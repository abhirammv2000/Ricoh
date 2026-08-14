"""
src/retriever.py - Hybrid Retrieval Engine (ChromaDB + BM25 + RRF).

This is the **heart of Phase 2**.  It:

1. Accepts the chunk list produced by ``ingest.py``.
2. Builds **two parallel indices**:
   a. A ChromaDB collection for dense/semantic vector search
      (embeddings computed locally via the bundled all-MiniLM-L6-v2).
   b. A BM25 index for sparse/keyword search, **persisted to disk**
      as pickle files so it survives process restarts.
3. At query time, runs *both* searches and fuses the ranked lists
   via **Reciprocal Rank Fusion (RRF)** - a simple, tuning-free
   method that combines ranks instead of raw scores.

Design decisions
- ChromaDB's default embedding function (all-MiniLM-L6-v2) runs
  entirely offline - critical because the hackathon disallows web
  search and we want zero API-key dependencies at retrieval time.
- BM25 adds keyword-exact-match strength that pure vector search
  misses on model numbers, error codes, and part names that are
  common in Ricoh technical manuals.
- BM25 is pickled to disk alongside ChromaDB so both indices
  persist across restarts - fixes the "BM25 not built" bug.
- RRF (k=60) is preferred over linear score fusion because the two
  score distributions are incommensurable.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from src.config import (
    BM25_CHUNKS_PATH,
    BM25_INDEX_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RERANK_CANDIDATE_POOL,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RETRIEVAL_FINAL_K,
    RETRIEVAL_TOP_K,
    RRF_K,
)

# Logging (configured centrally in config.py)
logger = logging.getLogger(__name__)

# Module-level cache so the (expensive) cross-encoder loads at most once.
_RERANKER = None

# Module-level cache for the retriever itself.  Constructing a
# HybridRetriever re-opens the Chroma client AND unpickles the entire
# BM25 index from disk (see __init__), so building a fresh one on every
# retrieval pass is pure waste.  get_retriever() returns a single shared
# instance instead.  Loads are idempotent, so reuse is behaviour-preserving.
_RETRIEVER: "HybridRetriever | None" = None


def get_retriever() -> "HybridRetriever":
    """Return a process-wide shared HybridRetriever (constructed once).

    The instance is cached at module level, so repeated calls avoid
    re-opening ChromaDB and re-unpickling the BM25 index from disk.
    Call ``reset_retriever()`` after (re)building the index to force a
    reload on the next access.
    """
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


def reset_retriever() -> None:
    """Clear the cached retriever so the next get_retriever() reloads.

    Used after build_index() writes a fresh index to disk, so callers
    never see a stale in-memory index.
    """
    global _RETRIEVER
    _RETRIEVER = None


def _get_embedding_function():
    """Return the ChromaDB embedding function for the configured model.

    ``None`` means "caller should omit the argument entirely" so ChromaDB
    applies its own default (all-MiniLM-L6-v2 via onnxruntime, no torch).
    Note this is *not* the same as passing ``embedding_function=None``:
    that explicitly overrides the default with nothing and makes every
    upsert fail with "You must provide an embedding function".  See
    ``HybridRetriever.__init__`` for the omit-the-kwarg handling.

    When ``EMBEDDING_MODEL`` is set, use a stronger sentence-transformers
    model instead (requires the optional extra).
    """
    if not EMBEDDING_MODEL:
        return None
    try:
        from chromadb.utils import embedding_functions
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "EMBEDDING_MODEL is set but the dependency is missing. "
            "Run: pip install -r requirements-enhanced.txt"
        ) from exc
    logger.info("Using embedding model '%s'.", EMBEDDING_MODEL)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def _get_reranker():
    """Lazily load the cross-encoder reranker (cached).

    Returns ``None`` if reranking is disabled.  Raises a helpful error
    if enabled but the optional dependency is not installed.
    """
    global _RERANKER
    if not RERANKER_ENABLED:
        return None
    if _RERANKER is None:
        try:
            from sentence_transformers import CrossEncoder  # lazy, heavy
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "RERANKER_ENABLED=true but sentence-transformers is not "
                "installed.  Run: pip install -r requirements-reranker.txt"
            ) from exc
        logger.info("Loading cross-encoder reranker '%s'…", RERANKER_MODEL)
        _RERANKER = CrossEncoder(RERANKER_MODEL)
    return _RERANKER


class HybridRetriever:
    """Unified retrieval interface: semantic + keyword + RRF fusion.

    Usage::

        retriever = HybridRetriever()
        retriever.build_index(chunks)          # one-time
        results = retriever.retrieve("query")  # per-question

    On subsequent runs, the constructor auto-loads the persisted
    BM25 index from disk - no need to call ``build_index`` again.
    """

    # ----------------------------------------------------------------
    # INITIALISATION
    # ----------------------------------------------------------------

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        """Create or load a persistent ChromaDB client + collection.

        Also attempts to load a previously pickled BM25 index so that
        keyword search works immediately without re-ingestion.

        Args:
            persist_dir:     Directory for ChromaDB's SQLite storage.
            collection_name: Name of the Chroma collection to use.
        """
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # BM25 lives beside the Chroma store it belongs to.
        #
        # These were previously module-level constants, so a retriever pointed
        # at a different persist_dir got an EMPTY vector store paired with the
        # DEFAULT BM25 index, two halves of different indexes, reporting
        # bm25_ready=True while holding no vectors. Deriving the paths keeps a
        # retriever internally consistent, and is identical to the old
        # behaviour for the default directory.
        self._bm25_index_path = self._persist_dir / BM25_INDEX_PATH.name
        self._bm25_chunks_path = self._persist_dir / BM25_CHUNKS_PATH.name

        # Persistent client → data survives process restarts
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
        )

        # get_or_create → idempotent; safe to call multiple times.
        # The embedding_function kwarg is OMITTED (not passed as None) when no
        # EMBEDDING_MODEL is configured, so ChromaDB applies its own onnx
        # MiniLM default.  Passing None explicitly overrides that default and
        # makes every upsert raise "You must provide an embedding function".
        collection_kwargs: dict[str, Any] = {
            "name": collection_name,
            "metadata": {"hnsw:space": "cosine"},  # cosine similarity
        }
        embedding_fn = _get_embedding_function()
        if embedding_fn is not None:
            collection_kwargs["embedding_function"] = embedding_fn

        self._collection = self._client.get_or_create_collection(**collection_kwargs)

        # BM25 index + backing store - try loading from disk first
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[dict[str, Any]] = []
        self._load_bm25()

        logger.info(
            "HybridRetriever ready - Chroma collection '%s' "
            "(%d existing docs) at '%s'.  BM25: %s",
            collection_name,
            self._collection.count(),
            self._persist_dir,
            "loaded" if self._bm25 is not None else "NOT loaded",
        )

    # ----------------------------------------------------------------
    # BM25 PERSISTENCE - save / load pickle files
    # ----------------------------------------------------------------

    def _save_bm25(self) -> None:
        """Persist the BM25 index and chunk list to disk as pickle."""
        self._bm25_index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._bm25_index_path, "wb") as f:
            pickle.dump(self._bm25, f)

        with open(self._bm25_chunks_path, "wb") as f:
            pickle.dump(self._bm25_chunks, f)

        logger.info(
            "BM25 index + chunks saved to '%s'.", self._bm25_index_path.parent
        )

    def _load_bm25(self) -> None:
        """Load previously pickled BM25 index + chunks from disk."""
        if self._bm25_index_path.exists() and self._bm25_chunks_path.exists():
            with open(self._bm25_index_path, "rb") as f:
                self._bm25 = pickle.load(f)

            with open(self._bm25_chunks_path, "rb") as f:
                self._bm25_chunks = pickle.load(f)

            logger.info(
                "BM25 index loaded from disk: %d chunks.",
                len(self._bm25_chunks),
            )
        else:
            logger.info("No persisted BM25 index found - will need build_index().")

    # ----------------------------------------------------------------
    # INDEX BUILDING
    # ----------------------------------------------------------------

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Populate ChromaDB and build a BM25 index from chunks.

        This is designed to be **idempotent**: if the ChromaDB
        collection already contains documents, we skip re-adding
        them (ChromaDB upserts by ID).

        The BM25 index is **pickled to disk** so it persists across
        process restarts - matching ChromaDB's persistence.

        Args:
            chunks: Flat list of chunk dicts from ``ingest.py``.
                    Required keys: ``id``, ``text``, ``source_document``,
                    ``page_number``, ``chunk_index``.
        """
        if not chunks:
            logger.warning("build_index called with empty chunk list.")
            return

        # 1. ChromaDB (vector index)
        # ChromaDB upsert accepts batches of up to ~41 666 docs
        # (limited by the underlying SQLite default).  We batch at
        # 5 000 to stay well within limits and keep memory reasonable.
        BATCH_SIZE = 5_000

        logger.info(
            "Upserting %d chunks into ChromaDB collection '%s'…",
            len(chunks),
            self._collection.name,
        )

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            self._collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "source_document": c["source_document"],
                        "page_number": c["page_number"],
                        "chunk_index": c["chunk_index"],
                    }
                    for c in batch
                ],
            )
            logger.info(
                "  ChromaDB upsert batch %d-%d done.",
                i,
                min(i + BATCH_SIZE, len(chunks)) - 1,
            )

        # 2. BM25 (keyword index)
        # Tokenisation: simple lowercase whitespace split.  This is
        # intentionally basic - BM25 does not need stemming to match
        # error codes like "SC542" or model names like "IM C3500".
        tokenised_corpus = [
            c["text"].lower().split() for c in chunks
        ]
        self._bm25 = BM25Okapi(tokenised_corpus)
        self._bm25_chunks = chunks  # keep a reference for lookup

        # 3. Persist BM25 to disk
        self._save_bm25()

        logger.info(
            "BM25 index built and persisted: %d chunks.", len(chunks)
        )

    # ----------------------------------------------------------------
    # SEARCH - VECTOR (SEMANTIC)
    # ----------------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """Query ChromaDB for semantically similar chunks.

        Returns:
            Ranked list of dicts with keys:
            ``id``, ``text``, ``source_document``, ``page_number``,
            ``chunk_index``, ``score`` (cosine distance → similarity).
        """
        count = self._collection.count()
        if count == 0:
            # Chroma raises an opaque "requested results 0" TypeError here.
            # An empty collection is an operational state (index not built
            # yet), not a programming error, so report it as one.
            logger.warning("Vector store is empty; returning no vector hits.")
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns nested lists (one per query text)
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        ranked: list[dict[str, Any]] = []
        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            ranked.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "source_document": meta["source_document"],
                    "page_number": meta["page_number"],
                    "chunk_index": meta["chunk_index"],
                    # ChromaDB cosine distance ∈ [0, 2].
                    # Convert to similarity ∈ [-1, 1] for readability.
                    "score": 1.0 - dist,
                }
            )

        return ranked

    # ----------------------------------------------------------------
    # SEARCH - BM25 (KEYWORD)
    # ----------------------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """Score every chunk against the BM25 index and return top-k.

        Returns:
            Same schema as ``_vector_search``, with ``score`` being
            the raw BM25 score (higher = more relevant).
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built; returning empty.")
            return []

        tokenised_query = query.lower().split()
        scores = self._bm25.get_scores(tokenised_query)

        # Pair scores with chunk indices, sort descending
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        ranked: list[dict[str, Any]] = []
        for idx, score in scored_indices:
            if score <= 0:
                break  # no point returning zero-relevance results
            chunk = self._bm25_chunks[idx]
            ranked.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source_document": chunk["source_document"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "score": float(score),
                }
            )

        return ranked

    # ----------------------------------------------------------------
    # FUSION - RECIPROCAL RANK FUSION (RRF)
    # ----------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        *ranked_lists: list[dict[str, Any]],
        k: int = RRF_K,
        final_k: int = RETRIEVAL_FINAL_K,
    ) -> list[dict[str, Any]]:
        """Merge multiple ranked lists via Reciprocal Rank Fusion.

        RRF score for document *d* = Σ  1 / (k + rank_i(d))
        across all ranked lists *i* where *d* appears.

        This method is ranking-based (not score-based), so it is
        immune to the scale-mismatch problem between cosine
        similarity and BM25 scores.

        Args:
            *ranked_lists: One or more ranked result lists.
            k:             Smoothing constant (default 60).
            final_k:       Number of results to return.

        Returns:
            Fused top-``final_k`` results, each dict gaining an
            ``rrf_score`` key.
        """
        fused_scores: dict[str, float] = {}
        doc_lookup: dict[str, dict[str, Any]] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                doc_id = doc["id"]
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (
                    1.0 / (k + rank)
                )
                # Keep the first occurrence's full dict
                if doc_id not in doc_lookup:
                    doc_lookup[doc_id] = doc

        # Sort by fused score descending, take top final_k
        sorted_ids = sorted(
            fused_scores, key=fused_scores.get, reverse=True  # type: ignore[arg-type]
        )[:final_k]

        results: list[dict[str, Any]] = []
        for doc_id in sorted_ids:
            entry = dict(doc_lookup[doc_id])  # shallow copy
            entry["rrf_score"] = fused_scores[doc_id]
            results.append(entry)

        return results

    # ----------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        final_k: int = RETRIEVAL_FINAL_K,
    ) -> list[dict[str, Any]]:
        """Run hybrid retrieval: vector + BM25 → RRF fusion.

        This is the **only method the LangGraph agent calls**.

        Args:
            query:   Natural-language question.
            top_k:   Candidates per retrieval method.
            final_k: How many fused results to return.

        Returns:
            Up to ``final_k`` chunk dicts, each containing:
            ``id``, ``text``, ``source_document``, ``page_number``,
            ``chunk_index``, ``rrf_score``.
        """
        logger.info("Retrieving for query: '%s'", query[:80])

        # Traced so a stored request can answer "which chunks produced this
        # answer?" after the fact.  Import is local to avoid a module-level
        # cycle (instrumentation imports llm_factory, which imports config).
        from src.instrumentation import span as _span

        with _span("retrieval", query=query[:200], top_k=top_k, final_k=final_k) as sp:
            vector_results = self._vector_search(query, top_k=top_k)
            bm25_results = self._bm25_search(query, top_k=top_k)

            logger.info(
                "  Vector hits: %d | BM25 hits: %d",
                len(vector_results),
                len(bm25_results),
            )

            reranker = _get_reranker()

            # With a reranker, fuse a LARGER pool first so the cross-encoder
            # has more candidates to reorder, then trim to final_k.
            fuse_k = max(final_k, RERANK_CANDIDATE_POOL) if reranker else final_k
            fused = self._rrf_fuse(vector_results, bm25_results, final_k=fuse_k)

            if reranker is not None and fused:
                fused = self._rerank(reranker, query, fused, final_k)
                logger.info("  Reranked to top %d.", len(fused))
            else:
                logger.info("  Fused results: %d", len(fused))

            sp.set(
                vector_hits=len(vector_results),
                bm25_hits=len(bm25_results),
                reranked=reranker is not None,
                # Chunk attribution: the exact chunks handed to the LLM.
                chunks=[
                    {
                        "id": d["id"],
                        "doc": d["source_document"],
                        "page": d["page_number"],
                        "rrf": round(d.get("rrf_score", 0.0), 5),
                    }
                    for d in fused
                ],
            )

        return fused

    @staticmethod
    def _rerank(
        reranker,
        query: str,
        docs: list[dict[str, Any]],
        final_k: int,
    ) -> list[dict[str, Any]]:
        """Re-score candidates with a cross-encoder and keep the top-k.

        Adds a ``rerank_score`` to each returned doc.
        """
        pairs = [(query, d["text"]) for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(docs, scores), key=lambda x: x[1], reverse=True
        )[:final_k]
        out: list[dict[str, Any]] = []
        for doc, score in ranked:
            entry = dict(doc)
            entry["rerank_score"] = float(score)
            out.append(entry)
        return out

    # ----------------------------------------------------------------
    # UTILITY - check if index is populated
    # ----------------------------------------------------------------

    @property
    def index_size(self) -> int:
        """Number of documents currently in the ChromaDB collection."""
        return self._collection.count()

    @property
    def bm25_ready(self) -> bool:
        """Whether the BM25 index is loaded and ready."""
        return self._bm25 is not None


# ====================================================================
# __main__ - End-to-end smoke test
# ====================================================================

if __name__ == "__main__":
    import sys
    import time

    from src.ingest import ingest_all

    print("=" * 70)
    print("  Citera - Phase 2 Hybrid Retrieval Smoke Test")
    print("=" * 70)

    # Step 1: Ingest all PDFs from data/
    print("\n📄  Ingesting PDFs…")
    chunks = ingest_all()

    if not chunks:
        print("❌  No chunks found. Place PDFs in data/ first.")
        sys.exit(1)

    print(f"   {len(chunks)} chunks ingested.")

    # Step 2: Build the hybrid index
    print("\n🔨  Building hybrid index (ChromaDB + BM25)…")
    t0 = time.perf_counter()
    retriever = HybridRetriever()
    retriever.build_index(chunks)
    elapsed = time.perf_counter() - t0
    print(f"   Index built in {elapsed:.1f}s - {retriever.index_size} docs in ChromaDB.")
    print(f"   BM25 ready: {retriever.bm25_ready}")

    # Step 3: Run sample queries
    sample_queries = [
        "How do I fix error SC542?",
        "What paper sizes does the bypass tray support?",
        "How to configure network settings?",
    ]

    for query in sample_queries:
        print(f"\n🔍  Query: \"{query}\"")
        print("-" * 60)
        results = retriever.retrieve(query)

        if not results:
            print("   (no results)")
            continue

        for i, r in enumerate(results, 1):
            snippet = r["text"][:120].replace("\n", " ") + "…"
            print(
                f"   {i}. [{r['source_document']} p.{r['page_number']}] "
                f"(RRF={r['rrf_score']:.4f})"
            )
            print(f"      {snippet}")

    print("\n" + "=" * 70)
    print("Phase 2 smoke test complete. Ready for Phase 3!")
    print("=" * 70)
