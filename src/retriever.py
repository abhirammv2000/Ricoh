"""Hybrid retrieval: ChromaDB vector search and BM25, merged with reciprocal rank fusion.

Vector search runs offline with chromadb's bundled MiniLM, so retrieval needs no API key. BM25
catches exact things the vectors miss, like error codes and model numbers. Both indexes are saved
to disk. RRF merges the two ranked lists by rank, because their scores aren't comparable.
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

logger = logging.getLogger(__name__)

# the cross-encoder is slow to load, so keep it
_reranker = None

# making a retriever reopens chroma and unpickles the whole BM25 index, so share one
_retriever: "HybridRetriever | None" = None


def get_retriever() -> "HybridRetriever":
    """The shared HybridRetriever, made on first use. Call reset_retriever() after rebuilding the index."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def reset_retriever() -> None:
    """Drop the cached retriever so the next get_retriever() reloads from disk."""
    global _retriever
    _retriever = None


def _get_embedding_function():
    """The chroma embedding function for EMBEDDING_MODEL.

    None means leave the argument out so chroma uses its default (see __init__). Passing
    embedding_function=None explicitly would break every upsert.
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


def _get_reranker(enabled: bool | None = None):
    """Load the cross-encoder the first time it's needed, or None if reranking is off.

    enabled defaults to RERANKER_ENABLED, the sweep passes it explicitly.
    """
    global _reranker
    want = RERANKER_ENABLED if enabled is None else enabled
    if not want:
        return None
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder  # lazy, heavy
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "RERANKER_ENABLED=true but sentence-transformers is not "
                "installed.  Run: pip install -r requirements-reranker.txt"
            ) from exc
        logger.info("Loading cross-encoder reranker '%s'...", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


class HybridRetriever:
    """Vector search plus BM25, merged with RRF.

    Call build_index(chunks) once, then retrieve(query). On later runs the constructor loads the
    saved BM25 index, so build_index isn't needed again.
    """

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_function: Any = None,
    ) -> None:
        """Open (or create) the chroma collection and load the saved BM25 index if there is one.

        embedding_function overrides EMBEDDING_MODEL. It's None in production, the sweep passes one.
        """
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # the BM25 files sit next to the chroma store they belong to. With fixed paths, a retriever
        # pointed at another folder got that folder's vectors and the default BM25 index.
        self._bm25_index_path = self._persist_dir / BM25_INDEX_PATH.name
        self._bm25_chunks_path = self._persist_dir / BM25_CHUNKS_PATH.name

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
        )

        # leave embedding_function out (don't pass None) when no model is set, so chroma uses its
        # own default. None would override it and every upsert would fail.
        collection_kwargs: dict[str, Any] = {
            "name": collection_name,
            "metadata": {"hnsw:space": "cosine"},  # cosine similarity
        }
        embedding_fn = embedding_function or _get_embedding_function()
        if embedding_fn is not None:
            collection_kwargs["embedding_function"] = embedding_fn

        self._collection = self._client.get_or_create_collection(**collection_kwargs)

        # load the saved BM25 index if there is one
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

    # bm25 save and load

    def _save_bm25(self) -> None:
        """Pickle the BM25 index and its chunk list."""
        self._bm25_index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._bm25_index_path, "wb") as f:
            pickle.dump(self._bm25, f)

        with open(self._bm25_chunks_path, "wb") as f:
            pickle.dump(self._bm25_chunks, f)

        logger.info(
            "BM25 index + chunks saved to '%s'.", self._bm25_index_path.parent
        )

    def _load_bm25(self) -> None:
        """Load the pickled BM25 index and chunks, if they exist."""
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

    # building the index

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Add the chunks from ingest.py to chroma and build the BM25 index.

        Safe to run twice, since chroma upserts by id. Chunks need id, text, source_document,
        page_number and chunk_index.
        """
        if not chunks:
            logger.warning("build_index called with empty chunk list.")
            return

        # chroma caps a single upsert at about 41k docs, so go in batches of 5000
        batch_size = 5_000

        logger.info(
            "Upserting %d chunks into ChromaDB collection '%s'...",
            len(chunks),
            self._collection.name,
        )

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
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
                min(i + batch_size, len(chunks)) - 1,
            )

        # plain lowercase split, no stemming, so codes like SC542 still match
        tokenised_corpus = [
            c["text"].lower().split() for c in chunks
        ]
        self._bm25 = BM25Okapi(tokenised_corpus)
        self._bm25_chunks = chunks
        self._save_bm25()

        logger.info(
            "BM25 index built and persisted: %d chunks.", len(chunks)
        )

    # vector search

    def _vector_search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """Top chunks from chroma, each with id, text, source, page, chunk_index and a similarity score."""
        count = self._collection.count()
        if count == 0:
            # an empty collection makes chroma raise a confusing TypeError, so return early
            logger.warning("Vector store is empty; returning no vector hits.")
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        # chroma returns one list per query text
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
                    # chroma gives a cosine distance, so flip it into a similarity
                    "score": 1.0 - dist,
                }
            )

        return ranked

    # bm25 search

    def _bm25_search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """Top chunks by BM25, in the same shape as _vector_search but with the raw BM25 score."""
        if self._bm25 is None:
            logger.warning("BM25 index not built; returning empty.")
            return []

        tokenised_query = query.lower().split()
        scores = self._bm25.get_scores(tokenised_query)

        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        ranked: list[dict[str, Any]] = []
        for idx, score in scored_indices:
            if score <= 0:
                break  # the rest score zero
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

    # rank fusion

    @staticmethod
    def _rrf_fuse(
        *ranked_lists: list[dict[str, Any]],
        k: int = RRF_K,
        final_k: int = RETRIEVAL_FINAL_K,
    ) -> list[dict[str, Any]]:
        """Merge ranked lists with reciprocal rank fusion and return the top final_k.

        A chunk's score is the sum of 1 / (k + rank) over the lists it appears in. It uses ranks
        only, so cosine and BM25 scores don't need to be on the same scale. Each result gets an
        rrf_score.
        """
        fused_scores: dict[str, float] = {}
        doc_lookup: dict[str, dict[str, Any]] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                doc_id = doc["id"]
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (
                    1.0 / (k + rank)
                )
                # keep the first copy we see
                if doc_id not in doc_lookup:
                    doc_lookup[doc_id] = doc

        sorted_ids = sorted(
            fused_scores, key=fused_scores.get, reverse=True  # type: ignore[arg-type]
        )[:final_k]

        results: list[dict[str, Any]] = []
        for doc_id in sorted_ids:
            entry = dict(doc_lookup[doc_id])  # shallow copy
            entry["rrf_score"] = fused_scores[doc_id]
            results.append(entry)

        return results

    # public api

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        final_k: int = RETRIEVAL_FINAL_K,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Vector search and BM25, fused with RRF. This is what the agent calls.

        top_k is the candidates per method, final_k how many fused results to return, and rerank
        overrides RERANKER_ENABLED for this call (only the sweep does that).
        """
        logger.info("Retrieving for query: '%s'", query[:80])

        # imported here to avoid a cycle (instrumentation imports llm_factory, which imports config)
        from src.instrumentation import span as _span

        with _span("retrieval", query=query[:200], top_k=top_k, final_k=final_k) as sp:
            vector_results = self._vector_search(query, top_k=top_k)
            bm25_results = self._bm25_search(query, top_k=top_k)

            logger.info(
                "  Vector hits: %d | BM25 hits: %d",
                len(vector_results),
                len(bm25_results),
            )

            reranker = _get_reranker(rerank)

            # with a reranker, fuse a bigger pool so it has more to reorder
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
                # which chunks went to the llm
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
        """Re-score with the cross-encoder and keep the top final_k, each with a rerank_score."""
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

    @property
    def index_size(self) -> int:
        """How many chunks are in the chroma collection."""
        return self._collection.count()

    @property
    def bm25_ready(self) -> bool:
        """Whether the BM25 index is loaded."""
        return self._bm25 is not None


# quick manual check

if __name__ == "__main__":
    import sys
    import time

    from src.ingest import ingest_all

    print("=" * 70)
    print("  Citera retrieval smoke test")
    print("=" * 70)

    print("\nIngesting PDFs...")
    chunks = ingest_all()

    if not chunks:
        print("No chunks found. Place PDFs in data/ first.")
        sys.exit(1)

    print(f"   {len(chunks)} chunks ingested.")

    print("\nBuilding hybrid index (ChromaDB + BM25)...")
    t0 = time.perf_counter()
    retriever = HybridRetriever()
    retriever.build_index(chunks)
    elapsed = time.perf_counter() - t0
    print(f"   Index built in {elapsed:.1f}s - {retriever.index_size} docs in ChromaDB.")
    print(f"   BM25 ready: {retriever.bm25_ready}")

    sample_queries = [
        "How do I fix error SC542?",
        "What paper sizes does the bypass tray support?",
        "How to configure network settings?",
    ]

    for query in sample_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 60)
        results = retriever.retrieve(query)

        if not results:
            print("   (no results)")
            continue

        for i, r in enumerate(results, 1):
            snippet = r["text"][:120].replace("\n", " ") + "..."
            print(
                f"   {i}. [{r['source_document']} p.{r['page_number']}] "
                f"(RRF={r['rrf_score']:.4f})"
            )
            print(f"      {snippet}")

    print("\n" + "=" * 70)
    print("Smoke test complete.")
    print("=" * 70)
