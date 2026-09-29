"""
src/azure_retriever.py - Azure AI Search backend for the same chunks.

Takes the chunk list from ``ingest.py`` and indexes it in Azure AI Search, so
the ChromaDB + BM25 retriever and Azure can be compared on identical input.
The embeddings come from the same model ChromaDB uses (all-MiniLM-L6-v2, 384
dimensions), so any ranking difference comes from the search engine and not
the vectors.

Four query modes, all on one index:

- ``keyword``: Azure's built-in BM25 over the chunk text.
- ``vector``: HNSW nearest neighbours on the embedding field.
- ``hybrid``: keyword and vector together, fused by Azure with RRF.
- ``hybrid_semantic``: hybrid, then Azure's semantic ranker re-scores the top
  results (Azure's counterpart to the cross-encoder in retriever.py).

Needs ``pip install -r requirements-azure.txt`` and the AZURE_SEARCH_* values
in .env. Nothing else in the project imports this module.

Usage:
    python -m src.azure_retriever --build     # create the index and upload all chunks
    python -m src.azure_retriever "how do I add a step to a workflow"
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Any

from src.config import (
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX,
    RETRIEVAL_FINAL_K,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)
# The Azure SDK logs every HTTP request and response at INFO.
logging.getLogger("azure").setLevel(logging.WARNING)

EMBEDDING_DIM = 384
HNSW_CONFIG = "citera-hnsw"
VECTOR_PROFILE = "citera-vector-profile"
SEMANTIC_CONFIG = "citera-semantic"

MODES = ("keyword", "vector", "hybrid", "hybrid_semantic")

_UPLOAD_BATCH = 200


def _clean_query(query: str) -> str:
    """Strip the characters Azure's simple query syntax treats as operators.

    A stray quote or a leading minus in a user question would otherwise be read
    as a phrase or a NOT. Hyphens inside a word (e-mail) are left alone.
    """
    query = re.sub(r'["()*+|~\\]', " ", query)
    query = re.sub(r"(^|\s)-+", r"\1", query)
    return query.strip()


class AzureRetriever:
    """Index chunks in Azure AI Search and query them in any of the four modes."""

    def __init__(
        self,
        endpoint: str = AZURE_SEARCH_ENDPOINT,
        api_key: str = AZURE_SEARCH_API_KEY,
        index_name: str = AZURE_SEARCH_INDEX,
    ) -> None:
        if not endpoint or not api_key:
            raise RuntimeError(
                "Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY in .env "
                "(see .env.example)."
            )
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
            from azure.search.documents.indexes import SearchIndexClient
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "azure-search-documents is not installed. "
                "Run: pip install -r requirements-azure.txt"
            ) from exc

        credential = AzureKeyCredential(api_key)
        self._index_name = index_name
        self._index_client = SearchIndexClient(endpoint, credential)
        self._client = SearchClient(endpoint, index_name, credential)
        self._embed = None

    # Embeddings

    def _embedder(self):
        """ChromaDB's default embedding function, the one the local index uses."""
        if self._embed is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._embed = DefaultEmbeddingFunction()
        return self._embed

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._embedder()(texts)]

    # Index management

    def create_index(self, recreate: bool = False) -> None:
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            HnswParameters,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
            VectorSearch,
            VectorSearchProfile,
        )

        if recreate:
            self._index_client.delete_index(self._index_name)

        index = SearchIndex(
            name=self._index_name,
            fields=[
                SearchField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SearchField(name="text", type=SearchFieldDataType.String, searchable=True),
                SearchField(name="source_document", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    hidden=True,
                    vector_search_dimensions=EMBEDDING_DIM,
                    vector_search_profile_name=VECTOR_PROFILE,
                ),
            ],
            vector_search=VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name=HNSW_CONFIG, parameters=HnswParameters(metric="cosine")
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name=VECTOR_PROFILE, algorithm_configuration_name=HNSW_CONFIG
                    )
                ],
            ),
            semantic_search=SemanticSearch(
                default_configuration_name=SEMANTIC_CONFIG,
                configurations=[
                    SemanticConfiguration(
                        name=SEMANTIC_CONFIG,
                        prioritized_fields=SemanticPrioritizedFields(
                            content_fields=[SemanticField(field_name="text")]
                        ),
                    )
                ],
            ),
        )
        self._index_client.create_or_update_index(index)
        logger.info("Azure index '%s' ready.", self._index_name)

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Embed and upload every chunk. Safe to re-run, documents upsert by id."""
        if not chunks:
            logger.warning("build_index called with empty chunk list.")
            return

        self.create_index()
        for i in range(0, len(chunks), _UPLOAD_BATCH):
            batch = chunks[i : i + _UPLOAD_BATCH]
            vectors = self.embed([c["text"] for c in batch])
            docs = [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "source_document": c["source_document"],
                    "page_number": c["page_number"],
                    "chunk_index": c["chunk_index"],
                    "embedding": v,
                }
                for c, v in zip(batch, vectors)
            ]
            results = self._client.upload_documents(documents=docs)
            failed = [r.key for r in results if not r.succeeded]
            if failed:
                raise RuntimeError(f"Azure rejected {len(failed)} documents, e.g. {failed[:3]}")
            logger.info("  uploaded %d-%d", i, min(i + _UPLOAD_BATCH, len(chunks)) - 1)

    @property
    def index_size(self) -> int:
        return self._client.get_document_count()

    # Search

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_semantic",
        top_k: int = RETRIEVAL_TOP_K,
        final_k: int = RETRIEVAL_FINAL_K,
    ) -> list[dict[str, Any]]:
        """Return up to ``final_k`` chunks in the same shape HybridRetriever uses.

        ``top_k`` is the number of vector neighbours fetched before fusion.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        from azure.search.documents.models import VectorizedQuery

        kwargs: dict[str, Any] = {
            "top": final_k,
            "select": ["id", "text", "source_document", "page_number", "chunk_index"],
        }
        if mode != "vector":
            kwargs["search_text"] = _clean_query(query)
        if mode != "keyword":
            kwargs["vector_queries"] = [
                VectorizedQuery(
                    vector=self.embed([query])[0],
                    k_nearest_neighbors=top_k,
                    fields="embedding",
                )
            ]
        if mode == "hybrid_semantic":
            kwargs["query_type"] = "semantic"
            kwargs["semantic_configuration_name"] = SEMANTIC_CONFIG

        out: list[dict[str, Any]] = []
        for r in self._client.search(**kwargs):
            entry = {
                "id": r["id"],
                "text": r["text"],
                "source_document": r["source_document"],
                "page_number": r["page_number"],
                "chunk_index": r["chunk_index"],
                "score": r["@search.score"],
            }
            if r.get("@search.reranker_score") is not None:
                entry["rerank_score"] = r["@search.reranker_score"]
            out.append(entry)
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Azure AI Search backend for Citera")
    ap.add_argument("--build", action="store_true", help="create the index and upload all chunks")
    ap.add_argument("--mode", default="hybrid_semantic", choices=MODES)
    ap.add_argument("query", nargs="?", help="a question to search for")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    retriever = AzureRetriever()

    if args.build:
        from src.ingest import ingest_all

        chunks = ingest_all()
        if not chunks:
            raise SystemExit("no PDFs found in data/. add them and retry.")
        print(f"uploading {len(chunks)} chunks to '{AZURE_SEARCH_INDEX}'")
        retriever.build_index(chunks)
        print(f"index now holds {retriever.index_size} documents")
    elif args.query:
        for i, r in enumerate(retriever.retrieve(args.query, mode=args.mode), 1):
            print(f"{i}. {r['source_document']} p.{r['page_number']}  score={r['score']:.4f}")
            print(f"   {r['text'][:120].replace(chr(10), ' ')}...")
    else:
        ap.error("pass --build or a query")
