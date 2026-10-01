"""Answer cache, off by default.

The synthesizer is where most of the cost and latency go, so a repeated or near-identical question
can skip it. Two layers: an exact match on the normalized question, and a semantic match on the
embeddings if the cosine similarity is above a high threshold. A cached answer for a question that
only looks similar would be wrong, so the threshold is high, the exact layer doesn't depend on
fuzzy matching, and the eval harness bypasses the cache so a hit can't change a measured number.
It lives in memory per process with least-recently-used eviction. Several replicas would need a
shared store like Redis. The embedder can be passed in so tests don't need a model.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from src.config import (
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_MAX_ENTRIES,
    SEMANTIC_CACHE_THRESHOLD,
)

Embedder = Callable[[str], Sequence[float]]

_WHITESPACE = re.compile(r"\s+")


def _normalize(query: str) -> str:
    """Lowercase and collapse whitespace, so those differences don't make a new question."""
    return _WHITESPACE.sub(" ", query.strip().lower())


def _unit(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return arr
    return arr / norm


@dataclass(frozen=True)
class CacheHit:
    answer: str
    kind: str  # "exact" | "semantic"
    similarity: float


@dataclass
class _Entry:
    vec: np.ndarray  # unit-normalized query embedding
    answer: str


class SemanticCache:
    """Answer cache with an exact layer and a semantic layer."""

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.95,
        max_entries: int = 512,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._embed = embedder
        self._threshold = threshold
        self._max_entries = max_entries
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._lock = threading.Lock()

    def lookup(self, query: str) -> CacheHit | None:
        """A hit for an identical or near-duplicate query, else None."""
        key = _normalize(query)
        with self._lock:
            exact = self._entries.get(key)
            if exact is not None:
                self._entries.move_to_end(key)
                return CacheHit(answer=exact.answer, kind="exact", similarity=1.0)

        # embed outside the lock, it can be slow and uses no shared state
        qvec = _unit(self._embed(query))

        with self._lock:
            best_key, best_sim = None, -1.0
            for k, entry in self._entries.items():
                sim = float(np.dot(qvec, entry.vec))
                if sim > best_sim:
                    best_key, best_sim = k, sim
            if best_key is not None and best_sim >= self._threshold:
                self._entries.move_to_end(best_key)
                return CacheHit(
                    answer=self._entries[best_key].answer,
                    kind="semantic",
                    similarity=round(best_sim, 4),
                )
        return None

    def store(self, query: str, answer: str) -> None:
        """Store an answer for this query, dropping the oldest if the cache is full."""
        key = _normalize(query)
        vec = _unit(self._embed(query))
        with self._lock:
            self._entries[key] = _Entry(vec=vec, answer=answer)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# the real embedder is chroma's MiniLM, the same one the retriever uses. It loads on first use
_onnx_ef = None


def _default_embedder(text: str) -> list[float]:
    global _onnx_ef
    if _onnx_ef is None:
        from chromadb.utils import embedding_functions

        _onnx_ef = embedding_functions.ONNXMiniLM_L6_V2()
    return list(_onnx_ef([text])[0])


_singleton: SemanticCache | None = None
_SINGLETON_LOCK = threading.Lock()


def get_semantic_cache() -> SemanticCache | None:
    """The shared cache, or None when caching is off (the default). run_agent skips the cache on None."""
    if not SEMANTIC_CACHE_ENABLED:
        return None
    global _singleton
    if _singleton is None:
        with _SINGLETON_LOCK:
            if _singleton is None:
                _singleton = SemanticCache(
                    embedder=_default_embedder,
                    threshold=SEMANTIC_CACHE_THRESHOLD,
                    max_entries=SEMANTIC_CACHE_MAX_ENTRIES,
                )
    return _singleton
