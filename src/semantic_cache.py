"""Semantic answer cache, opt-in and off by default.

The synthesizer is where almost all of a query's cost and latency go, so the
highest-leverage saving left is not calling it at all when the same question, or
a close paraphrase, has already been answered. This caches the final answer
keyed by the query, with two layers:

1. An exact layer on the normalized query string. Zero risk: identical questions
   (case and spacing aside) return the stored answer with no similarity math.
2. A semantic layer over query embeddings. A new query is embedded and, if its
   cosine similarity to a cached query clears a conservative threshold, the
   stored answer is returned.

The correctness risk is real and is taken seriously. A grounded QA system whose
whole value is not guessing must not serve a cached answer for a question that
only looks similar. Three things contain that risk: a high default threshold so
only near-duplicate phrasings match, the exact layer so identical queries never
depend on fuzzy matching, and the fact that the cache is opt-in and is bypassed
entirely by the eval harness (which invokes the graph directly, not run_agent),
so a cache hit can never contaminate a measured number.

Scope: in memory, per process, bounded with least-recently-used eviction. That
fits a single instance. Behind more than one replica the store would move to a
shared backend such as Redis, and the class is small so that swap is contained.
The embedder is injectable so the behaviour is testable without a model.
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
    """Fold away the differences that should never count as a new question:
    surrounding space, internal runs of whitespace, and case."""
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
    """Query-keyed answer cache with an exact layer and a semantic layer."""

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
        """Return a hit for an identical or near-duplicate query, else None."""
        key = _normalize(query)
        with self._lock:
            exact = self._entries.get(key)
            if exact is not None:
                self._entries.move_to_end(key)
                return CacheHit(answer=exact.answer, kind="exact", similarity=1.0)

        # Embedding is done outside the lock (it can be slow) since it touches
        # no shared state.
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
        """Remember an answer for this query, evicting the oldest if full."""
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


# Production embedder: ChromaDB's ONNX all-MiniLM-L6-v2, the same offline model
# the default retriever uses. Loaded lazily and once, so importing this module
# stays cheap and no model is loaded unless the cache is actually enabled.
_ONNX_EF = None


def _default_embedder(text: str) -> list[float]:
    global _ONNX_EF
    if _ONNX_EF is None:
        from chromadb.utils import embedding_functions

        _ONNX_EF = embedding_functions.ONNXMiniLM_L6_V2()
    return list(_ONNX_EF([text])[0])


_SINGLETON: SemanticCache | None = None
_SINGLETON_LOCK = threading.Lock()


def get_semantic_cache() -> SemanticCache | None:
    """Return the process-wide cache, or None when caching is disabled.

    Off by default. run_agent calls this and skips the cache entirely on None,
    so default behaviour is unchanged and the eval path never sees a cache.
    """
    if not SEMANTIC_CACHE_ENABLED:
        return None
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = SemanticCache(
                    embedder=_default_embedder,
                    threshold=SEMANTIC_CACHE_THRESHOLD,
                    max_entries=SEMANTIC_CACHE_MAX_ENTRIES,
                )
    return _SINGLETON
