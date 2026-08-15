"""Unit tests for the semantic answer cache (src/semantic_cache.py).

A fake embedder maps text to fixed 2D vectors, so hits, misses, thresholds, and
eviction are all deterministic and need no model.
"""

from __future__ import annotations

import math

import pytest

import src.agent as agentmod
import src.semantic_cache as sc
from src.semantic_cache import SemanticCache

_S = 1.0 / math.sqrt(2)  # ~0.7071


def vec_embed(text: str):
    t = text.lower()
    if "shutdown" in t or "shut down" in t:
        return [1.0, 0.0]
    if "network" in t:
        return [0.0, 1.0]
    if "paper" in t:
        return [_S, _S]  # 45 degrees from both axes, cosine 0.7071
    return [0.5, 0.5]


def test_exact_hit_ignores_case_and_spacing():
    c = SemanticCache(embedder=vec_embed, threshold=0.95)
    c.store("How do I shut down?", "STOP")
    hit = c.lookup("  how do I   SHUT DOWN? ")
    assert hit is not None
    assert hit.kind == "exact"
    assert hit.answer == "STOP"
    assert hit.similarity == 1.0


def test_semantic_hit_same_topic_different_words():
    c = SemanticCache(embedder=vec_embed, threshold=0.95)
    c.store("shutdown command", "STOP")
    hit = c.lookup("system shut down please")
    assert hit is not None
    assert hit.kind == "semantic"
    assert hit.answer == "STOP"
    assert hit.similarity >= 0.95


def test_semantic_miss_different_topic():
    c = SemanticCache(embedder=vec_embed, threshold=0.95)
    c.store("shutdown command", "STOP")
    assert c.lookup("network settings") is None


def test_threshold_is_respected_at_the_boundary():
    # paper vs shutdown cosine is 0.7071.
    strict = SemanticCache(embedder=vec_embed, threshold=0.95)
    strict.store("shutdown command", "STOP")
    assert strict.lookup("paper size") is None

    loose = SemanticCache(embedder=vec_embed, threshold=0.70)
    loose.store("shutdown command", "STOP")
    hit = loose.lookup("paper size")
    assert hit is not None
    assert hit.kind == "semantic"


def test_lru_evicts_oldest():
    c = SemanticCache(embedder=vec_embed, threshold=0.95, max_entries=2)
    c.store("network a", "A")
    c.store("paper b", "B")
    c.store("shutdown c", "C")  # over capacity, evicts "network a"
    assert len(c) == 2
    assert c.lookup("network a") is None


def test_lru_hit_refreshes_recency():
    c = SemanticCache(embedder=vec_embed, threshold=0.95, max_entries=2)
    c.store("network a", "A")
    c.store("paper b", "B")
    c.lookup("network a")       # touch A, so B becomes the oldest
    c.store("shutdown c", "C")  # evicts B, not A
    assert c.lookup("paper b") is None
    assert c.lookup("network a") is not None


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        SemanticCache(embedder=vec_embed, threshold=0.0)
    with pytest.raises(ValueError):
        SemanticCache(embedder=vec_embed, threshold=1.5)


def test_get_semantic_cache_disabled_by_default(monkeypatch):
    monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", False)
    monkeypatch.setattr(sc, "_SINGLETON", None)
    assert sc.get_semantic_cache() is None


def test_get_semantic_cache_singleton_when_enabled(monkeypatch):
    monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", True)
    monkeypatch.setattr(sc, "_SINGLETON", None)
    monkeypatch.setattr(sc, "_default_embedder", vec_embed)  # avoid loading ONNX
    c = sc.get_semantic_cache()
    assert c is not None
    assert sc.get_semantic_cache() is c


def test_run_agent_returns_cached_answer_without_running_graph(monkeypatch):
    cache = SemanticCache(embedder=vec_embed, threshold=0.95)
    cache.store("how do I shut down?", "CACHED [a.pdf, Page 1]")
    monkeypatch.setattr(agentmod, "get_semantic_cache", lambda: cache)

    def boom(*a, **k):
        raise AssertionError("graph must not run on a cache hit")

    monkeypatch.setattr(agentmod, "get_agent_graph", boom)
    assert agentmod.run_agent("How do I shut down?") == "CACHED [a.pdf, Page 1]"


def test_run_agent_stores_answer_on_miss(monkeypatch):
    cache = SemanticCache(embedder=vec_embed, threshold=0.95)
    monkeypatch.setattr(agentmod, "get_semantic_cache", lambda: cache)

    class FakeGraph:
        def invoke(self, state):
            return {"final_answer": "FRESH [a.pdf, Page 1]"}

    monkeypatch.setattr(agentmod, "get_agent_graph", lambda **k: FakeGraph())
    out = agentmod.run_agent("network settings question")
    assert out == "FRESH [a.pdf, Page 1]"
    hit = cache.lookup("network settings question")
    assert hit is not None and hit.answer == "FRESH [a.pdf, Page 1]"
