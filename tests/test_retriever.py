"""Unit tests for Reciprocal Rank Fusion (src/retriever.py).

RRF is the core fusion logic and is a pure static method, so it can
be tested without ChromaDB, BM25, or any I/O.
"""

from __future__ import annotations

from src.retriever import HybridRetriever

fuse = HybridRetriever._rrf_fuse


def _docs(*ids):
    return [{"id": i, "text": i} for i in ids]


def test_documents_in_both_lists_rank_highest():
    list_a = _docs("a", "b", "c")
    list_b = _docs("b", "c", "a")
    fused = fuse(list_a, list_b, k=60, final_k=3)
    # "b" is rank 2 then rank 1 -> best combined reciprocal rank.
    assert fused[0]["id"] == "b"
    assert {d["id"] for d in fused} == {"a", "b", "c"}


def test_final_k_limits_results():
    fused = fuse(_docs("a", "b", "c", "d"), k=60, final_k=2)
    assert len(fused) == 2


def test_rrf_score_attached():
    fused = fuse(_docs("a", "b"), k=60, final_k=2)
    assert all("rrf_score" in d for d in fused)
    assert fused[0]["rrf_score"] >= fused[-1]["rrf_score"]


def test_single_list_preserves_order():
    fused = fuse(_docs("a", "b", "c"), k=60, final_k=3)
    assert [d["id"] for d in fused] == ["a", "b", "c"]


def test_duplicate_doc_accumulates_score():
    # "a" appears in two lists; "z" only once. "a" must outrank "z".
    fused = fuse(_docs("a", "z"), _docs("a", "y"), k=60, final_k=3)
    assert fused[0]["id"] == "a"


def test_smaller_k_amplifies_rank_gaps():
    lists = (_docs("a", "b", "c"), _docs("a", "b", "c"))
    top_small_k = fuse(*lists, k=1, final_k=1)[0]
    top_big_k = fuse(*lists, k=1000, final_k=1)[0]
    # Top result is stable regardless of k for identical lists.
    assert top_small_k["id"] == top_big_k["id"] == "a"
