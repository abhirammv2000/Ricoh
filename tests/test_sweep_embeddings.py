"""Scoring maths for the embedding sweep (eval/sweep_embeddings). Pure and
offline; importing the module must not pull in torch."""

from __future__ import annotations

import math

import pytest

from eval.sweep_embeddings import _agg, _distinct_ranked_docs, _score_one


def test_distinct_ranked_docs_preserves_order_and_dedupes():
    results = [
        {"source_document": "a.pdf"},
        {"source_document": "a.pdf"},
        {"source_document": "b.pdf"},
        {"source_document": None},
        {"source_document": "c.pdf"},
    ]
    assert _distinct_ranked_docs(results) == ["a.pdf", "b.pdf", "c.pdf"]


def test_score_one_hit_at_rank_1():
    s = _score_one(["a.pdf", "b.pdf", "c.pdf"], ["a.pdf"], any_hit=True)
    assert s["recall@1"] == 1.0
    assert s["recall@5"] == 1.0
    assert s["mrr"] == 1.0
    assert s["ndcg@5"] == pytest.approx(1.0)


def test_score_one_hit_at_rank_3():
    s = _score_one(["x.pdf", "y.pdf", "a.pdf"], ["a.pdf"], any_hit=True)
    assert s["recall@1"] == 0.0
    assert s["recall@3"] == 1.0
    assert s["mrr"] == pytest.approx(1 / 3)
    # one relevant doc at position 3 -> dcg = 1/log2(4) = 0.5, idcg = 1.0
    assert s["ndcg@5"] == pytest.approx(0.5)


def test_score_one_complete_miss():
    s = _score_one(["x.pdf", "y.pdf"], ["a.pdf"], any_hit=True)
    assert s["recall@5"] == 0.0
    assert s["mrr"] == 0.0
    assert s["ndcg@5"] == 0.0


def test_score_one_partial_recall_when_not_any_hit():
    s = _score_one(["a.pdf", "b.pdf", "z.pdf"], ["a.pdf", "b.pdf", "c.pdf"], any_hit=False)
    assert s["recall@5"] == pytest.approx(2 / 3)
    # dcg over ranks 1,2 vs idcg over 3 ideal positions -> below 1
    dcg = 1 / math.log2(2) + 1 / math.log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert s["ndcg@5"] == pytest.approx(dcg / idcg)


def test_ndcg_never_exceeds_one_with_alternative_sources():
    # two alternative expected docs both retrieved near the top must not push
    # ndcg above 1 (the earlier bug)
    s = _score_one(["a.pdf", "b.pdf", "c.pdf"], ["a.pdf", "b.pdf"], any_hit=True)
    assert s["ndcg@5"] == 1.0


def test_agg_averages_each_key():
    rows = [{"recall@1": 1.0, "mrr": 0.5}, {"recall@1": 0.0, "mrr": 1.0}]
    assert _agg(rows) == {"recall@1": 0.5, "mrr": 0.75}
    assert _agg([]) == {}
