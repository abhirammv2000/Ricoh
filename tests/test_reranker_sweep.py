"""Offline checks for eval/reranker_sweep.py's aggregation helper.

The sweep itself loads real cross-encoder models against a real index; this
covers the pure part.
"""

from __future__ import annotations

from eval.reranker_sweep import DEFAULT_RERANKERS, _agg


def test_default_rerankers_include_the_production_model():
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in DEFAULT_RERANKERS


def test_agg_averages_each_metric_across_rows():
    rows = [
        {"recall@1": 1.0, "recall@5": 1.0},
        {"recall@1": 0.0, "recall@5": 1.0},
    ]
    out = _agg(rows)
    assert out == {"recall@1": 0.5, "recall@5": 1.0}


def test_agg_is_empty_on_no_rows():
    assert _agg([]) == {}
