"""Offline checks for eval/provider_bakeoff.py (aggregation + model overrides).

The bakeoff itself makes real API calls to three providers; this covers the
pure parts.
"""

from __future__ import annotations

from eval.provider_bakeoff import PROVIDER_MODELS, _aggregate, _distinct_docs, _mean


def test_provider_models_cover_all_four_providers():
    assert set(PROVIDER_MODELS) == {"anthropic", "openai", "google", "self_hosted"}
    assert PROVIDER_MODELS["anthropic"] == "claude-sonnet-4-6"


def test_distinct_docs_dedupes_in_order():
    ev = [{"source_document": "b.pdf"}, {"source_document": "a.pdf"}, {"source_document": "b.pdf"}]
    assert _distinct_docs(ev) == ["b.pdf", "a.pdf"]


def test_mean_ignores_none():
    assert _mean([1.0, None, 3.0]) == 2.0
    assert _mean([None, None]) is None


def test_aggregate_computes_per_provider_means():
    rows = [
        {"cost_usd": 0.01, "latency_seconds": 5.0, "output_tokens": 100,
         "behavior_match": True, "evidence_recall": 1.0,
         "groundedness": 0.9, "correctness": 1.0, "judge_cost_usd": 0.02},
        {"cost_usd": 0.03, "latency_seconds": 7.0, "output_tokens": 200,
         "behavior_match": False, "evidence_recall": 0.0,
         "groundedness": 0.7, "correctness": 0.8, "judge_cost_usd": 0.02},
    ]
    a = _aggregate("openai", "gpt-4o-mini", rows, use_judge=True)
    assert a["model"] == "gpt-4o-mini"
    assert a["mean_cost_per_query_usd"] == 0.02
    assert a["behavior_match_rate"] == 0.5
    assert a["groundedness"] == 0.8
    assert a["judge_cost_usd"] == 0.04


def test_aggregate_omits_judge_fields_when_disabled():
    rows = [{"cost_usd": 0.01, "latency_seconds": 5.0, "output_tokens": 100,
             "behavior_match": True, "evidence_recall": 1.0}]
    a = _aggregate("google", "gemini-3.6-flash", rows, use_judge=False)
    assert "groundedness" not in a
