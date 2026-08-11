"""Unit tests for the performance and cost summariser (src/perf.py).

These use synthetic trace records so they run offline and deterministically.
"""

from __future__ import annotations

from src.perf import _percentile, summarize


def _rec(cost, seconds, calls=1, in_tok=100, out_tok=50, by_stage=None):
    return {
        "total_cost_usd": cost,
        "total_traced_seconds": seconds,
        "llm_calls": calls,
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "by_stage": by_stage or {},
    }


def test_empty_records():
    assert summarize([]) == {"queries": 0}


def test_percentile_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(values, 50) == 3.0
    assert _percentile(values, 95) == 5.0
    assert _percentile([], 95) == 0.0


def test_summary_aggregates_cost_and_latency():
    records = [_rec(0.01, 5.0), _rec(0.03, 15.0), _rec(0.02, 10.0)]
    s = summarize(records)
    assert s["queries"] == 3
    assert s["cost_usd"]["total"] == 0.06
    assert s["cost_usd"]["mean"] == 0.02
    assert s["latency_seconds"]["p50"] == 10.0
    assert s["latency_seconds"]["mean"] == 10.0


def test_summary_rolls_up_stages():
    records = [
        _rec(0.02, 8.0, by_stage={"synthesizer": {"calls": 1, "seconds": 8.0, "cost_usd": 0.02}}),
        _rec(0.02, 8.0, by_stage={"synthesizer": {"calls": 1, "seconds": 8.0, "cost_usd": 0.02}}),
    ]
    s = summarize(records)
    assert s["by_stage"]["synthesizer"]["calls"] == 2
    assert s["by_stage"]["synthesizer"]["cost_usd"] == 0.04
    assert s["by_stage"]["synthesizer"]["seconds"] == 16.0
