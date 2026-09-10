"""Unit tests for the performance and cost summariser (src/perf.py).

These use synthetic trace records so they run offline and deterministically.
"""

from __future__ import annotations

from src.perf import _percentile, format_trace, summarize


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


# --- per-request drill-down ---

_TRACE = {
    "trace_id": "abc123",
    "started_at": "2026-09-10T14:00:00",
    "query": "how do I shut it down?",
    "llm_calls": 1,
    "total_cost_usd": 0.0157,
    "total_traced_seconds": 10.2,
    "total_input_tokens": 3800,
    "total_output_tokens": 240,
    "spans": [
        {
            "stage": "retrieval", "span_type": "retrieval", "latency_seconds": 0.3,
            "attributes": {
                "vector_hits": 10, "bm25_hits": 10, "reranked": False,
                "chunks": [
                    {"doc": "aiw00a13.pdf", "page": 1, "rrf": 0.0312},
                    {"doc": "aiw00p18.pdf", "page": 1, "rrf": 0.0164},
                ],
            },
        },
        {
            "stage": "synthesizer", "span_type": "llm", "latency_seconds": 9.9,
            "cost_usd": 0.0157, "input_tokens": 3800, "output_tokens": 240,
            "model": "claude-sonnet-4-6", "attributes": {},
        },
        {
            "stage": "citation_guardrail", "span_type": "retrieval", "latency_seconds": 0.0,
            "attributes": {"valid": False, "cited": ["ghost.pdf"], "fabricated": ["ghost.pdf"]},
        },
    ],
}


def test_format_trace_carries_chunk_attribution():
    t = format_trace(_TRACE)
    assert t["trace_id"] == "abc123"
    assert t["llm_calls"] == 1
    retrieval = next(s for s in t["spans"] if s["stage"] == "retrieval")
    assert retrieval["retrieval"]["chunks"][0] == {"doc": "aiw00a13.pdf", "page": 1, "rrf": 0.0312}
    assert retrieval["retrieval"]["reranked"] is False


def test_format_trace_surfaces_a_fabricated_citation():
    guard = next(s for s in format_trace(_TRACE)["spans"] if s["stage"] == "citation_guardrail")
    assert guard["citation_guardrail"]["valid"] is False
    assert guard["citation_guardrail"]["fabricated"] == ["ghost.pdf"]


def test_format_trace_is_safe_on_a_sparse_record():
    t = format_trace({"trace_id": "x"})
    assert t["spans"] == []
    assert t["total_cost_usd"] == 0.0
