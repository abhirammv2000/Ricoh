"""Aggregate performance and cost report, computed from recorded traces.

trace_view.py inspects one request at a time. This is the fleet view: read every
trace and answer the questions a cost dashboard answers. What does a query cost
on average and at the 95th percentile? Where does the latency go? How many calls
per query? The numbers come from the same traces the app and the API write, so
the dashboard and the per-request view can never disagree.

Usage:
    python -m src.perf
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

from src.instrumentation import TRACE_PATH


def load_traces(path: Path = TRACE_PATH) -> list[dict[str, Any]]:
    """Read all trace records from the JSONL file, skipping any torn line."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, so a reported figure is a real measurement."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[rank - 1]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll a list of trace records up into dashboard metrics."""
    n = len(records)
    if n == 0:
        return {"queries": 0}

    costs = [float(r.get("total_cost_usd", 0.0)) for r in records]
    latencies = [float(r.get("total_traced_seconds", 0.0)) for r in records]
    calls = [int(r.get("llm_calls", 0)) for r in records]
    in_tokens = [int(r.get("total_input_tokens", 0)) for r in records]
    out_tokens = [int(r.get("total_output_tokens", 0)) for r in records]

    # Per-stage rollup across every record, so we can see which stage spends the
    # time and the money rather than only the totals.
    stages: dict[str, dict[str, float]] = {}
    for r in records:
        for stage, agg in (r.get("by_stage") or {}).items():
            acc = stages.setdefault(stage, {"calls": 0, "seconds": 0.0, "cost_usd": 0.0})
            acc["calls"] += agg.get("calls", 0)
            acc["seconds"] += agg.get("seconds", 0.0)
            acc["cost_usd"] += agg.get("cost_usd", 0.0)

    return {
        "queries": n,
        "cost_usd": {
            "total": round(sum(costs), 4),
            "mean": round(sum(costs) / n, 5),
            "p95": round(_percentile(costs, 95), 5),
        },
        "latency_seconds": {
            "p50": round(_percentile(latencies, 50), 2),
            "p95": round(_percentile(latencies, 95), 2),
            "p99": round(_percentile(latencies, 99), 2),
            "mean": round(sum(latencies) / n, 2),
        },
        "llm_calls_mean": round(sum(calls) / n, 2),
        "tokens_mean": {
            "input": round(sum(in_tokens) / n),
            "output": round(sum(out_tokens) / n),
        },
        "by_stage": {
            stage: {
                "calls": int(acc["calls"]),
                "seconds": round(acc["seconds"], 2),
                "cost_usd": round(acc["cost_usd"], 5),
            }
            for stage, acc in stages.items()
        },
    }


def _print_report(s: dict[str, Any]) -> None:
    if s["queries"] == 0:
        print("No traces yet. Run a query through the app, the API, or the load test.")
        return
    print("=" * 60)
    print("  Performance and cost report")
    print("=" * 60)
    print(f"  queries              : {s['queries']}")
    print(f"  cost total / mean     : ${s['cost_usd']['total']} / ${s['cost_usd']['mean']}")
    print(f"  cost p95 per query    : ${s['cost_usd']['p95']}")
    lat = s["latency_seconds"]
    print(f"  latency p50/p95/p99   : {lat['p50']} / {lat['p95']} / {lat['p99']} s")
    print(f"  latency mean          : {lat['mean']} s")
    print(f"  LLM calls per query   : {s['llm_calls_mean']}")
    print(f"  tokens per query      : {s['tokens_mean']['input']} in / {s['tokens_mean']['output']} out")
    if s["by_stage"]:
        print("  by stage:")
        for stage, agg in s["by_stage"].items():
            print(f"    {stage:<14} {agg['calls']:>4} calls  {agg['seconds']:>7.2f}s  ${agg['cost_usd']:.5f}")
    print("=" * 60)


if __name__ == "__main__":
    _print_report(summarize(load_traces()))
