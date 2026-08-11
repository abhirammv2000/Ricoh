"""Concurrent load test for the agent.

You cannot claim a latency or throughput number you have not measured under
load. A single query timed once tells you nothing about tail latency or how the
system behaves when several requests arrive at the same time. This fires a batch
of real queries at a set concurrency and reports the numbers that actually
matter in production:

    p50 / p95 / p99 latency   what a typical and a worst-case user waits
    throughput (QPS)          completed queries per wall-clock second
    cost per query            mean and total dollar cost, from the traces
    error rate                failed queries as a fraction of the batch

Every run is wrapped in record_run, so the same traces the app writes are also
written here and can be inspected later with `python -m src.trace_view`.

Usage:
    python -m eval.loadtest                      # 12 queries, concurrency 4
    python -m eval.loadtest --n 24 --concurrency 8
    python -m eval.loadtest --planner --verifier # test the full agent path
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.config import PROJECT_ROOT
from src.agent import run_agent
from src.instrumentation import record_run

GROUND_TRUTH = PROJECT_ROOT / "eval" / "ground_truth.json"


@dataclass
class Result:
    query: str
    latency: float
    cost_usd: float
    ok: bool
    error: str = ""


def _load_queries(n: int) -> list[str]:
    """Take questions from the ground-truth set and repeat them up to n.

    Repeating is fine for a load test: we are measuring serving behaviour under
    concurrency, not answer quality, and repeats also exercise any cache once
    that lands.
    """
    data = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    base = [q["question"] for q in data["questions"]]
    out: list[str] = []
    while len(out) < n:
        out.extend(base)
    return out[:n]


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, so p95 of 20 samples is a real sample.

    statistics.quantiles interpolates, which invents a value between two
    measurements. For latency SLOs the nearest actual measurement is the honest
    choice, so we use the nearest-rank method here.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[rank - 1]


def _one_query(query: str, use_planner: bool, use_verifier: bool) -> Result:
    started = time.perf_counter()
    try:
        with record_run(query=query) as rec:
            run_agent(query, use_planner=use_planner, use_verifier=use_verifier)
        return Result(query, time.perf_counter() - started, rec.total_cost_usd, True)
    except Exception as exc:  # a single failed query must not abort the batch
        return Result(query, time.perf_counter() - started, 0.0, False, f"{type(exc).__name__}: {exc}")


def run_load_test(
    n: int, concurrency: int, use_planner: bool = False, use_verifier: bool = False
) -> dict:
    queries = _load_queries(n)

    # The agent prints progress to stdout on every node. Under concurrency those
    # lines interleave into noise, so we send them to a throwaway buffer for the
    # duration of the batch and print only the report afterwards.
    wall_start = time.perf_counter()
    results: list[Result] = []
    with contextlib.redirect_stdout(io.StringIO()):
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_one_query, q, use_planner, use_verifier) for q in queries
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
    wall = time.perf_counter() - wall_start

    ok = [r for r in results if r.ok]
    latencies = [r.latency for r in ok]
    costs = [r.cost_usd for r in ok]

    return {
        "requested": n,
        "concurrency": concurrency,
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "wall_seconds": round(wall, 2),
        "throughput_qps": round(len(ok) / wall, 3) if wall > 0 else 0.0,
        "latency_p50": round(_percentile(latencies, 50), 2),
        "latency_p95": round(_percentile(latencies, 95), 2),
        "latency_p99": round(_percentile(latencies, 99), 2),
        "latency_mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "latency_max": round(max(latencies), 2) if latencies else 0.0,
        "cost_per_query_mean": round(sum(costs) / len(costs), 5) if costs else 0.0,
        "cost_total": round(sum(costs), 4),
        "errors": [r.error for r in results if not r.ok][:5],
    }


def _print_report(s: dict) -> None:
    print("=" * 60)
    print("  Load test")
    print("=" * 60)
    print(f"  requested / completed / failed : {s['requested']} / {s['completed']} / {s['failed']}")
    print(f"  concurrency                    : {s['concurrency']}")
    print(f"  wall time                      : {s['wall_seconds']}s")
    print(f"  throughput                     : {s['throughput_qps']} queries/sec")
    print("  latency (seconds)")
    print(f"    p50 / p95 / p99              : {s['latency_p50']} / {s['latency_p95']} / {s['latency_p99']}")
    print(f"    mean / max                   : {s['latency_mean']} / {s['latency_max']}")
    print(f"  cost per query (mean)          : ${s['cost_per_query_mean']}")
    print(f"  cost total                     : ${s['cost_total']}")
    if s["errors"]:
        print(f"  sample errors                  : {s['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Concurrent load test for the agent")
    ap.add_argument("--n", type=int, default=12, help="total queries to send")
    ap.add_argument("--concurrency", type=int, default=4, help="concurrent workers")
    ap.add_argument("--planner", action="store_true", help="enable the planner stage")
    ap.add_argument("--verifier", action="store_true", help="enable the verifier stage")
    args = ap.parse_args()

    summary = run_load_test(
        args.n, args.concurrency, use_planner=args.planner, use_verifier=args.verifier
    )
    _print_report(summary)
