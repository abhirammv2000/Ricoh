"""Inspect stored request traces.

Observability is only useful if you can actually answer a question with it.
This is the read side: given a request that behaved oddly, reconstruct what
happened, which chunks were retrieved, in what order, what each stage cost,
and where the time went.

Traces are written by record_run() to traces/traces.jsonl, one JSON object per
line appended as it finishes, so nothing else needs to be running to read them.

Usage:
    python -m src.trace_view                    # summary of recent traces
    python -m src.trace_view --last             # full detail of the newest
    python -m src.trace_view --id 9f2a...       # one trace by id
    python -m src.trace_view --slowest 5        # the worst latencies
    python -m src.trace_view --doc aiw00a13.pdf # traces that used a document
"""

from __future__ import annotations

import argparse
import io
import json
from typing import Any

from src.instrumentation import TRACE_PATH


def _load() -> list[dict[str, Any]]:
    if not TRACE_PATH.exists():
        raise SystemExit(
            f"No traces at {TRACE_PATH}. Run a query first, for example "
            "`python -m src.eval_harness --no-judge` or the Streamlit app."
        )
    traces = []
    with io.open(TRACE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn final line must not break the reader
    return traces


def _wall(t: dict[str, Any]) -> float:
    return sum(s.get("latency_seconds", 0.0) for s in t.get("spans", []))


def show_detail(t: dict[str, Any]) -> None:
    print("=" * 74)
    print(f"trace {t.get('trace_id')}   {t.get('started_at')}")
    print(f"query: {t.get('query', '')[:100]}")
    print(
        f"{t.get('llm_calls', 0)} LLM calls | ${t.get('total_cost_usd', 0):.5f} | "
        f"{_wall(t):.2f}s traced | "
        f"{t.get('total_input_tokens', 0):,} in / {t.get('total_output_tokens', 0):,} out"
    )
    print("-" * 74)
    for i, s in enumerate(t.get("spans", []), 1):
        kind = s.get("span_type", "llm")
        head = (
            f"{i}. [{kind}] {s['stage']:<14} {s.get('latency_seconds', 0):>6.2f}s"
        )
        if kind == "llm":
            head += (
                f"  ${s.get('cost_usd', 0):.5f}"
                f"  {s.get('input_tokens', 0):,}in/{s.get('output_tokens', 0):,}out"
                f"  {s.get('model', '')}"
            )
        print(head)
        if s.get("error"):
            print(f"      ERROR: {s['error']}")
        attrs = s.get("attributes") or {}
        chunks = attrs.get("chunks")
        if chunks:
            print(
                f"      retrieved {len(chunks)} chunks "
                f"(vector {attrs.get('vector_hits')}, bm25 {attrs.get('bm25_hits')}"
                f"{', reranked' if attrs.get('reranked') else ''})"
            )
            for c in chunks:
                print(f"        - {c['doc']} p.{c['page']}  rrf={c['rrf']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect stored request traces")
    ap.add_argument("--last", action="store_true", help="full detail of the newest trace")
    ap.add_argument("--id", help="full detail of one trace id (prefix ok)")
    ap.add_argument("--slowest", type=int, help="show the N slowest traces")
    ap.add_argument("--doc", help="only traces that retrieved this document")
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args()

    traces = _load()

    if args.doc:
        def used(t: dict[str, Any]) -> bool:
            for s in t.get("spans", []):
                for c in (s.get("attributes") or {}).get("chunks", []):
                    if c.get("doc") == args.doc:
                        return True
            return False

        traces = [t for t in traces if used(t)]
        print(f"{len(traces)} trace(s) retrieved {args.doc}\n")

    if args.id:
        matches = [t for t in traces if str(t.get("trace_id", "")).startswith(args.id)]
        if not matches:
            raise SystemExit(f"No trace id starting with {args.id}")
        for t in matches:
            show_detail(t)
        return 0

    if args.last:
        show_detail(traces[-1])
        return 0

    if args.slowest:
        for t in sorted(traces, key=_wall, reverse=True)[: args.slowest]:
            show_detail(t)
        return 0

    print(f"{len(traces)} trace(s) in {TRACE_PATH}\n")
    print(f"{'trace':<18}{'when':<22}{'calls':<7}{'cost':<10}{'sec':<8}query")
    for t in traces[-args.limit :]:
        print(
            f"{str(t.get('trace_id'))[:16]:<18}{str(t.get('started_at'))[:19]:<22}"
            f"{t.get('llm_calls', 0):<7}${t.get('total_cost_usd', 0):<9.5f}"
            f"{_wall(t):<8.2f}{str(t.get('query', ''))[:40]}"
        )
    total = sum(t.get("total_cost_usd", 0.0) for t in traces)
    print(f"\ntotal recorded spend: ${total:.4f}")
    print("Detail: python -m src.trace_view --last  |  --slowest 3  |  --doc <file.pdf>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
