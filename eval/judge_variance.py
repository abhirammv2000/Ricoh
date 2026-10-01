"""Measure how noisy the LLM judge is.

Before saying a metric went from 0.97 to 0.98 you need to know whether the judge can resolve 0.01 (it
can't). This scores the same answer against the same evidence N times and reports the spread. Clear
answers score the same every time, and the variance sits on borderline answers, which are the ones that
move a mean. The evidence here is retrieved fresh, so the absolute scores differ from the harness's.
Compare the spread, not the values.

Usage:
    python -m eval.judge_variance            # default: 5 repeats
    python -m eval.judge_variance --repeats 10 --ids 7 9
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
from pathlib import Path

from src.config import PROJECT_ROOT, RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K
from src.eval_harness import _format_evidence_block, _judge
from src.retriever import get_retriever

METRICS_PATH: Path = PROJECT_ROOT / "eval" / "metrics.json"
GROUND_TRUTH_PATH: Path = PROJECT_ROOT / "eval" / "ground_truth.json"


def measure(ids: list[int], repeats: int) -> int:
    metrics = json.loads(io.open(METRICS_PATH, encoding="utf-8").read())
    rows = {r["id"]: r for r in metrics["per_question"]}
    truth = {
        q["id"]: q
        for q in json.loads(io.open(GROUND_TRUTH_PATH, encoding="utf-8").read())["questions"]
    }

    retriever = get_retriever()
    print(f"Judge: {metrics.get('judge_model')}   repeats: {repeats}\n")

    worst_spread = 0.0
    for qid in ids:
        row = rows.get(qid)
        if row is None:
            print(f"Q{qid}: not present in metrics.json - skipped")
            continue

        # retrieval is deterministic, so only the judge varies between repeats
        evidence = retriever.retrieve(
            query=row["question"], top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K
        )
        block = _format_evidence_block(evidence)
        expected = truth.get(qid, {}).get("expected_behavior", "answer")
        key_facts = truth.get(qid, {}).get("key_facts", [])

        grounded: list[float] = []
        correct: list[float] = []
        for _ in range(repeats):
            verdict = _judge(row["question"], row["answer"], block, key_facts, expected)
            grounded.append(verdict["groundedness"])
            correct.append(verdict["correctness"])

        for name, vals in (("groundedness", grounded), ("correctness", correct)):
            spread = max(vals) - min(vals)
            worst_spread = max(worst_spread, spread)
            print(
                f"Q{qid:<3}{name:<14}{vals}  "
                f"spread={spread:.2f}  stdev={statistics.pstdev(vals):.3f}"
            )
        print()

    print("=" * 68)
    print(f"Largest single-question spread observed: {worst_spread:.2f}")
    print(
        "Interpretation: differences in a per-question score smaller than this\n"
        "are indistinguishable from judge noise. Aggregate means inherit a\n"
        "smaller but non-zero share of it, on top of agent nondeterminism."
    )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure LLM-judge noise floor")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=[8, 9, 7],
        help="Question ids: mix an unambiguous case with borderline ones.",
    )
    args = ap.parse_args()
    raise SystemExit(measure(args.ids, args.repeats))
