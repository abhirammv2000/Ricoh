"""eval/judge_variance.py - Measure the LLM judge's own noise floor.

An LLM judge is a measuring instrument, and an instrument whose precision you
have not measured cannot support fine-grained claims.  Before saying "metric X
improved from 0.97 to 0.98", you have to know whether this judge can even
resolve a difference of 0.01.  (It cannot.)

This script scores the *same* answer against the *same* evidence N times and
reports the spread.  Anything smaller than that spread is noise, not signal.

Two properties worth knowing about the result:

* Unambiguous answers score with zero variance, the judge is not
  randomly jittering everything.
* Variance concentrates on borderline answers, which are exactly the ones
  that move an aggregate mean.  So the noise floor that matters is the
  borderline one, not the average one.

A caveat this script cannot remove: groundedness is only meaningful relative
to the evidence block the generator actually saw.  Reconstructing evidence
from a fresh retrieval (as here) yields different absolute scores than the
harness, which passes the agent's full accumulated evidence.  Compare the
*spread* across repeats, never these absolute values against harness output.

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

        # Retrieval is deterministic, so the evidence block is identical
        # across repeats and the only varying element is the judge itself.
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
