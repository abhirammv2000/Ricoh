"""Export some of a harness run so RAGAS can check it from its own environment.

RAGAS 0.4.x needs langchain-core and langgraph 1.x, which clash with this project's langgraph 0.2.74, so
they can't share a virtualenv. This runs in the project env and writes a jsonl, and ragas_eval.py reads
it from an env with RAGAS. Per sampled question it writes:
  question, answer   straight from the metrics file
  contexts           the passages, retrieved again here (deterministic for a plain run, so it's what
                     the synthesizer saw, same assumption as label_for_kappa.py)
  judge_groundedness our judge's score, to compare against RAGAS question by question

    python -m eval.ragas_export --n 12
    # then, in the RAGAS env:  python eval/ragas_eval.py
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

from src.config import PROJECT_ROOT, RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K

DEFAULT_METRICS: Path = PROJECT_ROOT / "eval" / "metrics_n100.json"
OUT_PATH: Path = PROJECT_ROOT / "eval" / "ragas_input.jsonl"


def _contexts(question: str) -> list[str]:
    """The passages for a question, retrieved again, as plain text."""
    from src.retriever import get_retriever

    passages = get_retriever().retrieve(
        query=question, top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K
    )
    return [(p.get("text", "") or "").strip() for p in passages if p.get("text")]


def export(metrics_path: Path, n: int, seed: int, out_path: Path = OUT_PATH) -> int:
    report = json.loads(io.open(metrics_path, encoding="utf-8").read())
    if report.get("config", {}).get("use_planner"):
        print("WARNING: this run used the planner, so re-retrieved contexts may")
        print("  not match what the judge saw. Use a config-A run.")

    rows = [
        r
        for r in report["per_question"]
        if r.get("groundedness") is not None and not r["answer"].startswith("ERROR")
    ]
    sample = random.Random(seed).sample(rows, min(n, len(rows)))

    written = 0
    with io.open(out_path, "w", encoding="utf-8") as f:
        for r in sample:
            record = {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "contexts": _contexts(r["question"]),
                "judge_groundedness": r["groundedness"],
                "judge_correctness": r.get("correctness"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"  Q{r['id']}: {len(record['contexts'])} contexts")

    print(f"\nwrote {written} rows to {out_path}")
    print(f"source: {metrics_path.name}, agent {report.get('agent_model')}, "
          f"judge {report.get('judge_model')}")
    print("next (in the RAGAS env): python eval/ragas_eval.py")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export a harness slice for RAGAS")
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    ap.add_argument("--n", type=int, default=12, help="questions to sample")
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()
    raise SystemExit(export(args.metrics, args.n, args.seed))
