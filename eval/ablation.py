"""Progressive-removal ablation: does each pipeline stage pay for itself?

Runs the same benchmark with the same retriever and judge, varying only how much
of the pipeline runs:

    A  retrieve_only   retrieve(raw question) -> synthesize        ~1 LLM call
    B  planner         + query decomposition + entity pass         ~2 calls
    C  full            + verifier and retry loop                   ~3 calls

Each rung adds one mechanism, so a difference between adjacent rungs is
attributable to it. Judge noise on borderline answers runs up to 0.10, so a
difference under that, or one or two questions flipping, is not a result; the
script prints per-question deltas for that reason.

The default run is the curated 10-question set. `--n100` runs the 70-question
dev split with the judge; that run revised the n=10 conclusion (README section
7): the verifier still earns nothing, the planner helps on dev but not holdout.

Usage:
    python -m eval.ablation                 # curated 10 questions
    python -m eval.ablation --n100          # 70-question dev split
    python -m eval.ablation --configs A C   # just the endpoints
    python -m eval.ablation --no-judge      # objective metrics only
    python -m eval.ablation --ground-truth eval/generated_questions.json --split holdout --configs A B --no-judge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.eval_harness import GROUND_TRUTH_PATH, evaluate, write_outputs

OUT_DIR: Path = PROJECT_ROOT / "eval" / "ablation"
GENERATED_QUESTIONS_PATH: Path = PROJECT_ROOT / "eval" / "generated_questions.json"

CONFIGS: dict[str, dict[str, Any]] = {
    "A": {
        "name": "retrieve_only",
        "use_planner": False,
        "use_verifier": False,
        "describes": "raw question -> retrieve -> synthesize",
    },
    "B": {
        "name": "planner",
        "use_planner": True,
        "use_verifier": False,
        "describes": "+ sub-query decomposition and entity pass",
    },
    "C": {
        "name": "full",
        "use_planner": True,
        "use_verifier": True,
        "describes": "+ verifier and retry loop",
    },
}


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _mean_of(summary: dict[str, Any], key: str) -> float | None:
    stat = summary.get(key)
    if isinstance(stat, dict):
        return stat.get("mean")
    return stat


def run(
    config_keys: list[str],
    use_judge: bool,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    split: str | None = None,
) -> int:
    # The 10-question run keeps writing into eval/ablation/ (the README links its
    # files). Anything else goes to a named subdir so they don't clobber.
    is_default = ground_truth_path == GROUND_TRUTH_PATH and split is None
    out_dir = OUT_DIR
    if not is_default:
        tag = ground_truth_path.stem
        if split:
            tag += f"_{split}"
        out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}

    for key in config_keys:
        cfg = CONFIGS[key]
        print("\n" + "=" * 72)
        print(f"  CONFIG {key} - {cfg['name']}: {cfg['describes']}")
        print("=" * 72)
        report = evaluate(
            use_judge=use_judge,
            use_planner=cfg["use_planner"],
            use_verifier=cfg["use_verifier"],
            ground_truth_path=ground_truth_path,
            split=split,
        )
        write_outputs(
            report,
            metrics_path=out_dir / f"{key}_{cfg['name']}.json",
            report_path=out_dir / f"{key}_{cfg['name']}.md",
        )
        reports[key] = report

    _print_comparison(reports, use_judge)
    (out_dir / "comparison.json").write_text(
        json.dumps(
            {k: r["summary"] for k, r in reports.items()}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(f"\nComparison written to {out_dir / 'comparison.json'}")
    return 0


def _print_comparison(reports: dict[str, dict[str, Any]], use_judge: bool) -> None:
    print("\n" + "=" * 72)
    print("  ABLATION COMPARISON")
    print("=" * 72)

    header = f"{'cfg':<5}{'calls':<8}{'cost/q':<11}{'latency':<10}{'evid.rec':<10}"
    if use_judge:
        header += f"{'grounded':<11}{'correct':<10}"
    header += "behaviour"
    print(header)

    for key, rep in reports.items():
        s = rep["summary"]
        cost = s.get("cost", {})
        row = (
            f"{key:<5}"
            f"{cost.get('mean_llm_calls_per_query', '?'):<8}"
            f"${cost.get('mean_cost_per_query_usd', 0):<10.5f}"
            f"{s.get('mean_latency_seconds', 0):<10.1f}"
            f"{_fmt(_mean_of(s, 'evidence_recall')):<10}"
        )
        if use_judge:
            row += (
                f"{_fmt(_mean_of(s, 'groundedness')):<11}"
                f"{_fmt(_mean_of(s, 'correctness')):<10}"
            )
        row += _fmt(s.get("behavior_match_rate"))
        print(row)

    # Per-question deltas against the cheapest config: a mean can hide a
    # config that wins on some questions and loses on others, which is
    # exactly the pattern that would justify a router.
    base_key = list(reports)[0]
    base_rows = {r["id"]: r for r in reports[base_key]["per_question"]}
    for key, rep in list(reports.items())[1:]:
        print(f"\nPer-question delta: {key} minus {base_key}")
        print(f"  {'Q':<4}{'d_evid_recall':<16}{'d_grounded':<14}{'d_correct':<12}{'d_cost':<12}")
        for row in rep["per_question"]:
            b = base_rows.get(row["id"], {})

            def delta(field: str) -> str:
                x, y = row.get(field), b.get(field)
                if x is None or y is None:
                    return "n/a"
                d = x - y
                return f"{d:+.2f}" if d else "  ."

            dc = row.get("cost_usd", 0) - b.get("cost_usd", 0)
            print(
                f"  {row['id']:<4}{delta('evidence_recall'):<16}"
                f"{delta('groundedness'):<14}{delta('correctness'):<12}"
                f"${dc:+.5f}"
            )

    n = next(iter(reports.values()))["summary"]["questions"]
    print("\n" + "-" * 72)
    print("Interpretation guard-rails:")
    print(f"  * n={n}. Judge noise on borderline answers reaches 0.10.")
    print(f"    A mean difference under ~0.10, or {'a single question' if n <= 20 else 'one or two questions'} flipping,")
    print("    is NOT evidence of a real effect.")
    print("  * If the cheap config ties, the expensive stages are not earning")
    print("    their cost on THIS benchmark - which is a statement about this")
    print("    corpus and question set, not about agentic RAG in general.")
    print("  * A split result (cheap wins some, expensive wins others) is the")
    print("    only outcome that actually justifies building a router.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline component ablation")
    ap.add_argument("--configs", nargs="+", default=["A", "B", "C"], choices=list(CONFIGS))
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument(
        "--ground-truth", type=Path, default=GROUND_TRUTH_PATH,
        help="Question set to run (default: the curated 10-question set)",
    )
    ap.add_argument(
        "--split", default=None,
        help="Only run questions with this split value, e.g. dev / holdout",
    )
    ap.add_argument(
        "--n100", action="store_true",
        help="Shorthand for --ground-truth eval/generated_questions.json --split dev",
    )
    args = ap.parse_args()

    gt_path = args.ground_truth
    split = args.split
    if args.n100:
        gt_path = GENERATED_QUESTIONS_PATH
        split = "dev"

    raise SystemExit(
        run(args.configs, use_judge=not args.no_judge, ground_truth_path=gt_path, split=split)
    )
