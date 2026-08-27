"""Progressive-removal ablation: does each pipeline stage pay for itself?

The pipeline is Planner -> Retriever -> Verifier -> retry -> Synthesizer, about
four LLM calls a query. A single retrieval on the raw question already finds
every expected document (8/8) while the full pipeline reaches 0.88, so on
retrieval alone the agentic layer is net negative.

That alone is not enough to remove it. The agent accumulates more evidence on
some questions because it retrieves once per sub-query, and the verifier exists
to catch insufficient evidence, which shows up in refusal behaviour rather than
in recall. So run the same benchmark with the same retriever and judge, varying
only how much of the pipeline runs:

    A  retrieve_only   retrieve(raw question) -> synthesize        ~1 LLM call
    B  planner         + query decomposition + entity pass         ~2 calls
    C  full            + verifier and retry loop                   ~4 calls

Each rung adds one mechanism, so a difference between adjacent rungs is
attributable to that mechanism. If A ties or beats C, the planner and verifier
go. If C wins on some questions, a router is justified and the data says what it
should key on.

Note there are only 10 questions here, and judge noise on borderline answers
runs up to 0.10 (see eval/judge_variance.py), so a one-question difference is
not a result. The script prints per-question deltas for that reason.

Cost: 3 configs x 10 questions = 30 agent runs plus 30 judge calls.

Usage:
    python -m eval.ablation                 # all three configs
    python -m eval.ablation --configs A C   # just the endpoints
    python -m eval.ablation --no-judge      # objective metrics only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.eval_harness import evaluate, write_outputs

OUT_DIR: Path = PROJECT_ROOT / "eval" / "ablation"

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
        "describes": "+ verifier and retry (production default)",
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


def run(config_keys: list[str], use_judge: bool) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
        )
        write_outputs(
            report,
            metrics_path=OUT_DIR / f"{key}_{cfg['name']}.json",
            report_path=OUT_DIR / f"{key}_{cfg['name']}.md",
        )
        reports[key] = report

    _print_comparison(reports, use_judge)
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(
            {k: r["summary"] for k, r in reports.items()}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(f"\nComparison written to {OUT_DIR / 'comparison.json'}")
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

    print("\n" + "-" * 72)
    print("Interpretation guard-rails:")
    print("  * n=10. Judge noise on borderline answers reaches 0.10.")
    print("    A mean difference under ~0.10, or a single question flipping,")
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
    args = ap.parse_args()
    raise SystemExit(run(args.configs, use_judge=not args.no_judge))
