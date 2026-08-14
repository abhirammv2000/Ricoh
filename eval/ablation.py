"""eval/ablation.py - Does each pipeline stage earn its cost?

The question this answers
The production pipeline is Planner -> Retriever -> Verifier -> (retry) ->
Synthesizer, roughly four LLM calls per query.  Deterministic measurement
already shows that a *single* retrieval on the raw question finds every
expected document (8/8 at production settings) while the full pipeline
reaches 0.88, so on retrieval alone, the agentic layer is net negative.

That is not sufficient grounds to remove it.  Retrieval recall is not answer
quality:

  * The agent accumulates MORE evidence on some questions (Q4: 17 chunks,
    Q7: 8) because it retrieves once per sub-query. Extra context can raise
    groundedness even when the *expected* document set looks worse.
  * The verifier + retry exist to catch insufficient evidence, which shows
    up in refusal behaviour, not in recall.

So the honest experiment is a **progressive-removal ablation** (the standard
method for isolating component contribution): run the same benchmark, same
retriever, same judge, varying exactly one thing, how much pipeline runs.

The ladder
    A  retrieve_only   retrieve(raw question) -> synthesize        ~1 LLM call
    B  planner         + query decomposition + entity pass         ~2 calls
    C  full            + verifier and retry loop  (production)     ~4 calls

Each rung adds one mechanism, so a quality difference between adjacent rungs
is attributable to that mechanism.

What would justify each outcome
    A ties or beats C on quality  ->  delete the planner and verifier.
                                      No router needed; less code, ~4x cheaper.
    C beats A on SOME questions   ->  a router is justified, and the ablation
                                      data tells you what the routing
                                      criterion should key on.
    C beats A everywhere          ->  keep the pipeline; the retrieval-recall
                                      result was misleading and worth writing
                                      up as such.

Reading the results honestly
There are 10 questions. Judge noise on borderline answers is up to 0.10
(see eval/judge_variance.py). A difference of one question, or a mean shift
below ~0.10, is NOT a result. This script prints the per-question deltas
precisely so that a small mean difference cannot be quietly promoted into a
conclusion.

Cost: 3 configs x 10 questions = 30 agent runs + 30 judge calls.

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
