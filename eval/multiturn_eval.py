"""Judged evaluation of multi-turn follow-up handling (src/conversation.py).

The unit tests prove condensation runs; this measures whether it works. For
every chain in eval/multiturn_questions.json it walks the turns in order,
carrying the real prior answers as history, and for each turn records:

  * what condensation rewrote the follow-up into, and how close that is to the
    hand-written 'standalone' target (offline MiniLM cosine)
  * retriever recall on the RAW follow-up vs the rewritten question, so the
    retrieval lift from condensation is visible and is free to compute
  * groundedness, correctness, evidence recall and behaviour match, judged with
    the same LLM judge as the main harness, scored against the standalone intent

Aggregates split turn 1 (already standalone, a control) from the follow-ups.

    python -m eval.multiturn_eval              # full judged run (needs API key)
    python -m eval.multiturn_eval --no-judge   # rewrite + retrieval metrics only
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import JUDGE_MODEL, PROJECT_ROOT, RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K
from src.conversation import Turn, condense_query
from src.instrumentation import record_run

QUESTIONS = PROJECT_ROOT / "eval" / "multiturn_questions.json"
RESULT_JSON = PROJECT_ROOT / "eval" / "multiturn_metrics.json"
RESULT_MD = PROJECT_ROOT / "eval" / "multiturn_report.md"

_EMBED = None


def _embed(texts: list[str]) -> np.ndarray:
    global _EMBED
    if _EMBED is None:
        from chromadb.utils import embedding_functions

        _EMBED = embedding_functions.ONNXMiniLM_L6_V2()
    v = np.asarray(_EMBED(texts), dtype=np.float64)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def _cosine(a: str, b: str) -> float:
    m = _embed([a, b])
    return round(float(m[0] @ m[1]), 3)


def _distinct_docs(evidence: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for e in evidence:
        d = e.get("source_document")
        if d and d not in out:
            out.append(d)
    return out


def _recall(expected: list[str], docs: list[str]) -> float:
    """1.0 if any expected document reached the top final_k, else 0.0.

    Any-hit, not fraction: for a support follow-up the question is "did the
    answer have a document that can answer it", and several turns list more
    than one document that independently would (e.g. the generic
    custom-properties article and the document-specific one).
    """
    if not expected:
        return 0.0
    return 1.0 if set(expected) & set(docs[:RETRIEVAL_FINAL_K]) else 0.0


def evaluate(use_judge: bool = True) -> dict[str, Any]:
    from src.eval_harness import _format_evidence_block, _judge, _run_agent_full
    from src.retriever import get_retriever

    chains = json.loads(io.open(QUESTIONS, encoding="utf-8").read())["chains"]
    retriever = get_retriever()
    rows: list[dict[str, Any]] = []

    for chain in chains:
        history: list[Turn] = []
        for pos, turn in enumerate(chain["turns"]):
            raw = turn["question"]
            standalone_target = turn["standalone"]
            is_followup = pos > 0

            with record_run(query=raw) as run:
                rewritten = condense_query(history, raw) if is_followup else raw
                state = _run_agent_full(rewritten, use_planner=False, use_verifier=False)
            answer = state.get("final_answer", "")
            evidence = state.get("retrieved_evidence", [])
            answer_docs = _distinct_docs(evidence)

            # Free retrieval control: raw follow-up vs the rewrite.
            raw_docs = _distinct_docs(
                retriever.retrieve(raw, top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K)
            )

            refused = "information unavailable" in " ".join(answer.lower().split())
            row: dict[str, Any] = {
                "chain": chain["id"],
                "turn": pos + 1,
                "is_followup": is_followup,
                "raw": raw,
                "rewritten": rewritten,
                "standalone_target": standalone_target,
                "rewrite_cosine_to_target": _cosine(rewritten, standalone_target)
                if is_followup
                else None,
                "recall_raw_followup": round(_recall(turn["expected_sources"], raw_docs), 3),
                "recall_after_condense": round(_recall(turn["expected_sources"], answer_docs), 3),
                "evidence_recall": round(_recall(turn["expected_sources"], answer_docs), 3),
                "expected_sources": turn["expected_sources"],
                "system_refused": refused,
                "behavior_match": (turn["expected_behavior"] == "answer") != refused,
                "answer": answer,
                "cost_usd": run.total_cost_usd,
            }

            if use_judge and not answer.startswith("ERROR"):
                with record_run(persist=False) as jrun:
                    verdict = _judge(
                        standalone_target,
                        answer,
                        _format_evidence_block(evidence),
                        turn["key_facts"],
                        turn["expected_behavior"],
                    )
                row["groundedness"] = round(verdict["groundedness"], 3)
                row["correctness"] = round(verdict["correctness"], 3)
                row["judge_cost_usd"] = jrun.total_cost_usd

            rows.append(row)
            print(
                f"  {chain['id']}/turn{pos + 1}: "
                f"rewrite~{row['rewrite_cosine_to_target']} "
                f"recall {row['recall_raw_followup']}->{row['recall_after_condense']} "
                f"grounded={row.get('groundedness')} correct={row.get('correctness')}"
            )
            history.append(Turn(question=raw, answer=answer))

    return _aggregate(rows, use_judge)


def _mean(vals: list[float | None]) -> float | None:
    nums = [v for v in vals if v is not None]
    return round(sum(nums) / len(nums), 3) if nums else None


def _aggregate(rows: list[dict[str, Any]], use_judge: bool) -> dict[str, Any]:
    followups = [r for r in rows if r["is_followup"]]
    firsts = [r for r in rows if not r["is_followup"]]

    def block(group: list[dict[str, Any]]) -> dict[str, Any]:
        b = {
            "n": len(group),
            "evidence_recall": _mean([r["evidence_recall"] for r in group]),
            "behavior_match_rate": _mean([float(r["behavior_match"]) for r in group]),
        }
        if use_judge:
            b["groundedness"] = _mean([r.get("groundedness") for r in group])
            b["correctness"] = _mean([r.get("correctness") for r in group])
        return b

    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": JUDGE_MODEL if use_judge else None,
        "turns": len(rows),
        "first_turns": block(firsts),
        "followups": block(followups),
        "condensation": {
            "mean_rewrite_cosine_to_target": _mean(
                [r["rewrite_cosine_to_target"] for r in followups]
            ),
            "mean_recall_raw_followup": _mean([r["recall_raw_followup"] for r in followups]),
            "mean_recall_after_condense": _mean(
                [r["recall_after_condense"] for r in followups]
            ),
        },
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows), 5),
        "judge_cost_usd": round(sum(r.get("judge_cost_usd", 0.0) for r in rows), 5),
    }
    return {"summary": summary, "per_turn": rows}


def _write_md(report: dict[str, Any]) -> None:
    s = report["summary"]
    c = s["condensation"]
    lines = [
        "# Multi-turn follow-up evaluation",
        "",
        f"**Generated:** {s['generated']}  ",
        f"**Judge:** `{s['judge_model'] or 'disabled'}`  ",
        f"**Turns:** {s['turns']} across {s['turns'] // 3} chains  ",
        "",
        "`standalone_target` is the hand-written ideal rewrite. Follow-up turns "
        "carry the real prior answers as history.",
        "",
        "## Condensation",
        "",
        f"- Mean cosine(rewrite, target) on follow-ups: **{c['mean_rewrite_cosine_to_target']}**",
        f"- Retriever recall on the raw follow-up: {c['mean_recall_raw_followup']}",
        f"- Retriever recall after condensation: **{c['mean_recall_after_condense']}**",
        "",
        "The recall lift is the point: a follow-up like \"what about those?\" "
        "retrieves nothing on its own; the rewrite makes it a real query.",
        "",
        "## Answer quality, first turn vs follow-ups",
        "",
        "| | first turns | follow-ups |",
        "|---|---|---|",
        f"| n | {s['first_turns']['n']} | {s['followups']['n']} |",
        f"| evidence recall | {s['first_turns']['evidence_recall']} | {s['followups']['evidence_recall']} |",
        f"| behaviour match | {s['first_turns']['behavior_match_rate']} | {s['followups']['behavior_match_rate']} |",
    ]
    if s["judge_model"]:
        lines += [
            f"| groundedness | {s['first_turns']['groundedness']} | {s['followups']['groundedness']} |",
            f"| correctness | {s['first_turns']['correctness']} | {s['followups']['correctness']} |",
        ]
    lines += [
        "",
        f"Agent cost ${s['total_cost_usd']}, judge overhead ${s['judge_cost_usd']}.",
        "",
        "## Per-turn",
        "",
        "| chain | turn | rewrite~target | recall raw->condensed | grounded | correct |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["per_turn"]:
        lines.append(
            f"| {r['chain']} | {r['turn']} | {r['rewrite_cosine_to_target'] if r['is_followup'] else '-'} "
            f"| {r['recall_raw_followup']}->{r['recall_after_condense']} "
            f"| {r.get('groundedness', '-')} | {r.get('correctness', '-')} |"
        )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Judged multi-turn evaluation")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    args = ap.parse_args()

    print("=" * 70)
    print("  Citera multi-turn evaluation")
    print("=" * 70)
    report = evaluate(use_judge=not args.no_judge)
    RESULT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(report)
    s = report["summary"]
    print(f"\nMetrics -> {RESULT_JSON}")
    print(f"Report  -> {RESULT_MD}")
    print(
        f"\nfollow-ups: recall {s['condensation']['mean_recall_raw_followup']} raw "
        f"-> {s['condensation']['mean_recall_after_condense']} condensed, "
        f"grounded {s['followups'].get('groundedness')}, "
        f"correct {s['followups'].get('correctness')}"
    )
