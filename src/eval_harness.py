"""
src/eval_harness.py - Quality-focused evaluation harness for RicohLibrary.

The original ``src/evaluate.py`` only measured *latency* and whether a
citation regex matched.  It never checked whether answers were
**correct** or **faithful** to the sources - so it could not answer the
one question any company will ask: *"How do you know it's good?"*

This harness measures RAG quality with four metrics:

1. **Retrieval recall@k** (objective, no LLM)
   Of the documents we EXPECT to be relevant (from ground_truth.json),
   how many actually showed up in the retrieved evidence?  This is the
   metric that catches retrieval misses - e.g. a question that gets
   refused only because the right document was never retrieved.

2. **Citation precision** (objective, no LLM)
   Of the documents the answer CITES, how many are actually present in
   the retrieved evidence?  Guards against fabricated citations.

3. **Groundedness / faithfulness** (LLM-judged, 0-1)
   The gold-standard RAG metric: is every claim in the answer supported
   by the retrieved evidence, with no hallucination?  Needs no external
   ground truth, so it is non-circular.

4. **Answer correctness** (LLM-judged, 0-1)
   Does the answer convey the curated ``key_facts`` (or correctly refuse
   when ``expected_behavior == 'refuse'``)?

It also records a **behaviour match** (did the system answer vs. refuse
when it should have?) which surfaces probable retrieval misses.

Outputs:
  • eval/metrics.json   - machine-readable aggregate + per-question scores
  • eval/eval_report.md - human-readable report

Usage:
    python -m src.eval_harness            # run the full harness
    python -m src.eval_harness --no-judge # objective metrics only (no API)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure config.py runs first (logging + telemetry silencing)
from src.config import PROJECT_ROOT

# Quiet the noisy ingest/retriever logs during evaluation
for _quiet in ("src.ingest", "src.retriever"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

from src.agent import AgentState, build_agent_graph  # noqa: E402
from src.ingest import ingest_all  # noqa: E402
from src.llm_factory import get_llm  # noqa: E402
from src.retriever import HybridRetriever  # noqa: E402

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH: Path = PROJECT_ROOT / "eval" / "ground_truth.json"
METRICS_PATH: Path = PROJECT_ROOT / "eval" / "metrics.json"
REPORT_PATH: Path = PROJECT_ROOT / "eval" / "eval_report.md"

# A refusal is detected by this canonical phrase (the synthesizer is
# instructed to emit it verbatim, in the user's language - we match the
# English form used by the test set).
REFUSAL_MARKER = "information unavailable"


# ====================================================================
# 1. LLM-AS-JUDGE
# ====================================================================

JUDGE_PROMPT = """\
You are a strict evaluator for a retrieval-augmented technical-support \
system. You will score an ANSWER against the EVIDENCE it was given and \
against a list of expected KEY FACTS.

Question:
\"\"\"{question}\"\"\"

Evidence the system retrieved:
{evidence_block}

System's answer:
\"\"\"{answer}\"\"\"

Expected key facts the answer should convey (may be empty):
{key_facts}

Expected behaviour: {expected_behavior}
  - "answer"  : the corpus should contain the answer; refusing is WRONG.
  - "refuse"  : the corpus does NOT contain the answer; refusing is CORRECT.

Score two things from 0.0 to 1.0:

1. groundedness: Is every factual claim in the answer supported by the \
EVIDENCE above? 1.0 = fully supported, 0.0 = mostly fabricated. A correct \
refusal ("information unavailable") is fully grounded (1.0).

2. correctness: Does the answer satisfy the expected behaviour and convey \
the key facts? If expected behaviour is "answer", a refusal scores 0.0. If \
"refuse", a correct refusal scores 1.0. If key facts are empty and \
behaviour is "answer", judge whether the answer plausibly and \
specifically addresses the question.

Respond with ONLY a valid JSON object, no markdown fences:
{{"groundedness": <float>, "correctness": <float>, "rationale": "<one short sentence>"}}
"""


def _judge(
    question: str,
    answer: str,
    evidence_block: str,
    key_facts: list[str],
    expected_behavior: str,
) -> dict[str, Any]:
    """Run the LLM judge for one question. Resilient to bad JSON."""
    llm = get_llm()
    prompt = JUDGE_PROMPT.format(
        question=question,
        evidence_block=evidence_block or "(no evidence)",
        answer=answer,
        key_facts=json.dumps(key_facts, ensure_ascii=False) or "[]",
        expected_behavior=expected_behavior,
    )
    raw = llm.invoke(prompt).content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        return {
            "groundedness": float(parsed.get("groundedness", 0.0)),
            "correctness": float(parsed.get("correctness", 0.0)),
            "rationale": str(parsed.get("rationale", "")),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Judge returned unparseable output: %s", raw[:120])
        return {"groundedness": 0.0, "correctness": 0.0, "rationale": "JUDGE_PARSE_ERROR"}


# ====================================================================
# 2. OBJECTIVE METRICS (no LLM)
# ====================================================================

_CITATION_RE = re.compile(r"\[([^\]]+?),\s*Page\s*\d+\]", re.IGNORECASE)


def _cited_docs(answer: str) -> set[str]:
    """Distinct document names cited in the answer text."""
    return {m.strip() for m in _CITATION_RE.findall(answer)}


def _evidence_docs(evidence: list[dict[str, Any]]) -> set[str]:
    """Distinct source documents present in the retrieved evidence."""
    return {e.get("source_document", "") for e in evidence if e.get("source_document")}


def _retrieval_recall(expected: list[str], evidence_docs: set[str]) -> float | None:
    """Fraction of expected source docs found in the retrieved evidence.

    Returns None when no expected sources are defined (can't be scored).
    """
    if not expected:
        return None
    hits = sum(1 for d in expected if d in evidence_docs)
    return hits / len(expected)


def _citation_precision(cited: set[str], evidence_docs: set[str]) -> float | None:
    """Fraction of cited docs that are actually in the evidence.

    Returns None when the answer cites nothing (e.g. a refusal).
    """
    if not cited:
        return None
    valid = sum(1 for d in cited if d in evidence_docs)
    return valid / len(cited)


def _is_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in answer.lower()


# ====================================================================
# 3. AGENT RUNNER (returns full state, like the Streamlit app)
# ====================================================================

def _run_agent_full(query: str) -> dict[str, Any]:
    agent = build_agent_graph()
    initial: AgentState = {
        "user_query": query,
        "sub_queries": [],
        "entities": [],
        "retrieved_evidence": [],
        "verification_status": "",
        "final_answer": "",
        "iterations": 0,
    }
    return dict(agent.invoke(initial))


def _format_evidence_block(evidence: list[dict[str, Any]], limit: int = 12) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for i, e in enumerate(evidence[:limit], 1):
        src = e.get("source_document", "unknown")
        page = e.get("page_number", "?")
        text = (e.get("text", "") or "")[:600]
        lines.append(f"[{i}] {src}, Page {page}\n{text}")
    return "\n\n".join(lines)


# ====================================================================
# 4. MAIN EVALUATION LOOP
# ====================================================================

def evaluate(use_judge: bool = True) -> dict[str, Any]:
    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    questions = gt["questions"]

    per_question: list[dict[str, Any]] = []

    for item in questions:
        qid = item["id"]
        question = item["question"]
        expected_behavior = item.get("expected_behavior", "answer")
        key_facts = item.get("key_facts", [])
        expected_sources = item.get("expected_sources", [])

        print(f"\n{'━' * 70}\n  Q{qid}: {question}\n{'━' * 70}")

        t0 = time.perf_counter()
        try:
            state = _run_agent_full(question)
            answer = state.get("final_answer", "")
            evidence = state.get("retrieved_evidence", [])
            iterations = state.get("iterations", 0)
        except Exception as exc:  # pragma: no cover - depends on live API
            logger.error("Agent failed on Q%d: %s", qid, exc)
            answer, evidence, iterations = f"ERROR: {exc}", [], 0
        latency = round(time.perf_counter() - t0, 2)

        ev_docs = _evidence_docs(evidence)
        cited = _cited_docs(answer)
        refused = _is_refusal(answer)
        behavior_match = (
            (expected_behavior == "refuse" and refused)
            or (expected_behavior == "answer" and not refused)
        )

        row: dict[str, Any] = {
            "id": qid,
            "question": question,
            "expected_behavior": expected_behavior,
            "system_refused": refused,
            "behavior_match": behavior_match,
            "latency_seconds": latency,
            "iterations": iterations,
            "evidence_chunks": len(evidence),
            "retrieval_recall": _retrieval_recall(expected_sources, ev_docs),
            "citation_precision": _citation_precision(cited, ev_docs),
            "needs_verification": item.get("needs_verification", False),
            "answer": answer,
        }

        if use_judge and not answer.startswith("ERROR"):
            verdict = _judge(
                question,
                answer,
                _format_evidence_block(evidence),
                key_facts,
                expected_behavior,
            )
            row.update(
                groundedness=round(verdict["groundedness"], 3),
                correctness=round(verdict["correctness"], 3),
                judge_rationale=verdict["rationale"],
            )

        per_question.append(row)
        _print_row(row)

    return _aggregate(per_question, use_judge)


def _print_row(row: dict[str, Any]) -> None:
    rr = row["retrieval_recall"]
    cp = row["citation_precision"]
    print(
        f"  behaviour_match={row['behavior_match']} | "
        f"recall@k={'n/a' if rr is None else f'{rr:.2f}'} | "
        f"cite_prec={'n/a' if cp is None else f'{cp:.2f}'} | "
        f"grounded={row.get('groundedness', 'n/a')} | "
        f"correct={row.get('correctness', 'n/a')} | {row['latency_seconds']}s"
    )


def _avg(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 3) if nums else None


def _aggregate(rows: list[dict[str, Any]], use_judge: bool) -> dict[str, Any]:
    n = len(rows)
    summary = {
        "questions": n,
        "behavior_match_rate": round(sum(r["behavior_match"] for r in rows) / n, 3),
        "mean_retrieval_recall": _avg([r["retrieval_recall"] for r in rows]),
        "mean_citation_precision": _avg([r["citation_precision"] for r in rows]),
        "mean_latency_seconds": round(sum(r["latency_seconds"] for r in rows) / n, 2),
    }
    if use_judge:
        summary["mean_groundedness"] = _avg([r.get("groundedness") for r in rows])
        summary["mean_correctness"] = _avg([r.get("correctness") for r in rows])

    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "judge_enabled": use_judge,
        "summary": summary,
        "per_question": rows,
    }


# ====================================================================
# 5. REPORT WRITERS
# ====================================================================

def write_outputs(report: dict[str, Any]) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📊  Metrics  → {METRICS_PATH}")

    s = report["summary"]
    lines: list[str] = [
        "# 📊 RicohLibrary — Quality Evaluation Report",
        "",
        f"**Generated:** {report['generated']}  ",
        f"**LLM judge:** {'enabled' if report['judge_enabled'] else 'disabled (objective metrics only)'}  ",
        f"**Questions:** {s['questions']}  ",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Score | What it means |",
        "|---|---|---|",
        f"| Behaviour-match rate | {s['behavior_match_rate']:.0%} | Answered vs. refused as expected |",
        f"| Mean retrieval recall@k | {_fmt(s['mean_retrieval_recall'])} | Expected source docs that were retrieved |",
        f"| Mean citation precision | {_fmt(s['mean_citation_precision'])} | Cited docs that are real (no fabricated cites) |",
    ]
    if report["judge_enabled"]:
        lines += [
            f"| Mean groundedness | {_fmt(s.get('mean_groundedness'))} | Claims supported by evidence (no hallucination) |",
            f"| Mean correctness | {_fmt(s.get('mean_correctness'))} | Conveys expected facts / refuses correctly |",
        ]
    lines += [
        f"| Mean latency | {s['mean_latency_seconds']}s | Per-question wall-clock |",
        "",
        "## Per-question results",
        "",
        "| # | Behaviour ✓ | Recall@k | Cite prec. | Grounded | Correct | Latency | Flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["per_question"]:
        flags = []
        if r.get("needs_verification"):
            flags.append("⚠ verify")
        if not r["behavior_match"]:
            flags.append("✗ behaviour")
        lines.append(
            f"| {r['id']} | {'✅' if r['behavior_match'] else '❌'} "
            f"| {_fmt(r['retrieval_recall'])} | {_fmt(r['citation_precision'])} "
            f"| {_fmt(r.get('groundedness'))} | {_fmt(r.get('correctness'))} "
            f"| {r['latency_seconds']}s | {', '.join(flags) or '—'} |"
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"📝  Report   → {REPORT_PATH}")


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


# ====================================================================
# __main__
# ====================================================================

def _ensure_index() -> None:
    retriever = HybridRetriever()
    if retriever.index_size == 0 or not retriever.bm25_ready:
        print("📄  Index missing — ingesting PDFs from data/ …")
        chunks = ingest_all()
        if not chunks:
            raise SystemExit("❌  No PDFs found in data/. Add PDFs and retry.")
        retriever.build_index(chunks)
    print(f"📄  Index ready: {retriever.index_size} chunks, BM25={retriever.bm25_ready}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RicohLibrary quality eval harness")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-judged metrics (objective metrics only, no API calls)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  RicohLibrary — Quality Evaluation Harness")
    print("=" * 70)
    _ensure_index()
    report = evaluate(use_judge=not args.no_judge)
    write_outputs(report)

    s = report["summary"]
    print(f"\n{'=' * 70}\n  SUMMARY")
    for k, v in s.items():
        print(f"  {k:28s}: {v}")
    print("=" * 70)
