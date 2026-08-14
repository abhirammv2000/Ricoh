"""
src/eval_harness.py - Quality-focused evaluation harness for Citera.

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
  - eval/metrics.json   - machine-readable aggregate + per-question scores
  - eval/eval_report.md - human-readable report

Usage:
    python -m src.eval_harness            # run the full harness
    python -m src.eval_harness --no-judge # objective metrics only (no API)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure config.py runs first (logging + telemetry silencing)
from src.config import (
    DEFAULT_LLM_PROVIDER,
    USE_PLANNER,
    USE_VERIFIER,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    PROJECT_ROOT,
    RETRIEVAL_FINAL_K,
    RETRIEVAL_TOP_K,
)

# Quiet the noisy ingest/retriever logs during evaluation
for _quiet in ("src.ingest", "src.retriever"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

from src.agent import get_agent_graph, initial_state  # noqa: E402
from src.ingest import ingest_all  # noqa: E402
from src.instrumentation import PRICING_SNAPSHOT_DATE  # noqa: E402
from src.instrumentation import invoke as instrumented_invoke  # noqa: E402
from src.instrumentation import record_run  # noqa: E402
from src.llm_factory import _DEFAULT_MODELS, get_llm  # noqa: E402
from src.retriever import HybridRetriever, get_retriever, reset_retriever  # noqa: E402

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
    """Run the LLM judge for one question. Resilient to bad JSON.

    The judge runs on ``JUDGE_MODEL``, deliberately a *different and
    stronger* model than the agent under test.  Letting a model grade its
    own output produces self-preference bias and makes the resulting
    groundedness/correctness scores uninterpretable.
    """
    llm = get_llm(model=JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS)
    prompt = JUDGE_PROMPT.format(
        question=question,
        evidence_block=evidence_block or "(no evidence)",
        answer=answer,
        key_facts=json.dumps(key_facts, ensure_ascii=False) or "[]",
        expected_behavior=expected_behavior,
    )
    raw = instrumented_invoke(llm, prompt, stage="judge")
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


def _evidence_recall(
    expected: list[str], evidence_docs: set[str], any_hit: bool = False
) -> float | None:
    """Fraction of expected source docs present in the FULL accumulated evidence.

    ⚠️  This is deliberately **not** called "recall@k".  ``evidence_docs`` is
    the union of every chunk the agent accumulated, across all planner
    sub-queries, the entity-boost pass, and any retry, which in practice
    ranges from 5 to 40+ chunks.  Reporting that as "recall@k" (where
    ``RETRIEVAL_FINAL_K`` is 5) overstates the retriever, because the net
    cast is far wider than k.

    Read this as: *did the right document reach the synthesizer at all?*
    For retriever quality in isolation, see ``_retriever_only_recall``.

    Returns None when no expected sources are defined (can't be scored).
    """
    if not expected:
        return None
    # Two different meanings share this field, so the caller must say which:
    #   any_hit=False (curated set)  - several documents are JOINTLY relevant;
    #                                  recall is the fraction retrieved.
    #   any_hit=True  (generated set) - the documents are ALTERNATIVES, each
    #                                  independently able to answer; retrieving
    #                                  any one of them is fully correct.
    # Applying the wrong one silently halves the score on correct retrievals.
    if any_hit:
        return 1.0 if set(expected) & evidence_docs else 0.0
    hits = sum(1 for d in expected if d in evidence_docs)
    return hits / len(expected)


# Depths for the retriever-only diagnostic.
#
# ⚠️  These MUST mirror production (`RETRIEVAL_TOP_K`), and an earlier version
# of this file got that wrong with real consequences.  It used a deliberately
# wider pool (top_k=50) so that recall@20 was "measurable", but RRF is not
# monotonic in pool size, so that measured a configuration the agent never
# runs and understated true recall@5 (0.81 measured vs 1.00 actual).  A
# diagnostic that does not mirror production is worse than no diagnostic: it
# produces confident, wrong conclusions about where the bottleneck is.
#
# Depths stop at RETRIEVAL_FINAL_K because that is all the synthesizer ever
# sees; deeper numbers would describe a system that does not exist.
RETRIEVER_DIAG_TOP_K: int = RETRIEVAL_TOP_K
RETRIEVER_DIAG_DEPTHS: tuple[int, ...] = (1, 3, RETRIEVAL_FINAL_K)


def _retriever_only_recall(
    question: str, expected: list[str], any_hit: bool = False
) -> dict[str, float]:
    """Recall@N of the retriever alone, on the RAW question.

    This bypasses the planner entirely: one retrieval call, original
    question, no sub-query decomposition, no entity boosting, no retry,
    run at **production settings** so it is a like-for-like control.

    Why it matters: ``_evidence_recall`` conflates two very different
    failure modes, the retriever cannot find the document, versus the
    retriever ranks it fine but the planner rewrote the question into
    something worse.  This is the control that separates them, and it
    doubles as the **do-nothing baseline**: if the full agentic pipeline
    does not beat a single retrieval on the raw question, the planner,
    verifier, and retry loop are not paying for themselves.

    Ranks are counted over distinct **documents**, not chunks, because on
    this corpus (mostly one-page help articles) the task is selecting the
    right document out of 733.
    """
    if not expected:
        return {}

    results = get_retriever().retrieve(
        query=question,
        top_k=RETRIEVER_DIAG_TOP_K,
        final_k=max(RETRIEVER_DIAG_DEPTHS),
    )

    ranked_docs: list[str] = []
    for r in results:
        doc = r.get("source_document")
        if doc and doc not in ranked_docs:
            ranked_docs.append(doc)

    out: dict[str, float] = {}
    for depth in RETRIEVER_DIAG_DEPTHS:
        top = set(ranked_docs[:depth])
        if any_hit:
            out[f"recall@{depth}"] = 1.0 if set(expected) & top else 0.0
        else:
            out[f"recall@{depth}"] = sum(1 for d in expected if d in top) / len(expected)
    return out


def _citation_precision(cited: set[str], evidence_docs: set[str]) -> float | None:
    """Fraction of cited docs that are actually in the evidence.

    Returns None when the answer cites nothing (e.g. a refusal).
    """
    if not cited:
        return None
    valid = sum(1 for d in cited if d in evidence_docs)
    return valid / len(cited)


def _is_refusal(answer: str) -> bool:
    """Detect the canonical refusal, robustly.

    The synthesizer answers in the *user's* language, so a refusal that was
    only ever emitted in translated form would be invisible here and would
    be silently scored as an answer, inflating behaviour-match and
    correctness on exactly the questions the system handled correctly.

    The synthesizer prompt therefore requires the English canonical
    sentence verbatim in every refusal, with any translation appended
    after it.  Whitespace is normalised so a line break inside the phrase
    does not defeat the match.
    """
    return REFUSAL_MARKER in " ".join(answer.lower().split())


# ====================================================================
# 3. AGENT RUNNER (returns full state, like the Streamlit app)
# ====================================================================

def _run_agent_full(
    query: str, use_planner: bool = True, use_verifier: bool = True
) -> dict[str, Any]:
    agent = get_agent_graph(use_planner=use_planner, use_verifier=use_verifier)
    return dict(agent.invoke(initial_state(query, use_planner=use_planner)))


def _format_evidence_block(evidence: list[dict[str, Any]]) -> str:
    """Render the FULL evidence the synthesizer saw, so the groundedness
    judge scores against the same material (no truncation / capping,
    an earlier capped version understated groundedness)."""
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for i, e in enumerate(evidence, 1):
        src = e.get("source_document", "unknown")
        page = e.get("page_number", "?")
        text = (e.get("text", "") or "").strip()
        lines.append(f"[{i}] {src}, Page {page}\n{text}")
    return "\n\n".join(lines)


# ====================================================================
# 4. MAIN EVALUATION LOOP
# ====================================================================

def evaluate(
    use_judge: bool = True,
    use_planner: bool | None = None,
    use_verifier: bool | None = None,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    split: str | None = None,
) -> dict[str, Any]:
    # Default to the PRODUCTION composition from config, not to True.
    # Hard-coding True here meant the harness silently benchmarked the full
    # agentic pipeline while production shipped the single-retrieval path,
    # i.e. the headline numbers would have described a system nobody runs.
    # Ablation callers still pass these explicitly.
    use_planner = USE_PLANNER if use_planner is None else use_planner
    use_verifier = USE_VERIFIER if use_verifier is None else use_verifier
    gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    questions = gt["questions"]
    if split:
        # Held-out slices exist so a final number can be reported that no
        # decision was fitted to. Filtering here keeps that discipline
        # enforceable from the command line rather than by convention.
        questions = [q for q in questions if q.get("split") == split]
    print(f"Questions: {len(questions)} from {ground_truth_path.name}"
          + (f" (split={split})" if split else ""))

    per_question: list[dict[str, Any]] = []

    for item in questions:
        qid = item["id"]
        question = item["question"]
        expected_behavior = item.get("expected_behavior", "answer")
        key_facts = item.get("key_facts", [])
        expected_sources = item.get("expected_sources", [])
        # Generated questions carry alternative sources; curated ones carry
        # jointly-relevant sources. See _evidence_recall.
        alternatives = item.get("provenance") == "generated"

        print(f"\n{'━' * 70}\n  Q{qid}: {question}\n{'━' * 70}")

        # The agent run is recorded separately from the judge: the agent is
        # what a user pays for, the judge is evaluation overhead. Reporting a
        # single blended figure would overstate the cost of serving a query.
        t0 = time.perf_counter()
        with record_run(query=question) as agent_run:
            try:
                state = _run_agent_full(
                    question, use_planner=use_planner, use_verifier=use_verifier
                )
                answer = state.get("final_answer", "")
                evidence = state.get("retrieved_evidence", [])
                iterations = state.get("iterations", 0)
                sub_queries = state.get("sub_queries", [])
                entities = state.get("entities", [])
            except Exception as exc:  # pragma: no cover - depends on live API
                logger.error("Agent failed on Q%d: %s", qid, exc)
                answer, evidence, iterations = f"ERROR: {exc}", [], 0
                sub_queries, entities = [], []
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
            "evidence_docs": sorted(ev_docs),
            "expected_sources": expected_sources,
            # Planner trace. Without these, a retrieval miss is unattributable:
            # you cannot tell whether the retriever failed or the planner
            # rewrote the question into something worse. Recording them is
            # what made the planner regression visible at all.
            "sub_queries": sub_queries,
            "entities": entities,
            # End-to-end: did the expected doc reach the synthesizer at all,
            # across every sub-query / entity pass / retry?
            "evidence_recall": _evidence_recall(
                expected_sources, ev_docs, any_hit=alternatives
            ),
            # Retriever in isolation: raw question, single pass, no planner.
            "retriever_recall": _retriever_only_recall(
                question, expected_sources, any_hit=alternatives
            ),
            "citation_precision": _citation_precision(cited, ev_docs),
            "needs_verification": item.get("needs_verification", False),
            "answer": answer,
            # Product cost: what serving this one query actually costs.
            "cost_usd": agent_run.total_cost_usd,
            "llm_calls": agent_run.llm_calls,
            "llm_seconds": agent_run.total_llm_seconds,
            "input_tokens": agent_run.total_input_tokens,
            "output_tokens": agent_run.total_output_tokens,
            "by_stage": agent_run.by_stage(),
        }

        if use_judge and not answer.startswith("ERROR"):
            # Judge spend is eval overhead; keep it out of the request trace log.
            with record_run(persist=False) as judge_run:
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
                # Eval overhead, tracked separately and never folded into cost_usd.
                judge_cost_usd=judge_run.total_cost_usd,
            )

        per_question.append(row)
        _print_row(row)

    return _aggregate(
        per_question, use_judge, use_planner=use_planner, use_verifier=use_verifier
    )


def _print_row(row: dict[str, Any]) -> None:
    er = row["evidence_recall"]
    cp = row["citation_precision"]
    r5 = row.get("retriever_recall", {}).get("recall@5")
    print(
        f"  behaviour_match={row['behavior_match']} | "
        f"evidence_recall={'n/a' if er is None else f'{er:.2f}'} | "
        f"retriever@5={'n/a' if r5 is None else f'{r5:.2f}'} | "
        f"cite_prec={'n/a' if cp is None else f'{cp:.2f}'} | "
        f"grounded={row.get('groundedness', 'n/a')} | "
        f"correct={row.get('correctness', 'n/a')} | {row['latency_seconds']}s"
    )


def _avg(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 3) if nums else None


def _bootstrap_ci(
    values: list[float | None],
    iterations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[float] | None:
    """Percentile bootstrap 95% CI for the mean.

    At n=10 the sampling error on any of these means is enormous, a single
    question moves a mean by 10 points.  Reporting a bare "0.98
    groundedness" invites the reader to believe a precision the sample size
    cannot support, so every mean ships with the interval around it.

    If the CI spans most of [0, 1], that is the honest finding: this eval
    set is too small to distinguish the system from a materially worse one.
    """
    nums = [v for v in values if v is not None]
    if len(nums) < 2:
        return None

    rng = random.Random(seed)  # fixed seed → reproducible intervals
    n = len(nums)
    means = []
    for _ in range(iterations):
        resample = [nums[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iterations)]
    hi = means[min(int((1 - alpha / 2) * iterations), iterations - 1)]
    return [round(lo, 3), round(hi, 3)]


def _cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cost and per-stage rollup across the run.

    Two figures are kept apart on purpose:
      * ``mean_cost_per_query_usd``: what serving one query costs. This is
        the number that scales with traffic and belongs in any unit-economics
        discussion.
      * ``eval_judge_cost_usd``: what grading the benchmark costs. It is
        overhead paid once per eval run, not per user query.
    """
    n = len(rows)
    stages: dict[str, dict[str, float]] = {}
    for r in rows:
        for stage, agg in (r.get("by_stage") or {}).items():
            acc = stages.setdefault(
                stage, {"calls": 0, "seconds": 0.0, "cost_usd": 0.0}
            )
            acc["calls"] += agg["calls"]
            acc["seconds"] += agg["seconds"]
            acc["cost_usd"] += agg["cost_usd"]

    # Share of total LLM wall-clock, which is what identifies the stage worth
    # optimising. Cost share and time share are usually NOT the same stage.
    total_secs = sum(s["seconds"] for s in stages.values()) or 1.0
    total_cost = sum(s["cost_usd"] for s in stages.values()) or 1.0
    for s in stages.values():
        s["seconds"] = round(s["seconds"], 2)
        s["cost_usd"] = round(s["cost_usd"], 6)
        s["pct_of_llm_seconds"] = round(100 * s["seconds"] / total_secs, 1)
        s["pct_of_cost"] = round(100 * s["cost_usd"] / total_cost, 1)

    costs = [r.get("cost_usd", 0.0) for r in rows]
    return {
        "pricing_snapshot": PRICING_SNAPSHOT_DATE,
        "mean_cost_per_query_usd": round(sum(costs) / n, 6),
        "max_cost_per_query_usd": round(max(costs), 6) if costs else 0.0,
        "total_agent_cost_usd": round(sum(costs), 6),
        "eval_judge_cost_usd": round(
            sum(r.get("judge_cost_usd", 0.0) for r in rows), 6
        ),
        "mean_llm_calls_per_query": round(
            sum(r.get("llm_calls", 0) for r in rows) / n, 2
        ),
        "mean_input_tokens": round(sum(r.get("input_tokens", 0) for r in rows) / n),
        "mean_output_tokens": round(sum(r.get("output_tokens", 0) for r in rows) / n),
        "by_stage": stages,
    }


def _worst(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """The single worst question for a metric, means hide these."""
    scored = [r for r in rows if r.get(key) is not None]
    if not scored:
        return None
    w = min(scored, key=lambda r: r[key])
    return {"id": w["id"], "question": w["question"], key: w[key]}


def _aggregate(
    rows: list[dict[str, Any]],
    use_judge: bool,
    use_planner: bool = True,
    use_verifier: bool = True,
) -> dict[str, Any]:
    n = len(rows)

    def stat(key: str) -> dict[str, Any]:
        vals = [r.get(key) for r in rows]
        return {
            "mean": _avg(vals),
            "ci95": _bootstrap_ci(vals),
            "scored_n": sum(1 for v in vals if v is not None),
        }

    retriever_at = {
        f"recall@{d}": _avg(
            [r.get("retriever_recall", {}).get(f"recall@{d}") for r in rows]
        )
        for d in RETRIEVER_DIAG_DEPTHS
    }

    summary = {
        "questions": n,
        "behavior_match_rate": round(sum(r["behavior_match"] for r in rows) / n, 3),
        "evidence_recall": stat("evidence_recall"),
        "retriever_only_recall": retriever_at,
        "citation_precision": stat("citation_precision"),
        "mean_latency_seconds": round(sum(r["latency_seconds"] for r in rows) / n, 2),
        "max_latency_seconds": max(r["latency_seconds"] for r in rows),
        "cost": _cost_summary(rows),
    }
    if use_judge:
        summary["groundedness"] = stat("groundedness")
        summary["correctness"] = stat("correctness")

    summary["worst_case"] = {
        k: _worst(rows, k)
        for k in ("evidence_recall", "groundedness", "correctness")
        if _worst(rows, k) is not None
    }

    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "judge_enabled": use_judge,
        # Recorded so a result is never ambiguous about who graded it.
        "judge_model": JUDGE_MODEL if use_judge else None,
        "agent_model": _DEFAULT_MODELS.get(DEFAULT_LLM_PROVIDER),
        "retrieval_final_k": RETRIEVAL_FINAL_K,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        # Pipeline configuration under test. Recorded so an ablation result
        # can never be mistaken for a production-config result.
        "config": {"use_planner": use_planner, "use_verifier": use_verifier},
        "summary": summary,
        "per_question": rows,
    }


# ====================================================================
# 5. REPORT WRITERS
# ====================================================================

def write_outputs(
    report: dict[str, Any],
    metrics_path: Path = METRICS_PATH,
    report_path: Path = REPORT_PATH,
) -> None:
    """Write the metrics JSON and the human-readable report.

    The paths are parameters rather than hard-coded constants so that
    ablation runs and smoke tests write somewhere harmless.  Hard-coding
    them once cost me the real artifacts when a synthetic test overwrote
    the production files.
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📊  Metrics  → {metrics_path}")

    s = report["summary"]

    def row(label: str, stat: dict[str, Any] | None, meaning: str) -> str:
        if not stat or stat.get("mean") is None:
            return f"| {label} | n/a | — | {meaning} |"
        ci = stat.get("ci95")
        ci_txt = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—"
        return (
            f"| {label} | {stat['mean']:.2f} | {ci_txt} "
            f"| {meaning} (n={stat['scored_n']}) |"
        )

    lines: list[str] = [
        "# 📊 Citera: Quality Evaluation Report",
        "",
        f"**Generated:** {report['generated']}  ",
        f"**Agent model:** `{report.get('agent_model')}`  ",
        f"**Judge model:** `{report.get('judge_model') or 'disabled'}`  ",
        f"**Questions:** {s['questions']}  ",
        "",
        "> Every mean carries a 95% percentile-bootstrap confidence interval.",
        "> At this sample size the intervals are wide by construction, they",
        "> are reported so the numbers are not read as more precise than the",
        "> eval set can support.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Mean | 95% CI | What it means |",
        "|---|---|---|---|",
        f"| Behaviour-match rate | {s['behavior_match_rate']:.2f} | — "
        f"| Answered vs. refused as expected |",
        row(
            "Evidence recall",
            s.get("evidence_recall"),
            "Expected doc reached the synthesizer, across ALL passes/retries",
        ),
        row(
            "Citation precision",
            s.get("citation_precision"),
            "Cited docs exist in the evidence (catches fabricated filenames only)",
        ),
    ]
    if report["judge_enabled"]:
        lines += [
            row("Groundedness", s.get("groundedness"), "Claims supported by evidence"),
            row("Correctness", s.get("correctness"), "Conveys expected facts / refuses correctly"),
        ]
    lines += [
        f"| Mean latency | {s['mean_latency_seconds']}s | — "
        f"| Per-question wall-clock (max {s['max_latency_seconds']}s) |",
        "",
        "### Retriever in isolation",
        "",
        "Single retrieval on the raw question, no planner, no sub-queries,",
        "no entity boost, no retry. Ranks counted over distinct documents.",
        "Comparing this against *Evidence recall* separates a retrieval",
        "failure from a planning failure.",
        "",
        "| Depth | Recall |",
        "|---|---|",
    ]
    for depth in RETRIEVER_DIAG_DEPTHS:
        lines.append(
            f"| recall@{depth} | {_fmt(s['retriever_only_recall'].get(f'recall@{depth}'))} |"
        )

    cost = s.get("cost", {})
    if cost:
        lines += [
            "",
            "### Cost and where the time goes",
            "",
            f"Prices are a dated snapshot ({cost['pricing_snapshot']}); token counts are",
            "the ground truth and cost is derived from them. Agent cost is what serving",
            "a query costs; judge cost is eval overhead and is never folded into it.",
            "",
            f"- **Mean cost per query: ${cost['mean_cost_per_query_usd']:.5f}** "
            f"(max ${cost['max_cost_per_query_usd']:.5f})",
            f"- Mean {cost['mean_llm_calls_per_query']} LLM calls, "
            f"{cost['mean_input_tokens']:,} in / {cost['mean_output_tokens']:,} out tokens",
            f"- Whole-benchmark agent cost ${cost['total_agent_cost_usd']:.4f}; "
            f"judge overhead ${cost['eval_judge_cost_usd']:.4f}",
            "",
            "| Stage | Calls | LLM seconds | % of time | Cost | % of cost |",
            "|---|---|---|---|---|---|",
        ]
        for stage, agg in sorted(
            cost["by_stage"].items(), key=lambda kv: -kv[1]["seconds"]
        ):
            lines.append(
                f"| {stage} | {agg['calls']} | {agg['seconds']}s "
                f"| {agg['pct_of_llm_seconds']}% | ${agg['cost_usd']:.5f} "
                f"| {agg['pct_of_cost']}% |"
            )

    worst = s.get("worst_case", {})
    if worst:
        lines += ["", "### Worst case (what the means hide)", ""]
        for metric, w in worst.items():
            lines.append(f"- **{metric}** = {w[metric]:.2f}, Q{w['id']}: {w['question']}")

    lines += [
        "",
        "## Per-question results",
        "",
        "| # | Behaviour ✓ | Evid. recall | Retr.@5 | Cite prec. | Grounded | Correct | Latency | Flags |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["per_question"]:
        flags = []
        if r.get("needs_verification"):
            flags.append("⚠ verify")
        if not r["behavior_match"]:
            flags.append("✗ behaviour")
        if r.get("evidence_recall") == 0.0:
            flags.append("✗ retrieval miss")
        lines.append(
            f"| {r['id']} | {'✅' if r['behavior_match'] else '❌'} "
            f"| {_fmt(r['evidence_recall'])} "
            f"| {_fmt(r.get('retriever_recall', {}).get('recall@5'))} "
            f"| {_fmt(r['citation_precision'])} "
            f"| {_fmt(r.get('groundedness'))} | {_fmt(r.get('correctness'))} "
            f"| {r['latency_seconds']}s | {', '.join(flags) or '—'} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📝  Report   → {report_path}")


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


# ====================================================================
# __main__
# ====================================================================

def _ensure_index() -> None:
    retriever = HybridRetriever()
    if retriever.index_size == 0 or not retriever.bm25_ready:
        print("📄  Index missing, ingesting PDFs from data/ …")
        chunks = ingest_all()
        if not chunks:
            raise SystemExit("❌  No PDFs found in data/. Add PDFs and retry.")
        retriever.build_index(chunks)
        reset_retriever()  # drop any cached instance so the agent reloads the fresh index
    print(f"📄  Index ready: {retriever.index_size} chunks, BM25={retriever.bm25_ready}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Citera quality eval harness")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-judged metrics (objective metrics only, no API calls)",
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=GROUND_TRUTH_PATH,
        help="Question set to evaluate (default: the curated 10-question set)",
    )
    parser.add_argument(
        "--split", default=None,
        help="Only evaluate questions with this split value, e.g. dev / holdout",
    )
    parser.add_argument(
        "--tag", default="", help="Suffix for output filenames, to avoid clobbering",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Citera: Quality Evaluation Harness")
    print("=" * 70)
    _ensure_index()
    report = evaluate(
        use_judge=not args.no_judge,
        ground_truth_path=args.ground_truth,
        split=args.split,
    )
    suffix = f"_{args.tag}" if args.tag else ""
    write_outputs(
        report,
        metrics_path=METRICS_PATH.with_name(f"metrics{suffix}.json"),
        report_path=REPORT_PATH.with_name(f"eval_report{suffix}.md"),
    )

    s = report["summary"]
    print(f"\n{'=' * 70}\n  SUMMARY")
    for k, v in s.items():
        print(f"  {k:28s}: {v}")
    print("=" * 70)
