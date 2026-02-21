"""
src/evaluate.py — Phase 4 Evaluation Pipeline.

Runs the 10 official DaSSA / HackVerse hackathon test questions
through the full agentic pipeline and produces:

1. **evaluation_results.csv** — machine-readable results.
2. **evaluation_report.md**  — judge-ready Markdown report.

Each question is sent to ``agent.run_agent()`` and we capture:
• The final answer (with citations)
• Latency (seconds)
• Retrieved evidence sources (document + page)

Design decisions
────────────────
• Questions are hardcoded — the hackathon forbids web search and
  the rubric specifies these exact queries.
• CSV uses pandas for clean formatting; Markdown is hand-assembled
  for maximum judge readability.
• We suppress ingestion logs during evaluation so only the agent
  "thoughts" (Planner / Verifier / Synthesizer) are visible.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure config.py runs first (logging + telemetry silencing)
from src.config import PROJECT_ROOT

# Suppress ingestion + retriever detail logs during eval
# (we only want agent-level "thoughts" in the terminal)
for _quiet in ("src.ingest", "src.retriever"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

from src.agent import run_agent  # noqa: E402 — after logging config
from src.ingest import ingest_all
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ====================================================================
# 1. OFFICIAL HACKATHON TEST QUESTIONS
# ====================================================================

HACKATHON_QUESTIONS: list[str] = [
    "What property do I set if I want the printers to enable after a restart?",
    "How much RAM does the primary server need if I will be doing document-level processing?",
    "How much hard drive space should I allocate for DB2 logs?",
    "Does RPD work with FusionPro?",
    "What operating system does RPD run on?",
    "How do I create a workflow?",
    "What programs does RPD integrate with?",
    "What is the command to shut down RPD?",
    "How do I use locations?",
    "What inserters does RPD support?",
]

# ── Output paths ──
CSV_PATH: Path = PROJECT_ROOT / "evaluation_results.csv"
REPORT_PATH: Path = PROJECT_ROOT / "evaluation_report.md"


# ====================================================================
# 2. EVALUATION RUNNER
# ====================================================================

def _extract_sources(answer: str) -> list[str]:
    """Pull unique [Document, Page X] citations from the answer text."""
    import re

    # Match citations like [filename.pdf, Page 3]
    pattern = r"\[([^\]]+?),\s*Page\s*\d+\]"
    matches = re.findall(pattern, answer, re.IGNORECASE)
    return sorted(set(matches)) if matches else ["(no citations found)"]


def run_evaluation() -> list[dict[str, Any]]:
    """Execute all hackathon questions and collect results.

    Returns:
        List of result dicts, one per question.
    """
    results: list[dict[str, Any]] = []

    total = len(HACKATHON_QUESTIONS)
    for idx, question in enumerate(HACKATHON_QUESTIONS, 1):
        print(f"\n{'━' * 70}")
        print(f"  Question {idx}/{total}")
        print(f"  {question}")
        print(f"{'━' * 70}")

        t0 = time.perf_counter()
        try:
            answer = run_agent(question)
        except Exception as e:
            logger.error("Agent failed on Q%d: %s", idx, e)
            answer = f"ERROR: {e}"
        elapsed = time.perf_counter() - t0

        sources = _extract_sources(answer)

        results.append(
            {
                "question_number": idx,
                "question": question,
                "answer": answer,
                "latency_seconds": round(elapsed, 2),
                "sources": "; ".join(sources),
            }
        )

        print(f"\n⏱️  Answered in {elapsed:.1f}s")
        print(f"📄  Sources: {', '.join(sources)}")

    return results


# ====================================================================
# 3. CSV WRITER
# ====================================================================

def save_csv(results: list[dict[str, Any]], path: Path = CSV_PATH) -> None:
    """Write evaluation results to a CSV file."""
    fieldnames = [
        "question_number",
        "question",
        "answer",
        "latency_seconds",
        "sources",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📊  CSV saved → {path}")


# ====================================================================
# 4. MARKDOWN REPORT WRITER
# ====================================================================

def save_markdown_report(
    results: list[dict[str, Any]], path: Path = REPORT_PATH
) -> None:
    """Generate a judge-ready Markdown evaluation report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = sum(r["latency_seconds"] for r in results)
    avg_time = total_time / len(results) if results else 0

    lines: list[str] = [
        "# 📊 RicohLibrary — Evaluation Report",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Team:** Neural Ninjas  ",
        f"**Track:** Ricoh Modern AI Solutions  ",
        f"**Total Questions:** {len(results)}  ",
        f"**Total Time:** {total_time:.1f}s  ",
        f"**Average Latency:** {avg_time:.1f}s per question  ",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| # | Question | Latency | Sources |",
        "|---|---|---|---|",
    ]

    for r in results:
        q_short = r["question"][:60] + ("…" if len(r["question"]) > 60 else "")
        lines.append(
            f"| {r['question_number']} "
            f"| {q_short} "
            f"| {r['latency_seconds']}s "
            f"| {r['sources'][:50]} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Detailed Answers",
            "",
        ]
    )

    for r in results:
        lines.extend(
            [
                f"### Q{r['question_number']}: {r['question']}",
                "",
                f"**Latency:** {r['latency_seconds']}s  ",
                f"**Sources:** {r['sources']}",
                "",
                r["answer"],
                "",
                "---",
                "",
            ]
        )

    # ── Compliance statement (required by hackathon) ──
    lines.extend(
        [
            "## Compliance Statement",
            "",
            "> We confirm that this project was developed during HackVerse 2026. "
            "We used only permitted datasets and tools. "
            "No private code sharing occurred between teams. "
            "All work is original.",
        ]
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📝  Markdown report saved → {path}")


# ====================================================================
# __main__ — Run the full evaluation
# ====================================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  RicohLibrary — Phase 4 Evaluation Pipeline")
    print("=" * 70)

    # ── Ensure index is ready ──
    print("\n📄  Checking retrieval index…")
    retriever = HybridRetriever()

    if retriever.index_size == 0 or not retriever.bm25_ready:
        reason = "empty" if retriever.index_size == 0 else "BM25 missing"
        print(f"   Index needs (re)build ({reason}) — ingesting PDFs…")
        # Temporarily restore ingest logging for visibility
        logging.getLogger("src.ingest").setLevel(logging.INFO)
        chunks = ingest_all()
        if not chunks:
            print("❌  No PDFs found in data/. Add PDFs and retry.")
            sys.exit(1)
        retriever.build_index(chunks)
        logging.getLogger("src.ingest").setLevel(logging.WARNING)
        print(f"   Index built: {retriever.index_size} docs.")
    else:
        print(f"   Index ready: {retriever.index_size} docs, BM25: ✅")

    # ── Run evaluation ──
    print(f"\n🚀  Running {len(HACKATHON_QUESTIONS)} hackathon questions…")
    results = run_evaluation()

    # ── Save outputs ──
    save_csv(results)
    save_markdown_report(results)

    # ── Final summary ──
    total = sum(r["latency_seconds"] for r in results)
    print(f"\n{'=' * 70}")
    print(f"  EVALUATION COMPLETE")
    print(f"  {len(results)} questions | {total:.1f}s total | {total/len(results):.1f}s avg")
    print(f"  CSV:      {CSV_PATH}")
    print(f"  Report:   {REPORT_PATH}")
    print(f"{'=' * 70}")
