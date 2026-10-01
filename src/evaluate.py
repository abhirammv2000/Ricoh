"""Quick latency and citation check over the 10 seed questions.

Runs each one through the agent and writes a CSV and a markdown summary (not checked in). It only
measures how long an answer took and whether it has a citation, not whether it's right (that's
eval_harness.py). It's cheap and catches gross breakage without spending judge tokens.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# config has to be imported first
from src.config import PROJECT_ROOT

# quiet the ingest and retriever logs, only the agent's output is useful here
for _quiet in ("src.ingest", "src.retriever"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

from src.agent import run_agent  # noqa: E402 - after logging config
from src.ingest import ingest_all
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# the 10 seed questions

SEED_QUESTIONS: list[str] = [
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

CSV_PATH: Path = PROJECT_ROOT / "evaluation_results.csv"
REPORT_PATH: Path = PROJECT_ROOT / "evaluation_report.md"


# running the questions

def _extract_sources(answer: str) -> list[str]:
    """The unique [Document, Page X] citations in the answer."""
    import re

    pattern = r"\[([^\]]+?),\s*Page\s*\d+\]"
    matches = re.findall(pattern, answer, re.IGNORECASE)
    return sorted(set(matches)) if matches else ["(no citations found)"]


def run_evaluation() -> list[dict[str, Any]]:
    """Run every seed question and return one result dict each."""
    results: list[dict[str, Any]] = []

    total = len(SEED_QUESTIONS)
    for idx, question in enumerate(SEED_QUESTIONS, 1):
        print(f"\n{'=' * 70}")
        print(f"  Question {idx}/{total}")
        print(f"  {question}")
        print(f"{'=' * 70}")

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

        print(f"\nAnswered in {elapsed:.1f}s")
        print(f"Sources: {', '.join(sources)}")

    return results


# csv

def save_csv(results: list[dict[str, Any]], path: Path = CSV_PATH) -> None:
    """Write the results to a csv."""
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

    print(f"\nCSV saved -> {path}")


# markdown report

def save_markdown_report(
    results: list[dict[str, Any]], path: Path = REPORT_PATH
) -> None:
    """Write the results as a markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = sum(r["latency_seconds"] for r in results)
    avg_time = total_time / len(results) if results else 0

    lines: list[str] = [
        "# Citera evaluation report",
        "",
        f"**Generated:** {timestamp}  ",
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
        q_short = r["question"][:60] + ("..." if len(r["question"]) > 60 else "")
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

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Markdown report saved -> {path}")



if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("  Citera latency and citation smoke test")
    print("=" * 70)

    print("\nChecking retrieval index...")
    retriever = HybridRetriever()

    if retriever.index_size == 0 or not retriever.bm25_ready:
        reason = "empty" if retriever.index_size == 0 else "BM25 missing"
        print(f"   Index needs (re)build ({reason}) - ingesting PDFs...")
        # show the ingest logs while it builds
        logging.getLogger("src.ingest").setLevel(logging.INFO)
        chunks = ingest_all()
        if not chunks:
            print("No PDFs found in data/. Add PDFs and retry.")
            sys.exit(1)
        retriever.build_index(chunks)
        logging.getLogger("src.ingest").setLevel(logging.WARNING)
        print(f"   Index built: {retriever.index_size} docs.")
    else:
        print(f"   Index ready: {retriever.index_size} docs, BM25: ")

    print(f"\nRunning {len(SEED_QUESTIONS)} seed questions...")
    results = run_evaluation()

    save_csv(results)
    save_markdown_report(results)

    total = sum(r["latency_seconds"] for r in results)
    print(f"\n{'=' * 70}")
    print(f"  EVALUATION COMPLETE")
    print(f"  {len(results)} questions | {total:.1f}s total | {total/len(results):.1f}s avg")
    print(f"  CSV:      {CSV_PATH}")
    print(f"  Report:   {REPORT_PATH}")
    print(f"{'=' * 70}")
