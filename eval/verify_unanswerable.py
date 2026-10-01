"""Check the "refuse" labels in ground_truth.json.

Some questions are labelled expected_behavior "refuse" because the corpus doesn't have the answer. If
that's wrong, a refusal scored as correct is really a retrieval miss. The harness can't tell, since it
only sees the top results. The only way is to read the whole corpus. This scans every PDF for the terms
that would have to appear if the answer existed, and prints what it finds so you can check it yourself.

Usage:
    python -m eval.verify_unanswerable
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from src.config import DATA_DIR


# each claim pairs a question with a regex that must be absent from the corpus for it to be unanswerable.
# Keep the pattern broad, then read what it finds
CLAIMS: list[dict[str, object]] = [
    {
        "qid": 2,
        "question": "How much RAM does the primary server need for document-level processing?",
        "pattern": re.compile(r"\bRAM\b|\bgigabytes?\b|\bGB\b|memory requirement", re.I),
        "expectation": "No RAM/memory sizing figure anywhere in the corpus.",
    },
    {
        "qid": 3,
        "question": "How much hard drive space should I allocate for DB2 logs?",
        "pattern": re.compile(r"DB2", re.I),
        "expectation": "DB2 is discussed, but no log disk-space allocation figure.",
    },
]

# lines that match this as well as the claim pattern would be an actual answer (a quantity), so they're highlighted
QUANTITY = re.compile(r"\d+\s*(?:MB|GB|TB|megabytes?|gigabytes?)", re.I)


def _iter_documents(data_dir: Path) -> Iterator[tuple[str, str]]:
    """Every PDF in the corpus as (filename, full text)."""
    for path in sorted(data_dir.glob("*.pdf")):
        try:
            doc = fitz.open(path)
            text = "\n".join(page.get_text("text") for page in doc)
            doc.close()
        except Exception as exc:  # pragma: no cover - corrupt/locked file
            print(f"  could not read {path.name}: {exc}")
            continue
        yield path.name, text


def _matching_lines(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [
        " ".join(m.group(0).split())
        for m in re.finditer(rf"[^\n]*{pattern.pattern}[^\n]*", text, pattern.flags)
    ]


def verify(data_dir: Path = DATA_DIR) -> int:
    documents = list(_iter_documents(data_dir))
    print(f"Scanned {len(documents)} documents in {data_dir}\n")

    quantified_total = 0

    for claim in CLAIMS:
        pattern: re.Pattern[str] = claim["pattern"]  # type: ignore[assignment]
        print("=" * 72)
        print(f"Q{claim['qid']}: {claim['question']}")
        print(f"Expectation: {claim['expectation']}")
        print("=" * 72)

        hits = 0
        quantified: list[str] = []

        for name, text in documents:
            lines = _matching_lines(text, pattern)
            if not lines:
                continue
            hits += 1
            print(f"  [{name}] {len(lines)} matching line(s)")
            for line in lines[:4]:
                flag = "  <-- QUANTITY" if QUANTITY.search(line) else ""
                if flag:
                    quantified.append(f"{name}: {line}")
                print(f"      {line[:150]}{flag}")

        print(f"\n  Documents matching: {hits}")
        if quantified:
            quantified_total += len(quantified)
            print(f"  {len(quantified)} line(s) contain a quantity, REVIEW THESE:")
            for q in quantified:
                print(f"      {q[:180]}")
            print("  -> If any is a real answer, this question is NOT unanswerable")
            print("    and its 'refuse' label is wrong.")
        else:
            print("  No quantity found, 'refuse' label is supported.\n")

    print("=" * 72)
    if quantified_total:
        print(f"RESULT: {quantified_total} line(s) need human review.")
        return 1
    print("RESULT: every 'refuse' label in this file is supported by the corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(verify())
