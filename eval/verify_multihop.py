"""Check that eval/multihop_questions.json is actually multi-hop.

Two things to confirm before the set is worth running an ablation on:

1. Every expected source document exists in the corpus.
2. A single retrieval on the raw question does NOT already pull every required
   document into the top-k. If it does, the question is not testing multi-hop
   retrieval, it is just a question the retriever handles in one pass.

Prints, per question, how many of its required documents a single retrieval
finds, and flags the ones that are not multi-hop.

    python -m eval.verify_multihop
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from src.config import DATA_DIR, PROJECT_ROOT, RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K

QUESTIONS = PROJECT_ROOT / "eval" / "multihop_questions.json"


def _ranked_docs(question: str) -> list[str]:
    from src.retriever import get_retriever

    results = get_retriever().retrieve(
        query=question, top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K
    )
    ordered: list[str] = []
    for r in results:
        d = r.get("source_document")
        if d and d not in ordered:
            ordered.append(d)
    return ordered


def verify() -> int:
    questions = json.loads(io.open(QUESTIONS, encoding="utf-8").read())["questions"]
    present = {p.name for p in DATA_DIR.glob("*.pdf")}

    missing_any = False
    single_pass_covers_all = 0
    n_multi = 0

    for q in questions:
        expected = q["expected_sources"]
        gone = [s for s in expected if s not in present]
        if gone:
            missing_any = True
            print(f"Q{q['id']}: MISSING FROM CORPUS: {gone}")
            continue

        docs = _ranked_docs(q["question"])
        found = [s for s in expected if s in docs[:RETRIEVAL_FINAL_K]]
        is_multi = len(expected) >= 2
        n_multi += is_multi
        covers_all = len(found) == len(expected)
        if is_multi and covers_all:
            single_pass_covers_all += 1
        tag = "  <- single pass already covers all" if (is_multi and covers_all) else ""
        print(
            f"Q{q['id']:>2}: {len(found)}/{len(expected)} required docs in top-{RETRIEVAL_FINAL_K}"
            f"  {found}{tag}"
        )

    print()
    print(f"{n_multi} multi-source questions; "
          f"{single_pass_covers_all} are fully covered by one retrieval "
          f"({n_multi - single_pass_covers_all} genuinely exercise multi-hop retrieval).")
    if missing_any:
        print("FAIL: some expected sources are not in the corpus.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(verify())
