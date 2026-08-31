"""Check whether a retrieval signal can tell a hit from a miss.

An earlier router escalated on a confidence signal off the single retrieval.
This records that signal per question against whether the retrieval actually
found an expected doc, and looks for a cutoff between the two groups. There
isn't one on this corpus (the ranges overlap), which is why src/router.py
escalates on the refusal marker instead. Kept as the record of that.

Retrieval only, so free and exact.

    python -m eval.calibrate_router [--split dev|holdout]
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from src.config import PROJECT_ROOT, RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K
from src.retriever import get_retriever
from src.router import confidence_signals

QUESTIONS_PATH: Path = PROJECT_ROOT / "eval" / "generated_questions.json"
OUT_PATH: Path = PROJECT_ROOT / "eval" / "router_calibration.json"


def _expected_hit(ranked_docs: list[str], expected: list[str]) -> bool:
    return bool(set(expected) & set(ranked_docs[:5]))


def calibrate(split: str) -> int:
    questions = [
        q for q in json.load(io.open(QUESTIONS_PATH, encoding="utf-8"))["questions"]
        if q.get("split") == split
    ]
    if not questions:
        raise SystemExit(f"no questions with split={split}")

    retriever = get_retriever()
    rows = []
    for q in questions:
        results = retriever.retrieve(
            query=q["question"], top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K
        )
        ranked: list[str] = []
        for r in results:
            d = r.get("source_document")
            if d and d not in ranked:
                ranked.append(d)
        sig = confidence_signals(results)
        rows.append({
            "id": q["id"],
            "hit": _expected_hit(ranked, q["expected_sources"]),
            **sig,
        })

    hits = [r for r in rows if r["hit"]]
    misses = [r for r in rows if not r["hit"]]
    print(f"split={split}: {len(rows)} questions, {len(misses)} retrieval misses\n")

    if not misses:
        print("no retrieval misses on this split, so there is nothing to calibrate against here.")
        print("run --split dev, where the misses are.")
        return 0

    def _describe(name: str, lower_is_worse: bool) -> None:
        h = sorted(r[name] for r in hits)
        m = sorted(r[name] for r in misses)
        print(f"{name}")
        print(f"  hits   : min {h[0]:.5f}  median {h[len(h)//2]:.5f}  max {h[-1]:.5f}")
        print(f"  misses : min {m[0]:.5f}  median {m[len(m)//2]:.5f}  max {m[-1]:.5f}")
        # Clean separation means max(miss) < min(hit) for a lower-is-worse signal.
        if lower_is_worse:
            gap = h[0] - m[-1]
            print(f"  separation (min hit - max miss): {gap:+.5f}"
                  f"  {'clean' if gap > 0 else 'overlap'}")
        print()

    _describe("top_rrf", lower_is_worse=True)
    _describe("margin", lower_is_worse=True)
    _describe("doc_spread", lower_is_worse=False)

    # Threshold sweep on the chosen signal (top_rrf).
    print("top_rrf threshold sweep (escalate when top_rrf < t):")
    print(f"  {'t':<10}{'escalated':<12}{'misses caught':<16}{'hits escalated':<16}")
    candidates = sorted({round(r["top_rrf"], 4) for r in rows})
    fine = candidates + [  # plus midpoints between observed values
        round((candidates[i] + candidates[i + 1]) / 2, 5)
        for i in range(len(candidates) - 1)
    ]
    best = None
    for t in sorted(set(fine)):
        esc = [r for r in rows if r["top_rrf"] < t]
        caught = sum(1 for r in esc if not r["hit"])
        hits_esc = sum(1 for r in esc if r["hit"])
        print(f"  {t:<10.5f}{len(esc):<12}{f'{caught}/{len(misses)}':<16}{f'{hits_esc}/{len(hits)}':<16}")
        score = (caught, -hits_esc)  # most misses caught, then fewest hits dragged along
        if best is None or score > best[0]:
            best = (score, t, caught, hits_esc, len(esc))

    _, t, caught, hits_esc, esc_n = best
    print(f"\nbest top_rrf cutoff on {split}: {t:.5f}")
    print(f"  escalates {esc_n}/{len(rows)} questions to catch {caught}/{len(misses)} "
          f"misses, and takes {hits_esc}/{len(hits)} that already hit along with them")
    if hits_esc > caught:
        print("  the cutoff escalates more good retrievals than bad ones: not usable.")

    OUT_PATH.write_text(json.dumps({
        "split": split,
        "n": len(rows),
        "misses": len(misses),
        "best_cutoff": t,
        "usable": caught > hits_esc,
        "at_best_cutoff": {
            "escalated": esc_n,
            "misses_caught": caught,
            "hits_escalated": hits_esc,
        },
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Can a retrieval signal drive the router?")
    ap.add_argument("--split", default="dev", help="dev or holdout")
    raise SystemExit(calibrate(ap.parse_args().split))
