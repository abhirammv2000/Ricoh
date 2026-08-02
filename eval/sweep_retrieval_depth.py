"""eval/sweep_retrieval_depth.py - Justify RETRIEVAL_TOP_K and RETRIEVAL_FINAL_K.

Both constants were originally set by intuition.  This script replaces the
intuition with a measurement, and — just as importantly — shows the cost side
of the trade so the choice is not simply "bigger is better".

The trade-off being measured
────────────────────────────
``final_k`` is how many fused chunks the synthesizer actually sees.

  Raising it   → higher chance the right document is in context (recall up)
  Raising it   → more input tokens per call, on EVERY LLM call downstream
                 (verifier and synthesizer both embed the evidence block),
                 and more irrelevant text competing with the answer.

So the honest question is not "which k maximises recall" but "where does
recall stop improving, and what does that k cost per query".

Why the retriever sweep is free
───────────────────────────────
Retrieval is deterministic (verified: identical results across repeated runs
with fresh clients). So the recall side of this sweep costs nothing and is
exactly reproducible. Only the downstream generation quality needs API spend,
which is why we settle k here first and validate once, rather than sweeping
the whole pipeline.

Overfitting caveat — read before trusting the output
────────────────────────────────────────────────────
There are only 8 scorable questions. Choosing k to squeeze out the last one
is fitting a hyperparameter to 8 samples and will not generalise. Prefer the
knee of the curve over the argmax, and treat any k justified by a single
question as unproven until the eval set is larger.

Usage:
    python -m eval.sweep_retrieval_depth
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from src.config import PROJECT_ROOT, RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K
from src.retriever import get_retriever

GROUND_TRUTH_PATH: Path = PROJECT_ROOT / "eval" / "ground_truth.json"

TOP_K_GRID = (10, 25, 50)
FINAL_K_GRID = (1, 3, 5, 8, 10, 15, 20)

# Rough proxy for what a chunk costs downstream. The evidence block is
# embedded in BOTH the verifier and synthesizer prompts, so each extra chunk
# is paid for more than once per query.
MEDIAN_CHUNK_WORDS = 307
TOKENS_PER_WORD = 1.3
PROMPTS_EMBEDDING_EVIDENCE = 2


def _questions() -> list[dict]:
    gt = json.loads(io.open(GROUND_TRUTH_PATH, encoding="utf-8").read())["questions"]
    return [q for q in gt if q.get("expected_sources")]


def sweep() -> int:
    questions = _questions()
    retriever = get_retriever()
    print(f"Scorable questions: {len(questions)} (of 10 - refusals have no expected source)")
    print(f"Current config: RETRIEVAL_TOP_K={RETRIEVAL_TOP_K}, "
          f"RETRIEVAL_FINAL_K={RETRIEVAL_FINAL_K}\n")

    max_final = max(FINAL_K_GRID)

    for top_k in TOP_K_GRID:
        # Retrieve once at the deepest final_k, then evaluate every shallower
        # depth as a prefix of that ranking - the ranking is stable, so this
        # is equivalent to re-running and far cheaper.
        per_question_docs: list[tuple[list[str], list[str]]] = []
        for q in questions:
            results = retriever.retrieve(
                query=q["question"], top_k=top_k, final_k=max_final
            )
            ranked: list[str] = []
            for r in results:
                d = r.get("source_document")
                if d and d not in ranked:
                    ranked.append(d)
            per_question_docs.append((ranked, q["expected_sources"]))

        print(f"top_k={top_k} (candidates per method before fusion)")
        print(f"  {'final_k':<9}{'recall':<9}{'full-hit':<10}{'~evidence tokens/query'}")
        for final_k in FINAL_K_GRID:
            recalls = []
            full = 0
            for ranked, expected in per_question_docs:
                top = set(ranked[:final_k])
                hits = sum(1 for e in expected if e in top)
                recalls.append(hits / len(expected))
                if hits == len(expected):
                    full += 1
            mean_recall = sum(recalls) / len(recalls)
            est_tokens = int(
                final_k * MEDIAN_CHUNK_WORDS * TOKENS_PER_WORD
                * PROMPTS_EMBEDDING_EVIDENCE
            )
            marker = "  <- current" if (
                final_k == RETRIEVAL_FINAL_K and top_k == RETRIEVAL_TOP_K
            ) else ""
            print(
                f"  {final_k:<9}{mean_recall:<9.3f}{f'{full}/{len(questions)}':<10}"
                f"{est_tokens:,}{marker}"
            )
        print()

    print("=" * 70)
    print("Reading this table:")
    print("  * 'recall' is mean fraction of expected docs inside the top final_k")
    print("    DOCUMENTS (not chunks) - the unit that matters on a corpus of")
    print("    mostly single-page articles.")
    print("  * 'full-hit' counts questions where EVERY expected doc was found;")
    print("    it is the stricter and more honest headline.")
    print("  * token estimates are indicative, not billed values - the harness")
    print("    reports real token counts from the API.")
    print("  * pick the knee, not the argmax: n is small enough that the last")
    print("    increment is probably one question, not a real effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(sweep())
