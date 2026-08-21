"""eval/sweep_rrf_k.py - Justify RRF_K for THIS corpus, not by citation.

`RRF_K = 60` is the constant from Cormack et al. (2009).  Citing the paper
explains where the number came from; it does not explain why it is right
here, and the two are not the same thing.

The number matters more than it looks
RRF scores a document as  Σ 1 / (k + rank_i)  over the lists it appears in.

With k = 60 and a candidate pool of 10:

    rank 1  ->  1/61 = 0.01639
    rank 10 ->  1/70 = 0.01429      ... only 15% apart

but a document appearing in *both* lists gets roughly twice the score of
one appearing in a single list, whatever its rank.  So at this pool size the
formula is closer to "did both retrievers vote for it" than to "how highly
did they rank it", k = 60 flattens rank almost entirely.

That is a real design choice, and it cuts both ways:

  * Agreement-dominant (high k) is robust when either retriever is noisy.
  * Rank-dominant (low k) is better when one retriever is decisively right
    and the other has no opinion, which is common on this corpus, where the
    answer document is often a strong *semantic* match with no distinctive
    keyword overlap.

This script measures which regime actually wins here instead of assuming.

Reminder on sample size: 8 scorable questions. Prefer a k that sits on a
plateau over one that wins by a single question.

Usage:
    python -m eval.sweep_rrf_k
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from src.config import PROJECT_ROOT, RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K, RRF_K
from src.retriever import HybridRetriever, get_retriever

GROUND_TRUTH_PATH: Path = PROJECT_ROOT / "eval" / "ground_truth.json"

K_GRID = (0, 1, 5, 10, 20, 60, 120)


def sweep() -> int:
    questions = [
        q
        for q in json.loads(io.open(GROUND_TRUTH_PATH, encoding="utf-8").read())["questions"]
        if q.get("expected_sources")
    ]
    retriever = get_retriever()

    print(f"Questions: {len(questions)}   top_k={RETRIEVAL_TOP_K}  "
          f"final_k={RETRIEVAL_FINAL_K}  (current RRF_K={RRF_K})\n")

    # Retrieve the per-method ranked lists ONCE; only fusion varies with k,
    # so re-querying per k would be wasted work and identical input.
    cached: list[tuple[list, list, list[str]]] = []
    for q in questions:
        vec = retriever._vector_search(q["question"], top_k=RETRIEVAL_TOP_K)
        bm = retriever._bm25_search(q["question"], top_k=RETRIEVAL_TOP_K)
        cached.append((vec, bm, q["expected_sources"]))

    print(f"{'RRF_K':<8}{'recall@' + str(RETRIEVAL_FINAL_K):<12}{'full-hit':<11}{'mean rank of expected'}")
    for k in K_GRID:
        recalls, full, ranks = [], 0, []
        for vec, bm, expected in cached:
            fused = HybridRetriever._rrf_fuse(
                vec, bm, k=k, final_k=RETRIEVAL_FINAL_K
            )
            docs: list[str] = []
            for d in fused:
                s = d["source_document"]
                if s not in docs:
                    docs.append(s)
            hits = sum(1 for e in expected if e in docs)
            recalls.append(hits / len(expected))
            if hits == len(expected):
                full += 1
            for e in expected:
                if e in docs:
                    ranks.append(docs.index(e) + 1)
        mean_recall = sum(recalls) / len(recalls)
        mean_rank = sum(ranks) / len(ranks) if ranks else float("nan")
        marker = "   <- current" if k == RRF_K else ""
        print(
            f"{k:<8}{mean_recall:<12.3f}{f'{full}/{len(questions)}':<11}"
            f"{mean_rank:.2f}{marker}"
        )

    print()
    print("=" * 70)
    print("k -> 0 makes RRF purely rank-driven (1/rank).")
    print("Large k flattens rank and rewards cross-retriever agreement instead.")
    print("'mean rank of expected' is the sharper signal: recall can be flat")
    print("while the expected document moves up or down inside the top-k, and")
    print("position matters once a reranker or an LLM reads the context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(sweep())
