"""Check RRF_K on this corpus. 60 is the constant from the original RRF paper, which says where it
came from but not whether it's right here.

A chunk's score is the sum of 1 / (k + rank) over the lists it is in. With k=60 and a pool of 10,
rank 1 gets 1/61 and rank 10 gets 1/70, only 15% apart, but a chunk in both lists scores about twice
one in a single list. So it behaves more like "did both retrievers pick it" than "how high did they
rank it". A high k favors agreement and a low k favors rank, which helps when one retriever is clearly
right and the other has no opinion. This measures which wins here. With only 8 scorable questions,
prefer a k on a plateau over one that wins by a single question.

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

    # get the ranked lists once, only the fusion changes with k
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
