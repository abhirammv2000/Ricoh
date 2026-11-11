"""Does fusing vector search and BM25 beat either one alone, and does the fusion method matter?

The retriever merges vector search and BM25 with reciprocal rank fusion. That was a design choice, and
this measures it on the 100 benchmark questions with no LLM: each method alone, RRF with several
constants, and a weighted blend of the two scores. It also looks up exact codes taken from the corpus
itself, because the 100 questions hardly contain any.

    python -m eval.retrieval_ablation --index /path/to/copy_of_chroma_db

Use a copy of the index. Opening a chroma folder can write files into it. Writes eval/retrieval_ablation.md.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.sweep_embeddings import QUESTIONS_PATH, _agg, _distinct_ranked_docs, _score_one
from src.config import PROJECT_ROOT
from src.retriever import HybridRetriever

TOP_K = 10
FINAL_K = 5
OUTPUT_PATH: Path = PROJECT_ROOT / "eval" / "retrieval_ablation.md"

# a token with letters and digits, such as IBM500 or AIWI6073E, in a question
CODE_IN_QUESTION = re.compile(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9-]{3,}\b")
# the shape of a message id in the documents
CODE_IN_CHUNK = re.compile(r"\b[A-Z]{2,6}[0-9]{3,6}[A-Z]?\b")


def _scaled(results: list[dict[str, Any]]) -> dict[str, float]:
    """Each result's score stretched to 0..1 within its own list, so two kinds of score can be added."""
    scores = [r["score"] for r in results]
    low, high = min(scores, default=0.0), max(scores, default=1.0)
    span = (high - low) or 1.0
    return {r["id"]: (r["score"] - low) / span for r in results}


def _blend(vector: list[dict], keyword: list[dict], vector_weight: float) -> list[dict]:
    """Add the two scaled scores with a weight on the vector side and keep the best FINAL_K."""
    scaled_vector, scaled_keyword = _scaled(vector), _scaled(keyword)
    by_id = {r["id"]: r for r in vector + keyword}
    total = {}
    for chunk_id in by_id:
        vector_part = vector_weight * scaled_vector.get(chunk_id, 0.0)
        keyword_part = (1 - vector_weight) * scaled_keyword.get(chunk_id, 0.0)
        total[chunk_id] = vector_part + keyword_part
    best = sorted(total, key=total.get, reverse=True)[:FINAL_K]
    return [by_id[chunk_id] for chunk_id in best]


def _methods(retriever: HybridRetriever) -> dict[str, Any]:
    def rrf(constant: int):
        return lambda vector, keyword: retriever._rrf_fuse(vector, keyword, k=constant, final_k=FINAL_K)

    return {
        "vector only": lambda vector, keyword: vector[:FINAL_K],
        "BM25 only": lambda vector, keyword: keyword[:FINAL_K],
        "RRF k=60 (current)": rrf(60),
        "RRF k=10": rrf(10),
        "RRF k=30": rrf(30),
        "RRF k=100": rrf(100),
        "score blend 50/50": lambda vector, keyword: _blend(vector, keyword, 0.5),
        "score blend 70/30 vector": lambda vector, keyword: _blend(vector, keyword, 0.7),
        "score blend 30/70 vector": lambda vector, keyword: _blend(vector, keyword, 0.3),
    }


def measure_questions(retriever: HybridRetriever) -> tuple[list[str], int]:
    """The table of each method on the 100 questions, and how many questions contain a code."""
    questions = json.load(open(QUESTIONS_PATH, encoding="utf-8"))["questions"]
    methods = _methods(retriever)
    rows: dict[str, dict[str, list]] = {name: {"all": [], "dev": [], "holdout": []} for name in methods}
    missed: dict[str, set[int]] = {name: set() for name in methods}
    with_code = 0
    for q in questions:
        vector = retriever._vector_search(q["question"], top_k=TOP_K)
        keyword = retriever._bm25_search(q["question"], top_k=TOP_K)
        with_code += bool(CODE_IN_QUESTION.search(q["question"]))
        any_hit = q.get("provenance") == "generated"
        for name, fuse in methods.items():
            score = _score_one(_distinct_ranked_docs(fuse(vector, keyword)), q["expected_sources"], any_hit)
            rows[name]["all"].append(score)
            rows[name][q.get("split", "dev")].append(score)
            if score["recall@5"] == 0.0:
                missed[name].add(q["id"])

    lines = [
        "| method | R@1 | R@3 | R@5 | MRR | missed | dev R@5 | holdout R@5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in methods:
        a, dev, hold = _agg(rows[name]["all"]), _agg(rows[name]["dev"]), _agg(rows[name]["holdout"])
        lines.append(
            f"| {name} | {a['recall@1']:.2f} | {a['recall@3']:.2f} | {a['recall@5']:.2f} | {a['mrr']:.2f} "
            f"| {len(missed[name])} | {dev['recall@5']:.3f} | {hold['recall@5']:.3f} |"
        )
    vector_miss, keyword_miss = missed["vector only"], missed["BM25 only"]
    both = vector_miss & keyword_miss
    lines += [
        "",
        f"BM25 alone misses {len(keyword_miss - vector_miss)} questions that vector search finds, and vector search "
        f"misses {len(vector_miss - keyword_miss)} that BM25 finds. Both miss {len(both)}.",
        f"RRF misses {len(missed['RRF k=60 (current)'])}: those {len(both)} and "
        f"{sorted(missed['RRF k=60 (current)'] - both)}, which one of the two methods alone finds.",
    ]
    return lines, with_code


def measure_codes(retriever: HybridRetriever) -> list[str]:
    """Look up codes taken from the corpus and count how often each method puts their chunk in the top 5."""
    where: dict[str, set[str]] = defaultdict(set)
    for chunk in retriever._bm25_chunks:
        for code in set(CODE_IN_CHUNK.findall(chunk["text"])):
            where[code].add(chunk["id"])
    specific = sorted(code for code, ids in where.items() if len(ids) <= 2)
    random.Random(7).shuffle(specific)
    hits = {"vector only": 0, "BM25 only": 0, "RRF (current)": 0}
    for code in specific:
        query = f"What does {code} mean?"
        vector = retriever._vector_search(query, top_k=TOP_K)
        keyword = retriever._bm25_search(query, top_k=TOP_K)
        fused = retriever._rrf_fuse(vector, keyword, final_k=FINAL_K)
        wanted = where[code]
        hits["vector only"] += bool(wanted & {r["id"] for r in vector[:FINAL_K]})
        hits["BM25 only"] += bool(wanted & {r["id"] for r in keyword[:FINAL_K]})
        hits["RRF (current)"] += bool(wanted & {r["id"] for r in fused})
    lines = [
        f"The corpus has {len(where)} distinct code-like tokens in {len(retriever._bm25_chunks)} chunks. "
        f"{len(specific)} of them are in at most 2 chunks, and each was looked up as \"What does <code> mean?\".",
        "",
        "| method | found in the top 5 |",
        "|---|---|",
    ]
    lines += [f"| {name} | {count} of {len(specific)} |" for name, count in hits.items()]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, required=True, help="a copy of the chroma_db folder")
    args = parser.parse_args()

    retriever = HybridRetriever(persist_dir=args.index)
    table, with_code = measure_questions(retriever)
    code_lines = measure_codes(retriever)
    report = [
        "# Retrieval ablation: each method alone, and the fusion method",
        "",
        f"Retrieval only, no LLM, top_k={TOP_K}, final_k={FINAL_K}, the 100 questions "
        f"({with_code} of them contain a code-like token).",
        "",
        *table,
        "",
        "## Exact codes from the corpus",
        "",
        *code_lines,
        "",
    ]
    OUTPUT_PATH.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
