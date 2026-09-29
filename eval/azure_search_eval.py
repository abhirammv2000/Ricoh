"""Compare Azure AI Search with the local ChromaDB + BM25 retriever, no LLM involved.

Same 100 questions, same chunks, same embeddings (all-MiniLM-L6-v2), same
scoring code as eval/sweep_embeddings.py. Only the search engine changes.
Retrieval is deterministic, so the numbers are free to reproduce.

Azure's semantic ranker is metered (a monthly free allowance, then a billing
error on the free plan), so it only runs with --semantic and the number of
requests it will send is printed first.

    python -m src.azure_retriever --build         # once, uploads the chunks
    python -m eval.azure_search_eval               # local baseline + Azure keyword/vector/hybrid
    python -m eval.azure_search_eval --semantic    # adds hybrid + semantic ranker
    python -m eval.azure_search_eval --chroma-rerank   # adds the local cross-encoder row

Writes eval/azure_search_eval.{json,md}.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from typing import Any

from eval.sweep_embeddings import (
    INDEX_ROOT,
    QUESTIONS_PATH,
    _agg,
    _distinct_ranked_docs,
    _score_one,
)
from src.config import PROJECT_ROOT

RESULT_JSON = PROJECT_ROOT / "eval" / "azure_search_eval.json"
RESULT_MD = PROJECT_ROOT / "eval" / "azure_search_eval.md"

FINAL_K = 5
TOP_K_GRID = (10, 20)


def _measure(label: str, run_query, questions: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, float]]] = {"dev": [], "holdout": []}
    missed: list[int] = []
    seconds = 0.0
    for q in questions:
        start = time.perf_counter()
        results = run_query(q["question"])
        seconds += time.perf_counter() - start
        ranked = _distinct_ranked_docs(results)
        s = _score_one(ranked, q["expected_sources"], q.get("provenance") == "generated")
        by_split.setdefault(q.get("split", "dev"), []).append(s)
        if s["recall@5"] == 0.0:
            missed.append(q["id"])
    all_rows = by_split["dev"] + by_split["holdout"]
    row = {
        "label": label,
        **extra,
        "n": {k: len(v) for k, v in {**by_split, "all": all_rows}.items()},
        "dev": _agg(by_split["dev"]),
        "holdout": _agg(by_split["holdout"]),
        "all": _agg(all_rows),
        "missed_ids": sorted(missed),
        "ms_per_query": round(1000 * seconds / len(questions), 1),
    }
    print(f"  {label}: R@5 {row['all']['recall@5']:.2f}, MRR {row['all']['mrr']:.2f}, "
          f"{row['ms_per_query']} ms/query", flush=True)
    return row


def run(with_semantic: bool, chroma_rerank: bool) -> list[dict[str, Any]]:
    from src.azure_retriever import AzureRetriever
    from src.retriever import HybridRetriever

    questions = json.load(io.open(QUESTIONS_PATH, encoding="utf-8"))["questions"]
    azure = AzureRetriever()
    print(f"azure index holds {azure.index_size} chunks, {len(questions)} questions", flush=True)

    rows: list[dict[str, Any]] = []

    print("== local ChromaDB + BM25 + RRF ==", flush=True)
    # The same index eval/sweep_embeddings.py measured, so these rows match the
    # README. A second HNSW build of the same chunks (the production chroma_db)
    # lands about one question away at top_k=20.
    if not (INDEX_ROOT / "minilm" / "chroma.sqlite3").exists():
        raise SystemExit("no local baseline index. run: python -m eval.sweep_embeddings --build minilm")
    local = HybridRetriever(persist_dir=INDEX_ROOT / "minilm")
    for top_k in TOP_K_GRID:
        rows.append(_measure(
            f"local hybrid (RRF) top_k={top_k}",
            lambda text, k=top_k: local.retrieve(text, top_k=k, final_k=FINAL_K, rerank=False),
            questions, {"backend": "local", "mode": "hybrid_rrf", "top_k": top_k},
        ))
    if chroma_rerank:
        rows.append(_measure(
            "local hybrid (RRF) + cross-encoder top_k=20",
            lambda text: local.retrieve(text, top_k=20, final_k=FINAL_K, rerank=True),
            questions, {"backend": "local", "mode": "hybrid_rrf_cross_encoder", "top_k": 20},
        ))

    print("== Azure AI Search ==", flush=True)
    for mode in ("keyword", "vector"):
        rows.append(_measure(
            f"azure {mode}",
            lambda text, m=mode: azure.retrieve(text, mode=m, top_k=20, final_k=FINAL_K),
            questions, {"backend": "azure", "mode": mode, "top_k": None},
        ))
    for top_k in TOP_K_GRID:
        rows.append(_measure(
            f"azure hybrid top_k={top_k}",
            lambda text, k=top_k: azure.retrieve(text, mode="hybrid", top_k=k, final_k=FINAL_K),
            questions, {"backend": "azure", "mode": "hybrid", "top_k": top_k},
        ))
    if with_semantic:
        for top_k in TOP_K_GRID:
            rows.append(_measure(
                f"azure hybrid + semantic ranker top_k={top_k}",
                lambda text, k=top_k: azure.retrieve(
                    text, mode="hybrid_semantic", top_k=k, final_k=FINAL_K
                ),
                questions, {"backend": "azure", "mode": "hybrid_semantic", "top_k": top_k},
            ))
    return rows


def _write_report(rows: list[dict[str, Any]], semantic_requests: int) -> None:
    RESULT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Azure AI Search vs local ChromaDB + BM25 (retrieval only)",
        "",
        "Same 100 questions, same 1,322 chunks, same MiniLM embeddings, same scoring as",
        "`eval/sweep_embeddings.py`. No LLM, so it is free to reproduce. Decide on `dev`",
        "(70 questions), confirm on `holdout` (30). `all` is over the 100.",
        "",
        f"Azure semantic ranker requests sent in this run: {semantic_requests}.",
        "",
        "| retriever | split | R@1 | R@3 | R@5 | MRR | nDCG@5 | missed | ms/query |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for split in ("dev", "holdout", "all"):
            m = r[split]
            if not m:
                continue
            missed = len(r["missed_ids"]) if split == "all" else ""
            ms = r["ms_per_query"] if split == "all" else ""
            lines.append(
                f"| {r['label']} | {split} | {m['recall@1']:.2f} | {m['recall@3']:.2f} "
                f"| {m['recall@5']:.2f} | {m['mrr']:.2f} | {m['ndcg@5']:.2f} | {missed} | {ms} |"
            )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {RESULT_JSON}\nwrote {RESULT_MD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Azure AI Search vs local retriever, retrieval only")
    ap.add_argument("--semantic", action="store_true",
                    help="also run the metered semantic ranker (sends 100 requests per top_k)")
    ap.add_argument("--chroma-rerank", action="store_true",
                    help="also run the local cross-encoder reranker row (slower)")
    args = ap.parse_args()

    n_questions = len(json.load(io.open(QUESTIONS_PATH, encoding="utf-8"))["questions"])
    planned = n_questions * len(TOP_K_GRID) if args.semantic else 0
    print(f"semantic ranker requests planned: {planned}", flush=True)

    _write_report(run(args.semantic, args.chroma_rerank), planned)
