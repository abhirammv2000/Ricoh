"""Compare reranker models on retrieval quality.

The embedding sweep (eval/sweep_embeddings.py) toggles ONE reranker,
`cross-encoder/ms-marco-MiniLM-L-6-v2` (a 2020 MS MARCO cross-encoder), on and
off. This varies the reranker model itself, against the `minilm` index (the best
embedder on this corpus per that sweep), over the 100-question set. Retrieval
only, no LLM, so the numbers are exact and free.

Each reranker is scored against the same fused candidate pool
(`top_k=20 -> final_k=5`), so a difference is the reranker.

    python -m eval.reranker_sweep
    python -m eval.reranker_sweep --rerankers BAAI/bge-reranker-base

Writes eval/reranker_sweep.{json,md}.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.config import PROJECT_ROOT  # noqa: E402
from eval.sweep_embeddings import _distinct_ranked_docs, _score_one  # noqa: E402

INDEX_ROOT: Path = PROJECT_ROOT / "eval" / "indexes"
QUESTIONS_PATH: Path = PROJECT_ROOT / "eval" / "generated_questions.json"
RESULT_JSON: Path = PROJECT_ROOT / "eval" / "reranker_sweep.json"
RESULT_MD: Path = PROJECT_ROOT / "eval" / "reranker_sweep.md"

TOP_K, FINAL_K = 20, 5

DEFAULT_RERANKERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",   # current default (2020, 6 layers)
    "cross-encoder/ms-marco-MiniLM-L-12-v2",  # same family, 12 layers
    "BAAI/bge-reranker-base",                 # 2024, ~278M params
    "BAAI/bge-reranker-v2-m3",                # 2024, ~568M params, multilingual
]


def _agg(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}


def _measure(retriever, questions: list[dict], rerank: bool) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, float]]] = {"dev": [], "holdout": []}
    missed: list[int] = []
    for q in questions:
        any_hit = q.get("provenance") == "generated"
        results = retriever.retrieve(
            query=q["question"], top_k=TOP_K, final_k=FINAL_K, rerank=rerank
        )
        s = _score_one(_distinct_ranked_docs(results), q["expected_sources"], any_hit)
        by_split.setdefault(q.get("split", "dev"), []).append(s)
        if s["recall@5"] == 0.0:
            missed.append(q["id"])
    all_rows = by_split["dev"] + by_split["holdout"]
    return {
        "dev": _agg(by_split["dev"]),
        "holdout": _agg(by_split["holdout"]),
        "all": _agg(all_rows),
        "missed_ids": sorted(missed),
    }


def run(rerankers: list[str], index_label: str) -> int:
    import src.retriever as rmod
    from src.retriever import HybridRetriever

    index_dir = INDEX_ROOT / index_label
    if not (index_dir / "chroma.sqlite3").exists():
        raise SystemExit(f"no index for '{index_label}'. run: python -m eval.sweep_embeddings --build {index_label}")

    questions = json.loads(io.open(QUESTIONS_PATH, encoding="utf-8").read())["questions"]
    retriever = HybridRetriever(persist_dir=index_dir)

    rows: list[dict[str, Any]] = []
    print(f"baseline: {index_label}, no reranker")
    base = _measure(retriever, questions, rerank=False)
    rows.append({"reranker": "none", **base})

    for model_id in rerankers:
        print(f"reranker: {model_id}")
        rmod._RERANKER = None
        rmod.RERANKER_MODEL = model_id
        t0 = time.time()
        try:
            r = _measure(retriever, questions, rerank=True)
        except Exception as exc:  # a model that will not load/predict here
            print(f"  skipped ({type(exc).__name__}: {str(exc)[:120]})")
            rows.append({"reranker": model_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        r["seconds_per_query"] = round((time.time() - t0) / len(questions), 2)
        rows.append({"reranker": model_id, **r})
        print(f"  all: {r['all']}  ({r['seconds_per_query']}s/query)")

    payload = {"index": index_label, "top_k": TOP_K, "final_k": FINAL_K, "rows": rows}
    RESULT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_md(payload)
    print(f"\nwrote {RESULT_JSON}\nwrote {RESULT_MD}")
    return 0


def _write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# Reranker sweep (retrieval only)",
        "",
        f"Reranker model varied against the `{payload['index']}` index, "
        f"`top_k={payload['top_k']} -> final_k={payload['final_k']}`, over the "
        "100-question set. No LLM. `none` is the fused pool with no reranking.",
        "",
        "| reranker | split | R@1 | R@3 | R@5 | MRR | nDCG@5 | missed | s/query |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["rows"]:
        if "error" in r:
            lines.append(f"| `{r['reranker']}` | - | did not run: {r['error'][:60]} |  |  |  |  |  |  |")
            continue
        spq = r.get("seconds_per_query", "")
        for split in ("dev", "holdout", "all"):
            m = r.get(split)
            if not m:
                continue
            missed = len(r["missed_ids"]) if split == "all" else ""
            name = f"`{r['reranker']}`" if split == "dev" or r["reranker"] == "none" else ""
            lines.append(
                f"| {name} | {split} | {m['recall@1']:.2f} | {m['recall@3']:.2f} "
                f"| {m['recall@5']:.2f} | {m['mrr']:.2f} | {m['ndcg@5']:.2f} | {missed} "
                f"| {spq if split == 'dev' else ''} |"
            )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare reranker models, retrieval only")
    ap.add_argument("--rerankers", nargs="+", default=DEFAULT_RERANKERS)
    ap.add_argument("--index", default="minilm")
    args = ap.parse_args()
    raise SystemExit(run(args.rerankers, args.index))
