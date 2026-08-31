"""Compare embedding models on retrieval quality, no LLM involved.

The production index uses all-MiniLM-L6-v2 (ChromaDB's default, a 2021 model).
This measures whether a newer model, a wider candidate pool, or the reranker
improve retrieval on the 100-question benchmark. Retrieval is deterministic, so
the numbers are free and reproduce exactly. Whether a retrieval gain carries
through to judged answer quality is a separate paid question for the harness.

Each model needs its own index (vectors are model-specific), built under
eval/indexes/<label>/ so production chroma_db/ is untouched.

    python -m eval.sweep_embeddings --build bge-small
    python -m eval.sweep_embeddings --measure           # + --rerank for the reranker rows

Writes eval/embedding_sweep.{json,md}.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Cap the thread pools before torch loads; without this, embedding thrashes on a
# CPU-only box under memory pressure.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.config import PROJECT_ROOT

INDEX_ROOT: Path = PROJECT_ROOT / "eval" / "indexes"
QUESTIONS_PATH: Path = PROJECT_ROOT / "eval" / "generated_questions.json"
RESULT_JSON: Path = PROJECT_ROOT / "eval" / "embedding_sweep.json"
RESULT_MD: Path = PROJECT_ROOT / "eval" / "embedding_sweep.md"

# label -> {model_id, kind: default|st|openai, query_prefix, doc_prefix, dim}
_BGE_Q = "Represent this sentence for searching relevant passages: "
MODELS: dict[str, dict[str, Any]] = {
    "minilm": {"model_id": "", "kind": "default", "dim": 384},
    "bge-small": {
        "model_id": "BAAI/bge-small-en-v1.5", "kind": "st",
        "query_prefix": _BGE_Q, "doc_prefix": "", "dim": 384,
    },
    "bge-base": {
        "model_id": "BAAI/bge-base-en-v1.5", "kind": "st",
        "query_prefix": _BGE_Q, "doc_prefix": "", "dim": 768,
    },
    "e5-base-v2": {
        "model_id": "intfloat/e5-base-v2", "kind": "st",
        "query_prefix": "query: ", "doc_prefix": "passage: ", "dim": 768,
    },
    "openai-3-large": {"model_id": "text-embedding-3-large", "kind": "openai", "dim": 3072},
}

TOP_K_GRID = (10, 20)
# Reranker rows are off by default (the cross-encoder is the slow part); --rerank adds them.


# Embedding functions

class _PrefixST:
    """Sentence-transformers embedding function that prepends a fixed prefix.

    bge and e5 want a different instruction on queries vs passages, and ChromaDB
    calls the embedding function the same way for both. So the prefix is baked in
    at construction: "doc" mode when building, "query" mode when measuring.
    """

    def __init__(self, model_id: str, prefix: str) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        self._model = SentenceTransformer(model_id)
        self._prefix = prefix
        self._id = model_id

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        texts = [self._prefix + t for t in input]
        vecs = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]

    def name(self) -> str:
        # Ignore the prefix so doc-mode and query-mode read as the same EF to ChromaDB.
        return f"prefix-st:{self._id}"


def _embedding_function(label: str, mode: str):
    """EF for a label (None = chroma default). mode is "doc" or "query"."""
    spec = MODELS[label]
    kind = spec["kind"]

    if kind == "default":
        return None

    if kind == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit(f"{label} needs OPENAI_API_KEY. Set it and retry.")
        from chromadb.utils import embedding_functions

        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"], model_name=spec["model_id"]
        )

    # _PrefixST even with an empty prefix, so doc and query vectors share the
    # same normalisation and pooling.
    prefix = spec.get("doc_prefix" if mode == "doc" else "query_prefix", "")
    return _PrefixST(spec["model_id"], prefix)


# Build

def build_index(label: str) -> None:
    if label not in MODELS:
        raise SystemExit(f"unknown model '{label}'. known: {', '.join(MODELS)}")
    from src.ingest import ingest_all
    from src.retriever import HybridRetriever

    index_dir = INDEX_ROOT / label
    chunks = ingest_all()
    if not chunks:
        raise SystemExit("no PDFs found in data/. add them and retry.")

    ef = _embedding_function(label, mode="doc")
    retriever = HybridRetriever(persist_dir=index_dir, embedding_function=ef)
    if retriever.index_size >= len(chunks) and retriever.bm25_ready:
        print(f"{label}: already built ({retriever.index_size} chunks), skipping")
        return
    print(f"building '{label}' at {index_dir} ({len(chunks)} chunks, dim {MODELS[label]['dim']})")
    retriever.build_index(chunks)
    print(f"  done: {retriever.index_size} chunks")


# Metrics

def _distinct_ranked_docs(results: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for r in results:
        d = r.get("source_document")
        if d and d not in seen:
            seen.append(d)
    return seen


def _score_one(ranked: list[str], expected: list[str], any_hit: bool) -> dict[str, float]:
    exp = set(expected)
    out: dict[str, float] = {}
    for k in (1, 3, 5):
        top = set(ranked[:k])
        if any_hit:
            out[f"recall@{k}"] = 1.0 if exp & top else 0.0
        else:
            out[f"recall@{k}"] = sum(1 for e in exp if e in top) / len(exp)

    first_rank = next((i + 1 for i, d in enumerate(ranked) if d in exp), None)
    out["mrr"] = 1.0 / first_rank if first_rank else 0.0

    if any_hit:
        # Alternatives: ideal is one at rank 1 (idcg = 1), only the first counts.
        out["ndcg@5"] = 1.0 / math.log2(first_rank + 1) if first_rank and first_rank <= 5 else 0.0
    else:
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:5]) if d in exp)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(exp), 5)))
        out["ndcg@5"] = dcg / idcg if idcg else 0.0
    return out


def _agg(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}


def measure_matrix(label: str) -> list[dict[str, Any]]:
    """Every (top_k, reranker) combo for one model, in this process."""
    from src.retriever import HybridRetriever

    index_dir = INDEX_ROOT / label
    if not (index_dir / "chroma.sqlite3").exists():
        raise SystemExit(f"no index for '{label}'. run --build {label} first.")

    ef = _embedding_function(label, mode="query")
    retriever = HybridRetriever(persist_dir=index_dir, embedding_function=ef)
    questions = json.load(io.open(QUESTIONS_PATH, encoding="utf-8"))["questions"]
    rerank_grid = (False, True) if os.getenv("SWEEP_RERANK") else (False,)

    out: list[dict[str, Any]] = []
    for top_k in TOP_K_GRID:
        for rerank in rerank_grid:
            print(f"  {label} top_k={top_k} rerank={rerank} ...", file=sys.stderr, flush=True)
            by_split: dict[str, list[dict[str, float]]] = {"dev": [], "holdout": []}
            missed: list[int] = []
            for q in questions:
                any_hit = q.get("provenance") == "generated"
                results = retriever.retrieve(
                    query=q["question"], top_k=top_k, final_k=5, rerank=rerank
                )
                ranked = _distinct_ranked_docs(results)
                s = _score_one(ranked, q["expected_sources"], any_hit)
                by_split.setdefault(q.get("split", "dev"), []).append(s)
                if s["recall@5"] == 0.0:
                    missed.append(q["id"])
            all_rows = by_split["dev"] + by_split["holdout"]
            out.append({
                "label": label,
                "top_k": top_k,
                "rerank": rerank,
                "n": {k: len(v) for k, v in {**by_split, "all": all_rows}.items()},
                "dev": _agg(by_split["dev"]),
                "holdout": _agg(by_split["holdout"]),
                "all": _agg(all_rows),
                "missed_ids": sorted(missed),
            })
    return out


# Orchestration

def _run_all(labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        print(f"== {label} ==", flush=True)
        rows_path = INDEX_ROOT / label / "_sweep_rows.json"
        rows_path.unlink(missing_ok=True)
        # A subprocess per model so the loaded models don't stack up in memory.
        proc = subprocess.run(
            [sys.executable, "-m", "eval.sweep_embeddings", "--one", label],
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0 or not rows_path.exists():
            raise SystemExit(f"sub-run failed for {label}")
        rows.extend(json.loads(rows_path.read_text()))
    return rows


def _write_report(rows: list[dict[str, Any]]) -> None:
    RESULT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Embedding sweep (retrieval only)",
        "",
        "Retriever metrics with the embedding model, candidate pool and reranker",
        "varied. No LLM, so these are exact and free to reproduce. `minilm` is the",
        "current production default.",
        "",
        "Decide on `dev` (70 questions), confirm on `holdout` (30). `all` is for",
        "comparison with the README, whose retriever numbers are over all 100.",
        "",
        "| model | top_k | rerank | split | R@1 | R@3 | R@5 | MRR | nDCG@5 | missed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for split in ("dev", "holdout", "all"):
            m = r[split]
            if not m:
                continue
            missed = len(r["missed_ids"]) if split == "all" else ""
            lines.append(
                f"| {r['label']} | {r['top_k']} | {'yes' if r['rerank'] else 'no'} | {split} "
                f"| {m['recall@1']:.2f} | {m['recall@3']:.2f} | {m['recall@5']:.2f} "
                f"| {m['mrr']:.2f} | {m['ndcg@5']:.2f} | {missed} |"
            )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {RESULT_JSON}\nwrote {RESULT_MD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Embedding model sweep, retrieval only")
    ap.add_argument("--build", metavar="LABEL", help="build the index for one model")
    ap.add_argument("--measure", nargs="*", metavar="LABEL",
                    help="measure the matrix for these models (default: all built)")
    ap.add_argument("--rerank", action="store_true",
                    help="also measure with the cross-encoder reranker (much slower)")
    ap.add_argument("--one", metavar="LABEL", help=argparse.SUPPRESS)  # internal
    args = ap.parse_args()

    if args.rerank:
        os.environ["SWEEP_RERANK"] = "1"  # sub-runs read this

    if args.build:
        build_index(args.build)
    elif args.one:
        rows = measure_matrix(args.one)
        (INDEX_ROOT / args.one / "_sweep_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    elif args.measure is not None:
        built = [d.name for d in sorted(INDEX_ROOT.glob("*")) if (d / "chroma.sqlite3").exists()]
        labels = args.measure or built
        if not labels:
            raise SystemExit("no indexes built yet. run --build LABEL first.")
        _write_report(_run_all(labels))
    else:
        ap.error("pass --build LABEL or --measure [LABEL ...]")
