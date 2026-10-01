"""Does rewriting the question before retrieval help? HyDE and multi-query against the raw question.

HyDE: an LLM writes a short passage that would answer the question and we search with that. Multi-query:
an LLM writes a few rewrites and we fuse the results. Both add an LLM call per question, so they only
belong in the pipeline if retrieval improves by more than the noise. Retrieval only, same 100 questions
and MiniLM index as the embedding sweep. The LLM output is cached in eval/query_transform_cache.json, so
a rerun is free and gives the same numbers.

    python -m eval.query_transform_sweep

Writes eval/query_transform_sweep.{json,md}.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

from eval.sweep_embeddings import (
    INDEX_ROOT,
    QUESTIONS_PATH,
    _agg,
    _distinct_ranked_docs,
    _embedding_function,
    _score_one,
)
from src.config import PROJECT_ROOT, RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K
from src.llm_factory import get_llm, response_text
from src.retriever import HybridRetriever

CACHE_PATH: Path = PROJECT_ROOT / "eval" / "query_transform_cache.json"
RESULT_JSON: Path = PROJECT_ROOT / "eval" / "query_transform_sweep.json"
RESULT_MD: Path = PROJECT_ROOT / "eval" / "query_transform_sweep.md"

PROVIDER = "google"  # gemini flash, the cheap model from the provider bakeoff
N_REWRITES = 3

HYDE_PROMPT = """\
Write a short passage (3 or 4 sentences) that could appear in the RICOH \
ProcessDirector documentation and would answer the question below. Write it as \
the manual would, with the product's own terms. Output only the passage.

Question: {question}"""

MULTI_PROMPT = """\
Rewrite the question below {n} different ways for searching the RICOH \
ProcessDirector documentation. Change the wording and use the vocabulary a \
manual would use, but keep the meaning. Output one rewrite per line, with no \
numbering and no extra text.

Question: {question}"""


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _llm_text(llm, prompt: str) -> str | None:
    """The reply text, or None if Gemini's safety filter blocked the request.

    A blocked request has a null message, which langchain_openai fails on with an AttributeError. It
    happens on harmless questions too (id 17), so it means "no rewrite for this one", not a crash.
    """
    try:
        return response_text(llm.invoke(prompt))
    except AttributeError:
        return None


def _generate(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate the rewrites for any question that isn't cached yet."""
    cache = _load_cache()
    # ids are ints but json keys are strings, so the cache is keyed by str(id)
    todo = [q for q in questions if str(q["id"]) not in cache]
    if not todo:
        return cache
    # gemini counts its thinking tokens in max_tokens, and at 400 the answer was cut off after a few words
    llm = get_llm(provider=PROVIDER, max_tokens=2000)
    print(f"generating HyDE passages and rewrites for {len(todo)} questions ({PROVIDER})", file=sys.stderr)
    for i, q in enumerate(todo, 1):
        hyde = _llm_text(llm, HYDE_PROMPT.format(question=q["question"]))
        multi = _llm_text(llm, MULTI_PROMPT.format(n=N_REWRITES, question=q["question"]))
        rewrites = [ln.strip() for ln in multi.splitlines() if ln.strip()][:N_REWRITES] if multi else []
        # a truncated output would quietly weaken the method, so stop instead of caching it
        if hyde is not None and len(hyde.split()) < 15:
            raise RuntimeError(f"question {q['id']}: HyDE passage is only {len(hyde.split())} words")
        if multi is not None and len(rewrites) < N_REWRITES:
            raise RuntimeError(f"question {q['id']}: only {len(rewrites)} rewrites")
        cache[str(q["id"])] = {"hyde": hyde, "rewrites": rewrites}
        if i % 10 == 0:
            CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            print(f"  {i}/{len(todo)}", file=sys.stderr, flush=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


def _fuse(*lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return HybridRetriever._rrf_fuse(*lists, final_k=RETRIEVAL_FINAL_K)


def _configs(retriever: HybridRetriever, question: str, extra: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    k = RETRIEVAL_TOP_K
    vec_q = retriever._vector_search(question, top_k=k)
    bm_q = retriever._bm25_search(question, top_k=k)
    # if the call was blocked, use the question itself, so that item acts like the baseline
    vec_h = retriever._vector_search(extra["hyde"] or question, top_k=k)

    multi_lists: list[list[dict[str, Any]]] = [vec_q, bm_q]
    for rw in extra["rewrites"]:
        multi_lists.append(retriever._vector_search(rw, top_k=k))
        multi_lists.append(retriever._bm25_search(rw, top_k=k))

    return {
        "baseline (raw question)": _fuse(vec_q, bm_q),
        "HyDE, passage replaces question for vector search": _fuse(vec_h, bm_q),
        "HyDE added as a third list": _fuse(vec_q, bm_q, vec_h),
        f"multi-query ({N_REWRITES} rewrites + original)": _fuse(*multi_lists),
    }


def sweep() -> int:
    questions = json.load(io.open(QUESTIONS_PATH, encoding="utf-8"))["questions"]
    cache = _generate(questions)

    index_dir = INDEX_ROOT / "minilm"
    if not (index_dir / "chroma.sqlite3").exists():
        raise SystemExit("no minilm index under eval/indexes. run: python -m eval.sweep_embeddings --build minilm")
    retriever = HybridRetriever(persist_dir=index_dir, embedding_function=_embedding_function("minilm", mode="query"))

    rows: dict[str, dict[str, list[dict[str, float]]]] = {}
    per_question: dict[str, dict[str, float]] = {}
    for q in questions:
        any_hit = q.get("provenance") == "generated"
        for name, results in _configs(retriever, q["question"], cache[str(q["id"])]).items():
            s = _score_one(_distinct_ranked_docs(results), q["expected_sources"], any_hit)
            split = q.get("split", "dev")
            rows.setdefault(name, {"dev": [], "holdout": []})[split].append(s)
            per_question.setdefault(name, {})[q["id"]] = s["recall@5"]

    base = "baseline (raw question)"
    summary = []
    for name, by_split in rows.items():
        both = by_split["dev"] + by_split["holdout"]
        wins = sum(1 for i, v in per_question[name].items() if v > per_question[base][i])
        losses = sum(1 for i, v in per_question[name].items() if v < per_question[base][i])
        summary.append({
            "config": name,
            "n": {"dev": len(by_split["dev"]), "holdout": len(by_split["holdout"])},
            "dev": _agg(by_split["dev"]),
            "holdout": _agg(by_split["holdout"]),
            "all": _agg(both),
            "recall5_wins_vs_baseline": wins,
            "recall5_losses_vs_baseline": losses,
        })
    blocked = {
        "hyde": sorted(k for k, v in cache.items() if v["hyde"] is None),
        "multi_query": sorted(k for k, v in cache.items() if not v["rewrites"]),
    }
    RESULT_JSON.write_text(json.dumps({"blocked_question_ids": blocked, "rows": summary}, indent=1), encoding="utf-8")
    _write_md(summary, blocked)
    print(f"questions where Gemini's safety filter blocked the call: HyDE {blocked['hyde']}, multi-query {blocked['multi_query']}")
    for r in summary:
        print(f"{r['config']:<52} dev R@5 {r['dev']['recall@5']:.3f}  holdout R@5 {r['holdout']['recall@5']:.3f}  "
              f"(+{r['recall5_wins_vs_baseline']}/-{r['recall5_losses_vs_baseline']} questions)")
    return 0


def _write_md(summary: list[dict[str, Any]], blocked: dict[str, list[str]]) -> None:
    lines = [
        "# Query rewriting before retrieval: HyDE and multi-query",
        "",
        "Retrieval only, no judge. MiniLM index, hybrid retrieval (vector + BM25, RRF), top_k 10, final_k 5,",
        "no reranker, 100 questions (70 dev, 30 holdout). The LLM for the rewrites is Gemini flash;",
        "its outputs are cached in `eval/query_transform_cache.json`, so this reproduces exactly.",
        "Decide on dev and confirm on holdout, as in the embedding sweep.",
        "",
        "| config | split | R@1 | R@3 | R@5 | MRR | nDCG@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        for split in ("dev", "holdout", "all"):
            m = r[split]
            lines.append(
                f"| {r['config']} | {split} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | "
                f"{m['recall@5']:.3f} | {m['mrr']:.3f} | {m['ndcg@5']:.3f} |"
            )
    lines += [
        "",
        "Gemini's safety filter blocked a few harmless requests (no output returned). For those questions the",
        f"raw question is used instead, so they behave like the baseline. HyDE blocked on ids {blocked['hyde'] or 'none'};",
        f"multi-query blocked on ids {blocked['multi_query'] or 'none'}.",
    ]
    lines += ["", "Questions whose recall@5 changed against the baseline (out of 100):", "",
              "| config | better | worse |", "|---|---|---|"]
    for r in summary[1:]:
        lines.append(f"| {r['config']} | {r['recall5_wins_vs_baseline']} | {r['recall5_losses_vs_baseline']} |")
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(sweep())
