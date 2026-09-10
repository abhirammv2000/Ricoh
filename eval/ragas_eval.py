"""Cross-check our LLM judge's groundedness against RAGAS faithfulness.

This project's whole claim is honest measurement, and the groundedness number
rests on a single hand-rolled judge. eval/label_for_kappa.py checks it against a
human and against a second Claude model; this adds a third angle: RAGAS, a
widely used third-party RAG-eval library, computing faithfulness (its analog of
groundedness) with its own prompts and its own decomposition of the answer into
claims.

Runs in a SEPARATE virtualenv. RAGAS 0.4.x pins langchain/langgraph to 1.x,
which collides with this project's langgraph 0.2.74, so the two cannot share an
env. This script therefore imports no ``src`` module: it reads the JSONL that
eval/ragas_export.py wrote from the project env, and writes results back.

    # in the project env:
    python -m eval.ragas_export --n 12

    # in a throwaway env with RAGAS:
    python -m venv .venv-ragas
    .venv-ragas/Scripts/pip install "ragas==0.4.3" langchain-anthropic
    ANTHROPIC_API_KEY=... .venv-ragas/Scripts/python eval/ragas_eval.py

Faithfulness needs only an LLM, no embeddings. It runs on Sonnet (cheap; this is
a spot check, not the headline judge), and the per-question comparison against
our Opus judge is written to eval/ragas_results.{json,md}.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "eval" / "ragas_input.jsonl"
RESULT_JSON = PROJECT_ROOT / "eval" / "ragas_results.json"
RESULT_MD = PROJECT_ROOT / "eval" / "ragas_results.md"

# A score at or above this is "acceptable"; matches label_for_kappa.py so the
# two calibration checks binarise the same way.
BINARY_THRESHOLD = 0.8

# Sonnet, not the Opus judge from the main harness. This is a consistency probe,
# and running RAGAS on a different model than our judge is the point.
RAGAS_MODEL = os.getenv("RAGAS_MODEL", "claude-sonnet-4-6")


def _kappa(a: list[int], b: list[int]) -> dict[str, float]:
    """Cohen's kappa for two binary label lists. Same formula as label_for_kappa."""
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    kappa = (observed - chance) / (1 - chance) if chance < 1 else float("nan")
    return {
        "n": n,
        "raw_agreement": round(observed, 3),
        "chance_agreement": round(chance, 3),
        "cohens_kappa": round(kappa, 3) if kappa == kappa else None,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return round(cov / (sx * sy), 3)


def _load_rows() -> list[dict]:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"No input at {INPUT_PATH}. Run `python -m eval.ragas_export` in the "
            "project env first."
        )
    return [json.loads(line) for line in INPUT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_ragas(rows: list[dict]) -> list[float]:
    """Return RAGAS faithfulness per row, in the same order (NaN if unscorable)."""
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness
    from ragas.run_config import RunConfig

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")

    llm = LangchainLLMWrapper(
        ChatAnthropic(model=RAGAS_MODEL, temperature=0, max_tokens=4096, timeout=90)
    )
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=list(r["contexts"]),
            )
            for r in rows
        ]
    )
    result = evaluate(
        dataset,
        metrics=[Faithfulness()],
        llm=llm,
        run_config=RunConfig(timeout=180, max_retries=3, max_workers=4),
        raise_exceptions=True,
        show_progress=True,
    )
    return [float(v) for v in result.to_pandas()["faithfulness"].tolist()]


def main() -> int:
    rows = _load_rows()
    faithfulness = _run_ragas(rows)

    per_question = []
    ragas_bin, judge_bin = [], []
    ragas_scores, judge_scores = [], []
    disagreements = []
    unscored = []
    for row, f in zip(rows, faithfulness):
        if f != f:  # NaN: RAGAS could not extract claims from the answer
            unscored.append(row["id"])
            per_question.append(
                {"id": row["id"], "question": row["question"],
                 "ragas_faithfulness": None,
                 "judge_groundedness": round(float(row["judge_groundedness"]), 3)}
            )
            continue
        g = float(row["judge_groundedness"])
        rb = 1 if f >= BINARY_THRESHOLD else 0
        jb = 1 if g >= BINARY_THRESHOLD else 0
        ragas_bin.append(rb)
        judge_bin.append(jb)
        ragas_scores.append(f)
        judge_scores.append(g)
        entry = {
            "id": row["id"],
            "question": row["question"],
            "ragas_faithfulness": round(f, 3),
            "judge_groundedness": round(g, 3),
        }
        per_question.append(entry)
        if rb != jb:
            disagreements.append(entry)

    if not ragas_scores:
        raise SystemExit("RAGAS scored no rows (all NaN). Nothing to compare.")

    results = {
        "source": INPUT_PATH.name,
        "ragas_model": RAGAS_MODEL,
        "ragas_version": _ragas_version(),
        "threshold": BINARY_THRESHOLD,
        "n": len(rows),
        "n_scored": len(ragas_scores),
        "unscored_ids": unscored,
        "mean_ragas_faithfulness": round(sum(ragas_scores) / len(ragas_scores), 3),
        "mean_judge_groundedness": round(sum(judge_scores) / len(judge_scores), 3),
        "pearson_r": _pearson(ragas_scores, judge_scores),
        "binary_agreement": _kappa(ragas_bin, judge_bin),
        "disagreements": disagreements,
        "per_question": per_question,
    }
    RESULT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_md(results)
    print(f"\nwrote {RESULT_JSON}\nwrote {RESULT_MD}")
    print(
        f"mean faithfulness {results['mean_ragas_faithfulness']} vs "
        f"mean groundedness {results['mean_judge_groundedness']}, "
        f"kappa {results['binary_agreement']['cohens_kappa']}, "
        f"{len(disagreements)} disagreement(s)"
    )
    return 0


def _ragas_version() -> str:
    try:
        import ragas

        return getattr(ragas, "__version__", "unknown")
    except Exception:
        return "unknown"


def _write_md(r: dict) -> None:
    k = r["binary_agreement"]
    lines = [
        "# RAGAS cross-check: faithfulness vs our groundedness judge",
        "",
        f"RAGAS {r['ragas_version']} faithfulness on `{r['ragas_model']}` against the "
        f"harness's `claude-opus-5` groundedness score, on {r['n_scored']} of {r['n']} "
        f"questions sampled from `{r['source']}`"
        + (f" ({len(r['unscored_ids'])} unscored by RAGAS)." if r["unscored_ids"] else "."),
        "",
        "This is a consistency probe, not validation: RAGAS decomposes the answer",
        "into claims and checks each against the contexts, with its own prompts, so",
        "agreement is evidence our judge is not idiosyncratic. It is still an",
        "LLM grading an LLM, so it cannot catch a bias the models share. Only the",
        "human labels in `label_for_kappa.py` close that gap.",
        "",
        "| | mean |",
        "|---|---|",
        f"| RAGAS faithfulness | {r['mean_ragas_faithfulness']:.3f} |",
        f"| Our groundedness judge | {r['mean_judge_groundedness']:.3f} |",
        "",
        f"- Pearson r: {r['pearson_r']}",
        f"- Binary agreement at >= {r['threshold']}: {k['raw_agreement']:.0%} raw, "
        f"chance {k['chance_agreement']:.0%}, Cohen's kappa {k['cohens_kappa']}",
        "",
        "With n=12 and both raters scoring almost everything above the threshold,",
        "chance agreement is near the raw number, so the kappa carries little",
        "information either way. Read the means and the disagreements, not the kappa.",
        "",
    ]
    if r["disagreements"]:
        lines.append(f"## Disagreements ({len(r['disagreements'])})")
        lines.append("")
        lines.append("| Q | RAGAS faithfulness | our groundedness | question |")
        lines.append("|---|---|---|---|")
        for d in r["disagreements"]:
            lines.append(
                f"| {d['id']} | {d['ragas_faithfulness']} | {d['judge_groundedness']} "
                f"| {d['question'][:70]} |"
            )
    else:
        lines.append("No binary disagreements: both rate every sampled answer the same side of the threshold.")
    lines.append("")
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def rewrite_md() -> int:
    """Regenerate the .md from an existing results .json, no API calls."""
    if not RESULT_JSON.exists():
        raise SystemExit(f"No {RESULT_JSON} to rewrite from.")
    _write_md(json.loads(RESULT_JSON.read_text(encoding="utf-8")))
    print(f"wrote {RESULT_MD}")
    return 0


if __name__ == "__main__":
    import sys

    if "--md-only" in sys.argv:
        raise SystemExit(rewrite_md())
    raise SystemExit(main())
