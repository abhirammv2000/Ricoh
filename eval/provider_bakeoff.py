"""Cross-provider bakeoff: same pipeline, same judge, different agent model.

The system runs on claude-sonnet-4-6. The README's section 6 says GPT-4o was
"considered"; this measures what actually changes if the synthesizer runs on a
cheaper non-Anthropic model instead.

Only the synthesizer model varies. Retrieval is identical and deterministic
(run once per question, shared across providers), the config is A
(retrieve -> synthesize, no planner), and the judge is always claude-opus-5, so
a difference is the model, not the harness.

Providers and default models (override with --models):
    anthropic    claude-sonnet-4-6   (the current production model, the baseline)
    openai       gpt-4o-mini
    google       gemini-3.6-flash    (via its OpenAI-compatible endpoint)
    self_hosted  citera-finetuned    (QLoRA-distilled Llama 3.1 8B, own vLLM
                                      server, see citera-finetune/. Cost shows
                                      as $0/query: real cost is GPU-hours, not
                                      per-token, so it is not comparable to the
                                      other rows' cost column as-is.)

    pip install -r requirements-providers.txt
    # set OPENAI_API_KEY and GEMINI_API_KEY in .env
    python -m eval.provider_bakeoff --n 20
    python -m eval.provider_bakeoff --providers anthropic openai --no-judge
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import JUDGE_MODEL, PROJECT_ROOT, RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K
from src.instrumentation import invoke as instrumented_invoke
from src.instrumentation import record_run
from src.llm_factory import _DEFAULT_MODELS, get_llm

GENERATED = PROJECT_ROOT / "eval" / "generated_questions.json"
RESULT_JSON = PROJECT_ROOT / "eval" / "provider_bakeoff.json"
RESULT_MD = PROJECT_ROOT / "eval" / "provider_bakeoff.md"

PROVIDER_MODELS = {
    "anthropic": _DEFAULT_MODELS["anthropic"],
    "openai": _DEFAULT_MODELS["openai"],
    "google": _DEFAULT_MODELS["google"],
    "self_hosted": _DEFAULT_MODELS["self_hosted"],
}


def _distinct_docs(evidence: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for e in evidence:
        d = e.get("source_document")
        if d and d not in out:
            out.append(d)
    return out


def run(providers: list[str], models: dict[str, str], n: int, seed: int, use_judge: bool) -> dict[str, Any]:
    from src.agent import SYNTHESIZER_PROMPT
    from src.eval_harness import _format_evidence_block, _judge
    from src.retriever import get_retriever

    gt = json.loads(io.open(GENERATED, encoding="utf-8").read())["questions"]
    dev = [q for q in gt if q.get("split") == "dev"]
    questions = random.Random(seed).sample(dev, min(n, len(dev)))
    retriever = get_retriever()

    # Retrieval once per question, shared across providers.
    shared: dict[int, dict[str, Any]] = {}
    for q in questions:
        ev = retriever.retrieve(q["question"], top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K)
        shared[q["id"]] = {
            "evidence": ev,
            "evidence_block": _format_evidence_block(ev),
            "retrieved_docs": _distinct_docs(ev),
        }

    per_provider: dict[str, Any] = {}
    for provider in providers:
        model = models[provider]
        print(f"\n=== {provider} / {model} ===")
        llm = get_llm(provider=provider, model=model)
        rows: list[dict[str, Any]] = []
        for q in questions:
            s = shared[q["id"]]
            prompt = SYNTHESIZER_PROMPT.format(
                user_query=q["question"], evidence_block=s["evidence_block"]
            )
            t0 = time.perf_counter()
            with record_run(query=q["question"]) as rec:
                answer = instrumented_invoke(llm, prompt, stage="synthesizer")
            latency = round(time.perf_counter() - t0, 2)

            refused = "information unavailable" in " ".join(answer.lower().split())
            expected = q.get("expected_sources", [])
            row: dict[str, Any] = {
                "id": q["id"],
                "answer": answer,
                "latency_seconds": latency,
                "cost_usd": rec.total_cost_usd,
                "input_tokens": rec.total_input_tokens,
                "output_tokens": rec.total_output_tokens,
                "system_refused": refused,
                "behavior_match": (q.get("expected_behavior", "answer") == "answer") != refused,
                # generated questions carry alternative sources: any hit is a hit.
                "evidence_recall": 1.0 if (set(expected) & set(s["retrieved_docs"])) else 0.0,
            }
            if use_judge and not answer.startswith("ERROR"):
                with record_run(persist=False) as jrec:
                    verdict = _judge(
                        q["question"], answer, s["evidence_block"],
                        q.get("key_facts", []), q.get("expected_behavior", "answer"),
                    )
                row["groundedness"] = round(verdict["groundedness"], 3)
                row["correctness"] = round(verdict["correctness"], 3)
                row["judge_cost_usd"] = jrec.total_cost_usd
            rows.append(row)
            print(f"  Q{q['id']:>3}: ${row['cost_usd']:.5f} {latency:>5}s "
                  f"grounded={row.get('groundedness')} correct={row.get('correctness')}")

        per_provider[provider] = _aggregate(provider, model, rows, use_judge)
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": JUDGE_MODEL if use_judge else None,
        "questions": [q["id"] for q in questions],
        "n": len(questions),
        "providers": per_provider,
    }


def _mean(vals: list[float | None]) -> float | None:
    nums = [v for v in vals if v is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def _aggregate(provider: str, model: str, rows: list[dict[str, Any]], use_judge: bool) -> dict[str, Any]:
    n = len(rows)
    out = {
        "model": model,
        "n": n,
        "mean_cost_per_query_usd": _mean([r["cost_usd"] for r in rows]),
        "mean_latency_seconds": _mean([r["latency_seconds"] for r in rows]),
        "mean_output_tokens": _mean([float(r["output_tokens"]) for r in rows]),
        "behavior_match_rate": _mean([float(r["behavior_match"]) for r in rows]),
        "evidence_recall": _mean([r["evidence_recall"] for r in rows]),
        "judge_cost_usd": round(sum(r.get("judge_cost_usd", 0.0) for r in rows), 5),
        "per_question": rows,
    }
    if use_judge:
        out["groundedness"] = _mean([r.get("groundedness") for r in rows])
        out["correctness"] = _mean([r.get("correctness") for r in rows])
    return out


def _write_md(report: dict[str, Any]) -> None:
    lines = [
        "# Cross-provider bakeoff",
        "",
        f"**Generated:** {report['generated']}  ",
        f"**Judge:** `{report['judge_model'] or 'disabled'}` (constant across providers)  ",
        f"**Questions:** {report['n']}, sampled from the generated dev split  ",
        "",
        "Config A (retrieve -> synthesize). Retrieval is shared and identical, so "
        "differences are the synthesizer model.",
        "",
        "| provider | model | cost/query | latency | out tok | grounded | correct | evid. recall | behaviour |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for prov, a in report["providers"].items():
        lines.append(
            f"| {prov} | `{a['model']}` | ${a['mean_cost_per_query_usd']:.5f} "
            f"| {a['mean_latency_seconds']}s | {int(a['mean_output_tokens'])} "
            f"| {a.get('groundedness', '-')} | {a.get('correctness', '-')} "
            f"| {a['evidence_recall']} | {a['behavior_match_rate']} |"
        )
    base = report["providers"].get("anthropic")
    if base and base.get("groundedness") is not None:
        lines += ["", "Baseline is `anthropic`. A cheaper model that holds groundedness and "
                  "correctness within the judge's ~0.10 noise floor would be a real "
                  "cost lever; one that drops either is not."]
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-provider synthesizer bakeoff")
    ap.add_argument("--providers", nargs="+", default=["anthropic", "openai", "google"],
                    choices=["anthropic", "openai", "google", "self_hosted"])
    ap.add_argument("--models", nargs="*", default=[],
                    help="provider=model overrides, e.g. openai=gpt-4o")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    models = dict(PROVIDER_MODELS)
    for pair in args.models:
        k, _, v = pair.partition("=")
        models[k] = v

    report = run(args.providers, models, args.n, args.seed, use_judge=not args.no_judge)
    RESULT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_md(report)
    print(f"\nwrote {RESULT_JSON}\nwrote {RESULT_MD}")
    for prov, a in report["providers"].items():
        print(f"  {prov:10} ${a['mean_cost_per_query_usd']:.5f}/q  "
              f"grounded {a.get('groundedness')}  correct {a.get('correctness')}")
