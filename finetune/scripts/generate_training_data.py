"""Build supervised fine-tuning data by distilling Citera's synthesizer.

The goal is not to teach a small model Ricoh facts from scratch. It is to
teach it the specific skill Citera's synthesizer node performs: given
retrieved evidence and a question, write a grounded, cited answer, or refuse
in the exact machine-readable way when the evidence does not cover it. So
each training example is built the same way a real synthesizer call is built
in production: real retrieval over the real index, then Claude Sonnet with
Citera's actual SYNTHESIZER_PROMPT as the teacher.

Leakage: every question here is generated from a document Citera's own eval
suite (eval/generated_questions.json, ground_truth.json, multihop_questions
.json, multiturn_questions.json) never uses, computed in
data/available_documents.txt. Judging the fine-tuned model on Citera's
existing 100-question benchmark afterwards is then a fair, unseen-document
test, not a train/test overlap.

Lives inside the Ricoh repo (finetune/) rather than its own project, and
needs the retrieval index already built one level up, since this reuses that
index and that prompt directly rather than a copy that could drift.

    python -m finetune.scripts.generate_training_data --pilot   # 3 examples, ~$0.10
    python -m finetune.scripts.generate_training_data --n 350
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.generate_questions import GENERATOR_PROMPT, _sample_chunks  # noqa: E402
from src.agent import SYNTHESIZER_PROMPT, _format_evidence_block  # noqa: E402
from src.config import (  # noqa: E402
    BM25_CHUNKS_PATH,
    RETRIEVAL_FINAL_K,
    RETRIEVAL_TOP_K,
)
from src.instrumentation import invoke as instrumented_invoke  # noqa: E402
from src.instrumentation import record_run  # noqa: E402
from src.llm_factory import get_llm  # noqa: E402
from src.retriever import get_retriever  # noqa: E402

AVAILABLE_DOCS_PATH = FINETUNE_ROOT / "data" / "available_documents.txt"
OUT_PATH = FINETUNE_ROOT / "data" / "train_examples.jsonl"
MANIFEST_PATH = FINETUNE_ROOT / "data" / "train_manifest.json"

DEFAULT_SEED = 20260911


def _load_available_chunks() -> list[dict[str, Any]]:
    if not BM25_CHUNKS_PATH.exists():
        raise SystemExit(
            f"No chunk store at {BM25_CHUNKS_PATH}. Build Citera's index "
            "first (in the Ricoh repo): python -m src.retriever"
        )
    with open(BM25_CHUNKS_PATH, "rb") as f:
        chunks: list[dict[str, Any]] = pickle.load(f)

    available = {
        line.strip() for line in AVAILABLE_DOCS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    filtered = [c for c in chunks if c["source_document"] in available]
    if not filtered:
        raise SystemExit("No chunks matched data/available_documents.txt. Was it built correctly?")
    return filtered


def generate(n: int, seed: int, skip: int = 0) -> int:
    chunks = _load_available_chunks()
    sample = _sample_chunks(chunks, n, seed)
    # _sample_chunks shuffles with this seed regardless of n, so the first
    # `skip` entries here are exactly the ones a prior run with the same seed
    # already produced. Skipping them resumes a partial run without paying to
    # regenerate examples already sitting in train_examples.jsonl.
    if skip:
        sample = sample[skip:]
    print(f"Sampled {len(sample)} chunks from {len(chunks)} available (leakage-safe) chunks (skipped {skip})\n")

    question_llm = get_llm()  # default provider/model, same as Citera's own generator
    retriever = get_retriever()

    # Continue id numbering from what is already on disk, so a resumed run's
    # ids don't collide with the prior run's (both would otherwise start at 1).
    generated = sum(1 for _ in open(OUT_PATH, encoding="utf-8")) if OUT_PATH.exists() else 0
    start_id = generated
    rejected = 0
    failed = 0

    # Opened once and flushed after every example, not batched to the end.
    # A run this long (roughly a call a second, ~350 examples means over an
    # hour of sequential API calls) has real odds of a transient network or
    # rate-limit error partway through. Writing incrementally means a crash
    # loses only the one in-flight example, not every example, and every
    # dollar, spent before it.
    out_f = open(OUT_PATH, "a" if OUT_PATH.exists() else "w", encoding="utf-8")

    with record_run() as rec:
        try:
            for i, chunk in enumerate(sample, 1):
                try:
                    gen_prompt = GENERATOR_PROMPT.format(
                        source_document=chunk["source_document"],
                        chunk_text=chunk["text"][:6000],
                    )
                    raw = instrumented_invoke(question_llm, gen_prompt, stage="question_generator")
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        rejected += 1
                        print(f"  [{i}] PARSE FAIL - skipped")
                        continue
                    if not parsed.get("usable") or not parsed.get("question"):
                        rejected += 1
                        print(f"  [{i}] rejected by generator: {parsed.get('reason', '')[:70]}")
                        continue

                    question = parsed["question"]

                    # Real retrieval, same settings the agent uses in production, so
                    # the teacher sees exactly the evidence the fine-tuned model will
                    # see at serving time, not the source chunk handed to it directly.
                    results = retriever.retrieve(
                        query=question, top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K
                    )
                    evidence_block = _format_evidence_block(results)

                    synth_prompt = SYNTHESIZER_PROMPT.format(user_query=question, evidence_block=evidence_block)
                    answer = instrumented_invoke(question_llm, synth_prompt, stage="synthesizer")

                    generated += 1
                    example = {
                        "id": generated,
                        "question": question,
                        "key_facts": parsed.get("key_facts", [])[:3],
                        "source_document": chunk["source_document"],
                        "source_chunk_id": chunk["id"],
                        "retrieved_documents": sorted({r["source_document"] for r in results}),
                        "messages": [
                            {"role": "user", "content": synth_prompt},
                            {"role": "assistant", "content": answer},
                        ],
                    }
                    out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    out_f.flush()
                    print(f"  [{i}] {question[:78]}")
                except Exception as exc:  # a single bad call should not sink the whole batch
                    failed += 1
                    print(f"  [{i}] FAILED ({type(exc).__name__}: {str(exc)[:120]}) - skipped")
                    continue
        finally:
            out_f.close()

    new_this_run = generated - start_id
    print(f"\nGenerated {new_this_run} training examples this run ({generated} total on disk), "
          f"rejected {rejected}, failed {failed}")
    print(f"Generation cost this run: ${rec.total_cost_usd:.4f}")

    manifest = {
        "seed": seed,
        "requested": n,
        "skip": skip,
        "generated_this_run": new_this_run,
        "total_on_disk": generated,
        "rejected": rejected,
        "failed": failed,
        "cost_usd": rec.total_cost_usd,
        "note": "Appended to train_examples.jsonl; see that file's line count for the running total.",
    }
    prior = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["runs"] if MANIFEST_PATH.exists() else []
    MANIFEST_PATH.write_text(json.dumps({"runs": prior + [manifest]}, indent=2), encoding="utf-8")

    print(f"Wrote/appended {OUT_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate SFT distillation data from Citera's synthesizer")
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--pilot", action="store_true", help="generate 3 and stop, to inspect quality before spending more")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N sampled chunks (already generated in a prior run with this seed)")
    args = ap.parse_args()
    raise SystemExit(generate(3 if args.pilot else args.n, args.seed, skip=args.skip))
