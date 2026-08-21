"""eval/generate_questions.py - Build a larger, honestly-labelled eval set.

Why this exists
Every claim in this project currently rests on 10 questions.  At n=10 a single
question moves any mean by 10 points, which is wider than most effects worth
detecting, so the project's strongest result (removing the agentic pipeline)
sits on its weakest evidence base.  Growing the set is the highest-value work
remaining.

How the labels are obtained without hand-labelling everything
Each question is generated from a specific chunk, so that chunk's document
is the expected source *by construction*.  That gives reliable retrieval labels
for free.  It does not give free *answer* labels, those still need the judge,
and the judge still needs human calibration (see eval/label_for_kappa.py).

The failure mode this guards against
Generated benchmarks are usually too easy.  If the question reuses distinctive
wording from its source chunk, BM25 matches it trivially and retrieval recall
looks perfect for reasons that have nothing to do with the system being good.
Two defences:

  1. The generator is instructed to paraphrase in a technician's own words and
     to avoid reusing rare exact strings from the chunk.
  2. ``--audit`` measures lexical overlap between each question and its source
     chunk, and reports retrieval recall on the generated set. If recall comes
     out at a suspiciously perfect 1.00 with high overlap, the set is too easy
     and the numbers it produces are not trustworthy.

A second guard: single-source assumption
Labelling the source chunk's document as the *only* expected source assumes no
other document answers the question equally well.  On a corpus with overlapping
help topics that is not always true, and a wrongly-narrow label penalises
correct retrieval, exactly the bug found in the original Q2 entry.  The
generator is therefore asked to reject questions it cannot make specific to one
document, and ``--audit`` flags any question whose top retrieval hit is a
*different* document for human review rather than silently scoring it wrong.

Held-out split
Questions are split into ``dev`` and ``holdout``.  Tune on dev only.  The
holdout exists so that a final number can be reported that no decision was
fitted to.

Usage:
    python -m eval.generate_questions --pilot          # 8 questions, inspect quality first
    python -m eval.generate_questions --n 100          # full run
    python -m eval.generate_questions --audit          # audit an existing set, no API cost
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import random
import re
from pathlib import Path
from typing import Any

from src.config import (
    BM25_CHUNKS_PATH,
    PROJECT_ROOT,
    RETRIEVAL_FINAL_K,
    RETRIEVAL_TOP_K,
)
from src.instrumentation import invoke as instrumented_invoke
from src.instrumentation import record_run
from src.llm_factory import get_llm

OUT_PATH: Path = PROJECT_ROOT / "eval" / "generated_questions.json"

# Chunks shorter than this are navigation stubs or boilerplate fragments and
# cannot support a real question.
MIN_CHUNK_WORDS = 120

# Fraction held out. Never tune against it.
HOLDOUT_FRACTION = 0.3

GENERATOR_PROMPT = """\
You are building an evaluation set for a technical-support retrieval system \
over RICOH ProcessDirector documentation.

Below is ONE documentation excerpt. Write a question that a real support \
technician would ask, which this excerpt answers.

Hard requirements:
1. The question must be answerable using ONLY this excerpt.
2. The question must be SPECIFIC to this excerpt, not a generic question that \
a dozen other documentation pages could also answer. If you cannot write such \
a question, set "usable" to false and explain why.
3. Phrase it the way a technician would actually type it. Do NOT copy \
distinctive phrases verbatim from the excerpt, paraphrase. A question that \
reuses rare exact strings makes keyword retrieval trivially easy and ruins the \
benchmark.
4. Do not reference "the excerpt", "the document", or "the text", the user \
asking the question cannot see it.
5. key_facts: 1-3 short factual strings the correct answer must convey.

Excerpt (from {source_document}):
\"\"\"{chunk_text}\"\"\"

Respond with ONLY a valid JSON object, no markdown fences:
{{"usable": true, "question": "...", "key_facts": ["...", "..."], "reason": ""}}
"""


def _load_chunks() -> list[dict[str, Any]]:
    if not BM25_CHUNKS_PATH.exists():
        raise SystemExit(
            f"No chunk store at {BM25_CHUNKS_PATH}. Build the index first: "
            "python -m src.retriever"
        )
    with open(BM25_CHUNKS_PATH, "rb") as f:
        return pickle.load(f)


def _sample_chunks(chunks: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """One chunk per document, sampled reproducibly.

    Sampling per-document rather than per-chunk stops multi-chunk documents
    from dominating the set and keeps the benchmark spread across the corpus.
    """
    usable = [c for c in chunks if len(c["text"].split()) >= MIN_CHUNK_WORDS]
    by_doc: dict[str, dict[str, Any]] = {}
    rng = random.Random(seed)
    for c in sorted(usable, key=lambda c: c["id"]):
        by_doc.setdefault(c["source_document"], c)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    return [by_doc[d] for d in docs[:n]]


_STOP = set(
    "the a an of to and or is are for in on with you your it this that if how "
    "what when where can do does i my be as at by from not no yes will".split()
)


def _content_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOP and len(w) > 2
    }


def generate(n: int, seed: int) -> int:
    chunks = _load_chunks()
    sample = _sample_chunks(chunks, n, seed)
    print(f"Sampled {len(sample)} chunks (1 per document, >= {MIN_CHUNK_WORDS} words)\n")

    llm = get_llm()
    questions: list[dict[str, Any]] = []
    rejected = 0

    with record_run() as rec:
        for i, chunk in enumerate(sample, 1):
            prompt = GENERATOR_PROMPT.format(
                source_document=chunk["source_document"],
                chunk_text=chunk["text"][:6000],
            )
            raw = instrumented_invoke(llm, prompt, stage="question_generator")
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
                print(f"  [{i}] rejected by generator: {parsed.get('reason','')[:70]}")
                continue

            questions.append(
                {
                    "question": parsed["question"],
                    "expected_behavior": "answer",
                    "key_facts": parsed.get("key_facts", [])[:3],
                    # Known by construction: the question was written FROM this doc.
                    "expected_sources": [chunk["source_document"]],
                    "source_chunk_id": chunk["id"],
                    "provenance": "generated",
                }
            )
            print(f"  [{i}] {parsed['question'][:78]}")

    print(f"\nGenerated {len(questions)}, rejected {rejected}")
    print(f"Generation cost: ${rec.total_cost_usd:.4f}")

    rng = random.Random(seed + 1)
    rng.shuffle(questions)
    n_holdout = int(len(questions) * HOLDOUT_FRACTION)
    for idx, q in enumerate(questions):
        q["id"] = idx + 1
        q["split"] = "holdout" if idx < n_holdout else "dev"

    payload = {
        "_description": (
            "Auto-generated evaluation questions. expected_sources is known by "
            "construction (each question was written from a chunk of that "
            "document). Tune on split=='dev' only; 'holdout' exists so a final "
            "number can be reported that no decision was fitted to. Run "
            "`python -m eval.generate_questions --audit` before trusting these."
        ),
        "seed": seed,
        "generation_cost_usd": rec.total_cost_usd,
        "counts": {
            "total": len(questions),
            "dev": len(questions) - n_holdout,
            "holdout": n_holdout,
        },
        "questions": questions,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_PATH}  (dev={payload['counts']['dev']}, holdout={n_holdout})")
    print("\nNext: python -m eval.generate_questions --audit")
    return 0


def audit() -> int:
    """Check the generated set is actually hard enough to be informative.

    Costs nothing: retrieval is deterministic and local.
    """
    if not OUT_PATH.exists():
        raise SystemExit(f"No generated set at {OUT_PATH}. Run without --audit first.")

    from src.retriever import get_retriever

    payload = json.loads(io.open(OUT_PATH, encoding="utf-8").read())
    questions = payload["questions"]
    chunks = {c["id"]: c for c in _load_chunks()}
    retriever = get_retriever()

    overlaps: list[float] = []
    recalls: list[float] = []
    at1 = 0
    mismatched: list[dict[str, Any]] = []

    for q in questions:
        chunk = chunks.get(q.get("source_chunk_id"))
        if chunk:
            qw = _content_words(q["question"])
            cw = _content_words(chunk["text"])
            overlaps.append(len(qw & cw) / len(qw) if qw else 0.0)

        results = retriever.retrieve(
            query=q["question"], top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K
        )
        docs: list[str] = []
        for r in results:
            d = r["source_document"]
            if d not in docs:
                docs.append(d)
        # ANY-HIT semantics, deliberately.
        #
        # For this generated set, multiple expected_sources are *alternatives*:
        # the question was written from one document, and label expansion added
        # others that an adjudicator confirmed also answer it. Retrieving any
        # one of them is a correct retrieval.
        #
        # Note this differs from the curated set in eval/ground_truth.json,
        # where multiple expected_sources mean several documents are jointly
        # relevant and recall is the fraction found. Same field name, different
        # meaning, which is why the two sets are audited separately rather
        # than concatenated.
        expected = set(q["expected_sources"])
        recalls.append(1.0 if expected & set(docs) else 0.0)
        if docs and docs[0] in expected:
            at1 += 1
        elif docs:
            mismatched.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "expected": ", ".join(sorted(expected)),
                    "top": docs[0],
                }
            )

    n = len(questions)
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    print(f"Questions: {n}  (dev={payload['counts']['dev']}, holdout={payload['counts']['holdout']})")
    print(f"\nDifficulty audit")
    print(f"  mean lexical overlap with source chunk : {mean_overlap:.1%}")
    print(f"  retrieval recall@{RETRIEVAL_FINAL_K}                    : {sum(recalls)/n:.3f}")
    print(f"  retrieval recall@1                     : {at1/n:.3f}")

    print("\nReading this:")
    if mean_overlap > 0.60:
        print("  HIGH overlap - questions echo their source chunk. Keyword")
        print("    retrieval will find them trivially and recall is inflated.")
    else:
        print("  Overlap is moderate; questions are paraphrased rather than copied.")
    if sum(recalls) / n >= 0.99:
        print("  recall@k is ~perfect. Either retrieval genuinely is this")
        print("    strong on this corpus (consistent with the 10-question set),")
        print("    or the set is too easy to discriminate between systems.")
    else:
        print("  recall@k below ceiling - the set can distinguish systems.")

    if mismatched:
        print(f"\n{len(mismatched)} question(s) whose TOP hit is a different document.")
        print("These need human review: the label may be wrongly narrow (another")
        print("document may answer equally well) rather than retrieval being wrong.")
        for m in mismatched[:10]:
            print(f"  Q{m['id']}: expected {m['expected']} | top {m['top']}")
            print(f"        {m['question'][:88]}")
    return 0


ADJUDICATOR_PROMPT = """\
You are auditing labels for a retrieval benchmark.

A question was written from document A, so document A is known to answer it. \
The retrieval system instead ranked document B first. The question is whether \
that is a retrieval error, or whether document B ALSO answers the question, \
in which case the benchmark label is too narrow and B should be accepted too.

Question:
\"\"\"{question}\"\"\"

Document B ({doc_b}) content:
\"\"\"{text_b}\"\"\"

Does document B contain enough information to correctly and specifically \
answer that question on its own?

Be strict. "Mentions the same topic" is NOT enough, it must actually answer \
the question. If it only partially answers, say false.

Respond with ONLY a valid JSON object, no markdown fences:
{{"also_answers": true, "why": "<one short sentence>"}}
"""


def expand_labels() -> int:
    """Accept additional correct source documents, verified by adjudication.

    The problem this solves
    Each generated question is labelled with the single document it was written
    from.  On a corpus of 733 overlapping help topics, other documents often
    answer the same question equally well, so that single-source label marks
    correct retrieval as wrong and understates recall.  This is the same bug
    that once made a correct refusal (Q2) look like a retrieval miss.

    Rather than assume the label is wrong OR that retrieval is wrong, this asks
    a strict adjudicator whether the competing document genuinely answers the
    question, and only then widens the label.  Every expansion is recorded with
    its justification so the change is auditable rather than a silent fix that
    happens to raise the score.

    Only questions whose TOP hit differs from the label are adjudicated, which
    bounds the cost to the cases that actually affect the metric.
    """
    from src.retriever import get_retriever

    payload = json.loads(io.open(OUT_PATH, encoding="utf-8").read())
    questions = payload["questions"]
    chunks = _load_chunks()
    by_doc: dict[str, str] = {}
    for c in chunks:
        by_doc.setdefault(c["source_document"], "")
        if len(by_doc[c["source_document"]]) < 6000:
            by_doc[c["source_document"]] += c["text"] + "\n"

    retriever = get_retriever()
    llm = get_llm()
    expanded = 0

    with record_run() as rec:
        for q in questions:
            results = retriever.retrieve(
                query=q["question"], top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K
            )
            docs: list[str] = []
            for r in results:
                if r["source_document"] not in docs:
                    docs.append(r["source_document"])
            if not docs or docs[0] in q["expected_sources"]:
                continue

            candidate = docs[0]
            raw = instrumented_invoke(
                llm,
                ADJUDICATOR_PROMPT.format(
                    question=q["question"],
                    doc_b=candidate,
                    text_b=by_doc.get(candidate, "")[:6000],
                ),
                stage="label_adjudicator",
            )
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            try:
                verdict = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if verdict.get("also_answers"):
                q["expected_sources"].append(candidate)
                q.setdefault("label_expansions", []).append(
                    {"added": candidate, "why": verdict.get("why", "")}
                )
                expanded += 1
                print(f"  Q{q['id']}: + {candidate}, {verdict.get('why','')[:70]}")

    payload["label_expansion"] = {
        "expanded_questions": expanded,
        "cost_usd": rec.total_cost_usd,
        "method": (
            "For questions whose top retrieval hit differed from the generated "
            "label, a strict adjudicator judged whether that document also "
            "answers the question. Only confirmed cases were added."
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nExpanded {expanded} labels. Cost ${rec.total_cost_usd:.4f}")
    print("Re-run --audit to see corrected recall.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate and audit eval questions")
    ap.add_argument(
        "--expand-labels",
        action="store_true",
        help="adjudicate whether competing top-hit documents also answer the question",
    )
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--pilot", action="store_true", help="generate 8 and stop, to inspect quality")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--audit", action="store_true", help="audit an existing set (no API cost)")
    args = ap.parse_args()
    if args.expand_labels:
        raise SystemExit(expand_labels())
    if args.audit:
        raise SystemExit(audit())
    raise SystemExit(generate(8 if args.pilot else args.n, args.seed))
