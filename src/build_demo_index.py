"""src/build_demo_index.py - Bake a small, shippable index for the live demo.

Why a subset rather than the full corpus
────────────────────────────────────────
The deployed container has no ``data/`` directory: the 733 source PDFs are
~223 MB and are RICOH's documentation, not ours to republish wholesale. But
an app that ships with no index is worse than useless — it looks healthy and
refuses every question, because the synthesizer correctly declines to answer
with no evidence.

So the demo ships a **small curated index**: large enough that the system
demonstrably works end to end, small enough to bake into the image and to
limit how much third-party documentation is republished.

How the subset is chosen
────────────────────────
Not at random. It includes:

1. Every document referenced by the curated benchmark
   (``eval/ground_truth.json``), so the demo can answer the questions the
   README talks about — including the two it should *refuse*.
2. A deterministic sample of documents from the generated benchmark, so the
   demo is not tuned to only the questions on show.

Honesty constraint
──────────────────
The live demo answers from ~N documents while the published metrics were
measured on all 733. Those are different systems, and conflating them would
overstate the demo. ``DEMO_MODE=true`` makes the UI say so, and this script
writes a manifest recording exactly what was included.

Usage:
    python -m src.build_demo_index                 # default subset
    python -m src.build_demo_index --extra 40      # widen the sample
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
from pathlib import Path

from src.config import CHROMA_COLLECTION_NAME, DATA_DIR, PROJECT_ROOT
from src.ingest import chunk_pages, extract_pages

DEMO_DIR: Path = PROJECT_ROOT / "demo_index"
GROUND_TRUTH: Path = PROJECT_ROOT / "eval" / "ground_truth.json"
GENERATED: Path = PROJECT_ROOT / "eval" / "generated_questions.json"


def _referenced_docs() -> set[str]:
    """Documents the curated benchmark depends on."""
    docs: set[str] = set()
    if GROUND_TRUTH.exists():
        for q in json.loads(io.open(GROUND_TRUTH, encoding="utf-8").read())["questions"]:
            docs.update(q.get("expected_sources", []))
    return docs


def _sampled_docs(exclude: set[str], n: int, seed: int) -> set[str]:
    """A deterministic spread of other documents from the generated set."""
    pool: list[str] = []
    if GENERATED.exists():
        for q in json.loads(io.open(GENERATED, encoding="utf-8").read())["questions"]:
            for d in q.get("expected_sources", []):
                if d not in exclude and d not in pool:
                    pool.append(d)
    pool.sort()
    random.Random(seed).shuffle(pool)
    return set(pool[:n])


def build(extra: int, seed: int) -> int:
    required = _referenced_docs()
    extras = _sampled_docs(required, extra, seed)
    wanted = required | extras

    present = {p.name for p in DATA_DIR.glob("*.pdf")}
    missing = wanted - present
    if missing:
        print(f"⚠ {len(missing)} referenced document(s) not in {DATA_DIR}:")
        for m in sorted(missing)[:10]:
            print(f"    {m}")
    wanted &= present
    if not wanted:
        raise SystemExit(
            f"No source PDFs found in {DATA_DIR}. The demo index is built from "
            "the real corpus; point RICOH_DATA_DIR at it and retry."
        )

    print(f"Building demo index from {len(wanted)} documents "
          f"({len(required & present)} benchmark-referenced, "
          f"{len(extras & present)} sampled)")

    # Rebuild from scratch so a shrunk subset never leaves stale documents
    # behind from a previous, larger run.
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    chunks = []
    for name in sorted(wanted):
        # extract_pages sets source_document/page_number, which chunk_pages
        # then carries onto every chunk — the provenance citations rely on.
        chunks.extend(chunk_pages(extract_pages(DATA_DIR / name)))
    print(f"  {len(chunks)} chunks")

    # Import after DEMO_DIR exists; retriever derives its BM25 paths from the
    # persist dir it is given, so the two halves stay consistent.
    from src.retriever import HybridRetriever

    retriever = HybridRetriever(persist_dir=DEMO_DIR, collection_name=CHROMA_COLLECTION_NAME)
    retriever.build_index(chunks)

    manifest = {
        "_description": (
            "Documents baked into the deployed demo image. The published "
            "metrics in README/ROADMAP were measured on the FULL 733-document "
            "corpus, not this subset — the demo shows the system working, it "
            "does not reproduce the benchmark."
        ),
        "document_count": len(wanted),
        "chunk_count": len(chunks),
        "benchmark_referenced": sorted(required & present),
        "sampled": sorted(extras & present),
        "seed": seed,
    }
    (DEMO_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    size = sum(f.stat().st_size for f in DEMO_DIR.rglob("*") if f.is_file())
    print(f"  index at {DEMO_DIR}  ({size / 1_048_576:.1f} MB, "
          f"{retriever.index_size} docs indexed, bm25={retriever.bm25_ready})")
    print("\nServe it with:  CHROMA_DIR=demo_index DEMO_MODE=true streamlit run app/main.py")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the shippable demo index")
    ap.add_argument("--extra", type=int, default=35, help="documents to sample beyond benchmark ones")
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()
    raise SystemExit(build(args.extra, args.seed))
