"""Build a small index to ship with the live demo.

The container has no data/ folder (the 733 PDFs are about 223 MB and Ricoh's, so they aren't ours to
republish), and an app with no index looks healthy but refuses everything. So the demo ships a small
index. It holds every document the curated benchmark uses (so the README's questions work, including
the two that should be refused) plus a fixed sample from the generated benchmark. The demo answers
from far fewer documents than the published metrics, so DEMO_MODE=true makes the UI say so, and a
manifest records what was included.

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
    """The documents the curated benchmark uses."""
    docs: set[str] = set()
    if GROUND_TRUTH.exists():
        for q in json.loads(io.open(GROUND_TRUTH, encoding="utf-8").read())["questions"]:
            docs.update(q.get("expected_sources", []))
    return docs


def _sampled_docs(exclude: set[str], n: int, seed: int) -> set[str]:
    """A fixed-seed sample of other documents from the generated set."""
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
        print(f"{len(missing)} referenced document(s) not in {DATA_DIR}:")
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

    # start from scratch so a smaller subset doesn't keep documents from an earlier run
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    chunks = []
    for name in sorted(wanted):
        # extract_pages sets source_document and page_number, and chunk_pages passes them on
        chunks.extend(chunk_pages(extract_pages(DATA_DIR / name)))
    print(f"  {len(chunks)} chunks")

    # imported after DEMO_DIR exists, the retriever puts its BM25 files next to the chroma store
    from src.retriever import HybridRetriever

    retriever = HybridRetriever(persist_dir=DEMO_DIR, collection_name=CHROMA_COLLECTION_NAME)
    retriever.build_index(chunks)

    manifest = {
        "_description": (
            "Documents baked into the deployed demo image. The published "
            "metrics in README/ROADMAP were measured on the FULL 733-document "
            "corpus, not this subset, the demo shows the system working, it "
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
