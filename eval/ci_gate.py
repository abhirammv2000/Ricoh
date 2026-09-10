"""Retrieval regression gate for CI.

The unit suite mocks the LLM and never touches a real index, so a change that
quietly breaks hybrid retrieval, an RRF bug, a fusion off-by-one, a chunk
metadata regression at ingest, would pass every test and still ship. This is
the check that would catch it.

It runs real hybrid retrieval against the committed ``demo_index/`` (46
documents, no API key, no model download beyond ChromaDB's bundled MiniLM) on
the ten seed questions in ``eval/ground_truth.json``, and compares retriever
recall@1/3/5 against a committed baseline. A drop fails the build.

What this is and is not. ``demo_index`` is a small, curated slice built around
these questions, so recall here is near-ceiling by construction. That makes it
a *smoke* gate: it proves the retrieval path still works end to end and did not
regress, not that retrieval quality on the full 733-document corpus is good.
The full-corpus numbers come from the paid harness (README section 7) and
cannot run in CI without the source PDFs.

    python -m eval.ci_gate            # check against eval/ci_baseline.json
    python -m eval.ci_gate --update   # rewrite the baseline (do this deliberately)
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.config import PROJECT_ROOT
from src.retriever import HybridRetriever

DEMO_INDEX: Path = PROJECT_ROOT / "demo_index"
GROUND_TRUTH: Path = PROJECT_ROOT / "eval" / "ground_truth.json"
BASELINE: Path = PROJECT_ROOT / "eval" / "ci_baseline.json"

DEPTHS: tuple[int, ...] = (1, 3, 5)
TOP_K: int = 10
FINAL_K: int = 5

# How far recall may fall below the baseline before the gate fails. Retrieval is
# deterministic on a fixed index, so this only absorbs a legitimate reindex of
# demo_index, not run-to-run noise. A real improvement should be committed with
# --update rather than tolerated.
TOLERANCE: float = 0.001


@contextmanager
def _readonly_copy(index_dir: Path) -> Iterator[Path]:
    """Yield a throwaway copy of the index.

    Opening a ChromaDB store for querying rewrites some of its HNSW files, so
    pointing the retriever straight at the committed ``demo_index/`` would leave
    the working tree dirty every time the gate runs. Querying a copy keeps the
    repo untouched.
    """
    tmp = Path(tempfile.mkdtemp(prefix="citera-ci-gate-"))
    try:
        dest = tmp / index_dir.name
        shutil.copytree(index_dir, dest)
        yield dest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _ranked_docs(retriever: HybridRetriever, question: str) -> list[str]:
    """Distinct source documents retrieval returns for one question, in order."""
    results = retriever.retrieve(question, top_k=TOP_K, final_k=FINAL_K)
    ordered: list[str] = []
    for r in results:
        doc = r.get("source_document")
        if doc and doc not in ordered:
            ordered.append(doc)
    return ordered


def measure(index_dir: Path = DEMO_INDEX) -> dict[str, Any]:
    """Retriever recall@1/3/5 over the answerable seed questions."""
    if not (index_dir / "chroma.sqlite3").exists():
        raise SystemExit(f"{index_dir} does not look like a ChromaDB index")

    questions = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["questions"]
    answerable = [q for q in questions if q.get("expected_behavior") == "answer"]

    per_question: list[dict[str, Any]] = []
    sums = {d: 0.0 for d in DEPTHS}
    with _readonly_copy(index_dir) as working:
        retriever = HybridRetriever(persist_dir=working)
        if retriever.index_size == 0 or not retriever.bm25_ready:
            raise SystemExit(
                f"{index_dir} is not a usable index "
                f"(vectors={retriever.index_size}, bm25={retriever.bm25_ready})"
            )
        for q in answerable:
            expected = q["expected_sources"]
            docs = _ranked_docs(retriever, q["question"])
            scores = {
                d: sum(1 for e in expected if e in docs[:d]) / len(expected)
                for d in DEPTHS
            }
            for d in DEPTHS:
                sums[d] += scores[d]
            per_question.append(
                {"id": q["id"], "recall": {f"@{d}": round(scores[d], 4) for d in DEPTHS}}
            )

    n = len(answerable)
    return {
        "index": index_dir.name,
        "questions": n,
        "recall": {f"@{d}": round(sums[d] / n, 4) for d in DEPTHS},
        "per_question": per_question,
    }


def manifest_drift() -> list[str]:
    """Ways the committed demo_index no longer matches what it should serve.

    demo_index is a binary artifact built by src/build_demo_index.py from the
    corpus. CI has neither the corpus nor a way to rebuild it, so this checks
    the cheap invariant instead: the documents the curated benchmark
    (eval/ground_truth.json) depends on must all be baked into the index. If a
    question gains a new expected source and the index is not rebuilt, the live
    demo silently cannot answer it, and this is what says so.
    """
    from src.build_demo_index import _referenced_docs

    problems: list[str] = []
    manifest_path = DEMO_INDEX / "manifest.json"
    if not manifest_path.exists():
        return [f"{manifest_path} is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    baked = set(manifest.get("benchmark_referenced", []))
    needed = _referenced_docs()
    missing = sorted(needed - baked)
    if missing:
        problems.append(
            f"ground_truth.json needs {missing} but they are not in the demo index"
        )

    counted = manifest.get("document_count")
    listed = len(manifest.get("benchmark_referenced", [])) + len(manifest.get("sampled", []))
    if isinstance(counted, int) and counted != listed:
        problems.append(f"manifest document_count is {counted} but it lists {listed} documents")
    return problems


def _load_baseline() -> dict[str, Any]:
    if not BASELINE.exists():
        raise SystemExit(
            f"No baseline at {BASELINE}. Create it with: python -m eval.ci_gate --update"
        )
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def regressions(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Recall depths that fell more than TOLERANCE below the baseline."""
    out: list[str] = []
    for depth in DEPTHS:
        key = f"@{depth}"
        now = current["recall"][key]
        was = baseline["recall"][key]
        if now < was - TOLERANCE:
            out.append(f"recall{key}: {was} -> {now}")
    return out


def check() -> int:
    drift = manifest_drift()
    if drift:
        print("demo_index is out of sync with the benchmark it serves:")
        for line in drift:
            print(f"  {line}")
        print("\nrebuild it: python -m src.build_demo_index  (needs the corpus in data/)")
        return 1

    current = measure()
    baseline = _load_baseline()

    print(f"index: {current['index']}  questions: {current['questions']}")
    for depth in DEPTHS:
        key = f"@{depth}"
        now = current["recall"][key]
        was = baseline["recall"][key]
        flag = "REGRESSION" if now < was - TOLERANCE else "ok"
        print(f"  recall{key}: {now:.4f}  (baseline {was:.4f})  {flag}")

    found = regressions(current, baseline)
    if found:
        print("\nretrieval regressed against the committed baseline:")
        for line in found:
            print(f"  {line}")
        print(
            "\nif this drop is expected (demo_index was rebuilt, a deliberate "
            "retrieval change), re-run with --update and commit the new baseline."
        )
        return 1

    print("\nno regression.")
    return 0


def update() -> int:
    current = measure()
    BASELINE.write_text(
        json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {BASELINE}")
    print(f"  recall: {current['recall']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval regression gate")
    parser.add_argument(
        "--update", action="store_true", help="rewrite eval/ci_baseline.json from the current index"
    )
    args = parser.parse_args()
    raise SystemExit(update() if args.update else check())
