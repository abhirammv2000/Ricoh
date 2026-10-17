"""A manifest for the retrieval index, and the checks that use it.

The index is two stores that have to agree: the Chroma vectors and the BM25 files. They are built from the
same chunks, with the same embedding model, and a service that serves from them has to know that they
still match. Four things go wrong without a record of how the index was built, and none of them raises:

* The embedding model changes. Chroma rejects a query vector of another size, but several models share a
  size (MiniLM, bge-small and e5-small are all 384), so the service keeps running and retrieval gets worse.
* A partial re-ingest. build_index upserts into Chroma but rebuilds BM25 from only the chunks it was given, so
  BM25 can cover a subset of what Chroma holds.
* A file is damaged or replaced. The BM25 files are pickles, and loading a pickle runs code, so a changed
  file has to be caught before it is loaded, not after.
* The chunking changes. Chunks of 500 words and chunks of 200 words are different corpora.

build_index writes index_manifest.json next to the index. verify_index compares it with the configuration
and with what is on disk. `python -m src.index_manifest write` creates one for an index built before this
existed, and `python -m src.index_manifest check` prints the result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_NAME = "index_manifest.json"
SCHEMA_VERSION = 1

# what chromadb uses when EMBEDDING_MODEL is empty. Recorded by name, so a change to a named model is seen
DEFAULT_EMBEDDING_NAME = "chromadb-default-all-MiniLM-L6-v2"


class IndexIntegrityError(RuntimeError):
    """An index file does not match its manifest. The file is not loaded."""


@dataclass(frozen=True)
class IndexManifest:
    schema_version: int
    built_at: str
    collection: str
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    document_count: int
    corpus_fingerprint: str
    bm25_index_sha256: str
    bm25_chunks_sha256: str

    @property
    def index_version(self) -> str:
        """A short id for what this index is made of. It is in every trace, so a change in answers can be
        matched to a change in the index. It does not change when the index is rebuilt from the same inputs."""
        basis = json.dumps(
            [self.embedding_model, self.embedding_dim, self.chunk_size, self.chunk_overlap, self.corpus_fingerprint]
        )
        return hashlib.sha256(basis.encode()).hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass
class IndexCheck:
    """The result of checking an index. ok is False if anything in problems would make answers wrong."""

    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: IndexManifest | None = None

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def index_version(self) -> str:
        return self.manifest.index_version if self.manifest else "unversioned"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_fingerprint(chunk_ids: list[str]) -> str:
    """One hash for the set of chunks. It changes when a document is added, removed or re-chunked."""
    digest = hashlib.sha256()
    for chunk_id in sorted(chunk_ids):
        digest.update(chunk_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def read_manifest(persist_dir: Path) -> IndexManifest | None:
    path = Path(persist_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return IndexManifest(**data)


def write_manifest(persist_dir: Path, manifest: IndexManifest) -> None:
    path = Path(persist_dir) / MANIFEST_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(manifest.to_json(), encoding="utf-8")
    tmp.replace(path)  # a reader never sees half a file


def make_manifest(
    persist_dir: Path,
    *,
    collection: str,
    embedding_model: str,
    embedding_dim: int,
    chunk_size: int,
    chunk_overlap: int,
    chunks: list[dict[str, Any]],
    bm25_index_path: Path,
    bm25_chunks_path: Path,
) -> IndexManifest:
    return IndexManifest(
        schema_version=SCHEMA_VERSION,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        collection=collection,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_count=len(chunks),
        document_count=len({c["source_document"] for c in chunks}),
        corpus_fingerprint=corpus_fingerprint([c["id"] for c in chunks]),
        bm25_index_sha256=sha256_file(bm25_index_path),
        bm25_chunks_sha256=sha256_file(bm25_chunks_path),
    )


def verify_files_before_loading(persist_dir: Path, bm25_index_path: Path, bm25_chunks_path: Path) -> None:
    """Raise IndexIntegrityError if a BM25 pickle differs from the manifest. Call it before unpickling.

    With no manifest there is nothing to compare with, and the files are loaded as before. A manifest that
    exists but cannot be read is treated as a failure, since that is what a damaged index looks like.
    """
    try:
        manifest = read_manifest(persist_dir)
    except (OSError, ValueError, TypeError) as exc:
        raise IndexIntegrityError(f"the index manifest could not be read: {exc}") from exc
    if manifest is None:
        return
    for path, expected in (
        (bm25_index_path, manifest.bm25_index_sha256),
        (bm25_chunks_path, manifest.bm25_chunks_sha256),
    ):
        if sha256_file(path) != expected:
            raise IndexIntegrityError(f"{path.name} does not match the index manifest, so it was not loaded")


def verify_index(
    persist_dir: Path,
    *,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    vector_count: int,
    bm25_count: int,
    embedding_dim: int | None,
    require_manifest: bool = True,
) -> IndexCheck:
    """Compare the index as it is now with its manifest and with the current configuration."""
    check = IndexCheck()
    if vector_count == 0:
        check.problems.append("the vector index is empty, so every question would be refused")
    if bm25_count == 0:
        check.problems.append("the BM25 index is empty or was not loaded")
    if vector_count and bm25_count and vector_count != bm25_count:
        check.problems.append(
            f"the vector index has {vector_count} chunks but the BM25 index has {bm25_count}. "
            "They were built from different chunks, for example by a partial re-ingest"
        )

    try:
        manifest = read_manifest(persist_dir)
    except (OSError, ValueError, TypeError) as exc:
        check.problems.append(f"the index manifest could not be read: {exc}")
        return check
    if manifest is None:
        message = (
            f"there is no {MANIFEST_NAME}, so the index cannot be checked against the configuration. "
            "Run: python -m src.index_manifest write"
        )
        (check.problems if require_manifest else check.warnings).append(message)
        return check
    check.manifest = manifest

    if manifest.schema_version != SCHEMA_VERSION:
        check.problems.append(f"the manifest is schema {manifest.schema_version}, this code reads {SCHEMA_VERSION}")
    if manifest.embedding_model != embedding_model:
        check.problems.append(
            f"the index was built with embedding model {manifest.embedding_model!r} but the service is "
            f"configured for {embedding_model!r}. Queries would be embedded differently from the documents"
        )
    if embedding_dim is not None and manifest.embedding_dim != embedding_dim:
        check.problems.append(
            f"the stored vectors have {embedding_dim} dimensions but the manifest says {manifest.embedding_dim}"
        )
    if (manifest.chunk_size, manifest.chunk_overlap) != (chunk_size, chunk_overlap):
        check.problems.append(
            f"the index was chunked at {manifest.chunk_size}/{manifest.chunk_overlap} words "
            f"but the configuration is {chunk_size}/{chunk_overlap}"
        )
    if manifest.chunk_count != vector_count:
        check.problems.append(
            f"the manifest records {manifest.chunk_count} chunks but the vector index holds {vector_count}"
        )
    if manifest.chunk_count != bm25_count:
        check.problems.append(
            f"the manifest records {manifest.chunk_count} chunks but the BM25 index holds {bm25_count}"
        )
    return check


# ---- command line: write a manifest for an existing index, or check one ----


def _open_existing(persist_dir: Path):
    """The retriever for an existing index, without writing anything."""
    from src.retriever import HybridRetriever

    return HybridRetriever(persist_dir=persist_dir)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.config import CHROMA_DIR

    parser = argparse.ArgumentParser(description="Write or check the manifest of a retrieval index")
    parser.add_argument("command", choices=["write", "check"])
    parser.add_argument("--persist-dir", default=str(CHROMA_DIR))
    args = parser.parse_args(argv)
    persist_dir = Path(args.persist_dir)

    retriever = _open_existing(persist_dir)
    if args.command == "write":
        manifest = retriever.write_manifest_for_existing_index()
        print(f"wrote {persist_dir / MANIFEST_NAME}: {manifest.chunk_count} chunks, "
              f"{manifest.document_count} documents, index_version {manifest.index_version}")
        return 0

    check = retriever.check_index()
    print(f"index_version: {check.index_version}")
    for message in check.warnings:
        print(f"warning: {message}")
    for message in check.problems:
        print(f"PROBLEM: {message}")
    print("ok" if check.ok else "NOT OK")
    return 0 if check.ok else 1


if __name__ == "__main__":
    sys.exit(main())
