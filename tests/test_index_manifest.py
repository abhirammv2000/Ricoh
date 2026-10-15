"""The index manifest: how it catches an index that no longer matches how the service is configured.

Each test builds a small real index (Chroma and BM25) in a temporary folder with a stub embedding function,
so nothing is downloaded and nothing touches the committed demo index.
"""

from __future__ import annotations

import hashlib
import json
import pickle

import pytest

from src import index_manifest
from src.index_manifest import (
    IndexIntegrityError,
    corpus_fingerprint,
    read_manifest,
    verify_files_before_loading,
)
from src.retriever import HybridRetriever


class StubEmbedding:
    """A deterministic embedding with a name and a size, so a test can have two models of the same size."""

    def __init__(self, name: str, dim: int = 8) -> None:
        self._name = name
        self._dim = dim

    def name(self) -> str:
        return self._name

    def __call__(self, input):  # chroma looks for exactly this parameter name
        out = []
        for text in input:
            digest = hashlib.sha256(text.encode()).digest()
            out.append([digest[i] / 255.0 for i in range(self._dim)])
        return out


def make_chunks(n: int, source: str = "guide.pdf", start: int = 0) -> list[dict]:
    return [
        {
            "id": f"{source}-{i}",
            "text": f"chunk {i} about network settings and error code SC{500 + i}",
            "source_document": source,
            "page_number": 1 + i // 3,
            "chunk_index": i,
        }
        for i in range(start, start + n)
    ]


def build(path, n=6, name="model-a", dim=8) -> HybridRetriever:
    retriever = HybridRetriever(persist_dir=path, embedding_function=StubEmbedding(name, dim))
    retriever.build_index(make_chunks(n))
    return retriever


def reopen(path, name="model-a", dim=8) -> HybridRetriever:
    return HybridRetriever(persist_dir=path, embedding_function=StubEmbedding(name, dim))


def test_a_built_index_writes_a_manifest_and_passes_its_own_check(tmp_path):
    retriever = build(tmp_path)
    manifest = read_manifest(tmp_path)
    assert manifest is not None
    assert (manifest.chunk_count, manifest.document_count) == (6, 1)
    assert manifest.embedding_model == "model-a" and manifest.embedding_dim == 8
    result = retriever.check_index()
    assert result.ok, result.problems
    assert result.index_version == manifest.index_version


def test_the_index_version_is_stable_for_the_same_inputs_and_changes_when_the_corpus_does(tmp_path):
    build(tmp_path / "a")
    build(tmp_path / "b")
    build(tmp_path / "c", n=7)
    first = read_manifest(tmp_path / "a").index_version
    again = read_manifest(tmp_path / "b").index_version
    bigger = read_manifest(tmp_path / "c").index_version
    assert first == again
    assert bigger != first


def test_a_different_embedding_model_of_the_same_size_is_caught(tmp_path):
    # the case that does not raise anywhere else: both models make 8 dimensional vectors
    build(tmp_path, name="model-a", dim=8)
    result = reopen(tmp_path, name="model-b", dim=8).check_index()
    assert not result.ok
    assert any("model-a" in p and "model-b" in p for p in result.problems), result.problems


def test_a_partial_reingest_that_leaves_bm25_smaller_than_the_vectors_is_caught(tmp_path):
    retriever = build(tmp_path, n=6)
    retriever.build_index(make_chunks(2, source="new.pdf"))  # upserts 2 into Chroma, rebuilds BM25 from only these 2
    result = retriever.check_index()
    assert not result.ok
    assert any("vector index has 8" in p and "BM25 index has 2" in p for p in result.problems), result.problems


def test_a_changed_chunking_configuration_is_caught(tmp_path, monkeypatch):
    retriever = build(tmp_path)
    monkeypatch.setattr("src.retriever.CHUNK_SIZE", 200)
    result = retriever.check_index()
    assert any("chunked at" in p for p in result.problems), result.problems


def test_an_empty_index_is_not_ready(tmp_path):
    result = reopen(tmp_path).check_index()
    assert not result.ok
    assert any("empty" in p for p in result.problems)


def test_a_missing_manifest_is_a_problem_unless_you_say_otherwise(tmp_path):
    retriever = build(tmp_path)
    (tmp_path / index_manifest.MANIFEST_NAME).unlink()
    strict = retriever.check_index()
    assert not strict.ok and "write" in strict.problems[0]
    lenient = retriever.check_index(require_manifest=False)
    assert lenient.ok and lenient.warnings and lenient.index_version == "unversioned"


def test_a_manifest_can_be_written_for_an_index_built_before_manifests_existed(tmp_path):
    build(tmp_path)
    (tmp_path / index_manifest.MANIFEST_NAME).unlink()
    reopened = reopen(tmp_path)
    assert not reopened.check_index().ok
    reopened.write_manifest_for_existing_index()
    assert reopened.check_index().ok


def test_a_corrupt_manifest_is_a_problem_not_a_crash(tmp_path):
    retriever = build(tmp_path)
    (tmp_path / index_manifest.MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    result = retriever.check_index()
    assert not result.ok and "could not be read" in result.problems[0]


# ---- the files are checked before a pickle is loaded ----


def test_a_replaced_bm25_pickle_is_refused_before_it_runs(tmp_path):
    build(tmp_path)
    marker = tmp_path / "pwned.txt"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))  # what loading this would do: create the file

    (tmp_path / "bm25_index.pkl").write_bytes(pickle.dumps(Evil()))
    with pytest.raises(IndexIntegrityError, match="bm25_index.pkl"):
        reopen(tmp_path)
    assert not marker.exists(), "the pickle was loaded before its checksum was compared"


def test_a_damaged_chunk_file_is_refused(tmp_path):
    build(tmp_path)
    path = tmp_path / "bm25_chunks.pkl"
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(IndexIntegrityError, match="bm25_chunks.pkl"):
        reopen(tmp_path)


def test_an_index_without_a_manifest_still_loads_so_old_indexes_keep_working(tmp_path):
    build(tmp_path)
    (tmp_path / index_manifest.MANIFEST_NAME).unlink()
    assert reopen(tmp_path).bm25_ready


def test_verify_files_with_no_manifest_does_nothing(tmp_path):
    (tmp_path / "a.pkl").write_bytes(b"x")
    (tmp_path / "b.pkl").write_bytes(b"y")
    verify_files_before_loading(tmp_path, tmp_path / "a.pkl", tmp_path / "b.pkl")


def test_the_corpus_fingerprint_ignores_order_and_changes_with_content():
    assert corpus_fingerprint(["a", "b", "c"]) == corpus_fingerprint(["c", "a", "b"])
    assert corpus_fingerprint(["a", "b"]) != corpus_fingerprint(["a", "b", "c"])


def test_the_manifest_file_is_readable_json_with_the_fields_an_operator_needs(tmp_path):
    build(tmp_path)
    data = json.loads((tmp_path / index_manifest.MANIFEST_NAME).read_text(encoding="utf-8"))
    for key in ("embedding_model", "embedding_dim", "chunk_count", "document_count", "built_at", "bm25_index_sha256"):
        assert key in data
