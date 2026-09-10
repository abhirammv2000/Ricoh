"""Tests for the retrieval regression gate's comparison logic (eval/ci_gate.py).

The retrieval measurement itself needs a real index and is exercised by the
separate CI job. This covers the pure part: deciding whether a set of recall
numbers has regressed against the baseline.
"""

from __future__ import annotations

import json

import eval.ci_gate as ci_gate
from eval.ci_gate import TOLERANCE, manifest_drift, regressions


def _m(at1: float, at3: float, at5: float) -> dict:
    return {"recall": {"@1": at1, "@3": at3, "@5": at5}}


def test_identical_numbers_do_not_regress():
    base = _m(0.81, 1.0, 1.0)
    assert regressions(base, base) == []


def test_a_drop_beyond_tolerance_is_a_regression():
    base = _m(0.81, 1.0, 1.0)
    worse = _m(0.81, 0.9, 1.0)
    found = regressions(worse, base)
    assert len(found) == 1 and "@3" in found[0]


def test_an_improvement_is_not_a_regression():
    base = _m(0.81, 0.9, 1.0)
    better = _m(1.0, 1.0, 1.0)
    assert regressions(better, base) == []


def test_a_drop_within_tolerance_is_ignored():
    base = _m(0.81, 1.0, 1.0)
    within = _m(0.81, 1.0 - TOLERANCE / 2, 1.0)
    assert regressions(within, base) == []


# --- manifest drift ---


def test_the_committed_demo_index_is_in_sync():
    # This runs against the real committed manifest + ground_truth.json.
    assert manifest_drift() == []


def test_drift_is_flagged_when_ground_truth_needs_a_doc_the_index_lacks(monkeypatch, tmp_path):
    manifest = {
        "document_count": 1,
        "benchmark_referenced": ["have.pdf"],
        "sampled": [],
    }
    fake_dir = tmp_path / "demo_index"
    fake_dir.mkdir()
    (fake_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(ci_gate, "DEMO_INDEX", fake_dir)
    monkeypatch.setattr(
        "src.build_demo_index._referenced_docs", lambda: {"have.pdf", "missing.pdf"}
    )

    problems = manifest_drift()
    assert any("missing.pdf" in p for p in problems)


def test_drift_flags_a_document_count_mismatch(monkeypatch, tmp_path):
    manifest = {"document_count": 9, "benchmark_referenced": ["a.pdf"], "sampled": ["b.pdf"]}
    fake_dir = tmp_path / "demo_index"
    fake_dir.mkdir()
    (fake_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(ci_gate, "DEMO_INDEX", fake_dir)
    monkeypatch.setattr("src.build_demo_index._referenced_docs", lambda: {"a.pdf"})

    assert any("document_count" in p for p in manifest_drift())
