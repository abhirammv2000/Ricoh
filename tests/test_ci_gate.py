"""Tests for the retrieval regression gate's comparison logic (eval/ci_gate.py).

The retrieval measurement itself needs a real index and is exercised by the
separate CI job. This covers the pure part: deciding whether a set of recall
numbers has regressed against the baseline.
"""

from __future__ import annotations

from eval.ci_gate import TOLERANCE, regressions


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
