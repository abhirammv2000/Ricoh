"""Tests for the pure comparison helpers in eval/ragas_eval.py.

The RAGAS call itself runs in a separate environment. The module's top level
imports nothing from RAGAS, so these statistics helpers are testable here.
"""

from __future__ import annotations

import math

from eval.ragas_eval import _kappa, _pearson


def test_kappa_is_one_on_perfect_agreement_with_a_split_base_rate():
    a = [1, 1, 1, 0, 0, 0]
    assert _kappa(a, a)["cohens_kappa"] == 1.0


def test_kappa_is_near_zero_when_agreement_matches_chance():
    # Both raters say "1" 5/6 of the time but disagree on which items.
    human = [1, 1, 1, 1, 1, 0]
    judge = [1, 1, 1, 1, 0, 1]
    stats = _kappa(human, judge)
    assert stats["cohens_kappa"] is not None
    assert abs(stats["cohens_kappa"]) < 0.3


def test_kappa_is_none_when_both_raters_are_constant_and_identical():
    # chance agreement is 1.0, so kappa is undefined (0/0).
    assert _kappa([1, 1, 1, 1], [1, 1, 1, 1])["cohens_kappa"] is None


def test_kappa_is_zero_when_one_rater_is_constant_but_the_other_varies():
    # chance agreement is < 1 here, and observed agreement equals it exactly.
    assert _kappa([1, 1, 1], [1, 0, 1])["cohens_kappa"] == 0.0


def test_pearson_perfect_positive():
    assert _pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0


def test_pearson_none_on_constant_input():
    assert _pearson([0.5, 0.5, 0.5], [0.1, 0.9, 0.5]) is None


def test_pearson_matches_reference_value():
    xs = [0.7, 0.8, 1.0, 0.9]
    ys = [1.0, 0.75, 0.95, 1.0]
    got = _pearson(xs, ys)
    # computed independently
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    assert got == round(cov / (sx * sy), 3)
