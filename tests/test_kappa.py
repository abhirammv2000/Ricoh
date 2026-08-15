"""Unit tests for the Cohen's kappa computation (eval/label_for_kappa._kappa).

Pure and offline. These pin the behaviour that matters most: that kappa corrects
for the base rate, so a judge which says "acceptable" every time scores near zero
however high its raw agreement looks.
"""

from __future__ import annotations

import math

from eval.label_for_kappa import _kappa


def test_perfect_agreement_balanced():
    s = _kappa([1, 1, 0, 0], [1, 1, 0, 0])
    assert s["raw_agreement"] == 1.0
    assert s["cohens_kappa"] == 1.0


def test_total_disagreement():
    s = _kappa([1, 1, 0, 0], [0, 0, 1, 1])
    assert s["raw_agreement"] == 0.0
    assert s["cohens_kappa"] == -1.0


def test_chance_level_agreement_is_zero():
    s = _kappa([1, 0, 1, 0], [1, 1, 0, 0])
    assert s["raw_agreement"] == 0.5
    assert s["cohens_kappa"] == 0.0


def test_base_rate_inflates_raw_but_not_kappa():
    # The judge labels every item acceptable: 90% raw agreement, zero real
    # information. Kappa must expose that as 0.0, which is the whole reason the
    # project reports kappa instead of raw agreement.
    s = _kappa([1] * 9 + [0], [1] * 10)
    assert s["raw_agreement"] == 0.9
    assert s["chance_agreement"] == 0.9
    assert s["cohens_kappa"] == 0.0


def test_single_category_throughout_is_undefined():
    s = _kappa([1, 1, 1], [1, 1, 1])
    assert math.isnan(s["cohens_kappa"])


def test_fragility_is_reported_and_nonnegative():
    s = _kappa([1] * 9 + [0], [1] * 10)
    assert s["kappa_drop_if_one_item_flipped"] >= 0
    assert s["n"] == 10
