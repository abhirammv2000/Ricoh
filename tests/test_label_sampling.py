"""The human-label worksheet has to contain answers the judge might get wrong.

A random 30 of a benchmark that mostly passes is mostly passes, and then even a perfect agreement gives a kappa
near zero. These tests pin the stratified sample: every failing answer first, then borderline ones, then passes,
a row is skipped when the passages differ from what the judge saw, and the worksheet never shows the judge's scores.
"""

from __future__ import annotations

import json
import random

import eval.label_for_kappa as lfk
from eval.label_for_kappa import pick_stratified


def make_rows():
    rows = []
    for i in range(100):
        if i < 5:
            score = 0.5          # fails the 0.8 line
        elif i < 25:
            score = 0.9          # borderline
        else:
            score = 1.0
        rows.append({
            "id": i, "question": f"q{i}", "answer": f"a{i}", "expected_behavior": "answer",
            "groundedness": score, "correctness": 1.0, "evidence_docs": [f"doc{i}.pdf"],
        })
    return rows


def same_documents(row):
    return [{"source": d, "page": 1, "text": "t"} for d in row["evidence_docs"]]


def bands(picked):
    out = {"fail": 0, "borderline": 0, "pass": 0}
    for row, _ in picked:
        low = min(row["groundedness"], row["correctness"])
        out["fail" if low < 0.8 else "borderline" if low < 0.95 else "pass"] += 1
    return out


def test_it_takes_every_failing_answer_and_mostly_borderline_ones():
    picked = pick_stratified(make_rows(), 30, seed=1, evidence_for=same_documents)

    assert len(picked) == 30
    assert bands(picked) == {"fail": 5, "borderline": 16, "pass": 9}


def test_it_is_repeatable_for_a_seed_and_differs_for_another():
    first = [r["id"] for r, _ in pick_stratified(make_rows(), 30, seed=1, evidence_for=same_documents)]
    again = [r["id"] for r, _ in pick_stratified(make_rows(), 30, seed=1, evidence_for=same_documents)]
    other = [r["id"] for r, _ in pick_stratified(make_rows(), 30, seed=2, evidence_for=same_documents)]

    assert first == again
    assert set(first) != set(other)


def test_a_row_whose_passages_changed_is_skipped_and_replaced():
    def evidence_for(row):
        if row["id"] == 0:   # one failing row now retrieves a different document
            return [{"source": "another.pdf", "page": 1, "text": "t"}]
        return same_documents(row)

    picked = pick_stratified(make_rows(), 30, seed=1, evidence_for=evidence_for)

    assert len(picked) == 30
    assert 0 not in [r["id"] for r, _ in picked]
    assert bands(picked)["fail"] == 4


def test_it_never_returns_more_than_the_rows_that_exist():
    rows = make_rows()[:12]

    picked = pick_stratified(rows, 30, seed=1, evidence_for=same_documents)

    assert sorted(r["id"] for r, _ in picked) == list(range(12))


def test_an_answer_that_fails_on_either_measure_counts_as_failing():
    rows = make_rows()
    rows[50]["correctness"] = 0.4       # a good groundedness score but a wrong answer

    picked = pick_stratified(rows, 30, seed=1, evidence_for=same_documents)

    assert 50 in [r["id"] for r, _ in picked]
    assert bands(picked)["fail"] == 6


def write_metrics(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"per_question": make_rows()}), encoding="utf-8")
    return path


def test_the_worksheet_does_not_show_the_judges_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(lfk, "WORKSHEET_PATH", tmp_path / "labels.json")
    monkeypatch.setattr(lfk, "_reconstruct_evidence", lambda question: same_documents(make_rows()[int(question[1:])]))

    lfk.make_worksheet(write_metrics(tmp_path), 30, seed=1, stratified=True)

    sheet = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert len(sheet["items"]) == 30
    assert sheet["sampling"] == "stratified by the judge's scores"
    for item in sheet["items"]:
        assert "groundedness" not in item and "correctness" not in item
        assert item["human_grounded"] is None and item["human_correct"] is None
    assert "0.5" not in json.dumps(sheet)


def test_the_default_is_still_the_same_random_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(lfk, "WORKSHEET_PATH", tmp_path / "labels.json")
    monkeypatch.setattr(lfk, "_reconstruct_evidence", lambda question: same_documents(make_rows()[int(question[1:])]))
    expected = [r["id"] for r in random.Random(7).sample(make_rows(), 10)]

    lfk.make_worksheet(write_metrics(tmp_path), 10, seed=7)

    sheet = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert [i["id"] for i in sheet["items"]] == expected
    assert sheet["sampling"] == "random"
