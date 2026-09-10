"""Tests for the RAGAS slice exporter (eval/ragas_export.py).

The RAGAS run itself lives in a separate environment and is not tested here.
This covers the part that runs in the project env: reading a harness metrics
file and writing the JSONL the RAGAS step consumes.
"""

from __future__ import annotations

import io
import json

import pytest

import eval.ragas_export as rx


@pytest.fixture
def metrics_file(tmp_path):
    report = {
        "config": {"use_planner": False, "use_verifier": False},
        "agent_model": "claude-sonnet-4-6",
        "judge_model": "claude-opus-5",
        "per_question": [
            {"id": i, "question": f"q{i}", "answer": f"a{i} [d.pdf, Page 1]",
             "groundedness": 0.95, "correctness": 1.0}
            for i in range(20)
        ],
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_export_writes_one_row_per_sampled_question(monkeypatch, metrics_file, tmp_path):
    monkeypatch.setattr(rx, "_contexts", lambda q: [f"passage for {q}"])
    out = tmp_path / "ragas_input.jsonl"

    rx.export(metrics_file, n=5, seed=1, out_path=out)

    rows = [json.loads(line) for line in io.open(out, encoding="utf-8")]
    assert len(rows) == 5
    assert set(rows[0]) == {
        "id", "question", "answer", "contexts", "judge_groundedness", "judge_correctness"
    }
    assert rows[0]["contexts"] == [f"passage for {rows[0]['question']}"]
    assert rows[0]["judge_groundedness"] == 0.95


def test_export_is_deterministic_for_a_seed(monkeypatch, metrics_file, tmp_path):
    monkeypatch.setattr(rx, "_contexts", lambda q: ["ctx"])
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"

    rx.export(metrics_file, n=6, seed=42, out_path=a)
    rx.export(metrics_file, n=6, seed=42, out_path=b)

    assert a.read_text() == b.read_text()


def test_export_skips_error_answers(monkeypatch, metrics_file, tmp_path):
    report = json.loads(metrics_file.read_text())
    report["per_question"][0]["answer"] = "ERROR: boom"
    metrics_file.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(rx, "_contexts", lambda q: ["ctx"])
    out = tmp_path / "o.jsonl"

    rx.export(metrics_file, n=20, seed=1, out_path=out)

    ids = {json.loads(line)["id"] for line in io.open(out, encoding="utf-8")}
    assert 0 not in ids
    assert len(ids) == 19


def test_export_warns_when_the_run_used_the_planner(monkeypatch, metrics_file, tmp_path, capsys):
    report = json.loads(metrics_file.read_text())
    report["config"]["use_planner"] = True
    metrics_file.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(rx, "_contexts", lambda q: ["ctx"])

    rx.export(metrics_file, n=3, seed=1, out_path=tmp_path / "o.jsonl")

    assert "planner" in capsys.readouterr().out.lower()
