"""Structural checks on eval/multihop_questions.json.

Retrieval quality is measured by eval/verify_multihop.py (needs the corpus);
this just guards the shape so a typo cannot silently break an ablation run.
"""

from __future__ import annotations

import io
import json

from src.config import PROJECT_ROOT

QUESTIONS = PROJECT_ROOT / "eval" / "multihop_questions.json"


def _load():
    return json.loads(io.open(QUESTIONS, encoding="utf-8").read())["questions"]


def test_ids_are_unique_and_contiguous():
    ids = [q["id"] for q in _load()]
    assert ids == list(range(1, len(ids) + 1))


def test_every_question_has_two_or_more_jointly_relevant_sources():
    for q in _load():
        assert len(q["expected_sources"]) >= 2, q["id"]
        assert len(set(q["expected_sources"])) == len(q["expected_sources"]), q["id"]


def test_all_answerable_and_curated_provenance():
    # provenance drives any_hit semantics in the harness: 'curated' means every
    # listed source is needed, which is what a multi-hop question is.
    for q in _load():
        assert q["expected_behavior"] == "answer", q["id"]
        assert q["provenance"] == "curated", q["id"]
        assert q["key_facts"], q["id"]


def test_source_filenames_look_like_pdfs():
    for q in _load():
        for s in q["expected_sources"]:
            assert s.endswith(".pdf"), (q["id"], s)
