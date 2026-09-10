"""Offline checks for eval/multiturn_eval.py and the chain file.

The judged run needs an API key; this covers the JSON shape and the pure
aggregation helpers.
"""

from __future__ import annotations

import io
import json

from src.config import PROJECT_ROOT
from eval.multiturn_eval import _aggregate, _distinct_docs, _recall

CHAINS = PROJECT_ROOT / "eval" / "multiturn_questions.json"


def _chains():
    return json.loads(io.open(CHAINS, encoding="utf-8").read())["chains"]


# --- chain file structure ---


def test_every_chain_has_a_standalone_first_turn_and_reference_followups():
    for chain in _chains():
        turns = chain["turns"]
        assert len(turns) >= 2, chain["id"]
        # Turn 1's raw question should equal its standalone (already self-contained).
        assert turns[0]["question"] == turns[0]["standalone"], chain["id"]
        # Later turns are rewritten, so raw != standalone.
        for t in turns[1:]:
            assert t["question"] != t["standalone"], (chain["id"], t["question"])


def test_turns_carry_judging_fields():
    for chain in _chains():
        for t in chain["turns"]:
            assert t["expected_behavior"] == "answer"
            assert t["key_facts"]
            assert t["expected_sources"] and all(s.endswith(".pdf") for s in t["expected_sources"])


# --- helpers ---


def test_distinct_docs_dedupes_preserving_order():
    ev = [{"source_document": "b.pdf"}, {"source_document": "a.pdf"}, {"source_document": "b.pdf"}]
    assert _distinct_docs(ev) == ["b.pdf", "a.pdf"]


def test_recall_is_any_hit_over_expected_sources():
    # Any expected source in the top-k is a hit; the multi-turn set lists
    # alternative valid documents on some turns.
    assert _recall(["a.pdf", "b.pdf"], ["a.pdf", "x.pdf", "y.pdf"]) == 1.0
    assert _recall(["a.pdf", "b.pdf"], ["x.pdf", "y.pdf"]) == 0.0
    assert _recall(["a.pdf"], ["a.pdf"]) == 1.0
    assert _recall([], ["a.pdf"]) == 0.0


def test_aggregate_splits_first_turns_from_followups():
    rows = [
        {"is_followup": False, "evidence_recall": 1.0, "behavior_match": True,
         "groundedness": 1.0, "correctness": 1.0, "rewrite_cosine_to_target": None,
         "recall_raw_followup": 1.0, "recall_after_condense": 1.0, "cost_usd": 0.01},
        {"is_followup": True, "evidence_recall": 0.5, "behavior_match": True,
         "groundedness": 0.9, "correctness": 0.8, "rewrite_cosine_to_target": 0.85,
         "recall_raw_followup": 0.0, "recall_after_condense": 0.5, "cost_usd": 0.02},
    ]
    out = _aggregate(rows, use_judge=True)["summary"]
    assert out["first_turns"]["n"] == 1
    assert out["followups"]["n"] == 1
    assert out["followups"]["groundedness"] == 0.9
    assert out["condensation"]["mean_recall_raw_followup"] == 0.0
    assert out["condensation"]["mean_recall_after_condense"] == 0.5
    assert out["condensation"]["mean_rewrite_cosine_to_target"] == 0.85
