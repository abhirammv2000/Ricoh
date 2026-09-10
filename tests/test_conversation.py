"""Unit tests for history-aware query condensation (src/conversation.py).

The LLM is a fake, so these are deterministic and offline. They cover the
contract that matters: no history means no call and no change, history means one
call and the model's rewrite is used, and a useless rewrite falls back to the
original question.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.conversation import (
    MAX_HISTORY_TURNS,
    Turn,
    _format_history,
    condense_query,
)
from tests.helpers import FakeLLM


def test_turn_is_immutable():
    turn = Turn(question="q", answer="a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        turn.question = "other"  # type: ignore[misc]


def test_no_history_returns_question_unchanged_without_calling_the_llm():
    llm = FakeLLM("SHOULD NOT BE USED")
    out = condense_query([], "How do I create a workflow?", llm=llm)
    assert out == "How do I create a workflow?"
    assert llm.calls == []


def test_history_triggers_one_rewrite_call_and_uses_its_output():
    llm = FakeLLM("How do I copy an existing workflow in RICOH ProcessDirector?")
    history = [Turn("How do I create a workflow?", "Click the Workflow tab ...")]

    out = condense_query(history, "Can I copy an existing one?", llm=llm)

    assert out == "How do I copy an existing workflow in RICOH ProcessDirector?"
    assert len(llm.calls) == 1
    # The follow-up and the prior turn both have to reach the model.
    assert "Can I copy an existing one?" in llm.calls[0]
    assert "How do I create a workflow?" in llm.calls[0]


def test_surrounding_quotes_are_stripped_from_the_rewrite():
    llm = FakeLLM('"What operating system does RICOH ProcessDirector run on?"')
    out = condense_query([Turn("q", "a")], "what OS does it run on?", llm=llm)
    assert out == "What operating system does RICOH ProcessDirector run on?"


def test_empty_rewrite_falls_back_to_the_original_question():
    llm = FakeLLM("   \n  ")
    out = condense_query([Turn("q", "a")], "the original follow-up", llm=llm)
    assert out == "the original follow-up"


def test_only_the_most_recent_turns_are_used():
    llm = FakeLLM("standalone")
    history = [Turn(f"question {i}", f"answer {i}") for i in range(MAX_HISTORY_TURNS + 3)]

    condense_query(history, "follow-up", llm=llm)

    prompt = llm.calls[0]
    assert "question 0" not in prompt
    assert f"question {MAX_HISTORY_TURNS + 2}" in prompt


def test_format_history_numbers_turns_and_truncates_long_answers():
    history = [
        Turn("first?", "short answer"),
        Turn("second?", "x" * 5000),
    ]
    rendered = _format_history(history)
    assert "1. User: first?" in rendered
    assert "2. User: second?" in rendered
    assert "..." in rendered
    assert len(rendered) < 1000
