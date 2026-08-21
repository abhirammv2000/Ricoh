"""Unit tests for tool-calling retrieval (src/tools.py).

A scripted fake model returns whatever sequence of tool calls a test needs, so
loop control, evidence accumulation, and the call budget are all deterministic
and cost nothing. What is NOT tested here is whether the model chooses good
queries: that is a measured question, not a unit-testable one, and the answer
lives in the eval harness.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.tools as tools


class ScriptedLLM:
    """Returns a canned response per invoke(), in order.

    Each script entry is either a list of tool calls or a final text answer.
    bind_tools() returns self and records that it was called, so a test can
    assert the tool was withdrawn on the last turn.
    """

    def __init__(self, script):
        self._script = list(script)
        self.invocations: list[object] = []
        self.bound_calls = 0

    def bind_tools(self, tools_arg):
        self.bound_calls += 1
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        step = self._script.pop(0)
        if isinstance(step, str):
            return SimpleNamespace(content=step, tool_calls=[], usage_metadata={})
        return SimpleNamespace(content="", tool_calls=step, usage_metadata={})


def _call(query, cid="t1"):
    return {"name": "search_docs", "args": {"query": query}, "id": cid}


@pytest.fixture
def fake_retrieval(monkeypatch):
    """search_docs returns one chunk named after the query."""
    def _search(query, **kw):
        return [{
            "id": f"chunk-{query}",
            "text": f"text for {query}",
            "source_document": f"{query}.pdf",
            "page_number": 1,
        }]
    monkeypatch.setattr(tools, "search_docs", _search)
    return _search


def test_answers_without_searching_when_model_does_not_call_the_tool(monkeypatch):
    llm = ScriptedLLM(["direct answer"])
    monkeypatch.setattr(tools, "get_llm", lambda: llm)
    out = tools.run_tool_loop("q")
    assert out["answer"] == "direct answer"
    assert out["tool_calls"] == 0
    assert out["evidence"] == []


def test_accumulates_evidence_across_multiple_searches(monkeypatch, fake_retrieval):
    llm = ScriptedLLM([[_call("alpha")], [_call("beta", "t2")], "final"])
    monkeypatch.setattr(tools, "get_llm", lambda: llm)
    out = tools.run_tool_loop("q")
    assert out["answer"] == "final"
    assert out["queries"] == ["alpha", "beta"]
    assert {c["source_document"] for c in out["evidence"]} == {"alpha.pdf", "beta.pdf"}


def test_duplicate_chunks_are_not_double_counted(monkeypatch, fake_retrieval):
    # The same query twice must not append the same chunk twice: an evidence
    # list with duplicates would inflate what the synthesizer appears to have.
    llm = ScriptedLLM([[_call("same")], [_call("same", "t2")], "final"])
    monkeypatch.setattr(tools, "get_llm", lambda: llm)
    out = tools.run_tool_loop("q")
    assert len(out["evidence"]) == 1
    assert out["tool_calls"] == 2


def test_tool_is_withdrawn_on_the_final_turn(monkeypatch, fake_retrieval):
    # A model that keeps calling the tool must still terminate. With
    # max_calls=2 the third invocation gets the unbound model, which cannot
    # emit tool calls, so the loop ends with an answer instead of running on.
    llm = ScriptedLLM([[_call("a")], [_call("b", "t2")], "forced answer"])
    monkeypatch.setattr(tools, "get_llm", lambda: llm)
    out = tools.run_tool_loop("q", max_calls=2)
    assert out["answer"] == "forced answer"
    assert out["tool_calls"] == 2
    # Three invocations: two searching turns, then the forced-answer turn.
    assert len(llm.invocations) == 3


def test_empty_query_does_not_reach_the_retriever(monkeypatch):
    called = []
    monkeypatch.setattr(tools, "search_docs", lambda q, **kw: called.append(q) or [])
    llm = ScriptedLLM([[_call("")], "done"])
    monkeypatch.setattr(tools, "get_llm", lambda: llm)
    out = tools.run_tool_loop("q")
    assert called == []
    assert out["evidence"] == []


def test_format_results_labels_every_passage_for_citation():
    text = tools._format_results([
        {"source_document": "a.pdf", "page_number": 3, "text": "body"},
    ])
    assert "[a.pdf, Page 3]" in text
    assert "body" in text


def test_format_results_says_so_when_nothing_was_found():
    assert "No passages" in tools._format_results([])
