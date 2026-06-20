"""Unit tests for agent control flow and parsing (src/agent.py).

The LLM is mocked, so these tests run offline and deterministically.
They cover the parts most likely to break silently: retry routing,
planner JSON parsing (including fenced / malformed output), and
verifier verdict normalisation.
"""

from __future__ import annotations

import src.agent as agent
from src.agent import (
    MAX_ITERATIONS,
    planner_node,
    should_retry_or_synthesize,
    verifier_node,
)
from tests.helpers import FakeLLM


# ── Routing logic ──────────────────────────────────────────────────

def test_routes_back_to_planner_when_insufficient_and_under_cap():
    state = {"verification_status": "INSUFFICIENT", "iterations": 1}
    assert should_retry_or_synthesize(state) == "planner"


def test_routes_to_synthesizer_when_cap_reached():
    state = {"verification_status": "INSUFFICIENT", "iterations": MAX_ITERATIONS}
    assert should_retry_or_synthesize(state) == "synthesizer"


def test_routes_to_synthesizer_when_sufficient():
    state = {"verification_status": "SUFFICIENT", "iterations": 0}
    assert should_retry_or_synthesize(state) == "synthesizer"


# ── Planner JSON parsing ───────────────────────────────────────────

def _patch_llm(monkeypatch, response, fake_cls):
    monkeypatch.setattr(agent, "get_llm", lambda *a, **k: fake_cls(response))


def test_planner_parses_clean_json(monkeypatch):
    resp = '{"entities": ["SC542"], "sub_queries": ["fix SC542", "fuser unit"]}'
    _patch_llm(monkeypatch, resp, FakeLLM)
    out = planner_node({"user_query": "fix SC542", "iterations": 0, "retrieved_evidence": []})
    assert out["entities"] == ["SC542"]
    assert out["sub_queries"] == ["fix SC542", "fuser unit"]


def test_planner_strips_markdown_fences(monkeypatch):
    resp = '```json\n{"entities": [], "sub_queries": ["q1"]}\n```'
    _patch_llm(monkeypatch, resp, FakeLLM)
    out = planner_node({"user_query": "q1", "iterations": 0, "retrieved_evidence": []})
    assert out["sub_queries"] == ["q1"]


def test_planner_falls_back_on_bad_json(monkeypatch):
    _patch_llm(monkeypatch, "not json at all", FakeLLM)
    out = planner_node({"user_query": "orig query", "iterations": 0, "retrieved_evidence": []})
    assert out["sub_queries"] == ["orig query"]
    assert out["entities"] == []


# ── Verifier verdict normalisation ─────────────────────────────────

def test_verifier_accepts_sufficient(monkeypatch):
    _patch_llm(monkeypatch, "SUFFICIENT", FakeLLM)
    out = verifier_node({"user_query": "q", "retrieved_evidence": [], "iterations": 1})
    assert out["verification_status"] == "SUFFICIENT"


def test_verifier_detects_insufficient_even_though_it_contains_sufficient(monkeypatch):
    # "INSUFFICIENT" contains the substring "SUFFICIENT" - must not misclassify.
    _patch_llm(monkeypatch, "INSUFFICIENT", FakeLLM)
    out = verifier_node({"user_query": "q", "retrieved_evidence": [], "iterations": 1})
    assert out["verification_status"] == "INSUFFICIENT"


def test_verifier_defaults_to_sufficient_on_garbage(monkeypatch):
    _patch_llm(monkeypatch, "maybe?", FakeLLM)
    out = verifier_node({"user_query": "q", "retrieved_evidence": [], "iterations": 1})
    assert out["verification_status"] == "SUFFICIENT"
