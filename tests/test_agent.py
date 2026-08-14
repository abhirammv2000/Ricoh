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


# Routing logic

def test_routes_back_to_planner_when_insufficient_and_under_cap():
    state = {"verification_status": "INSUFFICIENT", "iterations": 1}
    assert should_retry_or_synthesize(state) == "planner"


def test_routes_to_synthesizer_when_cap_reached():
    state = {"verification_status": "INSUFFICIENT", "iterations": MAX_ITERATIONS}
    assert should_retry_or_synthesize(state) == "synthesizer"


def test_routes_to_synthesizer_when_sufficient():
    state = {"verification_status": "SUFFICIENT", "iterations": 0}
    assert should_retry_or_synthesize(state) == "synthesizer"


# Planner JSON parsing

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


# Verifier verdict normalisation

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


# Pipeline configuration / ablation wiring
# These guard the ablation instrument itself. If graph construction or the
# state-seeding contract silently breaks, every ablation config would run
# the same pipeline and the comparison would read as "no effect", a
# false negative that is invisible in the results table.

import pytest

from src.agent import build_agent_graph, get_agent_graph, initial_state


def _node_names(graph):
    return {n for n in graph.get_graph().nodes if not n.startswith("__")}


def test_retrieve_only_config_omits_planner_and_verifier():
    g = build_agent_graph(use_planner=False, use_verifier=False)
    assert _node_names(g) == {"retriever", "synthesizer"}


def test_planner_only_config_omits_verifier():
    g = build_agent_graph(use_planner=True, use_verifier=False)
    assert _node_names(g) == {"planner", "retriever", "synthesizer"}


def test_full_config_has_every_node():
    g = build_agent_graph(use_planner=True, use_verifier=True)
    assert _node_names(g) == {"planner", "retriever", "verifier", "synthesizer"}


def test_default_config_is_the_full_pipeline():
    # Ablation support must not silently change production behaviour.
    assert _node_names(build_agent_graph()) == _node_names(
        build_agent_graph(use_planner=True, use_verifier=True)
    )


def test_state_seeds_raw_question_when_planner_disabled():
    # Without the planner nothing else populates sub_queries; an empty list
    # would make the retriever a no-op and the config would score 0 for
    # reasons unrelated to the design being tested.
    st = initial_state("how do I use locations?", use_planner=False)
    assert st["sub_queries"] == ["how do I use locations?"]


def test_state_leaves_sub_queries_empty_when_planner_enabled():
    assert initial_state("q", use_planner=True)["sub_queries"] == []


@pytest.mark.parametrize(
    "use_planner,use_verifier",
    [(False, False), (True, False), (True, True)],
)
def test_graph_cache_is_keyed_by_configuration(use_planner, use_verifier):
    # A single cached graph shared across configs would make every ablation
    # rung run identical code.
    a = get_agent_graph(use_planner=use_planner, use_verifier=use_verifier)
    b = get_agent_graph(use_planner=use_planner, use_verifier=use_verifier)
    assert a is b, "same config should reuse the compiled graph"
    other = get_agent_graph(use_planner=not use_planner, use_verifier=use_verifier)
    assert a is not other, "different config must not reuse a cached graph"


def test_harness_defaults_match_production_config():
    """The eval harness must benchmark what production actually runs.

    Regression guard: these defaults were once hard-coded to True while
    production shipped the single-retrieval path, so a full benchmark run
    silently measured a pipeline nobody uses. The failure is invisible in the
    output, the numbers look fine, they just describe the wrong system.
    """
    import inspect

    from src.config import USE_PLANNER, USE_VERIFIER
    from src.eval_harness import evaluate

    params = inspect.signature(evaluate).parameters
    assert params["use_planner"].default is None, (
        "use_planner must default to None so it resolves from config, "
        "not to a hard-coded literal"
    )
    assert params["use_verifier"].default is None

    # And the resolution must actually reach config.
    src = inspect.getsource(evaluate)
    assert "USE_PLANNER if use_planner is None" in src
    assert "USE_VERIFIER if use_verifier is None" in src
    assert isinstance(USE_PLANNER, bool) and isinstance(USE_VERIFIER, bool)
