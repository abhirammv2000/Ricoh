"""Router tests (src/router.py). Offline, with the retriever/synthesizer/tool
loop stubbed so the decision is deterministic."""

from __future__ import annotations

import pytest

import src.router as router

REFUSAL = "Information unavailable in provided documents."


def _fused(*scores: float, docs: list[str] | None = None) -> list[dict]:
    docs = docs or [f"d{i}.pdf" for i in range(len(scores))]
    return [
        {"id": f"c{i}", "source_document": docs[i], "page_number": 1,
         "text": f"chunk {i}", "rrf_score": s}
        for i, s in enumerate(scores)
    ]


def test_signals_empty():
    assert router.confidence_signals([]) == {"top_rrf": 0.0, "margin": 0.0, "doc_spread": 0, "n": 0}


def test_signals_basic():
    s = router.confidence_signals(_fused(0.03, 0.01, 0.01, docs=["a.pdf", "a.pdf", "b.pdf"]))
    assert s["top_rrf"] == 0.03
    assert s["margin"] == pytest.approx(0.02)
    assert s["doc_spread"] == 2
    assert s["n"] == 3


@pytest.fixture
def patched_paths(monkeypatch):
    """Stub retriever, synthesizer and tool loop; record which path ran."""
    calls: dict[str, object] = {}

    def _install(answer: str):
        monkeypatch.setattr(
            router, "get_retriever",
            lambda: type("R", (), {"retrieve": lambda self, *a, **k: _fused(0.03, 0.01)})(),
        )
        import src.agent as agent
        import src.tools as tools

        def _synth(state):
            calls["synth"] = state
            return {"final_answer": answer}

        def _tool(q):
            calls["tool"] = q
            return {"answer": "tool answer"}

        monkeypatch.setattr(agent, "synthesizer_node", _synth)
        monkeypatch.setattr(tools, "run_tool_loop", _tool)
        return calls

    return _install


def test_direct_answer_is_not_escalated(patched_paths):
    calls = patched_paths("Set the property to Yes [a.pdf, Page 1].")
    state = router.route("some question")
    assert state["final_answer"] == "Set the property to Yes [a.pdf, Page 1]."
    assert state["router_decision"] == "direct"
    assert state["retrieved_evidence"][0]["rrf_score"] == 0.03
    assert "tool" not in calls


def test_refusal_escalates_to_tool_loop(patched_paths):
    calls = patched_paths(REFUSAL)
    state = router.route("hard question")
    assert state["final_answer"] == "tool answer"
    assert state["router_decision"] == "escalate"
    assert calls["tool"] == "hard question"
    assert "synth" in calls  # cheap path ran first


def test_route_and_run_returns_just_the_string(patched_paths):
    patched_paths("A direct answer [x.pdf, Page 2].")
    assert router.route_and_run("q") == "A direct answer [x.pdf, Page 2]."
