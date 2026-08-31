"""Run the cheap path, escalate to the tool loop only when it refuses.

The default path is one retrieval then synthesize. The tool loop (src/tools.py)
rescues questions where that retrieval comes back thin, but it costs ~1.8x on
every question, so we only reach for it when the synthesizer says it cannot
answer from the evidence.

An earlier version escalated on a retrieval confidence signal instead;
eval/calibrate_router.py shows that signal does not separate the misses from the
hits on this corpus, so it was dropped. Off by default (USE_ROUTER).
"""

from __future__ import annotations

from typing import Any

from src.config import RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K
from src.instrumentation import span
from src.retriever import get_retriever


def confidence_signals(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Retrieval signals recorded on the trace (not used to branch, see module docstring)."""
    if not results:
        return {"top_rrf": 0.0, "margin": 0.0, "doc_spread": 0, "n": 0}

    rrf = [float(r.get("rrf_score", 0.0)) for r in results]
    docs: list[str] = []
    for r in results:
        d = r.get("source_document")
        if d and d not in docs:
            docs.append(d)

    return {
        "top_rrf": round(rrf[0], 5),
        "margin": round(rrf[0] - (rrf[1] if len(rrf) > 1 else 0.0), 5),
        "doc_spread": len(docs),
        "n": len(results),
    }


def route(query: str) -> dict[str, Any]:
    """Return a partial agent state so the eval harness can score a routed run."""
    from src.agent import is_refusal, synthesizer_node
    from src.tools import run_tool_loop

    results = get_retriever().retrieve(
        query, top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K
    )
    answer = synthesizer_node(
        {"user_query": query, "retrieved_evidence": results}
    )["final_answer"]
    signals = confidence_signals(results)

    if not is_refusal(answer):
        with span("router", decision="direct", **signals):
            pass
        return {
            "final_answer": answer,
            "retrieved_evidence": results,
            "router_decision": "direct",
        }

    with span("router", decision="escalate", **signals):
        pass
    loop = run_tool_loop(query)
    return {
        "final_answer": loop["answer"],
        "retrieved_evidence": loop.get("evidence", []),
        "sub_queries": loop.get("queries", []),
        "router_decision": "escalate",
    }


def route_and_run(query: str) -> str:
    return route(query)["final_answer"]
