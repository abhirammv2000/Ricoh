"""Tool-calling retrieval: the model issues its own searches.

The default pipeline retrieves once, on the raw question, and synthesizes. That
works because retrieval on this corpus is strong: recall@5 is 0.94 across the
100-question benchmark. The 6 failures are the interesting case. The model gets
one set of chunks and no way to say "that is not what I asked for, let me search
differently".

This module gives it that: `search_docs` is exposed as a tool, and the model
decides how many times to call it and with what query. Where the planner
rewrote the question up front and hoped, the model here sees the actual results
before deciding whether to search again.

Whether that helps is a measured question, not an assumed one, and here the
answer was yes, against expectation. A pre-build check replayed the six failing
questions through the retriever with four hand-written reformulations each and
recovered two, which was taken as a ceiling. The model beat it: 4 of 6 recovered,
because it reformulates after seeing the results rather than guessing up front,
which is exactly the mechanism the planner lacked. On a 20-question sample of
already-passing questions it lost none. Cost is 1.77x the single-retrieval path
($0.0278 vs $0.0157 per query).

That measurement covers retrieval only, whether the expected document reached
the model, not judged answer quality. Until a full judged run at n=100 settles
groundedness and correctness, the stage stays off by default (USE_TOOL_LOOP).
The evidence supports turning it on; it does not yet support making it the
default silently.

Design notes:

Tool results are capped. An unbounded loop that keeps searching is how a tool
agent turns a $0.016 query into a $0.50 one. MAX_TOOL_CALLS bounds it, and the
bound is enforced by stripping the tool rather than by trusting the model to
stop.

The loop is instrumented like every other stage. Each turn records a span via
invoke_messages(), so a tool run shows up in the cost dashboard as what it
actually costs rather than as a gap.

Evidence is accumulated across calls, not replaced. The synthesizer's citation
rules apply to everything the model saw, so a chunk retrieved on the second
search is as citable as one from the first.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K
from src.instrumentation import invoke_messages
from src.llm_factory import get_llm, response_text
from src.retriever import get_retriever

logger = logging.getLogger(__name__)

# Hard cap on searches per question. In the measured runs the model used 1 to 3
# searches and never needed a fourth, so this bounds the tail rather than the
# common case. Without a cap a tool agent turns a $0.016 query into a $0.50 one.
MAX_TOOL_CALLS: int = 4

SEARCH_DOCS_TOOL: dict[str, Any] = {
    "name": "search_docs",
    "description": (
        "Search the RICOH ProcessDirector documentation and return the most "
        "relevant passages with their source document and page number. Call "
        "this again with a differently worded query if the passages returned "
        "do not contain what the question asks for. Prefer specific technical "
        "terms over full sentences."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms. A short phrase works better than a sentence.",
            }
        },
        "required": ["query"],
    },
}

TOOL_SYSTEM_PROMPT = """\
You are a senior Ricoh technical support engineer answering from the RICOH \
ProcessDirector documentation.

Use the search_docs tool to find evidence. If the passages you get back do not \
answer the question, search again with different wording before giving up. You \
may search at most {max_calls} times.

When you have enough evidence, write the final answer using ONLY that evidence, \
citing every claim as [Document Name, Page X].

If the documentation genuinely does not contain the answer, say so plainly \
instead of guessing. A correct refusal is better than an unsupported answer.\
"""


def _format_results(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks the way the model is asked to cite them."""
    if not chunks:
        return "No passages found for that query."
    parts = []
    for c in chunks:
        parts.append(
            f"[{c.get('source_document', 'unknown')}, Page {c.get('page_number', '?')}]\n"
            f"{c.get('text', '')}"
        )
    return "\n\n".join(parts)


def search_docs(query: str) -> list[dict[str, Any]]:
    """Run one hybrid retrieval at production settings.

    Deliberately the same call the default pipeline makes, so a difference
    between the two paths is the model's querying, not a different retriever.
    """
    return get_retriever().retrieve(
        query, top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K
    )


def run_tool_loop(question: str, max_calls: int = MAX_TOOL_CALLS) -> dict[str, Any]:
    """Let the model search until it can answer, bounded by max_calls.

    Returns the answer, every chunk seen across all searches, the queries the
    model chose, and the number of searches it actually used.
    """
    llm = get_llm()
    bound = llm.bind_tools([SEARCH_DOCS_TOOL])

    messages: list[Any] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT.format(max_calls=max_calls)},
        {"role": "user", "content": question},
    ]

    evidence: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    queries: list[str] = []

    for turn in range(max_calls + 1):
        # On the final turn the tool is withdrawn, which forces an answer
        # instead of relying on the model to notice its own budget.
        active = bound if turn < max_calls else llm
        response = invoke_messages(active, messages, stage="tool_agent")

        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            return {
                "answer": response_text(response),
                "evidence": evidence,
                "queries": queries,
                "tool_calls": len(queries),
            }

        messages.append(response)
        for call in calls:
            q = (call.get("args") or {}).get("query", "")
            queries.append(q)
            logger.info("tool search %d: %s", len(queries), q[:80])
            chunks = search_docs(q) if q else []
            for c in chunks:
                cid = str(c.get("id") or f"{c.get('source_document')}#{c.get('page_number')}")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    evidence.append(c)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": _format_results(chunks),
                }
            )

    # Unreachable in practice: the final turn has no tool to call.
    return {
        "answer": response_text(response),
        "evidence": evidence,
        "queries": queries,
        "tool_calls": len(queries),
    }
