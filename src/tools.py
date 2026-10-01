"""Tool-calling retrieval: the model runs its own searches.

The default pipeline retrieves once and answers, and recall@5 is 0.94 on the 100 questions. The 6
misses get one set of chunks and no way to ask again. Here search_docs is a tool, so the model picks
how many searches to run and what to search for, and sees the results before deciding to search again.

It helped more than I expected. Four hand-written rewrites of each failing question recovered 2 of
the 6, which I took as the ceiling. The model recovered 4, and lost none of a 20 question sample that
already passed, at 1.77x the cost ($0.0278 vs $0.0157). That only shows the right document reached the
model, not that answers got better, so it stays off (USE_TOOL_LOOP) until a judged run at n=100.

Searches are capped by MAX_TOOL_CALLS by taking the tool away, not by asking the model to stop. Every
turn records a span through invoke_messages(), and evidence builds up across searches.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K
from src.instrumentation import invoke_messages
from src.llm_factory import get_llm, response_text
from src.retriever import get_retriever

logger = logging.getLogger(__name__)

# cap on searches per question. The model used 1 to 3 in the measured runs, so this only limits the
# worst case. Without it a $0.016 query could become a $0.50 one
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
    """The chunks formatted the way the model should cite them."""
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
    """One hybrid retrieval with the production settings, the same call the default pipeline makes."""
    return get_retriever().retrieve(
        query, top_k=RETRIEVAL_TOP_K, final_k=RETRIEVAL_FINAL_K
    )


def run_tool_loop(question: str, max_calls: int = MAX_TOOL_CALLS) -> dict[str, Any]:
    """Let the model search until it can answer, up to max_calls.

    Returns the answer, every chunk seen, the queries it chose and how many searches it used.
    """
    llm = get_llm()
    bound = llm.bind_tools([SEARCH_DOCS_TOOL])

    messages: list[Any] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT.format(max_calls=max_calls)},
        {"role": "user", "content": question},
    ]

    # imported here, not at the top, since agent.py also imports this module lazily
    from src.agent import record_citation_guardrail

    evidence: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    queries: list[str] = []

    def _result(response: Any) -> dict[str, Any]:
        answer = response_text(response)
        record_citation_guardrail(answer, evidence)
        return {
            "answer": answer,
            "evidence": evidence,
            "queries": queries,
            "tool_calls": len(queries),
        }

    for turn in range(max_calls + 1):
        # on the last turn the tool is taken away, which forces an answer
        active = bound if turn < max_calls else llm
        response = invoke_messages(active, messages, stage="tool_agent")

        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            return _result(response)

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

    # shouldn't get here, the last turn has no tool
    return _result(response)
