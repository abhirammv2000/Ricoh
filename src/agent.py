"""
src/agent.py - LangGraph Agentic State Machine.

Orchestrates a
Plan → Retrieve → Verify → Synthesize loop that:

1. Plans - decomposes a user question into focused sub-queries
   and extracts key entities (error codes, model numbers).
2. Retrieves - calls the HybridRetriever for each sub-query.
3. Verifies - asks the LLM whether the retrieved evidence is
   sufficient to answer the question definitively.
4. Synthesises - generates a grounded answer with strict
   page-level citations in [Document Name, Page X] format.

The agentic loop allows one retry: if the Verifier says
"INSUFFICIENT" and we haven't exhausted iterations, we loop
back to the Planner to broaden the search.

Design decisions
- We use LangGraph's StateGraph for explicit, auditable control
  flow - no hidden chains or prompt-chaining magic.
- ``iterations`` is capped at 2 to prevent runaway API costs.
- Temperature is 0.0 to reduce variance, NOT to make output
  deterministic, which temperature 0 has never guaranteed. The
  planner's sub-queries do vary run to run; retrieval does not.
- All prompts are defined as module-level constants for easy tuning.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.config import (  # noqa: F401 - triggers config.py logging setup
    RETRIEVAL_FINAL_K,
    RETRIEVAL_TOP_K,
    USE_PLANNER,
    USE_TOOL_LOOP,
    USE_VERIFIER,
)
from src.instrumentation import invoke as instrumented_invoke
from src.instrumentation import span
from src.llm_factory import get_llm
from src.retriever import get_retriever
from src.semantic_cache import get_semantic_cache

# Logging is configured centrally in config.py.
logger = logging.getLogger(__name__)

# Constants.
MAX_ITERATIONS: int = 2  # hard cap on agentic retries


# 1. STATE DEFINITION

class AgentState(TypedDict):
    """Typed state that flows through every node in the graph.

    Keeping it as a TypedDict lets LangGraph serialise / inspect
    the state at every step - invaluable for debugging.
    """
    user_query: str                          # original question
    sub_queries: list[str]                   # decomposed sub-questions
    entities: list[str]                      # error codes, model numbers, part names
    retrieved_evidence: list[dict[str, Any]] # chunks + metadata
    verification_status: str                 # "SUFFICIENT" | "INSUFFICIENT"
    final_answer: str                        # synthesised response
    iterations: int                          # loop counter (max 2)


# 2. PROMPT TEMPLATES

PLANNER_PROMPT = """\
You are a query planner for a Ricoh technical support system.

Given the user's question, do TWO things:
1. Extract any specific entities (error codes like SC542, model \
numbers like IM C3500, part names like "fusing unit").
2. Break the question into a list of simpler, focused sub-queries \
that can each be answered independently.  If the question is already \
simple, return a list with just that one query.

User question:
\"\"\"{user_query}\"\"\"

{retry_context}

Respond with ONLY a valid JSON object in this exact format - no \
markdown fences, no extra text:
{{
  "entities": ["entity1", "entity2"],
  "sub_queries": ["sub-query 1", "sub-query 2"]
}}
"""

VERIFIER_PROMPT = """\
You are an evidence verifier for a Ricoh technical support system.

User question:
\"\"\"{user_query}\"\"\"

Retrieved evidence:
{evidence_block}

Task: Determine whether the retrieved evidence contains enough \
information to definitively answer the user's question without \
guessing.  Consider whether specific steps, values, or procedures \
mentioned in the question are covered.

Respond with EXACTLY one word - either SUFFICIENT or INSUFFICIENT.
"""

SYNTHESIZER_PROMPT = """\
You are a senior Ricoh technical support engineer.  Answer the \
user's question using ONLY the evidence provided below.

Rules:
1. Detect the language of the user's input query. You MUST generate \
the Final Answer in that SAME language. However, keep the citation \
tags [Document Name, Page X] exactly as they are (do not translate \
filenames or citation format).
2. Provide clear, step-by-step instructions where applicable.
3. For EVERY factual claim, append a citation in this exact format: \
[Document Name, Page X].
4. If the evidence does not contain the answer, your reply MUST contain \
the following sentence verbatim, in English, exactly as written:
"Information unavailable in provided documents."
You may add a translation of that sentence into the user's language \
immediately afterwards, but the English sentence itself must always be \
present and unaltered, it is the machine-readable refusal marker. Never \
emit this sentence when the evidence does let you answer.
5. Do NOT invent information.  Do NOT guess.

User question:
\"\"\"{user_query}\"\"\"

Evidence:
{evidence_block}

Answer:
"""


# 3. HELPER - format evidence for prompts

def _format_evidence_block(evidence: list[dict[str, Any]]) -> str:
    """Render evidence chunks into a numbered text block for prompts."""
    if not evidence:
        return "(no evidence retrieved)"

    lines: list[str] = []
    for i, e in enumerate(evidence, 1):
        source = e.get("source_document", "unknown")
        page = e.get("page_number", "?")
        text = e.get("text", "").strip()
        lines.append(
            f"[{i}] Source: {source}, Page {page}\n{text}\n"
        )
    return "\n".join(lines)


# 4. GRAPH NODES

def planner_node(state: AgentState) -> dict[str, Any]:
    """Node 1 - Decompose the user query into sub-queries.

    On a retry (iterations > 0), we inject context about what has
    already been retrieved so the LLM can broaden its search.
    """
    llm = get_llm()

    # Build retry context if this is a second pass
    retry_context = ""
    if state["iterations"] > 0:
        already = set(
            e.get("source_document", "") for e in state["retrieved_evidence"]
        )
        retry_context = (
            "IMPORTANT: A previous retrieval attempt was INSUFFICIENT. "
            "The following sources were already searched: "
            f"{', '.join(already)}. "
            "Generate NEW, BROADER sub-queries that search for "
            "different angles or related topics."
        )

    prompt = PLANNER_PROMPT.format(
        user_query=state["user_query"],
        retry_context=retry_context,
    )

    content = instrumented_invoke(llm, prompt, stage="planner")

    # Parse the JSON response. Strip markdown fences if the model wraps them
    # despite the instructions.
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    try:
        parsed = json.loads(content)
        sub_queries = parsed.get("sub_queries", [state["user_query"]])
        entities = parsed.get("entities", [])
    except (json.JSONDecodeError, KeyError):
        logger.warning("Planner JSON parse failed. Using raw query.")
        sub_queries = [state["user_query"]]
        entities = []

    logger.info("Planner entities: %s", entities)
    logger.info("Planner sub-queries: %s", sub_queries)

    # Pretty-print for terminal visibility.
    print(f"\nPLANNER - Iteration {state['iterations'] + 1}")
    print(f"   Entities : {entities}")
    print(f"   Sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"     {i}. {sq}")

    return {"sub_queries": sub_queries, "entities": entities}


def retriever_node(state: AgentState) -> dict[str, Any]:
    """Node 2 - Run hybrid retrieval with TWO passes.

    Pass 1: Search using the sub-queries from the Planner.
    Pass 2: Search using entity-boosted refined queries
            (error codes, model numbers, part names).

    De-duplicates results by chunk ID across both passes.
    """
    retriever = get_retriever()

    # Collect existing IDs to avoid duplicates
    seen_ids: set[str] = {
        e["id"] for e in state["retrieved_evidence"] if "id" in e
    }
    new_evidence: list[dict[str, Any]] = list(state["retrieved_evidence"])

    # Pass 1: sub-query retrieval.
    for sq in state["sub_queries"]:
        results = retriever.retrieve(
            query=sq,
            top_k=RETRIEVAL_TOP_K,
            final_k=RETRIEVAL_FINAL_K,
        )
        for r in results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                new_evidence.append(r)

    pass1_count = len(new_evidence)
    print(f"\nRETRIEVER - Pass 1 (sub-queries): {pass1_count} chunks")

    # Pass 2: entity-boosted refined queries.
    entities = state.get("entities", [])
    if entities:
        for entity in entities:
            # Refine: combine entity with original question for context
            refined_query = f"{entity} {state['user_query']}"
            results = retriever.retrieve(
                query=refined_query,
                top_k=RETRIEVAL_TOP_K,
                final_k=RETRIEVAL_FINAL_K,
            )
            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    new_evidence.append(r)

        pass2_new = len(new_evidence) - pass1_count
        print(f"RETRIEVER - Pass 2 (entities: {entities}): +{pass2_new} new chunks")

    print(f"RETRIEVER - Total: {len(new_evidence)} unique evidence chunks")

    return {
        "retrieved_evidence": new_evidence,
        "iterations": state["iterations"] + 1,
    }


def verifier_node(state: AgentState) -> dict[str, Any]:
    """Node 3 - Check whether evidence is sufficient.

    Forces the LLM to output strictly "SUFFICIENT" or "INSUFFICIENT".
    """
    llm = get_llm()

    evidence_block = _format_evidence_block(state["retrieved_evidence"])
    prompt = VERIFIER_PROMPT.format(
        user_query=state["user_query"],
        evidence_block=evidence_block,
    )

    verdict = instrumented_invoke(llm, prompt, stage="verifier").upper()

    # Normalise: accept partial matches
    if "SUFFICIENT" in verdict and "INSUFFICIENT" not in verdict:
        status = "SUFFICIENT"
    elif "INSUFFICIENT" in verdict:
        status = "INSUFFICIENT"
    else:
        # Default to sufficient to avoid infinite loops
        logger.warning("Verifier gave unexpected output: '%s'", verdict)
        status = "SUFFICIENT"

    print(f"\nVERIFIER - {status} (iteration {state['iterations']})")

    return {"verification_status": status}


def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """Node 4 - Generate a grounded answer with strict citations.

    This is the final node.  It produces the user-facing answer
    with [Document Name, Page X] citations.
    """
    llm = get_llm()

    evidence_block = _format_evidence_block(state["retrieved_evidence"])
    prompt = SYNTHESIZER_PROMPT.format(
        user_query=state["user_query"],
        evidence_block=evidence_block,
    )

    answer = instrumented_invoke(llm, prompt, stage="synthesizer")

    print(f"\nSYNTHESIZER - Answer generated ({len(answer)} chars)")

    return {"final_answer": answer}


# 5. CONDITIONAL EDGE - Verifier routing logic

def should_retry_or_synthesize(state: AgentState) -> str:
    """Decide whether to loop back to the planner or move on.

    Returns "planner" if the evidence was INSUFFICIENT and we are still under
    MAX_ITERATIONS, otherwise "synthesizer" (evidence was sufficient, or we
    have used up our retries).
    """
    if (
        state["verification_status"] == "INSUFFICIENT"
        and state["iterations"] < MAX_ITERATIONS
    ):
        print(f"\nROUTING back to PLANNER (retry)")
        return "planner"

    if state["iterations"] >= MAX_ITERATIONS:
        print(f"\nROUTING to SYNTHESIZER (max iterations reached)")
    else:
        print(f"\nROUTING to SYNTHESIZER (evidence sufficient)")
    return "synthesizer"


# 6. GRAPH ASSEMBLY

def build_agent_graph(
    use_planner: bool = True,
    use_verifier: bool = True,
) -> Any:
    """Construct and compile the LangGraph state machine.

    With both flags on (the production default) the flow is: planner, then
    retriever, then verifier, which either loops back to the planner (when the
    evidence is insufficient and we are under 2 iterations) or goes on to the
    synthesizer and then END.

    Why the flags exist. Every node here costs an LLM call, and anything that
    costs money has to earn it. These flags run the same code path as a reduced
    pipeline, so each stage's contribution can be measured by removing it rather
    than argued about:

        use_planner=False, use_verifier=False   retrieve then synthesize
        use_planner=True,  use_verifier=False   adds query decomposition
        use_planner=True,  use_verifier=True    adds verify and retry (prod)

    This is the real graph rather than a separate "simple pipeline", on purpose:
    a benchmark that compares two different code paths measures the difference
    between the implementations as much as the difference between the designs.

    Args:
        use_planner:  Run the query-decomposition node. When False, the caller
                      has to seed sub_queries with the raw question (run_agent
                      and the harness both do this).
        use_verifier: Run the sufficiency check and the retry loop.

    Note: with use_planner=False there is no retry target, since the retry edge
    loops back to the planner, so the verifier (if enabled) always routes
    forward to the synthesizer. That combination measures the verifier's cost
    without its retry benefit, and is reported as such.
    """
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("synthesizer", synthesizer_node)

    if use_planner:
        graph.add_node("planner", planner_node)
        graph.set_entry_point("planner")
        graph.add_edge("planner", "retriever")
    else:
        graph.set_entry_point("retriever")

    if use_verifier:
        graph.add_node("verifier", verifier_node)
        graph.add_edge("retriever", "verifier")
        if use_planner:
            graph.add_conditional_edges(
                "verifier",
                should_retry_or_synthesize,
                {"planner": "planner", "synthesizer": "synthesizer"},
            )
        else:
            graph.add_edge("verifier", "synthesizer")
    else:
        graph.add_edge("retriever", "synthesizer")

    graph.add_edge("synthesizer", END)

    return graph.compile()


# The compiled graph is stateless (all run state is passed to .invoke()), so it
# can be built once and reused across queries instead of recompiling on every
# call. It is cached at module level for that reason, keyed by (use_planner,
# use_verifier) so an ablation run does not silently reuse a graph compiled for
# a different configuration, which would make every ablation result identical
# and look like the flags had no effect.
_COMPILED_GRAPHS: dict[tuple[bool, bool], Any] = {}


def get_agent_graph(
    use_planner: bool | None = None,
    use_verifier: bool | None = None,
) -> Any:
    """Return a process-wide compiled agent graph for this configuration.

    Defaults come from ``config.USE_PLANNER`` / ``config.USE_VERIFIER`` so
    production composition is a documented configuration decision backed by
    the ablation, not a hard-coded literal buried in a call site.
    """
    use_planner = USE_PLANNER if use_planner is None else use_planner
    use_verifier = USE_VERIFIER if use_verifier is None else use_verifier
    key = (use_planner, use_verifier)
    if key not in _COMPILED_GRAPHS:
        _COMPILED_GRAPHS[key] = build_agent_graph(
            use_planner=use_planner, use_verifier=use_verifier
        )
    return _COMPILED_GRAPHS[key]


# 7. PUBLIC API - convenience runner

def initial_state(query: str, use_planner: bool | None = None) -> AgentState:
    """Build the starting state for a run.

    When the planner is disabled nothing else will populate ``sub_queries``,
    so it is seeded with the raw question.  That is exactly what the planner
    already falls back to when its JSON fails to parse, so the disabled path
    is a real configuration rather than a special case.
    """
    use_planner = USE_PLANNER if use_planner is None else use_planner
    return {
        "user_query": query,
        "sub_queries": [] if use_planner else [query],
        "entities": [],
        "retrieved_evidence": [],
        "verification_status": "",
        "final_answer": "",
        "iterations": 0,
    }


def run_agent(
    query: str,
    use_planner: bool | None = None,
    use_verifier: bool | None = None,
) -> str:
    """Run the agentic pipeline on a single query.

    Args:
        query:        Natural-language technical support question.
        use_planner:  See :func:`build_agent_graph`.
        use_verifier: See :func:`build_agent_graph`.

    Returns:
        The final synthesised answer string with citations.

    When the semantic cache is enabled (off by default), an identical or
    near-duplicate query returns the stored answer without running the pipeline.
    The hit is recorded as a zero-cost span, so a trace shows the saving. The
    eval harness never reaches this path, so cache hits cannot affect measured
    numbers.
    """
    cache = get_semantic_cache()
    if cache is not None:
        hit = cache.lookup(query)
        if hit is not None:
            with span("cache", cache_hit=True, kind=hit.kind, similarity=hit.similarity):
                pass
            return hit.answer

    # Tool-calling path (USE_TOOL_LOOP, off by default). The model runs its own
    # searches instead of taking one fixed retrieval, so it bypasses the graph
    # rather than forming a node inside it: the whole point is that the control
    # flow is the model's, not the graph's.
    if USE_TOOL_LOOP:
        from src.tools import run_tool_loop

        answer = run_tool_loop(query)["answer"]
        if cache is not None:
            cache.store(query, answer)
        return answer

    agent = get_agent_graph(use_planner=use_planner, use_verifier=use_verifier)
    final_state = agent.invoke(initial_state(query, use_planner=use_planner))
    answer = final_state["final_answer"]

    if cache is not None:
        cache.store(query, answer)
    return answer


@dataclass
class StreamResult:
    """Carries the final state and timing out of stream_agent.

    A generator cannot easily return a value next to the items it yields, so the
    caller passes one of these in and reads it once the stream is exhausted.
    """

    final_state: dict[str, Any] | None = None
    ttft_seconds: float | None = None
    total_seconds: float | None = None


def _chunk_text(message: Any) -> str:
    """Text of one streamed chunk, without stripping.

    This deliberately does not strip, unlike response_text. A chunk boundary
    often falls on a space, so stripping each chunk would delete the spaces
    between words once the chunks are joined back together.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def stream_agent(
    query: str,
    result: StreamResult,
    use_planner: bool | None = None,
    use_verifier: bool | None = None,
) -> Iterator[str]:
    """Run the agent and stream the synthesizer's answer token by token.

    Yields answer text as it is generated. The upstream stages (planning,
    retrieval, verification) run first and do not stream, since the synthesizer
    output is the only text the user reads. We stream only the synthesizer node,
    filtered by the node name in the message metadata, so enabling the planner
    or verifier does not leak their internal LLM output into the answer.

    Grounding is not weakened by streaming: verification already happened
    upstream and gates whether we synthesize at all, so streaming the final
    answer only changes how it is delivered, not what it is based on.

    Once the generator is exhausted, `result` holds the full final state, the
    time to first token, and the total time.
    """
    graph = get_agent_graph(use_planner=use_planner, use_verifier=use_verifier)
    init = initial_state(query, use_planner=use_planner)

    started = time.perf_counter()
    for mode, data in graph.stream(init, stream_mode=["messages", "values"]):
        if mode == "messages":
            message, meta = data
            if meta.get("langgraph_node") == "synthesizer":
                text = _chunk_text(message)
                if text:
                    if result.ttft_seconds is None:
                        result.ttft_seconds = time.perf_counter() - started
                    yield text
        elif mode == "values":
            result.final_state = dict(data)
    result.total_seconds = time.perf_counter() - started


# __main__ - Smoke test with a complex multi-part question

if __name__ == "__main__":
    import sys

    from src.ingest import ingest_all
    from src.retriever import HybridRetriever

    print("=" * 70)
    print("  Citera agent smoke test")
    print("=" * 70)

    # Step 1: make sure the retrieval index is populated.
    print("\nChecking/building retrieval index…")
    retriever = HybridRetriever()

    if retriever.index_size == 0 or not retriever.bm25_ready:
        reason = "empty" if retriever.index_size == 0 else "BM25 missing"
        print(f"   Index needs (re)build ({reason}) - ingesting PDFs…")
        chunks = ingest_all()
        if not chunks:
            print("No PDFs found in data/. Add PDFs and retry.")
            sys.exit(1)
        retriever.build_index(chunks)
        print(f"   Index built: {retriever.index_size} docs, BM25: {retriever.bm25_ready}.")
    else:
        print(f"   Index already populated: {retriever.index_size} docs, BM25: ready.")

    # Step 2: run the agent on a complex multi-part question.
    test_query = (
        "How do I configure network settings and "
        "what paper does the bypass tray take?"
    )

    print(f"\n{'=' * 70}")
    print(f"  USER QUERY: {test_query}")
    print(f"{'=' * 70}")

    answer = run_agent(test_query)

    print(f"\n{'=' * 70}")
    print("  FINAL ANSWER")
    print(f"{'=' * 70}")
    print(answer)
    print(f"\n{'=' * 70}")
    print("Smoke test complete.")
    print(f"{'=' * 70}")
