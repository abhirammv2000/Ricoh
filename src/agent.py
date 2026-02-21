"""
src/agent.py - LangGraph Agentic State Machine.

This is the **brain** of RicohLibrary.  It orchestrates a
Plan → Retrieve → Verify → Synthesize loop that:

1. **Plans** - decomposes a user question into focused sub-queries
   and extracts key entities (error codes, model numbers).
2. **Retrieves** - calls the HybridRetriever for each sub-query.
3. **Verifies** - asks the LLM whether the retrieved evidence is
   sufficient to answer the question definitively.
4. **Synthesises** - generates a grounded answer with strict
   page-level citations in [Document Name, Page X] format.

The agentic loop allows one retry: if the Verifier says
"INSUFFICIENT" and we haven't exhausted iterations, we loop
back to the Planner to broaden the search.

Design decisions
────────────────
• We use LangGraph's StateGraph for explicit, auditable control
  flow - no hidden chains or prompt-chaining magic.
• ``iterations`` is capped at 2 to prevent runaway API costs.
• Temperature is 0.0 everywhere for deterministic, factual output.
• All prompts are defined as module-level constants for easy tuning.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.config import RETRIEVAL_FINAL_K, RETRIEVAL_TOP_K  # noqa: F401 - triggers config.py logging setup
from src.llm_factory import get_llm
from src.retriever import HybridRetriever

# ── Logging (configured centrally in config.py) ──
logger = logging.getLogger(__name__)

# ── Constants ──
MAX_ITERATIONS: int = 2  # hard cap on agentic retries


# ====================================================================
# 1. STATE DEFINITION
# ====================================================================

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


# ====================================================================
# 2. PROMPT TEMPLATES
# ====================================================================

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
information to **definitively** answer the user's question without \
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
4. If the evidence does not contain the answer, state: \
"Information unavailable in provided documents." (in the user's language).
5. Do NOT invent information.  Do NOT guess.

User question:
\"\"\"{user_query}\"\"\"

Evidence:
{evidence_block}

Answer:
"""


# ====================================================================
# 3. HELPER - format evidence for prompts
# ====================================================================

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


# ====================================================================
# 4. GRAPH NODES
# ====================================================================

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

    response = llm.invoke(prompt)
    content = response.content.strip()

    # ── Parse the JSON response ──
    # Strip markdown fences if the LLM wraps them despite instructions
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

    # ── Pretty-print for terminal visibility ──
    print(f"\n🧠  PLANNER - Iteration {state['iterations'] + 1}")
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
    retriever = HybridRetriever()

    # Collect existing IDs to avoid duplicates
    seen_ids: set[str] = {
        e["id"] for e in state["retrieved_evidence"] if "id" in e
    }
    new_evidence: list[dict[str, Any]] = list(state["retrieved_evidence"])

    # ── Pass 1: Sub-query retrieval ──
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
    print(f"\n📚  RETRIEVER - Pass 1 (sub-queries): {pass1_count} chunks")

    # ── Pass 2: Entity-boosted refined queries ──
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
        print(f"📚  RETRIEVER - Pass 2 (entities: {entities}): +{pass2_new} new chunks")

    print(f"📚  RETRIEVER - Total: {len(new_evidence)} unique evidence chunks")

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

    response = llm.invoke(prompt)
    verdict = response.content.strip().upper()

    # Normalise: accept partial matches
    if "SUFFICIENT" in verdict and "INSUFFICIENT" not in verdict:
        status = "SUFFICIENT"
    elif "INSUFFICIENT" in verdict:
        status = "INSUFFICIENT"
    else:
        # Default to sufficient to avoid infinite loops
        logger.warning("Verifier gave unexpected output: '%s'", verdict)
        status = "SUFFICIENT"

    print(f"\n✅  VERIFIER - {status} (iteration {state['iterations']})")

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

    response = llm.invoke(prompt)
    answer = response.content.strip()

    print(f"\n💬  SYNTHESIZER - Answer generated ({len(answer)} chars)")

    return {"final_answer": answer}


# ====================================================================
# 5. CONDITIONAL EDGE - Verifier routing logic
# ====================================================================

def should_retry_or_synthesize(state: AgentState) -> str:
    """Decide whether to loop back to the Planner or proceed.

    Routes to:
    • ``"planner"`` if INSUFFICIENT *and* iterations < MAX_ITERATIONS
    • ``"synthesizer"`` otherwise (sufficient or exhausted retries)
    """
    if (
        state["verification_status"] == "INSUFFICIENT"
        and state["iterations"] < MAX_ITERATIONS
    ):
        print(f"\n🔄  ROUTING → back to PLANNER (retry)")
        return "planner"

    if state["iterations"] >= MAX_ITERATIONS:
        print(f"\n⚠️   ROUTING → SYNTHESIZER (max iterations reached)")
    else:
        print(f"\n✅  ROUTING → SYNTHESIZER (evidence sufficient)")
    return "synthesizer"


# ====================================================================
# 6. GRAPH ASSEMBLY
# ====================================================================

def build_agent_graph() -> Any:
    """Construct and compile the LangGraph state machine.

    Graph topology::

        START → planner → retriever → verifier ─┬─→ synthesizer → END
                   ↑                             │
                   └──── (INSUFFICIENT & <2) ────┘
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("synthesizer", synthesizer_node)

    # ── Add edges ──
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "verifier")

    # Conditional edge: verifier → planner (retry) OR synthesizer
    graph.add_conditional_edges(
        "verifier",
        should_retry_or_synthesize,
        {
            "planner": "planner",
            "synthesizer": "synthesizer",
        },
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()


# ====================================================================
# 7. PUBLIC API - convenience runner
# ====================================================================

def run_agent(query: str) -> str:
    """Run the full agentic pipeline on a single query.

    Args:
        query: Natural-language technical support question.

    Returns:
        The final synthesised answer string with citations.
    """
    agent = build_agent_graph()

    initial_state: AgentState = {
        "user_query": query,
        "sub_queries": [],
        "entities": [],
        "retrieved_evidence": [],
        "verification_status": "",
        "final_answer": "",
        "iterations": 0,
    }

    # LangGraph's invoke runs the graph to completion
    final_state = agent.invoke(initial_state)
    return final_state["final_answer"]


# ====================================================================
# __main__ - Smoke test with a complex multi-part question
# ====================================================================

if __name__ == "__main__":
    import sys

    from src.ingest import ingest_all
    from src.retriever import HybridRetriever

    print("=" * 70)
    print("  RicohLibrary - Phase 3 Agent Smoke Test")
    print("=" * 70)

    # ── Step 1: Ensure the retrieval index is populated ──
    print("\n📄  Checking/building retrieval index…")
    retriever = HybridRetriever()

    if retriever.index_size == 0 or not retriever.bm25_ready:
        reason = "empty" if retriever.index_size == 0 else "BM25 missing"
        print(f"   Index needs (re)build ({reason}) - ingesting PDFs…")
        chunks = ingest_all()
        if not chunks:
            print("❌  No PDFs found in data/. Add PDFs and retry.")
            sys.exit(1)
        retriever.build_index(chunks)
        print(f"   Index built: {retriever.index_size} docs, BM25: {retriever.bm25_ready}.")
    else:
        print(f"   Index already populated: {retriever.index_size} docs, BM25: ready.")

    # ── Step 2: Run the agent on a complex test question ──
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
    print("Phase 3 smoke test complete. Ready for Phase 4!")
    print(f"{'=' * 70}")
