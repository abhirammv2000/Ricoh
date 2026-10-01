"""LangGraph agent: plan, retrieve, verify, synthesize.

The planner and verifier are off by default, so the normal path is just retrieve then
synthesize (see build_agent_graph and src/config.py). With both on, an insufficient
verdict sends the graph back to the planner once. The prompts are module-level constants.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

from src.config import (  # noqa: F401 - triggers config.py logging setup
    RETRIEVAL_FINAL_K,
    RETRIEVAL_TOP_K,
    USE_PLANNER,
    USE_ROUTER,
    USE_TOOL_LOOP,
    USE_VERIFIER,
)
from src.conversation import Turn, condense_query  # noqa: F401 - Turn re-exported for callers
from src.instrumentation import ainvoke as instrumented_ainvoke
from src.instrumentation import invoke as instrumented_invoke
from src.instrumentation import span
from src.llm_factory import get_llm
from src.retriever import get_retriever
from src.semantic_cache import get_semantic_cache

logger = logging.getLogger(__name__)

MAX_ITERATIONS: int = 2  # cap on planner retries


# state

class AgentState(TypedDict):
    """What gets passed between the nodes."""
    user_query: str                          # original question
    sub_queries: list[str]                   # decomposed sub-questions
    entities: list[str]                      # error codes, model numbers, part names
    retrieved_evidence: list[dict[str, Any]] # chunks + metadata
    verification_status: str                 # "SUFFICIENT" | "INSUFFICIENT"
    final_answer: str                        # synthesised response
    iterations: int                          # loop counter (max 2)


# prompts

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


# the synthesizer says this (rule 4 of its prompt) when it can't answer. The harness and the
# router look for it, ignoring case and whitespace.
REFUSAL_MARKER = "information unavailable"


def is_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in " ".join(answer.lower().split())


# matches the [Document Name, Page X] citations the prompt asks for (eval_harness imports it)
CITATION_RE = re.compile(r"\[([^\]]+?),\s*Page\s*\d+\]", re.IGNORECASE)


def cited_docs(answer: str) -> set[str]:
    """Distinct document names the answer cites, by the [Doc, Page X] format."""
    return {m.strip() for m in CITATION_RE.findall(answer)}


def record_citation_guardrail(answer: str, evidence: list[dict[str, Any]]) -> None:
    """Log whether every cited document was actually retrieved.

    It only records the result and never edits the answer. Citation precision was 1.00 on the
    100-question benchmark, so this hasn't fired yet. It costs a regex and a set comparison.
    """
    cited = cited_docs(answer)
    if not cited:
        return  # a refusal cites nothing
    evidence_docs = {e.get("source_document", "") for e in evidence if e.get("source_document")}
    fabricated = sorted(cited - evidence_docs)
    with span(
        "citation_guardrail",
        cited=sorted(cited),
        fabricated=fabricated,
        valid=not fabricated,
    ):
        pass
    if fabricated:
        logger.warning("Answer cites document(s) not in evidence: %s", fabricated)


# helpers

def _format_evidence_block(evidence: list[dict[str, Any]]) -> str:
    """Numbered text block of the evidence, for the prompts."""
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


# graph nodes

class PlannerOutput(BaseModel):
    """The shape the planner's JSON has to have.

    Valid json isn't enough: a string where the list should be would be searched one letter at
    a time. Checking the shape sends that to the same fallback as broken json.
    """

    sub_queries: list[str]
    entities: list[str] = []


def planner_node(state: AgentState) -> dict[str, Any]:
    """Split the question into sub-queries and pull out entities like error codes.

    On a retry it tells the model which sources were already searched.
    """
    llm = get_llm()

    # on a retry, say what was already searched
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

    # strip markdown fences if the model adds them anyway
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    try:
        parsed = PlannerOutput.model_validate_json(content)
        sub_queries = parsed.sub_queries or [state["user_query"]]
        entities = parsed.entities
    except ValidationError as exc:
        # covers bad json and valid json with the wrong shape
        logger.warning("Planner output failed validation (%s). Using raw query.", exc)
        sub_queries = [state["user_query"]]
        entities = []

    logger.info("Planner entities: %s", entities)
    logger.info("Planner sub-queries: %s", sub_queries)

    # show progress in the terminal
    print(f"\nPLANNER - Iteration {state['iterations'] + 1}")
    print(f"   Entities : {entities}")
    print(f"   Sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"     {i}. {sq}")

    return {"sub_queries": sub_queries, "entities": entities}


def retriever_node(state: AgentState) -> dict[str, Any]:
    """Search every sub-query, then every entity combined with the question, without duplicates."""
    retriever = get_retriever()

    # skip chunks we already have
    seen_ids: set[str] = {
        e["id"] for e in state["retrieved_evidence"] if "id" in e
    }
    new_evidence: list[dict[str, Any]] = list(state["retrieved_evidence"])

    # first pass: the sub-queries
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

    # second pass: each entity plus the original question
    entities = state.get("entities", [])
    if entities:
        for entity in entities:
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
    """Ask whether the evidence is enough to answer: SUFFICIENT or INSUFFICIENT."""
    llm = get_llm()

    evidence_block = _format_evidence_block(state["retrieved_evidence"])
    prompt = VERIFIER_PROMPT.format(
        user_query=state["user_query"],
        evidence_block=evidence_block,
    )

    verdict = instrumented_invoke(llm, prompt, stage="verifier").upper()

    # accept partial matches
    if "SUFFICIENT" in verdict and "INSUFFICIENT" not in verdict:
        status = "SUFFICIENT"
    elif "INSUFFICIENT" in verdict:
        status = "INSUFFICIENT"
    else:
        # default to sufficient so we can't loop forever
        logger.warning("Verifier gave unexpected output: '%s'", verdict)
        status = "SUFFICIENT"

    print(f"\nVERIFIER - {status} (iteration {state['iterations']})")

    return {"verification_status": status}


def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """Write the final answer with [Document Name, Page X] citations."""
    llm = get_llm()

    evidence_block = _format_evidence_block(state["retrieved_evidence"])
    prompt = SYNTHESIZER_PROMPT.format(
        user_query=state["user_query"],
        evidence_block=evidence_block,
    )

    answer = instrumented_invoke(llm, prompt, stage="synthesizer")
    record_citation_guardrail(answer, state["retrieved_evidence"])

    print(f"\nSYNTHESIZER - Answer generated ({len(answer)} chars)")

    return {"final_answer": answer}


# routing after the verifier

def should_retry_or_synthesize(state: AgentState) -> str:
    """Back to the planner if the evidence was insufficient and retries are left, else synthesize."""
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


# graph assembly

def build_agent_graph(
    use_planner: bool = True,
    use_verifier: bool = True,
) -> Any:
    """Build and compile the graph.

    The flags make the same code run as a smaller pipeline, so the ablation can measure a stage
    by turning it off. Both are off in production. use_planner=False means the caller seeds
    sub_queries with the raw question, and the verifier then always goes forward, because the
    retry edge points at the planner.
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


# the compiled graph holds no run state, so build it once per (use_planner, use_verifier)
# and reuse it. Keyed on both so an ablation never gets a graph built for another setting.
_compiled_graphs: dict[tuple[bool, bool], Any] = {}


def get_agent_graph(
    use_planner: bool | None = None,
    use_verifier: bool | None = None,
) -> Any:
    """The compiled graph for this configuration, built once. Defaults come from config."""
    use_planner = USE_PLANNER if use_planner is None else use_planner
    use_verifier = USE_VERIFIER if use_verifier is None else use_verifier
    key = (use_planner, use_verifier)
    if key not in _compiled_graphs:
        _compiled_graphs[key] = build_agent_graph(
            use_planner=use_planner, use_verifier=use_verifier
        )
    return _compiled_graphs[key]


# public api

def initial_state(query: str, use_planner: bool | None = None) -> AgentState:
    """Starting state for a run. Without the planner, sub_queries is just the raw question."""
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
    history: Sequence[Turn] | None = None,
) -> str:
    """Answer one question with the configured pipeline.

    history is the earlier turns. If given, the question is first rewritten to stand on its own
    (src/conversation.py), and everything after that, the cache included, sees the rewrite. With the
    semantic cache on, a near-duplicate question returns the stored answer. The eval harness never
    comes through here, so a cache hit can't change a measured number.
    """
    if history:
        query = condense_query(history, query)

    cache = get_semantic_cache()
    if cache is not None:
        hit = cache.lookup(query)
        if hit is not None:
            with span("cache", cache_hit=True, kind=hit.kind, similarity=hit.similarity):
                pass
            return hit.answer

    # router: cheap path first, tool loop only on a refusal. Wins over the other flags
    if USE_ROUTER:
        from src.router import route_and_run

        answer = route_and_run(query)
        if cache is not None:
            cache.store(query, answer)
        return answer

    # tool loop: the model runs its own searches, so it skips the graph
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


class UnsupportedAsyncConfig(RuntimeError):
    """arun_agent was called with a flag it can't handle.

    Its own type so a fallback to run_agent doesn't also catch unrelated RuntimeErrors.
    """


async def arun_agent(query: str) -> str:
    """Async version of run_agent, for the default pipeline only.

    The flags and the cache add calls that aren't async yet, so it raises UnsupportedAsyncConfig
    instead of mixing blocking and async work. Retrieval runs in a thread because it is disk and cpu
    bound. It mirrors the default path (one sub-query, no entities) rather than going through the
    graph. Single turn only.
    """
    if USE_PLANNER or USE_VERIFIER or USE_TOOL_LOOP or USE_ROUTER:
        raise UnsupportedAsyncConfig(
            "arun_agent only supports the default pipeline (no planner, "
            "verifier, tool loop, or router). Use run_agent for any other "
            "configuration."
        )
    if get_semantic_cache() is not None:
        raise UnsupportedAsyncConfig(
            "arun_agent does not support the semantic cache yet. Use "
            "run_agent, or disable SEMANTIC_CACHE_ENABLED."
        )

    llm = get_llm()
    evidence = await asyncio.to_thread(
        get_retriever().retrieve,
        query,
        top_k=RETRIEVAL_TOP_K,
        final_k=RETRIEVAL_FINAL_K,
    )
    evidence_block = _format_evidence_block(evidence)
    prompt = SYNTHESIZER_PROMPT.format(user_query=query, evidence_block=evidence_block)
    answer = await instrumented_ainvoke(llm, prompt, stage="synthesizer")
    record_citation_guardrail(answer, evidence)
    return answer


@dataclass
class StreamResult:
    """Filled in by stream_agent: the final state and the timings. Read it after the stream ends."""

    final_state: dict[str, Any] | None = None
    ttft_seconds: float | None = None
    total_seconds: float | None = None
    # the rewritten question, if history changed it (the UI shows it)
    condensed_query: str | None = None


def _chunk_text(message: Any) -> str:
    """Text of one streamed chunk. It doesn't strip, since a chunk can end on a space the next one needs."""
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
    history: Sequence[Turn] | None = None,
) -> Iterator[str]:
    """Stream the synthesizer's answer as it is written.

    The earlier stages run first and don't stream. Only the synthesizer node's text is yielded, so
    turning on the planner or verifier doesn't leak their output. With history, the rewritten
    question lands on result.condensed_query. When the generator is exhausted, result holds the
    final state, the time to first token and the total time.
    """
    if history:
        standalone = condense_query(history, query)
        if standalone != query:
            result.condensed_query = standalone
            query = standalone

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


# quick manual check

if __name__ == "__main__":
    import sys

    from src.ingest import ingest_all
    from src.retriever import HybridRetriever

    print("=" * 70)
    print("  Citera agent smoke test")
    print("=" * 70)

    # make sure the index exists
    print("\nChecking/building retrieval index...")
    retriever = HybridRetriever()

    if retriever.index_size == 0 or not retriever.bm25_ready:
        reason = "empty" if retriever.index_size == 0 else "BM25 missing"
        print(f"   Index needs (re)build ({reason}) - ingesting PDFs...")
        chunks = ingest_all()
        if not chunks:
            print("No PDFs found in data/. Add PDFs and retry.")
            sys.exit(1)
        retriever.build_index(chunks)
        print(f"   Index built: {retriever.index_size} docs, BM25: {retriever.bm25_ready}.")
    else:
        print(f"   Index already populated: {retriever.index_size} docs, BM25: ready.")

    # a two-part question
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
