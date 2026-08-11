"""FastAPI backend for the agent.

This is the API-first boundary in front of the same agent the Streamlit UI
uses. It exists so the system can be called by any client, load tested, rate
limited, and eventually put behind a gateway, rather than only through one UI.
Both surfaces call the same agent code, so there is one source of truth for how
a question gets answered.

Endpoints:
    GET  /health         liveness check for load balancers and container probes
    POST /query          answer a question, return the answer plus its trace
    POST /query/stream   stream the answer token by token as server-sent events

Run locally:
    uvicorn api.main:api --port 8000
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent import StreamResult, run_agent, stream_agent
from src.instrumentation import record_run

api = FastAPI(title="Citera RAG API", version="1.0.0")

# Cap the input length. This is a cheap first guardrail: it bounds the prompt
# size (and so the cost) of a single request and rejects obviously bad input at
# the edge instead of paying for it downstream.
MAX_QUERY_CHARS = 2000


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class QueryResponse(BaseModel):
    answer: str
    cost_usd: float
    llm_calls: int
    latency_seconds: float


@api.get("/health")
def health() -> dict:
    return {"status": "ok"}


@api.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answer one question and return the answer with its cost and latency.

    The whole call is wrapped in record_run, so it is traced exactly like a UI
    request and shows up in traces/traces.jsonl alongside the rest.
    """
    started = time.perf_counter()
    with record_run(query=req.query) as rec:
        answer = run_agent(req.query)
    return QueryResponse(
        answer=answer,
        cost_usd=rec.total_cost_usd,
        llm_calls=rec.llm_calls,
        latency_seconds=round(time.perf_counter() - started, 3),
    )


@api.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    """Stream the answer token by token as server-sent events.

    Starlette drives a streaming generator across threadpool contexts, one per
    next(), which breaks the ContextVar the tracer relies on: the token set on
    enter cannot be reset on exit, and spans recorded in a different context are
    lost. So the traced generation runs in a single worker thread, where the
    tracer's set, get, and reset all line up, and tokens are handed to the
    response through a queue.

    Each token arrives as a `data: {"token": "..."}` event. A final
    `data: {"done": true, ...}` event carries the cost, the call count, and the
    time to first token, so a client gets the same trace summary the UI shows.
    """
    channel: queue.Queue[tuple[str, Any]] = queue.Queue()
    result = StreamResult()
    summary: dict[str, Any] = {}

    def produce() -> None:
        try:
            with record_run(query=req.query) as rec:
                for chunk in stream_agent(req.query, result):
                    channel.put(("token", chunk))
                summary["cost_usd"] = rec.total_cost_usd
                summary["llm_calls"] = rec.llm_calls
        except Exception as exc:  # surface the failure instead of hanging
            channel.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            channel.put(("done", None))

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()

    def events():
        while True:
            kind, value = channel.get()
            if kind == "token":
                yield f"data: {json.dumps({'token': value})}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'error': value})}\n\n"
                break
            else:
                payload = {"done": True, "ttft_seconds": result.ttft_seconds, **summary}
                yield f"data: {json.dumps(payload)}\n\n"
                break
        worker.join(timeout=1)

    return StreamingResponse(events(), media_type="text/event-stream")
