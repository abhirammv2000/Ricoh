"""FastAPI backend for the agent, so anything can call it (the Streamlit app uses the same agent code).

    GET  /health         liveness check
    POST /query          answer a question, with its cost and latency
    POST /query/stream   stream the answer as server-sent events

Run it with: uvicorn api.main:api --port 8000
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import queue
import threading
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent import StreamResult, UnsupportedAsyncConfig, arun_agent, run_agent, stream_agent
from src.guardrails import screen_input
from src.instrumentation import record_run
from src.ratelimit import TokenBucketLimiter

api = FastAPI(title="Citera RAG API", version="1.0.0")

# cap the input length, which bounds the cost of one request and rejects bad input early
MAX_QUERY_CHARS = 2000

# rate limit per client ip: RATE_LIMIT_RPS requests a second with a burst of RATE_LIMIT_BURST.
# Both can be set in the environment. /health is left out so a load balancer can always probe it
_limiter = TokenBucketLimiter(
    rate_per_sec=float(os.getenv("RATE_LIMIT_RPS", "1")),
    capacity=int(os.getenv("RATE_LIMIT_BURST", "10")),
)


def rate_limit(request: Request) -> None:
    """Reject a client that is over its rate with 429 and a Retry-After hint."""
    client = request.client.host if request.client else "unknown"
    if not _limiter.allow(client):
        wait = math.ceil(_limiter.retry_after(client)) or 1
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(wait)},
        )


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


def _screen(query: str) -> None:
    """Turn away a known jailbreak or override pattern with a 400, before any LLM call."""
    verdict = screen_input(query)
    if not verdict.allowed:
        raise HTTPException(status_code=400, detail=verdict.reason)


class QueryResponse(BaseModel):
    answer: str
    cost_usd: float
    llm_calls: int
    latency_seconds: float


@api.get("/health")
def health() -> dict:
    return {"status": "ok"}


@api.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
async def query(req: QueryRequest) -> QueryResponse:
    """Answer one question and return the answer with its cost and latency.

    It is traced like a UI request (traces/traces.jsonl). It uses the async arun_agent, which only
    handles the default setup. For any other configuration it catches UnsupportedAsyncConfig and
    runs the sync run_agent in a worker thread.
    """
    _screen(req.query)
    started = time.perf_counter()
    with record_run(query=req.query) as rec:
        try:
            answer = await arun_agent(req.query)
        except UnsupportedAsyncConfig:
            answer = await asyncio.to_thread(run_agent, req.query)
    return QueryResponse(
        answer=answer,
        cost_usd=rec.total_cost_usd,
        llm_calls=rec.llm_calls,
        latency_seconds=round(time.perf_counter() - started, 3),
    )


@api.post("/query/stream", dependencies=[Depends(rate_limit)])
def query_stream(req: QueryRequest) -> StreamingResponse:
    """Stream the answer as server-sent events.

    Starlette runs each step of a generator in a different thread context, which breaks the tracer's
    ContextVar. So the traced generation runs in one worker thread and passes tokens to the response
    through a queue. Each token is a data: {"token": "..."} event, and a last data: {"done": true, ...}
    event carries the cost, call count and time to first token.
    """
    _screen(req.query)
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
