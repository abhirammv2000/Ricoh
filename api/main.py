"""FastAPI backend for the agent, so anything can call it (the Streamlit app uses the same agent code).

    GET  /health         liveness check
    POST /query          answer a question, with its cost and latency
    POST /query/stream   stream the answer as server-sent events

Run it with: uvicorn api.main:api --port 8000

What it does when things go wrong, in one place:
  * the model provider is slow, down or rate limiting: the caller gets a 503 or 504 with a short message,
    never the provider's error text. The full error is logged with the request id.
  * a question takes longer than REQUEST_TIMEOUT_SECONDS: 504, and the provider call is cancelled.
  * more than MAX_CONCURRENT_QUERIES questions at once: the extra ones wait QUEUE_TIMEOUT_SECONDS for a
    slot and then get a 503 with Retry-After.
  * a streaming client goes away: the model call is stopped, so nobody pays for an answer no one reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.errors import NOT_READY, OVERLOADED, TIMEOUT, ApiError, PublicError, classify_error
from api.limits import ConcurrencyGate, Lease
from api.readiness import Readiness, compute_readiness, needs_recheck
from api.settings import Settings
from src.agent import (
    StreamResult,
    UnsupportedAsyncConfig,
    arun_agent,
    astream_agent,
    async_pipeline_supported,
    run_agent,
    stream_agent,
)
from src.guardrails import screen_input
from src.instrumentation import record_run
from src.llm_factory import (
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_TIMEOUT_SECONDS,
    import_provider_sdk,
    warm_up_client,
)
from src.ratelimit import TokenBucketLimiter

log = logging.getLogger("citera.api")

# cap the input length, which bounds the cost of one request and rejects bad input early
MAX_QUERY_CHARS = 2000

# rate limit per client ip: RATE_LIMIT_RPS requests a second with a burst of RATE_LIMIT_BURST.
# Both can be set in the environment. /health is left out so a load balancer can always probe it
_limiter = TokenBucketLimiter(
    rate_per_sec=float(os.getenv("RATE_LIMIT_RPS", "1")),
    capacity=int(os.getenv("RATE_LIMIT_BURST", "10")),
)

# a request id from a caller is only trusted if it looks like an id, since it goes into logs and headers
_REQUEST_ID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIdMiddleware:
    """Gives every request an id (the caller's X-Request-ID if it is sane, else a new one) and sends it back.

    A plain ASGI middleware on purpose. The decorator-style middleware wraps the response in a way that
    interferes with streaming responses and with cancelling them when the client goes away.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = ""
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                supplied = value.decode("latin-1")
                break
        request_id = supplied if _REQUEST_ID_OK.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_id)


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


class QueryResponse(BaseModel):
    answer: str
    cost_usd: float
    llm_calls: int
    latency_seconds: float


def _screen(query: str) -> None:
    """Turn away a known jailbreak or override pattern with a 400, before any LLM call."""
    verdict = screen_input(query)
    if not verdict.allowed:
        raise HTTPException(status_code=400, detail=verdict.reason)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _error_response(error: PublicError, request_id: str) -> JSONResponse:
    headers = {"Retry-After": str(error.retry_after)} if error.retry_after else {}
    body = {"detail": error.message, "code": error.code, "request_id": request_id}
    return JSONResponse(body, status_code=error.status, headers=headers)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # check the index and load the models before the first question, not during it. The first request
        # used to take 2.7 s against 0.4 s for the rest, because it loaded the embedding model
        readiness = await refresh_readiness(force=True)
        if readiness.ready:
            await asyncio.to_thread(_warm_up)
            try:
                # the SDK import is slow and blocking, so it goes in a thread. The client has to be built in
                # this loop, which is the one that serves requests
                await asyncio.to_thread(import_provider_sdk)
                warm_up_client()
            except Exception:
                log.exception("warming up the model client failed. The first question will pay for it")
            log.info(
                "ready: index_version=%s chunks=%d embedding_model=%s",
                readiness.index_version, readiness.chunks, readiness.embedding_model,
            )
        yield

    app = FastAPI(title="Citera RAG API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.readiness = None
    app.state.readiness_lock = asyncio.Lock()
    worst_case = _DEFAULT_TIMEOUT_SECONDS * (_DEFAULT_MAX_RETRIES + 1)
    if settings.request_timeout_seconds < worst_case:
        # not an error: a provider that fails fast is retried inside the deadline. A provider that is slow
        # AND failing will hit the deadline first, and the caller sees a 504 instead of a 503
        log.warning(
            "REQUEST_TIMEOUT_SECONDS (%.0f) is shorter than the model client's worst case "
            "(LLM_TIMEOUT_SECONDS %.0f x %d tries = %.0f s). A slow, failing provider will give a 504.",
            settings.request_timeout_seconds, _DEFAULT_TIMEOUT_SECONDS, _DEFAULT_MAX_RETRIES + 1, worst_case,
        )
    app.state.gate = ConcurrencyGate(settings.max_concurrent_queries)
    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.error, _request_id(request))

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # the last stop for anything not handled below. The caller gets a generic 500 with the request id
        log.exception("unhandled error, request_id=%s", _request_id(request))
        return _error_response(classify_error(exc), _request_id(request))

    async def refresh_readiness(force: bool = False) -> Readiness:
        """The cached readiness, recomputed when there is none, when forced, or when a failure is old enough."""
        if not force and not needs_recheck(app.state.readiness):
            return app.state.readiness
        async with app.state.readiness_lock:
            if not force and not needs_recheck(app.state.readiness):
                return app.state.readiness
            readiness = await asyncio.to_thread(compute_readiness, settings.require_index_manifest)
            app.state.readiness = readiness
            if not readiness.ready:
                log.error("not ready: %s", "; ".join(readiness.problems))
            return readiness

    async def require_ready() -> None:
        """Refuse a question while the index is not fit to answer from, instead of calling the model with no
        evidence and returning a confident refusal."""
        if not (await refresh_readiness()).ready:
            raise ApiError(NOT_READY)

    async def take_slot(request: Request) -> Lease:
        """A slot to run one question, or a 503 if the service is full."""
        lease = await app.state.gate.acquire(settings.queue_timeout_seconds)
        if lease is None:
            log.warning("turned away, service full, request_id=%s", _request_id(request))
            raise ApiError(OVERLOADED)
        return lease

    @app.get("/health")
    def health() -> dict:
        """Is the process alive. It does not look at the index, so a bad index cannot make an orchestrator
        restart the process in a loop."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """Can this instance give a correct answer. 503 with the reasons when it cannot."""
        readiness = await refresh_readiness()
        if readiness.ready:
            return JSONResponse(
                {
                    "status": "ready",
                    "index_version": readiness.index_version,
                    "chunks": readiness.chunks,
                    "embedding_model": readiness.embedding_model,
                }
            )
        return JSONResponse(
            {"status": "not_ready", "problems": readiness.problems},
            status_code=503,
            headers={"Retry-After": str(NOT_READY.retry_after)},
        )

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
    async def query(request: Request, req: QueryRequest) -> QueryResponse:
        """Answer one question and return the answer with its cost and latency.

        It is traced like a UI request (traces/traces.jsonl). It uses the async arun_agent, which only
        handles the default setup. For any other configuration it catches UnsupportedAsyncConfig and
        runs the sync run_agent in a worker thread (a thread cannot be cancelled, so after a timeout it
        runs on until the provider call gives up, which the client timeout bounds).
        """
        _screen(req.query)
        await require_ready()
        lease = await take_slot(request)
        started = time.perf_counter()
        try:
            with record_run(query=req.query) as rec:
                try:
                    async with asyncio.timeout(settings.request_timeout_seconds):
                        try:
                            answer = await arun_agent(req.query)
                        except UnsupportedAsyncConfig:
                            answer = await asyncio.to_thread(run_agent, req.query)
                except Exception as exc:
                    public = classify_error(exc)
                    log.error(
                        "query failed, request_id=%s code=%s error=%s: %s",
                        _request_id(request), public.code, type(exc).__name__, exc,
                    )
                    raise ApiError(public) from exc
        finally:
            lease.release()
        return QueryResponse(
            answer=answer,
            cost_usd=rec.total_cost_usd,
            llm_calls=rec.llm_calls,
            latency_seconds=round(time.perf_counter() - started, 3),
        )

    async def async_events(query: str, request_id: str, lease: Lease):
        """The stream for the default pipeline. It runs in the task that serves the response, so when the
        client goes away that task is cancelled, the cancel reaches the provider's HTTP stream, and the
        provider stops generating."""
        outcome = "client_gone"  # stays this if the response is cancelled while a yield is waiting
        result = StreamResult()
        try:
            with record_run(query=query) as rec:
                try:
                    async with asyncio.timeout(settings.request_timeout_seconds):
                        async for text in astream_agent(query, result):
                            yield _sse({"token": text})
                except Exception as exc:
                    public = classify_error(exc)
                    outcome = "timeout" if public.code == "timeout" else "error"
                    log.error(
                        "stream failed, request_id=%s code=%s error=%s: %s",
                        request_id, public.code, type(exc).__name__, exc,
                    )
                    yield _sse({"error": public.message, "code": public.code, "request_id": request_id})
                    return
            outcome = "done"
            yield _sse(
                {
                    "done": True,
                    "ttft_seconds": result.ttft_seconds,
                    "cost_usd": rec.total_cost_usd,
                    "llm_calls": rec.llm_calls,
                }
            )
        finally:
            lease.release()
            log.info("stream finished, request_id=%s outcome=%s", request_id, outcome)

    @app.post("/query/stream", dependencies=[Depends(rate_limit)])
    async def query_stream(request: Request, req: QueryRequest) -> StreamingResponse:
        """Stream the answer as server-sent events.

        Each token is a data: {"token": "..."} event, and a last data: {"done": true, ...} event carries
        the cost, call count and time to first token. A failure is a data: {"error": ..., "code": ...}
        event with the same public message a /query failure would give.

        Starlette runs each step of a sync generator in a different thread context, which breaks the
        tracer's ContextVar. So the traced generation runs in one worker thread and passes tokens to the
        response through a queue. When the client goes away, or the deadline passes, the worker is told to
        stop and closes the agent's generator, which closes the provider connection.
        """
        _screen(req.query)
        await require_ready()
        lease = await take_slot(request)
        request_id = _request_id(request)
        if async_pipeline_supported():
            return _LeasedStreamingResponse(
                async_events(req.query, request_id, lease), lease, media_type="text/event-stream"
            )
        loop = asyncio.get_running_loop()
        channel: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        stop = threading.Event()
        result = StreamResult()
        summary: dict[str, Any] = {}

        def emit(kind: str, value: Any) -> None:
            loop.call_soon_threadsafe(channel.put_nowait, (kind, value))

        def produce() -> None:
            try:
                with record_run(query=req.query) as rec:
                    chunks = stream_agent(req.query, result)
                    try:
                        for chunk in chunks:
                            if stop.is_set():
                                log.info("stream worker stopping early, request_id=%s", request_id)
                                break
                            emit("token", chunk)
                    finally:
                        chunks.close()
                    summary["cost_usd"] = rec.total_cost_usd
                    summary["llm_calls"] = rec.llm_calls
            except Exception as exc:  # surface the failure instead of hanging
                public = classify_error(exc)
                log.error(
                    "stream failed, request_id=%s code=%s error=%s: %s",
                    request_id, public.code, type(exc).__name__, exc,
                )
                emit("error", public)
            finally:
                emit("done", None)

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        deadline = time.monotonic() + settings.request_timeout_seconds

        async def events():
            outcome = "client_gone"  # stays this if the response is cancelled while we wait
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    try:
                        kind, value = await asyncio.wait_for(channel.get(), timeout=max(remaining, 0.001))
                    except asyncio.TimeoutError:
                        stop.set()
                        outcome = "timeout"
                        log.error("stream timed out, request_id=%s", request_id)
                        yield _sse({"error": TIMEOUT.message, "code": TIMEOUT.code, "request_id": request_id})
                        break
                    if kind == "token":
                        yield _sse({"token": value})
                    elif kind == "error":
                        outcome = "error"
                        yield _sse({"error": value.message, "code": value.code, "request_id": request_id})
                        break
                    else:
                        outcome = "done"
                        payload = {"done": True, "ttft_seconds": result.ttft_seconds, **summary}
                        yield _sse(payload)
                        break
            finally:
                # runs on a normal end, on a timeout, and when the client disconnects and the response
                # is cancelled, so the worker always learns to stop
                stop.set()
                lease.release()
                log.info("stream finished, request_id=%s outcome=%s", request_id, outcome)

        return _LeasedStreamingResponse(events(), lease, media_type="text/event-stream")

    return app


def _warm_up() -> None:
    """Load the embedding model and touch the index with one real retrieval."""
    from src.agent import get_retriever

    get_retriever().retrieve("warm up", top_k=1, final_k=1)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class _LeasedStreamingResponse(StreamingResponse):
    """A streaming response that gives its slot back even if the body never starts.

    The body generator releases the slot when it finishes, but a generator that is never iterated never
    runs its cleanup (a client can disappear between the headers and the first read). Releasing is
    idempotent, so doing it here as well costs nothing.
    """

    def __init__(self, content, lease: Lease, **kwargs) -> None:
        super().__init__(content, **kwargs)
        self._lease = lease

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._lease.release()


api = create_app()
