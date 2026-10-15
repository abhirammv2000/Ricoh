"""What the API does when things go wrong: provider errors, slow answers, a full service, a client that leaves.

No network and no model. The agent functions are replaced, and requests run concurrently through an
in-process async client, so the concurrency gate and the deadline are exercised for real.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

import api.main as apimod
from api.errors import (
    INTERNAL,
    MISCONFIGURED,
    PROVIDER_REJECTED,
    PROVIDER_UNAVAILABLE,
    TIMEOUT,
    ApiError,
    classify_error,
)
from api.limits import ConcurrencyGate
from api.main import create_app
from api.readiness import Readiness
from api.settings import Settings, SettingsError
from src.ratelimit import TokenBucketLimiter

SECRET = "sk-ant-api03-SECRET-DO-NOT-LEAK"


@pytest.fixture(autouse=True)
def index_is_ready(monkeypatch):
    """These tests replace the agent, so they do not need an index. Readiness has its own tests."""
    monkeypatch.setattr(
        apimod, "compute_readiness", lambda require_manifest=True: Readiness(index_version="test", chunks=1)
    )


@pytest.fixture(autouse=True)
def open_rate_limit():
    apimod._limiter = TokenBucketLimiter(rate_per_sec=10_000.0, capacity=10_000)
    yield


class ProviderError(Exception):
    """Looks like an SDK status error: it carries a status code, and its text holds something private."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code} - key {SECRET} account limit reached")
        self.status_code = status_code


async def _call(app, method: str, url: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def post(app, path="/query", query="how do I shut it down?"):
    return asyncio.run(_call(app, "POST", path, json={"query": query}))


# ---- how an error becomes what the caller sees ----


@pytest.mark.parametrize(
    "status, expected",
    [
        (529, PROVIDER_UNAVAILABLE),  # the provider's "overloaded"
        (429, PROVIDER_UNAVAILABLE),
        (500, PROVIDER_UNAVAILABLE),
        (503, PROVIDER_UNAVAILABLE),
        (401, MISCONFIGURED),  # our own key is wrong, so not the caller's problem
        (403, MISCONFIGURED),
        (408, TIMEOUT),
        (400, PROVIDER_REJECTED),
        (404, PROVIDER_REJECTED),
    ],
)
def test_provider_status_codes_map_to_public_errors(status, expected):
    assert classify_error(ProviderError(status)) == expected


def test_timeouts_and_connection_errors_are_classified_by_name():
    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    assert classify_error(APITimeoutError()) == TIMEOUT
    assert classify_error(APIConnectionError()) == PROVIDER_UNAVAILABLE
    assert classify_error(asyncio.TimeoutError()) == TIMEOUT
    assert classify_error(TimeoutError()) == TIMEOUT


def test_an_unknown_error_is_a_generic_500():
    assert classify_error(RuntimeError("boom")) == INTERNAL


def test_public_messages_never_contain_provider_text():
    for status in (400, 401, 429, 500, 529):
        error = classify_error(ProviderError(status))
        assert SECRET not in error.message
        assert "account limit" not in error.message


# ---- /query ----


def test_provider_overload_returns_503_with_retry_after_and_no_provider_text(monkeypatch):
    async def overloaded(q):
        raise ProviderError(529)

    monkeypatch.setattr(apimod, "arun_agent", overloaded)
    r = post(create_app(Settings()))
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "model_unavailable"
    assert body["request_id"] == r.headers["x-request-id"]
    assert int(r.headers["retry-after"]) >= 1
    assert SECRET not in r.text


def test_an_unexpected_error_is_a_generic_500_with_a_request_id(monkeypatch):
    async def broken(q):
        raise RuntimeError(f"database password is {SECRET}")

    monkeypatch.setattr(apimod, "arun_agent", broken)
    r = post(create_app(Settings()))
    assert r.status_code == 500
    assert r.json()["code"] == "internal_error"
    assert r.json()["request_id"]
    assert SECRET not in r.text


def test_a_slow_answer_gets_a_504_at_the_deadline_and_the_call_is_cancelled(monkeypatch):
    state = {"cancelled": False}

    async def slow(q):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    monkeypatch.setattr(apimod, "arun_agent", slow)
    app = create_app(Settings(request_timeout_seconds=0.3))
    r = post(app)
    assert r.status_code == 504
    assert r.json()["code"] == "timeout"
    assert state["cancelled"], "the provider call kept running after the caller was told it failed"


def test_a_failed_request_gives_its_slot_back(monkeypatch):
    async def broken(q):
        raise ProviderError(529)

    monkeypatch.setattr(apimod, "arun_agent", broken)
    app = create_app(Settings(max_concurrent_queries=1))
    for _ in range(5):
        assert post(app).status_code == 503
    assert app.state.gate.in_flight == 0


def test_when_full_extra_requests_get_a_fast_503_and_the_admitted_one_still_finishes(monkeypatch):
    started = {}

    async def blocked(q):
        started["flag"].set()
        await started["go"].wait()
        return "answer [a.pdf, Page 1]"

    monkeypatch.setattr(apimod, "arun_agent", blocked)
    app = create_app(Settings(max_concurrent_queries=1, queue_timeout_seconds=0.05))

    async def scenario():
        started["flag"] = asyncio.Event()
        started["go"] = asyncio.Event()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(client.post("/query", json={"query": "first question"}))
            await started["flag"].wait()  # the first request now holds the only slot
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            second = await client.post("/query", json={"query": "second question"})
            refused_in = loop.time() - t0
            started["go"].set()
            return await first, second, refused_in

    first, second, refused_in = asyncio.run(scenario())
    assert second.status_code == 503
    assert second.json()["code"] == "overloaded"
    assert "retry-after" in second.headers
    assert refused_in < 0.5, "a full service should refuse quickly, not queue"
    assert first.status_code == 200
    assert app.state.gate.in_flight == 0
    assert app.state.gate.rejected == 1


# ---- request ids ----


def test_every_response_has_a_request_id(monkeypatch):
    async def fine(q):
        return "ok [a.pdf, Page 1]"

    monkeypatch.setattr(apimod, "arun_agent", fine)
    app = create_app(Settings())
    r = asyncio.run(_call(app, "GET", "/health"))
    assert len(r.headers["x-request-id"]) >= 8


def test_a_sane_request_id_from_the_caller_is_kept_and_a_bad_one_is_replaced():
    app = create_app(Settings())
    good = asyncio.run(_call(app, "GET", "/health", headers={"X-Request-ID": "trace-123.abc"}))
    assert good.headers["x-request-id"] == "trace-123.abc"
    bad = asyncio.run(_call(app, "GET", "/health", headers={"X-Request-ID": "x y\"z; drop"}))
    assert bad.headers["x-request-id"] != "x y\"z; drop"


# ---- /query/stream ----


@pytest.fixture
def threaded_stream(monkeypatch):
    """Streaming through the worker thread, which is what runs when the planner, verifier, tool loop, router
    or semantic cache is on. The default pipeline streams with astream_agent instead."""
    monkeypatch.setattr(apimod, "async_pipeline_supported", lambda: False)


def test_a_stream_failure_is_an_error_event_without_provider_text(monkeypatch, threaded_stream):
    def broken_stream(q, result):
        raise ProviderError(529)
        yield  # pragma: no cover

    monkeypatch.setattr(apimod, "stream_agent", broken_stream)
    r = post(create_app(Settings()), path="/query/stream")
    assert r.status_code == 200
    assert '"code": "model_unavailable"' in r.text
    assert SECRET not in r.text
    assert '"done": true' not in r.text


def test_a_stream_past_the_deadline_ends_with_a_timeout_event_and_stops_the_worker(monkeypatch, threaded_stream):
    import threading

    stopped = threading.Event()

    def slow_stream(q, result):
        try:
            yield "first "
            for _ in range(100):
                import time

                time.sleep(0.05)
                yield "word "
        finally:
            stopped.set()

    monkeypatch.setattr(apimod, "stream_agent", slow_stream)
    app = create_app(Settings(request_timeout_seconds=0.4))
    r = post(app, path="/query/stream")
    assert '"token": "first "' in r.text
    assert '"code": "timeout"' in r.text
    assert stopped.wait(timeout=3), "the worker kept producing after the stream ended"
    assert app.state.gate.in_flight == 0


def test_a_full_service_refuses_a_stream_before_it_starts(monkeypatch):
    app = create_app(Settings(max_concurrent_queries=1, queue_timeout_seconds=0.0))

    async def scenario():
        lease = await app.state.gate.acquire(0)  # something else holds the only slot
        assert lease is not None
        return await _call(app, "POST", "/query/stream", json={"query": "hello"})

    r = asyncio.run(scenario())
    assert r.status_code == 503
    assert r.json()["code"] == "overloaded"


# ---- /query/stream, default pipeline (async) ----


def test_the_default_pipeline_streams_tokens_then_a_done_event_with_cost(monkeypatch):
    async def fake_astream(q, result):
        result.ttft_seconds = 0.05
        yield "hel"
        yield "lo"

    monkeypatch.setattr(apimod, "async_pipeline_supported", lambda: True)
    monkeypatch.setattr(apimod, "astream_agent", fake_astream)
    app = create_app(Settings())
    r = post(app, path="/query/stream")
    assert r.status_code == 200
    assert '"token": "hel"' in r.text and '"token": "lo"' in r.text
    assert '"done": true' in r.text and '"ttft_seconds": 0.05' in r.text
    assert app.state.gate.in_flight == 0


def test_the_default_pipeline_reports_a_provider_failure_as_a_public_error_event(monkeypatch):
    async def broken(q, result):
        raise ProviderError(529)
        yield  # pragma: no cover

    monkeypatch.setattr(apimod, "async_pipeline_supported", lambda: True)
    monkeypatch.setattr(apimod, "astream_agent", broken)
    app = create_app(Settings())
    r = post(app, path="/query/stream")
    assert '"code": "model_unavailable"' in r.text
    assert SECRET not in r.text
    assert '"done": true' not in r.text
    assert app.state.gate.in_flight == 0


def test_the_default_pipeline_cancels_the_provider_stream_at_the_deadline(monkeypatch):
    state = {"cancelled": False}

    async def slow(q, result):
        try:
            yield "first "
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    monkeypatch.setattr(apimod, "async_pipeline_supported", lambda: True)
    monkeypatch.setattr(apimod, "astream_agent", slow)
    app = create_app(Settings(request_timeout_seconds=0.3))
    r = post(app, path="/query/stream")
    assert '"token": "first "' in r.text
    assert '"code": "timeout"' in r.text
    assert state["cancelled"], "the provider stream kept running after the deadline"
    assert app.state.gate.in_flight == 0


# ---- the gate and the settings ----


def test_the_gate_with_no_wait_takes_a_free_slot_and_refuses_when_none_is_free():
    async def scenario():
        gate = ConcurrencyGate(1)
        first = await gate.acquire(0)
        second = await gate.acquire(0)
        first.release()
        first.release()  # releasing twice must not hand out an extra slot
        third = await gate.acquire(0)
        fourth = await gate.acquire(0)
        return first, second, third, fourth, gate

    first, second, third, fourth, gate = asyncio.run(scenario())
    assert first is not None
    assert second is None
    assert third is not None
    assert fourth is None, "a double release created a second slot"
    assert gate.in_flight == 1


def test_a_waiting_request_gets_the_slot_when_it_frees_up():
    async def scenario():
        gate = ConcurrencyGate(1)
        held = await gate.acquire(0)

        async def free_soon():
            await asyncio.sleep(0.05)
            held.release()

        asyncio.get_running_loop().create_task(free_soon())
        return await gate.acquire(1.0)

    assert asyncio.run(scenario()) is not None


def test_settings_defaults_and_overrides():
    s = Settings.from_env({})
    assert s == Settings()
    s = Settings.from_env({"REQUEST_TIMEOUT_SECONDS": "12.5", "MAX_CONCURRENT_QUERIES": "4", "QUEUE_TIMEOUT_SECONDS": "0"})
    assert (s.request_timeout_seconds, s.max_concurrent_queries, s.queue_timeout_seconds) == (12.5, 4, 0.0)


@pytest.mark.parametrize(
    "env, name",
    [
        ({"REQUEST_TIMEOUT_SECONDS": "soon"}, "REQUEST_TIMEOUT_SECONDS"),
        ({"REQUEST_TIMEOUT_SECONDS": "0"}, "REQUEST_TIMEOUT_SECONDS"),
        ({"MAX_CONCURRENT_QUERIES": "0"}, "MAX_CONCURRENT_QUERIES"),
        ({"MAX_CONCURRENT_QUERIES": "2.5"}, "MAX_CONCURRENT_QUERIES"),
        ({"QUEUE_TIMEOUT_SECONDS": "-1"}, "QUEUE_TIMEOUT_SECONDS"),
        ({"RATE_LIMIT_RPS": "0"}, "RATE_LIMIT_RPS"),
        ({"RATE_LIMIT_BURST": "many"}, "RATE_LIMIT_BURST"),
    ],
)
def test_a_bad_setting_stops_startup_and_names_the_variable(env, name):
    with pytest.raises(SettingsError, match=name):
        Settings.from_env(env)


def test_api_error_carries_its_public_error():
    err = ApiError(TIMEOUT)
    assert err.error is TIMEOUT
