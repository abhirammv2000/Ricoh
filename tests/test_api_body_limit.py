"""A request body over the limit is refused before the service reads it, whether or not the caller has a key.

FastAPI reads a whole body before it checks the API key. Without this limit, one anonymous caller could make the
service hold hundreds of megabytes in memory. These tests check the refusal and, more important, that the service
stops reading once the limit is passed.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import api.main as apimod
from api.body_limit import BodyLimitMiddleware
from api.main import create_app
from api.readiness import Readiness
from api.settings import Settings, SettingsError

KEY = "alpha-secret-key-0001"
LIMIT = 2048


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(
        apimod, "compute_readiness", lambda require_manifest=True: Readiness(index_version="t", chunks=1)
    )
    asked = []

    async def fine(question):
        asked.append(question)
        return "Choose Network [network_guide.pdf, Page 3]."

    monkeypatch.setattr(apimod, "arun_agent", fine)
    return asked


def make_app():
    return create_app(Settings(api_keys=f"partner:{KEY}", fallback_to_retrieval=False, max_body_bytes=LIMIT))


def post(app, content: bytes, headers=None):
    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            all_headers = {"Content-Type": "application/json", **(headers or {})}
            return await client.post("/query", content=content, headers=all_headers)

    return asyncio.run(run())


def body_of(size: int) -> bytes:
    """A valid JSON body of exactly this many bytes."""
    padding = size - len(b'{"query": ""}')
    return b'{"query": "' + b"a" * padding + b'"}'


def test_a_normal_question_still_works(setup):
    response = post(make_app(), b'{"query": "how do I configure the network?"}', {"X-API-Key": KEY})

    assert response.status_code == 200
    assert setup == ["how do I configure the network?"]


def test_a_body_over_the_limit_is_refused_with_413_and_never_reaches_the_handler(setup):
    response = post(make_app(), body_of(LIMIT + 1), {"X-API-Key": KEY})

    assert response.status_code == 413
    assert response.json()["code"] == "request_too_large"
    assert response.json()["request_id"] != ""
    assert setup == []


def test_the_refusal_comes_before_the_key_check():
    # no key at all: the answer is 413, not 401, so a caller with no key cannot get the body read
    response = post(make_app(), body_of(LIMIT * 50))

    assert response.status_code == 413


def test_a_body_exactly_at_the_limit_is_accepted(setup):
    response = post(make_app(), body_of(LIMIT), {"X-API-Key": KEY})

    # it is a valid body, but the question is longer than 2000 characters, so it is the normal 422
    assert response.status_code == 422
    assert setup == []


def test_the_refusal_is_counted_as_an_error():
    app = make_app()
    post(app, body_of(LIMIT + 1))

    body, _ = app.state.metrics.render()
    assert b'citera_errors_total{code="request_too_large"} 1.0' in body


def test_a_body_that_does_not_say_its_size_is_refused_as_soon_as_it_passes_the_limit():
    """A chunked upload has no Content-Length. The service must stop reading at the limit, not at the end."""
    chunk = b"a" * 512
    reads = 0
    sent = []

    async def receive():
        nonlocal reads
        reads += 1
        if reads == 1:
            return {"type": "http.request", "body": b'{"query": "', "more_body": True}
        if reads <= 1000:
            return {"type": "http.request", "body": chunk, "more_body": True}
        return {"type": "http.request", "body": b'"}', "more_body": False}

    async def send(message):
        sent.append(message)

    async def application(scope, receive, send):
        # what FastAPI does: read the body to the end, then answer
        while True:
            message = await receive()
            if message["type"] == "http.disconnect" or not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await send({"type": "http.response.body", "body": b"read the whole body"})

    refused = []
    middleware = BodyLimitMiddleware(application, max_bytes=LIMIT, on_refused=refused.append)
    scope = {"type": "http", "headers": [], "state": {"request_id": "abc"}}

    asyncio.run(middleware(scope, receive, send))

    statuses = [m["status"] for m in sent if m["type"] == "http.response.start"]
    assert statuses == [413]
    assert reads <= 6, f"it kept reading after the limit: {reads} reads of a 512 KB body"
    assert refused == ["request_too_large"]
    answer = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert json.loads(answer)["request_id"] == "abc"


def test_a_lying_content_length_is_still_caught_by_the_count():
    """Content-Length says 100 bytes but more arrive. What counts is what arrived."""
    reads = 0
    sent = []

    async def receive():
        nonlocal reads
        reads += 1
        # the stream ends after 50 chunks, so a limit that stops working fails the test instead of hanging it
        return {"type": "http.request", "body": b"a" * 1024, "more_body": reads < 50}

    async def send(message):
        sent.append(message)

    async def application(scope, receive, send):
        while True:
            message = await receive()
            if message["type"] == "http.disconnect" or not message.get("more_body"):
                break

    middleware = BodyLimitMiddleware(application, max_bytes=LIMIT)
    scope = {"type": "http", "headers": [(b"content-length", b"100")], "state": {}}

    asyncio.run(middleware(scope, receive, send))

    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [413]
    assert reads <= 4


def test_the_setting_has_a_default_an_override_and_a_floor():
    assert Settings.from_env({}).max_body_bytes == 32768
    assert Settings.from_env({"MAX_BODY_BYTES": "4096"}).max_body_bytes == 4096
    with pytest.raises(SettingsError, match="MAX_BODY_BYTES"):
        Settings.from_env({"MAX_BODY_BYTES": "10"})


def test_the_longest_allowed_question_fits_under_the_default_limit(setup):
    """2000 emoji, escaped the way most JSON clients send them, is 24,013 bytes. It must not be refused."""
    question = "\U0001F600" * 2000
    body = json.dumps({"query": question}).encode()
    assert len(body) == 24013

    app = create_app(Settings(api_keys=f"partner:{KEY}", fallback_to_retrieval=False))
    response = post(app, body, {"X-API-Key": KEY})

    assert response.status_code == 200
    assert setup == [question]
