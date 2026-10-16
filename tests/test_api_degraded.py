"""When the model provider is down or too slow, the service answers with the passages retrieval found.

Retrieval does not depend on the model, so it still works, and a support engineer who is told "here are the
relevant pages" is better off than one who gets an error. The response says plainly that it is degraded, a
normal response is unchanged, and the cases where passages would not help (a full service, our own broken
credentials) still fail the way they did.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

import api.main as apimod
from api.main import DEGRADED_ANSWER, create_app
from api.readiness import Readiness
from api.settings import Settings

SECRET = "PROVIDER-SECRET-DO-NOT-LEAK"

PASSAGES = [
    {"source_document": "network_guide.pdf", "page_number": 3, "text": "Open the Settings menu.\n  Choose Network.", "rrf_score": 0.0328},
    {"source_document": "admin_guide.pdf", "page_number": 12, "text": "x" * 900, "rrf_score": 0.0161},
]


class ProviderError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code} - key {SECRET}")
        self.status_code = status_code


class FakeRetriever:
    def __init__(self, found=None, fail=False) -> None:
        self.found = PASSAGES if found is None else found
        self.fail = fail
        self.calls = 0

    def retrieve(self, query, top_k, final_k):
        self.calls += 1
        if self.fail:
            raise RuntimeError("index exploded")
        return list(self.found)


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(apimod, "compute_readiness", lambda require_manifest=True: Readiness(index_version="t", chunks=1))


def use_retriever(monkeypatch, retriever):
    monkeypatch.setattr("src.agent.get_retriever", lambda: retriever)
    return retriever


def failing_agent(monkeypatch, status):
    async def fail(q):
        raise ProviderError(status)

    monkeypatch.setattr(apimod, "arun_agent", fail)


async def _call(app, method, url, **kwargs):
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def post(app, path="/query", query="how do I configure the network?"):
    return asyncio.run(_call(app, "POST", path, json={"query": query}))


def test_provider_outage_returns_the_retrieved_passages_marked_as_degraded(monkeypatch):
    failing_agent(monkeypatch, 529)
    retriever = use_retriever(monkeypatch, FakeRetriever())
    r = post(create_app(Settings(auth_disabled=True)))
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["answer"] == DEGRADED_ANSWER
    assert [s["document"] for s in body["sources"]] == ["network_guide.pdf", "admin_guide.pdf"]
    assert body["sources"][0]["page"] == 3 and body["sources"][0]["score"] == 0.0328
    assert r.headers["x-citera-degraded"] == "model_unavailable"
    assert body["llm_calls"] == 0 and body["cost_usd"] == 0.0
    assert retriever.calls == 1
    assert SECRET not in r.text


def test_an_excerpt_is_short_and_has_its_whitespace_tidied(monkeypatch):
    failing_agent(monkeypatch, 529)
    use_retriever(monkeypatch, FakeRetriever())
    sources = post(create_app(Settings(auth_disabled=True))).json()["sources"]
    assert sources[0]["excerpt"] == "Open the Settings menu. Choose Network."
    assert len(sources[1]["excerpt"]) == 300


def test_a_normal_response_has_exactly_the_fields_it_always_had(monkeypatch):
    async def fine(q):
        return "Choose Network [network_guide.pdf, Page 3]."

    monkeypatch.setattr(apimod, "arun_agent", fine)
    r = post(create_app(Settings(auth_disabled=True)))
    assert set(r.json()) == {"answer", "cost_usd", "llm_calls", "latency_seconds"}
    assert "x-citera-degraded" not in r.headers


def test_a_slow_provider_that_hits_the_deadline_also_degrades(monkeypatch):
    async def slow(q):
        await asyncio.sleep(30)

    monkeypatch.setattr(apimod, "arun_agent", slow)
    use_retriever(monkeypatch, FakeRetriever())
    r = post(create_app(Settings(auth_disabled=True, request_timeout_seconds=0.3)))
    assert r.status_code == 200 and r.json()["degraded"] is True
    assert r.headers["x-citera-degraded"] == "timeout"


def test_with_the_fallback_off_an_outage_is_a_503(monkeypatch):
    failing_agent(monkeypatch, 529)
    retriever = use_retriever(monkeypatch, FakeRetriever())
    r = post(create_app(Settings(auth_disabled=True, fallback_to_retrieval=False)))
    assert r.status_code == 503 and r.json()["code"] == "model_unavailable"
    assert retriever.calls == 0


def test_when_retrieval_finds_nothing_it_is_still_a_503_not_an_empty_answer(monkeypatch):
    failing_agent(monkeypatch, 529)
    use_retriever(monkeypatch, FakeRetriever(found=[]))
    r = post(create_app(Settings(auth_disabled=True)))
    assert r.status_code == 503 and r.json()["code"] == "model_unavailable"


def test_when_retrieval_also_fails_the_caller_gets_the_original_error(monkeypatch):
    failing_agent(monkeypatch, 529)
    use_retriever(monkeypatch, FakeRetriever(fail=True))
    r = post(create_app(Settings(auth_disabled=True)))
    assert r.status_code == 503 and r.json()["code"] == "model_unavailable"
    assert "index exploded" not in r.text


@pytest.mark.parametrize("status, code", [(401, "service_misconfigured"), (400, "model_error")])
def test_errors_that_passages_would_not_fix_still_fail(monkeypatch, status, code):
    failing_agent(monkeypatch, status)
    retriever = use_retriever(monkeypatch, FakeRetriever())
    r = post(create_app(Settings(auth_disabled=True)))
    assert r.json()["code"] == code and r.status_code != 200
    assert retriever.calls == 0


def test_a_full_service_is_still_a_503_and_does_not_start_retrieval(monkeypatch):
    retriever = use_retriever(monkeypatch, FakeRetriever())
    app = create_app(Settings(auth_disabled=True, max_concurrent_queries=1, queue_timeout_seconds=0.0))

    async def scenario():
        lease = await app.state.gate.acquire(0)
        assert lease is not None
        return await _call(app, "POST", "/query", json={"query": "hello there"})

    r = asyncio.run(scenario())
    assert r.status_code == 503 and r.json()["code"] == "overloaded"
    assert retriever.calls == 0


def test_a_stream_degrades_with_the_passages_in_its_error_event(monkeypatch):
    async def broken(q, result):
        raise ProviderError(529)
        yield  # pragma: no cover

    monkeypatch.setattr(apimod, "async_pipeline_supported", lambda: True)
    monkeypatch.setattr(apimod, "astream_agent", broken)
    use_retriever(monkeypatch, FakeRetriever())
    r = post(create_app(Settings(auth_disabled=True)), path="/query/stream")
    assert r.status_code == 200
    assert '"degraded": true' in r.text and '"code": "model_unavailable"' in r.text
    assert "network_guide.pdf" in r.text
    assert SECRET not in r.text


def test_degraded_answers_are_counted_apart_from_real_ones(monkeypatch):
    failing_agent(monkeypatch, 529)
    use_retriever(monkeypatch, FakeRetriever())
    app = create_app(Settings(auth_disabled=True))
    post(app)
    text = asyncio.run(_call(app, "GET", "/metrics")).text
    assert 'citera_answers_total{outcome="degraded"} 1.0' in text
    assert 'citera_errors_total{code="model_unavailable"} 1.0' in text, "the outage must still show up as an error"
