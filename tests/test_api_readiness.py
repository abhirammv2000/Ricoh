"""Readiness: the service says when it cannot give a correct answer, and refuses to answer while that is true.

/health is liveness (is the process alive) and /ready is readiness (can it answer correctly). They are
separate on purpose: a bad index must stop traffic, and must not make an orchestrator restart the process.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as apimod
import api.readiness as readiness_mod
from api.main import create_app
from api.readiness import Readiness, compute_readiness, needs_recheck
from api.settings import Settings, SettingsError
from src.index_manifest import IndexIntegrityError
from src.ratelimit import TokenBucketLimiter
from tests.test_index_manifest import StubEmbedding, build, reopen

SECRET_KEY = "sk-ant-test-not-a-real-key"


@pytest.fixture(autouse=True)
def open_rate_limit():
    apimod._limiter = TokenBucketLimiter(rate_per_sec=10_000.0, capacity=10_000)
    yield


def ready() -> Readiness:
    return Readiness(index_version="abc123", chunks=88, embedding_model="model-a", checked_at=0.0)


def not_ready(problem="the vector index is empty, so every question would be refused") -> Readiness:
    return Readiness(problems=[problem], checked_at=time.monotonic())  # checked just now, as the real check is


def use(monkeypatch, readiness: Readiness):
    monkeypatch.setattr(apimod, "compute_readiness", lambda require_manifest=True: readiness)


async def _call(app, method, url, **kwargs):
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def call(app, method, url, **kwargs):
    return asyncio.run(_call(app, method, url, **kwargs))


# ---- the endpoints ----


def test_ready_reports_what_the_service_is_serving(monkeypatch):
    use(monkeypatch, ready())
    r = call(create_app(Settings()), "GET", "/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "index_version": "abc123", "chunks": 88, "embedding_model": "model-a"}


def test_not_ready_is_a_503_that_says_why(monkeypatch):
    use(monkeypatch, not_ready())
    r = call(create_app(Settings()), "GET", "/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"
    assert "empty" in r.json()["problems"][0]
    assert int(r.headers["retry-after"]) > 0


def test_health_stays_ok_when_the_index_is_bad(monkeypatch):
    # an orchestrator that restarted the process because of a bad index would restart it forever
    use(monkeypatch, not_ready())
    assert call(create_app(Settings()), "GET", "/health").status_code == 200


def test_queries_are_refused_before_any_model_call_while_not_ready(monkeypatch):
    use(monkeypatch, not_ready())
    called = {"model": False}

    async def must_not_run(q):
        called["model"] = True
        return "unreachable"

    monkeypatch.setattr(apimod, "arun_agent", must_not_run)
    app = create_app(Settings())
    r = call(app, "POST", "/query", json={"query": "how do I shut it down?"})
    assert r.status_code == 503 and r.json()["code"] == "not_ready"
    s = call(app, "POST", "/query/stream", json={"query": "how do I shut it down?"})
    assert s.status_code == 503 and s.json()["code"] == "not_ready"
    assert called["model"] is False, "the model was called with no usable index, which would cost money and refuse"
    assert app.state.gate.in_flight == 0


def test_a_ready_service_answers(monkeypatch):
    use(monkeypatch, ready())

    async def fine(q):
        return "ok [a.pdf, Page 1]"

    monkeypatch.setattr(apimod, "arun_agent", fine)
    r = call(create_app(Settings()), "POST", "/query", json={"query": "how do I shut it down?"})
    assert r.status_code == 200


def test_a_service_that_was_not_ready_recovers_without_a_restart(monkeypatch):
    results = [not_ready(), ready()]
    monkeypatch.setattr(apimod, "compute_readiness", lambda require_manifest=True: results.pop(0))

    async def fine(q):
        return "ok [a.pdf, Page 1]"

    monkeypatch.setattr(apimod, "arun_agent", fine)
    app = create_app(Settings())
    assert call(app, "GET", "/ready").status_code == 503
    # a failed check is not trusted forever: once it is old, the next request checks again
    app.state.readiness.checked_at -= readiness_mod.RECHECK_AFTER_FAILURE_SECONDS + 1
    assert call(app, "GET", "/ready").status_code == 200
    assert call(app, "POST", "/query", json={"query": "how do I shut it down?"}).status_code == 200


def test_a_recent_failure_is_not_rechecked_on_every_probe(monkeypatch):
    calls = []

    def counting(require_manifest=True):
        calls.append(1)
        return not_ready()

    monkeypatch.setattr(apimod, "compute_readiness", counting)
    app = create_app(Settings())
    for _ in range(5):
        call(app, "GET", "/ready")
    assert len(calls) == 1


def test_startup_checks_the_index_and_warms_up_once_when_ready(monkeypatch):
    use(monkeypatch, ready())
    warmed = []
    monkeypatch.setattr(apimod, "_warm_up", lambda: warmed.append(1))
    with TestClient(create_app(Settings())) as client:  # entering runs the lifespan
        assert client.get("/ready").status_code == 200
    assert warmed == [1]


def test_startup_does_not_warm_up_or_crash_when_not_ready(monkeypatch):
    use(monkeypatch, not_ready())
    warmed = []
    monkeypatch.setattr(apimod, "_warm_up", lambda: warmed.append(1))
    with TestClient(create_app(Settings())) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
    assert warmed == []


# ---- the checks themselves ----


def test_needs_recheck_rules():
    assert needs_recheck(None)
    assert not needs_recheck(Readiness(checked_at=0.0))  # ready stays ready
    failed = Readiness(problems=["x"], checked_at=100.0)
    assert not needs_recheck(failed, now=105.0)
    assert needs_recheck(failed, now=111.0)


def test_compute_readiness_passes_for_a_good_index(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retriever.get_retriever", lambda: build(tmp_path))
    result = compute_readiness()
    assert result.ready, result.problems
    assert result.chunks == 6 and result.embedding_model == "model-a"
    assert result.index_version != "unknown"


def test_compute_readiness_fails_when_the_model_provider_is_not_configured(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retriever.get_retriever", lambda: build(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = compute_readiness()
    assert not result.ready
    assert "ANTHROPIC_API_KEY" in result.problems[0]


def test_compute_readiness_never_puts_a_secret_in_a_message(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retriever.get_retriever", lambda: build(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET_KEY)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.llm_factory.DEFAULT_LLM_PROVIDER", "openai")
    result = compute_readiness()
    assert not result.ready
    assert SECRET_KEY not in " ".join(result.problems)


def test_compute_readiness_fails_for_an_index_built_with_another_embedding_model(tmp_path, monkeypatch):
    build(tmp_path, name="model-a")
    other = reopen(tmp_path, name="model-b")
    monkeypatch.setattr("src.retriever.get_retriever", lambda: other)
    result = compute_readiness()
    assert not result.ready
    assert any("model-a" in p and "model-b" in p for p in result.problems)


def test_compute_readiness_reports_a_tampered_index_file_without_loading_it(monkeypatch):
    def refuse():
        raise IndexIntegrityError("bm25_index.pkl does not match the index manifest, so it was not loaded")

    monkeypatch.setattr("src.retriever.get_retriever", refuse)
    result = compute_readiness()
    assert not result.ready
    assert any("bm25_index.pkl" in p for p in result.problems)


def test_compute_readiness_survives_an_index_that_cannot_be_opened(monkeypatch):
    def boom():
        raise RuntimeError("disk exploded at /secret/path with token abc")

    monkeypatch.setattr("src.retriever.get_retriever", boom)
    result = compute_readiness()
    assert not result.ready
    assert "RuntimeError" in result.problems[0]
    assert "/secret/path" not in " ".join(result.problems), "an internal detail reached the readiness message"


def test_the_manifest_requirement_can_be_relaxed_for_an_old_index(tmp_path, monkeypatch):
    retriever = build(tmp_path)
    (tmp_path / "index_manifest.json").unlink()
    monkeypatch.setattr("src.retriever.get_retriever", lambda: retriever)
    assert not compute_readiness(require_index_manifest=True).ready
    assert compute_readiness(require_index_manifest=False).ready


def test_the_manifest_setting_parses_booleans_and_rejects_nonsense():
    assert Settings.from_env({}).require_index_manifest is True
    assert Settings.from_env({"REQUIRE_INDEX_MANIFEST": "false"}).require_index_manifest is False
    assert Settings.from_env({"REQUIRE_INDEX_MANIFEST": "1"}).require_index_manifest is True
    with pytest.raises(SettingsError, match="REQUIRE_INDEX_MANIFEST"):
        Settings.from_env({"REQUIRE_INDEX_MANIFEST": "maybe"})
