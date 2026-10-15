"""The metrics: what each one counts, and that they carry no question text and cannot multiply without limit."""

from __future__ import annotations

import asyncio
import re

import httpx
import pytest

import api.main as apimod
from api.main import create_app
from api.metrics import Metrics, endpoint_label
from api.readiness import Readiness
from api.settings import Settings
from src.agent import PROMPT_VERSION
from src.instrumentation import RunRecord, Span


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(
        apimod, "compute_readiness", lambda require_manifest=True: Readiness(index_version="idx42", chunks=88, embedding_model="m")
    )


async def _call(app, method, url, **kwargs):
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def call(app, method, url, **kwargs):
    return asyncio.run(_call(app, method, url, **kwargs))


def value(text: str, series: str) -> float:
    """The value of one series in the exposition text, or 0 if it is not there."""
    match = re.search(rf"^{re.escape(series)} ([0-9.e+-]+)$", text, re.M)
    return float(match.group(1)) if match else 0.0


def run_record(chunks, valid=True, input_tokens=1200, output_tokens=40, cost=0.0123, cache_read=0) -> RunRecord:
    rec = RunRecord(trace_id="t", query="irrelevant")
    rec.spans.append(
        Span(stage="retrieval", model="-", span_type="retrieval", attributes={"chunks": chunks})
    )
    rec.spans.append(
        Span(stage="synthesizer", model="m", input_tokens=input_tokens, output_tokens=output_tokens,
             cost_usd=cost, cache_read_tokens=cache_read)
    )
    rec.spans.append(
        Span(stage="citation_guardrail", model="-", span_type="retrieval", attributes={"valid": valid})
    )
    return rec


def chunk(rrf):
    return {"id": "c", "doc": "d.pdf", "page": 1, "rrf": rrf}


# ---- what observe_run takes from a trace ----


def test_a_run_records_retrieval_strength_tokens_and_cost():
    m = Metrics()
    m.observe_run(run_record([chunk(0.0328), chunk(0.0161)], cache_read=300), "answered")
    text = m.render()[0].decode()
    assert value(text, 'citera_answers_total{outcome="answered"}') == 1
    assert value(text, "citera_retrieved_chunks_count") == 1 and value(text, "citera_retrieved_chunks_sum") == 2
    assert value(text, "citera_retrieval_top_rrf_score_sum") == pytest.approx(0.0328)
    assert value(text, "citera_llm_calls_total") == 1
    assert value(text, 'citera_llm_tokens_total{kind="input"}') == 1200
    assert value(text, 'citera_llm_tokens_total{kind="output"}') == 40
    assert value(text, 'citera_llm_tokens_total{kind="cache_read"}') == 300
    assert value(text, "citera_llm_cost_usd_total") == pytest.approx(0.0123)


def test_an_empty_retrieval_is_counted_and_has_no_top_score():
    m = Metrics()
    m.observe_run(run_record([]), "refused")
    text = m.render()[0].decode()
    assert value(text, "citera_empty_retrievals_total") == 1
    assert value(text, "citera_retrieval_top_rrf_score_count") == 0
    assert value(text, 'citera_answers_total{outcome="refused"}') == 1


def test_an_answer_citing_a_document_that_was_not_retrieved_is_counted():
    m = Metrics()
    m.observe_run(run_record([chunk(0.02)], valid=True), "answered")
    assert value(m.render()[0].decode(), "citera_invalid_citations_total") == 0
    m.observe_run(run_record([chunk(0.02)], valid=False), "answered")
    assert value(m.render()[0].decode(), "citera_invalid_citations_total") == 1


def test_every_outcome_is_visible_from_the_start_so_a_rate_has_a_baseline():
    text = Metrics().render()[0].decode()
    for outcome in ("answered", "refused", "degraded", "error", "abandoned"):
        assert f'citera_answers_total{{outcome="{outcome}"}} 0.0' in text


def test_two_metrics_objects_do_not_share_state():
    a, b = Metrics(), Metrics()
    a.observe_run(run_record([chunk(0.02)]), "answered")
    assert value(b.render()[0].decode(), 'citera_answers_total{outcome="answered"}') == 0


# ---- through the HTTP API ----


def test_metrics_are_served_in_the_prometheus_text_format():
    r = call(create_app(Settings(auth_disabled=True)), "GET", "/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "# TYPE citera_http_requests_total counter" in r.text


def test_requests_are_counted_by_endpoint_and_status(monkeypatch):
    async def fine(q):
        return "Choose Network [a.pdf, Page 1]."

    monkeypatch.setattr(apimod, "arun_agent", fine)
    app = create_app(Settings(auth_disabled=True))
    call(app, "POST", "/query", json={"query": "how do I shut it down?"})
    call(app, "POST", "/query", json={"query": ""})  # fails validation: 422
    call(app, "GET", "/health")
    text = call(app, "GET", "/metrics").text
    assert value(text, 'citera_http_requests_total{endpoint="/query",status="200"}') == 1
    assert value(text, 'citera_http_requests_total{endpoint="/query",status="422"}') == 1
    assert value(text, 'citera_http_requests_total{endpoint="/health",status="200"}') == 1
    assert value(text, 'citera_http_request_seconds_count{endpoint="/query"}') == 2


def test_an_answer_and_a_refusal_are_counted_apart(monkeypatch):
    answers = iter(["Choose Network [a.pdf, Page 1].", "Information unavailable in provided documents."])

    async def scripted(q):
        return next(answers)

    monkeypatch.setattr(apimod, "arun_agent", scripted)
    app = create_app(Settings(auth_disabled=True))
    call(app, "POST", "/query", json={"query": "how do I shut it down?"})
    call(app, "POST", "/query", json={"query": "what is the meaning of life?"})
    text = call(app, "GET", "/metrics").text
    assert value(text, 'citera_answers_total{outcome="answered"}') == 1
    assert value(text, 'citera_answers_total{outcome="refused"}') == 1


def test_a_failed_question_is_an_error_outcome_and_an_error_code(monkeypatch):
    async def broken(q):
        raise RuntimeError("boom")

    monkeypatch.setattr(apimod, "arun_agent", broken)
    app = create_app(Settings(auth_disabled=True, fallback_to_retrieval=False))
    call(app, "POST", "/query", json={"query": "how do I shut it down?"})
    text = call(app, "GET", "/metrics").text
    assert value(text, 'citera_answers_total{outcome="error"}') == 1
    assert value(text, 'citera_errors_total{code="internal_error"}') == 1


def test_unknown_paths_share_one_label_so_a_scanner_cannot_create_series():
    app = create_app(Settings(auth_disabled=True))
    for path in ("/wp-admin", "/.env", "/a/b/c", "/x1", "/x2"):
        call(app, "GET", path)
    text = call(app, "GET", "/metrics").text
    assert value(text, 'citera_http_requests_total{endpoint="other",status="404"}') == 5
    assert "wp-admin" not in text and ".env" not in text
    assert endpoint_label("/query") == "/query" and endpoint_label("/nope") == "other"


def test_no_question_text_ever_reaches_the_metrics(monkeypatch):
    async def fine(q):
        return "ok [a.pdf, Page 1]."

    monkeypatch.setattr(apimod, "arun_agent", fine)
    app = create_app(Settings(auth_disabled=True))
    call(app, "POST", "/query", json={"query": "my printer serial number is PX-99-SECRET"})
    assert "SECRET" not in call(app, "GET", "/metrics").text


def test_readiness_and_what_is_being_served_are_exported():
    app = create_app(Settings(auth_disabled=True))
    call(app, "GET", "/ready")
    text = call(app, "GET", "/metrics").text
    assert value(text, "citera_ready") == 1
    assert 'citera_index_info{embedding_model="m",index_version="idx42"} 1.0' in text
    assert f'prompt_version="{PROMPT_VERSION}"' in text


def test_not_ready_is_exported_as_zero(monkeypatch):
    monkeypatch.setattr(
        apimod, "compute_readiness", lambda require_manifest=True: Readiness(problems=["empty"], checked_at=1e12)
    )
    app = create_app(Settings(auth_disabled=True))
    call(app, "GET", "/ready")
    assert value(call(app, "GET", "/metrics").text, "citera_ready") == 0


def test_a_trace_records_what_produced_the_answer(monkeypatch):
    seen = {}
    real = apimod.record_run

    class Spy(real):
        def __exit__(self, *exc):
            seen["meta"] = dict(self.record.meta)
            return super().__exit__(*exc)

    async def fine(q):
        return "ok [a.pdf, Page 1]."

    monkeypatch.setattr(apimod, "record_run", Spy)
    monkeypatch.setattr(apimod, "arun_agent", fine)
    app = create_app(Settings(auth_disabled=True))
    call(app, "GET", "/ready")
    call(app, "POST", "/query", json={"query": "how do I shut it down?"})
    assert seen["meta"]["index_version"] == "idx42"
    assert seen["meta"]["prompt_version"] == PROMPT_VERSION
    assert seen["meta"]["model"].startswith("anthropic:")
    assert seen["meta"]["pipeline"] == "retrieve+synthesize"


def test_the_trace_file_format_includes_the_meta_block():
    rec = RunRecord(trace_id="t", meta={"index_version": "x"})
    assert rec.to_dict()["meta"] == {"index_version": "x"}
