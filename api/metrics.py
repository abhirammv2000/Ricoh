"""Prometheus metrics for the API, at /metrics.

Two groups, for two readers:

  Service health, for whoever is on call: request counts and latency by endpoint, requests in flight, errors
  by public code, how often the service is full or not ready.

  RAG quality, for whoever owns answer quality. A retrieval system gets worse without any request failing,
  so latency and error rate cannot see it. These can:
    * answers by outcome (answered, refused, degraded, error). A rising refusal rate means the corpus no
      longer covers what people ask, or retrieval has broken.
    * how many chunks were retrieved and the top fused score. A fall in either is retrieval drifting.
    * answers whose citations name a document that was not retrieved. That should be zero.
    * tokens and dollars, which is how a change in prompt size or traffic shows up as cost.

Nothing user-supplied is ever a label, so the number of series is fixed. No question text is stored here.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, Info, generate_latest

# every path the service serves. Anything else is one label, so a scanner cannot create series
KNOWN_ENDPOINTS = frozenset({"/query", "/query/stream", "/health", "/ready", "/metrics"})

# abandoned is a stream whose client left before the end
OUTCOMES = ("answered", "refused", "degraded", "error", "abandoned")

# reciprocal rank fusion with k=60 scores a chunk at most 2/61 = 0.0328 (first in both lists)
_RRF_BUCKETS = (0.004, 0.008, 0.012, 0.016, 0.02, 0.024, 0.028, 0.0325)
_SECONDS_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 60)


def endpoint_label(path: str) -> str:
    return path if path in KNOWN_ENDPOINTS else "other"


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry

        self.requests = Counter(
            "citera_http_requests_total", "HTTP requests by endpoint and status.", ["endpoint", "status"], registry=r
        )
        self.request_seconds = Histogram(
            "citera_http_request_seconds", "Time to complete a request, until the last byte for a stream.",
            ["endpoint"], buckets=_SECONDS_BUCKETS, registry=r,
        )
        self.in_flight = Gauge("citera_in_flight_queries", "Questions being answered now.", registry=r)
        self.errors = Counter(
            "citera_errors_total", "Public errors by code (overloaded, not_ready, timeout, model_unavailable...).",
            ["code"], registry=r,
        )

        self.answers = Counter(
            "citera_answers_total", "Finished questions by outcome.", ["outcome"], registry=r
        )
        for outcome in OUTCOMES:
            self.answers.labels(outcome)  # show up as 0 from the start, so a rate() has something to compare

        self.retrieved_chunks = Histogram(
            "citera_retrieved_chunks", "Chunks handed to the model per question.",
            buckets=(0, 1, 2, 3, 4, 5, 7, 10), registry=r,
        )
        self.top_rrf = Histogram(
            "citera_retrieval_top_rrf_score", "Fused score of the best chunk per question.",
            buckets=_RRF_BUCKETS, registry=r,
        )
        self.empty_retrievals = Counter(
            "citera_empty_retrievals_total", "Questions for which retrieval found nothing.", registry=r
        )
        self.invalid_citations = Counter(
            "citera_invalid_citations_total", "Answers that cite a document that was not retrieved.", registry=r
        )

        self.llm_calls = Counter("citera_llm_calls_total", "Model calls.", registry=r)
        self.llm_tokens = Counter("citera_llm_tokens_total", "Model tokens by kind.", ["kind"], registry=r)
        self.llm_cost = Counter("citera_llm_cost_usd_total", "Estimated model spend in dollars.", registry=r)
        self.ttft = Histogram(
            "citera_time_to_first_token_seconds", "Time from the request to the first streamed token.",
            buckets=_SECONDS_BUCKETS, registry=r,
        )

        self.budget_spent = Gauge("citera_budget_spent_usd", "Model spend so far today (UTC), in dollars.", registry=r)
        self.budget_limit = Gauge("citera_budget_limit_usd", "The daily spend limit, 0 if there is none.", registry=r)
        self.ready = Gauge("citera_ready", "1 if the service can give a correct answer, else 0.", registry=r)
        self.index_info = Info("citera_index", "What the service is serving.", registry=r)
        self.build_info = Info("citera_build", "The model and prompt in use.", registry=r)

    # ---- what the service records ----

    def observe_http(self, path: str, status: int, seconds: float) -> None:
        endpoint = endpoint_label(path)
        self.requests.labels(endpoint, str(status)).inc()
        self.request_seconds.labels(endpoint).observe(seconds)

    def observe_error(self, code: str) -> None:
        self.errors.labels(code).inc()

    def set_readiness(self, ready: bool, index_version: str, embedding_model: str) -> None:
        self.ready.set(1 if ready else 0)
        self.index_info.info({"index_version": index_version, "embedding_model": embedding_model})

    def set_build(self, model: str, prompt_version: str) -> None:
        self.build_info.info({"model": model, "prompt_version": prompt_version})

    def observe_ttft(self, seconds: float | None) -> None:
        if seconds is not None:
            self.ttft.observe(seconds)

    def observe_run(self, record: Any, outcome: str) -> None:
        """Take the RAG signals from one finished question's trace. `outcome` is one of OUTCOMES."""
        self.answers.labels(outcome).inc()

        for span in record.spans:
            if span.stage == "retrieval" and span.span_type == "retrieval":
                chunks = span.attributes.get("chunks", [])
                self.retrieved_chunks.observe(len(chunks))
                if chunks:
                    self.top_rrf.observe(max(c.get("rrf", 0.0) for c in chunks))
                else:
                    self.empty_retrievals.inc()
            elif span.stage == "citation_guardrail" and span.attributes.get("valid") is False:
                self.invalid_citations.inc()

        self.llm_calls.inc(record.llm_calls)
        self.llm_tokens.labels("input").inc(record.total_input_tokens)
        self.llm_tokens.labels("output").inc(record.total_output_tokens)
        self.llm_tokens.labels("cache_read").inc(sum(s.cache_read_tokens for s in record.llm_spans))
        self.llm_cost.inc(record.total_cost_usd)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


class MetricsMiddleware:
    """Counts every request and times it until the last byte was sent, which for a stream is the end of the
    stream. A plain ASGI middleware, so it does not interfere with streaming or cancellation."""

    def __init__(self, app, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        import time

        started = time.perf_counter()
        status = {"code": 500}

        async def send_and_note(message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_and_note)
        finally:
            # a client that disconnects mid-stream ends up here too, with the status that was already sent
            self.metrics.observe_http(scope["path"], status["code"], time.perf_counter() - started)
