"""Per-stage cost, token and latency tracking.

Records one span per LLM call (stage, model, tokens, cache hits, latency, cost) and adds them up
per question. Token counts come from what the API reports and cost is worked out from a price table.
Spans sit in a ContextVar so concurrent runs don't mix. A call that raises still records a span,
and missing usage data counts as zero tokens instead of raising.
"""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.llm_factory import response_text

# traces are appended to a jsonl file, so nothing needs a hosted service to run the repo
TRACE_PATH: Path = PROJECT_ROOT / "traces" / "traces.jsonl"


# prices in USD per million tokens, as of 2026-08-01. It's a fixed table so two eval runs stay
# comparable, update it by hand. Cache reads cost about 0.1x the input price and cache writes
# about 1.25x.
@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float

    @property
    def cache_read_per_mtok(self) -> float:
        return self.input_per_mtok * 0.1

    @property
    def cache_write_per_mtok(self) -> float:
        return self.input_per_mtok * 1.25


PRICING: dict[str, ModelPrice] = {
    "claude-fable-5": ModelPrice(10.0, 50.0),
    "claude-opus-5": ModelPrice(5.0, 25.0),
    "claude-opus-4-8": ModelPrice(5.0, 25.0),
    "claude-opus-4-7": ModelPrice(5.0, 25.0),
    "claude-opus-4-6": ModelPrice(5.0, 25.0),
    "claude-sonnet-5": ModelPrice(3.0, 15.0),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0),
    # non-Anthropic models, only for the provider bakeoff. _price() tries the longest key
    # first, so "gpt-4o-mini" wins over "gpt-4o"
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4o": ModelPrice(2.50, 10.0),
    "gemini-3.6-flash": ModelPrice(0.10, 0.40),
    "gemini-2.0-flash": ModelPrice(0.10, 0.40),
    "gemini-1.5-flash": ModelPrice(0.075, 0.30),
}
PRICING_SNAPSHOT_DATE = "2026-08-01"


@dataclass
class Span:
    """One step inside a request, either an LLM call or a retrieval (retrieval has latency but no tokens)."""

    stage: str
    model: str
    span_type: str = "llm"  # "llm" | "retrieval"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    # extra detail per stage. For retrieval it holds the chunk ids and documents behind the answer
    attributes: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""


@dataclass
class RunRecord:
    """All the spans for one request."""

    spans: list[Span] = field(default_factory=list)
    trace_id: str = ""
    query: str = ""
    started_at: str = ""

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.spans), 6)

    @property
    def llm_spans(self) -> list[Span]:
        """Just the LLM spans. Retrieval spans have to stay out of LLM counts."""
        return [s for s in self.spans if s.span_type == "llm"]

    @property
    def total_llm_seconds(self) -> float:
        return round(sum(s.latency_seconds for s in self.llm_spans), 3)

    @property
    def total_traced_seconds(self) -> float:
        """LLM and retrieval time together."""
        return round(sum(s.latency_seconds for s in self.spans), 3)

    @property
    def llm_calls(self) -> int:
        return len(self.llm_spans)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.llm_spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.llm_spans)

    def by_stage(self) -> dict[str, dict[str, Any]]:
        """Calls, time, cost and tokens per stage."""
        out: dict[str, dict[str, Any]] = {}
        for s in self.spans:
            agg = out.setdefault(
                s.stage,
                {"calls": 0, "seconds": 0.0, "cost_usd": 0.0,
                 "input_tokens": 0, "output_tokens": 0},
            )
            agg["calls"] += 1
            agg["seconds"] += s.latency_seconds
            agg["cost_usd"] += s.cost_usd
            agg["input_tokens"] += s.input_tokens
            agg["output_tokens"] += s.output_tokens
        for agg in out.values():
            agg["seconds"] = round(agg["seconds"], 3)
            agg["cost_usd"] = round(agg["cost_usd"], 6)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "started_at": self.started_at,
            "llm_calls": self.llm_calls,
            "total_cost_usd": self.total_cost_usd,
            "total_llm_seconds": self.total_llm_seconds,
            "total_traced_seconds": self.total_traced_seconds,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "by_stage": self.by_stage(),
            "spans": [asdict(s) for s in self.spans],
        }


# one per execution context, so concurrent runs don't mix their spans
_CURRENT: contextvars.ContextVar[RunRecord | None] = contextvars.ContextVar(
    "citera_run_record", default=None
)


class record_run:
    """Collects every instrumented call made inside the with block.

        with record_run(query="...") as rec:
            ...
        rec.total_cost_usd

    With persist on, the finished trace is appended to traces/traces.jsonl (look at it with
    python -m src.trace_view).
    """

    def __init__(self, query: str = "", persist: bool = True) -> None:
        self.record = RunRecord(
            trace_id=uuid.uuid4().hex[:16],
            query=query,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._persist = persist
        self._token: contextvars.Token | None = None

    def __enter__(self) -> RunRecord:
        self._token = _CURRENT.set(self.record)
        return self.record

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _CURRENT.reset(self._token)
        if self._persist and self.record.spans:
            try:
                TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(TRACE_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self.record.to_dict(), ensure_ascii=False) + "\n")
            except OSError:
                # losing a trace shouldn't break the request
                pass
        return None


class span:
    """Record a step that isn't an LLM call (right now, retrieval), so its time shows up in the trace."""

    def __init__(self, stage: str, **attributes: Any) -> None:
        self.stage = stage
        self.attributes = attributes
        self._started = 0.0
        self._span: Span | None = None

    def __enter__(self) -> "span":
        self._started = time.perf_counter()
        return self

    def set(self, **attributes: Any) -> None:
        """Add detail found during the span, like what was retrieved."""
        self.attributes.update(attributes)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        rec = _CURRENT.get()
        if rec is None:
            return None
        rec.spans.append(
            Span(
                stage=self.stage,
                model="-",
                span_type="retrieval",
                latency_seconds=round(time.perf_counter() - self._started, 3),
                attributes=self.attributes,
                started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                error=f"{exc_type.__name__}: {exc}" if exc_type else None,
            )
        )
        return None


def _price(model: str) -> ModelPrice | None:
    if model in PRICING:
        return PRICING[model]
    # handle ids with a date suffix (claude-haiku-4-5-20251001) by the longest matching prefix
    for known in sorted(PRICING, key=len, reverse=True):
        if model.startswith(known):
            return PRICING[known]
    return None


def _cost(model: str, usage: dict[str, Any]) -> float:
    p = _price(model)
    if p is None:
        return 0.0  # unknown model, keep the tokens but don't guess a price
    details = usage.get("input_token_details") or {}
    cache_read = int(details.get("cache_read", 0) or 0)
    cache_write = int(details.get("cache_creation", 0) or 0)
    # langchain's input_tokens leaves out the cached part, so it isn't counted twice
    plain_in = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    return (
        plain_in * p.input_per_mtok
        + cache_read * p.cache_read_per_mtok
        + cache_write * p.cache_write_per_mtok
        + out * p.output_per_mtok
    ) / 1_000_000


def invoke_messages(llm: Any, messages: Any, stage: str) -> Any:
    """Call an LLM with a message list and return the raw response, recording a span.

    The tool loop needs the response object (the tool_calls are on it) and sends a growing list of
    messages, so it can't use invoke(). Going through here keeps its calls in the cost numbers.
    """
    rec = _CURRENT.get()
    model = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")

    started = time.perf_counter()
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        if rec is not None:
            rec.spans.append(
                Span(
                    stage=stage,
                    model=str(model),
                    latency_seconds=round(time.perf_counter() - started, 3),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        raise
    elapsed = time.perf_counter() - started

    if rec is not None:
        usage = getattr(response, "usage_metadata", None) or {}
        details = usage.get("input_token_details") or {}
        rec.spans.append(
            Span(
                stage=stage,
                model=str(model),
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                cache_read_tokens=int(details.get("cache_read", 0) or 0),
                cache_write_tokens=int(details.get("cache_creation", 0) or 0),
                latency_seconds=round(elapsed, 3),
                cost_usd=round(_cost(str(model), usage), 6),
            )
        )

    return response


def invoke(llm: Any, prompt: str, stage: str) -> str:
    """Call an LLM, record a span and return the text. Same as response_text(llm.invoke(prompt)) when nothing is recording."""
    rec = _CURRENT.get()
    model = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")

    started = time.perf_counter()
    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        if rec is not None:
            rec.spans.append(
                Span(
                    stage=stage,
                    model=str(model),
                    latency_seconds=round(time.perf_counter() - started, 3),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        raise
    elapsed = time.perf_counter() - started

    if rec is not None:
        usage = getattr(response, "usage_metadata", None) or {}
        details = usage.get("input_token_details") or {}
        rec.spans.append(
            Span(
                stage=stage,
                model=str(model),
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                cache_read_tokens=int(details.get("cache_read", 0) or 0),
                cache_write_tokens=int(details.get("cache_creation", 0) or 0),
                latency_seconds=round(elapsed, 3),
                cost_usd=round(_cost(str(model), usage), 6),
            )
        )

    return response_text(response)


async def ainvoke(llm: Any, prompt: str, stage: str) -> str:
    """Async version of invoke(), same span. It only helps if the model's ainvoke is truly async (ChatAnthropic's is)."""
    rec = _CURRENT.get()
    model = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")

    started = time.perf_counter()
    try:
        response = await llm.ainvoke(prompt)
    except Exception as exc:
        if rec is not None:
            rec.spans.append(
                Span(
                    stage=stage,
                    model=str(model),
                    latency_seconds=round(time.perf_counter() - started, 3),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        raise
    elapsed = time.perf_counter() - started

    if rec is not None:
        usage = getattr(response, "usage_metadata", None) or {}
        details = usage.get("input_token_details") or {}
        rec.spans.append(
            Span(
                stage=stage,
                model=str(model),
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                cache_read_tokens=int(details.get("cache_read", 0) or 0),
                cache_write_tokens=int(details.get("cache_creation", 0) or 0),
                latency_seconds=round(elapsed, 3),
                cost_usd=round(_cost(str(model), usage), 6),
            )
        )

    return response_text(response)
