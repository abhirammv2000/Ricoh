"""src/instrumentation.py - Per-stage cost, token, and latency accounting.

Why this exists
───────────────
"It takes about 19 seconds" is not an engineering statement — it does not say
*where* the time goes, and it says nothing at all about money.  Without that
breakdown you cannot answer the two questions any reviewer will ask first:

    "What does one query cost you?"
    "Which stage would you optimise, and how do you know?"

This module records one **span** per LLM call — stage, model, tokens in/out,
cache hits, latency, derived cost — and aggregates them per question.  Every
optimisation claim elsewhere in this project is expected to cite these
numbers rather than assert an improvement.

Design decisions
────────────────
• **Token counts are ground truth; cost is derived.**  We record the tokens
  the API actually reported (``usage_metadata``) and multiply by a price
  table.  Prices drift, so the table is a dated snapshot and is treated as
  configuration, not fact — if it is stale, the token counts remain correct
  and only the dollar figure needs recomputing.

• **contextvars, not a global.**  A module-level mutable would break the
  moment anything runs concurrently (and parallelising retrieval is on the
  roadmap).  A ContextVar keeps each run's spans isolated without threading
  a recorder object through every function signature.

• **Failures are recorded, not swallowed.**  A call that raises still emits a
  span with ``error`` set, so a crash loop shows up as cost rather than as a
  gap in the data.

• **Instrumentation must not change behaviour.**  If usage metadata is
  missing, the span records zero tokens rather than raising: an accounting
  bug should never take down the pipeline it is measuring.
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

# Traces are append-only JSONL: cheap to write, greppable, and requiring no
# service to run the repo. A hosted backend (Langfuse / Phoenix) would give a
# UI, but would also make reproducing this project depend on someone else's
# account — so local-first is the default and export is left as an option.
TRACE_PATH: Path = PROJECT_ROOT / "traces" / "traces.jsonl"


# ── Price table ────────────────────────────────────────────────────
# USD per million tokens. Snapshot date: 2026-08-01.
#
# This is deliberately a plain dict rather than a live lookup: an eval run
# must be reproducible, and a price that changes underneath a stored result
# would make two runs incomparable. Update it explicitly and note the date.
#
# cache_read is billed at ~0.1x input; cache_creation at ~1.25x input for the
# default 5-minute TTL. We model both so that adding prompt caching later
# shows up as a cost *reduction* rather than as untracked spend.
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
}
PRICING_SNAPSHOT_DATE = "2026-08-01"


@dataclass
class Span:
    """One unit of work inside a request — an LLM call or a retrieval.

    Non-LLM spans (retrieval) carry zero tokens and zero cost but real
    latency, which is the point: without them the trace shows LLM time only
    and silently attributes retrieval latency to nothing.
    """

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
    # Free-form per-stage detail. For retrieval this carries the chunk IDs
    # and documents that fed the answer — "chunk attribution", i.e. the
    # ability to ask of any answer: which sources produced this?
    attributes: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""


@dataclass
class RunRecord:
    """All spans for a single request, under one trace id."""

    spans: list[Span] = field(default_factory=list)
    trace_id: str = ""
    query: str = ""
    started_at: str = ""

    # ---- aggregates -------------------------------------------------
    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.spans), 6)

    @property
    def llm_spans(self) -> list[Span]:
        """LLM spans only.

        Retrieval spans share the list but must be excluded from any
        LLM-specific count — including them silently inflated
        `llm_calls` from 1 to 2 the moment retrieval became traced.
        """
        return [s for s in self.spans if s.span_type == "llm"]

    @property
    def total_llm_seconds(self) -> float:
        return round(sum(s.latency_seconds for s in self.llm_spans), 3)

    @property
    def total_traced_seconds(self) -> float:
        """All traced work, LLM and retrieval alike."""
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
        """Per-stage rollup — this is what makes the numbers actionable.

        A single total tells you the system is slow; the rollup tells you
        which node to attack and what the ceiling on that fix is.
        """
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


# Isolated per execution context so concurrent runs cannot interleave spans.
_CURRENT: contextvars.ContextVar[RunRecord | None] = contextvars.ContextVar(
    "citera_run_record", default=None
)


class record_run:
    """Context manager collecting every instrumented call inside it.

    Usage::

        with record_run(query="...") as rec:
            ...                     # agent executes
        rec.total_cost_usd

    When ``persist`` is true the finished trace is appended to
    ``traces/traces.jsonl``.  Persisting is what turns instrumentation into
    observability: a number you printed once cannot be gone back to, but a
    stored trace lets you answer "why did *that* request behave that way?"
    after the fact — which is the question production debugging actually
    asks.  Inspect with ``python -m src.trace_view``.
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
                # Losing a trace must never take down the request it traces.
                pass
        return None


class span:
    """Record a non-LLM unit of work (currently: retrieval).

    Without this, a trace accounts only for LLM time and silently drops
    everything else, which makes the latency breakdown wrong in a way that
    is invisible — the percentages still add to 100%.
    """

    def __init__(self, stage: str, **attributes: Any) -> None:
        self.stage = stage
        self.attributes = attributes
        self._started = 0.0
        self._span: Span | None = None

    def __enter__(self) -> "span":
        self._started = time.perf_counter()
        return self

    def set(self, **attributes: Any) -> None:
        """Attach detail discovered during the span (e.g. what was retrieved)."""
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
    # Tolerate date-suffixed ids (e.g. claude-haiku-4-5-20251001) by longest
    # known-prefix match, so an unfamiliar snapshot id still costs correctly.
    for known in sorted(PRICING, key=len, reverse=True):
        if model.startswith(known):
            return PRICING[known]
    return None


def _cost(model: str, usage: dict[str, Any]) -> float:
    p = _price(model)
    if p is None:
        return 0.0  # unknown model: report tokens, decline to invent a price
    details = usage.get("input_token_details") or {}
    cache_read = int(details.get("cache_read", 0) or 0)
    cache_write = int(details.get("cache_creation", 0) or 0)
    # LangChain reports input_tokens as the UNCACHED portion, so cached
    # tokens are billed separately rather than double-counted here.
    plain_in = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    return (
        plain_in * p.input_per_mtok
        + cache_read * p.cache_read_per_mtok
        + cache_write * p.cache_write_per_mtok
        + out * p.output_per_mtok
    ) / 1_000_000


def invoke(llm: Any, prompt: str, stage: str) -> str:
    """Invoke an LLM, record a span, and return its text.

    Drop-in replacement for ``response_text(llm.invoke(prompt))``.  When no
    ``record_run()`` is active this is exactly that call plus a timer, so
    library use outside the harness is unaffected.
    """
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
