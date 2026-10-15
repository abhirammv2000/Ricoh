"""The daily spend cap: it counts what the model cost, stops spending when the limit is reached, and resets at midnight UTC."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

import api.main as apimod
from api.budget import BUDGET_EXCEEDED, DailyBudget
from api.main import create_app
from api.readiness import Readiness
from api.settings import Settings
from src import instrumentation
from src.instrumentation import Span
from src.ratelimit import TokenBucketLimiter


def at(day: int, hour: int = 12) -> float:
    return datetime(2026, 10, day, hour, 0, 0, tzinfo=timezone.utc).timestamp()


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# ---- the budget on its own ----


def test_spend_adds_up_and_the_limit_is_reached_exactly():
    budget = DailyBudget(1.0, clock=Clock(at(14)))
    budget.record(0.4)
    assert not budget.exceeded and budget.spent_usd == pytest.approx(0.4)
    budget.record(0.6)
    assert budget.exceeded, "spending exactly the limit should stop further spending"


def test_the_count_resets_when_the_utc_day_changes():
    clock = Clock(at(14, 23))
    budget = DailyBudget(1.0, clock=clock)
    budget.record(5.0)
    assert budget.exceeded
    clock.now = at(15, 0) + 1
    assert not budget.exceeded and budget.spent_usd == 0.0


def test_a_limit_of_zero_turns_the_cap_off_and_a_negative_cost_cannot_reduce_the_count():
    off = DailyBudget(0.0, clock=Clock(at(14)))
    off.record(1000.0)
    assert not off.enabled and not off.exceeded
    budget = DailyBudget(1.0, clock=Clock(at(14)))
    budget.record(0.5)
    budget.record(-10.0)
    assert budget.spent_usd == pytest.approx(0.5)


def test_it_says_how_long_until_the_budget_resets():
    budget = DailyBudget(1.0, clock=Clock(at(14, 23)))
    assert 3500 < budget.seconds_until_reset() <= 3600


# ---- through the API ----


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    apimod._limiter = TokenBucketLimiter(rate_per_sec=10_000.0, capacity=10_000)
    monkeypatch.setattr(apimod, "compute_readiness", lambda require_manifest=True: Readiness(index_version="t", chunks=1))


class FakeRetriever:
    def retrieve(self, query, top_k, final_k):
        return [{"source_document": "network_guide.pdf", "page_number": 3, "text": "Open Settings.", "rrf_score": 0.03}]


def costly_agent(monkeypatch, cost):
    """An agent whose answer costs `cost` dollars, recorded the way a real model call is."""
    calls = []

    async def run(q):
        calls.append(q)
        instrumentation._CURRENT.get().spans.append(Span(stage="synthesizer", model="m", cost_usd=cost))
        return "Choose Network [network_guide.pdf, Page 3]."

    monkeypatch.setattr(apimod, "arun_agent", run)
    return calls


async def _call(app, method, url, **kwargs):
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def post(app, query="how do I configure the network?"):
    return asyncio.run(_call(app, "POST", "/query", json={"query": query}))


def test_spending_stops_when_the_daily_limit_is_reached_and_the_model_is_not_called_again(monkeypatch):
    calls = costly_agent(monkeypatch, cost=0.6)
    app = create_app(Settings(auth_disabled=True, daily_budget_usd=1.0, fallback_to_retrieval=False))
    assert post(app).status_code == 200  # 0.6 spent
    assert post(app).status_code == 200  # 1.2 spent: the limit was crossed by this one
    blocked = post(app)
    assert blocked.status_code == 503 and blocked.json()["code"] == "budget_exceeded"
    assert int(blocked.headers["retry-after"]) > 0
    assert len(calls) == 2, "the model was called after the limit was reached"


def test_with_the_fallback_on_the_caller_still_gets_the_passages_when_the_budget_is_spent(monkeypatch):
    calls = costly_agent(monkeypatch, cost=2.0)
    monkeypatch.setattr("src.agent.get_retriever", lambda: FakeRetriever())
    app = create_app(Settings(auth_disabled=True, daily_budget_usd=1.0))
    assert post(app).status_code == 200
    spent = post(app)
    assert spent.status_code == 200 and spent.json()["degraded"] is True
    assert spent.headers["x-citera-degraded"] == "budget_exceeded"
    assert len(calls) == 1


def test_a_budget_of_zero_never_blocks(monkeypatch):
    calls = costly_agent(monkeypatch, cost=100.0)
    app = create_app(Settings(auth_disabled=True, daily_budget_usd=0.0, fallback_to_retrieval=False))
    assert all(post(app).status_code == 200 for _ in range(3))
    assert len(calls) == 3


def test_a_failed_question_that_cost_something_still_counts_against_the_budget(monkeypatch):
    async def fails_after_spending(q):
        instrumentation._CURRENT.get().spans.append(Span(stage="synthesizer", model="m", cost_usd=3.0))
        raise RuntimeError("late failure")

    monkeypatch.setattr(apimod, "arun_agent", fails_after_spending)
    app = create_app(Settings(auth_disabled=True, daily_budget_usd=1.0, fallback_to_retrieval=False))
    assert post(app).status_code == 500
    assert app.state.budget.exceeded


def test_the_spend_and_the_limit_are_exported(monkeypatch):
    costly_agent(monkeypatch, cost=0.25)
    app = create_app(Settings(auth_disabled=True, daily_budget_usd=5.0, fallback_to_retrieval=False))
    post(app)
    text = asyncio.run(_call(app, "GET", "/metrics")).text
    assert "citera_budget_spent_usd 0.25" in text
    assert "citera_budget_limit_usd 5.0" in text


def test_the_public_error_for_a_spent_budget_says_nothing_about_amounts():
    assert "$" not in BUDGET_EXCEEDED.message and BUDGET_EXCEEDED.status == 503
