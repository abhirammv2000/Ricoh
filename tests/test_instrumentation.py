"""Unit tests for the async instrumentation path (src/instrumentation.py).

invoke() already has coverage through the agent node tests. ainvoke() is new
and gets it directly: same span-recording contract, but through await and a
fake whose ainvoke() is itself a coroutine, so a test failure here would catch
a real "forgot to await" or "span never recorded" mistake rather than assume
the sync and async paths behave identically just because the code looks
parallel.

No pytest-asyncio dependency: the project has no async test infrastructure
yet, so each test wraps its coroutine in asyncio.run() rather than adding a
plugin and an asyncio_mode setting for four tests.
"""

from __future__ import annotations

import asyncio

import pytest

from src.instrumentation import ainvoke, record_run
from tests.helpers import FakeLLM


def test_ainvoke_returns_response_text():
    async def body():
        llm = FakeLLM("the answer")
        with record_run(query="q", persist=False):
            text = await ainvoke(llm, "prompt", stage="synthesizer")
        assert text == "the answer"
        assert llm.calls == ["prompt"]

    asyncio.run(body())


def test_ainvoke_records_a_span_on_the_active_run():
    async def body():
        llm = FakeLLM("grounded answer")
        with record_run(query="q", persist=False) as rec:
            await ainvoke(llm, "prompt", stage="synthesizer")
        assert len(rec.spans) == 1
        span = rec.spans[0]
        assert span.stage == "synthesizer"
        assert span.error is None
        assert span.latency_seconds >= 0.0

    asyncio.run(body())


def test_ainvoke_is_a_pure_timer_with_no_active_run():
    # No record_run() context: exactly what the sync invoke() promises for the
    # same case, so ainvoke() must not require one either.
    async def body():
        llm = FakeLLM("answer")
        text = await ainvoke(llm, "prompt", stage="synthesizer")
        assert text == "answer"

    asyncio.run(body())


class _FailingLLM:
    async def ainvoke(self, prompt: str):
        raise RuntimeError("upstream call failed")


def test_ainvoke_records_an_error_span_and_reraises():
    async def body():
        llm = _FailingLLM()
        with record_run(query="q", persist=False) as rec:
            with pytest.raises(RuntimeError, match="upstream call failed"):
                await ainvoke(llm, "prompt", stage="synthesizer")
        assert len(rec.spans) == 1
        assert rec.spans[0].error is not None
        assert "upstream call failed" in rec.spans[0].error

    asyncio.run(body())
