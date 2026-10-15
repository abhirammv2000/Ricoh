"""Shared pytest fixtures for the Citera test suite."""

from __future__ import annotations

import os

import pytest

# No test may reach a paid service. src.config loads .env, which can hold a real key, so these are set
# before anything imports it. A test that forgets to mock the model then fails at once with a connection
# error to a closed port, instead of spending money. Tests that need a provider use tests/fake_anthropic.py
# and set ANTHROPIC_BASE_URL themselves. Tracing is off so no question text is sent to a hosted service.
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-not-a-real-key"
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"
for _name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
    os.environ.pop(_name, None)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture
def make_page():
    """Factory for synthetic page dicts with N deterministic words."""

    def _make(n_words: int, page_number: int = 1, source: str = "doc.pdf"):
        text = " ".join(f"w{i}" for i in range(n_words))
        return {"text": text, "page_number": page_number, "source_document": source}

    return _make


def test_the_suite_cannot_reach_a_paid_api():
    """A guard on the guard: if someone removes the lines above, this fails before any test spends money."""
    # each check is a plain True or False first, because a failed assert prints the values it compared and
    # one of these is a key
    base_is_local = os.environ["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1")
    key_is_a_test_key = os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-not-a-real-key"
    tracing_is_off = os.environ["LANGSMITH_TRACING"] == "false"
    assert base_is_local, "ANTHROPIC_BASE_URL is not local"
    assert key_is_a_test_key, "ANTHROPIC_API_KEY is not the test key"
    assert tracing_is_off, "tracing is on"
