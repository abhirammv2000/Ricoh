"""Unit tests for LLM transport-resilience config (src/llm_factory.py).

These construct the model **offline** (no network call is made — building a
ChatAnthropic only validates config) and assert the retry/timeout knobs are
applied, so a future refactor cannot silently drop production resilience.
"""

from __future__ import annotations

import pytest

from src.llm_factory import (
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_TIMEOUT_SECONDS,
    get_llm,
)


@pytest.fixture(autouse=True)
def _dummy_key(monkeypatch):
    # A key must be present for construction; it is never used for a network
    # call in these tests (we never invoke the model).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-construction")


def test_default_retry_and_timeout_applied():
    """The production defaults reach the underlying client."""
    llm = get_llm()
    assert llm.max_retries == _DEFAULT_MAX_RETRIES
    assert llm.default_request_timeout == _DEFAULT_TIMEOUT_SECONDS


def test_explicit_values_override_defaults():
    """setdefault semantics: an explicit caller still wins over the defaults."""
    llm = get_llm(max_retries=0, timeout=5.0)
    assert llm.max_retries == 0
    assert llm.default_request_timeout == 5.0


def test_missing_api_key_raises_before_any_network_call():
    """No key is a clear, actionable error, not an obscure SDK failure."""
    import os

    os.environ.pop("ANTHROPIC_API_KEY", None)
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        get_llm()
