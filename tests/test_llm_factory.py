"""Unit tests for LLM transport-resilience config (src/llm_factory.py).

These construct the model offline (no network call is made, building a
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


# Cross-provider wiring. These build the client offline (no network) and only
# check that the right key is required and the right endpoint is used.


def test_openai_provider_builds_with_its_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-dummy")
    llm = get_llm(provider="openai")
    assert llm.model_name == "gpt-4o-mini"
    assert llm.openai_api_base in (None, "https://api.openai.com/v1")


def test_google_provider_uses_the_gemini_openai_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-dummy")
    llm = get_llm(provider="google")
    assert llm.model_name == "gemini-3.6-flash"
    assert "generativelanguage.googleapis.com" in llm.openai_api_base


def test_openai_provider_without_a_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        get_llm(provider="openai")


def test_google_provider_without_a_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        get_llm(provider="google")


def test_unknown_provider_still_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm(provider="fictional")


def test_self_hosted_provider_uses_its_base_url(monkeypatch):
    monkeypatch.setenv("SELF_HOSTED_LLM_BASE_URL", "http://localhost:8000/v1")
    llm = get_llm(provider="self_hosted")
    assert llm.model_name == "citera-finetuned"
    assert llm.openai_api_base == "http://localhost:8000/v1"


def test_self_hosted_provider_without_a_base_url_raises(monkeypatch):
    monkeypatch.delenv("SELF_HOSTED_LLM_BASE_URL", raising=False)
    with pytest.raises(EnvironmentError, match="SELF_HOSTED_LLM_BASE_URL"):
        get_llm(provider="self_hosted")


# ---- shared_llm: one client per configuration ----

import asyncio

from src.llm_factory import clear_shared_llms, shared_llm


@pytest.fixture
def anthropic_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-1")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    clear_shared_llms()
    yield
    clear_shared_llms()


def test_shared_llm_returns_the_same_client_for_the_same_arguments(anthropic_env):
    assert shared_llm() is shared_llm()


def test_shared_llm_matches_get_llm_settings(anthropic_env):
    llm = shared_llm()
    assert llm.max_retries == 3
    assert llm.default_request_timeout == 60.0


def test_shared_llm_gives_a_different_client_for_a_different_model_or_setting(anthropic_env):
    base = shared_llm()
    assert shared_llm(model="claude-haiku-4-5") is not base
    assert shared_llm(max_tokens=100) is not base
    assert shared_llm(timeout=5.0) is not base


def test_shared_llm_does_not_hand_out_a_client_built_for_another_key_or_endpoint(anthropic_env, monkeypatch):
    first = shared_llm()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-2")
    assert shared_llm() is not first
    other_key = shared_llm()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    assert shared_llm() is not other_key


def test_shared_llm_keeps_one_client_per_event_loop(anthropic_env):
    # an async client's connection pool belongs to its loop, so a second loop must not get the first one's
    async def grab():
        return shared_llm()

    first = asyncio.run(grab())
    second = asyncio.run(grab())
    assert first is not second
    inside_one_loop = asyncio.run(_twice())
    assert inside_one_loop


async def _twice():
    return shared_llm() is shared_llm()


def test_clear_shared_llms_forgets_clients(anthropic_env):
    first = shared_llm()
    clear_shared_llms()
    assert shared_llm() is not first


def test_shared_llm_does_not_keep_the_api_key_in_its_cache_key(anthropic_env):
    from src import llm_factory

    shared_llm()
    assert all("sk-test-1" not in repr(key) for key in llm_factory._shared)
