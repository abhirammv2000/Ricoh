"""Tests for the opt-in LangSmith tracing setup (src/config._configure_langsmith).

Offline. The autouse fixture fully snapshots and restores the relevant env vars,
so the function can write to os.environ directly without leaking across tests.
"""

from __future__ import annotations

import os

import pytest

from src.config import _configure_langsmith

_KEYS = [
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_PROJECT",
]


@pytest.fixture(autouse=True)
def clean_langsmith_env():
    saved = {k: os.environ.get(k) for k in _KEYS}
    for k in _KEYS:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_off_by_default():
    _configure_langsmith()
    assert not os.environ.get("LANGCHAIN_TRACING_V2")
    assert not os.environ.get("LANGCHAIN_PROJECT")


def test_opt_in_enables_and_defaults_project():
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = "lsv2_test"
    _configure_langsmith()
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "citera"


def test_opt_in_respects_custom_project():
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = "lsv2_test"
    os.environ["LANGCHAIN_PROJECT"] = "my-project"
    _configure_langsmith()
    assert os.environ["LANGCHAIN_PROJECT"] == "my-project"


def test_legacy_flag_name_also_works():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "lsv2_test"
    _configure_langsmith()
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_enabled_without_key_still_sets_flag():
    # No API key: tracing is still switched on (the warning path), which is what
    # lets a misconfiguration show up in the logs rather than silently no-op.
    os.environ["LANGSMITH_TRACING"] = "true"
    _configure_langsmith()
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
