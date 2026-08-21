"""Tests for the demo protection helpers (app/guard.py) and the password gate.

The helper tests are pure. The last test drives the Streamlit app headless to
confirm the gate actually stops the app before the chat renders.
"""

from __future__ import annotations

import app.guard as guard
from src.ratelimit import TokenBucketLimiter


def test_no_password_configured_allows_everything(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert guard.required_password() is None
    assert guard.password_ok("") is True
    assert guard.password_ok("whatever") is True


def test_password_required_and_checked(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    assert guard.required_password() == "s3cret"
    assert guard.password_ok("s3cret") is True
    assert guard.password_ok("wrong") is False
    assert guard.password_ok("") is False


def test_rate_limit_is_noop_outside_demo(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setattr(guard, "_APP_LIMITER", TokenBucketLimiter(rate_per_sec=0.001, capacity=1))
    assert all(guard.allow_query() for _ in range(5))


def test_rate_limit_enforced_in_demo(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setattr(guard, "_APP_LIMITER", TokenBucketLimiter(rate_per_sec=0.001, capacity=2))
    assert guard.allow_query() is True
    assert guard.allow_query() is True
    assert guard.allow_query() is False  # burst of 2 spent


def test_password_gate_stops_the_app_before_chat(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("APP_PASSWORD", "letmein")
    at = AppTest.from_file("app/main.py").run(timeout=60)
    assert not at.exception
    # The gate renders a password box and stops, so no chat input is reached.
    assert len(at.text_input) >= 1
    assert len(at.chat_input) == 0
