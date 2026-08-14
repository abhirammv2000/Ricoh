"""Unit tests for the token-bucket rate limiter (src/ratelimit.py).

A fake clock drives time, so the tests are deterministic and never sleep.
"""

from __future__ import annotations

import pytest

from src.ratelimit import TokenBucketLimiter


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_burst_up_to_capacity_then_blocks():
    clock = FakeClock()
    limiter = TokenBucketLimiter(rate_per_sec=1.0, capacity=3, clock=clock)
    assert [limiter.allow("ip") for _ in range(3)] == [True, True, True]
    assert limiter.allow("ip") is False


def test_tokens_refill_over_time():
    clock = FakeClock()
    limiter = TokenBucketLimiter(rate_per_sec=2.0, capacity=2, clock=clock)
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is False
    clock.advance(0.5)  # 0.5s * 2 tokens/s = 1 token back
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is False


def test_capacity_is_a_ceiling():
    clock = FakeClock()
    limiter = TokenBucketLimiter(rate_per_sec=5.0, capacity=2, clock=clock)
    clock.advance(100)  # idle a long time; tokens must not exceed capacity
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is False


def test_keys_are_independent():
    clock = FakeClock()
    limiter = TokenBucketLimiter(rate_per_sec=1.0, capacity=1, clock=clock)
    assert limiter.allow("a") is True
    assert limiter.allow("a") is False
    assert limiter.allow("b") is True  # b has its own bucket


def test_retry_after_reports_wait_when_blocked():
    clock = FakeClock()
    limiter = TokenBucketLimiter(rate_per_sec=2.0, capacity=1, clock=clock)
    assert limiter.allow("ip") is True
    assert limiter.allow("ip") is False
    # One token at 2/s means about half a second to wait.
    assert limiter.retry_after("ip") == pytest.approx(0.5, abs=1e-6)


def test_retry_after_is_zero_when_allowed():
    limiter = TokenBucketLimiter(rate_per_sec=1.0, capacity=2)
    assert limiter.retry_after("never-seen") == 0.0


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        TokenBucketLimiter(rate_per_sec=0.0, capacity=1)
    with pytest.raises(ValueError):
        TokenBucketLimiter(rate_per_sec=1.0, capacity=0)


def test_full_buckets_are_evicted_to_bound_memory():
    clock = FakeClock()
    limiter = TokenBucketLimiter(rate_per_sec=1.0, capacity=1, clock=clock, max_keys=2)
    limiter.allow("a")  # a is now empty (tokens 0)
    clock.advance(10)   # a refills to full
    limiter.allow("b")  # b empty; map holds a (full) and b (empty)
    limiter.allow("c")  # inserting c triggers eviction of the full bucket a
    assert "a" not in limiter._buckets
    assert set(limiter._buckets) <= {"b", "c"}
