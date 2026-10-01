"""Token bucket rate limiter, in memory and per process.

It limits how fast one client can hit the API, so one caller can't run up the bill. A token bucket
allows a small burst and then holds a client to the steady rate, where a fixed window hands out a fresh
quota the moment the window ticks over. The state lives in the process, which is enough for a single
instance. Several replicas would need a shared store like Redis. The clock can be passed in so tests
don't have to sleep.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    last: float


class TokenBucketLimiter:
    """A token bucket per key (usually a client ip). Each refills at rate_per_sec up to capacity, and a request costs one token."""

    def __init__(
        self,
        rate_per_sec: float,
        capacity: int,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if rate_per_sec <= 0 or capacity <= 0 or max_keys <= 0:
            raise ValueError("rate_per_sec, capacity, and max_keys must be positive")
        self._rate = rate_per_sec
        self._capacity = float(capacity)
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _refill(self, bucket: _Bucket, now: float) -> None:
        elapsed = max(0.0, now - bucket.last)
        bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rate)
        bucket.last = now

    def allow(self, key: str) -> bool:
        """Spend a token for this key if there is one. Returns whether the request is allowed."""
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    self._evict_full(now)
                # a new client starts with a full bucket
                self._buckets[key] = _Bucket(tokens=self._capacity - 1.0, last=now)
                return True
            self._refill(bucket, now)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False

    def retry_after(self, key: str) -> float:
        """Seconds until this key has a token. Zero if it has one now or isn't tracked."""
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return 0.0
            self._refill(bucket, self._clock())
            if bucket.tokens >= 1.0:
                return 0.0
            return (1.0 - bucket.tokens) / self._rate

    def _evict_full(self, now: float) -> None:
        """Drop keys whose bucket is full again. Nothing is lost, since a full bucket is the same as a new client, and it
        stops the map growing forever."""
        stale = [
            key
            for key, bucket in self._buckets.items()
            if min(self._capacity, bucket.tokens + max(0.0, now - bucket.last) * self._rate)
            >= self._capacity
        ]
        for key in stale:
            del self._buckets[key]
