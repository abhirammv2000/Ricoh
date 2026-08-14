"""Token-bucket rate limiter, in memory and per process.

This bounds how fast a single client can hit the API, so one caller cannot run
up the bill or crowd everyone else out. A token bucket is used rather than a
fixed window because it smooths bursts: a client may spend a small reserve of
saved tokens at once and is then held to the steady refill rate, instead of
getting a whole fresh quota the instant an arbitrary window boundary ticks over.

Scope, stated plainly. This state lives in the process. It is the right tool for
a single instance and it is exactly enough for this deployment. Behind more than
one replica the buckets would have to move to a shared store such as Redis, and
the class is kept deliberately small so that swap is a contained change rather
than a rewrite. The clock is injectable so the behavior can be tested without
sleeping.
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
    """A per-key token bucket. Each key (typically a client IP) refills at
    ``rate_per_sec`` tokens a second up to ``capacity``, and each allowed request
    spends one token."""

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
        """Consume a token for ``key`` if one is available. Returns whether the
        request is permitted."""
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    self._evict_full(now)
                # A new client starts with a full bucket, then spends one token.
                self._buckets[key] = _Bucket(tokens=self._capacity - 1.0, last=now)
                return True
            self._refill(bucket, now)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False

    def retry_after(self, key: str) -> float:
        """Seconds until at least one token is available for ``key``. Zero if a
        token is available now or the key is untracked."""
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return 0.0
            self._refill(bucket, self._clock())
            if bucket.tokens >= 1.0:
                return 0.0
            return (1.0 - bucket.tokens) / self._rate

    def _evict_full(self, now: float) -> None:
        """Drop keys whose bucket has refilled to full. This is lossless: a full
        bucket is indistinguishable from an untracked new client, since both
        start the next request with a full bucket. It keeps the map from growing
        without bound when many distinct clients pass through."""
        stale = [
            key
            for key, bucket in self._buckets.items()
            if min(self._capacity, bucket.tokens + max(0.0, now - bucket.last) * self._rate)
            >= self._capacity
        ]
        for key in stale:
            del self._buckets[key]
