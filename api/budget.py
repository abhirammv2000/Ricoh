"""A cap on how much the service spends on the model in one day.

Every other limit in the service bounds how fast money can go out. This one bounds how much, in total. A
bug, a loop in a caller or a leaked key can all send a steady stream of valid requests, and a rate limit
alone lets that run all day.

The count is in memory and is per process, and the day is a UTC calendar day. That is honest and enough for
one instance. With several replicas each has its own count, so the real ceiling is the limit times the
number of replicas, and a restart resets the count. Moving it to a shared store is the next step for a
larger deployment, and the limit should then be set from the platform's own spend alert, not trusted alone.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from api.errors import PublicError

BUDGET_EXCEEDED = PublicError(
    503, "budget_exceeded", "The daily limit on model usage has been reached. Try again tomorrow.", retry_after=3600
)


def _utc_day(now: float) -> str:
    return datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")


class DailyBudget:
    def __init__(self, limit_usd: float, clock: Callable[[], float] = time.time) -> None:
        """limit_usd of 0 or less turns the cap off."""
        self.limit_usd = limit_usd
        self._clock = clock
        self._lock = threading.Lock()
        self._day = _utc_day(clock())
        self._spent = 0.0

    @property
    def enabled(self) -> bool:
        return self.limit_usd > 0

    def _roll(self) -> None:
        today = _utc_day(self._clock())
        if today != self._day:
            self._day, self._spent = today, 0.0

    @property
    def spent_usd(self) -> float:
        with self._lock:
            self._roll()
            return self._spent

    def record(self, cost_usd: float) -> None:
        with self._lock:
            self._roll()
            self._spent += max(0.0, cost_usd)

    @property
    def exceeded(self) -> bool:
        return self.enabled and self.spent_usd >= self.limit_usd

    def seconds_until_reset(self) -> int:
        now = datetime.fromtimestamp(self._clock(), UTC)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(1, int((midnight - now).total_seconds()))
