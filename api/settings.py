"""Settings for the HTTP API, read from the environment once at startup.

Every value has a default that is safe for one small instance. A value that is set but wrong (a negative
timeout, text where a number goes) stops the service at startup with a message that names the variable.
Finding that out on the first request, in front of a user, is worse.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


class SettingsError(ValueError):
    """An environment variable has a value the service cannot use."""


def _number(env: Mapping[str, str], name: str, default: float, *, minimum: float, integer: bool = False) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        kind = "a whole number" if integer else "a number"
        raise SettingsError(f"{name} must be {kind}, got {raw!r}") from None
    if value < minimum:
        raise SettingsError(f"{name} must be at least {minimum}, got {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    # the longest one question may take, from the request arriving to the answer (or the stream ending)
    request_timeout_seconds: float = 30.0
    # how many questions are handled at once. Past this a request waits up to queue_timeout_seconds for a
    # slot and is then turned away with a 503, so a burst gets fast failures and the admitted work stays quick
    max_concurrent_queries: int = 16
    queue_timeout_seconds: float = 2.0
    # per client, requests a second with a burst
    rate_limit_rps: float = 1.0
    rate_limit_burst: int = 10

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        defaults = cls()
        return cls(
            request_timeout_seconds=_number(
                env, "REQUEST_TIMEOUT_SECONDS", defaults.request_timeout_seconds, minimum=0.1
            ),
            max_concurrent_queries=int(
                _number(env, "MAX_CONCURRENT_QUERIES", defaults.max_concurrent_queries, minimum=1, integer=True)
            ),
            queue_timeout_seconds=_number(
                env, "QUEUE_TIMEOUT_SECONDS", defaults.queue_timeout_seconds, minimum=0.0
            ),
            rate_limit_rps=_number(env, "RATE_LIMIT_RPS", defaults.rate_limit_rps, minimum=0.001),
            rate_limit_burst=int(
                _number(env, "RATE_LIMIT_BURST", defaults.rate_limit_burst, minimum=1, integer=True)
            ),
        )
