"""Settings for the HTTP API, read from the environment once at startup.

Every value has a default that is safe for one small instance. A value that is set but wrong (a negative timeout, text
where a number goes) stops the service at startup with a message that names the variable. That is better than finding
out on the first request, in front of a user.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field


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


def _choice(env: Mapping[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw not in allowed:
        raise SettingsError(f"{name} must be one of {', '.join(allowed)}, got {raw!r}")
    return raw


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise SettingsError(f"{name} must be true or false, got {raw!r}")


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
    # serve only from an index that has a manifest and passes its checks. Turn off only for an index built
    # before manifests existed that you cannot rebuild
    require_index_manifest: bool = True
    # when the model provider is down or too slow, answer with the passages retrieval found instead of an error
    fallback_to_retrieval: bool = True
    # who may call the service. A comma separated list, each `label:secret` or just a secret. Not shown
    # when settings are printed
    api_keys: str = field(default="", repr=False)
    # only for a laptop. With no keys and this off, the service refuses to start
    auth_disabled: bool = False
    # the most the service will spend on the model in one UTC day, in dollars. 0 turns it off
    daily_budget_usd: float = 20.0
    # text, or json (one object per line, with the request id). The container image sets json
    log_format: str = "text"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
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
            require_index_manifest=_flag(env, "REQUIRE_INDEX_MANIFEST", defaults.require_index_manifest),
            fallback_to_retrieval=_flag(env, "FALLBACK_TO_RETRIEVAL", defaults.fallback_to_retrieval),
            api_keys=env.get("API_KEYS", ""),
            auth_disabled=_flag(env, "API_AUTH_DISABLED", defaults.auth_disabled),
            daily_budget_usd=_number(env, "DAILY_BUDGET_USD", defaults.daily_budget_usd, minimum=0.0),
            log_format=_choice(env, "LOG_FORMAT", defaults.log_format, ("text", "json")),
        )
