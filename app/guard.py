"""Protection for the public Streamlit demo: a global query rate limit and an
optional shared-password gate.

The logic here is deliberately free of Streamlit calls so it can be unit tested.
app/main.py wraps it in the UI (a password box, a rate-limit notice). Both
controls are proportionate to a portfolio demo rather than enterprise auth: the
rate limit caps the whole app's query throughput, and so its Anthropic spend, on
a public URL, and the password keeps crawlers and casual traffic out when one is
set. Neither is active in local development unless you opt in.
"""

from __future__ import annotations

import hmac
import os

from src.ratelimit import TokenBucketLimiter

# One global bucket. It bounds the whole app's query rate, not a single user's,
# which is what actually caps cost on a shared public URL. Tunable by env.
_APP_LIMITER = TokenBucketLimiter(
    rate_per_sec=float(os.getenv("APP_RATE_LIMIT_RPS", "0.1")),
    capacity=int(os.getenv("APP_RATE_LIMIT_BURST", "10")),
)


def rate_limit_active() -> bool:
    """Enforce the query rate limit on the deployed demo (DEMO_MODE is set),
    not during local development."""
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")


def allow_query() -> bool:
    """Consume one global token per submitted query. True when the query is
    allowed, False when the app is over its rate limit."""
    if not rate_limit_active():
        return True
    return _APP_LIMITER.allow("global")


def required_password() -> str | None:
    """The shared demo password, or None when no gate is configured."""
    pw = os.getenv("APP_PASSWORD", "").strip()
    return pw or None


def password_ok(candidate: str) -> bool:
    """Constant-time check of a submitted password. True when no gate is set."""
    required = required_password()
    if required is None:
        return True
    return hmac.compare_digest((candidate or "").strip(), required)
