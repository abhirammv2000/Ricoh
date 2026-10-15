"""What a caller is told when something fails, and what stays in the log.

A model provider's error text can hold request details, account limits or a key fragment, so none of it
goes to the caller. The caller gets a stable code, a short sentence and, where retrying makes sense, a
Retry-After. The full error is logged with the request id so it can be found.

The classes of provider errors differ between SDKs (anthropic, openai), so this reads the attributes they
share (a status code, a name that says timeout or connection) and imports neither.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class PublicError:
    status: int
    code: str
    message: str
    retry_after: int | None = None


class ApiError(Exception):
    """Raised inside a request to answer with a specific public error."""

    def __init__(self, error: PublicError) -> None:
        super().__init__(error.message)
        self.error = error


OVERLOADED = PublicError(503, "overloaded", "The service is busy. Try again in a moment.", retry_after=1)
TIMEOUT = PublicError(504, "timeout", "The answer took too long. Try again.", retry_after=5)
PROVIDER_UNAVAILABLE = PublicError(
    503, "model_unavailable", "The language model is temporarily unavailable. Try again shortly.", retry_after=10
)
PROVIDER_REJECTED = PublicError(502, "model_error", "The language model could not answer this request.")
NOT_READY = PublicError(
    503, "not_ready", "The service is not ready to answer questions. Try again shortly.", retry_after=10
)
MISCONFIGURED = PublicError(500, "service_misconfigured", "The service is not configured correctly.")
INTERNAL = PublicError(500, "internal_error", "Something went wrong.")


def classify_error(exc: BaseException) -> PublicError:
    """Map any exception from the agent to the error the caller sees."""
    if isinstance(exc, ApiError):
        return exc.error
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return TIMEOUT

    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if isinstance(status, int):
        if status in (401, 403):
            # our own credential is wrong. The caller cannot fix that, and we need to hear about it
            return MISCONFIGURED
        if status == 408:
            return TIMEOUT
        if status == 429 or status >= 500:
            # 529 is the provider's "overloaded". Both clear on their own
            return PROVIDER_UNAVAILABLE
        return PROVIDER_REJECTED
    if "Timeout" in name:
        return TIMEOUT
    if "Connection" in name:
        return PROVIDER_UNAVAILABLE
    return INTERNAL
