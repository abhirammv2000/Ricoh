"""API key authentication, and who a request is for the purposes of rate limiting.

API_KEYS holds one or more keys, separated by commas. A key can carry a label, `partner-a:the-secret`, and a
label is what shows up in logs and what a client's rate limit is counted against. A key with no label gets
one made from a hash of the key, which names it without giving it away. More than one key is how a key is
rotated: add the new one, move callers over, then remove the old one.

Two details that are easy to get wrong:

* Keys are compared as SHA-256 digests with hmac.compare_digest, and every key is checked on every request.
  The comparison does not stop at the first differing byte, and the time taken does not say which key
  matched or how long the right one is.
* Failed attempts are counted per source address and limited, so the key cannot be guessed at speed. That
  is the one place the client address is used. A caller who authenticated is limited by their key, since
  behind a proxy every caller shares one address.

Authentication is on by default. The service will not start with no keys unless API_AUTH_DISABLED=true,
which is for a laptop. A default that quietly leaves the service open is how it ends up open in production.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from fastapi import Request

from api.errors import ApiError, PublicError
from src.ratelimit import TokenBucketLimiter

API_KEY_HEADER = "X-API-Key"

UNAUTHORIZED = PublicError(401, "unauthorized", "A valid API key is required.")
TOO_MANY_FAILURES = PublicError(
    429, "too_many_failed_attempts", "Too many failed attempts. Wait and try again.", retry_after=60
)

# these answer a probe or a load balancer, so they work with no key
OPEN_PATHS = frozenset({"/health", "/ready"})


@dataclass(frozen=True)
class Client:
    """Who a request is for. The label is safe to log."""

    label: str


ANONYMOUS = Client(label="anonymous")


@dataclass(frozen=True)
class ApiKey:
    label: str
    digest: bytes


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def parse_keys(raw: str) -> list[ApiKey]:
    """Turn the API_KEYS string into keys. Blank entries are dropped. Raises ValueError for a duplicate label."""
    keys: list[ApiKey] = []
    seen: set[str] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        label, sep, secret = entry.partition(":")
        if not sep:
            secret, label = entry, "key-" + _digest(entry).hex()[:8]
        label, secret = label.strip(), secret.strip()
        if not secret:
            raise ValueError(f"the API key labelled {label!r} is empty")
        if label in seen:
            raise ValueError(f"two API keys share the label {label!r}")
        seen.add(label)
        keys.append(ApiKey(label=label, digest=_digest(secret)))
    return keys


def match_key(supplied: str, keys: list[ApiKey]) -> Client | None:
    """The client whose key matches, or None. Every key is compared, so timing does not say which one matched."""
    supplied_digest = _digest(supplied)
    found: Client | None = None
    for key in keys:
        if hmac.compare_digest(supplied_digest, key.digest) and found is None:
            found = Client(label=key.label)
    return found


def supplied_key(request: Request) -> str:
    """The key from X-API-Key, or from an Authorization: Bearer header, which is what scrapers can send."""
    header = request.headers.get(API_KEY_HEADER)
    if header:
        return header.strip()
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


class Authenticator:
    def __init__(self, keys: list[ApiKey], enabled: bool, failures: TokenBucketLimiter | None = None) -> None:
        self.keys = keys
        self.enabled = enabled
        # ten wrong keys in a burst from one address, then one more a minute
        self.failures = failures or TokenBucketLimiter(rate_per_sec=1 / 60, capacity=10)

    def __call__(self, request: Request) -> Client:
        """A dependency. Returns the client, or raises 401 (wrong or no key) or 429 (too many wrong keys)."""
        if not self.enabled or request.url.path in OPEN_PATHS:
            client = ANONYMOUS
        else:
            source = request.client.host if request.client else "unknown"
            supplied = supplied_key(request)
            client = match_key(supplied, self.keys) if supplied else None
            if client is None:
                if not self.failures.allow(source):
                    raise ApiError(TOO_MANY_FAILURES)
                raise ApiError(UNAUTHORIZED)
        request.state.client = client
        return client
