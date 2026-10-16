"""What the service writes down about a user's question.

People paste serial numbers, names and error output into a support box, so a question is treated as personal
data: it does not go into logs, and what goes into traces is a setting.

  describe_query(q)   For logs. Only the length and a short hash, which is enough to match two log lines about
                      the same question without writing the question.
  trace_query(q)      For traces, by TRACE_QUERY_MODE:
                        full   the text as asked. For a laptop, where you want to read your own traces.
                        hash   the length and the hash, with no text. For a deployed service. The same question
                               always gets the same hash, so repeated questions are still countable.
                        none   nothing.

The hash is not a secret. Short questions can be guessed and checked against it, so "hash" keeps the text out
of the file and does not make it unrecoverable. If that matters, use "none".
"""

from __future__ import annotations

import hashlib
import os

MODES = ("full", "hash", "none")


def query_id(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:10]


def describe_query(query: str) -> str:
    """A description of a question that is safe to put in a log line."""
    return f"len={len(query)} id={query_id(query)}"


def trace_mode(environ: dict[str, str] | None = None) -> str:
    mode = ((environ if environ is not None else os.environ).get("TRACE_QUERY_MODE") or "full").strip().lower()
    if mode not in MODES:
        raise ValueError(f"TRACE_QUERY_MODE must be one of {', '.join(MODES)}, got {mode!r}")
    return mode


def trace_query(query: str, mode: str | None = None) -> str:
    """What to record in a trace for this question."""
    mode = mode or trace_mode()
    if mode == "full":
        return query
    if mode == "hash":
        return f"[{describe_query(query)}]"
    return ""
