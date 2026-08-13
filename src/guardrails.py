"""Input screening at the edge, before a query reaches the agent.

Why this exists, and what it is not. The primary defense against prompt
injection in this system is architectural, not a blocklist: the synthesizer is
instructed to answer only from retrieved evidence and to emit a fixed refusal
marker otherwise, so a query that tries to override the instructions retrieves
no supporting evidence from the Ricoh corpus and is refused on those grounds.

This module is the cheap outer layer of a defense in depth. It rejects a small
set of unambiguous override and jailbreak attempts at the API edge, before they
cost an LLM call. It is deliberately high precision: every pattern here is one a
genuine printer support question would essentially never contain, because a
guardrail that fires on normal questions is worse than no guardrail. It does not
claim to catch every injection. It catches the obvious ones for free and leaves
the rest to the grounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# High-precision prompt-injection and jailbreak signals. Each pattern is
# anchored on phrasing that overrides or exfiltrates instructions, never on a
# lone word like "system" that appears in legitimate questions such as "how do I
# open the system settings menu". A false positive here silently breaks a real
# support question, so the bar for adding a pattern is that a normal user would
# not phrase a question this way.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "ignore the previous instructions", "disregard all prior prompts"
    re.compile(
        r"(?:ignore|disregard)\s+(?:all\s+|any\s+|the\s+|your\s+)*"
        r"(?:previous|prior|above|preceding|earlier|foregoing)\s+"
        r"(?:instruction|prompt|rule|direction|message)s?",
        re.I,
    ),
    # "forget everything you were told", "forget your instructions", "forget the
    # above". Anchored on an imperative aimed at the assistant's own context, so
    # it does not fire on a user saying "I forget what the model number is".
    re.compile(
        r"forget\s+(?:all\s+|everything\s+)?"
        r"(?:(?:that\s+)?you(?:'ve|'re| have| were| are)?\s+(?:been\s+)?(?:told|learned|know|instructed)"
        r"|the\s+above"
        r"|your\s+(?:instructions|rules|prompt|training|guidelines)"
        r"|(?:all\s+)?previous\s+instructions)",
        re.I,
    ),
    # "reveal your system prompt", "print the system prompt"
    re.compile(
        r"(?:reveal|show|print|repeat|expose|leak|display|output)\s+"
        r"(?:me\s+)?(?:your\s+|the\s+)*system\s+prompt",
        re.I,
    ),
    # persona override, e.g. "you are now DAN", "you are now a pirate"
    re.compile(r"you\s+are\s+now\s+", re.I),
    # the classic "do anything now" jailbreak
    re.compile(r"do\s+anything\s+now", re.I),
    # "override your instructions", "override all safety rules"
    re.compile(
        r"override\s+(?:your\s+|all\s+|the\s+)*"
        r"(?:instruction|rule|guideline|programming|safety|restriction)s?",
        re.I,
    ),
)


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of screening one input.

    ``reason`` is intentionally generic. It is meant for the caller and for
    logs, and it does not echo the offending text or name the pattern that
    matched, so the screen does not become a description of how to get past it.
    """

    allowed: bool
    reason: str = ""


def screen_input(query: str) -> GuardrailResult:
    """Screen a raw user query before it reaches the agent.

    Returns an allowed result for anything that looks like a genuine question,
    and a rejected result for empty input or a known override or jailbreak
    pattern. This is a fast, side-effect-free check: no LLM call, no network.
    """
    text = query.strip()
    if not text:
        return GuardrailResult(allowed=False, reason="empty query")
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                allowed=False, reason="input rejected by prompt-injection screen"
            )
    return GuardrailResult(allowed=True)
