"""Screen a question before it reaches the agent.

The real defence against prompt injection is the grounding: the synthesizer only answers from
retrieved evidence, so a question that tries to override it finds no evidence and is refused. This
is a cheap outer layer. It rejects a few obvious override and jailbreak phrasings before they cost an
LLM call, and each pattern is one a real support question would almost never contain, since a
guardrail that blocks normal questions is worse than none. It doesn't catch everything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# patterns for jailbreak and override phrasing. They match the phrasing, never a single word like
# "system" (as in "open the system settings menu"), since a false positive blocks a real question
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "ignore the previous instructions", "disregard all prior prompts"
    re.compile(
        r"(?:ignore|disregard)\s+(?:all\s+|any\s+|the\s+|your\s+)*"
        r"(?:previous|prior|above|preceding|earlier|foregoing)\s+"
        r"(?:instruction|prompt|rule|direction|message)s?",
        re.I,
    ),
    # "forget everything you were told", "forget your instructions". Aimed at the assistant's
    # context, so "I forget what the model number is" doesn't match
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
    """The result of screening one input. reason is generic on purpose, it doesn't name the pattern that matched."""

    allowed: bool
    reason: str = ""


def screen_input(query: str) -> GuardrailResult:
    """Allow a normal question, reject empty input or a known jailbreak pattern. No LLM or network call."""
    text = query.strip()
    if not text:
        return GuardrailResult(allowed=False, reason="empty query")
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                allowed=False, reason="input rejected by prompt-injection screen"
            )
    return GuardrailResult(allowed=True)
