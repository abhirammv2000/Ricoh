"""Rewrite follow-up questions so they make sense on their own.

Each question is answered separately, so after "How do I create a workflow?" the follow-up "Can I
copy an existing one?" finds nothing, because "one" means nothing alone. condense_query turns it into
"Can I copy an existing workflow in RICOH ProcessDirector?" before retrieval. Only the question is
rewritten. The synthesizer still answers only from the evidence, so a bad rewrite gives a miss or a
refusal, not a made-up answer. With no history there is no extra LLM call, so the single-turn path
the benchmark measures is unchanged. (The multi-turn eval, multiturn_eval.py, measures this.)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.instrumentation import invoke as instrumented_invoke
from src.llm_factory import get_llm

# only the latest turns are used as context, enough to resolve a reference and short enough to keep the prompt cheap
MAX_HISTORY_TURNS: int = 4

# earlier answers are cut short in the prompt, the rewrite only needs to know what was discussed
_ANSWER_PREVIEW_CHARS: int = 400


@dataclass(frozen=True)
class Turn:
    """One finished exchange, the question and the answer."""

    question: str
    answer: str


CONDENSE_PROMPT = """\
You rewrite a follow-up question into a standalone one for a Ricoh technical \
support search system.

Given the conversation so far and a follow-up question, rewrite the follow-up \
so it stands on its own, resolving references ("it", "that step", "the second \
one") to what they point at in the conversation. Keep the user's wording and \
language. If the follow-up is already standalone, return it unchanged.

Output only the rewritten question: no preamble, no quotes, no explanation.

Conversation so far:
{history}

Follow-up question:
{question}

Standalone question:
"""


def _format_history(turns: Sequence[Turn]) -> str:
    """A numbered transcript of the turns, with each answer cut short."""
    lines: list[str] = []
    for i, turn in enumerate(turns, 1):
        answer = turn.answer.strip()
        if len(answer) > _ANSWER_PREVIEW_CHARS:
            answer = answer[:_ANSWER_PREVIEW_CHARS].rstrip() + " ..."
        lines.append(f"{i}. User: {turn.question.strip()}")
        lines.append(f"   Assistant: {answer}")
    return "\n".join(lines)


def condense_query(
    history: Sequence[Turn],
    question: str,
    *,
    llm: Any | None = None,
) -> str:
    """Rewrite the question to stand alone, using the history.

    With no history it returns the question as is and makes no LLM call. Otherwise it makes one cheap
    call (the condense stage) and falls back to the original if the model gives nothing usable. llm can
    be passed in for tests.
    """
    turns = list(history)[-MAX_HISTORY_TURNS:]
    if not turns:
        return question

    llm = llm or get_llm()
    prompt = CONDENSE_PROMPT.format(
        history=_format_history(turns), question=question.strip()
    )
    rewritten = instrumented_invoke(llm, prompt, stage="condense").strip()
    # sometimes the model wraps the line in quotes anyway
    rewritten = rewritten.strip('"').strip("'").strip()
    return rewritten or question
