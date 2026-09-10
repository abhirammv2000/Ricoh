"""History-aware query condensation for multi-turn conversations.

The default pipeline answers each question in isolation. That breaks the moment
someone asks a follow-up: "How do I create a workflow?" then "Can I copy an
existing one?" - the second question retrieves nothing useful because "one" has
no referent on its own.

``condense_query`` rewrites a follow-up into a standalone question before it
reaches retrieval, using the recent conversation as context. "Can I copy an
existing one?" becomes "Can I copy an existing workflow in RICOH
ProcessDirector?", which retrieves and synthesizes like any other standalone
question.

What this does not change. Condensation only rewrites the question fed to
retrieval and synthesis. The synthesizer still answers from retrieved evidence
only and still emits the refusal marker when the evidence does not support an
answer, so a poor rewrite degrades to a retrieval miss, usually a refusal,
rather than to a hallucination. The single-turn path is untouched: with no
history there is no extra LLM call and behaviour is identical to before, which
is what keeps the section 7 benchmark valid as a measure of that path.

Not yet measured. There is no multi-turn question set and no judged run, so this
ships as a mechanism with mocked tests. Whether condensation helps or hurts
end-to-end answer quality is a question the eval side has not paid for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.instrumentation import invoke as instrumented_invoke
from src.llm_factory import get_llm

# Only the most recent turns are used as context. Enough to resolve a reference
# a few exchanges back, bounded so a long chat does not grow the condense
# prompt (and its cost) without limit.
MAX_HISTORY_TURNS: int = 4

# Prior answers are truncated in the prompt. The rewrite needs to know what was
# discussed, not re-read every cited passage, and answers here run to hundreds
# of words with tables and citations.
_ANSWER_PREVIEW_CHARS: int = 400


@dataclass(frozen=True)
class Turn:
    """One completed exchange: what the user asked and what the system replied."""

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
    """Render turns as a numbered transcript, with each answer truncated."""
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
    """Rewrite ``question`` as standalone given the conversation ``history``.

    Returns ``question`` unchanged when there is no history, without an LLM
    call, so the single-turn path pays nothing for this. Otherwise runs one
    cheap LLM call, recorded under the ``condense`` stage, and falls back to the
    original question if the model returns nothing usable.

    ``llm`` is injectable for tests; production passes nothing and gets the
    configured model.
    """
    turns = list(history)[-MAX_HISTORY_TURNS:]
    if not turns:
        return question

    llm = llm or get_llm()
    prompt = CONDENSE_PROMPT.format(
        history=_format_history(turns), question=question.strip()
    )
    rewritten = instrumented_invoke(llm, prompt, stage="condense").strip()
    # Models occasionally wrap the line in quotes despite the instruction.
    rewritten = rewritten.strip('"').strip("'").strip()
    return rewritten or question
