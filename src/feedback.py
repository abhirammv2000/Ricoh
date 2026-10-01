"""Thumbs up or down on answers, stored next to the traces.

A vote is keyed by the request's trace id, so it leads back to the chunks that produced the answer
(src/trace_view.py). It's a jsonl file under traces/, which is gitignored because the question, answer
and comment are user data. Every event is kept and the reader uses the latest per trace id, since people
change their minds.

    python -m src.feedback            # counts and the up-rate with a 95% range

eval/feedback_candidates.py turns feedback into eval candidates. Nothing here touches the benchmark.
"""

from __future__ import annotations

import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

FEEDBACK_PATH: Path = PROJECT_ROOT / "traces" / "feedback.jsonl"

_MAX_ANSWER_CHARS = 4000
_MAX_COMMENT_CHARS = 1000


def vote_from_widget(value: int | None) -> int | None:
    """Streamlit's thumbs value (0 down, 1 up, None unset) as -1 or +1."""
    if value is None:
        return None
    return 1 if value == 1 else -1


def sources_from_state(state: dict[str, Any]) -> list[str]:
    """The distinct documents in the evidence behind an answer."""
    seen: list[str] = []
    for chunk in state.get("retrieved_evidence") or []:
        doc = chunk.get("source_document")
        if doc and doc not in seen:
            seen.append(doc)
    return seen


def record_feedback(
    trace_id: str,
    vote: int,
    *,
    query: str,
    answer: str,
    sources: list[str] | None = None,
    comment: str = "",
    path: Path = FEEDBACK_PATH,
) -> None:
    """Add one vote. Voting again on the same trace replaces the old one."""
    if vote not in (1, -1):
        raise ValueError("vote must be 1 (up) or -1 (down)")
    if not trace_id:
        raise ValueError("trace_id is required")
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "vote": vote,
        "query": query,
        "answer": answer[:_MAX_ANSWER_CHARS],
        "sources": sources or [],
        "comment": comment.strip()[:_MAX_COMMENT_CHARS],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # one write per event so two sessions don't interleave
    with io.open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def record_for_trace(
    messages: list[dict[str, Any]],
    trace_id: str,
    vote: int,
    comment: str = "",
    path: Path = FEEDBACK_PATH,
) -> bool:
    """Find the assistant message for a trace in the chat and record the vote. Returns False if the trace isn't there."""
    for i, msg in enumerate(messages):
        state = msg.get("agent_state") or {}
        if i > 0 and (state.get("trace") or {}).get("trace_id") == trace_id:
            record_feedback(
                trace_id,
                vote,
                query=messages[i - 1]["content"],
                answer=msg["content"],
                sources=sources_from_state(state),
                comment=comment,
                path=path,
            )
            return True
    return False


def load_feedback(path: Path = FEEDBACK_PATH) -> list[dict[str, Any]]:
    """The latest event for each trace id, oldest first. Half-written lines are skipped."""
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("vote") in (1, -1) and event.get("trace_id"):
                latest.pop(event["trace_id"], None)  # re-insert so order follows the latest vote
                latest[event["trace_id"]] = event
    return list(latest.values())


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion, (0, 0) with no data."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    up = sum(1 for e in events if e["vote"] == 1)
    total = len(events)
    low, high = wilson_interval(up, total)
    return {
        "total": total,
        "up": up,
        "down": total - up,
        "up_rate": (up / total) if total else None,
        "ci_low": low,
        "ci_high": high,
    }


if __name__ == "__main__":
    s = summarize(load_feedback())
    if not s["total"]:
        print("No feedback yet.")
    else:
        print(
            f"{s['total']} rated answers: {s['up']} up, {s['down']} down "
            f"({s['up_rate']:.0%}, 95% range {s['ci_low']:.0%} to {s['ci_high']:.0%})"
        )
