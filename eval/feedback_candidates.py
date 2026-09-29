"""Turn user feedback into eval candidates for a person to label.

The benchmark has one weakness the README names: production traffic never
flows back into it. This closes the first half of that loop. It takes the
questions people gave a thumbs down (and, optionally, a seeded sample of thumbs
ups so the set isn't all failures) and writes them out in the benchmark's shape.

It does not add anything to the benchmark. Every candidate comes out with
`needs_label: true` and empty `expected_sources`, because the label has to come
from reading the source documents. An earlier mislabeled entry in this
benchmark made a correct refusal look like a retrieval miss, and a candidate
that inherits the system's own answer as its label would repeat that mistake.

    python -m src.feedback                        # how much feedback there is
    python -m eval.feedback_candidates            # thumbs down only
    python -m eval.feedback_candidates --include-up 10

Writes eval/feedback_candidates.json, which is gitignored: it holds user text.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.feedback import load_feedback

OUTPUT_PATH: Path = PROJECT_ROOT / "eval" / "feedback_candidates.json"
_BENCHMARK_FILES = ("generated_questions.json", "ground_truth.json", "multihop_questions.json")


def _normalise(question: str) -> str:
    return " ".join(question.lower().split())


def known_questions(eval_dir: Path = PROJECT_ROOT / "eval") -> set[str]:
    """Every question already in a benchmark file, so a candidate isn't a duplicate."""
    known: set[str] = set()
    for name in _BENCHMARK_FILES:
        path = eval_dir / name
        if not path.exists():
            continue
        data = json.loads(io.open(path, encoding="utf-8").read())
        questions = data.get("questions", []) if isinstance(data, dict) else data
        known.update(_normalise(q["question"]) for q in questions if q.get("question"))
    return known


def build_candidates(
    events: list[dict[str, Any]],
    known: set[str],
    include_up: int = 0,
    seed: int = 20260929,
) -> list[dict[str, Any]]:
    """Thumbs-down questions first, then a seeded sample of thumbs-up ones."""
    downs = [e for e in events if e["vote"] == -1]
    ups = [e for e in events if e["vote"] == 1]
    sampled_ups = random.Random(seed).sample(ups, min(include_up, len(ups))) if include_up else []

    candidates: list[dict[str, Any]] = []
    seen = set(known)
    for event in downs + sampled_ups:
        key = _normalise(event["query"])
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append({
            "id": f"fb-{event['trace_id']}",
            "question": event["query"],
            "provenance": "feedback",
            "split": "candidate",
            "needs_label": True,
            "expected_behavior": None,
            "expected_sources": [],
            "key_facts": [],
            "feedback": {
                "vote": event["vote"],
                "comment": event.get("comment", ""),
                "system_answer": event.get("answer", ""),
                "system_sources": event.get("sources", []),
            },
        })
    return candidates


def main() -> int:
    ap = argparse.ArgumentParser(description="Export feedback as eval candidates")
    ap.add_argument("--include-up", type=int, default=0, help="also sample this many thumbs-up questions")
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    events = load_feedback()
    if not events:
        print("No feedback yet, nothing to export.")
        return 0

    candidates = build_candidates(events, known_questions(), args.include_up, args.seed)
    payload = {
        "_description": (
            "Questions from user feedback, waiting for a person to label them. "
            "Do not add to the benchmark until expected_behavior and expected_sources "
            "have been filled in from the source documents."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "feedback_events": len(events),
            "candidates": len(candidates),
            "thumbs_down": sum(1 for c in candidates if c["feedback"]["vote"] == -1),
            "thumbs_up": sum(1 for c in candidates if c["feedback"]["vote"] == 1),
        },
        "questions": candidates,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(candidates)} candidates to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
