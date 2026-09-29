"""User feedback storage and the export to eval candidates. Offline, no LLM."""

from __future__ import annotations

import json
import subprocess

import pytest

from eval.feedback_candidates import build_candidates, known_questions
from src import feedback as fb


@pytest.fixture
def path(tmp_path):
    return tmp_path / "feedback.jsonl"


def _vote(path, trace_id, vote, query="how do I add a step", **kw):
    fb.record_feedback(trace_id, vote, query=query, answer="an answer", path=path, **kw)


def test_widget_values_map_to_up_and_down():
    assert fb.vote_from_widget(1) == 1
    assert fb.vote_from_widget(0) == -1
    assert fb.vote_from_widget(None) is None


def test_a_vote_is_stored_with_its_context(path):
    _vote(path, "abc123", -1, sources=["a.pdf"], comment="  wrong document ")

    (event,) = fb.load_feedback(path)
    assert event["trace_id"] == "abc123"
    assert event["vote"] == -1
    assert event["sources"] == ["a.pdf"]
    assert event["comment"] == "wrong document"


def test_voting_again_replaces_the_earlier_vote(path):
    _vote(path, "t1", 1)
    _vote(path, "t2", 1)
    _vote(path, "t1", -1)

    events = fb.load_feedback(path)
    assert [(e["trace_id"], e["vote"]) for e in events] == [("t2", 1), ("t1", -1)]


def test_bad_input_is_refused(path):
    with pytest.raises(ValueError):
        _vote(path, "t1", 0)
    with pytest.raises(ValueError):
        _vote(path, "", 1)
    assert not path.exists()


def test_long_text_is_truncated(path):
    fb.record_feedback("t1", 1, query="q", answer="x" * 9000, comment="y" * 5000, path=path)

    (event,) = fb.load_feedback(path)
    assert len(event["answer"]) == 4000
    assert len(event["comment"]) == 1000


def test_a_missing_file_and_torn_lines_are_tolerated(path):
    assert fb.load_feedback(path) == []
    _vote(path, "t1", 1)
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"trace_id": "t2", "vote": 1, "que')  # a write cut off half way

    assert [e["trace_id"] for e in fb.load_feedback(path)] == ["t1"]


def test_summary_counts_and_interval(path):
    for i in range(10):
        _vote(path, f"t{i}", 1 if i < 8 else -1, query=f"q{i}")

    s = fb.summarize(fb.load_feedback(path))
    assert (s["total"], s["up"], s["down"]) == (10, 8, 2)
    assert s["up_rate"] == pytest.approx(0.8)
    assert s["ci_low"] == pytest.approx(0.4902, abs=1e-3)
    assert s["ci_high"] == pytest.approx(0.9433, abs=1e-3)


def test_summary_with_no_feedback():
    s = fb.summarize([])
    assert s["total"] == 0 and s["up_rate"] is None


def test_sources_are_the_distinct_documents_in_the_evidence():
    state = {"retrieved_evidence": [
        {"source_document": "a.pdf"}, {"source_document": "b.pdf"}, {"source_document": "a.pdf"}, {},
    ]}
    assert fb.sources_from_state(state) == ["a.pdf", "b.pdf"]
    assert fb.sources_from_state({}) == []


def _history(trace_id="trace-1"):
    return [
        {"role": "user", "content": "How do I add a step?"},
        {
            "role": "assistant",
            "content": "Use the workflow editor.",
            "agent_state": {
                "trace": {"trace_id": trace_id},
                "retrieved_evidence": [{"source_document": "a.pdf"}],
            },
        },
    ]


def test_a_vote_is_attached_to_the_right_question_and_answer(path):
    assert fb.record_for_trace(_history(), "trace-1", -1, "wrong doc", path=path) is True

    (event,) = fb.load_feedback(path)
    assert event["query"] == "How do I add a step?"
    assert event["answer"] == "Use the workflow editor."
    assert event["sources"] == ["a.pdf"]
    assert event["comment"] == "wrong doc"


def test_a_vote_for_a_trace_not_in_the_history_writes_nothing(path):
    assert fb.record_for_trace(_history(), "some-other-trace", 1, path=path) is False
    assert not path.exists()


def test_the_first_message_is_never_treated_as_an_answer(path):
    # An assistant message at index 0 has no question before it.
    history = [_history()[1]]
    assert fb.record_for_trace(history, "trace-1", 1, path=path) is False


# -- eval candidates -------------------------------------------------------


def _events(path, votes):
    for i, (vote, query) in enumerate(votes):
        _vote(path, f"trace{i}", vote, query=query)
    return fb.load_feedback(path)


def test_thumbs_down_questions_become_unlabelled_candidates(path):
    events = _events(path, [(-1, "How do I reset the fuser?"), (1, "What OS does it run on?")])

    (c,) = build_candidates(events, known=set())

    assert c["question"] == "How do I reset the fuser?"
    assert c["needs_label"] is True
    assert c["expected_sources"] == [] and c["expected_behavior"] is None
    assert c["provenance"] == "feedback"
    assert c["feedback"]["vote"] == -1


def test_questions_already_in_the_benchmark_are_skipped(path):
    events = _events(path, [(-1, "How do I  RESET the fuser?"), (-1, "A new question")])

    out = build_candidates(events, known={"how do i reset the fuser?"})

    assert [c["question"] for c in out] == ["A new question"]


def test_the_same_question_is_only_a_candidate_once(path):
    events = _events(path, [(-1, "same question"), (-1, "Same   question")])

    assert len(build_candidates(events, known=set())) == 1


def test_thumbs_up_are_sampled_deterministically_and_only_on_request(path):
    events = _events(path, [(-1, "down one")] + [(1, f"up {i}") for i in range(10)])

    assert len(build_candidates(events, known=set())) == 1
    a = build_candidates(events, known=set(), include_up=3, seed=1)
    b = build_candidates(events, known=set(), include_up=3, seed=1)
    assert a == b
    assert len(a) == 4
    assert a[0]["feedback"]["vote"] == -1  # failures come first


def test_include_up_larger_than_available_is_fine(path):
    events = _events(path, [(1, "only up")])

    assert len(build_candidates(events, known=set(), include_up=50)) == 1


def test_known_questions_reads_the_real_benchmark_files():
    known = known_questions()
    assert len(known) >= 100
    assert all(q == " ".join(q.lower().split()) for q in list(known)[:20])


def test_feedback_files_stay_out_of_git():
    for name in ("traces/feedback.jsonl", "eval/feedback_candidates.json"):
        result = subprocess.run(["git", "check-ignore", "-q", name], capture_output=True)
        assert result.returncode == 0, f"{name} is not gitignored, and it holds user text"


def test_events_round_trip_as_json(path):
    _vote(path, "t1", 1, query="unicode question éè")
    line = path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["query"] == "unicode question éè"
