"""Unit tests for the objective (non-LLM) metrics in the eval harness."""

from __future__ import annotations

from src.eval_harness import (
    _bootstrap_ci,
    _citation_precision,
    _cited_docs,
    _evidence_docs,
    _evidence_recall,
    _is_refusal,
    _worst,
)


def test_cited_docs_extracts_unique_filenames():
    answer = (
        "Set the property [aiw00p18.pdf, Page 1]. By default printers are "
        "disabled [aiw00a12.pdf, Page 3] [aiw00p18.pdf, Page 1]."
    )
    assert _cited_docs(answer) == {"aiw00p18.pdf", "aiw00a12.pdf"}


def test_cited_docs_empty_when_no_citations():
    assert _cited_docs("Information unavailable in provided documents.") == set()


def test_evidence_recall_fraction():
    ev = {"a.pdf", "c.pdf"}
    assert _evidence_recall(["a.pdf", "b.pdf"], ev) == 0.5


def test_evidence_recall_none_when_no_expected():
    assert _evidence_recall([], {"a.pdf"}) is None


def test_citation_precision_flags_fabricated_citation():
    # "ghost.pdf" was cited but never retrieved -> precision 0.5.
    assert _citation_precision({"a.pdf", "ghost.pdf"}, {"a.pdf"}) == 0.5


def test_citation_precision_none_when_no_citations():
    assert _citation_precision(set(), {"a.pdf"}) is None


def test_evidence_docs_dedup():
    evidence = [
        {"source_document": "a.pdf"},
        {"source_document": "a.pdf"},
        {"source_document": "b.pdf"},
        {"text": "no source"},
    ]
    assert _evidence_docs(evidence) == {"a.pdf", "b.pdf"}


def test_is_refusal_detection():
    assert _is_refusal("Information unavailable in provided documents.")
    assert not _is_refusal("Set the property to Yes [a.pdf, Page 1].")


# ── Refusal detection: the multilingual regression ──────────────────
# The synthesizer answers in the user's language. Before the fix, a
# refusal emitted only in translated form was invisible to _is_refusal
# and was silently scored as an *answer* — inflating behaviour-match on
# precisely the questions the system got right. The synthesizer prompt
# now requires the English canonical sentence verbatim in every refusal.


def test_is_refusal_survives_whitespace_and_line_breaks():
    assert _is_refusal("Information unavailable\nin provided documents.")
    assert _is_refusal("  INFORMATION   UNAVAILABLE in provided documents  ")


def test_is_refusal_detects_marker_alongside_translation():
    answer = (
        "Information unavailable in provided documents.\n\n"
        "Informacion no disponible en los documentos proporcionados."
    )
    assert _is_refusal(answer)


def test_is_refusal_false_for_translation_only():
    # Documents the contract: a translation WITHOUT the English marker is
    # not detectable. This is why the synthesizer prompt mandates it.
    assert not _is_refusal("Informacion no disponible en los documentos.")


# ── Statistics ──────────────────────────────────────────────────────


def test_bootstrap_ci_brackets_the_mean():
    vals = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]
    lo, hi = _bootstrap_ci(vals, iterations=2000)
    mean = sum(vals) / len(vals)
    assert lo <= mean <= hi
    assert hi - lo > 0.05, "n=10 CI should be visibly wide, not a point estimate"


def test_bootstrap_ci_is_deterministic():
    vals = [0.2, 0.9, 0.4, 1.0, 0.7]
    assert _bootstrap_ci(vals, iterations=500) == _bootstrap_ci(vals, iterations=500)


def test_bootstrap_ci_none_when_too_few_values():
    assert _bootstrap_ci([0.5]) is None
    assert _bootstrap_ci([None, None]) is None


def test_bootstrap_ci_ignores_none_entries():
    assert _bootstrap_ci([1.0, None, 1.0, None, 1.0], iterations=200) == [1.0, 1.0]


def test_worst_case_finds_the_minimum_row():
    rows = [
        {"id": 1, "question": "a", "correctness": 1.0},
        {"id": 9, "question": "how do I use locations?", "correctness": 0.7},
        {"id": 3, "question": "c", "correctness": None},
    ]
    assert _worst(rows, "correctness") == {
        "id": 9,
        "question": "how do I use locations?",
        "correctness": 0.7,
    }


def test_worst_case_none_when_nothing_scored():
    assert _worst([{"id": 1, "question": "a", "correctness": None}], "correctness") is None


# ── Instrumentation: LLM vs retrieval span accounting ───────────────
# Regression guard: when retrieval became traced, retrieval spans landed in
# the same list as LLM spans and silently doubled llm_calls. The metric still
# looked plausible, which is what makes it dangerous.

from src.instrumentation import RunRecord, Span


def _rec() -> RunRecord:
    r = RunRecord()
    r.spans = [
        Span(stage="retrieval", model="-", span_type="retrieval", latency_seconds=0.6),
        Span(stage="synthesizer", model="m", span_type="llm", latency_seconds=20.4,
             input_tokens=2971, output_tokens=1023, cost_usd=0.02426),
    ]
    return r


def test_llm_calls_excludes_retrieval_spans():
    assert _rec().llm_calls == 1


def test_token_totals_exclude_retrieval_spans():
    r = _rec()
    assert r.total_input_tokens == 2971
    assert r.total_output_tokens == 1023


def test_llm_seconds_and_traced_seconds_differ():
    r = _rec()
    assert r.total_llm_seconds == 20.4
    assert r.total_traced_seconds == 21.0, "retrieval latency must be visible somewhere"


def test_by_stage_still_covers_every_span():
    # The rollup is the latency breakdown; dropping retrieval from it would
    # attribute retrieval time to nothing while percentages still sum to 100.
    assert set(_rec().by_stage()) == {"retrieval", "synthesizer"}
