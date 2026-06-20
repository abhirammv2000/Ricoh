"""Unit tests for the objective (non-LLM) metrics in the eval harness."""

from __future__ import annotations

from src.eval_harness import (
    _citation_precision,
    _cited_docs,
    _evidence_docs,
    _is_refusal,
    _retrieval_recall,
)


def test_cited_docs_extracts_unique_filenames():
    answer = (
        "Set the property [aiw00p18.pdf, Page 1]. By default printers are "
        "disabled [aiw00a12.pdf, Page 3] [aiw00p18.pdf, Page 1]."
    )
    assert _cited_docs(answer) == {"aiw00p18.pdf", "aiw00a12.pdf"}


def test_cited_docs_empty_when_no_citations():
    assert _cited_docs("Information unavailable in provided documents.") == set()


def test_retrieval_recall_fraction():
    ev = {"a.pdf", "c.pdf"}
    assert _retrieval_recall(["a.pdf", "b.pdf"], ev) == 0.5


def test_retrieval_recall_none_when_no_expected():
    assert _retrieval_recall([], {"a.pdf"}) is None


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
