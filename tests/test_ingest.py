"""Unit tests for the chunking pipeline (src/ingest.py).

These guard the invariants the citation feature depends on:
chunks never exceed the size budget, overlap is preserved, page
provenance is never mixed, and IDs are deterministic.
"""

from __future__ import annotations

from src.ingest import _generate_chunk_id, chunk_pages


def test_short_page_emits_single_chunk(make_page):
    page = make_page(10)  # fewer words than CHUNK_SIZE
    chunks = chunk_pages([page], chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["text"] == page["text"]
    assert chunks[0]["page_number"] == 1


def test_chunks_never_exceed_size(make_page):
    page = make_page(500)
    chunks = chunk_pages([page], chunk_size=50, chunk_overlap=10)
    assert chunks  # produced something
    for c in chunks:
        assert len(c["text"].split()) <= 50


def test_overlap_is_preserved(make_page):
    page = make_page(120)
    chunks = chunk_pages([page], chunk_size=50, chunk_overlap=10)
    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    # The last 10 words of chunk 1 should equal the first 10 of chunk 2.
    assert first[-10:] == second[:10]


def test_page_provenance_never_mixed(make_page):
    pages = [
        make_page(200, page_number=1, source="a.pdf"),
        make_page(200, page_number=2, source="a.pdf"),
    ]
    chunks = chunk_pages(pages, chunk_size=50, chunk_overlap=10)
    by_page = {1: 0, 2: 0}
    for c in chunks:
        # Every chunk maps to exactly one page; words are page-local.
        assert c["page_number"] in (1, 2)
        by_page[c["page_number"]] += 1
    assert by_page[1] > 0 and by_page[2] > 0


def test_chunk_ids_are_unique_and_deterministic(make_page):
    pages = [make_page(300, page_number=p, source="a.pdf") for p in (1, 2, 3)]
    chunks_a = chunk_pages(pages, chunk_size=50, chunk_overlap=10)
    chunks_b = chunk_pages(pages, chunk_size=50, chunk_overlap=10)
    ids = [c["id"] for c in chunks_a]
    assert len(ids) == len(set(ids))  # unique
    assert ids == [c["id"] for c in chunks_b]  # deterministic across runs


def test_generate_chunk_id_stable():
    assert _generate_chunk_id("a.pdf", 1, 0) == _generate_chunk_id("a.pdf", 1, 0)
    assert _generate_chunk_id("a.pdf", 1, 0) != _generate_chunk_id("a.pdf", 1, 1)
