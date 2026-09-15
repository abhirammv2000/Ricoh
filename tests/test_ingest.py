"""Unit tests for the chunking pipeline (src/ingest.py).

These guard the invariants the citation feature depends on:
chunks never exceed the size budget, overlap is preserved, page
provenance is never mixed, and IDs are deterministic.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from src.ingest import _generate_chunk_id, chunk_pages, extract_pages


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


# extract_pages() + vision descriptions (src/vision_ingest.py)
#
# These use a real on-disk PDF (built with fitz, the same library
# extract_pages parses with) rather than mocking fitz itself, so the test
# exercises the actual extraction path. Only get_cached_description is
# faked, since it is a local cache lookup, no network/API call involved.

def _make_pdf(path: Path, page_texts: list[str]) -> Path:
    """Build a PDF with one page per string in page_texts (empty string =
    a page with no extractable text, standing in for an image-only page)."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page(width=595, height=842)
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 545, 792), text, fontsize=11, fontname="helv")
    doc.save(str(path))
    doc.close()
    return path


def test_vision_description_appended_to_page_text(tmp_path, monkeypatch):
    pdf_path = _make_pdf(tmp_path / "doc.pdf", ["Real extracted prose."])
    monkeypatch.setattr(
        "src.ingest.get_cached_description",
        lambda source, page_number: "Diagram shows step A leads to step B.",
    )

    pages = extract_pages(pdf_path)

    assert len(pages) == 1
    assert "Real extracted prose." in pages[0]["text"]
    assert "Diagram shows step A leads to step B." in pages[0]["text"]
    assert "[Embedded image/diagram content on this page]" in pages[0]["text"]


def test_page_with_no_text_but_a_vision_description_is_kept(tmp_path, monkeypatch):
    # An image-only page: previously silently dropped (no text at all). If a
    # vision description exists for it, it now carries real information and
    # should not be dropped just because the text layer was empty.
    pdf_path = _make_pdf(tmp_path / "doc.pdf", [""])
    monkeypatch.setattr(
        "src.ingest.get_cached_description",
        lambda source, page_number: "Screenshot of the configuration dialog.",
    )

    pages = extract_pages(pdf_path)

    assert len(pages) == 1
    assert "Screenshot of the configuration dialog." in pages[0]["text"]


def test_page_with_no_text_and_no_vision_description_is_still_skipped(tmp_path, monkeypatch):
    pdf_path = _make_pdf(tmp_path / "doc.pdf", [""])
    monkeypatch.setattr("src.ingest.get_cached_description", lambda source, page_number: None)

    pages = extract_pages(pdf_path)

    assert pages == []


def test_no_vision_description_leaves_text_unchanged(tmp_path, monkeypatch):
    pdf_path = _make_pdf(tmp_path / "doc.pdf", ["Just prose, no embedded image."])
    monkeypatch.setattr("src.ingest.get_cached_description", lambda source, page_number: None)

    pages = extract_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0]["text"] == "Just prose, no embedded image."
    assert "[Embedded image" not in pages[0]["text"]
