"""Shared pytest fixtures for the RicohLibrary test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def make_page():
    """Factory for synthetic page dicts with N deterministic words."""

    def _make(n_words: int, page_number: int = 1, source: str = "doc.pdf"):
        text = " ".join(f"w{i}" for i in range(n_words))
        return {"text": text, "page_number": page_number, "source_document": source}

    return _make
