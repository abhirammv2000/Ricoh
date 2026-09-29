"""Unit tests for src/azure_retriever.py that need no network or Azure account.

The query builder is checked against a fake search client, so we assert what
would be sent to Azure for each mode without sending anything.
"""

from __future__ import annotations

import pytest

from src.azure_retriever import MODES, AzureRetriever, _clean_query


def test_clean_query_strips_operator_characters():
    cleaned = _clean_query('what is "SC542" (error)?')
    assert not any(ch in cleaned for ch in '"()')
    assert "SC542" in cleaned and "error" in cleaned


def test_clean_query_drops_a_leading_minus_but_keeps_hyphenated_words():
    cleaned = _clean_query("-remove e-mail setup")
    assert cleaned.startswith("remove")
    assert "e-mail" in cleaned


def test_missing_credentials_raise():
    with pytest.raises(RuntimeError):
        AzureRetriever(endpoint="", api_key="", index_name="x")


class _FakeClient:
    def __init__(self, hits):
        self.hits = hits
        self.kwargs = None

    def search(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.hits)


def _retriever(hits):
    pytest.importorskip("azure.search.documents")
    r = object.__new__(AzureRetriever)
    r._client = _FakeClient(hits)
    r.embed = lambda texts: [[0.0] * 384 for _ in texts]
    return r


def _hit(**extra):
    hit = {
        "id": "a1",
        "text": "some text",
        "source_document": "doc.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "@search.score": 1.5,
    }
    hit.update(extra)
    return hit


def test_keyword_mode_sends_text_and_no_vector():
    r = _retriever([_hit()])
    r.retrieve("how do I add a step", mode="keyword")
    assert r._client.kwargs["search_text"] == "how do I add a step"
    assert "vector_queries" not in r._client.kwargs


def test_vector_mode_sends_vector_and_no_text():
    r = _retriever([_hit()])
    r.retrieve("how do I add a step", mode="vector", top_k=7)
    assert "search_text" not in r._client.kwargs
    assert len(r._client.kwargs["vector_queries"]) == 1


def test_hybrid_mode_sends_both_and_no_semantic_options():
    r = _retriever([_hit()])
    r.retrieve("q", mode="hybrid")
    assert "search_text" in r._client.kwargs
    assert "vector_queries" in r._client.kwargs
    assert "query_type" not in r._client.kwargs


def test_hybrid_semantic_turns_on_the_semantic_ranker():
    r = _retriever([_hit(**{"@search.reranker_score": 2.9})])
    out = r.retrieve("q", mode="hybrid_semantic", final_k=3)
    assert r._client.kwargs["query_type"] == "semantic"
    assert r._client.kwargs["semantic_configuration_name"]
    assert r._client.kwargs["top"] == 3
    assert out[0]["rerank_score"] == 2.9


def test_results_use_the_same_shape_as_the_local_retriever():
    r = _retriever([_hit()])
    out = r.retrieve("q", mode="hybrid")
    assert set(out[0]) >= {"id", "text", "source_document", "page_number", "chunk_index", "score"}
    assert "rerank_score" not in out[0]


def test_unknown_mode_is_rejected():
    r = _retriever([])
    with pytest.raises(ValueError):
        r.retrieve("q", mode="nonsense")


def test_all_documented_modes_are_accepted():
    r = _retriever([_hit()])
    for mode in MODES:
        r.retrieve("q", mode=mode)
