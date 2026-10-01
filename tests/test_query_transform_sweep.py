"""Offline checks for eval/query_transform_sweep.py.

The sweep calls Gemini and a real index. These cover the part that is easy to get
wrong: what happens when the model's safety filter blocks a call.
"""

from __future__ import annotations

from types import SimpleNamespace

from eval.query_transform_sweep import _configs, _llm_text

BASELINE = "baseline (raw question)"


class FakeRetriever:
    """Returns one chunk per query and remembers what it was asked."""

    def __init__(self) -> None:
        self.vector_queries: list[str] = []
        self.bm25_queries: list[str] = []

    def _vector_search(self, query, top_k):
        self.vector_queries.append(query)
        return [{"id": f"vec:{query}", "source_document": "a.pdf"}]

    def _bm25_search(self, query, top_k):
        self.bm25_queries.append(query)
        return [{"id": f"bm25:{query}", "source_document": "b.pdf"}]


class RaisesLikeABlockedCall:
    def invoke(self, prompt):
        # langchain_openai raises this when the reply has no message
        raise AttributeError("'NoneType' object has no attribute 'get'")


def test_a_blocked_call_gives_none_instead_of_crashing():
    assert _llm_text(RaisesLikeABlockedCall(), "anything") is None


def test_a_normal_call_gives_the_text():
    llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="a passage"))
    assert _llm_text(llm, "anything") == "a passage"


def test_blocked_hyde_searches_with_the_question_instead():
    r = FakeRetriever()
    _configs(r, "how do I copy a workflow", {"hyde": None, "rewrites": []})
    assert "" not in r.vector_queries
    assert set(r.vector_queries) == {"how do I copy a workflow"}


def test_no_rewrites_makes_multi_query_the_same_as_the_baseline():
    r = FakeRetriever()
    out = _configs(r, "q", {"hyde": None, "rewrites": []})
    multi = next(v for k, v in out.items() if k.startswith("multi-query"))
    assert [d["id"] for d in multi] == [d["id"] for d in out[BASELINE]]


def test_every_rewrite_is_searched_by_both_retrievers():
    r = FakeRetriever()
    _configs(r, "q", {"hyde": "a passage", "rewrites": ["r1", "r2", "r3"]})
    for rewrite in ("r1", "r2", "r3"):
        assert rewrite in r.vector_queries
        assert rewrite in r.bm25_queries
