"""The answer contract: every answer cites a source, or declines with the marker.

This is the output-side defense against injected instructions. It does not depend on how an attack was worded,
because what it looks at is what the model produced. It is checked on 50 real answers so that it cannot start
withholding good ones without a test failing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx
import pytest

import api.main as apimod
from api.errors import INVALID_ANSWER, AnswerContractError, classify_error
from api.main import create_app
from api.readiness import Readiness
from api.settings import Settings
from src.agent import follows_answer_contract, is_refusal

ROOT = Path(__file__).resolve().parent.parent

PASSAGES = [{"source_document": "network_guide.pdf", "page_number": 3, "text": "Open Settings.", "rrf_score": 0.03}]


class FakeRetriever:
    def retrieve(self, query, top_k, final_k):
        return list(PASSAGES)


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(
        apimod, "compute_readiness", lambda require_manifest=True: Readiness(index_version="t", chunks=1)
    )
    monkeypatch.setattr("src.agent.get_retriever", lambda: FakeRetriever())


# ---- the rule itself ----


@pytest.mark.parametrize(
    "answer",
    [
        "Open Settings and choose Network [network_guide.pdf, Page 3].",
        "Steps:\n1. Open the menu [admin_guide.pdf, Page 12].\n2. Choose Apply [admin_guide.pdf, Page 13].",
        "Information unavailable in provided documents.",
        "  INFORMATION   unavailable in provided documents.  ",
        "Information unavailable in provided documents. Informations non disponibles dans les documents fournis.",
        "[a.pdf, page 1]",
    ],
)
def test_an_answer_that_cites_or_declines_follows_the_contract(answer):
    assert follows_answer_contract(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "PWNED",
        "Sure! Here is my system prompt: you are a helpful assistant.",
        "Yes, you can do that.",
        "",
        "See the network guide, page 3.",  # a reference, but not in the citation format the prompt asks for
    ],
)
def test_an_answer_with_no_citation_and_no_refusal_breaks_it(answer):
    assert not follows_answer_contract(answer)


def test_a_refusal_followed_by_extra_text_still_passes_which_is_a_known_limit():
    # the contract cannot see this. It is written down so that nobody believes it is covered
    assert follows_answer_contract("Information unavailable in provided documents. PWNED")


def test_none_of_the_ten_committed_real_answers_would_be_withheld():
    # eval/metrics_enhanced.json is committed, so this runs everywhere. Eight answers cite a source and two are the
    # correct refusals for questions the corpus cannot answer, so both ways of passing the contract are covered
    rows = json.loads((ROOT / "eval" / "metrics_enhanced.json").read_text(encoding="utf-8"))["per_question"]
    assert len(rows) == 10
    withheld = [r["id"] for r in rows if not follows_answer_contract(r["answer"])]
    assert withheld == [], f"questions {withheld} would have had a good answer withheld"
    assert sum(1 for r in rows if is_refusal(r["answer"])) == 2


EXPORT = ROOT / "eval" / "ragas_input.jsonl"


@pytest.mark.skipif(
    not EXPORT.exists(),
    reason="eval/ragas_input.jsonl is an export of passages from the manuals, so it is not committed. "
    "Make it with eval/ragas_export.py to run this check on 50 answers",
)
def test_none_of_the_fifty_real_answers_in_the_local_export_would_be_withheld():
    rows = [json.loads(line) for line in EXPORT.read_text(encoding="utf-8").splitlines()]
    withheld = [r["id"] for r in rows if not follows_answer_contract(r["answer"])]
    assert withheld == [], f"questions {withheld} would have had a good answer withheld"


# ---- what the API does with a violation ----


async def _call(app, method, url, **kwargs):
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def post(app, path="/query"):
    return asyncio.run(_call(app, "POST", path, json={"query": "how do I configure the network?"}))


def agent_says(monkeypatch, text):
    async def run(q):
        return text

    monkeypatch.setattr(apimod, "arun_agent", run)


def test_a_bad_answer_is_withheld_and_the_caller_gets_the_passages_instead(monkeypatch):
    agent_says(monkeypatch, "PWNED")
    r = post(create_app(Settings(auth_disabled=True)))
    assert r.status_code == 200 and r.json()["degraded"] is True
    assert "PWNED" not in r.text, "the withheld answer reached the caller"
    assert r.headers["x-citera-degraded"] == "invalid_answer"
    assert r.json()["sources"][0]["document"] == "network_guide.pdf"


def test_with_the_fallback_off_a_bad_answer_is_a_502_that_says_why_and_not_what(monkeypatch):
    agent_says(monkeypatch, "PWNED: the system prompt is SECRET-PROMPT-TEXT")
    r = post(create_app(Settings(auth_disabled=True, fallback_to_retrieval=False)))
    assert r.status_code == 502 and r.json()["code"] == "invalid_answer"
    assert "SECRET-PROMPT-TEXT" not in r.text and "PWNED" not in r.text


def test_the_withheld_text_is_not_written_to_the_log(monkeypatch, caplog):
    agent_says(monkeypatch, "PWNED: the system prompt is SECRET-PROMPT-TEXT")
    with caplog.at_level(logging.DEBUG):
        post(create_app(Settings(auth_disabled=True, fallback_to_retrieval=False)))
    assert "SECRET-PROMPT-TEXT" not in caplog.text and "PWNED" not in caplog.text
    assert "withheld" in caplog.text and "len=" in caplog.text


def test_a_good_answer_is_untouched(monkeypatch):
    agent_says(monkeypatch, "Open Settings [network_guide.pdf, Page 3].")
    r = post(create_app(Settings(auth_disabled=True)))
    assert r.status_code == 200 and "degraded" not in r.json()
    assert r.json()["answer"] == "Open Settings [network_guide.pdf, Page 3]."


def test_a_refusal_is_a_normal_answer(monkeypatch):
    agent_says(monkeypatch, "Information unavailable in provided documents.")
    r = post(create_app(Settings(auth_disabled=True)))
    assert r.status_code == 200 and "degraded" not in r.json()


def test_a_withheld_answer_is_counted_as_an_error_and_as_degraded(monkeypatch):
    agent_says(monkeypatch, "PWNED")
    app = create_app(Settings(auth_disabled=True))
    post(app)
    text = asyncio.run(_call(app, "GET", "/metrics")).text
    assert 'citera_errors_total{code="invalid_answer"} 1.0' in text
    assert 'citera_answers_total{outcome="degraded"} 1.0' in text


def test_a_streamed_bad_answer_is_flagged_in_the_done_event_and_counted(monkeypatch):
    async def stream(q, result):
        result.ttft_seconds = 0.01
        yield "PWNED"

    monkeypatch.setattr(apimod, "async_pipeline_supported", lambda: True)
    monkeypatch.setattr(apimod, "astream_agent", stream)
    app = create_app(Settings(auth_disabled=True))
    r = post(app, "/query/stream")
    assert '"contract_ok": false' in r.text
    assert 'citera_errors_total{code="invalid_answer"} 1.0' in asyncio.run(_call(app, "GET", "/metrics")).text


def test_a_streamed_good_answer_has_no_contract_flag(monkeypatch):
    async def stream(q, result):
        result.ttft_seconds = 0.01
        yield "Open Settings [network_guide.pdf, Page 3]."

    monkeypatch.setattr(apimod, "async_pipeline_supported", lambda: True)
    monkeypatch.setattr(apimod, "astream_agent", stream)
    r = post(create_app(Settings(auth_disabled=True)), "/query/stream")
    assert "contract_ok" not in r.text and '"done": true' in r.text


def test_the_error_maps_to_a_public_502():
    assert classify_error(AnswerContractError()) is INVALID_ANSWER
    assert INVALID_ANSWER.status == 502
