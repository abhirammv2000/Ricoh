"""End to end: the real API process, the real retriever and the real model client, a fake model provider.

These are the failure drills. Each one is something the API did badly before it was hardened, and each
runs for real: a separate uvicorn process, a copy of the demo index, and the Anthropic client's own
retry and timeout code talking to tests/fake_anthropic.py. Nothing here is paid for.

They take a few seconds to start, because the embedding model and index load once at startup.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from tests.fake_anthropic import FakeAnthropic

ROOT = Path(__file__).resolve().parent.parent

# one client for every call, as a real caller has. httpx.post() builds a new client each time, and that includes loading
# the TLS certificates, which costs about 0.1 s and, with 8 threads at once, made the timings below measure the test.
CLIENT = httpx.Client(timeout=30)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def fake():
    with FakeAnthropic() as server:
        yield server


@pytest.fixture(scope="module")
def service(fake, tmp_path_factory):
    """The API in its own process. Its index is a copy, because opening a Chroma index writes to it."""
    work = tmp_path_factory.mktemp("e2e")
    shutil.copytree(ROOT / "demo_index", work / "demo_index")
    port = _free_port()
    env = {
        **os.environ,
        "CHROMA_DIR": str(work / "demo_index"),
        "DEMO_MODE": "true",
        "ANTHROPIC_API_KEY": "sk-fake",
        "ANTHROPIC_BASE_URL": fake.url,
        "REQUEST_TIMEOUT_SECONDS": "3",
        # one retry keeps a fast-failing provider inside the 3 s deadline, so overload is a 503 and not a 504
        "LLM_MAX_RETRIES": "1",
        "LLM_TIMEOUT_SECONDS": "20",
        "MAX_CONCURRENT_QUERIES": "2",
        "QUEUE_TIMEOUT_SECONDS": "0.1",
        "RATE_LIMIT_RPS": "1000",
        "RATE_LIMIT_BURST": "1000",
        "PYTHONIOENCODING": "utf-8",
    }
    log = open(work / "api.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:api", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=log, stderr=log,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            if httpx.get(base + "/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.3)
    else:
        proc.kill()
        pytest.fail("the API did not start:\n" + (work / "api.log").read_text(encoding="utf-8")[-2000:])
    # the first request loads the retriever, so do it before any timing
    httpx.post(base + "/query", json={"query": "warm up"}, timeout=60)
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(autouse=True)
def healthy_provider(fake):
    fake.behavior.mode = "ok"
    fake.behavior.latency = 0.05
    fake.behavior.chunk_delay = 0.0
    yield
    fake.behavior.mode = "ok"


def ask(base: str, query: str = "How do I configure the network settings?") -> httpx.Response:
    return CLIENT.post(base + "/query", json={"query": query})


def test_a_normal_question_is_answered(service):
    r = ask(service)
    assert r.status_code == 200
    assert "[network_guide.pdf, Page 3]" in r.json()["answer"]
    assert r.json()["llm_calls"] == 1


def test_provider_overload_is_a_clean_503_not_a_500_and_leaks_nothing(service, fake):
    fake.behavior.mode = "overloaded"
    r = ask(service)
    assert r.status_code == 503
    assert r.json()["code"] == "model_unavailable"
    assert r.headers["retry-after"]
    assert "overloaded_error" not in r.text and "Error code" not in r.text


def test_a_provider_that_never_answers_gives_a_504_at_the_deadline(service, fake):
    fake.behavior.mode = "hang"
    started = time.perf_counter()
    r = ask(service)
    took = time.perf_counter() - started
    assert r.status_code == 504
    assert r.json()["code"] == "timeout"
    assert 2.5 < took < 6, f"the 3 s deadline did not hold: {took:.1f} s"


def test_a_stream_hitting_provider_overload_ends_with_a_public_error_event(service, fake):
    fake.behavior.mode = "overloaded"
    with CLIENT.stream("POST", service + "/query/stream", json={"query": "network settings?"}) as s:
        body = "".join(s.iter_text())
    assert '"code": "model_unavailable"' in body
    assert "overloaded_error" not in body and "Error code" not in body


def test_when_full_extra_requests_are_refused_fast_and_admitted_ones_succeed(service, fake):
    fake.behavior.latency = 1.0  # each answer holds a slot for a second, and there are 2 slots

    def one(_):
        started = time.perf_counter()
        r = ask(service)
        return r.status_code, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(8)))
    ok = [t for status, t in results if status == 200]
    refused = [t for status, t in results if status == 503]
    assert len(ok) >= 2, results
    assert len(refused) >= 4, results
    assert max(refused) < 0.9, f"a full service should refuse fast, took {max(refused):.2f} s"
    assert min(ok) >= 0.9  # an admitted request still took the provider's second, not longer


def test_the_service_recovers_after_an_outage(service, fake):
    fake.behavior.mode = "overloaded"
    assert ask(service).status_code == 503
    fake.behavior.mode = "ok"
    assert ask(service).status_code == 200


def test_a_client_that_leaves_mid_stream_stops_the_model_call(service, fake):
    fake.behavior.chunk_delay = 0.4  # the answer takes about 4 s to stream
    done_before = fake.counters.completed
    cancelled_before = fake.counters.disconnected
    with CLIENT.stream("POST", service + "/query/stream", json={"query": "network settings?"}) as s:
        for line in s.iter_lines():
            if '"token"' in line:
                break  # the user closes the tab after the first word
    time.sleep(2.5)
    assert fake.counters.disconnected == cancelled_before + 1, "the provider connection was not closed"
    time.sleep(2.5)
    assert fake.counters.completed == done_before, "the model call ran to the end for nobody"
