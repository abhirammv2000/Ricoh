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
API_KEY = "e2e-test-key-0123456789"
CLIENT = httpx.Client(timeout=30, headers={"X-API-Key": API_KEY})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def fake():
    with FakeAnthropic() as server:
        yield server


def _launch(fake, work: Path, damage=None, env_extra=None):
    """Start the API in its own process on a copy of the demo index. `damage` is a function that is given the
    copied index folder and may break it. Returns (process, base_url, log_path)."""
    shutil.copytree(ROOT / "demo_index", work / "demo_index")
    if damage is not None:
        damage(work / "demo_index")
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
        "DOTENV_OVERRIDE": "0",
        "PYTHONIOENCODING": "utf-8",
        "API_KEYS": f"e2e:{API_KEY}",
        "API_AUTH_DISABLED": "false",
        "DAILY_BUDGET_USD": "0",
        **(env_extra or {}),
    }
    log_path = work / "api.log"
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:api", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=log, stderr=log,
    )
    base = f"http://127.0.0.1:{port}"
    # polled, so this costs nothing when the start is fast. The first import after the machine has been busy can take a
    # minute or more on a Windows laptop that has `transformers` installed (CI does not)
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            if httpx.get(base + "/health", timeout=2).status_code == 200:
                return proc, base, log_path
        except httpx.HTTPError:
            time.sleep(0.3)
    proc.kill()
    pytest.fail("the API did not start:\n" + log_path.read_text(encoding="utf-8")[-2000:])


def _stop(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def service(fake, tmp_path_factory):
    """The API in its own process. Its index is a copy, because opening a Chroma index writes to it."""
    proc, base, _ = _launch(fake, tmp_path_factory.mktemp("e2e"))
    yield base
    _stop(proc)


@pytest.fixture(scope="module")
def strict_service(fake, tmp_path_factory):
    """The API with the retrieval fallback off, so a provider failure is reported as an error."""
    proc, base, _ = _launch(fake, tmp_path_factory.mktemp("e2e_strict"), env_extra={"FALLBACK_TO_RETRIEVAL": "false"})
    yield base
    _stop(proc)


@pytest.fixture(scope="module")
def budget_service(fake, tmp_path_factory):
    """The API with a daily budget of half a cent. One answer from the fake costs 0.0042 dollars."""
    proc, base, _ = _launch(fake, tmp_path_factory.mktemp("e2e_budget"), env_extra={"DAILY_BUDGET_USD": "0.005"})
    yield base
    _stop(proc)


@pytest.fixture(scope="module")
def broken_service(fake, tmp_path_factory):
    """The API started on an index whose BM25 file has been damaged."""

    def damage(index_dir: Path) -> None:
        path = index_dir / "bm25_chunks.pkl"
        path.write_bytes(path.read_bytes()[:-7])

    work = tmp_path_factory.mktemp("e2e_broken")
    proc, base, log_path = _launch(fake, work, damage=damage)
    yield base, log_path
    _stop(proc)


@pytest.fixture(autouse=True)
def healthy_provider(fake):
    fake.behavior.mode = "ok"
    fake.behavior.latency = 0.05
    fake.behavior.chunk_delay = 0.0
    original_answer = fake.behavior.answer
    yield
    fake.behavior.mode = "ok"
    fake.behavior.answer = original_answer


def ask(base: str, query: str = "How do I configure the network settings?") -> httpx.Response:
    return CLIENT.post(base + "/query", json={"query": query})


def test_a_normal_question_is_answered(service):
    r = ask(service)
    assert r.status_code == 200
    assert "[network_guide.pdf, Page 3]" in r.json()["answer"]
    assert r.json()["llm_calls"] == 1


def test_provider_overload_is_a_clean_503_not_a_500_and_leaks_nothing(strict_service, fake):
    fake.behavior.mode = "overloaded"
    r = ask(strict_service)
    assert r.status_code == 503
    assert r.json()["code"] == "model_unavailable"
    assert r.headers["retry-after"]
    assert "overloaded_error" not in r.text and "Error code" not in r.text


def test_a_provider_that_never_answers_gives_a_504_at_the_deadline(strict_service, fake):
    fake.behavior.mode = "hang"
    started = time.perf_counter()
    r = ask(strict_service)
    took = time.perf_counter() - started
    assert r.status_code == 504
    assert r.json()["code"] == "timeout"
    assert 2.5 < took < 6, f"the 3 s deadline did not hold: {took:.1f} s"


def test_a_stream_hitting_provider_overload_ends_with_a_public_error_event(strict_service, fake):
    fake.behavior.mode = "overloaded"
    with CLIENT.stream("POST", strict_service + "/query/stream", json={"query": "network settings?"}) as s:
        body = "".join(s.iter_text())
    assert '"code": "model_unavailable"' in body
    assert "overloaded_error" not in body and "Error code" not in body
    assert "degraded" not in body


def test_a_stream_hitting_provider_overload_degrades_to_the_retrieved_passages(service, fake):
    fake.behavior.mode = "overloaded"
    with CLIENT.stream("POST", service + "/query/stream", json={"query": "network settings?"}) as s:
        body = "".join(s.iter_text())
    assert '"degraded": true' in body and '"sources"' in body
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


def test_the_service_recovers_after_an_outage(strict_service, fake):
    fake.behavior.mode = "overloaded"
    assert ask(strict_service).status_code == 503
    fake.behavior.mode = "ok"
    assert ask(strict_service).status_code == 200


def test_during_a_provider_outage_the_answer_is_the_retrieved_passages_and_then_normal_again(service, fake):
    fake.behavior.mode = "overloaded"
    down = ask(service, "What does the network guide say about settings?")
    assert down.status_code == 200
    body = down.json()
    # the failed attempt is recorded as a call, with no cost, since the provider never reported usage
    assert body["degraded"] is True and body["llm_calls"] == 1 and body["cost_usd"] == 0.0
    assert body["sources"], "retrieval found nothing in the demo index"
    assert all(s["document"].endswith(".pdf") and s["excerpt"] for s in body["sources"])
    assert down.headers["x-citera-degraded"] == "model_unavailable"
    assert "overloaded_error" not in down.text and "Error code" not in down.text
    fake.behavior.mode = "ok"
    up = ask(service)
    assert up.status_code == 200 and "degraded" not in up.json()


def test_the_metrics_show_what_happened(service, fake):
    def series(text, name):
        for line in text.splitlines():
            if line.startswith(name + " "):
                return float(line.split()[-1])
        return 0.0

    before = CLIENT.get(service + "/metrics").text
    assert ask(service).status_code == 200  # a normal answer
    fake.behavior.mode = "overloaded"
    assert ask(service).json()["degraded"] is True
    fake.behavior.mode = "ok"
    after = CLIENT.get(service + "/metrics").text

    assert series(after, 'citera_answers_total{outcome="answered"}') == series(before, 'citera_answers_total{outcome="answered"}') + 1
    assert series(after, 'citera_answers_total{outcome="degraded"}') == series(before, 'citera_answers_total{outcome="degraded"}') + 1
    assert series(after, "citera_retrieval_top_rrf_score_count") >= series(before, "citera_retrieval_top_rrf_score_count") + 2
    assert series(after, 'citera_llm_tokens_total{kind="input"}') > series(before, 'citera_llm_tokens_total{kind="input"}')
    assert series(after, "citera_llm_cost_usd_total") > series(before, "citera_llm_cost_usd_total")
    assert series(after, "citera_ready") == 1
    assert "How do I configure" not in after, "question text reached the metrics"


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


def test_a_ready_service_says_what_it_is_serving(service):
    r = CLIENT.get(service + "/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready" and body["chunks"] == 88
    assert body["index_version"] == "53b873d83374"  # from demo_index/index_manifest.json


def test_the_first_question_after_startup_is_not_slower_than_the_rest(fake, tmp_path_factory):
    # the embedding model loads at startup now. Before, the first question took about 2.7 s and the rest 0.4 s
    proc, base, _ = _launch(fake, tmp_path_factory.mktemp("e2e_first"))
    try:
        times = []
        for _ in range(4):
            started = time.perf_counter()
            assert CLIENT.post(base + "/query", json={"query": "How do I configure the network settings?"}).status_code == 200
            times.append(time.perf_counter() - started)
        assert times[0] < 1.0, f"the first question took {times[0]:.2f} s: {times}"
        assert times[0] < 3 * (sum(times[1:]) / 3) + 0.3, times
    finally:
        _stop(proc)


def test_a_damaged_index_keeps_the_process_alive_but_not_ready_and_answers_nothing(broken_service, fake):
    base, log_path = broken_service
    calls_before = fake.counters.requests
    assert CLIENT.get(base + "/health").status_code == 200, "a bad index must not look like a dead process"
    ready = CLIENT.get(base + "/ready")
    assert ready.status_code == 503
    assert "bm25_chunks.pkl" in " ".join(ready.json()["problems"])
    answer = CLIENT.post(base + "/query", json={"query": "How do I configure the network settings?"})
    assert answer.status_code == 503 and answer.json()["code"] == "not_ready"
    stream = CLIENT.post(base + "/query/stream", json={"query": "How do I configure the network settings?"})
    assert stream.status_code == 503
    assert fake.counters.requests == calls_before, "the model was called with no usable index"


def test_the_real_service_requires_an_api_key(service):
    body = {"query": "How do I configure the network settings?"}
    anonymous = httpx.post(service + "/query", json=body, timeout=30)
    assert anonymous.status_code == 401 and anonymous.json()["code"] == "unauthorized"
    wrong = httpx.post(service + "/query", json=body, headers={"X-API-Key": API_KEY + "x"}, timeout=30)
    assert wrong.status_code == 401 and API_KEY not in wrong.text
    assert httpx.post(service + "/query", json=body, headers={"Authorization": "Bearer " + API_KEY}, timeout=30).status_code == 200
    assert httpx.get(service + "/metrics", timeout=30).status_code == 401
    assert httpx.get(service + "/health", timeout=30).status_code == 200  # probes need no key
    assert httpx.get(service + "/ready", timeout=30).status_code == 200


def test_a_service_started_with_no_keys_and_auth_not_switched_off_refuses_to_start(fake, tmp_path_factory):
    work = tmp_path_factory.mktemp("e2e_nokeys")
    shutil.copytree(ROOT / "demo_index", work / "demo_index")
    env = {**os.environ, "CHROMA_DIR": str(work / "demo_index"), "ANTHROPIC_API_KEY": "sk-fake",
           "ANTHROPIC_BASE_URL": fake.url, "API_AUTH_DISABLED": "false", "PYTHONIOENCODING": "utf-8"}
    env.pop("API_KEYS", None)
    result = subprocess.run([sys.executable, "-m", "uvicorn", "api.main:api", "--port", str(_free_port())],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "API_KEYS is not set" in result.stdout + result.stderr


def test_the_daily_budget_stops_spending_on_the_real_service(budget_service, fake):
    body = {"query": "How do I configure the network settings?"}
    headers = {"X-API-Key": API_KEY}
    calls_before = fake.counters.requests
    first = httpx.post(budget_service + "/query", json=body, headers=headers, timeout=30)
    second = httpx.post(budget_service + "/query", json=body, headers=headers, timeout=30)
    assert first.status_code == 200 and "degraded" not in first.json()
    assert second.status_code == 200 and "degraded" not in second.json()  # 0.0084 spent: crossed on this one
    third = httpx.post(budget_service + "/query", json=body, headers=headers, timeout=30)
    assert third.status_code == 200 and third.json()["degraded"] is True
    assert third.headers["x-citera-degraded"] == "budget_exceeded"
    assert third.json()["sources"]
    assert fake.counters.requests - calls_before == 2, "the model was called after the budget was spent"
    metrics = httpx.get(budget_service + "/metrics", headers=headers, timeout=30).text
    assert "citera_budget_limit_usd 0.005" in metrics


def test_a_model_that_obeys_an_injected_instruction_has_its_answer_withheld(service, fake):
    # what a successful injection looks like from the outside: the model answers with the attacker's word and no citation
    fake.behavior.answer = "PWNED"
    r = ask(service)
    assert r.status_code == 200 and r.json()["degraded"] is True
    assert "PWNED" not in r.text
    assert r.headers["x-citera-degraded"] == "invalid_answer"
    assert r.json()["sources"], "the passages should replace the withheld answer"
    metrics = CLIENT.get(service + "/metrics").text
    assert 'citera_errors_total{code="invalid_answer"}' in metrics
