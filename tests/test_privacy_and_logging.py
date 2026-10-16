"""Question text stays out of logs, traces can be set to hide it, log lines carry the request id, and the trace
file stays valid when several processes write to it at once."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx
import pytest

import api.main as apimod
from api.logging_config import JsonFormatter, configure_logging, request_id_var
from api.main import create_app
from api.readiness import Readiness
from api.settings import Settings, SettingsError
from src import instrumentation
from src.instrumentation import RunRecord, append_trace_line, record_run
from src.privacy import describe_query, query_id, trace_mode, trace_query
from tests.test_index_manifest import build

QUESTION = "my printer serial number is PX-99-SECRET and my name is Dana"
ROOT = Path(__file__).resolve().parent.parent


# ---- what is safe to write about a question ----


def test_a_description_has_a_length_and_a_stable_id_and_none_of_the_text():
    text = describe_query(QUESTION)
    assert f"len={len(QUESTION)}" in text and query_id(QUESTION) in text
    assert "SECRET" not in text and "Dana" not in text
    assert describe_query(QUESTION) == describe_query(QUESTION)
    assert describe_query(QUESTION) != describe_query(QUESTION + "!")


def test_trace_modes():
    assert trace_query(QUESTION, "full") == QUESTION
    hashed = trace_query(QUESTION, "hash")
    assert "SECRET" not in hashed and query_id(QUESTION) in hashed
    assert trace_query(QUESTION, "none") == ""


def test_the_default_trace_mode_keeps_the_text_for_local_work_and_a_bad_mode_is_an_error(monkeypatch):
    monkeypatch.delenv("TRACE_QUERY_MODE", raising=False)
    assert trace_mode() == "full"
    monkeypatch.setenv("TRACE_QUERY_MODE", "redacted")
    with pytest.raises(ValueError, match="TRACE_QUERY_MODE"):
        trace_mode()


def test_a_trace_record_follows_the_mode(monkeypatch):
    monkeypatch.setenv("TRACE_QUERY_MODE", "hash")
    record = RunRecord(trace_id="t", query=QUESTION).to_dict()
    assert "SECRET" not in json.dumps(record)
    monkeypatch.setenv("TRACE_QUERY_MODE", "full")
    assert RunRecord(trace_id="t", query=QUESTION).to_dict()["query"] == QUESTION


def test_retrieval_does_not_log_the_question_and_does_not_put_it_in_the_span_in_hash_mode(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TRACE_QUERY_MODE", "hash")
    retriever = build(tmp_path)
    # the level is set on the retriever's own logger, because eval_harness.py and app/main.py silence it at import
    with caplog.at_level(logging.DEBUG, logger="src.retriever"), record_run(query=QUESTION, persist=False) as rec:
        retriever.retrieve(QUESTION, top_k=3, final_k=2)
    assert "SECRET" not in caplog.text and "Dana" not in caplog.text
    assert f"len={len(QUESTION)}" in caplog.text
    span_text = json.dumps([s.attributes for s in rec.spans], default=str)
    assert "SECRET" not in span_text


# ---- the trace file ----


def test_a_line_is_appended_whole_and_the_directory_is_created(tmp_path):
    path = tmp_path / "deep" / "traces.jsonl"
    append_trace_line(path, '{"a": 1}\n')
    append_trace_line(path, '{"a": 2}\n')
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [{"a": 1}, {"a": 2}]


def test_a_full_trace_file_is_rotated_so_a_long_running_service_cannot_fill_its_disk(tmp_path):
    path = tmp_path / "traces.jsonl"
    line = json.dumps({"x": "y" * 200}) + "\n"
    for _ in range(10):
        append_trace_line(path, line, max_bytes=1000)
    assert (tmp_path / "traces.jsonl.1").exists()
    assert path.stat().st_size < 1000 + len(line)
    assert all(json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())


def test_the_trace_directory_can_be_moved_for_a_read_only_container(tmp_path):
    code = "import src.instrumentation as i; print(i.TRACE_PATH)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "TRACE_DIR": str(tmp_path / "elsewhere")},
    )
    assert Path(out.stdout.strip().splitlines()[-1]) == tmp_path / "elsewhere" / "traces.jsonl", out.stderr[-500:]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows does not make appends atomic across processes (6 writers lost 85 of 240 lines when this was run there). "
    "The service runs on Linux, where this runs, in CI and in the container.",
)
def test_several_processes_writing_large_lines_at_once_never_interleave(tmp_path):
    # each line is about 60 KB, far past a buffered write's 8 KB, which is where two writers' halves would mix
    path = tmp_path / "traces.jsonl"
    worker = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        from src.instrumentation import append_trace_line
        path, me = Path(sys.argv[2]), sys.argv[3]
        for i in range(40):
            append_trace_line(path, json.dumps({"worker": me, "i": i, "pad": me * 60000}) + "\\n", max_bytes=0)
        """
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", worker, str(ROOT), str(path), letter])
        for letter in "ABCDEF"
    ]
    for p in procs:
        assert p.wait(timeout=120) == 0
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 * 40
    records = [json.loads(line) for line in lines]  # a mixed line would not parse
    assert all(r["pad"] == r["worker"] * 60000 for r in records)


# ---- structured logs ----


def log_record(message="hello", **extra) -> logging.LogRecord:
    return logging.LogRecord("citera.test", logging.WARNING, __file__, 1, message, (), None, extra=None) if not extra else \
        _with_extra(message, extra)


def _with_extra(message, extra):
    record = logging.LogRecord("citera.test", logging.WARNING, __file__, 1, message, (), None)
    record.__dict__.update(extra)
    return record


def test_a_json_log_line_has_the_standard_fields_and_the_request_id():
    token = request_id_var.set("req-123")
    try:
        line = JsonFormatter().format(log_record("something happened"))
    finally:
        request_id_var.reset(token)
    entry = json.loads(line)
    assert entry["message"] == "something happened" and entry["level"] == "WARNING"
    assert entry["logger"] == "citera.test" and entry["request_id"] == "req-123"
    assert entry["ts"].endswith("+00:00")


def test_extra_fields_are_included_and_an_odd_value_does_not_break_the_line():
    line = JsonFormatter().format(_with_extra("x", {"code": "timeout", "thing": object()}))
    entry = json.loads(line)
    assert entry["code"] == "timeout" and "thing" in entry


def test_an_exception_is_included_in_the_line():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys as _sys

        record = logging.LogRecord("citera.test", logging.ERROR, __file__, 1, "failed", (), _sys.exc_info())
    assert "RuntimeError: boom" in json.loads(JsonFormatter().format(record))["exception"]


def test_configure_logging_changes_existing_handlers_only_when_json_is_asked_for():
    root = logging.getLogger()
    handler = logging.StreamHandler()
    original = handler.formatter
    root.addHandler(handler)
    try:
        configure_logging("text")
        assert handler.formatter is original
        configure_logging("json")
        configure_logging("json")  # twice is fine
        assert isinstance(handler.formatter, JsonFormatter)
        assert root.handlers.count(handler) == 1
    finally:
        root.removeHandler(handler)


def test_the_log_format_setting_accepts_text_and_json_and_rejects_anything_else():
    assert Settings.from_env({}).log_format == "text"
    assert Settings.from_env({"LOG_FORMAT": "JSON"}).log_format == "json"
    with pytest.raises(SettingsError, match="LOG_FORMAT"):
        Settings.from_env({"LOG_FORMAT": "xml"})


# ---- the request id reaches log lines from code that knows nothing about HTTP ----


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.seen.append((record.getMessage(), request_id_var.get()))


@pytest.fixture
def capture():
    handler = Capture()
    logging.getLogger().addHandler(handler)
    yield handler
    logging.getLogger().removeHandler(handler)


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(apimod, "compute_readiness", lambda require_manifest=True: Readiness(index_version="t", chunks=1))


async def _call(app, method, url, **kwargs):
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def test_a_line_logged_deep_in_the_agent_carries_the_request_id(monkeypatch, capture):
    async def agent(q):
        logging.getLogger("src.somewhere.deep").warning("deep in the agent")
        await asyncio.sleep(0)
        await asyncio.to_thread(logging.getLogger("src.retriever").warning, "in a worker thread")
        return "ok [a.pdf, Page 1]."

    monkeypatch.setattr(apimod, "arun_agent", agent)
    app = create_app(Settings(auth_disabled=True))
    r = asyncio.run(_call(app, "POST", "/query", json={"query": "how do I shut it down?"}))
    rid = r.headers["x-request-id"]
    seen = dict(capture.seen)
    assert seen["deep in the agent"] == rid
    assert seen["in a worker thread"] == rid, "context did not reach a thread"


def test_two_requests_at_once_keep_their_own_ids(monkeypatch, capture):
    async def agent(q):
        await asyncio.sleep(0.05)
        logging.getLogger("src.agent").warning("answering " + q)
        return "ok [a.pdf, Page 1]."

    monkeypatch.setattr(apimod, "arun_agent", agent)
    app = create_app(Settings(auth_disabled=True))

    async def both():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await asyncio.gather(
                client.post("/query", json={"query": "question one"}),
                client.post("/query", json={"query": "question two"}),
            )

    r1, r2 = asyncio.run(both())
    seen = dict(capture.seen)
    assert seen["answering question one"] == r1.headers["x-request-id"]
    assert seen["answering question two"] == r2.headers["x-request-id"]
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


def test_the_id_does_not_leak_into_code_that_runs_after_the_request(monkeypatch):
    async def agent(q):
        return "ok [a.pdf, Page 1]."

    monkeypatch.setattr(apimod, "arun_agent", agent)
    asyncio.run(_call(create_app(Settings(auth_disabled=True)), "POST", "/query", json={"query": "how do I shut it down?"}))
    assert request_id_var.get() == ""
