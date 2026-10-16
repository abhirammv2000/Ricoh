"""A small fake of the Anthropic Messages API, for tests and load tests that must not spend money.

The real client code (langchain-anthropic and the Anthropic SDK, with their retries and timeouts) talks to
it, because ANTHROPIC_BASE_URL points the SDK at any server. That makes it possible to test what the service
does when the provider is slow, rate limiting or down, which a mock of the Python function never shows.

    python -m tests.fake_anthropic --port 9100 --latency 0.3

In a test, use the context manager and change its behavior while it runs:

    with FakeAnthropic() as fake:
        os.environ["ANTHROPIC_BASE_URL"] = fake.url
        fake.behavior.mode = "overloaded"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ANSWER = "Open the Settings menu and choose Network [network_guide.pdf, Page 3]."


@dataclass
class Behavior:
    """What the fake does with the next requests. Change it at any time."""

    # ok, rate_limit (429), overloaded (529), server_error (500), hang (never answers),
    # flaky (fails fail_every-th request with 529)
    mode: str = "ok"
    latency: float = 0.05
    answer: str = ANSWER
    input_tokens: int = 1200
    output_tokens: int = 40
    retry_after: float = 0.0
    fail_every: int = 2
    # for streaming: seconds between text chunks
    chunk_delay: float = 0.0
    # for streaming: stop sending after this many chunks, then hold the connection open
    stall_after_chunks: int | None = None


@dataclass
class Counters:
    requests: int = 0
    streamed: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    completed: int = 0
    disconnected: int = 0
    prompts: list[str] = field(default_factory=list)


def _usage(b: Behavior) -> dict:
    return {
        "input_tokens": b.input_tokens,
        "output_tokens": b.output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _message(b: Behavior, model: str) -> dict:
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": b.answer}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": _usage(b),
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def build_app(behavior: Behavior, counters: Counters) -> FastAPI:
    app = FastAPI()

    def error_response(status: int, kind: str, message: str, retry_after: float = 0.0) -> JSONResponse:
        headers = {"retry-after": str(retry_after)} if retry_after else {}
        body = {"type": "error", "error": {"type": kind, "message": message}}
        return JSONResponse(body, status_code=status, headers=headers)

    @app.post("/_admin/behavior")
    async def set_behavior(request: Request):
        """Change how the fake behaves while it runs, for a script on another machine or in another container.
        Not part of the Anthropic API."""
        changes = await request.json()
        for key, value in changes.items():
            if not hasattr(behavior, key):
                return JSONResponse({"error": f"unknown setting {key}"}, status_code=400)
            setattr(behavior, key, value)
        return JSONResponse({"mode": behavior.mode, "latency": behavior.latency})

    @app.get("/_admin/counters")
    async def get_counters():
        return JSONResponse({"requests": counters.requests, "completed": counters.completed, "in_flight": counters.in_flight})

    @app.post("/v1/messages")
    async def messages(request: Request):
        payload = await request.json()
        counters.requests += 1
        counters.in_flight += 1
        counters.max_in_flight = max(counters.max_in_flight, counters.in_flight)
        try:
            text = json.dumps(payload.get("messages", ""))
            counters.prompts.append(text[:2000])
            b = behavior
            mode = b.mode
            if mode == "flaky":
                mode = "overloaded" if counters.requests % max(1, b.fail_every) == 0 else "ok"
            if mode == "hang":
                await asyncio.sleep(3600)
            if mode == "rate_limit":
                return error_response(429, "rate_limit_error", "rate limited", b.retry_after)
            if mode == "overloaded":
                return error_response(529, "overloaded_error", "Overloaded")
            if mode == "server_error":
                return error_response(500, "api_error", "internal error")

            await asyncio.sleep(b.latency)
            model = payload.get("model", "claude-sonnet-4-6")
            if not payload.get("stream"):
                counters.completed += 1
                return JSONResponse(_message(b, model))

            counters.streamed += 1

            async def events():
                start = _message(b, model)
                start["content"] = []
                yield _sse("message_start", {"type": "message_start", "message": start})
                yield _sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                )
                words = b.answer.split(" ")
                try:
                    for i, word in enumerate(words):
                        if b.stall_after_chunks is not None and i >= b.stall_after_chunks:
                            await asyncio.sleep(3600)
                        piece = word if i == 0 else " " + word
                        yield _sse(
                            "content_block_delta",
                            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}},
                        )
                        if b.chunk_delay:
                            await asyncio.sleep(b.chunk_delay)
                except asyncio.CancelledError:
                    counters.disconnected += 1
                    raise
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                yield _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": _usage(b),
                    },
                )
                yield _sse("message_stop", {"type": "message_stop"})
                counters.completed += 1

            return StreamingResponse(events(), media_type="text/event-stream")
        finally:
            counters.in_flight -= 1

    return app


class FakeAnthropic:
    """The fake on a free local port, in a background thread."""

    def __init__(self, behavior: Behavior | None = None) -> None:
        self.behavior = behavior or Behavior()
        self.counters = Counters()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, port: int = 0) -> "FakeAnthropic":
        if port == 0:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        self.port = port
        config = uvicorn.Config(
            build_app(self.behavior, self.counters), host="127.0.0.1", port=port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("the fake Anthropic server did not start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> "FakeAnthropic":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fake Anthropic API")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--latency", type=float, default=0.3, help="seconds before each answer")
    parser.add_argument("--mode", default="ok")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to be reachable from a container")
    args = parser.parse_args()
    behavior = Behavior(latency=args.latency, mode=args.mode)
    app = build_app(behavior, Counters())
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
