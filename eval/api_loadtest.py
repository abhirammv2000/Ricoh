"""Load test of the running HTTP API, with latency percentiles and a status histogram.

eval/loadtest.py calls the agent in process. This one goes through the real server, so it includes the
HTTP layer, auth, rate limits and whatever the service does when it is busy or the model provider is not
well. Run the API against the fake Anthropic server (tests/fake_anthropic.py) and it costs nothing.

    python -m eval.api_loadtest --url http://127.0.0.1:8000 --n 200 --concurrency 20
    python -m eval.api_loadtest --n 100 --concurrency 10 --api-key local-dev-key --stream
    python -m eval.api_loadtest --n 60 --concurrency 6 --users 6      # each request from one of 6 clients

--users sends a different X-Forwarded-For per simulated client. The request counts as the same client to
a server that ignores that header, which is what happens to every user behind a proxy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import Counter

import httpx

QUESTIONS = [
    "How do I configure the network settings?",
    "What paper does the bypass tray take?",
    "How do I shut the system down?",
    "How do I add a printer to a workflow?",
    "What does error SC542 mean?",
    "How do I back up the database?",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


async def one_request(
    client: httpx.AsyncClient, i: int, args: argparse.Namespace
) -> tuple[int, float, float | None]:
    """Returns (status, seconds, seconds to the first streamed token)."""
    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    if args.users:
        headers["X-Forwarded-For"] = f"10.0.0.{i % args.users + 1}"
    body = {"query": QUESTIONS[i % len(QUESTIONS)]}
    started = time.perf_counter()
    ttft = None
    try:
        if args.stream:
            async with client.stream("POST", "/query/stream", json=body, headers=headers) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data:") and ttft is None and '"token"' in line:
                        ttft = time.perf_counter() - started
                status = resp.status_code
        else:
            resp = await client.post("/query", json=body, headers=headers)
            status = resp.status_code
    except httpx.TimeoutException:
        status = 0  # the client gave up before the server answered
    except httpx.HTTPError:
        status = -1
    return status, time.perf_counter() - started, ttft


async def run(args: argparse.Namespace) -> dict:
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency + 5)
    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout, limits=limits) as client:

        async def bounded(i: int):
            async with sem:
                return await one_request(client, i, args)

        started = time.perf_counter()
        results = await asyncio.gather(*(bounded(i) for i in range(args.n)))
        wall = time.perf_counter() - started

    statuses = Counter(r[0] for r in results)
    ok = [r[1] for r in results if r[0] == 200]
    ttfts = [r[2] for r in results if r[0] == 200 and r[2] is not None]
    report = {
        "requests": args.n,
        "concurrency": args.concurrency,
        "seconds": round(wall, 2),
        "throughput_rps": round(args.n / wall, 2) if wall else 0,
        "statuses": dict(sorted(statuses.items())),
        "success_rate": round(len(ok) / args.n, 4),
        "latency_ok_p50": round(percentile(ok, 50), 3),
        "latency_ok_p95": round(percentile(ok, 95), 3),
        "latency_ok_p99": round(percentile(ok, 99), 3),
        "latency_all_max": round(max((r[1] for r in results), default=0), 3),
    }
    if ttfts:
        report["ttft_p50"] = round(percentile(ttfts, 50), 3)
        report["ttft_p95"] = round(percentile(ttfts, 95), 3)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the Citera HTTP API")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--users", type=int, default=0, help="simulate this many clients with X-Forwarded-For")
    parser.add_argument("--timeout", type=float, default=300.0, help="client timeout in seconds")
    parser.add_argument("--json", action="store_true", help="print only the JSON report")
    args = parser.parse_args()

    report = asyncio.run(run(args))
    print(json.dumps(report, indent=None if args.json else 2))


if __name__ == "__main__":
    main()
