# Running Citera as a service

How to run the HTTP API, what it does when things go wrong, what to watch, and what to do when an alert fires.
The service is the code in `api/`. The Streamlit page in `app/` is a demo and is not covered here.

## What it is

A question goes in, an answer with page-level citations comes out.

```
POST /query  ->  check the key and the rate limit  ->  is the index fit?  ->  take a slot
                  ->  retrieve (vector + BM25, fused)  ->  withhold suspicious chunks
                  ->  one model call, which must cite  ->  record metrics and the trace  ->  answer
```

If the model provider fails or is too slow, the answer is the retrieved passages, marked as degraded. If the
index is not fit to answer from, the service refuses with a 503 and says why on `/ready`.

## Settings

All by environment variable. A value that is set and wrong stops the service at startup and names the variable.

| Variable | Default | What it does |
|---|---|---|
| `API_KEYS` | none | Who may call. Comma separated, each `label:secret` or just a secret. Required: the service will not start without it. |
| `API_AUTH_DISABLED` | `false` | `true` runs with no authentication. For a laptop only. |
| `ANTHROPIC_API_KEY` | none | The model provider's key. `/ready` fails without it. |
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, `google` or `self_hosted`. |
| `CHROMA_DIR` | `chroma_db` | The index folder. The image sets it to the bundled `demo_index`. |
| `REQUIRE_INDEX_MANIFEST` | `true` | Serve only an index that has a manifest and passes its checks. |
| `REQUEST_TIMEOUT_SECONDS` | `30` (image: `45`) | The longest one question may take. After it, a 504 or a degraded answer. |
| `MAX_CONCURRENT_QUERIES` | `16` | Questions handled at once. |
| `QUEUE_TIMEOUT_SECONDS` | `2` | How long an extra request waits for a slot before a 503. |
| `RATE_LIMIT_RPS`, `RATE_LIMIT_BURST` | `1`, `10` | Per caller (by key). |
| `DAILY_BUDGET_USD` | `20` | The most the model may cost in one UTC day. `0` turns it off. |
| `FALLBACK_TO_RETRIEVAL` | `true` | Answer with passages when the model is down, too slow or the budget is spent. |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | `60`, `3` (image: `30`, `1`) | The model client. Keep `timeout x (retries + 1)` under `REQUEST_TIMEOUT_SECONDS`, or a slow, failing provider gives a 504. The service warns at startup if not. |
| `LOG_FORMAT` | `text` (image: `json`) | `json` is one object per line, with the request id. |
| `TRACE_QUERY_MODE` | `full` (image: `hash`) | `full`, `hash` or `none`. What a trace records of the question. Use `hash` for a deployed service. |
| `TRACE_DIR`, `TRACE_MAX_BYTES` | `traces/`, 50 MB | Where traces go, and when the file is rotated. |
| `DOTENV_OVERRIDE` | `false` | `true` lets a `.env` file beat the real environment. For a laptop. |

## Endpoints

| | Needs a key | What it is |
|---|---|---|
| `GET /health` | no | Is the process alive. Never looks at the index. |
| `GET /ready` | no | Can it give a correct answer. 503 with the reasons when not. |
| `POST /query` | yes | `{"query": "..."}` returns `answer`, `cost_usd`, `llm_calls`, `latency_seconds`. Degraded answers also have `degraded` and `sources`. |
| `POST /query/stream` | yes | The same as server-sent events: `{"token": ...}`, then `{"done": true, ...}`. |
| `GET /metrics` | yes | Prometheus metrics. Meanings are in `api/metrics.py`. |

Pass the key as `X-API-Key: ...` or `Authorization: Bearer ...`. Every response has an `X-Request-ID`, and every
error body carries it, so a user who reports a problem can give you one string that finds every log line.

| Status, `code` | Meaning | Caller should |
|---|---|---|
| 400 | Input rejected by the prompt-injection screen. | Not retry. |
| 401 `unauthorized` | No key, or a wrong one. | Fix the key. |
| 422 | Empty question, or over 2000 characters. | Fix the request. |
| 429 | Rate limit, or too many wrong keys. `Retry-After` says when. | Wait. |
| 503 `overloaded` | The service is full. | Retry after `Retry-After`. |
| 503 `not_ready` | The index is not fit to answer from. | Retry later. Page someone. |
| 503 `model_unavailable`, 504 `timeout` | Only with `FALLBACK_TO_RETRIEVAL=false`. | Retry later. |
| 502 `invalid_answer` | The model's answer had no citation and no refusal, so it was withheld. Only with the fallback off. | Retry. |
| 200 with `degraded: true` | The model could not answer, so this is the passages. `X-Citera-Degraded` says why. | Show the passages, and say so. |

## The index

The index is two stores that must agree: the Chroma vectors and the BM25 files. `index_manifest.json` beside them
records how they were built (embedding model, chunk size, chunk count, a checksum of each BM25 file). At startup the
service compares the manifest with the files and with its own configuration, and refuses to answer if they differ.

```
python -m src.index_manifest check --persist-dir /path/to/index     # prints problems, exits 1 if any
python -m src.index_manifest write --persist-dir /path/to/index     # for an index built before manifests existed
```

To change the index, do not edit the live one. Build the new one in a new folder (`python -m src.ingest`, then
`check`), point `CHROMA_DIR` at it and restart. If the new index is bad the instance stays not ready and takes no
traffic, which is the safe way for a bad index to fail. Changing `EMBEDDING_MODEL` needs a new index built with it.

## Targets

Provisional, written before there was traffic. Set them from a month of real numbers.

| | Target | Why |
|---|---|---|
| Availability of `/query` | 99.5% of requests are not 5xx | A degraded answer counts as available. |
| Latency | 95th percentile under 25 s | The median is about 10 s, nearly all of it the model. |
| Refusals | under 35% of questions | Know your normal first. The benchmark has questions with no answer in the corpus. |
| Invalid citations | none | None in the 100-question benchmark. |
| Quarantined chunks | none | None in 1,322 real chunks. |

## Capacity

Measured against the fake model provider, one process, a 0.3 s provider, 100 questions at concurrency 10:

| | Before the work in this branch | Now |
|---|---|---|
| Throughput | 4.3 questions per second | 21.9 |
| Median latency | 2.32 s | 0.41 s |
| First question after start | 2.66 s | 0.18 s |

With the real provider the time is the model's, about 10 s a question, so one process holds
`MAX_CONCURRENT_QUERIES` questions open at once and CPU is not the limit. Add capacity with more containers, not
more workers. The rate limits, the concurrency cap and the daily budget are counted **per process**, so with N
containers the real ceilings are N times the settings, and a restart resets the budget count. Treat
`DAILY_BUDGET_USD` as a backstop and set a spend alert with the provider as well.

## When an alert fires

Every alert is in `ops/alerts.yml`. Look at the `X-Request-ID` of a failing request first, and search the logs for it.

### CiteraNotReady

The instance cannot give a correct answer and is refusing questions. `curl http://HOST:8000/ready`: the `problems`
list says which of these it is.
- *the model provider is not configured*: `ANTHROPIC_API_KEY` is missing in this environment.
- *vector index is empty / BM25 not loaded*: `CHROMA_DIR` points at the wrong place, or the index was never built.
- *vector index has N chunks but the BM25 index has M*: a partial re-ingest. Rebuild the whole index.
- *embedding model X but configured for Y*: `EMBEDDING_MODEL` changed without a rebuild. Set it back or rebuild.
- *does not match the index manifest, so it was not loaded*: a file changed or is damaged. Restore the index from where it was built. Do not delete the manifest to get past this.

### CiteraServerErrors

Questions are failing with 500, 502 or 504. Search the logs for `request_id` and `level=ERROR`.
`service_misconfigured` means our own provider key is rejected: rotate it. `internal_error` is a bug, so keep the
request id and open an issue.

### CiteraModelOutage

The provider is failing or too slow, and users are getting passages instead of answers. Check the provider's
status page and the `citera_errors_total` codes. Nothing needs doing here. It clears when the provider recovers.
If it lasts, `FALLBACK_TO_RETRIEVAL=false` makes it an error instead, which some callers prefer.

### CiteraSlowAnswers

The 95th percentile is over 25 s. Normal time is about 10 s. Check `citera_in_flight_queries` (is the service
full?), the provider's status, and `citera_llm_tokens_total` (did prompts get longer?). A planner or verifier turned
on makes each answer two to three times slower and dearer, and is off by default for that reason.

### CiteraTurningAwayCallers

All `MAX_CONCURRENT_QUERIES` slots are busy and extra questions get a 503. Raise `MAX_CONCURRENT_QUERIES` if the
provider's rate limits allow, or add a container.

### CiteraBudgetNearlyUsed

80% of today's model budget is gone. Check `citera_http_requests_total` for a spike and `X-Request-ID` logs for who.
A key that is looping is revoked by removing it from `API_KEYS`. Past the limit, answers become passages, and the
count resets at midnight UTC.

### CiteraCostPerQuestionUp

Cost per question is 50% above the same time last week. Compare `citera_llm_tokens_total{kind="input"}` per question
with last week: longer prompts mean longer chunks or a changed `RETRIEVAL_FINAL_K`. Compare `citera_llm_calls_total`
per question: more than one means a planner, verifier or tool loop is on.

### CiteraRefusalsUp

More than a third of questions are being refused ("Information unavailable in provided documents"). Either people
are asking about things the corpus does not cover, which is a content problem, or retrieval broke. Check
`citera_retrieval_top_rrf_score` and `citera_empty_retrievals_total`: if those fell too, it is retrieval. Look at
the questions: with `TRACE_QUERY_MODE=full` in a test environment, replay the refused ones.

### CiteraRetrievalFindingNothing

Retrieval returned no chunks for some questions. Check the index (`/ready`) and whether `CHROMA_DIR` changed.

### CiteraRetrievalWeaker

The best chunk scores lower than a week ago. Most likely the index changed (see `index_version` in `/ready` and in
the trace `meta`) or the questions changed. Compare `index_version` before and after.

### CiteraInvalidCitations

An answer named a document that was not in its evidence, so the citation is invented. Find it in the trace with the
`citation_guardrail` span, and check `prompt_version` and `model` in the trace `meta` for what changed.

### CiteraAnswersBreakingContract

The model returned an answer that neither cites a source nor declines, so it was withheld and the caller got the
retrieved passages (or a 502 `invalid_answer` with the fallback off). The log line has the request id and the length,
never the text. Look at the trace `meta` for `model` and `prompt_version`: a change in either is the likely cause.
If it follows a new document being added, check for `citera_quarantined_chunks_total` too. For a streamed answer the
text has already been sent, so the final event carries `contract_ok: false` and the client should discard it.

### CiteraChunksQuarantined

A retrieved chunk looked like an instruction to the model and was withheld. The log line names the document and page,
never the text. A document in the corpus is carrying an attack, or someone is poisoning it. Find the document,
remove it from the corpus, rebuild the index, and find out how it got in. `python -m src.index_manifest check`
also lists such chunks.

## What this does not cover

- **No dashboards.** The metrics are there. Drawing them is the next job.
- **Limits are per process.** See Capacity.
- **No automatic reindexing.** A changed document needs a rebuild and a restart, as above.
- **The answer-quality figures are not from this service.** They are from `docs/EVALUATION.md`, measured on 100
  questions with a model judge that has not been checked against human labels.
- **Never deployed to a cloud or used by real users.** It was proved with the fake provider, the Docker image and the
  local Kubernetes-free checks listed in the pull request, not under production traffic.
