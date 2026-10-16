# Making the API fit to run: decisions, evidence and what went wrong

This is the reasoning behind the work in `api/`, the container, and the checks around them. [OPERATIONS.md](OPERATIONS.md)
says how to run the service. This says why it is built the way it is, what each choice was measured against, and
what broke while I was building it.

**What "fit to run" means here, and what it does not.** The service is built so that when something fails it
fails safely and says so, and every claim about that is a test. Everything was proved against a fake model
provider, in the container, and in a local model experiment. It has not been deployed to a cloud, and it has not
served a real user. Nothing here says how it behaves under real traffic.

## Decisions, and what I rejected

**One model client for the whole process, not one per request.** The API built a new client on every request. A
profile of one request, with the provider made instant so only our own code was left, showed 276 ms of 330 ms spent
building it. At 10 concurrent requests that gave 4.3 questions a second and a 2.3 s median, against a provider that
took 0.3 s. One client per configuration gives 21.9 a second and a 0.41 s median. The cache is keyed on the
environment (so a changed key or endpoint gets a new client) and on the event loop, held weakly, because an async
client's connection pool belongs to one loop.

**An async stream for the default pipeline, not the graph's stream.** When a client closed the tab, the model
call ran to the end anyway and the money was spent for no one. A LangGraph node that is already running cannot be
cancelled. A plain async iterator over the model's stream can: cancelling the request closes the connection and the
provider stops. The other configurations (planner, verifier, tool loop, router, cache) keep the threaded path, and
that limit is written down.

**A deadline and a concurrency cap that refuses fast, not an unbounded queue.** With no cap, 30 simultaneous users
all waited 10 s. A request that cannot get a slot in `QUEUE_TIMEOUT_SECONDS` gets a 503 with `Retry-After`, so the
work that is admitted stays quick and the caller learns at once. The request deadline cancels the provider call.
The SDK's own timeout and retries are set from the environment and checked against the deadline at startup,
because a 529 retried for 3 s inside a 3 s deadline gave the wrong error (a 504 instead of a 503).

**Retrieval alone when the model fails, not an error.** Retrieval does not depend on the model, so when the
provider is down, slow, over budget or returns an answer that breaks the citation contract, the caller gets the
passages, marked `degraded`, with the reason in a header. It is not used for errors that passages would not fix
(a full service, our own bad key). A normal response is unchanged, so existing callers see nothing new.

**A manifest and checksums for the index, not trusting the folder.** Chroma rejects a query vector of another size
but MiniLM, bge-small and e5-small are all 384 wide, so a changed embedding model would run and retrieve worse
without any error. `build_index` also upserted into Chroma but rebuilt BM25 from only the chunks it was given, so a
partial re-ingest left them disagreeing. And the BM25 files are pickles, which run code when loaded. The manifest
records the model, chunking, counts and a checksum of each file. A changed model, a partial ingest or a replaced
file is refused, and the pickle check runs before the pickle is loaded (a test proves a malicious pickle never
runs). `/ready` is separate from `/health` so a bad index stops traffic without making an orchestrator restart the
process forever, and `/query` will not call the model with no usable index, which was the "healthy but refuses every
question" failure the old deployment notes described.

**Authentication on by default, limits by key, a daily budget.** The service refuses to start with no keys unless
`API_AUTH_DISABLED=true`, so it cannot be left open by forgetting a variable. A caller's rate limit follows their
key, because behind a proxy every caller shares one address. Wrong keys are throttled per address without
affecting callers who have a valid one. A rate limit bounds how fast money goes out and a budget bounds how much,
which matters for a looping client or a leaked key. The budget is per process: with N containers the ceiling is N times
the setting, and the docs say so.

**The question text stays out of logs, and traces have a mode.** People paste serial numbers and names into a support
box. The retriever logged the first 80 characters of every question, and every trace stored the full text.
Logs now record a length and a short hash, and `TRACE_QUERY_MODE` chooses full, hash or none. The container
defaults to hash.

**An output check, not a prompt change, as the second injection defense.** Changing the prompt would change the
answers whose quality is reported in [EVALUATION.md](EVALUATION.md), and re-measuring them costs money. A check on
the output (cite a source or decline) changes no clean answer, and it was run on real answers first (10 committed, 50 more
in a local export): none would have been withheld. It catches one of the two attacks that worked in the local-model experiment, and it
cannot catch the other (a refusal with a word added). That is written down in a test.

**One process and many containers, not many workers.** The time is the model's, so extra worker processes add
memory and split the limits without adding capacity. Capacity comes from more containers.

**A fake model provider, not mocks of the Python functions.** `ANTHROPIC_BASE_URL` points the real client at any
server, so a small fake that can return 429, 529, hang or stall mid-stream drives the real client's own retry and
timeout code. That is how the 529-versus-deadline interaction and the cancelled stream were found.

## What went wrong while building it

- **A failing assertion printed an API key.** A test compared the key from `.env` with a test value, and pytest prints the
  values a failed `assert` compared. The guard now reduces each check to a plain true or false first, so a failure
  cannot show a value. Anything that compares secrets in a test needs the same care.
- **Tests I had broken nearly spent money.** Once my async path took over, three tests that mocked the old path made
  real requests, because `src/config.py` loads `.env` and `override=True` made it beat everything. They were
  rejected by the API, so they likely cost nothing. The suite now sets a test key, points the client at a closed local
  port and turns tracing off before anything imports the code, and a test checks that.
- **`load_dotenv(override=True)` was wrong for a service.** The real environment (the platform, a secrets manager) should
  win over `.env`. It now does, with `DOTENV_OVERRIDE=1` to get the old behavior on a laptop.
- **A cache keyed on `id(loop)` handed a new event loop the old loop's client.** A closed loop's memory can be reused.
  It passed alone and failed in the full suite, which is how it was caught. It is now keyed on the loop object, held weakly.
- **A metric counted twice.** A failed `/query` raised an error that was counted where it was raised and again in its
  handler. A test of the metric caught it.
- **Six processes writing the trace file lost 85 of 240 lines on Windows.** The C runtime fakes append with a seek and a
  write. Appends are atomic on Linux, where the service runs, so the code and docstring say that, the test is skipped
  on Windows with the number in its reason, and it runs on Linux in CI.
- **My load test measured itself.** Refusals seemed to take a second. The server's own log showed they were refused 0.1 s
  after arriving; `httpx.post()` was building a new client, and loading TLS certificates, on every call. Fixed by sharing
  one client. The earlier before-and-after numbers were from a tool that creates its client once, so they were not affected.
- **The first warm-up did not remove the slow first request.** I had assumed retrieval was the cost. A profile showed
  2.3 s of it was importing the Anthropic SDK. Warming up the SDK and the client brought the first question from 2.66 s
  to 0.18 s, for about 5 s more at startup.
- **The injection attack mostly did not work.** Against this prompt and a 7B model, the recognised attacks were followed
  0 times in 30 even with no defense. The one that worked was a paraphrase the pattern screen cannot see. So the pattern
  screen is defense in depth and not the defense, and the report says so.
- **A smoke test caught a stale image.** The container passed everything except the newest feature, because I had built the
  image before adding it. Rebuilding fixed it, which is the point of testing the image and not the source.

## Not done

- No dashboards, and the alert thresholds are guesses made before there was traffic.
- Limits and the budget are per process. A shared store is the next step for more than one container.
- A changed document needs a manual rebuild and restart.
- The answer-quality figures come from [EVALUATION.md](EVALUATION.md). Their judge has not been checked against human
  labels (0 of 30 labelled), and the 100 questions were generated from the documents.
- The model was not compared against other models under the new defenses, and the production model was not used in
  the injection experiment, because both cost money.
