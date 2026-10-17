# Citera: an agentic RAG system for technical support

Citera answers questions about RICOH ProcessDirector using 733 help documents. Every answer has page-level citations. When the documents don't contain the answer, it says so instead of guessing.

It started as a group course project. Jayan Agarwal built the first version: ingestion, hybrid retrieval and the first agent loop. I rewrote much of the retrieval and agent code and added the evaluation harness, the ablation, the router and tool calling, per-request tracing, the API and the deployment path. `git log` shows who did what.

## What I found

I built a plan, retrieve, verify and retry pipeline and then measured it. One retrieval call turned out to be the best default, so the planner and verifier are off unless you switch them on.

Scores on 100 questions (70 dev, 30 held out). `claude-opus-5` judged `claude-sonnet-4-6` over the full corpus. Brackets are 95% bootstrap intervals.

| Metric | Score |
|---|---|
| Groundedness | 0.96 [0.95-0.98] |
| Correctness | 0.97 [0.94-0.99] |
| Citation precision | 1.00 |
| Answer vs refuse | 0.98 |
| Retrieval recall@5 | 0.94 |

Cost is $0.0157 and 10.1 s per query, with 1 LLM call.

The ablation, on the 70 dev questions:

| Config | Pipeline | Calls | Cost/query | Evidence recall | Correct |
|---|---|---|---|---|---|
| A | retrieve, synthesize | 1.0 | $0.0156 | 0.94 | 0.97 |
| B | A + planner | 2.0 | $0.0262 | 1.00 | 0.99 |
| C | B + verifier and retry | 3.0 | $0.0414 | 1.00 | 0.99 |

- The verifier adds nothing, so it stays off.
- The planner raises evidence recall from 0.94 to 1.00 on the dev questions. On the 30 held-out questions A and B both reach 0.93 evidence recall, so the gain doesn't carry over. It also costs 1.7x per query, so it stays behind `USE_PLANNER`.
- An earlier 10-question set gave flattering numbers. Growing it to 100 moved groundedness from 0.98 to 0.96.
- A cross-provider test on 20 questions found `gemini-3.6-flash` close to Sonnet on quality (0.95 against 0.99 correctness) at 1/50th the cost per query, while `gpt-4o-mini` loses 0.21 correctness.
- On the same 20 questions, a QLoRA fine-tune of Llama 3.1 8B on one GPU scored 0.94 correctness against Sonnet's 0.99. The judge's noise is about 0.10, so that gap is inside it and I cannot say they differ. Groundedness is lower (0.91 against 0.98) and latency is 25 s against 10 s.
- An optional Azure AI Search backend matches the local reranker on recall@5 and is much faster.

Everything is written up in [docs/EVALUATION.md](docs/EVALUATION.md), including the corrections I made to my own earlier results.

## About the data

The 733 PDFs are RICOH ProcessDirector's public help documents. They are not in this repository (`data/` is gitignored). Two things in the repository are derived from them: `demo_index/`, a small slice of text chunks that lets CI and the demo run, and `finetune/data/`, the examples used for the fine-tune, which include retrieved passages. That text belongs to Ricoh and is used here for a student project.

## How it works

```
question -> (rewrite if it is a follow-up) -> retrieve -> synthesize -> cited answer
                                               ChromaDB + BM25, merged with RRF
```

- **Ingest:** PyMuPDF reads the PDFs and keeps document and page. Text is cut into 500-word chunks with 50 words of overlap. 31 pages with screenshots also get a short description from a vision model.
- **Retrieve:** vector search (ChromaDB and MiniLM) and keyword search (BM25), merged with Reciprocal Rank Fusion. Keyword search catches exact error codes like `SC542`. An optional cross-encoder reranker is available.
- **Answer:** a LangGraph pipeline calls Claude, which must cite `[Document, Page]` for every claim. It answers in the language of the question.
- **MCP:** `python -m src.mcp_server` serves `search_docs` and `index_info` read-only over stdio, so Claude Desktop or any MCP client can search the documentation with page citations. It never calls a language model. I checked it through a real MCP client over stdio against the real index (1,322 passages).
- **UI:** a Streamlit dashboard that shows every step, the cost and the chunks used. Each request is traced to a JSONL file.

## Quick start

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-your-key" > .env     # then put the PDFs in data/
streamlit run app/main.py
```

To run it as a service instead of the demo page:

```bash
docker build -f Dockerfile.api -t citera-api .
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-your-key -e API_KEYS=me:a-long-random-secret citera-api
curl -H "X-API-Key: a-long-random-secret" -d '{"query": "How do I configure the network settings?"}' \
     -H 'Content-Type: application/json' http://localhost:8000/query
```

[docs/OPERATIONS.md](docs/OPERATIONS.md) has the settings, the endpoints, the alerts and what to do when each fires.

Run the tests with `pip install -r requirements-dev.txt` and `pytest`. They run offline, the LLM is mocked, and a guard makes the suite unable to reach a paid API even by mistake. CI also runs a retrieval regression check against a small demo index, starts the container against a fake model provider and runs the failure drills. More commands, the evaluation scripts and deployment steps are in [docs/RUNNING.md](docs/RUNNING.md).

## Docs

- [docs/OVERVIEW.md](docs/OVERVIEW.md): the problem, the design and the tech stack
- [docs/DESIGN.md](docs/DESIGN.md): data handling and modeling choices
- [docs/EVALUATION.md](docs/EVALUATION.md): all the measurements
- [docs/RUNNING.md](docs/RUNNING.md): running, testing, deploying
- [docs/OPERATIONS.md](docs/OPERATIONS.md): running the API as a service, its settings, alerts and runbook
- [docs/PRODUCTION_DESIGN.md](docs/PRODUCTION_DESIGN.md): why the API is built the way it is, what each choice was measured against, and what went wrong
- [docs/ROADMAP_STATUS.md](docs/ROADMAP_STATUS.md) and [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Limits

- It needs the PDFs ingested up front. It doesn't pick up document changes on its own.
- An answer takes about 10 to 15 seconds, which is too slow for live phone support.
- Declining is barely measured. The 100 generated questions are all answerable, and only 2 curated questions have no answer in the documents.
- The judge is a Claude model. A second Claude model agreed with it on 30 answers, but it has not been checked against human labels.
- The Streamlit page is a demo. It uses a shared password and has no rate limit.
- The API in `api/` is built to run as a service: keys, per-key limits, a deadline, a concurrency cap, a daily spend cap, a readiness check, metrics, and answers from retrieval alone when the model is down. It has been tested against a fake model provider and in its container. It has not served real users, and it has not been deployed to a cloud. Its rate limits, concurrency cap and budget are per process.
- [DEPLOYMENT.md](DEPLOYMENT.md) lists what is still missing, and [docs/OPERATIONS.md](docs/OPERATIONS.md) says how the service behaves when things go wrong.
