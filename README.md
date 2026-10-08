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
- The planner raises evidence recall from 0.94 to 1.00 on the dev questions. On the 30 held-out questions A and B both score 0.93, so the gain doesn't carry over. It also costs 1.7x per query, so it stays behind `USE_PLANNER`.
- An earlier 10-question set gave flattering numbers. Growing it to 100 moved groundedness from 0.98 to 0.96.
- A cross-provider test found `gemini-3.6-flash` keeps answer quality at 1/50th the cost per query, while `gpt-4o-mini` loses 0.21 correctness.
- A QLoRA fine-tune of Llama 3.1 8B on one GPU matched Sonnet's correctness within the noise, but groundedness is still worse.
- An optional Azure AI Search backend matches the local reranker on recall@5 and is much faster.

Everything is written up in [docs/EVALUATION.md](docs/EVALUATION.md), including the corrections I made to my own earlier results.

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

Run the tests with `pip install -r requirements-dev.txt` and `pytest`. They run offline and the LLM is mocked. CI also runs a retrieval regression check against a small demo index. More commands, the evaluation scripts and deployment steps are in [docs/RUNNING.md](docs/RUNNING.md).

## Docs

- [docs/OVERVIEW.md](docs/OVERVIEW.md): the problem, the design and the tech stack
- [docs/DESIGN.md](docs/DESIGN.md): data handling and modeling choices
- [docs/EVALUATION.md](docs/EVALUATION.md): all the measurements
- [docs/RUNNING.md](docs/RUNNING.md): running, testing, deploying
- [docs/ROADMAP_STATUS.md](docs/ROADMAP_STATUS.md) and [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)

## Limits

- It needs the PDFs ingested up front. It doesn't pick up document changes on its own.
- An answer takes about 10 to 15 seconds, which is too slow for live phone support.
- This is a demo, not a production system. The public demo uses a shared password, and there is no real auth or secrets manager. [DEPLOYMENT.md](DEPLOYMENT.md) lists what is missing.
