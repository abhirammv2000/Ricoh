# Running, testing and deploying

## 10. How to run the project

### Prerequisites
- Python 3.11+ (developed on 3.13)
- Anthropic API key

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/abhirammv2000/Ricoh.git
cd Ricoh

# 2. Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env

# 5. Place Ricoh PDFs in data/
# (Copy all provided PDF manuals into the data/ directory)
```

### Run the Application
```bash
# Launch the Streamlit dashboard
streamlit run app/main.py
# Or equivalently:
python -m streamlit run app/main.py
```

### Run the Evaluation
```bash
# Quality harness: groundedness, correctness, recall@k, citation precision
python -m src.eval_harness            # full run, uses the LLM judge (needs API key)
python -m src.eval_harness --no-judge # objective metrics only, no API calls

# Legacy latency/citation smoke test
python -m src.evaluate                # outputs evaluation_results.csv + evaluation_report.md
```
Results and methodology are in [§7 Evaluation & Metrics](EVALUATION.md#7-evaluation-and-metrics).

### Run Individual Components
```bash
python -m src.ingest       # PDF ingestion only
python -m src.retriever    # Retrieval smoke test
python -m src.agent        # Agent smoke test
```

### Live public demo (Ngrok)

To share a live demo link:

```bash
# 1. Install Ngrok (https://ngrok.com/download)
# Or via Chocolatey on Windows:
choco install ngrok

# 2. Run Streamlit locally
streamlit run app/main.py

# 3. In a separate terminal, expose port 8501
ngrok http 8501

# 4. Share the generated https://xxxx.ngrok-free.app link
```


## 11. Testing and CI

```bash
pip install -r requirements-dev.txt
pytest                      # offline unit tests (LLM is mocked, no API key needed)
```

The suite covers the logic most likely to break silently:
- **Chunking invariants**, size cap, overlap preservation, page-provenance isolation, deterministic IDs (`tests/test_ingest.py`)
- **RRF fusion math**, rank merging, score accumulation, `final_k` truncation (`tests/test_retriever.py`)
- **Agent control flow**, retry routing, planner JSON parsing (incl. fenced/malformed output), verifier verdict normalisation (`tests/test_agent.py`)
- **Eval metrics**, citation extraction, recall@k, citation precision, refusal detection (`tests/test_eval_metrics.py`)

[GitHub Actions](../.github/workflows/ci.yml) runs `pytest` on every push/PR (Python 3.11). No secrets required. The tests mock the LLM.

A second CI job is a **retrieval regression gate** (`python -m eval.ci_gate`): it runs real hybrid retrieval against the committed `demo_index/` on the seed questions and fails the build if recall@1/3/5 drops below `eval/ci_baseline.json`. The unit suite never touches a real index, so this is what would catch an RRF or fusion bug that still passes every mocked test. The same job also checks that `demo_index` has not drifted from the benchmark it serves: if `eval/ground_truth.json` gains a question whose expected document is not baked into the index, the build fails with a "rebuild demo_index" message. `demo_index` is a small curated slice, so the recall check is a smoke gate, not a quality measurement; the full-corpus numbers come from the paid harness in [§7](EVALUATION.md#7-evaluation-and-metrics).

## 12. Optional: cross-encoder reranker

A query-aware cross-encoder reranker (off by default to keep the base torch-free) can be enabled for higher precision@k:

```bash
pip install -r requirements-reranker.txt
RERANKER_ENABLED=true streamlit run app/main.py
```

It fuses a larger candidate pool, re-scores with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and trims to the top-k. Lazy-loaded and cached, so the default lightweight path is untouched. Whether it earns its latency on this corpus is measured by `python -m eval.sweep_embeddings --measure --rerank` ([§7](EVALUATION.md#7-evaluation-and-metrics)).

## 13. Deployment

Beyond the local Ngrok demo, the project ships a reproducible container path. See **[DEPLOYMENT.md](../DEPLOYMENT.md)** for Docker, Render (one-click `render.yaml`), and Streamlit Cloud instructions.

```bash
docker build -t citera .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... -v "$PWD/data:/app/data" citera
```

## 14. Project status: demo vs production

Being straight about what this is:

**Built and working:** hybrid retrieval + RRF, agentic verify-retry loop, grounded/cited generation with refusal, multi-lingual answers, Glass Box UI with a per-request cost/attribution drill-down, a quality eval harness, a judged multi-turn eval and a judged multi-hop ablation, a cross-provider (Anthropic / OpenAI / Gemini) abstraction with a bakeoff, a unit-test suite + CI with a retrieval regression gate, per-request tracing (local JSONL plus opt-in LangSmith), an optional semantic answer cache, SDK-level retry/timeout handling, a rate limit and optional password on the public demo, and a containerised deploy path.

**Deliberately out of scope (next steps for true production):** real auth (not just a shared demo password), a secrets manager, sampling production traffic back into the eval set, human-labelled judge calibration (worksheet prepared), and a CI-built full-corpus index instead of the baked demo subset. These are tracked in [DEPLOYMENT.md](../DEPLOYMENT.md).
