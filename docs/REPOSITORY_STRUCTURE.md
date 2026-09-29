# Citera: repository structure

This used to be section 15 of the [README](../README.md).

```
Ricoh/
├── app/
│   └── main.py                  # Streamlit Glass Box dashboard
├── data/
│   └── *.pdf                    # Ricoh RPD docs, 733 PDFs (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py                # Centralised configuration
│   ├── ingest.py                # PDF parsing + chunking pipeline
│   ├── retriever.py             # Hybrid retrieval (ChromaDB + BM25 + RRF + optional reranker)
│   ├── azure_retriever.py       # Optional Azure AI Search backend (keyword, vector, hybrid, semantic)
│   ├── llm_factory.py           # LLM provider abstraction
│   ├── agent.py                 # LangGraph agentic state machine
│   ├── conversation.py          # History-aware follow-up rewriting for multi-turn
│   ├── router.py                # Cheap path, escalate to the tool loop on a refusal
│   ├── tools.py                 # Tool-calling retrieval loop (USE_TOOL_LOOP)
│   ├── guardrails.py            # Prompt-injection screen at the API edge
│   ├── instrumentation.py       # Per-stage cost / token / latency spans
│   ├── perf.py / trace_view.py  # Dashboard rollups and per-request drill-down
│   ├── semantic_cache.py        # Optional answer cache (off by default)
│   ├── evaluate.py              # Latency/citation smoke test
│   └── eval_harness.py          # Quality eval harness (evidence recall, retriever recall@N, groundedness)
├── eval/
│   ├── ground_truth.json        # Curated expected answers/sources
│   ├── generated_questions.json # The 100-question set, 70/30 dev/holdout
│   ├── metrics.json             # Harness output, default config (generated)
│   ├── eval_report_n100.md      # Harness output, 100 questions (generated)
│   ├── ablation.py              # Progressive-removal pipeline ablation
│   ├── ci_gate.py               # Retrieval regression gate (runs in CI vs demo_index)
│   ├── ragas_export.py          # Export a harness slice for the RAGAS cross-check
│   ├── ragas_eval.py            # RAGAS faithfulness vs our judge (separate venv)
│   ├── multihop_questions.json  # 20 hand-written two-document questions
│   ├── verify_multihop.py       # Confirms the multi-hop set stresses retrieval
│   ├── multiturn_questions.json # 12 conversation chains for the follow-up eval
│   ├── multiturn_eval.py        # Judged multi-turn evaluation
│   ├── provider_bakeoff.py      # Cross-provider synthesizer comparison
│   ├── run_paid_batch.sh        # The judged runs that need Anthropic credits
│   ├── sweep_embeddings.py      # Retrieval-only embedding-model comparison
│   ├── reranker_sweep.py        # Retrieval-only reranker-model comparison
│   ├── azure_search_eval.py     # Azure AI Search vs the local retriever, retrieval only
│   ├── calibrate_router.py      # Whether a retrieval signal can drive the router
│   ├── label_for_kappa.py       # Judge-vs-human agreement worksheet + scoring
│   ├── redteam_guardrail.py     # Measures the prompt-injection screen, not just unit-tests it
│   ├── redteam_prompts.json     # 40 adversarial + benign prompts, 4 categories
│   └── verify_unanswerable.py   # Audits the "refuse" labels against the full corpus
├── finetune/                     # QLoRA distillation of the synthesizer, self-hosted, see finetune/README.md
│   ├── data/                     # Leakage-safe training set + the docs it's drawn from
│   ├── scripts/generate_training_data.py
│   ├── train/                    # finetune_qlora.py, merge_adapter.py (run on a GPU VM)
│   ├── serve/serve_vllm.sh
│   └── infra/                    # GCP VM setup
├── tests/                       # ~25 pytest modules, offline, LLM mocked
│   ├── test_ingest.py test_retriever.py test_agent.py test_router.py
│   ├── test_conversation.py test_tools.py test_guardrails.py
│   ├── test_eval_metrics.py test_ci_gate.py test_perf.py ...
│   └── (one per src/ and eval/ module worth guarding)
├── docs/                        # Longer write-ups moved out of the README
│   ├── EVALUATION.md            # Evaluation and metrics (section 7)
│   ├── DESIGN.md                # Observability, data handling, modeling
│   ├── ROADMAP_STATUS.md        # Roadmap table
│   └── REPOSITORY_STRUCTURE.md  # This file
├── .github/workflows/ci.yml     # GitHub Actions CI
├── notebooks/                   # Exploration notebooks
├── chroma_db/                   # Persisted ChromaDB + BM25 index (gitignored)
├── Dockerfile                   # Container build
├── .dockerignore
├── render.yaml                  # Render.com one-click deploy blueprint
├── DEPLOYMENT.md                # Docker / Render / Streamlit Cloud guide
├── .env.example                 # Copy to .env and fill in
├── .gitignore
├── requirements.txt             # Runtime deps
├── requirements-dev.txt         # + pytest (CI)
├── requirements-reranker.txt    # Optional cross-encoder extra
├── requirements-azure.txt       # Optional Azure AI Search extra
├── requirements-enhanced.txt    # Optional stronger embeddings + reranker
├── LICENSE                      # MIT
├── evaluation_results.csv       # Smoke-test output
├── evaluation_report.md         # Smoke-test output
└── README.md                    # Overview and quick start
```
