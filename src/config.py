"""
src/config.py - Centralised configuration for Citera.

All tuneable parameters live here so they can be adjusted in ONE
place.  We use python-dotenv to load any secrets (e.g. API keys)
from a `.env` file at the project root.
"""

import logging
import os
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr
# On Windows the console/redirected streams default to cp1252, which
# crashes on the emoji used in our log/print output (e.g. when piping
# to a file).  Reconfiguring to UTF-8 makes every entrypoint (app,
# evaluate, eval_harness, agent smoke test) robust across platforms.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# Silence ChromaDB telemetry BEFORE any chromadb import
# This MUST run before chromadb is ever imported in any module.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv

# Load .env (if present) so API keys are available via os.getenv
load_dotenv()

# Centralised logging configuration
# We configure logging ONCE here.  All modules use
# logging.getLogger(__name__) so this controls everything.
#
# Library loggers (chromadb, httpx, etc.) are forced to WARNING
# so only our agent "thoughts" and critical errors print.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
# Suppress noisy library loggers
for _noisy in (
    "chromadb",
    "chromadb.telemetry",
    "chromadb.telemetry.product.posthog",
    "httpx",
    "httpcore",
    "openai",
    "anthropic",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# Optional LangSmith tracing, opt-in and off by default.
# LangChain traces the LangGraph agent to LangSmith automatically when
# LANGSMITH_TRACING is truthy and an API key is present. Keeping it opt-in means
# the repo stays self-contained: nothing here needs a LangSmith account to run,
# and the file-based traces in traces/ work with or without it. When a key is
# configured this adds a hosted timeline of every run on top of those.
def _configure_langsmith() -> None:
    flag = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).lower()
    if flag not in ("1", "true", "yes"):
        return
    # Set both the current and the legacy variable names so any langchain-core
    # version picks the tracer up.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "citera"
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGSMITH_PROJECT"] = project
    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        logging.getLogger(__name__).info("LangSmith tracing enabled, project '%s'", project)
    else:
        logging.getLogger(__name__).warning(
            "LANGSMITH_TRACING is on but no LANGSMITH_API_KEY is set, so traces will not be sent."
        )


_configure_langsmith()


# Project root = parent of the `src/` package directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Paths
# DATA_DIR can be overridden via the RICOH_DATA_DIR env var so the
# ingestion source is never hardcoded-brittle (e.g. pointing at an
# external export folder).  Defaults to the in-repo ``data/`` dir.
DATA_DIR: Path = Path(os.getenv("RICOH_DATA_DIR", PROJECT_ROOT / "data"))  # Raw Ricoh PDFs go here
# CHROMA_DIR is overridable so a deployment can ship a pre-built index instead
# of ingesting PDFs on boot. The container has no data/ directory, so building
# at startup is not an option there, it would produce an empty index and the
# app would refuse every question while looking healthy.
CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
BM25_INDEX_PATH: Path = CHROMA_DIR / "bm25_index.pkl"
BM25_CHUNKS_PATH: Path = CHROMA_DIR / "bm25_chunks.pkl"

# Set when serving a reduced corpus so the UI can say so. A demo that quietly
# answers from 40 documents while the README reports metrics measured on 733
# would be misleading about what the live system is.
DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

# ChromaDB settings
CHROMA_COLLECTION_NAME: str = "ricoh_manuals"

# Embedding model
# Empty string ("") = ChromaDB's built-in default: all-MiniLM-L6-v2 via
# onnxruntime (fast, offline, NO torch).  Set EMBEDDING_MODEL to a
# sentence-transformers model id (e.g. "BAAI/bge-small-en-v1.5") for
# materially stronger retrieval, this pulls in torch + sentence-
# transformers (heavy), so it is opt-in:
#   pip install -r requirements-enhanced.txt
#   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 python -m src.ingest
# Changing this requires REBUILDING the index (vectors are model-specific):
# delete chroma_db/ and re-ingest.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "")

# Chunking hyper-parameters
# We approximate "tokens" as whitespace-delimited words (~1.3 tokens
# per word on average for English text).  Using word count is simpler
# and deterministic - no tokeniser dependency at ingest time.
#
# 500 words ≈ 650 tokens ≈ 2 000 characters.  This keeps each chunk
# small enough for the LLM context window while large enough to hold
# a coherent paragraph from a technical manual.
#
# 50-word overlap ensures sentence-boundary context is not lost when
# a question's answer straddles two chunks.
CHUNK_SIZE: int = 500        # words per chunk
CHUNK_OVERLAP: int = 50      # words of overlap between consecutive chunks

# Supported file types for ingestion
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf",)

# Retrieval hyper-parameters
# RETRIEVAL_TOP_K: how many candidates each method (vector / BM25)
#   returns before fusion.  More candidates -> better recall but
#   slower.  10 is a solid default for ~1 000-10 000 chunks.
RETRIEVAL_TOP_K: int = 10

# RETRIEVAL_FINAL_K: how many fused results the agent actually sees.
#   Keeping this small respects the LLM context window and forces
#   the retriever to surface only the most relevant evidence.
RETRIEVAL_FINAL_K: int = 5

# RRF_K: Reciprocal Rank Fusion smoothing constant.  The standard
#   value from the original Cormack et al. paper is 60.  Lowering it
#   amplifies the difference between ranks; raising it flattens it.
RRF_K: int = 60

# Optional cross-encoder reranker
# A reranker re-scores the fused candidates with a query-aware
# cross-encoder, which materially improves precision@k over RRF alone.
# It is OFF by default because it pulls in torch + sentence-transformers
# (a heavy dependency) and the base system is intentionally lightweight
# and torch-free.  Enable with RERANKER_ENABLED=true and install the
# extra:  pip install -r requirements-reranker.txt
RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# When reranking, fuse a larger candidate pool then rerank down to
# RETRIEVAL_FINAL_K.  More candidates -> better reranker headroom.
RERANK_CANDIDATE_POOL: int = 20

# LLM provider (overridden at runtime / via .env)
# Accepted values: "anthropic" | "google"
DEFAULT_LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")

# Pipeline composition
# Which agentic stages run in production.
#
# Both default to FALSE as of 2026-08-01, on evidence.  A three-config
# progressive-removal ablation (eval/ablation.py, results in
# eval/ablation/comparison.json) compared:
#
#   A  retrieve -> synthesize           1.0 calls  $0.0159/q   9.9s
#   B  + planner                        2.0 calls  $0.0207/q  13.9s
#   C  + verifier/retry  (old default)  3.4 calls  $0.0490/q  15.6s
#
# Config A matched or beat the full pipeline on every quality metric ,
# evidence recall 1.00 vs 0.88, correctness 1.00 vs 0.98, groundedness and
# behaviour-match tied within judge noise, while costing 3.1x less and
# running 1.6x faster.  No question was measurably better under C.
#
# Notably, behaviour-match stays 1.00 without the verifier: the synthesizer
# prompt's own refusal rule already handles insufficient evidence, so the
# verifier was duplicating work the synthesizer does anyway.
#
# Scope of that claim: it holds for THIS corpus (733 mostly single-page
# help articles, where a single retrieval already achieves recall@5 = 1.00)
# and THIS 10-question benchmark, which contains no genuinely multi-hop
# questions.  On a corpus where retrieval is weak, the planner's ability to
# re-query is exactly the mechanism that would earn its cost, which is why
# the stages are disabled by config rather than deleted from the codebase.
USE_PLANNER: bool = os.getenv("USE_PLANNER", "false").lower() in ("1", "true", "yes")
USE_VERIFIER: bool = os.getenv("USE_VERIFIER", "false").lower() in ("1", "true", "yes")

# Tool-calling retrieval (opt-in, off by default)
# When on, the agent exposes search_docs as a tool and the model runs its own
# searches, re-querying when the passages it gets back do not answer the
# question. This is the re-query mechanism the ablation section predicted would
# earn its cost only on a corpus with weak retrieval. It earns it here too:
# of the 6 questions that fail retrieval at n=100, the model recovered 4, and it
# lost none of a 20-question sample that already passed. Cost is 1.77x
# ($0.0278 vs $0.0157 per query).
#
# Off by default regardless, because that measurement is retrieval-only. Whether
# the recovered evidence actually improves judged groundedness and correctness
# needs the full n=100 judged run, and changing a default on unjudged evidence
# is the kind of move the rest of this project exists to argue against.
USE_TOOL_LOOP: bool = os.getenv("USE_TOOL_LOOP", "false").lower() in ("1", "true", "yes")

# Semantic answer cache (opt-in, off by default)
# When on, run_agent returns a stored answer for an identical or near-duplicate
# query instead of paying for the synthesizer again. Serving a cached answer for
# a merely similar question would be a correctness bug in a system whose whole
# value is grounded accuracy, so the threshold is set from measurement, not
# guessed: on this corpus, MiniLM scores genuine paraphrases at 0.87 to 0.91 and
# different questions at 0.48 or below, so 0.90 catches near-duplicate phrasings
# with a wide safety margin above anything a different question scored. The eval
# harness bypasses the cache entirely (it invokes the graph directly), so a hit
# can never corrupt a measured number.
SEMANTIC_CACHE_ENABLED: bool = os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() in ("1", "true", "yes")
SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90"))
SEMANTIC_CACHE_MAX_ENTRIES: int = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "512"))

# Evaluation judge
# The LLM-as-judge MUST NOT be the same model as the one generating the
# answers.  A model scoring its own output exhibits self-preference bias,
# which makes groundedness/correctness scores uninterpretable, you cannot
# tell a genuinely grounded answer from one the judge simply likes because
# it wrote it.  Published evaluations of LLM judges also find that raw
# agreement badly overstates chance-corrected agreement, so a single judge
# grading itself is the weakest possible configuration.
#
# We therefore default the judge to a *stronger, different* model than the
# agent (which runs Sonnet).  Override with JUDGE_MODEL to run a second
# judge and compare, cross-judge disagreement is itself a useful signal.
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "claude-opus-5")

# Judge output is a small JSON object, but on models with thinking enabled
# by default max_tokens covers thinking + text, so leave real headroom.
JUDGE_MAX_TOKENS: int = int(os.getenv("JUDGE_MAX_TOKENS", "8192"))
