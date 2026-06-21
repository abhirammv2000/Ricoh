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

# ── Force UTF-8 stdout/stderr ──────────────────────────────────────
# On Windows the console/redirected streams default to cp1252, which
# crashes on the emoji used in our log/print output (e.g. when piping
# to a file).  Reconfiguring to UTF-8 makes every entrypoint (app,
# evaluate, eval_harness, agent smoke test) robust across platforms.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# ── Silence ChromaDB telemetry BEFORE any chromadb import ──
# This MUST run before chromadb is ever imported in any module.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv

# ── Load .env (if present) so API keys are available via os.getenv ──
load_dotenv()

# ── Centralised logging configuration ─────────────────────────────
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

# ── Project root = parent of the `src/` package directory ──
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Paths ──
# DATA_DIR can be overridden via the RICOH_DATA_DIR env var so the
# ingestion source is never hardcoded-brittle (e.g. pointing at an
# external export folder).  Defaults to the in-repo ``data/`` dir.
DATA_DIR: Path = Path(os.getenv("RICOH_DATA_DIR", PROJECT_ROOT / "data"))  # Raw Ricoh PDFs go here
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"   # Local ChromaDB persistence
BM25_INDEX_PATH: Path = PROJECT_ROOT / "chroma_db" / "bm25_index.pkl"
BM25_CHUNKS_PATH: Path = PROJECT_ROOT / "chroma_db" / "bm25_chunks.pkl"

# ── ChromaDB settings ──
CHROMA_COLLECTION_NAME: str = "ricoh_manuals"

# ── Embedding model ────────────────────────────────────────────────
# Empty string ("") = ChromaDB's built-in default: all-MiniLM-L6-v2 via
# onnxruntime (fast, offline, NO torch).  Set EMBEDDING_MODEL to a
# sentence-transformers model id (e.g. "BAAI/bge-small-en-v1.5") for
# materially stronger retrieval — this pulls in torch + sentence-
# transformers (heavy), so it is opt-in:
#   pip install -r requirements-enhanced.txt
#   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 python -m src.ingest
# Changing this requires REBUILDING the index (vectors are model-specific):
# delete chroma_db/ and re-ingest.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "")

# ── Chunking hyper-parameters ──────────────────────────────────────
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

# ── Supported file types for ingestion ──
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf",)

# ── Retrieval hyper-parameters ─────────────────────────────────────
# RETRIEVAL_TOP_K: how many candidates each method (vector / BM25)
#   returns before fusion.  More candidates → better recall but
#   slower.  10 is a solid default for ~1 000–10 000 chunks.
RETRIEVAL_TOP_K: int = 10

# RETRIEVAL_FINAL_K: how many fused results the agent actually sees.
#   Keeping this small respects the LLM context window and forces
#   the retriever to surface only the most relevant evidence.
RETRIEVAL_FINAL_K: int = 5

# RRF_K: Reciprocal Rank Fusion smoothing constant.  The standard
#   value from the original Cormack et al. paper is 60.  Lowering it
#   amplifies the difference between ranks; raising it flattens it.
RRF_K: int = 60

# ── Optional cross-encoder reranker ────────────────────────────────
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
# RETRIEVAL_FINAL_K.  More candidates → better reranker headroom.
RERANK_CANDIDATE_POOL: int = 20

# ── LLM provider (overridden at runtime / via .env) ──
# Accepted values: "anthropic" | "google"
DEFAULT_LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
