"""
src/config.py — Centralised configuration for RicohLibrary.

All tuneable parameters live here so they can be adjusted in ONE
place.  We use python-dotenv to load any secrets (e.g. API keys)
from a `.env` file at the project root.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env (if present) so API keys are available via os.getenv ──
load_dotenv()

# ── Project root = parent of the `src/` package directory ──
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Paths ──
DATA_DIR: Path = PROJECT_ROOT / "data"          # Raw Ricoh PDFs go here
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"   # Local ChromaDB persistence

# ── ChromaDB settings ──
CHROMA_COLLECTION_NAME: str = "ricoh_manuals"

# ── Chunking hyper-parameters ──────────────────────────────────────
# We approximate "tokens" as whitespace-delimited words (~1.3 tokens
# per word on average for English text).  Using word count is simpler
# and deterministic — no tokeniser dependency at ingest time.
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

# ── LLM provider (overridden at runtime / via .env) ──
# Accepted values: "anthropic" | "google"
DEFAULT_LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google")
