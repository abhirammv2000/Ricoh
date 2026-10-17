"""Settings for Citera. Anything you might want to change lives here, and secrets come from .env."""

import logging
import os
import sys
from pathlib import Path

# windows pipes default to cp1252 and crash on the emoji in our log output
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# has to be set before chromadb is imported anywhere
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv


def env_flag(name: str) -> bool:
    """True when the variable is 1, true or yes, in any case. Unset, or anything else, is False."""
    return os.getenv(name, "").lower() in ("1", "true", "yes")


def env_number(name: str, default: float, cast=float):
    """A number from the environment, or the default. A bad or negative value stops startup and names itself."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {raw!r}")
    return value


# the real environment beats .env. A deployed service gets its settings from the platform, and .env is only a
# local default. If an old key in your shell is hiding the one in .env, set DOTENV_OVERRIDE=true.
load_dotenv(override=env_flag("DOTENV_OVERRIDE"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
for _noisy in (
    "chromadb",
    "chromadb.telemetry",
    "httpx",
    "httpcore",
    "openai",
    "anthropic",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# chromadb 0.6.3 logs a harmless posthog "capture() takes 1 positional argument" error on
# every call, even with telemetry off. Not ours to fix, so mute it.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


# LangSmith tracing is opt-in. Nothing needs a LangSmith account, and the file traces in
# traces/ work either way.
def _configure_langsmith() -> None:
    flag = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).lower()
    if flag not in ("1", "true", "yes"):
        return
    # set the new and the old variable names so any langchain-core version picks it up
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


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# paths
DATA_DIR: Path = Path(os.getenv("RICOH_DATA_DIR", PROJECT_ROOT / "data"))  # raw Ricoh PDFs go here
# can be overridden so a deployment can ship a prebuilt index (the container has no data/ folder)
CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
BM25_INDEX_PATH: Path = CHROMA_DIR / "bm25_index.pkl"
BM25_CHUNKS_PATH: Path = CHROMA_DIR / "bm25_chunks.pkl"

# set when serving a smaller corpus, so the UI can say so
DEMO_MODE: bool = env_flag("DEMO_MODE")

CHROMA_COLLECTION_NAME: str = "ricoh_manuals"

# empty means chromadb's default all-MiniLM-L6-v2 on onnxruntime (no torch). A different model
# needs the extra requirements and a fresh index: delete chroma_db/ and re-ingest.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "")

# chunk size is counted in words, not tokens, so ingest needs no tokenizer
CHUNK_SIZE: int = 500        # words per chunk
CHUNK_OVERLAP: int = 50      # words shared between neighbouring chunks

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf",)

# candidates each of vector and BM25 returns before fusion
RETRIEVAL_TOP_K: int = 10

# how many fused results the agent actually sees
RETRIEVAL_FINAL_K: int = 5

# smoothing constant for reciprocal rank fusion, 60 is the value from the original paper
RRF_K: int = 60

# cross-encoder reranker, off by default because it needs torch
# (pip install -r requirements-reranker.txt)
RERANKER_ENABLED: bool = env_flag("RERANKER_ENABLED")
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# with the reranker on, fuse this many candidates and rerank them down to RETRIEVAL_FINAL_K
RERANK_CANDIDATE_POOL: int = 20

# optional Azure AI Search backend, only read if src/azure_retriever.py is used
AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_API_KEY: str = os.getenv("AZURE_SEARCH_API_KEY", "")
AZURE_SEARCH_INDEX: str = os.getenv("AZURE_SEARCH_INDEX", "citera-chunks")

# anthropic, openai, google or self_hosted
DEFAULT_LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")

# planner and verifier are both off. The verifier never helped in the ablation. The planner helped
# on dev but not on holdout, which isn't worth 1.7x the cost per query (see docs/EVALUATION.md).
USE_PLANNER: bool = env_flag("USE_PLANNER")
USE_VERIFIER: bool = env_flag("USE_VERIFIER")

# lets the model run its own searches and re-query when the first ones don't answer. It recovered
# 4 of the 6 retrieval misses at n=100 for 1.77x the cost, but that was retrieval only, so it stays
# off until the judged run is done.
USE_TOOL_LOOP: bool = env_flag("USE_TOOL_LOOP")

# runs the cheap path and only escalates to the tool loop when the synthesizer refuses (src/router.py).
# Takes precedence over the flags above. Off because the gain was inside judge noise.
USE_ROUTER: bool = env_flag("USE_ROUTER")

# return a stored answer for a near-duplicate question. The 0.90 threshold comes from measuring
# MiniLM on this corpus: paraphrases scored 0.87 to 0.91, different questions 0.48 or lower.
# The eval harness skips the cache, so a hit can't change a measured number.
SEMANTIC_CACHE_ENABLED: bool = env_flag("SEMANTIC_CACHE_ENABLED")
SEMANTIC_CACHE_THRESHOLD: float = env_number("SEMANTIC_CACHE_THRESHOLD", 0.90)
SEMANTIC_CACHE_MAX_ENTRIES: int = int(env_number("SEMANTIC_CACHE_MAX_ENTRIES", 512, int))

# the judge is a different, stronger model than the agent (which runs Sonnet), so it isn't
# grading its own writing. Set JUDGE_MODEL to compare a second judge.
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "claude-opus-5")

# max_tokens covers thinking plus text on models that think by default, so leave room
JUDGE_MAX_TOKENS: int = int(env_number("JUDGE_MAX_TOKENS", 8192, int))
