"""Every setting the code reads from the environment is listed in .env.example, so nobody has to read the code."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# read on purpose but not listed, with the reason
UNLISTED = {
    "ANONYMIZED_TELEMETRY": "set by the code to turn chromadb telemetry off, not read",
    "LANGCHAIN_TRACING_V2": "the old name for LANGSMITH_TRACING",
    "LANGCHAIN_API_KEY": "the old name for LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT": "the old name for LANGCHAIN_PROJECT",
    "GOOGLE_API_KEY": "an alias for GEMINI_API_KEY, mentioned beside it",
}

READ_CALLS = re.compile(
    r"""(?:getenv|\.get|env_number|env_flag|_number|_choice|_flag)\(\s*(?:env,\s*)?["']([A-Z][A-Z0-9_]{3,})["']"""
)
LISTED = re.compile(r"^#?\s*([A-Z][A-Z0-9_]{3,})=", re.MULTILINE)


def names_read_by_the_code() -> set[str]:
    names: set[str] = set()
    for folder in ("src", "api", "app"):
        for path in (ROOT / folder).glob("*.py"):
            names |= set(READ_CALLS.findall(path.read_text(encoding="utf-8")))
    return names


def test_every_setting_the_code_reads_is_listed_in_env_example():
    listed = set(LISTED.findall((ROOT / ".env.example").read_text(encoding="utf-8")))
    missing = sorted(names_read_by_the_code() - listed - set(UNLISTED))
    assert not missing, f"add these to .env.example (or to UNLISTED with a reason): {missing}"


def test_env_example_lists_nothing_the_code_no_longer_reads():
    listed = set(LISTED.findall((ROOT / ".env.example").read_text(encoding="utf-8")))
    read = names_read_by_the_code()
    stale = sorted(listed - read)
    assert not stale, f"these are in .env.example but the code never reads them: {stale}"
