"""Is this instance fit to answer questions?

Two endpoints answer two different questions.

* /health: is the process alive? It is cheap and never looks at the index. If it failed because the index was bad, an
  orchestrator would restart the process in a loop and the index would still be bad.
* /ready: can it give a correct answer? The model provider is set up, the index opens, is not empty, and matches its
  manifest and the current settings. If not, no traffic is routed here and /query refuses with a 503 instead of calling
  the model with nothing to cite.

The check never calls the model provider, so it costs nothing and a probe can run it as often as it likes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.index_manifest import IndexIntegrityError
from src.llm_factory import missing_provider_settings

log = logging.getLogger("citera.api")

# how long a failed check is trusted before it is run again, so a fixed index is noticed without a restart
# and a broken one is not re-hashed on every probe
RECHECK_AFTER_FAILURE_SECONDS = 10.0


@dataclass
class Readiness:
    problems: list[str] = field(default_factory=list)
    index_version: str = "unknown"
    chunks: int = 0
    embedding_model: str = "unknown"
    checked_at: float = 0.0

    @property
    def ready(self) -> bool:
        return not self.problems


def compute_readiness(require_index_manifest: bool = True) -> Readiness:
    """Run every check once. Blocking, so call it from a thread."""
    result = Readiness(checked_at=time.monotonic())

    missing = missing_provider_settings()
    if missing:
        result.problems.append("the model provider is not configured: set " + ", ".join(missing))

    try:
        from src.retriever import get_retriever

        retriever = get_retriever()
    except IndexIntegrityError as exc:
        result.problems.append(str(exc))
        return result
    except Exception as exc:  # the index could not be opened at all
        log.exception("the retrieval index could not be opened")
        result.problems.append(f"the retrieval index could not be opened ({type(exc).__name__})")
        return result

    check = retriever.check_index(require_manifest=require_index_manifest)
    result.problems.extend(check.problems)
    result.index_version = check.index_version
    result.chunks = retriever.index_size
    result.embedding_model = retriever.embedding_name
    for message in check.warnings:
        log.warning("index: %s", message)
    return result


def needs_recheck(readiness: Readiness | None, now: float | None = None) -> bool:
    """A ready instance stays ready, because the index does not change while the process runs. One that is not
    ready is checked again after a short wait."""
    if readiness is None:
        return True
    if readiness.ready:
        return False
    return (now if now is not None else time.monotonic()) - readiness.checked_at >= RECHECK_AFTER_FAILURE_SECONDS
