"""One JSON object per log line, with the request id on every line that belongs to a request.

A log line that says "query failed" only helps if you can find the other lines about the same request. The request id is
set when a request arrives (see RequestIdMiddleware) and kept in a context variable, so every line logged while that
request is handled has it, even lines from the retriever and the agent, which know nothing about HTTP. The id is also in
the X-Request-ID response header and in every error body, so a user who reports a problem can give you the one string
that finds everything.

Off by default (LOG_FORMAT=text), because JSON is hard to read on a laptop. The container image turns it on.
"""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import UTC, datetime

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("citera_request_id", default="")

# attributes every LogRecord has, so anything else was passed as `extra=` and belongs in the JSON
_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            entry["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_logging(log_format: str = "text") -> None:
    """Switch the handlers that already exist to JSON. With "text" nothing changes.

    It changes the formatter on existing handlers and adds none, so it is safe to call more than once and
    does not duplicate lines. Uvicorn's own loggers are sent through the root handlers, so its lines are JSON too.
    """
    if log_format != "json":
        return
    formatter = JsonFormatter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(formatter)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        for handler in uvicorn_logger.handlers:
            handler.setFormatter(formatter)
