"""Structured JSON logging in the format Cloud Logging parses natively.

On Cloud Run, each stdout line that is a JSON object becomes a structured log entry:
`severity` sets the log level, `logging.googleapis.com/trace` links the entry to the
request's Cloud Trace, and every other key is queryable as jsonPayload.<key>.
Log-based metrics and alerts (monitoring/gcp/) are built on the `query_completed` event.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

# Per-request context (request_id, trace) attached to every log line emitted while serving it.
request_context: ContextVar[dict | None] = ContextVar("request_context", default=None)


class CloudLoggingFormatter(logging.Formatter):
    def __init__(self, project: str | None = None):
        super().__init__()
        self.project = project

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
        }
        ctx = dict(request_context.get() or {})
        trace_id = ctx.pop("trace_id", None)
        if trace_id and self.project:
            payload["logging.googleapis.com/trace"] = f"projects/{self.project}/traces/{trace_id}"
        payload.update(ctx)
        payload.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", project: str | None = None) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudLoggingFormatter(project))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "openai", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, event, extra={"fields": {"event": event, **fields}})
