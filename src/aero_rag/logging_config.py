"""JSON logs on stdout.

ECS sends a container's stdout to CloudWatch Logs. One JSON object per line means CloudWatch Logs
Insights can filter on fields (level, logger, request_id, ...) instead of searching free text.

Known secret values are replaced with '***' before a line is written, as a last line of defence in
case a library puts one into an error message.
"""

import json
import logging
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_MIN_REDACT_LENGTH = 6  # avoid redacting short, common strings


def redaction_list(values: Iterable[str]) -> list[str]:
    """The values to hide, longest first (so a secret containing another is hidden whole)."""
    return sorted({v for v in values if v and len(v) >= _MIN_REDACT_LENGTH}, key=len, reverse=True)


def redact_text(text: str, redact: list[str]) -> str:
    for secret in redact:
        # Compare against the JSON-escaped form too, e.g. a password containing a quote.
        for form in {secret, json.dumps(secret)[1:-1]}:
            text = text.replace(form, "***")
    return text


def redact_value(value: Any, redact: list[str]) -> Any:
    """Apply the same redaction to every string inside nested dicts, lists and tuples. Used for
    anything sent to Langfuse, so traces follow exactly the same rule as log lines."""
    if isinstance(value, str):
        return redact_text(value, redact)
    if isinstance(value, dict):
        return {redact_text(str(k), redact): redact_value(v, redact) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_value(v, redact) for v in value]
    return value


TRACING_LOGGERS = ("langfuse", "opentelemetry")


class TracingOnceFilter(logging.Filter):
    """Let through only the first record from the tracing libraries (and their child loggers),
    so a Langfuse outage produces one line instead of one per export attempt. It sits on the
    handler because a logger's own filters do not see records from its child loggers."""

    def __init__(self):
        super().__init__()
        self._seen = False

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith(TRACING_LOGGERS):
            return True
        if self._seen:
            return False
        self._seen = True
        return True


class JsonFormatter(logging.Formatter):
    def __init__(self, redact: Iterable[str] = ()):
        super().__init__()
        self._redact = redaction_list(redact)

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed as logger.info("...", extra={...}) becomes a top-level field.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            entry["exc"] = self.formatException(record.exc_info)
        return redact_text(json.dumps(entry, default=str, ensure_ascii=False), self._redact)


def setup_logging(level: str = "INFO", redact: Iterable[str] = ()) -> None:
    """Call once at process start (API, UI, ingestion job)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(redact))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # These log every HTTP request at INFO; keep them to warnings.
    # (openai 3.x uses the renamed httpx2/httpcore2 packages; both spellings are listed.)
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "opensearch",
                  "opensearchpy", "botocore", "boto3", "urllib3"):  # fmt: skip
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Tracing is optional: its libraries may report an outage once, at WARNING, and then go quiet.
    for tracing in TRACING_LOGGERS:
        logging.getLogger(tracing).setLevel(logging.WARNING)
    handler.addFilter(TracingOnceFilter())
