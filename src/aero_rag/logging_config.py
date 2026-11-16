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

_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_MIN_REDACT_LENGTH = 6  # avoid redacting short, common strings


class JsonFormatter(logging.Formatter):
    def __init__(self, redact: Iterable[str] = ()):
        super().__init__()
        self._redact = sorted(
            {v for v in redact if v and len(v) >= _MIN_REDACT_LENGTH}, key=len, reverse=True
        )

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
        line = json.dumps(entry, default=str, ensure_ascii=False)
        for secret in self._redact:
            # Compare against the JSON-escaped form too, e.g. a password containing a quote.
            for form in {secret, json.dumps(secret)[1:-1]}:
                line = line.replace(form, "***")
        return line


def setup_logging(level: str = "INFO", redact: Iterable[str] = ()) -> None:
    """Call once at process start (API, UI, ingestion job)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(redact))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # These log every HTTP request at INFO; keep them to warnings.
    for noisy in ("httpx", "httpcore", "opensearch", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
