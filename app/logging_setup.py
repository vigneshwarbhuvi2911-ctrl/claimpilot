"""JSON logs with run IDs, node timings, token counts and cost. No document text or personal data."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from app.guardrails import scrub_for_logs


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                   "level": record.levelname, "logger": record.name, "event": record.getMessage()}
        payload.update(getattr(record, "fields", {}) or {})
        return scrub_for_logs(json.dumps(payload, default=str))


_done = False


def configure_logging(level: str = "INFO") -> None:
    global _done
    if _done:
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "urllib3", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _done = True


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    logger.info(event, extra={"fields": fields})
