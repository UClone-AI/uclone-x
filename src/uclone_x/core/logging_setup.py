"""Application logging setup fulfilling #447 and Logging & Observability Policy.

Configures:
- Rotating JSONL file handler targeting ~/.uclone/logs/ucx.log (or UCX_LOG_DIR)
- Console stream handler with color/level formatting
- Environment variable overrides (UCX_LOG_LEVEL, UCX_LOG_DIR, UCX_LOG_CONSOLE)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from uclone_x.core.log_inspector import get_default_log_dir
from uclone_x.core.secrets import redact_credentials

MAX_LOG_BYTES = 50 * 1024 * 1024  # 50 MB
BACKUP_COUNT = 5


class UcxRotatingFileHandler(RotatingFileHandler):
    """Marker subclass for UClone-X managed file loggers."""

    ucx_managed: bool = True


class UcxStreamHandler(logging.StreamHandler[Any]):
    """Marker subclass for UClone-X managed console loggers."""

    ucx_managed: bool = True


class JsonLogFormatter(logging.Formatter):
    """Serialize log records into single-line JSON objects per #447, redacting credentials (#569)."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": redact_credentials(record.getMessage()),
        }
        sid = getattr(record, "session_id", None)
        if sid is not None:
            payload["session_id"] = str(sid)
        tid = getattr(record, "trace_id", None)
        if tid is not None:
            payload["trace_id"] = str(tid)
        if record.exc_info:
            payload["exception"] = redact_credentials(self.formatException(record.exc_info))
        return json.dumps(payload)


class RedactingConsoleFormatter(logging.Formatter):
    """Console formatter that redacts credential shapes on output (#569)."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_credentials(super().format(record))


def setup_application_logging(
    log_dir: Path | None = None,
    log_level: str | None = None,
    enable_console: bool | None = None,
    enable_file: bool = True,
) -> None:
    """Configure root logger with file rotation and console output per policy."""
    env_level = os.environ.get("UCX_LOG_LEVEL", "INFO").upper()
    eff_level_name = (log_level or env_level).upper()
    level = getattr(logging, eff_level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if setup called multiple times
    root_logger.handlers = [h for h in root_logger.handlers if not getattr(h, "ucx_managed", False)]

    target_dir = log_dir if log_dir is not None else get_default_log_dir()

    if enable_file:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            log_file = target_dir / "ucx.log"
            file_handler = UcxRotatingFileHandler(
                filename=log_file,
                maxBytes=MAX_LOG_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(JsonLogFormatter())
            root_logger.addHandler(file_handler)
        except OSError:
            pass

    # Console handler
    env_console = os.environ.get("UCX_LOG_CONSOLE", "1").lower() in ("1", "true", "yes")
    eff_console = enable_console if enable_console is not None else env_console
    if eff_console:
        console_handler = UcxStreamHandler()
        console_handler.setLevel(level)
        fmt = RedactingConsoleFormatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(fmt)
        root_logger.addHandler(console_handler)
