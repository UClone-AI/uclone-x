"""Log inspection, streaming, and filtering utilities for UClone-X.

Parses and filters structured JSONL and formatted text application logs
from ~/.uclone/logs/ucx.log (or UCX_LOG_DIR).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR_NAME = ".uclone"
DEFAULT_LOG_SUBDIR = "logs"
DEFAULT_LOG_FILENAME = "ucx.log"

LEVEL_WEIGHTS: dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARN": 30,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

TEXT_LOG_REGEX = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"
    r"\s+\[?([A-Za-z]+)\]?\s+\[?([a-zA-Z0-9_.]+)\]?\s*(.*)$"
)
SESSION_PATTERN = re.compile(r"""session(?:_id)?(?:=|['"]|:\s*)([a-zA-Z0-9_.-]+)""")


def _empty_log_data() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class LogEntry:
    """Structured representation of a parsed application log record."""

    timestamp: str
    level: str
    logger: str
    message: str
    session_id: str | None = None
    data: dict[str, Any] = field(default_factory=_empty_log_data)
    raw: str = ""


def get_default_log_dir() -> Path:
    """Resolve the default application logging directory, honouring UCX_LOG_DIR."""
    env_dir = os.environ.get("UCX_LOG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path.home() / DEFAULT_LOG_DIR_NAME / DEFAULT_LOG_SUBDIR).resolve()


def get_log_file(log_dir: Path | None = None) -> Path:
    """Resolve the active application log file path."""
    base_dir = log_dir if log_dir is not None else get_default_log_dir()
    return base_dir / DEFAULT_LOG_FILENAME


def parse_log_line(line: str) -> LogEntry:
    """Parse a single log line into a LogEntry.

    Accepts both single-line JSON records (the file standard per #447)
    and human-friendly console text lines.
    """
    raw = line.strip()
    if not raw:
        return LogEntry(timestamp="", level="INFO", logger="", message="", raw="")

    # 1. Try structured JSON
    if raw.startswith("{") and raw.endswith("}"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload: dict[str, Any] = cast(dict[str, Any], parsed)
                timestamp = str(payload.get("timestamp", ""))
                level = str(payload.get("level", "INFO")).upper()
                log_name = str(payload.get("logger", "root"))
                message = str(payload.get("message", ""))
                raw_sid = payload.get("session_id")
                session_id = str(raw_sid) if raw_sid is not None else None
                if session_id is None:
                    m = SESSION_PATTERN.search(message)
                    if m:
                        session_id = m.group(1)

                extra_data: dict[str, Any] = {
                    str(k): v
                    for k, v in payload.items()
                    if k not in ("timestamp", "level", "logger", "message", "session_id")
                }
                return LogEntry(
                    timestamp=timestamp,
                    level=level,
                    logger=log_name,
                    message=message,
                    session_id=session_id,
                    data=extra_data,
                    raw=raw,
                )
        except (ValueError, TypeError):
            pass

    # 2. Try structured text format
    match = TEXT_LOG_REGEX.match(raw)
    if match:
        timestamp, level_raw, log_name, message = match.groups()
        level = level_raw.upper()
        session_id = None
        m = SESSION_PATTERN.search(message)
        if m:
            session_id = m.group(1)
        return LogEntry(
            timestamp=timestamp,
            level=level,
            logger=log_name,
            message=message,
            session_id=session_id,
            raw=raw,
        )

    # 3. Plain unparsed text fallback
    m = SESSION_PATTERN.search(raw)
    session_id = m.group(1) if m else None
    return LogEntry(
        timestamp="",
        level="INFO",
        logger="root",
        message=raw,
        session_id=session_id,
        raw=raw,
    )


def read_logs(
    log_path: Path | None = None,
    tail: int = 50,
    min_level: str | None = None,
    session_id: str | None = None,
) -> list[LogEntry]:
    """Read and filter log entries from the specified log file.

    Parameters:
        log_path: Path to log file. If None, uses default ucx.log.
        tail: Maximum number of matching entries to return from the end.
        min_level: Minimum severity level to include (e.g. 'WARN', 'ERROR').
        session_id: Filter entries belonging to or referencing this session ID.
    """
    target_path = log_path if log_path is not None else get_log_file()
    if not target_path.is_file():
        return []

    min_weight = 0
    if min_level:
        min_weight = LEVEL_WEIGHTS.get(min_level.upper(), 0)

    matching: list[LogEntry] = []
    try:
        with target_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = parse_log_line(line)
                entry_weight = LEVEL_WEIGHTS.get(entry.level, 20)
                if min_weight > 0 and entry_weight < min_weight:
                    continue
                if session_id:
                    matches_sid = (
                        entry.session_id == session_id
                        or session_id in entry.message
                        or session_id in entry.raw
                    )
                    if not matches_sid:
                        continue
                matching.append(entry)
    except OSError as exc:
        logger.warning("Error reading log file %s: %s", target_path, exc)
        return []

    return matching[-tail:] if tail > 0 else matching
