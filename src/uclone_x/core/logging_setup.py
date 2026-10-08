"""Where a `ucx` command's log records go: `ucx.log` in `UCX_LOG_DIR`, as JSONL (#447, #1934)."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from uclone_x.core.log_inspector import get_default_log_dir, get_log_file
from uclone_x.core.secrets import redact_credentials

MAX_LOG_BYTES = 50 * 1024 * 1024  # 50 MB
BACKUP_COUNT = 5


class UcxRotatingFileHandler(RotatingFileHandler):
    """Marker subclass for UClone-X managed file loggers."""

    ucx_managed: bool = True


#: Whether a record this command sent to `ucx.log` was dropped: the log could not be opened,
#: or a write to it failed. A notice that would point at the log says so instead (#1945).
_records_dropped = False


def _drop_records() -> None:
    global _records_dropped
    _records_dropped = True


class CommandLogHandler(UcxRotatingFileHandler):
    """The file handler a `ucx` command logs through, which never writes to the terminal.

    `logging.Handler.handleError` prints a failed write's traceback to stderr, paths and
    `Errno` text included. Here a failed write only marks the record as dropped, so the
    command's notice can say the log does not hold its reason. A record that cannot be
    formatted reaches `handleError` too, and is dropped the same way: printing it would put
    the call's raw arguments on the terminal.
    """

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802 -- logging's name
        _drop_records()

    def close(self) -> None:
        """Close the log; a failed write's text still in the buffer is dropped, not printed.

        After a write fails, its text stays buffered, and closing retries it. On a full
        disk the retry fails the same way, and `logging` raises it out of the command's
        exit: a traceback with `Errno` and the log's path, after the command's notice (#1957).
        """
        try:
            super().close()
        except OSError:
            _drop_records()


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


@contextmanager
def log_to_file_not_terminal() -> Generator[None]:
    """While a command runs, write log records to `ucx.log`, not the person's terminal.

    With no handler configured anywhere -- the case for a `ucx` command that does not set
    logging up -- Python's last-resort handler prints every WARNING to stderr, raw: the
    file paths and error text a command's own plain message leaves out (#1921). This
    puts one file handler on the root logger for the duration, and removes it after. If
    logging is already configured (by the host, or by pytest) it changes nothing. If the
    log directory cannot be written, the records are dropped rather than printed, and
    `reason_is_in_the_log` stops pointing at the log (#1945).
    """
    global _records_dropped
    root = logging.getLogger()
    if root.handlers:
        yield
        return
    _records_dropped = False
    handler: logging.Handler
    try:
        target_dir = get_default_log_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        handler = CommandLogHandler(
            filename=target_dir / "ucx.log",
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(JsonLogFormatter())
    except OSError:
        _drop_records()  # the log cannot be opened
        handler = logging.NullHandler()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def reason_is_in_the_log() -> str:
    """The sentence a plain notice ends with when its cause went to the log (#1934).

    It names the file, so "the reason is in the log" says which log: `ucx.log` in
    `UCX_LOG_DIR`, or `~/.uclone/logs`. A path under the home directory is written from
    `~`, the way a person types it. When the log could not be written, the reason is not
    there, so the sentence says that instead, naming the log so the person can see which
    file to fix (#1945).
    """
    log_file = get_log_file()
    try:
        shown = f"~/{log_file.relative_to(Path.home().resolve()).as_posix()}"
    except ValueError:
        shown = str(log_file)
    if _records_dropped:
        return f"The reason could not be recorded, because the log, {shown}, could not be written."
    return f"The reason is in the log, {shown}."
