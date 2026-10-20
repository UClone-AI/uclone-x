"""Where a `ucx` command's log records go: `ucx.log` in `UCX_LOG_DIR`, as JSONL (#447, #1934)."""

from __future__ import annotations

import copy
import importlib
import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, cast

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


class RedactingFormatter(logging.Formatter):
    """A plain text line with credentials redacted, traceback included (#2319 review)."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_credentials(super().format(record))


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


def server_log_config() -> dict[str, Any]:
    """The logging config the dashboard server hands uvicorn: uvicorn's own, plus `ucx.log`.

    `ucx ui --dev` serves from a reload worker that uvicorn spawns as a fresh interpreter,
    so the handler `log_to_file_not_terminal` put on the launcher's root logger is not in
    it, and the worker wrote nothing anywhere but the terminal: on 2026-10-10 a failed
    turn's provider error and the reason for it were in no file (#2227). uvicorn applies
    this config in the serving process and again in every worker it spawns
    (`uvicorn._subprocess.subprocess_started`), so each one writes:

    - `uclone_x` records from INFO up -- a provider's raw answer (`connectors.failures`), a
      turn's failure with its session and room, an `a2a_call` child's reason;
    - anything else from WARNING up;
    - uvicorn's own log, an unhandled exception's traceback included ("Exception in ASGI
      application"). The access log is left to the terminal.

    The terminal that started the server still shows every WARNING and above, as uvicorn's
    own lines are shown (#2234); a `ucx` command, which logs through
    `log_to_file_not_terminal`, still does not (#1934).

    Records are JSON lines with credentials redacted (`JsonLogFormatter`). If the log
    directory cannot be made, uvicorn's config is returned as it is, so the server still
    starts. The file is opened on the first record, and a failed write is dropped, not
    printed (`CommandLogHandler`).
    """
    from uvicorn.config import LOGGING_CONFIG

    config: dict[str, Any] = copy.deepcopy(LOGGING_CONFIG)
    # uvicorn's own lines -- an ASGI exception's traceback, a request's URL -- go to the
    # terminal through its formatters; they are redacted too, before anything else can
    # return the config (#2327).
    _redact_uvicorn_formatters(config)
    try:
        target_dir = get_default_log_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return config
    config["formatters"]["ucx_json"] = {"()": f"{__name__}.JsonLogFormatter"}
    config["handlers"]["ucx_file"] = {
        "class": f"{__name__}.CommandLogHandler",
        "formatter": "ucx_json",
        "filename": str(target_dir / "ucx.log"),
        "maxBytes": MAX_LOG_BYTES,
        "backupCount": BACKUP_COUNT,
        "encoding": "utf-8",
        "delay": True,
    }
    config["root"] = {"handlers": ["ucx_file"], "level": "WARNING"}
    # The file handler took the place of Python's last-resort printer, so the server's own
    # WARNINGs stopped reaching the terminal that started it (#2234). uvicorn's loggers do
    # not propagate to the root, so their lines are not printed twice.
    # Formatted through the same redaction as the file: a line here is read off a screen,
    # and the file's `JsonLogFormatter` never sees it (#2319 review).
    config["formatters"]["ucx_redacted"] = {
        "()": "uclone_x.core.logging_setup.RedactingFormatter",
        "fmt": "%(levelname)s:     %(name)s: %(message)s",
    }
    config["handlers"]["ucx_terminal"] = {
        "class": "logging.StreamHandler",
        "formatter": "ucx_redacted",
        "stream": "ext://sys.stderr",
        "level": "WARNING",
    }
    cast(list[str], config["root"]["handlers"]).append("ucx_terminal")
    config["loggers"]["uclone_x"] = {"level": "INFO"}
    config["loggers"]["uvicorn"]["handlers"].append("ucx_file")
    return config


def redacting_formatter(base: str, **kwargs: Any) -> logging.Formatter:
    """An instance of the formatter class ``base`` names, its every line redacted (#2327).

    Keeps the class's own output (uvicorn's level prefix, colours, request line) and runs
    `redact_credentials` over the finished line, traceback included.
    """
    module_name, _, class_name = base.rpartition(".")
    cls = cast(type[logging.Formatter], getattr(importlib.import_module(module_name), class_name))

    class Redacting(cls):
        def format(self, record: logging.LogRecord) -> str:
            return redact_credentials(super().format(record))

    return Redacting(**kwargs)


def _redact_uvicorn_formatters(config: dict[str, Any]) -> None:
    """Swap each uvicorn formatter for `redacting_formatter` over the same class."""
    formatters = cast(dict[str, dict[str, Any]], config["formatters"])
    for name in ("default", "access"):
        spec = dict(formatters[name])
        base = spec.pop("()")
        formatters[name] = {"()": f"{__name__}.redacting_formatter", "base": base, **spec}


def shown_log_file() -> str:
    """`ucx.log`'s path as a person is shown it: from `~` when it is under the home
    directory, the way they type it and without their account name (#1934, #2234)."""
    log_file = get_log_file()
    try:
        return f"~/{log_file.relative_to(Path.home().resolve()).as_posix()}"
    except ValueError:
        return str(log_file)


def reason_is_in_the_log() -> str:
    """The sentence a plain notice ends with when its cause went to the log (#1934).

    It names the file, so "the reason is in the log" says which log: `ucx.log` in
    `UCX_LOG_DIR`, or `~/.uclone/logs`. A path under the home directory is written from
    `~`, the way a person types it. When the log could not be written, the reason is not
    there, so the sentence says that instead, naming the log so the person can see which
    file to fix (#1945).
    """
    shown = shown_log_file()
    if _records_dropped:
        return f"The reason could not be recorded, because the log, {shown}, could not be written."
    return f"The reason is in the log, {shown}."
