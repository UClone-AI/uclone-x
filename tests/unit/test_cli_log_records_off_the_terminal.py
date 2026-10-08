"""`ucx run` and `ucx loop`, run as a person runs them, print no log record (#1934).

`ucx` configured no log handler, so a WARNING reached stderr through Python's
last-resort handler. A session save that kept an unreadable record aside printed the
record's path, the `.unreadable-` copy's path and the parser's validation text. These
tests start the real command in its own process: in-process, pytest's logging plugin
holds a root handler, so the last-resort handler never runs and a caplog test would pass
with the leak present.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import uclone_x
from tests.support.clones import make_clones
from uclone_x.agent.session import SessionState, SessionStore, default_session_storage_dir
from uclone_x.core.agent_home import seat_id_for
from uclone_x.core.set_aside import SESSION_SET_ASIDE_NOTICE
from uclone_x.room.service import participant_session_id

#: What a log record carries and a person's terminal must not.
_INTERNALS = (".unreadable-", "Errno", "validation error", "extra_forbidden", "pydantic")


def _run_ucx(tmp_path: Path, *args: str) -> tuple[int, str, str]:
    """`ucx` in its own process, with its home, sessions and log under `tmp_path`."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("_API_KEY") and not k.startswith(("UCX_", "UCLONE_", "PYTEST", "LLM_"))
    }
    env |= {
        "HOME": str(tmp_path / "home"),
        "UCLONE_SESSION_DIR": str(tmp_path / "sessions"),
        "UCLONE_AGENTS_DIR": str(tmp_path / "agents"),
        "UCX_LOG_DIR": str(tmp_path / "logs"),
        "PYTHONPATH": str(Path(uclone_x.__file__).parents[1]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "COLUMNS": "400",
    }
    done = subprocess.run(
        [sys.executable, "-c", "from uclone_x.cli.main import main; main()", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return done.returncode, done.stdout, done.stderr


def _unreadable_seat_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, room_id: str, handle: str
) -> None:
    """A newer build's record for the clone's seat in `room_id`: valid JSON, unknown field."""
    make_clones(handle, root=tmp_path / "agents")  # a name no clone carries is refused
    # The seat is keyed by the clone's id, not the handle the command names (§4 step 3).
    clone_id = seat_id_for(handle, root=tmp_path / "agents")
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    store = SessionStore(storage_dir=default_session_storage_dir())
    seat = participant_session_id(room_id, clone_id)
    document = json.loads(SessionState.seed(seat, clone_id).model_dump_json())
    document["a_field_from_a_newer_build"] = True
    path = store.session_path(seat)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _assert_plain(tmp_path: Path, *streams: str) -> None:
    for text in streams:
        for root in {str(tmp_path), str(tmp_path.resolve())}:
            assert root not in text, text
        for internal in _INTERNALS:
            assert internal not in text, (internal, text)


def _assert_logged(tmp_path: Path) -> None:
    """The record went somewhere: the log has the copy's path and the parser's text."""
    logged = (tmp_path / "logs" / "ucx.log").read_text(encoding="utf-8")
    assert ".unreadable-" in logged
    assert "extra_forbidden" in logged


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "home").mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def test_a_run_over_an_unreadable_record_prints_only_the_plain_notice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace: Path
) -> None:
    """The record is kept aside; stderr says so plainly and names no file (#1934).

    Before, stderr carried the session store's two WARNINGs: the record's path, the
    `.unreadable-` copy's path and pydantic's validation text.

    Killed by: src/uclone_x/cli/main.py :: ctx.with_resource(log_to_file_not_terminal())
    Becomes: pass
    """
    _unreadable_seat_record(monkeypatch, tmp_path, "kept", "keeper")

    code, out, err = _run_ucx(
        tmp_path,
        "run",
        "keeper",
        "--provider",
        "mock",
        "--prompt",
        "hello",
        "--session-id",
        "kept",
        "--cwd",
        str(workspace),
    )

    assert code == 0, (out, err)
    assert SESSION_SET_ASIDE_NOTICE in " ".join(err.split())
    _assert_plain(tmp_path, out, err)
    _assert_logged(tmp_path)


def test_a_loop_over_an_unreadable_record_prints_only_the_plain_notice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace: Path
) -> None:
    """`ucx loop` keeps its seat's unreadable record aside and says so plainly (#1934).

    Killed by: src/uclone_x/cli/main.py :: ctx.with_resource(log_to_file_not_terminal())
    Becomes: pass
    """
    _unreadable_seat_record(monkeypatch, tmp_path, "nightly", "looper")

    code, out, err = _run_ucx(
        tmp_path,
        "loop",
        "run",
        "tick",
        "--interval",
        "1h",
        "--max-runs",
        "1",
        "--provider",
        "mock",
        "--agent",
        "looper",
        "--session-id",
        "nightly",
        "--cwd",
        str(workspace),
    )

    assert code == 0, (out, err)
    assert SESSION_SET_ASIDE_NOTICE in " ".join(err.split())
    _assert_plain(tmp_path, out, err)
    _assert_logged(tmp_path)


def test_the_log_a_notice_names_is_the_one_records_go_to(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A notice's "the reason is in the log" names `ucx.log` where it is written.

    Under the home directory it is written from `~`.

    Killed by: src/uclone_x/core/logging_setup.py :: shown = f"~/{log_file.relative_to(Path.home().resolve()).as_posix()}"
    Becomes: shown = "the log"
    """
    from uclone_x.core.logging_setup import reason_is_in_the_log

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("UCX_LOG_DIR", raising=False)
    assert reason_is_in_the_log() == "The reason is in the log, ~/.uclone/logs/ucx.log."

    elsewhere = tmp_path.parent / "elsewhere"
    monkeypatch.setenv("UCX_LOG_DIR", str(elsewhere))
    assert reason_is_in_the_log() == f"The reason is in the log, {elsewhere.resolve() / 'ucx.log'}."


def test_a_write_the_log_refuses_is_dropped_quietly_and_the_notice_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A write that fails after the log was opened prints nothing and is marked (#1945).

    `logging.Handler.handleError` would print the failure's traceback to stderr, with
    the log's path and the error's text; and the notice would still point at the log.

    Killed by: src/uclone_x/core/logging_setup.py :: def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802 -- logging's name
    Becomes: def _handle_error_unused(self, record: logging.LogRecord) -> None:
    """
    import logging

    from uclone_x.core import logging_setup

    monkeypatch.setenv("UCX_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(logging_setup, "_records_dropped", False)
    handler = logging_setup.CommandLogHandler(tmp_path / "ucx.log")
    monkeypatch.setattr(logging, "raiseExceptions", True)
    stream = handler.stream
    assert stream is not None
    stream.close()  # the next write raises, as a full disk's would
    try:
        handler.emit(logging.LogRecord("x", logging.WARNING, __file__, 1, "lost", None, None))
    finally:
        handler.stream = None
        handler.close()

    assert capsys.readouterr().err == ""
    assert logging_setup.reason_is_in_the_log() == (
        f"The reason could not be recorded, because the log, {tmp_path.resolve() / 'ucx.log'}, "
        "could not be written."
    )


def test_a_command_logs_through_the_handler_that_drops_failed_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The handler a command gets is the one whose failed writes stay off the terminal.

    A plain rotating file handler writes the same file, so every other test of the log's
    content passes with it; only its failures differ: it prints them (#1957).

    Killed by: src/uclone_x/core/logging_setup.py :: handler = CommandLogHandler(
    Becomes: handler = UcxRotatingFileHandler(
    """
    import logging

    from uclone_x.core import logging_setup

    monkeypatch.setenv("UCX_LOG_DIR", str(tmp_path))
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    with logging_setup.log_to_file_not_terminal():
        (installed,) = root.handlers
        assert type(installed) is logging_setup.CommandLogHandler
    assert root.handlers == []
