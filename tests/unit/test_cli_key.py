"""``ucx key``: keys are saved in the settings file, one per connection, never in ``.env``.

``ucx key setup`` used to write ``GEMINI_API_KEY`` into a ``.env`` file while the
dashboard's Settings wrote the settings file; which key a request used depended on which
was loaded first. These tests pin the one store and the key on each connection row
(model-gateway §3.2).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from uclone_x.cli.commands.key import mask_key, sanitize_key
from uclone_x.cli.main import app
from uclone_x.llm.connectors.saved_choice import settings_file

runner = CliRunner()

_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "VLLM_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_key_in_the_environment(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _saved() -> dict[str, Any]:
    return json.loads(settings_file().read_text(encoding="utf-8"))


def _keys() -> dict[str, str]:
    """The saved keys, by connection id."""
    return {row["id"]: row["key"] for row in _saved()["connections"] if "key" in row}


def _write(data: dict[str, Any]) -> None:
    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data), encoding="utf-8")


def test_sanitize_key() -> None:
    assert sanitize_key("  AIzaSy12345  ") == "AIzaSy12345"
    assert sanitize_key('"sk-ant-12345"') == "sk-ant-12345"
    assert sanitize_key("'sk-proj-12345'") == "sk-proj-12345"


def test_mask_key() -> None:
    assert mask_key("short") == "****"
    assert mask_key("AIzaSy1234567890abcdef") == "AIzaSy...cdef"


def test_set_saves_the_key_on_its_connection_in_the_settings_file(tmp_path: Path) -> None:
    """The key lands on the `gemini` connection, the file is private, and no `.env` is written."""
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["key", "set", "gemini", "--key", "AQ.gemini-key-0001"])
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0, result.output
    assert _saved()["connections"] == [
        {"id": "gemini", "kind": "gemini", "key": "AQ.gemini-key-0001"}
    ]
    assert stat.S_IMODE(settings_file().stat().st_mode) == 0o600
    assert not (tmp_path / ".env").exists()
    assert "GEMINI_API_KEY" not in os.environ
    assert "AQ.gemini-key-0001" not in result.output  # masked when echoed


def test_set_saves_the_key_on_a_named_connection_only() -> None:
    """A second connection of one kind gets its own key; the first keeps none.

    Killed by: src/uclone_x/cli/commands/key.py :: kept_aside = _save_over_unreadable(target, clean_key)
    Becomes: kept_aside = _save_over_unreadable(spec.id, clean_key)
    """
    _write({"connections": [{"id": "openai", "kind": "openai"}, {"id": "work", "kind": "openai"}]})

    result = runner.invoke(app, ["key", "set", "work", "--key", "sk-work-key-0001"])

    assert result.exit_code == 0, result.output
    assert _keys() == {"work": "sk-work-key-0001"}


def test_saving_one_connections_key_keeps_every_other_connections_key() -> None:
    assert (
        runner.invoke(app, ["key", "set", "openai", "--key", "sk-openai-key-0001"]).exit_code == 0
    )
    assert (
        runner.invoke(app, ["key", "set", "google", "--key", "AQ.gemini-key-0001"]).exit_code == 0
    )
    assert _keys() == {
        "openai": "sk-openai-key-0001",
        "gemini": "AQ.gemini-key-0001",
    }


def test_remove_deletes_only_that_connections_key() -> None:
    runner.invoke(app, ["key", "set", "openai", "--key", "sk-openai-key-0001"])
    runner.invoke(app, ["key", "set", "anthropic", "--key", "sk-ant-anthropic-0001"])
    result = runner.invoke(app, ["key", "remove", "openai"])
    assert result.exit_code == 0, result.output
    assert _keys() == {"anthropic": "sk-ant-anthropic-0001"}


def test_list_shows_saved_keys_masked_and_names_an_overriding_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An environment key outranks the saved one, and `list` says which variable does.

    Killed by: src/uclone_x/cli/commands/key.py :: f"{overriding[1]} ({mask_key(overriding[0])}) overrides the saved key"
    Becomes: f"[dim]{spec.key_env_vars[0]} not set[/dim]"
    """
    from rich.console import Console

    from uclone_x.cli.commands import key as key_module

    # Wide enough that no cell wraps: the module's console ignores the runner's width.
    monkeypatch.setattr(key_module, "console", Console(width=300))
    runner.invoke(app, ["key", "set", "openai", "--key", "sk-openai-key-0001"])
    monkeypatch.setenv("GOOGLE_API_KEY", "AQ.from-the-environment-9")
    result = runner.invoke(app, ["key", "list"], terminal_width=200)
    out = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "sk-ope...0001" in out
    assert "sk-openai-key-0001" not in out
    assert "GOOGLE_API_KEY" in out and "overrides the saved key" in out
    assert "AQ.from-the-environment-9" not in out


def test_set_warns_when_an_environment_variable_will_be_used_instead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-0001")
    result = runner.invoke(app, ["key", "set", "anthropic", "--key", "sk-ant-saved-0001"])
    assert result.exit_code == 0
    assert "ANTHROPIC_API_KEY is set in this environment" in _flat(result.output)


def test_unknown_provider_is_refused_in_plain_words() -> None:
    result = runner.invoke(app, ["key", "set", "acme", "--key", "whatever-key-1"])
    out = _flat(result.output)
    assert result.exit_code == 2
    assert "There is no connection or provider called 'acme' that takes a key." in out
    assert "openai" in out and "gemini" in out
    assert "Traceback" not in out and "ValueError" not in out
    assert not settings_file().exists()


def test_empty_key_is_refused_and_nothing_is_saved() -> None:
    result = runner.invoke(app, ["key", "set", "openai", "--key", "  ''  "])
    assert result.exit_code == 1
    assert "The key is empty, so nothing was saved." in _flat(result.output)
    assert not settings_file().exists()


def test_setup_saves_the_key_in_the_settings_file(tmp_path: Path) -> None:
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(
            app,
            [
                "key",
                "setup",
                "--provider",
                "gemini",
                "--key",
                "AIzaSy123456789012345678901234567890123",
                "--no-open-browser",
            ],
        )
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0, result.output
    assert _keys() == {"gemini": "AIzaSy123456789012345678901234567890123"}
    assert not (tmp_path / ".env").exists()


def _plain(text: str) -> None:
    """Copy for a person: no file location, no class name, no parser text (#1921).

    The one location allowed is the log's, which a notice names so the person can find
    the reason (#1934).
    """
    from uclone_x.core.logging_setup import reason_is_in_the_log

    flat = _flat(text).replace(reason_is_in_the_log(), "")
    assert str(settings_file()) not in flat, flat
    assert settings_file().name not in flat, flat
    assert "/" not in flat and "\\" not in flat, flat
    assert ".unreadable-" not in flat, flat
    for internal in ("Error", "Exception", "Traceback", "json", "Errno"):
        assert internal not in flat, (internal, flat)


def test_a_key_set_over_unreadable_settings_keeps_the_file_aside_and_says_so_plainly() -> None:
    """The file is kept unchanged beside a new one, the key is saved, and nothing names it.

    Before, `ucx key set` failed with the settings file's path as its reason (#1921), as
    the app's Settings did before #1918.

    Killed by: src/uclone_x/cli/commands/key.py :: if update_settings_file({}, replace_unreadable_with={}) is None:
    Becomes: if True:
    Killed by: src/uclone_x/cli/commands/key.py :: console.print(f"[yellow]{escape(SETTINGS_SET_ASIDE_NOTICE)}[/yellow]")
    Becomes: pass
    Killed by: src/uclone_x/cli/commands/key.py :: f"([green]{mask_key(clean_key)}[/green])."
    Becomes: f"([green]{mask_key(clean_key)}[/green]) in {escape(str(settings_file()))}."
    """
    from uclone_x.cli.commands.key import SETTINGS_SET_ASIDE_NOTICE
    from uclone_x.core.set_aside import set_aside_copies

    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"connections": [', encoding="utf-8")

    result = runner.invoke(
        app, ["key", "set", "openai", "--key", "sk-openai-key-1921"], terminal_width=400
    )

    assert result.exit_code == 0, result.output
    assert _keys() == {"openai": "sk-openai-key-1921"}
    (aside,) = set_aside_copies(target)
    assert aside.read_text(encoding="utf-8") == '{"connections": ['
    assert SETTINGS_SET_ASIDE_NOTICE in _flat(result.output)
    _plain(result.output)


def test_a_key_that_could_not_be_saved_is_reported_without_the_files_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settings file could not be kept aside: the key is not saved, and the reason the
    person reads names no file and quotes no error (#1921).

    Killed by: src/uclone_x/cli/commands/key.py :: refusal = f"{KEY_NOT_SAVED_NOTICE} {reason_is_in_the_log()}"
    Becomes: refusal = KEY_NOT_SAVED_NOTICE
    Killed by: src/uclone_x/cli/commands/key.py :: refusal = f"{KEY_NOT_SAVED_NOTICE} {reason_is_in_the_log()}"
    Becomes: refusal = f"The key was not saved: {exc}"
    """
    from uclone_x.cli.commands.key import KEY_NOT_SAVED_NOTICE
    from uclone_x.core.logging_setup import reason_is_in_the_log
    from uclone_x.llm.connectors import saved_choice

    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"connections": [', encoding="utf-8")

    def _cannot_move(path: Path) -> Path:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(saved_choice, "set_aside_unreadable", _cannot_move)

    result = runner.invoke(
        app, ["key", "set", "openai", "--key", "sk-openai-key-1921"], terminal_width=400
    )

    assert result.exit_code == 1, result.output
    assert target.read_text(encoding="utf-8") == '{"connections": ['
    assert f"{KEY_NOT_SAVED_NOTICE} {reason_is_in_the_log()}" in _flat(result.output)
    _plain(result.output)


def test_a_key_removed_over_unreadable_settings_is_refused_without_the_files_path() -> None:
    """Removing a key from a file this version cannot read leaves it as it is, plainly.

    Killed by: src/uclone_x/cli/commands/key.py :: refusal = f"{KEY_NOT_REMOVED_NOTICE} {reason_is_in_the_log()}"
    Becomes: refusal = KEY_NOT_REMOVED_NOTICE
    Killed by: src/uclone_x/cli/commands/key.py :: refusal = f"{KEY_NOT_REMOVED_NOTICE} {reason_is_in_the_log()}"
    Becomes: refusal = f"The key was not removed: {exc}"
    """
    from uclone_x.cli.commands.key import KEY_NOT_REMOVED_NOTICE
    from uclone_x.core.logging_setup import reason_is_in_the_log

    target = settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"connections": [', encoding="utf-8")

    result = runner.invoke(app, ["key", "remove", "openai"], terminal_width=400)

    assert result.exit_code == 1, result.output
    assert target.read_text(encoding="utf-8") == '{"connections": ['
    assert f"{KEY_NOT_REMOVED_NOTICE} {reason_is_in_the_log()}" in _flat(result.output)
    _plain(result.output)


def _run_ucx(tmp_path: Path, *args: str, no_file_growth: bool = False) -> tuple[int, str, str]:
    """`ucx` in its own process, as a person runs it: no pytest log handler on the root.

    In-process, pytest's logging plugin holds a root handler, so a WARNING never reaches
    Python's last-resort handler -- which, in a real `ucx`, prints it to stderr raw.
    `no_file_growth` sets the file-size limit to 0, so every write to a file fails the way
    a full disk's does; the pipes the streams are read from are not files.
    """
    import subprocess
    import sys

    import uclone_x

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("_API_KEY") and not k.startswith(("UCX_", "UCLONE_", "PYTEST"))
    }
    env |= {
        "HOME": str(tmp_path / "home"),
        "UCLONE_SESSION_DIR": str(tmp_path / "sessions"),
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
        preexec_fn=_no_file_growth if no_file_growth else None,
    )
    return done.returncode, done.stdout, done.stderr


def _no_file_growth() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (0, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))


def _unreadable_settings(tmp_path: Path) -> Path:
    target = tmp_path / "sessions" / settings_file().name
    target.parent.mkdir(parents=True)
    target.write_text('{"connections": [', encoding="utf-8")
    return target


def _terminal_is_plain(target: Path, *streams: str) -> None:
    for text in streams:
        assert str(target.parent) not in text, text
        assert ".unreadable-" not in text, text
        assert "Errno" not in text, text


def test_a_key_set_over_unreadable_settings_prints_no_path_to_the_terminal(
    tmp_path: Path,
) -> None:
    """Run as a person runs it, neither stream names a file; the log has the reason (#1921).

    Before, the set-aside's WARNING and `ucx key set`'s own reached stderr through
    Python's last-resort handler, each with the settings file's path.

    Killed by: src/uclone_x/cli/main.py :: ctx.with_resource(log_to_file_not_terminal())
    Becomes: pass
    """
    target = _unreadable_settings(tmp_path)

    code, out, err = _run_ucx(tmp_path, "key", "set", "openai", "--key", "sk-openai-key-1921")

    assert code == 0, (out, err)
    _terminal_is_plain(target, out, err)
    assert ".unreadable-" in (tmp_path / "logs" / "ucx.log").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions")
def test_a_key_that_could_not_be_saved_prints_no_error_text_to_the_terminal(
    tmp_path: Path,
) -> None:
    """The settings file cannot be kept aside: the refusal is plain on both streams.

    Before, stderr read `The openai key was not saved: [Errno 13] Permission denied: '<path>'`.

    Killed by: src/uclone_x/cli/main.py :: ctx.with_resource(log_to_file_not_terminal())
    Becomes: pass
    """
    target = _unreadable_settings(tmp_path)
    target.parent.chmod(0o500)
    try:
        code, out, err = _run_ucx(tmp_path, "key", "set", "openai", "--key", "sk-openai-1921")
        for_removal = _run_ucx(tmp_path, "key", "remove", "openai")
    finally:
        target.parent.chmod(0o700)

    assert code == 1, (out, err)
    assert for_removal[0] == 1, for_removal
    _terminal_is_plain(target, out, err, for_removal[1], for_removal[2])
    assert "Errno" in (tmp_path / "logs" / "ucx.log").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions")
def test_a_refusal_when_the_log_cannot_be_written_does_not_point_at_the_log(
    tmp_path: Path,
) -> None:
    """With `UCX_LOG_DIR` read-only the reason is dropped, and the notice says so (#1945).

    Before, the notice ended "The reason is in the log, <log>." over a log that did not
    have it. The log's location is the one path shown: it is the file to fix.

    Killed by: src/uclone_x/core/logging_setup.py :: if _records_dropped:
    Becomes: if False:
    Killed by: src/uclone_x/core/logging_setup.py :: _drop_records()  # the log cannot be opened
    Becomes: pass
    """
    from uclone_x.cli.commands.key import KEY_NOT_REMOVED_NOTICE, KEY_NOT_SAVED_NOTICE

    target = _unreadable_settings(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    target.parent.chmod(0o500)
    logs.chmod(0o500)
    try:
        saved = _run_ucx(tmp_path, "key", "set", "openai", "--key", "sk-openai-1945")
        removed = _run_ucx(tmp_path, "key", "remove", "openai")
    finally:
        target.parent.chmod(0o700)
        logs.chmod(0o700)

    dropped = (
        f"The reason could not be recorded, because the log, {logs.resolve() / 'ucx.log'}, "
        "could not be written."
    )
    assert saved == (1, f"{KEY_NOT_SAVED_NOTICE} {dropped}\n", "")
    assert removed == (1, f"{KEY_NOT_REMOVED_NOTICE} {dropped}\n", "")
    assert not (logs / "ucx.log").exists()


@pytest.mark.skipif(os.name == "nt", reason="needs RLIMIT_FSIZE")
def test_a_log_write_that_fails_midway_prints_no_traceback_when_the_command_ends(
    tmp_path: Path,
) -> None:
    """The log opens, then every write to it fails, as on a full disk (#1957).

    The failed write's text stayed buffered, and closing the log at exit retried it:
    after the notice, stderr carried a traceback with `OSError: [Errno 27] File too
    large` and the settings file's path. Now the notice is all there is, and it says the
    reason could not be recorded; the exit is the refusal's own.

    Killed by: src/uclone_x/core/logging_setup.py :: def close(self) -> None:
    Becomes: def _close_unused(self) -> None:
    """
    from uclone_x.cli.commands.key import KEY_NOT_REMOVED_NOTICE, KEY_NOT_SAVED_NOTICE

    _unreadable_settings(tmp_path)
    saved = _run_ucx(tmp_path, "key", "set", "openai", "--key", "sk-1957", no_file_growth=True)
    removed = _run_ucx(tmp_path, "key", "remove", "openai", no_file_growth=True)

    logs = tmp_path / "logs"
    dropped = (
        f"The reason could not be recorded, because the log, {logs.resolve() / 'ucx.log'}, "
        "could not be written."
    )
    assert saved == (1, f"{KEY_NOT_SAVED_NOTICE} {dropped}\n", "")
    assert removed == (1, f"{KEY_NOT_REMOVED_NOTICE} {dropped}\n", "")
    assert (logs / "ucx.log").read_bytes() == b""
