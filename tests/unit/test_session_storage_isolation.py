"""Unit tests for session storage isolation and subprocess propagation guards (#453)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.support.session_leak_guard import watching_session_writes
from uclone_x.agent.session import (
    DEFAULT_SESSION_STORAGE_DIR,
    SESSION_STORAGE_DIR_ENV_VAR,
    SessionState,
    SessionStore,
    default_session_root,
    default_session_storage_dir,
)
from uclone_x.core.agent_home import (
    AGENTS_DIR_ENV_VAR,
    DEFAULT_AGENTS_ROOT,
    AgentHome,
    default_agents_root,
)
from uclone_x.llm.models import ChatMessage, MessageRole
from uclone_x.ui.server import start_ui_server


def test_default_session_storage_resolves_to_isolated_tmp_directory() -> None:
    """Session storage resolution honors UCLONE_SESSION_DIR and stays out of real home."""
    session_dir_env = os.environ.get(SESSION_STORAGE_DIR_ENV_VAR)
    assert session_dir_env is not None, f"{SESSION_STORAGE_DIR_ENV_VAR} must be set by conftest"

    root = default_session_root()
    storage_dir = default_session_storage_dir()
    real_dir = DEFAULT_SESSION_STORAGE_DIR.resolve()

    assert root == Path(session_dir_env).resolve()
    assert not root.is_relative_to(real_dir)
    assert not storage_dir.is_relative_to(real_dir)


def test_subprocess_inherits_uclone_session_dir_when_env_is_none() -> None:
    """A subprocess spawned with default env inherits UCLONE_SESSION_DIR."""
    expected = os.environ[SESSION_STORAGE_DIR_ENV_VAR]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import os; print(os.environ.get('{SESSION_STORAGE_DIR_ENV_VAR}', ''))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == expected


def test_subprocess_propagates_uclone_session_dir_when_custom_env_passed() -> None:
    """A subprocess spawned with a sanitized or custom env dict receives UCLONE_SESSION_DIR."""
    expected = os.environ[SESSION_STORAGE_DIR_ENV_VAR]
    custom_env = {"PATH": os.environ.get("PATH", "")}
    assert SESSION_STORAGE_DIR_ENV_VAR not in custom_env

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import os; print(os.environ.get('{SESSION_STORAGE_DIR_ENV_VAR}', ''))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=custom_env,
    )
    assert proc.stdout.strip() == expected


@pytest.mark.asyncio
async def test_asyncio_subprocess_propagates_uclone_session_dir() -> None:
    """Asyncio subprocess creation propagates UCLONE_SESSION_DIR even with custom env."""
    expected = os.environ[SESSION_STORAGE_DIR_ENV_VAR]
    custom_env = {"PATH": os.environ.get("PATH", "")}
    assert SESSION_STORAGE_DIR_ENV_VAR not in custom_env

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        f"import os; print(os.environ.get('{SESSION_STORAGE_DIR_ENV_VAR}', ''))",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=custom_env,
    )
    stdout, _ = await proc.communicate()
    assert stdout.decode("utf-8").strip() == expected


def test_real_default_session_storage_has_no_test_fixture_files() -> None:
    """Real developer storage contains zero test fixture files after cleanup."""
    real_dir = DEFAULT_SESSION_STORAGE_DIR.resolve()
    if not real_dir.exists():
        return

    test_named_patterns = (
        "sess_agent-",
        "sess_test",
        "sess_repl-",
        "sess_sse-",
        "sess_span-",
        "sess_stream-",
    )
    all_files = [f for f in real_dir.rglob("*.json") if f.is_file()]

    for f in all_files:
        rel_str = str(f.relative_to(real_dir))
        # Genuine user sessions are preserved
        if rel_str in ("ui/sess_default.json", "core/sess_default.json"):
            continue
        # No test fixture files should exist in real storage
        assert not any(f.name.startswith(p) for p in test_named_patterns), (
            f"Found test fixture file {f.name} in real session storage at {rel_str}"
        )


@pytest.mark.usefixtures("frontend_build_suppressed")
def test_start_ui_server_sets_session_storage_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """start_ui_server exports storage_dir to UCLONE_SESSION_DIR and forwards to create_ui_app."""
    mock_create_app = MagicMock()
    mock_uvicorn = MagicMock()
    monkeypatch.setattr("uclone_x.ui.server.create_ui_app", mock_create_app)
    monkeypatch.setattr("uvicorn.run", mock_uvicorn)

    custom_storage = tmp_path / "custom_sessions"
    start_ui_server(port=5180, dev=False, host="127.0.0.1", storage_dir=custom_storage)

    assert os.environ.get(SESSION_STORAGE_DIR_ENV_VAR) == str(custom_storage.resolve())
    mock_create_app.assert_called_once_with(storage_dir=custom_storage, bind_host="127.0.0.1")


# The leak check's attribution (#963). Each test hands the guard a stand-in for real
# storage under `tmp_path`: these tests must not write to the invoking user's home, which is
# the very thing the check forbids, and a dashboard may be using it.


def test_a_session_another_process_writes_during_a_test_is_not_a_leak(tmp_path: Path) -> None:
    """A person's live dashboard saving a session mid-run must not fail the running test.

    The child process stands in for the dashboard: another process, writing a session into
    the storage the check watches while the guard is active. The listing comparison the
    check used before #963 reported this file as the test's leak.

    Killed by: tests/support/session_leak_guard.py :: f for f in new_files if f.resolve() in written
    Becomes: f for f in new_files
    """
    real_dir = tmp_path / "real-sessions"
    real_dir.mkdir()
    dashboard_session = real_dir / "sess_mu0t8qvs.json"

    with watching_session_writes(real_dir) as guard:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('{}')",
                str(dashboard_session),
            ],
            check=True,
        )

    assert dashboard_session.is_file()
    assert guard.leaked_files() == []
    guard.assert_no_leaks()


def test_a_session_the_test_process_saves_into_real_storage_is_a_leak(tmp_path: Path) -> None:
    """A session this process saves into watched storage still fails the check, by name.

    `SessionStore.save` lands a record with `os.replace` from a temporary file, so the file
    the check finds was never itself opened for writing: only the rename destination
    attributes it to this process.

    Killed by: tests/support/session_leak_guard.py :: frozenset({"os.rename", "os.link", "os.symlink"})
    Becomes: frozenset({"os.link", "os.symlink"})
    """
    real_dir = tmp_path / "real-sessions"
    real_dir.mkdir()

    with watching_session_writes(real_dir) as guard:
        SessionStore(storage_dir=real_dir).save(
            SessionState(
                session_id="sess_leaked",
                agent_id="agent-1",
                messages=(ChatMessage(role=MessageRole.USER, content="hello"),),
            )
        )

    assert [f.name for f in guard.leaked_files()] == ["sess_leaked.json"]
    with pytest.raises(AssertionError, match=r"leaked session files into .*sess_leaked\.json"):
        guard.assert_no_leaks()


def test_a_file_the_test_process_writes_directly_into_real_storage_is_a_leak(
    tmp_path: Path,
) -> None:
    """A plain write into watched storage, with no rename, is attributed to this process.

    Killed by: tests/support/session_leak_guard.py :: flags & _WRITE_FLAGS
    Becomes: flags & 0
    """
    real_dir = tmp_path / "real-sessions"
    real_dir.mkdir()

    with watching_session_writes(real_dir) as guard:
        (real_dir / "sess_direct.json").write_text("{}", encoding="utf-8")

    assert [f.name for f in guard.leaked_files()] == ["sess_direct.json"]


def test_a_hard_link_the_test_process_makes_into_real_storage_is_a_leak(tmp_path: Path) -> None:
    """A hard link into watched storage is attributed to this process by its destination.

    The linked file's content was written outside storage before the guard started, so no
    `open` during the guard names anything in storage: only the `os.link` event does.

    Killed by: tests/support/session_leak_guard.py :: frozenset({"os.rename", "os.link", "os.symlink"})
    Becomes: frozenset({"os.rename", "os.symlink"})
    """
    real_dir = tmp_path / "real-sessions"
    real_dir.mkdir()
    written_elsewhere = tmp_path / "sess_elsewhere.json"
    written_elsewhere.write_text("{}", encoding="utf-8")

    with watching_session_writes(real_dir) as guard:
        os.link(written_elsewhere, real_dir / "sess_linked.json")

    assert [f.name for f in guard.leaked_files()] == ["sess_linked.json"]


def test_a_write_through_a_symlinked_alias_of_real_storage_is_a_leak(tmp_path: Path) -> None:
    """A write that names storage by a symlinked alias is still attributed to this process.

    The hook records the path as the writer spelled it, through the alias; the file found in
    storage is named by the real path. Only resolving the recorded path joins the two.

    Killed by: tests/support/session_leak_guard.py :: {Path(os.path.realpath(p)) for p in set(self._written)}
    Becomes: {Path(p) for p in set(self._written)}
    """
    real_dir = tmp_path / "real-sessions"
    real_dir.mkdir()
    alias = tmp_path / "alias-sessions"
    alias.symlink_to(real_dir, target_is_directory=True)

    with watching_session_writes(real_dir) as guard:
        (alias / "sess_via_alias.json").write_text("{}", encoding="utf-8")

    assert [f.name for f in guard.leaked_files()] == ["sess_via_alias.json"]


def test_a_guard_watching_storage_by_a_symlinked_alias_still_finds_a_leak(tmp_path: Path) -> None:
    """A guard handed an unresolved storage path attributes a write made through the real one.

    The listing under the alias names the file through the alias, and the recorded write names
    it by the real path. Only resolving the listed file joins the two; the conftest fixture
    resolves the path it hands over, so this is the guard's own contract, not the fixture's.

    Killed by: tests/support/session_leak_guard.py :: f.resolve() in written
    Becomes: f in written
    """
    real_dir = tmp_path / "real-sessions"
    real_dir.mkdir()
    alias = tmp_path / "alias-sessions"
    alias.symlink_to(real_dir, target_is_directory=True)

    with watching_session_writes(alias) as guard:
        (real_dir / "sess_via_real_path.json").write_text("{}", encoding="utf-8")

    assert [f.name for f in guard.leaked_files()] == ["sess_via_real_path.json"]


def test_a_closed_guard_stops_recording_writes(tmp_path: Path) -> None:
    """A write made after the guard's block has ended is not attributed to that guard.

    Every test opens a guard, and a guard never taken off the active list would keep
    recording every write for the rest of the worker's life, charging a later write to a
    test that had already finished.

    Killed by: tests/support/session_leak_guard.py :: _active.remove(guard)
    Becomes: pass
    """
    real_dir = tmp_path / "real-sessions"
    real_dir.mkdir()

    with watching_session_writes(real_dir) as guard:
        pass
    (real_dir / "sess_after_close.json").write_text("{}", encoding="utf-8")

    assert guard.leaked_files() == []


# The fixture itself (#992). The guard tests above call the guard directly, so nothing in them
# fails when `tests/conftest.py` stops calling `assert_no_leaks()`. Only a pytest run can show
# the fixture failing a test, so a child pytest runs one leaking test with the repo conftest
# loaded. The child's `HOME` is a scratch directory, which moves `DEFAULT_SESSION_STORAGE_DIR`
# (`Path.home()`-based, fixed at import) under `tmp_path`, and the leaking test refuses to
# write unless it did: this must never write into the invoking user's real storage.

_STORAGE_ENV = "UCLONE_LEAK_META_STORAGE"

_LEAKING_TEST_MODULE = f"""
import os
from pathlib import Path

from uclone_x.agent.session import DEFAULT_SESSION_STORAGE_DIR


def test_saves_into_default_session_storage():
    scratch_storage = Path(os.environ["{_STORAGE_ENV}"])
    assert DEFAULT_SESSION_STORAGE_DIR == scratch_storage, DEFAULT_SESSION_STORAGE_DIR
    scratch_storage.mkdir(parents=True)
    (scratch_storage / "sess_fixture_leak.json").write_text("{{}}", encoding="utf-8")
"""


def test_the_conftest_fixture_fails_a_test_that_leaks_into_real_storage(tmp_path: Path) -> None:
    """A test that writes into default session storage fails at teardown, naming the file.

    Killed by: tests/conftest.py :: guard.assert_no_leaks()
    Becomes: pass
    """
    repo_root = Path(__file__).resolve().parents[2]
    scratch_home = tmp_path / "home"
    scratch_home.mkdir()
    scratch_storage = scratch_home / ".uclone" / "sessions"
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "test_leaks.py").write_text(_LEAKING_TEST_MODULE, encoding="utf-8")

    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    env["HOME"] = str(scratch_home)
    env[_STORAGE_ENV] = str(scratch_storage)
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), str(repo_root / "src")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.conftest",
            f"--basetemp={tmp_path / 'child-basetemp'}",
            str(workdir / "test_leaks.py"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(workdir),
        timeout=120,
    )

    output = result.stdout + result.stderr
    # Count outcomes, never the exit code: a child that collected nothing also exits non-zero.
    assert "1 passed, 1 error" in result.stdout, f"child pytest outcome:\n{output}"
    assert (scratch_storage / "sess_fixture_leak.json").is_file(), output
    assert "leaked session files into the real storage root" in result.stdout, output
    assert "sess_fixture_leak.json" in result.stdout, output


def test_agent_homes_resolve_into_an_isolated_tmp_directory() -> None:
    """The agents root is redirected for the suite, as session storage is.

    `default_cross_session_memory` mkdirs a home and mints an id, so *asking* for an
    agent's memory now writes. Without this redirection the suite did that in the real
    `~/.uclone/agents`: 89 directories named after test fixtures, one of them a traversal
    artefact from a mutation run. Session storage learned this in #453; the agents root
    arrived after, and inherited nothing.
    """
    override = os.environ.get(AGENTS_DIR_ENV_VAR)
    assert override is not None, f"{AGENTS_DIR_ENV_VAR} must be set by conftest"

    root = default_agents_root().resolve()
    assert root != DEFAULT_AGENTS_ROOT.resolve(), "the suite must not write real agent homes"
    assert Path.home() not in root.parents or ".uclone" not in root.parts


def test_an_agent_home_written_into_the_real_root_is_reported_as_one(tmp_path: Path) -> None:
    """The guard names what leaked, because "session" would send the reader to the wrong store.

    The same guard now watches two roots. A leaked agent home reported as a leaked
    *session* file points the person at `UCLONE_SESSION_DIR` and at the session fixtures,
    neither of which is involved -- the mis-attribution P6 forbids, inside the guard whose
    job is attribution.

    Killed by: tests/support/session_leak_guard.py :: f"Test leaked {self.label} files into the real storage root "
    Becomes: f"Test leaked session files into the real storage root "
    """
    real_root = tmp_path / "real-agents"
    real_root.mkdir()

    with watching_session_writes(real_root, label="agent home") as guard:
        home = AgentHome.for_username("leaky", root=real_root)
        home.agent_id()

        with pytest.raises(AssertionError) as caught:
            guard.assert_no_leaks()

    assert "agent home files" in str(caught.value)
    assert "leaky/id" in str(caught.value), "the message must name the home that leaked"
