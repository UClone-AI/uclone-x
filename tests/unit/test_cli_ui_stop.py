"""`ucx ui stop` and the process helpers it uses (#613, #881, #927).

The helpers live in `uclone_x.shells.ui_process` so that `stop` does not import the HTTP
shell. That the stop path stays clear of `uclone_x.ui` on a real `[cli]` wheel is pinned
by the distribution-install fitness lane, because this environment — every extra
installed — cannot express it. What is pinned here is the command's contract: it signals
only the launcher a dashboard record names, and only while the operating system still
reports that PID as the same process (#927); it says what it stopped and what it inspected;
and it refuses — naming why, and what to do instead — when it cannot tell.

**These tests send real signals if a guard misses.** `stop` used to SIGTERM whatever
listened on the port it was given, and review of #925 measured a mutation under which
these tests reached `os.kill` against two live dashboards on the product default port;
review of #951 then had an earlier revision of this change signal an unrelated process a
spoofer named. So `no_real_signals` sits under every test: an `os.kill` or `os.killpg`
fails the test unless it targets a PID *this test spawned*, and a PID may receive more than
signal 0 only when the test spawned it to be stopped. `subprocess.Popen.send_signal` routes
through `os.kill`, so a `Popen.terminate()` is covered by the same guard; `psutil` is not a
dependency. The test's own cleanup of its children uses the real `os.kill`, captured at
import, on those children alone.

Every port is ephemeral, every process is spawned here, and every record lives under the
per-test `UCLONE_UI_STATE_DIR` the root conftest sets, so no test can reach a developer's
dashboard or another product sharing the machine.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from uclone_x.cli import main
from uclone_x.errors import DashboardNotIdentifiedError, ListeningProcessLookupError
from uclone_x.room.store import ROOM_STORAGE_DIR_ENV_VAR
from uclone_x.shells import ui_process

# Imported here, before any test patches `ui_process`, so `ui.server`'s re-export binds
# the real functions. Imported lazily inside a test, it would bind whatever mock that test
# had installed and keep it for the rest of the session — the leak review of #925 saw.
from uclone_x.ui import server as ui_server

runner = CliRunner()

#: Captured before any test patches `os.kill`; used only to reap this module's own children.
_REAL_KILL = os.kill

#: How long a spawned dashboard may take to serve, or its processes to exit, before failing.
_DASHBOARD_TIMEOUT_S = 60.0


@dataclass
class SignalGuard:
    """What each PID the test spawned may be sent, and what was attempted."""

    probeable: set[int] = field(default_factory=lambda: set[int]())
    stoppable: set[int] = field(default_factory=lambda: set[int]())
    sent: list[tuple[int, int]] = field(default_factory=lambda: list[tuple[int, int]]())
    refused: list[tuple[str, int, int]] = field(
        default_factory=lambda: list[tuple[str, int, int]]()
    )


@pytest.fixture(autouse=True)
def no_real_signals(monkeypatch: pytest.MonkeyPatch) -> Iterator[SignalGuard]:
    """Fail any test whose code path signals a PID it did not spawn for that purpose.

    `pytest.fail` raises a `BaseException`, so neither `stop_ui_server`'s exception
    handling nor `CliRunner`'s exception capture can swallow it; the refusals are also
    re-checked at teardown in case something catches `BaseException`. A test that means to
    observe signals to fake PIDs patches `os.kill` itself, which replaces this guard.
    """
    guard = SignalGuard()

    def _kill(pid: int, sig: int) -> None:
        if pid in guard.stoppable or (sig == 0 and pid in guard.probeable):
            guard.sent.append((pid, sig))
            _REAL_KILL(pid, sig)
            return
        guard.refused.append(("kill", pid, sig))
        pytest.fail(f"test reached a real os.kill({pid}, {sig}); nothing may be signalled")

    def _killpg(pgid: int, sig: int) -> None:
        guard.refused.append(("killpg", pgid, sig))
        pytest.fail(f"test reached a real os.killpg({pgid}, {sig}); nothing may be signalled")

    monkeypatch.setattr(os, "kill", _kill)
    monkeypatch.setattr(os, "killpg", _killpg)
    yield guard
    assert not guard.refused, f"unpermitted signal attempts: {guard.refused}"


Spawn = Callable[..., "subprocess.Popen[bytes]"]


@pytest.fixture
def spawn(no_real_signals: SignalGuard) -> Iterator[Spawn]:
    """Start `python -c <code>` children, registered with the guard, and reap them after.

    A child spawned to be stopped is reaped as soon as it exits, as its real launcher's
    parent would; otherwise it would linger as a zombie still holding its PID.
    """
    children: list[subprocess.Popen[bytes]] = []

    def _spawn(
        code: str, *args: str, stoppable: bool = False, log: Path | None = None
    ) -> subprocess.Popen[bytes]:
        command = [sys.executable, "-c", code, *args]
        if log is None:
            proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        else:
            with log.open("wb") as out:
                proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        children.append(proc)
        if stoppable:
            no_real_signals.stoppable.add(proc.pid)
            threading.Thread(target=proc.wait, daemon=True).start()
        else:
            no_real_signals.probeable.add(proc.pid)
        return proc

    yield _spawn
    for proc in children:
        if proc.poll() is None:
            _REAL_KILL(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                stream.close()


@pytest.fixture
def idle_port() -> Iterator[int]:
    """A port this process has bound but is not listening on, held for the test.

    Ephemeral, so a fixed-port dashboard cannot be on it, and not listening, so nothing
    accepts a connection to it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
        reserved.bind(("127.0.0.1", 0))
        yield reserved.getsockname()[1]


_SLEEPER = "import time; time.sleep(300)"

# Prints its port, then for every line on stdin proves the socket still accepts, from
# inside the child — the unit suite's network guard keeps the test itself from connecting.
_PLAIN_LISTENER = """
import socket, sys
s = socket.socket()
s.bind(("127.0.0.1", 0))
s.listen(8)
print(s.getsockname()[1], flush=True)
for _ in sys.stdin:
    with socket.create_connection(s.getsockname(), timeout=5):
        conn, _peer = s.accept()
        conn.close()
    print("accepted", flush=True)
"""

# The real `start_ui_server`. `_ensure_frontend_built` is replaced because it may run
# `npm run build`, which rewrites the committed `ui_static` bundle. Replaced here by hand
# rather than by the `frontend_build_suppressed` fixture: this runs in a child interpreter,
# which the suite's in-process guard does not reach (#1067). For `--dev`, the
# module's `__file__` is pointed into `root`, where there is no `frontend/`, so no Vite is
# co-spawned and the reload supervisor watches an empty directory.
_DASHBOARD = """
import sys
from pathlib import Path
from uclone_x.ui import server
server._ensure_frontend_built = lambda: None
port, root, dev = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "dev"
if dev:
    server.__file__ = str(root / "src" / "uclone_x" / "ui" / "server.py")
server.start_ui_server(port=port, dev=dev, storage_dir=root / "sessions", workspace_dir=root)
"""


def _free_port() -> int:
    """An ephemeral port, released. Bound in this process, so the network guard allows it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


def _await_serving(port: int, proc: subprocess.Popen[bytes], log: Path) -> None:
    """Block until the dashboard answers `/api/health`, or fail naming its log."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + _DASHBOARD_TIMEOUT_S
    last = "no attempt"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"dashboard exited {proc.returncode} first:\n{log.read_text()[-3000:]}")
        try:
            with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=2.0):
                return
        except OSError as exc:
            last = repr(exc)
        threading.Event().wait(0.1)
    pytest.fail(f"dashboard never served port {port} ({last}):\n{log.read_text()[-3000:]}")


def _children_of(pid: int) -> dict[int, ui_process.ProcessIdentity]:
    """The live children of `pid`, each with the identity `ps` reports for it."""
    listing = subprocess.run(
        ["ps", "-A", "-o", "pid=", "-o", "ppid="], capture_output=True, text=True, check=True
    ).stdout
    children: dict[int, ui_process.ProcessIdentity] = {}
    for line in listing.splitlines():
        child, parent = (int(value) for value in line.split())
        if parent == pid:
            identity = ui_process.process_identity(child)
            if identity is not None:
                children[child] = identity
    return children


def _await_gone(processes: dict[int, ui_process.ProcessIdentity]) -> list[int]:
    """Wait for every process to exit; return those still running at the deadline."""
    deadline = time.monotonic() + _DASHBOARD_TIMEOUT_S / 4
    while True:
        alive = [
            pid
            for pid, identity in processes.items()
            if ui_process.process_identity(pid) == identity
        ]
        if not alive or time.monotonic() >= deadline:
            return alive
        threading.Event().wait(0.1)


@contextmanager
def _http_listener(status: int, body: bytes) -> Generator[int]:
    """Serve `body` with `status` for every GET on an ephemeral loopback port."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(httpd.server_address[1])
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _record_for(
    pid: int, port: int, *, process: ui_process.ProcessIdentity | None = None, instance: str = "i"
) -> ui_process.DashboardRecord:
    """A record naming `pid`, with its real identity unless one is given."""
    identity = process if process is not None else ui_process.process_identity(pid)
    return ui_process.DashboardRecord(
        pid=pid, port=port, host="127.0.0.1", instance=instance, process=identity
    )


def _outcome(*pids: int, port: int) -> ui_process.StopOutcome:
    return ui_process.StopOutcome(port=port, stopped_pids=pids)


def test_ui_stop_reports_the_pids_it_stopped(
    monkeypatch: pytest.MonkeyPatch, idle_port: int
) -> None:
    """A stopped server is named by PID and port."""
    stop = MagicMock(return_value=_outcome(12345, port=idle_port))
    monkeypatch.setattr(ui_process, "stop_ui_server", stop)

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 0, result.output
    stop.assert_called_once_with(port=idle_port)
    assert "Stopped UI dashboard server running on PID(s): 12345" in result.stdout


def test_ui_stop_says_what_it_inspected_when_it_finds_nothing(
    monkeypatch: pytest.MonkeyPatch, idle_port: int
) -> None:
    """Nothing to stop is a stated outcome, and the statement names what was looked at.

    Exit 0 is deliberate: the requested end state — no recorded dashboard on that port —
    holds. But the only evidence is a missing record and the loopback addresses, so a
    dashboard bound to a LAN address, or recorded under another `UCLONE_UI_STATE_DIR`,
    would get the same answer; review of #951 measured exactly that. The message says so
    rather than asserting an absence it never measured (P6).

    Killed by: src/uclone_x/cli/commands/ui.py ::
        f"{record_path}, and nothing accepts connections on 127.0.0.1 or ::1. A dashboard "
    Becomes: f"{record_path}. A dashboard "
    """
    monkeypatch.setattr(ui_process, "stop_ui_server", MagicMock(return_value=_outcome(port=1)))

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 0, result.output
    assert f"No active UI dashboard server found on port {idle_port}" in result.stdout
    assert f"no dashboard record at {ui_process.dashboard_record_path(idle_port)}" in result.stdout
    assert "nothing accepts connections on 127.0.0.1 or ::1" in result.stdout
    assert "bound only to another address is not detected" in result.stdout


def test_ui_stop_on_a_real_idle_port_inspects_it_and_signals_nothing(
    idle_port: int, no_real_signals: SignalGuard
) -> None:
    """End to end through real sockets: an idle port with no record is reported idle, exit 0.

    Unmocked on purpose, and portable: the decision path is a record file, `ps` and
    loopback connects, with no `lsof` (#927).
    """
    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 0, result.output
    assert f"No active UI dashboard server found on port {idle_port}" in result.stdout
    assert no_real_signals.sent == []


def test_ui_stop_refuses_an_unrelated_listener_and_leaves_it_running(
    spawn: Spawn, no_real_signals: SignalGuard
) -> None:
    """#927 itself: a listener no record names is refused, not signalled, with a remedy.

    `ucx ui stop --port 5432` used to SIGTERM a local Postgres. The listener here is a
    plain socket server the test spawned; after the refusal it must still be running and
    still accepting connections, and nothing may have been sent to it at all. The refusal
    names the commands that do record a dashboard — `ucx ui start` does not exist — and how
    to stop an unrecorded one.

    Killed by: src/uclone_x/shells/ui_process.py ::
        if _loopback_accepts_connections(port):
    Becomes: if False:
    """
    listener = spawn(_PLAIN_LISTENER)
    assert listener.stdin is not None and listener.stdout is not None
    port = int(listener.stdout.readline())

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(port)])

    assert result.exit_code == 1, result.output
    assert f"ucx ui stop: refusing to signal anything on port {port}" in result.stderr
    assert "no dashboard record at" in result.stderr
    assert "`ucx ui` or `ucx start`" in result.stderr
    assert "ucx ui start" not in result.stderr
    assert "Ctrl-C" in result.stderr
    assert "No active UI dashboard server" not in result.output
    assert listener.poll() is None
    listener.stdin.write(b"still listening?\n")
    listener.stdin.flush()
    assert listener.stdout.readline() == b"accepted\n"
    assert no_real_signals.sent == []


def test_ui_stop_stops_a_real_dashboard_gracefully_and_nothing_else(
    spawn: Spawn, no_real_signals: SignalGuard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The converse: a dashboard `start_ui_server` launched is identified and stopped.

    The dashboard is the real server in a child process on an ephemeral port, with its
    sessions, rooms and record under `tmp_path`. It records itself; `stop` finds that
    record, confirms through `ps` that the PID is still the process that wrote it, and
    signals exactly that PID — with SIGTERM only.

    A healthy dashboard must also get the time its shutdown needs. Review of #951 measured
    this server with the SIGTERM→SIGKILL wait at 0: SIGKILL followed within milliseconds,
    it exited −9 without running its lifespan shutdown (`room_stack.close()`, which keeps a
    running turn from being lost unsaved), and `stop` still printed "Stopped" with exit 0
    while every test passed. The evidence of a graceful stop is read from the dashboard: it
    logs the end of its lifespan shutdown, and exits by re-raising the SIGTERM it handled.

    Killed by: src/uclone_x/ui/server.py ::
        ui_process.write_dashboard_record(record)
    Becomes: pass
    Killed by: src/uclone_x/shells/ui_process.py ::
        TERMINATE_WAIT_S: float = 5.0
    Becomes: TERMINATE_WAIT_S: float = 0.0
    """
    monkeypatch.setenv(ROOM_STORAGE_DIR_ENV_VAR, str(tmp_path / "rooms"))
    port = _free_port()
    log = tmp_path / "dashboard.log"
    dashboard = spawn(_DASHBOARD, str(port), str(tmp_path), "plain", stoppable=True, log=log)
    _await_serving(port, dashboard, log)

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(port)])

    dashboard.wait(timeout=15)
    output = log.read_text()
    assert result.exit_code == 0, f"{result.output}\n{output[-3000:]}"
    assert f"PID(s): {dashboard.pid} (port {port})" in result.stdout
    assert (dashboard.pid, signal.SIGTERM) in no_real_signals.sent
    assert (dashboard.pid, signal.SIGKILL) not in no_real_signals.sent
    assert {pid for pid, _ in no_real_signals.sent} == {dashboard.pid}
    assert dashboard.returncode == -signal.SIGTERM, output[-3000:]
    assert "Application shutdown complete" in output, output[-3000:]
    assert not ui_process.dashboard_record_path(port).exists()


def test_ui_stop_stops_a_real_dev_dashboard_by_its_launcher_alone(
    spawn: Spawn, no_real_signals: SignalGuard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--dev`: the port is served by a reload worker; `stop` signals only the launcher.

    The launcher is the real `start_ui_server(dev=True)` in a child process, and the worker
    is uvicorn's, spawned by it. Only the launcher's PID is signalled — the worker is never
    registered with the guard, so a signal to it fails the test — and both must exit,
    because the launcher relays SIGTERM to its worker. Measured on Darwin 25: SIGKILL to the
    launcher instead orphans the worker, which keeps serving the port. The mutation sends
    exactly that.

    Killed by: src/uclone_x/shells/ui_process.py ::
        os.kill(record.pid, signal.SIGTERM)
    Becomes: os.kill(record.pid, signal.SIGKILL)
    """
    monkeypatch.setenv(ROOM_STORAGE_DIR_ENV_VAR, str(tmp_path / "rooms"))
    (tmp_path / "src").mkdir()
    port = _free_port()
    log = tmp_path / "dashboard.log"
    launcher = spawn(_DASHBOARD, str(port), str(tmp_path), "dev", stoppable=True, log=log)
    _await_serving(port, launcher, log)
    workers = _children_of(launcher.pid)
    assert workers, f"the --dev launcher has no worker:\n{log.read_text()[-3000:]}"

    try:
        result = runner.invoke(main.app, ["ui", "stop", "--port", str(port)])

        assert result.exit_code == 0, f"{result.output}\n{log.read_text()[-3000:]}"
        assert f"PID(s): {launcher.pid} (port {port})" in result.stdout
        launcher.wait(timeout=15)
        assert _await_gone(workers) == [], "the reload worker outlived its launcher"
        assert {pid for pid, _ in no_real_signals.sent} == {launcher.pid}
        assert not ui_process.dashboard_record_path(port).exists()
    finally:
        # On a failure, take down what this test started: the launcher (its own child)
        # first, then any worker still verifiably the one it spawned.
        if launcher.poll() is None:
            _REAL_KILL(launcher.pid, signal.SIGTERM)
            launcher.wait(timeout=15)
        for pid, identity in workers.items():
            if ui_process.process_identity(pid) == identity:
                _REAL_KILL(pid, signal.SIGKILL)


def test_ui_stop_signals_the_recorded_launcher_and_never_a_pid_the_port_names(
    spawn: Spawn, no_real_signals: SignalGuard
) -> None:
    """What answers on the port is not consulted about what to signal.

    Review of #951 had a process on the port answer `{pid: y, parent_pid: <launcher>}` and
    watched `stop` SIGTERM and SIGKILL the unrelated `y`. Here the record is valid, a
    spoofer on the port names an unrelated spawned process as the dashboard's child in that
    same shape, and only the recorded launcher may be signalled; the named process must
    survive having received nothing at all. No mutation of the current code restores the
    defect: the code that read the port's answer is gone, and this test is what keeps it so.
    """
    launcher = spawn(_SLEEPER, stoppable=True)
    unrelated = spawn(_SLEEPER)
    claim = {"product": "uclone-x-dashboard", "pid": unrelated.pid, "parent_pid": launcher.pid}

    with _http_listener(200, json.dumps(claim).encode()) as port:
        ui_process.write_dashboard_record(_record_for(launcher.pid, port))
        result = runner.invoke(main.app, ["ui", "stop", "--port", str(port)])

    assert result.exit_code == 0, result.output
    assert f"PID(s): {launcher.pid} (port {port})" in result.stdout
    launcher.wait(timeout=15)
    assert unrelated.poll() is None
    assert {pid for pid, _ in no_real_signals.sent} == {launcher.pid}


def test_ui_stop_removes_a_stale_record_and_says_so(
    spawn: Spawn, no_real_signals: SignalGuard, idle_port: int
) -> None:
    """A record whose PID is gone is removed, and the output says it was.

    A dashboard killed with SIGKILL cannot remove its own record. Nothing is listening
    here, so the end state holds and the exit is 0 — but the stale record is named rather
    than dropped silently, and only signal 0 went to its PID.

    Killed by: src/uclone_x/shells/ui_process.py ::
        except ProcessLookupError:  # exited; its PID is free for reuse
    Becomes: except ChildProcessError:  # exited; its PID is free for reuse
    """
    gone = spawn("pass")
    gone.wait(timeout=10)
    identity = ui_process.ProcessIdentity(started="Thu Jan  1 00:00:00 1970", command="gone")
    path = ui_process.write_dashboard_record(_record_for(gone.pid, idle_port, process=identity))

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 0, result.output
    assert f"Removed a stale dashboard record: PID {gone.pid} is not running" in result.stdout
    assert f"No active UI dashboard server found on port {idle_port}" in result.stdout
    assert not path.exists()
    assert no_real_signals.sent == [(gone.pid, 0)]


@pytest.mark.parametrize("differs", ["start-time", "command"])
def test_ui_stop_treats_a_reused_pid_as_stale_even_with_a_spoofer_on_the_port(
    spawn: Spawn, no_real_signals: SignalGuard, differs: str
) -> None:
    """A recorded PID that now names another process is never signalled.

    The dashboard was SIGKILLed and its PID reused by an unrelated process — here, one the
    test spawned, whose real start time or command line differs from the record's. A
    spoofer holds the port and answers in the shape a dashboard once did. The record is
    removed as stale, the listener with no valid record is refused, and the unrelated
    process survives having received only the liveness probe.

    Killed by: src/uclone_x/shells/ui_process.py ::
        if current.started != record.process.started or current.command != record.process.command:
    Becomes: if False:
    """
    reused = spawn(_SLEEPER)
    actual = ui_process.process_identity(reused.pid)
    assert actual is not None
    recorded = ui_process.ProcessIdentity(
        started="Thu Jan  1 00:00:00 1970" if differs == "start-time" else actual.started,
        command="python -m uclone_x.cli.main ui" if differs == "command" else actual.command,
    )
    claim = {"product": "uclone-x-dashboard", "pid": reused.pid, "parent_pid": reused.pid}

    with _http_listener(200, json.dumps(claim).encode()) as port:
        path = ui_process.write_dashboard_record(_record_for(reused.pid, port, process=recorded))
        result = runner.invoke(main.app, ["ui", "stop", "--port", str(port)])

    assert result.exit_code == 1, result.output
    assert f"Removed a stale dashboard record: PID {reused.pid}" not in result.stdout
    assert f"a stale record for PID {reused.pid} was removed" in result.stderr
    assert f"refusing to signal anything on port {port}" in result.stderr
    assert not path.exists()
    assert reused.poll() is None
    assert no_real_signals.sent == [(reused.pid, 0)]


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"pid": -1, "port": PORT, "host": "127.0.0.1", "instance": "i", "process": null}',
        '{"pid": 4242, "port": PORT, "host": "127.0.0.1", "instance": "i", "process": {"a": 1}}',
    ],
    ids=["unparseable", "non-positive-pid", "malformed-process"],
)
def test_ui_stop_refuses_a_record_it_cannot_read(
    idle_port: int, content: str, no_real_signals: SignalGuard
) -> None:
    """A malformed record is a refusal naming the file and the remedy; the file is kept.

    A PID of `-1` or `0` must never reach `os.kill`: signal 0 to `-1` probes every process
    the user owns, and the SIGTERM after it would terminate them all.

    Killed by: src/uclone_x/shells/ui_process.py ::
        or pid < 1
    Becomes: or False
    """
    path = ui_process.dashboard_record_path(idle_port)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(content.replace("PORT", str(idle_port)), encoding="utf-8")
    path.chmod(0o600)

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 1, result.output
    assert f"refusing to signal anything on port {idle_port}" in result.stderr
    assert f"delete {path}" in result.stderr.replace("\n", "")
    assert path.exists()
    assert no_real_signals.sent == []


def test_ui_stop_refuses_a_record_others_could_have_written(
    idle_port: int, no_real_signals: SignalGuard
) -> None:
    """A record is an instruction to signal a PID, so one others can write is not followed.

    Start time and command line are readable by any user through `ps`, so a record naming
    them proves nothing about who wrote it. The record here names a PID nobody spawned; were
    it followed, the guard would refuse the probe.

    Killed by: src/uclone_x/shells/ui_process.py ::
        return info.st_uid != os.getuid() or bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
    Becomes: return info.st_uid != os.getuid()
    """
    identity = ui_process.ProcessIdentity(started="Thu Jan  1 00:00:00 1970", command="x")
    path = ui_process.write_dashboard_record(_record_for(4242, idle_port, process=identity))
    path.chmod(0o666)

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 1, result.output
    assert "writable by others" in result.stderr
    assert path.exists()
    assert no_real_signals.sent == []


def test_ui_stop_refuses_a_record_without_a_process_identity(
    spawn: Spawn, idle_port: int, no_real_signals: SignalGuard
) -> None:
    """A launcher `ps` could not describe left a record that cannot tell it from a reuse.

    Its PID is running, so the record is not provably stale either. The refusal says how to
    stop the dashboard instead, and the record is kept.

    Killed by: src/uclone_x/shells/ui_process.py ::
        if record.process is None:
    Becomes: if False:
    """
    running = spawn(_SLEEPER)
    record = ui_process.DashboardRecord(
        pid=running.pid, port=idle_port, host="127.0.0.1", instance="i", process=None
    )
    path = ui_process.write_dashboard_record(record)

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 1, result.output
    assert "does not say which process it was" in result.stderr
    assert "Ctrl-C" in result.stderr
    assert path.exists()
    assert running.poll() is None
    assert no_real_signals.sent == [(running.pid, 0)]


@pytest.mark.parametrize("failure", ["not-runnable", "errored"])
def test_ui_stop_refuses_when_ps_cannot_confirm_the_launcher(
    spawn: Spawn,
    idle_port: int,
    no_real_signals: SignalGuard,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """No answer from `ps` is not "the process is gone" and not "it is the launcher" (P6).

    `ps` exits 1 with nothing on stdout or stderr for a PID it does not list — and exits 1
    too, with an error, when it rejects the query. Only the first is an answer.

    Killed by: src/uclone_x/shells/ui_process.py ::
        if res.returncode == 1 and not value and not res.stderr.strip():
    Becomes: if res.returncode != 0:
    """
    running = spawn(_SLEEPER)
    path = ui_process.write_dashboard_record(_record_for(running.pid, idle_port))
    if failure == "not-runnable":
        fake = MagicMock(side_effect=FileNotFoundError("ps"))
    else:
        fake = MagicMock(return_value=subprocess.CompletedProcess([], 1, "", "ps: bad option\n"))
    monkeypatch.setattr(ui_process, "_run_ps", fake)

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 1, result.output
    assert f"cannot confirm PID {running.pid} is the dashboard recorded at" in result.stderr
    assert "Ctrl-C" in result.stderr
    assert path.exists()
    assert running.poll() is None
    assert no_real_signals.sent == [(running.pid, 0)]


_IGNORES_SIGTERM = """
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("ready", flush=True)
time.sleep(300)
"""


def test_ui_stop_does_not_deny_the_sigterm_when_ps_fails_during_the_wait(
    spawn: Spawn,
    idle_port: int,
    no_real_signals: SignalGuard,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ps` failing after SIGTERM is reported as unconfirmed, never as "nothing signalled".

    The launcher here ignores SIGTERM, so the wait has to ask `ps` about it; `ps` answers
    the identity check before the signal and fails from then on. SIGKILL is not sent on
    an unconfirmed identity, and the record is kept.

    Killed by: src/uclone_x/shells/ui_process.py ::
        raise DashboardStopUnconfirmedError(
    Becomes: raise DashboardNotIdentifiedError(
    """
    launcher = spawn(_IGNORES_SIGTERM, stoppable=True, log=None)
    assert launcher.stdout is not None
    assert launcher.stdout.readline() == b"ready\n"
    path = ui_process.write_dashboard_record(_record_for(launcher.pid, idle_port))
    real_run_ps = ui_process._run_ps  # pyright: ignore[reportPrivateUsage]
    calls: list[str] = []

    def failing_after_the_signal(pid: int, field: str) -> subprocess.CompletedProcess[str]:
        calls.append(field)
        if len(calls) > 2:
            raise FileNotFoundError("ps")
        return real_run_ps(pid, field)

    monkeypatch.setattr(ui_process, "_run_ps", failing_after_the_signal)

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 1, result.output
    assert f"SIGTERM was sent to PID {launcher.pid}" in result.stderr
    assert "unconfirmed" in result.stderr
    assert "Nothing was signalled" not in result.stderr
    assert (launcher.pid, signal.SIGTERM) in no_real_signals.sent
    assert (launcher.pid, signal.SIGKILL) not in no_real_signals.sent
    assert launcher.poll() is None
    assert path.exists()


@pytest.mark.parametrize(
    ("after_the_wait", "expected"),
    [
        (None, [signal.SIGTERM, signal.SIGKILL]),
        ("now belongs to another process", [signal.SIGTERM]),
    ],
    ids=["still-the-launcher", "pid-reused-meanwhile"],
)
def test_stop_ui_server_escalates_only_against_the_same_launcher(
    monkeypatch: pytest.MonkeyPatch,
    idle_port: int,
    after_the_wait: str | None,
    expected: list[int],
) -> None:
    """SIGKILL goes only to a launcher that outlived the wait *and* is still itself.

    Once the launcher exits its PID is free, so a SIGKILL sent on the strength of the PID
    alone could land on whatever reused it. The PID is a fake, observed through a fake
    `os.kill`; identity is scripted: the launcher before SIGTERM, then the given answer.
    A SIGKILL is also recorded on the outcome, with what the port looked like afterwards.

    Killed by: src/uclone_x/shells/ui_process.py ::
        while _staleness(record) is None:
    Becomes: while True:
    Killed by: src/uclone_x/shells/ui_process.py ::
        still_accepting: bool | None = _loopback_accepts_connections(port)
    Becomes: still_accepting: bool | None = False
    """
    killed: list[tuple[int, int]] = []
    answers = iter([None, after_the_wait, after_the_wait])

    def scripted_staleness(record: ui_process.DashboardRecord) -> str | None:
        del record
        return next(answers)

    def fake_kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))

    monkeypatch.setattr(ui_process, "_staleness", scripted_staleness)
    monkeypatch.setattr(os, "kill", fake_kill)
    monkeypatch.setattr(ui_process, "TERMINATE_WAIT_S", 0.0)
    # After a SIGKILL the port is re-checked: here something (an orphaned worker) serves it.
    port_check = MagicMock(return_value=True)
    monkeypatch.setattr(ui_process, "_loopback_accepts_connections", port_check)
    identity = ui_process.ProcessIdentity(started="s", command="c")
    path = ui_process.write_dashboard_record(_record_for(3333, idle_port, process=identity))

    outcome = ui_process.stop_ui_server(idle_port)

    assert outcome.stopped_pids == (3333,)
    assert killed == [(3333, sig) for sig in expected]
    assert outcome.killed is (signal.SIGKILL in expected)
    assert outcome.still_accepting is (True if signal.SIGKILL in expected else None)
    assert port_check.call_count == (1 if signal.SIGKILL in expected else 0)
    assert not path.exists()


@pytest.mark.parametrize("still_accepting", [True, None, False])
def test_ui_stop_does_not_call_a_killed_dashboard_stopped(
    monkeypatch: pytest.MonkeyPatch, idle_port: int, still_accepting: bool | None
) -> None:
    """A SIGKILL is reported as one, exit 1, and a port still served is called out (P6).

    A launcher that needed SIGKILL did not run its shutdown, and under `--dev` its reload
    worker can outlive it and keep serving. "Stopped", exit 0, would claim the graceful
    outcome that did not happen.

    Killed by: src/uclone_x/cli/commands/ui.py ::
        if stopped.killed:
    Becomes: if False:
    """
    outcome = ui_process.StopOutcome(
        port=idle_port, stopped_pids=(3333,), killed=True, still_accepting=still_accepting
    )
    monkeypatch.setattr(ui_process, "stop_ui_server", MagicMock(return_value=outcome))

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 1, result.output
    assert "Stopped" not in result.output
    assert "were sent SIGKILL" in result.stderr
    assert "graceful shutdown did not complete" in result.stderr
    assert ("may have been orphaned" in result.stderr) is (still_accepting is not False)


def test_a_record_is_private_and_its_directory_is_left_as_the_user_made_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, idle_port: int
) -> None:
    """The file is `0600`; a directory this code creates is `0700`; an existing one is
    never re-permissioned — `UCLONE_UI_STATE_DIR` may name a directory the user owns and
    uses for other things.

    Killed by: src/uclone_x/shells/ui_process.py ::
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    Becomes: fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    """
    identity = ui_process.ProcessIdentity(started="s", command="c")
    previous = os.umask(0o022)
    try:
        fresh = tmp_path / "fresh"
        monkeypatch.setenv(ui_process.DASHBOARD_STATE_DIR_ENV_VAR, str(fresh))
        created = ui_process.write_dashboard_record(_record_for(3333, idle_port, process=identity))

        existing = tmp_path / "existing"
        existing.mkdir(mode=0o755)
        existing.chmod(0o755)
        monkeypatch.setenv(ui_process.DASHBOARD_STATE_DIR_ENV_VAR, str(existing))
        kept = ui_process.write_dashboard_record(_record_for(3333, idle_port, process=identity))
    finally:
        os.umask(previous)

    assert stat.S_IMODE(created.stat().st_mode) == 0o600
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o700
    assert stat.S_IMODE(kept.stat().st_mode) == 0o600
    assert stat.S_IMODE(existing.stat().st_mode) == 0o755


def test_a_record_is_not_written_into_a_directory_others_can_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, idle_port: int
) -> None:
    """Refused, and the directory's mode untouched, rather than silently narrowed.

    Killed by: src/uclone_x/shells/ui_process.py ::
        if _untrusted(existing):
    Becomes: if False:
    """
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    monkeypatch.setenv(ui_process.DASHBOARD_STATE_DIR_ENV_VAR, str(shared))
    identity = ui_process.ProcessIdentity(started="s", command="c")

    with pytest.raises(DashboardNotIdentifiedError, match="writable by others"):
        ui_process.write_dashboard_record(_record_for(3333, idle_port, process=identity))

    assert stat.S_IMODE(shared.stat().st_mode) == 0o777
    assert list(shared.iterdir()) == []


def test_ui_stop_refuses_a_record_owned_by_another_user(
    idle_port: int, monkeypatch: pytest.MonkeyPatch, no_real_signals: SignalGuard
) -> None:
    """Ownership is checked on its own, not only writability.

    Creating a file owned by another user needs root, so this user's ID is what is
    changed: to the check, every file here now belongs to someone else. The record names a
    PID nobody spawned; were it followed, the guard would refuse the probe.

    Killed by: src/uclone_x/shells/ui_process.py ::
        return info.st_uid != os.getuid() or bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
    Becomes: return bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
    """
    identity = ui_process.ProcessIdentity(started="Thu Jan  1 00:00:00 1970", command="x")
    path = ui_process.write_dashboard_record(_record_for(4242, idle_port, process=identity))
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 1, result.output
    assert "not owned by this user" in result.stderr
    assert path.exists()
    assert no_real_signals.sent == []


@pytest.mark.parametrize("where", ["record-is-a-symlink", "directory-writable-by-others"])
def test_ui_stop_refuses_a_record_reached_through_an_unsafe_path(
    tmp_path: Path, idle_port: int, no_real_signals: SignalGuard, where: str
) -> None:
    """A symlinked record, or a record in a directory others can write, is not followed.

    Either lets another user substitute the record between a check and the read. The
    record is otherwise valid and names a PID nobody spawned, so following it would reach
    the guard.

    Killed by: src/uclone_x/shells/ui_process.py ::
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    Becomes: fd = os.open(path, os.O_RDONLY)
    """
    identity = ui_process.ProcessIdentity(started="Thu Jan  1 00:00:00 1970", command="x")
    path = ui_process.write_dashboard_record(_record_for(4242, idle_port, process=identity))
    if where == "record-is-a-symlink":
        elsewhere = tmp_path / "elsewhere.json"
        path.rename(elsewhere)
        path.symlink_to(elsewhere)
        expected = "a symlink"
    else:
        path.parent.chmod(0o777)
        expected = "record directory"

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 1, result.output
    assert expected in result.stderr
    assert no_real_signals.sent == []


def test_removing_a_record_never_removes_a_later_starts_record(idle_port: int) -> None:
    """A dashboard exiting after another start took its port leaves that start's record.

    Killed by: src/uclone_x/shells/ui_process.py ::
        if current is not None and current.instance == record.instance:
    Becomes: if current is not None:
    """
    identity = ui_process.ProcessIdentity(started="s", command="c")
    earlier = _record_for(3333, idle_port, process=identity, instance="earlier")
    later = _record_for(4444, idle_port, process=identity, instance="later")
    ui_process.write_dashboard_record(earlier)
    path = ui_process.write_dashboard_record(later)

    ui_process.remove_dashboard_record(earlier)
    assert ui_process.read_dashboard_record(idle_port) == later

    ui_process.remove_dashboard_record(later)
    assert not path.exists()


@pytest.mark.parametrize("port", ["0", "-1", "65536", "70000"])
def test_ui_stop_rejects_a_port_outside_the_tcp_range(port: str) -> None:
    """`--port 70000` is a usage error, not "no server found on port 70000".

    Exit 2 with Click's range message is what says the *option* rejected the value. The
    helper refuses these as well, but only after Typer has accepted them, and it exits 1
    rather than 2. The mutation drops the upper bound, so the high ports reach the helper
    and exit 1.

    Killed by: src/uclone_x/cli/commands/ui.py ::
        max=_MAX_TCP_PORT,
    Becomes: max=None,
    """
    result = runner.invoke(main.app, ["ui", "stop", "--port", port])

    assert result.exit_code == 2, result.output
    assert "No active UI dashboard server" not in result.output
    assert "1<=x<=65535" in result.output


@pytest.mark.parametrize("port", [0, -1, 65536, 70000])
def test_stop_ui_server_rejects_a_port_outside_the_tcp_range(
    monkeypatch: pytest.MonkeyPatch, port: int
) -> None:
    """The helper refuses too, before any record or socket: the CLI is not its only caller.

    Killed by: src/uclone_x/shells/ui_process.py ::
        if not MIN_TCP_PORT <= port <= MAX_TCP_PORT:
    Becomes: if False:
    """
    opened = MagicMock()
    monkeypatch.setattr(ui_process, "_probe_socket", opened)

    with pytest.raises(ListeningProcessLookupError) as caught:
        ui_process.stop_ui_server(port)

    assert caught.value.port == port
    assert "1-65535" in str(caught.value)
    opened.assert_not_called()


class _RefusingSocket:
    """A socket whose connect reports `errno_value`, for classifying probe outcomes."""

    def __init__(self, errno_value: int) -> None:
        self._errno = errno_value

    def __enter__(self) -> _RefusingSocket:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def settimeout(self, timeout: float) -> None:
        del timeout

    def connect_ex(self, address: object) -> int:
        del address
        return self._errno


def _refusing_socket_factory(errno_value: int) -> Callable[[int], _RefusingSocket]:
    def _factory(family: int) -> _RefusingSocket:
        del family
        return _RefusingSocket(errno_value)

    return _factory


def test_stop_ui_server_treats_only_an_unaccepted_connect_as_nothing_listening(
    monkeypatch: pytest.MonkeyPatch, idle_port: int
) -> None:
    """A connect that fails for a reason that is not an answer leaves the port uninspected.

    `EACCES` says nothing about whether a process holds the port, so it must not read as
    "nothing is listening" — the substitution #881 removed from the `lsof` reading, which
    this probe replaced (P6).

    Killed by: src/uclone_x/shells/ui_process.py ::
        if result in _NOTHING_ACCEPTED or (family == socket.AF_INET6 and result in _NO_IPV6):
    Becomes: if result != 0:
    """
    monkeypatch.setattr(ui_process, "_probe_socket", _refusing_socket_factory(errno.EACCES))

    with pytest.raises(ListeningProcessLookupError) as caught:
        ui_process.stop_ui_server(idle_port)

    assert caught.value.port == idle_port


def test_ui_stop_fails_naming_the_lookup_when_the_port_cannot_be_inspected(
    monkeypatch: pytest.MonkeyPatch, idle_port: int
) -> None:
    """The command exits non-zero and says why, instead of claiming nothing was running.

    Killed by: src/uclone_x/cli/commands/ui.py ::
        except ListeningProcessLookupError as exc:
    Becomes: except LookupError as exc:
    """
    monkeypatch.setattr(ui_process, "_probe_socket", _refusing_socket_factory(errno.EACCES))

    result = runner.invoke(main.app, ["ui", "stop", "--port", str(idle_port)])

    assert result.exit_code == 1, result.output
    expected = f"ucx ui stop: cannot tell whether anything is listening on port {idle_port}"
    assert expected in result.stderr
    assert "No active UI dashboard server" not in result.output


def test_the_signal_guard_refuses_what_the_test_did_not_spawn_to_stop(
    spawn: Spawn, idle_port: int, no_real_signals: SignalGuard
) -> None:
    """Positive control: without it, the guard could be a no-op and every test above safe
    only by luck. A record naming a PID the test never spawned reaches `os.kill` and is
    refused; so is `Popen.terminate()` on a child spawned only to be probed.
    """
    identity = ui_process.ProcessIdentity(started="s", command="c")
    ui_process.write_dashboard_record(_record_for(4242, idle_port, process=identity))
    with pytest.raises(pytest.fail.Exception, match=r"os\.kill\(4242, 0\)"):
        ui_process.stop_ui_server(idle_port)

    probed_only = spawn(_SLEEPER)
    with pytest.raises(pytest.fail.Exception, match=rf"os\.kill\({probed_only.pid}, "):
        probed_only.terminate()

    assert no_real_signals.refused == [
        ("kill", 4242, 0),
        ("kill", probed_only.pid, signal.SIGTERM),
    ]
    assert probed_only.poll() is None
    no_real_signals.refused.clear()


def test_ui_server_reexports_the_shell_helper_rather_than_copying_it() -> None:
    """`uclone_x.ui.server` keeps its public name, bound to the one implementation."""
    assert ui_server.stop_ui_server is ui_process.stop_ui_server
