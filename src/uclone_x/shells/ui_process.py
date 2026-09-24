"""Record, identify and stop the process serving the developer UI, with the standard library only.

`ucx ui stop` needs a PID it can trust and a signal to send it — nothing from FastAPI or
uvicorn. These helpers used to live in `uclone_x.ui.server`, and importing that module
runs `uclone_x/ui/__init__.py`, which refuses to load without the `http` extra. So a
`uclone-x[cli]` install could not stop a dashboard, for want of a web framework it was not
going to use (#881).

## What `stop` is allowed to signal (#927)

`stop` used to SIGTERM whatever `lsof` said was listening on the port, so
`ucx ui stop --port 5432` would have stopped a local Postgres. It now signals exactly one
process: the one that launched the dashboard, identified by evidence the operating system
holds, not by anything a process on the port says about itself.

1. **A dashboard record.** `start_ui_server` — behind both `ucx ui` and `ucx start` —
   writes `dashboard-<port>.json` under `~/.uclone/ui` (`UCLONE_UI_STATE_DIR` redirects it)
   before it serves, and removes it on the way out. The record names the launcher's PID
   and that process's **identity as `ps` reports it**: its start time and its command line.
   The file is `0600`, in a directory created `0700` (an existing directory is never
   re-permissioned). A record that is a symlink, is not owned by this user, or is writable
   by anyone else — or sits in a directory that is — is refused: a record is an instruction
   to signal a PID.
2. **The same process, now.** `stop` asks `ps` for the recorded PID's start time and
   command line again. Equal: it is the process that wrote the record, so it is signalled.
   A PID that is gone, or that now names a process with a different start time or command
   line, was reused — the record is stale, it is removed, and the output says so.

The launcher is the only target. Under `--dev` a reload worker serves the port as the
launcher's child; the launcher relays SIGTERM to it, so `stop` never signals the worker,
and waits for the launcher to exit before escalating to SIGKILL. A SIGKILL is an outcome of
its own (`StopOutcome.killed`), never reported as a stop: the launcher's graceful shutdown
did not run, and a `--dev` worker may still be serving.

**Nothing the port answers is trusted.** An earlier revision also required an HTTP identity
answer carrying a nonce, and signalled a child PID the answer named. The dashboard served
that nonce to anyone who asked — any local process, any open web page, the LAN under
`0.0.0.0` — so after a crash and PID reuse, a process holding the port could replay it and
have `stop` kill an unrelated process it named (review of #951). The start-time check
defeats PID reuse without a secret, so the route, the nonce and the self-reported PIDs
are gone. What the port is still asked, only when no record exists, is whether anything
accepts a loopback connection there; that decides the exit status and the message, never
whether anything is signalled.

**`lsof` is not used.** `ps -o lstart= -p` and `ps -o command= -p` are present on macOS and
on Linux procps; where `ps` cannot answer, `stop` refuses rather than guesses (P6).

Placement mirrors `entry.py` beside it, for the same two reasons:

* Not under `uclone_x.ui`. Any module there runs the package's `http` guard first.
* Not at the top of `uclone_x`. A bare module beside `errors.py` counts as kernel under
  the layering rule, and signalling a local server process is a shell's concern.

`uclone_x.ui.server` re-exports `stop_ui_server` rather than keeping a copy, so there is
one implementation. The import graph of the stop path is pinned by the distribution-install
fitness lane, which runs `ucx ui stop` out of the built wheel with the `http` extra
withheld.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import secrets
import signal
import socket
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from uclone_x.errors import (
    DashboardNotIdentifiedError,
    DashboardStopUnconfirmedError,
    ListeningProcessLookupError,
)

#: The TCP port range, enforced before anything is read or connected to.
MIN_TCP_PORT: int = 1
MAX_TCP_PORT: int = 65535

#: Redirects where dashboard records are kept, as `UCLONE_SESSION_DIR` redirects sessions.
DASHBOARD_STATE_DIR_ENV_VAR = "UCLONE_UI_STATE_DIR"
DEFAULT_DASHBOARD_STATE_DIR = Path.home() / ".uclone" / "ui"

#: The address `ucx ui` bound the dashboard to, for `create_ui_app` built through uvicorn's
#: factory import string, which cannot carry arguments (#1413). Here rather than beside the
#: app so reading its name does not need the `http` extra.
UI_BIND_HOST_ENV_VAR = "UCLONE_UI_BIND_HOST"

#: How to stop a dashboard `stop` will not signal. Named in every refusal.
STOP_IT_YOURSELF = "press Ctrl-C in the terminal running `ucx ui` or `ucx start`"

#: Seconds the launcher gets to exit after SIGTERM before SIGKILL. Longer than the server's
#: 2 s graceful shutdown, which the earlier 0.3 s cut off. A `--dev` launcher must receive
#: SIGTERM, not SIGKILL, to take its reload worker down — SIGKILL orphans a worker that keeps
#: serving the port (measured on Darwin 25) — and it relays SIGTERM within milliseconds.
TERMINATE_WAIT_S: float = 5.0

#: Seconds between checks while waiting for the launcher to exit. It is not this process's
#: child, so there is no exit event to wait on; each check re-reads its identity.
_EXIT_POLL_S: float = 0.1

#: Seconds a loopback connect may take before it counts as not accepted. A listener with
#: room in its backlog completes a loopback handshake in well under a millisecond.
_CONNECT_TIMEOUT_S: float = 0.5

#: Connect outcomes that mean nothing accepted the connection. A refusal is the usual one.
#: The timeout is macOS: measured on Darwin 25, a SYN to a port that is bound but not
#: listening is dropped rather than refused, so the connect waits out its timeout. Either
#: way no process accepted a connection on the port, which is the question asked here.
_NOTHING_ACCEPTED = frozenset(
    {errno.ECONNREFUSED, errno.EAGAIN, errno.EWOULDBLOCK, errno.ETIMEDOUT}
)

#: Connect errors that mean "this host has no IPv6 loopback", which nothing can listen on.
_NO_IPV6 = frozenset({errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, errno.ENETUNREACH})

_LOOPBACK_ADDRESSES: tuple[tuple[int, str], ...] = (
    (socket.AF_INET, "127.0.0.1"),
    (socket.AF_INET6, "::1"),
)

#: `ps` output must not vary with the caller's locale or time zone, or a launcher and a
#: `stop` run from different shells would read one process as two.
_PS_ENV_OVERRIDES: dict[str, str] = {"LC_ALL": "C", "TZ": "UTC0"}


@dataclass(frozen=True)
class ProcessIdentity:
    """What the operating system says a PID is: when it started, and with what command.

    A PID can be reused once its process exits; the pair cannot, short of a new process
    started in the same second with the same command line.
    """

    started: str
    command: str


@dataclass(frozen=True)
class DashboardRecord:
    """What `start` wrote about the dashboard it launched on one port.

    `instance` tells two starts on one port apart; it is not a secret and is never served.
    `process` is `None` when `ps` could not describe the launcher, and such a record is
    refused rather than trusted.
    """

    pid: int
    port: int
    host: str
    instance: str
    process: ProcessIdentity | None


@dataclass(frozen=True)
class StopOutcome:
    """What `stop_ui_server` did on `port`."""

    port: int
    stopped_pids: tuple[int, ...]
    #: The PID named by a stale record that was removed, if there was one.
    stale_pid: int | None = None
    #: Why that record was stale, phrased to follow "PID <n> ".
    stale_reason: str | None = None
    #: The launcher outlived SIGTERM by `TERMINATE_WAIT_S` and was sent SIGKILL, so its
    #: graceful shutdown did not complete.
    killed: bool = False
    #: After a SIGKILL, whether anything still accepts loopback connections on the port —
    #: an orphaned `--dev` reload worker, for instance. `None` when not checked.
    still_accepting: bool | None = None


def default_dashboard_state_dir() -> Path:
    """Where dashboard records live, honouring `UCLONE_UI_STATE_DIR`."""
    override = os.environ.get(DASHBOARD_STATE_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_DASHBOARD_STATE_DIR


def dashboard_record_path(port: int) -> Path:
    """The record file for the dashboard on `port`."""
    return default_dashboard_state_dir() / f"dashboard-{port}.json"


class ProcessIdentityUnavailableError(OSError):
    """`ps` could not say whether, or as what, a PID is running."""


def process_identity(pid: int) -> ProcessIdentity | None:
    """The identity `ps` reports for `pid`; `None` when no such process exists.

    Raises `ProcessIdentityUnavailableError` when `ps` cannot be run or answers in any
    other way: an unanswered question is not evidence that the process is gone (P6).
    """
    started = _ps_field(pid, "lstart")
    if started is None:
        return None
    command = _ps_field(pid, "command")
    if command is None:
        return None
    return ProcessIdentity(started=started, command=command)


def _run_ps(pid: int, field: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ps", "-ww", "-o", f"{field}=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **_PS_ENV_OVERRIDES},
    )


def _ps_field(pid: int, field: str) -> str | None:
    try:
        res = _run_ps(pid, field)
    except OSError as exc:
        raise ProcessIdentityUnavailableError(f"`ps` could not be run ({exc})") from exc
    value = res.stdout.strip()
    if res.returncode == 0 and value:
        return value
    if res.returncode == 1 and not value and not res.stderr.strip():
        return None
    detail = res.stderr.strip().splitlines()[0] if res.stderr.strip() else "no error text"
    raise ProcessIdentityUnavailableError(
        f"`ps -o {field}= -p {pid}` exited {res.returncode} ({detail})"
    )


def new_dashboard_record(port: int, host: str) -> tuple[DashboardRecord, str | None]:
    """A record for the dashboard this process is about to serve on `port`.

    The second value says why the record carries no process identity, when `ps` could not
    describe this process. Such a record still marks the port as a dashboard's, but `stop`
    refuses to signal on it, so the caller should say so.
    """
    problem: str | None = None
    identity: ProcessIdentity | None = None
    try:
        identity = process_identity(os.getpid())
    except ProcessIdentityUnavailableError as exc:
        problem = str(exc)
    else:
        if identity is None:
            problem = f"`ps` does not list this process ({os.getpid()})"
    record = DashboardRecord(
        pid=os.getpid(),
        port=port,
        host=host,
        instance=secrets.token_hex(16),
        process=identity,
    )
    return record, problem


def _untrusted(info: os.stat_result) -> bool:
    """Whether a file or directory is owned by another user or writable by anyone else."""
    return info.st_uid != os.getuid() or bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def write_dashboard_record(record: DashboardRecord) -> Path:
    """Write `record` atomically as a `0600` file, replacing any record for the port.

    A record already there is stale by construction: the port was free for this start.

    The directory is created `0700` when it does not exist. An existing one — possibly a
    directory the user named with `UCLONE_UI_STATE_DIR` — is never re-permissioned: one
    owned by another user or writable by group or others raises
    `DashboardNotIdentifiedError` instead, because `stop` would refuse to follow a record
    kept there anyway.
    """
    path = dashboard_record_path(record.port)
    try:
        path.parent.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        existing = path.parent.stat()
        if _untrusted(existing):
            raise DashboardNotIdentifiedError(
                record.port,
                f"the dashboard record directory {path.parent} is not owned by this user or "
                f"is writable by others; point UCLONE_UI_STATE_DIR at a private directory",
            ) from None
    process = (
        None
        if record.process is None
        else {"started": record.process.started, "command": record.process.command}
    )
    body = json.dumps(
        {
            "pid": record.pid,
            "port": record.port,
            "host": record.host,
            "instance": record.instance,
            "process": process,
        }
    )
    partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(body)
    os.replace(partial, path)
    return path


def remove_dashboard_record(record: DashboardRecord) -> None:
    """Remove the record for `record.port` if it is still this one, never a later start's."""
    path = dashboard_record_path(record.port)
    try:
        current = read_dashboard_record(record.port)
    except DashboardNotIdentifiedError:
        return
    if current is not None and current.instance == record.instance:
        path.unlink(missing_ok=True)


def _json_object(text: str) -> dict[str, object]:
    """Parse `text` as a JSON object; anything else is a `ValueError`."""
    parsed: object = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("not a JSON object")
    return cast(dict[str, object], parsed)


def read_dashboard_record(port: int) -> DashboardRecord | None:
    """The record for `port`; `None` only when there is no file.

    A file that cannot be read as a record, or that someone other than this user could have
    written, raises: it is not evidence that no dashboard is running, and it is not an
    instruction `stop` may follow.
    """
    path = dashboard_record_path(port)
    remedy = f"delete {path} once no dashboard is running on port {port}"
    untrusted = (
        f"is not owned by this user or is writable by others, so it cannot be trusted; {remedy}"
    )
    try:
        # Not following a symlink, and checking the descriptor that is read: a path checked
        # and then reopened can be swapped between the two.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DashboardNotIdentifiedError(
            port, f"the dashboard record {path} is unreadable, or a symlink ({exc}); {remedy}"
        ) from exc
    with os.fdopen(fd, "r", encoding="utf-8") as record_file:
        if _untrusted(os.fstat(record_file.fileno())):
            raise DashboardNotIdentifiedError(port, f"the dashboard record {path} {untrusted}")
        if _untrusted(path.parent.stat()):
            raise DashboardNotIdentifiedError(
                port, f"the dashboard record directory {path.parent} {untrusted}"
            )
        text = record_file.read()
    try:
        data = _json_object(text)
    except ValueError as exc:
        raise DashboardNotIdentifiedError(
            port, f"the dashboard record {path} is not JSON; {remedy}"
        ) from exc
    pid = data.get("pid")
    instance = data.get("instance")
    host = data.get("host")
    process = data.get("process")
    identity: ProcessIdentity | None = None
    if isinstance(process, dict):
        fields = cast(dict[str, object], process)
        started, command = fields.get("started"), fields.get("command")
        if isinstance(started, str) and isinstance(command, str) and started and command:
            identity = ProcessIdentity(started=started, command=command)
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid < 1
        or data.get("port") != port
        or not isinstance(host, str)
        or not isinstance(instance, str)
        or not instance
        or (process is not None and identity is None)
    ):
        raise DashboardNotIdentifiedError(
            port, f"the dashboard record {path} is malformed; {remedy}"
        )
    return DashboardRecord(pid=pid, port=port, host=host, instance=instance, process=identity)


def _probe_socket(family: int) -> socket.socket:
    return socket.socket(family, socket.SOCK_STREAM)


def _loopback_accepts_connections(port: int) -> bool:
    """Whether anything accepts a TCP connection on `port` at the IPv4 or IPv6 loopback.

    Only a connect that nothing accepted reads as "nothing here" (`_NOTHING_ACCEPTED`).
    Any other failure leaves the port uninspected, and raises rather than reporting it
    idle (P6, #881). It is asked when no dashboard is recorded for the port, and after a
    SIGKILL, when a `--dev` worker may have been orphaned; it decides the exit status and
    the message, never whether anything is signalled.
    """
    for family, address in _LOOPBACK_ADDRESSES:
        try:
            probe = _probe_socket(family)
        except OSError as exc:
            if family == socket.AF_INET6 and exc.errno in _NO_IPV6:
                continue
            raise ListeningProcessLookupError(
                port, f"no socket to probe {address} ({exc})"
            ) from exc
        with probe:
            probe.settimeout(_CONNECT_TIMEOUT_S)
            result = probe.connect_ex((address, port))
        if result == 0:
            return True
        if result in _NOTHING_ACCEPTED or (family == socket.AF_INET6 and result in _NO_IPV6):
            continue
        raise ListeningProcessLookupError(
            port, f"connecting to {address} failed ({errno.errorcode.get(result, result)})"
        )
    return False


def _staleness(record: DashboardRecord) -> str | None:
    """Why `record`'s PID is not its launcher any more; `None` while it still is.

    Raises `DashboardNotIdentifiedError` for a record that cannot say which process it was,
    and `ProcessIdentityUnavailableError` when `ps` cannot answer; each caller words the
    latter, because only it knows whether a signal has been sent yet.
    """
    path = dashboard_record_path(record.port)
    try:
        os.kill(record.pid, 0)
    except ProcessLookupError:  # exited; its PID is free for reuse
        return "is not running"
    except PermissionError:
        # Running as another user, so it cannot be this user's launcher.
        return "now belongs to another user's process"
    if record.process is None:
        raise DashboardNotIdentifiedError(
            record.port,
            f"PID {record.pid} is running, but its record {path} does not say which process "
            f"it was, so it cannot be told apart from a reused PID. Nothing was signalled; "
            f"{STOP_IT_YOURSELF}, then delete the record",
        )
    current = process_identity(record.pid)
    if current is None:
        return "is not running"
    if current.started != record.process.started or current.command != record.process.command:
        return "now belongs to another process"
    return None


def _wait_for_exit(record: DashboardRecord, timeout_s: float) -> bool:
    """Whether the recorded launcher is gone within `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    while _staleness(record) is None:
        if time.monotonic() >= deadline:
            return False
        time.sleep(_EXIT_POLL_S)
    return True


def _no_record_note(path: Path, stale_pid: int | None) -> str:
    stale = f" (a stale record for PID {stale_pid} was removed)" if stale_pid is not None else ""
    return (
        f"something accepts connections on it, but there is no dashboard record at {path}"
        f"{stale}. `ucx ui stop` signals only a dashboard `ucx ui` or `ucx start` recorded, "
        f"so nothing was signalled. If it is a UClone-X dashboard started before dashboard "
        f"records existed, or with another UCLONE_UI_STATE_DIR, {STOP_IT_YOURSELF}"
    )


def stop_ui_server(port: int = 5180) -> StopOutcome:
    """Stop the UClone-X dashboard on `port`, and only its recorded launcher (#613, #927).

    Returns the PID sent SIGTERM; none means no dashboard was recorded for the port and
    nothing accepts loopback connections on it. A launcher still running `TERMINATE_WAIT_S`
    after SIGTERM — and still the same process — is sent SIGKILL. Raises
    `DashboardNotIdentifiedError` when something holds the port that cannot be identified as
    a recorded launcher, and `ListeningProcessLookupError` when the port could not be
    inspected.
    """
    if not MIN_TCP_PORT <= port <= MAX_TCP_PORT:
        raise ListeningProcessLookupError(
            port, f"not a TCP port number ({MIN_TCP_PORT}-{MAX_TCP_PORT})"
        )
    path = dashboard_record_path(port)
    record = read_dashboard_record(port)
    stale_pid: int | None = None
    stale_reason: str | None = None
    if record is not None:
        try:
            stale_reason = _staleness(record)
        except ProcessIdentityUnavailableError as exc:
            raise DashboardNotIdentifiedError(
                port,
                f"cannot confirm PID {record.pid} is the dashboard recorded at {path}: {exc}. "
                f"Nothing was signalled; {STOP_IT_YOURSELF}",
            ) from exc
        if stale_reason is not None:
            path.unlink(missing_ok=True)
            stale_pid = record.pid
            record = None

    if record is None:
        if _loopback_accepts_connections(port):
            raise DashboardNotIdentifiedError(port, _no_record_note(path, stale_pid))
        return StopOutcome(
            port=port, stopped_pids=(), stale_pid=stale_pid, stale_reason=stale_reason
        )

    try:
        os.kill(record.pid, signal.SIGTERM)
    except ProcessLookupError:
        # Exited between the identity check and the signal: the record is stale after all.
        remove_dashboard_record(record)
        return StopOutcome(
            port=port, stopped_pids=(), stale_pid=record.pid, stale_reason="is not running"
        )
    try:
        exited = _wait_for_exit(record, TERMINATE_WAIT_S)
    except ProcessIdentityUnavailableError as exc:
        raise DashboardStopUnconfirmedError(
            port,
            f"SIGTERM was sent to PID {record.pid}, but whether it exited cannot be confirmed "
            f"({exc}). No SIGKILL was sent and the record at {path} is kept; if the dashboard "
            f"is still running, {STOP_IT_YOURSELF}",
        ) from exc
    if exited:
        remove_dashboard_record(record)
        return StopOutcome(port=port, stopped_pids=(record.pid,))
    # Identity was re-read an instant ago: this is still the launcher, not a reused PID.
    with contextlib.suppress(ProcessLookupError):
        os.kill(record.pid, signal.SIGKILL)
    remove_dashboard_record(record)
    try:
        still_accepting: bool | None = _loopback_accepts_connections(port)
    except ListeningProcessLookupError:
        still_accepting = None
    return StopOutcome(
        port=port, stopped_pids=(record.pid,), killed=True, still_accepting=still_accepting
    )
