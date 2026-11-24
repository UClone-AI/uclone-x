"""Record, verify, and stop managed background daemons (gateway, worker) with the stdlib only.

Extends the OS-verified PID identity discipline of `uclone_x.shells.ui_process` (#927, #951) to
every background process the declarative stack reconciler (`ucx up` / `ucx down`) launches:
`gateway-<port>.json` and `worker-<slot>.json` under `default_dashboard_state_dir()`
(`UCLONE_UI_STATE_DIR` or `~/.uclone/ui`).

Like `ui_process.py`, this module imports nothing from `fastapi`, `uvicorn`, or the `cluster`
extra so that `uclone-x[cli]` alone can inspect or stop any managed daemon.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import signal
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from uclone_x.errors import UCloneXError
from uclone_x.shells.ui_process import (
    MAX_TCP_PORT,
    MIN_TCP_PORT,
    TERMINATE_WAIT_S,
    ProcessIdentity,
    ProcessIdentityUnavailableError,
    _loopback_accepts_connections,  # pyright: ignore[reportPrivateUsage]
    default_dashboard_state_dir,
    process_identity,
)

__all__ = [
    "DaemonNotIdentifiedError",
    "DaemonRecord",
    "DaemonStopOutcome",
    "DaemonStopUnconfirmedError",
    "daemon_record_path",
    "daemon_staleness",
    "list_daemon_records",
    "new_daemon_record",
    "read_daemon_record",
    "remove_daemon_record",
    "stop_daemon",
    "write_daemon_record",
]

_SAFE_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_EXIT_POLL_S: Final[float] = 0.1


class DaemonNotIdentifiedError(UCloneXError):
    """A daemon record or port listener cannot be verified as a UClone-X daemon we own."""

    def __init__(self, kind: str, slot: str, reason: str) -> None:
        super().__init__(f"{kind}-{slot}: {reason}")
        self.kind = kind
        self.slot = slot
        self.reason = reason


class DaemonStopUnconfirmedError(UCloneXError):
    """SIGTERM was sent to a verified daemon, and `ps` could not confirm whether it exited."""

    def __init__(self, kind: str, slot: str, reason: str) -> None:
        super().__init__(f"{kind}-{slot}: {reason}")
        self.kind = kind
        self.slot = slot
        self.reason = reason


@dataclass(frozen=True)
class DaemonRecord:
    """Recorded OS identity of a background daemon launched by `ucx`."""

    kind: str
    slot: str
    pid: int
    instance: str
    process: ProcessIdentity | None
    port: int | None = None
    host: str | None = None


@dataclass(frozen=True)
class DaemonStopOutcome:
    """Result of stopping one recorded daemon slot."""

    kind: str
    slot: str
    stopped_pids: tuple[int, ...]
    stale_pid: int | None = None
    stale_reason: str | None = None
    killed: bool = False
    still_accepting: bool | None = None


def _validate_token(label: str, value: str) -> str:
    if not _SAFE_TOKEN_RE.match(value):
        raise ValueError(f"Invalid daemon {label} {value!r}")
    return value


def _untrusted(info: os.stat_result) -> bool:
    return info.st_uid != os.getuid() or bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def daemon_record_path(kind: str, slot: str) -> Path:
    """Return the `0600` record path for `<kind>-<slot>.json`."""
    _validate_token("kind", kind)
    _validate_token("slot", slot)
    return default_dashboard_state_dir() / f"{kind}-{slot}.json"


def new_daemon_record(
    kind: str,
    slot: str,
    *,
    pid: int | None = None,
    port: int | None = None,
    host: str | None = None,
) -> tuple[DaemonRecord, str | None]:
    """Build a `DaemonRecord` capturing `ps` `lstart` and `command` for `pid` (default: self)."""
    _validate_token("kind", kind)
    _validate_token("slot", slot)
    if port is not None and not (MIN_TCP_PORT <= port <= MAX_TCP_PORT):
        raise ValueError(f"Invalid TCP port {port}")
    target_pid = os.getpid() if pid is None else pid
    problem: str | None = None
    identity: ProcessIdentity | None = None
    try:
        identity = process_identity(target_pid)
    except ProcessIdentityUnavailableError as exc:
        problem = str(exc)
    else:
        if identity is None:
            problem = f"`ps` does not list process ({target_pid})"
    record = DaemonRecord(
        kind=kind,
        slot=slot,
        pid=target_pid,
        port=port,
        host=host,
        instance=secrets.token_hex(16),
        process=identity,
    )
    return record, problem


def write_daemon_record(record: DaemonRecord) -> Path:
    """Write `record` atomically with mode `0600` inside a private `0700` state directory."""
    path = daemon_record_path(record.kind, record.slot)
    try:
        path.parent.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        existing = path.parent.stat()
        if _untrusted(existing):
            raise DaemonNotIdentifiedError(
                record.kind,
                record.slot,
                f"the daemon record directory {path.parent} is not owned by this user or "
                "is writable by others; point UCLONE_UI_STATE_DIR at a private directory",
            ) from None
    proc_payload = (
        None
        if record.process is None
        else {"started": record.process.started, "command": record.process.command}
    )
    body = json.dumps(
        {
            "kind": record.kind,
            "slot": record.slot,
            "pid": record.pid,
            "port": record.port,
            "host": record.host,
            "instance": record.instance,
            "process": proc_payload,
        }
    )
    partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(body)
    os.replace(partial, path)
    return path


def read_daemon_record(kind: str, slot: str) -> DaemonRecord | None:
    """Read and validate `<kind>-<slot>.json`; `None` only when no record file exists."""
    path = daemon_record_path(kind, slot)
    remedy = f"delete {path} once {kind}-{slot} is stopped"
    untrusted = (
        f"is not owned by this user or is writable by others, so it cannot be trusted; {remedy}"
    )
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DaemonNotIdentifiedError(
            kind, slot, f"the daemon record {path} is unreadable or a symlink ({exc}); {remedy}"
        ) from exc
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        if _untrusted(os.fstat(handle.fileno())):
            raise DaemonNotIdentifiedError(kind, slot, f"the daemon record {path} {untrusted}")
        if _untrusted(path.parent.stat()):
            raise DaemonNotIdentifiedError(
                kind, slot, f"the daemon record directory {path.parent} {untrusted}"
            )
        text = handle.read()
    try:
        raw: object = json.loads(text)
    except ValueError as exc:
        raise DaemonNotIdentifiedError(
            kind, slot, f"the daemon record {path} is not JSON; {remedy}"
        ) from exc
    if not isinstance(raw, dict):
        raise DaemonNotIdentifiedError(
            kind, slot, f"the daemon record {path} is malformed; {remedy}"
        )
    data = cast(dict[str, object], raw)
    pid = data.get("pid")
    instance = data.get("instance")
    port = data.get("port")
    host = data.get("host")
    proc_obj = data.get("process")
    identity: ProcessIdentity | None = None
    if isinstance(proc_obj, dict):
        fields = cast(dict[str, object], proc_obj)
        started, command = fields.get("started"), fields.get("command")
        if isinstance(started, str) and isinstance(command, str) and started and command:
            identity = ProcessIdentity(started=started, command=command)
    valid_port = port is None or (
        isinstance(port, int)
        and not isinstance(port, bool)
        and MIN_TCP_PORT <= port <= MAX_TCP_PORT
    )
    valid_host = host is None or isinstance(host, str)
    if (
        data.get("kind") != kind
        or data.get("slot") != slot
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid < 1
        or not valid_port
        or not valid_host
        or not isinstance(instance, str)
        or not instance
        or (proc_obj is not None and identity is None)
    ):
        raise DaemonNotIdentifiedError(
            kind, slot, f"the daemon record {path} is malformed; {remedy}"
        )
    return DaemonRecord(
        kind=kind,
        slot=slot,
        pid=pid,
        port=cast(int | None, port),
        host=cast(str | None, host),
        instance=instance,
        process=identity,
    )


def remove_daemon_record(record: DaemonRecord) -> None:
    """Remove `<kind>-<slot>.json` only when its `instance` still matches `record`."""
    path = daemon_record_path(record.kind, record.slot)
    try:
        current = read_daemon_record(record.kind, record.slot)
    except DaemonNotIdentifiedError:
        return
    if current is not None and current.instance == record.instance:
        path.unlink(missing_ok=True)


def list_daemon_records(kind: str) -> tuple[DaemonRecord, ...]:
    """Return every readable `<kind>-<slot>.json` record sorted by slot."""
    _validate_token("kind", kind)
    state_dir = default_dashboard_state_dir()
    if not state_dir.is_dir():
        return ()
    prefix = f"{kind}-"
    suffix = ".json"
    records: list[DaemonRecord] = []
    for entry in sorted(state_dir.iterdir(), key=lambda p: p.name):
        name = entry.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        slot = name[len(prefix) : -len(suffix)]
        if not _SAFE_TOKEN_RE.match(slot):
            continue
        rec = read_daemon_record(kind, slot)
        if rec is not None:
            records.append(rec)
    return tuple(records)


def daemon_staleness(record: DaemonRecord) -> str | None:
    """Return `None` while `record.pid` is still the recorded OS process, else why it is stale."""
    path = daemon_record_path(record.kind, record.slot)
    try:
        os.kill(record.pid, 0)
    except ProcessLookupError:
        return "is not running"
    except PermissionError:
        return "now belongs to another user's process"
    if record.process is None:
        raise DaemonNotIdentifiedError(
            record.kind,
            record.slot,
            f"PID {record.pid} is running, but its record {path} carries no OS process identity, "
            "so it cannot be distinguished from a reused PID; nothing was signalled",
        )
    current = process_identity(record.pid)
    if current is None:
        return "is not running"
    if current.started != record.process.started or current.command != record.process.command:
        return "now belongs to another process"
    return None


def _wait_for_exit(record: DaemonRecord, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while daemon_staleness(record) is None:
        if time.monotonic() >= deadline:
            return False
        time.sleep(_EXIT_POLL_S)
    return True


def stop_daemon(
    kind: str,
    slot: str,
    *,
    expected_port: int | None = None,
    terminate_wait_s: float = TERMINATE_WAIT_S,
) -> DaemonStopOutcome:
    """Stop a recorded daemon after verifying its OS `lstart` + `command` identity."""
    path = daemon_record_path(kind, slot)
    record = read_daemon_record(kind, slot)
    stale_pid: int | None = None
    stale_reason: str | None = None
    if record is not None:
        try:
            stale_reason = daemon_staleness(record)
        except ProcessIdentityUnavailableError as exc:
            raise DaemonNotIdentifiedError(
                kind,
                slot,
                f"cannot confirm PID {record.pid} matches {path}: {exc}; nothing was signalled",
            ) from exc
        if stale_reason is not None:
            path.unlink(missing_ok=True)
            stale_pid = record.pid
            record = None

    check_port = expected_port
    if record is None:
        if check_port is not None and _loopback_accepts_connections(check_port):
            stale_note = (
                f" (removed stale record for PID {stale_pid})" if stale_pid is not None else ""
            )
            raise DaemonNotIdentifiedError(
                kind,
                slot,
                f"something accepts connections on port {check_port}, but there is no verified "
                f"daemon record at {path}{stale_note}; refusing to signal an unknown process",
            )
        return DaemonStopOutcome(
            kind=kind,
            slot=slot,
            stopped_pids=(),
            stale_pid=stale_pid,
            stale_reason=stale_reason,
        )

    try:
        os.kill(record.pid, signal.SIGTERM)
    except ProcessLookupError:
        remove_daemon_record(record)
        return DaemonStopOutcome(
            kind=kind,
            slot=slot,
            stopped_pids=(),
            stale_pid=record.pid,
            stale_reason="is not running",
        )
    try:
        exited = _wait_for_exit(record, terminate_wait_s)
    except ProcessIdentityUnavailableError as exc:
        raise DaemonStopUnconfirmedError(
            kind,
            slot,
            f"SIGTERM was sent to PID {record.pid}, but `ps` could not confirm exit ({exc}); "
            f"no SIGKILL was sent and {path} was kept",
        ) from exc
    if exited:
        remove_daemon_record(record)
        return DaemonStopOutcome(kind=kind, slot=slot, stopped_pids=(record.pid,))

    with contextlib.suppress(ProcessLookupError):
        os.kill(record.pid, signal.SIGKILL)
    remove_daemon_record(record)
    port_to_check = record.port if record.port is not None else check_port
    still_accepting: bool | None = None
    if port_to_check is not None:
        with contextlib.suppress(Exception):
            still_accepting = _loopback_accepts_connections(port_to_check)
    return DaemonStopOutcome(
        kind=kind,
        slot=slot,
        stopped_pids=(record.pid,),
        killed=True,
        still_accepting=still_accepting,
    )
