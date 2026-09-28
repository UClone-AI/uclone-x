"""One runtime per links file: the dashboard and `ucx link run` do not both dial.

uClone2 keeps one live socket per clone. A second runtime dialling the same clone is closed
`4409` (replaced), and a session stops on 4409 rather than fight -- so two runtimes on one
machine, over one links file, would leave whichever dialled first offline, and the user
reading *이 클론은 다른 기기에서 연결되었습니다* about their own computer.

The rule: whichever process first has a link to dial holds this lock for as long as it
runs, and the other dials nothing. The lock is the gate lock's idiom (`cli/quality_gate.py`,
`hold_gate_lock`): an advisory `flock` on a sidecar the kernel releases when the process
dies, so a killed runtime leaves nothing to clean up, with the holder's role and PID written
into it so the one refused can say who has it. It sits next to the links file, so a
`UCLONE_LINKS_DIR` that points elsewhere is a separate runtime, as it is a separate store.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import IO

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX: no lock, as the store's own lock
    fcntl = None

__all__ = ["RuntimeLock", "RuntimeRole", "runtime_lock_path"]


class RuntimeRole(StrEnum):
    """Which head holds the links: what the refused one tells its user."""

    DASHBOARD = "dashboard"
    LINK_RUN = "link-run"


def runtime_lock_path(store_path: Path) -> Path:
    """`.uclone2.runtime.lock` beside the links file."""
    return store_path.parent / f".{store_path.stem}.runtime.lock"


class RuntimeLock:
    """Held from `acquire()` until `release()` or the process ends."""

    def __init__(self, path: Path, role: RuntimeRole) -> None:
        self._path = path
        self._role = role
        self._handle: IO[str] | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        """Take the lock without waiting; `False` when another process has it."""
        if self._handle is not None:
            return True
        if fcntl is None:  # pragma: no cover - not POSIX
            return True
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(fd, "r+", encoding="utf-8")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(f"{self._role.value} {os.getpid()}")
        handle.flush()
        self._handle = handle
        return True

    def held_by_another(self) -> bool:
        """Whether another process holds the lock now, without taking it.

        A shared, non-blocking try-lock on a descriptor of its own, dropped at once: it
        fails only while someone holds the exclusive lock. The file's text is not the
        answer -- a killed holder leaves its role written there with nothing held.
        """
        if self._handle is not None or fcntl is None:
            return False
        try:
            fd = os.open(self._path, os.O_RDONLY)
        except OSError:  # no file yet: nobody has ever held it
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None or fcntl is None:
            return
        handle.seek(0)
        handle.truncate()
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()

    def holder(self) -> RuntimeRole | None:
        """The role written by whoever holds the lock now; `None` if unknown or unheld."""
        try:
            text = self._path.read_text(encoding="utf-8").split()
        except OSError:
            return None
        try:
            return RuntimeRole(text[0]) if text else None
        except ValueError:
            return None
