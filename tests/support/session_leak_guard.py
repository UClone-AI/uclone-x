"""Attribute a file in real session storage to the process that wrote it (#453, #963).

`tests/conftest.py` fails a test that leaves a new file in the invoking user's real session
storage. It used to decide that by comparing two directory listings, which attributes a file
to whichever test was running when it appeared, whoever wrote it. The storage is shared: a
person chatting through a dashboard on the same machine writes a session there while the
suite runs, and the gate went red on a test that had touched nothing (#963; observed on
PRs #925, #934, #956, #962).

A leak is now a file that is both **new** in that storage and **written by this process** —
the pytest process or the parallel worker running the test. Writes are observed through a
`sys.addaudithook` hook, which sees every `open` for writing and every rename or link
destination this interpreter performs, and nothing another process does. A dashboard is
another process, so its sessions can no longer be attributed to a test.

**What this does not see: a child process the test starts.** The listing comparison did.
The children are covered by the structural guard in `tests/conftest.py`, which hands every
`subprocess` and `asyncio` child the test's redirected `UCLONE_SESSION_DIR`, so a child
reaches real storage only through code that ignores the variable — code the in-process
tests of the same module run under this hook. That trade is the fix: no listing of a shared
directory can tell a test's child from a person's dashboard.

**Nor two in-process writers** (probed in the PR #989 review; neither is reachable from `src`
today, so a change that adds one needs a check of its own):

* **A `dir_fd`-relative write or rename** — `os.open(name, flags, dir_fd=d)`,
  `os.rename(src, dst, dst_dir_fd=d)`. The `open` event carries only the relative name, and
  `os.rename` carries the descriptors as separate arguments that are not read; the name is made
  absolute against the working directory instead of `d`, so it never matches the new file.
  The one `dir_fd` in `src`, in `log/file_allocator.py`, opens a directory to fsync it.
* **`sqlite3`** — SQLite opens its database, journal and WAL files in C. The only event is
  `sqlite3.connect`, which is not watched, and the journal files raise none. `src` uses no
  `sqlite3`, `dbm` or `shelve`.

Audit hooks cannot be removed, so one dispatcher is installed per process and does nothing
unless a guard is active.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

#: Flags that let an `open` create or change a file. Reading cannot make a file new, so
#: skipping read-only opens changes no verdict; it keeps every module import out of the set.
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

#: Audit events whose second argument is a destination path. `os.replace` raises
#: `os.rename`, and `SessionStore.save` lands every record with `os.replace`.
_DESTINATION_EVENTS = frozenset({"os.rename", "os.link", "os.symlink"})

_EVENTS = _DESTINATION_EVENTS | {"open"}


def _files_under(directory: Path) -> set[Path]:
    return {f for f in directory.rglob("*") if f.is_file()} if directory.exists() else set()


def _as_path(value: object) -> str | None:
    """An absolute path string for an audit argument, or None for a file descriptor."""
    if isinstance(value, (str, bytes, os.PathLike)):
        return os.path.abspath(os.fsdecode(value))  # pyright: ignore[reportUnknownArgumentType]
    return None


class SessionLeakGuard:
    """Decide which files new in `real_dir` this process wrote while the guard was active."""

    def __init__(self, real_dir: Path, label: str = "session") -> None:
        self.real_dir = real_dir
        self.label = label
        self._before = _files_under(real_dir)
        self._written: set[str] = set()

    def observe(self, event: str, args: tuple[object, ...]) -> None:
        if event == "open":
            flags = args[2]
            if not (isinstance(flags, int) and flags & _WRITE_FLAGS):
                return
            path = _as_path(args[0])
        else:
            path = _as_path(args[1])
        if path is not None:
            self._written.add(path)

    def leaked_files(self) -> list[Path]:
        """Files new in `real_dir` since the guard started that this process wrote."""
        new_files = _files_under(self.real_dir) - self._before
        written = {Path(os.path.realpath(p)) for p in set(self._written)}
        return sorted(f for f in new_files if f.resolve() in written)

    def assert_no_leaks(self) -> None:
        leaked = self.leaked_files()
        assert not leaked, (
            f"Test leaked {self.label} files into the real storage root "
            f"({self.real_dir}): {sorted(str(f.relative_to(self.real_dir)) for f in leaked)}"
        )


_active: list[SessionLeakGuard] = []
_installed = False


def _dispatch(event: str, args: tuple[object, ...]) -> None:
    if event not in _EVENTS or not _active:
        return
    for guard in list(_active):
        guard.observe(event, args)


@contextmanager
def watching_session_writes(real_dir: Path, label: str = "session") -> Generator[SessionLeakGuard]:
    """Record this process's writes while the block runs; judge them with the yielded guard."""
    global _installed
    if not _installed:
        sys.addaudithook(_dispatch)
        _installed = True
    guard = SessionLeakGuard(real_dir, label=label)
    _active.append(guard)
    try:
        yield guard
    finally:
        _active.remove(guard)
