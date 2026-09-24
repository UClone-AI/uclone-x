"""A file-backed log-offset allocator: the desktop head's default.

Durability is one small JSON record per session holding the session's id and the highest
offset written. **The id is stored, and that is the point** — an earlier draft stored only an
integer, so identity was not recoverable from what was persisted and the "identity check"
compared cursors instead. That compared the wrong thing in both directions: a stale cursor for
the right session was reported as an id mismatch, and a batch for the *wrong* session was
silently accepted whenever the numbers happened to line up. A number agreeing is not evidence
that a session matches.

**Single-writer, and unenforced.** This backend does not admit concurrent allocators over one
root: `record_appended` reads the cursor and writes it back with no exclusion between, so two
writers can both claim the same offset and neither is told. There is no lock file and no
compare-and-swap — `SessionStore` carries one on `revision` for the same hazard (#219), and
this does not.

The precondition is therefore **declared, not checked**. The contiguity check catches a coarse
interleaving and misses a fine one, so its silence is not evidence of a single writer. Stating
it here is not the same as enforcing it; a backend that must admit concurrent writers owes an
exclusion mechanism, and adding one to this backend is #567's to specify, since the writer it
would exclude does not exist yet.

**The cursor lags the log, deliberately.** It is written *after* the events it counts, so a
crash between the log write and `record_appended` leaves the cursor behind the log, and the
next run re-issues an offset the log already holds. The other ordering fails the other way — a
cursor ahead of a log claims events that were never written, which a reader cannot distinguish
from a truncated log. Lagging was chosen because the log itself is the authority and can be
read back to find the true tail; leading is unrecoverable because nothing else records the
intent.

But that recovery is **not implemented here and this module cannot do it**: reconciling a
cursor against a log means reading the log, which this allocator has no handle on. A consumer
that survives crashes must reconcile on open. Until one exists the gap is real and is stated
rather than implied by the word "durable".

The medium is a choice, not the contract: `LogOffsetAllocatorProtocol` says nothing about
files.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, cast

from uclone_x.core.log_offset import FIRST_LOG_OFFSET, MAX_LOG_OFFSET, LogOffset
from uclone_x.errors import (
    LogOffsetContiguityError,
    LogOffsetCorruptCursorError,
    LogOffsetSessionMismatchError,
    PathTraversalError,
)

__all__ = ["FileLogOffsetAllocator"]

logger = logging.getLogger(__name__)

_FORBIDDEN_ID_SUBSTRINGS = ("..", "/", "\\", "\x00")
_TEMP_SUFFIX_PATTERN = re.compile(r"\.tmp\.(\d+)\.[0-9a-fA-F]{8}$")
"""Anchored at the end, so a *committed* record whose session id happens to contain
`.tmp.<pid>.<hex8>` is not mistaken for debris. `agent/session.py` anchors its own reaper
pattern for exactly this reason (#269), and dropping the anchor here would re-open the finding
that fix closed."""

_TEMP_MAX_AGE_SECONDS = 300.0


class FileLogOffsetAllocator:
    """Allocates per-session log offsets, durably, from a directory of cursor records."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _cursor_path(self, session_id: str) -> Path:
        # Checked before any path arithmetic: `Path` normalises a `..` segment away during
        # `resolve()`, so a containment check alone cannot report *why* an id was rejected.
        # `agent/session.py` guards its filenames the same way and for the same reason.
        if not session_id:
            raise PathTraversalError("Session id must not be empty.")
        for bad in _FORBIDDEN_ID_SUBSTRINGS:
            if bad in session_id:
                raise PathTraversalError(
                    f"Session id {session_id!r} contains {bad!r} and cannot name a cursor file."
                )
        return self._root / f"{session_id}.cursor"

    def _read_record(self, session_id: str) -> tuple[str, LogOffset] | None:
        """The stored `(session_id, cursor)`, or `None` when this session has no log.

        Returns the *stored* id rather than the requested one, so a caller can compare
        identities instead of inferring one from a number.
        """
        path = self._cursor_path(session_id)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except UnicodeDecodeError as exc:
            # Not a subclass of OSError, so it needs its own arm.
            raise LogOffsetCorruptCursorError(
                f"Cursor record at {path} is not valid UTF-8; refusing rather than treating "
                f"session {session_id!r}'s log as absent."
            ) from exc
        except OSError as exc:
            # A directory where the file belongs, a permission failure, a name too long. A
            # caller writing `except LogOffsetError` must catch these too: leaking the raw
            # `OSError` puts a reachable state outside the taxonomy this module defines, and
            # the caller's fallback for an uncaught error is the one thing that must not
            # happen here — treating the log as absent.
            raise LogOffsetCorruptCursorError(
                f"Cursor record at {path} could not be read ({exc.__class__.__name__}: {exc}); "
                f"refusing rather than treating session {session_id!r}'s log as absent."
            ) from exc

        # A record that cannot be read is refused, never treated as absent. Absent hands back
        # the first offset and lets a writer overwrite a log that exists — the silent-wipe
        # shape, worse than an error because the caller has no reason to look again.
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LogOffsetCorruptCursorError(
                f"Cursor record at {path} is not valid JSON; refusing rather than treating "
                f"session {session_id!r}'s log as absent."
            ) from exc

        if not isinstance(payload, dict):
            raise LogOffsetCorruptCursorError(f"Cursor record at {path} is not an object.")
        record = cast(dict[str, object], payload)
        stored_id: object = record.get("session_id")
        stored_cursor: object = record.get("cursor")
        if not isinstance(stored_id, str) or not stored_id:
            raise LogOffsetCorruptCursorError(f"Cursor record at {path} names no session.")
        # `bool` is a subclass of `int`; `True` must not read back as offset 1.
        if not isinstance(stored_cursor, int) or isinstance(stored_cursor, bool):
            raise LogOffsetCorruptCursorError(
                f"Cursor in {path} is {stored_cursor!r}, which is not an integer offset."
            )
        if stored_cursor < 0 or stored_cursor > MAX_LOG_OFFSET:
            raise LogOffsetCorruptCursorError(
                f"Cursor in {path} is {stored_cursor}, outside [0, {MAX_LOG_OFFSET}]."
            )
        return stored_id, LogOffset(stored_cursor)

    def _require_identity(self, session_id: str, record: tuple[str, LogOffset] | None) -> None:
        """Refuse when the stored log belongs to a different session.

        The comparison is between identities. A case-insensitive filesystem is why this is not
        redundant with the filename: `Sess-A.cursor` and `sess-a.cursor` are one file on
        macOS, so without this a distinct session silently inherits another's cursor —
        the door `SessionStore.delete` documents closing for sessions, reopened here if the
        id is not stored and compared.

        **The outcome of a collision is terminal, and that is a tradeoff rather than a fix.**
        On a case-insensitive filesystem, once `Sess-A` holds the record, `sess-a` can never
        obtain an offset from any method on this class — there is no delete and no reset, so
        the API offers no way out. Refusing is still right: inheriting another session's cursor
        corrupts two logs, and refusing wedges one. macOS is the shipped desktop head (P0), so
        this is reachable on the default platform, not a theoretical case. The repair is to
        remove the cursor file out of band, which is an operator action this class does not
        expose deliberately — a reset method is indistinguishable from the silent wipe the
        module refuses.
        """
        if record is not None and record[0] != session_id:
            raise LogOffsetSessionMismatchError(
                f"The log at {self._cursor_path(session_id)} belongs to session "
                f"{record[0]!r}, not {session_id!r}."
            )

    def next_offset(self, session_id: str) -> LogOffset:
        """The offset the next appended event will occupy."""
        record = self._read_record(session_id)
        self._require_identity(session_id, record)
        return FIRST_LOG_OFFSET if record is None else LogOffset(record[1] + 1)

    def _check_batch(
        self,
        session_id: str,
        offsets: tuple[LogOffset, ...],
        cursor: LogOffset,
        record: tuple[str, LogOffset] | None,
    ) -> None:
        """The batch rules, against a record the caller has already read.

        Both public methods share this so the file is read **once** per call. An earlier draft
        had `record_appended` read, then call `validate_batch`, which read twice more — three
        reads of one file per append, and the identity check and the ordering check could see
        different contents in between. Passing the record in makes the checks agree on what
        they are checking.
        """
        # Identity before ordering (#256): an id mismatch reported as an ordering conflict
        # names a position in a log the caller never meant.
        self._require_identity(session_id, record)

        expected = FIRST_LOG_OFFSET if record is None else LogOffset(record[1] + 1)
        if cursor != expected:
            raise LogOffsetContiguityError(
                f"Cursor {cursor} for session {session_id!r} is stale: the log continues at "
                f"{expected}."
            )
        if not offsets:
            return
        if offsets[0] != expected:
            raise LogOffsetContiguityError(
                f"Batch for session {session_id!r} starts at {offsets[0]}; the log expects "
                f"{expected}."
            )
        for previous, current in zip(offsets, offsets[1:], strict=False):
            if current != previous + 1:
                raise LogOffsetContiguityError(
                    f"Batch for session {session_id!r} jumps from {previous} to {current}; "
                    f"expected {previous + 1}."
                )
        # The bound lives here, in the check both public methods share, and not in
        # `record_appended` alone. An earlier fix put it only on the write path, which closed
        # one asymmetry by opening its mirror: `validate_batch` accepted a batch that
        # `record_appended` would then refuse, so a caller could validate, extend the log, and
        # be refused with the events already written — the leading-cursor failure this design
        # exists to avoid. A validator that disagrees with the writer it guards is worse than
        # no validator, in whichever direction it disagrees.
        if offsets[-1] > MAX_LOG_OFFSET:
            raise LogOffsetContiguityError(
                f"Batch for session {session_id!r} ends at {offsets[-1]}, above the largest "
                f"offset that reads back exactly ({MAX_LOG_OFFSET}); refusing rather than writing "
                f"a record this allocator could not then read."
            )

    def validate_batch(
        self, session_id: str, offsets: tuple[LogOffset, ...], cursor: LogOffset
    ) -> None:
        """Refuse a batch that does not continue this session's log."""
        self._check_batch(session_id, offsets, cursor, self._read_record(session_id))

    def record_appended(self, session_id: str, offsets: tuple[LogOffset, ...]) -> None:
        """Advance the cursor after a batch has been durably written.

        Validates before advancing. An earlier draft took `offsets[-1]` on trust, so a batch
        could move the cursor **backward** — after which the next batch re-issues offsets the
        log already holds. Duplicate positions in an append-only log are not recoverable by
        reading it back, and `validate_batch` cannot catch them afterwards because it reads
        the same cursor.

        Validation closes that door for a *malformed call*. It does not close it for a crash:
        if the process dies after the log write and before this method returns, the cursor
        still lags and the next run still re-issues. See the module docstring — that direction
        is chosen, not prevented, and recovering from it needs the log, which this allocator
        cannot see.

        **`offsets` must be the offsets actually written, not the batch attempted.** After a
        partial append, passing the whole batch advances the cursor past events that do not
        exist, which is the leading-cursor failure this design otherwise avoids. Nothing here
        can check that, because the store has no view of the log.
        """
        record = self._read_record(session_id)
        if not offsets:
            # Read *before* the early return, so an empty batch for the wrong session or for a
            # traversal-shaped id is refused here exactly as `validate_batch` refuses it.
            # Returning first made the two methods disagree about the same input: nothing was
            # written either way, but "the store accepted it" and "the store refused it" cannot
            # both be the answer for one call.
            self._require_identity(session_id, record)
            return
        expected = FIRST_LOG_OFFSET if record is None else LogOffset(record[1] + 1)
        # The `cursor` arm is trivially satisfied here, since `expected` is derived from the
        # same record it is compared against. The start and pairwise checks are what protect
        # this path, and they are not trivial — a rewind or a gap is refused by them.
        self._check_batch(session_id, offsets, cursor=expected, record=record)

        highest = int(offsets[-1])
        path = self._cursor_path(session_id)
        payload = json.dumps({"session_id": session_id, "cursor": highest})
        # Temp-and-replace with an fsync on the file *and* the directory. Without the
        # directory sync the rename itself may not survive a power loss, so the record would
        # be durable and its name would not.
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            dir_fd = os.open(self._root, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            # `except`, not `finally`, matching `SessionStore.save`: after a successful
            # `os.replace` the temp name is gone and unlinking it is dead code.
            tmp.unlink(missing_ok=True)
            raise

    def reap_orphaned_temp_files(self, max_age_seconds: float = _TEMP_MAX_AGE_SECONDS) -> int:
        """Remove temp records left by a process that died between `open` and `os.replace`.

        A temp file is unlinked only when it is **evidently** debris: the name matches the
        anchored temp pattern *and* either the pid encoded in it is gone, or the file is older
        than `max_age_seconds`. Returns the number unlinked.

        `max_age_seconds=0` **disables** the age guard rather than meaning "any age", so a live
        pid's temp file is then kept forever. That reading is not obvious from the name and is
        stated because it is the opposite of what "older than 0" suggests; it matches
        `agent/session.py`, and changing one without the other would be worse than the
        surprise.

        Both guards are load-bearing, and an earlier draft of this method had neither.

        * Without the anchor, the glob `*.cursor.tmp.*` also matches the *committed* record of
          a session whose id contains `.cursor.tmp.` — so reaping deleted a live cursor and the
          log silently restarted at offset 1. That is precisely the silent-wipe shape the
          module docstring says this backend refuses, reached through the code meant to keep
          the directory clean. `agent/session.py` anchors for the same reason (#269).
        * Without the liveness and age guards, reaping unlinks a *live* writer's in-flight temp
          file, and that writer's `os.replace` then fails. A reaper that can break a concurrent
          write is worse than the debris it collects.

        **Nothing calls this yet**, and it is deliberately not on `LogOffsetAllocatorProtocol`:
        temp-file debris is a property of this medium, and a database-backed allocator has no
        such thing to collect, so putting it on the protocol would oblige every backend to
        implement a no-op. It is this backend's own maintenance entry point, and the writer
        #567 introduces is what should call it on open.
        """
        removed = 0
        now = time.time()
        for candidate in self._root.glob("*.cursor.tmp.*"):
            match = _TEMP_SUFFIX_PATTERN.search(candidate.name)
            if match is None:
                continue
            try:
                # Not load-bearing for correctness: without it, `unlink` on a directory raises
                # `OSError`, which the handler below catches and continues past. It is here so
                # that case does not log a warning that reads like a failure.
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            try:
                pid = int(match.group(1))
            except ValueError:  # pragma: no cover - the pattern admits only digits
                continue
            # A pid field above `pid_t`'s range makes `os.kill` raise `OverflowError`, which
            # is an `ArithmeticError` and so escapes every `OSError` arm below — one junk
            # filename would then abort the reap for every real orphan in the directory and
            # throw outside this module's taxonomy. Measured boundary on this platform:
            # `os.kill(2**31 - 1, 0)` raises `ProcessLookupError`, `os.kill(2**31, 0)` raises
            # `OverflowError`. An earlier comment here blamed "a C long", which is wrong under
            # LP64 — a long is 64-bit and 2**31 fits it. `pid_t` is `int32_t`, and that is
            # where the range ends.
            #
            # Such a pid cannot name a process, so the file is debris. It is still put through
            # the age guard rather than deleted outright: an earlier version returned here and
            # called that caution, on the reasoning that "the pid is the only evidence of
            # ownership" — but three lines down a *live* pid's file is deleted on age alone, so
            # age is already accepted as sufficient evidence. Declining it here only made the
            # file uncollectable by any code path, forever. A leak is not caution.
            uninterpretable_pid = pid > _MAX_PID
            orphaned = not uninterpretable_pid and not _is_pid_alive(pid)
            if not orphaned and max_age_seconds > 0:
                try:
                    orphaned = (now - candidate.stat().st_mtime) >= max_age_seconds
                except OSError:
                    continue
            if not orphaned:
                continue
            try:
                candidate.unlink(missing_ok=True)
            except OSError as err:
                logger.warning("Failed to reap orphaned cursor temp file %s: %s", candidate, err)
                continue
            removed += 1
        return removed


def _is_pid_alive(pid: int) -> bool:
    """Whether a process with this pid exists.

    Signal 0 performs the permission and existence checks without delivering anything.
    `PermissionError` means the process exists and belongs to someone else, which is still
    alive for our purposes — the temp file is not debris.

    Duplicated from `agent/session.py` rather than imported: this module is a log adapter and
    `agent` sits above it, so importing upward would invert the dependency
    the core/shell architecture note fixes. The duplication is small and deliberate.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


_MAX_PID = 2**31 - 1
"""The largest pid `os.kill` will accept without raising `OverflowError`.

`pid_t` is `int32_t`, so this is its maximum. Measured on this platform:
`os.kill(2**31 - 1, 0)` raises `ProcessLookupError` (in range, no such process) and
`os.kill(2**31, 0)` raises `OverflowError` (out of range) — so the bound is exact and no
real pid is excluded by it.

`OverflowError` is an `ArithmeticError`, which is why an unguarded call escapes an `OSError`
handler. No process can hold a pid above this; a filename claiming one is malformed."""
