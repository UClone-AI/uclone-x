"""Durable storage for room transcripts, on the same contract as the session store.

A room has exactly one writer — its orchestrator — so the compare-and-swap here is not a
routine conflict path. It is the guard on that claim: a refusal means two orchestrators
are driving one room, which interleaves utterances and loses one side's turns rather than
merely colliding on a field.

Filesystem layout mirrors `SessionStore`: one JSON document per room, written to a
temporary file in the same directory and moved into place, so a reader never observes a
half-written transcript.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from uclone_x.agent.session import resolve_session_path
from uclone_x.errors import (
    PathTraversalError,
    RoomIdError,
    StaleRoomWriteError,
    UnreadableRoomRecordError,
)
from uclone_x.room.models import RoomState

__all__ = ["ROOM_STORAGE_DIR_ENV_VAR", "RoomStore", "default_room_storage_dir"]

#: Redirects the room store, as `UCLONE_SESSION_DIR` redirects the session store. Separate
#: variables for separate records: a headless run that wants its rooms elsewhere is not
#: necessarily the same request as moving its sessions, and one variable governing both
#: cannot express the difference.
ROOM_STORAGE_DIR_ENV_VAR = "UCLONE_ROOM_DIR"

DEFAULT_ROOM_STORAGE_DIR = Path.home() / ".uclone" / "rooms"


def default_room_storage_dir() -> Path:
    """Resolve the room store's root, honouring `UCLONE_ROOM_DIR`.

    A sibling of the session root rather than a subdirectory of it: the session store
    treats every `*.json` under its root as a session, and a room document placed there
    would read back as a session that will not validate.
    """
    override = os.environ.get(ROOM_STORAGE_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_ROOM_STORAGE_DIR


def _now_iso() -> str:
    """Current UTC instant as an ISO-8601 string, matching `RoomState`'s own stamps."""
    return datetime.now(UTC).isoformat()


class RoomStore:
    """Keeps rooms as JSON documents under `storage_dir`."""

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        """Keep rooms under `storage_dir`, or under the configured default when omitted.

        The default is resolved on each construction rather than captured at import, so a
        test or a headless run that sets `UCLONE_ROOM_DIR` is honoured by a store built
        afterwards — the failure the session root's own resolver documents.
        """
        self._dir = (
            default_room_storage_dir() if storage_dir is None else Path(storage_dir).expanduser()
        )

    @property
    def storage_dir(self) -> Path:
        """Directory holding the room documents."""
        return self._dir

    def room_path(self, room_id: str) -> Path:
        """Resolve `room_id` to its document path, refusing anything that escapes it.

        `resolve_session_path`, not `validate_session_id`: the guard has two halves and
        only one of them is a *name* rule. The other is containment of the resolved path,
        which is what refuses a lexically innocent id whose record is a planted symlink
        pointing out of the directory — a case the name rule cannot see, and which an
        earlier version of this method let through by reusing only the half that was easy
        to name. A room id and a session id become a single path component in exactly the
        same way, so this is one guard with two callers rather than two guards.

        The *message* is translated, though the check is not re-implemented. A person using
        `ucx room` has no notion of a session, and `resolve_session_path` naturally says
        "Session ID must be non-empty" — true of the control and meaningless to the caller,
        who asked about a room. Re-raising in the room's own vocabulary keeps one guard and
        one error the reader can act on.
        """
        try:
            return resolve_session_path(self._dir, room_id)
        except PathTraversalError as exc:
            msg = (
                str(exc)
                .replace("Session ID", "A room id")
                .replace("session ID", "room id")
                .replace("session id", "room id")
            )
            raise RoomIdError(f"{room_id!r} is not a usable room id: {msg}") from exc

    def load(self, room_id: str) -> RoomState | None:
        """Return the stored room, or `None` when there is none.

        Raises:
            UnreadableRoomRecordError: A record is there and will not validate -- a
                hand-edited file, one cut short, or a shape an older build wrote. The
                parser's `ValidationError` is its `__cause__`. Re-raised as our own class
                because a `ValidationError` reads, to every route above this, as a
                malformed *request*, and its field dump was being answered to the browser
                as the reason (#1411).
        """
        path = self.room_path(room_id)
        if not path.exists():
            return None
        try:
            return RoomState.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise UnreadableRoomRecordError(room_id) from exc

    def save(self, state: RoomState) -> RoomState:
        """Persist `state` at the next revision, refusing a write that would lose an update.

        Raises:
            StaleRoomWriteError: The stored record has moved past `state.revision`.
        """
        path = self.room_path(state.room_id)
        on_disk = self.load(state.room_id)
        if on_disk is not None and on_disk.revision != state.revision:
            raise StaleRoomWriteError(
                f"Refusing to persist room {state.room_id!r}: held revision "
                f"{state.revision}, but the record is at {on_disk.revision}. A room has "
                f"one writer by design, so this means two orchestrators are driving it."
            )

        stamped = state.model_copy(
            update={"revision": state.revision + 1, "updated_at": _now_iso()}
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._dir, prefix=f".{state.room_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(stamped.model_dump_json(indent=2))
                handle.flush()
                # `os.replace` is atomic with respect to a concurrent *reader*, not to a
                # crash: without this the rename can reach disk before the bytes do, and
                # the reader the comment above promises to protect finds a zero-length
                # document instead of either version.
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return stamped

    def delete(self, room_id: str) -> bool:
        """Remove the room; return whether one was there to remove."""
        path = self.room_path(room_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_room_ids(self) -> tuple[str, ...]:
        """Every stored room id, sorted."""
        if not self._dir.exists():
            return ()
        return tuple(sorted(p.stem for p in self._dir.glob("*.json")))
