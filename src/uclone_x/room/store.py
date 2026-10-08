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

import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from uclone_x.core.session import default_session_root, resolve_session_path
from uclone_x.errors import (
    PathTraversalError,
    RoomIdError,
    StaleRoomWriteError,
    UnreadableRoomRecordError,
)
from uclone_x.room.clone_ids import migrate_room_file
from uclone_x.room.models import RoomState

__all__ = [
    "ROOMS_SUBDIR",
    "ROOM_STORAGE_DIR_ENV_VAR",
    "RoomStore",
    "default_room_storage_dir",
    "room_storage_dir_under",
]

#: Redirects the room store, as `UCLONE_SESSION_DIR` redirects the session store. Separate
#: variables for separate records: a headless run that wants its rooms elsewhere is not
#: necessarily the same request as moving its sessions, and one variable governing both
#: cannot express the difference.
logger = logging.getLogger(__name__)
ROOM_STORAGE_DIR_ENV_VAR = "UCLONE_ROOM_DIR"

#: The folder, under the session family root, that the desktop app keeps its rooms in
#: (`ui.rooms.RoomStack`). The default store is that folder, so a room a head opens is one
#: the app lists (#1837). It was `~/.uclone/rooms` until then, which the app never read;
#: rooms written there are left in place.
ROOMS_SUBDIR = "rooms"


def default_room_storage_dir() -> Path:
    """Resolve the room store's root, honouring `UCLONE_ROOM_DIR`.

    Otherwise `<session family root>/rooms`, where the desktop app reads them, so the app
    and every CLI head share one store (#1837). A sibling of the Core session store
    (`<family root>/core`), not inside it: the session store treats every `*.json` under
    its root as a session, and a room document placed there would read back as a session
    that will not validate.
    """
    override = os.environ.get(ROOM_STORAGE_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return default_session_root() / ROOMS_SUBDIR


def room_storage_dir_under(session_root: Path) -> Path:
    """Where an app whose sessions are under `session_root` keeps its rooms (#1885).

    Over the default session root -- the one every CLI head resolves -- this is the CLI's
    own store, `UCLONE_ROOM_DIR` included, so the app lists the rooms `ucx run` and the
    other heads write. It ignored the variable before, and the two could use different
    folders. An app given a storage folder of its own keeps its rooms inside it, beside
    the sessions it was pointed at (author's choice): that folder is the app's whole
    record, and a variable set for the CLI does not move part of it elsewhere.
    """
    if session_root.resolve() == default_session_root().resolve():
        return default_room_storage_dir()
    return session_root / ROOMS_SUBDIR


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
            if not room_id.strip():
                raise RoomIdError("A conversation identifier must be non-empty.") from exc
            raise RoomIdError(
                "The conversation identifier is not valid: it must not contain path separators."
            ) from exc

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
        except (ValidationError, UnicodeDecodeError):
            # A room stored before seats were keyed by clone id carries `persona` on each
            # seat, which this build refuses: it is rewritten once, then read (§4 step 3).
            # `False` does not mean the room is broken: a reader that loses the race to
            # another finds nothing left to rewrite, and the record is now the winner's.
            # Either way the file is read once more, and only that read decides.
            migrate_room_file(path)
        try:
            return RoomState.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, UnicodeDecodeError) as rewritten:
            raise UnreadableRoomRecordError(room_id) from rewritten

    def save(self, state: RoomState) -> RoomState:
        """Persist `state` at the next revision, refusing a write that would lose an update.

        Raises:
            StaleRoomWriteError: The stored record has moved past `state.revision`.
        """
        path = self.room_path(state.room_id)
        on_disk = self.load(state.room_id)
        if on_disk is not None and on_disk.revision != state.revision:
            logger.warning(
                "Refusing to persist room %r: held revision %s, but the record is at %s. A room has one writer by design.",
                state.room_id,
                state.revision,
                on_disk.revision,
            )
            raise StaleRoomWriteError(
                f"The conversation {state.room_id!r} could not be saved because it was changed by another action. Refresh and try again."
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
