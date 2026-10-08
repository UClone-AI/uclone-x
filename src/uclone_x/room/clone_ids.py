"""Room seats keyed by clone id: migration step 3 of clone-data-scopes §3.8 (§4 step 3).

A room stored before this step seats each clone by its handle, and each seat carries a
`persona` that may name a different clone. This rewrites such a record once, in one save:

* every agent seat becomes a seat of the clone its `persona` names (else the clone its id
  names), keyed by that clone's id; a persona naming no clone is created as one;
* two seats of one clone in one room keep the first, and the other leaves the membership;
* every clone message records the session it was written in, derived from the handle it
  was written under -- before that sender is rewritten, since the trace is read from it;
* a kept seat's messages, decisions, marks and tool records follow it to the id, while a
  dropped seat's keep their original author;
* every kept seat keeps its stored `session_id`, which was derived from the old id.

The trigger is the `persona` key, which every seat written before this step carries and no
seat written after it does, so a record is rewritten once. Each move is reported to the
agents root's migration log (`.migration-<date>.log`), as the clone store's own are.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from uclone_x.core.agent_home import (
    AgentHomeError,
    clone_handles,
    clone_root_lock,
    create_clone,
    default_agents_root,
    is_agent_id,
)
from uclone_x.room.service import participant_session_id

__all__ = ["migrate_room_file", "needs_clone_id_migration", "rewrite_room_record"]

logger = logging.getLogger(__name__)

_AGENT = "agent"
_UTTERANCE = "utterance"


def _object(value: object) -> dict[str, Any] | None:
    """`value` as a JSON object, or None when it is not one."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _array(value: object) -> list[Any]:
    """`value` as a JSON array; anything else holds nothing to rewrite."""
    return cast("list[Any]", value) if isinstance(value, list) else []


def needs_clone_id_migration(data: object) -> bool:
    """Whether a stored room's JSON still seats anybody the way rooms did before step 3."""
    record = _object(data)
    if record is None:
        return False
    return any(
        (seat := _object(entry)) is not None and "persona" in seat
        for entry in _array(record.get("participants"))
    )


def rewrite_room_record(
    data: dict[str, Any], *, root: Path | None = None, create_missing: bool = True
) -> list[str]:
    """Rewrite the stored room `data` in place to seat clones by id; return what moved.

    `data` is the room's JSON object as stored. Nothing is written but the clones a seat's
    `persona` names and that do not exist yet (`create_missing`).
    """
    base = root if root is not None else default_agents_root()
    room_id = str(data.get("room_id", ""))
    lines: list[str] = []
    claims = clone_handles(base)

    def clone_of(name: str) -> str | None:
        if is_agent_id(name):
            return name if (base / name).is_dir() else None
        owners = claims.get(name, ())
        return owners[0] if len(owners) == 1 else None

    rename: dict[str, str] = {}
    dropped: dict[str, str] = {}
    kept_by_clone: dict[str, str] = {}
    seats: list[Any] = []
    for entry in _array(data.get("participants")):
        seat = _object(entry)
        if seat is None:
            seats.append(entry)
            continue
        persona: object = seat.pop("persona", "")
        if seat.get("kind") != _AGENT:
            seats.append(seat)
            continue
        old = str(seat.get("id", ""))
        named = persona if isinstance(persona, str) and persona and persona != old else old
        target = clone_of(named)
        if target is None and named != old and create_missing:
            target = _created(named, base, lines, room_id, old)
            if target is not None:
                claims = clone_handles(base)
        if target is None and named != old:
            target = clone_of(old)
            lines.append(
                f"room {room_id}: seat '{old}' named persona '{named}', which is no clone, so "
                "it stays a seat of the clone its own name names"
            )
        if target is None:
            lines.append(f"room {room_id}: seat '{old}' names no clone, so it was kept as it is")
            target = old
        first = kept_by_clone.get(target)
        if first is not None:
            dropped[old] = target
            lines.append(
                f"room {room_id}: seat '{old}' was a second seat of the clone {target} "
                f"(kept: '{first}'), so it left the room; its messages keep their author"
            )
            continue
        kept_by_clone[target] = old
        if target != old:
            rename[old] = target
            seat["id"] = target
            lines.append(f"room {room_id}: seat '{old}' is now the clone {target}")
        seats.append(seat)
    data["participants"] = seats

    for entry in _array(data.get("transcript")):
        message = _object(entry)
        if message is None:
            continue
        sender: object = message.get("sender_id")
        # Every clone message records the session it was written in, derived from the
        # sender it was written under -- before that sender is rewritten below.
        if (
            message.get("turn_id") is not None
            and message.get("kind", _UTTERANCE) == _UTTERANCE
            and not message.get("session_id")
            and isinstance(sender, str)
        ):
            message["session_id"] = participant_session_id(room_id, sender)
        if isinstance(sender, str) and sender in rename:
            message["sender_id"] = rename[sender]
        decision = _object(message.get("decision"))
        if decision is not None:
            _follow(decision, "speaker_id", rename)

    for holder_key, key in (("turn_state", "last_speaker_id"), ("last_decision", "speaker_id")):
        holder = _object(data.get(holder_key))
        if holder is not None:
            _follow(holder, key, rename, dropped)
    policy = _object(data.get("policy"))
    if policy is not None:
        _follow(policy, "default_responder_id", rename, dropped)
    for records in ("tool_uses", "written_files"):
        for entry in _array(data.get(records)):
            record = _object(entry)
            if record is not None:
                _follow(record, "participant_id", rename)
    seen = _object(data.get("last_seen_seq"))
    if seen is not None:
        marks: dict[str, Any] = {}
        for who, mark in seen.items():
            if who in dropped:
                continue  # the kept seat's mark is the clone's
            marks[rename.get(who, who)] = mark
        data["last_seen_seq"] = marks
    return lines


def _created(handle: str, base: Path, lines: list[str], room_id: str, seat: str) -> str | None:
    """A new clone called `handle`, for a seat whose persona names none; None if refused."""
    try:
        home = create_clone(handle, f"handle: {json.dumps(handle)}\n", root=base)
    except AgentHomeError as exc:
        lines.append(
            f"room {room_id}: seat '{seat}' named persona '{handle}', which is no clone and "
            f"could not be created: {exc}"
        )
        return None
    lines.append(
        f"room {room_id}: seat '{seat}' named persona '{handle}', which was no clone; "
        f"it was created as {home.path.name}"
    )
    return home.path.name


def _follow(
    holder: dict[str, Any],
    key: str,
    rename: dict[str, str],
    dropped: dict[str, str] | None = None,
) -> None:
    value = holder.get(key)
    if not isinstance(value, str):
        return
    if value in rename:
        holder[key] = rename[value]
    elif dropped is not None and value in dropped:
        holder[key] = dropped[value]


def migrate_room_file(path: Path, *, root: Path | None = None) -> bool:
    """Rewrite the room stored at `path` if it still seats by handle; True when it did.

    Under the agents root's lock, re-read inside it, so two processes opening one room
    rewrite it once. The record is replaced atomically at the same revision, and each move
    is appended to the migration log. A document that is not a room's JSON is left for the
    store's own validation to report.
    """
    base = root if root is not None else default_agents_root()
    try:
        with clone_root_lock(base):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if not needs_clone_id_migration(data):
                return False
            lines = rewrite_room_record(data, root=base)
            # The revision is left as it is: the conversation did not change, only how it
            # names its seats, and only `RoomStore.save` produces a revision.
            _replace(path, json.dumps(data, indent=2, ensure_ascii=False))
            _report(base, lines or [f"room {data.get('room_id')}: seats already by id"])
    except AgentHomeError:
        logger.warning("room %s could not be migrated to clone ids", path, exc_info=True)
        return False
    return True


def _replace(path: Path, text: str) -> None:
    fd, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise


def _report(base: Path, lines: list[str]) -> None:
    now = _dt.datetime.now(_dt.UTC)
    stamp = now.isoformat(timespec="seconds")
    try:
        base.mkdir(parents=True, exist_ok=True)
        with (base / f".migration-{now.date().isoformat()}.log").open("a", encoding="utf-8") as log:
            for line in lines:
                log.write(f"{stamp} {line}\n")
                logger.info("room migration: %s", line)
    except OSError:
        logger.warning("the room migration report could not be written", exc_info=True)
