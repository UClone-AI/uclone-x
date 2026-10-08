"""Tests for RoomStore persistence and loading behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.errors import UnreadableRoomRecordError
from uclone_x.room.store import RoomStore


def test_room_store_load_invalid_utf8_raises_unreadable_room_record_error(tmp_path: Path) -> None:
    """A room record containing invalid UTF-8 bytes raises UnreadableRoomRecordError (#1615).

    Killed by: src/uclone_x/room/store.py :: except (ValidationError, UnicodeDecodeError) as rewritten:
    Becomes: except ValidationError as rewritten:
    """
    store = RoomStore(tmp_path / "rooms")
    (tmp_path / "rooms").mkdir(parents=True, exist_ok=True)
    room_file = store.room_path("r1")
    room_file.write_bytes(b"\xff\xfe\xfd\x80")
    with pytest.raises(UnreadableRoomRecordError) as exc_info:
        store.load("r1")
    assert exc_info.value.room_id == "r1"
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)
