"""Fields retired with the per-seat rules engine still load from a stored room (step 6).

Clone-knowledge-graph step 6 removed a seat's own knowledge record and rules engine. Rooms
and rows written before it still carry three fields on disk: a seat's `ontology_namespace`
and a row's `knowledge_persist_error` / `knowledge_set_aside`. These pin that such a room
still loads (the models forbid unknown fields, so dropping the declarations would refuse
it), and that nothing written after step 6 carries the seat field or an unset row field.
"""

from __future__ import annotations

import json

from uclone_x.room.models import Participant, RoomMessage


def test_a_seat_stored_with_its_old_namespace_loads_and_is_written_without_it() -> None:
    stored = {
        "id": "scout",
        "kind": "agent",
        "display_name": "Scout",
        "aliases": ["sc"],
        "session_id": "sess_room__r__scout",
        "ontology_namespace": "room_r/scout",
    }
    seat = Participant.model_validate_json(json.dumps(stored))
    assert seat.aliases == ("sc",)
    assert "ontology_namespace" not in seat.model_dump(mode="json")


def test_a_row_stored_with_the_retired_notices_loads_and_keeps_what_it_said() -> None:
    stored = {
        "seq": 1,
        "sender_id": "scout",
        "content": "hi",
        "knowledge_persist_error": "OSError: disk full",
        "knowledge_set_aside": True,
    }
    row = RoomMessage.model_validate_json(json.dumps(stored))
    written = row.model_dump(mode="json")
    assert written["knowledge_persist_error"] == "OSError: disk full"
    assert written["knowledge_set_aside"] is True


def test_a_row_written_now_carries_neither_retired_notice() -> None:
    written = RoomMessage(seq=1, sender_id="scout", content="hi").model_dump(mode="json")
    assert "knowledge_persist_error" not in written
    assert "knowledge_set_aside" not in written
