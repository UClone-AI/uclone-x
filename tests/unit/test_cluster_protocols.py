"""Unit tests for `uclone_x.core.cluster_protocols` wire envelopes and error contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uclone_x.core.cluster_protocols import (
    InterruptControlEnvelope,
    LearnCommandEnvelope,
    RoomCommandEnvelope,
    SeatTurnRef,
    SSEStreamEnvelope,
    TokenBudgetSnapshot,
    TurnCommandEnvelope,
    TurnLeaseConflictError,
    TurnLeaseToken,
    TurnPreemptedError,
)


def test_token_budget_snapshot_computes_total_consumed_tokens() -> None:
    """`TokenBudgetSnapshot.total_consumed_tokens` sums prompt and completion counts.

    Killed by: src/uclone_x/core/cluster_protocols.py :: return self.consumed_prompt_tokens + self.consumed_completion_tokens
    Becomes: return self.consumed_prompt_tokens
    """
    snapshot = TokenBudgetSnapshot(
        max_tokens=4096,
        consumed_prompt_tokens=300,
        consumed_completion_tokens=120,
        turn_count=2,
    )
    assert snapshot.total_consumed_tokens == 420
    assert snapshot.max_tokens == 4096
    assert snapshot.turn_count == 2


def test_turn_command_envelope_rejects_blank_identifiers() -> None:
    """`TurnCommandEnvelope` refuses blank identifier strings at validation time.

    Killed by: src/uclone_x/core/cluster_protocols.py :: if not value.strip():
    Becomes: if False:
    """
    valid = TurnCommandEnvelope(
        turn_id="turn-1",
        session_id="sess-1",
        agent_id="researcher",
        user_id="user-1",
        prompt="hello",
        enqueued_at_ms=1000,
    )
    assert valid.session_id == "sess-1"

    with pytest.raises(ValidationError):
        TurnCommandEnvelope(
            turn_id="   ",
            session_id="sess-1",
            agent_id="researcher",
            user_id="user-1",
            prompt="hello",
            enqueued_at_ms=1000,
        )


def test_room_command_envelope_refuses_what_could_not_be_stored() -> None:
    """A room command crosses the same stores as a turn, and fails the same way on every worker.

    Killed by: src/uclone_x/core/cluster_protocols.py :: raise ValueError(f"An identifier on RoomCommandEnvelope {problem}")
    Becomes: pass
    Killed by: src/uclone_x/core/cluster_protocols.py :: raise ValueError(f"The utterance on RoomCommandEnvelope {problem}")
    Becomes: pass
    """
    fields: dict[str, object] = {
        "kind": "ROOM_POST",
        "cascade_id": "casc-1",
        "turn_id": "turn-1",
        "room_id": "room-1",
        "user_id": "user-1",
        "enqueued_at_ms": 1000,
    }
    assert RoomCommandEnvelope.model_validate(fields).room_id == "room-1"
    with pytest.raises(ValidationError):
        RoomCommandEnvelope.model_validate({**fields, "room_id": "bad\x00room"})
    with pytest.raises(ValidationError):
        RoomCommandEnvelope.model_validate({**fields, "human_utterance": "bad\x00text"})


def test_a_seat_turn_names_its_room_on_the_wire() -> None:
    seat = TurnCommandEnvelope(
        turn_id="seat-1",
        session_id="sess-1",
        agent_id="scout",
        user_id="user-1",
        prompt="",
        seat=SeatTurnRef(room_id="room-1", cascade_id="post-1"),
        enqueued_at_ms=1000,
    )
    again = TurnCommandEnvelope.model_validate(seat.model_dump(mode="json"))
    assert again.seat == SeatTurnRef(room_id="room-1", cascade_id="post-1")


def test_sse_stream_envelope_enforces_channel_namespace_prefix() -> None:
    """`SSEStreamEnvelope` only permits `ucx:session:` or `ucx:room:` channels.

    Killed by: src/uclone_x/core/cluster_protocols.py :: if not value.startswith(("ucx:session:", "ucx:room:")):
    Becomes: if False:
    """
    env = SSEStreamEnvelope(
        channel="ucx:session:sess-42",
        idempotency_key="sess-42:1",
        data={"seq": 1, "type": "token"},
    )
    assert env.channel == "ucx:session:sess-42"

    with pytest.raises(ValidationError):
        SSEStreamEnvelope(
            channel="invalid:channel:42",
            idempotency_key="sess-42:1",
            data={"seq": 1},
        )


def test_turn_lease_and_preempted_errors_preserve_attributes() -> None:
    """Cluster lease and preemption exceptions record conversation, turn, and holder metadata.

    Killed by: src/uclone_x/core/cluster_protocols.py :: detail = f" (holder={holder})" if holder else ""
    Becomes: detail = ""
    """
    conflict = TurnLeaseConflictError("sess-9", "turn-9", holder="worker-a")
    assert conflict.conversation_id == "sess-9"
    assert conflict.turn_id == "turn-9"
    assert conflict.holder == "worker-a"
    assert "(holder=worker-a)" in str(conflict)

    preempted = TurnPreemptedError("sess-9", "turn-9")
    assert preempted.conversation_id == "sess-9"
    assert preempted.turn_id == "turn-9"
    assert "preempted" in str(preempted)

    lease = TurnLeaseToken(
        conversation_id="sess-9",
        turn_id="turn-9",
        worker_id="worker-a",
        fencing_token=7,
    )
    learn = LearnCommandEnvelope(
        job_id="job-1", room_id="room-1", user_id="user-1", from_seq=1, up_to_seq=3
    )
    ctrl = InterruptControlEnvelope(
        conversation_id="sess-9",
        turn_id="turn-9",
        requested_by="user-1",
        issued_at_ms=500,
    )
    assert lease.fencing_token == 7
    assert learn.up_to_seq == 3
    assert ctrl.turn_id == "turn-9"
