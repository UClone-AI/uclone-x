"""IT-03, Invariant 2: what a turn saved reaches the next worker's agent, and #1366's audit.

Offline this runs the agent's own session handling over the file store, across two workers. The
two-phase `MGET` over Valkey and PostgreSQL, and that a turn then makes no synchronous fetch, are
in `cluster_live/`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.cluster_harness import ClusterRig, Reply
from uclone_x.adapters.cluster.kafka_broker import TOPIC_CMD_TURN
from uclone_x.core.cluster_protocols import RoomCommandEnvelope
from uclone_x.llm.models import MessageRole, ToolCallRequest
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomFileRecord,
    RoomPolicy,
    RoomState,
)


@pytest.mark.asyncio
async def test_a_tool_result_of_one_worker_is_in_the_next_workers_request(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rig = ClusterRig(tmp_path, workspace=workspace)
    rig.script.replies.extend(
        [
            Reply(
                tool_calls=(
                    ToolCallRequest(
                        id="c0",
                        name="file_write",
                        arguments={"path": "notes.txt", "content": "the marker is 73124"},
                    ),
                )
            ),
            Reply(
                tool_calls=(
                    ToolCallRequest(id="c1", name="file_read", arguments={"path": "notes.txt"}),
                )
            ),
            Reply(text="I read it."),
            Reply(text="Still remembered."),
        ]
    )
    await rig.worker("w1").execute_single_turn(rig.command("sess-body", "read the notes"))
    await rig.worker("w2").execute_single_turn(rig.command("sess-body", "what was the marker?"))

    second_turn = rig.script.requests[3]
    assert any(
        m.role is MessageRole.TOOL and "73124" in (m.content or "") for m in second_turn.messages
    )


@pytest.mark.asyncio
async def test_a_room_step_that_crashes_leaves_turns_started_ahead_of_turns_landed(
    tmp_path: Path,
) -> None:
    """#1366: `turns_started` is durable before the speaker runs, on the real worker."""
    rig = ClusterRig(tmp_path)
    rig.rooms.save(
        RoomState(
            room_id="room_crash_audit",
            title="Crash Audit Room",
            participants=(
                Participant(id="user-1", kind=ParticipantKind.HUMAN, display_name="User"),
                Participant(
                    id="researcher",
                    kind=ParticipantKind.AGENT,
                    display_name="Researcher",
                    session_id="sess-researcher",
                ),
            ),
            policy=RoomPolicy(auto_routing=False),
            file_record=RoomFileRecord(kept_since_creation=True, turns_started=0, turns_landed=0),
        )
    )
    worker = rig.worker()
    await worker.execute_room_command(
        RoomCommandEnvelope(
            kind="ROOM_POST",
            cascade_id="casc-crash",
            turn_id="turn-crash-1",
            room_id="room_crash_audit",
            user_id="user-1",
            sender_id="user-1",
            human_utterance="Run risky tool",
            enqueued_at_ms=2000,
        )
    )

    # The seat was told to run, and dies before it lands anything.
    room = rig.rooms.load("room_crash_audit")
    assert room is not None
    assert len(rig.broker.records(TOPIC_CMD_TURN)) == 1
    assert rig.script.calls == 0
    assert room.file_record.turns_started == 1
    assert room.file_record.turns_landed == 0
