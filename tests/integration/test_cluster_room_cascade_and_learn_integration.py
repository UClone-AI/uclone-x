"""IT-07: a room's cascade as one Kafka command per speaker, and `ucx.bg.learn` (Invariant 6).

The model is the only stand-in for the agent's side (`ScriptedProvider`): `RoomStepExecutor`,
`StatelessAgentWorker`, `AgentTurnExecutor`, `build_clone`, `BaseAgent` and `KnowledgeExtractor` are
real. `_Pump` hands each record to a worker in turn, as Kafka hands partitions out, and hands a
record over again when a test redelivers one. The real engines are in `cluster_live/`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.integration.cluster_harness import ClusterRig, Reply, eventually
from uclone_x.adapters.cluster.kafka_broker import (
    TOPIC_BG_LEARN,
    TOPIC_CMD_ROOM,
    TOPIC_CMD_TURN,
    ConsumedRecord,
)
from uclone_x.agent.models import PersonaDefinition
from uclone_x.core.cluster_protocols import (
    ClusterRoomStoreProtocol,
    InterruptControlEnvelope,
    LearnCommandEnvelope,
    RoomCommandEnvelope,
    TurnCommandEnvelope,
    is_worker_id,
)
from uclone_x.llm.models import ToolCallRequest
from uclone_x.llm.usage.limits import UsageLimits
from uclone_x.memory.extractor import EXTRACTION_FAILED
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomFileRecord,
    RoomMessage,
    RoomPolicy,
    RoomState,
    RoomToolUse,
    SelectionVerdict,
    SpeakerDecision,
)
from uclone_x.room.orchestrator import append_human_row, row_committed
from uclone_x.shells.cluster.worker_runner import StatelessAgentWorker

ROOM = "room-cascade"
HUMAN = "operator"
SPEAKERS = ("pioneer", "scout", "guardian")
#: Text that must never reach a person.
INTERNALS = ("AttributeError", "/srv/secret", "Traceback", "Error:", "LLMStream")


def _room(room_id: str = ROOM, *, speakers: Iterable[str] = SPEAKERS, **policy: Any) -> RoomState:
    fields: dict[str, Any] = {"auto_routing": False, **policy}
    return RoomState(
        room_id=room_id,
        title="A room",
        participants=(
            Participant(id=HUMAN, kind=ParticipantKind.HUMAN, display_name="Operator"),
            *(
                Participant(
                    id=name,
                    kind=ParticipantKind.AGENT,
                    display_name=name.capitalize(),
                    session_id=f"sess_room__{room_id}__{name}",
                )
                for name in speakers
            ),
        ),
        policy=RoomPolicy(**fields),
        file_record=RoomFileRecord(kept_since_creation=True),
    )


def _post(
    text: str, *, post_id: str = "post-1", room_id: str = ROOM, user_id: str = "user-1"
) -> RoomCommandEnvelope:
    return RoomCommandEnvelope(
        kind="ROOM_POST",
        turn_id=post_id,
        room_id=room_id,
        user_id=user_id,
        cascade_id=post_id,
        sender_id=HUMAN,
        human_utterance=text,
        enqueued_at_ms=1000,
    )


class _Pump:
    """Hands each command to the next worker: what Kafka does, with the order under test's hand."""

    def __init__(self, rig: ClusterRig, *worker_ids: str, **bounds: Any) -> None:
        self.rig = rig
        self.workers = [rig.worker(w) for w in (worker_ids or ("w1",))]
        self.ran_on: dict[str, str] = {}
        self._done: set[tuple[str, int, int]] = set()
        self._next = 0

    async def run(self, record: ConsumedRecord) -> None:
        self._done.add((record.topic, record.partition, record.offset))
        if record.topic == TOPIC_CMD_ROOM:
            await self.workers[0].execute_room_command(
                RoomCommandEnvelope.model_validate(record.payload), source=record
            )
        else:
            self._next += 1
            chosen = self.workers[self._next % len(self.workers)]
            cmd = TurnCommandEnvelope.model_validate(record.payload)
            self.ran_on[cmd.turn_id] = chosen.worker_id
            await chosen.execute_single_turn(cmd, source=record)

    def pending(self) -> list[ConsumedRecord]:
        return [
            record
            for topic in (TOPIC_CMD_ROOM, TOPIC_CMD_TURN)
            for record in self.rig.broker.records(topic)
            if (record.topic, record.partition, record.offset) not in self._done
        ]

    async def until_quiet(self) -> None:
        """Every command delivered, bounded: a room stuck retrying fails here and does not hang."""

        async def drain() -> None:
            while pending := self.pending():
                for record in pending:
                    await self.run(record)

        await asyncio.wait_for(drain(), timeout=30.0)

    def skip_all(self) -> None:
        self._done.update((r.topic, r.partition, r.offset) for r in self.pending())


def _state(rig: ClusterRig, room_id: str = ROOM) -> RoomState:
    state = rig.rooms.load(room_id)
    assert state is not None
    return state


def _spoken(state: RoomState) -> list[tuple[str, str]]:
    return [(m.sender_id, m.content) for m in state.transcript]


def _room_events(rig: ClusterRig, room_id: str = ROOM) -> list[Mapping[str, Any]]:
    return [e["payload"] for e in rig.room_stream(room_id)]


def _ended(rig: ClusterRig, room_id: str = ROOM) -> list[Mapping[str, Any]]:
    return [e for e in _room_events(rig, room_id) if e.get("event") == "ROOM_CASCADE_ENDED"]


def _replies(*texts: str) -> list[Reply]:
    return [Reply(text=text) for text in texts]


def _failing_seat(worker: StatelessAgentWorker, session_id: str) -> list[str]:
    """Make the worker's executor raise for one seat, with internals in the exception."""
    real = worker.turn_executor
    runs: list[str] = []

    async def executor(turn: TurnCommandEnvelope, bus: Any) -> Any:
        if turn.session_id != session_id:
            return await real(turn, bus)
        runs.append(turn.turn_id)
        raise AttributeError("/srv/secret/path")

    worker._turn_executor = executor  # pyright: ignore[reportPrivateUsage]
    return runs


def _assert_plain(text: object) -> None:
    for internal in INTERNALS:
        assert internal not in str(text)


async def test_a_three_speaker_cascade_runs_one_command_per_speaker_across_workers(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: for seat in outbox.seat_turns:
    Becomes: for seat in ():
    """
    rig = ClusterRig(tmp_path)
    assert isinstance(rig.rooms, ClusterRoomStoreProtocol)
    rig.rooms.save(_room(max_agent_turns_per_human_message=5))
    rig.script.replies.extend(_replies("Pioneer here.", "Scout here.", "Guardian here."))
    pump = _Pump(rig, "w1", "w2")

    await rig.ingress.submit_room_message(_post("@pioneer @scout @guardian report in"))
    await pump.until_quiet()

    state = _state(rig)
    assert _spoken(state) == [
        (HUMAN, "@pioneer @scout @guardian report in"),
        ("pioneer", "Pioneer here."),
        ("scout", "Scout here."),
        ("guardian", "Guardian here."),
    ]
    # Counted once each, and nobody holds the floor.
    assert state.file_record.turns_started == state.file_record.turns_landed == 3
    assert state.floor is None
    # One command per step: the post, then each seat's turn and its landing.
    assert [r.payload["kind"] for r in rig.broker.records(TOPIC_CMD_ROOM)] == [
        "ROOM_POST",
        "ROOM_TURN_LANDED",
        "ROOM_TURN_LANDED",
        "ROOM_TURN_LANDED",
    ]
    assert [r.payload["agent_id"] for r in rig.broker.records(TOPIC_CMD_TURN)] == list(SPEAKERS)
    # The seats were on different workers.
    assert len(set(pump.ran_on.values())) == 2

    # Each seat's own session committed the turn.
    for name, line in (("pioneer", "Pioneer here."), ("guardian", "Guardian here.")):
        assert ("assistant", line) in rig.history(f"sess_room__{ROOM}__{name}")

    # The one context builder: the room's framing, and the span the seat had not seen.
    pioneer, scout = rig.script.requests[0], rig.script.requests[1]
    assert "one participant in a shared" in (pioneer.messages[0].content or "")
    assert "Pioneer (pioneer)" in (pioneer.messages[0].content or "")
    last = scout.messages[-1].content or ""
    assert f"[{HUMAN}]: @pioneer @scout @guardian report in" in last
    assert "[pioneer]: Pioneer here." in last

    # SSE per step on the room's channel, in one rising sequence.
    stream = rig.room_stream(ROOM)
    assert [e["seq"] for e in stream] == list(range(1, len(stream) + 1))
    steps = [
        (e["payload"].get("agent_id"), e["payload"].get("status"))
        for e in stream
        if e["payload"].get("status") in {"generating", "final"}
    ]
    assert steps == [(name, status) for name in SPEAKERS for status in ("generating", "final")]
    assert any(e["payload"].get("delta") == "Pioneer here." for e in stream)
    assert _ended(rig)[-1]["reason"] == "silence"


async def test_a_cascade_runs_on_a_real_worker_process(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_process.py :: [TOPIC_CMD_TURN, TOPIC_CMD_ROOM], on_revoked=self._on_revoked
    Becomes: [TOPIC_CMD_TURN], on_revoked=self._on_revoked
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("One.", "Two."))
    worker = rig.start_worker()
    try:
        await rig.ingress.submit_room_message(_post("@pioneer @scout go"))
        await eventually(lambda: bool(_ended(rig)))
    finally:
        await worker.shut_down()
        await rig.close()
    assert _spoken(_state(rig))[1:] == [("pioneer", "One."), ("scout", "Two.")]


async def test_a_redelivered_command_of_any_kind_changes_nothing(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: if await self.send_if_committed(cmd):
    Becomes: if False:
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("One.", "Two.", "Never asked for."))
    pump = _Pump(rig, "w1", "w2")
    await rig.ingress.submit_room_message(_post("@pioneer @scout go"))
    await pump.until_quiet()
    before = _state(rig)
    calls = rig.script.calls
    commits = rig.rooms.commit_attempts

    # Everything again, on the other worker, in the order it was first delivered.
    for record in (
        *rig.broker.records(TOPIC_CMD_ROOM),
        *rig.broker.records(TOPIC_CMD_TURN),
    ):
        await pump.run(record)
    await pump.until_quiet()

    after = _state(rig)
    assert after.transcript == before.transcript
    assert after.file_record == before.file_record
    assert after.floor is None
    assert rig.script.calls == calls
    assert rig.rooms.commit_attempts == commits  # a receipt found means nothing is recomputed
    assert [m.sender_id for m in after.transcript].count(HUMAN) == 1


async def test_a_landing_delivered_twice_lands_once_while_the_next_seat_runs(
    tmp_path: Path,
) -> None:
    """The second copy arrives while the floor is the next speaker's."""
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("One.", "Two."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer @scout go")))
    await pump.run(rig.broker.records(TOPIC_CMD_TURN)[0])
    landed = next(r for r in pump.pending() if r.topic == TOPIC_CMD_ROOM)
    await pump.run(landed)
    assert _state(rig).floor is not None and _state(rig).floor.speaker_id == "scout"  # type: ignore[union-attr]

    await pump.run(landed)

    state = _state(rig)
    assert [m.sender_id for m in state.transcript] == [HUMAN, "pioneer"]
    assert state.file_record.turns_started == 2
    assert state.file_record.turns_landed == 1


async def _published(rig: ClusterRig, cmd: RoomCommandEnvelope) -> ConsumedRecord:
    await rig.ingress.submit_room_message(cmd)
    return rig.broker.records(TOPIC_CMD_ROOM)[-1]


async def test_the_start_is_durable_before_anything_is_told_to_run_it(tmp_path: Path) -> None:
    """A worker that dies on the publish leaves the start counted (#1366) and the room movable.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: update={"turns_started": record.turns_started + 1}
    Becomes: update={"turns_started": record.turns_started}
    """
    rig = ClusterRig(tmp_path, max_turn_attempts=2)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Answered."))
    seen: list[int] = []
    real_publish = rig.broker.publish_turn

    async def counted(cmd: TurnCommandEnvelope) -> int:
        seen.append(_state(rig).file_record.turns_started)
        return await real_publish(cmd)

    rig.broker.publish_turn = counted  # type: ignore[method-assign]
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer go"))
    await pump.until_quiet()
    assert seen == [1]


async def test_a_commit_whose_publish_failed_sends_again_and_does_not_stall(
    tmp_path: Path,
) -> None:
    """Commit succeeded, publish failed, the worker is gone: the redelivered copy sends it.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: await self._send(cmd.room_id, stored)
    Becomes: pass
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("One."))
    pump = _Pump(rig)
    post = await _published(rig, _post("@pioneer go"))
    real_publish = rig.broker.publish_turn

    async def down(cmd: TurnCommandEnvelope) -> int:
        raise ConnectionError("no broker")

    rig.broker.publish_turn = down  # type: ignore[method-assign]
    steps = pump.workers[0].room_steps
    assert steps is not None
    with pytest.raises(ConnectionError):
        await steps.run(RoomCommandEnvelope.model_validate(post.payload))
    state = _state(rig)
    assert state.file_record.turns_started == 1  # committed
    assert rig.broker.records(TOPIC_CMD_TURN) == []  # and nobody told

    rig.broker.publish_turn = real_publish  # type: ignore[method-assign]
    await pump.run(post)
    await pump.until_quiet()

    final = _state(rig)
    assert _spoken(final)[1:] == [("pioneer", "One.")]
    assert final.file_record.turns_started == final.file_record.turns_landed == 1
    assert final.floor is None


async def test_a_seat_commit_whose_landing_was_not_published_lands_on_redelivery(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_runner.py :: await self._broker.publish_room_step(_landed_command(cmd, cmd.seat))
    Becomes: pass
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("One.", "Never asked for."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer go")))
    seat = rig.broker.records(TOPIC_CMD_TURN)[0]
    real_publish = rig.broker.publish_room_step

    async def down(cmd: RoomCommandEnvelope) -> int:
        raise ConnectionError("no broker")

    worker = pump.workers[0]
    rig.broker.publish_room_step = down  # type: ignore[method-assign]
    worker._retry_delay_seconds = 0.0  # pyright: ignore[reportPrivateUsage]
    worker._max_turn_attempts = 1  # pyright: ignore[reportPrivateUsage]
    attempt = rig.worker("dying")
    attempt._retry_delay_seconds = 0.0  # pyright: ignore[reportPrivateUsage]

    async def die(seconds: float) -> None:
        raise SystemError("the process is gone")

    with pytest.raises(SystemError):
        await attempt.execute_single_turn(
            TurnCommandEnvelope.model_validate(seat.payload), source=seat, wait=die
        )
    assert rig.script.calls == 1
    assert _state(rig).floor is not None  # the room still waits

    rig.broker.publish_room_step = real_publish  # type: ignore[method-assign]
    await pump.run(seat)  # redelivered
    await pump.until_quiet()

    final = _state(rig)
    assert _spoken(final)[1:] == [("pioneer", "One.")]
    assert final.floor is None
    assert rig.script.calls == 1  # the model was not asked again


async def test_the_turn_ceiling_stops_the_cascade_and_is_recorded(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: stop = ceiling_decision(asked)  # the circuit breaker, before anyone is consulted
    Becomes: stop = None
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room(max_agent_turns_per_human_message=2))
    rig.script.replies.extend(_replies("One.", "Two.", "Never asked for."))
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer @scout @guardian go"))
    await pump.until_quiet()

    state = _state(rig)
    assert [m.sender_id for m in state.transcript] == [HUMAN, "pioneer", "scout"]
    assert state.file_record.turns_started == 2
    assert state.last_decision is not None
    assert "turn ceiling of 2" in state.last_decision.reasoning
    ended = _ended(rig)
    assert ended[-1]["reason"] == "turn_ceiling"
    _assert_plain(ended[-1]["message"])


async def test_an_autonomous_room_stops_at_the_twenty_turn_circuit_breaker(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    rig.rooms.save(
        _room(autonomous=True, auto_routing=True, transcript_window=40, speakers=("a1", "a2"))
    )

    def choose(name: str) -> Reply:
        return Reply(text=json.dumps({"speaker_id": name, "confidence": 1.0, "reasoning": "go"}))

    for hop in range(30):
        rig.script.replies.append(choose("a1" if hop % 2 == 0 else "a2"))
        rig.script.replies.append(Reply(text=f"Turn {hop}."))
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("Discuss among yourselves."))
    await pump.until_quiet()

    state = _state(rig)
    assert state.turn_state.agent_turns_since_human == 20
    assert state.file_record.turns_started == state.file_record.turns_landed == 20
    assert _ended(rig)[-1]["reason"] == "turn_ceiling"


async def test_a_seat_that_exhausts_its_bound_ends_the_cascade_and_the_person_is_told(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: if terminal is not None and terminal.get("dead_lettered"):
    Becomes: if False:
    """
    rig = ClusterRig(tmp_path, max_turn_attempts=2)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Never asked for."))
    pump = _Pump(rig)
    runs = _failing_seat(pump.workers[0], f"sess_room__{ROOM}__pioneer")
    await rig.ingress.submit_room_message(_post("@pioneer @scout go"))
    await pump.until_quiet()

    assert len(runs) == 2  # the turn's own policy: the bound is its attempts
    state = _state(rig)
    # The cascade ended: the next speaker named was not given the floor, and it is free.
    assert [m.sender_id for m in state.transcript] == [HUMAN, "pioneer"]
    assert state.floor is None
    assert state.file_record.turns_started == state.file_record.turns_landed == 1
    assert rig.script.calls == 0

    # The person is told, on the row and on the channel, in plain words.
    row = state.transcript[-1]
    assert row.error == "This message could not be answered. Please try again later."
    assert row.completed is True
    ended = _ended(rig)[-1]
    assert ended["reason"] == "seat_gave_up"
    assert ended["message"] == (
        "Pioneer could not answer, so the room has stopped. Send a message to try again."
    )
    _assert_plain(row.error)
    _assert_plain(_room_events(rig))
    assert [d.reason for d in rig.broker.dead_letters] == ["attempts_used"]

    # And the room is not wedged.
    rig.script.replies.clear()
    rig.script.replies.extend(_replies("Back again."))
    pump.workers[0]._turn_executor = rig.executor()  # pyright: ignore[reportPrivateUsage]
    await rig.ingress.submit_room_message(_post("@scout again?", post_id="post-2"))
    await pump.until_quiet()
    assert _spoken(_state(rig))[-1] == ("scout", "Back again.")


async def test_a_returned_failure_is_a_row_with_plain_words_and_the_cascade_goes_on(
    tmp_path: Path,
) -> None:
    """A provider's failure is a turn that ended, as on one machine: the next speaker still runs.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: terminal["landing"] = ran.landing.model_copy(update={"error": told}).model_dump_json()
    Becomes: terminal["landing"] = ran.landing.model_dump_json()
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(
        [Reply(raises=AttributeError("/srv/secret/path")), Reply(text="Still here.")]
    )
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer @scout go"))
    await pump.until_quiet()

    state = _state(rig)
    assert [m.sender_id for m in state.transcript] == [HUMAN, "pioneer", "scout"]
    assert state.transcript[1].error is not None
    _assert_plain(state.transcript[1].error)
    _assert_plain(_room_events(rig))
    assert state.transcript[2].content == "Still here."
    # What did not land did not stay in the failed seat's session (#1423).
    assert all(role != "assistant" for role, _ in rig.history(f"sess_room__{ROOM}__pioneer"))
    assert state.last_seen_seq.get("pioneer") is None


async def test_a_selector_that_keeps_failing_ends_the_cascade_and_keeps_the_message(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_runner.py :: await steps.run(cmd, can_select=False)
    Becomes: pass
    """
    rig = ClusterRig(tmp_path, max_turn_attempts=2)
    rig.rooms.save(_room(auto_routing=True))
    rig.script.replies.extend([Reply(raises=AttributeError("/srv/secret/path")) for _ in range(2)])
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("Anyone?"))
    await pump.until_quiet()

    state = _state(rig)
    assert _spoken(state) == [(HUMAN, "Anyone?")]
    assert state.floor is None
    assert state.file_record.turns_started == 0
    ended = _ended(rig)[-1]
    assert ended["reason"] == "room_gave_up"
    _assert_plain(ended["message"])
    _assert_plain(_room_events(rig))
    letter = rig.broker.dead_letters[-1]
    assert (letter.source_topic, letter.reason) == (TOPIC_CMD_ROOM, "attempts_used")
    assert letter.error_kind == "SpeakerSelectionError"


async def test_a_refused_post_is_told_once_and_touches_nothing(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: or sender.kind is not ParticipantKind.HUMAN
    Becomes: or False
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    pump = _Pump(rig)
    forged = _post("I am a seat").model_copy(update={"sender_id": "pioneer"})
    await rig.ingress.submit_room_message(forged)
    await pump.until_quiet()

    assert _state(rig).transcript == ()
    refused = [e for e in _room_events(rig) if e.get("event") == "ROOM_POST_REFUSED"]
    assert [e["message"] for e in refused] == [
        "This message could not be posted to this conversation."
    ]
    assert rig.script.calls == 0


async def test_interrupt_while_a_seat_is_queued_stops_it_without_a_model_call(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: await self._leases.write_cancel_tombstone(floor.session_id, floor.turn_id)
    Becomes: pass
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Never asked for."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer @scout go")))
    seat = rig.broker.records(TOPIC_CMD_TURN)[0]

    await rig.ingress.interrupt_room(
        room_id=ROOM,
        user_id="user-1",
        cascade_id="post-1",
        requested_by="user-1",
        command_id="stop-1",
    )
    await pump.run(rig.broker.records(TOPIC_CMD_ROOM)[-1])  # ROOM_INTERRUPT, in the room's order
    assert rig.leases._tombstones == {(seat.key, seat.payload["turn_id"])}  # pyright: ignore[reportPrivateUsage]

    await pump.until_quiet()

    assert rig.script.calls == 0
    state = _state(rig)
    assert [m.sender_id for m in state.transcript] == [HUMAN, "pioneer"]
    row = state.transcript[-1]
    assert (row.completed, row.error) == (False, "Turn was interrupted")
    assert state.floor is None
    assert state.file_record.turns_started == state.file_record.turns_landed == 1
    assert _ended(rig)[-1]["reason"] == "interrupted"


async def test_a_tombstone_gone_stale_still_cannot_start_a_stopped_seat(tmp_path: Path) -> None:
    """A command can outlive a tombstone in a queue: the room's own floor says no.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: and not floor.interrupted
    Becomes: and True
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Never asked for."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer go")))
    await rig.ingress.interrupt_room(
        room_id=ROOM,
        user_id="user-1",
        cascade_id="post-1",
        requested_by="user-1",
        command_id="stop-1",
    )
    await pump.run(rig.broker.records(TOPIC_CMD_ROOM)[-1])
    rig.leases._tombstones.clear()  # pyright: ignore[reportPrivateUsage]

    await pump.until_quiet()

    assert rig.script.calls == 0
    assert _ended(rig)[-1]["reason"] == "interrupted"


async def test_interrupt_while_a_seat_is_running_cancels_it_and_nobody_follows(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    entered, _release = rig.script.hold("stop me")
    rig.script.replies.extend(_replies("Never heard.", "Never asked for."))
    worker = rig.start_worker()
    try:
        await rig.ingress.submit_room_message(_post("@pioneer @scout stop me"))
        await eventually(entered.is_set)
        await rig.ingress.interrupt_room(
            room_id=ROOM,
            user_id="user-1",
            cascade_id="post-1",
            requested_by="user-1",
            command_id="stop-1",
        )
        await eventually(lambda: bool(_ended(rig)))
    finally:
        await worker.shut_down()
        await rig.close()

    state = _state(rig)
    assert [m.sender_id for m in state.transcript] == [HUMAN, "pioneer"]
    assert (state.transcript[-1].completed, state.transcript[-1].content) == (False, "")
    assert state.floor is None
    assert state.file_record.turns_started == state.file_record.turns_landed == 1
    assert _ended(rig)[-1]["reason"] == "interrupted"
    # Nothing of the turn stayed in the seat's session, and no other seat ran.
    assert rig.history(f"sess_room__{ROOM}__pioneer") == []
    assert rig.script.calls == 1


async def test_the_broadcast_naming_the_room_reaches_the_seat_running_in_it(
    tmp_path: Path,
) -> None:
    """It does not wait for the room's queue, so it stops a seat with nothing else sent.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: seated = [r for r in self._running.values() if r.room_id == conversation_id]
    Becomes: seated = []
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    entered, _release = rig.script.hold("stop me")
    rig.script.replies.extend(_replies("Never heard."))
    worker = rig.start_worker()
    try:
        await rig.ingress.submit_room_message(_post("@pioneer stop me"))
        await eventually(entered.is_set)
        await rig.broker.broadcast_interrupt(
            InterruptControlEnvelope(conversation_id=ROOM, requested_by="user-1", issued_at_ms=1)
        )
        await eventually(lambda: bool(_ended(rig)))
    finally:
        await worker.shut_down()
        await rig.close()
    assert _ended(rig)[-1]["reason"] == "interrupted"


async def test_interrupt_stops_the_next_speaker_when_the_turn_had_already_finished(
    tmp_path: Path,
) -> None:
    """The seat finished as the stop arrived: its reply stands, and nobody speaks after it.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: if floor.interrupted or message.completed is False:
    Becomes: if message.completed is False:
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Finished first.", "Never asked for."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer @scout go")))
    await pump.run(rig.broker.records(TOPIC_CMD_TURN)[0])
    landed = next(r for r in pump.pending() if r.topic == TOPIC_CMD_ROOM)
    # Stop reaches the room ahead of the landing.
    await rig.ingress.interrupt_room(
        room_id=ROOM,
        user_id="user-1",
        cascade_id="post-1",
        requested_by="user-1",
        command_id="stop-1",
    )
    await pump.run(rig.broker.records(TOPIC_CMD_ROOM)[-1])
    await pump.run(landed)
    await pump.until_quiet()

    state = _state(rig)
    assert _spoken(state)[1:] == [("pioneer", "Finished first.")]
    assert state.floor is None
    assert rig.script.calls == 1


async def test_a_message_after_a_stop_starts_a_cascade_of_its_own(tmp_path: Path) -> None:
    """A stop ends the cascade it stopped, and does not bind what the person says next.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: if RoomOrchestrator._interjected(cleared, said_by_then):  # pyright: ignore[reportPrivateUsage]
    Becomes: if False:
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("To the newer message."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer one")))
    await rig.ingress.interrupt_room(
        room_id=ROOM,
        user_id="user-1",
        cascade_id="post-1",
        requested_by="user-1",
        command_id="stop-1",
    )
    await pump.run(rig.broker.records(TOPIC_CMD_ROOM)[-1])
    await rig.ingress.submit_room_message(_post("@scout then you", post_id="post-2"))
    await pump.until_quiet()

    state = _state(rig)
    assert [m.sender_id for m in state.transcript] == [HUMAN, HUMAN, "pioneer", "scout"]
    assert state.transcript[-2].error == "Turn was interrupted"
    assert state.transcript[-1].content == "To the newer message."
    assert state.floor is None


async def test_a_person_who_speaks_while_a_seat_holds_the_floor_is_answered_after_it(
    tmp_path: Path,
) -> None:
    """One floor, one speaker: the newer message waits, then starts the next cascade."""
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("First.", "To the newer message.", "Never asked for."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer @guardian one")))
    seat = rig.broker.records(TOPIC_CMD_TURN)[0]
    await pump.run(await _published(rig, _post("@scout actually you", post_id="post-2")))

    # While the seat holds the floor, the message is written and nobody else is started.
    waiting = _state(rig)
    assert [m.sender_id for m in waiting.transcript] == [HUMAN, HUMAN]
    assert waiting.file_record.turns_started == 1
    assert len(rig.broker.records(TOPIC_CMD_TURN)) == 1

    await pump.run(seat)
    await pump.until_quiet()

    state = _state(rig)
    assert _spoken(state) == [
        (HUMAN, "@pioneer @guardian one"),
        (HUMAN, "@scout actually you"),
        ("pioneer", "First."),
        ("scout", "To the newer message."),
    ]
    assert state.floor is None


async def test_a_seat_turn_that_was_lost_cannot_wedge_the_room(tmp_path: Path) -> None:
    """Nothing lands: a person's next message, long after, takes the floor back.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: if age_seconds < self._floor_reclaim_seconds:
    Becomes: if True:
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Back.", "Never asked for."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer go")))
    pump.skip_all()  # the seat's command is never delivered
    held = _state(rig)
    assert held.floor is not None
    lost_turn = held.floor.turn_id
    rig.rooms.save(
        held.model_copy(update={"floor": held.floor.model_copy(update={"given_at_ms": 0})})
    )

    await rig.ingress.submit_room_message(_post("@scout anyone?", post_id="post-2"))
    await pump.until_quiet()

    state = _state(rig)
    assert [m.sender_id for m in state.transcript] == [HUMAN, HUMAN, "pioneer", "scout"]
    assert state.transcript[2].error == "This reply never arrived."
    assert state.transcript[3].content == "Back."
    assert state.floor is None
    assert state.file_record.turns_started == state.file_record.turns_landed == 2
    # If it was alive after all, it is stopped.
    assert (f"sess_room__{ROOM}__pioneer", lost_turn) in rig.leases._tombstones  # pyright: ignore[reportPrivateUsage]


_HOLD = "hold-me"


async def _say_while_a_seat_runs(rig: ClusterRig, first: str, then: str) -> None:
    """`first` puts a seat on the floor, and `then` is written while its model call is held."""
    entered, release = rig.script.hold(_HOLD)
    worker = rig.start_worker()
    try:
        await rig.ingress.submit_room_message(_post(first))
        await eventually(entered.is_set)
        await rig.ingress.submit_room_message(_post(then, post_id="post-2"))
        await eventually(lambda: len(_state(rig).transcript) == 2)
        release.set()
        await eventually(lambda: bool(_ended(rig)))
    finally:
        release.set()
        await worker.shut_down()
        await rig.close()


@pytest.mark.parametrize(
    ("speakers", "first", "then"),
    [
        (SPEAKERS, f"@pioneer {_HOLD}", "@pioneer second-ask"),
        (("pioneer",), _HOLD, "second-ask"),
    ],
    ids=["names-the-seat", "one-seat-room"],
)
async def test_a_message_sent_while_a_seat_runs_is_answered_by_that_seat(
    tmp_path: Path, speakers: tuple[str, ...], first: str, then: str
) -> None:
    """The seat's row lands after the message, and is not read as its answer.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: asked = _as_asked(state)
    Becomes: asked = state
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room(speakers=speakers))
    rig.script.replies.extend(_replies("First.", "To the newer message.", "Never asked for."))
    await _say_while_a_seat_runs(rig, first, then)

    state = _state(rig)
    assert _spoken(state) == [
        (HUMAN, first),
        (HUMAN, then),
        ("pioneer", "First."),
        ("pioneer", "To the newer message."),
    ]
    assert rig.script.calls == 2
    assert "second-ask" in "".join(rig.script.prompts_sent(1))
    assert state.floor is None
    assert _ended(rig)[-1]["reason"] == "silence"
    # The rows stay as they were written: only the choice read them the other way round.
    assert [m.seq for m in state.transcript] == [1, 2, 3, 4]


async def test_a_seats_late_row_does_not_retire_an_address_later_in_the_cascade(
    tmp_path: Path,
) -> None:
    """The newer message names two: the late row is still not an answer at the second choice."""
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("First.", "Scout.", "Pioneer again.", "Never asked for."))
    await _say_while_a_seat_runs(rig, f"@pioneer {_HOLD}", "@scout @pioneer both of you")

    assert [m.sender_id for m in _state(rig).transcript] == [
        HUMAN,
        HUMAN,
        "pioneer",
        "scout",
        "pioneer",
    ]
    assert rig.script.calls == 3


async def test_a_seats_late_row_does_not_spend_the_newer_messages_turn_budget(
    tmp_path: Path,
) -> None:
    """A ceiling of one still answers: the room's count is one before anyone answers it.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: budget = state.turn_state.agent_turns_since_human - len(overtaken)
    Becomes: budget = state.turn_state.agent_turns_since_human
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room(max_agent_turns_per_human_message=1))
    rig.script.replies.extend(_replies("First.", "To the newer message.", "Never asked for."))
    await _say_while_a_seat_runs(rig, f"@pioneer {_HOLD}", "@pioneer second-ask")

    assert _spoken(_state(rig))[-1] == ("pioneer", "To the newer message.")
    assert rig.script.calls == 2


async def test_a_seat_that_saw_the_newer_message_is_not_asked_again(tmp_path: Path) -> None:
    """The message came before the seat began, so its reply is that message's answer.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: overtaken = tuple(m for m in later if m.is_utterance and m.rendered_through < seq)
    Becomes: overtaken = tuple(m for m in later if m.is_utterance)
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Saw both.", "Never asked for."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer one")))
    seat = rig.broker.records(TOPIC_CMD_TURN)[0]
    await pump.run(await _published(rig, _post("@pioneer two", post_id="post-2")))
    await pump.run(seat)
    await pump.run(rig.broker.records(TOPIC_CMD_ROOM)[-1])  # its landing

    assert len(rig.broker.records(TOPIC_CMD_TURN)) == 1
    assert _spoken(_state(rig))[2:] == [("pioneer", "Saw both.")]
    assert rig.script.calls == 1
    assert "@pioneer two" in "".join(rig.script.prompts_sent(0))


@pytest.mark.parametrize(
    ("speakers", "first", "then"),
    [
        (SPEAKERS, "@pioneer go", "@pioneer anyone?"),
        (("pioneer",), "go", "anyone?"),
    ],
    ids=["names-the-seat", "one-seat-room"],
)
async def test_a_message_that_takes_the_floor_back_is_answered_by_the_seat_that_lost_it(
    tmp_path: Path, speakers: tuple[str, ...], first: str, then: str
) -> None:
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room(speakers=speakers))
    rig.script.replies.extend(_replies("Back.", "Never asked for."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post(first)))
    pump.skip_all()  # the seat's command is never delivered
    held = _state(rig)
    assert held.floor is not None
    rig.rooms.save(
        held.model_copy(update={"floor": held.floor.model_copy(update={"given_at_ms": 0})})
    )

    await rig.ingress.submit_room_message(_post(then, post_id="post-2"))
    await pump.until_quiet()

    state = _state(rig)
    assert [m.sender_id for m in state.transcript] == [HUMAN, HUMAN, "pioneer", "pioneer"]
    assert state.transcript[2].error == "This reply never arrived."
    assert state.transcript[3].content == "Back."
    assert rig.script.calls == 1
    assert then in "".join(rig.script.prompts_sent(0))
    assert state.floor is None


async def test_a_stop_after_a_waiting_message_starts_nobody(tmp_path: Path) -> None:
    """An interrupt wins over what was said before it: the running seat is the last to speak.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: said_by_then = max(floor.baseline_seq, floor.stopped_through)
    Becomes: said_by_then = floor.baseline_seq
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    entered, release = rig.script.hold(_HOLD)
    rig.script.replies.extend(_replies("Never heard.", "Never asked for."))
    worker = rig.start_worker()
    try:
        await rig.ingress.submit_room_message(_post(f"@pioneer {_HOLD}"))
        await eventually(entered.is_set)
        await rig.ingress.submit_room_message(_post("@scout second-msg", post_id="post-2"))
        await eventually(lambda: len(_state(rig).transcript) == 2)
        await rig.ingress.interrupt_room(
            room_id=ROOM,
            user_id="user-1",
            cascade_id="post-2",
            requested_by="user-1",
            command_id="stop-1",
        )
        await eventually(lambda: bool(_ended(rig)))
    finally:
        release.set()
        await worker.shut_down()
        await rig.close()

    state = _state(rig)
    assert [m.sender_id for m in state.transcript] == [HUMAN, HUMAN, "pioneer"]
    assert _ended(rig)[-1]["reason"] == "interrupted"
    assert rig.script.calls == 1
    assert state.file_record.turns_started == state.file_record.turns_landed == 1
    assert state.floor is None


async def test_a_stop_binds_what_came_before_it_and_not_what_comes_after(tmp_path: Path) -> None:
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("To the newest.", "Never asked for."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer one")))
    await pump.run(await _published(rig, _post("@guardian two", post_id="post-2")))
    await rig.ingress.interrupt_room(
        room_id=ROOM,
        user_id="user-1",
        cascade_id="post-2",
        requested_by="user-1",
        command_id="stop-1",
    )
    await pump.run(rig.broker.records(TOPIC_CMD_ROOM)[-1])
    await pump.run(await _published(rig, _post("@scout three", post_id="post-3")))
    await pump.until_quiet()

    state = _state(rig)
    assert [m.sender_id for m in state.transcript] == [HUMAN, HUMAN, HUMAN, "pioneer", "scout"]
    assert state.transcript[-1].content == "To the newest."
    assert rig.script.calls == 1


async def test_a_landing_that_is_not_the_floors_changes_nothing(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: if state is None or floor is None or floor.turn_id != cmd.seat_turn_id:
    Becomes: if state is None or floor is None:
    """
    rig = ClusterRig(tmp_path, max_turn_attempts=1)
    rig.rooms.save(_room())
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer go")))
    pump.skip_all()
    held = _state(rig)

    stale = _post("", post_id="landed-gone").model_copy(
        update={"kind": "ROOM_TURN_LANDED", "seat_turn_id": "seat-gone", "human_utterance": None}
    )
    await pump.run(await _published(rig, stale))

    state = _state(rig)
    assert state.floor == held.floor
    assert state.transcript == held.transcript
    assert _ended(rig) == []


async def test_a_seat_turns_account_stays_off_the_channels_a_person_reads(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_runner.py :: await self._emit(cmd, event_type, {k: v for k, v in terminal.items() if k != "landing"})
    Becomes: await self._emit(cmd, event_type, dict(terminal))
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Answered."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer go")))
    seat = TurnCommandEnvelope.model_validate(rig.broker.records(TOPIC_CMD_TURN)[0].payload)
    await pump.until_quiet()

    session = f"sess_room__{ROOM}__pioneer"
    assert rig.terminal_events(session, seat.turn_id) == ["TURN_COMPLETED"]
    for payload in (*(e["payload"] for e in rig.stream(session)), *_room_events(rig)):
        assert "landing" not in payload
    # The room read it from the receipt, which is where it is kept.
    settled = await rig.sessions.settled_terminal(session, seat.turn_id)
    assert settled is not None and "landing" in settled


async def test_a_seat_command_the_floor_does_not_name_runs_no_model(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: and floor.turn_id == cmd.turn_id
    Becomes: and True
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Never asked for."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer go")))
    real = TurnCommandEnvelope.model_validate(rig.broker.records(TOPIC_CMD_TURN)[0].payload)
    pump.skip_all()

    stale = real.model_copy(update={"turn_id": "seat-gone"})
    await rig.worker().execute_single_turn(stale)

    assert rig.script.calls == 0
    assert rig.terminal_events(real.session_id, stale.turn_id) == ["TURN_INTERRUPTED"]
    floor = _state(rig).floor
    assert floor is not None and floor.turn_id == real.turn_id


async def test_a_waiting_message_is_answered_under_the_user_who_sent_it(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: update={"cascade_id": cmd.cascade_id, "user_id": cmd.user_id}
    Becomes: update={"cascade_id": cmd.cascade_id}
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("To Alice.", "To Bob."))
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer one", user_id="alice")))
    seat = rig.broker.records(TOPIC_CMD_TURN)[0]
    await pump.run(await _published(rig, _post("@scout two", post_id="post-2", user_id="bob")))
    await pump.run(seat)
    await pump.until_quiet()

    next_seat = TurnCommandEnvelope.model_validate(rig.broker.records(TOPIC_CMD_TURN)[1].payload)
    assert (next_seat.agent_id, next_seat.user_id) == ("scout", "bob")
    assert sorted(row["scope"] for row in rig.usage.rows) == ["alice", "bob"]


def test_a_row_that_carries_an_error_is_not_committed_whatever_else_it_holds() -> None:
    """A failed turn's partial text teaches nothing.

    Killed by: src/uclone_x/room/orchestrator.py :: if row.turn_id is None or row.error is not None:
    Becomes: if row.turn_id is None:
    """
    decision = SpeakerDecision(
        verdict=SelectionVerdict.SPEAK, speaker_id="pioneer", selector="mention", reasoning="asked"
    )
    fields: dict[str, Any] = {
        "seq": 2,
        "sender_id": "pioneer",
        "content": "Half an answer",
        "decision": decision,
        "turn_id": "seat-1",
    }
    state = _room()
    assert row_committed(state, RoomMessage(**fields))
    assert not row_committed(state, RoomMessage(**fields, error="It failed."))


async def test_a_limit_reached_at_the_selector_tells_the_person_it_was_the_limit(
    tmp_path: Path,
) -> None:
    """Told at once, in the limit's own sentence, with no model call and no retry.

    Killed by: src/uclone_x/shells/cluster/room_steps.py :: if not isinstance(exc.__cause__, UsageLimitReachedError):
    Becomes: if True:
    """
    rig = ClusterRig(tmp_path, limits=UsageLimits(per_5_hours=50), max_turn_attempts=2)
    rig.rooms.save(_room(auto_routing=True))
    rig.script.replies.extend(_replies("Crosses the limit.", "Never asked for."))
    await rig.worker().execute_single_turn(rig.command("sess-other", "uses the tokens"))
    assert rig.script.calls == 1
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("Anyone?"))
    await pump.until_quiet()

    assert rig.script.calls == 1
    state = _state(rig)
    assert _spoken(state) == [(HUMAN, "Anyone?")]
    assert state.floor is None
    assert state.file_record.turns_started == 0
    ended = _ended(rig)
    assert [e["reason"] for e in ended] == ["usage_limit"]
    message = str(ended[0]["message"])
    assert message.startswith("You've reached your 5-hour limit for paid models.")
    assert "Settings" in message
    for internal in (*INTERNALS, "Selector", "provider", "SpeakerSelection", "UsageLimit"):
        assert internal not in message
    assert rig.broker.dead_letters == []


async def test_every_id_a_worker_derives_has_a_form_a_route_refuses(tmp_path: Path) -> None:
    """Named from a real cascade, so a new kind of id that is not reserved fails here.

    Killed by: src/uclone_x/core/cluster_protocols.py :: return value.startswith((SEAT_TURN_ID_PREFIX, LANDED_ID_PREFIX, LEARN_JOB_ID_PREFIX))
    Becomes: return False
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Answered."))
    pump = _Pump(rig)
    post = _post("@pioneer go")
    await rig.ingress.submit_room_message(post)
    await pump.until_quiet()

    seats = [str(r.payload["turn_id"]) for r in rig.broker.records(TOPIC_CMD_TURN)]
    room_commands = [str(r.payload["turn_id"]) for r in rig.broker.records(TOPIC_CMD_ROOM)]
    landings = [c for c in room_commands if c != post.turn_id]
    jobs = [job.job_id for job in _learn_jobs(rig)]
    assert (len(seats), len(landings), len(jobs)) == (1, 1, 1)
    for derived in (*seats, *landings, *jobs):
        assert is_worker_id(derived), derived
    assert not is_worker_id(post.turn_id)


async def test_a_seat_command_that_cannot_be_read_still_lands(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_runner.py :: await self._land_malformed_seat(record, session_id, turn_id)
    Becomes: pass
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    pump = _Pump(rig)
    await pump.run(await _published(rig, _post("@pioneer go")))
    seat = rig.broker.records(TOPIC_CMD_TURN)[0]
    unread = seat.model_copy(update={"payload": {**seat.payload, "unknown_field": 1}})
    pump.skip_all()

    await pump.workers[0].dead_letter_malformed(unread, error_kind="ValidationError")
    await pump.until_quiet()

    state = _state(rig)
    assert state.floor is None
    assert state.transcript[-1].sender_id == "pioneer"
    assert state.transcript[-1].error is not None
    _assert_plain(state.transcript[-1].error)


async def test_a_command_that_keeps_killing_its_worker_is_dead_lettered(tmp_path: Path) -> None:
    """A record that takes the process down with it cannot hold the partition for ever.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: kept_dying = starts > self._max_turn_starts
    Becomes: kept_dying = False
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.sessions.attempts[(ROOM, "post-1")] = {"starts": 10}
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer go"))
    await pump.until_quiet()

    assert [d.reason for d in rig.broker.dead_letters] == ["started_repeatedly"]
    state = _state(rig)
    assert _spoken(state) == [(HUMAN, "@pioneer go")]
    assert state.floor is None
    assert _ended(rig)[-1]["reason"] == "room_gave_up"
    assert rig.script.calls == 0


async def test_a_lost_race_on_the_revision_reads_the_room_again(tmp_path: Path) -> None:
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Answered."))
    pump = _Pump(rig)
    real = rig.rooms.commit_step
    raced: list[str] = []

    async def racing(state: RoomState | None, **fields: Any) -> RoomState | None:
        if not raced and state is not None:
            raced.append("once")
            current = _state(rig)
            rig.rooms.save(current.model_copy(update={"title": "Renamed meanwhile"}))
        return await real(state, **fields)

    rig.rooms.commit_step = racing  # type: ignore[method-assign]
    await rig.ingress.submit_room_message(_post("@pioneer go"))
    await pump.until_quiet()

    state = _state(rig)
    assert state.title == "Renamed meanwhile"
    assert [m.sender_id for m in state.transcript] == [HUMAN, "pioneer"]
    assert state.file_record.turns_started == 1


# -- what a seat may reach ----------------------------------------------------------------------


def _call(name: str, **arguments: Any) -> Reply:
    return Reply(tool_calls=(ToolCallRequest(id=f"c-{name}", name=name, arguments=arguments),))


async def _seat_writes(rig: ClusterRig, pump: _Pump, room_id: str, user: str, path: str) -> None:
    rig.script.replies.extend(
        [_call("file_write", path=path, content=f"written by {user}"), Reply(text="Done.")]
    )
    await rig.ingress.submit_room_message(
        _post("@pioneer write", room_id=room_id, user_id=user, post_id=f"post-{room_id}")
    )
    await pump.until_quiet()


async def test_a_seat_cannot_reach_another_users_files(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: _Cascade(cmd.cascade_id, cmd.user_id, row.seq, learn_from_seq, cmd.traceparent),
    Becomes: _Cascade(cmd.cascade_id, "everyone", row.seq, learn_from_seq, cmd.traceparent),
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rig = ClusterRig(tmp_path, workspace=workspace)
    rig.rooms.save(_room("room-a"))
    rig.rooms.save(_room("room-b"))
    pump = _Pump(rig)
    await _seat_writes(rig, pump, "room-a", "alice", "diary.txt")
    files = list((workspace / "users").rglob("diary.txt"))
    assert len(files) == 1
    alice_file = files[0]
    assert alice_file.read_text(encoding="utf-8") == "written by alice"

    rig.script.replies.extend(
        [
            _call("file_read", path="diary.txt"),
            _call("file_read", path=str(alice_file)),
            _call("file_read", path=f"../{alice_file.parent.name}/diary.txt"),
            _call("file_write", path=str(alice_file), content="bob was here"),
            _call("directory_list", path=".."),
            Reply(text="Done."),
        ]
    )
    await rig.ingress.submit_room_message(
        _post("@pioneer look around", room_id="room-b", user_id="bob", post_id="post-bob")
    )
    await pump.until_quiet()

    answered = "\n".join(m.content or "" for m in rig.script.requests[-1].messages)
    assert "written by alice" not in answered
    assert alice_file.read_text(encoding="utf-8") == "written by alice"
    # A human row of bob's own says what he asked, and his seat ended normally.
    assert _spoken(_state(rig, "room-b"))[-1] == ("pioneer", "Done.")


async def test_a_rooms_workspace_does_not_move_a_seat_out_of_its_users_folder(
    tmp_path: Path,
) -> None:
    """A room naming a folder is one person's machine; a worker's seat stays in its user's own.

    Killed by: src/uclone_x/shells/cluster/turn_executor.py :: own_folder = room.model_copy(update={"workspace": None})
    Becomes: own_folder = room
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    elsewhere = tmp_path / "someone-elses-folder"
    elsewhere.mkdir()
    rig = ClusterRig(tmp_path, workspace=workspace)
    rig.rooms.save(_room().model_copy(update={"workspace": str(elsewhere)}))
    pump = _Pump(rig)
    await _seat_writes(rig, pump, ROOM, "alice", "note.txt")

    assert not (elsewhere / "note.txt").exists()
    assert len(list((workspace / "users").rglob("note.txt"))) == 1


async def test_a_seat_is_billed_to_the_user_of_the_message_it_answers(tmp_path: Path) -> None:
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Answered."))
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer go", user_id="alice"))
    await pump.until_quiet()
    assert {row["scope"] for row in rig.usage.rows} == {"alice"}


# -- learning -----------------------------------------------------------------------------------


def _facts(*facts: tuple[str, str, str]) -> str:
    return json.dumps(
        {
            "facts": [
                {
                    "subject": subject,
                    "key": key,
                    "value": value,
                    "quote": value,
                    "confidence": 0.95,
                    "durable": True,
                    "why": "said so",
                }
                for subject, key, value in facts
            ]
        }
    )


def _memory_for(tmp_path: Path) -> Any:
    held: dict[str, CrossSessionMemory] = {}

    def store(clone_id: str, user_id: str) -> CrossSessionMemory:
        key = f"{user_id}-{clone_id}"
        if key not in held:
            held[key] = CrossSessionMemory(
                storage_path=tmp_path / f"{key}.sqlite3", agent_id=clone_id
            )
        return held[key]

    return store


def _learn_jobs(rig: ClusterRig) -> list[LearnCommandEnvelope]:
    return [
        LearnCommandEnvelope.model_validate(r.payload) for r in rig.broker.records(TOPIC_BG_LEARN)
    ]


async def test_a_finished_cascade_learns_off_the_critical_path_and_notes_it_on_the_row(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: for job in outbox.learn_jobs:
    Becomes: for job in ():
    """
    memory = _memory_for(tmp_path)
    rig = ClusterRig(tmp_path, fact_store_for=memory)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Noted, Seoul it is."))
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer I live in Seoul"))
    await pump.until_quiet()

    # The cascade ended and landed with nothing learned yet: the job is on its own topic.
    before = _state(rig)
    assert before.transcript[-1].knowledge_learned == ()
    jobs = _learn_jobs(rig)
    assert [(j.from_seq, j.up_to_seq, j.user_id) for j in jobs] == [(0, 2, "user-1")]

    rig.script.replies.append(Reply(text=_facts(("user", "lives_in", "Seoul"))))
    await rig.learner().process_learn_job(jobs[0])

    facts = memory("pioneer", "user-1").list_facts(subject="user")
    assert [(f.predicate, f.object_value) for f in facts] == [("lives_in", "Seoul")]
    row = _state(rig).transcript[-1]
    assert row.knowledge_learned == (facts[0].fact_id,)
    assert row.knowledge_extract_error is None
    # Under the room's user, in the usage ledger like any model call.
    assert {r["scope"] for r in rig.usage.rows} == {"user-1"}

    # A job redelivered is a job done.
    calls = rig.script.calls
    await rig.learner().process_learn_job(jobs[0])
    assert rig.script.calls == calls
    assert _state(rig).transcript[-1].knowledge_learned == (facts[0].fact_id,)


async def test_a_worker_with_nowhere_to_keep_facts_asks_no_model(tmp_path: Path) -> None:
    """`ucx worker` today. The job is acknowledged, and nothing pretends a fact was saved.

    Killed by: src/uclone_x/shells/cluster/learn_worker.py :: if self._fact_store_for is None:
    Becomes: if False:
    """
    rig = ClusterRig(tmp_path)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Noted."))
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer I live in Seoul"))
    await pump.until_quiet()
    calls = rig.script.calls
    job = _learn_jobs(rig)[0]

    await rig.learner().process_learn_job(job)

    assert rig.script.calls == calls
    assert (ROOM, job.job_id) in rig.rooms.receipts
    assert _state(rig).transcript[-1].knowledge_learned == ()
    assert _state(rig).transcript[-1].knowledge_extract_error is None


async def test_a_failing_learn_job_leaves_the_room_intact(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/learn_worker.py :: logger.exception("Learn job %s failed; its room is untouched", job.job_id)
    Becomes: raise
    """
    rig = ClusterRig(tmp_path, fact_store_for=_memory_for(tmp_path))
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Noted."))
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer I live in Seoul"))
    await pump.until_quiet()
    job = _learn_jobs(rig)[0]
    before = _state(rig)

    # The model fails: a plain note on the row, and the room otherwise as it was.
    rig.script.replies.append(Reply(raises=AttributeError("/srv/secret/path")))
    await rig.learner().process_learn_job(job)
    after = _state(rig)
    assert after.transcript[-1].knowledge_extract_error == EXTRACTION_FAILED
    _assert_plain(EXTRACTION_FAILED)
    assert after.transcript[:-1] == before.transcript[:-1]
    assert after.transcript[-1].content == before.transcript[-1].content
    assert after.floor is None

    # The store it would read fails: nothing is written, and nothing is raised.
    other = LearnCommandEnvelope(
        job_id="learn-other", room_id=ROOM, user_id="user-1", from_seq=0, up_to_seq=2
    )
    real_read = rig.rooms.read_room

    async def down(room_id: str) -> RoomState | None:
        raise ConnectionError("/srv/secret/path")

    learner = rig.learner()
    rig.rooms.read_room = down  # type: ignore[method-assign]
    await learner.process_learn_job(other)
    rig.rooms.read_room = real_read  # type: ignore[method-assign]
    assert _state(rig).revision == after.revision

    # And the room goes on.
    rig.script.replies.extend(_replies("Still here."))
    await rig.ingress.submit_room_message(_post("@pioneer still there?", post_id="post-2"))
    await pump.until_quiet()
    assert _spoken(_state(rig))[-1] == ("pioneer", "Still here.")


async def test_a_held_lesson_holds_up_no_room_on_a_real_worker_process(tmp_path: Path) -> None:
    """A lesson is off the critical path: one that never returns stops nothing.

    Killed by: src/uclone_x/shells/cluster/worker_process.py :: self._learn_dispatcher.dispatch(
    Becomes: self._dispatcher.dispatch(
    """
    rig = ClusterRig(tmp_path, fact_store_for=_memory_for(tmp_path))
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("First.", "Second.", _facts()))
    # Only the lesson's request holds the seat's own reply as text to read.
    in_lesson, release = rig.script.hold("First.")
    worker = rig.start_worker()
    try:
        await rig.ingress.submit_room_message(_post("@pioneer I live in Seoul"))
        await eventually(in_lesson.is_set)

        await rig.ingress.submit_room_message(_post("@pioneer and now?", post_id="post-2"))
        await eventually(lambda: _spoken(_state(rig))[-1] == ("pioneer", "Second."))
        release.set()
        await eventually(lambda: (ROOM, _learn_jobs(rig)[0].job_id) in rig.rooms.receipts)
    finally:
        release.set()
        await worker.shut_down()
        await rig.close()


async def test_the_learn_switch_off_produces_no_job(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: if not any(self._clone_learns(clone_id) for clone_id in speakers):
    Becomes: if False:
    """
    rig = ClusterRig(tmp_path, fact_store_for=_memory_for(tmp_path))
    rig.personas.register_persona(
        PersonaDefinition(
            name="pioneer", role="Pioneer", system_prompt="Go.", learn_from_conversations=False
        )
    )
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Answered."))
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer I live in Seoul"))
    await pump.until_quiet()

    assert _spoken(_state(rig))[-1] == ("pioneer", "Answered.")
    assert _learn_jobs(rig) == []


async def test_a_cascade_with_one_clone_learning_skips_the_one_that_does_not(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/learn_worker.py :: learns=lambda: self._clones.clone_learns(speaker.id),
    Becomes: learns=lambda: True,
    """
    memory = _memory_for(tmp_path)
    rig = ClusterRig(tmp_path, fact_store_for=memory)
    rig.personas.register_persona(
        PersonaDefinition(
            name="scout", role="Scout", system_prompt="Go.", learn_from_conversations=False
        )
    )
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("One.", "Two."))
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer @scout I live in Seoul"))
    await pump.until_quiet()
    jobs = _learn_jobs(rig)
    assert len(jobs) == 1

    calls = rig.script.calls
    rig.script.replies.append(Reply(text=_facts(("user", "lives_in", "Seoul"))))
    await rig.learner().process_learn_job(jobs[0])

    assert rig.script.calls == calls + 1  # the one that learns
    assert memory("scout", "user-1").list_facts(subject="user") == []


async def test_a_learn_job_that_cannot_be_made_does_not_hold_up_the_cascades_end(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/room_steps.py :: except Exception:  # a lesson lost costs a lesson; ending the cascade must not depend on it
    Becomes: except ZeroDivisionError:
    """
    rig = ClusterRig(tmp_path, max_turn_attempts=1)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Answered."))
    pump = _Pump(rig)

    def unreadable_persona(clone_id: str) -> bool:
        raise AttributeError("/srv/secret/path")

    steps = pump.workers[0].room_steps
    assert steps is not None
    steps._clone_learns = unreadable_persona  # pyright: ignore[reportPrivateUsage]
    await rig.ingress.submit_room_message(_post("@pioneer go"))
    await pump.until_quiet()

    state = _state(rig)
    assert _spoken(state)[-1] == ("pioneer", "Answered.")
    assert state.floor is None
    assert _ended(rig)[-1]["reason"] == "silence"
    assert _learn_jobs(rig) == []
    assert rig.broker.dead_letters == []


async def test_the_extractor_never_reads_its_own_tool_io(tmp_path: Path) -> None:
    """Its memory tools and a story's contents are not sources (#2092).

    Killed by: src/uclone_x/room/orchestrator.py :: and not use.tool_name.startswith(EXCLUDED_TOOL_PREFIXES)
    Becomes: and True
    """
    rig = ClusterRig(tmp_path, fact_store_for=_memory_for(tmp_path))
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Noted."))
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer I live in Seoul"))
    await pump.until_quiet()
    state = _state(rig)
    row = state.transcript[-1]

    def use(tool: str, output: str, status: str = "success") -> RoomToolUse:
        return RoomToolUse(
            turn_id=row.turn_id or "",
            participant_id="pioneer",
            tool_name=tool,
            tool_call_id=f"c-{tool}",
            status=status,
            duration_ms=1.0,
            output_preview=output,
        )

    rig.rooms.save(
        state.model_copy(
            update={
                "tool_uses": (
                    use("record_memory_fact", "ECHO-OF-OWN-MEMORY"),
                    use("story_library", "ECHO-OF-A-STORY"),
                    use("file_read", "FROM-A-FILE"),
                    use("file_write", "FAILED-CALL", status="error"),
                )
            }
        )
    )

    rig.script.replies.append(Reply(text=_facts()))
    await rig.learner().process_learn_job(_learn_jobs(rig)[0])

    asked = "\n".join(m.content or "" for m in rig.script.requests[-1].messages)
    assert "FROM-A-FILE" in asked
    assert "ECHO-OF-OWN-MEMORY" not in asked
    assert "ECHO-OF-A-STORY" not in asked
    assert "FAILED-CALL" not in asked


async def test_a_note_lost_to_a_persons_reply_goes_on_top_of_it(tmp_path: Path) -> None:
    """The learner never writes over a reply that landed while it was learning.

    Killed by: src/uclone_x/shells/cluster/learn_worker.py :: except StaleRoomWriteError:
    Becomes: except ZeroDivisionError:
    """
    memory = _memory_for(tmp_path)
    rig = ClusterRig(tmp_path, fact_store_for=memory)
    rig.rooms.save(_room())
    rig.script.replies.extend(_replies("Noted."))
    pump = _Pump(rig)
    await rig.ingress.submit_room_message(_post("@pioneer I live in Seoul"))
    await pump.until_quiet()
    job = _learn_jobs(rig)[0]

    learner = rig.learner()
    real_commit = rig.rooms.commit_step
    raced: list[str] = []

    async def racing(state: RoomState | None, **fields: Any) -> RoomState | None:
        if not raced:
            raced.append("once")
            rig.rooms.save(append_human_row(_state(rig), HUMAN, "A reply meanwhile."))
        return await real_commit(state, **fields)

    rig.rooms.commit_step = racing  # type: ignore[method-assign]
    rig.script.replies.append(Reply(text=_facts(("user", "lives_in", "Seoul"))))
    await learner.process_learn_job(job)

    state = _state(rig)
    assert state.transcript[-1].content == "A reply meanwhile."
    agent_row: RoomMessage = state.transcript[-2]
    assert len(agent_row.knowledge_learned) == 1
    assert (ROOM, job.job_id) in rig.rooms.receipts
