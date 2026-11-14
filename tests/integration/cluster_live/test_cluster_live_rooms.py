"""A room's cascade on real `ucx worker` processes, against the docker stack.

Kafka, PostgreSQL, Valkey and Centrifugo are real, and so is each worker's `GeminiConnector`, which
reaches a scripted model over a real socket. A worker is killed with SIGKILL, so nothing in it
gets to clean up.

Mutation evidence for these tests is recorded in the PR, not as `Killed by:` declarations: the
gate's lethality ratchet would run them offline, where they are skipped.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from aiokafka.admin import AIOKafkaAdminClient
from aiokafka.coordinator.protocol import ConsumerProtocolMemberAssignment
from aiokafka.partitioner import murmur2  # pyright: ignore[reportUnknownVariableType]

from tests.integration.cluster_live.conftest import eventually, raw_valkey
from tests.integration.cluster_live.scripted_gemini import ScriptedGemini, ScriptedTurn
from tests.integration.cluster_live.test_cluster_live_worker import WorkerProcesses
from uclone_x.adapters.cluster.config import ClusterRuntimeConfig
from uclone_x.adapters.cluster.kafka_broker import (
    DEFAULT_TOPIC_PARTITIONS,
    TOPIC_BG_LEARN,
    TOPIC_CMD_TURN,
    TOPIC_DEAD_LETTER,
    KafkaCommandBroker,
)
from uclone_x.adapters.cluster.turn_lease import ValkeyTurnLeaseManager
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.persona_store import PersonaDraft
from uclone_x.cli.commands.worker import WORKER_GROUP
from uclone_x.core.cluster_protocols import (
    RoomCommandEnvelope,
    RoomStepOutbox,
    SeatTurnRef,
    TurnCommandEnvelope,
    encode_versioned_head,
)
from uclone_x.errors import StaleRoomWriteError
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.usage.gate import UsageGate, gate_if_paid
from uclone_x.llm.usage.limits import UsageLimits
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomFileRecord,
    RoomPolicy,
    RoomState,
)
from uclone_x.shells.cluster.room_steps import (
    RoomStepExecutor,
    _seat_turn_id,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.shells.cluster.runtime_stack import ClusterRuntimeStack, create_cluster_http_app
from uclone_x.shells.cluster.turn_executor import AgentTurnExecutor
from uclone_x.shells.cluster.worker_runner import StatelessAgentWorker

pytestmark = [pytest.mark.cluster_live, pytest.mark.allow_network]

_WAIT_SECONDS = 60.0
HUMAN = "operator"
INTERNALS = ("JSON", "pydantic", "ValidationError", "state_json", "Traceback", "/")


def _uid(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


@pytest.fixture
def provider() -> Iterator[ScriptedGemini]:
    server = ScriptedGemini()
    yield server
    server.close()


@pytest.fixture
def workers(tmp_path: Path, provider: ScriptedGemini) -> Iterator[WorkerProcesses]:
    running = WorkerProcesses(tmp_path, provider)
    yield running
    running.close()


def _seat_session(room_id: str, name: str) -> str:
    return f"sess_room__{room_id}__{name}"


def _room(room_id: str, speakers: Sequence[str], **policy: Any) -> RoomState:
    fields: dict[str, Any] = {"auto_routing": False, "max_agent_turns_per_human_message": 5}
    return RoomState(
        room_id=room_id,
        title="A room",
        participants=(
            Participant(id=HUMAN, kind=ParticipantKind.HUMAN, display_name="Operator"),
            *(
                Participant(
                    id=name,
                    kind=ParticipantKind.AGENT,
                    display_name=name,
                    session_id=_seat_session(room_id, name),
                )
                for name in speakers
            ),
        ),
        policy=RoomPolicy(**{**fields, **policy}),
        file_record=RoomFileRecord(kept_since_creation=True),
    )


async def _seed(stack: ClusterRuntimeStack, room: RoomState) -> None:
    await asyncio.to_thread(stack.room_store.save, room)


def _post(room_id: str, text: str, *, post_id: str | None = None) -> RoomCommandEnvelope:
    turn_id = post_id or _uid("post-")
    return RoomCommandEnvelope(
        kind="ROOM_POST",
        turn_id=turn_id,
        room_id=room_id,
        user_id="room-user",
        cascade_id=turn_id,
        sender_id=HUMAN,
        human_utterance=text,
        enqueued_at_ms=int(time.time() * 1000),
    )


async def _state(stack: ClusterRuntimeStack, room_id: str) -> RoomState:
    """The room as PostgreSQL holds it: the cache's word is not evidence here."""
    async with stack.sql_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT state_json FROM room_states WHERE room_id = $1", room_id)
    assert row is not None
    return RoomState.model_validate_json(str(row["state_json"]))


async def _events(stack: ClusterRuntimeStack, room_id: str) -> list[Mapping[str, Any]]:
    """The room's channel, as it crossed Kafka into Centrifugo's history, in sequence order."""
    history = await stack.channel_history(f"ucx:room:{room_id}", limit=256)
    ordered = sorted(history, key=lambda e: int(e.data["seq"]))
    return [e.data["payload"] for e in ordered]


async def _ended(stack: ClusterRuntimeStack, room_id: str) -> Mapping[str, Any]:
    async def probe() -> Mapping[str, Any] | None:
        for payload in await _events(stack, room_id):
            if payload.get("event") == "ROOM_CASCADE_ENDED":
                return payload
        return None

    return await eventually(probe, timeout=_WAIT_SECONDS)


async def _entered(turn: ScriptedTurn) -> None:
    await eventually(lambda: asyncio.to_thread(turn.entered.wait, 0.1), timeout=_WAIT_SECONDS)


async def _lease_holder(stack: ClusterRuntimeStack, session_id: str) -> str:
    client = await raw_valkey(stack)
    raw = await client.get(ValkeyTurnLeaseManager.lease_key(session_id))
    assert raw is not None
    return str(json.loads(raw)["worker_id"])


def _spoken(state: RoomState) -> list[tuple[str, str]]:
    return [(m.sender_id, m.content) for m in state.transcript]


def _partition_of(session_id: str) -> int:
    partitions = DEFAULT_TOPIC_PARTITIONS[TOPIC_CMD_TURN]
    return (murmur2(session_id.encode()) & 0x7FFFFFFF) % partitions


async def _worker_partitions(config: ClusterRuntimeConfig, members: int) -> list[set[int]]:
    """Which partitions of `ucx.cmd.turn` each worker owns, once the group settles on `members`."""

    async def probe() -> list[set[int]] | None:
        admin: Any = AIOKafkaAdminClient(bootstrap_servers=",".join(config.kafka_bootstrap_servers))
        await admin.start()
        try:
            described = await admin.describe_consumer_groups([WORKER_GROUP])
        finally:
            await admin.close()
        group = described[0].groups[0]
        state, found = group[2], group[5]
        if state != "Stable" or len(found) != members:
            return None
        owned: list[set[int]] = []
        for member in found:
            assignment = ConsumerProtocolMemberAssignment.decode(member[4])
            owned.append({p for t, p in assignment.partitions() if t == TOPIC_CMD_TURN})
        return owned if all(owned) else None

    return await eventually(probe, timeout=_WAIT_SECONDS)


def _speaker_on(room_id: str, partitions: set[int]) -> str:
    while True:
        name = _uid("s")
        if _partition_of(_seat_session(room_id, name)) in partitions:
            return name


def _mentions(speakers: Sequence[str]) -> str:
    return " ".join(f"@{name}" for name in speakers)


async def test_a_three_speaker_cascade_runs_on_two_workers_with_a_stream_per_step(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    """Each seat's session is on a partition another worker owns, and the lease says who ran it."""
    first, second = workers.start(), workers.start()
    owned = await _worker_partitions(cluster_config, 2)
    room_id = _uid("room-")
    speakers = [
        _speaker_on(room_id, owned[0]),
        _speaker_on(room_id, owned[1]),
        _speaker_on(room_id, owned[0]),
    ]
    await _seed(stack, _room(room_id, speakers))
    holds = [threading.Event() for _ in speakers]
    turns = [
        provider.script("report-in", "Reply one.", hold=holds[0]),
        provider.script("Reply one.", "Reply two.", hold=holds[1]),
        provider.script("Reply two.", "Reply three.", hold=holds[2]),
    ]

    post = _post(room_id, f"{_mentions(speakers)} report-in")
    accepted = await stack.ingress.submit_room_message(post)
    assert accepted.sse_channel == f"ucx:room:{room_id}"

    ran_on: list[str] = []
    for index, name in enumerate(speakers):
        await _entered(turns[index])
        ran_on.append(await _lease_holder(stack, _seat_session(room_id, name)))
        held = await _state(stack, room_id)
        # Durable before the seat ran, and exactly this far (#1366).
        assert held.file_record.turns_started == index + 1
        assert held.file_record.turns_landed == index
        assert held.floor is not None and held.floor.speaker_id == name
        holds[index].set()
    assert len(set(ran_on)) == 2

    ended = await _ended(stack, room_id)
    assert ended["reason"] == "silence"
    state = await _state(stack, room_id)
    assert _spoken(state) == [
        (HUMAN, post.human_utterance),
        (speakers[0], "Reply one."),
        (speakers[1], "Reply two."),
        (speakers[2], "Reply three."),
    ]
    assert state.file_record.turns_started == state.file_record.turns_landed == 3
    assert state.floor is None
    # Each seat's own session committed, with its receipt.
    async with stack.sql_pool.acquire() as conn:
        for name in speakers:
            row = await conn.fetchrow(
                "SELECT outcome FROM session_turn_receipts WHERE session_id = $1",
                _seat_session(room_id, name),
            )
            assert row is not None and row["outcome"] == "committed"

    # SSE per step: a rising sequence on the room's channel, in Centrifugo's own history.
    history = await stack.channel_history(f"ucx:room:{room_id}", limit=256)
    seqs = sorted(int(e.data["seq"]) for e in history)
    assert seqs == list(range(1, len(seqs) + 1))
    payloads = await _events(stack, room_id)
    steps = [
        (p["agent_id"], p["status"]) for p in payloads if p.get("status") in {"generating", "final"}
    ]
    assert steps == [(name, status) for name in speakers for status in ("generating", "final")]
    assert any(p.get("delta") == "Reply two." for p in payloads)

    workers.assert_stopped_cleanly(first)
    workers.assert_stopped_cleanly(second)


async def test_a_worker_killed_mid_seat_is_redelivered_with_nothing_counted_twice(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    room_id = _uid("room-")
    speakers = [_uid("s") for _ in range(3)]
    await _seed(stack, _room(room_id, speakers))
    never_released = threading.Event()
    provider.script("survive-it", "Reply one.")
    stalled = provider.script("Reply one.", "The dead worker's reply.", hold=never_released)
    provider.script("Reply two.", "Reply three.")

    a, b = workers.start(), workers.start()
    await stack.ingress.submit_room_message(_post(room_id, f"{_mentions(speakers)} survive-it"))
    await _entered(stalled)

    held = await _state(stack, room_id)
    assert held.file_record.turns_started == 2
    assert held.file_record.turns_landed == 1
    holder = await _lease_holder(stack, _seat_session(room_id, speakers[1]))
    victim, survivor = (a, b) if workers.id_of(a) == holder else (b, a)
    assert workers.id_of(victim) == holder

    provider.script("Reply one.", "Reply two.")  # what the second worker hears
    provider.script("Reply two.", "Reply three.")
    workers.kill(victim)

    ended = await _ended(stack, room_id)
    assert ended["reason"] == "silence"
    state = await _state(stack, room_id)
    assert _spoken(state)[1:] == [
        (speakers[0], "Reply one."),
        (speakers[1], "Reply two."),
        (speakers[2], "Reply three."),
    ]
    assert [m.sender_id for m in state.transcript].count(HUMAN) == 1
    assert state.file_record.turns_started == state.file_record.turns_landed == 3
    assert state.floor is None
    # The killed call never completed, so only the second worker's is in the ledger.
    ledger = await stack.usage_store.list_ledger_entries(_seat_session(room_id, speakers[1]))
    assert len(ledger) == 1
    workers.assert_stopped_cleanly(survivor)


class _NoSeatPublish(KafkaCommandBroker):
    """A broker whose `ucx.cmd.turn` went away: what a commit followed by a failed publish meets."""

    async def publish_turn(self, cmd: TurnCommandEnvelope) -> int:
        raise ConnectionError("the broker went away")


class _NoLandingPublish(KafkaCommandBroker):
    async def publish_room_step(self, cmd: RoomCommandEnvelope) -> int:
        raise ConnectionError("the broker went away")


def _clones(stack: ClusterRuntimeStack, provider: ScriptedGemini) -> AgentTurnExecutor:
    def llm(gate: UsageGate) -> GeminiConnector:
        connector = GeminiConnector(
            api_key="scripted", base_url=provider.base_url, model="scripted-model"
        )
        gate_if_paid(connector, gate)
        return connector

    return AgentTurnExecutor(
        sessions=stack.session_store,
        usage=stack.usage_store,
        sse=stack.sse_publisher,
        llm_factory=llm,
        limits=UsageLimits,
        rooms=stack.room_store,
        room_sse=stack.room_sse,
    )


def _steps(
    stack: ClusterRuntimeStack, clones: AgentTurnExecutor, broker: KafkaCommandBroker
) -> RoomStepExecutor:
    return RoomStepExecutor(
        rooms=stack.room_store,
        sessions=stack.session_store,
        broker=broker,
        sse=stack.room_sse,
        leases=stack.lease_manager,
        decide=clones.decide_for_room,
        clone_learns=clones.clone_learns,
    )


async def test_a_room_commit_whose_publish_failed_does_not_stall_the_room(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    """The step commits, the seat's publish fails, the process is gone: a real worker sends it."""
    room_id = _uid("room-")
    speaker = _uid("s")
    await _seed(stack, _room(room_id, [speaker]))
    post = _post(room_id, f"@{speaker} go-on")
    provider.script("go-on", "Reply after the redelivery.")
    await stack.ingress.submit_room_message(post)

    broken = _NoSeatPublish(bootstrap_servers=cluster_config.kafka_bootstrap_servers)
    with pytest.raises(ConnectionError):
        await _steps(stack, _clones(stack, provider), broken).run(post)
    await broken.stop()
    committed = await _state(stack, room_id)
    assert committed.file_record.turns_started == 1
    assert committed.floor is not None
    assert len(provider.requests_for("go-on")) == 0

    worker = workers.start()
    assert (await _ended(stack, room_id))["reason"] == "silence"
    state = await _state(stack, room_id)
    assert _spoken(state)[1:] == [(speaker, "Reply after the redelivery.")]
    assert [m.sender_id for m in state.transcript].count(HUMAN) == 1
    assert state.file_record.turns_started == state.file_record.turns_landed == 1
    workers.assert_stopped_cleanly(worker)


async def test_a_seat_commit_whose_landing_was_not_published_does_not_stall_the_room(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    room_id = _uid("room-")
    speaker = _uid("s")
    await _seed(stack, _room(room_id, [speaker]))
    post = _post(room_id, f"@{speaker} land-it")
    turn = provider.script("land-it", "The one reply.")
    await stack.ingress.submit_room_message(post)

    clones = _clones(stack, provider)
    # The room's step runs here, so the seat's command goes on Kafka for whoever is next.
    await _steps(stack, clones, stack.broker).run(post)
    seat = TurnCommandEnvelope(
        turn_id=_seat_turn_of(await _state(stack, room_id)),
        session_id=_seat_session(room_id, speaker),
        agent_id=speaker,
        user_id=post.user_id,
        prompt="",
        seat=SeatTurnRef(room_id=room_id, cascade_id=post.cascade_id),
        enqueued_at_ms=int(time.time() * 1000),
    )
    broken = _NoLandingPublish(bootstrap_servers=cluster_config.kafka_bootstrap_servers)
    dying = StatelessAgentWorker(
        worker_id=_uid("dying-"),
        session_store=stack.session_store,
        lease_manager=stack.lease_manager,
        sse_publisher=stack.sse_publisher,
        turn_executor=clones,
        room_steps=_steps(stack, clones, broken),
        broker=broken,
        retry_delay_seconds=0.0,
    )

    async def die(seconds: float) -> None:
        raise SystemError("the process is gone")

    with pytest.raises(SystemError):
        await dying.execute_single_turn(seat, wait=die)
    await broken.stop()
    assert turn.calls == 1
    told = await stack.session_store.settled_terminal(seat.session_id, seat.turn_id)
    assert told is not None  # committed
    assert (await _state(stack, room_id)).floor is not None  # and the room still waits

    worker = workers.start()
    assert (await _ended(stack, room_id))["reason"] == "silence"
    state = await _state(stack, room_id)
    assert _spoken(state)[1:] == [(speaker, "The one reply.")]
    assert state.file_record.turns_started == state.file_record.turns_landed == 1
    assert turn.calls == 1  # the model was not asked again
    workers.assert_stopped_cleanly(worker)


def _seat_turn_of(state: RoomState) -> str:
    assert state.floor is not None
    return state.floor.turn_id


async def test_the_circuit_breaker_stops_an_autonomous_room_at_its_limit(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    room_id = _uid("room-")
    speakers = [_uid("s"), _uid("s")]
    await _seed(
        stack,
        _room(
            room_id,
            speakers,
            autonomous=True,
            auto_routing=True,
            transcript_window=40,
        ),
    )
    provider.script("[", "A reply.")
    choices = [
        json.dumps({"speaker_id": speakers[n % 2], "confidence": 1.0, "reasoning": "go"})
        for n in range(30)
    ]
    selector = provider.script("AGENT TURNS SINCE THE LAST HUMAN MESSAGE", *choices)

    worker = workers.start()
    await stack.ingress.submit_room_message(_post(room_id, "Discuss among yourselves."))
    ended = await _ended(stack, room_id)

    assert ended["reason"] == "turn_ceiling"
    state = await _state(stack, room_id)
    assert state.turn_state.agent_turns_since_human == 20
    assert state.file_record.turns_started == state.file_record.turns_landed == 20
    assert state.floor is None
    assert state.last_decision is not None
    assert "circuit breaker of 20" in state.last_decision.reasoning
    assert selector.calls == 20  # the ceiling is checked before anyone is consulted
    workers.assert_stopped_cleanly(worker)


async def test_a_seat_that_exhausts_its_bound_ends_the_cascade_and_tells_the_person(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    """A corrupt session fails on every worker: the deadline ends it, and the floor comes back."""
    room_id = _uid("room-")
    bad, other = _uid("s"), _uid("s")
    await _seed(stack, _room(room_id, [bad, other]))
    async with stack.sql_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO session_snapshots (session_id, agent_id, revision, state_json, updated_at)"
            " VALUES ($1, 'default', 1, '{not json at all', 'now')",
            _seat_session(room_id, bad),
        )
    never = provider.script("fail-here", "Never asked.")
    worker = workers.start("--turn-deadline-seconds", "2")

    await stack.ingress.submit_room_message(_post(room_id, f"@{bad} @{other} fail-here"))
    ended = await _ended(stack, room_id)

    assert ended["reason"] == "seat_gave_up"
    assert ended["message"] == (
        f"{bad} could not answer, so the room has stopped. Send a message to try again."
    )
    state = await _state(stack, room_id)
    assert [m.sender_id for m in state.transcript] == [HUMAN, bad]
    row = state.transcript[-1]
    assert row.error == "This message could not be answered. Please try again later."
    assert state.floor is None
    assert state.file_record.turns_started == state.file_record.turns_landed == 1
    assert never.calls == 0
    told = " ".join(str(p) for p in await _events(stack, room_id))
    for internal in ("pydantic", "ValidationError", "state_json", "Traceback", "not json"):
        assert internal not in told
    for internal in INTERNALS:
        assert internal not in str(row.error)

    # The floor came back: the same room answers the next message, through the next seat.
    provider.script("next-message", "Back again.")
    await stack.ingress.submit_room_message(_post(room_id, f"@{other} next-message"))

    async def answered() -> bool:
        return _spoken(await _state(stack, room_id))[-1] == (other, "Back again.")

    await eventually(answered, timeout=_WAIT_SECONDS)
    workers.assert_stopped_cleanly(worker)


async def _hold_seat_session(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, session_id: str
) -> tuple[ScriptedTurn, threading.Event]:
    """A turn of the seat's own session running, so the seat's command waits behind it."""
    release = threading.Event()
    held = provider.script("hold-the-session", "Released.", hold=release)
    await stack.ingress.submit_turn(
        TurnCommandEnvelope(
            turn_id=_uid("turn-"),
            session_id=session_id,
            agent_id="default",
            user_id="room-user",
            prompt="hold-the-session",
            enqueued_at_ms=int(time.time() * 1000),
        )
    )
    return held, release


async def test_interrupt_while_a_seat_is_queued_runs_nobody(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    room_id = _uid("room-")
    first, second = _uid("s"), _uid("s")
    await _seed(stack, _room(room_id, [first, second]))
    never = provider.script("stop-queued", "Never asked.")
    worker = workers.start()
    held, release = await _hold_seat_session(stack, provider, _seat_session(room_id, first))
    await _entered(held)

    post = _post(room_id, f"@{first} @{second} stop-queued")
    await stack.ingress.submit_room_message(post)

    async def queued() -> bool:
        floor = (await _state(stack, room_id)).floor if await _exists(stack, room_id) else None
        return floor is not None

    await eventually(queued, timeout=_WAIT_SECONDS)
    await stack.ingress.interrupt_room(
        room_id=room_id,
        user_id=post.user_id,
        cascade_id=post.cascade_id,
        requested_by="test",
        command_id=_uid("stop-"),
    )

    async def stopped() -> bool:
        state = await _state(stack, room_id)
        return state.floor is not None and state.floor.interrupted

    await eventually(stopped, timeout=_WAIT_SECONDS)
    release.set()

    ended = await _ended(stack, room_id)
    assert ended["reason"] == "interrupted"
    state = await _state(stack, room_id)
    assert [m.sender_id for m in state.transcript] == [HUMAN, first]
    row = state.transcript[-1]
    assert (row.completed, row.content, row.error) == (False, "", "Turn was interrupted")
    assert state.floor is None
    assert state.file_record.turns_started == state.file_record.turns_landed == 1
    assert never.calls == 0
    workers.assert_stopped_cleanly(worker)


async def _message_written(stack: ClusterRuntimeStack, room_id: str, rows: int) -> None:
    async def written() -> bool:
        return len((await _state(stack, room_id)).transcript) == rows

    await eventually(written, timeout=_WAIT_SECONDS)


@pytest.mark.parametrize(
    ("seats", "names_the_seat"), [(2, True), (1, False)], ids=["names-the-seat", "one-seat-room"]
)
async def test_a_message_sent_while_a_seat_runs_is_answered_by_that_seat(
    stack: ClusterRuntimeStack,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
    seats: int,
    names_the_seat: bool,
) -> None:
    room_id = _uid("room-")
    speakers = [_uid("s") for _ in range(seats)]
    seat = speakers[0]
    mention = f"@{seat} " if names_the_seat else ""
    await _seed(stack, _room(room_id, speakers))
    release = threading.Event()
    first = provider.script("first-ask", "Reply to first.", hold=release)
    second = provider.script("second-ask", "Reply to second.")
    worker = workers.start()

    await stack.ingress.submit_room_message(_post(room_id, f"{mention}first-ask"))
    await _entered(first)
    await stack.ingress.submit_room_message(_post(room_id, f"{mention}second-ask"))
    await _message_written(stack, room_id, 2)
    release.set()

    assert (await _ended(stack, room_id))["reason"] == "silence"
    state = await _state(stack, room_id)
    assert _spoken(state)[2:] == [(seat, "Reply to first."), (seat, "Reply to second.")]
    assert (first.calls, second.calls) == (1, 1)
    assert state.floor is None
    assert state.file_record.turns_started == state.file_record.turns_landed == 2
    workers.assert_stopped_cleanly(worker)


@pytest.mark.parametrize(
    ("seats", "names_the_seat"), [(2, True), (1, False)], ids=["names-the-seat", "one-seat-room"]
)
async def test_a_message_that_takes_the_floor_back_is_answered_by_the_seat_that_lost_it(
    stack: ClusterRuntimeStack,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
    seats: int,
    names_the_seat: bool,
) -> None:
    room_id = _uid("room-")
    speakers = [_uid("s") for _ in range(seats)]
    seat = speakers[0]
    mention = f"@{seat} " if names_the_seat else ""
    await _seed(stack, _room(room_id, speakers))
    release = threading.Event()
    first = provider.script("first-ask", "Never lands.", hold=release)
    second = provider.script("second-ask", "Reply to second.")
    worker = workers.start()

    await stack.ingress.submit_room_message(_post(room_id, f"{mention}first-ask"))
    await _entered(first)
    held = await _state(stack, room_id)
    assert held.floor is not None
    await stack.room_store.commit_step(
        held.model_copy(update={"floor": held.floor.model_copy(update={"given_at_ms": 0})}),
        room_id=room_id,
        command_id=_uid("age-"),
        outbox=RoomStepOutbox(),
    )
    await stack.ingress.submit_room_message(_post(room_id, f"{mention}second-ask"))

    assert (await _ended(stack, room_id))["reason"] == "silence"
    release.set()
    state = await _state(stack, room_id)
    assert [m.sender_id for m in state.transcript] == [HUMAN, HUMAN, seat, seat]
    assert state.transcript[2].error == "This reply never arrived."
    assert state.transcript[3].content == "Reply to second."
    assert second.calls == 1
    assert state.floor is None
    workers.assert_stopped_cleanly(worker)


async def test_a_stop_after_a_waiting_message_starts_nobody(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    room_id = _uid("room-")
    first, other = _uid("s"), _uid("s")
    await _seed(stack, _room(room_id, [first, other]))
    release = threading.Event()
    running = provider.script("first-ask", "Never heard.", hold=release)
    never = provider.script("second-msg", "Never asked.")
    worker = workers.start()

    post = _post(room_id, f"@{first} first-ask")
    await stack.ingress.submit_room_message(post)
    await _entered(running)
    await stack.ingress.submit_room_message(_post(room_id, f"@{other} second-msg"))
    await _message_written(stack, room_id, 2)
    await stack.ingress.interrupt_room(
        room_id=room_id,
        user_id=post.user_id,
        cascade_id=post.cascade_id,
        requested_by="test",
        command_id=_uid("stop-"),
    )

    assert (await _ended(stack, room_id))["reason"] == "interrupted"
    release.set()
    state = await _state(stack, room_id)
    assert [m.sender_id for m in state.transcript] == [HUMAN, HUMAN, first]
    assert never.calls == 0
    assert state.file_record.turns_started == state.file_record.turns_landed == 1
    assert state.floor is None
    workers.assert_stopped_cleanly(worker)


async def test_a_client_cannot_take_the_id_of_a_landing_or_of_the_next_seat(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    """Both ids are computed as a client could, and sent over the real routes while a seat runs."""
    room_id = _uid("room-")
    first, next_seat = _uid("s"), _uid("s")
    await _seed(stack, _room(room_id, [first, next_seat]))
    release = threading.Event()
    running = provider.script("first-ask", "Reply to first.", hold=release)
    following = provider.script("Reply to first", "Reply of the next seat.")
    worker = workers.start()
    post = _post(room_id, f"@{first} @{next_seat} first-ask")
    await stack.ingress.submit_room_message(post)
    await _entered(running)
    floor = (await _state(stack, room_id)).floor
    assert floor is not None
    landing_id = f"landed-{floor.turn_id}"
    next_seat_id = _seat_turn_id(room_id, landing_id)

    transport = httpx.ASGITransport(app=create_cluster_http_app(stack))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        as_post = await client.post(
            "/api/cluster/rooms",
            json=_post(room_id, "forged", post_id=landing_id).model_dump(mode="json"),
        )
        as_turn = await client.post(
            "/api/cluster/turns",
            json=TurnCommandEnvelope(
                turn_id=next_seat_id,
                session_id=_seat_session(room_id, next_seat),
                agent_id="default",
                user_id="room-user",
                prompt="forged",
                enqueued_at_ms=int(time.time() * 1000),
            ).model_dump(mode="json"),
        )
        # A third route takes a client's id, and runs its turn in the process that serves it.
        as_execute = await client.post(
            "/api/cluster/execute",
            json={
                "prompt": "forged",
                "session_id": _seat_session(room_id, next_seat),
                "turn_id": next_seat_id,
            },
        )
    assert (as_post.status_code, as_turn.status_code, as_execute.status_code) == (422, 422, 422)

    release.set()
    await _ended(stack, room_id)
    state = await _state(stack, room_id)
    assert _spoken(state)[1:] == [
        (first, "Reply to first."),
        (next_seat, "Reply of the next seat."),
    ]
    assert following.calls == 1
    assert state.floor is None
    workers.assert_stopped_cleanly(worker)


async def test_a_limit_reached_at_the_selector_tells_the_person_it_was_the_limit(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    user = _uid("user")
    room_id = _uid("room-")
    await _seed(stack, _room(room_id, [_uid("s"), _uid("s")], auto_routing=True))
    provider.script("cross-the-limit", "Crossed.")
    worker = workers.start(UCLONE_USAGE_LIMIT_5_HOURS="100")
    crossing = _uid("sess-")
    await stack.ingress.submit_turn(
        TurnCommandEnvelope(
            turn_id=_uid("turn-"),
            session_id=crossing,
            agent_id="default",
            user_id=user,
            prompt="cross-the-limit",
            enqueued_at_ms=int(time.time() * 1000),
        )
    )

    async def booked() -> bool:
        return bool(await stack.usage_store.list_ledger_entries(crossing))

    await eventually(booked, timeout=_WAIT_SECONDS)
    calls_before = len(provider.requests)

    post = _post(room_id, "Anyone?").model_copy(update={"user_id": user})
    await stack.ingress.submit_room_message(post)

    ended = await _ended(stack, room_id)
    assert ended["reason"] == "usage_limit"
    message = str(ended["message"])
    assert message.startswith("You've reached your 5-hour limit for paid models.")
    for internal in (*INTERNALS, "Selector", "provider", "UsageLimit"):
        assert internal not in message
    assert len(provider.requests) == calls_before
    state = await _state(stack, room_id)
    assert _spoken(state) == [(HUMAN, "Anyone?")]
    assert state.floor is None
    workers.assert_stopped_cleanly(worker)


async def _exists(stack: ClusterRuntimeStack, room_id: str) -> bool:
    return await stack.room_store.read_room(room_id) is not None


async def test_interrupt_while_a_seat_is_running_cancels_it_and_nobody_follows(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    room_id = _uid("room-")
    first, second = _uid("s"), _uid("s")
    await _seed(stack, _room(room_id, [first, second]))
    release = threading.Event()
    running = provider.script("stop-running", "Never heard.", hold=release)
    worker = workers.start()

    post = _post(room_id, f"@{first} @{second} stop-running")
    await stack.ingress.submit_room_message(post)
    await _entered(running)
    await stack.ingress.interrupt_room(
        room_id=room_id,
        user_id=post.user_id,
        cascade_id=post.cascade_id,
        requested_by="test",
        command_id=_uid("stop-"),
    )

    ended = await _ended(stack, room_id)
    assert ended["reason"] == "interrupted"
    release.set()
    state = await _state(stack, room_id)
    assert [m.sender_id for m in state.transcript] == [HUMAN, first]
    row = state.transcript[-1]
    assert (row.completed, row.content) == (False, "")
    assert state.floor is None
    assert state.file_record.turns_started == state.file_record.turns_landed == 1
    assert running.calls == 1
    # Nothing of the turn stayed in the seat's session.
    assert await _no_session(stack, _seat_session(room_id, first))
    workers.assert_stopped_cleanly(worker)


async def _no_session(stack: ClusterRuntimeStack, session_id: str) -> bool:
    async with stack.sql_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM session_snapshots WHERE session_id = $1", session_id
        )
    return row is None


async def _learn_job_receipt(stack: ClusterRuntimeStack, room_id: str) -> list[str]:
    async with stack.sql_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT command_id FROM room_step_receipts "
            "WHERE room_id = $1 AND command_id LIKE 'learn-%'",
            room_id,
        )
    return [str(r["command_id"]) for r in rows]


async def _learn_records(config: ClusterRuntimeConfig, room_id: str) -> list[Mapping[str, Any]]:
    """Every job of `room_id` on `ucx.bg.learn`, read with a group of its own."""
    reader = KafkaCommandBroker(
        bootstrap_servers=config.kafka_bootstrap_servers, group_id=_uid("learn-reader-")
    )
    await reader.subscribe([TOPIC_BG_LEARN])
    found: list[Mapping[str, Any]] = []
    try:
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            for record in await reader.poll(timeout_ms=500):
                if record.key == room_id:
                    found.append(record.payload)
    finally:
        await reader.stop()
    return found


async def test_a_learn_job_runs_after_the_cascade_and_a_failing_one_leaves_the_room_intact(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    """On a real worker there is nowhere yet to keep a fact (`learn_worker.py`): no model call."""
    room_id = _uid("room-")
    speaker = _uid("s")
    await _seed(stack, _room(room_id, [speaker]))
    turn = provider.script("learn-this", "Noted.")
    worker = workers.start()

    await stack.ingress.submit_room_message(_post(room_id, f"@{speaker} learn-this"))
    await _ended(stack, room_id)

    jobs = await _learn_records(cluster_config, room_id)
    assert [(j["from_seq"], j["up_to_seq"], j["user_id"]) for j in jobs] == [(0, 2, "room-user")]

    async def consumed() -> list[str]:
        return await _learn_job_receipt(stack, room_id)

    assert await eventually(consumed, timeout=_WAIT_SECONDS) == [jobs[0]["job_id"]]
    assert turn.calls == 1  # the seat's: the job asked no model, having nowhere to keep a fact
    state = await _state(stack, room_id)
    assert state.transcript[-1].knowledge_learned == ()
    assert state.transcript[-1].knowledge_extract_error is None

    # A job that cannot be read, and a job for a room that does not exist: a letter and a no-op.
    producer = await stack.broker._producers.get()  # pyright: ignore[reportPrivateUsage]
    await producer.send_and_wait(TOPIC_BG_LEARN, b"{not json", key=room_id.encode())
    await stack.broker._send(  # pyright: ignore[reportPrivateUsage]
        TOPIC_BG_LEARN,
        room_id,
        {
            "job_id": "learn-ghost",
            "room_id": _uid("no-such-room-"),
            "user_id": "room-user",
            "from_seq": 0,
            "up_to_seq": 1,
        },
    )
    reader = KafkaCommandBroker(
        bootstrap_servers=cluster_config.kafka_bootstrap_servers, group_id=_uid("dlq-reader-")
    )
    await reader.subscribe([TOPIC_DEAD_LETTER])
    try:

        async def letter() -> Mapping[str, Any] | None:
            for record in await reader.poll(timeout_ms=500):
                if record.key == room_id and record.payload["source_topic"] == TOPIC_BG_LEARN:
                    return record.payload
            return None

        malformed = await eventually(letter, timeout=_WAIT_SECONDS)
    finally:
        await reader.stop()
    assert malformed["reason"] == "malformed"

    # The room goes on, and its record did not move for either.
    before = await _state(stack, room_id)
    assert before.floor is None
    provider.script("after-the-bad-jobs", "Still here.")
    await stack.ingress.submit_room_message(_post(room_id, f"@{speaker} after-the-bad-jobs"))

    async def answered() -> bool:
        return _spoken(await _state(stack, room_id))[-1] == (speaker, "Still here.")

    await eventually(answered, timeout=_WAIT_SECONDS)
    assert worker.poll() is None
    workers.assert_stopped_cleanly(worker)


async def test_the_learn_switch_off_produces_no_job(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clone whose file says it does not learn, read by a real worker from its agents root."""
    agents = tmp_path / "agents"
    monkeypatch.setenv("UCLONE_AGENTS_DIR", str(agents))
    PersonaRegistry(include_defaults=False).save_persona(
        PersonaDraft(
            name="quietclone",
            role="Quiet",
            system_prompt="Say little.",
            learn_from_conversations=False,
        ),
        create=True,
    )
    room_id = _uid("room-")
    await _seed(stack, _room(room_id, ["quietclone"]))
    provider.script("no-lessons", "Heard.")
    worker = workers.start(UCLONE_AGENTS_DIR=str(agents))

    await stack.ingress.submit_room_message(_post(room_id, "@quietclone no-lessons"))
    await _ended(stack, room_id)

    assert _spoken(await _state(stack, room_id))[-1] == ("quietclone", "Heard.")
    assert await _learn_records(cluster_config, room_id) == []
    assert await _learn_job_receipt(stack, room_id) == []
    workers.assert_stopped_cleanly(worker)


async def test_a_step_commit_is_one_transaction_with_its_receipt(
    stack: ClusterRuntimeStack,
) -> None:
    """The real statement: a stale step writes no receipt, a repeated one moves no room."""
    room_id = _uid("room-")
    await _seed(stack, _room(room_id, ["s1"]))
    held = await _state(stack, room_id)
    outbox = RoomStepOutbox(events=[{"type": "agent_reply", "sender_id": "x", "payload": {"n": 1}}])

    moved = await stack.room_store.commit_step(
        held.model_copy(update={"title": "One"}),
        room_id=room_id,
        command_id="c1",
        outbox=outbox,
    )
    assert moved is not None and moved.revision == held.revision + 1
    assert await stack.room_store.step_receipt(room_id, "c1") == outbox

    # Built on the revision just replaced: refused, and no receipt says it happened.
    with pytest.raises(StaleRoomWriteError):
        await stack.room_store.commit_step(
            held.model_copy(update={"title": "Stale"}),
            room_id=room_id,
            command_id="c2",
            outbox=outbox,
        )
    assert await stack.room_store.step_receipt(room_id, "c2") is None

    # The same command on the current revision: refused, and the room did not move.
    with pytest.raises(StaleRoomWriteError):
        await stack.room_store.commit_step(
            moved.model_copy(update={"title": "Twice"}),
            room_id=room_id,
            command_id="c1",
            outbox=outbox,
        )
    current = await _state(stack, room_id)
    assert (current.title, current.revision) == ("One", moved.revision)


async def test_a_worker_reads_the_floor_from_the_source_of_truth_not_a_stale_head(
    stack: ClusterRuntimeStack,
) -> None:
    """A worker that dies between its commit and the head leaves the head behind."""
    room_id = _uid("room-")
    await _seed(stack, _room(room_id, ["s1"]))
    behind = await _state(stack, room_id)
    client = await raw_valkey(stack)
    stale_head = encode_versioned_head(behind.revision, behind.model_dump_json())

    ahead = await stack.room_store.commit_step(
        behind.model_copy(update={"title": "Committed"}),
        room_id=room_id,
        command_id="c1",
        outbox=RoomStepOutbox(),
    )
    assert ahead is not None
    await client.set(f"room:head:{{{room_id}}}".encode(), stale_head)

    read = await stack.room_store.read_room(room_id)
    assert read is not None and read.title == "Committed"
