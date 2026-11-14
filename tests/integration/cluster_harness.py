"""Offline rig for the distributed runtime: real worker, real agent, scripted provider.

What is real here: `ClusterWorkerProcess`, `KeyedPartitionDispatcher`, `StatelessAgentWorker`,
`AgentTurnExecutor`, `RoomStepExecutor`, `RoomLearnWorker`, `build_clone`, `BaseAgent`, `UsageGate`,
`KafkaSSEBridge`, the agent's own session handling, and the revision checks of `SessionStore` and
`RoomStore` underneath `MemorySessions` and `MemoryRooms`.

What is faked, at the protocol boundary only: the model (`ScriptedProvider`, a connector), Kafka
(`MemoryTopic`), Valkey's lease and sequence, and PostgreSQL's ledger. Nothing here parses SQL or
emulates a script; the adapters over the real services are covered by `tests/integration/
cluster_live/`, which is also what covers cache loss and the SQL revision check itself.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from uclone_x.adapters.cluster.kafka_broker import (
    TOPIC_BG_LEARN,
    TOPIC_CMD_ROOM,
    TOPIC_CMD_TURN,
    TOPIC_CTRL_INTERRUPT,
    ConsumedRecord,
)
from uclone_x.adapters.cluster.sse_bridge import KafkaSSEBridge
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.session import SessionStore
from uclone_x.core.cluster_protocols import (
    DeadLetterEnvelope,
    InterruptControlEnvelope,
    LearnCommandEnvelope,
    RoomCommandEnvelope,
    RoomStepOutbox,
    SSEStreamEnvelope,
    TokenBudgetSnapshot,
    TurnCommandEnvelope,
    TurnLeaseConflictError,
    TurnLeaseToken,
    TurnPreemptedError,
)
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.core.session_state import SessionState
from uclone_x.errors import StaleRoomWriteError, StaleSessionWriteError
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.llm.usage.gate import UsageGate, gate_if_paid
from uclone_x.llm.usage.limits import UsageLimits
from uclone_x.llm.usage.store import UsageEntry
from uclone_x.room.models import RoomState
from uclone_x.room.store import RoomStore
from uclone_x.shells.cluster.ingress_routes import ClusterIngressService
from uclone_x.shells.cluster.learn_worker import FactStoreFor, RoomLearnWorker
from uclone_x.shells.cluster.room_steps import RoomStepExecutor
from uclone_x.shells.cluster.turn_executor import AgentTurnExecutor
from uclone_x.shells.cluster.worker_process import ClusterWorkerProcess
from uclone_x.shells.cluster.worker_runner import StatelessAgentWorker
from uclone_x.skills.auditor import SkillRegistry

SCRIPTED_INPUT_TOKENS = 137
SCRIPTED_OUTPUT_TOKENS = 41
SHUTDOWN_SECONDS = 10.0


@dataclass(frozen=True)
class Reply:
    """One scripted model answer, with the counts a provider would report."""

    text: str = ""
    tool_calls: tuple[ToolCallRequest, ...] = ()
    input_tokens: int = SCRIPTED_INPUT_TOKENS
    output_tokens: int = SCRIPTED_OUTPUT_TOKENS
    #: Raised instead of answering, as a provider failing mid-turn does.
    raises: Exception | None = None


@dataclass
class Script:
    """What the model says and who asked, shared by the connector each turn builds."""

    replies: deque[Reply] = field(default_factory=deque[Reply])
    requests: list[LLMRequest] = field(default_factory=list[LLMRequest])
    #: Prompt substring -> released by the test; a call whose last user message holds the
    #: substring reports itself in `entered` and waits.
    holds: dict[str, asyncio.Event] = field(default_factory=dict[str, asyncio.Event])
    entered: dict[str, asyncio.Event] = field(default_factory=dict[str, asyncio.Event])

    def hold(self, prompt_part: str) -> tuple[asyncio.Event, asyncio.Event]:
        """Make calls answering `prompt_part` wait: returns `(entered, release)`."""
        self.entered[prompt_part] = asyncio.Event()
        self.holds[prompt_part] = asyncio.Event()
        return self.entered[prompt_part], self.holds[prompt_part]

    @property
    def calls(self) -> int:
        return len(self.requests)

    def prompts_sent(self, call: int) -> list[str]:
        """The text of every message of request number `call`, oldest first."""
        return [m.content or "" for m in self.requests[call].messages]


class ScriptedProvider(BaseLLMConnector):
    """The scripted model, at the same boundary a provider's connector sits at."""

    def __init__(self, script: Script, *, paid: bool = True) -> None:
        super().__init__(api_key="scripted", base_url="https://scripted.invalid", timeout=5.0)
        self._script = script
        self._paid = paid

    @property
    def paid(self) -> bool:
        return self._paid

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def _answer(self, request: LLMRequest) -> tuple[Reply, TokenUsage]:
        self._script.requests.append(request)
        asked = next(
            (m.content or "" for m in reversed(request.messages) if m.role is MessageRole.USER),
            "",
        )
        for part, release in list(self._script.holds.items()):
            if part in asked:
                self._script.entered[part].set()
                await release.wait()
        reply = self._script.replies.popleft() if self._script.replies else Reply(text="Done.")
        if reply.raises is not None:
            raise reply.raises
        return reply, TokenUsage(
            provider="scripted",
            model="scripted-model",
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            total_tokens=reply.input_tokens + reply.output_tokens,
            count_source=TokenCountSource.PROVIDER,
        )

    async def generate(self, request: LLMRequest) -> ModelResponse:
        reply, usage = await self._answer(request)
        ref = ServiceRef(provider="scripted", model="scripted-model")
        return ModelResponse(
            content=reply.text,
            tool_calls=reply.tool_calls,
            usage=usage,
            finish_reason=FinishReason.TOOL_CALLS if reply.tool_calls else FinishReason.STOP,
            model_name="scripted-model",
            provenance=Provenance(path=ExecutionPath.PRIMARY, requested=ref, served_by=ref),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        reply, usage = await self._answer(request)
        yield StreamChunk(
            delta_content=reply.text,
            tool_calls=reply.tool_calls,
            usage=usage,
            finish_reason=FinishReason.TOOL_CALLS if reply.tool_calls else FinishReason.STOP,
            model="scripted-model",
        )


class MemorySessions(SessionStore):
    """`ClusterSessionStoreProtocol` over the real file store, whose revision check is real."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._guard = threading.Lock()
        self._receipts: dict[tuple[str, str], str] = {}
        self._terminals: dict[tuple[str, str], Mapping[str, Any]] = {}
        self.attempts: dict[tuple[str, str], dict[str, int]] = {}
        self._budgets: dict[str, TokenBudgetSnapshot] = {}
        self.prefetched: list[str] = []
        self.evicted: list[str] = []
        self.commits: list[tuple[str, str]] = []
        self.invalidated: list[str] = []

    async def prefetch_session(self, session_id: str) -> SessionState | None:
        self.prefetched.append(session_id)
        return self.load(session_id)

    def evict_local_turn_cache(self, session_id: str) -> None:
        self.evicted.append(session_id)

    async def invalidate_head(self, session_id: str) -> None:
        self.invalidated.append(session_id)

    async def settled_terminal(self, session_id: str, turn_id: str) -> Mapping[str, Any] | None:
        return self._terminals.get((session_id, turn_id))

    async def settle_turn(self, session_id: str, turn_id: str, terminal: Mapping[str, Any]) -> None:
        with self._guard:
            self._receipts.setdefault((session_id, turn_id), str(terminal["event"]))
            self._terminals.setdefault((session_id, turn_id), json.loads(json.dumps(terminal)))
            self.attempts.pop((session_id, turn_id), None)

    async def record_turn_start(self, session_id: str, turn_id: str) -> int:
        counted = self.attempts.setdefault((session_id, turn_id), {"starts": 0})
        counted["starts"] += 1
        return counted["starts"]

    async def refund_turn_start(self, session_id: str, turn_id: str) -> None:
        counted = self.attempts.get((session_id, turn_id))
        if counted is not None:
            counted["starts"] = max(counted["starts"] - 1, 0)

    def receipt(self, session_id: str, turn_id: str) -> str | None:
        return self._receipts.get((session_id, turn_id))

    def commit_turn(
        self,
        state: SessionState,
        *,
        turn_id: str,
        terminal: Mapping[str, Any],
        budget: TokenBudgetSnapshot | None = None,
        pending_events: Sequence[Any] | None = None,
    ) -> SessionState:
        with self._guard:
            if (state.session_id, turn_id) in self._receipts:
                raise StaleSessionWriteError(
                    f"turn {turn_id} already ended",
                    session_id=state.session_id,
                    expected_revision=state.revision,
                    actual_revision=state.revision,
                    current=state,
                )
            saved = self.save(state, pending_events=pending_events)
            self._receipts[(state.session_id, turn_id)] = "committed"
            self._terminals[(state.session_id, turn_id)] = {
                **json.loads(json.dumps(terminal)),
                "revision": saved.revision,
            }
            self.attempts.pop((state.session_id, turn_id), None)
            self.commits.append((state.session_id, turn_id))
            if budget is not None:
                self._budgets[state.session_id] = budget
            return saved

    def save_budget_snapshot(
        self, session_id: str, snapshot: TokenBudgetSnapshot, *, expected_revision: int
    ) -> None:
        current = self.load(session_id)
        if current is None or current.revision != expected_revision:
            raise StaleSessionWriteError(
                f"budget of {session_id}: expected {expected_revision}",
                session_id=session_id,
                expected_revision=expected_revision,
                actual_revision=current.revision if current is not None else 0,
                current=current or SessionState(session_id=session_id, agent_id=""),
            )
        self._budgets[session_id] = snapshot

    def load_budget_snapshot(self, session_id: str) -> TokenBudgetSnapshot | None:
        return self._budgets.get(session_id)


class MemoryRooms(RoomStore):
    """`ClusterRoomStoreProtocol` over the real file store, whose revision check is real."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.receipts: dict[tuple[str, str], RoomStepOutbox] = {}
        self.commit_attempts = 0

    async def read_room(self, room_id: str) -> RoomState | None:
        return self.load(room_id)

    async def step_receipt(self, room_id: str, command_id: str) -> RoomStepOutbox | None:
        return self.receipts.get((room_id, command_id))

    async def commit_step(
        self, state: RoomState | None, *, room_id: str, command_id: str, outbox: RoomStepOutbox
    ) -> RoomState | None:
        self.commit_attempts += 1
        if (room_id, command_id) in self.receipts:
            raise StaleRoomWriteError(f"step {command_id} was already committed")
        saved = self.save(state) if state is not None else None
        self.receipts[(room_id, command_id)] = RoomStepOutbox.model_validate_json(
            outbox.model_dump_json()
        )
        return saved


class MemoryLeases:
    """`TurnLeaseProtocol` as a dict: one holder per conversation, expiry on request."""

    def __init__(self, ttl_seconds: int = 60) -> None:
        self._ttl = ttl_seconds
        self._held: dict[str, TurnLeaseToken] = {}
        self._fence: dict[str, int] = {}
        self._tombstones: set[tuple[str, str]] = set()
        self.released: list[TurnLeaseToken] = []

    def expire(self, conversation_id: str) -> None:
        """As if the holder's deadline passed: a crashed worker renews no more."""
        self._held.pop(conversation_id, None)

    def holder(self, conversation_id: str) -> TurnLeaseToken | None:
        return self._held.get(conversation_id)

    async def acquire(
        self, conversation_id: str, turn_id: str, worker_id: str, ttl_seconds: int | None = None
    ) -> TurnLeaseToken:
        if (conversation_id, turn_id) in self._tombstones:
            raise TurnPreemptedError(conversation_id, turn_id)
        held = self._held.get(conversation_id)
        if held is not None:
            raise TurnLeaseConflictError(conversation_id, turn_id, holder=held.worker_id)
        self._fence[conversation_id] = self._fence.get(conversation_id, 0) + 1
        token = TurnLeaseToken(
            conversation_id=conversation_id,
            turn_id=turn_id,
            worker_id=worker_id,
            fencing_token=self._fence[conversation_id],
            ttl_seconds=ttl_seconds or self._ttl,
        )
        self._held[conversation_id] = token
        return token

    async def renew(self, token: TurnLeaseToken) -> bool:
        return self._held.get(token.conversation_id) == token

    async def release(self, token: TurnLeaseToken) -> None:
        self.released.append(token)
        if self._held.get(token.conversation_id) == token:
            del self._held[token.conversation_id]

    async def write_cancel_tombstone(
        self, conversation_id: str, turn_id: str, ttl_seconds: int = 300
    ) -> None:
        self._tombstones.add((conversation_id, turn_id))

    async def is_cancelled(self, conversation_id: str, turn_id: str) -> bool:
        return (conversation_id, turn_id) in self._tombstones


class MemoryUsage:
    """`DistributedUsageStoreProtocol` as a list: the rows, in the order they were booked."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def record_completed_call(
        self,
        *,
        scope: str,
        session_id: str,
        turn_id: str,
        entry: UsageEntry,
        counts_against_limits: bool,
    ) -> None:
        self.rows.append(
            {
                "scope": scope,
                "session_id": session_id,
                "turn_id": turn_id,
                "entry": entry,
                "counts_against_limits": counts_against_limits,
            }
        )

    async def entries_since(self, scope: str, since: datetime) -> Sequence[UsageEntry]:
        return [
            row["entry"]
            for row in self.rows
            if row["scope"] == scope and row["counts_against_limits"] and row["entry"].at > since
        ]

    def tokens_of(self, turn_id: str) -> list[tuple[int | None, int | None]]:
        return [
            (row["entry"].input_tokens, row["entry"].output_tokens)
            for row in self.rows
            if row["turn_id"] == turn_id
        ]


class MemoryBroker:
    """Producer side of Kafka: one list per topic partition, keyed the way Kafka keys."""

    def __init__(self, partitions: int = 4) -> None:
        self._partitions = partitions
        self.log: dict[tuple[str, int], list[ConsumedRecord]] = {}
        self.sse: list[SSEStreamEnvelope] = []
        self.dead_letters: list[DeadLetterEnvelope] = []
        #: The clock a record is stamped by; a test ages a record by setting it.
        self.now_ms: Callable[[], int] = lambda: int(time.time() * 1000)

    def partition_of(self, key: str) -> int:
        return sum(key.encode()) % self._partitions

    def _append(self, topic: str, key: str, payload: Mapping[str, Any], partition: int) -> int:
        records = self.log.setdefault((topic, partition), [])
        records.append(
            ConsumedRecord(
                topic=topic,
                partition=partition,
                offset=len(records),
                key=key,
                payload=payload,
                raw_value=json.dumps(payload).encode(),
                timestamp_ms=self.now_ms(),
            )
        )
        return len(records) - 1

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def publish_turn(self, cmd: TurnCommandEnvelope) -> int:
        return self._append(
            TOPIC_CMD_TURN,
            cmd.session_id,
            cmd.model_dump(mode="json"),
            self.partition_of(cmd.session_id),
        )

    async def publish_room_step(self, cmd: RoomCommandEnvelope) -> int:
        return self._append(
            TOPIC_CMD_ROOM, cmd.room_id, cmd.model_dump(mode="json"), self.partition_of(cmd.room_id)
        )

    async def publish_learn_job(self, cmd: LearnCommandEnvelope) -> int:
        return self._append(
            TOPIC_BG_LEARN, cmd.room_id, cmd.model_dump(mode="json"), self.partition_of(cmd.room_id)
        )

    def records(self, topic: str) -> list[ConsumedRecord]:
        """Every record of `topic`, across its partitions, by partition and offset."""
        return [
            record
            for (name, _), records in sorted(self.log.items())
            if name == topic
            for record in records
        ]

    async def broadcast_interrupt(self, ctrl: InterruptControlEnvelope) -> None:
        self._append(TOPIC_CTRL_INTERRUPT, ctrl.conversation_id, ctrl.model_dump(mode="json"), 0)

    async def publish_sse(self, conversation_id: str, envelope: SSEStreamEnvelope) -> None:
        self.sse.append(envelope)

    async def publish_dead_letter(self, envelope: DeadLetterEnvelope) -> None:
        self.dead_letters.append(envelope)

    async def commit_partition_offset(self, topic: str, partition: int, offset: int) -> None:
        raise AssertionError("the producer does not commit; a MemoryGroupConsumer does")


class MemoryGroup:
    """One consumer group's committed offsets: what a new owner starts reading from."""

    def __init__(self) -> None:
        self.committed: dict[tuple[str, int], int] = {}


class MemoryGroupConsumer:
    """`CommandConsumerProtocol` over `MemoryBroker`: reads from the last committed offset."""

    def __init__(self, broker: MemoryBroker, group: MemoryGroup) -> None:
        self._broker = broker
        self._group = group
        self._position: dict[tuple[str, int], int] = {}
        self._topics: tuple[str, ...] = ()
        self.paused = False
        self.pause_count = 0

    async def subscribe(self, topics: Sequence[str], *, on_revoked: Any = None) -> None:
        self._topics = tuple(topics)

    async def poll(
        self, timeout_ms: int = 1000, max_records: int = 500
    ) -> tuple[ConsumedRecord, ...]:
        await asyncio.sleep(0.001)
        if self.paused:
            return ()
        out: list[ConsumedRecord] = []
        for key, records in self._broker.log.items():
            if key[0] in self._topics:
                start = self._position.get(key, self._group.committed.get(key, 0))
                out.extend(records[start:])
                self._position[key] = len(records)
        return tuple(out)

    async def commit_partition_offset(self, topic: str, partition: int, offset: int) -> None:
        self._group.committed[(topic, partition)] = offset

    def pause_all(self) -> None:
        self.paused = True
        self.pause_count += 1

    def resume_all(self) -> None:
        self.paused = False

    async def stop(self) -> None: ...


class MemoryControlConsumer:
    """`ControlConsumerProtocol`: every record published from now on, for every worker."""

    def __init__(self, broker: MemoryBroker) -> None:
        self._broker = broker
        self._seen = 0
        self._topic = TOPIC_CTRL_INTERRUPT

    async def subscribe_broadcast(self, topics: Sequence[str]) -> None:
        self._topic = topics[0]
        self._seen = len(self._broker.log.get((self._topic, 0), []))

    async def poll(
        self, timeout_ms: int = 1000, max_records: int = 500
    ) -> tuple[ConsumedRecord, ...]:
        await asyncio.sleep(0.001)
        records = self._broker.log.get((self._topic, 0), [])
        fresh = records[self._seen :]
        self._seen = len(records)
        return tuple(fresh)

    async def stop(self) -> None: ...


class _SequenceCache:
    """The one cache operation `KafkaSSEBridge` needs: `INCR` on `seq:sse:{id}`."""

    def __init__(self) -> None:
        self._counts: dict[bytes, int] = {}

    async def incr(self, key: bytes, amount: int = 1) -> int:
        self._counts[key] = self._counts.get(key, 0) + amount
        return self._counts[key]


@dataclass
class RunningWorker:
    """A `ClusterWorkerProcess` on a task; `kill` abandons it as a crash would."""

    process: ClusterWorkerProcess
    worker: StatelessAgentWorker
    consumer: MemoryGroupConsumer
    stop: asyncio.Event
    task: asyncio.Task[None]

    async def kill(self) -> None:
        """Gone at once: nothing drains and nothing more is committed."""
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)

    async def shut_down(self) -> None:
        """Ask it to stop, and fail rather than wait for ever if it does not."""
        self.stop.set()
        await asyncio.wait_for(asyncio.shield(self.task), timeout=SHUTDOWN_SECONDS)


class ClusterRig:
    """One stack's worth of memory services, and as many real workers as a test starts."""

    def __init__(
        self,
        root: Path,
        *,
        limits: UsageLimits | None = None,
        paid: bool = True,
        workspace: Path | None = None,
        skills: SkillRegistry | None = None,
        retry_delay_seconds: float = 0.0,
        fact_store_for: FactStoreFor | None = None,
        **bounds: float,
    ) -> None:
        self.script = Script()
        self.sessions = MemorySessions(root / "sessions")
        self.rooms = MemoryRooms(root / "rooms")
        self.leases = MemoryLeases()
        self.usage = MemoryUsage()
        self.broker = MemoryBroker()
        self.group = MemoryGroup()
        self.limits = limits if limits is not None else UsageLimits()
        self.ingress = ClusterIngressService(self.broker, self.leases)
        self.sse = KafkaSSEBridge(_SequenceCache(), self.broker)  # pyright: ignore[reportArgumentType]
        self.room_sse = KafkaSSEBridge(_SequenceCache(), self.broker, channel_kind="room")  # pyright: ignore[reportArgumentType]
        self.fact_store_for = fact_store_for
        self.learn_group = MemoryGroup()
        self._paid = paid
        self.personas = PersonaRegistry(workspace_root=root)
        self._workspace = workspace
        self._skills = skills
        self._retry_delay_seconds = retry_delay_seconds
        self._bounds = bounds
        self._running: list[RunningWorker] = []
        self._next_turn = 0

    def llm_factory(self, gate: UsageGate) -> BaseLLMConnector:
        return gate_if_paid(ScriptedProvider(self.script, paid=self._paid), gate)

    def executor(self) -> AgentTurnExecutor:
        return AgentTurnExecutor(
            sessions=self.sessions,
            usage=self.usage,
            sse=self.sse,
            llm_factory=self.llm_factory,
            workspace_root=self._workspace,
            limits=lambda: self.limits,
            persona_registry=self.personas,
            skills=self._skills,
            rooms=self.rooms,
            room_sse=self.room_sse,
        )

    def room_steps(
        self, executor: AgentTurnExecutor | None = None, **fields: Any
    ) -> RoomStepExecutor:
        clones = executor or self.executor()
        return RoomStepExecutor(
            rooms=self.rooms,
            sessions=self.sessions,
            broker=self.broker,
            sse=self.room_sse,
            leases=self.leases,
            decide=clones.decide_for_room,
            clone_learns=clones.clone_learns,
            **fields,
        )

    def learner(self) -> RoomLearnWorker:
        return RoomLearnWorker(
            rooms=self.rooms, clones=self.executor(), fact_store_for=self.fact_store_for
        )

    def worker(self, worker_id: str = "w1") -> StatelessAgentWorker:
        executor = self.executor()
        return StatelessAgentWorker(
            worker_id=worker_id,
            session_store=self.sessions,
            lease_manager=self.leases,
            sse_publisher=self.sse,
            turn_executor=executor,
            room_steps=self.room_steps(executor),
            broker=self.broker,
            retry_delay_seconds=self._retry_delay_seconds,
            **self._bounds,  # pyright: ignore[reportArgumentType]
        )

    def start_worker(self, worker_id: str = "w1", *, concurrency: int = 16) -> RunningWorker:
        worker = self.worker(worker_id)
        consumer = MemoryGroupConsumer(self.broker, self.group)
        process = ClusterWorkerProcess(
            worker=worker,
            consumer=consumer,
            control=MemoryControlConsumer(self.broker),
            control_topic=TOPIC_CTRL_INTERRUPT,
            concurrency=concurrency,
            drain_seconds=5.0,
            learner=self.learner(),
            learn_consumer=MemoryGroupConsumer(self.broker, self.learn_group),
        )
        stop = asyncio.Event()
        running = RunningWorker(
            process, worker, consumer, stop, asyncio.create_task(process.run(stop))
        )
        self._running.append(running)
        return running

    async def close(self) -> None:
        for running in self._running:
            if not running.task.done():
                await running.kill()

    def command(
        self,
        session_id: str,
        prompt: str,
        *,
        turn_id: str | None = None,
        user_id: str = "user-1",
        **fields: Any,
    ) -> TurnCommandEnvelope:
        self._next_turn += 1
        return TurnCommandEnvelope(
            turn_id=turn_id or f"turn-{self._next_turn}",
            session_id=session_id,
            agent_id="default",
            user_id=user_id,
            prompt=prompt,
            enqueued_at_ms=1000,
            **fields,
        )

    def room_stream(self, room_id: str) -> list[Mapping[str, Any]]:
        """Every SSE event of the room's channel, as published, in publish order."""
        channel = f"ucx:room:{room_id}"
        return [e.data for e in self.broker.sse if e.channel == channel]

    def committed_offset(self, session_id: str) -> int | None:
        return self.group.committed.get((TOPIC_CMD_TURN, self.broker.partition_of(session_id)))

    def stream(self, session_id: str) -> list[Mapping[str, Any]]:
        """Every SSE event of the session, as published, in publish order."""
        channel = f"ucx:session:{session_id}"
        return [e.data for e in self.broker.sse if e.channel == channel]

    def terminal_events(self, session_id: str, turn_id: str) -> list[str]:
        """The `TURN_*` events of `turn_id` that end a turn, in order."""
        names: list[str] = []
        for data in self.stream(session_id):
            payload: dict[str, Any] = data["payload"]
            event = str(payload.get("event"))
            if payload.get("turn_id") == turn_id and event.startswith("TURN_"):
                if event != "TURN_STARTED":
                    names.append(event)
        return names

    def history(self, session_id: str) -> list[tuple[str, str]]:
        """The committed session's messages as `(role, text)`, system anchor left out."""
        state = self.sessions.load(session_id)
        if state is None:
            return []
        return [
            (m.role.value, m.content or "")
            for m in state.messages
            if m.role is not MessageRole.SYSTEM
        ]


async def eventually(check: Callable[[], bool], *, timeout: float = 5.0) -> None:
    """Yield to the workers until `check()` holds; fail naming the timeout."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not check():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(0.005)
