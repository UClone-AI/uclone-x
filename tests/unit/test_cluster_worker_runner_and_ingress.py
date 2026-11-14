"""Unit tests for `StatelessAgentWorker`, `ClusterHost`, `StickyEndpointRouter`, and ingress."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from uclone_x.agent.models import TurnResult
from uclone_x.core.capability import Capability
from uclone_x.core.cluster_protocols import (
    ClusterSessionStoreProtocol,
    CommandBrokerProtocol,
    DeadLetterEnvelope,
    InterruptControlEnvelope,
    LearnCommandEnvelope,
    RoomCommandEnvelope,
    SSEStreamEnvelope,
    SSEStreamPublisherProtocol,
    TokenBudgetSnapshot,
    TurnCommandEnvelope,
    TurnLeaseProtocol,
    TurnLeaseToken,
    TurnPreemptedError,
)
from uclone_x.core.session_state import SessionState
from uclone_x.engine.event_bus import AgentEvent
from uclone_x.engine.protocols import EventBusProtocol
from uclone_x.errors import UCloneXError
from uclone_x.sandbox.models import IsolationLevel
from uclone_x.shells.cluster.host import ClusterHost, StickyEndpointRouter
from uclone_x.shells.cluster.ingress_routes import ClusterIngressService
from uclone_x.shells.cluster.turn_executor import TurnExecution
from uclone_x.shells.cluster.worker_runner import StatelessAgentWorker, TurnDisposition


def _command(turn_id: str, **fields: Any) -> TurnCommandEnvelope:
    return TurnCommandEnvelope(
        turn_id=turn_id,
        session_id="sess-101",
        agent_id="clone-a",
        user_id="user-1",
        prompt="hello cluster",
        enqueued_at_ms=1000,
        **fields,
    )


def _result(content: str) -> TurnResult:
    return TurnResult(turn_index=1, content=content, is_completed=True, provenance=None)


class _FakeClusterSessionStore(ClusterSessionStoreProtocol):
    def __init__(self) -> None:
        self.records: dict[str, SessionState] = {}
        self.budgets: dict[str, TokenBudgetSnapshot] = {}
        self.evicted: list[str] = []
        self.prefetched: list[str] = []
        self.receipts: set[tuple[str, str]] = set()

    async def prefetch_session(self, session_id: str) -> SessionState | None:
        self.prefetched.append(session_id)
        return self.records.get(session_id)

    def evict_local_turn_cache(self, session_id: str) -> None:
        self.evicted.append(session_id)

    def save_budget_snapshot(
        self, session_id: str, snapshot: TokenBudgetSnapshot, *, expected_revision: int
    ) -> None:
        self.budgets[session_id] = snapshot

    async def settled_terminal(self, session_id: str, turn_id: str) -> Mapping[str, Any] | None:
        return {"event": "TURN_COMPLETED"} if (session_id, turn_id) in self.receipts else None

    async def settle_turn(self, session_id: str, turn_id: str, terminal: Mapping[str, Any]) -> None:
        self.receipts.add((session_id, turn_id))

    async def record_turn_start(self, session_id: str, turn_id: str) -> int:
        return 1

    async def refund_turn_start(self, session_id: str, turn_id: str) -> None:
        return None

    def commit_turn(
        self,
        state: SessionState,
        *,
        turn_id: str,
        terminal: Mapping[str, Any],
        budget: TokenBudgetSnapshot | None = None,
        pending_events: object = None,
    ) -> SessionState:
        self.receipts.add((state.session_id, turn_id))
        return self.save(state)

    async def invalidate_head(self, session_id: str) -> None:
        return None

    def load_budget_snapshot(self, session_id: str) -> TokenBudgetSnapshot | None:
        return self.budgets.get(session_id)

    def load(self, session_id: str) -> SessionState | None:
        return self.records.get(session_id)

    def save(self, state: SessionState, pending_events: object = None) -> SessionState:
        stamped = state.model_copy(update={"revision": state.revision + 1})
        self.records[state.session_id] = stamped
        return stamped

    def delete(self, session_id: str) -> bool:
        return self.records.pop(session_id, None) is not None

    def list_session_ids(self) -> tuple[str, ...]:
        return tuple(self.records.keys())

    def clear_event_log(self, session_id: str) -> None:
        return None

    def save_context_body(self, session_id: str, digest: str, body: str) -> None:
        return None

    def load_context_body(self, session_id: str, digest: str) -> str | None:
        return None


class _FakeLeaseManager(TurnLeaseProtocol):
    def __init__(self) -> None:
        self.acquired: list[TurnLeaseToken] = []
        self.released: list[TurnLeaseToken] = []
        self.tombstones: set[tuple[str, str]] = set()

    async def acquire(
        self, conversation_id: str, turn_id: str, worker_id: str, ttl_seconds: int | None = None
    ) -> TurnLeaseToken:
        if await self.is_cancelled(conversation_id, turn_id):
            raise TurnPreemptedError(conversation_id, turn_id)
        token = TurnLeaseToken(
            conversation_id=conversation_id,
            turn_id=turn_id,
            worker_id=worker_id,
            fencing_token=len(self.acquired) + 1,
            ttl_seconds=ttl_seconds or 60,
        )
        self.acquired.append(token)
        return token

    async def renew(self, token: TurnLeaseToken) -> bool:
        return True

    async def release(self, token: TurnLeaseToken) -> None:
        self.released.append(token)

    async def write_cancel_tombstone(
        self, conversation_id: str, turn_id: str, ttl_seconds: int = 300
    ) -> None:
        self.tombstones.add((conversation_id, turn_id))

    async def is_cancelled(self, conversation_id: str, turn_id: str) -> bool:
        return (conversation_id, turn_id) in self.tombstones or (
            conversation_id,
            "*",
        ) in self.tombstones


class _FakeSSEPublisher(SSEStreamPublisherProtocol):
    def __init__(self) -> None:
        self.attached_conversations: list[str] = []
        self.events: list[AgentEvent] = []

    async def publish_event(self, conversation_id: str, event: AgentEvent) -> int:
        self.events.append(event)
        return len(self.events)

    @asynccontextmanager
    async def attach_to_bus(
        self, bus: EventBusProtocol, conversation_id: str
    ) -> AsyncGenerator[None]:
        self.attached_conversations.append(conversation_id)
        yield


class _FakeBroker(CommandBrokerProtocol):
    async def publish_dead_letter(self, envelope: DeadLetterEnvelope) -> None:
        self.dead_letters.append(envelope)

    def __init__(self) -> None:
        self.dead_letters: list[DeadLetterEnvelope] = []
        self.turns: list[TurnCommandEnvelope] = []
        self.room_steps: list[RoomCommandEnvelope] = []
        self.learn_jobs: list[LearnCommandEnvelope] = []
        self.interrupts: list[InterruptControlEnvelope] = []
        self.committed_offsets: list[tuple[str, int, int]] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def publish_turn(self, cmd: TurnCommandEnvelope) -> None:
        self.turns.append(cmd)

    async def publish_room_step(self, cmd: RoomCommandEnvelope) -> None:
        self.room_steps.append(cmd)

    async def publish_learn_job(self, cmd: LearnCommandEnvelope) -> None:
        self.learn_jobs.append(cmd)

    async def broadcast_interrupt(self, ctrl: InterruptControlEnvelope) -> None:
        self.interrupts.append(ctrl)

    async def publish_sse(self, conversation_id: str, envelope: SSEStreamEnvelope) -> None:
        _ = (conversation_id, envelope)

    async def commit_partition_offset(self, topic: str, partition: int, offset: int) -> None:
        self.committed_offsets.append((topic, partition, offset))


async def test_stateless_agent_worker_always_evicts_and_releases_and_says_how_a_turn_ended() -> (
    None
):
    """Every way out of a turn evicts its buffers and gives the lease back.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: self._session_store.evict_local_turn_cache(cmd.session_id)
    Becomes: pass
    """
    store = _FakeClusterSessionStore()
    leases = _FakeLeaseManager()
    sse = _FakeSSEPublisher()

    async def commit(terminal: Mapping[str, Any]) -> SessionState:
        return store.commit_turn(
            SessionState(session_id="sess-101", agent_id="clone-a"),
            turn_id="turn-101",
            terminal=terminal,
        )

    async def execute_ok(cmd: TurnCommandEnvelope, bus: EventBusProtocol) -> TurnExecution:
        return TurnExecution(result=_result("Fine."), commit=commit)

    worker = StatelessAgentWorker(
        worker_id="worker-1",
        session_store=store,
        lease_manager=leases,
        sse_publisher=sse,
        turn_executor=execute_ok,
    )
    outcome = await worker.execute_single_turn(_command("turn-101"))
    assert outcome.disposition is TurnDisposition.COMPLETED
    assert outcome.state is not None and outcome.state.revision == 1
    assert store.evicted == ["sess-101"]
    assert len(leases.released) == 1

    # A stale expected_revision is a turn that ends, not one that is run on the wrong state.
    stale = await worker.execute_single_turn(_command("turn-102", expected_revision=0))
    assert stale.disposition is TurnDisposition.FAILED
    assert store.evicted == ["sess-101", "sess-101"]
    assert len(leases.released) == 2

    # A turn that keeps failing is dead-lettered; the exception's own text stays in the log.
    async def execute_broken(cmd: TurnCommandEnvelope, bus: EventBusProtocol) -> TurnExecution:
        raise RuntimeError("secret internal detail")

    dead_letters = _FakeBroker()
    broken = StatelessAgentWorker(
        worker_id="worker-2",
        session_store=store,
        lease_manager=leases,
        sse_publisher=sse,
        turn_executor=execute_broken,
        broker=dead_letters,
        retry_delay_seconds=0.0,
    )
    failed = await broken.execute_single_turn(_command("turn-103"))
    assert failed.disposition is TurnDisposition.FAILED
    assert [(d.reason, d.attempts) for d in dead_letters.dead_letters] == [("attempts_used", 5)]
    assert store.evicted[-1] == "sess-101"
    assert len(leases.released) == 7  # five attempts of the broken turn
    assert "secret internal detail" not in str(sse.events[-1].payload)
    assert sse.events[-1].payload["event"] == TurnDisposition.FAILED.value


async def test_sticky_endpoint_router_and_ingress_interrupt_and_host_capabilities() -> None:
    """`StickyEndpointRouter` spills over on load/unhealthy; `ClusterHost` derives capabilities.

    Killed by: src/uclone_x/shells/cluster/host.py :: if self._inflight[candidate] >= self._max_inflight_per_endpoint:
    Becomes: if False:
    Killed by: src/uclone_x/shells/cluster/host.py :: if sandbox is not None and isolation_floor is not None:
    Becomes: if False:
    """
    router = StickyEndpointRouter(
        ["http://gpu-0:8000", "http://gpu-1:8000"],
        max_inflight_per_endpoint=2,
    )
    primary, headers = router.select_endpoint("sess-sticky-42")
    assert headers == {"X-Session-ID": "sess-sticky-42"}
    assert router.select_endpoint("sess-sticky-42")[0] == primary

    # Saturate primary endpoint -> spills over to neighbor.
    router.acquire_inflight(primary)
    router.acquire_inflight(primary)
    spillover, _ = router.select_endpoint("sess-sticky-42")
    assert spillover != primary

    # Mark spillover unhealthy while primary is saturated -> raises UCloneXError.
    router.mark_unhealthy(spillover)
    with pytest.raises(UCloneXError):
        router.select_endpoint("sess-sticky-42")

    # ClusterHost capability derivation.
    headless = ClusterHost(
        llm=MagicMock(),
        tools=MagicMock(),
        sessions=_FakeClusterSessionStore(),
        tracer=MagicMock(),
        bus=MagicMock(),
    )
    assert headless.capabilities == frozenset()

    full_host = ClusterHost(
        llm=MagicMock(),
        tools=MagicMock(),
        sessions=_FakeClusterSessionStore(),
        tracer=MagicMock(),
        bus=MagicMock(),
        workspace=MagicMock(),
        sandbox=MagicMock(),
        isolation_floor=IsolationLevel.CONTAINER,
        available_isolation=frozenset({IsolationLevel.CONTAINER}),
    )
    assert Capability.WORKSPACE_READ in full_host.capabilities
    assert Capability.WORKSPACE_WRITE in full_host.capabilities
    assert Capability.PROCESS_EXEC in full_host.capabilities

    # Ingress submission and dual-path interrupt.
    broker = _FakeBroker()
    leases = _FakeLeaseManager()
    ingress = ClusterIngressService(broker, leases)

    accepted = await ingress.submit_turn(
        TurnCommandEnvelope(
            turn_id="t-1",
            session_id="s-1",
            agent_id="a-1",
            user_id="u-1",
            prompt="hi",
            enqueued_at_ms=10,
        )
    )
    assert accepted.status_code == 202
    assert accepted.sse_channel == "ucx:session:s-1"

    ctrl = InterruptControlEnvelope(
        conversation_id="s-1",
        turn_id="t-1",
        requested_by="u-1",
        issued_at_ms=20,
    )
    await ingress.interrupt_turn(ctrl)
    assert await leases.is_cancelled("s-1", "t-1") is True
    assert broker.interrupts == [ctrl]

    # An interrupt naming no turn is broadcast, and leaves no tombstone behind to refuse the
    # person's next message.
    tombstones_before = len(leases.tombstones)
    anything = ctrl.model_copy(update={"turn_id": None})
    await ingress.interrupt_turn(anything)
    assert broker.interrupts == [ctrl, anything]
    assert len(leases.tombstones) == tombstones_before
