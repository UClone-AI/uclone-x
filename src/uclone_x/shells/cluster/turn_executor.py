"""One turn of a real agent on a stateless worker: compose it, run it, say how to commit it.

A room seat's turn comes through here too (`SEAT_TURN_EXECUTE`), so it has the one context builder:
the room adds its framing, its display name and the span the seat has not seen, as a room does on
one machine, and `execute_single_speaker_step` runs the turn.

The clone comes from `build_clone`, the one place a clone is assembled (owner ruling 2026-09-27:
one context builder per clone), through `ClusterHost`. A turn gets its own connector, gate,
budget ledger and agent, and all of them are dropped when it ends (Invariant: zero idle state).

The executor does not commit. It returns a `TurnExecution` whose `commit` the worker awaits
after its last tombstone check, which writes through the agent's own `persist_session`: what is
saved, and which bodies go with it, stay the kernel's decision.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from uclone_x.adapters.cluster.usage_store import TurnUsageStore
from uclone_x.agent.clone_builder import build_clone
from uclone_x.agent.models import TurnResult
from uclone_x.agent.persona_registry import PersonaRegistry, get_default_persona_registry
from uclone_x.core.agent_home import seat_id_for
from uclone_x.core.cluster_protocols import (
    ClusterRoomStoreProtocol,
    ClusterSessionStoreProtocol,
    SeatTurnRef,
    SSEStreamPublisherProtocol,
    TokenBudgetSnapshot,
    TurnCommandEnvelope,
)
from uclone_x.core.session_state import SessionState, content_digest
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.engine.protocols import EventBusProtocol
from uclone_x.errors import UCloneXError
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.factory import create_llm_connector
from uclone_x.llm.gateway import DefaultBinding, ModelGateway
from uclone_x.llm.models import TokenBudget
from uclone_x.llm.usage.gate import UsageGate, gate_connector
from uclone_x.llm.usage.limits import UsageLimits, load_limits
from uclone_x.room.models import (
    Participant,
    RoomState,
    SpeakerDecision,
    SpeakerStepOutcome,
    is_one_seat,
)
from uclone_x.room.orchestrator import (
    decide_speaker,
    execute_single_speaker_step,
    plan_speaker_step,
)
from uclone_x.room.resolver import room_participant_system_prompt
from uclone_x.room.selectors import build_selector_chain
from uclone_x.sandbox.models import AVAILABLE_ISOLATION_LEVELS, IsolationLevel
from uclone_x.shells.cluster.host import ClusterHost
from uclone_x.telemetry import TelemetryTracer
from uclone_x.tools.registry import ToolRegistry, create_default_registry

if TYPE_CHECKING:
    from uclone_x.agent.base import BaseAgent
    from uclone_x.core.cluster_protocols import DistributedUsageStoreProtocol
    from uclone_x.skills.protocols import SkillRegistryProtocol
    from uclone_x.tools.protocols import ToolRegistryProtocol

__all__ = [
    "WORKER_FILE_TOOLS",
    "AgentTurnExecutor",
    "ComposedClone",
    "LlmFactory",
    "TurnExecution",
    "environment_llm",
]

#: Every registry tool a worker offers, and only with a workspace. An allowlist, because a worker
#: serves many users on one OS account with no sandbox and nobody to answer an approval: a tool
#: added to the default registry later is not offered here until someone decides it should be.
#: These five go through `resolve_safe_path`, so they stay in the user's folder. The rest of the
#: default registry is not offered: `bash_run`, `run_command`, `install_package` and `browser`
#: run processes as the worker's OS user, and nothing else there was reviewed for this.
WORKER_FILE_TOOLS: Final[tuple[str, ...]] = (
    "file_read",
    "file_write",
    "file_edit",
    "file_search",
    "directory_list",
)

LlmFactory = Callable[[UsageGate], BaseLLMConnector]


def environment_llm(gate: UsageGate) -> BaseLLMConnector:
    """The connector the process's environment or saved settings name, passed through `gate`.

    The credential decision for a stateless worker (#2466 step 2): the provider's own variables
    (`LLM_PROVIDER`, `GEMINI_API_KEY`, ...) in the worker's environment, which is the "environment
    overrides" branch of the design's connection registry. Per-user connections in PostgreSQL
    are the other branch and are not built. Nothing named means `LLMProviderNotConfiguredError`.
    """
    return create_llm_connector(usage_gate=gate)


@dataclass(frozen=True)
class TurnExecution:
    """A finished turn, not yet committed."""

    result: TurnResult
    #: Takes the terminal event, which is stored in the receipt the commit writes.
    commit: Callable[[Mapping[str, Any]], Awaitable[SessionState]]
    #: A seat's turn: what the room's row will say of it.
    landing: SpeakerStepOutcome | None = None


@dataclass(frozen=True)
class ComposedClone:
    """A clone built for one piece of work on this worker, and what it holds until that ends."""

    agent: BaseAgent
    llm: BaseLLMConnector
    store: _TurnSessionStore
    budget: TokenBudgetManager
    turns_so_far: int


class _ReadOnlySkills:
    """A skill registry with no `store_root`, which is what `BaseAgent` offers `propose_skill` on.

    `load_skill` stays: it returns approved instructions by name and takes no path. A proposal
    would be written into the store every user of the worker shares, for a person to approve in
    a Settings page a worker does not have.
    """

    store_root = None

    def __init__(self, inner: SkillRegistryProtocol) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _TurnSessionStore(SessionStoreProtocol):
    """The session store as the agent of one turn sees it.

    Forwards everything. The one difference is the save `arm_commit` marks: that one is the
    turn's commit, carrying its receipt and budget. A save before it (a mid-turn compaction) is
    an ordinary revision of the record and says nothing about the turn having finished.
    """

    def __init__(self, inner: ClusterSessionStoreProtocol) -> None:
        self._inner = inner
        self._armed: tuple[str, TokenBudgetSnapshot, Mapping[str, Any]] | None = None

    def load(self, session_id: str) -> SessionState | None:
        return self._inner.load(session_id)

    def delete(self, session_id: str) -> bool:
        return self._inner.delete(session_id)

    def list_session_ids(self) -> tuple[str, ...]:
        return self._inner.list_session_ids()

    def clear_event_log(self, session_id: str) -> None:
        self._inner.clear_event_log(session_id)

    def save_context_body(self, session_id: str, digest: str, body: str) -> None:
        self._inner.save_context_body(session_id, digest, body)

    def load_context_body(self, session_id: str, digest: str) -> str | None:
        return self._inner.load_context_body(session_id, digest)

    def arm_commit(
        self, turn_id: str, budget: TokenBudgetSnapshot, terminal: Mapping[str, Any]
    ) -> None:
        self._armed = (turn_id, budget, terminal)

    def save(
        self, state: SessionState, pending_events: Sequence[Any] | None = None
    ) -> SessionState:
        armed, self._armed = self._armed, None
        if armed is None:
            return self._inner.save(state, pending_events=pending_events)
        turn_id, budget, terminal = armed
        return self._inner.commit_turn(
            state, turn_id=turn_id, terminal=terminal, budget=budget, pending_events=pending_events
        )


def _budget_snapshot(
    manager: TokenBudgetManager, session_id: str, turn_count: int
) -> TokenBudgetSnapshot:
    held = manager.get_budget(session_id)
    return TokenBudgetSnapshot(
        max_tokens=held.max_tokens if held is not None else None,
        consumed_prompt_tokens=held.used_input_tokens if held is not None else 0,
        consumed_completion_tokens=held.used_output_tokens if held is not None else 0,
        per_provider=dict(held.per_provider) if held is not None else {},
        turn_count=turn_count,
    )


class AgentTurnExecutor:
    """`TurnExecutorFn` running a real `BaseAgent`; holds nothing about any session."""

    def __init__(
        self,
        *,
        sessions: ClusterSessionStoreProtocol,
        usage: DistributedUsageStoreProtocol,
        sse: SSEStreamPublisherProtocol,
        llm_factory: LlmFactory = environment_llm,
        workspace_root: Path | None = None,
        limits: Callable[[], UsageLimits] = load_limits,
        persona_registry: PersonaRegistry | None = None,
        skills: SkillRegistryProtocol | None = None,
        rooms: ClusterRoomStoreProtocol | None = None,
        room_sse: SSEStreamPublisherProtocol | None = None,
    ) -> None:
        self._rooms = rooms
        self._room_sse = room_sse
        self._sessions = sessions
        self._usage = usage
        self._sse = sse
        self._llm_factory = llm_factory
        self._workspace_root = workspace_root
        self._limits = limits
        self._persona_registry = persona_registry
        self._skills = skills

    @cached_property
    def _registry(self) -> PersonaRegistry:
        # Not at construction: building a stack must not touch the session root.
        return self._persona_registry or get_default_persona_registry(self._workspace_root)

    @cached_property
    def _tools(self) -> ToolRegistryProtocol:
        """Only what is safe for many users on one OS account; see `WORKER_FILE_TOOLS`."""
        offered = ToolRegistry()
        if self._workspace_root is None:
            return offered
        every_tool = create_default_registry(workspace_root=self._workspace_root, enable_mcp=False)
        for name in WORKER_FILE_TOOLS:
            tool = every_tool.get(name)
            if tool is None:
                raise UCloneXError(f"The worker's tool {name!r} is not in the default registry")
            offered.register(tool)
        return offered

    def _user_workspace(self, user_id: str) -> Path | None:
        """This user's own folder under the workspace, created on first use.

        Named by a digest of the id, so no id can name a path, however odd it is.
        """
        if self._workspace_root is None:
            return None
        folder = self._workspace_root / "users" / content_digest(user_id)
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        return folder

    def _connector(
        self, *, user_id: str, session_id: str, turn_id: str
    ) -> tuple[BaseLLMConnector, UsageGate]:
        gate = UsageGate(
            store=TurnUsageStore(
                usage=self._usage,
                scope=user_id,
                session_id=session_id,
                turn_id=turn_id,
            ),
            limits=self._limits,
        )
        llm = self._llm_factory(gate)
        if not llm.paid:
            # Booked for the ledger, never limited: every completed call has a row.
            gate_connector(
                llm,
                UsageGate(
                    store=TurnUsageStore(
                        usage=self._usage,
                        scope=user_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        counts_against_limits=False,
                    ),
                    limits=UsageLimits,
                ),
            )
        return llm, gate

    def _restored_budget(self, session_id: str) -> tuple[TokenBudgetManager, int]:
        manager = TokenBudgetManager()
        held = self._sessions.load_budget_snapshot(session_id)
        if held is None:
            return manager, 0
        manager.set_budget(
            session_id,
            TokenBudget(
                max_tokens=held.max_tokens,
                used_input_tokens=held.consumed_prompt_tokens,
                used_output_tokens=held.consumed_completion_tokens,
                per_provider=dict(held.per_provider),
            ),
        )
        return manager, held.turn_count

    def _stream_callback(
        self, cmd: TurnCommandEnvelope
    ) -> Callable[[str, dict[str, Any]], Awaitable[None]]:
        async def publish(event_name: str, data: dict[str, Any]) -> None:
            await self._sse.publish_event(
                cmd.session_id,
                AgentEvent(
                    type=EventType.AGENT_REPLY,
                    topic=f"session.{cmd.session_id}",
                    session_id=cmd.session_id,
                    sender_id=cmd.agent_id,
                    payload={"event": event_name.upper(), "turn_id": cmd.turn_id, **data},
                ),
            )

        return publish

    def _seat_stream_callback(
        self, cmd: TurnCommandEnvelope, room_id: str
    ) -> Callable[[str, dict[str, Any]], Awaitable[None]]:
        """A seat's tokens and progress notices, on the room's channel.

        As `RoomOrchestrator._stream_callback` forwards them, and only those: a seat's thinking
        and its tool traces are its own session's (G1, G2).
        """

        async def publish(event_name: str, data: dict[str, Any]) -> None:
            detail = data.get("detail")
            delta = data.get("content")
            if event_name == "status" and data.get("status") == "calling_tool" and detail:
                update: dict[str, Any] = {"status": "status_update", "detail": str(detail)}
            elif event_name == "token" and delta:
                update = {"status": "streaming", "delta": delta}
            else:
                return
            await self._publish_room(cmd, room_id, update)

        return publish

    async def _publish_room(
        self, cmd: TurnCommandEnvelope, room_id: str, payload: Mapping[str, Any]
    ) -> None:
        if self._room_sse is None:
            raise UCloneXError("AgentTurnExecutor needs a room channel to run a seat turn")
        await self._room_sse.publish_event(
            room_id,
            AgentEvent(
                type=EventType.AGENT_REPLY,
                topic=f"room.{room_id}",
                session_id=cmd.session_id,
                sender_id=cmd.agent_id,
                payload={
                    "room_id": room_id,
                    "agent_id": cmd.agent_id,
                    "turn_id": cmd.turn_id,
                    **payload,
                },
            ),
        )

    def _compose(
        self,
        *,
        user_id: str,
        session_id: str,
        turn_id: str,
        clone_id: str,
        bus: EventBusProtocol,
        display_name: str | None = None,
        seat_framing: str = "",
    ) -> ComposedClone:
        """The one way a clone is built here: for a turn, a seat's turn or a lesson."""
        llm, gate = self._connector(user_id=user_id, session_id=session_id, turn_id=turn_id)
        store = _TurnSessionStore(self._sessions)
        budget, turns_so_far = self._restored_budget(session_id)
        has_files = self._workspace_root is not None
        host = ClusterHost(
            llm=llm,
            tools=self._tools,
            sessions=store,
            tracer=TelemetryTracer(),
            bus=bus,
            # What the file tools' calls are recorded under. No tool offered here executes.
            isolation_floor=IsolationLevel.WORKSPACE if has_files else None,
            available_isolation=AVAILABLE_ISOLATION_LEVELS if has_files else frozenset(),
        )
        gateway = ModelGateway(default_binding=DefaultBinding(llm), usage_gate=gate)
        agent = build_clone(
            host.app_scope(
                workspace_root=self._user_workspace(user_id),
                persona_registry=self._registry,
                gateway=gateway,
                budget=budget,
                skills=(
                    cast("SkillRegistryProtocol", _ReadOnlySkills(self._skills))
                    if self._skills is not None
                    else None
                ),
            ),
            clone_id=seat_id_for(clone_id),
            session_id=session_id,
            display_name=display_name,
            seat_framing=seat_framing,
        ).agent
        agent.hydrate_session()
        return ComposedClone(agent, llm, store, budget, turns_so_far)

    async def __call__(self, cmd: TurnCommandEnvelope, bus: EventBusProtocol) -> TurnExecution:
        """Rehydrate the session into a new agent and run one turn on it."""
        if cmd.seat is not None:
            return await self._run_seat(cmd, cmd.seat, bus)
        clone = self._compose(
            user_id=cmd.user_id,
            session_id=cmd.session_id,
            turn_id=cmd.turn_id,
            clone_id=cmd.agent_id,
            bus=bus,
        )
        agent = clone.agent
        checkpoint = agent.checkpoint_turn(cmd.session_id)
        stream = self._stream_callback(cmd)
        try:
            if cmd.continues_stopped_turn:
                result = await agent.continue_turn(
                    stream_callback=stream, caller_turn_id=cmd.turn_id
                )
            else:
                result = await agent.execute_turn(
                    cmd.prompt, stream_callback=stream, caller_turn_id=cmd.turn_id
                )
        finally:
            await clone.llm.aclose()
        if result.error is not None:
            # Nothing partial stays in the session, as a room seat rolls back a turn that did
            # not land (#1423). A turn its usage ceiling stopped has no error: it is kept,
            # because it continues in place (#2431).
            agent.roll_back_turn(checkpoint, reason="turn_not_committed")
        return TurnExecution(result=result, commit=self._commit_of(cmd, clone))

    @staticmethod
    def _commit_of(
        cmd: TurnCommandEnvelope, clone: ComposedClone
    ) -> Callable[[Mapping[str, Any]], Awaitable[SessionState]]:
        async def commit(terminal: Mapping[str, Any]) -> SessionState:
            clone.store.arm_commit(
                cmd.turn_id,
                _budget_snapshot(clone.budget, cmd.session_id, clone.turns_so_far + 1),
                terminal,
            )
            # On a thread: the agent's save is synchronous, and bridges to the store's clients.
            return await asyncio.to_thread(clone.agent.persist_session, cmd.session_id)

        return commit

    async def _run_seat(
        self, cmd: TurnCommandEnvelope, seat: SeatTurnRef, bus: EventBusProtocol
    ) -> TurnExecution:
        """One speaker's turn: the same clone, with what a room adds, on the shared step."""
        room, speaker = await self._seat_of(seat.room_id, cmd.agent_id)
        clone = self._compose(
            user_id=cmd.user_id,
            session_id=cmd.session_id,
            turn_id=cmd.turn_id,
            clone_id=speaker.id,
            bus=bus,
            display_name=speaker.display_name or speaker.id,
            seat_framing=self._framing_of(room, speaker),
        )
        # The worker's user's folder, whatever the room names: a room's `workspace` is a folder
        # on one person's machine, and a worker serves many people on one OS account.
        own_folder = room.model_copy(update={"workspace": None})
        plan = plan_speaker_step(clone.agent, speaker, own_folder)
        await self._publish_room(cmd, seat.room_id, {"status": "generating"})
        try:
            step = await execute_single_speaker_step(
                clone.agent,
                speaker,
                own_folder,
                plan,
                turn_id=cmd.turn_id,
                stream_callback=self._seat_stream_callback(cmd, seat.room_id),
            )
        finally:
            await clone.llm.aclose()
        if step.raised is not None:
            # A row records this on one machine, where nothing else would. Here the worker's
            # policy decides, as for any turn, and its cancellation stays a cancellation.
            raise step.raised
        if step.result is None:
            raise UCloneXError(f"The seat turn {cmd.turn_id} came to nothing")
        if step.outcome.persist_error is not None:
            raise UCloneXError(f"The seat turn {cmd.turn_id} could not be rolled back")
        return TurnExecution(
            result=step.result, commit=self._commit_of(cmd, clone), landing=step.outcome
        )

    async def _seat_of(self, room_id: str, speaker_id: str) -> tuple[RoomState, Participant]:
        if self._rooms is None:
            raise UCloneXError("AgentTurnExecutor needs a room store to run a seat turn")
        room = await self._rooms.read_room(room_id)
        seats = room.participants if room is not None else ()
        speaker = next((p for p in seats if p.id == speaker_id), None)
        if room is None or speaker is None:
            raise UCloneXError(f"Room {room_id} has no seat {speaker_id}")
        return room, speaker

    @staticmethod
    def _framing_of(room: RoomState, speaker: Participant) -> str:
        return "" if is_one_seat(room.participants) else room_participant_system_prompt(speaker)

    async def decide_for_room(
        self, state: RoomState, user_id: str, command_id: str
    ) -> SpeakerDecision:
        """The room's selector chain, on a connector billed to the room's user."""
        llm, _ = self._connector(user_id=user_id, session_id=state.room_id, turn_id=command_id)
        try:
            return await decide_speaker(build_selector_chain(state.policy, provider=llm), state)
        finally:
            await llm.aclose()

    def clone_learns(self, clone_id: str) -> bool:
        """The clone's "Learn from conversations" switch (#1840), as its persona says now."""
        persona = self._registry.get_persona(clone_id)
        return persona is None or persona.learn_from_conversations

    def compose_learner(
        self, room: RoomState, speaker: Participant, *, user_id: str, job_id: str
    ) -> ComposedClone:
        """`speaker`'s clone, for a model call of its own after its turns have ended."""
        return self._compose(
            user_id=user_id,
            session_id=speaker.session_id,
            turn_id=job_id,
            clone_id=speaker.id,
            bus=EventBus(),
            display_name=speaker.display_name or speaker.id,
            seat_framing=self._framing_of(room, speaker),
        )
