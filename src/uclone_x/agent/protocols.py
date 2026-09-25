"""Protocols for agent lifecycle, state machine, and subagent management.

`@runtime_checkable` is applied only where a runtime `isinstance` check is actually
performed. On a protocol with a `@property`, `issubclass()` raises `TypeError` and
`isinstance()` calls the object's getters as a side effect of the type test, and neither
form checks a signature — which is what actually drifted in issue 2026-09-02-035.
Conformance is enforced statically instead, by the bindings in
`tests/unit/test_protocol_conformance.py`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    AgentState,
    PersonaDefinition,
    PlanState,
    PlanStep,
    SubagentInvocation,
    SubAgentSpec,
    TurnResult,
)
from uclone_x.agent.session import CompactionResult, SessionState
from uclone_x.engine.event_bus import AgentEvent
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.skills.protocols import SkillRegistryProtocol
from uclone_x.telemetry.protocols import TracerProtocol


class BaseAgentProtocol(Protocol):
    """Protocol defining the core agent interface."""

    @property
    def agent_id(self) -> str:
        """Unique agent identifier."""
        ...

    @property
    def state(self) -> AgentState:
        """Current reactive lifecycle state."""
        ...

    @property
    def context(self) -> AgentContext:
        """Active execution context."""
        ...

    @property
    def config(self) -> AgentConfig:
        """Static agent configuration."""
        ...

    @property
    def ontology(self) -> OntologyEngineProtocol | None:
        """Active domain ontology service if configured."""
        ...

    @property
    def skills(self) -> SkillRegistryProtocol | None:
        """Active skill registry if configured (P9)."""
        ...

    @property
    def loaded_skills(self) -> frozenset[str]:
        """Names of skills loaded into session context (P9)."""
        ...

    @property
    def tracer(self) -> TracerProtocol | None:
        """Active telemetry tracer if configured."""
        ...

    @property
    def turn_counter(self) -> int:
        """The active session's turn counter."""
        ...

    @property
    def current_plan(self) -> PlanState | None:
        """Active interactive chat plan if any."""
        ...

    def create_plan(
        self,
        title: str,
        steps: Sequence[str | PlanStep | Mapping[str, Any]],
        plan_id: str | None = None,
    ) -> PlanState:
        """Create and initialize a new interactive chat plan."""
        ...

    def update_step_status(
        self,
        index: int,
        completed: bool = True,
        verification: str | None = None,
    ) -> PlanState:
        """Update the completion status and verification criteria of a step."""
        ...

    def clear_plan(self) -> None:
        """Clear and remove the currently active plan."""
        ...

    async def start(self) -> None:
        """Start the agent event loop and subscriptions."""
        ...

    async def stop(self) -> None:
        """Gracefully terminate agent execution."""
        ...

    async def process_event(self, event: AgentEvent) -> bool:
        """Ingest and react to an incoming event."""
        ...

    async def execute_turn(
        self,
        input_data: str | AgentEvent,
        *,
        stream_callback: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
        caller_turn_id: str | None = None,
        room_id: str | None = None,
        story_id: str | None = None,
    ) -> TurnResult:
        """Execute a single reasoning turn.

        `room_id` and `story_id` name the conversation (room) the turn runs in and the
        story it has open, for the turn's tool calls (#1555).

        The `dict[str, Any]` arm is gone: the turn entry point accepted an arbitrary
        untyped payload, which is the contract shape P8 forbids (issue 2026-09-02-036).
        A caller with structured input builds an `AgentEvent`.
        """
        ...

    def hydrate_session(self, session_id: str | None = None) -> SessionState | None:
        """Load one session from the Core store, replacing what is held in memory."""
        ...

    def persist_session(self, session_id: str | None = None) -> SessionState:
        """Write one session to the Core store."""
        ...

    def checkpoint_turn(self, session_id: str | None = None) -> SessionState:
        """The session as it stands before a turn, for `roll_back_turn` (#1423)."""
        ...

    def roll_back_turn(self, checkpoint: SessionState, *, reason: str) -> int:
        """Return a session's conversation to `checkpoint`; the number of messages dropped.

        For a caller that commits a turn together with state of its own and must undo the
        seat's half when its own half does not commit (#1423). The turn's events stay in
        the log, and a `TURN_ROLLED_BACK` event records the rewind.
        """
        ...

    def reset_session(self, session_id: str | None = None) -> SessionState:
        """Purge a session back to its seeded state and zero its turn counter (#183).

        On the protocol rather than only on `BaseAgent` because it is the API that
        replaces three divergent reset implementations, one of which reached into
        `agent._history` through a `reportPrivateUsage` pragma. A reset a caller can
        only perform by touching private state is not an interface.

        `session_id` defaults to the active session. Returns the reset state so a caller
        can report or persist it without a second read.
        """
        ...

    async def compact_session(
        self, session_id: str | None = None, reason: str = "manual_on_demand"
    ) -> CompactionResult:
        """Compact one session's context on demand and announce it (#183, P5).

        Automatic compaction runs inside `execute_turn` when the configured token
        threshold is reached; this is the explicit entry point. The Core's part is
        persisting the compacted sequence and publishing `CONTEXT_COMPACTED` — the LLM
        layer owns the algorithm and the token estimate.
        """
        ...


@runtime_checkable
class SubAgentSupervisorProtocol(Protocol):
    """Protocol for dynamic sub-agent instantiation, supervision, and lifecycle."""

    def define_persona(self, persona: PersonaDefinition) -> None:
        """Register a reusable dynamic persona definition."""
        ...

    def get_persona(self, name: str) -> PersonaDefinition | None:
        """Retrieve registered persona definition by name."""
        ...

    async def spawn_subagent(
        self,
        spec: SubAgentSpec,
        parent_agent_id: str,
    ) -> str:
        """Spawn a new dynamic sub-agent with isolated context."""
        ...

    async def invoke_subagent(
        self,
        invocation: SubagentInvocation,
        parent_agent_id: str,
    ) -> str:
        """Instantiate and invoke a sub-agent from a registered persona definition."""
        ...

    async def terminate_subagent(self, agent_id: str) -> bool:
        """Gracefully terminate an active sub-agent."""
        ...

    def list_subagents(self, parent_agent_id: str | None = None) -> list[str]:
        """List active subagent IDs."""
        ...
