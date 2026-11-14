"""Cluster worker HostProtocol implementation and bounded-load sticky LLM endpoint router."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from uclone_x.core.capability import Capability
from uclone_x.core.host import HostProtocol
from uclone_x.errors import UCloneXError
from uclone_x.sandbox.models import IsolationLevel

if TYPE_CHECKING:
    from pathlib import Path

    from uclone_x.agent.clone_builder import AppScope
    from uclone_x.agent.persona_registry import PersonaRegistry
    from uclone_x.core.session_store import SessionStoreProtocol
    from uclone_x.core.workspace import WorkspaceProtocol
    from uclone_x.engine.protocols import EventBusProtocol
    from uclone_x.llm.gateway import ModelGateway
    from uclone_x.llm.protocols import LLMProviderProtocol, TokenBudgetManagerProtocol
    from uclone_x.sandbox.protocols import SandboxRunnerProtocol
    from uclone_x.skills.protocols import SkillRegistryProtocol
    from uclone_x.telemetry.protocols import TraceRecorderProtocol
    from uclone_x.tools.protocols import ToolRegistryProtocol

__all__ = ["ClusterHost", "StickyEndpointRouter"]


class ClusterHost(HostProtocol):
    """What one turn on a stateless worker has, built per turn and dropped with it.

    `app_scope` hands these members to `build_clone`, the one place a clone is assembled, so a
    worker's clone gets its request built exactly as in a 1:1 chat or a room seat (owner ruling
    2026-09-27): the host says what the process has, and nothing about the agent is decided here.
    """

    def __init__(
        self,
        *,
        llm: LLMProviderProtocol,
        tools: ToolRegistryProtocol,
        sessions: SessionStoreProtocol,
        tracer: TraceRecorderProtocol,
        bus: EventBusProtocol,
        workspace: WorkspaceProtocol | None = None,
        sandbox: SandboxRunnerProtocol | None = None,
        isolation_floor: IsolationLevel | None = None,
        available_isolation: frozenset[IsolationLevel] = frozenset(),
        extra_capabilities: Iterable[Capability] = (),
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._sessions = sessions
        self._tracer = tracer
        self._bus = bus
        self._workspace = workspace
        self._sandbox = sandbox
        self._isolation_floor = isolation_floor
        self._available_isolation = available_isolation

        derived: set[Capability] = set(extra_capabilities)
        if workspace is not None:
            derived.add(Capability.WORKSPACE_READ)
            derived.add(Capability.WORKSPACE_WRITE)
        if sandbox is not None and isolation_floor is not None:
            derived.add(Capability.PROCESS_EXEC)
        self._capabilities = frozenset(derived)

    def app_scope(
        self,
        *,
        workspace_root: Path | None,
        persona_registry: PersonaRegistry,
        gateway: ModelGateway,
        budget: TokenBudgetManagerProtocol,
        skills: SkillRegistryProtocol | None,
    ) -> AppScope:
        """The `AppScope` a worker's `build_clone` runs on.

        No memory and no rules engine: both are local files today, and a worker keeps no
        state between turns, so a clone answering here has neither (`None` is a statement,
        core/host.py). Nobody answers an approval during a worker's turn.
        """
        from uclone_x.agent.clone_builder import AppScope, connector_tool_binder

        return AppScope.create(
            workspace_root=workspace_root,
            persona_registry=persona_registry,
            gateway=gateway,
            llm=self._llm,
            tools=self._tools,
            tracer=self._tracer,
            bus=self._bus,
            store=self._sessions,
            workspace=self._workspace,
            sandbox=self._sandbox,
            isolation_floor=self._isolation_floor,
            available_isolation=self._available_isolation,
            budget=budget,
            skills=skills,
            tool_binder=connector_tool_binder(self._llm),
            approvals_answered=False,
        )

    @property
    def llm(self) -> LLMProviderProtocol:
        return self._llm

    @property
    def tools(self) -> ToolRegistryProtocol:
        return self._tools

    @property
    def sessions(self) -> SessionStoreProtocol:
        return self._sessions

    @property
    def tracer(self) -> TraceRecorderProtocol:
        return self._tracer

    @property
    def bus(self) -> EventBusProtocol:
        return self._bus

    @property
    def workspace(self) -> WorkspaceProtocol | None:
        return self._workspace

    @property
    def sandbox(self) -> SandboxRunnerProtocol | None:
        return self._sandbox

    @property
    def isolation_floor(self) -> IsolationLevel | None:
        return self._isolation_floor

    @property
    def available_isolation(self) -> frozenset[IsolationLevel]:
        return self._available_isolation

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities


class StickyEndpointRouter:
    """Bounded-load consistent hash router for session-sticky GPU/LLM inference endpoints."""

    def __init__(
        self,
        endpoints: Sequence[str],
        max_inflight_per_endpoint: int = 8,
    ) -> None:
        cleaned = tuple(ep.strip() for ep in endpoints if ep.strip())
        if not cleaned:
            raise ValueError("StickyEndpointRouter requires at least one non-empty endpoint URL")
        if max_inflight_per_endpoint < 1:
            raise ValueError("max_inflight_per_endpoint must be >= 1")

        # Sort endpoints by deterministic SHA-256 ring position so ring neighbor traversal
        # is stable across all stateless worker processes.
        self._ring: tuple[str, ...] = tuple(sorted(dict.fromkeys(cleaned), key=self._hash_key))
        self._max_inflight_per_endpoint = max_inflight_per_endpoint
        self._inflight: dict[str, int] = dict.fromkeys(self._ring, 0)
        self._unhealthy: set[str] = set()

    @staticmethod
    def _hash_key(value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    @property
    def endpoints(self) -> tuple[str, ...]:
        """Ordered ring of distinct endpoint URLs."""
        return self._ring

    @property
    def max_inflight_per_endpoint(self) -> int:
        """Maximum concurrent requests before spilling over to the next ring neighbor."""
        return self._max_inflight_per_endpoint

    def mark_unhealthy(self, endpoint: str) -> None:
        """Mark `endpoint` unhealthy so traffic spills over to its ring neighbor."""
        if endpoint not in self._inflight:
            raise ValueError(f"Unknown endpoint: {endpoint!r}")
        self._unhealthy.add(endpoint)

    def mark_healthy(self, endpoint: str) -> None:
        """Restore `endpoint` to the healthy routing pool."""
        if endpoint not in self._inflight:
            raise ValueError(f"Unknown endpoint: {endpoint!r}")
        self._unhealthy.discard(endpoint)

    def is_healthy(self, endpoint: str) -> bool:
        """Return True if `endpoint` is currently marked healthy."""
        if endpoint not in self._inflight:
            raise ValueError(f"Unknown endpoint: {endpoint!r}")
        return endpoint not in self._unhealthy

    def acquire_inflight(self, endpoint: str) -> None:
        """Increment active in-flight request count for `endpoint`."""
        if endpoint not in self._inflight:
            raise ValueError(f"Unknown endpoint: {endpoint!r}")
        self._inflight[endpoint] += 1

    def release_inflight(self, endpoint: str) -> None:
        """Decrement active in-flight request count for `endpoint`."""
        if endpoint not in self._inflight:
            raise ValueError(f"Unknown endpoint: {endpoint!r}")
        if self._inflight[endpoint] > 0:
            self._inflight[endpoint] -= 1

    def inflight_count(self, endpoint: str) -> int:
        """Return current in-flight count for `endpoint`."""
        if endpoint not in self._inflight:
            raise ValueError(f"Unknown endpoint: {endpoint!r}")
        return self._inflight[endpoint]

    def select_endpoint(self, session_id: str) -> tuple[str, Mapping[str, str]]:
        """Select a healthy endpoint with bounded load and return `(url, headers)`.

        Hashes `session_id` deterministically onto the endpoint ring and walks clockwise
        until it finds a healthy endpoint with `inflight < max_inflight_per_endpoint`.
        Raises `UCloneXError` if all endpoints are unhealthy (or saturated).
        """
        if not session_id.strip():
            raise ValueError("session_id must not be blank")

        ring_size = len(self._ring)
        primary_index = self._hash_key(session_id) % ring_size
        headers: dict[str, str] = {"X-Session-ID": session_id}

        for offset in range(ring_size):
            candidate = self._ring[(primary_index + offset) % ring_size]
            if candidate in self._unhealthy:
                continue
            if self._inflight[candidate] >= self._max_inflight_per_endpoint:
                continue
            return candidate, headers

        raise UCloneXError(
            f"No available healthy endpoint under max_inflight={self._max_inflight_per_endpoint} "
            f"for session '{session_id}'"
        )
