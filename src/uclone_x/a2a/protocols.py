"""Protocols for A2A dual-transport and discovery.

`@runtime_checkable` is applied only where a runtime `isinstance` check is actually
performed. On a protocol with a `@property`, `issubclass()` raises `TypeError` and
`isinstance()` calls the object's getters as a side effect of the type test, and neither
form checks a signature — which is what actually drifted in issue 2026-09-02-035.
Conformance is enforced statically instead, by the bindings in
`tests/unit/test_protocol_conformance.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from uclone_x.a2a.models import (
    AgentCard,
    TaskMessage,
    TaskResult,
    WireProtocolType,
)


@runtime_checkable
class A2ADiscoveryProtocol(Protocol):
    """Protocol for discovering and publishing Agent Cards."""

    def get_local_agent_card(self) -> AgentCard:
        """Export local agent capabilities."""
        ...

    async def fetch_remote_agent_card(self, endpoint_url: str) -> AgentCard:
        """Fetch /.well-known/agent-card.json from remote peer."""
        ...


class A2ATransportProtocol(Protocol):
    """Protocol for sending tasks across local zero-copy or remote REST/SSE transport."""

    @property
    def transport_type(self) -> WireProtocolType:
        """Transport mechanism (local vs remote)."""
        ...

    async def send_task(self, target_endpoint: str, message: TaskMessage) -> TaskResult:
        """Dispatch task to local or remote agent."""
        ...

    def stream_task(
        self,
        target_endpoint: str,
        message: TaskMessage,
    ) -> AsyncIterator[str]:
        """Stream task results via SSE or in-memory generator.

        Declared `def`, not `async def`: an async generator function *is* a plain
        function returning an `AsyncIterator`. As `async def ... -> AsyncIterator` the
        caller had to await the call to obtain the iterator and then iterate it, which
        no async generator can satisfy — caught by the conformance stub in
        `tests/unit/test_protocol_conformance.py` (issue 2026-09-02-035).
        """
        ...
