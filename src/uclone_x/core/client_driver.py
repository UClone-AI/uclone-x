from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from uclone_x.engine.event_bus import AgentEvent


class ClientDriverProtocol(Protocol):
    """Protocol describing what an interactive client driver consumes from the kernel.

    Adopting ACP requires specifying the boundary between the ACP shell and the kernel.
    This protocol defines the exact contract an interactive client driver consumes,
    mapping ACP operations to explicit contract members or explicitly stating why
    certain operations are unsupported.

    ACP Operations Map:
    - `initialize`: Defined explicitly. Initializes the connection/handshake between the driver and kernel.
    - `new_session`: Not supported on the driver protocol. Session lifecycle is managed by session routers.
    - `load_session`: Not supported on the driver protocol. Managed by session/kernel routers.
    - `prompt`: Defined explicitly. Submits a user prompt and yields an `AgentEvent` stream.
    - `set_session_mode`: Not supported on the driver protocol. Mode is a config out-of-band concern.
    - `set_config_option`: Not supported on the driver protocol. Config applies outside driver scope.
    - `cancel`: Defined explicitly. Takes a mandatory named reason.

    Turn Cancellation (P6 compliance):
    Turn cancellation must be explicitly handled via `cancel(reason)`. It cannot be silently
    accepted and ignored. The kernel handles it using a cancellation token or event, resulting
    in explicit refusal with the named reason recorded.

    Streaming & Turn Events:
    The `prompt` method returns an `AsyncIterator[AgentEvent]`. The turn event stream is tightly
    coupled with durable events (Issue #566). Each `AgentEvent` (differentiated by `EventType`)
    forms a persistent ledger of the turn's progress and state.
    """

    async def initialize(self) -> None:
        """Initialize the client driver and perform any required handshake."""
        ...

    async def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        """Submit a prompt and stream back the turn events.

        Args:
            content: The user prompt.

        Returns:
            An async iterator yielding `AgentEvent` items, representing the durable turn event stream.
        """
        ...

    async def cancel(self, reason: str) -> None:
        """Signal turn cancellation with a mandatory reason.

        Args:
            reason: Explicit reason for cancellation. Never silently ignored (P6 compliance).
        """
        ...
