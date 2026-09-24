"""In-memory zero-copy fastpath transport for co-located A2A agents."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

from uclone_x.a2a.models import TaskMessage, TaskResult, WireProtocolType
from uclone_x.a2a.wire import task_result_to_wire_json
from uclone_x.errors import (
    MissingProvenanceError,
    TaskNotFoundError,
)

TaskHandler = Callable[[TaskMessage], Awaitable[TaskResult]]
StreamHandler = Callable[[TaskMessage], AsyncIterator[str]]


class A2AInMemoryTransport:
    """Zero-copy in-memory transport for co-located agents within the same process."""

    def __init__(self) -> None:
        self._handlers: dict[str, TaskHandler] = {}
        self._stream_handlers: dict[str, StreamHandler] = {}

    @property
    def transport_type(self) -> WireProtocolType:
        """Transport mechanism identifier."""
        return WireProtocolType.LOCAL_IN_MEMORY

    def register_handler(
        self,
        endpoint_or_agent_id: str,
        handler: TaskHandler,
    ) -> None:
        """Register a direct async task handler for an endpoint or agent id."""
        self._handlers[endpoint_or_agent_id] = handler

    def register_stream_handler(
        self,
        endpoint_or_agent_id: str,
        handler: StreamHandler,
    ) -> None:
        """Register a streaming task handler for an endpoint or agent id."""
        self._stream_handlers[endpoint_or_agent_id] = handler

    def unregister_handler(self, endpoint_or_agent_id: str) -> None:
        """Unregister handlers associated with an endpoint or agent id."""
        self._handlers.pop(endpoint_or_agent_id, None)
        self._stream_handlers.pop(endpoint_or_agent_id, None)

    async def send_task(self, target_endpoint: str, message: TaskMessage) -> TaskResult:
        """Dispatch task directly to local in-process agent without network or serialization."""
        handler = self._handlers.get(target_endpoint) or self._handlers.get(message.target_agent_id)
        if handler is None:
            raise TaskNotFoundError(
                f"No in-memory handler registered for endpoint '{target_endpoint}' "
                f"or agent '{message.target_agent_id}'"
            )
        result = await handler(message)
        if result.provenance is None:
            raise MissingProvenanceError(
                "TaskResult returned by in-memory handler must contain in-band provenance (P6)"
            )
        return result

    def stream_task(
        self,
        target_endpoint: str,
        message: TaskMessage,
    ) -> AsyncIterator[str]:
        """Stream task results via in-memory generator."""
        stream_handler = self._stream_handlers.get(target_endpoint) or self._stream_handlers.get(
            message.target_agent_id
        )
        if stream_handler is not None:
            return stream_handler(message)

        handler = self._handlers.get(target_endpoint) or self._handlers.get(message.target_agent_id)
        if handler is not None:

            async def _fallback_stream() -> AsyncIterator[str]:
                res = await handler(message)
                if res.provenance is None:
                    raise MissingProvenanceError(
                        "TaskResult returned by in-memory handler must contain in-band provenance (P6)"
                    )
                yield task_result_to_wire_json(res)

            return _fallback_stream()

        async def _not_found_stream() -> AsyncIterator[str]:
            raise TaskNotFoundError(
                f"No in-memory handler registered for endpoint '{target_endpoint}' "
                f"or agent '{message.target_agent_id}'"
            )
            # Make this an async generator function for type checkers
            if False:  # pragma: no cover
                yield ""

        return _not_found_stream()
