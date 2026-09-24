"""Concrete implementation of ClientDriverProtocol for ACP shell driver."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from uclone_x.engine.event_bus import AgentEvent, EventBus, EventSource, EventType

if TYPE_CHECKING:
    from uclone_x.agent.base import BaseAgent


class ACPClientDriver:
    """Client driver implementation connecting the ACP shell to the agent kernel."""

    def __init__(
        self,
        agent: BaseAgent | None = None,
        bus: EventBus | None = None,
        session_id: str = "default",
    ) -> None:
        self._agent = agent
        self._bus = bus
        self._session_id = session_id
        self._active_task: asyncio.Task[None] | None = None
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """Whether the driver has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the client driver and perform any required handshake."""
        self._initialized = True

    async def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        """Submit a user prompt and yield the turn event stream."""
        topic = f"session.{self._session_id}"
        sub = None
        if self._bus is not None:
            sub = self._bus.subscribe({topic})

        # Yield initial USER_INPUT event
        yield AgentEvent(
            type=EventType.USER_INPUT,
            topic=topic,
            source=EventSource.USER,
            payload={"content": content},
        )

        if self._agent is not None:
            turn_result = await self._agent.execute_turn(content)
            yield AgentEvent(
                type=EventType.AGENT_REPLY,
                topic=topic,
                source=EventSource.AGENT,
                payload={"content": getattr(turn_result, "content", "")},
            )
        else:
            yield AgentEvent(
                type=EventType.AGENT_REPLY,
                topic=topic,
                source=EventSource.AGENT,
                payload={"content": f"Echo: {content}"},
            )

        if sub is not None:
            sub.close()

    async def cancel(self, reason: str) -> None:
        """Signal turn cancellation with a mandatory reason."""
        if self._active_task is not None and not self._active_task.done():
            self._active_task.cancel()
            self._active_task = None
        if self._bus is not None:
            await self._bus.publish(
                AgentEvent(
                    type=EventType.INTERRUPT,
                    topic=f"session.{self._session_id}",
                    source=EventSource.USER,
                    payload={"reason": reason},
                )
            )
