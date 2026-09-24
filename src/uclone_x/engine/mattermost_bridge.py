from __future__ import annotations

import asyncio
from typing import Any, Protocol

from uclone_x.engine.event_bus import (
    AgentEvent,
    BackpressurePolicy,
    EventBus,
    EventSubscription,
    EventType,
)


class TransportProtocol(Protocol):
    """Abstracts the remote transport mechanism (WebSocket/HTTP)."""

    async def send_payload(self, payload: dict[str, Any]) -> None:
        """Send payload to remote. Must raise an exception on failure."""

    @property
    def is_connected(self) -> bool:
        """Return True if the transport is currently connected."""
        ...


class UnmappableEventBridgeError(Exception):
    """Raised when an event cannot be translated to a uclone2 payload (Principle 6)."""


class EventBusMattermostBridge:
    """Translates EventBus events to uclone2-compatible Mattermost/WebSocket payloads.

    Follows Principle 6 (Truth in State) by explicitly accounting for all dropped
    events. Non-blocking backpressure is enforced via `DROP_LOWEST_PRIORITY` rather
    than `BLOCK`, preventing remote stalls from accumulating unbounded background tasks
    in the dispatcher.
    """

    def __init__(self, bus: EventBus, transport: TransportProtocol) -> None:
        self.bus = bus
        self.transport = transport
        self.drop_reasons: dict[str, int] = {
            "unmappable": 0,
            "disconnected": 0,
            "send_failed": 0,
        }
        self._sub: EventSubscription | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def total_dropped_count(self) -> int:
        """Return the total number of events dropped by the bridge."""
        return sum(self.drop_reasons.values())

    async def start(self, topics: set[str] | None = None) -> None:
        """Start the bridge with a non-blocking backpressure subscription."""
        if topics is None:
            topics = {"default"}

        # P6: Explicit non-blocking backpressure policy.
        # DROP_LOWEST_PRIORITY is chosen to prevent backpressure from stalling
        # the bus or creating unbounded `deliver` tasks if the transport stalls.
        self._sub = self.bus.subscribe(
            topics=topics,
            backpressure_policy=BackpressurePolicy.DROP_LOWEST_PRIORITY,
            maxsize=1000,
        )
        self._task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        """Stop the bridge and unsubscribe from the bus."""
        if self._sub:
            self._sub.unsubscribe()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _translate(self, event: AgentEvent) -> dict[str, Any]:
        """Translate internal bus events to uclone2 payloads.

        Raises UnmappableEventBridgeError for unsupported event types or
        missing fields, enforcing Principle 6 fail-loud semantics.
        """
        if event.type == EventType.TOOL_CALL:
            if "tool_name" not in event.payload:
                raise UnmappableEventBridgeError(f"TOOL_CALL missing tool_name: {event.event_id}")
            return {
                "type": "status_update",
                "status": f"Calling tool {event.payload['tool_name']}",
                "session_id": event.session_id,
            }
        elif event.type == EventType.AGENT_REPLY:
            if "content" not in event.payload:
                raise UnmappableEventBridgeError(f"AGENT_REPLY missing content: {event.event_id}")
            return {
                "type": "reply",
                "text": event.payload["content"],
                "session_id": event.session_id,
            }
        elif event.type == EventType.PLAN_STATUS_UPDATE:
            if "status" not in event.payload:
                raise UnmappableEventBridgeError(
                    f"PLAN_STATUS_UPDATE missing status: {event.event_id}"
                )
            return {
                "type": "status_update",
                "status": event.payload["status"],
                "session_id": event.session_id,
            }
        else:
            raise UnmappableEventBridgeError(f"Cannot map event type: {event.type}")

    async def _pump(self) -> None:
        """Continuously pull events from the subscription and forward them."""
        if not self._sub:
            return

        async for event in self._sub:
            try:
                payload = self._translate(event)
            except UnmappableEventBridgeError:
                self.drop_reasons["unmappable"] += 1
                continue

            if not self.transport.is_connected:
                # Event is dropped if the remote transport is disconnected.
                self.drop_reasons["disconnected"] += 1
                continue

            try:
                await self.transport.send_payload(payload)
            except Exception:
                # Event is dropped if sending fails.
                self.drop_reasons["send_failed"] += 1
