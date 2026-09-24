import asyncio
from typing import Any

import pytest

from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.engine.mattermost_bridge import (
    EventBusMattermostBridge,
)


class MockTransport:
    def __init__(self) -> None:
        self._connected = True
        self.sent_payloads: list[dict[str, Any]] = []
        self.fail_send = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def send_payload(self, payload: dict[str, Any]) -> None:
        if self.fail_send:
            raise RuntimeError("Network error")
        self.sent_payloads.append(payload)

    def disconnect(self) -> None:
        self._connected = False

    def reconnect(self) -> None:
        self._connected = True


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def transport() -> MockTransport:
    return MockTransport()


@pytest.mark.asyncio
async def test_bridge_translates_and_sends(bus: EventBus, transport: MockTransport) -> None:
    bridge = EventBusMattermostBridge(bus, transport)
    await bridge.start({"default"})

    pub = bus.register_publisher("test")
    await pub.publish(
        AgentEvent(
            type=EventType.TOOL_CALL, payload={"tool_name": "calculator"}, session_id="sess_123"
        )
    )

    await asyncio.sleep(0.01)  # Allow pump to process

    assert len(transport.sent_payloads) == 1
    assert transport.sent_payloads[0] == {
        "type": "status_update",
        "status": "Calling tool calculator",
        "session_id": "sess_123",
    }
    assert bridge.total_dropped_count == 0
    await bridge.stop()


@pytest.mark.asyncio
async def test_bridge_drops_unmappable_loudly(bus: EventBus, transport: MockTransport) -> None:
    bridge = EventBusMattermostBridge(bus, transport)
    await bridge.start({"default"})

    pub = bus.register_publisher("test")

    # Missing tool_name
    await pub.publish(AgentEvent(type=EventType.TOOL_CALL, payload={}, session_id="sess_123"))

    # Unsupported event type
    await pub.publish(AgentEvent(type=EventType.INTERRUPT, payload={}, session_id="sess_123"))

    await asyncio.sleep(0.01)

    assert len(transport.sent_payloads) == 0
    assert bridge.total_dropped_count == 2
    assert bridge.drop_reasons["unmappable"] == 2

    await bridge.stop()


@pytest.mark.asyncio
async def test_bridge_disconnected_drop(bus: EventBus, transport: MockTransport) -> None:
    bridge = EventBusMattermostBridge(bus, transport)
    await bridge.start({"default"})

    transport.disconnect()

    pub = bus.register_publisher("test")
    await pub.publish(
        AgentEvent(type=EventType.AGENT_REPLY, payload={"content": "hello"}, session_id="sess_1")
    )

    await asyncio.sleep(0.01)
    assert len(transport.sent_payloads) == 0
    assert bridge.drop_reasons["disconnected"] == 1

    transport.reconnect()
    await pub.publish(
        AgentEvent(type=EventType.AGENT_REPLY, payload={"content": "world"}, session_id="sess_1")
    )

    await asyncio.sleep(0.01)
    assert len(transport.sent_payloads) == 1
    assert bridge.drop_reasons["disconnected"] == 1
    assert bridge.total_dropped_count == 1

    await bridge.stop()


@pytest.mark.asyncio
async def test_bridge_send_failed_drop(bus: EventBus, transport: MockTransport) -> None:
    bridge = EventBusMattermostBridge(bus, transport)
    await bridge.start({"default"})

    transport.fail_send = True

    pub = bus.register_publisher("test")
    await pub.publish(
        AgentEvent(
            type=EventType.PLAN_STATUS_UPDATE,
            payload={"status": "Planning..."},
            session_id="sess_1",
        )
    )

    await asyncio.sleep(0.01)
    assert len(transport.sent_payloads) == 0
    assert bridge.drop_reasons["send_failed"] == 1
    assert bridge.total_dropped_count == 1

    await bridge.stop()
