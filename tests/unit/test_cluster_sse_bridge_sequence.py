"""Unit tests for `KafkaSSEBridge` cluster-monotonic sequence stamping (#565) and its Kafka hand-off."""

from __future__ import annotations

from tests.support.cluster_fakes import FakeBroker, FakeCache
from uclone_x.adapters.cluster.sse_bridge import KafkaSSEBridge
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType


async def test_sse_bridge_allocates_monotonic_sequence_per_conversation_across_workers() -> None:
    """Two worker bridges sharing one cache stamp `1, 2, 3` per conversation, not per worker.

    Killed by: src/uclone_x/adapters/cluster/sse_bridge.py :: return f"seq:sse:{{{conversation_id}}}".encode()
    Becomes: return b"seq:sse:{global}"
    """
    shared = FakeCache()
    broker = FakeBroker()
    worker_a = KafkaSSEBridge(shared, broker, channel_kind="session")
    worker_b = KafkaSSEBridge(shared, broker, channel_kind="session")

    # Each worker's EventBus starts `sequence` at 0; the shared INCR is what orders them.
    e1 = AgentEvent(type=EventType.AGENT_REPLY, sender_id="clone-1", sequence=0, payload={"t": "a"})
    e2 = AgentEvent(type=EventType.AGENT_REPLY, sender_id="clone-1", sequence=0, payload={"t": "b"})
    e3 = AgentEvent(type=EventType.AGENT_REPLY, sender_id="clone-1", sequence=0, payload={"t": "c"})

    s1 = await worker_a.publish_event("sess-alpha", e1)
    s2 = await worker_b.publish_event("sess-alpha", e2)
    s3 = await worker_a.publish_event("sess-alpha", e3)
    assert (s1, s2, s3) == (1, 2, 3)
    assert await worker_b.publish_event("sess-beta", e1) == 1

    assert [env.idempotency_key for _, env in broker.sse] == [
        "sess-alpha:1",
        "sess-alpha:2",
        "sess-alpha:3",
        "sess-beta:1",
    ]
    # Partition key is the conversation, so one conversation's stream stays ordered in Kafka.
    assert [key for key, _ in broker.sse] == ["sess-alpha", "sess-alpha", "sess-alpha", "sess-beta"]
    # The sequence counter is coordination state: written by INCR, never given a TTL.
    assert shared.ttls[KafkaSSEBridge.sequence_key("sess-alpha")] is None


async def test_sse_bridge_attach_to_bus_forwards_only_while_attached() -> None:
    """`attach_to_bus` forwards every event of the turn to the broker, and stops after the turn.

    Killed by: src/uclone_x/adapters/cluster/sse_bridge.py :: await self._broker.publish_sse(clean_id, envelope)
    Becomes: pass
    """
    bridge_broker = FakeBroker()
    bridge = KafkaSSEBridge(FakeCache(), bridge_broker, channel_kind="room")

    async with EventBus() as bus:
        pub = bus.register_publisher("clone-1")
        async with bridge.attach_to_bus(bus, "room-7"):
            for step in (1, 2, 3):
                await pub.publish(
                    AgentEvent(type=EventType.TOOL_CALL, topic="room.step", payload={"step": step})
                )
        await pub.publish(
            AgentEvent(type=EventType.AGENT_REPLY, topic="room.step", payload={"step": 4})
        )
        await bus.wait_until_idle()

    envelopes = [env for _, env in bridge_broker.sse]
    assert [env.data["seq"] for env in envelopes] == [1, 2, 3]
    assert {env.channel for env in envelopes} == {"ucx:room:room-7"}
    assert [env.data["payload"] for env in envelopes] == [{"step": 1}, {"step": 2}, {"step": 3}]
