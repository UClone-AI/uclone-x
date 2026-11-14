"""IT-08, Invariant 7: one monotonic sequence per conversation, across workers.

Replay after a reconnect is Centrifugo's own history; `cluster_live/` reads it back.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.integration.cluster_harness import ClusterRig, Reply
from uclone_x.engine.event_bus import AgentEvent, EventType


@pytest.mark.asyncio
async def test_the_sequence_keeps_rising_when_the_next_turn_lands_on_another_worker(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    rig.script.replies.extend([Reply(text="one"), Reply(text="two")])
    first = rig.command("sess-seq", "first")
    second = rig.command("sess-seq", "second")
    await rig.worker("w1").execute_single_turn(first)
    await rig.worker("w2").execute_single_turn(second)

    stream = rig.stream("sess-seq")
    assert [e["seq"] for e in stream] == list(range(1, len(stream) + 1))
    by_turn = [e["payload"].get("turn_id") for e in stream if e["payload"].get("turn_id")]
    assert by_turn.index(second.turn_id) > max(
        i for i, turn in enumerate(by_turn) if turn == first.turn_id
    )


@pytest.mark.asyncio
async def test_events_published_at_once_reach_the_topic_in_sequence_order(tmp_path: Path) -> None:
    """A turn's bus events and its stream callback publish concurrently.

    Killed by: src/uclone_x/adapters/cluster/sse_bridge.py :: async with order.lock:
    Becomes: async with asyncio.Lock():
    """
    rig = ClusterRig(tmp_path)
    real_publish = rig.broker.publish_sse
    delays = iter([0.02, 0.0, 0.01, 0.0, 0.005, 0.0])

    async def slow_first_publish(conversation_id: str, envelope: object) -> None:
        await asyncio.sleep(next(delays))
        await real_publish(conversation_id, envelope)  # pyright: ignore[reportArgumentType]

    rig.broker.publish_sse = slow_first_publish  # type: ignore[method-assign]
    event = AgentEvent(
        type=EventType.AGENT_REPLY, topic="session.sess-race", session_id="sess-race", payload={}
    )
    await asyncio.gather(*(rig.sse.publish_event("sess-race", event) for _ in range(6)))

    assert [e["seq"] for e in rig.stream("sess-race")] == [1, 2, 3, 4, 5, 6]
