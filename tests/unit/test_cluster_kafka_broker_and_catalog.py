"""Unit tests for the Kafka broker's envelope mapping, the catalog cache and the Valkey TTL rule.

What needs a real server — partitioning, acks, consumer offsets, scripts, eviction — is in
`tests/integration/cluster_live/`. These tests stop at the library boundary.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.support.cluster_fakes import FakeCache
from uclone_x.adapters.cluster.catalog_cache import ValkeyCatalogCache
from uclone_x.adapters.cluster.kafka_broker import (
    TOPIC_CMD_TURN,
    TOPIC_STREAM_SSE,
    KafkaCommandBroker,
)
from uclone_x.adapters.cluster.valkey_cache import ValkeyClusterCache
from uclone_x.core.cluster_protocols import SSEStreamEnvelope, TurnCommandEnvelope
from uclone_x.core.loop_local import LoopLocal


class _RecordingProducer:
    """Stands in for `AIOKafkaProducer` at the library boundary."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes, bytes]] = []

    async def send_and_wait(self, topic: str, value: bytes, *, key: bytes) -> Any:
        self.sent.append((topic, key, value))

        class _Meta:
            offset = len(self.sent) - 1

        return _Meta()


def _broker_with(producer: _RecordingProducer) -> KafkaCommandBroker:
    broker = KafkaCommandBroker(bootstrap_servers=("127.0.0.1:1",))

    async def _start() -> Any:
        return producer

    async def _stop(_: Any) -> None:
        return None

    broker._producers = LoopLocal(_start, _stop)  # pyright: ignore[reportPrivateUsage]
    return broker


async def test_kafka_broker_writes_centrifugo_publish_commands_keyed_by_conversation() -> None:
    """`publish_sse` writes the Centrifugo `publish` API command its Kafka consumer executes.

    Killed by: src/uclone_x/adapters/cluster/kafka_broker.py :: await self._send(TOPIC_STREAM_SSE, conversation_id, command)
    Becomes: await self._send(TOPIC_STREAM_SSE, envelope.channel, command)
    """
    producer = _RecordingProducer()
    broker = _broker_with(producer)
    envelope = SSEStreamEnvelope(
        channel="ucx:session:s-1", idempotency_key="s-1:7", data={"seq": 7, "payload": {}}
    )
    await broker.publish_sse("s-1", envelope)
    offset = await broker.publish_turn(
        TurnCommandEnvelope(
            turn_id="t-1",
            session_id="s-1",
            agent_id="a",
            user_id="u",
            prompt="hi",
            enqueued_at_ms=1,
        )
    )

    (sse_topic, sse_key, sse_value), (turn_topic, turn_key, _) = producer.sent
    assert (sse_topic, sse_key) == (TOPIC_STREAM_SSE, b"s-1")
    assert json.loads(sse_value) == {
        "method": "publish",
        "payload": {
            "channel": "ucx:session:s-1",
            "data": {"payload": {}, "seq": 7},
            "idempotency_key": "s-1:7",
        },
    }
    assert (turn_topic, turn_key, offset) == (TOPIC_CMD_TURN, b"s-1", 1)


async def test_catalog_cache_writes_are_cache_class_with_a_ttl() -> None:
    """Catalog entries and the refresh lock are cache keys: always written with a TTL.

    Killed by: src/uclone_x/adapters/cluster/catalog_cache.py :: return await self._cache.add_cache(self._lock_key(provider), b"1", ttl_seconds=ttl_seconds)
    Becomes: return True
    """
    cache = FakeCache()
    catalog = ValkeyCatalogCache(cache, default_ttl_seconds=300)

    assert await catalog.get_models("openai") is None
    assert await catalog.acquire_refresh_lock("openai", ttl_seconds=15) is True
    assert await catalog.acquire_refresh_lock("openai", ttl_seconds=15) is False
    await catalog.set_models("openai", [{"id": "gpt-4o", "context_window": 128000}])
    assert await catalog.release_refresh_lock("openai") is True

    cached = await catalog.get_models("openai")
    assert cached is not None and cached[0]["id"] == "gpt-4o"
    assert cache.ttls == {b"catalog:models:{openai}": 300}


async def test_valkey_cache_refuses_a_cache_write_without_ttl() -> None:
    """A TTL-less cache write would be a key `volatile-lru` can never evict; it is refused.

    Killed by: src/uclone_x/adapters/cluster/valkey_cache.py :: if ttl_seconds < 1:
    Becomes: if False:
    """
    cache = ValkeyClusterCache(nodes=("127.0.0.1:1",), password="unused")
    with pytest.raises(ValueError, match="TTL"):
        await cache.set_cache(b"sess:head:{x}", b"1", ttl_seconds=0)
    with pytest.raises(ValueError, match="TTL"):
        await cache.add_cache(b"blob:{x}:d", b"1", ttl_seconds=0)
