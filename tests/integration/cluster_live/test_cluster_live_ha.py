"""High-availability behaviour of the compose cluster: failover, eviction, broker restart.

These tests kill and restart containers of the stack under test, and restore it before they
return: the killed Valkey primary is started again and rejoins as a replica, and the temporary
`maxmemory` is put back.
"""

from __future__ import annotations

import asyncio
import importlib
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from tests.integration.cluster_live.conftest import container, docker, eventually, raw_valkey
from uclone_x.adapters.cluster.config import ClusterRuntimeConfig
from uclone_x.adapters.cluster.kafka_broker import TOPIC_CMD_TURN, KafkaCommandBroker
from uclone_x.adapters.cluster.sse_bridge import KafkaSSEBridge
from uclone_x.adapters.cluster.turn_lease import ValkeyTurnLeaseManager
from uclone_x.core.cluster_protocols import TurnCommandEnvelope, TurnLeaseConflictError
from uclone_x.core.session_state import SessionState
from uclone_x.errors import StaleSessionWriteError
from uclone_x.llm.usage.store import UsageEntry
from uclone_x.shells.cluster.runtime_stack import ClusterRuntimeStack

pytestmark = [pytest.mark.cluster_live, pytest.mark.allow_network]


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _service_for_port(config: ClusterRuntimeConfig, port: int) -> str:
    for announced, published in config.valkey_address_map.items():
        if published.endswith(f":{port}"):
            return announced.split(":")[0]
    raise AssertionError(f"no valkey service publishes port {port}")


async def _cluster_ok(client: Any) -> bool:
    info: Any = await client.cluster_info()
    return str(info.get("cluster_state")) == "ok"


async def test_valkey_primary_failover_keeps_lease_and_fencing(
    stack: ClusterRuntimeStack, second_stack: ClusterRuntimeStack
) -> None:
    """Kill the primary that owns a held lease: its replica takes over with lease and fence intact.

    Replication is asynchronous, so the test first waits until the replica has the lease (an
    unreplicated write can be lost by design; PostgreSQL's revision check covers that case,
    asserted at the end).
    """
    conv = _uid("failover")
    lease_key = ValkeyTurnLeaseManager.lease_key(conv)
    fence_key = ValkeyTurnLeaseManager.fence_key(conv)
    client = await raw_valkey(stack)

    sid = conv
    base = await asyncio.to_thread(
        stack.session_store.save, SessionState(session_id=sid, agent_id="a")
    )
    token = await stack.lease_manager.acquire(conv, "t1", "worker-a")
    lease_value = await stack.cache.get(lease_key)

    # Backstop for a freshly formed cluster (`ucx cluster up` already waits for this): the
    # client's topology lists a replica only once it has attached, so refresh until it does.
    async def _primary_and_replica() -> tuple[Any, Any] | None:
        await client.nodes_manager.initialize()
        found_primary = client.get_node_from_key(lease_key)
        found_replica = client.get_node_from_key(lease_key, replica=True)
        if found_replica is None or found_replica.name == found_primary.name:
            return None
        return found_primary, found_replica

    primary, replica = await eventually(_primary_and_replica, timeout=60)

    valkey_asyncio: Any = importlib.import_module("valkey.asyncio")
    replica_client: Any = valkey_asyncio.Valkey(
        host="127.0.0.1", port=replica.port, password=stack.config.valkey_password
    )

    async def _replicated() -> bool:
        await replica_client.execute_command("READONLY")
        return await replica_client.get(lease_key) == lease_value

    try:
        await eventually(_replicated, timeout=10)
    finally:
        await replica_client.aclose()

    killed = container(_service_for_port(stack.config, primary.port))
    docker("kill", killed)
    try:
        # Worker B keeps trying until the promoted replica serves the slot: the held lease must
        # still refuse it, and the fencing counter must not have gone backwards.
        async def _refused_by_survivor() -> bool:
            try:
                await second_stack.lease_manager.acquire(conv, "t1", "worker-b")
            except TurnLeaseConflictError as conflict:
                return conflict.holder == "worker-a"
            raise AssertionError("worker-b acquired a lease worker-a still held")

        assert await eventually(_refused_by_survivor, timeout=60)
        survivor_fence = await stack.cache.get(fence_key)
        assert survivor_fence is not None and int(survivor_fence) >= token.fencing_token

        await stack.lease_manager.release(token)
        successor = await second_stack.lease_manager.acquire(conv, "t2", "worker-b")
        assert successor.fencing_token > token.fencing_token
        await second_stack.lease_manager.release(successor)

        # The last line of defence, whatever the cache lost: a zombie commit is refused.
        winner = await asyncio.to_thread(second_stack.session_store.save, base)
        assert winner.revision == 2
        with pytest.raises(StaleSessionWriteError):
            await asyncio.to_thread(stack.session_store.save, base)
    finally:
        docker("start", killed)

    await eventually(lambda: _cluster_ok(client), timeout=90)

    async def _six_nodes_connected() -> bool:
        nodes: Any = await client.cluster_nodes()
        return sum(1 for n in nodes.values() if n.get("connected")) >= 6

    await eventually(_six_nodes_connected, timeout=90)


async def test_flooding_the_cache_evicts_cache_keys_never_coordination_keys(
    stack: ClusterRuntimeStack,
) -> None:
    """Under `maxmemory`, `volatile-lru` evicts TTL-carrying cache keys and nothing else."""
    client = await raw_valkey(stack)
    primaries: Any = client.get_primaries()
    original: dict[str, str] = {}
    for node in primaries:
        conf: Any = await client.config_get("maxmemory", target_nodes=node)
        original[node.name] = str(conf["maxmemory"])
        policy: Any = await client.config_get("maxmemory-policy", target_nodes=node)
        assert policy["maxmemory-policy"] == "volatile-lru"

    convs = [_uid("coord") for _ in range(24)]
    flood = _uid("flood")
    flood_keys = [f"blob:{{{flood}-{i}}}:d".encode() for i in range(1600)]
    try:
        # 16 MB of room above what each primary already uses (scripts, buffers, Centrifugo).
        for node in primaries:
            memory: Any = await client.info("memory", target_nodes=node)
            limit = int(memory["used_memory"]) + 16 * 1024 * 1024
            await client.config_set("maxmemory", str(limit), target_nodes=node)

        for conv in convs:
            token = await stack.lease_manager.acquire(conv, "t1", "w", ttl_seconds=3600)
            assert token.fencing_token >= 1
            await stack.lease_manager.write_cancel_tombstone(conv, "t9", ttl_seconds=3600)
            await stack.cache.incr(KafkaSSEBridge.sequence_key(conv), 1)
            await stack.usage_store.record_completed_call(
                scope=conv,
                session_id=conv,
                turn_id="t1",
                entry=UsageEntry(at=datetime.now(UTC), tokens=10, provider="flood-test"),
                counts_against_limits=True,
            )

        evicted_before = await _evicted(client, primaries)
        payload = b"x" * 65536
        written = 0
        while written < len(flood_keys):  # ~100 MB against 3 x 16 MB of room
            batch = [
                stack.cache.set_cache(key, payload, ttl_seconds=600)
                for key in flood_keys[written : written + 8]
            ]
            try:
                await asyncio.gather(*batch)
            except Exception as refused:  # OOM: nothing evictable, i.e. cache keys lost their TTL
                raise AssertionError(
                    f"a cache write was refused at maxmemory ({refused}); cache keys must carry "
                    "a TTL so volatile-lru can evict them"
                ) from refused
            written += 8
        evicted = await _evicted(client, primaries) - evicted_before
        assert evicted > 0, "the flood never reached maxmemory"

        for conv in convs:
            for key in (
                ValkeyTurnLeaseManager.lease_key(conv),
                ValkeyTurnLeaseManager.fence_key(conv),
                ValkeyTurnLeaseManager.cancel_key(conv),
                KafkaSSEBridge.sequence_key(conv),
                stack.usage_store.window_key(conv),
            ):
                assert await client.exists(key) == 1, f"coordination key evicted: {key!r}"
                assert int(await client.ttl(key)) == -1, key
        with pytest.raises(TurnLeaseConflictError):
            await stack.lease_manager.acquire(convs[0], "t1", "intruder")
    finally:
        for node in primaries:
            await client.config_set("maxmemory", original[node.name], target_nodes=node)
        await stack.cache.delete(*flood_keys)  # leave the stack as it was found


async def _evicted(client: Any, primaries: Any) -> int:
    total = 0
    for node in primaries:
        stats: Any = await client.info("stats", target_nodes=node)
        total += int(stats.get("evicted_keys", 0))
    return total


async def test_kafka_broker_restart_loses_no_acknowledged_command(
    stack: ClusterRuntimeStack,
) -> None:
    """Commands acknowledged before a broker restart are all still readable after it."""
    run = _uid("restart")

    def _turn(i: int) -> TurnCommandEnvelope:
        return TurnCommandEnvelope(
            turn_id=f"{run}-{i}",
            session_id=f"{run}-s{i % 7}",
            agent_id="a",
            user_id="u",
            prompt=f"p{i}",
            enqueued_at_ms=1,
        )

    for i in range(40):
        await stack.broker.publish_turn(_turn(i))

    consumer = KafkaCommandBroker(
        bootstrap_servers=stack.config.kafka_bootstrap_servers, group_id=f"{run}-group"
    )
    await consumer.subscribe([TOPIC_CMD_TURN])
    seen: set[str] = set()
    try:
        # Consume and commit part of the stream before the restart.
        async def _first_part() -> bool:
            for rec in await consumer.poll(timeout_ms=1000):
                turn_id = str(rec.payload.get("turn_id", ""))
                if turn_id.startswith(run):
                    seen.add(turn_id)
                await consumer.commit_partition_offset(rec.topic, rec.partition, rec.offset + 1)
            return len(seen) >= 15

        await eventually(_first_part, timeout=60, interval=0.1)
        await consumer.stop()

        docker("restart", container("kafka"))
        await stack.broker.stop()

        async def _broker_back() -> bool:
            return await stack.broker.ping()

        await eventually(_broker_back, timeout=90)
        for i in range(40, 50):
            await stack.broker.publish_turn(_turn(i))

        resumed = KafkaCommandBroker(
            bootstrap_servers=stack.config.kafka_bootstrap_servers, group_id=f"{run}-group"
        )
        await resumed.subscribe([TOPIC_CMD_TURN])
        try:
            expected = {f"{run}-{i}" for i in range(50)}

            async def _all_seen() -> bool:
                for rec in await resumed.poll(timeout_ms=1000):
                    turn_id = str(rec.payload.get("turn_id", ""))
                    if turn_id.startswith(run):
                        seen.add(turn_id)
                return expected <= seen

            await eventually(_all_seen, timeout=90, interval=0.1)
        finally:
            await resumed.stop()
    finally:
        await consumer.stop()
    assert expected <= seen
