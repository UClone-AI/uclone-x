"""Store and adapter behaviour against the real compose cluster (Valkey, PostgreSQL, Kafka, Centrifugo).

Mutation evidence for these tests is recorded in the PR, not as `Killed by:` declarations:
the gate's lethality ratchet would run them offline, where they are skipped.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from tests.integration.cluster_live.conftest import raw_valkey
from uclone_x.adapters.cluster.knowledge_store import KnowledgeTripleRecord
from uclone_x.adapters.cluster.sse_bridge import KafkaSSEBridge
from uclone_x.adapters.cluster.turn_lease import ValkeyTurnLeaseManager
from uclone_x.core.cluster_protocols import (
    VERSIONED_HEAD_SET_SCRIPT,
    TokenBudgetSnapshot,
    TurnLeaseConflictError,
    TurnPreemptedError,
    decode_versioned_head,
)
from uclone_x.core.session_state import SessionState
from uclone_x.engine.event_bus import AgentEvent, EventType
from uclone_x.errors import StaleRoomWriteError, StaleSessionWriteError
from uclone_x.llm.usage.store import UsageEntry
from uclone_x.room.models import RoomState
from uclone_x.shells.cluster.runtime_stack import ClusterRuntimeStack

pytestmark = [pytest.mark.cluster_live, pytest.mark.allow_network]


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def test_postgres_occ_refuses_every_writer_but_one(
    stack: ClusterRuntimeStack, second_stack: ClusterRuntimeStack
) -> None:
    """Writers that loaded the same revision race; PostgreSQL commits exactly one of them."""
    sid = _uid("occ")
    base = await asyncio.to_thread(
        stack.session_store.save, SessionState(session_id=sid, agent_id="a")
    )
    assert base.revision == 1

    # Two workers (independent clients) both hold revision 1; the second commit must lose.
    first = await asyncio.to_thread(stack.session_store.save, base)
    assert first.revision == 2
    with pytest.raises(StaleSessionWriteError) as stale:
        await asyncio.to_thread(second_stack.session_store.save, base)
    assert stale.value.actual_revision == 2

    # Eight concurrent writers from revision 2: one wins, seven are refused.
    outcomes = await asyncio.gather(
        *(asyncio.to_thread(stack.session_store.save, first) for _ in range(8)),
        return_exceptions=True,
    )
    winners = [o for o in outcomes if isinstance(o, SessionState)]
    losers = [o for o in outcomes if isinstance(o, StaleSessionWriteError)]
    assert (len(winners), len(losers)) == (1, 7)

    async with stack.sql_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT revision FROM session_snapshots WHERE session_id = $1", sid
        )
    assert row is not None and int(row["revision"]) == 3

    # Creating a session that already exists is refused too (`INSERT ... DO NOTHING`).
    with pytest.raises(StaleSessionWriteError):
        await asyncio.to_thread(
            second_stack.session_store.save, SessionState(session_id=sid, agent_id="a")
        )


async def test_cached_head_never_regresses_and_cache_loss_reads_through(
    stack: ClusterRuntimeStack,
) -> None:
    """A stale reader cannot overwrite a newer cached head; a lost head is rebuilt from PostgreSQL."""
    sid = _uid("head")
    s1 = await asyncio.to_thread(
        stack.session_store.save, SessionState(session_id=sid, agent_id="a")
    )
    s2 = await asyncio.to_thread(stack.session_store.save, s1)
    head_key = f"sess:head:{{{sid}}}".encode()

    written = await stack.cache.run_script(
        VERSIONED_HEAD_SET_SCRIPT, [head_key], [1, s1.model_dump_json(), 60]
    )
    assert written == 0
    raw = await stack.cache.get(head_key)
    assert raw is not None and decode_versioned_head(raw)[0] == s2.revision

    client = await raw_valkey(stack)
    assert int(await client.ttl(head_key)) > 0  # cache class: evictable

    assert await stack.cache.delete(head_key) == 1
    stack.session_store.evict_local_turn_cache(sid)
    reloaded = await stack.session_store.prefetch_session(sid)
    stack.session_store.evict_local_turn_cache(sid)
    assert reloaded is not None and reloaded.revision == s2.revision
    raw_again = await stack.cache.get(head_key)
    assert raw_again is not None and decode_versioned_head(raw_again)[0] == s2.revision

    await asyncio.to_thread(stack.session_store.save_context_body, sid, "d" * 64, "body")
    blob_key = f"blob:{{{sid}}}:{'d' * 64}".encode()
    await stack.cache.delete(blob_key)
    body = await asyncio.to_thread(stack.session_store.load_context_body, sid, "d" * 64)
    assert body == "body"
    assert await stack.cache.get(blob_key) == b"body"


async def test_room_store_flushes_turns_started_and_refuses_stale_rooms(
    stack: ClusterRuntimeStack, second_stack: ClusterRuntimeStack
) -> None:
    """`turns_started` is in PostgreSQL as soon as `save` returns (#1366); stale saves are refused."""
    rid = _uid("room")
    room = await asyncio.to_thread(stack.room_store.save, RoomState(room_id=rid))
    bumped = room.model_copy(
        update={"file_record": room.file_record.model_copy(update={"turns_started": 1})}
    )
    saved = await asyncio.to_thread(stack.room_store.save, bumped)

    async with second_stack.sql_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT revision, turns_started FROM room_states WHERE room_id = $1", rid
        )
    assert row is not None
    assert (int(row["revision"]), int(row["turns_started"])) == (saved.revision, 1)

    with pytest.raises(StaleRoomWriteError):
        await asyncio.to_thread(second_stack.room_store.save, room)
    loaded = await asyncio.to_thread(second_stack.room_store.load, rid)
    assert loaded is not None and loaded.revision == saved.revision


async def test_lease_script_is_exclusive_fenced_and_honours_tombstones(
    stack: ClusterRuntimeStack, second_stack: ClusterRuntimeStack
) -> None:
    """One atomic script: tombstone check, exclusive acquire, monotonic fencing token."""
    conv = _uid("lease")
    leases_a, leases_b = stack.lease_manager, second_stack.lease_manager

    outcomes = await asyncio.gather(
        *(leases_a.acquire(conv, "t1", f"w{i}") for i in range(5)),
        *(leases_b.acquire(conv, "t1", f"v{i}") for i in range(5)),
        return_exceptions=True,
    )
    held = [o for o in outcomes if not isinstance(o, BaseException)]
    assert len(held) == 1
    assert all(isinstance(o, TurnLeaseConflictError) for o in outcomes if o not in held)
    token = held[0]

    # A stale holder (wrong fencing token) can neither renew nor release.
    stale = token.model_copy(update={"fencing_token": token.fencing_token + 100})
    assert await leases_b.renew(stale) is False
    await leases_b.release(stale)
    with pytest.raises(TurnLeaseConflictError):
        await leases_b.acquire(conv, "t1", "intruder")

    assert await leases_a.renew(token) is True
    await leases_a.release(token)
    successor = await leases_b.acquire(conv, "t2", "v-next")
    assert successor.fencing_token > token.fencing_token
    await leases_b.release(successor)

    await leases_a.write_cancel_tombstone(conv, "t3", ttl_seconds=60)
    assert await leases_b.is_cancelled(conv, "t3") is True
    with pytest.raises(TurnPreemptedError):
        await leases_b.acquire(conv, "t3", "v-cancelled")
    assert await leases_b.is_cancelled(conv, "t4") is False

    # Coordination keys carry no TTL, so volatile-lru can never evict them.
    client = await raw_valkey(stack)
    survivor = await leases_a.acquire(conv, "t4", "w-ttl")
    for key in (
        ValkeyTurnLeaseManager.lease_key(conv),
        ValkeyTurnLeaseManager.fence_key(conv),
        ValkeyTurnLeaseManager.cancel_key(conv),
    ):
        assert int(await client.ttl(key)) == -1, key
    await leases_a.release(survivor)


async def test_usage_is_one_window_across_workers_and_every_call_has_a_ledger_row(
    stack: ClusterRuntimeStack, second_stack: ClusterRuntimeStack
) -> None:
    """Two workers book into one window with no TTL; the ledger keeps the split and the turn."""
    scope = _uid("usage")
    sid = _uid("usage-sess")
    now = datetime.now(UTC)
    for owner, tokens in ((stack, 600), (second_stack, 500)):
        await owner.usage_store.record_completed_call(
            scope=scope,
            session_id=sid,
            turn_id="t1",
            entry=UsageEntry(
                at=now,
                tokens=tokens,
                provider="openai",
                model="m",
                input_tokens=tokens - 5,
                output_tokens=5,
                cached_tokens=2,
            ),
            counts_against_limits=True,
        )
    entries = await second_stack.usage_store.entries_since(scope, now - timedelta(minutes=5))
    assert sum(e.tokens for e in entries) == 1100
    assert sum(e.cached_tokens or 0 for e in entries) == 4

    client = await raw_valkey(stack)
    assert int(await client.ttl(stack.usage_store.window_key(scope))) == -1

    rows = await stack.usage_store.list_ledger_entries(sid)
    assert [(int(r["prompt_tokens"]), int(r["completion_tokens"])) for r in rows] == [
        (595, 5),
        (495, 5),
    ]
    assert all(r["scope"] == scope and r["turn_id"] == "t1" for r in rows)


async def test_a_lost_usage_window_reads_through_the_ledger_and_is_rebuilt(
    stack: ClusterRuntimeStack,
) -> None:
    scope = _uid("usage-lost")
    now = datetime.now(UTC)
    await stack.usage_store.record_completed_call(
        scope=scope,
        session_id=_uid("s"),
        turn_id="t1",
        entry=UsageEntry(at=now, tokens=700, provider="openai"),
        counts_against_limits=True,
    )
    await stack.usage_store.record_completed_call(
        scope=scope,
        session_id=_uid("s"),
        turn_id="t2",
        entry=UsageEntry(at=now, tokens=99, provider="ollama"),
        counts_against_limits=False,  # a local model: a row, never a window
    )
    client = await raw_valkey(stack)
    key = stack.usage_store.window_key(scope)
    since = now - timedelta(minutes=5)
    assert sum(e.tokens for e in await stack.usage_store.entries_since(scope, since)) == 700
    await client.delete(key)

    assert sum(e.tokens for e in await stack.usage_store.entries_since(scope, since)) == 700
    assert await client.hlen(key) >= 1  # rebuilt
    assert sum(e.tokens for e in await stack.usage_store.entries_since(scope, since)) == 700


async def test_lease_release_and_renew_are_scoped_to_the_turn(stack: ClusterRuntimeStack) -> None:
    """A failover can hand one worker the same fence for a later turn (#2468)."""
    conv = _uid("scoped")
    held = await stack.lease_manager.acquire(conv, "turn-new", "w1")
    old_turn = held.model_copy(update={"turn_id": "turn-old"})

    assert await stack.lease_manager.renew(old_turn) is False
    await stack.lease_manager.release(old_turn)
    still = await stack.lease_manager.current_holder(conv)
    assert still is not None and still.turn_id == "turn-new"

    assert await stack.lease_manager.renew(held) is True
    await stack.lease_manager.release(held)
    assert await stack.lease_manager.current_holder(conv) is None


async def test_budget_snapshot_writes_are_refused_past_their_revision(
    stack: ClusterRuntimeStack,
) -> None:
    sid = _uid("budget")
    first = await asyncio.to_thread(
        stack.session_store.save, SessionState(session_id=sid, agent_id="a")
    )
    newest = TokenBudgetSnapshot(consumed_prompt_tokens=10, turn_count=2)
    await asyncio.to_thread(
        stack.session_store.save_budget_snapshot, sid, newest, expected_revision=first.revision
    )
    second = await asyncio.to_thread(stack.session_store.save, first)
    assert second.revision == first.revision + 1

    stale = TokenBudgetSnapshot(consumed_prompt_tokens=1, turn_count=1)
    with pytest.raises(StaleSessionWriteError):
        await asyncio.to_thread(
            stack.session_store.save_budget_snapshot, sid, stale, expected_revision=first.revision
        )
    assert await asyncio.to_thread(stack.session_store.load_budget_snapshot, sid) == newest


async def test_knowledge_store_round_trips_through_postgres(stack: ClusterRuntimeStack) -> None:
    clone = _uid("clone")
    await stack.knowledge_store.upsert_triples(
        [
            KnowledgeTripleRecord(
                clone_id=clone, subject="user", predicate="prefers", object_val="Korean"
            ),
            KnowledgeTripleRecord(
                clone_id=clone,
                subject="user",
                predicate="prefers",
                object_val="Korean",
                confidence=0.5,
            ),
        ]
    )
    triples = await stack.knowledge_store.list_triples(clone)
    assert [(t.object_val, t.confidence) for t in triples] == [("Korean", 0.5)]


async def test_sse_travels_kafka_to_centrifugo_history_in_order(
    stack: ClusterRuntimeStack,
) -> None:
    """Events published by the bridge reach Centrifugo's Valkey-backed history through Kafka."""
    conv = _uid("sse")
    for step in range(5):
        await stack.sse_publisher.publish_event(
            conv,
            AgentEvent(type=EventType.AGENT_REPLY, sender_id="c", payload={"step": step}),
        )

    def _five(envs: tuple[object, ...]) -> bool:
        return len(envs) >= 5

    history = await stack.wait_for_history(f"ucx:session:{conv}", _five, timeout_seconds=30)
    assert sorted(int(e.data["seq"]) for e in history) == [1, 2, 3, 4, 5]
    client = await raw_valkey(stack)
    assert int(await client.ttl(KafkaSSEBridge.sequence_key(conv))) == -1


async def test_a_corrupt_cached_room_head_is_a_miss(stack: ClusterRuntimeStack) -> None:
    room_id = f"room-{uuid.uuid4().hex[:10]}"
    saved = await asyncio.to_thread(stack.room_store.save, RoomState(room_id=room_id))
    client = await raw_valkey(stack)
    key = f"room:head:{{{room_id}}}".encode()
    await client.set(key, b"\xff\xfe not a head")

    loaded = await asyncio.to_thread(stack.room_store.load, room_id)

    assert loaded is not None and loaded.revision == saved.revision
    assert (await client.get(key) or b"").startswith(f"{saved.revision}\n".encode())


async def test_a_corrupt_usage_window_is_a_miss_and_the_ledger_answers(
    stack: ClusterRuntimeStack,
) -> None:
    scope = f"user-{uuid.uuid4().hex[:10]}"
    client = await raw_valkey(stack)
    await client.hset(stack.usage_store.window_key(scope), mapping={"1": b"not a number"})

    entries = await stack.usage_store.entries_since(scope, datetime.now(UTC) - timedelta(hours=1))

    assert list(entries) == []


async def test_a_start_is_counted_and_refunded_and_never_goes_below_nothing(
    stack: ClusterRuntimeStack,
) -> None:
    sid = f"sess-{uuid.uuid4().hex[:10]}"
    store = stack.session_store
    await store.refund_turn_start(sid, "t")  # nothing to take back yet: writes nothing
    assert await store.record_turn_start(sid, "t") == 1
    assert await store.record_turn_start(sid, "t") == 2
    await store.refund_turn_start(sid, "t")
    assert await store.record_turn_start(sid, "t") == 2
    for _ in range(3):
        await store.refund_turn_start(sid, "t")
    assert await store.record_turn_start(sid, "t") == 1
