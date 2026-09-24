"""Unit tests for the Reactive Event Bus, AgentEvent models, and subscription mechanics."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    Provenance,
    ServiceRef,
)
from uclone_x.engine.event_bus import (
    AgentEvent,
    BackpressurePolicy,
    ErrorHandler,
    EventBus,
    EventBusError,
    EventPriority,
    EventSource,
    EventSubscription,
    EventType,
    PublisherHandle,
    QueueFullError,
    SubscriptionClosedError,
    UnauthorizedPublishError,
    UnauthorizedSubscriptionError,
    _drop_lowest_priority_from_queue,  # pyright: ignore[reportPrivateUsage]
    _drop_oldest_from_priority_queue,  # pyright: ignore[reportPrivateUsage]
)

# ==============================================================================
# 1. AgentEvent & EventPriority Model Tests
# ==============================================================================


def test_event_priority_values() -> None:
    """Validate standard priority values according to specification."""
    assert EventPriority.CRITICAL == 0
    assert EventPriority.INTERRUPT == 10
    assert EventPriority.NORMAL == 50
    assert EventPriority.BACKGROUND == 100
    assert (
        EventPriority.CRITICAL
        < EventPriority.INTERRUPT
        < EventPriority.NORMAL
        < EventPriority.BACKGROUND
    )


def test_agent_event_defaults_and_immutability() -> None:
    """Validate AgentEvent default values and immutability (frozen=True)."""
    event = AgentEvent(
        topic="agent.coder",
        type=EventType.USER_INPUT,
        payload={"message": "Build feature"},
    )

    assert event.event_id.startswith("evt_")
    assert event.topic == "agent.coder"
    assert event.type == "USER_INPUT"
    assert event.source == "system"
    assert event.priority == EventPriority.NORMAL
    assert event.sequence == 0
    assert event.payload == {"message": "Build feature"}
    assert event.schema_version == "1.0.0"
    assert not event.is_interrupt
    assert not event.is_critical

    # Verify frozen immutability
    with pytest.raises(ValidationError):
        # pyright: ignore[reportAttributeAccessIssue]
        event.topic = "new.topic"  # type: ignore[misc]


def test_agent_event_ordering() -> None:
    """Validate that AgentEvent instances are compared by priority then sequence."""
    e_critical = AgentEvent(
        type=EventType.USER_INPUT, event_id="e1", priority=EventPriority.CRITICAL, sequence=10
    )
    e_interrupt = AgentEvent(
        type=EventType.USER_INPUT, event_id="e2", priority=EventPriority.INTERRUPT, sequence=5
    )
    e_normal1 = AgentEvent(
        type=EventType.USER_INPUT, event_id="e3", priority=EventPriority.NORMAL, sequence=1
    )
    e_normal2 = AgentEvent(
        type=EventType.USER_INPUT, event_id="e4", priority=EventPriority.NORMAL, sequence=2
    )
    e_bg = AgentEvent(
        type=EventType.USER_INPUT, event_id="e5", priority=EventPriority.BACKGROUND, sequence=0
    )

    # Priority ordering (lower number = higher priority)
    assert e_critical < e_interrupt
    assert e_interrupt < e_normal1
    assert e_normal1 < e_normal2  # FIFO tie-break via sequence
    assert e_normal2 < e_bg

    assert e_critical.is_critical
    assert e_critical.is_interrupt
    assert e_interrupt.is_interrupt
    assert not e_normal1.is_interrupt


# ==============================================================================
# 2. Priority Dispatch Ordering Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_priority_queue_dispatch_order() -> None:
    """Verify that INTERRUPT and CRITICAL events are dispatched before NORMAL events."""
    async with EventBus() as bus:
        sub = bus.subscribe("task.run")

        # Publish in reverse priority order: NORMAL first, then INTERRUPT, then CRITICAL
        await bus.publish(
            AgentEvent(
                event_id="evt_norm1",
                topic="task.run",
                type=EventType.AGENT_REPLY,
                priority=EventPriority.NORMAL,
            )
        )
        await bus.publish(
            AgentEvent(
                event_id="evt_norm2",
                topic="task.run",
                type=EventType.AGENT_REPLY,
                priority=EventPriority.NORMAL,
            )
        )
        await bus.publish(
            AgentEvent(
                event_id="evt_intr",
                topic="task.run",
                type=EventType.INTERRUPT,
                priority=EventPriority.INTERRUPT,
            )
        )
        await bus.publish(
            AgentEvent(
                event_id="evt_crit",
                topic="task.run",
                type=EventType.INTERRUPT,
                priority=EventPriority.CRITICAL,
            )
        )
        await bus.publish(
            AgentEvent(
                event_id="evt_bg",
                topic="task.run",
                type=EventType.AGENT_REPLY,
                priority=EventPriority.BACKGROUND,
            )
        )

        await bus.wait_until_idle()

        # Collect events in priority order
        received: list[str] = []
        for _ in range(5):
            evt = await sub.get()
            received.append(evt.event_id)

        # Expected order: CRITICAL -> INTERRUPT -> NORMAL (FIFO) -> NORMAL (FIFO) -> BACKGROUND
        assert received == ["evt_crit", "evt_intr", "evt_norm1", "evt_norm2", "evt_bg"]


# ==============================================================================
# 3. Multi-Subscriber Pub/Sub Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_multi_subscriber_pub_sub() -> None:
    """Verify multiple subscribers receive events matching their topics."""
    async with EventBus() as bus:
        sub_all = bus.subscribe("*")
        sub_coder = bus.subscribe("agent.coder")
        sub_reviewer = bus.subscribe("agent.reviewer")
        sub_wildcard = bus.subscribe("agent.*")

        event_coder = AgentEvent(
            event_id="evt_c1",
            topic="agent.coder",
            type=EventType.TOOL_CALL,
            payload={"action": "write_code"},
        )
        event_reviewer = AgentEvent(
            event_id="evt_r1",
            topic="agent.reviewer",
            type=EventType.TOOL_CALL,
            payload={"action": "review_pr"},
        )

        await bus.publish(event_coder)
        await bus.publish(event_reviewer)
        await bus.wait_until_idle()

        # sub_all received both
        assert sub_all.qsize() == 2
        all_ids = [(await sub_all.get()).event_id, (await sub_all.get()).event_id]
        assert "evt_c1" in all_ids and "evt_r1" in all_ids

        # sub_coder received only coder event
        assert sub_coder.qsize() == 1
        assert (await sub_coder.get()).event_id == "evt_c1"

        # sub_reviewer received only reviewer event
        assert sub_reviewer.qsize() == 1
        assert (await sub_reviewer.get()).event_id == "evt_r1"

        # sub_wildcard received both coder and reviewer events
        assert sub_wildcard.qsize() == 2
        wildcard_ids = [(await sub_wildcard.get()).event_id, (await sub_wildcard.get()).event_id]
        assert "evt_c1" in wildcard_ids and "evt_r1" in wildcard_ids


@pytest.mark.asyncio
async def test_subscriber_unsubscribe() -> None:
    """Verify unsubscribing stops receiving further events."""
    async with EventBus() as bus:
        sub = bus.subscribe("events.channel")

        await bus.publish(
            AgentEvent(type=EventType.USER_INPUT, event_id="e1", topic="events.channel")
        )
        await bus.wait_until_idle()
        assert (await sub.get()).event_id == "e1"

        # Unsubscribe
        sub.unsubscribe()
        assert sub.is_closed

        # Subsequent publish should not route to sub
        await bus.publish(
            AgentEvent(type=EventType.USER_INPUT, event_id="e2", topic="events.channel")
        )
        await bus.wait_until_idle()

        with pytest.raises(SubscriptionClosedError):
            await sub.get()


@pytest.mark.asyncio
async def test_callback_subscription() -> None:
    """Verify callback-based subscriptions work for sync and async handlers."""
    async with EventBus() as bus:
        received_sync: list[str] = []
        received_async: list[str] = []

        def sync_cb(event: AgentEvent) -> None:
            received_sync.append(event.event_id)

        async def async_cb(event: AgentEvent) -> None:
            received_async.append(event.event_id)

        unsub_sync = bus.subscribe_callback("sync.topic", sync_cb)
        unsub_async = bus.subscribe_callback("async.*", async_cb)

        await bus.publish(AgentEvent(type=EventType.USER_INPUT, event_id="s1", topic="sync.topic"))
        await bus.publish(AgentEvent(type=EventType.USER_INPUT, event_id="a1", topic="async.test"))
        await bus.wait_until_idle()

        assert received_sync == ["s1"]
        assert received_async == ["a1"]

        # Unsubscribe callback
        unsub_sync()
        unsub_async()
        await bus.publish(AgentEvent(type=EventType.USER_INPUT, event_id="s2", topic="sync.topic"))
        await bus.publish(AgentEvent(type=EventType.USER_INPUT, event_id="a2", topic="async.test"))
        await bus.wait_until_idle()
        assert received_sync == ["s1"]
        assert received_async == ["a1"]


# ==============================================================================
# 4. Async Reactive Consumption & Iteration
# ==============================================================================


@pytest.mark.asyncio
async def test_async_iterator_consumption() -> None:
    """Verify reactive async for loop consumes events and terminates cleanly on close."""
    async with EventBus() as bus:
        sub = bus.subscribe("stream.data")
        consumed: list[str] = []

        async def consumer(s: EventSubscription) -> None:
            async for evt in s:
                consumed.append(evt.event_id)

        consumer_task = asyncio.create_task(consumer(sub))

        for i in range(5):
            await bus.publish(
                AgentEvent(type=EventType.USER_INPUT, event_id=f"evt_{i}", topic="stream.data")
            )

        await bus.wait_until_idle()
        # Give consumer loop time to receive
        await asyncio.sleep(0.01)

        sub.close()
        await consumer_task

        assert consumed == ["evt_0", "evt_1", "evt_2", "evt_3", "evt_4"]


@pytest.mark.asyncio
async def test_subscription_context_manager() -> None:
    """Verify subscription async context manager auto-closes on block exit."""
    async with EventBus() as bus:
        async with bus.subscribe("ctx.topic") as sub:
            await bus.publish(
                AgentEvent(type=EventType.USER_INPUT, event_id="ctx_1", topic="ctx.topic")
            )
            await bus.wait_until_idle()
            evt = await sub.get()
            assert evt.event_id == "ctx_1"

        assert sub.is_closed


# ==============================================================================
# 5. Backpressure Policies & Overflow Handling
# ==============================================================================


@pytest.mark.asyncio
async def test_backpressure_error_policy() -> None:
    """Verify QueueFullError is raised when maxsize is exceeded under ERROR policy."""
    bus = EventBus(maxsize=2, backpressure_policy=BackpressurePolicy.ERROR)
    # Publish without starting dispatcher so queue fills
    bus.publish_nowait(AgentEvent(type=EventType.USER_INPUT, event_id="e1"))
    bus.publish_nowait(AgentEvent(type=EventType.USER_INPUT, event_id="e2"))

    assert bus.full()

    with pytest.raises(QueueFullError):
        bus.publish_nowait(AgentEvent(type=EventType.USER_INPUT, event_id="e3"))

    with pytest.raises(QueueFullError):
        await bus.publish(AgentEvent(type=EventType.USER_INPUT, event_id="e4"))


@pytest.mark.asyncio
async def test_backpressure_drop_oldest_policy() -> None:
    """Verify oldest event is dropped when maxsize is exceeded under DROP_OLDEST policy."""
    bus = EventBus(maxsize=2, backpressure_policy=BackpressurePolicy.DROP_OLDEST)
    bus.publish_nowait(
        AgentEvent(type=EventType.USER_INPUT, event_id="e1", priority=EventPriority.NORMAL)
    )
    bus.publish_nowait(
        AgentEvent(type=EventType.USER_INPUT, event_id="e2", priority=EventPriority.NORMAL)
    )
    # Overflow with e3 -> should drop e1
    bus.publish_nowait(
        AgentEvent(type=EventType.USER_INPUT, event_id="e3", priority=EventPriority.NORMAL)
    )

    assert bus.qsize() == 2
    # e2 and e3 remain
    assert not bus.empty()


@pytest.mark.asyncio
async def test_backpressure_drop_incoming_policy() -> None:
    """Verify incoming event is discarded when full under DROP_INCOMING policy."""
    bus = EventBus(maxsize=2, backpressure_policy=BackpressurePolicy.DROP_INCOMING)
    bus.publish_nowait(AgentEvent(type=EventType.USER_INPUT, event_id="e1"))
    bus.publish_nowait(AgentEvent(type=EventType.USER_INPUT, event_id="e2"))
    # e3 should be silently dropped
    bus.publish_nowait(AgentEvent(type=EventType.USER_INPUT, event_id="e3"))

    assert bus.qsize() == 2


@pytest.mark.asyncio
async def test_backpressure_block_policy() -> None:
    """Verify BLOCK policy awaits until consumer frees space."""
    async with EventBus(maxsize=1, backpressure_policy=BackpressurePolicy.BLOCK) as bus:
        sub = bus.subscribe("*", maxsize=10)

        # Publish 1st item
        await bus.publish(AgentEvent(type=EventType.USER_INPUT, event_id="e1"))

        # Publish 2nd item in background (will block until dispatcher consumes 1st)
        task = asyncio.create_task(
            bus.publish(AgentEvent(type=EventType.USER_INPUT, event_id="e2"))
        )
        await bus.wait_until_idle()
        await task

        assert (await sub.get()).event_id == "e1"
        assert (await sub.get()).event_id == "e2"


# ==============================================================================
# 6. Lifecycle & Deadlock-Free Cancellation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_bus_stop_cancels_cleanly() -> None:
    """Verify stopping the bus closes subscriptions and shuts down dispatcher."""
    bus = EventBus()
    await bus.start()
    assert bus.is_running

    sub = bus.subscribe("any.topic")
    await bus.stop()

    assert not bus.is_running
    assert sub.is_closed

    with pytest.raises(EventBusError):
        await bus.publish(AgentEvent(type=EventType.USER_INPUT, event_id="e_stopped"))


@pytest.mark.asyncio
async def test_waiting_reader_unblocks_on_sub_close() -> None:
    """Verify an active awaiting get() immediately unblocks with error when subscription closes."""
    async with EventBus() as bus:
        sub = bus.subscribe("waiting.topic")

        reader_error: Any = None

        async def reader() -> None:
            nonlocal reader_error
            try:
                await sub.get()
            except SubscriptionClosedError as exc:
                reader_error = exc

        reader_task = asyncio.create_task(reader())
        await asyncio.sleep(0.01)  # Ensure reader is blocked awaiting

        sub.close()
        await reader_task

        assert isinstance(reader_error, SubscriptionClosedError)


# ==============================================================================
# 6. Envelope contract: strictness, immutability and total ordering
# ==============================================================================


def test_delivered_payload_cannot_be_mutated() -> None:
    """P2/A2A 10.3: an event shared by reference must not become shared mutable state, including nested structures (Issue #31)."""
    event = AgentEvent(
        type=EventType.USER_INPUT,
        payload={"plan": {"step": 1}, "items": [1, 2]},
    )

    with pytest.raises(TypeError):
        event.payload["top"] = "tampered"  # pyright: ignore[reportIndexIssue]

    plan: Any = event.payload["plan"]
    with pytest.raises(TypeError):
        plan["step"] = 999

    items: Any = event.payload["items"]
    with pytest.raises(AttributeError):
        items.append(3)

    assert plan["step"] == 1
    assert event.payload["items"] == (1, 2)


def test_payload_copies_the_caller_mapping() -> None:
    """A publisher must not retain a writable reference to what it published, including nested structures (Issue #31)."""
    nested: dict[str, Any] = {"step": 1}
    items: list[int] = [1, 2]
    source: dict[str, Any] = {"plan": nested, "items": items}
    event = AgentEvent(type=EventType.USER_INPUT, payload=source)

    nested["step"] = 999
    items.append(3)
    source["extra"] = "new"

    plan: Any = event.payload["plan"]
    assert plan["step"] == 1
    assert event.payload["items"] == (1, 2)
    assert "extra" not in event.payload


def test_model_copy_enforces_validation_and_forbid_rules() -> None:
    """Issue #31: model_copy(update=...) strictly validates and enforces extra='forbid', strict=True and deep freeze."""
    event = AgentEvent(type=EventType.USER_INPUT, payload={"a": 1})

    # 1. Reject undeclared extra attribute
    with pytest.raises(ValidationError):
        event.model_copy(update={"bogus": "x"})

    # 2. Reject invalid type under strict=True
    with pytest.raises(ValidationError):
        event.model_copy(update={"schema_version": 99})

    # 3. Preserves deep immutability on updated payload
    updated = event.model_copy(update={"payload": {"plan": {"step": 2}}})
    plan: Any = updated.payload["plan"]
    with pytest.raises(TypeError):
        plan["step"] = 999


def test_envelope_rejects_undeclared_fields() -> None:
    """`extra="forbid"` is what makes `schema_version` meaningful."""
    with pytest.raises(ValidationError):
        AgentEvent.model_validate({"type": "USER_INPUT", "unknown_field": 1})


def test_envelope_requires_a_type() -> None:
    """A defaulted type let an unclassified event pass as user input."""
    with pytest.raises(ValidationError, match="type"):
        AgentEvent.model_validate({"topic": "agent.coder"})


def test_envelope_rejects_an_unknown_type() -> None:
    with pytest.raises(ValidationError):
        AgentEvent.model_validate({"type": "NOT_A_REAL_EVENT_TYPE"})


def test_envelope_survives_a_json_round_trip() -> None:
    """An immutable payload must still serialise: A2A and OTel both need it."""
    event = AgentEvent(
        type=EventType.TOOL_RESULT,
        source=EventSource.TOOL,
        payload={"output": "ok", "count": 2},
    )

    restored = AgentEvent.model_validate_json(event.model_dump_json())

    assert restored.payload == {"output": "ok", "count": 2}
    assert restored.type is EventType.TOOL_RESULT
    assert restored.source is EventSource.TOOL


def test_ordering_is_total_for_events_sharing_priority_and_sequence() -> None:
    """Without the event_id tiebreaker every comparison is False and the heap wobbles."""
    first = AgentEvent(
        event_id="evt_a", type=EventType.USER_INPUT, priority=EventPriority.NORMAL, sequence=7
    )
    second = AgentEvent(
        event_id="evt_b", type=EventType.USER_INPUT, priority=EventPriority.NORMAL, sequence=7
    )

    assert first < second
    assert second > first
    assert first != second
    assert sorted([second, first])[0] is first


async def test_a_forged_close_sentinel_does_not_close_a_reader() -> None:
    """The reader detects close by identity: `type` is publishable by any component."""
    async with EventBus() as bus:
        sub = bus.subscribe("forge.test")
        await bus.publish(
            AgentEvent(
                type=EventType.SUBSCRIPTION_CLOSED,
                topic="forge.test",
                payload={"forged": True},
            )
        )
        await bus.wait_until_idle()

        received = await asyncio.wait_for(sub.get(), timeout=1.0)

        assert received.payload["forged"] is True
        assert sub.is_closed is False
        sub.close()


@pytest.mark.asyncio
async def test_multiple_concurrent_readers_wake_on_subscription_close() -> None:
    """Issue #33: Subscription.close() wakes all concurrent waiting readers, none stay hung."""
    async with EventBus() as bus:
        sub = bus.subscribe("multi.reader.test")

        results: list[str] = []

        async def reader(reader_id: str) -> None:
            try:
                await sub.get()
                results.append(f"{reader_id}:got_event")
            except SubscriptionClosedError:
                results.append(f"{reader_id}:closed")

        # Launch 3 concurrent tasks waiting on sub.get()
        task1 = asyncio.create_task(reader("reader1"))
        task2 = asyncio.create_task(reader("reader2"))
        task3 = asyncio.create_task(reader("reader3"))

        # Give tasks time to enter await sub.get()
        await asyncio.sleep(0.02)

        # Close the subscription
        sub.close()

        # All 3 readers must complete without hanging
        await asyncio.wait_for(asyncio.gather(task1, task2, task3), timeout=1.0)

        assert sorted(results) == ["reader1:closed", "reader2:closed", "reader3:closed"]


# ==============================================================================
# 7. Issue #17 Fix Verification: Backpressure, Isolation, and Lifecycle Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_drop_lowest_priority_mixed_priorities_preserves_critical() -> None:
    """Issue 2026-09-02-037: DROP_LOWEST_PRIORITY drops LOW/NORMAL priority and preserves CRITICAL."""
    async with EventBus() as bus:
        sub = bus.subscribe(
            "priority.test", maxsize=2, backpressure_policy=BackpressurePolicy.DROP_LOWEST_PRIORITY
        )

        # Publish 1 CRITICAL and 1 BACKGROUND event (queue is now full at maxsize=2)
        await bus.publish(
            AgentEvent(
                event_id="evt_crit1",
                topic="priority.test",
                type=EventType.INTERRUPT,
                priority=EventPriority.CRITICAL,
            )
        )
        await bus.publish(
            AgentEvent(
                event_id="evt_bg1",
                topic="priority.test",
                type=EventType.AGENT_REPLY,
                priority=EventPriority.BACKGROUND,
            )
        )
        await bus.wait_until_idle()
        assert sub.qsize() == 2

        # Publish a 2nd CRITICAL event -> BACKGROUND must be dropped, both CRITICAL preserved
        await bus.publish(
            AgentEvent(
                event_id="evt_crit2",
                topic="priority.test",
                type=EventType.INTERRUPT,
                priority=EventPriority.CRITICAL,
            )
        )
        await bus.wait_until_idle()

        assert sub.qsize() == 2
        r1 = await sub.get()
        r2 = await sub.get()

        received_ids = {r1.event_id, r2.event_id}
        assert received_ids == {"evt_crit1", "evt_crit2"}
        assert "evt_bg1" not in received_ids


def test_drop_lowest_priority_from_queue_heap_eviction() -> None:
    """Issue 2026-09-02-037: Helper _drop_lowest_priority_from_queue correctly pops max sort key."""
    q: asyncio.PriorityQueue[AgentEvent] = asyncio.PriorityQueue(maxsize=10)

    # Empty queue returns None
    assert _drop_lowest_priority_from_queue(q) is None

    e_crit = AgentEvent(
        event_id="e_crit", type=EventType.USER_INPUT, priority=EventPriority.CRITICAL, sequence=1
    )
    e_intr = AgentEvent(
        event_id="e_intr", type=EventType.USER_INPUT, priority=EventPriority.INTERRUPT, sequence=2
    )
    e_norm = AgentEvent(
        event_id="e_norm", type=EventType.USER_INPUT, priority=EventPriority.NORMAL, sequence=3
    )
    e_bg = AgentEvent(
        event_id="e_bg", type=EventType.USER_INPUT, priority=EventPriority.BACKGROUND, sequence=4
    )

    for e in (e_crit, e_intr, e_norm, e_bg):
        q.put_nowait(e)

    assert q.qsize() == 4

    # 1. Evicts BACKGROUND (priority=100)
    d1 = _drop_lowest_priority_from_queue(q)
    assert d1 is not None and d1.event_id == "e_bg"
    assert q.qsize() == 3

    # 2. Evicts NORMAL (priority=50)
    d2 = _drop_lowest_priority_from_queue(q)
    assert d2 is not None and d2.event_id == "e_norm"
    assert q.qsize() == 2

    # 3. Evicts INTERRUPT (priority=10)
    d3 = _drop_lowest_priority_from_queue(q)
    assert d3 is not None and d3.event_id == "e_intr"
    assert q.qsize() == 1

    # 4. Remaining item is CRITICAL (priority=0)
    remaining = q.get_nowait()
    assert remaining.event_id == "e_crit"
    assert q.empty()


def test_backpressure_policy_backward_compatibility() -> None:
    """Verify DROP_OLDEST is a backward-compatible alias of DROP_LOWEST_PRIORITY."""
    assert BackpressurePolicy.DROP_OLDEST == BackpressurePolicy.DROP_LOWEST_PRIORITY
    assert BackpressurePolicy("drop_oldest") is BackpressurePolicy.DROP_LOWEST_PRIORITY
    assert BackpressurePolicy("drop_lowest_priority") is BackpressurePolicy.DROP_LOWEST_PRIORITY
    assert _drop_oldest_from_priority_queue is _drop_lowest_priority_from_queue


@pytest.mark.asyncio
async def test_slow_blocked_subscriber_does_not_stall_other_subscribers() -> None:
    """Issue 2026-09-02-038: A blocked subscriber under BLOCK policy does not stall other subscribers."""
    async with EventBus() as bus:
        # sub_blocked has capacity 1 and uses BLOCK policy
        sub_blocked = bus.subscribe(
            "shared.topic", maxsize=1, backpressure_policy=BackpressurePolicy.BLOCK
        )
        # sub_active has capacity 10 and uses ERROR policy
        sub_active = bus.subscribe(
            "shared.topic", maxsize=10, backpressure_policy=BackpressurePolicy.ERROR
        )

        # Fill sub_blocked
        await bus.publish(
            AgentEvent(event_id="evt_0", topic="shared.topic", type=EventType.USER_INPUT)
        )
        await bus.wait_until_idle()
        assert sub_blocked.qsize() == 1
        assert sub_active.qsize() == 1

        # Now sub_blocked is full. Publish 3 more events.
        # Even though sub_blocked cannot accept new events immediately,
        # sub_active MUST receive all 3 events without hanging or blocking!
        for i in range(1, 4):
            await bus.publish(
                AgentEvent(event_id=f"evt_{i}", topic="shared.topic", type=EventType.USER_INPUT)
            )

        await bus.wait_until_idle()

        # sub_active received all 4 events (evt_0, evt_1, evt_2, evt_3)
        assert sub_active.qsize() == 4
        active_ids = [(await sub_active.get()).event_id for _ in range(4)]
        assert active_ids == ["evt_0", "evt_1", "evt_2", "evt_3"]


@pytest.mark.asyncio
async def test_wait_until_idle_after_stop_returns_promptly() -> None:
    """Issue 2026-09-02-038: wait_until_idle() after stop() returns immediately and never hangs."""
    bus = EventBus()
    await bus.start()

    # Publish events
    for i in range(5):
        await bus.publish(
            AgentEvent(event_id=f"e_{i}", topic="any.topic", type=EventType.USER_INPUT)
        )

    await bus.stop()

    # wait_until_idle and join must return immediately without timeout
    await asyncio.wait_for(bus.wait_until_idle(), timeout=1.0)
    await asyncio.wait_for(bus.join(), timeout=1.0)


@pytest.mark.asyncio
async def test_subscriber_error_policy_observability() -> None:
    """Issue 2026-09-02-040: Subscription ERROR policy records observable QueueFullError."""
    handled_errors: list[QueueFullError] = []

    def _handler(exc: QueueFullError, event: AgentEvent, sub: EventSubscription | None) -> None:
        handled_errors.append(exc)

    handler: ErrorHandler = _handler

    async with EventBus() as bus:
        bus.set_error_handler(handler)
        sub = bus.subscribe("err.topic", maxsize=1, backpressure_policy=BackpressurePolicy.ERROR)

        # 1st event fits in sub queue
        await bus.publish(AgentEvent(event_id="e1", topic="err.topic", type=EventType.USER_INPUT))
        await bus.wait_until_idle()
        assert sub.qsize() == 1
        assert bus.error_count == 0

        # 2nd event overflows sub queue -> raises QueueFullError in delivery
        await bus.publish(AgentEvent(event_id="e2", topic="err.topic", type=EventType.USER_INPUT))
        await bus.wait_until_idle()
        await asyncio.sleep(0.01)

        # Observable on subscription
        assert sub.error_count == 1
        assert isinstance(sub.last_error, QueueFullError)
        assert len(sub.errors) == 1

        # Observable on bus
        assert bus.error_count == 1
        assert isinstance(bus.last_error, QueueFullError)
        assert len(bus.errors) == 1

        # Error handler callback was invoked
        assert len(handled_errors) == 1

        # Clear errors on bus
        bus.clear_errors()
        assert bus.error_count == 0
        assert bus.last_error is None


@pytest.mark.asyncio
async def test_subscription_deliver_nowait_policies() -> None:
    """Verify direct deliver_nowait under various backpressure policies."""
    bus = EventBus()
    sub_err = EventSubscription(
        bus=bus, topics={"test"}, maxsize=1, backpressure_policy=BackpressurePolicy.ERROR
    )
    sub_drop = EventSubscription(
        bus=bus,
        topics={"test"},
        maxsize=1,
        backpressure_policy=BackpressurePolicy.DROP_LOWEST_PRIORITY,
    )
    sub_inc = EventSubscription(
        bus=bus, topics={"test"}, maxsize=1, backpressure_policy=BackpressurePolicy.DROP_INCOMING
    )
    sub_blk = EventSubscription(
        bus=bus, topics={"test"}, maxsize=1, backpressure_policy=BackpressurePolicy.BLOCK
    )

    # Fill all 1-slot subscriptions
    for s in (sub_err, sub_drop, sub_inc, sub_blk):
        s.deliver_nowait(AgentEvent(event_id="base", topic="test", type=EventType.USER_INPUT))
        assert s.full()

    # ERROR raises QueueFullError
    with pytest.raises(QueueFullError):
        sub_err.deliver_nowait(
            AgentEvent(event_id="overflow", topic="test", type=EventType.USER_INPUT)
        )

    # BLOCK nowait raises QueueFullError
    with pytest.raises(QueueFullError):
        sub_blk.deliver_nowait(
            AgentEvent(event_id="overflow", topic="test", type=EventType.USER_INPUT)
        )

    # DROP_LOWEST_PRIORITY succeeds
    sub_drop.deliver_nowait(
        AgentEvent(
            event_id="overflow_crit",
            topic="test",
            type=EventType.USER_INPUT,
            priority=EventPriority.CRITICAL,
        )
    )
    assert sub_drop.qsize() == 1
    assert sub_drop.get_nowait().event_id == "overflow_crit"

    # DROP_INCOMING succeeds without altering existing
    sub_inc.deliver_nowait(AgentEvent(event_id="overflow", topic="test", type=EventType.USER_INPUT))
    assert sub_inc.qsize() == 1
    assert sub_inc.get_nowait().event_id == "base"


@pytest.mark.asyncio
async def test_async_callback_does_not_stall_dispatch_loop() -> None:
    """Issue 2026-09-02-038: Slow async callback does not block the dispatch loop for subsequent events."""
    async with EventBus() as bus:
        sub = bus.subscribe("async.cb.topic")
        callback_finished: list[str] = []

        async def slow_callback(event: AgentEvent) -> None:
            await asyncio.sleep(0.05)
            callback_finished.append(event.event_id)

        bus.subscribe_callback("async.cb.topic", slow_callback)

        await bus.publish(
            AgentEvent(event_id="e1", topic="async.cb.topic", type=EventType.USER_INPUT)
        )
        await bus.publish(
            AgentEvent(event_id="e2", topic="async.cb.topic", type=EventType.USER_INPUT)
        )

        await bus.wait_until_idle()

        # Subscriber receives events immediately before slow callback finishes
        evt1 = await sub.get()
        evt2 = await sub.get()
        assert evt1.event_id == "e1"
        assert evt2.event_id == "e2"

        # Allow slow callback task to finish
        await asyncio.sleep(0.08)
        assert callback_finished == ["e1", "e2"]


# ==============================================================================
# 9. Issue #18 (2026-09-02-039): Event Bus Authority & Identity Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_unverified_user_origin_spoofing_rejected() -> None:
    """Unverified publishers cannot publish events claiming source='user' (Issue #18, 2026-09-02-039)."""
    async with EventBus() as bus:
        # 1. Async publish without publisher handle claiming source=USER
        with pytest.raises(UnauthorizedPublishError, match="Unverified publisher"):
            await bus.publish(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    source=EventSource.USER,
                    topic="system.approval",
                    payload={"approved": True},
                )
            )

        # 2. Synchronous publish_nowait without publisher handle claiming source=USER
        with pytest.raises(UnauthorizedPublishError, match="Unverified publisher"):
            bus.publish_nowait(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    source=EventSource.USER,
                    topic="system.approval",
                    payload={"approved": True},
                )
            )


@pytest.mark.asyncio
async def test_publisher_handle_authoritatively_stamps_identity() -> None:
    """The bus authoritatively stamps source and sender_id from registered publisher handle, preventing spoofing."""
    async with EventBus() as bus:
        sub = bus.subscribe("agent.task")
        agent_pub = bus.register_publisher(
            sender_id="agent_worker_1",
            source=EventSource.AGENT,
        )

        assert isinstance(agent_pub, PublisherHandle)
        assert agent_pub.sender_id == "agent_worker_1"
        assert agent_pub.source == EventSource.AGENT

        # Caller attempts to forge source=USER and a fake sender_id
        forged_event = AgentEvent(
            type=EventType.USER_INPUT,
            source=EventSource.USER,
            sender_id="spoofed_admin",
            topic="agent.task",
            payload={"action": "reboot"},
        )

        # Published via agent publisher handle
        published = await agent_pub.publish(forged_event)
        await bus.wait_until_idle()

        # The bus authoritatively overrides source to AGENT and sender_id to agent_worker_1
        assert published.source == EventSource.AGENT
        assert published.sender_id == "agent_worker_1"
        assert published.sequence > 0

        received = await sub.get()
        assert received.source == EventSource.AGENT
        assert received.sender_id == "agent_worker_1"
        assert received.payload["action"] == "reboot"


@pytest.mark.asyncio
async def test_authorized_user_publisher_can_publish_user_source() -> None:
    """Registered USER publisher handle is authorized to publish events with source='user'."""
    async with EventBus() as bus:
        sub = bus.subscribe("agent.chat")
        user_pub = bus.register_publisher(
            sender_id="console_user",
            source=EventSource.USER,
            capabilities={"user_source"},
        )

        event = AgentEvent(
            type=EventType.USER_INPUT,
            topic="agent.chat",
            payload={"message": "Deploy to staging"},
        )

        published = await user_pub.publish(event)
        await bus.wait_until_idle()

        assert published.source == EventSource.USER
        assert published.sender_id == "console_user"

        received = await sub.get()
        assert received.source == EventSource.USER
        assert received.sender_id == "console_user"
        assert received.payload["message"] == "Deploy to staging"


@pytest.mark.asyncio
async def test_publisher_allowed_topics_enforcement() -> None:
    """Publisher handle enforces allowed_topics restrictions."""
    async with EventBus() as bus:
        tool_pub = bus.register_publisher(
            sender_id="tool_runner",
            source=EventSource.TOOL,
            allowed_topics=["tool.result.*", "agent.notifications"],
        )

        # Authorized topic matches glob
        evt_valid = await tool_pub.publish(
            AgentEvent(
                type=EventType.TOOL_RESULT,
                topic="tool.result.grep",
                payload={"matches": 5},
            )
        )
        assert evt_valid.topic == "tool.result.grep"
        assert evt_valid.source == EventSource.TOOL
        assert evt_valid.sender_id == "tool_runner"

        # Unauthorized topic raises UnauthorizedPublishError
        with pytest.raises(UnauthorizedPublishError, match="not authorized to publish to topic"):
            await tool_pub.publish(
                AgentEvent(
                    type=EventType.TOOL_RESULT,
                    topic="system.admin.execute",
                    payload={"cmd": "rm -rf"},
                )
            )

        with pytest.raises(UnauthorizedPublishError, match="not authorized to publish to topic"):
            tool_pub.publish_nowait(
                AgentEvent(
                    type=EventType.TOOL_RESULT,
                    topic="system.admin.execute",
                    payload={"cmd": "rm -rf"},
                )
            )


@pytest.mark.asyncio
async def test_wildcard_subscription_auth_required() -> None:
    """Wildcard subscriptions require 'wildcard_subscribe' capability when require_wildcard_auth is enabled."""
    bus = EventBus(require_wildcard_auth=True)
    await bus.start()

    try:
        # Subscribing to full wildcard without capability raises UnauthorizedSubscriptionError
        with pytest.raises(UnauthorizedSubscriptionError, match="requires 'wildcard_subscribe'"):
            bus.subscribe("*")

        # Subscribing to pattern wildcard without capability raises UnauthorizedSubscriptionError
        with pytest.raises(UnauthorizedSubscriptionError, match="requires 'wildcard_subscribe'"):
            bus.subscribe("agent.*")

        # Subscribing with capability succeeds
        sub_auth = bus.subscribe("*", capabilities={"wildcard_subscribe"})
        assert not sub_auth.is_closed
        assert "*" in sub_auth.topics

        # Subscribing to exact topic without wildcards succeeds even without capability
        sub_exact = bus.subscribe("agent.coder")
        assert "agent.coder" in sub_exact.topics
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_topic_allowlist_enforcement() -> None:
    """Bus with topic_allowlist rejects subscriptions outside the allowlist."""
    bus = EventBus(topic_allowlist=["agent.*", "session.*", "system.health"])
    await bus.start()

    try:
        # Allowed subscriptions
        sub1 = bus.subscribe("agent.coder")
        sub2 = bus.subscribe("session.42")
        sub3 = bus.subscribe("system.health")
        assert not sub1.is_closed
        assert not sub2.is_closed
        assert not sub3.is_closed

        # Disallowed subscription
        with pytest.raises(UnauthorizedSubscriptionError, match="not allowed by topic allowlist"):
            bus.subscribe("credentials.vault")

        with pytest.raises(UnauthorizedSubscriptionError, match="not allowed by topic allowlist"):
            bus.subscribe_callback("credentials.vault", lambda e: None)
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_addressing_recipient_and_session_routing() -> None:
    """Events addressed to specific recipient_id or session_id are routed accurately to matching subscribers."""
    async with EventBus() as bus:
        sub_agent1 = bus.subscribe("chat.topic", recipient_id="agent_1")
        sub_agent2 = bus.subscribe("chat.topic", recipient_id="agent_2")
        sub_broadcast = bus.subscribe("chat.topic")  # Receives all on this topic

        # 1. Event addressed to agent_1
        await bus.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                topic="chat.topic",
                recipient_id="agent_1",
                payload={"msg": "For agent 1 only"},
            )
        )
        await bus.wait_until_idle()

        assert sub_agent1.qsize() == 1
        assert sub_agent2.qsize() == 0
        assert sub_broadcast.qsize() == 1

        evt_a1 = await sub_agent1.get()
        assert evt_a1.recipient_id == "agent_1"
        assert evt_a1.payload["msg"] == "For agent 1 only"

        evt_b1 = await sub_broadcast.get()
        assert evt_b1.recipient_id == "agent_1"

        # 2. Broadcast event (recipient_id="")
        await bus.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                topic="chat.topic",
                recipient_id="",
                payload={"msg": "For everyone"},
            )
        )
        await bus.wait_until_idle()

        assert sub_agent1.qsize() == 1
        assert sub_agent2.qsize() == 1
        assert sub_broadcast.qsize() == 1

        assert (await sub_agent1.get()).payload["msg"] == "For everyone"
        assert (await sub_agent2.get()).payload["msg"] == "For everyone"
        assert (await sub_broadcast.get()).payload["msg"] == "For everyone"


@pytest.mark.asyncio
async def test_session_id_addressing_routing() -> None:
    """Events with session_id are filtered to matching session subscriptions."""
    async with EventBus() as bus:
        sub_sess1 = bus.subscribe("workflow.step", session_id="sess_alpha")
        sub_sess2 = bus.subscribe("workflow.step", session_id="sess_beta")

        await bus.publish(
            AgentEvent(
                type=EventType.AGENT_REPLY,
                topic="workflow.step",
                session_id="sess_alpha",
                payload={"step": 1},
            )
        )
        await bus.wait_until_idle()

        assert sub_sess1.qsize() == 1
        assert sub_sess2.qsize() == 0

        evt = await sub_sess1.get()
        assert evt.session_id == "sess_alpha"


def test_create_response_preserves_correlation_and_trace_across_hops() -> None:
    """AgentEvent.create_response derives causally-linked events preserving correlation_id and trace_id across hops."""
    root_event = AgentEvent(
        event_id="evt_root_100",
        type=EventType.USER_INPUT,
        topic="agent.ingress",
        sender_id="user_alice",
        session_id="sess_999",
        trace_id="trace_otel_abc",
        payload={"query": "Calculate total"},
    )

    # Hop 1: Agent creates AGENT_REPLY responding to root_event
    reply = root_event.create_response(
        type=EventType.AGENT_REPLY,
        topic="agent.egress",
        payload={"total": 42},
    )

    assert reply.type == EventType.AGENT_REPLY
    assert reply.correlation_id == "evt_root_100"
    assert reply.trace_id == "trace_otel_abc"
    assert reply.session_id == "sess_999"
    assert reply.recipient_id == "user_alice"
    assert reply.payload["total"] == 42
    assert reply.event_id != root_event.event_id

    # Hop 2: Downstream component creates sub-action from reply
    sub_action = reply.create_response(
        type=EventType.TOOL_CALL,
        topic="tool.exec",
        payload={"func": "format"},
    )

    # correlation_id must point back to root_event (evt_root_100) across hops
    assert sub_action.correlation_id == "evt_root_100"
    assert sub_action.trace_id == "trace_otel_abc"
    assert sub_action.session_id == "sess_999"


# ==============================================================================
# Typed provenance on the envelope (#117, decision #51 Option B)
# ==============================================================================


def _failover_provenance() -> Provenance:
    return Provenance(
        path=ExecutionPath.FAILOVER,
        requested=ServiceRef(provider="primary_llm", model="p-1"),
        served_by=ServiceRef(provider="secondary_llm", model="s-1"),
        attempts=(AttemptRecord(provider="primary_llm", error_class="RateLimitError"),),
    )


def test_provenance_is_a_typed_optional_envelope_field() -> None:
    """`AgentEvent.provenance` is typed `Provenance | None` and defaults to absent."""
    assert "provenance" in AgentEvent.model_fields

    # The accepted trade of Option B: every event declares the field, and the ones with
    # no meaningful provenance carry `None` — "not stated", never "nothing went wrong".
    assert AgentEvent(type=EventType.INTERRUPT).provenance is None

    stamped = AgentEvent(type=EventType.AGENT_REPLY, provenance=_failover_provenance())
    assert isinstance(stamped.provenance, Provenance)
    assert stamped.provenance.path is ExecutionPath.FAILOVER

    # Strict typing still applies. A well-formed Python-mode mapping is accepted and
    # *coerced* to `Provenance` (which is what makes `with_sequence`'s dump-and-revalidate
    # work); a non-conformant one is a `ValidationError` located inside `provenance`
    # rather than a shape that gets carried through untyped.
    coerced = AgentEvent(
        type=EventType.AGENT_REPLY,
        provenance=Provenance.primary("anthropic", "claude").model_dump(),  # type: ignore[arg-type]
    )
    assert isinstance(coerced.provenance, Provenance)
    assert coerced.provenance.path is ExecutionPath.PRIMARY

    with pytest.raises(ValidationError) as excinfo:
        AgentEvent(type=EventType.AGENT_REPLY, provenance={"path": "primary"})  # type: ignore[arg-type]
    assert {err["loc"] for err in excinfo.value.errors()} == {
        ("provenance", "path"),
        ("provenance", "requested"),
        ("provenance", "served_by"),
    }


def test_provenance_survives_bus_sequence_stamping() -> None:
    """`with_sequence` / `model_copy` re-validate the whole envelope; provenance is preserved."""
    provenance = _failover_provenance()
    event = AgentEvent(type=EventType.AGENT_REPLY, provenance=provenance)

    stamped = event.with_sequence(17)
    assert stamped.sequence == 17
    assert stamped.provenance == provenance
    assert stamped.provenance is not None
    # Recomputed, not carried through the dump: `degraded` is a derived property.
    assert stamped.provenance.degraded is True

    identity_stamped = event.model_copy(update={"sender_id": "agt_1", "sequence": 3})
    assert identity_stamped.provenance == provenance


async def test_publisher_stamping_preserves_provenance() -> None:
    """The bus stamps identity and sequence on publish without dropping provenance."""
    bus = EventBus()
    publisher = bus.register_publisher(sender_id="agt_reply", source=EventSource.AGENT)
    sub = bus.subscribe("results")

    provenance = _failover_provenance()
    published = await publisher.publish(
        AgentEvent(type=EventType.AGENT_REPLY, topic="results", provenance=provenance)
    )
    assert published.sender_id == "agt_reply"
    assert published.provenance == provenance

    delivered = await sub.get()
    assert delivered.provenance == provenance


def test_create_response_forwards_a_nested_payload_from_the_causing_event() -> None:
    """`create_response(payload=causing.payload)` must not raise on a nested payload.

    `AgentEvent.payload` is an `ImmutableMapping`, frozen *recursively*: nested objects
    are `MappingProxyType` and nested arrays are `tuple`. `dict(payload)` copies only the
    top level, and the `payload` field of the event being constructed then **rejects**
    the result — `AfterValidator(freeze_mapping)` runs after `JsonValue` validation, so a
    nested proxy (and a nested tuple) fails validation rather than being re-frozen.

    Forwarding or amending the causing event's payload is the most natural use of this
    public method, and it raised. The single in-repo caller (`agent/base.py`, the
    `USER_INPUT` reply path) is safe only because it builds `reply_payload` as a plain
    dict 25 lines earlier — a guarantee living in a different module that nothing
    enforces. The fitness sweep cannot see this line either: `dict(payload)` is a bare
    name, not `dict(<expr>.<field>)`.

    Killed by: src/uclone_x/engine/event_bus.py :: payload=cast(dict[str, Any], unwrap_immutable(payload)) if payload is not None else {},
    """
    nested: dict[str, Any] = {
        "query": "Calculate total",
        "options": {"units": {"currency": "KRW"}},
        "items": [{"sku": "a"}, {"sku": "b"}],
    }
    causing = AgentEvent(
        type=EventType.USER_INPUT,
        sender_id="user_alice",
        payload=nested,
    )

    reply = causing.create_response(type=EventType.AGENT_REPLY, payload=causing.payload)

    assert unwrap_immutable(reply.payload) == nested


def test_create_response_does_not_inherit_provenance() -> None:
    """Attribution belongs to the producer of *this* result, so it is never inherited."""
    incoming = AgentEvent(
        type=EventType.USER_INPUT,
        sender_id="caller",
        provenance=_failover_provenance(),
    )

    inherited = incoming.create_response(type=EventType.AGENT_REPLY)
    assert inherited.provenance is None

    own = _failover_provenance()
    stated = incoming.create_response(type=EventType.AGENT_REPLY, provenance=own)
    assert stated.provenance == own


@pytest.mark.asyncio
async def test_event_bus_and_subscription_dropped_events_accounting() -> None:
    """Verify dropped_event_count and drop_reasons on EventBus and Subscription (#198)."""
    async with EventBus(maxsize=100, backpressure_policy=BackpressurePolicy.DROP_OLDEST) as bus:
        sub_drop_oldest = bus.subscribe(
            "metrics.test", maxsize=2, backpressure_policy=BackpressurePolicy.DROP_OLDEST
        )
        sub_drop_incoming = bus.subscribe(
            "metrics.test", maxsize=2, backpressure_policy=BackpressurePolicy.DROP_INCOMING
        )

        assert bus.dropped_event_count == 0
        assert bus.drop_reasons == {}
        assert sub_drop_oldest.dropped_event_count == 0
        assert sub_drop_incoming.dropped_event_count == 0

        # Publish 4 events (causing overflow in subscribers)
        for i in range(4):
            await bus.publish(
                AgentEvent(
                    event_id=f"evt_drop_{i}",
                    topic="metrics.test",
                    type=EventType.USER_INPUT,
                    priority=EventPriority.NORMAL,
                )
            )

        await bus.wait_until_idle()

        # Both subscribers had maxsize=2, so 2 events were dropped in each
        assert sub_drop_oldest.dropped_event_count == 2
        assert sub_drop_oldest.drop_reasons.get("drop_lowest_priority") == 2
        assert sub_drop_incoming.dropped_event_count == 2
        assert sub_drop_incoming.drop_reasons.get("drop_incoming") == 2

        # Aggregate on bus
        assert bus.dropped_event_count == 4
        assert bus.drop_reasons.get("drop_lowest_priority") == 2
        assert bus.drop_reasons.get("drop_incoming") == 2


# ==============================================================================
# Retargeting a live subscription (#225)
# ==============================================================================


def _evt(topic: str, *, session_id: str = "", recipient_id: str = "") -> AgentEvent:
    return AgentEvent(
        type=EventType.USER_INPUT,
        topic=topic,
        session_id=session_id,
        recipient_id=recipient_id,
    )


@pytest.mark.asyncio
async def test_retarget_repoints_a_live_subscription_without_closing_it() -> None:
    """The subscription object and its queue survive; only the target moves.

    `EventBus` dispatches by scanning its subscriber list and calling `matches` per
    event, with no topic index to invalidate, so a retarget takes effect on the next
    dispatched event with no bus bookkeeping and nothing closed. Asserted through a real
    dispatch, so it is the bus's routing under test and not just the predicate.
    """
    bus = EventBus()
    await bus.start()
    try:
        sub = bus.subscribe(topics={"agent.a", "session.s1", "broadcast"}, session_id="s1")

        assert sub.matches(_evt("session.s1", session_id="s1")) is True
        assert sub.matches(_evt("session.s2", session_id="s2")) is False

        stranded = sub.retarget({"agent.a", "session.s2", "broadcast"}, session_id="s2")

        assert stranded == ()
        assert sub.is_closed is False
        assert sub.topics == frozenset({"agent.a", "session.s2", "broadcast"})
        assert sub.session_id == "s2"

        # The same object now receives the new session's traffic off the live bus.
        published = await bus.publish(_evt("session.s2", session_id="s2"))
        delivered = await asyncio.wait_for(sub.get(), timeout=2.0)
        assert delivered.event_id == published.event_id
    finally:
        await bus.stop()


def test_retarget_requeues_what_still_matches_and_returns_what_does_not() -> None:
    """Closing would lose the whole queue; retargeting loses only the departing session.

    The events that survive here are exactly the ones option A's close-and-resubscribe
    would have destroyed as collateral: they were never addressed to either session.
    """
    bus = EventBus()
    sub = bus.subscribe(topics={"agent.a", "session.s1", "broadcast"}, session_id="s1")

    keep_agent = _evt("agent.a")
    keep_broadcast = _evt("broadcast")
    leaving = _evt("session.s1", session_id="s1")
    for event in (keep_agent, keep_broadcast, leaving):
        sub.deliver_nowait(event)
    assert sub.qsize() == 3

    stranded = sub.retarget({"agent.a", "session.s2", "broadcast"}, session_id="s2")

    assert [e.event_id for e in stranded] == [leaving.event_id]
    survivors = {sub.get_nowait().event_id for _ in range(sub.qsize())}
    assert survivors == {keep_agent.event_id, keep_broadcast.event_id}


def test_retarget_counts_stranded_events_under_a_reason_of_their_own() -> None:
    """A discard has to be countable and attributable, not merely returned once.

    The stranded set is handed back *and* recorded on the subscription's existing drop
    accounting, so a caller that ignores the return value still leaves a trace, and the
    reason distinguishes a switch from a backpressure eviction.
    """
    bus = EventBus()
    sub = bus.subscribe(topics={"session.s1"}, session_id="s1")
    for _ in range(2):
        sub.deliver_nowait(_evt("session.s1", session_id="s1"))

    assert sub.dropped_event_count == 0
    stranded = sub.retarget({"session.s2"}, session_id="s2")

    assert len(stranded) == 2
    assert sub.dropped_event_count == 2
    assert sub.drop_reasons == {"retarget_stranded": 2}


def test_retarget_refuses_a_closed_subscription() -> None:
    """Refusing here is what keeps the close sentinel out of the drain.

    `close` is the only writer of `_CLOSED_SENTINEL` and it sets `_closed` first, so a
    retarget that refuses closed subscriptions can never meet the sentinel — and so can
    never strand the wake-up that makes `get()` raise instead of blocking forever.
    """
    bus = EventBus()
    sub = bus.subscribe(topics={"session.s1"}, session_id="s1")
    sub.close()

    assert sub.is_closed is True
    with pytest.raises(SubscriptionClosedError):
        sub.retarget({"session.s2"}, session_id="s2")


def test_retarget_refuses_a_topic_the_allowlist_would_have_refused() -> None:
    """A retarget must not be a route around the subscription allowlist.

    It runs the same authorization `subscribe` runs, with the subscription's own
    capabilities, so it can never reach a topic a fresh subscribe would reject.
    """
    bus = EventBus(topic_allowlist={"session.s1", "agent.*"})
    sub = bus.subscribe(topics={"session.s1"}, session_id="s1")

    with pytest.raises(UnauthorizedSubscriptionError):
        sub.retarget({"session.s2"}, session_id="s2")


def test_a_refused_retarget_leaves_topics_filter_and_queue_untouched() -> None:
    """Validation runs before any mutation, so a refusal is a no-op, not a half-move."""
    bus = EventBus(topic_allowlist={"session.s1"})
    sub = bus.subscribe(topics={"session.s1"}, session_id="s1")
    queued = _evt("session.s1", session_id="s1")
    sub.deliver_nowait(queued)

    with pytest.raises(UnauthorizedSubscriptionError):
        sub.retarget({"session.s2"}, session_id="s2")

    assert sub.topics == frozenset({"session.s1"})
    assert sub.session_id == "s1"
    assert sub.qsize() == 1
    assert sub.get_nowait().event_id == queued.event_id
    assert sub.dropped_event_count == 0


def test_retarget_cannot_grant_a_wildcard_the_subscription_lacks_capability_for() -> None:
    """Wildcard authorization is re-checked against the subscription's own capabilities."""
    bus = EventBus(require_wildcard_auth=True)
    sub = bus.subscribe(topics={"session.s1"}, session_id="s1")

    with pytest.raises(UnauthorizedSubscriptionError):
        sub.retarget({"session.*"}, session_id=None)

    assert sub.topics == frozenset({"session.s1"})


# ------------------------------------------------------------------------------
# Why #225 option B — one session-independent wildcard subscription — is not safe
# on its own. Measured rather than asserted, because the card leaves the
# interaction between the pattern and the recipient/session filters open.
# ------------------------------------------------------------------------------


def test_a_session_wildcard_with_no_session_filter_matches_a_foreign_session() -> None:
    """`session.*` plus `session_id=None` is not "every session this agent hosts".

    It is *every session on the bus*. `matches` short-circuits the session filter when
    the subscription's own is unset, and the recipient filter when the event names no
    recipient, so an unaddressed event for a conversation this agent has never heard of
    matches. That is the leak the topic pattern alone cannot close, and it is why option
    B was not taken in the wildcard form the card describes.
    """
    bus = EventBus()
    sub = bus.subscribe(topics={"session.*"}, recipient_id="agent-a", session_id=None)

    # A session this agent hosts: intended.
    assert sub.matches(_evt("session.mine", session_id="mine", recipient_id="agent-a")) is True
    # Another agent's session, unaddressed: matched anyway.
    assert sub.matches(_evt("session.someone-elses", session_id="someone-elses")) is True


def test_a_scalar_session_filter_cannot_express_a_set_of_hosted_sessions() -> None:
    """The other half of the same finding: the filter is one id, not a set.

    Narrowing the pattern to the sessions actually hosted means naming them as exact
    topics, and then `session_id` has nothing to hold — one scalar cannot filter for two
    sessions at once. So option B's "subscribe once" reduces to either a filter that is
    off (above) or an exact topic set that has to be updated on every switch, which is
    what `retarget` does.
    """
    bus = EventBus()
    sub = bus.subscribe(topics={"session.s1", "session.s2"}, session_id="s1")

    # The topic set admits s2, but the scalar session filter rejects it.
    assert sub.matches(_evt("session.s2", session_id="s2")) is False
    assert sub.matches(_evt("session.s1", session_id="s1")) is True


# ==============================================================================
# Drop exits that used to be silent (#291)
#
# Every other discard in `EventSubscription` records itself — `drop_lowest_priority`,
# `drop_incoming`, `retarget_stranded`, and the `ERROR`/`BLOCK`-nowait paths, which
# `record_error` and raise. These pin the exits that did not.
# ==============================================================================


async def _wait_for(predicate: Any, timeout: float = 2.0, what: str = "condition") -> None:
    """Await a condition on the running loop, failing loudly rather than hanging."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"{what} was not reached within {timeout}s")


@pytest.mark.asyncio
async def test_an_event_parked_in_delivery_across_a_retarget_is_counted() -> None:
    """The #291 race itself: a parked putter, a retarget, and the event that lands unmatched.

    Under BLOCK a full subscriber queue makes `_dispatch_loop` hand delivery to a
    background task, which parks in `await queue.put`. `retarget` is synchronous and
    partitions only what is in the queue at that instant, so the parked event is
    invisible to it; when the putter resumes, `put` does not re-apply `matches` and the
    event enters the queue of a subscription that stopped listening for it.

    This drives the real race — the assertion below waits on the queue's actual parked
    putter, not on a stand-in — and pins the repair: the landing is *counted*. Option C
    on #291 deliberately does not prevent it, which the final assertions record.
    """
    bus = EventBus()
    await bus.start()
    try:
        sub = bus.subscribe(
            topics={"session.s1"},
            maxsize=1,
            backpressure_policy=BackpressurePolicy.BLOCK,
            session_id="s1",
        )
        queue = sub._queue  # pyright: ignore[reportPrivateUsage]

        first = await bus.publish(_evt("session.s1", session_id="s1"))
        await _wait_for(lambda: sub.full(), what="subscriber queue saturated")

        await bus.publish(_evt("session.s1", session_id="s1"))
        # The actual race condition: a putter blocked inside the subscriber's queue.
        await _wait_for(
            lambda: bool(getattr(queue, "_putters", None)),
            what="a delivery parked in queue.put",
        )
        assert sub.drop_reasons == {}

        stranded = sub.retarget({"session.s2"}, session_id="s2")

        assert tuple(e.event_id for e in stranded) == (first.event_id,)
        await _wait_for(
            lambda: sub.drop_reasons.get("refused_stale_delivery", 0) == 1,
            what="the parked event refused on landing",
        )

        # Both events are attributable drops: the drained one is retarget_stranded,
        # the parked one is refused_stale_delivery (#307 Option A).
        assert sub.drop_reasons == {"retarget_stranded": 1, "refused_stale_delivery": 1}
        assert sub.dropped_event_count == 2
        assert bus.drop_reasons == {"retarget_stranded": 1, "refused_stale_delivery": 1}
        assert bus.dropped_event_count == 2

        # Option A refuses the landing, so the event is not in the queue.
        assert sub.qsize() == 0
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_delivery_into_a_closed_subscription_is_counted_not_silent() -> None:
    """Both deliver doors: a closed subscription discards the event, and says so."""
    bus = EventBus()
    sync_sub = bus.subscribe(topics={"session.s1"}, session_id="s1")
    async_sub = bus.subscribe(topics={"session.s1"}, session_id="s1")
    sync_sub.close()
    async_sub.close()

    sync_sub.deliver_nowait(_evt("session.s1", session_id="s1"))
    await async_sub.deliver(_evt("session.s1", session_id="s1"))

    for sub in (sync_sub, async_sub):
        assert sub.drop_reasons == {"subscription_closed": 1}
        assert sub.dropped_event_count == 1
        assert sub.qsize() == 1  # the close sentinel only; the event was not queued


@pytest.mark.asyncio
async def test_the_bus_lets_the_subscription_count_its_own_closed_discard() -> None:
    """The bus must not short-circuit `is_closed` ahead of the counter.

    `_dispatch_loop` and `_deliver_to_subscriber_async` both used to return early on a
    closed subscriber, which is *before* the deliver methods can record anything — so
    counting inside `EventSubscription` alone would never fire on the paths the bus
    actually uses. Reinstating a closed subscription stands in for the window between
    the dispatch loop's snapshot of `_subscribers` and its delivery call.
    """
    bus = EventBus()
    await bus.start()
    try:
        dispatch_sub = bus.subscribe(topics={"session.s1"}, session_id="s1")
        dispatch_sub.close()
        bus._subscribers.append(dispatch_sub)  # pyright: ignore[reportPrivateUsage]

        await bus.publish(_evt("session.s1", session_id="s1"))
        await bus.wait_until_idle()
        await _wait_for(
            lambda: dispatch_sub.drop_reasons.get("subscription_closed") == 1,
            what="the dispatch-loop discard recorded",
        )

        # The BLOCK background path has the same obligation.
        block_sub = bus.subscribe(
            topics={"session.s1"},
            backpressure_policy=BackpressurePolicy.BLOCK,
            session_id="s1",
        )
        block_sub.close()
        await bus._deliver_to_subscriber_async(  # pyright: ignore[reportPrivateUsage]
            block_sub, _evt("session.s1", session_id="s1")
        )
        assert block_sub.drop_reasons == {"subscription_closed": 1}
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_parked_block_putter_on_close_is_counted_and_completes_cleanly() -> None:
    """Issue #311: An event parked in `await put` under BLOCK is counted on subscription close.

    When `EventSubscription.close()` is called, pending BLOCK delivery tasks parked in
    `await put` are cancelled, recorded under `subscription_closed`, and complete cleanly
    without task leakage or unhandled exceptions.
    """
    bus = EventBus()
    sub = bus.subscribe(
        topics={"test.topic"},
        maxsize=1,
        backpressure_policy=BackpressurePolicy.BLOCK,
    )
    # Fill the queue with 1 event
    sub.deliver_nowait(_evt("test.topic"))
    assert sub.full()
    assert sub.dropped_event_count == 0

    # Start a second delivery in a background task which will park in await put
    event2 = _evt("test.topic")
    delivery_task = asyncio.create_task(sub.deliver(event2))

    # Assert that while parked, the delivery task is pending and tracked
    await _wait_for(
        lambda: len(sub._pending_put_tasks) == 1,  # pyright: ignore[reportPrivateUsage]
        what="delivery task tracked in _pending_put_tasks",
    )
    assert not delivery_task.done()

    # Close the subscription
    sub.close()

    # Assert that the parked delivery task completes cleanly within timeout
    await asyncio.wait_for(delivery_task, timeout=1.0)
    assert delivery_task.done()
    assert not delivery_task.cancelled()
    assert delivery_task.exception() is None

    # Assert drop accounting:
    # 1 resident event evicted for close sentinel + 1 parked event cancelled on close
    assert sub.drop_reasons == {"close_sentinel_eviction": 1, "subscription_closed": 1}
    assert sub.drop_reasons.get("subscription_closed") == 1
    assert sub.dropped_event_count == 2
    assert sub.dropped_event_count == sum(sub.drop_reasons.values())
    assert len(sub._pending_put_tasks) == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_multiple_parked_block_putters_on_close_are_all_counted() -> None:
    """Multiple tasks parked in BLOCK delivery across close are all cancelled and counted (#311)."""
    bus = EventBus()
    sub = bus.subscribe(
        topics={"test.topic"},
        maxsize=1,
        backpressure_policy=BackpressurePolicy.BLOCK,
    )
    sub.deliver_nowait(_evt("test.topic"))
    assert sub.full()

    tasks = [asyncio.create_task(sub.deliver(_evt("test.topic"))) for _ in range(3)]
    await _wait_for(
        lambda: len(sub._pending_put_tasks) == 3,  # pyright: ignore[reportPrivateUsage]
        what="all 3 delivery tasks tracked in _pending_put_tasks",
    )
    for task in tasks:
        assert not task.done()

    sub.close()

    await asyncio.gather(*tasks)
    for task in tasks:
        assert task.done()
        assert not task.cancelled()
        assert task.exception() is None

    # 1 resident event evicted + 3 parked putters cancelled
    assert sub.drop_reasons == {"close_sentinel_eviction": 1, "subscription_closed": 3}
    assert sub.dropped_event_count == 4
    assert sub.dropped_event_count == sum(sub.drop_reasons.values())
    assert len(sub._pending_put_tasks) == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_event_bus_dispatch_parked_block_putter_on_close_is_counted_and_completes() -> None:
    """Full bus dispatch: parked BLOCK delivery across sub.close() records drop and completes."""
    bus = EventBus()
    await bus.start()
    try:
        sub = bus.subscribe(
            topics={"session.s1"},
            maxsize=1,
            backpressure_policy=BackpressurePolicy.BLOCK,
            session_id="s1",
        )
        await bus.publish(_evt("session.s1", session_id="s1"))
        await _wait_for(lambda: sub.full(), what="first event enqueued")

        await bus.publish(_evt("session.s1", session_id="s1"))
        await _wait_for(
            lambda: len(sub._pending_put_tasks) == 1,  # pyright: ignore[reportPrivateUsage]
            what="delivery task parked in _pending_put_tasks",
        )

        sub.close()

        await _wait_for(
            lambda: sub.drop_reasons.get("subscription_closed") == 1,
            what="parked delivery recorded on close",
        )
        assert sub.drop_reasons == {"close_sentinel_eviction": 1, "subscription_closed": 1}
        assert sub.dropped_event_count == 2
        assert len(sub._pending_put_tasks) == 0  # pyright: ignore[reportPrivateUsage]
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_parked_block_putter_cancelled_externally_when_not_closed_reraises() -> None:
    """External cancellation of a parked BLOCK putter re-raises CancelledError when not closed."""
    bus = EventBus()
    sub = bus.subscribe(
        topics={"test.topic"},
        maxsize=1,
        backpressure_policy=BackpressurePolicy.BLOCK,
    )
    sub.deliver_nowait(_evt("test.topic"))
    assert sub.full()

    delivery_task = asyncio.create_task(sub.deliver(_evt("test.topic")))
    await _wait_for(
        lambda: len(sub._pending_put_tasks) == 1,  # pyright: ignore[reportPrivateUsage]
        what="delivery task tracked in _pending_put_tasks",
    )

    delivery_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await delivery_task

    assert sub.dropped_event_count == 0
    assert sub.drop_reasons == {}
    assert len(sub._pending_put_tasks) == 0  # pyright: ignore[reportPrivateUsage]


def test_close_counts_the_event_it_evicts_to_seat_the_sentinel() -> None:
    """Closing a full subscription discards a real queued event to make room.

    `close` puts `_CLOSED_SENTINEL` on the queue to wake readers, and on a full queue it
    first evicts the least urgent event. That eviction is a loss like any other, and was
    the third silent exit in this class alongside the two `_closed` returns in the
    deliver methods.
    """
    bus = EventBus()
    sub = bus.subscribe(topics={"session.s1"}, maxsize=1, session_id="s1")
    sub.deliver_nowait(_evt("session.s1", session_id="s1"))
    assert sub.full()
    assert sub.dropped_event_count == 0

    sub.close()

    assert sub.drop_reasons == {"close_sentinel_eviction": 1}
    assert sub.dropped_event_count == 1


def test_drop_oldest_is_a_value_alias_not_a_fifth_policy() -> None:
    """#17's compatibility shim, pinned so it is not "fixed" into a distinct member.

    `DROP_OLDEST` shares `DROP_LOWEST_PRIORITY`'s value, so the enum has four canonical
    members and `DROP_OLDEST.name` reads back as `'DROP_LOWEST_PRIORITY'`. Dropping the
    *oldest* event from a priority queue was the inverted behaviour repaired in #17; the
    name survives only as an accepted spelling.
    """
    assert BackpressurePolicy.DROP_OLDEST is BackpressurePolicy.DROP_LOWEST_PRIORITY
    assert BackpressurePolicy.DROP_OLDEST.name == "DROP_LOWEST_PRIORITY"
    assert len(list(BackpressurePolicy)) == 4
    assert "DROP_OLDEST" in BackpressurePolicy.__members__
    assert BackpressurePolicy("drop_oldest") is BackpressurePolicy.DROP_LOWEST_PRIORITY


@pytest.mark.asyncio
async def test_an_unmatched_delivery_is_refused_and_recorded_as_a_drop() -> None:
    """An event that no longer matches the subscription at delivery is refused (#307).

    Refusing the event is a true drop (it is never delivered to the reader), so it
    must be recorded under `drop_reasons` and increment `dropped_event_count`. It
    never reaches the queue or the backpressure policy.
    """
    bus = EventBus()

    # Even with ERROR policy and a full queue, the event does not raise QueueFullError
    # because it is refused before the queue bounds are checked.
    sub = bus.subscribe(
        topics={"session.s1"},
        maxsize=1,
        backpressure_policy=BackpressurePolicy.ERROR,
        session_id="s1",
    )
    sub.deliver_nowait(_evt("session.s1", session_id="s1"))
    assert sub.full()

    # The unmatched event is refused and dropped, not raised as QueueFullError
    sub.deliver_nowait(_evt("session.other", session_id="other"))

    assert sub.drop_reasons == {"refused_stale_delivery": 1}
    assert sub.dropped_event_count == 1
    assert sub.error_count == 0
    assert sub.dropped_event_count == sum(sub.drop_reasons.values())


def test_the_two_close_path_reasons_are_readable_only_on_the_subscription() -> None:
    """`close` leaves the bus's subscriber list before recording, so the aggregate misses them.

    Not silent — both write a `logger.warning` and both land on the subscription — but
    §6.6 has to say which surface carries which reason, or an operator checks the wrong
    one. The live-subscription control below is what shows the aggregate works at all.
    """
    bus = EventBus()

    live = bus.subscribe(
        topics={"session.s1"},
        maxsize=1,
        backpressure_policy=BackpressurePolicy.DROP_INCOMING,
        session_id="s1",
    )
    live.deliver_nowait(_evt("session.s1", session_id="s1"))
    live.deliver_nowait(_evt("session.s1", session_id="s1"))
    # Control: a drop on a live subscription does reach the bus aggregate.
    assert bus.drop_reasons.get("drop_incoming") == 1

    closing = bus.subscribe(topics={"session.s2"}, maxsize=1, session_id="s2")
    closing.deliver_nowait(_evt("session.s2", session_id="s2"))
    closing.close()
    closing.deliver_nowait(_evt("session.s2", session_id="s2"))

    assert closing.drop_reasons == {"close_sentinel_eviction": 1, "subscription_closed": 1}
    assert "close_sentinel_eviction" not in bus.drop_reasons
    assert "subscription_closed" not in bus.drop_reasons


def test_dropped_event_count_reconciles_with_drop_reasons_on_every_reason() -> None:
    """`dropped_event_count == sum(drop_reasons.values())` on each of the five reasons.

    The sentence on `dropped_event_count` rests on this holding at every call site, not
    just the one a given test happens to exercise. Routing an unmatched delivery through
    `record_drop` broke it under `DROP_INCOMING` by counting one event twice, so it is
    pinned here per reason rather than measured once.

    Also pins the axis `dropped_event_count`'s docstring distinguishes: only
    `close_sentinel_eviction` requires a full queue; the other two non-backpressure
    reasons fire on a queue with room to spare.
    """
    bus = EventBus()
    seen: dict[str, int] = {}

    def check(sub: EventSubscription, reason: str) -> None:
        assert reason in sub.drop_reasons, f"{reason} was not recorded"
        assert sub.dropped_event_count == sum(sub.drop_reasons.values())
        seen[reason] = sub.drop_reasons[reason]

    evict = bus.subscribe(
        topics={"t"}, maxsize=1, backpressure_policy=BackpressurePolicy.DROP_LOWEST_PRIORITY
    )
    evict.deliver_nowait(_evt("t"))
    evict.deliver_nowait(_evt("t"))
    check(evict, "drop_lowest_priority")

    incoming = bus.subscribe(
        topics={"t"}, maxsize=1, backpressure_policy=BackpressurePolicy.DROP_INCOMING
    )
    incoming.deliver_nowait(_evt("t"))
    incoming.deliver_nowait(_evt("t"))
    check(incoming, "drop_incoming")

    # Not full: room to spare, and the drop is recorded anyway.
    stranded = bus.subscribe(topics={"session.s1"}, maxsize=8, session_id="s1")
    stranded.deliver_nowait(_evt("session.s1", session_id="s1"))
    assert not stranded.full()
    stranded.retarget({"session.s2"}, session_id="s2")
    check(stranded, "retarget_stranded")

    closed = bus.subscribe(topics={"t"}, maxsize=8)
    closed.close()
    assert not closed.full()
    closed.deliver_nowait(_evt("t"))
    check(closed, "subscription_closed")

    # The one reason that genuinely cannot fire without a full queue.
    roomy = bus.subscribe(topics={"t"}, maxsize=4)
    roomy.deliver_nowait(_evt("t"))
    roomy.close()
    assert roomy.drop_reasons == {}

    evicting = bus.subscribe(topics={"t"}, maxsize=1)
    evicting.deliver_nowait(_evt("t"))
    assert evicting.full()
    evicting.close()
    check(evicting, "close_sentinel_eviction")

    assert sorted(seen) == [
        "close_sentinel_eviction",
        "drop_incoming",
        "drop_lowest_priority",
        "retarget_stranded",
        "subscription_closed",
    ]
