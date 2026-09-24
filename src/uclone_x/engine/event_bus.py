"""Reactive, zero-copy, in-memory Event Bus for UClone-X.

Implements Principle 1 (Reactive Event-Driven, zero busy polling) and
Principle 8 (Strict Typing).

Features:
- `AgentEvent` and `EventPriority` immutable Pydantic v2 models.
- Priority levels: CRITICAL (0), INTERRUPT (10), NORMAL (50), BACKGROUND (100).
- `EventBus` backed by `asyncio.PriorityQueue` with bounded capacity.
- Explicit backpressure policies (ERROR, DROP_OLDEST, DROP_INCOMING, BLOCK).
- Topic / channel subscription with exact and glob-pattern matching.
- Multi-subscriber pub/sub and non-blocking async consumption.
- Clean cancellation and lifecycle management without deadlocks.
"""

from __future__ import annotations

import asyncio
import fnmatch
import functools
import heapq
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from enum import IntEnum, StrEnum
from types import TracebackType
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.core.immutable import ImmutableMapping, unwrap_immutable
from uclone_x.core.provenance import Provenance
from uclone_x.errors import UCloneXError

logger = logging.getLogger(__name__)


def _drop_lowest_priority_from_queue(q: asyncio.PriorityQueue[AgentEvent]) -> AgentEvent | None:
    """Drop the least urgent (lowest priority / highest sort key) event from a PriorityQueue.

    Under asyncio.PriorityQueue, get_nowait() pops the highest-priority (lowest
    priority integer) event, which inverts DROP_OLDEST/DROP_LOWEST_PRIORITY into dropping the
    most urgent event (issue 2026-09-02-037). This helper inspects the underlying heap list,
    removes the entry that compares greatest under `AgentEvent.__lt__` (lowest priority,
    largest sequence), and re-heapifies the queue in O(N).
    """
    underlying: list[AgentEvent] | None = getattr(q, "_queue", None)
    if not underlying:
        return None

    max_idx = 0
    for i in range(1, len(underlying)):
        if underlying[max_idx] < underlying[i]:
            max_idx = i

    dropped = underlying.pop(max_idx)
    heapq.heapify(underlying)
    q.task_done()
    return dropped


# Backward-compatible alias for the queue eviction helper
_drop_oldest_from_priority_queue = _drop_lowest_priority_from_queue


class EventPriority(IntEnum):
    """Priority levels for event dispatch ordering (lower values = higher priority)."""

    CRITICAL = 0
    INTERRUPT = 10
    NORMAL = 50
    BACKGROUND = 100


class EventType(StrEnum):
    """Closed classification of events on the bus.

    The first seven are the types enumerated in `docs/event-driven-agent-core.md`
    section 3. `PROVIDER_FAILOVER` and `RETRY` are the decision-plane notices Principle 6
    requires to be first-class rather than free-form strings. `CONTEXT_COMPACTED` joins
    them on that same standard (issue #183): a compaction irreversibly changes what every
    subsequent turn of the session can see, it carries a P6-attributable result — an
    LLM-written ledger is a value a model produced — and a subscriber that cannot
    distinguish "the context was rewritten" from ordinary traffic cannot reason about why
    an agent's later answers stopped referring to earlier turns. `SUBSCRIPTION_CLOSED` is
    the bus's own wake-up sentinel and is never published by a component.
    """

    USER_INPUT = "USER_INPUT"
    AGENT_REPLY = "AGENT_REPLY"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_APPROVAL_REQUEST = "tool.approval.request"
    TOOL_APPROVAL_RESPONSE = "tool.approval.response"
    SUBAGENT_SPAWN = "SUBAGENT_SPAWN"
    SUBAGENT_DONE = "SUBAGENT_DONE"
    INTERRUPT = "INTERRUPT"
    PROVIDER_FAILOVER = "PROVIDER_FAILOVER"
    RETRY = "RETRY"
    CONTEXT_COMPACTED = "CONTEXT_COMPACTED"
    PLAN_STATUS_UPDATE = "PLAN_STATUS_UPDATE"
    PLAN_CREATED = "plan.created"
    PLAN_UPDATED = "plan.updated"
    SUBSCRIPTION_CLOSED = "SUBSCRIPTION_CLOSED"
    HOOK_EXECUTED = "hook.executed"
    HOOK_FAILED = "hook.failed"
    SETTINGS_UPDATED = "settings.updated"


class EventSource(StrEnum):
    """Origin of an event, per `docs/event-driven-agent-core.md` section 3."""

    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    TIMER = "timer"
    SYSTEM = "system"


class BackpressurePolicy(StrEnum):
    """Policies for handling capacity overflow when queues reach maxsize."""

    ERROR = "error"
    DROP_LOWEST_PRIORITY = "drop_lowest_priority"
    DROP_OLDEST = "drop_lowest_priority"
    DROP_INCOMING = "drop_incoming"
    BLOCK = "block"

    @classmethod
    def _missing_(cls, value: object) -> BackpressurePolicy | None:
        if value == "drop_oldest":
            return cls.DROP_LOWEST_PRIORITY
        return None


class EventBusError(UCloneXError):
    """Base exception for EventBus errors."""


class QueueFullError(EventBusError):
    """Raised when an EventBus queue is full under BackpressurePolicy.ERROR."""


class SubscriptionClosedError(EventBusError):
    """Raised when operations are attempted on a closed EventSubscription."""


class UnauthorizedPublishError(EventBusError):
    """Raised when an event publisher is not authorized or attempts unverified origin spoofing."""


class UnauthorizedSubscriptionError(EventBusError):
    """Raised when a subscription attempts unauthorized topic access or wildcard subscription."""


@functools.total_ordering
class AgentEvent(BaseModel):
    """Immutable event envelope passed through the reactive EventBus.

    Follows Principle 8 (Strict Typing) and Principle 1 (Reactive Event-Driven).

    `frozen=True` blocks attribute rebinding and `payload` is validated into a read-only
    mapping, so an event delivered by reference to several subscribers cannot be mutated
    by any of them — the immutability the fastpath is required to provide by
    `docs/a2a-protocol-spec.md` section 10.3. `extra="forbid"` makes `schema_version`
    meaningful: a field this version does not declare is rejected instead of passing
    through unnoticed.

    `provenance` is the P6 in-band attribution channel for a result travelling as a bus
    event. It is a typed optional field on the envelope rather than a payload key by
    convention; see its own `Field` description for the decision and the trade it makes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    event_id: str = Field(
        default_factory=lambda: f"evt_{uuid.uuid4().hex}",
        description="Unique event identifier.",
    )
    topic: str = Field(
        default="default",
        description="Topic or channel string for pub/sub routing.",
    )
    type: EventType = Field(
        description="Event classification. Required: a defaulted type would let an "
        "unclassified event pass as user input.",
    )
    source: EventSource = Field(
        default=EventSource.SYSTEM,
        description="Origin of the event. Publisher-supplied and therefore forgeable "
        "today; see issue 2026-09-02-039, which moves identity stamping to the bus.",
    )
    sender_id: str = Field(
        default="",
        description="Identifier of the sending agent or component.",
    )
    recipient_id: str = Field(
        default="",
        description="Target recipient agent ID or empty for topic-level broadcast.",
    )
    session_id: str = Field(
        default="",
        description="Session context identifier.",
    )
    priority: EventPriority = Field(
        default=EventPriority.NORMAL,
        description="Priority level for dispatch ordering.",
    )
    sequence: int = Field(
        default=0,
        description=(
            "Monotonic sequence number assigned by the bus for deterministic FIFO "
            "tie-breaking. **Not a session log offset** — it is bus-global rather than "
            "per-session, restarts at zero with the process, and is preserved when a "
            "publisher supplies its own. The log's offset is `uclone_x.core.log_offset."
            "LogOffset`, allocated at the persistence boundary. Reading this field as an "
            "offset is the error that made an earlier design claim the envelope was already "
            "log-ready (#565)."
        ),
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Epoch timestamp when the event was created.",
    )
    payload: ImmutableMapping = Field(
        default_factory=dict,
        description="Structured typed payload data. Read-only after validation.",
    )
    provenance: Provenance | None = Field(
        default=None,
        description="In-band P6 attribution for the result this event carries. Typed and "
        "optional on the envelope (issue #51, Option B): a result-bearing publisher "
        "stamps it after `require_provenance`, and a consumer reads `event.provenance` "
        "under ordinary validation instead of decoding an untyped payload key. The "
        "interim `payload['provenance']` convention this replaces is retired. "
        "ACCEPTED TRADE: every event declares the field, including the ones with no "
        "meaningful provenance — `INTERRUPT`, a `TOOL_RESULT` echo, the close sentinel — "
        "which carry `None`, so `None` here means 'not stated', never 'nothing went "
        "wrong'. Option C — a validator requiring the field for result-bearing types and "
        "forbidding it elsewhere — is the intended destination but is blocked on the "
        "event taxonomy (`docs/event-driven-agent-core.md` section 10, rows 3 and 10), "
        "which is still open. Do not add that validator here until the taxonomy closes.",
    )
    trace_id: str | None = Field(
        default=None,
        description="OpenTelemetry trace correlation ID.",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Event ID or correlation ID of the causing input event (Issue #60).",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Optional idempotency key for deduplication.",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Contract schema version.",
    )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, AgentEvent):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.sequence != other.sequence:
            return self.sequence < other.sequence
        # `total_ordering` derives the rest from `__lt__` and Pydantic's field-wise
        # `__eq__`. Without a final tiebreaker two distinct events sharing
        # (priority, sequence) compare False under every operator, so the order is not
        # total and heap behaviour depends on insertion accident.
        return self.event_id < other.event_id

    @property
    def is_interrupt(self) -> bool:
        """Return True if this event is an interrupt or critical event."""
        return self.priority <= EventPriority.INTERRUPT

    @property
    def is_critical(self) -> bool:
        """Return True if this event has CRITICAL priority."""
        return self.priority == EventPriority.CRITICAL

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Override model_copy so updates are strictly validated through model_validate (Issue #31)."""
        if update:
            dumped: dict[str, Any] = dict(self.model_dump())
            dumped.update(update)
            return type(self).model_validate(dumped)
        return super().model_copy(deep=deep)

    def with_sequence(self, sequence: int) -> AgentEvent:
        """Derive an event stamped with a sequence number, preserving strict validation."""
        return self.model_copy(update={"sequence": sequence})

    def create_response(
        self,
        type: EventType,
        *,
        payload: Mapping[str, Any] | None = None,
        topic: str | None = None,
        priority: EventPriority | None = None,
        recipient_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        provenance: Provenance | None = None,
    ) -> AgentEvent:
        """Derive a response event causally linked to this event, preserving correlation_id and trace_id.

        `provenance` is **not** inherited from the causing event: attribution belongs to
        whoever produced this result, so a response carries only what its own producer
        states. Omitting it leaves the field `None` (P6: "not stated", not "clean").

        `payload` is unwrapped **recursively** rather than shallow-copied. The most
        natural use of this method — forwarding or amending the causing event's payload,
        `e.create_response(..., payload=e.payload)` — hands it an `ImmutableMapping`
        whose nested values are `MappingProxyType`/`tuple`; `dict(payload)` copies only
        the top level, and `AgentEvent.payload` then *rejects* it, because
        `AfterValidator(freeze_mapping)` runs after `JsonValue` validation rather than
        repairing the shape (#665, #673). The single in-repo caller
        (`agent/base.py`, the `USER_INPUT` reply path) happens to build a plain dict, so
        this never fired; that is a guarantee living in another module and enforced by
        nothing. The textual sweep in
        `tests/fitness/test_no_shallow_unwrap_of_frozen_mappings.py` cannot see this
        occurrence either — `dict(payload)` is a bare name, not `dict(<expr>.<field>)`.
        """
        return AgentEvent(
            type=type,
            topic=topic if topic is not None else self.topic,
            payload=cast(dict[str, Any], unwrap_immutable(payload)) if payload is not None else {},
            priority=priority if priority is not None else self.priority,
            recipient_id=recipient_id if recipient_id is not None else (self.sender_id or ""),
            session_id=session_id if session_id is not None else self.session_id,
            correlation_id=self.correlation_id or self.event_id,
            trace_id=trace_id if trace_id is not None else self.trace_id,
            provenance=provenance,
        )


# Sentinel event to unblock consumers when a subscription is closed. Readers detect it
# by object identity, not by `type`: a closed enum stops a typo, but any component can
# publish an event carrying the sentinel's type, and comparing on type would let a
# publisher stop an arbitrary reader while `is_closed` stayed False (issue -039).
_CLOSED_SENTINEL = AgentEvent(
    event_id="__closed_sentinel__",
    type=EventType.SUBSCRIPTION_CLOSED,
    priority=EventPriority.BACKGROUND,
    sequence=2**62,
)


class PublisherHandle:
    """Authenticated publisher handle bound to an EventBus instance."""

    def __init__(
        self,
        bus: EventBus,
        sender_id: str,
        source: EventSource = EventSource.AGENT,
        allowed_topics: Sequence[str] | set[str] | frozenset[str] | None = None,
        capabilities: Sequence[str] | set[str] | frozenset[str] | None = None,
    ) -> None:
        self._bus = bus
        self._sender_id = sender_id
        self._source = source
        self._allowed_topics = frozenset(allowed_topics) if allowed_topics is not None else None
        self._capabilities: frozenset[str] = (
            frozenset(capabilities) if capabilities is not None else frozenset()
        )

    @property
    def sender_id(self) -> str:
        """Return the authoritative sender identifier."""
        return self._sender_id

    @property
    def source(self) -> EventSource:
        """Return the authoritative event source."""
        return self._source

    @property
    def allowed_topics(self) -> frozenset[str] | None:
        """Return allowed topics for this publisher, or None for unrestricted."""
        return self._allowed_topics

    @property
    def capabilities(self) -> frozenset[str]:
        """Return capabilities granted to this publisher."""
        return self._capabilities

    def is_topic_allowed(self, topic: str) -> bool:
        """Check if this publisher is authorized to publish to the given topic."""
        if self._allowed_topics is None:
            return True
        for pattern in self._allowed_topics:
            if pattern == "*" or pattern == topic or fnmatch.fnmatch(topic, pattern):
                return True
        return False

    async def publish(self, event: AgentEvent) -> AgentEvent:
        """Publish an event using this publisher's authoritative identity."""
        return await self._bus.publish(event, publisher=self)

    def publish_nowait(self, event: AgentEvent) -> AgentEvent:
        """Synchronously publish an event using this publisher's authoritative identity."""
        return self._bus.publish_nowait(event, publisher=self)


class EventSubscription:
    """Subscription handle representing a queue of events matching subscribed topics."""

    def __init__(
        self,
        bus: EventBus,
        topics: set[str],
        maxsize: int = 1000,
        backpressure_policy: BackpressurePolicy = BackpressurePolicy.ERROR,
        recipient_id: str | None = None,
        session_id: str | None = None,
        capabilities: Sequence[str] | set[str] | frozenset[str] | None = None,
    ) -> None:
        self._bus = bus
        self._topics = topics
        self._maxsize = maxsize
        self._backpressure_policy = backpressure_policy
        self._recipient_id = recipient_id
        self._session_id = session_id
        self._capabilities: frozenset[str] = (
            frozenset(capabilities) if capabilities is not None else frozenset()
        )
        self._queue: asyncio.PriorityQueue[AgentEvent] = asyncio.PriorityQueue(maxsize=maxsize)
        self._closed: bool = False
        self._pending_put_tasks: set[asyncio.Task[Any]] = set()
        self._errors: list[Exception] = []
        self._dropped_event_count: int = 0
        self._drop_reasons: dict[str, int] = {}

    @property
    def topics(self) -> frozenset[str]:
        """Return the set of subscribed topics or topic patterns."""
        return frozenset(self._topics)

    @property
    def recipient_id(self) -> str | None:
        """Return target recipient filter, if any."""
        return self._recipient_id

    @property
    def session_id(self) -> str | None:
        """Return session filter, if any."""
        return self._session_id

    @property
    def capabilities(self) -> frozenset[str]:
        """Return capabilities attached to this subscription."""
        return self._capabilities

    @property
    def is_closed(self) -> bool:
        """Return True if this subscription is closed."""
        return self._closed

    @property
    def backpressure_policy(self) -> BackpressurePolicy:
        """Return the active backpressure policy."""
        return self._backpressure_policy

    @property
    def maxsize(self) -> int:
        """Return the maximum queue capacity."""
        return self._maxsize

    @property
    def errors(self) -> tuple[Exception, ...]:
        """Return the history of delivery errors encountered by this subscription."""
        return tuple(self._errors)

    @property
    def last_error(self) -> Exception | None:
        """Return the most recent delivery error, or None if no errors occurred."""
        return self._errors[-1] if self._errors else None

    @property
    def error_count(self) -> int:
        """Return the number of delivery errors recorded."""
        return len(self._errors)

    @property
    def dropped_event_count(self) -> int:
        """Return the number of events this subscription dropped, for any reason.

        Across the call sites in this module, one increment is one event this
        subscription did not deliver, so this equals `sum(drop_reasons.values())` and the
        two reconcile against each other. That is a property of those call sites, not one
        this type enforces: `record_drop` is public and increments unconditionally, so
        what would falsify the sentence is a caller that records without an event, or
        records one event twice. The second is not hypothetical — routing an unmatched
        delivery through `record_drop` did exactly that under `DROP_INCOMING` until it
        was given its own counter (#198).

        Overflow under a backpressure policy is no longer the only cause:
        `retarget_stranded`, `subscription_closed` and `close_sentinel_eviction` are
        drops that **no backpressure policy governs**. Queue pressure is a separate axis
        and does not line up with that one — `close_sentinel_eviction` is recorded only
        inside `except asyncio.QueueFull` and so cannot fire *without* a full queue,
        while the other two fire regardless of how full the queue is (§6.6 of
        `docs/event-driven-agent-core.md`).

        Events counted by `unmatched_delivery_count` are deliberately **not** counted
        here: such an event is mis-filed rather than dropped, and on most policies it is
        delivered normally. Counting it here would make one event increment this twice.
        """
        return self._dropped_event_count

    @property
    def drop_reasons(self) -> Mapping[str, int]:
        """Return the count of event drops grouped by reason."""
        return dict(self._drop_reasons)

    def record_drop(self, reason: str = "subscription_queue_overflow") -> None:
        """Record one event dropped by this subscription under `reason`."""
        self._dropped_event_count += 1
        self._drop_reasons[reason] = self._drop_reasons.get(reason, 0) + 1

    def record_error(self, exc: Exception) -> None:
        """Record an observable delivery failure."""
        self._errors.append(exc)

    def _record_closed_drop(self, event: AgentEvent) -> None:
        """Record an event discarded because this subscription is already closed.

        Delivery into a closed subscription is a real loss: the event is neither
        queued nor raised, and no reader will ever see it. P6 forbids that happening
        without a trace, so it is counted under `drop_reasons["subscription_closed"]`
        exactly as the backpressure evictions and `retarget_stranded` are.
        """
        self.record_drop("subscription_closed")
        logger.warning(
            "Discarded event %s (type=%s, topic=%s, session=%s) delivered to a closed "
            "subscription for topics %s",
            event.event_id,
            event.type.value,
            event.topic,
            event.session_id or "<unset>",
            sorted(self._topics),
        )

    def _refuse_stale_delivery(self, event: AgentEvent) -> None:
        """Refuse an event that no longer matches its subscription at delivery (#307).

        `EventBus` matches at dispatch time; a retarget in the scheduling gap before
        `deliver` runs, or during the parked `await put` under BLOCK policy, can
        repoint the subscription. Under option A on #291, such events are refused
        outright rather than being delivered to a target that no longer listens for them.

        This is a true drop (the event never reaches the reader), so it is recorded
        under `drop_reasons` as `refused_stale_delivery` and increments
        `dropped_event_count`.
        """
        self.record_drop("refused_stale_delivery")
        logger.warning(
            "Refused stale delivery of event %s (type=%s, topic=%s, session=%s); "
            "no longer matches subscription (topics=%s, session_id=%s)",
            event.event_id,
            event.type.value,
            event.topic,
            event.session_id or "<unset>",
            sorted(self._topics),
            self._session_id or "<unset>",
        )

    def matches_topic(self, topic: str) -> bool:
        """Check if any of the subscription's topic patterns match the given topic."""
        for pattern in self._topics:
            if pattern == "*" or pattern == topic:
                return True
            if fnmatch.fnmatch(topic, pattern):
                return True
        return False

    def matches(self, event: AgentEvent) -> bool:
        """Check if this subscription matches the given event (topic, recipient, session)."""
        if not self.matches_topic(event.topic):
            return False
        if self._recipient_id and event.recipient_id and self._recipient_id != event.recipient_id:
            return False
        if self._session_id and event.session_id and self._session_id != event.session_id:
            return False
        return True

    def retarget(
        self,
        topics: str | Sequence[str] | set[str],
        *,
        session_id: str | None,
    ) -> tuple[AgentEvent, ...]:
        """Repoint this **live** subscription at a new topic set and session filter (#225).

        The subscription object, its queue and every event in it survive. `EventBus`
        dispatches by scanning its subscriber list and calling `matches` per event, with
        no topic index to update, so the new target takes effect on the next dispatched
        event and nothing has to be closed to change where this agent listens.

        That is the whole point. Closing the old subscription and opening a new one — the
        obvious way to follow a session switch — discards **everything queued on it**, and
        not only the departing session's events: a subscriber to
        `{agent.X, session.S, broadcast}` loses its `agent.X` and `broadcast` traffic too,
        because the queue dies with the object. Here those events are re-queued.

        Returns:
            The queued events the new target does **not** match — the *stranded* set —
            removed from the queue and handed back rather than swallowed, so the caller
            can state and attribute what it discarded. They are also counted on
            `dropped_event_count` and under `drop_reasons["retarget_stranded"]`.

            This is the events **in the queue** at this instant, which is not every event
            the old target loses. A delivery already parked in `deliver` under `BLOCK` is
            not in the queue and so cannot be partitioned here; it lands after this
            returns and is counted separately on `unmatched_delivery_count` (#291) —
            not as a drop, because it is still enqueued.

        Raises:
            SubscriptionClosedError: If this subscription is already closed. Refusing
                here is also what keeps the partition below simple: `close` is the only
                thing that puts `_CLOSED_SENTINEL` on the queue, and it sets `_closed`
                first, so the drain can never meet the sentinel and cannot swallow the
                wake-up `get` relies on.
            UnauthorizedSubscriptionError: If the new topics fail the bus's allowlist or
                wildcard-capability check. Validation runs **before** any mutation and
                the subscription's own `capabilities` are used, so a retarget can never
                reach a topic a fresh `subscribe` would have refused, and a refusal
                leaves topics, session filter and queue exactly as they were.
        """
        if self._closed:
            raise SubscriptionClosedError("Cannot retarget a closed subscription.")

        topic_set = {topics} if isinstance(topics, str) else set(topics)
        self._bus.authorize_subscription_topics(topic_set, self._capabilities)

        self._topics = topic_set
        self._session_id = session_id

        kept: list[AgentEvent] = []
        stranded: list[AgentEvent] = []
        while True:
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            if self.matches(event):
                kept.append(event)
            else:
                stranded.append(event)

        for event in kept:
            self._queue.put_nowait(event)
        for event in stranded:
            self.record_drop("retarget_stranded")
            logger.warning(
                "Stranded event %s (type=%s, topic=%s, session=%s) on subscription "
                "retarget to topics=%s session_id=%s",
                event.event_id,
                event.type.value,
                event.topic,
                event.session_id or "<unset>",
                sorted(topic_set),
                session_id,
            )
        return tuple(stranded)

    async def get(self) -> AgentEvent:
        """Await and return the next event in priority order.

        Raises:
            SubscriptionClosedError: If the subscription is closed and no more events remain.
        """
        if self._closed and self._queue.empty():
            raise SubscriptionClosedError("Subscription is closed.")

        event = await self._queue.get()
        if event is _CLOSED_SENTINEL:
            # Re-enqueue sentinel so all concurrent readers on this subscription wake up
            try:
                self._queue.put_nowait(_CLOSED_SENTINEL)
            except asyncio.QueueFull:
                pass
            self._queue.task_done()
            raise SubscriptionClosedError("Subscription is closed.")

        self._queue.task_done()
        return event

    def get_nowait(self) -> AgentEvent:
        """Return the next event in priority order without awaiting.

        Raises:
            SubscriptionClosedError: If the subscription is closed and empty.
            asyncio.QueueEmpty: If no events are currently buffered.
        """
        if self._closed and self._queue.empty():
            raise SubscriptionClosedError("Subscription is closed.")

        event = self._queue.get_nowait()
        if event is _CLOSED_SENTINEL:
            try:
                self._queue.put_nowait(_CLOSED_SENTINEL)
            except asyncio.QueueFull:
                pass
            self._queue.task_done()
            raise SubscriptionClosedError("Subscription is closed.")

        self._queue.task_done()
        return event

    def qsize(self) -> int:
        """Return the current number of buffered events in the subscriber queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if the subscriber queue is empty."""
        return self._queue.empty()

    def full(self) -> bool:
        """Return True if the subscriber queue has reached capacity."""
        return self._queue.full()

    async def deliver(self, event: AgentEvent) -> None:
        """Enqueue an event into the subscription's priority queue according to policy.

        **Every exit that returns** records itself first (P6): a closed subscription
        under `subscription_closed`, the backpressure evictions under their own reasons,
        and an event the current target no longer matches on
        `unmatched_delivery_count`. The `ERROR` and `BLOCK`-nowait paths raise instead,
        which is already observable.

        Parked `BLOCK` putters across `close()` are deterministically resolved and
        recorded under `subscription_closed` without task leakage (#311).
        """
        if self._closed:
            self._record_closed_drop(event)
            return

        # Two windows can repoint this subscription between `EventBus`'s dispatch-time
        # match and the enqueue below: the scheduling gap before this coroutine runs, and
        # the parked `await put` under BLOCK. The first is checked here, the second after
        # the put; `matched_at_entry` keeps an event from being counted on both.
        matched_at_entry = self.matches(event)
        if not matched_at_entry:
            self._refuse_stale_delivery(event)
            return

        if not self._queue.full():
            self._queue.put_nowait(event)
            return

        # Queue is full, handle backpressure
        if self._backpressure_policy == BackpressurePolicy.ERROR:
            exc = QueueFullError(
                f"Subscription queue is full (maxsize={self._maxsize}) for topics {self._topics}"
            )
            self.record_error(exc)
            raise exc
        elif self._backpressure_policy in (
            BackpressurePolicy.DROP_LOWEST_PRIORITY,
            BackpressurePolicy.DROP_OLDEST,
        ):
            dropped = _drop_lowest_priority_from_queue(self._queue)
            if dropped is not None:
                self.record_drop("drop_lowest_priority")
                logger.warning(
                    "Dropped lowest priority event %s (priority=%s, seq=%s) due to subscriber queue overflow",
                    dropped.event_id,
                    dropped.priority,
                    dropped.sequence,
                )
            self._queue.put_nowait(event)
        elif self._backpressure_policy == BackpressurePolicy.DROP_INCOMING:
            self.record_drop("drop_incoming")
            logger.warning(
                "Dropped incoming event %s (priority=%s) due to subscriber queue overflow",
                event.event_id,
                event.priority,
            )
            return
        elif self._backpressure_policy == BackpressurePolicy.BLOCK:
            if self._closed:
                self._record_closed_drop(event)
                return
            current_task = asyncio.current_task()
            if current_task is not None:
                self._pending_put_tasks.add(current_task)
            try:
                await self._queue.put(event)
            except asyncio.CancelledError:
                if self._closed:
                    self._record_closed_drop(event)
                    return
                raise
            finally:
                if current_task is not None:
                    self._pending_put_tasks.discard(current_task)
            # This put parks while the queue is full, and `retarget` is synchronous and
            # partitions only what is in the queue at that instant — a parked putter is
            # not in it, and `put` does not re-apply `matches`. So the event can land on
            # a subscription that stopped listening for it while it waited. Under option
            # A on #291, we re-apply `matches` here and refuse the stale event.
            if matched_at_entry and not self.matches(event):
                underlying: list[AgentEvent] | None = getattr(self._queue, "_queue", None)
                if underlying is not None:
                    try:
                        underlying.remove(event)
                        import heapq

                        heapq.heapify(underlying)
                        self._queue.task_done()
                        if hasattr(self._queue, "_wakeup_next") and hasattr(
                            self._queue, "_putters"
                        ):
                            self._queue._wakeup_next(self._queue._putters)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownArgumentType]
                    except ValueError:
                        pass
                self._refuse_stale_delivery(event)

    def deliver_nowait(self, event: AgentEvent) -> None:
        """Synchronously enqueue an event into the subscription's queue.

        Every exit that returns records itself first (P6), on the same reasons `deliver`
        uses. This method never parks, so `deliver`'s caveat about an event stranded in a
        blocked `put` (#311) does not apply here.

        Raises:
            QueueFullError: If queue is full under BackpressurePolicy.ERROR or BLOCK.
        """
        if self._closed:
            self._record_closed_drop(event)
            return

        if not self.matches(event):
            self._refuse_stale_delivery(event)
            return

        if not self._queue.full():
            self._queue.put_nowait(event)
            return

        if self._backpressure_policy == BackpressurePolicy.ERROR:
            exc = QueueFullError(
                f"Subscription queue is full (maxsize={self._maxsize}) for topics {self._topics}"
            )
            self.record_error(exc)
            raise exc
        elif self._backpressure_policy in (
            BackpressurePolicy.DROP_LOWEST_PRIORITY,
            BackpressurePolicy.DROP_OLDEST,
        ):
            dropped = _drop_lowest_priority_from_queue(self._queue)
            if dropped is not None:
                self.record_drop("drop_lowest_priority")
                logger.warning(
                    "Dropped lowest priority event %s (priority=%s, seq=%s) due to subscriber queue overflow",
                    dropped.event_id,
                    dropped.priority,
                    dropped.sequence,
                )
            self._queue.put_nowait(event)
        elif self._backpressure_policy == BackpressurePolicy.DROP_INCOMING:
            self.record_drop("drop_incoming")
            logger.warning(
                "Dropped incoming event %s (priority=%s) due to subscriber queue overflow",
                event.event_id,
                event.priority,
            )
        elif self._backpressure_policy == BackpressurePolicy.BLOCK:
            exc = QueueFullError(
                f"Subscription queue is full (maxsize={self._maxsize}) under BLOCK policy (nowait)"
            )
            self.record_error(exc)
            raise exc

    def close(self) -> None:
        """Close the subscription and wake up pending readers."""
        if self._closed:
            return
        self._closed = True
        self._bus.remove_subscription(self)
        for task in list(self._pending_put_tasks):
            task.cancel()
        self._pending_put_tasks.clear()
        try:
            self._queue.put_nowait(_CLOSED_SENTINEL)
        except asyncio.QueueFull:
            # Making room for the sentinel evicts a real queued event. That is a drop and
            # is counted like any other (P6); it was previously the third silent exit in
            # this class, alongside the two `_closed` returns in the deliver methods.
            evicted = _drop_lowest_priority_from_queue(self._queue)
            if evicted is not None:
                self.record_drop("close_sentinel_eviction")
                logger.warning(
                    "Evicted event %s (priority=%s, seq=%s) to make room for the close "
                    "sentinel on a full subscription queue",
                    evicted.event_id,
                    evicted.priority,
                    evicted.sequence,
                )
            try:
                self._queue.put_nowait(_CLOSED_SENTINEL)
            except Exception as exc:
                logger.warning(
                    "Failed to enqueue close sentinel on full subscription queue: %s", exc
                )
        except Exception as exc:
            logger.warning("Unexpected error during subscription close: %s", exc)

    def unsubscribe(self) -> None:
        """Alias for close()."""
        self.close()

    async def __aiter__(self) -> AsyncIterator[AgentEvent]:
        """Iterate reactively over incoming events until subscription is closed."""
        while not self._closed:
            try:
                event = await self.get()
                yield event
            except SubscriptionClosedError:
                break
            except asyncio.CancelledError:
                break

    async def __aenter__(self) -> EventSubscription:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


EventCallback = Callable[[AgentEvent], Awaitable[None] | None]
ErrorHandler = Callable[
    [QueueFullError, AgentEvent, EventSubscription | None], Awaitable[None] | None
]


class EventBus:
    """High-performance in-memory Reactive Event Bus.

    Adheres to:
    - Principle 1: Reactive Event-Driven (backed by asyncio.PriorityQueue, zero busy polling)
    - Principle 8: Strict Typing (Pydantic v2 immutable contracts, strict annotations)
    """

    def __init__(
        self,
        maxsize: int = 10000,
        backpressure_policy: BackpressurePolicy = BackpressurePolicy.ERROR,
        topic_allowlist: Sequence[str] | set[str] | frozenset[str] | None = None,
        require_wildcard_auth: bool = False,
    ) -> None:
        self._maxsize = maxsize
        self._backpressure_policy = backpressure_policy
        self._topic_allowlist = frozenset(topic_allowlist) if topic_allowlist is not None else None
        self._require_wildcard_auth = require_wildcard_auth
        self._queue: asyncio.PriorityQueue[AgentEvent] = asyncio.PriorityQueue(maxsize=maxsize)
        self._subscribers: list[EventSubscription] = []
        self._callbacks: list[tuple[str, EventCallback]] = []
        self._publishers: dict[str, PublisherHandle] = {}
        self._sequence_counter: int = 0
        self._dispatch_task: asyncio.Task[None] | None = None
        self._delivery_tasks: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()
        self._stopped: bool = False
        self._errors: list[QueueFullError] = []
        self._error_handler: ErrorHandler | None = None
        self._dropped_event_count: int = 0
        self._drop_reasons: dict[str, int] = {}

    @property
    def maxsize(self) -> int:
        """Return the maximum capacity of the ingress event queue."""
        return self._maxsize

    @property
    def backpressure_policy(self) -> BackpressurePolicy:
        """Return the active backpressure policy."""
        return self._backpressure_policy

    @property
    def topic_allowlist(self) -> frozenset[str] | None:
        """Return the configured topic allowlist, if any."""
        return self._topic_allowlist

    @property
    def require_wildcard_auth(self) -> bool:
        """Return True if wildcard subscriptions require explicit authorization."""
        return self._require_wildcard_auth

    @property
    def is_running(self) -> bool:
        """Return True if the background dispatch loop is actively running."""
        return self._dispatch_task is not None and not self._dispatch_task.done()

    @property
    def errors(self) -> tuple[QueueFullError, ...]:
        """Return all subscription delivery errors recorded by the bus."""
        return tuple(self._errors)

    @property
    def last_error(self) -> QueueFullError | None:
        """Return the most recent delivery error recorded by the bus."""
        return self._errors[-1] if self._errors else None

    @property
    def error_count(self) -> int:
        """Return the count of delivery errors recorded by the bus."""
        return len(self._errors)

    @property
    def dropped_event_count(self) -> int:
        """Return total events dropped across ingress queue and active subscriptions.

        P6 sanctions the drop under backpressure policies (DROP_OLDEST, DROP_INCOMING)
        while requiring in-band result provenance to survive. This property exposes the
        aggregate drop count at runtime for operator visibility (#198).

        **Active** is the operative word. `EventSubscription.close` removes the
        subscription from the bus before recording anything, so the two close-path
        reasons — `subscription_closed` and `close_sentinel_eviction` — are counted on
        the subscription and can never appear here. They are not lost: each writes a
        `logger.warning`, and the subscription object still carries them. §6.6 of
        `docs/event-driven-agent-core.md` says which surface to read for which reason.
        """
        sub_drops = sum(sub.dropped_event_count for sub in self._subscribers)
        return self._dropped_event_count + sub_drops

    @property
    def drop_reasons(self) -> Mapping[str, int]:
        """Return aggregate event drop counts grouped by reason across ingress and subscriptions."""
        reasons: dict[str, int] = dict(self._drop_reasons)
        for sub in self._subscribers:
            for r, count in sub.drop_reasons.items():
                reasons[r] = reasons.get(r, 0) + count
        return reasons

    def record_drop(self, reason: str = "ingress_queue_overflow") -> None:
        """Record an absorbed event drop at the ingress queue."""
        self._dropped_event_count += 1
        self._drop_reasons[reason] = self._drop_reasons.get(reason, 0) + 1

    def clear_errors(self) -> None:
        """Clear recorded delivery errors."""
        self._errors.clear()

    def set_error_handler(self, handler: ErrorHandler | None) -> None:
        """Register a handler for observable subscription delivery failures under ERROR policy."""
        self._error_handler = handler

    def qsize(self) -> int:
        """Return the number of events waiting in the ingress queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if the ingress queue is empty."""
        return self._queue.empty()

    def full(self) -> bool:
        """Return True if the ingress queue has reached capacity."""
        return self._queue.full()

    def _next_sequence(self) -> int:
        """Generate a monotonically increasing sequence number."""
        self._sequence_counter += 1
        return self._sequence_counter

    def _ensure_running(self) -> None:
        """Lazily ensure the dispatch task is running if an event loop is active."""
        if self._stopped:
            raise EventBusError("EventBus has been stopped.")
        if self._dispatch_task is None or self._dispatch_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._dispatch_task = loop.create_task(
                    self._dispatch_loop(), name="ucx-eventbus-dispatcher"
                )
            except RuntimeError:
                pass

    def register_publisher(
        self,
        sender_id: str,
        source: EventSource = EventSource.AGENT,
        allowed_topics: Sequence[str] | set[str] | frozenset[str] | None = None,
        capabilities: Sequence[str] | set[str] | frozenset[str] | None = None,
    ) -> PublisherHandle:
        """Register an authenticated publisher identity with the bus."""
        pub = PublisherHandle(
            bus=self,
            sender_id=sender_id,
            source=source,
            allowed_topics=allowed_topics,
            capabilities=capabilities,
        )
        self._publishers[sender_id] = pub
        return pub

    def _validate_event_for_publishing(
        self,
        event: AgentEvent,
        publisher: PublisherHandle | None = None,
    ) -> AgentEvent:
        """Validate event and authoritatively stamp identity and sequence."""
        if publisher is not None:
            if not publisher.is_topic_allowed(event.topic):
                raise UnauthorizedPublishError(
                    f"Publisher '{publisher.sender_id}' is not authorized to publish to topic '{event.topic}'"
                )
            updates: dict[str, Any] = {
                "source": publisher.source,
                "sender_id": publisher.sender_id,
            }
            if event.sequence == 0:
                updates["sequence"] = self._next_sequence()
            return event.model_copy(update=updates)

        # Publisher is None (unverified / direct call)
        if event.source == EventSource.USER:
            raise UnauthorizedPublishError(
                "Unverified publisher cannot publish event with source 'user'. Register an authenticated USER publisher handle."
            )

        if event.sequence == 0:
            return event.with_sequence(self._next_sequence())
        return event

    async def start(self) -> None:
        """Explicitly start the event bus dispatcher."""
        self._stopped = False
        if self._dispatch_task is None or self._dispatch_task.done():
            loop = asyncio.get_running_loop()
            self._dispatch_task = loop.create_task(
                self._dispatch_loop(), name="ucx-eventbus-dispatcher"
            )

    async def stop(self) -> None:
        """Gracefully stop the event bus and close all active subscriptions."""
        if self._stopped:
            return
        self._stopped = True

        # Cancel background dispatch loop
        if self._dispatch_task is not None and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
            self._dispatch_task = None

        # Cancel any pending async delivery / callback tasks
        for t in list(self._delivery_tasks):
            t.cancel()
        if self._delivery_tasks:
            await asyncio.gather(*self._delivery_tasks, return_exceptions=True)
        self._delivery_tasks.clear()

        # Drain and discard remaining ingress queue items so wait_until_idle() never hangs
        while not self._queue.empty():
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                logger.warning(
                    "Discarded undelivered ingress event %s during stop()", dropped.event_id
                )
            except (asyncio.QueueEmpty, ValueError):
                break

        # Close all active subscriptions
        async with self._lock:
            subscribers_to_close = list(self._subscribers)
            self._subscribers.clear()
            self._callbacks.clear()

        for sub in subscribers_to_close:
            sub.close()

    async def publish(
        self,
        event: AgentEvent,
        *,
        publisher: PublisherHandle | None = None,
    ) -> AgentEvent:
        """Publish an event to the bus asynchronously.

        Identity (source, sender_id) is authoritatively stamped when published via a
        PublisherHandle. Unverified direct publishing claiming source='user' is rejected.

        Raises:
            EventBusError: If the bus is stopped.
            UnauthorizedPublishError: If origin spoofing or unauthorized topic is detected.
            QueueFullError: If the queue is full under BackpressurePolicy.ERROR.
        """
        self._ensure_running()
        stamped_event = self._validate_event_for_publishing(event, publisher=publisher)

        if not self._queue.full():
            self._queue.put_nowait(stamped_event)
            return stamped_event

        # Handle backpressure
        if self._backpressure_policy == BackpressurePolicy.ERROR:
            raise QueueFullError(f"EventBus queue is full (maxsize={self._maxsize})")
        elif self._backpressure_policy in (
            BackpressurePolicy.DROP_LOWEST_PRIORITY,
            BackpressurePolicy.DROP_OLDEST,
        ):
            dropped = _drop_lowest_priority_from_queue(self._queue)
            if dropped is not None:
                self.record_drop("ingress_drop_lowest_priority")
                logger.warning(
                    "Dropped lowest priority event %s (priority=%s, seq=%s) from ingress queue",
                    dropped.event_id,
                    dropped.priority,
                    dropped.sequence,
                )
            self._queue.put_nowait(stamped_event)
        elif self._backpressure_policy == BackpressurePolicy.DROP_INCOMING:
            self.record_drop("ingress_drop_incoming")
            logger.warning(
                "Dropped incoming event %s (priority=%s) at ingress queue",
                stamped_event.event_id,
                stamped_event.priority,
            )
        elif self._backpressure_policy == BackpressurePolicy.BLOCK:
            await self._queue.put(stamped_event)

        return stamped_event

    def publish_nowait(
        self,
        event: AgentEvent,
        *,
        publisher: PublisherHandle | None = None,
    ) -> AgentEvent:
        """Synchronously enqueue an event without awaiting.

        Identity (source, sender_id) is authoritatively stamped when published via a
        PublisherHandle. Unverified direct publishing claiming source='user' is rejected.

        Raises:
            EventBusError: If the bus is stopped.
            UnauthorizedPublishError: If origin spoofing or unauthorized topic is detected.
            QueueFullError: If queue is full under BackpressurePolicy.ERROR or BLOCK.
        """
        self._ensure_running()
        stamped_event = self._validate_event_for_publishing(event, publisher=publisher)

        if not self._queue.full():
            self._queue.put_nowait(stamped_event)
            return stamped_event

        if self._backpressure_policy in (BackpressurePolicy.ERROR, BackpressurePolicy.BLOCK):
            raise QueueFullError(f"EventBus queue is full (maxsize={self._maxsize})")
        elif self._backpressure_policy in (
            BackpressurePolicy.DROP_LOWEST_PRIORITY,
            BackpressurePolicy.DROP_OLDEST,
        ):
            dropped = _drop_lowest_priority_from_queue(self._queue)
            if dropped is not None:
                self.record_drop("ingress_drop_lowest_priority")
                logger.warning(
                    "Dropped lowest priority event %s (priority=%s, seq=%s) from ingress queue",
                    dropped.event_id,
                    dropped.priority,
                    dropped.sequence,
                )
            self._queue.put_nowait(stamped_event)
        elif self._backpressure_policy == BackpressurePolicy.DROP_INCOMING:
            self.record_drop("ingress_drop_incoming")
            logger.warning(
                "Dropped incoming event %s (priority=%s) at ingress queue",
                stamped_event.event_id,
                stamped_event.priority,
            )

        return stamped_event

    def authorize_subscription_topics(
        self,
        topics: set[str],
        capabilities: frozenset[str],
    ) -> None:
        """Validate topic patterns against allowlist and wildcard authorization rules.

        Public because `EventSubscription.retarget` must run exactly this check before
        repointing a live subscription (#225). A retarget that skipped it would be a
        route to a topic a fresh `subscribe` would have refused, which is the same
        widening the allowlist exists to prevent — reached one call later.
        """
        for pattern in topics:
            if not pattern:
                raise UnauthorizedSubscriptionError("Topic pattern cannot be empty.")

            if self._topic_allowlist is not None:
                allowed = False
                for allowed_pat in self._topic_allowlist:
                    if (
                        allowed_pat == "*"
                        or pattern == allowed_pat
                        or fnmatch.fnmatch(pattern, allowed_pat)
                        or (allowed_pat.endswith(".*") and pattern.startswith(allowed_pat[:-2]))
                    ):
                        allowed = True
                        break
                if not allowed:
                    raise UnauthorizedSubscriptionError(
                        f"Subscription to topic '{pattern}' is not allowed by topic allowlist."
                    )

            if self._require_wildcard_auth and ("*" in pattern or "?" in pattern):
                if (
                    "wildcard_subscribe" not in capabilities
                    and "admin" not in capabilities
                    and "*" not in capabilities
                ):
                    raise UnauthorizedSubscriptionError(
                        f"Wildcard subscription to '{pattern}' requires 'wildcard_subscribe' capability."
                    )

    def subscribe(
        self,
        topics: str | Sequence[str] | set[str] = "*",
        maxsize: int = 1000,
        backpressure_policy: BackpressurePolicy | None = None,
        *,
        recipient_id: str | None = None,
        session_id: str | None = None,
        capabilities: Sequence[str] | set[str] | frozenset[str] | None = None,
    ) -> EventSubscription:
        """Subscribe to one or more topic patterns.

        Supports exact matching ("agent.coder") and wildcard globbing ("agent.*", "*").
        Validates topics against topic allowlists and checks required capabilities.
        """
        self._ensure_running()

        if isinstance(topics, str):
            topic_set = {topics}
        else:
            topic_set = set(topics)

        caps: frozenset[str] = frozenset(capabilities) if capabilities is not None else frozenset()
        self.authorize_subscription_topics(topic_set, caps)

        policy = self._backpressure_policy if backpressure_policy is None else backpressure_policy
        sub = EventSubscription(
            bus=self,
            topics=topic_set,
            maxsize=maxsize,
            backpressure_policy=policy,
            recipient_id=recipient_id,
            session_id=session_id,
            capabilities=caps,
        )
        self._subscribers.append(sub)
        return sub

    def subscribe_callback(
        self,
        topic_pattern: str,
        callback: EventCallback,
        *,
        capabilities: Sequence[str] | set[str] | frozenset[str] | None = None,
    ) -> Callable[[], None]:
        """Register a callback for events matching `topic_pattern`.

        Returns an unsubscribe function that unregisters the callback.
        """
        self._ensure_running()
        caps: frozenset[str] = frozenset(capabilities) if capabilities is not None else frozenset()
        self.authorize_subscription_topics({topic_pattern}, caps)

        entry = (topic_pattern, callback)
        self._callbacks.append(entry)

        def unsubscribe() -> None:
            if entry in self._callbacks:
                self._callbacks.remove(entry)

        return unsubscribe

    def remove_subscription(self, sub: EventSubscription) -> None:
        """Remove a subscription from the active subscriber list."""
        if sub in self._subscribers:
            self._subscribers.remove(sub)

    async def wait_until_idle(self) -> None:
        """Wait until all currently queued ingress events are dispatched.

        If the bus is stopped, returns immediately without hanging.
        """
        if self._stopped:
            return
        await self._queue.join()

    async def join(self) -> None:
        """Alias for wait_until_idle()."""
        await self.wait_until_idle()

    def _handle_delivery_error(
        self,
        exc: QueueFullError,
        event: AgentEvent,
        sub: EventSubscription | None = None,
    ) -> None:
        """Handle observable delivery failure when a subscriber queue overflows under ERROR policy."""
        self._errors.append(exc)
        if sub is not None and exc not in sub.errors:
            sub.record_error(exc)
        logger.error(
            "Subscriber queue overflow (ERROR policy) for event %s on topics %s: %s",
            event.event_id,
            sub.topics if sub else "unknown",
            exc,
        )
        if self._error_handler is not None:
            try:
                res = self._error_handler(exc, event, sub)
                if asyncio.iscoroutine(res):
                    task = asyncio.create_task(res, name="ucx-bus-error-handler")
                    self._delivery_tasks.add(task)
                    task.add_done_callback(self._delivery_tasks.discard)
            except Exception as handler_exc:
                logger.exception("Error executing event bus error handler: %s", handler_exc)

    async def _deliver_to_subscriber_async(self, sub: EventSubscription, event: AgentEvent) -> None:
        """Deliver an event to a subscriber asynchronously without blocking main dispatch.

        A closed subscription is *not* short-circuited here. `EventSubscription.deliver`
        owns that check and counts the discard under `subscription_closed`; returning
        early would skip the counter and put the silent drop back (P6).
        """
        try:
            await sub.deliver(event)
        except asyncio.CancelledError:
            return
        except QueueFullError as exc:
            self._handle_delivery_error(exc, event, sub)
        except Exception as exc:
            logger.exception("Error during async delivery to subscriber: %s", exc)

    async def _dispatch_loop(self) -> None:
        """Continuous reactive dispatch loop delivering events to subscribers."""
        while not self._stopped:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                break

            try:
                # Match subscriptions using topic, recipient_id, and session_id addressing
                matching_subs = [sub for sub in list(self._subscribers) if sub.matches(event)]
                for sub in matching_subs:
                    # A subscription closed between this snapshot and delivery is handled
                    # by the deliver methods, which count the discard under
                    # `subscription_closed`. Skipping it here would drop it silently (P6).
                    if not sub.full() or sub.backpressure_policy != BackpressurePolicy.BLOCK:
                        try:
                            sub.deliver_nowait(event)
                        except QueueFullError as exc:
                            self._handle_delivery_error(exc, event, sub)
                        except Exception as exc:
                            logger.exception(
                                "Error delivering event %s to subscriber: %s",
                                event.event_id,
                                exc,
                            )
                    else:
                        # Subscriber queue is full under BLOCK policy.
                        # Deliver concurrently in background to prevent stalling the whole bus.
                        task = asyncio.create_task(
                            self._deliver_to_subscriber_async(sub, event),
                            name=f"ucx-deliver-{event.event_id}",
                        )
                        self._delivery_tasks.add(task)
                        task.add_done_callback(self._delivery_tasks.discard)

                # Match callbacks
                for pattern, cb in list(self._callbacks):
                    if (
                        pattern == "*"
                        or pattern == event.topic
                        or fnmatch.fnmatch(event.topic, pattern)
                    ):
                        try:
                            res = cb(event)
                            if asyncio.iscoroutine(res):
                                cb_task = asyncio.create_task(res, name=f"ucx-cb-{event.event_id}")
                                self._delivery_tasks.add(cb_task)
                                cb_task.add_done_callback(self._delivery_tasks.discard)
                        except Exception:
                            logger.exception(
                                "Error executing event callback for topic %s", event.topic
                            )
            finally:
                self._queue.task_done()

    async def __aenter__(self) -> EventBus:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()
