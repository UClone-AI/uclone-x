"""Engine protocols for event bus, scheduler, and timer services.

`@runtime_checkable` is applied only where a runtime `isinstance` check is actually
performed. On a protocol with a `@property`, `issubclass()` raises `TypeError` and
`isinstance()` calls the object's getters as a side effect of the type test, and neither
form checks a signature — which is what actually drifted in issue 2026-09-02-035.
Conformance is enforced statically instead, by the bindings in
`tests/unit/test_protocol_conformance.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from uclone_x.engine.event_bus import (
    AgentEvent,
    BackpressurePolicy,
    ErrorHandler,
    EventCallback,
    EventPriority,
    EventSource,
    QueueFullError,
)

T = TypeVar("T")


@runtime_checkable
class PublisherHandleProtocol(Protocol):
    """Protocol for authenticated event bus publisher handles."""

    @property
    def sender_id(self) -> str:
        """Return the authoritative sender identifier."""
        ...

    @property
    def source(self) -> EventSource:
        """Return the authoritative event source."""
        ...

    @property
    def allowed_topics(self) -> frozenset[str] | None:
        """Return allowed topics for this publisher, or None for unrestricted."""
        ...

    @property
    def capabilities(self) -> frozenset[str]:
        """Return capabilities granted to this publisher."""
        ...

    def is_topic_allowed(self, topic: str) -> bool:
        """Check if this publisher is authorized to publish to the given topic."""
        ...

    async def publish(self, event: AgentEvent) -> AgentEvent:
        """Publish an event using this publisher's authoritative identity."""
        ...

    def publish_nowait(self, event: AgentEvent) -> AgentEvent:
        """Synchronously publish an event using this publisher's authoritative identity."""
        ...


class EventSubscriptionProtocol(Protocol):
    """Protocol for event subscriptions."""

    @property
    def topics(self) -> frozenset[str]:
        """Return the set of subscribed topics or topic patterns."""
        ...

    @property
    def recipient_id(self) -> str | None:
        """Return target recipient filter, if any."""
        ...

    @property
    def session_id(self) -> str | None:
        """Return session filter, if any."""
        ...

    @property
    def capabilities(self) -> frozenset[str]:
        """Return capabilities attached to this subscription."""
        ...

    @property
    def is_closed(self) -> bool:
        """Return True if this subscription is closed."""
        ...

    @property
    def backpressure_policy(self) -> BackpressurePolicy:
        """Return active backpressure policy."""
        ...

    @property
    def maxsize(self) -> int:
        """Return maximum queue capacity."""
        ...

    @property
    def errors(self) -> tuple[Exception, ...]:
        """Return recorded delivery errors."""
        ...

    @property
    def last_error(self) -> Exception | None:
        """Return most recent delivery error."""
        ...

    @property
    def error_count(self) -> int:
        """Return count of delivery errors."""
        ...

    @property
    def dropped_event_count(self) -> int:
        """Return the number of events dropped by this subscription."""
        ...

    @property
    def drop_reasons(self) -> Mapping[str, int]:
        """Return event drop counts grouped by reason."""
        ...

    def record_error(self, exc: Exception) -> None:
        """Record an observable delivery failure."""
        ...

    def matches_topic(self, topic: str) -> bool:
        """Check if any of the subscription's topic patterns match the given topic."""
        ...

    def matches(self, event: AgentEvent) -> bool:
        """Check if this subscription matches the given event (topic, recipient, session)."""
        ...

    async def get(self) -> AgentEvent:
        """Retrieve next event in priority order."""
        ...

    def get_nowait(self) -> AgentEvent:
        """Retrieve next event without awaiting."""
        ...

    def qsize(self) -> int:
        """Return buffered queue size."""
        ...

    def empty(self) -> bool:
        """Check if queue is empty."""
        ...

    def full(self) -> bool:
        """Check if queue is at capacity."""
        ...

    def close(self) -> None:
        """Close subscription."""
        ...

    def unsubscribe(self) -> None:
        """Alias for close()."""
        ...

    def deliver_nowait(self, event: AgentEvent) -> None:
        """Synchronously enqueue an event directly into this subscription, bypassing the bus."""
        ...

    def retarget(
        self,
        topics: str | Sequence[str] | set[str],
        *,
        session_id: str | None,
    ) -> tuple[AgentEvent, ...]:
        """Atomically redirect this subscription to a new session and topic pattern."""
        ...

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        """Async iteration support."""
        ...

    async def __aenter__(self) -> EventSubscriptionProtocol:
        """Async context manager entry."""
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        ...


@runtime_checkable
class EventBusProtocol(Protocol):
    """Protocol for the reactive in-memory event bus."""

    @property
    def maxsize(self) -> int:
        """Return maximum capacity of ingress event queue."""
        ...

    @property
    def backpressure_policy(self) -> BackpressurePolicy:
        """Return active backpressure policy."""
        ...

    @property
    def topic_allowlist(self) -> frozenset[str] | None:
        """Return configured topic allowlist, if any."""
        ...

    @property
    def require_wildcard_auth(self) -> bool:
        """Return True if wildcard subscriptions require explicit authorization."""
        ...

    @property
    def is_running(self) -> bool:
        """Return True if background dispatch loop is running."""
        ...

    @property
    def errors(self) -> tuple[QueueFullError, ...]:
        """Return recorded delivery errors."""
        ...

    @property
    def last_error(self) -> QueueFullError | None:
        """Return most recent delivery error."""
        ...

    @property
    def error_count(self) -> int:
        """Return count of delivery errors."""
        ...

    @property
    def dropped_event_count(self) -> int:
        """Return total events dropped across ingress queue and subscriptions."""
        ...

    @property
    def drop_reasons(self) -> Mapping[str, int]:
        """Return aggregate event drop counts grouped by reason."""
        ...

    def clear_errors(self) -> None:
        """Clear recorded delivery errors."""
        ...

    def set_error_handler(self, handler: ErrorHandler | None) -> None:
        """Register a handler for observable subscription delivery failures."""
        ...

    def qsize(self) -> int:
        """Return number of pending events."""
        ...

    def empty(self) -> bool:
        """Check if ingress queue is empty."""
        ...

    def full(self) -> bool:
        """Check if ingress queue is full."""
        ...

    def register_publisher(
        self,
        sender_id: str,
        source: EventSource = EventSource.AGENT,
        allowed_topics: Sequence[str] | set[str] | frozenset[str] | None = None,
        capabilities: Sequence[str] | set[str] | frozenset[str] | None = None,
    ) -> PublisherHandleProtocol:
        """Register an authenticated publisher identity with the bus."""
        ...

    async def publish(
        self,
        event: AgentEvent,
        *,
        publisher: Any = None,
    ) -> AgentEvent:
        """Publish an event to the bus asynchronously."""
        ...

    def publish_nowait(
        self,
        event: AgentEvent,
        *,
        publisher: Any = None,
    ) -> AgentEvent:
        """Synchronously enqueue an event without awaiting."""
        ...

    def subscribe(
        self,
        topics: str | Sequence[str] | set[str] = "*",
        maxsize: int = 1000,
        backpressure_policy: BackpressurePolicy | None = None,
        *,
        recipient_id: str | None = None,
        session_id: str | None = None,
        capabilities: Sequence[str] | set[str] | frozenset[str] | None = None,
    ) -> EventSubscriptionProtocol:
        """Subscribe to one or more topic patterns."""
        ...

    def subscribe_callback(
        self,
        topic_pattern: str,
        callback: EventCallback,
        *,
        capabilities: Sequence[str] | set[str] | frozenset[str] | None = None,
    ) -> Callable[[], None]:
        """Register a callback for events matching pattern."""
        ...

    async def wait_until_idle(self) -> None:
        """Wait until all currently queued ingress events are dispatched."""
        ...

    async def join(self) -> None:
        """Alias for wait_until_idle()."""
        ...

    async def start(self) -> None:
        """Start the background event dispatch loop."""
        ...

    async def stop(self) -> None:
        """Stop the event bus cleanly."""
        ...

    async def __aenter__(self) -> EventBusProtocol:
        """Async context manager entry."""
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        ...


@runtime_checkable
class TimerServiceProtocol(Protocol):
    """Protocol for reactive timer and alarm scheduling."""

    def schedule_once(
        self,
        delay_seconds: float,
        callback: Callable[[], Awaitable[None]],
        timer_id: str | None = None,
    ) -> str:
        """Schedule a one-shot async callback."""
        ...

    def cancel(self, timer_id: str) -> bool:
        """Cancel an active timer."""
        ...


@runtime_checkable
class SchedulerProtocol(Protocol):
    """Protocol for priority-based task scheduling in the local engine."""

    async def submit(
        self,
        coro_fn: Callable[..., Awaitable[T]],
        *args: object,
        priority: EventPriority = EventPriority.NORMAL,
        **kwargs: object,
    ) -> T:
        """Submit a coroutine for prioritized execution."""
        ...
