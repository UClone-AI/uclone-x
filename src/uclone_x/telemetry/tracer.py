"""OpenTelemetry-compatible tracer implementation for agent reasoning, tools, and A2A events."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import secrets
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import JsonValue

from uclone_x.telemetry.models import SpanKind, SpanRecord, SpanStatus
from uclone_x.telemetry.protocols import SpanStreamProtocol, TraceRecorderProtocol

logger = logging.getLogger(__name__)

#: Completed spans retained before the oldest are evicted. A tracer is a buffer, not a
#: store: something has to export or read the spans, and nothing that fails to do so
#: should be able to exhaust memory. Chosen large enough that a REPL turn or an HTTP
#: request never overflows it in practice, so an eviction means a real consumer gap
#: rather than ordinary throughput.
DEFAULT_MAX_COMPLETED_SPANS = 2048

#: Spans a single `stream_spans` subscriber may have pending before the oldest of *its*
#: pending spans is discarded. `asyncio.Queue()` defaults to `maxsize=0`, i.e. unbounded,
#: which made every subscriber a second unbounded span buffer that #187's bound did not
#: cover: measured against a stalled subscriber, the completed-span buffer held at 4
#: while the subscriber's queue reached 20,000 (issue #197). Sized so a subscriber that
#: is merely slow for a moment loses nothing, and only one that has stopped draining
#: does.
DEFAULT_MAX_SUBSCRIBER_QUEUE_SPANS = 512

#: Identities of dropped `failover.event` spans retained for check-5 attribution before
#: the oldest is forgotten (issue #199). Bounded by how many failovers occurred rather
#: than by total span volume, because only failover spans are retained at all; a failover
#: is an exceptional path, so this is reached only when a great many of them were lost.
DEFAULT_MAX_RETAINED_DROPPED_FAILOVER_IDS = 256

#: The span name P6 check 5 resolves `provenance.attempts[*].span_id` against. Duplicated
#: as a literal at the emitting call site in `agent/base.py`; that file has concurrent
#: writers, so the two are not yet folded together.
FAILOVER_EVENT_SPAN_NAME = "failover.event"

_active_span_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "uclone_x_active_span_id", default=None
)


class SpanFate(StrEnum):
    """What this tracer can say about a span id it is asked to resolve (issue #199).

    Deliberately three states and not two: "this tracer has no record of it" is not the
    same claim as "it was never emitted", and collapsing them would be the absence-read-
    as-success shape P6 forbids.
    """

    #: This tracer still holds a span with this id. Says nothing about the span's *name*
    #: — check 5's own name filter remains the check for a wrongly threaded span id.
    RETAINED = "retained"

    #: This tracer held it and stopped holding it. `drop_reason` says which loss path.
    DROPPED = "dropped"

    #: This tracer has no record of it. See `FailoverSpanAttribution` for what this state
    #: cannot separate.
    UNRECORDED = "unrecorded"


@dataclass(frozen=True)
class FailoverSpanAttribution:
    """Why a `provenance.attempts[*].span_id` does or does not resolve here (issue #199).

    P6 check 5 requires every `attempts[*].span_id` on a `failover` result to resolve to
    an emitted `failover.event` span. When one does not, the reader needs to know whether
    *this* span was lost by the tracer or never emitted at all — a count keyed by reason
    answers that only in aggregate.

    **What `UNRECORDED` still cannot separate.** A fabricated `span_id` and one that was
    emitted, exported, and then acknowledged through `discard_exported` both leave no
    trace here, because a consumed span is not a loss and is not counted as one. Closing
    that would mean retaining every successfully exported failover id for the life of the
    tracer, which is the "a tracer is a buffer, not a store" line #187 drew. When an id
    was exported, check 5 is answerable at the collector that received it.

    `identities_forgotten` is the honest qualifier on `UNRECORDED`: the retained-identity
    set is itself bounded, so when it has evicted, `UNRECORDED` no longer rules out
    "dropped, and its identity forgotten too". A caller that reads `UNRECORDED` while
    this is non-zero has a maybe, not a no.
    """

    span_id: str
    fate: SpanFate
    drop_reason: str | None = None
    identities_forgotten: int = 0


@dataclass
class _InFlightSpan:
    span_id: str
    trace_id: str
    name: str
    kind: SpanKind
    parent_span_id: str | None
    start_time_ns: int
    attributes: dict[str, JsonValue]


class TelemetryTracer(TraceRecorderProtocol, SpanStreamProtocol):
    """Tracer implementation managing hierarchical spans, active context, and span streaming."""

    def __init__(
        self,
        trace_id: str | None = None,
        default_attributes: Mapping[str, JsonValue] | None = None,
        max_completed_spans: int = DEFAULT_MAX_COMPLETED_SPANS,
        max_subscriber_queue_spans: int = DEFAULT_MAX_SUBSCRIBER_QUEUE_SPANS,
        max_retained_dropped_failover_ids: int = DEFAULT_MAX_RETAINED_DROPPED_FAILOVER_IDS,
    ) -> None:
        self._trace_id: str = trace_id or f"trc_{secrets.token_hex(12)}"
        self._default_attributes: dict[str, JsonValue] = (
            dict(default_attributes) if default_attributes is not None else {}
        )
        self._active_spans: dict[str, _InFlightSpan] = {}
        # Bounded, because nothing outside this class is obliged to drain it (issue
        # #187). The `ui/app.py` tracer is a module-global with no exporter and no
        # reader, so an unbounded list there grows for the life of the server process.
        self._completed_spans: deque[SpanRecord] = deque(maxlen=max_completed_spans)
        # Every span this tracer stops accounting for, and why. A bound that discards
        # silently is the P6 shape -- a control that loses data without saying so -- and
        # a log line alone is what `BaseAgent.processing_errors` exists to avoid: "P6
        # forbids a failure that is visible only in telemetry". So drops are counted on
        # an attribute a caller and a test can read, not only warned about.
        #
        # Three counters and not one, because these are three different losses and a
        # single number that mixed them could not be read as anything (#197): a span can
        # leave the buffer while a subscriber still holds it, and forgetting a dropped
        # span's *identity* loses no span at all. See each property's docstring.
        self._buffer_evicted_span_count: int = 0
        self._drop_reasons: dict[str, int] = {}
        self._undelivered_span_count: int = 0
        self._undelivered_reasons: dict[str, int] = {}
        self._forgotten_drop_identity_count: int = 0
        # Insertion-ordered, so the bound evicts first-in first-out. Maps a dropped
        # `failover.event` span id to the reason it was dropped (#199).
        self._dropped_failover_reasons: dict[str, str] = {}
        self._max_retained_dropped_failover_ids: int = max_retained_dropped_failover_ids
        self._max_subscriber_queue_spans: int = max_subscriber_queue_spans
        self._stream_subscribers: set[asyncio.Queue[SpanRecord]] = set()

    @property
    def trace_id(self) -> str:
        """Return the root trace ID for this tracer."""
        return self._trace_id

    @property
    def active_span_count(self) -> int:
        """Return the number of currently active in-flight spans."""
        return len(self._active_spans)

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: str | None = None,
        attributes: Mapping[str, JsonValue] | None = None,
    ) -> str:
        """Start a new span and return its unique span ID."""
        span_id = f"spn_{secrets.token_hex(8)}"
        resolved_parent = (
            parent_span_id if parent_span_id is not None else _active_span_id_var.get()
        )

        merged_attrs: dict[str, JsonValue] = dict(self._default_attributes)
        if attributes is not None:
            merged_attrs.update(attributes)

        in_flight = _InFlightSpan(
            span_id=span_id,
            trace_id=self._trace_id,
            name=name,
            kind=kind,
            parent_span_id=resolved_parent,
            start_time_ns=time.time_ns(),
            attributes=merged_attrs,
        )
        self._active_spans[span_id] = in_flight
        return span_id

    def end_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.OK,
        error_message: str | None = None,
    ) -> SpanRecord | None:
        """Complete an in-flight span, record its end time, and notify span streamers."""
        in_flight = self._active_spans.pop(span_id, None)
        if in_flight is None:
            return None

        record = SpanRecord(
            trace_id=in_flight.trace_id,
            span_id=in_flight.span_id,
            name=in_flight.name,
            parent_span_id=in_flight.parent_span_id,
            kind=in_flight.kind,
            start_time_ns=in_flight.start_time_ns,
            end_time_ns=time.time_ns(),
            attributes=in_flight.attributes,
            status=status,
            error_message=error_message,
        )
        if (
            self._completed_spans.maxlen is not None
            and len(self._completed_spans) == self._completed_spans.maxlen
        ):
            # `deque` would evict the oldest silently. Account for it first -- and name
            # it, so a `failover.event` evicted here stays attributable (#199) instead of
            # becoming one anonymous tick of `buffer_overflow`.
            self._record_drop((self._completed_spans[0],), "buffer_overflow")
        self._completed_spans.append(record)

        for queue in list(self._stream_subscribers):
            self._offer_to_subscriber(queue, record)

        return record

    def _offer_to_subscriber(self, queue: asyncio.Queue[SpanRecord], record: SpanRecord) -> None:
        """Hand a span to one subscriber's queue, dropping its oldest if it is full.

        The subscriber's queue is bounded, so a subscriber that has stopped draining
        cannot grow tracer-held memory without limit (#197). `DROP_OLDEST` rather than
        `DROP_INCOMING`, matching the completed-span `deque` and `EventBus`'s policy of
        the same name: a live span stream is a tail, and the newest spans are the ones a
        reader reconnecting to it wants.

        The discarded span is counted as *undelivered*, not as dropped: it is still in
        the completed-span buffer, so it is still readable through
        `get_completed_spans()` and still resolvable by check 5. Only this subscriber
        lost it.
        """
        if queue.maxsize > 0 and queue.full():
            # Sync, on one event loop, with no await between the two calls, so the space
            # this frees cannot be taken by anyone else before the `put_nowait` below.
            self._record_undelivered(queue.get_nowait(), "subscriber_queue_overflow")
        queue.put_nowait(record)

    @asynccontextmanager
    async def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: str | None = None,
        attributes: Mapping[str, JsonValue] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Scope a span to an `async with` block, automatically managing parent/child context."""
        span_id = self.start_span(
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            attributes=attributes,
        )
        token = _active_span_id_var.set(span_id)
        try:
            yield span_id
            self.end_span(span_id=span_id, status=SpanStatus.OK)
        except Exception as exc:
            self.end_span(span_id=span_id, status=SpanStatus.ERROR, error_message=str(exc))
            raise
        finally:
            _active_span_id_var.reset(token)

    def agent_turn_span(
        self,
        agent_id: str,
        session_id: str,
        turn_index: int | None = None,
        parent_span_id: str | None = None,
        extra_attributes: Mapping[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[str]:
        """Async context manager helper for agent reasoning turn spans."""
        attrs: dict[str, Any] = {
            "agent_id": agent_id,
            "session_id": session_id,
        }
        if turn_index is not None:
            attrs["turn_index"] = turn_index
        if extra_attributes is not None:
            attrs.update(extra_attributes)
        return self.span(
            name="agent.run",
            kind=SpanKind.INTERNAL,
            parent_span_id=parent_span_id,
            attributes=attrs,
        )

    def tool_call_span(
        self,
        tool_name: str,
        parent_span_id: str | None = None,
        tool_path: str | None = None,
        extra_attributes: Mapping[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[str]:
        """Async context manager helper for tool invocation spans."""
        attrs: dict[str, Any] = {"tool": tool_name}
        if tool_path is not None:
            attrs["path"] = tool_path
        if extra_attributes is not None:
            attrs.update(extra_attributes)
        return self.span(
            name="tool.execute",
            kind=SpanKind.INTERNAL,
            parent_span_id=parent_span_id,
            attributes=attrs,
        )

    def llm_request_span(
        self,
        provider: str,
        model: str,
        parent_span_id: str | None = None,
        tokens: int | None = None,
        extra_attributes: Mapping[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[str]:
        """Async context manager helper for GenAI LLM generation spans."""
        attrs: dict[str, Any] = {
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
            "provider": provider,
            "model": model,
        }
        if tokens is not None:
            attrs["tokens"] = tokens
        if extra_attributes is not None:
            attrs.update(extra_attributes)
        return self.span(
            name="llm.generate",
            kind=SpanKind.CLIENT,
            parent_span_id=parent_span_id,
            attributes=attrs,
        )

    def a2a_event_span(
        self,
        target_agent_id: str,
        protocol: str = "google-a2a/v1",
        parent_span_id: str | None = None,
        extra_attributes: Mapping[str, Any] | None = None,
    ) -> AbstractAsyncContextManager[str]:
        """Async context manager helper for A2A delegation spans."""
        attrs: dict[str, Any] = {
            "target": target_agent_id,
            "target_agent_id": target_agent_id,
            "protocol": protocol,
        }
        if extra_attributes is not None:
            attrs.update(extra_attributes)
        return self.span(
            name="a2a.delegate",
            kind=SpanKind.PRODUCER,
            parent_span_id=parent_span_id,
            attributes=attrs,
        )

    async def stream_spans(self) -> AsyncIterator[SpanRecord]:
        """Stream completed span records in real-time as they finish.

        **The subscriber's queue is owned by the tracer and is bounded** (#197). A
        subscriber that stops draining loses its oldest pending spans, counted on
        `undelivered_span_count`; it is not disconnected, and this generator does not
        raise or end early because of backpressure. A slow SSE client is the ordinary
        case, so the ordinary case degrades that client's stream rather than terminating
        it -- and terminating it is what could not be done honestly, since an
        `AsyncIterator` that simply stops reads to its consumer as "the stream ended",
        which is a normal outcome standing in for a failure.

        A span discarded here is *undelivered to this subscriber*, not lost: it is still
        in the completed-span buffer for `get_completed_spans()` to return.
        """
        queue: asyncio.Queue[SpanRecord] = asyncio.Queue(maxsize=self._max_subscriber_queue_spans)
        self._stream_subscribers.add(queue)
        try:
            while True:
                record = await queue.get()
                yield record
        finally:
            self._stream_subscribers.discard(queue)

    def get_completed_spans(self) -> tuple[SpanRecord, ...]:
        """Return the completed spans currently retained by this tracer.

        "Currently retained", not "all recorded": the buffer is bounded and callers may
        have discarded exported batches. `buffer_evicted_span_count` says how many are missing.
        """
        return tuple(self._completed_spans)

    @property
    def buffer_evicted_span_count(self) -> int:
        """Completed spans that left **this tracer's completed-span buffer** unconsumed.

        Scoped deliberately, because the unscoped reading of this counter was wrong. It
        means "the buffer stopped holding these"; **it does not mean the spans were
        freed**, and a rise in it is not evidence that tracer-held memory is bounded.
        A span counted here may still be retained by a `stream_spans` subscriber's
        pending queue, and it may still be held by a caller that took it from
        `get_completed_spans()` and has not finished with it (#187, #197, #206).

        The name scopes itself at the call site to prevent reading it as a total-memory
        claim. The three separate counters -- this one, `undelivered_span_count`, and
        `forgotten_drop_identity_count` -- are what stop any single number from being
        read as a claim about total memory.
        """
        return self._buffer_evicted_span_count

    @property
    def dropped_span_count(self) -> int:
        """Deprecated alias for `buffer_evicted_span_count` (#206)."""
        return self._buffer_evicted_span_count

    @property
    def drop_reasons(self) -> Mapping[str, int]:
        """Buffer-departure counts by reason — `buffer_overflow`, or a caller's reason.

        Sums to `buffer_evicted_span_count`. Subscriber non-delivery is *not* in here; it is in
        `undelivered_reasons`, because a span can be in both losses at once and a shared
        bucket would double-count it.
        """
        return dict(self._drop_reasons)

    @property
    def undelivered_span_count(self) -> int:
        """Spans a `stream_spans` subscriber's bounded queue could not keep (#197).

        A different loss from `buffer_evicted_span_count`: the span is still in the completed-
        span buffer and still resolvable by P6 check 5. What was lost is one subscriber's
        view of it. Counted separately so neither number has to mean two things.
        """
        return self._undelivered_span_count

    @property
    def undelivered_reasons(self) -> Mapping[str, int]:
        """Non-delivery counts by reason — currently `subscriber_queue_overflow`.

        Paired with `undelivered_span_count` for subscriber queue overflows, distinct from
        buffer-drop accounting (`drop_reasons`, `buffer_evicted_span_count`).
        """
        return dict(self._undelivered_reasons)

    @property
    def forgotten_drop_identity_count(self) -> int:
        """Dropped `failover.event` ids the retained-identity set could no longer hold.

        The retention added for #199 is itself bounded, and a bound whose eviction went
        unaccounted would rebuild #187 one level up -- inside the fix for its own
        follow-up. So the set's own losses are counted here, and
        `attribute_failover_span` reports this number alongside every `UNRECORDED`
        verdict, where it is the difference between "no" and "maybe".
        """
        return self._forgotten_drop_identity_count

    def attribute_failover_span(self, span_id: str) -> FailoverSpanAttribution:
        """Say what became of one `provenance.attempts[*].span_id` here (issue #199).

        The question P6 check 5 leaves a reader holding when an `attempts[*].span_id`
        does not resolve: was this span dropped by the tracer, or never emitted? An
        aggregate count keyed by reason cannot answer it for a *specific* id.

        Only dropped `failover.event` identities are retained, which is why this method
        names failover in its own name. That is the check-5-relevant subset and no other
        check resolves a span id, so retaining every dropped id would multiply the
        retention by total span volume for no additional answer -- and would push this
        set's own eviction into the ordinary case rather than the exceptional one.
        A consequence worth stating plainly: a dropped span that was *not* a
        `failover.event` reports `UNRECORDED` here. For a compliant producer that cannot
        arise, because check 5 obliges the emitter to thread the id of a `failover.event`
        span; for a non-compliant one, `UNRECORDED` is the correct answer to "did this
        tracer lose your failover span" -- it never had one under that id.

        Read `FailoverSpanAttribution` for what `UNRECORDED` cannot separate.
        """
        for span in self._completed_spans:
            if span.span_id == span_id:
                return FailoverSpanAttribution(
                    span_id=span_id,
                    fate=SpanFate.RETAINED,
                    identities_forgotten=self._forgotten_drop_identity_count,
                )
        reason = self._dropped_failover_reasons.get(span_id)
        if reason is not None:
            return FailoverSpanAttribution(
                span_id=span_id,
                fate=SpanFate.DROPPED,
                drop_reason=reason,
                identities_forgotten=self._forgotten_drop_identity_count,
            )
        return FailoverSpanAttribution(
            span_id=span_id,
            fate=SpanFate.UNRECORDED,
            identities_forgotten=self._forgotten_drop_identity_count,
        )

    def _record_undelivered(self, span: SpanRecord, reason: str) -> None:
        """Account for a span a subscriber's bounded queue discarded (#197)."""
        first_of_this_reason = reason not in self._undelivered_reasons
        self._undelivered_span_count += 1
        self._undelivered_reasons[reason] = self._undelivered_reasons.get(reason, 0) + 1
        # Warn once per reason, for the same argument as `_record_drop`: a subscriber
        # that has stopped draining discards one span per span forever, and a warning
        # each time would bury the first -- the one that says a consumer is stuck.
        if first_of_this_reason:
            logger.warning(
                "A span stream subscriber is not draining; discarding its oldest "
                "pending span (%s, span_id=%s); first occurrence. Further "
                "non-deliveries are counted on `undelivered_span_count`.",
                reason,
                span.span_id,
            )
        else:
            logger.debug(
                "Discarded another pending span for a stalled subscriber (%s, "
                "span_id=%s); %d total",
                reason,
                span.span_id,
                self._undelivered_span_count,
            )

    def _remember_dropped_failover(self, span_id: str, reason: str) -> None:
        """Retain a dropped `failover.event` id so check 5 stays per-span (#199)."""
        if span_id in self._dropped_failover_reasons:
            return
        if len(self._dropped_failover_reasons) >= self._max_retained_dropped_failover_ids:
            forgotten = next(iter(self._dropped_failover_reasons))
            del self._dropped_failover_reasons[forgotten]
            self._forgotten_drop_identity_count += 1
            if self._forgotten_drop_identity_count == 1:
                logger.warning(
                    "Retained dropped-failover identities are full (%d); forgetting the "
                    "oldest (span_id=%s). `attribute_failover_span` can no longer "
                    "distinguish 'never emitted' from 'dropped' for forgotten ids; the "
                    "count is on `forgotten_drop_identity_count`.",
                    self._max_retained_dropped_failover_ids,
                    forgotten,
                )
        self._dropped_failover_reasons[span_id] = reason

    def _record_drop(self, spans: Sequence[SpanRecord | _InFlightSpan], reason: str) -> None:
        """Account for spans leaving the buffer without being consumed.

        Takes the records rather than a count so a dropped `failover.event` can be
        remembered by id (#199), not only tallied.
        """
        count = len(spans)
        if count <= 0:
            return
        first_of_this_reason = reason not in self._drop_reasons
        self._buffer_evicted_span_count += count
        self._drop_reasons[reason] = self._drop_reasons.get(reason, 0) + count
        for span in spans:
            if span.name == FAILOVER_EVENT_SPAN_NAME:
                self._remember_dropped_failover(span.span_id, reason)
        # The counter is the observable; the log is a hint towards it. Warn once per
        # reason and stay quiet afterwards: a bounded buffer with no consumer evicts one
        # span per span forever, and a warning each time would bury the first occurrence
        # -- which is the one that tells an operator a consumer is missing.
        if first_of_this_reason:
            logger.warning(
                "Dropping completed spans (%s); first occurrence, %d so far. Further "
                "drops for this reason are counted on `buffer_evicted_span_count` and logged at "
                "debug level.",
                reason,
                self._buffer_evicted_span_count,
            )
        else:
            logger.debug(
                "Dropped %d more completed span(s) (%s); %d total",
                count,
                reason,
                self._buffer_evicted_span_count,
            )

    def discard_exported(self, spans: tuple[SpanRecord, ...]) -> None:
        """Forget spans that a consumer has successfully taken (#187).

        Removes exactly the given records, so a span completed between the caller's read
        and this call survives — which `clear()` cannot promise. Not counted as a drop:
        these were consumed, not lost.

        Paired with `drop_unexported` so a caller cannot express "stop tracking these"
        without saying whether they were delivered. That pairing is the fix for the real
        defect behind #187's CLI half: `cli/commands/run.py` called `clear()` in the
        REPL's `finally` block **after** a failed `export_spans`, so a collector outage
        discarded every span of that turn with nothing but an export warning to show for
        it.
        """
        exported = {span.span_id for span in spans}
        if not exported:
            return
        kept = [span for span in self._completed_spans if span.span_id not in exported]
        self._completed_spans.clear()
        self._completed_spans.extend(kept)

    def drop_unexported(self, spans: tuple[SpanRecord, ...], reason: str) -> None:
        """Forget spans that were *not* delivered, and account for the loss (#187).

        The deliberate choice here is at-most-once over at-least-once: a failed export
        drops rather than retrying, because retrying re-sends spans a collector may
        already have accepted, and "a backend receiving the duplicates cannot distinguish
        a re-export from a real repeat" (#187). Dropping is the honest half only if it is
        counted, which is what `buffer_evicted_span_count` is for.
        """
        present = {span.span_id for span in self._completed_spans}
        losing = tuple(span for span in spans if span.span_id in present)
        self.discard_exported(spans)
        self._record_drop(losing, reason)

    def clear(self) -> None:
        """Clear all active and completed spans, counting both as dropped (#205).

        A reset, not a consume: anything still buffered or in flight is being thrown
        away, so it is accounted for. Prefer `discard_exported` / `drop_unexported`, which
        say what happened to the spans.
        """
        self._record_drop(tuple(self._completed_spans), "cleared")
        self._record_drop(tuple(self._active_spans.values()), "cleared_in_flight")
        self._active_spans.clear()
        self._completed_spans.clear()


Tracer = TelemetryTracer

__all__ = [
    "DEFAULT_MAX_COMPLETED_SPANS",
    "DEFAULT_MAX_RETAINED_DROPPED_FAILOVER_IDS",
    "DEFAULT_MAX_SUBSCRIBER_QUEUE_SPANS",
    "FAILOVER_EVENT_SPAN_NAME",
    "FailoverSpanAttribution",
    "SpanFate",
    "TelemetryTracer",
    "Tracer",
]
