"""Protocols for OpenTelemetry tracers, metric recorders, and exporters.

`@runtime_checkable` is applied only where a runtime `isinstance` check is actually
performed; structural conformance is enforced statically by the bindings in
`tests/unit/test_protocol_conformance.py` (issue 2026-09-02-035).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from uclone_x.telemetry.models import (
    MetricKind,
    MetricRecord,
    SpanKind,
    SpanRecord,
    SpanStatus,
)


@runtime_checkable
class TraceRecorderProtocol(Protocol):
    """Protocol for recording nested agent turns and tool execution spans."""

    @property
    def trace_id(self) -> str:
        """Return the root trace ID for this tracer."""
        ...

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: str | None = None,
        attributes: Mapping[str, JsonValue] | None = None,
    ) -> str:
        """Start a new span and return its span_id."""
        ...

    def end_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.OK,
        error_message: str | None = None,
    ) -> SpanRecord | None:
        """Complete a span."""
        ...

    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: str | None = None,
        attributes: Mapping[str, JsonValue] | None = None,
    ) -> AbstractAsyncContextManager[str]:
        """Scope a span to an `async with` block, yielding the span_id.

        Typed as an async context manager, not as an `AsyncIterator`: the previous
        annotation could not be used with `async with` at all, so the documented usage
        was impossible (issue 2026-09-02-036).
        """
        ...


@runtime_checkable
class MetricRecorderProtocol(Protocol):
    """Protocol for producing metric measurements.

    `MetricRecord` and `export_metrics` existed with nothing in between: the module
    advertised "metrics collectors" while offering no way to record a measurement.
    """

    def record(
        self,
        name: str,
        value: float,
        kind: MetricKind,
        unit: str = "1",
        attributes: Mapping[str, str] | None = None,
    ) -> MetricRecord:
        """Record one measurement and return the record produced."""
        ...

    def drain(self) -> tuple[MetricRecord, ...]:
        """Take the buffered measurements, leaving the buffer empty."""
        ...


@runtime_checkable
class TelemetryExporterProtocol(Protocol):
    """Protocol for exporting spans and metrics to Langfuse, Jaeger, or an OTLP collector."""

    async def export_spans(self, spans: tuple[SpanRecord, ...]) -> None:
        """Push span records to remote observability backends.

        Returns None and raises on failure. The previous `-> bool` gave a caller no way
        to know why an export failed, and a discarded False is a silent loss of the
        very trace P6 relies on.
        """
        ...

    async def export_metrics(self, metrics: tuple[MetricRecord, ...]) -> None:
        """Push metric measurements to remote collectors, raising on failure."""
        ...


@runtime_checkable
class SpanStreamProtocol(Protocol):
    """Protocol for consuming spans as they complete, for the developer UI."""

    def stream_spans(self) -> AsyncIterator[SpanRecord]:
        """Yield span records as they are completed."""
        ...


TracerProtocol = TraceRecorderProtocol
MetricsCollectorProtocol = MetricRecorderProtocol

__all__ = [
    "MetricRecorderProtocol",
    "MetricsCollectorProtocol",
    "SpanStreamProtocol",
    "TelemetryExporterProtocol",
    "TraceRecorderProtocol",
    "TracerProtocol",
]
