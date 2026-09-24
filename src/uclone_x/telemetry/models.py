"""Data models for OpenTelemetry tracing, metrics, and failover observability."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.core.immutable import ImmutableJsonMapping, ImmutableStrMapping


class SpanKind(StrEnum):
    """OpenTelemetry span category."""

    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class SpanStatus(StrEnum):
    """OpenTelemetry span status, as an enum rather than a commented string."""

    OK = "OK"
    ERROR = "ERROR"
    UNSET = "UNSET"


class MetricKind(StrEnum):
    """Instrument kind a measurement came from."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class SpanRecord(BaseModel):
    """Trace span data envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    trace_id: str
    span_id: str
    name: str
    parent_span_id: str | None = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time_ns: int = 0
    end_time_ns: int = 0
    attributes: ImmutableJsonMapping = Field(default_factory=dict)
    status: SpanStatus = SpanStatus.UNSET
    error_message: str | None = None


class MetricRecord(BaseModel):
    """Telemetry counter / gauge / histogram measurement."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    kind: MetricKind
    value: float
    unit: str = "1"
    timestamp_ns: int = 0
    attributes: ImmutableStrMapping = Field(default_factory=dict)
