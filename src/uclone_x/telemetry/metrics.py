"""Metrics collection and recording implementation for UClone-X."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping

from uclone_x.telemetry.models import MetricKind, MetricRecord
from uclone_x.telemetry.protocols import MetricRecorderProtocol


class MetricsCollector(MetricRecorderProtocol):
    """Metrics collector buffer recording counters, gauges, and histograms."""

    def __init__(self, default_attributes: Mapping[str, str] | None = None) -> None:
        self._default_attributes: dict[str, str] = (
            dict(default_attributes) if default_attributes is not None else {}
        )
        self._records: list[MetricRecord] = []
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        """Return the number of buffered metric measurements."""
        with self._lock:
            return len(self._records)

    def record(
        self,
        name: str,
        value: float,
        kind: MetricKind,
        unit: str = "1",
        attributes: Mapping[str, str] | None = None,
    ) -> MetricRecord:
        """Record one metric measurement and append it to the buffer."""
        merged_attrs: dict[str, str] = dict(self._default_attributes)
        if attributes is not None:
            merged_attrs.update({k: str(v) for k, v in attributes.items()})

        record = MetricRecord(
            name=name,
            kind=kind,
            value=float(value),
            unit=unit,
            timestamp_ns=time.time_ns(),
            attributes=merged_attrs,
        )
        with self._lock:
            self._records.append(record)
        return record

    def drain(self) -> tuple[MetricRecord, ...]:
        """Take all buffered metric measurements and clear the internal buffer."""
        with self._lock:
            buffered = tuple(self._records)
            self._records.clear()
            return buffered

    def get_records(self) -> tuple[MetricRecord, ...]:
        """Inspect all buffered metric measurements without clearing them."""
        with self._lock:
            return tuple(self._records)

    def clear(self) -> None:
        """Clear all buffered metric measurements."""
        with self._lock:
            self._records.clear()

    def record_token_count(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
        extra_attributes: Mapping[str, str] | None = None,
    ) -> tuple[MetricRecord, MetricRecord]:
        """Record input and output token count counters for an LLM call."""
        attrs: dict[str, str] = {
            "provider": provider,
            "model": model,
        }
        if extra_attributes is not None:
            attrs.update(extra_attributes)

        input_attrs = dict(attrs)
        input_attrs["token_type"] = "input"
        rec_in = self.record(
            name="tokens.input",
            value=float(input_tokens),
            kind=MetricKind.COUNTER,
            unit="tokens",
            attributes=input_attrs,
        )

        output_attrs = dict(attrs)
        output_attrs["token_type"] = "output"
        rec_out = self.record(
            name="tokens.output",
            value=float(output_tokens),
            kind=MetricKind.COUNTER,
            unit="tokens",
            attributes=output_attrs,
        )
        return rec_in, rec_out

    def record_execution_latency(
        self,
        operation_name: str,
        duration_ms: float,
        attributes: Mapping[str, str] | None = None,
    ) -> MetricRecord:
        """Record an execution latency measurement histogram in milliseconds."""
        attrs: dict[str, str] = {"operation": operation_name}
        if attributes is not None:
            attrs.update(attributes)
        return self.record(
            name="execution.latency",
            value=float(duration_ms),
            kind=MetricKind.HISTOGRAM,
            unit="ms",
            attributes=attrs,
        )

    def record_error(
        self,
        error_type: str,
        operation_name: str,
        attributes: Mapping[str, str] | None = None,
    ) -> MetricRecord:
        """Record an error occurrence counter."""
        attrs: dict[str, str] = {
            "error_type": error_type,
            "operation": operation_name,
        }
        if attributes is not None:
            attrs.update(attributes)
        return self.record(
            name="errors.count",
            value=1.0,
            kind=MetricKind.COUNTER,
            unit="1",
            attributes=attrs,
        )

    def record_memory_usage(
        self,
        bytes_used: float,
        attributes: Mapping[str, str] | None = None,
    ) -> MetricRecord:
        """Record a memory usage gauge in bytes."""
        return self.record(
            name="memory.usage",
            value=float(bytes_used),
            kind=MetricKind.GAUGE,
            unit="bytes",
            attributes=attributes,
        )


MetricCollector = MetricsCollector
MetricRecorder = MetricsCollector

__all__ = [
    "MetricCollector",
    "MetricRecorder",
    "MetricsCollector",
]
