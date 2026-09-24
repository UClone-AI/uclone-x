"""OpenTelemetry OTLP exporter for spans and metrics over HTTP."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, cast

import httpx

import uclone_x
from uclone_x.errors import TelemetryExportError
from uclone_x.telemetry.exporter import redact_span_attributes
from uclone_x.telemetry.models import (
    MetricKind,
    MetricRecord,
    SpanKind,
    SpanRecord,
    SpanStatus,
)
from uclone_x.telemetry.protocols import TelemetryExporterProtocol

DEFAULT_OTLP_ENDPOINT: str = "http://localhost:4318"
DEFAULT_SERVICE_NAME: str = "uclone-x"
DEFAULT_TIMEOUT_SECONDS: float = 10.0


def _instrumentation_scope(name: str) -> dict[str, str]:
    """Build an OTLP InstrumentationScope stamped with this package's live version.

    The version is read as `uclone_x.__version__` through the module rather than via
    `from uclone_x import __version__`. An import-time binding reports the same
    string but is a second binding no test can move, so nothing could demonstrate
    that the exported scope follows the declaration (#1131, following #1121).
    """
    return {"name": name, "version": uclone_x.__version__}


def _parse_header_string(header_str: str) -> dict[str, str]:
    """Parse comma-separated key=value pairs into a header dictionary."""
    headers: dict[str, str] = {}
    for item in header_str.split(","):
        stripped = item.strip()
        if not stripped or "=" not in stripped:
            continue
        key, val = stripped.split("=", 1)
        headers[key.strip()] = val.strip()
    return headers


def _span_kind_to_otlp_int(kind: SpanKind) -> int:
    """Map SpanKind to OTLP Span.SpanKind integer."""
    match kind:
        case SpanKind.INTERNAL:
            return 1
        case SpanKind.SERVER:
            return 2
        case SpanKind.CLIENT:
            return 3
        case SpanKind.PRODUCER:
            return 4
        case SpanKind.CONSUMER:
            return 5
        case _:
            return 1


def _span_status_to_otlp_code(status: SpanStatus) -> int:
    """Map SpanStatus to OTLP Status.StatusCode integer."""
    match status:
        case SpanStatus.UNSET:
            return 0
        case SpanStatus.OK:
            return 1
        case SpanStatus.ERROR:
            return 2
        case _:
            return 0


def _value_to_otlp_any_value(val: object) -> dict[str, Any]:
    """Convert a Python value to an OTLP AnyValue dictionary."""
    if isinstance(val, bool):
        return {"boolValue": val}
    if isinstance(val, int):
        return {"intValue": str(val)}
    if isinstance(val, float):
        return {"doubleValue": float(val)}
    if isinstance(val, str):
        return {"stringValue": val}
    if isinstance(val, (list, tuple)):
        sequence = cast("list[object] | tuple[object, ...]", val)
        return {"arrayValue": {"values": [_value_to_otlp_any_value(item) for item in sequence]}}
    if isinstance(val, Mapping):
        mapping = cast(Mapping[object, object], val)
        return {
            "kvlistValue": {
                "values": [
                    {"key": str(k), "value": _value_to_otlp_any_value(v)}
                    for k, v in mapping.items()
                ]
            }
        }
    if val is None:
        return {"stringValue": "null"}
    return {"stringValue": str(val)}


def _attributes_to_otlp_key_values(attributes: Mapping[str, object]) -> list[dict[str, Any]]:
    """Convert an attributes mapping to a list of OTLP KeyValue dictionaries."""
    return [{"key": str(k), "value": _value_to_otlp_any_value(v)} for k, v in attributes.items()]


class OTLPTelemetryExporter(TelemetryExporterProtocol):
    """OpenTelemetry OTLP exporter transmitting spans and metrics to an OTel Collector over HTTP."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        traces_endpoint: str | None = None,
        metrics_endpoint: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
        export_full_payloads: bool = False,
        redact_secrets: bool = True,
        service_name: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._export_full_payloads: bool = export_full_payloads
        self._redact_secrets: bool = redact_secrets
        self._http_client: httpx.AsyncClient | None = http_client

        # Service name resolution
        self._service_name: str = (
            service_name or os.environ.get("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME
        )

        # Timeout resolution
        if timeout_seconds is not None:
            self._timeout_seconds: float = timeout_seconds
        else:
            env_timeout = os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT")
            if env_timeout:
                try:
                    self._timeout_seconds = float(env_timeout)
                except ValueError:
                    self._timeout_seconds = DEFAULT_TIMEOUT_SECONDS
            else:
                self._timeout_seconds = DEFAULT_TIMEOUT_SECONDS

        # Base endpoint resolution
        base_endpoint = (
            endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or DEFAULT_OTLP_ENDPOINT
        ).rstrip("/")
        self._endpoint: str = base_endpoint

        # Traces endpoint resolution
        if traces_endpoint:
            self._traces_endpoint: str = traces_endpoint
        elif os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
            self._traces_endpoint = os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        elif self._endpoint.endswith("/v1/traces"):
            self._traces_endpoint = self._endpoint
        elif self._endpoint.endswith("/v1/metrics"):
            self._traces_endpoint = self._endpoint[:-11] + "/v1/traces"
        else:
            self._traces_endpoint = f"{self._endpoint}/v1/traces"

        # Metrics endpoint resolution
        if metrics_endpoint:
            self._metrics_endpoint: str = metrics_endpoint
        elif os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
            self._metrics_endpoint = os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"]
        elif self._endpoint.endswith("/v1/metrics"):
            self._metrics_endpoint = self._endpoint
        elif self._endpoint.endswith("/v1/traces"):
            self._metrics_endpoint = self._endpoint[:-10] + "/v1/metrics"
        else:
            self._metrics_endpoint = f"{self._endpoint}/v1/metrics"

        # Headers resolution
        resolved_headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        env_headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
        if env_headers:
            resolved_headers.update(_parse_header_string(env_headers))
        if headers:
            resolved_headers.update(headers)
        self._headers: dict[str, str] = resolved_headers

    @property
    def endpoint(self) -> str:
        """Return base OTLP endpoint."""
        return self._endpoint

    @property
    def traces_endpoint(self) -> str:
        """Return OTLP HTTP traces ingestion endpoint."""
        return self._traces_endpoint

    @property
    def metrics_endpoint(self) -> str:
        """Return OTLP HTTP metrics ingestion endpoint."""
        return self._metrics_endpoint

    @property
    def headers(self) -> Mapping[str, str]:
        """Return HTTP headers sent with OTLP export requests."""
        return self._headers

    @property
    def timeout_seconds(self) -> float:
        """Return request timeout in seconds."""
        return self._timeout_seconds

    @property
    def service_name(self) -> str:
        """Return configured service name for OTel resource envelope."""
        return self._service_name

    @property
    def export_full_payloads(self) -> bool:
        """Return True if full payloads are exported without redaction."""
        return self._export_full_payloads

    @property
    def redact_secrets(self) -> bool:
        """Return True if secrets are redacted from attributes."""
        return self._redact_secrets

    def serialize_spans(self, spans: tuple[SpanRecord, ...]) -> dict[str, Any]:
        """Serialize SpanRecord objects into standard OTLP ResourceSpans JSON structure.

        The `service.version` resource attribute and the tracer scope both report
        `uclone_x.__version__`; see `_instrumentation_scope` for why it is read
        through the module (#1131).
        """
        otlp_spans: list[dict[str, Any]] = []

        for span in spans:
            redacted_attrs = redact_span_attributes(
                span.attributes,
                export_full_payloads=self._export_full_payloads,
                redact_secrets=self._redact_secrets,
            )
            otlp_span: dict[str, Any] = {
                "traceId": span.trace_id,
                "spanId": span.span_id,
                "parentSpanId": span.parent_span_id or "",
                "name": span.name,
                "kind": _span_kind_to_otlp_int(span.kind),
                "startTimeUnixNano": str(span.start_time_ns),
                "endTimeUnixNano": str(span.end_time_ns),
                "attributes": _attributes_to_otlp_key_values(redacted_attrs),
                "status": {
                    "code": _span_status_to_otlp_code(span.status),
                    "message": span.error_message or "",
                },
            }
            otlp_spans.append(otlp_span)

        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": self._service_name},
                            },
                            {
                                "key": "service.version",
                                "value": {"stringValue": uclone_x.__version__},
                            },
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": _instrumentation_scope("uclone-x.tracer"),
                            "spans": otlp_spans,
                        }
                    ],
                }
            ]
        }

    def serialize_metrics(self, metrics: tuple[MetricRecord, ...]) -> dict[str, Any]:
        """Serialize MetricRecord objects into standard OTLP ResourceMetrics JSON structure.

        The meter scope reports `uclone_x.__version__`; see `_instrumentation_scope`.
        """
        otlp_metrics: list[dict[str, Any]] = []

        for m in metrics:
            dp_attributes = _attributes_to_otlp_key_values(m.attributes)
            metric_entry: dict[str, Any] = {
                "name": m.name,
                "unit": m.unit,
            }
            if m.kind == MetricKind.COUNTER:
                metric_entry["sum"] = {
                    "dataPoints": [
                        {
                            "timeUnixNano": str(m.timestamp_ns),
                            "asDouble": float(m.value),
                            "attributes": dp_attributes,
                        }
                    ],
                    "isMonotonic": True,
                    "aggregationTemporality": 2,
                }
            elif m.kind == MetricKind.GAUGE:
                metric_entry["gauge"] = {
                    "dataPoints": [
                        {
                            "timeUnixNano": str(m.timestamp_ns),
                            "asDouble": float(m.value),
                            "attributes": dp_attributes,
                        }
                    ]
                }
            elif m.kind == MetricKind.HISTOGRAM:
                metric_entry["histogram"] = {
                    "dataPoints": [
                        {
                            "timeUnixNano": str(m.timestamp_ns),
                            "count": 1,
                            "sum": float(m.value),
                            "attributes": dp_attributes,
                        }
                    ],
                    "aggregationTemporality": 2,
                }
            otlp_metrics.append(metric_entry)

        return {
            "resourceMetrics": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": self._service_name},
                            }
                        ]
                    },
                    "scopeMetrics": [
                        {
                            "scope": _instrumentation_scope("uclone-x.metrics"),
                            "metrics": otlp_metrics,
                        }
                    ],
                }
            ]
        }

    async def export_spans(self, spans: tuple[SpanRecord, ...]) -> None:
        """Export span records to OpenTelemetry Collector traces endpoint."""
        if not spans:
            return

        payload = self.serialize_spans(spans)
        try:
            if self._http_client is not None:
                resp = await self._http_client.post(
                    self._traces_endpoint,
                    json=payload,
                    headers=self._headers,
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        self._traces_endpoint,
                        json=payload,
                        headers=self._headers,
                        timeout=self._timeout_seconds,
                    )
        except Exception as exc:
            raise TelemetryExportError(
                f"OTLP span export failed to {self._traces_endpoint}: {exc}"
            ) from exc

        if resp.status_code >= 400:
            raise TelemetryExportError(
                f"OTLP span export rejected by {self._traces_endpoint} with HTTP {resp.status_code}: {resp.text}"
            )

    async def export_metrics(self, metrics: tuple[MetricRecord, ...]) -> None:
        """Export metric records to OpenTelemetry Collector metrics endpoint."""
        if not metrics:
            return

        payload = self.serialize_metrics(metrics)
        try:
            if self._http_client is not None:
                resp = await self._http_client.post(
                    self._metrics_endpoint,
                    json=payload,
                    headers=self._headers,
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        self._metrics_endpoint,
                        json=payload,
                        headers=self._headers,
                        timeout=self._timeout_seconds,
                    )
        except Exception as exc:
            raise TelemetryExportError(
                f"OTLP metrics export failed to {self._metrics_endpoint}: {exc}"
            ) from exc

        if resp.status_code >= 400:
            raise TelemetryExportError(
                f"OTLP metrics export rejected by {self._metrics_endpoint} with HTTP {resp.status_code}: {resp.text}"
            )


__all__ = [
    "DEFAULT_OTLP_ENDPOINT",
    "DEFAULT_SERVICE_NAME",
    "DEFAULT_TIMEOUT_SECONDS",
    "OTLPTelemetryExporter",
]
