"""Telemetry subsystem: OpenTelemetry tracing, span recording, and exporter integration."""

from uclone_x.telemetry.exporter import (
    BENIGN_METRIC_EXACT_KEYS,
    DEPTH_LIMITED_PLACEHOLDER,
    MAX_REDACTION_DEPTH,
    REDACTED_PLACEHOLDER,
    CompositeTelemetryExporter,
    InMemoryExporter,
    InMemoryTelemetryExporter,
    TelemetryExporter,
    create_telemetry_exporter,
    is_benign_metric_attribute_key,
    is_payload_attribute_key,
    is_sensitive_attribute_key,
    redact_span_attributes,
)
from uclone_x.telemetry.langfuse import (
    DEFAULT_LANGFUSE_HOST,
    LangfuseTelemetryExporter,
)
from uclone_x.telemetry.metrics import (
    MetricCollector,
    MetricRecorder,
    MetricsCollector,
)
from uclone_x.telemetry.models import (
    MetricKind,
    MetricRecord,
    SpanKind,
    SpanRecord,
    SpanStatus,
)
from uclone_x.telemetry.otel_sdk import (
    create_opentelemetry_sdk_exporter,
    get_opentelemetry_tracer,
    require_opentelemetry_api,
)
from uclone_x.telemetry.otlp import (
    DEFAULT_OTLP_ENDPOINT,
    DEFAULT_SERVICE_NAME,
    OTLPTelemetryExporter,
)
from uclone_x.telemetry.protocols import (
    MetricRecorderProtocol,
    MetricsCollectorProtocol,
    SpanStreamProtocol,
    TelemetryExporterProtocol,
    TraceRecorderProtocol,
    TracerProtocol,
)
from uclone_x.telemetry.tracer import (
    FAILOVER_EVENT_SPAN_NAME,
    FailoverSpanAttribution,
    SpanFate,
    TelemetryTracer,
    Tracer,
)

__all__ = [
    "BENIGN_METRIC_EXACT_KEYS",
    "CompositeTelemetryExporter",
    "DEFAULT_LANGFUSE_HOST",
    "DEFAULT_OTLP_ENDPOINT",
    "DEFAULT_SERVICE_NAME",
    "DEPTH_LIMITED_PLACEHOLDER",
    "FAILOVER_EVENT_SPAN_NAME",
    "FailoverSpanAttribution",
    "InMemoryExporter",
    "InMemoryTelemetryExporter",
    "LangfuseTelemetryExporter",
    "MAX_REDACTION_DEPTH",
    "MetricCollector",
    "MetricKind",
    "MetricRecord",
    "MetricRecorder",
    "MetricRecorderProtocol",
    "MetricsCollector",
    "MetricsCollectorProtocol",
    "OTLPTelemetryExporter",
    "REDACTED_PLACEHOLDER",
    "SpanFate",
    "SpanKind",
    "SpanRecord",
    "SpanStatus",
    "SpanStreamProtocol",
    "TelemetryExporter",
    "TelemetryExporterProtocol",
    "TelemetryTracer",
    "TraceRecorderProtocol",
    "Tracer",
    "TracerProtocol",
    "create_opentelemetry_sdk_exporter",
    "create_telemetry_exporter",
    "get_opentelemetry_tracer",
    "is_benign_metric_attribute_key",
    "is_payload_attribute_key",
    "is_sensitive_attribute_key",
    "redact_span_attributes",
    "require_opentelemetry_api",
]
