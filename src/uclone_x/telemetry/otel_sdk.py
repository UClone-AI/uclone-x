"""OpenTelemetry SDK integration and adapter contracts."""

from __future__ import annotations

import importlib
from typing import Any

from uclone_x.errors import MissingDependencyError

__all__ = [
    "create_opentelemetry_sdk_exporter",
    "get_opentelemetry_tracer",
    "require_opentelemetry_api",
]


def require_opentelemetry_api() -> Any:
    """Import and return opentelemetry.trace API, or raise MissingDependencyError (P6)."""
    try:
        otel_trace: Any = importlib.import_module("opentelemetry.trace")
        return otel_trace
    except ImportError as exc:
        raise MissingDependencyError(
            extra="telemetry",
            package="opentelemetry-api",
            feature="OpenTelemetry tracing API",
        ) from exc


def get_opentelemetry_tracer(instrumenting_module_name: str = "uclone-x") -> Any:
    """Return an OpenTelemetry SDK tracer, or raise MissingDependencyError (P6)."""
    otel_trace: Any = require_opentelemetry_api()
    tracer: Any = otel_trace.get_tracer(instrumenting_module_name)
    return tracer


def create_opentelemetry_sdk_exporter(
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """Create an OpenTelemetry OTLP span exporter using the OTel SDK, or raise MissingDependencyError (P6)."""
    try:
        exporter_mod: Any = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        )
        exporter_cls: Any = exporter_mod.OTLPSpanExporter
    except ImportError as exc:
        pkg = "opentelemetry-exporter-otlp" if "exporter" in str(exc) else "opentelemetry-sdk"
        raise MissingDependencyError(
            extra="telemetry",
            package=pkg,
            feature="OpenTelemetry OTLP SDK exporter",
        ) from exc
    exporter: Any = exporter_cls(endpoint=endpoint, headers=headers)
    return exporter
