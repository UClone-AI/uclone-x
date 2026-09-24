"""In-memory telemetry exporter with credential and payload redaction."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

import httpx

from uclone_x.core.secrets import is_secret_env_name, redact_credentials
from uclone_x.errors import TelemetryExportError
from uclone_x.telemetry.models import MetricRecord, SpanRecord
from uclone_x.telemetry.protocols import TelemetryExporterProtocol

SENSITIVE_KEY_SUBSTRINGS: tuple[str, ...] = (
    "api_key",
    "token",
    "secret",
    "password",
    "credentials",
    "authorization",
    "auth",
    "private_key",
    "bearer",
)
"""Substrings that make a *span attribute key* sensitive, on top of the env predicate.

Deliberately broader than `uclone_x.sandbox.models.SECRET_NAME_TAILS`: bare `token` and
`auth` here will redact patterns like `git.author` and `authorization`, which costs one
`[REDACTED]` in a trace. The same breadth in the environment predicate would be a hard
construction refusal of `GIT_AUTHOR_NAME` — a broken tool. Operational token count and
performance metrics (e.g. `llm.max_tokens`, `input_tokens`, `token_count`) are explicitly
exempted via `is_benign_metric_attribute_key` after real credentials have been caught by
`is_secret_env_name`. The two lists therefore differ in breadth on purpose while sharing one
call site, `is_sensitive_attribute_key`, so a credential too sensitive to appear in a trace can
never be *less* than too sensitive to inherit into an untrusted child process.
"""

BENIGN_METRIC_EXACT_KEYS: tuple[str, ...] = (
    "token_count",
    "tokens_per_second",
    "num_tokens",
    "tokens",
)
"""Exact attribute key names representing token counts or performance metrics."""

PAYLOAD_ATTRIBUTE_KEYS: tuple[str, ...] = (
    "prompt",
    "completion",
    "messages",
    "tool_arguments",
    "tool_result",
    "arguments",
    "result",
    "stdout",
    "stderr",
    "input_data",
    "output_data",
    "content",
    "payload",
)

REDACTED_PLACEHOLDER: str = "[REDACTED]"

DEPTH_LIMITED_PLACEHOLDER: str = "[DEPTH-LIMITED]"
"""Stands in for a subtree the walk refused to descend into.

Distinct from `REDACTED_PLACEHOLDER` on purpose: a reader of the trace must be able to
tell "this was a credential" from "this was never inspected". They are different
statements about the exported span, and a single placeholder would conflate them.
"""

MAX_REDACTION_DEPTH: int = 8
"""How much nested structure the exporter will inspect. A policy cap, not a safety net.

This is a deliberate choice about how much of an open-ended attribute tree is worth
walking before export, set far below anything the runtime would refuse on its own. Its
one observable effect: a span nesting at or beyond this depth is truncated, and the
truncation **withholds rather than emits**, so the failure mode is missing data and
never a leak.

It is explicitly *not* a guard against unbounded recursion reaching this function,
because nothing that reaches an exporter can be unbounded. `SpanRecord.attributes` is
`ImmutableJsonMapping`, and pydantic-core's recursion guard rejects both a cyclic
mapping and an over-deep one at validation, before `freeze_mapping` or this function
ever runs. Cycle tolerance and very-deep tolerance are therefore properties of this
public helper's own contract — reachable only by calling it directly and bypassing
`SpanRecord` — and not controls on the export path.

That distinction has now invalidated two successive rationales written for this
constant, one about cycles and one about the recursion limit, both of which asserted a
threat this validator had already eliminated. The general lesson is worth more than the
constant: `SpanRecord` construction is a far stronger gate than code written about the
export path tends to assume, so a claim about what can reach an exporter should be
measured against the model, not reasoned from the annotation. (Measured, not
contractual: the validator's own ceiling sits at 254 levels on pydantic-core today,
fixed regardless of `sys.setrecursionlimit`.)
"""


def is_benign_metric_attribute_key(key: str) -> bool:
    """Return True if an attribute key represents a token count or performance metric.

    Recognizes operational metrics (e.g., keys ending in `_tokens`, namespaced metric
    keys like `llm.input_tokens`, and exact terms like `token_count` or `tokens_per_second`)
    so they survive span redaction instead of being matched by the bare `token` substring.
    """
    lower = key.lower()
    return (
        lower.endswith("_tokens")
        or lower in BENIGN_METRIC_EXACT_KEYS
        or any(lower.endswith(f".{term}") for term in BENIGN_METRIC_EXACT_KEYS)
    )


def is_sensitive_attribute_key(key: str) -> bool:
    """Return True if an attribute key matches credential or secret naming patterns."""
    if is_secret_env_name(key):
        return True
    if is_benign_metric_attribute_key(key):
        return False
    lower = key.lower()
    return any(sub in lower for sub in SENSITIVE_KEY_SUBSTRINGS)


def is_payload_attribute_key(key: str) -> bool:
    """Return True if an attribute key is a prompt/completion or tool argument/result payload."""
    lower = key.lower()
    return lower in PAYLOAD_ATTRIBUTE_KEYS or any(
        lower.endswith(f".{p}") for p in PAYLOAD_ATTRIBUTE_KEYS
    )


def _redact_entry(
    key: str,
    value: object,
    *,
    export_full_payloads: bool,
    redact_secrets: bool,
    depth: int,
) -> object:
    """Decide one `key -> value` entry, at any depth.

    Both predicates apply at every depth. The nested case is the *common* one, not an
    edge case: `{"llm": {"prompt": ...}}` is the ordinary shape of an `llm.generate`
    span, and checking only the credential predicate on nested keys (as this function's
    predecessor did) exported that prompt verbatim.
    """
    if redact_secrets and is_sensitive_attribute_key(key):
        return REDACTED_PLACEHOLDER
    if not export_full_payloads and is_payload_attribute_key(key):
        return REDACTED_PLACEHOLDER
    return _redact_value(
        value,
        export_full_payloads=export_full_payloads,
        redact_secrets=redact_secrets,
        depth=depth,
    )


def _redact_value(
    value: object,
    *,
    export_full_payloads: bool,
    redact_secrets: bool,
    depth: int,
) -> object:
    """Walk a value whose own key has already been cleared, rebuilding plain containers."""
    if depth >= MAX_REDACTION_DEPTH:
        if isinstance(value, (Mapping, list, tuple)):
            return DEPTH_LIMITED_PLACEHOLDER
        return value

    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(sub_key): _redact_entry(
                str(sub_key),
                sub_value,
                export_full_payloads=export_full_payloads,
                redact_secrets=redact_secrets,
                depth=depth + 1,
            )
            for sub_key, sub_value in mapping.items()
        }

    if isinstance(value, (list, tuple)):
        sequence = cast("list[object] | tuple[object, ...]", value)
        return [
            _redact_value(
                item,
                export_full_payloads=export_full_payloads,
                redact_secrets=redact_secrets,
                depth=depth + 1,
            )
            for item in sequence
        ]

    if isinstance(value, str) and redact_secrets:
        return redact_credentials(value)

    return value


def redact_span_attributes(
    attributes: Mapping[str, Any],
    *,
    export_full_payloads: bool = False,
    redact_secrets: bool = True,
) -> dict[str, Any]:
    """Scrub sensitive credentials and payload bodies from span attributes before export.

    Recurses through mappings, lists and tuples, applying the credential predicate and
    the payload predicate to every key at every depth, and rebuilding plain `dict` and
    `list` on the way out. Beyond `MAX_REDACTION_DEPTH` nested containers are withheld
    behind `DEPTH_LIMITED_PLACEHOLDER`.
    """
    return {
        key: _redact_entry(
            key,
            value,
            export_full_payloads=export_full_payloads,
            redact_secrets=redact_secrets,
            depth=1,
        )
        for key, value in attributes.items()
    }


class InMemoryTelemetryExporter(TelemetryExporterProtocol):
    """In-memory telemetry sink collecting spans and metrics with mandatory scrubbing."""

    def __init__(
        self,
        *,
        export_full_payloads: bool = False,
        redact_secrets: bool = True,
        fail_on_export: bool = False,
    ) -> None:
        self._export_full_payloads: bool = export_full_payloads
        self._redact_secrets: bool = redact_secrets
        self._fail_on_export: bool = fail_on_export
        self._spans: list[SpanRecord] = []
        self._metrics: list[MetricRecord] = []

    @property
    def export_full_payloads(self) -> bool:
        """Return True if raw payloads are allowed to be exported."""
        return self._export_full_payloads

    @property
    def redact_secrets(self) -> bool:
        """Return True if secrets are redacted on export."""
        return self._redact_secrets

    @property
    def span_count(self) -> int:
        """Return the number of exported spans."""
        return len(self._spans)

    @property
    def metric_count(self) -> int:
        """Return the number of exported metrics."""
        return len(self._metrics)

    async def export_spans(self, spans: tuple[SpanRecord, ...]) -> None:
        """Sanitize and record spans in memory. Raises TelemetryExportError if fail_on_export is True."""
        if self._fail_on_export:
            raise TelemetryExportError("Telemetry span export failed (fail_on_export is True)")

        for span in spans:
            redacted_attrs = redact_span_attributes(
                span.attributes,
                export_full_payloads=self._export_full_payloads,
                redact_secrets=self._redact_secrets,
            )
            clean_span = SpanRecord(
                trace_id=span.trace_id,
                span_id=span.span_id,
                name=span.name,
                parent_span_id=span.parent_span_id,
                kind=span.kind,
                start_time_ns=span.start_time_ns,
                end_time_ns=span.end_time_ns,
                attributes=redacted_attrs,
                status=span.status,
                error_message=span.error_message,
            )
            self._spans.append(clean_span)

    async def export_metrics(self, metrics: tuple[MetricRecord, ...]) -> None:
        """Record metrics in memory. Raises TelemetryExportError if fail_on_export is True."""
        if self._fail_on_export:
            raise TelemetryExportError("Telemetry metrics export failed (fail_on_export is True)")

        self._metrics.extend(metrics)

    def get_exported_spans(self) -> tuple[SpanRecord, ...]:
        """Return all spans exported to this sink."""
        return tuple(self._spans)

    def get_exported_metrics(self) -> tuple[MetricRecord, ...]:
        """Return all metrics exported to this sink."""
        return tuple(self._metrics)

    def clear(self) -> None:
        """Clear all exported spans and metrics."""
        self._spans.clear()
        self._metrics.clear()


class CompositeTelemetryExporter(TelemetryExporterProtocol):
    """Composite telemetry exporter broadcasting spans and metrics to multiple exporters."""

    def __init__(self, exporters: Sequence[TelemetryExporterProtocol]) -> None:
        self._exporters: tuple[TelemetryExporterProtocol, ...] = tuple(exporters)

    @property
    def exporters(self) -> tuple[TelemetryExporterProtocol, ...]:
        """Return the sequence of registered child exporters."""
        return self._exporters

    async def export_spans(self, spans: tuple[SpanRecord, ...]) -> None:
        """Export spans across all configured child exporters. Raises if any exporter fails."""
        if not spans or not self._exporters:
            return
        errors: list[Exception] = []
        for exporter in self._exporters:
            try:
                await exporter.export_spans(spans)
            except Exception as exc:
                errors.append(exc)
        if errors:
            if len(errors) == 1:
                raise errors[0]
            err_msgs = "; ".join(str(e) for e in errors)
            raise TelemetryExportError(
                f"Multiple telemetry exporters failed to export spans ({len(errors)}): {err_msgs}"
            )

    async def export_metrics(self, metrics: tuple[MetricRecord, ...]) -> None:
        """Export metrics across all configured child exporters. Raises if any exporter fails."""
        if not metrics or not self._exporters:
            return
        errors: list[Exception] = []
        for exporter in self._exporters:
            try:
                await exporter.export_metrics(metrics)
            except Exception as exc:
                errors.append(exc)
        if errors:
            if len(errors) == 1:
                raise errors[0]
            err_msgs = "; ".join(str(e) for e in errors)
            raise TelemetryExportError(
                f"Multiple telemetry exporters failed to export metrics ({len(errors)}): {err_msgs}"
            )


def create_telemetry_exporter(
    *,
    export_full_payloads: bool = False,
    redact_secrets: bool = True,
    otlp_endpoint: str | None = None,
    otlp_traces_endpoint: str | None = None,
    otlp_metrics_endpoint: str | None = None,
    otlp_headers: Mapping[str, str] | None = None,
    langfuse_public_key: str | None = None,
    langfuse_secret_key: str | None = None,
    langfuse_host: str | None = None,
    service_name: str | None = None,
    fail_on_export: bool = False,
    http_client: httpx.AsyncClient | None = None,
    use_otel_sdk: bool = False,
) -> TelemetryExporterProtocol:
    """Auto-detect telemetry configuration from environment variables and return appropriate exporter.

    Detects OTLP Collector configuration from `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`,
    and Langfuse from `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`. If multiple backends are detected,
    a `CompositeTelemetryExporter` is returned. If neither is detected, falls back to `InMemoryTelemetryExporter`.
    """
    if use_otel_sdk:
        from uclone_x.telemetry.otel_sdk import create_opentelemetry_sdk_exporter

        return cast(
            TelemetryExporterProtocol,
            create_opentelemetry_sdk_exporter(endpoint=otlp_endpoint),
        )

    from uclone_x.telemetry.langfuse import LangfuseTelemetryExporter
    from uclone_x.telemetry.otlp import OTLPTelemetryExporter

    # Check for OTLP configuration
    has_otlp = bool(
        otlp_endpoint
        or otlp_traces_endpoint
        or otlp_metrics_endpoint
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )

    # Check for Langfuse configuration
    lf_public = langfuse_public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    lf_secret = langfuse_secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    has_langfuse = bool(lf_public and lf_secret)

    configured_exporters: list[TelemetryExporterProtocol] = []

    if has_otlp:
        configured_exporters.append(
            OTLPTelemetryExporter(
                endpoint=otlp_endpoint,
                traces_endpoint=otlp_traces_endpoint,
                metrics_endpoint=otlp_metrics_endpoint,
                headers=otlp_headers,
                export_full_payloads=export_full_payloads,
                redact_secrets=redact_secrets,
                service_name=service_name,
                http_client=http_client,
            )
        )

    if has_langfuse:
        configured_exporters.append(
            LangfuseTelemetryExporter(
                public_key=lf_public,
                secret_key=lf_secret,
                host=langfuse_host,
                export_full_payloads=export_full_payloads,
                redact_secrets=redact_secrets,
                http_client=http_client,
            )
        )

    if not configured_exporters:
        return InMemoryTelemetryExporter(
            export_full_payloads=export_full_payloads,
            redact_secrets=redact_secrets,
            fail_on_export=fail_on_export,
        )
    if len(configured_exporters) == 1:
        return configured_exporters[0]
    return CompositeTelemetryExporter(configured_exporters)


InMemoryExporter = InMemoryTelemetryExporter
TelemetryExporter = InMemoryTelemetryExporter

__all__ = [
    "BENIGN_METRIC_EXACT_KEYS",
    "CompositeTelemetryExporter",
    "DEPTH_LIMITED_PLACEHOLDER",
    "InMemoryExporter",
    "InMemoryTelemetryExporter",
    "MAX_REDACTION_DEPTH",
    "PAYLOAD_ATTRIBUTE_KEYS",
    "REDACTED_PLACEHOLDER",
    "SENSITIVE_KEY_SUBSTRINGS",
    "TelemetryExporter",
    "create_telemetry_exporter",
    "is_benign_metric_attribute_key",
    "is_payload_attribute_key",
    "is_sensitive_attribute_key",
    "redact_span_attributes",
]
