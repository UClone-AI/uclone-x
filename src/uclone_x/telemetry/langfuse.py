"""Langfuse telemetry exporter for spans, generations, and traces over HTTP."""

from __future__ import annotations

import datetime
import os
from collections.abc import Mapping
from typing import Any

import httpx

from uclone_x.errors import TelemetryExportError
from uclone_x.telemetry.exporter import redact_span_attributes
from uclone_x.telemetry.models import (
    MetricRecord,
    SpanKind,
    SpanRecord,
    SpanStatus,
)
from uclone_x.telemetry.protocols import TelemetryExporterProtocol

DEFAULT_LANGFUSE_HOST: str = "http://localhost:3000"
DEFAULT_TIMEOUT_SECONDS: float = 10.0


def _ns_to_iso(timestamp_ns: int) -> str:
    """Convert nanosecond integer timestamp to ISO-8601 UTC string."""
    if timestamp_ns <= 0:
        return datetime.datetime.now(tz=datetime.UTC).isoformat()
    return datetime.datetime.fromtimestamp(
        timestamp_ns / 1_000_000_000.0, tz=datetime.UTC
    ).isoformat()


class LangfuseTelemetryExporter(TelemetryExporterProtocol):
    """Langfuse telemetry exporter ingesting hierarchical traces and LLM generations."""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        *,
        host: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        export_full_payloads: bool = False,
        redact_secrets: bool = True,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._public_key: str | None = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        self._secret_key: str | None = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        raw_host = (
            host
            or os.environ.get("LANGFUSE_HOST")
            or os.environ.get("LANGFUSE_BASEURL")
            or DEFAULT_LANGFUSE_HOST
        )
        self._host: str = raw_host.rstrip("/")
        self._timeout_seconds: float = timeout_seconds
        self._export_full_payloads: bool = export_full_payloads
        self._redact_secrets: bool = redact_secrets
        self._http_client: httpx.AsyncClient | None = http_client

    @property
    def public_key(self) -> str | None:
        """Return configured Langfuse public key."""
        return self._public_key

    @property
    def secret_key(self) -> str | None:
        """Return configured Langfuse secret key."""
        return self._secret_key

    @property
    def host(self) -> str:
        """Return configured Langfuse host endpoint."""
        return self._host

    @property
    def timeout_seconds(self) -> float:
        """Return request timeout in seconds."""
        return self._timeout_seconds

    @property
    def export_full_payloads(self) -> bool:
        """Return True if full payloads are exported without redaction."""
        return self._export_full_payloads

    @property
    def redact_secrets(self) -> bool:
        """Return True if secrets are redacted from attributes."""
        return self._redact_secrets

    def serialize_spans(self, spans: tuple[SpanRecord, ...]) -> dict[str, Any]:
        """Map SpanRecord hierarchy into Langfuse batch ingestion events."""
        events: list[dict[str, Any]] = []

        for span in spans:
            redacted_attrs = redact_span_attributes(
                span.attributes,
                export_full_payloads=self._export_full_payloads,
                redact_secrets=self._redact_secrets,
            )
            start_iso = _ns_to_iso(span.start_time_ns)
            end_iso = _ns_to_iso(span.end_time_ns)

            # If root span or parent is None, emit trace-create
            if span.parent_span_id is None:
                session_id = redacted_attrs.get("session_id")
                user_id = redacted_attrs.get("user_id") or redacted_attrs.get("agent_id")
                trace_event: dict[str, Any] = {
                    "id": f"evt_trc_{span.span_id}",
                    "type": "trace-create",
                    "timestamp": start_iso,
                    "body": {
                        "id": span.trace_id,
                        "name": span.name,
                        "sessionId": str(session_id) if session_id is not None else None,
                        "userId": str(user_id) if user_id is not None else None,
                        "metadata": redacted_attrs,
                        "tags": ["uclone-x"],
                    },
                }
                events.append(trace_event)

            # Check if this span is an LLM generation
            is_generation = (
                span.name == "llm.generate"
                or span.kind == SpanKind.CLIENT
                or "model" in redacted_attrs
                or "provider" in redacted_attrs
            )

            level = "ERROR" if span.status == SpanStatus.ERROR else "DEFAULT"
            status_msg = span.error_message or (
                span.status.value if span.status != SpanStatus.OK else None
            )

            if is_generation:
                model_name = (
                    redacted_attrs.get("model")
                    or redacted_attrs.get("gen_ai.request.model")
                    or "unknown"
                )
                input_tokens = redacted_attrs.get("input_tokens") or redacted_attrs.get(
                    "llm.input_tokens"
                )
                output_tokens = redacted_attrs.get("output_tokens") or redacted_attrs.get(
                    "llm.output_tokens"
                )
                total_tokens = (
                    redacted_attrs.get("total_tokens")
                    or redacted_attrs.get("tokens")
                    or redacted_attrs.get("token_count")
                )

                usage_dict: dict[str, Any] = {}
                if input_tokens is not None:
                    usage_dict["input"] = input_tokens
                if output_tokens is not None:
                    usage_dict["output"] = output_tokens
                if total_tokens is not None:
                    usage_dict["total"] = total_tokens

                gen_event: dict[str, Any] = {
                    "id": f"evt_gen_{span.span_id}",
                    "type": "generation-create",
                    "timestamp": start_iso,
                    "body": {
                        "id": span.span_id,
                        "traceId": span.trace_id,
                        "parentObservationId": span.parent_span_id,
                        "name": span.name,
                        "startTime": start_iso,
                        "endTime": end_iso,
                        "model": str(model_name),
                        "modelParameters": {
                            "temperature": redacted_attrs.get("temperature"),
                            "max_tokens": redacted_attrs.get("max_tokens"),
                        },
                        "usage": usage_dict if usage_dict else None,
                        "level": level,
                        "statusMessage": status_msg,
                        "metadata": redacted_attrs,
                        "input": redacted_attrs.get("prompt")
                        if self._export_full_payloads
                        else None,
                        "output": redacted_attrs.get("completion")
                        if self._export_full_payloads
                        else None,
                    },
                }
                events.append(gen_event)
            else:
                span_event: dict[str, Any] = {
                    "id": f"evt_spn_{span.span_id}",
                    "type": "span-create",
                    "timestamp": start_iso,
                    "body": {
                        "id": span.span_id,
                        "traceId": span.trace_id,
                        "parentObservationId": span.parent_span_id,
                        "name": span.name,
                        "startTime": start_iso,
                        "endTime": end_iso,
                        "level": level,
                        "statusMessage": status_msg,
                        "metadata": redacted_attrs,
                        "input": redacted_attrs.get("tool_arguments")
                        if self._export_full_payloads
                        else None,
                        "output": redacted_attrs.get("tool_result")
                        if self._export_full_payloads
                        else None,
                    },
                }
                events.append(span_event)

        return {"batch": events}

    def serialize_metrics(self, metrics: tuple[MetricRecord, ...]) -> dict[str, Any]:
        """Map MetricRecord measurements into Langfuse score ingestion events."""
        events: list[dict[str, Any]] = []

        for idx, m in enumerate(metrics):
            metric_iso = _ns_to_iso(m.timestamp_ns)
            score_event: dict[str, Any] = {
                "id": f"evt_score_{idx}_{m.name}",
                "type": "score-create",
                "timestamp": metric_iso,
                "body": {
                    "name": m.name,
                    "value": float(m.value),
                    "comment": f"unit: {m.unit}, kind: {m.kind.value}",
                    "metadata": dict(m.attributes),
                },
            }
            events.append(score_event)

        return {"batch": events}

    async def _send_batch(self, payload: Mapping[str, Any]) -> None:
        """Send a batch payload to the Langfuse ingestion API endpoint."""
        if not self._public_key or not self._secret_key:
            raise TelemetryExportError("Langfuse public_key and secret_key are required for export")

        ingestion_url = f"{self._host}/api/public/ingestion"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        auth = httpx.BasicAuth(self._public_key, self._secret_key)

        try:
            if self._http_client is not None:
                resp = await self._http_client.post(
                    ingestion_url,
                    json=payload,
                    headers=headers,
                    auth=auth,
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        ingestion_url,
                        json=payload,
                        headers=headers,
                        auth=auth,
                        timeout=self._timeout_seconds,
                    )
        except Exception as exc:
            raise TelemetryExportError(
                f"Langfuse ingestion export failed to {ingestion_url}: {exc}"
            ) from exc

        if resp.status_code >= 400:
            raise TelemetryExportError(
                f"Langfuse export rejected by {ingestion_url} with HTTP {resp.status_code}: {resp.text}"
            )

    async def export_spans(self, spans: tuple[SpanRecord, ...]) -> None:
        """Export span records to Langfuse ingestion endpoint."""
        if not spans:
            return
        payload = self.serialize_spans(spans)
        await self._send_batch(payload)

    async def export_metrics(self, metrics: tuple[MetricRecord, ...]) -> None:
        """Export metric records to Langfuse ingestion endpoint."""
        if not metrics:
            return
        payload = self.serialize_metrics(metrics)
        await self._send_batch(payload)


__all__ = [
    "DEFAULT_LANGFUSE_HOST",
    "DEFAULT_TIMEOUT_SECONDS",
    "LangfuseTelemetryExporter",
]
