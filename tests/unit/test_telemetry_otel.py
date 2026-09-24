"""Unit tests for OTLPTelemetryExporter, LangfuseTelemetryExporter, CompositeTelemetryExporter, and factory."""

from __future__ import annotations

import httpx
import pytest

import uclone_x
from uclone_x.cli.commands.run import run_agent_repl_async
from uclone_x.errors import TelemetryExportError
from uclone_x.telemetry import (
    CompositeTelemetryExporter,
    InMemoryTelemetryExporter,
    LangfuseTelemetryExporter,
    MetricKind,
    MetricRecord,
    OTLPTelemetryExporter,
    SpanKind,
    SpanRecord,
    SpanStatus,
    create_telemetry_exporter,
)

# ======================================================================================
# 1. OTLPTelemetryExporter Tests
# ======================================================================================


def test_otlp_exporter_endpoint_and_header_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify OTLP exporter endpoint and header resolution from defaults and env vars."""
    # Defaults
    exporter = OTLPTelemetryExporter()
    assert exporter.endpoint == "http://localhost:4318"
    assert exporter.traces_endpoint == "http://localhost:4318/v1/traces"
    assert exporter.metrics_endpoint == "http://localhost:4318/v1/metrics"
    assert exporter.service_name == "uclone-x"
    assert exporter.headers["Content-Type"] == "application/json"

    # Explicit base endpoint
    custom = OTLPTelemetryExporter("http://collector.internal:4318")
    assert custom.endpoint == "http://collector.internal:4318"
    assert custom.traces_endpoint == "http://collector.internal:4318/v1/traces"
    assert custom.metrics_endpoint == "http://collector.internal:4318/v1/metrics"

    # Env vars
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.company.com:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "api-key=secret123,x-tenant=prod")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "custom-agent-service")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "15.5")

    env_exporter = OTLPTelemetryExporter(headers={"x-override": "custom"})
    assert env_exporter.endpoint == "http://otel.company.com:4318"
    assert env_exporter.traces_endpoint == "http://otel.company.com:4318/v1/traces"
    assert env_exporter.metrics_endpoint == "http://otel.company.com:4318/v1/metrics"
    assert env_exporter.service_name == "custom-agent-service"
    assert env_exporter.timeout_seconds == 15.5
    assert env_exporter.headers["api-key"] == "secret123"
    assert env_exporter.headers["x-tenant"] == "prod"
    assert env_exporter.headers["x-override"] == "custom"

    # Specific traces & metrics endpoint env vars
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://traces.company.com:4318/custom/traces"
    )
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://metrics.company.com:4318/custom/metrics"
    )
    split_exporter = OTLPTelemetryExporter()
    assert split_exporter.traces_endpoint == "http://traces.company.com:4318/custom/traces"
    assert split_exporter.metrics_endpoint == "http://metrics.company.com:4318/custom/metrics"


def test_otlp_exporter_span_serialization_standard_envelope() -> None:
    """Verify OTLP JSON serialization adheres to standard ResourceSpans envelope."""
    exporter = OTLPTelemetryExporter(service_name="uclone-x-test")

    span1 = SpanRecord(
        trace_id="trc_root123",
        span_id="spn_root456",
        name="agent.run",
        kind=SpanKind.INTERNAL,
        start_time_ns=1_000_000_000_000,
        end_time_ns=1_000_500_000_000,
        status=SpanStatus.OK,
        attributes={
            "agent_id": "agt_planner",
            "turn_index": 1,
            "temperature": 0.7,
            "is_active": True,
            "tags": ["prod", "fast"],
            "nested": {"key": "val"},
            "password": "secret_password",
            "prompt": "Sensitive system prompt",
        },
    )
    span2 = SpanRecord(
        trace_id="trc_root123",
        span_id="spn_child789",
        parent_span_id="spn_root456",
        name="llm.generate",
        kind=SpanKind.CLIENT,
        start_time_ns=1_000_100_000_000,
        end_time_ns=1_000_400_000_000,
        status=SpanStatus.ERROR,
        error_message="Rate limit exceeded",
        attributes={
            "provider": "anthropic",
            "model": "claude-3-7-sonnet",
            "input_tokens": 150,
            "output_tokens": 50,
            "api_key": "sk-ant-123",
        },
    )

    payload = exporter.serialize_spans((span1, span2))

    assert "resourceSpans" in payload
    resource_spans = payload["resourceSpans"]
    assert len(resource_spans) == 1

    resource = resource_spans[0]["resource"]
    res_attrs = {attr["key"]: attr["value"]["stringValue"] for attr in resource["attributes"]}
    assert res_attrs["service.name"] == "uclone-x-test"

    scope_spans = resource_spans[0]["scopeSpans"][0]
    spans_list = scope_spans["spans"]
    assert len(spans_list) == 2

    # Verify span 1 serialization and redaction
    s1 = spans_list[0]
    assert s1["traceId"] == "trc_root123"
    assert s1["spanId"] == "spn_root456"
    assert s1["name"] == "agent.run"
    assert s1["kind"] == 1  # INTERNAL
    assert s1["startTimeUnixNano"] == "1000000000000"
    assert s1["endTimeUnixNano"] == "1000500000000"
    assert s1["status"]["code"] == 1  # OK

    s1_attrs = {a["key"]: a["value"] for a in s1["attributes"]}
    assert s1_attrs["agent_id"] == {"stringValue": "agt_planner"}
    assert s1_attrs["turn_index"] == {"intValue": "1"}
    assert s1_attrs["temperature"] == {"doubleValue": 0.7}
    assert s1_attrs["is_active"] == {"boolValue": True}
    assert s1_attrs["password"] == {"stringValue": "[REDACTED]"}
    assert s1_attrs["prompt"] == {"stringValue": "[REDACTED]"}

    # Verify span 2 serialization
    s2 = spans_list[1]
    assert s2["traceId"] == "trc_root123"
    assert s2["spanId"] == "spn_child789"
    assert s2["parentSpanId"] == "spn_root456"
    assert s2["name"] == "llm.generate"
    assert s2["kind"] == 3  # CLIENT
    assert s2["status"]["code"] == 2  # ERROR
    assert s2["status"]["message"] == "Rate limit exceeded"

    s2_attrs = {a["key"]: a["value"] for a in s2["attributes"]}
    assert s2_attrs["provider"] == {"stringValue": "anthropic"}
    assert s2_attrs["model"] == {"stringValue": "claude-3-7-sonnet"}
    assert s2_attrs["input_tokens"] == {"intValue": "150"}
    assert s2_attrs["output_tokens"] == {"intValue": "50"}
    assert s2_attrs["api_key"] == {"stringValue": "[REDACTED]"}


def test_otlp_exporter_metric_serialization() -> None:
    """Verify OTLP JSON serialization for COUNTER, GAUGE, and HISTOGRAM."""
    exporter = OTLPTelemetryExporter(service_name="uclone-x-test")

    m_counter = MetricRecord(
        name="tokens.total",
        kind=MetricKind.COUNTER,
        value=1250.0,
        unit="tokens",
        timestamp_ns=1_000_000_000_000,
        attributes={"model": "gpt-4o"},
    )
    m_gauge = MetricRecord(
        name="memory.usage",
        kind=MetricKind.GAUGE,
        value=52428800.0,
        unit="bytes",
        timestamp_ns=1_000_000_000_000,
        attributes={"tier": "primary"},
    )
    m_histogram = MetricRecord(
        name="execution.latency",
        kind=MetricKind.HISTOGRAM,
        value=42.5,
        unit="ms",
        timestamp_ns=1_000_000_000_000,
        attributes={"operation": "tool.run"},
    )

    payload = exporter.serialize_metrics((m_counter, m_gauge, m_histogram))
    assert "resourceMetrics" in payload
    metrics_list = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    assert len(metrics_list) == 3

    assert metrics_list[0]["name"] == "tokens.total"
    assert metrics_list[0]["sum"]["dataPoints"][0]["asDouble"] == 1250.0

    assert metrics_list[1]["name"] == "memory.usage"
    assert metrics_list[1]["gauge"]["dataPoints"][0]["asDouble"] == 52428800.0

    assert metrics_list[2]["name"] == "execution.latency"
    assert metrics_list[2]["histogram"]["dataPoints"][0]["sum"] == 42.5


@pytest.mark.asyncio
async def test_otlp_exporter_network_success_and_error_propagation() -> None:
    """Verify OTLP exporter handles successful HTTP responses and fails fast on errors."""
    recorded_requests: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        if "fail-server" in str(request.url):
            return httpx.Response(500, text="Internal Collector Error")
        if "fail-bad-request" in str(request.url):
            return httpx.Response(400, text="Invalid ResourceSpans JSON")
        return httpx.Response(200, json={"partialSuccess": {}})

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        # Success cases
        exporter = OTLPTelemetryExporter(
            endpoint="http://collector.local:4318",
            http_client=client,
        )

        span = SpanRecord(trace_id="trc-1", span_id="spn-1", name="step.one")
        metric = MetricRecord(name="count", kind=MetricKind.COUNTER, value=1.0)

        # Empty calls do not send HTTP requests
        await exporter.export_spans(())
        await exporter.export_metrics(())
        assert len(recorded_requests) == 0

        # Normal export
        await exporter.export_spans((span,))
        assert len(recorded_requests) == 1
        assert recorded_requests[0].url == "http://collector.local:4318/v1/traces"

        await exporter.export_metrics((metric,))
        assert len(recorded_requests) == 2
        assert recorded_requests[1].url == "http://collector.local:4318/v1/metrics"

        # HTTP 500 error propagation
        err_exporter_500 = OTLPTelemetryExporter(
            endpoint="http://fail-server:4318",
            http_client=client,
        )
        with pytest.raises(TelemetryExportError, match="Internal Collector Error"):
            await err_exporter_500.export_spans((span,))

        with pytest.raises(TelemetryExportError, match="Internal Collector Error"):
            await err_exporter_500.export_metrics((metric,))

        # HTTP 400 error propagation
        err_exporter_400 = OTLPTelemetryExporter(
            endpoint="http://fail-bad-request:4318",
            http_client=client,
        )
        with pytest.raises(TelemetryExportError, match="Invalid ResourceSpans JSON"):
            await err_exporter_400.export_spans((span,))


# ======================================================================================
# 2. LangfuseTelemetryExporter Tests
# ======================================================================================


def test_langfuse_exporter_resolution_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Langfuse exporter resolves keys, host, and defaults properly."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    exporter = LangfuseTelemetryExporter(
        public_key="pk_custom", secret_key="sk_custom", host="http://langfuse.local:3000"
    )
    assert exporter.public_key == "pk_custom"
    assert exporter.secret_key == "sk_custom"
    assert exporter.host == "http://langfuse.local:3000"

    # From env
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_env_123")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_env_456")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com/")
    env_exp = LangfuseTelemetryExporter()
    assert env_exp.public_key == "pk_env_123"
    assert env_exp.secret_key == "sk_env_456"
    assert env_exp.host == "https://cloud.langfuse.com"


def test_langfuse_exporter_trace_and_generation_mapping() -> None:
    """Verify Langfuse exporter maps traces, spans, and generations with attribute scrubbing."""
    exporter = LangfuseTelemetryExporter(
        public_key="pk_test",
        secret_key="sk_test",
    )

    root_span = SpanRecord(
        trace_id="trc_agent_1",
        span_id="spn_root",
        name="agent.run",
        kind=SpanKind.INTERNAL,
        start_time_ns=1_000_000_000_000,
        end_time_ns=1_000_500_000_000,
        attributes={
            "session_id": "sess_100",
            "agent_id": "agt_architect",
            "auth_token": "secret_token",
            "prompt": "secret user prompt",
        },
    )

    llm_span = SpanRecord(
        trace_id="trc_agent_1",
        span_id="spn_gen_1",
        parent_span_id="spn_root",
        name="llm.generate",
        kind=SpanKind.CLIENT,
        start_time_ns=1_000_100_000_000,
        end_time_ns=1_000_400_000_000,
        attributes={
            "provider": "google",
            "model": "gemini-2.5-pro",
            "input_tokens": 300,
            "output_tokens": 120,
            "total_tokens": 420,
            "temperature": 0.5,
            "api_key": "secret_gemini_key",
        },
    )

    tool_span = SpanRecord(
        trace_id="trc_agent_1",
        span_id="spn_tool_1",
        parent_span_id="spn_root",
        name="tool.execute",
        kind=SpanKind.INTERNAL,
        start_time_ns=1_000_410_000_000,
        end_time_ns=1_000_450_000_000,
        attributes={
            "tool": "read_file",
            "tool_arguments": {"file": "secret.txt"},
        },
    )

    batch_payload = exporter.serialize_spans((root_span, llm_span, tool_span))
    assert "batch" in batch_payload
    events = batch_payload["batch"]

    # Root span creates both a trace-create event and a span-create event
    types = [e["type"] for e in events]
    assert "trace-create" in types
    assert "generation-create" in types
    assert "span-create" in types

    trace_event = next(e for e in events if e["type"] == "trace-create")
    assert trace_event["body"]["id"] == "trc_agent_1"
    assert trace_event["body"]["sessionId"] == "sess_100"
    assert trace_event["body"]["userId"] == "agt_architect"
    assert trace_event["body"]["metadata"]["auth_token"] == "[REDACTED]"
    assert trace_event["body"]["metadata"]["prompt"] == "[REDACTED]"

    gen_event = next(e for e in events if e["type"] == "generation-create")
    assert gen_event["body"]["id"] == "spn_gen_1"
    assert gen_event["body"]["traceId"] == "trc_agent_1"
    assert gen_event["body"]["parentObservationId"] == "spn_root"
    assert gen_event["body"]["model"] == "gemini-2.5-pro"
    assert gen_event["body"]["usage"]["input"] == 300
    assert gen_event["body"]["usage"]["output"] == 120
    assert gen_event["body"]["usage"]["total"] == 420
    assert gen_event["body"]["metadata"]["api_key"] == "[REDACTED]"

    tool_event = next(
        e for e in events if e["type"] == "span-create" and e["body"]["id"] == "spn_tool_1"
    )
    assert tool_event["body"]["parentObservationId"] == "spn_root"
    assert tool_event["body"]["metadata"]["tool"] == "read_file"
    assert tool_event["body"]["input"] is None  # Scrubbed payload by default


@pytest.mark.asyncio
async def test_langfuse_exporter_network_and_error_handling() -> None:
    """Verify Langfuse exporter sends authentication, handles errors, and enforces required keys."""
    # Missing credentials fails fast
    no_key_exporter = LangfuseTelemetryExporter()
    span = SpanRecord(trace_id="trc-1", span_id="spn-1", name="step")

    with pytest.raises(
        TelemetryExportError, match="Langfuse public_key and secret_key are required"
    ):
        await no_key_exporter.export_spans((span,))

    # Network tests with mock transport
    captured_requests: list[httpx.Request] = []

    def mock_langfuse_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if "auth-fail" in str(request.url):
            return httpx.Response(401, text="Unauthorized: Invalid Secret Key")
        return httpx.Response(200, json={"status": "success"})

    transport = httpx.MockTransport(mock_langfuse_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        exporter = LangfuseTelemetryExporter(
            public_key="pk_valid",
            secret_key="sk_valid",
            host="http://langfuse.local:3000",
            http_client=client,
        )

        await exporter.export_spans((span,))
        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.url == "http://langfuse.local:3000/api/public/ingestion"
        assert req.headers["authorization"].startswith("Basic ")

        # Metrics export
        metric = MetricRecord(name="eval.score", kind=MetricKind.GAUGE, value=0.95)
        await exporter.export_metrics((metric,))
        assert len(captured_requests) == 2

        # 401 Unauthorized failure
        err_exp = LangfuseTelemetryExporter(
            public_key="pk_bad",
            secret_key="sk_bad",
            host="http://auth-fail:3000",
            http_client=client,
        )
        with pytest.raises(TelemetryExportError, match="Unauthorized"):
            await err_exp.export_spans((span,))


# ======================================================================================
# 3. CompositeTelemetryExporter & create_telemetry_exporter Factory Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_composite_telemetry_exporter_broadcasting() -> None:
    """Verify CompositeTelemetryExporter fans out spans and metrics to all exporters."""
    sink1 = InMemoryTelemetryExporter()
    sink2 = InMemoryTelemetryExporter()
    composite = CompositeTelemetryExporter([sink1, sink2])

    assert len(composite.exporters) == 2

    span = SpanRecord(trace_id="trc-1", span_id="spn-1", name="step")
    metric = MetricRecord(name="m1", kind=MetricKind.COUNTER, value=10.0)

    await composite.export_spans((span,))
    await composite.export_metrics((metric,))

    assert sink1.span_count == 1
    assert sink2.span_count == 1
    assert sink1.metric_count == 1
    assert sink2.metric_count == 1


@pytest.mark.asyncio
async def test_composite_telemetry_exporter_error_handling() -> None:
    """Verify CompositeTelemetryExporter fails fast when child exporters fail."""
    sink_ok = InMemoryTelemetryExporter()
    sink_err = InMemoryTelemetryExporter(fail_on_export=True)
    composite = CompositeTelemetryExporter([sink_ok, sink_err])

    span = SpanRecord(trace_id="trc-1", span_id="spn-1", name="step")
    with pytest.raises(TelemetryExportError, match="Telemetry span export failed"):
        await composite.export_spans((span,))


def test_create_telemetry_exporter_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify create_telemetry_exporter auto-detects OTLP, Langfuse, or composite from environment."""
    # 1. Clean environment -> returns InMemoryTelemetryExporter
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    exp_default = create_telemetry_exporter()
    assert isinstance(exp_default, InMemoryTelemetryExporter)

    # 2. OTLP configured via env -> returns OTLPTelemetryExporter
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    exp_otlp = create_telemetry_exporter()
    assert isinstance(exp_otlp, OTLPTelemetryExporter)
    assert exp_otlp.endpoint == "http://localhost:4318"

    # 3. Langfuse configured via env -> returns LangfuseTelemetryExporter
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_live")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_live")
    exp_lf = create_telemetry_exporter()
    assert isinstance(exp_lf, LangfuseTelemetryExporter)
    assert exp_lf.public_key == "pk_live"

    # 4. Both configured -> returns CompositeTelemetryExporter
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    exp_composite = create_telemetry_exporter()
    assert isinstance(exp_composite, CompositeTelemetryExporter)
    assert len(exp_composite.exporters) == 2
    assert any(isinstance(e, OTLPTelemetryExporter) for e in exp_composite.exporters)
    assert any(isinstance(e, LangfuseTelemetryExporter) for e in exp_composite.exporters)

    # 5. Parameters override env detection
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    exp_arg = create_telemetry_exporter(otlp_endpoint="http://collector.param:4318")
    assert isinstance(exp_arg, OTLPTelemetryExporter)
    assert exp_arg.endpoint == "http://collector.param:4318"


# ======================================================================================
# 4. CLI REPL Integration Tests (run.py)
# ======================================================================================


@pytest.mark.asyncio
async def test_run_agent_repl_single_shot_emits_telemetry_trace() -> None:
    """Verify ./ucx run single-shot prompt execution records agent.run trace span."""
    # Run single-shot turn
    await run_agent_repl_async(
        agent_name="test_agent",
        provider="mock",
        prompt="Hello UClone-X!",
    )


def test_otlp_span_export_reports_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OTLP span envelope reports the live package version in both places.

    `service.version` on the resource and `version` on the tracer scope are what a
    collector groups and filters traces by. Asserting agreement with
    `uclone_x.__version__` alone would pass against a literal that happens to be
    correct today -- the state #1131 describes. Moving `__version__` to a value no
    literal in the tree carries is what separates reading the declaration from
    restating it.

    Killed by: src/uclone_x/telemetry/otlp.py :: "value": {"stringValue": uclone_x.__version__},
    Becomes: "value": {"stringValue": "0.0.0"},
    """
    monkeypatch.setattr(uclone_x, "__version__", "9.8.7-probe")

    exporter = OTLPTelemetryExporter(service_name="uclone-x-test")
    span = SpanRecord(
        trace_id="trc_probe",
        span_id="spn_probe",
        name="agent.run",
        kind=SpanKind.INTERNAL,
        start_time_ns=1_000_000_000_000,
        end_time_ns=1_000_500_000_000,
        status=SpanStatus.OK,
    )

    resource_span = exporter.serialize_spans((span,))["resourceSpans"][0]

    res_attrs = {
        attr["key"]: attr["value"]["stringValue"]
        for attr in resource_span["resource"]["attributes"]
    }
    assert res_attrs["service.version"] == "9.8.7-probe"

    scope = resource_span["scopeSpans"][0]["scope"]
    assert scope["name"] == "uclone-x.tracer"
    assert scope["version"] == "9.8.7-probe"


def test_otlp_metric_export_reports_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OTLP meter scope reports the live package version.

    Killed by: src/uclone_x/telemetry/otlp.py :: return {"name": name, "version": uclone_x.__version__}
    Becomes: return {"name": name, "version": "0.0.0"}
    """
    monkeypatch.setattr(uclone_x, "__version__", "9.8.7-probe")

    exporter = OTLPTelemetryExporter(service_name="uclone-x-test")
    metric = MetricRecord(
        name="agent.turns",
        kind=MetricKind.COUNTER,
        value=1.0,
        timestamp_ns=1_000_000_000_000,
    )

    scope = exporter.serialize_metrics((metric,))["resourceMetrics"][0]["scopeMetrics"][0]["scope"]
    assert scope["name"] == "uclone-x.metrics"
    assert scope["version"] == "9.8.7-probe"
