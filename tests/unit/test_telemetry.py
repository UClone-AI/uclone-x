"""Unit tests for OpenTelemetry tracer, metrics collector, and in-memory exporter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from uclone_x.sandbox.models import is_secret_env_name
from uclone_x.telemetry import (
    BENIGN_METRIC_EXACT_KEYS,
    DEPTH_LIMITED_PLACEHOLDER,
    FAILOVER_EVENT_SPAN_NAME,
    MAX_REDACTION_DEPTH,
    REDACTED_PLACEHOLDER,
    FailoverSpanAttribution,
    InMemoryExporter,
    InMemoryTelemetryExporter,
    MetricCollector,
    MetricKind,
    MetricRecord,
    MetricRecorder,
    MetricRecorderProtocol,
    MetricsCollector,
    MetricsCollectorProtocol,
    SpanFate,
    SpanKind,
    SpanRecord,
    SpanStatus,
    TelemetryExporter,
    TelemetryTracer,
    Tracer,
    TraceRecorderProtocol,
    TracerProtocol,
    is_benign_metric_attribute_key,
    is_payload_attribute_key,
    is_sensitive_attribute_key,
    redact_span_attributes,
)


def test_telemetry_type_aliases() -> None:
    """Verify all class and protocol aliases are consistent."""
    assert Tracer is TelemetryTracer
    assert TracerProtocol is TraceRecorderProtocol
    assert MetricCollector is MetricsCollector
    assert MetricRecorder is MetricsCollector
    assert MetricsCollectorProtocol is MetricRecorderProtocol
    assert InMemoryExporter is InMemoryTelemetryExporter
    assert TelemetryExporter is InMemoryTelemetryExporter


@pytest.mark.asyncio
async def test_tracer_start_and_end_span() -> None:
    tracer = TelemetryTracer(trace_id="trc_test123", default_attributes={"env": "test"})
    assert tracer.trace_id == "trc_test123"
    assert tracer.active_span_count == 0

    span_id = tracer.start_span("agent.run", kind=SpanKind.INTERNAL, attributes={"user": "alice"})
    assert span_id.startswith("spn_")
    assert tracer.active_span_count == 1

    record = tracer.end_span(span_id, status=SpanStatus.OK)
    assert record is not None
    assert record.span_id == span_id
    assert record.trace_id == "trc_test123"
    assert record.name == "agent.run"
    assert record.kind is SpanKind.INTERNAL
    assert record.status is SpanStatus.OK
    assert record.attributes["env"] == "test"
    assert record.attributes["user"] == "alice"
    assert record.start_time_ns > 0
    assert record.end_time_ns >= record.start_time_ns
    assert tracer.active_span_count == 0

    # Ending a non-existent span returns None
    assert tracer.end_span("spn_unknown") is None

    # Verify history
    completed = tracer.get_completed_spans()
    assert len(completed) == 1
    assert completed[0] == record

    tracer.clear()
    assert len(tracer.get_completed_spans()) == 0


@pytest.mark.asyncio
async def test_tracer_async_context_manager() -> None:
    tracer = TelemetryTracer()

    async with tracer.span("step.one", attributes={"k1": "v1"}) as s1:
        assert s1.startswith("spn_")
        assert tracer.active_span_count == 1

    assert tracer.active_span_count == 0
    spans = tracer.get_completed_spans()
    assert len(spans) == 1
    assert spans[0].name == "step.one"
    assert spans[0].status is SpanStatus.OK
    assert spans[0].attributes["k1"] == "v1"


@pytest.mark.asyncio
async def test_tracer_async_context_manager_exception_handling() -> None:
    tracer = TelemetryTracer()

    with pytest.raises(ValueError, match="Something failed"):
        async with tracer.span("failing.step") as s1:
            assert s1.startswith("spn_")
            raise ValueError("Something failed")

    assert tracer.active_span_count == 0
    spans = tracer.get_completed_spans()
    assert len(spans) == 1
    assert spans[0].name == "failing.step"
    assert spans[0].status is SpanStatus.ERROR
    assert spans[0].error_message == "Something failed"


@pytest.mark.asyncio
async def test_tracer_nested_spans_and_hierarchy() -> None:
    tracer = TelemetryTracer()

    async with tracer.span("parent.turn") as parent_id:
        async with tracer.span("child.tool") as child_id:
            async with tracer.span("grandchild.llm") as grandchild_id:
                pass

    spans = tracer.get_completed_spans()
    assert len(spans) == 3

    # Completed in LIFO order (grandchild, child, parent)
    grandchild_span = spans[0]
    child_span = spans[1]
    parent_span = spans[2]

    assert grandchild_span.name == "grandchild.llm"
    assert grandchild_span.parent_span_id == child_id
    assert grandchild_span.span_id == grandchild_id

    assert child_span.name == "child.tool"
    assert child_span.parent_span_id == parent_id
    assert child_span.span_id == child_id

    assert parent_span.name == "parent.turn"
    assert parent_span.parent_span_id is None
    assert parent_span.span_id == parent_id


@pytest.mark.asyncio
async def test_tracer_semantic_helpers() -> None:
    tracer = TelemetryTracer()

    async with tracer.agent_turn_span(
        agent_id="agt_architect",
        session_id="sess_123",
        turn_index=1,
        extra_attributes={"role": "planner"},
    ) as s_turn:
        async with tracer.tool_call_span(
            tool_name="view_file", tool_path="src/main.py", extra_attributes={"arg": "x"}
        ) as s_tool:
            async with tracer.llm_request_span(
                provider="anthropic", model="claude-3-7-sonnet", tokens=350
            ) as s_llm:
                pass

        async with tracer.a2a_event_span(
            target_agent_id="agt_security", protocol="google-a2a/v1"
        ) as s_a2a:
            pass

    spans = {s.name: s for s in tracer.get_completed_spans()}

    assert "agent.run" in spans
    assert spans["agent.run"].attributes["agent_id"] == "agt_architect"
    assert spans["agent.run"].attributes["session_id"] == "sess_123"
    assert spans["agent.run"].attributes["turn_index"] == 1
    assert spans["agent.run"].attributes["role"] == "planner"
    assert spans["agent.run"].kind is SpanKind.INTERNAL

    assert "tool.execute" in spans
    assert spans["tool.execute"].attributes["tool"] == "view_file"
    assert spans["tool.execute"].attributes["path"] == "src/main.py"
    assert spans["tool.execute"].parent_span_id == s_turn

    assert "llm.generate" in spans
    assert spans["llm.generate"].span_id == s_llm
    assert spans["llm.generate"].attributes["provider"] == "anthropic"
    assert spans["llm.generate"].attributes["model"] == "claude-3-7-sonnet"
    assert spans["llm.generate"].attributes["gen_ai.system"] == "anthropic"
    assert spans["llm.generate"].attributes["tokens"] == 350
    assert spans["llm.generate"].kind is SpanKind.CLIENT
    assert spans["llm.generate"].parent_span_id == s_tool

    assert "a2a.delegate" in spans
    assert spans["a2a.delegate"].span_id == s_a2a
    assert spans["a2a.delegate"].attributes["target"] == "agt_security"
    assert spans["a2a.delegate"].attributes["protocol"] == "google-a2a/v1"
    assert spans["a2a.delegate"].kind is SpanKind.PRODUCER
    assert spans["a2a.delegate"].parent_span_id == s_turn


@pytest.mark.asyncio
async def test_tracer_stream_spans() -> None:
    tracer = TelemetryTracer()
    received_spans: list[SpanRecord] = []

    async def consumer() -> None:
        async for span in tracer.stream_spans():
            received_spans.append(span)
            if len(received_spans) == 2:
                break

    consumer_task = asyncio.create_task(consumer())

    # Give consumer task a moment to initialize the queue
    await asyncio.sleep(0.01)

    async with tracer.span("span_1"):
        pass

    async with tracer.span("span_2"):
        pass

    await asyncio.wait_for(consumer_task, timeout=1.0)
    assert len(received_spans) == 2
    assert received_spans[0].name == "span_1"
    assert received_spans[1].name == "span_2"


def test_metrics_collector_record_and_drain() -> None:
    collector = MetricsCollector(default_attributes={"service": "uclone-x"})
    assert collector.count == 0

    m1 = collector.record(
        "http.requests", 1.0, MetricKind.COUNTER, unit="1", attributes={"method": "GET"}
    )
    assert m1.name == "http.requests"
    assert m1.kind is MetricKind.COUNTER
    assert m1.value == 1.0
    assert m1.unit == "1"
    assert m1.attributes["service"] == "uclone-x"
    assert m1.attributes["method"] == "GET"
    assert m1.timestamp_ns > 0

    assert collector.count == 1
    assert len(collector.get_records()) == 1

    drained = collector.drain()
    assert len(drained) == 1
    assert drained[0] == m1
    assert collector.count == 0
    assert len(collector.get_records()) == 0


def test_metrics_collector_helpers() -> None:
    collector = MetricsCollector()

    # 1. Token count
    rec_in, rec_out = collector.record_token_count(
        input_tokens=1500,
        output_tokens=300,
        provider="openai",
        model="gpt-4o",
        extra_attributes={"tier": "pro"},
    )
    assert rec_in.name == "tokens.input"
    assert rec_in.value == 1500.0
    assert rec_in.attributes["provider"] == "openai"
    assert rec_in.attributes["token_type"] == "input"
    assert rec_in.attributes["tier"] == "pro"

    assert rec_out.name == "tokens.output"
    assert rec_out.value == 300.0
    assert rec_out.attributes["token_type"] == "output"

    # 2. Latency
    rec_lat = collector.record_execution_latency(
        operation_name="sandbox.run",
        duration_ms=45.2,
        attributes={"sandbox_level": "workspace"},
    )
    assert rec_lat.name == "execution.latency"
    assert rec_lat.kind is MetricKind.HISTOGRAM
    assert rec_lat.value == 45.2
    assert rec_lat.unit == "ms"
    assert rec_lat.attributes["operation"] == "sandbox.run"
    assert rec_lat.attributes["sandbox_level"] == "workspace"

    # 3. Error
    rec_err = collector.record_error(
        error_type="TimeoutError",
        operation_name="llm.generate",
        attributes={"provider": "anthropic"},
    )
    assert rec_err.name == "errors.count"
    assert rec_err.kind is MetricKind.COUNTER
    assert rec_err.value == 1.0
    assert rec_err.attributes["error_type"] == "TimeoutError"
    assert rec_err.attributes["operation"] == "llm.generate"

    # 4. Memory usage
    rec_mem = collector.record_memory_usage(bytes_used=104_857_600.0)
    assert rec_mem.name == "memory.usage"
    assert rec_mem.kind is MetricKind.GAUGE
    assert rec_mem.value == 104_857_600.0
    assert rec_mem.unit == "bytes"

    assert collector.count == 5
    collector.clear()
    assert collector.count == 0


def test_redaction_utilities() -> None:
    assert is_sensitive_attribute_key("OPENAI_API_KEY") is True
    assert is_sensitive_attribute_key("auth_token") is True
    assert is_sensitive_attribute_key("user_password") is True
    assert is_sensitive_attribute_key("client_secret") is True
    assert is_sensitive_attribute_key("non_sensitive_field") is False

    assert is_payload_attribute_key("prompt") is True
    assert is_payload_attribute_key("completion") is True
    assert is_payload_attribute_key("tool_arguments") is True
    assert is_payload_attribute_key("gen_ai.prompt") is True
    assert is_payload_attribute_key("model_name") is False

    raw_attrs = {
        "OPENAI_API_KEY": "sk-12345",
        "prompt": "Tell me a secret",
        "tool_result": "Success file contents",
        "model": "gpt-4o",
        "nested": {"token": "xyz", "normal": "abc"},
    }

    # Default: redact secrets and payload bodies
    redacted = redact_span_attributes(raw_attrs, export_full_payloads=False, redact_secrets=True)
    assert redacted["OPENAI_API_KEY"] == "[REDACTED]"
    assert redacted["prompt"] == "[REDACTED]"
    assert redacted["tool_result"] == "[REDACTED]"
    assert redacted["model"] == "gpt-4o"
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert redacted["nested"]["normal"] == "abc"

    # Opt-in full payloads: keep payload bodies, still scrub secrets
    full_payloads = redact_span_attributes(
        raw_attrs, export_full_payloads=True, redact_secrets=True
    )
    assert full_payloads["OPENAI_API_KEY"] == "[REDACTED]"
    assert full_payloads["prompt"] == "Tell me a secret"
    assert full_payloads["tool_result"] == "Success file contents"
    assert full_payloads["model"] == "gpt-4o"
    assert full_payloads["nested"]["token"] == "[REDACTED]"


def test_nested_payload_keys_are_redacted() -> None:
    """`{"llm": {"prompt": ...}}` is the ordinary shape of an `llm.generate` span.

    Nested keys used to be checked against the credential predicate only, so the payload
    predicate never ran below the top level and the prompt was exported verbatim —
    §7 requirements 1 and 2 of `docs/telemetry-opentelemetry.md` were both unmet on any
    span with nested attributes, which is the common case rather than an edge one.
    """
    redacted = redact_span_attributes({"llm": {"prompt": "SECRET"}}, export_full_payloads=False)
    assert redacted["llm"]["prompt"] == REDACTED_PLACEHOLDER

    # The credential predicate still applies at the same depth.
    creds = redact_span_attributes({"llm": {"api_key": "sk-1"}}, export_full_payloads=False)
    assert creds["llm"]["api_key"] == REDACTED_PLACEHOLDER

    # Opt-in full payloads releases the payload body and nothing else.
    opted_in = redact_span_attributes(
        {"llm": {"prompt": "hello", "api_key": "sk-1", "model": "gpt-4o"}},
        export_full_payloads=True,
    )
    assert opted_in["llm"]["prompt"] == "hello"
    assert opted_in["llm"]["api_key"] == REDACTED_PLACEHOLDER
    assert opted_in["llm"]["model"] == "gpt-4o"


def test_redaction_recurses_through_depth_and_sequences() -> None:
    """Both predicates apply at every depth, through mappings, lists and tuples."""
    attrs: dict[str, Any] = {
        "a": {"b": {"c": {"prompt": "SECRET", "keep": "visible"}}},
        "messages_list": [{"content": "SECRET"}, {"role": "user"}],
        "as_tuple": ({"x-api-key": "sk-1"}, "plain"),
        "mixed": [[{"stdout": "SECRET"}]],
    }
    redacted = redact_span_attributes(attrs, export_full_payloads=False)

    assert redacted["a"]["b"]["c"]["prompt"] == REDACTED_PLACEHOLDER
    assert redacted["a"]["b"]["c"]["keep"] == "visible"
    assert redacted["messages_list"][0]["content"] == REDACTED_PLACEHOLDER
    assert redacted["messages_list"][1]["role"] == "user"
    assert redacted["as_tuple"][0]["x-api-key"] == REDACTED_PLACEHOLDER
    assert redacted["as_tuple"][1] == "plain"
    assert redacted["mixed"][0][0]["stdout"] == REDACTED_PLACEHOLDER

    # Plain containers on the way out: a tuple becomes a list, a Mapping becomes a dict.
    assert isinstance(redacted["as_tuple"], list)
    assert isinstance(redacted["a"], dict)


def test_redaction_depth_bound_withholds_and_is_distinguishable() -> None:
    """The depth bound is a policy cap on how much structure is inspected.

    The bound withholds behind its own placeholder rather than the redaction one: "this
    was a credential" and "this was never inspected" are different statements.

    The cycle case below is a property of the *public helper*, not of the export path:
    `SpanRecord.attributes` is `ImmutableJsonMapping`, and Pydantic rejects a cyclic
    mapping with `recursion_loop` before any exporter runs — asserted in
    `test_unbounded_attributes_are_rejected_before_the_exporter`. This test reaches the
    walk by calling `redact_span_attributes` directly, which is exactly the path that
    bypasses that validation.
    """
    assert DEPTH_LIMITED_PLACEHOLDER != REDACTED_PLACEHOLDER

    deep: dict[str, Any] = {"prompt": "SECRET"}
    for level in range(MAX_REDACTION_DEPTH + 4):
        deep = {f"level_{level}": deep}

    redacted = redact_span_attributes(deep, export_full_payloads=False)
    assert DEPTH_LIMITED_PLACEHOLDER in json.dumps(redacted)
    assert "SECRET" not in json.dumps(redacted)

    # A direct call with a cycle terminates rather than recursing without bound.
    cyclic: dict[str, Any] = {"name": "span"}
    cyclic["self"] = cyclic
    assert DEPTH_LIMITED_PLACEHOLDER in json.dumps(redact_span_attributes(cyclic))

    # A scalar sitting at the bound is still emitted; only containers are withheld.
    scalar_at_bound: dict[str, Any] = {"value": "visible"}
    for level in range(MAX_REDACTION_DEPTH - 2):
        scalar_at_bound = {f"level_{level}": scalar_at_bound}
    assert "visible" in json.dumps(redact_span_attributes(scalar_at_bound))


def test_unbounded_attributes_are_rejected_before_the_exporter() -> None:
    """Nothing unbounded — cyclic or over-deep — can reach an exporter.

    Pins the boundary that two successive rationales for `MAX_REDACTION_DEPTH` got
    wrong, first claiming attributes could be cyclic and then that they could nest past
    Python's recursion limit. `attributes` is `ImmutableJsonMapping`, so pydantic-core's
    recursion guard rejects *both* shapes at `SpanRecord` construction, before
    `freeze_mapping` and long before `redact_span_attributes`. The redactor's tolerance
    of them is a property of the public helper's contract, not a control on the export
    path — which is why the bound is documented as a policy cap instead.

    Asserts the *direction* only. The validator's exact ceiling is a pydantic-core
    internal (254 levels today, fixed regardless of `sys.setrecursionlimit`); pinning
    the number here would couple the suite to that internal for no benefit.
    """
    cyclic: dict[str, Any] = {"name": "span"}
    cyclic["self"] = cyclic

    with pytest.raises(ValidationError, match="cyclic reference detected"):
        SpanRecord(trace_id="trc-1", span_id="spn-1", name="llm.generate", attributes=cyclic)

    over_deep: dict[str, Any] = {"prompt": "SECRET"}
    for level in range(1000):
        over_deep = {f"level_{level}": over_deep}
    with pytest.raises(ValidationError, match="recursion"):
        SpanRecord(trace_id="trc-1", span_id="spn-1", name="llm.generate", attributes=over_deep)

    # Modestly deep and acyclic *is* accepted, and the policy cap still truncates it.
    deep: dict[str, Any] = {"prompt": "SECRET"}
    for level in range(MAX_REDACTION_DEPTH + 4):
        deep = {f"level_{level}": deep}
    span = SpanRecord(trace_id="trc-1", span_id="spn-1", name="llm.generate", attributes=deep)
    redacted = json.dumps(redact_span_attributes(span.attributes))
    assert "SECRET" not in redacted
    assert DEPTH_LIMITED_PLACEHOLDER in redacted


def test_dotted_vendor_attributes_survive_export() -> None:
    """#95: dotted vendor resource attributes are not credentials and must survive.

    PR #87 normalised `-`/`.` to `_` so `x-api-key` and `llm.api_key` would be caught.
    That normalisation also fed dotted keys through the `AWS_*`/`GH_*`/`GITHUB_*` prefix
    globs, redacting an entire namespace of legitimate OTel resource attributes — the
    ones that say which repository, which commit and which region a span came from.
    Replacing those globs with exact names (#95) fixes it at the root: there is no
    prefix rule left for a dotted key to match.
    """
    attrs: dict[str, Any] = {
        "aws.region": "us-east-1",
        "aws.profile": "prod",
        "aws.s3.bucket": "my-bucket",
        "gh.repo": "UClone-AI/uclone-x",
        "gh.host": "github.com",
        "github.repository": "UClone-AI/uclone-x",
        "github.sha": "1e29a5c",
        "github.workflow": "ci",
    }
    redacted = redact_span_attributes(attrs, export_full_payloads=False)
    assert redacted == attrs

    # The normalisation still does its job on genuinely credential-shaped dotted keys.
    creds: dict[str, Any] = {
        "github.token": "ghp_x",
        "llm.api_key": "sk-1",
        "x-api-key": "sk-2",
        "aws.secret_access_key": "abc",
    }
    scrubbed = redact_span_attributes(creds, export_full_payloads=False)
    assert all(v == REDACTED_PLACEHOLDER for v in scrubbed.values()), scrubbed

    # Nested and in-list, since recursion is what caused the original leak (#87) and a
    # top-level-only assertion would not have caught it.
    nested: dict[str, Any] = {
        "resource": {"aws.region": "us-east-1", "github.repository": "UClone-AI/uclone-x"},
        "spans": [{"gh.repo": "UClone-AI/uclone-x", "github.token": "ghp_x"}],
    }
    out = redact_span_attributes(nested, export_full_payloads=False)
    assert out["resource"]["aws.region"] == "us-east-1"
    assert out["resource"]["github.repository"] == "UClone-AI/uclone-x"
    assert out["spans"][0]["gh.repo"] == "UClone-AI/uclone-x"
    assert out["spans"][0]["github.token"] == REDACTED_PLACEHOLDER


def test_exporter_key_list_is_broader_than_the_env_predicate() -> None:
    """The two lists differ in breadth on purpose, and only in one direction.

    Over-matching a trace attribute key costs a `[REDACTED]`; over-matching an
    environment name is a construction refusal that breaks a tool. So the exporter may
    be a superset of `is_secret_env_name` and never a subset — a name too sensitive for
    a trace is therefore never inheritable into an untrusted child process (FR-12.3).
    """
    # Superset, never subset: everything the env predicate refuses, the exporter redacts.
    for name in ("API_KEY", "TOKEN", "SSH_AUTH_SOCK", "DATABASE_URL", "GITHUB_TOKEN"):
        assert is_secret_env_name(name) is True
        assert is_sensitive_attribute_key(name) is True

    # Broader on the exporter side only, where the cost is one redacted trace value.
    for name in ("GIT_AUTHOR_NAME", "authorization"):
        assert is_sensitive_attribute_key(name) is True
        assert is_secret_env_name(name) is False


def test_is_benign_metric_attribute_key() -> None:
    """Verify is_benign_metric_attribute_key identifies metric keys and rejects credential keys."""
    # Keys ending with _tokens
    ending_with_tokens = (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "max_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "INPUT_TOKENS",
        "MAX_TOKENS",
    )
    for key in ending_with_tokens:
        assert is_benign_metric_attribute_key(key) is True

    # Namespaced metric keys
    namespaced = (
        "llm.input_tokens",
        "llm.output_tokens",
        "llm.max_tokens",
        "llm.total_tokens",
        "llm.prompt_tokens",
        "llm.completion_tokens",
        "llm.tokens",
        "llm.token_count",
        "gen_ai.usage.input_tokens",
        "gen_ai.client.token_count",
        "LLM.INPUT_TOKENS",
        "LLM.OUTPUT_TOKENS",
    )
    for key in namespaced:
        assert is_benign_metric_attribute_key(key) is True

    # Exact metric terms
    exact_terms = (
        "token_count",
        "tokens_per_second",
        "num_tokens",
        "tokens",
        "TOKEN_COUNT",
        "TOKENS_PER_SECOND",
        "NUM_TOKENS",
        "TOKENS",
    )
    for key in exact_terms:
        assert is_benign_metric_attribute_key(key) is True

    # Verify exact keys tuple contents
    assert set(BENIGN_METRIC_EXACT_KEYS) == {
        "token_count",
        "tokens_per_second",
        "num_tokens",
        "tokens",
    }

    # Credential token keys must NOT be recognized as benign metric keys
    credentials = (
        "token",
        "api_token",
        "auth_token",
        "access_token",
        "github.token",
        "session_token",
        "GITHUB_TOKEN",
        "API_TOKEN",
        "AUTH_TOKEN",
        "user_token",
        "TOKEN",
    )
    for key in credentials:
        assert is_benign_metric_attribute_key(key) is False

    # Other non-metric keys
    non_metrics = (
        "model",
        "user_id",
        "prompt",
        "completion",
        "password",
        "secret",
        "api_key",
    )
    for key in non_metrics:
        assert is_benign_metric_attribute_key(key) is False


def test_token_count_metrics_survive_redaction_top_level_and_nested() -> None:
    """Verify that token count telemetry metrics survive span redaction at top level and nested."""
    attrs: dict[str, Any] = {
        # Top-level benign metrics
        "llm.input_tokens": 120,
        "llm.output_tokens": 45,
        "input_tokens": 120,
        "output_tokens": 45,
        "max_tokens": 2048,
        "token_count": 165,
        "tokens": 165,
        "total_tokens": 165,
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "tokens_per_second": 32.5,
        "num_tokens": 165,
        # Nested benign metrics
        "usage": {
            "input_tokens": 120,
            "output_tokens": 45,
            "total_tokens": 165,
        },
        "llm": {
            "max_tokens": 2048,
            "token_count": 165,
            "prompt_tokens": 120,
            "completion_tokens": 45,
        },
        "metrics_list": [
            {"input_tokens": 100, "output_tokens": 50},
            {"total_tokens": 150, "token_count": 150},
        ],
    }

    redacted = redact_span_attributes(attrs, export_full_payloads=False, redact_secrets=True)

    # Top-level checks
    assert redacted["llm.input_tokens"] == 120
    assert redacted["llm.output_tokens"] == 45
    assert redacted["input_tokens"] == 120
    assert redacted["output_tokens"] == 45
    assert redacted["max_tokens"] == 2048
    assert redacted["token_count"] == 165
    assert redacted["tokens"] == 165
    assert redacted["total_tokens"] == 165
    assert redacted["prompt_tokens"] == 120
    assert redacted["completion_tokens"] == 45
    assert redacted["tokens_per_second"] == 32.5
    assert redacted["num_tokens"] == 165

    # Nested checks
    assert redacted["usage"]["input_tokens"] == 120
    assert redacted["usage"]["output_tokens"] == 45
    assert redacted["usage"]["total_tokens"] == 165

    assert redacted["llm"]["max_tokens"] == 2048
    assert redacted["llm"]["token_count"] == 165
    assert redacted["llm"]["prompt_tokens"] == 120
    assert redacted["llm"]["completion_tokens"] == 45

    assert redacted["metrics_list"][0]["input_tokens"] == 100
    assert redacted["metrics_list"][0]["output_tokens"] == 50
    assert redacted["metrics_list"][1]["total_tokens"] == 150
    assert redacted["metrics_list"][1]["token_count"] == 150


def test_credential_tokens_are_strictly_redacted_top_level_and_nested() -> None:
    """Verify that actual credential tokens are strictly redacted to [REDACTED]."""
    attrs: dict[str, Any] = {
        # Top-level credentials
        "token": "secret-token-value",
        "api_token": "secret-api-token",
        "auth_token": "secret-auth-token",
        "access_token": "secret-access-token",
        "github.token": "ghp_secret12345",
        "session_token": "sess_secret_abc",
        "GITHUB_TOKEN": "ghp_secret67890",
        "API_TOKEN": "api_secret_key",
        "AUTH_TOKEN": "bearer_secret",
        # Nested credentials inside a non-sensitive container mapping
        "configs": {
            "token": "nested-token",
            "api_token": "nested-api-token",
            "auth_token": "nested-auth-token",
            "access_token": "nested-access-token",
            "github.token": "nested-gh-token",
            "session_token": "nested-session-token",
        },
        "items": [
            {"token": "list-token-1", "user": "alice"},
            {"api_token": "list-token-2", "user": "bob"},
        ],
    }

    redacted = redact_span_attributes(attrs, export_full_payloads=False, redact_secrets=True)

    # Top-level assertions
    assert redacted["token"] == REDACTED_PLACEHOLDER
    assert redacted["api_token"] == REDACTED_PLACEHOLDER
    assert redacted["auth_token"] == REDACTED_PLACEHOLDER
    assert redacted["access_token"] == REDACTED_PLACEHOLDER
    assert redacted["github.token"] == REDACTED_PLACEHOLDER
    assert redacted["session_token"] == REDACTED_PLACEHOLDER
    assert redacted["GITHUB_TOKEN"] == REDACTED_PLACEHOLDER
    assert redacted["API_TOKEN"] == REDACTED_PLACEHOLDER
    assert redacted["AUTH_TOKEN"] == REDACTED_PLACEHOLDER

    # Nested assertions
    assert redacted["configs"]["token"] == REDACTED_PLACEHOLDER
    assert redacted["configs"]["api_token"] == REDACTED_PLACEHOLDER
    assert redacted["configs"]["auth_token"] == REDACTED_PLACEHOLDER
    assert redacted["configs"]["access_token"] == REDACTED_PLACEHOLDER
    assert redacted["configs"]["github.token"] == REDACTED_PLACEHOLDER
    assert redacted["configs"]["session_token"] == REDACTED_PLACEHOLDER

    assert redacted["items"][0]["token"] == REDACTED_PLACEHOLDER
    assert redacted["items"][0]["user"] == "alice"
    assert redacted["items"][1]["api_token"] == REDACTED_PLACEHOLDER
    assert redacted["items"][1]["user"] == "bob"


@pytest.mark.asyncio
async def test_in_memory_exporter_preserves_token_metrics_while_redacting_credentials() -> None:
    """Verify that InMemoryTelemetryExporter exports spans preserving token counts while redacting credentials."""
    exporter = InMemoryTelemetryExporter(export_full_payloads=False, redact_secrets=True)

    span = SpanRecord(
        trace_id="trc-metrics-1",
        span_id="spn-metrics-1",
        name="llm.generate",
        attributes={
            "provider": "anthropic",
            "model": "claude-3-7-sonnet",
            "llm.input_tokens": 512,
            "llm.output_tokens": 128,
            "total_tokens": 640,
            "token_count": 640,
            "api_token": "sk-ant-secret123",
            "github.token": "ghp_topsecret",
            "usage": {
                "input_tokens": 512,
                "output_tokens": 128,
                "session_token": "sess-hidden",
            },
        },
    )

    await exporter.export_spans((span,))

    exported = exporter.get_exported_spans()
    assert len(exported) == 1
    exported_attrs = exported[0].attributes

    # Metrics preserved
    assert exported_attrs["provider"] == "anthropic"
    assert exported_attrs["model"] == "claude-3-7-sonnet"
    assert exported_attrs["llm.input_tokens"] == 512
    assert exported_attrs["llm.output_tokens"] == 128
    assert exported_attrs["total_tokens"] == 640
    assert exported_attrs["token_count"] == 640

    usage = exported_attrs["usage"]
    assert isinstance(usage, Mapping)
    assert usage["input_tokens"] == 512
    assert usage["output_tokens"] == 128

    # Credentials redacted
    assert exported_attrs["api_token"] == REDACTED_PLACEHOLDER
    assert exported_attrs["github.token"] == REDACTED_PLACEHOLDER
    assert usage["session_token"] == REDACTED_PLACEHOLDER


@pytest.mark.asyncio
async def test_in_memory_telemetry_exporter_spans_and_metrics() -> None:
    exporter = InMemoryTelemetryExporter(export_full_payloads=False, redact_secrets=True)
    assert exporter.export_full_payloads is False
    assert exporter.redact_secrets is True
    assert exporter.span_count == 0
    assert exporter.metric_count == 0

    span = SpanRecord(
        trace_id="trc-1",
        span_id="spn-1",
        name="tool.execute",
        attributes={
            "ANTHROPIC_API_KEY": "sk-secret",
            "prompt": "sensitive prompt",
            "tool": "view_file",
        },
    )
    metric = MetricRecord(
        name="tokens.total",
        kind=MetricKind.COUNTER,
        value=500.0,
        unit="tokens",
    )

    await exporter.export_spans((span,))
    await exporter.export_metrics((metric,))

    assert exporter.span_count == 1
    assert exporter.metric_count == 1

    exported_spans = exporter.get_exported_spans()
    assert exported_spans[0].span_id == "spn-1"
    assert exported_spans[0].attributes["ANTHROPIC_API_KEY"] == "[REDACTED]"
    assert exported_spans[0].attributes["prompt"] == "[REDACTED]"
    assert exported_spans[0].attributes["tool"] == "view_file"

    exported_metrics = exporter.get_exported_metrics()
    assert exported_metrics[0].name == "tokens.total"
    assert exported_metrics[0].value == 500.0

    exporter.clear()
    assert exporter.span_count == 0
    assert exporter.metric_count == 0


@pytest.mark.asyncio
async def test_in_memory_telemetry_exporter_fail_on_export() -> None:
    exporter = InMemoryTelemetryExporter(fail_on_export=True)

    span = SpanRecord(trace_id="trc-1", span_id="spn-1", name="test")
    metric = MetricRecord(name="test", kind=MetricKind.COUNTER, value=1.0)

    with pytest.raises(RuntimeError, match="Telemetry span export failed"):
        await exporter.export_spans((span,))

    with pytest.raises(RuntimeError, match="Telemetry metrics export failed"):
        await exporter.export_metrics((metric,))


@pytest.mark.asyncio
async def test_end_to_end_telemetry_pipeline() -> None:
    tracer = TelemetryTracer(default_attributes={"app": "uclone-x"})
    collector = MetricsCollector()
    exporter = InMemoryTelemetryExporter(export_full_payloads=False)

    # Simulate agent turn
    async with tracer.agent_turn_span(agent_id="agt_1", session_id="sess_1", turn_index=0):
        # Latency metric
        collector.record_execution_latency("agent.turn", 120.5)

        # Tool execution
        async with tracer.tool_call_span(
            "grep_search", extra_attributes={"query": "TODO", "password": "pass"}
        ):
            pass

        # LLM call
        async with tracer.llm_request_span("google", "gemini-2.5-pro", tokens=250):
            collector.record_token_count(200, 50, "google", "gemini-2.5-pro")

    # Export
    await exporter.export_spans(tracer.get_completed_spans())
    await exporter.export_metrics(collector.drain())

    assert exporter.span_count == 3
    assert exporter.metric_count == 3

    # Check that grep_search span had password redacted
    spans_by_name = {s.name: s for s in exporter.get_exported_spans()}
    assert spans_by_name["tool.execute"].attributes["password"] == "[REDACTED]"
    assert spans_by_name["tool.execute"].attributes["query"] == "TODO"


@pytest.mark.asyncio
async def test_semantic_helpers_defaults_and_options() -> None:
    tracer = TelemetryTracer()

    # Agent turn without turn_index or extra_attributes
    async with tracer.agent_turn_span(agent_id="agt_simple", session_id="sess_simple") as s_turn:
        # Tool call without path or extra_attributes
        async with tracer.tool_call_span(tool_name="list_dir") as s_tool:
            # LLM request without tokens or extra_attributes
            async with tracer.llm_request_span(provider="ollama", model="llama3") as s_llm:
                # A2A without extra_attributes
                async with tracer.a2a_event_span(target_agent_id="agt_peer") as s_a2a:
                    assert s_turn and s_tool and s_llm and s_a2a

    spans = {s.name: s for s in tracer.get_completed_spans()}
    assert "agent.run" in spans
    assert "turn_index" not in spans["agent.run"].attributes

    assert "tool.execute" in spans
    assert "path" not in spans["tool.execute"].attributes

    assert "llm.generate" in spans
    assert "tokens" not in spans["llm.generate"].attributes

    assert "a2a.delegate" in spans
    assert spans["a2a.delegate"].attributes["protocol"] == "google-a2a/v1"


def test_metrics_collector_record_error_with_custom_attributes() -> None:
    collector = MetricsCollector()
    rec = collector.record_error(
        error_type="CustomError",
        operation_name="op",
        attributes={"severity": "high"},
    )
    assert rec.attributes["error_type"] == "CustomError"
    assert rec.attributes["operation"] == "op"
    assert rec.attributes["severity"] == "high"


# ======================================================================================
# Span buffer lifecycle: bounded, drained, and accounted (issue #187)
# ======================================================================================


def test_the_completed_span_buffer_is_bounded_and_evictions_are_counted() -> None:
    """An undrained tracer stops growing, and says how much it discarded (#187).

    The `ui/app.py` tracer is a module-global with no exporter and no reader, so an
    unbounded list there grew for the life of the server process. Bounding it silently
    would be the P6 shape — a control that loses data without saying so — so the
    eviction count is readable rather than log-only, following
    `BaseAgent.processing_errors` ("P6 forbids a failure that is visible only in
    telemetry") rather than the event bus's log-only drop accounting.
    """
    tracer = TelemetryTracer(max_completed_spans=64)

    for _ in range(500):
        span_id = tracer.start_span("failover.event")
        tracer.end_span(span_id=span_id)

    assert len(tracer.get_completed_spans()) == 64
    assert tracer.dropped_span_count == 500 - 64
    assert tracer.drop_reasons == {"buffer_overflow": 436}


def test_discarding_exported_spans_keeps_ones_completed_since_the_read() -> None:
    """`discard_exported` removes exactly what was taken, which `clear()` cannot (#187).

    A span completed between a consumer's read and its acknowledgement is not the
    consumer's to throw away. `clear()` wiped the whole buffer, so that span was lost
    with no record; this is the reason the drain contract names the batch.
    """
    tracer = TelemetryTracer()
    first = tracer.end_span(span_id=tracer.start_span("early"))
    taken = tracer.get_completed_spans()
    late = tracer.end_span(span_id=tracer.start_span("late"))
    assert first is not None
    assert late is not None

    tracer.discard_exported(taken)

    remaining = tracer.get_completed_spans()
    assert [s.span_id for s in remaining] == [late.span_id]
    assert first.span_id not in {s.span_id for s in remaining}
    assert tracer.dropped_span_count == 0, "consumed spans are not losses"


def test_clear_counts_what_it_throws_away() -> None:
    """`clear()` is a reset, so anything still buffered is accounted as dropped (#187).

    It stays available — tests use it — but it no longer discards silently, which is what
    let the CLI lose a turn's spans behind an export warning.
    """
    tracer = TelemetryTracer()
    for _ in range(3):
        tracer.end_span(span_id=tracer.start_span("agent.run"))

    tracer.clear()

    assert tracer.get_completed_spans() == ()
    assert tracer.dropped_span_count == 3
    assert tracer.drop_reasons == {"cleared": 3}


# ======================================================================================
# The subscriber queue is the tracer's *second* buffer, and it is bounded too (issue #197)
# ======================================================================================


@pytest.mark.asyncio
async def test_a_subscriber_that_never_drains_cannot_grow_tracer_memory() -> None:
    """A stalled `stream_spans` subscriber is bounded and its losses are counted (#197).

    The case the suite did not have. #192 bounded `_completed_spans` and counted its
    evictions, but `stream_spans` handed each subscriber a plain `asyncio.Queue()` --
    `maxsize=0`, unbounded. Measured before the fix, with the completed-span buffer
    capped at 4 and one subscriber attached and never draining: the buffer held at 4
    while the subscriber's queue reached 20,000 and `dropped_span_count` read 19,996.
    So the counter that looks like a memory bound was reporting on one of two paths.
    """
    tracer = TelemetryTracer(max_completed_spans=4, max_subscriber_queue_spans=8)
    stream = tracer.stream_spans()

    async def stalled_subscriber() -> None:
        # Subscribe and then never consume: `__anext__` is awaited once so the queue is
        # registered, and the task is parked before the first record is delivered.
        async for _ in stream:  # pragma: no cover - the body never runs
            await asyncio.sleep(3600)

    subscriber = asyncio.create_task(stalled_subscriber())
    await asyncio.sleep(0.01)
    assert len(tracer._stream_subscribers) == 1  # pyright: ignore[reportPrivateUsage]

    for index in range(2000):
        tracer.end_span(span_id=tracer.start_span(f"agent.run.{index}"))

    pending = [
        queue.qsize()
        for queue in tracer._stream_subscribers  # pyright: ignore[reportPrivateUsage]
    ]
    assert pending == [8], "a stalled subscriber's queue grew past its bound"
    assert len(tracer.get_completed_spans()) == 4

    # The non-delivery is observable, in the reason-keyed form #192 established, and on
    # its own counter rather than folded into the buffer's.
    assert tracer.undelivered_span_count == 2000 - 8
    assert tracer.undelivered_reasons == {"subscriber_queue_overflow": 1992}
    assert tracer.drop_reasons == {"buffer_overflow": 1996}
    assert "subscriber_queue_overflow" not in tracer.drop_reasons

    subscriber.cancel()
    with pytest.raises(asyncio.CancelledError):
        await subscriber


@pytest.mark.asyncio
async def test_a_stalled_subscriber_keeps_the_newest_spans_not_the_oldest() -> None:
    """Subscriber overflow is DROP_OLDEST, matching the `deque` and `EventBus` (#197).

    A live span stream is a tail: a reader that comes back wants what just happened, not
    the first few spans of a stall that has been running for an hour.
    """
    tracer = TelemetryTracer(max_subscriber_queue_spans=3)
    stream = tracer.stream_spans()

    async def stalled_subscriber() -> None:
        async for _ in stream:  # pragma: no cover - the body never runs
            await asyncio.sleep(3600)

    subscriber = asyncio.create_task(stalled_subscriber())
    await asyncio.sleep(0.01)

    for index in range(6):
        tracer.end_span(span_id=tracer.start_span(f"span_{index}"))

    (queue,) = tracer._stream_subscribers  # pyright: ignore[reportPrivateUsage]
    retained = [queue.get_nowait().name for _ in range(queue.qsize())]
    assert retained == ["span_3", "span_4", "span_5"]

    subscriber.cancel()
    with pytest.raises(asyncio.CancelledError):
        await subscriber


@pytest.mark.asyncio
async def test_a_subscriber_drop_is_not_counted_as_a_buffer_drop() -> None:
    """`undelivered_span_count` and `dropped_span_count` measure different losses (#197).

    The distinction is the whole reason there are two counters: a span a subscriber never
    received is still in the completed-span buffer, still returned by
    `get_completed_spans()`, and still resolvable by P6 check 5. Only that subscriber's
    view of it was lost. A shared bucket would double-count the spans that suffer both.
    """
    tracer = TelemetryTracer(max_completed_spans=1024, max_subscriber_queue_spans=2)
    stream = tracer.stream_spans()

    async def stalled_subscriber() -> None:
        async for _ in stream:  # pragma: no cover - the body never runs
            await asyncio.sleep(3600)

    subscriber = asyncio.create_task(stalled_subscriber())
    await asyncio.sleep(0.01)

    for index in range(10):
        tracer.end_span(span_id=tracer.start_span(f"span_{index}"))

    assert tracer.undelivered_span_count == 8
    assert tracer.dropped_span_count == 0, "nothing left the buffer; it holds 1024"
    assert len(tracer.get_completed_spans()) == 10

    subscriber.cancel()
    with pytest.raises(asyncio.CancelledError):
        await subscriber


@pytest.mark.asyncio
async def test_a_draining_subscriber_still_receives_every_span() -> None:
    """The bound costs a subscriber that keeps up nothing (#197).

    Bounding the queue rather than disconnecting the subscriber was chosen so that a
    momentarily slow SSE client degrades instead of dying; this pins the other half of
    that -- a client that drains loses nothing, and the generator does not end early.
    """
    tracer = TelemetryTracer(max_subscriber_queue_spans=2)
    received: list[str] = []

    async def consumer() -> None:
        async for span in tracer.stream_spans():
            received.append(span.name)
            if len(received) == 5:
                break

    consumer_task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)

    for index in range(5):
        tracer.end_span(span_id=tracer.start_span(f"span_{index}"))
        await asyncio.sleep(0)

    await asyncio.wait_for(consumer_task, timeout=1.0)
    assert received == [f"span_{index}" for index in range(5)]
    assert tracer.undelivered_span_count == 0
    assert tracer.undelivered_reasons == {}


# ======================================================================================
# A dropped failover span stays attributable by id (issue #199)
# ======================================================================================


def test_a_failover_span_evicted_by_the_bound_is_still_attributable_by_id() -> None:
    """The `buffer_overflow` path keeps the dropped `failover.event` id (#199).

    P6 check 5 requires every `attempts[*].span_id` on a `failover` result to resolve to
    an emitted `failover.event` span. Before this, an unresolved id had four possible
    causes and only an aggregate count keyed by reason to separate them, so the reader
    could not ask about *this* span. The genuinely new loss cause #192 introduced is
    `buffer_overflow`, and it is the one driven here.
    """
    tracer = TelemetryTracer(max_completed_spans=2)
    failover_span_id = tracer.start_span(FAILOVER_EVENT_SPAN_NAME)
    tracer.end_span(span_id=failover_span_id)

    # Still buffered: check 5 resolves it the ordinary way.
    assert tracer.attribute_failover_span(failover_span_id) == FailoverSpanAttribution(
        span_id=failover_span_id,
        fate=SpanFate.RETAINED,
    )

    for _ in range(10):
        tracer.end_span(span_id=tracer.start_span("agent.run"))

    assert failover_span_id not in {s.span_id for s in tracer.get_completed_spans()}
    assert tracer.attribute_failover_span(failover_span_id) == FailoverSpanAttribution(
        span_id=failover_span_id,
        fate=SpanFate.DROPPED,
        drop_reason="buffer_overflow",
    )


def test_a_span_id_this_tracer_never_held_is_unrecorded_not_dropped() -> None:
    """Fabricated provenance is separable from eviction, for that specific id (#199)."""
    tracer = TelemetryTracer(max_completed_spans=2)
    for _ in range(10):
        tracer.end_span(span_id=tracer.start_span(FAILOVER_EVENT_SPAN_NAME))

    assert tracer.dropped_span_count == 8
    assert tracer.attribute_failover_span("spn_fabricated") == FailoverSpanAttribution(
        span_id="spn_fabricated",
        fate=SpanFate.UNRECORDED,
    )


def test_the_other_two_drop_paths_also_keep_the_failover_id() -> None:
    """`drop_unexported` and `clear()` are attributable per span as well (#199)."""
    export_failed = TelemetryTracer()
    lost = export_failed.end_span(span_id=export_failed.start_span(FAILOVER_EVENT_SPAN_NAME))
    assert lost is not None
    export_failed.drop_unexported((lost,), "cli_export_failed")
    assert export_failed.attribute_failover_span(lost.span_id).fate is SpanFate.DROPPED
    assert export_failed.attribute_failover_span(lost.span_id).drop_reason == ("cli_export_failed")

    cleared = TelemetryTracer()
    wiped = cleared.end_span(span_id=cleared.start_span(FAILOVER_EVENT_SPAN_NAME))
    assert wiped is not None
    cleared.clear()
    assert cleared.attribute_failover_span(wiped.span_id).drop_reason == "cleared"


def test_an_exported_and_acknowledged_failover_span_stays_indistinguishable() -> None:
    """The residual #199 records rather than closes, pinned so it is not overread.

    A span taken through `discard_exported` was *consumed*, not lost, so it is not
    counted as a drop and its identity is not retained. It therefore reads exactly like a
    fabricated id. Closing that would mean retaining every successfully exported failover
    id for the tracer's life, which is the "a tracer is a buffer, not a store" line #187
    drew; when an id was exported, check 5 is answerable at the collector that took it.
    """
    tracer = TelemetryTracer()
    exported = tracer.end_span(span_id=tracer.start_span(FAILOVER_EVENT_SPAN_NAME))
    assert exported is not None
    tracer.discard_exported((exported,))

    assert tracer.dropped_span_count == 0, "consumed spans are not losses"
    assert tracer.attribute_failover_span(exported.span_id).fate is SpanFate.UNRECORDED
    assert tracer.attribute_failover_span("spn_fabricated").fate is SpanFate.UNRECORDED


def test_the_retained_identity_bound_accounts_for_its_own_eviction() -> None:
    """The fix's own bound must not rebuild #187 one level up (#199).

    A capped set of retained ids evicts too. If that eviction were unaccounted, an
    `UNRECORDED` verdict would silently start meaning "possibly dropped, forgotten" --
    the same control-that-loses-data-without-saying-so shape #187 was filed for, inside
    the fix for its follow-up. So the set counts what it forgets, and every verdict
    carries that count: `UNRECORDED` alongside a non-zero `identities_forgotten` is a
    maybe, not a no.
    """
    tracer = TelemetryTracer(max_completed_spans=1, max_retained_dropped_failover_ids=2)
    failover_ids: list[str] = []
    for _ in range(5):
        span_id = tracer.start_span(FAILOVER_EVENT_SPAN_NAME)
        tracer.end_span(span_id=span_id)
        failover_ids.append(span_id)

    assert tracer.forgotten_drop_identity_count == 2
    assert tracer.attribute_failover_span(failover_ids[0]) == FailoverSpanAttribution(
        span_id=failover_ids[0],
        fate=SpanFate.UNRECORDED,
        identities_forgotten=2,
    )
    newest_dropped = tracer.attribute_failover_span(failover_ids[3])
    assert newest_dropped.fate is SpanFate.DROPPED
    assert newest_dropped.drop_reason == "buffer_overflow"
    assert newest_dropped.identities_forgotten == 2


def test_only_failover_identities_are_retained_and_that_is_the_stated_scope() -> None:
    """Retention is bounded by failovers occurred, not by span volume (#199).

    Chosen over retaining every dropped id: `failover.event` is the only span name any
    P6 check resolves an id against, so the wide set would multiply retention by total
    span throughput for no additional answer. The cost is stated here rather than left
    to be discovered -- a dropped non-failover span reports `UNRECORDED`.
    """
    tracer = TelemetryTracer(max_completed_spans=1)
    ordinary = tracer.end_span(span_id=tracer.start_span("agent.run"))
    assert ordinary is not None
    for _ in range(5):
        tracer.end_span(span_id=tracer.start_span("agent.run"))

    assert tracer.dropped_span_count == 5
    assert tracer.forgotten_drop_identity_count == 0, "no failover span was ever dropped"
    assert tracer.attribute_failover_span(ordinary.span_id).fate is SpanFate.UNRECORDED


def test_clear_discards_and_accounts_for_in_flight_spans() -> None:
    """clear() accounts for in-flight spans and retains in-flight failover span IDs (#205)."""
    tracer = TelemetryTracer()

    # Start an in-flight failover span and an ordinary in-flight span
    failover_span_id = tracer.start_span(FAILOVER_EVENT_SPAN_NAME)
    ordinary_span_id = tracer.start_span("agent.step")

    # Start and end a completed span
    completed_span_id = tracer.start_span("agent.completed")
    tracer.end_span(span_id=completed_span_id)

    assert tracer.active_span_count == 2
    assert len(tracer.get_completed_spans()) == 1
    assert tracer.dropped_span_count == 0

    tracer.clear()

    assert tracer.active_span_count == 0
    assert len(tracer.get_completed_spans()) == 0
    assert tracer.dropped_span_count == 3
    assert tracer.drop_reasons.get("cleared") == 1
    assert tracer.drop_reasons.get("cleared_in_flight") == 2

    # In-flight failover span attribution should report DROPPED with cleared_in_flight reason
    attr = tracer.attribute_failover_span(failover_span_id)
    assert attr.fate is SpanFate.DROPPED
    assert attr.drop_reason == "cleared_in_flight"
    assert attr.identities_forgotten == 0

    # Ordinary in-flight span reports UNRECORDED
    assert tracer.attribute_failover_span(ordinary_span_id).fate is SpanFate.UNRECORDED


def test_buffer_evicted_span_count_and_dropped_span_count_alias_are_identical() -> None:
    """`buffer_evicted_span_count` is the primary counter name and `dropped_span_count` is an alias (#206)."""
    tracer = TelemetryTracer(max_completed_spans=2)
    for _ in range(5):
        tracer.end_span(span_id=tracer.start_span("agent.run"))

    assert tracer.buffer_evicted_span_count == 3
    assert tracer.dropped_span_count == 3
    assert tracer.buffer_evicted_span_count == tracer.dropped_span_count
