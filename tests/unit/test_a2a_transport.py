"""Unit tests for A2A dual-transport engine, discovery service, and HTTP/SSE wire server."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import ValidationError

from uclone_x.a2a import (
    A2ADiscoveryService,
    A2AHttpTransport,
    A2AInMemoryTransport,
    AgentCard,
    TaskMessage,
    TaskResult,
    TaskStatus,
    WireProtocolType,
)
from uclone_x.a2a.wire import (
    agent_event_from_wire_json,
    agent_event_to_wire,
    agent_event_to_wire_json,
    task_result_from_wire_json,
    task_result_to_wire,
    task_result_to_wire_json,
)
from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    Provenance,
    ServiceRef,
)
from uclone_x.engine.event_bus import (
    AgentEvent,
    EventPriority,
    EventSource,
    EventType,
)
from uclone_x.errors import (
    InvalidAgentResponseError,
    MissingProvenanceError,
    TaskNotFoundError,
)
from uclone_x.shells.a2a_server import A2AServer


@pytest.fixture
def sample_agent_card() -> AgentCard:
    return AgentCard(
        name="SecurityReviewer",
        description="Static security scanner for repository analysis",
        version="1.0.1",
        skills=("cve_scan", "ast_lint"),
        endpoints={"rest": "http://127.0.0.1:8000/a2a"},
    )


@pytest.fixture
def sample_task_message() -> TaskMessage:
    return TaskMessage(
        task_id="task-001",
        session_id="session-001",
        sender_agent_id="supervisor",
        target_agent_id="security_auditor",
        input_data={"scan_dir": "/workspace"},
        metadata={"priority": "high"},
    )


# ======================================================================================
# 1. In-Memory Transport Tests (Zero-Copy Fastpath)
# ======================================================================================


async def test_in_memory_transport_type() -> None:
    transport = A2AInMemoryTransport()
    assert transport.transport_type == WireProtocolType.LOCAL_IN_MEMORY


async def test_in_memory_send_task_success(sample_task_message: TaskMessage) -> None:
    transport = A2AInMemoryTransport()
    received_messages: list[TaskMessage] = []

    async def mock_handler(msg: TaskMessage) -> TaskResult:
        received_messages.append(msg)
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            output_data={"vulnerabilities_found": 0},
            provenance=Provenance.primary("security_auditor"),
        )

    transport.register_handler("security_auditor", mock_handler)
    result = await transport.send_task("security_auditor", sample_task_message)

    assert result.task_id == "task-001"
    assert result.status == TaskStatus.COMPLETED
    assert result.output_data["vulnerabilities_found"] == 0
    assert result.provenance is not None
    # Verify zero-copy object identity
    assert len(received_messages) == 1
    assert received_messages[0] is sample_task_message


async def test_in_memory_send_task_missing_provenance_fails(
    sample_task_message: TaskMessage,
) -> None:
    transport = A2AInMemoryTransport()

    async def mock_handler(msg: TaskMessage) -> TaskResult:
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            provenance=None,
        )

    transport.register_handler("security_auditor", mock_handler)
    with pytest.raises(MissingProvenanceError, match="must contain in-band provenance"):
        await transport.send_task("security_auditor", sample_task_message)


async def test_in_memory_send_task_not_found(sample_task_message: TaskMessage) -> None:
    transport = A2AInMemoryTransport()
    with pytest.raises(TaskNotFoundError, match="No in-memory handler registered"):
        await transport.send_task("unknown_endpoint", sample_task_message)


async def test_in_memory_unregister_handler(sample_task_message: TaskMessage) -> None:
    transport = A2AInMemoryTransport()

    async def mock_handler(msg: TaskMessage) -> TaskResult:
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            provenance=Provenance.primary("agent"),
        )

    transport.register_handler("endpoint-1", mock_handler)
    transport.unregister_handler("endpoint-1")

    with pytest.raises(TaskNotFoundError):
        await transport.send_task("endpoint-1", sample_task_message)


async def test_in_memory_stream_task_with_custom_stream_handler(
    sample_task_message: TaskMessage,
) -> None:
    transport = A2AInMemoryTransport()

    async def mock_stream_handler(msg: TaskMessage) -> AsyncIterator[str]:
        yield f"chunk-1 for {msg.task_id}"
        yield f"chunk-2 for {msg.task_id}"

    transport.register_stream_handler("security_auditor", mock_stream_handler)

    chunks: list[str] = []
    async for chunk in transport.stream_task("security_auditor", sample_task_message):
        chunks.append(chunk)

    assert chunks == ["chunk-1 for task-001", "chunk-2 for task-001"]


async def test_in_memory_stream_task_fallback_to_single_turn_handler(
    sample_task_message: TaskMessage,
) -> None:
    transport = A2AInMemoryTransport()

    async def mock_handler(msg: TaskMessage) -> TaskResult:
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            output_data={"done": True},
            provenance=Provenance.primary("agent"),
        )

    transport.register_handler("security_auditor", mock_handler)

    chunks: list[str] = []
    async for chunk in transport.stream_task("security_auditor", sample_task_message):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "task-001" in chunks[0]


async def test_in_memory_stream_task_not_found(sample_task_message: TaskMessage) -> None:
    transport = A2AInMemoryTransport()
    with pytest.raises(TaskNotFoundError):
        async for _ in transport.stream_task("unregistered", sample_task_message):
            pass


# ======================================================================================
# 2. A2ADiscoveryService Tests
# ======================================================================================


def test_discovery_service_local_card(sample_agent_card: AgentCard) -> None:
    discovery = A2ADiscoveryService(local_agent_card=sample_agent_card)
    assert discovery.get_local_agent_card() == sample_agent_card

    new_card = AgentCard(name="NewAgent", description="Updated")
    discovery.set_local_agent_card(new_card)
    assert discovery.get_local_agent_card().name == "NewAgent"


async def test_discovery_service_default_card() -> None:
    discovery = A2ADiscoveryService()
    card = discovery.get_local_agent_card()
    assert card.name == "uclone-x-agent"
    assert card.version == "1.0.1"


# ======================================================================================
# 3. Remote Server & HTTP Transport Integration (Dynamic Port port=0)
# ======================================================================================


async def test_server_unstarted_port_raises(sample_agent_card: AgentCard) -> None:
    server = A2AServer(agent_card=sample_agent_card, port=0)
    with pytest.raises(RuntimeError, match="Server is not running"):
        _ = server.port


async def test_server_and_http_transport_full_lifecycle(
    sample_agent_card: AgentCard,
    sample_task_message: TaskMessage,
) -> None:
    async def mock_handler(msg: TaskMessage) -> TaskResult:
        if msg.input_data.get("trigger_error"):
            raise TaskNotFoundError(f"Subtask not found in {msg.task_id}")
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            output_data={"result": "analysis_ok"},
            provenance=Provenance.primary("remote_security_auditor"),
        )

    async def mock_stream_handler(msg: TaskMessage) -> AsyncIterator[str]:
        yield f"stream-event-1:{msg.task_id}"
        yield f"stream-event-2:{msg.task_id}"

    server = A2AServer(
        agent_card=sample_agent_card,
        handler=mock_handler,
        stream_handler=mock_stream_handler,
        host="127.0.0.1",
        port=0,  # dynamic ephemeral port
    )

    await server.start()
    try:
        assert server.port > 0
        assert server.url == f"http://127.0.0.1:{server.port}"
        assert server.get_local_agent_card() == sample_agent_card

        transport = A2AHttpTransport(default_agent_card=sample_agent_card)
        assert transport.transport_type == WireProtocolType.REST_SSE
        assert transport.get_local_agent_card() == sample_agent_card

        # 1. Discovery fetch
        remote_card = await transport.fetch_remote_agent_card(server.url)
        assert remote_card.name == sample_agent_card.name
        assert remote_card.version == "1.0.1"
        assert remote_card.skills == sample_agent_card.skills

        # 2. send_task success
        result = await transport.send_task(server.url, sample_task_message)
        assert result.task_id == "task-001"
        assert result.status == TaskStatus.COMPLETED
        assert result.output_data["result"] == "analysis_ok"
        assert result.provenance is not None
        assert result.provenance.served_by.provider == "remote_security_auditor"

        # 3. stream_task success
        stream_lines: list[str] = []
        async for line in transport.stream_task(server.url, sample_task_message):
            stream_lines.append(line)

        assert any("stream-event-1:task-001" in s for s in stream_lines)
        assert any("stream-event-2:task-001" in s for s in stream_lines)

        # 4. Error mapping: TaskNotFoundError
        error_msg = TaskMessage(
            task_id="err-task",
            session_id="s1",
            sender_agent_id="sender",
            target_agent_id="target",
            input_data={"trigger_error": True},
        )
        with pytest.raises(TaskNotFoundError, match="Subtask not found in err-task"):
            await transport.send_task(server.url, error_msg)

    finally:
        await server.stop()


async def test_server_version_negotiation_enforcement(
    sample_agent_card: AgentCard,
    sample_task_message: TaskMessage,
) -> None:
    server = A2AServer(agent_card=sample_agent_card, port=0)
    await server.start()
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            # Bad A2A-Version header
            resp = await client.get(
                f"{server.url}/.well-known/agent-card.json",
                headers={"A2A-Version": "0.3"},
            )
            assert resp.status_code == 400
            assert "VersionNotSupportedError" in resp.text

            # Send task with unsupported version
            resp_post = await client.post(
                f"{server.url}/message:send",
                headers={"A2A-Version": "2.0"},
                json=sample_task_message.model_dump(mode="json"),
            )
            assert resp_post.status_code == 400
            assert "VersionNotSupportedError" in resp_post.text
    finally:
        await server.stop()


async def test_http_transport_fetch_card_errors() -> None:
    def _fail_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Network error: connection refused", request=request)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(_fail_connect))
    transport = A2AHttpTransport(http_client=mock_client)

    # Network error on unreachable port
    with pytest.raises(InvalidAgentResponseError, match="Network error"):
        await transport.fetch_remote_agent_card("http://127.0.0.1:59999")


async def test_discovery_service_fetch_card_invalid_payload(
    sample_agent_card: AgentCard,
) -> None:
    server = A2AServer(agent_card=sample_agent_card, port=0)
    await server.start()
    try:
        discovery = A2ADiscoveryService()
        # Fetching non-existent endpoint
        with pytest.raises(TaskNotFoundError):
            await discovery.fetch_remote_agent_card(f"{server.url}/non_existent")
    finally:
        await server.stop()


async def test_server_without_handler_returns_404(
    sample_agent_card: AgentCard,
    sample_task_message: TaskMessage,
) -> None:
    server = A2AServer(agent_card=sample_agent_card, handler=None, port=0)
    await server.start()
    try:
        transport = A2AHttpTransport()
        with pytest.raises(TaskNotFoundError, match="No task handler registered"):
            await transport.send_task(server.url, sample_task_message)

        with pytest.raises(TaskNotFoundError):
            async for _ in transport.stream_task(server.url, sample_task_message):
                pass
    finally:
        await server.stop()


async def test_server_handler_missing_provenance_returns_500(
    sample_agent_card: AgentCard,
    sample_task_message: TaskMessage,
) -> None:
    async def bad_handler(msg: TaskMessage) -> TaskResult:
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            provenance=None,
        )

    server = A2AServer(agent_card=sample_agent_card, handler=bad_handler, port=0)
    await server.start()
    try:
        transport = A2AHttpTransport()
        with pytest.raises(InvalidAgentResponseError):
            await transport.send_task(server.url, sample_task_message)
    finally:
        await server.stop()


async def test_server_stream_fallback_handler(
    sample_agent_card: AgentCard,
    sample_task_message: TaskMessage,
) -> None:
    async def single_handler(msg: TaskMessage) -> TaskResult:
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            output_data={"single": True},
            provenance=Provenance.primary("agent"),
        )

    server = A2AServer(
        agent_card=sample_agent_card,
        handler=single_handler,
        stream_handler=None,
        port=0,
    )
    await server.start()
    try:
        transport = A2AHttpTransport()
        chunks: list[str] = []
        async for chunk in transport.stream_task(server.url, sample_task_message):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert any("single" in c for c in chunks)
    finally:
        await server.stop()


# ======================================================================================
# 5. Wire-Serialisation Boundary (P2 interoperability, P6 in-band provenance)
# ======================================================================================


def _failover_provenance() -> Provenance:
    """A provenance whose every awkward shape is exercised: enum, tuple, degraded."""
    return Provenance(
        path=ExecutionPath.FAILOVER,
        requested=ServiceRef(provider="primary_llm", model="p-1"),
        served_by=ServiceRef(provider="secondary_llm", model="s-1"),
        attempts=(
            AttemptRecord(
                provider="primary_llm",
                model="p-1",
                error_class="RateLimitError",
                status_code=429,
            ),
        ),
    )


def test_wire_payload_omits_derived_provenance_field() -> None:
    """`degraded` is computed, so it is never asserted on the wire."""
    result = TaskResult(
        task_id="task-001",
        status=TaskStatus.COMPLETED,
        provenance=_failover_provenance(),
    )

    payload = task_result_to_wire(result)
    provenance_payload = payload["provenance"]
    assert isinstance(provenance_payload, dict)
    assert "degraded" not in provenance_payload
    assert "degraded" not in task_result_to_wire_json(result)

    # The authoritative fields that `degraded` is derived from are still carried.
    assert provenance_payload["requested"] == {"provider": "primary_llm", "model": "p-1"}
    assert provenance_payload["served_by"] == {"provider": "secondary_llm", "model": "s-1"}


def test_wire_round_trip_recomputes_degraded_and_preserves_attempts() -> None:
    """A JSON round trip through the boundary validates under the strict models."""
    original = TaskResult(
        task_id="task-001",
        status=TaskStatus.COMPLETED,
        output_data={"result": "analysis_ok"},
        provenance=_failover_provenance(),
    )

    parsed = task_result_from_wire_json(task_result_to_wire_json(original))

    assert parsed == original
    assert parsed.status is TaskStatus.COMPLETED
    provenance = parsed.provenance
    assert provenance is not None
    assert provenance.path is ExecutionPath.FAILOVER
    assert isinstance(provenance.attempts, tuple)
    assert provenance.attempts[0].error_class == "RateLimitError"
    # Recomputed by the receiving model rather than trusted from the payload.
    assert provenance.degraded is True


def test_wire_ignores_peer_asserted_degraded() -> None:
    """A peer cannot report a substituted result as clean (P6): peer-supplied degraded is ignored and recomputed.

    Named for what the code does. `Provenance` strips the key and recomputes the value;
    it does not raise. `a2a/wire.py`'s module docstring cites this test by name as the
    evidence for "corrected, not raised on", so the name has to match the assertion.
    """
    payload = task_result_to_wire(TaskResult(task_id="task-001", provenance=_failover_provenance()))
    provenance_payload = payload["provenance"]
    assert isinstance(provenance_payload, dict)
    provenance_payload["degraded"] = False

    restored = task_result_from_wire_json(json.dumps(payload))
    assert restored.provenance is not None
    assert restored.provenance.degraded is True


async def test_http_transport_round_trips_failover_provenance(
    sample_agent_card: AgentCard,
    sample_task_message: TaskMessage,
) -> None:
    """End-to-end: a degraded result survives the HTTP binding without being flattened."""

    async def failover_handler(msg: TaskMessage) -> TaskResult:
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            provenance=_failover_provenance(),
        )

    server = A2AServer(agent_card=sample_agent_card, handler=failover_handler, port=0)
    await server.start()
    try:
        transport = A2AHttpTransport()
        result = await transport.send_task(server.url, sample_task_message)

        provenance = result.provenance
        assert provenance is not None
        assert provenance.path is ExecutionPath.FAILOVER
        assert provenance.degraded is True
        assert provenance.served_by.provider == "secondary_llm"
        assert len(provenance.attempts) == 1
        assert provenance.attempts[0].status_code == 429
    finally:
        await server.stop()


def test_task_message_rejects_unserializable_payload_at_construction() -> None:
    """Issue #35: A2A payload fields reject unserializable objects at construction time (P8 / P6)."""

    class NonJsonCustomObject:
        pass

    with pytest.raises(ValidationError):
        TaskMessage(
            task_id="task-001",
            session_id="session-001",
            sender_agent_id="agent-001",
            target_agent_id="agent-002",
            input_data={"bad_set": {1, 2}},  # type: ignore[dict-item]
        )

    with pytest.raises(ValidationError):
        TaskMessage(
            task_id="task-002",
            session_id="session-001",
            sender_agent_id="agent-001",
            target_agent_id="agent-002",
            input_data={"bad_obj": NonJsonCustomObject()},  # type: ignore[dict-item]
        )


def _reply_event() -> AgentEvent:
    return AgentEvent(
        event_id="evt_wire_1",
        topic="session.sess_1",
        type=EventType.AGENT_REPLY,
        source=EventSource.AGENT,
        sender_id="agt_1",
        recipient_id="caller_1",
        session_id="sess_1",
        priority=EventPriority.NORMAL,
        sequence=9,
        payload={"content": "answered", "tokens": [1, 2, 3]},
        provenance=_failover_provenance(),
    )


def test_agent_event_wire_payload_omits_derived_provenance_field() -> None:
    """The event envelope obeys the same derived-field policy as `TaskResult` (#117)."""
    event = _reply_event()

    payload = agent_event_to_wire(event)
    provenance_payload = payload["provenance"]
    assert isinstance(provenance_payload, dict)
    assert "degraded" not in provenance_payload
    assert "degraded" not in agent_event_to_wire_json(event)

    # The authoritative fields `degraded` is derived from are still carried.
    assert provenance_payload["requested"] == {"provider": "primary_llm", "model": "p-1"}
    assert provenance_payload["served_by"] == {"provider": "secondary_llm", "model": "s-1"}


def test_agent_event_wire_round_trip_preserves_envelope_and_provenance() -> None:
    """A full event round trip validates under `strict=True` and rebuilds the frozen payload."""
    event = _reply_event()

    parsed = agent_event_from_wire_json(agent_event_to_wire_json(event))

    assert parsed == event
    assert parsed.type is EventType.AGENT_REPLY
    assert parsed.source is EventSource.AGENT
    assert parsed.priority is EventPriority.NORMAL
    assert parsed.sequence == 9
    # Payload comes back deep-frozen, as it is on the fastpath (event_bus section 3.4).
    assert parsed.payload["tokens"] == (1, 2, 3)
    provenance = parsed.provenance
    assert provenance is not None
    assert provenance.path is ExecutionPath.FAILOVER
    assert isinstance(provenance.attempts, tuple)
    assert provenance.degraded is True


def test_agent_event_wire_ignores_peer_asserted_degraded() -> None:
    """A peer cannot report a substituted result as clean on the event envelope either."""
    payload = agent_event_to_wire(_reply_event())
    provenance_payload = payload["provenance"]
    assert isinstance(provenance_payload, dict)
    provenance_payload["degraded"] = False

    restored = agent_event_from_wire_json(json.dumps(payload))
    assert restored.provenance is not None
    assert restored.provenance.degraded is True


def test_agent_event_wire_ingress_needs_json_not_a_decoded_dict() -> None:
    """Asymmetry (2) in `a2a/wire.py`: strict mode accepts the JSON forms only while parsing JSON."""
    decoded = agent_event_to_wire(_reply_event())

    with pytest.raises(ValidationError):
        AgentEvent.model_validate(decoded)

    assert agent_event_from_wire_json(json.dumps(decoded)).provenance is not None


def test_wire_unattributed_task_result_round_trip() -> None:
    """TaskResult with provenance=None serializes with provenance: null on wire and parses back to provenance=None."""
    result = TaskResult(
        task_id="task-unattr-001",
        status=TaskStatus.COMPLETED,
        output_data={"result": "ok"},
        provenance=None,
    )
    wire_dict = task_result_to_wire(result)
    assert wire_dict["provenance"] is None

    wire_json = task_result_to_wire_json(result)
    assert '"provenance":null' in wire_json or '"provenance": null' in wire_json

    parsed = task_result_from_wire_json(wire_json)
    assert parsed.provenance is None
