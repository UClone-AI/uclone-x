"""Unit and integration tests for A2A CLI commands and federated gateway (Issue #69)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest
from typer.testing import CliRunner

from uclone_x.a2a.models import AgentCard, TaskMessage, TaskResult, TaskStatus
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, TurnResult
from uclone_x.cli import main
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import AgentEvent, EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.shells.a2a_server import A2AServer, ManagedTaskRecord

runner = CliRunner()


@pytest.fixture
def sample_agent_card() -> AgentCard:
    return AgentCard(
        name="FederatedAuditor",
        description="Autonomous A2A agent node for security analysis",
        version="1.0.1",
        skills=("cve_scan", "threat_modeling"),
        endpoints={"http": "http://127.0.0.1:8080"},
    )


# ======================================================================================
# 1. A2A Server Discovery and Task Endpoints
# ======================================================================================


async def test_a2a_server_discovery_endpoint(sample_agent_card: AgentCard) -> None:
    """Test RFC 8615 GET /.well-known/agent-card.json endpoint on ephemeral port."""
    server = A2AServer(agent_card=sample_agent_card, port=0)
    await server.start()
    try:
        assert server.port > 0
        async with httpx.AsyncClient() as client:
            # 1. Success with default or explicit A2A-Version
            resp = await client.get(
                f"{server.url}/.well-known/agent-card.json",
                headers={"A2A-Version": "1.0"},
            )
            assert resp.status_code == 200
            card_json = resp.json()
            assert card_json["name"] == "FederatedAuditor"
            assert card_json["version"] == "1.0.1"
            assert "cve_scan" in card_json["skills"]
            assert resp.headers["cache-control"] == "max-age=3600"
            assert resp.headers["etag"] == '"1.0.1"'

            # 2. Version check rejection
            bad_resp = await client.get(
                f"{server.url}/.well-known/agent-card.json",
                headers={"A2A-Version": "0.3"},
            )
            assert bad_resp.status_code == 400
            assert "VersionNotSupportedError" in bad_resp.text
    finally:
        await server.stop()


async def test_a2a_server_task_lifecycle_with_base_agent(
    sample_agent_card: AgentCard,
) -> None:
    """Test full task creation, execution, polling, and SSE streaming with live BaseAgent."""
    agent_config = AgentConfig(
        agent_id="test_agent",
        name="TestAgent",
        system_prompt="You are a test agent.",
    )
    bus = EventBus()
    agent = BaseAgent(config=agent_config, bus=bus, llm=MockLLMConnector())

    server = A2AServer(
        agent_card=sample_agent_card,
        agent=agent,
        bus=bus,
        port=0,
    )
    await server.start()
    try:
        async with httpx.AsyncClient() as client:
            # 1. POST /a2a/v1/tasks
            payload = {
                "task_id": "task-test-100",
                "input_data": {"prompt": "Analyze security vulnerabilities"},
                "metadata": {"priority": "high"},
            }
            create_resp = await client.post(
                f"{server.url}/a2a/v1/tasks",
                headers={"A2A-Version": "1.0"},
                json=payload,
            )
            assert create_resp.status_code == 201
            data = create_resp.json()
            assert data["task_id"] == "task-test-100"
            assert data["status"] in ("working", "completed")

            # 2. Poll GET /a2a/v1/tasks/{task_id} until completed
            poll_data: dict[str, Any] = {}
            for _ in range(30):
                poll_resp = await client.get(
                    f"{server.url}/a2a/v1/tasks/task-test-100",
                    headers={"A2A-Version": "1.0"},
                )
                assert poll_resp.status_code == 200
                poll_data = cast(dict[str, Any], poll_resp.json())
                if poll_data.get("status") == "completed":
                    break
                await asyncio.sleep(0.05)

            assert poll_data.get("status") == "completed"
            output_data = cast(dict[str, Any], poll_data.get("output_data", {}))
            assert "result" in output_data
            artifacts = cast(list[dict[str, Any]], poll_data.get("artifacts", []))
            assert len(artifacts) > 0
            assert poll_data.get("provenance") is not None

            # 3. GET /a2a/v1/tasks/{task_id}/events (SSE stream replay)
            async with client.stream(
                "GET",
                f"{server.url}/a2a/v1/tasks/task-test-100/events",
                headers={"A2A-Version": "1.0"},
            ) as sse_resp:
                assert sse_resp.status_code == 200
                assert "text/event-stream" in sse_resp.headers["content-type"]
                events: list[dict[str, Any]] = []
                async for line in sse_resp.aiter_lines():
                    if line.startswith("data: "):
                        evt_json = json.loads(line[6:])
                        events.append(evt_json)

                assert len(events) >= 2
                event_names = [e.get("event") for e in events]
                assert "status_changed" in event_names
                assert "completed" in event_names
    finally:
        await server.stop()


async def test_a2a_server_task_cancellation(sample_agent_card: AgentCard) -> None:
    """Test canceling a running or pending task via POST /a2a/v1/tasks/{task_id}/cancel."""

    async def slow_handler(msg: TaskMessage) -> TaskResult:
        await asyncio.sleep(5.0)
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            provenance=Provenance.primary("agent"),
        )

    server = A2AServer(
        agent_card=sample_agent_card,
        handler=slow_handler,
        port=0,
    )
    await server.start()
    try:
        async with httpx.AsyncClient() as client:
            # 1. Create slow task
            create_resp = await client.post(
                f"{server.url}/a2a/v1/tasks",
                headers={"A2A-Version": "1.0"},
                json={"task_id": "cancel-task-1", "input_data": {"slow": True}},
            )
            assert create_resp.status_code == 201
            assert create_resp.json()["status"] == "working"

            # 2. Cancel task immediately
            cancel_resp = await client.post(
                f"{server.url}/a2a/v1/tasks/cancel-task-1/cancel",
                headers={"A2A-Version": "1.0"},
            )
            assert cancel_resp.status_code == 200
            assert cancel_resp.json()["status"] == "canceled"

            # 3. Verify status after cancellation
            get_resp = await client.get(
                f"{server.url}/a2a/v1/tasks/cancel-task-1",
                headers={"A2A-Version": "1.0"},
            )
            assert get_resp.status_code == 200
            assert get_resp.json()["status"] == "canceled"

            # 4. Canceling already terminal task returns 200
            cancel_again = await client.post(
                f"{server.url}/a2a/v1/tasks/cancel-task-1/cancel",
                headers={"A2A-Version": "1.0"},
            )
            assert cancel_again.status_code == 200
            assert cancel_again.json()["status"] == "canceled"
    finally:
        await server.stop()


async def test_a2a_server_task_not_found_errors(sample_agent_card: AgentCard) -> None:
    """Test 404 responses for non-existent tasks across all task endpoints."""
    server = A2AServer(agent_card=sample_agent_card, port=0)
    await server.start()
    try:
        async with httpx.AsyncClient() as client:
            get_resp = await client.get(
                f"{server.url}/a2a/v1/tasks/unknown-task",
                headers={"A2A-Version": "1.0"},
            )
            assert get_resp.status_code == 404
            assert "TaskNotFoundError" in get_resp.text

            sse_resp = await client.get(
                f"{server.url}/a2a/v1/tasks/unknown-task/events",
                headers={"A2A-Version": "1.0"},
            )
            assert sse_resp.status_code == 404
            assert "TaskNotFoundError" in sse_resp.text

            cancel_resp = await client.post(
                f"{server.url}/a2a/v1/tasks/unknown-task/cancel",
                headers={"A2A-Version": "1.0"},
            )
            assert cancel_resp.status_code == 404
            assert "TaskNotFoundError" in cancel_resp.text
    finally:
        await server.stop()


async def test_a2a_server_task_default_echo_execution(
    sample_agent_card: AgentCard,
) -> None:
    """Test fallback echo execution when neither agent nor handler is configured."""
    server = A2AServer(agent_card=sample_agent_card, port=0)
    await server.start()
    try:
        async with httpx.AsyncClient() as client:
            # Auto-generated task_id and session_id with direct prompt field
            create_resp = await client.post(
                f"{server.url}/a2a/v1/tasks",
                headers={"A2A-Version": "1.0"},
                json={"prompt": "Echo this prompt"},
            )
            assert create_resp.status_code == 201
            task_id = create_resp.json()["task_id"]
            assert task_id.startswith("task_")

            await asyncio.sleep(0.05)
            get_resp = await client.get(
                f"{server.url}/a2a/v1/tasks/{task_id}",
                headers={"A2A-Version": "1.0"},
            )
            assert get_resp.status_code == 200
            data = get_resp.json()
            assert data["status"] == "completed"
            assert "Executed: Echo this prompt" in data["output_data"]["result"]
    finally:
        await server.stop()


async def test_a2a_server_custom_handler_failure(
    sample_agent_card: AgentCard,
) -> None:
    """Test error handling when task handler raises an exception."""

    async def failing_handler(msg: TaskMessage) -> TaskResult:
        raise ValueError("Simulated handler crash")

    server = A2AServer(
        agent_card=sample_agent_card,
        handler=failing_handler,
        port=0,
    )
    await server.start()
    try:
        async with httpx.AsyncClient() as client:
            create_resp = await client.post(
                f"{server.url}/a2a/v1/tasks",
                headers={"A2A-Version": "1.0"},
                json={"task_id": "fail-task-1", "input_data": {}},
            )
            assert create_resp.status_code == 201

            await asyncio.sleep(0.05)
            get_resp = await client.get(
                f"{server.url}/a2a/v1/tasks/fail-task-1",
                headers={"A2A-Version": "1.0"},
            )
            assert get_resp.status_code == 200
            data = get_resp.json()
            assert data["status"] == "failed"
            assert data["error"] == "InternalError: ValueError: Simulated handler crash"
            assert data["provenance"] is None
            assert len(server.processing_errors) == 1
            err = server.processing_errors[0]
            assert err["task_id"] == "fail-task-1"
            assert err["error_class"] == "ValueError"
            assert err["error_message"] == "Simulated handler crash"
            assert "ValueError: Simulated handler crash" in err["traceback"]
            assert isinstance(err["timestamp"], float)
    finally:
        await server.stop()


async def test_a2a_server_unattributed_turn_failure_preserves_null_provenance(
    sample_agent_card: AgentCard,
) -> None:
    """Test that failed unattributed turns report provenance: null and are NOT fabricated (Issue #157, P6)."""
    agent_config = AgentConfig(
        agent_id="failing_unattributed_agent",
        name="FailingUnattributedAgent",
        system_prompt="Test agent with failed unattributed turns.",
    )
    bus = EventBus()

    class FailingUnattributedAgent(BaseAgent):
        async def execute_turn(  # noqa: D102
            self, input_data: str | AgentEvent, *, continuation: bool = False, **kwargs: Any
        ) -> TurnResult:
            return TurnResult(
                turn_index=1,
                content="",
                is_completed=False,
                error="Turn execution failed without attribution",
                provenance=None,
            )

    agent = FailingUnattributedAgent(config=agent_config, bus=bus, llm=MockLLMConnector())

    server = A2AServer(
        agent_card=sample_agent_card,
        agent=agent,
        bus=bus,
        port=0,
    )
    await server.start()
    try:
        async with httpx.AsyncClient() as client:
            create_resp = await client.post(
                f"{server.url}/a2a/v1/tasks",
                headers={"A2A-Version": "1.0"},
                json={
                    "task_id": "unattr-fail-task-1",
                    "input_data": {"prompt": "Run failing turn"},
                },
            )
            assert create_resp.status_code == 201

            poll_data: dict[str, Any] = {}
            for _ in range(30):
                poll_resp = await client.get(
                    f"{server.url}/a2a/v1/tasks/unattr-fail-task-1",
                    headers={"A2A-Version": "1.0"},
                )
                assert poll_resp.status_code == 200
                poll_data = cast(dict[str, Any], poll_resp.json())
                if poll_data.get("status") == "failed":
                    break
                await asyncio.sleep(0.05)

            assert poll_data.get("status") == "failed"
            assert "Turn execution failed without attribution" in str(poll_data.get("error"))
            assert poll_data.get("provenance") is None
    finally:
        await server.stop()


# ======================================================================================
# 2. CLI Command Integration Tests ('./ucx a2a serve')
# ======================================================================================


def test_cli_a2a_serve_default_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test './ucx a2a serve' command invocation with default arguments."""
    mock_uvicorn = MagicMock()
    monkeypatch.setattr("uvicorn.run", mock_uvicorn)

    result = runner.invoke(main.app, ["a2a", "serve", "--port", "8080"])
    assert result.exit_code == 0
    assert mock_uvicorn.called
    _, kwargs = mock_uvicorn.call_args
    assert kwargs.get("port") == 8080
    assert kwargs.get("host") == "127.0.0.1"
    assert kwargs.get("log_level") == "info"


def test_cli_a2a_serve_with_custom_card_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test './ucx a2a serve' with custom JSON AgentCard configuration file."""
    mock_uvicorn = MagicMock()
    monkeypatch.setattr("uvicorn.run", mock_uvicorn)

    card_file = tmp_path / "agent-card.json"
    card_data = {
        "name": "CustomSecurityAgent",
        "description": "Custom agent card for test",
        "version": "1.0.1",
        "skills": ["static_analysis"],
        "endpoints": {"http": "http://127.0.0.1:9090"},
    }
    card_file.write_text(json.dumps(card_data), encoding="utf-8")

    result = runner.invoke(
        main.app,
        [
            "a2a",
            "serve",
            "--port",
            "9090",
            "--host",
            "0.0.0.0",
            "--agent-card",
            str(card_file),
        ],
    )
    assert result.exit_code == 0
    assert mock_uvicorn.called
    _, kwargs = mock_uvicorn.call_args
    assert kwargs.get("port") == 9090
    assert kwargs.get("host") == "0.0.0.0"


def test_cli_a2a_serve_missing_card_file_error() -> None:
    """Test './ucx a2a serve' fails fast when specified card file does not exist."""
    result = runner.invoke(
        main.app,
        ["a2a", "serve", "--agent-card", "/non/existent/path/agent.json"],
    )
    assert result.exit_code == 1
    assert "Agent card file not found" in result.output


def test_cli_a2a_serve_invalid_json_card_error(tmp_path: Path) -> None:
    """Test './ucx a2a serve' fails fast when specified card file has invalid JSON."""
    bad_card = tmp_path / "bad-card.json"
    bad_card.write_text("invalid json content", encoding="utf-8")

    result = runner.invoke(
        main.app,
        ["a2a", "serve", "--agent-card", str(bad_card)],
    )
    assert result.exit_code == 1
    assert "Error parsing agent card" in result.output


def test_cli_a2a_serve_dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test './ucx a2a serve --dev' sets reload and debug log level."""
    mock_uvicorn = MagicMock()
    monkeypatch.setattr("uvicorn.run", mock_uvicorn)

    result = runner.invoke(
        main.app,
        ["a2a", "serve", "--dev", "--agent-id", "dev_agent", "--port", "8085"],
    )
    assert result.exit_code == 0
    assert mock_uvicorn.called
    _, kwargs = mock_uvicorn.call_args
    assert kwargs.get("reload") is True
    assert kwargs.get("log_level") == "debug"
    assert kwargs.get("port") == 8085


# ======================================================================================
# P6: the gateway refuses to attribute what it did not produce (issue #157)
# ======================================================================================


class _UnattributedAgent:
    """An agent whose turn carries no provenance — the normal input after #136.

    #136 stopped `execute_turn` substituting a synthetic `agent.core/BaseAgent` value
    and made it propagate a connector's `None` verbatim, so this is not a contrived
    state: it is what a connector that states nothing produces today.
    """

    def __init__(self, agent_id: str = "agent-alpha", is_completed: bool = True) -> None:
        self.agent_id = agent_id
        self._is_completed = is_completed

    async def execute_turn(  # noqa: D102
        self, input_data: object, *, continuation: bool = False
    ) -> TurnResult:
        return TurnResult(
            turn_index=1,
            content="an answer nobody claims",
            is_completed=self._is_completed,
            error=None if self._is_completed else "provider down",
            provenance=None,
        )


async def test_gateway_refuses_to_attribute_an_unattributed_turn(
    sample_agent_card: AgentCard,
) -> None:
    """An unattributed turn fails the task; it is never fabricated onto the wire (#157).

    The gateway used to write `turn_result.provenance or Provenance.primary(agent_id)`,
    so a turn that stated nothing reached a remote peer as
    `path=primary, requested == served_by == <agent id>, degraded=False` — an
    attribution invented by the component forwarding the value, indistinguishable at the
    peer from one the producer asserted. This is the same shape #136 deleted from
    `agent/base.py`, and it crosses a trust boundary.
    """
    server = A2AServer(
        agent_card=sample_agent_card,
        agent=cast(BaseAgent, _UnattributedAgent()),
        port=0,
    )
    record = ManagedTaskRecord(
        task_id="task-unattributed",
        session_id="sess-1",
        input_data={"prompt": "who answered this?"},
    )
    await server._execute_managed_task(record)  # pyright: ignore[reportPrivateUsage]

    wire = record.to_dict()
    # The peer sees an explicit absence and a stated reason — not a clean primary.
    assert wire["provenance"] is None
    assert wire["status"] == TaskStatus.FAILED.value
    assert "no provenance" in str(wire["error"])
    assert "agent-alpha" in str(wire["error"])
    # And the unattributed content is not published as a completed result.
    assert wire["output_data"] == {}
    assert wire["artifacts"] == []


async def test_gateway_refusal_preserves_the_turns_own_error(
    sample_agent_card: AgentCard,
) -> None:
    """A turn that both failed and stated no provenance reports both facts (#157).

    The refusal must not overwrite the diagnosis the caller actually needs.
    """
    server = A2AServer(
        agent_card=sample_agent_card,
        agent=cast(BaseAgent, _UnattributedAgent(is_completed=False)),
        port=0,
    )
    record = ManagedTaskRecord(
        task_id="task-unattributed-failed",
        session_id="sess-1",
        input_data={"prompt": "hi"},
    )
    await server._execute_managed_task(record)  # pyright: ignore[reportPrivateUsage]

    assert record.provenance is None
    assert record.status is TaskStatus.FAILED
    assert "no provenance" in str(record.error)
    assert "provider down" in str(record.error)


async def test_gateway_forwards_a_stated_provenance_unchanged(
    sample_agent_card: AgentCard,
) -> None:
    """The refusal is not a blanket rejection: a stated attribution passes through (#157).

    Guards the other direction — that #157's fix did not turn every task into a failure.
    The value that reaches the wire is the producer's own, not a copy the gateway built.
    """
    stated = Provenance.primary("ollama", "qwen2.5-coder:14b")

    class _AttributedAgent(_UnattributedAgent):
        async def execute_turn(  # noqa: D102
            self, input_data: object, *, continuation: bool = False
        ) -> TurnResult:
            return TurnResult(
                turn_index=1,
                content="a real answer",
                is_completed=True,
                provenance=stated,
            )

    server = A2AServer(
        agent_card=sample_agent_card,
        agent=cast(BaseAgent, _AttributedAgent()),
        port=0,
    )
    record = ManagedTaskRecord(
        task_id="task-attributed", session_id="sess-1", input_data={"prompt": "hi"}
    )
    await server._execute_managed_task(record)  # pyright: ignore[reportPrivateUsage]

    assert record.status is TaskStatus.COMPLETED
    assert record.provenance == stated
    wire = record.to_dict()
    assert wire["provenance"] is not None
    assert wire["provenance"]["served_by"]["provider"] == "ollama"


async def test_gateway_refuses_an_unattributed_handler_result(
    sample_agent_card: AgentCard,
) -> None:
    """The handler path refuses too, matching `send_task_endpoint`'s existing 500 (#157).

    `Provenance.primary("handler")` named a provider that does not exist — there is no
    service called "handler" — while the synchronous endpoint for the same handler
    already answered `MissingProvenanceError`. One file, two opposite policies.
    """

    async def unattributed_handler(msg: TaskMessage) -> TaskResult:
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            output_data={"result": "ok"},
            provenance=None,
        )

    server = A2AServer(agent_card=sample_agent_card, handler=unattributed_handler, port=0)
    record = ManagedTaskRecord(
        task_id="task-handler", session_id="sess-1", input_data={"prompt": "hi"}
    )
    await server._execute_managed_task(record)  # pyright: ignore[reportPrivateUsage]

    assert record.provenance is None
    assert record.status is TaskStatus.FAILED
    assert "no provenance" in str(record.error)
    assert record.to_dict()["provenance"] is None


async def test_unattributed_turn_reaches_a_peer_as_a_failure_over_real_http(
    sample_agent_card: AgentCard,
) -> None:
    """The wire document a remote peer actually receives (#157), over a real socket.

    The sibling tests drive `_execute_managed_task` directly, which is where the policy
    lives; this one goes through the HTTP surface a peer sees — create, poll, read the
    serialised document — because #157 is a trust-boundary defect and the boundary is
    the thing worth asserting.

    This test shape is inherited from #167, which fixed the same issue concurrently with
    the opposite policy (task `completed`, `provenance: null`). Its end-to-end shape was
    better than mine; its policy conflicted with the refusal that `send_task_endpoint`
    and the SSE generator already apply to this exact condition. The shape is kept here,
    the assertion is inverted, and #167's three tests are removed.
    """

    class _UnattributedBaseAgent(BaseAgent):
        async def execute_turn(  # noqa: D102
            self, input_data: object, *, continuation: bool = False, **kwargs: Any
        ) -> TurnResult:
            return TurnResult(
                turn_index=1,
                content="Unattributed result content",
                is_completed=True,
                provenance=None,
            )

    bus = EventBus()
    agent = _UnattributedBaseAgent(
        config=AgentConfig(agent_id="unattributed_agent", name="UnattributedAgent"),
        bus=bus,
        llm=MockLLMConnector(),
    )
    server = A2AServer(agent_card=sample_agent_card, agent=agent, bus=bus, port=0)
    await server.start()
    try:
        async with httpx.AsyncClient() as client:
            create = await client.post(
                f"{server.url}/a2a/v1/tasks",
                headers={"A2A-Version": "1.0"},
                json={
                    "task_id": "wire-unattributed-1",
                    "input_data": {"prompt": "who answered this?"},
                },
            )
            assert create.status_code == 201

            poll: dict[str, Any] = {}
            for _ in range(30):
                resp = await client.get(
                    f"{server.url}/a2a/v1/tasks/wire-unattributed-1",
                    headers={"A2A-Version": "1.0"},
                )
                assert resp.status_code == 200
                poll = cast(dict[str, Any], resp.json())
                if poll.get("status") in ("failed", "completed"):
                    break
                await asyncio.sleep(0.05)

            # The peer is told the task failed and why, and is handed no attribution at
            # all -- rather than a fabricated clean primary it could not have detected.
            assert poll.get("status") == "failed"
            assert poll.get("provenance") is None
            assert "no provenance" in str(poll.get("error"))
            assert "unattributed_agent" in str(poll.get("error"))
            # The unattributed content is not published as a result.
            assert poll.get("output_data") == {}
    finally:
        await server.stop()


async def test_a2a_server_unexpected_exception_logging_and_resilience(
    sample_agent_card: AgentCard,
) -> None:
    """When a task handler raises an unexpected exception (Issue #176, P1, P6):

    1) logger.exception is invoked with the task ID and preserves the full traceback.
    2) server.processing_errors captures the internal error details and traceback.
    3) record.error reflects the internal error and record.provenance remains None.
    4) The server task loop stays alive (P1) and subsequent tasks execute normally.
    """
    from unittest.mock import patch

    async def flaky_handler(msg: TaskMessage) -> TaskResult:
        if msg.task_id == "task-fail-db":
            raise RuntimeError("database crash")
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            output_data={"result": f"processed {msg.task_id}"},
            provenance=Provenance.primary("test_handler"),
        )

    server = A2AServer(
        agent_card=sample_agent_card,
        handler=flaky_handler,
        port=0,
    )
    await server.start()
    try:
        with patch("uclone_x.shells.a2a_server.logger.exception") as mock_log_exception:
            async with httpx.AsyncClient() as client:
                # 1. Dispatch task that raises an unexpected exception
                create_fail = await client.post(
                    f"{server.url}/a2a/v1/tasks",
                    headers={"A2A-Version": "1.0"},
                    json={"task_id": "task-fail-db", "input_data": {"action": "query"}},
                )
                assert create_fail.status_code == 201

                # Poll until terminal
                fail_data: dict[str, Any] = {}
                for _ in range(30):
                    resp = await client.get(
                        f"{server.url}/a2a/v1/tasks/task-fail-db",
                        headers={"A2A-Version": "1.0"},
                    )
                    assert resp.status_code == 200
                    fail_data = cast(dict[str, Any], resp.json())
                    if fail_data.get("status") in ("failed", "completed"):
                        break
                    await asyncio.sleep(0.05)

                # Verify failure status, error formatting, and provenance absence (P6)
                assert fail_data["status"] == "failed"
                assert fail_data["error"] == "InternalError: RuntimeError: database crash"
                assert fail_data["provenance"] is None

                # 1) & 4) logger.exception invoked
                mock_log_exception.assert_called_once_with(
                    "Task %s failed with unexpected exception", "task-fail-db"
                )

                # 2) server.processing_errors captured
                assert len(server.processing_errors) == 1
                recorded_err = server.processing_errors[0]
                assert recorded_err["task_id"] == "task-fail-db"
                assert recorded_err["error_class"] == "RuntimeError"
                assert recorded_err["error_message"] == "database crash"
                assert "database crash" in recorded_err["traceback"]
                assert "RuntimeError" in recorded_err["traceback"]
                assert isinstance(recorded_err["timestamp"], float)

                # 3) Server task loop stays alive (P1) and subsequent task executes normally
                create_ok = await client.post(
                    f"{server.url}/a2a/v1/tasks",
                    headers={"A2A-Version": "1.0"},
                    json={"task_id": "task-ok-next", "input_data": {"action": "query"}},
                )
                assert create_ok.status_code == 201

                ok_data: dict[str, Any] = {}
                for _ in range(30):
                    resp = await client.get(
                        f"{server.url}/a2a/v1/tasks/task-ok-next",
                        headers={"A2A-Version": "1.0"},
                    )
                    assert resp.status_code == 200
                    ok_data = cast(dict[str, Any], resp.json())
                    if ok_data.get("status") in ("failed", "completed"):
                        break
                    await asyncio.sleep(0.05)

                assert ok_data["status"] == "completed"
                assert ok_data["output_data"] == {"result": "processed task-ok-next"}
                assert ok_data["provenance"] is not None
                assert ok_data["provenance"]["served_by"]["provider"] == "test_handler"
    finally:
        await server.stop()


async def test_a2a_server_agent_turn_exception_records_processing_error(
    sample_agent_card: AgentCard,
) -> None:
    """When agent.execute_turn raises an unexpected exception (Issue #176):

    The server records the failure in processing_errors and marks task failed without crashing.
    """

    class _CrashingAgent(BaseAgent):
        async def execute_turn(  # noqa: D102
            self, input_data: object, *, continuation: bool = False, **kwargs: Any
        ) -> TurnResult:
            raise KeyError("unexpected internal dictionary key missing")

    bus = EventBus()
    agent = _CrashingAgent(
        config=AgentConfig(agent_id="crash_agent", name="CrashAgent"),
        bus=bus,
        llm=MockLLMConnector(),
    )
    server = A2AServer(agent_card=sample_agent_card, agent=agent, bus=bus, port=0)
    record = ManagedTaskRecord(
        task_id="task-crash-agent",
        session_id="sess-crash",
        input_data={"prompt": "do something"},
    )
    await server._execute_managed_task(record)  # pyright: ignore[reportPrivateUsage]

    assert record.status is TaskStatus.FAILED
    assert record.error == "InternalError: KeyError: 'unexpected internal dictionary key missing'"
    assert record.provenance is None
    assert len(server.processing_errors) == 1
    err = server.processing_errors[0]
    assert err["task_id"] == "task-crash-agent"
    assert err["error_class"] == "KeyError"
    assert "unexpected internal dictionary key missing" in err["error_message"]
    assert "KeyError" in err["traceback"]
    assert isinstance(err["timestamp"], float)


# ======================================================================================
# Issue #665 (follow-up) — the shallow unwrap in the managed-task path
# ======================================================================================


async def test_a2a_managed_task_returns_nested_output_data(
    sample_agent_card: AgentCard,
) -> None:
    """A handler's nested `output_data` survives the managed-task record and the wire.

    `TaskResult.output_data` is `ImmutableJsonMapping` and therefore frozen recursively.
    `ManagedTaskRecord` is a plain object, not a pydantic model, so nothing revalidates
    what is assigned to it: the shallow `dict(result.output_data)` parked nested
    `MappingProxyType`s on the record, and `GET /a2a/v1/tasks/{id}` then handed them to
    `JSONResponse`, whose `json.dumps` raised `TypeError` — a 500 on task status for any
    handler that returns structured output.

    This site is not in `llm/connectors/` and was not among the three the review of #673
    listed; it was found by sweeping every `Immutable*Mapping` field to its consumers
    rather than sweeping the connectors.

    Killed by: src/uclone_x/shells/a2a_server.py :: record.output_data = cast(dict[str, Any], unwrap_immutable(result.output_data))
    """
    nested_output: dict[str, Any] = {
        "summary": "scan complete",
        "findings": {"critical": 0, "detail": {"cve": "none"}},
        "hosts": [{"name": "h1", "open_ports": [22, 443]}],
    }

    async def handler(message: TaskMessage) -> TaskResult:
        return TaskResult(
            task_id=message.task_id,
            status=TaskStatus.COMPLETED,
            output_data=nested_output,
            provenance=Provenance.primary("test-handler"),
        )

    server = A2AServer(agent_card=sample_agent_card, handler=handler, port=0)
    await server.start()
    try:
        async with httpx.AsyncClient() as client:
            create_resp = await client.post(
                f"{server.url}/a2a/v1/tasks",
                headers={"A2A-Version": "1.0"},
                json={"task_id": "task-nested-1", "input_data": {"prompt": "scan"}},
            )
            assert create_resp.status_code == 201

            poll_data: dict[str, Any] = {}
            for _ in range(30):
                poll_resp = await client.get(
                    f"{server.url}/a2a/v1/tasks/task-nested-1",
                    headers={"A2A-Version": "1.0"},
                )
                # The assertion that matters: the status endpoint can serialize itself.
                assert poll_resp.status_code == 200, poll_resp.text
                poll_data = cast(dict[str, Any], poll_resp.json())
                if poll_data.get("status") in ("completed", "failed"):
                    break
                await asyncio.sleep(0.05)

            assert poll_data.get("status") == "completed"
            assert poll_data.get("output_data") == nested_output
            artifacts = cast(list[dict[str, Any]], poll_data.get("artifacts", []))
            assert artifacts and artifacts[0]["content"] == nested_output
    finally:
        await server.stop()
