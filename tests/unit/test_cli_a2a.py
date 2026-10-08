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

from tests.support.clones import make_clones
from uclone_x.a2a.models import AgentCard, TaskMessage, TaskResult, TaskStatus
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, TurnResult
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.cli import main
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import AgentEvent, EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.room.store import RoomStore
from uclone_x.shells.a2a_server import (
    TURN_FAILED_MESSAGE,
    TURN_UNATTRIBUTED_MESSAGE,
    A2AServer,
    ManagedTaskRecord,
)

# The agent ids these tests run as. Each is a clone now, since a name no clone carries is
# refused rather than given a home (clone-data-scopes §3.4); persona-less, so each speaks as
# the prompt the test gives it.
_A2A_CLONES = ("dev_agent",)


@pytest.fixture(autouse=True)
def _a2a_clones() -> None:  # pyright: ignore[reportUnusedFunction]
    make_clones(*_A2A_CLONES)


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


async def test_a2a_a_task_that_finishes_while_its_cancel_waits_keeps_its_own_end(
    sample_agent_card: AgentCard,
) -> None:
    """A task that completes during the cancel's wait stays completed, with no `canceled`.

    The cancel waits for the task to handle its cancellation (#1921). A task that finishes
    instead is already terminal when the wait ends; the cancel answers with that status and
    does not re-mark it or tell listeners it was cancelled (#1934).

    Killed by: src/uclone_x/shells/a2a_server.py :: if record.is_terminal:  # it ended on its own while the cancel waited
    Becomes: if False:  # it ended on its own while the cancel waited
    """
    started = asyncio.Event()

    async def finishes_anyway(msg: TaskMessage) -> TaskResult:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass  # a handler that finishes its work rather than stop
        return TaskResult(
            task_id=msg.task_id,
            status=TaskStatus.COMPLETED,
            provenance=Provenance.primary("agent"),
        )

    server = A2AServer(agent_card=sample_agent_card, handler=finishes_anyway, port=0)
    record = ManagedTaskRecord(task_id="task-finishes", session_id="sess", input_data={})
    server._tasks[record.task_id] = record  # pyright: ignore[reportPrivateUsage]
    record.async_task = asyncio.create_task(server._execute_managed_task(record))  # pyright: ignore[reportPrivateUsage]
    await started.wait()

    answer = await server.cancel_task_endpoint(record.task_id, a2a_version="1.0")

    assert json.loads(bytes(answer.body)) == {"task_id": record.task_id, "status": "completed"}
    assert record.status is TaskStatus.COMPLETED
    assert "canceled" not in [e["event"] for e in record.events]


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
            assert data["error"] == TURN_FAILED_MESSAGE  # the cause is logged (#1885)
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
            assert poll_data.get("error") == TURN_UNATTRIBUTED_MESSAGE
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
        self, input_data: object, *, continuation: bool = False, **kwargs: Any
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
    # The peer sees an explicit absence and a plain reason — not a clean primary. Which
    # agent and turn it was goes to the server's log (#1885), not over the wire.
    assert wire["provenance"] is None
    assert wire["status"] == TaskStatus.FAILED.value
    assert wire["error"] == TURN_UNATTRIBUTED_MESSAGE
    # And the unattributed content is not published as a completed result.
    assert wire["output_data"] == {}
    assert wire["artifacts"] == []


async def test_gateway_refusal_preserves_the_turns_own_error(
    sample_agent_card: AgentCard, caplog: pytest.LogCaptureFixture
) -> None:
    """A turn that both failed and stated no provenance keeps both facts (#157).

    The refusal must not overwrite the diagnosis: the turn's own error is in the server's
    log beside the refusal, and the peer is told plainly (#1885).

    Killed by: src/uclone_x/shells/a2a_server.py :: turn_result.error or "no error",
    Becomes: "no error",
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
    assert record.error == TURN_UNATTRIBUTED_MESSAGE
    logged = caplog.text
    assert "no provenance" in logged
    assert "provider down" in logged


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
            self, input_data: object, *, continuation: bool = False, **kwargs: Any
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
    assert record.error == TURN_UNATTRIBUTED_MESSAGE
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
            assert poll.get("error") == TURN_UNATTRIBUTED_MESSAGE
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
                assert fail_data["error"] == TURN_FAILED_MESSAGE
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
    assert record.error == TURN_FAILED_MESSAGE
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


# ======================================================================================
# Each A2A contextId is a one-seat room (#1836)
# ======================================================================================


def _a2a_room_server(
    tmp_path: Path, sample_agent_card: AgentCard, clone_id: str = "ada"
) -> tuple[A2AServer, SessionStore, RoomStore]:
    """An A2A server serving `clone_id`, built per context the way `ucx a2a serve` builds it."""
    from uclone_x.agent.clone_builder import AppScope
    from uclone_x.agent.composition import HostDependencies
    from uclone_x.agent.models import AgentLLMConfig
    from uclone_x.agent.persona_registry import PersonaRegistry
    from uclone_x.cli.commands.a2a import a2a_turn_recorder, context_agent_factory
    from uclone_x.telemetry.tracer import TelemetryTracer
    from uclone_x.tools.registry import ToolRegistry

    store = SessionStore(tmp_path / "sessions")
    rooms = RoomStore(tmp_path / "rooms")
    app = AppScope(
        host=HostDependencies(
            bus=EventBus(),
            llm=MockLLMConnector(),
            tools=ToolRegistry(),
            tracer=TelemetryTracer(),
            store=store,
        ),
        workspace_root=tmp_path,
        persona_registry=PersonaRegistry(include_defaults=False),
    )
    server = A2AServer(
        agent_card=sample_agent_card,
        port=0,
        context_agent_factory=context_agent_factory(
            app,
            clone_id=clone_id,
            room_store=rooms,
            fallback_llm=AgentLLMConfig(model_name="mock"),
            config_update={"name": clone_id, "max_steps": 3},
        ),
        turn_recorder=a2a_turn_recorder(clone_id, rooms),
    )
    return server, store, rooms


async def _ask(server: A2AServer, context_id: str, prompt: str) -> ManagedTaskRecord:
    record = ManagedTaskRecord(
        task_id=f"task-{context_id}-{prompt}",
        session_id="sess-caller",
        input_data={"prompt": prompt},
        context_id=context_id,
    )
    await server._execute_managed_task(record)  # pyright: ignore[reportPrivateUsage]
    return record


def _asked(store: SessionStore, clone_id: str, context_id: str) -> list[str]:
    from uclone_x.cli.commands.a2a import a2a_room_id
    from uclone_x.llm.models import MessageRole
    from uclone_x.room.service import participant_session_id

    state = store.load(participant_session_id(a2a_room_id(clone_id, context_id), clone_id))
    assert state is not None, f"no seat session for {context_id!r}"
    return [m.content or "" for m in state.messages if m.role == MessageRole.USER]


async def test_a2a_context_continues_its_room_and_a_new_context_starts_one(
    tmp_path: Path, sample_agent_card: AgentCard
) -> None:
    """A known `contextId` continues its room's seat session; a new one starts a room.

    The `sess_<clone>` session the gateway used before is neither written nor resumed.

    Killed by: src/uclone_x/shells/a2a_server.py :: record.context_id, prompt, on_set_aside=record
    Becomes: record.session_id, prompt, on_set_aside=record
    Killed by: src/uclone_x/cli/commands/a2a.py :: agent.hydrate_session()
    Becomes: pass
    """
    from uclone_x.cli.commands.a2a import a2a_room_id

    old = SessionState(session_id="sess_ada", agent_id="ada")
    server, store, rooms = _a2a_room_server(tmp_path, sample_agent_card)
    store.save(old)
    before = store.session_path("sess_ada").read_bytes()

    first = await _ask(server, "ctx-1", "one")
    # A fresh server: continuing must come from the stored seat, not a held agent.
    server, _, _ = _a2a_room_server(tmp_path, sample_agent_card)
    second = await _ask(server, "ctx-1", "two")
    other = await _ask(server, "ctx-2", "three")

    assert [r.status for r in (first, second, other)] == [TaskStatus.COMPLETED] * 3
    assert _asked(store, "ada", "ctx-1") == ["one", "two"]
    assert _asked(store, "ada", "ctx-2") == ["three"]
    assert set(rooms.list_room_ids()) == {a2a_room_id("ada", "ctx-1"), a2a_room_id("ada", "ctx-2")}
    state = rooms.load(a2a_room_id("ada", "ctx-1"))
    assert state is not None
    assert state.head == "a2a"  # the app will not post into it (#1885)
    humans = [
        m.content for m in state.transcript if m.kind == "utterance" and m.sender_id == "user"
    ]
    assert humans == ["one", "two"]
    assert store.session_path("sess_ada").read_bytes() == before


async def test_a2a_two_clones_under_one_context_never_share_a_room(
    tmp_path: Path, sample_agent_card: AgentCard
) -> None:
    """Two clones called with one `contextId` each keep a room of their own (#1836).

    Killed by: src/uclone_x/cli/commands/a2a.py :: return conversation_room_id("a2a", clone_id, context_id)
    Becomes: return conversation_room_id("a2a", "shared", context_id)
    """
    ada, store, rooms = _a2a_room_server(tmp_path, sample_agent_card, "ada")
    bo, _, _ = _a2a_room_server(tmp_path, sample_agent_card, "bo")

    asked_ada = await _ask(ada, "ctx", "for ada")
    asked_bo = await _ask(bo, "ctx", "for bo")

    assert asked_ada.status is TaskStatus.COMPLETED, asked_ada.error
    assert asked_bo.status is TaskStatus.COMPLETED, asked_bo.error
    assert len(rooms.list_room_ids()) == 2
    assert _asked(store, "ada", "ctx") == ["for ada"]
    assert _asked(store, "bo", "ctx") == ["for bo"]


async def test_a2a_says_once_that_a_turns_save_kept_a_record_aside(
    tmp_path: Path, sample_agent_card: AgentCard
) -> None:
    """A task whose save kept a newer build's record aside says so, once (#1921).

    Before, A2A set the record aside, recorded its row without the flag and said nothing,
    while `ucx run`, `ucx loop` and ACP each told the person. The task carries the notice
    as its `status_message` and one `notice` event, in the words every head uses; its
    result stays in the artifacts. The room's row carries the flag the app renders.

    Killed by: src/uclone_x/shells/a2a_server.py :: set_aside = self._took_set_aside(context_id, agent)  # this turn's save
    Becomes: set_aside = False
    Killed by: src/uclone_x/shells/a2a_server.py :: on_set_aside.status_message = SESSION_SET_ASIDE_NOTICE
    Becomes: pass
    Killed by: src/uclone_x/shells/a2a_server.py :: await record.emit_event({"event": "notice", "message": record.status_message})
    Becomes: pass
    Killed by: src/uclone_x/cli/commands/a2a.py :: turn = replace(turn, session_set_aside=session_set_aside)
    Becomes: pass
    """
    from uclone_x.cli.commands.a2a import a2a_room_id
    from uclone_x.core.set_aside import SESSION_SET_ASIDE_NOTICE, set_aside_copies
    from uclone_x.room.service import participant_session_id

    server, store, rooms = _a2a_room_server(tmp_path, sample_agent_card)
    first = await _ask(server, "ctx", "one")
    seat = store.session_path(participant_session_id(a2a_room_id("ada", "ctx"), "ada"))
    document = json.loads(seat.read_text(encoding="utf-8"))
    document["a_field_from_a_newer_build"] = True
    seat.write_text(json.dumps(document), encoding="utf-8")

    second = await _ask(server, "ctx", "two")
    third = await _ask(server, "ctx", "three")

    assert [r.status for r in (first, second, third)] == [TaskStatus.COMPLETED] * 3
    assert len(set_aside_copies(seat)) == 1
    assert [r.to_dict()["status_message"] for r in (first, second, third)] == [
        None,
        SESSION_SET_ASIDE_NOTICE,
        None,
    ]
    notices = [
        [e["message"] for e in r.events if e["event"] == "notice"] for r in (first, second, third)
    ]
    assert notices == [[], [SESSION_SET_ASIDE_NOTICE], []]
    assert second.artifacts == [
        {"name": "response", "content": second.output_data["result"], "type": "text"}
    ]
    room = rooms.load(a2a_room_id("ada", "ctx"))
    assert room is not None
    rows = [
        m.session_set_aside
        for m in room.transcript
        if m.sender_id == "ada" and m.kind == "utterance"
    ]
    assert rows == [False, True, False]


async def _a_context_whose_next_save_sets_its_record_aside(
    tmp_path: Path, sample_agent_card: AgentCard
) -> tuple[A2AServer, RoomStore, Path]:
    """A server whose `ctx` agent is held, with its seat record since written by a newer build."""
    from uclone_x.cli.commands.a2a import a2a_room_id
    from uclone_x.room.service import participant_session_id

    server, store, rooms = _a2a_room_server(tmp_path, sample_agent_card)
    first = await _ask(server, "ctx", "one")
    assert first.status is TaskStatus.COMPLETED, first.error
    seat = store.session_path(participant_session_id(a2a_room_id("ada", "ctx"), "ada"))
    document = json.loads(seat.read_text(encoding="utf-8"))
    document["a_field_from_a_newer_build"] = True
    seat.write_text(json.dumps(document), encoding="utf-8")
    return server, rooms, seat


def _last_row_flag(rooms: RoomStore) -> bool:
    from uclone_x.cli.commands.a2a import a2a_room_id

    room = rooms.load(a2a_room_id("ada", "ctx"))
    assert room is not None
    return room.transcript[-1].session_set_aside


async def test_a2a_a_failed_turn_still_says_its_save_kept_a_record_aside(
    tmp_path: Path, sample_agent_card: AgentCard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn that raised is saved and recorded, and the task says so like a completed one.

    Killed by: src/uclone_x/shells/a2a_server.py :: await self._tell_set_aside(record)  # the failed turn's save
    Becomes: pass
    """
    from uclone_x.core.set_aside import SESSION_SET_ASIDE_NOTICE, set_aside_copies

    server, rooms, seat = await _a_context_whose_next_save_sets_its_record_aside(
        tmp_path, sample_agent_card
    )

    async def _raises(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("the turn broke")

    monkeypatch.setattr(server._context_agents["ctx"], "execute_turn", _raises)  # pyright: ignore[reportPrivateUsage]
    failed = await _ask(server, "ctx", "two")

    assert failed.status is TaskStatus.FAILED
    assert len(set_aside_copies(seat)) == 1
    assert failed.status_message == SESSION_SET_ASIDE_NOTICE
    assert [e["event"] for e in failed.events][-2:] == ["notice", "error"]
    assert _last_row_flag(rooms) is True


async def test_a2a_a_cancelled_turn_says_it_to_a_live_listener(
    tmp_path: Path, sample_agent_card: AgentCard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled turn is saved; a subscriber still listening is told a record was kept aside.

    Before, the cancel closed the listeners before the task had handled its cancellation,
    so the `notice` reached only the stored events (#1921).

    Killed by: src/uclone_x/shells/a2a_server.py :: await self._tell_set_aside(record)  # the cancelled turn's save
    Becomes: pass
    Killed by: src/uclone_x/shells/a2a_server.py :: await asyncio.wait({task}, timeout=CANCEL_SETTLE_SECONDS)
    Becomes: pass
    """
    from uclone_x.core.set_aside import SESSION_SET_ASIDE_NOTICE, set_aside_copies

    server, rooms, seat = await _a_context_whose_next_save_sets_its_record_aside(
        tmp_path, sample_agent_card
    )
    started = asyncio.Event()

    async def _hangs(*_args: object, **_kwargs: object) -> object:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(server._context_agents["ctx"], "execute_turn", _hangs)  # pyright: ignore[reportPrivateUsage]
    record = ManagedTaskRecord(
        task_id="task-cancelled",
        session_id="sess-caller",
        input_data={"prompt": "two"},
        context_id="ctx",
    )
    heard: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    record.listeners.append(heard)
    server._tasks[record.task_id] = record  # pyright: ignore[reportPrivateUsage]
    record.async_task = asyncio.create_task(server._execute_managed_task(record))  # pyright: ignore[reportPrivateUsage]
    await started.wait()

    answer = await server.cancel_task_endpoint(record.task_id, a2a_version="1.0")

    assert json.loads(bytes(answer.body)) == {"task_id": record.task_id, "status": "canceled"}
    live: list[str] = []
    while (event := heard.get_nowait()) is not None:
        live.append(event["event"])
    assert live[-2:] == ["notice", "canceled"]
    assert record.status_message == SESSION_SET_ASIDE_NOTICE
    assert len(set_aside_copies(seat)) == 1
    assert _last_row_flag(rooms) is True


def test_a2a_task_request_reads_context_id_and_reports_it() -> None:
    """`contextId` on the wire names the conversation; the task reports which one it joined.

    Killed by: src/uclone_x/shells/a2a_server.py :: "context_id": self.context_id,
    Becomes: "context_id": self.session_id,
    """
    from uclone_x.shells.a2a_server import TaskCreateRequest

    payload = TaskCreateRequest.model_validate({"contextId": "ctx-9", "session_id": "sess-9"})
    record = ManagedTaskRecord(
        task_id="t", session_id="sess-9", input_data={}, context_id=payload.context_id
    )
    unkeyed = ManagedTaskRecord(task_id="u", session_id="sess-8", input_data={})

    assert payload.context_id == "ctx-9"
    assert record.to_dict()["context_id"] == "ctx-9"
    assert unkeyed.to_dict()["context_id"] == "sess-8"


class _PooledAgent:
    """A context's agent that can be held mid-turn, counting the turns it runs at once."""

    def __init__(self, context_id: str, pool: _Pool) -> None:
        self.agent_id = f"agent-{context_id}"
        self._pool = pool
        self._context_id = context_id

    async def execute_turn(  # noqa: D102
        self, input_data: object, *, continuation: bool = False, **kwargs: Any
    ) -> TurnResult:
        pool = self._pool
        pool.running += 1
        pool.most_at_once = max(pool.most_at_once, pool.running)
        pool.started.set()
        if self._context_id in pool.held_mid_turn:
            await pool.release.wait()
        pool.running -= 1
        return TurnResult(
            turn_index=1,
            content=f"answered {input_data}",
            is_completed=True,
            provenance=Provenance.primary(self.agent_id),
        )

    def persist_session(self) -> None:  # noqa: D102
        return None


class _Pool:
    def __init__(self, *held_mid_turn: str) -> None:
        self.held_mid_turn = set(held_mid_turn)
        self.running = 0
        self.most_at_once = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def factory(self, context_id: str) -> BaseAgent:
        return cast(BaseAgent, _PooledAgent(context_id, self))


def _pooled_server(sample_agent_card: AgentCard, pool: _Pool, cap: int) -> A2AServer:
    return A2AServer(
        agent_card=sample_agent_card,
        port=0,
        context_agent_factory=pool.factory,
        max_context_agents=cap,
    )


async def test_a2a_past_the_cap_the_idle_saved_context_is_released(
    sample_agent_card: AgentCard,
) -> None:
    """At the cap, a new context releases the least recently used idle one (#1836).

    Killed by: src/uclone_x/shells/a2a_server.py :: if len(self._context_agents) >= self._max_context_agents:
    Becomes: if False:
    """
    server = _pooled_server(sample_agent_card, _Pool(), cap=1)

    first = await _ask(server, "c1", "hello")
    second = await _ask(server, "c2", "hello")

    assert (first.status, second.status) == (TaskStatus.COMPLETED, TaskStatus.COMPLETED)
    assert server.context_agent("c1") is None
    assert server.context_agent("c2") is not None


async def test_a2a_a_context_mid_turn_is_not_released_and_the_newcomer_is_told_plainly(
    sample_agent_card: AgentCard,
) -> None:
    """A context whose turn is running is never released; the new one is refused (#1836).

    Killed by: src/uclone_x/shells/a2a_server.py :: if held not in self._unsaved_contexts and not self._context_locks[held].locked()
    Becomes: if held not in self._unsaved_contexts
    """
    from uclone_x.shells.a2a_server import CONTEXTS_BUSY_MESSAGE

    pool = _Pool("c1")
    server = _pooled_server(sample_agent_card, pool, cap=1)
    running = asyncio.create_task(_ask(server, "c1", "slow"))
    await pool.started.wait()

    refused = await _ask(server, "c2", "hello")
    held = server.context_agent("c1")
    pool.release.set()
    finished = await running

    assert (refused.status, refused.error) == (TaskStatus.FAILED, CONTEXTS_BUSY_MESSAGE)
    assert held is not None
    assert finished.status == TaskStatus.COMPLETED


async def test_a2a_one_context_runs_one_turn_at_a_time(sample_agent_card: AgentCard) -> None:
    """Two tasks in one context take turns rather than running on its agent at once (#1836).

    Killed by: src/uclone_x/shells/a2a_server.py :: async with lock:
    Becomes: if True:
    """
    pool = _Pool("c1")
    server = _pooled_server(sample_agent_card, pool, cap=2)
    first = asyncio.create_task(_ask(server, "c1", "one"))
    second = asyncio.create_task(_ask(server, "c1", "two"))
    await pool.started.wait()
    for _ in range(5):
        await asyncio.sleep(0)
    pool.release.set()
    done = await asyncio.gather(first, second)

    assert [task.status for task in done] == [TaskStatus.COMPLETED, TaskStatus.COMPLETED]
    assert pool.most_at_once == 1


async def test_a2a_a_released_context_leaves_no_lock_behind(sample_agent_card: AgentCard) -> None:
    """The lock map holds the held contexts and no more, however many callers came (#1885).

    It kept a lock for every context the server had ever answered, so a long-lived server
    grew by one per caller conversation. A refused context leaves none either.

    Killed by: src/uclone_x/shells/a2a_server.py :: del self._context_locks[victim]
    Becomes: pass
    Killed by: src/uclone_x/shells/a2a_server.py :: del self._context_locks[context_id]
    Becomes: pass
    """
    from uclone_x.shells.a2a_server import CONTEXTS_BUSY_MESSAGE

    pool = _Pool("busy")
    server = _pooled_server(sample_agent_card, pool, cap=1)
    for context_id in ("c1", "c2", "c3", "c4"):
        answered = await _ask(server, context_id, "hello")
        assert answered.status == TaskStatus.COMPLETED

    locks = server._context_locks  # pyright: ignore[reportPrivateUsage]
    users = server._context_users  # pyright: ignore[reportPrivateUsage]
    assert set(locks) == {"c4"} and users == {}

    pool.started.clear()  # the four turns above set it
    running = asyncio.create_task(_ask(server, "busy", "slow"))
    await pool.started.wait()
    refused = await _ask(server, "late", "hello")
    pool.release.set()
    await running

    assert refused.error == CONTEXTS_BUSY_MESSAGE
    assert set(locks) == {"busy"} and users == {}


# ======================================================================================
# #1893 item 1 -- a context's turn knows who the person is; #1885 -- failures read plainly
# ======================================================================================


class _NamedAgent:
    """A context's agent that keeps the names each turn was given."""

    def __init__(self, context_id: str, seen: dict[str, list[tuple[str, ...]]]) -> None:
        self.agent_id = f"agent-{context_id}"
        self._seen = seen.setdefault(context_id, [])

    async def execute_turn(  # noqa: D102
        self, input_data: object, *, person_names: tuple[str, ...] = (), **kwargs: Any
    ) -> TurnResult:
        self._seen.append(person_names)
        return TurnResult(
            turn_index=1,
            content="noted",
            is_completed=True,
            provenance=Provenance.primary(self.agent_id),
        )

    def persist_session(self) -> None:  # noqa: D102
        return None


async def test_a2a_context_turn_is_given_the_person_s_names(sample_agent_card: AgentCard) -> None:
    """Read per turn with the context id; a reader that fails gives none, not a failed task.

    Killed by: src/uclone_x/shells/a2a_server.py :: prompt, person_names=self._turn_person_names(context_id)
    Becomes: prompt
    Killed by: src/uclone_x/shells/a2a_server.py :: return self._person_names(context_id)
    Becomes: return (context_id,)
    """
    seen: dict[str, list[tuple[str, ...]]] = {}

    def names(context_id: str) -> tuple[str, ...]:
        if context_id == "broken":
            raise OSError("room record unreadable")
        return ("user", f"Kenny of {context_id}")

    server = A2AServer(
        agent_card=sample_agent_card,
        port=0,
        context_agent_factory=lambda c: cast(BaseAgent, _NamedAgent(c, seen)),
        person_names=names,
    )

    done = [await _ask(server, c, "remember me") for c in ("c1", "broken")]

    assert [r.status for r in done] == [TaskStatus.COMPLETED] * 2
    assert seen == {"c1": [("user", "Kenny of c1")], "broken": [()]}


#: What a caller must never be shown: a class name, a traceback, a path, the raw cause.
_A2A_INTERNALS = ("Error", "Traceback", "/", "\\", "KeyError", "secret-db", "InternalError")


def _plain(text: object) -> None:
    assert isinstance(text, str) and text, text
    assert not [w for w in _A2A_INTERNALS if w in text], text


class _FailingAgent:
    def __init__(self, how: str) -> None:
        self.agent_id = "agent-failing"
        self._how = how

    async def execute_turn(  # noqa: D102
        self, input_data: object, **kwargs: Any
    ) -> TurnResult:
        if self._how == "raises":
            raise KeyError("/Users/someone/secret-db/index.json")
        return TurnResult(
            turn_index=1,
            content="",
            is_completed=False,
            error="LLMProviderError: 401 at /Users/someone/secret-db",
            provenance=None if self._how == "unattributed" else Provenance.primary("x"),
        )

    def persist_session(self) -> None:  # noqa: D102
        return None


@pytest.mark.parametrize(
    ("how", "copy"),
    [
        ("raises", TURN_FAILED_MESSAGE),
        ("fails", TURN_FAILED_MESSAGE),
        ("unattributed", TURN_UNATTRIBUTED_MESSAGE),
    ],
)
async def test_a2a_failed_task_tells_the_caller_plainly_and_logs_the_cause(
    sample_agent_card: AgentCard, caplog: pytest.LogCaptureFixture, how: str, copy: str
) -> None:
    """The caller's owner reads the task's error; the cause is for the server's log (#1885).

    Killed by: src/uclone_x/shells/a2a_server.py :: record.error = TURN_FAILED_MESSAGE
    Becomes: record.error = turn_result.error or TURN_FAILED_MESSAGE
    Killed by: src/uclone_x/shells/a2a_server.py :: record.error = TURN_UNATTRIBUTED_MESSAGE
    Becomes: record.error = f"{TURN_UNATTRIBUTED_MESSAGE} {turn_result.error}"
    Killed by: src/uclone_x/shells/a2a_server.py :: record.error, record.provenance = TURN_FAILED_MESSAGE, None
    Becomes: record.error, record.provenance = str(exc), None
    """
    server = A2AServer(
        agent_card=sample_agent_card,
        port=0,
        context_agent_factory=lambda c: cast(BaseAgent, _FailingAgent(how)),
    )

    record = await _ask(server, "c1", "go")

    assert record.status is TaskStatus.FAILED
    assert record.error == copy
    _plain(record.to_dict()["error"])
    assert "secret-db" in caplog.text


def test_a2a_failure_copy_is_plain() -> None:
    """Both sentences, whatever the failure: no class names, paths or tracebacks."""
    _plain(TURN_FAILED_MESSAGE)
    _plain(TURN_UNATTRIBUTED_MESSAGE)


async def test_a2a_handler_crash_over_send_and_stream_reads_plainly(
    sample_agent_card: AgentCard, caplog: pytest.LogCaptureFixture
) -> None:
    r"""`/send` and `/stream` answer a crashed handler with the plain sentence, not `str(exc)`.

    Killed by: src/uclone_x/shells/a2a_server.py :: detail={"error": "InvalidAgentResponseError", "message": TURN_FAILED_MESSAGE},
    Becomes: detail={"error": "InvalidAgentResponseError", "message": str(e)},
    Killed by: src/uclone_x/shells/a2a_server.py :: yield f"data: {json.dumps({'error': TURN_FAILED_MESSAGE})}\n\n"
    Becomes: yield f"data: {json.dumps({'error': traceback.format_exc()})}\n\n"
    """

    async def crashing(msg: TaskMessage) -> TaskResult:
        raise ValueError("/Users/someone/secret-db is locked")

    server = A2AServer(agent_card=sample_agent_card, handler=crashing, port=0)
    await server.start()
    try:
        async with httpx.AsyncClient() as client:
            body: dict[str, object] = {
                "task_id": "t-crash",
                "session_id": "s-crash",
                "input_data": {},
                "sender_agent_id": "caller",
                "target_agent_id": "target",
            }
            headers = {"A2A-Version": "1.0"}
            sent = await client.post(f"{server.url}/tasks/send", headers=headers, json=body)
            streamed = await client.post(f"{server.url}/tasks/stream", headers=headers, json=body)
    finally:
        await server.stop()

    assert sent.status_code == 500
    assert sent.json()["detail"]["message"] == TURN_FAILED_MESSAGE
    event = json.loads(streamed.text.strip().removeprefix("data: "))
    assert event == {"error": TURN_FAILED_MESSAGE}
    assert "secret-db" in caplog.text
