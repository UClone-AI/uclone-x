"""Unit tests for the ACP (Agent Client Protocol) server shell adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field

import uclone_x
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentConfig, AgentLLMConfig, TurnResult
from uclone_x.agent.persona_store import DEFAULT_PERSONA_NAME
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.cli.commands.acp import session_agent_factory
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.log.reader import read_session_log
from uclone_x.shells.acp.models import (
    ACP_PROTOCOL_VERSION,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    SERVER_NAME,
    SESSION_NOT_FOUND,
    UNSUPPORTED_MCP_TRANSPORT,
)
from uclone_x.shells.acp.server import (
    BLOCKED_MESSAGE,
    CANNOT_OPEN_MESSAGE,
    CANNOT_START_MESSAGE,
    NOT_SAVED_MESSAGE,
    REQUEST_FAILED_MESSAGE,
    TOO_MANY_BUSY_MESSAGE,
    TOO_MANY_STEPS_MESSAGE,
    TURN_FAILED_MESSAGE,
    TURN_IN_FLIGHT_MESSAGE,
    UNKNOWN_SESSION_MESSAGE,
    USAGE_LIMIT_MESSAGE,
    ACPServer,
)
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

# --------------------------------------------------------------------------------------
# A stand-in agent, for tests about the server's own bookkeeping.
# --------------------------------------------------------------------------------------


class _FakeSessionAgent:
    """Serves one session: records every save, answers with `execute`."""

    def __init__(
        self,
        session_id: str,
        execute: Callable[[str], Awaitable[TurnResult]],
        *,
        fail_saves_after: int | None = None,
    ) -> None:
        self.session_id = session_id
        self._execute = execute
        self._fail_saves_after = fail_saves_after
        self.saves: list[str | None] = []
        self.hydrated: list[str | None] = []

    def persist_session(self, session_id: str | None = None) -> None:
        if self._fail_saves_after is not None and len(self.saves) >= self._fail_saves_after:
            raise OSError("disk full")
        self.saves.append(session_id)

    def hydrate_session(self, session_id: str | None = None) -> SessionState:
        self.hydrated.append(session_id)
        return SessionState(session_id=self.session_id, agent_id="fake")

    async def execute_turn(self, prompt: str) -> TurnResult:
        return await self._execute(prompt)


async def _answer(prompt: str) -> TurnResult:
    return TurnResult(turn_index=1, content=f"re: {prompt}", provenance=Provenance.primary("fake"))


def _fake_factory(
    agents: list[_FakeSessionAgent],
    execute: Callable[[str], Awaitable[TurnResult]] = _answer,
    *,
    fail_saves_after: int | None = None,
) -> Callable[[str], BaseAgent]:
    """A factory that builds a fresh fake per call and keeps each one in `agents`."""

    def build(session_id: str) -> BaseAgent:
        agent = _FakeSessionAgent(session_id, execute, fail_saves_after=fail_saves_after)
        agents.append(agent)
        return cast(BaseAgent, agent)

    return build


def _capture(server: ACPServer) -> list[dict[str, Any]]:
    """Replace the server's writer with a list, and return the list."""
    sent: list[dict[str, Any]] = []

    async def send(msg: dict[str, Any]) -> None:
        sent.append(msg)

    server.send_response = send  # type: ignore[method-assign]
    return sent


async def _prompt(server: ACPServer, session_id: str, text: str, req_id: int) -> None:
    """Send one prompt and wait for its turn to finish."""
    resp = await server.dispatch_method(
        "prompt", {"sessionId": session_id, "prompt": text}, req_id=req_id
    )
    assert resp is None, resp
    task = server.get_in_flight_task(session_id)
    assert task is not None
    await task


# --------------------------------------------------------------------------------------
# Real agents, built the way the CLI builds them, over a real store.
# --------------------------------------------------------------------------------------

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="scripted", model="scripted"),
    served_by=ServiceRef(provider="scripted", model="scripted"),
    attempts=(),
)
_USAGE = TokenUsage(provider="scripted", model="scripted", input_tokens=0, output_tokens=0)


class _RecordingLLM(BaseLLMConnector):
    """Keeps every request. Calls `echo` once when the user asks for it, else answers."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.calls.append(request)
        last = request.messages[-1]
        if last.role == MessageRole.USER and "use echo" in (last.content or ""):
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                content=None,
                tool_calls=(ToolCallRequest(id="tc_echo", name="echo", arguments={"x": 7}),),
                usage=_USAGE,
                provenance=_PROV,
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content=f"answer {len(self.calls)}",
            tool_calls=(),
            usage=_USAGE,
            provenance=_PROV,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta_content="unused")


class _EchoParams(BaseModel):
    x: int = Field(default=0)


class _EchoTool(BaseTool[_EchoParams]):
    name = "echo"
    description = "Echo tool"

    def run(self, params: _EchoParams, context: ToolContext) -> dict[str, Any]:
        return {"echoed": params.x}


def _real_server(store: SessionStore, llm: _RecordingLLM, **kwargs: Any) -> ACPServer:
    """An `ACPServer` whose sessions get real agents from the CLI's own factory."""
    tools = ToolRegistry()
    tools.register(_EchoTool())
    bus = EventBus()
    host = HostDependencies(bus=bus, llm=llm, tools=tools, tracer=TelemetryTracer(), store=store)
    config = AgentConfig(
        agent_id="acp_agent",
        name="ACP Agent",
        llm_config=AgentLLMConfig(model_name="scripted"),
        max_steps=5,
    )
    return ACPServer(
        agent_factory=session_agent_factory(config, host), bus=bus, store=store, **kwargs
    )


def _request_text(request: LLMRequest) -> str:
    return "\n".join(m.content or "" for m in request.messages)


@pytest.mark.asyncio
async def test_acp_initialize_negotiation() -> None:
    """Test ACP initialize handler returns correct capabilities and server metadata.

    Killed by: src/uclone_x/shells/acp/server.py :: "protocolVersion": ACP_PROTOCOL_VERSION,
    Becomes: "protocolVersion": "999.0.0",
    """
    server = ACPServer()
    resp = await server.dispatch_method("initialize", {}, req_id=1)
    assert resp is not None
    assert resp["id"] == 1
    assert "result" in resp
    res = resp["result"]
    assert res["protocolVersion"] == ACP_PROTOCOL_VERSION
    assert res["serverInfo"]["name"] == SERVER_NAME
    assert res["serverInfo"]["version"] == uclone_x.__version__
    assert res["capabilities"]["session"]["load"] is True
    assert res["capabilities"]["session"]["cancel"] is True
    assert res["capabilities"]["prompt"]["streaming"] is True
    assert res["capabilities"]["permissions"]["request"] is True


@pytest.mark.asyncio
async def test_acp_initialize_reports_package_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """initialize must *track* `uclone_x.__version__`, not merely agree with it today.

    `test_acp_initialize_negotiation` asserts agreement between two live values. That
    assertion cannot fail while the server holds a literal that happens to be current,
    which is exactly the state issue #1119 describes: a literal and a test pinning the
    same stale constant, green until someone reads the wire. Agreement only starts
    failing after a bump, when nobody is looking.

    So this test moves `__version__` to a value no declaration in the tree carries and
    requires the report to move with it. A reintroduced literal -- correct or not --
    fails here immediately; the declaration below mutates in an arbitrary one rather
    than today's version, so that `git grep -F` for the current version finds only the
    one place that declares it.

    Killed by: src/uclone_x/shells/acp/server.py :: "version": uclone_x.__version__,
    Becomes: "version": "0.0.0",
    """
    monkeypatch.setattr(uclone_x, "__version__", "9.8.7-probe")

    server = ACPServer()
    resp = await server.dispatch_method("initialize", {}, req_id=1)
    assert resp is not None
    assert resp["result"]["serverInfo"]["version"] == "9.8.7-probe"


@pytest.mark.asyncio
async def test_acp_new_session_refuses_acp_mcp_server() -> None:
    """Test new_session refuses AcpMcpServer per spec §5 / §7.

    Killed by: src/uclone_x/shells/acp/server.py :: "AcpMcpServer is not implementable against today's model: "
    Becomes: "bogus message",
    """
    server = ACPServer()
    # Test transport="acp" refusal
    resp = await server.dispatch_method(
        "new_session",
        {"mcpServers": [{"name": "remote-editor-mcp", "transport": "acp", "serverId": "edit-1"}]},
        req_id=1,
    )
    assert resp is not None
    assert "error" in resp
    assert resp["error"]["code"] == UNSUPPORTED_MCP_TRANSPORT
    assert "AcpMcpServer is not implementable" in resp["error"]["message"]

    # Test serverId presence refusal
    resp2 = await server.dispatch_method(
        "new_session",
        {"mcpServers": [{"name": "remote-editor-mcp", "serverId": "edit-2"}]},
        req_id=2,
    )
    assert resp2 is not None
    assert "error" in resp2
    assert resp2["error"]["code"] == UNSUPPORTED_MCP_TRANSPORT


@pytest.mark.asyncio
async def test_acp_new_session_creation() -> None:
    """Test creating a valid new ACP session.

    Killed by: src/uclone_x/shells/acp/server.py :: {"sessionId": session_id, "mode": state.mode},
    Becomes: {"sessionId": "fixed_bogus", "mode": state.mode},
    """
    store = SessionStore()
    server = ACPServer(store=store)
    resp = await server.dispatch_method(
        "new_session",
        {"sessionId": "test_sess_01", "cwd": "/tmp"},
        req_id=1,
    )
    assert resp is not None
    assert "result" in resp
    assert resp["result"]["sessionId"] == "test_sess_01"
    assert resp["result"]["mode"] == "auto"


@pytest.mark.asyncio
async def test_acp_load_session_success_and_missing() -> None:
    """Test load_session handles existing and missing sessions properly.

    Killed by: src/uclone_x/shells/acp/server.py :: f"Session not found: {session_id}",
    Becomes: "Custom error not found",
    """
    store = SessionStore()
    store.save(SessionState(session_id="saved_session_xyz", agent_id="default"))
    server = ACPServer(store=store)

    # Missing session
    missing_resp = await server.dispatch_method(
        "load_session",
        {"sessionId": "nonexistent_session"},
        req_id=1,
    )
    assert missing_resp is not None
    assert "error" in missing_resp
    assert missing_resp["error"]["code"] == SESSION_NOT_FOUND
    assert missing_resp["error"]["message"] == "Session not found: nonexistent_session"

    # Existing session
    found_resp = await server.dispatch_method(
        "load_session",
        {"sessionId": "saved_session_xyz"},
        req_id=2,
    )
    assert found_resp is not None
    assert "result" in found_resp
    assert found_resp["result"]["sessionId"] == "saved_session_xyz"
    assert found_resp["result"]["mode"] == "auto"


@pytest.mark.asyncio
async def test_acp_set_session_mode() -> None:
    """Test setting session mode with validation.

    Killed by: src/uclone_x/shells/acp/server.py :: valid_modes = {"auto", "ask", "readonly"}
    Becomes: valid_modes = {"auto"}
    """
    server = ACPServer()
    await server.dispatch_method("new_session", {"sessionId": "s1"}, req_id=1)

    # Valid mode change to readonly
    resp = await server.dispatch_method(
        "set_session_mode",
        {"sessionId": "s1", "mode": "readonly"},
        req_id=2,
    )
    assert resp is not None
    assert resp["result"]["mode"] == "readonly"

    # Invalid mode
    bad_resp = await server.dispatch_method(
        "set_session_mode",
        {"sessionId": "s1", "mode": "invalid_mode"},
        req_id=3,
    )
    assert bad_resp is not None
    assert "error" in bad_resp


@pytest.mark.asyncio
async def test_acp_set_config_option() -> None:
    """Test setting session configuration option.

    Killed by: src/uclone_x/shells/acp/server.py :: self._sessions[session_id].config[str(name)] = value
    Becomes: pass
    """
    server = ACPServer()
    await server.dispatch_method("new_session", {"sessionId": "s1"}, req_id=1)

    resp = await server.dispatch_method(
        "set_config_option",
        {"sessionId": "s1", "name": "temperature", "value": 0.7},
        req_id=2,
    )
    assert resp is not None
    assert resp["result"]["name"] == "temperature"
    assert resp["result"]["value"] == 0.7
    session = server.get_session("s1")
    assert session is not None
    assert session.config["temperature"] == 0.7


@pytest.mark.asyncio
async def test_acp_cancel_turn_scoped() -> None:
    """Test turn-scoped cancellation interrupts in-flight task without killing agent.

    Killed by: src/uclone_x/shells/acp/server.py :: task = self.get_in_flight_task(session_id)
    Becomes: task = None
    """
    bus = EventBus()
    agents: list[_FakeSessionAgent] = []
    calls: list[str] = []

    async def hang_first_time(prompt: str) -> TurnResult:
        calls.append(prompt)
        if len(calls) == 1:
            await asyncio.sleep(10.0)
        return TurnResult(
            turn_index=len(calls),
            content="Ready again",
            provenance=Provenance.primary("fake_agent"),
        )

    server = ACPServer(agent_factory=_fake_factory(agents, hang_first_time), bus=bus)
    await server.dispatch_method("new_session", {"sessionId": "cancel_sess"}, req_id=1)
    assert agents[0].saves == ["cancel_sess"]

    # Start prompt turn in background
    await server.dispatch_method("prompt", {"sessionId": "cancel_sess", "prompt": "hang"}, req_id=2)
    assert server.is_turn_in_flight("cancel_sess")
    # Let the turn start, so the cancel interrupts a running turn rather than one that
    # never ran (which has nothing of its own to save).
    await asyncio.sleep(0.01)
    assert calls == ["hang"]

    # Cancel turn
    cancel_resp = await server.dispatch_method("cancel", {"sessionId": "cancel_sess"}, req_id=3)
    assert cancel_resp is not None
    assert cancel_resp["result"]["status"] == "canceled"

    # Wait briefly for task to observe cancellation
    await asyncio.sleep(0.05)
    assert not server.is_turn_in_flight("cancel_sess")
    # What the cancelled turn already added is kept: the session is saved on cancel (#1454).
    assert agents[0].saves == ["cancel_sess", "cancel_sess"]

    # Next prompt should still work on same session, with the same agent
    await server.dispatch_method("prompt", {"sessionId": "cancel_sess", "prompt": "next"}, req_id=4)
    # Wait for completion
    task = server.get_in_flight_task("cancel_sess")
    if task:
        await task
    assert not server.is_turn_in_flight("cancel_sess")
    assert len(agents) == 1
    assert calls == ["hang", "next"]


@pytest.mark.asyncio
async def test_acp_permission_prompt_flow() -> None:
    """Test HookAction.ASK permission prompt integration via ACP request_permission.

    Killed by: src/uclone_x/shells/acp/server.py :: "method": "request_permission",
    Becomes: "method": "wrong_permission_method",
    """
    bus = EventBus()
    server = ACPServer(bus=bus)
    await server.dispatch_method("new_session", {"sessionId": "perm_sess"}, req_id=1)

    sent_messages: list[dict[str, Any]] = []

    async def mock_send(msg: dict[str, Any]) -> None:
        sent_messages.append(msg)
        if msg.get("method") == "request_permission":
            # Emulate client approving permission request
            resp = {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "result": {"allowed": True},
            }
            await server.process_raw_message(json.dumps(resp))

    server.send_response = mock_send  # type: ignore[method-assign]

    sub = bus.subscribe({"session.perm_sess"})

    # Trigger TOOL_APPROVAL_REQUEST event on bus
    req_event = AgentEvent(
        type=EventType.TOOL_APPROVAL_REQUEST,
        topic="session.perm_sess",
        payload={
            "request_id": "appr_123",
            "tool_call_id": "tc_456",
            "tool_name": "bash_run",
            "arguments": {"command": "rm -rf /"},
            "reason": "Dangerous command requires user approval",
        },
    )

    await server.prompt_client_permission("perm_sess", req_event)

    # Verify ACP request_permission was dispatched
    perm_requests = [m for m in sent_messages if m.get("method") == "request_permission"]
    assert len(perm_requests) == 1
    assert perm_requests[0]["params"]["tool"] == "bash_run"
    assert perm_requests[0]["params"]["sessionId"] == "perm_sess"

    # Verify TOOL_APPROVAL_RESPONSE was published on bus
    response_event = await asyncio.wait_for(sub.get(), timeout=2.0)
    assert response_event.type == EventType.TOOL_APPROVAL_RESPONSE
    assert response_event.payload["request_id"] == "appr_123"
    assert response_event.payload["action"] == "allow"


@pytest.mark.asyncio
async def test_acp_stdio_framing_and_lifecycle() -> None:
    """Test reading Content-Length header and newline delimited messages in stdio loop.

    Killed by: src/uclone_x/shells/acp/server.py :: length_str = line.split(":", 1)[1].strip()
    Becomes: length_str = "0"
    """
    reader = asyncio.StreamReader()
    out_lines: list[str] = []

    class MockWriter:
        def write(self, data: bytes) -> None:
            out_lines.append(data.decode("utf-8"))

        async def drain(self) -> None:
            pass

    server = ACPServer(reader=reader, writer=MockWriter())  # type: ignore[arg-type]

    # Feed Content-Length framed initialize request
    req1 = json.dumps({"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {}})
    framed1 = f"Content-Length: {len(req1.encode('utf-8'))}\r\n\r\n{req1}".encode()
    reader.feed_data(framed1)

    # Feed newline-delimited new_session request
    req2 = json.dumps(
        {"jsonrpc": "2.0", "id": 20, "method": "new_session", "params": {"sessionId": "s_stdio"}}
    )
    reader.feed_data((req2 + "\n").encode("utf-8"))

    # Feed EOF
    reader.feed_eof()

    await server.run_stdio()

    assert len(out_lines) == 2
    parsed_resp1 = json.loads(out_lines[0].strip())
    assert parsed_resp1["id"] == 10
    assert parsed_resp1["result"]["serverInfo"]["name"] == SERVER_NAME

    parsed_resp2 = json.loads(out_lines[1].strip())
    assert parsed_resp2["id"] == 20
    assert parsed_resp2["result"]["sessionId"] == "s_stdio"


@pytest.mark.asyncio
async def test_acp_prompt_streaming_and_completion() -> None:
    """Test prompt execution streams semantic updates and returns completion response.

    Killed by: src/uclone_x/shells/acp/server.py :: "status": "completed",
    Becomes: "status": "failed",
    """
    bus = EventBus()

    async def publish_tool_call_then_answer(prompt: str) -> TurnResult:
        # Publish intermediate tool call and result
        await bus.publish(
            AgentEvent(
                type=EventType.TOOL_CALL,
                topic="session.prompt_stream_sess",
                sender_id="mock_agent",
                payload={"tool_name": "test_search", "arguments": {"q": "query"}},
            )
        )
        await asyncio.sleep(0.01)
        await bus.publish(
            AgentEvent(
                type=EventType.TOOL_RESULT,
                topic="session.prompt_stream_sess",
                sender_id="mock_agent",
                payload={"result": "found matches"},
            )
        )
        await asyncio.sleep(0.01)
        return TurnResult(
            turn_index=1,
            content="Completed successfully",
            provenance=Provenance.primary("mock_agent"),
        )

    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_fake_factory(agents, publish_tool_call_then_answer), bus=bus)
    await server.dispatch_method("new_session", {"sessionId": "prompt_stream_sess"}, req_id=1)

    sent_notifications: list[dict[str, Any]] = []

    async def mock_send(msg: dict[str, Any]) -> None:
        sent_notifications.append(msg)

    server.send_response = mock_send  # type: ignore[method-assign]

    # Run prompt via dispatch_method
    await server.dispatch_method(
        "prompt",
        {"sessionId": "prompt_stream_sess", "prompt": [{"type": "text", "text": "Do task"}]},
        req_id=100,
    )

    # Wait for turn task
    task = server.get_in_flight_task("prompt_stream_sess")
    if task:
        await task

    # Verify semantic notifications and final response
    states = [
        n
        for n in sent_notifications
        if n.get("method") == "session_update"
        and n.get("params", {}).get("update", {}).get("type") == "state"
    ]
    assert any(s["params"]["update"]["state"] == "working" for s in states)

    tool_calls = [
        n
        for n in sent_notifications
        if n.get("method") == "session_update"
        and n.get("params", {}).get("update", {}).get("type") == "tool_call"
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0]["params"]["update"]["name"] == "test_search"

    final_resp = [n for n in sent_notifications if n.get("id") == 100]
    assert len(final_resp) == 1
    assert final_resp[0]["result"]["status"] == "completed"
    assert "Completed successfully" in final_resp[0]["result"]["output"]


@pytest.mark.asyncio
async def test_acp_client_driver() -> None:
    """Test ACPClientDriver protocol methods.

    Killed by: src/uclone_x/shells/acp/driver.py :: self._initialized = True
    Becomes: self._initialized = False
    """
    from uclone_x.shells.acp.driver import ACPClientDriver

    driver = ACPClientDriver(session_id="drv_sess")
    await driver.initialize()
    assert driver.is_initialized is True

    events: list[AgentEvent] = []
    async for evt in driver.prompt("Hello agent"):
        events.append(evt)

    assert len(events) == 2
    assert events[0].type == EventType.USER_INPUT
    assert events[1].type == EventType.AGENT_REPLY

    # Test cancel
    await driver.cancel("User aborted")


def test_acp_cli_commands() -> None:
    """Test CLI commands for acp and acp-server.

    Killed by: src/uclone_x/cli/commands/acp.py :: @acp_app.command("serve")
    Becomes: @acp_app.command("bogus_serve")
    """
    from typer.testing import CliRunner

    from uclone_x.cli.main import app

    runner = CliRunner()
    res1 = runner.invoke(app, ["acp-server", "--help"])
    assert res1.exit_code == 0
    assert "acp-server" in res1.output

    res2 = runner.invoke(app, ["acp", "--help"])
    assert res2.exit_code == 0
    assert "serve" in res2.output

    res3 = runner.invoke(app, ["acp", "serve", "--help"])
    assert res3.exit_code == 0


# --------------------------------------------------------------------------------------
# Sessions are saved, and kept apart (#1454)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acp_turn_with_a_tool_call_is_saved_with_its_events(tmp_path: Path) -> None:
    """A prompt turn that calls a tool leaves its messages and its events in the store.

    Killed by: src/uclone_x/shells/acp/server.py :: agent.persist_session(session_id=session_id)
    Becomes: pass
    """
    store = SessionStore(tmp_path / "sessions")
    llm = _RecordingLLM()
    server = _real_server(store, llm)
    sent = _capture(server)

    resp = await server.dispatch_method("new_session", {"sessionId": "acp_saved"}, req_id=1)
    assert resp is not None and "result" in resp, resp
    await _prompt(server, "acp_saved", "please use echo", req_id=2)

    final = [m for m in sent if m.get("id") == 2]
    assert len(final) == 1 and final[0]["result"]["status"] == "completed", final

    record = store.load("acp_saved")
    assert record is not None
    roles = [m.role for m in record.messages]
    assert MessageRole.TOOL in roles, roles
    assert any(
        m.role == MessageRole.USER and m.content == "please use echo" for m in record.messages
    ), record.messages
    assert any(
        m.role == MessageRole.ASSISTANT and m.content == "answer 2" for m in record.messages
    ), record.messages

    log_path = store.event_log_path("acp_saved")
    assert log_path is not None and log_path.is_file(), log_path
    types = [str(e["type"]) for e in read_session_log(log_path)]
    assert "TOOL_CALL" in types and "TOOL_RESULT" in types, types


@pytest.mark.asyncio
async def test_acp_sessions_do_not_share_history(tmp_path: Path) -> None:
    """Session B's request carries none of session A's conversation, and vice versa.

    Killed by: src/uclone_x/shells/acp/server.py :: agent = factory(session_id)
    Becomes: agent = next(iter(self._agents.values()), None) or factory(session_id); agent._context = agent._context.model_copy(update={"session_id": session_id})
    """
    store = SessionStore(tmp_path / "sessions")
    llm = _RecordingLLM()
    server = _real_server(store, llm)
    _capture(server)

    for sid in ("sess_a", "sess_b"):
        resp = await server.dispatch_method("new_session", {"sessionId": sid}, req_id=sid)
        assert resp is not None and "result" in resp, resp

    await _prompt(server, "sess_a", "alpha secret", req_id=10)
    await _prompt(server, "sess_b", "bravo secret", req_id=11)
    await _prompt(server, "sess_a", "alpha again", req_id=12)

    assert len(llm.calls) == 3
    b_request = _request_text(llm.calls[1])
    assert "bravo secret" in b_request
    assert "alpha secret" not in b_request, b_request
    second_a_request = _request_text(llm.calls[2])
    assert "alpha secret" in second_a_request
    assert "bravo secret" not in second_a_request, second_a_request

    a_record = store.load("sess_a")
    b_record = store.load("sess_b")
    assert a_record is not None and b_record is not None
    assert "bravo secret" not in [m.content for m in a_record.messages]
    assert "alpha secret" not in [m.content for m in b_record.messages]


@pytest.mark.asyncio
async def test_acp_load_session_restores_saved_history(tmp_path: Path) -> None:
    """A session reopened by a new server continues its conversation, not a blank one.

    Killed by: src/uclone_x/shells/acp/server.py :: restored = agent.hydrate_session(session_id)
    Becomes: restored = agent.get_session(session_id)
    """
    store = SessionStore(tmp_path / "sessions")
    first = _real_server(store, _RecordingLLM())
    _capture(first)
    await first.dispatch_method("new_session", {"sessionId": "kept"}, req_id=1)
    await _prompt(first, "kept", "remember the lighthouse", req_id=2)

    llm = _RecordingLLM()
    second = _real_server(store, llm)
    _capture(second)
    resp = await second.dispatch_method("load_session", {"sessionId": "kept"}, req_id=3)
    assert resp is not None and "result" in resp, resp
    await _prompt(second, "kept", "what did I say?", req_id=4)

    assert len(llm.calls) == 1
    request = _request_text(llm.calls[0])
    assert "remember the lighthouse" in request, request
    assert "what did I say?" in request


@pytest.mark.asyncio
async def test_acp_prompt_on_an_unknown_session_is_refused_plainly() -> None:
    """A prompt for a session nobody opened is refused, in words, with nothing run.

    Killed by: src/uclone_x/shells/acp/server.py :: return make_jsonrpc_error(req_id, SESSION_NOT_FOUND, UNKNOWN_SESSION_MESSAGE)
    Becomes: self._sessions[session_id] = ACPSessionState(session_id=session_id)
    """
    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_fake_factory(agents))

    resp = await server.dispatch_method(
        "prompt", {"sessionId": "never_opened", "prompt": "hi"}, req_id=1
    )
    assert resp is not None
    assert resp["error"]["code"] == SESSION_NOT_FOUND
    assert resp["error"]["message"] == UNKNOWN_SESSION_MESSAGE
    assert "never_opened" not in resp["error"]["message"]
    assert agents == []
    assert not server.is_turn_in_flight("never_opened")
    assert server.get_session("never_opened") is None


@pytest.mark.asyncio
async def test_acp_idle_agent_is_released_and_rebuilt_from_the_store() -> None:
    """Past the cap an idle, saved agent is dropped; its next prompt rebuilds it.

    Killed by: src/uclone_x/shells/acp/server.py :: del self._agents[victim]
    Becomes: pass
    """
    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_fake_factory(agents), max_live_agents=1)
    _capture(server)

    await server.dispatch_method("new_session", {"sessionId": "one"}, req_id=1)
    resp = await server.dispatch_method("new_session", {"sessionId": "two"}, req_id=2)
    assert resp is not None and "result" in resp, resp
    assert server.session_agent("one") is None
    assert server.session_agent("two") is agents[1]

    await _prompt(server, "one", "back again", req_id=3)
    rebuilt = server.session_agent("one")
    assert rebuilt is agents[2]
    assert agents[2].hydrated == ["one"]
    assert server.session_agent("two") is None


@pytest.mark.asyncio
async def test_acp_unsaved_agent_is_never_released_to_make_room() -> None:
    """An agent whose save failed holds history the store lacks, so it is kept.

    Killed by: src/uclone_x/shells/acp/server.py :: if held_id in self._unsaved or self.is_turn_in_flight(held_id):
    Becomes: if self.is_turn_in_flight(held_id):
    """
    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_fake_factory(agents, fail_saves_after=1), max_live_agents=1)
    _capture(server)

    await server.dispatch_method("new_session", {"sessionId": "one"}, req_id=1)
    await _prompt(server, "one", "not saved", req_id=2)

    resp = await server.dispatch_method("new_session", {"sessionId": "two"}, req_id=3)
    assert resp is not None and "error" in resp, resp
    assert resp["error"]["code"] == INTERNAL_ERROR
    assert resp["error"]["message"] == TOO_MANY_BUSY_MESSAGE
    assert server.session_agent("one") is agents[0]


@pytest.mark.asyncio
async def test_acp_turn_that_cannot_be_saved_is_not_reported_completed() -> None:
    """The reply is shown, but the prompt answers with an error saying it was not saved.

    Killed by: src/uclone_x/shells/acp/server.py :: if req_id is not None and not saved:
    Becomes: if False:
    """
    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_fake_factory(agents, fail_saves_after=1))
    sent = _capture(server)

    await server.dispatch_method("new_session", {"sessionId": "lossy"}, req_id=1)
    await _prompt(server, "lossy", "hello", req_id=2)

    final = [m for m in sent if m.get("id") == 2]
    assert len(final) == 1
    assert "result" not in final[0]
    assert final[0]["error"]["message"] == NOT_SAVED_MESSAGE
    texts = [
        m["params"]["update"]
        for m in sent
        if m.get("method") == "session_update"
        and m.get("params", {}).get("update", {}).get("type") == "text"
    ]
    assert texts, sent


def _final_and_texts(
    sent: list[dict[str, Any]], req_id: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final = [m for m in sent if m.get("id") == req_id]
    texts = [
        m["params"]["update"]
        for m in sent
        if m.get("method") == "session_update"
        and m.get("params", {}).get("update", {}).get("type") == "text"
    ]
    return final, texts


@pytest.mark.asyncio
async def test_acp_refused_turn_is_not_reported_completed_with_empty_output() -> None:
    """A step the agent refused for the context window answers the prompt with that
    refusal, not with `completed` and an empty output that says nothing happened (#1509).
    The refusal is written for the user, so it is sent as it is; no empty text is shown.

    Killed by: src/uclone_x/shells/acp/server.py :: if turn_result.error is not None:
    Becomes: if False:
    """
    from uclone_x.core.tool_results import STEP_OVER_WINDOW_MESSAGE

    async def refuse(prompt: str) -> TurnResult:
        return TurnResult(
            turn_index=1,
            content="",
            error=STEP_OVER_WINDOW_MESSAGE,
            stop_reason="step_results_over_window",
            provenance=Provenance.primary("fake"),
        )

    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_fake_factory(agents, refuse))
    sent = _capture(server)

    await server.dispatch_method("new_session", {"sessionId": "full"}, req_id=1)
    await _prompt(server, "full", "read everything", req_id=2)

    final, texts = _final_and_texts(sent, 2)
    assert len(final) == 1
    assert "result" not in final[0], final[0]
    assert final[0]["error"]["message"] == STEP_OVER_WINDOW_MESSAGE
    assert texts == []
    assert agents[0].saves  # what the turn added is kept, as for any turn


@pytest.mark.asyncio
async def test_acp_turn_error_with_internals_is_answered_plainly() -> None:
    """Any other turn error can carry a provider's text or a path; the client is told
    plainly that the turn failed, and the raw text stays in the log. The partial reply
    the turn did produce is still shown.

    Killed by: src/uclone_x/shells/acp/server.py :: failure = _STOP_REASON_MESSAGES.get(stop_reason, TURN_FAILED_MESSAGE)
    Becomes: failure = turn_result.error
    """

    async def fail(prompt: str) -> TurnResult:
        return TurnResult(
            turn_index=1,
            content="partial",
            error="Error: [Errno 13] Permission denied: '/Users/someone/.uclone/secret'",
            stop_reason="not_started",
            provenance=Provenance.primary("fake"),
        )

    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_fake_factory(agents, fail))
    sent = _capture(server)

    await server.dispatch_method("new_session", {"sessionId": "broken"}, req_id=1)
    await _prompt(server, "broken", "hello", req_id=2)

    final, texts = _final_and_texts(sent, 2)
    assert len(final) == 1
    assert "result" not in final[0], final[0]
    assert final[0]["error"]["message"] == TURN_FAILED_MESSAGE
    _assert_plain(final[0]["error"]["message"])
    assert [t["content"] for t in texts] == ["partial"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("budget_exceeded", USAGE_LIMIT_MESSAGE),
        ("step_budget_exceeded", TOO_MANY_STEPS_MESSAGE),
        ("blocked_by_hook", BLOCKED_MESSAGE),
    ],
)
async def test_acp_turn_a_retry_would_meet_again_is_not_answered_try_again(
    stop_reason: str, expected: str
) -> None:
    """A usage limit, the step limit and a blocking rule each get their own plain words,
    not "please try again" and not the error's own text.

    Only the usage limit is sure to stop a retry: the step budget resets every turn, so
    its message promises nothing about one, and a hook may be time- or rate-based, so its
    message says only that a resend is likely to be stopped.

    Killed by: src/uclone_x/shells/acp/server.py :: "budget_exceeded": USAGE_LIMIT_MESSAGE,
    Becomes: "budget_exceeded": TURN_FAILED_MESSAGE,
    """

    async def stopped(prompt: str) -> TurnResult:
        return TurnResult(
            turn_index=1,
            content="",
            error="BudgetExceededError: ledger /Users/someone/.uclone/budget.json at 100%",
            stop_reason=stop_reason,  # type: ignore[arg-type]
            provenance=Provenance.primary("fake"),
        )

    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_fake_factory(agents, stopped))
    sent = _capture(server)

    await server.dispatch_method("new_session", {"sessionId": "limited"}, req_id=1)
    await _prompt(server, "limited", "hello", req_id=2)

    final, _ = _final_and_texts(sent, 2)
    assert len(final) == 1
    message = final[0]["error"]["message"]
    assert message == expected
    assert message != TURN_FAILED_MESSAGE
    if stop_reason == "step_budget_exceeded":
        for false_claim in ("again", "same way", "will stop"):
            assert false_claim not in message, false_claim
    if stop_reason == "blocked_by_hook":
        assert "will be stopped" not in message
    _assert_plain(message)
    for internal in ("Error", "budget.json", "ledger", "{"):
        assert internal not in message


def _factory_failing_from(
    agents: list[_FakeSessionAgent], nth: int, *, wrong_id: bool = False
) -> Callable[[str], BaseAgent]:
    """Builds fakes until the `nth` call (0-based), which raises -- or, with `wrong_id`,
    builds an agent for another session."""
    build = _fake_factory(agents)
    calls = 0

    def maybe_fail(session_id: str) -> BaseAgent:
        nonlocal calls
        calls += 1
        if calls - 1 >= nth:
            if wrong_id:
                return build("someone_else")
            raise PermissionError(13, "Permission denied", "/Users/someone/.uclone/secret")
        return build(session_id)

    return maybe_fail


def _assert_plain(message: str) -> None:
    for internal in ("Permission denied", "/Users/", "Errno", "Internal error", "factory"):
        assert internal not in message, message


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_id", [False, True])
async def test_acp_new_session_whose_agent_cannot_be_built_is_refused_plainly(
    wrong_id: bool,
) -> None:
    """A factory that fails -- or builds for another id -- is answered in words.

    Killed by: src/uclone_x/shells/acp/server.py :: except _AgentBuildError:
    Becomes: except _NoRoomForAgentError:
    """
    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_factory_failing_from(agents, 0, wrong_id=wrong_id))
    _capture(server)

    resp = await server.dispatch_method("new_session", {"sessionId": "fresh"}, req_id=1)
    assert resp is not None
    assert resp["error"]["message"] == CANNOT_START_MESSAGE
    _assert_plain(resp["error"]["message"])
    assert server.session_agent("fresh") is None
    assert server.get_session("fresh") is None


@pytest.mark.asyncio
async def test_acp_released_session_whose_agent_cannot_be_rebuilt_is_refused_plainly() -> None:
    """A prompt after its agent was released, with a failing factory, says so in words
    and runs no turn.

    This covers the prompt path only. `load_session`'s rebuild with a failing factory is
    `test_acp_load_session_whose_agent_cannot_be_built_is_answered_plainly_on_the_wire`.

    Killed by: src/uclone_x/shells/acp/server.py :: except _AgentBuildError as exc:
    Becomes: except _NoRoomForAgentError as exc:
    """
    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_factory_failing_from(agents, 2), max_live_agents=1)
    _capture(server)

    await server.dispatch_method("new_session", {"sessionId": "one"}, req_id=1)
    await server.dispatch_method("new_session", {"sessionId": "two"}, req_id=2)
    assert server.session_agent("one") is None

    resp = await server.dispatch_method("prompt", {"sessionId": "one", "prompt": "hi"}, req_id=3)
    assert resp is not None
    assert resp["error"]["message"] == CANNOT_OPEN_MESSAGE
    _assert_plain(resp["error"]["message"])
    assert not server.is_turn_in_flight("one")


# --------------------------------------------------------------------------------------
# A live turn keeps its agent; one turn at a time per session (#1494)
# --------------------------------------------------------------------------------------


def _gated(calls: list[str], gate: asyncio.Event) -> Callable[[str], Awaitable[TurnResult]]:
    """An answer that waits for `gate`, so a test can hold a turn in flight."""

    async def execute(prompt: str) -> TurnResult:
        calls.append(prompt)
        await gate.wait()
        return TurnResult(
            turn_index=len(calls), content=f"re: {prompt}", provenance=Provenance.primary("fake")
        )

    return execute


@pytest.mark.asyncio
async def test_acp_agent_mid_turn_is_never_released_to_make_room() -> None:
    """With the cap reached, an agent whose turn is still running is not released.

    Killed by: src/uclone_x/shells/acp/server.py :: if held_id in self._unsaved or self.is_turn_in_flight(held_id):
    Becomes: if held_id in self._unsaved:
    """
    agents: list[_FakeSessionAgent] = []
    calls: list[str] = []
    gate = asyncio.Event()
    server = ACPServer(agent_factory=_fake_factory(agents, _gated(calls, gate)), max_live_agents=1)
    _capture(server)

    await server.dispatch_method("new_session", {"sessionId": "one"}, req_id=1)
    await server.dispatch_method("prompt", {"sessionId": "one", "prompt": "long"}, req_id=2)
    await asyncio.sleep(0.01)
    assert calls == ["long"]

    resp = await server.dispatch_method("new_session", {"sessionId": "two"}, req_id=3)
    assert resp is not None and "error" in resp, resp
    assert resp["error"]["code"] == INTERNAL_ERROR
    assert resp["error"]["message"] == TOO_MANY_BUSY_MESSAGE
    assert server.session_agent("one") is agents[0]
    assert server.get_session("two") is None

    gate.set()
    task = server.get_in_flight_task("one")
    assert task is not None
    await task
    # Once the turn is over the agent is idle and saved, so it can make room.
    resp = await server.dispatch_method("new_session", {"sessionId": "two"}, req_id=4)
    assert resp is not None and "result" in resp, resp
    assert server.session_agent("one") is None


@pytest.mark.asyncio
async def test_acp_second_prompt_while_a_turn_is_in_flight_is_refused_plainly() -> None:
    """A second prompt on a busy session is refused in words; the first turn is untouched.

    It used to start a second task that replaced the first in the record of in-flight
    turns, so whichever finished first removed the other's entry and left a live turn
    unrecorded -- releasable while still answering.

    Killed by: src/uclone_x/shells/acp/server.py :: return make_jsonrpc_error(req_id, INVALID_PARAMS, TURN_IN_FLIGHT_MESSAGE)
    Becomes: pass
    """
    agents: list[_FakeSessionAgent] = []
    calls: list[str] = []
    gate = asyncio.Event()
    server = ACPServer(agent_factory=_fake_factory(agents, _gated(calls, gate)))
    sent = _capture(server)

    await server.dispatch_method("new_session", {"sessionId": "busy_sess"}, req_id=1)
    await server.dispatch_method("prompt", {"sessionId": "busy_sess", "prompt": "first"}, req_id=2)
    first_task = server.get_in_flight_task("busy_sess")
    assert first_task is not None
    await asyncio.sleep(0.01)

    resp = await server.dispatch_method(
        "prompt", {"sessionId": "busy_sess", "prompt": "second"}, req_id=3
    )
    assert resp is not None
    assert resp["error"]["code"] == INVALID_PARAMS
    assert resp["error"]["message"] == TURN_IN_FLIGHT_MESSAGE
    _assert_plain(resp["error"]["message"])
    assert "busy_sess" not in resp["error"]["message"]
    assert server.get_in_flight_task("busy_sess") is first_task

    gate.set()
    await first_task
    assert calls == ["first"]
    final = [m for m in sent if m.get("id") == 2]
    assert len(final) == 1 and final[0]["result"]["status"] == "completed", final
    assert [m for m in sent if m.get("id") == 3] == []


@pytest.mark.asyncio
async def test_acp_cancelled_turn_that_is_still_running_keeps_its_agent() -> None:
    """`cancel` does not remove a turn from the record; the task does, when it ends.

    A turn inside a call that ignores the cancel for a while is still using its agent.
    `cancel` used to pop it from the record at once, so a new session could release that
    agent mid-turn, and a second prompt could start beside it.

    Killed by: src/uclone_x/shells/acp/server.py :: task = self.get_in_flight_task(session_id)
    Becomes: task = self._in_flight_tasks.pop(session_id, None)
    """
    agents: list[_FakeSessionAgent] = []
    gate = asyncio.Event()
    started = asyncio.Event()

    async def ignores_cancel_until_gate(prompt: str) -> TurnResult:
        started.set()
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            await gate.wait()
        return TurnResult(turn_index=1, content="late", provenance=Provenance.primary("fake"))

    server = ACPServer(
        agent_factory=_fake_factory(agents, ignores_cancel_until_gate), max_live_agents=1
    )
    _capture(server)

    await server.dispatch_method("new_session", {"sessionId": "one"}, req_id=1)
    await server.dispatch_method("prompt", {"sessionId": "one", "prompt": "slow"}, req_id=2)
    await started.wait()

    cancel_resp = await server.dispatch_method("cancel", {"sessionId": "one"}, req_id=3)
    assert cancel_resp is not None
    assert cancel_resp["result"]["status"] == "canceled"
    await asyncio.sleep(0.01)
    assert server.is_turn_in_flight("one")

    resp = await server.dispatch_method("new_session", {"sessionId": "two"}, req_id=4)
    assert resp is not None and "error" in resp, resp
    assert resp["error"]["message"] == TOO_MANY_BUSY_MESSAGE
    assert server.session_agent("one") is agents[0]

    gate.set()
    task = server.get_in_flight_task("one")
    assert task is not None
    await task
    assert not server.is_turn_in_flight("one")


# --------------------------------------------------------------------------------------
# A request that fails releases nobody's agent (#1494)
# --------------------------------------------------------------------------------------


def _second_agent_fails(agents: list[_FakeSessionAgent], how: str) -> Callable[[str], BaseAgent]:
    """Builds the first agent normally; the second raises, is for another id, or cannot
    save its first record."""

    def build(session_id: str) -> BaseAgent:
        if agents and how == "raises":
            raise PermissionError(13, "Permission denied", "/Users/someone/.uclone/secret")
        target = "someone_else" if agents and how == "wrong_id" else session_id
        fail_saves_after = 0 if agents and how == "save_fails" else None
        agent = _FakeSessionAgent(target, _answer, fail_saves_after=fail_saves_after)
        agents.append(agent)
        return cast(BaseAgent, agent)

    return build


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["raises", "wrong_id", "save_fails"])
async def test_acp_failed_new_session_releases_no_idle_agent(how: str) -> None:
    """At the cap, a new session that cannot be started leaves the idle agent held.

    The release waits until the new agent is built and its first record saved; releasing
    first dropped an agent for a session that then never opened.

    Killed by: src/uclone_x/shells/acp/server.py :: self._check_room()
    Becomes: self._agents.pop(self._idle_agent_to_release()) if len(self._agents) >= self._max_live_agents else None
    """
    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_second_agent_fails(agents, how), max_live_agents=1)
    _capture(server)

    resp = await server.dispatch_method("new_session", {"sessionId": "one"}, req_id=1)
    assert resp is not None and "result" in resp, resp

    resp = await server.dispatch_method("new_session", {"sessionId": "two"}, req_id=2)
    assert resp is not None and "error" in resp, resp
    assert resp["error"]["message"] == CANNOT_START_MESSAGE
    _assert_plain(resp["error"]["message"])
    assert server.session_agent("one") is agents[0]
    assert server.session_agent("two") is None
    assert server.get_session("two") is None


# --------------------------------------------------------------------------------------
# The reply a client receives, through the message entrypoint (#1494)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acp_new_session_whose_agent_cannot_be_built_is_answered_plainly_on_the_wire() -> (
    None
):
    """The client receives exactly one JSON-RPC error, in words, for a failed build.

    Killed by: src/uclone_x/shells/acp/server.py :: await self.send_response(result)
    Becomes: await self.send_response({"jsonrpc": "2.0", "id": req_id, "result": result})
    """
    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_factory_failing_from(agents, 0))
    sent = _capture(server)

    await server.process_raw_message(
        json.dumps(
            {"jsonrpc": "2.0", "id": 7, "method": "new_session", "params": {"sessionId": "fresh"}}
        )
    )

    assert sent == [
        {
            "jsonrpc": "2.0",
            "id": 7,
            "error": {"code": INTERNAL_ERROR, "message": CANNOT_START_MESSAGE},
        }
    ]
    _assert_plain(json.dumps(sent))
    assert "fresh" not in json.dumps(sent)


@pytest.mark.asyncio
async def test_acp_load_session_whose_agent_cannot_be_built_is_answered_plainly_on_the_wire(
    tmp_path: Path,
) -> None:
    """Reopening a saved session with a failing factory answers the client in words.

    Killed by: src/uclone_x/shells/acp/server.py :: except _AgentBuildError as exc:
    Becomes: except _NoRoomForAgentError as exc:
    """
    store = SessionStore(tmp_path / "sessions")
    store.save(SessionState(session_id="kept", agent_id="fake"))
    agents: list[_FakeSessionAgent] = []
    server = ACPServer(agent_factory=_factory_failing_from(agents, 0), store=store)
    sent = _capture(server)

    await server.process_raw_message(
        json.dumps(
            {"jsonrpc": "2.0", "id": 8, "method": "load_session", "params": {"sessionId": "kept"}}
        )
    )

    assert sent == [
        {
            "jsonrpc": "2.0",
            "id": 8,
            "error": {"code": INTERNAL_ERROR, "message": CANNOT_OPEN_MESSAGE},
        }
    ]
    _assert_plain(json.dumps(sent))
    assert server.get_session("kept") is None
    assert server.session_agent("kept") is None


@pytest.mark.asyncio
async def test_acp_unexpected_failure_is_answered_plainly_on_the_wire() -> None:
    """A handler that raises is answered in words; the exception text goes to the log only.

    The catch-all used to send `Internal error: <exception>`, which can carry a path.

    Killed by: src/uclone_x/shells/acp/server.py :: make_jsonrpc_error(req_id, INTERNAL_ERROR, REQUEST_FAILED_MESSAGE)
    Becomes: make_jsonrpc_error(req_id, INTERNAL_ERROR, f"Internal error: {sys.exc_info()[1]}")
    """
    server = ACPServer()
    sent = _capture(server)

    async def broken(params: dict[str, Any], req_id: int | str | None) -> dict[str, Any]:
        raise PermissionError(13, "Permission denied", "/Users/someone/.uclone/secret")

    server._handle_initialize = broken  # type: ignore[method-assign]

    await server.process_raw_message(
        json.dumps({"jsonrpc": "2.0", "id": 9, "method": "initialize", "params": {}})
    )

    assert sent == [
        {
            "jsonrpc": "2.0",
            "id": 9,
            "error": {"code": INTERNAL_ERROR, "message": REQUEST_FAILED_MESSAGE},
        }
    ]
    _assert_plain(json.dumps(sent))


# --------------------------------------------------------------------------------------
# The editor's clone keeps its memory under the id the desktop app uses (#1494)
# --------------------------------------------------------------------------------------


class _MemoryOpened(Exception):
    """Stops `start_acp_server` once it has named the memory it would open."""


def _memory_ids_opened(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[str]:
    """Run the CLI with `argv` up to the point it opens memory; the ids it asked for."""
    from typer.testing import CliRunner

    from uclone_x.cli.commands import acp as acp_command
    from uclone_x.cli.main import app

    opened: list[str] = []

    def record(agent_id: str) -> Any:
        opened.append(agent_id)
        raise _MemoryOpened()

    monkeypatch.setattr(acp_command, "memory_for_agent_id", record)
    result = CliRunner().invoke(app, argv)
    assert isinstance(result.exception, _MemoryOpened), result.output
    return opened


def test_acp_serve_keeps_memory_under_the_persona_name_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ucx acp serve` with no `--agent-id` keeps memory under the persona's name.

    The desktop app keys each agent by its persona's name, and the agent id names the
    memory store; the old default, `default`, kept the editor's memory apart from the app's.

    Killed by: src/uclone_x/cli/commands/acp.py :: agent_id: Annotated[str | None, typer.Option("--agent-id", help=AGENT_ID_HELP)] = None,
    Becomes: agent_id: Annotated[str | None, typer.Option("--agent-id", help=AGENT_ID_HELP)] = "default",
    """
    assert _memory_ids_opened(monkeypatch, ["acp", "serve"]) == [DEFAULT_PERSONA_NAME]


def test_acp_server_alias_keeps_memory_under_the_persona_name_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ucx acp-server`, the older spelling, follows the persona the same way.

    Killed by: src/uclone_x/cli/main.py :: agent_id: str | None = typer.Option(None, "--agent-id", help=ACP_AGENT_ID_HELP),
    Becomes: agent_id: str | None = typer.Option("default", "--agent-id", help=ACP_AGENT_ID_HELP),
    """
    assert _memory_ids_opened(monkeypatch, ["acp-server"]) == [DEFAULT_PERSONA_NAME]


def test_acp_serve_keeps_an_explicit_agent_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit `--agent-id` still names the memory, whatever the persona.

    Killed by: src/uclone_x/cli/commands/acp.py :: start_acp_server(agent_id=agent_id, persona=persona)
    Becomes: start_acp_server(agent_id=None, persona=persona)
    """
    opened = _memory_ids_opened(monkeypatch, ["acp", "serve", "--agent-id", "editor-clone"])
    assert opened == ["editor-clone"]
