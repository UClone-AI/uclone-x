# pyright: reportPrivateUsage=false
"""Unit tests for BaseAgent P3 sandbox isolation levels and workspace path validation (Issue #127)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    AgentState,
)
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.errors import PathTraversalError
from uclone_x.llm.models import (
    FinishReason,
    MessageRole,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.sandbox.models import (
    ContainerIsolation,
    IsolationLevel,
    NoIsolation,
    WorkspaceIsolation,
)
from uclone_x.tools import (
    BaseTool,
    LocalTool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    create_default_registry,
)


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    ws = tmp_path / "agent_ws"
    ws.mkdir()
    return ws


def _response(
    content: str = "",
    tool_calls: tuple[ToolCallRequest, ...] = (),
    finish_reason: FinishReason = FinishReason.STOP,
    provenance: Provenance | None = None,
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=TokenUsage(provider="mock", input_tokens=10, output_tokens=10),
        provenance=provenance or Provenance.primary("mock"),
    )


def _create_mock_llm(responses: list[ModelResponse]) -> MagicMock:
    """A stub model. If the script ends on a tool call, a terminal answer is appended.

    The agent takes agent steps until the model stops asking for tools (P4, amended
    2026-09-05), so a script whose last entry requests a tool describes a model that never
    answers. Left as-is these stubs exhausted `side_effect` on the second step. Appending
    the answer models what a real model does and keeps each test about its own subject.
    """
    # A terminal answer after *every* tool-calling response, not just the last. Each
    # entry in these scripts is written as one turn, and a turn is now a step loop: the
    # model asks for a tool, sees the result, and answers. Appending only at the end let
    # one turn consume the next turn's entry and then exhaust the script.
    script: list[ModelResponse] = []
    for resp in responses:
        script.append(resp)
        if resp.tool_calls:
            script.append(_response(content="Done."))
    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(side_effect=script)
    return mock_llm


@pytest.mark.asyncio
async def test_agent_tool_execution_receives_valid_tool_context(workspace_dir: Path) -> None:
    """BaseAgent tool execution constructs ToolContext with workspace_root and isolation level."""
    received_contexts: list[ToolContext] = []

    async def custom_handler(params: dict[str, Any], context: ToolContext) -> ToolResult:
        received_contexts.append(context)
        return ToolResult(
            success=True,
            output={"result": "ok"},
            isolation_level=context.isolation.level,
            provenance=Provenance.primary("custom_tool"),
        )

    registry = ToolRegistry()
    registry.register(LocalTool(name="test_tool", description="Test tool", handler=custom_handler))

    config = AgentConfig(
        agent_id="test_agent",
        name="Test Agent",
        workspace_dir=workspace_dir,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    tool_call = ToolCallRequest(
        id="call_ctx_1",
        name="test_tool",
        arguments={"x": 1},
    )

    mock_llm = _create_mock_llm(
        [
            _response(
                tool_calls=(tool_call,),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
        ]
    )

    agent = BaseAgent(config=config, llm=mock_llm, tools=registry)
    result = await agent.execute_turn("Run test tool")

    assert result.is_completed is True
    assert len(received_contexts) == 1
    ctx = received_contexts[0]
    assert ctx.agent_id == "test_agent"
    assert ctx.require_workspace() == workspace_dir.resolve()
    assert ctx.isolation.level is IsolationLevel.WORKSPACE
    assert isinstance(ctx.isolation, WorkspaceIsolation)


@pytest.mark.asyncio
async def test_agent_workspace_root_resolution_precedence(
    workspace_dir: Path, tmp_path: Path
) -> None:
    """AgentContext.require_workspace() overrides AgentConfig.workspace_dir, and defaults to cwd."""
    received_contexts: list[ToolContext] = []

    async def handler(params: dict[str, Any], context: ToolContext) -> ToolResult:
        received_contexts.append(context)
        return ToolResult(success=True, output="ok", provenance=Provenance.primary("tool"))

    registry = ToolRegistry()
    registry.register(LocalTool(name="prec_tool", description="Test", handler=handler))

    override_ws = tmp_path / "override_ws"
    override_ws.mkdir()

    # Case 1: Runtime context override takes precedence over config
    config1 = AgentConfig(
        agent_id="agent1",
        name="Agent 1",
        workspace_dir=workspace_dir,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    context1 = AgentContext(
        session_id="sess_override",
        agent_id="agent1",
        workspace_root=override_ws,
    )

    # One stub per agent. A single script shared by both was consumed entirely by the
    # first agent once a turn became a step loop, so `received_contexts[1]` held that
    # agent's second tool round rather than the second agent's first.
    def _tool_once(call_id: str) -> MagicMock:
        return _create_mock_llm(
            [
                _response(
                    tool_calls=(ToolCallRequest(id=call_id, name="prec_tool", arguments={}),),
                    finish_reason=FinishReason.TOOL_CALLS,
                )
            ]
        )

    agent1 = BaseAgent(config=config1, llm=_tool_once("c1"), tools=registry, context=context1)
    await agent1.execute_turn("First turn")
    assert received_contexts[0].require_workspace() == override_ws.resolve()

    # Case 2: No workspace_dir and no context override results in None
    config2 = AgentConfig(
        agent_id="agent2",
        name="Agent 2",
        workspace_dir=None,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    agent2 = BaseAgent(config=config2, llm=_tool_once("c2"), tools=registry)
    await agent2.execute_turn("Second turn")
    assert received_contexts[1].workspace_root is None


@pytest.mark.asyncio
async def test_agent_isolation_clamping_prevents_weaker_isolation(workspace_dir: Path) -> None:
    """P3 Containment: Requesting NoIsolation in AgentConfig is clamped to WorkspaceIsolation floor."""
    received_contexts: list[ToolContext] = []

    async def handler(params: dict[str, Any], context: ToolContext) -> ToolResult:
        received_contexts.append(context)
        return ToolResult(success=True, output="ok", provenance=Provenance.primary("tool"))

    registry = ToolRegistry()
    registry.register(LocalTool(name="clamp_tool", description="Test", handler=handler))

    # Requesting NoIsolation() must be clamped to WorkspaceIsolation
    config = AgentConfig(
        agent_id="weak_agent",
        name="Weak Agent",
        workspace_dir=workspace_dir,
        isolation=NoIsolation(),
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    mock_llm = _create_mock_llm(
        [
            _response(
                tool_calls=(ToolCallRequest(id="c1", name="clamp_tool", arguments={}),),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        ]
    )

    agent = BaseAgent(config=config, llm=mock_llm, tools=registry)
    result = await agent.execute_turn("Run clamp tool")

    assert result.is_completed is True
    assert len(received_contexts) == 1
    # Effective isolation clamped to WorkspaceIsolation
    assert received_contexts[0].isolation.level is IsolationLevel.WORKSPACE
    assert isinstance(received_contexts[0].isolation, WorkspaceIsolation)


@pytest.mark.asyncio
async def test_agent_unsupported_isolation_level_fails_fast(workspace_dir: Path) -> None:
    """P6 Fail-Fast: Requesting container isolation when unavailable raises SandboxViolationError in turn."""
    registry = ToolRegistry()
    registry.register(LocalTool(name="tool", description="tool", handler=AsyncMock()))

    config = AgentConfig(
        agent_id="container_agent",
        name="Container Agent",
        workspace_dir=workspace_dir,
        isolation=ContainerIsolation(image="alpine:latest"),
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    mock_llm = _create_mock_llm(
        [
            _response(
                tool_calls=(ToolCallRequest(id="c1", name="tool", arguments={}),),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        ]
    )

    agent = BaseAgent(config=config, llm=mock_llm, tools=registry)
    result = await agent.execute_turn("Trigger turn")

    assert result.is_completed is False
    assert result.error is not None
    assert "has no available runner backend" in result.error
    assert agent.state is AgentState.ERROR


@pytest.mark.asyncio
async def test_path_traversal_exception_is_caught_and_formatted(
    workspace_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """BaseAgent catches PathTraversalError from tools, logs structured warning, and informs LLM."""

    class TraversalParams(BaseModel):
        path: str = ""

    class TraversalTool(BaseTool[TraversalParams]):
        name = "traversal_tool"
        description = "Simulates path traversal"

        def run(self, params: TraversalParams, context: ToolContext) -> Any:
            raise PathTraversalError(
                f"Path '/etc/shadow' resolves outside workspace '{context.require_workspace()}'"
            )

    registry = ToolRegistry()
    registry.register(TraversalTool())

    config = AgentConfig(
        agent_id="agent_traversal",
        name="Agent Traversal",
        workspace_dir=workspace_dir,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    tool_call = ToolCallRequest(
        id="call_escape_1",
        name="traversal_tool",
        arguments={"path": "/etc/shadow"},
    )

    mock_llm = _create_mock_llm(
        [
            _response(
                content="I will try to read the system file",
                tool_calls=(tool_call,),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
        ]
    )

    agent = BaseAgent(config=config, llm=mock_llm, tools=registry)

    with caplog.at_level(logging.WARNING):
        result = await agent.execute_turn("Read shadow file")

    # Turn completes without crashing
    assert result.is_completed is True
    assert agent.state is AgentState.IDLE

    # History contains the tool response with Path traversal violation
    tool_messages = [m for m in agent._history if m.role is MessageRole.TOOL]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_escape_1"
    assert tool_messages[0].name == "traversal_tool"
    assert "Path traversal violation" in str(tool_messages[0].content)

    # Structured warning was logged
    assert any(
        "Path traversal violation during tool 'traversal_tool'" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_direct_path_traversal_exception_from_tool_is_caught(
    workspace_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """BaseAgent catches raw PathTraversalError raised directly from a tool's execute method."""

    async def raw_traversal_handler(params: dict[str, Any], context: ToolContext) -> ToolResult:
        raise PathTraversalError("Direct escaping path violation")

    registry = ToolRegistry()
    registry.register(
        LocalTool(name="raw_traversal", description="Raw", handler=raw_traversal_handler)
    )

    config = AgentConfig(
        agent_id="raw_agent",
        name="Raw Agent",
        workspace_dir=workspace_dir,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    tool_call = ToolCallRequest(
        id="call_raw_escape",
        name="raw_traversal",
        arguments={},
    )

    mock_llm = _create_mock_llm(
        [
            _response(
                content="Triggering raw traversal",
                tool_calls=(tool_call,),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
        ]
    )

    agent = BaseAgent(config=config, llm=mock_llm, tools=registry)

    with caplog.at_level(logging.WARNING):
        result = await agent.execute_turn("Test raw traversal")

    assert result.is_completed is True
    tool_messages = [m for m in agent._history if m.role is MessageRole.TOOL]
    assert len(tool_messages) == 1
    assert "Path traversal violation: Direct escaping path violation" in str(
        tool_messages[0].content
    )
    assert any(
        "Path traversal violation during tool 'raw_traversal'" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_builtin_filesystem_tools_with_base_agent(workspace_dir: Path) -> None:
    """End-to-end integration: BaseAgent executing builtin filesystem tools within workspace."""
    registry = create_default_registry()

    config = AgentConfig(
        agent_id="fs_agent",
        name="FS Agent",
        workspace_dir=workspace_dir,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    # 1. First turn: Agent creates a file via file_write
    write_call = ToolCallRequest(
        id="call_write_1",
        name="file_write",
        arguments={"path": "notes.txt", "content": "Hello UClone-X\nLine 2\n"},
    )
    # 2. Second turn: Agent attempts path traversal via file_read
    traversal_call = ToolCallRequest(
        id="call_read_escape",
        name="file_read",
        arguments={"path": "../../outside_secret.txt"},
    )
    # 3. Third turn: Agent reads the valid file via file_read
    read_call = ToolCallRequest(
        id="call_read_ok",
        name="file_read",
        arguments={"path": "notes.txt"},
    )

    mock_llm = _create_mock_llm(
        [
            _response(
                content="Writing notes",
                tool_calls=(write_call,),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
            _response(
                content="Reading outside file",
                tool_calls=(traversal_call,),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
            _response(
                content="Reading notes",
                tool_calls=(read_call,),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
        ]
    )

    agent = BaseAgent(config=config, llm=mock_llm, tools=registry)

    # Turn 1: Write file
    res1 = await agent.execute_turn("Write notes")
    assert res1.is_completed is True
    assert (workspace_dir / "notes.txt").exists()
    assert (workspace_dir / "notes.txt").read_text() == "Hello UClone-X\nLine 2\n"

    # Turn 2: Traversal attempt blocked
    res2 = await agent.execute_turn("Read outside")
    assert res2.is_completed is True
    tool_msg_traversal = [m for m in agent._history if m.tool_call_id == "call_read_escape"][0]
    assert "Path traversal violation" in str(tool_msg_traversal.content)

    # Turn 3: Read valid file
    res3 = await agent.execute_turn("Read notes")
    assert res3.is_completed is True
    tool_msg_read = [m for m in agent._history if m.tool_call_id == "call_read_ok"][0]
    assert "Hello UClone-X" in str(tool_msg_read.content)


@pytest.mark.asyncio
async def test_agent_tool_unexpected_exception_handling(workspace_dir: Path) -> None:
    """BaseAgent catches unexpected tool exceptions, records tool failure message, and continues."""

    async def exploding_handler(params: dict[str, Any], context: ToolContext) -> ToolResult:
        raise RuntimeError("Database connection suddenly dropped")

    registry = ToolRegistry()
    registry.register(
        LocalTool(name="exploding_tool", description="Explodes", handler=exploding_handler)
    )

    config = AgentConfig(
        agent_id="error_agent",
        name="Error Agent",
        workspace_dir=workspace_dir,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    mock_llm = _create_mock_llm(
        [
            _response(
                content="Calling tool",
                tool_calls=(ToolCallRequest(id="call_err_1", name="exploding_tool", arguments={}),),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        ]
    )

    agent = BaseAgent(config=config, llm=mock_llm, tools=registry)
    result = await agent.execute_turn("Run exploding tool")

    assert result.is_completed is True
    tool_messages = [m for m in agent._history if m.role is MessageRole.TOOL]
    assert len(tool_messages) == 1
    assert "Tool execution failed: RuntimeError: Database connection suddenly dropped" in str(
        tool_messages[0].content
    )


@pytest.mark.asyncio
async def test_subagent_inherits_workspace_dir_and_isolation(workspace_dir: Path) -> None:
    """Subagents dynamically spawned inherit workspace_dir and isolation from parent agent."""
    parent_config = AgentConfig(
        agent_id="parent_leader",
        name="Parent Leader",
        workspace_dir=workspace_dir,
        isolation=WorkspaceIsolation(write_paths=(Path("sub_output"),)),
    )

    parent = BaseAgent(config=parent_config)
    subagent = await parent.spawn_subagent(
        role="worker",
        goal="Process workspace items",
    )

    assert subagent.config.workspace_dir == workspace_dir
    assert subagent.config.isolation.level is IsolationLevel.WORKSPACE
    assert isinstance(subagent.config.isolation, WorkspaceIsolation)
    assert subagent.config.isolation.write_paths == (Path("sub_output"),)
    assert subagent.context.parent_agent_id == "parent_leader"


@pytest.mark.asyncio
async def test_event_loop_survives_tool_traversal_and_continues_consuming(
    workspace_dir: Path,
) -> None:
    """Reactive event loop does not crash on tool path traversal and continues consuming subsequent events."""
    bus = EventBus()
    await bus.start()

    registry = create_default_registry()

    config = AgentConfig(
        agent_id="loop_agent",
        name="Loop Agent",
        workspace_dir=workspace_dir,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    # Tool call with traversal
    traversal_call = ToolCallRequest(
        id="call_trav_loop",
        name="file_read",
        arguments={"path": "../../forbidden.txt"},
    )
    # Tool call with valid write
    valid_call = ToolCallRequest(
        id="call_valid_loop",
        name="file_write",
        arguments={"path": "result.txt", "content": "loop ok"},
    )

    mock_llm = _create_mock_llm(
        [
            _response(
                content="Attempting bad read",
                tool_calls=(traversal_call,),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
            _response(
                content="Attempting good write",
                tool_calls=(valid_call,),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
        ]
    )

    agent = BaseAgent(config=config, bus=bus, llm=mock_llm, tools=registry)
    await agent.start()

    caller = bus.register_publisher(sender_id="caller_1")
    replies = bus.subscribe(f"session.sess_{config.agent_id}")

    try:
        # Send first event (causes path traversal attempt in tool)
        await caller.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                recipient_id=agent.agent_id,
                topic=f"agent.{agent.agent_id}",
                payload={"message": "read forbidden"},
            )
        )
        await bus.wait_until_idle()

        reply1 = await asyncio.wait_for(replies.get(), timeout=1.0)
        assert reply1.type is EventType.AGENT_REPLY
        assert reply1.payload["is_completed"] == "True"

        # Loop is healthy and processing errors is empty
        assert agent.processing_errors == ()
        assert agent.state is AgentState.IDLE

        # Send second event (valid file write)
        await caller.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                recipient_id=agent.agent_id,
                topic=f"agent.{agent.agent_id}",
                payload={"message": "write result"},
            )
        )
        await bus.wait_until_idle()

        reply2 = await asyncio.wait_for(replies.get(), timeout=1.0)
        assert reply2.type is EventType.AGENT_REPLY
        assert reply2.payload["is_completed"] == "True"
        assert (workspace_dir / "result.txt").exists()
        assert (workspace_dir / "result.txt").read_text() == "loop ok"

    finally:
        replies.close()
        await agent.stop()
        await bus.stop()
