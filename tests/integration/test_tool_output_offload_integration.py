"""Integration tests for oversized tool output offloading and recovery (Issue #472)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.errors import PathTraversalError
from uclone_x.llm.compactor import ContextCompactor
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest
from uclone_x.tools.builtin.filesystem import FileReadParams, FileReadTool
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import LocalTool, ToolRegistry


@pytest.mark.asyncio
async def test_tool_output_offload_recovery_and_cleanup(tmp_path: Path) -> None:
    """L2 integration test:
    1. Tool produces oversized output exceeding max_tool_output_chars.
    2. Context compactor offloads to sandbox filesystem instead of truncating.
    3. FileReadTool reads back full content including middle section.
    4. Path traversal attempt during offload is refused.
    5. Session deletion cleans up tool artifacts from disk.
    """
    ws = tmp_path / "workspace"
    ws.mkdir()
    sessions_dir = tmp_path / "sessions"

    bus = EventBus(maxsize=100)
    await bus.start()

    store = SessionStore(storage_dir=sessions_dir, workspace_root=ws)
    tools = ToolRegistry()

    # Register FileReadTool
    read_tool = FileReadTool()
    tools.register(read_tool)

    # Register custom tool that returns oversized data with a secret middle section
    large_payload = (
        "START_CHUNK\n" + ("A" * 2000) + "\nRECOVERABLE_KEY_#472\n" + ("B" * 2000) + "\nEND_CHUNK"
    )

    async def fetch_big_data(args: dict[str, object], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            output=large_payload,
            success=True,
            provenance=Provenance.primary("test.fetch_big_data"),
        )

    tools.register(
        LocalTool(
            name="fetch_big_data",
            description="Returns large tool output",
            parameters_schema={"type": "object", "properties": {}},
            handler=fetch_big_data,
        )
    )

    tc = ToolCallRequest(
        id="call_big_1",
        name="fetch_big_data",
        arguments={},
    )
    llm = MockLLMConnector(
        responses=["Data fetched, inspecting..."],
        tool_calls=[tc],
    )

    config = AgentConfig(
        agent_id="test-offload-agent",
        name="OffloadAgent",
        workspace_dir=str(ws),
        llm_config=AgentLLMConfig(
            model_name="mock-model",
            auto_compact=False,
        ),
    )

    agent = BaseAgent(
        config=config,
        bus=bus,
        llm=llm,
        tools=tools,
        store=store,
    )

    await agent.start()
    sub = bus.subscribe({f"session.{agent.session_id}"}, recipient_id="tester")

    try:
        # 1. Trigger agent turn
        await bus.publish(
            AgentEvent(
                type=EventType.USER_INPUT,
                topic=f"session.{agent.session_id}",
                sender_id="tester",
                payload={"message": "Run fetch_big_data"},
            )
        )

        # Wait for agent reply
        reply_evt = None
        for _ in range(50):
            if not sub.empty():
                evt = await sub.get()
                if evt.type == EventType.AGENT_REPLY:
                    reply_evt = evt
                    break
            await asyncio.sleep(0.05)

        assert reply_evt is not None, "Agent did not produce reply"

        # 2. Compact session to trigger offloading of oversized tool outputs
        result = await agent.compact_session()
        assert result.saved_tokens > 0

        # Find the tool message in live session
        messages = agent._live_session(agent.session_id).messages  # pyright: ignore[reportPrivateUsage]
        tool_msg = next(m for m in messages if m.role == MessageRole.TOOL)
        assert tool_msg.content is not None
        assert "[Tool Output Offloaded" in tool_msg.content
        assert "path=offload" in tool_msg.content

        # 3. Extract the artifact path and verify recovery via FileReadTool
        match = re.search(r"Full output saved to '([^']+)'", tool_msg.content)
        assert match is not None, f"Artifact path not found in: {tool_msg.content}"
        artifact_rel_path = match.group(1)

        tool_ctx = ToolContext(
            agent_id=agent.agent_id, workspace_root=ws, session_id=agent.session_id
        )
        read_result = read_tool.run(FileReadParams(path=artifact_rel_path), tool_ctx)

        assert read_result["content"] == large_payload
        assert "RECOVERABLE_KEY_#472" in read_result["content"]

        # 4. Assert traversal refusal
        compactor_bad = ContextCompactor(
            workspace_root=ws,
            session_id="../../escaped",
            max_tool_output_chars=100,
        )
        oversized_msg = ChatMessage(
            role=MessageRole.TOOL,
            name="evil_tool",
            content="E" * 500,
            tool_call_id="call_evil",
        )
        with pytest.raises(PathTraversalError):
            compactor_bad.prune_tool_message(oversized_msg)

        # 5. Verify session artifact cleanup
        artifact_file = ws / artifact_rel_path
        assert artifact_file.is_file()

        # Delete session
        deleted = agent.delete_session()
        assert deleted is True

        # Artifact file and session directory should now be removed
        assert not artifact_file.exists()
        assert not artifact_file.parent.exists()

    finally:
        sub.close()
        await agent.stop()
        await bus.stop()
