"""Integration tests for oversized tool output offloading and recovery (Issue #472)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.core.tool_results import handle_in
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest
from uclone_x.tools.builtin.filesystem import FileReadTool
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import LocalTool, ToolRegistry


@pytest.mark.asyncio
async def test_tool_output_offload_recovery_and_cleanup(tmp_path: Path) -> None:
    """L2 integration test:
    1. Tool produces oversized output exceeding max_tool_output_chars.
    2. Context compactor offloads to the session's result store instead of truncating.
    3. The stored body holds the full content, including the middle section, and it
       is kept in the session store, not the workspace (#1848).
    4. Session deletion removes the kept body from disk.
    """
    ws = tmp_path / "workspace"
    ws.mkdir()
    sessions_dir = tmp_path / "sessions"

    bus = EventBus(maxsize=100)
    await bus.start()

    store = SessionStore(storage_dir=sessions_dir)
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
        # Since #1640 the offload goes to the `tr_` result store. This agent has
        # `file_read` but not `tool_result_read`, so the stub names neither.
        handle = handle_in(tool_msg.content)
        assert handle is not None, f"No stored-result handle in: {tool_msg.content}"
        assert "file_read" not in tool_msg.content
        assert "tool_result_read" not in tool_msg.content

        # 3. The stored body is the whole output, middle section included, and it is
        # kept among the session's context bodies rather than under the workspace.
        agent.persist_session()
        bodies = agent._result_bodies(agent.session_id)  # pyright: ignore[reportPrivateUsage]
        stored = bodies.read(handle)
        assert stored is not None
        assert stored == large_payload
        assert "RECOVERABLE_KEY_#472" in stored
        # Stored once (#2013): the result fit the history whole, so the body that holds it
        # is its message's entry, and compaction names that entry instead of writing the
        # text again as a body of its own.
        body_dir = store.context_body_dir(agent.session_id)
        files = list(body_dir.iterdir())
        assert all(f.read_text(encoding="utf-8") != large_payload for f in files)
        kept = [
            f
            for f in files
            if f.read_text(encoding="utf-8").startswith("{")
            and (m := ChatMessage.model_validate_json(f.read_text(encoding="utf-8"))).role
            == MessageRole.TOOL
            and m.content == large_payload
        ]
        assert len(kept) == 1
        assert not (ws / ".sandbox" / "tool_artifacts").exists()

        # 4. Deleting the session removes the kept body
        deleted = agent.delete_session()
        assert deleted is True
        assert not kept[0].exists()
        assert not body_dir.exists()

    finally:
        sub.close()
        await agent.stop()
        await bus.stop()
