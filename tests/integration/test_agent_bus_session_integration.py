"""L2 Integration test: real BaseAgent + EventBus + SessionStore + Tools assembly."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ToolCallRequest
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import LocalTool, ToolRegistry


@pytest.mark.asyncio
async def test_agent_event_bus_tool_session_real_assembly(tmp_path: Path) -> None:
    """Verify end-to-end event flow across real EventBus, BaseAgent, SessionStore and ToolRegistry."""
    # 1. Assembled components
    bus = EventBus(maxsize=100)
    await bus.start()
    store = SessionStore(storage_dir=tmp_path / "sessions")
    tools = ToolRegistry()

    # Register a real tool that performs state mutation
    async def sample_tool_handler(args: dict[str, object], ctx: ToolContext) -> ToolResult:
        key = str(args.get("key", ""))
        val = str(args.get("value", ""))
        return ToolResult(
            output={"stored_key": key, "stored_val": val, "status": "ok"},
            success=True,
            provenance=Provenance.primary(provider="tool.record_metric", model="record_metric"),
        )

    tools.register(
        LocalTool(
            name="record_metric",
            description="Record a key-value metric",
            parameters_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
            handler=sample_tool_handler,
        )
    )

    # Mock connector returning scripted tool call on turn 1, then completion on turn 2
    tc = ToolCallRequest(
        id="call_test_123",
        name="record_metric",
        arguments={"key": "cpu_util", "value": "42%"},
    )
    mock_llm = MockLLMConnector(
        responses=["Observation recorded. Metric stored successfully."],
        tool_calls=[tc],
    )

    config = AgentConfig(
        agent_id="test-assembler",
        name="TestAssembler",
        system_prompt="You are a test coordinator.",
    )

    agent = BaseAgent(
        config=config,
        bus=bus,
        llm=mock_llm,
        tools=tools,
        store=store,
    )

    # 2. Subscribe to agent reply events on the bus
    sub = bus.subscribe(topics={f"session.{agent.session_id}"})

    # 3. Start the agent reactive consumer
    await agent.start()

    try:
        # Publish USER_INPUT to the agent
        input_event = AgentEvent(
            type=EventType.USER_INPUT,
            topic=f"session.{agent.session_id}",
            sender_id="test-client",
            recipient_id=agent.agent_id,
            session_id=agent.session_id,
            payload={"message": "Please record the CPU metric"},
        )
        await bus.publish(input_event)

        # Await AGENT_REPLY from the bus
        reply_event: AgentEvent | None = None
        for _ in range(50):
            if not sub.empty():
                evt = await sub.get()
                if evt.type == EventType.AGENT_REPLY:
                    reply_event = evt
                    break
            await asyncio.sleep(0.05)

        assert reply_event is not None, "Agent failed to publish AGENT_REPLY to EventBus"
        assert reply_event.payload.get("turn_index") == 1
        # The reply is the model's answer *after* seeing the tool result, not the text it
        # emitted alongside the tool call (P4, amended 2026-09-05). The assertion is
        # stronger for it: the answer demonstrably contains what the tool returned, which
        # is the property the step loop exists to produce.
        reply_content = str(reply_event.payload.get("content"))
        assert "cpu_util" in reply_content, reply_content
        assert "42%" in reply_content, reply_content

        # Verify tool executions reached the event payload
        from collections.abc import Mapping, Sequence

        tool_execs = reply_event.payload.get("tool_executions")
        assert isinstance(tool_execs, Sequence)
        assert len(tool_execs) == 1
        first_tool = tool_execs[0]
        assert isinstance(first_tool, Mapping)
        assert first_tool.get("tool_name") == "record_metric"
        assert first_tool.get("status") == "success"

        # 4. Verify SessionStore persistence
        agent.persist_session()
        persisted = store.load(agent.session_id)
        assert persisted is not None
        assert persisted.turn_counter == 1
        # Check messages: SYSTEM, USER, ASSISTANT (with tool call), TOOL, and ASSISTANT final
        roles = [m.role for m in persisted.messages]
        assert "system" in roles
        assert "user" in roles
        assert "tool" in roles
        assert "assistant" in roles

    finally:
        await agent.stop()

        await bus.stop()
