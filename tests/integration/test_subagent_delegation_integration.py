from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.llm.models import ToolCallRequest
from uclone_x.tools.builtin.subagent import SubagentDelegationTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_subagent_delegation_bus_events_and_budget():
    bus = EventBus()
    await bus.start()

    registry = ToolRegistry(tools=[SubagentDelegationTool()])

    config = AgentConfig(agent_id="test_parent", name="parent", max_steps=10)
    agent = BaseAgent(config=config, bus=bus, tools=registry)
    agent._turn_counter = 0  # pyright: ignore[reportPrivateUsage]

    # We will mock agent.delegate_task to just increment the subagent's turn_counter by 2 and return success

    async def mock_delegate(subagent: Any, prompt: str) -> Any:
        subagent._turn_counter = 2  # pyright: ignore[reportPrivateUsage]
        res = MagicMock()
        res.error = None
        res.content = "Subagent done"
        res.provenance = Provenance.primary(provider="test", model="test")
        return res

    agent.delegate_task = AsyncMock(side_effect=mock_delegate)

    await agent.start()

    events: list[AgentEvent] = []
    original_publish = bus.publish

    async def capture_publish(event: Any) -> None:
        events.append(event)
        await original_publish(event)

    bus.publish = AsyncMock(side_effect=capture_publish)
    assert agent._publisher is not None  # pyright: ignore[reportPrivateUsage]
    agent._publisher.publish = AsyncMock(side_effect=capture_publish)  # pyright: ignore[reportPrivateUsage]

    tc = ToolCallRequest(
        id="call_1",
        name="delegate_subagent",
        arguments={"role": "helper", "goal": "help", "prompt": "do it", "max_turns": 4},
    )
    ctx = ToolContext(
        agent_id="test_parent", session_id="s1", workspace_root=Path("/tmp"), agent_delegate=agent
    )

    _, rec = await agent._execute_single_tool(tc, ctx)  # pyright: ignore[reportPrivateUsage]

    assert rec.status == "success", f"Failed with {rec.error}"
    # Delegation does not charge the parent's interaction-turn counter for the child's
    # work; the parent spent one step of its own run budget by calling the tool.
    assert agent._turn_counter == 0  # pyright: ignore[reportPrivateUsage]

    spawn_events = [e for e in events if e.type == EventType.SUBAGENT_SPAWN]
    assert len(spawn_events) == 1

    from collections.abc import Mapping

    payload = spawn_events[0].payload
    assert isinstance(payload, (dict, Mapping))
    assert payload["role"] == "helper"
    assert payload["goal"] == "help"

    # Also test recursion cap
    agent._context = agent._context.model_copy(update={"depth": 3})  # pyright: ignore[reportPrivateUsage]
    tc2 = ToolCallRequest(
        id="call_2",
        name="delegate_subagent",
        arguments={"role": "helper", "goal": "help", "prompt": "do it", "max_turns": 4},
    )
    _, rec2 = await agent._execute_single_tool(tc2, ctx)  # pyright: ignore[reportPrivateUsage]
    assert rec2.status == "error"
    assert rec2.error is not None and "Recursion depth exceeded" in rec2.error

    await agent.stop()
    await bus.stop()
