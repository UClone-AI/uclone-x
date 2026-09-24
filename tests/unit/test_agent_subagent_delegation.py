# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
"""Unit tests for dynamic subagent spawning and in-memory A2A delegation in BaseAgent (Issue #56)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import EventBus, EventType
from uclone_x.llm.models import FinishReason, ModelResponse, TokenUsage


@pytest.mark.asyncio
async def test_spawn_subagent_context_isolation() -> None:
    bus = EventBus()
    parent_config = AgentConfig(
        agent_id="parent_agent",
        name="Parent Agent",
        system_prompt="Parent system instruction",
    )
    parent = BaseAgent(config=parent_config, bus=bus)
    await parent.start()

    sub = await parent.spawn_subagent(
        role="researcher",
        goal="Analyze repo structure",
    )

    assert sub.agent_id.startswith("parent_agent_sub_")
    assert sub.context.parent_agent_id == "parent_agent"
    assert sub.context.depth == 1
    assert sub.context.session_id != parent.context.session_id
    assert "researcher" in (sub.config.system_prompt or "")
    assert len(sub._history) == 1
    assert "researcher" in (sub._history[0].content or "")

    await parent.stop()


@pytest.mark.asyncio
async def test_subagent_task_delegation_and_events() -> None:
    bus = EventBus()
    sub = bus.subscribe("swarm.subagent.*")

    # A connector is wired because delegation runs a real turn on the sub-agent. This
    # test used to run the parent (and so the inherited sub-agent) with `llm=None` and
    # assert `"test_foo" in result.content` — which passed only because `execute_turn`
    # echoed the prompt back. It asserted the defect of issue #136a, not delegation.
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=ModelResponse(
            finish_reason=FinishReason.STOP,
            content="Wrote test_foo",
            usage=TokenUsage(provider="mock", input_tokens=5, output_tokens=5, total_tokens=10),
            provenance=Provenance.primary("mock"),
        )
    )

    parent_config = AgentConfig(
        agent_id="lead_agent",
        name="Lead Agent",
        system_prompt="Lead coordinator",
    )
    parent = BaseAgent(config=parent_config, bus=bus, llm=mock_llm)
    await parent.start()

    worker = await parent.spawn_subagent(
        role="coder",
        goal="Write unit test",
    )

    result = await parent.delegate_task(worker, "Please write test_foo")
    assert result.is_completed is True
    assert result.content == "Wrote test_foo"

    # Verify SUBAGENT_SPAWN and SUBAGENT_DONE events received on EventBus
    event1 = await asyncio.wait_for(sub.get(), timeout=1.0)
    event2 = await asyncio.wait_for(sub.get(), timeout=1.0)
    event_types = {event1.type, event2.type}
    assert EventType.SUBAGENT_SPAWN in event_types
    assert EventType.SUBAGENT_DONE in event_types

    sub.close()
    await parent.stop()


@pytest.mark.asyncio
async def test_concurrent_subagent_delegation_with_llm() -> None:
    bus = EventBus()
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        side_effect=[
            ModelResponse(
                finish_reason=FinishReason.STOP,
                content="Result from worker 1",
                usage=TokenUsage(provider="mock", input_tokens=5, output_tokens=5, total_tokens=10),
                provenance=Provenance.primary("mock"),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                content="Result from worker 2",
                usage=TokenUsage(provider="mock", input_tokens=5, output_tokens=5, total_tokens=10),
                provenance=Provenance.primary("mock"),
            ),
        ]
    )

    parent_config = AgentConfig(
        agent_id="manager",
        name="Manager Agent",
    )
    parent = BaseAgent(config=parent_config, bus=bus, llm=mock_llm)
    await parent.start()

    worker1 = await parent.spawn_subagent(role="worker1", goal="Goal 1")
    worker2 = await parent.spawn_subagent(role="worker2", goal="Goal 2")

    # Delegate concurrently
    results = await asyncio.gather(
        parent.delegate_task(worker1, "Task 1"),
        parent.delegate_task(worker2, "Task 2"),
    )

    assert len(results) == 2
    assert results[0].content == "Result from worker 1"
    assert results[1].content == "Result from worker 2"
    assert results[0].is_completed is True
    assert results[1].is_completed is True

    await parent.stop()


@pytest.mark.asyncio
async def test_p4_bounded_execution_and_config() -> None:
    """Verify Principle 4 bounded execution limits: max_turns, max_subagent_depth, and depth progression."""
    config = AgentConfig(
        agent_id="p4_lead",
        name="P4 Lead",
        max_turns=10,
        max_subagent_depth=2,
        max_concurrent_subagents=3,
    )
    assert config.max_turns == 10
    assert config.max_subagent_depth == 2
    assert config.max_concurrent_subagents == 3

    bus = EventBus()
    agent = BaseAgent(config=config, bus=bus)
    await agent.start()

    # Spawn child at depth 1
    child = await agent.spawn_subagent(role="child_worker", goal="Task 1")
    assert child.context.depth == 1
    assert child.context.parent_agent_id == "p4_lead"

    # Spawn grandchild at depth 2
    grandchild = await child.spawn_subagent(role="grandchild_worker", goal="Task 2")
    assert grandchild.context.depth == 2
    assert grandchild.context.parent_agent_id == child.agent_id

    await agent.stop()
