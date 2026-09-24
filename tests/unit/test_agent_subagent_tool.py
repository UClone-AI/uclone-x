from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig
from uclone_x.core.provenance import Provenance
from uclone_x.tools.builtin.subagent import SubagentDelegationTool
from uclone_x.tools.models import ToolContext


@pytest.mark.asyncio
async def test_subagent_delegation_callable():
    # Setup mock agent
    config = AgentConfig(agent_id="parent_agent", name="parent", role="parent", max_turns=10)
    agent = BaseAgent(config=config)
    agent._turn_counter = 0  # pyright: ignore[reportPrivateUsage]

    subagent_mock = AsyncMock()
    subagent_mock.agent_id = "sub_1"
    subagent_mock.turn_counter = 3
    subagent_mock._config = AgentConfig(agent_id="sub_1", name="sub", role="sub")  # pyright: ignore[reportPrivateUsage]

    agent.spawn_subagent = AsyncMock(return_value=subagent_mock)

    turn_result = MagicMock()
    turn_result.error = None
    turn_result.content = "Success result"
    turn_result.provenance = Provenance.primary(provider="test", model="test")
    agent.delegate_task = AsyncMock(return_value=turn_result)

    tool = SubagentDelegationTool()
    tool_ctx = ToolContext(
        agent_id="parent_agent",
        session_id="sess_1",
        trace_id="trace_1",
        workspace_root=Path("/tmp"),
        agent_delegate=agent,
    )

    result = await tool.execute(
        params={"role": "researcher", "goal": "Find X", "prompt": "Search X", "max_turns": 5},
        context=tool_ctx,
    )

    assert result.success is True
    assert (
        result.output is not None
        and isinstance(result.output, dict)
        and result.output.get("response") == "Success result"
    )
    # The parent's interaction-turn counter is NOT charged for the subagent's work.
    # (The previous assertion here read `assert agent._turn_counter  # ... == 3`, with the
    # comparison inside the pyright comment, so it asserted truthiness of a counter the
    # code had just inflated — it would have passed for any non-zero value.)
    assert agent._turn_counter == 0  # pyright: ignore[reportPrivateUsage]
    # Child steps consumed are charged to the parent run budget (P4)
    assert agent.run_steps == 3
    assert agent.steps_remaining == 7

    # The child's step ceiling is the one the caller asked for.
    assert subagent_mock._config.max_steps == 5  # pyright: ignore[reportPrivateUsage]
    assert subagent_mock._config.max_turns == 5  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_subagent_recursion_depth_capped():
    config = AgentConfig(agent_id="parent_agent", name="parent", role="parent", max_turns=10)
    agent = BaseAgent(config=config)
    agent._context = agent._context.model_copy(update={"depth": 3})  # pyright: ignore[reportPrivateUsage]

    tool = SubagentDelegationTool()
    tool_ctx = ToolContext(
        agent_id="parent_agent",
        session_id="sess_1",
        trace_id="trace_1",
        workspace_root=Path("/tmp"),
        agent_delegate=agent,
    )

    result = await tool.execute(
        params={"role": "researcher", "goal": "Find X", "prompt": "Search X"},
        context=tool_ctx,
    )

    assert result.success is False
    assert result.error is not None and "Recursion depth exceeded" in result.error


@pytest.mark.asyncio
async def test_subagent_budget_inherited():
    config = AgentConfig(agent_id="parent_agent", name="parent", role="parent", max_steps=10)
    agent = BaseAgent(config=config)
    # Spent the run's step budget. `turn_counter` is deliberately left high as well, to
    # prove the refusal keys off steps rather than off conversation length.
    agent._run_steps = 10  # pyright: ignore[reportPrivateUsage]
    agent._turn_counter = 99  # pyright: ignore[reportPrivateUsage]

    tool = SubagentDelegationTool()
    tool_ctx = ToolContext(
        agent_id="parent_agent",
        session_id="sess_1",
        trace_id="trace_1",
        workspace_root=Path("/tmp"),
        agent_delegate=agent,
    )

    result = await tool.execute(
        params={"role": "researcher", "goal": "Find X", "prompt": "Search X"},
        context=tool_ctx,
    )

    assert result.success is False
    assert result.error is not None and "exhausted its step budget" in result.error


@pytest.mark.asyncio
async def test_subagent_failure_surfaced():
    config = AgentConfig(agent_id="parent_agent", name="parent", role="parent", max_turns=10)
    agent = BaseAgent(config=config)
    agent._turn_counter = 0  # pyright: ignore[reportPrivateUsage]

    subagent_mock = AsyncMock()
    subagent_mock.agent_id = "sub_1"
    subagent_mock.turn_counter = 1
    subagent_mock._config = AgentConfig(agent_id="sub_1", name="sub", role="sub")  # pyright: ignore[reportPrivateUsage]

    agent.spawn_subagent = AsyncMock(return_value=subagent_mock)

    turn_result = MagicMock()
    turn_result.error = "Agent error"
    turn_result.content = "Failed"
    turn_result.provenance = Provenance.primary(provider="test", model="test")
    agent.delegate_task = AsyncMock(return_value=turn_result)

    tool = SubagentDelegationTool()
    tool_ctx = ToolContext(
        agent_id="parent_agent",
        session_id="sess_1",
        trace_id="trace_1",
        workspace_root=Path("/tmp"),
        agent_delegate=agent,
    )

    result = await tool.execute(
        params={"role": "researcher", "goal": "Find X", "prompt": "Search X"},
        context=tool_ctx,
    )

    assert result.success is False
    assert result.error is not None and "Subagent failed: Agent error" in result.error


@pytest.mark.asyncio
async def test_long_conversation_does_not_exhaust_the_delegation_budget():
    """A long conversation must not block delegation (issue 2026-09-05-001).

    The budget was `config.max_turns - agent.turn_counter`: the per-run step ceiling minus
    the lifetime interaction-turn counter. Once a session had exchanged `max_steps`
    messages, `turn_counter` alone pushed the difference to zero and every later
    delegation failed with "exhausted", no matter how few steps the current run had spent.

    Killed by: src/uclone_x/tools/builtin/subagent.py :: parent_remaining_steps = agent.steps_remaining
    """
    config = AgentConfig(agent_id="parent_agent", name="parent", role="parent", max_steps=10)
    agent = BaseAgent(config=config)
    # A perfectly ordinary long conversation, at the very start of a fresh run.
    agent._turn_counter = 250  # pyright: ignore[reportPrivateUsage]
    agent._run_steps = 0  # pyright: ignore[reportPrivateUsage]

    subagent_mock = AsyncMock()
    subagent_mock.agent_id = "sub_1"
    subagent_mock.turn_counter = 2
    subagent_mock._config = AgentConfig(agent_id="sub_1", name="sub", role="sub")  # pyright: ignore[reportPrivateUsage]
    agent.spawn_subagent = AsyncMock(return_value=subagent_mock)

    turn_result = MagicMock()
    turn_result.error = None
    turn_result.content = "Delegated fine"
    turn_result.provenance = Provenance.primary(provider="test", model="test")
    agent.delegate_task = AsyncMock(return_value=turn_result)

    tool = SubagentDelegationTool()
    tool_ctx = ToolContext(
        agent_id="parent_agent",
        session_id="sess_1",
        trace_id="trace_1",
        workspace_root=Path("/tmp"),
        agent_delegate=agent,
    )

    result = await tool.execute(
        params={"role": "researcher", "goal": "Find X", "prompt": "Search X"},
        context=tool_ctx,
    )

    assert result.success is True, result.error
    # The child inherits the parent's remaining steps, not `max_steps - turn_counter`.
    assert subagent_mock._config.max_steps == 10  # pyright: ignore[reportPrivateUsage]
    # And the conversation counter is left exactly where the conversation put it.
    assert agent._turn_counter == 250  # pyright: ignore[reportPrivateUsage]
    # While child steps consumed are charged to the parent run steps
    assert agent.run_steps == 2
    assert agent.steps_remaining == 8


@pytest.mark.asyncio
async def test_delegation_params_accept_the_deprecated_max_turns_spelling():
    """A model that learned the old tool schema still lands on the canonical field."""
    from uclone_x.tools.builtin.subagent import SubagentDelegationParams

    legacy = SubagentDelegationParams(role="r", goal="g", prompt="p", max_turns=7)
    assert legacy.max_steps == 7
    assert legacy.max_turns == 7

    canonical = SubagentDelegationParams(role="r", goal="g", prompt="p", max_steps=7)
    assert canonical.max_steps == 7
    assert canonical.max_turns == 7
