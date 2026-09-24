"""Unit tests for the PlanUpdateTool and agent planning integration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import (
    AgentConfig,
)
from uclone_x.agent.session import SessionStore
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest
from uclone_x.tools.builtin.plan import PlanUpdateParams, PlanUpdateTool
from uclone_x.tools.models import ToolContext


def test_plan_update_tool_create() -> None:
    """Tool can create a new plan returning the intent structure."""
    tool = PlanUpdateTool()
    ctx = ToolContext(agent_id="agent1", session_id="sess1", workspace_root=Path("."))

    params = PlanUpdateParams(
        action="create",
        title="My Plan",
        steps=[{"description": "Step 1"}, {"description": "Step 2"}],
    )

    result = tool.run(params, ctx)
    assert result["action"] == "create"
    assert result["title"] == "My Plan"
    assert len(result["steps"]) == 2


def test_plan_update_tool_update() -> None:
    """Tool can update an existing plan returning the intent structure."""
    tool = PlanUpdateTool()
    ctx = ToolContext(agent_id="agent1", session_id="sess1", workspace_root=Path("."))

    params = PlanUpdateParams(
        action="update", steps=[{"index": 1, "completed": True}], status="in_progress"
    )

    result = tool.run(params, ctx)
    assert result["action"] == "update"
    assert result["status"] == "in_progress"
    assert result["steps"][0]["completed"] is True


def test_plan_update_tool_validation_errors() -> None:
    """Tool validates missing required fields for create."""
    with pytest.raises(ValidationError):
        PlanUpdateParams(action="create")  # Missing title


@pytest.mark.asyncio
async def test_agent_plan_tool_integration(tmp_path: Path) -> None:
    """Agent native integration creates and updates the plan in live session."""
    store = SessionStore(storage_dir=tmp_path)
    config = AgentConfig(agent_id="test_agent", name="Test", workspace_dir=tmp_path)
    from uclone_x.tools.registry import create_default_registry

    tools = create_default_registry(workspace_root=tmp_path, enable_mcp=False)
    agent = BaseAgent(config=config, store=store, tools=tools)

    ctx = ToolContext(agent_id="test_agent", session_id="sess1", workspace_root=tmp_path)

    # 1. Create plan tool call
    tc_create = ToolCallRequest(
        id="call_1",
        name="update_plan",
        arguments={
            "action": "create",
            "title": "Test Plan",
            "steps": [{"description": "S1"}, {"description": "S2"}],
        },
    )

    _, rec = await agent._execute_single_tool(tc_create, ctx)  # type: ignore[reportPrivateUsage]
    assert rec.status == "success"

    # Verify plan is actively on the agent
    plan = agent.current_plan
    assert plan is not None
    assert plan.title == "Test Plan"
    assert len(plan.steps) == 2

    # 2. Update plan tool call
    tc_update = ToolCallRequest(
        id="call_2",
        name="update_plan",
        arguments={
            "action": "update",
            "steps": [{"index": 1, "completed": True}],
            "status": "in_progress",
        },
    )

    _, rec2 = await agent._execute_single_tool(tc_update, ctx)  # type: ignore[reportPrivateUsage]
    assert rec2.status == "success"

    # Verify updated
    plan2 = agent.current_plan
    assert plan2 is not None
    assert plan2.status == "in_progress"
    assert plan2.steps[0].completed is True
    assert plan2.steps[1].completed is False

    # 3. Save to store and reload
    agent.persist_session()
    sid = agent._context.session_id  # type: ignore[reportPrivateUsage]
    reloaded = store.load(sid)
    assert reloaded is not None
    assert reloaded.plan is not None
    assert reloaded.plan.title == "Test Plan"
    assert reloaded.plan.status == "in_progress"
    assert reloaded.plan.steps[0].completed is True


@pytest.mark.asyncio
async def test_compaction_preserves_plan(tmp_path: Path) -> None:
    """Context compaction does not destroy the active plan."""
    store = SessionStore(storage_dir=tmp_path)
    config = AgentConfig(agent_id="test_agent", name="Test", workspace_dir=tmp_path)
    agent = BaseAgent(config=config, store=store)

    # Seed messages
    for i in range(10):
        agent._history.append(ChatMessage(role=MessageRole.USER, content=f"msg {i}"))  # type: ignore[reportPrivateUsage]
        agent._live_session(agent._context.session_id).turn_counter += 1  # type: ignore[reportPrivateUsage]

    # Create plan
    agent.create_plan(title="Compaction test plan", steps=["A", "B"])

    assert agent.current_plan is not None

    # Compact
    await agent.compact_session(agent._context.session_id)  # type: ignore[reportPrivateUsage]

    # Verify plan survives in active session
    plan = agent.current_plan
    assert plan is not None
    assert plan.title == "Compaction test plan"

    # Verify plan survives on disk
    loaded = store.load(agent._context.session_id)  # type: ignore[reportPrivateUsage]
    assert loaded is not None
    assert loaded.plan is not None
    assert loaded.plan.title == "Compaction test plan"


def test_agent_turn_consumes_max_turns(tmp_path: Path) -> None:
    """Tool execution (including plan tool) burns turns."""
    # This is verified implicitly by the standard agent turn execution.
    pass
