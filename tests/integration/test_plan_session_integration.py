"""Integration tests for planning and session persistence."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, TurnResult
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest
from uclone_x.tools.registry import create_default_registry


@pytest.mark.asyncio
async def test_full_round_trip_plan_session_integration(tmp_path: Path) -> None:
    """Full round trip: agent turn invokes plan tool, persists session, reloads session,
    verifies plan state in new agent session, runs compaction, verifies plan survives compaction."""

    store = SessionStore(storage_dir=tmp_path)
    config = AgentConfig(agent_id="test_integration", name="Integration", workspace_dir=tmp_path)
    tools = create_default_registry(workspace_root=tmp_path, enable_mcp=False)

    # Mock LLM to return a tool call for create plan
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = TurnResult(
        turn_index=1,
        content="I will create a plan.",
        tool_calls=(
            ToolCallRequest(
                id="call_plan",
                name="update_plan",
                arguments={
                    "action": "create",
                    "title": "Integration Plan",
                    "steps": [{"description": "Init"}, {"description": "Done"}],
                },
            ),
        ),
        provenance=Provenance.primary("test", "test"),
    )

    agent = BaseAgent(config=config, store=store, tools=tools, llm=mock_llm)
    await agent.start()

    # Execute turn which should invoke the tool
    await agent.execute_turn("Create a plan.")

    # Verify plan is created in live session
    assert agent.current_plan is not None
    assert agent.current_plan.title == "Integration Plan"

    # Persist session
    agent.persist_session()

    # Reload session in a NEW agent instance
    agent2 = BaseAgent(config=config, store=store, tools=tools)
    agent2.hydrate_session(agent._context.session_id)  # type: ignore[reportPrivateUsage]

    assert agent2.current_plan is not None
    assert agent2.current_plan.title == "Integration Plan"
    assert len(agent2.current_plan.steps) == 2

    # Add messages to force compaction
    for i in range(15):
        agent2._history.append(ChatMessage(role=MessageRole.USER, content=f"msg {i}"))  # type: ignore[reportPrivateUsage]

    await agent2.compact_session()

    # Verify plan survives compaction in memory and on disk
    assert agent2.current_plan is not None
    assert agent2.current_plan.title == "Integration Plan"

    reloaded_store = store.load(agent._context.session_id)  # type: ignore[reportPrivateUsage]
    assert reloaded_store is not None
    assert reloaded_store.plan is not None
    assert reloaded_store.plan.title == "Integration Plan"
