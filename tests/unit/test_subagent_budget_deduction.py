"""Regression tests for subagent step budget deduction and sibling pooling (P4, Issue #555)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentState
from uclone_x.core.provenance import Provenance
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import FinishReason, LLMRequest, ModelResponse, TokenUsage, ToolCallRequest
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.subagent import SubagentDelegationTool
from uclone_x.tools.models import IsolationLevel, ToolContext, ToolResult
from uclone_x.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_child_steps_deducted_from_parent_budget_on_tool_execute() -> None:
    """Child steps spent during delegation are charged to parent run steps (P4).

    When a subagent completes, its consumed steps are deducted from the parent's
    remaining budget, incrementing parent._run_steps and decreasing steps_remaining.

    Killed by: src/uclone_x/tools/builtin/subagent.py :: agent.consume_steps(child_steps)
    Becomes: agent.consume_steps(0)
    """
    config = AgentConfig(agent_id="parent_agent", name="parent", role="parent", max_steps=10)
    agent = BaseAgent(config=config)
    agent._run_steps = 0  # pyright: ignore[reportPrivateUsage]
    agent._turn_counter = 0  # pyright: ignore[reportPrivateUsage]

    subagent_mock = AsyncMock()
    subagent_mock.agent_id = "sub_1"
    subagent_mock.turn_counter = 4
    subagent_mock._config = AgentConfig(agent_id="sub_1", name="sub", role="sub")  # pyright: ignore[reportPrivateUsage]
    agent.spawn_subagent = AsyncMock(return_value=subagent_mock)

    turn_result = MagicMock()
    turn_result.error = None
    turn_result.content = "Child finished"
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
        params={"role": "researcher", "goal": "Find X", "prompt": "Search X", "max_steps": 5},
        context=tool_ctx,
    )

    assert result.success is True
    # The parent's run budget must be charged the 4 steps consumed by the child
    assert agent.run_steps == 4
    assert agent.steps_remaining == 6
    # The parent's human interaction turn counter is NOT charged for child execution
    assert agent.turn_counter == 0


@pytest.mark.asyncio
async def test_sibling_subagents_share_parent_budget_ceiling() -> None:
    """Sibling delegations jointly share the parent step ceiling per P4.

    Child 1 consumes 4 steps -> parent remaining is 6.
    Child 2 requests 10 steps -> capped at launch to 6.
    Child 2 consumes 3 steps -> parent remaining is 3.
    Child 3 requests 5 steps -> capped at launch to 3.
    Child 3 consumes 3 steps -> parent remaining is 0.
    Child 4 attempts delegation -> immediately fails with exhausted budget.

    Killed by: src/uclone_x/tools/builtin/subagent.py :: if sub_max_steps is None or sub_max_steps > parent_remaining_steps:
    """
    config = AgentConfig(agent_id="parent_agent", name="parent", role="parent", max_steps=10)
    agent = BaseAgent(config=config)
    agent._run_steps = 0  # pyright: ignore[reportPrivateUsage]

    tool = SubagentDelegationTool()
    tool_ctx = ToolContext(
        agent_id="parent_agent",
        session_id="sess_1",
        trace_id="trace_1",
        workspace_root=Path("/tmp"),
        agent_delegate=agent,
    )

    # --- Sibling 1 ---
    sub1 = AsyncMock()
    sub1.agent_id = "sub_1"
    sub1.turn_counter = 4
    sub1._config = AgentConfig(agent_id="sub_1", name="sub1", role="sub")  # pyright: ignore[reportPrivateUsage]
    agent.spawn_subagent = AsyncMock(return_value=sub1)

    turn_res1 = MagicMock()
    turn_res1.error = None
    turn_res1.content = "Child 1 done"
    turn_res1.provenance = Provenance.primary(provider="test", model="test")
    agent.delegate_task = AsyncMock(return_value=turn_res1)

    res1 = await tool.execute(
        params={"role": "researcher", "goal": "Task 1", "prompt": "P1", "max_steps": 5},
        context=tool_ctx,
    )
    assert res1.success is True
    assert agent.run_steps == 4
    assert agent.steps_remaining == 6

    # --- Sibling 2: requests 10 steps, capped at parent.steps_remaining (6) ---
    sub2 = AsyncMock()
    sub2.agent_id = "sub_2"
    sub2.turn_counter = 3
    sub2._config = AgentConfig(agent_id="sub_2", name="sub2", role="sub")  # pyright: ignore[reportPrivateUsage]
    agent.spawn_subagent = AsyncMock(return_value=sub2)

    turn_res2 = MagicMock()
    turn_res2.error = None
    turn_res2.content = "Child 2 done"
    turn_res2.provenance = Provenance.primary(provider="test", model="test")
    agent.delegate_task = AsyncMock(return_value=turn_res2)

    res2 = await tool.execute(
        params={"role": "analyst", "goal": "Task 2", "prompt": "P2", "max_steps": 10},
        context=tool_ctx,
    )
    assert res2.success is True
    # Capped at launch to parent.steps_remaining (6)
    assert sub2._config.max_steps == 6  # pyright: ignore[reportPrivateUsage]
    assert agent.run_steps == 7
    assert agent.steps_remaining == 3

    # --- Sibling 3: requests 5 steps, capped at parent.steps_remaining (3) ---
    sub3 = AsyncMock()
    sub3.agent_id = "sub_3"
    sub3.turn_counter = 3
    sub3._config = AgentConfig(agent_id="sub_3", name="sub3", role="sub")  # pyright: ignore[reportPrivateUsage]
    agent.spawn_subagent = AsyncMock(return_value=sub3)

    turn_res3 = MagicMock()
    turn_res3.error = None
    turn_res3.content = "Child 3 done"
    turn_res3.provenance = Provenance.primary(provider="test", model="test")
    agent.delegate_task = AsyncMock(return_value=turn_res3)

    res3 = await tool.execute(
        params={"role": "coder", "goal": "Task 3", "prompt": "P3", "max_steps": 5},
        context=tool_ctx,
    )
    assert res3.success is True
    assert sub3._config.max_steps == 3  # pyright: ignore[reportPrivateUsage]
    assert agent.run_steps == 10
    assert agent.steps_remaining == 0

    # --- Sibling 4: attempts delegation when parent budget is 0 ---
    res4 = await tool.execute(
        params={"role": "tester", "goal": "Task 4", "prompt": "P4", "max_steps": 2},
        context=tool_ctx,
    )
    assert res4.success is False
    assert res4.error is not None and "exhausted its step budget" in res4.error


@pytest.mark.asyncio
async def test_direct_delegate_task_deducts_child_steps_from_parent() -> None:
    """Calling agent.delegate_task directly charges child steps to parent budget.

    Killed by: src/uclone_x/agent/base.py :: self.consume_steps(child_steps)
    """
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=ModelResponse(
            finish_reason=FinishReason.STOP,
            content="Child completed turn",
            usage=TokenUsage(provider="mock", input_tokens=5, output_tokens=5, total_tokens=10),
            provenance=Provenance.primary("mock"),
        )
    )

    parent_config = AgentConfig(agent_id="lead", name="Lead", max_steps=10)
    parent = BaseAgent(config=parent_config, llm=mock_llm)
    await parent.start()

    worker = await parent.spawn_subagent(role="worker", goal="Work")
    result = await parent.delegate_task(worker, "Do task")

    assert result.is_completed is True
    assert result.content == "Child completed turn"
    # Worker executed 1 step in execute_turn, which was deducted from parent
    assert parent.run_steps == 1
    assert parent.steps_remaining == 9

    await parent.stop()


def test_agent_consume_steps_method_bounds() -> None:
    """agent.consume_steps increments _run_steps and reduces steps_remaining.

    Killed by: src/uclone_x/agent/base.py :: self._run_steps += count
    """
    config = AgentConfig(agent_id="test_agent", name="test", max_steps=10)
    agent = BaseAgent(config=config)
    assert agent.run_steps == 0
    assert agent.steps_remaining == 10

    # Positive count consumes steps
    agent.consume_steps(4)
    assert agent.run_steps == 4
    assert agent.steps_remaining == 6

    # Non-positive counts are no-ops
    agent.consume_steps(0)
    assert agent.run_steps == 4
    agent.consume_steps(-3)
    assert agent.run_steps == 4

    # Consuming past max_steps bottoms steps_remaining out at zero
    agent.consume_steps(10)
    assert agent.run_steps == 14
    assert agent.steps_remaining == 0


@pytest.mark.asyncio
async def test_execute_turn_loop_synchronizes_with_deducted_steps() -> None:
    """The execute_turn agentic loop synchronizes step with _run_steps.

    When tools consume child steps during a step, the next loop iteration
    accounts for those consumed steps so the session ceiling cannot be breached.

    Killed by: src/uclone_x/agent/base.py :: step = self._run_steps + 1
    """
    from pydantic import BaseModel

    class DummyParams(BaseModel):
        pass

    class StepConsumingTool(BaseTool[DummyParams]):
        name = "step_consumer"
        description = "Consumes steps from agent delegate"
        params_type = DummyParams

        async def execute(
            self,
            params: dict[str, Any] | ToolContext | None = None,
            context: ToolContext | None = None,
            **kwargs: Any,
        ) -> ToolResult:
            actual_context = params if isinstance(params, ToolContext) else context
            if actual_context and actual_context.agent_delegate:
                actual_context.agent_delegate.consume_steps(4)
            return ToolResult(
                success=True,
                output={"consumed": 4},
                isolation_level=IsolationLevel.WORKSPACE,
                provenance=Provenance.primary("test", "test"),
            )

        def run(self, params: Any, context: ToolContext) -> Any:
            pass

    reg = ToolRegistry()
    reg.register(StepConsumingTool())

    tc = ToolCallRequest(id="tc_1", name="step_consumer", arguments={})

    class NeverSatisfiedConnector(MockLLMConnector):
        async def generate(self, request: LLMRequest) -> ModelResponse:
            return ModelResponse(
                content="calling again",
                tool_calls=(tc,),
                usage=TokenUsage(
                    provider="mock", model="mock-model", input_tokens=1, output_tokens=1
                ),
                finish_reason=FinishReason.TOOL_CALLS,
                model_name="mock-model",
                provenance=Provenance.primary("mock", "mock-model"),
            )

    # Parent budget is 5 steps
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="loop_agent",
            name="LoopAgent",
            max_steps=5,
        ),
        llm=NeverSatisfiedConnector(),
        tools=reg,
    )

    res = await agent.execute_turn("trigger loop")
    assert res.is_completed is False
    assert res.error == "Agent step budget exceeded: maximum 5 steps in a single request"
    assert agent.state == AgentState.ERROR
    # 1 parent step taken + 4 steps consumed by tool = 5 total steps taken
    assert agent.run_steps == 5
    assert agent.steps_remaining == 0


@pytest.mark.asyncio
async def test_subagent_exception_still_charges_consumed_steps() -> None:
    """If a subagent delegation raises an exception, steps taken are still charged.

    Killed by: src/uclone_x/tools/builtin/subagent.py :: val > steps
    Becomes: val < steps
    """
    config = AgentConfig(agent_id="parent_agent", name="parent", role="parent", max_steps=10)
    agent = BaseAgent(config=config)
    agent._run_steps = 0  # pyright: ignore[reportPrivateUsage]

    subagent_mock = AsyncMock()
    subagent_mock.agent_id = "sub_crash"
    subagent_mock.turn_counter = 3
    subagent_mock._config = AgentConfig(agent_id="sub_crash", name="sub", role="sub")  # pyright: ignore[reportPrivateUsage]
    agent.spawn_subagent = AsyncMock(return_value=subagent_mock)
    agent.delegate_task = AsyncMock(side_effect=RuntimeError("Subagent crashed midway"))

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
    assert result.error is not None and "Subagent crashed midway" in result.error
    # Even on crash, the 3 steps taken prior to crash are deducted
    assert agent.run_steps == 3
    assert agent.steps_remaining == 7
