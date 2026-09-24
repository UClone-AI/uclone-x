"""Unit tests for parallel tool execution in BaseAgent (Issue #185, P3)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from tests.conftest import finish_after_tools
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.models import FinishReason, ModelResponse, TokenUsage, ToolCallRequest
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry


class SlowParams(BaseModel):
    delay: float = Field(default=0.05, description="Sleep delay")
    tag: str = Field(default="", description="Tag")


class SlowAsyncTool(BaseTool[SlowParams]):
    """Tool that sleeps to simulate async I/O."""

    name = "slow_tool"
    description = "Sleeps for a duration"

    async def run(self, params: SlowParams, context: ToolContext) -> str:
        await asyncio.sleep(params.delay)
        return f"Done {params.tag}"


class FailParams(BaseModel):
    pass


class FailingTool(BaseTool[FailParams]):
    """Tool that raises an error."""

    name = "failing_tool"
    description = "Always fails"

    def run(self, params: FailParams, context: ToolContext) -> str:
        raise ValueError("Simulated tool crash")


@pytest.mark.asyncio
async def test_parallel_tool_execution_benchmark_speedup(tmp_path: Path) -> None:
    """Verify that multiple tool calls execute concurrently via asyncio.gather."""
    reg = ToolRegistry()
    slow_tool = SlowAsyncTool()
    reg.register(slow_tool)

    cfg = AgentConfig(
        agent_id="test_parallel",
        name="Parallel Test",
        workspace_dir=str(tmp_path),
        llm_config=AgentLLMConfig(model_name="mock"),
    )

    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="test", model="mock"),
        served_by=ServiceRef(provider="test", model="mock"),
    )

    llm = MagicMock(spec=LLMProviderProtocol)
    # Return 3 tool calls, each sleeping 0.06s
    llm.generate = AsyncMock(
        return_value=ModelResponse(
            finish_reason=FinishReason.STOP,
            content="Tools called",
            tool_calls=(
                ToolCallRequest(
                    id="call_1", name="slow_tool", arguments={"delay": 0.06, "tag": "A"}
                ),
                ToolCallRequest(
                    id="call_2", name="slow_tool", arguments={"delay": 0.06, "tag": "B"}
                ),
                ToolCallRequest(
                    id="call_3", name="slow_tool", arguments={"delay": 0.06, "tag": "C"}
                ),
            ),
            usage=TokenUsage(
                provider="test",
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
            ),
            provenance=prov,
        )
    )

    finish_after_tools(llm)

    agent = BaseAgent(config=cfg, llm=llm, tools=reg)

    t0 = asyncio.get_running_loop().time()
    res = await agent.execute_turn("Run tools")
    total_time = asyncio.get_running_loop().time() - t0

    # If sequential, 3 * 0.06s = 0.18s+. If parallel, ~0.06s-0.12s.
    assert total_time < 0.16, f"Expected parallel execution < 0.16s, got {total_time:.3f}s"
    assert len(res.tool_executions) == 3

    # Assert deterministic ordering matching input tool_calls sequence
    assert res.tool_executions[0].tool_call_id == "call_1"
    assert res.tool_executions[0].output == "Done A"
    assert res.tool_executions[1].tool_call_id == "call_2"
    assert res.tool_executions[1].output == "Done B"
    assert res.tool_executions[2].tool_call_id == "call_3"
    assert res.tool_executions[2].output == "Done C"


@pytest.mark.asyncio
async def test_parallel_tool_execution_sibling_isolation_on_failure(tmp_path: Path) -> None:
    """Verify that an exception in one tool does not cancel sibling parallel tools."""
    reg = ToolRegistry()
    reg.register(SlowAsyncTool())
    reg.register(FailingTool())

    cfg = AgentConfig(
        agent_id="test_iso",
        name="Iso Test",
        workspace_dir=str(tmp_path),
        llm_config=AgentLLMConfig(model_name="mock"),
    )

    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="test", model="mock"),
        served_by=ServiceRef(provider="test", model="mock"),
    )

    llm = MagicMock(spec=LLMProviderProtocol)
    llm.generate = AsyncMock(
        return_value=ModelResponse(
            finish_reason=FinishReason.STOP,
            content="Mixed tools",
            tool_calls=(
                ToolCallRequest(
                    id="call_ok_1", name="slow_tool", arguments={"delay": 0.02, "tag": "1"}
                ),
                ToolCallRequest(id="call_fail", name="failing_tool", arguments={}),
                ToolCallRequest(
                    id="call_ok_2", name="slow_tool", arguments={"delay": 0.02, "tag": "2"}
                ),
            ),
            usage=TokenUsage(
                provider="test",
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
            ),
            provenance=prov,
        )
    )

    finish_after_tools(llm)

    agent = BaseAgent(config=cfg, llm=llm, tools=reg)
    res = await agent.execute_turn("Run mixed")

    assert len(res.tool_executions) == 3
    assert res.tool_executions[0].status == "success"
    assert res.tool_executions[0].output == "Done 1"
    assert res.tool_executions[1].status == "error"
    assert "Simulated tool crash" in str(res.tool_executions[1].error)
    assert res.tool_executions[2].status == "success"
    assert res.tool_executions[2].output == "Done 2"
