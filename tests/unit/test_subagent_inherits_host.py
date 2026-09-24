"""A sub-agent is built from its parent's host, so the parent's enforcement binds it (#1449).

`spawn_subagent` used to construct the child with eight of the parent's dependencies and
none of what `HostDependencies` carries besides, so a `PRE_TOOL_USE` hook -- where a human
approval step runs -- and the token budget stopped at the delegation boundary.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import (
    SUBAGENT_EXCLUDED_HOST_FIELDS,
    HostDependencies,
    compose_agent,
)
from uclone_x.agent.hooks import BaseHook, HookAction, HookContext, HookDecision
from uclone_x.agent.models import AgentConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import TokenUsage, ToolCallRequest
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools import LocalTool, ToolRegistry
from uclone_x.tools.models import ToolContext, ToolResult


class _CountingTool(LocalTool):
    def __init__(self) -> None:
        super().__init__(name="guarded_tool", description="A tool a hook refuses")
        self.runs = 0

    async def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        self.runs += 1
        return ToolResult(success=True, output="ran", provenance=Provenance.primary("test"))


class _RefuseGuardedTool(BaseHook):
    """Refuses `guarded_tool` and records which agent asked."""

    def __init__(self) -> None:
        super().__init__(name="refuse_guarded_tool")
        self.refused_for: list[str] = []

    async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
        if context.payload.get("tool_name") == "guarded_tool":
            self.refused_for.append(context.agent_id)
            return HookDecision(action=HookAction.BLOCK, reason="not approved")
        return HookDecision(action=HookAction.ALLOW)


def _parent(
    tmp_path: Path,
    llm: MockLLMConnector,
    tools: ToolRegistry,
    *,
    hooks: list[BaseHook] | None = None,
    budget: TokenBudgetManager | None = None,
) -> BaseAgent:
    host = HostDependencies(
        bus=EventBus(),
        llm=llm,
        tools=tools,
        tracer=TelemetryTracer(),
        store=SessionStore(storage_dir=tmp_path),
        hooks=hooks,
        budget=budget,
    )
    config = AgentConfig(agent_id="lead", name="lead", enable_subagent_tools=True)
    return compose_agent(config=config, host=host)


@pytest.mark.asyncio
async def test_a_hook_on_the_parent_refuses_the_same_tool_call_made_by_its_child(
    tmp_path: Path,
) -> None:
    """The child's call reaches the parent's hook, and the tool never runs.

    Killed by: src/uclone_x/agent/base.py :: "hooks": self._hook_runner.hooks,
    Becomes: "hooks": None,
    """
    tool = _CountingTool()
    tools = ToolRegistry()
    tools.register(tool)
    llm = MockLLMConnector(
        responses=["", "done"],
        tool_calls=[ToolCallRequest(id="c1", name="guarded_tool", arguments={})],
    )
    hook = _RefuseGuardedTool()
    parent = _parent(tmp_path, llm, tools, hooks=[hook])

    child = await parent.spawn_subagent(role="helper", goal="help")
    result = await child.execute_turn("use the tool")

    assert [te.status for te in result.tool_executions] == ["error"]
    assert tool.runs == 0
    assert hook.refused_for == [child.agent_id]


@pytest.mark.asyncio
async def test_a_child_s_token_spend_is_booked_on_the_parent_s_session(tmp_path: Path) -> None:
    """Not only on the parent's manager: on the parent's *session*, whose ceiling it is.

    Killed by: src/uclone_x/agent/base.py :: "budget": _ParentSessionBudget(self._budget, self._context.session_id)
    Becomes: "budget": None
    Killed by: src/uclone_x/agent/base.py :: self._parent.record_usage(self._session_id, usage)
    Becomes: self._parent.record_usage(session_id, usage)
    """
    budget = TokenBudgetManager()
    parent = _parent(tmp_path, MockLLMConnector(responses=["done"]), ToolRegistry(), budget=budget)

    child = await parent.spawn_subagent(role="helper", goal="help")
    await child.execute_turn("answer")

    booked = budget.get_budget(parent.context.session_id)
    assert booked is not None
    history = budget.get_turn_history(parent.context.session_id)
    assert len(history) == 1
    assert booked.used_input_tokens + booked.used_output_tokens == history[0].total_tokens > 0
    assert budget.get_budget(child.context.session_id) is None


@pytest.mark.asyncio
async def test_a_child_is_refused_once_the_parent_s_ceiling_is_spent(tmp_path: Path) -> None:
    """The parent's ceiling is the child's: nothing is asked of the model past it.

    Killed by: src/uclone_x/agent/base.py :: self._parent.enforce_budget(self._session_id, provider=provider)
    Becomes: self._parent.enforce_budget(session_id, provider=provider)
    """
    budget = TokenBudgetManager()
    llm = MockLLMConnector(responses=["done"])
    parent = _parent(tmp_path, llm, ToolRegistry(), budget=budget)
    budget.configure_session(parent.context.session_id, max_tokens=10)
    budget.record_usage(
        parent.context.session_id,
        TokenUsage(provider="mock", input_tokens=10, output_tokens=0, total_tokens=10),
    )

    child = await parent.spawn_subagent(role="helper", goal="help")
    result = await child.execute_turn("answer")

    assert result.stop_reason == "budget_exceeded"
    assert llm.call_count == 0


def test_spawn_subagent_has_no_option_to_share_the_parent_s_session() -> None:
    assert "context_isolation" not in inspect.signature(BaseAgent.spawn_subagent).parameters


def test_every_host_field_is_either_given_to_a_child_or_withheld_on_purpose(
    tmp_path: Path,
) -> None:
    """A field added to `HostDependencies` fails here until someone decides about children.

    Killed by: src/uclone_x/agent/base.py :: "sandbox": self._host.sandbox,
    Becomes: "sandbox_": self._host.sandbox,
    """
    parent = _parent(tmp_path, MockLLMConnector(responses=["done"]), ToolRegistry())
    forwarded = set(parent._subagent_host_fields(None, None))  # pyright: ignore[reportPrivateUsage]
    every_field = {f.name for f in dataclasses.fields(HostDependencies)}

    assert forwarded.isdisjoint(SUBAGENT_EXCLUDED_HOST_FIELDS)
    assert forwarded | SUBAGENT_EXCLUDED_HOST_FIELDS == every_field
