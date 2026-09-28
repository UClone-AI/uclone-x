"""`agent/tool_execution.py` on its own: running one tool call (#1736).

The call path moved out of `agent/base.py` unchanged, and the behavioural tests that drive
it through `BaseAgent._execute_single_tool` stay where they were (`test_agent_hooks.py`,
`test_story_audit.py`, `test_tool_approval_integration.py`, ...). This file pins the
module directly, and pins what the move itself introduced: the executor holds no copy of
agent state, and the plan callables the agent hands it pass their arguments through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.hooks import BaseHook, HookAction, HookContext, HookDecision, HookRunner
from uclone_x.agent.models import AgentConfig, AgentContext, PlanState
from uclone_x.agent.tool_execution import (
    ExecutionScope,
    PlanSession,
    ToolCallExecutor,
    _modified_arguments,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.agent.tool_invoker import ToolInvoker
from uclone_x.llm.models import ToolCallRequest
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import create_default_registry


class _BlockingHook(BaseHook):
    async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
        return HookDecision(action=HookAction.BLOCK, reason="not today")


class _AskingHook(BaseHook):
    async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
        return HookDecision(action=HookAction.ASK, reason="check with a person")


class _EmptyInvoker:
    """A catalog in whose range every name is, and which holds no tool at all."""

    def in_tool_range(self, name: str) -> bool:
        return True

    def resolve(self, name: str) -> None:
        return None


@dataclass
class _Session:
    plan: PlanState | None = None


@dataclass
class _State:
    hook_runner: HookRunner = field(default_factory=HookRunner)
    session: _Session = field(default_factory=_Session)


def _unexpected(*_: Any) -> Any:
    raise AssertionError("a call that never reaches a tool must not touch the plan")


def _executor(state: _State) -> ToolCallExecutor:
    config = AgentConfig(agent_id="executor", name="Executor")
    return ToolCallExecutor(
        ExecutionScope(
            agent_id=lambda: "executor",
            context=lambda: AgentContext(session_id="s", agent_id="executor"),
            config=lambda: config,
            tool_invoker=lambda: cast(ToolInvoker, _EmptyInvoker()),
            hook_runner=lambda: state.hook_runner,
            approvals_answered=lambda: True,
            bus=lambda: None,
            publisher=lambda: None,
            capability_refusal=lambda tool: None,
            current_plan=lambda: state.session.plan,
            create_plan=_unexpected,
            update_step_status=_unexpected,
            live_session=lambda session_id: cast(PlanSession, state.session),
            publish_plan_update=_unexpected,
        )
    )


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(agent_id="executor", session_id="s", workspace_root=tmp_path)


def test_a_hook_payload_naming_neither_key_is_itself_the_arguments() -> None:
    """`{"path": ...}` from a hook is the rewritten call's arguments, not "no rewrite".

    Killed by: src/uclone_x/agent/tool_execution.py :: return cast(dict[str, Any], unwrap_immutable(modified_payload))
    Becomes: return None
    """
    assert _modified_arguments({"path": "a.txt"}) == {"path": "a.txt"}
    assert _modified_arguments({"arguments": {"path": "b.txt"}}) == {"path": "b.txt"}
    assert _modified_arguments({"tool_name": "other"}) is None
    assert _modified_arguments(None) is None


@pytest.mark.asyncio
async def test_a_name_in_range_that_no_catalog_holds_is_reported_not_found(
    tmp_path: Path,
) -> None:
    """In range but held nowhere: an error record naming the tool, and nothing runs.

    Killed by: src/uclone_x/agent/tool_execution.py :: err_msg = f"Tool '{effective_tc.name}' not found"
    Becomes: err_msg = "unavailable"
    """
    call = ToolCallRequest(id="c1", name="ghost", arguments={})

    msg, rec = await _executor(_State()).execute_single_tool(call, _ctx(tmp_path))

    assert rec.status == ToolResultStatus.ERROR
    assert rec.error == "Tool 'ghost' not found"
    assert msg.content == "Tool 'ghost' not found"
    assert msg.tool_call_id == "c1"


@pytest.mark.asyncio
async def test_the_executor_asks_the_hook_runner_it_is_given_at_call_time(
    tmp_path: Path,
) -> None:
    """A hook runner swapped in after construction is the one the next call consults.

    Killed by: src/uclone_x/agent/tool_execution.py :: return self._scope.hook_runner()
    Becomes: return HookRunner()
    """
    state = _State()
    executor = _executor(state)
    state.hook_runner = HookRunner(hooks=[_BlockingHook()])

    msg, rec = await executor.execute_single_tool(
        ToolCallRequest(id="c1", name="ghost", arguments={}), _ctx(tmp_path)
    )

    assert rec.error == "Tool execution blocked by hook: not today"
    assert msg.content == "Tool execution blocked by hook: not today"


@pytest.mark.asyncio
async def test_an_agent_that_stops_answering_approvals_refuses_before_asking(
    tmp_path: Path,
) -> None:
    """The agent hands the executor a live read of `_approvals_answered`, not its first value.

    Answered at construction, then not: the next call that needs approval is refused at
    once in the "cannot ask" words, rather than asked and timed out.

    Killed by: src/uclone_x/agent/base.py :: approvals_answered=lambda: self._approvals_answered,
    Becomes: approvals_answered=lambda _a=self._approvals_answered: _a,
    """
    agent = BaseAgent(
        config=AgentConfig(agent_id="executor", name="Executor"),
        hooks=[_AskingHook()],
        approvals_answered=True,
    )
    agent._approvals_answered = False  # pyright: ignore[reportPrivateUsage]

    _, rec = await agent._execute_single_tool(  # pyright: ignore[reportPrivateUsage]
        ToolCallRequest(id="c1", name="file_read", arguments={"path": "a.txt"}),
        _ctx(tmp_path),
    )

    assert rec.status == ToolResultStatus.ERROR
    assert rec.error is not None and "cannot ask for it" in rec.error


@pytest.mark.asyncio
async def test_the_plan_tool_reaches_the_agents_plan_with_every_argument(
    tmp_path: Path,
) -> None:
    """`update_plan` creates with all its steps and marks a step with what it was sent.

    The agent's plan methods reach the executor through lambdas that re-spell their
    arguments; this pins that nothing is dropped or defaulted on the way.

    Killed by: src/uclone_x/agent/base.py :: index=index, completed=completed, verification=verification
    Becomes: index=index, completed=False, verification=verification
    """
    agent = BaseAgent(
        config=AgentConfig(agent_id="executor", name="Executor", workspace_dir=tmp_path),
        tools=create_default_registry(workspace_root=tmp_path, enable_mcp=False),
    )
    ctx = _ctx(tmp_path)
    create = ToolCallRequest(
        id="c1",
        name="update_plan",
        arguments={
            "action": "create",
            "title": "Ship",
            "steps": [{"description": "one"}, {"description": "two"}],
        },
    )
    update = ToolCallRequest(
        id="c2",
        name="update_plan",
        arguments={
            "action": "update",
            "steps": [{"index": 1, "completed": True, "verification": "seen"}],
        },
    )

    _, created = await agent._execute_single_tool(create, ctx)  # pyright: ignore[reportPrivateUsage]
    _, updated = await agent._execute_single_tool(update, ctx)  # pyright: ignore[reportPrivateUsage]

    assert (created.status, updated.status) == (ToolResultStatus.SUCCESS,) * 2
    plan = agent.current_plan
    assert plan is not None and plan.title == "Ship"
    assert [s.description for s in plan.steps] == ["one", "two"]
    assert (plan.steps[0].completed, plan.steps[0].verification) == (True, "seen")
    assert plan.steps[1].completed is False
