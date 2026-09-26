"""Regression and unit tests for max_steps and deprecated max_turns alias validation (P6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.agent.models import AgentConfig, SubAgentSpec
from uclone_x.errors import StepBudgetExceededError
from uclone_x.sandbox.models import WorkspaceIsolation
from uclone_x.tools.builtin.subagent import SubagentDelegationParams, SubagentDelegationTool
from uclone_x.tools.models import ToolContext


def test_agent_config_alias_both_equal() -> None:
    """When both max_steps and max_turns are provided and equal, both are preserved."""
    config = AgentConfig(agent_id="test", name="Test", max_steps=15, max_turns=15)
    assert config.max_steps == 15
    assert config.max_turns == 15


def test_agent_config_alias_only_steps_set() -> None:
    """When only max_steps is provided, max_turns is populated with the same value."""
    config = AgentConfig(agent_id="test", name="Test", max_steps=12)
    assert config.max_steps == 12
    assert config.max_turns == 12


def test_agent_config_alias_only_turns_set() -> None:
    """When only deprecated max_turns is provided, max_steps is populated with the same value."""
    config = AgentConfig(agent_id="test", name="Test", max_turns=18)
    assert config.max_steps == 18
    assert config.max_turns == 18


def test_agent_config_alias_defaults() -> None:
    """When neither is provided, defaults match."""
    config = AgentConfig(agent_id="test", name="Test")
    assert config.max_steps == 50
    assert config.max_turns == 50


def test_agent_config_alias_conflicting_values_raises_value_error() -> None:
    """Contradictory max_steps and max_turns must fail fast rather than silently resolve per P6.

    Killed by: src/uclone_x/agent/models.py :: if steps != turns:  # AgentConfig budget conflict check
    """
    with pytest.raises(
        ValueError,
        match=r"Conflicting values for max_steps and deprecated alias max_turns: 1 != 99",
    ):
        AgentConfig(agent_id="test", name="Test", max_steps=1, max_turns=99)


def test_subagent_spec_alias_both_equal() -> None:
    """When both max_steps and max_turns are provided and equal on SubAgentSpec, both match."""
    spec = SubAgentSpec(
        name="sub",
        role="worker",
        system_prompt="sys",
        max_steps=25,
        max_turns=25,
    )
    assert spec.max_steps == 25
    assert spec.max_turns == 25


def test_subagent_spec_alias_only_steps_set() -> None:
    """When only max_steps is provided on SubAgentSpec, max_turns is populated."""
    spec = SubAgentSpec(
        name="sub",
        role="worker",
        system_prompt="sys",
        max_steps=30,
    )
    assert spec.max_steps == 30
    assert spec.max_turns == 30


def test_subagent_spec_alias_only_turns_set() -> None:
    """When only deprecated max_turns is provided on SubAgentSpec, max_steps is populated."""
    spec = SubAgentSpec(
        name="sub",
        role="worker",
        system_prompt="sys",
        max_turns=35,
    )
    assert spec.max_steps == 35
    assert spec.max_turns == 35


def test_subagent_spec_alias_defaults() -> None:
    """When neither is provided on SubAgentSpec, defaults match."""
    spec = SubAgentSpec(name="sub", role="worker", system_prompt="sys")
    assert spec.max_steps == 20
    assert spec.max_turns == 20


def test_subagent_spec_alias_conflicting_values_raises_value_error() -> None:
    """Contradictory max_steps and max_turns on SubAgentSpec must fail fast per P6.

    Killed by: src/uclone_x/agent/models.py :: if steps != turns:  # SubAgentSpec budget conflict check
    """
    with pytest.raises(
        ValueError,
        match=r"Conflicting values for max_steps and deprecated alias max_turns: 10 != 20",
    ):
        SubAgentSpec(
            name="sub",
            role="worker",
            system_prompt="sys",
            max_steps=10,
            max_turns=20,
        )


def test_subagent_delegation_params_both_equal() -> None:
    """When both max_steps and max_turns are provided and equal on SubagentDelegationParams."""
    params = SubagentDelegationParams(
        role="researcher",
        goal="find answer",
        prompt="do query",
        max_steps=8,
        max_turns=8,
    )
    assert params.max_steps == 8
    assert params.max_turns == 8


def test_subagent_delegation_params_only_steps_set() -> None:
    """When only max_steps is provided on SubagentDelegationParams, max_turns is populated."""
    params = SubagentDelegationParams(
        role="researcher",
        goal="find answer",
        prompt="do query",
        max_steps=8,
    )
    assert params.max_steps == 8
    assert params.max_turns == 8


def test_subagent_delegation_params_only_turns_set() -> None:
    """When only max_turns is provided on SubagentDelegationParams, max_steps is populated."""
    params = SubagentDelegationParams(
        role="researcher",
        goal="find answer",
        prompt="do query",
        max_turns=8,
    )
    assert params.max_steps == 8
    assert params.max_turns == 8


def test_subagent_delegation_params_explicit_none_with_value() -> None:
    """Passing None for one field and a value for the other normalizes cleanly."""
    steps_set = SubagentDelegationParams(
        role="r",
        goal="g",
        prompt="p",
        max_steps=14,
        max_turns=None,
    )
    assert steps_set.max_steps == 14
    assert steps_set.max_turns == 14

    turns_set = SubagentDelegationParams(
        role="r",
        goal="g",
        prompt="p",
        max_steps=None,
        max_turns=14,
    )
    assert turns_set.max_steps == 14
    assert turns_set.max_turns == 14


def test_subagent_delegation_params_both_none() -> None:
    """When neither max_steps nor max_turns is provided, both remain None."""
    params = SubagentDelegationParams(role="r", goal="g", prompt="p")
    assert params.max_steps is None
    assert params.max_turns is None


def test_subagent_delegation_params_conflicting_values_raises_value_error() -> None:
    """Contradictory max_steps and max_turns on SubagentDelegationParams must fail fast per P6.

    Killed by: src/uclone_x/tools/builtin/subagent.py :: if steps != turns:  # SubagentDelegationParams budget conflict check
    """
    with pytest.raises(
        ValueError,
        match=r"Conflicting values for max_steps and deprecated alias max_turns: 4 != 9",
    ):
        SubagentDelegationParams(
            role="researcher",
            goal="find answer",
            prompt="do query",
            max_steps=4,
            max_turns=9,
        )


def test_subagent_delegation_schema_marks_max_turns_deprecated() -> None:
    """The JSON schema for SubagentDelegationTool must clearly mark max_turns as deprecated."""
    tool = SubagentDelegationTool()
    schema = tool.parameters_schema
    props = schema["properties"]
    assert "max_turns" in props
    assert props["max_turns"].get("deprecated") is True
    assert "max_steps" in props
    assert props["max_steps"].get("deprecated") is not True


@pytest.mark.asyncio
async def test_subagent_delegation_tool_execution_rejects_conflicting_parameters() -> None:
    """Executing delegate_subagent with conflicting step and turn budgets fails fast per P6."""
    tool = SubagentDelegationTool()
    from pathlib import Path

    context = ToolContext(
        agent_id="agent_1",
        session_id="s1",
        workspace_root=Path("/tmp"),
        isolation=WorkspaceIsolation(),
    )
    result = await tool.execute(
        params={
            "role": "researcher",
            "goal": "research",
            "prompt": "search",
            "max_steps": 3,
            "max_turns": 7,
        },
        context=context,
    )
    assert result.success is False
    assert "Conflicting values for max_steps and deprecated alias max_turns: 3 != 7" in (
        result.error or ""
    )


@pytest.mark.asyncio
async def test_a_refused_delegation_says_so_in_plain_words(tmp_path: Path) -> None:
    """The tool's own sentence reaches the model, without pydantic's report around it and
    with what was not done -- no helper started (#1570).

    Killed by: src/uclone_x/tools/builtin/subagent.py :: error=describe_invalid_arguments(
    Becomes: error=str(e) + describe_invalid_arguments(
    Killed by: src/uclone_x/tools/base.py :: if kind in {"value_error", "assertion_error"} and ctx.get("error") is not None:
    Becomes: if False:
    """
    context = ToolContext(
        agent_id="agent_1",
        session_id="s1",
        workspace_root=tmp_path,
        isolation=WorkspaceIsolation(),
    )
    result = await SubagentDelegationTool().execute(
        params={"role": "r", "goal": "g", "prompt": "p", "max_steps": 3, "max_turns": 7},
        context=context,
    )

    assert result.success is False
    assert result.error == (
        "The call to 'delegate_subagent' was refused because its arguments did not fit: "
        "Conflicting values for max_steps and deprecated alias max_turns: 3 != 7. No helper "
        "was started, so nothing was done. Call it again with the arguments corrected."
    )
    for internal in ("pydantic", "http", "Value error", "validation error", "SubagentDelegation"):
        assert internal not in result.error


def test_step_budget_exceeded_error_conflicting_values() -> None:
    """StepBudgetExceededError rejects contradictory max_steps and max_turns per P6.

    Killed by: src/uclone_x/errors.py :: if max_steps is not None and max_turns is not None and max_steps != max_turns:
    """
    with pytest.raises(
        ValueError,
        match=r"Conflicting values for max_steps and deprecated alias max_turns: 5 != 10",
    ):
        StepBudgetExceededError("budget exceeded", max_steps=5, max_turns=10)


def test_step_budget_exceeded_error_conflicting_current_values() -> None:
    """StepBudgetExceededError rejects contradictory current_steps and current_turns per P6.

    Killed by: src/uclone_x/errors.py :: and current_steps != current_turns
    """
    with pytest.raises(
        ValueError,
        match=r"Conflicting values for current_steps and deprecated alias current_turns: 3 != 6",
    ):
        StepBudgetExceededError("budget exceeded", current_steps=3, current_turns=6)
