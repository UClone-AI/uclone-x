"""Regression tests asserting active_skills and loaded_skills on all execute_turn return sites (Issue #514).

BaseAgent.execute_turn has five TurnResult return sites:
1) Pre-turn hook BLOCK
2) Agent-step budget refusal
3) BudgetExceededError refusal
4) Generic failover envelope
5) Successful turn execution

active_skills and loaded_skills are declared with default_factory=tuple in TurnResult,
so a return site that omits either field defaults to an empty tuple (). Each test
exercises a return site with a real skill registry where a skill is loaded through the
load_skill tool, proving that both fields are populated from agent state rather than
relying on default values.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import uuid
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.hooks.models import HookAction, HookDecision
from uclone_x.agent.hooks.protocols import BaseHook, HookContext
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, AgentState
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.budget import TokenBudget, TokenBudgetManager
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.skills.auditor import Skill, SkillRegistry
from uclone_x.skills.models import (
    AuditVerdict,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)

SKILL_NAME = "tactical_analysis"
LOAD_TC = ToolCallRequest(
    id="tc_load",
    name="load_skill",
    arguments={"skill_name": SKILL_NAME},
)


def _make_skills_registry() -> SkillRegistry:
    registry = SkillRegistry()
    manifest = SkillManifest(
        name=SKILL_NAME,
        description="Tactical plan evaluation",
        origin=SkillOrigin.HUMAN,
        status=SkillStatus.ACTIVE,
        content_sha256="sha_tac",
    )
    registry.register(
        Skill(manifest=manifest, instructions_markdown="# Tactical\n1. Evaluate."),
        SkillAuditReport(
            skill_name=SKILL_NAME,
            is_safe=True,
            recommendation=AuditVerdict.APPROVE,
            content_sha256="sha_tac",
            auditor_version="0.1.0",
        ),
    )
    return registry


class _BlockingHook(BaseHook):
    async def on_pre_turn(self, context: HookContext) -> HookDecision:
        return HookDecision(action=HookAction.BLOCK, reason="blocked by test hook")


class _AlwaysToolCallsConnector(MockLLMConnector):
    """Requests load_skill with a fresh tool call id on each invocation to exhaust step budget."""

    async def generate(self, request: LLMRequest) -> ModelResponse:
        call_id = f"tc_{uuid.uuid4().hex[:6]}"
        return ModelResponse(
            content="requesting skill",
            tool_calls=(
                ToolCallRequest(
                    id=call_id,
                    name="load_skill",
                    arguments={"skill_name": SKILL_NAME},
                ),
            ),
            usage=TokenUsage(
                provider="mock",
                model="mock-model",
                input_tokens=1,
                output_tokens=1,
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            model_name="mock-model",
            provenance=Provenance.primary("mock", "mock-model"),
        )


async def _prime_agent_with_loaded_skill(agent: BaseAgent) -> None:
    """Execute one turn where load_skill is invoked so agent state holds the skill."""
    res = await agent.execute_turn("prime loaded skill")
    assert res.is_completed is True, f"priming turn failed: {res.error}"
    assert SKILL_NAME in agent.loaded_skills, "priming did not load the skill"
    if agent.state is not AgentState.IDLE:
        agent.transition_to(AgentState.IDLE)


@pytest.mark.asyncio
async def test_execute_turn_success_propagates_skills() -> None:
    """Site 1 (Success): TurnResult propagates active_skills and loaded_skills on success."""
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="agent-success",
            name="SuccessAgent",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=MockLLMConnector(
            responses=["loading", "done"],
            tool_calls=[LOAD_TC],
        ),
        skills=_make_skills_registry(),
    )
    res = await agent.execute_turn("please load the skill")
    assert res.is_completed is True
    assert res.active_skills == (SKILL_NAME,)
    assert res.loaded_skills == (SKILL_NAME,)


@pytest.mark.asyncio
async def test_execute_turn_pre_turn_hook_block_propagates_skills() -> None:
    """Site 2 (Pre-turn Hook BLOCK): TurnResult propagates active_skills and loaded_skills when blocked."""
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="agent-block",
            name="BlockAgent",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=MockLLMConnector(
            responses=["loading", "done"],
            tool_calls=[LOAD_TC],
        ),
        skills=_make_skills_registry(),
    )
    await _prime_agent_with_loaded_skill(agent)
    agent.hook_runner.register_hook(_BlockingHook())

    res = await agent.execute_turn("blocked input")
    assert res.is_completed is False
    assert res.error == "blocked by test hook"
    assert res.active_skills == (SKILL_NAME,)
    assert res.loaded_skills == (SKILL_NAME,)


@pytest.mark.asyncio
async def test_execute_turn_step_budget_refusal_propagates_skills() -> None:
    """Site 3 (Agent-Step Budget Refusal): TurnResult propagates active_skills and loaded_skills."""
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="agent-step-budget",
            name="StepBudgetAgent",
            max_turns=3,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=_AlwaysToolCallsConnector(),
        skills=_make_skills_registry(),
    )
    res = await agent.execute_turn("exhaust step budget")
    assert res.is_completed is False
    assert res.error is not None
    assert "Agent step budget exceeded" in res.error
    assert SKILL_NAME in agent.loaded_skills
    assert res.active_skills == (SKILL_NAME,)
    assert res.loaded_skills == (SKILL_NAME,)


@pytest.mark.asyncio
async def test_execute_turn_budget_exceeded_refusal_propagates_skills() -> None:
    """Site 4 (BudgetExceededError Refusal): TurnResult propagates active_skills and loaded_skills."""
    manager = TokenBudgetManager()
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="agent-budget-exceeded",
            name="BudgetExceededAgent",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=MockLLMConnector(
            responses=["loading", "done"],
            tool_calls=[LOAD_TC],
        ),
        context=AgentContext(
            session_id="sess_budget_skills",
            agent_id="agent-budget-exceeded",
        ),
        skills=_make_skills_registry(),
        budget=manager,
    )
    await _prime_agent_with_loaded_skill(agent)
    manager.set_budget(
        "sess_budget_skills",
        TokenBudget(
            max_tokens=1000,
            used_input_tokens=600,
            used_output_tokens=500,
        ),
    )
    res = await agent.execute_turn("over budget turn")
    assert res.is_completed is False
    assert res.provenance is None
    assert res.active_skills == (SKILL_NAME,)
    assert res.loaded_skills == (SKILL_NAME,)


@pytest.mark.asyncio
async def test_execute_turn_generic_failover_propagates_skills() -> None:
    """Site 5 (Generic Failover Envelope): TurnResult propagates active_skills and loaded_skills on unhandled exception."""
    bus = EventBus()
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="agent-failover",
            name="FailoverAgent",
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=MockLLMConnector(
            responses=["loading", "done"],
            tool_calls=[LOAD_TC],
        ),
        skills=_make_skills_registry(),
        bus=bus,
    )
    await _prime_agent_with_loaded_skill(agent)

    broken_llm = MagicMock(spec=LLMProviderProtocol)
    broken_llm.generate = AsyncMock(side_effect=ConnectionResetError("mock connection reset"))
    broken_llm.provider_name = "mock"
    agent._llm = cast(LLMProviderProtocol, broken_llm)

    res = await agent.execute_turn("trigger exception")
    assert res.is_completed is False
    assert res.provenance is not None
    assert res.active_skills == (SKILL_NAME,)
    assert res.loaded_skills == (SKILL_NAME,)
