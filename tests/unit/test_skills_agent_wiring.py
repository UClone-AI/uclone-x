"""Unit tests for wiring approved skill registry into BaseAgent runtime (Issue #473, P9)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, MessageRole, ModelResponse, ToolCallRequest
from uclone_x.skills.auditor import (
    Skill,
    SkillAuditor,
    SkillRegistry,
    save_skill,
)
from uclone_x.skills.models import (
    AuditVerdict,
    AutoApprovalPolicy,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.tools.builtin.skill_loader import LoadSkillTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry


def _make_active_skill(name: str, desc: str, instructions: str) -> tuple[Skill, SkillAuditReport]:
    """Helper creating a valid active skill and matching passing audit report."""
    manifest = SkillManifest(
        name=name,
        description=desc,
        origin=SkillOrigin.HUMAN,
        status=SkillStatus.ACTIVE,
        content_sha256=f"sha_{name}",
    )
    skill = Skill(manifest=manifest, instructions_markdown=instructions)
    report = SkillAuditReport(
        skill_name=name,
        is_safe=True,
        recommendation=AuditVerdict.APPROVE,
        content_sha256=f"sha_{name}",
        auditor_version="0.1.0",
    )
    return skill, report


@pytest.mark.asyncio
async def test_base_agent_injects_approved_skills_into_system_prompt() -> None:
    """BaseAgent injects approved active skills into system prompt via progressive disclosure."""
    registry = SkillRegistry()
    skill1, report1 = _make_active_skill(
        "code_review",
        "Perform rigorous code reviews and invariant checks",
        "### Full Code Review Instructions\nStep 1: Check P1 to P9.",
    )
    skill2, report2 = _make_active_skill(
        "sql_optimize",
        "Analyze SQL queries and generate index advice",
        "### Full SQL Instructions\nStep 1: Explain analyze.",
    )
    registry.register(skill1, report1)
    registry.register(skill2, report2)

    config = AgentConfig(
        agent_id="test_skill_agent",
        name="SkillAgent",
        system_prompt="Base system instructions.",
        llm_config=AgentLLMConfig(model_name="mock"),
    )
    agent = BaseAgent(config=config, skills=registry)

    # Inspect prepared turn messages
    messages = agent._prepare_turn_messages()  # pyright: ignore[reportPrivateUsage]
    assert len(messages) >= 1
    sys_msg = messages[0]
    assert sys_msg.role == MessageRole.SYSTEM
    content = sys_msg.content or ""

    assert "[Available Approved Skills]" in content
    assert "- code_review: Perform rigorous code reviews and invariant checks" in content
    assert "- sql_optimize: Analyze SQL queries and generate index advice" in content


@pytest.mark.asyncio
async def test_per_turn_prompt_does_not_grow_linearly_with_skill_body() -> None:
    """Progressive disclosure guarantees skill bodies are NOT injected into system prompt."""
    registry = SkillRegistry()
    huge_body = "EXHAUSTIVE PROCEDURAL KNOWLEDGE STEP " * 1000  # ~38KB body
    skill, report = _make_active_skill("heavy_skill", "Lightweight summary description", huge_body)
    registry.register(skill, report)

    config = AgentConfig(
        agent_id="test_progressive_agent",
        name="ProgressiveAgent",
        system_prompt="Base prompt.",
        llm_config=AgentLLMConfig(model_name="mock"),
    )
    agent = BaseAgent(config=config, skills=registry)

    messages = agent._prepare_turn_messages()  # pyright: ignore[reportPrivateUsage]
    sys_content = messages[0].content or ""

    assert "Lightweight summary description" in sys_content
    # The huge instructions body must NOT be in the system prompt
    assert huge_body not in sys_content
    assert len(sys_content) < 500


@pytest.mark.asyncio
async def test_quarantined_and_pending_skills_are_never_injected() -> None:
    """Only ACTIVE skills reach the prompt; quarantined or pending skills are strictly excluded."""
    registry = SkillRegistry()

    # Active skill
    skill_active, report_active = _make_active_skill(
        "active_tool", "Approved active skill", "Instructions"
    )
    registry.register(skill_active, report_active)

    # Manually inject a quarantined skill into registry's internal storage
    # (register() rejects quarantined skills per P9 fail-closed auditor policy)
    quarantined_manifest = SkillManifest(
        name="quarantined_tool",
        description="Dangerous unapproved skill",
        origin=SkillOrigin.SYNTHESIZED,
        status=SkillStatus.QUARANTINED,
    )
    registry._skills["quarantined_tool"] = Skill(  # pyright: ignore[reportPrivateUsage]
        manifest=quarantined_manifest, instructions_markdown="Dangerous code"
    )

    config = AgentConfig(
        agent_id="test_quarantine_agent",
        name="QuarantineAgent",
        system_prompt="Base.",
        llm_config=AgentLLMConfig(model_name="mock"),
    )
    agent = BaseAgent(config=config, skills=registry)

    messages = agent._prepare_turn_messages()  # pyright: ignore[reportPrivateUsage]
    sys_content = messages[0].content or ""

    assert "active_tool" in sys_content
    assert "quarantined_tool" not in sys_content


@pytest.mark.asyncio
async def test_load_skill_tool_execution() -> None:
    """LoadSkillTool retrieves instructions for approved skills and rejects unapproved ones."""
    registry = SkillRegistry()
    skill, report = _make_active_skill(
        "git_bisect",
        "Perform binary search debugging",
        "# Git Bisect Guide\nRun git bisect start.",
    )
    registry.register(skill, report)

    loaded_records: list[str] = []
    tool = LoadSkillTool(registry=registry, on_load=loaded_records.append)

    ctx = ToolContext(agent_id="test_agent", session_id="test_sess", workspace_root=Path("/tmp"))

    # Success case
    result = await tool.execute(params={"skill_name": "git_bisect"}, context=ctx)
    assert result.success is True
    assert result.output == "# Git Bisect Guide\nRun git bisect start."
    assert "git_bisect" in loaded_records

    # Failure case: unknown skill
    fail_result = await tool.execute(params={"skill_name": "non_existent"}, context=ctx)
    assert fail_result.success is False
    assert "not found or has not been approved" in (fail_result.error or "")


@pytest.mark.asyncio
async def test_turn_result_in_band_attribution() -> None:
    """TurnResult records active_skills and loaded_skills in-band per Principle 6."""
    registry = SkillRegistry()
    skill, report = _make_active_skill("unit_test_author", "Write pytest tests", "pytest docs")
    registry.register(skill, report)

    # Tool call to load_skill
    tc = ToolCallRequest(
        id="call_load_1",
        name="load_skill",
        arguments={"skill_name": "unit_test_author"},
    )
    llm = MockLLMConnector(
        responses=["Skill loaded, ready to write tests."],
        tool_calls=[tc],
    )

    config = AgentConfig(
        agent_id="test_turn_attrib",
        name="AttribAgent",
        system_prompt="Base prompt.",
        llm_config=AgentLLMConfig(model_name="mock"),
    )
    agent = BaseAgent(config=config, llm=llm, skills=registry)

    result = await agent.execute_turn("Please load unit test author skill")
    assert result.is_completed is True
    assert "unit_test_author" in result.active_skills
    assert "unit_test_author" in result.loaded_skills


@pytest.mark.asyncio
async def test_agent_reload_skills_from_disk(tmp_path: Path) -> None:
    """Agent can reload newly approved skills dynamically from disk."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Write a pending skill to disk
    manifest = SkillManifest(
        name="dynamic_refactor",
        description="Refactor modules with AST safety",
        origin=SkillOrigin.HUMAN,
        status=SkillStatus.PENDING,
    )
    save_skill(skills_dir / "dynamic_refactor", manifest, "# Refactoring guide")

    registry = SkillRegistry(skills_dir=skills_dir)
    config = AgentConfig(
        agent_id="test_reload_agent",
        name="ReloadAgent",
        system_prompt="Base prompt.",
        llm_config=AgentLLMConfig(model_name="mock"),
    )
    agent = BaseAgent(config=config, skills=registry)

    # Before approval: registry has 0 active skills
    assert len(registry.list_skills()) == 0
    reloaded_before = await agent.reload_skills()
    assert len(reloaded_before) == 0

    # Simulate approval: audit and update status to ACTIVE
    auditor = SkillAuditor(policy=AutoApprovalPolicy.SAFE_ONLY)
    audit_report = await auditor.audit_skill(skills_dir / "dynamic_refactor")
    approved_manifest = manifest.model_copy(
        update={
            "status": SkillStatus.ACTIVE,
            "approved_by": "developer:human",
            "content_sha256": audit_report.content_sha256,
        }
    )
    save_skill(skills_dir / "dynamic_refactor", approved_manifest, "# Refactoring guide")

    # Reload skills
    reloaded_after = await agent.reload_skills()
    assert len(reloaded_after) == 1
    assert reloaded_after[0].manifest.name == "dynamic_refactor"
    assert registry.get("dynamic_refactor") is not None

    # Prompt now includes newly approved skill
    messages = agent._prepare_turn_messages()  # pyright: ignore[reportPrivateUsage]
    assert "- dynamic_refactor: Refactor modules with AST safety" in (messages[0].content or "")


class TestLoadSkillIsBoundToItsOwnAgent:
    """`load_skill` cannot be shared, for the same reason the memory tools cannot (#1098).

    The instance closes over two things that belong to one agent: the `SkillRegistry` that
    decides which skills are approved *for it* (P9), and the `on_load` callback that
    records the load in that agent's `_loaded_skills`. Every head gives its agents one
    `ToolRegistry` -- `ui/app.py` seats the champion and its sub-agents on `self._tools`,
    `ui/rooms.py` seats every participant on `mgr.tools` -- so the first agent composed
    registered the only `load_skill` in it and every later agent resolved *that one*.
    """

    def test_a_second_agent_loads_from_its_own_approval_list(self) -> None:
        """...and the load is recorded against the agent that made it.

        Killed by: src/uclone_x/agent/base.py :: self._agent_local_tools[skill_tool.name] = skill_tool
        Becomes: self._agent_local_tools.pop(skill_tool.name, None)
        """
        shared_tools = ToolRegistry()
        first_skills, second_skills = SkillRegistry(), SkillRegistry()
        first_skills.register(*_make_active_skill("first_only", "Only the first agent's", "# A"))
        second_skills.register(*_make_active_skill("second_only", "Only the second agent's", "# B"))

        first = BaseAgent(
            config=AgentConfig(agent_id="first", name="First"),
            tools=shared_tools,
            skills=first_skills,
        )
        second = BaseAgent(
            config=AgentConfig(agent_id="second", name="Second"),
            tools=shared_tools,
            skills=second_skills,
        )

        record = asyncio.run(second.execute_tool_call("load_skill", {"skill_name": "second_only"}))

        assert record.status == "success", record.error
        assert record.output == "# B"
        # The shared registry holds one advertised copy -- the first agent's -- which is
        # what makes the resolution order load-bearing rather than incidental.
        assert second.loaded_skills == frozenset({"second_only"})
        assert first.loaded_skills == frozenset()

    def test_an_agent_given_no_skills_cannot_borrow_another_agents(self) -> None:
        """An agent composed without a registry has no approved skills, not someone's.

        Resolved from the shared registry it could load any skill the *first* agent's
        registry approved -- a P9 approval gate answered by the wrong registry -- and the
        name would land in the first agent's `_loaded_skills`. "No such tool" is the honest
        answer, and a tool whose every call is refused must not be advertised either.

        Killed by: src/uclone_x/agent/base.py ::     LoadSkillTool,
        Becomes:     QueryMemoryFactsTool,
        """
        shared_tools = ToolRegistry()
        owner_skills = SkillRegistry()
        owner_skills.register(*_make_active_skill("owners_skill", "The owner's", "# owned"))

        BaseAgent(
            config=AgentConfig(agent_id="owner", name="Owner"),
            tools=shared_tools,
            skills=owner_skills,
        )
        skill_less = BaseAgent(
            config=AgentConfig(agent_id="skill_less", name="SkillLess"), tools=shared_tools
        )

        with pytest.raises(KeyError, match="not registered"):
            asyncio.run(skill_less.execute_tool_call("load_skill", {"skill_name": "owners_skill"}))

        capturing = _ToolCapturingLLM()
        watched = BaseAgent(
            config=AgentConfig(agent_id="watched", name="Watched"),
            llm=capturing,
            tools=shared_tools,
        )
        asyncio.run(watched.execute_turn("hello"))

        assert "load_skill" not in capturing.advertised[0]


class _ToolCapturingLLM(MockLLMConnector):
    """Records the tool definitions each turn advertises."""

    def __init__(self) -> None:
        super().__init__(default_response="ok")
        self.advertised: list[tuple[str, ...]] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.advertised.append(tuple(tool.name for tool in request.tools))
        return await super().generate(request)
