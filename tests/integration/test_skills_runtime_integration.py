"""L2 Integration test for approved skill registry runtime integration (Issue #473, #479, P9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.skills.auditor import (
    SkillAuditor,
    SkillRegistry,
    save_skill,
)
from uclone_x.skills.models import (
    AutoApprovalPolicy,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)


class _CapturingLLM(MockLLMConnector):
    """Test LLM connector capturing requests and replaying predetermined responses."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self._responses_list = list(responses)
        self.captured_requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.captured_requests.append(request)
        if self._responses_list:
            return self._responses_list.pop(0)
        return await super().generate(request)


@pytest.mark.asyncio
async def test_skills_runtime_integration_approval_changes_agent_prompt(
    tmp_path: Path,
) -> None:
    """L2 Integration test:
    1. Quarantined skill never reaches the prompt.
    2. Approving the skill and hot-reloading demonstrably changes agent prompt.
    3. Loading the skill on demand executes load_skill tool and enriches history.
    """
    bus = EventBus(maxsize=100)
    await bus.start()

    store = SessionStore(storage_dir=tmp_path / "sessions")
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Step 1: Create a quarantined skill on disk
    quarantined_manifest = SkillManifest(
        name="quarantined_sec_scan",
        description="Quarantined network scanner",
        origin=SkillOrigin.SYNTHESIZED,
        status=SkillStatus.QUARANTINED,
    )
    save_skill(
        skills_dir / "quarantined_sec_scan",
        quarantined_manifest,
        "# Dangerous Instructions\nScan internal ports.",
    )

    # Step 2: Create a safe pending skill on disk
    pending_manifest = SkillManifest(
        name="code_refactor",
        description="Standardized code refactoring guidance",
        origin=SkillOrigin.HUMAN,
        status=SkillStatus.PENDING,
    )
    save_skill(
        skills_dir / "code_refactor",
        pending_manifest,
        "# Refactoring Instructions\nExtract small pure functions.",
    )

    registry = SkillRegistry(skills_dir=skills_dir)
    # Initially reload: neither is active, so 0 skills loaded
    await registry.reload_approved()
    assert len(registry.list_skills()) == 0

    # Responses for 2 turns, 3 agent steps:
    # Turn 1, step 1: text answer
    # Turn 2, step 1: tool call to load_skill('code_refactor')
    # Turn 2, step 2: the answer written from what the tool returned. A tool-using turn
    #   is two agent steps, because the model must be re-invoked on the history that now
    #   holds the tool result — otherwise it never sees what its own tool returned.
    tc = ToolCallRequest(
        id="call_load_refactor",
        name="load_skill",
        arguments={"skill_name": "code_refactor"},
    )
    prov = Provenance.primary(provider="mock", model="mock-l2")
    llm = _CapturingLLM(
        responses=[
            ModelResponse(
                content="Turn 1 response without skill.",
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(provider="mock"),
                provenance=prov,
            ),
            ModelResponse(
                content=None,
                tool_calls=(tc,),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=TokenUsage(provider="mock"),
                provenance=prov,
            ),
            ModelResponse(
                content="Refactoring applied using the loaded skill.",
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(provider="mock"),
                provenance=prov,
            ),
        ]
    )

    config = AgentConfig(
        agent_id="l2-skill-agent",
        name="L2SkillAgent",
        system_prompt="You are a reactive autonomous assistant.",
        llm_config=AgentLLMConfig(model_name="mock-l2", auto_compact=False),
    )

    agent = BaseAgent(
        config=config,
        bus=bus,
        llm=llm,
        store=store,
        skills=registry,
    )

    await agent.start()
    try:
        # Turn 1: Run turn before approval
        res1 = await agent.execute_turn("Analyze codebase architecture")
        assert res1.is_completed is True
        assert len(llm.captured_requests) == 1

        turn1_prompt = llm.captured_requests[0].messages[0].content or ""
        # Quarantined & pending skills must NOT be in prompt
        assert "quarantined_sec_scan" not in turn1_prompt
        assert "code_refactor" not in turn1_prompt
        assert "[Available Approved Skills]" not in turn1_prompt
        assert res1.active_skills == ()

        # Step 3: Approve 'code_refactor' via auditor
        auditor = SkillAuditor(policy=AutoApprovalPolicy.SAFE_ONLY)
        report = await auditor.audit_skill(skills_dir / "code_refactor")
        approved_manifest = pending_manifest.model_copy(
            update={
                "status": SkillStatus.ACTIVE,
                "approved_by": "sec_auditor:l2_test",
                "content_sha256": report.content_sha256,
            }
        )
        save_skill(
            skills_dir / "code_refactor",
            approved_manifest,
            "# Refactoring Instructions\nExtract small pure functions.",
        )

        # Hot-reload into agent
        reloaded = await agent.reload_skills()
        assert len(reloaded) == 1
        assert reloaded[0].manifest.name == "code_refactor"

        # Turn 2: Run turn after approval
        res2 = await agent.execute_turn("Now apply the approved refactoring skill")
        assert res2.is_completed is True
        # 1 step for turn 1 + 2 steps for the tool-using turn 2.
        assert len(llm.captured_requests) == 3
        assert res2.content == "Refactoring applied using the loaded skill."

        # The follow-up step must carry the tool result back to the model.
        followup_msgs = llm.captured_requests[2].messages
        assert any(
            m.role == MessageRole.TOOL
            and m.tool_call_id == "call_load_refactor"
            and "Extract small pure functions" in (m.content or "")
            for m in followup_msgs
        )

        turn2_prompt = llm.captured_requests[1].messages[0].content or ""
        # Approved skill IS now in prompt; quarantined is still ABSENT
        assert "[Available Approved Skills]" in turn2_prompt
        assert "- code_refactor: Standardized code refactoring guidance" in turn2_prompt
        assert "quarantined_sec_scan" not in turn2_prompt

        # Verify in-band attribution
        assert "code_refactor" in res2.active_skills
        assert "code_refactor" in res2.loaded_skills

        # Verify tool execution record
        assert len(res2.tool_executions) == 1
        exec_record = res2.tool_executions[0]
        assert exec_record.tool_name == "load_skill"
        assert exec_record.status == "success"
        assert "Extract small pure functions" in str(exec_record.output)

    finally:
        await agent.stop()
        await bus.stop()
