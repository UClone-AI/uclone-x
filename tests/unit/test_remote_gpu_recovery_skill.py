"""Tests for the remote_gpu_recovery runtime skill package."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.skills.auditor import SkillAuditor, load_skill_from_dir
from uclone_x.skills.models import AuditVerdict, SkillOrigin, SkillStatus

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "ucx-agent-skills" / "remote_gpu_recovery"


def test_remote_gpu_recovery_skill_manifest() -> None:
    """Verify that remote_gpu_recovery package exists, parses, and meets safety criteria."""
    assert SKILL_DIR.is_dir()
    skill_file = SKILL_DIR / "SKILL.md"
    assert skill_file.is_file()

    skill = load_skill_from_dir(SKILL_DIR)
    assert skill is not None

    manifest = skill.manifest
    assert manifest.name == "remote_gpu_recovery"
    assert manifest.version == "0.1.0"
    assert manifest.origin == SkillOrigin.HUMAN
    assert manifest.status == SkillStatus.ACTIVE
    assert "comfyui" in manifest.tags
    assert "ollama" in manifest.tags

    # Instructions contain required playbooks and safety constraints
    instructions = skill.instructions_markdown
    assert "Mandatory Execution Rules" in instructions
    assert "~/ComfyUI/venv/bin/pip" in instructions
    assert "Playbook 1: Ollama Model Not Found" in instructions
    assert "Playbook 2: ComfyUI Missing Python Module" in instructions
    assert "Playbook 3: ComfyUI Port 8188 Inactive" in instructions
    assert "nohup ~/ComfyUI/venv/bin/python3" in instructions
    assert "</dev/null >~/ComfyUI/comfy.log 2>&1 &" in instructions


@pytest.mark.asyncio
async def test_remote_gpu_recovery_skill_audits_cleanly() -> None:
    """Verify that SkillAuditor verifies remote_gpu_recovery without security red flags."""
    auditor = SkillAuditor()
    report = await auditor.audit_skill(SKILL_DIR)

    assert report.is_safe is True
    assert report.verdict in (AuditVerdict.APPROVE, AuditVerdict.REQUIRE_HUMAN_REVIEW)
    assert report.risk_score is not None and report.risk_score < 0.5
