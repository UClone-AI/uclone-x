"""Unit tests for Skill CLI commands (ucx skill list, approve, reject, audit)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from uclone_x.cli.main import app
from uclone_x.sandbox.models import IsolationLevel
from uclone_x.skills.auditor import load_skill_from_dir, save_skill
from uclone_x.skills.models import SkillManifest, SkillOrigin, SkillStatus

runner = CliRunner()


def test_cli_skill_list_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["skill", "list", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No skills found" in result.output


def test_cli_skill_list_and_filtering(tmp_path: Path) -> None:
    # 1. Create a pending skill
    pending_dir = tmp_path / "pending_skill"
    save_skill(
        pending_dir,
        SkillManifest(
            name="pending_skill",
            description="Pending synthesis",
            origin=SkillOrigin.SYNTHESIZED,
            status=SkillStatus.PENDING,
        ),
        "# Pending",
    )

    # 2. Create an active skill
    active_dir = tmp_path / "active_skill"
    save_skill(
        active_dir,
        SkillManifest(
            name="active_skill",
            description="Active human skill",
            origin=SkillOrigin.HUMAN,
            status=SkillStatus.ACTIVE,
            approved_by="human:admin",
        ),
        "# Active",
    )

    # 3. Create a rejected skill
    rejected_dir = tmp_path / "rejected_skill"
    save_skill(
        rejected_dir,
        SkillManifest(
            name="rejected_skill",
            description="Rejected skill",
            origin=SkillOrigin.SYNTHESIZED,
            status=SkillStatus.REJECTED,
        ),
        "# Rejected",
    )

    # Normal list (shows pending and active, hides rejected without --all)
    res_list = runner.invoke(app, ["skill", "list", "--dir", str(tmp_path)], env={"COLUMNS": "200"})
    assert res_list.exit_code == 0
    assert "pending_skill" in res_list.output
    assert "active_skill" in res_list.output
    assert "rejected_skill" not in res_list.output

    # List with --all includes rejected
    res_all = runner.invoke(
        app, ["skill", "list", "--all", "--dir", str(tmp_path)], env={"COLUMNS": "200"}
    )
    assert res_all.exit_code == 0
    assert "rejected_skill" in res_all.output

    # Filter by --pending
    res_pending = runner.invoke(
        app, ["skill", "list", "--pending", "--dir", str(tmp_path)], env={"COLUMNS": "200"}
    )
    assert res_pending.exit_code == 0
    assert "pending_skill" in res_pending.output
    assert "active_skill" not in res_pending.output

    # Empty pending filter when no pending skills exist
    empty_tmp = tmp_path / "empty_root"
    empty_tmp.mkdir()
    save_skill(
        empty_tmp / "only_active",
        SkillManifest(
            name="only_active",
            description="Active",
            origin=SkillOrigin.HUMAN,
            status=SkillStatus.ACTIVE,
        ),
        "# Active",
    )
    res_no_pending = runner.invoke(app, ["skill", "list", "--pending", "--dir", str(empty_tmp)])
    assert res_no_pending.exit_code == 0
    assert "No pending skills awaiting review" in res_no_pending.output


def test_cli_skill_approve_promotes_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "data_exporter"
    save_skill(
        skill_dir,
        SkillManifest(
            name="data_exporter",
            description="Export data",
            origin=SkillOrigin.SYNTHESIZED,
            status=SkillStatus.PENDING,
            requested_isolation=IsolationLevel.WORKSPACE,
        ),
        "# Data Exporter",
    )

    result = runner.invoke(
        app,
        ["skill", "approve", "data_exporter", "--approver", "human:alice", "--dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "Approved skill: data_exporter" in result.output
    assert "human:alice" in result.output

    # Verify on-disk promotion
    reloaded = load_skill_from_dir(skill_dir)
    assert reloaded.manifest.status is SkillStatus.ACTIVE
    assert reloaded.manifest.approved_by == "human:alice"
    assert reloaded.manifest.approved_at is not None
    assert reloaded.manifest.content_sha256 is not None

    # Idempotent re-approve notice
    result2 = runner.invoke(
        app,
        ["skill", "approve", "data_exporter", "--dir", str(tmp_path)],
    )
    assert result2.exit_code == 0
    assert "already active" in result2.output


def test_cli_skill_approve_blocks_dangerous_unless_forced(tmp_path: Path) -> None:
    skill_dir = tmp_path / "dangerous_skill"
    save_skill(
        skill_dir,
        SkillManifest(
            name="dangerous_skill",
            description="Dangerous script",
            origin=SkillOrigin.SYNTHESIZED,
            status=SkillStatus.PENDING,
        ),
        "# Dangerous",
    )
    (skill_dir / "bad.py").write_text("import os\nos.system('rm -rf /')\n", encoding="utf-8")

    # Regular approve is blocked by auditor
    res_block = runner.invoke(app, ["skill", "approve", "dangerous_skill", "--dir", str(tmp_path)])
    assert res_block.exit_code == 1
    assert "Cannot approve skill" in res_block.output
    assert "REJECT" in res_block.output

    # Force approve bypasses auditor rejection
    res_force = runner.invoke(
        app, ["skill", "approve", "dangerous_skill", "--force", "--dir", str(tmp_path)]
    )
    assert res_force.exit_code == 0
    assert "Approved skill: dangerous_skill" in res_force.output
    assert load_skill_from_dir(skill_dir).manifest.status is SkillStatus.ACTIVE


def test_cli_skill_approve_missing_skill(tmp_path: Path) -> None:
    result = runner.invoke(app, ["skill", "approve", "ghost_skill", "--dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_skill_reject_records_provenance(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad_skill"
    save_skill(
        skill_dir,
        SkillManifest(
            name="bad_skill",
            description="Bad logic",
            origin=SkillOrigin.SYNTHESIZED,
            status=SkillStatus.PENDING,
        ),
        "# Bad",
    )

    result = runner.invoke(
        app,
        [
            "skill",
            "reject",
            "bad_skill",
            "--reason",
            "Insecure logic design",
            "--rejecter",
            "human:security_lead",
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "Rejected skill: bad_skill" in result.output
    assert "Insecure logic design" in result.output

    reloaded = load_skill_from_dir(skill_dir)
    assert reloaded.manifest.status is SkillStatus.REJECTED
    assert reloaded.manifest.rejected_by == "human:security_lead"
    assert reloaded.manifest.rejected_at is not None
    assert reloaded.manifest.rejection_reason == "Insecure logic design"


def test_cli_skill_reject_missing_skill(tmp_path: Path) -> None:
    result = runner.invoke(app, ["skill", "reject", "non_existent", "--dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_skill_audit_command(tmp_path: Path) -> None:
    # 1. Safe skill
    safe_dir = tmp_path / "safe_skill"
    save_skill(
        safe_dir,
        SkillManifest(
            name="safe_skill",
            description="Safe skill",
            origin=SkillOrigin.HUMAN,
            requested_isolation=IsolationLevel.WORKSPACE,
        ),
        "# Safe",
    )

    res_safe = runner.invoke(
        app, ["skill", "audit", "safe_skill", "--policy", "safe_only", "--dir", str(tmp_path)]
    )
    assert res_safe.exit_code == 0
    assert "Audit passed" in res_safe.output
    assert "approve" in res_safe.output

    # 2. Safe skill under NEVER policy
    res_never = runner.invoke(
        app, ["skill", "audit", "safe_skill", "--policy", "never", "--dir", str(tmp_path)]
    )
    assert res_never.exit_code == 0
    assert "Human Review Required" in res_never.output

    # 3. Dangerous skill
    evil_dir = tmp_path / "evil_skill"
    save_skill(
        evil_dir,
        SkillManifest(
            name="evil_skill",
            description="Evil",
            origin=SkillOrigin.SYNTHESIZED,
        ),
        "# Evil",
    )
    (evil_dir / "exec.py").write_text(
        "eval('import os; os.system(\"rm -rf /\")')", encoding="utf-8"
    )

    res_evil = runner.invoke(app, ["skill", "audit", "evil_skill", "--dir", str(tmp_path)])
    assert res_evil.exit_code == 0
    assert "Audit Failed" in res_evil.output
    assert "reject" in res_evil.output

    # 4. Invalid policy
    res_bad_policy = runner.invoke(
        app, ["skill", "audit", "safe_skill", "--policy", "invalid_mode", "--dir", str(tmp_path)]
    )
    assert res_bad_policy.exit_code == 1
    assert "Invalid policy" in res_bad_policy.output

    # 5. Missing skill
    res_missing = runner.invoke(app, ["skill", "audit", "ghost", "--dir", str(tmp_path)])
    assert res_missing.exit_code == 1
    assert "not found" in res_missing.output


def test_cli_skill_synthesize_from_steps(tmp_path: Path) -> None:
    """CLI synthesize creates a quarantined skill package from explicit steps."""
    result = runner.invoke(
        app,
        [
            "skill",
            "synthesize",
            "--name",
            "csv_exporter",
            "--step",
            "Read input SQL database table",
            "--step",
            "Format records into CSV",
            "--step",
            "Write output file to disk",
            "--description",
            "Exports SQL tables to CSV",
            "--dir",
            str(tmp_path),
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "Synthesized skill package in quarantine" in result.output
    assert "csv_exporter" in result.output
    assert "Audit Verdict" in result.output
    assert "approve" in result.output

    # Check files on disk
    skill_dir = tmp_path / "csv_exporter"
    assert skill_dir.is_dir()
    assert (skill_dir / "SKILL.md").is_file()
    assert (skill_dir / "manifest.yaml").is_file()
    assert (skill_dir / "main.py").is_file()

    loaded = load_skill_from_dir(skill_dir)
    assert loaded.manifest.status is SkillStatus.PENDING
    assert loaded.manifest.origin is SkillOrigin.SYNTHESIZED


def test_cli_skill_synthesize_from_trace_file(tmp_path: Path) -> None:
    """CLI synthesize extracts workflow from a trace JSON file."""
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(
        '{"workflow_steps": ["Extract system metrics", "Filter CPU anomalies", "Generate alert"]}',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "skill",
            "synthesize",
            "--name",
            "cpu_monitor",
            "--from-trace",
            str(trace_file),
            "--dir",
            str(tmp_path),
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "Synthesized skill package in quarantine" in result.output
    assert "cpu_monitor" in result.output

    skill_dir = tmp_path / "cpu_monitor"
    assert (skill_dir / "SKILL.md").is_file()


def test_cli_skill_synthesize_from_session_id(tmp_path: Path) -> None:
    """CLI synthesize extracts workflow from session identifier."""
    result = runner.invoke(
        app,
        [
            "skill",
            "synthesize",
            "--name",
            "session_task_skill",
            "--session-id",
            "sess_prod_101",
            "--dir",
            str(tmp_path),
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "session_task_skill" in result.output
    assert (tmp_path / "session_task_skill" / "SKILL.md").is_file()


def test_cli_skill_synthesize_auto_approve(tmp_path: Path) -> None:
    """CLI synthesize with --auto-approve automatically activates safe skills."""
    result = runner.invoke(
        app,
        [
            "skill",
            "synthesize",
            "--name",
            "auto_promoted_skill",
            "--step",
            "Step 1: Compute hash",
            "--step",
            "Step 2: Validate token",
            "--auto-approve",
            "--dir",
            str(tmp_path),
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "Auto-approved and registered skill" in result.output
    assert "active" in result.output

    skill_dir = tmp_path / "auto_promoted_skill"
    loaded = load_skill_from_dir(skill_dir)
    assert loaded.manifest.status is SkillStatus.ACTIVE
    assert loaded.manifest.approved_by == "synthesizer:auto"


def test_cli_skill_synthesize_missing_inputs_fails(tmp_path: Path) -> None:
    """CLI synthesize fails fast when no workflow source is specified."""
    res_no_source = runner.invoke(
        app,
        ["skill", "synthesize", "--name", "empty_skill", "--dir", str(tmp_path)],
    )
    assert res_no_source.exit_code == 1
    assert "Must provide either --session-id, --from-trace, or --step" in res_no_source.output

    res_empty_name = runner.invoke(
        app,
        ["skill", "synthesize", "--name", "   ", "--step", "Step 1", "--dir", str(tmp_path)],
    )
    assert res_empty_name.exit_code == 1
    assert "Skill name cannot be empty" in res_empty_name.output


def test_cli_skill_synthesize_invalid_policy_fails(tmp_path: Path) -> None:
    """CLI synthesize fails fast when an invalid policy is supplied."""
    result = runner.invoke(
        app,
        [
            "skill",
            "synthesize",
            "--name",
            "policy_test",
            "--step",
            "Step 1",
            "--policy",
            "nonexistent_policy",
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "Invalid policy" in result.output
