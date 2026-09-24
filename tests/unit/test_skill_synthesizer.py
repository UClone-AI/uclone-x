"""Unit tests for SkillSynthesizer pipeline conforming to Principle 9."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from uclone_x.engine.event_bus import AgentEvent, EventSource, EventType
from uclone_x.errors import SkillAuditError
from uclone_x.sandbox.models import IsolationLevel
from uclone_x.skills.auditor import (
    SkillAuditor,
    SkillRegistry,
    compute_skill_sha256,
    load_skill_from_dir,
    save_skill,
)
from uclone_x.skills.models import (
    AuditVerdict,
    AutoApprovalPolicy,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.skills.synthesizer import SkillSynthesizer


def test_skill_synthesizer_initialization_and_properties() -> None:
    """SkillSynthesizer initializes with proper defaults and custom parameters."""
    synth = SkillSynthesizer()
    assert synth.default_isolation is IsolationLevel.WORKSPACE
    assert synth.synthesizer_version == "0.1.0"
    assert isinstance(synth.auditor, SkillAuditor)

    custom_auditor = SkillAuditor(policy=AutoApprovalPolicy.NEVER)
    custom_synth = SkillSynthesizer(
        default_isolation=IsolationLevel.NONE,
        synthesizer_version="1.2.3",
        auditor=custom_auditor,
    )
    assert custom_synth.default_isolation is IsolationLevel.NONE
    assert custom_synth.synthesizer_version == "1.2.3"
    assert custom_synth.auditor is custom_auditor


def test_extract_workflow_from_events_agent_events() -> None:
    """extract_workflow_from_events correctly extracts steps from typed AgentEvents."""
    synth = SkillSynthesizer()

    events = [
        AgentEvent(
            type=EventType.USER_INPUT,
            source=EventSource.USER,
            payload={"message": "Analyze PostgreSQL slow query log and suggest indexes"},
        ),
        AgentEvent(
            type=EventType.TOOL_CALL,
            source=EventSource.AGENT,
            payload={"tool_name": "read_log_file", "arguments": {"path": "/var/log/pg.log"}},
        ),
        AgentEvent(
            type=EventType.TOOL_RESULT,
            source=EventSource.TOOL,
            payload={"tool_name": "read_log_file", "success": True},
        ),
        AgentEvent(
            type=EventType.SUBAGENT_SPAWN,
            source=EventSource.AGENT,
            payload={"role": "index_optimizer", "goal": "Find optimal B-Tree indexes"},
        ),
        AgentEvent(
            type=EventType.AGENT_REPLY,
            source=EventSource.AGENT,
            payload={"content": "Proposed composite index on (user_id, created_at)"},
        ),
    ]

    steps = synth.extract_workflow_from_events(events)
    assert len(steps) == 5
    assert "Process user task intent: 'Analyze PostgreSQL" in steps[0]
    assert "Execute tool 'read_log_file' with arguments:" in steps[1]
    assert "Verify result of tool 'read_log_file' (status: success)" in steps[2]
    assert "Delegate task to subagent 'index_optimizer'" in steps[3]
    assert "Synthesize reasoning stage result:" in steps[4]


def test_extract_workflow_from_events_dict_events() -> None:
    """extract_workflow_from_events extracts steps from serialized event dicts."""
    synth = SkillSynthesizer()

    dict_events = [
        {
            "type": "TOOL_CALL",
            "payload": {"name": "fetch_api", "args": {"url": "https://api.test"}},
        },
        {"type": "TOOL_RESULT", "payload": {"name": "fetch_api", "success": False}},
        {"step": "Custom intermediate transformation step"},
        {"action": "Export report to destination S3 bucket"},
    ]

    steps = synth.extract_workflow_from_events(dict_events)
    assert len(steps) == 4
    assert "Execute tool 'fetch_api'" in steps[0]
    assert "Verify result of tool 'fetch_api' (status: failed)" in steps[1]
    assert steps[2] == "Custom intermediate transformation step"
    assert steps[3] == "Export report to destination S3 bucket"


def test_extract_workflow_from_events_empty_raises() -> None:
    """extract_workflow_from_events raises SkillAuditError on empty or invalid event list."""
    synth = SkillSynthesizer()
    with pytest.raises(SkillAuditError, match="No actionable workflow steps"):
        synth.extract_workflow_from_events([])

    with pytest.raises(SkillAuditError, match="No actionable workflow steps"):
        synth.extract_workflow_from_events([{"type": "UNKNOWN_EVENT_TYPE", "payload": {}}])


def test_extract_workflow_from_trace_json_list(tmp_path: Path) -> None:
    """extract_workflow_from_trace extracts steps from JSON file containing a string list or event list."""
    synth = SkillSynthesizer()

    # 1. List of string steps
    trace_file_str = tmp_path / "trace_str.json"
    trace_file_str.write_text(
        json.dumps(
            ["Step 1: Ingest records", "Step 2: Transform schema", "Step 3: Save to parquet"]
        ),
        encoding="utf-8",
    )
    steps1 = synth.extract_workflow_from_trace(trace_file_str)
    assert len(steps1) == 3
    assert steps1[0] == "Step 1: Ingest records"

    # 2. Nested dict with workflow_steps
    trace_file_dict = tmp_path / "trace_dict.json"
    trace_file_dict.write_text(
        json.dumps({"workflow_steps": ["Extract DB dump", "Anonymize PII", "Upload artifact"]}),
        encoding="utf-8",
    )
    steps2 = synth.extract_workflow_from_trace(trace_file_dict)
    assert len(steps2) == 3
    assert steps2[1] == "Anonymize PII"


def test_extract_workflow_from_trace_yaml(tmp_path: Path) -> None:
    """extract_workflow_from_trace extracts steps from YAML formatted trace."""
    synth = SkillSynthesizer()

    trace_yaml = tmp_path / "trace.yaml"
    trace_yaml.write_text(
        "steps:\n  - Query metrics from Prometheus\n  - Detect anomalous spike\n  - Trigger pager notification\n",
        encoding="utf-8",
    )
    steps = synth.extract_workflow_from_trace(trace_yaml)
    assert len(steps) == 3
    assert steps[0] == "Query metrics from Prometheus"


def test_extract_workflow_from_trace_plain_text(tmp_path: Path) -> None:
    """extract_workflow_from_trace falls back to non-empty lines for plain text traces."""
    synth = SkillSynthesizer()

    trace_txt = tmp_path / "trace.txt"
    trace_txt.write_text(
        "Initialize cluster client\n\nVerify pod health\nScale replicas to 5\n",
        encoding="utf-8",
    )
    steps = synth.extract_workflow_from_trace(trace_txt)
    assert len(steps) == 3
    assert steps[1] == "Verify pod health"


def test_extract_workflow_from_trace_invalid_raises(tmp_path: Path) -> None:
    """extract_workflow_from_trace raises SkillAuditError on missing or empty file."""
    synth = SkillSynthesizer()

    # Missing file
    with pytest.raises(SkillAuditError, match="does not exist"):
        synth.extract_workflow_from_trace(tmp_path / "non_existent.json")

    # Empty file
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(SkillAuditError, match="is empty"):
        synth.extract_workflow_from_trace(empty_file)

    # File with unparseable structure
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"unrelated_key": 123}), encoding="utf-8")
    with pytest.raises(SkillAuditError, match="contains no identifiable workflow steps"):
        synth.extract_workflow_from_trace(bad_file)


def test_extract_workflow_from_session(tmp_path: Path) -> None:
    """extract_workflow_from_session handles both synthetic fallback and on-disk session files."""
    synth = SkillSynthesizer()

    # 1. Fallback deterministic session steps
    steps = synth.extract_workflow_from_session("sess_alpha_99")
    assert len(steps) == 3
    assert "sess_alpha_99" in steps[0]
    assert "sess_alpha_99" in steps[1]

    # 2. Existing session file on disk
    sess_file = tmp_path / "custom_sess.json"
    sess_file.write_text(json.dumps(["Step A", "Step B"]), encoding="utf-8")
    steps_disk = synth.extract_workflow_from_session("custom_sess", session_file=sess_file)
    assert steps_disk == ["Step A", "Step B"]


def test_generate_skill_code() -> None:
    """generate_skill_code produces clean, type-checked Python source code."""
    synth = SkillSynthesizer()
    steps = ["Read input configuration", "Filter invalid records", "Save clean dataset"]
    code = synth.generate_skill_code("data_cleaner", steps, "Clean datasets")

    # AST validation
    tree = ast.parse(code)
    func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "execute_workflow" in func_names
    assert "main" in func_names
    assert "data_cleaner" in code


def test_generate_manifest_yaml() -> None:
    """generate_manifest_yaml produces LinkML-compatible YAML manifest specification."""
    synth = SkillSynthesizer()
    manifest = SkillManifest(
        name="log_analyzer",
        description="Analyzes structured logs",
        origin=SkillOrigin.SYNTHESIZED,
        status=SkillStatus.PENDING,
        requested_isolation=IsolationLevel.WORKSPACE,
        content_sha256="abc123def456",
    )
    steps = ["Step 1", "Step 2"]
    manifest_yaml = synth.generate_manifest_yaml(manifest, steps)
    parsed: dict[str, Any] = yaml.safe_load(manifest_yaml)

    assert parsed["name"] == "log_analyzer"
    assert parsed["origin"] == "synthesized"
    assert parsed["status"] == "pending"
    assert parsed["requested_isolation"] == "workspace"
    assert parsed["workflow_steps"] == ["Step 1", "Step 2"]
    assert parsed["content_sha256"] == "abc123def456"


@pytest.mark.asyncio
async def test_synthesize_skill_quarantine_lifecycle(tmp_path: Path) -> None:
    """synthesize_skill generates package files in quarantine with SHA-256 binding (P9)."""
    synth = SkillSynthesizer()
    steps = [
        "Connect to database replica",
        "Profile sequential table scans",
        "Generate CREATE INDEX DDL",
    ]

    manifest = await synth.synthesize_skill(
        task_name="db_indexer",
        workflow_steps=steps,
        quarantine_dir=tmp_path,
        description="Synthesizes PostgreSQL index definitions",
        tags=("sql", "perf"),
    )

    # 1. Manifest properties
    assert manifest.name == "db_indexer"
    assert manifest.origin is SkillOrigin.SYNTHESIZED
    assert manifest.status is SkillStatus.PENDING
    assert manifest.requested_isolation is IsolationLevel.WORKSPACE
    assert manifest.entrypoint == "main.py"
    assert manifest.scripts == ("main.py",)
    assert manifest.tags == ("sql", "perf")
    assert manifest.content_sha256 is not None
    assert len(manifest.content_sha256) == 64

    # 2. Package files created
    skill_dir = tmp_path / "db_indexer"
    assert skill_dir.is_dir()
    assert (skill_dir / "SKILL.md").is_file()
    assert (skill_dir / "manifest.yaml").is_file()
    assert (skill_dir / "main.py").is_file()

    # 3. Verify load_skill_from_dir loads cleanly
    loaded = load_skill_from_dir(skill_dir)
    assert loaded.manifest.name == "db_indexer"
    assert loaded.manifest.status is SkillStatus.PENDING
    assert "Connect to database replica" in loaded.instructions_markdown

    # 4. Hash verification
    disk_hash = compute_skill_sha256(skill_dir)
    assert manifest.content_sha256 == disk_hash


@pytest.mark.asyncio
async def test_synthesize_skill_invalid_inputs_raises(tmp_path: Path) -> None:
    """synthesize_skill validates inputs and raises ValueError on invalid parameters."""
    synth = SkillSynthesizer()

    # Invalid name (empty or invalid characters)
    with pytest.raises(ValueError, match="Invalid skill name"):
        await synth.synthesize_skill(
            task_name="", workflow_steps=["Step 1"], quarantine_dir=tmp_path
        )

    with pytest.raises(ValueError, match="Invalid skill name"):
        await synth.synthesize_skill(
            task_name="invalid name with spaces",
            workflow_steps=["Step 1"],
            quarantine_dir=tmp_path,
        )

    # Empty workflow steps
    with pytest.raises(ValueError, match="workflow_steps cannot be empty"):
        await synth.synthesize_skill(
            task_name="valid_name",
            workflow_steps=[],
            quarantine_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_synthesize_and_audit_safe_skill(tmp_path: Path) -> None:
    """synthesize_and_audit runs SkillAuditor on synthesized skill and yields APPROVE verdict."""
    synth = SkillSynthesizer()
    steps = ["Read input CSV", "Compute column mean", "Write summary JSON"]

    manifest, report = await synth.synthesize_and_audit(
        task_name="csv_mean_calculator",
        workflow_steps=steps,
        quarantine_dir=tmp_path,
        description="Calculates CSV column statistics",
    )

    # Verification
    assert manifest.name == "csv_mean_calculator"
    assert report.skill_name == "csv_mean_calculator"
    assert report.is_safe is True
    assert report.recommendation is AuditVerdict.APPROVE
    assert report.risk_score == 0.0
    assert len(report.detected_risks) == 0
    assert report.content_sha256 == manifest.content_sha256


@pytest.mark.asyncio
async def test_synthesize_audit_and_registry_admission(tmp_path: Path) -> None:
    """Full synthesis, audit, promotion, and admission pipeline into SkillRegistry."""
    synth = SkillSynthesizer()
    registry = SkillRegistry()

    manifest, report = await synth.synthesize_and_audit(
        task_name="pipeline_runner",
        workflow_steps=["Step A", "Step B"],
        quarantine_dir=tmp_path,
    )

    skill_dir = tmp_path / "pipeline_runner"
    skill = load_skill_from_dir(skill_dir)

    # In quarantine (status: pending), registration succeeds on matching report
    # Promote to active status
    promoted_manifest = manifest.model_copy(
        update={
            "status": SkillStatus.ACTIVE,
            "approved_by": "synthesizer:auto",
            "approved_at": "2026-09-02T12:00:00Z",
            "content_sha256": report.content_sha256,
        }
    )
    save_skill(skill_dir, promoted_manifest, skill.instructions_markdown)

    promoted_skill = load_skill_from_dir(skill_dir)
    # Register with audit report
    registry.register(promoted_skill, report)

    assert registry.get("pipeline_runner") is promoted_skill
    assert len(registry.list_skills()) == 1
