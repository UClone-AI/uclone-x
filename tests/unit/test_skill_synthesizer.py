"""Unit tests for SkillSynthesizer pipeline conforming to Principle 9."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uclone_x.engine.event_bus import AgentEvent, EventSource, EventType
from uclone_x.errors import SkillAuditError
from uclone_x.sandbox.models import IsolationLevel
from uclone_x.skills.auditor import (
    SkillAuditor,
    SkillRegistry,
    compute_skill_sha256,
    load_skill_from_dir,
    parse_skill_markdown,
    save_skill,
)
from uclone_x.skills.models import (
    AuditVerdict,
    AutoApprovalPolicy,
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


def test_a_session_with_no_trace_is_refused_rather_than_given_invented_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No trace for the session means no steps, said in plain words (#1723, P6).

    It used to return three generic steps naming the session, so `ucx skill synthesize
    --session` built a skill out of nothing and reported it as distilled.

    Killed by: src/uclone_x/skills/synthesizer.py :: raise SkillAuditError(_NO_SESSION_STEPS.format(session_id=session_id))
    Becomes: return []
    """
    monkeypatch.chdir(tmp_path)  # no session folders here, so no candidate trace exists
    synth = SkillSynthesizer()

    with pytest.raises(SkillAuditError) as caught:
        synth.extract_workflow_from_session("sess_alpha_99")

    message = str(caught.value)
    assert "sess_alpha_99" in message
    assert "nothing to turn into a skill" in message
    for internal in ("/", "trace", "json", "Error", "Exception"):
        assert internal not in message, f"the refusal shows `{internal}`"


def test_extract_workflow_from_session(tmp_path: Path) -> None:
    """extract_workflow_from_session reads an on-disk session file."""
    synth = SkillSynthesizer()

    sess_file = tmp_path / "custom_sess.json"
    sess_file.write_text(json.dumps(["Step A", "Step B"]), encoding="utf-8")
    steps_disk = synth.extract_workflow_from_session("custom_sess", session_file=sess_file)
    assert steps_disk == ["Step A", "Step B"]


def test_the_instructions_say_when_to_use_the_skill_then_give_the_steps() -> None:
    """The `SKILL.md` body is a prompt, not code: a title, when to use it, the steps (#1810).

    Killed by: src/uclone_x/skills/synthesizer.py :: title = task_name.replace("_", " ").replace("-", " ").title()
    Becomes: title = task_name
    """
    body = SkillSynthesizer().format_instructions("db_indexer", ["Connect to replica", "Write DDL"])

    assert body == (
        "# Db Indexer\n\n"
        "Synthesized from a recorded session. It is instructions only and carries no script.\n\n"
        "## When to use\n\n"
        "When a task calls for the steps below, as the session this skill was recorded from did."
        "\n\n## Steps\n\n"
        "Follow these steps in order. They were recorded from one session, so adapt names, "
        "paths and values to the task at hand.\n\n"
        "1. Connect to replica\n"
        "2. Write DDL\n"
    )


#: Step text a recorded session can hold, and the one line of Markdown each becomes.
_AWKWARD_STEPS = {
    "<!-- hide --> keep": "\\<!-- hide --\\> keep",
    "# Not a heading": "\\# Not a heading",
    "one\nline\u2028two": "one line two",
    "1. not an item": "1\\. not an item",
    "- nor this": "\\- nor this",
    "```sh\nrm -rf /\n```": "\\`\\`\\`sh rm -rf / \\`\\`\\`",
    "a\u202ereversed": "areversed",
    "*bold* [link](http://e) a_b & |c|": "\\*bold\\* \\[link\\](http://e) a\\_b \\& \\|c\\|",
}


def test_step_text_cannot_hide_itself_or_pose_as_another_part_of_the_skill() -> None:
    """Each step is one line of escaped text, so the approver reads what the agent will (#1810).

    An HTML comment would hide text from the rendered view; a heading, a fence or a
    list marker would pose as another section or another step; a bidirectional override
    would make a step read differently from what it holds.

    Killed by: src/uclone_x/skills/synthesizer.py :: escaped = _MARKDOWN_PUNCTUATION.sub(r"\\\1", one_line(text))
    Becomes: escaped = one_line(text)

    Killed by: src/uclone_x/skills/synthesizer.py :: flat = _BREAKS.sub(" ", text)
    Becomes: flat = text

    Killed by: src/uclone_x/skills/synthesizer.py :: if unicodedata.category(ch) not in ("Cc", "Cf")
    Becomes: if unicodedata.category(ch) not in ("Cc",)

    Killed by: src/uclone_x/skills/synthesizer.py :: lambda m: f"{m[1]}\\{m[2]}" if m[1] is not None else f"\\{m[3]}", escaped
    Becomes: lambda m: m[0], escaped
    """
    body = SkillSynthesizer().format_instructions(
        "awkward", [*_AWKWARD_STEPS, "   ", "\x00"], "Use <b>it</b>\n# now"
    )

    steps = body.split("adapt names, paths and values to the task at hand.\n\n")[1]
    assert steps.splitlines() == [
        f"{i}. {line}" for i, line in enumerate(_AWKWARD_STEPS.values(), start=1)
    ]
    when = body.split("## When to use\n\n")[1].split("\n\n")[0]
    assert when == r"Use \<b\>it\</b\> \# now"


@pytest.mark.asyncio
async def test_synthesize_skill_quarantine_lifecycle(tmp_path: Path) -> None:
    """synthesize_skill writes one pending SKILL.md in quarantine with SHA-256 binding (P9).

    Killed by: src/uclone_x/skills/synthesizer.py :: save_skill(skill_dir, manifest, instructions)
    Becomes: save_skill(skill_dir, manifest, instructions); (skill_dir / "main.py").write_text("")
    """
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
    assert manifest.entrypoint is None
    assert manifest.scripts == ()
    assert manifest.tags == ("sql", "perf")
    assert manifest.content_sha256 is not None
    assert len(manifest.content_sha256) == 64

    # 2. The package is the one prompt-only file: no script, no second manifest (#1810)
    skill_dir = tmp_path / "db_indexer"
    assert sorted(p.name for p in skill_dir.iterdir()) == ["SKILL.md"]

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
async def test_a_description_holding_the_header_delimiter_round_trips(tmp_path: Path) -> None:
    """`---` inside the description no longer ends the `SKILL.md` header (#1826).

    The header ends only at a line that is exactly `---`, so the synthesizer need not refuse
    the text, and the written file reads back with the description whole.

    Killed by: src/uclone_x/skills/auditor.py :: (?P<yaml>.*?)^---
    Becomes: (?P<yaml>.*?)---
    """
    await SkillSynthesizer().synthesize_skill(
        task_name="cutter",
        workflow_steps=["Step 1"],
        quarantine_dir=tmp_path,
        description="Before --- after ---",
    )

    data, _ = parse_skill_markdown((tmp_path / "cutter" / "SKILL.md").read_text("utf-8"))
    assert data["description"] == "Before --- after ---"


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
