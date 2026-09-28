"""Skill synthesis from a recorded session (Principle 9).

Extracts the steps a session took from its trace or events and writes them into quarantine
as a prompt-only `SKILL.md` package (`status: pending`): when to use the skill, then the
steps as instructions, with no script (#1810, owner ruling 2026-09-27). The package does
nothing until a person approves it with `ucx skill approve`, which pins its digest.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from uclone_x.engine.event_bus import AgentEvent, EventType
from uclone_x.errors import SkillAuditError
from uclone_x.sandbox.models import IsolationLevel
from uclone_x.skills.auditor import (
    SkillAuditor,
    compute_skill_sha256,
    save_skill,
)
from uclone_x.skills.models import (
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.skills.protocols import SkillSynthesizerProtocol

__all__ = [
    "SkillSynthesizer",
    "one_line",
]

_IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_\-]+$")

#: What `ucx skill synthesize --session` says when the session left no steps to distil.
_NO_SESSION_STEPS = (
    "No recorded steps were found for session '{session_id}', so there is nothing to turn "
    "into a skill."
)


def _to_json_serializable(obj: object) -> Any:
    """Recursively convert MappingProxy and sequences into JSON-serializable primitives."""
    if isinstance(obj, Mapping):
        mapping_obj = cast(Mapping[object, object], obj)
        return {str(k): _to_json_serializable(v) for k, v in mapping_obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_json_serializable(x) for x in cast(Sequence[object], obj)]
    return obj


#: Characters that end a line, or a paragraph, in some reader: each becomes a space.
_BREAKS: re.Pattern[str] = re.compile(r"[\t\n\v\f\r\x1c-\x1f\x85\u2028\u2029]+")

#: Markdown punctuation that can start a heading, emphasis, a link, a code span, a table
#: cell, strikethrough, an entity or inline HTML (an HTML comment hides its text when the
#: package is shown rendered). Each is written with a backslash before it, which CommonMark
#: reads as the plain character.
_MARKDOWN_PUNCTUATION: re.Pattern[str] = re.compile(r"([\\`*_\[\]<>#|~&])")

#: A line that starts with one of these would open a list item or a setext underline.
_BLOCK_START: re.Pattern[str] = re.compile(r"^(\d+)([.)])|^([-+=])")

#: What a synthesized skill says about itself, under its title.
_SYNTHESIZED_NOTE = (
    "Synthesized from a recorded session. It is instructions only and carries no script."
)

#: The "When to use" text when the person gave no description.
_DEFAULT_WHEN_TO_USE = (
    "When a task calls for the steps below, as the session this skill was recorded from did."
)

#: The line before the steps.
_STEPS_LEAD = (
    "Follow these steps in order. They were recorded from one session, so adapt names, "
    "paths and values to the task at hand."
)


def one_line(text: str) -> str:
    """`text` as one line: breaks become spaces, control and format characters are dropped.

    Format characters (Unicode category Cf) include the bidirectional overrides that can
    make text read differently from what it holds.
    """
    flat = _BREAKS.sub(" ", text)
    kept = "".join(ch for ch in flat if unicodedata.category(ch) not in ("Cc", "Cf"))
    return " ".join(kept.split())


def _markdown_text(text: str) -> str:
    """`text` as one line of Markdown that reads as the plain text it holds."""
    escaped = _MARKDOWN_PUNCTUATION.sub(r"\\\1", one_line(text))
    return _BLOCK_START.sub(
        lambda m: f"{m[1]}\\{m[2]}" if m[1] is not None else f"\\{m[3]}", escaped
    )


class SkillSynthesizer(SkillSynthesizerProtocol):
    """Skill distillation implementing SkillSynthesizerProtocol (P9).

    Extracts the steps from a session's trace or turn events and writes them as a
    quarantined, prompt-only `SKILL.md` package (`status: pending`); `synthesize_and_audit`
    also runs the `SkillAuditor` over it.
    """

    def __init__(
        self,
        default_isolation: IsolationLevel = IsolationLevel.WORKSPACE,
        synthesizer_version: str = "0.1.0",
        auditor: SkillAuditor | None = None,
    ) -> None:
        self._default_isolation = default_isolation
        self._synthesizer_version = synthesizer_version
        self._auditor = auditor or SkillAuditor()

    @property
    def default_isolation(self) -> IsolationLevel:
        """Default isolation level requested for synthesized skills."""
        return self._default_isolation

    @property
    def synthesizer_version(self) -> str:
        """Version of the skill synthesizer."""
        return self._synthesizer_version

    @property
    def auditor(self) -> SkillAuditor:
        """Security auditor instance."""
        return self._auditor

    def extract_workflow_from_events(
        self,
        events: Sequence[AgentEvent | dict[str, Any]],
    ) -> list[str]:
        """Extract sequential workflow steps from a list of AgentEvents or event dicts."""
        steps: list[str] = []

        for item in events:
            if isinstance(item, AgentEvent):
                ev_type = item.type
                payload: dict[str, Any] = dict(item.payload)
                dict_item: dict[str, Any] | None = None
            else:
                dict_item = item
                raw_type = dict_item.get("type", "UNKNOWN")
                try:
                    ev_type = EventType(str(raw_type))
                except ValueError:
                    ev_type = None
                raw_payload: object = dict_item.get("payload", dict_item)
                if isinstance(raw_payload, Mapping):
                    payload = {
                        str(k): v for k, v in cast(Mapping[object, object], raw_payload).items()
                    }
                else:
                    payload = {}

            if ev_type is EventType.TOOL_CALL:
                tool_name = str(
                    payload.get("tool_name")
                    or payload.get("name")
                    or payload.get("tool")
                    or "generic_tool"
                )
                args_obj: object = payload.get("arguments") or payload.get("args") or {}
                serializable_args = _to_json_serializable(args_obj)
                args_summary = (
                    json.dumps(serializable_args, sort_keys=True) if serializable_args else "{}"
                )
                steps.append(f"Execute tool '{tool_name}' with arguments: {args_summary}")

            elif ev_type is EventType.TOOL_RESULT:
                tool_name = str(
                    payload.get("tool_name") or payload.get("name") or payload.get("tool") or "tool"
                )
                status = "success" if payload.get("success", True) else "failed"
                steps.append(f"Verify result of tool '{tool_name}' (status: {status})")

            elif ev_type is EventType.USER_INPUT:
                msg = str(payload.get("message") or payload.get("content") or "").strip()
                if msg:
                    summary = msg if len(msg) <= 80 else f"{msg[:77]}..."
                    steps.append(f"Process user task intent: '{summary}'")

            elif ev_type is EventType.AGENT_REPLY:
                content = str(payload.get("content") or "").strip()
                if content:
                    summary = content if len(content) <= 80 else f"{content[:77]}..."
                    steps.append(f"Synthesize reasoning stage result: '{summary}'")

            elif ev_type is EventType.SUBAGENT_SPAWN:
                role = str(payload.get("role") or "subagent")
                goal = str(payload.get("goal") or "subtask")
                steps.append(f"Delegate task to subagent '{role}' with goal: '{goal}'")

            elif dict_item is not None and "step" in dict_item:
                steps.append(str(dict_item["step"]))
            elif dict_item is not None and "action" in dict_item:
                steps.append(str(dict_item["action"]))

        if not steps:
            raise SkillAuditError(
                "No actionable workflow steps could be extracted from the provided events."
            )

        return steps

    def extract_workflow_from_trace(self, trace_source: Path | str) -> list[str]:
        """Extract sequential workflow steps from a trace file (JSON, YAML, or plain text)."""
        trace_path = Path(trace_source)
        if not trace_path.exists() or not trace_path.is_file():
            raise SkillAuditError(f"Trace file does not exist or is not a file: {trace_path}")

        raw_content = trace_path.read_text(encoding="utf-8").strip()
        if not raw_content:
            raise SkillAuditError(f"Trace file is empty: {trace_path}")

        # Try parsing as JSON
        try:
            parsed_json: object = json.loads(raw_content)
            return self._extract_from_parsed_structure(parsed_json)
        except json.JSONDecodeError:
            pass

        # Try parsing as YAML
        try:
            parsed_yaml: object = yaml.safe_load(raw_content)
            if isinstance(parsed_yaml, (dict, list)):
                return self._extract_from_parsed_structure(cast(object, parsed_yaml))
        except yaml.YAMLError:
            pass

        # Fallback to plain text lines
        lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
        if not lines:
            raise SkillAuditError(f"No valid workflow steps found in trace file: {trace_path}")
        return lines

    def _extract_from_parsed_structure(self, data: object) -> list[str]:
        """Helper to extract workflow steps from parsed JSON/YAML data structures."""
        if isinstance(data, list):
            # Check if list of strings
            data_list = cast(list[object], data)
            if all(isinstance(x, str) for x in data_list):
                return [str(x) for x in data_list if str(x).strip()]
            # List of dicts (events or steps)
            dict_list: list[dict[str, Any]] = [
                cast(dict[str, Any], x) for x in data_list if isinstance(x, dict)
            ]
            if dict_list:
                return self.extract_workflow_from_events(dict_list)

        elif isinstance(data, dict):
            dict_data = cast(dict[str, Any], data)
            for key in (
                "workflow_steps",
                "steps",
                "workflow",
                "events",
                "tool_calls",
                "turns",
                "spans",
                "messages",
            ):
                if key in dict_data and isinstance(dict_data[key], list):
                    return self._extract_from_parsed_structure(dict_data[key])

        raise SkillAuditError("Trace data structure contains no identifiable workflow steps.")

    def extract_workflow_from_session(
        self,
        session_id: str,
        session_file: Path | None = None,
    ) -> list[str]:
        """Extract workflow steps for a session from its trace file.

        Raises:
            SkillAuditError: No trace file is found for the session.
        """
        if session_file is not None and session_file.is_file():
            return self.extract_workflow_from_trace(session_file)

        # Look for standard session trace file locations
        candidates = [
            Path(f".uclone_x/sessions/{session_id}.json"),
            Path(f"sessions/{session_id}.json"),
            Path(f".uclone_x/traces/{session_id}.json"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return self.extract_workflow_from_trace(candidate)

        # No trace, no steps. This used to return three generic steps naming the session,
        # which synthesized a skill out of nothing and reported it as distilled (P6).
        raise SkillAuditError(_NO_SESSION_STEPS.format(session_id=session_id))

    def format_instructions(
        self,
        task_name: str,
        workflow_steps: Sequence[str],
        when_to_use: str | None = None,
        *,
        note: str = _SYNTHESIZED_NOTE,
        steps_lead: str = _STEPS_LEAD,
    ) -> str:
        """The `SKILL.md` body: when to use the skill, then the steps as instructions.

        `note` (the line under the title) and `steps_lead` (the line before the steps) are
        fixed text the caller owns, not session text, and are written as given: a clone's
        proposal (`uclone_x.skills.proposals`) says it was proposed rather than recorded.

        Every step is free text from a recorded session, so each is written as one line of
        escaped Markdown text (`_markdown_text`): it cannot open a heading, a code block, an
        HTML comment or a new list item, and so cannot hide text from the person reading
        the package before approving it, or pose as another section of it.

        Raises:
            ValueError: No step has any text left once it is made one plain line.
        """
        steps = [text for text in (_markdown_text(step) for step in workflow_steps) if text]
        if not steps:
            raise ValueError("workflow_steps cannot be empty.")
        title = task_name.replace("_", " ").replace("-", " ").title()
        when = _markdown_text(when_to_use or "") or _DEFAULT_WHEN_TO_USE
        steps_list = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
        return (
            f"# {_markdown_text(title)}\n\n"
            f"{note}\n\n"
            f"## When to use\n\n{when}\n\n"
            f"## Steps\n\n{steps_lead}\n\n{steps_list}\n"
        )

    async def synthesize_skill(
        self,
        task_name: str,
        workflow_steps: list[str],
        quarantine_dir: Path,
        description: str | None = None,
        tags: tuple[str, ...] | None = None,
        requested_isolation: IsolationLevel | None = None,
        author: str | None = None,
    ) -> SkillManifest:
        """Write a prompt-only `SKILL.md` package into quarantine and return its manifest.

        The package is the one file `SKILL.md`: its frontmatter says `origin: synthesized`
        and `status: pending`, and its body says when to use the skill and gives the steps
        as instructions (`format_instructions`). It carries no script (#1810). A pending
        package is inert: it becomes active only through `ucx skill approve`, which pins its
        digest in the approvals ledger.

        Raises:
            ValueError: The name is not an identifier, or no step has any text.
        """
        clean_name = task_name.strip().lower()
        if not clean_name or not _IDENTIFIER_PATTERN.match(clean_name):
            raise ValueError(
                f"Invalid skill name '{task_name}'. Name must be non-empty and alphanumeric "
                f"with underscores or hyphens."
            )

        # A `---` inside the description is safe: the header ends only at a line that is
        # exactly `---` (#1826), and `one_line` leaves the description one line.
        given = one_line(description or "")
        instructions = self.format_instructions(clean_name, workflow_steps, given)

        # Determine target skill directory
        if quarantine_dir.name == clean_name:
            skill_dir = quarantine_dir
        else:
            skill_dir = quarantine_dir / clean_name

        skill_dir.mkdir(parents=True, exist_ok=True)

        manifest = SkillManifest(
            name=clean_name,
            description=given or f"Synthesized from a recorded session: {clean_name}",
            version="0.1.0",
            author=author or "agent:synthesizer",
            origin=SkillOrigin.SYNTHESIZED,
            status=SkillStatus.PENDING,
            requested_isolation=requested_isolation or self._default_isolation,
            tags=tags or ("synthesized", clean_name),
        )
        save_skill(skill_dir, manifest, instructions)

        content_hash = compute_skill_sha256(skill_dir)
        return manifest.model_copy(update={"content_sha256": content_hash})

    async def synthesize_and_audit(
        self,
        task_name: str,
        workflow_steps: list[str],
        quarantine_dir: Path,
        description: str | None = None,
        tags: tuple[str, ...] | None = None,
        requested_isolation: IsolationLevel | None = None,
        author: str | None = None,
    ) -> tuple[SkillManifest, SkillAuditReport]:
        """Synthesize a quarantined skill and perform immediate security audit via SkillAuditor."""
        manifest = await self.synthesize_skill(
            task_name=task_name,
            workflow_steps=workflow_steps,
            quarantine_dir=quarantine_dir,
            description=description,
            tags=tags,
            requested_isolation=requested_isolation,
            author=author,
        )
        skill_dir = (
            quarantine_dir
            if quarantine_dir.name == manifest.name
            else quarantine_dir / manifest.name
        )
        report = await self._auditor.audit_skill(skill_dir)
        return manifest, report
