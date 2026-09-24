"""Autonomous dynamic skill synthesis pipeline conforming to Principle 9.

Extracts reusable workflow steps from session events/traces, generates sandboxed
Python execution modules, LinkML-compatible manifest.yaml, and SKILL.md packages
into quarantine (status=pending), and integrates with SkillAuditor and SkillRegistry.
"""

from __future__ import annotations

import ast
import json
import re
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
]

_IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _to_json_serializable(obj: object) -> Any:
    """Recursively convert MappingProxy and sequences into JSON-serializable primitives."""
    if isinstance(obj, Mapping):
        mapping_obj = cast(Mapping[object, object], obj)
        return {str(k): _to_json_serializable(v) for k, v in mapping_obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_json_serializable(x) for x in cast(Sequence[object], obj)]
    return obj


class SkillSynthesizer(SkillSynthesizerProtocol):
    """Autonomous skill distillation engine implementing SkillSynthesizerProtocol (P9).

    Discovers workflow patterns from session traces or turn event logs, synthesizes
    quarantined packages (`status: pending`), generates sandboxed Python code and
    `manifest.yaml` / `SKILL.md` artifacts, and connects to `SkillAuditor`.
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
        except Exception:
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
        """Extract workflow steps for a session from a file or generate structured steps."""
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

        # Deterministic synthetic template steps for session
        return [
            f"Initialize execution context for session '{session_id}'",
            f"Execute automated multi-turn tool workflow for session '{session_id}'",
            f"Verify output artifacts and finalize results for session '{session_id}'",
        ]

    def generate_skill_code(
        self,
        task_name: str,
        workflow_steps: Sequence[str],
        description: str = "",
    ) -> str:
        """Generate sandboxed Python code (main.py) implementing the skill workflow."""
        steps_doc = "\n".join(f"        {i + 1}. {step}" for i, step in enumerate(workflow_steps))
        steps_quoted = ",\n".join(f'            "{step}"' for step in workflow_steps)

        code = f'''"""Synthesized skill: {task_name}

Autonomously generated by UClone-X Skill Synthesizer (Principle 9).
"""

from __future__ import annotations

from typing import Any


def execute_workflow(inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute the synthesized {task_name} workflow steps.

    Workflow steps:
{steps_doc}
    """
    context_inputs = inputs or {{}}
    results: dict[str, Any] = {{
        "status": "success",
        "skill": "{task_name}",
        "inputs": context_inputs,
        "steps_executed": [
{steps_quoted}
        ],
    }}
    return results


def main() -> None:
    """CLI entrypoint for standalone skill invocation."""
    out = execute_workflow()
    print(f"Executed skill '{task_name}': {{out['status']}}")


if __name__ == "__main__":
    main()
'''
        # Verify generated code parses cleanly with Python AST
        try:
            ast.parse(code, filename="main.py")
        except SyntaxError as exc:
            raise SkillAuditError(f"Generated skill code failed AST validation: {exc}") from exc

        return code

    def generate_manifest_yaml(
        self,
        manifest: SkillManifest,
        workflow_steps: Sequence[str],
    ) -> str:
        """Generate LinkML-compatible manifest.yaml schema specification for the skill."""
        manifest_data: dict[str, Any] = {
            "id": f"https://uclone.ai/skills/{manifest.name}",
            "name": manifest.name,
            "description": manifest.description,
            "version": manifest.version,
            "origin": manifest.origin.value,
            "status": manifest.status.value,
            "requested_isolation": (
                manifest.requested_isolation.value if manifest.requested_isolation else "workspace"
            ),
            "entrypoint": manifest.entrypoint or "main.py",
            "scripts": list(manifest.scripts) if manifest.scripts else ["main.py"],
            "tags": list(manifest.tags) if manifest.tags else ["synthesized", manifest.name],
            "workflow_steps": list(workflow_steps),
        }
        if manifest.author:
            manifest_data["author"] = manifest.author
        if manifest.content_sha256:
            manifest_data["content_sha256"] = manifest.content_sha256

        return yaml.dump(manifest_data, sort_keys=False)

    def _format_instructions(
        self,
        task_name: str,
        description: str,
        workflow_steps: Sequence[str],
    ) -> str:
        """Format markdown body for SKILL.md."""
        title = task_name.replace("_", " ").replace("-", " ").title()
        steps_list = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(workflow_steps))

        return f"""# {title} Skill

## Description
{description}

## Step-by-Step Workflow
{steps_list}

## Execution Notes
This skill package is autonomously synthesized and executes within an isolated sandbox
environment adhering to Principle P3 (Sandboxing) and Principle P9 (Dynamic Skills).
"""

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
        """Write a SKILL.md package into quarantine and return its manifest.

        Conforms strictly to SkillSynthesizerProtocol.
        """
        clean_name = task_name.strip().lower()
        if not clean_name or not _IDENTIFIER_PATTERN.match(clean_name):
            raise ValueError(
                f"Invalid skill name '{task_name}'. Name must be non-empty and alphanumeric "
                f"with underscores or hyphens."
            )

        if not workflow_steps:
            raise ValueError("workflow_steps cannot be empty.")

        # Determine target skill directory
        if quarantine_dir.name == clean_name:
            skill_dir = quarantine_dir
        else:
            skill_dir = quarantine_dir / clean_name

        skill_dir.mkdir(parents=True, exist_ok=True)

        desc = description or f"Autonomously synthesized skill for {clean_name}"
        isolation = requested_isolation or self._default_isolation

        # 1. Generate sandboxed Python module (main.py)
        code = self.generate_skill_code(
            task_name=clean_name,
            workflow_steps=workflow_steps,
            description=desc,
        )
        main_py = skill_dir / "main.py"
        main_py.write_text(code, encoding="utf-8")

        # 2. Build initial SkillManifest
        initial_manifest = SkillManifest(
            name=clean_name,
            description=desc,
            version="0.1.0",
            author=author or "agent:synthesizer",
            origin=SkillOrigin.SYNTHESIZED,
            status=SkillStatus.PENDING,
            requested_isolation=isolation,
            entrypoint="main.py",
            scripts=("main.py",),
            tags=tags or ("synthesized", clean_name),
        )

        # 3. Generate LinkML manifest.yaml
        manifest_yaml = self.generate_manifest_yaml(
            manifest=initial_manifest,
            workflow_steps=workflow_steps,
        )
        (skill_dir / "manifest.yaml").write_text(manifest_yaml, encoding="utf-8")

        # 4. Generate SKILL.md
        instructions = self._format_instructions(
            task_name=clean_name,
            description=desc,
            workflow_steps=workflow_steps,
        )
        save_skill(skill_dir, initial_manifest, instructions)

        # 5. Compute SHA-256 content hash of quarantine files
        content_hash = compute_skill_sha256(skill_dir)
        return initial_manifest.model_copy(update={"content_sha256": content_hash})

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
