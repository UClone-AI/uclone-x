"""Skill Auditor security verification and registry implementation.

Implements SkillAuditorProtocol and SkillRegistryProtocol with fail-closed
security evaluation, AST static analysis, and quarantine lifecycle gating.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, cast

import yaml

from uclone_x.errors import SkillAuditError, SkillNotApprovedError
from uclone_x.sandbox.models import (
    AVAILABLE_ISOLATION_LEVELS,
    IsolationLevel,
    is_weaker_isolation,
)
from uclone_x.skills.models import (
    AuditVerdict,
    AutoApprovalPolicy,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.skills.protocols import SkillAuditorProtocol, SkillProtocol

logger = logging.getLogger(__name__)

__all__ = [
    "Skill",
    "SkillAuditor",
    "SkillRegistry",
    "compute_skill_sha256",
    "load_skill_from_dir",
    "manifest_from_dict",
    "parse_skill_markdown",
    "save_skill",
    "serialize_skill_markdown",
]

# Critical dangerous function calls / builtins
DANGEROUS_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "__import__",
        "compile",
        "globals",
        "locals",
    }
)

# Dangerous OS/subprocess functions
DANGEROUS_OS_CALLS: frozenset[str] = frozenset(
    {
        "system",
        "popen",
        "popen2",
        "popen3",
        "popen4",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "kill",
        "killpg",
        "remove",
        "unlink",
        "rmdir",
    }
)

# Dangerous subprocess methods
DANGEROUS_SUBPROCESS_CALLS: frozenset[str] = frozenset(
    {
        "call",
        "check_call",
        "check_output",
        "run",
        "Popen",
    }
)

# Dangerous modules
DANGEROUS_MODULES: frozenset[str] = frozenset(
    {
        "pty",
        "ctypes",
    }
)

# Network modules (medium risk if not sandboxed)
NETWORK_MODULES: frozenset[str] = frozenset(
    {
        "socket",
        "http.client",
        "urllib.request",
        "requests",
        "httpx",
        "aiohttp",
    }
)

# Prompt injection patterns
PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard all previous instructions",
    "disregard previous instructions",
    "disregard system prompt",
    "override system prompt",
    "bypass security policy",
    "bypass safety filters",
    "<system>",
    "</system>",
    "[system_prompt]",
    "[/system_prompt]",
)

# Shell command escalation patterns
DANGEROUS_SHELL_PATTERNS: tuple[str, ...] = (
    "rm -rf",
    "sudo ",
    "curl ",
    "wget ",
    "chmod +x",
    "nc -e",
    "dd if=",
    ":(){ :|:& };:",
)


class Skill:
    """Concrete implementation of SkillProtocol representing a loaded modular skill package."""

    def __init__(
        self,
        manifest: SkillManifest,
        instructions_markdown: str,
        directory: Path | None = None,
    ) -> None:
        self._manifest = manifest
        self._instructions_markdown = instructions_markdown
        self._directory = directory

    @property
    def manifest(self) -> SkillManifest:
        """Skill metadata and frontmatter."""
        return self._manifest

    @property
    def instructions_markdown(self) -> str:
        """Markdown procedural knowledge."""
        return self._instructions_markdown

    @property
    def directory(self) -> Path | None:
        """Skill package root directory on disk, if loaded from disk."""
        return self._directory


def parse_skill_markdown(text: str) -> tuple[dict[str, Any], str]:
    """Parse a SKILL.md file into frontmatter dictionary and markdown body."""
    if not text.startswith("---"):
        raise ValueError("Document has no leading YAML frontmatter block starting with '---'")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Document frontmatter block is not closed with '---'")
    yaml_content = parts[1]
    instructions = parts[2].lstrip()
    raw_data: object = yaml.safe_load(yaml_content)
    if not isinstance(raw_data, dict):
        raise ValueError("YAML frontmatter must be a mapping/dictionary")
    raw_dict = cast(dict[object, object], raw_data)
    data_dict: dict[str, Any] = {str(k): v for k, v in raw_dict.items()}
    return data_dict, instructions


def manifest_from_dict(data: dict[str, Any]) -> SkillManifest:
    """Construct a SkillManifest from a dictionary extracted from frontmatter."""
    unknown_keys = set(data.keys()) - set(SkillManifest.model_fields.keys())
    if unknown_keys:
        raise ValueError(
            f"Unknown field(s) in SKILL.md frontmatter: {sorted(unknown_keys)}. "
            f"Allowed fields are: {sorted(SkillManifest.model_fields.keys())}"
        )

    name = data.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("Skill manifest is missing required 'name' field")

    origin_val = data.get("origin", SkillOrigin.SYNTHESIZED)
    if isinstance(origin_val, str):
        origin = SkillOrigin(origin_val)
    elif isinstance(origin_val, SkillOrigin):
        origin = origin_val
    else:
        origin = SkillOrigin.SYNTHESIZED

    status_val = data.get("status", SkillStatus.PENDING)
    if isinstance(status_val, str):
        status = SkillStatus(status_val)
    elif isinstance(status_val, SkillStatus):
        status = status_val
    else:
        status = SkillStatus.PENDING

    isolation_val = data.get("requested_isolation")
    if isolation_val is not None and isinstance(isolation_val, str):
        requested_isolation = IsolationLevel(isolation_val)
    elif isinstance(isolation_val, IsolationLevel):
        requested_isolation = isolation_val
    else:
        requested_isolation = None

    scripts_val: object = data.get("scripts")
    scripts_list: list[str] = []
    if isinstance(scripts_val, list):
        for item in cast(list[object], scripts_val):
            scripts_list.append(str(item))
    elif isinstance(scripts_val, tuple):
        for item in cast(tuple[object, ...], scripts_val):
            scripts_list.append(str(item))

    tags_val: object = data.get("tags")
    tags_list: list[str] = []
    if isinstance(tags_val, list):
        for item in cast(list[object], tags_val):
            tags_list.append(str(item))
    elif isinstance(tags_val, tuple):
        for item in cast(tuple[object, ...], tags_val):
            tags_list.append(str(item))

    return SkillManifest(
        name=name,
        description=str(data.get("description", "")),
        version=str(data.get("version", "0.1.0")),
        author=str(data["author"]) if data.get("author") else None,
        origin=origin,
        status=status,
        requested_isolation=requested_isolation,
        scripts=tuple(scripts_list),
        tags=tuple(tags_list),
        entrypoint=str(data["entrypoint"]) if data.get("entrypoint") else None,
        content_sha256=str(data["content_sha256"]) if data.get("content_sha256") else None,
        approved_by=str(data["approved_by"]) if data.get("approved_by") else None,
        approved_at=str(data["approved_at"]) if data.get("approved_at") else None,
        rejected_by=str(data["rejected_by"]) if data.get("rejected_by") else None,
        rejected_at=str(data["rejected_at"]) if data.get("rejected_at") else None,
        rejection_reason=str(data["rejection_reason"]) if data.get("rejection_reason") else None,
    )


def compute_skill_sha256(skill_dir: Path) -> str:
    """Compute a deterministic SHA-256 digest of all files in a skill package."""
    hasher = hashlib.sha256()
    if skill_dir.is_file():
        hasher.update(skill_dir.read_bytes())
        return hasher.hexdigest()

    if not skill_dir.exists() or not skill_dir.is_dir():
        raise SkillAuditError(f"Cannot compute hash for invalid directory: {skill_dir}")

    for path in sorted(skill_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            rel_path = path.relative_to(skill_dir).as_posix()
            hasher.update(rel_path.encode("utf-8"))
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def serialize_skill_markdown(manifest: SkillManifest, instructions: str) -> str:
    """Serialize a SkillManifest and instructions markdown into standard SKILL.md format."""
    data: dict[str, Any] = {
        "name": manifest.name,
        "description": manifest.description,
        "version": manifest.version,
    }
    if manifest.author:
        data["author"] = manifest.author
    data["origin"] = manifest.origin.value
    data["status"] = manifest.status.value
    if manifest.requested_isolation is not None:
        data["requested_isolation"] = manifest.requested_isolation.value
    if manifest.entrypoint:
        data["entrypoint"] = manifest.entrypoint
    if manifest.scripts:
        data["scripts"] = list(manifest.scripts)
    if manifest.tags:
        data["tags"] = list(manifest.tags)
    if manifest.content_sha256:
        data["content_sha256"] = manifest.content_sha256
    if manifest.approved_by:
        data["approved_by"] = manifest.approved_by
    if manifest.approved_at:
        data["approved_at"] = manifest.approved_at
    if manifest.rejected_by:
        data["rejected_by"] = manifest.rejected_by
    if manifest.rejected_at:
        data["rejected_at"] = manifest.rejected_at
    if manifest.rejection_reason:
        data["rejection_reason"] = manifest.rejection_reason

    yaml_str = yaml.dump(data, sort_keys=False)
    instructions_clean = instructions.strip()
    if instructions_clean:
        return f"---\n{yaml_str}---\n\n{instructions_clean}\n"
    return f"---\n{yaml_str}---\n"


def load_skill_from_dir(skill_dir: Path) -> Skill:
    """Load a skill package from a directory containing SKILL.md."""
    if not skill_dir.exists() or not skill_dir.is_dir():
        raise SkillAuditError(f"Skill directory '{skill_dir}' does not exist or is not a directory")

    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        raise SkillAuditError(f"Missing required 'SKILL.md' in '{skill_dir}'")

    text = skill_file.read_text(encoding="utf-8")
    try:
        data, instructions = parse_skill_markdown(text)
        manifest = manifest_from_dict(data)
    except Exception as exc:
        raise SkillAuditError(f"Failed to parse skill package in '{skill_dir}': {exc}") from exc

    return Skill(manifest=manifest, instructions_markdown=instructions, directory=skill_dir)


def save_skill(skill_dir: Path, manifest: SkillManifest, instructions: str) -> None:
    """Save/update a skill package SKILL.md in the given directory."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    content = serialize_skill_markdown(manifest, instructions)
    skill_file.write_text(content, encoding="utf-8")


class _PythonASTSecurityVisitor(ast.NodeVisitor):
    """AST visitor inspecting Python source code for security violations and high-risk operations."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.critical_risks: list[str] = []
        self.medium_risks: list[str] = []
        self.low_risks: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name
            if name in DANGEROUS_MODULES:
                self.critical_risks.append(
                    f"Dangerous module import '{name}' in {self.filename}:{node.lineno}"
                )
            elif name in NETWORK_MODULES:
                self.medium_risks.append(
                    f"Network module import '{name}' in {self.filename}:{node.lineno}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            if node.module in DANGEROUS_MODULES:
                self.critical_risks.append(
                    f"Dangerous module import '{node.module}' in {self.filename}:{node.lineno}"
                )
            elif node.module in NETWORK_MODULES:
                self.medium_risks.append(
                    f"Network module import '{node.module}' in {self.filename}:{node.lineno}"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct calls (e.g. eval(), exec())
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in DANGEROUS_CALLS:
                self.critical_risks.append(
                    f"Dangerous builtin function call '{func_name}()' in {self.filename}:{node.lineno}"
                )

        # Check attribute calls (e.g. os.system(), subprocess.run(), shutil.rmtree())
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            # Check os.system, os.popen, etc.
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                if module_name == "os" and attr_name in DANGEROUS_OS_CALLS:
                    self.critical_risks.append(
                        f"Dangerous OS call 'os.{attr_name}()' in {self.filename}:{node.lineno}"
                    )
                elif module_name == "subprocess" and attr_name in DANGEROUS_SUBPROCESS_CALLS:
                    self.medium_risks.append(
                        f"Process execution 'subprocess.{attr_name}()' in {self.filename}:{node.lineno}"
                    )
                elif module_name == "shutil" and attr_name == "rmtree":
                    self.critical_risks.append(
                        f"Recursive deletion 'shutil.rmtree()' in {self.filename}:{node.lineno}"
                    )

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            val_lower = node.value.lower()
            for pattern in DANGEROUS_SHELL_PATTERNS:
                if pattern in val_lower:
                    self.critical_risks.append(
                        f"Dangerous shell command pattern '{pattern}' in {self.filename}:{node.lineno}"
                    )
        self.generic_visit(node)


class SkillAuditor:
    """Security auditor for dynamic skills, implementing SkillAuditorProtocol.

    Enforces fail-closed evaluation, AST static analysis, isolation ceiling checks,
    and configurable auto-approval policies (P9 / Issue 2026-09-02-002 / 2026-09-02-041).
    """

    def __init__(
        self,
        policy: AutoApprovalPolicy = AutoApprovalPolicy.SAFE_ONLY,
        isolation_floor: IsolationLevel = IsolationLevel.WORKSPACE,
        auditor_version: str = "0.1.0",
        available_levels: frozenset[IsolationLevel] = AVAILABLE_ISOLATION_LEVELS,
    ) -> None:
        if isolation_floor not in available_levels:
            raise SkillAuditError(
                f"Configured isolation_floor '{isolation_floor.value}' has no available backend runner "
                f"(available: {sorted(lvl.value for lvl in available_levels)})"
            )
        self._policy = policy
        self._isolation_floor = isolation_floor
        self._auditor_version = auditor_version
        self._available_levels = available_levels

    @property
    def policy(self) -> AutoApprovalPolicy:
        """Current auto-approval mode."""
        return self._policy

    @property
    def isolation_floor(self) -> IsolationLevel:
        """The weakest isolation a skill may run under, decided by the runtime."""
        return self._isolation_floor

    async def audit_skill(self, skill_dir: Path) -> SkillAuditReport:
        """Audit a skill package before activation.

        Performs fail-closed static AST analysis, prompt injection detection,
        isolation clamping checks, and policy enforcement.
        """
        return await asyncio.to_thread(self._audit_sync, skill_dir)

    def _audit_sync(self, skill_dir: Path) -> SkillAuditReport:
        if not skill_dir.exists() or not skill_dir.is_dir():
            raise SkillAuditError(f"Skill directory does not exist: {skill_dir}")

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            raise SkillAuditError(f"Missing required SKILL.md in {skill_dir}")

        content_sha256 = compute_skill_sha256(skill_dir)
        skill = load_skill_from_dir(skill_dir)
        manifest = skill.manifest

        critical_risks: list[str] = []
        medium_risks: list[str] = []
        low_risks: list[str] = []

        # 1. Isolation Policy Check (P3 / Issue 2026-09-02-002, #62)
        # Check if requested isolation has an available runner backend
        if (
            manifest.requested_isolation is not None
            and manifest.requested_isolation not in self._available_levels
        ):
            critical_risks.append(
                f"Requested isolation level '{manifest.requested_isolation.value}' has no available backend runner "
                f"(available: {sorted(lvl.value for lvl in self._available_levels)})"
            )

        # Synthesized skills cannot grant themselves host execution (IsolationLevel.NONE)
        if manifest.origin == SkillOrigin.SYNTHESIZED:
            if manifest.requested_isolation is IsolationLevel.NONE:
                critical_risks.append(
                    "Synthesized skill requested unisolated host execution (IsolationLevel.NONE)"
                )
            if manifest.requested_isolation is not None and is_weaker_isolation(
                manifest.requested_isolation, self._isolation_floor
            ):
                medium_risks.append(
                    f"Requested isolation '{manifest.requested_isolation.value}' is weaker "
                    f"than runtime floor '{self._isolation_floor.value}'"
                )

        # 2. Prompt Injection Check on markdown instructions
        instructions_lower = skill.instructions_markdown.lower()
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern in instructions_lower:
                critical_risks.append(
                    f"Prompt injection / override pattern detected in instructions: '{pattern}'"
                )

        for pattern in DANGEROUS_SHELL_PATTERNS:
            if pattern in instructions_lower:
                critical_risks.append(
                    f"Dangerous shell command pattern detected in instructions: '{pattern}'"
                )

        # 3. Static AST Analysis on all Python scripts in the package
        for py_file in skill_dir.rglob("*.py"):
            if py_file.is_file():
                try:
                    code = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(code, filename=py_file.name)
                    visitor = _PythonASTSecurityVisitor(filename=py_file.name)
                    visitor.visit(tree)
                    critical_risks.extend(visitor.critical_risks)
                    medium_risks.extend(visitor.medium_risks)
                    low_risks.extend(visitor.low_risks)
                except SyntaxError as exc:
                    critical_risks.append(f"Syntax error in script '{py_file.name}': {exc}")
                except Exception as exc:
                    critical_risks.append(f"Failed to analyze script '{py_file.name}': {exc}")

        all_risks = tuple(critical_risks + medium_risks + low_risks)

        # 4. Calculate Risk Score (0.0 to 1.0)
        risk_score: float = 0.0
        if critical_risks:
            risk_score = min(1.0, 0.8 + (len(critical_risks) - 1) * 0.1)
        elif medium_risks:
            risk_score = min(0.7, 0.4 + (len(medium_risks) - 1) * 0.1)
        elif low_risks:
            risk_score = min(0.3, 0.1 * len(low_risks))
        else:
            risk_score = 0.0

        # 5. Determine Verdict and Safety according to Policy
        is_safe: bool
        verdict: AuditVerdict

        if critical_risks or risk_score >= 0.7:
            is_safe = False
            verdict = AuditVerdict.REJECT
        elif medium_risks or risk_score >= 0.2:
            is_safe = False
            verdict = AuditVerdict.REQUIRE_HUMAN_REVIEW
        else:
            # Clean skill with low/zero risk
            if self._policy is AutoApprovalPolicy.NEVER:
                # NEVER auto-approve policy: safe skills still require explicit human review
                is_safe = True
                verdict = AuditVerdict.REQUIRE_HUMAN_REVIEW
            else:
                # SAFE_ONLY or ALWAYS
                is_safe = True
                verdict = AuditVerdict.APPROVE

        return SkillAuditReport(
            skill_name=manifest.name,
            is_safe=is_safe,
            recommendation=verdict,
            risk_score=risk_score,
            detected_risks=all_risks,
            auditor_version=self._auditor_version,
            content_sha256=content_sha256,
        )


class SkillRegistry:
    """Registry for hot-reloading skill discovery and runtime management.

    Implements SkillRegistryProtocol with quarantine enforcement.
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._skills_dir = skills_dir
        self._skills: dict[str, SkillProtocol] = {}
        self._audit_reports: dict[str, SkillAuditReport] = {}

    def register(self, skill: SkillProtocol, report: SkillAuditReport) -> None:
        """Register a skill, admitting it only on a passing audit of *this* code."""
        if report.skill_name != skill.manifest.name:
            raise SkillNotApprovedError(
                f"Audit report for '{report.skill_name}' does not match skill '{skill.manifest.name}'"
            )

        if not report.content_sha256 or not skill.manifest.content_sha256:
            raise SkillNotApprovedError(
                f"Missing content hash binding for skill '{skill.manifest.name}': "
                f"report={report.content_sha256}, manifest={skill.manifest.content_sha256}"
            )

        if report.content_sha256 != skill.manifest.content_sha256:
            raise SkillNotApprovedError(
                f"Audit report content hash '{report.content_sha256}' does not match "
                f"skill content hash '{skill.manifest.content_sha256}'"
            )

        if report.is_safe and report.recommendation is AuditVerdict.APPROVE:
            self._skills[skill.manifest.name] = skill
            self._audit_reports[skill.manifest.name] = report
            return

        raise SkillNotApprovedError(
            f"Skill '{skill.manifest.name}' is not approved for registration: "
            f"verdict={report.recommendation.value}, is_safe={report.is_safe}"
        )

    def get(self, name: str) -> SkillProtocol | None:
        """Retrieve an *active* skill by name. Quarantined packages are not returned."""
        return self._skills.get(name)

    def get_audit_report(self, name: str) -> SkillAuditReport | None:
        """Retrieve the security audit report for a registered skill."""
        return self._audit_reports.get(name)

    def list_skills(self) -> list[SkillProtocol]:
        """List all active skills."""
        return list(self._skills.values())

    def get_summary(self) -> dict[str, Any]:
        """Return JSON-serializable list of registered skills and security audit summary."""
        skills_list: list[dict[str, Any]] = []
        for skill in self._skills.values():
            manifest = skill.manifest
            report = self._audit_reports.get(manifest.name)
            if report is not None:
                audit_report_dict: dict[str, Any] = {
                    "skill_name": report.skill_name,
                    "is_safe": report.is_safe,
                    "recommendation": (
                        report.recommendation.value
                        if hasattr(report.recommendation, "value")
                        else str(report.recommendation)
                    ),
                    "risk_score": report.risk_score if report.risk_score is not None else 0.0,
                    "detected_risks": list(report.detected_risks),
                    "auditor_version": report.auditor_version or "0.1.0",
                    "content_sha256": report.content_sha256 or "",
                }
            else:
                audit_report_dict = {
                    "skill_name": manifest.name,
                    "is_safe": manifest.status == SkillStatus.ACTIVE,
                    "recommendation": (
                        AuditVerdict.APPROVE.value
                        if manifest.status == SkillStatus.ACTIVE
                        else AuditVerdict.REQUIRE_HUMAN_REVIEW.value
                    ),
                    "risk_score": 0.0 if manifest.status == SkillStatus.ACTIVE else 0.5,
                    "detected_risks": [],
                    "auditor_version": "0.1.0",
                    "content_sha256": manifest.content_sha256 or "",
                }

            skill_dict: dict[str, Any] = {
                "name": manifest.name,
                "description": manifest.description,
                "version": manifest.version,
                "author": manifest.author or "unknown",
                "origin": (
                    manifest.origin.value
                    if hasattr(manifest.origin, "value")
                    else str(manifest.origin)
                ),
                "status": (
                    manifest.status.value
                    if hasattr(manifest.status, "value")
                    else str(manifest.status)
                ),
                "isolation_level": (
                    manifest.requested_isolation.value
                    if manifest.requested_isolation is not None
                    else "workspace"
                ),
                "content_sha256": manifest.content_sha256 or "",
                "scripts": list(manifest.scripts),
                "tags": list(manifest.tags),
                "approved_by": manifest.approved_by,
                "approved_at": manifest.approved_at,
                "rejected_by": manifest.rejected_by,
                "rejected_at": manifest.rejected_at,
                "rejection_reason": manifest.rejection_reason,
                "audit_report": audit_report_dict,
            }
            skills_list.append(skill_dict)

        active_count = sum(1 for s in skills_list if s["status"] == SkillStatus.ACTIVE.value)
        pending_count = sum(1 for s in skills_list if s["status"] == SkillStatus.PENDING.value)
        quarantined_count = sum(
            1 for s in skills_list if s["status"] == SkillStatus.QUARANTINED.value
        )

        return {
            "skills": skills_list,
            "total": len(skills_list),
            "summary": {
                "total_skills": len(skills_list),
                "active_count": active_count,
                "pending_count": pending_count,
                "quarantined_count": quarantined_count,
            },
        }

    async def scan(self, skills_dir: Path) -> tuple[SkillManifest, ...]:
        """Discover packages on disk and return their manifests without activating any."""
        return await asyncio.to_thread(self._scan_sync, skills_dir)

    def _scan_sync(self, skills_dir: Path) -> tuple[SkillManifest, ...]:
        if not skills_dir.exists() or not skills_dir.is_dir():
            return ()

        manifests: list[SkillManifest] = []
        for child in sorted(skills_dir.iterdir()):
            if child.is_dir():
                skill_file = child / "SKILL.md"
                if skill_file.is_file():
                    try:
                        skill = load_skill_from_dir(child)
                        manifests.append(skill.manifest)
                    except Exception as exc:
                        manifests.append(
                            SkillManifest(
                                name=child.name,
                                description=f"Unparseable skill package: {exc}",
                                origin=SkillOrigin.SYNTHESIZED,
                                status=SkillStatus.REJECTED,
                                rejection_reason=f"Package failed to parse: {exc}",
                            )
                        )
        return tuple(manifests)

    async def reload_approved(
        self,
        skills_dir: Path | None = None,
        auditor: SkillAuditorProtocol | None = None,
    ) -> tuple[SkillProtocol, ...]:
        """Scan skills_dir on disk, audit all packages, and hot-reload approved ones into the registry."""
        target_dir = skills_dir or self._skills_dir
        if target_dir is None or not target_dir.exists() or not target_dir.is_dir():
            return ()

        active_auditor = auditor or SkillAuditor(policy=AutoApprovalPolicy.SAFE_ONLY)
        reloaded: list[SkillProtocol] = []
        for child in sorted(target_dir.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file():
                try:
                    skill = load_skill_from_dir(child)
                    if skill.manifest.status == SkillStatus.ACTIVE:
                        report = await active_auditor.audit_skill(child)
                        if report.is_safe and report.recommendation is AuditVerdict.APPROVE:
                            bound_manifest = skill.manifest.model_copy(
                                update={"content_sha256": report.content_sha256}
                            )
                            bound_skill = Skill(
                                manifest=bound_manifest,
                                instructions_markdown=skill.instructions_markdown,
                                directory=child,
                            )
                            self.register(bound_skill, report)
                            reloaded.append(bound_skill)
                except Exception as exc:
                    logger.debug("Failed to reload skill package %s: %s", child.name, exc)
        return tuple(reloaded)
