"""Protocols for skills, dynamic synthesizer, registry, and Skill Auditor.

`@runtime_checkable` is applied only where a runtime `isinstance` check is actually
performed. On a protocol with a `@property`, `issubclass()` raises `TypeError` and
`isinstance()` calls the object's getters as a side effect of the type test, and neither
form checks a signature — which is what actually drifted in issue 2026-09-02-035.
Conformance is enforced statically instead, by the bindings in
`tests/unit/test_protocol_conformance.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from uclone_x.sandbox.models import IsolationLevel
from uclone_x.skills.models import (
    AutoApprovalPolicy,
    SkillAuditReport,
    SkillManifest,
)


class SkillProtocol(Protocol):
    """Protocol representing a loaded modular skill package."""

    @property
    def manifest(self) -> SkillManifest:
        """Skill metadata and frontmatter."""
        ...

    @property
    def instructions_markdown(self) -> str:
        """Markdown procedural knowledge."""
        ...


@runtime_checkable
class SkillRegistryProtocol(Protocol):
    """Protocol for hot-reloading skill discovery and runtime management."""

    def register(self, skill: SkillProtocol, report: SkillAuditReport) -> None:
        """Register a skill, admitting it only on a passing audit of *this* code.

        The report is a required argument rather than a separate step a caller may
        forget: with `register(skill)` alone, "synthesize, register, hot-reload into
        every agent" was a legal call sequence with the auditor skipped entirely
        (issue 2026-09-02-034).

        Raises:
            SkillNotApprovedError: If the report does not approve the skill, or covers
                different content than the package being registered.
        """
        ...

    def get(self, name: str) -> SkillProtocol | None:
        """Retrieve an *active* skill by name. Quarantined packages are not returned."""
        ...

    def list_skills(self) -> list[SkillProtocol]:
        """List all active skills."""
        ...

    def get_summary(self) -> dict[str, Any]:
        """Return JSON-serializable list of registered skills and security summary."""
        ...

    async def scan(self, skills_dir: Path) -> tuple[SkillManifest, ...]:
        """Discover packages on disk and return their manifests without activating any.

        Asynchronous because it touches the filesystem, which under P1 must not block
        the event loop; and it returns what it found rather than a count, because a
        caller needs to know *which* packages appeared before deciding anything.
        """
        ...

    async def reload_approved(
        self,
        skills_dir: Path | None = None,
        auditor: SkillAuditorProtocol | None = None,
    ) -> tuple[SkillProtocol, ...]:
        """Scan skills_dir on disk, audit all packages, and hot-reload approved ones into the registry."""
        ...


class SkillAuditorProtocol(Protocol):
    """Protocol for automated skill security audit and policy enforcement."""

    @property
    def policy(self) -> AutoApprovalPolicy:
        """Current auto-approval mode."""
        ...

    @property
    def isolation_floor(self) -> IsolationLevel:
        """The weakest isolation a skill may run under, decided by the runtime.

        A synthesized `SKILL.md` states what it wants; the runtime resolves that
        against this floor with `effective_isolation_level`, so a request can only
        strengthen isolation. P3, as amended for issue 2026-09-02-001: a requesting
        artifact never raises its own ceiling.
        """
        ...

    async def audit_skill(self, skill_dir: Path) -> SkillAuditReport:
        """Audit a package before activation.

        Raises:
            SkillAuditError: If the audit cannot be completed. It must not return a
                report claiming safety it did not establish.
        """
        ...


@runtime_checkable
class SkillSynthesizerProtocol(Protocol):
    """Protocol for autonomous skill distillation from successful turns."""

    async def synthesize_skill(
        self,
        task_name: str,
        workflow_steps: list[str],
        quarantine_dir: Path,
    ) -> SkillManifest:
        """Write a SKILL.md package into quarantine and return its manifest.

        The destination is named `quarantine_dir`, not `output_dir`: the returned
        manifest is `SkillStatus.PENDING` and synthesis has no path that produces an
        active skill. Promotion is a separate, audited act.
        """
        ...
