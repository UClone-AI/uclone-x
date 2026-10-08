"""Skills a clone proposes, and the person's approve, reject and revoke of them (#1827).

A clone never makes a skill active. `propose_skill` writes a prompt-only `SKILL.md` into
the store's `.pending/<name>/<version>/` folder, which nothing loads: the store and every
scan skip dot-named folders, so a proposal is not in the catalog and `load_skill` cannot
reach it. A person approves it in Settings (`POST /api/skills/{name}/approve`), which runs
the same flow as `ucx skill approve` on a copy: audit, write it active, take the digest,
pin it in the approvals ledger. There is no path, flag or setting that approves a proposal
without that request (owner ruling 2026-09-27).

Layout (design: the skill system architecture document)::

    <root>/<name>/                 the active version (+ .proposal.json when proposed)
    <root>/.pending/<name>/<v>/    a proposal
    <root>/.versions/<name>/<v>/   an approved version a newer one replaced
    <root>/.rejected/<name>/<v>/   a proposal the person turned down

Every refusal is a `SkillProposalError` whose text is one plain sentence for the person:
no path, exception name or refusal code.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast, get_args

from uclone_x.errors import PlainRefusalError, SkillAuditError
from uclone_x.skills.approvals import SkillApprovalLedger, SkillPin
from uclone_x.skills.auditor import (
    Skill,
    SkillAuditor,
    compute_skill_sha256,
    copy_skill_package,
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
from uclone_x.skills.shipped_pins import SHIPPED_SKILL_PINS
from uclone_x.skills.synthesizer import SkillSynthesizer, one_line

__all__ = [
    "PENDING_DIRNAME",
    "PROPOSAL_FILENAME",
    "REJECTED_DIRNAME",
    "VERSIONS_DIRNAME",
    "SkillDecisionCode",
    "SkillProposal",
    "SkillProposalChangedError",
    "SkillProposalError",
    "SkillProposalStore",
]

logger = logging.getLogger(__name__)

PENDING_DIRNAME: Final[str] = ".pending"
VERSIONS_DIRNAME: Final[str] = ".versions"
REJECTED_DIRNAME: Final[str] = ".rejected"
#: The proposal's provenance. Dot-named, so it is outside the digest and never copied by
#: `copy_skill_package`.
PROPOSAL_FILENAME: Final[str] = ".proposal.json"

_NAME: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_VERSION: Final[re.Pattern[str]] = re.compile(r"(\d+)\.(\d+)\.(\d+)")

#: The line under a proposal's title, in place of the synthesizer's "recorded" note.
_PROPOSED_NOTE: Final[str] = "Proposed by a clone. It is instructions only and carries no script."
_PROPOSED_STEPS_LEAD: Final[str] = (
    "Follow these steps in order, adapting names, paths and values to the task at hand."
)

#: What the ledger records as the approver of a Settings decision.
SETTINGS_PERSON: Final[str] = "human:settings"

REVOKED_REASON: Final[str] = "Revoked in Settings."
REJECTED_REASON: Final[str] = "Turned down in Settings."

# Plain sentences for the person and the model.
BAD_NAME = (
    "A skill name must be lowercase letters, digits, '-' or '_', at most 64 characters, "
    "and start with a letter or digit."
)
NO_STEPS = "A skill needs at least one step with some text in it."
NO_DESCRIPTION = "A skill needs a description that says when to use it."
SHIPPED = "This skill comes with UClone-X, so it cannot be changed or revoked here."
HAS_FILES = (
    "This skill has files besides its instructions, so a clone cannot propose a new "
    "version of it. A person can change it by hand."
)
BAD_TOOLS = "The list of tools a skill needs must name each tool once, by its exact name."
NOT_FOUND = "That proposal was not found. It may already have been approved or turned down."
NOT_ACTIVE = "That skill is not in use, so there is nothing to revoke."
FAILED_CHECK = (
    "The safety check did not pass this proposal, so it cannot be approved. "
    "You can turn it down instead."
)
CHECK_NOT_FINISHED = "The safety check could not read this proposal, so it cannot be approved."
CHANGED_MEANWHILE = (
    "This proposal changed while it was being checked, so it was not approved. "
    "Look at it again, then approve it."
)
SEEN_CHANGED = (
    "This proposal changed after it was shown to you, so it was not approved. "
    "Look at it again, then approve it."
)
SEEN_CHANGED_TURN_DOWN = (
    "This proposal changed after it was shown to you, so it was not turned down. "
    "Look at it again, then decide."
)
EXTRA_FILES = (
    "This proposal holds files besides its instructions, so it cannot be approved. "
    "You can turn it down instead."
)
NOT_SAVED = "The skill folder could not be changed, so nothing was saved."

#: Why Settings' approve, turn down or revoke was refused: a stable name the head renders in
#: the reader's language (`skills.refusals.<code>` in each catalog, #1865). The sentence the
#: route sends as `detail` is the English one; the code travels beside it, never inside it.
SkillDecisionCode = Literal[
    "shipped",
    "has_files",
    "extra_files",
    "not_found",
    "not_active",
    "failed_check",
    "check_not_finished",
    "changed_meanwhile",
    "seen_changed",
    "not_saved",
    # Set by the Settings routes, not the store: no skill folder, an unexpected failure, or
    # a request that did not say which version, or which text the person saw.
    "no_store",
    "not_changed",
    "no_version",
    "not_seen",
]
SKILL_DECISION_CODES: Final[tuple[SkillDecisionCode, ...]] = get_args(SkillDecisionCode)


class SkillProposalError(PlainRefusalError):
    """A proposal, approve, reject or revoke refused; the message is for a person."""

    def __init__(self, message: str, *, reason_code: SkillDecisionCode | None = None) -> None:
        super().__init__(message, reason_code=reason_code)
        #: `reason_code`, typed as the closed set the head's catalogs cover.
        self.decision_code: SkillDecisionCode | None = reason_code


class SkillProposalChangedError(SkillProposalError):
    """The proposal is not the one the person was shown; they should look at it again."""


@dataclass(frozen=True)
class SkillProposal:
    """One pending proposal, as the Settings panel shows it."""

    name: str
    version: str
    description: str
    requires_tools: tuple[str, ...]
    agent_id: str
    session_id: str
    proposed_at: str
    instructions: str
    current_version: str | None
    diff: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        """The JSON `/api/skills` sends under `proposals`."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "requires_tools": list(self.requires_tools),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "proposed_at": self.proposed_at,
            "instructions": self.instructions,
            "current_version": self.current_version,
            "diff": self.diff,
            "digest": self.digest,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _version_key(version: str) -> tuple[int, int, int] | None:
    match = _VERSION.fullmatch(version.strip())
    if match is None:
        return None
    return (int(match[1]), int(match[2]), int(match[3]))


def _review_text(manifest: SkillManifest, instructions: str) -> list[str]:
    """What a person compares between two versions: the parts a proposal can change."""
    tools = ", ".join(manifest.requires_tools) or "(none)"
    return [
        f"description: {manifest.description}",
        f"requires_tools: {tools}",
        "",
        *instructions.strip().splitlines(),
    ]


def _holds_more_than_instructions(package: Path) -> bool:
    """Whether a copied package holds any file but its top-level `SKILL.md`."""
    return any(
        path.relative_to(package).as_posix() != "SKILL.md"
        for path in package.rglob("*")
        if not path.is_dir()
    )


class SkillProposalStore:
    """Proposals under a skill store's root, and the person's decisions on them."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        """The skill store folder (`ucx-agent-skills/`)."""
        return self._root

    def _pending(self, name: str) -> Path:
        return self._root / PENDING_DIRNAME / name

    # --- proposing ------------------------------------------------------------------

    def _versions_of(self, name: str) -> list[tuple[int, int, int]]:
        found: list[tuple[int, int, int]] = []
        for area in (PENDING_DIRNAME, VERSIONS_DIRNAME, REJECTED_DIRNAME):
            folder = self._root / area / name
            if folder.is_dir():
                for child in folder.iterdir():
                    key = _version_key(child.name)
                    if key is not None:
                        found.append(key)
        current = self._current(name)
        if current is not None:
            key = _version_key(current.manifest.version)
            if key is not None:
                found.append(key)
        return found

    def _next_version(self, name: str) -> str:
        versions = self._versions_of(name)
        if not versions:
            return "0.1.0"
        major, minor, patch = max(versions)
        return f"{major}.{minor}.{patch + 1}"

    def _current(self, name: str) -> Skill | None:
        folder = self._root / name
        if not (folder / "SKILL.md").is_file():
            return None
        try:
            return load_skill_from_dir(folder)
        except SkillAuditError:
            return None

    def _refuse_replacing(self, name: str) -> None:
        """Refuse to replace a shipped skill, or one that holds more than its `SKILL.md`."""
        if name in SHIPPED_SKILL_PINS:
            raise SkillProposalError(SHIPPED, reason_code="shipped")
        folder = self._root / name
        if folder.is_dir():
            extra = [
                child
                for child in folder.iterdir()
                if not child.name.startswith(".") and child.name != "SKILL.md"
            ]
            if extra:
                raise SkillProposalError(HAS_FILES, reason_code="has_files")

    def propose(
        self,
        *,
        name: str,
        description: str,
        steps: Sequence[str],
        requires_tools: Sequence[str] = (),
        agent_id: str,
        session_id: str,
    ) -> SkillProposal:
        """Write a pending, prompt-only proposal and return it. Nothing is made active.

        The status is always `pending` and the author is the proposing clone: the caller
        cannot choose either (the tool has no such argument).
        """
        clean = name.strip().lower()
        if not _NAME.fullmatch(clean):
            raise SkillProposalError(BAD_NAME)
        flat_description = one_line(description)
        if not flat_description:
            raise SkillProposalError(NO_DESCRIPTION)
        self._refuse_replacing(clean)
        try:
            instructions = SkillSynthesizer().format_instructions(
                clean,
                steps,
                flat_description,
                note=_PROPOSED_NOTE,
                steps_lead=_PROPOSED_STEPS_LEAD,
            )
        except ValueError as exc:
            raise SkillProposalError(NO_STEPS) from exc
        try:
            manifest = SkillManifest(
                name=clean,
                description=flat_description,
                version=self._next_version(clean),
                author=f"agent:{one_line(agent_id) or 'unknown'}",
                origin=SkillOrigin.SYNTHESIZED,
                status=SkillStatus.PENDING,
                tags=("proposed",),
                requires_tools=tuple(requires_tools),
            )
        except ValueError as exc:
            raise SkillProposalError(BAD_TOOLS) from exc
        current = self._current(clean)
        target = self._pending(clean) / manifest.version
        try:
            target.mkdir(parents=True, exist_ok=False)
            save_skill(target, manifest, instructions)
            (target / PROPOSAL_FILENAME).write_text(
                json.dumps(
                    {
                        "agent_id": agent_id,
                        "session_id": session_id,
                        "proposed_at": _now(),
                        "base_version": current.manifest.version if current else None,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("A skill proposal for '%s' could not be written: %s", clean, exc)
            raise SkillProposalError(NOT_SAVED, reason_code="not_saved") from exc
        return self._read_proposal(clean, target)

    # --- listing --------------------------------------------------------------------

    def _read_proposal(self, name: str, folder: Path) -> SkillProposal:
        # Read from one copy, so the text shown and the digest Approve sends back are of the
        # same bytes even while a clone's file tools write into the store (#1827).
        with tempfile.TemporaryDirectory(prefix="ucx-skill-proposal-") as scratch:
            snapshot = Path(scratch) / name
            copy_skill_package(folder, snapshot)
            digest = compute_skill_sha256(snapshot)
            skill = load_skill_from_dir(snapshot)
        provenance: dict[str, Any] = {}
        try:
            raw: object = json.loads((folder / PROPOSAL_FILENAME).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                provenance = {str(k): v for k, v in cast("dict[object, Any]", raw).items()}
        except (OSError, ValueError):
            provenance = {}
        current = self._current(name)
        diff = ""
        if current is not None:
            diff = "\n".join(
                difflib.unified_diff(
                    _review_text(current.manifest, current.instructions_markdown),
                    _review_text(skill.manifest, skill.instructions_markdown),
                    fromfile=f"{name} {current.manifest.version}",
                    tofile=f"{name} {skill.manifest.version}",
                    lineterm="",
                )
            )
        return SkillProposal(
            name=skill.manifest.name,
            version=folder.name,
            description=skill.manifest.description,
            requires_tools=skill.manifest.requires_tools,
            agent_id=str(provenance.get("agent_id") or ""),
            session_id=str(provenance.get("session_id") or ""),
            proposed_at=str(provenance.get("proposed_at") or ""),
            instructions=skill.instructions_markdown,
            current_version=current.manifest.version if current else None,
            diff=diff,
            digest=digest,
        )

    def list_proposals(self) -> list[SkillProposal]:
        """Every readable pending proposal, by name then version. An unreadable one is logged."""
        area = self._root / PENDING_DIRNAME
        if not area.is_dir():
            return []
        proposals: list[SkillProposal] = []
        for name_dir in sorted(area.iterdir()):
            if not name_dir.is_dir() or name_dir.name.startswith("."):
                continue
            for folder in sorted(name_dir.iterdir()):
                if not folder.is_dir() or _version_key(folder.name) is None:
                    continue
                try:
                    proposals.append(self._read_proposal(name_dir.name, folder))
                except SkillAuditError as exc:
                    logger.warning("A skill proposal could not be read: %s", exc)
        return proposals

    def _proposal_dir(self, name: str, version: str) -> Path:
        if not _NAME.fullmatch(name) or _version_key(version) is None:
            raise SkillProposalError(NOT_FOUND, reason_code="not_found")
        folder = self._pending(name) / version
        if not (folder / "SKILL.md").is_file():
            raise SkillProposalError(NOT_FOUND, reason_code="not_found")
        return folder

    # --- deciding -------------------------------------------------------------------

    async def approve(
        self,
        name: str,
        version: str,
        *,
        seen_digest: str,
        approver: str,
        ledger: SkillApprovalLedger,
    ) -> str:
        """Make the proposal the active version of `name` and pin its digest; return it.

        The flow of `ucx skill approve`, on a copy of the proposal read once: audit it,
        refuse it unless the copy's digest is `seen_digest` (the one the person was shown),
        refuse any verdict but a safe `APPROVE` (the only one `load_approved` loads), write
        it active, take the digest of the file as that
        write left it, check the proposal did not change meanwhile, keep the current
        version under `.versions/`, install the copy, and pin the digest.
        """
        pending = self._proposal_dir(name, version)
        self._refuse_replacing(name)
        with tempfile.TemporaryDirectory(prefix="ucx-skill-proposal-") as scratch:
            checked_dir = Path(scratch) / name
            try:
                copy_skill_package(pending, checked_dir)
                checked = load_skill_from_dir(checked_dir)
                report = await SkillAuditor(policy=AutoApprovalPolicy.SAFE_ONLY).audit_skill(
                    checked_dir
                )
            except SkillAuditError as exc:
                logger.warning("The skill proposal '%s' could not be checked: %s", name, exc)
                raise SkillProposalError(
                    CHECK_NOT_FINISHED, reason_code="check_not_finished"
                ) from exc
            if report.content_sha256 != seen_digest:
                raise SkillProposalChangedError(SEEN_CHANGED, reason_code="seen_changed")
            if _holds_more_than_instructions(checked_dir):
                # Only `SKILL.md` is installed, so a digest over more than that would pin a
                # package that is never on disk, and the person is shown only the
                # instructions: approving the rest unseen is what this refuses (#1858).
                raise SkillProposalError(EXTRA_FILES, reason_code="extra_files")
            manifest = checked.manifest
            if (
                manifest.name != name
                or manifest.scripts
                or manifest.entrypoint
                or not (report.is_safe and report.recommendation is AuditVerdict.APPROVE)
            ):
                raise SkillProposalError(FAILED_CHECK, reason_code="failed_check")
            approved_at = _now()
            active = manifest.model_copy(
                update={
                    "status": SkillStatus.ACTIVE,
                    "approved_by": approver,
                    "approved_at": approved_at,
                    "content_sha256": None,
                    "rejected_by": None,
                    "rejected_at": None,
                    "rejection_reason": None,
                }
            )
            try:
                save_skill(checked_dir, active, checked.instructions_markdown)
                digest = compute_skill_sha256(checked_dir)
                save_skill(
                    checked_dir,
                    active.model_copy(update={"content_sha256": digest}),
                    checked.instructions_markdown,
                )
                if compute_skill_sha256(pending) != report.content_sha256:
                    raise SkillProposalChangedError(
                        CHANGED_MEANWHILE, reason_code="changed_meanwhile"
                    )
                data = (checked_dir / "SKILL.md").read_bytes()
                await asyncio.to_thread(
                    self._install, name, pending, data, digest, approver, approved_at, ledger
                )
            except (SkillAuditError, OSError) as exc:
                logger.warning("The skill proposal '%s' could not be approved: %s", name, exc)
                raise SkillProposalError(NOT_SAVED, reason_code="not_saved") from exc
        return digest

    def _install(
        self,
        name: str,
        pending: Path,
        data: bytes,
        digest: str,
        approver: str,
        approved_at: str,
        ledger: SkillApprovalLedger,
    ) -> None:
        live = self._root / name
        archived: Path | None = None
        current = self._current(name)
        if live.exists():
            old_version = current.manifest.version if current else "unknown"
            archived = self._free(self._root / VERSIONS_DIRNAME / name / old_version)
            archived.parent.mkdir(parents=True, exist_ok=True)
            live.rename(archived)
        try:
            live.mkdir(parents=True)
            (live / "SKILL.md").write_bytes(data)
            provenance = pending / PROPOSAL_FILENAME
            if provenance.is_file():
                shutil.copyfile(provenance, live / PROPOSAL_FILENAME)
            ledger.pin(
                name,
                SkillPin(content_sha256=digest, approved_by=approver, approved_at=approved_at),
            )
        except (OSError, SkillAuditError):
            # Put the previous version back: a half-installed approval must not replace it.
            shutil.rmtree(live, ignore_errors=True)
            if archived is not None:
                archived.rename(live)
            raise
        shutil.rmtree(pending, ignore_errors=True)
        self._drop_if_empty(pending.parent)

    @staticmethod
    def _free(path: Path) -> Path:
        """`path`, or `path-2`, `path-3`, ... when it is taken: nothing kept is overwritten."""
        candidate, n = path, 1
        while candidate.exists():
            n += 1
            candidate = path.with_name(f"{path.name}-{n}")
        return candidate

    @staticmethod
    def _drop_if_empty(folder: Path) -> None:
        try:
            folder.rmdir()
        except OSError:
            pass

    def reject(
        self,
        name: str,
        version: str,
        *,
        seen_digest: str,
        rejecter: str,
        reason: str | None,
    ) -> None:
        """Move the proposal to `.rejected/`, marked rejected with who, when and why.

        Bound to what the person saw, as `approve` is (#1865): unless the proposal's digest
        is still `seen_digest`, nothing moves and `SkillProposalChangedError` says so.
        """
        pending = self._proposal_dir(name, version)
        try:
            current_digest = compute_skill_sha256(pending)
        except SkillAuditError as exc:
            logger.warning("The skill proposal '%s' could not be checked: %s", name, exc)
            raise SkillProposalError(CHECK_NOT_FINISHED, reason_code="check_not_finished") from exc
        if current_digest != seen_digest:
            raise SkillProposalChangedError(SEEN_CHANGED_TURN_DOWN, reason_code="seen_changed")
        try:
            skill = load_skill_from_dir(pending)
            save_skill(
                pending,
                skill.manifest.model_copy(
                    update={
                        "status": SkillStatus.REJECTED,
                        "rejected_by": rejecter,
                        "rejected_at": _now(),
                        "rejection_reason": one_line(reason or "") or REJECTED_REASON,
                    }
                ),
                skill.instructions_markdown,
            )
            target = self._free(self._root / REJECTED_DIRNAME / name / version)
            target.parent.mkdir(parents=True, exist_ok=True)
            pending.rename(target)
        except (SkillAuditError, OSError) as exc:
            logger.warning("The skill proposal '%s' could not be turned down: %s", name, exc)
            raise SkillProposalError(NOT_SAVED, reason_code="not_saved") from exc
        self._drop_if_empty(pending.parent)

    def revoke(self, name: str, *, revoker: str, ledger: SkillApprovalLedger) -> None:
        """Stop an active skill: remove its pin, then mark it rejected. Its files stay."""
        if not _NAME.fullmatch(name):
            raise SkillProposalError(NOT_ACTIVE, reason_code="not_active")
        if name in SHIPPED_SKILL_PINS:
            raise SkillProposalError(SHIPPED, reason_code="shipped")
        current = self._current(name)
        if current is None or current.manifest.status is not SkillStatus.ACTIVE:
            raise SkillProposalError(NOT_ACTIVE, reason_code="not_active")
        try:
            ledger.revoke(name)
            save_skill(
                self._root / name,
                current.manifest.model_copy(
                    update={
                        "status": SkillStatus.REJECTED,
                        "rejected_by": revoker,
                        "rejected_at": _now(),
                        "rejection_reason": REVOKED_REASON,
                    }
                ),
                current.instructions_markdown,
            )
        except (SkillAuditError, OSError) as exc:
            logger.warning("The skill '%s' could not be revoked: %s", name, exc)
            raise SkillProposalError(NOT_SAVED, reason_code="not_saved") from exc
