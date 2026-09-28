"""Where a skill's approval is pinned: a ledger outside the skill packages (#1720, #1751).

An approval is only worth something if it names the bytes that were approved and a later
edit cannot carry it along. A digest written into the skill's own `SKILL.md` cannot do that:
whoever edits the skill can rewrite the digest in the same edit. So `ucx skill approve`
records the approved digest here, in the person's data folder (`~/.uclone/skills/
approvals.json`, or `$UCLONE_SKILL_APPROVALS_DIR/approvals.json`), and the store loads an
active skill only when its current digest is the one pinned for its name -- here, or, for a
skill that ships with the code, in `SHIPPED_SKILL_PINS`.

The ledger holds one pin per skill name: the version the person last approved. Approving
again replaces it; rejecting removes it.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from uclone_x.errors import SkillAuditError
from uclone_x.skills.shipped_pins import SHIPPED_SKILL_PINS

__all__ = [
    "APPROVALS_DIR_ENV_VAR",
    "APPROVALS_FILENAME",
    "DEFAULT_APPROVALS_DIR",
    "LEDGER_UNUSABLE",
    "SkillApprovalLedger",
    "SkillPin",
    "skill_approvals_path",
]

#: Overrides the folder the ledger lives in; tests point it at a temporary folder.
APPROVALS_DIR_ENV_VAR: Final[str] = "UCLONE_SKILL_APPROVALS_DIR"
DEFAULT_APPROVALS_DIR: Final[Path] = Path.home() / ".uclone" / "skills"
APPROVALS_FILENAME: Final[str] = "approvals.json"

#: What the person reads when the ledger cannot be used; no path, no parser message.
LEDGER_UNUSABLE = (
    "The list of skills you approved could not be read or saved, so the approval could not "
    "be recorded."
)


def skill_approvals_path() -> Path:
    """The ledger file, honouring `UCLONE_SKILL_APPROVALS_DIR`."""
    override = os.environ.get(APPROVALS_DIR_ENV_VAR)
    folder = Path(override).expanduser() if override else DEFAULT_APPROVALS_DIR
    return folder / APPROVALS_FILENAME


@dataclass(frozen=True)
class SkillPin:
    """One approval: the digest approved for a skill name, and who approved it when."""

    content_sha256: str
    approved_by: str
    approved_at: str


class SkillApprovalLedger:
    """The person's approvals, one pin per skill name, in a JSON file outside any package.

    The file is resolved when it is read, not when the ledger is built, so a ledger made at
    startup follows `UCLONE_SKILL_APPROVALS_DIR` as it is at load time.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """The ledger file."""
        return self._path if self._path is not None else skill_approvals_path()

    def read(self) -> dict[str, SkillPin]:
        """Every pin in the ledger; an absent ledger holds none.

        Raises:
            SkillAuditError: The ledger exists but cannot be read or is not a ledger. A
                caller that loads skills treats that as "nothing approved here"; one that
                writes must not overwrite it.
        """
        path = self.path
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise SkillAuditError(LEDGER_UNUSABLE) from exc
        try:
            raw: object = json.loads(text)
        except ValueError as exc:
            raise SkillAuditError(LEDGER_UNUSABLE) from exc
        skills: object = (
            cast(dict[str, object], raw).get("skills") if isinstance(raw, dict) else None
        )
        if not isinstance(skills, dict):
            raise SkillAuditError(LEDGER_UNUSABLE)
        pins: dict[str, SkillPin] = {}
        for name, entry in cast(dict[object, object], skills).items():
            if not isinstance(entry, dict):
                continue
            fields = cast(dict[object, object], entry)
            digest = fields.get("content_sha256")
            if isinstance(name, str) and isinstance(digest, str):
                pins[name] = SkillPin(
                    content_sha256=digest,
                    approved_by=str(fields.get("approved_by", "")),
                    approved_at=str(fields.get("approved_at", "")),
                )
        return pins

    def approved_digests(self, name: str, pins: dict[str, SkillPin]) -> frozenset[str]:
        """The digests `name` may load under: its pin in `pins`, and its shipped pin."""
        digests: set[str] = set()
        if name in pins:
            digests.add(pins[name].content_sha256)
        if name in SHIPPED_SKILL_PINS:
            digests.add(SHIPPED_SKILL_PINS[name])
        return frozenset(digests)

    def pin(self, name: str, pin: SkillPin) -> None:
        """Record `pin` as the approved version of `name`, replacing any earlier one."""
        pins = self.read()
        pins[name] = pin
        self._write(pins)

    def revoke(self, name: str) -> None:
        """Remove the pin for `name`, if there is one."""
        pins = self.read()
        if pins.pop(name, None) is not None:
            self._write(pins)

    def _write(self, pins: dict[str, SkillPin]) -> None:
        """Replace the ledger in one step, so a reader never sees half of it."""
        path = self.path
        payload = {
            "version": 1,
            "skills": {
                name: {
                    "content_sha256": pin.content_sha256,
                    "approved_by": pin.approved_by,
                    "approved_at": pin.approved_at,
                }
                for name, pin in sorted(pins.items())
            },
        }
        temp_name: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".approvals-", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temp_name, path)
        except OSError as exc:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
            raise SkillAuditError(LEDGER_UNUSABLE) from exc
