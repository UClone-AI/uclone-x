"""An active skill loads only while it is the version that was approved (#1720, #1751).

The approved digest is pinned outside the package, in the person's approvals ledger, so an
edit to the package cannot carry its approval along -- not even by rewriting the digest the
package itself records. These tests run the file-system store the way every head does and
check what the Settings Skills panel is given (`SkillRegistry.get_summary`).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import get_args

import pytest

from uclone_x.skills.approvals import SkillApprovalLedger, SkillPin
from uclone_x.skills.auditor import (
    SkillRegistry,
    compute_skill_sha256,
    save_skill,
)
from uclone_x.skills.models import SkillManifest, SkillOrigin, SkillStatus
from uclone_x.skills.refusals import SKILL_REFUSAL_FALLBACK, SkillRefusalCode, refusal_reason

NAME = "pour_over"
INSTRUCTIONS = "# Pour over\nBloom the grounds for thirty seconds."

#: What the panel must not show a person: a digest, a path, a field name or error text.
_INTERNALS = re.compile(
    r"[0-9a-f]{16}|/|\\|sha|digest|hash|ledger|json|error|exception|none\b", re.IGNORECASE
)


def _active_skill(
    root: Path, instructions: str = INSTRUCTIONS, approved_by: str | None = "human:alice"
) -> Path:
    skill_dir = root / NAME
    save_skill(
        skill_dir,
        SkillManifest(
            name=NAME,
            description="Brew one cup by hand.",
            origin=SkillOrigin.HUMAN,
            status=SkillStatus.ACTIVE,
            approved_by=approved_by,
        ),
        instructions,
    )
    return skill_dir


def _approve(skill_dir: Path) -> str:
    digest = compute_skill_sha256(skill_dir)
    SkillApprovalLedger().pin(NAME, SkillPin(digest, "human:alice", "2026-09-27T00:00:00Z"))
    return digest


def _summary_entry(registry: SkillRegistry) -> dict[str, object]:
    [entry] = [s for s in registry.get_summary()["skills"] if s["name"] == NAME]
    return entry


@pytest.mark.asyncio
async def test_a_skill_edited_after_approval_is_refused_and_says_so_in_plain_words(
    tmp_path: Path,
) -> None:
    """The edit also rewrites the digest the file records; the pin is what decides.

    Killed by: src/uclone_x/skills/auditor.py :: elif report.content_sha256 not in digests:
    Becomes: elif False:

    Killed by: src/uclone_x/skills/auditor.py :: self._skills.pop(name, None)
    Becomes: pass

    Killed by: src/uclone_x/skills/auditor.py :: "not_loaded_reason": refusal.reason if refusal is not None else None,
    Becomes: "not_loaded_reason": None,

    Killed by: src/uclone_x/skills/auditor.py :: if digests:
    Becomes: if False:

    Killed by: src/uclone_x/skills/auditor.py :: "not_loaded_code": refusal.code if refusal is not None else None,
    Becomes: "not_loaded_code": None,
    """
    skill_dir = _active_skill(tmp_path)
    _approve(skill_dir)
    registry = SkillRegistry(skills_dir=tmp_path)
    assert [s.manifest.name for s in await registry.reload_approved()] == [NAME]

    edited = INSTRUCTIONS + "\nSkip the bloom; pour it all at once."
    manifest = SkillManifest(
        name=NAME,
        description="Brew one cup by hand.",
        origin=SkillOrigin.HUMAN,
        status=SkillStatus.ACTIVE,
        approved_by="human:alice",
    )
    save_skill(skill_dir, manifest, edited)
    forged = compute_skill_sha256(skill_dir)
    save_skill(skill_dir, manifest.model_copy(update={"content_sha256": forged}), edited)
    assert compute_skill_sha256(skill_dir) == forged  # the file vouches for itself...

    assert await registry.reload_approved() == ()  # ...and is refused anyway
    assert registry.get(NAME) is None
    entry = _summary_entry(registry)
    assert entry["status"] == SkillStatus.QUARANTINED.value
    assert entry["not_loaded_code"] == "changed_after_approval"
    assert entry["not_loaded_params"] == {"name": NAME}
    reason = entry["not_loaded_reason"]
    assert reason == refusal_reason("changed_after_approval", NAME)
    assert isinstance(reason, str)
    without_command = reason.replace(f"ucx skill approve {NAME}", "")
    assert _INTERNALS.search(without_command) is None, reason


@pytest.mark.asyncio
async def test_an_active_skill_never_approved_here_is_refused_with_its_own_words(
    tmp_path: Path,
) -> None:
    """`status: active` in the file is not an approval; only a pin is."""
    _active_skill(tmp_path, approved_by=None)
    registry = SkillRegistry(skills_dir=tmp_path)

    assert await registry.reload_approved() == ()
    entry = _summary_entry(registry)
    assert entry["not_loaded_code"] == "never_approved"
    assert entry["not_loaded_reason"] == refusal_reason("never_approved", NAME)


@pytest.mark.asyncio
async def test_a_skill_approved_before_approvals_were_pinned_is_asked_to_be_approved_again(
    tmp_path: Path,
) -> None:
    """The file names who approved it, but the ledger has no pin for it (#1776, #1777).

    Killed by: src/uclone_x/skills/auditor.py :: elif skill.manifest.approved_by:
    Becomes: elif False:
    """
    _active_skill(tmp_path, approved_by="human:alice")
    registry = SkillRegistry(skills_dir=tmp_path)

    assert await registry.reload_approved() == ()
    entry = _summary_entry(registry)
    assert entry["status"] == SkillStatus.QUARANTINED.value
    assert entry["not_loaded_code"] == "approved_before_pins"
    assert entry["not_loaded_reason"] == refusal_reason("approved_before_pins", NAME)
    assert f"ucx skill approve {NAME}" in str(entry["not_loaded_reason"])


@pytest.mark.asyncio
async def test_a_skill_whose_instructions_cannot_be_read_is_listed_with_a_plain_reason(
    tmp_path: Path,
) -> None:
    """A package that does not parse is listed under its folder, not dropped silently (#1777).

    Killed by: src/uclone_x/skills/auditor.py :: refused.append(SkillRefusal(unreadable, None, "unreadable"))
    Becomes: pass
    """
    broken = tmp_path / NAME
    broken.mkdir()
    (broken / "SKILL.md").write_text("---\nname: [unclosed\n---\n# Pour over\n", encoding="utf-8")
    registry = SkillRegistry(skills_dir=tmp_path)

    assert await registry.reload_approved() == ()
    entry = _summary_entry(registry)
    assert entry["status"] == SkillStatus.QUARANTINED.value
    assert entry["not_loaded_code"] == "unreadable"
    reason = entry["not_loaded_reason"]
    assert reason == refusal_reason("unreadable", NAME)
    assert _INTERNALS.search(str(reason)) is None, reason


@pytest.mark.asyncio
async def test_an_approved_skill_the_check_cannot_read_in_full_is_listed_with_a_plain_reason(
    tmp_path: Path,
) -> None:
    """The audit raising is a refusal the panel hears about, not only the log (#1777).

    Killed by: src/uclone_x/skills/auditor.py :: refused.append(SkillRefusal(skill.manifest, None, "check_not_finished"))
    Becomes: pass
    """
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("root lists any folder, and the test needs one it cannot")
    skill_dir = _active_skill(tmp_path)
    _approve(skill_dir)
    (skill_dir / "resources" / "beans").mkdir(parents=True)
    (skill_dir / "resources").chmod(0o311)
    try:
        registry = SkillRegistry(skills_dir=tmp_path)
        assert await registry.reload_approved() == ()
    finally:
        (skill_dir / "resources").chmod(0o755)

    entry = _summary_entry(registry)
    assert entry["status"] == SkillStatus.QUARANTINED.value
    assert entry["not_loaded_code"] == "check_not_finished"
    reason = entry["not_loaded_reason"]
    assert reason == refusal_reason("check_not_finished", NAME)
    assert _INTERNALS.search(str(reason)) is None, reason


def test_every_refusal_reason_is_plain_words() -> None:
    """The one technical thing a reason may carry is the command to run."""
    for code, template in SKILL_REFUSAL_FALLBACK.items():
        reason = template.format(name=NAME).replace(f"ucx skill approve {NAME}", "")
        assert "{" not in reason, code
        assert _INTERNALS.search(reason) is None, (code, reason)


def test_the_head_words_exactly_the_codes_the_core_sends() -> None:
    """Codes = the catalog's keys in every language; the fallback is the English catalog.

    Killed by: src/uclone_x/skills/refusals.py :: "unreadable",
    Becomes: "unreadable", "renamed",

    Killed by: src/uclone_x/skills/refusals.py :: "failed_safety_check": "The safety check did not pass this skill, so it is not used.",
    Becomes: "failed_safety_check": "The safety check did not pass this skill.",
    """
    locales = Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n" / "locales"
    codes: set[str] = set(get_args(SkillRefusalCode))
    catalogs = {
        lang: json.loads((locales / lang / "skills.json").read_text(encoding="utf-8"))["notLoaded"]
        for lang in ("en", "ko")
    }
    assert set(SKILL_REFUSAL_FALLBACK) == codes
    for lang, catalog in catalogs.items():
        assert set(catalog["codes"]) == codes, lang
        assert catalog["label"].strip(), lang
    assert catalogs["en"]["codes"] == SKILL_REFUSAL_FALLBACK


@pytest.mark.asyncio
async def test_an_unedited_approved_skill_loads(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/skills/approvals.py :: digests.add(pins[name].content_sha256)
    Becomes: digests.add(name)
    """
    _approve(_active_skill(tmp_path))
    registry = SkillRegistry(skills_dir=tmp_path)

    loaded = await registry.reload_approved()

    assert [s.manifest.name for s in loaded] == [NAME]
    entry = _summary_entry(registry)
    assert entry["status"] == SkillStatus.ACTIVE.value
    assert entry["not_loaded_reason"] is None


def test_the_digest_does_not_change_when_the_file_records_it(tmp_path: Path) -> None:
    r"""The rule leaves out only the frontmatter line holding 64 hex digits (#1751).

    Killed by: src/uclone_x/skills/auditor.py :: if _OWN_DIGEST_LINE.fullmatch(line) is None
    Becomes: if True

    Killed by: src/uclone_x/skills/auditor.py :: rb"content_sha256: (?:[0-9a-f]{64}|'[0-9a-f]{64}')\r?\n?"
    Becomes: rb"content_sha256: .*\r?\n?"
    """
    skill_dir = _active_skill(tmp_path)
    skill_md = skill_dir / "SKILL.md"
    without = compute_skill_sha256(skill_dir)
    text = skill_md.read_text(encoding="utf-8")
    head, sep, body = text.partition("\n---")
    assert sep, text

    for spelling in (without, f"'{without}'", "f" * 64):
        skill_md.write_text(f"{head}\ncontent_sha256: {spelling}{sep}{body}", encoding="utf-8")
        assert compute_skill_sha256(skill_dir) == without, spelling

    # Any other value is hashed: it could say something the rest of the frontmatter reads.
    skill_md.write_text(f"{head}\ncontent_sha256: &a {without}{sep}{body}", encoding="utf-8")
    assert compute_skill_sha256(skill_dir) != without
    # And the same line in the body is instructions, so it is hashed too.
    skill_md.write_text(f"{text}\ncontent_sha256: {without}\n", encoding="utf-8")
    skill_md_plain = compute_skill_sha256(skill_dir)
    skill_md.write_text(f"{text}\n", encoding="utf-8")
    assert skill_md_plain != compute_skill_sha256(skill_dir)
