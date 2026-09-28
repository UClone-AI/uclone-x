"""A skill registry loads from a `SkillStoreProtocol` store, not only from a folder (#1733).

`SkillRegistry(skills_dir=...)` still reads `<name>/SKILL.md` packages through
`FileSystemSkillStore`, which is what every head uses. These tests run the registry and
`load_skill` over `InMemorySkillStore`, with no directory anywhere, and pin that the
registry's own admission rule still holds for what such a store hands it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from uclone_x.skills import (
    AuditVerdict,
    InMemorySkillStore,
    Skill,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillRegistry,
    SkillStatus,
)
from uclone_x.tools.builtin.skill_loader import LoadSkillParams, LoadSkillTool
from uclone_x.tools.models import ToolContext

DIGEST = "d" * 64


def _entry(
    name: str,
    *,
    status: SkillStatus = SkillStatus.ACTIVE,
    verdict: AuditVerdict = AuditVerdict.APPROVE,
) -> tuple[Skill, SkillAuditReport]:
    skill = Skill(
        manifest=SkillManifest(
            name=name,
            description=f"The {name} procedure.",
            origin=SkillOrigin.HUMAN,
            status=status,
            content_sha256=DIGEST,
        ),
        instructions_markdown=f"# {name}\nDo the {name} steps.",
    )
    report = SkillAuditReport(
        skill_name=name,
        is_safe=verdict is AuditVerdict.APPROVE,
        recommendation=verdict,
        content_sha256=DIGEST,
    )
    return skill, report


@pytest.mark.asyncio
async def test_load_skill_serves_a_skill_from_an_in_memory_store(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/skills/auditor.py :: store = FileSystemSkillStore(skills_dir) if skills_dir is not None else self._store
    Becomes: store = FileSystemSkillStore(skills_dir) if skills_dir is not None else None
    """
    registry = SkillRegistry(store=InMemorySkillStore([_entry("pour_over")]))

    loaded = await registry.reload_approved()

    assert [s.manifest.name for s in loaded] == ["pour_over"]
    tool = LoadSkillTool(registry)
    context = ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path)
    text = await tool.run(LoadSkillParams(skill_name="pour_over"), context)
    assert text == "# pour_over\nDo the pour_over steps."


@pytest.mark.asyncio
async def test_an_in_memory_store_offers_only_its_active_skills() -> None:
    """Killed by: src/uclone_x/skills/auditor.py :: if skill.manifest.status is SkillStatus.ACTIVE
    Becomes: if True
    """
    registry = SkillRegistry(
        store=InMemorySkillStore([_entry("draft", status=SkillStatus.PENDING), _entry("live")])
    )

    loaded = await registry.reload_approved()

    assert [s.manifest.name for s in loaded] == ["live"]
    assert registry.get("draft") is None


@pytest.mark.asyncio
async def test_the_registry_refuses_and_logs_a_stored_skill_its_report_does_not_approve(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A store cannot activate a skill: `register` still judges the report it hands over.

    Killed by: src/uclone_x/skills/auditor.py :: logger.warning("The skill '%s' was not loaded: %s", skill.manifest.name, exc)
    Becomes: logger.debug("The skill '%s' was not loaded: %s", skill.manifest.name, exc)
    """
    registry = SkillRegistry(
        store=InMemorySkillStore([_entry("risky", verdict=AuditVerdict.REJECT), _entry("ok")])
    )

    with caplog.at_level(logging.WARNING, logger="uclone_x.skills.auditor"):
        loaded = await registry.reload_approved()

    assert [s.manifest.name for s in loaded] == ["ok"]
    assert registry.get("risky") is None
    assert [r.getMessage() for r in caplog.records] == [
        "The skill 'risky' was not loaded: Skill 'risky' is not approved for registration: "
        "verdict=reject, is_safe=False"
    ]


def test_a_registry_takes_a_folder_or_a_store_not_both(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/skills/auditor.py :: raise ValueError("Give a SkillRegistry a skills_dir or a store, not both")
    Becomes: pass
    """
    with pytest.raises(ValueError, match="not both"):
        SkillRegistry(tmp_path, store=InMemorySkillStore())
