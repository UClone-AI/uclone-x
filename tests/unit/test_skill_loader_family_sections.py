"""`load_skill` resolves a family-sections skill against the active image model.

An image domain skill carries one section per prompt family, and the loader returns the
preamble plus the one section for the model `generate_image` would use now: never another
family's rules, and never a silent fallback (P6).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig
from uclone_x.errors import PlainRefusalError
from uclone_x.skills.auditor import (
    Skill,
    SkillRegistry,
    manifest_from_dict,
    parse_skill_markdown,
    serialize_skill_markdown,
)
from uclone_x.skills.models import (
    AuditVerdict,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.tools.builtin.image import GenerateImageTool, ImagePipelineDispatcher
from uclone_x.tools.builtin.media_registry import ModelProfile, PromptFamily
from uclone_x.tools.builtin.skill_loader import LoadSkillParams, LoadSkillTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

ARCHITECTURE_BODY = """# Architecture Prompt Rules
PREAMBLE_SCOPE: exteriors and interiors.

## family: danbooru
DANBOORU_RULES: modern architecture, glass facade
### Negatives
DANBOORU_NEGATIVES: people, tilted horizon

## family: prose
PROSE_RULES: a wide-angle photograph of a concrete house

## family: generic
GENERIC_RULES: short English phrases

## model: special_model
MODEL_RULES: special tags only
"""

FAMILY_MARKERS = {
    "danbooru": "DANBOORU_RULES",
    "prose": "PROSE_RULES",
    "generic": "GENERIC_RULES",
}


def _skill(name: str, body: str, *, family_sections: bool) -> tuple[Skill, SkillAuditReport]:
    manifest = SkillManifest(
        name=name,
        description=f"{name} rules",
        origin=SkillOrigin.HUMAN,
        status=SkillStatus.ACTIVE,
        content_sha256=f"sha_{name}",
        family_sections=family_sections,
    )
    report = SkillAuditReport(
        skill_name=name,
        is_safe=True,
        recommendation=AuditVerdict.APPROVE,
        content_sha256=f"sha_{name}",
        auditor_version="0.1.0",
    )
    return Skill(manifest=manifest, instructions_markdown=body), report


def _registry(*skills: tuple[Skill, SkillAuditReport]) -> SkillRegistry:
    registry = SkillRegistry()
    for skill, report in skills:
        registry.register(skill, report)
    return registry


def _profile(
    family: PromptFamily,
    model_id: str = "some_model",
    *,
    skill_name: str | None = None,
    prompt_notes: str = "",
) -> ModelProfile:
    return ModelProfile(
        model_id=model_id,
        display_name=model_id,
        family=family,
        skill_name=skill_name,
        prompt_notes=prompt_notes,
    )


def _load(tool: LoadSkillTool, name: str) -> str:
    context = ToolContext(agent_id="a", session_id="s", workspace_root=Path("/tmp"))
    return asyncio.run(tool.run(LoadSkillParams(skill_name=name), context))


def _architecture_tool(
    profile: ModelProfile, *extra: tuple[Skill, SkillAuditReport], loaded: list[str] | None = None
) -> LoadSkillTool:
    registry = _registry(
        _skill("media-architecture", ARCHITECTURE_BODY, family_sections=True), *extra
    )
    return LoadSkillTool(
        registry,
        on_load=loaded.append if loaded is not None else None,
        profile_provider=lambda: profile,
    )


@pytest.mark.parametrize("family", ["danbooru", "prose", "generic"])
def test_each_family_gets_its_own_section_and_no_other_familys(family: str) -> None:
    """Objective 4: the active family's rules only, with the preamble and nested headings."""
    text = _load(_architecture_tool(_profile(PromptFamily(family))), "media-architecture")

    assert "PREAMBLE_SCOPE" in text
    assert FAMILY_MARKERS[family] in text
    for other, marker in FAMILY_MARKERS.items():
        if other != family:
            assert marker not in text
    assert "MODEL_RULES" not in text
    assert ("DANBOORU_NEGATIVES" in text) is (family == "danbooru")
    assert "generic rules follow" not in text


def test_a_model_section_wins_over_the_family_section() -> None:
    """Killed by: src/uclone_x/tools/builtin/skill_loader.py :: wanted = [f"model: {profile.model_id}", f"family: {family}"]
    Becomes: wanted = [f"family: {family}", f"model: {profile.model_id}"]
    """
    text = _load(
        _architecture_tool(_profile(PromptFamily.DANBOORU, "special_model")), "media-architecture"
    )

    assert "MODEL_RULES" in text
    assert "DANBOORU_RULES" not in text


def test_the_generic_fallback_is_announced() -> None:
    """P6: a family with no section of its own is told it is reading generic rules.

    Killed by: src/uclone_x/tools/builtin/skill_loader.py :: if key == "family: generic" and profile.family is not PromptFamily.GENERIC:
    Becomes: if False:
    """
    text = _load(_architecture_tool(_profile(PromptFamily.HYBRID)), "media-architecture")

    assert "No hybrid rules for this domain; generic rules follow." in text
    assert "GENERIC_RULES" in text
    assert "DANBOORU_RULES" not in text and "PROSE_RULES" not in text


def test_a_skill_with_neither_section_raises_naming_what_it_looked_for() -> None:
    body = "# Engineering\nPREAMBLE\n\n## family: prose\nPROSE_RULES\n"
    registry = _registry(_skill("media-engineering", body, family_sections=True))
    tool = LoadSkillTool(registry, profile_provider=lambda: _profile(PromptFamily.DANBOORU))

    with pytest.raises(ValueError, match="media-engineering") as caught:
        _load(tool, "media-engineering")

    message = str(caught.value)
    assert "'## model: some_model'" in message
    assert "'## family: danbooru'" in message
    assert "'## family: generic'" in message


def test_prompt_notes_are_appended() -> None:
    """Killed by: src/uclone_x/tools/builtin/skill_loader.py :: if notes:
    Becomes: if False:
    """
    profile = _profile(PromptFamily.DANBOORU, prompt_notes="Quality tags: score_9, score_8_up")

    text = _load(_architecture_tool(profile), "media-architecture")

    assert text.index("DANBOORU_RULES") < text.index("Quality tags: score_9, score_8_up")


def test_a_skill_without_family_sections_is_returned_unchanged() -> None:
    """Even one whose body happens to hold family headings, and with no provider wired."""
    body = "# Plain\n\n## family: danbooru\nKEEP\n\n## family: prose\nALSO_KEEP\n"
    tool = LoadSkillTool(_registry(_skill("plain", body, family_sections=False)))

    assert _load(tool, "plain") == body


def test_a_family_sections_skill_with_no_provider_raises() -> None:
    """P6: without a profile the loader cannot choose, and must not return every family.

    Killed by: src/uclone_x/tools/builtin/skill_loader.py :: if self._profile_provider is None:
    Becomes: if False:
    """
    registry = _registry(_skill("media-architecture", ARCHITECTURE_BODY, family_sections=True))
    tool = LoadSkillTool(registry)

    with pytest.raises(ValueError, match="no active image model"):
        _load(tool, "media-architecture")


class TestOverride:
    """`profile.skill_name` replaces family x domain routing (design §3.2 step 1, §3.4)."""

    OVERRIDE_BODY = "# Whole-model rules\nOVERRIDE_RULES\n\n## family: danbooru\nKEPT_WHOLE\n"

    def test_a_domain_load_returns_the_override_skill_whole(self) -> None:
        """Killed by: src/uclone_x/tools/builtin/skill_loader.py :: if profile.skill_name is not None:
        Becomes: if False:
        """
        loaded: list[str] = []
        profile = _profile(
            PromptFamily.DANBOORU, "odd_model", skill_name="odd-rules", prompt_notes="NOTE_LINE"
        )
        tool = _architecture_tool(
            profile, _skill("odd-rules", self.OVERRIDE_BODY, family_sections=False), loaded=loaded
        )

        text = _load(tool, "media-architecture")

        assert "OVERRIDE_RULES" in text and "KEPT_WHOLE" in text
        assert "DANBOORU_RULES" not in text and "PREAMBLE_SCOPE" not in text
        assert "odd-rules" in text.splitlines()[0]
        assert text.rstrip().endswith("NOTE_LINE")
        assert loaded == ["odd-rules"]

    def test_an_override_naming_no_active_skill_raises(self) -> None:
        profile = _profile(PromptFamily.DANBOORU, "odd_model", skill_name="missing-rules")

        with pytest.raises(ValueError, match="'missing-rules'"):
            _load(_architecture_tool(profile), "media-architecture")

    def test_the_override_skill_loaded_by_name_is_an_ordinary_load(self) -> None:
        profile = _profile(PromptFamily.DANBOORU, "odd_model", skill_name="odd-rules")
        tool = _architecture_tool(
            profile, _skill("odd-rules", self.OVERRIDE_BODY, family_sections=False)
        )

        assert _load(tool, "odd-rules") == self.OVERRIDE_BODY

    @pytest.mark.parametrize(
        "retired", ["media-prompt-danbooru", "media-prompt-flux", "media-prompt-generic"]
    )
    def test_a_stale_override_naming_a_retired_skill_says_to_remove_it(self, retired: str) -> None:
        """A user models.yaml or sidecar written before the domain skills still names one.

        Killed by: src/uclone_x/tools/builtin/skill_loader.py :: if override_name in RETIRED_MEDIA_SKILLS:
        Becomes: if False:
        """
        profile = _profile(PromptFamily.DANBOORU, "old_model", skill_name=retired)

        with pytest.raises(PlainRefusalError) as caught:
            _load(_architecture_tool(profile), "media-architecture")

        message = str(caught.value)
        assert retired in message and "retired" in message
        assert f"skill_name: {retired}" in message and "models.yaml" in message

    def test_an_override_with_family_sections_is_refused(self) -> None:
        """Returned whole, a family-sections override would hand the model every family.

        Killed by: src/uclone_x/tools/builtin/skill_loader.py :: if override.manifest.family_sections:
        Becomes: if False:
        """
        profile = _profile(PromptFamily.DANBOORU, "odd_model", skill_name="odd-rules")
        tool = _architecture_tool(
            profile, _skill("odd-rules", self.OVERRIDE_BODY, family_sections=True)
        )

        with pytest.raises(ValueError, match="split by prompt family"):
            _load(tool, "media-architecture")


def test_a_heading_inside_a_code_fence_is_not_a_section() -> None:
    body = (
        "# Rules\nPREAMBLE\n\n## family: danbooru\nDANBOORU_RULES\n"
        "```markdown\n## family: prose\nEXAMPLE_IN_FENCE\n```\n\n## family: prose\nPROSE_RULES\n"
    )
    registry = _registry(_skill("media-x", body, family_sections=True))
    tool = LoadSkillTool(registry, profile_provider=lambda: _profile(PromptFamily.DANBOORU))

    text = _load(tool, "media-x")

    assert "EXAMPLE_IN_FENCE" in text
    assert "PROSE_RULES" not in text


def test_a_heading_naming_no_family_raises() -> None:
    body = "# Rules\n\n## family: danbooro\nTYPO\n\n## family: generic\nGENERIC\n"
    registry = _registry(_skill("media-x", body, family_sections=True))
    tool = LoadSkillTool(registry, profile_provider=lambda: _profile(PromptFamily.NATURAL_PROSE))

    with pytest.raises(ValueError, match="danbooro"):
        _load(tool, "media-x")


def test_a_heading_that_appears_twice_raises() -> None:
    """The second section would otherwise silently replace the first.

    Killed by: src/uclone_x/tools/builtin/skill_loader.py :: if key in sections:
    Becomes: if False:
    """
    body = "# Rules\n\n## family: danbooru\nFIRST\n\n## family: danbooru\nSECOND\n"
    registry = _registry(_skill("media-x", body, family_sections=True))
    tool = LoadSkillTool(registry, profile_provider=lambda: _profile(PromptFamily.DANBOORU))

    with pytest.raises(ValueError, match="'## family: danbooru' twice"):
        _load(tool, "media-x")


def test_family_sections_round_trips_through_the_frontmatter() -> None:
    """Killed by: src/uclone_x/skills/auditor.py :: data["family_sections"] = True
    Becomes: data["family_sections"] = False
    """
    manifest = SkillManifest(
        name="media-architecture",
        description="Architecture",
        origin=SkillOrigin.HUMAN,
        family_sections=True,
    )

    data, body = parse_skill_markdown(serialize_skill_markdown(manifest, "# Body"))

    assert manifest_from_dict(data).family_sections is True
    assert body.strip() == "# Body"
    plain = manifest.model_copy(update={"family_sections": False})
    plain_data, _ = parse_skill_markdown(serialize_skill_markdown(plain, "# Body"))
    assert "family_sections" not in plain_data
    assert manifest_from_dict(plain_data).family_sections is False


def test_a_non_boolean_family_sections_is_refused() -> None:
    with pytest.raises(ValueError, match="family_sections"):
        manifest_from_dict({"name": "x", "family_sections": "yes"})


class _FixedDispatcher(ImagePipelineDispatcher):
    def __init__(self, profile: ModelProfile) -> None:
        super().__init__()
        self.profile = profile

    def get_active_profile(self, own: str | None = None) -> ModelProfile:
        return self.profile


def test_the_agent_wires_load_skill_to_the_image_tools_active_profile() -> None:
    """The agent's `load_skill` reads the profile the registered `generate_image` resolves.

    Killed by: src/uclone_x/agent/base.py :: profile_provider = clone_profile
    Becomes: profile_provider = None
    """
    dispatcher = _FixedDispatcher(_profile(PromptFamily.NATURAL_PROSE))
    tools = ToolRegistry(tools=[GenerateImageTool(dispatcher=dispatcher)])
    skills = _registry(_skill("media-architecture", ARCHITECTURE_BODY, family_sections=True))
    agent = BaseAgent(config=AgentConfig(agent_id="a", name="A"), tools=tools, skills=skills)

    record = asyncio.run(
        agent.execute_tool_call("load_skill", {"skill_name": "media-architecture"})
    )
    assert record.status == "success", record.error
    assert "PROSE_RULES" in str(record.output)
    assert "DANBOORU_RULES" not in str(record.output)

    dispatcher.profile = _profile(PromptFamily.DANBOORU)
    record = asyncio.run(
        agent.execute_tool_call("load_skill", {"skill_name": "media-architecture"})
    )
    assert "DANBOORU_RULES" in str(record.output)
    assert "PROSE_RULES" not in str(record.output)


def test_the_agent_gives_the_image_tool_its_skill_store() -> None:
    """Killed by: src/uclone_x/agent/base.py :: image_source.bind_skill_registry(self._skills)
    Becomes: pass
    """
    image_tool = GenerateImageTool(dispatcher=_FixedDispatcher(_profile(PromptFamily.DANBOORU)))
    skills = _registry(_skill("media-architecture", ARCHITECTURE_BODY, family_sections=True))

    BaseAgent(
        config=AgentConfig(agent_id="a", name="A"),
        tools=ToolRegistry(tools=[image_tool]),
        skills=skills,
    )

    assert "load_skill('media-architecture')" in image_tool.description
