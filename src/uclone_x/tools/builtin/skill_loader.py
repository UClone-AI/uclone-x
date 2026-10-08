"""Built-in tool for on-demand progressive loading of approved skills (P9)."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import PlainRefusalError
from uclone_x.skills.models import ROUTED_SKILL_TAG, SkillStatus, missing_required_tools
from uclone_x.skills.protocols import SkillProtocol, SkillRegistryProtocol
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.media_registry import ModelProfile, PromptFamily
from uclone_x.tools.models import ToolContext

#: Supplies the profile of the image model that is active *now*; called on every load.
ProfileProvider = Callable[[], ModelProfile]

#: A family-sections heading: `## family: <family>` or `## model: <model_id>`, alone on
#: its line. Any other `##`/`###` heading belongs to the section it sits in.
_SECTION_HEADING = re.compile(r"^## (family|model): *(\S.*?)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")

#: Per-family image skills replaced by the `media-<domain>` skills. A user `models.yaml`
#: or sidecar written before then may still name one as `skill_name`.
RETIRED_MEDIA_SKILLS = frozenset(
    {"media-prompt-danbooru", "media-prompt-flux", "media-prompt-generic"}
)


class LoadSkillParams(BaseModel):
    """Parameters for loading an approved skill on demand."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    skill_name: str = Field(
        description="The exact name of the approved skill to load into context.",
    )


@dataclass(frozen=True)
class FamilySections:
    """A family-sections SKILL.md body, split at its section headings.

    `sections` is keyed by the heading's own text, `"family: danbooru"` or
    `"model: anillustrious_v4"`, and each value is the section including its heading line.
    """

    preamble: str
    sections: dict[str, str]


def split_family_sections(skill_name: str, body: str) -> FamilySections:
    """Split `body` into its preamble and its `## family:` / `## model:` sections.

    A heading inside a fenced code block is text, not a heading. A `## family:` naming no
    `PromptFamily`, or a heading that appears twice, raises: either would otherwise make a
    section silently unreachable (P6).
    """
    valid_families = {family.value for family in PromptFamily}
    preamble_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current: list[str] = preamble_lines
    in_fence = False
    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _SECTION_HEADING.match(line)
        if match is None:
            current.append(line)
            continue
        kind, value = match.group(1), match.group(2)
        if kind == "family" and value not in valid_families:
            raise ValueError(
                f"Skill '{skill_name}' has a section heading '## family: {value}', but "
                f"'{value}' is not a prompt family (one of {sorted(valid_families)})."
            )
        key = f"{kind}: {value}"
        if key in sections:
            raise ValueError(f"Skill '{skill_name}' has the section '## {key}' twice.")
        current = [line]
        sections[key] = current
    return FamilySections(
        preamble="\n".join(preamble_lines).strip(),
        sections={key: "\n".join(lines).strip() for key, lines in sections.items()},
    )


class LoadSkillTool(BaseTool[LoadSkillParams]):
    """Tool for loading approved modular skill procedural instructions on demand.

    A skill whose manifest says `family_sections: true` (the `media-<domain>` image
    skills) is not returned whole: it is resolved against the active image model's
    profile, so a model never reads another prompt family's rules.

    1. **Override.** When `profile.skill_name` is set, that skill is returned whole in
       place of the requested one, headed by a line saying so. The override replaces
       family x domain routing for *every* family-sections skill, whichever domain was
       asked for: a model whose grammar differs throughout has no use for any domain's
       sections (§3.4). An override naming a skill that is not active, a retired
       `media-prompt-*` skill, or a skill that itself has family sections (returned
       whole it would mix families) raises. Loading
       the override skill by its own name, or any skill without family sections, is an
       ordinary load and returns it unchanged.
    2. **Sections.** Otherwise the result is the preamble (everything before the first
       section heading) plus the first section found of `model: <model_id>`,
       `family: <family>`, `family: generic`. The generic section, used for a model of
       another family, is headed by a plain line saying no rules for that family exist.
       A skill with none of the three raises, naming the skill and the sections looked
       for.
    3. **Notes.** `profile.prompt_notes`, when not empty, is appended to either result.

    A family-sections skill loaded by a tool built without a profile provider raises
    rather than returning the whole file: every family's rules in one context is the
    contamination the design exists to prevent (P6).
    """

    name: str = "load_skill"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    description: str = (
        "Load the full procedural instructions and guidance for an approved skill into context. "
        "Only approved skills listed in [Available Approved Skills] can be loaded."
    )

    def __init__(
        self,
        registry: SkillRegistryProtocol,
        on_load: Callable[[str], None] | None = None,
        profile_provider: ProfileProvider | None = None,
        tool_scope: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._on_load = on_load
        self._profile_provider = profile_provider
        self._tool_scope = tool_scope

    def _active_skill(self, name: str) -> SkillProtocol:
        skill = self._registry.get(name)
        if skill is None or skill.manifest.status != SkillStatus.ACTIVE:
            raise ValueError(f"Skill '{name}' is not found or has not been approved.")
        # A case-routed skill is handed over by code for the request it fits (#1865); the
        # catalog leaves it out, and this closes the path of a name the model recalls.
        if ROUTED_SKILL_TAG in skill.manifest.tags:
            raise ValueError(
                f"Skill '{name}' is added automatically when a request needs it, "
                "so it cannot be loaded by name."
            )
        return skill

    def _require_scope(self, skill: SkillProtocol) -> None:
        """Refuse a skill whose `requires_tools` this agent's tool scope does not grant (#1826).

        The catalog already leaves such a skill out; this closes the path of a name the
        model recalls or guesses.
        """
        if self._tool_scope is None:
            return
        missing = missing_required_tools(skill.manifest, self._tool_scope())
        if missing:
            raise ValueError(
                f"Skill '{skill.manifest.name}' needs the tool(s) {', '.join(missing)}, "
                "which this agent cannot use, so it is not available here."
            )

    async def run(self, params: LoadSkillParams, context: ToolContext) -> str:
        skill = self._active_skill(params.skill_name)
        self._require_scope(skill)
        if not skill.manifest.family_sections:
            self._record(params.skill_name)
            return skill.instructions_markdown
        if self._profile_provider is None:
            raise ValueError(
                f"Skill '{params.skill_name}' is split by image model family, and this agent "
                "has no active image model to choose a section for."
            )
        profile = self._profile_provider()
        if profile.skill_name is not None:
            override = self._active_skill_for_override(profile, params.skill_name)
            text = (
                f"The active image model '{profile.model_id}' replaces the image domain "
                f"skills with skill '{profile.skill_name}'; its rules follow in full.\n\n"
                f"{override.instructions_markdown.strip()}"
            )
            self._record(profile.skill_name)
        else:
            text = self._resolve_sections(params.skill_name, skill, profile)
            self._record(params.skill_name)
        notes = profile.prompt_notes.strip()
        if notes:
            text = f"{text}\n\n## Notes for {profile.model_id}\n{notes}"
        return text

    def _active_skill_for_override(self, profile: ModelProfile, requested: str) -> SkillProtocol:
        override_name = profile.skill_name or ""
        try:
            override = self._active_skill(override_name)
        except ValueError:
            if override_name in RETIRED_MEDIA_SKILLS:
                raise PlainRefusalError(
                    f"The image model '{profile.model_id}' is set up to use the prompt guide "
                    f"'{override_name}', which has been retired. Delete the line "
                    f"'skill_name: {override_name}' from that model's entry in models.yaml "
                    "(or from the settings file next to the model), then try again."
                ) from None
            raise ValueError(
                f"The active image model '{profile.model_id}' names skill '{override_name}' "
                f"in place of '{requested}', and '{override_name}' is not found or has not "
                "been approved."
            ) from None
        if override.manifest.family_sections:
            raise ValueError(
                f"The active image model '{profile.model_id}' names skill '{override_name}' "
                f"in place of '{requested}', but '{override_name}' is split by prompt family "
                "and cannot replace the image domain skills whole. Name a skill without "
                "family sections, or remove `skill_name` from that model's entry."
            )
        self._require_scope(override)
        return override

    @staticmethod
    def _resolve_sections(name: str, skill: SkillProtocol, profile: ModelProfile) -> str:
        return resolve_family_sections(name, skill.instructions_markdown, profile)

    def _record(self, name: str) -> None:
        if self._on_load is not None:
            self._on_load(name)


def resolve_family_sections(skill_name: str, body: str, profile: ModelProfile) -> str:
    """The preamble plus the one section of `body` that `profile` selects.

    The first section found of `model: <model_id>`, `family: <family>`, `family: generic`;
    the generic section, used for a model of another family, is headed by a plain line
    saying no rules for that family exist. None of the three raises, naming the skill and
    the sections looked for (P6).
    """
    split = split_family_sections(skill_name, body)
    family = profile.family.value
    wanted = [f"model: {profile.model_id}", f"family: {family}"]
    if profile.family is not PromptFamily.GENERIC:
        wanted.append(f"family: {PromptFamily.GENERIC.value}")
    for key in wanted:
        section = split.sections.get(key)
        if section is None:
            continue
        if key == "family: generic" and profile.family is not PromptFamily.GENERIC:
            section = f"No {family} rules for this domain; generic rules follow.\n\n{section}"
        return f"{split.preamble}\n\n{section}" if split.preamble else section
    looked_for = ", ".join(f"'## {key}'" for key in wanted)
    raise ValueError(
        f"Skill '{skill_name}' has no section for the active image model "
        f"'{profile.model_id}': looked for {looked_for}."
    )
