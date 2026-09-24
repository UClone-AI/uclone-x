"""Built-in tool for on-demand progressive loading of approved skills (P9)."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.skills.models import SkillStatus
from uclone_x.skills.protocols import SkillRegistryProtocol
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext


class LoadSkillParams(BaseModel):
    """Parameters for loading an approved skill on demand."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    skill_name: str = Field(
        description="The exact name of the approved skill to load into context.",
    )


class LoadSkillTool(BaseTool[LoadSkillParams]):
    """Tool for loading approved modular skill procedural instructions on demand."""

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
    ) -> None:
        super().__init__()
        self._registry = registry
        self._on_load = on_load

    async def run(self, params: LoadSkillParams, context: ToolContext) -> str:
        skill = self._registry.get(params.skill_name)
        if skill is None or skill.manifest.status != SkillStatus.ACTIVE:
            raise ValueError(f"Skill '{params.skill_name}' is not found or has not been approved.")
        if self._on_load is not None:
            self._on_load(params.skill_name)
        return skill.instructions_markdown
