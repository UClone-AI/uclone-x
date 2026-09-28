"""`propose_skill`: a clone writes a skill a person may approve in Settings (#1827, P9).

The proposal is pending and inert. It lands in the store's `.pending/` area, which nothing
loads, and becomes active only when a person approves it in Settings. The tool takes no
status, version, author or approval argument (`extra="forbid"` refuses them), so a clone
cannot ask for its own proposal to be approved (owner ruling 2026-09-27).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext

__all__ = ["ProposeSkillParams", "ProposeSkillTool"]


class ProposeSkillParams(BaseModel):
    """What a clone may say about a skill it proposes; nothing about its approval."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(
        max_length=64,
        description="Short identifier: lowercase letters, digits, '-' or '_'. Proposing an "
        "existing skill's name proposes a new version of it.",
    )
    description: str = Field(
        max_length=500,
        description="One sentence: when a clone should use this skill.",
    )
    steps: list[str] = Field(
        min_length=1,
        max_length=40,
        description="The instructions, in order, one plain step per entry.",
    )
    requires_tools: list[str] = Field(
        default_factory=list,
        max_length=32,
        description="Exact names of the tools the steps call. A clone lacking any of them is "
        "not offered the skill.",
    )


#: What the model is told once the proposal is written.
_PROPOSED = (
    "Proposed skill '{name}' version {version}. It is not in use: it waits for the person "
    "to approve it in Settings > Skills. Do not tell them it is available until they have."
)


class ProposeSkillTool(BaseTool[ProposeSkillParams]):
    """Write a pending, prompt-only skill proposal for a person to review."""

    name: str = "propose_skill"
    writes_files: ClassVar[bool] = True  # writes the proposal into the skill store
    description: str = (
        "Propose a reusable skill (instructions only, no scripts) for the person to review. "
        "It is not used until they approve it in Settings; you cannot approve it."
    )

    def __init__(self, store_root: Path) -> None:
        super().__init__()
        self._store_root = store_root

    async def run(self, params: ProposeSkillParams, context: ToolContext) -> str:
        # Deferred: the store writes files and runs the auditor, an adapter this kernel
        # tool reaches only when a clone actually proposes (docs/core-shell-architecture.md §4).
        from uclone_x.skills.proposals import SkillProposalStore

        proposal = await asyncio.to_thread(
            SkillProposalStore(self._store_root).propose,
            name=params.name,
            description=params.description,
            steps=params.steps,
            requires_tools=params.requires_tools,
            agent_id=context.agent_id,
            session_id=context.session_id,
        )
        return _PROPOSED.format(name=proposal.name, version=proposal.version)
