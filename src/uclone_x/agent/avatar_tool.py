"""`set_avatar`: a clone changes its own picture.

The picture is the calling clone's, and only its own: the caller is the persona of the
agent running the tool (`agent_delegate.persona_name`), not `ToolContext.agent_id`, which in
a room is the seat and differs from the persona. There is no argument naming another
clone; changing another clone's face is a person's act, done from that clone's profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.agent.persona_avatar import AvatarRefused, PersonaAvatarStore, avatar_url
from uclone_x.agent.persona_registry import get_default_persona_registry
from uclone_x.errors import PathTraversalError, PlainRefusalError
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext

SET_AVATAR_TOOL = "set_avatar"


class SetAvatarParams(BaseModel):
    """Which picture to use, or whether to go back to the shipped one."""

    model_config = ConfigDict(extra="forbid", strict=True)

    image_path: str | None = Field(
        default=None,
        description=(
            "The picture to use, as a path in the workspace -- normally one generate_image "
            "just returned, or the previous_path an earlier set_avatar returned, to undo."
        ),
    )
    reset: bool = Field(
        default=False,
        description="True to stop using a chosen picture and show the shipped one again.",
    )
    undo_of: int | None = Field(
        default=None,
        description=(
            "Only when undoing: the change_id the change being undone returned. The undo "
            "is refused, and nothing changes, if the picture was changed again since."
        ),
    )


class SetAvatarTool(BaseTool[SetAvatarParams]):
    """Set or reset the calling clone's own picture.

    `writes_files` is False. By `tool_writes_files`' own definition, writing the agent's own
    state does not count, and this writes only the calling clone's picture, in the
    workspace personas folder, which no general writing tool can reach
    (`BaseTool.resolve_write_path` refuses it). So a clone without write tools can still
    change its own face.
    """

    name = SET_AVATAR_TOOL
    writes_files: ClassVar[bool] = False
    description = (
        "Set your own profile picture from an image in the workspace, usually one "
        "generate_image just made. Pass image_path to set it, or reset=true to go back to "
        "your shipped picture. Returns avatar_url, previous_path and change_id; to undo, "
        "call set_avatar again with image_path set to previous_path (or reset=true when "
        "previous_path is null) and undo_of set to that change_id. An undo is refused if "
        "the picture was changed again since. It changes only your own picture."
    )
    params_type = SetAvatarParams

    def run(self, params: SetAvatarParams, context: ToolContext) -> dict[str, str | int | None]:
        """Set or reset the picture and say where it is shown from."""
        agent = context.agent_delegate
        persona: object = getattr(agent, "persona_name", None) if agent is not None else None
        if not isinstance(persona, str) or not persona:
            raise PlainRefusalError(
                "Only a clone can change its own picture, and this conversation has none. "
                "Set the picture from the clone's profile instead."
            )
        if params.reset == (params.image_path is not None):
            raise PlainRefusalError(
                "Give either image_path, to set a picture, or reset=true, to go back to the "
                "shipped one -- one of the two."
            )
        workspace = context.require_workspace()
        registry = get_default_persona_registry(workspace)
        store = PersonaAvatarStore(registry)
        if params.image_path is None:
            change = store.reset(persona, undo_of=params.undo_of)
        else:
            source = store.named_previous(persona, params.image_path)
            if source is None:
                try:
                    source = self.resolve_safe_path(params.image_path, workspace)
                except PathTraversalError as exc:
                    raise AvatarRefused(
                        f"'{params.image_path}' is outside the workspace, so it was not used. "
                        "Use a picture in the workspace."
                    ) from exc
            change = store.set_from_path(persona, source, undo_of=params.undo_of)
        return {
            "avatar_url": avatar_url(registry.id_of(persona) or persona, store.find(persona)),
            "previous_path": _relative(change.previous, workspace),
            "change_id": change.change_id,
        }


def _relative(path: Path | None, workspace: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path)
