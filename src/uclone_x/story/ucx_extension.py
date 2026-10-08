"""The story extension: what stories add to the core, declared in one place (#2205).

The core never imports `uclone_x.story`. It finds this module by its name
(`extensions.registry`, in-tree discovery) and reads `EXTENSION`:

* **Tools** -- the story tools, and `character_sheet` in the core's place with the story's
  behaviour while one is open (`story/character.py`).
* **Turn lifecycle hooks** -- `StoryLifecycleHook`, which moves the story a turn has open
  between its steps (#1732), and `SceneTurnHook`, a Writer turn's next scene (#1808).
* **A protected folder** -- `stories/`, which general tools may read and never write
  (#1583, #1589).
* **A leased folder kind** -- the story library, which the Files screen lists one story
  per folder and opens into a conversation (#1554).
* **The `a2a_call` character lookup** -- the open story's characters a task names (#1808).
* **Routes** -- the story view (#1560).

Everything that loads more than this module and the story library is imported when the
core asks for it, so finding the extension does not load the story tools.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from uclone_x.artifacts.kinds import (
    FolderItem,
    FolderItemUnreadableError,
    UnknownFolderItemError,
)
from uclone_x.extensions import Extension, ProtectedRoot
from uclone_x.story.library import (
    STORIES_DIRNAME,
    STORY_FILE,
    StoryError,
    StoryLibrary,
    StoryRecord,
    UnknownStoryError,
)

if TYPE_CHECKING:
    from uclone_x.agent.protocols import TurnLifecycleHookProtocol
    from uclone_x.tools.models import ToolContext
    from uclone_x.tools.protocols import ToolProtocol

__all__ = ["EXTENSION", "StoryFolderKind"]

#: A story folder's name, as the story library makes one.
_STORY_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

#: What a general tool is told when it tries to write in the story library (#1583).
_WRITE_REFUSAL = (
    "{path} is in the story library, so it was not written. Stories are changed with the "
    "story tools (story_manuscript, story_outline, story_codex), which check which "
    "conversation is writing the story and ask a person before a codex change. Reading "
    "the file is still allowed."
)


def _item(record: StoryRecord) -> FolderItem:
    return FolderItem(
        title=record.title,
        holder=record.lease.holder if record.lease is not None else None,
    )


class StoryFolderKind:
    """The story library as the Files screen's leased folder kind (`LeasedFolderKind`)."""

    @property
    def root(self) -> str:
        """`stories`, under the workspace and under `.archive/`."""
        return STORIES_DIRNAME

    def is_item(self, folder: Path) -> bool:
        """A story folder: a story id for a name, not a link, with its `story.yaml`."""
        return (
            _STORY_ID.fullmatch(folder.name) is not None
            and not folder.is_symlink()
            and (folder / STORY_FILE).is_file()
        )

    def load(self, base: Path, item_id: str) -> FolderItem:
        """The story `item_id` under `<base>/stories`."""
        try:
            return _item(StoryLibrary(base).load(item_id))
        except StoryError as exc:
            raise FolderItemUnreadableError(str(exc)) from exc

    def open(self, workspace: Path, item_id: str, room_id: str) -> tuple[FolderItem, bool]:
        """Open the story in `room_id`: writable when its lease is free or already the room's."""
        try:
            opening = StoryLibrary(workspace).open(item_id, room_id)
        except UnknownStoryError as exc:
            raise UnknownFolderItemError(str(exc)) from exc
        return _item(opening.story), opening.writable

    def take_over(self, workspace: Path, item_id: str, room_id: str) -> FolderItem:
        """Give `room_id` the lease of a story whose writer no longer exists."""
        record, _ = StoryLibrary(workspace).take_over(item_id, room_id)
        return _item(record)

    def release(self, workspace: Path, item_id: str, holder: str) -> bool:
        """Give up the story's lease if `holder` holds it."""
        return StoryLibrary(workspace).release(item_id, holder)


def _tools() -> Sequence[ToolProtocol]:
    from uclone_x.story.character import StoryCharacterSheetTool
    from uclone_x.story.muse import MuseSparkTool
    from uclone_x.story.start import StoryStartTool
    from uclone_x.story.tool import StoryLibraryTool
    from uclone_x.story.tools import (
        StoryAuditTool,
        StoryCodexTool,
        StoryContextTool,
        StoryManuscriptTool,
        StoryOutlineTool,
    )

    return (
        StoryCharacterSheetTool(),
        StoryLibraryTool(),
        MuseSparkTool(),
        StoryOutlineTool(),
        StoryCodexTool(),
        StoryManuscriptTool(),
        StoryContextTool(),
        StoryAuditTool(),
        StoryStartTool(),
    )


def _lifecycle_hooks() -> Sequence[TurnLifecycleHookProtocol]:
    from uclone_x.story import StoryLifecycleHook
    from uclone_x.story.scene_turn import SceneTurnHook

    return (StoryLifecycleHook(), SceneTurnHook())


def _a2a_characters() -> Callable[[ToolContext, Sequence[str]], list[dict[str, Any]]]:
    from uclone_x.story.tools import story_characters

    return story_characters


def _routes(context: Any) -> None:
    from uclone_x.story.routes import register_story_routes

    register_story_routes(context)


EXTENSION = Extension(
    name="story",
    tools=_tools,
    replaces_tools=("character_sheet",),
    lifecycle_hooks=_lifecycle_hooks,
    protected_roots=(ProtectedRoot(dirname=STORIES_DIRNAME, refusal=_WRITE_REFUSAL),),
    leased_folders=lambda: (StoryFolderKind(),),
    a2a_characters=_a2a_characters,
    routes=_routes,
)
