"""The video extension: what video projects add to the core, declared in one place.

The core never imports `uclone_x_video`. It finds `EXTENSION` through the entry point this
example's `pyproject.toml` declares in the group `uclone_x.extensions`:

* **Tools** -- `video_library`, `video_start`, `video_media_vault`, `video_search`,
  `video_manifest`, `video_timeline`, `video_fx_color`, `video_context`, `video_proposal`,
  and `video_audit`.
* **Turn lifecycle hooks** -- `VideoLifecycleHook`, which moves the video project a turn has
  open between its steps, and `VideoTurnHook`, which provides turn-level timeline context and
  selection parsing.
* **A protected folder** -- `videos/`, which general tools may read and never write.
* **A leased folder kind** -- the video project library, which the Files screen lists one
  project per folder and opens into a conversation when no other leased folder kind is active.
* **Routes** -- the interactive `VideoView` endpoints.
* **Personas** -- Director (`personas/director.yaml`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from uclone_x.artifacts.kinds import (
    FolderItem,
    FolderItemUnreadableError,
    UnknownFolderItemError,
)
from uclone_x.extensions import Extension, ProtectedRoot
from uclone_x_video.library import (
    PROJECT_FILE,
    VIDEOS_DIRNAME,
    UnknownVideoError,
    VideoError,
    VideoLibrary,
    VideoProjectRecord,
)

if TYPE_CHECKING:
    from uclone_x.agent.protocols import TurnLifecycleHookProtocol
    from uclone_x.tools.protocols import ToolProtocol

__all__ = ["EXTENSION", "PERSONAS_DIR", "VideoFolderKind"]

PERSONAS_DIR = Path(__file__).resolve().parent / "personas"

_PROJECT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

_WRITE_REFUSAL = (
    "{path} is in the video project library, so it was not written. Video projects are "
    "changed with the video tools (video_media_vault, video_manifest, video_timeline, "
    "video_fx_color, video_proposal), which preserve immutable raw media in vault/, check "
    "which conversation is directing the project, and ask a person before applying a proposal. "
    "Reading the file is still allowed."
)


def _item(record: VideoProjectRecord) -> FolderItem:
    return FolderItem(
        title=record.title,
        holder=record.lease.holder if record.lease is not None else None,
    )


class VideoFolderKind:
    """The video project library as the Files screen's leased folder kind (`LeasedFolderKind`)."""

    @property
    def root(self) -> str:
        """`videos`, under the workspace and under `.archive/`."""
        return VIDEOS_DIRNAME

    def is_item(self, folder: Path) -> bool:
        """A video project folder: a project id for a name, not a link, with its `project.yaml`."""
        return (
            _PROJECT_ID.fullmatch(folder.name) is not None
            and not folder.is_symlink()
            and (folder / PROJECT_FILE).is_file()
        )

    def load(self, base: Path, item_id: str) -> FolderItem:
        """The video project `item_id` under `<base>/videos`."""
        try:
            return _item(VideoLibrary(base).load(item_id))
        except VideoError as exc:
            raise FolderItemUnreadableError(str(exc)) from exc

    def open(self, workspace: Path, item_id: str, room_id: str) -> tuple[FolderItem, bool]:
        """Open the video project in `room_id`: writable when its lease is free or already the room's."""
        try:
            opening = VideoLibrary(workspace).open(item_id, room_id)
        except UnknownVideoError as exc:
            raise UnknownFolderItemError(str(exc)) from exc
        return _item(opening.project), opening.writable

    def take_over(self, workspace: Path, item_id: str, room_id: str) -> FolderItem:
        """Give `room_id` the lease of a video project whose writer no longer exists."""
        record, _ = VideoLibrary(workspace).take_over(item_id, room_id)
        return _item(record)

    def release(self, workspace: Path, item_id: str, holder: str) -> bool:
        """Give up the video project's lease if `holder` holds it."""
        return VideoLibrary(workspace).release(item_id, holder)


def _tools() -> Sequence[ToolProtocol]:
    from uclone_x_video.start import VideoStartTool
    from uclone_x_video.tool import VideoLibraryTool
    from uclone_x_video.tools import (
        VideoAuditTool,
        VideoContextTool,
        VideoFxColorTool,
        VideoManifestTool,
        VideoMediaVaultTool,
        VideoProposalTool,
        VideoSearchTool,
        VideoTimelineTool,
    )

    return (
        VideoLibraryTool(),
        VideoStartTool(),
        VideoMediaVaultTool(),
        VideoSearchTool(),
        VideoManifestTool(),
        VideoTimelineTool(),
        VideoFxColorTool(),
        VideoContextTool(),
        VideoProposalTool(),
        VideoAuditTool(),
    )


def _lifecycle_hooks() -> Sequence[TurnLifecycleHookProtocol]:
    from uclone_x_video import VideoLifecycleHook
    from uclone_x_video.section_turn import VideoTurnHook

    return (VideoLifecycleHook(), VideoTurnHook())


def _routes(context: Any) -> None:
    from uclone_x_video.routes import register_video_routes

    register_video_routes(context)


_in_leased_check = False


def _leased_folders() -> tuple[VideoFolderKind, ...]:
    global _in_leased_check
    if _in_leased_check:
        return ()
    from uclone_x.extensions import loaded_extensions

    _in_leased_check = True
    try:
        if any(ext.name != "video" and bool(ext.leased_folders()) for ext in loaded_extensions()):
            return ()
    finally:
        _in_leased_check = False
    return (VideoFolderKind(),)


EXTENSION = Extension(
    name="video",
    tools=_tools,
    lifecycle_hooks=_lifecycle_hooks,
    protected_roots=(ProtectedRoot(dirname=VIDEOS_DIRNAME, refusal=_WRITE_REFUSAL),),
    leased_folders=_leased_folders,
    routes=_routes,
    personas=(PERSONAS_DIR,),
)
