"""`video_library`: list, create, open, take over, and close video projects in the workspace.

A video project is a workspace artifact under `videos/<project_id>/` that a conversation
opens; see `uclone_x_video.library`. This tool is how a conversation moves between video
projects. It takes no filesystem path from the model: the only identifier it accepts is a
`project_id`, which `VideoLibrary.project_dir` validates.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x_video import OPEN_VIDEO_KEY
from uclone_x_video.library import (
    PROJECT_FILE,
    VIDEOS_DIRNAME,
    NoConversationError,
    NoOpenVideoError,
    VideoError,
    VideoLibrary,
    VideoProjectRecord,
)
from uclone_x_video.schemas import StyleProfile
from uclone_x_video.work import VideoWork

__all__ = ["VideoLibraryParams", "VideoLibraryTool"]


class VideoLibraryParams(BaseModel):
    """Parameters for `video_library`."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["list", "create", "open", "take_over", "close"] = Field(
        description="'list' video projects in the workspace; 'create' a new project "
        "(requires 'title'); 'open' an existing project (requires 'project_id'); "
        "'take_over' a project lease held by another conversation (requires 'project_id'); "
        "'close' the currently open project."
    )
    title: str | None = Field(default=None, description="The new project's title, for 'create'.")
    logline: str | None = Field(
        default=None,
        description="A short description of the video's concept and mood, for 'create'.",
    )
    style_profile: StyleProfile | None = Field(
        default=None,
        description="Visual editing style profile ('cinematic_film', 'documentary_clean', "
        "'fast_social_cut', or 'vintage_analog') for 'create'.",
    )
    fps: int | None = Field(
        default=None,
        ge=1,
        le=240,
        description="Sequence frame rate (default 30) for 'create'.",
    )
    project_id: str | None = Field(
        default=None,
        description="The video project id (for 'open', 'take_over', or an explicit id on 'create').",
    )


def _project_file(project_id: str) -> str:
    return f"{VIDEOS_DIRNAME}/{project_id}/{PROJECT_FILE}"


def _summary(record: VideoProjectRecord, conversation_id: str | None) -> dict[str, Any]:
    lease = record.lease
    if lease is None:
        writer = "nobody"
    elif lease.holder == conversation_id:
        writer = "this conversation"
    else:
        writer = "another conversation"
    return {
        "project_id": record.id,
        "title": record.title,
        "style_profile": record.style_profile,
        "fps": record.fps,
        "active_playback_tier": record.active_playback_tier,
        "being_written_by": writer,
    }


class VideoLibraryTool(BaseTool[VideoLibraryParams]):
    """List, create, open, take over, or close workspace video projects."""

    name = "video_library"
    description = (
        "List, create, open, take over, or close video projects in the workspace. A video "
        "project lives under videos/<project_id>/ and outlives the conversation. One "
        "conversation holds the write lease at a time; opening a project leased to another "
        "conversation opens it read-only until 'take_over' is called."
    )
    params_type = VideoLibraryParams
    writes_files: ClassVar[bool] = True
    opens_story: ClassVar[bool] = True
    needs_room: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"list"})

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=VideoLibraryParams,
        )

    async def run(self, params: VideoLibraryParams, context: ToolContext) -> dict[str, Any]:
        library = VideoLibrary(context.require_workspace())
        conversation = context.room_id
        if params.action == "list":
            listing = library.list_projects()
            result: dict[str, Any] = {
                "projects": [_summary(r, conversation) for r in listing.projects],
                "open_here": context.story_id,
            }
            if listing.unreadable:
                result["unreadable"] = [
                    {"project_id": u.id, "reason": u.reason} for u in listing.unreadable
                ]
            return result

        if conversation is None or not conversation.strip():
            raise NoConversationError(
                "Video projects are opened inside a conversation, and this call has no active conversation."
            )

        if params.action == "close":
            return self._close(library, context.story_id, conversation)

        if params.action == "create":
            if params.title is None or not params.title.strip():
                raise VideoError("A new video project requires a non-empty 'title'.")
            profile: StyleProfile = params.style_profile or "cinematic_film"
            logline = params.logline.strip() if params.logline else ""
            fps = params.fps or 30
            explicit_id = (
                params.project_id.strip()
                if params.project_id and params.project_id.strip()
                else None
            )
            record = library.create(
                params.title,
                conversation,
                style_profile=profile,
                logline=logline,
                fps=fps,
                project_id=explicit_id,
            )
            released = self._leave(library, context.story_id, conversation, keep=None)
            VideoWork(library, record.id, conversation).record_session("Created video project")
            return {
                OPEN_VIDEO_KEY: record.id,
                "project_id": record.id,
                "title": record.title,
                "style_profile": record.style_profile,
                "fps": record.fps,
                "writable": True,
                "path": _project_file(record.id),
                **released,
            }

        if params.project_id is None or not params.project_id.strip():
            raise VideoError(f"Specify 'project_id' for action {params.action!r}.")
        project_id = params.project_id.strip()

        if params.action == "open":
            opening = library.open(project_id, conversation)
            released = self._leave(library, context.story_id, conversation, keep=project_id)
            result = {
                OPEN_VIDEO_KEY: project_id,
                "project_id": project_id,
                "title": opening.project.title,
                "writable": opening.writable,
                "path": _project_file(project_id),
                **released,
            }
            if opening.writable:
                VideoWork(library, project_id, conversation).record_session("Opened video project")
            else:
                result["note"] = (
                    "Another conversation holds the write lease on this project, so it is open "
                    "read-only. Use 'take_over' if the person wants to edit it here."
                )
            return result

        record, previous = library.take_over(project_id, conversation)
        released = self._leave(library, context.story_id, conversation, keep=project_id)
        VideoWork(library, project_id, conversation).record_session(
            f"Took over lease from {previous or 'none'}"
        )
        return {
            OPEN_VIDEO_KEY: project_id,
            "project_id": project_id,
            "title": record.title,
            "writable": True,
            "taken_from_another_conversation": previous not in (None, conversation),
            "path": _project_file(project_id),
            **released,
        }

    @staticmethod
    def _close(library: VideoLibrary, project_id: str | None, conversation: str) -> dict[str, Any]:
        if project_id is None:
            raise NoOpenVideoError(
                "No video project is open in this conversation, so none was closed."
            )
        result: dict[str, Any] = {OPEN_VIDEO_KEY: None, "closed": project_id}
        try:
            released = library.release(project_id, conversation)
        except VideoError as exc:
            result["not_released"] = str(exc)
            return result
        if released:
            result["path"] = _project_file(project_id)
        return result

    @staticmethod
    def _leave(
        library: VideoLibrary,
        current: str | None,
        conversation: str,
        *,
        keep: str | None,
    ) -> dict[str, Any]:
        if current is None or current == keep:
            return {}
        try:
            released = library.release(current, conversation)
        except VideoError:
            return {"previous_project_not_released": current}
        return {"released": current} if released else {}
