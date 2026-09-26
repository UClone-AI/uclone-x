"""`story_library`: list, create, open, take over and close stories (#1555).

A story is a workspace artifact under `stories/<story_id>/` that a conversation opens; see
`uclone_x.story.library`. This tool is how a conversation moves between stories. It takes no path
from the model: the only name it accepts is a story id, which `StoryLibrary.root` checks.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.story import OPEN_STORY_KEY
from uclone_x.story.library import (
    STORIES_DIRNAME,
    STORY_FILE,
    NoConversationError,
    NoOpenStoryError,
    StoryError,
    StoryLibrary,
    StoryRecord,
)
from uclone_x.story.schemas import StoryFileError
from uclone_x.story.work import StoryWork
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext

__all__ = ["StoryLibraryParams", "StoryLibraryTool"]


class StoryLibraryParams(BaseModel):
    """What to do in the story library."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["list", "create", "open", "take_over", "close"] = Field(
        description="'list' the stories; 'create' a new one (needs 'title'); 'open' one "
        "(needs 'story_id'); 'take_over' a story another conversation is writing (needs "
        "'story_id'); 'close' the story open in this conversation."
    )
    title: str | None = Field(default=None, description="The new story's title, for 'create'.")
    genre: str | None = Field(default=None, description="The story's genre, for 'create'.")
    logline: str | None = Field(
        default=None, description="The story in one or two sentences, for 'create'."
    )
    style_notes: str | None = Field(
        default=None,
        description="How the story is written -- voice, tense, point of view, what to avoid "
        "-- for 'create'.",
    )
    story_id: str | None = Field(
        default=None,
        description="The story's id as 'list' shows it, for 'open' and 'take_over'.",
    )


def _story_file(story_id: str) -> str:
    return f"{STORIES_DIRNAME}/{story_id}/{STORY_FILE}"


def _summary(record: StoryRecord, conversation_id: str | None) -> dict[str, Any]:
    lease = record.lease
    if lease is None:
        writer = "nobody"
    elif lease.holder == conversation_id:
        writer = "this conversation"
    else:
        writer = "another conversation"
    return {"story_id": record.story_id, "title": record.title, "being_written_by": writer}


class StoryLibraryTool(BaseTool[StoryLibraryParams]):
    """The conversation's way into its stories."""

    name = "story_library"
    description = (
        "List, create, open, take over or close stories. A story is kept in the workspace "
        "and outlives the conversation. One conversation writes a story at a time: opening "
        "a story another conversation is writing opens it read-only, and 'take_over' moves "
        "the writing here when the person asks for it."
    )
    params_type = StoryLibraryParams
    # It writes `story.yaml`: a new story, and the lease that says who writes it.
    writes_files: ClassVar[bool] = True
    opens_story: ClassVar[bool] = True
    # 'list' works outside a conversation, but nothing there can open what it lists, so
    # the tool is kept out of such a turn like the other story tools (#1576).
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name, description=self.description, params_type=StoryLibraryParams
        )

    async def run(self, params: StoryLibraryParams, context: ToolContext) -> dict[str, Any]:
        # No await below: each action runs to the end before another call can start, so
        # two conversations on one Core cannot interleave inside a lease change.
        library = StoryLibrary(context.require_workspace())
        conversation = context.room_id
        if params.action == "list":
            listing = library.list()
            result: dict[str, Any] = {
                "stories": [_summary(r, conversation) for r in listing.stories],
                "open_here": context.story_id,
            }
            if listing.unreadable:
                result["unreadable"] = [
                    {"story_id": u.story_id, "reason": u.reason} for u in listing.unreadable
                ]
            return result
        if conversation is None:
            raise NoConversationError(
                "Stories are opened inside a conversation, and this call is not part of one, "
                "so nothing was done."
            )
        if params.action == "close":
            return self._close(library, context.story_id, conversation)
        if params.action == "create":
            if params.title is None or not params.title.strip():
                raise StoryError("A new story needs a title, so nothing was created.")
            record = library.create(
                params.title,
                conversation,
                genre=params.genre,
                logline=params.logline,
                style_notes=params.style_notes,
            )
            # The story it had open is given back only once the new one exists: a refused
            # create leaves the conversation where it was, lease and all (#1556).
            released = self._leave(library, context.story_id, conversation, keep=None)
            return {
                OPEN_STORY_KEY: record.story_id,
                "title": record.title,
                "writable": True,
                "path": _story_file(record.story_id),
                **released,
                **self._session_opened(library, record.story_id, conversation),
            }
        if params.story_id is None or not params.story_id.strip():
            raise StoryError(f"Say which story to {params.action.replace('_', ' ')}.")
        story_id = params.story_id.strip()
        if params.action == "open":
            opening = library.open(story_id, conversation)
            released = self._leave(library, context.story_id, conversation, keep=story_id)
            result = {
                OPEN_STORY_KEY: story_id,
                "title": opening.story.title,
                "writable": opening.writable,
                **released,
            }
            if opening.acquired:
                result["path"] = _story_file(story_id)
            if opening.writable:
                result.update(self._session_opened(library, story_id, conversation))
            if not opening.writable:
                result["note"] = (
                    "Another conversation is writing this story, so it is open here to "
                    "read only. If the person wants to continue it here, use 'take_over'."
                )
            return result
        # take_over
        record, previous = library.take_over(story_id, conversation)
        released = self._leave(library, context.story_id, conversation, keep=story_id)
        result = {
            OPEN_STORY_KEY: story_id,
            "title": record.title,
            "writable": True,
            "taken_from_another_conversation": previous not in (None, conversation),
            **released,
        }
        if previous != conversation:
            result["path"] = _story_file(story_id)
        result.update(self._session_opened(library, story_id, conversation, taken_from=previous))
        return result

    @staticmethod
    def _close(library: StoryLibrary, story_id: str | None, conversation: str) -> dict[str, Any]:
        if story_id is None:
            raise NoOpenStoryError("No story is open in this conversation, so none was closed.")
        result: dict[str, Any] = {OPEN_STORY_KEY: None, "closed": story_id}
        result.update(StoryLibraryTool._session_closed(library, story_id, conversation))
        try:
            released = library.release(story_id, conversation)
        except StoryError as exc:
            # A story removed or broken outside the conversation still closes here, or the
            # conversation could never let go of it; the reason travels with the result.
            result["not_released"] = str(exc)
            return result
        if released:
            result["path"] = _story_file(story_id)
        return result

    @staticmethod
    def _leave(
        library: StoryLibrary, current: str | None, conversation: str, *, keep: str | None
    ) -> dict[str, Any]:
        """Release the story this conversation had open before it moves to another."""
        if current is None or current == keep:
            return {}
        noted = StoryLibraryTool._session_closed(library, current, conversation)
        try:
            released = library.release(current, conversation)
        except StoryError:
            # The story it had open is gone or unreadable; there is no lease to give back.
            return {"previous_story_not_released": current}
        return {"released": current, **noted} if released else noted

    @staticmethod
    def _session_opened(
        library: StoryLibrary, story_id: str, conversation: str, *, taken_from: str | None = None
    ) -> dict[str, Any]:
        """Start this conversation's entry in `sessions.yaml`, for a later recap.

        After a take-over, the entry of the conversation it was taken from is closed too.
        The story is open either way; a record that could not be written is said in the
        result rather than dropped (P6).
        """
        try:
            StoryWork(library, story_id).note_session(conversation, taken_from=taken_from)
        except (StoryError, StoryFileError) as exc:
            return {"session_not_recorded": str(exc)}
        return {}

    @staticmethod
    def _session_closed(library: StoryLibrary, story_id: str, conversation: str) -> dict[str, Any]:
        """Close this conversation's entry in `sessions.yaml`, while it still holds the lease."""
        try:
            StoryWork(library, story_id).note_session(conversation, closing=True)
        except (StoryError, StoryFileError) as exc:
            return {"session_not_recorded": str(exc)}
        return {}
