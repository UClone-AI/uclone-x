"""`music_library`: list, create, open, take over, and close songs in the workspace.

A song is a workspace artifact under `songs/<song_id>/` that a conversation opens; see
`uclone_x_music.library`. This tool is how a conversation moves between songs. It takes no
filesystem path from the model: the only identifier it accepts is a `song_id`, which
`SongLibrary.song_dir` validates.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x_music import OPEN_SONG_KEY
from uclone_x_music.library import (
    SONG_FILE,
    SONGS_DIRNAME,
    MusicError,
    NoConversationError,
    NoOpenSongError,
    SongLibrary,
    SongRecord,
)
from uclone_x_music.schemas import StyleProfile
from uclone_x_music.work import SongWork

__all__ = ["MusicLibraryParams", "MusicLibraryTool"]


class MusicLibraryParams(BaseModel):
    """Parameters for `music_library`."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["list", "create", "open", "take_over", "close"] = Field(
        description="'list' songs in the workspace; 'create' a new song (requires 'title'); "
        "'open' an existing song (requires 'song_id'); 'take_over' a song lease held by "
        "another conversation (requires 'song_id'); 'close' the currently open song."
    )
    title: str | None = Field(default=None, description="The new song's title, for 'create'.")
    genre: str | None = Field(
        default=None, description="The song's genre description, for 'create'."
    )
    logline: str | None = Field(
        default=None,
        description="A short description of the song's mood and concept, for 'create'.",
    )
    style_profile: StyleProfile | None = Field(
        default=None,
        description="Style axiom profile ('pop_ballad', 'hiphop_groove', 'jazz_swing', "
        "or 'classical_strict') for 'create'.",
    )
    song_id: str | None = Field(
        default=None,
        description="The song id (for 'open', 'take_over', or an explicit id on 'create').",
    )


def _song_file(song_id: str) -> str:
    return f"{SONGS_DIRNAME}/{song_id}/{SONG_FILE}"


def _summary(record: SongRecord, conversation_id: str | None) -> dict[str, Any]:
    lease = record.lease
    if lease is None:
        writer = "nobody"
    elif lease.holder == conversation_id:
        writer = "this conversation"
    else:
        writer = "another conversation"
    return {
        "song_id": record.id,
        "title": record.title,
        "genre": record.genre,
        "style_profile": record.style_axioms.profile,
        "being_written_by": writer,
    }


class MusicLibraryTool(BaseTool[MusicLibraryParams]):
    """List, create, open, take over, or close workspace songs."""

    name = "music_library"
    description = (
        "List, create, open, take over, or close songs in the workspace. A song lives under "
        "songs/<song_id>/ and outlives the conversation. One conversation holds the write "
        "lease at a time; opening a song leased to another conversation opens it read-only "
        "until 'take_over' is called."
    )
    params_type = MusicLibraryParams
    writes_files: ClassVar[bool] = True
    opens_story: ClassVar[bool] = True
    needs_room: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"list"})

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=MusicLibraryParams,
        )

    async def run(self, params: MusicLibraryParams, context: ToolContext) -> dict[str, Any]:
        library = SongLibrary(context.require_workspace())
        conversation = context.room_id
        if params.action == "list":
            listing = library.list_songs()
            result: dict[str, Any] = {
                "songs": [_summary(r, conversation) for r in listing.songs],
                "open_here": context.story_id,
            }
            if listing.unreadable:
                result["unreadable"] = [
                    {"song_id": u.id, "reason": u.reason} for u in listing.unreadable
                ]
            return result

        if conversation is None or not conversation.strip():
            raise NoConversationError(
                "Songs are opened inside a conversation, and this call has no active conversation."
            )

        if params.action == "close":
            return self._close(library, context.story_id, conversation)

        if params.action == "create":
            if params.title is None or not params.title.strip():
                raise MusicError("A new song requires a non-empty 'title'.")
            profile: StyleProfile = params.style_profile or "pop_ballad"
            genre = params.genre.strip() if params.genre and params.genre.strip() else profile
            logline = params.logline.strip() if params.logline else ""
            explicit_id = (
                params.song_id.strip() if params.song_id and params.song_id.strip() else None
            )
            record = library.create(
                params.title,
                conversation,
                genre=genre,
                logline=logline,
                style_profile=profile,
                song_id=explicit_id,
            )
            released = self._leave(library, context.story_id, conversation, keep=None)
            SongWork(library, record.id, conversation).record_session("Created song")
            return {
                OPEN_SONG_KEY: record.id,
                "song_id": record.id,
                "title": record.title,
                "style_profile": record.style_axioms.profile,
                "writable": True,
                "path": _song_file(record.id),
                **released,
            }

        if params.song_id is None or not params.song_id.strip():
            raise MusicError(f"Specify 'song_id' for action {params.action!r}.")
        song_id = params.song_id.strip()

        if params.action == "open":
            opening = library.open(song_id, conversation)
            released = self._leave(library, context.story_id, conversation, keep=song_id)
            result = {
                OPEN_SONG_KEY: song_id,
                "song_id": song_id,
                "title": opening.song.title,
                "writable": opening.writable,
                "path": _song_file(song_id),
                **released,
            }
            if opening.writable:
                SongWork(library, song_id, conversation).record_session("Opened song")
            else:
                result["note"] = (
                    "Another conversation holds the write lease on this song, so it is open "
                    "read-only. Use 'take_over' if the person wants to edit it here."
                )
            return result

        # take_over
        record, previous = library.take_over(song_id, conversation)
        released = self._leave(library, context.story_id, conversation, keep=song_id)
        SongWork(library, song_id, conversation).record_session(
            f"Took over lease from {previous or 'none'}"
        )
        return {
            OPEN_SONG_KEY: song_id,
            "song_id": song_id,
            "title": record.title,
            "writable": True,
            "taken_from_another_conversation": previous not in (None, conversation),
            "path": _song_file(song_id),
            **released,
        }

    @staticmethod
    def _close(library: SongLibrary, song_id: str | None, conversation: str) -> dict[str, Any]:
        if song_id is None:
            raise NoOpenSongError("No song is open in this conversation, so none was closed.")
        result: dict[str, Any] = {OPEN_SONG_KEY: None, "closed": song_id}
        try:
            released = library.release(song_id, conversation)
        except MusicError as exc:
            result["not_released"] = str(exc)
            return result
        if released:
            result["path"] = _song_file(song_id)
        return result

    @staticmethod
    def _leave(
        library: SongLibrary,
        current: str | None,
        conversation: str,
        *,
        keep: str | None,
    ) -> dict[str, Any]:
        if current is None or current == keep:
            return {}
        try:
            released = library.release(current, conversation)
        except MusicError:
            return {"previous_song_not_released": current}
        return {"released": current} if released else {}
