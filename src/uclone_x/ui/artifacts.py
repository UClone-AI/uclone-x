"""HTTP routes for the Files screen (#1554) and its story view (#1560): thin translations.

Every decision -- what is listed, what may move, whether a story is in use, whether a
proposed change may be applied -- is made by `uclone_x.artifacts.library.ArtifactLibrary`
or `uclone_x.story.view.StoryView` (P8). A route builds the library for the
workspace, calls one method, and maps a refusal to a status code carrying the refusal's
own plain sentence. A failure that is not a refusal is answered with a fixed sentence and
logged with its cause: no class name, path or traceback reaches the screen.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool

from uclone_x.artifacts.library import (
    ArtifactError,
    ArtifactLibrary,
    ArtifactNotFoundError,
    StoryInUseError,
)
from uclone_x.errors import PlainRefusalError, StaleRoomWriteError
from uclone_x.story.view import StoryNotFoundError, StoryView, WriterBusyError

if TYPE_CHECKING:
    from uclone_x.ui.rooms import RoomStack

logger = logging.getLogger(__name__)

#: What a failure that is not a refusal says. The cause is in the server log.
FILES_FAILURE_DETAIL = "The files could not be changed. The reason is in the server log."

#: The same, for the two reads.
FILES_READ_FAILURE_DETAIL = "The files could not be read. The reason is in the server log."

#: What a room written by someone else in the meantime says.
FILES_CONFLICT_DETAIL = "The conversation changed while this was being done. Refresh and try again."


class _PathRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    #: Strict: only a JSON `true` goes ahead over a writer, never `"yes"` or `1` (#1578).
    release_writer: StrictBool = False


class _DeleteRequest(_PathRequest):
    #: Strict: only a JSON `true` confirms a permanent delete, never `"yes"` or `1` (#1578).
    confirm: StrictBool = False


class _OpenStoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str


class _DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: The digest of the proposal as the view showed it.
    seen_digest: str


class _RejectRequest(_DecideRequest):
    reason: str | None = None


def _http_error(exc: Exception, *, failed: str = FILES_FAILURE_DETAIL) -> HTTPException:
    """A refusal keeps its sentence; anything else is logged and answered plainly."""
    if isinstance(exc, (StoryInUseError, WriterBusyError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (ArtifactNotFoundError, StoryNotFoundError)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, StaleRoomWriteError):
        return HTTPException(status_code=409, detail=FILES_CONFLICT_DETAIL)
    if isinstance(exc, (ArtifactError, PlainRefusalError)):
        return HTTPException(status_code=400, detail=str(exc))
    logger.error("A Files screen request failed", exc_info=exc)
    return HTTPException(status_code=500, detail=failed)


def register_artifact_routes(app: FastAPI, stack: RoomStack) -> None:
    """Mount the Files screen's routes under `/api/artifacts/library`."""

    def _turn_busy(room_id: str) -> bool:
        # A retry runs its turn inside the request, not as a cascade, so
        # `turn_in_flight` alone misses it; the dock asks the same two (#1578).
        return stack.turn_in_flight(room_id) or stack.turn_unlanded(room_id)

    def _library() -> ArtifactLibrary:
        return ArtifactLibrary(
            stack.session_manager().workspace_dir,
            stack.service,
            turn_in_flight=_turn_busy,
        )

    @app.get("/api/artifacts/library")
    async def survey_artifacts() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Every file in the artifact folders, current and archived, with what it covers."""
        try:
            return _library().survey().model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc, failed=FILES_READ_FAILURE_DETAIL) from exc

    @app.get("/api/artifacts/library/file")
    async def open_artifact(path: str = "") -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """One listed file's text, or for an image its kind (the bytes are at `/content`)."""
        try:
            return _library().open_file(path).model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc, failed=FILES_READ_FAILURE_DETAIL) from exc

    @app.post("/api/artifacts/library/archive")
    async def archive_artifact(body: _PathRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Move a file or story under the archive; 409 when a conversation is writing it."""
        try:
            moved = _library().archive(body.path, release_writer=body.release_writer)
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"path": moved.path, "note": moved.note}

    @app.post("/api/artifacts/library/restore")
    async def restore_artifact(body: _PathRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Move an archived file or story back where it was."""
        try:
            moved = _library().restore(body.path)
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"path": moved}

    @app.post("/api/artifacts/library/delete")
    async def delete_artifact(body: _DeleteRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Remove a file or story for good; refused unless `confirm` is true."""
        try:
            removed = _library().delete(
                body.path, confirm=body.confirm, release_writer=body.release_writer
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"deleted": body.path, "note": removed.note}

    @app.post("/api/artifacts/library/stories/{story_id}/open")
    async def open_story(story_id: str, body: _OpenStoryRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Open a story in the conversation `room_id` and make it that room's story."""
        try:
            opened = _library().open_story_in_conversation(story_id, body.room_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        return opened.model_dump(mode="json")

    def _story_view() -> StoryView:
        return StoryView(
            stack.session_manager().workspace_dir,
            stack.service,
            turn_busy=_turn_busy,
        )

    @app.get("/api/artifacts/library/stories/{story_id}")
    async def show_story(story_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """The story whole: outline, codex, and the proposed changes waiting for a person."""
        try:
            return _story_view().show(story_id).model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc, failed=FILES_READ_FAILURE_DETAIL) from exc

    @app.post("/api/artifacts/library/stories/{story_id}/proposals/{proposal_id}/approve")
    async def approve_proposal(  # pyright: ignore[reportUnusedFunction]
        story_id: str, proposal_id: str, body: _DecideRequest
    ) -> dict[str, Any]:
        """Apply a proposed change a person approved in the story view."""
        try:
            decided = _story_view().approve(story_id, proposal_id, seen_digest=body.seen_digest)
        except Exception as exc:
            raise _http_error(exc) from exc
        return decided.model_dump(mode="json")

    @app.post("/api/artifacts/library/stories/{story_id}/proposals/{proposal_id}/reject")
    async def reject_proposal(  # pyright: ignore[reportUnusedFunction]
        story_id: str, proposal_id: str, body: _RejectRequest
    ) -> dict[str, Any]:
        """Mark a proposed change rejected by a person in the story view."""
        try:
            decided = _story_view().reject(
                story_id, proposal_id, seen_digest=body.seen_digest, reason=body.reason
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return decided.model_dump(mode="json")
