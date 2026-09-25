"""HTTP routes for the Files screen (#1554): thin translations over `ArtifactLibrary`.

Every decision -- what is listed, what may move, whether a story is in use -- is made by
`uclone_x.artifacts.library.ArtifactLibrary` (P8). A route builds the library for the
workspace, calls one method, and maps a refusal to a status code carrying the refusal's
own plain sentence. A failure that is not a refusal is answered with a fixed sentence and
logged with its cause: no class name, path or traceback reaches the screen.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from uclone_x.artifacts.library import (
    ArtifactError,
    ArtifactLibrary,
    ArtifactNotFoundError,
    StoryInUseError,
)
from uclone_x.errors import PlainRefusalError, StaleRoomWriteError

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
    release_writer: bool = False


class _DeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    confirm: bool = False
    release_writer: bool = False


class _OpenStoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str


def _http_error(exc: Exception, *, failed: str = FILES_FAILURE_DETAIL) -> HTTPException:
    """A refusal keeps its sentence; anything else is logged and answered plainly."""
    if isinstance(exc, StoryInUseError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ArtifactNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, StaleRoomWriteError):
        return HTTPException(status_code=409, detail=FILES_CONFLICT_DETAIL)
    if isinstance(exc, (ArtifactError, PlainRefusalError)):
        return HTTPException(status_code=400, detail=str(exc))
    logger.error("A Files screen request failed", exc_info=exc)
    return HTTPException(status_code=500, detail=failed)


def register_artifact_routes(app: FastAPI, stack: RoomStack) -> None:
    """Mount the Files screen's routes under `/api/artifacts/library`."""

    def _library() -> ArtifactLibrary:
        return ArtifactLibrary(
            stack.session_manager().workspace_dir,
            stack.service,
            turn_in_flight=stack.turn_in_flight,
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
        return {"path": moved}

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
            _library().delete(body.path, confirm=body.confirm, release_writer=body.release_writer)
        except Exception as exc:
            raise _http_error(exc) from exc
        return {"deleted": body.path}

    @app.post("/api/artifacts/library/stories/{story_id}/open")
    async def open_story(story_id: str, body: _OpenStoryRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Open a story in the conversation `room_id` and make it that room's story."""
        try:
            opened = _library().open_story_in_conversation(story_id, body.room_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        return opened.model_dump(mode="json")
