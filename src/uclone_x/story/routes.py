"""HTTP routes for the story view (#1560): thin translations, mounted by the story extension.

Every decision -- whether a proposed change may be applied, by whom -- is made by
`uclone_x.story.view.StoryView` (P8). A route builds the view for the workspace, calls one
method, and maps a refusal to a status code carrying the refusal's own plain sentence, the
way the Files screen's routes do (`ui.artifacts.http_error`). The web head mounts these
through the story extension (`story/ucx_extension.py`, #2205); the paths are unchanged.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from uclone_x.story.view import (
    StoryNotFoundError,
    StoryNotWritableError,
    StoryView,
    WriterBusyError,
)
from uclone_x.ui.artifacts import FILES_FAILURE_DETAIL, FILES_READ_FAILURE_DETAIL, http_error
from uclone_x.ui.extension_routes import ExtensionRouteContext

__all__ = ["register_story_routes"]


class _DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: The digest of the proposal as the view showed it.
    seen_digest: str


class _RejectRequest(_DecideRequest):
    reason: str | None = None


def _story_refusal(exc: Exception, *, failed: str = FILES_FAILURE_DETAIL) -> JSONResponse:
    """A story view refusal as `{"detail": <English sentence>, "code": <which reason>}`: the
    head words it from `code` in the reader's language (`dock.story.refusals`), and shows
    `detail` for a code it does not know. Anything else is answered as `http_error` does."""
    http = http_error(exc, failed=failed, busy=(WriterBusyError,), missing=(StoryNotFoundError,))
    body: dict[str, Any] = {"detail": http.detail}
    if isinstance(exc, (StoryNotFoundError, StoryNotWritableError, WriterBusyError)):
        body["code"] = exc.code
    return JSONResponse(body, status_code=http.status_code)


def register_story_routes(context: ExtensionRouteContext) -> None:
    """Mount the story view's routes under `/api/artifacts/library/stories/{story_id}`.

    Approve and reject record that a person decided (`decided_in: story_view`), so both
    first ask the person gate whether the request came from a window this server
    confirmed, and refuse before anything is read or written when it did not (#1589 item 6).
    """
    app, stack, person = context.app, context.stack, context.person
    refuse_cross_origin = context.refuse_cross_origin

    def _story_view() -> StoryView:
        return StoryView(
            stack.session_manager().workspace_dir,
            stack.service,
            turn_busy=context.turn_busy,
        )

    @app.get("/api/artifacts/library/stories/{story_id}")
    async def show_story(story_id: str, request: Request) -> Any:  # pyright: ignore[reportUnusedFunction]
        """The story whole: outline, codex, and the proposed changes waiting for a person."""
        refuse_cross_origin(request)
        try:
            return _story_view().show(story_id).model_dump(mode="json")
        except Exception as exc:
            return _story_refusal(exc, failed=FILES_READ_FAILURE_DETAIL)

    @app.post("/api/artifacts/library/stories/{story_id}/proposals/{proposal_id}/approve")
    async def approve_proposal(  # pyright: ignore[reportUnusedFunction]
        request: Request, story_id: str, proposal_id: str, body: _DecideRequest
    ) -> Any:
        """Apply a proposed change a person approved in the story view."""
        person.require(request)
        try:
            decided = _story_view().approve(story_id, proposal_id, seen_digest=body.seen_digest)
        except Exception as exc:
            return _story_refusal(exc)
        return decided.model_dump(mode="json")

    @app.post("/api/artifacts/library/stories/{story_id}/proposals/{proposal_id}/reject")
    async def reject_proposal(  # pyright: ignore[reportUnusedFunction]
        request: Request, story_id: str, proposal_id: str, body: _RejectRequest
    ) -> Any:
        """Mark a proposed change rejected by a person in the story view."""
        person.require(request)
        try:
            decided = _story_view().reject(
                story_id, proposal_id, seen_digest=body.seen_digest, reason=body.reason
            )
        except Exception as exc:
            return _story_refusal(exc)
        return decided.model_dump(mode="json")
