"""HTTP routes for the interactive music score view (`MusicView`).

Every decision -- proposal approval/rejection, direct sheet-music bar edits, and orchestra
stem mix adjustments -- is executed by `uclone_x_music.view.MusicView`. Mutation endpoints
require person confirmation (`person.require(request)`), and errors are mapped via `http_error`.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from uclone_x.ui.artifacts import FILES_FAILURE_DETAIL, FILES_READ_FAILURE_DETAIL, http_error
from uclone_x.ui.extension_routes import ExtensionRouteContext
from uclone_x_music.arranger import ArrangementIntent
from uclone_x_music.view import (
    ComposerBusyError,
    DirectEditAction,
    MusicView,
    SongNotFoundError,
    SongNotWritableError,
    StructureEditAction,
)

__all__ = ["register_music_routes"]


class _DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seen_digest: str
    # The conversation the score view is docked beside. It takes the song over when no
    # live conversation holds it, so a decision is never refused only for that.
    room_id: str | None = None


class _ApproveAllRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str | None = None


class _RejectRequest(_DecideRequest):
    reason: str | None = None


class _DirectEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: DirectEditAction
    start_bar: int = 1
    end_bar: int = 1
    semitones: int = 0
    muted: bool = True
    notes: list[str] | None = None
    chord: str | None = None
    seen_digest: str


class _StructureEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: StructureEditAction
    section_id: str | None = None
    at_bar: int = 1
    count: int = 2
    start_bar: int = 1
    end_bar: int = 1
    chords: list[str] | None = None
    intent: ArrangementIntent | None = None


class _MixUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volume: int | None = None
    pan: float | None = None
    muted: bool | None = None
    solo: bool | None = None
    seen_digest: str | None = None


def _music_refusal(exc: Exception, *, failed: str = FILES_FAILURE_DETAIL) -> JSONResponse:
    """Format a `MusicView` refusal as `{"detail": ..., "code": ...}`."""
    http = http_error(
        exc,
        failed=failed,
        busy=(ComposerBusyError,),
        missing=(SongNotFoundError,),
    )
    body: dict[str, Any] = {"detail": http.detail}
    if isinstance(exc, (SongNotFoundError, SongNotWritableError, ComposerBusyError)):
        body["code"] = exc.code
    return JSONResponse(body, status_code=http.status_code)


def register_music_routes(context: ExtensionRouteContext) -> None:
    """Mount the music score view routes under `/api/artifacts/library/songs/{song_id}`."""
    app, stack, person = context.app, context.stack, context.person
    refuse_cross_origin = context.refuse_cross_origin
    # Every route that changes a song asks for the person's confirmed window and nothing
    # else. A same-origin check is not a substitute: a request with no `Origin` header, as
    # any program on this computer can send, passes it.
    require_person = person.require

    def _music_view() -> MusicView:
        return MusicView(
            stack.session_manager().workspace_dir,
            stack.service,
            turn_busy=context.turn_busy,
        )

    @app.get("/api/artifacts/library/songs/{song_id}")
    async def show_song(song_id: str, request: Request) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Load the song's score, orchestra, tracks, notes, and pending proposals."""
        refuse_cross_origin(request)
        try:
            return _music_view().show(song_id).model_dump(mode="json")
        except Exception as exc:
            return _music_refusal(exc, failed=FILES_READ_FAILURE_DETAIL)

    @app.post("/api/artifacts/library/songs/{song_id}/proposals/approve-all")
    async def approve_all_song_proposals(  # pyright: ignore[reportUnusedFunction]
        request: Request, song_id: str, body: _ApproveAllRequest | None = None
    ) -> Any:
        """Approve all pending music proposals in order from the score view."""
        require_person(request)
        try:
            shown = _music_view().approve_all(
                song_id, room_id=body.room_id if body is not None else None
            )
        except Exception as exc:
            return _music_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/songs/{song_id}/proposals/{proposal_id}/approve")
    async def approve_song_proposal(  # pyright: ignore[reportUnusedFunction]
        request: Request, song_id: str, proposal_id: str, body: _DecideRequest
    ) -> Any:
        """Approve a pending music proposal from the score view."""
        require_person(request)
        try:
            shown = _music_view().approve(
                song_id, proposal_id, seen_digest=body.seen_digest, room_id=body.room_id
            )
        except Exception as exc:
            return _music_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/songs/{song_id}/proposals/{proposal_id}/reject")
    async def reject_song_proposal(  # pyright: ignore[reportUnusedFunction]
        request: Request, song_id: str, proposal_id: str, body: _RejectRequest
    ) -> Any:
        """Reject a pending music proposal from the score view."""
        require_person(request)
        try:
            shown = _music_view().reject(
                song_id,
                proposal_id,
                seen_digest=body.seen_digest,
                reason=body.reason,
                room_id=body.room_id,
            )
        except Exception as exc:
            return _music_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/songs/{song_id}/structure")
    async def edit_song_structure(  # pyright: ignore[reportUnusedFunction]
        request: Request, song_id: str, body: _StructureEditRequest
    ) -> Any:
        """Insert, append, delete, or duplicate bars across all tracks of a section."""
        require_person(request)
        try:
            shown = _music_view().edit_structure(
                song_id,
                action=body.action,
                section_id=body.section_id,
                at_bar=body.at_bar,
                count=body.count,
                start_bar=body.start_bar,
                end_bar=body.end_bar,
                chords=body.chords,
                intent=body.intent,
            )
        except Exception as exc:
            return _music_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/songs/{song_id}/tracks/{track_id}/edit")
    async def edit_song_track(  # pyright: ignore[reportUnusedFunction]
        request: Request, song_id: str, track_id: str, body: _DirectEditRequest
    ) -> Any:
        """Apply a direct sheet-music edit (clear, mute, transpose, set_bar_notes, or revert) to a track."""
        require_person(request)
        try:
            shown = _music_view().direct_edit_track(
                song_id,
                track_id,
                action=body.action,
                start_bar=body.start_bar,
                end_bar=body.end_bar,
                semitones=body.semitones,
                muted=body.muted,
                notes=body.notes,
                chord=body.chord,
                seen_digest=body.seen_digest,
            )
        except Exception as exc:
            return _music_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/songs/{song_id}/instruments/{instrument_id}/mix")
    async def update_song_instrument_mix(  # pyright: ignore[reportUnusedFunction]
        request: Request, song_id: str, instrument_id: str, body: _MixUpdateRequest
    ) -> Any:
        """Update an instrument's mix parameters (volume, pan, muted, solo)."""
        require_person(request)
        try:
            shown = _music_view().update_mix(
                song_id,
                instrument_id,
                volume=body.volume,
                pan=body.pan,
                muted=body.muted,
                solo=body.solo,
                seen_digest=body.seen_digest,
            )
        except Exception as exc:
            return _music_refusal(exc)
        return shown.model_dump(mode="json")
