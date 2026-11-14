"""HTTP routes for the interactive DaVinci-style NLE view (`VideoView`).

Every decision -- proposal approval/rejection, direct timeline cuts/grades/titles, quality
tier switching, master rendering, and derived asset baking -- is executed by
`uclone_x_video.view.VideoView`. Mutation endpoints require person confirmation
(`person.require(request)`), and errors are mapped via `http_error`.
"""

from __future__ import annotations

import base64
from typing import Any

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from uclone_x.ui.artifacts import FILES_FAILURE_DETAIL, FILES_READ_FAILURE_DETAIL, http_error
from uclone_x.ui.extension_routes import ExtensionRouteContext
from uclone_x_video.schemas import BakeOpKind, QualityTier, StyleProfile
from uclone_x_video.view import (
    DirectorBusyError,
    DirectTimelineAction,
    VideoNotFoundError,
    VideoNotWritableError,
    VideoView,
)

__all__ = ["register_video_routes"]


class _CreateProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = "New Studio Reel"
    style_profile: StyleProfile = "cinematic_film"
    fps: int = 30
    width: int = 1920
    height: int = 1080
    project_id: str | None = None
    seed_sample: bool = True
    room_id: str | None = None


class _OpenProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str | None = None


class _DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seen_digest: str
    room_id: str | None = None


class _ApproveAllRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str | None = None


class _RejectRequest(_DecideRequest):
    reason: str | None = None


class _TierSwitchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: QualityTier
    room_id: str | None = None


class _RenderMasterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str | None = None


class _IngestMediaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    filename: str
    fps: int = 30
    total_frames: int = 300
    width: int = 1920
    height: int = 1080
    payload_b64: str | None = None
    room_id: str | None = None


class _BakeDerivedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_asset_id: str
    source_in_frame: int = 0
    source_out_frame: int = 150
    operation_kind: BakeOpKind = "optical_flow_slowmo"
    rate: float = 0.5
    strength: float = 0.8
    derived_asset_id: str | None = None
    track_id: str | None = None
    clip_id: str | None = None
    room_id: str | None = None


class _SearchFramesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    asset_id: str | None = None
    shot_type: str | None = None
    top_k: int = 6


class _DirectTimelineEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: DirectTimelineAction
    seen_digest: str
    clip_id: str | None = None
    asset_id: str | None = None
    label: str | None = None
    split_at_frame: int | None = None
    timeline_in_frame: int | None = None
    timeline_out_frame: int | None = None
    source_in_frame: int | None = None
    source_out_frame: int | None = None
    ripple: bool = False
    color_grade: dict[str, Any] | None = None
    transition_out: dict[str, Any] | None = None
    title_clip: dict[str, Any] | None = None
    title_id: str | None = None
    derived_asset_id: str | None = None
    muted: bool | None = None
    solo: bool | None = None
    locked: bool | None = None
    room_id: str | None = None


def _video_refusal(exc: Exception, *, failed: str = FILES_FAILURE_DETAIL) -> JSONResponse:
    """Format a `VideoView` refusal as `{"detail": ..., "code": ...}`."""
    http = http_error(
        exc,
        failed=failed,
        busy=(DirectorBusyError,),
        missing=(VideoNotFoundError,),
    )
    body: dict[str, Any] = {"detail": http.detail}
    if isinstance(exc, (VideoNotFoundError, VideoNotWritableError, DirectorBusyError)):
        body["code"] = exc.code
    return JSONResponse(body, status_code=http.status_code)


def register_video_routes(context: ExtensionRouteContext) -> None:
    """Mount the video NLE studio routes under `/api/artifacts/library/videos/{project_id}`."""
    app, stack, person = context.app, context.stack, context.person
    refuse_cross_origin = context.refuse_cross_origin
    # Every route that changes a video project asks for the person's confirmed window and
    # nothing else.
    require_person = person.require

    def _video_view() -> VideoView:
        return VideoView(
            stack.session_manager().workspace_dir,
            stack.service,
            turn_busy=context.turn_busy,
        )

    @app.get("/api/artifacts/library/videos")
    async def list_video_projects(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ) -> Any:
        """List all video projects in the workspace library."""
        refuse_cross_origin(request)
        try:
            return _video_view().list_projects()
        except Exception as exc:
            return _video_refusal(exc, failed=FILES_READ_FAILURE_DETAIL)

    @app.post("/api/artifacts/library/videos")
    async def create_video_project(  # pyright: ignore[reportUnusedFunction]
        request: Request, body: _CreateProjectRequest
    ) -> Any:
        """Create a new video project (optionally seeded with sample footage) and bind it to `room_id`."""
        refuse_cross_origin(request)
        try:
            shown = _video_view().create_project(
                title=body.title,
                style_profile=body.style_profile,
                fps=body.fps,
                width=body.width,
                height=body.height,
                project_id=body.project_id,
                seed_sample=body.seed_sample,
                room_id=body.room_id,
            )
            return shown.model_dump(mode="json")
        except Exception as exc:
            return _video_refusal(exc)

    @app.post("/api/artifacts/library/videos/{project_id}/open")
    async def open_video_project_in_room(  # pyright: ignore[reportUnusedFunction]
        request: Request, project_id: str, body: _OpenProjectRequest | None = None
    ) -> Any:
        """Open `project_id` in `room_id` and bind `room.story_id = project_id`."""
        refuse_cross_origin(request)
        try:
            shown = _video_view().open_project(
                project_id,
                room_id=body.room_id if body is not None else None,
            )
            return shown.model_dump(mode="json")
        except Exception as exc:
            return _video_refusal(exc)

    @app.get("/api/artifacts/library/videos/{project_id}/stream/{tier}/{target_id}")
    async def stream_video_tier_file(  # pyright: ignore[reportUnusedFunction]
        project_id: str, tier: QualityTier, target_id: str, request: Request
    ) -> Any:
        """Stream the on-disk media file (`proxies/`, `vault/`, or `exports/`) with HTTP Range support."""
        refuse_cross_origin(request)
        try:
            path = _video_view().resolve_stream_path(project_id, tier, target_id)
            return FileResponse(path, media_type="video/mp4", filename=path.name)
        except Exception as exc:
            return _video_refusal(exc, failed=FILES_READ_FAILURE_DETAIL)

    @app.get("/api/artifacts/library/videos/{project_id}")
    async def show_video_project(  # pyright: ignore[reportUnusedFunction]
        project_id: str, request: Request, tier: QualityTier | None = None
    ) -> Any:
        """Load the video project's manifest, media pool, tracks, playback bundle, and pending proposals."""
        refuse_cross_origin(request)
        try:
            return _video_view().show(project_id, tier=tier).model_dump(mode="json")
        except Exception as exc:
            return _video_refusal(exc, failed=FILES_READ_FAILURE_DETAIL)

    @app.post("/api/artifacts/library/videos/{project_id}/search")
    async def search_video_frames(  # pyright: ignore[reportUnusedFunction]
        project_id: str, request: Request, body: _SearchFramesRequest
    ) -> Any:
        """Search indexed frames semantically or lexically across the video project."""
        refuse_cross_origin(request)
        try:
            ranking = await _video_view().search_frames(
                project_id,
                query=body.query,
                asset_id=body.asset_id,
                shot_type=body.shot_type,
                top_k=body.top_k,
            )
            return ranking.model_dump(mode="json")
        except Exception as exc:
            return _video_refusal(exc, failed=FILES_READ_FAILURE_DETAIL)

    @app.post("/api/artifacts/library/videos/{project_id}/tier")
    async def switch_video_tier(  # pyright: ignore[reportUnusedFunction]
        request: Request, project_id: str, body: _TierSwitchRequest
    ) -> Any:
        """Switch the active resolution/quality tier (`preview` 540p, `source` raw, `master` final)."""
        refuse_cross_origin(request)
        try:
            shown = _video_view().set_tier(project_id, body.tier, room_id=body.room_id)
        except Exception as exc:
            return _video_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/videos/{project_id}/render-master")
    async def render_video_master(  # pyright: ignore[reportUnusedFunction]
        request: Request, project_id: str, body: _RenderMasterRequest | None = None
    ) -> Any:
        """Render the Tier 3 `master` deliverable by relinking from `preview` proxies to Tier 2 `source` raw/derived vault assets."""
        refuse_cross_origin(request)
        try:
            shown = _video_view().render_master(
                project_id,
                room_id=body.room_id if body is not None else None,
            )
        except Exception as exc:
            return _video_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/videos/{project_id}/media")
    async def ingest_video_media(  # pyright: ignore[reportUnusedFunction]
        request: Request, project_id: str, body: _IngestMediaRequest
    ) -> Any:
        """Ingest an immutable raw video into `vault/`, generate its Tier 1 540p proxy, and index its frames."""
        refuse_cross_origin(request)
        try:
            raw_bytes: bytes | None = None
            if body.payload_b64:
                raw_bytes = base64.b64decode(body.payload_b64)
            shown = _video_view().ingest_media(
                project_id,
                asset_id=body.asset_id,
                filename=body.filename,
                fps=body.fps,
                total_frames=body.total_frames,
                width=body.width,
                height=body.height,
                raw_bytes=raw_bytes,
                room_id=body.room_id,
            )
        except Exception as exc:
            return _video_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/videos/{project_id}/bake")
    async def bake_video_derived_asset(  # pyright: ignore[reportUnusedFunction]
        request: Request, project_id: str, body: _BakeDerivedRequest
    ) -> Any:
        """Bake a new immutable derived asset (`kind='derived'`) from a raw asset while preserving the original raw untouched."""
        refuse_cross_origin(request)
        try:
            shown = _video_view().bake_derived(
                project_id,
                parent_asset_id=body.parent_asset_id,
                source_in_frame=body.source_in_frame,
                source_out_frame=body.source_out_frame,
                operation_kind=body.operation_kind,
                rate=body.rate,
                strength=body.strength,
                derived_asset_id=body.derived_asset_id,
                track_id=body.track_id,
                clip_id=body.clip_id,
                room_id=body.room_id,
            )
        except Exception as exc:
            return _video_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/videos/{project_id}/tracks/{track_id}/edit")
    async def edit_video_track(  # pyright: ignore[reportUnusedFunction]
        request: Request, project_id: str, track_id: str, body: _DirectTimelineEditRequest
    ) -> Any:
        """Apply a direct NLE edit (split, trim, delete, grade, title, swap_to_derived, unbake_to_raw, revert) to a track."""
        refuse_cross_origin(request)
        try:
            shown = _video_view().direct_edit_track(
                project_id,
                track_id,
                action=body.action,
                seen_digest=body.seen_digest,
                clip_id=body.clip_id,
                asset_id=body.asset_id,
                label=body.label,
                split_at_frame=body.split_at_frame,
                timeline_in_frame=body.timeline_in_frame,
                timeline_out_frame=body.timeline_out_frame,
                source_in_frame=body.source_in_frame,
                source_out_frame=body.source_out_frame,
                ripple=body.ripple,
                color_grade=body.color_grade,
                transition_out=body.transition_out,
                title_clip=body.title_clip,
                title_id=body.title_id,
                derived_asset_id=body.derived_asset_id,
                muted=body.muted,
                solo=body.solo,
                locked=body.locked,
                room_id=body.room_id,
            )
        except Exception as exc:
            return _video_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/videos/{project_id}/proposals/approve-all")
    async def approve_all_video_proposals(  # pyright: ignore[reportUnusedFunction]
        request: Request, project_id: str, body: _ApproveAllRequest | None = None
    ) -> Any:
        """Approve all pending video proposals in order from `VideoView`."""
        require_person(request)
        try:
            shown = _video_view().approve_all(
                project_id, room_id=body.room_id if body is not None else None
            )
        except Exception as exc:
            return _video_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/videos/{project_id}/proposals/{proposal_id}/approve")
    async def approve_video_proposal(  # pyright: ignore[reportUnusedFunction]
        request: Request, project_id: str, proposal_id: str, body: _DecideRequest
    ) -> Any:
        """Approve a pending video proposal from `VideoView`."""
        require_person(request)
        try:
            shown = _video_view().approve(
                project_id,
                proposal_id,
                seen_digest=body.seen_digest,
                room_id=body.room_id,
            )
        except Exception as exc:
            return _video_refusal(exc)
        return shown.model_dump(mode="json")

    @app.post("/api/artifacts/library/videos/{project_id}/proposals/{proposal_id}/reject")
    async def reject_video_proposal(  # pyright: ignore[reportUnusedFunction]
        request: Request, project_id: str, proposal_id: str, body: _RejectRequest
    ) -> Any:
        """Reject a pending video proposal from `VideoView`."""
        require_person(request)
        try:
            shown = _video_view().reject(
                project_id,
                proposal_id,
                seen_digest=body.seen_digest,
                reason=body.reason,
                room_id=body.room_id,
            )
        except Exception as exc:
            return _video_refusal(exc)
        return shown.model_dump(mode="json")
