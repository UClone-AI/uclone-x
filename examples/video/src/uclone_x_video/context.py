"""Pull-based timeline context bundle for interactive video editing (`uclone_x_video.context`).

Computes the active sequence's track summary, media pool with 3-tier and derived lineage
status, selected region window, and context manifest so the Director agent can inspect and
edit a timeline without overflowing its context window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from uclone_x_video.digests import asset_line, clip_view, title_view
from uclone_x_video.schemas import (
    ColorGrade,
    ManifestOutline,
    QualityTier,
    StyleProfile,
    TimelineRegionSelection,
    TrackFile,
    TrackKind,
    frames_to_timecode,
)

if TYPE_CHECKING:
    from uclone_x_video.work import VideoWork

__all__ = [
    "TimelineContextBundle",
    "TimelineContextManifest",
    "TrackContextSummary",
    "build_timeline_context",
    "context_digest_line",
]


def context_digest_line(digest: str) -> str:
    """How the `[Video timeline context]` names its bundle's digest (#2429)."""
    return f"(context digest {digest})"


class TrackContextSummary(BaseModel):
    """Summary of one timeline track in the sequence."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    track_id: str
    kind: TrackKind
    name: str
    index: int
    clip_count: int
    title_count: int
    end_frame: int
    muted: bool
    solo: bool
    locked: bool


class TimelineContextManifest(BaseModel):
    """Explicit manifest of which tracks, assets, and proposals are in the context bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    included_tracks: list[str]
    omitted_tracks: list[str]
    included_assets: list[str]
    pending_proposal_ids: list[str]


class TimelineContextBundle(BaseModel):
    """The pull-based timeline context for a video project."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    project_id: str
    title: str
    style_profile: StyleProfile
    fps: int
    width: int
    height: int
    active_playback_tier: QualityTier
    total_frames: int
    total_timecode: str
    master_grade: ColorGrade
    selected_region: TimelineRegionSelection | None = None
    tracks: list[TrackContextSummary]
    media_pool: list[dict[str, Any]]
    window_clips: dict[str, list[dict[str, Any]]]
    window_titles: dict[str, list[dict[str, Any]]]
    manifest: TimelineContextManifest


def _track_end_frame(track: TrackFile) -> int:
    clip_end = max((c.timeline_out_frame for c in track.clips), default=0)
    title_end = max((t.timeline_out_frame for t in track.titles), default=0)
    return max(clip_end, title_end)


def build_timeline_context(
    work: VideoWork,
    *,
    selected_region: TimelineRegionSelection | None = None,
    max_tracks: int = 8,
    max_clips_per_track: int = 8,
) -> TimelineContextBundle:
    """Build the compact horizontal and vertical timeline context bundle for `work`."""
    record = work.record()
    manifest_pair = work.read_manifest()
    outline = manifest_pair[0] if manifest_pair is not None else ManifestOutline(fps=record.fps)

    all_tracks = [t for t, _ in work.list_tracks()]
    if selected_region is not None:
        all_tracks.sort(
            key=lambda t: (0 if t.track_id == selected_region.track_id else 1, t.index, t.track_id)
        )

    limit = max(1, max_tracks)
    included = all_tracks[:limit]
    omitted = all_tracks[limit:]

    track_summaries: list[TrackContextSummary] = []
    total_frames = 0
    window_clips: dict[str, list[dict[str, Any]]] = {}
    window_titles: dict[str, list[dict[str, Any]]] = {}

    for track in included:
        end_f = _track_end_frame(track)
        total_frames = max(total_frames, end_f)
        track_summaries.append(
            TrackContextSummary(
                track_id=track.track_id,
                kind=track.kind,
                name=track.name or track.track_id.upper(),
                index=track.index,
                clip_count=len(track.clips),
                title_count=len(track.titles),
                end_frame=end_f,
                muted=track.muted,
                solo=track.solo,
                locked=track.locked,
            )
        )
        if selected_region is not None:
            r_start = max(0, selected_region.start_frame - outline.fps)
            r_end = selected_region.end_frame + outline.fps
            matching_clips = [
                clip_view(c)
                for c in track.clips
                if c.timeline_out_frame > r_start and c.timeline_in_frame < r_end
            ]
            matching_titles = [
                title_view(t)
                for t in track.titles
                if t.timeline_out_frame > r_start and t.timeline_in_frame < r_end
            ]
        else:
            matching_clips = [clip_view(c) for c in track.clips[-max_clips_per_track:]]
            matching_titles = [title_view(t) for t in track.titles[-max_clips_per_track:]]
        if matching_clips:
            window_clips[track.track_id] = matching_clips[:max_clips_per_track]
        if matching_titles:
            window_titles[track.track_id] = matching_titles[:max_clips_per_track]

    assets = [asset for asset, _ in work.list_assets()]
    media_pool = [asset_line(a) for a in assets[:20]]
    pending_ids = [prop.id for prop, _ in work.list_proposals(status="pending")]

    ctx_manifest = TimelineContextManifest(
        included_tracks=[t.track_id for t in included],
        omitted_tracks=[t.track_id for t in omitted],
        included_assets=[a.asset_id for a in assets[:20]],
        pending_proposal_ids=pending_ids,
    )

    return TimelineContextBundle(
        project_id=record.id,
        title=record.title,
        style_profile=record.style_profile,
        fps=outline.fps,
        width=outline.width,
        height=outline.height,
        active_playback_tier=outline.active_playback_tier,
        total_frames=total_frames,
        total_timecode=frames_to_timecode(total_frames, outline.fps),
        master_grade=outline.master_grade,
        selected_region=selected_region,
        tracks=track_summaries,
        media_pool=media_pool,
        window_clips=window_clips,
        window_titles=window_titles,
        manifest=ctx_manifest,
    )
