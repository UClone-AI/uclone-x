"""Interactive video editing, media vault, semantic frame search, timeline, FX/color, proposal, and audit tools.

All decisions come from structured tool arguments (#2204); modifications to existing
non-empty tracks stage a `VideoProposal` requiring person approval (#2432); tool results
carry compact digests and slices (#2429).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x_video.audit import audit_project
from uclone_x_video.compiler import render_final_master
from uclone_x_video.context import build_timeline_context, context_digest_line
from uclone_x_video.digests import (
    LIST_ITEMS_SHOWN,
    asset_line,
    audit_digest,
    bundle_view,
    clip_view,
    digest_of_value,
    proposal_line,
    title_view,
    write_result,
)
from uclone_x_video.indexer import (
    FrameSeedSpec,
    index_asset_frames,
    search_project_frames,
)
from uclone_x_video.library import VideoError
from uclone_x_video.proposals import (
    apply_proposal,
    propose_manifest_change,
    propose_track_change,
    reject_proposal,
)
from uclone_x_video.schemas import (
    BakeOperation,
    BakeOpKind,
    ClipTransform,
    ClipTransition,
    ColorGrade,
    LutPreset,
    ManifestOutline,
    MediaAsset,
    QualityTier,
    SequenceMarker,
    SequenceSection,
    TimelineRegionSelection,
    TitleAnimation,
    TitleClip,
    TitlePosition,
    TitleStylePreset,
    TrackFile,
    TrackKind,
    TransitionKind,
    VideoClip,
)
from uclone_x_video.vault import (
    bake_derived_asset,
    ingest_raw_video,
    swap_clip_to_derived,
    unbake_clip_to_raw,
)
from uclone_x_video.work import VideoWork

if TYPE_CHECKING:
    from uclone_x.llm.protocols import EmbedderProtocol

__all__ = [
    "APPLY_PROPOSAL_NOT_APPROVED_NOTE",
    "VideoAuditParams",
    "VideoAuditTool",
    "VideoContextParams",
    "VideoContextTool",
    "VideoFxColorParams",
    "VideoFxColorTool",
    "VideoManifestParams",
    "VideoManifestTool",
    "VideoMediaVaultParams",
    "VideoMediaVaultTool",
    "VideoProposalParams",
    "VideoProposalTool",
    "VideoSearchParams",
    "VideoSearchTool",
    "VideoTimelineParams",
    "VideoTimelineTool",
    "append_or_insert_clip",
    "cut_track_range",
    "ripple_delete_clip",
    "split_track_clip",
    "trim_track_clip",
]

_ARGUMENTS = ConfigDict(extra="forbid")

APPLY_PROPOSAL_NOT_APPROVED_NOTE = (
    "The person has not approved applying this video proposal yet. Leave it pending so "
    "they can review the A/B preview and approve or reject it in VideoView."
)


def _save_or_propose_track(
    work: VideoWork,
    proposed_track: TrackFile,
    *,
    summary: str,
    direct_when_empty: bool = True,
) -> dict[str, Any]:
    """Write `proposed_track` directly only when creating a new/empty track; otherwise stage a proposal (#2432)."""
    existing = work.read_track_by_id(proposed_track.track_id)
    is_empty = existing is None or (not existing[0].clips and not existing[0].titles)
    if direct_when_empty and is_empty:
        prev_digest = existing[1] if existing is not None else None
        written = work.write_track(
            proposed_track,
            expected_digest=prev_digest,
            summary=summary,
        )
        return write_result(
            written.relative_path,
            prev_digest,
            written.digest,
            summary,
            proposed=False,
            track_id=proposed_track.track_id,
            clip_count=len(proposed_track.clips),
            title_count=len(proposed_track.titles),
        )

    proposal, written_prop = propose_track_change(
        work,
        proposed_track=proposed_track,
        summary=summary,
    )
    return {
        "proposed": True,
        "changed": True,
        "proposal_id": proposal.id,
        "track_id": proposed_track.track_id,
        "summary": f"Proposed {proposal.id}: {summary}. Waiting for approval in VideoView.",
        "path": written_prop.relative_path,
        "digest": written_prop.digest,
    }


# -- Pure Timeline Editing Operations ----------------------------------------------------


def _next_clip_id(track: TrackFile, asset_id: str) -> str:
    existing_ids = {c.clip_id for c in track.clips}
    idx = len(track.clips) + 1
    candidate = f"clip_{idx:02d}_{asset_id[:16]}"
    while candidate in existing_ids:
        idx += 1
        candidate = f"clip_{idx:02d}_{asset_id[:16]}"
    return candidate


def append_or_insert_clip(
    track: TrackFile,
    *,
    asset: MediaAsset,
    source_in_frame: int,
    source_out_frame: int,
    timeline_in_frame: int | None = None,
    clip_id: str | None = None,
    speed: float = 1.0,
    transition_kind: TransitionKind | None = None,
    transition_frames: int = 15,
    color_grade: ColorGrade | None = None,
) -> tuple[TrackFile, VideoClip]:
    """Append or insert a clip onto `track`, shifting subsequent clips forward to prevent overlap."""
    if source_in_frame < 0 or source_out_frame <= source_in_frame:
        raise VideoError(
            f"Clip source frame range {source_in_frame}..{source_out_frame} must have "
            "0 <= source_in_frame < source_out_frame."
        )
    if source_out_frame > asset.total_frames:
        raise VideoError(
            f"Clip source_out_frame ({source_out_frame}) exceeds asset {asset.asset_id!r} "
            f"total_frames ({asset.total_frames})."
        )

    cid = clip_id or _next_clip_id(track, asset.asset_id)
    transition = (
        ClipTransition(kind=transition_kind, duration_frames=transition_frames)
        if transition_kind is not None and transition_kind != "none"
        else None
    )
    end_of_track = max((c.timeline_out_frame for c in track.clips), default=0)
    target_timeline_in = end_of_track if timeline_in_frame is None else timeline_in_frame

    new_clip = VideoClip(
        clip_id=cid,
        asset_id=asset.asset_id,
        source_in_frame=source_in_frame,
        source_out_frame=source_out_frame,
        timeline_in_frame=target_timeline_in,
        speed=speed,
        color_grade=color_grade or ColorGrade(),
        transition_out=transition,
    )

    if not track.clips or target_timeline_in >= end_of_track:
        return track.model_copy(update={"clips": [*track.clips, new_clip]}), new_clip

    before: list[VideoClip] = []
    after: list[VideoClip] = []
    for existing in track.clips:
        if existing.timeline_out_frame <= target_timeline_in:
            before.append(existing)
        else:
            after.append(existing)

    if before and before[-1].timeline_out_frame > target_timeline_in:
        target_timeline_in = before[-1].timeline_out_frame
        new_clip = new_clip.model_copy(update={"timeline_in_frame": target_timeline_in})

    cursor = new_clip.timeline_out_frame
    shifted_after: list[VideoClip] = []
    for item in after:
        placed_in = max(cursor, item.timeline_in_frame + new_clip.duration_frames)
        shifted = item.model_copy(update={"timeline_in_frame": placed_in})
        shifted_after.append(shifted)
        cursor = shifted.timeline_out_frame

    return track.model_copy(update={"clips": [*before, new_clip, *shifted_after]}), new_clip


def split_track_clip(
    track: TrackFile,
    *,
    clip_id: str,
    split_timeline_frame: int,
) -> TrackFile:
    """Split `clip_id` at `split_timeline_frame` into two contiguous clips (`<id>_a`, `<id>_b`)."""
    target_idx = -1
    for idx, clip in enumerate(track.clips):
        if clip.clip_id == clip_id:
            target_idx = idx
            break
    if target_idx < 0:
        raise VideoError(f"Clip {clip_id!r} was not found on track {track.track_id!r}.")

    clip = track.clips[target_idx]
    if not (clip.timeline_in_frame < split_timeline_frame < clip.timeline_out_frame):
        raise VideoError(
            f"split_timeline_frame ({split_timeline_frame}) must be strictly inside clip "
            f"{clip_id!r} timeline bounds ({clip.timeline_in_frame}..{clip.timeline_out_frame})."
        )

    offset_timeline = split_timeline_frame - clip.timeline_in_frame
    source_split = clip.source_in_frame + max(1, round(offset_timeline * clip.speed))
    if source_split <= clip.source_in_frame or source_split >= clip.source_out_frame:
        raise VideoError(
            f"Cannot split clip {clip_id!r} at frame {split_timeline_frame}: sub-frame span."
        )

    left_clip = clip.model_copy(
        update={
            "clip_id": f"{clip.clip_id}_a",
            "source_out_frame": source_split,
            "transition_out": None,
        }
    )
    right_clip = clip.model_copy(
        update={
            "clip_id": f"{clip.clip_id}_b",
            "source_in_frame": source_split,
            "timeline_in_frame": left_clip.timeline_out_frame,
        }
    )
    new_clips = [
        *track.clips[:target_idx],
        left_clip,
        right_clip,
        *track.clips[target_idx + 1 :],
    ]
    return track.model_copy(update={"clips": new_clips})


def trim_track_clip(
    track: TrackFile,
    *,
    clip_id: str,
    new_source_in_frame: int | None = None,
    new_source_out_frame: int | None = None,
    max_asset_frames: int | None = None,
) -> TrackFile:
    """Trim `clip_id`'s `source_in_frame` and/or `source_out_frame` and ripple subsequent clips."""
    target_idx = -1
    for idx, clip in enumerate(track.clips):
        if clip.clip_id == clip_id:
            target_idx = idx
            break
    if target_idx < 0:
        raise VideoError(f"Clip {clip_id!r} was not found on track {track.track_id!r}.")

    clip = track.clips[target_idx]
    s_in = clip.source_in_frame if new_source_in_frame is None else new_source_in_frame
    s_out = clip.source_out_frame if new_source_out_frame is None else new_source_out_frame
    if s_in < 0 or s_out <= s_in:
        raise VideoError(
            f"Trimmed frame range {s_in}..{s_out} for clip {clip_id!r} must satisfy 0 <= in < out."
        )
    if max_asset_frames is not None and s_out > max_asset_frames:
        raise VideoError(
            f"Trimmed source_out_frame ({s_out}) exceeds asset total_frames ({max_asset_frames})."
        )

    old_dur = clip.duration_frames
    trimmed = clip.model_copy(update={"source_in_frame": s_in, "source_out_frame": s_out})
    delta = trimmed.duration_frames - old_dur
    updated: list[VideoClip] = [*track.clips[:target_idx], trimmed]
    for later in track.clips[target_idx + 1 :]:
        updated.append(
            later.model_copy(update={"timeline_in_frame": max(0, later.timeline_in_frame + delta)})
        )
    return track.model_copy(update={"clips": updated})


def ripple_delete_clip(track: TrackFile, *, clip_id: str) -> TrackFile:
    """Remove `clip_id` from `track` and shift subsequent clips left to close the gap."""
    target_idx = -1
    for idx, clip in enumerate(track.clips):
        if clip.clip_id == clip_id:
            target_idx = idx
            break
    if target_idx < 0:
        raise VideoError(f"Clip {clip_id!r} was not found on track {track.track_id!r}.")

    removed = track.clips[target_idx]
    shift = removed.duration_frames
    updated: list[VideoClip] = list(track.clips[:target_idx])
    for later in track.clips[target_idx + 1 :]:
        updated.append(
            later.model_copy(update={"timeline_in_frame": max(0, later.timeline_in_frame - shift)})
        )
    return track.model_copy(update={"clips": updated})


def cut_track_range(
    track: TrackFile,
    *,
    start_frame: int,
    end_frame: int,
) -> TrackFile:
    """Ripple-cut the timeline frame span `[start_frame, end_frame)` out of `track`."""
    if start_frame < 0 or end_frame <= start_frame:
        raise VideoError(
            f"Cut range {start_frame}..{end_frame} must have 0 <= start_frame < end_frame."
        )
    cut_span = end_frame - start_frame
    remaining: list[VideoClip] = []
    for clip in track.clips:
        if clip.timeline_out_frame <= start_frame:
            remaining.append(clip)
        elif clip.timeline_in_frame >= end_frame:
            remaining.append(
                clip.model_copy(
                    update={"timeline_in_frame": max(0, clip.timeline_in_frame - cut_span)}
                )
            )
        else:
            # Clip intersects `[start_frame, end_frame)`
            if clip.timeline_in_frame < start_frame:
                left_frames = start_frame - clip.timeline_in_frame
                left_source_out = min(
                    clip.source_out_frame,
                    clip.source_in_frame + max(1, round(left_frames * clip.speed)),
                )
                remaining.append(
                    clip.model_copy(
                        update={
                            "clip_id": f"{clip.clip_id}_pre",
                            "source_out_frame": left_source_out,
                            "transition_out": None,
                        }
                    )
                )
            if clip.timeline_out_frame > end_frame:
                cut_into_clip = end_frame - clip.timeline_in_frame
                right_source_in = min(
                    clip.source_out_frame - 1,
                    clip.source_in_frame + max(1, round(cut_into_clip * clip.speed)),
                )
                remaining.append(
                    clip.model_copy(
                        update={
                            "clip_id": f"{clip.clip_id}_post",
                            "source_in_frame": right_source_in,
                            "timeline_in_frame": start_frame,
                        }
                    )
                )
    return track.model_copy(update={"clips": remaining})


# -- 1. video_media (`VideoMediaVaultTool`) ----------------------------------------------


class VideoMediaVaultParams(BaseModel):
    """Parameters for `video_media`."""

    model_config = _ARGUMENTS

    action: Literal[
        "register_upload",
        "bake_derived_asset",
        "list_assets",
        "inspect_lineage",
        "index_frames",
    ] = Field(
        description="'register_upload' an immutable raw video into the Tier 2 vault and build its "
        "Tier 1 preview proxy; 'bake_derived_asset' to render a heavy edit into a new immutable "
        "derived asset while keeping the parent raw asset untouched; 'list_assets' in the media "
        "pool; 'inspect_lineage' of an asset; 'index_frames' for semantic frame search."
    )
    asset_id: str | None = Field(
        default=None,
        description="Asset identifier (e.g. 'asset_cafe_talk').",
    )
    filename: str | None = Field(
        default=None,
        description="Original filename for 'register_upload' (e.g. 'cafe_talk.mp4').",
    )
    fps: int = Field(default=30, ge=1, le=240, description="Frames per second.")
    total_frames: int = Field(
        default=300, ge=1, le=10_000_000, description="Total frames in the uploaded raw asset."
    )
    width: int = Field(default=1920, ge=64, le=7680, description="Source video width.")
    height: int = Field(default=1080, ge=64, le=4320, description="Source video height.")
    parent_asset_id: str | None = Field(
        default=None,
        description="Parent immutable asset id for 'bake_derived_asset'.",
    )
    source_in_frame: int | None = Field(
        default=None,
        ge=0,
        description="Start frame in the parent asset for 'bake_derived_asset'.",
    )
    source_out_frame: int | None = Field(
        default=None,
        ge=1,
        description="End frame in the parent asset for 'bake_derived_asset'.",
    )
    operations: list[BakeOperation] | None = Field(
        default=None,
        description="Heavy edit operations (e.g. [{'op': 'stabilize'}, {'op': 'optical_flow_slowmo', 'rate': 0.5}]) for 'bake_derived_asset'.",
    )
    op: BakeOpKind | None = Field(
        default=None,
        description="Shorthand single operation kind for 'bake_derived_asset' when 'operations' is omitted.",
    )
    rate: float = Field(
        default=0.5,
        gt=0.05,
        le=8.0,
        description="Playback rate for 'optical_flow_slowmo' or 'speed_ramp' shorthand.",
    )
    recipe_kind: Literal["heavy_transform", "compound_bake", "audio_stem_clean"] = Field(
        default="heavy_transform",
        description="Lineage recipe category for 'bake_derived_asset'.",
    )
    frame_seeds: list[FrameSeedSpec] | None = Field(
        default=None,
        description="Optional frame/shot seed specifications for 'register_upload' or 'index_frames'.",
    )


class VideoMediaVaultTool(BaseTool[VideoMediaVaultParams]):
    """Manage the 3-tier immutable media vault, preview proxies, and derived asset baking."""

    name = "video_media"
    description = (
        "Register immutable raw video uploads into the Tier 2 SHA-256 vault with automatic "
        "Tier 1 preview proxy generation, bake heavy edits into new immutable derived assets "
        "with full lineage while keeping source media untouched, inspect asset lineage, and "
        "index frames for semantic search."
    )
    params_type = VideoMediaVaultParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"list_assets", "inspect_lineage"})
    needs_room: ClassVar[bool] = True

    def __init__(self, embedder: EmbedderProtocol | None = None) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=VideoMediaVaultParams,
        )
        self._embedder = embedder

    async def run(self, params: VideoMediaVaultParams, context: ToolContext) -> dict[str, Any]:
        work = VideoWork.for_context(context)

        if params.action == "list_assets":
            assets = [asset_line(a) for a, _ in work.list_assets()]
            return {
                "assets": assets[:LIST_ITEMS_SHOWN],
                "total_assets": len(assets),
            }

        if params.action == "inspect_lineage":
            if not params.asset_id:
                raise VideoError("Specify 'asset_id' for action='inspect_lineage'.")
            loaded = work.read_asset(params.asset_id.strip())
            if loaded is None:
                raise VideoError(f"Media asset {params.asset_id!r} was not found.")
            asset, digest = loaded
            return {
                "asset": asset_line(asset),
                "immutable": asset.immutable,
                "tiers": asset.tiers.model_dump(mode="json"),
                "lineage": asset.lineage.model_dump(mode="json") if asset.lineage else None,
                "digest": digest,
            }

        if params.action == "register_upload":
            if not params.asset_id:
                raise VideoError("Specify 'asset_id' for action='register_upload'.")
            aid = params.asset_id.strip()
            fname = (params.filename or f"{aid}.mp4").strip()
            asset, written = ingest_raw_video(
                work,
                asset_id=aid,
                filename=fname,
                fps=params.fps,
                total_frames=params.total_frames,
                width=params.width,
                height=params.height,
            )
            indexed = await index_asset_frames(
                work,
                asset_id=aid,
                seeds=params.frame_seeds,
                embedder=self._embedder,
            )
            reloaded = work.read_asset(aid)
            final_asset = reloaded[0] if reloaded is not None else asset
            final_digest = reloaded[1] if reloaded is not None else written.digest
            return write_result(
                written.relative_path,
                None,
                final_digest,
                f"Registered immutable raw asset {aid!r} (Tier 2 vault + Tier 1 preview proxy, {len(indexed)} shots indexed).",
                asset=asset_line(final_asset),
            )

        if params.action == "index_frames":
            if not params.asset_id:
                raise VideoError("Specify 'asset_id' for action='index_frames'.")
            aid = params.asset_id.strip()
            indexed = await index_asset_frames(
                work,
                asset_id=aid,
                seeds=params.frame_seeds,
                embedder=self._embedder,
            )
            return {
                "changed": True,
                "asset_id": aid,
                "indexed_frame_count": len(indexed),
                "summary": f"Indexed {len(indexed)} frame/shot entries for {aid!r}.",
            }

        # bake_derived_asset
        parent_id = (params.parent_asset_id or "").strip()
        if not parent_id:
            raise VideoError("Specify 'parent_asset_id' for action='bake_derived_asset'.")
        parent_loaded = work.read_asset(parent_id)
        if parent_loaded is None:
            raise VideoError(f"Parent media asset {parent_id!r} was not found.")
        parent_asset, _ = parent_loaded

        in_f = params.source_in_frame if params.source_in_frame is not None else 0
        out_f = (
            params.source_out_frame
            if params.source_out_frame is not None
            else parent_asset.total_frames
        )
        ops: Sequence[BakeOperation]
        if params.operations:
            ops = params.operations
        elif params.op is not None:
            ops = [BakeOperation(op=params.op, rate=params.rate)]
        else:
            ops = [
                BakeOperation(op="stabilize", strength=0.75),
                BakeOperation(op="optical_flow_slowmo", rate=params.rate),
            ]

        derived_asset, written = bake_derived_asset(
            work,
            parent_asset_id=parent_id,
            source_in_frame=in_f,
            source_out_frame=out_f,
            operations=ops,
            derived_asset_id=params.asset_id.strip() if params.asset_id else None,
            recipe_kind=params.recipe_kind,
        )
        await index_asset_frames(
            work,
            asset_id=derived_asset.asset_id,
            embedder=self._embedder,
        )
        return write_result(
            written.relative_path,
            None,
            written.digest,
            f"Baked derived immutable asset {derived_asset.asset_id!r} from {parent_id!r} "
            f"[{in_f}..{out_f}] ({', '.join(o.op for o in ops)}); source asset remains untouched.",
            asset=asset_line(derived_asset),
        )


# -- 2. video_search (`VideoSearchTool`) -------------------------------------------------


class VideoSearchParams(BaseModel):
    """Parameters for `video_search`."""

    model_config = _ARGUMENTS

    action: Literal["search_frames", "list_shots"] = Field(
        default="search_frames",
        description="'search_frames' by natural language query across indexed frames/shots; "
        "'list_shots' to inspect all indexed shots for an asset or project.",
    )
    query: str | None = Field(
        default=None,
        description="Semantic or visual search query (for example 'barista smiling with coffee cup').",
    )
    asset_id: str | None = Field(
        default=None,
        description="Optional filter restricting search to one media asset.",
    )
    shot_type: (
        Literal["wide", "medium", "close_up", "extreme_close_up", "aerial", "insert"] | None
    ) = Field(default=None, description="Optional shot framing filter.")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of ranked hits.")


class VideoSearchTool(BaseTool[VideoSearchParams]):
    """Search video frames and shots semantically or inspect shot boundaries."""

    name = "video_search"
    description = (
        "Search indexed video frames and shots by natural language description, visual tags, "
        "or transcript text, returning frame-accurate (in_frame, out_frame, timecode) ranges "
        "and explicit ranking method attribution."
    )
    params_type = VideoSearchParams
    writes_files: ClassVar[bool] = False
    needs_room: ClassVar[bool] = True

    def __init__(self, embedder: EmbedderProtocol | None = None) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=VideoSearchParams,
        )
        self._embedder = embedder

    async def run(self, params: VideoSearchParams, context: ToolContext) -> dict[str, Any]:
        work = VideoWork.for_context(context)
        if params.action == "list_shots":
            entries = work.read_frame_index()
            if params.asset_id:
                entries = [e for e in entries if e.asset_id == params.asset_id.strip()]
            shots = [
                {
                    "frame_id": e.frame_id,
                    "asset_id": e.asset_id,
                    "shot_id": e.shot_id,
                    "in_frame": e.shot_in_frame,
                    "out_frame": e.shot_out_frame,
                    "peak_frame": e.frame_no,
                    "timecode": e.timecode,
                    "shot_type": e.shot_type,
                    "caption": e.caption,
                    "visual_tags": e.visual_tags,
                }
                for e in entries[:LIST_ITEMS_SHOWN]
            ]
            return {
                "total_indexed_frames": len(entries),
                "shots": shots,
            }

        if not params.query or not params.query.strip():
            raise VideoError("Specify a non-empty 'query' for action='search_frames'.")
        ranking = await search_project_frames(
            work,
            query=params.query,
            asset_id=params.asset_id.strip() if params.asset_id else None,
            shot_type=params.shot_type,
            top_k=params.top_k,
            embedder=self._embedder,
        )
        return ranking.model_dump(mode="json")


# -- 3. video_manifest (`VideoManifestTool`) ---------------------------------------------


class VideoManifestParams(BaseModel):
    """Parameters for `video_manifest`."""

    model_config = _ARGUMENTS

    action: Literal[
        "read",
        "set_sequence",
        "set_playback_tier",
        "set_master_grade",
        "add_marker",
        "render_master",
    ] = Field(
        description="'read' manifest.yaml; 'set_sequence' resolution/fps/sections; "
        "'set_playback_tier' ('preview' or 'source'); 'set_master_grade' on the sequence; "
        "'add_marker' at a frame; 'render_master' to relink all clips to Tier 2 source vault "
        "files and render the Tier 3 final master MP4."
    )
    fps: int | None = Field(default=None, ge=1, le=240)
    width: int | None = Field(default=None, ge=160, le=7680)
    height: int | None = Field(default=None, ge=90, le=4320)
    tier: QualityTier | None = Field(
        default=None,
        description="Playback quality tier ('preview', 'source', or 'master') for 'set_playback_tier'.",
    )
    master_grade: ColorGrade | None = Field(
        default=None,
        description="Master sequence color grade for 'set_master_grade'.",
    )
    sections: list[SequenceSection] | None = Field(
        default=None,
        description="Sequence sections for 'set_sequence'.",
    )
    marker: SequenceMarker | None = Field(
        default=None,
        description="Timeline marker to add or update for 'add_marker'.",
    )


class VideoManifestTool(BaseTool[VideoManifestParams]):
    """Read or update `manifest.yaml`, switch quality tiers, or render the Tier 3 final master."""

    name = "video_manifest"
    description = (
        "Read or update the video sequence manifest (resolution, frame rate, active playback "
        "tier 'preview' vs. 'source', master color grade, timeline markers) or render the "
        "Tier 3 final master MP4 relinked to full-resolution source assets."
    )
    params_type = VideoManifestParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"read"})
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=VideoManifestParams,
        )

    async def run(self, params: VideoManifestParams, context: ToolContext) -> dict[str, Any]:
        work = VideoWork.for_context(context)
        record = work.record()
        loaded = work.read_manifest()
        current = loaded[0] if loaded is not None else ManifestOutline(fps=record.fps)
        prev_digest = loaded[1] if loaded is not None else None

        if params.action == "read":
            return {
                "manifest": current.model_dump(mode="json"),
                "quality_tiers": record.quality_tiers.model_dump(mode="json"),
                "digest": prev_digest,
            }

        if params.action == "render_master":
            rendered = render_final_master(work)
            return {
                "changed": True,
                "master_render": rendered.model_dump(mode="json"),
                "summary": (
                    f"Rendered Tier 3 final master ({rendered.width}x{rendered.height}, "
                    f"{rendered.total_frames} frames) to {rendered.output_path} after relinking "
                    f"{len(rendered.relinked_source_assets)} source stream(s)."
                ),
            }

        if params.action == "set_playback_tier":
            if params.tier is None:
                raise VideoError(
                    "Specify 'tier' ('preview', 'source', or 'master') for 'set_playback_tier'."
                )
            updated_manifest = current.model_copy(update={"active_playback_tier": params.tier})
            written = work.write_manifest(updated_manifest, expected_digest=prev_digest)
            work.update_project_settings(active_playback_tier=params.tier)
            return write_result(
                written.relative_path,
                prev_digest,
                written.digest,
                f"Switched active playback tier to {params.tier!r}.",
                active_playback_tier=params.tier,
            )

        if params.action == "set_sequence":
            updates: dict[str, Any] = {}
            if params.fps is not None:
                updates["fps"] = params.fps
                work.update_project_settings(fps=params.fps)
            if params.width is not None:
                updates["width"] = params.width
            if params.height is not None:
                updates["height"] = params.height
            if params.sections is not None:
                updates["sections"] = list(params.sections)
            updated_manifest = current.model_copy(update=updates)
            written = work.write_manifest(updated_manifest, expected_digest=prev_digest)
            return write_result(
                written.relative_path,
                prev_digest,
                written.digest,
                f"Updated sequence manifest ({updated_manifest.width}x{updated_manifest.height} @ {updated_manifest.fps}fps).",
            )

        if params.action == "set_master_grade":
            if params.master_grade is None:
                raise VideoError("Specify 'master_grade' for action='set_master_grade'.")
            updated_manifest = current.model_copy(update={"master_grade": params.master_grade})
            if prev_digest is not None and current.master_grade != FilterDefaultGrade:
                proposal, written_prop = propose_manifest_change(
                    work,
                    proposed_manifest=updated_manifest,
                    summary=f"Update master grade to LUT {params.master_grade.lut}",
                )
                return {
                    "proposed": True,
                    "changed": True,
                    "proposal_id": proposal.id,
                    "summary": f"Proposed {proposal.id}: Update master grade ({params.master_grade.lut}). Waiting for approval in VideoView.",
                    "path": written_prop.relative_path,
                    "digest": written_prop.digest,
                }
            written = work.write_manifest(updated_manifest, expected_digest=prev_digest)
            return write_result(
                written.relative_path,
                prev_digest,
                written.digest,
                f"Set sequence master color grade (LUT: {params.master_grade.lut}).",
            )

        # add_marker
        if params.marker is None:
            raise VideoError("Specify 'marker' for action='add_marker'.")
        others = [m for m in current.markers if m.marker_id != params.marker.marker_id]
        ordered = sorted([*others, params.marker], key=lambda m: m.frame)
        updated_manifest = current.model_copy(update={"markers": ordered})
        written = work.write_manifest(updated_manifest, expected_digest=prev_digest)
        return write_result(
            written.relative_path,
            prev_digest,
            written.digest,
            f"Added timeline marker {params.marker.marker_id!r} at frame {params.marker.frame}.",
        )


FilterDefaultGrade = ColorGrade()


# -- 4. video_timeline (`VideoTimelineTool`) ---------------------------------------------


class VideoTimelineParams(BaseModel):
    """Parameters for `video_timeline`."""

    model_config = _ARGUMENTS

    action: Literal[
        "read",
        "append_clip",
        "insert_clip",
        "split_clip",
        "trim_clip",
        "ripple_delete",
        "cut_range",
        "swap_to_derived",
        "unbake_to_raw",
        "set_transition",
        "adjust_track",
    ] = Field(
        description="Non-destructive manifest timeline operations: 'read' a track; "
        "'append_clip' or 'insert_clip' from an asset; 'split_clip' at a frame; 'trim_clip' "
        "in/out frames; 'ripple_delete' a clip; 'cut_range' a frame span; 'swap_to_derived' "
        "to replace a clip with a baked derived asset; 'unbake_to_raw' to restore a baked "
        "clip back to its raw source; 'set_transition' on a clip; 'adjust_track' mute/solo/lock."
    )
    track_id: str = Field(
        default="v1_main",
        description="Target track id (e.g. 'v1_main', 'v2_overlay', 'a1_dialogue').",
    )
    track_kind: TrackKind = Field(
        default="video", description="Track kind when creating a new track."
    )
    asset_id: str | None = Field(
        default=None,
        description="Source or derived media asset id for 'append_clip' or 'insert_clip'.",
    )
    derived_asset_id: str | None = Field(
        default=None,
        description="Derived asset id for 'swap_to_derived'.",
    )
    clip_id: str | None = Field(
        default=None,
        description="Target clip id on the track.",
    )
    source_in_frame: int | None = Field(
        default=None,
        ge=0,
        description="Clip start frame in the source asset.",
    )
    source_out_frame: int | None = Field(
        default=None,
        ge=1,
        description="Clip end frame (exclusive) in the source asset.",
    )
    timeline_in_frame: int | None = Field(
        default=None,
        ge=0,
        description="Placement frame on the sequence timeline for 'insert_clip'.",
    )
    split_frame: int | None = Field(
        default=None,
        ge=1,
        description="Timeline frame at which to split 'clip_id' for 'split_clip'.",
    )
    start_frame: int | None = Field(
        default=None,
        ge=0,
        description="Start timeline frame for 'cut_range'.",
    )
    end_frame: int | None = Field(
        default=None,
        ge=1,
        description="End timeline frame for 'cut_range'.",
    )
    speed: float = Field(default=1.0, gt=0.1, le=8.0, description="Clip playback speed multiplier.")
    transition_kind: TransitionKind | None = Field(
        default=None,
        description="Outgoing transition kind ('cross_dissolve', 'dip_to_black', 'whip_pan', 'wipe_left', or 'none').",
    )
    transition_frames: int = Field(
        default=15, ge=1, le=120, description="Transition duration in frames."
    )
    muted: bool | None = Field(default=None, description="Mute state for 'adjust_track'.")
    solo: bool | None = Field(default=None, description="Solo state for 'adjust_track'.")
    locked: bool | None = Field(default=None, description="Lock state for 'adjust_track'.")
    summary: str | None = Field(default=None, description="Optional human-readable edit summary.")


class VideoTimelineTool(BaseTool[VideoTimelineParams]):
    """Cut, insert, append, split, trim, ripple-delete, or swap derived clips on timeline tracks."""

    name = "video_timeline"
    description = (
        "Edit multi-track video timeline manifests non-destructively: append or insert clips "
        "by frame range, split clips, trim in/out points, ripple-delete clips, cut frame "
        "ranges, swap clips to baked derived assets or unbake back to raw, and set transitions."
    )
    params_type = VideoTimelineParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"read"})
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=VideoTimelineParams,
        )

    async def run(self, params: VideoTimelineParams, context: ToolContext) -> dict[str, Any]:
        work = VideoWork.for_context(context)
        tid = params.track_id.strip()
        loaded = work.read_track_by_id(tid)
        track = (
            loaded[0]
            if loaded is not None
            else TrackFile(track_id=tid, kind=params.track_kind, name=tid.upper(), index=1)
        )
        prev_digest = loaded[1] if loaded is not None else None

        if params.action == "read":
            if loaded is None:
                raise VideoError(f"Track {tid!r} does not exist yet.")
            return {
                "track_id": track.track_id,
                "kind": track.kind,
                "muted": track.muted,
                "solo": track.solo,
                "locked": track.locked,
                "clips": [clip_view(c) for c in track.clips[:LIST_ITEMS_SHOWN]],
                "titles": [title_view(t) for t in track.titles[:LIST_ITEMS_SHOWN]],
                "digest": prev_digest,
            }

        if params.action == "adjust_track":
            updates: dict[str, Any] = {}
            if params.muted is not None:
                updates["muted"] = params.muted
            if params.solo is not None:
                updates["solo"] = params.solo
            if params.locked is not None:
                updates["locked"] = params.locked
            updated_track = track.model_copy(update=updates)
            written = work.write_track(
                updated_track,
                expected_digest=prev_digest,
                summary=f"Adjusted track {tid} controls",
            )
            return write_result(
                written.relative_path,
                prev_digest,
                written.digest,
                f"Adjusted track {tid!r} controls (muted={updated_track.muted}, solo={updated_track.solo}, locked={updated_track.locked}).",
                track_id=tid,
                muted=updated_track.muted,
                solo=updated_track.solo,
                locked=updated_track.locked,
            )

        if track.locked:
            raise VideoError(f"Track {tid!r} is locked; unlock it before editing its clips.")

        if params.action in ("append_clip", "insert_clip"):
            if not params.asset_id:
                raise VideoError(f"Specify 'asset_id' for action={params.action!r}.")
            asset_loaded = work.read_asset(params.asset_id.strip())
            if asset_loaded is None:
                raise VideoError(f"Media asset {params.asset_id!r} was not found.")
            asset, _ = asset_loaded
            s_in = params.source_in_frame if params.source_in_frame is not None else 0
            s_out = (
                params.source_out_frame
                if params.source_out_frame is not None
                else asset.total_frames
            )
            t_in = params.timeline_in_frame if params.action == "insert_clip" else None
            updated_track, new_clip = append_or_insert_clip(
                track,
                asset=asset,
                source_in_frame=s_in,
                source_out_frame=s_out,
                timeline_in_frame=t_in,
                clip_id=params.clip_id.strip() if params.clip_id else None,
                speed=params.speed,
                transition_kind=params.transition_kind,
                transition_frames=params.transition_frames,
            )
            summary = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Place {new_clip.clip_id} ({asset.asset_id} [{s_in}..{s_out}]) on {tid}"
            )
            res = _save_or_propose_track(work, updated_track, summary=summary)
            res["clip"] = clip_view(new_clip)
            return res

        if params.action == "split_clip":
            if not params.clip_id or params.split_frame is None:
                raise VideoError("Specify 'clip_id' and 'split_frame' for action='split_clip'.")
            updated_track = split_track_clip(
                track,
                clip_id=params.clip_id.strip(),
                split_timeline_frame=params.split_frame,
            )
            summary = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Split clip {params.clip_id} at timeline frame {params.split_frame} on {tid}"
            )
            return _save_or_propose_track(work, updated_track, summary=summary)

        if params.action == "trim_clip":
            if not params.clip_id:
                raise VideoError("Specify 'clip_id' for action='trim_clip'.")
            target_clip = next(
                (c for c in track.clips if c.clip_id == params.clip_id.strip()), None
            )
            max_frames: int | None = None
            if target_clip is not None:
                asset_loaded = work.read_asset(target_clip.asset_id)
                if asset_loaded is not None:
                    max_frames = asset_loaded[0].total_frames
            updated_track = trim_track_clip(
                track,
                clip_id=params.clip_id.strip(),
                new_source_in_frame=params.source_in_frame,
                new_source_out_frame=params.source_out_frame,
                max_asset_frames=max_frames,
            )
            summary = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Trim clip {params.clip_id} on {tid}"
            )
            return _save_or_propose_track(work, updated_track, summary=summary)

        if params.action == "ripple_delete":
            if not params.clip_id:
                raise VideoError("Specify 'clip_id' for action='ripple_delete'.")
            updated_track = ripple_delete_clip(track, clip_id=params.clip_id.strip())
            summary = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Ripple delete clip {params.clip_id} from {tid}"
            )
            return _save_or_propose_track(work, updated_track, summary=summary)

        if params.action == "cut_range":
            if params.start_frame is None or params.end_frame is None:
                raise VideoError("Specify 'start_frame' and 'end_frame' for action='cut_range'.")
            updated_track = cut_track_range(
                track, start_frame=params.start_frame, end_frame=params.end_frame
            )
            summary = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Cut timeline frames {params.start_frame}..{params.end_frame} on {tid}"
            )
            return _save_or_propose_track(work, updated_track, summary=summary)

        if params.action == "swap_to_derived":
            if not params.clip_id or not params.derived_asset_id:
                raise VideoError(
                    "Specify 'clip_id' and 'derived_asset_id' for action='swap_to_derived'."
                )
            derived_loaded = work.read_asset(params.derived_asset_id.strip())
            if derived_loaded is None:
                raise VideoError(f"Derived asset {params.derived_asset_id!r} was not found.")
            updated_track = swap_clip_to_derived(
                track,
                clip_id=params.clip_id.strip(),
                derived_asset=derived_loaded[0],
            )
            summary = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Swap clip {params.clip_id} on {tid} to derived asset {params.derived_asset_id}"
            )
            return _save_or_propose_track(work, updated_track, summary=summary)

        if params.action == "unbake_to_raw":
            if not params.clip_id:
                raise VideoError("Specify 'clip_id' for action='unbake_to_raw'.")
            assets_by_id = {a.asset_id: a for a, _ in work.list_assets()}
            updated_track = unbake_clip_to_raw(
                track,
                clip_id=params.clip_id.strip(),
                assets_by_id=assets_by_id,
            )
            summary = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Restore baked clip {params.clip_id} on {tid} back to raw source"
            )
            return _save_or_propose_track(work, updated_track, summary=summary)

        # set_transition
        if not params.clip_id or params.transition_kind is None:
            raise VideoError("Specify 'clip_id' and 'transition_kind' for action='set_transition'.")
        cid = params.clip_id.strip()
        found = False
        new_clips: list[VideoClip] = []
        for clip in track.clips:
            if clip.clip_id == cid:
                found = True
                trans = (
                    None
                    if params.transition_kind == "none"
                    else ClipTransition(
                        kind=params.transition_kind,
                        duration_frames=params.transition_frames,
                    )
                )
                new_clips.append(clip.model_copy(update={"transition_out": trans}))
            else:
                new_clips.append(clip)
        if not found:
            raise VideoError(f"Clip {cid!r} was not found on track {tid!r}.")
        updated_track = track.model_copy(update={"clips": new_clips})
        summary = (
            params.summary.strip()
            if params.summary and params.summary.strip()
            else f"Set transition {params.transition_kind} on {cid} ({tid})"
        )
        return _save_or_propose_track(work, updated_track, summary=summary)


# -- 5. video_fx (`VideoFxColorTool`) ----------------------------------------------------


class VideoFxColorParams(BaseModel):
    """Parameters for `video_fx`."""

    model_config = _ARGUMENTS

    action: Literal[
        "grade_clip",
        "add_title",
        "update_title",
        "remove_title",
        "set_speed",
        "set_transform",
    ] = Field(
        description="DaVinci-style color grading, title overlays, speed, and spatial transforms: "
        "'grade_clip' 3-way color wheels & LUT; 'add_title', 'update_title', or 'remove_title' "
        "on a title track; 'set_speed' or 'set_transform' on a video clip."
    )
    track_id: str = Field(
        default="v1_main",
        description="Target track id ('v1_main' for clips, 't1_titles' for titles).",
    )
    clip_id: str | None = Field(default=None, description="Target clip id for clip FX/grading.")
    lut: LutPreset | None = Field(default=None, description="Color LUT preset.")
    lift: tuple[float, float, float] | None = Field(
        default=None, description="Shadows RGB lift offset (e.g. [0.0, 0.0, 0.02])."
    )
    gamma: tuple[float, float, float] | None = Field(
        default=None, description="Midtones RGB gamma offset."
    )
    gain: tuple[float, float, float] | None = Field(
        default=None, description="Highlights RGB gain multiplier (e.g. [1.05, 1.02, 0.98])."
    )
    contrast: float | None = Field(default=None, ge=0.2, le=3.0)
    saturation: float | None = Field(default=None, ge=0.0, le=3.0)
    temperature: float | None = Field(default=None, ge=-100.0, le=100.0)
    tint: float | None = Field(default=None, ge=-100.0, le=100.0)
    vignette: float | None = Field(default=None, ge=0.0, le=1.0)
    title_id: str | None = Field(default=None, description="Title clip id (e.g. 'title_01').")
    timeline_in_frame: int | None = Field(default=None, ge=0, description="Title start frame.")
    timeline_out_frame: int | None = Field(default=None, ge=1, description="Title end frame.")
    text: str | None = Field(default=None, description="Primary title text.")
    subtitle: str | None = Field(default=None, description="Optional subtitle / lower-third line.")
    style_preset: TitleStylePreset = Field(default="minimal_cinema")
    position: TitlePosition = Field(default="center")
    animation_in: TitleAnimation = Field(default="fade_up")
    color_hex: str = Field(default="#F8FAFC")
    font_size_px: int = Field(default=48, ge=12, le=200)
    speed: float | None = Field(default=None, gt=0.1, le=8.0)
    scale: float | None = Field(default=None, ge=0.1, le=5.0)
    position_x: float | None = Field(default=None, ge=-1.0, le=1.0)
    position_y: float | None = Field(default=None, ge=-1.0, le=1.0)
    rotation_deg: float | None = Field(default=None, ge=-180.0, le=180.0)
    summary: str | None = Field(default=None, description="Optional human-readable summary.")


class VideoFxColorTool(BaseTool[VideoFxColorParams]):
    """Apply DaVinci-style 3-way color grading, LUTs, title overlays, speed ramps, and transforms."""

    name = "video_fx"
    description = (
        "Apply 3-way color grading (Lift/Gamma/Gain, Contrast, Saturation, Temperature, Tint, "
        "LUT, Vignette), add or edit animated title/lower-third graphics, and adjust clip "
        "speed or spatial transforms."
    )
    params_type = VideoFxColorParams
    writes_files: ClassVar[bool] = True
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=VideoFxColorParams,
        )

    async def run(self, params: VideoFxColorParams, context: ToolContext) -> dict[str, Any]:
        work = VideoWork.for_context(context)

        if params.action in ("add_title", "update_title", "remove_title"):
            tid = params.track_id.strip() if params.track_id != "v1_main" else "t1_titles"
            loaded = work.read_track_by_id(tid)
            track = (
                loaded[0]
                if loaded is not None
                else TrackFile(track_id=tid, kind="title", name="TITLES", index=3)
            )
            if params.action == "add_title":
                if not params.text or not params.text.strip():
                    raise VideoError("Specify non-empty 'text' for action='add_title'.")
                t_in = params.timeline_in_frame if params.timeline_in_frame is not None else 0
                t_out = (
                    params.timeline_out_frame
                    if params.timeline_out_frame is not None
                    else t_in + 90
                )
                title_id = (
                    params.title_id.strip()
                    if params.title_id
                    else f"title_{len(track.titles) + 1:02d}"
                )
                new_title = TitleClip(
                    title_id=title_id,
                    timeline_in_frame=t_in,
                    timeline_out_frame=t_out,
                    text=params.text.strip(),
                    subtitle=(params.subtitle or "").strip(),
                    style_preset=params.style_preset,
                    font_size_px=params.font_size_px,
                    color_hex=params.color_hex,
                    position=params.position,
                    animation_in=params.animation_in,
                )
                ordered = sorted([*track.titles, new_title], key=lambda t: t.timeline_in_frame)
                updated_track = track.model_copy(update={"titles": ordered})
                summary = (
                    params.summary.strip()
                    if params.summary and params.summary.strip()
                    else f"Add title {new_title.text!r} ({t_in}..{t_out}) on {tid}"
                )
                return _save_or_propose_track(work, updated_track, summary=summary)

            if not params.title_id:
                raise VideoError(f"Specify 'title_id' for action={params.action!r}.")
            target_tid = params.title_id.strip()
            if params.action == "remove_title":
                remaining = [t for t in track.titles if t.title_id != target_tid]
                if len(remaining) == len(track.titles):
                    raise VideoError(f"Title {target_tid!r} was not found on track {tid!r}.")
                updated_track = track.model_copy(update={"titles": remaining})
                return _save_or_propose_track(
                    work,
                    updated_track,
                    summary=f"Remove title {target_tid} from {tid}",
                )

            # update_title
            updated_titles: list[TitleClip] = []
            found_title = False
            for t_clip in track.titles:
                if t_clip.title_id == target_tid:
                    found_title = True
                    t_updates: dict[str, Any] = {}
                    if params.text is not None:
                        t_updates["text"] = params.text.strip()
                    if params.subtitle is not None:
                        t_updates["subtitle"] = params.subtitle.strip()
                    if params.timeline_in_frame is not None:
                        t_updates["timeline_in_frame"] = params.timeline_in_frame
                    if params.timeline_out_frame is not None:
                        t_updates["timeline_out_frame"] = params.timeline_out_frame
                    t_updates["style_preset"] = params.style_preset
                    t_updates["position"] = params.position
                    t_updates["animation_in"] = params.animation_in
                    t_updates["color_hex"] = params.color_hex
                    t_updates["font_size_px"] = params.font_size_px
                    updated_titles.append(t_clip.model_copy(update=t_updates))
                else:
                    updated_titles.append(t_clip)
            if not found_title:
                raise VideoError(f"Title {target_tid!r} was not found on track {tid!r}.")
            updated_track = track.model_copy(
                update={"titles": sorted(updated_titles, key=lambda t: t.timeline_in_frame)}
            )
            return _save_or_propose_track(
                work,
                updated_track,
                summary=f"Update title {target_tid} on {tid}",
            )

        # Clip-level FX (`grade_clip`, `set_speed`, `set_transform`)
        tid = params.track_id.strip()
        loaded = work.read_track_by_id(tid)
        if loaded is None:
            raise VideoError(f"Track {tid!r} was not found.")
        track, _ = loaded
        if not params.clip_id:
            raise VideoError(f"Specify 'clip_id' for action={params.action!r}.")
        cid = params.clip_id.strip()

        target_idx = next((idx for idx, c in enumerate(track.clips) if c.clip_id == cid), -1)
        if target_idx < 0:
            raise VideoError(f"Clip {cid!r} was not found on track {tid!r}.")
        clip = track.clips[target_idx]

        if params.action == "grade_clip":
            g = clip.color_grade
            new_grade = ColorGrade(
                lut=params.lut if params.lut is not None else g.lut,
                lift=params.lift if params.lift is not None else g.lift,
                gamma=params.gamma if params.gamma is not None else g.gamma,
                gain=params.gain if params.gain is not None else g.gain,
                contrast=params.contrast if params.contrast is not None else g.contrast,
                saturation=params.saturation if params.saturation is not None else g.saturation,
                temperature=params.temperature if params.temperature is not None else g.temperature,
                tint=params.tint if params.tint is not None else g.tint,
                vignette=params.vignette if params.vignette is not None else g.vignette,
            )
            updated_clip = clip.model_copy(update={"color_grade": new_grade})
            new_clips = [
                *track.clips[:target_idx],
                updated_clip,
                *track.clips[target_idx + 1 :],
            ]
            updated_track = track.model_copy(update={"clips": new_clips})
            summary = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Color grade {cid} on {tid} (LUT: {new_grade.lut}, temp: {new_grade.temperature:+.0f})"
            )
            return _save_or_propose_track(
                work, updated_track, summary=summary, direct_when_empty=False
            )

        if params.action == "set_speed":
            if params.speed is None:
                raise VideoError("Specify 'speed' for action='set_speed'.")
            old_dur = clip.duration_frames
            updated_clip = clip.model_copy(update={"speed": params.speed})
            delta = updated_clip.duration_frames - old_dur
            new_clips = [*track.clips[:target_idx], updated_clip]
            for later in track.clips[target_idx + 1 :]:
                new_clips.append(
                    later.model_copy(
                        update={"timeline_in_frame": max(0, later.timeline_in_frame + delta)}
                    )
                )
            updated_track = track.model_copy(update={"clips": new_clips})
            summary = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Set speed={params.speed}x on {cid} ({tid})"
            )
            return _save_or_propose_track(
                work, updated_track, summary=summary, direct_when_empty=False
            )

        # set_transform
        tr = clip.transform
        new_transform = ClipTransform(
            scale=params.scale if params.scale is not None else tr.scale,
            position_x=params.position_x if params.position_x is not None else tr.position_x,
            position_y=params.position_y if params.position_y is not None else tr.position_y,
            rotation_deg=params.rotation_deg
            if params.rotation_deg is not None
            else tr.rotation_deg,
        )
        updated_clip = clip.model_copy(update={"transform": new_transform})
        new_clips = [
            *track.clips[:target_idx],
            updated_clip,
            *track.clips[target_idx + 1 :],
        ]
        updated_track = track.model_copy(update={"clips": new_clips})
        summary = (
            params.summary.strip()
            if params.summary and params.summary.strip()
            else f"Transform {cid} on {tid} (scale={new_transform.scale})"
        )
        return _save_or_propose_track(work, updated_track, summary=summary, direct_when_empty=False)


# -- 6. video_context (`VideoContextTool`) -----------------------------------------------


class VideoContextParams(BaseModel):
    """Parameters for `video_context`."""

    model_config = _ARGUMENTS

    selected_region: TimelineRegionSelection | None = Field(
        default=None,
        description="Optional selected track and frame span from VideoView.",
    )
    max_tracks: int = Field(default=8, ge=1, le=32)


class VideoContextTool(BaseTool[VideoContextParams]):
    """Pull the active video sequence's timeline context, media pool, and context manifest."""

    name = "video_context"
    description = (
        "Pull the active video sequence's timeline context bundle: quality tier, track stack, "
        "media pool with raw and derived asset lineage, window clips/titles, and context manifest."
    )
    params_type = VideoContextParams
    writes_files: ClassVar[bool] = False
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=VideoContextParams,
        )

    async def run(self, params: VideoContextParams, context: ToolContext) -> dict[str, Any]:
        work = VideoWork.for_context(context)
        bundle = build_timeline_context(
            work,
            selected_region=params.selected_region,
            max_tracks=params.max_tracks,
        )
        payload = bundle_view(bundle)
        digest = digest_of_value(payload)
        mark = context_digest_line(digest)
        if any(mark in section for section in context.turn_aid_sections):
            return {
                "same_as_turn_context": True,
                "digest": digest,
                "summary": "This is the [Video timeline context] already in this request "
                f"(context digest {digest[:12]}); it is not repeated here.",
            }
        return {**payload, "digest": digest}


# -- 7. video_proposal (`VideoProposalTool`) ---------------------------------------------


class VideoProposalParams(BaseModel):
    """Parameters for `video_proposal`."""

    model_config = _ARGUMENTS

    action: Literal["list", "propose", "apply", "reject"] = Field(
        description="'list' proposals; 'propose' a track change; 'apply' a pending proposal "
        "(requires person approval); 'reject' a pending proposal."
    )
    proposal_id: str | None = Field(
        default=None, description="Proposal id (e.g. 'p001') for 'apply' or 'reject'."
    )
    track: TrackFile | None = Field(
        default=None, description="Proposed TrackFile for action='propose'."
    )
    summary: str | None = Field(default=None, description="Summary for action='propose'.")
    status: Literal["pending", "approved", "rejected"] | None = Field(default=None)
    reason: str | None = Field(default=None, description="Optional rejection reason.")
    seen_digest: str | None = Field(default=None)


class VideoProposalTool(BaseTool[VideoProposalParams]):
    """List, propose, apply, or reject staged video timeline proposals (`proposals/pNNN.yaml`)."""

    name = "video_proposal"
    description = (
        "List, propose, apply, or reject staged video editing proposals. Applying a proposal "
        "requires person approval in the conversation or in VideoView."
    )
    params_type = VideoProposalParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"list"})
    needs_room: ClassVar[bool] = True
    approval_actions: ClassVar[frozenset[str]] = frozenset({"apply"})
    approval_timeout_note: ClassVar[str | None] = APPLY_PROPOSAL_NOT_APPROVED_NOTE
    approval_unavailable_note: ClassVar[str | None] = APPLY_PROPOSAL_NOT_APPROVED_NOTE
    approval_unavailable_ends_turn: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=VideoProposalParams,
        )

    async def run(self, params: VideoProposalParams, context: ToolContext) -> dict[str, Any]:
        work = VideoWork.for_context(context)
        if params.action == "list":
            items = work.list_proposals(status=params.status or "pending")
            listed: dict[str, Any] = {
                "status": params.status or "pending",
                "proposals": [proposal_line(p, d) for p, d in items[-LIST_ITEMS_SHOWN:]],
            }
            if len(items) > LIST_ITEMS_SHOWN:
                listed["older_not_shown"] = len(items) - LIST_ITEMS_SHOWN
            return listed

        if params.action == "propose":
            if params.track is None:
                raise VideoError("Specify 'track' for action='propose'.")
            text = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Update track {params.track.track_id}"
            )
            proposal, written = propose_track_change(
                work, proposed_track=params.track, summary=text
            )
            return {
                "proposed": True,
                "changed": True,
                "proposal_id": proposal.id,
                "summary": f"Proposed {proposal.id}: {text}. Waiting for approval in VideoView.",
                "path": written.relative_path,
                "digest": written.digest,
            }

        if not params.proposal_id or not params.proposal_id.strip():
            raise VideoError(f"Specify 'proposal_id' for action={params.action!r}.")
        pid = params.proposal_id.strip()

        if params.action == "apply":
            approved, written = apply_proposal(
                work,
                pid,
                decided_in="conversation",
                seen_digest=params.seen_digest,
            )
            return {
                "changed": True,
                "proposal_id": approved.id,
                "status": approved.status,
                "summary": f"Approved and applied proposal {approved.id}: {approved.summary}",
                "path": written.relative_path,
                "digest": written.digest,
            }

        rejected, written = reject_proposal(
            work,
            pid,
            decided_in="conversation",
            reason=params.reason,
            seen_digest=params.seen_digest,
        )
        return {
            "changed": True,
            "proposal_id": rejected.id,
            "status": rejected.status,
            "summary": f"Rejected proposal {rejected.id}: {rejected.summary}",
            "path": written.relative_path,
            "digest": written.digest,
        }


# -- 8. video_audit (`VideoAuditTool`) ---------------------------------------------------


class VideoAuditParams(BaseModel):
    """Parameters for `video_audit`."""

    model_config = _ARGUMENTS


class VideoAuditTool(BaseTool[VideoAuditParams]):
    """Run deterministic vault SHA-256 immutability, derived lineage, and frame boundary checks."""

    name = "video_audit"
    description = (
        "Run deterministic video project audits: verify that all Tier 2 vault media files and "
        "Tier 1 proxies match their SHA-256 digests and remain read-only, check derived asset "
        "lineage integrity, and verify clip frame bounds and timeline continuity."
    )
    params_type = VideoAuditParams
    writes_files: ClassVar[bool] = False
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=VideoAuditParams,
        )

    async def run(self, params: VideoAuditParams, context: ToolContext) -> dict[str, Any]:
        del params
        work = VideoWork.for_context(context)
        report = audit_project(work)
        return audit_digest(report.findings, passed=report.passed)
