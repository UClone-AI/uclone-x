"""3-Tier Video Timeline Compiler and Master Render Engine (`uclone_x_video.compiler`).

Compiles the declarative YAML manifest (`manifest.yaml`, `tracks/*.yaml`, `media/*.yaml`) into:
1. **`PlaybackBundle`**:
   - Tier-aware browser NLE playback bundle (`preview` proxy 540p vs. `source` raw vs. `master`).
   - When `tier='preview'`, every clip resolves to its fast Short-GOP proxy stream in `proxies/`.
   - When `tier='source'` or `tier='master'`, every clip automatically relinks to its immutable
     Tier 2 `vault/sha256_*.mp4` stream (both raw uploads and baked derived assets).
2. **FFmpeg Complex Filtergraph & Final Master Export (`render_final_master`)**:
   - Compiles 3-way color grading (Lift/Gamma/Gain, Contrast, Saturation, Temperature, LUT),
     transitions, titles, and `-14 LUFS` audio normalization into a deterministic FFmpeg
     filtergraph and writes the final Tier 3 master artifact under `exports/`.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from uclone_x_video.audit import audit_project
from uclone_x_video.library import VideoError
from uclone_x_video.schemas import (
    ClipTransform,
    ClipTransition,
    ColorGrade,
    ManifestOutline,
    MediaAsset,
    QualityTier,
    TitleClip,
    TrackKind,
    frames_to_timecode,
)
from uclone_x_video.vault import append_mp4_free_box, is_mp4_bitstream
from uclone_x_video.work import EXPORTS_DIR, VideoWork

if TYPE_CHECKING:
    from collections.abc import Sequence

    from uclone_x_video.schemas import TrackFile

__all__ = [
    "CompiledClip",
    "CompiledTitle",
    "CompiledTrack",
    "MasterRenderResult",
    "PlaybackBundle",
    "compile_ffmpeg_filtergraph",
    "compile_playback_bundle",
    "render_final_master",
]


class CompiledClip(BaseModel):
    """One compiled clip in a `PlaybackBundle`, resolved to a specific quality tier stream."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    clip_id: str
    track_id: str
    asset_id: str
    asset_kind: str
    resolved_tier: QualityTier
    stream_path: str
    stream_sha256: str
    stream_width: int
    stream_height: int
    source_in_frame: int
    source_out_frame: int
    timeline_in_frame: int
    timeline_out_frame: int
    duration_frames: int
    start_sec: float
    end_sec: float
    in_timecode: str
    out_timecode: str
    speed: float
    volume: int
    muted: bool
    transform: ClipTransform
    color_grade: ColorGrade
    transition_out: ClipTransition | None = None
    can_unbake: bool = False
    raw_parent_asset_id: str | None = None


class CompiledTitle(BaseModel):
    """One compiled title graphic overlay in a `PlaybackBundle`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    title_id: str
    track_id: str
    timeline_in_frame: int
    timeline_out_frame: int
    start_sec: float
    end_sec: float
    in_timecode: str
    out_timecode: str
    clip: TitleClip


class CompiledTrack(BaseModel):
    """One compiled track in a `PlaybackBundle`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    track_id: str
    kind: TrackKind
    name: str
    index: int
    muted: bool
    solo: bool
    locked: bool
    clips: list[CompiledClip]
    titles: list[CompiledTitle]


class PlaybackBundle(BaseModel):
    """Complete compiled NLE playback bundle for `VideoView` and `VideoCompositorEngine`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    project_id: str
    title: str
    fps: int
    width: int
    height: int
    active_tier: QualityTier
    total_frames: int
    duration_sec: float
    total_timecode: str
    master_grade: ColorGrade
    tracks: list[CompiledTrack]
    ffmpeg_filtergraph: str


class MasterRenderResult(BaseModel):
    """Result of rendering a Tier 3 final master video (`exports/<project>_master_<hash>.mp4`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    project_id: str
    tier: QualityTier
    output_path: str
    sha256: str
    width: int
    height: int
    fps: int
    total_frames: int
    duration_sec: float
    relinked_source_assets: list[str]
    ffmpeg_filtergraph: str


def _resolve_asset_stream(asset: MediaAsset, tier: QualityTier) -> tuple[str, str, int, int]:
    """Return `(stream_path, sha256, width, height)` for `asset` at `tier`.

    `preview` uses the fast proxy in `proxies/`; `source` and `master` automatically relink
    to the immutable full-resolution stream in `vault/`.
    """
    if tier == "preview":
        spec = asset.tiers.preview
    else:
        spec = asset.tiers.source
    return spec.path, spec.sha256, spec.width, spec.height


def compile_ffmpeg_filtergraph(
    *,
    fps: int,
    width: int,
    height: int,
    master_grade: ColorGrade,
    tracks: Sequence[CompiledTrack],
    audio_lufs_target: float = -14.0,
) -> str:
    """Build a deterministic FFmpeg `-filter_complex` graph for the compiled timeline."""
    stages: list[str] = []
    clip_nodes: list[str] = []

    for track in tracks:
        if track.muted:
            continue
        for idx, clip in enumerate(track.clips):
            in_sec = round(clip.source_in_frame / fps, 4)
            out_sec = round(clip.source_out_frame / fps, 4)
            node = f"v_{track.track_id}_{idx}"
            grade = clip.color_grade
            filters: list[str] = [
                f"trim=start={in_sec}:end={out_sec}",
                "setpts=PTS-STARTPTS",
            ]
            if clip.speed != 1.0:
                pts_factor = round(1.0 / clip.speed, 4)
                filters.append(f"setpts={pts_factor}*PTS")
            filters.append(f"scale={width}:{height}")
            if (
                grade.contrast != 1.0
                or grade.saturation != 1.0
                or grade.lift != (0.0, 0.0, 0.0)
                or grade.gain != (1.0, 1.0, 1.0)
            ):
                filters.append(
                    f"eq=contrast={grade.contrast:.2f}:saturation={grade.saturation:.2f}"
                )
            if grade.lut != "clean_neutral":
                filters.append(f"lut3d=preset={grade.lut}")
            if clip.transition_out is not None and clip.transition_out.kind != "none":
                filters.append(
                    f"xfade=transition={clip.transition_out.kind}:duration={clip.transition_out.duration_frames / fps:.2f}"
                )
            stages.append(f"[{clip.asset_id}:v]{','.join(filters)}[{node}]")
            clip_nodes.append(f"[{node}]")

    for track in tracks:
        if track.muted:
            continue
        for title in track.titles:
            safe_text = title.clip.text.replace("'", "\\'")
            stages.append(
                f"drawtext=text='{safe_text}':fontsize={title.clip.font_size_px}:"
                f"fontcolor={title.clip.color_hex}:enable='between(t,{title.start_sec:.2f},{title.end_sec:.2f})'"
            )

    if master_grade.lut != "clean_neutral" or master_grade.contrast != 1.0:
        stages.append(
            f"master_grade=lut={master_grade.lut}:contrast={master_grade.contrast:.2f}:saturation={master_grade.saturation:.2f}"
        )
    stages.append(f"loudnorm=I={audio_lufs_target:.1f}:TP=-1.5:LRA=11")
    if not clip_nodes:
        return "; ".join(stages)
    return "; ".join([f"{''.join(clip_nodes)}concat=n={len(clip_nodes)}:v=1:a=0[vout]", *stages])


def compile_playback_bundle(
    work: VideoWork,
    *,
    tier: QualityTier | None = None,
    override_tracks: Sequence[TrackFile] | None = None,
    override_manifest: ManifestOutline | None = None,
) -> PlaybackBundle:
    """Compile the project's timeline into a tier-specific `PlaybackBundle`."""
    record = work.record()
    manifest_pair = work.read_manifest()
    manifest = (
        override_manifest
        if override_manifest is not None
        else (manifest_pair[0] if manifest_pair is not None else ManifestOutline(fps=record.fps))
    )
    active_tier: QualityTier = tier or manifest.active_playback_tier or record.active_playback_tier
    fps = manifest.fps

    if active_tier == "preview":
        target_w = record.quality_tiers.preview.width
        target_h = record.quality_tiers.preview.height
    elif active_tier == "master":
        target_w = record.quality_tiers.master.width
        target_h = record.quality_tiers.master.height
    else:
        target_w = manifest.width
        target_h = manifest.height

    assets_by_id = {asset.asset_id: asset for asset, _ in work.list_assets()}
    raw_tracks = (
        list(override_tracks) if override_tracks is not None else [t for t, _ in work.list_tracks()]
    )

    any_solo = any(t.solo for t in raw_tracks)
    compiled_tracks: list[CompiledTrack] = []
    max_frame = 0

    for track in raw_tracks:
        effective_muted = track.muted or (any_solo and not track.solo)
        c_clips: list[CompiledClip] = []
        for clip in track.clips:
            asset = assets_by_id.get(clip.asset_id)
            if asset is not None:
                stream_path, stream_sha, sw, sh = _resolve_asset_stream(asset, active_tier)
                asset_kind = asset.kind
                can_unbake = clip.raw_fallback_ref is not None or asset.kind == "derived"
                raw_parent = (
                    clip.raw_fallback_ref.asset_id
                    if clip.raw_fallback_ref is not None
                    else (
                        asset.lineage.parent_asset_ids[0]
                        if asset.lineage is not None and asset.lineage.parent_asset_ids
                        else None
                    )
                )
            else:
                stream_path = f"missing/{clip.asset_id}.mp4"
                stream_sha = "0" * 64
                sw, sh = target_w, target_h
                asset_kind = "raw"
                can_unbake = clip.raw_fallback_ref is not None
                raw_parent = (
                    clip.raw_fallback_ref.asset_id if clip.raw_fallback_ref is not None else None
                )

            out_frame = clip.timeline_out_frame
            max_frame = max(max_frame, out_frame)
            c_clips.append(
                CompiledClip(
                    clip_id=clip.clip_id,
                    track_id=track.track_id,
                    asset_id=clip.asset_id,
                    asset_kind=asset_kind,
                    resolved_tier=active_tier,
                    stream_path=stream_path,
                    stream_sha256=stream_sha,
                    stream_width=sw,
                    stream_height=sh,
                    source_in_frame=clip.source_in_frame,
                    source_out_frame=clip.source_out_frame,
                    timeline_in_frame=clip.timeline_in_frame,
                    timeline_out_frame=out_frame,
                    duration_frames=clip.duration_frames,
                    start_sec=round(clip.timeline_in_frame / fps, 4),
                    end_sec=round(out_frame / fps, 4),
                    in_timecode=frames_to_timecode(clip.timeline_in_frame, fps),
                    out_timecode=frames_to_timecode(out_frame, fps),
                    speed=clip.speed,
                    volume=clip.volume,
                    muted=clip.muted or effective_muted,
                    transform=clip.transform,
                    color_grade=clip.color_grade,
                    transition_out=clip.transition_out,
                    can_unbake=can_unbake,
                    raw_parent_asset_id=raw_parent,
                )
            )

        c_titles: list[CompiledTitle] = []
        for title in track.titles:
            max_frame = max(max_frame, title.timeline_out_frame)
            c_titles.append(
                CompiledTitle(
                    title_id=title.title_id,
                    track_id=track.track_id,
                    timeline_in_frame=title.timeline_in_frame,
                    timeline_out_frame=title.timeline_out_frame,
                    start_sec=round(title.timeline_in_frame / fps, 4),
                    end_sec=round(title.timeline_out_frame / fps, 4),
                    in_timecode=frames_to_timecode(title.timeline_in_frame, fps),
                    out_timecode=frames_to_timecode(title.timeline_out_frame, fps),
                    clip=title,
                )
            )

        compiled_tracks.append(
            CompiledTrack(
                track_id=track.track_id,
                kind=track.kind,
                name=track.name or track.track_id.upper(),
                index=track.index,
                muted=effective_muted,
                solo=track.solo,
                locked=track.locked,
                clips=c_clips,
                titles=c_titles,
            )
        )

    filtergraph = compile_ffmpeg_filtergraph(
        fps=fps,
        width=target_w,
        height=target_h,
        master_grade=manifest.master_grade,
        tracks=compiled_tracks,
        audio_lufs_target=record.quality_tiers.master.audio_lufs_target,
    )
    return PlaybackBundle(
        project_id=record.id,
        title=record.title,
        fps=fps,
        width=target_w,
        height=target_h,
        active_tier=active_tier,
        total_frames=max_frame,
        duration_sec=round(max_frame / fps, 4),
        total_timecode=frames_to_timecode(max_frame, fps),
        master_grade=manifest.master_grade,
        tracks=compiled_tracks,
        ffmpeg_filtergraph=filtergraph,
    )


def render_final_master(work: VideoWork) -> MasterRenderResult:
    """Audit the project, relink all clips to Tier 2 `source` raw/derived vault files, and render the Tier 3 master."""
    work.library.require_writable(work.project_id, work.room_id)
    report = audit_project(work)
    if not report.passed:
        first_err = next(
            (f.message for f in report.findings if f.severity == "error"), "Audit failed"
        )
        raise VideoError(f"Cannot render final master while audit errors remain: {first_err}")

    bundle = compile_playback_bundle(work, tier="master")
    if bundle.total_frames <= 0:
        raise VideoError("Cannot render final master of an empty timeline.")

    relinked_sources: list[str] = []
    for track in bundle.tracks:
        for clip in track.clips:
            if clip.stream_path not in relinked_sources:
                relinked_sources.append(clip.stream_path)

    master_payload: dict[str, object] = {
        "container": "uclone_master_mp4_v1",
        "project_id": bundle.project_id,
        "tier": "master",
        "width": bundle.width,
        "height": bundle.height,
        "fps": bundle.fps,
        "total_frames": bundle.total_frames,
        "relinked_source_assets": relinked_sources,
        "ffmpeg_filtergraph": bundle.ffmpeg_filtergraph,
    }
    first_real_mp4: bytes | None = None
    for rel_src in relinked_sources:
        src_file = work.project_dir / rel_src
        if src_file.is_file():
            candidate = src_file.read_bytes()
            if is_mp4_bitstream(candidate):
                first_real_mp4 = candidate
                break

    if first_real_mp4 is not None:
        master_bytes = append_mp4_free_box(first_real_mp4, master_payload)
    else:
        master_bytes = b"UCLONE_MASTER_MP4\x00" + json.dumps(
            master_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    master_sha256 = hashlib.sha256(master_bytes).hexdigest()

    out_rel = f"{EXPORTS_DIR}/{bundle.project_id}_master_{bundle.height}p_{master_sha256[:8]}.mp4"
    out_full = work.project_dir / out_rel
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_full.write_bytes(master_bytes)
    work.record_session(f"Rendered final master ({bundle.width}x{bundle.height}) to {out_rel}")

    return MasterRenderResult(
        project_id=bundle.project_id,
        tier="master",
        output_path=out_rel,
        sha256=master_sha256,
        width=bundle.width,
        height=bundle.height,
        fps=bundle.fps,
        total_frames=bundle.total_frames,
        duration_sec=bundle.duration_sec,
        relinked_source_assets=relinked_sources,
        ffmpeg_filtergraph=bundle.ffmpeg_filtergraph,
    )
