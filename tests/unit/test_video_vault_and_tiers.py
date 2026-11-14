"""Tests for the 3-tier immutable media vault, derived asset lineage, and timeline compiler (`#2439`).

Pins:
- `frames_to_timecode` and `timecode_to_frames` round-trip frame counts to `HH:MM:SS:FF`.
- `TrackFile` rejects overlapping clips on the same track.
- `ingest_raw_video` writes an immutable read-only (`0444`) Tier 2 source file in `vault/` and a
  Tier 1 540p short-GOP proxy in `proxies/`, and refuses overwriting an existing `asset_id` with
  different content.
- `bake_derived_asset` renders a new immutable derived asset (`kind="derived"`) with full
  `AssetLineage` while leaving the parent raw asset in `vault/` untouched.
- `swap_clip_to_derived` and `unbake_clip_to_raw` swap a timeline clip to a derived asset and
  restore it back to the original raw asset and frame range.
- `compile_playback_bundle` resolves clips to Tier 1 `proxies/` when `tier="preview"` and
  automatically relinks to Tier 2 `vault/` when `tier="source"` or `tier="master"`.
- `render_final_master` verifies vault integrity, relinks all active clips to Tier 2 `vault/`
  streams, and writes the Tier 3 master deliverable under `exports/`.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from uclone_x_video.audit import audit_project
from uclone_x_video.compiler import compile_playback_bundle, render_final_master
from uclone_x_video.library import VideoChangedError, VideoError, VideoLibrary
from uclone_x_video.schemas import (
    BakeOperation,
    ColorGrade,
    ManifestOutline,
    SequenceSection,
    TitleClip,
    TrackFile,
    VideoClip,
    frames_to_timecode,
    timecode_to_frames,
)
from uclone_x_video.vault import (
    READ_ONLY_MODE,
    VaultIntegrityError,
    bake_derived_asset,
    ingest_raw_video,
    swap_clip_to_derived,
    unbake_clip_to_raw,
    verify_asset_vault_integrity,
)
from uclone_x_video.work import VideoWork


def _make_work(tmp_path: Path, room_id: str = "room-director") -> VideoWork:
    library = VideoLibrary(tmp_path)
    record = library.create(
        "Seoul Night Cut",
        room_id,
        project_id="seoul-night-cut",
        fps=30,
    )
    return VideoWork(library, record.id, room_id)


def test_timecode_roundtrip_and_non_overlapping_track_validation() -> None:
    """Killed by: examples/video/src/uclone_x_video/schemas.py :: total_seconds, ff = divmod(frame, fps)
    Becomes: total_seconds, ff = (0, 0)
    """
    tc = frames_to_timecode(75, 30)
    assert tc == "00:00:02:15"
    assert timecode_to_frames(tc, 30) == 75

    with pytest.raises(ValueError, match="overlapping clips"):
        TrackFile(
            track_id="v1_main",
            kind="video",
            clips=[
                VideoClip(
                    clip_id="c1",
                    asset_id="a_cam",
                    timeline_in_frame=0,
                    source_in_frame=0,
                    source_out_frame=100,
                ),
                VideoClip(
                    clip_id="c2",
                    asset_id="a_cam",
                    timeline_in_frame=50,
                    source_in_frame=0,
                    source_out_frame=100,
                ),
            ],
        )


def test_ingest_raw_video_locks_vault_readonly_and_refuses_mutation(tmp_path: Path) -> None:
    """Killed by: examples/video/src/uclone_x_video/vault.py :: if existing_asset.tiers.source.sha256 != source_sha256:
    Becomes: if False:
    """
    work = _make_work(tmp_path)
    raw_asset, _ = ingest_raw_video(
        work,
        asset_id="a_cam_01",
        filename="a_cam_01.mp4",
        payload=b"RAW_CAMERA_BITSTREAM_01",
        fps=30,
        total_frames=300,
        width=1920,
        height=1080,
    )

    assert raw_asset.kind == "raw"
    assert raw_asset.immutable is True
    assert raw_asset.tiers.source.path.startswith("vault/sha256_")
    assert raw_asset.tiers.preview.path.startswith("proxies/sha256_")
    assert raw_asset.tiers.preview.width == 960
    assert raw_asset.tiers.preview.height == 540

    vault_file = work.project_dir / raw_asset.tiers.source.path
    assert vault_file.is_file()
    mode = stat.S_IMODE(vault_file.stat().st_mode)
    assert mode == READ_ONLY_MODE
    verify_asset_vault_integrity(work, raw_asset)

    # Attempting to overwrite the same asset_id with different bytes is refused
    with pytest.raises(VaultIntegrityError, match="immutable"):
        ingest_raw_video(
            work,
            asset_id="a_cam_01",
            filename="a_cam_01.mp4",
            payload=b"DIFFERENT_BITSTREAM_ATTACK",
            fps=30,
            total_frames=300,
        )


def test_bake_derived_asset_preserves_raw_and_supports_swap_and_unbake(tmp_path: Path) -> None:
    """Killed by: examples/video/src/uclone_x_video/vault.py :: if op.op in ("optical_flow_slowmo", "speed_ramp"):
    Becomes: if False:
    """
    work = _make_work(tmp_path)
    raw_bytes = b"ORIGINAL_RAW_VIDEO_STREAM_1080P"
    raw_asset, _ = ingest_raw_video(
        work,
        asset_id="a_cam_01",
        filename="a_cam_01.mp4",
        payload=raw_bytes,
        fps=30,
        total_frames=300,
        width=1920,
        height=1080,
    )
    raw_vault_path = work.project_dir / raw_asset.tiers.source.path

    # Bake a 0.5x optical-flow slow-motion derived asset from frames 30..90 (60 source frames -> 120 derived frames)
    derived_asset, _ = bake_derived_asset(
        work,
        parent_asset_id="a_cam_01",
        source_in_frame=30,
        source_out_frame=90,
        operations=[BakeOperation(op="optical_flow_slowmo", rate=0.5, strength=0.9)],
        derived_asset_id="a_cam_01_slowmo_2x",
    )

    assert derived_asset.kind == "derived"
    assert derived_asset.immutable is True
    assert derived_asset.total_frames == 120
    assert derived_asset.lineage is not None
    assert derived_asset.lineage.parent_asset_ids == ["a_cam_01"]
    assert derived_asset.lineage.source_in_frame == 30
    assert derived_asset.lineage.source_out_frame == 90
    assert derived_asset.tiers.source.sha256 != raw_asset.tiers.source.sha256

    # Original raw asset in vault/ is completely untouched
    assert raw_vault_path.read_bytes() == raw_bytes
    verify_asset_vault_integrity(work, raw_asset)
    verify_asset_vault_integrity(work, derived_asset)

    # Swap clip on track to derived asset and ripple-shift subsequent clip
    track = TrackFile(
        track_id="v1_main",
        kind="video",
        clips=[
            VideoClip(
                clip_id="clip_01",
                asset_id="a_cam_01",
                timeline_in_frame=0,
                source_in_frame=30,
                source_out_frame=90,
            ),
            VideoClip(
                clip_id="clip_02",
                asset_id="a_cam_01",
                timeline_in_frame=60,
                source_in_frame=100,
                source_out_frame=160,
            ),
        ],
    )
    swapped_track = swap_clip_to_derived(
        track,
        clip_id="clip_01",
        derived_asset=derived_asset,
    )
    assert swapped_track.clips[0].asset_id == "a_cam_01_slowmo_2x"
    assert swapped_track.clips[0].duration_frames == 120
    assert swapped_track.clips[0].raw_fallback_ref is not None
    assert swapped_track.clips[0].raw_fallback_ref.asset_id == "a_cam_01"
    # Second clip shifted from 60 to 120
    assert swapped_track.clips[1].timeline_in_frame == 120

    # Unbake clip_01 back to original raw asset
    assets_map = {"a_cam_01": raw_asset, "a_cam_01_slowmo_2x": derived_asset}
    unbaked_track = unbake_clip_to_raw(
        swapped_track,
        clip_id="clip_01",
        assets_by_id=assets_map,
    )
    assert unbaked_track.clips[0].asset_id == "a_cam_01"
    assert unbaked_track.clips[0].source_in_frame == 30
    assert unbaked_track.clips[0].source_out_frame == 90
    assert unbaked_track.clips[0].raw_fallback_ref is None
    assert unbaked_track.clips[1].timeline_in_frame == 60


def test_three_tier_compiler_relinks_from_proxy_to_source_vault_on_master_render(
    tmp_path: Path,
) -> None:
    """Killed by: examples/video/src/uclone_x_video/compiler.py :: if tier == "preview":
    Becomes: if True:
    """
    work = _make_work(tmp_path)
    raw_asset, _ = ingest_raw_video(
        work,
        asset_id="a_cam_01",
        filename="a_cam_01.mp4",
        fps=30,
        total_frames=300,
        width=1920,
        height=1080,
    )
    derived_asset, _ = bake_derived_asset(
        work,
        parent_asset_id="a_cam_01",
        source_in_frame=0,
        source_out_frame=60,
        operations=[BakeOperation(op="stabilize", rate=1.0, strength=0.8)],
        derived_asset_id="a_cam_01_stab",
    )

    work.write_manifest(
        ManifestOutline(
            fps=30,
            width=1920,
            height=1080,
            active_playback_tier="preview",
            sections=[
                SequenceSection(
                    id="sec_main",
                    title="Main",
                    kind="main",
                    start_frame=0,
                    end_frame=180,
                )
            ],
        ),
        expected_digest=None,
    )
    work.write_track(
        TrackFile(
            track_id="v1_main",
            kind="video",
            clips=[
                VideoClip(
                    clip_id="clip_01",
                    asset_id="a_cam_01_stab",
                    timeline_in_frame=0,
                    source_in_frame=0,
                    source_out_frame=60,
                    color_grade=ColorGrade(lut="teal_orange_cinema", contrast=1.15),
                ),
                VideoClip(
                    clip_id="clip_02",
                    asset_id="a_cam_01",
                    timeline_in_frame=60,
                    source_in_frame=60,
                    source_out_frame=180,
                ),
            ],
        ),
        expected_digest=None,
    )
    work.write_track(
        TrackFile(
            track_id="t1_titles",
            kind="title",
            titles=[
                TitleClip(
                    title_id="title_01",
                    text="SEOUL DIRECTOR CUT",
                    timeline_in_frame=0,
                    timeline_out_frame=60,
                )
            ],
        ),
        expected_digest=None,
    )

    # Tier 1 (`preview`): resolves to 540p proxies in `proxies/`
    preview_bundle = compile_playback_bundle(work, tier="preview")
    assert preview_bundle.active_tier == "preview"
    assert preview_bundle.width == 960
    assert preview_bundle.height == 540
    v1_prev = next(t for t in preview_bundle.tracks if t.track_id == "v1_main")
    c1_prev = v1_prev.clips[0]
    assert c1_prev.stream_path == derived_asset.tiers.preview.path
    assert c1_prev.stream_path.startswith("proxies/")
    assert c1_prev.stream_width == 960

    # Tier 2 (`source`): automatically relinks to immutable `vault/` streams
    source_bundle = compile_playback_bundle(work, tier="source")
    assert source_bundle.active_tier == "source"
    v1_src = next(t for t in source_bundle.tracks if t.track_id == "v1_main")
    c1_src = v1_src.clips[0]
    assert c1_src.stream_path == derived_asset.tiers.source.path
    assert c1_src.stream_path.startswith("vault/")
    assert c1_src.stream_width == 1920

    # Tier 3 (`master`): renders final deliverable in `exports/` using relinked Tier 2 vault sources
    master_res = render_final_master(work)
    assert master_res.tier == "master"
    assert master_res.output_path.startswith("exports/")
    assert (work.project_dir / master_res.output_path).is_file()
    assert master_res.relinked_source_assets == [
        derived_asset.tiers.source.path,
        raw_asset.tiers.source.path,
    ]
    assert "lut3d=preset=teal_orange_cinema" in master_res.ffmpeg_filtergraph
    assert "loudnorm=I=-14.0" in master_res.ffmpeg_filtergraph


def test_audit_detects_vault_tampering_and_track_revert_restores_snapshot(
    tmp_path: Path,
) -> None:
    """Killed by: examples/video/src/uclone_x_video/audit.py :: verify_asset_vault_integrity(work, asset)
    Becomes: pass
    """
    work = _make_work(tmp_path)
    asset, _ = ingest_raw_video(
        work,
        asset_id="a_cam_01",
        filename="a_cam_01.mp4",
        fps=30,
        total_frames=200,
    )
    first_write = work.write_track(
        TrackFile(
            track_id="v1_main",
            kind="video",
            clips=[
                VideoClip(
                    clip_id="clip_01",
                    asset_id="a_cam_01",
                    timeline_in_frame=0,
                    source_in_frame=0,
                    source_out_frame=100,
                )
            ],
        ),
        expected_digest=None,
        summary="v1 initial",
    )
    second_write = work.write_track(
        TrackFile(
            track_id="v1_main",
            kind="video",
            clips=[
                VideoClip(
                    clip_id="clip_01",
                    asset_id="a_cam_01",
                    timeline_in_frame=0,
                    source_in_frame=0,
                    source_out_frame=50,
                )
            ],
        ),
        expected_digest=first_write.digest,
        summary="v1 trimmed",
    )
    restored, _ = work.revert_track("v1_main", expected_digest=second_write.digest)
    assert restored.clips[0].source_out_frame == 100

    # Make vault file writable -> audit reports vault_integrity_error
    vault_file = work.project_dir / asset.tiers.source.path
    os.chmod(vault_file, stat.S_IRUSR | stat.S_IWUSR)
    try:
        report = audit_project(work)
        assert report.passed is False
        assert any(f.code == "VAULT_INTEGRITY_ERROR" for f in report.findings)
    finally:
        os.chmod(vault_file, READ_ONLY_MODE)


def test_take_over_if_held_by_refuses_when_another_room_took_lease(tmp_path: Path) -> None:
    """Killed by: examples/video/src/uclone_x_video/library.py :: if current != expected_holder:
    Becomes: if False:
    """
    library = VideoLibrary(tmp_path)
    library.create("Demo Cut", "room-old", project_id="demo-cut")
    library.take_over("demo-cut", "room-live")

    transferred = library.take_over_if_held_by("demo-cut", "room-dock", expected_holder="room-old")
    assert transferred is False
    assert library.load("demo-cut").lease is not None
    assert library.load("demo-cut").lease.holder == "room-live"  # type: ignore[union-attr]

    with pytest.raises(VideoChangedError):
        work = VideoWork(library, "demo-cut", "room-live")
        work.write_manifest(ManifestOutline(fps=30), expected_digest="stale_digest")

    with pytest.raises(VideoError):
        library.load("unknown-project")
