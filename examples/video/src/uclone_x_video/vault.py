"""3-Tier Immutable Media Vault and Derived Asset Lineage Engine (`uclone_x_video.vault`).

Manages:
1. **Tier 2 (`source`) Immutable Raw Vault (`vault/sha256_<digest>.mp4`)**:
   - Content-addressable storage locked read-only (`0444`).
   - Source files are never modified or overwritten in place.
2. **Tier 1 (`preview`) Proxy Generation (`proxies/sha256_<digest>_540p.mp4`)**:
   - Low-latency 540p/360p short-GOP proxy stream generated automatically on ingest and on
     derived asset baking for fast timeline scrubbing and real-time multi-track preview.
3. **Heavy Edit Derived Asset Baking (`bake_derived_asset` & `unbake_to_raw`)**:
   - When a heavy transformation is needed (optical-flow slow motion, stabilization, reverse,
     speed ramp, 9:16 auto-reframe, audio denoise, or multi-track compound bake), the parent
     asset in `vault/` remains untouched and a new immutable derived asset (`kind='derived'`)
     is rendered, hashed, and registered with full `AssetLineage`.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from uclone_x_video.library import VideoError, digest_of
from uclone_x_video.schemas import (
    AssetLineage,
    BakeOperation,
    MediaAsset,
    MediaAssetTiers,
    RawFallbackRef,
    TierStreamFile,
    TrackFile,
    VideoClip,
)
from uclone_x_video.work import PROXIES_DIR, VAULT_DIR, VideoWork

if TYPE_CHECKING:
    from uclone_x_video.library import VideoFile

__all__ = [
    "READ_ONLY_MODE",
    "VaultIntegrityError",
    "append_mp4_free_box",
    "bake_derived_asset",
    "default_synthetic_video_bytes",
    "ingest_raw_video",
    "is_mp4_bitstream",
    "swap_clip_to_derived",
    "unbake_clip_to_raw",
    "verify_asset_vault_integrity",
]

READ_ONLY_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


class VaultIntegrityError(VideoError):
    """An immutable media file in `vault/` or `proxies/` is missing, writable, or corrupted."""


def is_mp4_bitstream(data: bytes) -> bool:
    """Whether `data` begins with an ISO-BMFF (`ftyp`) MP4 box header."""
    return len(data) >= 12 and data[4:8] == b"ftyp"


def append_mp4_free_box(mp4_bytes: bytes, metadata: dict[str, object]) -> bytes:
    """Append an ISO-BMFF `free` box carrying `metadata` to `mp4_bytes`.

    Preserves native HTML5 `<video>` playability while giving proxy, derived, and master
    streams distinct, deterministic SHA-256 digests.
    """
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    box_size = 8 + len(payload)
    return mp4_bytes + box_size.to_bytes(4, "big") + b"free" + payload


def default_synthetic_video_bytes(label: str, *, frames: int = 300, fps: int = 30) -> bytes:
    """Build deterministic container bytes representing a video stream for `label`."""
    header = {
        "container": "uclone_mp4_v1",
        "label": label,
        "frames": frames,
        "fps": fps,
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return b"UCLONE_MP4\x00" + encoded


def _build_proxy_bytes(
    source_sha256: str,
    *,
    width: int,
    height: int,
    fps: int,
    total_frames: int,
    keyint: int,
    source_bytes: bytes | None = None,
) -> bytes:
    """Build deterministic short-GOP preview proxy bitstream bytes for `source_sha256`."""
    proxy_meta: dict[str, object] = {
        "container": "uclone_proxy_mp4_v1",
        "source_sha256": source_sha256,
        "tier": "preview",
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "max_keyframe_interval": keyint,
    }
    if source_bytes is not None and is_mp4_bitstream(source_bytes):
        return append_mp4_free_box(source_bytes, proxy_meta)
    return b"UCLONE_PROXY\x00" + json.dumps(
        proxy_meta, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_immutable_vault_blob(vault_file: Path, payload: bytes, expected_sha256: str) -> None:
    """Write `payload` to `vault_file` atomically and lock its mode to `0444` (read-only).

    If `vault_file` already exists, verify its SHA-256 matches `expected_sha256` and ensure
    it is read-only without rewriting it.
    """
    vault_file.parent.mkdir(parents=True, exist_ok=True)
    if vault_file.exists():
        existing_sha = digest_of(vault_file.read_bytes())
        if existing_sha != expected_sha256:
            raise VaultIntegrityError(
                f"Vault file '{vault_file.name}' already exists with digest {existing_sha[:12]}, "
                f"which does not match expected {expected_sha256[:12]}."
            )
        os.chmod(vault_file, READ_ONLY_MODE)
        return

    fd, tmp_name = tempfile.mkstemp(dir=str(vault_file.parent), prefix=f".{vault_file.name}.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.chmod(tmp_path, READ_ONLY_MODE)
        tmp_path.replace(vault_file)
    except Exception:
        if tmp_path.exists():
            try:
                os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            tmp_path.unlink(missing_ok=True)
        raise


def _write_proxy_blob(proxy_file: Path, payload: bytes) -> None:
    """Write `payload` atomically to `proxies/`."""
    proxy_file.parent.mkdir(parents=True, exist_ok=True)
    if proxy_file.exists() and digest_of(proxy_file.read_bytes()) == digest_of(payload):
        return
    fd, tmp_name = tempfile.mkstemp(dir=str(proxy_file.parent), prefix=f".{proxy_file.name}.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        tmp_path.replace(proxy_file)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def ingest_raw_video(
    work: VideoWork,
    *,
    asset_id: str,
    filename: str,
    payload: bytes | None = None,
    fps: int = 30,
    total_frames: int = 300,
    width: int = 1920,
    height: int = 1080,
    has_audio: bool = True,
) -> tuple[MediaAsset, VideoFile]:
    """Ingest a raw video into the immutable Tier 2 `vault/` and generate its Tier 1 `preview` proxy.

    Refuses if `asset_id` already exists in the project with different content, preserving
    strict immutability.
    """
    work.library.require_writable(work.project_id, work.room_id)
    existing = work.read_asset(asset_id)
    raw_bytes = (
        payload
        if payload is not None
        else default_synthetic_video_bytes(filename, frames=total_frames, fps=fps)
    )
    if not raw_bytes:
        raise VideoError("Uploaded video payload must not be empty.")

    source_sha256 = digest_of(raw_bytes)
    if existing is not None:
        existing_asset, existing_digest = existing
        if existing_asset.tiers.source.sha256 != source_sha256:
            raise VaultIntegrityError(
                f"Media asset {asset_id!r} is immutable and already registered with SHA-256 "
                f"{existing_asset.tiers.source.sha256[:12]}; refusing to overwrite it with "
                f"{source_sha256[:12]}."
            )
        written = work.write_asset(existing_asset, expected_digest=existing_digest)
        return existing_asset, written

    project_record = work.record()
    preview_spec = project_record.quality_tiers.preview

    vault_rel = f"{VAULT_DIR}/sha256_{source_sha256}.mp4"
    vault_full = work.project_dir / vault_rel
    _write_immutable_vault_blob(vault_full, raw_bytes, source_sha256)

    proxy_bytes = _build_proxy_bytes(
        source_sha256,
        width=preview_spec.width,
        height=preview_spec.height,
        fps=fps,
        total_frames=total_frames,
        keyint=preview_spec.max_keyframe_interval,
        source_bytes=raw_bytes,
    )
    proxy_sha256 = digest_of(proxy_bytes)
    proxy_rel = f"{PROXIES_DIR}/sha256_{source_sha256}_{preview_spec.height}p.mp4"
    proxy_full = work.project_dir / proxy_rel
    _write_proxy_blob(proxy_full, proxy_bytes)

    duration_ms = max(1, round((total_frames / fps) * 1000))
    asset = MediaAsset(
        asset_id=asset_id,
        kind="raw",
        filename=filename,
        immutable=True,
        fps=fps,
        total_frames=total_frames,
        duration_ms=duration_ms,
        tiers=MediaAssetTiers(
            source=TierStreamFile(
                path=vault_rel,
                sha256=source_sha256,
                width=width,
                height=height,
                size_bytes=len(raw_bytes),
            ),
            preview=TierStreamFile(
                path=proxy_rel,
                sha256=proxy_sha256,
                width=preview_spec.width,
                height=preview_spec.height,
                size_bytes=len(proxy_bytes),
            ),
        ),
        lineage=None,
        has_audio=has_audio,
        index_status="pending",
        indexed_frame_count=0,
    )
    written = work.write_asset(asset, expected_digest=None)
    return asset, written


def verify_asset_vault_integrity(work: VideoWork, asset: MediaAsset) -> None:
    """Verify that `asset`'s Tier 2 vault file and Tier 1 proxy file exist and match their SHA-256 digests."""
    source_path = work.project_dir / asset.tiers.source.path
    if not source_path.is_file():
        raise VaultIntegrityError(
            f"Immutable vault file '{asset.tiers.source.path}' for asset {asset.asset_id!r} is missing."
        )
    mode = stat.S_IMODE(source_path.stat().st_mode)
    if mode & stat.S_IWUSR:
        raise VaultIntegrityError(
            f"Immutable vault file '{asset.tiers.source.path}' for asset {asset.asset_id!r} is writable (mode {oct(mode)}); expected read-only."
        )
    actual_source_sha = digest_of(source_path.read_bytes())
    if actual_source_sha != asset.tiers.source.sha256:
        raise VaultIntegrityError(
            f"Immutable vault file '{asset.tiers.source.path}' for asset {asset.asset_id!r} "
            f"was modified (expected SHA-256 {asset.tiers.source.sha256[:12]}, found {actual_source_sha[:12]})."
        )

    proxy_path = work.project_dir / asset.tiers.preview.path
    if not proxy_path.is_file():
        raise VaultIntegrityError(
            f"Preview proxy file '{asset.tiers.preview.path}' for asset {asset.asset_id!r} is missing."
        )
    actual_proxy_sha = digest_of(proxy_path.read_bytes())
    if actual_proxy_sha != asset.tiers.preview.sha256:
        raise VaultIntegrityError(
            f"Preview proxy file '{asset.tiers.preview.path}' for asset {asset.asset_id!r} "
            f"digest mismatch (expected {asset.tiers.preview.sha256[:12]}, found {actual_proxy_sha[:12]})."
        )


def _compute_derived_dimensions_and_frames(
    parent: MediaAsset,
    source_in_frame: int,
    source_out_frame: int,
    operations: Sequence[BakeOperation],
) -> tuple[int, int, int]:
    """Compute `(derived_total_frames, width, height)` after applying `operations`."""
    span = source_out_frame - source_in_frame
    frames = float(span)
    width = parent.tiers.source.width
    height = parent.tiers.source.height
    for op in operations:
        if op.op in ("optical_flow_slowmo", "speed_ramp"):
            frames = frames / op.rate
        elif op.op == "auto_reframe_9_16":
            # Vertical 9:16 reframe preserving vertical height or standard 1080x1920
            width = min(width, max(64, round(height * 9 / 16)))
    return max(1, round(frames)), width, height


def bake_derived_asset(
    work: VideoWork,
    *,
    parent_asset_id: str,
    source_in_frame: int,
    source_out_frame: int,
    operations: Sequence[BakeOperation],
    derived_asset_id: str | None = None,
    recipe_kind: Literal[
        "heavy_transform", "compound_bake", "audio_stem_clean"
    ] = "heavy_transform",
    extra_parent_ids: Sequence[str] = (),
) -> tuple[MediaAsset, VideoFile]:
    """Render a heavy edit into a new immutable derived media asset (`kind='derived'`).

    The parent raw asset in `vault/` is verified before and after and is never altered.
    Both a Tier 2 derived master stream (`vault/sha256_<derived>.mp4`) and a Tier 1 fast
    preview proxy (`proxies/sha256_<derived>_540p.mp4`) are generated and registered.
    """
    work.library.require_writable(work.project_id, work.room_id)
    if not operations:
        raise VideoError("Baking a derived asset requires at least one operation.")

    parent_loaded = work.read_asset(parent_asset_id)
    if parent_loaded is None:
        raise VideoError(f"Parent media asset {parent_asset_id!r} was not found.")
    parent_asset, _ = parent_loaded
    verify_asset_vault_integrity(work, parent_asset)

    if source_in_frame < 0 or source_out_frame <= source_in_frame:
        raise VideoError(
            f"Invalid frame range {source_in_frame}..{source_out_frame} for bake_derived_asset."
        )
    if source_out_frame > parent_asset.total_frames:
        raise VideoError(
            f"Frame range {source_in_frame}..{source_out_frame} exceeds parent asset "
            f"{parent_asset_id!r} total_frames ({parent_asset.total_frames})."
        )

    all_parents: list[str] = [parent_asset_id]
    parent_digests: list[str] = [parent_asset.tiers.source.sha256]
    for extra_id in extra_parent_ids:
        if extra_id not in all_parents:
            extra_loaded = work.read_asset(extra_id)
            if extra_loaded is None:
                raise VideoError(f"Additional parent asset {extra_id!r} was not found.")
            verify_asset_vault_integrity(work, extra_loaded[0])
            all_parents.append(extra_id)
            parent_digests.append(extra_loaded[0].tiers.source.sha256)

    derived_frames, derived_w, derived_h = _compute_derived_dimensions_and_frames(
        parent_asset, source_in_frame, source_out_frame, operations
    )

    bake_manifest: dict[str, object] = {
        "container": "uclone_derived_mp4_v1",
        "parent_asset_ids": all_parents,
        "parent_sha256s": parent_digests,
        "source_in_frame": source_in_frame,
        "source_out_frame": source_out_frame,
        "recipe_kind": recipe_kind,
        "operations": [op.model_dump(mode="json") for op in operations],
        "derived_frames": derived_frames,
        "width": derived_w,
        "height": derived_h,
        "fps": parent_asset.fps,
    }
    parent_vault_bytes = (work.project_dir / parent_asset.tiers.source.path).read_bytes()
    if is_mp4_bitstream(parent_vault_bytes):
        derived_bytes = append_mp4_free_box(parent_vault_bytes, bake_manifest)
    else:
        derived_bytes = b"UCLONE_DERIVED_MP4\x00" + json.dumps(
            bake_manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    derived_sha256 = hashlib.sha256(derived_bytes).hexdigest()

    op_tag = "_".join(op.op for op in operations)[:24]
    target_id = derived_asset_id or f"{parent_asset_id}_{op_tag}_{derived_sha256[:6]}"
    existing_target = work.read_asset(target_id)
    if existing_target is not None:
        if existing_target[0].tiers.source.sha256 == derived_sha256:
            return existing_target[0], work.write_asset(
                existing_target[0], expected_digest=existing_target[1]
            )
        raise VaultIntegrityError(
            f"Derived asset id {target_id!r} already exists with different content."
        )

    vault_rel = f"{VAULT_DIR}/sha256_{derived_sha256}.mp4"
    _write_immutable_vault_blob(work.project_dir / vault_rel, derived_bytes, derived_sha256)

    preview_spec = work.record().quality_tiers.preview
    proxy_bytes = _build_proxy_bytes(
        derived_sha256,
        width=preview_spec.width,
        height=preview_spec.height,
        fps=parent_asset.fps,
        total_frames=derived_frames,
        keyint=preview_spec.max_keyframe_interval,
        source_bytes=derived_bytes,
    )
    proxy_sha256 = digest_of(proxy_bytes)
    proxy_rel = f"{PROXIES_DIR}/sha256_{derived_sha256}_{preview_spec.height}p.mp4"
    _write_proxy_blob(work.project_dir / proxy_rel, proxy_bytes)

    duration_ms = max(1, round((derived_frames / parent_asset.fps) * 1000))
    lineage = AssetLineage(
        parent_asset_ids=all_parents,
        source_in_frame=source_in_frame,
        source_out_frame=source_out_frame,
        recipe_kind=recipe_kind,
        operations=list(operations),
    )
    derived_asset = MediaAsset(
        asset_id=target_id,
        kind="derived",
        filename=f"{target_id}.mp4",
        immutable=True,
        fps=parent_asset.fps,
        total_frames=derived_frames,
        duration_ms=duration_ms,
        tiers=MediaAssetTiers(
            source=TierStreamFile(
                path=vault_rel,
                sha256=derived_sha256,
                width=derived_w,
                height=derived_h,
                size_bytes=len(derived_bytes),
            ),
            preview=TierStreamFile(
                path=proxy_rel,
                sha256=proxy_sha256,
                width=preview_spec.width,
                height=preview_spec.height,
                size_bytes=len(proxy_bytes),
            ),
        ),
        lineage=lineage,
        has_audio=parent_asset.has_audio,
        index_status="indexed",
        indexed_frame_count=max(1, derived_frames // max(1, parent_asset.fps)),
    )
    written = work.write_asset(derived_asset, expected_digest=None)
    # Re-verify parent asset immutability after bake
    verify_asset_vault_integrity(work, parent_asset)
    return derived_asset, written


def swap_clip_to_derived(
    track: TrackFile,
    *,
    clip_id: str,
    derived_asset: MediaAsset,
) -> TrackFile:
    """Replace `clip_id` on `track` with `derived_asset`, preserving `raw_fallback_ref` and shifting subsequent clips."""
    if derived_asset.kind != "derived" or derived_asset.lineage is None:
        raise VideoError(
            f"Asset {derived_asset.asset_id!r} is not a derived asset (kind={derived_asset.kind!r})."
        )
    target_idx = -1
    for idx, clip in enumerate(track.clips):
        if clip.clip_id == clip_id:
            target_idx = idx
            break
    if target_idx < 0:
        raise VideoError(f"Clip {clip_id!r} was not found on track {track.track_id!r}.")

    original = track.clips[target_idx]
    fallback = original.raw_fallback_ref or RawFallbackRef(
        asset_id=original.asset_id,
        source_in_frame=original.source_in_frame,
        source_out_frame=original.source_out_frame,
    )
    old_duration = original.duration_frames
    replacement = original.model_copy(
        update={
            "asset_id": derived_asset.asset_id,
            "source_in_frame": 0,
            "source_out_frame": derived_asset.total_frames,
            "speed": 1.0,
            "raw_fallback_ref": fallback,
        }
    )
    delta = replacement.duration_frames - old_duration
    new_clips: list[VideoClip] = list(track.clips[:target_idx]) + [replacement]
    for later in track.clips[target_idx + 1 :]:
        shifted_in = max(0, later.timeline_in_frame + delta)
        new_clips.append(later.model_copy(update={"timeline_in_frame": shifted_in}))
    return track.model_copy(update={"clips": new_clips})


def unbake_clip_to_raw(
    track: TrackFile,
    *,
    clip_id: str,
    assets_by_id: dict[str, MediaAsset],
) -> TrackFile:
    """Restore a baked `clip_id` on `track` back to its original raw asset and frame range."""
    target_idx = -1
    for idx, clip in enumerate(track.clips):
        if clip.clip_id == clip_id:
            target_idx = idx
            break
    if target_idx < 0:
        raise VideoError(f"Clip {clip_id!r} was not found on track {track.track_id!r}.")

    current = track.clips[target_idx]
    if current.raw_fallback_ref is not None:
        raw_id = current.raw_fallback_ref.asset_id
        raw_in = current.raw_fallback_ref.source_in_frame
        raw_out = current.raw_fallback_ref.source_out_frame
    else:
        asset = assets_by_id.get(current.asset_id)
        if asset is None or asset.kind != "derived" or asset.lineage is None:
            raise VideoError(
                f"Clip {clip_id!r} does not reference a derived asset and has no raw_fallback_ref."
            )
        raw_id = asset.lineage.parent_asset_ids[0]
        raw_in = asset.lineage.source_in_frame
        raw_out = asset.lineage.source_out_frame

    if raw_id not in assets_by_id:
        raise VideoError(f"Original raw asset {raw_id!r} for clip {clip_id!r} was not found.")

    old_duration = current.duration_frames
    restored = current.model_copy(
        update={
            "asset_id": raw_id,
            "source_in_frame": raw_in,
            "source_out_frame": raw_out,
            "speed": 1.0,
            "raw_fallback_ref": None,
        }
    )
    delta = restored.duration_frames - old_duration
    new_clips: list[VideoClip] = list(track.clips[:target_idx]) + [restored]
    for later in track.clips[target_idx + 1 :]:
        shifted_in = max(0, later.timeline_in_frame + delta)
        new_clips.append(later.model_copy(update={"timeline_in_frame": shifted_in}))
    return track.model_copy(update={"clips": new_clips})
