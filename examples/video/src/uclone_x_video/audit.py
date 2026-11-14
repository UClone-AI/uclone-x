"""Deterministic video project audit engine (`uclone_x_video.audit`).

Audits:
1. **Immutable Media Vault Integrity (`VAULT_INTEGRITY_ERROR`)**:
   - Every asset's Tier 2 `source` file in `vault/` exists, is read-only (`0444`), and
     matches its SHA-256 digest; its Tier 1 `preview` proxy in `proxies/` matches its SHA-256.
2. **Derived Asset Lineage Integrity (`DERIVED_LINEAGE_BROKEN`)**:
   - Every `kind='derived'` asset references existing parent assets and valid frame bounds.
3. **Timeline Clip Frame Bounds (`CLIP_SOURCE_OUT_OF_BOUNDS`, `MISSING_MEDIA_ASSET`)**:
   - Every clip references a registered `MediaAsset` and stays within `0..asset.total_frames`.
4. **Transition & Timeline Continuity (`TRANSITION_EXCEEDS_CLIP`, `TIMELINE_BLACK_GAP`, `AUDIO_GAIN_CLIPPING`)**:
   - Outgoing transition durations do not exceed the clip's duration; unintentional black-frame
     gaps on primary video tracks and excessive audio gain (`>125%`) are flagged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from uclone_x_video.vault import VaultIntegrityError, verify_asset_vault_integrity

if TYPE_CHECKING:
    from uclone_x_video.work import VideoWork

__all__ = [
    "AuditCode",
    "AuditSeverity",
    "VideoAuditFinding",
    "VideoAuditReport",
    "audit_project",
]

AuditSeverity = Literal["error", "warning"]
AuditCode = Literal[
    "VAULT_INTEGRITY_ERROR",
    "DERIVED_LINEAGE_BROKEN",
    "MISSING_MEDIA_ASSET",
    "CLIP_SOURCE_OUT_OF_BOUNDS",
    "TRANSITION_EXCEEDS_CLIP",
    "TIMELINE_BLACK_GAP",
    "AUDIO_GAIN_CLIPPING",
]


class VideoAuditFinding(BaseModel):
    """One deterministic finding from `audit_project`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    code: AuditCode
    severity: AuditSeverity
    frame: int = 0
    track_id: str | None = None
    clip_id: str | None = None
    asset_id: str | None = None
    message: str


class VideoAuditReport(BaseModel):
    """Complete deterministic audit report for a video project."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    passed: bool
    findings: list[VideoAuditFinding]


def audit_project(work: VideoWork) -> VideoAuditReport:
    """Run all deterministic vault, lineage, and timeline checks on `work`."""
    findings: list[VideoAuditFinding] = []
    assets_list = work.list_assets()
    assets_by_id = {asset.asset_id: asset for asset, _ in assets_list}

    # 1. Vault SHA-256 and read-only immutability checks + derived lineage checks
    for asset, _ in assets_list:
        try:
            verify_asset_vault_integrity(work, asset)
        except VaultIntegrityError as exc:
            findings.append(
                VideoAuditFinding(
                    code="VAULT_INTEGRITY_ERROR",
                    severity="error",
                    frame=0,
                    asset_id=asset.asset_id,
                    message=str(exc),
                )
            )
        if asset.kind == "derived" and asset.lineage is not None:
            for parent_id in asset.lineage.parent_asset_ids:
                parent = assets_by_id.get(parent_id)
                if parent is None:
                    findings.append(
                        VideoAuditFinding(
                            code="DERIVED_LINEAGE_BROKEN",
                            severity="error",
                            frame=asset.lineage.source_in_frame,
                            asset_id=asset.asset_id,
                            message=(
                                f"Derived asset {asset.asset_id!r} references missing parent "
                                f"asset {parent_id!r}."
                            ),
                        )
                    )
                elif asset.lineage.source_out_frame > parent.total_frames:
                    findings.append(
                        VideoAuditFinding(
                            code="DERIVED_LINEAGE_BROKEN",
                            severity="error",
                            frame=asset.lineage.source_out_frame,
                            asset_id=asset.asset_id,
                            message=(
                                f"Derived asset {asset.asset_id!r} lineage source_out_frame "
                                f"({asset.lineage.source_out_frame}) exceeds parent {parent_id!r} "
                                f"total_frames ({parent.total_frames})."
                            ),
                        )
                    )

    # 2. Timeline tracks and clip checks
    for track, _ in work.list_tracks():
        cursor_frame = 0
        for idx, clip in enumerate(track.clips):
            asset = assets_by_id.get(clip.asset_id)
            if asset is None:
                findings.append(
                    VideoAuditFinding(
                        code="MISSING_MEDIA_ASSET",
                        severity="error",
                        frame=clip.timeline_in_frame,
                        track_id=track.track_id,
                        clip_id=clip.clip_id,
                        asset_id=clip.asset_id,
                        message=(
                            f"Clip {clip.clip_id!r} on track {track.track_id!r} references "
                            f"unknown asset {clip.asset_id!r}."
                        ),
                    )
                )
            elif clip.source_out_frame > asset.total_frames:
                findings.append(
                    VideoAuditFinding(
                        code="CLIP_SOURCE_OUT_OF_BOUNDS",
                        severity="error",
                        frame=clip.timeline_in_frame,
                        track_id=track.track_id,
                        clip_id=clip.clip_id,
                        asset_id=clip.asset_id,
                        message=(
                            f"Clip {clip.clip_id!r} on track {track.track_id!r} requests "
                            f"source_out_frame {clip.source_out_frame}, exceeding asset "
                            f"{asset.asset_id!r} total_frames ({asset.total_frames})."
                        ),
                    )
                )

            if (
                clip.transition_out is not None
                and clip.transition_out.kind != "none"
                and clip.transition_out.duration_frames >= clip.duration_frames
            ):
                findings.append(
                    VideoAuditFinding(
                        code="TRANSITION_EXCEEDS_CLIP",
                        severity="error",
                        frame=clip.timeline_in_frame,
                        track_id=track.track_id,
                        clip_id=clip.clip_id,
                        message=(
                            f"Clip {clip.clip_id!r} transition duration "
                            f"({clip.transition_out.duration_frames} frames) must be shorter "
                            f"than the clip duration ({clip.duration_frames} frames)."
                        ),
                    )
                )

            if track.kind == "video" and track.index == 1:
                if idx == 0 and clip.timeline_in_frame > 0:
                    findings.append(
                        VideoAuditFinding(
                            code="TIMELINE_BLACK_GAP",
                            severity="warning",
                            frame=0,
                            track_id=track.track_id,
                            clip_id=clip.clip_id,
                            message=(
                                f"Primary video track {track.track_id!r} starts with a black "
                                f"gap of {clip.timeline_in_frame} frames before {clip.clip_id!r}."
                            ),
                        )
                    )
                elif idx > 0 and clip.timeline_in_frame > cursor_frame:
                    gap = clip.timeline_in_frame - cursor_frame
                    findings.append(
                        VideoAuditFinding(
                            code="TIMELINE_BLACK_GAP",
                            severity="warning",
                            frame=cursor_frame,
                            track_id=track.track_id,
                            clip_id=clip.clip_id,
                            message=(
                                f"Primary video track {track.track_id!r} has a {gap}-frame "
                                f"black gap between frame {cursor_frame} and {clip.timeline_in_frame}."
                            ),
                        )
                    )

            if not clip.muted and clip.volume > 125:
                findings.append(
                    VideoAuditFinding(
                        code="AUDIO_GAIN_CLIPPING",
                        severity="warning",
                        frame=clip.timeline_in_frame,
                        track_id=track.track_id,
                        clip_id=clip.clip_id,
                        message=(
                            f"Clip {clip.clip_id!r} on track {track.track_id!r} has volume "
                            f"{clip.volume}%, which may cause audio clipping above 125%."
                        ),
                    )
                )

            cursor_frame = clip.timeline_out_frame

    passed = not any(f.severity == "error" for f in findings)
    return VideoAuditReport(passed=passed, findings=findings)
