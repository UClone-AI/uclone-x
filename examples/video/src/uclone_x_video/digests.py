"""The short forms the video tools return to the model (#2429).

A tool result in context is a digest and a short summary, or the slice the model asked for;
a timeline manifest appears in context at most once. Writes return what changed and the
file's new digest, never the whole file; lists return one compact entry per item; audits
return their first findings and the count of the rest.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uclone_x_video.audit import VideoAuditFinding
    from uclone_x_video.context import TimelineContextBundle
    from uclone_x_video.schemas import MediaAsset, TitleClip, VideoClip, VideoProposal

__all__ = [
    "AUDIT_FINDINGS_SHOWN",
    "LIST_ITEMS_SHOWN",
    "NOTHING_CHANGED",
    "asset_line",
    "audit_digest",
    "bundle_view",
    "clip_view",
    "digest_of_value",
    "proposal_line",
    "title_view",
    "write_result",
]

#: The opening words of every result whose write left the file as it was.
NOTHING_CHANGED = "Nothing changed"
#: How many findings an audit result lists; the rest are counted by kind.
AUDIT_FINDINGS_SHOWN = 8
#: How many items a list result shows before it says how many more there are.
LIST_ITEMS_SHOWN = 20


def digest_of_value(value: object) -> str:
    """SHA-256 of `value`'s canonical JSON."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_result(
    path: str,
    previous: str | None,
    digest: str,
    summary: str,
    **extra: Any,
) -> dict[str, Any]:
    """A write's result: whether it changed the file, the digests, and what changed."""
    changed = previous != digest
    text = (
        summary
        if changed
        else f"{NOTHING_CHANGED}: {path} already held this, so nothing was written."
    )
    return {
        "changed": changed,
        "path": path,
        "digest": digest,
        "previous_digest": previous,
        "summary": text,
        **extra,
    }


def asset_line(asset: MediaAsset) -> dict[str, Any]:
    """One media asset as the fields a timeline or bake decision needs."""
    line: dict[str, Any] = {
        "asset_id": asset.asset_id,
        "kind": asset.kind,
        "filename": asset.filename,
        "fps": asset.fps,
        "total_frames": asset.total_frames,
        "duration_ms": asset.duration_ms,
        "source_resolution": f"{asset.tiers.source.width}x{asset.tiers.source.height}",
        "preview_resolution": f"{asset.tiers.preview.width}x{asset.tiers.preview.height}",
        "source_sha256": asset.tiers.source.sha256[:16],
        "indexed_frame_count": asset.indexed_frame_count,
    }
    if asset.lineage is not None:
        line["parent_asset_ids"] = list(asset.lineage.parent_asset_ids)
        line["source_frames"] = f"{asset.lineage.source_in_frame}..{asset.lineage.source_out_frame}"
        line["operations"] = [op.op for op in asset.lineage.operations]
    return line


def clip_view(clip: VideoClip) -> dict[str, Any]:
    """A compact view of one `VideoClip` without default transform/grade noise."""
    info: dict[str, Any] = {
        "clip_id": clip.clip_id,
        "asset_id": clip.asset_id,
        "source_frames": f"{clip.source_in_frame}..{clip.source_out_frame}",
        "timeline_frames": f"{clip.timeline_in_frame}..{clip.timeline_out_frame}",
        "speed": clip.speed,
    }
    if clip.color_grade.lut != "clean_neutral":
        info["lut"] = clip.color_grade.lut
    if clip.transition_out is not None and clip.transition_out.kind != "none":
        info["transition_out"] = (
            f"{clip.transition_out.kind}({clip.transition_out.duration_frames}f)"
        )
    if clip.raw_fallback_ref is not None:
        info["raw_fallback_asset_id"] = clip.raw_fallback_ref.asset_id
    return info


def title_view(title: TitleClip) -> dict[str, Any]:
    """A compact view of one `TitleClip`."""
    return {
        "title_id": title.title_id,
        "timeline_frames": f"{title.timeline_in_frame}..{title.timeline_out_frame}",
        "text": title.text,
        "style_preset": title.style_preset,
        "position": title.position,
    }


def proposal_line(proposal: VideoProposal, digest: str) -> dict[str, Any]:
    """One proposal as its id, target, kind, and summary."""
    line: dict[str, Any] = {
        "id": proposal.id,
        "status": proposal.status,
        "kind": proposal.kind,
        "summary": proposal.summary,
        "digest": digest,
    }
    if proposal.track_id is not None:
        line["track_id"] = proposal.track_id
    return line


def bundle_view(bundle: TimelineContextBundle) -> dict[str, Any]:
    """Serialize a `TimelineContextBundle` concisely."""
    return bundle.model_dump(mode="json", exclude_none=True)


def audit_digest(findings: Sequence[VideoAuditFinding], *, passed: bool) -> dict[str, Any]:
    """An audit as its verdict, its first findings, and a count of every finding by kind."""
    ordered = sorted(findings, key=lambda f: (f.severity != "error", f.frame))
    shown = [
        {
            "code": f.code,
            "severity": f.severity,
            "frame": f.frame,
            "message": f.message,
        }
        for f in ordered[:AUDIT_FINDINGS_SHOWN]
    ]
    result: dict[str, Any] = {
        "passed": passed,
        "findings_total": len(findings),
        "findings": shown,
    }
    if len(findings) > len(shown):
        result["counts_by_code"] = dict(Counter(f.code for f in findings))
        result["not_shown"] = len(findings) - len(shown)
    return result
