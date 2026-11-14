"""Proposing, previewing, applying, and rejecting video timeline and manifest changes.

Every modification to an existing non-empty track or sequence manifest is staged as a
`VideoProposal` (`proposals/pNNN.yaml`) carrying the `target_digest` of the file it was
proposed against. Applying a proposal verifies both the proposal's own digest (when
`seen_digest` is given) and the target file's current digest on disk, refusing with
`VideoChangedError` if the underlying file changed since the proposal was created.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import ValidationError

from uclone_x_video.library import VideoChangedError, VideoError, VideoFile
from uclone_x_video.schemas import (
    ManifestOutline,
    TrackFile,
    VideoProposal,
    describe_invalid,
)
from uclone_x_video.work import MANIFEST_FILE, VideoWork, track_file_by_id

__all__ = [
    "DecidedIn",
    "apply_proposal",
    "preview_proposed_track",
    "propose_manifest_change",
    "propose_track_change",
    "reject_proposal",
]

DecidedIn = Literal["conversation", "video_view"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def preview_proposed_track(proposal: VideoProposal) -> TrackFile:
    """Return the validated `TrackFile` held by a `track_replace` proposal."""
    if proposal.kind != "track_replace" or proposal.proposed_track is None:
        raise VideoError(
            f"Proposal {proposal.id!r} does not carry a proposed track (kind={proposal.kind!r})."
        )
    return proposal.proposed_track


def propose_track_change(
    work: VideoWork,
    *,
    proposed_track: TrackFile,
    summary: str,
) -> tuple[VideoProposal, VideoFile]:
    """Create and save a pending `track_replace` proposal against `tracks/<track_id>.yaml`."""
    clean_summary = summary.strip()
    if not clean_summary:
        raise VideoError("Proposal summary must not be empty.")
    existing = work.read_track_by_id(proposed_track.track_id)
    target_digest = existing[1] if existing is not None else None
    pid = work.next_proposal_id()
    target = track_file_by_id(proposed_track.track_id)
    try:
        proposal = VideoProposal(
            id=pid,
            kind="track_replace",
            target_file=target,
            target_digest=target_digest,
            track_id=proposed_track.track_id,
            summary=clean_summary,
            proposed_track=proposed_track,
            status="pending",
            created_by=work.room_id,
            created_at=_now_iso(),
        )
    except ValidationError as exc:
        raise VideoError(describe_invalid(f"proposals/{pid}.yaml", exc)) from exc

    written = work.write_proposal(proposal, expected_digest=None)
    work.record_session(f"Proposed {pid}: {clean_summary}")
    return proposal, written


def propose_manifest_change(
    work: VideoWork,
    *,
    proposed_manifest: ManifestOutline,
    summary: str,
) -> tuple[VideoProposal, VideoFile]:
    """Create and save a pending `manifest_replace` proposal against `manifest.yaml`."""
    clean_summary = summary.strip()
    if not clean_summary:
        raise VideoError("Proposal summary must not be empty.")
    existing = work.read_manifest()
    target_digest = existing[1] if existing is not None else None
    pid = work.next_proposal_id()
    try:
        proposal = VideoProposal(
            id=pid,
            kind="manifest_replace",
            target_file=MANIFEST_FILE,
            target_digest=target_digest,
            summary=clean_summary,
            proposed_manifest=proposed_manifest,
            status="pending",
            created_by=work.room_id,
            created_at=_now_iso(),
        )
    except ValidationError as exc:
        raise VideoError(describe_invalid(f"proposals/{pid}.yaml", exc)) from exc

    written = work.write_proposal(proposal, expected_digest=None)
    work.record_session(f"Proposed {pid}: {clean_summary}")
    return proposal, written


def _load_pending(
    work: VideoWork,
    proposal_id: str,
    seen_digest: str | None,
) -> tuple[VideoProposal, str]:
    loaded = work.read_proposal(proposal_id)
    if loaded is None:
        raise VideoError(f"Proposal {proposal_id!r} was not found.")
    proposal, prop_digest = loaded
    if proposal.status != "pending":
        raise VideoError(
            f"Proposal {proposal_id!r} is already {proposal.status!r} and cannot be decided again."
        )
    if seen_digest is not None and prop_digest != seen_digest:
        raise VideoChangedError(
            f"Proposal {proposal_id!r} changed on disk since it was read "
            f"(expected {seen_digest[:12]}, found {prop_digest[:12]})."
        )
    return proposal, prop_digest


def apply_proposal(
    work: VideoWork,
    proposal_id: str,
    *,
    decided_in: DecidedIn,
    seen_digest: str | None = None,
) -> tuple[VideoProposal, VideoFile]:
    """Apply a pending proposal to its target file after verifying digests, then mark it approved."""
    proposal, prop_digest = _load_pending(work, proposal_id, seen_digest)

    if proposal.kind == "track_replace":
        if proposal.proposed_track is None:
            raise VideoError(f"Proposal {proposal_id!r} is missing proposed_track.")
        current = work.read_track_by_id(proposal.proposed_track.track_id)
        current_digest = current[1] if current is not None else None
        if current_digest != proposal.target_digest:
            raise VideoChangedError(
                f"Target track {proposal.target_file!r} changed on disk since proposal "
                f"{proposal_id!r} was created (expected {(proposal.target_digest or 'none')[:12]}, "
                f"found {(current_digest or 'none')[:12]})."
            )
        work.write_track(
            proposal.proposed_track,
            expected_digest=proposal.target_digest,
            summary=f"Applied {proposal.id}: {proposal.summary}",
        )
    elif proposal.kind == "manifest_replace":
        if proposal.proposed_manifest is None:
            raise VideoError(f"Proposal {proposal_id!r} is missing proposed_manifest.")
        current_m = work.read_manifest()
        current_digest = current_m[1] if current_m is not None else None
        if current_digest != proposal.target_digest:
            raise VideoChangedError(
                f"Target manifest {proposal.target_file!r} changed on disk since proposal "
                f"{proposal_id!r} was created."
            )
        work.write_manifest(
            proposal.proposed_manifest,
            expected_digest=proposal.target_digest,
        )

    approved = proposal.model_copy(
        update={
            "status": "approved",
            "decided_in": decided_in,
            "decided_at": _now_iso(),
        }
    )
    written_prop = work.write_proposal(approved, expected_digest=prop_digest)
    work.record_session(f"Approved proposal {proposal_id} in {decided_in}")
    return approved, written_prop


def reject_proposal(
    work: VideoWork,
    proposal_id: str,
    *,
    decided_in: DecidedIn,
    reason: str | None = None,
    seen_digest: str | None = None,
) -> tuple[VideoProposal, VideoFile]:
    """Reject a pending proposal and record `rejection_reason`."""
    proposal, prop_digest = _load_pending(work, proposal_id, seen_digest)
    clean_reason = reason.strip() if reason is not None and reason.strip() else None
    rejected = proposal.model_copy(
        update={
            "status": "rejected",
            "decided_in": decided_in,
            "decided_at": _now_iso(),
            "rejection_reason": clean_reason,
        }
    )
    written_prop = work.write_proposal(rejected, expected_digest=prop_digest)
    work.record_session(f"Rejected proposal {proposal_id} in {decided_in}")
    return rejected, written_prop
