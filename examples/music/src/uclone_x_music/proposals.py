"""Proposing, previewing, applying, and rejecting music score changes.

Every destructive or multi-bar modification to an existing track or instrument can be
staged as a `MusicProposal` (`proposals/pNNN.yaml`) carrying the `target_digest` of the
file it was proposed against. Applying a proposal verifies both the proposal's own digest
(when `seen_digest` is given) and the target file's current digest on disk, refusing with
`SongChangedError` if the underlying file changed since the proposal was created.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import ValidationError

from uclone_x_music.library import MusicError, SongChangedError, SongFile
from uclone_x_music.schemas import (
    InstrumentEntry,
    MusicFileError,
    MusicProposal,
    TrackBar,
    TrackFile,
    describe_invalid,
)
from uclone_x_music.work import SongWork, instrument_file, track_file

__all__ = [
    "DecidedIn",
    "apply_proposal",
    "preview_patched_track",
    "propose_instrument_change",
    "propose_track_patch",
    "reject_proposal",
]

DecidedIn = Literal["conversation", "music_view"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def preview_patched_track(track: TrackFile, proposal: MusicProposal) -> TrackFile:
    """Return a new `TrackFile` with `proposal.proposed_bars` spliced at `start_bar..end_bar`.

    All bars outside `start_bar..end_bar` are preserved untouched.
    """
    if proposal.kind == "track_replace":
        if proposal.proposed_track is None:
            raise MusicError(f"Proposal {proposal.id!r} has no proposed_track.")
        return proposal.proposed_track
    if proposal.kind != "track_patch":
        raise MusicError(
            f"Proposal {proposal.id!r} has kind {proposal.kind!r}; expected 'track_patch'."
        )
    if proposal.start_bar is None or proposal.end_bar is None:
        raise MusicError(f"Proposal {proposal.id!r} is missing start_bar or end_bar.")
    total_bars = len(track.bars)
    if (
        proposal.start_bar < 1
        or proposal.end_bar < proposal.start_bar
        or proposal.end_bar > total_bars
    ):
        raise MusicError(
            f"Proposal {proposal.id!r} bar range {proposal.start_bar}..{proposal.end_bar} "
            f"is outside track {track.track_id!r} (1..{total_bars})."
        )
    before = list(track.bars[: proposal.start_bar - 1])
    middle = list(proposal.proposed_bars)
    after = list(track.bars[proposal.end_bar :])
    spliced = [*before, *middle, *after]
    try:
        return TrackFile(
            track_id=track.track_id,
            section_id=track.section_id,
            instrument_id=track.instrument_id,
            time_signature=track.time_signature,
            groove_swing=track.groove_swing,
            bars=spliced,
        )
    except ValidationError as exc:
        raise MusicFileError(describe_invalid(proposal.target_file, exc)) from exc


def propose_track_patch(
    work: SongWork,
    *,
    section_id: str,
    instrument_id: str,
    start_bar: int,
    end_bar: int,
    proposed_bars: Sequence[TrackBar],
    summary: str,
) -> tuple[MusicProposal, SongFile]:
    """Create and save a pending `track_patch` proposal against an existing track."""
    clean_summary = summary.strip()
    if not clean_summary:
        raise MusicError("Proposal summary must not be empty.")
    existing = work.read_track(section_id, instrument_id)
    if existing is None:
        raise MusicError(
            f"Track for section {section_id!r} and instrument {instrument_id!r} does not exist yet."
        )
    track, digest = existing
    total_bars = len(track.bars)
    if start_bar < 1 or end_bar < start_bar or end_bar > total_bars:
        raise MusicError(
            f"Bar range {start_bar}..{end_bar} is outside track {track.track_id!r} (1..{total_bars})."
        )
    pid = work.next_proposal_id()
    target = track_file(section_id, instrument_id)
    try:
        proposal = MusicProposal(
            id=pid,
            kind="track_patch",
            target_file=target,
            target_digest=digest,
            section_id=section_id,
            instrument_id=instrument_id,
            start_bar=start_bar,
            end_bar=end_bar,
            summary=clean_summary,
            proposed_bars=list(proposed_bars),
            status="pending",
            created_by=work.room_id,
            created_at=_now_iso(),
        )
    except ValidationError as exc:
        raise MusicError(describe_invalid(f"proposals/{pid}.yaml", exc)) from exc

    # Validate that applying the patch produces a valid TrackFile before saving the proposal.
    preview_patched_track(track, proposal)
    written = work.write_proposal(proposal, expected_digest=None)
    work.record_session(f"Proposed {pid}: {clean_summary}")
    return proposal, written


def propose_instrument_change(
    work: SongWork,
    *,
    instrument: InstrumentEntry,
    summary: str,
) -> tuple[MusicProposal, SongFile]:
    """Create and save a pending `instrument_change` proposal."""
    clean_summary = summary.strip()
    if not clean_summary:
        raise MusicError("Proposal summary must not be empty.")
    existing = work.read_instrument(instrument.id)
    target_digest = existing[1] if existing is not None else None
    pid = work.next_proposal_id()
    target = instrument_file(instrument.id)
    try:
        proposal = MusicProposal(
            id=pid,
            kind="instrument_change",
            target_file=target,
            target_digest=target_digest,
            instrument_id=instrument.id,
            summary=clean_summary,
            proposed_instrument=instrument,
            status="pending",
            created_by=work.room_id,
            created_at=_now_iso(),
        )
    except ValidationError as exc:
        raise MusicError(describe_invalid(f"proposals/{pid}.yaml", exc)) from exc

    written = work.write_proposal(proposal, expected_digest=None)
    work.record_session(f"Proposed {pid}: {clean_summary}")
    return proposal, written


def _load_pending(
    work: SongWork,
    proposal_id: str,
    seen_digest: str | None,
) -> tuple[MusicProposal, str]:
    loaded = work.read_proposal(proposal_id)
    if loaded is None:
        raise MusicError(f"Proposal {proposal_id!r} was not found.")
    proposal, prop_digest = loaded
    if proposal.status != "pending":
        raise MusicError(
            f"Proposal {proposal_id!r} is already {proposal.status!r} and cannot be decided again."
        )
    if seen_digest is not None and prop_digest != seen_digest:
        raise SongChangedError(
            f"Proposal {proposal_id!r} changed on disk since it was read "
            f"(expected {seen_digest[:12]}, found {prop_digest[:12]})."
        )
    return proposal, prop_digest


def apply_proposal(
    work: SongWork,
    proposal_id: str,
    *,
    decided_in: DecidedIn,
    seen_digest: str | None = None,
) -> tuple[MusicProposal, SongFile]:
    """Apply a pending proposal to its target file after verifying digests, then mark it approved."""
    proposal, prop_digest = _load_pending(work, proposal_id, seen_digest)

    if proposal.kind == "track_patch":
        if proposal.section_id is None or proposal.instrument_id is None:
            raise MusicError(f"Proposal {proposal_id!r} is missing section_id or instrument_id.")
        current = work.read_track(proposal.section_id, proposal.instrument_id)
        if current is None:
            raise SongChangedError(
                f"Target track {proposal.target_file!r} no longer exists on disk."
            )
        track, current_digest = current
        if proposal.target_digest is not None and current_digest != proposal.target_digest:
            raise SongChangedError(
                f"Target track {proposal.target_file!r} changed on disk since proposal "
                f"{proposal_id!r} was created (expected {proposal.target_digest[:12]}, "
                f"found {current_digest[:12]})."
            )
        patched = preview_patched_track(track, proposal)
        work.write_track(
            patched,
            expected_digest=current_digest,
            summary=f"Applied {proposal.id}: {proposal.summary}",
        )
    elif proposal.kind == "track_replace":
        if proposal.proposed_track is None:
            raise MusicError(f"Proposal {proposal_id!r} is missing proposed_track.")
        current = work.read_track_by_id(proposal.proposed_track.track_id)
        current_digest = current[1] if current is not None else None
        if current_digest != proposal.target_digest:
            raise SongChangedError(
                f"Target track {proposal.target_file!r} changed on disk since proposal "
                f"{proposal_id!r} was created."
            )
        work.write_track(
            proposal.proposed_track,
            expected_digest=proposal.target_digest,
            summary=f"Applied {proposal.id}: {proposal.summary}",
        )
    elif proposal.kind == "instrument_change":
        if proposal.proposed_instrument is None:
            raise MusicError(f"Proposal {proposal_id!r} is missing proposed_instrument.")
        current_inst = work.read_instrument(proposal.proposed_instrument.id)
        current_digest = current_inst[1] if current_inst is not None else None
        if current_digest != proposal.target_digest:
            raise SongChangedError(
                f"Target instrument {proposal.target_file!r} changed on disk since proposal "
                f"{proposal_id!r} was created."
            )
        work.write_instrument(
            proposal.proposed_instrument,
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
    work: SongWork,
    proposal_id: str,
    *,
    decided_in: DecidedIn,
    reason: str | None = None,
    seen_digest: str | None = None,
) -> tuple[MusicProposal, SongFile]:
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
