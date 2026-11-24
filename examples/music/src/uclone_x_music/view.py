"""A song's interactive score view: outline, tracks, notes, and pending proposals.

`MusicView` reads one song whole for the interactive sheet music and multi-track studio view,
and applies user-confirmed proposal decisions, direct score edits, and orchestra mix adjustments
through the song's writing lease.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import RoomNotFoundError, UnreadableRoomRecordError
from uclone_x_music.arranger import (
    ArrangementIntent,
    append_section_bars,
    delete_section_bars,
    duplicate_section_bars,
    insert_section_bars,
)
from uclone_x_music.library import (
    MusicError,
    SongLibrary,
    SongRecord,
    UnknownSongError,
)
from uclone_x_music.schemas import (
    Articulation,
    BarNoteEvent,
    InstrumentEntry,
    MusicProposal,
    ProposalKind,
    ScoreOutline,
    ScoreSection,
    StyleAxioms,
    TrackBar,
    TrackFile,
    beats_per_bar_for,
    format_note_token,
    normalize_bar_notes,
    pitch_to_name,
    resolve_track_bars,
    validate_track_against_section,
)
from uclone_x_music.work import SongWork, track_id_for

__all__ = [
    "ComposerBusyError",
    "DirectEditAction",
    "MusicNoteCode",
    "MusicRefusalCode",
    "MusicView",
    "ShownBarView",
    "ShownNoteView",
    "ShownProposalView",
    "ShownSong",
    "ShownTrackView",
    "SongNotFoundError",
    "SongNotWritableError",
    "StructureEditAction",
]

_MODEL = ConfigDict(frozen=True, extra="forbid", strict=True)

MusicNoteCode = Literal["no_writer", "writer_gone"]
MusicRefusalCode = Literal["song_gone", "no_writer", "writer_gone", "writer_busy"]
DirectEditAction = Literal[
    "clear_bars",
    "mute_bars",
    "transpose_bars",
    "set_bar_notes",
    "revert",
]
StructureEditAction = Literal[
    "insert_bars",
    "append_bars",
    "delete_bars",
    "duplicate_bars",
]


class SongNotFoundError(MusicError):
    """The song is not in the library."""

    code: MusicRefusalCode = "song_gone"


class SongNotWritableError(MusicError):
    """No existing conversation holds the song's writing lease."""

    def __init__(self, message: str, code: MusicRefusalCode = "no_writer") -> None:
        super().__init__(message)
        self.code: MusicRefusalCode = code


class ComposerBusyError(MusicError):
    """The conversation composing the song is answering right now."""

    code: MusicRefusalCode = "writer_busy"


class ShownNoteView(BaseModel):
    """A compiled note event positioned in bar/beat and seconds, with concert and written pitch."""

    model_config = _MODEL

    track_id: str
    section_id: str
    instrument_id: str
    bar: int = Field(ge=1)
    global_bar: int = Field(ge=1)
    beat: float = Field(ge=1.0)
    duration: float = Field(gt=0.0)
    concert_midi: int = Field(ge=-1, le=127)
    written_pitch: str
    concert_pitch: str
    velocity: int = Field(ge=0, le=127)
    articulation: Articulation
    start_seconds: float = Field(ge=0.0)
    duration_seconds: float = Field(gt=0.0)


class ShownBarView(BaseModel):
    """A compiled bar in the global timeline of the song."""

    model_config = _MODEL

    global_bar: int = Field(ge=1)
    section_id: str
    section_title: str
    section_bar: int = Field(ge=1)
    chord: str
    bpm: int = Field(ge=20, le=300)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)


class ShownTrackView(BaseModel):
    """A track's current bars, digest, and undo availability."""

    model_config = _MODEL

    track_id: str
    section_id: str
    instrument_id: str
    digest: str
    groove_swing: float
    bars: list[TrackBar]
    can_revert: bool


class ShownProposalView(BaseModel):
    """A pending proposal with original vs proposed bars for A/B audition and diff."""

    model_config = _MODEL

    id: str
    kind: ProposalKind
    summary: str
    section_id: str | None = None
    instrument_id: str | None = None
    start_bar: int | None = None
    end_bar: int | None = None
    seen_digest: str
    original_bars: list[TrackBar] = Field(default_factory=list[TrackBar])
    proposed_bars: list[TrackBar] = Field(default_factory=list[TrackBar])


class ShownSong(BaseModel):
    """The complete song state for MusicView."""

    model_config = _MODEL

    song_id: str
    title: str
    genre: str
    style_axioms: StyleAxioms
    writable: bool
    holder: str | None
    decide_code: str | None
    decide_note: str | None = None
    tempo: int
    time_signature: str
    key_signature: str
    total_bars: int
    duration_seconds: float
    sections: list[ScoreSection]
    instruments: list[InstrumentEntry]
    tracks: list[ShownTrackView]
    bars: list[ShownBarView]
    notes: list[ShownNoteView]
    proposals: list[ShownProposalView]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _written_pitch_for(concert_midi: int, transposition_semitones: int) -> str:
    """Compute written pitch for a transposing instrument.

    `transposition_semitones` is `concert_midi - written_midi` (e.g. Bb Trumpet has `-2`,
    so a concert Bb4 sounds 2 semitones lower than written C5: `written_midi = concert_midi - (-2)`).
    """
    if concert_midi < 0:
        return "REST"
    written_midi = max(0, min(127, concert_midi - transposition_semitones))
    return pitch_to_name(written_midi)


def _bar_bpm(default_tempo: int, section: ScoreSection, section_bar: int) -> int:
    bpm = default_tempo
    for change in section.tempo_map:
        if change.bar <= section_bar:
            bpm = change.bpm
    return bpm


def _pad_track_bars_to(
    track: TrackFile,
    target_bars: int,
    chords: list[str],
    patch_by_bar: dict[int, TrackBar] | None = None,
) -> TrackFile:
    """Return `track` with `bars` reconciled to `1..target_bars` and `patch_by_bar` applied."""
    patches = patch_by_bar or {}
    source_bars = list(track.bars)
    bpb = beats_per_bar_for(track.time_signature)
    rest_dur = f"{bpb:g}" if bpb % 1 != 0 else f"{int(bpb)}.0"
    fallback_bar = (
        source_bars[0]
        if source_bars
        else TrackBar(bar=1, chord=None, notes=[f"1.0:REST:{rest_dur}"])
    )
    merged: list[TrackBar] = []
    for bar_num in range(1, target_bars + 1):
        if bar_num in patches:
            patched_bar = patches[bar_num]
            chord = (
                patched_bar.chord
                if patched_bar.chord
                else (chords[bar_num - 1] if bar_num - 1 < len(chords) else "")
            )
            merged.append(patched_bar.model_copy(update={"bar": bar_num, "chord": chord}))
        elif bar_num <= len(source_bars):
            existing_bar = source_bars[bar_num - 1]
            chord = (
                existing_bar.chord
                if existing_bar.chord
                else (chords[bar_num - 1] if bar_num - 1 < len(chords) else "")
            )
            merged.append(existing_bar.model_copy(update={"bar": bar_num, "chord": chord}))
        else:
            template = (
                source_bars[(bar_num - 1) % len(source_bars)] if source_bars else fallback_bar
            )
            chord = chords[bar_num - 1] if bar_num - 1 < len(chords) else template.chord
            merged.append(
                TrackBar(
                    bar=bar_num,
                    chord=chord,
                    notes=list(template.notes),
                    repeat_bar=None,
                    transpose_semitones=template.transpose_semitones,
                    muted=template.muted,
                )
            )
    return TrackFile(
        track_id=track.track_id,
        section_id=track.section_id,
        instrument_id=track.instrument_id,
        time_signature=track.time_signature,
        groove_swing=track.groove_swing,
        bars=merged,
    )


def _reconcile_section_and_track_bars(
    work: SongWork,
    section_id: str,
    track: TrackFile,
    patch_bars: list[TrackBar] | None = None,
) -> TrackFile:
    """Ensure `score.yaml` section `section_id`, `track`, and sibling tracks share the same bar count."""
    patch_by_bar = {b.bar: b for b in (patch_bars or [])}
    max_patch_bar = max(patch_by_bar.keys(), default=0)
    score_pair = work.read_score()
    outline = score_pair[0] if score_pair is not None else None
    score_digest = score_pair[1] if score_pair is not None else None
    sec = (
        next((s for s in outline.sections if s.id == section_id), None)
        if outline is not None
        else None
    )

    target_bars = max(
        len(track.bars),
        max_patch_bar,
        sec.bars if sec is not None else 0,
    )
    base_chords = (
        list(sec.chords)
        if sec is not None and sec.chords
        else [b.chord for b in track.bars if b.chord]
    )
    chords = list(sec.chords) if sec is not None else []
    # A chordless song stays chordless: no progression is invented to fill the score.
    while base_chords and len(chords) < target_bars:
        bar_num = len(chords) + 1
        patch_chord = patch_by_bar[bar_num].chord if bar_num in patch_by_bar else ""
        chords.append(patch_chord or base_chords[(bar_num - 1) % len(base_chords)])

    if (
        outline is not None
        and sec is not None
        and (sec.bars < target_bars or len(sec.chords) < target_bars)
    ):
        updated_sec = ScoreSection(
            id=sec.id,
            title=sec.title,
            kind=sec.kind,
            bars=target_bars,
            chords=chords[:target_bars] if len(chords) >= target_bars else [],
            tempo_map=list(sec.tempo_map),
        )
        new_sections = [updated_sec if s.id == section_id else s for s in outline.sections]
        outline = outline.model_copy(update={"sections": new_sections})
        work.write_score(outline, expected_digest=score_digest)

    reconciled_track = _pad_track_bars_to(track, target_bars, chords, patch_by_bar)
    if outline is not None and reconciled_track.time_signature != outline.time_signature:
        reconciled_track = reconciled_track.model_copy(
            update={"time_signature": outline.time_signature}
        )

    for other_track, other_digest in work.list_tracks():
        if (
            other_track.section_id == section_id
            and other_track.track_id != track.track_id
            and len(other_track.bars) < target_bars
        ):
            padded_other = _pad_track_bars_to(other_track, target_bars, chords)
            work.write_track(
                padded_other,
                expected_digest=other_digest,
                summary=f"Extended {other_track.track_id} to {target_bars} bars",
            )

    return reconciled_track


class MusicView:
    """Reads a song whole and executes person-initiated decisions and direct score edits."""

    def __init__(
        self,
        workspace_dir: Path,
        room_service: Any = None,
        *,
        turn_busy: Callable[[str], bool] = lambda _room_id: False,
    ) -> None:
        self._library = SongLibrary(workspace_dir.resolve())
        self._rooms = room_service
        self._turn_busy = turn_busy

    def _load_record(self, song_id: str) -> SongRecord:
        try:
            return self._library.load(song_id)
        except UnknownSongError as exc:
            raise SongNotFoundError(
                f"There is no song called '{song_id}' any more. Refresh the list to see the songs in this workspace."
            ) from exc

    def _room_status(self, room_id: str) -> tuple[bool, str | None]:
        if self._rooms is None:
            return True, room_id
        try:
            room = self._rooms.get(room_id)
            return True, getattr(room, "title", None)
        except RoomNotFoundError:
            return False, None
        except UnreadableRoomRecordError:
            return True, None

    def _check_writable_state(
        self, record: SongRecord
    ) -> tuple[bool, str | None, MusicNoteCode | None, str | None]:
        if record.lease is None or not record.lease.holder:
            return (
                False,
                None,
                "no_writer",
                f"No conversation is composing “{record.title}”, so changes cannot be saved. Open the song in a conversation first.",
            )
        holder = record.lease.holder
        exists, _ = self._room_status(holder)
        if not exists:
            return (
                False,
                holder,
                "writer_gone",
                f"The conversation that was composing “{record.title}” no longer exists. Open the song in a conversation first.",
            )
        return True, holder, None, None

    def _require_decider(self, song_id: str, room_id: str | None = None) -> tuple[SongWork, str]:
        """The song's writer, for a decision the person made in the score view.

        When no conversation holds the song any more (`no_writer`, `writer_gone`) and the
        person decided from the view docked beside conversation `room_id`, that conversation
        takes the song over first: the person's click is the decision, and it is applied in
        the conversation they are looking at. A song another live conversation holds is never
        taken from it.
        """
        record = self._load_record(song_id)
        writable, holder, code, message = self._check_writable_state(record)
        if (
            not writable
            and code in ("no_writer", "writer_gone")
            and room_id is not None
            and room_id.strip()
            and self._room_status(room_id)[0]
        ):
            # A docked conversation that is answering cannot take the song: refuse before
            # anything moves, so the lease stays where it was.
            self._refuse_if_busy(room_id)
            # Taken only if the gone (or absent) writer still holds it, checked and written
            # under one lock: a conversation that took it meanwhile keeps it.
            self._library.take_over_if_held_by(song_id, room_id, expected_holder=holder)
            record = self._load_record(song_id)
            writable, holder, code, message = self._check_writable_state(record)
        if not writable or holder is None:
            raise SongNotWritableError(
                message or f"Song '{song_id}' has no active writer conversation.",
                code or "no_writer",
            )
        self._refuse_if_busy(holder)
        return SongWork(self._library, song_id, holder), holder

    def _refuse_if_busy(self, room_id: str) -> None:
        if self._turn_busy(room_id):
            _, title = self._room_status(room_id)
            label = f"“{title}”" if title else f"'{room_id}'"
            raise ComposerBusyError(
                f"The conversation {label} is answering right now, so the change was not saved. Wait for the turn to finish, then try again."
            )

    def show(self, song_id: str) -> ShownSong:
        """Compile the full `ShownSong` view model for `song_id`."""
        record = self._load_record(song_id)
        writable, holder, decide_code, decide_note = self._check_writable_state(record)
        work = SongWork(self._library, song_id, holder)

        score_pair = work.read_score()
        outline = score_pair[0] if score_pair is not None else ScoreOutline()
        bpb = beats_per_bar_for(outline.time_signature)

        instruments_with_digest = work.list_instruments()
        instruments = [inst for inst, _ in instruments_with_digest]

        tracks_with_digest = work.list_tracks()
        track_map: dict[tuple[str, str], tuple[TrackFile, str]] = {}
        shown_tracks: list[ShownTrackView] = []
        for track, digest in tracks_with_digest:
            track_map[(track.section_id, track.instrument_id)] = (track, digest)
            rev_log, _ = work.read_revision_log(track.track_id)
            shown_tracks.append(
                ShownTrackView(
                    track_id=track.track_id,
                    section_id=track.section_id,
                    instrument_id=track.instrument_id,
                    digest=digest,
                    groove_swing=track.groove_swing,
                    bars=list(track.bars),
                    can_revert=len(rev_log.revisions) >= 2,
                )
            )

        bars: list[ShownBarView] = []
        notes: list[ShownNoteView] = []
        current_time = 0.0
        global_bar_idx = 0

        for section in outline.sections:
            resolved_by_inst: dict[str, tuple[TrackFile, dict[int, list[BarNoteEvent]]]] = {}
            for inst in instruments:
                found_track = track_map.get((section.id, inst.id))
                if found_track is not None:
                    t_file = found_track[0]
                    if len(t_file.bars) < section.bars:
                        t_file = _pad_track_bars_to(t_file, section.bars, list(section.chords))
                    resolved_by_inst[inst.id] = (t_file, resolve_track_bars(t_file))

            for sec_bar in range(1, section.bars + 1):
                global_bar_idx += 1
                bpm = _bar_bpm(outline.tempo, section, sec_bar)
                sec_per_beat = 60.0 / float(bpm)
                bar_duration_sec = bpb * sec_per_beat
                bar_start_sec = round(current_time, 6)
                bar_end_sec = round(current_time + bar_duration_sec, 6)
                chord = section.chords[sec_bar - 1] if sec_bar - 1 < len(section.chords) else ""

                bars.append(
                    ShownBarView(
                        global_bar=global_bar_idx,
                        section_id=section.id,
                        section_title=section.title,
                        section_bar=sec_bar,
                        chord=chord,
                        bpm=bpm,
                        start_seconds=bar_start_sec,
                        end_seconds=bar_end_sec,
                    )
                )

                for inst in instruments:
                    if inst.id not in resolved_by_inst:
                        continue
                    t_file, resolved_bars = resolved_by_inst[inst.id]
                    events = resolved_bars.get(sec_bar, [])
                    for ev in events:
                        note_start = round(bar_start_sec + (ev.beat - 1.0) * sec_per_beat, 6)
                        note_dur = round(ev.duration * sec_per_beat, 6)
                        concert_pitch = "REST" if ev.midi == -1 else pitch_to_name(ev.midi)
                        written_pitch = _written_pitch_for(ev.midi, inst.transposition_semitones)
                        notes.append(
                            ShownNoteView(
                                track_id=t_file.track_id,
                                section_id=section.id,
                                instrument_id=inst.id,
                                bar=sec_bar,
                                global_bar=global_bar_idx,
                                beat=ev.beat,
                                duration=ev.duration,
                                concert_midi=ev.midi,
                                written_pitch=written_pitch,
                                concert_pitch=concert_pitch,
                                velocity=ev.velocity,
                                articulation=ev.articulation,
                                start_seconds=note_start,
                                duration_seconds=note_dur,
                            )
                        )

                current_time += bar_duration_sec

        proposals_raw = work.list_proposals(status="pending")
        shown_proposals: list[ShownProposalView] = []
        for prop, prop_digest in proposals_raw:
            orig_bars: list[TrackBar] = []
            prop_bars: list[TrackBar] = []
            sec_id = prop.section_id
            inst_id = prop.instrument_id
            start_b = prop.start_bar
            end_b = prop.end_bar

            if prop.kind == "track_patch":
                if sec_id and inst_id and (sec_id, inst_id) in track_map:
                    existing_track = track_map[(sec_id, inst_id)][0]
                    if start_b is not None and end_b is not None:
                        orig_bars = [b for b in existing_track.bars if start_b <= b.bar <= end_b]
                prop_bars = list(prop.proposed_bars)
            elif prop.kind == "track_replace" and prop.proposed_track is not None:
                pt = prop.proposed_track
                sec_id = sec_id or pt.section_id
                inst_id = inst_id or pt.instrument_id
                start_b = 1
                end_b = len(pt.bars)
                if (pt.section_id, pt.instrument_id) in track_map:
                    orig_bars = list(track_map[(pt.section_id, pt.instrument_id)][0].bars)
                prop_bars = list(pt.bars)
            elif prop.kind == "instrument_change" and prop.proposed_instrument is not None:
                inst_id = inst_id or prop.proposed_instrument.id

            shown_proposals.append(
                ShownProposalView(
                    id=prop.id,
                    kind=prop.kind,
                    summary=prop.summary,
                    section_id=sec_id,
                    instrument_id=inst_id,
                    start_bar=start_b,
                    end_bar=end_b,
                    seen_digest=prop_digest,
                    original_bars=orig_bars,
                    proposed_bars=prop_bars,
                )
            )

        return ShownSong(
            song_id=record.id,
            title=record.title,
            genre=record.genre,
            style_axioms=record.style_axioms,
            writable=writable,
            holder=holder,
            decide_code=decide_code,
            decide_note=decide_note,
            tempo=outline.tempo,
            time_signature=outline.time_signature,
            key_signature=outline.key_signature,
            total_bars=global_bar_idx,
            duration_seconds=round(current_time, 4),
            sections=list(outline.sections),
            instruments=instruments,
            tracks=shown_tracks,
            bars=bars,
            notes=notes,
            proposals=shown_proposals,
        )

    def approve(
        self,
        song_id: str,
        proposal_id: str,
        *,
        seen_digest: str,
        room_id: str | None = None,
    ) -> ShownSong:
        """Approve pending `proposal_id` and apply its changes to the target track or instrument."""
        work, _ = self._require_decider(song_id, room_id)
        loaded = work.read_proposal(proposal_id)
        if loaded is None:
            raise MusicError(f"Proposal '{proposal_id}' does not exist in song '{song_id}'.")
        proposal, current_digest = loaded
        if current_digest != seen_digest:
            raise MusicError(
                f"Proposal '{proposal_id}' changed after it was shown, so nothing was applied. Refresh and review it again."
            )
        if proposal.status != "pending":
            raise MusicError(
                f"Proposal '{proposal_id}' was already {proposal.status}, so nothing was changed."
            )

        self._apply_proposal_target(work, proposal)

        updated_proposal = proposal.model_copy(
            update={
                "status": "approved",
                "decided_in": "music_view",
                "decided_at": _now_iso(),
            }
        )
        work.write_proposal(updated_proposal, expected_digest=current_digest)
        return self.show(song_id)

    def _apply_proposal_target(self, work: SongWork, proposal: MusicProposal) -> None:
        if proposal.kind == "track_patch":
            assert proposal.section_id is not None
            assert proposal.instrument_id is not None
            track_loaded = work.read_track(proposal.section_id, proposal.instrument_id)
            if track_loaded is None:
                raise MusicError(
                    f"Target track for section '{proposal.section_id}' and instrument '{proposal.instrument_id}' does not exist."
                )
            track, track_digest = track_loaded
            updated_track = _reconcile_section_and_track_bars(
                work,
                track.section_id,
                track,
                list(proposal.proposed_bars),
            )
            score_pair = work.read_score()
            if score_pair is not None:
                outline = score_pair[0]
                sec_map = {s.id: s for s in outline.sections}
                if track.section_id in sec_map:
                    validate_track_against_section(
                        updated_track, sec_map[track.section_id], outline
                    )
            written_track = work.write_track(
                updated_track,
                expected_digest=track_digest,
                summary=f"Approved proposal {proposal.id}: {proposal.summary}",
            )
            for other_prop, other_digest in work.list_proposals(status="pending"):
                if (
                    other_prop.id != proposal.id
                    and other_prop.target_file == proposal.target_file
                    and other_prop.target_digest == track_digest
                ):
                    work.write_proposal(
                        other_prop.model_copy(update={"target_digest": written_track.digest}),
                        expected_digest=other_digest,
                    )
        elif proposal.kind == "track_replace":
            assert proposal.proposed_track is not None
            pt = proposal.proposed_track
            existing = work.read_track(pt.section_id, pt.instrument_id)
            existing_digest = existing[1] if existing is not None else None
            pt = _reconcile_section_and_track_bars(work, pt.section_id, pt)
            score_pair = work.read_score()
            if score_pair is not None:
                outline = score_pair[0]
                sec_map = {s.id: s for s in outline.sections}
                if pt.section_id in sec_map:
                    validate_track_against_section(pt, sec_map[pt.section_id], outline)
            work.write_track(
                pt,
                expected_digest=existing_digest,
                summary=f"Approved proposal {proposal.id}: {proposal.summary}",
            )
        elif proposal.kind == "instrument_change":
            assert proposal.proposed_instrument is not None
            pi = proposal.proposed_instrument
            existing_inst = work.read_instrument(pi.id)
            existing_digest = existing_inst[1] if existing_inst is not None else None
            if proposal.target_digest is not None and existing_digest != proposal.target_digest:
                raise MusicError(
                    f"Instrument '{pi.id}' changed after proposal '{proposal.id}' was created. Reject it and propose again."
                )
            work.write_instrument(pi, expected_digest=existing_digest)

    def reject(
        self,
        song_id: str,
        proposal_id: str,
        *,
        seen_digest: str,
        reason: str | None = None,
        room_id: str | None = None,
    ) -> ShownSong:
        """Mark pending `proposal_id` rejected."""
        work, _ = self._require_decider(song_id, room_id)
        loaded = work.read_proposal(proposal_id)
        if loaded is None:
            raise MusicError(f"Proposal '{proposal_id}' does not exist in song '{song_id}'.")
        proposal, current_digest = loaded
        if current_digest != seen_digest:
            raise MusicError(
                f"Proposal '{proposal_id}' changed after it was shown, so nothing was changed. Refresh and review it again."
            )
        if proposal.status != "pending":
            raise MusicError(
                f"Proposal '{proposal_id}' was already {proposal.status}, so nothing was changed."
            )
        clean_reason = reason.strip() if reason and reason.strip() else None
        updated_proposal = proposal.model_copy(
            update={
                "status": "rejected",
                "decided_in": "music_view",
                "decided_at": _now_iso(),
                "rejection_reason": clean_reason,
            }
        )
        work.write_proposal(updated_proposal, expected_digest=current_digest)
        return self.show(song_id)

    def approve_all(self, song_id: str, *, room_id: str | None = None) -> ShownSong:
        """Approve all pending proposals in `song_id` in order and return the updated `ShownSong`."""
        work, _ = self._require_decider(song_id, room_id)
        pending = work.list_proposals(status="pending")
        for proposal, _ in pending:
            reloaded = work.read_proposal(proposal.id)
            if reloaded is None or reloaded[0].status != "pending":
                continue
            prop_obj, prop_digest = reloaded
            self._apply_proposal_target(work, prop_obj)
            updated_proposal = prop_obj.model_copy(
                update={
                    "status": "approved",
                    "decided_in": "music_view",
                    "decided_at": _now_iso(),
                }
            )
            work.write_proposal(updated_proposal, expected_digest=prop_digest)
        return self.show(song_id)

    def edit_structure(
        self,
        song_id: str,
        *,
        action: StructureEditAction,
        section_id: str | None = None,
        at_bar: int = 1,
        count: int = 2,
        start_bar: int = 1,
        end_bar: int = 1,
        chords: list[str] | None = None,
        intent: ArrangementIntent | None = None,
    ) -> ShownSong:
        """Insert, append, delete, or duplicate bars across all tracks of `section_id`."""
        work, _ = self._require_decider(song_id)
        score_pair = work.read_score()
        default_sec = (
            score_pair[0].sections[0].id
            if score_pair is not None and score_pair[0].sections
            else "theme_a"
        )
        sec_id = section_id.strip() if section_id and section_id.strip() else default_sec
        if action == "insert_bars":
            insert_section_bars(
                work,
                sec_id,
                at_bar=at_bar,
                count=count,
                chords=chords,
                intent=intent or ("gentle_intro" if at_bar == 1 else "continuation"),
            )
        elif action == "append_bars":
            append_section_bars(
                work,
                sec_id,
                count=count,
                chords=chords,
                intent=intent or "continuation",
            )
        elif action == "delete_bars":
            delete_section_bars(work, sec_id, start_bar=start_bar, end_bar=end_bar)
        elif action == "duplicate_bars":
            duplicate_section_bars(
                work,
                sec_id,
                start_bar=start_bar,
                end_bar=end_bar,
                insert_at=at_bar if at_bar >= 1 else None,
            )
        return self.show(song_id)

    def direct_edit_track(
        self,
        song_id: str,
        track_id: str,
        *,
        action: DirectEditAction,
        start_bar: int = 1,
        end_bar: int = 1,
        semitones: int = 0,
        muted: bool = True,
        notes: list[str] | None = None,
        chord: str | None = None,
        seen_digest: str,
    ) -> ShownSong:
        """Execute a direct sheet-music manipulation on `track_id` in `MusicView`."""
        work, _ = self._require_decider(song_id)
        if action == "revert":
            work.revert_track(track_id, expected_digest=seen_digest)
            return self.show(song_id)

        loaded = work.read_track_by_id(track_id)
        if loaded is None:
            raise MusicError(f"Track '{track_id}' does not exist in song '{song_id}'.")
        track, current_digest = loaded
        if current_digest != seen_digest:
            raise MusicError(
                f"Track '{track_id}' changed on disk since it was shown. Refresh before editing."
            )
        score_pair = work.read_score()
        if score_pair is not None:
            sec = next((s for s in score_pair[0].sections if s.id == track.section_id), None)
            if sec is not None and len(track.bars) < sec.bars:
                track = _pad_track_bars_to(track, sec.bars, list(sec.chords))
        if start_bar < 1 or end_bar < start_bar or end_bar > len(track.bars):
            raise MusicError(
                f"Bar range {start_bar}..{end_bar} is outside track '{track_id}' (1..{len(track.bars)})."
            )

        bpb = beats_per_bar_for(track.time_signature)
        rest_dur = f"{bpb:g}" if bpb % 1 != 0 else f"{int(bpb)}.0"
        rest_token = f"1.0:REST:{rest_dur}"
        resolved_events = resolve_track_bars(track)
        clean_chord = chord.strip() if chord and chord.strip() else None

        new_bars: list[TrackBar] = []
        for bar in track.bars:
            if not (start_bar <= bar.bar <= end_bar):
                new_bars.append(bar)
                continue

            if action == "clear_bars":
                new_bars.append(
                    TrackBar(
                        bar=bar.bar,
                        chord=bar.chord,
                        notes=[rest_token],
                        repeat_bar=None,
                        transpose_semitones=0,
                        muted=False,
                    )
                )
            elif action == "mute_bars":
                new_bars.append(bar.model_copy(update={"muted": muted}))
            elif action == "set_bar_notes":
                norm_tokens = (
                    normalize_bar_notes(notes, track.time_signature)
                    if notes is not None
                    else list(bar.notes)
                )
                new_bars.append(
                    TrackBar(
                        bar=bar.bar,
                        chord=clean_chord or bar.chord,
                        notes=norm_tokens,
                        repeat_bar=None,
                        transpose_semitones=0,
                        muted=False,
                    )
                )
            elif action == "transpose_bars":
                if bar.repeat_bar is not None:
                    events = resolved_events[bar.bar]
                    shifted_tokens: list[str] = []
                    for ev in events:
                        if ev.midi == -1:
                            shifted_tokens.append(format_note_token(ev))
                        else:
                            shifted_midi = max(0, min(127, ev.midi + semitones))
                            shifted_ev = BarNoteEvent(
                                beat=ev.beat,
                                pitch=pitch_to_name(shifted_midi),
                                midi=shifted_midi,
                                duration=ev.duration,
                                velocity=ev.velocity,
                                articulation=ev.articulation,
                            )
                            shifted_tokens.append(format_note_token(shifted_ev))
                    new_bars.append(
                        TrackBar(
                            bar=bar.bar,
                            chord=bar.chord,
                            notes=shifted_tokens,
                            repeat_bar=None,
                            transpose_semitones=0,
                            muted=bar.muted,
                        )
                    )
                else:
                    shifted_tokens = []
                    for ev in bar.parsed_events():
                        if ev.midi == -1:
                            shifted_tokens.append(format_note_token(ev))
                        else:
                            shifted_midi = max(0, min(127, ev.midi + semitones))
                            shifted_ev = BarNoteEvent(
                                beat=ev.beat,
                                pitch=pitch_to_name(shifted_midi),
                                midi=shifted_midi,
                                duration=ev.duration,
                                velocity=ev.velocity,
                                articulation=ev.articulation,
                            )
                            shifted_tokens.append(format_note_token(shifted_ev))
                    new_bars.append(bar.model_copy(update={"notes": shifted_tokens}))

        updated_track = TrackFile(
            track_id=track_id_for(track.section_id, track.instrument_id),
            section_id=track.section_id,
            instrument_id=track.instrument_id,
            time_signature=track.time_signature,
            groove_swing=track.groove_swing,
            bars=new_bars,
        )
        work.write_track(
            updated_track,
            expected_digest=current_digest,
            summary=f"Direct edit ({action}) on bars {start_bar}..{end_bar}",
        )
        if action == "set_bar_notes" and clean_chord is not None and score_pair is not None:
            outline_obj, outline_digest = score_pair
            sec_obj = next((s for s in outline_obj.sections if s.id == track.section_id), None)
            if sec_obj is not None and sec_obj.chords:
                updated_chords = list(sec_obj.chords)
                for b_idx in range(start_bar, end_bar + 1):
                    if 1 <= b_idx <= len(updated_chords):
                        updated_chords[b_idx - 1] = clean_chord
                updated_sec = sec_obj.model_copy(update={"chords": updated_chords})
                updated_outline = outline_obj.model_copy(
                    update={
                        "sections": [
                            updated_sec if s.id == track.section_id else s
                            for s in outline_obj.sections
                        ]
                    }
                )
                work.write_score(updated_outline, expected_digest=outline_digest)
        return self.show(song_id)

    def update_mix(
        self,
        song_id: str,
        instrument_id: str,
        *,
        volume: int | None = None,
        pan: float | None = None,
        muted: bool | None = None,
        solo: bool | None = None,
        seen_digest: str | None = None,
    ) -> ShownSong:
        """Update mix parameters (`volume`, `pan`, `muted`, `solo`) on `instrument_id`."""
        work, _ = self._require_decider(song_id)
        loaded = work.read_instrument(instrument_id)
        if loaded is None:
            raise MusicError(f"Instrument '{instrument_id}' does not exist in song '{song_id}'.")
        entry, current_digest = loaded
        if seen_digest is not None and seen_digest != "" and current_digest != seen_digest:
            raise MusicError(
                f"Instrument '{instrument_id}' changed on disk since it was shown. Refresh before adjusting mix."
            )

        updates: dict[str, Any] = {}
        if volume is not None:
            updates["volume"] = volume
        if pan is not None:
            updates["pan"] = pan
        if muted is not None:
            updates["muted"] = muted
        if solo is not None:
            updates["solo"] = solo

        updated_entry = InstrumentEntry.model_validate({**entry.model_dump(), **updates})
        work.write_instrument(updated_entry, expected_digest=current_digest)
        return self.show(song_id)
