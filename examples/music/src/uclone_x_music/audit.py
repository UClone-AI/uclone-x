"""Pure music theory, range, voicing, and groove-lock auditing for song tracks.

Every check operates on parsed score sections, instrument definitions, track files,
and style axioms without touching the filesystem.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x_music.schemas import (
    BarNoteEvent,
    InstrumentEntry,
    ScoreSection,
    StyleAxioms,
    TrackFile,
    parse_pitch,
    resolve_track_bars,
)

__all__ = [
    "AuditCode",
    "AuditFinding",
    "AuditReport",
    "AuditSeverity",
    "audit_bar_patch_continuity",
    "audit_section",
    "parse_chord_pitch_classes",
]

AuditCode = Literal[
    "range_violation",
    "harmonic_clash",
    "parallel_fifths_octaves",
    "voice_crossing",
    "low_end_mud",
    "kick_bass_drift",
    "boundary_jump",
]

AuditSeverity = Literal["error", "warning"]

_NOTE_OFFSETS: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_CHORD_RE = re.compile(r"^([A-Ga-g])([#b]?)(.*?)(?:/[A-Ga-g][#b]?)?$")


class AuditFinding(BaseModel):
    """One theory, range, or groove-lock issue located in a section and bar."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    code: AuditCode
    severity: AuditSeverity
    section_id: str
    bar: int = Field(ge=1)
    beat: float | None = None
    instrument_ids: list[str]
    message: str


class AuditReport(BaseModel):
    """Aggregate audit report for a section or track edit."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    passed: bool
    findings: list[AuditFinding]


def _parse_chord_root_and_quality(chord_symbol: str) -> tuple[int, str] | None:
    cleaned = chord_symbol.strip()
    if not cleaned or cleaned.upper() in ("NC", "N.C.", "REST"):
        return None
    match = _CHORD_RE.fullmatch(cleaned)
    if match is None:
        return None
    letter, accidental, quality = match.groups()
    root = _NOTE_OFFSETS[letter.upper()]
    if accidental == "#":
        root = (root + 1) % 12
    elif accidental == "b":
        root = (root - 1) % 12
    return root, quality


def parse_chord_pitch_classes(chord_symbol: str) -> set[int]:
    """Parse a chord symbol (e.g. `Dm7`, `Fmaj7`, `Em7b5`, `A7alt`) into pitch classes `0..11`."""
    parsed = _parse_chord_root_and_quality(chord_symbol)
    if parsed is None:
        return set[int]()
    root, quality = parsed
    q = quality.strip()

    is_dim_triad = q.startswith(("dim", "o"))
    is_half_dim = "m7b5" in q or "ø" in q
    is_aug = q.startswith(("aug", "+"))
    is_sus2 = "sus2" in q
    is_sus4 = "sus4" in q or ("sus" in q and not is_sus2)
    is_minor = (
        (q.startswith("m") and not q.startswith("maj"))
        or q.startswith("min")
        or q.startswith("-")
        or is_dim_triad
        or is_half_dim
    )

    if is_sus2:
        third_offset = 2
    elif is_sus4:
        third_offset = 5
    elif is_minor:
        third_offset = 3
    else:
        third_offset = 4

    if is_dim_triad or is_half_dim or "b5" in q:
        fifth_offset = 6
    elif is_aug or "#5" in q or "alt" in q:
        fifth_offset = 8
    else:
        fifth_offset = 7

    pcs: set[int] = {
        root,
        (root + third_offset) % 12,
        (root + fifth_offset) % 12,
    }

    if "maj7" in q or "maj9" in q or "M7" in q:
        pcs.add((root + 11) % 12)
    elif is_dim_triad and "7" in q:
        pcs.add((root + 9) % 12)
    elif any(tok in q for tok in ("7", "9", "11", "13", "alt")):
        pcs.add((root + 10) % 12)

    if "6" in q:
        pcs.add((root + 9) % 12)
    if "alt" in q:
        pcs.add((root + 1) % 12)
        pcs.add((root + 3) % 12)
        pcs.add((root + 6) % 12)

    return pcs


def _chord_root_and_third(chord_symbol: str) -> set[int]:
    parsed = _parse_chord_root_and_quality(chord_symbol)
    if parsed is None:
        return set[int]()
    root, quality = parsed
    q = quality.strip()
    is_minor = (
        (q.startswith("m") and not q.startswith("maj"))
        or q.startswith("min")
        or q.startswith("-")
        or q.startswith(("dim", "o"))
        or "m7b5" in q
    )
    third = (root + (3 if is_minor else 4)) % 12
    return {root, third}


def audit_section(
    section: ScoreSection,
    tracks: Sequence[TrackFile],
    instruments: Mapping[str, InstrumentEntry],
    axioms: StyleAxioms,
) -> AuditReport:
    """Audit all tracks in `section` against `instruments` and `axioms`."""
    findings: list[AuditFinding] = []
    section_tracks = [t for t in tracks if t.section_id == section.id]
    resolved_by_track: dict[str, dict[int, list[BarNoteEvent]]] = {
        t.track_id: resolve_track_bars(t) for t in section_tracks
    }

    # 1. Per-track checks: range violation, harmonic clash, and low-end mud.
    for track in section_tracks:
        inst = instruments.get(track.instrument_id)
        if inst is None:
            continue
        is_drums = inst.category == "drums"
        low_midi = parse_pitch(inst.min_pitch)
        high_midi = parse_pitch(inst.max_pitch)
        bars_map = resolved_by_track[track.track_id]

        for bar_obj in track.bars:
            bar_num = bar_obj.bar
            events = bars_map.get(bar_num, [])
            chord_sym = bar_obj.chord
            if chord_sym is None and 1 <= bar_num <= len(section.chords):
                chord_sym = section.chords[bar_num - 1]
            guide_pcs: set[int] = _chord_root_and_third(chord_sym) if chord_sym else set[int]()

            for ev in events:
                if ev.midi < 0:
                    continue

                if not is_drums and (ev.midi < low_midi or ev.midi > high_midi):
                    findings.append(
                        AuditFinding(
                            code="range_violation",
                            severity="error",
                            section_id=section.id,
                            bar=bar_num,
                            beat=ev.beat,
                            instrument_ids=[inst.id],
                            message=(
                                f"Note {ev.pitch} (MIDI {ev.midi}) on {inst.id} at bar {bar_num} "
                                f"beat {ev.beat:g} is outside instrument range "
                                f"{inst.min_pitch}..{inst.max_pitch}."
                            ),
                        )
                    )

                if not is_drums and guide_pcs and ev.beat in (1.0, 3.0) and ev.duration >= 1.0:
                    pc = ev.midi % 12
                    if pc not in parse_chord_pitch_classes(chord_sym or "") and any(
                        abs(pc - g) in (1, 11) for g in guide_pcs
                    ):
                        findings.append(
                            AuditFinding(
                                code="harmonic_clash",
                                severity="warning",
                                section_id=section.id,
                                bar=bar_num,
                                beat=ev.beat,
                                instrument_ids=[inst.id],
                                message=(
                                    f"Note {ev.pitch} on {inst.id} at bar {bar_num} beat {ev.beat:g} "
                                    f"forms a minor-2nd clash against chord {chord_sym!r}."
                                ),
                            )
                        )

                if (
                    axioms.profile != "classical_strict"
                    and not is_drums
                    and inst.category != "bass"
                    and inst.role != "bass_foundation"
                    and ev.midi < 36
                ):
                    findings.append(
                        AuditFinding(
                            code="low_end_mud",
                            severity="warning",
                            section_id=section.id,
                            bar=bar_num,
                            beat=ev.beat,
                            instrument_ids=[inst.id],
                            message=(
                                f"Non-bass instrument {inst.id} plays {ev.pitch} (MIDI {ev.midi} < 36) "
                                f"at bar {bar_num} beat {ev.beat:g}, causing low-end mud."
                            ),
                        )
                    )

    # 2. Pairwise melodic / SATB checks: parallel 5ths/8ves and voice crossing.
    if axioms.forbid_parallel_fifths_octaves or axioms.forbid_voice_crossing:
        pitched_tracks = [
            t
            for t in section_tracks
            if (inst := instruments.get(t.instrument_id)) is not None and inst.category != "drums"
        ]
        for idx_a in range(len(pitched_tracks)):
            for idx_b in range(idx_a + 1, len(pitched_tracks)):
                track_a = pitched_tracks[idx_a]
                track_b = pitched_tracks[idx_b]
                inst_a = instruments[track_a.instrument_id]
                inst_b = instruments[track_b.instrument_id]
                min_a = (
                    min(parse_pitch(inst_a.min_pitch), 12)
                    if inst_a.category == "bass" or inst_a.role == "bass_foundation"
                    else parse_pitch(inst_a.min_pitch)
                )
                min_b = (
                    min(parse_pitch(inst_b.min_pitch), 12)
                    if inst_b.category == "bass" or inst_b.role == "bass_foundation"
                    else parse_pitch(inst_b.min_pitch)
                )

                prev_pair: tuple[int, int] | None = None
                for bar_num in range(1, section.bars + 1):
                    events_a = {
                        ev.beat: ev
                        for ev in resolved_by_track[track_a.track_id].get(bar_num, [])
                        if ev.midi >= 0
                    }
                    events_b = {
                        ev.beat: ev
                        for ev in resolved_by_track[track_b.track_id].get(bar_num, [])
                        if ev.midi >= 0
                    }
                    shared_beats = sorted(set(events_a).intersection(events_b))
                    for beat in shared_beats:
                        midi_a = events_a[beat].midi
                        midi_b = events_b[beat].midi

                        if axioms.forbid_voice_crossing and min_a != min_b:
                            if (min_a < min_b and midi_a > midi_b) or (
                                min_b < min_a and midi_b > midi_a
                            ):
                                findings.append(
                                    AuditFinding(
                                        code="voice_crossing",
                                        severity="error",
                                        section_id=section.id,
                                        bar=bar_num,
                                        beat=beat,
                                        instrument_ids=[inst_a.id, inst_b.id],
                                        message=(
                                            f"Voice crossing between {inst_a.id} ({events_a[beat].pitch}) "
                                            f"and {inst_b.id} ({events_b[beat].pitch}) at bar {bar_num} beat {beat:g}."
                                        ),
                                    )
                                )

                        if axioms.forbid_parallel_fifths_octaves and prev_pair is not None:
                            prev_a, prev_b = prev_pair
                            da = midi_a - prev_a
                            db = midi_b - prev_b
                            if (da > 0 and db > 0) or (da < 0 and db < 0):
                                int_prev = abs(prev_a - prev_b) % 12
                                int_curr = abs(midi_a - midi_b) % 12
                                if (
                                    int_prev == int_curr
                                    and int_curr in (0, 7)
                                    and prev_a != prev_b
                                    and midi_a != midi_b
                                ):
                                    interval_label = "fifths" if int_curr == 7 else "octaves"
                                    findings.append(
                                        AuditFinding(
                                            code="parallel_fifths_octaves",
                                            severity="error",
                                            section_id=section.id,
                                            bar=bar_num,
                                            beat=beat,
                                            instrument_ids=[inst_a.id, inst_b.id],
                                            message=(
                                                f"Parallel {interval_label} between {inst_a.id} and {inst_b.id} "
                                                f"into bar {bar_num} beat {beat:g}."
                                            ),
                                        )
                                    )
                        prev_pair = (midi_a, midi_b)

    # 3. Kick-Bass lock check.
    if axioms.require_kick_bass_lock:
        bass_tracks = [
            t
            for t in section_tracks
            if (inst := instruments.get(t.instrument_id)) is not None
            and (inst.category == "bass" or inst.role == "bass_foundation")
        ]
        drum_tracks = [
            t
            for t in section_tracks
            if (inst := instruments.get(t.instrument_id)) is not None and inst.category == "drums"
        ]
        if bass_tracks and drum_tracks:
            bass_track = bass_tracks[0]
            drum_track = drum_tracks[0]
            bass_bars = resolved_by_track[bass_track.track_id]
            drum_bars = resolved_by_track[drum_track.track_id]

            for bar_num in range(1, section.bars + 1):
                b_events = [ev for ev in bass_bars.get(bar_num, []) if ev.midi >= 0]
                if not b_events:
                    continue
                d_events = drum_bars.get(bar_num, [])
                kick_beats = {ev.beat for ev in d_events if ev.midi == 36}
                bass_beat_one = any(abs(ev.beat - 1.0) < 1e-4 for ev in b_events)
                if not kick_beats or (bass_beat_one and 1.0 not in kick_beats):
                    findings.append(
                        AuditFinding(
                            code="kick_bass_drift",
                            severity="warning",
                            section_id=section.id,
                            bar=bar_num,
                            beat=1.0,
                            instrument_ids=[bass_track.instrument_id, drum_track.instrument_id],
                            message=(
                                f"Bar {bar_num} bass attack on {bass_track.instrument_id} lacks a "
                                f"locked kick drum (MIDI 36) on {drum_track.instrument_id}."
                            ),
                        )
                    )

    passed = not any(f.severity == "error" for f in findings)
    return AuditReport(passed=passed, findings=findings)


def audit_bar_patch_continuity(
    track: TrackFile,
    start_bar: int,
    end_bar: int,
    max_leap_semitones: int = 19,
) -> list[AuditFinding]:
    """Check melodic continuity across the boundaries of a patched bar range `start_bar..end_bar`."""
    resolved = resolve_track_bars(track)
    findings: list[AuditFinding] = []

    def _sounding(bar_num: int) -> list[BarNoteEvent]:
        return [ev for ev in resolved.get(bar_num, []) if ev.midi >= 0]

    if start_bar > 1:
        prev_notes = _sounding(start_bar - 1)
        curr_notes = _sounding(start_bar)
        if prev_notes and curr_notes:
            leap = abs(curr_notes[0].midi - prev_notes[-1].midi)
            if leap > max_leap_semitones:
                findings.append(
                    AuditFinding(
                        code="boundary_jump",
                        severity="warning",
                        section_id=track.section_id,
                        bar=start_bar,
                        beat=curr_notes[0].beat,
                        instrument_ids=[track.instrument_id],
                        message=(
                            f"Patch entry at bar {start_bar} leaps {leap} semitones from "
                            f"bar {start_bar - 1} ({prev_notes[-1].pitch} -> {curr_notes[0].pitch})."
                        ),
                    )
                )

    if end_bar < len(track.bars):
        curr_notes = _sounding(end_bar)
        next_notes = _sounding(end_bar + 1)
        if curr_notes and next_notes:
            leap = abs(next_notes[0].midi - curr_notes[-1].midi)
            if leap > max_leap_semitones:
                findings.append(
                    AuditFinding(
                        code="boundary_jump",
                        severity="warning",
                        section_id=track.section_id,
                        bar=end_bar + 1,
                        beat=next_notes[0].beat,
                        instrument_ids=[track.instrument_id],
                        message=(
                            f"Patch exit after bar {end_bar} leaps {leap} semitones into "
                            f"bar {end_bar + 1} ({curr_notes[-1].pitch} -> {next_notes[0].pitch})."
                        ),
                    )
                )

    return findings
