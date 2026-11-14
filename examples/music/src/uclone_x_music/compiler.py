"""Pure playback timeline, MusicXML 4.0, and Standard MIDI File (SMF) compiler.

Takes a song's `ScoreOutline`, `InstrumentEntry` definitions, and `TrackFile`s and
compiles a deterministic `PlaybackBundle` for the browser Web Audio synthesizer,
sheet music renderer, and binary `.mid` export.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from xml.sax.saxutils import escape

from pydantic import BaseModel, ConfigDict, Field

from uclone_x_music.schemas import (
    BarNoteEvent,
    InstrumentEntry,
    ScoreOutline,
    TrackFile,
    beats_per_bar_for,
    pitch_to_name,
    resolve_track_bars,
)

__all__ = [
    "CompiledBarCursor",
    "CompiledNote",
    "PlaybackBundle",
    "compile_midi_bytes",
    "compile_song",
]

_DIVISIONS = 4  # 4 divisions per quarter note (16th-note resolution)
_MIDI_TPQ = 480


class CompiledNote(BaseModel):
    """A single scheduled note event on the compiled playback timeline."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    track_id: str
    section_id: str
    instrument_id: str
    bar: int = Field(ge=1)
    global_bar: int = Field(ge=1)
    beat: float = Field(ge=1.0)
    start_seconds: float = Field(ge=0.0)
    duration_seconds: float = Field(gt=0.0)
    concert_midi: int = Field(ge=0, le=127)
    written_pitch: str
    velocity: int = Field(ge=0, le=127)
    articulation: str
    pan: float = Field(ge=-1.0, le=1.0)
    volume: int = Field(ge=0, le=100)


class CompiledBarCursor(BaseModel):
    """Bar timing and chord metadata for synchronized playhead highlighting."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    global_bar: int = Field(ge=1)
    section_id: str
    section_bar: int = Field(ge=1)
    chord: str | None = None
    bpm: int = Field(ge=20, le=300)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)


class PlaybackBundle(BaseModel):
    """Compiled song timeline, bar cursors, and MusicXML score."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    song_id: str
    title: str
    tempo: int
    time_signature: str
    key_signature: str
    total_bars: int
    duration_seconds: float
    instruments: list[InstrumentEntry]
    notes: list[CompiledNote]
    bar_cursors: list[CompiledBarCursor]
    musicxml: str


def compile_song(
    song_id: str,
    title: str,
    outline: ScoreOutline,
    instruments: Sequence[InstrumentEntry],
    tracks: Sequence[TrackFile],
) -> PlaybackBundle:
    """Compile `outline`, `instruments`, and `tracks` into a deterministic `PlaybackBundle`."""
    bpb = beats_per_bar_for(outline.time_signature)
    inst_by_id = {inst.id: inst for inst in instruments}
    any_solo = any(inst.solo for inst in instruments)

    bar_cursors: list[CompiledBarCursor] = []
    cursor_by_sec_bar: dict[tuple[str, int], CompiledBarCursor] = {}
    current_time = 0.0
    current_bpm = outline.tempo
    global_bar = 0

    for section in outline.sections:
        tempo_changes = {tc.bar: tc.bpm for tc in section.tempo_map}
        for sec_bar in range(1, section.bars + 1):
            global_bar += 1
            if sec_bar in tempo_changes:
                current_bpm = tempo_changes[sec_bar]
            sec_per_beat = 60.0 / float(current_bpm)
            bar_dur = bpb * sec_per_beat
            chord = section.chords[sec_bar - 1] if 1 <= sec_bar <= len(section.chords) else None
            cursor = CompiledBarCursor(
                global_bar=global_bar,
                section_id=section.id,
                section_bar=sec_bar,
                chord=chord,
                bpm=current_bpm,
                start_seconds=round(current_time, 6),
                end_seconds=round(current_time + bar_dur, 6),
            )
            bar_cursors.append(cursor)
            cursor_by_sec_bar[(section.id, sec_bar)] = cursor
            current_time += bar_dur

    compiled_notes: list[CompiledNote] = []
    resolved_Lookup: dict[tuple[str, str], dict[int, list[BarNoteEvent]]] = {}

    for track in tracks:
        inst = inst_by_id.get(track.instrument_id)
        if inst is None:
            continue
        resolved = resolve_track_bars(track)
        resolved_Lookup[(track.section_id, track.instrument_id)] = resolved

        if inst.muted or (any_solo and not inst.solo):
            continue

        for sec_bar, events in resolved.items():
            cursor = cursor_by_sec_bar.get((track.section_id, sec_bar))
            if cursor is None:
                continue
            sec_per_beat = 60.0 / float(cursor.bpm)
            for ev in events:
                if ev.midi < 0:
                    continue
                beat_offset = ev.beat - 1.0
                # Apply groove swing to off-beat 16th/8th subdivisions
                frac = round(beat_offset % 0.5, 4)
                if abs(frac - 0.25) < 1e-3 and track.groove_swing > 0.5:
                    beat_offset += (track.groove_swing - 0.5) * 0.5

                start_sec = cursor.start_seconds + beat_offset * sec_per_beat
                dur_factor = (
                    0.55
                    if ev.articulation == "staccato"
                    else (1.02 if ev.articulation == "legato" else 0.92)
                )
                dur_sec = max(0.02, ev.duration * sec_per_beat * dur_factor)

                if inst.category == "drums":
                    written = ev.pitch
                else:
                    written_midi = max(0, min(127, ev.midi - inst.transposition_semitones))
                    written = pitch_to_name(written_midi)

                compiled_notes.append(
                    CompiledNote(
                        track_id=track.track_id,
                        section_id=track.section_id,
                        instrument_id=inst.id,
                        bar=sec_bar,
                        global_bar=cursor.global_bar,
                        beat=ev.beat,
                        start_seconds=round(start_sec, 6),
                        duration_seconds=round(dur_sec, 6),
                        concert_midi=ev.midi,
                        written_pitch=written,
                        velocity=ev.velocity,
                        articulation=ev.articulation,
                        pan=inst.pan,
                        volume=inst.volume,
                    )
                )

    compiled_notes.sort(
        key=lambda n: (n.start_seconds, n.global_bar, n.instrument_id, n.concert_midi)
    )

    musicxml = _build_musicxml(
        title=title,
        outline=outline,
        instruments=instruments,
        bar_cursors=bar_cursors,
        resolved_lookup=resolved_Lookup,
    )

    return PlaybackBundle(
        song_id=song_id,
        title=title,
        tempo=outline.tempo,
        time_signature=outline.time_signature,
        key_signature=outline.key_signature,
        total_bars=global_bar,
        duration_seconds=round(current_time, 6),
        instruments=list(instruments),
        notes=compiled_notes,
        bar_cursors=bar_cursors,
        musicxml=musicxml,
    )


def _duration_type(duration_beats: float) -> str:
    if duration_beats >= 3.5:
        return "whole"
    if duration_beats >= 1.75:
        return "half"
    if duration_beats >= 0.85:
        return "quarter"
    if duration_beats >= 0.4:
        return "eighth"
    return "16th"


def _midi_to_step_alter_octave(midi: int) -> tuple[str, int, int]:
    name = pitch_to_name(midi)
    step = name[0]
    if "#" in name:
        alter = 1
        octave = int(name[2:])
    else:
        alter = 0
        octave = int(name[1:])
    return step, alter, octave


def _build_musicxml(
    *,
    title: str,
    outline: ScoreOutline,
    instruments: Sequence[InstrumentEntry],
    bar_cursors: Sequence[CompiledBarCursor],
    resolved_lookup: dict[tuple[str, str], dict[int, list[BarNoteEvent]]],
) -> str:
    num_str, den_str = outline.time_signature.split("/")
    bpb = beats_per_bar_for(outline.time_signature)

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<score-partwise version="4.0">',
        f"  <work><work-title>{escape(title)}</work-title></work>",
        "  <part-list>",
    ]
    for inst in instruments:
        lines.append(f'    <score-part id="{escape(inst.id)}">')
        lines.append(f"      <part-name>{escape(inst.name)}</part-name>")
        lines.append("    </score-part>")
    lines.append("  </part-list>")

    for inst in instruments:
        lines.append(f'  <part id="{escape(inst.id)}">')
        for cursor in bar_cursors:
            lines.append(f'    <measure number="{cursor.global_bar}">')
            if cursor.global_bar == 1:
                lines.append("      <attributes>")
                lines.append(f"        <divisions>{_DIVISIONS}</divisions>")
                lines.append(
                    f"        <time><beats>{num_str}</beats><beat-type>{den_str}</beat-type></time>"
                )
                lines.append("      </attributes>")
            if cursor.chord:
                lines.append(
                    f"      <direction><direction-type><words>{escape(cursor.chord)}</words></direction-type></direction>"
                )

            events = resolved_lookup.get((cursor.section_id, inst.id), {}).get(
                cursor.section_bar, []
            )
            if not events:
                divs = max(1, int(round(bpb * _DIVISIONS)))
                lines.append(
                    f"      <note><rest/><duration>{divs}</duration><type>whole</type></note>"
                )
            else:
                for ev in events:
                    divs = max(1, int(round(ev.duration * _DIVISIONS)))
                    ntype = _duration_type(ev.duration)
                    if ev.midi < 0:
                        lines.append(
                            f"      <note><rest/><duration>{divs}</duration><type>{ntype}</type></note>"
                        )
                    else:
                        written_midi = (
                            ev.midi
                            if inst.category == "drums"
                            else max(0, min(127, ev.midi - inst.transposition_semitones))
                        )
                        step, alter, octave = _midi_to_step_alter_octave(written_midi)
                        alter_xml = f"<alter>{alter}</alter>" if alter != 0 else ""
                        lines.append(
                            f"      <note><pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"
                            f"<duration>{divs}</duration><type>{ntype}</type></note>"
                        )
            lines.append("    </measure>")
        lines.append("  </part>")

    lines.append("</score-partwise>")
    return "\n".join(lines)


def _encode_vlq(value: int) -> bytes:
    """Encode a non-negative integer as a MIDI Variable-Length Quantity."""
    val = max(0, value)
    buf = [val & 0x7F]
    val >>= 7
    while val > 0:
        buf.append((val & 0x7F) | 0x80)
        val >>= 7
    buf.reverse()
    return bytes(buf)


def compile_midi_bytes(bundle: PlaybackBundle) -> bytes:
    """Serialize `bundle` into a Standard MIDI File (SMF Format 1) byte stream."""
    sec_per_beat = 60.0 / float(max(20, bundle.tempo))

    # Track 0: tempo and time signature metadata
    meta_events = bytearray()
    us_per_quarter = int(round(60_000_000 / max(20, bundle.tempo)))
    meta_events.extend(_encode_vlq(0))
    meta_events.extend(b"\xff\x51\x03")
    meta_events.extend(us_per_quarter.to_bytes(3, "big"))
    meta_events.extend(_encode_vlq(0))
    meta_events.extend(b"\xff\x2f\x00")

    tracks_bytes: list[bytes] = [b"MTrk" + struct.pack(">I", len(meta_events)) + bytes(meta_events)]

    # One track per instrument with notes
    for inst in bundle.instruments:
        inst_notes = [n for n in bundle.notes if n.instrument_id == inst.id]
        track_buf = bytearray()
        channel = 9 if inst.category == "drums" else ((inst.midi_channel - 1) % 16)
        if channel == 9 and inst.category != "drums":
            channel = 0

        # Program change at tick 0
        track_buf.extend(_encode_vlq(0))
        track_buf.extend(bytes([0xC0 | channel, inst.midi_program & 0x7F]))

        raw_midi_events: list[tuple[int, int, int, int]] = []
        for note in inst_notes:
            start_tick = max(0, int(round((note.start_seconds / sec_per_beat) * _MIDI_TPQ)))
            dur_ticks = max(1, int(round((note.duration_seconds / sec_per_beat) * _MIDI_TPQ)))
            end_tick = start_tick + dur_ticks
            raw_midi_events.append((start_tick, 1, note.concert_midi & 0x7F, note.velocity & 0x7F))
            raw_midi_events.append((end_tick, 0, note.concert_midi & 0x7F, 0))

        raw_midi_events.sort(key=lambda item: (item[0], item[1], item[2]))
        last_tick = 0
        for tick, is_on, pitch, vel in raw_midi_events:
            delta = max(0, tick - last_tick)
            last_tick = tick
            track_buf.extend(_encode_vlq(delta))
            status = (0x90 if is_on == 1 else 0x80) | channel
            track_buf.extend(bytes([status, pitch, vel]))

        track_buf.extend(_encode_vlq(0))
        track_buf.extend(b"\xff\x2f\x00")
        tracks_bytes.append(b"MTrk" + struct.pack(">I", len(track_buf)) + bytes(track_buf))

    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks_bytes), _MIDI_TPQ)
    return header + b"".join(tracks_bytes)
