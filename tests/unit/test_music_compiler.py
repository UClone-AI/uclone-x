"""Unit tests for the song timeline, MusicXML 4.0, and Standard MIDI File compiler."""

from __future__ import annotations

from uclone_x_music.compiler import compile_midi_bytes, compile_song
from uclone_x_music.schemas import (
    InstrumentEntry,
    ScoreOutline,
    ScoreSection,
    TempoChange,
    TrackBar,
    TrackFile,
)


def test_compile_song_computes_swing_transposition_and_musicxml() -> None:
    """Verify timeline compilation applies groove swing, instrument transposition, and MusicXML output.

    Killed by: examples/music/src/uclone_x_music/compiler.py :: beat_offset += (track.groove_swing - 0.5) * 0.5
    Becomes: beat_offset += 0.0
    """
    outline = ScoreOutline(
        tempo=120,
        time_signature="4/4",
        key_signature="D minor",
        sections=[
            ScoreSection(
                id="verse_1",
                title="Verse 1",
                bars=2,
                chords=["Dm7", "G7"],
                tempo_map=[TempoChange(bar=2, bpm=60)],
            )
        ],
    )
    sax = InstrumentEntry(
        id="tenor_sax",
        name="Tenor Sax",
        category="woodwinds",
        role="lead_melody",
        transposition_semitones=-2,
        min_pitch="Bb2",
        max_pitch="F6",
    )
    track = TrackFile(
        track_id="verse_1_tenor_sax",
        section_id="verse_1",
        instrument_id="tenor_sax",
        groove_swing=0.60,
        bars=[
            TrackBar(
                bar=1,
                notes=[
                    "1.0:D4:0.25:v90",
                    "1.25:F4:0.25:v85",  # Off-beat 16th note gets swing shift
                    "1.5:A4:3.5:v88",
                ],
            ),
            TrackBar(bar=2, repeat_bar=1, transpose_semitones=2),
        ],
    )
    bundle = compile_song("song_1", "Late Night", outline, [sax], [track])
    assert bundle.total_bars == 2
    # Bar 1 at 120 BPM = 2.0s; Bar 2 at 60 BPM = 4.0s -> total 6.0s
    assert bundle.duration_seconds == 6.0
    assert len(bundle.notes) == 6

    first_note = bundle.notes[0]
    assert first_note.concert_midi == 62
    # Concert D4 transposed by -2 semitones produces written E4
    assert first_note.written_pitch == "E4"

    second_note = bundle.notes[1]
    # Unswung 1.25 beat at 120 BPM (0.5s/beat) is 0.125s; swing 0.60 adds (0.10 * 0.5) = 0.05 beats = 0.025s -> 0.15s
    assert abs(second_note.start_seconds - 0.15) < 1e-4

    assert '<score-partwise version="4.0">' in bundle.musicxml
    assert '<part id="tenor_sax">' in bundle.musicxml
    assert '<measure number="2">' in bundle.musicxml


def test_compile_midi_bytes_emits_valid_smf_format_1() -> None:
    """Verify binary Standard MIDI File serializer outputs MThd and MTrk chunks.

    Killed by: examples/music/src/uclone_x_music/compiler.py :: header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks_bytes), _MIDI_TPQ)
    Becomes: header = b"MThd" + struct.pack(">IHHH", 6, 0, len(tracks_bytes), _MIDI_TPQ)
    """
    outline = ScoreOutline(
        tempo=100,
        time_signature="4/4",
        sections=[ScoreSection(id="intro", title="Intro", bars=1, chords=["C"])],
    )
    piano = InstrumentEntry(
        id="piano",
        name="Piano",
        category="piano",
        role="harmonic_comping",
    )
    track = TrackFile(
        track_id="intro_piano",
        section_id="intro",
        instrument_id="piano",
        bars=[TrackBar(bar=1, notes=["1.0:C4:2.0:v85", "3.0:G4:2.0:v85"])],
    )
    bundle = compile_song("song_1", "Prelude", outline, [piano], [track])
    midi_bytes = compile_midi_bytes(bundle)
    assert midi_bytes.startswith(b"MThd\x00\x00\x00\x06\x00\x01")
    assert midi_bytes.count(b"MTrk") == 2
