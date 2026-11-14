"""Unit tests for `uclone_x_music.schemas`: Bar-Grid Step DSL, timing validation, and YAML parsing."""

from __future__ import annotations

import pytest

from uclone_x_music.schemas import (
    InstrumentEntry,
    MusicFileError,
    MusicProposal,
    RegionSelection,
    ScoreOutline,
    ScoreSection,
    TrackBar,
    TrackFile,
    default_style_axioms,
    dump_file,
    format_note_token,
    parse_file,
    parse_note_token,
    parse_pitch,
    pitch_to_name,
    resolve_track_bars,
    validate_track_against_section,
)


def test_parse_pitch_and_roundtrip() -> None:
    """Scientific pitches, rests, and drum aliases parse into expected MIDI numbers.

    Killed by: examples/music/src/uclone_x_music/schemas.py :: midi = (octave + 1) * 12 + semitone
    Becomes: midi = octave * 12 + semitone
    """
    assert parse_pitch("C4") == 60
    assert parse_pitch("D2") == 38
    f_sharp_4 = "F#4"
    assert parse_pitch(f_sharp_4) == 66
    assert parse_pitch("Bb3") == 58
    assert parse_pitch("REST") == -1
    assert parse_pitch("R") == -1
    assert parse_pitch("KICK") == 36
    assert parse_pitch("SNARE") == 38
    assert parse_pitch("HIHAT") == 42

    assert pitch_to_name(60) == "C4"
    assert pitch_to_name(38) == "D2"
    assert pitch_to_name(-1) == "REST"

    with pytest.raises(ValueError, match="Invalid pitch"):
        parse_pitch("H4")


def test_parse_and_format_note_token() -> None:
    """Bar-Grid Step DSL tokens parse beat, pitch, duration, velocity, and articulation.

    Killed by: examples/music/src/uclone_x_music/schemas.py :: velocity = 0 if midi == -1 else vel
    Becomes: velocity = vel
    """
    ev = parse_note_token("1.0:D2:1.5:v92:staccato")
    assert ev.beat == 1.0
    assert ev.pitch == "D2"
    assert ev.midi == 38
    assert ev.duration == 1.5
    assert ev.velocity == 92
    assert ev.articulation == "staccato"
    assert format_note_token(ev) == "1.0:D2:1.5:v92:staccato"

    rest = parse_note_token("2.5:REST:0.5:v90")
    assert rest.midi == -1
    assert rest.velocity == 0
    assert format_note_token(rest) == "2.5:REST:0.5"


def test_track_bar_rejects_timing_drift() -> None:
    """A bar whose notes do not span the time signature's beat count is refused (Fail-Fast).

    Killed by: examples/music/src/uclone_x_music/schemas.py :: if abs(total_span - bpb) > 1e-3:
    Becomes: if False:
    """
    with pytest.raises(ValueError, match="requires 4 beats per bar"):
        TrackFile(
            track_id="sec_verse_bass",
            section_id="sec_verse",
            instrument_id="bass",
            time_signature="4/4",
            bars=[
                TrackBar(
                    bar=1,
                    notes=["1.0:D2:1.5:v90", "2.5:A2:0.5:v80"],  # spans only 2.0 beats
                )
            ],
        )

    with pytest.raises(ValueError, match="overflows the bar boundary"):
        TrackFile(
            track_id="sec_verse_bass",
            section_id="sec_verse",
            instrument_id="bass",
            time_signature="4/4",
            bars=[
                TrackBar(
                    bar=1,
                    notes=["1.0:D2:3.0:v90", "4.0:A2:1.5:v80"],  # ends at beat 5.5 > 5.0
                )
            ],
        )


def test_resolve_track_bars_expands_repeat_and_transpose() -> None:
    """`repeat_bar` and `transpose_semitones` expand deterministically and respect `muted`.

    Killed by: examples/music/src/uclone_x_music/schemas.py :: new_midi = max(0, min(127, ev.midi + shift))
    Becomes: new_midi = max(0, min(127, ev.midi))
    """
    track = TrackFile(
        track_id="sec_verse_bass",
        section_id="sec_verse",
        instrument_id="bass",
        bars=[
            TrackBar(bar=1, chord="Dm7", notes=["1.0:D2:2.0:v90", "3.0:A2:2.0:v85"]),
            TrackBar(bar=2, chord="C7", repeat_bar=1, transpose_semitones=-2),
            TrackBar(bar=3, chord="Bb", repeat_bar=1, muted=True),
        ],
    )
    resolved = resolve_track_bars(track)
    assert [e.midi for e in resolved[1]] == [38, 45]
    assert [e.midi for e in resolved[2]] == [36, 43]
    assert [e.midi for e in resolved[3]] == [-1]
    assert resolved[3][0].duration == 4.0


def test_parse_file_and_validate_against_section() -> None:
    """`parse_file` names the file and field on invalid YAML, and section checks bar counts.

    Killed by: examples/music/src/uclone_x_music/schemas.py :: if len(track.bars) != section.bars:
    Becomes: if False:
    """
    section = ScoreSection(id="verse_1", title="Verse 1", bars=2, chords=["Dm7", "G7"])
    outline = ScoreOutline(
        tempo=96, time_signature="4/4", key_signature="D minor", sections=[section]
    )
    track = TrackFile(
        track_id="verse_1_bass",
        section_id="verse_1",
        instrument_id="bass",
        bars=[
            TrackBar(bar=1, notes=["1.0:D2:4.0:v90"]),
            TrackBar(bar=2, notes=["1.0:G2:4.0:v90"]),
        ],
    )
    validate_track_against_section(track, section, outline)

    text = dump_file(track)
    parsed = parse_file(text, TrackFile, relative_path="tracks/verse_1_bass.yaml")
    assert parsed == track

    with pytest.raises(MusicFileError, match="tracks/bad.yaml"):
        parse_file("tempo: 'not_an_int'\n", ScoreOutline, relative_path="tracks/bad.yaml")

    short_section = ScoreSection(id="verse_1", title="Verse 1", bars=4)
    with pytest.raises(MusicFileError, match="requires 4 bars"):
        validate_track_against_section(
            track,
            short_section,
            ScoreOutline(sections=[short_section]),
        )


def test_style_axioms_and_proposal_validation() -> None:
    """Style axioms configure genre rules, and `track_patch` proposals check bar ranges.

    Killed by: examples/music/src/uclone_x_music/schemas.py :: if actual != expected:
    Becomes: if False:
    """
    classical = default_style_axioms("classical_strict")
    assert classical.forbid_parallel_fifths_octaves is True
    assert classical.forbid_voice_crossing is True

    hiphop = default_style_axioms("hiphop_groove")
    assert hiphop.require_kick_bass_lock is True
    assert hiphop.default_swing == 0.58

    sel = RegionSelection(section_id="verse_1", start_bar=3, end_bar=4, instrument_ids=["violin"])
    assert sel.start_bar == 3

    prop = MusicProposal(
        id="p001",
        kind="track_patch",
        target_file="tracks/verse_1_violin.yaml",
        section_id="verse_1",
        instrument_id="violin",
        start_bar=3,
        end_bar=4,
        summary="Double-time violin on bars 3-4",
        proposed_bars=[
            TrackBar(bar=3, notes=["1.0:A4:4.0:v90"]),
            TrackBar(bar=4, notes=["1.0:D5:4.0:v90"]),
        ],
    )
    assert prop.id == "p001"
    with pytest.raises(ValueError, match="must cover bars"):
        MusicProposal(
            id="p002",
            kind="track_patch",
            target_file="tracks/verse_1_violin.yaml",
            section_id="verse_1",
            instrument_id="violin",
            start_bar=3,
            end_bar=4,
            summary="Bars that skip bar 4",
            proposed_bars=[
                TrackBar(bar=3, notes=["1.0:A4:4.0:v90"]),
                TrackBar(bar=5, notes=["1.0:D5:4.0:v90"]),
            ],
        )

    inst = InstrumentEntry(
        id="violin_solo",
        name="Solo Violin",
        category="strings",
        role="lead_melody",
        min_pitch="G3",
        max_pitch="E7",
    )
    assert inst.id == "violin_solo"
