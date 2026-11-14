"""Unit tests for music theory, range, voicing, and groove-lock auditing."""

from __future__ import annotations

from uclone_x_music.audit import (
    audit_bar_patch_continuity,
    audit_section,
    parse_chord_pitch_classes,
)
from uclone_x_music.schemas import (
    InstrumentEntry,
    ScoreSection,
    TrackBar,
    TrackFile,
    default_style_axioms,
)


def test_parse_chord_pitch_classes_covers_triads_and_sevenths() -> None:
    """Verify chord symbol parsing into pitch classes 0..11.

    Killed by: examples/music/src/uclone_x_music/audit.py :: (root + third_offset) % 12
    Becomes: (root + third_offset + 1) % 12
    """
    assert parse_chord_pitch_classes("C") == {0, 4, 7}
    assert parse_chord_pitch_classes("Dm7") == {2, 5, 9, 0}
    assert parse_chord_pitch_classes("Fmaj7") == {5, 9, 0, 4}
    assert parse_chord_pitch_classes("Em7b5") == {4, 7, 10, 2}
    assert parse_chord_pitch_classes("A7alt") == {9, 1, 5, 7, 10, 0, 3}
    assert parse_chord_pitch_classes("NC") == set()


def test_audit_section_detects_range_violation_and_harmonic_clash() -> None:
    """Verify out-of-range notes and strong-beat semitone clashes are reported.

    Killed by: examples/music/src/uclone_x_music/audit.py :: code="range_violation"
    Becomes: code="low_end_mud"
    """
    section = ScoreSection(id="verse_1", title="Verse 1", bars=1, chords=["C"])
    flute = InstrumentEntry(
        id="flute",
        name="Flute",
        category="woodwinds",
        role="lead_melody",
        min_pitch="C4",
        max_pitch="C7",
    )
    track = TrackFile(
        track_id="verse_1_flute",
        section_id="verse_1",
        instrument_id="flute",
        bars=[
            TrackBar(
                bar=1,
                chord="C",
                notes=[
                    "1.0:C#5:2.0:v85",  # C# clashes with C root
                    "3.0:G2:2.0:v85",  # G2 is below flute min_pitch C4 and < MIDI 36? G2 is 43
                    "3.0:A1:2.0:v85",  # A1 is MIDI 33 (< C4 and < 36)
                ],
            )
        ],
    )
    report = audit_section(
        section,
        [track],
        {"flute": flute},
        default_style_axioms("pop_ballad"),
    )
    assert not report.passed
    codes = {f.code for f in report.findings}
    assert "range_violation" in codes
    assert "harmonic_clash" in codes
    assert "low_end_mud" in codes


def test_audit_section_detects_parallel_fifths_and_voice_crossing() -> None:
    """Verify classical strict axioms catch parallel fifths and voice crossing.

    Killed by: examples/music/src/uclone_x_music/audit.py :: code="parallel_fifths_octaves"
    Becomes: code="low_end_mud"
    """
    section = ScoreSection(id="movement_1", title="Movement 1", bars=2, chords=["C", "D"])
    cello = InstrumentEntry(
        id="cello",
        name="Cello",
        category="strings",
        role="satb_bass",
        min_pitch="C2",
        max_pitch="E5",
    )
    viola = InstrumentEntry(
        id="viola",
        name="Viola",
        category="strings",
        role="satb_tenor",
        min_pitch="C3",
        max_pitch="A5",
    )
    cello_track = TrackFile(
        track_id="movement_1_cello",
        section_id="movement_1",
        instrument_id="cello",
        bars=[
            TrackBar(bar=1, notes=["1.0:C3:2.0:v80", "3.0:D3:2.0:v80"]),
            TrackBar(bar=2, notes=["1.0:E4:4.0:v80"]),  # Crosses above viola G3
        ],
    )
    viola_track = TrackFile(
        track_id="movement_1_viola",
        section_id="movement_1",
        instrument_id="viola",
        bars=[
            TrackBar(
                bar=1, notes=["1.0:G3:2.0:v80", "3.0:A3:2.0:v80"]
            ),  # C3-G3 -> D3-A3 parallel 5th
            TrackBar(bar=2, notes=["1.0:G3:4.0:v80"]),
        ],
    )
    report = audit_section(
        section,
        [cello_track, viola_track],
        {"cello": cello, "viola": viola},
        default_style_axioms("classical_strict"),
    )
    assert not report.passed
    codes = {f.code for f in report.findings}
    assert "parallel_fifths_octaves" in codes
    assert "voice_crossing" in codes


def test_audit_section_detects_kick_bass_drift() -> None:
    """Verify kick-bass lock axiom warns when bass beat 1 has no matching kick hit.

    Killed by: examples/music/src/uclone_x_music/audit.py :: code="kick_bass_drift"
    Becomes: code="low_end_mud"
    """
    section = ScoreSection(id="verse_1", title="Verse", bars=1, chords=["Dm7"])
    bass = InstrumentEntry(
        id="bass_synth",
        name="Bass",
        category="bass",
        role="bass_foundation",
        min_pitch="C1",
        max_pitch="C5",
    )
    drums = InstrumentEntry(
        id="drum_kit",
        name="Drums",
        category="drums",
        role="rhythm_percussion",
    )
    bass_track = TrackFile(
        track_id="verse_1_bass_synth",
        section_id="verse_1",
        instrument_id="bass_synth",
        bars=[TrackBar(bar=1, notes=["1.0:D2:4.0:v95"])],
    )
    drum_track = TrackFile(
        track_id="verse_1_drum_kit",
        section_id="verse_1",
        instrument_id="drum_kit",
        bars=[TrackBar(bar=1, notes=["1.0:HIHAT:2.0:v80", "3.0:SNARE:2.0:v90"])],
    )
    report = audit_section(
        section,
        [bass_track, drum_track],
        {"bass_synth": bass, "drum_kit": drums},
        default_style_axioms("hiphop_groove"),
    )
    assert any(f.code == "kick_bass_drift" for f in report.findings)


def test_audit_bar_patch_continuity_detects_boundary_jump() -> None:
    """Verify large melodic leaps across patched bar boundaries emit boundary_jump.

    Killed by: examples/music/src/uclone_x_music/audit.py :: if start_bar > 1:
    Becomes: if False:
    """
    track = TrackFile(
        track_id="verse_1_lead",
        section_id="verse_1",
        instrument_id="lead",
        bars=[
            TrackBar(bar=1, notes=["1.0:C4:4.0:v85"]),
            TrackBar(bar=2, notes=["1.0:C6:4.0:v85"]),  # 24 semitone jump from C4
            TrackBar(bar=3, notes=["1.0:C4:4.0:v85"]),  # 24 semitone jump back to C4
        ],
    )
    findings = audit_bar_patch_continuity(track, start_bar=2, end_bar=2, max_leap_semitones=19)
    assert len(findings) == 2
    assert all(f.code == "boundary_jump" for f in findings)
