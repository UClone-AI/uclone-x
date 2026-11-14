"""Unit tests for deterministic bass groove, hip-hop drums, double-time subdivision, and SATB strings."""

from __future__ import annotations

import pytest

from uclone_x_music.audit import audit_section
from uclone_x_music.groove import (
    generate_bass_groove,
    generate_classical_satb_strings,
    generate_hiphop_drums,
    subdivide_bars_faster,
)
from uclone_x_music.schemas import (
    InstrumentEntry,
    ScoreSection,
    default_style_axioms,
    resolve_track_bars,
)


def test_generate_bass_groove_and_hiphop_drums_lock_kick_and_bass() -> None:
    """Verify generated hip-hop bass and drums lock on beat 1.0 and pass audit.

    Killed by: examples/music/src/uclone_x_music/groove.py :: swing = 0.58 if substyle == "boombap" else 0.50
    Becomes: swing = 0.50 if substyle == "boombap" else 0.50
    """
    section = ScoreSection(
        id="verse_1",
        title="Verse 1",
        bars=4,
        chords=["Dm7", "Gm7", "Bbmaj7", "A7alt"],
    )
    bass_track = generate_bass_groove(section, instrument_id="bass_synth", style="hiphop_groove")
    drum_track = generate_hiphop_drums(
        section, bass_track=bass_track, instrument_id="drum_kit", substyle="boombap"
    )
    assert bass_track.groove_swing == 0.58
    assert drum_track.groove_swing == 0.58

    instruments = {
        "bass_synth": InstrumentEntry(
            id="bass_synth",
            name="Bass Synth",
            category="bass",
            role="bass_foundation",
            min_pitch="C1",
            max_pitch="C5",
        ),
        "drum_kit": InstrumentEntry(
            id="drum_kit",
            name="Drum Kit",
            category="drums",
            role="rhythm_percussion",
        ),
    }
    report = audit_section(
        section,
        [bass_track, drum_track],
        instruments,
        default_style_axioms("hiphop_groove"),
    )
    assert report.passed
    assert not any(f.code == "kick_bass_drift" for f in report.findings)


def test_subdivide_bars_faster_splits_notes_and_preserves_bar_span() -> None:
    """Verify double-time subdivision splits notes into faster figurations while preserving rests.

    Killed by: examples/music/src/uclone_x_music/groove.py :: sub_dur = round(ev.duration / float(factor), 4)
    Becomes: sub_dur = round(ev.duration, 4)
    """
    section = ScoreSection(id="verse_1", title="Verse 1", bars=2, chords=["Dm7", "Gm7"])
    bass_track = generate_bass_groove(section, instrument_id="bass_synth", style="hiphop_groove")
    orig_bar_1 = resolve_track_bars(bass_track)[1]
    patched_bars = subdivide_bars_faster(bass_track, start_bar=1, end_bar=2, factor=2)
    assert len(patched_bars) == 2

    updated = bass_track.model_copy(update={"bars": patched_bars})
    new_bar_1 = resolve_track_bars(updated)[1]
    assert len(new_bar_1) > len(orig_bar_1)
    assert any(ev.midi == -1 for ev in new_bar_1)
    # The bar still spans the same beats: each note is split, not repeated at full length.
    assert sum(ev.duration for ev in new_bar_1) == pytest.approx(
        sum(ev.duration for ev in orig_bar_1)
    )


def test_generate_classical_satb_strings_passes_strict_counterpoint_audit() -> None:
    """Verify generated 4-part classical strings have zero parallel 5ths/8ves or voice crossing.

    Killed by: examples/music/src/uclone_x_music/groove.py :: cello_midi, viola_midi, vln2_midi, vln1_midi = voicing
    Becomes: vln1_midi, viola_midi, vln2_midi, cello_midi = voicing
    """
    section = ScoreSection(
        id="movement_1",
        title="Movement I",
        kind="movement",
        bars=4,
        chords=["C", "Am", "F", "G7"],
    )
    instruments, tracks = generate_classical_satb_strings(section)
    assert len(instruments) == 4
    assert len(tracks) == 4
    inst_map = {inst.id: inst for inst in instruments}
    report = audit_section(
        section,
        tracks,
        inst_map,
        default_style_axioms("classical_strict"),
    )
    assert report.passed
    assert report.findings == []
