"""Unit tests for `uclone_x_music.context` and `uclone_x_music.proposals`."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x_music.context import build_score_context, classify_register_band
from uclone_x_music.library import MusicError, SongChangedError, SongLibrary
from uclone_x_music.proposals import (
    apply_proposal,
    preview_patched_track,
    propose_instrument_change,
    propose_track_patch,
    reject_proposal,
)
from uclone_x_music.schemas import (
    InstrumentEntry,
    RegionSelection,
    ScoreOutline,
    ScoreSection,
    TrackBar,
    TrackFile,
    parse_note_token,
)
from uclone_x_music.work import SongWork


def _setup_song(tmp_path: Path) -> SongWork:
    lib = SongLibrary(tmp_path)
    lib.create("Autumn Sketch", "room-1", song_id="autumn-sketch")
    work = SongWork(lib, "autumn-sketch", "room-1")
    outline = ScoreOutline(
        tempo=92,
        time_signature="4/4",
        key_signature="D minor",
        sections=[
            ScoreSection(
                id="verse_1",
                title="Verse 1",
                bars=4,
                chords=["Dm7", "Gm7", "A7", "Dm7"],
            )
        ],
    )
    work.write_score(outline, expected_digest=None)
    work.write_instrument(
        InstrumentEntry(
            id="bass",
            name="Electric Bass",
            category="bass",
            role="bass_foundation",
            min_pitch="E1",
            max_pitch="G4",
        ),
        expected_digest=None,
    )
    work.write_instrument(
        InstrumentEntry(
            id="violin",
            name="Solo Violin",
            category="strings",
            role="lead_melody",
            min_pitch="G3",
            max_pitch="E7",
        ),
        expected_digest=None,
    )
    work.write_track(
        TrackFile(
            track_id="verse_1_bass",
            section_id="verse_1",
            instrument_id="bass",
            bars=[
                TrackBar(bar=1, chord="Dm7", notes=["1.0:D2:4.0:v90"]),
                TrackBar(bar=2, chord="Gm7", notes=["1.0:G2:4.0:v90"]),
                TrackBar(bar=3, chord="A7", notes=["1.0:A2:4.0:v90"]),
                TrackBar(bar=4, chord="Dm7", notes=["1.0:D2:4.0:v90"]),
            ],
        ),
        expected_digest=None,
    )
    work.write_track(
        TrackFile(
            track_id="verse_1_violin",
            section_id="verse_1",
            instrument_id="violin",
            bars=[
                TrackBar(bar=1, chord="Dm7", notes=["1.0:A4:2.0:v85", "3.0:F5:2.0:v85"]),
                TrackBar(bar=2, chord="Gm7", notes=["1.0:G4:4.0:v85"]),
                TrackBar(bar=3, chord="A7", notes=["1.0:E5:4.0:v90"]),
                TrackBar(bar=4, chord="Dm7", notes=["1.0:D5:4.0:v85"]),
            ],
        ),
        expected_digest=None,
    )
    return work


def test_build_score_context_computes_vertical_stack_and_manifest(tmp_path: Path) -> None:
    """`build_score_context` classifies registers, identifies free registers/missing roles, and extracts tail bars.

    Killed by: examples/music/src/uclone_x_music/context.py :: if avg_midi < 52.0:
    Becomes: if avg_midi < 0.0:
    """
    work = _setup_song(tmp_path)
    bundle = build_score_context(work, section_id="verse_1")

    assert bundle.song_id == "autumn-sketch"
    assert bundle.active_section_id == "verse_1"
    assert bundle.section_bars == 4
    assert bundle.chords == ["Dm7", "Gm7", "A7", "Dm7"]

    by_inst = {s.instrument_id: s for s in bundle.polyphonic_stack}
    assert by_inst["bass"].register_band == "low"
    assert by_inst["violin"].register_band == "high"
    assert bundle.free_registers == ["mid"]
    assert "harmonic_comping" in bundle.missing_roles
    assert "rhythm_percussion" in bundle.missing_roles
    assert "bass_foundation" not in bundle.missing_roles
    assert "lead_melody" not in bundle.missing_roles

    # Default tail_bars without selected_region holds the last 2 bars (bars 3 and 4)
    assert bundle.manifest.included_bars == [3, 4]
    assert [b.bar for b in bundle.tail_bars["bass"]] == [3, 4]

    # With selected_region at bar 2, window is bars 1..3
    region = RegionSelection(section_id="verse_1", start_bar=2, end_bar=2)
    region_bundle = build_score_context(work, selected_region=region)
    assert region_bundle.manifest.included_bars == [1, 2, 3]
    assert [b.bar for b in region_bundle.tail_bars["violin"]] == [1, 2, 3]

    # Percussion classification check
    drum_band = classify_register_band(
        category="drums",
        role="rhythm_percussion",
        sounding_events=[parse_note_token("1.0:KICK:1.0:v95")],
    )
    assert drum_band == "percussion"


def test_propose_preview_and_apply_track_patch_preserves_outside_bars(tmp_path: Path) -> None:
    """`preview_patched_track` and `apply_proposal` splice only `start_bar..end_bar` and leave bars 1 and 4 untouched.

    Killed by: examples/music/src/uclone_x_music/proposals.py :: before = list(track.bars[: proposal.start_bar - 1])
    Becomes: before = [b.model_copy(update={"notes": ["1.0:REST:4.0"]}) for b in track.bars[: proposal.start_bar - 1]]
    """
    work = _setup_song(tmp_path)
    orig_loaded = work.read_track("verse_1", "violin")
    assert orig_loaded is not None
    orig_track, _ = orig_loaded

    patch_bars = [
        TrackBar(bar=2, chord="Gm7", notes=["1.0:Bb4:2.0:v92", "3.0:D5:2.0:v92"]),
        TrackBar(bar=3, chord="A7", notes=["1.0:C#5:2.0:v95", "3.0:E5:2.0:v95"]),
    ]
    proposal, prop_file = propose_track_patch(
        work,
        section_id="verse_1",
        instrument_id="violin",
        start_bar=2,
        end_bar=3,
        proposed_bars=patch_bars,
        summary="More active melody on bars 2-3",
    )
    assert proposal.id == "p001"
    assert proposal.status == "pending"
    assert prop_file.digest

    preview = preview_patched_track(orig_track, proposal)
    assert preview.bars[0] == orig_track.bars[0]
    assert preview.bars[1] == patch_bars[0]
    assert preview.bars[2] == patch_bars[1]
    assert preview.bars[3] == orig_track.bars[3]

    approved, _ = apply_proposal(
        work,
        "p001",
        decided_in="music_view",
        seen_digest=prop_file.digest,
    )
    assert approved.status == "approved"
    assert approved.decided_in == "music_view"

    after_loaded = work.read_track("verse_1", "violin")
    assert after_loaded is not None
    updated_track, _ = after_loaded
    assert updated_track.bars[0] == orig_track.bars[0]
    assert updated_track.bars[3] == orig_track.bars[3]
    assert updated_track.bars[1] == patch_bars[0]
    assert updated_track.bars[2] == patch_bars[1]


def test_apply_proposal_refuses_stale_target_digest_and_reject_records_reason(
    tmp_path: Path,
) -> None:
    """If the target track changed after a proposal was created, `apply_proposal` raises `SongChangedError`.

    Killed by: examples/music/src/uclone_x_music/proposals.py :: if proposal.target_digest is not None and current_digest != proposal.target_digest:
    Becomes: if False:
    """
    work = _setup_song(tmp_path)
    proposal, _ = propose_track_patch(
        work,
        section_id="verse_1",
        instrument_id="bass",
        start_bar=2,
        end_bar=2,
        proposed_bars=[TrackBar(bar=2, chord="Gm7", notes=["1.0:G2:2.0:v90", "3.0:D2:2.0:v85"])],
        summary="Walking bass in bar 2",
    )

    # Modify the underlying bass track on disk before applying the proposal
    loaded = work.read_track("verse_1", "bass")
    assert loaded is not None
    current_track, current_digest = loaded
    modified = current_track.model_copy(
        update={
            "bars": [
                TrackBar(bar=1, chord="Dm7", notes=["1.0:D2:2.0:v95", "3.0:A2:2.0:v90"]),
                *current_track.bars[1:],
            ]
        }
    )
    work.write_track(modified, expected_digest=current_digest, summary="Manual edit on bar 1")

    with pytest.raises(SongChangedError, match="changed on disk"):
        apply_proposal(work, proposal.id, decided_in="music_view")

    # Rejecting the stale proposal succeeds and records the reason
    rejected, _ = reject_proposal(
        work,
        proposal.id,
        decided_in="music_view",
        reason="Superseded by manual edit",
    )
    assert rejected.status == "rejected"
    assert rejected.rejection_reason == "Superseded by manual edit"

    # Cannot decide an already rejected proposal again
    with pytest.raises(MusicError, match="already 'rejected'"):
        apply_proposal(work, proposal.id, decided_in="music_view")


def test_propose_and_apply_instrument_change(tmp_path: Path) -> None:
    """`propose_instrument_change` stages an instrument addition or update and `apply_proposal` writes it.

    Killed by: examples/music/src/uclone_x_music/proposals.py :: elif proposal.kind == "instrument_change":
    Becomes: elif False:
    """
    work = _setup_song(tmp_path)
    new_inst = InstrumentEntry(
        id="piano_comp",
        name="Grand Piano",
        category="piano",
        role="harmonic_comping",
        min_pitch="A0",
        max_pitch="C8",
        volume=80,
    )
    prop, _ = propose_instrument_change(
        work,
        instrument=new_inst,
        summary="Add grand piano for mid-register comping",
    )
    assert prop.kind == "instrument_change"
    assert prop.target_digest is None

    # If the instrument is created behind the proposal's back, applying must fail with SongChangedError
    work.write_instrument(new_inst, expected_digest=None)
    with pytest.raises(SongChangedError, match="changed on disk"):
        apply_proposal(work, prop.id, decided_in="conversation")
