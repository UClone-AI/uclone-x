"""Tests for the interactive music score view (`MusicView`) and its HTTP routes (`register_music_routes`).

Pins:
- `MusicView.show` compiles global bars, start/end times, and notes with both concert and written
  pitch for transposing instruments.
- `MusicView.approve` and `MusicView.reject` enforce the song's writing lease, composer busy state,
  proposal `seen_digest`, and target track/instrument `target_digest`.
- `MusicView.direct_edit_track` supports clearing bars to full-bar rests (`clear_bars`), muting bars
  (`mute_bars`), transposing bars (`transpose_bars`), and reverting to the previous snapshot (`revert`).
- `MusicView.update_mix` updates instrument volume, pan, muted, and solo state.
- `register_music_routes` exposes the REST endpoints and maps refusals cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.ui.extension_routes import ExtensionRouteContext
from uclone_x_music.library import MusicError, SongLibrary
from uclone_x_music.routes import register_music_routes
from uclone_x_music.schemas import (
    InstrumentEntry,
    MusicProposal,
    ScoreOutline,
    ScoreSection,
    TrackBar,
    TrackFile,
)
from uclone_x_music.view import (
    ComposerBusyError,
    MusicView,
    SongNotFoundError,
    SongNotWritableError,
)
from uclone_x_music.work import SongWork


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def rooms(tmp_path: Path) -> RoomService:
    return RoomService(RoomStore(tmp_path / "rooms"))


@pytest.fixture
def busy_rooms() -> set[str]:
    return set()


@pytest.fixture
def view(workspace: Path, rooms: RoomService, busy_rooms: set[str]) -> MusicView:
    return MusicView(workspace, rooms, turn_busy=busy_rooms.__contains__)


def _seed_song(workspace: Path, rooms: RoomService) -> tuple[str, str]:
    room_id = rooms.create("Studio Room").room_id
    library = SongLibrary(workspace)
    record = library.create(
        "Midnight Horizon",
        room_id,
        genre="jazz_swing",
        style_profile="jazz_swing",
        song_id="midnight-horizon",
    )
    work = SongWork(library, record.id, room_id)

    outline = ScoreOutline(
        tempo=120,
        time_signature="4/4",
        key_signature="C major",
        sections=[
            ScoreSection(
                id="verse1",
                title="Verse 1",
                kind="verse",
                bars=2,
                chords=["Cmaj7", "G7"],
            )
        ],
    )
    work.write_score(outline, expected_digest=None)

    piano = InstrumentEntry(
        id="piano",
        name="Grand Piano",
        category="piano",
        role="harmonic_comping",
        transposition_semitones=0,
        volume=85,
    )
    trumpet = InstrumentEntry(
        id="trumpet",
        name="Bb Trumpet",
        category="brass",
        role="lead_melody",
        transposition_semitones=-2,
        volume=80,
    )
    work.write_instrument(piano, expected_digest=None)
    work.write_instrument(trumpet, expected_digest=None)

    piano_track = TrackFile(
        track_id="verse1_piano",
        section_id="verse1",
        instrument_id="piano",
        time_signature="4/4",
        bars=[
            TrackBar(bar=1, chord="Cmaj7", notes=["1.0:C4:2.0:v80", "3.0:E4:2.0:v80"]),
            TrackBar(bar=2, chord="G7", notes=["1.0:G3:4.0:v85"]),
        ],
    )
    trumpet_track = TrackFile(
        track_id="verse1_trumpet",
        section_id="verse1",
        instrument_id="trumpet",
        time_signature="4/4",
        bars=[
            TrackBar(bar=1, chord="Cmaj7", notes=["1.0:Bb4:4.0:v90"]),
            TrackBar(bar=2, chord="G7", notes=["1.0:REST:4.0"]),
        ],
    )
    work.write_track(piano_track, expected_digest=None, summary="Initial piano")
    work.write_track(trumpet_track, expected_digest=None, summary="Initial trumpet")

    return record.id, room_id


def test_show_compiles_bars_notes_and_transposed_written_pitch(
    workspace: Path, rooms: RoomService, view: MusicView
) -> None:
    """Killed by: examples/music/src/uclone_x_music/view.py :: written_midi = max(0, min(127, concert_midi - transposition_semitones))
    Becomes: written_midi = concert_midi
    """
    song_id, room_id = _seed_song(workspace, rooms)

    shown = view.show(song_id)

    assert shown.song_id == "midnight-horizon"
    assert shown.title == "Midnight Horizon"
    assert shown.writable is True
    assert shown.holder == room_id
    assert shown.decide_code is None
    assert shown.total_bars == 2
    assert shown.duration_seconds == 4.0
    assert [(b.global_bar, b.chord, b.start_seconds, b.end_seconds) for b in shown.bars] == [
        (1, "Cmaj7", 0.0, 2.0),
        (2, "G7", 2.0, 4.0),
    ]
    trumpet_notes = [n for n in shown.notes if n.instrument_id == "trumpet" and n.bar == 1]
    assert len(trumpet_notes) == 1
    assert trumpet_notes[0].concert_pitch == "A#4"
    assert trumpet_notes[0].written_pitch == "C5"


def test_show_unknown_song_raises_song_not_found(view: MusicView) -> None:
    """Killed by: examples/music/src/uclone_x_music/view.py :: code: MusicRefusalCode = "song_gone"
    Becomes: code: MusicRefusalCode = "no_writer"
    """
    with pytest.raises(SongNotFoundError) as exc_info:
        view.show("non-existent-song")
    assert exc_info.value.code == "song_gone"


def test_approve_and_reject_proposals_check_digest_and_apply_changes(
    workspace: Path, rooms: RoomService, view: MusicView
) -> None:
    """Killed by: examples/music/src/uclone_x_music/view.py :: self._apply_proposal_target(work, proposal)
    Becomes: pass
    """
    song_id, room_id = _seed_song(workspace, rooms)
    work = SongWork(SongLibrary(workspace), song_id, room_id)
    track_loaded = work.read_track("verse1", "piano")
    assert track_loaded is not None
    _, track_digest = track_loaded

    prop1 = MusicProposal(
        id="p001",
        kind="track_patch",
        target_file="tracks/verse1_piano.yaml",
        target_digest=track_digest,
        section_id="verse1",
        instrument_id="piano",
        start_bar=2,
        end_bar=2,
        summary="Add passing tone in bar 2",
        proposed_bars=[TrackBar(bar=2, chord="G7", notes=["1.0:G3:2.0:v85", "3.0:B3:2.0:v85"])],
    )
    prop1_file = work.write_proposal(prop1, expected_digest=None)

    prop2 = MusicProposal(
        id="p002",
        kind="instrument_change",
        target_file="orchestra/piano.yaml",
        instrument_id="piano",
        summary="Increase piano volume",
        proposed_instrument=InstrumentEntry(
            id="piano",
            name="Grand Piano",
            category="piano",
            role="harmonic_comping",
            volume=95,
        ),
    )
    prop2_file = work.write_proposal(prop2, expected_digest=None)

    shown_before = view.show(song_id)
    assert [p.id for p in shown_before.proposals] == ["p001", "p002"]

    with pytest.raises(MusicError, match="changed after it was shown"):
        view.approve(song_id, "p001", seen_digest="wrong-digest")

    after_approve = view.approve(song_id, "p001", seen_digest=prop1_file.digest)
    assert [p.id for p in after_approve.proposals] == ["p002"]
    piano_track = next(t for t in after_approve.tracks if t.track_id == "verse1_piano")
    assert piano_track.bars[1].notes == ["1.0:G3:2.0:v85", "3.0:B3:2.0:v85"]
    assert piano_track.can_revert is True

    after_reject = view.reject(
        song_id, "p002", seen_digest=prop2_file.digest, reason="Keep original mix"
    )
    assert after_reject.proposals == []
    p2_loaded = work.read_proposal("p002")
    assert p2_loaded is not None
    saved_p2, _ = p2_loaded
    assert saved_p2.status == "rejected"
    assert saved_p2.rejection_reason == "Keep original mix"


def test_busy_or_missing_writer_refuses_mutations(
    workspace: Path, rooms: RoomService, busy_rooms: set[str], view: MusicView
) -> None:
    """Killed by: examples/music/src/uclone_x_music/view.py :: self._refuse_if_busy(holder)
    Becomes: pass
    """
    song_id, room_id = _seed_song(workspace, rooms)
    shown = view.show(song_id)
    piano_digest = next(t.digest for t in shown.tracks if t.track_id == "verse1_piano")

    busy_rooms.add(room_id)
    with pytest.raises(ComposerBusyError) as busy_exc:
        view.direct_edit_track(
            song_id,
            "verse1_piano",
            action="clear_bars",
            start_bar=1,
            end_bar=1,
            seen_digest=piano_digest,
        )
    assert busy_exc.value.code == "writer_busy"
    busy_rooms.discard(room_id)

    rooms.delete(room_id)
    assert view.show(song_id).decide_code == "writer_gone"
    with pytest.raises(SongNotWritableError) as gone_exc:
        view.update_mix(song_id, "piano", volume=50)
    assert gone_exc.value.code == "writer_gone"

    SongLibrary(workspace).release(song_id, room_id)
    assert view.show(song_id).decide_code == "no_writer"
    with pytest.raises(SongNotWritableError) as no_writer_exc:
        view.update_mix(song_id, "piano", volume=50)
    assert no_writer_exc.value.code == "no_writer"


def test_direct_edit_track_clear_mute_transpose_and_revert(
    workspace: Path, rooms: RoomService, view: MusicView
) -> None:
    """Killed by: examples/music/src/uclone_x_music/view.py :: rest_token = f"1.0:REST:{rest_dur}"
    Becomes: rest_token = "1.0:C4:4.0"
    """
    song_id, _ = _seed_song(workspace, rooms)
    initial = view.show(song_id)
    track0 = next(t for t in initial.tracks if t.track_id == "verse1_piano")

    # 1. Transpose bar 1 up by 12 semitones (C4 -> C5, E4 -> E5)
    after_transpose = view.direct_edit_track(
        song_id,
        "verse1_piano",
        action="transpose_bars",
        start_bar=1,
        end_bar=1,
        semitones=12,
        seen_digest=track0.digest,
    )
    track1 = next(t for t in after_transpose.tracks if t.track_id == "verse1_piano")
    assert track1.bars[0].notes == ["1.0:C5:2.0:v80", "3.0:E5:2.0:v80"]

    # 2. Mute bar 2
    after_mute = view.direct_edit_track(
        song_id,
        "verse1_piano",
        action="mute_bars",
        start_bar=2,
        end_bar=2,
        muted=True,
        seen_digest=track1.digest,
    )
    track2 = next(t for t in after_mute.tracks if t.track_id == "verse1_piano")
    assert track2.bars[1].muted is True

    # 3. Clear bar 1 to full-bar rest
    after_clear = view.direct_edit_track(
        song_id,
        "verse1_piano",
        action="clear_bars",
        start_bar=1,
        end_bar=1,
        seen_digest=track2.digest,
    )
    track3 = next(t for t in after_clear.tracks if t.track_id == "verse1_piano")
    assert track3.bars[0].notes == ["1.0:REST:4.0"]

    # 4. Revert back to previous revision (before clear_bars)
    after_revert = view.direct_edit_track(
        song_id,
        "verse1_piano",
        action="revert",
        seen_digest=track3.digest,
    )
    track4 = next(t for t in after_revert.tracks if t.track_id == "verse1_piano")
    assert track4.bars[0].notes == ["1.0:C5:2.0:v80", "3.0:E5:2.0:v80"]


def test_update_mix_persists_instrument_settings(
    workspace: Path, rooms: RoomService, view: MusicView
) -> None:
    """Killed by: examples/music/src/uclone_x_music/view.py :: updates["solo"] = solo
    Becomes: updates["solo"] = False
    """
    song_id, _ = _seed_song(workspace, rooms)

    updated = view.update_mix(
        song_id,
        "piano",
        volume=65,
        pan=-0.25,
        muted=True,
        solo=True,
    )
    piano = next(i for i in updated.instruments if i.id == "piano")
    assert (piano.volume, piano.pan, piano.muted, piano.solo) == (65, -0.25, True, True)


class _StubSessionManager:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir


class _StubStack:
    def __init__(self, workspace_dir: Path, service: RoomService) -> None:
        self._sm = _StubSessionManager(workspace_dir)
        self.service = service

    def session_manager(self) -> _StubSessionManager:
        return self._sm

    def turn_in_flight(self, _room_id: str) -> bool:
        return False

    def turn_unlanded(self, _room_id: str) -> bool:
        return False


class _StubPersonGate:
    def __init__(self) -> None:
        self.calls: int = 0

    def require(self, _request: Any) -> None:
        self.calls += 1


def test_routes_mount_and_serve_show_edit_mix_and_proposals(
    workspace: Path, rooms: RoomService
) -> None:
    """Killed by: examples/music/src/uclone_x_music/routes.py :: song_id, proposal_id, seen_digest=body.seen_digest, room_id=body.room_id
    Becomes: "no-such-song", proposal_id, seen_digest=body.seen_digest, room_id=body.room_id
    """
    song_id, room_id = _seed_song(workspace, rooms)
    work = SongWork(SongLibrary(workspace), song_id, room_id)
    track_loaded = work.read_track("verse1", "piano")
    assert track_loaded is not None
    _, track_digest = track_loaded
    prop = MusicProposal(
        id="p001",
        kind="track_patch",
        target_file="tracks/verse1_piano.yaml",
        target_digest=track_digest,
        section_id="verse1",
        instrument_id="piano",
        start_bar=1,
        end_bar=1,
        summary="Simple C4 whole note",
        proposed_bars=[TrackBar(bar=1, chord="Cmaj7", notes=["1.0:C4:4.0:v85"])],
    )
    prop_file = work.write_proposal(prop, expected_digest=None)

    app = FastAPI()
    person = _StubPersonGate()
    stack = _StubStack(workspace, rooms)
    ctx = ExtensionRouteContext(
        app=app,
        stack=stack,  # type: ignore[arg-type]
        person=person,  # type: ignore[arg-type]
        refuse_cross_origin=lambda _req: None,
    )
    register_music_routes(ctx)
    client = TestClient(app)

    res_get = client.get(f"/api/artifacts/library/songs/{song_id}")
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["song_id"] == song_id
    assert len(data["proposals"]) == 1

    res_mix = client.post(
        f"/api/artifacts/library/songs/{song_id}/instruments/piano/mix",
        json={"solo": True, "volume": 90},
    )
    assert res_mix.status_code == 200
    assert person.calls == 1

    res_approve = client.post(
        f"/api/artifacts/library/songs/{song_id}/proposals/p001/approve",
        json={"seen_digest": prop_file.digest},
    )
    assert res_approve.status_code == 200
    assert res_approve.json()["proposals"] == []
    assert person.calls == 2

    new_digest = next(
        t["digest"] for t in res_approve.json()["tracks"] if t["track_id"] == "verse1_piano"
    )
    res_edit = client.post(
        f"/api/artifacts/library/songs/{song_id}/tracks/verse1_piano/edit",
        json={
            "action": "clear_bars",
            "start_bar": 1,
            "end_bar": 1,
            "seen_digest": new_digest,
        },
    )
    assert res_edit.status_code == 200
    assert person.calls == 3

    res_missing = client.get("/api/artifacts/library/songs/missing-song")
    assert res_missing.status_code == 404
    assert res_missing.json()["code"] == "song_gone"


def test_approve_reconciles_expanded_section_bars_and_sequential_proposals(
    workspace: Path, rooms: RoomService, view: MusicView
) -> None:
    """When `score.yaml` expands a section (e.g. from 2 to 6 bars for an intro/outro) before tracks are rewritten, `MusicView.approve` and `MusicView.direct_edit_track` reconcile bar counts and rebase sibling pending proposals.

    Killed by: examples/music/src/uclone_x_music/view.py :: reconciled_track = _pad_track_bars_to(track, target_bars, chords, patch_by_bar)
    Becomes: reconciled_track = track
    """
    song_id, room_id = _seed_song(workspace, rooms)
    work = SongWork(SongLibrary(workspace), song_id, room_id)

    # Expand verse1 in score.yaml from 2 bars to 6 bars (like adding an intro/outro extension)
    score_loaded = work.read_score()
    assert score_loaded is not None
    outline, score_digest = score_loaded
    expanded_sec = ScoreSection(
        id="verse1",
        title="Extended Verse 1 with Intro",
        kind="intro",
        bars=6,
        chords=["Am7", "Dm7", "Cmaj7", "G7", "Fmaj7", "G7"],
    )
    work.write_score(
        outline.model_copy(update={"sections": [expanded_sec]}), expected_digest=score_digest
    )

    piano_loaded = work.read_track("verse1", "piano")
    assert piano_loaded is not None
    _, piano_digest = piano_loaded

    # Stage two sequential proposals (p001 on bar 1, p002 on bar 2) while piano_track still has 2 bars
    p001 = MusicProposal(
        id="p001",
        kind="track_patch",
        target_file="tracks/verse1_piano.yaml",
        target_digest=piano_digest,
        section_id="verse1",
        instrument_id="piano",
        start_bar=1,
        end_bar=1,
        summary="Gentle intro bar 1",
        proposed_bars=[TrackBar(bar=1, chord="Am7", notes=["1.0:A3:4.0:v72:legato"])],
    )
    p002 = MusicProposal(
        id="p002",
        kind="track_patch",
        target_file="tracks/verse1_piano.yaml",
        target_digest=piano_digest,
        section_id="verse1",
        instrument_id="piano",
        start_bar=2,
        end_bar=2,
        summary="Gentle intro bar 2",
        proposed_bars=[TrackBar(bar=2, chord="Dm7", notes=["1.0:D4:4.0:v74:legato"])],
    )
    f1 = work.write_proposal(p001, expected_digest=None)
    work.write_proposal(p002, expected_digest=None)

    # Approving p001 reconciles verse1_piano (and sibling verse1_trumpet) to 6 bars and rebases p002
    shown1 = view.approve(song_id, "p001", seen_digest=f1.digest)
    assert shown1.total_bars == 6
    piano_after_1 = next(t for t in shown1.tracks if t.track_id == "verse1_piano")
    assert len(piano_after_1.bars) == 6
    assert piano_after_1.bars[0].notes == ["1.0:A3:4.0:v72:legato"]

    # Approving p002 next using the fresh seen_digest succeeds cleanly
    p002_shown = next(p for p in shown1.proposals if p.id == "p002")
    shown2 = view.approve(song_id, "p002", seen_digest=p002_shown.seen_digest)
    assert shown2.proposals == []
    piano_after_2 = next(t for t in shown2.tracks if t.track_id == "verse1_piano")
    assert piano_after_2.bars[0].notes == ["1.0:A3:4.0:v72:legato"]
    assert piano_after_2.bars[1].notes == ["1.0:D4:4.0:v74:legato"]
    assert len(piano_after_2.bars) == 6
