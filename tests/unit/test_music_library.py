"""Unit tests for `uclone_x_music.library` and `uclone_x_music.work`."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x_music.library import (
    MusicError,
    SongChangedError,
    SongLibrary,
    SongReadOnlyError,
    UnknownSongError,
)
from uclone_x_music.schemas import (
    InstrumentEntry,
    ScoreOutline,
    ScoreSection,
    TrackBar,
    TrackFile,
)
from uclone_x_music.work import SongWork


def test_song_library_lease_and_take_over(tmp_path: Path) -> None:
    """Only the conversation holding the lease may write; another opens read-only until `take_over`.

    Killed by: examples/music/src/uclone_x_music/library.py :: if record.lease.holder == room_id:
    Becomes: if False:
    """
    lib = SongLibrary(tmp_path)
    created = lib.create("Midnight Breeze", "room-a", song_id="midnight-breeze")
    assert created.id == "midnight-breeze"
    assert created.lease is not None
    assert created.lease.holder == "room-a"

    assert lib.open("midnight-breeze", "room-a").writable is True
    opened_b = lib.open("midnight-breeze", "room-b")
    assert opened_b.writable is False

    work_b = SongWork(lib, "midnight-breeze", "room-b")
    with pytest.raises(SongReadOnlyError, match="read-only"):
        work_b.write_score(ScoreOutline(tempo=96), expected_digest=None)

    taken, prev = lib.take_over("midnight-breeze", "room-b")
    assert prev == "room-a"
    assert taken.lease is not None and taken.lease.holder == "room-b"

    written = work_b.write_score(ScoreOutline(tempo=100), expected_digest=None)
    assert written.digest


def test_song_work_digest_guard_and_track_history_revert(tmp_path: Path) -> None:
    """Writing with a stale digest raises `SongChangedError`, and `revert_track` restores history.

    Killed by: examples/music/src/uclone_x_music/library.py :: if current_digest != expected_digest:
    Becomes: if False:
    """
    lib = SongLibrary(tmp_path)
    lib.create("Nocturne", "room-1", song_id="nocturne-01")
    work = SongWork(lib, "nocturne-01", "room-1")

    outline = ScoreOutline(
        tempo=90,
        time_signature="4/4",
        key_signature="D minor",
        sections=[ScoreSection(id="verse_1", title="Verse 1", bars=2)],
    )
    score_file = work.write_score(outline, expected_digest=None)

    with pytest.raises(SongChangedError, match="already exists"):
        work.write_score(outline, expected_digest=None)

    with pytest.raises(SongChangedError, match="changed on disk"):
        work.write_score(outline, expected_digest="0" * 64)

    inst = InstrumentEntry(
        id="violin_solo",
        name="Solo Violin",
        category="strings",
        role="lead_melody",
        min_pitch="G3",
        max_pitch="E7",
    )
    work.write_instrument(inst, expected_digest=None)
    assert len(work.list_instruments()) == 1

    v1 = TrackFile(
        track_id="verse_1_violin_solo",
        section_id="verse_1",
        instrument_id="violin_solo",
        bars=[
            TrackBar(bar=1, notes=["1.0:D4:4.0:v85"]),
            TrackBar(bar=2, notes=["1.0:A4:4.0:v85"]),
        ],
    )
    f1 = work.write_track(v1, expected_digest=None, summary="Initial melody")

    v2 = TrackFile(
        track_id="verse_1_violin_solo",
        section_id="verse_1",
        instrument_id="violin_solo",
        bars=[
            TrackBar(bar=1, notes=["1.0:F4:2.0:v90", "3.0:G4:2.0:v90"]),
            TrackBar(bar=2, notes=["1.0:A4:4.0:v95"]),
        ],
    )
    f2 = work.write_track(v2, expected_digest=f1.digest, summary="Faster bar 1")

    restored, f3 = work.revert_track("verse_1_violin_solo", expected_digest=f2.digest)
    assert restored == v1
    assert f3.digest == f1.digest
    assert score_file.digest


def test_unknown_song_and_path_traversal_refused(tmp_path: Path) -> None:
    """Invalid song ids and path traversal inside a song folder are refused.

    The refusals are layered (the song-id pattern, the resolved-parent check, the `..` check
    and the path validator), so no single-line mutation reaches this test and it carries no
    kill declaration: removing one layer leaves the next one refusing.
    """
    lib = SongLibrary(tmp_path)
    with pytest.raises(UnknownSongError):
        lib.load("../outside")
    lib.create("Inside", "room-1", song_id="inside")
    for escape in ("../escape.yaml", "/etc/passwd", "tracks/../../escape.yaml"):
        with pytest.raises(MusicError):
            lib.read_file("inside", escape)
