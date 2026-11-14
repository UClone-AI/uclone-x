"""Unit tests for the music extension (`ucx_extension.py`), write guard (`songs/`), and `MusicTheoryAuditTool`."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.artifacts.kinds import FolderItemUnreadableError, UnknownFolderItemError
from uclone_x.errors import PlainRefusalError
from uclone_x.extensions import leased_folder_kinds, protected_roots, use_extensions
from uclone_x.tools.builtin.filesystem import FileWriteParams, FileWriteTool
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import create_default_registry
from uclone_x_music import OPEN_SONG_KEY, MusicLifecycleHook
from uclone_x_music.library import SongLibrary
from uclone_x_music.schemas import (
    InstrumentEntry,
    ScoreOutline,
    ScoreSection,
    TrackBar,
    TrackFile,
)
from uclone_x_music.tools import MusicTheoryAuditParams, MusicTheoryAuditTool
from uclone_x_music.ucx_extension import EXTENSION as MUSIC_EXTENSION
from uclone_x_music.ucx_extension import SongFolderKind
from uclone_x_music.work import SongWork


@pytest.mark.asyncio
async def test_general_file_write_is_refused_inside_songs_library(tmp_path: Path) -> None:
    """`FileWriteTool` refuses to write inside `songs/` when the music extension is active.

    Killed by: examples/music/src/uclone_x_music/ucx_extension.py :: protected_roots=(ProtectedRoot(dirname=SONGS_DIRNAME, refusal=_WRITE_REFUSAL),),
    Becomes: protected_roots=(),
    """
    with use_extensions([MUSIC_EXTENSION]):
        roots = protected_roots()
        assert any(r.dirname == "songs" for r in roots)

        tool = FileWriteTool()
        ctx = ToolContext(
            agent_id="composer",
            session_id="s1",
            workspace_root=tmp_path,
            room_id="room-1",
        )
        with pytest.raises(PlainRefusalError, match="song library"):
            tool.run(
                FileWriteParams(path="songs/midnight-breeze/score.yaml", content="tempo: 120\n"),
                ctx,
            )


def test_music_extension_registers_tools_and_leased_folder_kind(tmp_path: Path) -> None:
    """The music extension registers all 8 music tools and `SongFolderKind`.

    Killed by: examples/music/src/uclone_x_music/ucx_extension.py :: return _item(opening.song), opening.writable
    Becomes: return _item(opening.song), True
    """
    with use_extensions([MUSIC_EXTENSION]):
        registry = create_default_registry()
        for name in (
            "music_library",
            "music_start",
            "music_score",
            "music_orchestra",
            "music_track",
            "music_context",
            "music_proposal",
            "music_audit",
        ):
            assert registry.get(name) is not None

        kinds = leased_folder_kinds()
        assert len(kinds) == 1
        kind = kinds[0]
        assert isinstance(kind, SongFolderKind)
        assert kind.root == "songs"

        lib = SongLibrary(tmp_path)
        lib.create("Prelude in D", "room-a", song_id="prelude-d")
        song_folder = tmp_path / "songs" / "prelude-d"
        assert kind.is_item(song_folder) is True

        loaded = kind.load(tmp_path, "prelude-d")
        assert loaded.title == "Prelude in D"
        assert loaded.holder == "room-a"

        item_b, writable_b = kind.open(tmp_path, "prelude-d", "room-b")
        assert item_b.title == "Prelude in D"
        assert writable_b is False

        taken = kind.take_over(tmp_path, "prelude-d", "room-b")
        assert taken.holder == "room-b"

        assert kind.release(tmp_path, "prelude-d", "room-b") is True

        with pytest.raises(UnknownFolderItemError):
            kind.open(tmp_path, "no-such-song", "room-b")

        (song_folder / "song.yaml").write_text("not: [valid", encoding="utf-8")
        with pytest.raises(FolderItemUnreadableError):
            kind.load(tmp_path, "prelude-d")


@pytest.mark.asyncio
async def test_music_theory_audit_tool_and_lifecycle_hook(tmp_path: Path) -> None:
    """`MusicTheoryAuditTool` audits section tracks and `MusicLifecycleHook` updates `story_id`.

    Killed by: examples/music/src/uclone_x_music/tools.py :: passed = all(f.severity != "error" for f in findings)
    Becomes: passed = True
    Killed by: examples/music/src/uclone_x_music/__init__.py :: return context.model_copy(update={"story_id": song})
    Becomes: return context
    """
    lib = SongLibrary(tmp_path)
    lib.create("Audit Song", "room-1", song_id="audit-song", style_profile="classical_strict")
    work = SongWork(lib, "audit-song", "room-1")
    work.write_score(
        ScoreOutline(
            tempo=96,
            time_signature="4/4",
            key_signature="D minor",
            sections=[ScoreSection(id="verse_1", title="Verse 1", bars=2, chords=["Dm", "Am"])],
        ),
        expected_digest=None,
    )
    work.write_instrument(
        InstrumentEntry(
            id="violin_solo",
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
            track_id="verse_1_violin_solo",
            section_id="verse_1",
            instrument_id="violin_solo",
            bars=[
                TrackBar(bar=1, notes=["1.0:D3:4.0:v85"]),  # below G3 -> range_violation error
                TrackBar(bar=2, notes=["1.0:A5:4.0:v85"]),
            ],
        ),
        expected_digest=None,
    )

    audit_tool = MusicTheoryAuditTool()
    ctx = ToolContext(
        agent_id="composer",
        session_id="s1",
        workspace_root=tmp_path,
        room_id="room-1",
        story_id="audit-song",
    )
    res = await audit_tool.run(
        MusicTheoryAuditParams(
            section_id="verse_1",
            instrument_id="violin_solo",
            start_bar=2,
            end_bar=2,
        ),
        ctx,
    )
    assert res["passed"] is False
    codes = {f["code"] for f in res["findings"]}
    assert "range_violation" in codes
    assert "boundary_jump" in codes

    hook = MusicLifecycleHook()
    rec = ToolExecutionRecord(
        tool_name="music_library",
        arguments={"action": "open", "song_id": "new-song"},
        output={OPEN_SONG_KEY: "new-song"},
        status=ToolResultStatus.SUCCESS,
        duration_ms=1.0,
        opens_story=True,
    )
    updated_ctx = hook.after_tool_step([rec], ctx)
    assert updated_ctx.story_id == "new-song"
