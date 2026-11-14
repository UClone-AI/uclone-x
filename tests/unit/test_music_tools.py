"""Unit tests for `uclone_x_music.tool` and `uclone_x_music.tools`."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.tools.models import ToolContext
from uclone_x_music import OPEN_SONG_KEY
from uclone_x_music.library import MusicError
from uclone_x_music.schemas import TempoChange, TrackBar
from uclone_x_music.start import MusicStartState
from uclone_x_music.tool import MusicLibraryParams, MusicLibraryTool
from uclone_x_music.tools import (
    MusicContextParams,
    MusicContextTool,
    MusicOrchestraParams,
    MusicOrchestraTool,
    MusicProposalParams,
    MusicProposalTool,
    MusicScoreParams,
    MusicScoreTool,
    MusicTrackParams,
    MusicTrackTool,
)
from uclone_x_music.work import SongWork


def _ctx(
    tmp_path: Path,
    *,
    room_id: str = "room-alpha",
    song_id: str | None = None,
    approved: bool = False,
) -> ToolContext:
    return ToolContext(
        agent_id="composer",
        session_id="sess_music",
        workspace_root=tmp_path,
        room_id=room_id,
        story_id=song_id,
        approved_by_person=approved,
    )


def _work(ctx: ToolContext) -> SongWork:
    return SongWork.for_context(ctx)


def _track_notes(ctx: ToolContext, section_id: str, instrument_id: str) -> list[list[str]]:
    """Every bar's notes, read from the track file: write results carry no bars (#2429)."""
    loaded = _work(ctx).read_track(section_id, instrument_id)
    assert loaded is not None
    return [list(b.notes) for b in loaded[0].bars]


@pytest.mark.asyncio
async def test_music_library_tool_create_open_list_take_over_and_close(tmp_path: Path) -> None:
    """`MusicLibraryTool` creates, opens, lists, takes over, and closes songs with `OPEN_SONG_KEY`.

    Killed by: examples/music/src/uclone_x_music/tool.py :: OPEN_SONG_KEY: record.id,
    Becomes: OPEN_SONG_KEY: None,
    """
    lib_tool = MusicLibraryTool()
    ctx_a = _ctx(tmp_path, room_id="room-a")

    created = await lib_tool.run(
        MusicLibraryParams(
            action="create",
            title="Starlight Sonata",
            style_profile="classical_strict",
            song_id="starlight-sonata",
        ),
        ctx_a,
    )
    assert created[OPEN_SONG_KEY] == "starlight-sonata"
    assert created["writable"] is True
    assert created["style_profile"] == "classical_strict"

    ctx_b = _ctx(tmp_path, room_id="room-b")
    opened_b = await lib_tool.run(
        MusicLibraryParams(action="open", song_id="starlight-sonata"),
        ctx_b,
    )
    assert opened_b[OPEN_SONG_KEY] == "starlight-sonata"
    assert opened_b["writable"] is False

    taken_b = await lib_tool.run(
        MusicLibraryParams(action="take_over", song_id="starlight-sonata"),
        ctx_b,
    )
    assert taken_b[OPEN_SONG_KEY] == "starlight-sonata"
    assert taken_b["writable"] is True
    assert taken_b["taken_from_another_conversation"] is True

    listed = await lib_tool.run(MusicLibraryParams(action="list"), ctx_b)
    assert len(listed["songs"]) == 1
    assert listed["songs"][0]["being_written_by"] == "this conversation"

    ctx_b_open = _ctx(tmp_path, room_id="room-b", song_id="starlight-sonata")
    closed = await lib_tool.run(MusicLibraryParams(action="close"), ctx_b_open)
    assert closed[OPEN_SONG_KEY] is None
    assert closed["closed"] == "starlight-sonata"


@pytest.mark.asyncio
async def test_music_score_and_orchestra_tools(tmp_path: Path) -> None:
    """`MusicScoreTool` and `MusicOrchestraTool` configure sections, tempo maps, and instruments.

    Killed by: examples/music/src/uclone_x_music/tools.py :: if params.style_profile is not None:
    Becomes: if False:
    """
    lib_tool = MusicLibraryTool()
    score_tool = MusicScoreTool()
    orch_tool = MusicOrchestraTool()

    await lib_tool.run(
        MusicLibraryParams(
            action="create",
            title="Neon Groove",
            song_id="neon-groove",
        ),
        _ctx(tmp_path, room_id="room-1"),
    )
    ctx = _ctx(tmp_path, room_id="room-1", song_id="neon-groove")

    meta_res = await score_tool.run(
        MusicScoreParams(
            action="set_metadata",
            tempo=94,
            time_signature="4/4",
            key_signature="F minor",
            style_profile="hiphop_groove",
        ),
        ctx,
    )
    assert meta_res["changed"] is True
    score_pair = _work(ctx).read_score()
    assert score_pair is not None and score_pair[0].tempo == 94
    axioms = _work(ctx).record().style_axioms
    assert axioms.profile == "hiphop_groove"
    assert axioms.require_kick_bass_lock is True

    # Repeating the same call must read as "nothing changed", not as a fresh success: a
    # model that is told every repeat succeeded keeps repeating it (27-step loop, 2026-11-01).
    repeat_res = await score_tool.run(
        MusicScoreParams(
            action="set_metadata",
            tempo=94,
            time_signature="4/4",
            key_signature="F minor",
            style_profile="hiphop_groove",
        ),
        ctx,
    )
    assert repeat_res["changed"] is False
    from uclone_x_music.start import read_start_state, write_start_state

    start_pair = read_start_state(_work(ctx))
    if start_pair is None:
        write_start_state(
            _work(ctx),
            MusicStartState(
                stage="layer_melody",
                active_section_id="verse_1",
                style_profile="pop_ballad",
                completed_layers=[],
                recommended_next="",
            ),
            expected_digest=None,
        )
    await score_tool.run(
        MusicScoreParams(action="set_metadata", style_profile="classical_strict"), ctx
    )
    synced = read_start_state(_work(ctx))
    assert synced is not None and synced[0].style_profile == "classical_strict"

    sec_res = await score_tool.run(
        MusicScoreParams(
            action="upsert_section",
            section_id="verse_1",
            title="Verse 1",
            kind="verse",
            bars=4,
            chords=["Fm7", "Bbm7", "C7", "Fm7"],
        ),
        ctx,
    )
    assert sec_res["created"] is True
    assert "4 bars" in sec_res["summary"]

    tmap_res = await score_tool.run(
        MusicScoreParams(
            action="set_tempo_map",
            section_id="verse_1",
            tempo_map=[TempoChange(bar=3, bpm=98, curve="accelerando")],
        ),
        ctx,
    )
    assert tmap_res["changed"] is True
    score_after = _work(ctx).read_score()
    assert score_after is not None
    assert len(next(s for s in score_after[0].sections if s.id == "verse_1").tempo_map) == 1

    await orch_tool.run(
        MusicOrchestraParams(
            action="add_instrument",
            instrument_id="sub_bass",
            name="Sub Bass",
            category="bass",
            role="bass_foundation",
            min_pitch="C1",
            max_pitch="C4",
            volume=88,
        ),
        ctx,
    )
    mix_res = await orch_tool.run(
        MusicOrchestraParams(
            action="adjust_mix",
            instrument_id="sub_bass",
            volume=92,
            pan=-0.2,
        ),
        ctx,
    )
    assert mix_res["changed"] is True
    inst = _work(ctx).read_instrument("sub_bass")
    assert inst is not None
    assert inst[0].volume == 92
    assert inst[0].pan == -0.2


@pytest.mark.asyncio
async def test_music_track_context_and_proposal_tools_workflow(tmp_path: Path) -> None:
    """`MusicTrackTool` writes, clears with full-bar rests, transposes, patches, and `MusicProposalTool` enforces approval.

    Killed by: examples/music/src/uclone_x_music/tools.py :: if not context.approved_by_person:
    Becomes: if False:
    """
    lib_tool = MusicLibraryTool()
    score_tool = MusicScoreTool()
    orch_tool = MusicOrchestraTool()
    track_tool = MusicTrackTool()
    context_tool = MusicContextTool()
    prop_tool = MusicProposalTool()

    await lib_tool.run(
        MusicLibraryParams(action="create", title="Chamber Piece", song_id="chamber-piece"),
        _ctx(tmp_path, room_id="room-1"),
    )
    ctx = _ctx(tmp_path, room_id="room-1", song_id="chamber-piece")

    await score_tool.run(
        MusicScoreParams(
            action="upsert_section",
            section_id="verse_1",
            title="Verse 1",
            bars=3,
            chords=["Dm", "Gm", "A7"],
        ),
        ctx,
    )
    await orch_tool.run(
        MusicOrchestraParams(
            action="add_instrument",
            instrument_id="cello",
            name="Cello",
            category="strings",
            role="bass_foundation",
            min_pitch="C2",
            max_pitch="A5",
        ),
        ctx,
    )

    bars = [
        TrackBar(bar=1, chord="Dm", notes=["1.0:D2:4.0:v85"]),
        TrackBar(bar=2, chord="Gm", notes=["1.0:G2:4.0:v85"]),
        TrackBar(bar=3, chord="A7", notes=["1.0:A2:4.0:v85"]),
    ]
    await track_tool.run(
        MusicTrackParams(
            action="write_track",
            section_id="verse_1",
            instrument_id="cello",
            bars=bars,
        ),
        ctx,
    )

    # Clear bar 2 -> replaces notes with a full-bar rest ("1.0:REST:4.0") while preserving bars 1 and 3
    cleared = await track_tool.run(
        MusicTrackParams(
            action="clear_bars",
            section_id="verse_1",
            instrument_id="cello",
            start_bar=2,
            end_bar=2,
        ),
        ctx,
    )
    assert cleared["changed_bars"] == [2]
    notes = _track_notes(ctx, "verse_1", "cello")
    assert notes[0] == ["1.0:D2:4.0:v85"]
    assert notes[1] == ["1.0:REST:4.0"]
    assert notes[2] == ["1.0:A2:4.0:v85"]

    # Transpose bar 3 up by 2 semitones (A2 -> B2)
    transposed = await track_tool.run(
        MusicTrackParams(
            action="transpose_bars",
            section_id="verse_1",
            instrument_id="cello",
            start_bar=3,
            end_bar=3,
            semitones=2,
        ),
        ctx,
    )
    assert transposed["changed_bars"] == [3]
    assert _track_notes(ctx, "verse_1", "cello")[2] == ["1.0:B2:4.0:v85"]

    # Propose a bar patch on bar 2
    patched = await track_tool.run(
        MusicTrackParams(
            action="patch_bars",
            section_id="verse_1",
            instrument_id="cello",
            start_bar=2,
            end_bar=2,
            bars=[TrackBar(bar=2, chord="Gm", notes=["1.0:G2:2.0:v90", "3.0:Bb2:2.0:v90"])],
            summary="Restore bar 2 with two half notes",
        ),
        ctx,
    )
    assert patched["proposed"] is True
    pid = patched["proposal_id"]

    # Context bundle lists the pending proposal and classifies cello in low register
    bundle = await context_tool.run(MusicContextParams(section_id="verse_1"), ctx)
    assert pid in bundle["manifest"]["pending_proposal_ids"]
    assert bundle["polyphonic_stack"][0]["register_band"] == "low"

    # Applying without approved_by_person=True is refused
    with pytest.raises(MusicError, match="not applied"):
        await prop_tool.run(
            MusicProposalParams(action="apply", proposal_id=pid),
            ctx,
        )

    # Applying with approved_by_person=True succeeds
    approved_ctx = _ctx(tmp_path, room_id="room-1", song_id="chamber-piece", approved=True)
    applied = await prop_tool.run(
        MusicProposalParams(action="apply", proposal_id=pid),
        approved_ctx,
    )
    assert applied["applied"] == pid

    read_after = await track_tool.run(
        MusicTrackParams(action="read", section_id="verse_1", instrument_id="cello"),
        ctx,
    )
    assert read_after["bars"][1]["notes"] == ["1.0:G2:2.0:v90", "3.0:Bb2:2.0:v90"]
    assert read_after["bars"][0]["notes"] == ["1.0:D2:4.0:v85"]


@pytest.mark.asyncio
async def test_music_proposal_and_orchestra_extend_bars_seed_intro_and_exclusive_solo(
    tmp_path: Path,
) -> None:
    """`MusicProposalTool(action="propose")` seeds new intro sections, extends bar ranges beyond 4 bars, and `MusicOrchestraTool` exclusively solos instruments via aliases.

    Killed by: examples/music/src/uclone_x_music/tools.py :: if updates.get("solo") is True:
    Becomes: if False:
    """
    lib_tool = MusicLibraryTool()
    score_tool = MusicScoreTool()
    orch_tool = MusicOrchestraTool()
    track_tool = MusicTrackTool()
    prop_tool = MusicProposalTool()

    await lib_tool.run(
        MusicLibraryParams(action="create", title="Nocturne Study", song_id="nocturne-study"),
        _ctx(tmp_path, room_id="room-1"),
    )
    ctx = _ctx(tmp_path, room_id="room-1", song_id="nocturne-study")

    await score_tool.run(
        MusicScoreParams(
            action="upsert_section",
            section_id="theme_a",
            title="Theme A",
            bars=2,
            chords=["Dm7", "Gm7"],
        ),
        ctx,
    )
    await orch_tool.run(
        MusicOrchestraParams(
            action="add_instrument",
            instrument_id="violin_solo",
            name="Solo Violin",
            category="strings",
            role="lead_melody",
            solo=True,
        ),
        ctx,
    )
    await orch_tool.run(
        MusicOrchestraParams(
            action="add_instrument",
            instrument_id="drum_kit",
            name="808 Drum Kit",
            category="drums",
            role="rhythm_percussion",
            solo=False,
        ),
        ctx,
    )
    await track_tool.run(
        MusicTrackParams(
            action="write_track",
            section_id="theme_a",
            instrument_id="violin_solo",
            bars=[
                TrackBar(bar=1, chord="Dm7", notes=["1.0:A4:2.0:v85", "3.0:D5:2.0:v85"]),
                TrackBar(bar=2, chord="Gm7", notes=["1.0:G4:2.0:v85", "3.0:Bb4:2.0:v85"]),
            ],
        ),
        ctx,
    )

    # 1. Proposing bars 3..4 on a 2-bar section automatically extends theme_a and its track to 4 bars
    ext_prop = await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="violin",
            start_bar=3,
            end_bar=4,
            summary="Extend violin melody into bars 3..4",
        ),
        ctx,
    )
    assert ext_prop["proposed"] is True
    violin = _work(ctx).read_track("theme_a", "violin_solo")
    assert violin is not None and len(violin[0].bars) == 4

    # 2. Proposing on a brand-new 'intro' section seeds the section and track automatically
    intro_prop = await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="intro",
            instrument_id="violin",
            start_bar=1,
            end_bar=2,
            summary="Gentle violin intro",
        ),
        ctx,
    )
    assert intro_prop["proposed"] is True
    assert intro_prop["section_id"] == "intro"

    # 3. Soloing 'drums' via alias resolves to 'drum_kit' and clears solo on 'violin_solo'
    await orch_tool.run(
        MusicOrchestraParams(
            action="adjust_mix",
            instrument_id="drums",
            solo=True,
        ),
        ctx,
    )
    listed = await orch_tool.run(MusicOrchestraParams(action="list"), ctx)
    by_id = {i["id"]: i for i in listed["instruments"]}
    assert by_id["drum_kit"]["solo"] is True
    assert by_id["violin_solo"]["solo"] is False
