"""End-to-End AI-to-Editor Scenario Tests (Scenarios S1..S11) for `uclone_x_music`.

Verifies that every AI editing instruction pattern observed in live sessions — underfilled or
overfilled bar beat sums, gentle intro vs ornate climax intent, `instrument_id="all"` multi-stem
proposals, structural bar `insert_bars`/`append_bars`/`delete_bars`/`duplicate_bars`, articulation
synonyms (`:trill`, `:fermata`, `:sustain`, `:marcato`), sequential proposal digest reconciliation,
and direct `set_bar_notes` / `approve_all` / `structure` HTTP routes — reflects cleanly in
`MusicView` without silent refusal or destructive fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.tools.models import ToolContext
from uclone_x.ui.extension_routes import ExtensionRouteContext
from uclone_x_music.library import MusicError, SongLibrary
from uclone_x_music.routes import register_music_routes
from uclone_x_music.schemas import TrackBar, parse_note_token
from uclone_x_music.section_turn import MusicTurnHook
from uclone_x_music.tool import MusicLibraryParams, MusicLibraryTool
from uclone_x_music.tools import (
    MusicOrchestraParams,
    MusicOrchestraTool,
    MusicProposalParams,
    MusicProposalTool,
    MusicScoreParams,
    MusicScoreTool,
    MusicTrackParams,
    MusicTrackTool,
)
from uclone_x_music.view import MusicView
from uclone_x_music.work import SongWork


def _ctx(
    tmp_path: Path,
    *,
    room_id: str = "room-scenario",
    song_id: str | None = None,
    approved: bool = False,
) -> ToolContext:
    return ToolContext(
        agent_id="composer",
        session_id="sess_scenario",
        workspace_root=tmp_path,
        room_id=room_id,
        story_id=song_id,
        approved_by_person=approved,
    )


def _proposed_notes(ctx: ToolContext, res: dict[str, Any], bar_index: int = 0) -> list[str]:
    """The notes a staged proposal holds, read from its file: results carry no bars (#2429)."""
    loaded = SongWork.for_context(ctx).read_proposal(str(res["proposal_id"]))
    assert loaded is not None
    return list(loaded[0].proposed_bars[bar_index].notes)


def _section(ctx: ToolContext, section_id: str = "theme_a") -> Any:
    loaded = SongWork.for_context(ctx).read_score()
    assert loaded is not None
    return next(s for s in loaded[0].sections if s.id == section_id)


async def _seed_four_stem_song(tmp_path: Path, song_id: str = "midnight-nocturne") -> ToolContext:
    """Create a 4-bar, 4-stem song (`violin_solo`, `grand_piano`, `bass_synth`, `drum_kit`)."""
    lib_tool = MusicLibraryTool()
    score_tool = MusicScoreTool()
    orch_tool = MusicOrchestraTool()
    track_tool = MusicTrackTool()

    await lib_tool.run(
        MusicLibraryParams(action="create", title="Midnight Nocturne", song_id=song_id),
        _ctx(tmp_path),
    )
    ctx = _ctx(tmp_path, song_id=song_id)

    await score_tool.run(
        MusicScoreParams(
            action="upsert_section",
            section_id="theme_a",
            title="Theme A",
            kind="verse",
            bars=4,
            chords=["Dm7", "Gm7", "A7", "Dm7"],
        ),
        ctx,
    )
    stems: list[tuple[str, str, str, str, list[str]]] = [
        (
            "violin_solo",
            "Solo Violin",
            "strings",
            "lead_melody",
            ["1.0:A4:2.0:v85:legato", "3.0:D5:2.0:v85:tenuto"],
        ),
        (
            "grand_piano",
            "Grand Piano",
            "piano",
            "harmonic_comping",
            ["1.0:F4:2.0:v78:legato", "3.0:A4:2.0:v76:tenuto"],
        ),
        (
            "bass_synth",
            "Synth Bass",
            "bass",
            "bass_foundation",
            ["1.0:D2:2.0:v90", "3.0:A2:2.0:v86"],
        ),
        (
            "drum_kit",
            "808 Drum Kit",
            "drums",
            "rhythm_percussion",
            ["1.0:KICK:1.0:v95", "2.0:SNARE:1.0:v90", "3.0:KICK:1.0:v92", "4.0:SNARE:1.0:v90"],
        ),
    ]
    for inst_id, name, cat, role, notes in stems:
        await orch_tool.run(
            MusicOrchestraParams(
                action="add_instrument",
                instrument_id=inst_id,
                name=name,
                category=cat,  # type: ignore[arg-type]
                role=role,  # type: ignore[arg-type]
            ),
            ctx,
        )
        await track_tool.run(
            MusicTrackParams(
                action="write_track",
                section_id="theme_a",
                instrument_id=inst_id,
                bars=[
                    TrackBar(bar=1, chord="Dm7", notes=list(notes)),
                    TrackBar(bar=2, chord="Gm7", notes=list(notes)),
                    TrackBar(bar=3, chord="A7", notes=list(notes)),
                    TrackBar(bar=4, chord="Dm7", notes=list(notes)),
                ],
            ),
            ctx,
        )
    return ctx


@pytest.mark.asyncio
async def test_scenario_s1_underfilled_bar_auto_padded_with_rest(tmp_path: Path) -> None:
    """Scenario S1: AI passes underfilled 2-beat bar to `music_track(patch_bars)`; auto-pads trailing `REST` and reflects in `MusicView`.

    Killed by: examples/music/src/uclone_x_music/schemas.py :: if max_end < bar_end - 1e-3:
    Becomes: if False:
    """
    ctx = await _seed_four_stem_song(tmp_path)
    track_tool = MusicTrackTool()
    view = MusicView(tmp_path)

    res = await track_tool.run(
        MusicTrackParams(
            action="patch_bars",
            section_id="theme_a",
            instrument_id="bass",
            start_bar=1,
            end_bar=1,
            bars=[
                TrackBar(
                    bar=1,
                    chord="Am7",
                    notes=["1.0:A2:1.5:v60:legato", "2.5:D3:0.5:v55:legato"],
                )
            ],
            summary="Soft bass intro motif on bar 1",
        ),
        ctx,
    )
    assert res["proposed"] is True
    proposed_notes = _proposed_notes(ctx, res)
    assert proposed_notes == [
        "1.0:A2:1.5:v60:legato",
        "2.5:D3:0.5:v55:legato",
        "3.0:REST:2.0",
    ]

    shown_before = view.show("midnight-nocturne")
    shown_after = view.approve(
        "midnight-nocturne",
        res["proposal_id"],
        seen_digest=shown_before.proposals[0].seen_digest,
    )
    bass_track = next(t for t in shown_after.tracks if t.instrument_id == "bass_synth")
    assert bass_track.bars[0].notes == [
        "1.0:A2:1.5:v60:legato",
        "2.5:D3:0.5:v55:legato",
        "3.0:REST:2.0",
    ]


@pytest.mark.asyncio
async def test_scenario_s2_overfilled_bar_clamped_at_bar_boundary(tmp_path: Path) -> None:
    """Scenario S2: AI passes overfilled bar (`4.0:E5:2.0:v87:tenuto`) to `music_proposal(propose)`; clamps to `1.0` beat instead of falling back to 16th-note staccato.

    Killed by: examples/music/src/uclone_x_music/schemas.py :: dur = min(ev.duration, max_dur)
    Becomes: dur = ev.duration
    """
    ctx = await _seed_four_stem_song(tmp_path)
    prop_tool = MusicProposalTool()

    res = await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="violin",
            start_bar=1,
            end_bar=1,
            bars=[
                TrackBar(
                    bar=1,
                    chord="Dm7",
                    notes=[
                        "1.0:F4:1.0:v80:legato",
                        "2.0:A4:2.0:v82:legato",
                        "4.0:E5:2.0:v87:tenuto",
                    ],
                )
            ],
            summary="Expressive violin phrase on bar 1",
        ),
        ctx,
    )
    assert res["proposed"] is True
    notes = _proposed_notes(ctx, res)
    assert notes == [
        "1.0:F4:1.0:v80:legato",
        "2.0:A4:2.0:v82:legato",
        "4.0:E5:1.0:v87:tenuto",
    ]


@pytest.mark.asyncio
async def test_scenario_s3_gentle_intro_intent_synthesizes_soft_sustained_notes(
    tmp_path: Path,
) -> None:
    """Scenario S3: AI calls `music_proposal(propose)` without `bars` and with `intent='gentle_intro'`; synthesizes soft sustained notes instead of 16th-note staccato.

    The intent is the model's structured argument. The same words in `summary` alone choose
    nothing: without `intent` the default ornate variation is generated.

    Killed by: examples/music/src/uclone_x_music/tools.py :: synth_intent: ArrangementIntent = params.intent or "climax_ornate"
    Becomes: synth_intent: ArrangementIntent = "climax_ornate"
    """
    ctx = await _seed_four_stem_song(tmp_path)
    prop_tool = MusicProposalTool()

    res = await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="piano",
            start_bar=1,
            end_bar=2,
            summary="Soft opening chords",
            intent="gentle_intro",
        ),
        ctx,
    )
    assert res["proposed"] is True
    bar1_notes = _proposed_notes(ctx, res)
    parsed = [parse_note_token(t) for t in bar1_notes]
    assert all(ev.duration >= 1.5 for ev in parsed)
    assert all(ev.articulation in ("legato", "tenuto") for ev in parsed)

    words_only = await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="violin_solo",
            start_bar=3,
            end_bar=4,
            summary="Gentle intro, soft and sustained, 도입부를 부드럽고 잔잔하게",
        ),
        ctx,
    )
    words_notes = [parse_note_token(t) for t in _proposed_notes(ctx, words_only)]
    assert any(ev.duration < 1.0 for ev in words_notes)


@pytest.mark.asyncio
async def test_scenario_s4_instrument_all_stages_proposals_for_every_stem_and_approve_all_applies_them(
    tmp_path: Path,
) -> None:
    """Scenario S4: `music_proposal(propose, instrument_id='all')` stages proposals for all 4 stems, and `MusicView.approve_all` applies all of them atomically.

    Killed by: examples/music/src/uclone_x_music/tools.py :: if cleaned.casefold() in {"all", "*", "ensemble", "full"}:
    Becomes: if False:
    """
    ctx = await _seed_four_stem_song(tmp_path)
    prop_tool = MusicProposalTool()
    view = MusicView(tmp_path)

    res = await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="all",
            start_bar=1,
            end_bar=2,
            summary="Gentle intro across all instruments",
        ),
        ctx,
    )
    assert res["proposed"] is True
    assert len(res["proposal_ids"]) == 4

    shown_pending = view.show("midnight-nocturne")
    assert len(shown_pending.proposals) == 4

    shown_approved = view.approve_all("midnight-nocturne")
    assert len(shown_approved.proposals) == 0


@pytest.mark.asyncio
async def test_scenario_s5_insert_bars_prepends_intro_and_shifts_existing_bars(
    tmp_path: Path,
) -> None:
    """Scenario S5: `music_score(action='insert_bars', at_bar=1, count=2)` prepends 2 intro bars and shifts existing bars 1..4 to 3..6 across all tracks.

    Killed by: examples/music/src/uclone_x_music/arranger.py :: bars=sec.bars + count,
    Becomes: bars=sec.bars,
    """
    ctx = await _seed_four_stem_song(tmp_path)
    score_tool = MusicScoreTool()
    view = MusicView(tmp_path)

    res = await score_tool.run(
        MusicScoreParams(
            action="insert_bars",
            section_id="theme_a",
            at_bar=1,
            count=2,
            chords=["Am7", "Dm7"],
            intent="gentle_intro",
        ),
        ctx,
    )
    assert res["changed"] is True
    assert _section(ctx).bars == 6
    assert _section(ctx).chords[:2] == ["Am7", "Dm7"]

    shown = view.show("midnight-nocturne")
    assert shown.total_bars == 6
    for track in shown.tracks:
        assert len(track.bars) == 6
        assert [b.bar for b in track.bars] == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_scenario_s6_append_bars_and_delete_duplicate_bars_keep_section_and_tracks_in_sync(
    tmp_path: Path,
) -> None:
    """Scenario S6: `append_bars`, `duplicate_bars`, and `delete_bars` keep `score.yaml` and all tracks synchronized in bar count.

    Killed by: examples/music/src/uclone_x_music/arranger.py :: new_bars_count = sec.bars - remove_count
    Becomes: new_bars_count = sec.bars
    """
    ctx = await _seed_four_stem_song(tmp_path)
    score_tool = MusicScoreTool()
    view = MusicView(tmp_path)

    # 1. Append 4 continuation bars (4 -> 8)
    app_res = await score_tool.run(
        MusicScoreParams(
            action="append_bars",
            section_id="theme_a",
            count=4,
            intent="outro_cadence",
        ),
        ctx,
    )
    assert app_res["changed"] is True
    assert _section(ctx).bars == 8
    assert view.show("midnight-nocturne").total_bars == 8

    # 2. Duplicate bars 1..2 (8 -> 10)
    dup_res = await score_tool.run(
        MusicScoreParams(
            action="duplicate_bars",
            section_id="theme_a",
            start_bar=1,
            end_bar=2,
        ),
        ctx,
    )
    assert dup_res["changed"] is True
    assert _section(ctx).bars == 10
    assert view.show("midnight-nocturne").total_bars == 10

    # 3. Delete bars 9..10 (10 -> 8)
    del_res = await score_tool.run(
        MusicScoreParams(
            action="delete_bars",
            section_id="theme_a",
            start_bar=9,
            end_bar=10,
        ),
        ctx,
    )
    assert del_res["changed"] is True
    assert _section(ctx).bars == 8
    shown = view.show("midnight-nocturne")
    assert shown.total_bars == 8
    assert all(len(t.bars) == 8 for t in shown.tracks)


def test_scenario_s7_articulation_aliases_trill_fermata_marcato_parse_cleanly() -> None:
    """Scenario S7: `:trill`, `:fermata`, `:sustain`, `:marcato`, `:glissando`, `:muted` map to canonical articulations.

    Killed by: examples/music/src/uclone_x_music/schemas.py :: elif extra.lower() in _ARTICULATION_ALIASES:
    Becomes: elif False:
    """
    assert parse_note_token("1.0:E5:2.0:v88:trill").articulation == "vibrato"
    assert parse_note_token("1.0:D5:4.0:v84:fermata").articulation == "tenuto"
    assert parse_note_token("1.0:A4:2.0:v80:sustain").articulation == "tenuto"
    assert parse_note_token("1.0:C4:1.0:v96:marcato").articulation == "accent"
    assert parse_note_token("1.0:G4:1.0:v82:glissando").articulation == "legato"
    assert parse_note_token("1.0:E3:0.5:v75:muted").articulation == "staccato"


@pytest.mark.asyncio
async def test_scenario_s8_sequential_same_track_proposals_reconcile_digest_on_approve(
    tmp_path: Path,
) -> None:
    """Scenario S8: Two pending proposals on the same track: approving one in `MusicView` moves the other's `target_digest`, so it still applies.

    Killed by: examples/music/src/uclone_x_music/view.py :: other_prop.model_copy(update={"target_digest": written_track.digest}),
    Becomes: other_prop.model_copy(update={"target_digest": "stale"}),
    """
    ctx = await _seed_four_stem_song(tmp_path)
    prop_tool = MusicProposalTool()
    view = MusicView(tmp_path)

    p1 = await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="violin",
            start_bar=1,
            end_bar=2,
            summary="Gentle intro violin bars 1..2",
        ),
        ctx,
    )
    p2 = await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="violin",
            start_bar=3,
            end_bar=4,
            summary="Climax ornate violin bars 3..4",
        ),
        ctx,
    )

    shown1 = view.show("midnight-nocturne")
    p1_view = next(p for p in shown1.proposals if p.id == p1["proposal_id"])
    shown2 = view.approve("midnight-nocturne", p1_view.id, seen_digest=p1_view.seen_digest)

    assert [p.id for p in shown2.proposals] == [p2["proposal_id"]]

    # The second proposal is applied in conversation, which checks its `target_digest`
    # against the track: approving the first must have moved it to the written track.
    approved_ctx = ctx.model_copy(update={"approved_by_person": True})
    applied = await prop_tool.run(
        MusicProposalParams(action="apply", proposal_id=str(p2["proposal_id"])), approved_ctx
    )
    assert applied["applied"] == p2["proposal_id"]
    assert len(view.show("midnight-nocturne").proposals) == 0


@pytest.mark.asyncio
async def test_scenario_s9_upsert_section_bar_increase_extends_all_tracks_for_music_view(
    tmp_path: Path,
) -> None:
    """Scenario S9: Increasing a section's `bars` via `upsert_section` automatically extends all tracks in that section.

    Killed by: examples/music/src/uclone_x_music/tools.py :: _extend_section_and_tracks_to_bar(work, sec_id, new_sec.bars)
    Becomes: pass
    """
    ctx = await _seed_four_stem_song(tmp_path)
    score_tool = MusicScoreTool()
    view = MusicView(tmp_path)

    await score_tool.run(
        MusicScoreParams(
            action="upsert_section",
            section_id="theme_a",
            bars=8,
            chords=["Dm7", "Gm7", "A7", "Dm7", "Bbmaj7", "Gm7", "A7", "Dm7"],
        ),
        ctx,
    )
    shown = view.show("midnight-nocturne")
    assert shown.total_bars == 8
    assert all(len(t.bars) == 8 for t in shown.tracks)


@pytest.mark.asyncio
async def test_scenario_s10_DirectEdit_set_bar_notes_normalizes_and_updates_chord(
    tmp_path: Path,
) -> None:
    """Scenario S10: `MusicView.direct_edit_track(action='set_bar_notes')` normalizes underfilled notes and updates the bar chord in `score.yaml`.

    Killed by: examples/music/src/uclone_x_music/view.py :: elif action == "set_bar_notes":
    Becomes: elif False:
    """
    await _seed_four_stem_song(tmp_path)
    view = MusicView(tmp_path)
    shown = view.show("midnight-nocturne")
    piano_track = next(t for t in shown.tracks if t.instrument_id == "grand_piano")

    updated = view.direct_edit_track(
        "midnight-nocturne",
        piano_track.track_id,
        action="set_bar_notes",
        start_bar=1,
        end_bar=1,
        chord="Fmaj7",
        notes=["1.0:F4:1.5:v84:legato", "2.5:A4:0.5:v80:trill"],
        seen_digest=piano_track.digest,
    )
    updated_piano = next(t for t in updated.tracks if t.instrument_id == "grand_piano")
    assert updated_piano.bars[0].chord == "Fmaj7"
    assert updated_piano.bars[0].notes == [
        "1.0:F4:1.5:v84:legato",
        "2.5:A4:0.5:v80:vibrato",
        "3.0:REST:2.0",
    ]
    assert updated.bars[0].chord == "Fmaj7"


@pytest.mark.asyncio
async def test_scenario_s11_http_routes_structure_approve_all_and_set_bar_notes(
    tmp_path: Path,
) -> None:
    """Scenario S11: HTTP routes `/structure`, `/proposals/approve-all`, and `/tracks/{id}/edit` (`set_bar_notes`) execute end-to-end.

    Killed by: examples/music/src/uclone_x_music/routes.py :: song_id, room_id=body.room_id if body is not None else None
    Becomes: "no-such-song", room_id=None
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rooms = RoomService(RoomStore(tmp_path / "rooms"))
    room_id = rooms.create("Composer Room").room_id

    await _seed_four_stem_song(workspace, song_id="route-song")
    # Transfer lease to created room_id
    lib = SongLibrary(workspace)
    lib.take_over("route-song", room_id)

    prop_tool = MusicProposalTool()
    await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="all",
            start_bar=1,
            end_bar=2,
            summary="Gentle intro across all stems",
        ),
        _ctx(workspace, room_id=room_id, song_id="route-song"),
    )

    app = FastAPI()

    class _FakeSessionManager:
        workspace_dir = workspace

    class _FakeStack:
        service = rooms

        def session_manager(self) -> Any:
            return _FakeSessionManager()

        def turn_in_flight(self, _room_id: str) -> bool:
            return False

        def turn_unlanded(self, _room_id: str) -> bool:
            return False

    class _FakePerson:
        def require(self, _request: Any) -> None:
            return None

    register_music_routes(
        ExtensionRouteContext(
            app=app,
            stack=_FakeStack(),  # type: ignore[arg-type]
            person=_FakePerson(),  # type: ignore[arg-type]
            refuse_cross_origin=lambda _req: None,
        )
    )
    client = TestClient(app)

    # 1. Approve all 4 proposals via HTTP route
    resp_all = client.post("/api/artifacts/library/songs/route-song/proposals/approve-all")
    assert resp_all.status_code == 200
    assert len(resp_all.json()["proposals"]) == 0

    # 2. Insert 2 intro bars via HTTP `/structure` route
    resp_struct = client.post(
        "/api/artifacts/library/songs/route-song/structure",
        json={
            "action": "insert_bars",
            "section_id": "theme_a",
            "at_bar": 1,
            "count": 2,
            "intent": "gentle_intro",
        },
    )
    assert resp_struct.status_code == 200
    assert resp_struct.json()["total_bars"] == 6


@pytest.mark.asyncio
async def test_scenario_s13_swan_lake_intro_across_time_signature_change(
    tmp_path: Path,
) -> None:
    """Scenario S13: Changing `time_signature` from 4/4 to 3/4, upserting `swan_intro` (`kind='intro'`), and patching `violin_solo` succeeds: the patch's 4/4-shaped bars are normalized to 3/4 instead of overflowing the bar.

    Killed by: examples/music/src/uclone_x_music/tools.py :: norm_bar = normalize_track_bar(tmpl, time_signature)
    Becomes: norm_bar = tmpl
    """
    ctx = await _seed_four_stem_song(tmp_path)
    score_tool = MusicScoreTool()
    track_tool = MusicTrackTool()
    view = MusicView(tmp_path)

    # 1. AI switches time signature from 4/4 to 3/4
    await score_tool.run(
        MusicScoreParams(action="set_metadata", tempo=68, time_signature="3/4"),
        ctx,
    )

    # 2. AI upserts a new intro section `swan_intro`
    await score_tool.run(
        MusicScoreParams(
            action="upsert_section",
            section_id="swan_intro",
            title="Swan Lake Intro",
            kind="intro",
            bars=4,
            chords=["Dm7", "Bm7", "F#m7", "Dmaj7"],
        ),
        ctx,
    )

    # 3. AI patches `swan_intro` violin_solo directly with Swan Lake motif notes (including 4-beat notes that need 3/4 normalization)
    patch_res = await track_tool.run(
        MusicTrackParams(
            action="patch_bars",
            section_id="swan_intro",
            instrument_id="violin_solo",
            start_bar=1,
            end_bar=4,
            propose=False,
            bars=[
                TrackBar(
                    bar=1,
                    chord="Dm7",
                    notes=[
                        "1.0:D5:1.5:v72:legato",
                        "2.5:C#5:0.5:v68:tenuto",
                        "3.0:B4:1.0:v65:legato",
                        "4.0:A4:1.0:v62:tenuto",
                    ],
                ),
                TrackBar(
                    bar=2,
                    chord="Bm7",
                    notes=[
                        "1.0:B4:2.0:v70:legato",
                        "3.0:F#4:2.0:v64:tenuto",
                    ],
                ),
                TrackBar(
                    bar=3,
                    chord="F#m7",
                    notes=[
                        "1.0:F#4:1.5:v68:legato",
                        "2.5:A4:0.5:v66:tenuto",
                        "3.0:C#5:2.0:v72:trill",
                    ],
                ),
                TrackBar(
                    bar=4,
                    chord="Dmaj7",
                    notes=[
                        "1.0:D5:4.0:v75:fermata",
                    ],
                ),
            ],
        ),
        ctx,
    )
    assert patch_res["proposed"] is False

    shown = view.show("midnight-nocturne")
    assert shown.sections[0].id == "swan_intro"
    assert shown.total_bars == 8
    swan_tracks = [t for t in shown.tracks if t.section_id == "swan_intro"]
    assert len(swan_tracks) == 4


@pytest.mark.asyncio
async def test_scenario_s14_solo_mute_and_apply_go_through_explicit_tool_calls(
    tmp_path: Path,
) -> None:
    """Scenario S14: solo, unsolo and applying a proposal happen through the model's tool calls.

    `music_orchestra(adjust_mix)` sets solo; `music_proposal(apply)` applies only with the
    person's approval on the call (`approved_by_person`), and refuses without it.

    Killed by: examples/music/src/uclone_x_music/tools.py :: if not context.approved_by_person:
    Becomes: if False:
    """
    ctx = await _seed_four_stem_song(tmp_path)
    orch_tool = MusicOrchestraTool()
    prop_tool = MusicProposalTool()
    view = MusicView(tmp_path)

    await orch_tool.run(
        MusicOrchestraParams(action="adjust_mix", instrument_id="drum_kit", solo=True), ctx
    )
    inst_map = {i.id: i for i in view.show("midnight-nocturne").instruments}
    assert inst_map["drum_kit"].solo is True
    assert inst_map["violin_solo"].solo is False

    await orch_tool.run(
        MusicOrchestraParams(action="adjust_mix", instrument_id="drum_kit", solo=False), ctx
    )
    assert all(not i.solo for i in view.show("midnight-nocturne").instruments)

    proposed = await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="violin_solo",
            start_bar=1,
            end_bar=2,
            summary="Gentle intro",
            intent="gentle_intro",
        ),
        ctx,
    )
    proposal_id = str(proposed["proposal_id"])
    with pytest.raises(MusicError):
        await prop_tool.run(MusicProposalParams(action="apply", proposal_id=proposal_id), ctx)
    assert len(view.show("midnight-nocturne").proposals) == 1

    approved_ctx = ctx.model_copy(update={"approved_by_person": True})
    await prop_tool.run(MusicProposalParams(action="apply", proposal_id=proposal_id), approved_ctx)
    assert len(view.show("midnight-nocturne").proposals) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Don't approve yet",
        "적용 말고 다시 제안해줘",
        "적용하라",
        "approve all",
        "피아노만 빼고",
        "드럼만 연주해줘",
        "전체 악기 연주해줘",
        "mute the bass",
    ],
)
async def test_a_user_message_alone_changes_no_proposal_and_no_mix(
    tmp_path: Path, message: str
) -> None:
    """The turn hook reads no meaning from the person's words: a message applies nothing.

    Whatever the message says about approving, applying, soloing or muting, the pending
    proposal stays pending and every instrument's solo and mute stay as they were. Only the
    model's tool calls change them.

    Killed by: examples/music/src/uclone_x_music/section_turn.py :: bundle = build_score_context(work, section_id=default_sec)
    Becomes: bundle = build_score_context(work, section_id=default_sec); [__import__("uclone_x_music.proposals").proposals.apply_proposal(work, p.id, seen_digest=d, decided_in="conversation") for p, d in work.list_proposals()]
    """
    ctx = await _seed_four_stem_song(tmp_path)
    prop_tool = MusicProposalTool()
    view = MusicView(tmp_path)
    await prop_tool.run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="violin_solo",
            start_bar=1,
            end_bar=2,
            summary="Gentle intro",
            intent="gentle_intro",
        ),
        ctx,
    )
    before = view.show("midnight-nocturne")
    assert len(before.proposals) == 1
    mix_before = {i.id: (i.solo, i.muted) for i in before.instruments}

    aid = MusicTurnHook().turn_aid(
        message=message,
        story_id="midnight-nocturne",
        room_id=ctx.room_id,
        workspace_root=tmp_path,
        tool_names=frozenset({"music_proposal", "music_orchestra"}),
    )
    assert aid is not None

    after = view.show("midnight-nocturne")
    assert [p.id for p in after.proposals] == [p.id for p in before.proposals]
    assert {i.id: (i.solo, i.muted) for i in after.instruments} == mix_before


def _music_routes_client(
    workspace: Path, rooms: RoomService, *, answering: frozenset[str] = frozenset()
) -> TestClient:
    """The music score routes over `workspace`, with conversations from `rooms`.

    The conversations in `answering` have a turn running; no other does.
    """
    app = FastAPI()

    class _FakeSessionManager:
        workspace_dir = workspace

    class _FakeStack:
        service = rooms

        def session_manager(self) -> Any:
            return _FakeSessionManager()

        def turn_in_flight(self, room_id: str) -> bool:
            return room_id in answering

        def turn_unlanded(self, _room_id: str) -> bool:
            return False

    class _FakePerson:
        def require(self, _request: Any) -> None:
            return None

    register_music_routes(
        ExtensionRouteContext(
            app=app,
            stack=_FakeStack(),  # type: ignore[arg-type]
            person=_FakePerson(),  # type: ignore[arg-type]
            refuse_cross_origin=lambda _req: None,
        )
    )
    return TestClient(app)


async def song_with_pending_violin_patch(
    workspace: Path, room_id: str
) -> tuple[list[TrackBar], str]:
    """A song held by `room_id` whose violin bars 1..2 have a pending `track_patch` proposal.

    The proposal is staged the way Composer stages one: `music_track(patch_bars)` with the
    bars, which proposes by default. Returns the proposed bars and the proposal id.
    """
    await _seed_four_stem_song(workspace)
    SongLibrary(workspace).take_over("midnight-nocturne", room_id)
    proposed = [
        TrackBar(bar=1, chord="Dm7", notes=["1.0:D5:1.5:v72:legato", "2.5:C#5:2.5:v68:tenuto"]),
        TrackBar(bar=2, chord="Gm7", notes=["1.0:B4:2.0:v70:legato", "3.0:F#4:2.0:v64:tenuto"]),
    ]
    staged = await MusicTrackTool().run(
        MusicTrackParams(
            action="patch_bars",
            section_id="theme_a",
            instrument_id="violin_solo",
            start_bar=1,
            end_bar=2,
            bars=proposed,
            summary="Lyrical violin opening",
        ),
        _ctx(workspace, room_id=room_id, song_id="midnight-nocturne"),
    )
    assert staged["proposed"] is True
    return proposed, str(staged["proposal_id"])


@pytest.mark.asyncio
async def test_approving_in_the_score_view_applies_the_patch_it_showed(tmp_path: Path) -> None:
    """Approve in MusicView: the route takes the `seen_digest` the view returned and applies it.

    The violin track gets the proposed bars, and the proposal leaves the pending list.

    Killed by: examples/music/src/uclone_x_music/routes.py :: song_id, proposal_id, seen_digest=body.seen_digest, room_id=body.room_id
    Becomes: song_id, proposal_id, seen_digest="", room_id=body.room_id
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rooms = RoomService(RoomStore(tmp_path / "rooms"))
    room_id = rooms.create("Composer Room").room_id
    proposed, proposal_id = await song_with_pending_violin_patch(workspace, room_id)
    client = _music_routes_client(workspace, rooms)

    shown = client.get("/api/artifacts/library/songs/midnight-nocturne").json()
    assert shown["writable"] is True
    shown_proposal = next(p for p in shown["proposals"] if p["id"] == proposal_id)

    resp = client.post(
        f"/api/artifacts/library/songs/midnight-nocturne/proposals/{proposal_id}/approve",
        json={"seen_digest": shown_proposal["seen_digest"], "room_id": room_id},
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["proposals"] == []

    work = SongWork(SongLibrary(workspace), "midnight-nocturne", room_id)
    track = work.read_track("theme_a", "violin_solo")
    assert track is not None
    assert [b.notes for b in track[0].bars[:2]] == [b.notes for b in proposed]
    decided = work.read_proposal(proposal_id)
    assert decided is not None and decided[0].status == "approved"


@pytest.mark.asyncio
async def test_approving_from_the_dock_opens_a_song_whose_conversation_is_gone(
    tmp_path: Path,
) -> None:
    """A song whose writer conversation was deleted is approved from the dock's conversation.

    Without a conversation the approval is refused in plain words; sent from the dock it moves
    the song into that conversation and applies the proposal.

    Killed by: examples/music/src/uclone_x_music/view.py :: self._library.take_over_if_held_by(song_id, room_id, expected_holder=holder)
    Becomes: pass
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rooms = RoomService(RoomStore(tmp_path / "rooms"))
    docked_room = rooms.create("Composer Room").room_id
    _, proposal_id = await song_with_pending_violin_patch(workspace, "room_deleted_000")
    client = _music_routes_client(workspace, rooms)

    shown = client.get("/api/artifacts/library/songs/midnight-nocturne").json()
    assert shown["writable"] is False
    assert shown["decide_code"] == "writer_gone"
    assert "no longer exists" in shown["decide_note"]
    seen = next(p for p in shown["proposals"] if p["id"] == proposal_id)["seen_digest"]
    url = f"/api/artifacts/library/songs/midnight-nocturne/proposals/{proposal_id}/approve"

    refused = client.post(url, json={"seen_digest": seen})
    assert refused.status_code >= 400
    assert "no longer exists" in refused.json()["detail"]

    resp = client.post(url, json={"seen_digest": seen, "room_id": docked_room})
    assert resp.status_code == 200, resp.json()
    assert resp.json()["proposals"] == []
    assert resp.json()["holder"] == docked_room


@pytest.mark.asyncio
async def test_a_docked_conversation_that_is_answering_does_not_take_the_song(
    tmp_path: Path,
) -> None:
    """Approving from a conversation mid-turn is refused before the lease moves.

    The song stays with its gone writer, the proposal stays pending, and the refusal says the
    conversation is answering.

    Killed by: examples/music/src/uclone_x_music/view.py :: self._refuse_if_busy(room_id)
    Becomes: pass
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rooms = RoomService(RoomStore(tmp_path / "rooms"))
    docked_room = rooms.create("Composer Room").room_id
    _, proposal_id = await song_with_pending_violin_patch(workspace, "room_deleted_000")
    client = _music_routes_client(workspace, rooms, answering=frozenset({docked_room}))
    shown = client.get("/api/artifacts/library/songs/midnight-nocturne").json()
    seen = next(p for p in shown["proposals"] if p["id"] == proposal_id)["seen_digest"]

    resp = client.post(
        f"/api/artifacts/library/songs/midnight-nocturne/proposals/{proposal_id}/approve",
        json={"seen_digest": seen, "room_id": docked_room},
    )
    assert resp.status_code == 409, resp.json()
    assert "answering right now" in resp.json()["detail"]
    record = SongLibrary(workspace).load("midnight-nocturne")
    assert record.lease is not None and record.lease.holder == "room_deleted_000"
    after = client.get("/api/artifacts/library/songs/midnight-nocturne").json()
    assert [p["id"] for p in after["proposals"]] == [proposal_id]
