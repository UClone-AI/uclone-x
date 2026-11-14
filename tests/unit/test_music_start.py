"""Unit tests for guided step-by-step co-composition (`music_start`) and `MusicTurnHook`."""

from __future__ import annotations

from pathlib import Path

import yaml

from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x_music import OPEN_SONG_KEY
from uclone_x_music.library import SongLibrary
from uclone_x_music.schemas import parse_pitch, resolve_track_bars
from uclone_x_music.section_turn import MUSIC_SECTION_HEADING, MusicTurnHook
from uclone_x_music.start import (
    BUNDLED_STYLES_PATH,
    MusicStartParams,
    MusicStartTool,
    load_bundled_styles,
    read_start_state,
)
from uclone_x_music.work import SongWork


def test_load_bundled_styles_and_composer_persona_yaml() -> None:
    """Bundled style presets and `personas/composer.yaml` validate with expected tools and schema.

    Killed by: examples/music/src/uclone_x_music/start.py :: styles = cast(dict[str, object], raw).get("styles")
    Becomes: styles = None
    """
    assert BUNDLED_STYLES_PATH.is_file()
    styles = load_bundled_styles()
    for profile in ("pop_ballad", "hiphop_groove", "jazz_swing", "classical_strict"):
        assert profile in styles
        preset = styles[profile]
        assert isinstance(preset["tempo"], int)
        assert preset["time_signature"] == "4/4"
        assert preset["chords"] and isinstance(preset["chords"], list)
        assert preset["layering_order"] == ["bass", "violin", "piano", "drums"]

    persona_path = BUNDLED_STYLES_PATH.parent.parent / "personas" / "composer.yaml"
    assert persona_path.is_file()
    persona_raw = yaml.safe_load(persona_path.read_text(encoding="utf-8"))
    assert persona_raw["name"] == "composer"
    assert persona_raw["display_name"]["en"] == "Composer"
    allowed = set(persona_raw["allowed_tools"])
    for expected_tool in (
        "music_library",
        "music_start",
        "music_score",
        "music_orchestra",
        "music_track",
        "music_context",
        "music_proposal",
        "music_audit",
    ):
        assert expected_tool in allowed

    from uclone_x.agent.persona_store import YamlFilePersonaStore

    store = YamlFilePersonaStore(persona_path.parent, read_only=True)
    composer_def = store.get_persona("composer")
    assert composer_def is not None
    assert composer_def.name == "composer"


async def test_music_start_tool_full_guided_layering_workflow(tmp_path: Path) -> None:
    """`MusicStartTool` walks through foundation bass -> violin -> piano -> hip-hop drums -> completed.

    Killed by: examples/music/src/uclone_x_music/start.py :: recommended_next="violin_solo"
    Becomes: recommended_next=""
    """
    tool = MusicStartTool()
    ctx = ToolContext(
        agent_id="composer",
        session_id="sess_music",
        room_id="room_music_01",
        workspace_root=tmp_path,
    )

    # Stage 1: setup_foundation -> creates song, score.yaml, bass_synth, and sec_verse_01_bass_synth
    step1 = await tool.run(
        MusicStartParams(action="start", request="새 곡 작곡 시작해줘", title="Midnight Groove"),
        ctx,
    )
    song_id = step1[OPEN_SONG_KEY]
    assert isinstance(song_id, str) and song_id
    assert step1["stage"] == "layer_melody"
    assert step1["completed_layers"] == ["bass_synth"]
    assert step1["recommended_next"] == "violin_solo"

    ctx = ctx.model_copy(update={"story_id": song_id})
    work = SongWork(SongLibrary(tmp_path), song_id, "room_music_01")
    score_pair = work.read_score()
    assert score_pair is not None
    outline = score_pair[0]
    assert outline.tempo == 96
    assert outline.time_signature == "4/4"
    assert outline.key_signature == "D minor"
    assert outline.sections[0].chords == ["Dm7", "Gm7", "C7", "Fmaj7"]

    # Stage 2: layer_melody -> the person agreed, so the model continues -> violin_solo in D5..A5
    step2 = await tool.run(MusicStartParams(action="continue"), ctx)
    assert step2["stage"] == "layer_harmony"
    assert step2["completed_layers"] == ["bass_synth", "violin_solo"]
    assert step2["recommended_next"] == "grand_piano"

    violin_inst_pair = work.read_instrument("violin_solo")
    assert violin_inst_pair is not None
    assert violin_inst_pair[0].min_pitch == "G3"
    assert violin_inst_pair[0].max_pitch == "E7"
    assert violin_inst_pair[0].role == "lead_melody"

    violin_track_pair = work.read_track("sec_verse_01", "violin_solo")
    assert violin_track_pair is not None
    violin_events = [
        ev
        for bar_evs in resolve_track_bars(violin_track_pair[0]).values()
        for ev in bar_evs
        if ev.midi >= 0
    ]
    assert all(parse_pitch("D5") <= ev.midi <= parse_pitch("A5") for ev in violin_events)

    # Stage 3: layer_harmony -> continue adds grand_piano in C3..G4
    step3 = await tool.run(MusicStartParams(action="continue"), ctx)
    assert step3["stage"] == "layer_drums"
    assert step3["completed_layers"] == ["bass_synth", "violin_solo", "grand_piano"]
    assert step3["recommended_next"] == "drum_kit"

    piano_inst_pair = work.read_instrument("grand_piano")
    assert piano_inst_pair is not None
    assert piano_inst_pair[0].min_pitch == "A0"
    assert piano_inst_pair[0].max_pitch == "C8"
    assert piano_inst_pair[0].role == "arpeggio"

    piano_track_pair = work.read_track("sec_verse_01", "grand_piano")
    assert piano_track_pair is not None
    piano_events = [
        ev
        for bar_evs in resolve_track_bars(piano_track_pair[0]).values()
        for ev in bar_evs
        if ev.midi >= 0
    ]
    assert all(parse_pitch("C3") <= ev.midi <= parse_pitch("G4") for ev in piano_events)

    # Stage 4: layer_drums -> the model states the hip-hop style -> drum_kit with swing 0.58
    step4 = await tool.run(
        MusicStartParams(action="continue", style_profile="hiphop_groove"),
        ctx,
    )
    assert step4["stage"] == "completed"
    assert step4["style_profile"] == "hiphop_groove"
    assert step4["completed_layers"] == [
        "bass_synth",
        "violin_solo",
        "grand_piano",
        "drum_kit",
    ]
    assert step4["recommended_next"] == ""
    assert step4["audit"]["passed"] is True

    drum_inst_pair = work.read_instrument("drum_kit")
    assert drum_inst_pair is not None
    assert drum_inst_pair[0].midi_channel == 10
    assert drum_inst_pair[0].role == "rhythm_percussion"

    drum_track_pair = work.read_track("sec_verse_01", "drum_kit")
    assert drum_track_pair is not None
    drum_track = drum_track_pair[0]
    assert drum_track.groove_swing == 0.58
    bar1_events = resolve_track_bars(drum_track)[1]
    kick_beats = {ev.beat for ev in bar1_events if ev.midi == 36}
    snare_beats = {ev.beat for ev in bar1_events if ev.midi == 38}
    assert {1.0, 2.5}.issubset(kick_beats)
    assert {2.0, 4.0}.issubset(snare_beats)

    saved_state = read_start_state(work)
    assert saved_state is not None
    assert saved_state[0].stage == "completed"


async def test_music_start_tool_regional_fast_edit_subdivides_selected_bars_only(
    tmp_path: Path,
) -> None:
    """Regional fast-edit subdivides only bars 3..4 of `violin_solo` into 16th-note patterns.

    Killed by: examples/music/src/uclone_x_music/start.py :: output["patched_bars"] = [b.bar for b in patched_bars]
    Becomes: output["patched_bars"] = []
    """
    tool = MusicStartTool()
    ctx = ToolContext(
        agent_id="composer",
        session_id="sess_edit",
        room_id="room_edit_01",
        workspace_root=tmp_path,
    )

    step1 = await tool.run(MusicStartParams(action="start", request="곡 시작"), ctx)
    song_id = str(step1[OPEN_SONG_KEY])
    ctx = ctx.model_copy(update={"story_id": song_id})
    await tool.run(MusicStartParams(action="continue"), ctx)

    work = SongWork(SongLibrary(tmp_path), song_id, "room_edit_01")
    before_violin, _ = work.read_track("sec_verse_01", "violin_solo") or (None, "")
    before_bass, bass_digest_before = work.read_track("sec_verse_01", "bass_synth") or (None, "")
    assert before_violin is not None and before_bass is not None

    bar1_before = before_violin.bars[0].notes
    bar2_before = before_violin.bars[1].notes
    bar3_count_before = len(before_violin.bars[2].notes)

    edit_res = await tool.run(
        MusicStartParams(
            action="edit_region",
            section_id="sec_verse_01",
            start_bar=3,
            end_bar=4,
            instrument_id="violin_solo",
        ),
        ctx,
    )
    assert edit_res["patched_bars"] == [3, 4]
    assert edit_res["instrument_id"] == "violin_solo"

    after_violin, _ = work.read_track("sec_verse_01", "violin_solo") or (None, "")
    after_bass, bass_digest_after = work.read_track("sec_verse_01", "bass_synth") or (None, "")
    assert after_violin is not None and after_bass is not None

    # Untouched bars 1..2 and untouched bass track remain identical
    assert after_violin.bars[0].notes == bar1_before
    assert after_violin.bars[1].notes == bar2_before
    assert bass_digest_after == bass_digest_before

    # Selected bars 3..4 have faster subdivided 16th-note (0.25 beat) events
    assert len(after_violin.bars[2].notes) > bar3_count_before
    bar3_events = after_violin.bars[2].parsed_events()
    assert any(abs(ev.duration - 0.25) < 1e-6 for ev in bar3_events)


async def test_music_turn_hook_moves_open_song_and_injects_context_aid(tmp_path: Path) -> None:
    """`MusicTurnHook` updates `story_id` after tool execution and builds `MusicTurnAid`.

    Killed by: examples/music/src/uclone_x_music/section_turn.py :: song = song_after(records, context.story_id)
    Becomes: song = context.story_id
    """
    tool = MusicStartTool()
    ctx = ToolContext(
        agent_id="composer",
        session_id="sess_hook",
        room_id="room_hook_01",
        workspace_root=tmp_path,
    )
    out = await tool.run(MusicStartParams(action="start", request="곡 시작"), ctx)
    song_id = str(out[OPEN_SONG_KEY])

    hook = MusicTurnHook()
    record = ToolExecutionRecord(
        tool_name="music_start",
        arguments={"action": "start"},
        output={OPEN_SONG_KEY: song_id},
        status=ToolResultStatus.SUCCESS,
        opens_story=True,
    )
    updated_ctx = hook.after_tool_step([record], ctx)
    assert updated_ctx.story_id == song_id

    aid = hook.turn_aid(
        message="[sec_verse_01 | Bar 3-4 | bass_synth] 이 구간 확인해줘",
        story_id=song_id,
        room_id="room_hook_01",
        workspace_root=tmp_path,
        tool_names=frozenset({"music_start", "music_context"}),
    )
    assert aid is not None
    assert aid.song_id == song_id
    assert aid.start_stage == "layer_melody"
    # The badge in the message is for the model to read; the hook selects nothing from it.
    assert "Selected region" not in aid.section
    assert MUSIC_SECTION_HEADING in aid.section
    assert "layer_melody" in aid.section
    assert aid.reply_lines([record], korean=True) == []


async def test_a_request_s_words_choose_no_style_and_no_region(tmp_path: Path) -> None:
    """Words in `request` are recorded, never read: only the structured arguments decide.

    "힙합" in the request picks no style, and "3~4마디 바이올린 빠르게" on a `continue` edits
    no bars: the call adds the next layer, as `action` says.

    Killed by: examples/music/src/uclone_x_music/start.py :: params.action == "edit_region"
    Becomes: params.action in ("edit_region", "continue")
    """
    tool = MusicStartTool()
    ctx = ToolContext(
        agent_id="composer",
        session_id="sess_words",
        room_id="room_words_01",
        workspace_root=tmp_path,
    )
    step1 = await tool.run(
        MusicStartParams(action="start", request="힙합 비트로 곡 시작해줘 hip-hop 808"), ctx
    )
    assert step1["style_profile"] == "pop_ballad"
    song_id = str(step1[OPEN_SONG_KEY])
    assert SongLibrary(tmp_path).load(song_id).logline == "힙합 비트로 곡 시작해줘 hip-hop 808"
    ctx = ctx.model_copy(update={"story_id": song_id})

    step2 = await tool.run(
        MusicStartParams(action="continue", request="3~4마디 바이올린 빠르게 16분 double-time"),
        ctx,
    )
    assert step2["stage"] == "layer_harmony"
    assert step2["added_instrument"] == "violin_solo"
    assert "patched_bars" not in step2


async def test_music_start_melody_and_piano_adapt_to_style_chords_and_bass_rhythm(
    tmp_path: Path,
) -> None:
    """`violin_solo` and `grand_piano` follow the section's chords, style profile, and bass rhythm.

    Killed by: examples/music/src/uclone_x_music/start.py :: style=state.style_profile,
    Becomes: style="pop_ballad",
    """
    from uclone_x_music.arranger import synthesize_bars_for_intent
    from uclone_x_music.groove import chord_root_pc, subdivide_bars_faster
    from uclone_x_music.schemas import TrackBar, TrackFile

    tool = MusicStartTool()
    ctx_pop = ToolContext(
        agent_id="composer",
        session_id="sess_pop",
        room_id="room_pop",
        workspace_root=tmp_path,
    )
    pop_s1 = await tool.run(
        MusicStartParams(action="start", style_profile="pop_ballad", title="Pop Song"),
        ctx_pop,
    )
    pop_id = str(pop_s1[OPEN_SONG_KEY])
    ctx_pop = ctx_pop.model_copy(update={"story_id": pop_id})
    await tool.run(MusicStartParams(action="continue"), ctx_pop)
    await tool.run(MusicStartParams(action="continue"), ctx_pop)

    ctx_cls = ToolContext(
        agent_id="composer",
        session_id="sess_cls",
        room_id="room_cls",
        workspace_root=tmp_path,
    )
    cls_s1 = await tool.run(
        MusicStartParams(action="start", style_profile="classical_strict", title="Classical Song"),
        ctx_cls,
    )
    cls_id = str(cls_s1[OPEN_SONG_KEY])
    ctx_cls = ctx_cls.model_copy(update={"story_id": cls_id})
    await tool.run(MusicStartParams(action="continue"), ctx_cls)
    step_cls_piano = await tool.run(MusicStartParams(action="continue"), ctx_cls)
    assert step_cls_piano["audit"]["passed"] is True

    work_pop = SongWork(SongLibrary(tmp_path), pop_id, "room_pop")
    work_cls = SongWork(SongLibrary(tmp_path), cls_id, "room_cls")

    v_pop, _ = work_pop.read_track("sec_verse_01", "violin_solo") or (None, "")
    v_cls, _ = work_cls.read_track("sec_verse_01", "violin_solo") or (None, "")
    p_pop, _ = work_pop.read_track("sec_verse_01", "grand_piano") or (None, "")
    p_cls, _ = work_cls.read_track("sec_verse_01", "grand_piano") or (None, "")
    assert v_pop is not None and v_cls is not None
    assert p_pop is not None and p_cls is not None

    # Style and chord differences produce distinct violin melodies and piano accompaniment
    assert [b.notes for b in v_pop.bars] != [b.notes for b in v_cls.bars]
    assert [b.notes for b in p_pop.bars] != [b.notes for b in p_cls.bars]
    # Bars 1..4 within a single song are not 4 copies of the same bar
    assert len({tuple(b.notes) for b in v_cls.bars}) >= 3

    # Arranger chord root extraction handles Am7, Dm7, A7 accurately (not sorted lowest pitch class C/C#)
    assert chord_root_pc("Am7") == 9
    assert chord_root_pc("Dm7") == 2
    assert chord_root_pc("A7") == 9
    b_cls, _ = work_cls.read_track("sec_verse_01", "bass_synth") or (None, "")
    assert b_cls is not None
    bass_bars = synthesize_bars_for_intent(
        b_cls,
        1,
        2,
        chords=["Am7", "Dm7"],
        category="bass",
        role="bass_foundation",
        intent="continuation",
    )
    assert bass_bars[0].notes[0].startswith("1.0:A2:")
    assert bass_bars[1].notes[0].startswith("1.0:D2:")

    # Drum faster subdivision keeps 0.5-beat KICK/SNARE intact while subdividing HIHAT
    drum_track = TrackFile(
        track_id="sec_verse_01_drum_kit",
        section_id="sec_verse_01",
        instrument_id="drum_kit",
        bars=[
            TrackBar(
                bar=1,
                notes=[
                    "1.0:KICK:0.5:v100",
                    "1.0:HIHAT:0.5:v70",
                    "1.5:HIHAT:0.5:v65",
                    "2.0:SNARE:0.5:v96",
                    "2.5:HIHAT:2.5:v70",
                ],
            )
        ],
    )
    subdivided = subdivide_bars_faster(drum_track, start_bar=1, end_bar=1)
    assert "1.0:KICK:0.5:v100" in subdivided[0].notes
    assert "2.0:SNARE:0.5:v96" in subdivided[0].notes
    assert any(n.startswith("1.25:HIHAT:0.25") for n in subdivided[0].notes)
