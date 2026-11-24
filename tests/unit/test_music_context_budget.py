"""What the music tools put in a model's context, on a full-size song (#2429).

Owner ruling, 2026-10-30: no call may put its whole result in context, and no score may
appear in context more than once. A tool result is a digest and a short summary, or the
slice the model asked for. These tests hold every music tool to a size cap on a song of
three 32-bar sections and twelve full tracks (`tests/support/music_budget.py`), and run a
scripted three-turn conversation through the real agent turn path to check that no bar
reaches one request twice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from tests.support.music_budget import (
    BARS,
    ROOM,
    SONG,
    ScriptedComposer,
    bar_fingerprints,
    bar_notes,
    build_full_song,
    composer_agent,
    repeated_bodies,
    run_scenario,
    scripted_reply,
)
from uclone_x.core.tool_results import canonical_tool_text
from uclone_x.llm.models import ToolCallRequest
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x_music.library import SongLibrary
from uclone_x_music.start import MusicStartTool
from uclone_x_music.tools import (
    APPLY_PROPOSAL_NOT_APPROVED_NOTE,
    MusicContextParams,
    MusicContextTool,
    MusicOrchestraParams,
    MusicOrchestraTool,
    MusicProposalParams,
    MusicProposalTool,
    MusicScoreParams,
    MusicScoreTool,
    MusicTheoryAuditTool,
    MusicTrackParams,
    MusicTrackTool,
)
from uclone_x_music.work import SongWork

#: The most a music tool's result may put in context, as canonical JSON characters. A
#: whole 32-bar track is about 5,400; the investigation's largest single result was 7,761.
RESULT_CAP = 3_000


def _ctx(workspace: Path, *, approved: bool = False) -> ToolContext:
    return ToolContext(
        agent_id="composer",
        session_id="s-budget",
        workspace_root=workspace,
        room_id=ROOM,
        story_id=SONG,
        approved_by_person=approved,
    )


@pytest.fixture
def song(tmp_path: Path) -> Path:
    build_full_song(tmp_path)
    return tmp_path


def _bars(instrument_index: int, *bars: int, shift: int = 5) -> list[dict[str, Any]]:
    return [{"bar": b, "notes": bar_notes(0, instrument_index, b, shift=shift)} for b in bars]


async def _propose(song: Path) -> str:
    out = await MusicProposalTool().run(
        MusicProposalParams(
            action="propose",
            section_id="theme_a",
            instrument_id="violin_solo",
            start_bar=1,
            end_bar=BARS,
            sketch=True,
        ),
        _ctx(song),
    )
    return str(out["proposal_id"])


_CASES: list[tuple[str, BaseTool[Any], dict[str, Any]]] = [
    ("score.read", MusicScoreTool(), {"action": "read"}),
    ("score.set_metadata", MusicScoreTool(), {"action": "set_metadata", "tempo": 120}),
    ("orchestra.list", MusicOrchestraTool(), {"action": "list"}),
    (
        "orchestra.adjust_mix_all",
        MusicOrchestraTool(),
        {"action": "adjust_mix", "instrument_id": "all", "volume": 70},
    ),
    (
        "track.read",
        MusicTrackTool(),
        {"action": "read", "section_id": "theme_a", "instrument_id": "violin_solo"},
    ),
    (
        "track.write_track",
        MusicTrackTool(),
        {
            "action": "write_track",
            "section_id": "theme_a",
            "instrument_id": "violin_solo",
            "force_direct": True,
            "bars": [{"bar": b, "notes": bar_notes(0, 0, b, shift=3)} for b in range(1, BARS + 1)],
        },
    ),
    (
        "track.patch_bars_propose",
        MusicTrackTool(),
        {
            "action": "patch_bars",
            "section_id": "theme_a",
            "instrument_id": "violin_solo",
            "start_bar": 4,
            "end_bar": 4,
            "bars": _bars(0, 4),
        },
    ),
    (
        "track.patch_bars_direct_all",
        MusicTrackTool(),
        {
            "action": "patch_bars",
            "section_id": "theme_a",
            "instrument_id": "all",
            "start_bar": 1,
            "end_bar": BARS,
            "propose": False,
            "sketch": True,
        },
    ),
    (
        "track.clear_bars",
        MusicTrackTool(),
        {
            "action": "clear_bars",
            "section_id": "theme_a",
            "instrument_id": "bass_synth",
            "start_bar": 1,
            "end_bar": 2,
        },
    ),
    (
        "track.transpose_bars",
        MusicTrackTool(),
        {
            "action": "transpose_bars",
            "section_id": "theme_a",
            "instrument_id": "bass_synth",
            "start_bar": 1,
            "end_bar": BARS,
            "semitones": 2,
        },
    ),
    (
        "context.region",
        MusicContextTool(),
        {"selected_region": {"section_id": "theme_a", "start_bar": 1, "end_bar": BARS}},
    ),
    (
        "proposal.propose_all",
        MusicProposalTool(),
        {"action": "propose", "instrument_id": "all", "sketch": True},
    ),
    ("audit", MusicTheoryAuditTool(), {"section_id": "theme_a"}),
    ("start.status", MusicStartTool(), {"action": "status"}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("label", "tool", "arguments"), _CASES, ids=[c[0] for c in _CASES])
async def test_a_music_tool_result_stays_under_the_cap_on_a_full_size_song(
    song: Path, label: str, tool: BaseTool[Any], arguments: dict[str, Any]
) -> None:
    """Killed by: examples/music/src/uclone_x_music/tools.py :: "bars": shown,
    Becomes: "bars": [bar_view(b) for b in track.bars],
    """
    params_type = tool.params_type
    assert params_type is not None
    out = await tool.run(params_type.model_validate(arguments), _ctx(song))

    text = canonical_tool_text(out)
    assert len(text) <= RESULT_CAP, f"{label}: {len(text)} chars: {text[:300]}"


@pytest.mark.asyncio
async def test_proposal_list_apply_and_reject_stay_under_the_cap(song: Path) -> None:
    """`list` shows pending proposals as short summaries, without their bars.

    Killed by: examples/music/src/uclone_x_music/tools.py :: items = work.list_proposals(status=params.status or "pending")
    Becomes: items = work.list_proposals(status=params.status)
    """
    tool = MusicProposalTool()
    first = await _propose(song)
    second = await _propose(song)
    for _ in range(4):
        await _propose(song)
    await tool.run(MusicProposalParams(action="reject", proposal_id=second), _ctx(song))

    listed = await tool.run(MusicProposalParams(action="list"), _ctx(song))
    applied = await tool.run(
        MusicProposalParams(action="apply", proposal_id=first), _ctx(song, approved=True)
    )

    ids = [p["id"] for p in listed["proposals"]]
    assert second not in ids
    assert first in ids
    for out in (listed, applied):
        text = canonical_tool_text(out)
        assert len(text) <= RESULT_CAP, text[:300]


@pytest.mark.asyncio
async def test_a_track_read_returns_the_bars_asked_for_and_says_what_it_left_out(
    song: Path,
) -> None:
    """Killed by: examples/music/src/uclone_x_music/tools.py :: for bar in [b for b in track.bars if first <= b.bar <= last]:
    Becomes: for bar in list(track.bars):
    """
    out = await MusicTrackTool().run(
        MusicTrackParams(
            action="read",
            section_id="theme_a",
            instrument_id="violin_solo",
            start_bar=5,
            end_bar=6,
        ),
        _ctx(song),
    )

    assert [b["bar"] for b in out["bars"]] == [5, 6]
    assert out["bars_total"] == BARS


@pytest.mark.asyncio
async def test_a_write_says_what_changed_and_never_returns_the_track(song: Path) -> None:
    """Killed by: examples/music/src/uclone_x_music/digests.py :: "previous_digest": previous,
    Becomes: "previous_digest": digest,
    """
    out = await MusicTrackTool().run(
        MusicTrackParams.model_validate(
            {
                "action": "patch_bars",
                "section_id": "theme_a",
                "instrument_id": "bass_synth",
                "start_bar": 3,
                "end_bar": 3,
                "propose": False,
                "bars": _bars(2, 3),
            }
        ),
        _ctx(song),
    )

    assert out["changed"] is True
    assert out["changed_bars"] == [3]
    assert "track" not in out
    assert out["digest"] != out["previous_digest"]


@pytest.mark.asyncio
async def test_a_patch_that_repeats_unchanged_bars_carries_only_the_changed_ones(
    song: Path,
) -> None:
    """A patch over bars 1..4 that changes only bar 4 is staged as bar 4 alone.

    Killed by: examples/music/src/uclone_x_music/tools.py :: start_bar, end_bar = min(changed_bars), max(changed_bars)
    Becomes: start_bar, end_bar = bar_range
    """
    unchanged = [{"bar": b, "notes": bar_notes(0, 0, b)} for b in (1, 2, 3)]
    out = await MusicTrackTool().run(
        MusicTrackParams.model_validate(
            {
                "action": "patch_bars",
                "section_id": "theme_a",
                "instrument_id": "violin_solo",
                "start_bar": 1,
                "end_bar": 4,
                "bars": [*unchanged, *_bars(0, 4)],
            }
        ),
        _ctx(song),
    )

    work = SongWork(SongLibrary(song), SONG, ROOM)
    loaded = work.read_proposal(str(out["proposal_id"]))
    assert loaded is not None
    proposal, _ = loaded
    assert (proposal.start_bar, proposal.end_bar) == (4, 4)
    assert [b.bar for b in proposal.proposed_bars] == [4]
    assert out["changed_bars"] == [4]


_NOTHING = "Nothing changed"


@pytest.mark.asyncio
async def test_a_patch_identical_to_the_track_says_nothing_changed_and_writes_nothing(
    song: Path,
) -> None:
    """Killed by: examples/music/src/uclone_x_music/tools.py :: if not changed_bars:
    Becomes: if False:
    """
    work = SongWork(SongLibrary(song), SONG, ROOM)
    before = work.read_track("theme_a", "violin_solo")
    assert before is not None
    for propose in (True, False):
        out = await MusicTrackTool().run(
            MusicTrackParams.model_validate(
                {
                    "action": "patch_bars",
                    "section_id": "theme_a",
                    "instrument_id": "violin_solo",
                    "start_bar": 2,
                    "end_bar": 3,
                    "propose": propose,
                    "bars": [{"bar": b, "notes": bar_notes(0, 0, b)} for b in (2, 3)],
                }
            ),
            _ctx(song),
        )
        assert out["changed"] is False
        assert out["summary"].startswith(_NOTHING)
    assert work.list_proposals() == []
    after = work.read_track("theme_a", "violin_solo")
    assert after is not None and after[1] == before[1]


@pytest.mark.asyncio
async def test_writing_the_same_track_again_says_nothing_changed_and_adds_no_revision(
    song: Path,
) -> None:
    """Killed by: examples/music/src/uclone_x_music/work.py :: if written.digest == expected_digest:
    Becomes: if False:
    """
    work = SongWork(SongLibrary(song), SONG, ROOM)
    log_before, _ = work.read_revision_log("theme_a_violin_solo")
    stored = work.read_track("theme_a", "violin_solo")
    assert stored is not None

    out = await MusicTrackTool().run(
        MusicTrackParams.model_validate(
            {
                "action": "write_track",
                "section_id": "theme_a",
                "instrument_id": "violin_solo",
                "force_direct": True,
                "bars": [b.model_dump(mode="json") for b in stored[0].bars],
            }
        ),
        _ctx(song),
    )

    assert out["changed"] is False
    assert out["summary"].startswith(_NOTHING)
    log_after, _ = work.read_revision_log("theme_a_violin_solo")
    assert len(log_after.revisions) == len(log_before.revisions)


@pytest.mark.asyncio
async def test_a_repeated_section_upsert_and_mix_change_say_nothing_changed(song: Path) -> None:
    """Killed by: examples/music/src/uclone_x_music/digests.py :: changed = previous != digest
    Becomes: changed = True
    """
    upsert = MusicScoreParams(action="upsert_section", section_id="theme_a", bars=BARS)
    mix = MusicOrchestraParams(action="adjust_mix", instrument_id="violin_solo", volume=85)
    for out in (
        await MusicScoreTool().run(upsert, _ctx(song)),
        await MusicOrchestraTool().run(mix, _ctx(song)),
    ):
        assert out["changed"] is False
        assert out["summary"].startswith(_NOTHING)


@pytest.mark.asyncio
async def test_music_context_already_in_the_turn_context_is_not_sent_again(song: Path) -> None:
    """Killed by: examples/music/src/uclone_x_music/tools.py :: if any(mark in section for section in context.turn_aid_sections):
    Becomes: if False:
    """
    section = _aid_section(song, room_id=ROOM)
    ctx = _ctx(song).model_copy(update={"turn_aid_sections": (section,)})

    out = await MusicContextTool().run(MusicContextParams(), ctx)

    assert out["same_as_turn_context"] is True
    assert out["digest"] in section
    assert "tail_bars" not in out


def _aid_section(workspace: Path, *, room_id: str | None) -> str:
    from uclone_x_music.section_turn import MusicTurnHook

    aid = MusicTurnHook().turn_aid(
        message="hi",
        story_id=SONG,
        room_id=room_id,
        workspace_root=workspace,
        tool_names=frozenset({"music_context"}),
    )
    assert aid is not None
    return aid.section


def _conversation(workspace: Path, session_id: str, room_id: str | None) -> ToolContext:
    return ToolContext(
        agent_id="composer",
        session_id=session_id,
        workspace_root=workspace,
        room_id=room_id,
        story_id=SONG,
    )


@pytest.mark.asyncio
async def test_a_new_conversation_on_a_song_shown_earlier_gets_the_whole_context(
    song: Path,
) -> None:
    """What an earlier conversation was shown says nothing about this request (#2434 review).

    A 1:1 conversation has no room. The first conversation's request carried the song's
    score context; a later one whose request carries none must get the bundle itself.

    Killed by: examples/music/src/uclone_x_music/tools.py :: if any(mark in section for section in context.turn_aid_sections):
    Becomes: if True:
    """
    _aid_section(song, room_id=None)

    out = await MusicContextTool().run(MusicContextParams(), _conversation(song, "s-later", None))

    assert "same_as_turn_context" not in out
    assert out["tail_bars"]


@pytest.mark.asyncio
async def test_two_open_conversations_on_one_song_each_get_the_context_on_first_ask(
    song: Path,
) -> None:
    """Two conversations open on the same song at once: neither's request speaks for the
    other's. Each gets the bundle when its own request does not carry it, and a reference
    only when it does.

    Killed by: examples/music/src/uclone_x_music/tools.py :: if any(mark in section for section in context.turn_aid_sections):
    Becomes: if True:
    """
    shown_to_a = _aid_section(song, room_id=None)
    first = _conversation(song, "s-a", None)
    second = _conversation(song, "s-b", None)

    out_b = await MusicContextTool().run(MusicContextParams(), second)
    out_a_bare = await MusicContextTool().run(MusicContextParams(), first)
    out_a = await MusicContextTool().run(
        MusicContextParams(), first.model_copy(update={"turn_aid_sections": (shown_to_a,)})
    )

    assert out_b["tail_bars"] and "same_as_turn_context" not in out_b
    assert out_a_bare["tail_bars"] and "same_as_turn_context" not in out_a_bare
    assert out_a["same_as_turn_context"] is True


@pytest.mark.asyncio
async def test_no_bar_reaches_one_request_twice_over_a_scripted_conversation(
    tmp_path: Path,
) -> None:
    """Killed by: examples/music/src/uclone_x_music/tools.py :: "proposals": [proposal_line(p, d) for p, d in items[-LIST_ITEMS_SHOWN:]],
    Becomes: "proposals": [{**p.model_dump(mode="json"), "digest": d} for p, d in items[-LIST_ITEMS_SHOWN:]],
    Killed by: src/uclone_x/agent/turn_executor.py :: turn_aid_sections=tuple(self._turn_aid_sections),
    Becomes: turn_aid_sections=(),
    """
    llm, _ = await run_scenario(tmp_path)

    fingerprints = bar_fingerprints()
    assert len(llm.requests) >= 10
    for index, request in enumerate(llm.requests, start=1):
        repeats = repeated_bodies(request, fingerprints)
        assert repeats == {}, f"request {index} repeats {len(repeats)} bars"


@pytest.mark.asyncio
async def test_an_apply_nobody_here_can_approve_ends_the_turn_with_a_plain_sentence(
    tmp_path: Path,
) -> None:
    """The refused `apply` leaves the model no tools to work around it with.

    Killed by: src/uclone_x/agent/turn_executor.py :: if any(record.ends_turn for record in step_executions):
    Becomes: if False:
    Killed by: examples/music/src/uclone_x_music/tools.py :: approval_unavailable_ends_turn: ClassVar[bool] = True
    Becomes: approval_unavailable_ends_turn: ClassVar[bool] = False
    """
    llm, results = await run_scenario(tmp_path)

    turn = results[1]
    apply = next(e for e in turn.tool_executions if dict(e.arguments).get("action") == "apply")
    assert apply.error == APPLY_PROPOSAL_NOT_APPROVED_NOTE
    after_apply = next(
        r
        for r in llm.requests
        if any(m.tool_call_id == "c6" for m in r.messages if m.tool_call_id is not None)
    )
    assert after_apply.tools == ()
    assert turn.tool_executions[-1] is apply
    assert "MusicView" in turn.content


def test_the_apply_refusal_is_plain_words_without_internals() -> None:
    """Killed by: examples/music/src/uclone_x_music/tools.py :: "approve a proposal, and they do that in MusicView. Tell them it is waiting there for "
    Becomes: "approve a proposal (approved_by_person=True) in MusicView. Tell them it waits for "
    """
    note = APPLY_PROPOSAL_NOT_APPROVED_NOTE
    for internal in ("`", "approved_by_person", "=", "True", "_"):
        assert internal not in note
    assert "MusicView" in note
    assert "not applied" in note


@pytest.mark.asyncio
async def test_the_score_context_shows_a_write_from_the_step_before(tmp_path: Path) -> None:
    """The `[Music score context]` is rebuilt after a step that changed the song.

    It is turn context, sent at the tail of each request and never kept in history, so the
    step after a chord change sees the new chords and no earlier message changes.

    Killed by: src/uclone_x/agent/turn_executor.py :: sections[sections.index(current)] = fresh
    Becomes: pass
    """
    build_full_song(tmp_path)
    chords: list[JsonValue] = ["Em9"] * BARS
    call = ToolCallRequest(
        id="u1",
        name="music_score",
        arguments={"action": "upsert_section", "section_id": "theme_a", "chords": chords},
    )
    llm = ScriptedComposer([scripted_reply(None, call), scripted_reply("Done.")])
    agent = composer_agent(tmp_path, llm)

    await agent.execute_turn(
        "Change theme A to Em9.", caller_turn_id="t1", room_id=ROOM, story_id=SONG
    )

    def context_of(index: int) -> str:
        texts = [m.content or "" for m in llm.requests[index].messages]
        return next(t for t in reversed(texts) if "[Music score context]" in t)

    assert '"Em9"' not in context_of(0)
    assert '"Em9", "Em9"' in context_of(1)
    assert agent.history
    assert not any("[Music score context]" in (m.content or "") for m in agent.history)


@pytest.mark.asyncio
async def test_each_request_extends_the_last_while_the_score_context_refreshes(
    tmp_path: Path,
) -> None:
    """The refreshed `[Music score context]` stays at the tail, so the history is append-only.

    Every request, minus its last message, is a byte-identical prefix of the next one,
    across steps and turns. Only the last message carries the score context, and it
    changes at least once over the conversation, so the prefix holds while it refreshes.

    Killed by: src/uclone_x/agent/request_record.py :: messages.append(ChatMessage(role=MessageRole.USER, content=block))
    Becomes: messages.insert(1, ChatMessage(role=MessageRole.USER, content=block))
    """
    from uclone_x_music.section_turn import MUSIC_SECTION_HEADING

    llm, _ = await run_scenario(tmp_path)

    heading = f"{MUSIC_SECTION_HEADING}\n"
    tails: list[str] = []
    for earlier, later in zip(llm.requests, llm.requests[1:], strict=False):
        kept = [m.model_dump_json() for m in earlier.messages[:-1]]
        assert [m.model_dump_json() for m in later.messages[: len(kept)]] == kept
    for request in llm.requests:
        holding = [m for m in request.messages if heading in (m.content or "")]
        assert holding == [request.messages[-1]]
        tails.append(request.messages[-1].content or "")
    shown = [t[t.index(heading) :] for t in tails]
    assert len(set(shown)) > 1
