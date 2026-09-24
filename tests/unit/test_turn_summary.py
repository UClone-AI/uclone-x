"""Unit tests for TurnSummary and summarize_turn (#1491)."""

from __future__ import annotations

import pytest

from uclone_x.core.provenance import AttemptRecord, ExecutionPath, Provenance, ServiceRef
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomMessage,
    RoomState,
    RoomToolUse,
    SelectionVerdict,
    SpeakerDecision,
)
from uclone_x.room.turn_summary import TurnNotFoundError, summarize_turn


def _make_agent(agent_id: str = "scout") -> Participant:
    return Participant(
        id=agent_id,
        kind=ParticipantKind.AGENT,
        display_name="Scout",
        session_id=f"sess_{agent_id}",
    )


def _make_human(user_id: str = "kenny") -> Participant:
    return Participant(
        id=user_id,
        kind=ParticipantKind.HUMAN,
        display_name="Kenny",
        session_id=f"sess_{user_id}",
    )


def test_turn_summary_with_steps_and_documents() -> None:
    """Killed by: src/uclone_x/room/turn_summary.py :: docs_map[u.written_path] = docs_map.get(u.written_path, 0) + 1
    Becomes: docs_map[u.written_path] = 1
    """
    agent = _make_agent("scout")
    msg = RoomMessage(
        seq=1,
        sender_id="scout",
        content="I wrote notes.",
        turn_id="turn_1",
        tools_recorded=True,
    )
    u1 = RoomToolUse(
        turn_id="turn_1",
        participant_id="scout",
        tool_name="file_write",
        status="success",
        written_path="docs/notes.md",
        recorded_at="2026-09-23T10:00:00Z",
    )
    u2 = RoomToolUse(
        turn_id="turn_1",
        participant_id="scout",
        tool_name="file_edit",
        status="success",
        written_path="docs/notes.md",
        recorded_at="2026-09-23T10:00:01Z",
    )
    state = RoomState(
        room_id="room_1",
        participants=(agent,),
        transcript=(msg,),
        tool_uses=(u1, u2),
    )

    summary = summarize_turn(state, 1)

    assert summary.seq == 1
    assert summary.turn_id == "turn_1"
    assert summary.steps_absent_reason is None
    assert len(summary.steps) == 2
    assert len(summary.documents) == 1
    assert summary.documents[0].path == "docs/notes.md"
    assert summary.documents[0].writes == 2


def test_turn_summary_tools_not_recorded() -> None:
    """Killed by: src/uclone_x/room/turn_summary.py :: if not message.tools_recorded:
    Becomes: if False:
    """
    agent = _make_agent("scout")
    msg = RoomMessage(
        seq=1,
        sender_id="scout",
        content="An unrecorded turn.",
        turn_id="turn_unrecorded",
        tools_recorded=False,
    )
    state = RoomState(
        room_id="room_1",
        participants=(agent,),
        transcript=(msg,),
    )

    summary = summarize_turn(state, 1)

    assert summary.steps_absent_reason == "not_recorded"
    assert summary.steps == []
    assert summary.documents == []


def test_turn_summary_no_tools_used() -> None:
    agent = _make_agent("scout")
    msg = RoomMessage(
        seq=1,
        sender_id="scout",
        content="A clean conversational answer.",
        turn_id="turn_no_tools",
        tools_recorded=True,
    )
    state = RoomState(
        room_id="room_1",
        participants=(agent,),
        transcript=(msg,),
        tool_uses=(),
    )

    summary = summarize_turn(state, 1)

    assert summary.steps == []
    assert summary.documents == []
    assert summary.steps_absent_reason is None


def test_turn_summary_human_message() -> None:
    """Killed by: src/uclone_x/room/turn_summary.py :: if not is_agent:
    Becomes: if False:
    """
    human = _make_human("kenny")
    msg = RoomMessage(
        seq=1,
        sender_id="kenny",
        content="Hello scout",
        turn_id=None,
        tools_recorded=False,
    )
    state = RoomState(
        room_id="room_1",
        participants=(human,),
        transcript=(msg,),
    )

    summary = summarize_turn(state, 1)

    assert summary.steps_absent_reason == "not_an_agent_turn"
    assert summary.steps == []
    assert summary.documents == []


def test_turn_summary_notable_degraded() -> None:
    """Killed by: src/uclone_x/room/turn_summary.py :: if getattr(prov, "degraded", False):
    Becomes: if False:
    """
    agent = _make_agent("scout")
    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="openai", model="gpt-4o"),
        served_by=ServiceRef(provider="openai", model="gpt-4o-mini"),
    )
    msg = RoomMessage(
        seq=1,
        sender_id="scout",
        content="Substituted answer.",
        turn_id="turn_degraded",
        tools_recorded=True,
        provenance=prov,
        decision=SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id="scout",
            selector="sole_agent",
        ),
    )
    state = RoomState(
        room_id="room_1",
        participants=(agent,),
        transcript=(msg,),
    )

    summary = summarize_turn(state, 1)
    assert summary.notable == ["degraded"]


def test_turn_summary_notable_failover() -> None:
    """Killed by: src/uclone_x/room/turn_summary.py :: if path_str == "failover":
    Becomes: if False:
    """
    agent = _make_agent("scout")
    attempt = AttemptRecord(provider="openai", model="gpt-4o", error_class="ConnectionError")
    prov = Provenance(
        path=ExecutionPath.FAILOVER,
        requested=ServiceRef(provider="openai", model="gpt-4o"),
        served_by=ServiceRef(provider="anthropic", model="claude-3-5-sonnet"),
        attempts=(attempt,),
    )
    msg = RoomMessage(
        seq=1,
        sender_id="scout",
        content="Failover answer.",
        turn_id="turn_failover",
        tools_recorded=True,
        provenance=prov,
        decision=SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id="scout",
            selector="sole_agent",
        ),
    )
    state = RoomState(
        room_id="room_1",
        participants=(agent,),
        transcript=(msg,),
    )

    summary = summarize_turn(state, 1)
    assert "failover" in summary.notable
    assert "retried" not in summary.notable


def test_turn_summary_notable_retried() -> None:
    """Killed by: src/uclone_x/room/turn_summary.py :: elif path_str == "retry":
    Becomes: elif False:
    """
    agent = _make_agent("scout")
    attempt = AttemptRecord(provider="openai", model="gpt-4o", error_class="RateLimitError")
    prov = Provenance(
        path=ExecutionPath.RETRY,
        requested=ServiceRef(provider="openai", model="gpt-4o"),
        served_by=ServiceRef(provider="openai", model="gpt-4o"),
        attempts=(attempt,),
    )
    msg = RoomMessage(
        seq=1,
        sender_id="scout",
        content="Retried answer.",
        turn_id="turn_retried",
        tools_recorded=True,
        provenance=prov,
        decision=SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id="scout",
            selector="sole_agent",
        ),
    )
    state = RoomState(
        room_id="room_1",
        participants=(agent,),
        transcript=(msg,),
    )

    summary = summarize_turn(state, 1)
    assert summary.notable == ["retried"]


def test_turn_summary_notable_contested() -> None:
    """Killed by: src/uclone_x/room/turn_summary.py :: if is_contested:
    Becomes: if False:
    """
    agent = _make_agent("scout")
    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="openai", model="gpt-4o"),
        served_by=ServiceRef(provider="openai", model="gpt-4o"),
    )
    msg = RoomMessage(
        seq=1,
        sender_id="scout",
        content="Mentioned answer.",
        turn_id="turn_contested",
        tools_recorded=True,
        provenance=prov,
        decision=SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id="scout",
            selector="MentionSelector",
        ),
    )
    state = RoomState(
        room_id="room_1",
        participants=(agent,),
        transcript=(msg,),
    )

    summary = summarize_turn(state, 1)
    assert summary.notable == ["contested"]


def test_turn_summary_notable_sole_agent_uncontested() -> None:
    agent = _make_agent("scout")
    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="openai", model="gpt-4o"),
        served_by=ServiceRef(provider="openai", model="gpt-4o"),
    )
    msg = RoomMessage(
        seq=1,
        sender_id="scout",
        content="Ordinary answer.",
        turn_id="turn_uncontested",
        tools_recorded=True,
        provenance=prov,
        decision=SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id="scout",
            selector="sole_agent",
        ),
    )
    state = RoomState(
        room_id="room_1",
        participants=(agent,),
        transcript=(msg,),
    )

    summary = summarize_turn(state, 1)
    assert summary.notable == []


def test_turn_summary_subagents_and_unnamed_writes() -> None:
    """Killed by: src/uclone_x/room/turn_summary.py :: unnamed_writes = sum(1 for u in steps if u.wrote_unnamed)
    Becomes: unnamed_writes = 0
    """
    agent = _make_agent("scout")
    msg = RoomMessage(
        seq=1,
        sender_id="scout",
        content="Subagent and unnamed write.",
        turn_id="turn_sub",
        tools_recorded=True,
    )
    u1 = RoomToolUse(
        turn_id="turn_sub",
        participant_id="scout",
        tool_name="spawn_subagent",
        status="success",
        subagent_id="sub_helper_1",
        recorded_at="2026-09-23T10:00:00Z",
    )
    u2 = RoomToolUse(
        turn_id="turn_sub",
        participant_id="scout",
        tool_name="bash_run",
        status="success",
        wrote_unnamed=True,
        recorded_at="2026-09-23T10:00:01Z",
    )
    state = RoomState(
        room_id="room_1",
        participants=(agent,),
        transcript=(msg,),
        tool_uses=(u1, u2),
    )

    summary = summarize_turn(state, 1)
    assert len(summary.subagent_steps) == 1
    assert summary.subagent_steps[0].subagent_id == "sub_helper_1"
    assert summary.unnamed_writes == 1


def test_turn_summary_turn_not_found() -> None:
    """Killed by: src/uclone_x/room/turn_summary.py :: if message is None:
    Becomes: if False:
    """
    state = RoomState(room_id="room_empty")
    with pytest.raises(TurnNotFoundError):
        summarize_turn(state, 999)
