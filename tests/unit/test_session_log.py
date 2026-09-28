"""The session log: every message that entered the history, and every tool body in full (#1443).

PR 1 of 3. The log is written beside `messages` on the one path every history change
takes, persisted on the record, and read by nothing that builds a request, so requests are
unchanged (`test_room_request_conformance.py` and `test_one_clone_builder.py` pass as they
were). These tests pin what the log itself must hold:

*   every message `messages` holds has an entry whose stored body is that message, and an
    over-cap tool result's entry names the full body, not the excerpt the history keeps;
*   a record written before the log loads with its messages backfilled as `migrated`
    entries, and loading it again adds none;
*   a rolled-back turn's messages leave the history and stay in the log;
*   after compaction or a replaced history, a message identical to one that left is
    logged when it enters again, at its own turn;
*   a room seat's session is logged the same way as a 1:1 session.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, AgentState
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.core.session_log import SessionLogEntry, SessionLogKind, SessionLogProvenance
from uclone_x.core.tool_results import (
    TOOL_RESULT_CAP_BYTES,
    artifacts_dir_for,
    handle_in,
    load_tool_result,
)
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ModelResponse, ToolCallRequest
from uclone_x.room.models import Participant, ParticipantKind
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import LocalTool, ToolRegistry

_SID = "sess_log"


class _ScriptedLLM(MockLLMConnector):
    """Asks for the tool calls of step N on its N-th request, then answers `done`."""

    def __init__(self, steps: Sequence[Sequence[ToolCallRequest]] = ()) -> None:
        super().__init__(default_response="done")
        self._steps = [list(s) for s in steps]
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        self._tool_calls = self._steps.pop(0) if self._steps else []
        return await super().generate(request)


def _returning(name: str, output: Any) -> LocalTool:
    async def handler(params: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True, output=output, provenance=Provenance.primary(provider="t", model=name)
        )

    return LocalTool(name, f"Returns a fixed {name} result.", handler=handler, writes_files=False)


def _big_text() -> str:
    return "\n".join(f"line {i:05d}: payload" for i in range(2_000))


def _agent(
    workspace: Path, store: SessionStore, llm: MockLLMConnector, tools: Sequence[LocalTool] = ()
) -> BaseAgent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return BaseAgent(
        config=AgentConfig(
            agent_id="agent_log",
            name="Logger",
            system_prompt="You keep a log.",
            workspace_dir=workspace,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=llm,
        tools=registry,
        store=store,
        context=AgentContext(session_id=_SID, agent_id="agent_log", current_state=AgentState.IDLE),
    )


def _logged_messages(store: SessionStore, state: SessionState) -> list[ChatMessage]:
    """Each entry's stored body, read back as the message it names."""
    out: list[ChatMessage] = []
    for entry in state.session_log:
        body = store.load_context_body(state.session_id, entry.digest)
        assert body is not None, f"{entry.id} names a body that is not stored"
        assert len(body.encode("utf-8")) == entry.size
        out.append(ChatMessage.model_validate_json(body))
    return out


def _log_lines(store: SessionStore, state: SessionState) -> list[tuple[str, int | None, str]]:
    """The log in order: each entry's kind, turn and message text, the summary shortened."""
    return [
        (entry.kind.value, entry.turn, (message.content or "")[:10])
        for entry, message in zip(state.session_log, _logged_messages(store, state), strict=True)
    ]


def _legacy_record(store: SessionStore, tmp_path: Path) -> None:
    """A record in the shape written before #1443: no `session_log` key."""
    store.save(
        SessionState(
            session_id=_SID,
            agent_id="agent_log",
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content="You keep a log."),
                ChatMessage(role=MessageRole.USER, content="hello"),
                ChatMessage(role=MessageRole.ASSISTANT, content="hi"),
                ChatMessage(role=MessageRole.USER, content="hello"),
                ChatMessage(role=MessageRole.ASSISTANT, content="hi again"),
            ),
            turn_counter=2,
        )
    )
    path = tmp_path / "sessions" / f"{_SID}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("session_log")
    path.write_text(json.dumps(raw), encoding="utf-8")


@pytest.mark.asyncio
async def test_the_log_holds_every_history_message_and_every_tool_body_in_full(
    tmp_path: Path,
) -> None:
    """Every message `messages` holds is a logged body; an excerpted result names its blob.

    The history keeps an over-cap result as an excerpt (#1422). The log entry for it names
    the `tr_` handle, and that handle loads the whole result, so nothing the tool returned
    is lost to the log even where the request carries only its head and tail.

    Killed by: src/uclone_x/core/session_log.py :: blob = handle_in(message.content) if message.role == MessageRole.TOOL else None
    Becomes: blob = None
    """
    body = _big_text()
    store = SessionStore(tmp_path / "sessions")
    llm = _ScriptedLLM(
        [
            [
                ToolCallRequest(id="c1", name="dump", arguments={}),
                ToolCallRequest(id="c2", name="small", arguments={}),
            ]
        ]
    )
    agent = _agent(tmp_path, store, llm, [_returning("dump", body), _returning("small", "ok")])
    assert (await agent.execute_turn("go")).is_completed
    agent.persist_session()

    state = store.load(_SID)
    assert state is not None
    assert [e.id for e in state.session_log] == [f"e{i}" for i in range(len(state.session_log))]
    assert _logged_messages(store, state) == list(state.messages)
    assert [e.kind for e in state.session_log] == [
        SessionLogKind.SYSTEM,
        SessionLogKind.UTTERANCE,
        SessionLogKind.TOOL_CALL,
        SessionLogKind.TOOL_RESULT,
        SessionLogKind.TOOL_RESULT,
        SessionLogKind.UTTERANCE,
    ]
    assert {e.provenance for e in state.session_log} == {SessionLogProvenance.RECORDED}
    assert [e.turn for e in state.session_log] == [0, 1, 1, 1, 1, 1]

    dump_entry, small_entry = state.session_log[3], state.session_log[4]
    excerpt = state.messages[3].content
    assert excerpt is not None and len(excerpt.encode("utf-8")) <= TOOL_RESULT_CAP_BYTES
    assert dump_entry.blob is not None and dump_entry.blob == handle_in(excerpt)
    assert load_tool_result(artifacts_dir_for(tmp_path), _SID, dump_entry.blob) == body
    assert small_entry.blob is None  # kept whole: the entry's body is the result


@pytest.mark.asyncio
async def test_a_record_from_before_the_log_is_backfilled_once(tmp_path: Path) -> None:
    """A legacy record's messages become `migrated` entries; loading it again adds none.

    Two identical user prompts are two entries: accounting is by occurrence, not by text.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: if entry is None and pools.get(item.digest):
    Becomes: if False:
    """
    store = SessionStore(tmp_path / "sessions")
    _legacy_record(store, tmp_path)

    agent = _agent(tmp_path, store, _ScriptedLLM())
    assert agent.hydrate_session() is not None
    agent.persist_session()
    first = store.load(_SID)
    assert first is not None
    assert len(first.session_log) == 5
    assert {e.provenance for e in first.session_log} == {SessionLogProvenance.MIGRATED}
    assert {e.turn for e in first.session_log} == {None}
    assert _logged_messages(store, first) == list(first.messages)

    again = _agent(tmp_path, store, _ScriptedLLM())
    assert again.hydrate_session() is not None
    again.persist_session()
    second = store.load(_SID)
    assert second is not None
    assert second.session_log == first.session_log

    assert (await again.execute_turn("once more")).is_completed
    again.persist_session()
    third = store.load(_SID)
    assert third is not None
    assert third.session_log[:5] == first.session_log
    added = third.session_log[5:]
    assert [(e.kind, e.turn, e.provenance) for e in added] == [
        (SessionLogKind.UTTERANCE, 3, SessionLogProvenance.RECORDED),
        (SessionLogKind.UTTERANCE, 3, SessionLogProvenance.RECORDED),
    ]


@pytest.mark.asyncio
async def test_a_rolled_back_turn_leaves_the_history_and_stays_in_the_log(
    tmp_path: Path,
) -> None:
    """The undone turn's prompt, call, result and reply are gone from `messages`, not the log."""
    store = SessionStore(tmp_path / "sessions")
    llm = _ScriptedLLM([[ToolCallRequest(id="c1", name="small", arguments={})]])
    agent = _agent(tmp_path, store, llm, [_returning("small", "ok")])
    checkpoint = agent.checkpoint_turn()
    assert (await agent.execute_turn("undo me")).is_completed
    assert agent.roll_back_turn(checkpoint, reason="test") == 4
    agent.persist_session()

    state = store.load(_SID)
    assert state is not None
    assert len(state.messages) == 1
    logged = _logged_messages(store, state)
    assert logged[: len(state.messages)] == list(state.messages)
    assert [m.role for m in logged[1:]] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert logged[1].content == "undo me"


@pytest.mark.asyncio
async def test_after_compaction_a_repeated_message_is_logged_at_its_own_turn(
    tmp_path: Path,
) -> None:
    """The summary is logged at the turn that made it; turn 5's `yes` and `done` are logged
    although identical messages were folded away, because they entered again.

    Compared in order, entry by entry: a count of messages against `messages` misses both.
    What decides it is that the log is matched against the history it last held, the
    compacted one, not against every entry it ever took: matched against the whole log,
    turn 5's `yes` and `done` match entries of earlier turns and are not logged. (Keeping the
    compacted history out of the driver's working copy fails this test too, but by
    crashing: the derived entries no longer match the history.)

    Killed by: src/uclone_x/agent/session_lifecycle.py :: if self.aligned is None:
    Becomes: if True:
    """
    store = SessionStore(tmp_path / "sessions")
    agent = _agent(tmp_path, store, _ScriptedLLM())
    for prompt in ("yes", "a", "b", "c"):
        assert (await agent.execute_turn(prompt)).is_completed
    result = await agent.compact_session()
    assert result.messages_after < result.messages_before, "nothing was folded"
    assert (await agent.execute_turn("yes")).is_completed
    agent.persist_session()

    state = store.load(_SID)
    assert state is not None
    assert _log_lines(store, state) == [
        ("system", 0, "You keep a"),
        ("utterance", 1, "yes"),
        ("utterance", 1, "done"),
        ("utterance", 2, "a"),
        ("utterance", 2, "done"),
        ("utterance", 3, "b"),
        ("utterance", 3, "done"),
        ("utterance", 4, "c"),
        ("utterance", 4, "done"),
        ("summary", 4, "[Context A"),
        ("utterance", 5, "yes"),
        ("utterance", 5, "done"),
    ]


@pytest.mark.asyncio
async def test_after_the_history_is_replaced_a_repeated_message_is_logged(
    tmp_path: Path,
) -> None:
    """Turn 1 is replaced away; turn 2 says the same and is logged at turn 2.

    Killed by: src/uclone_x/agent/base.py :: self._active_session.log_history()  # and what replaced it
    Becomes: pass
    """
    store = SessionStore(tmp_path / "sessions")
    agent = _agent(tmp_path, store, _ScriptedLLM())
    assert (await agent.execute_turn("yes")).is_completed
    agent._history = agent._history[:1]  # pyright: ignore[reportPrivateUsage]
    assert (await agent.execute_turn("yes")).is_completed
    agent.persist_session()

    state = store.load(_SID)
    assert state is not None
    assert _log_lines(store, state) == [
        ("system", 0, "You keep a"),
        ("utterance", 1, "yes"),
        ("utterance", 1, "done"),
        ("utterance", 2, "yes"),
        ("utterance", 2, "done"),
    ]


@pytest.mark.asyncio
async def test_a_room_seat_is_logged_on_the_same_path(tmp_path: Path) -> None:
    """A seat the room resolver builds logs its session exactly as a 1:1 agent does."""
    store = SessionStore(tmp_path / "sessions")
    host = HostDependencies(
        bus=EventBus(),
        llm=_ScriptedLLM(),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=store,
    )
    seat = Participant(
        id="author",
        kind=ParticipantKind.AGENT,
        display_name="Author",
        session_id="sess_room__r1__author",
    )
    agent = await RoomAgentResolver(host).resolve(seat)
    assert isinstance(agent, BaseAgent)
    assert (await agent.execute_turn("Open the chapter.")).is_completed
    agent.persist_session(seat.session_id)

    state = store.load(seat.session_id)
    assert state is not None
    assert len(state.session_log) == len(state.messages) >= 3
    assert _logged_messages(store, state) == list(state.messages)


def test_a_log_with_a_missing_or_reordered_entry_is_refused() -> None:
    """Entry n is `e<n>`: a dropped or reordered entry cannot reach the store."""
    entry = SessionLogEntry(
        id="e1",
        kind=SessionLogKind.UTTERANCE,
        turn=1,
        digest="0" * 64,
        size=1,
        provenance=SessionLogProvenance.RECORDED,
    )
    with pytest.raises(ValidationError, match="append-only"):
        SessionState(session_id=_SID, agent_id="a", session_log=(entry,))
