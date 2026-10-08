"""The session log: every message that entered the history, and every tool body in full (#1443).

PR 1 of 3. The log is written beside `messages` on the one path every history change
takes, persisted on the record, and read by nothing that builds a request, so requests are
unchanged (`test_room_request_conformance.py` and `test_one_clone_builder.py` pass as they
were). These tests pin what the log itself must hold:

*   every message `messages` holds has an entry whose stored body is that message, and an
    over-cap tool result's entry names the full body, not the excerpt the history keeps;
*   a record holding an entry an older build backfilled as `migrated` is unreadable,
    not empty;
*   a rolled-back turn's messages leave the history and stay in the log;
*   after compaction or a replaced history, a message identical to one that left is
    logged when it enters again, at its own turn;
*   a room seat's session is logged the same way as a 1:1 session.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, AgentState
from uclone_x.agent.request_record import LogReader
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.agent.session_lifecycle import (
    AnchorWriter,
    _LiveSession,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.core.context_state import ContextEntry, ContextForm
from uclone_x.core.provenance import Provenance
from uclone_x.core.session_log import (
    SessionLogEntry,
    SessionLogKind,
    SessionLogProvenance,
    is_result_message,
    kept_result_text,
    logged_message,
    new_entry,
    stored_result_entry,
    tool_result_kind,
)
from uclone_x.core.tool_results import (
    TOOL_RESULT_CAP_BYTES,
    handle_in,
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


def _is_full_text(entry: SessionLogEntry) -> bool:
    """Whether `entry`'s body is a kept tool result's full text, not a message (#1848)."""
    return entry.blob is not None and entry.digest.startswith(entry.blob.removeprefix("tr_"))


def _message_entries(state: SessionState) -> list[SessionLogEntry]:
    return [e for e in state.session_log if not _is_full_text(e)]


def _logged_messages(store: SessionStore, state: SessionState) -> list[ChatMessage]:
    """Each message entry's stored body, read back as the message it names."""
    out: list[ChatMessage] = []
    for entry in _message_entries(state):
        body = store.load_context_body(state.session_id, entry.digest)
        assert body is not None, f"{entry.id} names a body that is not stored"
        assert len(body.encode("utf-8")) == entry.size
        out.append(ChatMessage.model_validate_json(body))
    return out


def _as_logged(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """The messages as the log keeps them: a form records its source, not its text (#1848)."""
    return [m.model_copy(update={"content": None}) if m.form is not None else m for m in messages]


def _log_lines(store: SessionStore, state: SessionState) -> list[tuple[str, int | None, str]]:
    """The log in order: each entry's kind, turn and message text, the summary shortened."""
    return [
        (entry.kind.value, entry.turn, (message.content or "")[:10])
        for entry, message in zip(
            _message_entries(state), _logged_messages(store, state), strict=True
        )
    ]


@pytest.mark.asyncio
async def test_the_log_holds_every_history_message_and_every_tool_body_in_full(
    tmp_path: Path,
) -> None:
    """Every message `messages` holds is a logged body; an excerpted result names its blob.

    The history keeps an over-cap result as an excerpt (#1422). The log entry for it records
    the form and the `tr_` handle it was rendered from, not the excerpt's text (#1848), and that handle loads the whole result, so nothing the tool returned
    is lost to the log even where the request carries only its head and tail.

    Killed by: src/uclone_x/core/session_log.py :: blob: str | None = message.rendered_from.handle
    Becomes: blob: str | None = None
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
    assert _logged_messages(store, state) == _as_logged(state.messages)
    messages = _message_entries(state)
    assert [e.kind for e in messages] == [
        SessionLogKind.SYSTEM,
        SessionLogKind.UTTERANCE,
        SessionLogKind.TOOL_CALL,
        SessionLogKind.TOOL_RESULT,
        SessionLogKind.TOOL_RESULT,
        SessionLogKind.UTTERANCE,
    ]
    assert {e.provenance for e in state.session_log} == {SessionLogProvenance.RECORDED}
    assert [e.turn for e in messages] == [0, 1, 1, 1, 1, 1]

    dump_entry, small_entry = messages[3], messages[4]
    # The record holds the excerpt as its form, with no text; the request carried the text.
    assert state.messages[3].content is None and state.messages[3].form == "excerpt"
    (excerpt,) = [m.content for m in llm.requests[-1].messages if m.tool_call_id == "c1"]
    assert excerpt is not None and len(excerpt.encode("utf-8")) <= TOOL_RESULT_CAP_BYTES
    assert dump_entry.blob is not None and dump_entry.blob == handle_in(excerpt)
    logged = _logged_messages(store, state)[3]
    assert logged.content is None and logged.form == "excerpt"
    assert logged.rendered_from is not None and logged.rendered_from.handle == dump_entry.blob
    # The whole result is an entry of its own, whose body is the text itself (#1848).
    (full,) = [e for e in state.session_log if _is_full_text(e)]
    assert full is stored_result_entry(state.session_log, dump_entry.blob)
    assert full.kind is SessionLogKind.TOOL_RESULT and full.turn == 1
    assert store.load_context_body(_SID, full.digest) == body
    assert small_entry.blob is None  # kept whole: the entry's body is the result


@pytest.mark.asyncio
async def test_a_record_with_a_migrated_entry_is_unreadable_not_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An entry an older build backfilled (`migrated`, no turn) is not a shape this build reads.

    It takes the store's unreadable-record path: `load` reports it and reads it as absent,
    the listing omits it with a warning, and nothing on disk changes -- it is not shown as
    an empty session, and the listing does not fail.
    """
    store = SessionStore(tmp_path / "sessions")
    agent = _agent(tmp_path, store, _ScriptedLLM())
    assert (await agent.execute_turn("hello")).is_completed
    agent.persist_session()
    path = store.session_path(_SID)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["session_log"][0] = {**raw["session_log"][0], "turn": None, "provenance": "migrated"}
    written = json.dumps(raw)
    path.write_text(written, encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert store.load(_SID) is None
        assert store.list_session_ids() == ()
        assert _agent(tmp_path, store, _ScriptedLLM()).hydrate_session() is None
    assert "cannot be read" in caplog.text
    assert "omitting from listing" in caplog.text
    assert path.read_text(encoding="utf-8") == written


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
    What decides it is that a message appended is logged because it entered, whatever its
    text: matched by digest against every entry the log ever took, turn 5's `yes` and
    `done` would reuse entries of earlier turns and not be logged. (Keeping the compacted
    history out of the driver's working copy fails this test too, but by crashing: the
    derived entries no longer match the history.)

    Killed by: src/uclone_x/agent/session_lifecycle.py :: self._log(self._logged(message))
    Becomes: any(e.digest == self._logged(message).digest for e in self.session_log) or self._log(self._logged(message))
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

    Killed by: src/uclone_x/agent/session_lifecycle.py :: self._log(self._logged(message))
    Becomes: any(e.digest == self._logged(message).digest for e in self.session_log) or self._log(self._logged(message))
    """
    store = SessionStore(tmp_path / "sessions")
    agent = _agent(tmp_path, store, _ScriptedLLM())
    assert (await agent.execute_turn("yes")).is_completed
    agent.load_history(agent.history[:1])
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
    assert _logged_messages(store, state) == _as_logged(state.messages)


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


# --------------------------------------------------------------------------------------
# #1848: the history's one door. Every write logs as it writes, and declares a cause
# only when it changes something the current epoch showed.
# --------------------------------------------------------------------------------------


def _door_session(*contents: str) -> _LiveSession:
    state = SessionState(
        session_id=_SID,
        agent_id="a",
        messages=tuple(ChatMessage(role=MessageRole.USER, content=c) for c in contents),
    )
    return _LiveSession.from_state(
        state, anchor_provenance=AnchorWriter.CALLER, load_body=lambda _digest: None
    )


def _show_all(live: _LiveSession) -> None:
    """Record a request that showed the whole history, as a turn's request does."""
    live.record_shown(
        [ContextEntry(entry=entry, form=ContextForm.FULL) for entry in live.entry_ids()],
        step=0,
    )


def _user(content: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content=content)


def test_the_history_is_derived_from_the_log_and_cannot_be_assigned() -> None:
    """`messages` is each history entry's logged body, decoded; nothing assigns it."""
    live = _door_session("one")
    live.append(_user("two"))

    bodies = {entry.id: entry.digest for entry in live.session_log}
    assert [bodies[e] for e in live.entry_ids()] == [
        logged_message(m).digest for m in live.messages
    ]
    assert [m.content for m in live.messages] == ["one", "two"]
    with pytest.raises(AttributeError):
        live.messages = ()  # pyright: ignore[reportAttributeAccessIssue]


def test_an_append_logs_at_the_write_and_declares_nothing() -> None:
    """Killed by: src/uclone_x/agent/session_lifecycle.py :: self._log(self._logged(message))
    Becomes: self._logged(message)
    """
    live = _door_session("one")
    _show_all(live)
    before = len(live.session_log)

    live.append(_user("one"), _user("two"))

    assert len(live.session_log) == before + 2
    assert live.entry_ids()[-2:] == [e.id for e in live.session_log[-2:]]
    assert live.epoch_causes == []


def test_a_rewrite_declares_its_cause_only_for_a_message_the_epoch_showed() -> None:
    """Killed by: src/uclone_x/agent/session_lifecycle.py :: if self._shows_any((entry,)):
    Becomes: if False:
    """
    live = _door_session("shown")
    _show_all(live)
    live.append(_user("not shown yet"))

    live.replace(1, _user("rewritten before any request"), cause="post_turn_hook")
    assert live.epoch_causes == []
    live.replace(0, _user("rewritten after a request"), cause="artifact_sanitized")
    assert live.epoch_causes == ["artifact_sanitized"]
    assert [m.content for m in live.messages] == [
        "rewritten after a request",
        "rewritten before any request",
    ]
    # What was replaced stays in the log.
    assert len(live.session_log) == 4


def test_a_rewrite_of_a_message_a_compaction_pruned_declares_its_cause() -> None:
    """A message a compaction pruned shows an entry the epoch showed, in a smaller form;
    rewriting it before the next request records it still changes what the epoch showed,
    so the cause is declared (#1971, item 6).

    Killed by: src/uclone_x/agent/session_lifecycle.py :: return derived is not None and derived.entry in ids
    Becomes: return False
    """
    live = _door_session("asked", "a long answer")
    _show_all(live)
    shown = live.entry_ids()
    live.replace_history([_user("asked"), _user("a shorter answer")], cause="compaction")
    pruned = live.entry_ids()[1]
    assert pruned not in shown
    live.compacted_entries = {
        pruned: ContextEntry(entry=shown[1], form=ContextForm.FULL, rendering=pruned)
    }

    live.replace(1, _user("rewritten after the compaction"), cause="artifact_sanitized")

    assert live.epoch_causes == ["compaction", "artifact_sanitized"]


def test_a_rewrite_to_the_same_text_changes_nothing() -> None:
    live = _door_session("same")
    _show_all(live)
    entries, logged = live.entry_ids(), len(live.session_log)

    live.replace(0, _user("same"), cause="reply_lines")

    assert (live.entry_ids(), len(live.session_log), live.epoch_causes) == (entries, logged, [])


def test_a_cut_declares_its_cause_only_when_it_takes_out_what_was_shown() -> None:
    """Killed by: src/uclone_x/agent/session_lifecycle.py :: if self._shows_any(entries[length:]):
    Becomes: if True:
    """
    live = _door_session("shown")
    _show_all(live)
    live.append(_user("refused step"))

    live.truncate(1, cause="step_refused")
    assert live.epoch_causes == []
    assert [m.content for m in live.messages] == ["shown"]
    live.truncate(0, cause="tool_step_dropped")
    assert live.epoch_causes == ["tool_step_dropped"]


def test_a_replaced_history_keeps_the_entries_of_what_stayed() -> None:
    """Killed by: src/uclone_x/agent/session_lifecycle.py :: entries.append(pool.pop() if pool else self._log(item))
    Becomes: entries.append(self._log(item))
    """
    live = _door_session("a", "b", "c")
    _show_all(live)
    a, _b, c = live.entry_ids()
    logged = len(live.session_log)

    live.replace_history([_user("a"), _user("summary"), _user("c")], cause="compaction")

    assert live.entry_ids()[0] == a and live.entry_ids()[2] == c
    assert len(live.session_log) == logged + 1
    assert live.epoch_causes == ["compaction"]


def test_a_replaced_history_that_only_extends_declares_nothing() -> None:
    live = _door_session("a")
    _show_all(live)

    live.replace_history([_user("a"), _user("b")], cause="history_replaced")

    assert live.epoch_causes == []
    assert [m.content for m in live.messages] == ["a", "b"]


def test_a_kept_result_is_read_once_by_the_readers_of_one_session() -> None:
    """Kept results are shared by every reader of a session, like decoded messages (#1971)."""
    live = _door_session("a")
    text = "\n".join(f"row {i}" for i in range(50))
    handle = live.keep_result_body(text, kind=tool_result_kind(None))
    reads: list[str] = []

    def load(digest: str) -> str | None:
        reads.append(digest)
        return live.pending_bodies.get(digest)

    for _ in range(3):
        reader = LogReader(
            live.session_log, load, purpose="send", decoded=live.decoded, results=live.results
        )
        assert reader.result_of(handle) == text
    assert len(reads) == 1
    live.forget_result_bodies()
    assert live.results == {}


def _tool_message(content: str, name: str = "fetch") -> ChatMessage:
    return ChatMessage(role=MessageRole.TOOL, content=content, name=name, tool_call_id="c1")


def test_a_result_the_history_holds_whole_is_kept_by_naming_its_message() -> None:
    """Compaction's offload of a result that fit logs no second copy of it (#2013).

    The handle names the message's own entry -- its digest's first 16 hex digits -- and
    reading it gives the message's text, through the live session and the log reader.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: if whole is not None:
    Becomes: if False:
    """
    live = _door_session("q")
    text = _big_text()
    live.append(_tool_message(text))
    entry = live.session_log[-1]
    before = len(live.session_log)

    handle = live.keep_result_body(text, kind=tool_result_kind("fetch"))

    assert handle == "tr_" + entry.digest[:16]
    assert len(live.session_log) == before
    assert text not in live.pending_bodies.values()
    assert live.result_body(handle, lambda _d: None) == text
    reader = LogReader(live.session_log, live.pending_bodies.get, purpose="send")
    assert reader.result_of(handle) == text


def test_a_text_the_history_does_not_hold_whole_is_still_kept_as_its_own_body() -> None:
    """Text no message of the history holds -- an over-cap result at ingest, a memory
    section, or a message's text of another kind -- is kept as before (#1848, #2013).
    """
    live = _door_session("q")
    live.append(_tool_message("short"))
    text = _big_text()
    before = len(live.session_log)

    handle = live.keep_result_body(text, kind=tool_result_kind("fetch"))
    assert len(live.session_log) == before + 1
    assert live.session_log[-1].blob == handle
    assert live.pending_bodies[live.session_log[-1].digest] == text

    # A user message with the same text is not a tool result the handle may name.
    live.append(_user("typed by a person"))
    kept = live.keep_result_body("typed by a person", kind=tool_result_kind(None))
    assert live.session_log[-1].blob == kept


def test_a_handle_to_a_message_is_refused_after_a_delete() -> None:
    """A delete forgets a result held by its message as it forgets a kept one (#1848)."""
    live = _door_session("q")
    text = _big_text()
    live.append(_tool_message(text))
    handle = live.keep_result_body(text, kind=tool_result_kind("fetch"))

    live.forget_result_bodies()

    assert live.result_body(handle, lambda _d: None) is None


def test_a_handle_to_an_entry_that_is_not_a_whole_result_reads_nothing() -> None:
    """Only a tool result message logged whole reads as a result by its digest (#2013).

    A user message's entry, a form's entry, or a body that is not a message's JSON is
    never read as the text a handle names.
    """
    tool = logged_message(_tool_message("whole"))
    user = logged_message(_user("hello"))
    entries = [
        SessionLogEntry(
            id=f"e{i}",
            kind=rendered.kind,
            turn=0,
            digest=rendered.digest,
            size=len(rendered.body),
            provenance=SessionLogProvenance.RECORDED,
            blob=rendered.blob,
        )
        for i, rendered in enumerate((tool, user))
    ]
    assert stored_result_entry(entries, "tr_" + tool.digest[:16]) is entries[0]
    assert stored_result_entry(entries, "tr_" + user.digest[:16]) is None
    assert kept_result_text(entries[0], tool.body) == "whole"
    assert kept_result_text(entries[0], "not json") is None
    assert is_result_message(entries[0]) and not is_result_message(entries[1])


def test_saved_session_omits_messages_and_reconstructs_from_context_bodies(tmp_path: Path) -> None:
    """A saved session with history_entries and session_log omits messages from disk,
    and SessionStore.load reconstructs them from the context body store (#2081).

    Killed by: src/uclone_x/core/session_state.py :: if data.get("history_entries") and data.get("session_log"):
    Becomes: if False:
    """
    store = SessionStore(tmp_path / "store")
    user_msg = ChatMessage(role=MessageRole.USER, content="hello from user")
    asst_msg = ChatMessage(role=MessageRole.ASSISTANT, content="reply from agent")
    log_entries: list[SessionLogEntry] = []
    for i, m in enumerate((user_msg, asst_msg)):
        logged = logged_message(m)
        store.save_context_body("sess_2081", logged.digest, logged.body)
        log_entries.append(new_entry(i, logged, turn=1, provenance=SessionLogProvenance.RECORDED))

    state = SessionState(
        session_id="sess_2081",
        agent_id="agent_2081",
        messages=(user_msg, asst_msg),
        session_log=tuple(log_entries),
        history_entries=("e0", "e1"),
    )
    saved = store.save(state)
    assert saved.messages == (user_msg, asst_msg)

    # On disk: session.json does NOT contain "messages"
    path = store.session_path("sess_2081")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "messages" not in raw
    assert raw["history_entries"] == ["e0", "e1"]

    # On load: store.load reconstructs messages from context bodies
    loaded = store.load("sess_2081")
    assert loaded is not None
    assert loaded.messages == (user_msg, asst_msg)
    assert loaded.history_entries == ("e0", "e1")


def test_saved_session_with_missing_context_body_is_unreadable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If a context body is missing, load treats the record as unreadable (#2081).

    Killed by: src/uclone_x/agent/session.py :: if body is None:
    Becomes: if False:
    """
    store = SessionStore(tmp_path / "store")
    user_msg = ChatMessage(role=MessageRole.USER, content="hello")
    logged = logged_message(user_msg)
    # Don't save the body to context_body_dir
    entry = new_entry(0, logged, turn=1, provenance=SessionLogProvenance.RECORDED)
    state = SessionState(
        session_id="sess_missing_body",
        agent_id="agent",
        messages=(user_msg,),
        session_log=(entry,),
        history_entries=("e0",),
    )
    store.save(state)

    with caplog.at_level(logging.WARNING, logger="uclone_x.agent.session"):
        assert store.load("sess_missing_body") is None
    assert "no context body" in caplog.text
