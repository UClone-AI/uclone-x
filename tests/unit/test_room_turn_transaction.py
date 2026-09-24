"""A seat's turn commits in three places or in none (#1423).

A room turn changes three records: the seat's own session (what the model is shown next),
the transcript row (what the room says happened), and the seat's `last_seen_seq` (how
much of the room it has answered). Only the last of these followed the rule "a failed
turn changes nothing": the session kept whatever the turn had appended before it failed.
The audit that opened #1423 found two consequences, and each is reproduced here against a
real `BaseAgent`:

* **The span went in twice.** A turn that ran a tool and then failed left its prompt, the
  tool call and the result in the seat's history. The retry rendered the same unseen span
  and appended it again -- the guard from #969 recognises only an identical prompt left
  *last*, and here a tool result came after it. The same happened on the seat's next turn
  after a human spoke, because that span is the old one plus the new message.
* **A stopped turn left a call nothing answered.** The step appends the assistant message
  that asks for tools before it awaits them, so Stop during a tool left that message in
  history with no `TOOL` result after it, and every provider refuses such a conversation.

What these pin, beside the two reproductions: the room undoes a turn it could not save out
of the seat as well; the rollback is in the log and the requests still rebuild from it;
and outside a room, where nothing rolls a turn back, the agent itself drops an unanswered
tool step rather than inventing results for it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.agent.request_record import rebuild_requests
from uclone_x.agent.session import EVENT_LOG_SUBDIR, SessionStore
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.errors import LLMProviderError, TurnNotLandedError
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.log.reader import read_session_log
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomPolicy,
    RoomState,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerRequest,
)
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.store import RoomStore
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry
from uclone_x.ui.rooms import (
    _http_error,  # pyright: ignore[reportPrivateUsage]
    reader_facing_reason,
)

ALICE = Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice")
SCOUT = Participant(
    id="scout",
    kind=ParticipantKind.AGENT,
    display_name="Scout",
    persona_summary="scout does scout things",
    session_id="sess_room__r1__scout",
    ontology_namespace="https://uclone-x.ai/ontology/r1/scout",
)

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="scripted", model="scripted"),
    served_by=ServiceRef(provider="scripted", model="scripted"),
    attempts=(),
)
_USAGE = TokenUsage(provider="scripted", model="scripted", input_tokens=0, output_tokens=0)

#: One model call: a reply, a tool call, a whole response, or a provider failure.
Step = str | ToolCallRequest | ModelResponse | Exception


class _Script(BaseLLMConnector):
    """Answers each call with the next step, and keeps every request it was sent."""

    def __init__(self, steps: Sequence[Step]) -> None:
        super().__init__()
        self._steps = list(steps)
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ModelResponse):
            return step
        if isinstance(step, ToolCallRequest):
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                content=None,
                tool_calls=(step,),
                usage=_USAGE,
                provenance=_PROV,
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content=step,
            tool_calls=(),
            usage=_USAGE,
            provenance=_PROV,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        response = await self.generate(request)
        yield StreamChunk(delta_content=response.content or "")


class _NoteParams(BaseModel):
    text: str = Field(default="")


class _NoteTool(BaseTool[_NoteParams]):
    """Returns at once: a tool step that completes."""

    name = "note"
    description = "Takes a note"

    def run(self, params: _NoteParams, context: ToolContext) -> dict[str, Any]:
        return {"noted": params.text}


class _MemoTool(_NoteTool):
    """A second tool, so two attempts can call different tools under one call id."""

    name = "memo"
    description = "Keeps a memo"


class _HeldTool(BaseTool[_NoteParams]):
    """Runs until released, so a turn can be stopped while its tool is running."""

    name = "held"
    description = "Waits until released"

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, params: _NoteParams, context: ToolContext) -> dict[str, Any]:
        self.started.set()
        await self.release.wait()
        return {"done": True}


def _call(call_id: str, name: str) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments={"text": "x"})


def _seat(llm: _Script, sessions: SessionStore | None, *tools: BaseTool[Any]) -> BaseAgent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return BaseAgent(
        config=AgentConfig(
            agent_id=SCOUT.id,
            name=SCOUT.id,
            llm_config=AgentLLMConfig(model_name="scripted"),
        ),
        llm=llm,
        tools=registry,
        context=AgentContext(session_id=SCOUT.session_id, agent_id=SCOUT.id),
        store=sessions,
    )


class _Seats:
    def __init__(self, agent: BaseAgent) -> None:
        self.agent = agent

    async def resolve(self, participant: Participant) -> BaseAgent:
        return self.agent


class _ScoutSpeaks:
    """Gives the floor to the scout every time it is asked; the room's cap ends the loop."""

    @property
    def name(self) -> str:
        return "scout-speaks"

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        return SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id=SCOUT.id,
            selector=self.name,
            confidence=1.0,
        )


def _room(tmp_path: Path, agent: BaseAgent, rooms: RoomStore | None = None) -> RoomOrchestrator:
    store = rooms if rooms is not None else RoomStore(tmp_path / "rooms")
    store.save(
        RoomState(
            room_id="r1",
            participants=(ALICE, SCOUT),
            policy=RoomPolicy(max_agent_turns_per_human_message=1),
        )
    )
    return RoomOrchestrator(store=store, selectors=[_ScoutSpeaks()], resolver=_Seats(agent))


def _unanswered(messages: Sequence[ChatMessage]) -> list[str]:
    """Every tool call in `messages` that no `TOOL` message answers."""
    answered = {m.tool_call_id for m in messages if m.role is MessageRole.TOOL}
    return [
        call.id
        for m in messages
        if m.role is MessageRole.ASSISTANT
        for call in m.tool_calls
        if call.id not in answered
    ]


def _before_last_user(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """What a request sends ahead of its newest prompt: the prefix a retry must not change."""
    last = max(i for i, m in enumerate(messages) if m.role is MessageRole.USER)
    return list(messages[:last])


def _times_shown(request: LLMRequest, words: str) -> int:
    return sum(str(m.content).count(words) for m in request.messages)


def _only_log(sessions: SessionStore) -> Path:
    logs = sorted((sessions.storage_dir / EVENT_LOG_SUBDIR).glob("*.jsonl"))
    assert len(logs) == 1, logs
    return logs[0]


# --------------------------------------------------------------------------------------
# Finding 1: a failed turn left its span in the seat, so the span was sent twice.
# --------------------------------------------------------------------------------------


class TestAFailedTurnLeavesNothingInTheSeat:
    @pytest.mark.asyncio
    async def test_a_retry_after_a_tool_step_shows_the_span_once_on_an_unchanged_prefix(
        self, tmp_path: Path
    ) -> None:
        """The audit's first finding, reproduced: the retry sent the span a second time.

        The failed turn ran a tool before its provider failed, so its history ended on a
        `TOOL` result and #969's guard -- an identical prompt left *last* -- did not see
        the prompt as repeated. The retry appended it again, after the failed turn's call
        and result: the model was shown the question twice and a tool round it had not
        asked for in this attempt, and the prefix it had been shown was no longer the
        prefix of the next request.

        Killed by: src/uclone_x/room/orchestrator.py :: rollback_error = None if committed else self._roll_back_seat(agent, speaker, checkpoint)
        Becomes: rollback_error = None
        Killed by: src/uclone_x/agent/base.py :: live.messages = list(checkpoint.messages)
        Becomes: pass
        """
        llm = _Script(
            [
                "hello, Alice",
                _call("c1", "note"),
                LLMProviderError("provider unavailable (scripted)"),
                "scout says TTL",
            ]
        )
        agent = _seat(llm, SessionStore(tmp_path / "sessions"), _NoteTool())
        orch = _room(tmp_path, agent)
        answered = await orch.post("r1", "alice", "hi")
        before = agent.history
        seen_before = answered.last_seen_seq.get(SCOUT.id)

        failed = await orch.post("r1", "alice", "discuss the cache")

        assert failed.transcript[-1].error is not None
        assert agent.history == before, "the failed turn stayed in the seat's history"
        assert failed.last_seen_seq.get(SCOUT.id) == seen_before

        retried = await orch.retry("r1")

        assert retried.transcript[-1].content == "scout says TTL"
        failed_first, retry = llm.requests[1], llm.requests[3]
        assert _times_shown(retry, "discuss the cache") == 1, [m.content for m in retry.messages]
        assert _unanswered(retry.messages) == []
        assert [m.model_dump_json() for m in _before_last_user(retry.messages)] == [
            m.model_dump_json() for m in _before_last_user(failed_first.messages)
        ]
        assert retried.last_seen_seq.get(SCOUT.id) != seen_before

    @pytest.mark.asyncio
    async def test_the_next_turn_after_a_human_speaks_does_not_show_the_failed_span_again(
        self, tmp_path: Path
    ) -> None:
        """The same finding by the other door: no retry, the human just speaks again.

        The failed turn did not advance `last_seen_seq`, so the next span is the failed
        one plus the new message -- correct for the room, and a second copy of the old
        message for a seat that had kept the first. The prompts differ, so #969's guard
        never applied here at all.

        Killed by: src/uclone_x/room/orchestrator.py :: rollback_error = None if committed else self._roll_back_seat(agent, speaker, checkpoint)
        Becomes: rollback_error = None
        """
        llm = _Script([LLMProviderError("provider unavailable (scripted)"), "noted both"])
        agent = _seat(llm, SessionStore(tmp_path / "sessions"))
        orch = _room(tmp_path, agent)

        failed = await orch.post("r1", "alice", "discuss the cache")
        assert failed.transcript[-1].error is not None
        answered = await orch.post("r1", "alice", "and the queue")

        assert answered.transcript[-1].content == "noted both"
        second = llm.requests[1]
        assert _times_shown(second, "discuss the cache") == 1, [m.content for m in second.messages]
        assert [m.role for m in second.messages].count(MessageRole.USER) == 1

    @pytest.mark.asyncio
    async def test_the_rollback_is_in_the_log_and_every_request_still_rebuilds(
        self, tmp_path: Path
    ) -> None:
        """Undone in the conversation, kept in the record (P6).

        The failed turn's events stay in the log, a `TURN_ROLLED_BACK` event says what
        left the conversation, and the request log (#1421) still rebuilds the retry's
        request exactly: the next `REQUEST_CONTEXT` records the rewind as a shorter kept
        prefix, not as a gap.

        Killed by: src/uclone_x/agent/base.py :: "type": "TURN_ROLLED_BACK",
        Becomes: "type": "TURN_UNDONE",
        """
        sessions = SessionStore(tmp_path / "sessions")
        llm = _Script(
            [_call("c1", "note"), LLMProviderError("provider unavailable (scripted)"), "done"]
        )
        agent = _seat(llm, sessions, _NoteTool())
        orch = _room(tmp_path, agent)

        await orch.post("r1", "alice", "discuss the cache")
        await orch.retry("r1")

        events = list(read_session_log(_only_log(sessions)))
        rolled_back = [e for e in events if e["type"] == "TURN_ROLLED_BACK"]
        assert len(rolled_back) == 1, [e["type"] for e in events]
        # The prompt, the step that asked for the tool, and its result.
        assert rolled_back[0]["dropped_message_count"] == 3
        assert "TOOL_CALL" in [e["type"] for e in events]

        state = sessions.load(SCOUT.session_id)
        assert state is not None
        assert _unanswered(state.messages) == []
        rebuilt = rebuild_requests(sessions, state, events)
        assert len(rebuilt) == len(llm.requests)
        assert [m.model_dump_json() for m in rebuilt[-1].request.messages] == [
            m.model_dump_json() for m in llm.requests[-1].messages
        ]


# --------------------------------------------------------------------------------------
# Finding 2: a stopped turn left an assistant message whose tool calls nothing answered.
# --------------------------------------------------------------------------------------


class TestAStoppedTurnLeavesNoUnansweredCall:
    @pytest.mark.asyncio
    async def test_stop_during_a_tool_leaves_the_seat_as_it_was_on_disk_and_in_memory(
        self, tmp_path: Path
    ) -> None:
        """The audit's second finding, reproduced through the room's own Stop.

        Stop cancelled the turn while its tool ran, after the step had put its request for
        the tool in history and before any result. The seat was written that way, so the
        next turn sent a conversation with a call nothing answered. Now the turn is rolled
        back like any other that did not commit, and the mark is not advanced either:
        the session, the row and `last_seen_seq` agree that the turn did not happen.

        Killed by: src/uclone_x/room/orchestrator.py :: rollback_error = None if committed else self._roll_back_seat(agent, speaker, checkpoint)
        Becomes: rollback_error = None
        """
        sessions = SessionStore(tmp_path / "sessions")
        held = _HeldTool()
        agent = _seat(_Script([_call("c1", "held"), "unused"]), sessions, held)
        before = agent.history
        orch = _room(tmp_path, agent)

        post = asyncio.create_task(orch.post("r1", "alice", "write it down"))
        await asyncio.wait_for(held.started.wait(), 2.0)
        await orch.interrupt("r1", reason="stopped")
        state = await asyncio.wait_for(post, 2.0)

        row = state.transcript[-1]
        assert row.sender_id == SCOUT.id and row.completed is False
        assert SCOUT.id not in state.last_seen_seq
        assert agent.history == before
        record = sessions.load(SCOUT.session_id)
        assert record is not None
        assert _unanswered(record.messages) == []
        assert record.messages == before


class TestTheAgentDropsAnUnansweredStep:
    """Outside a room nothing rolls a turn back, so the agent keeps the floor itself.

    **Dropped, not closed with invented results.** A tool may have run and changed
    something before the stop, so a `TOOL` message saying "cancelled" would be a result no
    tool produced, and possibly a false one (P6). The dropped message was never sent as
    input to any request, so removing it edits nothing a model was shown. The calls stay
    in the log as `TOOL_CALL` events, with a `TOOL_STEP_DROPPED` event beside them.
    """

    @pytest.mark.asyncio
    async def test_a_cancelled_turn_keeps_its_prompt_and_drops_the_unanswered_step(
        self,
    ) -> None:
        """Stop in a one-to-one chat: the question stays, the half step goes.

        Killed by: src/uclone_x/agent/base.py :: del turn_messages[dangling_index:]
        Becomes: pass
        Killed by: src/uclone_x/agent/base.py :: return (index, missing) if missing else None
        Becomes: return None
        """
        held = _HeldTool()
        agent = _seat(_Script([_call("c1", "held"), "unused"]), None, held)

        turn = asyncio.create_task(agent.execute_turn("write it down"))
        await asyncio.wait_for(held.started.wait(), 2.0)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

        assert _unanswered(agent.history) == []
        last = agent.history[-1]
        assert last.role is MessageRole.USER and last.content == "write it down"
        dropped = [e for e in agent.pending_durable_events if e["type"] == "TOOL_STEP_DROPPED"]
        assert len(dropped) == 1
        assert dropped[0]["unanswered_tool_call_ids"] == ["c1"]
        assert dropped[0]["outcome"] == "cancelled"

    @pytest.mark.asyncio
    async def test_a_step_whose_tools_all_answered_is_kept_when_the_turn_fails_later(
        self,
    ) -> None:
        """Only an unanswered step is dropped: a completed round is a real exchange.

        Killed by: src/uclone_x/agent/base.py :: return (index, missing) if missing else None
        Becomes: return (index, missing)
        """
        agent = _seat(
            _Script([_call("c1", "note"), LLMProviderError("provider unavailable (scripted)")]),
            None,
            _NoteTool(),
        )

        result = await agent.execute_turn("take a note")

        assert result.error is not None
        roles = [m.role for m in agent.history]
        assert roles[-2:] == [MessageRole.ASSISTANT, MessageRole.TOOL], roles
        assert not [e for e in agent.pending_durable_events if e["type"] == "TOOL_STEP_DROPPED"]


# --------------------------------------------------------------------------------------
# The room's own write fails after a turn that committed.
# --------------------------------------------------------------------------------------


class _LandingFails(RoomStore):
    """Refuses the write that lands the scout's reply, as a full disk or a lost CAS would."""

    def save(self, state: RoomState) -> RoomState:
        if state.transcript and state.transcript[-1].sender_id == SCOUT.id:
            raise OSError("disk full (scripted)")
        return super().save(state)


class TestARoomThatCannotSaveTheTurnUndoesTheSeat:
    @pytest.mark.asyncio
    async def test_the_seat_does_not_keep_a_reply_the_room_did_not_record(
        self, tmp_path: Path
    ) -> None:
        """The room's failure propagates as itself, and the seat forgets the turn too.

        The seat's session is written before the room's row, so a landing save that fails
        used to leave a seat that remembered answering while the transcript had no answer
        -- and it answered the next span from that turn.

        Killed by: src/uclone_x/room/orchestrator.py :: self._undo_committed_seat(agent, speaker, checkpoint)
        Becomes: pass
        """
        sessions = SessionStore(tmp_path / "sessions")
        agent = _seat(_Script(["scout says TTL"]), sessions)
        before = agent.history
        orch = _room(tmp_path, agent, _LandingFails(tmp_path / "rooms"))

        with pytest.raises(TurnNotLandedError) as raised:
            await orch.post("r1", "alice", "discuss the cache")
        assert isinstance(raised.value.__cause__, OSError)

        assert agent.history == before
        record = sessions.load(SCOUT.session_id)
        assert record is not None
        assert not any("scout says TTL" in str(m.content) for m in record.messages)


# --------------------------------------------------------------------------------------
# Follow-ups from the review of #1487 (#1495).
# --------------------------------------------------------------------------------------


def _toolless_seat(llm: _Script, sessions: SessionStore) -> BaseAgent:
    """A seat with no tool registry: nothing can answer a call its model makes."""
    return BaseAgent(
        config=AgentConfig(
            agent_id=SCOUT.id,
            name=SCOUT.id,
            llm_config=AgentLLMConfig(model_name="scripted"),
        ),
        llm=llm,
        tools=None,
        context=AgentContext(session_id=SCOUT.session_id, agent_id=SCOUT.id),
        store=sessions,
    )


def _answer_with_call(text: str, call: ToolCallRequest) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=text,
        tool_calls=(call,),
        usage=_USAGE,
        provenance=_PROV,
    )


def _tail(request: LLMRequest) -> str:
    """The text of a request's last message, where its turn context travels."""
    return str(request.messages[-1].content)


class TestACompletedTurnKeepsItsAnswer:
    @pytest.mark.asyncio
    async def test_a_seat_without_tools_remembers_the_answer_the_room_landed(
        self, tmp_path: Path
    ) -> None:
        """A completed turn loses only the calls nothing answered, not its text.

        With no tool registry, a reply that also asks for a tool ends the turn: nothing
        can run it, and the reply's text is the answer. #1487 dropped that whole message
        as an unanswered step, so the room's transcript kept the answer while the seat's
        own history ended on the question -- and its next turn was asked it again.

        Killed by: src/uclone_x/agent/base.py :: kept_text = outcome == "completed" and bool(asked.content)
        Becomes: kept_text = False
        """
        sessions = SessionStore(tmp_path / "sessions")
        agent = _toolless_seat(
            _Script([_answer_with_call("the cache TTL is 60s", _call("c1", "note"))]), sessions
        )
        orch = _room(tmp_path, agent)

        state = await orch.post("r1", "alice", "what is the TTL?")

        assert state.transcript[-1].content == "the cache TTL is 60s"
        last = agent.history[-1]
        assert last.role is MessageRole.ASSISTANT
        assert last.content == "the cache TTL is 60s"
        assert last.tool_calls == ()
        assert _unanswered(agent.history) == []
        dropped = [
            e for e in read_session_log(_only_log(sessions)) if e["type"] == "TOOL_STEP_DROPPED"
        ]
        assert [(e["dropped_message_count"], e["assistant_text_kept"]) for e in dropped] == [
            (0, True)
        ]


class _LandingFailsOnce(RoomStore):
    """Refuses the first write of the scout's reply with a real-looking disk error."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.failed = False

    def save(self, state: RoomState) -> RoomState:
        if not self.failed and state.transcript and state.transcript[-1].sender_id == SCOUT.id:
            self.failed = True
            raise OSError(28, "No space left on device", "/home/reader/.uclone/rooms/r1.json")
        return super().save(state)


class TestARetryIsToldWhatTheUndoneAttemptRan:
    """The rollback undoes the conversation, not the tools, and the retry is told so.

    The statement travels in the turn context, never as a `TOOL` message: a result the
    runtime wrote would be a substituted result (P6).
    """

    @pytest.mark.asyncio
    async def test_the_retry_names_the_call_the_failed_attempt_made(self, tmp_path: Path) -> None:
        """A retry after a rollback sees which tool already ran, without its output.

        Killed by: src/uclone_x/agent/base.py :: live.undone_tool_calls.extend(undone)
        Becomes: pass
        """
        sessions = SessionStore(tmp_path / "sessions")
        llm = _Script([_call("c1", "note"), LLMProviderError("provider unavailable"), "done"])
        agent = _seat(llm, sessions, _NoteTool())
        orch = _room(tmp_path, agent)

        await orch.post("r1", "alice", "write it down")
        await orch.retry("r1")

        retry = llm.requests[2]
        assert "[Undone Attempt]" in _tail(retry)
        assert 'note(text="x")' in _tail(retry)
        # Named, not replayed: no result in any form, and no message the tool did not write.
        assert "noted" not in _tail(retry)
        assert [m for m in retry.messages if m.role is MessageRole.TOOL] == []
        assert "[Undone Attempt]" not in str(agent.history)
        rolled_back = [
            e for e in read_session_log(_only_log(sessions)) if e["type"] == "TURN_ROLLED_BACK"
        ]
        assert [e["undone_tool_calls"] for e in rolled_back] == [
            [{"tool_call_id": "c1", "name": "note"}]
        ]

    @pytest.mark.asyncio
    async def test_the_statement_ends_once_a_turn_that_carried_it_is_kept(
        self, tmp_path: Path
    ) -> None:
        """The retry that stayed is in history now; later turns are not told again.

        Killed by: src/uclone_x/agent/base.py :: turn_live.undone_tool_calls.clear()
        Becomes: pass
        """
        llm = _Script(
            [_call("c1", "note"), LLMProviderError("provider unavailable"), "done", "next"]
        )
        agent = _seat(llm, SessionStore(tmp_path / "sessions"), _NoteTool())
        orch = _room(tmp_path, agent)

        await orch.post("r1", "alice", "write it down")
        await orch.retry("r1")
        await orch.post("r1", "alice", "and now?")

        assert "[Undone Attempt]" in _tail(llm.requests[2])
        assert "[Undone Attempt]" not in _tail(llm.requests[3])

    @pytest.mark.asyncio
    async def test_a_turn_the_room_could_not_save_is_named_to_the_next_one(
        self, tmp_path: Path
    ) -> None:
        """The landing-save rollback runs after the seat's events were drained.

        So the calls cannot be read back off the pending queue; the turn keeps them on
        the live session for the rollback to name.

        Killed by: src/uclone_x/agent/base.py :: turn_live.last_turn_tool_calls = all_tool_calls[
        Becomes: turn_live.last_turn_tool_calls = [] and all_tool_calls[
        """
        llm = _Script([_call("c1", "note"), "noted it", "again"])
        agent = _seat(llm, SessionStore(tmp_path / "sessions"), _NoteTool())
        orch = _room(tmp_path, agent, _LandingFailsOnce(tmp_path / "rooms"))

        with pytest.raises(TurnNotLandedError):
            await orch.post("r1", "alice", "write it down")
        await orch.post("r1", "alice", "did you?")

        assert 'note(text="x")' in _tail(llm.requests[2])


class TestUndoneCallsAreNotMatchedByProviderIds:
    """Call ids are not unique across steps or attempts (#1509 review B1).

    Ollama and Gemini number a response's calls `call_0`, `call_1`, and an id-less
    OpenAI-compatible server gives `""`, so two different calls that both ran can share
    an id. Each must still be stated to the retry.
    """

    @pytest.mark.asyncio
    async def test_two_undone_attempts_with_the_same_call_id_are_both_stated(
        self, tmp_path: Path
    ) -> None:
        """Attempt one runs `note` as `call_0`, attempt two runs `memo` as `call_0`.

        Killed by: src/uclone_x/agent/base.py :: live.undone_tool_calls.extend(undone)
        Becomes: live.undone_tool_calls.extend(call for call in undone if call.id not in {c.id for c in live.undone_tool_calls})
        """
        llm = _Script(
            [
                _call("call_0", "note"),
                LLMProviderError("provider unavailable"),
                _call("call_0", "memo"),
                LLMProviderError("provider unavailable"),
                "done",
            ]
        )
        agent = _seat(llm, SessionStore(tmp_path / "sessions"), _NoteTool(), _MemoTool())
        orch = _room(tmp_path, agent)

        await orch.post("r1", "alice", "write it down")
        await orch.retry("r1")
        await orch.retry("r1")

        third = _tail(llm.requests[4])
        assert 'note(text="x")' in third
        assert 'memo(text="x")' in third

    @pytest.mark.asyncio
    async def test_a_refused_step_rolled_back_keeps_the_earlier_step_with_its_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Step one runs `memo` as `call_0`; step two runs `note`, also as `call_0`, and is
        refused for the window and withheld. The room then rolls the turn back.

        The retry is told about both calls, and about the withheld one once: the
        rollback leaves out the calls the refusal already stated, by position.

        Killed by: src/uclone_x/agent/base.py :: : len(all_tool_calls) - withheld_calls
        Becomes: : len(all_tool_calls)
        """
        from uclone_x.core.tool_results import STEP_OVER_WINDOW_MESSAGE

        llm = _Script([_call("call_0", "memo"), _call("call_0", "note"), "done"])
        agent = _seat(llm, SessionStore(tmp_path / "sessions"), _NoteTool(), _MemoTool())
        steps = 0

        def refuse_the_second_step(*args: Any, **kwargs: Any) -> str | None:
            nonlocal steps
            steps += 1
            return STEP_OVER_WINDOW_MESSAGE if steps == 2 else None

        monkeypatch.setattr(agent, "_fit_step_to_window", refuse_the_second_step)
        orch = _room(tmp_path, agent)

        await orch.post("r1", "alice", "write it down")
        await orch.retry("r1")

        retry = _tail(llm.requests[2])
        assert retry.count('memo(text="x")') == 1
        assert retry.count('note(text="x")') == 1

    @pytest.mark.asyncio
    async def test_a_refused_step_is_stated_beside_an_earlier_attempts_call_with_its_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Attempt one runs `note` as `call_0` and is rolled back. Its retry runs `memo`,
        also as `call_0`, and that step is refused and withheld while `note` is still
        waiting to be stated. The next retry is told about both.

        Killed by: src/uclone_x/agent/base.py :: live.undone_tool_calls.extend(withheld)
        Becomes: live.undone_tool_calls.extend(call for call in withheld if call.id not in {c.id for c in live.undone_tool_calls})
        """
        from uclone_x.core.tool_results import STEP_OVER_WINDOW_MESSAGE

        llm = _Script(
            [
                _call("call_0", "note"),
                LLMProviderError("provider unavailable"),
                _call("call_0", "memo"),
                "done",
            ]
        )
        agent = _seat(llm, SessionStore(tmp_path / "sessions"), _NoteTool(), _MemoTool())
        steps = 0

        def refuse_the_second_step(*args: Any, **kwargs: Any) -> str | None:
            nonlocal steps
            steps += 1
            return STEP_OVER_WINDOW_MESSAGE if steps == 2 else None

        monkeypatch.setattr(agent, "_fit_step_to_window", refuse_the_second_step)
        orch = _room(tmp_path, agent)

        await orch.post("r1", "alice", "write it down")
        await orch.retry("r1")
        await orch.retry("r1")

        last = _tail(llm.requests[3])
        assert last.count('note(text="x")') == 1
        assert last.count('memo(text="x")') == 1


class TestAFailedLandingSaveIsReportedPlainly:
    @pytest.mark.asyncio
    async def test_the_reason_a_reader_sees_names_no_path_or_exception(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """What reaches the conversation is a sentence; the cause is in the log.

        The landing save's own `OSError` used to propagate as itself, so the CLI printed
        it and a head could only say "the runtime failed". It is now a `RoomError` in plain
        words, with the fault chained on it and logged.

        Killed by: src/uclone_x/room/orchestrator.py :: if isinstance(exc, RoomError):
        Becomes: if True:
        """
        agent = _seat(_Script(["scout says TTL"]), SessionStore(tmp_path / "sessions"))
        orch = _room(tmp_path, agent, _LandingFailsOnce(tmp_path / "rooms"))

        with caplog.at_level("ERROR"), pytest.raises(TurnNotLandedError) as raised:
            await orch.post("r1", "alice", "discuss the cache")

        text = reader_facing_reason(raised.value)
        assert text == str(raised.value)
        assert text.startswith("Scout's reply could not be saved")
        for internal in ("/", "OSError", "Errno", "No space left", "Traceback"):
            assert internal not in text, internal
        assert _http_error(raised.value).status_code == 503
        assert isinstance(raised.value.__cause__, OSError)
        assert "/home/reader/.uclone/rooms/r1.json" in caplog.text
