"""Tests for re-running a turn that failed, without spending a fresh turn.

Written before the implementation. What they pin:

* **A retry reuses the turn the failure already spent.** The budget bounds *conversation*
  — how many agent utterances one human message may produce — and a failed turn produced
  none. Charging it means a transient provider error silently shortens the exchange, and
  the only remedy the user had before this (retype the question) resets the budget to zero
  anyway, so charging was a penalty that the workaround skipped.
* **The failure stays in the record.** A retry appends; it never edits the transcript. A
  room that erases a failure once it is repaired cannot answer "did this room have to be
  asked twice", which is exactly what G8 records failures for.
* **A retry is not a re-selection.** The floor goes back to the agent that failed, under
  the decision that gave it the floor, because the chain would now see the failed row as
  that agent having already spoken and would route somewhere else.
"""

from __future__ import annotations

from typing import Any

import pytest

from uclone_x.errors import NothingToRetryError, UnknownRoomParticipantError
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomPolicy,
    RoomState,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerRequest,
)


class FakeAgent:
    """Minimal `BaseAgentProtocol` surface the orchestrator is allowed to touch."""

    def __init__(self, agent_id: str, reply: str = "ok") -> None:
        self.agent_id = agent_id
        self._reply = reply
        self.prompts: list[str] = []
        self.fail_with: Exception | None = None

    async def execute_turn(self, prompt: str, *, stream_callback: Any = None) -> Any:
        from uclone_x.agent.models import TurnResult

        self.prompts.append(prompt)
        if self.fail_with is not None:
            raise self.fail_with
        return TurnResult(turn_index=len(self.prompts), content=self._reply, provenance=None)

    def checkpoint_turn(self, session_id: str | None = None) -> Any:
        from uclone_x.agent.session import SessionState

        return SessionState(session_id=session_id or "unused", agent_id=self.agent_id)

    def roll_back_turn(self, checkpoint: Any, *, reason: str) -> int:
        return 0

    def persist_session(self, session_id: str | None = None) -> Any:
        from uclone_x.agent.session import SessionState

        return SessionState(session_id=session_id or "unused", agent_id=self.agent_id)


class FakeResolver:
    def __init__(self, agents: dict[str, FakeAgent]) -> None:
        self._agents = agents

    async def resolve(self, participant: Participant) -> Any:
        return self._agents[participant.id]


class ScriptedSelector:
    """Returns queued decisions, then abstains. Counts how often it was consulted."""

    def __init__(self, name: str, decisions: list[SpeakerDecision] | None = None) -> None:
        self._name = name
        self._queue = list(decisions or [])
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        self.calls += 1
        if self._queue:
            return self._queue.pop(0)
        return SpeakerDecision(verdict=SelectionVerdict.ABSTAIN, selector=self._name)


ALICE = Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice")
SCOUT = Participant(
    id="scout",
    kind=ParticipantKind.AGENT,
    display_name="Scout",
    session_id="sess_room__r1__scout",
    ontology_namespace="https://uclone-x.ai/ontology/r1/scout",
)


def speak(agent_id: str, selector: str = "scripted") -> SpeakerDecision:
    return SpeakerDecision(
        verdict=SelectionVerdict.SPEAK,
        speaker_id=agent_id,
        selector=selector,
        reasoning="addressed by name",
    )


@pytest.fixture
def built(tmp_path: Any) -> Any:
    """A one-agent room capped at one turn, plus its store and agent."""
    from uclone_x.room.store import RoomStore

    store = RoomStore(tmp_path)
    store.save(
        RoomState(
            room_id="r1",
            participants=(ALICE, SCOUT),
            policy=RoomPolicy(max_agent_turns_per_human_message=1),
        )
    )
    agents = {"scout": FakeAgent("scout", reply="TTL is simplest")}
    return store, FakeResolver(agents), agents


def _orchestrator(built: Any, selectors: list[Any]) -> Any:
    from uclone_x.room.orchestrator import RoomOrchestrator

    store, resolver, _ = built
    return RoomOrchestrator(store=store, selectors=selectors, resolver=resolver)


async def _fail_one_turn(built: Any, selector: ScriptedSelector) -> Any:
    """Post a message whose only turn fails, and return the orchestrator that ran it."""
    _, _, agents = built
    agents["scout"].fail_with = RuntimeError("model exploded")
    orch = _orchestrator(built, [selector])
    await orch.post("r1", "alice", "what about caching?")
    agents["scout"].fail_with = None
    return orch


class TestRetryRunsTheFailedTurnAgain:
    @pytest.mark.asyncio
    async def test_retry_gives_the_floor_back_to_the_agent_that_failed(self, built: Any) -> None:
        """The answer the failure owed is recorded, without the human retyping.

        Killed by: src/uclone_x/room/orchestrator.py :: return await self._take_turn(room_id, speaker, failed.decision or retried)
        Becomes: return state
        """
        store, _, _ = built
        orch = await _fail_one_turn(built, ScriptedSelector("s", [speak("scout")]))

        state = await orch.retry("r1")

        assert state.transcript[-1].sender_id == "scout"
        assert state.transcript[-1].content == "TTL is simplest"
        assert state.transcript[-1].error is None
        assert store.load("r1") is not None

    @pytest.mark.asyncio
    async def test_retry_reuses_the_turn_the_failure_spent(self, built: Any) -> None:
        """A failed turn produced no conversation, so the budget it charged is given back.

        The room is capped at one turn. Were the retry to spend a fresh one, the room
        would end at two agent turns for one human message — over its own ceiling, by the
        path a user reaches for precisely when the room has already gone wrong.

        Killed by: src/uclone_x/room/orchestrator.py :: max(0, state.turn_state.agent_turns_since_human - 1)
        Becomes: state.turn_state.agent_turns_since_human
        """
        store, _, _ = built
        orch = await _fail_one_turn(built, ScriptedSelector("s", [speak("scout")]))
        spent = store.load("r1")
        assert spent is not None and spent.turn_state.agent_turns_since_human == 1

        state = await orch.retry("r1")

        assert state.turn_state.agent_turns_since_human == 1, (
            "a retried turn reuses the budget slot the failure spent, never a fresh one"
        )

    @pytest.mark.asyncio
    async def test_retry_keeps_the_failed_row_in_the_transcript(self, built: Any) -> None:
        """Repairing a failure must not erase it: the record is what G8 exists for."""
        orch = await _fail_one_turn(built, ScriptedSelector("s", [speak("scout")]))

        state = await orch.retry("r1")

        failed = [m for m in state.transcript if m.error is not None]
        assert len(failed) == 1
        assert failed[0].seq < state.transcript[-1].seq

    @pytest.mark.asyncio
    async def test_retry_hands_the_speaker_the_span_the_failure_did_not_consume(
        self, built: Any
    ) -> None:
        """`last_seen_seq` never advanced, so the unseen span is still waiting."""
        _, _, agents = built
        orch = await _fail_one_turn(built, ScriptedSelector("s", [speak("scout")]))

        state = await orch.retry("r1")

        assert "what about caching?" in agents["scout"].prompts[-1]
        assert state.last_seen_seq.get("scout") is not None

    @pytest.mark.asyncio
    async def test_retry_does_not_consult_the_selector_chain(self, built: Any) -> None:
        """Re-selecting would route elsewhere: the failed row reads as that agent's turn.

        Killed by: src/uclone_x/room/orchestrator.py :: speaker = self._participant(state, failed.sender_id)
        Becomes: speaker = await self._decide(state) and None
        """
        selector = ScriptedSelector("s", [speak("scout")])
        orch = await _fail_one_turn(built, selector)
        consulted = selector.calls

        await orch.retry("r1")

        assert selector.calls == consulted

    @pytest.mark.asyncio
    async def test_retry_records_the_decision_that_caused_the_failed_turn(self, built: Any) -> None:
        """Same floor, same reason — "why did this one answer" survives the retry."""
        orch = await _fail_one_turn(built, ScriptedSelector("s", [speak("scout", "mention")]))

        state = await orch.retry("r1")

        decision = state.transcript[-1].decision
        assert decision is not None
        assert decision.selector == "mention"
        assert decision.reasoning == "addressed by name"

    @pytest.mark.asyncio
    async def test_a_retry_that_fails_again_is_recorded_as_a_second_failure(
        self, built: Any
    ) -> None:
        """And leaves the room retryable again rather than swallowing the second error."""
        _, _, agents = built
        orch = await _fail_one_turn(built, ScriptedSelector("s", [speak("scout")]))
        agents["scout"].fail_with = RuntimeError("still broken")

        state = await orch.retry("r1")

        assert state.transcript[-1].error is not None
        assert "still broken" in state.transcript[-1].error
        assert state.turn_state.agent_turns_since_human == 1


class TestRetryRefuses:
    @pytest.mark.asyncio
    async def test_retry_refuses_when_nothing_failed(self, built: Any) -> None:
        """A room whose last row is an answer has nothing to re-run.

        Killed by: src/uclone_x/room/orchestrator.py :: if failed is None or failed.error is None:
        Becomes: if failed is None:
        """
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        await orch.post("r1", "alice", "what about caching?")

        with pytest.raises(NothingToRetryError, match="scout"):
            await orch.retry("r1")

    @pytest.mark.asyncio
    async def test_retry_refuses_when_someone_has_spoken_since_the_failure(
        self, built: Any
    ) -> None:
        """Only the last *utterance* is retryable; later speech has moved the room on.

        A join or a leave written after the failure does not: the room narrating its own
        roster is not somebody taking the conversation somewhere else, which is why the
        search is over `is_utterance` rather than over the last row.
        """
        orch = await _fail_one_turn(built, ScriptedSelector("s", [speak("scout")]))
        store, _, _ = built
        state = store.load("r1")
        assert state is not None
        store.save(orch.append_human(state, "alice", "never mind"))

        with pytest.raises(NothingToRetryError):
            await orch.retry("r1")

    @pytest.mark.asyncio
    async def test_retry_refuses_when_the_failed_speaker_has_left(self, built: Any) -> None:
        """The floor cannot be given to nobody — and the leave row must not hide the failure.

        `remove_participant` appends a LEAVE row, so the transcript's *last row* is no
        longer the failed turn. Searching rows rather than utterances would report "nothing
        to retry" for a room whose failure is still sitting there unanswered, which is a
        different and wronger thing to tell the user than "the agent has left".

        Killed by: src/uclone_x/room/models.py :: if m.is_utterance), None
        Becomes: ), None
        """
        from uclone_x.room.service import RoomService

        store, _, _ = built
        orch = await _fail_one_turn(built, ScriptedSelector("s", [speak("scout")]))
        RoomService(store).remove_participant("r1", "scout")

        with pytest.raises(UnknownRoomParticipantError, match="scout"):
            await orch.retry("r1")
