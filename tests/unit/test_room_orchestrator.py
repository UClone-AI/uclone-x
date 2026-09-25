"""Tests for the room store, the default-responder selector, and the orchestrator.

Written before the implementations they describe. What they pin, in order of how much
they would cost to get wrong:

* **Per-agent isolation** — two participants never share a session or an ontology. The
  requirement the whole design exists to satisfy, and the one a plausible-looking
  implementation can violate invisibly.
* **The transcript holds final utterances only** — an agent's tool traffic stays in its
  own session.
* **Who decides, and when** — the chain advises, the orchestrator decides; the turn cap is
  checked before anyone is asked; an exhausted chain becomes *recorded* silence.
* **Interjection** — a human message arriving mid-loop abandons the remaining turns, and
  does so without cancelling the turn already running.
* **Failure is visible** — a failed turn is recorded and does not consume the unseen span.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, cast

import pytest

from uclone_x.errors import (
    ParticipantNotResolvableError,
    SpeakerSelectionError,
    StaleRoomWriteError,
)
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomPolicy,
    RoomState,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerRequest,
)

# --------------------------------------------------------------------------------------
# Fakes. Deliberately not mocks: these assert on real recorded state, so a test that passes
# says the orchestrator did the thing, not that it called a method.
# --------------------------------------------------------------------------------------


def unwrap(payload: Any) -> Any:
    """A payload as plain data, so a text search over it cannot miss a nested mapping."""
    return {str(k): v for k, v in dict(payload).items()}


class FakeAgent:
    """Minimal `BaseAgentProtocol` surface the orchestrator is allowed to touch."""

    def __init__(self, agent_id: str, session_id: str, reply: str = "ok") -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self._reply = reply
        self.prompts: list[str] = []
        self.fail_with: Exception | None = None
        self.delay: float = 0.0
        #: Set the instant a turn begins, and awaited by an interjecting task. A `sleep`
        #: long enough to "probably" land mid-turn is a guess about scheduling that fails
        #: differently on a loaded machine; this is the fact itself.
        self.turn_started = asyncio.Event()
        #: Held closed while an interjection is arranged, so the turn cannot finish first.
        self.release: asyncio.Event | None = None
        #: Token deltas this agent emits when it is given a `stream_callback`.
        self.chunks: list[str] = []
        #: One entry per turn: the callback the orchestrator passed, or `None`.
        self.stream_callbacks: list[Any] = []
        #: A failure *returned* on the result, after streaming, as `BaseAgent.execute_turn`
        #: reports one -- as opposed to `fail_with`, which raises before any chunk.
        self.result_error: str | None = None
        #: The session id of every write the orchestrator asked for, in order.
        self.persisted: list[str | None] = []
        #: The tool executions this agent's next turns report on their `TurnResult`, as
        #: `BaseAgent.execute_turn` reports the tools its loop ran (#1353, #1354).
        self.tool_executions: tuple[Any, ...] = ()
        #: False as `BaseAgent.execute_turn` reports a turn that failed mid-step (#1366).
        self.tool_executions_complete: bool = True
        #: Every checkpoint the orchestrator handed back to undo a turn (#1423), in order.
        self.rolled_back: list[Any] = []
        self.caller_turn_ids: list[str | None] = []

    async def execute_turn(
        self,
        prompt: str,
        *,
        stream_callback: Any = None,
        caller_turn_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from uclone_x.agent.models import TurnResult

        self.prompts.append(prompt)
        self.caller_turn_ids.append(caller_turn_id)
        #: What the orchestrator handed this turn. `None` is a real answer and not a
        #: missing value -- it is how a room with no bus says "do not pay for chunks".
        self.stream_callbacks.append(stream_callback)
        self.turn_started.set()
        if self.release is not None:
            await self.release.wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_with is not None:
            raise self.fail_with
        if stream_callback is not None:
            for chunk in self.chunks:
                res = stream_callback("token", {"content": chunk})
                if asyncio.iscoroutine(res):
                    await res
            # Not forwarded: an agent's private reasoning is not the room's transcript
            # (G1). It carries `content` on purpose -- a payload the room's own forwarder
            # would happily publish if it stopped discriminating by event name, which is
            # what makes the discrimination testable at all.
            res = stream_callback(
                "status",
                {"status": "thinking", "detail": "private", "content": "PRIVATE-SCRATCHPAD"},
            )
            if asyncio.iscoroutine(res):
                await res
        if self.result_error is not None:
            return TurnResult(
                turn_index=len(self.prompts),
                content="",
                error=self.result_error,
                provenance=None,
                tool_executions=self.tool_executions,
                tool_executions_complete=self.tool_executions_complete,
            )
        return TurnResult(
            turn_index=len(self.prompts),
            content=self._reply,
            provenance=None,
            tool_executions=self.tool_executions,
        )

    def checkpoint_turn(self, session_id: str | None = None) -> Any:
        """The seat before a turn; the orchestrator takes one before every turn (#1423)."""
        from uclone_x.agent.session import SessionState

        return SessionState(session_id=session_id or self.session_id, agent_id=self.agent_id)

    def roll_back_turn(self, checkpoint: Any, *, reason: str) -> int:
        """Recorded: the orchestrator undoes a turn that did not commit (#1423)."""
        self.rolled_back.append(checkpoint)
        return 0

    def persist_session(self, session_id: str | None = None) -> Any:
        """Accepted and recorded: the orchestrator writes a seat's session after every turn."""
        from uclone_x.agent.session import SessionState

        self.persisted.append(session_id)
        return SessionState(session_id=session_id or self.session_id, agent_id=self.agent_id)


class FakeResolver:
    """Hands out one agent per participant, and records what it was asked for."""

    def __init__(self, agents: dict[str, FakeAgent]) -> None:
        self._agents = agents
        self.resolved: list[tuple[str, str, str]] = []

    async def resolve(self, participant: Participant) -> Any:
        agent = self._agents.get(participant.id)
        if agent is None:
            raise ParticipantNotResolvableError(f"no agent for {participant.id!r}")
        self.resolved.append(
            (participant.id, participant.session_id, participant.ontology_namespace)
        )
        return agent


class ScriptedSelector:
    """Returns queued decisions, then abstains. Counts how often it was consulted."""

    def __init__(self, name: str, decisions: list[SpeakerDecision] | None = None) -> None:
        self._name = name
        self._queue = list(decisions or [])
        self.calls = 0
        self.raise_with: Exception | None = None

    @property
    def name(self) -> str:
        return self._name

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        self.calls += 1
        if self.raise_with is not None:
            raise self.raise_with
        if self._queue:
            return self._queue.pop(0)
        return SpeakerDecision(verdict=SelectionVerdict.ABSTAIN, selector=self._name)


def speak(agent_id: str, selector: str = "scripted", confidence: float = 1.0) -> SpeakerDecision:
    return SpeakerDecision(
        verdict=SelectionVerdict.SPEAK,
        speaker_id=agent_id,
        selector=selector,
        confidence=confidence,
    )


def silence(selector: str = "scripted") -> SpeakerDecision:
    return SpeakerDecision(verdict=SelectionVerdict.SILENCE, selector=selector)


ALICE = Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice")


def agent_participant(agent_id: str, room_id: str = "r1") -> Participant:
    return Participant(
        id=agent_id,
        kind=ParticipantKind.AGENT,
        display_name=agent_id.title(),
        persona_summary=f"{agent_id} does {agent_id} things",
        session_id=f"sess_room__{room_id}__{agent_id}",
        ontology_namespace=f"https://uclone-x.ai/ontology/{room_id}/{agent_id}",
    )


SCOUT = agent_participant("scout")
CRITIC = agent_participant("critic")


# --------------------------------------------------------------------------------------
# RoomStore
# --------------------------------------------------------------------------------------


class TestRoomStore:
    def test_load_returns_none_for_an_unknown_room(self, tmp_path: Any) -> None:
        from uclone_x.room.store import RoomStore

        assert RoomStore(tmp_path).load("nope") is None

    def test_save_then_load_roundtrips(self, tmp_path: Any) -> None:
        from uclone_x.room.store import RoomStore

        store = RoomStore(tmp_path)
        saved = store.save(RoomState(room_id="r1", participants=(ALICE, SCOUT)))
        loaded = store.load("r1")
        assert loaded is not None
        assert loaded.participants[1].session_id == "sess_room__r1__scout"
        assert loaded.revision == saved.revision

    def test_save_stamps_the_next_revision(self, tmp_path: Any) -> None:
        from uclone_x.room.store import RoomStore

        store = RoomStore(tmp_path)
        first = store.save(RoomState(room_id="r1"))
        second = store.save(first)
        assert (first.revision, second.revision) == (1, 2)

    def test_a_stale_write_is_refused(self, tmp_path: Any) -> None:
        """Two orchestrators on one room is a design violation, so it must not merge."""
        from uclone_x.room.store import RoomStore

        store = RoomStore(tmp_path)
        first = store.save(RoomState(room_id="r1"))
        store.save(first)  # someone else advances the record
        with pytest.raises(StaleRoomWriteError):
            store.save(first)  # our stale handle

    def test_delete_reports_whether_there_was_one(self, tmp_path: Any) -> None:
        from uclone_x.room.store import RoomStore

        store = RoomStore(tmp_path)
        store.save(RoomState(room_id="r1"))
        assert store.delete("r1") is True
        assert store.delete("r1") is False

    def test_list_room_ids(self, tmp_path: Any) -> None:
        from uclone_x.room.store import RoomStore

        store = RoomStore(tmp_path)
        store.save(RoomState(room_id="r1"))
        store.save(RoomState(room_id="r2"))
        assert store.list_room_ids() == ("r1", "r2")

    def test_a_room_id_that_escapes_the_directory_is_refused(self, tmp_path: Any) -> None:
        from uclone_x.errors import PathTraversalError
        from uclone_x.room.store import RoomStore

        with pytest.raises(PathTraversalError):
            RoomStore(tmp_path).load("../escape")


# --------------------------------------------------------------------------------------
# DefaultResponderSelector
# --------------------------------------------------------------------------------------


def _request(*participants: Participant, policy: RoomPolicy | None = None) -> SpeakerRequest:
    from uclone_x.room.models import RoomMessage, TurnState

    return SpeakerRequest(
        room_id="r1",
        participants=participants,
        transcript=(RoomMessage(seq=1, sender_id="alice", content="hello"),),
        turn_state=TurnState(),
        policy=policy or RoomPolicy(),
    )


class TestDefaultResponderSelector:
    @pytest.mark.asyncio
    async def test_routes_to_the_configured_responder(self) -> None:
        from uclone_x.room.selectors import DefaultResponderSelector

        policy = RoomPolicy(default_responder_id="critic")
        decision = await DefaultResponderSelector().select(
            _request(ALICE, SCOUT, CRITIC, policy=policy)
        )
        assert decision.verdict is SelectionVerdict.SPEAK
        assert decision.speaker_id == "critic"

    @pytest.mark.asyncio
    async def test_abstains_when_no_responder_is_configured(self) -> None:
        """No implicit default: answering in an unchosen voice is the defect this replaces."""
        from uclone_x.room.selectors import DefaultResponderSelector

        decision = await DefaultResponderSelector().select(_request(ALICE, SCOUT, CRITIC))
        assert decision.verdict is SelectionVerdict.ABSTAIN

    @pytest.mark.asyncio
    async def test_a_responder_who_left_the_room_raises(self) -> None:
        from uclone_x.room.selectors import DefaultResponderSelector

        policy = RoomPolicy(default_responder_id="ghost")
        with pytest.raises(SpeakerSelectionError, match="ghost"):
            await DefaultResponderSelector().select(_request(ALICE, SCOUT, policy=policy))

    @pytest.mark.asyncio
    async def test_a_human_responder_raises(self) -> None:
        from uclone_x.room.selectors import DefaultResponderSelector

        policy = RoomPolicy(default_responder_id="alice")
        with pytest.raises(SpeakerSelectionError):
            await DefaultResponderSelector().select(_request(ALICE, SCOUT, policy=policy))


# --------------------------------------------------------------------------------------
# RoomOrchestrator
# --------------------------------------------------------------------------------------


@pytest.fixture
def built(tmp_path: Any) -> Any:
    """A two-agent room, its store, resolver and agents, with no selectors wired yet."""
    from uclone_x.room.store import RoomStore

    store = RoomStore(tmp_path)
    store.save(RoomState(room_id="r1", participants=(ALICE, SCOUT, CRITIC), policy=RoomPolicy()))
    agents = {
        "scout": FakeAgent("scout", SCOUT.session_id, reply="scout says TTL"),
        "critic": FakeAgent("critic", CRITIC.session_id, reply="critic disagrees"),
    }
    return store, FakeResolver(agents), agents


def _cap(built: Any, turns: int) -> None:
    """Set the room's per-message turn ceiling."""
    store, _, _ = built
    state = store.load("r1")
    assert state is not None
    store.save(
        state.model_copy(update={"policy": RoomPolicy(max_agent_turns_per_human_message=turns)})
    )


def _orchestrator(built: Any, selectors: list[Any]) -> Any:
    from uclone_x.room.orchestrator import RoomOrchestrator

    store, resolver, _ = built
    return RoomOrchestrator(store=store, selectors=selectors, resolver=resolver)


class TestOrchestratorDecides:
    @pytest.mark.asyncio
    async def test_the_first_non_abstaining_selector_wins(self, built: Any) -> None:
        _cap(built, 1)  # isolate a single decision; continuation has its own test
        abstainer = ScriptedSelector("first")
        decider = ScriptedSelector("second", [speak("critic")])
        never = ScriptedSelector("third", [speak("scout")])
        orch = _orchestrator(built, [abstainer, decider, never])

        state = await orch.post("r1", "alice", "who is there?")

        assert state.transcript[-1].sender_id == "critic"
        assert never.calls == 0, "a selector after the decision must not be consulted"

    @pytest.mark.asyncio
    async def test_the_loop_re_decides_after_every_turn(self, built: Any) -> None:
        """One human message can yield several turns, each separately decided."""
        selector = ScriptedSelector("s", [speak("scout"), speak("critic")])
        orch = _orchestrator(built, [selector])

        state = await orch.post("r1", "alice", "discuss")

        assert [m.sender_id for m in state.transcript] == ["alice", "scout", "critic"]
        assert selector.calls == 3, "consulted once per turn, plus the round that ends it"

    @pytest.mark.asyncio
    async def test_an_exhausted_chain_records_silence(self, built: Any) -> None:
        """Recorded, not implicit: a quiet room must be distinguishable from a broken one."""
        orch = _orchestrator(built, [ScriptedSelector("a"), ScriptedSelector("b")])

        state = await orch.post("r1", "alice", "...")

        assert state.transcript[-1].sender_id == "alice"
        assert state.last_decision is not None
        assert state.last_decision.verdict is SelectionVerdict.SILENCE
        assert "exhaust" in state.last_decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_a_decided_silence_stops_the_loop(self, built: Any) -> None:
        selector = ScriptedSelector("s", [silence()])
        orch = _orchestrator(built, [selector])

        state = await orch.post("r1", "alice", "thanks, bye")

        assert [m.sender_id for m in state.transcript] == ["alice"]
        assert selector.calls == 1

    @pytest.mark.asyncio
    async def test_a_speaker_who_is_not_in_the_room_is_refused(self, built: Any) -> None:
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("ghost")])])

        with pytest.raises(SpeakerSelectionError, match="ghost"):
            await orch.post("r1", "alice", "hi")

    @pytest.mark.asyncio
    async def test_a_human_named_as_speaker_is_refused(self, built: Any) -> None:
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("alice")])])

        with pytest.raises(SpeakerSelectionError):
            await orch.post("r1", "alice", "hi")

    @pytest.mark.asyncio
    async def test_a_broken_selector_propagates_and_is_not_silence(self, built: Any) -> None:
        selector = ScriptedSelector("s")
        selector.raise_with = SpeakerSelectionError("provider down")
        orch = _orchestrator(built, [selector])

        with pytest.raises(SpeakerSelectionError, match="provider down"):
            await orch.post("r1", "alice", "hi")

    @pytest.mark.asyncio
    async def test_the_decision_is_recorded_on_the_utterance_it_caused(self, built: Any) -> None:
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout", selector="mention")])])

        state = await orch.post("r1", "alice", "@scout hi")

        assert state.transcript[-1].decision is not None
        assert state.transcript[-1].decision.selector == "mention"

    @pytest.mark.asyncio
    async def test_each_seat_turn_is_called_with_the_id_its_row_carries(self, built: Any) -> None:
        """The seat's `TURN_START` link to the room row (turn inspection design §4.2.1).

        Each agent turn is run with `caller_turn_id` equal to the `turn_id` its transcript
        row records, and two turns get two ids -- the join the trace reader makes.

        Killed by: src/uclone_x/room/orchestrator.py :: caller_turn_id=turn_id,
        Becomes: caller_turn_id=None,
        """
        _, _, agents = built
        selector = ScriptedSelector("s", [speak("scout"), speak("critic")])
        orch = _orchestrator(built, [selector])

        state = await orch.post("r1", "alice", "discuss")

        scout_row, critic_row = state.transcript[1], state.transcript[2]
        assert scout_row.turn_id is not None and critic_row.turn_id is not None
        assert scout_row.turn_id != critic_row.turn_id
        assert agents["scout"].caller_turn_ids == [scout_row.turn_id]
        assert agents["critic"].caller_turn_ids == [critic_row.turn_id]


class TestOrchestratorBoundsTurns:
    @pytest.mark.asyncio
    async def test_the_cap_stops_an_agent_to_agent_run(self, built: Any) -> None:
        _cap(built, 2)
        selector = ScriptedSelector("s", [speak("scout"), speak("critic"), speak("scout")])
        orch = _orchestrator(built, [selector])

        result = await orch.post("r1", "alice", "discuss")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout", "critic"]
        assert selector.calls == 2, "the cap is checked before a selector is consulted"

    @pytest.mark.asyncio
    async def test_a_zero_cap_asks_nobody(self, built: Any) -> None:
        _cap(built, 0)
        selector = ScriptedSelector("s", [speak("scout")])
        orch = _orchestrator(built, [selector])

        await orch.post("r1", "alice", "quiet please")

        assert selector.calls == 0

    @pytest.mark.asyncio
    async def test_a_human_message_resets_the_turn_budget(self, built: Any) -> None:
        _cap(built, 1)
        selector = ScriptedSelector("s", [speak("scout"), speak("critic")])
        orch = _orchestrator(built, [selector])

        first = await orch.post("r1", "alice", "first")
        assert first.turn_state.agent_turns_since_human == 1
        state = await orch.post("r1", "alice", "second")

        assert state.turn_state.agent_turns_since_human == 1, (
            "a human message resets the budget rather than accumulating across messages"
        )


class TestOrchestratorKeepsAgentsApart:
    @pytest.mark.asyncio
    async def test_each_speaker_is_resolved_with_its_own_session_and_ontology(
        self, built: Any
    ) -> None:
        """G3 and G4, the orchestrator's half of them.

        What this proves is bounded, and the bound is worth stating: the orchestrator
        resolves each speaker by *its own* participant record, so it can never hand one
        agent another's session or namespace. Whether the resolver then honours those ids
        is the resolver's own obligation (`RoomAgentResolverProtocol`), and it has no
        implementation yet — §7.3.
        """
        _, resolver, _ = built
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])

        await orch.post("r1", "alice", "discuss")

        sessions = {sid for _, sid, _ in resolver.resolved}
        namespaces = {ns for _, _, ns in resolver.resolved}
        assert sessions == {"sess_room__r1__scout", "sess_room__r1__critic"}
        assert len(namespaces) == 2

    @pytest.mark.asyncio
    async def test_a_speaker_sees_only_the_span_it_has_not_seen(self, built: Any) -> None:
        _, _, agents = built
        _cap(built, 1)
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("scout")])])

        await orch.post("r1", "alice", "first question")
        await orch.post("r1", "alice", "second question")

        assert "first question" in agents["scout"].prompts[0]
        assert "first question" not in agents["scout"].prompts[1], (
            "already-seen messages live in the agent's own session and must not be resent"
        )
        assert "second question" in agents["scout"].prompts[1]

    @pytest.mark.asyncio
    async def test_a_speaker_sees_what_the_other_agent_said(self, built: Any) -> None:
        _, _, agents = built
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])

        await orch.post("r1", "alice", "discuss")

        assert "scout says TTL" in agents["critic"].prompts[0]
        assert "[scout]" in agents["critic"].prompts[0], "speakers must be named"


class TestOrchestratorRecordsFailure:
    @pytest.mark.asyncio
    async def test_a_failed_turn_is_recorded_rather_than_skipped(self, built: Any) -> None:
        _, _, agents = built
        agents["scout"].fail_with = RuntimeError("model exploded")
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])

        state = await orch.post("r1", "alice", "hi")

        failed = state.transcript[-1]
        assert failed.sender_id == "scout"
        assert failed.error is not None and "model exploded" in failed.error
        assert failed.content == ""

    @pytest.mark.asyncio
    async def test_a_failed_turn_does_not_consume_the_unseen_span(self, built: Any) -> None:
        _, _, agents = built
        _cap(built, 1)
        agents["scout"].fail_with = RuntimeError("transient")
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("scout")])])
        state = await orch.post("r1", "alice", "the question")
        assert state.last_seen_seq.get("scout") is None

        agents["scout"].fail_with = None
        await orch.post("r1", "alice", "again")

        assert "the question" in agents["scout"].prompts[1], (
            "a retried speaker must see the span its failed turn did not consume"
        )

    @pytest.mark.asyncio
    async def test_a_failed_turn_still_spends_the_budget(self, built: Any) -> None:
        """Otherwise a persistently failing agent loops until the cap never advances."""
        _, _, agents = built
        agents["scout"].fail_with = RuntimeError("always")
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")] * 5)])

        state = await orch.post("r1", "alice", "hi")

        assert state.turn_state.agent_turns_since_human == 3

    @pytest.mark.asyncio
    async def test_an_unresolvable_participant_propagates(self, built: Any) -> None:
        _, _, agents = built
        del agents["critic"]
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("critic")])])

        with pytest.raises(ParticipantNotResolvableError):
            await orch.post("r1", "alice", "hi")


class TestOrchestratorYieldsToHumans:
    @pytest.mark.asyncio
    async def test_a_human_message_mid_loop_abandons_the_remaining_turns(self, built: Any) -> None:
        """Window W3: the human typed while an agent was generating."""
        store, _, agents = built
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])
        agents["scout"].release = asyncio.Event()

        async def interject() -> None:
            await agents["scout"].turn_started.wait()
            state = store.load("r1")
            assert state is not None
            store.save(orch.append_human(state, "alice", "wait, actually"))
            agents["scout"].release.set()

        await asyncio.gather(orch.post("r1", "alice", "discuss"), interject())

        final = store.load("r1")
        assert final is not None
        senders = [m.sender_id for m in final.transcript]
        assert "critic" not in senders, "the remaining turns must be abandoned"

    @pytest.mark.asyncio
    async def test_the_running_turn_is_not_cancelled(self, built: Any) -> None:
        """Only the remaining turns are abandoned; work already spent is kept."""
        store, _, agents = built
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])
        agents["scout"].release = asyncio.Event()

        async def interject() -> None:
            await agents["scout"].turn_started.wait()
            state = store.load("r1")
            assert state is not None
            store.save(orch.append_human(state, "alice", "wait"))
            agents["scout"].release.set()

        await asyncio.gather(orch.post("r1", "alice", "discuss"), interject())

        final = store.load("r1")
        assert final is not None
        # The ORDER is the discriminating assertion, not scout's mere presence: if the
        # interjection never landed mid-turn, scout's message would be there anyway and
        # this test would pass while proving nothing. The human message sitting *before*
        # scout's is what says the turn was already running and was allowed to finish.
        assert [m.sender_id for m in final.transcript] == ["alice", "alice", "scout"]
        assert final.transcript[-1].content, "the running turn's work was discarded"

    @pytest.mark.asyncio
    async def test_a_new_human_message_clears_the_decision_that_ended_the_last_one(
        self, built: Any
    ) -> None:
        """The previous message's silence does not describe the message that followed it.

        The head says "no one answered your last message" from `last_decision` (#920), so a
        silence that outlived the message it was decided about would be reported under a
        message nobody has judged yet. Nothing pinned this: keeping the old decision passed
        every unit test and the room E2E file (#929), because the tests that read
        `last_decision` after an append all started from a room that had none.

        Killed by: src/uclone_x/room/orchestrator.py :: "last_decision": None,
        Becomes: "last_decision": state.last_decision,
        """
        store, _, _ = built
        orch = _orchestrator(built, [ScriptedSelector("s", [silence()])])
        ended = await orch.post("r1", "alice", "anyone?")
        assert ended.last_decision is not None
        assert ended.last_decision.verdict is SelectionVerdict.SILENCE

        accepted = await orch.accept("r1", "alice", "hello?")

        stored = store.load("r1")
        assert stored is not None
        assert accepted.last_decision is None
        assert stored.last_decision is None, "the silence was about the previous message"

    @pytest.mark.asyncio
    async def test_typing_alone_does_not_abandon_the_loop(self, built: Any) -> None:
        """A hint can be abandoned, so it must not silence the room on its own."""
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])

        await orch.note_human_activity("r1")
        state = await orch.post("r1", "alice", "discuss")

        assert [m.sender_id for m in state.transcript] == ["alice", "scout", "critic"]

    @pytest.mark.asyncio
    async def test_typing_advances_the_activity_watermark(self, built: Any) -> None:
        store, _, _ = built
        orch = _orchestrator(built, [ScriptedSelector("s")])
        before = store.load("r1")
        assert before is not None and before.turn_state.last_activity_ts == 0.0

        await orch.note_human_activity("r1")

        after = store.load("r1")
        assert after is not None and after.turn_state.last_activity_ts > 0.0


class TestOrchestratorTranscript:
    @pytest.mark.asyncio
    async def test_sequence_numbers_are_contiguous(self, built: Any) -> None:
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])

        state = await orch.post("r1", "alice", "discuss")

        assert [m.seq for m in state.transcript] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_an_unknown_room_is_refused(self, built: Any) -> None:
        from uclone_x.errors import RoomNotFoundError

        orch = _orchestrator(built, [ScriptedSelector("s")])
        with pytest.raises(RoomNotFoundError):
            await orch.post("no-such-room", "alice", "hi")

    @pytest.mark.asyncio
    async def test_a_sender_who_is_not_in_the_room_is_refused(self, built: Any) -> None:
        from uclone_x.errors import UnknownRoomParticipantError

        orch = _orchestrator(built, [ScriptedSelector("s")])
        with pytest.raises(UnknownRoomParticipantError):
            await orch.post("r1", "mallory", "hi")


# --------------------------------------------------------------------------------------
# Regressions from the PR #690 review. Each of these passed the original suite because
# `ScriptedSelector` pops a queue and then abstains — it is stateful in exactly the way the
# real selectors are not, so no test ever ran a real selector twice.
# --------------------------------------------------------------------------------------


class TestRealSelectorsDoNotRefire:
    """A rule answers the human's message once, not once per remaining turn."""

    @pytest.mark.asyncio
    async def test_a_one_to_one_room_takes_exactly_one_turn(self, built: Any) -> None:
        from uclone_x.room.selectors import SoleAgentSelector

        store, _, agents = built
        state = store.load("r1")
        assert state is not None
        store.save(state.model_copy(update={"participants": (ALICE, SCOUT)}))
        orch = _orchestrator(built, [SoleAgentSelector()])

        result = await orch.post("r1", "alice", "hello")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout"], (
            "the sole agent answered, and then answered its own answer twice more"
        )
        assert agents["scout"].prompts == ["[alice]: hello"]

    @pytest.mark.asyncio
    async def test_a_mention_is_not_re_read_on_every_turn(self, built: Any) -> None:
        from uclone_x.room.selectors import MentionSelector

        _, _, agents = built
        orch = _orchestrator(built, [MentionSelector()])

        result = await orch.post("r1", "alice", "@scout have a look")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout"]
        assert "" not in agents["scout"].prompts, "an agent was handed an empty prompt"

    @pytest.mark.asyncio
    async def test_the_default_responder_answers_once(self, built: Any) -> None:
        from uclone_x.room.selectors import DefaultResponderSelector

        store, _, _ = built
        state = store.load("r1")
        assert state is not None
        store.save(state.model_copy(update={"policy": RoomPolicy(default_responder_id="critic")}))
        orch = _orchestrator(built, [DefaultResponderSelector()])

        result = await orch.post("r1", "alice", "anyone?")

        assert [m.sender_id for m in result.transcript] == ["alice", "critic"]

    @pytest.mark.asyncio
    async def test_a_rule_still_answers_a_second_human_message(self, built: Any) -> None:
        """Abstaining after a turn must not make the selector permanently silent."""
        from uclone_x.room.selectors import MentionSelector

        orch = _orchestrator(built, [MentionSelector()])

        await orch.post("r1", "alice", "@scout first")
        result = await orch.post("r1", "alice", "@critic second")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout", "alice", "critic"]


class TestInterjectionIsNotSkipped:
    @pytest.mark.asyncio
    async def test_a_message_appended_during_a_turn_is_still_delivered(self, built: Any) -> None:
        """The high-water mark must cover the span actually shown, not the latest seq."""
        store, _, agents = built
        _cap(built, 1)
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("scout")])])
        agents["scout"].release = asyncio.Event()

        async def interject() -> None:
            await agents["scout"].turn_started.wait()
            state = store.load("r1")
            assert state is not None
            store.save(orch.append_human(state, "alice", "WAIT - do X instead"))
            agents["scout"].release.set()

        await asyncio.gather(orch.post("r1", "alice", "first question"), interject())
        await orch.post("r1", "alice", "carry on")

        assert any("WAIT - do X instead" in p for p in agents["scout"].prompts), (
            "a message appended while the speaker was working was jumped over forever"
        )


class TestPostRejectsNonHumans:
    @pytest.mark.asyncio
    async def test_an_agent_cannot_post_and_reset_the_turn_budget(self, built: Any) -> None:
        """G6's ceiling is the only structural bound; an agent must not be able to lift it."""
        from uclone_x.errors import UnknownRoomParticipantError

        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")] * 9)])
        await orch.post("r1", "alice", "go")

        with pytest.raises(UnknownRoomParticipantError, match="human"):
            await orch.post("r1", "scout", "and another thing")


class TestStoreContainment:
    def test_a_record_symlinked_out_of_the_directory_is_refused(self, tmp_path: Any) -> None:
        from uclone_x.errors import PathTraversalError
        from uclone_x.room.store import RoomStore

        rooms = tmp_path / "rooms"
        rooms.mkdir()
        outside = tmp_path / "outside.json"
        outside.write_text('{"room_id": "evil"}', encoding="utf-8")
        (rooms / "evil.json").symlink_to(outside)

        with pytest.raises(PathTraversalError) as excinfo:
            RoomStore(rooms).load("evil")

        # Still a containment refusal for anything treating that class as a boundary, and
        # now worded for the caller that asked about a room rather than a session.
        assert "room id" in str(excinfo.value).lower()

    def test_save_advances_updated_at(self, tmp_path: Any) -> None:
        from uclone_x.room.store import RoomStore

        store = RoomStore(tmp_path)
        first = store.save(RoomState(room_id="r1"))
        second = store.save(first)
        assert second.updated_at > first.updated_at


# --------------------------------------------------------------------------------------
# Coverage mostly over gaps no review had named. PR #690's one comment names a single gap —
# that no test composed a real selector with the real orchestrator — and records it closed
# inside #690; `TestTheWholeChainDrivesARealDebate` strengthens that composition rather than
# opening it. The other classes came out of reading the module against its own claims in the
# orchestration design document (§7.2a), and each closes a gap where the behaviour was
# implemented but nothing held it in place — the state in which a refactor silently changes
# it and the suite still reports green.
# --------------------------------------------------------------------------------------


class TestTranscriptWindow:
    """`policy.transcript_window` bounds what a selector sees, and nothing pinned it."""

    @pytest.mark.asyncio
    async def test_a_selector_sees_only_the_trailing_window(self, built: Any) -> None:
        store, _, _ = built
        state = store.load("r1")
        assert state is not None
        store.save(
            state.model_copy(
                update={
                    "policy": RoomPolicy(transcript_window=2, max_agent_turns_per_human_message=1)
                }
            )
        )
        seen: list[tuple[int, ...]] = []

        class Recording(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                seen.append(tuple(m.seq for m in request.transcript))
                return await super().select(request)

        orch = _orchestrator(built, [Recording("rec", [speak("scout")] * 4)])

        await orch.post("r1", "alice", "one")
        await orch.post("r1", "alice", "two")
        await orch.post("r1", "alice", "three")

        assert all(len(window) <= 2 for window in seen), seen
        assert seen[-1] == (4, 5), "the window must be the trailing slice, not the head"

    @pytest.mark.asyncio
    async def test_the_window_does_not_truncate_what_the_speaker_is_shown(self, built: Any) -> None:
        """The window bounds *selection* input. A speaker's own unseen span is separate."""
        store, _, agents = built
        state = store.load("r1")
        assert state is not None
        store.save(
            state.model_copy(
                update={
                    "policy": RoomPolicy(transcript_window=2, max_agent_turns_per_human_message=1)
                }
            )
        )
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")] * 3)])

        await orch.post("r1", "alice", "first")
        # scout does not speak to the second message; critic would, but nothing selects it.
        store.save(orch.append_human(store.load("r1"), "alice", "second"))  # type: ignore[arg-type]
        await orch.post("r1", "alice", "third")

        last = agents["scout"].prompts[-1]
        assert "second" in last and "third" in last, (
            "the speaker's span is bounded by its own high-water mark, not by the "
            f"selector's window: {last!r}"
        )


class TestProvenanceReachesTheTranscript:
    """§3.10's P6 claim was pinned on the decision side only; the utterance side was not."""

    @pytest.mark.asyncio
    async def test_a_turn_result_provenance_lands_on_the_room_message(self, built: Any) -> None:
        from uclone_x.core.provenance import Provenance

        _, _, agents = built
        prov = Provenance.primary(provider="mock", model="mock-model")

        class Attributing(FakeAgent):
            async def execute_turn(
                self, prompt: str, *, stream_callback: Any = None, **kwargs: Any
            ) -> Any:
                from uclone_x.agent.models import TurnResult

                self.prompts.append(prompt)
                return TurnResult(turn_index=1, content="answered", provenance=prov)

        agents["scout"] = Attributing("scout", SCOUT.session_id)
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])

        state = await orch.post("r1", "alice", "hi")

        message = state.transcript[-1]
        assert message.provenance is not None
        assert message.provenance.served_by.model == "mock-model"

    @pytest.mark.asyncio
    async def test_provenance_survives_the_store_round_trip(self, built: Any) -> None:
        """It is carried on a frozen model through JSON, not held only in memory."""
        from uclone_x.core.provenance import Provenance

        store, _, agents = built
        prov = Provenance.primary(provider="mock", model="mock-model")

        class Attributing(FakeAgent):
            async def execute_turn(
                self, prompt: str, *, stream_callback: Any = None, **kwargs: Any
            ) -> Any:
                from uclone_x.agent.models import TurnResult

                self.prompts.append(prompt)
                return TurnResult(turn_index=1, content="answered", provenance=prov)

        agents["scout"] = Attributing("scout", SCOUT.session_id)
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        await orch.post("r1", "alice", "hi")

        reloaded = store.load("r1")
        assert reloaded is not None
        assert reloaded.transcript[-1].provenance is not None
        assert reloaded.transcript[-1].decision is not None


class TestRenderedSpan:
    @pytest.mark.asyncio
    async def test_a_failed_turn_is_not_rendered_to_the_next_speaker(self, built: Any) -> None:
        """An empty utterance attributed to an agent is not something another can read."""
        _, _, agents = built
        agents["scout"].fail_with = RuntimeError("boom")
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])

        await orch.post("r1", "alice", "question")

        assert "[scout]" not in agents["critic"].prompts[0], agents["critic"].prompts[0]
        assert "question" in agents["critic"].prompts[0]

    @pytest.mark.asyncio
    async def test_a_speaker_is_not_shown_its_own_prior_utterance(self, built: Any) -> None:
        """It is already in that agent's session as its own assistant turn."""
        _, _, agents = built
        _cap(built, 1)
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")] * 2)])

        await orch.post("r1", "alice", "first")
        await orch.post("r1", "alice", "second")

        assert "[scout]" not in agents["scout"].prompts[1], agents["scout"].prompts[1]


class TestRosterChangesUnderTheLoop:
    @pytest.mark.asyncio
    async def test_a_speaker_that_left_during_selection_is_re_selected(self, built: Any) -> None:
        """A race is not a defect, and it is not a lost turn either (§3.8.1).

        A selector cannot know the roster moved after it answered; the orchestrator can.
        The first answer to #710 recorded a `SILENCE` and ended the loop, which spent the
        human's question on a participant who was never given the floor. The floor now goes
        back to the chain against the live roster: the chain is re-consulted, and only what
        it says the second time decides the room. Here it has nothing left to say, so the
        room falls to the chain's own recorded silence — and nobody is resolved, which is
        the part that says the departed participant never spoke.

        Killed by: src/uclone_x/room/orchestrator.py :: self._forgive_departure_race(state)  # (e)
        Becomes: return state  # (e)
        """
        store, resolver, _ = built

        class Departing(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                state = store.load("r1")
                assert state is not None
                store.save(state.model_copy(update={"participants": (ALICE, SCOUT)}))
                return await super().select(request)

        selector = Departing("dep", [speak("critic")])
        orch = _orchestrator(built, [selector])

        result = await orch.post("r1", "alice", "hi")

        assert [m.sender_id for m in result.transcript] == ["alice"]
        assert resolver.resolved == [], "a departed participant must not be given the floor"
        assert selector.calls == 2, "the chain must be re-consulted against the live roster"

    @pytest.mark.asyncio
    async def test_the_recorded_reason_is_the_chains_and_not_a_fabricated_departure(
        self, built: Any
    ) -> None:
        """The room reports why it actually stopped, which is no longer the departure.

        While a departure ended the loop, `last_decision` named who had left. Re-selecting
        makes that claim false: the departure was survived, and what stopped the room was
        the chain running out of judgement one round later. Recording the departure anyway
        would tell an operator the room went quiet because `critic` left when in fact the
        chain was asked again and declined — two different faults with two different
        remedies. The count on `TurnState.races_forgiven` is where the departure is
        recorded now, and this pins that the decision does not also try to.

        Killed by: src/uclone_x/room/orchestrator.py :: self._forgive_departure_race(state)  # (e)
        Becomes: return state  # (e)
        """
        store, _, _ = built

        class Departing(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                state = store.load("r1")
                assert state is not None
                store.save(state.model_copy(update={"participants": (ALICE, SCOUT)}))
                return await super().select(request)

        orch = _orchestrator(built, [Departing("dep", [speak("critic")])])

        result = await orch.post("r1", "alice", "hi")

        assert result.last_decision is not None
        assert result.last_decision.verdict is SelectionVerdict.SILENCE
        assert "abstained" in result.last_decision.reasoning, result.last_decision.reasoning
        assert "left" not in result.last_decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_departure_race_increments_races_forgiven(self, built: Any) -> None:
        """A forgiven departure race is observable via TurnState.races_forgiven (P6, #755).

        The counter outlived the `SILENCE` row that used to carry the same fact, and is now
        the **only** trace: a race the room recovers from leaves no utterance, no decision
        and no error behind it, so without the count a room that re-selected ten times and
        a room that never raced would be indistinguishable.

        Killed by: src/uclone_x/room/orchestrator.py :: update={"races_forgiven": state.turn_state.races_forgiven + 1}
        Becomes: update={"races_forgiven": state.turn_state.races_forgiven}
        """
        store, _, _ = built

        class Departing(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                state = store.load("r1")
                assert state is not None
                store.save(state.model_copy(update={"participants": (ALICE, SCOUT)}))
                return await super().select(request)

        orch = _orchestrator(built, [Departing("dep", [speak("critic")])])

        before = store.load("r1")
        assert before is not None
        assert before.turn_state.races_forgiven == 0

        result = await orch.post("r1", "alice", "hi")

        assert result.turn_state.races_forgiven == 1

    @pytest.mark.asyncio
    async def test_the_re_selected_round_can_still_give_the_floor_to_somebody_else(
        self, built: Any
    ) -> None:
        """Re-selecting is worth doing only if the recovered round can actually speak.

        The sibling tests end in silence because their chain is exhausted, so on their own
        they are also satisfied by a loop that falls through and then stops. This one gives
        the chain a second answer and pins that the turn the departure would have cost is
        taken by the agent the live roster still holds.

        Killed by: src/uclone_x/room/orchestrator.py :: self._forgive_departure_race(state)  # (e)
        Becomes: return state  # (e)
        """
        store, resolver, _ = built

        class DepartingOnce(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                if self.calls == 0:
                    state = store.load("r1")
                    assert state is not None
                    store.save(state.model_copy(update={"participants": (ALICE, SCOUT)}))
                return await super().select(request)

        orch = _orchestrator(built, [DepartingOnce("dep", [speak("critic"), speak("scout")])])

        result = await orch.post("r1", "alice", "hi")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout"]
        assert [r[0] for r in resolver.resolved] == ["scout"]
        assert result.turn_state.races_forgiven == 1

    @pytest.mark.asyncio
    async def test_a_departure_during_the_hesitation_pause_is_also_re_selected(
        self, built: Any
    ) -> None:
        """The second validation site is the same decision, and must answer it the same way.

        A low-confidence speaker is validated twice: once when the chain answers, and again
        after the hesitation pause, which is a second window of up to
        `RoomPolicy.hesitation_seconds` in which the roster can move. The two sites are far
        apart in the loop and the first answer to #710 was applied to both; leaving either
        on the old behaviour makes the room's response to a departure depend on whether
        hesitation happened to be enabled.

        Killed by: src/uclone_x/room/orchestrator.py :: self._forgive_departure_race(state)  # (f)
        Becomes: return state  # (f)
        """
        store, resolver, _ = built
        state = store.load("r1")
        assert state is not None
        store.save(
            state.model_copy(
                update={"policy": state.policy.model_copy(update={"hesitation_seconds": 0.5})}
            )
        )

        async def depart_during_the_pause() -> None:
            await asyncio.sleep(0.05)
            live = store.load("r1")
            assert live is not None
            store.save(live.model_copy(update={"participants": (ALICE, SCOUT)}))

        orch = _orchestrator(
            built, [ScriptedSelector("s", [speak("critic", confidence=0.0), speak("scout")])]
        )
        departure = asyncio.create_task(depart_during_the_pause())

        result = await asyncio.wait_for(orch.post("r1", "alice", "hi"), 5.0)
        await departure

        assert [m.sender_id for m in result.transcript] == ["alice", "scout"]
        assert [r[0] for r in resolver.resolved] == ["scout"]
        assert result.turn_state.races_forgiven == 1

    @pytest.mark.asyncio
    async def test_a_selector_that_keeps_naming_the_departed_agent_is_refused(
        self, built: Any
    ) -> None:
        """The fall-through is bounded by the very distinction that grants it (§3.8.1).

        Once the departure has landed, the roster the chain is *given* no longer holds
        `critic`, so naming it a second time is a selector defect rather than a race and
        the raise the first round withheld fires. That bound is not decoration: no turn is
        taken on a fall-through, so the step ceiling cannot stop a selector stuck on a
        departed participant — only this can.

        Killed by: src/uclone_x/room/orchestrator.py :: self._forgive_departure_race(state)  # (e)
        Becomes: return state  # (e)
        """
        store, _, _ = built

        class Stuck(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                self.calls += 1
                state = store.load("r1")
                assert state is not None
                store.save(state.model_copy(update={"participants": (ALICE, SCOUT)}))
                return speak("critic", selector=self.name)

        selector = Stuck("stuck")
        orch = _orchestrator(built, [selector])

        with pytest.raises(SpeakerSelectionError, match="critic"):
            await orch.post("r1", "alice", "hi")

        assert selector.calls == 2, "the race is forgiven once, and exactly once"

    @pytest.mark.asyncio
    async def test_the_fall_through_reaches_the_next_snapshot_without_suspending(
        self, built: Any
    ) -> None:
        """The bound rests on there being no suspension point on the fall-through path.

        §3.8.1 bounds the fall-through by re-reading the pre-selection snapshot: on the
        next iteration it no longer holds the departed agent, so naming it again raises.
        That argument is only sound while nothing can *put the agent back* between the
        `continue` and that re-read — and what guarantees it is not a lock but the plain
        absence of an `await` on that path, which no reader is obliged to notice and a
        later backoff `sleep` would quietly remove.

        So this test supplies the adversary the sibling test does not have: a task that
        re-admits `critic` at every opportunity the event loop gives it. While the path
        does not suspend, that task never gets its opportunity inside the window, the
        second round's snapshot lacks `critic`, and the raise still fires. Add one
        suspension point there and the re-admission lands inside the window, every round
        becomes a race again, and the loop no longer terminates — which is why the timeout
        is part of the assertion rather than a safety net: the suite configures no global
        timeout, so without it a regression hangs the suite instead of failing it.

        Killed by: src/uclone_x/room/orchestrator.py :: self._forgive_departure_race(state)  # (e)
        Becomes: return state  # (e)
        """
        store, resolver, _ = built

        class Stuck(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                self.calls += 1
                state = store.load("r1")
                assert state is not None
                store.save(state.model_copy(update={"participants": (ALICE, SCOUT)}))
                return speak("critic", selector=self.name)

        async def readmit_critic() -> None:
            """Put `critic` back the instant the loop yields anywhere."""
            while True:
                state = store.load("r1")
                assert state is not None
                if CRITIC not in state.participants:
                    store.save(state.model_copy(update={"participants": (ALICE, SCOUT, CRITIC)}))
                await asyncio.sleep(0)

        selector = Stuck("stuck")
        orch = _orchestrator(built, [selector])
        adversary = asyncio.create_task(readmit_critic())
        await asyncio.sleep(0)  # let the adversary reach its own suspension point first

        try:
            async with asyncio.timeout(5):
                with pytest.raises(SpeakerSelectionError, match="critic"):
                    await orch.post("r1", "alice", "hi")
        finally:
            adversary.cancel()
            with pytest.raises(asyncio.CancelledError):
                await adversary

        assert selector.calls == 2, "the race is still forgiven exactly once"
        assert resolver.resolved == [], "a departed participant must not be given the floor"

    @pytest.mark.asyncio
    async def test_a_speaker_who_joined_during_selection_is_still_a_defect(
        self, built: Any
    ) -> None:
        """A name becoming valid afterwards does not make choosing it a judgement.

        The selector is shown exactly the roster in its request, so a name outside that
        roster is a defect by construction — whatever the roster did next. Checking the
        live roster first would short-circuit this and reward the selector for a
        coincidence.
        """
        store, _, agents = built
        agents["newbie"] = FakeAgent("newbie", "sess_room__r1__newbie")
        newbie = agent_participant("newbie")

        class Joining(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                state = store.load("r1")
                assert state is not None
                store.save(state.model_copy(update={"participants": (*state.participants, newbie)}))
                return await super().select(request)

        orch = _orchestrator(built, [Joining("join", [speak("newbie")])])

        with pytest.raises(SpeakerSelectionError, match="newbie"):
            await orch.post("r1", "alice", "hi")

    @pytest.mark.asyncio
    async def test_a_human_the_chain_saw_is_a_defect_even_if_now_an_agent(self, built: Any) -> None:
        store, _, agents = built
        agents["alice"] = FakeAgent("alice", "sess_room__r1__alice")

        class Promoting(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                state = store.load("r1")
                assert state is not None
                promoted = ALICE.model_copy(
                    update={"kind": ParticipantKind.AGENT, "session_id": "sess_room__r1__alice"}
                )
                store.save(state.model_copy(update={"participants": (promoted, SCOUT, CRITIC)}))
                return await super().select(request)

        orch = _orchestrator(built, [Promoting("promote", [speak("alice")])])

        with pytest.raises(SpeakerSelectionError, match="alice"):
            await orch.post("r1", "alice", "hi")

    @pytest.mark.asyncio
    async def test_a_speaker_the_roster_never_had_still_raises(self, built: Any) -> None:
        """The distinction the code can make is exactly race versus selector defect.

        Nobody named `ghost` ever belonged to this room, so no roster change explains the
        decision: the selector is wrong, and a wrong selector must be loud.
        """
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("ghost")])])

        with pytest.raises(SpeakerSelectionError, match="ghost"):
            await orch.post("r1", "alice", "hi")

    @pytest.mark.asyncio
    async def test_a_room_with_no_agents_falls_silent_without_resolving_anyone(
        self, built: Any
    ) -> None:
        store, resolver, _ = built
        state = store.load("r1")
        assert state is not None
        store.save(state.model_copy(update={"participants": (ALICE,)}))
        orch = _orchestrator(built, [ScriptedSelector("s")])

        result = await orch.post("r1", "alice", "anyone?")

        assert result.last_decision is not None
        assert result.last_decision.verdict is SelectionVerdict.SILENCE
        assert resolver.resolved == []


class TestInterjectionDuringSelection:
    """#710(a): the re-read made a mid-selection message visible; the loop walked past it."""

    @pytest.mark.asyncio
    async def test_a_message_posted_during_selection_abandons_the_pending_turn(
        self, built: Any
    ) -> None:
        store, resolver, _ = built

        class Interjecting(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                decision = await super().select(request)
                state = store.load("r1")
                assert state is not None
                store.save(orch.append_human(state, "alice", "wait - different question"))
                return decision

        orch = _orchestrator(built, [Interjecting("int", [speak("scout")])])

        result = await orch.post("r1", "alice", "first question")

        assert [m.sender_id for m in result.transcript] == ["alice", "alice"]
        assert resolver.resolved == [], (
            "the human spoke while the chain was deciding; the pending turn is theirs to "
            "supersede, and the selection that produced it answered a superseded message"
        )

    @pytest.mark.asyncio
    async def test_the_check_runs_before_the_speaker_is_validated(self, built: Any) -> None:
        """Interjection outranks a departure: both happened, and the human wins."""
        store, resolver, _ = built

        class Both(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                decision = await super().select(request)
                state = store.load("r1")
                assert state is not None
                state = state.model_copy(update={"participants": (ALICE, SCOUT)})
                store.save(orch.append_human(state, "alice", "never mind"))
                return decision

        orch = _orchestrator(built, [Both("both", [speak("critic")])])

        result = await orch.post("r1", "alice", "hi")

        assert [m.sender_id for m in result.transcript] == ["alice", "alice"]
        # The loop returned on the interjection, so no departure decision was recorded.
        assert result.last_decision is None
        assert resolver.resolved == []


class TestInterjectionOutranksADecidedSilence:
    """Which of two stops wins, pinned rather than left to line order.

    A chain may decide `SILENCE` while a human is mid-sentence. The decision answers a
    message the human has already superseded, and their `append_human` has already nulled
    `last_decision`, so writing the silence back would attach a stale judgement to the new
    message. The interjection wins and nothing is recorded — which is what guard (c)
    sitting before the `SILENCE` branch means, and what this holds in place.
    """

    @pytest.mark.asyncio
    async def test_a_silence_decided_during_an_interjection_is_not_recorded(
        self, built: Any
    ) -> None:
        store, resolver, _ = built

        class SilentAndInterrupted(ScriptedSelector):
            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                decision = await super().select(request)
                state = store.load("r1")
                assert state is not None
                store.save(orch.append_human(state, "alice", "actually, never mind"))
                return decision

        orch = _orchestrator(built, [SilentAndInterrupted("s", [silence()])])

        result = await orch.post("r1", "alice", "anyone?")

        assert [m.sender_id for m in result.transcript] == ["alice", "alice"]
        assert result.last_decision is None, (
            "the silence answered a superseded message; recording it would attach a stale "
            "judgement to the message that replaced it"
        )
        assert resolver.resolved == []


class TestALateReplyDoesNotClobberANewerSilence:
    """A turn that started before a newer message must not overwrite that message's own
    decision when it finally lands (#945).

    Scout is given the floor for "hi". While its turn is still running, "anyone?" arrives,
    is unaddressed, and the chain decides `SILENCE` for it -- recorded immediately, since a
    decided silence needs no floor. Only then does scout's reply -- which answered "hi",
    and never saw "anyone?" -- land. `last_decision` must still read the silence that was
    decided about the newer message, not scout's stale speak decision about the older one.
    """

    @pytest.mark.asyncio
    async def test_a_reply_that_started_before_a_newer_silence_does_not_overwrite_it(
        self, built: Any
    ) -> None:
        """
        Killed by: src/uclone_x/room/orchestrator.py :: "last_decision": state.last_decision if superseded else decision,
        Becomes: "last_decision": decision,
        """
        store, _, agents = built
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), silence()])])
        agents["scout"].release = asyncio.Event()

        async def interject() -> None:
            await agents["scout"].turn_started.wait()
            await orch.post("r1", "alice", "anyone?")
            agents["scout"].release.set()

        await asyncio.gather(orch.post("r1", "alice", "hi"), interject())

        final = store.load("r1")
        assert final is not None
        assert [m.sender_id for m in final.transcript] == ["alice", "alice", "scout"]
        assert final.last_decision is not None
        assert final.last_decision.verdict is SelectionVerdict.SILENCE, (
            "scout's late reply to 'hi' overwrote the silence already decided about "
            "'anyone?', the newer message"
        )
        scout_row = final.transcript[-1]
        anyone_seq = final.transcript[1].seq
        assert scout_row.rendered_through < anyone_seq, (
            "scout's turn must have started before 'anyone?' existed for this to pin anything"
        )


class TestActivityWatermark:
    @pytest.mark.asyncio
    async def test_note_human_activity_on_an_unknown_room_refuses(self, built: Any) -> None:
        from uclone_x.errors import RoomNotFoundError

        orch = _orchestrator(built, [ScriptedSelector("s")])
        with pytest.raises(RoomNotFoundError):
            await orch.note_human_activity("no-such-room")

    @pytest.mark.asyncio
    async def test_typing_does_not_disturb_the_transcript_or_the_budget(self, built: Any) -> None:
        store, _, _ = built
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        await orch.post("r1", "alice", "hi")
        before = store.load("r1")
        assert before is not None

        await orch.note_human_activity("r1")

        after = store.load("r1")
        assert after is not None
        assert after.transcript == before.transcript
        assert after.turn_state.agent_turns_since_human == (
            before.turn_state.agent_turns_since_human
        )
        assert after.last_decision == before.last_decision


class TestTheWholeChainDrivesARealDebate:
    """The composition the review found untested: real selectors, real orchestrator."""

    @pytest.mark.asyncio
    async def test_mention_then_default_responder_over_two_messages(self, built: Any) -> None:
        from uclone_x.room.selectors import (
            DefaultResponderSelector,
            MentionSelector,
            SoleAgentSelector,
        )

        store, _, agents = built
        state = store.load("r1")
        assert state is not None
        store.save(state.model_copy(update={"policy": RoomPolicy(default_responder_id="critic")}))
        chain = [SoleAgentSelector(), MentionSelector(), DefaultResponderSelector()]
        orch = _orchestrator(built, chain)

        await orch.post("r1", "alice", "@scout what do you see?")
        result = await orch.post("r1", "alice", "and overall?")

        assert [m.sender_id for m in result.transcript] == [
            "alice",
            "scout",
            "alice",
            "critic",
        ]
        # critic's first turn must carry the whole conversation it has not seen.
        assert "what do you see?" in agents["critic"].prompts[0]
        assert "[scout]" in agents["critic"].prompts[0]

    @pytest.mark.asyncio
    async def test_the_model_is_first_consulted_only_after_the_address_was_answered_by_rule(
        self, built: Any
    ) -> None:
        """G5: the rule routes the address for free; the model's single call comes after.

        The LLM link *is* reached — once — and what it is asked is whether anyone should
        continue, with scout's answer already in the prompt it receives.
        """
        from uclone_x.llm.connectors.mock import MockLLMConnector
        from uclone_x.room.selectors import (
            LLMSpeakerSelector,
            MentionSelector,
            SoleAgentSelector,
        )

        captured: list[str] = []

        class Capturing(MockLLMConnector):
            async def generate(self, request):  # type: ignore[no-untyped-def,override]
                captured.append(request.messages[-1].content or "")
                return await super().generate(request)

        llm = Capturing(responses=['{"speaker_id": null, "reasoning": "answered"}'])
        chain = [SoleAgentSelector(), MentionSelector(), LLMSpeakerSelector(llm)]
        orch = _orchestrator(built, chain)

        result = await orch.post("r1", "alice", "@scout please")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout"]
        # The routing of the addressed message cost nothing: by the time the model was
        # first consulted, scout had already answered, and the only question left was
        # whether anyone should continue.
        assert llm.call_count == 1
        assert "[scout]" in captured[0], (
            "the model was asked to route the address itself, not to continue past it"
        )

    @pytest.mark.asyncio
    async def test_the_llm_link_continues_an_unaddressed_discussion_then_stops(
        self, built: Any
    ) -> None:
        """The LLM selector alone may continue after an agent spoke — and may end it."""
        from uclone_x.llm.connectors.mock import MockLLMConnector
        from uclone_x.room.selectors import LLMSpeakerSelector, MentionSelector

        llm = MockLLMConnector(
            responses=[
                '{"speaker_id": "scout", "confidence": 0.9}',
                '{"speaker_id": "critic", "confidence": 0.7}',
                '{"speaker_id": null, "reasoning": "the exchange has concluded"}',
            ]
        )
        orch = _orchestrator(built, [MentionSelector(), LLMSpeakerSelector(llm)])

        result = await orch.post("r1", "alice", "discuss caching, whoever fits")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout", "critic"]
        assert result.last_decision is not None
        assert result.last_decision.verdict is SelectionVerdict.SILENCE
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_the_cap_stops_a_real_llm_driven_discussion(self, built: Any) -> None:
        """A model that never chooses silence is bounded by G6, not by its own judgement."""
        from uclone_x.llm.connectors.mock import MockLLMConnector
        from uclone_x.room.selectors import LLMSpeakerSelector

        llm = MockLLMConnector(default_response='{"speaker_id": "scout"}')
        orch = _orchestrator(built, [LLMSpeakerSelector(llm)])

        result = await orch.post("r1", "alice", "go forever")

        assert sum(1 for m in result.transcript if m.sender_id == "scout") == 3
        assert llm.call_count == 3


class TestNonAsciiContent:
    @pytest.mark.asyncio
    async def test_content_and_speaker_names_survive_the_round_trip(self, built: Any) -> None:
        store, _, agents = built
        agents["scout"]._reply = "TTL 기반이 단순합니다"
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])

        await orch.post("r1", "alice", "캐시 무효화 전략은?")

        reloaded = store.load("r1")
        assert reloaded is not None
        assert reloaded.transcript[0].content == "캐시 무효화 전략은?"
        assert "캐시 무효화 전략은?" in agents["critic"].prompts[0]
        assert "TTL 기반이 단순합니다" in agents["critic"].prompts[0]


class TestStoreDeclaresItsEncoding:
    """`RoomStore` names `encoding="utf-8"` on both sides, and #727 found neither held.

    Deleting the argument from `save`, or from `load`, left the suite green: the machines
    that run it already default to UTF-8, so the explicit argument is a no-op locally and
    matters only on a host where it is not. `TestNonAsciiContent` documented the intent and
    pinned none of it.

    The pin has to be a subprocess. `-X warn_default_encoding` is an interpreter flag, and
    monkeypatching `locale.getencoding` does not change what `open()` and `read_text()`
    default to — measured, and the write still came out UTF-8. Under the flag plus
    `-W error::EncodingWarning`, any `open`/`read_text` with no explicit `encoding` raises,
    so each deletion fails this test.
    """

    @staticmethod
    def _run(tmp_path: Any) -> Any:
        import os
        import subprocess
        import sys
        import textwrap
        from pathlib import Path

        src_root = Path(__file__).resolve().parents[2] / "src"
        script = textwrap.dedent(
            """
            import pathlib, sys

            import uclone_x.room.store as store_mod

            # The shared `.venv` carries a `.pth` putting the PRIMARY workspace's `src` on
            # the path, so a subprocess that inherits it measures the wrong tree and keeps
            # passing after the mutation. Assert the tree under test before touching it.
            src_root = pathlib.Path(sys.argv[2]).resolve()
            loaded = pathlib.Path(store_mod.__file__).resolve()
            assert loaded.is_relative_to(src_root), f"measured {loaded}, expected under {src_root}"

            from uclone_x.room.models import RoomMessage, RoomState
            from uclone_x.room.store import RoomStore

            store = RoomStore(sys.argv[1])
            saved = store.save(
                RoomState(
                    room_id="enc",
                    transcript=(RoomMessage(seq=1, sender_id="alice", content="캐시 무효화 전략"),),
                )
            )
            back = store.load("enc")
            assert back is not None
            assert back.transcript[0].content == "캐시 무효화 전략"
            assert back.revision == saved.revision
            print("ok")
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(src_root)
        env.pop("PYTHONWARNINGS", None)
        return subprocess.run(
            [
                sys.executable,
                "-X",
                "warn_default_encoding",
                "-W",
                "error::EncodingWarning",
                "-c",
                script,
                str(tmp_path),
                str(src_root),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def test_neither_save_nor_load_relies_on_the_host_default(self, tmp_path: Any) -> None:
        result = self._run(tmp_path)
        assert result.returncode == 0, (
            "a round trip through RoomStore raised under -X warn_default_encoding, so one "
            f"side reads or writes at the host's default encoding.\n{result.stderr}"
        )
        assert result.stdout.strip().endswith("ok")


# --------------------------------------------------------------------------------------
# UX review: three defects a user meets before any head exists. Each is behaviour in the
# Core, not a surface, so each is wrong whichever head is built on top.
# --------------------------------------------------------------------------------------


class TestTheSpanIsBounded:
    """A span had no ceiling, so a long room's cost grew without bound per new speaker."""

    @pytest.mark.asyncio
    async def test_a_long_unseen_span_is_truncated_to_the_most_recent(self, built: Any) -> None:
        store, _, agents = built
        state = store.load("r1")
        assert state is not None
        store.save(
            state.model_copy(
                update={"policy": RoomPolicy(max_span_messages=3, transcript_window=50)}
            )
        )
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")] * 20)])

        for i in range(6):
            store.save(orch.append_human(store.load("r1"), "alice", f"line {i}"))  # type: ignore[arg-type]
        await orch.post("r1", "alice", "now answer")

        prompt = agents["scout"].prompts[0]
        assert "line 0" not in prompt, "the oldest unseen messages must fall off the front"
        assert "now answer" in prompt, "the most recent must always be present"
        assert prompt.count("\n") == 3, prompt

    @pytest.mark.asyncio
    async def test_the_truncation_is_announced_rather_than_silent(self, built: Any) -> None:
        """Dropping conversation an agent has never seen is a real loss; it must say so."""
        store, _, agents = built
        state = store.load("r1")
        assert state is not None
        store.save(state.model_copy(update={"policy": RoomPolicy(max_span_messages=2)}))
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")] * 20)])

        for i in range(4):
            store.save(orch.append_human(store.load("r1"), "alice", f"line {i}"))  # type: ignore[arg-type]
        await orch.post("r1", "alice", "go")

        assert "3 earlier message" in agents["scout"].prompts[0]

    @pytest.mark.asyncio
    async def test_a_short_span_is_untouched_and_unannotated(self, built: Any) -> None:
        _, _, agents = built
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])

        await orch.post("r1", "alice", "just the one")

        assert agents["scout"].prompts[0] == "[alice]: just the one"


class TestEveryAddressedAgentAnswers:
    """`@scout @critic compare` answered with one voice and said nothing about the other."""

    @pytest.mark.asyncio
    async def test_two_mentioned_agents_both_speak_in_mention_order(self, built: Any) -> None:
        from uclone_x.room.selectors import MentionSelector

        orch = _orchestrator(built, [MentionSelector()])

        result = await orch.post("r1", "alice", "@critic @scout compare your views")

        assert [m.sender_id for m in result.transcript] == ["alice", "critic", "scout"]

    @pytest.mark.asyncio
    async def test_an_agent_that_has_already_answered_is_not_re_elected(self, built: Any) -> None:
        """The no-re-firing property must survive the change that lets the second speak."""
        from uclone_x.room.selectors import MentionSelector

        orch = _orchestrator(built, [MentionSelector()])

        result = await orch.post("r1", "alice", "@scout have a look")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout"]

    @pytest.mark.asyncio
    async def test_the_second_speaker_sees_what_the_first_said(self, built: Any) -> None:
        from uclone_x.room.selectors import MentionSelector

        _, _, agents = built
        orch = _orchestrator(built, [MentionSelector()])

        await orch.post("r1", "alice", "@scout @critic compare")

        assert "[scout]" in agents["critic"].prompts[0]

    @pytest.mark.asyncio
    async def test_the_turn_cap_still_bounds_a_long_mention_list(self, built: Any) -> None:
        from uclone_x.room.selectors import MentionSelector

        store, _, agents = built
        for extra in ("a", "b", "c"):
            agents[extra] = FakeAgent(extra, f"sess_room__r1__{extra}")
        state = store.load("r1")
        assert state is not None
        store.save(
            state.model_copy(
                update={
                    "participants": (
                        *state.participants,
                        *(agent_participant(x) for x in ("a", "b", "c")),
                    )
                }
            )
        )
        orch = _orchestrator(built, [MentionSelector()])

        result = await orch.post("r1", "alice", "@scout @critic @a @b @c all of you")

        assert sum(1 for m in result.transcript if m.sender_id != "alice") == 3


class TestTheHeadCanFollowAlong:
    """`post()` returned only after every turn, and published nothing in between."""

    @pytest.mark.asyncio
    async def test_each_turn_is_published_as_it_lands(self, built: Any) -> None:
        from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType

        store, resolver, _ = built
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            from uclone_x.room.orchestrator import RoomOrchestrator

            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout"), speak("critic")])],
                resolver=resolver,
                bus=bus,
            )
            await orch.post("r1", "alice", "discuss")

            events: list[AgentEvent] = []
            while True:
                try:
                    events.append(await asyncio.wait_for(sub.get(), 0.3))
                except TimeoutError:
                    break

        starts = [
            e
            for e in events
            if e.type is EventType.AGENT_REPLY and e.payload.get("status") == "generating"
        ]
        assert [e.payload["agent_id"] for e in starts] == ["scout", "critic"]
        # No `seq`: the row is not known until the turn is over. See
        # `test_a_turn_is_identified_by_the_turn_and_not_by_a_guessed_row`.
        assert "seq" not in starts[0].payload
        assert starts[0].payload["turn_id"]

        replies = [
            e
            for e in events
            if e.type is EventType.AGENT_REPLY and e.payload.get("status") == "final"
        ]
        assert [e.payload["agent_id"] for e in replies] == ["scout", "critic"]
        assert replies[0].payload["content"] == "scout says TTL"
        assert replies[0].payload["seq"] == 2

    @pytest.mark.asyncio
    async def test_a_failed_turn_is_published_too(self, built: Any) -> None:
        from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType

        store, resolver, agents = built
        agents["scout"].fail_with = RuntimeError("boom")
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            from uclone_x.room.orchestrator import RoomOrchestrator

            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )
            await orch.post("r1", "alice", "hi")

            events: list[AgentEvent] = []
            while True:
                try:
                    events.append(await asyncio.wait_for(sub.get(), 0.3))
                except TimeoutError:
                    break

        # Selected by the status it has, not by the one it does not: the topic carries
        # four kinds of AGENT_REPLY, and "not generating" silently began matching every
        # token delta the moment streaming was added. This one survived that correction
        # only because its agent fails before emitting a chunk.
        replies = [
            e
            for e in events
            if e.type is EventType.AGENT_REPLY and e.payload.get("status") == "final"
        ]
        assert len(replies) == 1
        assert "boom" in str(replies[0].payload["error"])

    @pytest.mark.asyncio
    async def test_a_turn_that_returns_its_failure_after_streaming_lands_as_a_failed_row(
        self, built: Any
    ) -> None:
        """The partial reply is superseded by a row that says the turn failed (#938).

        `BaseAgent.execute_turn` does not raise a turn's failure; it returns it on
        `TurnResult.error`. The room read only `content` and `provenance` from a result, so
        a real agent's failure landed as a *successful* empty row: no `error` on the landed
        event, the partial deltas replaced by nothing, the unseen span consumed, and no
        failed turn for `retry` to re-run. A stream that fails mid-reply is exactly such a
        failure, and the landed row is the only signal a head has that the words it just
        showed are not the answer.

        Killed by: src/uclone_x/room/orchestrator.py :: error = result.error
        Becomes: error = None
        """
        from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType

        store, resolver, agents = built
        _cap(built, 1)
        agents["scout"].chunks = ["The ", "partial "]
        failure = "The streamed reply from ollama/scripted-model was interrupted after 2 chunk(s)"
        agents["scout"].result_error = failure
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            from uclone_x.room.orchestrator import RoomOrchestrator

            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )
            state = await orch.post("r1", "alice", "discuss")

            events: list[AgentEvent] = []
            while True:
                try:
                    events.append(await asyncio.wait_for(sub.get(), 0.3))
                except TimeoutError:
                    break

        replies = [e.payload for e in events if e.type is EventType.AGENT_REPLY]
        assert [r["status"] for r in replies] == ["generating", "streaming", "streaming", "final"]
        landed = replies[-1]
        assert landed["error"] == failure
        assert landed["content"] == ""
        assert landed["turn_id"] == replies[1]["turn_id"]

        failed = state.transcript[-1]
        assert failed.error == failure
        assert state.last_seen_seq.get("scout") is None, "a failed turn consumed the span"

    @pytest.mark.asyncio
    async def test_tokens_reach_the_topic_as_they_are_produced(self, built: Any) -> None:
        """Without this the head shows a blank screen for the whole of a turn.

        `post()` returns once every turn has landed, and the landed-reply event is
        published after the write, so a room that forwards nothing in between is a
        conversation that appears frozen for as long as the model takes -- while the same
        model streamed token by token on the retired `/api/chat/stream` (#1208). Unifying
        kinds on rooms without this makes the product's main screen worse, which is why it
        is a prerequisite and not an enhancement.

        Killed by: src/uclone_x/room/orchestrator.py :: status": "streaming
        Becomes: status": "generating
        """
        from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType

        store, resolver, agents = built
        agents["scout"].chunks = ["The ", "composite ", "index"]
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            from uclone_x.room.orchestrator import RoomOrchestrator

            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )
            await orch.post("r1", "alice", "discuss")

            events: list[AgentEvent] = []
            while True:
                try:
                    events.append(await asyncio.wait_for(sub.get(), 0.3))
                except TimeoutError:
                    break

        statuses = [e.payload.get("status") for e in events if e.type is EventType.AGENT_REPLY]
        assert statuses == ["generating", "streaming", "streaming", "streaming", "final"], (
            "a delta must arrive between the floor being taken and the utterance landing"
        )

        deltas = [e.payload for e in events if e.payload.get("status") == "streaming"]
        assert [d["delta"] for d in deltas] == ["The ", "composite ", "index"]
        # Addressed to the same speaker and the same row the final reply will carry, so a
        # head can accumulate deltas against the row rather than guessing where they go.
        assert [d["agent_id"] for d in deltas] == ["scout", "scout", "scout"]
        # Addressed by turn, not by a row number guessed before the turn ran, and the
        # landed reply names the same turn so a head can close the bubble it opened.
        turn_ids = [str(d["turn_id"]) for d in deltas]
        landed = [str(e.payload["turn_id"]) for e in events if e.payload.get("status") == "final"]
        assert len(set(turn_ids)) == 1
        assert set(turn_ids) == set(landed)

    @pytest.mark.asyncio
    async def test_private_reasoning_is_not_forwarded_to_the_room(self, built: Any) -> None:
        """G1/G2: the room's channel carries what was said, not how it was reached.

        Rewritten after a mutation run showed the first version could not fail: it looked
        for a forwarded event with `status == "thinking"`, and the forwarder hard-codes
        `status: "streaming"` on everything it publishes, so the collection it asserted
        over was empty by construction. Removing the discriminator entirely left it
        green. It now counts what arrives and looks for the scratchpad's own text, both
        of which change the moment a non-token event is let through.

        Killed by: src/uclone_x/room/orchestrator.py :: if event_name != "token":
        Becomes: if False:
        """
        from uclone_x.engine.event_bus import AgentEvent, EventBus

        store, resolver, agents = built
        agents["scout"].chunks = ["hi"]
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            from uclone_x.room.orchestrator import RoomOrchestrator

            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )
            await orch.post("r1", "alice", "discuss")

            events: list[AgentEvent] = []
            while True:
                try:
                    events.append(await asyncio.wait_for(sub.get(), 0.3))
                except TimeoutError:
                    break

        deltas = [e.payload for e in events if e.payload.get("status") == "streaming"]
        assert [d["delta"] for d in deltas] == ["hi"], (
            "exactly the utterance's own chunks reach the room, one event each"
        )
        assert not [e for e in events if "PRIVATE-SCRATCHPAD" in json.dumps(unwrap(e.payload))], (
            "an agent's scratchpad reached the topic every participant's head reads"
        )

    @pytest.mark.asyncio
    async def test_a_room_without_a_bus_hands_the_agent_no_stream_callback(
        self, built: Any
    ) -> None:
        """No listener, no per-token cost. The callback is the whole of the overhead."""
        _store, _resolver, agents = built
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])

        await orch.post("r1", "alice", "hi")

        assert agents["scout"].stream_callbacks == [None]

    @pytest.mark.asyncio
    async def test_a_stream_that_cannot_be_published_is_dropped_not_retried(
        self, built: Any
    ) -> None:
        """A failing head costs the room its streaming, never its turn -- or its speed.

        Per-chunk publishing multiplies the one failure `_publish` already tolerates by
        however many tokens the answer runs to. Retrying each one turns a broken
        subscriber into a per-token exception storm in the middle of a turn, so the first
        failure ends streaming for that turn and the utterance still lands whole.
        """
        from uclone_x.engine.event_bus import EventBus

        store, resolver, agents = built
        agents["scout"].chunks = ["a", "b", "c", "d"]
        attempts: list[str] = []

        class BrokenBus(EventBus):
            def register_publisher(self, sender_id: str, *args: Any, **kwargs: Any) -> Any:
                attempts.append(sender_id)
                raise RuntimeError("the bus is down")

        from uclone_x.room.orchestrator import RoomOrchestrator

        async with BrokenBus() as bus:
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )
            result = await orch.post("r1", "alice", "discuss")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout"]
        assert result.transcript[-1].content == "scout says TTL"
        # Three publishing sites attempt once each -- the floor being taken, the stream,
        # and the landed reply. Four chunks would make six attempts if the stream retried.
        assert len(attempts) <= 3, f"registration is per publishing site, not per token: {attempts}"

    @pytest.mark.asyncio
    async def test_a_turn_is_identified_by_the_turn_and_not_by_a_guessed_row(
        self, built: Any
    ) -> None:
        """Deltas name the turn that produced them, not a row number guessed before it.

        The row an utterance lands on is not known until the turn is over: `append_human`
        is a documented mid-turn seam, so anything appended while the agent generates
        shifts it. Publishing a row number up front addressed every delta of the agent's
        reply to whichever row that number turned out to be -- the human's own
        interjection, in the case below -- and a head matching the landed reply against
        it never cleared the live bubble, so the conversation stayed stuck mid-turn with
        no way back.

        `turn_id` is knowable at the start and does not move, which is why the three
        statuses carry it and only `final` carries a `seq`.

        Killed by: src/uclone_x/room/orchestrator.py :: turn_id = uuid.uuid4().hex
        Becomes: turn_id = ""
        """
        from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType

        store, resolver, agents = built
        scout = agents["scout"]
        scout.chunks = ["The ", "index"]
        scout.release = asyncio.Event()

        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            from uclone_x.room.orchestrator import RoomOrchestrator

            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )
            cascade = asyncio.create_task(orch.post("r1", "alice", "q"))
            await scout.turn_started.wait()
            # The interjection lands between the floor being given and the reply landing,
            # which is exactly what the seam exists for.
            state = store.load("r1")
            assert state is not None
            store.save(orch.append_human(state, "alice", "WAIT"))
            scout.release.set()
            await cascade

            events: list[AgentEvent] = []
            while True:
                try:
                    events.append(await asyncio.wait_for(sub.get(), 0.3))
                except TimeoutError:
                    break

        replies = [e.payload for e in events if e.type is EventType.AGENT_REPLY]
        starts = [p for p in replies if p.get("status") == "generating"]
        deltas = [p for p in replies if p.get("status") == "streaming"]
        final = [p for p in replies if p.get("status") == "final"]
        assert len(starts) == 1 and len(final) == 1

        turn_id = starts[0]["turn_id"]
        assert turn_id, "a turn must be identifiable before it has a row"
        assert [d["turn_id"] for d in deltas] == [turn_id, turn_id]
        assert final[0]["turn_id"] == turn_id, (
            "the landed reply must name the turn its deltas named, or a head cannot "
            "match them and the live bubble never clears"
        )

        # The provisional row number is gone rather than wrong: the human's interjection
        # took the row this turn was announced against.
        assert "seq" not in starts[0]
        assert all("seq" not in d for d in deltas)
        stored = store.load("r1")
        assert stored is not None
        assert [m.sender_id for m in stored.transcript] == ["alice", "alice", "scout"]
        assert final[0]["seq"] == 3

    @pytest.mark.asyncio
    async def test_stream_callback_forwards_status_updates(self, built: Any) -> None:
        """Lightweight tool-calling and progress notices are published as status_update.

        Allows the UI to display live status (e.g. 'Running tool: generate_image...')
        instead of generic 'Thinking...' while an agent is executing tools.

        Killed by: src/uclone_x/room/orchestrator.py :: "status": "status_update",
        Becomes: "status": "streaming",
        """
        from uclone_x.engine.event_bus import EventBus, EventType
        from uclone_x.room.orchestrator import RoomOrchestrator

        store, resolver, _agents = built
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            orch = RoomOrchestrator(
                store=store,
                selectors=[],
                resolver=resolver,
                bus=bus,
            )
            callback = orch._stream_callback("r1", "scout", "turn-42")  # pyright: ignore[reportPrivateUsage]
            assert callback is not None

            # Stream a status update
            await callback(
                "status",
                {"status": "calling_tool", "detail": "Running tool: generate_image..."},
            )

            event = await asyncio.wait_for(sub.get(), timeout=1.0)
            assert event.type is EventType.AGENT_REPLY
            assert event.payload == {
                "room_id": "r1",
                "agent_id": "scout",
                "turn_id": "turn-42",
                "status": "status_update",
                "detail": "Running tool: generate_image...",
            }

    @pytest.mark.asyncio
    async def test_the_heads_payload_fixture_is_what_the_room_publishes(
        self, built: Any, tmp_path: Any
    ) -> None:
        """`frontend/src/test/room-stream-events.json` is this run's output, and stays so.

        The head folds these payloads into its live bubble, and its only accumulation test
        once fed it a `seq` on `generating` and `streaming` -- a field the room never sends
        on either. The fold keyed on that absent field replaced the bubble on every delta,
        and the test stayed green because its input could not occur. So the head's tests
        read their payloads from this file, and this test fails the moment the file and the
        room disagree: a field added, dropped or renamed on the Core side reaches the head's
        tests instead of passing them by.

        **Every kind of event the topic carries, not only the landed turns** (#929). The
        head's `error` case was once the one payload its tests wrote by hand, and the
        composing notice and the interrupt were in no test at all -- which is how a head
        that read them as landed replies went unnoticed. So the run is a conversation that
        produces each of them: a human composing, two turns, a turn stopped by Stop while it
        writes, and a cascade that fails before anyone speaks, announced by the HTTP layer
        (`RoomStack.drive`) exactly as a send would be.

        Serialized the way `/api/stream` serializes -- `event_type` from the event, the
        payload through `unwrap_immutable` and JSON -- since those are the two fields
        `App.tsx` hands to `applyRoomEvent`. `turn_id` is a fresh uuid per turn,
        so it is replaced by its order of first appearance; what the head relies on is that
        the three statuses of one turn carry the same value and the next turn's differs.

        Killed by: src/uclone_x/room/orchestrator.py :: "status": "streaming",
        Becomes: "status": "streaming", "seq": 0,
        """
        from pathlib import Path
        from types import SimpleNamespace

        from uclone_x.core.immutable import unwrap_immutable
        from uclone_x.engine.event_bus import AgentEvent, EventBus
        from uclone_x.room.orchestrator import RoomOrchestrator
        from uclone_x.room.selectors import MentionSelector
        from uclone_x.ui.rooms import RoomStack

        store, resolver, agents = built
        scout, critic = agents["scout"], agents["critic"]
        scout.chunks = ["scout ", "says ", "TTL"]
        critic.chunks = ["critic ", "disagrees"]
        events: list[AgentEvent] = []
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            orch = RoomOrchestrator(
                store=store,
                selectors=[
                    ScriptedSelector(
                        "s", [speak("scout"), speak("critic"), silence(), speak("scout")]
                    )
                ],
                resolver=resolver,
                bus=bus,
            )

            # The human composing, then two turns.
            await orch.note_human_activity("r1", "alice")
            await orch.post("r1", "alice", "discuss")

            # Stop, pressed while scout is writing its next turn.
            scout.turn_started.clear()
            scout.release = asyncio.Event()
            stopped = asyncio.create_task(orch.post("r1", "alice", "and the other index?"))
            await asyncio.wait_for(scout.turn_started.wait(), 2.0)
            await orch.interrupt("r1", reason="Stopped by the user")
            await asyncio.wait_for(stopped, 2.0)

            # A cascade the HTTP layer runs and announces the failure of: an address that
            # names nobody, refused by the chain before any turn is given.
            def ignore_llm(_: object) -> None:
                return None

            stack = RoomStack(
                cast(
                    Any,
                    SimpleNamespace(
                        storage_dir=tmp_path / "stack", bus=bus, on_llm_replaced=ignore_llm
                    ),
                )
            )
            addressed = RoomOrchestrator(
                store=store, selectors=[MentionSelector()], resolver=resolver, bus=bus
            )
            stack.drive("r1", lambda: addressed.post("r1", "alice", "@phantm are you there?"))

            while not any(e.payload.get("status") == "error" for e in events):
                events.append(await asyncio.wait_for(sub.get(), 2.0))
            while True:
                try:
                    events.append(await asyncio.wait_for(sub.get(), 0.3))
                except TimeoutError:
                    break
            await stack.close()

        published: list[dict[str, Any]] = [
            {
                "event_type": e.type.value,
                "payload": json.loads(json.dumps(unwrap_immutable(e.payload))),
            }
            for e in events
        ]
        turns: dict[str, str] = {}
        for event in published:
            payload = event["payload"]
            if "turn_id" in payload:
                payload["turn_id"] = turns.setdefault(payload["turn_id"], f"turn-{len(turns) + 1}")

        assert [e["event_type"] for e in published].count("INTERRUPT") == 1
        assert [e["event_type"] for e in published].count("USER_INPUT") == 1
        assert [e["payload"].get("status") for e in published].count("error") == 1

        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "frontend"
            / "src"
            / "test"
            / "room-stream-events.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert fixture["events"] == published, (
            f"{fixture_path} no longer matches what the room publishes, so the head's tests "
            "are folding payloads the server does not send. Replace its `events` with:\n"
            + json.dumps(published, indent=2)
        )

    @pytest.mark.asyncio
    async def test_a_room_without_a_bus_still_runs(self, built: Any) -> None:
        """The bus is optional: a CLI that wants no streaming must not have to fake one."""
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])

        result = await orch.post("r1", "alice", "hi")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout"]


class TestAnUnknownMentionIsDecidedBeforeAnyTurn:
    """#757 review, finding 1: the refusal landed after a turn had already been given."""

    @pytest.mark.asyncio
    async def test_a_typo_after_a_valid_name_still_refuses_before_speaking(
        self, built: Any
    ) -> None:
        from uclone_x.errors import UnknownRoomParticipantError
        from uclone_x.room.selectors import MentionSelector

        store, resolver, _ = built
        orch = _orchestrator(built, [MentionSelector()])

        with pytest.raises(UnknownRoomParticipantError, match="phantm"):
            await orch.post("r1", "alice", "@scout and @phantm go")

        after = store.load("r1")
        assert after is not None
        assert [m.sender_id for m in after.transcript] == ["alice"], (
            "a turn was given for an utterance the room then refused"
        )
        assert resolver.resolved == []

    @pytest.mark.asyncio
    async def test_the_order_of_the_names_does_not_change_the_outcome(self, built: Any) -> None:
        """`@scout @phantm` and `@phantm @scout` are the same utterance class."""
        from uclone_x.errors import UnknownRoomParticipantError
        from uclone_x.room.selectors import MentionSelector

        store, _, _ = built
        orch = _orchestrator(built, [MentionSelector()])

        for text in ("@scout and @phantm go", "@phantm and @scout go"):
            store.save(
                (store.load("r1")).model_copy(update={"transcript": ()})  # type: ignore[union-attr]
            )
            with pytest.raises(UnknownRoomParticipantError):
                await orch.post("r1", "alice", text)
            state = store.load("r1")
            assert state is not None
            assert [m.sender_id for m in state.transcript] == ["alice"], text


class TestThePresentationChannelCannotStopTheRoom:
    """#757 review, finding 2: an optional channel could abort orchestration."""

    @pytest.mark.asyncio
    async def test_a_bus_that_raises_does_not_truncate_the_conversation(self, built: Any) -> None:
        from uclone_x.engine.event_bus import EventBus

        store, resolver, _ = built

        class BrokenBus(EventBus):
            def register_publisher(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("the bus is down")

        from uclone_x.room.orchestrator import RoomOrchestrator

        async with BrokenBus() as bus:
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout"), speak("critic")])],
                resolver=resolver,
                bus=bus,
            )
            result = await orch.post("r1", "alice", "discuss")

        assert [m.sender_id for m in result.transcript] == ["alice", "scout", "critic"], (
            "a head's channel failing must not cost the room a turn"
        )

    @pytest.mark.asyncio
    async def test_a_publisher_is_registered_once_per_speaker(self, built: Any) -> None:
        """Re-registering replaces the handle with an unrestricted one on every turn."""
        from uclone_x.engine.event_bus import EventBus

        store, resolver, _ = built
        registrations: list[str] = []

        class CountingBus(EventBus):
            def register_publisher(self, sender_id: str, *args: Any, **kwargs: Any) -> Any:
                registrations.append(sender_id)
                return super().register_publisher(sender_id, *args, **kwargs)

        from uclone_x.room.orchestrator import RoomOrchestrator

        async with CountingBus() as bus:
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")] * 3)],
                resolver=resolver,
                bus=bus,
            )
            await orch.post("r1", "alice", "go")

        assert registrations.count("scout") == 1, registrations


class TestTwoSendsDoNotRunTwoCascades:
    """`post` was awaited by its one caller, so an overlap was unreachable.

    Splitting it into `accept` + `resume` made it reachable, and a head that answers 202
    and drives the rest behind the response produces it from a double-clicked send
    button. Two loops then selected and gave the floor independently, which breaks the
    invariant the floor exists to state, and a finishing turn evicted the live turn of
    the *other* loop, so Stop reported success against an agent that kept generating.
    """

    @pytest.mark.asyncio
    async def test_a_second_send_does_not_put_two_agents_on_the_floor(self, built: Any) -> None:
        """One floor, one speaker -- including while a second cascade is starting.

        Killed by: src/uclone_x/room/orchestrator.py :: floor = self._floor.setdefault(room_id, asyncio.Lock())
        Becomes: floor = asyncio.Lock()
        """
        store, _resolver, agents = built
        scout, critic = agents["scout"], agents["critic"]
        scout.release = asyncio.Event()

        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")] * 4)])

        first = await orch.accept("r1", "alice", "one")
        cascade_a = asyncio.create_task(orch.resume("r1", first.transcript[-1].seq))
        await scout.turn_started.wait()

        # The second message arrives while scout is still generating.
        second = await orch.accept("r1", "alice", "two")
        cascade_b = asyncio.create_task(orch.resume("r1", second.transcript[-1].seq))
        # Give the second cascade every chance to select and give the floor.
        for _ in range(20):
            await asyncio.sleep(0)

        assert not critic.turn_started.is_set(), (
            "a second cascade gave the floor while the first agent was still generating"
        )

        scout.release.set()
        await asyncio.gather(cascade_a, cascade_b)

        # The older loop was abandoned rather than resumed: it recorded the turn it had
        # already given and then stopped, so nobody speaks twice for one message.
        final = store.load("r1")
        assert final is not None
        assert [m.sender_id for m in final.transcript][:3] == ["alice", "alice", "scout"]

    @pytest.mark.asyncio
    async def test_a_loop_waiting_for_the_floor_stands_down_for_a_newer_message(
        self, built: Any
    ) -> None:
        """The generation counter, in the one window nothing else guards.

        A loop checks for an interjection before selection and again after it, and then
        waits for the floor. A human message that arrives *during that wait* is invisible
        to both checks, and the loop would take the floor to answer a message the human has
        already superseded. The only thing that sees it is the generation `accept` bumps,
        re-read under the floor. The loop-top generation check is not what this pins: every
        `accept` also appends a human row, so guard (a) in `_run_turns` answers the same
        question there.

        Here scout holds the floor for the first message. The second message's loop selects
        critic and queues for the floor; a third message is accepted while it waits. When
        scout lands, the second loop must stand down, and only the third is answered.

        Killed by: src/uclone_x/room/orchestrator.py :: self._generation[room_id] = self._generation.get(room_id, 0) + 1
        Becomes: self._generation[room_id] = self._generation.get(room_id, 0)
        """
        store, _resolver, agents = built
        scout, critic = agents["scout"], agents["critic"]
        scout.release = asyncio.Event()

        class Announcing(ScriptedSelector):
            """Says when the second loop has decided -- after which it only awaits the floor."""

            def __init__(self, decisions: list[SpeakerDecision]) -> None:
                super().__init__("s", decisions)
                self.second_decision = asyncio.Event()

            async def select(self, request: SpeakerRequest) -> SpeakerDecision:
                decision = await super().select(request)
                if self.calls == 2:
                    self.second_decision.set()
                return decision

        selector = Announcing([speak("scout"), speak("critic"), speak("scout")])
        orch = _orchestrator(built, [selector])

        first = asyncio.create_task(orch.post("r1", "alice", "one"))
        await scout.turn_started.wait()

        second = await orch.accept("r1", "alice", "two")
        superseded = asyncio.create_task(orch.resume("r1", second.transcript[-1].seq))
        # Between a decision and `async with floor` there is no await (confidence 1.0, so
        # no hesitation pause): once this is set, that loop is queued on the floor.
        await selector.second_decision.wait()

        third = await orch.accept("r1", "alice", "three")
        current = asyncio.create_task(orch.resume("r1", third.transcript[-1].seq))
        try:
            scout.release.set()
            await asyncio.wait_for(asyncio.gather(first, superseded, current), timeout=5)
        finally:
            scout.release.set()

        assert critic.prompts == [], (
            "a loop took the floor to answer 'two' after 'three' had been sent"
        )
        final = store.load("r1")
        assert final is not None
        assert [(m.sender_id, m.content) for m in final.transcript] == [
            ("alice", "one"),
            ("alice", "two"),
            ("alice", "three"),
            # Scout's first turn lands after both later messages were recorded; the second
            # is the answer to "three", given by the loop that "three" started.
            ("scout", "scout says TTL"),
            ("scout", "scout says TTL"),
        ]

    @pytest.mark.asyncio
    async def test_stop_cancels_the_turn_that_is_actually_running(self, built: Any) -> None:
        """Stop reaches the agent that holds the floor after an overlap has retired one.

        The registry is keyed by room, so a finishing turn used to evict whichever turn
        was in the slot. The floor now serialises both doors into `_take_turn`, which is
        what makes that unreachable; this pins the outcome the eviction produced.
        """
        store, _resolver, agents = built
        scout, critic = agents["scout"], agents["critic"]
        scout.release = asyncio.Event()
        critic.release = asyncio.Event()

        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")] * 4)])

        first = await orch.accept("r1", "alice", "one")
        cascade_a = asyncio.create_task(orch.resume("r1", first.transcript[-1].seq))
        await scout.turn_started.wait()
        second = await orch.accept("r1", "alice", "two")
        cascade_b = asyncio.create_task(orch.resume("r1", second.transcript[-1].seq))

        # Let scout land, which retires cascade A and hands the floor to cascade B.
        scout.release.set()
        await cascade_a
        await critic.turn_started.wait()

        await orch.interrupt("r1")
        critic.release.set()
        await cascade_b

        final = store.load("r1")
        assert final is not None
        interrupted = [m for m in final.transcript if m.sender_id == "critic"]
        assert interrupted, "critic never took the floor, so the test proves nothing"
        assert interrupted[-1].completed is False, (
            "Stop answered as though it had cancelled a turn that kept generating"
        )

    @pytest.mark.asyncio
    async def test_a_retry_waits_for_the_floor_instead_of_speaking_beside_a_cascade(
        self, built: Any
    ) -> None:
        """`retry` is the other door into `_take_turn`, and it bypassed the floor.

        The loop's own generation guard cannot see a retry, so this was the one route by
        which two agents could hold the floor after the guard was added.

        Killed by: src/uclone_x/room/orchestrator.py :: async with self._floor.setdefault(room_id, asyncio.Lock()):
        Becomes: if True:
        """
        store, _resolver, agents = built
        scout, critic = agents["scout"], agents["critic"]

        critic.fail_with = RuntimeError("boom")
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("critic")])])
        await orch.post("r1", "alice", "go")
        critic.fail_with = None
        failed = store.load("r1")
        assert failed is not None
        assert failed.transcript[-1].error is not None, "the retry has nothing to retry"

        # A cascade holds the floor with scout generating. Driven through `resume` with
        # no fresh `accept`, so critic's failed row stays last and the retry stays valid.
        scout.release = asyncio.Event()
        cascade_orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        cascade_orch._floor = orch._floor  # noqa: SLF001 - one room has one floor
        cascade = asyncio.create_task(cascade_orch.resume("r1", 1))
        await scout.turn_started.wait()

        retrying = asyncio.create_task(orch.retry("r1"))
        try:
            for _ in range(20):
                await asyncio.sleep(0)
            assert not retrying.done(), "a retry took the floor while scout was generating"
            assert len(critic.prompts) == 1, "critic spoke a second time beside the cascade"
        finally:
            # Always, so a failed assertion cannot leave a turn awaiting for the rest of
            # the suite -- which is how the first draft of this test hung the run.
            scout.release.set()
            await asyncio.gather(cascade, retrying, return_exceptions=True)

        assert len(critic.prompts) == 2, "the retry never ran once the floor was free"


class TestTheCeilingSaysThatItStopped:
    """#757 review, finding 3: the cap returned with nothing recorded."""

    @pytest.mark.asyncio
    async def test_reaching_the_ceiling_is_recorded(self, built: Any) -> None:
        _cap(built, 2)
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")] * 5)])

        result = await orch.post("r1", "alice", "go")

        assert result.last_decision is not None
        assert result.last_decision.verdict is SelectionVerdict.SILENCE
        assert "ceiling" in result.last_decision.reasoning.lower()
        assert "2" in result.last_decision.reasoning


class TestThePolicyKnobsCannotStarveAnAddress:
    """#757 review, finding 4: the fan-out silently depended on the selector's window."""

    def test_a_window_too_small_for_the_turn_ceiling_is_refused(self) -> None:
        import pytest as _pytest

        with _pytest.raises(ValueError, match="transcript_window"):
            RoomPolicy(transcript_window=2, max_agent_turns_per_human_message=3)

    def test_the_defaults_satisfy_the_relation(self) -> None:
        policy = RoomPolicy()
        assert policy.transcript_window > policy.max_agent_turns_per_human_message


# --------------------------------------------------------------------------------------
# Membership events share the transcript with speech, and must never be read as speech
# --------------------------------------------------------------------------------------


def _membership(seq: int, sender_id: str, content: str, kind: Any) -> Any:
    from uclone_x.room.models import RoomMessage

    return RoomMessage(seq=seq, sender_id=sender_id, content=content, kind=kind)


class TestMembershipIsNotSpeech:
    """A join or leave line sits in the transcript; three consumers must skip it.

    The transcript gained a second kind of row when the roster became editable. Every
    place that reads it looking for *speech* had been written when speech was the only
    thing in there, so each is a place a membership line can be mistaken for an utterance.
    """

    @pytest.mark.asyncio
    async def test_a_join_line_is_not_rendered_into_a_speakers_span(self, built: Any) -> None:
        """Otherwise the agent is shown `[critic]: critic joined the room` as a remark.

        Killed by: src/uclone_x/room/orchestrator.py :: m.is_utterance and m.sender_id != speaker.id
        Becomes: m.sender_id != speaker.id
        """
        from uclone_x.room.models import RoomMessageKind

        store, _, agents = built
        state = store.load("r1")
        assert state is not None
        store.save(
            state.model_copy(
                update={
                    "transcript": (
                        _membership(1, "critic", "critic joined the room", RoomMessageKind.JOIN),
                    )
                }
            )
        )
        _cap(built, 1)
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])

        await orch.post("r1", "alice", "what do you think?")

        prompt = agents["scout"].prompts[0]
        assert "joined the room" not in prompt
        assert prompt == "[alice]: what do you think?"

    @pytest.mark.asyncio
    async def test_a_membership_line_is_not_an_interjection(self, built: Any) -> None:
        """Somebody joining mid-loop is not somebody speaking, and must not stop the turns.

        `_interjected` asks whether a human has said anything past the baseline seq, and it
        answers that from the row itself: a row nobody was given the floor for was posted
        by a human. A membership row carries no decision either — nobody selects a join —
        so without the `is_utterance` guard the room's own prose about the roster would
        read as a human cutting in, and the remaining turns would be abandoned for a
        message nobody sent.

        Killed by: src/uclone_x/room/orchestrator.py :: m.is_utterance and m.decision is None
        Becomes: m.decision is None
        """
        from uclone_x.room.models import RoomMessageKind

        _cap(built, 2)
        store, _, agents = built
        agents["scout"].release = asyncio.Event()
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])

        post = asyncio.create_task(orch.post("r1", "alice", "go"))
        await asyncio.wait_for(agents["scout"].turn_started.wait(), 1.0)

        # Another agent is seated while the first turn runs.
        mid = store.load("r1")
        assert mid is not None
        store.save(
            mid.model_copy(
                update={
                    "transcript": (
                        *mid.transcript,
                        _membership(
                            len(mid.transcript) + 1,
                            "archivist",
                            "archivist joined the room",
                            RoomMessageKind.JOIN,
                        ),
                    ),
                    "participants": (*mid.participants, agent_participant("archivist")),
                }
            )
        )
        agents["scout"].release.set()
        result = await asyncio.wait_for(post, 2.0)

        spoken = [m.sender_id for m in result.transcript if m.is_utterance]
        assert spoken == ["alice", "scout", "critic"], (
            "a join is not an interjection, so the second turn must still have run"
        )

    @pytest.mark.asyncio
    async def test_membership_row_in_window_does_not_starve_an_address(self, built: Any) -> None:
        """A membership row must not displace an address from the selector's window.

        `RoomPolicy._window_must_outlast_the_turn_ceiling` accepts `transcript_window=4`
        and `max_agent_turns_per_human_message=3` on the promise that the turns themselves
        cannot push the human utterance out of the window. But if membership rows count
        towards `transcript_window`, an intervening JOIN row reduces the effective
        utterance window to 3 rows, causing scout's turn to push the human utterance out
        of view and dropping critic's outstanding address unannounced (#806).

        Killed by: src/uclone_x/room/orchestrator.py :: windowed = self.window_over_utterances(state.transcript, state.policy.transcript_window)
        Becomes: windowed = state.transcript[-state.policy.transcript_window :]
        """
        from uclone_x.room.models import RoomMessageKind
        from uclone_x.room.selectors import MentionSelector

        store, _, agents = built
        state = store.load("r1")
        assert state is not None
        store.save(
            state.model_copy(
                update={
                    "policy": RoomPolicy(transcript_window=3, max_agent_turns_per_human_message=2)
                }
            )
        )
        orch = _orchestrator(built, [MentionSelector()])

        # Alice addresses both scout and critic. While scout generates its turn, two JOIN
        # events are written into the transcript.
        # Under raw slicing state.transcript[-3:], the transcript seen before critic speaks
        # is [JOIN, JOIN, scout], dropping alice and starving critic.
        # Under window_over_utterances(transcript, 3), the window contains [alice, JOIN, JOIN, scout],
        # preserving alice's address so critic speaks.
        agents["scout"].release = asyncio.Event()
        post = asyncio.create_task(orch.post("r1", "alice", "@scout @critic compare"))
        await asyncio.wait_for(agents["scout"].turn_started.wait(), 1.0)

        mid = store.load("r1")
        assert mid is not None
        store.save(
            mid.model_copy(
                update={
                    "transcript": (
                        *mid.transcript,
                        _membership(
                            len(mid.transcript) + 1,
                            "archivist",
                            "archivist joined the room",
                            RoomMessageKind.JOIN,
                        ),
                        _membership(
                            len(mid.transcript) + 2,
                            "observer",
                            "observer joined the room",
                            RoomMessageKind.JOIN,
                        ),
                    ),
                    "participants": (
                        *mid.participants,
                        agent_participant("archivist"),
                        agent_participant("observer"),
                    ),
                }
            )
        )
        agents["scout"].release.set()
        result = await asyncio.wait_for(post, 2.0)

        spoken = [m.sender_id for m in result.transcript if m.is_utterance]
        assert spoken == ["alice", "scout", "critic"], (
            "critic was starved because the JOIN row pushed alice's address out of the "
            "selector window"
        )

    @pytest.mark.asyncio
    async def test_window_over_utterances_preserves_interleaved_membership(
        self, built: Any
    ) -> None:
        """The window preserves interleaved membership lines while counting utterances.

        Killed by: src/uclone_x/room/orchestrator.py :: utterances_seen == window
        Becomes: len(transcript) - idx == window
        """
        from uclone_x.room.models import RoomMessageKind
        from uclone_x.room.orchestrator import RoomOrchestrator

        store, _, _ = built
        state = store.load("r1")
        assert state is not None
        t = (
            _membership(1, "alice", "hello 1", RoomMessageKind.UTTERANCE),
            _membership(2, "scout", "reply 1", RoomMessageKind.UTTERANCE),
            _membership(3, "critic", "critic joined", RoomMessageKind.JOIN),
            _membership(4, "alice", "hello 2", RoomMessageKind.UTTERANCE),
            _membership(5, "scout", "reply 2", RoomMessageKind.UTTERANCE),
        )
        # Window of 2 utterances must return from seq 4 onwards (hello 2 and reply 2).
        win2 = RoomOrchestrator.window_over_utterances(t, 2)
        assert [m.seq for m in win2] == [4, 5]

        # Window of 3 utterances must include seq 2 (reply 1), seq 3 (JOIN), seq 4 (hello 2), seq 5 (reply 2).
        win3 = RoomOrchestrator.window_over_utterances(t, 3)
        assert [m.seq for m in win3] == [2, 3, 4, 5]


# --------------------------------------------------------------------------------------
# An interjection is a property of the message, not of who is still in the room
# --------------------------------------------------------------------------------------


class TestAPosterWhoLeftStillInterjected:
    @pytest.mark.asyncio
    async def test_a_human_who_leaves_after_speaking_still_stops_the_loop(self, built: Any) -> None:
        """Leaving the room must not retract what was already said.

        `_interjected` used to derive the human set from the *live* roster, so a poster who
        left during selection made their own interjection invisible and the loop carried on
        answering the message they had superseded. Whether a row was a human's is a fact
        about the row — nobody was given the floor for it — and reading it off a roster that
        has moved since is the one way to get it wrong.

        Killed by: src/uclone_x/room/orchestrator.py :: m.decision is None
        Becomes: m.sender_id in {p.id for p in state.participants if p.kind is ParticipantKind.HUMAN}
        """
        _cap(built, 2)
        store, _, agents = built
        agents["scout"].release = asyncio.Event()
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])

        post = asyncio.create_task(orch.post("r1", "alice", "go"))
        await asyncio.wait_for(agents["scout"].turn_started.wait(), 1.0)

        # Alice cuts in and then leaves the room, both while scout is still generating.
        mid = store.load("r1")
        assert mid is not None
        interjected = orch.append_human(mid, "alice", "WAIT - never mind")
        store.save(
            interjected.model_copy(
                update={
                    "participants": tuple(p for p in interjected.participants if p.id != "alice"),
                }
            )
        )
        agents["scout"].release.set()
        result = await asyncio.wait_for(post, 2.0)

        spoken = [m.sender_id for m in result.transcript if m.is_utterance]
        # Alice's interjection lands while scout is still generating, so it precedes
        # scout's own utterance in the transcript; what matters is that critic is absent.
        assert spoken == ["alice", "alice", "scout"], (
            "critic answered a message alice had already superseded, because alice had "
            "left the roster the check read"
        )


# --------------------------------------------------------------------------------------
# Room Interruption (Issue #759)
# --------------------------------------------------------------------------------------


class TestRoomInterruption:
    @pytest.mark.asyncio
    async def test_interrupt_cancels_running_turn_and_stops_loop(self, built: Any) -> None:
        """Signaling interrupt cancels in-flight turn, records completed=False, and halts.

        Killed by: src/uclone_x/room/orchestrator.py :: completed = False
        Becomes: completed = True
        """
        _, _, agents = built
        agents["scout"].release = asyncio.Event()
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])

        post_task = asyncio.create_task(orch.post("r1", "alice", "please discuss"))
        await asyncio.wait_for(agents["scout"].turn_started.wait(), 1.0)

        # Signal interrupt while scout is executing its turn
        await orch.interrupt("r1", reason="User stopped the conversation")

        # Allow post to return
        state = await asyncio.wait_for(post_task, 2.0)

        # Scout's turn was interrupted: completed is False, error is set
        assert len(state.transcript) == 2
        interrupted_msg = state.transcript[1]
        assert interrupted_msg.sender_id == "scout"
        assert interrupted_msg.completed is False
        assert interrupted_msg.error == "Turn was interrupted"

        # Critic was never given the floor because the loop halted
        spoken = [m.sender_id for m in state.transcript if m.is_utterance]
        assert spoken == ["alice", "scout"]

    @pytest.mark.asyncio
    async def test_interrupt_publishes_interrupt_event_on_bus(self, built: Any) -> None:
        """Signaling interrupt publishes EventType.INTERRUPT event on the room's bus topic.

        Killed by: src/uclone_x/room/orchestrator.py :: type=EventType.INTERRUPT,
        Becomes: type=EventType.AGENT_REPLY,
        """
        from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
        from uclone_x.room.orchestrator import RoomOrchestrator

        store, resolver, agents = built
        agents["scout"].release = asyncio.Event()

        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )

            post_task = asyncio.create_task(orch.post("r1", "alice", "hi"))
            await asyncio.wait_for(agents["scout"].turn_started.wait(), 1.0)

            await orch.interrupt("r1", reason="Stop now")
            await asyncio.wait_for(post_task, 2.0)

            events: list[AgentEvent] = []
            while True:
                try:
                    events.append(await asyncio.wait_for(sub.get(), 0.3))
                except TimeoutError:
                    break

        interrupt_events = [e for e in events if e.type is EventType.INTERRUPT]
        assert len(interrupt_events) == 1
        assert interrupt_events[0].payload["room_id"] == "r1"
        assert interrupt_events[0].payload["reason"] == "Stop now"


class TestAgentTurnStartAnnouncement:
    """Announce that a selected agent has begun generating before execute_turn awaits."""

    @pytest.mark.asyncio
    async def test_agent_turn_start_is_announced_before_turn_completes(self, built: Any) -> None:
        """
        Killed by: src/uclone_x/room/orchestrator.py :: "status": "generating"
        Becomes: "status": "pending"
        """
        from uclone_x.engine.event_bus import EventBus, EventType
        from uclone_x.room.orchestrator import RoomOrchestrator

        store, resolver, agents = built
        agents["scout"].release = asyncio.Event()

        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )

            post_task = asyncio.create_task(orch.post("r1", "alice", "hello"))
            await asyncio.wait_for(agents["scout"].turn_started.wait(), 1.0)

            event = await asyncio.wait_for(sub.get(), 1.0)
            assert event.type is EventType.AGENT_REPLY
            assert event.payload["room_id"] == "r1"
            assert event.payload["agent_id"] == "scout"
            assert event.payload["turn_id"]
            assert "seq" not in event.payload
            assert event.payload["status"] == "generating"

            agents["scout"].release.set()
            await asyncio.wait_for(post_task, 2.0)

            landed = await asyncio.wait_for(sub.get(), 1.0)
            assert landed.type is EventType.AGENT_REPLY
            assert landed.payload["room_id"] == "r1"
            assert landed.payload["agent_id"] == "scout"
            assert landed.payload["seq"] == 2
            assert landed.payload["content"] == "scout says TTL"
            assert landed.payload["completed"] is True


class TestHumanTypingAnnouncement:
    """Note human typing on the presentation channel."""

    @pytest.mark.asyncio
    async def test_note_human_activity_announces_typing_on_bus(self, built: Any) -> None:
        """
        Killed by: src/uclone_x/room/orchestrator.py :: "status": "typing"
        Becomes: "status": "idle"
        """
        from uclone_x.engine.event_bus import EventBus, EventType
        from uclone_x.room.orchestrator import RoomOrchestrator

        store, resolver, _ = built
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s")],
                resolver=resolver,
                bus=bus,
            )

            await orch.note_human_activity("r1", sender_id="alice")

            event = await asyncio.wait_for(sub.get(), 1.0)
            assert event.type is EventType.USER_INPUT
            assert event.payload["room_id"] == "r1"
            assert event.payload["sender_id"] == "alice"
            assert event.payload["status"] == "typing"

    @pytest.mark.asyncio
    async def test_note_human_activity_default_sender_is_human(self, built: Any) -> None:
        """
        Killed by: src/uclone_x/room/orchestrator.py :: async def note_human_activity(self, room_id: str, sender_id: str = "human") -> None:
        Becomes: async def note_human_activity(self, room_id: str, sender_id: str = "default_user") -> None:
        """
        from uclone_x.engine.event_bus import EventBus, EventType
        from uclone_x.room.orchestrator import RoomOrchestrator

        store, resolver, _ = built
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s")],
                resolver=resolver,
                bus=bus,
            )

            await orch.note_human_activity("r1")

            event = await asyncio.wait_for(sub.get(), 1.0)
            assert event.type is EventType.USER_INPUT
            assert event.payload["sender_id"] == "human"
            assert event.payload["status"] == "typing"


class TestHesitationPause:
    """Hesitation pause before low-confidence turns, yielding the floor on activity."""

    @pytest.mark.asyncio
    async def test_hesitation_pause_proceeds_when_no_human_activity(self, built: Any) -> None:
        """
        Killed by: src/uclone_x/room/orchestrator.py :: pause = max(0.0, state.policy.hesitation_seconds * (1.0 - decision.confidence))
        Becomes: pause = 0.0
        """
        store, _, _ = built
        state = store.load("r1")
        assert state is not None
        store.save(
            state.model_copy(
                update={"policy": state.policy.model_copy(update={"hesitation_seconds": 0.05})}
            )
        )
        orch = _orchestrator(
            built,
            [ScriptedSelector("s", [speak("scout", confidence=0.0)])],
        )

        t0 = time.monotonic()
        result = await orch.post("r1", "alice", "go")
        elapsed = time.monotonic() - t0

        assert elapsed >= 0.03
        assert [m.sender_id for m in result.transcript] == ["alice", "scout"]

    @pytest.mark.asyncio
    async def test_hesitation_pause_yields_when_human_activity_occurs(self, built: Any) -> None:
        """
        Killed by: src/uclone_x/room/orchestrator.py :: if state.turn_state.last_activity_ts > pre_wait_activity:
        Becomes: if False and state.turn_state.last_activity_ts > pre_wait_activity:
        """
        store, _, _ = built
        state = store.load("r1")
        assert state is not None
        store.save(
            state.model_copy(
                update={"policy": state.policy.model_copy(update={"hesitation_seconds": 0.5})}
            )
        )
        orch = _orchestrator(
            built,
            [ScriptedSelector("s", [speak("scout", confidence=0.0)])],
        )

        post_task = asyncio.create_task(orch.post("r1", "alice", "hello"))
        await asyncio.sleep(0.05)
        await orch.note_human_activity("r1", sender_id="alice")

        final_state = await asyncio.wait_for(post_task, 1.0)
        assert [m.sender_id for m in final_state.transcript] == ["alice"]

    @pytest.mark.asyncio
    async def test_hesitation_pause_yields_when_interjection_arrives(self, built: Any) -> None:
        """
        Killed by: src/uclone_x/room/orchestrator.py :: if room_id in self._interrupted_rooms:  # hesitation interrupted
        Becomes: if False and room_id in self._interrupted_rooms:  # hesitation interrupted
        """
        store, _, _ = built
        state = store.load("r1")
        assert state is not None
        store.save(
            state.model_copy(
                update={"policy": state.policy.model_copy(update={"hesitation_seconds": 0.5})}
            )
        )
        orch = _orchestrator(
            built,
            [ScriptedSelector("s", [speak("scout", confidence=0.0)])],
        )

        post_task = asyncio.create_task(orch.post("r1", "alice", "hello"))
        await asyncio.sleep(0.05)
        await orch.interrupt("r1", reason="interrupted during pause")

        final_state = await asyncio.wait_for(post_task, 1.0)
        assert [m.sender_id for m in final_state.transcript] == ["alice"]


# --------------------------------------------------------------------------------------
# A real agent's failures, through the room (#969): a refusal a retry would meet again is
# told apart from a failure a retry can get past, and a retry does not repeat the prompt.
# --------------------------------------------------------------------------------------


def _real_agent(participant: Participant, llm: Any, budget: Any = None) -> Any:
    from uclone_x.agent import BaseAgent
    from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig

    return BaseAgent(
        config=AgentConfig(
            agent_id=participant.id,
            name=participant.id,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=llm,
        context=AgentContext(session_id=participant.session_id, agent_id=participant.id),
        budget=budget,
    )


def _scripted_connector(fail_on: set[int], responses: list[str]) -> Any:
    """A mock connector that raises a provider error on the calls in `fail_on` (1-based)
    and records every request it was sent."""
    from uclone_x.errors import LLMProviderError
    from uclone_x.llm import MockLLMConnector
    from uclone_x.llm.models import LLMRequest, ModelResponse

    class Scripted(MockLLMConnector):
        def __init__(self) -> None:
            super().__init__(responses=responses)
            self.requests: list[LLMRequest] = []

        async def generate(self, request: LLMRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) in fail_on:
                raise LLMProviderError("provider unavailable (scripted)")
            return await super().generate(request)

    return Scripted()


def _spent_budget(session_id: str) -> Any:
    from uclone_x.llm.budget import TokenBudgetManager

    budget = TokenBudgetManager()
    budget.configure_session(session_id, max_tokens=0)
    return budget


class TestARealAgentsFailuresInTheRoom:
    @pytest.mark.asyncio
    async def test_a_budget_refusal_lands_as_a_refusal_and_an_outage_does_not(
        self, built: Any
    ) -> None:
        """The row says which failures a retry cannot get past (#969).

        A spent budget landed as an ordinary failed row, so the head offered Retry on it:
        the retry refunded the turn slot, was refused by the same ceiling, and appended
        another failed row. What decides it is the turn's `stop_reason`, never the text of
        `error`.

        Killed by: src/uclone_x/room/orchestrator.py :: refusal = turn_refusal(result.stop_reason)
        Becomes: refusal = None
        """
        from uclone_x.room.models import RoomMessage, RoomTurnRefusal

        store, _, agents = built
        _cap(built, 1)
        agents["scout"] = _real_agent(
            SCOUT, _scripted_connector(set(), ["unused"]), _spent_budget(SCOUT.session_id)
        )
        agents["critic"] = _real_agent(CRITIC, _scripted_connector({1}, []))

        refused = (
            await _orchestrator(built, [ScriptedSelector("s", [speak("scout")])]).post(
                "r1", "alice", "discuss"
            )
        ).transcript[-1]
        assert refused.error is not None and "limit exceeded" in refused.error.lower()
        assert refused.refusal is RoomTurnRefusal.BUDGET_EXCEEDED

        failed = (
            await _orchestrator(built, [ScriptedSelector("s", [speak("critic")])]).post(
                "r1", "alice", "and you?"
            )
        ).transcript[-1]
        assert failed.error == "provider unavailable (scripted)"
        assert failed.refusal is None

        # A room written before the field existed still loads, as a failure Retry can try.
        stored = store.load("r1")
        assert stored is not None
        old = stored.transcript[-1].model_dump(mode="json")
        del old["refusal"]
        assert RoomMessage.model_validate_json(json.dumps(old)).refusal is None

    @pytest.mark.asyncio
    async def test_a_model_without_tools_lands_as_a_refusal_in_plain_words(
        self, built: Any
    ) -> None:
        """The room row states the refusal and its remedy, and none of the plumbing.

        Killed by: src/uclone_x/room/models.py :: return RoomTurnRefusal.MODEL_WITHOUT_TOOLS
        Becomes: return None
        """
        from uclone_x.errors import ModelLacksToolSupportError
        from uclone_x.llm import MockLLMConnector
        from uclone_x.llm.models import LLMRequest, ModelResponse
        from uclone_x.room.models import RoomTurnRefusal

        class NoTools(MockLLMConnector):
            async def generate(self, request: LLMRequest) -> ModelResponse:
                raise ModelLacksToolSupportError("deepseek-r1:14b")

        _, _, agents = built
        _cap(built, 1)
        agents["scout"] = _real_agent(SCOUT, NoTools(responses=[]))

        row = (
            await _orchestrator(built, [ScriptedSelector("s", [speak("scout")])]).post(
                "r1", "alice", "discuss"
            )
        ).transcript[-1]

        assert row.refusal is RoomTurnRefusal.MODEL_WITHOUT_TOOLS
        assert row.error is not None
        assert row.error.startswith("The model deepseek-r1:14b can't use tools")
        assert "qwen3:8b" in row.error
        for internal in ("Traceback", "status 400", "{", "LLMProviderError"):
            assert internal not in row.error, internal

    @pytest.mark.asyncio
    async def test_the_landed_event_carries_the_refusal(self, built: Any) -> None:
        """A head listening on the topic learns the refusal where it learns the failure.

        Killed by: src/uclone_x/room/orchestrator.py :: payload["refusal"] = message.refusal.value
        Becomes: pass
        """
        from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
        from uclone_x.room.orchestrator import RoomOrchestrator

        store, resolver, agents = built
        _cap(built, 1)
        agents["scout"] = _real_agent(
            SCOUT, _scripted_connector(set(), ["unused"]), _spent_budget(SCOUT.session_id)
        )
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )
            await orch.post("r1", "alice", "discuss")
            events: list[AgentEvent] = []
            while True:
                try:
                    events.append(await asyncio.wait_for(sub.get(), 0.3))
                except TimeoutError:
                    break

        landed = [
            e.payload
            for e in events
            if e.type is EventType.AGENT_REPLY and e.payload.get("status") == "final"
        ]
        assert len(landed) == 1
        assert landed[0]["refusal"] == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_a_retry_puts_the_prompt_in_front_of_the_model_once(self, built: Any) -> None:
        """Retry re-runs the failed turn; it does not ask the same thing twice (#969).

        The failed turn left its prompt in the agent's history, and the retry renders the
        same unseen span and sends it again, so the model was shown two copies.

        No `Killed by:` declaration, because two guards now hold this and neither alone is
        what the test sees: the room rolls the failed turn out of the seat's session
        (#1423), and the agent does not re-append an identical unanswered prompt (#969).
        Each is killed where the other cannot stand in -- the rollback by
        `test_room_turn_transaction.py`, whose failed turn ends on a tool result the #969
        guard does not recognise, and the guard by the chat route's retry in
        `test_ui_app.py`, which has no room to roll anything back.
        """
        from uclone_x.llm.models import MessageRole

        _, _, agents = built
        _cap(built, 1)
        connector = _scripted_connector({1}, ["scout says TTL"])
        agents["scout"] = _real_agent(SCOUT, connector)
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])

        failed = await orch.post("r1", "alice", "discuss")
        assert failed.transcript[-1].error is not None
        retried = await orch.retry("r1")

        assert retried.transcript[-1].content == "scout says TTL"
        prompts = [m for m in connector.requests[1].messages if m.role is MessageRole.USER]
        assert len(prompts) == 1, [m.content for m in prompts]


# --------------------------------------------------------------------------------------
# A room turn's tools reach the bus and the room's record (#1353, #1354)
# --------------------------------------------------------------------------------------


def _execution(
    name: str,
    call_id: str,
    *,
    output: Any = None,
    writes_files: bool = False,
    spawns_subagents: bool = False,
    status: Any = None,
    error: str | None = None,
) -> Any:
    from uclone_x.agent.models import ToolExecutionRecord
    from uclone_x.tools.models import ToolResultStatus

    return ToolExecutionRecord(
        tool_name=name,
        arguments={"path": "notes.md"} if name == "file_write" else {},
        output=output,
        status=status if status is not None else ToolResultStatus.SUCCESS,
        error=error,
        duration_ms=12.5,
        tool_call_id=call_id,
        writes_files=writes_files,
        spawns_subagents=spawns_subagents,
    )


async def _drain(sub: Any) -> list[Any]:
    events: list[Any] = []
    while True:
        try:
            events.append(await asyncio.wait_for(sub.get(), 0.3))
        except TimeoutError:
            return events


class TestARoomTurnsToolsAreRecorded:
    """A seat's tools reached neither the bus nor any record the dock could read (#1353).

    `_take_turn` read `content`, `provenance` and `error` off the `TurnResult` and dropped
    `tool_executions`, so the Activity surface had nothing to show for a room turn, and the
    Docs surface could not be scoped because nothing recorded which files a room wrote.
    """

    @pytest.mark.asyncio
    async def test_each_tool_call_is_published_with_the_room_and_the_seats_session(
        self, built: Any
    ) -> None:
        """One `TOOL_CALL` and one `TOOL_RESULT` per call, naming the room and the seat.

        Killed by: src/uclone_x/room/orchestrator.py :: await self._publish_tools(saved, speaker, message.seq, uses)
        Becomes: pass
        """
        from uclone_x.engine.event_bus import EventBus, EventType
        from uclone_x.room.orchestrator import RoomOrchestrator

        store, resolver, agents = built
        agents["scout"].tool_executions = (
            _execution("file_read", "call_1", output={"path": "a.md", "content": "x"}),
            _execution("file_write", "call_2", output={"path": "notes.md"}, writes_files=True),
        )
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1.tool"})
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )
            await orch.post("r1", "alice", "write it down")
            events = await _drain(sub)

        calls = [e for e in events if e.type is EventType.TOOL_CALL]
        results = [e for e in events if e.type is EventType.TOOL_RESULT]
        assert [e.payload["tool_call_id"] for e in calls] == ["call_1", "call_2"]
        assert [e.payload["name"] for e in calls] == ["file_read", "file_write"]
        for event in (*calls, *results):
            assert event.topic == "room.r1.tool"
            assert event.payload["room_id"] == "r1"
            assert event.payload["participant_id"] == "scout"
            assert event.payload["session_id"] == SCOUT.session_id
        assert [e.payload["written_path"] for e in results] == [None, "notes.md"]
        # The row the turn landed as, so a head can place the calls under the reply.
        assert {e.payload["seq"] for e in (*calls, *results)} == {2}

    @pytest.mark.asyncio
    async def test_tool_events_stay_off_the_rooms_own_topic(self, built: Any) -> None:
        """The head folds `room.{id}` into the transcript and rejects unknown types there.

        Killed by: src/uclone_x/room/orchestrator.py :: topic = f"room.{state.room_id}.tool"
        Becomes: topic = f"room.{state.room_id}"
        """
        from uclone_x.engine.event_bus import EventBus, EventType
        from uclone_x.room.orchestrator import RoomOrchestrator

        store, resolver, agents = built
        agents["scout"].tool_executions = (_execution("file_read", "call_1"),)
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1"})
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )
            await orch.post("r1", "alice", "look")
            events = await _drain(sub)

        assert {e.type for e in events} == {EventType.AGENT_REPLY}

    @pytest.mark.asyncio
    async def test_a_turns_tool_events_precede_its_final_reply(self, built: Any) -> None:
        """A head sees the calls ahead of the answer they fed (design §3.6, Rev 31).

        Killed by: src/uclone_x/room/orchestrator.py :: await self._publish_tools(saved, speaker, message.seq, uses)
        Becomes: asyncio.get_running_loop().call_later(0.05, lambda: asyncio.ensure_future(self._publish_tools(saved, speaker, message.seq, uses)))
        """
        from uclone_x.engine.event_bus import EventBus, EventType
        from uclone_x.room.orchestrator import RoomOrchestrator

        store, resolver, agents = built
        agents["scout"].tool_executions = (_execution("file_read", "call_1"),)
        async with EventBus() as bus:
            sub = bus.subscribe({"room.r1", "room.r1.tool"})
            orch = RoomOrchestrator(
                store=store,
                selectors=[ScriptedSelector("s", [speak("scout")])],
                resolver=resolver,
                bus=bus,
            )
            await orch.post("r1", "alice", "look")
            events = await _drain(sub)

        kinds = [
            e.type
            for e in events
            if e.type in (EventType.TOOL_CALL, EventType.TOOL_RESULT)
            or (e.type is EventType.AGENT_REPLY and e.payload.get("status") == "final")
        ]
        assert kinds == [EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.AGENT_REPLY]

    @pytest.mark.asyncio
    async def test_a_written_path_is_recorded_on_the_room_and_survives_a_reload(
        self, built: Any, tmp_path: Any
    ) -> None:
        """P8: which files a room wrote is Core state, persisted with the room (#1354).

        A reading tool that names a path is not a write. The flag is the tool's own
        declaration (#1167), never the shape of its output.

        Killed by: src/uclone_x/room/orchestrator.py :: if execution.writes_files and succeeded:
        Becomes: if succeeded:
        """
        from uclone_x.room.store import RoomStore

        _, _, agents = built
        agents["scout"].tool_executions = (
            _execution("file_read", "call_1", output={"path": "a.md", "content": "x"}),
            _execution("file_write", "call_2", output={"path": "notes.md"}, writes_files=True),
        )
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        await orch.post("r1", "alice", "write it down")

        reloaded = RoomStore(tmp_path).load("r1")
        assert reloaded is not None
        assert [(f.path, f.participant_id, f.tool_name) for f in reloaded.written_files] == [
            ("notes.md", "scout", "file_write")
        ]
        uses = reloaded.tool_uses
        assert [(u.tool_name, u.tool_call_id, u.written_path) for u in uses] == [
            ("file_read", "call_1", None),
            ("file_write", "call_2", "notes.md"),
        ]
        # Joined to the row by the turn's own id, not by a row number a rewind reuses.
        reply = reloaded.transcript[-1]
        assert reply.turn_id is not None
        assert {u.turn_id for u in uses} == {reply.turn_id}

    @pytest.mark.asyncio
    async def test_a_failed_write_records_no_file_and_counts_a_possible_one(
        self, built: Any
    ) -> None:
        """A failed write lists no file -- and may have written before failing (#1366).

        Killed by: src/uclone_x/room/orchestrator.py :: succeeded = execution.status is ToolResultStatus.SUCCESS
        Becomes: succeeded = True
        """
        from uclone_x.tools.models import ToolResultStatus

        _, _, agents = built
        agents["scout"].tool_executions = (
            _execution(
                "file_write",
                "call_1",
                output={"path": "notes.md"},
                writes_files=True,
                status=ToolResultStatus.ERROR,
                error="disk full",
            ),
        )
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        saved = await orch.post("r1", "alice", "write it down")

        assert saved.written_files == ()
        assert saved.tool_uses[0].error == "disk full"
        assert saved.tool_uses[0].wrote_unnamed is True
        assert saved.file_record.unattributed_writes == 1

    @pytest.mark.asyncio
    async def test_every_file_a_peer_call_names_is_in_the_rooms_file_list(self, built: Any) -> None:
        """A peer asked through `a2a_call` reports its files as a `paths` list (#1558).

        Each one is a file the conversation wrote, attributed to the seat that asked, and
        a peer that named everything it wrote is not counted as a possible unnamed write.

        Killed by: src/uclone_x/room/orchestrator.py :: candidates.extend(cast(Sequence[object], listed))
        Becomes: pass
        """
        _, _, agents = built
        agents["scout"].tool_executions = (
            _execution(
                "a2a_call",
                "call_1",
                output={
                    "agent": "artist",
                    "response": "Drew both.",
                    "paths": ["images/hero.png", "images/villain.png"],
                    "unnamed_writes": False,
                },
                writes_files=True,
            ),
        )
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        saved = await orch.post("r1", "alice", "draw them")

        assert [(f.path, f.participant_id, f.tool_name) for f in saved.written_files] == [
            ("images/hero.png", "scout", "a2a_call"),
            ("images/villain.png", "scout", "a2a_call"),
        ]
        use = saved.tool_uses[0]
        assert use.written_paths == ("images/hero.png", "images/villain.png")
        assert use.written_path == "images/hero.png"
        assert use.wrote_unnamed is False
        assert saved.file_record.unattributed_writes == 0

    @pytest.mark.asyncio
    async def test_a_peer_that_may_have_written_unnamed_files_is_counted(self, built: Any) -> None:
        """A peer call that succeeded but says it may have written more than it named is
        counted the way a helper is, so the room does not claim a complete list (#1558).

        Killed by: src/uclone_x/room/orchestrator.py :: or (execution.writes_files and mapping.get("unnamed_writes") is True)
        Becomes: or False
        """
        _, _, agents = built
        agents["scout"].tool_executions = (
            _execution(
                "a2a_call",
                "call_1",
                output={"agent": "artist", "paths": ["images/hero.png"], "unnamed_writes": True},
                writes_files=True,
            ),
        )
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        saved = await orch.post("r1", "alice", "draw them")

        assert [f.path for f in saved.written_files] == ["images/hero.png"]
        assert saved.tool_uses[0].wrote_unnamed is True
        assert saved.file_record.unattributed_writes == 1

    @pytest.mark.asyncio
    async def test_a_failed_peer_call_leaves_a_trace(self, built: Any) -> None:
        """A peer that failed may have written before it stopped; the room counts it (#1558).

        Killed by: src/uclone_x/room/orchestrator.py :: wrote_unnamed=(written_path is None and execution.writes_files)
        Becomes: wrote_unnamed=(False)
        """
        from uclone_x.tools.models import ToolResultStatus

        _, _, agents = built
        agents["scout"].tool_executions = (
            _execution(
                "a2a_call",
                "call_1",
                output=None,
                writes_files=True,
                status=ToolResultStatus.ERROR,
                error="'artist' could not do the task: it could not finish.",
            ),
        )
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        saved = await orch.post("r1", "alice", "draw them")

        assert saved.written_files == ()
        assert saved.tool_uses[0].wrote_unnamed is True
        assert saved.file_record.unattributed_writes == 1

    @pytest.mark.asyncio
    async def test_a_write_that_names_no_path_is_counted_rather_than_dropped(
        self, built: Any
    ) -> None:
        """A shell can put bytes on the host without saying where (P6: say so).

        Killed by: src/uclone_x/room/orchestrator.py :: wrote_unnamed=(written_path is None and execution.writes_files)
        Becomes: wrote_unnamed=(False)
        """
        _, _, agents = built
        agents["scout"].tool_executions = (
            _execution("bash_run", "call_1", output={"stdout": ""}, writes_files=True),
        )
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        saved = await orch.post("r1", "alice", "run it")

        assert saved.written_files == ()
        assert saved.tool_uses[0].wrote_unnamed is True

    @pytest.mark.asyncio
    async def test_a_failed_helper_is_counted_and_a_call_that_never_ran_is_not(
        self, built: Any
    ) -> None:
        """A helper that failed may have written first; a call that reached no tool did not.

        Killed by: src/uclone_x/room/orchestrator.py :: or execution.spawns_subagents,
        Becomes: or (execution.spawns_subagents and succeeded),
        """
        from uclone_x.tools.models import ToolResultStatus

        _, _, agents = built
        agents["scout"].tool_executions = (
            _execution(
                "delegate_subagent",
                "call_d",
                spawns_subagents=True,
                status=ToolResultStatus.ERROR,
                error="helper stopped",
            ),
            _execution("file_write", "call_x", status=ToolResultStatus.ERROR, error="Unknown tool"),
        )
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        saved = await orch.post("r1", "alice", "go")

        by_call = {u.tool_call_id: u for u in saved.tool_uses}
        assert by_call["call_d"].wrote_unnamed is True
        assert by_call["call_x"].wrote_unnamed is False
        assert saved.file_record.unattributed_writes == 1

    @pytest.mark.asyncio
    async def test_a_turn_that_raised_says_its_tools_were_not_recorded(self, built: Any) -> None:
        """No `TurnResult`, no executions: an empty list there would claim no tools ran.

        Killed by: src/uclone_x/room/orchestrator.py :: tools_recorded = result.tool_executions_complete
        Becomes: tools_recorded = False
        """
        _, _, agents = built
        agents["critic"].fail_with = RuntimeError("boom")
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout"), speak("critic")])])
        saved = await orch.post("r1", "alice", "go")

        by_sender = {m.sender_id: m for m in saved.transcript if m.sender_id != "alice"}
        assert by_sender["scout"].tools_recorded is True
        assert by_sender["critic"].tools_recorded is False

    @pytest.mark.asyncio
    async def test_a_turn_that_failed_mid_step_keeps_its_tools_as_a_lower_bound(
        self, built: Any
    ) -> None:
        """An errored turn's recorded calls are kept, and the turn still counts as partial.

        Killed by: src/uclone_x/room/orchestrator.py :: tools_recorded = result.tool_executions_complete
        Becomes: tools_recorded = True
        """
        _, _, agents = built
        agents["scout"].result_error = "provider went away"
        agents["scout"].tool_executions = (
            _execution("file_write", "call_1", output={"path": "a.md"}, writes_files=True),
        )
        agents["scout"].tool_executions_complete = False
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        saved = await orch.post("r1", "alice", "write it")

        row = next(m for m in saved.transcript if m.sender_id == "scout")
        assert [(u.tool_name, u.turn_id) for u in saved.tool_uses] == [("file_write", row.turn_id)]
        assert [f.path for f in saved.written_files] == ["a.md"]
        assert row.tools_recorded is False
        assert saved.file_record.unrecorded_turns == 1
        assert saved.file_record.turns_started == saved.file_record.turns_landed == 1

    @pytest.mark.asyncio
    async def test_a_spawned_subagent_is_recorded_from_the_declaring_tool(self, built: Any) -> None:
        """P4: the topology can only draw a sub-agent whose parent recorded it.

        Killed by: src/uclone_x/room/orchestrator.py :: if execution.spawns_subagents and succeeded and isinstance(child, str) and child:
        Becomes: if False:
        """
        _, _, agents = built
        agents["scout"].tool_executions = (
            _execution(
                "delegate_subagent",
                "call_1",
                output={"subagent_id": "scout_sub_1", "response": "done"},
                spawns_subagents=True,
            ),
        )
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        saved = await orch.post("r1", "alice", "delegate")

        assert saved.tool_uses[0].subagent_id == "scout_sub_1"


class TestATurnIsCountedBeforeItRuns:
    """A lost landing save left no trace, so a room read as "nothing written" (#1366).

    The turn is now counted in a save of its own before it runs. If that save fails, the
    turn is refused with its cause rather than run unrecorded.
    """

    @pytest.mark.asyncio
    async def test_the_start_is_saved_before_the_agent_runs(self, built: Any) -> None:
        """Killed by: src/uclone_x/room/orchestrator.py :: update={"turns_started": record.turns_started + 1}
        Becomes: update={"turns_started": record.turns_started + 0}
        Killed by: src/uclone_x/room/orchestrator.py :: self._unlanded[room_id] = turn_id
        Becomes: pass
        """
        store, _, agents = built
        seen: list[tuple[int, bool]] = []
        original = agents["scout"].execute_turn

        async def _observe(prompt: str, **kwargs: Any) -> Any:
            state = store.load("r1")
            assert state is not None
            seen.append((state.file_record.turns_started, orch.turn_unlanded("r1")))
            return await original(prompt, **kwargs)

        agents["scout"].execute_turn = _observe
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        saved = await orch.post("r1", "alice", "go")

        assert seen == [(1, True)]
        assert saved.file_record.turns_landed == 1
        assert orch.turn_unlanded("r1") is False

    @pytest.mark.asyncio
    async def test_a_failed_start_save_refuses_the_turn_and_says_why(self, built: Any) -> None:
        """Nothing runs, nothing is written, and the refusal names its cause (P6).

        Killed by: src/uclone_x/room/orchestrator.py :: except Exception as exc:  # any failure to count the start refuses the turn
        Becomes: except ImportError as exc:  # any failure to count the start refuses the turn
        Killed by: src/uclone_x/room/orchestrator.py :: self._record_turn_started(room_id, speaker, turn_id)
        Becomes: pass
        """
        from uclone_x.errors import TurnNotStartedError

        store, _, agents = built
        real_save = store.save

        def _refuse_the_start(state: RoomState) -> Any:
            before = store.load(state.room_id)
            if (
                before is not None
                and state.file_record.turns_started > before.file_record.turns_started
            ):
                raise OSError("disk full")
            return real_save(state)

        store.save = _refuse_the_start
        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        with pytest.raises(TurnNotStartedError, match="was not started.*Nothing was run"):
            await orch.post("r1", "alice", "go")

        assert agents["scout"].prompts == []
        after = store.load("r1")
        assert after is not None
        assert [m.sender_id for m in after.transcript] == ["alice"]
        assert after.file_record.turns_started == 0
        assert orch.turn_unlanded("r1") is False


# --------------------------------------------------------------------------------------
# A save that failed is on the row, whatever the reply claims (#1375)
# --------------------------------------------------------------------------------------


def _remembering_agent(participant: Participant, tmp_path: Any, calls: list[Any]) -> Any:
    """A real agent with its own memory, whose model asks for `calls` and then claims success."""
    from uclone_x.agent import BaseAgent
    from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
    from uclone_x.llm import MockLLMConnector
    from uclone_x.memory.store import CrossSessionMemory

    return BaseAgent(
        config=AgentConfig(
            agent_id=participant.id,
            name=participant.id,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=MockLLMConnector(
            responses=["", "I've saved your favourite colour."],
            tool_calls=calls,
        ),
        context=AgentContext(session_id=participant.session_id, agent_id=participant.id),
        memory=CrossSessionMemory(storage_path=tmp_path / f"{participant.id}-memory.json"),
    )


def _save_call(call_id: str, arguments: dict[str, Any]) -> Any:
    from uclone_x.llm.models import ToolCallRequest

    return ToolCallRequest(id=call_id, name="record_memory_fact", arguments=arguments)


class TestAFailedSaveIsOnTheRow:
    def test_the_orchestrator_matches_the_memory_tool_by_its_registered_name(self) -> None:
        """The kernel spells the adapter's tool name instead of importing it; this keeps them equal.

        Killed by: src/uclone_x/room/orchestrator.py :: MEMORY_SAVE_TOOL_NAME = "record_memory_fact"
        Becomes: MEMORY_SAVE_TOOL_NAME = "record_memory"
        """
        from uclone_x.memory.tools import RecordMemoryFactTool
        from uclone_x.room.orchestrator import MEMORY_SAVE_TOOL_NAME

        assert MEMORY_SAVE_TOOL_NAME == RecordMemoryFactTool.name

    @pytest.mark.asyncio
    async def test_a_turn_whose_every_save_failed_says_so_beside_the_claim(
        self, built: Any, tmp_path: Any
    ) -> None:
        """The reply says it saved; the row says nothing was.

        Observed 2026-09-22 on qwen3:8b: both memory saves came back as errors and the
        clone answered that it had saved the colour. The conversation is what the user
        reads, and the errors were only in Activity. The model's words are left alone --
        the row carries the failure next to them, as counts and not as the tool's error.

        Killed by: src/uclone_x/room/orchestrator.py :: memory_facts_unsaved=memory_facts_unsaved,
        Becomes: memory_facts_unsaved=0,
        """
        _, _, agents = built
        _cap(built, 1)
        agents["scout"] = _remembering_agent(
            SCOUT,
            tmp_path,
            [
                _save_call("c1", {"subject": "user", "predicate": "colour"}),
                _save_call("c2", {"subject": "user", "predicate": "colour", "object_value": " "}),
            ],
        )

        state = await _orchestrator(built, [ScriptedSelector("s", [speak("scout")])]).post(
            "r1", "alice", "remember my favourite colour is teal"
        )

        row = state.transcript[-1]
        assert row.content == "I've saved your favourite colour."
        assert row.error is None
        # Two attempts at one fact: one fact tried, one never saved.
        assert (row.memory_facts_tried, row.memory_facts_unsaved) == (1, 1)
        assert agents["scout"].memory.list_facts() == []
        # The tool's errors -- a pydantic dump and a `ValueError` -- are the model's to read.
        # None of it is on the row, which is what the conversation renders.
        dumped = row.model_dump_json()
        for internal in ("RecordMemoryFactParams", "object_value", "ValueError", "pydantic"):
            assert internal not in dumped, f"{internal!r} reached the row: {dumped}"

    @pytest.mark.asyncio
    async def test_a_turn_where_the_retry_landed_carries_no_failure(
        self, built: Any, tmp_path: Any
    ) -> None:
        """A failed attempt followed by one that landed, for the same fact, is a save.

        Killed by: src/uclone_x/room/orchestrator.py :: if any(_agrees(key, done) for done in landed):
        Becomes: if False:
        """
        _, _, agents = built
        _cap(built, 1)
        agents["scout"] = _remembering_agent(
            SCOUT,
            tmp_path,
            [
                _save_call("c1", {"subject": "user", "predicate": "colour"}),
                _save_call(
                    "c2", {"subject": "user", "predicate": "colour", "object_value": "teal"}
                ),
            ],
        )

        state = await _orchestrator(built, [ScriptedSelector("s", [speak("scout")])]).post(
            "r1", "alice", "remember my favourite colour is teal"
        )

        row = state.transcript[-1]
        assert (row.memory_facts_tried, row.memory_facts_unsaved) == (1, 0)
        assert [f.object_value for f in agents["scout"].memory.list_facts()] == ["teal"]

    @pytest.mark.asyncio
    async def test_a_turn_that_saved_one_fact_and_failed_another_says_one_was_not_saved(
        self, built: Any, tmp_path: Any
    ) -> None:
        """The reviewer's case (#1400): two different facts, one saved, one refused.

        Before, one success anywhere in the turn silenced the row, while the reply could
        claim both.
        """
        _, _, agents = built
        _cap(built, 1)
        agents["scout"] = _remembering_agent(
            SCOUT,
            tmp_path,
            [
                _save_call(
                    "c1", {"subject": "user", "predicate": "colour", "object_value": "teal"}
                ),
                _save_call("c2", {"subject": "user", "predicate": "city"}),
            ],
        )

        state = await _orchestrator(built, [ScriptedSelector("s", [speak("scout")])]).post(
            "r1", "alice", "remember my colour is teal and I live in Seoul"
        )

        row = state.transcript[-1]
        assert (row.memory_facts_tried, row.memory_facts_unsaved) == (2, 1)

    @pytest.mark.asyncio
    async def test_a_turn_that_saved_nothing_and_tried_nothing_carries_no_failure(
        self, built: Any
    ) -> None:
        """No save attempted is not a failed save, and a row stored before the fields loads."""
        from uclone_x.room.models import RoomMessage

        store, _, _ = built
        _cap(built, 1)
        state = await _orchestrator(built, [ScriptedSelector("s", [speak("scout")])]).post(
            "r1", "alice", "hello"
        )
        row = state.transcript[-1]
        assert (row.memory_facts_tried, row.memory_facts_unsaved) == (0, 0)

        stored = store.load("r1")
        assert stored is not None
        old = stored.transcript[-1].model_dump(mode="json")
        del old["memory_facts_tried"]
        del old["memory_facts_unsaved"]
        legacy = RoomMessage.model_validate_json(json.dumps(old))
        assert (legacy.memory_facts_tried, legacy.memory_facts_unsaved) == (0, 0)


def _save_record(arguments: dict[str, Any], *, ok: bool) -> Any:
    from uclone_x.agent.models import ToolExecutionRecord
    from uclone_x.tools.models import ToolResultStatus

    return ToolExecutionRecord(
        tool_name="record_memory_fact",
        arguments=arguments,
        status=ToolResultStatus.SUCCESS if ok else ToolResultStatus.ERROR,
        error=None if ok else "refused",
    )


class TestMemorySaveOutcome:
    """`_memory_save_outcome` counts facts, not calls, and only what its names can tell."""

    def test_a_failure_the_retry_completed_by_naming_a_missing_subject_is_resolved(self) -> None:
        """A failed call that left a part out agrees with a success on the parts it named.

        Killed by: src/uclone_x/room/orchestrator.py :: return all(mine == "" or mine == theirs for mine, theirs in zip(named, other, strict=True))
        Becomes: return all(mine == theirs for mine, theirs in zip(named, other, strict=True))
        """
        from uclone_x.room.orchestrator import (
            _memory_save_outcome,  # pyright: ignore[reportPrivateUsage]
        )

        records = [
            _save_record({"predicate": "colour", "object_value": "teal"}, ok=False),
            _save_record(
                {"subject": "User ", "predicate": "Colour", "object_value": "teal"}, ok=True
            ),
        ]
        assert _memory_save_outcome(records) == (1, 0)

    def test_two_failures_of_one_fact_are_one_unsaved_fact(self) -> None:
        """Killed by: src/uclone_x/room/orchestrator.py :: if any(_agrees(key, seen) or _agrees(seen, key) for seen in unsaved):
        Becomes: if False:
        """
        from uclone_x.room.orchestrator import (
            _memory_save_outcome,  # pyright: ignore[reportPrivateUsage]
        )

        records = [
            _save_record({"subject": "user", "predicate": "colour"}, ok=False),
            _save_record({"predicate": "colour", "object_value": "teal"}, ok=False),
        ]
        assert _memory_save_outcome(records) == (1, 1)

    def test_a_success_does_not_resolve_a_different_fact(self) -> None:
        """Killed by: src/uclone_x/room/orchestrator.py :: return all(mine == "" or mine == theirs for mine, theirs in zip(named, other, strict=True))
        Becomes: return True
        """
        from uclone_x.room.orchestrator import (
            _memory_save_outcome,  # pyright: ignore[reportPrivateUsage]
        )

        records = [
            _save_record(
                {"subject": "user", "predicate": "colour", "object_value": "teal"}, ok=True
            ),
            _save_record({"subject": "user", "predicate": "city"}, ok=False),
        ]
        assert _memory_save_outcome(records) == (2, 1)

    def test_other_tools_are_not_counted(self) -> None:
        """Killed by: src/uclone_x/room/orchestrator.py :: saves = [e for e in executions if e.tool_name == MEMORY_SAVE_TOOL_NAME]
        Becomes: saves = list(executions)
        """
        from uclone_x.agent.models import ToolExecutionRecord
        from uclone_x.room.orchestrator import (
            _memory_save_outcome,  # pyright: ignore[reportPrivateUsage]
        )
        from uclone_x.tools.models import ToolResultStatus

        records = [
            ToolExecutionRecord(
                tool_name="plan", arguments={}, status=ToolResultStatus.ERROR, error="x"
            )
        ]
        assert _memory_save_outcome(records) == (0, 0)


class TestAutonomousDiscussionMode:
    """Autonomous discussion runs under user presence and circuit breaker ceiling."""

    @pytest.mark.asyncio
    async def test_autonomous_mode_pauses_when_user_not_present(self, built: Any) -> None:
        store, _, _ = built
        state = store.load("r1")
        assert state is not None
        store.save(state.model_copy(update={"policy": RoomPolicy(autonomous=True)}))

        orch = _orchestrator(built, [ScriptedSelector("s", [speak("scout")])])
        # Presence is not noted, so user is inactive
        result = await orch.post("r1", "alice", "hello")

        assert result.last_decision is not None
        assert result.last_decision.verdict is SelectionVerdict.SILENCE
        assert "autonomous discussion paused" in result.last_decision.reasoning

    @pytest.mark.asyncio
    async def test_autonomous_mode_runs_with_active_presence_and_stops_at_circuit_breaker(
        self, built: Any
    ) -> None:
        from uclone_x.room.orchestrator import AUTONOMOUS_CIRCUIT_BREAKER_TURNS

        store, _, _ = built
        state = store.load("r1")
        assert state is not None
        store.save(state.model_copy(update={"policy": RoomPolicy(autonomous=True)}))

        # Script 25 speak decisions
        decisions = [speak("scout" if i % 2 == 0 else "critic") for i in range(25)]
        orch = _orchestrator(built, [ScriptedSelector("s", decisions)])
        orch.note_presence("r1", active=True)

        result = await orch.post("r1", "alice", "let's discuss")

        assert result.turn_state.agent_turns_since_human == AUTONOMOUS_CIRCUIT_BREAKER_TURNS
        assert result.last_decision is not None
        assert result.last_decision.verdict is SelectionVerdict.SILENCE
        assert "circuit breaker" in result.last_decision.reasoning
