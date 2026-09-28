"""A clone learns durable facts from a room turn, after the turn (#1404).

Clone-knowledge-graph design §3.3, as built in step 4: when a turn commits, the room queues
what the speaker saw and did; once the room's cascade has ended and nobody holds the floor,
the clone's own model is asked once for the durable facts in it; they are curated and saved
straight into the clone's memory, and the turn's row says how many (or that it failed).

Every model here is a `BaseLLMConnector` fake and every seat a real `BaseAgent`: no network,
no clock, no randomness. The fake tells the extraction call from a reply by the extraction
prompt's first words, so a test controls each independently.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.errors import LLMProviderError
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
)
from uclone_x.memory.extractor import (
    EXTRACTED_CONFIDENCE_CAP,
    EXTRACTION_FAILED,
    KnowledgeExtractor,
    Lesson,
    SpanLine,
)
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomMessage,
    RoomPolicy,
    RoomState,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerRequest,
)
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.store import RoomStore
from uclone_x.tools.registry import ToolRegistry

ALICE = Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice")
SCOUT = Participant(
    id="scout",
    kind=ParticipantKind.AGENT,
    display_name="Scout",
    session_id="sess_room__r1__scout",
)
SAGE = Participant(
    id="sage",
    kind=ParticipantKind.AGENT,
    display_name="Sage",
    session_id="sess_room__r1__sage",
)

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="scripted", model="scripted"),
    served_by=ServiceRef(provider="scripted", model="scripted"),
    attempts=(),
)
_USAGE = TokenUsage(provider="scripted", model="scripted", input_tokens=0, output_tokens=0)

#: The extraction prompt's opening words: how the fake knows which call it is answering.
_EXTRACTION_MARK = "You pick out durable facts"


def _fact(
    subject: str, relation: str, value: str, source: str = "person", **extra: Any
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "subject": subject,
        "relation": relation,
        "value": value,
        "source": source,
        "confidence": 0.95,
        "durable": True,
        "why": "said so",
    }
    item.update(extra)
    return item


class _Model(BaseLLMConnector):
    """Replies with `reply`; answers the extraction call with the next queued answer.

    An extraction answer is a list of facts (sent as JSON), a raw string, or an exception
    to raise. Every request is kept, split into replies and extractions.
    """

    def __init__(self, reply: str = "Noted.", extractions: Sequence[Any] = ()) -> None:
        super().__init__()
        self.reply = reply
        self.extractions = list(extractions)
        self.replies: list[LLMRequest] = []
        self.extraction_requests: list[LLMRequest] = []
        #: Set while an extraction call is in progress, so a test can pause it there.
        self.hold: asyncio.Event | None = None

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        system = str(request.messages[0].content) if request.messages else ""
        if system.startswith(_EXTRACTION_MARK):
            self.extraction_requests.append(request)
            if self.hold is not None:
                await self.hold.wait()
            answer: Any = self.extractions.pop(0) if self.extractions else []
            if isinstance(answer, Exception):
                raise answer
            content = answer if isinstance(answer, str) else json.dumps(answer)
        else:
            self.replies.append(request)
            content = self.reply
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content=content,
            tool_calls=(),
            usage=_USAGE,
            provenance=_PROV,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        response = await self.generate(request)
        yield StreamChunk(delta_content=response.content or "")


def _seat(participant: Participant, llm: _Model, memory: CrossSessionMemory | None) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id=participant.id,
            name=participant.display_name,
            llm_config=AgentLLMConfig(model_name="scripted"),
        ),
        llm=llm,
        tools=ToolRegistry(),
        context=AgentContext(session_id=participant.session_id, agent_id=participant.id),
        memory=memory,
        store=SessionStore(_SESSIONS[0] / participant.id),
    )


#: Where the current test's seats keep their sessions; set per test under its `tmp_path`.
_SESSIONS: list[Path] = [Path()]


@pytest.fixture(autouse=True)
def _sessions_under_tmp(tmp_path: Path) -> None:  # pyright: ignore[reportUnusedFunction]
    _SESSIONS[0] = tmp_path / "sessions"


class _Seats:
    def __init__(self, *agents: tuple[Participant, BaseAgent]) -> None:
        self.agents = {participant.id: agent for participant, agent in agents}

    async def resolve(self, participant: Participant, *, one_seat: bool = False) -> BaseAgent:
        return self.agents[participant.id]


class _InOrder:
    """Gives the floor to each named clone in turn, then is silent. `None` is one silence."""

    def __init__(self, *speakers: str | None) -> None:
        self._speakers = list(speakers)

    @property
    def name(self) -> str:
        return "in-order"

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        speaker = self._speakers.pop(0) if self._speakers else None
        if speaker is None:
            return SpeakerDecision(verdict=SelectionVerdict.SILENCE, selector=self.name)
        return SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id=speaker,
            selector=self.name,
            confidence=1.0,
        )


def _room(
    tmp_path: Path,
    seats: _Seats,
    *speakers: str | None,
    room_id: str = "r1",
    bus: EventBus | None = None,
    extractor: KnowledgeExtractor | None = None,
) -> RoomOrchestrator:
    store = RoomStore(tmp_path / "rooms")
    agents = tuple(p for p in (SCOUT, SAGE) if p.id in seats.agents)
    store.save(
        RoomState(
            room_id=room_id,
            participants=(ALICE, *agents),
            policy=RoomPolicy(max_agent_turns_per_human_message=len(speakers) or 1),
        )
    )
    return RoomOrchestrator(
        store=store,
        selectors=[_InOrder(*(speakers or (SCOUT.id,)))],
        resolver=seats,
        bus=bus,
        extractor=extractor if extractor is not None else KnowledgeExtractor(),
    )


def _memory(tmp_path: Path, name: str = "scout") -> CrossSessionMemory:
    return CrossSessionMemory(storage_path=tmp_path / f"{name}-memory.json")


def _row(state: RoomState, sender: str) -> RoomMessage:
    return [m for m in state.transcript if m.sender_id == sender][-1]


async def _say(orch: RoomOrchestrator, content: str, room_id: str = "r1") -> RoomState:
    await orch.post(room_id, ALICE.id, content)
    await orch.wait_for_learning(room_id)
    return orch._require_room(room_id)  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------------------
# What a person said is saved to the clone's memory, with where it was learned.
# --------------------------------------------------------------------------------------


async def test_a_told_fact_is_saved_with_its_origin_room_and_turn(tmp_path: Path) -> None:
    """The fact lands in the speaker's own memory, `told`, naming the room and the turn.

    Killed by: src/uclone_x/memory/extractor.py :: _SOURCE_ORIGIN: Final[dict[str, FactOrigin]] = {"person": "told", "tool": "found"}
    Becomes: _SOURCE_ORIGIN: Final[dict[str, FactOrigin]] = {"person": "found", "tool": "found"}
    Killed by: src/uclone_x/memory/extractor.py :: source_turn_id=lesson.turn_id,
    Becomes: source_turn_id=None,
    Killed by: src/uclone_x/memory/extractor.py :: confidence=min(candidate.confidence, EXTRACTED_CONFIDENCE_CAP),
    Becomes: confidence=candidate.confidence,
    """
    memory = _memory(tmp_path)
    llm = _Model(extractions=[[_fact("Alice", "prefers", "green tea")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    state = await _say(orch, "I only drink green tea, by the way.")

    row = _row(state, SCOUT.id)
    [fact] = memory.list_facts()
    assert (fact.subject, fact.predicate, fact.object_value) == ("user", "prefers", "green tea")
    assert fact.origin == "told"
    assert fact.source_room_id == "r1"
    assert fact.source_turn_id == row.turn_id
    assert fact.confidence == EXTRACTED_CONFIDENCE_CAP
    assert fact.metadata["observations"] == [row.turn_id]
    assert row.knowledge_learned == (fact.fact_id,)
    assert row.knowledge_extract_error is None
    # Saved to disk, not only held: a fresh load of the clone's memory has it.
    assert [f.fact_id for f in _memory(tmp_path).list_facts()] == [fact.fact_id]


async def test_the_extraction_sees_the_span_labelled_by_source(tmp_path: Path) -> None:
    """The model is shown the person's words, the clone's reply, and facts already held.

    Killed by: src/uclone_x/room/orchestrator.py :: lines.append(SpanLine("self", speaker.display_name, content))
    Becomes: pass
    """
    memory = _memory(tmp_path)
    memory.record_fact("user", "lives_in", "Busan", _PROV, "s0")
    llm = _Model(reply="Green tea it is.", extractions=[[]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    await _say(orch, "I only drink green tea.")

    [request] = llm.extraction_requests
    span = str(request.messages[1].content)
    assert "[person: Alice] I only drink green tea." in span
    assert "[Scout (you)] Green tea it is." in span
    assert "user | lives_in | Busan" in str(request.messages[0].content)
    # The clone's configured model, filled in because the extractor names none.
    assert request.model == "scripted"


async def test_a_learned_fact_reaches_the_next_conversation(tmp_path: Path) -> None:
    """Another room, the same clone: its prompt carries what the first room taught it."""
    llm = _Model(extractions=[[_fact("user", "prefers", "green tea")]])
    first = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, _memory(tmp_path)))))
    await _say(first, "I only drink green tea.")

    again = _Model()
    second = _room(
        tmp_path / "two", _Seats((SCOUT, _seat(SCOUT, again, _memory(tmp_path)))), room_id="r2"
    )
    await _say(second, "What should I order?", room_id="r2")

    # The fact as the memory section renders it: words no message in either room used.
    assert any("user: prefers -> green tea" in str(m.content) for m in again.replies[-1].messages)


async def test_two_clones_in_one_room_keep_their_own_facts(tmp_path: Path) -> None:
    """Each clone's extraction lands in its own memory, never the other's."""
    scout_memory, sage_memory = _memory(tmp_path, "scout"), _memory(tmp_path, "sage")
    scout_llm = _Model(extractions=[[_fact("user", "works_at", "a bakery")]])
    sage_llm = _Model(extractions=[[_fact("user", "has_pet", "a cat")]])
    orch = _room(
        tmp_path,
        _Seats(
            (SCOUT, _seat(SCOUT, scout_llm, scout_memory)),
            (SAGE, _seat(SAGE, sage_llm, sage_memory)),
        ),
        SCOUT.id,
        SAGE.id,
    )

    await _say(orch, "I work at a bakery and I have a cat.")

    assert [f.object_value for f in scout_memory.list_facts()] == ["a bakery"]
    assert [f.object_value for f in sage_memory.list_facts()] == ["a cat"]


# --------------------------------------------------------------------------------------
# Only the person and the clone's own tool results ground a fact.
# --------------------------------------------------------------------------------------


async def test_another_clones_message_is_never_a_source(tmp_path: Path) -> None:
    """Sage's line is shown to Scout's extraction as context; a fact sourced to it is dropped.

    Killed by: src/uclone_x/memory/extractor.py :: if source not in _SOURCE_ORIGIN:
    Becomes: if False:
    Killed by: src/uclone_x/room/orchestrator.py :: kind = "person" if kinds.get(row.sender_id) is ParticipantKind.HUMAN else "other"
    Becomes: kind = "person"
    """
    scout_memory = _memory(tmp_path, "scout")
    sage_llm = _Model(reply="Alice told me she moved to Seoul.")
    scout_llm = _Model(
        extractions=[
            [
                _fact("user", "lives_in", "Seoul", source="other"),
                _fact("user", "likes", "maps", source="person"),
            ]
        ]
    )
    orch = _room(
        tmp_path,
        _Seats((SAGE, _seat(SAGE, sage_llm, None)), (SCOUT, _seat(SCOUT, scout_llm, scout_memory))),
        SAGE.id,
        SCOUT.id,
    )

    await _say(orch, "I like maps.")

    span = str(scout_llm.extraction_requests[0].messages[1].content)
    assert "[other clone: Sage] Alice told me she moved to Seoul." in span
    assert [f.object_value for f in scout_memory.list_facts()] == ["maps"]


async def test_a_clone_echoing_another_clone_does_not_make_it_a_fact(tmp_path: Path) -> None:
    """Sage claims where Alice works; Scout agrees. Scout's agreement grounds nothing.

    Saving it would let one clone's guess, repeated by another, come back as a found fact
    -- and then be repeated again from memory. The clone's own words are context only.

    Killed by: src/uclone_x/memory/extractor.py :: _SOURCE_ORIGIN: Final[dict[str, FactOrigin]] = {"person": "told", "tool": "found"}
    Becomes: _SOURCE_ORIGIN: Final[dict[str, FactOrigin]] = {"person": "told", "tool": "found", "self": "found"}
    """
    scout_memory = _memory(tmp_path, "scout")
    sage_llm = _Model(reply="Alice works at Acme, I think.")
    scout_llm = _Model(
        reply="Right, you work at Acme.",
        extractions=[
            [
                _fact("user", "works_at", "Acme", source="self"),
                _fact("user", "likes", "maps", source="person"),
            ]
        ],
    )
    orch = _room(
        tmp_path,
        _Seats((SAGE, _seat(SAGE, sage_llm, None)), (SCOUT, _seat(SCOUT, scout_llm, scout_memory))),
        SAGE.id,
        SCOUT.id,
    )

    state = await _say(orch, "I like maps.")

    request = scout_llm.extraction_requests[0]
    assert "[Scout (you)] Right, you work at Acme." in str(request.messages[1].content)
    assert "context only, never a source" in str(request.messages[0].content)
    [fact] = scout_memory.list_facts()
    assert (fact.predicate, fact.object_value, fact.origin) == ("likes", "maps", "told")
    assert _row(state, SCOUT.id).knowledge_learned == (fact.fact_id,)


async def test_a_source_the_span_does_not_hold_is_dropped(tmp_path: Path) -> None:
    """A fact claimed from a tool result, in a turn that ran no tool, has nothing under it.

    Killed by: src/uclone_x/memory/extractor.py :: if candidate.source not in kinds:
    Becomes: if False:
    """
    memory = _memory(tmp_path)
    llm = _Model(extractions=[[_fact("user", "birthday", "May 3", source="tool")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    state = await _say(orch, "Hello there.")

    assert memory.list_facts() == []
    assert _row(state, SCOUT.id).knowledge_learned == ()


async def test_nothing_that_could_ground_a_fact_means_no_model_call(tmp_path: Path) -> None:
    """A span holding only clones' words -- another's and its own -- is not sent at all.

    Killed by: src/uclone_x/memory/extractor.py :: if not any(line.kind in _SOURCE_ORIGIN for line in lesson.lines):
    Becomes: if False:
    """
    memory = _memory(tmp_path)
    calls: list[LLMRequest] = []

    async def generate(request: LLMRequest) -> ModelResponse:
        calls.append(request)
        raise AssertionError("the model must not be asked")

    lesson = Lesson(
        clone_id="scout",
        clone_name="Scout",
        room_id="r1",
        session_id="s1",
        turn_id="t1",
        lines=(
            SpanLine("other", "Sage", "Alice lives in Seoul."),
            SpanLine("self", "Scout", "Right, Seoul."),
        ),
        memory=memory,
        generate=generate,
    )
    outcome = await KnowledgeExtractor().extract(lesson)

    assert calls == []
    assert outcome.error is None
    assert outcome.added == ()


# --------------------------------------------------------------------------------------
# Curation: durable only, deduplicated, superseding -- and a person's correction wins.
# --------------------------------------------------------------------------------------


async def test_not_durable_and_unsure_facts_are_dropped(tmp_path: Path) -> None:
    """
    Killed by: src/uclone_x/memory/extractor.py :: if fields.get("durable") is not True:
    Becomes: if False:
    Killed by: src/uclone_x/memory/extractor.py :: if candidate.confidence < MIN_CONFIDENCE:
    Becomes: if False:
    """
    memory = _memory(tmp_path)
    llm = _Model(
        extractions=[
            [
                _fact("user", "is_tired", "today", durable=False),
                _fact("user", "maybe_likes", "jazz", confidence=0.3),
                _fact("user", "likes", "jazz"),
            ]
        ]
    )
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    await _say(orch, "Tired today. I like jazz.")

    assert [(f.predicate, f.object_value) for f in memory.list_facts()] == [("likes", "jazz")]


async def test_a_fact_seen_again_is_reinforced_not_repeated(tmp_path: Path) -> None:
    """The held fact gains the turn as an observation; no second fact, nothing on the row.

    Killed by: src/uclone_x/memory/extractor.py :: same = [f for f in active if _value_key(f.object_value) == candidate.keys[2]]
    Becomes: same = []
    Killed by: src/uclone_x/memory/store.py :: "metadata": {**current.metadata, "observations": [*observations, turn_id]},
    Becomes: "metadata": {**current.metadata, "observations": observations},
    """
    memory = _memory(tmp_path)
    held = memory.record_fact(
        "user", "prefers", "Green Tea", _PROV, "s0", metadata={"observations": ["t0"]}
    )
    llm = _Model(extractions=[[_fact("User", "Prefers", "green  tea")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    state = await _say(orch, "Green tea again, please.")

    row = _row(state, SCOUT.id)
    [fact] = memory.list_facts()
    assert fact.fact_id == held.fact_id
    assert fact.object_value == "Green Tea"
    assert fact.metadata["observations"] == ["t0", row.turn_id]
    assert row.knowledge_learned == ()
    assert _memory(tmp_path).get_fact(held.fact_id).metadata["observations"] == ["t0", row.turn_id]  # type: ignore[union-attr]


async def test_a_changed_fact_supersedes_the_old_one(tmp_path: Path) -> None:
    """
    Killed by: src/uclone_x/memory/extractor.py :: auto_retract_conflicts=True,
    Becomes: auto_retract_conflicts=False,
    """
    memory = _memory(tmp_path)
    old = memory.record_fact("user", "lives_in", "Busan", _PROV, "s0", origin="told")
    llm = _Model(extractions=[[_fact("user", "lives_in", "Seoul")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    await _say(orch, "I moved to Seoul last month.")

    [new] = memory.list_facts()
    assert new.object_value == "Seoul"
    assert new.contradicts_fact_id == old.fact_id
    assert memory.get_fact(old.fact_id).retracted  # type: ignore[union-attr]


async def test_a_persons_correction_survives_a_conflicting_extraction(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The corrected fact stays; the extracted one is dropped and the drop is logged.

    The correction is made through a second memory object, as the Remembers panel's server
    route would, after the seat's memory was loaded: the extractor re-reads before it
    decides.

    Killed by: src/uclone_x/memory/extractor.py :: corrected = [f for f in active if f.origin == "corrected"]
    Becomes: corrected = []
    """
    seat_memory = _memory(tmp_path)
    original = seat_memory.record_fact("user", "lives_in", "Busan", _PROV, "s0", origin="told")
    corrected = _memory(tmp_path).correct_fact(original.fact_id, "Daegu", _PROV)
    llm = _Model(extractions=[[_fact("user", "lives_in", "Busan")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, seat_memory))))

    with caplog.at_level(logging.INFO, logger="uclone_x.memory.extractor"):
        state = await _say(orch, "Back home in Busan for the weekend.")

    assert [f.fact_id for f in _memory(tmp_path).list_facts()] == [corrected.fact_id]
    assert _row(state, SCOUT.id).knowledge_learned == ()
    assert "corrected by a person" in caplog.text


#: "José" as a build before #1895 could store it (`e` and a combining accent); the extractor
#: saves the composed form (`person_subject`), so the two meet only when compared folded.
_JOSE_DECOMPOSED = "Jose\u0301"
_JOSE_COMPOSED = "Jos\u00e9"


async def test_a_fact_stored_decomposed_is_superseded_by_the_composed_extraction(
    tmp_path: Path,
) -> None:
    """The old subject spelling is still the same subject, so the new value replaces it (#1893).

    Killed by: src/uclone_x/memory/models.py :: same_subject = fold_name(self.subject) == fold_name(other.subject)
    Becomes: same_subject = self.subject.strip().lower() == other.subject.strip().lower()
    """
    memory = _memory(tmp_path)
    old = memory.record_fact(_JOSE_DECOMPOSED, "lives_in", "Busan", _PROV, "s0", origin="told")
    llm = _Model(extractions=[[_fact(_JOSE_COMPOSED, "lives_in", "Seoul")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    await _say(orch, f"{_JOSE_COMPOSED} moved to Seoul last month.")

    [new] = memory.list_facts()
    assert new.subject == _JOSE_COMPOSED
    assert new.object_value == "Seoul"
    assert new.contradicts_fact_id == old.fact_id


async def test_a_fact_stored_decomposed_is_shown_when_the_span_names_it_composed(
    tmp_path: Path,
) -> None:
    """The extraction prompt lists what the clone already holds about a named subject (#1899).

    Killed by: src/uclone_x/memory/extractor.py :: if (subject := fold_name(fact.subject)) == "user" or subject in text
    Becomes: if (subject := fact.subject.casefold()) == "user" or subject in text
    """
    memory = _memory(tmp_path)
    memory.record_fact(_JOSE_DECOMPOSED, "lives_in", "Busan", _PROV, "s0", origin="told")
    llm = _Model(extractions=[[]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    await _say(orch, f"{_JOSE_COMPOSED} is visiting next week.")

    [request] = llm.extraction_requests
    assert f"{_JOSE_DECOMPOSED} | lives_in | Busan" in str(request.messages[0].content)


async def test_a_fact_stored_composed_is_shown_when_the_span_names_it_decomposed(
    tmp_path: Path,
) -> None:
    """The reverse direction: the stored subject is folded too (#1899).

    Killed by: src/uclone_x/memory/extractor.py :: text = fold_name(" ".join(line.text for line in lesson.lines))
    Becomes: text = " ".join(line.text for line in lesson.lines).casefold()
    """
    memory = _memory(tmp_path)
    memory.record_fact(_JOSE_COMPOSED, "lives_in", "Busan", _PROV, "s0", origin="told")
    llm = _Model(extractions=[[]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    await _say(orch, f"{_JOSE_DECOMPOSED} is visiting next week.")

    [request] = llm.extraction_requests
    assert f"{_JOSE_COMPOSED} | lives_in | Busan" in str(request.messages[0].content)


async def test_a_correction_stored_decomposed_still_outranks_the_model(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A person's correction under the old spelling beats a composed extraction (#1893).

    Killed by: src/uclone_x/memory/store.py :: if target_subj and fold_name(fact.subject) != target_subj:
    Becomes: if target_subj and fact.subject.strip().lower() != target_subj:
    """
    seat_memory = _memory(tmp_path)
    original = seat_memory.record_fact(
        _JOSE_DECOMPOSED, "lives_in", "Busan", _PROV, "s0", origin="told"
    )
    corrected = _memory(tmp_path).correct_fact(original.fact_id, "Daegu", _PROV)
    llm = _Model(extractions=[[_fact(_JOSE_COMPOSED, "lives_in", "Busan")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, seat_memory))))

    with caplog.at_level(logging.INFO, logger="uclone_x.memory.extractor"):
        state = await _say(orch, f"{_JOSE_COMPOSED} is back in Busan for the weekend.")

    assert [f.fact_id for f in _memory(tmp_path).list_facts()] == [corrected.fact_id]
    assert _row(state, SCOUT.id).knowledge_learned == ()
    assert "corrected by a person" in caplog.text


async def test_a_correction_made_while_the_model_thinks_still_wins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A person corrects the fact after the extraction call went out; the save sees it.

    Not reading again would act on the pre-correction view: reinforce a fact that is gone,
    fail, and put a failure on the row for a turn that taught nothing new.

    Killed by: src/uclone_x/memory/extractor.py :: memory.refresh()  # the model call awaited; a person may have acted meanwhile
    Becomes: pass
    """
    seat_memory = _memory(tmp_path)
    original = seat_memory.record_fact("user", "lives_in", "Busan", _PROV, "s0", origin="told")
    llm = _Model(extractions=[[_fact("user", "lives_in", "Busan")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, seat_memory))))
    llm.hold = asyncio.Event()

    await orch.post("r1", ALICE.id, "Back home in Busan for the weekend.")
    learning = asyncio.create_task(orch.wait_for_learning("r1"))
    for _ in range(50):
        if llm.extraction_requests:
            break
        await asyncio.sleep(0)
    assert len(llm.extraction_requests) == 1  # the call is out, and paused
    corrected = _memory(tmp_path).correct_fact(original.fact_id, "Daegu", _PROV)
    with caplog.at_level(logging.INFO, logger="uclone_x.memory.extractor"):
        llm.hold.set()
        await learning

    assert [f.fact_id for f in _memory(tmp_path).list_facts()] == [corrected.fact_id]
    row = _row(orch._require_room("r1"), SCOUT.id)  # pyright: ignore[reportPrivateUsage]
    assert (row.knowledge_learned, row.knowledge_extract_error) == ((), None)
    assert "corrected by a person" in caplog.text


# --------------------------------------------------------------------------------------
# A failure is one plain sentence on the row; the reply is untouched.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        LLMProviderError("upstream 503 for request req_9f8e7d at /v1/chat"),
        "Sure! Here is what I learned: mem_0123abcd",
    ],
    ids=["model-error", "not-json"],
)
async def test_a_failed_extraction_says_so_plainly_and_leaves_the_reply(
    tmp_path: Path, answer: Any
) -> None:
    """The row carries the plain sentence -- no exception text, no ids -- and the reply stands.

    Killed by: src/uclone_x/memory/extractor.py :: return replace(outcome, error=EXTRACTION_FAILED)
    Becomes: return replace(outcome, error=f"{type(exc).__name__}: {exc}")
    Killed by: src/uclone_x/memory/extractor.py :: raise ExtractionFailed("the reply holds no JSON array")
    Becomes: return []
    """
    memory = _memory(tmp_path)
    llm = _Model(reply="Happy to help.", extractions=[answer])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    state = await _say(orch, "I run marathons.")

    row = _row(state, SCOUT.id)
    assert row.content == "Happy to help."
    assert row.error is None
    assert row.knowledge_extract_error == EXTRACTION_FAILED
    shown = row.knowledge_extract_error
    for internal in (
        "503",
        "req_",
        "mem_",
        "LLMProviderError",
        "/v1",
        "Error",
        "JSON",
        row.turn_id or "-",
    ):
        assert internal not in shown
    assert memory.list_facts() == []


async def test_a_failed_save_says_so_plainly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Killed by: src/uclone_x/memory/extractor.py :: error=EXTRACTION_FAILED,
    Becomes: error=f"{type(exc).__name__}: {exc}",
    """
    memory = _memory(tmp_path)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device", str(tmp_path / "scout-memory.json"))

    llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))
    await orch.post("r1", ALICE.id, "I like jazz.")
    monkeypatch.setattr(memory, "save", refuse)
    await orch.wait_for_learning("r1")

    row = _row(orch._require_room("r1"), SCOUT.id)  # pyright: ignore[reportPrivateUsage]
    assert row.knowledge_extract_error == EXTRACTION_FAILED
    assert "space" not in row.knowledge_extract_error
    assert str(tmp_path) not in row.knowledge_extract_error


# --------------------------------------------------------------------------------------
# Off the reply path: after the turn landed, when the floor is free, once per cascade.
# --------------------------------------------------------------------------------------


async def test_the_turn_lands_before_learning_starts(tmp_path: Path) -> None:
    """`post` returns with the reply saved and nothing learned yet; learning follows.

    Killed by: src/uclone_x/room/orchestrator.py :: self._start_learning(room_id)  # after the cascade, never inside it (#1404)
    Becomes: await self._learn(room_id)
    """
    memory = _memory(tmp_path)
    llm = _Model(reply="Got it.", extractions=[[_fact("user", "likes", "jazz")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    state = await orch.post("r1", ALICE.id, "I like jazz.")

    assert _row(state, SCOUT.id).content == "Got it."
    assert llm.extraction_requests == []
    assert memory.list_facts() == []

    await orch.wait_for_learning("r1")
    assert [f.object_value for f in memory.list_facts()] == ["jazz"]


async def test_learning_waits_while_a_turn_holds_the_floor(tmp_path: Path) -> None:
    """
    Killed by: src/uclone_x/room/orchestrator.py :: async with waited:  # wait for the floor, then let it go
    Becomes: if True:
    """
    memory = _memory(tmp_path)
    llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))
    floor = orch._floor.setdefault("r1", asyncio.Lock())  # pyright: ignore[reportPrivateUsage]

    await orch.post("r1", ALICE.id, "I like jazz.")
    await floor.acquire()  # a turn is on the floor
    learning = asyncio.create_task(orch.wait_for_learning("r1"))
    for _ in range(20):
        await asyncio.sleep(0)
    assert llm.extraction_requests == []
    assert not learning.done()

    floor.release()
    await learning
    assert [f.object_value for f in memory.list_facts()] == ["jazz"]


async def test_the_floor_is_free_while_the_extraction_call_is_out(tmp_path: Path) -> None:
    """Learning waits for the floor but does not keep it: the next turn runs meanwhile.

    Holding it through the model call would make the person's next reply wait on learning.

    Killed by: src/uclone_x/room/orchestrator.py :: outcome = await extractor.extract(lesson)
    Becomes: async with waited: outcome = await extractor.extract(lesson)
    """
    memory = _memory(tmp_path)
    llm = _Model(reply="Noted.", extractions=[[_fact("user", "likes", "jazz")], []])
    # Scout answers the first message, the room falls quiet, Scout answers the second.
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))), SCOUT.id, None, SCOUT.id)
    llm.hold = asyncio.Event()

    await orch.post("r1", ALICE.id, "I like jazz.")
    learning = asyncio.create_task(orch.wait_for_learning("r1"))
    for _ in range(50):
        if llm.extraction_requests:
            break
        await asyncio.sleep(0)
    assert len(llm.extraction_requests) == 1  # the call is out, and paused
    floor = orch._floor["r1"]  # pyright: ignore[reportPrivateUsage]
    assert not floor.locked()

    # A second message is answered while the first turn's extraction is still out.
    state = await asyncio.wait_for(orch.post("r1", ALICE.id, "And tea."), timeout=5)
    assert [m.content for m in state.transcript if m.sender_id == SCOUT.id] == ["Noted."] * 2
    assert not learning.done()

    llm.hold.set()
    await learning
    assert [f.object_value for f in memory.list_facts()] == ["jazz"]


def test_a_row_that_learned_nothing_is_saved_as_an_older_build_wrote_it(tmp_path: Path) -> None:
    """Empty learning fields stay off disk, so rolling back past #1404 still reads the room.

    A row that did learn carries them, and both shapes load.

    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda learned: not learned,
    Becomes: exclude_if=None,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda error: error is None,
    Becomes: exclude_if=None,
    """
    store = RoomStore(tmp_path / "rooms")
    quiet = RoomMessage(seq=1, sender_id="scout", content="Hello.")
    taught = RoomMessage(seq=2, sender_id="scout", content="Noted.", knowledge_learned=("mem_a",))
    failed = RoomMessage(
        seq=3, sender_id="scout", content="Hm.", knowledge_extract_error=EXTRACTION_FAILED
    )
    store.save(
        RoomState(room_id="r1", participants=(ALICE, SCOUT), transcript=(quiet, taught, failed))
    )

    rows = json.loads(store.room_path("r1").read_text(encoding="utf-8"))["transcript"]
    assert "knowledge_learned" not in rows[0] and "knowledge_extract_error" not in rows[0]
    assert rows[1]["knowledge_learned"] == ["mem_a"]
    assert "knowledge_extract_error" not in rows[1]
    assert rows[2]["knowledge_extract_error"] == EXTRACTION_FAILED
    loaded = store.load("r1")
    assert loaded is not None
    assert [(m.knowledge_learned, m.knowledge_extract_error) for m in loaded.transcript] == [
        ((), None),
        (("mem_a",), None),
        ((), EXTRACTION_FAILED),
    ]


async def test_one_clone_speaking_twice_is_learned_once_under_the_last_turn(tmp_path: Path) -> None:
    """A cascade's turns by one clone are one extraction; the facts name the latest turn.

    Killed by: src/uclone_x/memory/extractor.py :: lines=(*held.lines, *lesson.lines),
    Becomes: lines=lesson.lines,
    """
    memory = _memory(tmp_path)
    llm = _Model(reply="Mm.", extractions=[[_fact("user", "likes", "jazz")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))), SCOUT.id, SCOUT.id)
    llm.reply = "First."
    extractor_llm_reply = ["First.", "Second."]

    async def reply_in_turn(request: LLMRequest) -> ModelResponse:
        system = str(request.messages[0].content) if request.messages else ""
        if not system.startswith(_EXTRACTION_MARK):
            llm.reply = extractor_llm_reply.pop(0)
        return await _Model.generate(llm, request)

    llm.generate = reply_in_turn  # type: ignore[method-assign]

    state = await _say(orch, "I like jazz.")

    rows = [m for m in state.transcript if m.sender_id == SCOUT.id]
    assert [r.content for r in rows] == ["First.", "Second."]
    [request] = llm.extraction_requests
    span = str(request.messages[1].content)
    assert "[Scout (you)] First." in span and "[Scout (you)] Second." in span
    [fact] = memory.list_facts()
    assert fact.source_turn_id == rows[-1].turn_id
    assert rows[-1].knowledge_learned == (fact.fact_id,)
    assert rows[0].knowledge_learned == ()


async def test_learning_is_announced_on_the_rooms_knowledge_topic(tmp_path: Path) -> None:
    """
    Killed by: src/uclone_x/room/orchestrator.py :: topic=f"room.{outcome.room_id}.knowledge",
    Becomes: topic=f"room.{outcome.room_id}",
    """
    memory = _memory(tmp_path)
    llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    async with EventBus() as bus:
        sub = bus.subscribe({"room.r1.knowledge"})
        orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))), bus=bus)
        await _say(orch, "I like jazz.")
        event: AgentEvent = await asyncio.wait_for(sub.get(), 5)

    assert event.type is EventType.KNOWLEDGE_UPDATED
    assert event.payload["agent_id"] == SCOUT.id
    assert event.payload["added"] == 1
    assert event.payload["failed"] is False


async def test_a_seat_with_no_memory_learns_nothing_and_is_not_asked(tmp_path: Path) -> None:
    llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, None))))

    state = await _say(orch, "I like jazz.")

    assert llm.extraction_requests == []
    assert _row(state, SCOUT.id).knowledge_learned == ()


async def test_a_failed_turn_is_not_learned_from(tmp_path: Path) -> None:
    """Only a committed turn is queued.

    Killed by: src/uclone_x/room/orchestrator.py :: if committed:  # a failed turn teaches nothing
    Becomes: if True:
    """

    class _Failing(_Model):
        async def generate(self, request: LLMRequest) -> ModelResponse:
            system = str(request.messages[0].content) if request.messages else ""
            if not system.startswith(_EXTRACTION_MARK):
                raise LLMProviderError("down")
            return await super().generate(request)

    memory = _memory(tmp_path)
    llm = _Failing(extractions=[[_fact("user", "likes", "jazz")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    state = await _say(orch, "I like jazz.")

    assert _row(state, SCOUT.id).error is not None
    assert llm.extraction_requests == []
    assert memory.list_facts() == []


def test_fenced_and_wrapped_answers_are_read() -> None:
    """A model that fences its JSON, or wraps the list in an object, is still read."""
    from uclone_x.memory.extractor import _parse  # pyright: ignore[reportPrivateUsage]

    fenced = "```json\n" + json.dumps([_fact("user", "likes", "jazz")]) + "\n```"
    wrapped = json.dumps({"facts": [_fact("user", "likes", "tea"), "junk", {"subject": ""}]})
    assert [c.value for c in _parse(fenced)] == ["jazz"]
    assert [c.value for c in _parse(wrapped)] == ["tea"]
