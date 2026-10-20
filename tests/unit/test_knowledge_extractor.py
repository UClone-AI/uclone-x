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
from pydantic import BaseModel

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    ToolExecutionRecord,
)
from uclone_x.agent.session import SessionStore
from uclone_x.core.models import PersonaDefinition
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
    ToolCallRequest,
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
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
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
    subject: str, relation: str, value: str, quote: str | None = None, **extra: Any
) -> dict[str, Any]:
    """One fact as the extraction prompt asks for it; its quote is its value unless given."""
    item: dict[str, Any] = {
        "subject": subject,
        "key": relation,
        "value": value,
        "quote": value if quote is None else quote,
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
            content = answer if isinstance(answer, str) else json.dumps({"facts": answer})
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


def _seat(
    participant: Participant,
    llm: _Model,
    memory: CrossSessionMemory | None,
    *,
    learns: bool | None = None,
) -> BaseAgent:
    """A seat for `participant`. `learns` gives it a persona with that switch (#1840);
    `None` gives it none, as most seats here have."""
    personas = (
        ()
        if learns is None
        else (
            PersonaDefinition(
                name=participant.id,
                role="Helper",
                system_prompt="Help.",
                learn_from_conversations=learns,
            ),
        )
    )
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
        personas=personas,
        persona=participant.id if personas else None,
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

    Killed by: src/uclone_x/memory/extractor.py :: grounds = [line for line in lesson.lines if line.kind in _SOURCE_ORIGIN]
    Becomes: grounds = list(lesson.lines)
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
    """A fact claimed from a tool result, in a turn that ran no tool, has nothing under it:
    its words are in no line of the span (#2235).

    Killed by: src/uclone_x/knowledge/extraction.py :: if not quote_found(fact.quote, text):
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
    Killed by: src/uclone_x/memory/extractor.py :: if item.get("durable") is not True:
    Becomes: if False:
    Killed by: src/uclone_x/memory/extractor.py :: return confidence >= MIN_CONFIDENCE
    Becomes: return True
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
    Killed by: src/uclone_x/memory/store.py :: metadata = {**current.metadata, "observations": [*observations, turn_id]}
    Becomes: metadata = {**current.metadata, "observations": observations}
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
    Killed by: src/uclone_x/memory/extractor.py :: replaces=tuple(f.fact_id for f in replaced),
    Becomes: replaces=(),
    """
    memory = _memory(tmp_path)
    old = memory.record_fact("user", "lives_in", "Busan", _PROV, "s0", origin="told")
    llm = _Model(extractions=[[_fact("user", "lives_in", "Seoul", replaces="Busan")]])
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

    Killed by: src/uclone_x/memory/extractor.py :: corrected = [f for f in (*active, *replaced) if f.origin == "corrected"]
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


async def _learn(
    memory: CrossSessionMemory,
    said: str,
    facts: list[dict[str, Any]],
    functional: frozenset[str] = frozenset(),
) -> Any:
    """What `extract` does with the person saying `said` and the model answering `facts`."""

    async def generate(request: LLMRequest) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content=json.dumps({"facts": facts}),
            tool_calls=(),
            usage=_USAGE,
            provenance=_PROV,
        )

    lesson = Lesson(
        clone_id="scout",
        clone_name="Scout",
        room_id="r1",
        session_id="s1",
        turn_id="t1",
        lines=(SpanLine("person", "Alice", said),),
        memory=memory,
        generate=generate,
        person_names=("Alice",),
        functional=lambda: functional,
    )
    return await KnowledgeExtractor().extract(lesson)


@pytest.mark.parametrize("replaces", [None, "", "Rock"], ids=["null", "empty", "not-held"])
async def test_a_value_that_replaces_nothing_held_supersedes_nothing(
    tmp_path: Path, replaces: str | None
) -> None:
    """ "Prefers jazz" held, "prefers Korean" learned: both stay, and nothing is reported
    replaced (#2224). Only a held value the model names is replaced.

    Killed by: src/uclone_x/memory/extractor.py :: single_valued=False,
    Becomes: single_valued=True,
    """
    memory = _memory(tmp_path)
    jazz = memory.record_fact(
        "user", "prefers", "jazz", _PROV, "s0", origin="told", single_valued=False
    )
    extra: dict[str, Any] = {} if replaces is None else {"replaces": replaces}

    outcome = await _learn(
        memory, "I like Korean food.", [_fact("user", "prefers", "Korean", **extra)]
    )

    [korean] = outcome.added
    assert outcome.superseded == ()
    assert {f.fact_id for f in memory.list_facts()} == {jazz.fact_id, korean}
    assert memory.get_fact(korean).single_valued is False  # type: ignore[union-attr]


async def test_the_held_value_the_model_names_is_replaced_under_any_relation(
    tmp_path: Path,
) -> None:
    """ "I stopped Rust; I'm on Zig now", filed under a key that holds nothing yet, the value
    given bare: Rust is replaced, and the other things held about the person stay.

    Killed by: src/uclone_x/memory/extractor.py :: replaces=_value_key(replaces) if isinstance(replaces, str) else "",
    Becomes: replaces="",
    Killed by: src/uclone_x/memory/extractor.py :: return named
    Becomes: return []
    Killed by: src/uclone_x/memory/extractor.py :: superseded.extend(f.fact_id for f in replaced)
    Becomes: pass
    """
    memory = _memory(tmp_path)
    rust = memory.record_fact("user", "learning", "Rust", _PROV, "s0", origin="told")
    jazz = memory.record_fact(
        "user", "prefers", "jazz", _PROV, "s0", origin="told", single_valued=False
    )

    outcome = await _learn(
        memory,
        "I've stopped learning Rust; I'm on Zig now.",
        [_fact("user", "current_language", "Zig", replaces=" rust ")],
    )

    [zig] = outcome.added
    assert outcome.superseded == (rust.fact_id,)
    assert {f.fact_id for f in memory.list_facts()} == {jazz.fact_id, zig}
    assert memory.get_fact(zig).contradicts_fact_id == rust.fact_id  # type: ignore[union-attr]


async def test_of_two_held_values_named_the_one_under_the_same_relation_is_replaced(
    tmp_path: Path,
) -> None:
    """ "Korean" is held as a preference and as the reply language; a new reply language
    replacing "Korean" replaces only the reply language.

    Killed by: src/uclone_x/memory/extractor.py :: return [f for f in named if relation_key(f.predicate) == candidate.relation]
    Becomes: return named
    """
    memory = _memory(tmp_path)
    food = memory.record_fact("user", "prefers", "Korean", _PROV, "s0", origin="told")
    reply = memory.record_fact("user", "reply_language", "Korean", _PROV, "s0", origin="told")

    outcome = await _learn(
        memory,
        "Please answer in English from now on.",
        [_fact("user", "reply_language", "English", replaces="Korean")],
    )

    assert outcome.superseded == (reply.fact_id,)
    assert food.fact_id in {f.fact_id for f in memory.list_facts()}


async def test_a_bare_value_reaches_another_relation_only_while_its_own_holds_nothing(
    tmp_path: Path,
) -> None:
    """ "Moved to Busan from Seoul" with "Seoul" bare: "likes Seoul" is not replaced while
    "lives_in" holds a value of its own (#2328).

    Killed by: src/uclone_x/memory/extractor.py :: if any(relation_key(f.predicate) == candidate.relation for f in live):
    Becomes: if False:
    """
    memory = _memory(tmp_path)
    likes = memory.record_fact("user", "likes", "Seoul", _PROV, "s0", origin="told")
    ulsan = memory.record_fact("user", "lives_in", "Ulsan", _PROV, "s0", origin="told")

    outcome = await _learn(
        memory,
        "I moved to Busan from Seoul.",
        [_fact("user", "lives_in", "Busan", quote="moved to Busan", replaces="Seoul")],
    )

    [busan] = outcome.added
    assert outcome.superseded == ()
    assert {f.fact_id for f in memory.list_facts()} == {likes.fact_id, ulsan.fact_id, busan}


async def test_under_a_functional_relation_a_new_value_replaces_every_current_one(
    tmp_path: Path,
) -> None:
    """A person declared `lives_in` one value at a time: Seoul replaces Busan and Ulsan
    though the model named neither, and `prefers` keeps both its values (#2332).

    Killed by: src/uclone_x/memory/extractor.py :: if candidate.relation in functional:
    Becomes: if False:
    Killed by: src/uclone_x/memory/extractor.py :: f for f in active if f.valid_until is None and f.fact_id not in named
    Becomes: f for f in active if f.fact_id not in named
    """
    memory = _memory(tmp_path)
    busan = memory.record_fact("user", "lives_in", "Busan", _PROV, "s0", origin="told")
    ulsan = memory.record_fact(
        "user", "lives_in", "Ulsan", _PROV, "s0", origin="told", single_valued=False
    )
    then = memory.record_fact(
        "user",
        "lives_in",
        "Daegu",
        _PROV,
        "s0",
        origin="told",
        single_valued=False,
        valid_until="2020-01-01T00:00:00Z",
    )
    jazz = memory.record_fact(
        "user", "prefers", "jazz", _PROV, "s0", origin="told", single_valued=False
    )

    outcome = await _learn(
        memory,
        "I live in Seoul and I like Korean food.",
        [_fact("user", "lives_in", "Seoul"), _fact("user", "prefers", "Korean")],
        functional=frozenset({"lives_in"}),
    )

    seoul, korean = outcome.added
    assert set(outcome.superseded) == {busan.fact_id, ulsan.fact_id}
    assert {f.fact_id for f in memory.list_facts()} == {then.fact_id, jazz.fact_id, seoul, korean}


async def test_a_learned_fact_keeps_the_words_it_was_learned_from(tmp_path: Path) -> None:
    """Remembers' history shows them (#2332).

    Killed by: src/uclone_x/memory/extractor.py :: "quote": candidate.quote,
    Becomes: "quote": "",
    """
    memory = _memory(tmp_path)

    outcome = await _learn(
        memory,
        "I moved to Busan last spring.",
        [_fact("user", "lives_in", "Busan", quote="moved to Busan")],
    )

    [busan] = outcome.added
    assert memory.get_fact(busan).metadata["quote"] == "moved to Busan"  # type: ignore[union-attr]


async def test_a_value_replacing_a_corrected_one_under_another_relation_is_dropped(
    tmp_path: Path,
) -> None:
    """A person corrected "lives_in"; the model filing a move under "location" and naming
    the corrected value does not replace it.

    Killed by: src/uclone_x/memory/extractor.py :: corrected = [f for f in (*active, *replaced) if f.origin == "corrected"]
    Becomes: corrected = [f for f in active if f.origin == "corrected"]
    """
    memory = _memory(tmp_path)
    busan = memory.record_fact("user", "lives_in", "Busan", _PROV, "s0", origin="told")
    daegu = memory.correct_fact(busan.fact_id, "Daegu", _PROV)

    outcome = await _learn(
        memory, "I live in Ulsan now.", [_fact("user", "location", "Ulsan", replaces="Daegu")]
    )

    assert outcome.added == ()
    assert [f.fact_id for f in memory.list_facts()] == [daegu.fact_id]


def test_the_model_is_asked_which_held_value_a_fact_replaces() -> None:
    """The extraction prompt asks for "replaces", and when to leave it null (#2224).

    Killed by: src/uclone_x/memory/extractor.py :: '"durable" (true or false), "replaces" (only when the line says a fact already held '
    Becomes: '"durable" (true or false), (only when the line says a fact already held '
    """
    from uclone_x.memory.extractor import _instructions  # pyright: ignore[reportPrivateUsage]

    told = _instructions(_lesson(_SAID))

    assert '"replaces" (only when the line says a fact already held' in told
    assert "otherwise null" in told


#: "José" as a build before #1895 could store it (`e` and a combining accent); the extractor
#: saves the composed form (`person_subject`), so the two meet only when compared folded.
_JOSE_DECOMPOSED = "Jose\u0301"
_JOSE_COMPOSED = "Jos\u00e9"


async def test_a_fact_stored_decomposed_is_superseded_by_the_composed_extraction(
    tmp_path: Path,
) -> None:
    """The old subject spelling is still the same subject, so the new value replaces it (#1893).

    The composed name resolves to the subject the clone holds (#2235), so the fact is filed
    under the spelling already held.

    Killed by: src/uclone_x/memory/extractor.py :: *(Entity(s, scope, "subject", s) for s in sorted(held) if name_key(s) not in taken),
    Becomes: *(),
    """
    memory = _memory(tmp_path)
    old = memory.record_fact(_JOSE_DECOMPOSED, "lives_in", "Busan", _PROV, "s0", origin="told")
    llm = _Model(extractions=[[_fact(_JOSE_COMPOSED, "lives_in", "Seoul", replaces="Busan")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))

    await _say(orch, f"{_JOSE_COMPOSED} moved to Seoul last month.")

    [new] = memory.list_facts()
    assert new.subject == _JOSE_DECOMPOSED
    assert new.object_value == "Seoul"
    assert new.contradicts_fact_id == old.fact_id


async def test_a_fact_stored_decomposed_is_shown_when_the_span_names_it_composed(
    tmp_path: Path,
) -> None:
    """The extraction prompt lists what the clone already holds about a named subject (#1899).

    Killed by: src/uclone_x/memory/extractor.py :: if (subject := fold_name(fact.subject)) in (PERSON_SUBJECT, SELF_SUBJECT) or subject in text
    Becomes: if (subject := fact.subject.casefold()) in (PERSON_SUBJECT, SELF_SUBJECT) or subject in text
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
    Killed by: src/uclone_x/memory/extractor.py :: raise ExtractionFailed(str(exc)) from exc
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
        raise OSError(28, "No space left on device", str(tmp_path / "scout-knowledge.sqlite3"))

    llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory))))
    await orch.post("r1", ALICE.id, "I like jazz.")
    monkeypatch.setattr(memory, "record_fact", refuse)
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
    quiet = RoomMessage(seq=1, sender_id="scout", content="Hello.", turn_id="t1")
    taught = RoomMessage(
        seq=2, sender_id="scout", content="Noted.", turn_id="t2", knowledge_learned=("mem_a",)
    )
    failed = RoomMessage(
        seq=3,
        sender_id="scout",
        content="Hm.",
        turn_id="t3",
        knowledge_extract_error=EXTRACTION_FAILED,
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


# --------------------------------------------------------------------------------------
# "Learn from conversations", one switch per clone (#1840, design §3.3 "Opting out").
# --------------------------------------------------------------------------------------


async def test_a_clone_switched_off_is_not_asked_what_to_remember(tmp_path: Path) -> None:
    """Off stops the extraction for that clone: no model call, nothing saved, no row line.

    Killed by: src/uclone_x/room/orchestrator.py :: if persona is not None and not persona.learn_from_conversations:
    Becomes: if False:
    """
    llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    memory = _memory(tmp_path)
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory, learns=False))))

    state = await _say(orch, "I like jazz.")

    assert llm.extraction_requests == []
    assert memory.list_facts() == []
    assert _row(state, SCOUT.id).knowledge_learned == ()


async def test_a_clone_switched_on_still_learns(tmp_path: Path) -> None:
    """The switch on is the default behaviour, not a second way of turning learning off."""
    llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    memory = _memory(tmp_path)
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory, learns=True))))

    await _say(orch, "I like jazz.")

    assert [(f.subject, f.object_value) for f in memory.list_facts()] == [("user", "jazz")]


async def test_switching_one_clone_off_leaves_the_other_in_the_room_learning(
    tmp_path: Path,
) -> None:
    """The switch is the clone's, not the room's: the other seat is asked and saves."""
    scout_llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    sage_llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    scout_memory, sage_memory = _memory(tmp_path, "scout"), _memory(tmp_path, "sage")
    orch = _room(
        tmp_path,
        _Seats(
            (SCOUT, _seat(SCOUT, scout_llm, scout_memory, learns=False)),
            (SAGE, _seat(SAGE, sage_llm, sage_memory)),
        ),
        SCOUT.id,
        SAGE.id,
    )

    await _say(orch, "I like jazz.")

    assert scout_llm.extraction_requests == []
    assert scout_memory.list_facts() == []
    assert len(sage_llm.extraction_requests) == 1
    assert [f.object_value for f in sage_memory.list_facts()] == ["jazz"]


def _switched_off(participant: Participant) -> PersonaDefinition:
    """What saving the clone with the switch off hands its live seat (`stage_persona_edit`)."""
    return PersonaDefinition(
        name=participant.id,
        role="Helper",
        system_prompt="Help.",
        learn_from_conversations=False,
    )


async def test_a_lesson_queued_before_the_switch_is_turned_off_is_not_learned(
    tmp_path: Path,
) -> None:
    """Queued, then switched off, then drained: nothing is asked and nothing is saved (#2228).

    The seat takes the saved edit only at its next turn, so the switch is read from the edit
    it holds, not only from the persona in force.

    Killed by: src/uclone_x/memory/extractor.py :: if not lesson.learns():  # switched off since the lesson was queued
    Becomes: if False:  # switched off since the lesson was queued
    """
    memory = _memory(tmp_path)
    llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    seat = _seat(SCOUT, llm, memory, learns=True)
    orch = _room(tmp_path, _Seats((SCOUT, seat)))
    floor = orch._floor.setdefault("r1", asyncio.Lock())  # pyright: ignore[reportPrivateUsage]

    await orch.post("r1", ALICE.id, "I like jazz.")
    await floor.acquire()  # the lesson is queued and waits for the floor
    learning = asyncio.create_task(orch.wait_for_learning("r1"))
    seat.stage_persona_edit(_switched_off(SCOUT))
    floor.release()
    await learning

    assert llm.extraction_requests == []
    assert memory.list_facts() == []


async def test_turning_the_switch_off_while_the_model_is_asked_saves_nothing(
    tmp_path: Path,
) -> None:
    """The answer of an extraction call already out when learning is turned off is dropped.

    Killed by: src/uclone_x/memory/extractor.py :: if not lesson.learns():  # turned off while the model was asked
    Becomes: if False:  # turned off while the model was asked
    """
    memory = _memory(tmp_path)
    llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    seat = _seat(SCOUT, llm, memory, learns=True)
    orch = _room(tmp_path, _Seats((SCOUT, seat)))
    llm.hold = asyncio.Event()

    await orch.post("r1", ALICE.id, "I like jazz.")
    learning = asyncio.create_task(orch.wait_for_learning("r1"))
    for _ in range(50):
        if llm.extraction_requests:
            break
        await asyncio.sleep(0)
    assert len(llm.extraction_requests) == 1  # the call is out, and paused
    seat.stage_persona_edit(_switched_off(SCOUT))
    llm.hold.set()
    await learning

    assert memory.list_facts() == []
    assert _row(orch._require_room("r1"), SCOUT.id).knowledge_learned == ()  # pyright: ignore[reportPrivateUsage]


def _persona(name: str, *, learns: bool) -> PersonaDefinition:
    return PersonaDefinition(
        name=name, role="Helper", system_prompt="Help.", learn_from_conversations=learns
    )


def test_the_latest_persona_is_the_one_in_force_when_no_edit_is_held(tmp_path: Path) -> None:
    seat = _seat(SCOUT, _Model(), _memory(tmp_path), learns=True)

    latest = seat.latest_persona_definition

    assert latest is not None and latest.learn_from_conversations is True
    assert latest == seat.persona_definition


def test_the_latest_persona_is_a_held_edit_of_the_persona_the_seat_speaks_as(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/base.py :: if staged is not None and staged.name == self._persona:
    Becomes: if False:
    """
    seat = _seat(SCOUT, _Model(), _memory(tmp_path), learns=True)
    seat.stage_persona_edit(_persona(SCOUT.id, learns=False))

    latest = seat.latest_persona_definition

    assert latest is not None and latest.learn_from_conversations is False
    applied = seat.persona_definition
    assert applied is not None and applied.learn_from_conversations is True  # not yet taken


def test_a_held_edit_of_a_persona_the_seat_no_longer_speaks_as_is_not_the_latest(
    tmp_path: Path,
) -> None:
    """An edit held for the seat's old persona says nothing about the one it speaks as now
    (#2233): `stage_persona_edit` is keyed by the persona the seat had when it was saved.

    Killed by: src/uclone_x/agent/base.py :: if staged is not None and staged.name == self._persona:
    Becomes: if staged is not None:
    """
    seat = _seat(SCOUT, _Model(), _memory(tmp_path), learns=True)
    seat.define_persona(_persona("other", learns=True))
    seat.stage_persona_edit(_persona(SCOUT.id, learns=False))
    seat.persona = "other"

    latest = seat.latest_persona_definition

    assert latest is not None and latest.name == "other"
    assert latest.learn_from_conversations is True


class _ReplacingSeats(_Seats):
    """A resolver that can put a new seat in a participant's place, as a rebuilt one would."""

    def built_seat(self, participant: Participant) -> BaseAgent | None:
        return self.agents.get(participant.id)


async def test_a_queued_lesson_reads_the_switch_from_the_seat_in_place_when_it_is_learned(
    tmp_path: Path,
) -> None:
    """The switch is read from the participant's current seat, not the one the lesson was
    queued from, which a rebuilt seat no longer is (#2233).

    Killed by: src/uclone_x/room/orchestrator.py :: current = lookup(speaker) if callable(lookup) else None
    Becomes: current = None
    """
    memory = _memory(tmp_path)
    llm = _Model(extractions=[[_fact("user", "likes", "jazz")]])
    seats = _ReplacingSeats((SCOUT, _seat(SCOUT, llm, memory, learns=True)))
    orch = _room(tmp_path, seats)
    floor = orch._floor.setdefault("r1", asyncio.Lock())  # pyright: ignore[reportPrivateUsage]

    await orch.post("r1", ALICE.id, "I like jazz.")
    await floor.acquire()  # the lesson is queued and waits for the floor
    learning = asyncio.create_task(orch.wait_for_learning("r1"))
    seats.agents[SCOUT.id] = _seat(SCOUT, _Model(), memory, learns=False)
    floor.release()
    await learning

    assert llm.extraction_requests == []
    assert memory.list_facts() == []


class _SavesWhenAsked(_Model):
    """Answers the first reply request with a `record_memory_fact` call, then replies."""

    def __init__(self) -> None:
        super().__init__(reply="Saved.")
        self._asked = False

    async def generate(self, request: LLMRequest) -> ModelResponse:
        system = str(request.messages[0].content) if request.messages else ""
        if self._asked or system.startswith(_EXTRACTION_MARK):
            return await super().generate(request)
        self._asked = True
        self.replies.append(request)
        save = ToolCallRequest(
            id="call_save",
            name="record_memory_fact",
            arguments={"subject": "user", "predicate": "editor", "object_value": "helix"},
        )
        return ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            content="",
            tool_calls=(save,),
            usage=_USAGE,
            provenance=_PROV,
        )


async def test_a_clone_switched_off_still_saves_corrects_and_forgets(tmp_path: Path) -> None:
    """Off stops only the automatic extraction. Being asked to remember still saves, and a
    person's Correct and Forget still change what the clone knows.

    An "off" built by withholding the seat's memory would pass the tests above and fail here.
    """
    llm = _SavesWhenAsked()
    memory = _memory(tmp_path)
    orch = _room(tmp_path, _Seats((SCOUT, _seat(SCOUT, llm, memory, learns=False))))

    await _say(orch, "Remember that I use helix.")

    assert llm.extraction_requests == []
    (saved,) = memory.list_facts()
    assert (saved.subject, saved.predicate, saved.object_value) == ("user", "editor", "helix")

    person = Provenance.primary("person")
    corrected = memory.correct_fact(saved.fact_id, "vim", person)
    assert [f.object_value for f in memory.list_facts()] == ["vim"]
    memory.retract_fact(corrected.fact_id, "forgotten by the user", person)
    assert memory.list_facts() == []


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


def _lesson(
    *lines: SpanLine,
    memory: CrossSessionMemory | None = None,
    person_names: tuple[str, ...] = ("Alice",),
    clone_names: tuple[str, ...] = ("Scout",),
) -> Lesson:
    return Lesson(
        clone_id="scout",
        clone_name="Scout",
        room_id="r1",
        session_id="s1",
        turn_id="t1",
        lines=lines,
        memory=memory if memory is not None else CrossSessionMemory(),
        generate=_never,
        person_names=person_names,
        clone_names=clone_names,
    )


def _read_reply(content: str, lesson: Lesson) -> list[tuple[str, str, str, str]]:
    from uclone_x.memory.extractor import _read  # pyright: ignore[reportPrivateUsage]

    return [(c.subject, c.relation, c.value, c.source) for c in _read(content, lesson)]


_SAID = SpanLine("person", "Alice", "I like jazz and my sister Mina likes tea.")


def test_fenced_and_wrapped_answers_are_read() -> None:
    """A model that fences its JSON, answers a bare array, or puts junk in the list, is still
    read: the reply is read as every vessel's is (`knowledge.extraction.listed_facts`)."""
    lesson = _lesson(_SAID)
    fenced = "```json\n" + json.dumps({"facts": [_fact("user", "likes", "jazz")]}) + "\n```"
    bare = json.dumps([_fact("user", "likes", "jazz")])
    wrapped = json.dumps({"facts": [_fact("user", "likes", "jazz"), "junk", {"subject": ""}]})
    for content in (fenced, bare, wrapped):
        assert _read_reply(content, lesson) == [("user", "likes", "jazz", "person")]


# --------------------------------------------------------------------------------------
# Read and verified as every vessel's facts are (uGraph step 4, #2235).
# --------------------------------------------------------------------------------------


def test_a_reply_cut_off_by_its_token_cap_keeps_the_facts_it_finished() -> None:
    """A model that ran to its cap wrote one whole fact before the cut; it is kept, typed
    fields and all, and the half-written one is not.

    Killed by: src/uclone_x/knowledge/extraction.py :: if salvaged:
    Becomes: if False:
    """
    whole = json.dumps(_fact("user", "likes", "jazz"))
    cut = '{"facts": [' + whole + ', {"subject": "Mina", "key": "likes", "val'
    assert _read_reply(cut, _lesson(_SAID)) == [("user", "likes", "jazz", "person")]


def test_a_quote_that_is_not_in_a_person_or_tool_line_is_dropped() -> None:
    """The clone said it, or the model paraphrased: neither grounds a fact (#2235).

    Killed by: src/uclone_x/memory/extractor.py :: grounds = [line for line in lesson.lines if line.kind in _SOURCE_ORIGIN]
    Becomes: grounds = list(lesson.lines)
    """
    lesson = _lesson(_SAID, SpanLine("self", "Scout", "You also love opera, I think."))
    reply = json.dumps(
        {
            "facts": [
                _fact("user", "loves", "opera", quote="You also love opera"),
                _fact("user", "enjoys", "jazz music", quote="she enjoys jazz music"),
                _fact("user", "likes", "jazz", quote="I like jazz"),
            ]
        }
    )
    assert _read_reply(reply, lesson) == [("user", "likes", "jazz", "person")]


def test_a_name_two_subjects_share_is_dropped_not_guessed() -> None:
    """The clone holds two subjects that fold to one name ("Mina" and "MINA", filed before
    names were folded): a fact about "mina" is filed under neither.

    Killed by: src/uclone_x/knowledge/extraction.py :: return None, "ambiguous_name"  # nothing is guessed: not the first, not any
    Becomes: return ids[0], None
    """
    memory = CrossSessionMemory()
    memory.record_fact("Mina", "is", "a friend", _PROV, "s0")
    memory.record_fact("MINA", "works_at", "Acme", _PROV, "s0")
    lesson = _lesson(SpanLine("person", "Alice", "mina likes jazz."), memory=memory)
    reply = json.dumps({"facts": [_fact("mina", "likes", "jazz", quote="mina likes jazz")]})
    assert _read_reply(reply, lesson) == []


def test_a_held_subject_is_named_as_held_and_a_new_name_is_a_new_subject() -> None:
    """The clone's names policy (§3.4): a name it holds, written another way, is that
    subject; a name it does not hold is filed as written; a near name is not merged.

    Killed by: src/uclone_x/memory/extractor.py :: *(Entity(s, scope, "subject", s) for s in sorted(held) if name_key(s) not in taken),
    Becomes: *(),
    Killed by: src/uclone_x/knowledge/extraction.py :: if new_names:
    Becomes: if False:
    """
    memory = CrossSessionMemory()
    memory.record_fact("Mina Park", "is", "Alice's sister", _PROV, "s0")
    lesson = _lesson(
        SpanLine("person", "Alice", "mina park likes tea, and Mina's friend Jo likes jazz."),
        memory=memory,
    )
    reply = json.dumps(
        {
            "facts": [
                _fact("mina  park", "likes", "tea", quote="mina park likes tea"),
                _fact("Jo", "likes", "jazz", quote="Jo likes jazz"),
                _fact("Mina", "has_friend", "Jo", quote="Mina's friend Jo"),
            ]
        }
    )
    assert _read_reply(reply, lesson) == [
        ("Mina Park", "likes", "tea", "person"),
        ("Jo", "likes", "jazz", "person"),
        ("Mina", "has_friend", "Jo", "person"),
    ]


def test_a_fact_given_twice_is_kept_once() -> None:
    """The same fact twice in one reply, the second with its relation spelled with a space,
    is one fact, not two to file.

    Killed by: src/uclone_x/memory/extractor.py :: facts = [replace(f, key=relation_key(f.key)) for f in facts]
    Becomes: facts = facts
    """
    reply = json.dumps(
        {
            "facts": [
                _fact("user", "likes", "jazz", quote="I like jazz"),
                _fact("user", "Likes", "Jazz", quote="I like jazz"),
                _fact("Mina", "likes tea", "yes", quote="Mina likes tea"),
                _fact("Mina", "likes_tea", "yes", quote="Mina likes tea"),
            ]
        }
    )
    assert _read_reply(reply, _lesson(_SAID)) == [
        ("user", "likes", "jazz", "person"),
        ("Mina", "likes_tea", "yes", "person"),
    ]


def test_a_tool_result_grounds_a_found_fact_and_the_person_comes_first() -> None:
    """A quote in a tool result is `tool`; one in both the person's line and a tool's is the
    person's.

    Killed by: src/uclone_x/memory/extractor.py :: for line in sorted(grounds, key=lambda line: line.kind != "person"):
    Becomes: for line in grounds:
    """
    lesson = _lesson(
        SpanLine("tool", "web_search", "Busan is the second largest city. Alice lives in Busan."),
        SpanLine("person", "Alice", "Yes, Alice lives in Busan."),
    )
    reply = json.dumps(
        {
            "facts": [
                _fact("Busan", "is", "the second largest city", quote="second largest city"),
                _fact("user", "lives_in", "Busan", quote="Alice lives in Busan"),
            ]
        }
    )
    assert _read_reply(reply, lesson) == [
        ("Busan", "is", "the second largest city", "tool"),
        ("user", "lives_in", "Busan", "person"),
    ]


def test_generate_image_results_are_excluded_from_lesson_lines() -> None:
    # Killed by: src/uclone_x/memory/extractor.py :: "generate_image",
    # Becomes: "unused_generate_image",
    from uclone_x.room.orchestrator import _lesson_lines  # pyright: ignore[reportPrivateUsage]

    state = RoomState(room_id="r1", participants=(ALICE, SCOUT))
    rec_img = ToolExecutionRecord(
        tool_name="generate_image",
        output={"status": "success", "image_path": "/tmp/img.png"},
    )
    rec_comfy = ToolExecutionRecord(
        tool_name="comfy_image_gen",
        output={"status": "success", "image_path": "/tmp/comfy.png"},
    )
    lines = _lesson_lines(
        state,
        speaker=SCOUT,
        since=0,
        rendered_through=0,
        content="Here is the picture.",
        executions=(rec_img, rec_comfy),
    )
    assert not any(line.kind == "tool" for line in lines)


async def _never(_request: LLMRequest) -> ModelResponse:
    raise AssertionError("must not be called")


def test_self_fact_from_tool_source_is_dropped_by_curate() -> None:
    # Killed by: src/uclone_x/memory/extractor.py :: if candidate.subject == SELF_SUBJECT and candidate.source != "person":
    # Becomes: if False:
    from uclone_x.memory.extractor import _Candidate, _curate  # pyright: ignore[reportPrivateUsage]

    candidate = _Candidate(
        subject="self",
        relation="hair",
        value="blonde",
        source="tool",
        confidence=0.9,
    )
    lesson = Lesson(
        clone_id="fiona",
        clone_name="Fiona",
        room_id="r1",
        session_id="s1",
        turn_id="t1",
        lines=(),
        memory=CrossSessionMemory(),
        generate=_never,
        clone_names=("Fiona",),
    )
    kept = _curate([candidate], lesson)
    assert kept == []


async def test_a_clone_turn_whose_generate_image_arguments_describe_its_looks_yields_no_self_fact(
    tmp_path: Path,
) -> None:
    fiona = Participant(
        id="fiona",
        kind=ParticipantKind.AGENT,
        display_name="Fiona",
        session_id="sess_room__r1__fiona",
    )

    class _Params(BaseModel):
        prompt: str = ""

    class _Draw(BaseTool[_Params]):
        name = "generate_image"
        description = "draws"

        def run(self, params: _Params, context: ToolContext) -> dict[str, Any]:
            return {"status": "success", "image_path": "/tmp/fiona.png"}

    tools = ToolRegistry()
    tools.register(_Draw())

    class _DrawingModel(_Model):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._call_made = False

        async def generate(self, request: LLMRequest) -> ModelResponse:
            system = str(request.messages[0].content) if request.messages else ""
            if system.startswith(_EXTRACTION_MARK):
                return await super().generate(request)
            self.replies.append(request)
            if not self._call_made:
                self._call_made = True
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    content="",
                    tool_calls=(
                        ToolCallRequest(
                            id="call_img",
                            name="generate_image",
                            arguments={
                                "prompt": "A portrait of Fiona, 25 years old, with blonde hair"
                            },
                        ),
                    ),
                    usage=_USAGE,
                    provenance=_PROV,
                )
            return ModelResponse(
                finish_reason=FinishReason.STOP,
                content="Here is a picture of me.",
                tool_calls=(),
                usage=_USAGE,
                provenance=_PROV,
            )

    memory = _memory(tmp_path, "fiona")
    llm = _DrawingModel(
        extractions=[
            [
                _fact("self", "hair", "blonde", source="tool"),
                _fact("Fiona", "age", "25", source="tool"),
            ]
        ]
    )

    agent = BaseAgent(
        config=AgentConfig(
            agent_id=fiona.id,
            name=fiona.display_name,
            llm_config=AgentLLMConfig(model_name="scripted"),
        ),
        llm=llm,
        tools=tools,
        context=AgentContext(session_id=fiona.session_id, agent_id=fiona.id),
        memory=memory,
        store=SessionStore(_SESSIONS[0] / fiona.id),
    )

    store = RoomStore(tmp_path / "rooms")
    store.save(
        RoomState(
            room_id="r1",
            participants=(ALICE, fiona),
            policy=RoomPolicy(max_agent_turns_per_human_message=1),
        )
    )
    orch = RoomOrchestrator(
        store=store,
        selectors=[_InOrder(fiona.id)],
        resolver=_Seats((fiona, agent)),
        extractor=KnowledgeExtractor(),
    )

    state = await _say(orch, "Draw your looks")

    [req] = llm.extraction_requests
    span = str(req.messages[1].content)
    assert "generate_image" not in span
    assert "blonde hair" not in span
    assert "25 years old" not in span

    assert memory.list_facts(subject="self") == []
    assert memory.list_facts() == []

    row = _row(state, fiona.id)
    assert row.knowledge_learned == ()


def test_a_durable_copy_of_a_fact_is_kept_though_a_later_copy_is_not_durable() -> None:
    """The same fact twice, the later copy marked not durable: the durable one is kept. The
    gates ran after the duplicate was folded away, which kept the later copy and then lost
    the fact to it (#2240 item 3).

    Killed by: src/uclone_x/memory/extractor.py :: items = [item for item in items if _passes_gates(item)]
    Becomes: items = items
    """
    reply = json.dumps(
        {
            "facts": [
                _fact("user", "likes", "jazz", quote="I like jazz"),
                _fact("user", "likes", "jazz", quote="I like jazz", durable=False, confidence=0.6),
                _fact("Mina", "likes", "tea", quote="Mina likes tea", confidence=0.2),
                _fact("Mina", "likes", "tea", quote="Mina likes tea"),
            ]
        }
    )
    from uclone_x.memory.extractor import _read  # pyright: ignore[reportPrivateUsage]

    kept = _read(reply, _lesson(_SAID))
    assert [(c.subject, c.relation, c.value, c.confidence) for c in kept] == [
        ("user", "likes", "jazz", 0.95),  # the durable copy, not the later one
        ("Mina", "likes", "tea", 0.95),  # the sure copy, not the earlier one
    ]


def test_a_name_the_person_and_the_clone_share_is_the_persons() -> None:
    """Decided (#2240 item 4): as `fact_subject` files it for `record_memory_fact`, the person
    is read first, so a fact about the person never becomes the clone's. Dropping it as
    ambiguous, as #2235 did, lost a fact the person told about themselves.

    Killed by: src/uclone_x/memory/extractor.py :: clone = tuple(n for n in (*SELF_WORDS, *lesson.clone_names) if name_key(n) not in person)
    Becomes: clone = (*SELF_WORDS, *lesson.clone_names)
    """
    lesson = _lesson(
        SpanLine("person", "Sam", "Sam likes jazz."), person_names=("Sam",), clone_names=("Sam",)
    )
    reply = json.dumps({"facts": [_fact("Sam", "likes", "jazz", quote="Sam likes jazz")]})
    assert _read_reply(reply, lesson) == [("user", "likes", "jazz", "person")]


def test_a_fact_that_does_not_say_it_is_durable_is_not_kept_and_one_at_the_threshold_is() -> None:
    """The gates' edges (#2272): a fact with no `durable` field is not durable, and a
    confidence of exactly `MIN_CONFIDENCE` is sure enough.

    Killed by: src/uclone_x/memory/extractor.py :: if item.get("durable") is not True:
    Becomes: if item.get("durable") is False:
    Killed by: src/uclone_x/memory/extractor.py :: return confidence >= MIN_CONFIDENCE
    Becomes: return confidence > MIN_CONFIDENCE
    """
    from uclone_x.memory.extractor import MIN_CONFIDENCE

    undecided = _fact("user", "likes", "jazz", quote="I like jazz")
    del undecided["durable"]
    reply = json.dumps(
        {
            "facts": [
                undecided,
                _fact("Mina", "likes", "tea", quote="Mina likes tea", confidence=MIN_CONFIDENCE),
            ]
        }
    )
    assert _read_reply(reply, _lesson(_SAID)) == [("Mina", "likes", "tea", "person")]
