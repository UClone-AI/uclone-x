"""A clone learns who it is: facts about itself, filed under `self` and always recalled (#2016).

The person defines the clone by talking to it ("너는 빨간 머리야", "나이는 24이야"). Those
statements are filed under the subject `self`, whatever word the model used for the clone,
and every turn's memory section carries them first.

Deterministic and offline: in-memory stores and scripted connectors. Test content is
appearance and personality only.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from tests.support.memory_seed import seed_fact
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.memory import extractor as extractor_module
from uclone_x.memory.extractor import KnowledgeExtractor, Lesson, SpanLine
from uclone_x.memory.models import (
    PERSON_SUBJECT,
    SELF_SUBJECT,
    MemoryFact,
    fact_subject,
)
from uclone_x.memory.recall import SELF_INTRO, recall_prompt_section
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.memory.tools import RecordMemoryFactParams, RecordMemoryFactTool
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
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

_PROV = Provenance.primary(provider="scripted", model="scripted")
_USAGE = TokenUsage(provider="scripted", model="scripted", input_tokens=0, output_tokens=0)


# --------------------------------------------------------------------------------------
# Folding: which subject a fact is filed under.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "written",
    ["너", "넌", "네가", "당신", "You", "yourself", "the clone", "Clone", "assistant", "SELF"],
)
def test_second_person_words_fold_to_self(written: str) -> None:
    """Korean and English second person, and the words for a clone, all mean the clone.

    Killed by: src/uclone_x/memory/models.py :: if fold_name(filed) in SELF_WORDS | names:
    Becomes: if fold_name(filed) in names:
    """
    assert fact_subject(written, ("u_kenny", "Kenny"), ("scout", "Scout")) == SELF_SUBJECT


@pytest.mark.parametrize("written", ["Scout", "scout", "  SCOUT ", "스카우트"])
def test_the_clones_own_names_fold_to_self(written: str) -> None:
    """The clone's id, display name and aliases are the clone.

    Killed by: src/uclone_x/memory/models.py :: names = {fold_name(name) for name in clone_names if name.strip()}
    Becomes: names = {fold_name(name) for name in () if name.strip()}
    """
    assert fact_subject(written, ("u_kenny",), ("scout", "Scout", "스카우트")) == SELF_SUBJECT


@pytest.mark.parametrize(
    ("written", "filed"),
    [
        ("user", PERSON_SUBJECT),
        ("me", PERSON_SUBJECT),
        ("the user", PERSON_SUBJECT),
        ("Kenny", PERSON_SUBJECT),
        ("Seoul", "Seoul"),
        ("Sage", "Sage"),
    ],
)
def test_the_persons_words_and_third_parties_do_not_become_self(written: str, filed: str) -> None:
    """The person's words keep folding to `user`; any other subject is kept as written."""
    assert fact_subject(written, ("u_kenny", "Kenny"), ("scout", "Scout")) == filed


def test_a_name_both_could_go_by_stays_the_persons() -> None:
    """The person is checked first: a shared name never moves a person's fact to the clone.

    Killed by: src/uclone_x/memory/models.py :: filed = person_subject(subject, person_names)
    Becomes: filed = person_subject(subject, ())
    """
    assert fact_subject("Kim", ("Kim",), ("Kim",)) == PERSON_SUBJECT


def _participant(pid: str, kind: ParticipantKind, name: str, *aliases: str) -> Participant:
    return Participant(id=pid, kind=kind, display_name=name, aliases=aliases)


def test_the_room_leaves_out_a_clone_name_another_participant_goes_by() -> None:
    """ "Kim" is one word of another clone's name here, so it does not mean this clone.

    Killed by: src/uclone_x/room/orchestrator.py :: _add_unshared_names(state, clone, names, set())
    Becomes: names.extend((clone.id, clone.display_name, *clone.aliases))
    """
    from uclone_x.room.orchestrator import (
        _clone_names,  # pyright: ignore[reportPrivateUsage]
    )

    me = _participant("scout", ParticipantKind.AGENT, "Scout", "Kim", "스카우트")
    state = RoomState(
        room_id="r1",
        participants=(
            _participant("alice", ParticipantKind.HUMAN, "Alice"),
            me,
            _participant("sage", ParticipantKind.AGENT, "Kim Sage"),
        ),
    )

    assert _clone_names(state, me) == ("scout", "스카우트")


# --------------------------------------------------------------------------------------
# The two write paths.
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_memory_fact_files_the_clone_under_self() -> None:
    """The model saving "you" or its own name lands in the self slot, as the extractor does.

    Killed by: src/uclone_x/memory/tools.py :: subject=fact_subject(params.subject, context.person_names, context.clone_names),
    Becomes: subject=fact_subject(params.subject, context.person_names, ()),
    """
    memory = CrossSessionMemory(max_facts_in_prompt=10)
    tool = RecordMemoryFactTool(memory)
    context = ToolContext(
        agent_id="scout", session_id="sess", person_names=("Kenny",), clone_names=("Rin",)
    )
    for subject, predicate, value in (
        ("Rin", "hair", "red"),
        ("you", "age", "24"),
        ("Kenny", "editor", "helix"),
    ):
        await tool.run(
            RecordMemoryFactParams(subject=subject, predicate=predicate, object_value=value),
            context,
        )

    assert sorted((f.subject, f.predicate) for f in memory.list_facts()) == [
        ("self", "age"),
        ("self", "hair"),
        ("user", "editor"),
    ]


class _ScriptedLLM(MockLLMConnector):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self._queue = list(responses)

    async def generate(self, request: LLMRequest) -> ModelResponse:
        if self._queue:
            return self._queue.pop(0)
        return await super().generate(request)


def _reply(text: str, calls: tuple[ToolCallRequest, ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=text,
        tool_calls=calls,
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        usage=TokenUsage(provider="mock"),
        provenance=Provenance.primary("mock"),
    )


@pytest.mark.asyncio
async def test_an_agents_turn_gives_the_tool_its_own_name() -> None:
    """The agent puts its configured name on the tool context, so saving under it is `self`.

    Killed by: src/uclone_x/agent/turn_executor.py :: clone_names=tuple(n for n in (self.agent_id, self._config.name) if n),
    Becomes: clone_names=(),
    """
    memory = CrossSessionMemory(max_facts_in_prompt=10)
    save = ToolCallRequest(
        id="call_1",
        name="record_memory_fact",
        arguments={"subject": "Rin", "predicate": "hair", "object_value": "red"},
    )
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="clone_rin",
            name="Rin",
            llm_config=AgentLLMConfig(model_name="mock", auto_compact=False),
        ),
        llm=_ScriptedLLM([_reply("", (save,)), _reply("Saved.")]),
        memory=memory,
    )

    await agent.execute_turn("remember: your hair is red")

    assert [(f.subject, f.predicate) for f in memory.list_facts()] == [("self", "hair")]


# --------------------------------------------------------------------------------------
# Recall: the self slot comes first, has its own limit, and never starves the person's.
# --------------------------------------------------------------------------------------


def _put(
    memory: CrossSessionMemory,
    fact_id: str,
    subject: str,
    predicate: str,
    value: str,
    *,
    created_at: str = "2026-09-20T00:00:00+00:00",
) -> None:
    seed_fact(
        memory,
        MemoryFact(
            fact_id=fact_id,
            subject=subject,
            predicate=predicate,
            object_value=value,
            provenance=_PROV,
            source_session_id="sess_a",
            confidence=0.8,
            created_at=created_at,
        ),
    )


def _fact_lines(section: str) -> list[str]:
    return [line for line in section.splitlines() if line.startswith("- ")]


@pytest.mark.asyncio
async def test_self_facts_lead_under_their_own_line_newest_first_and_capped_at_eight() -> None:
    """Up to eight facts about the clone, newest first, before everything else, once each.

    Killed by: src/uclone_x/memory/recall.py :: SELF_FACTS_LIMIT = 8
    Becomes: SELF_FACTS_LIMIT = 9
    Killed by: src/uclone_x/memory/recall.py :: lines.append(SELF_INTRO)
    Becomes: lines.append(RECALL_INTRO)
    Killed by: src/uclone_x/memory/recall.py :: chosen.update(fact.fact_id for fact in active if fold_name(fact.subject) == SELF_SUBJECT)
    Becomes: chosen.update(fact.fact_id for fact in active if fold_name(fact.subject) == "none")
    """
    memory = CrossSessionMemory(max_facts_in_prompt=10)
    for day in range(1, 10):
        _put(
            memory,
            f"mem_s{day}",
            "self",
            f"trait_{day}",
            f"hair colour {day}",
            created_at=f"2026-09-0{day}T00:00:00+00:00",
        )
    _put(memory, "mem_u", "user", "name", "Kenny")
    _put(memory, "mem_db", "database", "engine", "postgres")

    section = await recall_prompt_section(memory, "what hair colour 1")

    lines = section.splitlines()
    assert lines[1] == SELF_INTRO
    assert "persona description" in SELF_INTRO and "the fact wins" in SELF_INTRO
    assert _fact_lines(section) == [
        *(f"- self: trait_{day} -> hair colour {day}" for day in range(9, 1, -1)),
        "- user: name -> Kenny",
    ]
    # The oldest fact about the clone is not ranked back in, though the message names it.
    assert "(2 more facts not selected for this message" in section


@pytest.mark.asyncio
async def test_self_facts_do_not_count_against_the_prompt_budget() -> None:
    """Eight facts about the clone and a budget of five: the person still gets all five.

    Killed by: src/uclone_x/memory/recall.py :: about_user = _newest_about(about_others, USER_SUBJECT, min(USER_FACTS_LIMIT, limit))
    Becomes: about_user = _newest_about(about_others, USER_SUBJECT, min(USER_FACTS_LIMIT, limit - len(about_self)))
    """
    memory = CrossSessionMemory(max_facts_in_prompt=5)
    for day in range(1, 9):
        _put(memory, f"mem_s{day}", "self", f"trait_{day}", "kind")
    for day in range(1, 6):
        _put(memory, f"mem_u{day}", "user", f"likes_{day}", "tea")

    section = await recall_prompt_section(memory, "hello")

    lines = _fact_lines(section)
    assert sum(line.startswith("- self:") for line in lines) == 8
    assert sum(line.startswith("- user:") for line in lines) == 5


@pytest.mark.asyncio
async def test_a_changed_self_fact_leaves_one_active_and_recall_shows_only_it() -> None:
    """ "빨간 머리" then "아니 파란 머리야": the store supersedes, recall shows blue alone."""
    memory = CrossSessionMemory(max_facts_in_prompt=10)
    red = memory.record_fact("self", "hair", "빨간 머리", _PROV, "s0", origin="told")
    blue = memory.record_fact("self", "hair", "파란 머리", _PROV, "s0", origin="told")

    assert [f.fact_id for f in memory.list_facts(subject="self", predicate="hair")] == [
        blue.fact_id
    ]
    assert blue.contradicts_fact_id == red.fact_id
    section = await recall_prompt_section(memory, "hi")
    assert _fact_lines(section) == ["- self: hair -> 파란 머리"]


@pytest.mark.asyncio
async def test_without_self_facts_the_section_is_as_before() -> None:
    """A clone nobody has described gets no self line."""
    memory = CrossSessionMemory(max_facts_in_prompt=10)
    _put(memory, "mem_u", "user", "name", "Kenny")

    section = await recall_prompt_section(memory, "hi")

    assert SELF_INTRO not in section
    assert _fact_lines(section) == ["- user: name -> Kenny"]


# --------------------------------------------------------------------------------------
# The extractor: told to file what the person says about the clone under `self`.
# --------------------------------------------------------------------------------------


async def _never(_request: LLMRequest) -> ModelResponse:
    raise AssertionError("no model call is made to render the instructions")


def test_the_extractor_is_told_to_use_the_self_subject_and_sees_self_facts() -> None:
    """The prompt names `self` for what a person says about the clone, and lists what is held.

    Killed by: src/uclone_x/memory/extractor.py :: f'Use the subject "{SELF_SUBJECT}", with source "person", when a person line tells '
    Becomes: f'Use the subject "user", with source "person", when a person line tells '
    Killed by: src/uclone_x/memory/extractor.py :: if (subject := fold_name(fact.subject)) in (PERSON_SUBJECT, SELF_SUBJECT)
    Becomes: if (subject := fold_name(fact.subject)) in (PERSON_SUBJECT,)
    """
    memory = CrossSessionMemory()
    memory.record_fact("self", "hair", "red", _PROV, "s0")
    lesson = Lesson(
        clone_id="scout",
        clone_name="Scout",
        room_id="r1",
        session_id="sess",
        turn_id="t1",
        lines=(SpanLine(kind="person", speaker="Kenny", text="아니 파란 머리야"),),
        memory=memory,
        generate=_never,
    )
    render = extractor_module._instructions  # pyright: ignore[reportPrivateUsage]
    told = " ".join(render(lesson).split())

    assert 'Use the subject "self", with source "person", when a person line tells Scout' in told
    assert '"hair", "eyes", "age", "appearance", "personality" or "speech_style"' in told
    assert "Scout's own lines never define it." in told
    assert "the contents of a story" in told
    assert "self | hair | red" in told


class _Model(BaseLLMConnector):
    """Replies "Okay."; answers each extraction call with the next queued list of facts."""

    def __init__(self, extractions: list[list[dict[str, Any]]]) -> None:
        super().__init__()
        self.extractions = extractions

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        system = str(request.messages[0].content) if request.messages else ""
        if system.startswith(extractor_module.INSTRUCTIONS_OPENING):
            content = json.dumps(self.extractions.pop(0) if self.extractions else [])
        else:
            content = "Okay."
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


class _Seats:
    def __init__(self, participant: Participant, agent: BaseAgent) -> None:
        self.agents = {participant.id: agent}

    async def resolve(self, participant: Participant, *, one_seat: bool = False) -> BaseAgent:
        return self.agents[participant.id]


class _Always:
    def __init__(self, speaker: str) -> None:
        self._speaker = speaker
        self._spoke = False

    @property
    def name(self) -> str:
        return "always"

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        if self._spoke:
            self._spoke = False
            return SpeakerDecision(verdict=SelectionVerdict.SILENCE, selector=self.name)
        self._spoke = True
        return SpeakerDecision(
            verdict=SelectionVerdict.SPEAK,
            speaker_id=self._speaker,
            selector=self.name,
            confidence=1.0,
        )


def _told(subject: str, relation: str, value: str) -> dict[str, Any]:
    return {
        "subject": subject,
        "relation": relation,
        "value": value,
        "source": "person",
        "confidence": 0.95,
        "durable": True,
        "why": "said so",
    }


@pytest.mark.asyncio
async def test_a_person_defining_the_clone_in_a_room_leaves_one_active_hair_fact(
    tmp_path: Path,
) -> None:
    """Told "너는 빨간 머리야", then "아니 파란 머리야" under its alias: one `self` hair fact.

    The model writes "너" the first time and the clone's alias the second; both are the
    clone, so the second supersedes the first and the next turn recalls only blue.

    Killed by: src/uclone_x/room/orchestrator.py :: clone_names=_clone_names(state, speaker),
    Becomes: clone_names=(),
    Killed by: src/uclone_x/memory/extractor.py :: subject = fact_subject(candidate.subject, lesson.person_names, lesson.clone_names)
    Becomes: subject = fact_subject(candidate.subject, lesson.person_names, ())
    """
    person = _participant("alice", ParticipantKind.HUMAN, "Alice")
    clone = Participant(
        id="rin",
        kind=ParticipantKind.AGENT,
        display_name="Rin",
        aliases=("린",),
        session_id="sess_room__r1__rin",
    )
    memory = CrossSessionMemory(storage_path=tmp_path / "rin-memory.json")
    llm = _Model([[_told("너", "hair", "빨간 머리")], [_told("린", "hair", "파란 머리")]])
    agent = BaseAgent(
        config=AgentConfig(
            agent_id=clone.id, name="Rin", llm_config=AgentLLMConfig(model_name="scripted")
        ),
        llm=llm,
        tools=ToolRegistry(),
        context=AgentContext(session_id=clone.session_id, agent_id=clone.id),
        memory=memory,
        store=SessionStore(tmp_path / "sessions" / clone.id),
    )
    store = RoomStore(tmp_path / "rooms")
    store.save(RoomState(room_id="r1", participants=(person, clone), policy=RoomPolicy()))
    orch = RoomOrchestrator(
        store=store,
        selectors=[_Always(clone.id)],
        resolver=_Seats(clone, agent),
        extractor=KnowledgeExtractor(),
    )

    for said in ("너는 빨간 머리야", "아니 파란 머리야"):
        await orch.post("r1", person.id, said)
        await orch.wait_for_learning("r1")

    [hair] = memory.list_facts(subject="self", predicate="hair")
    assert hair.object_value == "파란 머리"
    assert hair.origin == "told"
    section = await recall_prompt_section(memory, "안녕")
    assert _fact_lines(section) == ["- self: hair -> 파란 머리"]
