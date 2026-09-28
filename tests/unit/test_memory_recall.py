"""Relevance-ranked recall (clone-knowledge-graph §3.5, step 5).

Deterministic and offline: the embedders here are in-process fakes, and the agent tests use
the mock connector.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.core.provenance import Provenance
from uclone_x.errors import EmbeddingError
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.memory import extractor as extractor_module
from uclone_x.memory.models import MemoryFact
from uclone_x.memory.recall import EMBEDDER_FAILED_METHOD, recall_prompt_section
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.memory.tools import RecordMemoryFactParams, RecordMemoryFactTool
from uclone_x.tools.models import ToolContext


class _TopicEmbedder:
    """One axis per topic word, so a paraphrase with no shared word can still match."""

    TOPICS = (("database", "postgres", "datastore"), ("deploy", "release", "ship"))

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "topic-axes"

    @property
    def dimensions(self) -> int:
        return len(self.TOPICS) + 1

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            lowered = text.lower()
            axes = [1.0 if any(word in lowered for word in topic) else 0.0 for topic in self.TOPICS]
            vectors.append((*axes, 0.01))
        return tuple(vectors)


class _BrokenEmbedder(_TopicEmbedder):
    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        raise EmbeddingError("embedding endpoint unreachable")


def _prov() -> Provenance:
    return Provenance.primary(provider="agent.test", model="memory")


def _put(
    memory: CrossSessionMemory,
    fact_id: str,
    subject: str,
    predicate: str,
    value: str,
    *,
    confidence: float = 0.8,
    created_at: str = "2026-09-20T00:00:00+00:00",
) -> None:
    memory._facts[fact_id] = MemoryFact(  # pyright: ignore[reportPrivateUsage]
        fact_id=fact_id,
        subject=subject,
        predicate=predicate,
        object_value=value,
        provenance=_prov(),
        source_session_id="sess_a",
        confidence=confidence,
        created_at=created_at,
    )


def _fact_lines(section: str) -> list[str]:
    return [line for line in section.splitlines() if line.startswith("- ")]


@pytest.mark.asyncio
async def test_user_facts_lead_newest_first_and_are_capped_at_five() -> None:
    """Facts about the person come first, newest first, at most five (§3.5 item 1).

    Killed by: src/uclone_x/memory/recall.py :: USER_FACTS_LIMIT = 5
    Becomes: USER_FACTS_LIMIT = 9
    Killed by: src/uclone_x/memory/recall.py :: key=lambda fact: (fact.created_at, fact.fact_id),
    Becomes: key=lambda fact: (fact.fact_id, fact.created_at),
    """
    memory = CrossSessionMemory(max_facts_in_prompt=10)
    for day in range(1, 8):
        _put(
            memory,
            f"mem_z{8 - day}",
            "User",
            f"likes_{day}",
            f"thing {day}",
            created_at=f"2026-09-0{day}T00:00:00+00:00",
        )
    _put(memory, "mem_db", "database", "engine", "postgres")

    section = await recall_prompt_section(memory, "which database engine")

    assert _fact_lines(section) == [
        "- User: likes_7 -> thing 7",
        "- User: likes_6 -> thing 6",
        "- User: likes_5 -> thing 5",
        "- User: likes_4 -> thing 4",
        "- User: likes_3 -> thing 3",
        "- database: engine -> postgres",
    ]


@pytest.mark.asyncio
async def test_relevant_fact_outranks_higher_confidence_ones_and_the_rest_are_counted() -> None:
    """The message, not confidence, picks the facts; what is left out is counted (C2).

    Killed by: src/uclone_x/agent/prompt_assembler.py :: return None if memory is None else await recall_prompt_section(memory, message)
    Becomes: return None if memory is None else memory.format_prompt_section()
    """
    memory = CrossSessionMemory(max_facts_in_prompt=2)
    _put(memory, "mem_a", "theme", "colour", "dark", confidence=0.95)
    _put(memory, "mem_b", "editor", "font", "mono", confidence=0.9)
    _put(memory, "mem_c", "project_x", "database", "postgres 16", confidence=0.6)

    agent = BaseAgent(config=AgentConfig(agent_id="recall_a", name="A"), memory=memory)
    section = await agent._prompt_assembler.recall_memory(  # pyright: ignore[reportPrivateUsage]
        "which database does project x use?"
    )

    assert section is not None
    assert _fact_lines(section) == ["- project_x: database -> postgres 16"]
    assert "2 more facts not selected for this message" in section
    assert "Ranked by lexical overlap (no embedder configured" in section


@pytest.mark.asyncio
async def test_snake_case_keys_match_spaced_words() -> None:
    """A `gate_command` key is found by "gate command": underscores split lexical tokens.

    Killed by: src/uclone_x/memory/retrieval.py :: _]+", re.UNICODE)
    Becomes: ]+", re.UNICODE)
    """
    memory = CrossSessionMemory(max_facts_in_prompt=1)
    _put(memory, "mem_a", "theme", "colour", "dark", confidence=0.95)
    _put(memory, "mem_b", "ci_pipeline", "gate_command", "ucx test check", confidence=0.6)

    section = await recall_prompt_section(memory, "What is the gate command?")

    assert _fact_lines(section) == ["- ci_pipeline: gate_command -> ucx test check"]


@pytest.mark.asyncio
async def test_an_embedder_ranks_by_meaning_and_says_so() -> None:
    """With an embedder, a paraphrase sharing no word still recalls the fact."""
    embedder = _TopicEmbedder()
    memory = CrossSessionMemory(max_facts_in_prompt=1, embedder=embedder)
    _put(memory, "mem_a", "release", "window", "friday", confidence=0.95)
    _put(memory, "mem_b", "service", "backend", "postgres 16", confidence=0.6)

    section = await recall_prompt_section(memory, "what datastore do we run?")

    assert _fact_lines(section) == ["- service: backend -> postgres 16"]
    assert "Ranked by embedding similarity (topic-axes)" in section
    assert embedder.calls > 0


@pytest.mark.asyncio
async def test_a_failing_embedder_falls_back_to_lexical_and_says_so() -> None:
    """An embedder error never fails the turn: lexical ranking, named as a fallback (P6).

    Killed by: src/uclone_x/memory/recall.py :: ranked=ranking.ranked, method=EMBEDDER_FAILED_METHOD, considered=ranking.considered
    Becomes: ranked=ranking.ranked, method=ranking.method, considered=ranking.considered
    """
    memory = CrossSessionMemory(max_facts_in_prompt=1, embedder=_BrokenEmbedder())
    _put(memory, "mem_a", "theme", "colour", "dark", confidence=0.95)
    _put(memory, "mem_b", "service", "database", "postgres 16", confidence=0.6)

    section = await recall_prompt_section(memory, "which database?")

    assert _fact_lines(section) == ["- service: database -> postgres 16"]
    assert EMBEDDER_FAILED_METHOD in section


@pytest.mark.asyncio
async def test_empty_and_unreadable_stores_keep_their_answers(tmp_path: object) -> None:
    """No active fact: the same answer `format_prompt_section` gives, empty or UNAVAILABLE."""
    assert await recall_prompt_section(CrossSessionMemory(), "anything") == ""

    broken = CrossSessionMemory()
    broken._load_failure = "memory.json is not valid JSON"  # pyright: ignore[reportPrivateUsage]
    section = await recall_prompt_section(broken, "anything")
    assert section.startswith("[Cross-Session Memory Facts]\nUNAVAILABLE")


class _ScriptedLLM(MockLLMConnector):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self._queue = list(responses)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
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


def _memory_part(request: LLMRequest) -> str:
    tail = request.messages[-1]
    assert tail.role is MessageRole.USER
    content = tail.content or ""
    return content[content.index("[Cross-Session Memory Facts]") :]


@pytest.mark.asyncio
async def test_recall_is_in_the_tail_follows_the_message_and_holds_within_a_turn() -> None:
    """Per turn the tail recalls for that message; the system turn and in-turn tails hold still.

    Killed by: src/uclone_x/agent/turn_executor.py :: turn_live.recalled_memory = await self._prompt_assembler.recall_memory(
    Becomes: turn_live.recalled_memory = None and await self._prompt_assembler.recall_memory(
    """
    memory = CrossSessionMemory(max_facts_in_prompt=1)
    _put(memory, "mem_a", "theme", "colour", "dark", confidence=0.95)
    _put(memory, "mem_b", "service", "database", "postgres 16", confidence=0.6)
    lookup = ToolCallRequest(id="call_1", name="query_memory_facts", arguments={"subject": "x"})
    llm = _ScriptedLLM([_reply("", (lookup,)), _reply("Postgres."), _reply("Dark.")])
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="recall_turns",
            name="R",
            llm_config=AgentLLMConfig(model_name="mock", auto_compact=False),
        ),
        llm=llm,
        memory=memory,
    )

    await agent.execute_turn("which database does the project use?")
    await agent.execute_turn("which colour theme?")

    first_step, second_step, next_turn = llm.requests
    assert "[Cross-Session Memory Facts]" not in (first_step.messages[0].content or "")
    assert first_step.messages[0] == next_turn.messages[0]
    assert "- service: database -> postgres 16" in _memory_part(first_step)
    assert _memory_part(second_step) == _memory_part(first_step)
    assert "- theme: colour -> dark" in _memory_part(next_turn)
    assert "postgres 16" not in _memory_part(next_turn)


@pytest.mark.asyncio
async def test_a_fact_saved_mid_turn_leaves_the_next_steps_tail_unchanged() -> None:
    """A save between steps reaches the tail at the next turn, never at the next step (§5.5).

    Re-rendering the memory section on every step would put the saved fact into the second
    step's tail and move the tail inside the turn.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: memory_section = self._active_session.recalled_memory
    Becomes: memory_section = None
    """
    memory = CrossSessionMemory(max_facts_in_prompt=5)
    _put(memory, "mem_a", "user", "name", "Kenny")
    save = ToolCallRequest(
        id="call_1",
        name="record_memory_fact",
        arguments={"subject": "user", "predicate": "editor", "object_value": "helix"},
    )
    llm = _ScriptedLLM([_reply("", (save,)), _reply("Saved."), _reply("Hello.")])
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="recall_save",
            name="S",
            llm_config=AgentLLMConfig(model_name="mock", auto_compact=False),
        ),
        llm=llm,
        memory=memory,
    )

    await agent.execute_turn("remember that I use helix")
    await agent.execute_turn("hi")

    first_step, second_step, next_turn = llm.requests
    assert [fact.object_value for fact in memory.list_facts(subject="user")] == [
        "Kenny",
        "helix",
    ], "the save did not land, so this test would compare nothing"
    assert _memory_part(second_step) == _memory_part(first_step)
    assert "helix" not in _memory_part(second_step)
    assert "- user: editor -> helix" in _memory_part(next_turn)


@pytest.mark.asyncio
async def test_a_fact_below_half_confidence_is_not_recalled() -> None:
    """The 0.5 floor holds for the person's facts and for ranked ones alike.

    Killed by: src/uclone_x/memory/recall.py :: RECALL_MIN_CONFIDENCE = 0.5
    Becomes: RECALL_MIN_CONFIDENCE = 0.0
    """
    memory = CrossSessionMemory(max_facts_in_prompt=10)
    _put(memory, "mem_a", "user", "name", "Kenny", confidence=0.9)
    _put(memory, "mem_b", "user", "nickname", "K", confidence=0.4)
    _put(memory, "mem_c", "database", "engine", "postgres", confidence=0.4)
    _put(memory, "mem_d", "database", "version", "16", confidence=0.6)

    section = await recall_prompt_section(memory, "which database engine version")

    assert _fact_lines(section) == ["- user: name -> Kenny", "- database: version -> 16"]


@pytest.mark.asyncio
async def test_a_short_message_still_gets_the_projects_standing_preferences() -> None:
    """ "continue" shares no word with "reply in Korean"; the project's facts are included anyway.

    Newest first and capped, after the person's facts; a fact on another subject that the
    message does not match is still left out and counted (#1857).

    Killed by: src/uclone_x/memory/recall.py :: PROJECT_FACTS_LIMIT = 3
    Becomes: PROJECT_FACTS_LIMIT = 0
    Killed by: src/uclone_x/memory/models.py :: PROJECT_SUBJECT = "project"
    Becomes: PROJECT_SUBJECT = "projekt"
    """
    memory = CrossSessionMemory(max_facts_in_prompt=10)
    _put(memory, "mem_u", "user", "name", "Kenny")
    for day in range(1, 5):
        _put(
            memory,
            f"mem_p{day}",
            "Project",
            f"rule_{day}",
            f"rule {day}",
            created_at=f"2026-09-0{day}T00:00:00+00:00",
        )
    _put(memory, "mem_lang", "project", "language", "reply in Korean")
    _put(memory, "mem_db", "database", "engine", "postgres")

    for message in ("continue", "계속"):
        section = await recall_prompt_section(memory, message)
        assert _fact_lines(section) == [
            "- user: name -> Kenny",
            "- project: language -> reply in Korean",
            "- Project: rule_4 -> rule 4",
            "- Project: rule_3 -> rule 3",
        ], message
        assert "(3 more facts not selected for this message" in section


@pytest.mark.asyncio
async def test_a_key_of_one_letter_parts_still_matches() -> None:
    """`a_b` splits into parts too short to be tokens; the whole key is kept as one.

    Killed by: src/uclone_x/memory/retrieval.py :: tokens.update(key for key in _KEY_RE.findall(lowered) if key.strip("_"))
    Becomes: tokens.update(key for key in _KEY_RE.findall(lowered) if not key.strip("_"))
    """
    memory = CrossSessionMemory(max_facts_in_prompt=1)
    _put(memory, "mem_a", "theme", "colour", "dark", confidence=0.95)
    _put(memory, "mem_b", "config", "x_y", "on", confidence=0.6)

    section = await recall_prompt_section(memory, "is x_y set?")

    assert _fact_lines(section) == ["- config: x_y -> on"]


@pytest.mark.asyncio
async def test_a_fact_saved_under_the_persons_name_is_a_user_fact() -> None:
    """The model's `record_memory_fact` under the person's name lands in the user slots.

    Same rule as the extractor: the person's id, display name and aliases, and the words
    `PERSON_WORDS` lists, all file under `user` (#1857).

    Killed by: src/uclone_x/memory/tools.py :: subject=person_subject(params.subject, context.person_names),
    Becomes: subject=person_subject(params.subject, ()),
    """
    memory = CrossSessionMemory(max_facts_in_prompt=10)
    tool = RecordMemoryFactTool(memory)
    context = ToolContext(
        agent_id="clone", session_id="sess", person_names=("u_kenny", "Kenny Lim")
    )
    for subject, predicate in (("Kenny  Lim", "editor"), ("the user", "shell"), ("Seoul", "is")):
        await tool.run(
            RecordMemoryFactParams(subject=subject, predicate=predicate, object_value="x"),
            context,
        )

    assert sorted((f.subject, f.predicate) for f in memory.list_facts()) == [
        ("Seoul", "is"),
        ("user", "editor"),
        ("user", "shell"),
    ]
    section = await recall_prompt_section(memory, "hi")
    # Saved in the same second, so their order is the fact ids'; which two, not the order.
    assert sorted(_fact_lines(section)) == ["- user: editor -> x", "- user: shell -> x"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored", "written"),
    [("Jose\u0301", "Jos\u00e9"), ("Jos\u00e9", "Jose\u0301")],
    ids=["stored-decomposed", "written-decomposed"],
)
async def test_the_persons_name_matches_in_either_unicode_form(stored: str, written: str) -> None:
    """ "José" with a combining accent and "José" composed are one name (#1893).

    The person's name as stored and the subject the model writes are compared in NFC, and a
    subject that is not the person is kept in NFC, so the two forms of a third party's name
    are one subject.

    Killed by: src/uclone_x/memory/models.py :: return " ".join(unicodedata.normalize("NFC", name).split()).casefold()
    Becomes: return " ".join(name.split()).casefold()
    Killed by: src/uclone_x/memory/models.py :: collapsed = " ".join(unicodedata.normalize("NFC", subject).split())
    Becomes: collapsed = " ".join(subject.split())
    """
    memory = CrossSessionMemory(max_facts_in_prompt=10)
    tool = RecordMemoryFactTool(memory)
    context = ToolContext(agent_id="clone", session_id="sess", person_names=("u_1", stored))
    for subject, predicate in ((written, "lives_in"), ("Zoe\u0308", "role"), ("Zo\u00eb", "team")):
        await tool.run(
            RecordMemoryFactParams(subject=subject, predicate=predicate, object_value="x"),
            context,
        )

    assert sorted((f.subject, f.predicate) for f in memory.list_facts()) == [
        ("Zo\u00eb", "role"),
        ("Zo\u00eb", "team"),
        ("user", "lives_in"),
    ]


@pytest.mark.asyncio
async def test_standing_facts_never_overrun_the_limit_nor_repeat() -> None:
    """Five person facts and three project facts under a limit of six: six lines, each once.

    The project slots take only what the person's facts left, and a project fact the
    message also matches is not ranked in a second time.

    Killed by: src/uclone_x/memory/recall.py :: active, PROJECT_SUBJECT, min(PROJECT_FACTS_LIMIT, limit - len(about_user))
    Becomes: active, PROJECT_SUBJECT, min(PROJECT_FACTS_LIMIT, limit)
    Killed by: src/uclone_x/memory/recall.py :: chosen = {fact.fact_id for fact in standing}
    Becomes: chosen = {fact.fact_id for fact in about_user}
    """
    memory = CrossSessionMemory(max_facts_in_prompt=6)
    for day in range(1, 6):
        _put(
            memory,
            f"mem_u{day}",
            "user",
            f"likes_{day}",
            f"thing {day}",
            created_at=f"2026-09-0{day}T00:00:00+00:00",
        )
    for day in range(1, 4):
        _put(
            memory,
            f"mem_p{day}",
            "project",
            f"style_{day}",
            f"style {day}",
            created_at=f"2026-09-0{day}T00:00:00+00:00",
        )

    section = await recall_prompt_section(memory, "style 3")

    lines = _fact_lines(section)
    assert len(lines) == 6
    assert lines[-1] == "- project: style_3 -> style 3"
    assert len(set(lines)) == len(lines)

    roomy = CrossSessionMemory(max_facts_in_prompt=10)
    _put(roomy, "mem_p", "project", "language", "reply in Korean")
    _put(roomy, "mem_x", "theme", "colour", "dark")
    section = await recall_prompt_section(roomy, "reply language")
    assert _fact_lines(section) == ["- project: language -> reply in Korean"]


def test_the_model_is_told_to_use_the_project_subject() -> None:
    """Nothing files a fact under `project` unless the model is told to, on both write paths.

    Killed by: src/uclone_x/memory/tools.py :: f'"{PROJECT_SUBJECT}" for durable working preferences of this workspace, such as the '
    Becomes: f'"user" for durable working preferences of this workspace, such as the '
    Killed by: src/uclone_x/memory/extractor.py :: f'person. Use the subject "{PROJECT_SUBJECT}" for durable working preferences of '
    Becomes: f'person. Use the subject "user" for durable working preferences of '
    """
    from uclone_x.memory.extractor import Lesson, SpanLine

    async def _never(_request: LLMRequest) -> ModelResponse:
        raise AssertionError("no model call is made to render the instructions")

    lesson = Lesson(
        clone_id="scout",
        clone_name="Scout",
        room_id="r1",
        session_id="sess",
        turn_id="t1",
        lines=(SpanLine(kind="person", speaker="Kenny", text="reply in Korean"),),
        memory=CrossSessionMemory(),
        generate=_never,
    )
    render = extractor_module._instructions  # pyright: ignore[reportPrivateUsage]
    told = " ".join(render(lesson).split())
    described = " ".join(RecordMemoryFactTool(CrossSessionMemory()).description.split())

    line = 'the subject "project" for durable working preferences of this workspace'
    assert line in told
    assert line in described


@pytest.mark.parametrize(
    ("stored", "wanted"),
    [("José", "JOSÉ"), ("José", "josé"), ("Kenny  Lim", "kenny lim")],
    ids=["stored-decomposed", "wanted-decomposed", "inner-whitespace"],
)
def test_a_standing_subject_is_matched_folded(stored: str, wanted: str) -> None:
    """`_newest_about` compares subjects as every other subject lookup does (#1904).

    Today it is asked only for the ASCII constants `user` and `project`, and through
    `recall_prompt_section` a revert to `strip().casefold()` is not seen: neither constant
    has inner whitespace, and a scan of every code point found none whose NFC form
    casefolds into their letters when it did not already. So the fold is pinned here, at
    the function, with a subject the constants cannot exercise: one name in the other
    Unicode form, case or spacing.

    Killed by: src/uclone_x/memory/recall.py :: (fact for fact in active if fold_name(fact.subject) == wanted),
    Becomes: (fact for fact in active if fact.subject.strip().casefold() == wanted),
    """
    from uclone_x.memory.recall import (
        _newest_about,  # pyright: ignore[reportPrivateUsage]
    )

    memory = CrossSessionMemory()
    _put(memory, "mem_a", stored, "lives_in", "Seoul")
    _put(memory, "mem_b", "someone else", "lives_in", "Busan")

    found = _newest_about(memory.list_facts(), wanted, 5)

    assert [fact.fact_id for fact in found] == ["mem_a"]
