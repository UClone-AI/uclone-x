"""Case-skill routing for the Artist (#1828): the case, the texts and where they go.

The routing functions are tested on their own; the agent-level tests drive a real
`BaseAgent` turn with a fake connector that answers the extraction call with given
facts and records every request, so what is asserted is the request the model saw.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from uclone_x.agent.artist_skill_router import (
    CASE_SKILL_HEADER,
    CASE_SKILL_NAMES,
    ROUTED_SKILL_TAG,
    RequestFacts,
    case_skill_section,
    case_skill_texts,
    grounded_facts,
    is_follow_up,
    latest_span_message,
    route_first_turn,
    route_follow_up,
)
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, PersonaDefinition
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.errors import LLMProviderError
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
from uclone_x.skills.auditor import SkillRegistry, load_runtime_skill_registry
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.skill_loader import LoadSkillParams, LoadSkillTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The shipped skill store, loaded the way a head loads it: audited, and each package
#: admitted only while its bytes match its pin in `SHIPPED_SKILL_PINS`.
_STORE: SkillRegistry = asyncio.run(load_runtime_skill_registry(REPO_ROOT / "ucx-agent-skills"))
CASE_SKILLS = case_skill_texts(_STORE)

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="fake", model="fake"),
    served_by=ServiceRef(provider="fake", model="fake"),
    attempts=(),
)
_USAGE = TokenUsage(provider="fake", model="fake", input_tokens=1, output_tokens=1)
_IMAGE_URL = "/api/artifacts/content?path=artifacts/images/img_a1.png"


def _reply(content: str | None, tool_calls: tuple[ToolCallRequest, ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP,
        usage=_USAGE,
        provenance=_PROV,
    )


def _tool_message(content: str, name: str = "generate_image") -> ChatMessage:
    return ChatMessage(role=MessageRole.TOOL, content=content, name=name, tool_call_id="c1")


# ------------------------------------------------------------------ the case


def test_a_drawn_image_in_history_makes_a_follow_up() -> None:
    """Read from state: a successful `generate_image` result, not the wording.

    Killed by: src/uclone_x/agent/artist_skill_router.py ::         and '"relative_url"' in (m.content or "")
    Becomes:         and True
    """
    drawn = json.dumps({"status": "success", "relative_url": _IMAGE_URL})
    failed = json.dumps({"status": "error", "error": "No image engine is available."})

    assert is_follow_up([_tool_message(drawn)])
    assert not is_follow_up([_tool_message(failed)])
    assert not is_follow_up([_tool_message(drawn, name="set_avatar")])
    assert not is_follow_up([ChatMessage(role=MessageRole.USER, content="밤으로 바꿔")])


def test_a_quote_the_message_does_not_contain_is_dropped() -> None:
    """Grounding: a small model invents media and details; code keeps only real quotes.

    Killed by: src/uclone_x/agent/artist_skill_router.py ::     return bool(text) and text not in ("null", "none") and text in message.lower()
    Becomes:     return bool(text) and text not in ("null", "none")
    """
    message = "Draw a dragon with Red Scales"
    facts = RequestFacts(
        medium_quote="lineart", minimal_quote="none", attribute_quotes=("red scales", "wings")
    )

    assert grounded_facts(facts, message) == RequestFacts(attribute_quotes=("red scales",))


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        (RequestFacts(), ("art-brief-expansion", "art-genre-vocab")),
        (
            RequestFacts(attribute_quotes=("a", "b", "c")),
            ("art-brief-expansion", "art-genre-vocab"),
        ),
        (RequestFacts(attribute_quotes=("a", "b", "c", "d")), ("art-literal-spec",)),
        (RequestFacts(medium_quote="연필 스케치"), ("art-medium-restraint",)),
        (RequestFacts(minimal_quote="배경 없이"), ("art-medium-restraint",)),
        (
            RequestFacts(minimal_quote="흰 배경", attribute_quotes=("a", "b", "c", "d")),
            ("art-medium-restraint", "art-literal-spec"),
        ),
    ],
)
def test_first_turn_route(facts: RequestFacts, expected: tuple[str, ...]) -> None:
    """Killed by: src/uclone_x/agent/artist_skill_router.py ::     detailed = len(facts.attribute_quotes) >= LITERAL_ATTRIBUTES
    Becomes:     detailed = len(facts.attribute_quotes) > LITERAL_ATTRIBUTES
    """
    assert route_first_turn(facts) == expected


def test_a_follow_up_asking_for_detail_also_gets_vocabulary() -> None:
    """Killed by: src/uclone_x/agent/artist_skill_router.py ::     if asks_for_more_detail(message):
    Becomes:     if False:
    """
    assert route_follow_up("밤으로 바꿔") == ("art-iterative-edit",)
    assert route_follow_up("좀 더 화려하게, 디테일 채워줘") == (
        "art-iterative-edit",
        "art-genre-vocab",
    )


# ------------------------------------------------------------------ the texts


def test_the_texts_have_nothing_the_model_could_paste_as_a_tag() -> None:
    """A prototype defect (#1828): `environment:` and `palette:` labels and a "no props"
    instruction were copied into prompts verbatim. The only `no ...` string allowed is
    the real tag. Quality tags are left to the call-time fill.
    """
    for name, text in CASE_SKILLS.items():
        assert not re.search(r"\b[a-z_]+:(?!\d)", text), name
        for match in re.finditer(r"\b(?:no|without)\s+(\w+)", text, re.IGNORECASE):
            assert match.group(0) == "no humans", (name, match.group(0))
        assert "masterpiece" not in text and "best quality" not in text, name


def test_each_culture_keeps_its_own_vocabulary_line() -> None:
    """A prototype defect (#1828): one "East Asian traditional" line had the model dress
    a kimono request in hanbok. Each line names one culture's clothing only.
    """
    lines = CASE_SKILLS["art-genre-vocab"].replace("\\\n", "").splitlines()
    japanese = next(line for line in lines if line.startswith("- Japanese"))
    korean = next(line for line in lines if line.startswith("- Korean"))
    chinese = next(line for line in lines if line.startswith("- Chinese"))

    assert "kimono" in japanese and "hanbok" not in japanese and "hanfu" not in japanese
    assert "hanbok" in korean and "kimono" not in korean and "hanfu" not in korean
    assert "hanfu" in chinese and "kimono" not in chinese and "hanbok" not in chinese


def test_every_case_skill_is_read_from_the_pinned_store() -> None:
    """The texts are store packages (#1865): all five load, pinned, and are tagged routed.

    Killed by: src/uclone_x/agent/artist_skill_router.py ::         if skill is not None and skill.manifest.status is SkillStatus.ACTIVE:
    Becomes:         if skill is not None and skill.manifest.status is not SkillStatus.ACTIVE:
    """
    assert tuple(CASE_SKILLS) == CASE_SKILL_NAMES
    for name in CASE_SKILL_NAMES:
        skill = _STORE.get(name)
        assert skill is not None
        assert skill.manifest.requires_tools == ("generate_image",)
        assert ROUTED_SKILL_TAG in skill.manifest.tags
        assert CASE_SKILLS[name] and not CASE_SKILLS[name].startswith("---")
    assert case_skill_texts(None) == {}


def test_the_literal_spec_leaves_the_negative_prompt_to_the_tool() -> None:
    """H5 regression (#1865): without this line, 4 of 10 qwen3:8b runs at 50ec13fa stopped
    at the 1500-token length cap with no tool call; with it, 10 of 10 called
    (`docs/eval/runs/2026-09-28-artist-case-routing-judged/h5_ab_*.jsonl`).
    """
    text = CASE_SKILLS["art-literal-spec"]
    assert (
        "Leave negative_prompt empty unless the person asked for something to be left out" in text
    )


@pytest.mark.parametrize("name", CASE_SKILL_NAMES)
def test_load_skill_refuses_a_case_routed_skill_in_plain_words(name: str) -> None:
    """A case skill is handed over by code; the model cannot pull one in by name (#1865).

    Killed by: src/uclone_x/tools/builtin/skill_loader.py ::         if ROUTED_SKILL_TAG in skill.manifest.tags:
    Becomes:         if False:
    """
    tool = LoadSkillTool(_STORE, tool_scope=lambda: ("generate_image",))
    context = ToolContext(agent_id="artist", session_id="s", workspace_root=Path("/tmp"))

    with pytest.raises(ValueError) as refused:
        asyncio.run(tool.run(LoadSkillParams(skill_name=name), context))
    message = str(refused.value)
    assert message == (
        f"Skill '{name}' is added automatically when a request needs it, "
        "so it cannot be loaded by name."
    )
    assert CASE_SKILLS[name] not in message
    # An unrouted skill in the same store still loads under the same scope.
    plain = _STORE.get("remote_gpu_recovery")
    assert plain is not None
    loaded = asyncio.run(tool.run(LoadSkillParams(skill_name="remote_gpu_recovery"), context))
    assert loaded == plain.instructions_markdown


def test_the_section_names_each_skill_and_says_the_person_wins() -> None:
    section = case_skill_section(("art-literal-spec",), CASE_SKILLS)

    assert section.startswith(CASE_SKILL_HEADER)
    assert "[art-literal-spec]" in section and CASE_SKILLS["art-literal-spec"] in section
    assert "always wins" in section
    assert case_skill_section((), CASE_SKILLS) == ""
    assert case_skill_section(("art-literal-spec",), {}) == ""


# ------------------------------------------------------------------ in a turn


class _ImageParams(BaseModel):
    prompt: str = ""


class _FakeGenerateImage(BaseTool[_ImageParams]):
    name = "generate_image"
    description = "generates images"

    def run(self, params: _ImageParams, context: ToolContext) -> dict[str, str]:
        return {"status": "success", "relative_url": _IMAGE_URL}


class _RoutingLLM(BaseLLMConnector):
    """Answers the extraction call with `facts` (or raises), and draws once per turn.

    A step's reply is keyed on whether the last message before the runtime's turn
    context is a tool result, not on a queue running out.
    """

    def __init__(self, facts: dict[str, object] | Exception) -> None:
        super().__init__()
        self._facts = facts
        self.structured: list[LLMRequest] = []
        self.turns: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        if request.response_schema is not None:
            self.structured.append(request)
            if isinstance(self._facts, Exception):
                raise self._facts
            return _reply(json.dumps(self._facts, ensure_ascii=False))
        self.turns.append(request)
        spoken = [m for m in request.messages if not (m.content or "").startswith("[Turn Context]")]
        if spoken[-1].role is MessageRole.TOOL:
            return _reply("done")
        call = ToolCallRequest(id="c1", name="generate_image", arguments={"prompt": "1girl"})
        return _reply(None, (call,))

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        resp = await self.generate(request)
        yield StreamChunk(
            delta_content=resp.content, finish_reason=resp.finish_reason, tool_calls=resp.tool_calls
        )


def _agent(
    llm: BaseLLMConnector,
    tmp_path: Path,
    persona: str = "artist",
    *,
    skills: SkillRegistry | None = _STORE,
) -> BaseAgent:
    registry = ToolRegistry()
    registry.register(_FakeGenerateImage())
    return BaseAgent(
        config=AgentConfig(
            agent_id="artist",
            name="Artist",
            system_prompt="You are Artist.",
            persona=persona,
            llm_config=AgentLLMConfig(model_name="qwen3:8b"),
        ),
        llm=llm,
        tools=registry,
        skills=skills,
        context=AgentContext(agent_id="artist", session_id="s1", workspace_root=tmp_path),
        personas=(
            *(
                PersonaDefinition(
                    name=name,
                    role=name,
                    system_prompt=name,
                    allowed_tools=("generate_image",),
                    enable_write_tools=True,
                )
                for name in ("artist", "writer")
            ),
            # A clone whose range does not reach `generate_image`: never offered it.
            PersonaDefinition(
                name="critic", role="critic", system_prompt="critic", allowed_tools=("file_read",)
            ),
        ),
    )


_DETAILED = "금발 트윈테일에 파란 눈, 메이드복을 입은 소녀가 은색 쟁반을 들고 서 있다"
_FACTS: dict[str, object] = {
    "medium_quote": "",
    "minimal_quote": "",
    "attribute_quotes": ["금발", "트윈테일", "파란 눈", "메이드복", "은색 쟁반", "watercolor"],
}


def _prefix(request: LLMRequest) -> tuple[list[str], list[tuple[str, str]]]:
    system = [m.content or "" for m in request.messages if m.role is MessageRole.SYSTEM]
    tools = [(t.name, t.description) for t in request.tools]
    return system, tools


@pytest.mark.asyncio
async def test_the_skill_follows_the_latest_user_message_and_the_prefix_is_unchanged(
    tmp_path: Path,
) -> None:
    """The routed skill rides in the turn context; the cached prefix is byte-identical.

    The same agent setup is run twice: routed (the extraction answers) and unrouted (the
    extraction fails). System prompt and tool descriptions must not differ by a byte.

    Killed by: src/uclone_x/agent/turn_executor.py ::                     present_sections(undone_section, image_set_section, case_section)
    Becomes:                     present_sections(undone_section, image_set_section)
    Killed by: src/uclone_x/agent/prompt_assembler.py ::             and ROUTED_SKILL_TAG not in s.manifest.tags
    Becomes:             and True
    """
    routed = _RoutingLLM(_FACTS)
    unrouted = _RoutingLLM(LLMProviderError("connection refused"))
    await _agent(routed, tmp_path / "a").execute_turn(_DETAILED)
    await _agent(unrouted, tmp_path / "b").execute_turn(_DETAILED)

    first = routed.turns[0]
    assert _prefix(first) == _prefix(unrouted.turns[0])
    system, tools = _prefix(first)
    assert all(CASE_SKILL_HEADER not in s for s in system)
    assert all("art-literal-spec" not in d for _, d in tools)
    # The store's other skills are listed; the case-routed ones are not (#1865).
    assert any("media-character" in s for s in system)
    assert all(name not in s for name in CASE_SKILL_NAMES for s in system)

    last = first.messages[-1]
    assert last.role is MessageRole.USER
    content = last.content or ""
    assert content.index(_DETAILED) < content.index("[art-literal-spec]")
    assert "[art-brief-expansion]" not in content
    assert all(CASE_SKILL_HEADER not in (m.content or "") for m in unrouted.turns[0].messages)


@pytest.mark.asyncio
async def test_the_skill_stays_for_every_step_and_never_enters_history(tmp_path: Path) -> None:
    """Same position on every step of the turn (layering §5.5); built per request."""
    llm = _RoutingLLM(_FACTS)
    agent = _agent(llm, tmp_path)
    await agent.execute_turn(_DETAILED)

    assert len(llm.turns) == 2
    assert "[art-literal-spec]" in (llm.turns[1].messages[-1].content or "")
    history = agent._history  # pyright: ignore[reportPrivateUsage]
    assert all(CASE_SKILL_HEADER not in (m.content or "") for m in history)


@pytest.mark.asyncio
async def test_a_second_turn_after_a_drawn_image_is_an_edit_without_extraction(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/base.py ::         if is_follow_up(self._history):
    Becomes:         if False:
    """
    llm = _RoutingLLM(_FACTS)
    agent = _agent(llm, tmp_path)
    await agent.execute_turn(_DETAILED)
    assert len(llm.structured) == 1

    await agent.execute_turn("밤으로 바꿔")

    assert len(llm.structured) == 1
    content = llm.turns[2].messages[-1].content or ""
    assert content.index("밤으로 바꿔") < content.index("[art-iterative-edit]")


@pytest.mark.asyncio
async def test_a_failed_extraction_leaves_the_turn_as_it_was(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/base.py ::                 return None  # an unread request is drawn unrouted
    Becomes:                 return case_skill_section(("art-brief-expansion",), texts)
    """
    llm = _RoutingLLM(LLMProviderError("connection refused at 127.0.0.1:11434"))
    result = await _agent(llm, tmp_path).execute_turn("고양이 그려줘")

    assert result.is_completed and result.error is None
    assert len(llm.structured) == 1
    assert "127.0.0.1" not in result.content
    assert all(CASE_SKILL_HEADER not in (m.content or "") for m in llm.turns[0].messages)


@pytest.mark.asyncio
async def test_another_clone_offered_generate_image_is_routed_and_not_told_it_is_artist(
    tmp_path: Path,
) -> None:
    """Any clone offered `generate_image` gets the case skills, under a neutral header (#2091).

    Killed by: src/uclone_x/agent/base.py ::         if llm is None:  # any clone offered generate_image is routed, not only the Artist
    Becomes:         if llm is None or self._persona != "artist":
    Killed by: src/uclone_x/agent/artist_skill_router.py :: CASE_SKILL_HEADER = "[Drawing Case Skills]"
    Becomes: CASE_SKILL_HEADER = "[Artist Case Skills]"
    """
    llm = _RoutingLLM(_FACTS)
    await _agent(llm, tmp_path, persona="writer").execute_turn(_DETAILED)

    assert len(llm.structured) == 1
    content = llm.turns[0].messages[-1].content or ""
    section = content[content.index(CASE_SKILL_HEADER) :]
    assert "[art-literal-spec]" in section
    assert "artist" not in section.lower()


@pytest.mark.asyncio
async def test_a_clone_not_offered_generate_image_is_not_routed(tmp_path: Path) -> None:
    """With the persona gate gone, the offered tool is what decides: no extraction call.

    Killed by: src/uclone_x/agent/base.py ::         if not any(tool.name == "generate_image" for tool in tool_defs):  # nothing to route for
    Becomes:         if False:
    """
    llm = _RoutingLLM(_FACTS)
    await _agent(llm, tmp_path, persona="critic").execute_turn(_DETAILED)

    assert "generate_image" not in [d.name for d in llm.turns[0].tools]
    assert llm.structured == []
    assert all(CASE_SKILL_HEADER not in (m.content or "") for m in llm.turns[0].messages)


@pytest.mark.asyncio
async def test_an_agent_without_the_skill_store_is_not_routed(tmp_path: Path) -> None:
    """No store, no case texts: no extraction call is spent on a section that cannot exist.

    Killed by: src/uclone_x/agent/base.py ::         if not texts:
    Becomes:         if False:
    """
    llm = _RoutingLLM(_FACTS)
    await _agent(llm, tmp_path, skills=None).execute_turn(_DETAILED)

    assert llm.structured == []
    assert all(CASE_SKILL_HEADER not in (m.content or "") for m in llm.turns[0].messages)


# ------------------------------------------------------------------ in a room


def test_the_latest_message_of_a_room_span_is_read_without_its_sender() -> None:
    span = (
        "[...2 earlier messages in this room are not shown]\n"
        "[critic]: 금발에 파란 눈이면 좋겠어요\n"
        "[user]: 고양이 그려줘\n배경은 하얗게"
    )

    assert latest_span_message(span) == "고양이 그려줘\n배경은 하얗게"
    assert latest_span_message("고양이 그려줘") == "고양이 그려줘"


def test_the_latest_message_of_a_room_span_reads_the_newest_user_message_even_when_another_clone_spoke_after() -> (
    None
):
    """In a room, another clone's message does not steer routing (#1865).

    Killed by: src/uclone_x/agent/artist_skill_router.py :: if _is_person(sender):
    Becomes: if False:
    """
    span = (
        "[...2 earlier messages in this room are not shown]\n"
        "[user]: 고양이 그려줘\n배경은 하얗게\n"
        "[critic]: 금발에 파란 눈이면 좋겠어요"
    )
    assert latest_span_message(span) == "고양이 그려줘\n배경은 하얗게"
    assert latest_span_message(span, person_names=("user",)) == "고양이 그려줘\n배경은 하얗게"
    assert latest_span_message(span, person_names=("kenny",)) == "금발에 파란 눈이면 좋겠어요"

    span_with_custom_name = (
        "[...2 earlier messages in this room are not shown]\n"
        "[kenny]: 강아지 그려줘\n"
        "[critic]: 고양이 그려줘"
    )
    assert latest_span_message(span_with_custom_name, person_names=("kenny",)) == "강아지 그려줘"


@pytest.mark.asyncio
async def test_in_a_room_only_the_latest_message_grounds_the_route(tmp_path: Path) -> None:
    """An earlier speaker's details are not this request's (#1865).

    The span holds a critic's line with every quoted detail; the latest message is a
    two-word brief. Grounded against the whole span, five details survive and the turn
    is routed as a detailed specification; against the latest message, none do.

    Killed by: src/uclone_x/agent/base.py ::             message = latest_span_message(message, self._turn_person_names)
    Becomes:             message = message
    """
    span = f"[critic]: {_DETAILED}\n[user]: 고양이 그려줘"
    llm = _RoutingLLM(_FACTS)
    await _agent(llm, tmp_path).execute_turn(span, room_id="room-1")

    assert [m.content for m in llm.structured[0].messages][-1] == "고양이 그려줘"
    content = llm.turns[0].messages[-1].content or ""
    assert "[art-brief-expansion]" in content
    assert "[art-literal-spec]" not in content


@pytest.mark.asyncio
async def test_in_a_room_another_clone_speaking_after_user_does_not_steer_routing(
    tmp_path: Path,
) -> None:
    """An intervening clone's words do not decide which case skill gets routed (#1865)."""
    span = f"[user]: 고양이 그려줘\n[critic]: {_DETAILED}"
    llm = _RoutingLLM(_FACTS)
    await _agent(llm, tmp_path).execute_turn(span, room_id="room-1")
    assert [m.content for m in llm.structured[0].messages][-1] == "고양이 그려줘"


# ------------------------------------------------------------------ beside an image set


class _SetLLM(_RoutingLLM):
    """`_RoutingLLM`, with the image-set planning call answered by `plan` (or raising)."""

    def __init__(self, plan: str | Exception) -> None:
        super().__init__(_FACTS)
        self._plan = plan
        self.plans = 0

    async def generate(self, request: LLMRequest) -> ModelResponse:
        schema = request.response_schema
        if schema is not None and "medium_quote" not in str(schema.get("properties")):
            self.plans += 1
            if isinstance(self._plan, Exception):
                raise self._plan
            return _reply(self._plan)
        return await super().generate(request)


_SET_ASK = "고양이로 다양한 포즈 3장 그려줘"
_SET_PLAN = json.dumps(
    {
        "shared": ["cat"],
        "style": ["anime coloring"],
        "variants": [
            {
                "subject": [],
                "action": [f"pose {i}"],
                "expression": ["calm"],
                "location": ["garden"],
                "camera": ["full body"],
            }
            for i in range(3)
        ],
    }
)


@pytest.mark.asyncio
async def test_a_failed_plan_still_routes_the_case_skills(tmp_path: Path) -> None:
    """Review of #1966: a failed plan's note names no prompts, so it must not cost the routing.

    Killed by: src/uclone_x/agent/turn_executor.py :: if image_set_section is not None and is_planned_set(image_set_section):
    Becomes: if image_set_section:
    """
    llm = _SetLLM(LLMProviderError("connection refused at 127.0.0.1:11434"))
    result = await _agent(llm, tmp_path).execute_turn(_SET_ASK)

    assert result.is_completed and result.error is None
    assert llm.plans == 1
    assert len(llm.structured) == 1  # the extraction ran as well
    content = llm.turns[0].messages[-1].content or ""
    assert "[Image Set Request (3 images)]" in content
    assert "[art-brief-expansion]" in content


@pytest.mark.asyncio
async def test_a_planned_set_is_not_routed_as_well(tmp_path: Path) -> None:
    """A plan carries its own prompts; routing it too would spend an extraction call on it.

    Killed by: src/uclone_x/agent/image_set_planner.py ::     return section.startswith(_PLAN_HEADING)
    Becomes:     return False
    """
    llm = _SetLLM(_SET_PLAN)
    result = await _agent(llm, tmp_path).execute_turn(_SET_ASK)

    assert result.is_completed and result.error is None
    assert llm.plans == 1
    assert llm.structured == []
    content = llm.turns[0].messages[-1].content or ""
    assert "[Image Set Plan]" in content
    assert CASE_SKILL_HEADER not in content
