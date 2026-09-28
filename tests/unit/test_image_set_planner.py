"""Image-set planning: detection, prompt assembly, the repair turn, and the agent hand-off."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.image_set_planner import (
    QUALITY_TAGS,
    ImageSetPlan,
    ImageSetPlanError,
    ImageVariant,
    assemble_prompts,
    bad_entries,
    detect_image_set,
    plan_image_set,
    plan_schema,
    repeated_locations,
)
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, PersonaDefinition
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.errors import LLMProviderError, StructuredOutputUnsupportedError
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.connectors.openai import OpenAIConnector
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
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.image_set_intent import wants_varied_locations
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import ToolRegistry

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="fake", model="fake"),
    served_by=ServiceRef(provider="fake", model="fake"),
    attempts=(),
)
_USAGE = TokenUsage(provider="fake", model="fake", input_tokens=1, output_tokens=1)


def _reply(content: str | None, tool_calls: tuple[ToolCallRequest, ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP,
        usage=_USAGE,
        provenance=_PROV,
    )


def _plan_json(count: int, *, korean: bool = False) -> str:
    return json.dumps(
        {
            "shared": ["1girl", "solo", "silver hair", "elf"],
            "style": ["anime coloring"],
            "variants": [
                {
                    "subject": [],
                    "action": ["은발" if korean and i == 0 else f"pose {i}"],
                    "expression": ["smile"],
                    "location": ["forest"],
                    "camera": ["full body"],
                }
                for i in range(count)
            ],
        },
        ensure_ascii=False,
    )


# ----------------------------------------------------------------------------- detection

_DETECTION_TABLE = [
    # Review: more of a picture on screen is not a new set; a known character still is.
    ("이 장면 3장 더", None),
    ("이 의상 그대로 4장", None),
    ("more of this, 3 different shots", None),
    ("이 캐릭터로 다양한 포즈 5장", 5),
    ("다섯 장면으로 그려줘", 5),
    ("은발 엘프 궁수 캐릭터 하나 만들고 포즈 5개 테스트해줘", 5),
    ("내 캐릭터로 다양한 포즈 6장: 검은 단발, 빨간 눈", 6),
    ("비 오는 밤 네온사인 골목 배경은 그대로 두고, 다른 캐릭터 4명을 세워봐", 4),
    ("벚꽃 핀 학교 옥상 배경 고정하고 캐릭터만 바꿔서 3장: 남학생, 여학생, 선생님", 3),
    ("여러 표정 다섯 장 그려줘", 5),
    ("서로 다른 장면 4장", 4),
    ("같은 캐릭터로 다양한 포즈 5장", 5),
    ("draw 5 different scenes of a knight", 5),
    ("Make 3 images of different characters", 3),
    ("draw 4 poses of my elf", 4),
    ("12 different poses", 10),
    # Not a set: one picture repeated, seed variations, a single image, people in one image.
    ("같은 그림 5장", None),
    ("같은 구도로 다양한 표정 5장", None),
    ("시드만 바꿔서 다양하게 5장", None),
    ("고양이 그림 5장", None),
    ("5 variations of this cat", None),
    ("the same image 4 times with different seeds", None),
    ("a cat, 1 image", None),
    ("이미지 1장, 다양한 색감", None),
    ("두 명이 싸우는 장면 그려줘", None),
    ("a scene with 3 characters", None),
    ("세일러 교복 소녀 여러 장", None),
]


@pytest.mark.parametrize(("message", "expected"), _DETECTION_TABLE)
def test_detect_image_set(message: str, expected: int | None) -> None:
    """Killed by: src/uclone_x/tools/builtin/image_set_intent.py :: if _SAME_PICTURE.search(message):
    Becomes: if False:
    """
    assert detect_image_set(message) == expected


# ----------------------------------------------------------------------------- assembly


def test_assembly_puts_count_tags_then_differences_then_shared_then_quality() -> None:
    """Killed by: src/uclone_x/agent/image_set_planner.py :: ordered = [*own, *repeated, *plan.shared, *plan.style, *QUALITY_TAGS]
    Becomes: ordered = [*plan.shared, *plan.style, *own, *repeated, *QUALITY_TAGS]
    """
    plan = ImageSetPlan(
        shared=("1girl", "solo", "silver hair"),
        style=("anime coloring",),
        variants=(
            ImageVariant(action=("sitting",), location=("forest",), camera=("from above",)),
            ImageVariant(action=("running",), location=("forest",), camera=("from side",)),
        ),
    )
    prompts = assemble_prompts(plan)
    assert prompts == [
        "1girl, solo, sitting, from above, forest, silver hair, anime coloring, "
        + ", ".join(QUALITY_TAGS),
        "1girl, solo, running, from side, forest, silver hair, anime coloring, "
        + ", ".join(QUALITY_TAGS),
    ]


def test_assembly_takes_a_changing_subject_first_and_dedupes() -> None:
    plan = ImageSetPlan(
        shared=(),
        style=("Masterpiece",),
        variants=(
            ImageVariant(subject=("1boy", "black hair"), action=("standing",)),
            ImageVariant(subject=("1girl", "red hair"), action=("standing",)),
        ),
    )
    first, second = assemble_prompts(plan)
    assert first.startswith("1boy, black hair, standing")
    assert second.startswith("1girl, red hair, standing")
    assert first.count("masterpiece") == 1


def test_assembly_spells_tags_with_spaces_and_merges_spellings() -> None:
    """The model wrote `neon_street` and `pink_long_hair` live; CLIP was trained on spaces.

    Killed by: src/uclone_x/agent/image_set_planner.py :: return " ".join(tag.replace("_", " ").split()).lower()
    Becomes: return tag.strip().lower()
    """
    plan = ImageSetPlan(
        shared=("1girl", "neon_street", "neon street"),
        style=(),
        variants=(ImageVariant(action=("looking_at_viewer",)), ImageVariant(action=("running",))),
    )
    first, _ = assemble_prompts(plan)
    assert "_" not in first
    assert first.count("neon street") == 1
    assert "looking at viewer" in first


def test_schema_pins_the_variant_count() -> None:
    variants = plan_schema(4)["properties"]["variants"]
    assert (variants["minItems"], variants["maxItems"]) == (4, 4)


def test_bad_entries_names_non_tags() -> None:
    plan = ImageSetPlan(
        shared=("은발", "hair_black", "16:9", "black hair"),
        variants=(ImageVariant(action=("she is running very fast through the woods",)),),
    )
    assert bad_entries(plan) == [
        "은발",
        "hair_black",
        "16:9",
        "she is running very fast through the woods",
    ]


# ----------------------------------------------------------------------------- planning


class _Replies:
    """A model call answering from a fixed list and recording what it was sent."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.requests: list[LLMRequest] = []

    async def __call__(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return _reply(self.replies[len(self.requests) - 1])


@pytest.mark.asyncio
async def test_clean_plan_takes_one_structured_call() -> None:
    generate = _Replies(_plan_json(3))
    prompts = await plan_image_set(generate, "포즈 3개", 3, model="qwen3:8b")
    assert len(prompts) == 3
    (request,) = generate.requests
    assert request.response_schema is not None
    assert request.thinking is False
    assert request.tools == ()


@pytest.mark.asyncio
async def test_non_ascii_entry_triggers_one_repair_turn() -> None:
    """Killed by: src/uclone_x/agent/image_set_planner.py :: if bad or same_place or len(plan.variants) != count:
    Becomes: if False:
    """
    generate = _Replies(_plan_json(3, korean=True), _plan_json(3))
    prompts = await plan_image_set(generate, "포즈 3개", 3)
    assert len(generate.requests) == 2
    repair = generate.requests[1].messages[-1]
    assert repair.role is MessageRole.USER
    assert "은발" in (repair.content or "")
    assert generate.requests[1].messages[-2].role is MessageRole.ASSISTANT
    assert all(p.isascii() for p in prompts)


@pytest.mark.asyncio
async def test_non_ascii_left_after_repair_is_dropped() -> None:
    generate = _Replies(_plan_json(2, korean=True), _plan_json(2, korean=True))
    prompts = await plan_image_set(generate, "포즈 2개", 2)
    assert all(p.isascii() for p in prompts)
    assert len(generate.requests) == 2


@pytest.mark.asyncio
async def test_wrong_count_after_repair_raises() -> None:
    generate = _Replies(_plan_json(2), _plan_json(2))
    with pytest.raises(ImageSetPlanError):
        await plan_image_set(generate, "포즈 3개", 3)


@pytest.mark.asyncio
async def test_earlier_conversation_reaches_the_planner() -> None:
    """Killed by: src/uclone_x/agent/image_set_planner.py :: ChatMessage(role=MessageRole.USER, content=_with_earlier(message, earlier)),
    Becomes: ChatMessage(role=MessageRole.USER, content=message),
    """
    generate = _Replies(_plan_json(3))
    await plan_image_set(
        generate, "철수 다양한 포즈 3장", 3, earlier="user: 철수는 붉은 머리, 초록 눈"
    )
    sent = generate.requests[0].messages[-1].content or ""
    assert "붉은 머리" in sent
    assert sent.endswith("Request: 철수 다양한 포즈 3장")


@pytest.mark.asyncio
async def test_non_json_reply_twice_raises() -> None:
    generate = _Replies("sure! here are five poses", '{"shared": [')
    with pytest.raises(ImageSetPlanError):
        await plan_image_set(generate, "포즈 3개", 3)
    assert len(generate.requests) == 2


@pytest.mark.asyncio
async def test_one_broken_reply_gets_one_fresh_attempt() -> None:
    """qwen3:8b broke the JSON on 1 of 10 live plans; a second try is cheaper than no plan.

    Killed by: src/uclone_x/agent/image_set_planner.py :: second = await generate(request())
    Becomes: raise
    """
    generate = _Replies('{"shared": ["1girl"', _plan_json(3))
    prompts = await plan_image_set(generate, "포즈 3개", 3)
    assert len(prompts) == 3
    assert len(generate.requests) == 2
    assert generate.requests[1].messages == generate.requests[0].messages


# ----------------------------------------------------------------------------- connectors


def test_ollama_sends_the_schema_as_format() -> None:
    request = LLMRequest(
        model="qwen3:8b",
        messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
        response_schema=plan_schema(2),
    )
    payload = OllamaConnector()._build_payload(request)  # pyright: ignore[reportPrivateUsage]
    assert payload["format"] == plan_schema(2)


def test_connectors_without_a_mapping_refuse_the_schema() -> None:
    request = LLMRequest(
        model="m",
        messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
        response_schema=plan_schema(2),
    )
    with pytest.raises(StructuredOutputUnsupportedError):
        OpenAIConnector(api_key="k")._build_payload(request)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_mock_connector_refuses_the_schema_without_spending_a_reply() -> None:
    mock = MockLLMConnector(responses=["scripted"])
    request = LLMRequest(
        messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
        response_schema=plan_schema(2),
    )
    with pytest.raises(StructuredOutputUnsupportedError):
        await mock.generate(request)
    assert mock.call_count == 0


# ----------------------------------------------------------------------------- the agent


class _ImageParams(BaseModel):
    prompt: str | None = Field(default=None, description="one prompt")
    prompts: list[str] | None = Field(default=None, description="one prompt per image")
    count: int = Field(default=1, description="seeds of one prompt")


class _FakeGenerateImage(BaseTool[_ImageParams]):
    name = "generate_image"
    description = "generates images"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[_ImageParams] = []

    def run(self, params: _ImageParams, context: ToolContext) -> ToolResult:
        self.calls.append(params)
        return ToolResult(success=True, output="ok", provenance=_PROV)


class _PlanningLLM(BaseLLMConnector):
    """Answers a structured request with `plan` (or raises), and turns by the step.

    The turn's step replies are keyed on whether a tool result is already in the request,
    not on a queue running out: the first step calls the tool, the step after the result
    answers in text, exactly as a model that obeys would.
    """

    def __init__(self, plan: str | Exception) -> None:
        super().__init__()
        self._plan = plan
        self.structured: list[LLMRequest] = []
        self.turns: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        if request.response_schema is not None:
            self.structured.append(request)
            if isinstance(self._plan, Exception):
                raise self._plan
            return _reply(self._plan)
        self.turns.append(request)
        if any(m.role is MessageRole.TOOL for m in request.messages):
            return _reply("done")
        call = ToolCallRequest(id="c1", name="generate_image", arguments={"prompts": ["a", "b"]})
        return _reply(None, (call,))

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        resp = await self.generate(request)
        yield StreamChunk(
            delta_content=resp.content, finish_reason=resp.finish_reason, tool_calls=resp.tool_calls
        )


class _SheetParams(BaseModel):
    action: str = "get"


class _FakeCharacterSheet(BaseTool[_SheetParams]):
    name = "character_sheet"
    description = "reads a character sheet"

    def run(self, params: _SheetParams, context: ToolContext) -> ToolResult:
        return ToolResult(success=True, output="red hair, green eyes", provenance=_PROV)


class _SheetFirstLLM(_PlanningLLM):
    """Reads the character sheet first, as the artist persona is told to, then draws."""

    async def generate(self, request: LLMRequest) -> ModelResponse:
        if request.response_schema is not None:
            return await super().generate(request)
        self.turns.append(request)
        results = sum(m.role is MessageRole.TOOL for m in request.messages)
        if results == 0:
            return _reply(None, (ToolCallRequest(id="s1", name="character_sheet", arguments={}),))
        if results == 1:
            call = ToolCallRequest(
                id="g1", name="generate_image", arguments={"prompts": ["a", "b"]}
            )
            return _reply(None, (call,))
        return _reply("done")


def _agent(llm: BaseLLMConnector, tmp_path: Path, persona: str | None = "artist") -> BaseAgent:
    registry = ToolRegistry()
    registry.register(_FakeGenerateImage())
    registry.register(_FakeCharacterSheet())
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
        context=AgentContext(agent_id="artist", session_id="s1", workspace_root=tmp_path),
        personas=tuple(
            PersonaDefinition(
                name=name,
                role=name,
                system_prompt=name,
                allowed_tools=("generate_image",),
                enable_write_tools=True,
            )
            for name in ("artist", "writer")
        ),
    )


_ASK = "은발 엘프 궁수 캐릭터로 다양한 포즈 3장 그려줘"


@pytest.mark.asyncio
async def test_plan_note_follows_the_latest_user_message(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: turn_extra_sections = list(present_sections(undone_section, image_set_section))
    Becomes: turn_extra_sections = list(present_sections(undone_section))
    """
    llm = _PlanningLLM(_plan_json(3))
    result = await _agent(llm, tmp_path).execute_turn(_ASK)
    assert result.is_completed
    assert len(llm.structured) == 1

    first = llm.turns[0].messages
    system = [m for m in first if m.role is MessageRole.SYSTEM]
    assert all("[Image Set Plan]" not in (m.content or "") for m in system)
    last = first[-1]
    assert last.role is MessageRole.USER
    content = last.content or ""
    assert content.index(_ASK) < content.index("[Image Set Plan]")
    assert "prompts=" in content and "Do not use count" in content
    assert "pose 0" in content and "pose 2" in content

    # The step after the tool ran no longer carries the plan: it has been used.
    assert all("[Image Set Plan]" not in (m.content or "") for m in llm.turns[-1].messages)


@pytest.mark.asyncio
async def test_plan_survives_a_character_sheet_step(tmp_path: Path) -> None:
    """Review: the plan was dropped after any tool step, so a sheet read cost it.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: if plan in sections and any(e.tool_name == "generate_image" for e in executions):
    Becomes: if plan in sections:
    """
    llm = _SheetFirstLLM(_plan_json(3))
    result = await _agent(llm, tmp_path).execute_turn(_ASK)
    assert result.is_completed
    assert len(llm.turns) == 3
    assert "[Image Set Plan]" in (llm.turns[1].messages[-1].content or "")
    assert all("[Image Set Plan]" not in (m.content or "") for m in llm.turns[2].messages)


@pytest.mark.asyncio
async def test_planning_failure_leaves_the_turn_as_it_was(tmp_path: Path) -> None:
    llm = _PlanningLLM(LLMProviderError("connection refused at 127.0.0.1:11434"))
    result = await _agent(llm, tmp_path).execute_turn(_ASK)
    assert result.is_completed
    assert result.error is None
    assert len(llm.structured) == 1
    assert "127.0.0.1" not in result.content
    assert all("[Image Set Plan]" not in (m.content or "") for m in llm.turns[0].messages)


@pytest.mark.asyncio
async def test_unusable_plan_leaves_the_turn_as_it_was(tmp_path: Path) -> None:
    llm = _PlanningLLM("not json")
    result = await _agent(llm, tmp_path).execute_turn(_ASK)
    assert result.is_completed
    assert all("[Image Set Plan]" not in (m.content or "") for m in llm.turns[0].messages)


@pytest.mark.asyncio
async def test_other_personas_and_plain_requests_are_not_planned(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/base.py :: if llm is None or self._persona not in IMAGE_SET_PERSONAS:
    Becomes: if llm is None:
    """
    writer = _PlanningLLM(_plan_json(3))
    await _agent(writer, tmp_path, persona="writer").execute_turn(_ASK)
    assert writer.structured == []
    assert [d.name for d in writer.turns[0].tools] == ["generate_image"]

    plain = _PlanningLLM(_plan_json(3))
    await _agent(plain, tmp_path).execute_turn("은발 엘프 궁수 한 장 그려줘")
    assert plain.structured == []


def test_earlier_skips_the_request_and_system_turns(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/base.py :: if not picked and entry.role is MessageRole.USER and entry.content == message:
    Becomes: if False:
    """
    agent = _agent(_PlanningLLM(_plan_json(3)), tmp_path)
    agent._history.extend(  # pyright: ignore[reportPrivateUsage]
        [
            ChatMessage(role=MessageRole.SYSTEM, content="persona"),
            ChatMessage(role=MessageRole.USER, content="철수는 붉은 머리"),
            ChatMessage(role=MessageRole.ASSISTANT, content="알겠습니다"),
            ChatMessage(role=MessageRole.USER, content=_ASK),
        ]
    )
    earlier = agent._image_set_earlier(_ASK)  # pyright: ignore[reportPrivateUsage]
    assert earlier == "user: 철수는 붉은 머리\nassistant: 알겠습니다"


# ----------------------------------------------------------------------------- scene sets move

_SCENE = "은발 엘프 궁수로 다양한 장면 3장 그려줘"


def _scene_plan(*places: str, shared: tuple[str, ...] = ()) -> str:
    return json.dumps(
        {
            "shared": ["1girl", "silver hair", *shared],
            "style": ["anime coloring"],
            "variants": [
                {
                    "subject": [],
                    "action": [f"pose {i}"],
                    "expression": ["smile"],
                    "location": [place],
                    "camera": ["full body"],
                }
                for i, place in enumerate(places)
            ],
        }
    )


_LOCATION_TABLE = [
    (_SCENE, True),
    ("다섯 장면으로 그려줘", True),
    ("draw 3 different scenes of a knight and a dragon", True),
    ("여러 장소에서 여행하는 소녀 4장", True),
    # A pose or expression set stays where it is.
    ("은발 엘프 궁수 다양한 포즈 5장", False),
    ("고양이 소녀 표정 4가지", False),
    # The user pinned the place.
    ("배경은 그대로 두고 다른 장면 3개", False),
    ("교실에서만 다양한 장면 4장", False),
    ("same background, 3 different scenes", False),
    ("비 오는 네온 골목 배경 고정하고 서로 다른 캐릭터 4명", False),
]


@pytest.mark.parametrize(("message", "expected"), _LOCATION_TABLE)
def test_wants_varied_locations(message: str, expected: bool) -> None:
    """Killed by: src/uclone_x/tools/builtin/image_set_intent.py :: return bool(_SCENE_WORD.search(message)) and not _FIXED_LOCATION.search(message)
    Becomes: return bool(_SCENE_WORD.search(message))
    """
    assert wants_varied_locations(message) is expected


def test_repeated_locations() -> None:
    def plan(*places: tuple[str, ...]) -> ImageSetPlan:
        return ImageSetPlan(variants=tuple(ImageVariant(location=p) for p in places))

    assert not repeated_locations(plan(("forest",), ("castle gate",)))
    assert repeated_locations(plan(("forest",), ("Forest",)))
    assert repeated_locations(plan(("forest",), ()))


@pytest.mark.asyncio
async def test_a_scene_set_is_told_to_put_each_image_somewhere_else() -> None:
    """Live on qwen3:8b, "다양한 장면 5장" put one forest clearing in all five images.

    Killed by: src/uclone_x/agent/image_set_planner.py :: + (_VARIED_LOCATIONS if vary_locations else "")
    Becomes: + ""
    """
    generate = _Replies(_scene_plan("forest", "castle gate", "tavern"))
    prompts = await plan_image_set(generate, _SCENE, 3)
    assert len(generate.requests) == 1
    assert "Every variant needs its own location" in (
        generate.requests[0].messages[0].content or ""
    )
    assert ["forest" in p for p in prompts] == [True, False, False]


@pytest.mark.asyncio
async def test_a_pose_set_keeps_one_place() -> None:
    """Killed by: src/uclone_x/agent/image_set_planner.py :: vary_locations = wants_varied_locations(message)
    Becomes: vary_locations = True
    """
    generate = _Replies(_plan_json(3))
    prompts = await plan_image_set(generate, "은발 엘프 궁수 다양한 포즈 3장", 3)
    assert len(generate.requests) == 1
    assert "own location" not in (generate.requests[0].messages[0].content or "")
    assert all("forest" in p for p in prompts)


@pytest.mark.asyncio
async def test_a_scene_set_in_one_place_gets_a_repair_turn() -> None:
    """Killed by: src/uclone_x/agent/image_set_planner.py :: same_place = vary_locations and repeated_locations(plan)
    Becomes: same_place = False
    """
    generate = _Replies(
        _scene_plan("forest", "forest", "forest"), _scene_plan("forest", "cave", "tavern")
    )
    prompts = await plan_image_set(generate, _SCENE, 3)
    assert len(generate.requests) == 2
    repair = generate.requests[1].messages[-1].content or ""
    assert "different place" in repair
    # The tags were fine; the repair asks only about places.
    assert "not valid English Danbooru tags" not in repair
    assert ["cave" in p for p in prompts] == [False, True, False]


@pytest.mark.asyncio
async def test_a_place_left_in_shared_does_not_reach_the_other_scenes() -> None:
    """Live, qwen3:8b kept "forest" in shared beside "castle gate" and "tavern" as locations.

    Killed by: src/uclone_x/agent/image_set_planner.py :: plan = _without_shared_places(plan)
    Becomes: pass
    """
    generate = _Replies(
        _scene_plan("forest clearing", "castle gate", "tavern", shared=("forest", "long hair"))
    )
    prompts = await plan_image_set(generate, _SCENE, 3)
    assert "forest" not in prompts[1]
    assert "forest" not in prompts[2]
    assert all("long hair" in p for p in prompts)
