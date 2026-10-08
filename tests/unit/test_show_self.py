"""`show_self`: a clone draws itself in a scene from its own self facts (#2017).

clone-self-and-scenes §4. What these pin:

* **Projection** (§4.1): hair, eyes, build, appearance and outfit facts become tags in that
  order; gender sets gender; other facts and retracted facts do not draw; an `outfit`
  argument replaces the outfit facts for one picture.
* **One person** (§4.2): one base seed per clone, a Danbooru tag prompt through the
  character composer or a plain sentence for other families, a fresh `self_` file in the
  artifacts folder.
* **Cost** (§4.3): at most one picture per reply, per agent and session.
* **The guard** (§4.4): no age or an age under 18 draws nothing; a sexual scene argument is
  refused when a fact or argument codes the clone as a minor. Tests classify neutral
  markers through the term-set parameters; no test writes sexual content.
* **Plain words**: every refusal reads as a sentence, never a class name or a path.
* **Wiring**: only an agent with memory and an image tool has `show_self`; it reads only
  the clone's own self facts; another agent sharing the registry cannot reach it; a clone
  built by `build_clone` knows whether it has an avatar.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.clone_builder import AppScope, build_clone
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.self_scene import (
    DRAW_FAILED_TEXT,
    MINOR_SEXUAL_TEXT,
    MINOR_TERMS,
    NO_AGE_TEXT,
    NO_REFERENCE_TEXT,
    ONCE_PER_TURN_TEXT,
    REFERENCE_UNKNOWN_TEXT,
    REFERENCE_UNUSED_TEXT,
    SHOW_SELF_TOOL,
    ShowSelfTool,
    check_self_scene,
    clone_seed,
    project_self,
    self_scene_path,
)
from uclone_x.agent.session import SessionStore
from uclone_x.core.models import AGENT_COMPOSED_TOOLS
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import EventBus
from uclone_x.errors import PlainRefusalError
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.memory.models import MemoryFact
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.media_registry import ModelProfile, PromptFamily
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import ToolRegistry, create_default_registry

_PROV = Provenance.primary(provider="scripted", model="scripted")
_MARKERS: dict[str, Any] = {
    "minor_terms": {"marker-minor"},
    "sexual_terms": {"marker-sexual"},
}


def _fact(
    predicate: str, value: str, *, subject: str = "self", retracted: bool = False
) -> MemoryFact:
    return MemoryFact(
        subject=subject,
        predicate=predicate,
        object_value=value,
        provenance=_PROV,
        source_session_id="s0",
        retracted=retracted,
    )


_ADULT = _fact("age", "30")


class _ImageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    negative_prompt: str | None = None
    style: str | None = None
    seed_override: int | None = None
    aspect_ratio: str | None = None
    output_path: str | None = None


class _FakeImageTool(BaseTool[_ImageParams]):
    """A `generate_image` stand-in: records each request and answers like the real one."""

    name = "generate_image"
    writes_files: ClassVar[bool] = False
    description = "Draws a picture."
    params_type = _ImageParams

    def __init__(
        self,
        family: PromptFamily = PromptFamily.DANBOORU,
        *,
        error: str | None = None,
        output: dict[str, Any] | None = None,
        families: dict[str, PromptFamily] | None = None,
    ):
        super().__init__(name=self.name, params_type=_ImageParams)
        self.family = family
        #: A clone's own picture model -> its family; any other follows ``family`` (the default's).
        self.families = families or {}
        self.error = error
        self.output = output
        self.requests: list[_ImageParams] = []

    def active_profile(self, image_model: str | None = None) -> ModelProfile:
        family = self.families.get(image_model, self.family) if image_model else self.family
        return ModelProfile(model_id="m", display_name="M", family=family)

    def bind_skill_registry(self, registry: Any) -> None:
        return None

    async def run(self, params: _ImageParams, context: ToolContext) -> ToolResult:
        self.requests.append(params)
        if self.output is not None:
            return ToolResult(
                success=self.error is None,
                output=self.output,
                error=self.error,
                provenance=_PROV,
            )
        if self.error is not None:
            return ToolResult(success=False, error=self.error, provenance=_PROV)
        path = params.output_path or "artifacts/images/x.png"
        return ToolResult(
            success=True,
            output={"status": "success", "relative_url": path, "prompt": params.prompt},
            artifacts=(path,),
            provenance=_PROV,
        )


def _tool(
    facts: list[MemoryFact],
    image: _FakeImageTool | None = None,
    *,
    avatar: bool | None = False,
) -> tuple[ShowSelfTool, _FakeImageTool]:
    image = image or _FakeImageTool()
    tool = ShowSelfTool(
        clone_id="rin",
        image_tool=image,
        facts=lambda: facts,
        avatar_present=None if avatar is None else (lambda: avatar),
    )
    return tool, image


def _ctx(turn: int = 1, session: str = "sess") -> ToolContext:
    return ToolContext(agent_id="rin", session_id=session, turn_index=turn)


_SCENE = {"place": "a cafe", "action": "reading"}


def _assert_plain(text: str) -> None:
    for internal in ("Error", "Exception", "Traceback", "show_self", "/", "\\", "_"):
        assert internal not in text, (internal, text)


# --------------------------------------------------------------------------------------
# Projection (§4.1)
# --------------------------------------------------------------------------------------


def test_appearance_facts_draw_in_relation_order_and_nothing_else_does() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: if fact.retracted:
    Becomes: if False:
    """
    visual = project_self(
        [
            _fact("outfit", "raincoat"),
            _fact("personality", "curious"),
            _fact("hair", "red hair, long hair"),
            _fact("hair", "blue hair", retracted=True),
            _fact("eyes", "green eyes"),
            _fact("age", "30"),
            _fact("eyes", "Green eyes"),
        ]
    )

    assert visual.tags == ("red hair", "long hair", "green eyes", "raincoat")
    assert visual.gender is None


def test_an_outfit_argument_replaces_the_outfit_facts_for_one_picture() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: by_relation["outfit"] = _tags_of(outfit)
    Becomes: by_relation["outfit"].extend(_tags_of(outfit))
    """
    facts = [_fact("gender", "여성"), _fact("hair", "red hair"), _fact("outfit", "raincoat")]

    visual = project_self(facts, outfit="swimsuit")

    assert visual.tags == ("red hair", "swimsuit")
    assert visual.gender == "female"
    assert project_self(facts).tags == ("red hair", "raincoat")


def test_the_base_seed_is_one_fixed_number_per_clone() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: return int.from_bytes(hashlib.sha256(clone_id.encode("utf-8")).digest()[:4], "big")
    Becomes: return int.from_bytes(hashlib.sha256(clone_id.encode("utf-8")).digest()[:4], "little")
    """
    # The first four bytes of sha256(clone id), read big-endian: a restart draws the same person.
    prefix = hashlib.sha256(b"rin").digest()[:4]
    assert clone_seed("rin") == int.from_bytes(prefix, "big")
    assert clone_seed("rin") != clone_seed("mio")


# --------------------------------------------------------------------------------------
# The guard (§4.4)
# --------------------------------------------------------------------------------------


def test_a_clone_with_no_age_is_not_drawn() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: if not ages:
    Becomes: if False:
    """
    with pytest.raises(PlainRefusalError) as refused:
        check_self_scene([_fact("hair", "red hair"), _fact("age", "unknown")], ["a cafe"])

    assert refused.value.reason_code == "no_age"
    assert str(refused.value) == NO_AGE_TEXT


def test_birth_year_is_parsed_against_current_year_and_under_eighteen_is_refused() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: age = current_year - year
    Becomes: age = current_year + year
    """
    fixed_now = datetime(2026, 6, 1, tzinfo=UTC)
    for minor_age in ("born 2010", "2010년생"):
        with pytest.raises(PlainRefusalError) as refused:
            check_self_scene([_fact("age", minor_age)], ["a cafe"], now=fixed_now)
        assert refused.value.reason_code == "under_age"
        assert "16" in str(refused.value)
        _assert_plain(str(refused.value))

    for adult_age in ("born 1995", "1995년생"):
        check_self_scene([_fact("age", adult_age)], ["a cafe"], now=fixed_now)


def test_unclear_or_relative_age_is_refused_as_no_age() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: if _UNCLEAR_AGE_PATTERN.search(folded):
    Becomes: if False:
    """
    for unclear in ("under 18", "18세 미만"):
        with pytest.raises(PlainRefusalError) as refused:
            check_self_scene([_fact("age", unclear)], ["a cafe"])
        assert refused.value.reason_code == "no_age"
        assert str(refused.value) == NO_AGE_TEXT


def test_a_clone_under_eighteen_is_not_drawn_and_the_youngest_age_counts() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: youngest = min(ages)
    Becomes: youngest = max(ages)
    """
    with pytest.raises(PlainRefusalError) as refused:
        check_self_scene([_fact("age", "30"), _fact("age", "17 years old")], ["a cafe"])

    assert refused.value.reason_code == "under_age"
    assert "17" in str(refused.value)
    _assert_plain(str(refused.value))


def test_eighteen_is_drawn() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: if youngest < ADULT_AGE:
    Becomes: if youngest <= ADULT_AGE:
    """
    check_self_scene([_fact("age", "18")], ["a cafe"])


def test_a_sexual_scene_of_a_minor_coded_clone_is_refused_whatever_the_age() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: described = [fact.object_value for fact in active] + list(scene)
    Becomes: described = list(scene)
    """
    facts = [_ADULT, _fact("outfit", "marker-minor")]

    with pytest.raises(PlainRefusalError) as refused:
        check_self_scene(facts, ["a cafe", "marker-sexual pose"], **_MARKERS)

    assert refused.value.reason_code == "minor_coded"
    assert str(refused.value) == MINOR_SEXUAL_TEXT


def test_a_minor_marker_alone_or_a_sexual_marker_alone_is_not_refused() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: if not any(_mentions(part, sexual_terms) for part in scene):
    Becomes: if False:
    """
    check_self_scene([_ADULT, _fact("outfit", "marker-minor")], ["a cafe"], **_MARKERS)
    check_self_scene([_ADULT], ["marker-sexual pose"], **_MARKERS)


def test_a_latin_term_matches_only_as_a_whole_word() -> None:
    r"""Killed by: src/uclone_x/agent/self_scene.py :: if re.search(rf"(?<![\w]){re.escape(key)}(?![\w])", folded):
    Becomes: if key in folded:
    """
    facts = [_ADULT, _fact("appearance", "kidney-shaped hairclip")]

    check_self_scene(facts, ["marker-sexual"], minor_terms={"kid"}, sexual_terms={"marker-sexual"})
    with pytest.raises(PlainRefusalError):
        check_self_scene(
            [_ADULT, _fact("appearance", "a Kid at heart")],
            ["marker-sexual"],
            minor_terms={"kid"},
            sexual_terms={"marker-sexual"},
        )


def test_a_hangul_term_matches_inside_a_word() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: elif key in folded:
    Becomes: elif False:
    """
    with pytest.raises(PlainRefusalError):
        check_self_scene(
            [_ADULT, _fact("outfit", "교복을 입음")],
            ["marker-sexual"],
            minor_terms={"교복"},
            sexual_terms={"marker-sexual"},
        )


def test_minor_terms_include_student_and_teen_terms() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: "teen",
    Becomes: "not_a_teen",
    """
    expected = {"high school", "student", "teen", "teenager", "학생", "10대"}
    assert expected.issubset(MINOR_TERMS)


def test_student_and_teen_descriptions_refuse_sexual_scene_with_neutral_marker() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: "학생",
    Becomes: "not_a_student",
    """
    for desc in ("high school student", "학생", "teen", "10대"):
        with pytest.raises(PlainRefusalError) as refused:
            check_self_scene(
                [_ADULT, _fact("appearance", desc)],
                ["marker-sexual"],
                sexual_terms={"marker-sexual"},
            )
        assert refused.value.reason_code == "minor_coded"
        assert str(refused.value) == MINOR_SEXUAL_TEXT

    check_self_scene([_ADULT, _fact("appearance", "high school student")], ["a cafe"])


# --------------------------------------------------------------------------------------
# The tool: plain refusals, once per turn, the file, the prompt, the result
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refusal_reaches_the_model_as_plain_words_and_nothing_is_drawn() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: raise PlainRefusalError(NO_AGE_TEXT, reason_code="no_age")
    Becomes: raise ValueError(NO_AGE_TEXT)
    """
    tool, image = _tool([_fact("hair", "red hair")])

    result = await tool.execute(_SCENE, _ctx())

    assert not result.success
    assert result.error == NO_AGE_TEXT
    _assert_plain(result.error or "")
    assert image.requests == []


@pytest.mark.asyncio
async def test_one_picture_per_reply() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: if self._drawn_turns.get(key) == turn:
    Becomes: if False:
    """
    tool, image = _tool([_ADULT])

    first = await tool.execute(_SCENE, _ctx(turn=3))
    second = await tool.execute(_SCENE, _ctx(turn=3))
    next_reply = await tool.execute(_SCENE, _ctx(turn=4))

    assert first.success and next_reply.success
    assert second.error == ONCE_PER_TURN_TEXT
    _assert_plain(second.error or "")
    assert len(image.requests) == 2


@pytest.mark.asyncio
async def test_the_once_per_reply_limit_is_per_session() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: key = (context.agent_id, context.session_id)
    Becomes: key = (context.agent_id, "")
    """
    tool, image = _tool([_ADULT])

    await tool.execute(_SCENE, _ctx(turn=3, session="a"))
    other = await tool.execute(_SCENE, _ctx(turn=3, session="b"))

    assert other.success
    assert len(image.requests) == 2


@pytest.mark.asyncio
async def test_each_picture_gets_a_fresh_self_file_never_the_avatar() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: return f"artifacts/images/self_{_slug(clone_id)}_{stamp}_{suffix}.png"
    Becomes: return "artifacts/images/avatar.png"
    """
    tool, image = _tool([_ADULT])

    await tool.execute(_SCENE, _ctx(turn=1))
    await tool.execute(_SCENE, _ctx(turn=2))

    paths = [r.output_path or "" for r in image.requests]
    for path in paths:
        assert re.fullmatch(r"artifacts/images/self_rin_\d{14}_[0-9a-f]{6}\.png", path), path
    assert paths[0] != paths[1]


def test_the_file_stamp_is_utc() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: stamp = now.astimezone(UTC).strftime("%Y%m%d%H%M%S")
    Becomes: stamp = now.strftime("%Y%m%d%H%M%S")
    """
    seoul = datetime(2026, 9, 30, 9, 5, 7, tzinfo=timezone(timedelta(hours=9)))

    assert self_scene_path("Rin Clone", seoul, "abc123") == (
        "artifacts/images/self_rin-clone_20260930000507_abc123.png"
    )


@pytest.mark.asyncio
async def test_a_tag_model_gets_the_composed_tag_prompt_with_the_clone_seed() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: if self._family(image_model) is PromptFamily.DANBOORU:
    Becomes: if False:
    """
    facts = [_ADULT, _fact("gender", "female"), _fact("hair", "red hair")]
    tool, image = _tool(facts, _FakeImageTool(PromptFamily.DANBOORU))

    await tool.execute({**_SCENE, "expression": "smiling"}, _ctx())

    request = image.requests[0]
    assert request.prompt.startswith(
        "1girl, solo, red hair, reading, in a cafe, smiling expression"
    )
    assert request.negative_prompt
    assert request.style == "anime"
    assert request.seed_override == clone_seed("rin")
    assert request.aspect_ratio == "3:4"


@pytest.mark.asyncio
async def test_a_prose_model_gets_one_plain_sentence() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: if self._family(image_model) is PromptFamily.DANBOORU:
    Becomes: if True:
    """
    facts = [
        _ADULT,
        _fact("gender", "female"),
        _fact("hair", "red hair"),
        _fact("eyes", "green eyes"),
    ]
    tool, image = _tool(facts, _FakeImageTool(PromptFamily.NATURAL_PROSE))

    await tool.execute({"place": "at the station", "action": "waiting"}, _ctx())

    assert image.requests[0].prompt == (
        "A woman (red hair, green eyes), waiting, at the station. One person, the same character."
    )
    assert image.requests[0].seed_override == clone_seed("rin")


@pytest.mark.asyncio
async def test_the_prompt_follows_the_clones_own_picture_model_not_the_default() -> None:
    """A clone on a prose model over a tag default gets a sentence, not tags (#2176).

    Killed by: src/uclone_x/agent/self_scene.py :: return self._image_tool.active_profile(image_model).family
    Becomes: return self._image_tool.active_profile().family
    """
    facts = [_ADULT, _fact("gender", "female"), _fact("hair", "red hair")]
    image = _FakeImageTool(
        PromptFamily.DANBOORU,
        families={"gemini/gemini-2.5-flash-image": PromptFamily.NATURAL_PROSE},
    )
    tool, _ = _tool(facts, image)
    own = ToolContext(
        agent_id="rin",
        session_id="sess",
        turn_index=1,
        image_model="gemini/gemini-2.5-flash-image",
    )

    await tool.execute({"place": "at the station", "action": "waiting"}, own)

    request = image.requests[0]
    assert request.prompt == (
        "A woman (red hair), waiting, at the station. One person, the same character."
    )
    assert request.style != "anime"


@pytest.mark.asyncio
async def test_the_result_is_compact_and_keeps_the_declared_file() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: return drawn.model_copy(update={"output": result})
    Becomes: return ToolResult(success=True, output=result, provenance=drawn.provenance)
    """
    tool, image = _tool([_ADULT, _fact("hair", "red hair")], avatar=True)

    result = await tool.execute({**_SCENE, "expression": "calm", "outfit": "a raincoat"}, _ctx())

    url = image.requests[0].output_path
    assert result.output == {
        "status": "success",
        "relative_url": url,
        "drawn": "you, reading, in a cafe, calm, wearing a raincoat",
        "reference_image": REFERENCE_UNUSED_TEXT,
    }
    assert result.artifacts == (url,)


@pytest.mark.asyncio
async def test_the_result_says_whether_an_avatar_exists() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: return REFERENCE_UNUSED_TEXT if self._avatar_present() else NO_REFERENCE_TEXT
    Becomes: return REFERENCE_UNUSED_TEXT
    """
    without, _ = _tool([_ADULT], avatar=False)
    unknown, _ = _tool([_ADULT], avatar=None)

    assert _reference(await without.execute(_SCENE, _ctx())) == NO_REFERENCE_TEXT
    assert _reference(await unknown.execute(_SCENE, _ctx())) == REFERENCE_UNKNOWN_TEXT


def _reference(result: ToolResult) -> object:
    assert isinstance(result.output, dict)
    return result.output["reference_image"]


@pytest.mark.asyncio
async def test_an_internal_image_failure_reads_as_one_plain_sentence() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: if not error or error.startswith(_INTERNAL_FAILURE_PREFIXES) or "Traceback" in error:
    Becomes: if not error:
    """
    broken = _FakeImageTool(
        error="Tool execution failed for 'generate_image': RuntimeError: /srv/engine/x.py"
    )
    plain = _FakeImageTool(error="The image engine is not running. Start it in Settings.")
    first, _ = _tool([_ADULT], broken)
    second, _ = _tool([_ADULT], plain)

    assert (await first.execute(_SCENE, _ctx())).error == DRAW_FAILED_TEXT
    assert (await second.execute(_SCENE, _ctx())).error == plain.error


@pytest.mark.asyncio
async def test_plain_failure_preserves_plain_message_without_url() -> None:
    """Killed by: src/uclone_x/agent/self_scene.py :: if not error and isinstance(result.output, dict):
    Becomes: if False:
    """
    tool, _ = _tool([_ADULT], _FakeImageTool(output={"message": "Model is busy. Try again."}))

    result = await tool.execute(_SCENE, _ctx())

    assert not result.success
    assert result.error == "Model is busy. Try again."


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


def _agent(
    registry: ToolRegistry, memory: CrossSessionMemory | None, agent_id: str = "rin"
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id=agent_id,
            name=agent_id,
            llm_config=AgentLLMConfig(model_name="mock", auto_compact=False),
        ),
        llm=MockLLMConnector(default_response="Done."),
        tools=registry,
        memory=memory,
    )


@pytest.mark.asyncio
async def test_the_agent_draws_from_its_own_self_facts_only() -> None:
    """Killed by: src/uclone_x/agent/base.py :: facts=lambda: memory.list_facts(subject=SELF_SUBJECT),
    Becomes: facts=lambda: memory.list_facts(),
    """
    memory = CrossSessionMemory()
    memory.record_fact("self", "age", "30", _PROV, "s0", origin="told")
    memory.record_fact("user", "age", "12", _PROV, "s0", origin="told")
    registry = ToolRegistry()
    registry.register(_FakeImageTool())

    record = await _agent(registry, memory).execute_tool_call("show_self", _SCENE)

    assert record.error is None
    assert record.status.value == "success"


@pytest.mark.asyncio
async def test_an_agent_without_an_image_tool_has_no_show_self() -> None:
    """Killed by: src/uclone_x/agent/base.py :: if isinstance(image_tool, ImageModelSource):
    Becomes: if True:
    """
    agent = _agent(ToolRegistry(), CrossSessionMemory())

    with pytest.raises(KeyError):
        await agent.execute_tool_call("show_self", _SCENE)


@pytest.mark.asyncio
async def test_another_agent_on_the_shared_registry_cannot_reach_this_clone_s_show_self() -> None:
    """Killed by: src/uclone_x/agent/base.py :: ShowSelfTool,
    Becomes: LoadSkillTool,
    """
    registry = ToolRegistry()
    registry.register(_FakeImageTool())
    _agent(registry, CrossSessionMemory())
    stranger = _agent(registry, None, agent_id="mio")

    with pytest.raises(KeyError):
        await stranger.execute_tool_call("show_self", _SCENE)


def test_a_built_clone_knows_whether_it_has_an_avatar(tmp_path: Path) -> None:
    """The shipped writer has a picture, so its `show_self` reports one (§4.2).

    Killed by: src/uclone_x/agent/clone_builder.py :: avatar_present=_avatar_lookup(app.persona_registry, persona_def.name),
    Becomes: avatar_present=None,
    """
    registry = ToolRegistry()
    registry.register(_FakeImageTool())
    host = HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(default_response="Done."),
        tools=registry,
        tracer=TelemetryTracer(),
        store=SessionStore(tmp_path / "sessions"),
    )
    app = AppScope(
        host=host,
        workspace_root=tmp_path,
        persona_registry=PersonaRegistry(workspace_root=tmp_path),
        memory_for=lambda _clone: CrossSessionMemory(),
    )

    built = build_clone(app, clone_id="writer", session_id="sess")

    tool = built.agent._tool_invoker.resolve("show_self")  # pyright: ignore[reportPrivateUsage]
    assert isinstance(tool, ShowSelfTool)
    assert tool._reference_image() == REFERENCE_UNUSED_TEXT  # pyright: ignore[reportPrivateUsage]


def test_a_persona_naming_show_self_loads_against_a_registry_that_lacks_it(
    tmp_path: Path,
) -> None:
    """`show_self` is composed per agent, so no shared registry holds it when artist loads.

    Killed by: src/uclone_x/agent/persona_registry.py :: frozenset(tool_names) | AGENT_COMPOSED_TOOLS if tool_names is not None else None
    Becomes: frozenset(tool_names) if tool_names is not None else None
    """
    names = [t.name for t in create_default_registry(enable_mcp=False).list_tools()]
    assert SHOW_SELF_TOOL not in names
    assert SHOW_SELF_TOOL in AGENT_COMPOSED_TOOLS

    artist = PersonaRegistry(workspace_root=tmp_path, tool_names=names).get_persona("artist")

    assert artist is not None
    assert SHOW_SELF_TOOL in artist.allowed_tools
