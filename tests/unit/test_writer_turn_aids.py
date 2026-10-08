"""What code adds to a Writer turn instead of asking a small model to (#1808).

Three aids, each for a failure the 2026-09-28 qwen3:8b Writer eval measured: the next
scene's material put in the turn context (`story/scene_turn.py`), lines appended to the
final reply (`agent/reply_lines.py`: a tool's `reply_note`, and the story-facts line),
and an empty reply asked again once (`TurnExecutor._late_nudge`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, JsonValue

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.clone_builder import with_app_lifecycle_hooks
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    ToolExecutionRecord,
)
from uclone_x.agent.protocols import TurnAidHookProtocol, TurnAidProtocol
from uclone_x.agent.reply_lines import is_korean, lines_to_add, reply_notes, with_lines
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.compactor import estimate_text_tokens
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.story.library import StoryLibrary
from uclone_x.story.muse import MuseSparkTool
from uclone_x.story.scene_turn import SceneTurnHook, scene_section
from uclone_x.story.work import StoryWork
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import REPLY_NOTE_KEY, ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "writer"
STORY = "moon-seal"
CONFLICTING = "예린과 월광검으로 싸우는 장면을 써줘"
WITH_STORY_CONTEXT = frozenset({"story_context", "story_manuscript"})

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="scripted", model="m"),
    served_by=ServiceRef(provider="scripted", model="m"),
    attempts=(),
)
_USAGE = TokenUsage(provider="scripted", model="m", input_tokens=0, output_tokens=0)

# Statically checked: the story's hook is a turn-aid hook, and its aid a turn aid.
_hook: TurnAidHookProtocol = SceneTurnHook()


def _workspace(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    return tmp_path


def _aid(tmp_path: Path, message: str = CONFLICTING, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "message": message,
        "story_id": STORY,
        "room_id": None,
        "workspace_root": _workspace(tmp_path),
        "tool_names": WITH_STORY_CONTEXT,
    }
    kwargs.update(overrides)
    return SceneTurnHook().turn_aid(**kwargs)


def _bundle_of(section: str) -> dict[str, Any]:
    return json.loads(section.splitlines()[-1])


def _write_record(scene_id: str, status: ToolResultStatus = ToolResultStatus.SUCCESS) -> Any:
    return ToolExecutionRecord(
        tool_name="story_manuscript",
        arguments={"action": "write", "scene_id": scene_id},
        output={"scene_id": scene_id},
        status=status,
    )


# --- the scene section ------------------------------------------------------------------


def test_an_open_story_gives_the_next_scene_with_the_request_conflicts_first(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/story/scene_turn.py :: request=message or None,
    Becomes: request=None,
    """
    aid = _aid(tmp_path)
    _conforms: TurnAidProtocol = aid

    assert aid.scene_id == "ch02.s02"
    assert "ch02.s02 「모닥불」" in aid.section
    assert "the prose says they are dead" in aid.section
    bundle = _bundle_of(aid.section)
    assert next(iter(bundle)) == "request_conflicts"
    assert any("예린" in line for line in bundle["request_conflicts"])
    assert bundle["scene"]["id"] == "ch02.s02"


@pytest.mark.parametrize(
    ("message", "scene_id"),
    [
        ("'달의 봉인'의 씬 ch02.s04 「새벽 수련」을 써 줘.", "ch02.s04"),
        ("ch02.s03를 다시 써 줘", "ch02.s03"),
        ("「새벽 수련」 장면을 써 줘", "ch02.s04"),
    ],
    ids=["id-and-title", "id-with-particle", "title-only"],
)
def test_a_message_that_names_a_scene_gets_that_scene(
    tmp_path: Path, message: str, scene_id: str
) -> None:
    """Killed by: src/uclone_x/story/scene_turn.py :: named = _named_scene(outline, message)
    Becomes: named = None
    """
    aid = _aid(tmp_path, message)

    assert aid.scene_id == scene_id
    assert f"The scene this message names is {scene_id}" in aid.section
    assert _bundle_of(aid.section)["scene"]["id"] == scene_id


def test_an_id_inside_a_longer_id_is_not_a_name(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/story/scene_turn.py :: {re.escape(scene.id)}(?![A-Za-z0-9_])
    Becomes: {re.escape(scene.id)}
    """
    aid = _aid(tmp_path, "xch02.s04 와 ch02.s040 은 없는 씬이야")

    assert aid.scene_id == "ch02.s02"
    assert "The next unwritten scene of the open story is ch02.s02" in aid.section


@pytest.mark.parametrize(
    "overrides",
    [
        {"story_id": None},
        {"workspace_root": None},
        {"tool_names": frozenset({"story_manuscript"})},
    ],
    ids=["no-story", "no-workspace", "no-story-context"],
)
def test_no_open_story_or_no_story_context_adds_nothing(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    """Killed by: src/uclone_x/story/scene_turn.py :: or _STORY_CONTEXT not in tool_names:
    Becomes: or False:
    """
    assert _aid(tmp_path, **overrides) is None


def test_a_story_that_cannot_be_read_adds_nothing_and_does_not_raise(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Killed by: src/uclone_x/story/scene_turn.py :: except Exception:
    Becomes: except KeyError:
    """
    assert _aid(tmp_path, story_id="no-such-story") is None
    assert "could not be read" in caplog.text


def test_a_story_with_every_scene_written_adds_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Said in the log as what it is, not as a story that could not be read.

    Killed by: src/uclone_x/story/scene_turn.py :: if following is None:
    Becomes: if False:
    """
    caplog.set_level(logging.INFO, logger="uclone_x.story.scene_turn")
    workspace = _workspace(tmp_path)
    manuscripts = workspace / "stories" / STORY / "manuscript"
    work = StoryWork(StoryLibrary(workspace), STORY)
    current = work.outline()
    assert current is not None
    for _, scene in current[0].scenes_in_order():
        (manuscripts / f"{scene.id}.md").write_text("Written.", encoding="utf-8")

    aid = SceneTurnHook().turn_aid(
        message=CONFLICTING,
        story_id=STORY,
        room_id=None,
        workspace_root=workspace,
        tool_names=WITH_STORY_CONTEXT,
    )
    assert aid is None
    assert "no unwritten scene left" in caplog.text
    assert "could not be read" not in caplog.text


def test_a_section_over_the_cap_drops_the_neighbours_text_first_and_says_so() -> None:
    """Killed by: src/uclone_x/story/scene_turn.py :: _DROP_FIRST = ("end_of_previous_scene", "previous_scene", "next_scene", "manifest")
    Becomes: _DROP_FIRST = ("manifest", "end_of_previous_scene", "previous_scene", "next_scene")
    """
    bundle = {
        "request_conflicts": ["kept"],
        "scene": {"id": "s2"},
        "end_of_previous_scene": "x" * 800,
        "manifest": ["m" * 40],
        "codex": [{"id": "a"}],
    }
    whole = scene_section("s2", "T", bundle, cap=100_000)
    assert whole is not None
    section = scene_section("s2", "T", bundle, cap=estimate_text_tokens(whole) - 50)

    assert section is not None
    cut = _bundle_of(section)
    assert "end_of_previous_scene" not in cut
    assert cut["manifest"] == ["m" * 40]
    assert "Left out to save room: end_of_previous_scene." in section


def test_a_section_that_cannot_fit_even_cut_down_is_none() -> None:
    """Killed by: src/uclone_x/story/scene_turn.py :: return section if estimate_text_tokens(section) <= cap else None
    Becomes: return section
    """
    bundle = {"request_conflicts": ["r" * 4000], "scene": {"id": "s"}, "codex": [{"id": "a"}]}
    assert scene_section("s", "T", bundle, cap=200) is None


# --- the story-facts line ------------------------------------------------------------


def test_a_saved_conflicting_scene_gets_the_kept_line_in_the_persons_language(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/story/scene_turn.py :: if not self.conflicts or not any(self._saved(record) for record in records):
    Becomes: if not self.conflicts:
    """
    aid = _aid(tmp_path)

    (korean,) = aid.reply_lines([_write_record("ch02.s02")], korean=True)
    (english,) = aid.reply_lines([_write_record("ch02.s02")], korean=False)

    assert "예린은 「스승의 최후」 장면에서 죽었습니다" in korean
    assert "라온은 「빼앗긴 검」 장면에서 월광검을 잃었습니다" in korean
    assert "승인" in korean
    # The line states the story's facts; it does not claim the scene kept them (#1808).
    assert korean.startswith("요청이 이야기 설정과 어긋납니다: ")
    assert "두었" not in korean and "kept" not in english
    assert "예린 died in the scene 「스승의 최후」" in english
    assert "ch01" not in korean + english
    assert aid.reply_lines([], korean=True) == []
    assert aid.reply_lines([_write_record("ch02.s03")], korean=True) == []
    assert aid.reply_lines([_write_record("ch02.s02", ToolResultStatus.ERROR)], korean=True) == []


def test_a_request_the_story_does_not_contradict_gets_no_kept_line(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/story/scene_turn.py :: if not self.conflicts or not any(
    Becomes: if not any(
    """
    aid = _aid(tmp_path, message="다음 장면을 써줘")

    assert aid is not None
    assert aid.reply_lines([_write_record("ch02.s02")], korean=True) == []


# --- reply lines ------------------------------------------------------------------------


def test_reply_notes_follow_the_persons_language_and_skip_failed_calls() -> None:
    """Killed by: src/uclone_x/agent/reply_lines.py :: note = cast(Mapping[str, object], note).get("ko" if korean else "en")
    Becomes: note = cast(Mapping[str, object], note).get("en")
    """
    note: dict[str, JsonValue] = {REPLY_NOTE_KEY: {"en": "Seed 7.", "ko": "시드 7."}}
    records = [
        ToolExecutionRecord(tool_name="muse_spark", output=note),
        ToolExecutionRecord(
            tool_name="muse_spark",
            output={REPLY_NOTE_KEY: "Failed."},
            status=ToolResultStatus.ERROR,
        ),
        ToolExecutionRecord(tool_name="other", output={REPLY_NOTE_KEY: "Plain."}),
    ]

    assert is_korean(CONFLICTING) and not is_korean("Write the next scene.")
    assert reply_notes(records, korean=True) == ["시드 7.", "Plain."]
    assert reply_notes(records, korean=False) == ["Seed 7.", "Plain."]


def test_a_line_is_added_once_and_not_when_the_reply_already_says_it() -> None:
    """Killed by: src/uclone_x/agent/reply_lines.py :: if text and text not in added and text not in reply:
    Becomes: if text:
    """
    assert lines_to_add("Done. Seed 7.", ["Seed 7.", "Kept.", " Kept. ", ""]) == ["Kept."]


def test_the_reply_stays_the_prefix_of_the_reply_with_lines() -> None:
    """What streamed of the reply stays true: the lines stream as what follows it.

    Killed by: src/uclone_x/agent/reply_lines.py :: max(0, 2 - (len(reply)
    Becomes: max(0, 0 - (len(reply)
    """
    assert with_lines("Scene.\n", ["A.", "B."]) == "Scene.\n\nA.\nB."
    assert with_lines("Scene.  ", ["A."]) == "Scene.  \n\nA."
    assert with_lines("", ["A."]) == "A."
    assert with_lines("Scene.", []) == "Scene."


def test_a_muse_draw_asks_the_reply_to_name_its_seed(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/story/muse.py :: REPLY_NOTE_KEY: {
    Becomes: "unused_note": {
    """
    context = ToolContext(agent_id="writer", session_id="s1", workspace_root=tmp_path)
    result = asyncio.run(
        MuseSparkTool().execute({"action": "draw", "genre": "fantasy", "seed": 81123}, context)
    )
    assert result.success, result.error

    record = ToolExecutionRecord(tool_name="muse_spark", output=result.output)
    (korean,) = reply_notes([record], korean=True)
    (english,) = reply_notes([record], korean=False)
    assert "시드 81123" in korean
    assert "seed 81123" in english


# --- the executor -----------------------------------------------------------------------


class _Scripted(BaseLLMConnector):
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return self._responses.pop(0)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        resp = await self.generate(request)
        yield StreamChunk(
            delta_content=resp.content,
            finish_reason=resp.finish_reason,
            tool_calls=resp.tool_calls,
        )


def _reply(content: str | None, *calls: ToolCallRequest) -> ModelResponse:
    return ModelResponse(
        content=content,
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        tool_calls=calls,
        usage=_USAGE,
        provenance=_PROV,
    )


class _NoParams(BaseModel):
    pass


class _NotingTool(BaseTool[_NoParams]):
    name = "noting"
    description = "Returns a reply note."

    def run(self, params: _NoParams, context: ToolContext) -> dict[str, Any]:
        del params, context
        return {REPLY_NOTE_KEY: {"en": "Noted.", "ko": "적었습니다."}}


class _Aid:
    section = "[Aid]\nThe aid's section."

    def reply_lines(self, records: Sequence[ToolExecutionRecord], *, korean: bool) -> list[str]:
        return ["Kept."] if records and not korean else []


class _AidHook:
    def __init__(self) -> None:
        self.calls = 0

    def after_tool_step(
        self, records: Sequence[ToolExecutionRecord], context: ToolContext
    ) -> ToolContext:
        del records
        return context

    def turn_aid(self, **kwargs: Any) -> _Aid:
        del kwargs
        self.calls += 1
        return _Aid()


def _agent(tmp_path: Path, llm: BaseLLMConnector, *hooks: Any) -> BaseAgent:
    tools = ToolRegistry()
    tools.register(_NotingTool())
    return BaseAgent(
        config=AgentConfig(
            agent_id="writer", name="Writer", llm_config=AgentLLMConfig(model_name="m")
        ),
        llm=llm,
        tools=tools,
        context=AgentContext(agent_id="writer", session_id="s", workspace_root=tmp_path),
        lifecycle_hooks=hooks,
    )


@pytest.mark.asyncio
async def test_the_aid_is_worked_out_once_and_its_section_sent_with_every_request(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: sections.extend(aid.section for aid in self._turn_aids)
    Becomes: sections.extend(())
    """
    hook = _AidHook()
    llm = _Scripted(
        [_reply(None, ToolCallRequest(id="c1", name="noting", arguments={})), _reply("Scene.")]
    )
    agent = _agent(tmp_path, llm, hook)

    await agent.execute_turn("Write the next scene.")

    assert hook.calls == 1
    assert len(llm.requests) == 2
    for request in llm.requests:
        assert any("[Aid]" in (m.content or "") for m in request.messages)


@pytest.mark.asyncio
async def test_reply_lines_are_appended_streamed_and_stored_alike(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: await emit("token", {"content": full[len(content) :]})
    Becomes: pass
    """
    llm = _Scripted(
        [_reply(None, ToolCallRequest(id="c1", name="noting", arguments={})), _reply("Scene.")]
    )
    agent = _agent(tmp_path, llm, _AidHook())
    streamed: list[str] = []

    async def on_event(kind: str, data: dict[str, Any]) -> None:
        if kind == "token":
            streamed.append(data["content"])

    result = await agent.execute_turn("Write the next scene.", stream_callback=on_event)

    assert result.content == "Scene.\n\nNoted.\nKept."
    assert "".join(streamed) == result.content
    final = [m for m in agent.history if m.role is MessageRole.ASSISTANT][-1]
    assert final.content == result.content
    events = [e for e in agent.pending_durable_events if e.get("type") == "REPLY_LINES_ADDED"]
    assert [e["lines"] for e in events] == [["Noted.", "Kept."]]


@pytest.mark.asyncio
async def test_a_korean_message_gets_its_lines_in_korean(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: korean = is_korean(message)
    Becomes: korean = False
    """
    llm = _Scripted(
        [_reply(None, ToolCallRequest(id="c1", name="noting", arguments={})), _reply("장면.")]
    )
    agent = _agent(tmp_path, llm)

    result = await agent.execute_turn("다음 장면을 써줘")

    assert result.content == "장면.\n\n적었습니다."


@pytest.mark.asyncio
async def test_an_empty_reply_is_asked_again_once(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: if not asked and not content and not called_tools:
    Becomes: if False:
    """
    llm = _Scripted([_reply(""), _reply("Here it is.")])
    agent = _agent(tmp_path, llm)

    result = await agent.execute_turn("Write the next scene.")

    assert result.content == "Here it is."
    assert len(llm.requests) == 2
    assert any("Your last reply was empty" in (m.content or "") for m in llm.requests[1].messages)
    nudges = [e for e in agent.pending_durable_events if e.get("type") == "EMPTY_REPLY_NUDGE"]
    assert len(nudges) == 1


@pytest.mark.asyncio
async def test_a_second_empty_reply_stops_the_turn(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: if not asked and not content and not called_tools:
    Becomes: if not content and not called_tools:
    """
    llm = _Scripted([_reply(""), _reply(""), _reply("Never asked.")])
    agent = _agent(tmp_path, llm)

    result = await agent.execute_turn("Write the next scene.")

    assert len(llm.requests) == 2
    assert result.stop_reason == "model_stopped_after_nudge"


def test_compose_empty_reply_nudge_names_tools_or_instructs_text() -> None:
    """The empty reply nudge names available tools to prevent reasoning traps (#2168).

    Killed by: src/uclone_x/agent/nudges.py :: f"If you call a tool, choose only from the available tools: {tools_str}. "
    Becomes: ""
    """
    from uclone_x.agent.nudges import compose_empty_reply_nudge

    with_tools = compose_empty_reply_nudge(["generate_image", "story_codex"])
    assert "Your last reply was empty" in with_tools
    assert "'generate_image'" in with_tools
    assert "'story_codex'" in with_tools
    assert "choose only from the available tools" in with_tools

    without_tools = compose_empty_reply_nudge([])
    assert "Your last reply was empty" in without_tools
    assert "Do not attempt tool calls" in without_tools


# --- composition ------------------------------------------------------------------------


def test_composing_the_app_hooks_adds_the_scene_hook_once(tmp_path: Path) -> None:
    """Both heads compose their hooks through `with_app_lifecycle_hooks`, 1:1 and room alike.

    Killed by: src/uclone_x/agent/clone_builder.py :: added.append(hook)
    Becomes: pass
    """
    host = HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(default_response="Done."),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(tmp_path / "sessions"),
    )
    twice = with_app_lifecycle_hooks(with_app_lifecycle_hooks(host))

    assert sum(isinstance(h, SceneTurnHook) for h in twice.lifecycle_hooks) == 1
