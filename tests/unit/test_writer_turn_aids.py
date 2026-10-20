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
from uclone_x.agent.reply_lines import (
    NOTE_ASKS_KEY,
    NOTE_LISTS_KEY,
    is_korean,
    lines_to_add,
    reply_notes,
    with_lines,
)
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
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import REPLY_NOTE_KEY, ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry
from uclone_x_story.library import StoryLibrary
from uclone_x_story.muse import MuseSparkTool
from uclone_x_story.scene_turn import SceneTurnHook, scene_section
from uclone_x_story.work import StoryWork

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
    """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: request=message or None,
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
    """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: named = _named_scene(outline, message)
    Becomes: named = None
    """
    aid = _aid(tmp_path, message)

    assert aid.scene_id == scene_id
    assert f"The scene this message names is {scene_id}" in aid.section
    assert _bundle_of(aid.section)["scene"]["id"] == scene_id


def test_an_id_inside_a_longer_id_is_not_a_name(tmp_path: Path) -> None:
    """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: {re.escape(scene.id)}(?![A-Za-z0-9_])
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
    """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: or not offered:
    Becomes: or False:
    """
    assert _aid(tmp_path, **overrides) is None


def test_a_story_that_cannot_be_read_adds_nothing_and_does_not_raise(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: except Exception:
    Becomes: except KeyError:
    """
    assert _aid(tmp_path, story_id="no-such-story") is None
    assert "could not be read" in caplog.text


def test_a_story_with_every_scene_written_adds_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Said in the log as what it is, not as a story that could not be read.

    Killed by: examples/story/src/uclone_x_story/scene_turn.py :: if following is None:
    Becomes: if False:
    """
    caplog.set_level(logging.INFO, logger="uclone_x_story.scene_turn")
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
    # No scene is added; the story's ids are still kept out of the reply (#2300).
    assert aid is not None and aid.section == ""
    assert "no unwritten scene left" in caplog.text
    assert "could not be read" not in caplog.text


def test_a_section_over_the_cap_drops_the_neighbours_text_first_and_says_so() -> None:
    """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: _DROP_FIRST = ("end_of_previous_scene", "previous_scene", "next_scene", "manifest")
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
    """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: return section if estimate_text_tokens(section) <= cap else None
    Becomes: return section
    """
    bundle = {"request_conflicts": ["r" * 4000], "scene": {"id": "s"}, "codex": [{"id": "a"}]}
    assert scene_section("s", "T", bundle, cap=200) is None


# --- the story-facts line ------------------------------------------------------------


def test_a_saved_conflicting_scene_gets_the_kept_line_in_the_persons_language(
    tmp_path: Path,
) -> None:
    """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: if not self.conflicts or not any(self._saved(record) for record in records):
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
    """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: if not self.conflicts or not any(
    Becomes: if not any(
    """
    aid = _aid(tmp_path, message="다음 장면을 써줘")

    assert aid is not None
    assert aid.reply_lines([_write_record("ch02.s02")], korean=True) == []


# --- reply lines ------------------------------------------------------------------------


def test_reply_notes_follow_the_persons_language_and_skip_failed_calls() -> None:
    """Killed by: src/uclone_x/agent/reply_lines.py :: note = found.get("ko" if korean else "en")
    Becomes: note = found.get("en")
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
    """Killed by: src/uclone_x/agent/reply_lines.py :: if text and text not in added and _plain(text) not in plain_reply:
    Becomes: if text:
    """
    assert lines_to_add("Done. Seed 7.", ["Seed 7.", "Kept.", " Kept. ", ""]) == ["Kept."]


def test_a_note_whose_list_the_reply_already_names_gives_only_what_follows_it() -> None:
    """#2301, picky T2: the Writer listed the three premises in its own words (renumbered,
    bolded, another dash), and the note listed them again under it.

    Killed by: src/uclone_x/agent/reply_lines.py :: if not all(name and name in plain_reply for name in names):
    Becomes: if True:
    """
    note: dict[str, JsonValue] = {
        "ko": "판타지 이야기의 핵심 발상 세 가지입니다.\n1. 용기의 약속 — 산맥의 수도원.\n"
        "2. 가시 외벽 — 강의 페리맨.\n마음에 드는 발상의 번호를 골라 주세요.",
        "en": "Two premises:\n1. Oath — a monastery.\n2. Wall — a ferryman.\nChoose one.",
        NOTE_LISTS_KEY: ["용기의 약속", "가시 외벽"],
    }
    record = ToolExecutionRecord(tool_name="story_start", output={REPLY_NOTE_KEY: note})
    own = "추천드려요.\n\n1) **용기의 약속**: 수도원 이야기  \n2) _가시 외벽_: 페리맨 이야기"

    assert reply_notes([record], korean=True, reply=own) == [
        "마음에 드는 발상의 번호를 골라 주세요."
    ]
    # A reply that leaves one of them out gets the whole note, as before.
    (whole,) = reply_notes([record], korean=True, reply="1) **용기의 약속**: 수도원 이야기")
    assert whole == note["ko"]


def test_a_marked_question_the_reply_already_asks_in_its_words_is_not_added_again() -> None:
    """#2305: the model asked the stop's question with straight quotes and bold, and the
    note's own copy, with curly quotes, was added under it.

    Killed by: src/uclone_x/agent/reply_lines.py :: if not (line.strip() in marked and _words(line) and _asked(line, held))
    Becomes: if True
    """
    question = "마음에 드는 발상의 번호를 골라 주세요. ‘네’라고만 하셔도 1번으로 진행합니다."
    note: dict[str, JsonValue] = {
        "ko": f"발상을 새로 만들었습니다.\n{question}",
        "en": "New premises.\nChoose one.",
        NOTE_ASKS_KEY: [question, "Choose one."],
    }
    record = ToolExecutionRecord(tool_name="story_start", output={REPLY_NOTE_KEY: note})
    reply = "**마음에 드는 발상의 번호를 골라 주세요.** '네'라고만 하셔도 1번으로 진행합니다!"

    assert reply_notes([record], korean=True, reply=reply) == ["발상을 새로 만들었습니다."]
    # A line the note does not mark is compared as before, and is added.
    unmarked = {**note, NOTE_ASKS_KEY: []}
    plain = ToolExecutionRecord(tool_name="story_start", output={REPLY_NOTE_KEY: unmarked})
    assert reply_notes([plain], korean=True, reply=reply) == [note["ko"]]


def test_a_marked_question_is_not_found_across_two_of_the_replys_sentences() -> None:
    """#2317 item 5: the reply's letters were compared joined, so a short marked line could
    match across a sentence boundary. Each sentence of the marked line must be in one of the
    reply's sentences.

    Killed by: src/uclone_x/agent/reply_lines.py :: return all(any(own in said for said in held) for own in wanted)
    Becomes: return "".join(wanted) in "".join(held)
    """
    question = "Shall I go on?"
    note: dict[str, JsonValue] = {
        "ko": "새 장을 썼습니다.",
        "en": f"Chapter written.\n{question}",
        NOTE_ASKS_KEY: ["새 장을 썼습니다.", question],
    }
    record = ToolExecutionRecord(tool_name="story_start", output={REPLY_NOTE_KEY: note})

    split = "That is what I shall. I go on to the next chapter now."
    assert reply_notes([record], korean=False, reply=split) == [note["en"]]
    asked = "Here it is. **Shall I go on?**"
    assert reply_notes([record], korean=False, reply=asked) == ["Chapter written."]


def test_a_line_the_reply_copied_with_markdown_and_hard_breaks_is_not_added_again() -> None:
    r"""#2263: the Writer copied a stop's proposal into its reply with bold marks and
    markdown hard breaks, and the exact-substring check added the same lines again.

    Killed by: src/uclone_x/agent/reply_lines.py :: lines = (" ".join(line.split()) for line in _WRAPPED.sub(r"\2", text).splitlines())
    Becomes: lines = (line for line in text.splitlines())
    """
    note = "No genre was named, so here are some:\n1. Horror (recommended)\n2. Fantasy"
    copied = "No genre was named, so here are some:  \n1. **Horror** (recommended)  \n2. _Fantasy_"

    assert lines_to_add(copied, [note]) == []
    # Different words are still added, marks or not.
    assert lines_to_add(copied, ["1. **Mystery**"]) == ["1. **Mystery**"]


def test_a_mark_inside_a_token_is_not_markdown() -> None:
    r"""#2272: `_` and `~` inside a path were stripped, so `my_story.md` read as `mystory.md`.

    Killed by: src/uclone_x/agent/reply_lines.py :: r"(?<![A-Za-z0-9*_`~])([*_`~]+)(?=[^\s*_`~])(.+?)(?<=[^\s*_`~])\1(?![A-Za-z0-9*_`~])"
    Becomes: r"()[*_`~]+()"
    """
    assert lines_to_add("Saved to mystory.md.", ["Saved to my_story.md."]) == [
        "Saved to my_story.md."
    ]
    assert lines_to_add("Open /notes.", ["Open ~/notes."]) == ["Open ~/notes."]
    assert lines_to_add("Saved to `my_story.md`.", ["Saved to my_story.md."]) == []


def test_bold_before_a_korean_particle_is_still_markdown() -> None:
    r"""#2279 review: `\w` in the token edge matched Hangul, so `**미스터리**로` kept its
    marks and a copied note was added again (#2263 regressed for Korean replies).

    Killed by: src/uclone_x/agent/reply_lines.py :: (?<=[^\s*_`~])\1(?![A-Za-z0-9*_`~])
    Becomes: (?<=[^\s*_`~])\1(?![\w*_`~])
    """
    assert (
        lines_to_add(
            "‘네’라고 하시면 **미스터리**로 발상을 만들겠습니다.",
            ["‘네’라고 하시면 미스터리로 발상을 만들겠습니다."],
        )
        == []
    )
    assert lines_to_add("추천은 **1번**입니다.", ["추천은 1번입니다."]) == []
    assert lines_to_add("**하린**은 등대지기입니다.", ["하린은 등대지기입니다."]) == []
    # Nor a mark right after Hangul (the opening edge).
    assert lines_to_add("이번엔**SF**로 갑니다.", ["이번엔SF로 갑니다."]) == []
    # The path cases still hold with ASCII edges.
    assert lines_to_add("Saved to mystory.md.", ["Saved to my_story.md."]) == [
        "Saved to my_story.md."
    ]
    assert lines_to_add("Open /notes.", ["Open ~/notes."]) == ["Open ~/notes."]


def test_two_lines_never_read_as_one() -> None:
    r"""#2272: whitespace collapse joined lines across a newline.

    Killed by: src/uclone_x/agent/reply_lines.py :: return "\n".join(line for line in lines if line)
    Becomes: return " ".join(line for line in lines if line)
    """
    assert lines_to_add("Pick one\nnow.", ["Pick one now."]) == ["Pick one now."]
    assert lines_to_add("Pick one now.", ["Pick one\nnow."]) == ["Pick one\nnow."]
    # Blank lines and a hard break between the same lines are spacing, not words.
    assert lines_to_add("A.  \n\n\nB.", ["A.\nB."]) == []


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
    """Killed by: examples/story/src/uclone_x_story/muse.py :: REPLY_NOTE_KEY: {
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
    # Ended with the plain sentence, not an empty reply (#2264).
    assert result.stop_reason == "empty_reply"
    assert result.error_code == "step.empty_reply"


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


def test_the_storys_ids_are_put_as_their_names_in_the_reply(tmp_path: Path) -> None:
    r"""#2300: a live reply said "현재 진행 중인 장면은 **ch01.s02**입니다".

    Killed by: examples/story/src/uclone_x_story/scene_turn.py :: rf"(?<![A-Za-z0-9_.-]){re.escape(found)}(?![A-Za-z0-9_]|\.[A-Za-z0-9])",
    Becomes: rf"(?<![\w.-]){re.escape(found)}(?![\w]|\.[A-Za-z0-9])",
    """
    aid = _aid(tmp_path)
    assert aid is not None

    said = aid.reply_text(
        "현재 진행 중인 장면은 **ch01.s02**입니다. ch01.s02입니다, ch01장과 moon-seal을 봅니다."
    )

    assert (
        said
        == "현재 진행 중인 장면은 **습격**입니다. 습격입니다, 은월성의 밤장과 달의 봉인을 봅니다."
    )
    # A token that only contains an id is not it.
    assert aid.reply_text("xch01.s02 ch01.s022") == "xch01.s02 ch01.s022"


class _RaisingAid(_Aid):
    def reply_text(self, reply: str) -> str:
        raise RuntimeError(f"a domain bug on {reply!r}")

    def reply_lines(self, records: Sequence[ToolExecutionRecord], *, korean: bool) -> list[str]:
        raise RuntimeError("a domain bug")


class _RaisingHook(_AidHook):
    def turn_aid(self, **kwargs: Any) -> _RaisingAid:
        del kwargs
        self.calls += 1
        return _RaisingAid()


@pytest.mark.asyncio
async def test_an_aid_that_raises_costs_its_own_part_and_not_the_reply(tmp_path: Path) -> None:
    """#2303 item 1: a domain's bug in `reply_text` or `reply_lines` failed the turn's finish.

    Killed by: src/uclone_x/agent/turn_executor.py :: rewritten = self._of_aid(lambda a=aid, c=content: a.reply_text(c), "rewrite the reply")
    Becomes: rewritten = aid.reply_text(content)
    Killed by: src/uclone_x/agent/turn_executor.py :: lambda a=aid: a.reply_lines(records, korean=korean), "give its reply lines"
    Becomes: aid.reply_lines(records, korean=korean), "give its reply lines"
    """
    llm = _Scripted(
        [_reply(None, ToolCallRequest(id="c1", name="noting", arguments={})), _reply("Scene.")]
    )
    agent = _agent(tmp_path, llm, _RaisingHook(), _AidHook())

    result = await agent.execute_turn("Write the next scene.")

    assert result.content == "Scene.\n\nNoted.\nKept."
    final = [m for m in agent.history if m.role is MessageRole.ASSISTANT][-1]
    assert final.content == result.content


class _RenamingAid(_Aid):
    def reply_text(self, reply: str) -> str:
        return reply.replace("ch01.s02", "습격")


class _RenamingHook(_AidHook):
    def turn_aid(self, **kwargs: Any) -> _RenamingAid:
        del kwargs
        self.calls += 1
        return _RenamingAid()


@pytest.mark.asyncio
async def test_an_aid_that_rewrites_the_reply_is_what_the_turn_keeps(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: sanitized = self._aid_reply_text(apply_artifact_sanitization(shown, self.workspace_root))
    Becomes: sanitized = apply_artifact_sanitization(shown, self.workspace_root)
    """
    llm = _Scripted([_reply("Next is ch01.s02.")])
    agent = _agent(tmp_path, llm, _RenamingHook())

    result = await agent.execute_turn("Where are we?")

    assert result.content == "Next is 습격."
    final = [m for m in agent.history if m.role is MessageRole.ASSISTANT][-1]
    assert final.content == result.content
