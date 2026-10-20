"""`story_start`: a request made into a story through fixed stages, stopping for the person.

What these pin, in order of what it would cost to get wrong:

* **The order is code, and the flow stops for the person** after the premises, the cast
  and the outline -- unless the request asks for the story at once.
* **The person's choice is what the later stages build on**, and nothing is chosen for them.
* **The state lives in the story's files**: a new tool instance continues where the last
  call stopped.
* **The brief fills only what the request left open.**
* **Each premise comes from its own call, engine and cards**, is checked for the
  must-haves, and the three are measured against each other by one measure.
* **The cast waits for approval, and so does the chapter**: it is written over pending
  proposals only when the person says so or asked for the story at once, and then the
  reply says so. Each character starts with the status the premise gives them.
* The outline has real beats; the chapter is written through the manuscript path.

The model is a scripted fake behind the `StoryModel` seam, and the embedder a fake behind
`PremiseEmbedder` (P5).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from pydantic import JsonValue

import uclone_x_story.start as start_module
from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.agent.reply_lines import NOTE_ASKS_KEY, NOTE_LISTS_KEY, reply_notes
from uclone_x.i18n.scripts import words_outside
from uclone_x.knowledge.extraction import parse_extraction
from uclone_x.tools.models import (
    REPLY_NOTE_KEY,
    NoIsolation,
    ToolContext,
    ToolResult,
)
from uclone_x_story.arc import CHECK_MARKER, Arc, Spine, arc_for, chapter_problems, chapter_text
from uclone_x_story.enrich import ENRICH_MARKER
from uclone_x_story.library import StoryError, StoryLibrary
from uclone_x_story.plan_check import PLAN_MARKER
from uclone_x_story.proposals import apply_proposal, reject_proposal
from uclone_x_story.scene_turn import SceneTurnHook, StartTurnAid
from uclone_x_story.schemas import Proposal, StoryFileError, status_of
from uclone_x_story.skill_data import BUNDLED_DATA_ROOT, load_structure_templates_sourced
from uclone_x_story.start import (  # pyright: ignore[reportPrivateUsage]
    CHAPTER_PROMPT_MARK,
    CHAPTERS_PER_CALL,
    DEFAULTS,
    DETAILS_MARK,
    DETAILS_REVISE_MARK,
    ENGINES,
    GENRE_LABELS,
    KG_MARK,
    MAX_CAST_TRIES,
    MAX_DRAFTS,
    MAX_PREMISE_BIGRAM,
    MAX_PREMISE_COSINE,
    REPEAT_MARK,
    REVISE_MARK,
    SCENE_CHOICES_MARK,
    SCENE_CUSTOM_MARK,
    SCENE_VISUAL_MARK,
    SPINE_MARK,
    START_FILE,
    StartState,
    StoryStartTool,
    _reply_note,  # pyright: ignore[reportPrivateUsage]
    _Run,  # pyright: ignore[reportPrivateUsage]
    _with_entry_fields,  # pyright: ignore[reportPrivateUsage]
    _without_names,  # pyright: ignore[reportPrivateUsage]
    abstract_item,
    bigram_overlap,
    brief_from,
    cosine,
    genre_label,
    genre_options,
    json_object,
    names_genre,
    outline_from,
    premise_similarity,
    scene_title,
    scenes_for,
    story_scripts,
    structure_for,
    wants_it_now,
)
from uclone_x_story.tool import StoryLibraryTool
from uclone_x_story.tools import (
    NewEntryParams,
    StoryCodexTool,
    StoryManuscriptTool,
    StoryOutlineTool,
    new_entry_draft,
)
from uclone_x_story.work import StoryWork

ROOM = "room_a"
_HANGUL = re.compile(r"[가-힣]")

REQUEST = "등대지기 소녀가 나오는 호러 단편을 써 줘. 고양이가 꼭 나오고, 잔인한 장면은 빼 줘."

_PREMISES = (
    ("안개 등대", "등대지기 소녀 하린은 안개 속에서 들려오는 목소리를 쫓는다."),
    ("검은 고양이의 밤", "섬의 고양이들이 하나씩 사라지고, 남은 한 마리가 하린을 이끈다."),
    ("꺼지지 않는 불", "폭풍의 밤, 꺼진 등대에 누군가 다시 불을 켠다."),
)

_SCENE = (
    "하린은 등대 계단을 올랐다. 바람이 창을 두드렸고, 고양이 먹물이 발밑을 따라왔다. "
    "꼭대기에 닿자 렌즈 너머로 안개가 밀려들었다. 그 안에서 누군가 이름을 불렀다. "
    "하린은 등불을 들어 올렸다. 먹물이 낮게 울었다."
)


#: A second scene, and a rewrite of the first that a person asked for (#2361): neither
#: repeats the scene before it, so the prose check keeps each.
_SCENE_TWO = (
    "아침이 밝자 하린은 마을 우물가로 내려갔다. 장터의 사람들이 등대 이야기를 수군거렸다. "
    "먹물은 생선 가게 앞에서 꼬리를 흔들었다. 하린은 빵 한 덩이를 사서 품에 넣었다."
)
#: A draft the prose check refuses: Hanja glued inside a Hangul word (#2365).
_SCENE_GLUED = (
    "하린은 등대에서 눈을眯하며 웃었다. 바람이 불었다. 먹물이 울었다. 하린은 계단을 내려왔다."
)
_SCENE_CRIES = (
    "하린은 등대 계단에 주저앉아 울었다. 눈물이 손등에 떨어졌고, 먹물이 무릎에 머리를 "
    "기댔다. 한참 뒤에야 하린은 일어나 창밖의 어둠을 바라보았다. 바다는 조용했다."
)


#: A premise the fake gives after its first three, under a title of its own.
_REDRAWN = ("등대의 손님", "폭풍이 지난 아침, 낯선 선원이 등대 문을 두드린다.")

_REWRITE = (
    "폭풍이 지난 아침, 등대지기 소녀 하린은 사라진 고양이 먹물의 방울을 등대 꼭대기에서 줍는다."
)

_SPINE = {
    "central_conflict": "하린은 안개 속 목소리의 정체를 밝히려 한다.",
    "twist": "목소리는 사라진 고양이를 찾던 하린 자신의 것이었다.",
    "clue": "계단에 하린의 젖은 발자국이 먼저 나 있다.",
    "resolution": "하린은 목소리를 따라 안개 속으로 들어가 고양이를 되찾는다.",
}

_DEFAULT_CHAPTERS: dict[int, dict[str, Any]] = {
    1: {
        "title": "안개가 오는 밤",
        "stakes": "등대의 불",
        "scenes": [
            {
                "title": "계단",
                "summary": "하린이 등대에 오른다.",
                "beats": ["바람이 분다", "누군가 이름을 부른다"],
                "characters": ["하린", "먹물", "낯선 이"],
                "places": ["안개섬 등대"],
            },
            {"title": "목소리", "summary": "목소리를 쫓는다.", "beats": ["문이 열린다"]},
        ],
    },
    2: {"title": "새벽", "scenes": [{"title": "끝", "summary": "안개가 걷힌다."}]},
}


#: What the fake reads from a settled outline for the codex (`KG_MARK`).
_KG: dict[str, Any] = {
    "threads": [
        {
            "name": "등대의 약속",
            "profile": "하린이 먹물을 꼭 찾겠다고 한 약속.",
            "characters": ["하린", "먹물", "없는 사람"],
            "planted_chapter": 1,
            "pay_off_chapter": 2,
        }
    ],
    "items": [{"name": "등불", "profile": "하린이 드는 오래된 등불."}],
}

#: The fake's revision of the outline (`REVISE_MARK`): the first chapter renamed.
_REVISED: dict[str, Any] = {
    "central_conflict": _SPINE["central_conflict"],
    "twist": _SPINE["twist"],
    "chapters": [{**_DEFAULT_CHAPTERS[1], "title": "첫날 밤의 목소리"}, _DEFAULT_CHAPTERS[2]],
}


#: The fake's plan of a chapter (`DETAILS_MARK`): two concrete scenes.
_DETAILS: dict[str, Any] = {
    "scenes": [
        {
            "title": "젖은 발자국",
            "summary": "하린이 계단의 발자국을 따라간다.",
            "beats": ["발자국이 꼭대기로 이어진다", "먹물의 방울 소리가 난다"],
            "characters": ["하린", "먹물", "없는 사람"],
            "places": ["안개섬 등대"],
        },
        {"title": "걷히는 안개", "summary": "안개가 걷힌다.", "beats": ["해가 뜬다"]},
    ]
}

#: The fake's revision of a plan (`DETAILS_REVISE_MARK`): one scene, renamed.
_DETAILS_REVISED: dict[str, Any] = {
    "scenes": [
        {
            "title": "방울 소리",
            "summary": "하린이 방울 소리를 따라 먹물을 찾는다.",
            "beats": ["방울 소리가 멀어진다"],
            "characters": ["하린"],
        }
    ]
}


def _outlined_chapter(prompt: str) -> dict[str, Any]:
    """The chapter a plan prompt shows as the settled outline has it, as a plan's JSON."""
    head = "as the settled outline has it, as JSON:\n"
    shown, _ = json.JSONDecoder().raw_decode(prompt, prompt.index(head) + len(head))
    return {"scenes": shown["chapters"][0]["scenes"]}


class FakeEmbedder:
    """A fake `PremiseEmbedder`: the vectors given, in turn; or it fails."""

    def __init__(self, *vectors: list[list[float]], fails: bool = False) -> None:
        self._vectors = list(vectors)
        self._fails = fails
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self._fails:
            raise ConnectionError("http://127.0.0.1:11434 refused")
        return self._vectors.pop(0)


_APART = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


# qwen3:8b wrote this on 2026-09-29: a Chinese character glued inside a Korean word.
_GLUED = "할머니는 편지를 펼쳤다. 무언가 잘못되었다는 직觉을 느꼈다. 바다가 조용했다."


class ScriptedModel:
    """A fake `StoryModel`: answers each stage's prompt by what the prompt asks for."""

    def __init__(
        self,
        *,
        brief: dict[str, Any] | None = None,
        scenes: list[str] | None = None,
        lacking: int = 0,
        cast_replies: list[str] | None = None,
        facts: list[list[dict[str, Any]] | str] | None = None,
        chapters: dict[int, list[dict[str, Any]]] | None = None,
        spine: dict[str, Any] | None = None,
        checks: Sequence[tuple[str, ...]] = (),
        kg: dict[str, Any] | str | None = None,
        revised: dict[str, Any] | str | None = None,
        details: dict[str, Any] | str | None = None,
        details_revised: dict[str, Any] | str | None = None,
        growth: Sequence[dict[str, Any] | Exception] = (),
        plans: Sequence[dict[str, Any] | str] = (),
        later_briefs: Sequence[dict[str, Any]] = (),
        later_premises: Sequence[tuple[str, str]] = (),
        premises: Sequence[tuple[str, str]] = _PREMISES,
        scene_choices: Sequence[dict[str, Any] | str] = (),
        scene_custom: Sequence[dict[str, Any] | str] = (),
        scene_visuals: Sequence[dict[str, Any] | str] = (),
    ) -> None:
        self.prompts: list[str] = []
        # Readings of the person's later words as a request (`_direction` after the first),
        # in turn; then `brief` again.
        self._later_briefs = list(later_briefs)
        self._briefs_read = 0
        # The three premises the first calls give, in turn.
        self._first_premises = tuple(premises)
        # Premises given after the first three calls, in turn, before cycling back.
        self._later_premises = list(later_premises)
        self._scene_choices = list(scene_choices)
        self._scene_custom = list(scene_custom)
        self._scene_visuals = list(scene_visuals)
        #: How many scene readings were asked for (`ENRICH_MARKER`).
        self.readings = 0
        # What the plan check reads from each written scene, in turn; then nothing amiss.
        self._plans = list(plans)
        # What the codex-growth reading of each saved scene returns, in turn (or raises);
        # then nothing new.
        self._growth = list(growth)
        self._details = details if details is not None else _DETAILS
        # With no plan given, the first chapter's plan keeps the outline's scenes, so a
        # test of the written chapter reads the scenes it outlined.
        self._echo_first = details is None
        self._details_revised = details_revised if details_revised is not None else _DETAILS_REVISED
        self._kg = kg if kg is not None else _KG
        self._revised = revised if revised is not None else _REVISED
        # Replies to the call for chapter n before the default one; each is used once.
        self._chapters = {n: list(replies) for n, replies in (chapters or {}).items()}
        self._spine = spine if spine is not None else _SPINE
        # Outline check questions answered yes (or B) when they hold every text of one entry.
        self._checks = tuple(checks)
        # Replies to the cast prompt before the good one; each is used once.
        self._cast_replies = list(cast_replies or [])
        # The facts each written scene's reading states, in turn, beside its additions.
        self._facts = list(facts or [])
        self._scenes = list(scenes or [])
        # The scene last written, and each draft asked again for repeating the scene before
        # (`REPEAT_MARK`): this model writes the same again, as the scripted scenes repeat.
        self._last_scene = _SCENE
        self.redrafts: list[str] = []
        self._premise_calls = 0
        # The must-have check answers no this many times first, then yes.
        self._lacking = lacking
        self._brief = (
            brief
            if brief is not None
            else {
                "title": "안개 등대",
                "genre": "호러",
                "tone": None,
                "audience": None,
                "length": "단편",
                "must_have": ["고양이"],
                "avoid": ["잔인한 장면"],
            }
        )

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        if REPEAT_MARK in prompt:
            self.redrafts.append(prompt)
            return self._last_scene
        self.prompts.append(prompt)
        if SCENE_VISUAL_MARK in prompt:
            if self._scene_visuals:
                item = self._scene_visuals.pop(0)
                return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            return json.dumps(
                {
                    "characters": ["하린"],
                    "action": "holding brass lantern, climbing stone stairs",
                    "expression": "focused expression",
                    "location": "lighthouse interior, fog outside window",
                    "lighting": "warm lantern glow, night",
                    "camera": "medium shot",
                },
                ensure_ascii=False,
            )
        if SCENE_CHOICES_MARK in prompt:
            if self._scene_choices:
                item = self._scene_choices.pop(0)
                return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            return json.dumps(
                {
                    "choices": [
                        {
                            "number": 1,
                            "title": "정면으로 오르는 계단",
                            "approach": "정면 돌파 / 행동",
                            "summary": "하린이 등불을 들고 계단을 단숨에 오른다.",
                            "beats": ["등불을 높이 든다", "꼭대기 문을 밀어 연다"],
                            "characters": ["하린", "먹물"],
                            "places": ["안개섬 등대"],
                            "transitions": [
                                {
                                    "subject": "하린",
                                    "set": {"location": "안개섬 등대", "stance": "단호함"},
                                    "note": "하린이 등대 꼭대기로 진입한다.",
                                }
                            ],
                        },
                        {
                            "number": 2,
                            "title": "젖은 발자국 추적",
                            "approach": "탐색 / 단서 추적",
                            "summary": "하린이 계단에 찍힌 젖은 발자국과 방울을 살핀다.",
                            "beats": ["발자국을 조사한다", "방울 조각을 줍는다"],
                            "characters": ["하린", "먹물"],
                            "places": ["안개섬 등대"],
                            "transitions": [
                                {
                                    "subject": "하린",
                                    "set": {"possesses": "은방울"},
                                    "note": "하린이 계단에서 은방울을 발견한다.",
                                }
                            ],
                            "new_entries": [
                                {
                                    "kind": "items",
                                    "name": "은방울",
                                    "profile": "먹물의 목걸이에 달려 있던 작은 은방울.",
                                }
                            ],
                        },
                        {
                            "number": 3,
                            "title": "안개 속 이름을 부르다",
                            "approach": "대화 / 관계 변화",
                            "summary": "하린이 창밖 안개를 향해 먹물의 이름을 부르며 교감한다.",
                            "beats": ["창문을 연다", "먹물의 울음소리에 답한다"],
                            "characters": ["하린", "먹물"],
                            "places": ["안개섬 등대"],
                            "transitions": [
                                {
                                    "subject": "하린",
                                    "relations": {"먹물": "서로의 위치를 확인함"},
                                    "note": "하린과 먹물이 소리로 교감한다.",
                                }
                            ],
                        },
                    ]
                },
                ensure_ascii=False,
            )
        if SCENE_CUSTOM_MARK in prompt:
            if self._scene_custom:
                item = self._scene_custom.pop(0)
                return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            return json.dumps(
                {
                    "title": "지하 비밀 문 개방",
                    "summary": "하린이 등대 바닥의 낡은 열쇠로 비밀 지하실 문을 연다.",
                    "beats": ["바닥 틈에서 열쇠를 찾는다", "지하실 문을 연다"],
                    "characters": ["하린", "먹물"],
                    "places": ["안개섬 등대"],
                    "transitions": [
                        {
                            "subject": "하린",
                            "set": {"possesses": "황동 열쇠", "stance": "경계"},
                            "note": "하린이 황동 열쇠를 얻어 지하실을 연다.",
                        }
                    ],
                    "new_entries": [
                        {
                            "kind": "items",
                            "name": "황동 열쇠",
                            "profile": "등대 지하실 문을 여는 오래된 황동 열쇠.",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        if prompt.startswith(ENRICH_MARKER):
            # One reading per written scene (#2207): what it adds (`growth`) and the facts
            # it states (`facts`), in one reply.
            self.readings += 1
            grown: dict[str, Any] | Exception = (
                self._growth.pop(0) if self._growth else {"new": [], "facts": []}
            )
            if isinstance(grown, Exception):
                raise grown
            facts = self._facts.pop(0) if self._facts else []
            if isinstance(facts, str):
                return facts
            reply = {"new": grown.get("new", []), "facts": [*grown.get("facts", []), *facts]}
            return json.dumps(reply, ensure_ascii=False)
        if prompt.startswith(PLAN_MARKER):
            held: dict[str, Any] | str = (
                self._plans.pop(0) if self._plans else {"beats": [], "unplanned": []}
            )
            return held if isinstance(held, str) else json.dumps(held, ensure_ascii=False)
        if DETAILS_REVISE_MARK in prompt:
            plan = self._details_revised
            return plan if isinstance(plan, str) else json.dumps(plan, ensure_ascii=False)
        if DETAILS_MARK in prompt:
            if self._echo_first and f"{DETAILS_MARK} 1," in prompt:
                return json.dumps(_outlined_chapter(prompt), ensure_ascii=False)
            plan = self._details
            return plan if isinstance(plan, str) else json.dumps(plan, ensure_ascii=False)
        if KG_MARK in prompt:
            kg = self._kg
            return kg if isinstance(kg, str) else json.dumps(kg, ensure_ascii=False)
        if REVISE_MARK in prompt:
            revised = self._revised
            return revised if isinstance(revised, str) else json.dumps(revised, ensure_ascii=False)
        if "write down ONLY" in prompt:
            self._briefs_read += 1
            brief = self._brief
            if self._briefs_read > 1 and self._later_briefs:
                brief = self._later_briefs.pop(0)
            return "```json\n" + json.dumps(brief, ensure_ascii=False) + "\n```"
        if "Write one premise" in prompt:
            if "had words outside" in prompt and self._later_premises:
                # A draft asked again for its words: answered from the later premises.
                title, premise = self._later_premises.pop(0)
                return json.dumps({"title": title, "premise": premise}, ensure_ascii=False)
            title, premise = self._first_premises[self._premise_calls % 3]
            if self._premise_calls >= 3 and self._later_premises:
                title, premise = self._later_premises.pop(0)
            self._premise_calls += 1
            return json.dumps({"title": title, "premise": premise}, ensure_ascii=False)
        if "Does this story premise contain" in prompt:
            if self._lacking > 0:
                self._lacking -= 1
                return "no"
            return "yes"
        if "Rewrite the premise" in prompt:
            return json.dumps({"title": "고양이 등대", "premise": _REWRITE}, ensure_ascii=False)
        if "List the main characters" in prompt:
            if self._cast_replies:
                return self._cast_replies.pop(0)
            return json.dumps(
                {
                    "characters": [
                        {
                            "name": "하린",
                            "profile": "등대지기의 딸.",
                            "appearance": "짧은 머리",
                            "gender": "female",
                            "status": "alive",
                        },
                        {
                            "name": "먹물",
                            "profile": "사라진 검은 고양이.",
                            "gender": "unknown",
                            "status": "실종",
                        },
                    ],
                    "places": [{"name": "안개섬 등대", "profile": "섬 끝의 등대."}],
                },
                ensure_ascii=False,
            )
        if CHECK_MARKER in prompt:
            hit = any(all(part in prompt for part in c) for c in self._checks)
            if "Answer with one letter" in prompt:
                return "B" if hit else "A"
            return "yes" if hit else "no"
        if SPINE_MARK in prompt:
            return json.dumps(self._spine, ensure_ascii=False)
        if CHAPTER_PROMPT_MARK in prompt:
            number = int(prompt.split(CHAPTER_PROMPT_MARK)[1].split()[0])
            queued = self._chapters.get(number)
            if queued:
                return json.dumps(queued.pop(0), ensure_ascii=False)
            # Two chapters by default; the third call comes back with no scene.
            return json.dumps(_DEFAULT_CHAPTERS.get(number, {}), ensure_ascii=False)
        if "Outline the story" in prompt:
            return json.dumps({"chapters": list(_DEFAULT_CHAPTERS.values())}, ensure_ascii=False)
        if "Write scene" in prompt:
            self._last_scene = self._scenes.pop(0) if self._scenes else _SCENE
            return self._last_scene
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


def _last(model: ScriptedModel, text: str) -> str:
    """The last prompt the model was sent that contains `text`."""
    return [p for p in model.prompts if text in p][-1]


def _ctx(workspace: Path, *, story: str | None = None, turn: str | None = None) -> ToolContext:
    return ToolContext(
        agent_id="writer",
        session_id=f"sess_room__{ROOM}__writer",
        workspace_root=workspace,
        room_id=ROOM,
        story_id=story,
        turn_id=turn,
        isolation=NoIsolation(),
    )


def _tool(model: ScriptedModel, embedder: FakeEmbedder | None = None) -> StoryStartTool:
    return StoryStartTool(
        model_factory=lambda _context: model, embedder_factory=lambda _context: embedder
    )


async def _ok(tool: StoryStartTool, ctx: ToolContext, **args: Any) -> dict[str, Any]:
    """`story_start` called with `args`, which must succeed.

    A bare yes (`go_on` alone) at the premise stop is passed with the recommended number,
    as the Writer is told to pass it: a premise is taken only by its number (#2360). A
    test of that refusal calls `execute` itself."""
    if args.get("action") == "continue" and args.get("go_on") and "choice" not in args:
        args = {**args, **_recommended_choice(ctx, args)}
    result: ToolResult = await tool.execute(args, ctx)
    assert result.success, result.error
    assert isinstance(result.output, dict)
    return result.output


def _recommended_choice(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """`{"choice": <recommended>}` when the story waits at its premises and `args` turn
    nothing down, else nothing."""
    root = ctx.workspace_root
    if root is None or ctx.story_id is None or args.get("changes") or args.get("avoid"):
        return {}
    path = Path(root) / "stories" / ctx.story_id / START_FILE
    state = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    if not state or state.get("stage") != "choose_premise" or state.get("cast_failed"):
        return {}
    return {"choice": state.get("recommended") or 1}


def _state(workspace: Path, story_id: str) -> dict[str, Any]:
    text = (workspace / "stories" / story_id / START_FILE).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _work(workspace: Path, story_id: str) -> StoryWork:
    return StoryWork(StoryLibrary(workspace), story_id)


class TestTheFlowStopsForThePerson:
    async def test_start_stops_at_the_three_premises_and_asks_for_a_number(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if state.quick:
        Becomes: if True:
        """
        out = await _ok(_tool(ScriptedModel()), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        assert out["stage"] == "choose_premise"
        assert [p["number"] for p in out["premises"]] == [1, 2, 3]
        assert "번호를 골라 주세요" in out[REPLY_NOTE_KEY]["ko"]
        for title, _ in _PREMISES:
            assert title in out[REPLY_NOTE_KEY]["ko"]
        work = _work(tmp_path, story_id)
        assert work.outline() is None
        assert work.proposals()[0] == []
        assert work.written_scenes() == []

    async def test_continue_without_a_choice_is_refused_and_changes_nothing(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if number is None and self.state.quick:
        Becomes: if number is None:
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        before = _state(tmp_path, story_id)
        result = await _tool(model).execute(
            {"action": "continue", "request": "음, 잘 모르겠어요"}, _ctx(tmp_path, story=story_id)
        )
        assert not result.success
        assert result.error is not None and "choice" in result.error
        assert _state(tmp_path, story_id) == before
        assert _work(tmp_path, story_id).proposals()[0] == []


class TestTheChoiceCarriesThrough:
    async def test_each_stage_stops_and_builds_on_the_chosen_premise(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: return next((p for p in self.premises if p.number == number), self.premises[0])
        Becomes: return self.premises[0]
        Killed by: examples/story/src/uclone_x_story/start.py :: self._save(chosen=number)
        Becomes: pass
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)

        # A fresh tool each call: what the flow knows is in the story's files.
        cast = await _ok(_tool(model), ctx, action="continue", choice=2)
        assert cast["stage"] == "review_cast"
        assert cast["chosen"] == 2
        assert _PREMISES[1][1] in model.prompts[-1]
        work = _work(tmp_path, story_id)
        pending = [p for p, _ in work.proposals()[0]]
        assert {p.entry_id for p in pending} == {"harin", "meokmul", "angaeseom_deungdae"}
        assert all(p.status == "pending" for p in pending)
        assert not work.codex().items  # nothing is in the codex until a person approves
        assert work.outline() is None

        outline_out = await _ok(_tool(model), ctx, action="continue")
        assert outline_out["stage"] == "review_outline"
        outline, _ = work.require_outline()
        first = outline.chapters[0].scenes[0]
        assert first.beats == ["바람이 분다", "누군가 이름을 부른다"]
        assert first.characters == ["harin", "meokmul"]  # a name not in the cast is dropped
        assert first.places == ["angaeseom_deungdae"]
        assert _PREMISES[1][1] in _last(model, CHAPTER_PROMPT_MARK)
        assert work.written_scenes() == []

        # The person approves 하린 and the lighthouse and turns the cat down.
        for proposal in pending:
            if proposal.entry_id == "meokmul":
                reject_proposal(
                    work, proposal.id, room_id=ROOM, reason=None, decided_in="story_view"
                )
            else:
                apply_proposal(work, proposal.id, room_id=ROOM, decided_in="story_view")
        # The first chapter's plan is shown before it is written, as every later one's is.
        plan = await _ok(_tool(model), ctx, action="continue")
        assert plan["stage"] == "review_chapter" and plan["details_chapter"] == 1
        assert plan["done_now"] == ["plot", "details"]
        assert work.written_scenes() == []
        assert "1장의 장면 계획" in plan[REPLY_NOTE_KEY]["ko"]
        chapter = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert chapter["stage"] == "review_chapter"
        assert chapter["scenes_written"] == ["ch01.s01", "ch01.s02"]
        assert work.written_scenes() == ["ch01.s01", "ch01.s02"]
        found = work.manuscript("ch01.s01")
        assert found is not None and found.text.strip() == _SCENE
        last_scene = [p for p in model.prompts if "Write scene" in p][-1]
        cast_part = last_scene.split("Cast and places:")[1].split("What to know")[0]
        assert "하린" in cast_part and "먹물" not in cast_part  # the rejected entry is out
        assert "cast_assumed" not in chapter
        assert "승인된 것으로 보고" not in chapter[REPLY_NOTE_KEY]["ko"]

    async def test_asking_for_it_at_once_runs_every_stage_in_one_call(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: done.extend(await run.forward(choice=None, go_on=True))
        Becomes: pass
        """
        model = ScriptedModel()
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        assert out["stage"] == "review_chapter"
        assert out["done_now"] == [
            "direction",
            "premises",
            "cast",
            "outline",
            "plot",
            "details",
            "chapter",
            "continuity",
            "details",
        ]
        assert out["chosen"] == 1
        work = _work(tmp_path, out["open_story_id"])
        assert work.written_scenes() == ["ch01.s01", "ch01.s02"]
        # Written over the cast as proposed, and the person is told so in plain words;
        # nothing is approved for them.
        assert len(out["cast_assumed"]) == 3
        assert "승인된 것으로 보고 썼습니다" in out[REPLY_NOTE_KEY]["ko"]
        assert all(p.status == "pending" for p, _ in work.proposals()[0])


class TestTheChapterWaitsForTheCast:
    async def _at_outline(self, tmp_path: Path, model: ScriptedModel) -> tuple[str, ToolContext]:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_tool(model), ctx, action="continue", choice=1)
        await _ok(_tool(model), ctx, action="continue")
        return out["open_story_id"], ctx

    async def test_pending_cast_stops_the_chapter_and_asks(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if pending and not (go_on or self.state.quick):
        Becomes: if False:
        """
        model = ScriptedModel()
        story_id, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue")
        assert out["stage"] == "review_outline"
        assert out["done_now"] == ["plot"]  # the plot is settled; the chapter waits
        assert len(out["cast_waiting"]) == 3
        assert "아직 승인되지 않아 첫 장을 계획하지 않았습니다" in out[REPLY_NOTE_KEY]["ko"]
        assert "approval" in out["next"]
        assert _work(tmp_path, story_id).written_scenes() == []
        assert not any("Write scene" in p for p in model.prompts)

    async def test_saying_to_go_on_writes_it_and_says_so(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: result["cast_assumed"] = state.cast_assumed
        Becomes: pass
        """
        model = ScriptedModel()
        story_id, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert out["stage"] == "review_chapter" and out["details_chapter"] == 1
        assert "cast_waiting" not in out
        assert len(out["cast_assumed"]) == 3
        assert _work(tmp_path, story_id).written_scenes() == []  # the plan is shown first
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert "승인된 것으로 보고 썼습니다" in out[REPLY_NOTE_KEY]["ko"]
        assert _work(tmp_path, story_id).written_scenes() == ["ch01.s01", "ch01.s02"]


class TestTheCast:
    async def test_each_character_starts_with_the_status_the_premise_gives(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_cast.py :: facts: dict[str, Any] = {"status": status}
        Becomes: facts: dict[str, Any] = {"status": "alive"}
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        await _ok(
            _tool(model), _ctx(tmp_path, story=out["open_story_id"]), action="continue", choice=2
        )
        entries = {
            p.entry_id: p.new_entry()
            for p, _ in _work(tmp_path, out["open_story_id"]).proposals()[0]
        }
        assert entries["harin"].state == {"status": "alive", "gender": "female"}
        assert entries["meokmul"].state == {"status": "missing"}
        assert entries["angaeseom_deungdae"].state == {}
        assert '"missing", "dead" or "unknown"' in model.prompts[-1]


class TestTheChapter:
    async def test_a_refused_draft_is_rewritten_with_the_reason(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: for _ in range(MAX_DRAFTS):
        Becomes: for _ in range(1):
        """
        model = ScriptedModel(scenes=[_GLUED])
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        assert out["stage"] == "review_chapter"
        assert out["scenes_written"] == ["ch01.s01", "ch01.s02"]
        scene_prompts = [p for p in model.prompts if "Write scene" in p]
        assert "was not saved" not in scene_prompts[0]
        assert "was not saved" in scene_prompts[1]
        assert '"직觉을"' in scene_prompts[1]  # the check's own words, quoting the fault

    async def test_a_chapter_with_nothing_saved_stays_open_to_try_again(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: if not this_chapter:
        Becomes: if False:
        """
        model = ScriptedModel(scenes=[_GLUED] * 3)
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        assert out["stage"] == "review_chapter" and out["details_chapter"] == 1
        assert "scenes_written" not in out
        assert not _work(tmp_path, out["open_story_id"]).written_scenes()
        assert any("after 3 drafts" in n for n in out["notes"])
        assert "저장하지 못했습니다" in out[REPLY_NOTE_KEY]["ko"]
        again = await _ok(
            _tool(ScriptedModel()), _ctx(tmp_path, story=out["open_story_id"]), action="continue"
        )
        assert again["stage"] == "review_chapter"
        assert again["scenes_written"] == ["ch01.s01", "ch01.s02"]


# qwen3:8b on 2026-09-29, cut to its fault: a double quote inside a value ends the string.
_BROKEN_CAST = '{"characters": [{"name": "리안", "profile": "전설의 "용의 눈"을 찾는 소년"}]}'

_DEAD_LINE = "하린은 이미 죽은 몸이었다"
_DEAD_SCENE = f"{_SCENE} {_DEAD_LINE}. 그래도 등불은 꺼지지 않았다."
_HARIN_DEAD = {
    "subject": "harin",
    "key": "status",
    "value": "dead",
    "quote": _DEAD_LINE,
    "time": "now",
    "polarity": "asserted",
}
#: 하린 on the stairs in the next scene, after the scene before left her dead.
_BACK_LINE = "하린은 등대 계단을 올랐다"
_HARIN_BACK = {**_HARIN_DEAD, "value": "alive", "quote": _BACK_LINE}


class TestTheCastIsNeverSkipped:
    async def test_an_unreadable_cast_is_asked_again_told_why(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_cast.py :: for _ in range(MAX_CAST_TRIES):
        Becomes: for _ in range(1):
        """
        model = ScriptedModel(cast_replies=[_BROKEN_CAST])
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        cast = await _ok(
            _tool(model), _ctx(tmp_path, story=out["open_story_id"]), action="continue", choice=1
        )
        assert cast["stage"] == "review_cast"
        assert "cast_failed" not in cast
        asked = [p for p in model.prompts if "List the main characters" in p]
        assert len(asked) == 2
        assert "could not be used" not in asked[0]
        assert "could not be used" in asked[1] and "‘ ’" in asked[1]
        assert "The cast was asked for 2 times." in cast["notes"]

    async def test_a_cast_that_never_comes_stops_the_story_even_asked_at_once(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if self.state.cast_failed:
        Becomes: if False:
        """
        model = ScriptedModel(cast_replies=[_BROKEN_CAST] * MAX_CAST_TRIES)
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        assert out["stage"] == "choose_premise"
        assert out["cast_failed"] == "unreadable"
        assert "cast" not in out["done_now"]
        story_id = out["open_story_id"]
        work = _work(tmp_path, story_id)
        assert work.outline() is None and work.written_scenes() == []
        assert not any(
            m in p for p in model.prompts for m in ("Write scene", SPINE_MARK, CHAPTER_PROMPT_MARK)
        )
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "인물을 뽑지 못했습니다" in ko and "형식이 깨져" in ko
        assert "인물 없이 개요나 첫 장을 만들지는 않았습니다" in ko
        assert not any(w in ko for w in ("JSON", "unreadable", "{", "Traceback"))

    async def test_asked_again_with_no_number_it_tries_the_premise_chosen(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: retry = self.state.chosen if self.state.cast_failed else None
        Becomes: retry = None
        """
        model = ScriptedModel(cast_replies=["{}"] * MAX_CAST_TRIES)
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        failed = await _ok(_tool(model), ctx, action="continue", choice=2)
        assert failed["stage"] == "choose_premise"
        assert failed["cast_failed"] == "empty"
        assert "이름이 있는 인물이 한 명도 나오지 않았습니다" in failed[REPLY_NOTE_KEY]["ko"]
        again = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert again["stage"] == "review_cast"
        assert again["chosen"] == 2
        assert "cast_failed" not in again


class TestTheOutlineUsesAStructure:
    def _known(self) -> dict[str, Any]:
        found = load_structure_templates_sourced([BUNDLED_DATA_ROOT])
        return {k: v.item for k, v in found.items()}

    def test_the_named_structure_wins_and_the_genre_fills_only_what_is_left(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: return structure_id, "genre"
        Becomes: return DEFAULT_STRUCTURE, "genre"
        """
        known = self._known()
        assert structure_for("기승전결로 써 줘", "호러", known) == ("kishotenketsu", "request")
        assert structure_for("영웅의 여정 구조로", "일상 드라마", known) == (
            "heros-journey",
            "request",
        )
        assert structure_for("save the cat please", "", known) == ("save-the-cat", "request")
        assert structure_for("", "미스터리", known) == ("three-act", "genre")
        assert structure_for("", "판타지 모험", known) == ("heros-journey", "genre")
        assert structure_for("", "일상 드라마", known) == ("kishotenketsu", "genre")
        assert structure_for("", "서사시", known) == ("three-act", "default")
        assert structure_for("기승전결", "호러", {}) is None

    async def test_the_outline_is_the_structure_the_person_named_act_by_act(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_outline.py :: acts = [act.id for act in kept_acts]
        Becomes: acts = []
        """
        model = ScriptedModel()
        # The brief says horror, which alone would give the three-act structure.
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 기승전결로 짜 줘."
        )
        assert out["structure"] == {"id": "kishotenketsu", "source": "request"}
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_tool(model), ctx, action="continue", choice=1)
        outline_out = await _ok(_tool(model), ctx, action="continue")
        spine = _last(model, SPINE_MARK)
        assert "'Kishotenketsu', 4 chapters" in spine and "learns it in chapter 3" in spine
        asked = [p for p in model.prompts if CHAPTER_PROMPT_MARK in p]
        # One call per act, in order, until one came back with no scene.
        assert [f"{CHAPTER_PROMPT_MARK} {n} of 4" in p for n, p in enumerate(asked, 1)] == [
            True
        ] * 3
        assert "(act 'Twist')" in asked[2]
        outline, _ = _work(tmp_path, out["open_story_id"]).require_outline()
        assert [c.act for c in outline.chapters] == ["ki", "sho"]
        assert outline_out["central_conflict"].startswith("하린은 안개 속")
        assert outline_out["twist"].startswith("목소리는")
        assert "The outline covers 2 of the structure's 4 acts." in outline_out["notes"]

    async def test_the_twist_is_told_only_to_the_chapter_that_reveals_it(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/arc.py :: if number < arc.twist_chapter:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/arc.py :: if number >= arc.climax_chapter and self.resolution:
        Becomes: if self.resolution:
        Killed by: examples/story/src/uclone_x_story/start_outline.py :: f"Chapter {n}:\n{chapter_text(c)}" for n, c in enumerate(chapters, start=1)
        Becomes: "" for n, c in enumerate(chapters, start=1)
        """
        third = {"title": "거울", "scenes": [{"title": "안개 속", "summary": "하린이 들어간다."}]}
        model = ScriptedModel(
            chapters={3: [third]}, checks=[("state the core fact of the twist", "거울")]
        )
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 기승전결로 짜 줘."
        )
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_tool(model), ctx, action="continue", choice=1)
        outline_out = await _ok(_tool(model), ctx, action="continue")
        asked = [p for p in model.prompts if CHAPTER_PROMPT_MARK in p]
        twist, clue, resolution = _SPINE["twist"], _SPINE["clue"], _SPINE["resolution"]
        # Chapters 1-2 get the clue and must keep the twist; chapter 3 reveals it.
        assert all(twist not in p and clue in p for p in asked[:2])
        assert "It must not reveal the twist" in asked[1]
        assert twist in asked[2] and "the reader learns it in THIS chapter" in asked[2]
        assert clue not in asked[2]
        # Only the climax is told how the conflict ends; nothing before it.
        assert all(resolution not in p for p in asked[:3])
        assert resolution in asked[3] and "It must not" not in asked[3]
        # Each call is told the chapters before it.
        assert "Chapter 1:\n안개가 오는 밤" in asked[1] and "Chapter 2:\n새벽" in asked[2]
        assert outline_out["spine"] == {"clue": clue, "resolution": resolution}

    def test_an_outline_on_a_structure_has_no_more_chapters_than_acts(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_outline.py :: for chapter in items[: len(acts) or MAX_CHAPTERS]:
        Becomes: for chapter in items[:MAX_CHAPTERS]:
        """
        data = {"chapters": [{"title": str(n), "scenes": [{"title": "s"}]} for n in range(5)]}
        outline = outline_from(data, [], acts=("a", "b", "c"))
        assert [(c.id, c.act) for c in outline.chapters] == [
            ("ch01", "a"),
            ("ch02", "b"),
            ("ch03", "c"),
        ]


class TestContinuityAfterWriting:
    async def test_a_contradiction_is_shown_with_its_quote_and_a_rewrite_offered(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: kept += [f.record() for f in findings]
        Becomes: kept += []
        Killed by: examples/story/src/uclone_x_story/start.py :: if rewrite and flagged:
        Becomes: if flagged:
        Killed by: examples/story/src/uclone_x_story/start_scene.py :: store = checked_store(self.work.story_id, outline, codex, proposals)
        Becomes: store = checked_store(self.work.story_id, outline, codex, [])
        """
        # 하린 is still a pending proposal, and so is the death the first scene wrote: the
        # check holds the second scene to both, through the store's proposed view.
        model = ScriptedModel(scenes=[_DEAD_SCENE], facts=[[_HARIN_DEAD], [_HARIN_BACK]])
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        assert out["stage"] == "review_chapter"
        [finding] = out["continuity"]
        assert finding["scene_id"] == "ch01.s02"
        assert finding["quote"] == _BACK_LINE
        assert finding["note"] == (
            f"‘목소리’ 장면의 “{_BACK_LINE}” — 하린은 앞의 ‘계단’ 장면에서 죽었는데, "
            "이 장면에서 다시 움직입니다."
        )
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "맞지 않는 곳이 1곳" in ko and finding["note"] in ko
        assert "다시 쓸까요?" in ko
        # Never rewritten on its own: each scene was drafted once.
        assert len([p for p in model.prompts if "Write scene" in p]) == 2
        found = _work(tmp_path, out["open_story_id"]).manuscript("ch01.s01")
        assert found is not None and _DEAD_LINE in found.text

        # Going on writes the next chapter; the flagged scene is left as it was.
        again = await _ok(
            _tool(model), _ctx(tmp_path, story=out["open_story_id"]), action="continue"
        )
        assert "rewrite" not in again["done_now"] and "rewritten" not in again
        found = _work(tmp_path, out["open_story_id"]).manuscript("ch01.s02")
        assert found is not None and _BACK_LINE in found.text

    async def test_asked_to_rewrite_only_the_flagged_scene_is_written_again(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if rewrite and flagged:
        Becomes: if False:
        """
        model = ScriptedModel(
            scenes=[_DEAD_SCENE, _SCENE, _SCENE.replace("올랐다", "올랐다고 꿈꾸었다")],
            facts=[[_HARIN_DEAD], [_HARIN_BACK]],
        )
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        story_id = out["open_story_id"]
        again = await _ok(
            _tool(model), _ctx(tmp_path, story=story_id), action="continue", rewrite=True
        )
        assert again["done_now"] == ["rewrite"]
        assert again["rewritten"] == ["ch01.s02"]
        asked = [p for p in model.prompts if "Write scene" in p][-1]
        assert "disagree with the story's codex" in asked and _BACK_LINE in asked
        found = _work(tmp_path, story_id).manuscript("ch01.s02")
        assert found is not None and "꿈꾸었다" in found.text
        assert again["continuity"] == []  # checked again, and it agrees now
        assert "장면 1개를 다시 써서 저장했습니다" in again[REPLY_NOTE_KEY]["ko"]

    async def test_each_written_scene_is_read_by_the_model_once(self, tmp_path: Path) -> None:
        """#2207: one reading per written scene feeds the codex growth and the check.

        Before, each scene was read twice -- for what it adds (`enrich_prompt`) and again for
        the check (`continuity_prompt`) -- so a written scene cost a draft, two readings and
        the plan check: four calls. Now three.

        Killed by: examples/story/src/uclone_x_story/start_scene.py :: reading = self.readings.get(scene_id)
        Becomes: reading = None
        """
        model = ScriptedModel(scenes=[_DEAD_SCENE], facts=[[_HARIN_DEAD], [_HARIN_BACK]])
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        written = out["scenes_written"]
        assert len(written) == 2
        drafts = [p for p in model.prompts if "Write scene" in p]
        readings = [p for p in model.prompts if p.startswith(ENRICH_MARKER)]
        plans = [p for p in model.prompts if p.startswith(PLAN_MARKER)]
        assert len(drafts) == len(readings) == len(plans) == len(written)
        assert model.readings == len(written)
        assert (len(drafts) + len(readings) + len(plans)) / len(written) == 3
        # And the check ran, on the reading's facts.
        assert [f["scene_id"] for f in out["continuity"]] == ["ch01.s02"]
        assert _state(tmp_path, out["open_story_id"])["continuity_checked"] == written

    async def test_a_scene_whose_facts_cannot_be_read_is_not_called_consistent(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: unread.append(scene_id)
        Becomes: pass
        """
        model = ScriptedModel(facts=["모르겠습니다", "모르겠습니다"])
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        assert out["continuity"] == []
        assert out["continuity_unread"] == ["ch01.s01", "ch01.s02"]
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "장면 2개는 설정집과 대조하지 못했습니다" in ko
        assert "어긋나는 곳은 찾지 못했습니다" not in ko


_POSTMAN_LINE = "문 앞에는 우체부 도윤이 젖은 편지를 들고 서 있었다."
_POSTMAN_SCENE = f"{_SCENE} {_POSTMAN_LINE}"
_POSTMAN_GROWTH: dict[str, Any] = {
    "new": [
        {
            "kind": "characters",
            "name": "도윤",
            "profile": "섬에 편지를 나르는 우체부.",
            "gender": "male",
            "quote": "우체부 도윤이 젖은 편지를 들고",
        }
    ],
    "facts": [],
}
_BELL_SCENE = f"{_SCENE} 계단 끝에서 하린은 작은 은방울 하나를 주웠다."
_BELL_GROWTH: dict[str, Any] = {
    "new": [
        {
            "kind": "items",
            "name": "은방울",
            "profile": "계단 끝에 떨어져 있던 방울.",
            "quote": "작은 은방울 하나를 주웠다",
        }
    ],
    "facts": [],
}


class TestEachWrittenSceneGrowsTheCodex:
    """§1.1 row 7: what `story_start` writes is read for the codex, as a manuscript write is."""

    async def _first_chapter(
        self, tmp_path: Path, model: ScriptedModel
    ) -> tuple[dict[str, Any], str]:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)
        await _ok(_tool(model), ctx, action="continue", choice=1)
        await _ok(_tool(model), ctx, action="continue", go_on=True)
        await _ok(_tool(model), ctx, action="continue", go_on=True)  # the first plan
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        return out, story_id

    async def test_a_written_chapter_proposes_what_its_scenes_add(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: await self._grow(scene.id, text, notes)
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/start_state.py :: update.setdefault("growth_now", list(self.grown))
        Becomes: update.setdefault("growth_now", [])
        """
        model = ScriptedModel(scenes=[_POSTMAN_SCENE], growth=[_POSTMAN_GROWTH])
        out, story_id = await self._first_chapter(tmp_path, model)
        assert out["chapter_written"] == 1
        read = [p for p in model.prompts if p.startswith(ENRICH_MARKER)]
        assert len(read) == 2 and _POSTMAN_LINE in read[0]
        [pid] = out["codex_growth"]
        work = _work(tmp_path, story_id)
        [proposal] = [p for p, _ in work.proposals()[0] if p.id == pid]
        assert proposal.status == "pending" and proposal.room_id == ROOM
        assert proposal.new_entry().name == "도윤"
        assert proposal.evidence[0].scene_id == "ch01.s01"
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "이번에 쓴 장면에서 새로 생긴 인물, 장소, 물건, 관계, 상태 변화 1건을" in ko
        assert pid not in ko and "ch01" not in ko

    async def test_a_failed_reading_does_not_stop_the_chapter(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/enrich.py :: except Exception:
        Becomes: except ValueError:
        """
        model = ScriptedModel(growth=[ConnectionError("refused"), ConnectionError("refused")])
        out, story_id = await self._first_chapter(tmp_path, model)
        assert out["stage"] == "review_chapter" and out["chapter_written"] == 1
        assert _work(tmp_path, story_id).written_scenes() == ["ch01.s01", "ch01.s02"]
        assert "codex_growth" not in out
        assert any("could not be read for what it adds" in n for n in out["notes"])
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "첫 장을 썼습니다" in ko
        assert "보강안" not in ko and "refused" not in ko

    async def test_a_rewritten_scene_is_read_again(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: await self._grow(scene_id, text, notes)
        Becomes: pass
        """
        rewritten = f"{_SCENE} {_POSTMAN_LINE}"
        model = ScriptedModel(
            scenes=[_DEAD_SCENE, _SCENE, rewritten],
            facts=[[_HARIN_DEAD], [_HARIN_BACK]],
            growth=[{"new": [], "facts": []}, {"new": [], "facts": []}, _POSTMAN_GROWTH],
        )
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        again = await _ok(
            _tool(model),
            _ctx(tmp_path, story=out["open_story_id"]),
            action="continue",
            rewrite=True,
        )
        assert again["done_now"] == ["rewrite"]
        # The first call proposed the death the first scene wrote; the rewrite, the postman.
        assert len(again["codex_growth"]) == 2
        assert "상태 변화 1건을 설정집 보강안으로 올렸습니다" in again[REPLY_NOTE_KEY]["ko"]


class TestTheBrief:
    def test_only_what_the_request_left_open_is_filled_in(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: fields[key] = value
        Becomes: fields[key] = DEFAULTS[language][key]
        """
        brief = brief_from({"genre": "호러", "tone": "", "length": "단편"}, REQUEST)
        assert brief.genre == "호러"
        assert brief.length == "단편"
        assert brief.tone == DEFAULTS["ko"]["tone"]
        assert brief.audience == DEFAULTS["ko"]["audience"]
        assert brief.from_request == ["genre", "length"]

    def test_a_reply_with_no_json_leaves_every_field_at_its_default(self) -> None:
        brief = brief_from(json_object("모르겠습니다"), REQUEST)
        assert brief.genre == DEFAULTS["ko"]["genre"]
        assert brief.from_request == []
        assert brief.title


class TestThePremises:
    async def test_each_premise_is_its_own_call_with_its_own_cards(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_premise.py :: seed = secrets.randbelow(MAX_SEED + 1)
        Becomes: seed = 7
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        state = _state(tmp_path, out["open_story_id"])
        premise_prompts = [p for p in model.prompts if "Write one premise" in p]
        assert len(premise_prompts) == 3
        assert state["premises"][0]["muse_genre"] == "horror"
        cards = [json.dumps(p["card"], sort_keys=True) for p in state["premises"]]
        assert len(set(cards)) == 3
        assert len({p["seed"] for p in state["premises"]}) == 3
        assert len(set(premise_prompts)) == 3

    async def test_each_premise_is_built_around_its_own_engine(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_premise.py :: premises = [await draft(n, engines[n - 1]) for n in range(1, PREMISE_COUNT + 1)]
        Becomes: premises = [await draft(n, engines[0]) for n in range(1, PREMISE_COUNT + 1)]
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        state = _state(tmp_path, out["open_story_id"])
        engines = [p["engine"] for p in state["premises"]]
        assert len(set(engines)) == 3 and set(engines) <= set(ENGINES)
        prompts = [p for p in model.prompts if "Write one premise" in p]
        for engine, prompt in zip(engines, prompts, strict=True):
            assert f"whose engine is {engine}" in prompt
            assert "do not retell the request" in prompt

    async def test_a_premise_that_drops_a_must_have_is_rewritten_told_all_of_them(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_premise.py :: if len(still) < len(lacking):
        Becomes: if False:
        """
        brief = {"genre": "호러", "must_have": ["고양이", "등대지기 소녀"]}
        model = ScriptedModel(brief=brief, lacking=1)
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        first = _state(tmp_path, out["open_story_id"])["premises"][0]
        assert first["premise"] == _REWRITE
        assert first["title"] == "고양이 등대"
        assert first["rewritten_for"] == ["고양이"]
        assert first["missing"] == []
        (revise,) = [p for p in model.prompts if "Rewrite the premise" in p]
        assert "- 고양이\n- 등대지기 소녀\n" in revise  # every must-have, not only the lost one
        assert "leaves out: 고양이." in revise

    async def test_a_rewrite_that_holds_no_more_is_not_kept(self, tmp_path: Path) -> None:
        brief = {"genre": "호러", "must_have": ["고양이"]}
        model = ScriptedModel(brief=brief, lacking=2)
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        first = _state(tmp_path, out["open_story_id"])["premises"][0]
        assert first["premise"] == _PREMISES[0][1]
        assert first["missing"] == ["고양이"]

    async def test_the_premises_are_measured_by_the_embedder(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: if embedder is not None and texts:
        Becomes: if False:
        """
        embedder = FakeEmbedder(_APART)
        out = await _ok(
            _tool(ScriptedModel(), embedder), _ctx(tmp_path), action="start", request=REQUEST
        )
        # Each premise is measured with its title (#2301).
        assert embedder.calls == [[f"{t}\n{p}" for t, p in _PREMISES]]
        similarity = out["premise_similarity"]
        assert similarity["measure"] == "embedding_cosine"
        assert similarity["pairs"] == [0.0, 0.0, 0.0]
        assert similarity["limit"] == MAX_PREMISE_COSINE

    async def test_two_premises_alike_are_drawn_again_once(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_premise.py :: if (same is not None or not similarity.distinct) and len(engines) > PREMISE_COUNT:
        Becomes: if False:
        """
        alike = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        model = ScriptedModel(later_premises=[_REDRAWN])
        out = await _ok(
            _tool(model, FakeEmbedder(alike, _APART)),
            _ctx(tmp_path),
            action="start",
            request=REQUEST,
        )
        state = _state(tmp_path, out["open_story_id"])
        prompts = [p for p in model.prompts if "Write one premise" in p]
        assert len(prompts) == 4
        # The later premise of the closest pair, the second, around a fourth engine.
        assert state["premises"][1]["premise"] == _REDRAWN[1]
        assert len({p["engine"] for p in state["premises"]}) == 3
        assert state["premise_similarity"]["closest"] == 0.0
        assert not any("read alike" in n for n in state["notes"])

    async def test_with_no_embedder_or_a_failing_one_the_measure_is_character_pairs(
        self, tmp_path: Path
    ) -> None:
        for embedder in (None, FakeEmbedder(fails=True)):
            out = await _ok(
                _tool(ScriptedModel(), embedder), _ctx(tmp_path), action="start", request=REQUEST
            )
            similarity = out["premise_similarity"]
            assert similarity["measure"] == "bigram_jaccard"
            assert similarity["limit"] == MAX_PREMISE_BIGRAM
            assert similarity["closest"] < MAX_PREMISE_BIGRAM


class TestPureHelpers:
    def test_a_stray_quote_before_a_key_does_not_lose_the_reply(self) -> None:
        r"""Killed by: examples/story/src/uclone_x_story/start_state.py :: for attempt in (body, _STRAY_QUOTE.sub(r"\1", body)):
        Becomes: for attempt in (body,):
        """
        # The three shapes qwen3:8b wrote on 2026-09-29, each after a place entry.
        for stray in ('" "name"', '""name"', '"\n      "name"'):
            reply = (
                '{"characters": [{"name": "리안", "profile": "견습 마법사."}], '
                f'"places": [{{"name": "마을"}}, {{\n      {stray}: "용의 눈의 장소"}}]}}'
            )
            found = json_object(reply)
            assert found is not None, stray
            assert found["characters"][0]["name"] == "리안"
            assert found["places"][1] == {"name": "용의 눈의 장소"}
        assert json_object('{"a": "그는 " "말했다"}') is None  # nothing else is repaired

    def test_json_is_read_from_a_fence_or_prose(self) -> None:
        assert json_object('앞말 {"a": 1} 뒷말') == {"a": 1}
        assert json_object('```json\n{"b": [1]}\n```') == {"b": [1]}
        assert json_object("[1, 2]") is None

    def test_quick_phrases(self) -> None:
        assert wants_it_now("그냥 바로 써줘")
        assert wants_it_now("알아서 써 줘")
        assert not wants_it_now("천천히 같이 만들어 봐요")

    def test_overlap_is_one_for_the_same_text_and_low_for_different_ones(self) -> None:
        assert bigram_overlap("안개 속 등대", "안개 속 등대") == 1.0
        assert bigram_overlap("안개 속 등대", "우주 정거장의 반란") < 0.1

    async def test_the_same_text_is_alike_by_either_measure(self) -> None:
        same = ["안개 속 등대", "안개 속 등대"]
        by_pairs = await premise_similarity(same, None)
        assert by_pairs.pairs == (1.0,) and not by_pairs.distinct
        by_vectors = await premise_similarity(same, FakeEmbedder([[3.0, 4.0], [3.0, 4.0]]))
        assert by_vectors.measure == "embedding_cosine" and not by_vectors.distinct
        assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_a_status_is_read_from_the_model_word_and_is_unknown_otherwise(self) -> None:
        assert status_of("missing") == "missing"
        assert status_of("실종 상태") == "missing"
        assert status_of("행방불명") == "missing"  # it holds "불명" too
        assert status_of("사망") == "dead"
        assert status_of("생존 여부 불명") == "unknown"
        assert status_of("Alive") == "alive"
        assert status_of(None) == "unknown"
        assert status_of("잠들어 있음") == "unknown"

    def test_an_outline_takes_its_ids_from_code(self) -> None:
        outline = outline_from(
            {"chapters": [{"title": "하나", "scenes": [{"title": "a"}, {"summary": ""}]}, {}]},
            [],
        )
        assert [c.id for c in outline.chapters] == ["ch01"]
        assert [s.id for s in outline.chapters[0].scenes] == ["ch01.s01"]


_TWIST = "목소리의 주인은 실종된 등대지기, 하린의 아버지였다."


_THREE_SPINE = {
    "central_conflict": "하린은 안개 속 목소리의 정체를 밝히려 한다.",
    "twist": _TWIST,
    "clue": "계단에 낡은 장갑 한 짝이 떨어져 있다.",
    "resolution": "하린은 등대 불을 켜 아버지를 안개 밖으로 이끈다.",
}


def _chapter(title: str, stakes: str, summary: str) -> dict[str, Any]:
    return {
        "title": title,
        "stakes": stakes,
        "scenes": [{"title": title, "summary": summary, "characters": ["하린"]}],
    }


def _three(first: str) -> dict[int, list[dict[str, Any]]]:
    """Replies to a three-act outline's chapter calls, the first chapter's summary `first`."""
    return {
        1: [_chapter("안개가 오는 밤", "등대의 불", first)],
        2: [_chapter("젖은 발자국", "하린의 목숨", "발자국 끝에서 아버지의 목소리를 알아본다.")],
        3: [_chapter("마지막 불빛", "섬 전체", "하린이 안개 속으로 들어가 목소리와 마주한다.")],
    }


_EARLY = "하린은 목소리가 아버지의 것임을 알게 된다."
_CLUE = "하린이 계단에서 낡은 장갑 한 짝을 줍는다."


class TestSceneTitles:
    """`scene_title`: a planned scene's title is a name, never a clause cut from its summary."""

    def test_a_real_title_is_kept(self) -> None:
        assert scene_title("장로의 속삭임", "하린은 장로를 찾아간다.") == "장로의 속삭임"

    def test_a_clause_ending_mid_sentence_becomes_an_of_phrase_from_the_summary(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_outline.py :: if _CLAUSE_END.search(title) or len(title) > MAX_SCENE_TITLE:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/start_outline.py :: if m.group(1) not in _PRONOUN_OWNERS:
        Becomes: if True:
        """
        # A clause cut off at its connective, not the summary's opening.
        summary = "하린은 자신의 몸에서 기척을 느끼고, 장로의 계략을 깨닫고, 맞서기로 결심한다."
        title = scene_title("이상한 기척을 느끼며,", summary)
        assert title == "장로의 계략"
        assert title != summary and not summary.startswith(title)

    def test_the_summary_opening_is_not_a_title(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_outline.py :: return len(title) >= 8 and summary.startswith(title) and title != summary
        Becomes: return False
        """
        title = scene_title(
            "먹물이 방울을 울리며 계단",
            "먹물이 방울을 울리며 계단을 오르는 동안 등대의 불빛이 꺼진다.",
        )
        assert title == "등대의 불빛"

    def test_without_an_of_phrase_the_clause_is_cut_short_without_its_subject(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_outline.py :: head = head[1:]  # the subject ("이진호는") names the scene no better than its cast
        Becomes: pass  # the subject ("이진호는") names the scene no better than its cast
        """
        assert scene_title("", "하린은 문을 열고 들어가며, 먹물을 부른다.") == "문을 열고 들어가"
        assert scene_title("", "") == ""


class TestTheOutlineEscalates:
    def test_the_twist_is_never_before_the_middle_and_never_last(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/arc.py :: twist = marked or min(middle, count - 1)
        Becomes: twist = min(middle, count - 1)
        Killed by: examples/story/src/uclone_x_story/arc.py :: climax = count if count == twist + 1 else count - 1
        Becomes: climax = count
        """
        three = arc_for([["hook"], ["rising-action", "midpoint"], ["climax"]])
        assert three is not None
        assert three.functions == (
            ("setup",),
            ("complication", "reversal"),
            ("climax", "resolution"),
        )
        assert (three.twist_chapter, three.climax_chapter) == (2, 3)
        four = arc_for([["introduction"], ["development"], ["twist"], ["reconciliation"]])
        assert four is not None and (four.twist_chapter, four.climax_chapter) == (3, 4)
        # Five acts, the turning beat in act 4: the twist goes there, not the middle.
        five = arc_for([["a"], ["b"], ["c"], ["ordeal"], ["e"]])
        assert five is not None and (five.twist_chapter, five.climax_chapter) == (4, 5)
        # Six acts and no turning beat: the middle turns, and the climax is before the end.
        six = arc_for([["a"]] * 6)
        assert six is not None and (six.twist_chapter, six.climax_chapter) == (4, 5)
        assert six.functions[-1] == ("resolution",)
        assert all(a != b for a, b in zip(six.functions, six.functions[1:], strict=False))
        assert arc_for([["a"], ["b"]]) is None

    def test_a_chapter_before_the_twist_is_told_to_keep_it_hidden(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/arc.py :: if number < self.twist_chapter:
        Becomes: if False:
        """
        arc = arc_for([["introduction"], ["development"], ["twist"], ["reconciliation"]])
        assert arc is not None
        assert "the twist stays hidden" in arc.does(2)
        assert "the twist stays hidden" not in arc.does(3)
        assert "reveal the twist" in arc.does(3)
        assert "the conflict stays open" in arc.does(3) and "stays open" not in arc.does(4)

    async def test_the_check_finds_an_early_twist_and_an_early_ending(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/arc.py :: if shown and number < arc.twist_chapter:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/arc.py :: if await _settled(model, spine.conflict, text):
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/arc.py :: return reply.strip()[:1].upper() == "B"
        Becomes: return reply.strip().casefold().startswith("yes")
        Killed by: examples/story/src/uclone_x_story/arc.py :: if again:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/arc.py :: if spine.conflict.strip() and arc.twist_chapter <= number < arc.climax_chapter:
        Becomes: if spine.conflict.strip() and number < arc.climax_chapter:
        Killed by: examples/story/src/uclone_x_story/arc.py :: "Does this chapter state the core fact of the twist, even in different words "
        Becomes: "Does the reader learn this twist in this chapter -- is it revealed, stated "
        """
        arc = arc_for([["hook"], ["midpoint"], ["climax"]])
        assert arc is not None
        spine = Spine(conflict=_THREE_SPINE["central_conflict"], twist=_TWIST)
        model = ScriptedModel(
            checks=[
                ("state the core fact of the twist", _EARLY),
                ("How does this chapter end", "젖은 발자국"),
                ("Does chapter B only repeat", "Chapter B, the next one:\n마지막 불빛"),
            ]
        )
        texts = [chapter_text(_three(_EARLY)[n][0]) for n in (1, 2, 3)]
        found = [
            await chapter_problems(
                model, arc, n, texts[n - 1], spine=spine, previous=texts[n - 2] if n > 1 else None
            )
            for n in (1, 2, 3)
        ]
        assert found == [
            [{"kind": "twist_early", "chapter": 1}],
            [{"kind": "twist_missing", "chapter": 2}, {"kind": "resolved_early", "chapter": 2}],
            [{"kind": "repeats", "chapter": 3}],
        ]
        asked = [p for p in model.prompts if CHECK_MARKER in p]
        # Twist: chapters 1-2; settled: chapter 2 only (from the twist to the climax);
        # repeats: chapters 2-3.
        assert len(asked) == 5
        assert "Stakes: 등대의 불" in asked[0]
        # Settled is a choice between two endings, not a yes/no question.
        settled = [p for p in asked if "How does this chapter end" in p]
        assert len(settled) == 1 and "Answer with one letter, A or B" in settled[0]
        assert "젖은 발자국" in settled[0]

    async def test_a_chapter_that_gives_the_twist_away_is_asked_for_again_alone(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_outline.py :: if first:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/start_outline.py :: chapter = retried
        Becomes: pass
        """
        chapters = _three(_EARLY)
        chapters[1].append(_chapter("안개가 오는 밤", "등대의 불", _CLUE))
        model = ScriptedModel(
            spine=_THREE_SPINE,
            chapters=chapters,
            checks=[
                ("state the core fact of the twist", _EARLY),
                ("state the core fact of the twist", "젖은 발자국"),
            ],
        )
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_tool(model), ctx, action="continue", choice=1)
        outline_out = await _ok(_tool(model), ctx, action="continue")
        asked = [p for p in model.prompts if CHAPTER_PROMPT_MARK in p]
        # Chapter 1 twice, then chapters 2 and 3 once each.
        assert [p.split(CHAPTER_PROMPT_MARK)[1].split()[0] for p in asked] == ["1", "1", "2", "3"]
        assert "Chapter 1 already reveals the twist" in asked[1]
        assert "This chapter must set the conflict going" in asked[0]
        # The kept chapter 1, not the one that gave the twist away, is what chapter 2 sees.
        assert _CLUE in asked[2] and _EARLY not in asked[2]
        # The record keeps the codes; the result says them in words (#2371).
        check = _state(tmp_path, out["open_story_id"])["outline_check"]
        assert check["first"] == [{"kind": "twist_early", "chapter": 1}]
        assert check["final"] == [] and check["asked"] == 4
        assert check["chapters"][0]["asked"] == 2
        assert outline_out["arc"]["stakes"] == ["등대의 불", "하린의 목숨", "섬 전체"]
        outline, _ = _work(tmp_path, out["open_story_id"]).require_outline()
        assert outline.chapters[0].scenes[0].summary == _CLUE
        ko = outline_out[REPLY_NOTE_KEY]["ko"]
        assert "반전은 2장에서 처음 드러나고, 갈등은 3장에서 결판나도록" in ko
        assert "고칠 곳" not in ko

    async def test_what_the_check_still_finds_is_told_to_the_person(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_outline.py :: if len(later) < len(first):
        Becomes: if True:
        """
        chapters = _three(_EARLY)
        chapters[1].append(_chapter("안개가 오는 밤", "등대의 불", _EARLY + " 그리고 운다."))
        model = ScriptedModel(
            spine=_THREE_SPINE,
            chapters=chapters,
            checks=[
                ("state the core fact of the twist", _EARLY),
                ("state the core fact of the twist", "젖은 발자국"),
            ],
        )
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_tool(model), ctx, action="continue", choice=1)
        outline_out = await _ok(_tool(model), ctx, action="continue")
        # The record keeps the codes; the result says them in words (#2371).
        check = _state(tmp_path, out["open_story_id"])["outline_check"]
        # No better, so the first is kept.
        assert check["chapters"][0]["second"] == [{"kind": "twist_early", "chapter": 1}]
        assert check["final"] == [{"kind": "twist_early", "chapter": 1}]
        outline, _ = _work(tmp_path, out["open_story_id"]).require_outline()
        assert outline.chapters[0].scenes[0].summary == _EARLY
        ko = outline_out[REPLY_NOTE_KEY]["ko"]
        assert "아직 고칠 곳이 있습니다: 1장에서 반전이 너무 일찍 드러납니다." in ko
        assert "twist_early" not in ko


def _family_cast(harin: dict[str, Any], doyun: dict[str, Any]) -> str:
    return json.dumps(
        {
            "characters": [
                {"name": "하린", "profile": "등대지기의 딸.", "status": "alive", **harin},
                {"name": "도윤", "profile": "실종된 등대지기.", "status": "alive", **doyun},
            ],
            "places": [],
        },
        ensure_ascii=False,
    )


# The father lists his daughter; 하린 states no gender, so "daughter" gives her one, and a
# relative not in the cast is dropped.
_FATHER_CAST = _family_cast(
    {"family": []},
    {
        "gender": "male",
        "family": [{"relative": "하린", "is": "daughter"}, {"relative": "누군가", "is": "딸"}],
    },
)
_MOTHER_LINE = "도윤은 하린의 어머니였다"


class TestGenderAndKinship:
    async def test_the_cast_records_gender_and_family_on_both_sides(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_cast.py :: relations[other].setdefault(holder_id, INVERSE_ROLE[role])
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/start_cast.py :: gender = family_gender.get(name, "")
        Becomes: gender = ""
        """
        model = ScriptedModel(cast_replies=[_FATHER_CAST])
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        await _ok(
            _tool(model), _ctx(tmp_path, story=out["open_story_id"]), action="continue", choice=1
        )
        entries = {
            p.entry_id: p.new_entry()
            for p, _ in _work(tmp_path, out["open_story_id"]).proposals()[0]
        }
        harin, doyun = "harin", "doyun"
        # "daughter" gives 하린 a gender the reply left out, and 도윤 the other side of it;
        # a name not in the cast is dropped.
        assert entries[harin].state == {"status": "alive", "gender": "female"}
        assert entries[harin].relations == {doyun: "child"}
        assert entries[doyun].state == {"status": "alive", "gender": "male"}
        assert entries[doyun].relations == {harin: "parent"}
        assert '"family"' in _last(model, "List the main characters")

    async def test_the_cast_records_the_names_the_prose_will_use(self, tmp_path: Path) -> None:
        """#2217: the cast is asked for the names its prose will call each character by, and
        they are the entry's aliases, so a scene's reading finds them there; the name itself
        and a repeat are not kept twice.

        Killed by: examples/story/src/uclone_x_story/start_cast.py :: aliases=_aliases(entry.get("aliases"), name),
        Becomes: aliases=[],
        """
        cast_reply = _family_cast({"aliases": ["Harin", "하린", "Harin", 3]}, {"gender": "male"})
        model = ScriptedModel(cast_replies=[cast_reply])
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        await _ok(
            _tool(model), _ctx(tmp_path, story=out["open_story_id"]), action="continue", choice=1
        )
        entries = {
            p.entry_id: p.new_entry()
            for p, _ in _work(tmp_path, out["open_story_id"]).proposals()[0]
        }
        assert entries["harin"].aliases == ["Harin"]
        assert entries["doyun"].aliases == []
        assert '"aliases"' in _last(model, "List the main characters")

    @pytest.mark.parametrize(
        ("harin", "doyun", "kept"),
        [
            # 하린 lists her father as what she is to him: "daughter" contradicts his stated
            # gender and matches hers, so it is read the other way round.
            ({"gender": "female", "family": [{"relative": "도윤", "is": "daughter"}]}, {}, True),
            # The word matches neither side's stated gender: which way it runs is unknown.
            ({"gender": "male", "family": [{"relative": "도윤", "is": "daughter"}]}, {}, False),
        ],
    )
    async def test_a_relation_given_from_the_wrong_side_is_turned_or_dropped(
        self, tmp_path: Path, harin: dict[str, Any], doyun: dict[str, Any], kept: bool
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_cast.py :: holder, holder_id, other, other_id = name, own, relative, relative_id
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/start_cast.py :: if stated.get(name.casefold()) != word_gender:
        Becomes: if False:
        """
        cast_reply = _family_cast(harin, {"gender": "male", "family": [], **doyun})
        model = ScriptedModel(cast_replies=[cast_reply])
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        await _ok(
            _tool(model), _ctx(tmp_path, story=out["open_story_id"]), action="continue", choice=1
        )
        entries = {
            p.entry_id: p.new_entry()
            for p, _ in _work(tmp_path, out["open_story_id"]).proposals()[0]
        }
        assert entries["harin"].relations.get("doyun") == ("child" if kept else None)
        assert entries["doyun"].relations.get("harin") == ("parent" if kept else None)

    async def test_a_scene_that_calls_the_father_a_mother_is_caught(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/continuity.py :: axioms=[*axioms, *identity_axioms(submitted, axioms)],
        Becomes: axioms=axioms,
        Killed by: examples/story/src/uclone_x_story/continuity.py :: held.append(SubmittedFact(subject, "gender", said, fact.quote))
        Becomes: pass
        """
        doyun, harin = "doyun", "harin"
        scene = f"{_SCENE} {_MOTHER_LINE}."
        mother = {
            "subject": doyun,
            "key": "family",
            "of": harin,
            "value": "mother",
            "quote": _MOTHER_LINE,
            "time": "now",
            "polarity": "asserted",
        }
        model = ScriptedModel(cast_replies=[_FATHER_CAST], scenes=[scene], facts=[[mother]])
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        [finding] = out["continuity"]
        assert finding["quote"] == _MOTHER_LINE
        assert "설정집에는 도윤이 남성인 것으로" in finding["note"]
        assert "gender" not in finding["note"] and doyun not in finding["note"]
        asked = _last(model, ENRICH_MARKER)
        assert "- family: what the subject is to" in asked and "- gender:" in asked

    async def test_a_brother_called_a_son_is_a_kinship_contradiction(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/continuity.py :: held.append(SubmittedFact(subject, relation_predicate(other), role, fact.quote))
        Becomes: pass
        """
        doyun, harin = "doyun", "harin"
        line = "하린은 도윤의 누나였다"
        sister = {
            "subject": harin,
            "key": "family",
            "of": doyun,
            "value": "누나",
            "quote": line,
            "time": "now",
            "polarity": "asserted",
        }
        model = ScriptedModel(
            cast_replies=[_FATHER_CAST], scenes=[f"{_SCENE} {line}."], facts=[[sister]]
        )
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        [finding] = out["continuity"]
        assert "하린이 도윤의 자녀인 것으로" in finding["note"]

    async def test_a_kinship_read_from_a_line_that_names_none_is_dropped(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/continuity.py :: if not found or not _names_family(fact.quote):
        Becomes: if not found:
        """
        doyun, harin = "doyun", "harin"
        line = "도윤은 하린에게 등대를 지키는 법을 가르쳐 주었다"
        wife = {
            "subject": harin,
            "key": "family",
            "of": doyun,
            "value": "wife",
            "quote": line,
            "time": "now",
            "polarity": "asserted",
        }
        model = ScriptedModel(
            cast_replies=[_FATHER_CAST], scenes=[f"{_SCENE} {line}."], facts=[[wife]]
        )
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        assert out["continuity"] == []

    async def test_when_some_scenes_went_unread_the_rest_are_not_called_all_clean(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: elif checked and unread:
        Becomes: elif False:
        """
        model = ScriptedModel(facts=[[], "모르겠습니다"])
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "대조한 장면 1개에서는" in ko and "장면 1개는 설정집과 대조하지 못했습니다" in ko
        assert "쓴 장면의 인물 생사" not in ko

    def test_a_reply_with_quotes_inside_a_value_is_still_read(self) -> None:
        r"""Killed by: src/uclone_x/knowledge/extraction.py :: fields = _fields_read(STRAY_QUOTE.sub(r"\1", body))
        Becomes: fields = None
        """
        reply = (
            '{"facts": [{"subject": "doyun", "key": "gender", "value": "female", '
            '"quote": "그가 "딸아" 하고 불렀다", "time": "now", "polarity": "asserted"}]}'
        )
        [fact] = parse_extraction(reply)
        assert fact.quote == '그가 "딸아" 하고 불렀다' and fact.value == "female"


# -- "yes" alone, and the genre proposed when the request names none (§1.1 1 and 2) ------

BARE = "소설 하나 써줘"
_BARE_BRIEF: dict[str, Any] = {"title": "새 이야기", "genre": None, "must_have": [], "avoid": []}


def _seeded(model: ScriptedModel, seed: int = 7) -> StoryStartTool:
    return StoryStartTool(
        model_factory=lambda _context: model,
        embedder_factory=lambda _context: None,
        rng=random.Random(seed),
    )


def _story_genre(workspace: Path, story_id: str) -> object:
    return StoryLibrary(workspace).load(story_id).genre


class _NoRandom(random.Random):
    def sample(self, population: Any, k: int, *, counts: Any = None) -> list[Any]:
        raise AssertionError("a genre was drawn for a request that named one")


class TestYesAloneCarriesTheStory:
    async def test_a_bare_request_reaches_a_written_chapter_on_yes_alone(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_premise.py :: if params.go_on:
        Becomes: if False:
        """
        model = ScriptedModel(
            brief=_BARE_BRIEF,
            scenes=[_POSTMAN_SCENE, _SCENE, _BELL_SCENE, _SCENE],
            growth=[_POSTMAN_GROWTH, {"new": [], "changes": []}, _BELL_GROWTH],
        )
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=BARE)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)
        assert out["stage"] == "choose_genre"
        assert out["premises"] == []  # nothing is drawn before the genre is taken
        proposed = out["genre_options"][0]
        assert "(추천)" in out[REPLY_NOTE_KEY]["ko"] and proposed in out[REPLY_NOTE_KEY]["ko"]

        outs: list[dict[str, Any]] = []
        for _ in range(8):
            out = await _ok(_seeded(model), ctx, action="continue", go_on=True)
            outs.append(out)
            if out["stage"] == "done":
                break
        stages = [o["stage"] for o in outs]
        assert stages == [
            "choose_premise",
            "review_cast",
            "review_outline",
            "review_chapter",
            "review_chapter",
            "done",
        ]
        assert out["chosen"] == 1 and out["recommended"] == 1
        assert out["brief"]["genre"] == proposed
        assert _story_genre(tmp_path, story_id) == proposed
        # "Yes" at the outline showed the first chapter's plan; "yes" to it wrote the
        # first chapter, and "yes" to the second's plan wrote the second.
        settled, first = outs[3], outs[4]
        assert settled["details_chapter"] == 1 and "chapter_written" not in settled
        assert first["chapter_written"] == 1 and first["details_chapter"] == 2
        assert _work(tmp_path, story_id).written_scenes() == [
            "ch01.s01",
            "ch01.s02",
            "ch02.s01",
            "ch02.s02",
        ]
        assert out["chapter_written"] == 2
        assert "마지막 장까지 모두 썼습니다" in out[REPLY_NOTE_KEY]["ko"]
        # "Yes" wrote over the cast as proposed and said so; it approved nothing.
        assert len(first["cast_assumed"]) == 3
        assert "승인된 것으로 보고 썼습니다" in first[REPLY_NOTE_KEY]["ko"]
        # The "yes" at the outline settled the plot and proposed its key events.
        assert settled["plot_locked"] is True and _state(tmp_path, story_id)["plot_locked"]
        assert "줄거리를 확정했습니다" in settled[REPLY_NOTE_KEY]["ko"]
        kg = {k["proposal"] for k in out["plot_kg"]}
        assert kg and kg <= {p.id for p, _ in _work(tmp_path, story_id).proposals()[0]}
        assert all(p.status == "pending" for p, _ in _work(tmp_path, story_id).proposals()[0])
        # Each chapter's scenes grew the codex, and the proposals add up across chapters;
        # "yes" carried the flow over them and approved none of them.
        assert len(first["codex_growth"]) == 1 and len(out["codex_growth"]) == 2
        assert out["codex_growth"][0] == first["codex_growth"][0]
        grown = {
            p.id: p
            for p, _ in _work(tmp_path, story_id).proposals()[0]
            if p.id in out["codex_growth"]
        }
        assert sorted(p.new_entry().name for p in grown.values()) == ["도윤", "은방울"]
        assert all(p.status == "pending" for p in grown.values())
        for said in (first, out):
            ko = said[REPLY_NOTE_KEY]["ko"]
            assert "상태 변화 1건을 설정집 보강안으로 올렸습니다" in ko
            assert not any(i in ko for i in out["codex_growth"])

    async def test_the_premise_stop_says_which_premise_yes_takes(self, tmp_path: Path) -> None:
        out = await _ok(_tool(ScriptedModel()), _ctx(tmp_path), action="start", request=REQUEST)
        assert out["recommended"] == 1
        assert "‘네’라고만 하셔도 1번으로 진행합니다" in out[REPLY_NOTE_KEY]["ko"]

    async def test_the_cast_stop_says_what_yes_does_and_yes_does_it(self, tmp_path: Path) -> None:
        """The cast stop says one thing in both languages: "yes" makes the outline, and a
        bare "yes" at the outline plans (and the next "yes" writes) the first chapter over
        proposals still undecided, which stay pending. The flow then does exactly that.

        Killed by: examples/story/src/uclone_x_story/start_notes.py :: "고칠 수 있습니다. ‘네’라고 하시면 개요를 만들겠습니다. 첫 장은 제안을 모두 "
        Becomes: "고칠 수 있습니다. 첫 장은 제안을 승인하신 뒤에 쓰겠습니다. 첫 장은 제안을 모두 "
        Killed by: examples/story/src/uclone_x_story/start_notes.py :: "chapter waits until every proposal is decided; but if you just say yes at the "
        Becomes: "chapter waits for your approval. "
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        cast_out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert cast_out["stage"] == "review_cast"
        ko, en = cast_out[REPLY_NOTE_KEY]["ko"], cast_out[REPLY_NOTE_KEY]["en"]
        assert "‘네’라고 하시면 개요를 만들겠습니다" in ko
        assert (
            "개요에서 ‘네’라고만 하시면 정하지 않은 제안도 제안대로 보고 첫 장을 계획하고 쓰며"
            in ko
        )
        assert "승인하신 뒤에 쓰겠습니다" not in ko
        assert "Say yes and I will make the outline" in en
        assert "if you just say yes at the outline" in en
        assert "waits for your approval" not in en

        outline = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert outline["stage"] == "review_outline"
        assert _work(tmp_path, out["open_story_id"]).written_scenes() == []
        plan = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert plan["details_chapter"] == 1 and len(plan["cast_assumed"]) == 3
        chapter = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert chapter["chapter_written"] == 1
        assert len(chapter["cast_assumed"]) == 3
        work = _work(tmp_path, out["open_story_id"])
        assert all(p.status == "pending" for p, _ in work.proposals()[0])

    async def test_a_number_beats_yes(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_premise.py :: if params.choice is not None and params.choice <= len(options):
        Becomes: if False:
        """
        model = ScriptedModel(brief=_BARE_BRIEF)
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=BARE)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        third = out["genre_options"][2]
        premises = await _ok(_seeded(model), ctx, action="continue", choice=3, go_on=True)
        assert premises["brief"]["genre"] == third
        cast = await _ok(_seeded(model), ctx, action="continue", choice=2, go_on=True)
        assert cast["chosen"] == 2

    async def test_a_request_that_is_more_than_yes_does_not_skip_the_cast_wait(
        self, tmp_path: Path
    ) -> None:
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_tool(model), ctx, action="continue", go_on=True)
        await _ok(_tool(model), ctx, action="continue", go_on=True)
        waiting = await _ok(_tool(model), ctx, action="continue")
        assert waiting["stage"] == "review_outline" and len(waiting["cast_waiting"]) == 3


class TestTheGenreIsProposedOnlyWhenLeftOpen:
    async def test_a_stated_genre_is_never_replaced(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if "genre" in brief.from_request
        Becomes: if False
        """
        model = ScriptedModel()  # the request says 호러
        tool = StoryStartTool(
            model_factory=lambda _context: model,
            embedder_factory=lambda _context: None,
            rng=_NoRandom(),
        )
        out = await _ok(tool, _ctx(tmp_path), action="start", request=REQUEST)
        assert out["stage"] == "choose_premise"
        assert "genre_options" not in out
        assert out["brief"]["genre"] == "호러"
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        for _ in range(3):
            out = await _ok(tool, ctx, action="continue", go_on=True)
        assert out["stage"] == "review_chapter" and out["brief"]["genre"] == "호러"
        assert _story_genre(tmp_path, out["story_id"]) == "호러"

    async def test_a_named_genre_replaces_the_proposal_before_premises_are_drawn(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_premise.py :: if params.genre and params.genre.strip():
        Becomes: if False:
        """
        model = ScriptedModel(brief=_BARE_BRIEF)
        out = await _ok(_seeded(model, 3), _ctx(tmp_path), action="start", request=BARE)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        wanted = next(g for g in ("로맨스", "미스터리") if g != out["genre_options"][0])
        premises = await _ok(_seeded(model), ctx, action="continue", genre=wanted)
        assert premises["stage"] == "choose_premise"
        assert premises["brief"]["genre"] == wanted
        assert f"Genre: {wanted}" in _last(model, "Write one premise")
        assert _story_genre(tmp_path, out["open_story_id"]) == wanted

    async def test_words_that_name_no_genre_are_refused_and_change_nothing(
        self, tmp_path: Path
    ) -> None:
        model = ScriptedModel(brief=_BARE_BRIEF)
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=BARE)
        story_id = out["open_story_id"]
        before = _state(tmp_path, story_id)
        result = await _seeded(model).execute(
            {"action": "continue"}, _ctx(tmp_path, story=story_id)
        )
        assert not result.success and result.error is not None and "genre" in result.error
        assert _state(tmp_path, story_id) == before

    async def test_asked_at_once_it_takes_the_drawn_genre_and_says_so(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: if first and state.quick and state.genre_options and not state.genre_answered:
        Becomes: if False:
        """
        model = ScriptedModel(brief=_BARE_BRIEF)
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=BARE + " 바로 써줘")
        genre = out["genre_options"][0]
        assert out["stage"] == "review_chapter" and out["brief"]["genre"] == genre
        assert f"요청에 장르가 없어 {genre}로 정해 썼습니다" in out[REPLY_NOTE_KEY]["ko"]

    def test_the_proposal_is_drawn_at_random_from_the_muse_tables(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: return [GENRE_LABELS[k][language] for k in rng.sample(keys, min(GENRE_OPTIONS, len(keys)))]
        Becomes: return [GENRE_LABELS[k][language] for k in keys[:GENRE_OPTIONS]]
        """
        tables = ["fantasy", "horror", "mystery", "romance", "science-fiction", "western"]
        drawn = [genre_options(tables, random.Random(seed), "ko") for seed in range(20)]
        assert drawn[0] == genre_options(tables, random.Random(0), "ko")  # the seed decides
        assert len({d[0] for d in drawn}) > 1  # the recommendation is not fixed
        for options in drawn:
            assert len(options) == 3 == len(set(options))
            assert set(options) <= {"판타지", "호러", "미스터리", "로맨스", "SF"}
        assert genre_options(["romance"], random.Random(0), "ko") == ["로맨스"]


class TestAGenreCountsAsStatedOnlyWhenTheRequestNamesIt:
    async def test_a_genre_the_model_made_up_is_proposed_over(self, tmp_path: Path) -> None:
        """qwen3:8b read "소설 하나 써줘" as genre "일반", stated (2026-09-29).

        Killed by: examples/story/src/uclone_x_story/start_state.py :: if key == "genre" and value and not names_genre(request, value):
        Becomes: if False:
        """
        model = ScriptedModel(brief={**_BARE_BRIEF, "genre": "일반"})
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=BARE)
        assert out["stage"] == "choose_genre"
        assert len(out["genre_options"]) == 3
        assert "genre" not in out["brief"]["from_request"]

    def test_the_request_s_own_words_decide(self) -> None:
        for genre, request in (
            ("일반", "소설 하나 써줘"),
            ("스릴러", "무서운 이야기 하나"),
            ("SF", "a story about a transfer student"),
        ):
            assert "genre" not in brief_from({"genre": genre}, request).from_request, request
        for genre, request in (
            ("드라마", "가족 드라마"),
            ("가족 드라마", "가족 드라마 써 줘"),
            ("미스터리", "추리 단편"),
            ("일반", "일반 소설 하나"),
            ("science fiction", "an SF story"),
        ):
            brief = brief_from({"genre": genre}, request)
            assert "genre" in brief.from_request and brief.genre == genre, request


class TestTheOutlineTakesFeedbackUntilThePlotIsSettled:
    async def _at_outline(self, tmp_path: Path, model: ScriptedModel) -> tuple[str, ToolContext]:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_tool(model), ctx, action="continue", choice=1)
        outline = await _ok(_tool(model), ctx, action="continue")
        assert outline["stage"] == "review_outline"
        assert "‘네’라고 하시면 이 줄거리로 확정하고" in outline[REPLY_NOTE_KEY]["ko"]
        return out["open_story_id"], ctx

    async def test_feedback_revises_the_outline_and_stops_again(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if feedback and not self.state.quick:
        Becomes: if False:
        """
        model = ScriptedModel()
        story_id, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", changes="반전을 더 일찍 드러내 줘")
        assert out["stage"] == "review_outline"
        assert out["done_now"] == ["outline_revised"]
        assert out["outline_feedback"] == ["반전을 더 일찍 드러내 줘"]
        assert "plot_locked" not in out and not _state(tmp_path, story_id)["plot_locked"]
        assert "말씀하신 대로 개요를 고쳐 저장했습니다" in out[REPLY_NOTE_KEY]["ko"]
        asked = _last(model, REVISE_MARK)
        assert "반전을 더 일찍 드러내 줘" in asked and "안개가 오는 밤" in asked
        assert '"하린"' in asked  # the cast by name, not by id
        outline = _work(tmp_path, story_id).outline()
        assert outline is not None
        assert outline[0].chapters[0].title == "첫날 밤의 목소리"
        # The revision went through the arc check, and nothing was written or proposed.
        assert out["outline_check"]["revised"] == 1
        assert not any("Write scene" in p for p in model.prompts)
        assert not any(KG_MARK in p for p in model.prompts)
        assert len(_work(tmp_path, story_id).proposals()[0]) == 3

        settled = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert settled["stage"] == "review_chapter" and settled["plot_locked"] is True
        assert settled["details_chapter"] == 1
        assert _work(tmp_path, story_id).written_scenes() == []

    async def test_an_unreadable_revision_keeps_the_outline_and_says_so(
        self, tmp_path: Path
    ) -> None:
        model = ScriptedModel(revised="개요를 고쳤습니다!")
        story_id, ctx = await self._at_outline(tmp_path, model)
        before = _work(tmp_path, story_id).outline()
        out = await _ok(_tool(model), ctx, action="continue", changes="3장 빼 줘")
        assert out["stage"] == "review_outline"
        assert "반영하지 못해 개요를 그대로 두었습니다" in out[REPLY_NOTE_KEY]["ko"]
        assert _work(tmp_path, story_id).outline() == before

    async def test_yes_settles_the_plot_and_proposes_its_key_events(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_details.py :: wanted += read
        Becomes: pass
        """
        model = ScriptedModel()
        story_id, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert out["done_now"][0] == "plot"
        assert _state(tmp_path, story_id)["plot_locked"] is True
        assert [k["name"] for k in out["plot_kg"]] == ["중심 갈등", "반전", "등대의 약속", "등불"]
        assert "줄거리를 확정했습니다" in out[REPLY_NOTE_KEY]["ko"]
        assert "설정집에 넣을 항목 4개" in out[REPLY_NOTE_KEY]["ko"]
        proposals = {p.id: p for p, _ in _work(tmp_path, story_id).proposals()[0]}
        entries: dict[str, dict[str, Any]] = {}
        for k in out["plot_kg"]:
            proposal = proposals[k["proposal"]]
            assert proposal.status == "pending"  # nothing approves itself (§5.3)
            entries[k["name"]] = proposal.new_entry().model_dump()
        assert proposals[out["plot_kg"][3]["proposal"]].kind == "items"
        conflict = entries["중심 갈등"]
        assert conflict["profile"] == _SPINE["central_conflict"]
        assert conflict["planted_in"] == "ch01.s01"
        assert conflict["notes"] == _SPINE["resolution"]
        assert entries["반전"]["profile"] == _SPINE["twist"]
        assert entries["반전"]["notes"] == _SPINE["clue"]
        promise = entries["등대의 약속"]
        assert promise["planted_in"] == "ch01.s01" and promise["pay_off_by"] == "ch02.s01"
        cast_ids = {c["name"]: c["id"] for c in _state(tmp_path, story_id)["cast"]}
        # A name not in the cast is left out, not invented.
        assert promise["state"] == {"involves": [cast_ids["하린"], cast_ids["먹물"]]}
        # The chapter is written after, over the same outline.
        assert out["stage"] == "review_chapter"

    async def test_words_that_settle_it_wait_for_the_cast_and_do_not_settle_twice(
        self, tmp_path: Path
    ) -> None:
        model = ScriptedModel()
        _, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue")
        assert out["stage"] == "review_outline" and len(out["cast_waiting"]) == 3
        assert out["plot_locked"] is True
        assert out[REPLY_NOTE_KEY]["ko"].startswith("줄거리를 확정했습니다.")
        again = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert again["stage"] == "review_chapter" and "plot" not in again["done_now"]
        assert "줄거리를 확정했습니다" not in again[REPLY_NOTE_KEY]["ko"]
        assert sum(KG_MARK in p for p in model.prompts) == 1

    async def test_an_unreadable_reading_proposes_the_conflict_and_twist_alone(
        self, tmp_path: Path
    ) -> None:
        model = ScriptedModel(kg="모르겠습니다")
        _, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert [k["name"] for k in out["plot_kg"]] == ["중심 갈등", "반전"]
        assert "The settled outline's key events could not be read." in out["notes"]


class TestTheNextChapterIsPlannedBeforeItIsWritten:
    """Stage 4b: after a chapter is written and checked, the next one's plan waits."""

    async def _after_first(
        self, tmp_path: Path, model: ScriptedModel
    ) -> tuple[dict[str, Any], str, ToolContext]:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)
        await _ok(_tool(model), ctx, action="continue", choice=1)
        await _ok(_tool(model), ctx, action="continue", go_on=True)
        await _ok(_tool(model), ctx, action="continue", go_on=True)  # the first plan
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        return out, story_id, ctx

    async def test_the_next_chapter_is_planned_and_shown_before_it_is_written(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: await self._propose_details(number + 1)
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/start_details.py :: before = self._last_text(outline, number - 1)
        Becomes: before = ""
        """
        model = ScriptedModel()
        out, story_id, _ = await self._after_first(tmp_path, model)
        assert out["stage"] == "review_chapter"
        assert out["done_now"][-3:] == ["chapter", "continuity", "details"]
        assert out["chapter_written"] == 1 and out["details_chapter"] == 2
        plan = out["chapter_details"]
        assert [s["title"] for s in plan] == ["젖은 발자국", "걷히는 안개"]
        assert plan[0]["characters"] == ["하린", "먹물"]  # a name not in the cast is dropped
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "첫 장을 썼습니다" in ko and "다음은 2장의 장면 계획입니다" in ko
        assert "1. 젖은 발자국 — 하린이 계단의 발자국을 따라간다." in ko
        assert "‘네’라고 하시면 이대로 2장을 쓰겠습니다" in ko
        # The plan is made from the settled chapter and the end of the one before it.
        prompt = _last(model, DETAILS_MARK)
        assert "새벽" in prompt and "안개가 걷힌다." in prompt
        assert "How chapter 1 ended" in prompt and _SCENE[-40:] in prompt
        # Nothing of chapter 2 is written, and the outline is as settled until it is.
        work = _work(tmp_path, story_id)
        assert work.written_scenes() == ["ch01.s01", "ch01.s02"]
        outline, _ = work.require_outline()
        assert [s.title for s in outline.chapters[1].scenes] == ["끝"]

    async def test_a_clause_given_as_a_title_is_shortened_and_not_repeated(
        self, tmp_path: Path
    ) -> None:
        """A scene title the model gave as the opening clause of its summary (a live
        qwen3:8b run showed one) becomes a short title, and the plan shows the summary once.

        Killed by: examples/story/src/uclone_x_story/start_outline.py :: title = scene_title(_text(sc.get("title")), _text(sc.get("summary")))
        Becomes: title = _text(sc.get("title")) or _text(sc.get("summary"))[:30]
        Killed by: examples/story/src/uclone_x_story/start_notes.py :: + (f" — {s['summary']}" if s.get("summary") and s["summary"] != s.get("title") else "")
        Becomes: + (f" — {s['summary']}" if s.get("summary") else "")
        """
        clause = "하린은 계단에서 등대지기의 발소리를 듣고 있음을 느끼며,"
        details = {
            "scenes": [
                {"title": clause, "summary": f"{clause} 먹물의 방울을 찾아 나선다."},
                {"title": "걷히는 안개", "summary": "걷히는 안개"},
            ]
        }
        model = ScriptedModel(details=details)
        out, _, _ = await self._after_first(tmp_path, model)
        assert [s["title"] for s in out["chapter_details"]] == ["등대지기의 발소리", "걷히는 안개"]
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert f"1. 등대지기의 발소리 — {clause} 먹물의 방울을 찾아 나선다." in ko
        assert f"1. {clause}" not in ko
        assert "2. 걷히는 안개\n" in ko  # a summary that only repeats the title is not shown

    async def test_feedback_revises_the_plan_and_stops_again(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if not self.state.quick:  # the revised plan is shown first
        Becomes: if False:  # the revised plan is shown first
        """
        model = ScriptedModel()
        _, story_id, ctx = await self._after_first(tmp_path, model)
        feedback = "첫 장면은 빼고 방울 소리를 따라가는 장면으로 바꿔 줘"
        out = await _ok(_tool(model), ctx, action="continue", changes=feedback)
        assert out["stage"] == "review_chapter"
        assert out["done_now"] == ["details_revised"]
        assert [s["title"] for s in out["chapter_details"]] == ["방울 소리"]
        assert out["details_feedback"] == [feedback]
        assert feedback in _last(model, DETAILS_REVISE_MARK)
        assert "젖은 발자국" in _last(model, DETAILS_REVISE_MARK)  # the plan as it stood
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "말씀하신 대로 2장의 장면 계획을 고쳤습니다" in ko and "1. 방울 소리" in ko
        assert "썼습니다" not in ko
        assert _work(tmp_path, story_id).written_scenes() == ["ch01.s01", "ch01.s02"]

    async def test_an_unreadable_revision_keeps_the_plan_and_says_so(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_details.py :: details_revision_failed=True,
        Becomes: details_revision_failed=False,
        """
        model = ScriptedModel(details_revised="고칠 수 없습니다")
        _, _, ctx = await self._after_first(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", changes="더 무섭게 해 줘")
        assert [s["title"] for s in out["chapter_details"]] == ["젖은 발자국", "걷히는 안개"]
        assert "반영하지 못해 2장의 장면 계획을 그대로 두었습니다" in out[REPLY_NOTE_KEY]["ko"]

    async def test_yes_writes_the_planned_chapter_and_the_last_one_ends_the_flow(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: await self._apply_details(number)
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/start_scene.py :: if number < len(outline.chapters):
        Becomes: if True:
        """
        model = ScriptedModel()
        _, story_id, ctx = await self._after_first(tmp_path, model)
        await _ok(_tool(model), ctx, action="continue", changes="첫 장면은 빼 줘")
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert out["stage"] == "done"
        assert out["done_now"] == ["chapter", "continuity"]
        work = _work(tmp_path, story_id)
        outline, _ = work.require_outline()
        # Chapter 2 keeps its id and title; its scenes are the plan the person agreed to.
        assert outline.chapters[1].id == "ch02" and outline.chapters[1].title == "새벽"
        assert [(s.id, s.title) for s in outline.chapters[1].scenes] == [("ch02.s01", "방울 소리")]
        assert outline.chapters[1].scenes[0].characters == ["harin"]
        assert work.written_scenes() == ["ch01.s01", "ch01.s02", "ch02.s01"]
        written = _last(model, "Write scene")
        assert "방울 소리" in written and _SCENE[-20:] in written  # after chapter 1's end
        assert out["chapter_written"] == 2 and out["chapter_scenes"] == ["ch02.s01"]
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "2장을 썼습니다. 장면 1개를 저장했습니다." in ko
        assert "마지막 장까지 모두 썼습니다" in ko
        assert "details_chapter" not in out

    async def test_an_unreadable_plan_shows_the_outlines_scenes(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_details.py :: from_outline = not scenes
        Becomes: from_outline = False
        """
        model = ScriptedModel(details="계획을 세울 수 없습니다")
        out, _, _ = await self._after_first(tmp_path, model)
        assert out["stage"] == "review_chapter"
        assert [s["title"] for s in out["chapter_details"]] == ["끝"]
        assert "개요에 있는 장면을 그대로 보여 드립니다" in out[REPLY_NOTE_KEY]["ko"]

    async def test_asked_at_once_it_writes_one_chapter_per_call_and_says_how_to_go_on(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: and done.count("chapter") < CHAPTERS_PER_CALL
        Becomes: and True
        """
        model = ScriptedModel()
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        story_id = out["open_story_id"]
        assert CHAPTERS_PER_CALL == 1
        assert out["done_now"].count("chapter") == 1 and out["stage"] == "review_chapter"
        assert _work(tmp_path, story_id).written_scenes() == ["ch01.s01", "ch01.s02"]
        assert "‘계속’이라고 하시면 2장을 이어 쓰겠습니다" in out[REPLY_NOTE_KEY]["ko"]
        # In quick mode feedback revises the plan and writes it in the same call.
        again = await _ok(
            _tool(model),
            _ctx(tmp_path, story=story_id),
            action="continue",
            changes="첫 장면은 빼 줘",
        )
        assert again["done_now"] == ["details_revised", "chapter", "continuity"]
        assert again["stage"] == "done"
        assert _work(tmp_path, story_id).written_scenes() == ["ch01.s01", "ch01.s02", "ch02.s01"]


class TestEachWrittenSceneIsHeldToItsPlan:
    """§1.1 row 6: after a chapter is written, each scene is held to its plan and the twist."""

    async def _quick(self, tmp_path: Path, model: ScriptedModel) -> dict[str, Any]:
        return await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )

    async def test_a_missing_beat_is_shown_and_yes_still_goes_on(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: kept += [f.record() for f in held]
        Becomes: kept += []
        Killed by: examples/story/src/uclone_x_story/start_scene.py :: return findings + plan_findings(scene, text, reply, **where), True
        Becomes: return findings, True
        """
        model = ScriptedModel(plans=[{"beats": [{"beat": 2, "done": False}]}])
        out = await self._quick(tmp_path, model)
        [finding] = out["continuity"]
        assert finding["scene_id"] == "ch01.s01"
        assert "“누군가 이름을 부른다” 부분이 쓴 장면에 보이지 않습니다" in finding["note"]
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "장면 계획에 대조해 보니 맞지 않는 곳이 1곳" in ko and finding["note"] in ko
        assert "ch01" not in ko and "plan_beat_missing" not in ko
        # The scene was read against its plan once, at temperature 0, and never rewritten.
        assert sum(p.startswith(PLAN_MARKER) for p in model.prompts) == 2
        assert len([p for p in model.prompts if "Write scene" in p]) == 2
        # "네" takes the next chapter's plan; the finding does not hold the flow.
        again = await _ok(
            _tool(model),
            _ctx(tmp_path, story=out["open_story_id"]),
            action="continue",
            go_on=True,
        )
        assert "chapter" in again["done_now"] and "rewrite" not in again["done_now"]

    async def test_the_twist_is_asked_about_only_before_its_chapter(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: "twist_chapter": arc.twist_chapter if arc else None,
        Becomes: "twist_chapter": None,
        """
        model = ScriptedModel()
        out = await self._quick(tmp_path, model)
        twist_chapter = _state(tmp_path, out["open_story_id"])["arc"]["twist_chapter"]
        assert twist_chapter > 1
        assert _SPINE["twist"] in _last(model, PLAN_MARKER)

    async def test_a_planned_character_the_scene_never_names_is_shown(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: findings = absent_cast(codex, scene, text)
        Becomes: findings = []
        """
        alone = "하린은 등대 계단을 올랐다. 바람이 창을 두드렸다. 꼭대기에서 누군가 이름을 불렀다."
        model = ScriptedModel(scenes=[alone])
        out = await self._quick(tmp_path, model)
        notes = [f["note"] for f in out["continuity"]]
        assert "‘계단’ 장면 계획에 있던 먹물이 쓴 장면에는 나오지 않습니다." in notes

    async def test_an_unreadable_reading_is_never_called_consistent(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: not_held.append(scene_id)
        Becomes: pass
        """
        model = ScriptedModel(plans=["모르겠습니다", "모르겠습니다"])
        out = await self._quick(tmp_path, model)
        assert out["plan_unread"] == ["ch01.s01", "ch01.s02"]
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "장면 2개는 장면 계획과 대조하지 못했습니다" in ko
        assert "반전을 드러낼 때와도 어긋나는 곳을 찾지 못했습니다" not in ko

    async def test_a_scene_that_keeps_its_plan_is_said_to(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: if not shown and scenes and len(plan_unread) < len(scenes):
        Becomes: if False:
        """
        model = ScriptedModel()
        out = await self._quick(tmp_path, model)
        assert out["continuity"] == [] and out["plan_unread"] == []
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert (
            "장면 계획에 있던 인물과 사건, 반전을 드러낼 때와도 어긋나는 곳을 찾지 못했습니다" in ko
        )


# A trimmed, paraphrased case from a live yes-only run (qwen3:8b): a mechanic takes up a
# broken clock in a grave and falls there, then walks the village in the next scene, while
# the village elder holds the same clock. The run's check told the person nothing.
_CLOCK_CAST = json.dumps(
    {
        "characters": [
            {
                "name": "이진호",
                "profile": "시계를 쫓는 기계공.",
                "gender": "male",
                "status": "alive",
            },
            {
                "name": "마을 장로",
                "profile": "의식을 이끈 장로.",
                "gender": "male",
                "status": "alive",
            },
        ],
        "places": [{"name": "무명 마을", "profile": "산속의 마을."}],
    },
    ensure_ascii=False,
)
_CLOCK_KG: dict[str, Any] = {
    "threads": [],
    "items": [{"name": "조각난 시계", "profile": "무덤에서 나온 깨진 시계."}],
}
_FALLS = "이진호의 숨은 그 자리에서 멎었다"
_GRAVE = f"이진호는 무덤 아래로 내려가 깨진 시계를 집어 들었다. {_FALLS}."
_WALKS = "이진호는 마을 한가운데로 걸어갔다"
_CARRIES = "그는 시계를 품에 넣고 다녔다"
_HOLDS = "마을 장로는 그 시계를 손에 쥔 채 앉아 있었다"
_VILLAGE = f"{_WALKS}. {_CARRIES}. 장로의 집에서 {_HOLDS}."


def _said(subject: str, key: str, value: str, quote: str, **more: str) -> dict[str, Any]:
    return {
        **more,
        "subject": subject,
        "key": key,
        "value": value,
        "quote": quote,
        "time": "now",
        "polarity": "asserted",
    }


class TestAWrittenDeathAndAHeldItemCarryAcrossScenes:
    """§1.1 row 6: a scene is held to what earlier scenes wrote, and one item has one owner."""

    async def _clock(self, tmp_path: Path, village_facts: list[dict[str, Any]]) -> dict[str, Any]:
        model = ScriptedModel(
            cast_replies=[_CLOCK_CAST],
            kg=_CLOCK_KG,
            scenes=[_GRAVE, _VILLAGE],
            facts=[
                [
                    _said("이진호", "status", "alive", "이진호는 무덤 아래로 내려가"),
                    _said("이진호", "status", "dead", _FALLS),
                ],
                village_facts,
            ],
        )
        return await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )

    async def test_the_live_case_is_flagged_in_the_scene_that_breaks_it(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: kept += [f.record() for f in findings]
        Becomes: kept += []
        Killed by: examples/story/src/uclone_x_story/continuity.py :: return f"이 장면에서 {both[0]} 나오는데, {both[1]}도 나옵니다."
        Becomes: return "이 문장이 설정집의 기록과 맞지 않습니다."
        """
        out = await self._clock(
            tmp_path,
            [
                _said("이진호", "status", "alive", _WALKS),
                # "시계", the words naming the clock, as the reading is asked for them (#2249).
                _said("이진호", "possesses", "조각난 시계", _CARRIES, object_named="시계"),
                _said("마을 장로", "possesses", "조각난 시계", _HOLDS, object_named="시계"),
            ],
        )
        village = [f for f in out["continuity"] if f["scene_id"] == "ch01.s02"]
        notes = [f["note"] for f in village]
        assert (
            f"‘목소리’ 장면의 “{_WALKS}” — 이진호는 앞의 ‘계단’ 장면에서 죽었는데, "
            "이 장면에서 다시 움직입니다."
        ) in notes
        [owners] = [f for f in village if f["kind"] != "dead_then_acting"]
        said = owners["note"].split(" — ", 1)[1]
        assert said.startswith("이 장면에서 ") and said.endswith(" 가진 것으로도 나옵니다.")
        assert "이진호가 ‘조각난 시계’를" in said and "마을 장로가 ‘조각난 시계’를" in said
        for note in notes:
            assert "jogak" not in note and "ch01" not in note and "설정집에는" not in note
        # No state of its own beside the store (#2207).
        assert "written_deaths" not in _state(tmp_path, out["open_story_id"])

    async def test_a_death_that_holds_is_a_proposal_and_not_flagged(self, tmp_path: Path) -> None:
        """The death of a cast member still waiting for approval is proposed, quoting the
        digest the cast's file will have once added, and it applies after the cast does.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: return digest_once_added(adding.new_entry()) if adding is not None else None
        Becomes: return None
        """
        out = await self._clock(tmp_path, [_said("마을 장로", "status", "alive", _HOLDS)])
        village = [f for f in out["continuity"] if f["scene_id"] == "ch01.s02"]
        assert village == []
        work = _work(tmp_path, out["open_story_id"])
        [death] = [
            p
            for p, _ in work.proposals()[0]
            if p.change.progression is not None and p.change.progression.set.get("status")
        ]
        assert death.change.progression is not None
        assert death.change.progression.at == "ch01.s01"
        assert death.change.progression.set == {"status": "dead"}
        assert [e.quote for e in death.evidence] == [_FALLS]
        cast = next(
            p for p, _ in work.proposals()[0] if p.change.new_entry and p.entry_id == death.entry_id
        )
        # Approved in turn, the cast and then the death: both apply.
        apply_proposal(work, cast.id, room_id=ROOM, decided_in="story_view")
        apply_proposal(work, death.id, room_id=ROOM, decided_in="story_view")
        loaded = work.entry("characters", death.entry_id)
        assert loaded is not None and loaded[0].progressions[0].set == {"status": "dead"}


async def _to_outline(tmp_path: Path, model: ScriptedModel) -> tuple[str, ToolContext]:
    """A story started, premise 1 chosen and the outline shown: the plot not yet settled."""
    out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
    story_id = out["open_story_id"]
    ctx = _ctx(tmp_path, story=story_id)
    await _ok(_tool(model), ctx, action="continue", choice=1)
    await _ok(_tool(model), ctx, action="continue", go_on=True)
    return story_id, ctx


class TestTheFirstChapterIsPlannedBeforeItIsWritten:
    """§1.1: the first chapter's plan stops for the person, as every later chapter's does."""

    async def test_settling_the_plot_shows_the_first_plan_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: await self._propose_details(1)
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/start_details.py :: "The first scene opens the story."
        Becomes: ""
        """
        model = ScriptedModel(details=_DETAILS)
        story_id, ctx = await _to_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert out["stage"] == "review_chapter" and out["details_chapter"] == 1
        assert out["done_now"] == ["plot", "details"]
        assert [s["title"] for s in out["chapter_details"]] == ["젖은 발자국", "걷히는 안개"]
        assert _work(tmp_path, story_id).written_scenes() == []
        prompt = _last(model, DETAILS_MARK)
        assert f"{DETAILS_MARK} 1," in prompt
        assert "The first scene opens the story." in prompt
        assert "How chapter 0 ended" not in prompt
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "1장의 장면 계획" in ko and "1. 젖은 발자국" in ko
        assert "p0" not in ko and "ch01" not in ko

    async def test_feedback_on_the_first_plan_revises_it_and_yes_writes_it(
        self, tmp_path: Path
    ) -> None:
        model = ScriptedModel(details=_DETAILS)
        story_id, ctx = await _to_outline(tmp_path, model)
        await _ok(_tool(model), ctx, action="continue", go_on=True)
        feedback = "방울 소리를 따라가는 한 장면으로 줄여 줘"
        revised = await _ok(_tool(model), ctx, action="continue", changes=feedback)
        assert revised["done_now"] == ["details_revised"] and revised["details_chapter"] == 1
        assert [s["title"] for s in revised["chapter_details"]] == ["방울 소리"]
        assert _work(tmp_path, story_id).written_scenes() == []
        written = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert written["chapter_written"] == 1
        work = _work(tmp_path, story_id)
        assert work.written_scenes() == ["ch01.s01"]
        outline, _ = work.require_outline()
        assert [s.title for s in outline.chapters[0].scenes] == ["방울 소리"]


class TestTheSceneCountFollowsTheArc:
    """§1.1: how many scenes a chapter has is decided by code from its arc functions."""

    def test_a_turning_chapter_is_longer_and_a_setup_chapter_shorter(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: if functions & _HEAVY:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/start_state.py :: if functions and functions <= _LIGHT:
        Becomes: if False:
        """
        arc = Arc(
            functions=(
                ("setup",),
                ("complication", "escalation"),
                ("reversal",),
                ("climax", "resolution"),
                ("resolution",),
            ),
            twist_chapter=3,
            climax_chapter=4,
        )
        assert [scenes_for(arc, n) for n in range(1, 6)] == [2, 3, 4, 4, 2]
        assert scenes_for(None, 1) == 3
        assert scenes_for(arc, 0) == 3 and scenes_for(arc, 6) == 3

    async def test_the_outline_and_the_plan_are_told_and_clipped_to_the_count(
        self, tmp_path: Path
    ) -> None:
        """The default test arc makes chapter 1 a setup (2 scenes) and chapter 2 a
        complication and reversal (4 scenes).

        Killed by: examples/story/src/uclone_x_story/start_outline.py :: "scenes": _items(chapter.get("scenes"))[: scenes_for(arc, number)],
        Becomes: "scenes": _items(chapter.get("scenes")),
        Killed by: examples/story/src/uclone_x_story/start_details.py :: count = scenes_for(_arc_of(self.state.arc), number)
        Becomes: count = MAX_SCENES_PER_CHAPTER
        Killed by: examples/story/src/uclone_x_story/start_details.py :: for scene in _scene_dicts(found.get("scenes"), "plan", by_name)[:limit]
        Becomes: for scene in _scene_dicts(found.get("scenes"), "plan", by_name)
        """
        four = [{"title": f"장면 {i}", "summary": f"하린이 {i}층에 오른다."} for i in range(1, 5)]
        model = ScriptedModel(
            chapters={1: [{**_DEFAULT_CHAPTERS[1], "scenes": four}]},
            details={"scenes": four},
        )
        story_id, ctx = await _to_outline(tmp_path, model)
        asked = [p for p in model.prompts if CHAPTER_PROMPT_MARK in p]
        assert "It has 2 scenes" in asked[0] and "It has 4 scenes" in asked[1]
        outline, _ = _work(tmp_path, story_id).require_outline()
        assert [s.title for s in outline.chapters[0].scenes] == ["장면 1", "장면 2"]
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert "It has 2 scenes" in _last(model, DETAILS_MARK)
        assert [s["title"] for s in out["chapter_details"]] == ["장면 1", "장면 2"]


class TestPlotItemsAreThingsNotIdeas:
    """§1.1: the settled plot proposes items a character can hold, not motifs."""

    def test_an_idea_is_told_from_an_object(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_details.py :: return any(len(w) >= 2 and last.endswith(w) for w in ABSTRACT_ITEM_WORDS if not w.isascii())
        Becomes: return False
        Killed by: examples/story/src/uclone_x_story/start_details.py :: if whole in ABSTRACT_ITEM_WORDS or last in ABSTRACT_ITEM_WORDS:
        Becomes: if whole in ABSTRACT_ITEM_WORDS:
        """
        for idea in ("기억", "아버지의 비밀", "잃어버린기억", "a promise", "Hope"):
            assert abstract_item(idea), idea
        for thing in ("등불", "the locket", "물 용기", "promised ring", "열쇠", ""):
            assert not abstract_item(thing), thing

    async def test_an_idea_named_as_an_item_is_dropped_and_counted(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_details.py :: dropped = len(abstract)
        Becomes: dropped = 0
        Killed by: examples/story/src/uclone_x_story/start_details.py :: if kind == "items" and abstract_item(name):
        Becomes: if False:
        """
        kg = {
            "threads": [],
            "items": [
                {"name": "기억", "profile": "하린이 잊은 밤."},
                {"name": "등불", "profile": "하린이 드는 오래된 등불."},
            ],
        }
        model = ScriptedModel(kg=kg)
        story_id, ctx = await _to_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert [k["name"] for k in out["plot_kg"]] == ["중심 갈등", "반전", "등불"]
        assert out["plot_kg_dropped"] == 1
        assert _state(tmp_path, story_id)["plot_kg_dropped"] == 1
        assert any("named an idea, not an object" in n and "기억" in n for n in out["notes"])
        assert "physical objects" in _last(model, KG_MARK)
        names = [p.new_entry().name for p, _ in _work(tmp_path, story_id).proposals()[0]]
        assert "기억" not in names and "등불" in names


_FATHER_PLAN: dict[str, Any] = {
    "scenes": _DETAILS["scenes"],
    "new_characters": [
        {
            "name": "도윤",
            "profile": "하린의 아버지, 오래전에 섬을 떠난 어부.",
            "gender": "male",
            "family": [
                {"relative": "하린", "is": "daughter"},
                {"relative": "없는 사람", "is": "son"},
            ],
        },
        {"name": "뱃사람", "profile": "항구의 뱃사람.", "family": []},
    ],
}


class TestANewCharacterInThePlanIsProposedWithFamily:
    """§1.1 row 7: a plan that brings in a relative of the cast proposes both sides."""

    async def _plan(self, tmp_path: Path, model: ScriptedModel) -> tuple[dict[str, Any], str]:
        story_id, ctx = await _to_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        return out, story_id

    async def test_the_plan_names_the_new_character_and_writing_proposes_both_sides(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_cast.py :: relations = {f["relative_id"]: INVERSE_ROLE[f["role"]] for f in family}
        Becomes: relations = {f["relative_id"]: f["role"] for f in family}
        Killed by: examples/story/src/uclone_x_story/start.py :: if self._introduce(number):
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/start_cast.py :: if k.get("change") == "new_entry":
        Becomes: if False:
        """
        model = ScriptedModel(details=_FATHER_PLAN)
        out, story_id = await self._plan(tmp_path, model)
        work = _work(tmp_path, story_id)
        [new] = out["details_new"]  # one with no family among the cast is not kept
        assert new["name"] == "도윤" and new["gender"] == "male"
        assert [(f["relative"], f["role"]) for f in new["family"]] == [("하린", "child")]
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "이 장에서 새로 나오는 인물은 도윤입니다" in ko
        assert "new_characters" in _last(model, DETAILS_MARK)
        # The person approves 하린, then accepts the plan.
        [harin] = [p for p, _ in work.proposals()[0] if p.entry_id == "harin"]
        apply_proposal(work, harin.id, room_id=ROOM, decided_in="story_view")
        ctx = _ctx(tmp_path, story=story_id)
        written = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert written["chapter_written"] == 1
        assert "introduced" in written["done_now"]
        by_id = {p.id: p for p, _ in work.proposals()[0]}
        [entry] = [k for k in written["introduced"] if k["change"] == "new_entry"]
        [kin] = [k for k in written["introduced"] if k["change"] == "relation"]
        father = by_id[entry["proposal"]]
        assert father.status == "pending"
        assert father.new_entry().name == "도윤"
        assert father.new_entry().state == {"status": "alive"}
        assert father.new_entry().relations == {"harin": "parent"}
        back = by_id[kin["proposal"]]
        assert back.status == "pending" and back.entry_id == "harin"
        assert back.change.progression is not None
        assert back.change.progression.at == "ch01.s01"
        assert back.change.progression.relations == {father.entry_id: "child"}
        assert back.change.progression.set == {}
        assert back.entry_digest == work.entry("characters", "harin")[1]  # type: ignore[index]
        ko = written[REPLY_NOTE_KEY]["ko"]
        assert "계획에 나온 새 인물(도윤)과 그 가족 관계를 설정집에 제안했습니다" in ko
        assert father.id not in ko and "relation" not in ko
        # The next plan knows 도윤 now: it does not bring him in again.
        assert written["details_chapter"] == 2 and "details_new" not in written

    async def test_a_pending_relative_gets_the_relation_on_the_new_side_only(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_cast.py :: if loaded is None:  # a relative still pending: this side only
        Becomes: if False:  # a relative still pending: this side only
        """
        model = ScriptedModel(details=_FATHER_PLAN)
        _, story_id = await self._plan(tmp_path, model)
        ctx = _ctx(tmp_path, story=story_id)
        written = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert [k["change"] for k in written["introduced"]] == ["new_entry"]
        work = _work(tmp_path, story_id)
        [father] = [
            p for p, _ in work.proposals()[0] if p.id == written["introduced"][0]["proposal"]
        ]
        assert father.new_entry().relations == {"harin": "parent"}
        assert all(p.status == "pending" for p, _ in work.proposals()[0])
        assert not work.codex().items


#: What a person says at the premise stop when none of the three will do (#2262).
_TURNED_DOWN = "셋 다 별로야. 우주 정거장에 혼자 남은 정비사가 나오는 SF로 해 줘."


class TestTheFlowKeepsItsStops:
    """#2262: a live Writer left the flow after its first stop and wrote the story by hand,
    and turning every premise down derailed it; the outline stop hid the outline."""

    async def _at(self, tmp_path: Path, stage: str) -> tuple[ToolContext, str]:
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)
        while out["stage"] != stage:
            out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        return ctx, story_id

    async def test_the_outline_and_a_scene_by_hand_are_refused_while_the_start_waits(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/tools.py :: if waiting is not None:
        Becomes: if False:
        """
        ctx, story_id = await self._at(tmp_path, "choose_premise")
        before = _state(tmp_path, story_id)
        outline = await StoryOutlineTool().execute(
            {"action": "init", "chapter_titles": ["안개"]}, ctx
        )
        scene = await StoryManuscriptTool(model_factory=lambda _c: None).execute(
            {"action": "write", "scene_id": "ch01.s01", "text": _SCENE}, ctx
        )
        for refused in (outline, scene):
            assert not refused.success
            assert refused.error is not None
            assert "story_start 'continue'" in refused.error
            assert "the three premises it offered" in refused.error
        work = _work(tmp_path, story_id)
        assert work.outline() is None and work.written_scenes() == []
        assert _state(tmp_path, story_id) == before

    async def test_at_a_chapters_plan_its_new_scenes_are_the_starts_to_write(
        self, tmp_path: Path
    ) -> None:
        """#2300: at the first chapter's plan a live model wrote ch01.s01 by hand, and
        story_start could no longer write the chapter.

        Killed by: examples/story/src/uclone_x_story/start_notes.py :: return start_waiting(work)
        Becomes: return None
        """
        ctx, story_id = await self._at(tmp_path, "review_chapter")
        outline = await StoryOutlineTool().execute(
            {"action": "set_scene", "scene_id": "ch01.s01", "summary": "하린이 계단을 오른다."},
            ctx,
        )
        scene = await StoryManuscriptTool(model_factory=lambda _c: None).execute(
            {"action": "write", "scene_id": "ch01.s01", "text": _SCENE}, ctx
        )
        for refused in (outline, scene):
            assert not refused.success and refused.error is not None
            assert "the chapter's scene plan it showed" in refused.error
        assert _work(tmp_path, story_id).written_scenes() == []
        written = await _ok(_tool(ScriptedModel()), ctx, action="continue", go_on=True)
        assert written["chapter_written"] == 1

    async def test_a_written_scene_is_not_overwritten_by_hand_while_the_start_waits(
        self, tmp_path: Path
    ) -> None:
        """#2361: a live Writer, asked to rewrite a written scene at a chapter's plan, wrote
        its new text by hand into the next chapter's first scene. While the start waits, the
        flow rewrites a written scene itself (`changes`), so a by-hand write over one is
        refused and names that call.

        Killed by: examples/story/src/uclone_x_story/tools.py :: written = scene_id is not None and work.manuscript(scene_id) is not None
        Becomes: written = False
        """
        ctx, story_id = await self._at(tmp_path, "review_chapter")
        await _ok(_tool(ScriptedModel()), ctx, action="continue", go_on=True)
        work = _work(tmp_path, story_id)
        found = work.manuscript("ch01.s01")
        assert found is not None and _state(tmp_path, story_id)["stage"] == "review_chapter"

        edited = await StoryManuscriptTool(model_factory=lambda _c: None).execute(
            {
                "action": "write",
                "scene_id": "ch01.s01",
                "text": _SCENE + " 끝.",
                "digest": found.digest,
            },
            ctx,
        )

        assert not edited.success and edited.error is not None
        assert "a scene already written is rewritten by it, not by hand" in edited.error
        assert "story_start 'continue'" in edited.error and "'changes'" in edited.error
        assert work.manuscript("ch01.s01") == found

    async def test_no_second_story_is_made_by_hand_while_the_start_waits(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/tool.py :: if waiting is not None:
        Becomes: if False:
        """
        ctx, story_id = await self._at(tmp_path, "review_cast")
        made = await StoryLibraryTool().execute({"action": "create", "title": "우주 정비사"}, ctx)
        assert not made.success and made.error is not None
        assert "story_start 'start'" in made.error and "story_start 'continue'" in made.error
        stories = [p.name for p in (tmp_path / "stories").iterdir() if p.is_dir()]
        assert stories == [story_id]

    async def test_turning_every_premise_down_draws_three_new_ones_from_the_words(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: changes = (params.changes or "").strip() or ", ".join(wanted["avoid"])
        Becomes: changes = ""
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)
        drawn = len([p for p in model.prompts if "Write one premise" in p])

        again = await _ok(
            _tool(model),
            ctx,
            action="continue",
            changes=_TURNED_DOWN,
            genre="SF",
            must_have=["우주 정거장에 혼자 남은 정비사"],
        )

        assert again["stage"] == "choose_premise" and "chosen" not in again
        assert len([p for p in model.prompts if "Write one premise" in p]) == drawn + 3
        assert again["brief"]["genre"] == "SF"
        assert _story_genre(tmp_path, story_id) == "SF"
        assert "우주 정거장에 혼자 남은 정비사" in again["brief"]["must_have"]
        assert "고양이" in again["brief"]["must_have"]  # the request's own stays
        # Nothing past the premises was made, and the person is shown the new three.
        assert _work(tmp_path, story_id).proposals()[0] == []
        assert again[REPLY_NOTE_KEY]["ko"].startswith("말씀하신 내용으로 발상을 새로 만들었습니다.")
        for premise in again["premises"]:
            assert premise["title"] in again[REPLY_NOTE_KEY]["ko"]

    async def test_the_next_draw_reads_the_words_and_the_premises_turned_down(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_premise.py :: f"{_turned_down_lines(state)}"
        Becomes: ""
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_tool(model), ctx, action="continue", changes=_TURNED_DOWN)
        prompt = _last(model, "Write one premise")
        assert _TURNED_DOWN in prompt
        for title, _ in _PREMISES:
            assert title in prompt.split("turned down the premises offered before")[1]

    async def test_the_outline_stop_shows_the_outline_in_the_reply(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: shown = _outline_lines(outline)
        Becomes: shown = ""
        """
        ctx, story_id = await self._at(tmp_path, "review_outline")
        out = await _ok(_tool(ScriptedModel()), ctx, action="status")
        outline, _ = _work(tmp_path, story_id).require_outline()
        for note in (out[REPLY_NOTE_KEY]["ko"], out[REPLY_NOTE_KEY]["en"]):
            for chapter in outline.chapters:
                assert chapter.title in note
                assert all(scene.title in note for scene in chapter.scenes)
        assert "이야기 보기에서 확인할 수 있습니다" not in out[REPLY_NOTE_KEY]["ko"]


class TestAnEnglishRequestGetsAnEnglishStory:
    """#2263: an English speaker was offered "**호러**" and Korean premise titles, because the
    genre labels and the brief's defaults were Korean only."""

    _ENGLISH = "Write me a story."
    _BRIEF: dict[str, Any] = {"title": None, "genre": None, "must_have": [], "avoid": []}

    async def test_the_genre_proposal_is_worded_from_the_english_catalog(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: return "ko" if is_korean(request) else "en"
        Becomes: return "ko"
        """
        model = ScriptedModel(brief=self._BRIEF)
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=self._ENGLISH)
        english = {labels["en"] for labels in GENRE_LABELS.values()}
        assert out["stage"] == "choose_genre"
        assert set(out["genre_options"]) <= english, out["genre_options"]
        assert out["brief"]["tone"] == DEFAULTS["en"]["tone"]
        assert not _HANGUL.search(out[REPLY_NOTE_KEY]["en"]), out[REPLY_NOTE_KEY]["en"]
        # The premises are asked for with nothing Korean in the direction they are given.
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_seeded(model), ctx, action="continue", go_on=True)
        # (The fake's premises are Korean, so each is asked for again naming those words;
        # the first ask is the one the direction is given in.)
        assert not _HANGUL.search(
            model.prompts[[i for i, p in enumerate(model.prompts) if "Write one premise" in p][0]]
        )

    def test_each_note_names_the_genres_in_its_own_language(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: en_options = [genre_label(g, "en") for g in state.genre_options]
        Becomes: en_options = list(state.genre_options)
        """
        state = StartState(
            request="소설 하나 써줘",
            stage="choose_genre",
            brief=brief_from({}, "소설 하나 써줘"),
            genre_options=["호러", "판타지", "SF"],
        )
        note = _reply_note(state)
        assert "1. Horror (recommended)" in note["en"] and "Science fiction" in note["en"]
        assert not _HANGUL.search(note["en"]), note["en"]
        assert "1. 호러 (추천)" in note["ko"]


class TestTheStartFollowUps:
    """#2270: the review of #2268."""

    async def test_a_question_that_names_an_element_is_refused_not_redrawn(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if changes:
        Becomes: if True:
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        before = _state(tmp_path, story_id)

        asked = await _tool(model).execute(
            {
                "action": "continue",
                "must_have": ["여자 주인공"],
            },
            _ctx(tmp_path, story=story_id),
        )

        assert not asked.success and asked.error is not None and "choice" in asked.error
        assert _state(tmp_path, story_id) == before  # the three offered are still there

    async def test_the_refusal_names_the_stop_in_words_not_by_its_id(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: f"{_STOP_WORDS.get(state.stage, 'what it showed')}, and until every chapter is "
        Becomes: f"its '{state.stage}' stop, and until every chapter is "
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        for stage in ("review_cast", "review_outline"):
            out = await _ok(_tool(model), ctx, action="continue", go_on=True)
            assert out["stage"] == stage
            refused = await StoryOutlineTool().execute(
                {"action": "init", "chapter_titles": ["a"]}, ctx
            )
            assert refused.error is not None
            assert stage not in refused.error and "_" not in refused.error.split("story_")[0]

    async def test_after_a_failed_cast_the_refusal_says_to_try_the_cast_again(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: if state.cast_failed:
        Becomes: if False:
        """
        bad = ["not json"] * MAX_CAST_TRIES
        model = ScriptedModel(cast_replies=bad)
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        failed = await _ok(_tool(model), ctx, action="continue", choice=1)
        assert failed["cast_failed"] == "unreadable"

        refused = await StoryManuscriptTool(model_factory=lambda _c: None).execute(
            {"action": "write", "scene_id": "ch01.s01", "text": _SCENE}, ctx
        )

        assert refused.error is not None
        assert "No character could be proposed" in refused.error
        assert "Show the person the three premises" not in refused.error

    def test_a_word_for_another_genre_does_not_let_a_made_up_one_through(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: known == table and _has_word(lowered, word) for word, known in _GENRE_WORDS
        Becomes: _has_word(lowered, word) for word, known in _GENRE_WORDS
        """
        assert not names_genre("호러는 싫어", "로맨스")
        assert not brief_from({"genre": "로맨스"}, "호러는 싫어, 다른 걸로").from_request
        assert names_genre("추리 단편", "미스터리")  # another word for the same genre counts

    async def test_leave_lets_the_person_write_the_story_by_hand(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: if state is None or state.stage == "done" or state.left_request is not None:
        Becomes: if state is None or state.stage == "done":
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)

        left = await _ok(_tool(model), ctx, action="leave", request="그냥 내가 직접 쓸게")
        made = await StoryOutlineTool().execute({"action": "init", "chapter_titles": ["안개"]}, ctx)
        again = await _tool(model).execute({"action": "continue", "go_on": True}, ctx)

        assert left["left"] is True and "멈췄습니다" in left[REPLY_NOTE_KEY]["ko"]
        assert made.success, made.error
        assert _work(tmp_path, story_id).outline() is not None
        assert not again.success and again.error is not None and "left" in again.error
        assert _state(tmp_path, story_id)["stage"] == "choose_premise"
        assert _state(tmp_path, story_id)["left_request"] == "그냥 내가 직접 쓸게"

    async def test_leave_without_the_persons_words_is_refused(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if not asked_by:
        Becomes: if False:
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)
        before = _state(tmp_path, story_id)

        for request in (None, "  "):
            args: dict[str, Any] = {"action": "leave"}
            if request is not None:
                args["request"] = request
            refused = await _tool(model).execute(args, ctx)
            assert not refused.success and refused.error is not None
            assert "'request'" in refused.error

        assert _state(tmp_path, story_id) == before
        made = await StoryOutlineTool().execute({"action": "init", "chapter_titles": ["a"]}, ctx)
        assert not made.success  # the guard still holds

    async def test_no_refusal_after_a_by_hand_write_names_leave(self, tmp_path: Path) -> None:
        """A model that just tried to write by hand is not told how to switch the guard off.

        Killed by: examples/story/src/uclone_x_story/start_notes.py :: f"written it makes the outline and the new scenes itself. {_next_of(state)}"
        Becomes: f"written it makes the outline and the new scenes itself. {_next_of(state)} Call story_start 'leave' to write it yourself."
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        errors: list[str] = []
        for stage in ("choose_premise", "review_cast", "review_outline"):
            if stage != "choose_premise":
                out = await _ok(_tool(model), ctx, action="continue", go_on=True)
            assert out["stage"] == stage
            for refused in (
                await StoryOutlineTool().execute({"action": "init", "chapter_titles": ["a"]}, ctx),
                await StoryManuscriptTool(model_factory=lambda _c: None).execute(
                    {"action": "write", "scene_id": "ch01.s01", "text": _SCENE}, ctx
                ),
                await StoryLibraryTool().execute({"action": "create", "title": "다른 이야기"}, ctx),
            ):
                assert not refused.success and refused.error is not None
                errors.append(refused.error)
        assert errors and not [e for e in errors if "leave" in e.casefold()], errors


class TestAnEnglishStoryInBothNotes:
    """#2272 item 3: the two mutations #2271's review saw escape."""

    async def _english(self, tmp_path: Path) -> dict[str, Any]:
        model = ScriptedModel(brief={"title": None, "genre": None, "must_have": [], "avoid": []})
        return await _ok(
            _seeded(model), _ctx(tmp_path), action="start", request="Write me a story."
        )

    async def test_the_korean_note_names_an_english_storys_genres_in_korean(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: ko_options = [genre_label(g, "ko") for g in state.genre_options]
        Becomes: ko_options = list(state.genre_options)
        """
        out = await self._english(tmp_path)
        korean = {labels["ko"] for labels in GENRE_LABELS.values()}
        options = [
            line
            for line in out[REPLY_NOTE_KEY]["ko"].splitlines()
            if line[:2] in ("1.", "2.", "3.")
        ]
        assert len(options) == 3
        for line in options:
            assert line.split(". ", 1)[1].removesuffix(" (추천)") in korean, line

    async def test_an_english_storys_style_notes_have_english_headings(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: tone, audience, length, avoid = _STYLE_HEADINGS[brief.language]
        Becomes: tone, audience, length, avoid = _STYLE_HEADINGS["ko"]
        """
        out = await self._english(tmp_path)
        notes = StoryLibrary(tmp_path).load(out["open_story_id"]).style_notes or ""
        assert notes.startswith("Tone: "), notes
        assert "Audience: " in notes and "Length: " in notes
        assert not _HANGUL.search(notes), notes


class TestTheStopIsThePersonsToAnswer:
    """#2300: in a live run one turn ran start, continue "1" and continue "yes", so the
    premise, cast and outline stops were all answered by the model."""

    async def test_a_continue_in_the_turn_that_made_the_stop_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if context.turn_id is not None and context.turn_id == state.stop_turn_id:
        Becomes: if False:
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path, turn="t1"), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        before = _state(tmp_path, story_id)

        same = await _tool(model).execute(
            {"action": "continue", "choice": 1}, _ctx(tmp_path, story=story_id, turn="t1")
        )

        assert not same.success and same.error is not None and "next message" in same.error
        assert _state(tmp_path, story_id) == before
        moved = await _ok(
            _tool(model), _ctx(tmp_path, story=story_id, turn="t2"), action="continue", choice=1
        )
        assert moved["stage"] == "review_cast"

    async def test_the_turn_that_moved_the_flow_is_the_one_kept(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: state = state.model_copy(update={"stop_turn_id": context.turn_id})
        Becomes: state = state.model_copy(update={})
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path, turn="t1"), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"], turn="t2")
        await _ok(_tool(model), ctx, action="continue", choice=1)

        again = await _tool(model).execute({"action": "continue", "go_on": True}, ctx)

        assert not again.success
        assert _state(tmp_path, out["open_story_id"])["stop_turn_id"] == "t2"

    async def test_the_start_keeps_the_turn_that_made_its_first_stop(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: stop_turn_id=context.turn_id,
        Becomes: stop_turn_id=None,
        """
        out = await _ok(
            _tool(ScriptedModel()), _ctx(tmp_path, turn="t1"), action="start", request=REQUEST
        )
        assert _state(tmp_path, out["open_story_id"])["stop_turn_id"] == "t1"


class TestAWaitingStartGetsThePersonsAnswer:
    """#2300: at the outline stop the turn's context offered the first scene to write, and a
    live model wrote the chapter in chat; nothing was saved."""

    _TOOLS = frozenset({"story_start", "story_context", "story_manuscript"})

    async def _at_outline(self, tmp_path: Path) -> str:
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        while out["stage"] != "review_outline":
            out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        return str(out["story_id"])

    async def test_the_turn_is_told_the_message_answers_the_stop_and_gets_no_scene(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: waiting = start_waiting(work) if _STORY_START in tool_names else None
        Becomes: waiting = None
        """
        story_id = await self._at_outline(tmp_path)

        aid = SceneTurnHook().turn_aid(
            message="네, 첫 장을 써줘",
            story_id=story_id,
            room_id=ROOM,
            workspace_root=tmp_path,
            tool_names=self._TOOLS,
        )

        assert isinstance(aid, StartTurnAid)
        assert aid.section.startswith("[Story start waiting]")
        assert "the outline" in aid.section and "story_start 'continue'" in aid.section
        assert "[Scene to write]" not in aid.section

    async def test_a_turn_that_did_not_move_the_start_says_nothing_was_saved(
        self, tmp_path: Path
    ) -> None:
        """The start's record decides, not the calls: the agent records a skipped call as a
        success, and the live run's reply then claimed nothing.

        Killed by: examples/story/src/uclone_x_story/scene_turn.py :: moved = found is None or found.digest != self.start_digest
        Becomes: moved = True
        """
        story_id = await self._at_outline(tmp_path)
        aid = SceneTurnHook().turn_aid(
            message="네",
            story_id=story_id,
            room_id=ROOM,
            workspace_root=tmp_path,
            tool_names=self._TOOLS,
        )
        assert isinstance(aid, StartTurnAid)
        skipped = ToolExecutionRecord(tool_name="story_start", output={"status": "skipped"})

        assert aid.reply_text("알겠습니다.") == "알겠습니다."
        [line] = aid.reply_lines([skipped], korean=True)
        assert line == (
            "이번 답으로 이야기에 저장된 것은 없습니다. 이야기는 개요에 대한 답을 기다리고 있습니다."
        )
        assert "the outline" in aid.reply_lines([skipped], korean=False)[0]

    async def test_a_turn_that_did_not_touch_the_story_is_not_told_it_saved_nothing(
        self, tmp_path: Path
    ) -> None:
        """A live turn asked only for a picture, and its whole reply was the line.

        Killed by: examples/story/src/uclone_x_story/scene_turn.py :: tried = any(record.tool_name.startswith("story_") for record in records)
        Becomes: tried = True
        """
        story_id = await self._at_outline(tmp_path)
        aid = SceneTurnHook().turn_aid(
            message="첫 장면을 그려 줘",
            story_id=story_id,
            room_id=ROOM,
            workspace_root=tmp_path,
            tool_names=self._TOOLS,
        )
        assert isinstance(aid, StartTurnAid)
        aid.reply_text("그림을 그렸습니다.")

        assert aid.reply_lines([ToolExecutionRecord(tool_name="a2a_call")], korean=True) == []

    async def test_a_moved_start_notes_only_a_scene_the_writer_wrote_itself(
        self, tmp_path: Path
    ) -> None:
        """#2300: the outline stop moved, and the same reply held a scene in prose that was
        never saved.

        Killed by: examples/story/src/uclone_x_story/scene_turn.py :: long_text = own_sentences(self.reply, [*given, *saved]) >= OWN_SENTENCES_SHOWN
        Becomes: long_text = False
        """
        story_id = await self._at_outline(tmp_path)
        aid = SceneTurnHook().turn_aid(
            message="네",
            story_id=story_id,
            room_id=ROOM,
            workspace_root=tmp_path,
            tool_names=self._TOOLS,
        )
        assert isinstance(aid, StartTurnAid)
        ctx = _ctx(tmp_path, story=story_id)
        moved = await _ok(_tool(ScriptedModel()), ctx, action="continue", go_on=True)
        assert moved["stage"] == "review_chapter"

        aid.reply_text("첫 장의 계획을 보여 드립니다.")
        assert aid.reply_lines([], korean=True) == []
        aid.reply_text("하린은 등대 계단을 올랐다. " * 40)
        [line] = aid.reply_lines([], korean=True)
        assert line == (
            "제가 직접 쓴 글은 이야기에 저장되지 않았습니다. 이야기는 1장의 장면 계획에 대한 "
            "답을 기다리고 있습니다."
        )

    async def test_a_finished_start_gives_the_scene_again(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: if state is None or state.stage == "done" or state.left_request is not None:
        Becomes: if state is None or state.left_request is not None:
        """
        story_id = await self._at_outline(tmp_path)
        record = yaml.safe_load((tmp_path / "stories" / story_id / START_FILE).read_text())
        record["stage"] = "done"
        (tmp_path / "stories" / story_id / START_FILE).write_text(
            yaml.safe_dump(record, allow_unicode=True)
        )

        aid = SceneTurnHook().turn_aid(
            message="다음 장면",
            story_id=story_id,
            room_id=ROOM,
            workspace_root=tmp_path,
            tool_names=self._TOOLS,
        )

        assert not isinstance(aid, StartTurnAid)
        assert aid is not None and aid.section.startswith("[Scene to write]")


class TestAnAnswerThatTakesAPremiseAndAsksForMore:
    """#2305 / #2351: taking a premise via structured parameters (`go_on`, `choice`,
    `must_have`) advances from `choose_premise` and carries `must_have` to later stages."""

    _WANTED = "다음 장에서 먹물이 돌아온다"

    async def _at_premises(self, tmp_path: Path, model: ScriptedModel) -> ToolContext:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        assert out["stage"] == "choose_premise"
        return _ctx(tmp_path, story=out["open_story_id"])

    async def test_a_yes_with_a_request_takes_the_recommended_premise(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: go_on=params.go_on,
        Becomes: go_on=False,
        """
        model = ScriptedModel()
        ctx = await self._at_premises(tmp_path, model)

        out = await _ok(_tool(model), ctx, action="continue", go_on=True, must_have=[self._WANTED])

        assert out["stage"] == "review_cast"
        assert out["chosen"] == out["recommended"]

    async def test_the_premise_they_name_is_the_one_taken(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: number = choice if choice is not None else retry
        Becomes: number = retry
        """
        model = ScriptedModel()
        ctx = await self._at_premises(tmp_path, model)

        out = await _ok(_tool(model), ctx, action="continue", choice=3)

        assert out["stage"] == "review_cast" and out["chosen"] == 3

    async def test_a_number_that_cannot_be_read_is_asked_about_not_guessed(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if number is None:
        Becomes: if False:
        """
        model = ScriptedModel()
        ctx = await self._at_premises(tmp_path, model)
        story_id = str(ctx.story_id)
        before = _state(tmp_path, story_id)

        result = await _tool(model).execute({"action": "continue"}, ctx)

        assert not result.success and result.error is not None
        assert "which of the three" in result.error
        assert _state(tmp_path, story_id)["stage"] == before["stage"] == "choose_premise"

    async def test_what_they_asked_for_reaches_the_stages_after_it(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: self._save(brief=_with_wanted(self.state.brief, answer))
        Becomes: pass
        """
        model = ScriptedModel()
        ctx = await self._at_premises(tmp_path, model)

        out = await _ok(_tool(model), ctx, action="continue", go_on=True, must_have=[self._WANTED])
        assert self._WANTED in out["brief"]["must_have"]
        assert "고양이" in out["brief"]["must_have"]  # the request's own stays
        await _ok(_tool(model), ctx, action="continue", go_on=True)  # the outline

        assert self._WANTED in _last(model, "List the main characters")
        assert self._WANTED in _last(model, CHAPTER_PROMPT_MARK)


class TestWhatTheGenreAnswerAlsoAsksForIsKept:
    """#2305 / #2351: picky's "SF는 좋지만 … 여성 정비사가 주인공인 이야기" passes `genre` and
    `must_have` to `story_start 'continue'` without a secondary LLM parser."""

    async def test_the_genre_answers_must_haves_reach_the_premises(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: wanted = _wanted_from(params)
        Becomes: wanted = {"must_have": [], "avoid": []}
        """
        model = ScriptedModel(brief=_BARE_BRIEF)
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=BARE)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        wanted = next(g for g in ("로맨스", "미스터리") if g != out["genre_options"][0])

        premises = await _ok(
            _seeded(model),
            ctx,
            action="continue",
            genre=wanted,
            must_have=["여성 정비사가 주인공"],
        )

        assert premises["brief"]["genre"] == wanted
        assert "여성 정비사가 주인공" in premises["brief"]["must_have"]
        assert "Must include: 여성 정비사가 주인공" in _last(model, "Write one premise")


class TestTheStartSaysWhatItMeans:
    """#2301: what the start proposed and said in the 2026-10-12 Dell run (#2299)."""

    async def test_a_genre_the_person_turns_down_is_not_taken_over_the_one_they_ask_for(
        self, tmp_path: Path
    ) -> None:
        """picky T2: a turned-down genre in `avoid` is not accepted via `genre`.

        Killed by: examples/story/src/uclone_x_story/start_premise.py :: if not _genre_turned_down(genre, refused):
        Becomes: if True:
        """
        brief = {**_BARE_BRIEF, "avoid": ["판타지 장르"]}
        model = ScriptedModel(brief=brief)
        out = await _ok(_seeded(model, 3), _ctx(tmp_path), action="start", request=BARE)
        assert out["stage"] == "choose_genre"
        ctx = _ctx(tmp_path, story=out["open_story_id"])

        premises = await _ok(_seeded(model, 3), ctx, action="continue", genre="판타지", go_on=True)

        assert premises["stage"] == "choose_premise"
        assert premises["brief"]["genre"] == out["genre_options"][0] != "판타지"
        assert _story_genre(tmp_path, out["open_story_id"]) != "판타지"

    def test_a_request_that_turns_a_genre_down_does_not_state_it(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: if key == "genre" and _genre_turned_down(value, turned_down_genres(avoid)):
        Becomes: if False:
        """
        brief = brief_from({"genre": "판타지", "avoid": ["판타지"]}, "판타지 말고 아무거나 써 줘")
        assert "genre" not in brief.from_request
        assert brief.genre == DEFAULTS["ko"]["genre"]  # a default, proposed over at the stop

    async def test_the_premises_are_asked_for_in_the_storys_language(self, tmp_path: Path) -> None:
        r"""picky T2: "2. eclipsed melody" in a Korean story.

        Killed by: examples/story/src/uclone_x_story/start_premise.py :: f"stands in the way.\n{_in_language(state)}{fix}\n{_PREMISE_REPLY}"
        Becomes: f"stands in the way.\n{fix}\n{_PREMISE_REPLY}"
        """
        model = ScriptedModel()
        await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        assert "Write the title and the premise in Korean only." in _last(
            model, "Write one premise"
        )

    async def test_a_premise_with_words_outside_the_storys_script_is_drawn_once_more(
        self, tmp_path: Path
    ) -> None:
        """picky T2: one premise in English, another with "죄孽" in it. Each is drafted
        again, told its words, and the cleaner draft is kept.

        Killed by: examples/story/src/uclone_x_story/start_premise.py :: if text and foreign:
        Becomes: if False:
        """
        english = ("eclipsed melody", "During the rare eclipse market, a smuggler plays a song.")
        hanja = ("가시 외벽", "강을 지키는 페리맨은 그의 죄孽을 드러내야만 했다.")
        clean_english = ("일식의 노래", "드문 일식 장터에서 밀수꾼이 금지된 노래를 연주한다.")
        clean_hanja = ("가시 외벽", "강을 지키는 페리맨은 그의 죄를 드러내야만 했다.")
        model = ScriptedModel(
            premises=[_PREMISES[0], english, hanja], later_premises=[clean_english, clean_hanja]
        )

        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)

        texts = [f"{p['title']} {p['premise']}" for p in out["premises"]]
        assert out["premises"][1]["title"] == "일식의 노래"
        assert out["premises"][2]["premise"] == clean_hanja[1]
        assert all(words_outside(t, {"HANGUL"}) == [] for t in texts), texts
        again = [p for p in model.prompts if "had words outside Korean" in p]
        assert len(again) == 2
        assert "eclipsed, melody" in again[0] and "죄孽을" in again[1]

    def test_a_script_the_persons_own_words_use_is_the_storys_too(self) -> None:
        """ "SF로 해줘": a Korean story may say "SF".

        Killed by: examples/story/src/uclone_x_story/start_state.py :: return LANGUAGE_SCRIPTS[language] | frozenset(scripts_of(said))
        Becomes: return LANGUAGE_SCRIPTS[language]
        """
        scripts = story_scripts("ko", ["SF 이야기로 해줘"])
        assert words_outside("우주 정거장의 SF 이야기", scripts) == []
        assert words_outside("그의 죄孽을", scripts) == ["죄孽을"]

    async def test_two_premises_with_one_title_are_not_both_offered(self, tmp_path: Path) -> None:
        """continuity probe T1: "은시계의 유산" twice.

        Killed by: examples/story/src/uclone_x_story/start_premise.py :: if (same is not None or not similarity.distinct) and len(engines) > PREMISE_COUNT:
        Becomes: if not similarity.distinct and len(engines) > PREMISE_COUNT:
        """
        first = ("은시계의 유산", "노인이 숨을 거두며 손녀에게 낡은 시계를 남긴다.")
        second = ("은시계의 유산", "시장 골목의 수리공이 멈춘 시계 속에서 편지를 찾는다.")
        other = ("시계탑의 손님", "폭풍우 치는 밤, 낯선 여행자가 시계탑의 문을 두드린다.")
        model = ScriptedModel(premises=[first, second, _PREMISES[2]], later_premises=[other])

        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)

        titles = [p["title"] for p in out["premises"]]
        assert titles == ["은시계의 유산", "시계탑의 손님", _PREMISES[2][0]]
        redraw = _last(model, "Write one premise")
        assert "titled 은시계의 유산" in redraw and "give this one another title" in redraw

    async def test_the_premises_are_measured_with_their_titles(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_premise.py :: similarity = await premise_similarity(_premise_texts(premises), embedder)
        Becomes: similarity = await premise_similarity([p.premise for p in premises], embedder)
        """
        embedder = FakeEmbedder(_APART)
        await _ok(_tool(ScriptedModel(), embedder), _ctx(tmp_path), action="start", request=REQUEST)
        assert embedder.calls == [[f"{t}\n{p}" for t, p in _PREMISES]]

    async def test_an_object_the_cast_reply_also_lists_as_an_item_is_not_cast(
        self, tmp_path: Path
    ) -> None:
        """continuity probe T1: "인물과 장소를 제안했습니다: 이진호, 하린, 은시계, …".

        Killed by: examples/story/src/uclone_x_story/start_cast.py :: if name.casefold() in objects:
        Becomes: if False:
        """
        reply = {
            "characters": [
                {"name": "하린", "profile": "손녀.", "gender": "female", "status": "alive"},
                {"name": "은시계", "profile": "할아버지의 시계.", "gender": "other"},
            ],
            "places": [{"name": "시장", "profile": "오래된 시장."}],
            "items": [{"name": "은시계", "profile": "할아버지가 남긴 시계."}],
        }
        model = ScriptedModel(cast_replies=[json.dumps(reply, ensure_ascii=False)])
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])

        cast_out = await _ok(_tool(model), ctx, action="continue", choice=1)

        assert cast_out["stage"] == "review_cast"
        state = _state(tmp_path, out["open_story_id"])
        assert [c["name"] for c in state["cast"]] == ["하린", "시장"]
        assert "은시계" not in cast_out[REPLY_NOTE_KEY]["ko"]
        assert '"items"' in _last(model, "List the main characters")

    async def test_a_reply_that_lists_the_premises_gets_only_the_question_after_them(
        self, tmp_path: Path
    ) -> None:
        """picky T2: the model's list, then the code's list of the same premises.

        Killed by: examples/story/src/uclone_x_story/start.py :: note[NOTE_LISTS_KEY] = [p.title for p in state.premises]
        Becomes: pass
        """
        out = await _ok(_tool(ScriptedModel()), _ctx(tmp_path), action="start", request=REQUEST)
        assert out[REPLY_NOTE_KEY][NOTE_LISTS_KEY] == [t for t, _ in _PREMISES]
        record = ToolExecutionRecord(tool_name="story_start", output=out)
        listed = "\n".join(f"{n}. **{t}** - {p}" for n, (t, p) in enumerate(_PREMISES, start=1))

        (note,) = reply_notes([record], korean=True, reply=f"발상입니다.\n\n{listed}")

        assert note.startswith("마음에 드는 발상의 번호를 골라 주세요.")
        assert not any(t in note for t, _ in _PREMISES)


class TestAnEditOfAWrittenSceneIsNotPlanFeedback:
    """#2303 item 3: at a chapter's plan, words asking to fix a scene already written were
    read as feedback on the next plan."""

    _TOOLS = frozenset({"story_start", "story_context", "story_manuscript"})

    async def _after_first(self, tmp_path: Path, model: ScriptedModel) -> tuple[str, ToolContext]:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        while int(out.get("chapter_written") or 0) < 1:
            out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        assert out["stage"] == "review_chapter" and out["details_chapter"] == 2
        return str(out["story_id"]), ctx

    async def test_the_plan_is_kept_and_the_written_scene_is_rewritten_by_the_flow(
        self, tmp_path: Path
    ) -> None:
        """#2361: the flow rewrites the written scene the person's words ask to change, with
        their words; it no longer tells the Writer to edit it by hand.

        Killed by: examples/story/src/uclone_x_story/start_details.py :: if edits:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/start_details.py :: await self._rewrite_asked(edits, asked)
        Becomes: pass
        """
        model = ScriptedModel(
            details_revised={**_DETAILS_REVISED, "edits_written": ["계단"]},
            scenes=[_SCENE, _SCENE_TWO, _SCENE_CRIES],
        )
        story_id, ctx = await self._after_first(tmp_path, model)
        work = _work(tmp_path, story_id)
        before = _state(tmp_path, story_id)["chapter_details"]
        second = work.manuscript("ch01.s02")

        out = await _ok(
            _tool(model), ctx, action="continue", changes="「계단」 장면에서 하린이 울게 고쳐 줘"
        )

        assert out["stage"] == "review_chapter"
        assert out["scene_edits"] == [{"id": "ch01.s01", "title": "계단"}]
        assert out["chapter_details"] == before
        rewritten = work.manuscript("ch01.s01")
        assert rewritten is not None and rewritten.text == _SCENE_CRIES
        # Only the scene asked about: the other written scene is as it was.
        assert work.manuscript("ch01.s02") == second
        asked = [p for p in model.prompts if "Write scene" in p][-1]
        assert "「계단」 장면에서 하린이 울게 고쳐 줘" in asked
        assert (
            "「계단」" in out[REPLY_NOTE_KEY]["ko"]
            and "다시 써서 저장했습니다" in out[REPLY_NOTE_KEY]["ko"]
        )
        assert "story_manuscript" not in out["next"]
        assert "- 계단\n- 목소리\n" in _last(model, DETAILS_REVISE_MARK)

    @staticmethod
    def _retitle(tmp_path: Path, story_id: str, scene_id: str, title: str) -> None:
        """Give `scene_id` another scene's title in the outline, as a model's outline can."""
        work = _work(tmp_path, story_id)
        found = work.outline()
        assert found is not None
        outline, digest = found
        chapters = [
            chapter.model_copy(
                update={
                    "scenes": [
                        s.model_copy(update={"title": title}) if s.id == scene_id else s
                        for s in chapter.scenes
                    ]
                }
            )
            for chapter in outline.chapters
        ]
        changed = outline.model_copy(update={"chapters": chapters})
        work.save_outline(changed, room_id=ROOM, expected_digest=digest)

    async def test_a_title_two_written_scenes_share_rewrites_neither_and_asks_which(
        self, tmp_path: Path
    ) -> None:
        """#2365 item 1: since #2363 the flow overwrites the scene a title names, and a
        title two written scenes share named the last of them. Neither is rewritten; the
        person is asked which one.

        Killed by: examples/story/src/uclone_x_story/start_details.py :: if ambiguous:
        Becomes: if False:
        """
        model = ScriptedModel(
            details_revised={**_DETAILS_REVISED, "edits_written": ["계단"]},
            scenes=[_SCENE, _SCENE_TWO, _SCENE_CRIES],
        )
        story_id, ctx = await self._after_first(tmp_path, model)
        self._retitle(tmp_path, story_id, "ch01.s02", "계단")
        work = _work(tmp_path, story_id)
        first, second = work.manuscript("ch01.s01"), work.manuscript("ch01.s02")

        out = await _ok(
            _tool(model), ctx, action="continue", changes="「계단」 장면에서 하린이 울게 고쳐 줘"
        )

        assert work.manuscript("ch01.s01") == first and work.manuscript("ch01.s02") == second
        assert out["stage"] == "review_chapter"
        ko, en = out[REPLY_NOTE_KEY]["ko"], out[REPLY_NOTE_KEY]["en"]
        assert "「계단」" in ko and "어느 장면" in ko and "1장 1번째" in ko and "1장 2번째" in ko
        assert "「" not in en and "chapter 1, scene 2" in en
        assert "which" in out["next"] and "story_manuscript" not in out["next"]
        # The plan prompt tells the two apart, so an answer can name one.
        asked = _last(model, DETAILS_REVISE_MARK)
        assert "- 계단 (chapter 1, scene 1)\n- 계단 (chapter 1, scene 2)\n" in asked

    async def test_the_answer_to_which_one_rewrites_with_the_words_first_asked(
        self, tmp_path: Path
    ) -> None:
        """#2366 review: after "which one?", the person answers with a place only ("1장
        2번째"). The rewrite must still carry what they first asked for ("하린이 울게"), or
        the scene is replaced by one written to a place and told it is as they asked.

        Killed by: examples/story/src/uclone_x_story/start_details.py :: asked = "\n".join([*(a["words"] for a in pending), feedback])
        Becomes: asked = feedback
        """
        model = ScriptedModel(
            details_revised={**_DETAILS_REVISED, "edits_written": ["계단"]},
            scenes=[_SCENE, _SCENE_TWO, _SCENE_CRIES],
        )
        story_id, ctx = await self._after_first(tmp_path, model)
        self._retitle(tmp_path, story_id, "ch01.s02", "계단")
        first = "「계단」 장면에서 하린이 울게 고쳐 줘"
        await _ok(_tool(model), ctx, action="continue", changes=first)
        assert _state(tmp_path, story_id)["details_scene_edits_ambiguous"][0]["words"] == first

        model._details_revised = {  # pyright: ignore[reportPrivateUsage]
            **_DETAILS_REVISED,
            "edits_written": ["계단 (chapter 1, scene 2)"],
        }
        out = await _ok(_tool(ctx_model := model), ctx, action="continue", changes="1장 2번째")

        assert out["scene_edits"] == [{"id": "ch01.s02", "title": "계단"}]
        rewrite = [p for p in ctx_model.prompts if "Write scene" in p][-1]
        assert first in rewrite and "1장 2번째" in rewrite
        # The plan question was asked with both, so the place could be matched to the title.
        assert first in _last(model, DETAILS_REVISE_MARK)
        assert _state(tmp_path, story_id)["details_scene_edits_ambiguous"] == []

    async def test_two_answers_later_the_rewrite_still_carries_the_first_words(
        self, tmp_path: Path
    ) -> None:
        """#2368 item 1: asked which one twice, the person answers with a place the third
        time. The words of the first turn reach the rewrite, not only the last answer's.

        Killed by: examples/story/src/uclone_x_story/start_details.py :: _where(outline, written, title) | {"words": asked}
        Becomes: _where(outline, written, title) | {"words": feedback}
        """
        model = ScriptedModel(
            details_revised={**_DETAILS_REVISED, "edits_written": ["계단"]},
            scenes=[_SCENE, _SCENE_TWO, _SCENE_CRIES],
        )
        story_id, ctx = await self._after_first(tmp_path, model)
        self._retitle(tmp_path, story_id, "ch01.s02", "계단")
        first = "「계단」 장면에서 하린이 울게 고쳐 줘"
        await _ok(_tool(model), ctx, action="continue", changes=first)
        await _ok(_tool(model), ctx, action="continue", changes="그 계단 장면 말이야")
        assert _state(tmp_path, story_id)["details_scene_edits_ambiguous"] != []

        model._details_revised = {  # pyright: ignore[reportPrivateUsage]
            **_DETAILS_REVISED,
            "edits_written": ["계단 (chapter 1, scene 2)"],
        }
        await _ok(_tool(model), ctx, action="continue", changes="1장 2번째")

        rewrite = [p for p in model.prompts if "Write scene" in p][-1]
        assert first in rewrite and "그 계단 장면 말이야" in rewrite and "1장 2번째" in rewrite

    async def test_words_left_unanswered_are_not_feedback_on_the_next_plan(
        self, tmp_path: Path
    ) -> None:
        """#2368 item 2: after "which one?" the person moves on to the next chapter's plan.
        The scene words still waiting are given apart, labelled as only for telling which
        written scene was meant, not as feedback on this plan.

        Killed by: examples/story/src/uclone_x_story/start_details.py :: f"{_feedback_with_brief(feedback, state.brief)}\n\n"
        Becomes: f"{_feedback_with_brief(asked, state.brief)}\n\n"
        """
        model = ScriptedModel(
            details_revised={**_DETAILS_REVISED, "edits_written": ["계단"]},
            scenes=[_SCENE, _SCENE_TWO, _SCENE_CRIES],
        )
        story_id, ctx = await self._after_first(tmp_path, model)
        self._retitle(tmp_path, story_id, "ch01.s02", "계단")
        first = "「계단」 장면에서 하린이 울게 고쳐 줘"
        await _ok(_tool(model), ctx, action="continue", changes=first)

        model._details_revised = {**_DETAILS_REVISED, "edits_written": []}  # pyright: ignore[reportPrivateUsage]
        moved_on = "2장은 바닷가에서 시작해 줘"
        await _ok(_tool(model), ctx, action="continue", changes=moved_on)

        asked = _last(model, DETAILS_REVISE_MARK)
        said = asked.split("The person's feedback, word for word:\n", 1)[1].split("\n\n", 1)[0]
        assert moved_on in said and first not in said
        # Still given, apart and labelled, so a place in the answer can be matched.
        assert first in asked and "only to tell which written scene" in asked
        assert _state(tmp_path, story_id)["details_feedback"][-1] == moved_on

    async def test_a_title_told_apart_by_its_place_rewrites_that_scene_only(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: title if seen[title] == 1 else f"{title} (chapter {chapter_no}, scene {scene_no})"
        Becomes: title
        """
        model = ScriptedModel(
            details_revised={**_DETAILS_REVISED, "edits_written": ["계단 (chapter 1, scene 2)"]},
            scenes=[_SCENE, _SCENE_TWO, _SCENE_CRIES],
        )
        story_id, ctx = await self._after_first(tmp_path, model)
        self._retitle(tmp_path, story_id, "ch01.s02", "계단")
        work = _work(tmp_path, story_id)
        first = work.manuscript("ch01.s01")

        out = await _ok(
            _tool(model), ctx, action="continue", changes="1장 두 번째 계단 장면을 고쳐 줘"
        )

        assert out["scene_edits"] == [{"id": "ch01.s02", "title": "계단"}]
        assert work.manuscript("ch01.s01") == first
        rewritten = work.manuscript("ch01.s02")
        assert rewritten is not None and rewritten.text == _SCENE_CRIES

    async def test_a_rewrite_every_draft_of_which_is_refused_leaves_the_scene_and_says_so(
        self, tmp_path: Path
    ) -> None:
        """#2365 item 2: the refused-draft path and the person's copy.

        Killed by: examples/story/src/uclone_x_story/start_scene.py :: failed.append(scene_id)  # every draft refused: the scene stays as it was
        Becomes: pass  # every draft refused: the scene stays as it was
        """
        model = ScriptedModel(
            details_revised={**_DETAILS_REVISED, "edits_written": ["계단"]},
            scenes=[_SCENE, _SCENE_TWO, *([_SCENE_GLUED] * MAX_DRAFTS)],
        )
        story_id, ctx = await self._after_first(tmp_path, model)
        work = _work(tmp_path, story_id)
        before = work.manuscript("ch01.s01")

        out = await _ok(_tool(model), ctx, action="continue", changes="「계단」 장면을 다시 써 줘")

        assert work.manuscript("ch01.s01") == before
        assert _state(tmp_path, story_id)["details_scene_edits_failed"] == ["ch01.s01"]
        ko, en = out[REPLY_NOTE_KEY]["ko"], out[REPLY_NOTE_KEY]["en"]
        assert "이미 쓴 장면(「계단」)은 다시 쓰지 못해 그대로 두었습니다." in ko
        assert "다시 써서 저장했습니다" not in ko
        assert "could not be written again" in en and "「" not in en
        assert any("was not written again" in n for n in out["notes"])

    async def test_a_rewrite_that_brings_the_dead_back_is_caught(self, tmp_path: Path) -> None:
        """#2365 item 2: the rewritten scene is read again and checked against the story.

        Killed by: examples/story/src/uclone_x_story/start_scene.py :: await self._check(done)  # the scenes asked about, checked again
        Becomes: pass
        """
        model = ScriptedModel(
            details_revised={**_DETAILS_REVISED, "edits_written": ["목소리"]},
            scenes=[_DEAD_SCENE, _SCENE_TWO, _SCENE],
            facts=[[_HARIN_DEAD], [], [_HARIN_BACK]],
        )
        story_id, ctx = await self._after_first(tmp_path, model)
        assert _state(tmp_path, story_id)["continuity"] == []

        out = await _ok(
            _tool(model), ctx, action="continue", changes="「목소리」 장면을 다시 써 줘"
        )

        found = _work(tmp_path, story_id).manuscript("ch01.s02")
        assert found is not None and found.text == _SCENE
        assert [c["scene_id"] for c in out["continuity"]] == ["ch01.s02"]

    async def test_the_waiting_turn_is_told_which_scenes_are_written(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: _start_section(stop["en"], _written_titles(work, waiting.stage)),
        Becomes: _start_section(stop["en"]),
        """
        story_id, _ = await self._after_first(tmp_path, ScriptedModel())

        aid = SceneTurnHook().turn_aid(
            message="계단 장면을 고쳐 줘",
            story_id=story_id,
            room_id=ROOM,
            workspace_root=tmp_path,
            tool_names=self._TOOLS,
        )

        assert isinstance(aid, StartTurnAid)
        assert "Scenes already written: 「계단」, 「목소리」" in aid.section
        # #2361: an edit of one goes to story_start, which rewrites it; not by hand.
        assert "story_manuscript" not in aid.section.split("Scenes already written")[1]
        assert "'changes'" in aid.section.split("Scenes already written")[1]


class TestOwnTextIsCountedInSentences:
    """#2303 item 2: the reply line counted characters, so the same note was a scene in
    Hangul and seventy words in English; and a reply that relayed the premises it was
    given was told it wrote them."""

    _TOOLS = frozenset({"story_start", "story_context", "story_manuscript"})
    _RELAYED = [f"{n}번째 발상에서 하린은 등대의 불빛을 따라 걸었다." for n in range(1, 13)]

    async def _moved(self, tmp_path: Path) -> StartTurnAid:
        """An aid whose turn moved the start, from the outline to the first chapter's plan."""
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        while out["stage"] != "review_outline":
            out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        aid = SceneTurnHook().turn_aid(
            message="네",
            story_id=str(out["story_id"]),
            room_id=ROOM,
            workspace_root=tmp_path,
            tool_names=self._TOOLS,
        )
        assert isinstance(aid, StartTurnAid)
        await _ok(_tool(model), ctx, action="continue", go_on=True)
        return aid

    async def test_a_reply_that_relays_what_the_tools_returned_is_not_told_it_wrote_it(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: given = [text for record in records for text in _strings(record.output)]
        Becomes: given = []
        """
        aid = await self._moved(tmp_path)
        output: dict[str, JsonValue] = {"premises": [{"premise": " ".join(self._RELAYED)}]}
        record = ToolExecutionRecord(tool_name="story_start", output=output)
        relayed = "\n".join(f"{n}. **{s}**" for n, s in enumerate(self._RELAYED, start=1))

        aid.reply_text(relayed)

        assert aid.reply_lines([record], korean=True) == []
        aid.reply_text(relayed.replace("하린", "먹물"))  # not what the tools returned
        assert aid.reply_lines([record], korean=True) != []

    async def test_an_english_explanation_of_a_few_sentences_is_not_a_scene(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: long_text = own_sentences(self.reply, [*given, *saved]) >= OWN_SENTENCES_SHOWN
        Becomes: long_text = len(self.reply.strip()) >= 400
        """
        aid = await self._moved(tmp_path)
        said = (
            "The outline is settled, and the first chapter's plan is ready for you. "
            "It opens at the lighthouse on the night the fog comes in. "
            "Harin climbs the stairs and hears someone call her name. "
            "The cat follows her up, and the lamp is lit at the top. "
            "The second scene follows the voice into the fog, and the cat is lost there. "
            "Nothing in it is written yet; the plan is only what each scene will do. "
            "Tell me what to change, or say yes and I will write it."
        )
        assert len(said) > 400

        aid.reply_text(said)

        assert aid.reply_lines([], korean=False) == []


class TestTheStopsQuestionIsAskedOnce:
    """#2305 item 3, and a bare number at the genre stop."""

    async def test_the_premise_stops_question_is_marked_and_not_added_again(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: note[NOTE_ASKS_KEY] = [str(note[lang]).splitlines()[-1] for lang in ("ko", "en")]
        Becomes: pass
        """
        out = await _ok(_tool(ScriptedModel()), _ctx(tmp_path), action="start", request=REQUEST)
        question = out[REPLY_NOTE_KEY]["ko"].splitlines()[-1]
        assert out[REPLY_NOTE_KEY][NOTE_ASKS_KEY][0] == question
        record = ToolExecutionRecord(tool_name="story_start", output=out)
        reply = question.replace("‘", "'").replace("’", "'") + " 어느 것이 좋으세요?"

        (note,) = reply_notes([record], korean=True, reply=reply)

        # The list is still added (the reply names no title); only the question is not.
        assert note == "\n".join(out[REPLY_NOTE_KEY]["ko"].splitlines()[:-1])

    async def test_a_bare_number_at_the_genre_stop_is_not_read_as_a_request(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: genre = _genre_choice(state, params, wanted)
        Becomes: genre = state.genre_options[0]
        """
        model = ScriptedModel(brief=_BARE_BRIEF)
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=BARE)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        read_before = sum("write down ONLY" in p for p in model.prompts)

        premises = await _ok(_seeded(model), ctx, action="continue", choice=2)

        assert premises["brief"]["genre"] == out["genre_options"][1]
        assert sum("write down ONLY" in p for p in model.prompts) == read_before


class TestATakenOutlineOrPlanWithARequestMovesOn:
    """#2014 / #2351: structured parameters (`go_on`, `changes`, `must_have`, `rewrite`)
    drive the outline, cast wait, and chapter plan stops without secondary LLM parsers."""

    _WANTED = "다음 장에서 먹물이 돌아온다"

    async def _at(self, tmp_path: Path, model: ScriptedModel, stage: str) -> ToolContext:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        while out["stage"] != stage:
            out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        return ctx

    async def test_words_that_take_the_outline_settle_it_and_keep_the_request(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: feedback = (changes or "").strip()
        Becomes: feedback = "revise"
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        out = await _ok(_tool(model), ctx, action="continue", go_on=True, must_have=[self._WANTED])

        assert out["plot_locked"] is True and out["stage"] == "review_chapter"
        assert self._WANTED in out["brief"]["must_have"]
        assert not any(REVISE_MARK in p for p in model.prompts)

    async def test_words_that_ask_for_a_change_still_revise_it(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: feedback = (changes or "").strip()
        Becomes: feedback = ""
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        out = await _ok(_tool(model), ctx, action="continue", changes="반전을 더 일찍 넣어 줘")

        assert out["stage"] == "review_outline" and "outline_revised" in out["done_now"]
        assert "반전을 더 일찍 넣어 줘" in _last(model, REVISE_MARK)

    async def test_words_that_take_a_chapters_plan_write_it(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: said = (changes or "").strip()
        Becomes: said = ""
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_chapter")

        revised = await _ok(
            _tool(model), ctx, action="continue", changes="첫 장면을 부두로 바꿔 줘"
        )
        assert revised["stage"] == "review_chapter" and "details_revised" in revised["done_now"]

        out = await _ok(_tool(model), ctx, action="continue", go_on=True, must_have=[self._WANTED])

        assert out["chapter_written"] == 1
        assert self._WANTED in out["brief"]["must_have"]

    async def test_words_that_take_the_waiting_cast_go_on_over_it(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: if pending and not (go_on or self.state.quick):
        Becomes: if pending and not self.state.quick:
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")
        waiting = await _ok(_tool(model), ctx, action="continue")
        assert waiting["stage"] == "review_outline" and waiting["cast_waiting"]

        out = await _ok(_tool(model), ctx, action="continue", go_on=True)

        assert out["stage"] == "review_chapter" and out["cast_assumed"]


class TestWhatATakenStopKeepsIsSaidBack:
    """#2313 / #2351: the elements passed via `must_have` and `avoid` at a stop are said
    back in the reply note, and the waiting turn aid instructs the Writer not to call
    `story_start` when the person only asks a question."""

    _MUST = "다음 장에서 먹물이 돌아온다"
    _AVOID = "2장의 잔인한 장면"
    _TOOLS = frozenset({"story_start", "story_context", "story_manuscript"})

    async def _at(self, tmp_path: Path, model: ScriptedModel, stage: str) -> ToolContext:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        while out["stage"] != stage:
            out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        return ctx

    async def test_the_reply_names_what_taking_the_outline_kept_once(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: self.kept[key] = list(dict.fromkeys([*self.kept[key], *new]))
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/start.py :: note[NOTE_ASKS_KEY] = [*note.get(NOTE_ASKS_KEY, []), ko, en]
        Becomes: pass
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        out = await _ok(
            _tool(model),
            ctx,
            action="continue",
            go_on=True,
            must_have=[self._MUST],
            avoid=[self._AVOID],
        )

        note = out[REPLY_NOTE_KEY]
        said = [line for line in note["ko"].splitlines() if self._MUST in line]
        assert len(said) == 1 and self._AVOID in said[0]
        assert self._MUST in note["en"] and self._AVOID in note["en"]
        # Marked as the stop's question lines are: a reply that already says it in the same
        # words does not get it again.
        record = ToolExecutionRecord(tool_name="story_start", output=out)
        added = reply_notes([record], korean=True, reply=f"알겠습니다.\n{said[0]}")
        assert added and self._MUST not in added[0]

    async def test_the_said_back_line_says_how_to_change_it_now(self, tmp_path: Path) -> None:
        """#2342 item 1: a misread change is undone in one turn only if the line says how.

        Killed by: examples/story/src/uclone_x_story/start_notes.py :: ko += f" {_KEPT_CHANGE_NOW[0]}"
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/start_notes.py :: en += f" {_KEPT_CHANGE_NOW[1]}"
        Becomes: pass
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        out = await _ok(_tool(model), ctx, action="continue", go_on=True, must_have=[self._MUST])

        note = out[REPLY_NOTE_KEY]
        [ko] = [line for line in note["ko"].splitlines() if self._MUST in line]
        [en] = [line for line in note["en"].splitlines() if self._MUST in line]
        assert ko.endswith("지금 바꾸고 싶으시면 말씀해 주세요.")
        assert en.endswith("Tell me now if you would like it changed instead.")

    async def test_taking_with_nothing_asked_says_nothing_kept(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_notes.py :: if not any(kept.get(k) for k in ("must_have", "avoid")):
        Becomes: if False:
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        out = await _ok(_tool(model), ctx, action="continue", go_on=True)

        assert NOTE_ASKS_KEY not in out[REPLY_NOTE_KEY]
        assert "반영하겠습니다" not in out[REPLY_NOTE_KEY]["ko"]

    async def test_rewrite_parameter_rewrites_flagged_scenes(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start.py :: rewrite=params.rewrite,
        Becomes: rewrite=False,
        """
        model = ScriptedModel(
            scenes=[_DEAD_SCENE, _SCENE, _SCENE.replace("올랐다", "올랐다고 꿈꾸었다")],
            facts=[[_HARIN_DEAD], [_HARIN_BACK]],
        )
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        story_id = out["open_story_id"]

        again = await _ok(
            _tool(model), _ctx(tmp_path, story=story_id), action="continue", rewrite=True
        )

        assert "rewrite" in again["done_now"] and again["rewritten"] == ["ch01.s02"]

    async def test_waiting_turn_aid_tells_writer_not_to_call_story_start_on_questions(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/scene_turn.py :: "current stop, answer them in chat and do not call story_start.",
        Becomes: "",
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        aid = SceneTurnHook().turn_aid(
            message="주인공이 누구야?",
            story_id=str(ctx.story_id),
            room_id=ROOM,
            workspace_root=tmp_path,
            tool_names=self._TOOLS,
        )

        assert isinstance(aid, StartTurnAid)
        assert "answer them in chat and do not call story_start" in aid.section
        assert "'choice'" in aid.section and "'changes'" in aid.section

    async def test_avoid_without_choice_redraws_premises_even_when_go_on_is_true(
        self, tmp_path: Path
    ) -> None:
        """#2353: when the model passes `avoid` without `choice` at `choose_premise`,
        even with `go_on=True`, three new premises are drawn instead of taking premise #1.

        Killed by: examples/story/src/uclone_x_story/start.py :: changes = (params.changes or "").strip() or ", ".join(wanted["avoid"])
        Becomes: changes = (params.changes or "").strip()
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)
        drawn = len([p for p in model.prompts if "Write one premise" in p])

        again = await _ok(
            _tool(model),
            ctx,
            action="continue",
            go_on=True,
            must_have=["우주 정거장"],
            avoid=["리차드 지도자"],
        )

        assert again["stage"] == "choose_premise" and "chosen" not in again
        assert len([p for p in model.prompts if "Write one premise" in p]) == drawn + 3
        assert "리차드 지도자" in again["brief"]["avoid"]
        assert "우주 정거장" in again["brief"]["must_have"]

    async def test_avoided_cast_is_dropped_and_named_in_outline_revision(
        self, tmp_path: Path
    ) -> None:
        """#2353: characters/places matching `avoid` are removed from `state.cast` and
        explicitly listed in the `_revise_outline` prompt when `changes` is generic.

        Killed by: examples/story/src/uclone_x_story/start.py :: kept_cast = _without_avoided_cast(self.state.cast, self.state.brief.avoid)
        Becomes: kept_cast = list(self.state.cast)
        Killed by: examples/story/src/uclone_x_story/start_outline.py :: asked_feedback = _feedback_with_brief(feedback, state.brief)
        Becomes: asked_feedback = feedback
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        out = await _ok(
            _tool(model),
            ctx,
            action="continue",
            changes="개요 완전 재구성",
            must_have=["여성 정비사"],
            avoid=["먹물"],
        )

        assert out["stage"] == "review_outline" and "outline_revised" in out["done_now"]
        cast_names = [c["name"] for c in _state(tmp_path, str(ctx.story_id))["cast"]]
        assert "먹물" not in cast_names and "하린" in cast_names
        asked = _last(model, REVISE_MARK)
        assert "Must NOT include (remove any mention of these): 잔인한 장면, 먹물" in asked
        assert "Must include: 고양이, 여성 정비사" in asked


class TestTheStartAfterTheWriterReadsTheAnswer:
    """#2360: the 2026-10-15 live recheck, after #2352 moved reading the person's answer
    into the Writer's own parameters."""

    async def _at(self, tmp_path: Path, model: ScriptedModel, stage: str) -> ToolContext:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        while out["stage"] != stage:
            taken = {"choice": 1} if out["stage"] == "choose_premise" else {}
            out = await _ok(_tool(model), ctx, action="continue", go_on=True, **taken)
        return ctx

    async def test_go_on_alone_does_not_take_a_premise(self, tmp_path: Path) -> None:
        """picky T3: "제안된 세 가지 발상 모두 마음에 들지 않아." came as `go_on` with only
        `must_have`, and premise 1 was taken. A premise is taken only by its number; `go_on`
        alone is refused.

        Killed by: examples/story/src/uclone_x_story/start.py :: if number is None and self.state.quick:
        Becomes: if number is None and (self.state.quick or go_on):
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])

        refused = await _tool(model).execute({"action": "continue", "go_on": True}, ctx)

        assert not refused.success and "'choice'" in (refused.error or "")
        assert "'changes'" in (refused.error or "")
        kept = _state(tmp_path, out["open_story_id"])
        assert kept["stage"] == "choose_premise" and kept.get("chosen") is None
        taken = await _ok(_tool(model), ctx, action="continue", choice=2, go_on=True)
        assert taken["chosen"] == 2

    async def test_wants_with_no_number_at_the_premises_draw_new_ones(self, tmp_path: Path) -> None:
        """The #2360 live check: refused for want of a number, the Writer sent the same
        `go_on` and `must_have` 12 times over 6 turns, and the premise stop never moved.
        Wants with no number are what the person wants instead: new premises are drawn
        with them, and nothing is taken.

        Killed by: examples/story/src/uclone_x_story/start.py :: changes = ", ".join(wanted["must_have"])
        Becomes: pass
        """
        model = ScriptedModel(later_premises=[_REDRAWN])
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        before = [p["title"] for p in out["premises"]]

        result = await _tool(model).execute(
            {"action": "continue", "go_on": True, "must_have": ["외로운 정비사"]}, ctx
        )

        assert result.success, result.error
        assert isinstance(result.output, dict)
        redrawn = cast("dict[str, Any]", result.output)
        assert redrawn["stage"] == "choose_premise" and redrawn["done_now"] == ["premises"]
        assert redrawn.get("chosen") is None
        assert [p["title"] for p in redrawn["premises"]] != before
        assert "외로운 정비사" in redrawn["brief"]["must_have"]

    async def test_a_yes_with_a_request_at_the_outline_settles_it(self, tmp_path: Path) -> None:
        """The redraw on `go_on` with `must_have` and no number is the premise stop's only:
        at the outline, "yes, and add X" settles the plot and keeps X (#2364 review).

        Killed by: examples/story/src/uclone_x_story/start.py :: at_premises = state.stage == "choose_premise"
        Becomes: at_premises = True
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        out = await _ok(_tool(model), ctx, action="continue", go_on=True, must_have=["등불"])

        assert out["stage"] == "review_chapter" and out.get("plot_locked") is True
        assert "outline_revised" not in out["done_now"] and "premises" not in out["done_now"]
        assert "등불" in out["brief"]["must_have"]

    async def test_the_outline_stop_says_what_is_not_made_yet(self, tmp_path: Path) -> None:
        """mind_changer T4: "개요를 수정하여 저장했습니다. … 이제 첫 번째 장의 장면 계획을
        보여드리겠습니다." with the stage still at the outline. The stop's note says what is
        not made yet, from the story's state.

        Killed by: examples/story/src/uclone_x_story/start_notes.py :: ko_not_yet, en_not_yet = _NOT_YET_OUTLINE
        Becomes: ko_not_yet, en_not_yet = "", ""
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        out = await _ok(_tool(model), ctx, action="continue", changes="반전을 더 일찍")

        assert out["stage"] == "review_outline" and "outline_revised" in out["done_now"]
        note = out[REPLY_NOTE_KEY]
        assert "줄거리는 아직 확정하지 않았고, 첫 장의 장면 계획도 아직 없습니다." in note["ko"]
        assert "no scene plan for the first chapter yet" in note["en"]

    async def test_a_chapters_plan_says_the_chapter_is_not_written_yet(
        self, tmp_path: Path
    ) -> None:
        """vague T5: chapter prose in the reply while `chapter_written` was 0.

        Killed by: examples/story/src/uclone_x_story/start_notes.py :: ko += f"{number}장은 아직 쓰지 않았습니다. "
        Becomes: pass
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        out = await _ok(_tool(model), ctx, action="continue", go_on=True)

        assert out["stage"] == "review_chapter" and out.get("chapter_written", 0) == 0
        assert "1장은 아직 쓰지 않았습니다." in out[REPLY_NOTE_KEY]["ko"]
        assert "Chapter 1 is not written yet." in out[REPLY_NOTE_KEY]["en"]

    async def test_a_request_already_kept_is_not_said_back_again(self, tmp_path: Path) -> None:
        """The live replies said "덧붙이신 요청은 … 반영하겠습니다" on every turn: the Writer
        passes the same `must_have` each time, and nothing new was kept.

        Killed by: examples/story/src/uclone_x_story/start.py :: new = [t for t in _texts(answer.get(key)) if t not in known]
        Becomes: new = _texts(answer.get(key))
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")
        first = await _ok(_tool(model), ctx, action="continue", changes="x", must_have=["등불"])
        assert "‘등불’" in first[REPLY_NOTE_KEY]["ko"]

        again = await _ok(_tool(model), ctx, action="continue", changes="y", must_have=["등불"])

        assert "덧붙이신 요청" not in again[REPLY_NOTE_KEY]["ko"]

    async def test_the_story_id_and_stage_the_writer_adds_are_taken_and_ignored(
        self, tmp_path: Path
    ) -> None:
        """mind_changer T2: `{"action": "continue", "go_on": true, "story_id": "story-c606"}`
        was refused; 7 such refusals in one run, a step each.

        Killed by: examples/story/src/uclone_x_story/start_state.py :: story_id: str | None = Field(
        Becomes: story_ref: str | None = Field(
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])

        taken = await _ok(
            _tool(model),
            ctx,
            action="continue",
            choice=1,
            story_id="story-elsewhere",
            stage="review_outline",
        )

        assert taken["story_id"] == out["open_story_id"] and taken["chosen"] == 1
        refused = await _tool(model).execute({"action": "continue", "choise": 2}, ctx)
        assert not refused.success  # any other unknown argument is still refused

    async def test_a_reply_that_listed_the_genres_gets_only_the_question(
        self, tmp_path: Path
    ) -> None:
        """vague T1: the model's "1. 로맨스 (추천) / 2. 미스터리 / 3. 판타지", then the
        code's "요청에 장르가 없어 장르부터 제안합니다. / 1. 로맨스 (추천) …".

        Killed by: examples/story/src/uclone_x_story/start.py :: note[NOTE_LISTS_KEY] = _listed_genres(state)
        Becomes: pass
        """
        model = ScriptedModel(brief=_BARE_BRIEF)
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=BARE)
        assert out["stage"] == "choose_genre"
        labels = [genre_label(g, "ko") for g in out["genre_options"]]
        listed = "\n".join(f"{n}. **{g}**" for n, g in enumerate(labels, start=1))
        record = ToolExecutionRecord(tool_name="story_start", output=out)

        (note,) = reply_notes([record], korean=True, reply=f"장르를 골라 주세요.\n{listed}")

        assert not any(f"{n}. {g}" in note for n, g in enumerate(labels, start=1)), note
        assert note.startswith("‘네’라고 하시면"), note


class TestTheStopTurnFindingsOfTheLiveRun:
    """#2371: what the 2026-10-15 live `writer_stop_turn` run (#2370) found the tool saying
    to the Writer, or not saying."""

    async def _at(self, tmp_path: Path, model: ScriptedModel, stage: str) -> ToolContext:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        while out["stage"] != stage:
            out = await _ok(_tool(model), ctx, action="continue", go_on=True)
        return ctx

    async def test_go_on_with_changes_says_nothing_was_settled(self, tmp_path: Path) -> None:
        """Item 1: `go_on` and `changes` together revised the outline, and the Writer said
        the plot was settled. The change is made and `go_on` is not taken (#2355 lets a
        change win over `go_on`), and the result says so first.

        Killed by: examples/story/src/uclone_x_story/start_notes.py :: result["go_on_not_taken"] = True
        Becomes: pass
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_outline")

        out = await _ok(
            _tool(model),
            ctx,
            action="continue",
            go_on=True,
            changes="2장 제목을 '등대의 손님'으로 수정합니다.",
        )

        assert out["stage"] == "review_outline" and "plot_locked" not in out
        assert out["go_on_not_taken"] is True
        assert out["next"].startswith("Nothing was settled")

    async def test_a_quick_call_that_revises_the_plan_and_writes_says_no_such_thing(
        self, tmp_path: Path
    ) -> None:
        """#2372 review: on the quick path the plan is revised and the chapter written in the
        same call; a result that says nothing was settled contradicts it. Only a call that
        did not move on says so.

        Killed by: examples/story/src/uclone_x_story/start_notes.py :: and not done & _MOVED
        Becomes: and True
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "review_chapter")

        out = await _ok(
            _tool(model),
            ctx,
            action="continue",
            go_on=True,
            changes="1장 첫 장면을 밤으로 바꾸고 바로 써 줘",
        )

        assert out["chapter_written"] == 1 and "chapter" in out["done_now"]
        assert "go_on_not_taken" not in out
        assert not out["next"].startswith("Nothing was settled")

    async def test_the_persons_words_in_request_reach_the_redraw(self, tmp_path: Path) -> None:
        """Item 2: 'changes' held the placeholder "premises rejected" and the person's
        words were in 'request', which a 'continue' ignored. A 'continue' that carries
        'request' with a change takes those words as the change.

        Killed by: examples/story/src/uclone_x_story/start.py :: params = params.model_copy(update={"changes": params.request.strip()})
        Becomes: pass
        """
        model = ScriptedModel()
        ctx = await self._at(tmp_path, model, "choose_premise")
        words = "우주 정거장에 혼자 남은 정비사 이야기로 다시 보여 주세요."

        await _ok(_tool(model), ctx, action="continue", changes="premises rejected", request=words)

        prompt = [p for p in model.prompts if "Write one premise" in p][-1]
        assert words in prompt

    async def test_no_check_code_reaches_the_writer(self, tmp_path: Path) -> None:
        """Item 3: "이 설정은 현재 개요의 \"twist_missing\"을 해결하며" -- the Writer was given
        the outline check's codes. The result and the notes say each in words.

        Killed by: examples/story/src/uclone_x_story/start.py :: result["outline_check"] = _plain_check(state.outline_check)
        Becomes: result["outline_check"] = state.outline_check
        """
        chapters = _three(_EARLY)
        chapters[1].append(_chapter("안개가 오는 밤", "등대의 불", _EARLY + " 그리고 운다."))
        model = ScriptedModel(
            spine=_THREE_SPINE,
            chapters=chapters,
            checks=[
                ("state the core fact of the twist", _EARLY),
                ("state the core fact of the twist", "젖은 발자국"),
            ],
        )
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_tool(model), ctx, action="continue", choice=1)
        outline_out = await _ok(_tool(model), ctx, action="continue")

        shown = json.dumps(outline_out, ensure_ascii=False)
        for code in (
            "twist_early",
            "twist_missing",
            "resolved_early",
            "no_twist",
            "missing_chapters",
        ):
            assert code not in shown, code
        assert "Chapter 1 reveals the twist too early" in outline_out[REPLY_NOTE_KEY]["en"]
        assert outline_out["outline_check"]["final"] == [
            {"problem": "Chapter 1 reveals the twist too early", "chapter": 1}
        ]
        # The record keeps the codes, for the eval and the flow.
        assert _state(tmp_path, out["open_story_id"])["outline_check"]["final"] == [
            {"kind": "twist_early", "chapter": 1}
        ]

    async def test_an_empty_codex_says_the_story_has_started(self, tmp_path: Path) -> None:
        """Item 4: at the premises the codex is empty, and "It has no entries yet" read as a
        story not yet started: "이 질문에 답변하기 위해 먼저 이야기가 시작되어야 합니다."

        Killed by: examples/story/src/uclone_x_story/tools.py :: started=_started_line(StoryWork.open_in(context)),
        Becomes: started="",
        """
        ctx = await self._at(tmp_path, ScriptedModel(), "choose_premise")

        got = await StoryCodexTool().execute({"action": "get", "entry_id": "등대지기_소녀"}, ctx)

        assert not got.success and got.error is not None
        assert "It has no entries yet." in got.error
        assert "has been started" in got.error and "the three premises proposed" in got.error


class TestNotesNameNoInternals:
    """#2382: a note the Writer reads names an entry by its name, in plain words: no id,
    no exception text; the guards #2381 added hold when their call raises."""

    async def test_a_new_character_that_does_not_fit_is_named_in_plain_words(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_cast.py :: f"'{person['name']}' was not proposed: what was read for them does not fit a codex entry."
        Becomes: f"'{person['name']}' was not proposed: 1 validation error for NewEntryParams"
        """
        real = start_module.new_entry_draft

        def refusing(asked: NewEntryParams, **kwargs: Any) -> Proposal:
            if asked.name == "도윤":
                NewEntryParams.model_validate({"kind": "nothing", "name": "도윤"})
            return real(asked, **kwargs)

        monkeypatch.setattr(start_module, "new_entry_draft", refusing)
        model = ScriptedModel(details=_FATHER_PLAN)
        story_id, ctx = await _to_outline(tmp_path, model)
        await _ok(_tool(model), ctx, action="continue", go_on=True)
        work = _work(tmp_path, story_id)
        [harin] = [p for p, _ in work.proposals()[0] if p.entry_id == "harin"]
        apply_proposal(work, harin.id, room_id=ROOM, decided_in="story_view")
        written = await _ok(
            _tool(model), _ctx(tmp_path, story=story_id), action="continue", go_on=True
        )
        said = " ".join(written["notes"])
        assert "'도윤' was not proposed" in said
        for internal in ("validation error", "input_value", "pydantic", "Input should"):
            assert internal not in said, internal

    def test_an_entry_whose_fields_do_not_fit_is_named_by_its_name(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_details.py :: raise StoryError(f"'{name}' was not proposed: it does not fit a codex entry.") from exc
        Becomes: raise StoryError(f"'{draft.entry_id}' was not proposed: it does not fit.") from exc
        """
        draft = new_entry_draft(
            NewEntryParams(kind="threads", name="안개섬 등대 (줄거리)", profile="등대의 약속."),
            room_id=ROOM,
            context=_ctx(tmp_path),
        )
        with pytest.raises(StoryError) as refused:
            _with_entry_fields(draft, {"planted_in": "not an id!"})
        assert "안개섬 등대 (줄거리)" in str(refused.value)
        assert draft.entry_id not in str(refused.value)

    async def test_a_renamed_thread_whose_draft_fails_leaves_a_note_and_the_start_goes_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = start_module._with_entry_fields  # pyright: ignore[reportPrivateUsage]

        def refusing(draft: Proposal, fields: Mapping[str, str | None]) -> Proposal:
            if draft.new_entry().name.endswith("(줄거리)"):
                raise StoryError("'안개섬 등대 (줄거리)' was not proposed: it does not fit.")
            return real(draft, fields)

        monkeypatch.setattr(start_module, "_with_entry_fields", refusing)
        kg: dict[str, Any] = {
            "threads": [
                {
                    "name": "안개섬 등대",
                    "profile": "등대에 얽힌 약속.",
                    "planted_chapter": 1,
                    "pay_off_chapter": 2,
                }
            ],
            "items": [],
            "transitions": {},
        }
        model = ScriptedModel(kg=kg)
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        assert out["chapter_written"] == 1
        assert any(
            "(줄거리)' was not proposed" in n
            for n in _state(tmp_path, str(out["open_story_id"]))["notes"]
        )

    def test_a_scene_title_is_asked_of_an_outline_that_cannot_be_read(self) -> None:
        class _Unreadable:
            def outline(self) -> object:
                raise StoryFileError("outline.yaml could not be read.")

        run = cast(Any, SimpleNamespace(work=_Unreadable()))
        assert _Run._scene_title(run, "ch01.s01") == "a scene"  # pyright: ignore[reportPrivateUsage]

    def test_an_empty_name_takes_nothing_out(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: if not name.strip():
        Becomes: if False:
        """
        assert _without_names("Annual 하린", ["", " ", "Ann"]) == "Annual 하린"
