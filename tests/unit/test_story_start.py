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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from uclone_x.story.arc import CHECK_MARKER, Arc, Spine, arc_for, chapter_problems, chapter_text
from uclone_x.story.context import CodexIndex, CodexItem
from uclone_x.story.continuity import CONTINUITY_MARKER, came_back, parse_extraction
from uclone_x.story.enrich import ENRICH_MARKER
from uclone_x.story.library import StoryLibrary
from uclone_x.story.plan_check import PLAN_MARKER
from uclone_x.story.proposals import apply_proposal, reject_proposal
from uclone_x.story.schemas import CharacterEntry
from uclone_x.story.skill_data import BUNDLED_DATA_ROOT, load_structure_templates_sourced
from uclone_x.story.start import (
    CHAPTER_PROMPT_MARK,
    CHAPTERS_PER_CALL,
    DEFAULTS,
    DETAILS_MARK,
    DETAILS_REVISE_MARK,
    ENGINES,
    KG_MARK,
    MAX_CAST_TRIES,
    MAX_PREMISE_BIGRAM,
    MAX_PREMISE_COSINE,
    REVISE_MARK,
    SPINE_MARK,
    START_FILE,
    StoryStartTool,
    abstract_item,
    asks_rewrite,
    bigram_overlap,
    brief_from,
    choice_in,
    cosine,
    genre_options,
    json_object,
    outline_from,
    premise_similarity,
    says_go_on,
    says_yes,
    scene_title,
    scenes_for,
    settles_plot,
    status_of,
    structure_for,
    wants_it_now,
    writes_chapter,
)
from uclone_x.story.work import StoryWork
from uclone_x.tools.models import REPLY_NOTE_KEY, NoIsolation, ToolContext, ToolResult

ROOM = "room_a"

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
    ) -> None:
        self.prompts: list[str] = []
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
        # The facts the continuity check reads from each written scene, in turn.
        self._facts = list(facts or [])
        self._scenes = list(scenes or [])
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
        self.prompts.append(prompt)
        if prompt.startswith(ENRICH_MARKER):
            grown: dict[str, Any] | Exception = (
                self._growth.pop(0) if self._growth else {"new": [], "changes": []}
            )
            if isinstance(grown, Exception):
                raise grown
            return json.dumps(grown, ensure_ascii=False)
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
            return "```json\n" + json.dumps(self._brief, ensure_ascii=False) + "\n```"
        if "Write one premise" in prompt:
            title, premise = _PREMISES[self._premise_calls % 3]
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
            return self._scenes.pop(0) if self._scenes else _SCENE
        if CONTINUITY_MARKER in prompt:
            facts = self._facts.pop(0) if self._facts else []
            return facts if isinstance(facts, str) else json.dumps({"facts": facts})
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


def _last(model: ScriptedModel, text: str) -> str:
    """The last prompt the model was sent that contains `text`."""
    return [p for p in model.prompts if text in p][-1]


def _ctx(workspace: Path, *, story: str | None = None) -> ToolContext:
    return ToolContext(
        agent_id="writer",
        session_id=f"sess_room__{ROOM}__writer",
        workspace_root=workspace,
        room_id=ROOM,
        story_id=story,
        isolation=NoIsolation(),
    )


def _tool(model: ScriptedModel, embedder: FakeEmbedder | None = None) -> StoryStartTool:
    return StoryStartTool(
        model_factory=lambda _context: model, embedder_factory=lambda _context: embedder
    )


async def _ok(tool: StoryStartTool, ctx: ToolContext, **args: Any) -> dict[str, Any]:
    result: ToolResult = await tool.execute(args, ctx)
    assert result.success, result.error
    assert isinstance(result.output, dict)
    return result.output


def _state(workspace: Path, story_id: str) -> dict[str, Any]:
    text = (workspace / "stories" / story_id / START_FILE).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _work(workspace: Path, story_id: str) -> StoryWork:
    return StoryWork(StoryLibrary(workspace), story_id)


class TestTheFlowStopsForThePerson:
    async def test_start_stops_at_the_three_premises_and_asks_for_a_number(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/start.py :: if state.quick:
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
        """Killed by: src/uclone_x/story/start.py :: if number is None and (self.state.quick or yes):
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
        """Killed by: src/uclone_x/story/start.py :: return next((p for p in self.premises if p.number == number), self.premises[0])
        Becomes: return self.premises[0]
        Killed by: src/uclone_x/story/start.py :: self._save(chosen=number)
        Becomes: pass
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)

        # A fresh tool each call: what the flow knows is in the story's files.
        cast = await _ok(_tool(model), ctx, action="continue", request="2번이요")
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
        chapter = await _ok(_tool(model), ctx, action="continue", request="네")
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
        """Killed by: src/uclone_x/story/start.py :: done.extend(await run.forward(choice=None, go_on=True))
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
        """Killed by: src/uclone_x/story/start.py :: if pending and not (go_on or self.state.quick):
        Becomes: if False:
        """
        model = ScriptedModel()
        story_id, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", request="좋아요, 첫 장 써 주세요")
        assert out["stage"] == "review_outline"
        assert out["done_now"] == ["plot"]  # the plot is settled; the chapter waits
        assert len(out["cast_waiting"]) == 3
        assert "아직 승인되지 않아 첫 장을 계획하지 않았습니다" in out[REPLY_NOTE_KEY]["ko"]
        assert "approval" in out["next"]
        assert _work(tmp_path, story_id).written_scenes() == []
        assert not any("Write scene" in p for p in model.prompts)

    async def test_saying_to_go_on_writes_it_and_says_so(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: go_on = bool(params.request and says_go_on(params.request))
        Becomes: go_on = False
        Killed by: src/uclone_x/story/start.py :: result["cast_assumed"] = state.cast_assumed
        Becomes: pass
        """
        model = ScriptedModel()
        story_id, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", request="제안대로 진행해 주세요")
        assert out["stage"] == "review_chapter" and out["details_chapter"] == 1
        assert "cast_waiting" not in out
        assert len(out["cast_assumed"]) == 3
        assert _work(tmp_path, story_id).written_scenes() == []  # the plan is shown first
        out = await _ok(_tool(model), ctx, action="continue", request="네")
        assert "승인된 것으로 보고 썼습니다" in out[REPLY_NOTE_KEY]["ko"]
        assert _work(tmp_path, story_id).written_scenes() == ["ch01.s01", "ch01.s02"]


class TestTheCast:
    async def test_each_character_starts_with_the_status_the_premise_gives(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/start.py :: facts: dict[str, Any] = {"status": status}
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
        """Killed by: src/uclone_x/story/start.py :: for _ in range(MAX_DRAFTS):
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
        """Killed by: src/uclone_x/story/start.py :: if not this_chapter:
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


class TestTheCastIsNeverSkipped:
    async def test_an_unreadable_cast_is_asked_again_told_why(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: for _ in range(MAX_CAST_TRIES):
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
        """Killed by: src/uclone_x/story/start.py :: if self.state.cast_failed:
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
        """Killed by: src/uclone_x/story/start.py :: retry = self.state.chosen if self.state.cast_failed else None
        Becomes: retry = None
        """
        model = ScriptedModel(cast_replies=["{}"] * MAX_CAST_TRIES)
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        failed = await _ok(_tool(model), ctx, action="continue", choice=2)
        assert failed["stage"] == "choose_premise"
        assert failed["cast_failed"] == "empty"
        assert "이름이 있는 인물이 한 명도 나오지 않았습니다" in failed[REPLY_NOTE_KEY]["ko"]
        again = await _ok(_tool(model), ctx, action="continue", request="다시 해 봐")
        assert again["stage"] == "review_cast"
        assert again["chosen"] == 2
        assert "cast_failed" not in again


class TestTheOutlineUsesAStructure:
    def _known(self) -> dict[str, Any]:
        found = load_structure_templates_sourced([BUNDLED_DATA_ROOT])
        return {k: v.item for k, v in found.items()}

    def test_the_named_structure_wins_and_the_genre_fills_only_what_is_left(self) -> None:
        """Killed by: src/uclone_x/story/start.py :: return structure_id, "genre"
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
        """Killed by: src/uclone_x/story/start.py :: acts = [act.id for act in kept_acts]
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
        """Killed by: src/uclone_x/story/arc.py :: if number < arc.twist_chapter:
        Becomes: if False:
        Killed by: src/uclone_x/story/arc.py :: if number >= arc.climax_chapter and self.resolution:
        Becomes: if self.resolution:
        Killed by: src/uclone_x/story/start.py :: f"Chapter {n}:\n{chapter_text(c)}" for n, c in enumerate(chapters, start=1)
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
        """Killed by: src/uclone_x/story/start.py :: for chapter in items[: len(acts) or MAX_CHAPTERS]:
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
        """Killed by: src/uclone_x/story/start.py :: kept += [f.record() for f in findings]
        Becomes: kept += []
        Killed by: src/uclone_x/story/start.py :: if rewrite and flagged:
        Becomes: if flagged:
        Killed by: src/uclone_x/story/start.py :: return codex_with(self.work.codex(), assumed)
        Becomes: return self.work.codex()
        """
        # 하린 is still a pending proposal: the check holds the scene to the cast assumed.
        model = ScriptedModel(scenes=[_DEAD_SCENE], facts=[[_HARIN_DEAD]])
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        assert out["stage"] == "review_chapter"
        [finding] = out["continuity"]
        assert finding["scene_id"] == "ch01.s01"
        assert finding["quote"] == _DEAD_LINE
        assert f"“{_DEAD_LINE}”" in finding["note"]
        assert "설정집에는 하린이 살아 있는 것으로" in finding["note"]
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
        found = _work(tmp_path, out["open_story_id"]).manuscript("ch01.s01")
        assert found is not None and _DEAD_LINE in found.text

    async def test_asked_to_rewrite_only_the_flagged_scene_is_written_again(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/start.py :: rewrite = bool(params.request and asks_rewrite(params.request))
        Becomes: rewrite = False
        """
        model = ScriptedModel(scenes=[_DEAD_SCENE], facts=[[_HARIN_DEAD]])
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        story_id = out["open_story_id"]
        again = await _ok(
            _tool(model), _ctx(tmp_path, story=story_id), action="continue", request="다시 써 줘"
        )
        assert again["done_now"] == ["rewrite"]
        assert again["rewritten"] == ["ch01.s01"]
        asked = [p for p in model.prompts if "Write scene" in p][-1]
        assert "disagree with the story's codex" in asked and _DEAD_LINE in asked
        found = _work(tmp_path, story_id).manuscript("ch01.s01")
        assert found is not None and found.text.strip() == _SCENE
        assert again["continuity"] == []  # checked again, and it agrees now
        assert "장면 1개를 다시 써서 저장했습니다" in again[REPLY_NOTE_KEY]["ko"]

    async def test_a_scene_whose_facts_cannot_be_read_is_not_called_consistent(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/start.py :: unread.append(scene_id)
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

    def test_words_that_ask_for_a_rewrite(self) -> None:
        assert asks_rewrite("그 장면 다시 써 줘")
        assert asks_rewrite("고쳐 주세요")
        assert asks_rewrite("please rewrite it")
        assert not asks_rewrite("다시 생각해 볼게요")
        assert not asks_rewrite("좋아요, 계속해 주세요")


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
    "changes": [],
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
    "changes": [],
}


class TestEachWrittenSceneGrowsTheCodex:
    """§1.1 row 7: what `story_start` writes is read for the codex, as a manuscript write is."""

    async def _first_chapter(
        self, tmp_path: Path, model: ScriptedModel
    ) -> tuple[dict[str, Any], str]:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)
        await _ok(_tool(model), ctx, action="continue", request="1번")
        await _ok(_tool(model), ctx, action="continue", request="네")
        await _ok(_tool(model), ctx, action="continue", request="네")  # the first plan
        out = await _ok(_tool(model), ctx, action="continue", request="네")
        return out, story_id

    async def test_a_written_chapter_proposes_what_its_scenes_add(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: await self._grow(scene.id, text, notes)
        Becomes: pass
        Killed by: src/uclone_x/story/start.py :: update.setdefault("growth_now", list(self.grown))
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
        """Killed by: src/uclone_x/story/enrich.py :: except Exception:
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
        """Killed by: src/uclone_x/story/start.py :: await self._grow(scene_id, text, notes)
        Becomes: pass
        """
        rewritten = f"{_SCENE} {_POSTMAN_LINE}"
        model = ScriptedModel(
            scenes=[_DEAD_SCENE, _SCENE, rewritten],
            facts=[[_HARIN_DEAD]],
            growth=[{"new": [], "changes": []}, {"new": [], "changes": []}, _POSTMAN_GROWTH],
        )
        out = await _ok(
            _tool(model), _ctx(tmp_path), action="start", request=REQUEST + " 바로 써줘."
        )
        again = await _ok(
            _tool(model),
            _ctx(tmp_path, story=out["open_story_id"]),
            action="continue",
            request="다시 써 줘",
        )
        assert again["done_now"] == ["rewrite"]
        assert len(again["codex_growth"]) == 1
        assert "보강안으로 올렸습니다" in again[REPLY_NOTE_KEY]["ko"]


class TestTheBrief:
    def test_only_what_the_request_left_open_is_filled_in(self) -> None:
        """Killed by: src/uclone_x/story/start.py :: fields[key] = value
        Becomes: fields[key] = DEFAULTS[key]
        """
        brief = brief_from({"genre": "호러", "tone": "", "length": "단편"}, REQUEST)
        assert brief.genre == "호러"
        assert brief.length == "단편"
        assert brief.tone == DEFAULTS["tone"]
        assert brief.audience == DEFAULTS["audience"]
        assert brief.from_request == ["genre", "length"]

    def test_a_reply_with_no_json_leaves_every_field_at_its_default(self) -> None:
        brief = brief_from(json_object("모르겠습니다"), REQUEST)
        assert brief.genre == DEFAULTS["genre"]
        assert brief.from_request == []
        assert brief.title


class TestThePremises:
    async def test_each_premise_is_its_own_call_with_its_own_cards(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: seed = secrets.randbelow(MAX_SEED + 1)
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
        """Killed by: src/uclone_x/story/start.py :: premises = [await draft(n, engines[n - 1]) for n in range(1, PREMISE_COUNT + 1)]
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
        """Killed by: src/uclone_x/story/start.py :: if len(still) < len(lacking):
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
        """Killed by: src/uclone_x/story/start.py :: if embedder is not None and texts:
        Becomes: if False:
        """
        embedder = FakeEmbedder(_APART)
        out = await _ok(
            _tool(ScriptedModel(), embedder), _ctx(tmp_path), action="start", request=REQUEST
        )
        assert embedder.calls == [[p for _, p in _PREMISES]]
        similarity = out["premise_similarity"]
        assert similarity["measure"] == "embedding_cosine"
        assert similarity["pairs"] == [0.0, 0.0, 0.0]
        assert similarity["limit"] == MAX_PREMISE_COSINE

    async def test_two_premises_alike_are_drawn_again_once(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: if not similarity.distinct and len(engines) > PREMISE_COUNT:
        Becomes: if False:
        """
        alike = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        model = ScriptedModel()
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
        assert state["premises"][1]["premise"] == _PREMISES[0][1]
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
        r"""Killed by: src/uclone_x/story/start.py :: for attempt in (body, _STRAY_QUOTE.sub(r"\1", body)):
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

    def test_quick_phrases_and_choices(self) -> None:
        assert wants_it_now("그냥 바로 써줘")
        assert wants_it_now("알아서 써 줘")
        assert not wants_it_now("천천히 같이 만들어 봐요")
        assert choice_in("2번이 좋아요") == 2
        assert choice_in("1번이랑 3번 중에") is None
        assert choice_in(None) is None

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

    def test_words_that_say_go_on_without_approving(self) -> None:
        assert says_go_on("제안대로 진행해 주세요")
        assert says_go_on("승인하지 않아도 되니 그대로 써 주세요")
        assert says_go_on("바로 써줘")
        assert not says_go_on("좋아요, 첫 장 써 주세요")

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
        """Killed by: src/uclone_x/story/start.py :: if _CLAUSE_END.search(title) or len(title) > MAX_SCENE_TITLE:
        Becomes: if False:
        Killed by: src/uclone_x/story/start.py :: if m.group(1) not in _PRONOUN_OWNERS:
        Becomes: if True:
        """
        # A clause cut off at its connective, not the summary's opening.
        summary = "하린은 자신의 몸에서 기척을 느끼고, 장로의 계략을 깨닫고, 맞서기로 결심한다."
        title = scene_title("이상한 기척을 느끼며,", summary)
        assert title == "장로의 계략"
        assert title != summary and not summary.startswith(title)

    def test_the_summary_opening_is_not_a_title(self) -> None:
        """Killed by: src/uclone_x/story/start.py :: return len(title) >= 8 and summary.startswith(title) and title != summary
        Becomes: return False
        """
        title = scene_title(
            "먹물이 방울을 울리며 계단",
            "먹물이 방울을 울리며 계단을 오르는 동안 등대의 불빛이 꺼진다.",
        )
        assert title == "등대의 불빛"

    def test_without_an_of_phrase_the_clause_is_cut_short_without_its_subject(self) -> None:
        """Killed by: src/uclone_x/story/start.py :: head = head[1:]  # the subject ("이진호는") names the scene no better than its cast
        Becomes: pass  # the subject ("이진호는") names the scene no better than its cast
        """
        assert scene_title("", "하린은 문을 열고 들어가며, 먹물을 부른다.") == "문을 열고 들어가"
        assert scene_title("", "") == ""


class TestTheOutlineEscalates:
    def test_the_twist_is_never_before_the_middle_and_never_last(self) -> None:
        """Killed by: src/uclone_x/story/arc.py :: twist = marked or min(middle, count - 1)
        Becomes: twist = min(middle, count - 1)
        Killed by: src/uclone_x/story/arc.py :: climax = count if count == twist + 1 else count - 1
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
        """Killed by: src/uclone_x/story/arc.py :: if number < self.twist_chapter:
        Becomes: if False:
        """
        arc = arc_for([["introduction"], ["development"], ["twist"], ["reconciliation"]])
        assert arc is not None
        assert "the twist stays hidden" in arc.does(2)
        assert "the twist stays hidden" not in arc.does(3)
        assert "reveal the twist" in arc.does(3)
        assert "the conflict stays open" in arc.does(3) and "stays open" not in arc.does(4)

    async def test_the_check_finds_an_early_twist_and_an_early_ending(self) -> None:
        """Killed by: src/uclone_x/story/arc.py :: if shown and number < arc.twist_chapter:
        Becomes: if False:
        Killed by: src/uclone_x/story/arc.py :: if await _settled(model, spine.conflict, text):
        Becomes: if False:
        Killed by: src/uclone_x/story/arc.py :: return reply.strip()[:1].upper() == "B"
        Becomes: return reply.strip().casefold().startswith("yes")
        Killed by: src/uclone_x/story/arc.py :: if again:
        Becomes: if False:
        Killed by: src/uclone_x/story/arc.py :: if spine.conflict.strip() and arc.twist_chapter <= number < arc.climax_chapter:
        Becomes: if spine.conflict.strip() and number < arc.climax_chapter:
        Killed by: src/uclone_x/story/arc.py :: "Does this chapter state the core fact of the twist, even in different words "
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
        """Killed by: src/uclone_x/story/start.py :: if first:
        Becomes: if False:
        Killed by: src/uclone_x/story/start.py :: chapter = retried
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
        check = outline_out["outline_check"]
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
        """Killed by: src/uclone_x/story/start.py :: if len(later) < len(first):
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
        check = outline_out["outline_check"]
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
        """Killed by: src/uclone_x/story/start.py :: relations[other].setdefault(holder_id, INVERSE_ROLE[role])
        Becomes: pass
        Killed by: src/uclone_x/story/start.py :: gender = family_gender.get(name, "")
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
        """Killed by: src/uclone_x/story/start.py :: holder, holder_id, other, other_id = name, own, relative, relative_id
        Becomes: pass
        Killed by: src/uclone_x/story/start.py :: if stated.get(name.casefold()) != word_gender:
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
        """Killed by: src/uclone_x/story/continuity.py :: axioms=[*axioms, *identity_axioms(codex, submitted)],
        Becomes: axioms=axioms,
        Killed by: src/uclone_x/story/continuity.py :: held.append(SubmittedFact(subject, "gender", said, fact.quote))
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
        asked = _last(model, CONTINUITY_MARKER)
        assert "- family: what the subject is to" in asked and "- gender:" in asked

    async def test_a_brother_called_a_son_is_a_kinship_contradiction(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/continuity.py :: held.append(SubmittedFact(subject, relation_predicate(other), role, fact.quote))
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
        """Killed by: src/uclone_x/story/continuity.py :: if not _names_family(fact.quote):
        Becomes: if False:
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
        """Killed by: src/uclone_x/story/start.py :: elif checked and unread:
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
        r"""Killed by: src/uclone_x/story/continuity.py :: fields = _fields_read(STRAY_QUOTE.sub(r"\1", body))
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
        """Killed by: src/uclone_x/story/start.py :: go_on = bool(params.request and says_go_on(params.request)) or yes
        Becomes: go_on = bool(params.request and says_go_on(params.request))
        Killed by: src/uclone_x/story/start.py :: if number is None and (self.state.quick or yes):
        Becomes: if number is None and self.state.quick:
        Killed by: src/uclone_x/story/start.py :: if yes:
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
            out = await _ok(_seeded(model), ctx, action="continue", request="네")
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

        Killed by: src/uclone_x/story/start.py :: "고칠 수 있습니다. ‘네’라고 하시면 개요를 만들겠습니다. 첫 장은 제안을 모두 "
        Becomes: "고칠 수 있습니다. 첫 장은 제안을 승인하신 뒤에 쓰겠습니다. 첫 장은 제안을 모두 "
        Killed by: src/uclone_x/story/start.py :: "chapter waits until every proposal is decided; but if you just say yes at the "
        Becomes: "chapter waits for your approval. "
        """
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        cast_out = await _ok(_tool(model), ctx, action="continue", request="네")
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

        outline = await _ok(_tool(model), ctx, action="continue", request="네")
        assert outline["stage"] == "review_outline"
        assert _work(tmp_path, out["open_story_id"]).written_scenes() == []
        plan = await _ok(_tool(model), ctx, action="continue", request="네")
        assert plan["details_chapter"] == 1 and len(plan["cast_assumed"]) == 3
        chapter = await _ok(_tool(model), ctx, action="continue", request="네")
        assert chapter["chapter_written"] == 1
        assert len(chapter["cast_assumed"]) == 3
        work = _work(tmp_path, out["open_story_id"])
        assert all(p.status == "pending" for p, _ in work.proposals()[0])

    async def test_a_number_beats_yes(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: if choice is not None and choice <= len(options):
        Becomes: if False:
        """
        model = ScriptedModel(brief=_BARE_BRIEF)
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=BARE)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        third = out["genre_options"][2]
        premises = await _ok(_seeded(model), ctx, action="continue", request="네, 3번이요")
        assert premises["brief"]["genre"] == third
        cast = await _ok(_seeded(model), ctx, action="continue", request="네 2번으로 할게요")
        assert cast["chosen"] == 2

    async def test_a_request_that_is_more_than_yes_does_not_skip_the_cast_wait(
        self, tmp_path: Path
    ) -> None:
        model = ScriptedModel()
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        await _ok(_tool(model), ctx, action="continue", request="네")
        await _ok(_tool(model), ctx, action="continue", request="네")
        waiting = await _ok(_tool(model), ctx, action="continue", request="이대로 확정해 주세요")
        assert waiting["stage"] == "review_outline" and len(waiting["cast_waiting"]) == 3

    def test_only_a_reply_made_of_agreement_is_yes(self) -> None:
        """Killed by: src/uclone_x/story/start.py :: return bool(tokens) and all(_YES_TOKEN.fullmatch(t) for t in tokens)
        Becomes: return bool(tokens) and any(_YES_TOKEN.fullmatch(t) for t in tokens)
        """
        for words in ("네", "응", "좋아요!", "그래", "yes", "OK", "진행", "네, 진행해 주세요"):
            assert says_yes(words), words
        for words in ("", "2번", "좋아요, 첫 장 써 주세요", "네 로맨스로요", "아니요"):
            assert not says_yes(words), words


class TestTheGenreIsProposedOnlyWhenLeftOpen:
    async def test_a_stated_genre_is_never_replaced(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: [] if "genre" in brief.from_request else genre_options(sorted(tables), self._rng)
        Becomes: genre_options(sorted(tables), self._rng)
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
            out = await _ok(tool, ctx, action="continue", request="네")
        assert out["stage"] == "review_chapter" and out["brief"]["genre"] == "호러"
        assert _story_genre(tmp_path, out["story_id"]) == "호러"

    async def test_a_named_genre_replaces_the_proposal_before_premises_are_drawn(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/start.py :: named = named_genre(words)
        Becomes: named = None
        """
        model = ScriptedModel(brief=_BARE_BRIEF)
        out = await _ok(_seeded(model, 3), _ctx(tmp_path), action="start", request=BARE)
        ctx = _ctx(tmp_path, story=out["open_story_id"])
        wanted = next(g for g in ("로맨스", "미스터리") if g != out["genre_options"][0])
        premises = await _ok(_seeded(model), ctx, action="continue", request=f"{wanted}로 해 줘")
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
            {"action": "continue", "request": "음, 글쎄요"}, _ctx(tmp_path, story=story_id)
        )
        assert not result.success and result.error is not None and "genre" in result.error
        assert _state(tmp_path, story_id) == before

    async def test_asked_at_once_it_takes_the_drawn_genre_and_says_so(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: if first and state.quick and state.genre_options and not state.genre_answered:
        Becomes: if False:
        """
        model = ScriptedModel(brief=_BARE_BRIEF)
        out = await _ok(_seeded(model), _ctx(tmp_path), action="start", request=BARE + " 바로 써줘")
        genre = out["genre_options"][0]
        assert out["stage"] == "review_chapter" and out["brief"]["genre"] == genre
        assert f"요청에 장르가 없어 {genre}로 정해 썼습니다" in out[REPLY_NOTE_KEY]["ko"]

    def test_the_proposal_is_drawn_at_random_from_the_muse_tables(self) -> None:
        """Killed by: src/uclone_x/story/start.py :: return [GENRE_LABELS[k] for k in rng.sample(keys, min(GENRE_OPTIONS, len(keys)))]
        Becomes: return [GENRE_LABELS[k] for k in keys[:GENRE_OPTIONS]]
        """
        tables = ["fantasy", "horror", "mystery", "romance", "science-fiction", "western"]
        drawn = [genre_options(tables, random.Random(seed)) for seed in range(20)]
        assert drawn[0] == genre_options(tables, random.Random(0))  # the seed decides
        assert len({d[0] for d in drawn}) > 1  # the recommendation is not fixed
        for options in drawn:
            assert len(options) == 3 == len(set(options))
            assert set(options) <= {"판타지", "호러", "미스터리", "로맨스", "SF"}
        assert genre_options(["romance"], random.Random(0)) == ["로맨스"]


class TestAGenreCountsAsStatedOnlyWhenTheRequestNamesIt:
    async def test_a_genre_the_model_made_up_is_proposed_over(self, tmp_path: Path) -> None:
        """qwen3:8b read "소설 하나 써줘" as genre "일반", stated (2026-09-29).

        Killed by: src/uclone_x/story/start.py :: if key == "genre" and value and not names_genre(request, value):
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
        """Killed by: src/uclone_x/story/start.py :: if feedback and not (self.state.quick or settles_plot(feedback)):
        Becomes: if False:
        """
        model = ScriptedModel()
        story_id, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", request="반전을 더 일찍 드러내 줘")
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

        settled = await _ok(_tool(model), ctx, action="continue", request="네")
        assert settled["stage"] == "review_chapter" and settled["plot_locked"] is True
        assert settled["details_chapter"] == 1
        assert _work(tmp_path, story_id).written_scenes() == []

    async def test_an_unreadable_revision_keeps_the_outline_and_says_so(
        self, tmp_path: Path
    ) -> None:
        model = ScriptedModel(revised="개요를 고쳤습니다!")
        story_id, ctx = await self._at_outline(tmp_path, model)
        before = _work(tmp_path, story_id).outline()
        out = await _ok(_tool(model), ctx, action="continue", request="3장 빼 줘")
        assert out["stage"] == "review_outline"
        assert "반영하지 못해 개요를 그대로 두었습니다" in out[REPLY_NOTE_KEY]["ko"]
        assert _work(tmp_path, story_id).outline() == before

    async def test_yes_settles_the_plot_and_proposes_its_key_events(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: wanted += read
        Becomes: pass
        """
        model = ScriptedModel()
        story_id, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", request="네")
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
        out = await _ok(_tool(model), ctx, action="continue", request="이대로 확정해 주세요")
        assert out["stage"] == "review_outline" and len(out["cast_waiting"]) == 3
        assert out["plot_locked"] is True
        assert out[REPLY_NOTE_KEY]["ko"].startswith("줄거리를 확정했습니다.")
        again = await _ok(_tool(model), ctx, action="continue", request="그대로 진행해 주세요")
        assert again["stage"] == "review_chapter" and "plot" not in again["done_now"]
        assert "줄거리를 확정했습니다" not in again[REPLY_NOTE_KEY]["ko"]
        assert sum(KG_MARK in p for p in model.prompts) == 1

    async def test_an_unreadable_reading_proposes_the_conflict_and_twist_alone(
        self, tmp_path: Path
    ) -> None:
        model = ScriptedModel(kg="모르겠습니다")
        _, ctx = await self._at_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", request="네")
        assert [k["name"] for k in out["plot_kg"]] == ["중심 갈등", "반전"]
        assert "The settled outline's key events could not be read." in out["notes"]

    def test_only_agreement_or_asking_for_the_chapter_settles_it(self) -> None:
        """Killed by: src/uclone_x/story/start.py :: return bool(_LOCK.search(words)) or says_yes(words) or says_go_on(words)
        Becomes: return says_yes(words) or says_go_on(words)
        """
        for words in (
            "네",
            "확정해 주세요",
            "이대로 가요",
            "좋아요, 첫 장 써 주세요",
            "그대로 진행해",
        ):
            assert settles_plot(words), words
        for words in ("반전을 더 일찍", "주인공을 여자로", "3장 빼 줘", "1장을 더 짧게 해 줘"):
            assert not settles_plot(words), words


class TestTheNextChapterIsPlannedBeforeItIsWritten:
    """Stage 4b: after a chapter is written and checked, the next one's plan waits."""

    async def _after_first(
        self, tmp_path: Path, model: ScriptedModel
    ) -> tuple[dict[str, Any], str, ToolContext]:
        out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
        story_id = out["open_story_id"]
        ctx = _ctx(tmp_path, story=story_id)
        await _ok(_tool(model), ctx, action="continue", request="1번")
        await _ok(_tool(model), ctx, action="continue", request="네")
        await _ok(_tool(model), ctx, action="continue", request="네")  # the first plan
        out = await _ok(_tool(model), ctx, action="continue", request="네")
        return out, story_id, ctx

    async def test_the_next_chapter_is_planned_and_shown_before_it_is_written(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/start.py :: await self._propose_details(number + 1)
        Becomes: pass
        Killed by: src/uclone_x/story/start.py :: before = self._last_text(outline, number - 1)
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

        Killed by: src/uclone_x/story/start.py :: title = scene_title(_text(sc.get("title")), _text(sc.get("summary")))
        Becomes: title = _text(sc.get("title")) or _text(sc.get("summary"))[:30]
        Killed by: src/uclone_x/story/start.py :: + (f" — {s['summary']}" if s.get("summary") and s["summary"] != s.get("title") else "")
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
        """Killed by: src/uclone_x/story/start.py :: if said and not writes_chapter(said):
        Becomes: if False:
        Killed by: src/uclone_x/story/start.py :: if not self.state.quick:  # the revised plan is shown first
        Becomes: if False:  # the revised plan is shown first
        """
        model = ScriptedModel()
        _, story_id, ctx = await self._after_first(tmp_path, model)
        feedback = "첫 장면은 빼고 방울 소리를 따라가는 장면으로 바꿔 줘"
        out = await _ok(_tool(model), ctx, action="continue", request=feedback)
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
        """Killed by: src/uclone_x/story/start.py :: self._save(details_feedback=said, details_revision_failed=True, notes=notes)
        Becomes: self._save(details_feedback=said, details_revision_failed=False, notes=notes)
        """
        model = ScriptedModel(details_revised="고칠 수 없습니다")
        _, _, ctx = await self._after_first(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", request="더 무섭게 해 줘")
        assert [s["title"] for s in out["chapter_details"]] == ["젖은 발자국", "걷히는 안개"]
        assert "반영하지 못해 2장의 장면 계획을 그대로 두었습니다" in out[REPLY_NOTE_KEY]["ko"]

    async def test_yes_writes_the_planned_chapter_and_the_last_one_ends_the_flow(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/start.py :: await self._apply_details(number)
        Becomes: pass
        Killed by: src/uclone_x/story/start.py :: if number < len(outline.chapters):
        Becomes: if True:
        """
        model = ScriptedModel()
        _, story_id, ctx = await self._after_first(tmp_path, model)
        await _ok(_tool(model), ctx, action="continue", request="첫 장면은 빼 줘")
        out = await _ok(_tool(model), ctx, action="continue", request="네")
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
        assert "방울 소리" in written and _SCENE[-40:] in written  # after chapter 1's end
        assert out["chapter_written"] == 2 and out["chapter_scenes"] == ["ch02.s01"]
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "2장을 썼습니다. 장면 1개를 저장했습니다." in ko
        assert "마지막 장까지 모두 썼습니다" in ko
        assert "details_chapter" not in out

    async def test_write_words_write_it_and_other_words_are_feedback(self) -> None:
        for words in ("네", "써 줘", "2장 써 주세요", "이어서 써 줘", "계속", "continue"):
            assert writes_chapter(words), words
        for words in ("첫 장면은 빼 줘", "주인공 이름을 바꿔 줘", "하린이 죽지 않게 써 줘"):
            assert not writes_chapter(words), words

    async def test_an_unreadable_plan_shows_the_outlines_scenes(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: from_outline = not scenes
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
        """Killed by: src/uclone_x/story/start.py :: and done.count("chapter") < CHAPTERS_PER_CALL
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
            request="첫 장면은 빼 줘",
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
        """Killed by: src/uclone_x/story/start.py :: kept += [f.record() for f in held]
        Becomes: kept += []
        Killed by: src/uclone_x/story/start.py :: return findings + plan_findings(scene, text, reply, **where), True
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
            request="네",
        )
        assert "chapter" in again["done_now"] and "rewrite" not in again["done_now"]

    async def test_the_twist_is_asked_about_only_before_its_chapter(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: "twist_chapter": arc.twist_chapter if arc else None,
        Becomes: "twist_chapter": None,
        """
        model = ScriptedModel()
        out = await self._quick(tmp_path, model)
        twist_chapter = _state(tmp_path, out["open_story_id"])["arc"]["twist_chapter"]
        assert twist_chapter > 1
        assert _SPINE["twist"] in _last(model, PLAN_MARKER)

    async def test_a_planned_character_the_scene_never_names_is_shown(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: findings = absent_cast(codex, scene, text)
        Becomes: findings = []
        """
        alone = "하린은 등대 계단을 올랐다. 바람이 창을 두드렸다. 꼭대기에서 누군가 이름을 불렀다."
        model = ScriptedModel(scenes=[alone])
        out = await self._quick(tmp_path, model)
        notes = [f["note"] for f in out["continuity"]]
        assert "‘계단’ 장면 계획에 있던 먹물이 쓴 장면에는 나오지 않습니다." in notes

    async def test_an_unreadable_reading_is_never_called_consistent(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: not_held.append(scene_id)
        Becomes: pass
        """
        model = ScriptedModel(plans=["모르겠습니다", "모르겠습니다"])
        out = await self._quick(tmp_path, model)
        assert out["plan_unread"] == ["ch01.s01", "ch01.s02"]
        ko = out[REPLY_NOTE_KEY]["ko"]
        assert "장면 2개는 장면 계획과 대조하지 못했습니다" in ko
        assert "반전을 드러낼 때와도 어긋나는 곳을 찾지 못했습니다" not in ko

    async def test_a_scene_that_keeps_its_plan_is_said_to(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: if not shown and scenes and len(plan_unread) < len(scenes):
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


def _said(subject: str, key: str, value: str, quote: str) -> dict[str, Any]:
    return {
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
        """Killed by: src/uclone_x/story/start.py :: kept += [f.record() for f in back]
        Becomes: kept += []
        Killed by: src/uclone_x/story/start.py :: proposal.id in items and proposal.kind == "items"
        Becomes: False
        Killed by: src/uclone_x/story/continuity.py :: said = f"이 장면에서 {both[0]} 나오는데, {both[1]}도 나옵니다."
        Becomes: said = "이 문장이 설정집의 기록과 맞지 않습니다."
        """
        out = await self._clock(
            tmp_path,
            [
                _said("이진호", "status", "alive", _WALKS),
                _said("이진호", "possesses", "조각난 시계", _CARRIES),
                _said("마을 장로", "possesses", "조각난 시계", _HOLDS),
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
        state = _state(tmp_path, out["open_story_id"])
        assert state["written_deaths"] == {}  # he walks again: one finding, not one a scene

    async def test_a_death_that_holds_is_remembered_and_not_flagged(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/continuity.py :: if value == "alive" and dead is not None and dead["scene_id"] != scene_id:
        Becomes: if value == "alive":
        """
        out = await self._clock(tmp_path, [_said("마을 장로", "status", "alive", _HOLDS)])
        village = [f for f in out["continuity"] if f["scene_id"] == "ch01.s02"]
        assert village == []
        [(who, died)] = _state(tmp_path, out["open_story_id"])["written_deaths"].items()
        assert died == {"scene_id": "ch01.s01", "scene_title": "계단", "quote": _FALLS}
        assert who != "마을 장로"

    def test_a_death_later_in_the_same_scene_or_an_earlier_one_is_not_a_return(self) -> None:
        """Killed by: src/uclone_x/story/continuity.py :: dead = ends[subject] if subject in ends else deaths.get(subject)
        Becomes: dead = deaths.get(subject)
        """
        codex = CodexIndex(
            (CodexItem(kind="characters", entry=CharacterEntry(id="jinho", name="이진호")),)
        )
        facts = parse_extraction(
            json.dumps(
                {
                    "facts": [
                        _said("jinho", "status", "dead", _FALLS),
                        _said("jinho", "status", "alive", "깨진 시계를 집어 들었다"),
                    ]
                },
                ensure_ascii=False,
            )
        )
        # Alive then dead in its own scene: a change, and he ends it dead.
        found, ends = came_back(
            codex, scene_id="s1", scene_title="계단", text=_GRAVE, facts=facts, deaths={}
        )
        assert found == [] and ends["jinho"] is not None
        # Dead in an earlier scene and seen twice here: one return, and he ends it alive.
        earlier = {"jinho": {"scene_id": "s0", "scene_title": "앞", "quote": "..."}}
        twice = parse_extraction(
            json.dumps(
                {
                    "facts": [
                        _said("jinho", "status", "alive", _CARRIES),
                        _said("jinho", "status", "alive", _WALKS),
                    ]
                },
                ensure_ascii=False,
            )
        )
        found, ends = came_back(
            codex, scene_id="s1", scene_title="목소리", text=_VILLAGE, facts=twice, deaths=earlier
        )
        assert [f.quote for f in found] == [_WALKS]
        assert ends == {"jinho": None}


async def _to_outline(tmp_path: Path, model: ScriptedModel) -> tuple[str, ToolContext]:
    """A story started, premise 1 chosen and the outline shown: the plot not yet settled."""
    out = await _ok(_tool(model), _ctx(tmp_path), action="start", request=REQUEST)
    story_id = out["open_story_id"]
    ctx = _ctx(tmp_path, story=story_id)
    await _ok(_tool(model), ctx, action="continue", request="1번")
    await _ok(_tool(model), ctx, action="continue", request="네")
    return story_id, ctx


class TestTheFirstChapterIsPlannedBeforeItIsWritten:
    """§1.1: the first chapter's plan stops for the person, as every later chapter's does."""

    async def test_settling_the_plot_shows_the_first_plan_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/start.py :: await self._propose_details(1)
        Becomes: pass
        Killed by: src/uclone_x/story/start.py :: "The first scene opens the story."
        Becomes: ""
        """
        model = ScriptedModel(details=_DETAILS)
        story_id, ctx = await _to_outline(tmp_path, model)
        out = await _ok(_tool(model), ctx, action="continue", request="네")
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
        await _ok(_tool(model), ctx, action="continue", request="네")
        feedback = "방울 소리를 따라가는 한 장면으로 줄여 줘"
        revised = await _ok(_tool(model), ctx, action="continue", request=feedback)
        assert revised["done_now"] == ["details_revised"] and revised["details_chapter"] == 1
        assert [s["title"] for s in revised["chapter_details"]] == ["방울 소리"]
        assert _work(tmp_path, story_id).written_scenes() == []
        written = await _ok(_tool(model), ctx, action="continue", request="네")
        assert written["chapter_written"] == 1
        work = _work(tmp_path, story_id)
        assert work.written_scenes() == ["ch01.s01"]
        outline, _ = work.require_outline()
        assert [s.title for s in outline.chapters[0].scenes] == ["방울 소리"]


class TestTheSceneCountFollowsTheArc:
    """§1.1: how many scenes a chapter has is decided by code from its arc functions."""

    def test_a_turning_chapter_is_longer_and_a_setup_chapter_shorter(self) -> None:
        """Killed by: src/uclone_x/story/start.py :: if functions & _HEAVY:
        Becomes: if False:
        Killed by: src/uclone_x/story/start.py :: if functions and functions <= _LIGHT:
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

        Killed by: src/uclone_x/story/start.py :: "scenes": _items(chapter.get("scenes"))[: scenes_for(arc, number)],
        Becomes: "scenes": _items(chapter.get("scenes")),
        Killed by: src/uclone_x/story/start.py :: count = scenes_for(_arc_of(self.state.arc), number)
        Becomes: count = MAX_SCENES_PER_CHAPTER
        Killed by: src/uclone_x/story/start.py :: for scene in _scene_dicts(found.get("scenes"), "plan", by_name)[:limit]
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
        out = await _ok(_tool(model), ctx, action="continue", request="네")
        assert "It has 2 scenes" in _last(model, DETAILS_MARK)
        assert [s["title"] for s in out["chapter_details"]] == ["장면 1", "장면 2"]


class TestPlotItemsAreThingsNotIdeas:
    """§1.1: the settled plot proposes items a character can hold, not motifs."""

    def test_an_idea_is_told_from_an_object(self) -> None:
        """Killed by: src/uclone_x/story/start.py :: return any(len(w) >= 2 and last.endswith(w) for w in ABSTRACT_ITEM_WORDS if not w.isascii())
        Becomes: return False
        Killed by: src/uclone_x/story/start.py :: if whole in ABSTRACT_ITEM_WORDS or last in ABSTRACT_ITEM_WORDS:
        Becomes: if whole in ABSTRACT_ITEM_WORDS:
        """
        for idea in ("기억", "아버지의 비밀", "잃어버린기억", "a promise", "Hope"):
            assert abstract_item(idea), idea
        for thing in ("등불", "the locket", "물 용기", "promised ring", "열쇠", ""):
            assert not abstract_item(thing), thing

    async def test_an_idea_named_as_an_item_is_dropped_and_counted(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/start.py :: dropped = len(abstract)
        Becomes: dropped = 0
        Killed by: src/uclone_x/story/start.py :: if kind == "items" and abstract_item(name):
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
        out = await _ok(_tool(model), ctx, action="continue", request="네")
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
        out = await _ok(_tool(model), ctx, action="continue", request="네")
        return out, story_id

    async def test_the_plan_names_the_new_character_and_writing_proposes_both_sides(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/start.py :: relations = {f["relative_id"]: INVERSE_ROLE[f["role"]] for f in family}
        Becomes: relations = {f["relative_id"]: f["role"] for f in family}
        Killed by: src/uclone_x/story/start.py :: if self._introduce(number):
        Becomes: if False:
        Killed by: src/uclone_x/story/start.py :: if k.get("change") == "new_entry":
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
        written = await _ok(_tool(model), ctx, action="continue", request="네")
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
        """Killed by: src/uclone_x/story/start.py :: if loaded is None:  # a relative still pending: this side only
        Becomes: if False:  # a relative still pending: this side only
        """
        model = ScriptedModel(details=_FATHER_PLAN)
        _, story_id = await self._plan(tmp_path, model)
        ctx = _ctx(tmp_path, story=story_id)
        written = await _ok(_tool(model), ctx, action="continue", request="네")
        assert [k["change"] for k in written["introduced"]] == ["new_entry"]
        work = _work(tmp_path, story_id)
        [father] = [
            p for p, _ in work.proposals()[0] if p.id == written["introduced"][0]["proposal"]
        ]
        assert father.new_entry().relations == {"harin": "parent"}
        assert all(p.status == "pending" for p, _ in work.proposals()[0])
        assert not work.codex().items
