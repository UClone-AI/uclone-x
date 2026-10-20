"""Unit tests for Novelist (`writer`) scene-by-scene illustration generation.

Verifies:
1. `character_sheet_at_scene`: folds initial `visual.tags` and `visual.progressions`
   (`add_tags` / `remove_tags`) in story-time order up through `scene_id`, strips non-ASCII
   tokens, and computes a deterministic `base_seed` per `(story_id, entry_id)`.
2. `effective_codex`: attaches pending `visual` and `visual_progression` proposals onto
   `CharacterEntry.visual` so scene illustrations reflect unapproved appearance changes.
3. `compose_scene_image_request`: composes Danbooru prompts via `compose_character_prompt`
   (matching the Drawer clone's `ShowSelfTool` architecture) or natural prose prompts for
   `PromptFamily.PROSE`.
4. `StoryStartTool` integration: generates scene illustrations during chapter writing,
   rewrites, and RPG single-scene progression, persisting the URL in `outline.yaml` and
   `start.yaml`, embedding markdown images in `REPLY_NOTE_KEY`, and recording artifact
   provenance on `ToolResult.artifacts`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from tests.unit.test_story_start import REQUEST, ScriptedModel
from uclone_x.core.provenance import Provenance
from uclone_x.tools.builtin.media_registry import PromptFamily
from uclone_x.tools.models import REPLY_NOTE_KEY, NoIsolation, ToolContext, ToolResult
from uclone_x_story.context import CodexIndex, CodexItem
from uclone_x_story.kg_plan import effective_codex
from uclone_x_story.library import StoryLibrary
from uclone_x_story.scene_illustration import (
    SceneVisualPlan,
    can_illustrate,
    character_seed,
    character_sheet_at_scene,
    compose_scene_image_request,
    extract_scene_visual,
    illustrate_scene,
    strip_non_ascii,
    style_for_genre,
)
from uclone_x_story.schemas import (
    Chapter,
    CharacterEntry,
    Outline,
    PlaceEntry,
    Proposal,
    ProposalChange,
    Scene,
    Visual,
    VisualChange,
    VisualProgression,
)
from uclone_x_story.start import START_FILE, StoryStartTool
from uclone_x_story.timeline import place_scenes
from uclone_x_story.tools import NewEntryParams, new_entry_draft
from uclone_x_story.work import StoryWork

ROOM = "room_illus"
_SCENE_TEXT = (
    "하린은 등대 계단을 올랐다. 바람이 창을 두드렸고, 고양이 먹물이 발밑을 따라왔다. "
    "꼭대기에 닿자 렌즈 너머로 안개가 밀려들었다. 그 안에서 누군가 이름을 불렀다. "
    "하린은 등불을 들어 올렸다. 먹물이 낮게 울었다."
)


class FakeImageTool:
    """Test double for `generate_image` that records requests and returns deterministic URLs."""

    def __init__(self, *, succeed: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        self._succeed = succeed

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return "Fake image tool"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(
        self,
        params: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        req = dict(params)
        self.calls.append(req)
        idx = len(self.calls)
        prov = Provenance.primary(provider="local.builtin", model=self.name)
        if not self._succeed:
            return ToolResult(success=False, error="engine busy", provenance=prov)
        rel_path = f"artifacts/sess_room__{ROOM}__writer/images/scene_{idx}.png"
        url = f"/api/artifacts/content?path={rel_path}"
        return ToolResult(
            success=True,
            output={
                "status": "success",
                "relative_url": url,
                "seed": req.get("seed_override", 123),
                "prompt": req.get("prompt", ""),
                "style": req.get("style", "anime"),
                "aspect_ratio": req.get("aspect_ratio", "16:9"),
            },
            artifacts=(rel_path,),
            provenance=prov,
        )


def _ctx(workspace: Path, *, story: str | None = None, turn_id: str | None = None) -> ToolContext:
    return ToolContext(
        agent_id="writer",
        session_id=f"sess_room__{ROOM}__writer",
        workspace_root=workspace,
        room_id=ROOM,
        story_id=story,
        turn_id=turn_id,
        isolation=NoIsolation(),
    )


def _outline() -> Outline:
    return Outline(
        chapters=[
            Chapter(
                id="ch01",
                title="안개",
                scenes=[
                    Scene(
                        id="ch01.s01",
                        title="계단",
                        summary="하린이 등대에 오른다.",
                        beats=["등불을 켠다", "계단을 오른다"],
                        characters=["harin", "meokmul"],
                        places=["angaeseom_deungdae"],
                    ),
                    Scene(
                        id="ch01.s02",
                        title="꼭대기",
                        summary="하린이 안개 속에서 상처를 입는다.",
                        beats=["창문이 깨진다", "등불을 든다"],
                        characters=["harin"],
                        places=["angaeseom_deungdae"],
                    ),
                ],
            )
        ]
    )


def test_strip_non_ascii_and_style_for_genre() -> None:
    """Killed by: examples/story/src/uclone_x_story/scene_illustration.py :: ascii_only = "".join(ch for ch in value if ord(ch) < 128)
    Becomes: ascii_only = value
    Killed by: examples/story/src/uclone_x_story/scene_illustration.py :: if keyword in combined:
    Becomes: if keyword == combined:
    """
    assert strip_non_ascii("short black hair, 검은 머리, raincoat, 노란 우비") == (
        "short black hair, raincoat"
    )
    assert strip_non_ascii("한글만 있음") is None
    assert style_for_genre("미스터리 판타지", "수채화 동화풍") == "artistic"
    assert style_for_genre("사이버펑크 스릴러", "실사 영화") == "photorealistic"
    assert style_for_genre("판타지", "차분함") == "anime"


def test_character_sheet_at_scene_folds_temporal_visual_progressions() -> None:
    """`character_sheet_at_scene` applies `visual.progressions` up through `scene_id` in
    story-time order and derives a deterministic `base_seed`.

    Killed by: examples/story/src/uclone_x_story/scene_illustration.py :: snap = entry_snapshot(entry, placements, scene_id, through_scene=True)
    Becomes: snap = entry_snapshot(entry, placements, scene_id, through_scene=False)
    Killed by: examples/story/src/uclone_x_story/scene_illustration.py :: digest = hashlib.sha256(f"{story_id}:{entry_id}".encode()).digest()[:4]
    Becomes: digest = hashlib.sha256(entry_id.encode()).digest()[:4]
    """
    outline = _outline()
    placements = place_scenes(outline)
    harin = CharacterEntry(
        id="harin",
        name="하린",
        profile="등대지기의 딸.",
        state={"status": "alive", "gender": "female"},
        visual=Visual(
            gender="female",
            tags=["short black hair", "yellow raincoat", "clean coat"],
            progressions=[
                VisualProgression(
                    at="ch01.s02",
                    add_tags=["bandage on cheek", "torn sleeve"],
                    remove_tags=["clean coat"],
                )
            ],
        ),
    )

    sheet_s01 = character_sheet_at_scene(harin, placements, "ch01.s01", story_id="story_a")
    sheet_s02 = character_sheet_at_scene(harin, placements, "ch01.s02", story_id="story_a")
    sheet_other_story = character_sheet_at_scene(harin, placements, "ch01.s01", story_id="story_b")

    assert sheet_s01["core_tags"] == ["short black hair", "yellow raincoat", "clean coat"]
    assert "bandage on cheek" not in sheet_s01["danbooru_tags"]
    assert sheet_s02["core_tags"] == [
        "short black hair",
        "yellow raincoat",
        "bandage on cheek",
        "torn sleeve",
    ]
    assert "clean coat" not in sheet_s02["danbooru_tags"]
    assert sheet_s01["base_seed"] == character_seed("story_a", "harin")
    assert sheet_s01["base_seed"] == sheet_s02["base_seed"]
    assert sheet_s01["base_seed"] != sheet_other_story["base_seed"]


def test_effective_codex_folds_pending_visual_and_visual_progression_proposals() -> None:
    """Pending `visual` and `visual_progression` proposals are folded into `effective_codex`
    so scene illustrations reflect unapproved appearance updates immediately.

    Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if extra_vprogs:
    Becomes: if False:
    """
    outline = _outline()
    placements = place_scenes(outline)
    harin = CharacterEntry(
        id="harin",
        name="하린",
        profile="등대지기의 딸.",
        state={"status": "alive", "gender": "female"},
    )
    base_codex = CodexIndex(items=(CodexItem(kind="characters", entry=harin),))
    p_vis = Proposal(
        id="p001",
        kind="characters",
        entry_id="harin",
        entry_digest="sha256:abc",
        change=ProposalChange(
            visual=VisualChange(
                gender="female",
                tags=["short black hair", "yellow raincoat"],
            )
        ),
        proposed_at="2026-10-19T00:00:00Z",
        room_id=ROOM,
    )
    p_vprog = Proposal(
        id="p002",
        kind="characters",
        entry_id="harin",
        entry_digest="sha256:abc",
        change=ProposalChange(
            visual_progression=VisualProgression(
                at="ch01.s02",
                add_tags=["wet hair", "muddy boots"],
            )
        ),
        proposed_at="2026-10-19T00:01:00Z",
        room_id=ROOM,
    )

    merged = effective_codex(base_codex, [p_vis, p_vprog], ())
    found = merged.find("harin", "characters")
    assert found is not None
    char_entry = found[0].entry
    assert isinstance(char_entry, CharacterEntry)
    sheet = character_sheet_at_scene(char_entry, placements, "ch01.s02", story_id="story_a")
    assert sheet["core_tags"] == [
        "short black hair",
        "yellow raincoat",
        "wet hair",
        "muddy boots",
    ]


def test_new_entry_draft_populates_visual_tags_when_given() -> None:
    """`NewEntryParams.visual_tags` populates `looks.tags` on character proposals.

    Killed by: examples/story/src/uclone_x_story/tools.py :: clean_tags = [t.strip() for t in asked.visual_tags if t.strip()]
    Becomes: clean_tags = []
    """
    ctx = ToolContext(agent_id="writer", session_id="sess_1", room_id=ROOM)
    params = NewEntryParams(
        kind="characters",
        name="하린",
        profile="등대지기의 딸.",
        gender="female",
        visual_tags=["short black hair", "yellow raincoat"],
        base_seed=4242,
    )
    draft = new_entry_draft(params, room_id=ROOM, context=ctx)
    entry = draft.new_entry()
    assert isinstance(entry, CharacterEntry)
    assert entry.visual is not None
    assert entry.visual.tags == ["short black hair", "yellow raincoat"]
    assert entry.visual.base_seed == 4242


def test_compose_scene_image_request_danbooru_and_prose() -> None:
    """`compose_scene_image_request` builds a Danbooru prompt via `compose_character_prompt`
    or a natural prose prompt when `PromptFamily.NATURAL_PROSE` is selected.

    Killed by: examples/story/src/uclone_x_story/scene_illustration.py :: if prompt_family == PromptFamily.NATURAL_PROSE:
    Becomes: if False:
    Killed by: examples/story/src/uclone_x_story/scene_illustration.py :: request["seed_override"] = seed
    Becomes: request["seed_override"] = 0
    """
    outline = _outline()
    placements = place_scenes(outline)
    scene = outline.chapters[0].scenes[0]
    harin = CharacterEntry(
        id="harin",
        name="하린",
        profile="등대지기의 딸.",
        state={"status": "alive", "gender": "female"},
        visual=Visual(
            gender="female",
            tags=["short black hair", "yellow raincoat"],
            base_seed=777,
        ),
    )
    sheet = character_sheet_at_scene(harin, placements, scene.id, story_id="story_a")
    plan = SceneVisualPlan(
        characters=("harin",),
        action="holding brass lantern, climbing spiral stairs",
        expression="focused expression",
        location="dark lighthouse interior, stone wall",
        lighting="warm lantern glow, deep shadows",
        camera="medium shot, low angle",
    )

    danbooru_req = compose_scene_image_request(
        plan,
        [sheet],
        scene=scene,
        genre="판타지",
        prompt_family=PromptFamily.DANBOORU,
    )
    assert "1girl, solo" in danbooru_req["prompt"]
    assert "short black hair, yellow raincoat" in danbooru_req["prompt"]
    assert "holding brass lantern" in danbooru_req["prompt"]
    assert danbooru_req["aspect_ratio"] == "16:9"
    assert danbooru_req["seed_override"] == 777
    assert "worst quality" in danbooru_req["negative_prompt"]

    prose_req = compose_scene_image_request(
        plan,
        [sheet],
        scene=scene,
        genre="판타지",
        prompt_family=PromptFamily.NATURAL_PROSE,
    )
    assert "Scene illustration for '계단'." in prose_req["prompt"]
    assert "하린 (short black hair, yellow raincoat)" in prose_req["prompt"]
    assert prose_req["seed_override"] == 777


async def test_illustrate_scene_executes_image_tool_and_handles_missing_engine(
    tmp_path: Path,
) -> None:
    """`illustrate_scene` returns `None` when no image tool is available and returns the
    generated `relative_url` when `image_tool` succeeds.

    Killed by: examples/story/src/uclone_x_story/scene_illustration.py :: if not await can_illustrate(resolved_tool, context) or resolved_tool is None:
    Becomes: if True:
    """
    outline = _outline()
    harin = CharacterEntry(
        id="harin",
        name="하린",
        profile="등대지기의 딸.",
        state={"status": "alive", "gender": "female"},
        visual=Visual(gender="female", tags=["short black hair", "yellow raincoat"]),
    )
    place = PlaceEntry(id="angaeseom_deungdae", name="안개섬 등대", profile="바위 위 등대.")
    codex = CodexIndex(
        items=(
            CodexItem(kind="characters", entry=harin),
            CodexItem(kind="places", entry=place),
        )
    )
    visual_entry = {
        "characters": ["하린"],
        "action": "holding lantern on spiral stairs",
        "expression": "determined expression",
        "location": "lighthouse interior, fog outside window",
        "lighting": "warm lantern light",
        "camera": "medium shot",
    }
    model = ScriptedModel(scene_visuals=[visual_entry, visual_entry])
    ctx = _ctx(tmp_path)

    # No image tool -> returns None without calling model
    assert await can_illustrate(None, ctx) is False
    url_none = await illustrate_scene(
        model=model,
        image_tool=None,
        context=ctx,
        story_id="story_1",
        outline=outline,
        codex=codex,
        cast_lines=[],
        scene_id="ch01.s01",
        text=_SCENE_TEXT,
        genre="판타지",
    )
    assert url_none is None
    assert model.prompts == []

    # With FakeImageTool -> extracts SceneVisualPlan and generates illustration
    fake_img = FakeImageTool()
    plan = await extract_scene_visual(
        model,
        scene=outline.chapters[0].scenes[0],
        text=_SCENE_TEXT,
        characters=[harin],
        places=["안개섬 등대: 바위 위 등대."],
        genre="판타지",
    )
    assert plan.action == "holding lantern on spiral stairs"

    url = await illustrate_scene(
        model=model,
        image_tool=fake_img,
        context=ctx,
        story_id="story_1",
        outline=outline,
        codex=codex,
        cast_lines=[],
        scene_id="ch01.s01",
        text=_SCENE_TEXT,
        genre="판타지",
    )
    assert (
        url == f"/api/artifacts/content?path=artifacts/sess_room__{ROOM}__writer/images/scene_1.png"
    )
    assert len(fake_img.calls) == 1
    assert "short black hair, yellow raincoat" in fake_img.calls[0]["prompt"]
    assert "holding lantern on spiral stairs" in fake_img.calls[0]["prompt"]


async def test_story_start_persists_scene_illustrations_and_embeds_in_reply_note(
    tmp_path: Path,
) -> None:
    """When `StoryStartTool` writes scenes with an active `image_tool`, each written scene's
    illustration URL is saved in `outline.yaml` and `start.yaml`, embedded as markdown in
    `REPLY_NOTE_KEY`, and recorded in `ToolResult.artifacts`.

    Killed by: examples/story/src/uclone_x_story/start_scene.py :: outline = await self._illustrate_written_scene(scene.id, text)
    Becomes: pass
    Killed by: examples/story/src/uclone_x_story/start_notes.py :: url = state.scene_illustrations.get(sid)
    Becomes: url = None
    Killed by: examples/story/src/uclone_x_story/start.py :: result["scene_illustrations"] = dict(state.scene_illustrations)
    Becomes: pass
    """
    fake_img = FakeImageTool()
    model = ScriptedModel(
        scene_visuals=[
            {
                "characters": ["하린"],
                "action": "holding brass lantern, climbing stone stairs",
                "expression": "focused expression",
                "location": "lighthouse staircase, fog outside",
                "lighting": "warm lantern glow, night",
                "camera": "medium shot",
            }
        ]
    )
    tool = StoryStartTool(model_factory=lambda _: model, image_tool=fake_img)

    # Quick mode ("바로 써줘") writes chapter 1 in a single call
    quick_req = f"{REQUEST} 바로 써줘"
    exec_res = await tool.execute({"action": "start", "request": quick_req}, _ctx(tmp_path))
    assert exec_res.success
    out = cast("dict[str, Any]", exec_res.output)
    story_id = str(out["story_id"])

    # Every written scene in chapter 1 gets an illustration
    written = cast("list[str]", out["chapter_scenes"])
    assert len(written) >= 2
    illus_map = cast("dict[str, str]", out["scene_illustrations"])
    assert set(illus_map.keys()) == set(written)
    first_url = illus_map["ch01.s01"]
    assert first_url.startswith("/api/artifacts/content?path=artifacts/")

    # Reply note embeds markdown image links
    reply_note = cast("dict[str, str]", out[REPLY_NOTE_KEY])
    assert f"![1장 1번째 장면]({first_url})" in reply_note["ko"]
    assert f"![chapter 1, scene 1]({first_url})" in reply_note["en"]

    # ToolResult.artifacts records the workspace paths
    assert f"artifacts/sess_room__{ROOM}__writer/images/scene_1.png" in exec_res.artifacts

    # outline.yaml and start.yaml both persist the illustration URL
    work = StoryWork(StoryLibrary(tmp_path), story_id)
    saved_outline, _ = work.require_outline()
    s01 = saved_outline.find("ch01.s01")
    assert s01 is not None
    assert s01[1].illustration == first_url

    start_file = work.read(START_FILE)
    assert start_file is not None
    raw_start = cast("dict[str, Any]", yaml.safe_load(start_file.text))
    assert raw_start["scene_illustrations"]["ch01.s01"] == first_url


async def test_story_start_rpg_mode_illustrates_each_scene_on_choice(
    tmp_path: Path,
) -> None:
    """In RPG (`interactive=True`) mode, `_write_single_scene` generates an illustration for
    the enacted scene and includes it in the reply note.

    Killed by: examples/story/src/uclone_x_story/start_rpg.py :: await self._illustrate_written_scene(scene.id, text)
    Becomes: pass
    """
    fake_img = FakeImageTool()
    model = ScriptedModel(
        scene_visuals=[
            {
                "characters": ["하린"],
                "action": "opening lighthouse door, holding lantern",
                "expression": "surprised expression",
                "location": "misty lighthouse entrance",
                "lighting": "moonlight and lantern glow",
                "camera": "wide shot",
            }
        ]
    )
    tool = StoryStartTool(model_factory=lambda _: model, image_tool=fake_img)

    res0 = await tool.execute(
        {"action": "start", "request": REQUEST, "interactive": True},
        _ctx(tmp_path, turn_id="t1"),
    )
    out0 = cast("dict[str, Any]", res0.output)
    story_id = str(out0["story_id"])
    await tool.execute(
        {"action": "continue", "choice": 1},
        _ctx(tmp_path, story=story_id, turn_id="t2"),
    )
    await tool.execute(
        {"action": "continue", "go_on": True},
        _ctx(tmp_path, story=story_id, turn_id="t3"),
    )
    res_rpg = await tool.execute(
        {"action": "continue", "go_on": True},
        _ctx(tmp_path, story=story_id, turn_id="t4"),
    )
    out_rpg = cast("dict[str, Any]", res_rpg.output)
    assert out_rpg["stage"] == "review_scene"
    assert out_rpg["scene_id"] == "ch01.s01"

    # Choose option 1 for ch01.s01 -> writes ch01.s01 and generates 1 illustration
    res_scene1 = await tool.execute(
        {"action": "continue", "choice": 1},
        _ctx(tmp_path, story=story_id, turn_id="t5"),
    )
    out1 = cast("dict[str, Any]", res_scene1.output)
    assert out1["scene_written_last"] == "ch01.s01"
    illus1 = cast("dict[str, str]", out1["scene_illustrations"])
    assert "ch01.s01" in illus1
    url1 = illus1["ch01.s01"]
    note1 = cast("dict[str, str]", out1[REPLY_NOTE_KEY])
    assert f"![1장 1번째 장면]({url1})" in note1["ko"]
    assert len(fake_img.calls) == 1
