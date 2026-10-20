"""Unit tests for Temporal Knowledge Graph (TKG) state-transition story planning (#2358).

Verifies:
1. `effective_codex`: combines approved entries, pending `new_entry`/`aliases`/`progression`
   proposals, and `planned_transitions` so `scene_context` populates both `progressions_applied`
   ($G_{t-1}$) and `changes_in_this_scene` ($\\Delta G_t$) even when proposals are still pending.
2. `validate_kg_trajectory`: deterministically simulates $G_0 \\to G_1 \\to \\dots \\to G_n$,
   filtering `dead_actor`, `premature_twist`, `premature_resolution`, `immutable_identity`,
   and `redundant_transition`.
3. `story_start` integration: `_lock_plot` and `_apply_details` compute and validate the
   temporal KG trajectory, and `_scene_prompt` injects `kg_scene_guidance` with human-readable
   entity names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import yaml

from tests.unit.test_story_start import REQUEST, ScriptedModel
from uclone_x.tools.models import NoIsolation, ToolContext
from uclone_x_story.arc import Arc
from uclone_x_story.context import CodexIndex, CodexItem, scene_context
from uclone_x_story.kg_plan import (
    SceneTransition,
    chronological_kg_log,
    effective_codex,
    format_chronological_kg_log,
    kg_scene_guidance,
    merge_transitions,
    parse_detail_transitions,
    parse_plot_transitions,
    parse_scene_transitions,
    thread_transitions,
    unplaced_transitions,
    validate_kg_trajectory,
)
from uclone_x_story.plan_check import PLAN_MARKER, plan_prompt, scene_threads
from uclone_x_story.schemas import (
    Chapter,
    CharacterEntry,
    ItemEntry,
    Outline,
    PlaceEntry,
    Progression,
    Proposal,
    Scene,
    ThreadEntry,
)
from uclone_x_story.start import KG_MARK, START_FILE, StoryStartTool, wants_interactive
from uclone_x_story.tools import NewEntryParams, new_entry_draft

ROOM = "room_kg"
#: A scene the prose check passes, in Korean.
_SCENE = (
    "하린은 등대 계단을 올랐다. 바람이 창을 두드렸고, 고양이 먹물이 발밑을 따라왔다. "
    "꼭대기에 닿자 렌즈 너머로 안개가 밀려들었다. 그 안에서 누군가 이름을 불렀다. "
    "하린은 등불을 들어 올렸다. 먹물이 낮게 울었다."
)
_REASON_CODES = (
    "unknown_scene",
    "unknown_entity",
    "dead_actor",
    "immutable_identity",
    "premature_twist",
    "premature_resolution",
    "redundant_transition",
)


def _ctx(workspace: Path, *, story: str | None = None) -> ToolContext:
    return ToolContext(
        agent_id="writer",
        session_id=f"sess_room__{ROOM}__writer",
        workspace_root=workspace,
        room_id=ROOM,
        story_id=story,
        isolation=NoIsolation(),
    )


def _sample_outline() -> Outline:
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
                        beats=["등불을 켠다"],
                        characters=["harin", "meokmul"],
                        places=["angaeseom_deungdae"],
                    ),
                    Scene(
                        id="ch01.s02",
                        title="목소리",
                        summary="하린이 목소리를 쫓는다.",
                        beats=["문이 열린다"],
                        characters=["harin"],
                        places=["angaeseom_deungdae"],
                    ),
                ],
            ),
            Chapter(
                id="ch02",
                title="새벽",
                scenes=[
                    Scene(
                        id="ch02.s01",
                        title="진실",
                        summary="반전이 드러난다.",
                        beats=["발자국을 본다"],
                        characters=["harin", "meokmul"],
                    ),
                ],
            ),
            Chapter(
                id="ch03",
                title="아침",
                scenes=[
                    Scene(
                        id="ch03.s01",
                        title="결말",
                        summary="갈등이 해결된다.",
                        beats=["고양이를 되찾는다"],
                        characters=["harin", "meokmul"],
                    ),
                ],
            ),
        ]
    )


def _sample_codex() -> CodexIndex:
    harin = CharacterEntry(
        id="harin",
        name="하린",
        aliases=["소녀"],
        profile="등대지기의 딸.",
        state={"status": "alive", "gender": "female"},
        relations={"meokmul": "owner"},
    )
    meokmul = CharacterEntry(
        id="meokmul",
        name="먹물",
        profile="검은 고양이.",
        state={"status": "missing"},
    )
    lantern = ItemEntry(
        id="deungbul",
        name="등불",
        profile="오래된 등불.",
    )
    conflict = ThreadEntry(
        id="central_conflict",
        name="중심 갈등",
        profile="안개 속 목소리의 정체를 밝히려 한다.",
        planted_in="ch01.s01",
        pay_off_by="ch03.s01",
        always_include=True,
    )
    twist = ThreadEntry(
        id="twist",
        name="반전",
        profile="목소리는 하린 자신의 것이었다.",
        pay_off_by="ch02.s01",
        always_include=True,
    )
    return CodexIndex(
        items=(
            CodexItem(kind="characters", entry=harin),
            CodexItem(kind="characters", entry=meokmul),
            CodexItem(kind="items", entry=lantern),
            CodexItem(kind="threads", entry=conflict),
            CodexItem(kind="threads", entry=twist),
        )
    )


class TestEffectiveCodexAndGuidance:
    """`effective_codex` attaches pending proposals and planned transitions onto `CodexEntry`."""

    def test_pending_proposals_and_planned_transitions_populate_scene_context(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if isinstance(entry, ThreadEntry) and entry.id in ("central_conflict", "twist"):
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if not extra_progs:
        Becomes: if True:
        """
        ctx = _ctx(tmp_path)
        harin_draft = new_entry_draft(
            NewEntryParams(
                kind="characters",
                id="harin",
                name="하린",
                profile="등대지기의 딸.",
                state={"status": "alive", "gender": "female"},
            ),
            room_id=ROOM,
            context=ctx,
        )
        conflict_draft = new_entry_draft(
            NewEntryParams(
                kind="threads",
                id="central_conflict",
                name="중심 갈등",
                profile="목소리의 정체를 찾는다.",
            ),
            room_id=ROOM,
            context=ctx,
        )
        prog_proposal = Proposal.model_validate(
            {
                "id": "p010",
                "kind": "characters",
                "entry_id": "harin",
                "entry_digest": "sha256:0000",
                "change": {
                    "progression": {
                        "at": "ch01.s01",
                        "set": {"possesses": "deungbul"},
                        "note": "등불을 집어 든다",
                    }
                },
                "proposed_at": "2026-10-15T00:00:00+00:00",
                "room_id": ROOM,
            }
        )
        planned = [
            SceneTransition(
                scene_id="ch01.s02",
                entity_id="harin",
                kind="characters",
                set={"stance": "determined"},
                note="안개 속으로 나아갈 결심을 굳힌다",
            ),
            SceneTransition(
                scene_id="ch01.s02",
                entity_id="central_conflict",
                kind="threads",
                set={"thread_stage": "escalated"},
                note="목소리가 등대 문을 두드린다",
            ),
        ]
        outline = _sample_outline()
        codex = effective_codex(
            CodexIndex(),
            [harin_draft, conflict_draft, prog_proposal],
            planned,
        )
        bundle = scene_context(
            outline,
            "ch01.s02",
            codex,
            previous_text=None,
            story={},
        )
        manifest = bundle["manifest"]
        applied_ids = {item["id"] for item in manifest["progressions_applied"]}
        in_scene_ids = {item["id"] for item in manifest["changes_in_this_scene"]}
        assert "harin" in applied_ids
        assert {"harin", "central_conflict"} <= in_scene_ids

        guidance = kg_scene_guidance(bundle, codex)
        assert "World state entering this scene" in guidance
        assert "하린: possesses=deungbul" in guidance
        assert "Target state/relationship transitions in THIS scene" in guidance
        assert "하린: stance=determined" in guidance
        assert "중심 갈등: thread_stage=escalated" in guidance


class TestValidateKgTrajectory:
    """`validate_kg_trajectory` enforces causal invariants across $G_0 \\to G_1 \\to \\dots \\to G_n$."""

    def test_dead_actor_premature_twist_immutable_and_redundant_transitions_are_filtered(
        self,
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if status_of(entering.get("status")) == "dead":
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if eid == "twist" and scene_chapter < arc.twist_chapter:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if eid == "central_conflict" and scene_chapter < arc.climax_chapter:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if k in _IMMUTABLE_KEYS and k in state_now:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if state_now.get(k) == v:
        Becomes: if False:
        """
        outline = _sample_outline()
        codex = _sample_codex()
        arc = Arc(
            functions=(("setup",), ("reversal",), ("climax", "resolution")),
            twist_chapter=2,
            climax_chapter=3,
        )
        raw = [
            # Invalid (`immutable_identity`): trying to overwrite harin's gender before she dies
            SceneTransition(
                scene_id="ch01.s01",
                entity_id="harin",
                kind="characters",
                set={"gender": "male"},
            ),
            # Valid: Harin dies at the end of ch01.s01
            SceneTransition(
                scene_id="ch01.s01",
                entity_id="harin",
                kind="characters",
                set={"status": "dead"},
                note="추락한다",
            ),
            # Invalid (`dead_actor`): Harin tries to act in ch01.s02 after dying in ch01.s01
            SceneTransition(
                scene_id="ch01.s02",
                entity_id="harin",
                kind="characters",
                set={"location": "roof"},
            ),
            # Invalid (`premature_twist`): twist paid off in chapter 1 (< twist_chapter=2)
            SceneTransition(
                scene_id="ch01.s01",
                entity_id="twist",
                kind="threads",
                set={"thread_stage": "paid_off"},
            ),
            # Invalid (`premature_resolution`): conflict paid off in chapter 2 (< climax_chapter=3)
            SceneTransition(
                scene_id="ch02.s01",
                entity_id="central_conflict",
                kind="threads",
                set={"thread_stage": "paid_off"},
            ),
            # Invalid (`redundant_transition`): status: missing already holds in G_0
            SceneTransition(
                scene_id="ch01.s01",
                entity_id="meokmul",
                kind="characters",
                set={"status": "missing"},
            ),
            # Valid: meokmul becomes found in ch02.s01
            SceneTransition(
                scene_id="ch02.s01",
                entity_id="meokmul",
                kind="characters",
                set={"status": "alive"},
                relations={"harin": "companion"},
            ),
        ]
        kept, report = validate_kg_trajectory(outline, codex, raw, arc=arc)
        reasons = {d["reason"] for d in report.dropped}
        assert {
            "dead_actor",
            "premature_twist",
            "premature_resolution",
            "immutable_identity",
            "redundant_transition",
        } <= reasons
        assert [(t.scene_id, t.entity_id, t.set) for t in kept] == [
            ("ch01.s01", "harin", {"status": "dead"}),
            ("ch02.s01", "meokmul", {"status": "alive"}),
        ]
        assert report.active_scenes == ("ch01.s01", "ch02.s01")
        assert "ch01.s02" in report.stagnant_scenes
        assert report.edge_deltas == 3

    def test_parsers_and_merge_coalesce_and_replace_scene_transitions(self) -> None:
        outline = _sample_outline()
        codex = _sample_codex()
        cast_lines = [
            {"id": "harin", "name": "하린", "kind": "characters", "aliases": ["소녀"]},
            {"id": "meokmul", "name": "먹물", "kind": "characters"},
        ]

        plot_found = {
            "transitions": [
                {
                    "scene": "[ch01.s01]",
                    "subject": "소녀",
                    "set": {"possesses": "등불"},
                    "relations": {"먹물": "searching_for"},
                    "note": "등불을 들고 계단을 오른다",
                },
                {
                    "scene": "ch01.s02",
                    "subject": "먹물",
                    "set": {"location": "deungdae"},
                    "note": "목소리가 들리는 곳에서 기다린다",
                },
            ]
        }
        parsed_plot = parse_plot_transitions(plot_found, outline, cast_lines, codex)
        assert len(parsed_plot) == 2
        assert parsed_plot[0].scene_id == "ch01.s01"
        assert parsed_plot[0].entity_id == "harin"
        assert parsed_plot[0].relations == {"meokmul": "searching_for"}
        assert parsed_plot[1].scene_id == "ch01.s02"
        assert parsed_plot[1].entity_id == "meokmul"

        prompt_text = plan_prompt(
            outline.chapters[0].scenes[0],
            "하린은 등불을 들었다.",
            chapter=1,
            twist=None,
            twist_chapter=None,
            transitions=["하린: possesses=등불"],
        )
        assert "Planned state/relationship transitions for this scene:" in prompt_text
        assert "- 하린: possesses=등불" in prompt_text

        details_found = {
            "scenes": [
                {
                    "title": "계단",
                    "transitions": [{"subject": "하린", "set": {"stance": "alert"}}],
                },
                {
                    "title": "목소리",
                    "transitions": [{"subject": "먹물", "note": "울음소리가 들린다"}],
                },
            ]
        }
        parsed_details = parse_detail_transitions(details_found, outline, 1, cast_lines, codex)
        assert [(t.scene_id, t.entity_id) for t in parsed_details] == [
            ("ch01.s01", "harin"),
            ("ch01.s02", "meokmul"),
        ]
        merged = merge_transitions(
            parsed_plot,
            parsed_details,
            replace_scenes={"ch01.s01", "ch01.s02"},
        )
        assert [(t.scene_id, t.entity_id, t.set) for t in merged] == [
            ("ch01.s01", "harin", {"stance": "alert"}),
            ("ch01.s02", "meokmul", {"milestone": "울음소리가 들린다"}),
        ]


class TestStoryStartKgTrajectoryIntegration:
    """`story_start` plans, validates, and injects the temporal KG trajectory into scene prompts."""

    async def test_lock_plot_and_scene_prompt_carry_kg_trajectory_into_writing(
        self, tmp_path: Path
    ) -> None:
        r"""Killed by: examples/story/src/uclone_x_story/start_scene.py :: kg_block = f"{guidance}\n\n" if guidance else ""
        Becomes: kg_block = ""
        Killed by: examples/story/src/uclone_x_story/start_details.py :: planned_transitions=[t.record() for t in validated],
        Becomes: planned_transitions=[],
        """
        kg_reply: dict[str, Any] = {
            "threads": [
                {
                    "name": "등대의 약속",
                    "profile": "하린이 먹물을 찾겠다고 한 약속.",
                    "characters": ["하린", "먹물"],
                    "planted_chapter": 1,
                    "pay_off_chapter": 2,
                }
            ],
            "items": [{"name": "등불", "profile": "오래된 등불."}],
            "transitions": [
                {
                    "scene": "[ch01.s01]",
                    "subject": "하린",
                    "set": {"possesses": "등불"},
                    "note": "하린이 등불을 켜고 계단을 오른다",
                }
            ],
        }
        details_reply: dict[str, Any] = {
            "scenes": [
                {
                    "title": "젖은 발자국",
                    "summary": "하린이 계단의 발자국을 따라간다.",
                    "beats": ["발자국이 이어진다"],
                    "characters": ["하린", "먹물"],
                    "places": ["안개섬 등대"],
                    "transitions": [
                        {
                            "subject": "하린",
                            "set": {"stance": "resolute"},
                            "note": "발자국 끝까지 가기로 결심한다",
                        }
                    ],
                },
                {
                    "title": "걷히는 안개",
                    "summary": "안개가 걷힌다.",
                    "beats": ["목소리가 들린다"],
                    "characters": ["하린"],
                    "places": ["안개섬 등대"],
                    "transitions": [
                        {
                            "subject": "먹물",
                            "set": {"status": "alive"},
                            "note": "안개 너머에서 먹물의 방울 소리가 울린다",
                        }
                    ],
                },
            ]
        }
        model = ScriptedModel(kg=kg_reply, details=details_reply)
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success and isinstance(started.output, dict)
        out = cast("dict[str, Any]", started.output)
        # The trajectory is the flow's own record: no check code or internal id of it
        # reaches the Writer (#2374).
        assert "kg_trajectory" not in out
        shown = json.dumps(out, ensure_ascii=False)
        for internal in (*_REASON_CODES, '"scene_id"', '"entity_id"', "edge_deltas"):
            assert internal not in shown, internal
        kg_prompt = next(p for p in model.prompts if KG_MARK in p)
        assert "[ch01.s01]" in kg_prompt

        scene_prompts = [p for p in model.prompts if "Write scene" in p]
        assert len(scene_prompts) == 2
        # Scene 1 prompt has target transitions in THIS scene
        assert "Target state/relationship transitions in THIS scene" in scene_prompts[0]
        assert "중심 갈등: thread_stage=planted" in scene_prompts[0]
        # Scene 2 prompt has entering world state from Scene 1 AND its own target transitions
        assert "World state entering this scene" in scene_prompts[1]
        assert "Target state/relationship transitions in THIS scene" in scene_prompts[1]
        assert "먹물: status=alive" in scene_prompts[1]

        start_file = tmp_path / "stories" / str(out["open_story_id"]) / START_FILE
        saved = cast("dict[str, Any]", yaml.safe_load(start_file.read_text(encoding="utf-8")))
        assert saved["planned_transitions"]
        traj = cast("dict[str, Any]", saved["kg_trajectory"])
        assert int(traj["total_scenes"]) >= 2
        assert int(traj["edge_deltas"]) >= 3


class TestTheReviewFindingsOf2373:
    """#2374: a transition's scene is the label the code wrote, never a guess from words; a
    death already on record stops a planned act; the flow's notes follow the story's
    language; a planned status is one of the statuses."""

    def test_a_transition_is_placed_by_its_label_and_never_by_its_words(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: text = _text(label).strip("[] ")
        Becomes: text = _text(label)
        """
        outline = _sample_outline()
        cast_lines = [{"id": "harin", "name": "하린", "kind": "characters"}]
        found = {
            "transitions": [
                # Its note shares words with ch01.s01 ("등불을 켠다"); its label says s02.
                {"scene": "[ch01.s02]", "subject": "하린", "note": "등불을 켠다"},
                # No label: a chapter number and words the first scene shares are not a place.
                {"chapter": 1, "subject": "하린", "note": "등불을 켠다"},
                # A label the outline does not have is not a place either.
                {"scene": "[ch09.s01]", "subject": "하린", "note": "등불을 켠다"},
            ]
        }
        parsed = parse_plot_transitions(found, outline, cast_lines)
        assert [t.scene_id for t in parsed] == ["ch01.s02"]

    def test_a_death_already_on_record_stops_a_planned_act(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: entry = item.entry.model_copy(update={"progressions": [*item.entry.progressions, *added]})
        Becomes: entry = item.entry.model_copy(update={"progressions": [*added]})
        """
        outline = _sample_outline()
        harin = CharacterEntry(
            id="harin",
            name="하린",
            profile="등대지기의 딸.",
            state={"status": "alive"},
            progressions=[Progression(at="ch01.s01", set={"status": "dead"})],
        )
        codex = CodexIndex(items=(CodexItem(kind="characters", entry=harin),))
        planned = [
            SceneTransition(scene_id="ch01.s02", entity_id="harin", set={"location": "roof"})
        ]
        kept, report = validate_kg_trajectory(outline, codex, planned)
        assert kept == []
        assert [d["reason"] for d in report.dropped] == ["dead_actor"]

    def test_a_planned_death_in_the_storys_words_is_a_death(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: cleaned = status_of(cleaned) if key == "status" else cleaned
        Becomes: cleaned = cleaned
        """
        outline = _sample_outline()
        codex = _sample_codex()
        cast_lines = [{"id": "harin", "name": "하린", "kind": "characters"}]
        found = {
            "transitions": [
                {"scene": "[ch01.s01]", "subject": "하린", "set": {"status": "사망"}},
                {"scene": "[ch01.s02]", "subject": "하린", "set": {"location": "옥상"}},
            ]
        }
        parsed = parse_plot_transitions(found, outline, cast_lines, codex)
        assert parsed[0].set == {"status": "dead"}
        kept, report = validate_kg_trajectory(outline, codex, parsed)
        assert [(t.scene_id, t.set) for t in kept] == [("ch01.s01", {"status": "dead"})]
        assert [d["reason"] for d in report.dropped] == ["dead_actor"]

    def test_a_threads_notes_are_in_the_storys_language(self) -> None:
        outline = _sample_outline()

        class _Thread:
            kind = "threads"
            name = "The promise"
            id = "the_promise"

        threads = [(_Thread(), {"planted_in": "ch01.s01", "pay_off_by": "ch03.s01"})]
        english = thread_transitions(
            threads, outline, id_for_name_fn=lambda _n: None, language="en"
        )
        korean = thread_transitions(threads, outline, id_for_name_fn=lambda _n: None, language="ko")
        assert [t.note for t in english] == [
            "The promise: set up as a clue or a conflict",
            "The promise: paid off",
        ]
        assert [t.note for t in korean] == ["The promise: 단서/갈등 제시", "The promise: 회수/결말"]
        assert not any(c >= "\uac00" for t in english for c in t.note or "")

    def test_bracketed_scene_header_meta_keys_and_non_status_condition_are_normalized(
        self,
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: text = text.partition("]")[0].strip()
        Becomes: text = text
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: state_items = {state_items["key"]: state_items["value"]}
        Becomes: state_items = state_items
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: key = "condition"
        Becomes: key = "status"
        Killed by: examples/story/src/uclone_x_story/start_outline.py :: if body_key in seen_body:
        Becomes: if False:
        """
        from uclone_x_story.start_outline import outline_from

        outline = _sample_outline()
        cast_lines = [{"id": "harin", "name": "하린", "kind": "characters"}]
        found = {
            "transitions": [
                {
                    "scene": "[ch01.s01] 계단",
                    "subject": "하린",
                    "set": {"key": "goal", "value": "고양이를 찾기"},
                },
                {
                    "scene": "[ch01.s02] 목소리",
                    "subject": "하린",
                    "set": {"status": "기억을 잃고 흩어진 마을 사람들과 함께 이동"},
                },
            ]
        }
        parsed = parse_plot_transitions(found, outline, cast_lines)
        assert [(t.scene_id, t.set) for t in parsed] == [
            ("ch01.s01", {"goal": "고양이를 찾기"}),
            ("ch01.s02", {"condition": "기억을 잃고 흩어진 마을 사람들과 함께 이동"}),
        ]

        deduped = outline_from(
            {
                "chapters": [
                    {
                        "title": "반복 없는 장",
                        "scenes": [
                            {
                                "title": "고양이의 선택",
                                "summary": "소녀가 숲 깊은 곳으로 들어간다.",
                                "beats": ["흔적을 발견한다", "울음소리를 듣는다"],
                            },
                            {
                                "title": "마지막 선택",
                                "summary": "소녀가 숲 깊은 곳으로 들어간다.",
                                "beats": ["흔적을 발견한다", "울음소리를 듣는다"],
                            },
                        ],
                    }
                ]
            },
            cast_lines,
        )
        assert [s.id for s in deduped.chapters[0].scenes] == ["ch01.s01"]

    def test_subgenre_muse_mapping_and_dict_unwrapping(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: for words in (_GENRE_WORDS, _MUSE_GENRE_WORDS):
        Becomes: for words in (_GENRE_WORDS,):
        Killed by: examples/story/src/uclone_x_story/start_state.py :: if len(mapping) == 1:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/start_premise.py :: if (key := _title_key(premises[i].title)) and key == _title_key(premises[j].title)
        Becomes: if False
        """
        from uclone_x_story.start_premise import same_title_count
        from uclone_x_story.start_state import Premise, brief_from, muse_genre

        tables = ("fantasy", "horror", "mystery", "romance", "science-fiction")
        assert muse_genre("따뜻한 드라마", tables) == "romance"
        assert muse_genre("사이버펑크 느와르", tables) == "science-fiction"
        assert muse_genre("느와르 스릴러", tables) == "mystery"
        assert muse_genre("비극", tables) is None

        req = "안개 낀 항구 마을의 낡은 시계탑에서 일하는 소녀가 잃어버린 검은 고양이 먹물을 찾는 이야기"
        brief = brief_from(
            {
                "must_have": [
                    {"배경": "안개 낀 항구 마을"},
                    "검은 고양이 '먹물'",
                ]
            },
            req,
        )
        assert brief.must_have == ["안개 낀 항구 마을", "검은 고양이 '먹물'"]
        p_all_same = [Premise(number=i, title="같은 제목", premise=f"내용 {i}") for i in (1, 2, 3)]
        p_one_fixed = [
            Premise(number=1, title="같은 제목", premise="내용 1"),
            Premise(number=2, title="다른 제목", premise="내용 2"),
            Premise(number=3, title="같은 제목", premise="내용 3"),
        ]
        assert same_title_count(p_all_same) == 3 > same_title_count(p_one_fixed) == 1

    def test_cast_quoted_name_and_outline_alias_resolution(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_cast.py :: out["name"] = proper
        Becomes: out["name"] = raw_name
        Killed by: examples/story/src/uclone_x_story/start_cast.py :: out.setdefault((kind, alias.casefold()), cid)
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if m := _PAREN_ENTITY.match(text):
        Becomes: if False and (m := _PAREN_ENTITY.match(text)):
        Killed by: examples/story/src/uclone_x_story/start_state.py :: if a.casefold() != name.casefold() and not words_outside(a, allowed)
        Becomes: if a.casefold() != name.casefold()
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: return word in _EXACT_STATUS_WORDS
        Becomes: return True
        Killed by: examples/story/src/uclone_x_story/start_details.py :: if kind == "threads" and name.casefold() in physical_item_names:
        Becomes: if False:
        """
        from uclone_x_story.start_cast import normalize_cast_entry
        from uclone_x_story.start_details import _kg_params
        from uclone_x_story.start_outline import outline_from
        from uclone_x_story.start_state import _aliases

        assert _aliases(["장大爷", "역장"], "노년의 기차역장") == ["역장"]

        normalized = normalize_cast_entry({"name": "검은 고양이 '먹물'", "aliases": ["고양이"]})
        assert normalized["name"] == "먹물"
        assert normalized["aliases"] == ["검은 고양이", "고양이"]
        assert normalize_cast_entry({"name": "'먹물'"})["name"] == "먹물"

        cast_lines = [
            {
                "id": "meokmul",
                "kind": "characters",
                **normalized,
            },
            {
                "id": "junho",
                "kind": "characters",
                "name": "이준호",
                "aliases": ["준호"],
            },
            {
                "id": "old_house",
                "kind": "places",
                "name": "민호의 과거 집",
            },
        ]
        outline = outline_from(
            {
                "chapters": [
                    {
                        "title": "1장",
                        "scenes": [
                            {
                                "title": "고양이의 흔적",
                                "summary": "검은 고양이가 골목을 스친다.",
                                "beats": ["발자국이 남는다"],
                                "characters": ["검은 고양이", "이준호 (시계 수리공)"],
                            }
                        ],
                    }
                ]
            },
            cast_lines,
        )
        assert outline.chapters[0].scenes[0].characters == ["meokmul", "junho"]

        transitions = parse_plot_transitions(
            {
                "transitions": [
                    {
                        "scene": "[ch01.s01]",
                        "subject": "이준호 (시계 수리공)",
                        "set": {"status": "죽을 만큼 괴로움"},
                        "relations": {"검은 고양이 '먹물'": "추적"},
                    }
                ]
            },
            outline,
            cast_lines,
        )
        assert len(transitions) == 1
        assert transitions[0].entity_id == "junho"
        assert transitions[0].set == {"condition": "죽을 만큼 괴로움"}
        assert transitions[0].relations == {"meokmul": "추적"}

        wanted, dropped = _kg_params(
            {
                "threads": [
                    {
                        "name": "붉은 인주함",
                        "profile": "인주함의 행방을 둘러싼 추적.",
                        "characters": ["준호"],
                        "planted_chapter": 1,
                        "pay_off_chapter": 1,
                    },
                    {
                        "name": "인주함의 행방",
                        "profile": "사라진 인주함을 찾는 추적.",
                        "characters": ["준호"],
                        "planted_chapter": 1,
                        "pay_off_chapter": 1,
                    },
                ],
                "items": [
                    {"name": "붉은 인주함", "profile": "왕실의 비밀이 담긴 함."},
                    {"name": "민호의 과거 집", "profile": "이미 장소에 있는 집."},
                ],
            },
            cast_lines,
            lambda _: "ch01.s01",
        )
        assert dropped == []
        assert [(p.kind, p.name) for p, _ in wanted] == [
            ("threads", "인주함의 행방"),
            ("items", "붉은 인주함"),
        ]

    async def test_temporal_kg_delta_and_repetitive_scene_detection(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: and bigram_overlap(prev_v, v) >= 0.65
        Becomes: and False
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if prev is not None and prev != v and k not in ("gender",):
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if kind in ("state", "relations"):
        Becomes: if kind == "relations":
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: at = _text(change.get("at")) if show_at else ""
        Becomes: at = ""
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if item_match is None or item_match[1] != "items":
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if not kept_set and not kept_rel:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: return ", ".join(items) if len(items) == 1 else f"[{', '.join(items)}]"
        Becomes: return v
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: known_entities[cid] = "characters"
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: rel_items.setdefault(rel_target, rel_is)
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/start_outline.py :: is_rep = bigram_overlap(prev_body, cur_body) >= MAX_SCENE_BIGRAM
        Becomes: is_rep = False
        Killed by: examples/story/src/uclone_x_story/start_outline.py :: is_rep = cosine(vectors[global_j], vectors[global_i]) >= MAX_SCENE_COSINE
        Becomes: is_rep = False
        Killed by: examples/story/src/uclone_x_story/start_outline.py :: if len(body_text) >= MIN_SCENE_COMPARE_CHARS and any(
        Becomes: if False and any(
        """
        from collections.abc import Sequence

        from uclone_x_story.kg_plan import entering_world_state
        from uclone_x_story.start_outline import _scene_dicts, repetitive_scenes

        outline = Outline(
            chapters=[
                Chapter(
                    id="ch01",
                    title="1장",
                    scenes=[
                        Scene(
                            id="ch01.s01",
                            title="첫 장면",
                            summary="첫 요약",
                            characters=["harin", "yunho"],
                        ),
                        Scene(
                            id="ch01.s02",
                            title="둘째 장면",
                            summary="둘째 요약",
                            characters=["harin", "yunho"],
                        ),
                    ],
                ),
                Chapter(
                    id="ch02",
                    title="2장",
                    scenes=[
                        Scene(
                            id="ch02.s01",
                            title="셋째 장면",
                            summary="셋째 요약",
                            characters=["harin", "yunho"],
                        ),
                    ],
                ),
            ]
        )
        base_codex = CodexIndex(
            items=(
                CodexItem(
                    kind="characters",
                    entry=CharacterEntry(
                        id="harin",
                        name="하린",
                        state={
                            "status": "alive",
                            "stance": "경계하며 거리를 둠",
                            "possesses": ["deungbul"],
                        },
                    ),
                ),
                CodexItem(
                    kind="characters",
                    entry=CharacterEntry(
                        id="yunho",
                        name="윤호",
                        state={"status": "alive"},
                    ),
                ),
                CodexItem(
                    kind="items",
                    entry=ItemEntry(
                        id="deungbul",
                        name="등불",
                    ),
                ),
            )
        )
        # Non-item `possesses` normalizes to `knowledge` when known items exist in lookup,
        # and `relation` + `is` inside `set` normalizes to `relations`
        parsed_non_item = parse_plot_transitions(
            {
                "transitions": {
                    "[ch01.s01]": [
                        {
                            "subject": "하린",
                            "set": {
                                "possesses": "해독된 주파수",
                                "relation": "윤호",
                                "is": "동료",
                            },
                        }
                    ]
                }
            },
            outline,
            [],
            base_codex,
        )
        assert len(parsed_non_item) == 1
        assert parsed_non_item[0].set == {"knowledge": "해독된 주파수"}
        assert parsed_non_item[0].relations == {"yunho": "동료"}

        # `cast_lines` baseline fallback allows transitions for characters not yet in `base_codex`
        cast_only, _ = validate_kg_trajectory(
            outline,
            CodexIndex(),
            [
                SceneTransition(
                    scene_id="ch01.s01",
                    entity_id="minsu",
                    set={"stance": "진실을 추적함"},
                )
            ],
            cast_lines=[{"id": "minsu", "name": "민수", "kind": "characters", "status": "alive"}],
        )
        assert len(cast_only) == 1

        transitions = [
            SceneTransition(
                scene_id="ch01.s01",
                entity_id="harin",
                set={"stance": "윤호의 비밀을 파헤치기로 결심함", "location": "등대 입구"},
                relations={"yunho": "의심하며 추궁함"},
                note="하린이 윤호를 추궁하기 시작한다",
            ),
            SceneTransition(
                scene_id="ch01.s02",
                entity_id="harin",
                # Cross-field near-duplicate (stance -> goal with bigram_overlap >= 0.65) is filtered out,
                # while new location supersedes ch01.s01's location!
                set={"goal": "윤호의 비밀을 파헤치기로 굳게 결심함", "location": "등대 꼭대기"},
                relations={"yunho": "단서를 공유하는 공조 관계"},
                note="하린과 윤호가 단서를 공유한다",
            ),
        ]
        validated, _ = validate_kg_trajectory(outline, base_codex, transitions)
        assert len(validated) == 2
        assert validated[0].set == {
            "stance": "윤호의 비밀을 파헤치기로 결심함",
            "location": "등대 입구",
        }
        assert validated[1].set == {"location": "등대 꼭대기"}
        assert validated[1].relations == {"yunho": "단서를 공유하는 공조 관계"}

        codex_now = effective_codex(base_codex, [], [t.record() for t in validated])
        bundle_s01 = scene_context(
            outline, "ch01.s01", codex_now, previous_text=None, story={}, request=None
        )
        guidance_s01 = kg_scene_guidance(bundle_s01, codex_now)
        assert "stance=윤호의 비밀을 파헤치기로 결심함 (from 경계하며 거리를 둠)" in guidance_s01
        assert "relation(윤호)=의심하며 추궁함" in guidance_s01

        bundle_s02 = scene_context(
            outline, "ch01.s02", codex_now, previous_text=None, story={}, request=None
        )
        guidance_s02 = kg_scene_guidance(bundle_s02, codex_now)
        assert "[ch01.s01]" in guidance_s02
        assert "relation(윤호)=단서를 공유하는 공조 관계 (from 의심하며 추궁함)" in guidance_s02

        # Entering ch02.s01, ch01.s01's superseded location ("등대 입구") and relation ("의심하며 추궁함")
        # are folded away; only active values appear. Also check when an earlier progression is
        # completely superseded (only set location="등대 입구" in ch01.s01 and location="등대 꼭대기" in ch01.s02).
        bundle_ch02 = scene_context(
            outline, "ch02.s01", codex_now, previous_text=None, story={}, request=None
        )
        guidance_ch02 = kg_scene_guidance(bundle_ch02, codex_now)
        assert "location=등대 입구" not in guidance_ch02
        assert "location=등대 꼭대기" in guidance_ch02

        codex_superseded = effective_codex(
            base_codex,
            [],
            [
                SceneTransition(
                    scene_id="ch01.s01",
                    entity_id="harin",
                    set={"location": "등대 입구"},
                    note="입구 도착",
                ).record(),
                SceneTransition(
                    scene_id="ch01.s02",
                    entity_id="harin",
                    set={"location": "등대 꼭대기"},
                    note="꼭대기 이동",
                ).record(),
            ],
        )
        bundle_superseded = scene_context(
            outline, "ch02.s01", codex_superseded, previous_text=None, story={}, request=None
        )
        guidance_superseded = kg_scene_guidance(bundle_superseded, codex_superseded)
        assert "입구 도착" not in guidance_superseded
        assert "[ch01.s01]" not in guidance_superseded
        assert "location=등대 꼭대기; (꼭대기 이동) [ch01.s02]" in guidance_superseded

        entering_ch02 = entering_world_state(codex_now, outline, "ch02.s01")
        assert (
            "하린 (characters): status=alive, stance=윤호의 비밀을 파헤치기로 결심함, possesses=등불, location=등대 꼭대기"
            in entering_ch02
        )
        assert "relation(윤호)=단서를 공유하는 공조 관계" in entering_ch02

        # Check repetitive_scenes via bigram overlap and semantic embedding cosine
        prior_scenes = [
            {
                "title": "폐역의 재회",
                "summary": "수현이 비 내리는 폐역 대합실에서 지훈의 낡은 수첩을 발견하고 과거의 진실을 떠올린다.",
                "beats": [
                    "수현이 벤치 위에서 수첩을 줍는다",
                    "수첩 속 기록을 읽으며 오해를 깨닫는다",
                ],
            }
        ]
        draft_scenes = [
            {
                "title": "수첩의 기록",
                "summary": "수현이 비 내리는 폐역 대합실에서 지훈의 낡은 수첩을 발견하고 과거의 진실을 마주한다.",
                "beats": [
                    "수현이 벤치 위에서 수첩을 펼친다",
                    "수첩 속 기록을 읽으며 진실을 깨닫는다",
                ],
            },
            {
                "title": "밤열차의 도착",
                "summary": "청량리행 마지막 열차가 플랫폼으로 들어오며 기적 소리가 밤공기를 가른다.",
                "beats": ["역무원이 깃발을 흔든다", "승객들이 우산을 접고 객차에 오른다"],
            },
        ]

        class _FakeEmbedder:
            async def embed(self, texts: Sequence[str]) -> list[list[float]]:
                # Give prior_scenes[0] and draft_scenes[1] identical unit vectors (cosine = 1.0)
                out: list[list[float]] = []
                for idx, _ in enumerate(texts):
                    if idx in (0, 2):
                        out.append([1.0, 0.0, 0.0])
                    else:
                        out.append([0.0, 1.0, 0.0])
                return out

        rep = await repetitive_scenes(draft_scenes, prior_scenes, _FakeEmbedder())
        assert len(rep) == 2
        assert rep[0][:2] == ("폐역의 재회", "수첩의 기록")
        assert rep[1][:2] == ("폐역의 재회", "밤열차의 도착")

        # Check _scene_dicts intra-chapter deduplication
        deduped = _scene_dicts(
            [prior_scenes[0], draft_scenes[0], draft_scenes[1]],
            "ch01",
            {},
        )
        assert len(deduped) == 2
        assert [d["title"] for d in deduped] == [
            "폐역의 재회",
            "밤열차의 도착",
        ]


class TestHowManyTransitionsWentUnplaced:
    """#2376: the start record says how many of the plot reading's transitions named no scene
    the outline has, so a live run can measure what #2375's exact label match drops."""

    def test_a_transition_without_a_scene_label_is_counted(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: unplaced["label_missing"] += 1
        Becomes: pass
        """
        outline = _sample_outline()
        found = {
            "transitions": [
                {"scene": "[ch01.s02]", "subject": "하린"},
                {"chapter": 1, "subject": "하린"},
                {"scene": "[ch09.s01]", "subject": "하린"},
                "not a transition",
            ]
        }
        assert unplaced_transitions(found, outline) == {
            "label_missing": 1,
            "label_unknown": 1,
            "not_a_transition": 1,
        }
        nothing = {"label_missing": 0, "label_unknown": 0, "not_a_transition": 0}
        assert unplaced_transitions(None, outline) == nothing
        assert unplaced_transitions({"transitions": "x"}, outline) == nothing

    async def test_the_start_record_keeps_the_unplaced_count(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_details.py :: plot_kg_unplaced=unplaced_transitions(found, outline),
        Becomes: plot_kg_unplaced=0,
        """
        kg_reply: dict[str, Any] = {
            "threads": [],
            "items": [],
            "transitions": [
                {"scene": "[ch01.s01]", "subject": "하린", "set": {"stance": "resolute"}},
                {"chapter": 1, "subject": "하린", "set": {"stance": "afraid"}},
            ],
        }
        model = ScriptedModel(kg=kg_reply)
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success and isinstance(started.output, dict)
        out = cast("dict[str, Any]", started.output)
        start_file = tmp_path / "stories" / str(out["open_story_id"]) / START_FILE
        saved = cast("dict[str, Any]", yaml.safe_load(start_file.read_text(encoding="utf-8")))
        assert saved["plot_kg_unplaced"] == {
            "label_missing": 1,
            "label_unknown": 0,
            "not_a_transition": 0,
        }
        assert "plot_kg_unplaced" not in out


class TestTheFindingsOfThe2376LiveRun:
    """#2378: a transition is placed by the scene key the code wrote; a thread never takes
    another kind's entry; the plan check is told the threads a scene plants; a draft in
    another script is drawn again."""

    def test_transitions_keyed_by_the_scene_label_are_placed_and_each_drop_has_its_reason(
        self,
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: unplaced["label_unknown"] += len(entries)
        Becomes: pass
        """
        outline = _sample_outline()
        cast_lines = [{"id": "harin", "name": "하린", "kind": "characters"}]
        found = {
            "transitions": {
                "[ch01.s02]": [{"subject": "하린", "set": {"stance": "resolute"}}],
                "[ch01.s01]": [],
                "[ch09.s01]": [{"subject": "하린"}, {"subject": "하린"}],
            }
        }
        parsed = parse_plot_transitions(found, outline, cast_lines)
        assert [(t.scene_id, t.set) for t in parsed] == [("ch01.s02", {"stance": "resolute"})]
        assert unplaced_transitions(found, outline) == {
            "label_missing": 0,
            "label_unknown": 2,
            "not_a_transition": 0,
        }

    async def test_the_reading_is_asked_for_its_transitions_under_each_scene_label(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_details.py :: slots = ", ".join(f'"[{s.id}]": []' for c in outline.chapters for s in c.scenes)
        Becomes: slots = ""
        """
        model = ScriptedModel()
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success
        kg_prompt = next(p for p in model.prompts if KG_MARK in p)
        assert '"transitions": {"[ch01.s01]": [], ' in kg_prompt

    def test_a_threads_stage_is_not_planned_on_a_place_of_the_same_id(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if tr.kind != known_entities[tr.entity_id]:
        Becomes: if False:
        """
        outline = _sample_outline()
        codex = CodexIndex(
            items=(
                CodexItem(
                    kind="places",
                    entry=PlaceEntry(id="ieobuui_bae", name="이어부의 배", profile="배."),
                ),
            )
        )
        planned = [
            SceneTransition(
                scene_id="ch01.s01",
                entity_id="ieobuui_bae",
                kind="threads",
                set={"thread_stage": "planted"},
            )
        ]
        kept, report = validate_kg_trajectory(outline, codex, planned)
        assert kept == []
        assert [d["reason"] for d in report.dropped] == ["kind_mismatch"]

    def test_the_plan_check_is_told_the_threads_the_scene_plants_and_pays_off(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/plan_check.py :: if item.kind == "threads" and scene_id in (entry.planted_in, entry.pay_off_by)
        Becomes: if False
        """
        outline = _sample_outline()
        codex = _sample_codex()
        scene = outline.chapters[0].scenes[0]
        threads = scene_threads(codex, "ch01.s01")
        assert threads == ["중심 갈등: 안개 속 목소리의 정체를 밝히려 한다. (set up in this scene)"]
        assert scene_threads(codex, "ch02.s01") == [
            "반전: 목소리는 하린 자신의 것이었다. (paid off in this scene)"
        ]
        prompt = plan_prompt(
            scene,
            "하린은 등불을 들었다.",
            chapter=1,
            twist=None,
            twist_chapter=None,
            threads=threads,
        )
        assert "- 중심 갈등: 안개 속 목소리의 정체를 밝히려 한다. (set up in this scene)" in prompt
        assert "nor any planned thread" in prompt

    async def test_the_plan_check_of_a_written_scene_carries_its_threads(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: threads=scene_threads(codex, scene_id),
        Becomes: threads=[],
        """
        model = ScriptedModel()
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success
        checks = [p for p in model.prompts if p.startswith(PLAN_MARKER)]
        assert checks and "(set up in this scene)" in checks[0]

    async def test_a_scene_drafted_in_another_script_is_drawn_again(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: if refused is None and foreign:
        Becomes: if False:
        """
        foreign = _SCENE + " 因为我爱你"
        model = ScriptedModel(scenes=[foreign, _SCENE + " 먹물이 웃었다.", _SCENE])
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success and isinstance(started.output, dict)
        out = cast("dict[str, Any]", started.output)
        redraw = [p for p in model.prompts if "因为我爱你" in p and "Write scene" in p]
        assert len(redraw) == 1
        assert "outside Korean" in redraw[0]
        written = (tmp_path / "stories" / str(out["open_story_id"]) / "manuscript").glob("*.md")
        assert not any("因为我爱你" in p.read_text(encoding="utf-8") for p in written)


def _proposals(workspace: Path, story_id: str) -> list[dict[str, Any]]:
    folder = workspace / "stories" / story_id / "proposals"
    return [
        cast("dict[str, Any]", yaml.safe_load(p.read_text(encoding="utf-8")))
        for p in sorted(folder.glob("*.yaml"))
    ]


def _start_record(workspace: Path, story_id: str) -> dict[str, Any]:
    start_file = workspace / "stories" / story_id / START_FILE
    return cast("dict[str, Any]", yaml.safe_load(start_file.read_text(encoding="utf-8")))


def _written(workspace: Path, story_id: str) -> list[str]:
    folder = workspace / "stories" / story_id / "manuscript"
    return [p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))]


class TestTheFindingsOfThe2379Review:
    """#2380: a cast alias in another script is the story's own; every keyed transition is
    placed or counted; a redraft is kept only when it is better, and a scene still in
    another script is said; a thread named like a place keeps its lifecycle."""

    async def test_a_korean_scene_using_a_latin_alias_is_not_drawn_again(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: names = [n for n in self._story_names() if words_outside(n, scripts)]
        Becomes: names = []
        """
        cast_reply = json.dumps(
            {
                "characters": [
                    {
                        "name": "하린",
                        "aliases": ["Harin"],
                        "profile": "등대지기의 딸.",
                        "gender": "female",
                        "status": "alive",
                    },
                    {"name": "먹물", "profile": "사라진 검은 고양이.", "status": "실종"},
                ],
                "places": [{"name": "안개섬 등대", "profile": "섬 끝의 등대."}],
            },
            ensure_ascii=False,
        )
        scene = _SCENE + " 등대지기는 그녀를 Harin 이라고 불렀다."
        model = ScriptedModel(cast_replies=[cast_reply], scenes=[scene, scene, scene])
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success and isinstance(started.output, dict)
        assert not [p for p in model.prompts if "outside Korean" in p]
        story = str(cast("dict[str, Any]", started.output)["open_story_id"])
        assert any("Harin" in text for text in _written(tmp_path, story))

    async def test_a_latin_sentence_beside_a_latin_alias_is_still_drawn_again(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: foreign = words_outside(_without_names(text, names), scripts) if refused is None else []
        Becomes: foreign = words_outside(text, scripts) if refused is None else []
        """
        cast_reply = json.dumps(
            {
                "characters": [
                    {"name": "하린", "aliases": ["Harin"], "profile": "등대지기의 딸."},
                    {"name": "먹물", "profile": "사라진 검은 고양이."},
                ],
                "places": [{"name": "안개섬 등대", "profile": "섬 끝의 등대."}],
            },
            ensure_ascii=False,
        )
        scene = _SCENE + " Harin smiled, and suddenly everything changed."
        model = ScriptedModel(cast_replies=[cast_reply], scenes=[scene, _SCENE, _SCENE, _SCENE])
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success
        asked = [p for p in model.prompts if "outside Korean" in p]
        assert len(asked) == 1
        told = asked[0].split("outside Korean:", 1)[1].split(". Write", 1)[0]
        assert "smiled," in told and "Harin" not in told

    async def test_a_name_is_taken_out_only_as_a_whole_name(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: if any(c and script_of(c) in own for c in (before, after)):
        Becomes: if False:
        """
        cast_reply = json.dumps(
            {
                "characters": [
                    {"name": "하린", "aliases": ["Ann", "Harin"], "profile": "등대지기의 딸."},
                    {"name": "먹물", "aliases": ["Mo"], "profile": "사라진 검은 고양이."},
                ],
                "places": [{"name": "안개섬 등대", "profile": "섬 끝의 등대."}],
            },
            ensure_ascii=False,
        )
        scene = _SCENE + " Harin은 Annual MoMo 를 보았다."
        model = ScriptedModel(cast_replies=[cast_reply], scenes=[scene, _SCENE, _SCENE, _SCENE])
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success
        asked = [p for p in model.prompts if "outside Korean" in p]
        assert len(asked) == 1
        told = asked[0].split("outside Korean:", 1)[1].split(". Write", 1)[0]
        assert told.strip().split(", ") == ["Annual", "MoMo"]

    def test_a_single_object_under_a_label_is_placed_and_a_string_is_counted(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: listed = [value] if isinstance(value, dict) else listed
        Becomes: listed = listed
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: unplaced["not_a_transition"] += len(listed) - len(entries)
        Becomes: pass
        """
        outline = _sample_outline()
        cast_lines = [{"id": "harin", "name": "하린", "kind": "characters"}]
        found = {
            "transitions": {
                "[ch01.s01]": {"subject": "하린", "set": {"stance": "alert"}},
                "[ch01.s02]": ["하린이 문을 연다"],
            }
        }
        parsed = parse_plot_transitions(found, outline, cast_lines)
        assert [(t.scene_id, t.set) for t in parsed] == [("ch01.s01", {"stance": "alert"})]
        assert unplaced_transitions(found, outline) == {
            "label_missing": 0,
            "label_unknown": 0,
            "not_a_transition": 1,
        }

    async def test_a_redraft_with_more_foreign_words_is_not_kept_and_the_scene_says_so(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: and len(words_outside(_without_names(again, names), scripts)) < len(foreign)
        Becomes: and True
        Killed by: examples/story/src/uclone_x_story/start_scene.py :: notes.append(_still_foreign(self._scene_title(scene_id), still))
        Becomes: pass
        """
        first = _SCENE + " 因为我爱你"
        worse = _SCENE + " 因为 我爱 你好"
        model = ScriptedModel(scenes=[first, worse, _SCENE, _SCENE])
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success and isinstance(started.output, dict)
        story = str(cast("dict[str, Any]", started.output)["open_story_id"])
        written = _written(tmp_path, story)
        assert first in written and worse not in written
        said = cast("list[str]", _start_record(tmp_path, story).get("notes") or [])
        notes = " ".join(said)
        assert "still has words outside the story's language: 因为我爱你" in notes
        # The note names the scene as the person reads it, never by its id (#2381).
        assert re.search(r"ch\d\d\.s\d\d", notes) is None

    async def test_a_redraft_the_prose_check_refuses_is_not_kept(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_scene.py :: and prose_refusal(again) is None  # a redraft the prose check refuses is not kept
        Becomes: and True
        """
        first = _SCENE + " 因为我爱你"
        # No word outside Korean, so only the prose check (a looping sentence) refuses it.
        refused = "하린은 등대 계단을 올랐다. " + "먹물이 낮게 울었고 바람이 창을 두드렸다. " * 6
        model = ScriptedModel(scenes=[first, refused, _SCENE, _SCENE])
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success and isinstance(started.output, dict)
        story = str(cast("dict[str, Any]", started.output)["open_story_id"])
        written = _written(tmp_path, story)
        assert first in written and refused not in written

    async def test_a_thread_named_like_a_place_is_proposed_and_keeps_its_lifecycle(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_details.py :: params = _as_thread_of_its_own(params, state.brief.language)
        Becomes: pass
        """
        kg_reply: dict[str, Any] = {
            "threads": [
                {
                    "name": "안개섬 등대",
                    "profile": "등대에 얽힌 약속.",
                    "characters": ["하린"],
                    "planted_chapter": 1,
                    "pay_off_chapter": 2,
                }
            ],
            "items": [],
            "transitions": {},
        }
        model = ScriptedModel(kg=kg_reply)
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)
        started = await tool.execute(
            {"action": "start", "request": REQUEST + " 바로 써줘."}, _ctx(tmp_path)
        )
        assert started.success and isinstance(started.output, dict)
        story = str(cast("dict[str, Any]", started.output)["open_story_id"])

        def profile(p: dict[str, Any]) -> object:
            change = cast("dict[str, Any]", p.get("change") or {})
            return cast("dict[str, Any]", change.get("new_entry") or {}).get("profile")

        threads = [
            p
            for p in _proposals(tmp_path, story)
            if p.get("kind") == "threads" and profile(p) == "등대에 얽힌 약속."
        ]
        assert len(threads) == 1
        thread_id = str(threads[0]["entry_id"])
        planned = _start_record(tmp_path, story)["planned_transitions"]
        stages = [
            t["set"]["thread_stage"]
            for t in planned
            if t["entity_id"] == thread_id and t["kind"] == "threads"
        ]
        assert stages == ["planted", "paid_off"]


class TestRpgInteractiveSceneAndChronologicalKg:
    """Tests for RPG scene-by-scene interactive story building (`review_scene`) and dynamic
    chronological KG updates across scenes."""

    def test_chronological_kg_log_and_formatting_respect_scene_order(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/kg_plan.py :: if sid == before_scene_id:
        Becomes: if False:
        Killed by: examples/story/src/uclone_x_story/kg_plan.py :: f"- [{entry['scene_id']}] {entry['title']}: " + " | ".join(entry["changes"])
        Becomes: ""
        """
        outline = _sample_outline()
        codex = CodexIndex(
            items=(
                CodexItem(
                    kind="characters",
                    entry=CharacterEntry(
                        id="harin",
                        name="하린",
                        state={"status": "alive"},
                        progressions=[
                            Progression(
                                at="ch01.s01",
                                set={"location": "안개섬 등대", "possesses": "은방울"},
                                relations={"meokmul": "추적 중"},
                                note="계단에서 은방울을 줍는다.",
                            ),
                            Progression(
                                at="ch01.s02",
                                set={"stance": "결연함"},
                                note="목소리의 정체를 직감한다.",
                            ),
                        ],
                    ),
                ),
                CodexItem(
                    kind="characters",
                    entry=CharacterEntry(id="meokmul", name="먹물", state={"status": "실종"}),
                ),
                CodexItem(
                    kind="threads",
                    entry=ThreadEntry(
                        id="promise",
                        name="등대의 약속",
                        planted_in="ch01.s01",
                        pay_off_by="ch02.s01",
                        state={"thread_stage": "open"},
                        relations={"harin": "involved"},
                        progressions=[
                            Progression(
                                at="ch01.s01",
                                set={"thread_stage": "planted"},
                                note="약속이 심어진다.",
                            ),
                        ],
                    ),
                ),
            )
        )
        before_s02 = chronological_kg_log(codex, outline, before_scene_id="ch01.s02")
        assert len(before_s02) == 1
        assert before_s02[0]["scene_id"] == "ch01.s01"
        assert any("은방울" in c for c in before_s02[0]["changes"])

        full_log = chronological_kg_log(codex, outline)
        assert [entry["scene_id"] for entry in full_log] == ["ch01.s01", "ch01.s02"]

        formatted = format_chronological_kg_log(codex, outline, before_scene_id="ch01.s02")
        assert "- [ch01.s01] 계단:" in formatted
        assert "[ch01.s02]" not in formatted
        assert (
            "등대의 약속: stage=planted (planted_in=ch01.s01, pay_off_by=ch02.s01, involves=[하린])"
            in formatted
        )

        parsed_single = parse_scene_transitions(
            [
                {"subject": "하린", "set": {"stance": "집중"}, "note": "소리에 귀를 기울인다."},
                {"subject": "하린과 먹물", "set": {"location": "안개섬 등대"}},
            ],
            "ch01.s02",
            [
                {"id": "harin", "name": "하린", "kind": "characters"},
                {"id": "meokmul", "name": "먹물", "kind": "characters"},
            ],
            codex,
        )
        assert len(parsed_single) == 3
        assert parsed_single[0].scene_id == "ch01.s02"
        assert parsed_single[0].set == {"stance": "집중"}
        assert [t.entity_id for t in parsed_single[1:]] == ["harin", "meokmul"]

    def test_unplanned_death_filter_and_choice_milestone_persistence(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_rpg.py :: return (not in_current_cast) or in_later_cast
        Becomes: return False
        Killed by: examples/story/src/uclone_x_story/start_rpg.py :: updated_set.setdefault("milestone", decision)
        Becomes: pass
        """
        from uclone_x_story.schemas import Proposal
        from uclone_x_story.start_rpg import (
            ensure_choice_transition,
            is_unplanned_death_proposal,
        )

        outline = _sample_outline()
        hallucinated = Proposal.model_validate(
            {
                "id": "p019",
                "kind": "characters",
                "entry_id": "harin",
                "status": "pending",
                "change": {"progression": {"at": "ch01.s01", "set": {"status": "dead"}}},
                "evidence": [{"scene_id": "ch01.s01", "quote": "하린은 안개 속으로 사라졌다."}],
                "proposed_at": "2026-10-17T00:00:00Z",
                "room_id": "room_a",
                "entry_digest": "0123456789abcdef",
            }
        )
        # harin appears in later outline scenes (ch01.s02, ch02.s01) and was not planned to die in ch01.s01
        assert is_unplanned_death_proposal(hallucinated, outline, []) is True
        # When planned_transitions explicitly plans the death in ch01.s01, it is NOT filtered out
        assert (
            is_unplanned_death_proposal(
                hallucinated,
                outline,
                [{"scene_id": "ch01.s01", "entity_id": "harin", "set": {"status": "dead"}}],
            )
            is False
        )

        ensured = ensure_choice_transition(
            [
                SceneTransition(
                    scene_id="ch01.s01",
                    entity_id="harin",
                    set={"goal": "고양이 찾기"},
                    note="수색 시작",
                )
            ],
            scene_id="ch01.s01",
            lead_id="harin",
            title="발자국 추적",
            summary="하린이 젖은 발자국을 따라간다.",
        )
        assert ensured[0].set["milestone"] == "발자국 추적 — 하린이 젖은 발자국을 따라간다."
        assert ensured[0].set["goal"] == "고양이 찾기"

    def test_wants_interactive_detects_rpg_and_scene_by_scene_requests(self) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_state.py :: return bool(_INTERACTIVE.search(request))
        Becomes: return False
        """
        assert wants_interactive("RPG 스타일로 선택지를 주며 씬별로 진행해줘")
        assert wants_interactive("Build an interactive scene by scene fantasy story")
        assert not wants_interactive(REQUEST)

    async def test_rpg_interactive_mode_steps_scene_by_scene_and_updates_kg_dynamically(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/start_rpg.py :: stage="review_scene",
        Becomes: stage="review_chapter",
        Killed by: examples/story/src/uclone_x_story/start_rpg.py :: done.append("kg_updated")
        Becomes: pass
        Killed by: examples/story/src/uclone_x_story/start_rpg.py :: done.append("scene_adapted")
        Becomes: pass
        """
        second_scene = (
            "아침이 밝자 하린은 마을 우물가로 내려갔다. 장터의 사람들이 등대 이야기를 수군거렸다. "
            "먹물은 생선 가게 앞에서 꼬리를 흔들었다. 하린은 빵 한 덩이를 사서 품에 넣었다."
        )
        third_scene = (
            "새벽빛이 등대 유리창을 비추었다. 하린은 마침내 안개 속에서 걸어 나오는 먹물을 안아 올렸다. "
            "방울 소리가 맑게 울려 퍼졌고 바다는 잔잔해졌다."
        )
        model = ScriptedModel(scenes=[_SCENE, second_scene, third_scene])
        tool = StoryStartTool(model_factory=lambda _c: model, embedder_factory=lambda _c: None)

        # 1. Start with interactive=True (or RPG phrasing)
        res1 = await tool.execute(
            {"action": "start", "request": REQUEST, "interactive": True},
            _ctx(tmp_path),
        )
        assert res1.success and isinstance(res1.output, dict)
        out1 = cast("dict[str, Any]", res1.output)
        story = str(out1["open_story_id"])
        assert out1["stage"] == "choose_premise"
        assert out1["interactive"] is True

        # 2. Choose premise 1 -> review_cast
        res2 = await tool.execute(
            {"action": "continue", "choice": 1},
            _ctx(tmp_path, story=story),
        )
        assert res2.success and isinstance(res2.output, dict)
        assert cast("dict[str, Any]", res2.output)["stage"] == "review_cast"

        # 3. Continue from cast -> review_outline
        res3 = await tool.execute(
            {"action": "continue", "go_on": True},
            _ctx(tmp_path, story=story),
        )
        assert res3.success and isinstance(res3.output, dict)
        assert cast("dict[str, Any]", res3.output)["stage"] == "review_outline"

        # 4. Lock outline in interactive mode -> stops at review_scene for ch01.s01 with 3 RPG choices
        res4 = await tool.execute(
            {"action": "continue", "go_on": True},
            _ctx(tmp_path, story=story),
        )
        assert res4.success and isinstance(res4.output, dict)
        out4 = cast("dict[str, Any]", res4.output)
        assert out4["stage"] == "review_scene"
        assert out4["scene_id"] == "ch01.s01"
        choices_s01 = cast("list[dict[str, Any]]", out4["scene_choices"])
        assert len(choices_s01) == 3
        assert choices_s01[1]["title"] == "젖은 발자국 추적"

        # 5. Select choice 2 (which dynamically adds new item '은방울' and sets possesses='은방울')
        res5 = await tool.execute(
            {"action": "continue", "choice": 2},
            _ctx(tmp_path, story=story),
        )
        assert res5.success and isinstance(res5.output, dict)
        out5 = cast("dict[str, Any]", res5.output)
        assert "kg_updated" in out5["done_now"]
        assert "scene" in out5["done_now"]
        assert out5["scene_written_last"] == "ch01.s01"
        assert out5["stage"] == "review_scene"
        assert out5["scene_id"] == "ch01.s02"
        # Verify the dynamic KG transition from ch01.s01 is now visible in ch01.s02's entering state & timeline!
        assert "은방울" in str(out5.get("scene_entering_kg") or "")
        assert "젖은 발자국 추적" in str(out5.get("scene_entering_kg") or "")
        s02_choice_prompts = [
            p
            for p in model.prompts
            if "Propose three distinct RPG choices for scene [ch01.s02]" in p
        ]
        assert (
            s02_choice_prompts
            and "Completed scenes & enacted decisions so far" in s02_choice_prompts[-1]
        )
        timeline_s02 = cast("list[dict[str, Any]]", out5.get("scene_kg_timeline") or [])
        assert any(
            entry["scene_id"] == "ch01.s01" and any("은방울" in c for c in entry["changes"])
            for entry in timeline_s02
        )

        # 6. Pass custom direction (`changes`) on ch01.s02 -> adapts scene & dynamically adds '황동 열쇠'
        res6 = await tool.execute(
            {"action": "continue", "changes": "바닥 틈에서 열쇠를 찾아 지하실 문을 연다"},
            _ctx(tmp_path, story=story),
        )
        assert res6.success and isinstance(res6.output, dict)
        out6 = cast("dict[str, Any]", res6.output)
        assert "scene_adapted" in out6["done_now"]
        assert "kg_updated" in out6["done_now"]
        assert out6["scene_written_last"] == "ch01.s02"
        assert out6["stage"] == "review_scene"
        assert out6["scene_id"] == "ch02.s01"
        # Chronological timeline entering ch02.s01 now has both ch01.s01 and ch01.s02 transitions in order
        timeline_ch02 = cast("list[dict[str, Any]]", out6.get("scene_kg_timeline") or [])
        assert [entry["scene_id"] for entry in timeline_ch02] == ["ch01.s01", "ch01.s02"]

        # 7. Complete the final scene ch02.s01 -> reaches stage="done"
        res7 = await tool.execute(
            {"action": "continue", "choice": 1},
            _ctx(tmp_path, story=story),
        )
        assert res7.success and isinstance(res7.output, dict)
        out7 = cast("dict[str, Any]", res7.output)
        assert out7["stage"] == "done"
        assert out7["scenes_written"] == ["ch01.s01", "ch01.s02", "ch02.s01"]
