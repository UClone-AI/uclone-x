"""A written scene held to its plan and to the settled plot (`uclone_x.story.plan_check`)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from uclone_x.story.context import CodexIndex, CodexItem
from uclone_x.story.plan_check import (
    MAX_UNPLANNED,
    PLAN_MARKER,
    absent_cast,
    asks_model,
    plan_findings,
    plan_prompt,
)
from uclone_x.story.schemas import CharacterEntry, Scene

_CODEX = CodexIndex(
    (
        CodexItem(kind="characters", entry=CharacterEntry(id="harin", name="하린")),
        CodexItem(
            kind="characters",
            entry=CharacterEntry(id="meokmul", name="먹물", aliases=["검은 고양이"]),
        ),
    )
)

_SCENE = Scene(
    id="ch01.s01",
    title="계단",
    summary="하린이 등대에 오른다.",
    characters=["harin", "meokmul"],
    beats=["바람이 창을 두드린다", "누군가 하린의 이름을 부른다"],
)

_TEXT = (
    "하린은 등대 계단을 올랐다. 바람이 창을 두드렸고, 먹물이 발밑을 따라왔다. "
    "꼭대기에 닿자 안개 속에서 누군가 하린의 이름을 불렀다."
)
_TWIST = "목소리는 사라진 고양이를 찾던 하린 자신의 것이었다."
_TWIST_LINE = "그 목소리는 바로 하린 자신의 목소리였다"
_TWIST_TEXT = f"{_TEXT} {_TWIST_LINE}."


def _reply(**found: Any) -> str:
    return json.dumps(found, ensure_ascii=False)


def _held(reply: str, text: str = _TEXT, *, chapter: int = 1, twist_chapter: int | None = 3):
    return plan_findings(
        _SCENE, text, reply, chapter=chapter, twist=_TWIST, twist_chapter=twist_chapter
    )


class TestAPlannedCharacterWhoNeverAppears:
    def test_is_found_by_code_and_named_for_the_person(self) -> None:
        """Killed by: src/uclone_x/story/plan_check.py :: if not keys or any(key in haystack for key in keys):
        Becomes: if True:
        """
        [finding] = absent_cast(_CODEX, _SCENE, "하린은 등대 계단을 올랐다. 바람이 불었다.")
        assert finding.kind == "plan_cast_absent"
        assert finding.note == "‘계단’ 장면 계획에 있던 먹물이 쓴 장면에는 나오지 않습니다."
        assert "meokmul" not in finding.note and "ch01" not in finding.note

    def test_a_name_with_its_particle_or_an_alias_counts_as_appearing(self) -> None:
        assert absent_cast(_CODEX, _SCENE, _TEXT) == []
        assert absent_cast(_CODEX, _SCENE, "하린은 검은 고양이를 안았다.") == []

    def test_one_word_of_a_longer_name_counts_as_appearing(self) -> None:
        """Killed by: src/uclone_x/story/plan_check.py :: words = [w for n in called for w in (n, *n.split())]
        Becomes: words = list(called)
        """
        codex = CodexIndex(
            (CodexItem(kind="characters", entry=CharacterEntry(id="elder", name="마을 장로")),)
        )
        scene = _SCENE.model_copy(update={"characters": ["elder"]})
        assert absent_cast(codex, scene, "장로는 의자에 앉아 조용히 웃었다.") == []
        [finding] = absent_cast(codex, scene, "이진호는 혼자 산길을 걸었다.")
        assert "마을 장로가 쓴 장면에는 나오지 않습니다" in finding.note

    def test_a_planned_id_the_codex_does_not_have_is_not_looked_for(self) -> None:
        """Killed by: src/uclone_x/story/plan_check.py :: if found is None:
        Becomes: if False:
        """
        scene = _SCENE.model_copy(update={"characters": ["harin", "rejected_one"]})
        assert absent_cast(_CODEX, scene, _TEXT) == []


class TestAPlannedBeatTheSceneLeavesOut:
    def test_the_beat_the_reading_says_is_not_done_is_shown(self) -> None:
        """Killed by: src/uclone_x/story/plan_check.py :: if 1 <= number <= len(scene.beats) and _false(read.get("done")):
        Becomes: if False:
        """
        reply = _reply(
            beats=[
                {"beat": 1, "done": True, "quote": "바람이 창을 두드렸고"},
                {"beat": 2, "done": False, "quote": ""},
            ]
        )
        [finding] = _held(reply)
        assert finding.kind == "plan_beat_missing"
        assert "“누군가 하린의 이름을 부른다” 부분이 쓴 장면에 보이지 않습니다" in finding.note

    def test_a_beat_number_outside_the_plan_or_an_unclear_answer_is_not_a_finding(
        self,
    ) -> None:
        reply = _reply(
            beats=[{"beat": 7, "done": False}, {"beat": 1, "done": "maybe"}, {"beat": True}]
        )
        assert _held(reply) == []


class TestTheTwistToldEarly:
    def test_a_quoted_reveal_before_the_twist_chapter_is_shown(self) -> None:
        """Killed by: src/uclone_x/story/plan_check.py :: return bool(twist) and twist_chapter is not None and chapter < twist_chapter
        Becomes: return bool(twist) and twist_chapter is not None and chapter > twist_chapter
        Killed by: src/uclone_x/story/plan_check.py :: if quote_found(quote, text):
        Becomes: if True:
        """
        reply = _reply(twist_revealed=True, twist_quote=_TWIST_LINE)
        [finding] = _held(reply, _TWIST_TEXT)
        assert finding.kind == "twist_early"
        assert finding.quote == _TWIST_LINE
        assert "반전은 3장에서 드러나기로 정했는데, 이 문장이 1장에서 먼저 드러냅니다" in (
            finding.note
        )
        # A quote the scene does not hold is not evidence.
        assert _held(_reply(twist_revealed=True, twist_quote="하린은 모든 것을 알았다")) == []

    def test_from_the_twist_chapter_on_it_is_neither_asked_nor_shown(self) -> None:
        reply = _reply(twist_revealed=True, twist_quote=_TWIST_LINE)
        assert _held(reply, _TWIST_TEXT, chapter=3) == []
        assert _TWIST in plan_prompt(_SCENE, _TEXT, chapter=1, twist=_TWIST, twist_chapter=3)
        assert _TWIST not in plan_prompt(_SCENE, _TEXT, chapter=3, twist=_TWIST, twist_chapter=3)
        bare = _SCENE.model_copy(update={"beats": []})
        assert asks_model(bare, chapter=1, twist=_TWIST, twist_chapter=3)
        assert not asks_model(bare, chapter=3, twist=_TWIST, twist_chapter=3)
        assert not asks_model(bare, chapter=1, twist=None, twist_chapter=3)


class TestAnUnplannedTurn:
    def test_a_quoted_event_is_shown_and_an_unquoted_one_is_not(self) -> None:
        """Killed by: src/uclone_x/story/plan_check.py :: if quote in kept or not quote_found(quote, text):
        Becomes: if quote in kept:
        """
        line = "먹물은 계단 아래로 떨어져 다시 일어나지 않았다"
        text = f"{_TEXT} {line}."
        reply = _reply(unplanned=[{"quote": line}, {"quote": "하린은 섬을 영영 떠났다"}])
        [finding] = _held(reply, text)
        assert finding.kind == "plan_unplanned_event"
        assert finding.note == f"‘계단’ 장면의 “{line}” — 장면 계획에 없던 큰 사건입니다."

    def test_at_most_a_few_are_shown_per_scene(self) -> None:
        """Killed by: src/uclone_x/story/plan_check.py :: if len(kept) > MAX_UNPLANNED:
        Becomes: if False:
        """
        lines = ["하린은 등대 계단을 올랐다", "먹물이 발밑을 따라왔다", "꼭대기에 닿자 안개 속에서"]
        found = _held(_reply(unplanned=[{"quote": q} for q in lines]))
        assert len(found) == MAX_UNPLANNED


class TestACleanScene:
    def test_a_scene_that_keeps_its_plan_has_no_finding(self) -> None:
        reply = _reply(
            beats=[
                {"beat": 1, "done": True, "quote": "바람이 창을 두드렸고"},
                {"beat": 2, "done": True, "quote": "누군가 하린의 이름을 불렀다"},
            ],
            twist_revealed=False,
            twist_quote="",
            unplanned=[],
        )
        assert absent_cast(_CODEX, _SCENE, _TEXT) == []
        assert _held(reply) == []

    def test_an_unreadable_reply_is_an_error_never_a_clean_scene(self) -> None:
        with pytest.raises(ValueError):
            _held("잘 모르겠습니다")

    def test_the_prompt_is_marked_and_lists_the_beats(self) -> None:
        prompt = plan_prompt(_SCENE, _TEXT, chapter=1, twist=None, twist_chapter=None)
        assert prompt.startswith(PLAN_MARKER)
        assert "1. 바람이 창을 두드린다\n2. 누군가 하린의 이름을 부른다" in prompt
        assert _TEXT in prompt
