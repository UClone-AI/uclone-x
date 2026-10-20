"""#2307: a scene that repeats the scene before is drafted once more.

In the 2026-10-12 live run of #2305, qwen3:8b wrote `ch02.s04` as `ch02.s03` again with a
paragraph added (`tests/fixtures/story_scene_repeat/scenes.json`). The cause: a scene
shorter than the 1,200 characters the Writer is given of "the end of the previous scene"
reached it whole, and the model wrote it out again. Now the end is at most a third of the
scene before, and a draft that shares most of its character runs with the scene before
(`prose.repeated_share`) is drafted once more. The measure reads characters, not words, so
it reads a script written without spaces as one written with them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.unit import test_story_start as T
from uclone_x_story.context import TAIL_PARTS, CodexIndex, scene_context
from uclone_x_story.prose import REPEAT_SHARE, repeated_share
from uclone_x_story.schemas import Outline
from uclone_x_story.start import REPEAT_MARK

SCENES = json.loads(
    (
        Path(__file__).resolve().parents[1] / "fixtures" / "story_scene_repeat" / "scenes.json"
    ).read_text("utf-8")
)["scenes"]


class TestTheMeasure:
    def test_the_live_copy_repeats_and_the_scene_before_it_does_not(self) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/prose.py :: return len(runs & _runs(before)) / len(runs)
        Becomes: return len(runs | _runs(before)) / len(runs)
        """
        assert repeated_share(SCENES["ch02.s03"], SCENES["ch02.s04"]) >= REPEAT_SHARE
        assert repeated_share(SCENES["ch02.s02"], SCENES["ch02.s03"]) < 0.1

    def test_the_measure_reads_characters_in_any_script(self) -> None:
        before = "She climbed the stairs. The wind beat on the glass. The cat followed her up."
        copy = before + " At the top, someone called her name."
        assert repeated_share(before, copy) >= REPEAT_SHARE
        assert repeated_share(before, "The lamp went dark, and the sea was quiet again.") < 0.1
        japanese = "彼女は階段を上った。風が窓を叩いた。猫が後をついてきた。"
        assert repeated_share(japanese, japanese + "上で誰かが名前を呼んだ。") >= REPEAT_SHARE
        assert repeated_share(japanese, "灯りが消え、海はまた静かになった。") < 0.1

    def test_a_text_too_short_to_measure_repeats_nothing(self) -> None:
        assert repeated_share("하린", "하린") == 0.0


class _Again(T.ScriptedModel):
    """The scripted model, with what it writes when told its draft repeats the scene
    before."""

    def __init__(self, again: str, **parts: Any) -> None:
        super().__init__(**parts)
        self.again = again

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        if REPEAT_MARK in prompt:
            self.redrafts.append(prompt)
            return self.again
        return await super().complete(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )


async def _first_chapter(model: T.ScriptedModel, workspace: Path) -> dict[str, str]:
    """The story asked for at once, so chapter 1's two scenes are written: their texts."""
    result = await T._tool(model).execute(  # pyright: ignore[reportPrivateUsage]
        {"action": "start", "request": T.REQUEST + " 바로 써줘."},
        T._ctx(workspace),  # pyright: ignore[reportPrivateUsage]
    )
    assert result.success, result.error
    assert isinstance(result.output, dict)
    assert result.output["scenes_written"] == ["ch01.s01", "ch01.s02"]
    work = T._work(workspace, str(result.output["open_story_id"]))  # pyright: ignore[reportPrivateUsage]
    texts: dict[str, str] = {}
    for scene_id in ("ch01.s01", "ch01.s02"):
        found = work.manuscript(scene_id)
        assert found is not None
        texts[scene_id] = found.text.strip()
    return texts


class TestARepeatIsDraftedOnceMore:
    async def test_the_second_draft_is_saved_when_it_goes_on(self, tmp_path: Path) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/start_scene.py :: if share < REPEAT_SHARE:
        Becomes: if True:
        """
        model = _Again(SCENES["ch02.s02"], scenes=[SCENES["ch02.s03"], SCENES["ch02.s04"]])

        texts = await _first_chapter(model, tmp_path)

        assert len(model.redrafts) == 1
        assert texts == {"ch01.s01": SCENES["ch02.s03"], "ch01.s02": SCENES["ch02.s02"]}
        # Asked again with the scene's own prompt, told why.
        assert "Write scene" in model.redrafts[0]

    async def test_a_second_draft_that_repeats_as_much_is_not_taken(self, tmp_path: Path) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/start_scene.py :: if again and prose_refusal(again) is None and repeated_share(previous, again) < share:
        Becomes: if again and prose_refusal(again) is None:
        """
        model = _Again(SCENES["ch02.s03"], scenes=[SCENES["ch02.s03"], SCENES["ch02.s04"]])

        texts = await _first_chapter(model, tmp_path)

        assert len(model.redrafts) == 1
        assert texts["ch01.s02"] == SCENES["ch02.s04"]

    async def test_a_scene_that_goes_on_is_not_drafted_again(self, tmp_path: Path) -> None:
        model = _Again("unused", scenes=[SCENES["ch02.s02"], SCENES["ch02.s03"]])

        texts = await _first_chapter(model, tmp_path)

        assert model.redrafts == []
        assert texts == {"ch01.s01": SCENES["ch02.s02"], "ch01.s02": SCENES["ch02.s03"]}


def test_a_short_scene_before_is_not_given_whole_as_its_end() -> None:
    """The cause: `ch02.s03` (891 characters) was under the 1,200 the Writer is given of the
    scene before, so it reached the next scene's prompt whole.

    Killed by: examples/story/src/uclone_x_story/context.py :: tail(previous_text, min(TAIL_CHARS, len(previous_text) // TAIL_PARTS))
    Becomes: tail(previous_text, min(TAIL_CHARS, len(previous_text)))
    """
    outline = Outline.model_validate(
        {
            "chapters": [
                {
                    "id": "ch02",
                    "title": "ch02",
                    "scenes": [{"id": "ch02.s03", "title": "a"}, {"id": "ch02.s04", "title": "b"}],
                }
            ]
        }
    )
    before = SCENES["ch02.s03"]

    end = scene_context(outline, "ch02.s04", CodexIndex(), previous_text=before, story={})[
        "end_of_previous_scene"
    ]

    assert len(end.removeprefix("...")) <= len(before) // TAIL_PARTS
    assert before.endswith(end.removeprefix("..."))
