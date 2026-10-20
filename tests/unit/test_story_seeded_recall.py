"""#2014 recall, seeded: a dead man who acts, and an item a second holder has with no word
of a handover, are caught by the story's own check, through the real scene readings.

Live runs could not test this: asked for the dead man's return, qwen3:8b wrote it as a
memory or a ghost every time (manager ruling, #2014 issuecomment-6134794134). So the case
is seeded, as the 2026-10-11 acceptance run seeded it: `StoryStartTool` with scripted
cast, outline and prose, and two scenes:

* `ch01.s01`: the village elder holds the broken clock at the grave, and 이진호 dies there;
* `ch01.s02`: the next morning 이진호 walks the village, carrying the clock.

A second seeded story pins the handover on its own, where nobody dies, so only the
one-holder rule can catch it:

* `ch01.s01`: the elder gives the clock to 이진호, who puts it in his coat;
* `ch01.s02`: that night the elder takes the clock out and checks the time.

What makes it more than a scripted test is the reading. Every scene-reading call the
check makes (`ENRICH_MARKER`, the alias question `ALIAS_MARKER`, and the presence question
`PRESENCE_MARKER`) is answered from a recording of live qwen3:8b on a remote GPU host
(`tests/fixtures/story_seeded_recall/`, one file per story), call by call. Each recorded
call keeps its prompt's SHA-256, and the replay refuses a prompt that differs: a change to
the reading prompt, the cast or the scenes makes this test fail and say to record again,
rather than pass on replies to a prompt nobody sends any more.

Pinned: on `ch01.s02` of the first story the check reports the dead man moving
(`dead_then_acting`, or the store's `disjoint` alive-vs-dead it is named from) and the
clock in a second hand (`holder_changed`); on `ch01.s02` of the second, the old holder
using the clock he gave away (`holder_changed`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from tests.support.story_seeded_recall import (
    GRAVE,
    VILLAGE,
    Replay,
    recording,
    run_seeded,
    stale_message,
)
from uclone_x_story.enrich import ENRICH_MARKER, PRESENCE_MARKER

DEATH_KINDS = frozenset({"dead_then_acting", "disjoint"})
#: The lines the presence question asks about, per case, from the live readings: the dead
#: man alive and holding the clock; the elder holding it twice and 이진호 once.
ASKED = {"death": 2, "handover": 3}
_LINE = re.compile(r"^\d+\. .+ -- read as: ", re.MULTILINE)


def _lines_asked(model: Replay) -> int:
    return sum(len(_LINE.findall(p)) for p in model.presence_prompts)


async def _replayed(tmp_path: Path, case: str = "death") -> tuple[list[dict[str, Any]], Replay]:
    model = Replay(recording(case)["calls"], case)
    result = await run_seeded(model, tmp_path)
    assert result.success, result.error
    assert isinstance(result.output, dict)
    assert model.mismatched == [], stale_message(case)
    assert model.unrecorded == []
    assert _lines_asked(model) == ASKED[case]
    assert result.output["scenes_written"] == ["ch01.s01", "ch01.s02"]
    found = cast("list[dict[str, Any]]", result.output.get("continuity") or [])
    return found, model


class TestSeededRecall:
    async def test_the_recording_is_of_live_qwen3_and_every_call_is_replayed(
        self, tmp_path: Path
    ) -> None:
        recorded = recording()
        assert recorded["model"] == "qwen3:8b"
        _, model = await _replayed(tmp_path)
        assert model.replayed == len(recorded["calls"]) >= 3
        assert model.calls == []  # none left over

    async def test_a_reading_prompt_that_is_not_the_recorded_one_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A recording replayed against another prompt would pin replies nobody gets.

        Killed by: tests/support/story_seeded_recall.py :: if (call["marker"], call["prompt_sha256"]) != (marker, digest):
        Becomes: if False:
        """
        model = Replay(recording()["calls"])
        model._scenes = [GRAVE, VILLAGE.replace("마을 한가운데로", "강가로")]  # pyright: ignore[reportPrivateUsage]

        await run_seeded(model, tmp_path)

        assert model.mismatched == [
            f"{ENRICH_MARKER}: not the prompt recorded",
            f"{PRESENCE_MARKER}: not the prompt recorded",
        ]

    def test_a_stale_recording_names_its_own_file(self) -> None:
        """
        Killed by: tests/support/story_seeded_recall.py :: f"tests/fixtures/story_seeded_recall/{CASES[case][1]} again"
        Becomes: "tests/fixtures/story_seeded_recall/readings.json again"
        """
        assert stale_message("handover").endswith("story_seeded_recall/handover.json again")
        assert stale_message("death").endswith("story_seeded_recall/readings.json again")

    async def test_the_presence_question_is_asked_about_each_contested_fact(
        self, tmp_path: Path
    ) -> None:
        """Killed by: examples/story/src/uclone_x_story/enrich.py :: answers = await _ask_presence(model, reader, contested) if contested else None
        Becomes: answers = None
        """
        for case in ASKED:
            await _replayed(tmp_path / case, case)

    async def test_the_dead_man_who_walks_is_caught(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/continuity.py :: if key == "status":
        Becomes: if False:

        The alive-vs-dead rule is written twice: `continuity._CHECK_RULES` and
        `schemas.DEFAULT_AXIOMS`. Turning one copy off changes nothing. Turning both off
        (`object="Dead"` -> `object="Missing"` in each, one two-site mutation, which the
        declaration form above cannot state) kills this test: only the clock's finding is
        left. The runner is kept with the recording's run record in the eval ledger.
        """
        findings, _ = await _replayed(tmp_path)

        caught = [f for f in findings if f["scene_id"] == "ch01.s02" and f["kind"] in DEATH_KINDS]
        assert caught, findings
        assert all("이진호" in f["note"] for f in caught)

    async def test_the_clock_in_a_second_hand_is_caught(self, tmp_path: Path) -> None:
        """Killed by: examples/story/src/uclone_x_story/continuity.py :: held.append(SubmittedFact(subject, "possesses", item.entity_id, fact.quote))
        Becomes: pass

        The one-holder rule is written twice too, and turning both copies off
        (`subject="possesses"` -> `subject="owns"` in `continuity._CHECK_RULES` and
        `schemas.DEFAULT_AXIOMS`, one two-site mutation) kills this test: only the dead
        man's finding is left (the same runner).
        """
        findings, _ = await _replayed(tmp_path)

        caught = [
            f for f in findings if f["scene_id"] == "ch01.s02" and f["kind"] == "holder_changed"
        ]
        assert caught, findings
        assert all("조각난 시계" in f["note"] for f in caught)

    async def test_the_old_holder_using_the_clock_he_gave_away_is_caught(
        self, tmp_path: Path
    ) -> None:
        """The handover case (#2312 review): the old holder uses the item after a handover.

        Killed by: examples/story/src/uclone_x_story/continuity.py :: and here[2] == there[2]
        Becomes: and False

        The one-holder rule off in both its copies (the two-site mutation of the clock test
        above) kills this test too; the alive-vs-dead rule off in both does not, since
        nobody dies here.
        """
        findings, model = await _replayed(tmp_path, "handover")

        assert model.replayed == len(recording("handover")["calls"])
        caught = [
            f for f in findings if f["scene_id"] == "ch01.s02" and f["kind"] == "holder_changed"
        ]
        assert caught, findings
        assert all("조각난 시계" in f["note"] and "마을 장로" in f["note"] for f in caught)
