"""#2311: a dead man only mentioned, remembered or spoken of is not read as alive again.

Three scenes qwen3:8b wrote in live runs after 이진호 had died made the store report him
alive (`disjoint`, alive vs dead) though he was not there: "…삶을 바쳐온 이진호
할아버지의 유산이었다" (his legacy), "…살아 있을 때 … 장면이 기억났다" (a memory), and
황의 telling 하린 of the past. A fact's own ``time`` asks "now or past?" of one quote, and
the model answered "now".

The scene reading now also answers, per character, how the scene has them at its own time
(``present``: acting, mentioned, remembered or reported), and a character alive or holding
an item counts only where the scene has them acting. Nothing reads the words of the scene:
the model's structured answer decides, and the code holds facts to it.

The three scenes are replayed from a recording of live qwen3:8b on a remote GPU host
(`tests/fixtures/story_mentioned_dead/`), call by call, each with its prompt's SHA-256: a
change to the reading prompt makes these tests fail and say to record again. Read with the
reading as it was, the same model reported him moving again in the legacy and the spoken-of
scenes; in the memory scene, cut to this story, it did not. That recording is kept with
the run record in the eval ledger.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.support.story_mentioned_dead import (
    DEAD,
    Replay,
    cases,
    read_and_check,
    recording,
    story,
)
from uclone_x_story.continuity import Finding
from uclone_x_story.enrich import ENRICH_MARKER, PRESENCE_MARKER, enrich_scene

DEATH_KINDS = frozenset({"dead_then_acting", "disjoint"})
CASES = ("legacy", "memory", "reported")
#: The lines the presence question asks about in each case, from the live readings:
#: 이진호 alive in `legacy`; none in `memory`; 하린 and 황의 holding the watch and
#: 이진호 alive in `reported`.
ASKED = {"legacy": 1, "memory": 0, "reported": 3}
_LINE = re.compile(r"^\d+\. .+ -- read as: ", re.MULTILINE)


def _lines_asked(model: Replay) -> int:
    return sum(len(_LINE.findall(p)) for p in model.presence_prompts)


def _death_findings(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.kind in DEATH_KINDS]


async def _replayed(tmp_path: Path, case: str) -> tuple[list[Finding], Any, Replay]:
    calls = recording()["cases"][case]
    model = Replay(calls)
    findings, read = await read_and_check(story(tmp_path, case), model, case)
    assert model.mismatched == [], (
        "the reading prompts are not the ones recorded: record "
        "tests/fixtures/story_mentioned_dead/readings.json again"
    )

    # Recorded before the presence question (#2317): that question goes unanswered here so
    # the reading's own ``present`` decides, as it did when they were recorded, and it is
    # asked about each fact that would contradict the store (#2342).
    assert model.unrecorded == ([PRESENCE_MARKER] if ASKED[case] else [])
    assert _lines_asked(model) == ASKED[case]
    assert model.replayed == len(calls) and model.calls == []
    return findings, read, model


class TestTheThreeFalseFindings:
    def test_the_recording_is_of_live_qwen3_and_quotes_the_scenes(self) -> None:
        recorded = recording()
        assert recorded["model"] == "qwen3:8b"
        assert sorted(recorded["cases"]) == sorted(CASES) == sorted(cases())
        for case, found in cases().items():
            assert found["quote"] in found["text"], case

    @pytest.mark.parametrize("case", CASES)
    async def test_the_dead_man_is_not_read_as_moving_again(
        self, tmp_path: Path, case: str
    ) -> None:
        """And the presence question is asked about each line that would contradict the
        store (`ASKED`, #2342), checked in `_replayed`.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: answers = await _ask_presence(model, reader, contested) if contested else None
        Becomes: answers = None
        """
        findings, _, _ = await _replayed(tmp_path, case)

        assert _death_findings(findings) == []

    @pytest.mark.parametrize("case", ["legacy", "reported"])
    async def test_what_the_reading_said_of_him_is_dropped_by_its_own_answer(
        self, tmp_path: Path, case: str
    ) -> None:
        """The recorded readings still give him alive (the legacy) or holding the watch
        (what 황의 says); their ``present`` has him only mentioned, so neither is checked.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: return present.get(read.subject) == ACTING
        Becomes: return True
        """
        findings, read, _ = await _replayed(tmp_path, case)

        assert read.dropped.get("not_acting_here") == 1
        assert not [f for f in read.facts if f.subject == DEAD and f.fact.key != "condition"]
        assert _death_findings(findings) == []

    async def test_a_reading_prompt_that_is_not_the_recorded_one_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A recording replayed against another prompt would pin replies nobody gets.

        Killed by: tests/support/story_mentioned_dead.py :: if (call["marker"], call["prompt_sha256"]) != (marker, digest):
        Becomes: if False:
        """
        work = story(tmp_path, "legacy")
        model = Replay(recording()["cases"]["legacy"])
        text = cases()["legacy"]["text"].replace("노인을 만났다", "노인을 보았다")
        work.write_scene(
            "ch02.s02", text, room_id="room_a", expected_digest=None, agent_id=None, turn_index=None
        )
        await enrich_scene(work, model=model, scene_id="ch02.s02", text=text, room_id="room_a")

        assert model.mismatched == [f"{ENRICH_MARKER}: not the prompt recorded"]


class Scripted:
    """A fake reading: one reply for the scene, as given."""

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        del system, temperature, max_tokens
        assert prompt.startswith(ENRICH_MARKER)
        return json.dumps(self.reply, ensure_ascii=False)


ACTING_LINE = "다음 날 아침, 이진호는 마을 한가운데로 걸어갔다."
WATCH_LINE = "이진호는 은시계를 품에 넣고 다녔다."


def _said(key: str, value: str, quote: str) -> dict[str, str]:
    return {"subject": DEAD, "named": "이진호", "key": key, "value": value, "quote": quote}


async def _scene_read(tmp_path: Path, reply: dict[str, Any]) -> tuple[list[Finding], Any]:
    """The legacy case's story, with its scene written as one where 이진호 walks and
    carries the watch, read by `reply`."""
    text = f"{ACTING_LINE} {WATCH_LINE} {cases()['legacy']['text']}"
    return await read_and_check(story(tmp_path, "legacy"), Scripted(reply), "legacy", text)


class TestRecallStands:
    """What the reading says of a man there and acting is still checked."""

    async def test_a_dead_man_the_reading_has_acting_is_caught(self, tmp_path: Path) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/enrich.py :: return present.get(read.subject) == ACTING
        Becomes: return False
        """
        findings, read = await _scene_read(
            tmp_path,
            {
                "present": {DEAD: "acting", "harin": "acting"},
                "facts": [_said("status", "alive", ACTING_LINE)],
            },
        )

        assert "not_acting_here" not in read.dropped
        [back] = _death_findings(findings)
        assert back.kind == "dead_then_acting" and back.quote == ACTING_LINE

    async def test_a_reading_with_no_present_map_is_checked_as_before(self, tmp_path: Path) -> None:
        """An unanswered question is no answer that he is absent.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: return True  # unanswered: no answer that they are absent
        Becomes: return False
        """
        findings, _ = await _scene_read(
            tmp_path, {"facts": [_said("status", "alive", ACTING_LINE)]}
        )

        assert [f.kind for f in _death_findings(findings)] == ["dead_then_acting"]

    async def test_one_the_map_leaves_out_is_not_acting(self, tmp_path: Path) -> None:
        findings, read = await _scene_read(
            tmp_path,
            {
                "present": {"harin": "acting"},
                "facts": [
                    _said("status", "alive", ACTING_LINE),
                    _said("possesses", "eunsigye", WATCH_LINE),
                ],
            },
        )

        assert read.dropped.get("not_acting_here") == 2
        assert findings == []


class TestADeathIsKeptHoweverItIsTold:
    async def test_a_reported_death_is_still_proposed(self, tmp_path: Path) -> None:
        """Only being alive or holding something needs him there: a death someone reports
        is a death, and the codex learns it.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: return key == "possesses" or (key == "status" and folded(read.fact.value) == "alive")
        Becomes: return True
        """
        work = story(tmp_path, "reported")
        found = cases()["reported"]
        hwangui_death = "황의는 그 자리에서 숨을 거두었다고 마을 사람들이 말했다."
        text = f"{found['text']} {hwangui_death}"
        work.write_scene(
            "ch02.s03", text, room_id="room_a", expected_digest=None, agent_id=None, turn_index=None
        )
        read = await enrich_scene(
            work,
            model=Scripted(
                {
                    "present": {"harin": "acting", "hwangui": "reported"},
                    "facts": [
                        {
                            "subject": "hwangui",
                            "named": "황의",
                            "key": "status",
                            "value": "dead",
                            "quote": hwangui_death,
                        }
                    ],
                }
            ),
            scene_id="ch02.s03",
            text=text,
            room_id="room_a",
        )

        [death] = read.proposed
        assert death.entry_id == "hwangui"
        assert death.change.progression is not None
        assert death.change.progression.set == {"status": "dead"}


class TestTheTwoLaterScenesAreNotReadAsHimAlive:
    """#2361: two scenes of the 2026-10-15 live run on which the store reported 이진호 alive
    again, on lines that only name him: "그녀는 할아버지의 유산을…" (his legacy) and
    "하린은 이진호의 집에서…" (his house). The scenes are hand-assembled from that run's
    report (`cases-2361.json`); the readings are live qwen3:8b, recorded on them
    (`readings-2361.json`), one case with 할아버지 given as his alias. The run's own
    readings were not kept, and these do not read him alive: the finding is pinned as not
    arising from these scenes, not as explained."""

    @pytest.mark.parametrize("case", ["legacy_line", "house", "legacy_line_alias"])
    async def test_no_death_finding_on_the_line_that_only_names_him(
        self, tmp_path: Path, case: str
    ) -> None:
        calls = recording("2361")["cases"][case]
        model = Replay(calls)
        work = story(tmp_path, case, cases_set="2361")

        findings, _ = await read_and_check(work, model, case, cases_set="2361")

        assert model.mismatched == [] and model.replayed == len(calls)
        assert _death_findings(findings) == []

    def test_the_recording_is_of_live_qwen3_and_quotes_the_scenes(self) -> None:
        recorded = recording("2361")
        assert recorded["model"] == "qwen3:8b"
        for case, found in cases("2361").items():
            assert found["quote"] in found["text"], case
