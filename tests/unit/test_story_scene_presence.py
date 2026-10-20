"""#2317 items 6-8: what the scene reading says of a person decides a contradiction only
when one question about that person, answered with the scene's own words, says so.

The reading's ``present`` (#2311) labels everyone the scene names at once, and at qwen3:8b
it is noisy both ways: a ghost who acts was labelled ``acting`` and reported alive (item 6),
and the walking dead man of the seeded recall case could as well have been labelled
``reported`` and dropped (item 8). And a character who only wants an item was read as
holding it (item 7). Each of those facts is one the store would report as a contradiction.

So a fact that would contradict the store as the scene starts -- alive where the store has
the character dead, holding an item someone else holds -- is asked about once more, on its
own (`PRESENCE_MARKER`): how the scene has that character at its own time (acting, a ghost
or other ``apparition``, mentioned, remembered, reported), whether they hold the item or
only want it, and the words that show it. The fact is checked only when the answer is
acting (and holding), with words that are in the scene and name the character. A fact
that contradicts nothing is held to ``present`` as before. No rule reads the words.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tests.support.story_mentioned_dead import DEAD, read_and_check, story
from uclone_x_story.continuity import Finding
from uclone_x_story.enrich import ENRICH_MARKER, PRESENCE_MARKER

WALKS = "다음 날 아침, 이진호는 마을 한가운데로 걸어갔다."
GHOST = "그날 밤 이진호의 유령이 하린 앞에 나타나 은시계를 내밀었다."
HOLDS = "이진호는 은시계를 품에 넣고 다녔다."
WANTS = "황의는 은시계를 갖고 싶어 했다."
HELD = "하린은 은시계를 손에 꼭 쥐었다."
SCENE = f"{WALKS} {GHOST} {HOLDS} {WANTS} {HELD} 하린은 아무 말도 하지 않았다."


class Scripted:
    """The scene reading, then the presence question, each answered as given; the
    questions asked are kept."""

    def __init__(self, reading: dict[str, Any], check: dict[str, Any] | str | None) -> None:
        self.reading = reading
        self.check = check
        self.asked: list[str] = []

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        del system, temperature, max_tokens
        if prompt.startswith(ENRICH_MARKER):
            return json.dumps(self.reading, ensure_ascii=False)
        assert prompt.startswith(PRESENCE_MARKER), prompt[:40]
        self.asked.append(prompt)
        if self.check is None:
            raise RuntimeError("the model is down")
        return self.check if isinstance(self.check, str) else json.dumps(self.check)


def _fact(subject: str, key: str, value: str, quote: str) -> dict[str, str]:
    return {"subject": subject, "key": key, "value": value, "quote": quote}


def _deaths(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.kind in ("dead_then_acting", "disjoint")]


async def _read(
    tmp_path: Path,
    reading: dict[str, Any],
    check: dict[str, Any] | str | None,
    *,
    holder: str | None = None,
) -> tuple[list[Finding], Any, Scripted]:
    model = Scripted(reading, check)
    work = story(tmp_path, "reported", holder=holder)
    findings, read = await read_and_check(work, model, "reported", SCENE)
    return findings, read, model


class TestADeadManIsAskedAboutOnce:
    async def test_one_labelled_reported_who_walks_is_caught(self, tmp_path: Path) -> None:
        """Item 8: ``present`` says reported; asked alone, the model says he acts, and
        quotes the scene. The contradiction is caught.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: ok = said if said is not None else _by_present(read, present)
        Becomes: ok = _by_present(read, present)
        """
        findings, read, model = await _read(
            tmp_path,
            {"present": {DEAD: "reported"}, "facts": [_fact(DEAD, "status", "alive", WALKS)]},
            {"1": {"answer": "acting", "quote": "이진호는 마을 한가운데로 걸어갔다"}},
        )

        assert len(model.asked) == 1 and "이진호" in model.asked[0]
        [back] = _deaths(findings)
        assert back.kind == "dead_then_acting" and back.quote == WALKS
        assert "not_acting_here" not in read.dropped

    async def test_a_ghost_labelled_acting_is_not_read_as_alive(self, tmp_path: Path) -> None:
        """Item 6: ``present`` says acting; asked alone, the model says it is his ghost.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: ok = said if said is not None else _by_present(read, present)
        Becomes: ok = _by_present(read, present)
        """
        findings, read, _ = await _read(
            tmp_path,
            {"present": {DEAD: "acting"}, "facts": [_fact(DEAD, "status", "alive", GHOST)]},
            {"1": {"answer": "apparition", "quote": "이진호의 유령이 하린 앞에 나타나"}},
        )

        assert _deaths(findings) == []
        assert read.dropped.get("not_acting_here") == 1

    async def test_an_answer_whose_words_are_not_in_the_scene_does_not_keep_the_fact(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/enrich.py :: shown = quote_found(quote, reader.text)
        Becomes: shown = True
        """
        findings, _, _ = await _read(
            tmp_path,
            {"present": {DEAD: "acting"}, "facts": [_fact(DEAD, "status", "alive", WALKS)]},
            {"1": {"answer": "acting", "quote": "이진호는 강가로 달려갔다"}},
        )

        assert _deaths(findings) == []

    async def test_an_answer_that_cannot_be_read_leaves_present_to_decide(
        self, tmp_path: Path
    ) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/enrich.py :: return None  # unanswered: present decides
        Becomes: return [False] * len(contested)

        And an answer that leaves a number out leaves that fact to ``present`` too:

        Killed by: examples/story/src/uclone_x_story/enrich.py :: out.append(None)
        Becomes: out.append(False)
        """
        for check in (None, "not json", "{}"):
            findings, _, model = await _read(
                tmp_path / str(check),
                {"present": {DEAD: "acting"}, "facts": [_fact(DEAD, "status", "alive", WALKS)]},
                check,
            )
            assert len(model.asked) == 1
            assert [f.kind for f in _deaths(findings)] == ["dead_then_acting"]

    async def test_an_answer_about_someone_else_does_not_keep_the_fact(
        self, tmp_path: Path
    ) -> None:
        """#2342 item 3: the answer says acting and quotes the scene, but the words are about
        하린, not the dead man; they do not show him acting.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: out.append(shown and reader.names_subject(read.subject, quote))
        Becomes: out.append(shown)
        """
        findings, read, _ = await _read(
            tmp_path,
            {"present": {DEAD: "acting"}, "facts": [_fact(DEAD, "status", "alive", WALKS)]},
            {"1": {"answer": "acting", "quote": "하린은 은시계를 손에 꼭 쥐었다"}},
        )

        assert _deaths(findings) == []
        assert read.dropped.get("not_acting_here") == 1

    async def test_a_word_it_was_not_offered_leaves_present_to_decide(self, tmp_path: Path) -> None:
        """#2342 item 2: "alive" is not one of the answers offered, so it says neither that he
        acts nor that he is not there; ``present`` (acting) decides, and he is caught.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: out.append(None)  # no word it was offered: present decides
        Becomes: out.append(False)
        """
        findings, read, _ = await _read(
            tmp_path,
            {"present": {DEAD: "acting"}, "facts": [_fact(DEAD, "status", "alive", WALKS)]},
            {"1": {"answer": "alive", "quote": "이진호는 마을 한가운데로 걸어갔다"}},
        )

        assert [f.kind for f in _deaths(findings)] == ["dead_then_acting"]
        assert "not_acting_here" not in read.dropped


class TestHoldingIsAskedAboutOnce:
    async def test_one_who_only_wants_the_watch_another_holds_is_no_second_holder(
        self, tmp_path: Path
    ) -> None:
        """Item 7: read as holding it; asked alone, the model says 황의 only wants it.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: ok = said if said is not None else _by_present(read, present)
        Becomes: ok = _by_present(read, present)
        """
        findings, read, model = await _read(
            tmp_path,
            {
                "present": {"hwangui": "acting", "harin": "acting"},
                "facts": [_fact("hwangui", "possesses", "eunsigye", WANTS)],
            },
            {"1": {"answer": "wants", "quote": "황의는 은시계를 갖고 싶어 했다"}},
            holder="harin",
        )

        assert len(model.asked) == 1 and "황의" in model.asked[0]
        assert [f for f in findings if f.kind in ("holder_changed", "inverse_functional")] == []
        assert read.dropped.get("not_acting_here") == 1

    async def test_two_holders_read_in_one_scene_are_each_asked_about(self, tmp_path: Path) -> None:
        """The #2311 case itself: 하린 holds the watch and 황의 wants it, both in one scene,
        and the reading gave both as holding. Asked alone, only 하린 holds it.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: s != read.subject for s in claims.get(_held_item(store, read) or "", ())
        Becomes: False for s in claims.get(_held_item(store, read) or "", ())
        """
        findings, read, model = await _read(
            tmp_path,
            {
                "present": {"hwangui": "acting", "harin": "acting"},
                "facts": [
                    _fact("harin", "possesses", "eunsigye", HELD),
                    _fact("hwangui", "possesses", "eunsigye", WANTS),
                ],
            },
            {
                "1": {"answer": "holds", "quote": "하린은 은시계를 손에 꼭 쥐었다"},
                "2": {"answer": "wants", "quote": "황의는 은시계를 갖고 싶어 했다"},
            },
        )

        assert len(model.asked) == 1
        assert [f for f in findings if f.kind in ("holder_changed", "inverse_functional")] == []
        assert [(r.subject, r.fact.key) for r in read.facts] == [("harin", "possesses")]

    async def test_one_who_holds_it_is_a_second_holder(self, tmp_path: Path) -> None:
        """
        Killed by: examples/story/src/uclone_x_story/enrich.py :: expected = _HOLDS if _key(read.fact.key) == "possesses" else ACTING
        Becomes: expected = ACTING
        """
        findings, _, _ = await _read(
            tmp_path,
            {
                "present": {DEAD: "acting"},
                "facts": [_fact("harin", "possesses", "eunsigye", HELD)],
            },
            {"1": {"answer": "holds", "quote": "하린은 은시계를 손에 꼭 쥐었다"}},
            holder="hwangui",
        )

        assert [f.kind for f in findings] == ["holder_changed"]

    async def test_acting_on_a_holding_line_leaves_present_to_decide(self, tmp_path: Path) -> None:
        """#2342 item 2: a holding line is answered "holds" or "wants", or a not-there word;
        "acting" is none of them, so ``present`` (acting) decides and the second holder is
        caught.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: elif answer in not_there:
        Becomes: elif True:
        """
        findings, _, _ = await _read(
            tmp_path,
            {
                "present": {"harin": "acting"},
                "facts": [_fact("harin", "possesses", "eunsigye", HELD)],
            },
            {"1": {"answer": "acting", "quote": "하린은 은시계를 손에 꼭 쥐었다"}},
            holder="hwangui",
        )

        assert [f.kind for f in findings] == ["holder_changed"]


class TestNoQuestionWithoutAContradiction:
    async def test_a_fact_that_contradicts_nothing_is_not_asked_about(self, tmp_path: Path) -> None:
        """Alive where the store has the character alive: ``present`` decides, as before.

        Killed by: examples/story/src/uclone_x_story/enrich.py :: return "dead" in before
        Becomes: return True
        """
        findings, read, model = await _read(
            tmp_path,
            {
                "present": {"harin": "acting"},
                "facts": [_fact("harin", "status", "alive", "하린은 아무 말도 하지 않았다")],
            },
            None,
        )

        assert model.asked == [] and findings == []
        assert "not_acting_here" not in read.dropped

    def test_the_question_lists_the_answers_it_reads(self) -> None:
        from uclone_x_story.enrich import presence_prompt

        prompt = presence_prompt(
            [("이진호", "alive"), ("황의", "possesses 은시계")], ["a", "b"], "scene text"
        )
        assert prompt.startswith(PRESENCE_MARKER)
        for word in ("acting", "apparition", "mentioned", "remembered", "reported", "holds"):
            assert re.search(rf'"{word}"', prompt), word
