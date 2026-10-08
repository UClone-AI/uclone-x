"""The codex grows from each written scene, as proposals a person decides (§1.1 row 7).

`story_manuscript write` reads every saved scene once more for what it adds to the codex
(`uclone_x.story.enrich`). A fake model stands in for the reading; everything after the
reply is code, and that is what these pin, in order of what it would cost to get wrong:

* **Nothing is applied**: every addition is a pending proposal, and the codex files are
  as they were.
* **A quote the scene does not have proposes nothing**: a made-up fact looks like that.
* **What the codex already says, or contradicts, is not proposed**: the first is noise, the
  second is the continuity check's to report.
* **It runs on the write, without the model asking**, and a failed reading never fails the
  saved scene.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from uclone_x.story.enrich import ENRICH_MARKER
from uclone_x.story.library import StoryLibrary
from uclone_x.story.tool import StoryLibraryTool
from uclone_x.story.tools import StoryManuscriptTool, StoryOutlineTool
from uclone_x.story.work import StoryWork
from uclone_x.tools.models import REPLY_NOTE_KEY, NoIsolation, ToolContext

ROOM = "room_a"

SCENE = (
    "Mara lifted the brass lantern and led Doyun across the salt flats. "
    "A stranger named Ilsa waited at the Salt Crossing with a bow on her back. "
    "The arrow took Mara in the throat, and she did not rise again."
)


def _ctx(workspace: Path, story: str | None = None) -> ToolContext:
    return ToolContext(
        agent_id="writer",
        session_id=f"sess_room__{ROOM}__writer",
        workspace_root=workspace,
        room_id=ROOM,
        story_id=story,
        isolation=NoIsolation(),
    )


class FakeModel:
    """Answers every request with one reply, and keeps the prompts it was sent."""

    def __init__(self, reply: dict[str, Any] | str) -> None:
        self.reply = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        self.prompts: list[str] = []

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        del system, max_tokens
        assert temperature == 0.0
        self.prompts.append(prompt)
        return self.reply


class BrokenModel:
    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        raise RuntimeError("connector down")


def _yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")


async def _story(workspace: Path) -> tuple[str, ToolContext]:
    """A story with one chapter of two scenes and a codex of Mara, Doyun and the lantern."""
    out = await StoryLibraryTool().execute(
        {"action": "create", "title": "The Salt Road"}, _ctx(workspace)
    )
    assert out.success, out.error
    assert isinstance(out.output, dict)
    story_id = out.output["open_story_id"]
    assert isinstance(story_id, str)
    ctx = _ctx(workspace, story=story_id)
    outline = StoryOutlineTool()
    for args in (
        {"action": "init", "chapter_titles": ["The Crossing"]},
        {"action": "set_scene", "chapter_id": "ch01", "title": "Dusk"},
        {"action": "set_scene", "chapter_id": "ch01", "title": "Night"},
    ):
        done = await outline.execute(args, ctx)
        assert done.success, done.error
    codex = workspace / "stories" / story_id / "codex"
    _yaml(
        codex / "characters" / "mara.yaml",
        {
            "id": "mara",
            "name": "Mara",
            "state": {"status": "alive", "possesses": ["lantern"]},
        },
    )
    _yaml(
        codex / "characters" / "doyun.yaml",
        {
            "id": "doyun",
            "name": "Doyun",
            "state": {"status": "alive"},
            "relations": {"mara": "child"},
        },
    )
    _yaml(codex / "items" / "lantern.yaml", {"id": "lantern", "name": "brass lantern"})
    return story_id, ctx


def _codex_files(workspace: Path, story_id: str) -> dict[str, str]:
    root = workspace / "stories" / story_id / "codex"
    return {p.relative_to(root).as_posix(): p.read_text("utf-8") for p in root.rglob("*.yaml")}


async def _write(
    ctx: ToolContext, model: Any, text: str = SCENE, *, digest: str | None = None
) -> dict[str, Any]:
    tool = StoryManuscriptTool(model_factory=lambda _: model)
    args: dict[str, Any] = {"action": "write", "scene_id": "ch01.s02", "text": text}
    if digest is not None:
        args["digest"] = digest
    result = await tool.execute(args, ctx)
    assert result.success, result.error
    assert isinstance(result.output, dict)
    return result.output


def _pending(workspace: Path, story_id: str) -> list[dict[str, Any]]:
    work = StoryWork(StoryLibrary(workspace), story_id)
    return [
        p.model_dump(mode="json", exclude_defaults=True)
        for p, _ in work.proposals()[0]
        if p.status == "pending"
    ]


ILSA = {
    "kind": "characters",
    "name": "Ilsa",
    "profile": "A stranger with a bow.",
    "gender": "female",
    "quote": "A stranger named Ilsa waited at the Salt Crossing",
}
DEATH = {
    "subject": "mara",
    "key": "status",
    "value": "dead",
    "quote": "she did not rise again",
    "time": "now",
    "polarity": "asserted",
}


class TestAScenesAdditionsBecomeProposals:
    async def test_a_new_character_is_a_new_entry_proposal_and_the_codex_is_untouched(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/enrich.py :: result.proposed.append(work.add_proposal(draft, room_id=room_id))
        Becomes: pass
        """
        story_id, ctx = await _story(tmp_path)
        before = _codex_files(tmp_path, story_id)
        out = await _write(ctx, FakeModel({"new": [ILSA], "changes": []}))
        assert out["codex_growth"]["proposed"] == ["p001"]
        [proposal] = _pending(tmp_path, story_id)
        assert proposal["kind"] == "characters"
        assert proposal["change"]["new_entry"]["name"] == "Ilsa"
        assert proposal["change"]["new_entry"]["visual"] == {"gender": "female"}
        assert proposal["evidence"] == [{"scene_id": "ch01.s02", "quote": ILSA["quote"]}]
        assert _codex_files(tmp_path, story_id) == before

    async def test_a_death_with_its_quote_is_a_progression_at_the_scene(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/enrich.py :: return "status", "dead"
        Becomes: return None
        """
        story_id, ctx = await _story(tmp_path)
        await _write(ctx, FakeModel({"new": [], "changes": [DEATH]}))
        [proposal] = _pending(tmp_path, story_id)
        assert proposal["entry_id"] == "mara"
        assert proposal["change"]["progression"]["at"] == "ch01.s02"
        assert proposal["change"]["progression"]["set"] == {"status": "dead"}
        assert proposal["evidence"] == [{"scene_id": "ch01.s02", "quote": "she did not rise again"}]
        assert proposal["entry_digest"]

    async def test_the_person_is_told_in_plain_words_where_to_decide(self, tmp_path: Path) -> None:
        _, ctx = await _story(tmp_path)
        out = await _write(ctx, FakeModel({"new": [ILSA], "changes": [DEATH]}))
        note = out[REPLY_NOTE_KEY]
        assert "2건" in note["ko"] and "이야기 보기에서 승인" in note["ko"]
        assert note["ko"].endswith("니다.")
        for said in (note["ko"], note["en"]):
            assert "p00" not in said and "mara" not in said and "codex/" not in said


class TestWhatIsNotProposed:
    async def test_a_quote_the_scene_does_not_have_proposes_nothing(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/enrich.py :: if not quote_found(quote, self.text):
        Becomes: if False:
        """
        story_id, ctx = await _story(tmp_path)
        made_up = {**DEATH, "quote": "Mara was buried beneath the old oak"}
        ghost = {**ILSA, "name": "Borin", "quote": "Borin the smith sharpened his axe"}
        out = await _write(ctx, FakeModel({"new": [ghost], "changes": [made_up]}))
        assert _pending(tmp_path, story_id) == []
        assert out["codex_growth"]["dropped"] == {"quote_not_in_scene": 2}
        assert REPLY_NOTE_KEY not in out

    async def test_a_real_line_about_someone_else_proposes_no_change(self, tmp_path: Path) -> None:
        """qwen3:8b proposed a death from "a scream was heard", a line that names no one.

        Killed by: src/uclone_x/story/enrich.py :: if not self.names_subject(subject, quote):
        Becomes: if False:
        """
        story_id, ctx = await _story(tmp_path)
        elsewhere = {**DEATH, "quote": "with a bow on her back"}
        out = await _write(ctx, FakeModel({"new": [], "changes": [elsewhere]}))
        assert _pending(tmp_path, story_id) == []
        assert out["codex_growth"]["dropped"] == {"subject_not_named": 1}

    async def test_what_the_codex_has_or_a_proposal_waits_on_is_not_proposed_again(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/enrich.py :: if clash is not None:
        Becomes: if False:
        Killed by: src/uclone_x/story/enrich.py :: if (subject, pending_key, repr(new_value)) in self.pending_sets:
        Becomes: if False:
        """
        story_id, ctx = await _story(tmp_path)
        known = {**ILSA, "name": "Mara", "quote": "Mara lifted the brass lantern"}
        await _write(ctx, FakeModel({"new": [known, ILSA], "changes": [DEATH]}))
        assert [p["id"] for p in _pending(tmp_path, story_id)] == ["p001", "p002"]
        # The same scene written again: Ilsa and the death already wait for a person.
        written = StoryWork(StoryLibrary(tmp_path), story_id).manuscript("ch01.s02")
        assert written is not None
        again = FakeModel({"new": [ILSA], "changes": [DEATH]})
        out = await _write(ctx, again, digest=written.digest)
        assert out["codex_growth"]["proposed"] == []
        assert out["codex_growth"]["dropped"] == {"already_known": 1, "already_pending": 1}
        assert len(_pending(tmp_path, story_id)) == 2

    async def test_what_contradicts_the_codex_is_left_to_the_continuity_check(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/enrich.py :: if other in have:
        Becomes: if False:
        """
        story_id, ctx = await _story(tmp_path)
        text = f"{SCENE} Doyun wept over his wife Mara."
        husband = {
            "subject": "doyun",
            "key": "family",
            "of": "mara",
            "value": "husband",
            "quote": "Doyun wept over his wife Mara",
        }
        out = await _write(ctx, FakeModel({"new": [], "changes": [husband]}), text)
        assert _pending(tmp_path, story_id) == []
        assert out["codex_growth"]["dropped"] == {"contradicts_codex": 1}

    async def test_a_remembered_death_is_not_a_change_now(self, tmp_path: Path) -> None:
        story_id, ctx = await _story(tmp_path)
        remembered = {**DEATH, "time": "past"}
        out = await _write(ctx, FakeModel({"new": [], "changes": [remembered]}))
        assert _pending(tmp_path, story_id) == []
        assert out["codex_growth"]["dropped"] == {"not_in_the_present": 1}


class TestItRunsOnTheWrite:
    async def test_the_write_reads_the_saved_scene_without_the_model_asking(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/tools.py :: await self._grow(result, params, context)
        Becomes: pass
        """
        _, ctx = await _story(tmp_path)
        model = FakeModel({"new": [], "changes": []})
        out = await _write(ctx, model)
        [prompt] = model.prompts
        assert prompt.startswith(ENRICH_MARKER)
        assert SCENE in prompt and "- mara: Mara" in prompt
        assert out["codex_growth"] == {"scene_id": "ch01.s02", "proposed": []}

    async def test_with_no_model_the_scene_is_saved_and_not_said_to_be_read(
        self, tmp_path: Path
    ) -> None:
        _, ctx = await _story(tmp_path)
        out = await _write(ctx, None)
        assert out["scene_id"] == "ch01.s02"
        assert "codex_growth" not in out

    async def test_a_failed_reading_never_fails_the_saved_scene(self, tmp_path: Path) -> None:
        story_id, ctx = await _story(tmp_path)
        out = await _write(ctx, BrokenModel())
        assert "codex_growth" not in out
        assert any("could not be read for what it adds" in n for n in out["notes"])
        written = StoryWork(StoryLibrary(tmp_path), story_id).manuscript("ch01.s02")
        assert written is not None and written.text == SCENE

    async def test_a_reply_that_is_not_json_is_said_unread_not_empty(self, tmp_path: Path) -> None:
        _, ctx = await _story(tmp_path)
        out = await _write(ctx, FakeModel("I could not find anything."))
        assert out["codex_growth"]["unread"] is True


class TestAFamilyRelationIsProposedAsARelation:
    TEXT = f"{SCENE} Mara, Doyun's mother, held the lantern high."
    QUOTE = "Mara, Doyun's mother, held the lantern high"

    async def test_a_relation_between_two_known_characters_is_a_relations_progression(
        self, tmp_path: Path
    ) -> None:
        """The relation goes to the entry's `relations:`, never to a state key.

        Killed by: src/uclone_x/story/enrich.py :: pending_key, change = _relation_key(other), {"relations": {other: role}}
        Becomes: pending_key, change = _relation_key(other), {"set": {other: role}}
        """
        story_id, ctx = await _story(tmp_path)
        mother = {
            "subject": "mara",
            "key": "family",
            "of": "doyun",
            "value": "mother",
            "quote": self.QUOTE,
        }
        await _write(ctx, FakeModel({"new": [], "changes": [mother]}), self.TEXT)
        [proposal] = _pending(tmp_path, story_id)
        assert proposal["entry_id"] == "mara"
        assert proposal["change"]["progression"] == {
            "at": "ch01.s02",
            "relations": {"doyun": "parent"},
        }


class TestANewCharactersFamilyIsProposedOnBothSides:
    """A new character the scene says is family of one the codex has: the kinship is kept
    on both sides, as the cast's is -- in the new entry, and as a progression on the other."""

    TEXT = f"{SCENE} Ilsa, Doyun's daughter, had walked from the hills to find him."
    QUOTE = "Ilsa, Doyun's daughter, had walked from the hills"

    async def test_the_new_entry_and_its_relative_both_hold_the_kinship(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/enrich.py :: entry["relations"] = {other: role for other, role, _ in family}
        Becomes: pass
        Killed by: src/uclone_x/story/enrich.py :: inverse = INVERSE_ROLE[role]
        Becomes: inverse = role
        """
        story_id, ctx = await _story(tmp_path)
        before = _codex_files(tmp_path, story_id)
        daughter = {**ILSA, "family": {"doyun": "daughter"}, "quote": self.QUOTE}
        out = await _write(ctx, FakeModel({"new": [daughter], "changes": []}), self.TEXT)
        new, back = _pending(tmp_path, story_id)
        assert new["change"]["new_entry"]["state"] == {"status": "alive"}
        assert new["change"]["new_entry"]["relations"] == {"doyun": "child"}
        assert back["entry_id"] == "doyun" and back["entry_digest"]
        assert back["change"]["progression"] == {
            "at": "ch01.s02",
            "relations": {"ilsa": "parent"},
        }
        assert back["evidence"] == [{"scene_id": "ch01.s02", "quote": self.QUOTE}]
        assert out["codex_growth"]["proposed"] == [new["id"], back["id"]]
        assert _codex_files(tmp_path, story_id) == before  # nothing approved itself

    async def test_a_relation_the_quote_does_not_name_or_to_no_one_known_is_dropped(
        self, tmp_path: Path
    ) -> None:
        """The new character is still proposed, without the relation, and the drop counted.

        Killed by: src/uclone_x/story/enrich.py :: if word is None or not names_family(quote):
        Becomes: if word is None:
        Killed by: src/uclone_x/story/enrich.py :: if other is None or other == entry_id or self.kinds.get(other) != "characters":
        Becomes: if other is None or other == entry_id:
        """
        story_id, ctx = await _story(tmp_path)
        unnamed = {**ILSA, "family": {"mara": "sister", "lantern": "sister"}}
        out = await _write(ctx, FakeModel({"new": [unnamed], "changes": []}))
        [new] = _pending(tmp_path, story_id)
        assert new["change"]["new_entry"]["state"] == {"status": "alive"}
        assert "relations" not in new["change"]["new_entry"]
        assert out["codex_growth"]["dropped"] == {"not_readable": 1, "not_an_entry": 1}
