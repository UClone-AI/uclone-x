"""The Writer's story tools: outline, codex, manuscript, context, and the sheet over a story (#1556).

What these pin, in order of what it would cost to get wrong:

* **A new conversation continues from the recap alone.** What the story is, what the last
  conversation wrote and left as a summary, and the next scene's context.
* **The story's own record is not a file a tool writes**, however it is spelled or linked.
* **No lost text.** A rewrite keeps the text it replaced; a rewrite of text changed since
  it was read is refused.
* **A file that does not fit names the file and the field**, in words.
* **Two stories' codex ids do not collide.**
* **What a scene's context leaves out is said**, with the reason.
* **`character_sheet` reads the open story**, refuses to save into it, and shows the
  workspace's own sheets read-only.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse
from uclone_x.story.character import StoryCharacterSheetTool as CharacterSheetTool
from uclone_x.story.context import MAX_ENTRIES, CodexIndex, CodexItem, last_and_next
from uclone_x.story.library import StoryError, StoryLibrary
from uclone_x.story.schemas import CharacterEntry, Outline
from uclone_x.story.tool import StoryLibraryTool
from uclone_x.story.tools import (
    StoryCodexTool,
    StoryContextTool,
    StoryManuscriptTool,
    StoryOutlineTool,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import NoIsolation, ToolContext, ToolResult
from uclone_x.tools.registry import create_default_registry

ROOM_A = "room_a"
ROOM_B = "room_b"


def _ctx(
    workspace: Path, *, conversation: str | None = ROOM_A, story: str | None = None
) -> ToolContext:
    return ToolContext(
        agent_id="writer",
        session_id=f"sess_room__{conversation}__writer",
        workspace_root=workspace,
        room_id=conversation,
        story_id=story,
        isolation=NoIsolation(),
    )


async def _call(tool: BaseTool[Any], ctx: ToolContext, **args: Any) -> ToolResult:
    return await tool.execute(args, ctx)


async def _ok(tool: BaseTool[Any], ctx: ToolContext, **args: Any) -> dict[str, Any]:
    result = await _call(tool, ctx, **args)
    assert result.success, result.error
    assert isinstance(result.output, dict)
    return result.output


async def _new_story(workspace: Path, title: str = "The Salt Road", **extra: Any) -> str:
    out = await _ok(StoryLibraryTool(), _ctx(workspace), action="create", title=title, **extra)
    story_id = out["open_story_id"]
    assert isinstance(story_id, str)
    return story_id


def _put(workspace: Path, story_id: str, relative: str, text: str) -> Path:
    """A file written by hand into the story, as a person editing it would."""
    path = workspace / "stories" / story_id / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _codex(workspace: Path, story_id: str, kind: str, entry: dict[str, Any]) -> None:
    _put(
        workspace,
        story_id,
        f"codex/{kind}/{entry['id']}.yaml",
        yaml.safe_dump(entry, allow_unicode=True),
    )


async def _outlined(workspace: Path, story_id: str) -> ToolContext:
    """A story with chapter ch01 holding ch01.s01 and ch01.s02; returns the writer's context."""
    ctx = _ctx(workspace, story=story_id)
    outline = StoryOutlineTool()
    await _ok(outline, ctx, action="init", chapter_titles=["The Crossing"])
    await _ok(
        outline,
        ctx,
        action="set_scene",
        chapter_id="ch01",
        title="Ferry at dusk",
        summary="Mara boards the ferry.",
        characters=["mara"],
    )
    await _ok(
        outline,
        ctx,
        action="set_scene",
        chapter_id="ch01",
        title="The toll",
        summary="Vane demands the toll.",
        characters=["mara", "vane"],
    )
    return ctx


# --------------------------------------------------------------------------------------
# The acceptance: a new conversation continues from the recap alone
# --------------------------------------------------------------------------------------


class TestANewConversationContinuesFromTheRecap:
    async def test_the_recap_carries_the_story_the_summary_and_the_next_scene(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/work.py :: "summary": summary if summary is not None else entry.summary,
        Becomes: "summary": entry.summary,

        Killed by: src/uclone_x/story/context.py :: recent = list(sessions.sessions[-RECENT_SESSIONS:])
        Becomes: recent = list(reversed(sessions.sessions[-RECENT_SESSIONS:]))
        """
        story_id = await _new_story(
            tmp_path, genre="fantasy", style_notes="Past tense, close third."
        )
        ctx = await _outlined(tmp_path, story_id)
        _codex(
            tmp_path,
            story_id,
            "characters",
            {"id": "mara", "name": "Mara", "profile": "A smuggler."},
        )
        _codex(
            tmp_path,
            story_id,
            "characters",
            {"id": "vane", "name": "Lord Vane", "profile": "Holds the ferry."},
        )
        first = " ".join(f"Mara counted plank {n} of the ferry." for n in range(80))
        first += " The river went black behind her."
        await _ok(
            StoryManuscriptTool(),
            ctx,
            action="write",
            scene_id="ch01.s01",
            text=first,
            session_summary="Mara is aboard; the toll is next.",
        )
        await _ok(StoryLibraryTool(), ctx, action="close")

        # A different conversation, which remembers none of that.
        opened = await _ok(
            StoryLibraryTool(),
            _ctx(tmp_path, conversation=ROOM_B),
            action="open",
            story_id=story_id,
        )
        assert opened["writable"] is True
        recap = await _ok(
            StoryContextTool(), _ctx(tmp_path, conversation=ROOM_B, story=story_id), action="recap"
        )

        assert recap["story"]["title"] == "The Salt Road"
        assert recap["story"]["style_notes"] == "Past tense, close third."
        assert recap["recent_sessions"][0]["room_id"] == ROOM_A
        assert recap["recent_sessions"][0]["summary"] == "Mara is aboard; the toll is next."
        assert recap["recent_sessions"][0]["scenes_written"] == ["ch01.s01"]
        assert recap["recent_sessions"][0]["closed_at"]
        assert recap["progress"] == {"scenes_in_outline": 2, "scenes_written": 1}
        assert recap["last_written_scene"]["scene_id"] == "ch01.s01"
        assert recap["last_written_scene"]["end_of_text"].endswith(
            "The river went black behind her."
        )
        following = recap["next_scene"]
        assert following["scene"]["id"] == "ch01.s02"
        assert following["end_of_previous_scene"].endswith("The river went black behind her.")
        assert {e["id"] for e in following["codex"]} == {"mara", "vane"}
        assert "ch01.s02" in recap["how_to_continue"]

        # Room B writes the next scene from the recap alone: the scene to write, what it is
        # about and who is in it all come from the recap, not from room A's conversation.
        ctx_b = _ctx(tmp_path, conversation=ROOM_B, story=story_id)
        scene = following["scene"]
        cast = " and ".join(e["name"] for e in following["codex"])
        second = f"{cast} met at the rail. {scene['summary']} The ferry did not stop."
        written = await _ok(
            StoryManuscriptTool(),
            ctx_b,
            action="write",
            scene_id=scene["id"],
            text=second,
            session_summary="The toll is paid; the crossing is done.",
        )
        assert written["scene_id"] == "ch01.s02"
        await _ok(StoryLibraryTool(), ctx_b, action="close")

        # A third conversation sees both sessions, oldest first and then its own, and the
        # whole outline written.
        await _ok(
            StoryLibraryTool(),
            _ctx(tmp_path, conversation="room_c"),
            action="open",
            story_id=story_id,
        )
        after = await _ok(
            StoryContextTool(),
            _ctx(tmp_path, conversation="room_c", story=story_id),
            action="recap",
        )
        assert after["progress"] == {"scenes_in_outline": 2, "scenes_written": 2}
        assert after["last_written_scene"]["scene_id"] == "ch01.s02"
        assert after["last_written_scene"]["end_of_text"].endswith("The ferry did not stop.")
        assert [s["room_id"] for s in after["recent_sessions"]] == [ROOM_A, ROOM_B, "room_c"]
        by_room = {s["room_id"]: s for s in after["recent_sessions"]}
        assert by_room[ROOM_B]["scenes_written"] == ["ch01.s02"]
        assert by_room[ROOM_B]["summary"] == "The toll is paid; the crossing is done."
        assert by_room[ROOM_B]["closed_at"]
        assert after["next_scene"] is None

    async def test_a_story_with_no_outline_says_where_to_start(self, tmp_path: Path) -> None:
        story_id = await _new_story(tmp_path)
        recap = await _ok(StoryContextTool(), _ctx(tmp_path, story=story_id), action="recap")
        assert "story_outline 'init'" in recap["how_to_continue"]
        assert "next_scene" not in recap

    def test_the_next_scene_is_after_the_last_written_one_not_the_first_gap(self) -> None:
        """Killed by: src/uclone_x/story/context.py :: start = 0 if last_index is None else last_index + 1
        Becomes: start = 0
        """
        outline = Outline.model_validate(
            {
                "chapters": [
                    {
                        "id": "ch01",
                        "title": "One",
                        "scenes": [
                            {"id": "a", "title": "A"},
                            {"id": "b", "title": "B"},
                            {"id": "c", "title": "C"},
                        ],
                    }
                ]
            }
        )
        last, following = last_and_next(outline, ["b"])
        assert last is not None and last[1].id == "b"
        assert following is not None and following[1].id == "c"


# --------------------------------------------------------------------------------------
# The story's own record, however it is named (#1564 review, item 1)
# --------------------------------------------------------------------------------------


class TestTheStoryRecordIsNotWrittenByAPath:
    async def test_a_link_inside_the_story_to_its_record_is_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/library.py :: if target == record or (target.exists() and target.samefile(record)):
        Becomes: if pure.name == STORY_FILE:
        """
        story_id = await _new_story(tmp_path)
        folder = tmp_path / "stories" / story_id
        (folder / "notes.yaml").symlink_to(folder / "story.yaml")
        before = (folder / "story.yaml").read_text(encoding="utf-8")

        with pytest.raises(StoryError, match="the story's own record"):
            StoryLibrary(tmp_path).write_file(
                story_id,
                "notes.yaml",
                "lease: null\n",
                conversation_id=ROOM_A,
                expected_digest=None,
            )
        assert (folder / "story.yaml").read_text(encoding="utf-8") == before

    async def test_the_record_spelled_in_another_case_is_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/library.py :: (target.exists() and target.samefile(record))
        Becomes: False
        """
        story_id = await _new_story(tmp_path)
        folder = tmp_path / "stories" / story_id
        if not (folder / "Story.yaml").exists():
            pytest.skip("this disk tells 'Story.yaml' from 'story.yaml', so they are two files")
        before = (folder / "story.yaml").read_text(encoding="utf-8")
        with pytest.raises(StoryError, match="the story's own record"):
            StoryLibrary(tmp_path).write_file(
                story_id,
                "Story.yaml",
                "lease: null\n",
                conversation_id=ROOM_A,
                expected_digest=None,
            )
        assert (folder / "story.yaml").read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------------------
# Moving between stories keeps the old lease until the new one opens (item 3)
# --------------------------------------------------------------------------------------


class TestAFailedMoveKeepsTheStoryItHad:
    async def test_opening_an_unreadable_story_keeps_the_lease_on_the_open_one(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/tool.py :: opening = library.open(story_id, conversation)
        Becomes: self._leave(library, context.story_id, conversation, keep=story_id); opening = library.open(story_id, conversation)
        """
        kept = await _new_story(tmp_path, "Kept")
        broken = await _new_story(tmp_path, "Broken")
        _put(tmp_path, broken, "story.yaml", "title: [unclosed\n")

        result = await _call(
            StoryLibraryTool(), _ctx(tmp_path, story=kept), action="open", story_id=broken
        )

        assert not result.success
        record = StoryLibrary(tmp_path).load(kept)
        assert record.lease is not None and record.lease.holder == ROOM_A

    async def test_taking_over_an_unreadable_story_keeps_the_lease_on_the_open_one(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/tool.py :: record, previous = library.take_over(story_id, conversation)
        Becomes: self._leave(library, context.story_id, conversation, keep=story_id); record, previous = library.take_over(story_id, conversation)
        """
        kept = await _new_story(tmp_path, "Kept")
        broken = await _new_story(tmp_path, "Broken")
        _put(tmp_path, broken, "story.yaml", "- not a record\n")

        result = await _call(
            StoryLibraryTool(), _ctx(tmp_path, story=kept), action="take_over", story_id=broken
        )

        assert not result.success
        record = StoryLibrary(tmp_path).load(kept)
        assert record.lease is not None and record.lease.holder == ROOM_A


# --------------------------------------------------------------------------------------
# Text that is not UTF-8 (item 4)
# --------------------------------------------------------------------------------------


class TestTextThatIsNotUtf8:
    async def test_a_scene_saved_in_another_encoding_is_refused_in_words(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/library.py :: text = data.decode("utf-8")
        Becomes: text = data.decode("utf-8", "replace")
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        path = tmp_path / "stories" / story_id / "manuscript" / "ch01.s01.md"
        path.parent.mkdir(parents=True)
        path.write_bytes("마라는 배에 올랐다.".encode("euc-kr"))

        result = await _call(StoryManuscriptTool(), ctx, action="read", scene_id="ch01.s01")

        assert not result.success
        assert result.error == (
            "'manuscript/ch01.s01.md' is not saved as UTF-8 text, so it was not read. Save it "
            "as UTF-8 text and try again."
        )


# --------------------------------------------------------------------------------------
# A file that does not fit names the file and the field
# --------------------------------------------------------------------------------------


class TestAFileThatDoesNotFit:
    async def test_a_codex_entry_names_its_file_and_field(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/schemas.py :: field = _where(loc)
        Becomes: field = ""
        """
        story_id = await _new_story(tmp_path)
        _codex(
            tmp_path,
            story_id,
            "characters",
            {"id": "vane", "name": "Vane", "visual": {"base_seed": "seven"}},
        )

        out = await _ok(StoryCodexTool(), _ctx(tmp_path, story=story_id), action="search")

        assert out["entries"] == []
        [problem] = out["unreadable_files"]
        assert problem["file"] == "codex/characters/vane.yaml"
        assert problem["reason"] == (
            "codex/characters/vane.yaml does not fit its shape: 'visual.base_seed' should be "
            "a whole number."
        )

    async def test_an_outline_names_the_scene_field(self, tmp_path: Path) -> None:
        story_id = await _new_story(tmp_path)
        _put(
            tmp_path,
            story_id,
            "outline.yaml",
            "chapters:\n- id: ch01\n  title: One\n  scenes:\n  - id: Bad Id\n    title: A\n",
        )

        result = await _call(StoryOutlineTool(), _ctx(tmp_path, story=story_id), action="get")

        assert not result.success
        assert result.error is not None
        assert result.error.startswith("outline.yaml does not fit its shape:")
        assert "'chapters[0].scenes[0].id' is not a valid id" in result.error
        assert "pattern" not in result.error and "Error" not in result.error

    async def test_a_file_named_other_than_its_id_is_reported(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/work.py :: unreadable.append(UnreadableFile(relative, str(exc)))
        Becomes: raise
        """
        story_id = await _new_story(tmp_path)
        _put(tmp_path, story_id, "codex/places/ferry.yaml", "id: dock\nname: The Dock\n")

        out = await _ok(StoryCodexTool(), _ctx(tmp_path, story=story_id), action="search")

        [problem] = out["unreadable_files"]
        assert "says its id is 'dock', but the file is named 'ferry'" in problem["reason"]

    async def test_files_and_folders_the_codex_does_not_read_are_reported(
        self, tmp_path: Path
    ) -> None:
        """A person who saved `mara.yml` is told why Mara is missing (#1576).

        Killed by: src/uclone_x/story/work.py :: if name.suffix != ".yaml":
        Becomes: if False:

        Killed by: src/uclone_x/story/work.py :: for relative in folders:
        Becomes: for relative in []:

        Killed by: src/uclone_x/story/work.py :: if PurePosixPath(relative).name in CODEX_KINDS:
        Becomes: if True:

        Killed by: src/uclone_x/story/work.py :: for relative in self._library.files_in(self._story_id, CODEX_DIR):
        Becomes: for relative in []:
        """
        story_id = await _new_story(tmp_path)
        _codex(tmp_path, story_id, "places", {"id": "dock", "name": "The Dock"})
        _put(tmp_path, story_id, "codex/characters/mara.yml", "id: mara\nname: Mara\n")
        _put(tmp_path, story_id, "codex/characters/old/vane.yaml", "id: vane\nname: Vane\n")
        _put(tmp_path, story_id, "codex/character/pell.yaml", "id: pell\nname: Pell\n")
        _put(tmp_path, story_id, "codex/notes.md", "loose notes\n")
        _put(tmp_path, story_id, "codex/characters/.DS_Store", "")

        out = await _ok(StoryCodexTool(), _ctx(tmp_path, story=story_id), action="search")

        assert [e["id"] for e in out["entries"]] == ["dock"]
        kinds = "codex/characters/, codex/places/, codex/items/, codex/threads/"
        assert {u["file"]: u["reason"] for u in out["unreadable_files"]} == {
            "codex/notes.md": "'codex/notes.md' was not read: codex entries are kept in one "
            f"of the folders {kinds}. Move it into the right one.",
            "codex/character": "The folder 'codex/character/' was not read: the codex reads "
            f"only the folders {kinds}. Move its entries into the right one.",
            "codex/characters/old": "The folder 'codex/characters/old/' was not read: codex "
            "entries are files directly in 'codex/characters/'. Move its entries up into "
            "'codex/characters/'.",
            "codex/characters/mara.yml": "'codex/characters/mara.yml' was not read: codex "
            "entries are read only from files ending in '.yaml'. If it is an entry, rename "
            "it to 'mara.yaml'.",
        }
        # A lookup of the missing entry points at the report instead of a bare "not found".
        missing = await _call(
            StoryCodexTool(), _ctx(tmp_path, story=story_id), action="get", entry_id="mara"
        )
        assert missing.error is not None
        assert "4 codex file(s) or folder(s) could not be read" in missing.error

    async def test_a_rename_is_suggested_only_when_the_name_is_plain_and_free(
        self, tmp_path: Path
    ) -> None:
        """An editor's backup is not told to become `mara.yaml.yaml`, nor to replace the
        entry it is a copy of (#1595).

        Killed by: src/uclone_x/story/work.py :: if match is not None:
        Becomes: if False:

        Killed by: src/uclone_x/story/library.py :: and entry.name.lower() not in _SYSTEM_FILES
        Becomes: and True
        """
        story_id = await _new_story(tmp_path)
        _codex(tmp_path, story_id, "characters", {"id": "mara", "name": "Mara"})
        _put(tmp_path, story_id, "codex/characters/mara.yaml~", "id: mara\nname: Mara\n")
        _put(tmp_path, story_id, "codex/places/dock.yaml.bak", "id: dock\nname: Dock\n")
        _put(tmp_path, story_id, "codex/items/Old Map.txt", "a map\n")
        _put(tmp_path, story_id, "codex/characters/Thumbs.db", "")
        _put(tmp_path, story_id, "codex/places/desktop.ini", "")

        out = await _ok(StoryCodexTool(), _ctx(tmp_path, story=story_id), action="search")

        assert [e["id"] for e in out["entries"]] == ["mara"]
        ending = "codex entries are read only from files ending in '.yaml'."
        fallback = (
            "If it is an entry, give it an unused name ending in '.yaml' using only lowercase "
            "letters, digits, and hyphens (up to 80 characters)."
        )
        assert {u["file"]: u["reason"] for u in out["unreadable_files"]} == {
            "codex/characters/mara.yaml~": f"'codex/characters/mara.yaml~' was not read: "
            f"{ending} {fallback}",
            "codex/places/dock.yaml.bak": f"'codex/places/dock.yaml.bak' was not read: "
            f"{ending} If it is an entry, rename it to 'dock.yaml'.",
            "codex/items/Old Map.txt": f"'codex/items/Old Map.txt' was not read: {ending} "
            f"{fallback}",
        }

    async def test_a_rename_never_targets_a_name_taken_in_other_capitals_or_by_a_twin(
        self, tmp_path: Path
    ) -> None:
        """On a Mac, `mv mara.yml mara.yaml` beside `Mara.yaml` replaces it, and two files
        both told to become `bo.yaml` would have the second replace the first (#1595).

        Killed by: src/uclone_x/story/work.py :: taken = any(name.casefold() == wanted for name in others)
        Becomes: taken = any(name == target for name in others)

        Killed by: src/uclone_x/story/work.py :: if not taken and not shared:
        Becomes: if not taken:

        Killed by: src/uclone_x/story/work.py :: if not taken and not shared:
        Becomes: if not shared:

        Killed by: src/uclone_x/story/work.py :: others = [PurePosixPath(b).name for b in beside if b != relative]
        Becomes: others = [PurePosixPath(b).name for b in beside]
        """
        story_id = await _new_story(tmp_path)
        _put(tmp_path, story_id, "codex/characters/Mara.yaml", "id: mara\nname: Mara\n")
        _put(tmp_path, story_id, "codex/characters/mara.yml", "id: mara\nname: Mara\n")
        _put(tmp_path, story_id, "codex/places/bo.yml", "id: bo\nname: Bo\n")
        _put(tmp_path, story_id, "codex/places/bo.yaml~", "id: bo\nname: Bo\n")
        _put(tmp_path, story_id, "codex/items/map.YAML", "id: map\nname: Map\n")

        out = await _ok(StoryCodexTool(), _ctx(tmp_path, story=story_id), action="search")

        reasons = {u["file"]: u["reason"] for u in out["unreadable_files"]}
        ending = "codex entries are read only from files ending in '.yaml'."
        no_name = (
            "If it is an entry, give it an unused name ending in '.yaml' using only lowercase "
            "letters, digits, and hyphens (up to 80 characters)."
        )
        assert reasons["codex/characters/mara.yml"] == (
            f"'codex/characters/mara.yml' was not read: {ending} {no_name}"
        )
        for twin in ("codex/places/bo.yml", "codex/places/bo.yaml~"):
            assert reasons[twin] == f"'{twin}' was not read: {ending} {no_name}"
        # A file differing only in capitals is not in its own way.
        assert reasons["codex/items/map.YAML"] == (
            f"'codex/items/map.YAML' was not read: {ending} If it is an entry, rename it to "
            "'map.yaml'."
        )

    async def test_icon_file_is_excluded_from_story_library_files_in(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/library.py :: _SYSTEM_FILES = frozenset({"thumbs.db", "ehthumbs.db", "desktop.ini", "icon\r"})
        Becomes: _SYSTEM_FILES = frozenset({"thumbs.db", "ehthumbs.db", "desktop.ini"})
        """
        story_id = await _new_story(tmp_path)
        library = StoryLibrary(tmp_path)
        _put(tmp_path, story_id, "codex/characters/Icon\r", "")
        _put(tmp_path, story_id, "codex/characters/Thumbs.db", "")
        _put(tmp_path, story_id, "codex/characters/mara.yaml", "id: mara\nname: Mara\n")

        files = library.files_in(story_id, "codex/characters")
        assert files == ["codex/characters/mara.yaml"]

    async def test_a_backup_with_multiple_extensions_suggests_yaml_not_nested(
        self, tmp_path: Path
    ) -> None:
        story_id = await _new_story(tmp_path)
        _put(tmp_path, story_id, "codex/characters/mara.yaml.bak.1", "id: mara\nname: Mara\n")

        out = await _ok(StoryCodexTool(), _ctx(tmp_path, story=story_id), action="search")

        [problem] = out["unreadable_files"]
        assert problem["file"] == "codex/characters/mara.yaml.bak.1"
        assert problem["reason"] == (
            "'codex/characters/mara.yaml.bak.1' was not read: "
            "codex entries are read only from files ending in '.yaml'. "
            "If it is an entry, rename it to 'mara.yaml'."
        )

    async def test_a_name_over_eighty_characters_does_not_suggest_a_rename(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/work.py :: if len(base_str) > 80:
        Becomes: if False:
        """
        story_id = await _new_story(tmp_path)
        long_id = "a" * 81
        _put(tmp_path, story_id, f"codex/characters/{long_id}.txt", "id: long\n")

        out = await _ok(StoryCodexTool(), _ctx(tmp_path, story=story_id), action="search")

        [problem] = out["unreadable_files"]
        assert problem["file"] == f"codex/characters/{long_id}.txt"
        assert problem["reason"] == (
            f"'codex/characters/{long_id}.txt' was not read: "
            "codex entries are read only from files ending in '.yaml'. "
            "If it is an entry, give it an unused name ending in '.yaml' using only lowercase "
            "letters, digits, and hyphens (up to 80 characters)."
        )

    async def test_an_existing_folder_named_target_prevents_suggesting_rename(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/work.py :: _rename_advice(relative, files + folders)
        Becomes: _rename_advice(relative, files)
        """
        story_id = await _new_story(tmp_path)
        folder = tmp_path / "stories" / story_id / "codex/characters/mara.yaml"
        folder.mkdir(parents=True, exist_ok=True)
        _put(tmp_path, story_id, "codex/characters/mara.yml", "id: mara\nname: Mara\n")

        out = await _ok(StoryCodexTool(), _ctx(tmp_path, story=story_id), action="search")

        reasons = {u["file"]: u["reason"] for u in out["unreadable_files"]}
        ending = "codex entries are read only from files ending in '.yaml'."
        fallback = (
            "If it is an entry, give it an unused name ending in '.yaml' using only lowercase "
            "letters, digits, and hyphens (up to 80 characters)."
        )
        assert reasons["codex/characters/mara.yml"] == (
            f"'codex/characters/mara.yml' was not read: {ending} {fallback}"
        )

    async def test_a_recap_over_a_broken_sessions_file_says_so_and_goes_on(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/tools.py :: problems.append(UnreadableFile(SESSIONS_FILE, str(exc)))
        Becomes: raise
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        _put(tmp_path, story_id, "sessions.yaml", "sessions:\n- opened_at: today\n")

        out = await _ok(StoryContextTool(), ctx, action="recap")

        assert out["unreadable_files"] == [
            {
                "file": "sessions.yaml",
                "reason": "sessions.yaml does not fit its shape: 'sessions[0].room_id' is missing.",
            }
        ]
        assert out["recent_sessions"] == []
        assert out["next_scene"]["scene"]["id"] == "ch01.s01"


# --------------------------------------------------------------------------------------
# Two stories, one character id
# --------------------------------------------------------------------------------------


class TestTwoStoriesDoNotCollide:
    async def test_the_same_character_id_is_its_own_in_each_story(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/work.py :: return f"{STORIES_DIRNAME}/{self._story_id}/{relative}"
        Becomes: return f"{STORIES_DIRNAME}/{relative}"
        """
        first = await _new_story(tmp_path, "First")
        second = await _new_story(tmp_path, "Second")
        _codex(tmp_path, first, "characters", {"id": "vane", "name": "Lord Vane"})
        _codex(tmp_path, second, "characters", {"id": "vane", "name": "Vane the Younger"})

        one = await _ok(
            StoryCodexTool(), _ctx(tmp_path, story=first), action="get", entry_id="vane"
        )
        two = await _ok(
            StoryCodexTool(), _ctx(tmp_path, story=second), action="get", entry_id="vane"
        )
        assert one["entries"][0]["name"] == "Lord Vane"
        assert two["entries"][0]["name"] == "Vane the Younger"

        sheet = await _ok(
            CharacterSheetTool(), _ctx(tmp_path, story=second), action="get", character_id="vane"
        )
        assert sheet["character"]["name"] == "Vane the Younger"

        # The same scene id in both, written to each story's own file.
        paths: list[str] = []
        for story_id in (first, second):
            ctx = _ctx(tmp_path, conversation=ROOM_B, story=story_id)
            await _ok(StoryLibraryTool(), ctx, action="take_over", story_id=story_id)
            await _ok(StoryOutlineTool(), ctx, action="init", chapter_titles=["One"])
            await _ok(StoryOutlineTool(), ctx, action="set_scene", chapter_id="ch01", title="Start")
            out = await _ok(
                StoryManuscriptTool(), ctx, action="write", scene_id="ch01.s01", text=story_id
            )
            paths.append(out["path"])
        assert paths == [
            f"stories/{first}/manuscript/ch01.s01.md",
            f"stories/{second}/manuscript/ch01.s01.md",
        ]
        for story_id in (first, second):
            assert (tmp_path / "stories" / story_id / "manuscript" / "ch01.s01.md").read_text(
                encoding="utf-8"
            ) == story_id


# --------------------------------------------------------------------------------------
# The manuscript: no lost text
# --------------------------------------------------------------------------------------


class TestTheManuscriptKeepsWhatItReplaces:
    async def test_a_rewrite_keeps_the_old_text_and_logs_both_writes(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/work.py :: if current is not None and current.digest == expected_digest:
        Becomes: if False:
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        tool = StoryManuscriptTool()
        await _ok(tool, ctx, action="write", scene_id="ch01.s01", text="First draft.")
        read = await _ok(tool, ctx, action="read", scene_id="ch01.s01")

        out = await _ok(
            tool,
            ctx,
            action="write",
            scene_id="ch01.s01",
            text="Second draft.",
            digest=read["digest"],
        )

        history = tmp_path / "stories" / story_id / "manuscript" / ".history" / "ch01.s01"
        assert (history / f"{read['digest']}.md").read_text(encoding="utf-8") == "First draft."
        log = yaml.safe_load((history / "revisions.yaml").read_text(encoding="utf-8"))
        assert [r["digest"] for r in log["revisions"]] == [read["digest"], out["digest"]]
        assert (
            log["revisions"][1]["replaced"] == f"manuscript/.history/ch01.s01/{read['digest']}.md"
        )
        assert log["revisions"][1]["room_id"] == ROOM_A
        assert "notes" not in out

    async def test_writing_over_text_without_its_digest_names_the_digest(
        self, tmp_path: Path
    ) -> None:
        """#1613: a model that had just read the scene was told only to read it again, did,
        and repeated the same digest-less write until the step cap. The refusal names the
        argument to pass, and no file.

        Killed by: src/uclone_x/story/tools.py :: if digest is None:
        Becomes: if False:
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        tool = StoryManuscriptTool()
        await _ok(tool, ctx, action="write", scene_id="ch01.s01", text="First draft.")

        result = await _call(tool, ctx, action="write", scene_id="ch01.s01", text="Oops.")

        assert not result.success
        assert result.error == (
            "Scene 'ch01.s01' already has text, so nothing was written. To replace it, pass "
            "the digest that reading the scene returned, as 'digest'."
        )
        scene = tmp_path / "stories" / story_id / "manuscript" / "ch01.s01.md"
        assert scene.read_text(encoding="utf-8") == "First draft."

    async def test_writing_with_a_stale_digest_says_to_read_again(self, tmp_path: Path) -> None:
        """A digest that no longer matches is a different case from a missing one: the
        scene changed since it was read (here, by hand), so reading again is the fix.

        Killed by: src/uclone_x/story/tools.py :: if digest is None:
        Becomes: if True:
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        tool = StoryManuscriptTool()
        await _ok(tool, ctx, action="write", scene_id="ch01.s01", text="First draft.")
        read = await _ok(tool, ctx, action="read", scene_id="ch01.s01")
        _put(tmp_path, story_id, "manuscript/ch01.s01.md", "Edited by hand.")

        result = await _call(
            tool, ctx, action="write", scene_id="ch01.s01", text="Mine.", digest=read["digest"]
        )

        assert not result.success
        assert result.error == (
            "Scene 'ch01.s01' changed after it was read, so nothing was written. Read it "
            "again and make the change on the current version, with the digest that returns."
        )
        scene = tmp_path / "stories" / story_id / "manuscript" / "ch01.s01.md"
        assert scene.read_text(encoding="utf-8") == "Edited by hand."

    async def test_a_scene_not_in_the_outline_is_refused(self, tmp_path: Path) -> None:
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        result = await _call(
            StoryManuscriptTool(), ctx, action="write", scene_id="ch09.s01", text="Lost."
        )
        assert result.error == (
            "The outline has no scene 'ch09.s01', so nothing was written. Add it with "
            "story_outline 'set_scene' first."
        )

    async def test_a_read_only_conversation_reads_but_does_not_write(self, tmp_path: Path) -> None:
        story_id = await _new_story(tmp_path)
        await _outlined(tmp_path, story_id)
        other = _ctx(tmp_path, conversation=ROOM_B, story=story_id)
        opened = await _ok(StoryLibraryTool(), other, action="open", story_id=story_id)
        assert opened["writable"] is False

        listing = await _ok(StoryManuscriptTool(), other, action="list")
        assert [s["scene_id"] for s in listing["scenes"]] == ["ch01.s01", "ch01.s02"]
        result = await _call(
            StoryManuscriptTool(), other, action="write", scene_id="ch01.s01", text="Mine."
        )
        assert not result.success
        assert result.error is not None and "Another conversation" in result.error
        assert not (tmp_path / "stories" / story_id / "manuscript").exists()

    async def test_story_tools_outside_an_open_story_say_so(self, tmp_path: Path) -> None:
        result = await _call(StoryContextTool(), _ctx(tmp_path), action="recap")
        assert (
            result.error == "No story is open in this conversation. Open or create a story first."
        )


# --------------------------------------------------------------------------------------
# The outline
# --------------------------------------------------------------------------------------


class TestAWriteSaysWhatTheStoryChangedBefore:
    """A saved scene is told when it names the dead or a lost thing, without being asked (#1613).

    The eval's moon-seal story: 예린 dies at ch01.s03 (story time 3), 라온 loses 월광검 to
    카엘 at ch02.s01 (time 4), and ch02.s03 is a flashback at time 0, before both.
    """

    FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "writer"

    def _moon_seal(self, workspace: Path) -> ToolContext:
        shutil.copytree(self.FIXTURE, workspace, dirs_exist_ok=True)
        return _ctx(workspace, conversation="writer-eval-room", story="moon-seal")

    async def _write(self, workspace: Path, scene_id: str, text: str) -> dict[str, Any]:
        return await _ok(
            StoryManuscriptTool(), self._moon_seal(workspace), action="write",
            scene_id=scene_id, text=text,
        )  # fmt: skip

    async def test_a_scene_after_the_death_that_names_the_dead_is_told(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/context.py :: and "status" in snapshot.set_at
        Becomes: and "status" not in snapshot.set_at
        """
        out = await self._write(tmp_path, "ch02.s04", "예린이 웃으며 라온의 어깨를 두드렸다.")

        assert out["digest"]
        assert (tmp_path / out["path"]).read_text(encoding="utf-8").startswith("예린이")
        [line] = out["continuity"]
        assert "예린" in line and "ch01.s03" in line and "dead" in line
        assert "고쳐 다시 쓰세요" in out["continuity_note"]

    async def test_a_lost_thing_names_who_lost_it_last_and_who_has_it(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: max(lost, key=lambda found: found[0])
        Becomes: min(lost, key=lambda found: found[0])
        """
        out = await self._write(tmp_path, "ch02.s04", "라온은 월광검을 높이 들었다.")

        [line] = out["continuity"]
        assert line.startswith("라온 no longer has 월광검")
        assert "ch02.s01" in line and "카엘 has it now" in line

    async def test_a_flashback_before_the_death_is_not_told(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: item.entry, placements, scene_id, through_scene=False
        Becomes: item.entry, placements, list(placements)[-1], through_scene=False
        """
        out = await self._write(
            tmp_path, "ch02.s03", "어린 라온은 예린에게서 월광검을 처음 받았다."
        )

        assert "continuity" not in out and "continuity_note" not in out
        assert "notes" not in out

    async def test_the_notice_is_in_plain_words(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: name = item.entry.name
        Becomes: name = item.entry.id
        """
        out = await self._write(tmp_path, "ch02.s04", "예린이 월광검을 라온에게 돌려주었다.")

        again = await _ok(
            StoryManuscriptTool(), self._moon_seal(tmp_path), action="write",
            scene_id="ch02.s04", text="예린이 월광검을 쥐었다.", digest=out["digest"],
        )  # fmt: skip
        words = " ".join(
            [
                *out["continuity"],
                out["continuity_note"],
                *again["continuity"],
                again["continuity_note"],
            ]
        )
        assert len(out["continuity"]) == 2
        for internal in (
            "yerin", "raon", "kael", "moon_sword", "progression", "codex", "status",
            "possesses", "state", ".yaml", "stories/", "Traceback",
        ):  # fmt: skip
            assert internal not in words, internal

    async def test_the_notice_quotes_the_sentence_that_names_them(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: found.append(f'"{sentence}"')
        Becomes: pass
        """
        out = await self._write(
            tmp_path, "ch02.s04", "새벽이 밝았다. 예린이 라온의 자세를 바로잡았다. 바람이 불었다."
        )

        [line] = out["continuity"]
        assert '"예린이 라온의 자세를 바로잡았다."' in line
        assert "새벽이 밝았다" not in line and "바람이 불었다" not in line

    async def test_the_first_notice_says_to_rewrite_now_in_the_scenes_language(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/context.py :: korean = bool(_HANGUL.search(text))
        Becomes: korean = False
        """
        out = await self._write(tmp_path, "ch02.s04", "예린이 웃으며 라온의 어깨를 두드렸다.")

        note = out["continuity_note"]
        assert note.startswith("아직 끝나지 않았습니다. 이 장면을 고쳐 다시 쓰세요")
        assert "'ch02.s04'" in note and f"'{out['digest']}'" in note
        assert "회상" in note  # a flashback may stay as it is: a notice, not a refusal

    async def test_a_rewrite_of_its_own_text_is_not_sent_round_again(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/tools.py :: rewrite_of_own = work.last_writer(scene_id) == room_id
        Becomes: rewrite_of_own = False
        """
        first = await self._write(tmp_path, "ch02.s04", "예린이 웃으며 라온의 어깨를 두드렸다.")
        again = await _ok(
            StoryManuscriptTool(), self._moon_seal(tmp_path), action="write",
            scene_id="ch02.s04", text="예린이 라온의 검을 바로잡았다.", digest=first["digest"],
        )  # fmt: skip

        assert again["continuity"]
        note = again["continuity_note"]
        assert "다시 고치지는 마세요" in note and "고쳐 다시 쓰세요" not in note
        assert again["digest"] not in note

    async def test_another_conversations_text_is_a_first_notice(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/tools.py :: rewrite_of_own = work.last_writer(scene_id) == room_id
        Becomes: rewrite_of_own = work.last_writer(scene_id) is not None
        """
        first = await self._write(tmp_path, "ch02.s04", "예린이 웃으며 라온의 어깨를 두드렸다.")
        [log] = (tmp_path / "stories" / "moon-seal").rglob("revisions.yaml")
        text = log.read_text(encoding="utf-8")
        assert "writer-eval-room" in text
        log.write_text(text.replace("writer-eval-room", "another-room"), encoding="utf-8")
        again = await _ok(
            StoryManuscriptTool(), self._moon_seal(tmp_path), action="write",
            scene_id="ch02.s04", text="예린이 다시 웃었다.", digest=first["digest"],
        )  # fmt: skip

        assert "고쳐 다시 쓰세요" in again["continuity_note"]


class TestAMemoryOfTheDeadIsNotTold:
    """A sentence that only remembers the dead is the story's own, not a slip (#1808).

    The moon-seal story again: 예린 is dead since ch01.s03, and ch02.s04 comes after.
    """

    FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "writer"

    async def _write(self, workspace: Path, text: str) -> dict[str, Any]:
        shutil.copytree(self.FIXTURE, workspace, dirs_exist_ok=True)
        return await _ok(
            StoryManuscriptTool(),
            _ctx(workspace, conversation="writer-eval-room", story="moon-seal"),
            action="write",
            scene_id="ch02.s04",
            text=text,
        )

    async def test_a_sentence_remembering_the_dead_is_not_told(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: _RECALL.search(sentence)
        Becomes: False
        """
        out = await self._write(tmp_path, "라온은 스승 예린을 떠올렸다. 바람이 차가웠다.")

        assert out["digest"]
        assert "continuity" not in out and "continuity_note" not in out

    async def test_a_sentence_where_the_dead_act_is_told(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: if quoted is None:
        Becomes: if quoted is not None:
        """
        out = await self._write(tmp_path, "예린이 라온의 검을 바로잡았다.")

        [line] = out["continuity"]
        assert '"예린이 라온의 검을 바로잡았다."' in line

    async def test_in_mixed_text_only_the_acting_sentence_is_quoted(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: if passed_over and not found:
        Becomes: if passed_over:
        """
        out = await self._write(
            tmp_path, "라온은 예린을 기억했다. 그때 예린이 라온의 어깨를 두드렸다."
        )

        [line] = out["continuity"]
        assert '"그때 예린이 라온의 어깨를 두드렸다."' in line
        assert "기억했다" not in line


class TestASentenceThatSaysTheChangeIsNotTold:
    """A sentence that says the dead are dead, the lost is lost, or that the act did not
    happen keeps the scene true to the change, so it is not quoted as a slip (#1808).

    The sentences are from a qwen3:8b Writer run on the moon-seal story, where 예린 is dead
    since ch01.s03 and 월광검 lost at ch02.s01, both before ch02.s04.
    """

    async def _write(self, workspace: Path, text: str) -> dict[str, Any]:
        shutil.copytree(TestAMemoryOfTheDeadIsNotTold.FIXTURE, workspace, dirs_exist_ok=True)
        return await _ok(
            StoryManuscriptTool(),
            _ctx(workspace, conversation="writer-eval-room", story="moon-seal"),
            action="write",
            scene_id="ch02.s04",
            text=text,
        )

    async def test_the_dead_said_to_be_gone_is_not_told_and_an_act_still_is(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/context.py :: 떠났|떠난
        Becomes: 떠난
        """
        out = await self._write(
            tmp_path,
            '도윤이 모닥불 앞에서 조용히 말했다. "예린이 떠났을 때, 그의 검을 지키려고 했어."\n'
            "예린이 칼을 들고 라온에게 다가왔다.",
        )

        [line] = out["continuity"]
        assert '"예린이 칼을 들고 라온에게 다가왔다."' in line
        assert "떠났을" not in line

    async def test_departure_and_loss_stated_together_are_not_told(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: or _STATED_OR_DENIED.search(sentence)
        Becomes: or False
        """
        out = await self._write(
            tmp_path,
            "그는 라온의 자세를 바라보며, 그녀가 스승 예린의 교훈을 잊지 않았다는 걸 느낀다. "
            "그러나 예린은 이미 영원히 떠났고, 월광검도 카엘에게 빼앗겼다.",
        )

        assert out["digest"]
        assert "continuity" not in out and "continuity_note" not in out

    async def test_the_lost_sword_said_to_be_taken_is_not_told(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: 잃은|빼앗겼|
        Becomes: 잃은|
        """
        for workspace, text in (
            (tmp_path / "run", "하지만 이제 그는 죽었고, 월광검도 빼앗겼다."),
            (tmp_path / "taken", "월광검은 카엘에게 빼앗겼다."),
        ):
            out = await self._write(workspace, text)

            assert "continuity" not in out and "continuity_note" not in out, text

    async def test_a_use_of_the_lost_sword_negated_is_not_told(self, tmp_path: Path) -> None:
        """A word between the name and the negated verb is ordinary narration (#1808).

        Killed by: src/uclone_x/story/context.py :: 않았|지
        Becomes: 않았음|지
        """
        for name, text in (
            ("drawn", "라온은 허리에 손을 얹었지만, 월광검을 뽑아내지 않았다."),
            ("sheathed", "월광검은 끝내 칼집에서 나오지 않았다."),
            ("never", "라온은 월광검을 다시는 뽑지 않았다."),
            ("unseen", "라온은 예린을 다시 보지 않았다."),
        ):
            out = await self._write(tmp_path / name, text)

            assert "continuity" not in out and "continuity_note" not in out, text

    async def test_a_use_refused_or_replaced_is_not_told(self, tmp_path: Path) -> None:
        r"""Two sentences the 417602e6 re-measure quoted that keep the sword lost (#1808).

        Killed by: src/uclone_x/story/context.py :: 꺼내)\S*지\s*않|아닌)
        Becomes: 꺼내)\S*지\s*않았|아닌)
        Killed by: src/uclone_x/story/context.py :: 않|아닌)"
        Becomes: 않)"
        Killed by: src/uclone_x/story/context.py :: or (lost and _use_denied(
        Becomes: or (False and _use_denied(
        """
        for workspace, text in (
            (
                tmp_path / "refused",
                "라온은 월광검을 뽑지 않고, 몸으로 안개를 뚫으며 자세를 잡는다.",
            ),
            (tmp_path / "replaced", "라온은 허리에 월광검이 아닌 칼을 차고 있었다."),
            (tmp_path / "used_not", "라온은 월광검을 쓰지 않고 맨손으로 싸웠다."),
            (tmp_path / "drawn_not", "라온은 월광검을 꺼내지 않고 주먹을 쥐었다."),
            (tmp_path / "held_not", "라온은 월광검을 잡지 않고 등을 돌렸다."),
            (tmp_path / "applied_not", "라온은 월광검을 사용하지 않고 피했다."),
        ):
            out = await self._write(workspace, text)

            assert "continuity" not in out and "continuity_note" not in out, text

    async def test_a_use_after_a_negation_that_is_not_its_own_is_told(self, tmp_path: Path) -> None:
        r"""A "~지 않고" is also "without ~ing": only a negation right after the name excuses
        a use, and these three are slips (#1808).

        Killed by: src/uclone_x/story/context.py :: r"|지\s*않았|지\s*않는|지\s*못"
        Becomes: r"|지\s*않|지\s*못|아닌"
        """
        for name, text in (
            ("hesitate", "라온은 주저하지 않고 월광검을 뽑아 들었다."),
            ("stop", "라온은 멈추지 않고 월광검을 휘둘렀다."),
            ("dream", "꿈이 아닌 듯, 라온은 월광검을 쥐었다."),
        ):
            out = await self._write(tmp_path / name, text)

            [line] = out["continuity"]
            assert f'"{text}"' in line, text

    async def test_a_negation_after_the_lost_sword_that_is_not_its_use_is_told(
        self, tmp_path: Path
    ) -> None:
        r"""Only the sword's own use verbs, negated, excuse it: hesitating is not one (#1808).

        Killed by: src/uclone_x/story/context.py :: (?:(?:뽑|쥐|휘두르|들|차|겨누|베|끼|착용하|챙기|쓰|사용하|잡|꺼내)\S*지
        Becomes: (?:\S*지
        """
        text = "라온은 월광검을 주저하지 않고 뽑았다."
        out = await self._write(tmp_path, text)

        [line] = out["continuity"]
        assert f'"{text}"' in line

    async def test_a_negation_after_the_dead_is_an_act_and_is_told(self, tmp_path: Path) -> None:
        r"""A refused use excuses only a lost item: the dead doing anything is the slip (#1808).

        Killed by: src/uclone_x/story/context.py :: or (lost and _use_denied(
        Becomes: or (_use_denied(
        Killed by: src/uclone_x/story/context.py :: |꿈에"
        Becomes: |꿈에|잊지\s*(?:않|못)"
        """
        for name, text in (
            ("hesitate", "예린은 주저하지 않고 칼을 들었다."),
            ("sheathe", "예린은 휘두르지 않고 검을 거두었다."),
            ("remember", "예린은 잊지 않고 매일 라온을 찾아왔다."),
        ):
            out = await self._write(tmp_path / name, text)

            [line] = out["continuity"]
            assert f'"{text}"' in line, text

    async def test_english_that_says_the_dead_died_is_not_told(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: (?:died|
        Becomes: (?:
        """
        out = await self._write(
            tmp_path, "예린 died at the gate. Raon did not draw 월광검. 예린 smiled at Raon."
        )

        [line] = out["continuity"]
        assert '"예린 smiled at Raon."' in line
        assert "died" not in line


class TestAWriteRefusesWhatASmallModelLeavesInProse:
    """Four faults are refused at write time, in plain words, with nothing written (#1808)."""

    async def _refused(self, workspace: Path, text: str) -> str:
        story_id = await _new_story(workspace)
        ctx = await _outlined(workspace, story_id)
        result = await _call(
            StoryManuscriptTool(), ctx, action="write", scene_id="ch01.s01", text=text
        )
        assert not result.success
        assert result.error is not None
        assert not list((workspace / "stories" / story_id).rglob("ch01.s01.md"))
        for internal in ("Error", "Traceback", "regex", "pattern", "_", "\\", "[가-힣]"):
            assert internal not in result.error, internal
        return result.error

    async def _saved(self, workspace: Path, text: str) -> None:
        story_id = await _new_story(workspace)
        ctx = await _outlined(workspace, story_id)
        await _ok(StoryManuscriptTool(), ctx, action="write", scene_id="ch01.s01", text=text)

    async def test_a_chinese_character_stuck_in_a_korean_word_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/prose.py :: glued = _glued(text, _GLUED_HAN)
        Becomes: glued = []
        Killed by: src/uclone_x/story/tools.py :: raise StoryError(refused)
        Becomes: pass
        """
        error = await self._refused(
            tmp_path, "불씨가 꺼지고 잔烬만 남았다. 그는 둔钝한 칼을 들었다."
        )

        assert error == (
            '"잔烬만", "둔钝한": 한글 낱말 안에 한자가 붙어 있습니다. 한글로 고치거나, 한자를 '
            "꼭 쓰려면 한자(漢字)처럼 괄호 안에 쓰세요. 장면은 저장되지 않았습니다. 고친 글로 "
            "다시 저장하세요."
        )

    async def test_latin_glued_to_a_korean_word_is_refused_in_plain_words(
        self, tmp_path: Path
    ) -> None:
        """Two words of the 417602e6 re-measure (#1808).

        Killed by: src/uclone_x/story/prose.py :: latin = _glued(text, _GLUED_LATIN)
        Becomes: latin = []
        """
        error = await self._refused(
            tmp_path, "라온은 왼blick 손을 들었다. 그는 왼linkplain 쪽으로 걸었다."
        )

        assert error == (
            '"왼blick", "왼linkplain": 한글 낱말 안에 영문이 붙어 있습니다. 한글로 고치거나, '
            "영문을 꼭 쓰려면 띄어 쓰세요. 장면은 저장되지 않았습니다. 고친 글로 다시 저장하세요."
        )

    async def test_a_foreign_name_before_a_particle_or_two_letters_is_saved(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/prose.py :: [a-z]{3,}
        Becomes: [a-zA-Z]{2,}
        """
        await self._saved(
            tmp_path, "그는 AI를 믿지 않았다. 한국vs일본 경기가 열렸다. 도윤은 웃었다."
        )

    async def test_hanja_in_brackets_or_before_a_particle_is_saved(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/prose.py :: for raw in _BRACKETED.sub(" ", text).split():
        Becomes: for raw in text.split():
        Killed by: src/uclone_x/story/prose.py :: _GLUED_HAN = re.compile(r"[가-힣]
        Becomes: _GLUED_HAN = re.compile(r"[㐀-䶿一-鿿][가-힣]|[가-힣]
        """
        await self._saved(
            tmp_path, "그는 검(劍)을 들었다. 사람들은 그 불을 (잔烬이라) 불렀다. 韓國은 멀었다."
        )

    async def test_the_same_sentence_four_times_is_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/prose.py :: _SAME_SENTENCE = 4
        Becomes: _SAME_SENTENCE = 99
        """
        loop = "그는 문을 열고 다시 밖으로 나갔다. "
        text = (
            "비가 그치고 새벽이 왔다. " + loop * 2 + "마당에는 아무도 없었다. " + loop * 2
            + "등불이 하나씩 꺼져 갔다. 멀리서 종소리가 울렸다."
        )  # fmt: skip

        error = await self._refused(tmp_path, text)

        assert error.startswith(
            '"그는 문을 열고 다시 밖으로 나갔다." 같은 문장이 되풀이됩니다(이 문장만 4번).'
        )
        assert error.endswith("장면은 저장되지 않았습니다. 고친 글로 다시 저장하세요.")

    async def test_a_few_sentences_said_over_and_over_are_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/prose.py :: len(times) / len(counted) < _DISTINCT_SHARE
        Becomes: len(times) / len(counted) < 0
        """
        text = " ".join(
            ["The wind came down the hill.", "She waited by the gate.", "Nobody came for her."] * 3
        )

        error = await self._refused(tmp_path, text)

        assert '"The wind came down the hill." 3 times' in error
        assert error.endswith("The scene was not saved. Save it again once it is fixed.")

    async def test_a_refrain_said_three_times_is_saved(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/prose.py :: _SAME_SENTENCE = 4
        Becomes: _SAME_SENTENCE = 3
        """
        refrain = "달은 아직 지지 않았다.\n"
        await self._saved(
            tmp_path,
            refrain + "라온은 성벽을 따라 걸었다.\n" + refrain + "도윤이 뒤에서 불렀다.\n"
            + refrain + "둘은 말없이 동쪽을 보았다.",
        )  # fmt: skip

    async def test_short_lines_repeated_in_dialogue_are_saved(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/prose.py :: _MIN_COUNTED = 8
        Becomes: _MIN_COUNTED = 1
        """
        await self._saved(
            tmp_path,
            '"갈 거야?" "응." "정말?" "응." "혼자서?" "응." "그래." "응." 라온은 문을 닫았다.',
        )

    async def test_a_note_in_brackets_after_the_story_is_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/prose.py :: if len(lines) < 2 or not _whole_note(lines[-1]):
        Becomes: if True:
        """
        note = "(이 장면은 라온의 상실감을 강조하기 위해 짧게 썼습니다.)"
        error = await self._refused(tmp_path, f"라온은 빈 칼집을 쥐었다.\n\n{note}")

        assert error == (
            f'장면이 이야기 뒤에 괄호로 된 메모로 끝납니다: "{note}". 글에 대한 설명이라면 '
            "빼고, 하고 싶은 말은 답장에 쓰세요. 장면은 저장되지 않았습니다. 고친 글로 다시 "
            "저장하세요."
        )

    async def test_a_scripts_stage_directions_and_a_short_close_are_saved(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/prose.py :: if any(_whole_note(line) for line in lines[:-1]):
        Becomes: if False:
        Killed by: src/uclone_x/story/prose.py :: _MIN_NOTE = 20
        Becomes: _MIN_NOTE = 0
        """
        await self._saved(
            tmp_path,
            "(무대 왼쪽, 불 꺼진 성벽 위에 라온이 홀로 서 있다.)\n라온: 스승님.\n"
            "(도윤이 등불을 들고 천천히 오른쪽에서 걸어 들어온다.)",
        )
        await self._saved(tmp_path / "second", "라온은 등불을 껐다.\n\n(암전)")


class TestTheCodexTakesANameForAnId:
    """'entry_id' takes the name or an alias the prose uses, as well as the id (#1808)."""

    FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "writer"

    def _moon_seal(self, workspace: Path) -> ToolContext:
        shutil.copytree(self.FIXTURE, workspace, dirs_exist_ok=True)
        return _ctx(workspace, conversation="writer-eval-room", story="moon-seal")

    async def _propose(self, ctx: ToolContext, entry_id: str) -> ToolResult:
        return await _call(
            StoryCodexTool(), ctx, action="propose", entry_id=entry_id, at="ch02.s01",
            set={"wounded": True}, quote="도윤의 화살이 날아갔을 때",
        )  # fmt: skip

    async def test_a_proposal_by_name_is_made_for_the_entry_of_that_name(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/tools.py :: named = _named_entries(work.codex(), entry_id, kind, done="nothing was proposed")
        Becomes: raise StoryError("nothing was proposed")
        """
        result = await self._propose(self._moon_seal(tmp_path), "도윤")

        assert result.success, result.error
        assert isinstance(result.output, dict)
        path = result.output["path"]
        assert isinstance(path, str)
        saved = (tmp_path / path).read_text(encoding="utf-8")
        assert "doyun" in saved

    async def test_a_name_two_entries_share_is_refused_with_both(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/tools.py :: if len({(item.kind, item.entry.id) for item in named}) > 1:
        Becomes: if False:
        """
        ctx = self._moon_seal(tmp_path)
        _codex(
            tmp_path, "moon-seal", "characters",
            {"id": "doyun_elder", "name": "도윤 할아버지", "aliases": ["도윤"]},
        )  # fmt: skip

        result = await self._propose(ctx, "도윤")

        assert result.error == (
            "'도윤' names more than one entry: 'doyun' (도윤), 'doyun_elder' (도윤 할아버지). "
            "Say which, by its id in 'entry_id', so nothing was proposed."
        )
        assert not (tmp_path / "stories" / "moon-seal" / "proposals").exists()

    async def test_a_name_no_entry_has_is_refused_with_the_entries_there_are(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/tools.py :: hint = f" Its entries are: {_candidates(listed)}." if listed else " It has no entries yet."
        Becomes: hint = ""
        """
        result = await _call(
            StoryCodexTool(), self._moon_seal(tmp_path), action="get", entry_id="세린",
            kind="characters",
        )  # fmt: skip

        assert result.error == (
            "The story's characters has no entry called '세린', so nothing was read. Its "
            "entries are: 'doyun' (도윤), 'kael' (카엘), 'raon' (라온), 'yerin' (예린)."
        )

    async def test_get_by_name_reads_the_entry(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/names.py :: for name in (item.entry.id, item.entry.name, *item.entry.aliases)
        Becomes: for name in (item.entry.id,)
        """
        out = await _ok(StoryCodexTool(), self._moon_seal(tmp_path), action="get", entry_id="카엘")

        [entry] = out["entries"]
        assert entry["id"] == "kael"


class TestForSceneNamesWhatTheRequestAsksThatTheStoryEnded:
    """'for_scene' with the person's request lists, first, what it asks that the story ended (#1613).

    The eval's `tempt` request at ch02.s04 (story time 6) asks for 예린, dead at ch01.s03,
    to stand beside 라온 and correct him, and for 라온 to draw 월광검, lost to 카엘 at
    ch02.s01. ch02.s03 is a flashback at time 0, before both.
    """

    TEMPT = (
        "1) 라온이 늘 그랬듯 허리의 월광검을 뽑아 자세를 잡는다 "
        "2) 스승 예린이 곁에서 라온의 자세를 바로잡아 준다 3) 도윤이 멀리서 지켜본다"
    )

    async def _for_scene(self, workspace: Path, scene_id: str, **args: Any) -> dict[str, Any]:
        shutil.copytree(
            TestAWriteSaysWhatTheStoryChangedBefore.FIXTURE, workspace, dirs_exist_ok=True
        )
        ctx = _ctx(workspace, conversation="writer-eval-room", story="moon-seal")
        return await _ok(StoryContextTool(), ctx, action="for_scene", scene_id=scene_id, **args)

    async def test_the_dead_teacher_and_the_lost_sword_come_first_one_line_each(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/context.py :: request_conflicts(outline, scene_id, codex, request) if request else ([], None)
        Becomes: request_conflicts(outline, scene_id, codex, "") if request else ([], None)
        """
        out = await self._for_scene(tmp_path, "ch02.s04", request=self.TEMPT)

        assert list(out)[:2] == ["request_conflicts", "request_conflicts_note"]
        dead, lost = out["request_conflicts"]
        assert dead.startswith("예린:") and "「스승의 최후」" in dead and "기억이나 슬픔" in dead
        assert lost.startswith("월광검:") and "「빼앗긴 검」" in lost and "라온" in lost
        assert "승인" in out["request_conflicts_note"]  # how the person changes it

    async def test_the_lost_sword_line_says_who_has_it_now(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: now_ko = f" 지금은 {', '.join(change.holders)}에게 있습니다." if change.holders else ""
        Becomes: now_ko = ""
        """
        out = await self._for_scene(tmp_path, "ch02.s04", request="라온이 월광검을 뽑는다")

        [lost] = out["request_conflicts"]
        assert "지금은 카엘에게 있습니다" in lost

    async def test_a_request_for_a_living_character_raises_nothing(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: and _mentions(item.entry, haystack)
        Becomes: and True
        """
        out = await self._for_scene(tmp_path, "ch02.s04", request="도윤이 멀리서 지켜본다")

        assert "request_conflicts" not in out and "request_conflicts_note" not in out

    async def test_a_flashback_before_the_death_raises_nothing(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: for change in named_changes(outline, scene_id, codex, request):
        Becomes: for change in named_changes(outline, "ch02.s04", codex, request):
        """
        out = await self._for_scene(tmp_path, "ch02.s03", request=self.TEMPT)

        assert "request_conflicts" not in out and "request_conflicts_note" not in out

    async def test_without_a_request_the_bundle_is_as_it_was(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: if conflicts:
        Becomes: if conflicts is not None:
        """
        plain = await self._for_scene(tmp_path, "ch02.s04")
        empty = await self._for_scene(tmp_path, "ch02.s04", request="")

        assert list(plain)[0] == "story"
        assert "request_conflicts" not in plain and plain == empty

    async def test_the_lines_are_in_plain_words(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/context.py :: title = found[1].title if found and found[1].title else at
        Becomes: title = at
        """
        out = await self._for_scene(tmp_path, "ch02.s04", request=self.TEMPT)

        words = " ".join([*out["request_conflicts"], out["request_conflicts_note"]])
        for internal in (
            "yerin", "raon", "kael", "moon_sword", "ch01", "ch02", "progression", "codex",
            "status", "possesses", "state", ".yaml", "stories/", "Traceback", "_",
        ):  # fmt: skip
            assert internal not in words, internal


class TestTheOutline:
    async def test_new_scenes_get_the_next_free_id_and_move_keeps_order(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/tools.py :: while f"{chapter_id}.s{number:02d}" in taken:
        Becomes: while False:
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        tool = StoryOutlineTool()
        added = await _ok(
            tool,
            ctx,
            action="set_scene",
            chapter_id="ch02",
            chapter_title="The Far Bank",
            title="Landing",
        )
        assert added["outline"] == "scene 'ch02.s01' added to chapter 'ch02'"
        await _ok(
            tool, ctx, action="move", scene_id="ch02.s01", chapter_id="ch01", before="ch01.s02"
        )

        got = await _ok(tool, ctx, action="get")
        assert [[s["id"] for s in c["scenes"]] for c in got["chapters"]] == [
            ["ch01.s01", "ch02.s01", "ch01.s02"],
            [],
        ]

    async def test_a_second_init_does_not_replace_the_outline(self, tmp_path: Path) -> None:
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        result = await _call(StoryOutlineTool(), ctx, action="init", chapter_titles=["Again"])
        assert result.error is not None and "already has an outline" in result.error

    async def test_changing_a_scenes_chapter_goes_through_move(self, tmp_path: Path) -> None:
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        result = await _call(
            StoryOutlineTool(),
            ctx,
            action="set_scene",
            scene_id="ch01.s01",
            chapter_id="ch02",
            title="X",
        )
        assert (
            result.error
            == "Scene 'ch01.s01' is in chapter 'ch01'. Use 'move' to put it in another chapter."
        )


# --------------------------------------------------------------------------------------
# One scene's context, and what it leaves out
# --------------------------------------------------------------------------------------


class TestASceneContextSaysWhatItLeftOut:
    async def test_named_always_and_mentioned_entries_go_in_with_their_reasons(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/context.py :: for name in (entry.name, *entry.aliases)
        Becomes: for name in (entry.name,)
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        _codex(tmp_path, story_id, "characters", {"id": "mara", "name": "Mara"})
        _codex(
            tmp_path, story_id, "places", {"id": "ferry", "name": "The Ferry", "aliases": ["toll"]}
        )
        _codex(
            tmp_path, story_id, "items", {"id": "law", "name": "River Law", "always_include": True}
        )
        _codex(tmp_path, story_id, "items", {"id": "sword", "name": "Sword"})
        _codex(
            tmp_path,
            story_id,
            "threads",
            {
                "id": "debt",
                "name": "The debt",
                "progressions": [{"at": "ch01.s02", "set": {"paid": True}}],
            },
        )

        out = await _ok(StoryContextTool(), ctx, action="for_scene", scene_id="ch01.s02")

        manifest = out["manifest"]
        assert {(i["id"], i["reason"]) for i in manifest["included"]} == {
            ("mara", "named in the scene"),
            ("law", "always included"),
            ("ferry", "mentioned in the scene or the end of the scene before"),
        }
        assert {e["id"] for e in manifest["left_out"]} == {"sword", "debt"}
        assert manifest["named_without_an_entry"] == [{"id": "vane", "kind": "characters"}]
        assert out["previous_scene"]["scene_id"] == "ch01.s01"
        assert out["next_scene"] is None
        # 'debt' is not in the bundle, so its change at this scene is not listed.
        assert "changes_in_this_scene" not in manifest

    def test_entries_over_the_budget_are_listed_as_left_out(self) -> None:
        """Killed by: src/uclone_x/story/context.py :: if len(entries) >= MAX_ENTRIES:
        Becomes: if False:
        """
        from uclone_x.story.context import scene_context

        outline = Outline.model_validate(
            {"chapters": [{"id": "c", "title": "C", "scenes": [{"id": "s", "title": "S"}]}]}
        )
        items = tuple(
            CodexItem(
                "characters", CharacterEntry(id=f"p{i:02d}", name=f"P{i}", always_include=True)
            )
            for i in range(MAX_ENTRIES + 2)
        )
        out = scene_context(outline, "s", CodexIndex(items), previous_text=None, story={})
        assert len(out["codex"]) == MAX_ENTRIES
        assert [e["id"] for e in out["manifest"]["left_out"]] == ["p12", "p13"]
        assert out["manifest"]["left_out"][0]["reason"].startswith("over the budget of 12 entries")

    async def test_an_earlier_scene_s_change_is_applied_and_this_scene_s_is_listed(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/context.py :: entries.append(render_entry(item, snapshot))
        Becomes: entries.append(render_entry(item))
        Killed by: src/uclone_x/story/context.py :: manifest["changes_in_this_scene"] = in_this_scene
        Becomes: pass
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        _codex(
            tmp_path,
            story_id,
            "characters",
            {
                "id": "mara",
                "name": "Mara",
                "state": {"arm": "whole", "mood": "calm"},
                "progressions": [
                    {"at": "ch01.s01", "set": {"mood": "wary"}},
                    {"at": "ch01.s02", "set": {"arm": "broken"}},
                ],
            },
        )
        out = await _ok(StoryContextTool(), ctx, action="for_scene", scene_id="ch01.s02")
        assert [e["state"] for e in out["codex"] if e["id"] == "mara"] == [
            {"arm": "whole", "mood": "wary"}
        ]
        manifest = out["manifest"]
        assert manifest["progressions_applied"] == [
            {
                "id": "mara",
                "kind": "characters",
                "applied": [{"at": "ch01.s01", "kind": "state", "set": {"mood": "wary"}}],
            }
        ]
        assert manifest["changes_in_this_scene"] == [
            {
                "id": "mara",
                "kind": "characters",
                "changes": [{"at": "ch01.s02", "kind": "state", "set": {"arm": "broken"}}],
            }
        ]
        assert "progressions_not_applied" not in manifest

    async def test_a_character_s_appearance_reaches_the_writer_at_the_scene_s_moment(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/context.py :: if visual.prose:
        Becomes: if False:
        Killed by: src/uclone_x/story/context.py :: if visual.gender is not None:
        Becomes: if False:
        Killed by: src/uclone_x/story/context.py :: if snapshot is not None and any(p["kind"] == "visual" for p in snapshot.applied):
        Becomes: if snapshot is not None:
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        prose = "허리까지 오는 은발에 푸른 눈. 왼뺨에 가는 흉터가 있다."
        _codex(
            tmp_path,
            story_id,
            "characters",
            {
                "id": "mara",
                "name": "Mara",
                "visual": {
                    "tags": ["silver_hair", "blue_eyes"],
                    "prose": prose,
                    "gender": "female",
                    "base_seed": 41027,
                    "negative_tags": ["hat"],
                    "progressions": [{"at": "ch01.s01", "add_tags": ["scar_on_cheek"]}],
                },
            },
        )
        _codex(tmp_path, story_id, "characters", {"id": "vane", "name": "Vane"})
        tool = StoryContextTool()

        first = await _ok(tool, ctx, action="for_scene", scene_id="ch01.s01")
        [mara] = [e for e in first["codex"] if e["id"] == "mara"]
        assert mara["appearance"] == prose
        assert mara["gender"] == "female"
        assert mara["visual_tags"] == ["silver_hair", "blue_eyes"]
        assert "appearance_note" not in mara
        # Only what a prose writer uses: the illustrator's fields stay in the codex.
        assert not {"base_seed", "negative_tags", "prose", "visual"} & mara.keys()

        second = await _ok(tool, ctx, action="for_scene", scene_id="ch01.s02")
        entries = {e["id"]: e for e in second["codex"]}
        assert entries["mara"]["visual_tags"] == ["silver_hair", "blue_eyes", "scar_on_cheek"]
        assert entries["mara"]["appearance"] == prose
        assert "visual_tags" in entries["mara"]["appearance_note"]
        vane = entries["vane"]
        assert not {"appearance", "gender", "visual_tags", "appearance_note"} & vane.keys()

    async def test_a_visual_block_without_prose_gives_no_appearance(self, tmp_path: Path) -> None:
        """#1613 item 7: a character drawn by tags alone has no `appearance`, not a null one.

        Killed by: src/uclone_x/story/context.py :: if visual.prose:
        Becomes: if True:
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        _codex(
            tmp_path,
            story_id,
            "characters",
            {"id": "mara", "name": "Mara", "visual": {"tags": ["silver_hair"]}},
        )
        out = await _ok(StoryContextTool(), ctx, action="for_scene", scene_id="ch01.s01")
        [mara] = [e for e in out["codex"] if e["id"] == "mara"]
        assert mara["visual_tags"] == ["silver_hair"]
        assert "appearance" not in mara

    async def test_an_earlier_death_or_loss_outside_the_scene_s_list_is_one_line(
        self, tmp_path: Path
    ) -> None:
        """#1613: the live writer eval's Writer, handed 예린's full entry only when the scene's
        beats happened to name her, wrote the dead teacher correcting a stance.

        Killed by: src/uclone_x/story/context.py :: if (item.kind, item.entry.id) in listed_in_full:
        Becomes: if False:
        Killed by: src/uclone_x/story/context.py :: outside = entry_snapshot(item.entry, placements, scene.id, through_scene=False)
        Becomes: outside = entry_snapshot(item.entry, placements, scene.id, through_scene=True)
        Killed by: src/uclone_x/story/context.py :: dated.sort(key=lambda pair: pair[0], reverse=True)
        Becomes: dated.sort(key=lambda pair: pair[0])
        Killed by: src/uclone_x/story/context.py :: bundle["earlier_changes"] = earlier_changes
        Becomes: pass
        Killed by: src/uclone_x/story/context.py :: where = f"since {change['at']}" + (f", {change['note']}" if change.get("note") else "")
        Becomes: where = f"since {change['at']}"
        Killed by: src/uclone_x/story/context.py :: return f"{item.entry.name} ('{item.entry.id}', {item.kind}): " + "; ".join(reversed(parts))
        Becomes: return f"{item.entry.name} ('{item.entry.id}', {item.kind}): " + "; ".join(parts)
        """
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        await _ok(
            StoryOutlineTool(),
            ctx,
            action="set_scene",
            chapter_id="ch01",
            title="Dawn",
            summary="Mara crosses at the toll.",
            characters=["mara"],
        )
        _codex(
            tmp_path,
            story_id,
            "characters",
            {
                "id": "ines",
                "name": "Ines",
                "state": {"status": "alive", "holds": "the oar"},
                "progressions": [
                    {"at": "ch01.s01", "set": {"status": "dead"}, "note": "drowned"},
                    {"at": "ch01.s02", "set": {"holds": None}, "note": "the oar washed away"},
                ],
            },
        )
        # Listed in the scene: its full entry carries the change, so no line.
        _codex(
            tmp_path,
            story_id,
            "characters",
            {
                "id": "mara",
                "name": "Mara",
                "progressions": [{"at": "ch01.s01", "set": {"mood": "grim"}}],
            },
        )
        # Only mentioned ("toll"): in the bundle in full, and one line as well.
        _codex(
            tmp_path,
            story_id,
            "places",
            {
                "id": "ferry",
                "name": "The Ferry",
                "aliases": ["toll"],
                "progressions": [{"at": "ch01.s01", "set": {"state": "burned"}}],
            },
        )
        # Changed at this scene: the scene starts before it.
        _codex(
            tmp_path,
            story_id,
            "threads",
            {
                "id": "debt",
                "name": "The debt",
                "progressions": [{"at": "ch01.s03", "set": {"paid": True}}],
            },
        )

        out = await _ok(StoryContextTool(), ctx, action="for_scene", scene_id="ch01.s03")

        assert {e["id"] for e in out["codex"]} == {"mara", "ferry"}
        # Latest first; within a line, in story order.
        assert out["earlier_changes"] == [
            "Ines ('ines', characters): status: dead (since ch01.s01, drowned); "
            "holds: (cleared) (since ch01.s02, the oar washed away)",
            "The Ferry ('ferry', places): state: burned (since ch01.s01)",
        ]
        assert "dead here" in out["manifest"]["earlier_changes_note"]
        assert "earlier_changes_not_shown" not in out["manifest"]

        # A scene before every change is given none.
        first = await _ok(StoryContextTool(), ctx, action="for_scene", scene_id="ch01.s01")
        assert "earlier_changes" not in first
        assert "earlier_changes_note" not in first["manifest"]

    def test_a_listed_entry_over_the_budget_still_gets_its_line(self) -> None:
        """A scene that lists more entries than `MAX_ENTRIES` drops the rest from the bundle;
        a death in a dropped one still reaches the Writer as one line (#1613 item 8).

        Killed by: src/uclone_x/story/context.py :: listed_in_full = {key for key in listed if key in snapshots}
        Becomes: listed_in_full = listed
        """
        from uclone_x.story.context import scene_context

        outline = Outline.model_validate(
            {
                "chapters": [
                    {
                        "id": "c",
                        "title": "C",
                        "scenes": [
                            {"id": "a", "title": "A"},
                            {
                                "id": "b",
                                "title": "B",
                                "characters": [f"p{i:02d}" for i in range(MAX_ENTRIES + 1)],
                            },
                        ],
                    }
                ]
            }
        )
        items = tuple(
            CodexItem(
                "characters",
                CharacterEntry.model_validate(
                    {
                        "id": f"p{i:02d}",
                        "name": f"P{i}",
                        "progressions": [{"at": "a", "set": {"status": "dead"}}],
                    }
                ),
            )
            for i in range(MAX_ENTRIES + 1)
        )
        out = scene_context(outline, "b", CodexIndex(items), previous_text=None, story={})
        dropped = f"p{MAX_ENTRIES:02d}"
        assert dropped not in {e["id"] for e in out["codex"]}
        assert [e["id"] for e in out["manifest"]["left_out"]] == [dropped]
        # Only the dropped entry: the listed ones in the bundle carry the change in full.
        assert out["earlier_changes"] == [
            f"P{MAX_ENTRIES} ('{dropped}', characters): status: dead (since a)"
        ]

    def test_earlier_changes_over_the_budget_are_counted(self) -> None:
        """Killed by: src/uclone_x/story/context.py :: earlier_changes = [line for _, line in dated[:MAX_EARLIER_CHANGES]]
        Becomes: earlier_changes = [line for _, line in dated]
        """
        from uclone_x.story.context import MAX_EARLIER_CHANGES, scene_context

        outline = Outline.model_validate(
            {
                "chapters": [
                    {
                        "id": "c",
                        "title": "C",
                        "scenes": [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}],
                    }
                ]
            }
        )
        items = tuple(
            CodexItem(
                "characters",
                CharacterEntry.model_validate(
                    {
                        "id": f"p{i:02d}",
                        "name": f"P{i}",
                        "progressions": [{"at": "a", "set": {"status": "dead"}}],
                    }
                ),
            )
            for i in range(MAX_EARLIER_CHANGES + 2)
        )
        out = scene_context(outline, "b", CodexIndex(items), previous_text=None, story={})
        assert len(out["earlier_changes"]) == MAX_EARLIER_CHANGES
        assert out["manifest"]["earlier_changes_not_shown"] == 2


# --------------------------------------------------------------------------------------
# The record of conversations
# --------------------------------------------------------------------------------------


class TestSessionsAreRecorded:
    async def test_opening_and_closing_are_written_to_sessions(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/tool.py :: result.update(StoryLibraryTool._session_closed(library, story_id, conversation))
        Becomes: result.update({})
        """
        story_id = await _new_story(tmp_path)
        await _ok(StoryLibraryTool(), _ctx(tmp_path, story=story_id), action="close")

        sessions = yaml.safe_load(
            (tmp_path / "stories" / story_id / "sessions.yaml").read_text(encoding="utf-8")
        )
        [entry] = sessions["sessions"]
        assert entry["room_id"] == ROOM_A
        assert entry["opened_at"] and entry["closed_at"]

    async def test_a_take_over_closes_the_entry_of_the_conversation_it_was_taken_from(
        self, tmp_path: Path
    ) -> None:
        """Room A can no longer write the story, so it cannot close its own entry (#1576).

        Killed by: src/uclone_x/story/tool.py :: result.update(self._session_opened(library, story_id, conversation, taken_from=previous))
        Becomes: result.update(self._session_opened(library, story_id, conversation))
        """
        story_id = await _new_story(tmp_path)
        taken = await _ok(
            StoryLibraryTool(),
            _ctx(tmp_path, conversation=ROOM_B),
            action="take_over",
            story_id=story_id,
        )
        assert taken["taken_from_another_conversation"] is True
        assert "session_not_recorded" not in taken

        sessions = yaml.safe_load(
            (tmp_path / "stories" / story_id / "sessions.yaml").read_text(encoding="utf-8")
        )
        entries = {e["room_id"]: e for e in sessions["sessions"]}
        assert entries[ROOM_A]["closed_at"]
        assert entries[ROOM_B]["closed_at"] is None


class TestTheLibraryOutsideAConversation:
    async def test_list_works_and_the_other_actions_are_refused_in_words(
        self, tmp_path: Path
    ) -> None:
        """A direct call of 'list' works outside a room; the tool is not offered there (#1576).

        Killed by: src/uclone_x/story/tool.py :: if params.action == "list":
        Becomes: if False:
        """
        story_id = await _new_story(tmp_path)
        outside = _ctx(tmp_path, conversation=None)

        listed = await _ok(StoryLibraryTool(), outside, action="list")
        assert listed["stories"] == [
            {
                "story_id": story_id,
                "title": "The Salt Road",
                "being_written_by": "another conversation",
            }
        ]
        refused = await _call(StoryLibraryTool(), outside, action="create", title="Another")
        assert refused.error == (
            "Stories are opened inside a conversation, and this call is not part of one, "
            "so nothing was done."
        )


# --------------------------------------------------------------------------------------
# character_sheet over an open story (1b)
# --------------------------------------------------------------------------------------


class TestCharacterSheetOverAStory:
    async def test_save_proposes_the_change_instead_of_writing_a_sheet(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/character.py :: return _propose_visual(params, context)
        Becomes: raise PlainRefusalError("no")
        """
        story_id = await _new_story(tmp_path)
        _codex(tmp_path, story_id, "characters", {"id": "mara", "name": "Mara"})
        before = (tmp_path / "stories" / story_id / "codex/characters/mara.yaml").read_text()
        out = await _ok(
            CharacterSheetTool(),
            _ctx(tmp_path, story=story_id),
            action="save",
            character_id="mara",
            danbooru_tags="red hair, scar",
        )
        assert out["status"] == "proposed"
        assert out["proposal"]["change"] == {"visual": {"tags": ["red hair", "scar"]}}
        assert (tmp_path / "stories" / story_id / "proposals" / f"{out['proposed']}.yaml").is_file()
        # Nothing changed until a person applies it, and no workspace sheet was written.
        assert (
            tmp_path / "stories" / story_id / "codex/characters/mara.yaml"
        ).read_text() == before
        assert not (tmp_path / "characters").exists()

    async def test_get_and_compose_read_the_codex_visual_block(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/character.py :: "danbooru_tags": ", ".join(visual.tags) if visual else "",
        Becomes: "danbooru_tags": "",
        """
        story_id = await _new_story(tmp_path)
        _codex(
            tmp_path,
            story_id,
            "characters",
            {
                "id": "mara",
                "name": "Mara",
                "visual": {"tags": ["red hair", "scar"], "gender": "female", "base_seed": 7},
            },
        )
        ctx = _ctx(tmp_path, story=story_id)
        got = await _ok(CharacterSheetTool(), ctx, action="get", character_id="mara")
        assert got["character"]["danbooru_tags"] == "red hair, scar"
        assert got["character"]["gender"] == "female"

        composed = await _ok(CharacterSheetTool(), ctx, action="compose", character_ids=["mara"])
        assert composed["composed_danbooru_prompt"].startswith("1girl, solo, red hair, scar")
        assert composed["seeds"] == [7]

    async def test_a_workspace_sheet_is_shown_read_only_with_a_migration_hint(
        self, tmp_path: Path
    ) -> None:
        # The sheet saved before any story was open, as the tool does today.
        await _ok(
            CharacterSheetTool(),
            _ctx(tmp_path),
            action="save",
            character_id="kaito",
            danbooru_tags="black hair",
        )
        story_id = await _new_story(tmp_path)
        ctx = _ctx(tmp_path, story=story_id)

        got = await _ok(CharacterSheetTool(), ctx, action="get", character_id="kaito")
        assert got["status"] == "not_in_story"
        assert got["workspace_sheet_read_only"]["danbooru_tags"] == "black hair"
        assert "codex/characters/<id>.yaml" in got["migration_hint"]

        listed = await _ok(CharacterSheetTool(), ctx, action="list")
        assert listed["characters"] == []
        assert listed["workspace_sheets_read_only"] == ["kaito"]

        result = await _call(CharacterSheetTool(), ctx, action="compose", character_ids=["kaito"])
        assert result.error is not None
        assert result.error.startswith("The story's codex has no character 'kaito'")
        assert "codex/characters/<id>.yaml" in result.error


# --------------------------------------------------------------------------------------
# Outside a conversation, the story tools are not offered
# --------------------------------------------------------------------------------------


class _RecordingLLM(MockLLMConnector):
    def __init__(self) -> None:
        super().__init__(default_response="ok")
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


class TestStoryToolsAreOfferedOnlyInARoom:
    async def test_a_turn_outside_a_room_is_not_sent_the_story_tools(self, tmp_path: Path) -> None:
        """Their schemas would cost every request room in the window for calls that are
        refused there or, for `story_library`'s 'list', of no use there; in a room they are
        offered as before.

        Killed by: src/uclone_x/agent/tool_invoker.py :: if self._scope.room_id() is None and tool_needs_room(t):
        Becomes: if False:
        """
        story_tools = {
            "story_library",
            "story_outline",
            "story_codex",
            "story_manuscript",
            "story_context",
        }
        llm = _RecordingLLM()
        agent = BaseAgent(
            config=AgentConfig(
                agent_id="writer",
                name="Writer",
                workspace_dir=tmp_path,
                llm_config=AgentLLMConfig(model_name="mock-model"),
            ),
            llm=llm,
            tools=create_default_registry(workspace_root=tmp_path, enable_mcp=False),
        )

        await agent.execute_turn("hello")
        await agent.execute_turn("hello", room_id=ROOM_A)

        outside, inside = ({t.name for t in r.tools} for r in llm.requests)
        assert outside & story_tools == set()
        assert story_tools <= inside
        assert "character_sheet" in outside  # not a story tool: it works with no story open
