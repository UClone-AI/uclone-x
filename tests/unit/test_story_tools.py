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

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse
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
from uclone_x.tools.builtin.character import CharacterSheetTool
from uclone_x.tools.models import NoIsolation, ToolContext, ToolResult
from uclone_x.tools.registry import create_default_registry
from uclone_x.ui.app import AgentSessionManager, create_ui_app

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
        first = "Mara stepped onto the ferry. " * 80 + "The river went black behind her."
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

        Killed by: src/uclone_x/story/work.py :: for relative in self._library.folders_in(self._story_id, folder):
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

        Killed by: src/uclone_x/story/work.py :: if base.suffix in (".yaml", ".yml"):
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
        assert {u["file"]: u["reason"] for u in out["unreadable_files"]} == {
            "codex/characters/mara.yaml~": f"'codex/characters/mara.yaml~' was not read: "
            f"{ending} If it is an entry, give it a name ending in '.yaml' that no other "
            "file in 'codex/characters/' has, even with different capitals.",
            "codex/places/dock.yaml.bak": f"'codex/places/dock.yaml.bak' was not read: "
            f"{ending} If it is an entry, rename it to 'dock.yaml'.",
            "codex/items/Old Map.txt": f"'codex/items/Old Map.txt' was not read: {ending} "
            "If it is an entry, give it a name ending in '.yaml' that no other file in "
            "'codex/items/' has, even with different capitals.",
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
        no_name = "If it is an entry, give it a name ending in '.yaml' that no other file in "
        assert reasons["codex/characters/mara.yml"] == (
            f"'codex/characters/mara.yml' was not read: {ending} {no_name}"
            "'codex/characters/' has, even with different capitals."
        )
        for twin in ("codex/places/bo.yml", "codex/places/bo.yaml~"):
            assert reasons[twin] == (
                f"'{twin}' was not read: {ending} {no_name}'codex/places/' has, even with "
                "different capitals."
            )
        # A file differing only in capitals is not in its own way.
        assert reasons["codex/items/map.YAML"] == (
            f"'codex/items/map.YAML' was not read: {ending} If it is an entry, rename it to "
            "'map.yaml'."
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

    async def test_writing_over_text_without_its_digest_is_refused(self, tmp_path: Path) -> None:
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        tool = StoryManuscriptTool()
        await _ok(tool, ctx, action="write", scene_id="ch01.s01", text="First draft.")

        result = await _call(tool, ctx, action="write", scene_id="ch01.s01", text="Oops.")

        assert not result.success
        assert result.error is not None and "nothing was written" in result.error
        scene = tmp_path / "stories" / story_id / "manuscript" / "ch01.s01.md"
        assert scene.read_text(encoding="utf-8") == "First draft."

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
        """Killed by: src/uclone_x/tools/builtin/character.py :: return _propose_visual(params, context)
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
        """Killed by: src/uclone_x/tools/builtin/character.py :: "danbooru_tags": ", ".join(visual.tags) if visual else "",
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

        Killed by: src/uclone_x/agent/base.py :: if self._turn_room_id is None and tool_needs_room(t):
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


class TestTheDashboardListsWhatTheAgentHolds:
    async def test_an_agent_not_yet_in_a_room_shows_the_story_tools_as_needing_one(
        self, tmp_path: Path
    ) -> None:
        """What `/api/agents` lists does not depend on the room of the last turn (#1576).

        Killed by: src/uclone_x/ui/app.py :: held = ag.held_tools()
        Becomes: held = ag.available_tools()

        Killed by: src/uclone_x/ui/app.py :: capabilities_needing_room = [tool.name for tool in held if tool_needs_room(tool)]
        Becomes: capabilities_needing_room = [tool.name for tool in held]
        """
        registry = create_default_registry(workspace_root=tmp_path, enable_mcp=False)
        session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", tools=registry)
        agent = BaseAgent(
            config=AgentConfig(
                agent_id="writer",
                name="Writer",
                workspace_dir=tmp_path,
                llm_config=AgentLLMConfig(model_name="mock-model"),
            ),
            llm=MockLLMConnector(default_response="ok"),
            tools=registry,
        )
        agents = session_mgr._agents  # pyright: ignore[reportPrivateUsage]
        agents[f"{agent.agent_id}:{agent.context.session_id}"] = agent
        client = TestClient(
            create_ui_app(
                static_dir=tmp_path / "static",
                storage_dir=tmp_path / "sessions",
                llm=MockLLMConnector(),
                session_manager=session_mgr,
            )
        )
        needing_room = {
            "story_library",
            "story_outline",
            "story_codex",
            "story_manuscript",
            "story_context",
        }

        def row() -> dict[str, Any]:
            response = client.get("/api/agents")
            assert response.status_code == 200, response.text
            [only] = response.json()["agents"]
            [node] = response.json()["topology"]["nodes"]
            assert node["capabilities_needing_room"] == only["capabilities_needing_room"]
            return only

        before = row()
        assert needing_room <= set(before["capabilities"])
        assert needing_room <= set(before["capabilities_needing_room"])
        assert "character_sheet" not in before["capabilities_needing_room"]

        await agent.execute_turn("hello", room_id=ROOM_A)
        await agent.execute_turn("hello")
        assert row()["capabilities"] == before["capabilities"]
