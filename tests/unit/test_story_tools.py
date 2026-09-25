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
        assert "progressions_not_applied" not in manifest  # 'debt' is not in the bundle

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

    async def test_a_progression_on_an_included_entry_is_named_as_not_applied(
        self, tmp_path: Path
    ) -> None:
        story_id = await _new_story(tmp_path)
        ctx = await _outlined(tmp_path, story_id)
        _codex(
            tmp_path,
            story_id,
            "characters",
            {
                "id": "mara",
                "name": "Mara",
                "state": {"arm": "whole"},
                "progressions": [{"at": "ch01.s02", "set": {"arm": "broken"}}],
            },
        )
        out = await _ok(StoryContextTool(), ctx, action="for_scene", scene_id="ch01.s02")
        assert out["manifest"]["progressions_not_applied"] == [
            {"id": "mara", "kind": "characters", "at_scenes": ["ch01.s02"]}
        ]
        assert [e["state"] for e in out["codex"] if e["id"] == "mara"] == [{"arm": "whole"}]


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


# --------------------------------------------------------------------------------------
# character_sheet over an open story (1b)
# --------------------------------------------------------------------------------------


class TestCharacterSheetOverAStory:
    async def test_save_is_refused_while_a_story_is_open(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/tools/builtin/character.py :: if context.story_id is not None:
        Becomes: if False:
        """
        story_id = await _new_story(tmp_path)
        result = await _call(
            CharacterSheetTool(),
            _ctx(tmp_path, story=story_id),
            action="save",
            character_id="mara",
            danbooru_tags="red hair",
        )
        assert not result.success
        assert result.error is not None
        assert result.error.startswith("A story is open, so character sheets are not saved here")
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
        """Their schemas would cost every request room in the window for calls that can only
        be refused; in a room they are offered as before.

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
