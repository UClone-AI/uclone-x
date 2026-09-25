"""Stories as workspace artifacts, their single writer, and the `story_library` tool (#1555).

What these pin, in order of what it would cost to get wrong:

* **One writer.** A second conversation opens a story read-only, and a conversation that
  was taken over cannot write over its successor.
* **No lost edit.** A file changed after it was read -- a person editing it by hand -- is
  not overwritten.
* **No path from the model.** A story id that names somewhere outside the library is
  refused, with a reason.
* **Plain refusals.** A story tool that refuses says so in words, not as a class name.
* **The conversation's story reaches every tool call**, and a story opened in a turn is the
  one the rest of that turn works on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig, ToolExecutionRecord
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse, ToolCallRequest
from uclone_x.story import OPEN_STORY_KEY, story_after
from uclone_x.story.library import (
    NoConversationError,
    NoOpenStoryError,
    StoryChangedError,
    StoryLibrary,
    StoryReadOnlyError,
    UnknownStoryError,
)
from uclone_x.story.tool import StoryLibraryTool
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import NoIsolation, ToolContext, ToolResult, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry

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


async def _call(tool: StoryLibraryTool, ctx: ToolContext, **args: Any) -> ToolResult:
    return await tool.execute(args, ctx)


# --------------------------------------------------------------------------------------
# The library: where a story is
# --------------------------------------------------------------------------------------


class TestWhereAStoryIs:
    def test_a_created_story_lives_under_stories(self, tmp_path: Path) -> None:
        library = StoryLibrary(tmp_path)
        record = library.create("The Salt Road", ROOM_A)

        assert record.story_id.startswith("the-salt-road-")
        assert library.root(record.story_id) == (tmp_path / "stories" / record.story_id).resolve()
        assert [s.story_id for s in library.list().stories] == [record.story_id]

    def test_an_unknown_id_is_refused_with_a_reason(self, tmp_path: Path) -> None:
        library = StoryLibrary(tmp_path)
        with pytest.raises(UnknownStoryError, match="no story called 'nothing-here'"):
            library.root("nothing-here")

    @pytest.mark.parametrize("bad", ["../escape", "a/b", "/etc", "UPPER", ""])
    def test_an_id_that_is_not_a_name_is_refused(self, tmp_path: Path, bad: str) -> None:
        with pytest.raises(UnknownStoryError, match="not a story name"):
            StoryLibrary(tmp_path).root(bad)

    def test_a_story_folder_linked_outside_the_library_is_refused(self, tmp_path: Path) -> None:
        """A well-formed id can still name a symlink that leaves the library.

        Killed by: src/uclone_x/story/library.py :: if folder.parent != library:
        Becomes: if False:
        """
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "story.yaml").write_text(
            "story_id: linked\ntitle: T\ncreated_at: '2026-09-24T00:00:00+00:00'\n",
            encoding="utf-8",
        )
        (tmp_path / "stories").mkdir()
        (tmp_path / "stories" / "linked").symlink_to(outside, target_is_directory=True)

        with pytest.raises(UnknownStoryError, match="outside the story library"):
            StoryLibrary(tmp_path).root("linked")

    def test_an_unreadable_story_is_listed_with_its_reason(self, tmp_path: Path) -> None:
        library = StoryLibrary(tmp_path)
        good = library.create("Good", ROOM_A)
        broken = tmp_path / "stories" / "broken-0000"
        broken.mkdir()
        (broken / "story.yaml").write_text("title: [unclosed\n", encoding="utf-8")

        listing = library.list()
        assert [s.story_id for s in listing.stories] == [good.story_id]
        assert [(u.story_id, "YAML" in u.reason) for u in listing.unreadable] == [
            ("broken-0000", True)
        ]


# --------------------------------------------------------------------------------------
# The lease: one writer at a time
# --------------------------------------------------------------------------------------


class TestOneWriter:
    def test_a_second_conversation_opens_the_story_read_only(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/library.py :: return StoryOpening(record, writable=False, acquired=False)
        Becomes: return StoryOpening(record, writable=True, acquired=False)
        """
        library = StoryLibrary(tmp_path)
        story = library.create("Tide", ROOM_A).story_id

        assert library.open(story, ROOM_A).writable is True
        later = library.open(story, ROOM_B)
        assert later.writable is False
        with pytest.raises(StoryReadOnlyError):
            library.write_file(
                story, "chapter-01.md", "B", conversation_id=ROOM_B, expected_digest=None
            )

    def test_after_take_over_the_earlier_conversation_is_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/library.py :: if lease.holder != conversation_id:
        Becomes: if False:
        """
        library = StoryLibrary(tmp_path)
        story = library.create("Tide", ROOM_A).story_id
        rev = library.write_file(
            story, "chapter-01.md", "A1", conversation_id=ROOM_A, expected_digest=None
        )

        _, previous = library.take_over(story, ROOM_B)
        assert previous == ROOM_A

        with pytest.raises(StoryReadOnlyError) as refused:
            library.write_file(
                story, "chapter-01.md", "A2", conversation_id=ROOM_A, expected_digest=rev
            )
        message = str(refused.value)
        assert "Another conversation" in message and "nothing was written" in message
        assert library.read_file(story, "chapter-01.md").text == "A1"
        # The new holder writes on the same digest.
        library.write_file(
            story, "chapter-01.md", "B1", conversation_id=ROOM_B, expected_digest=rev
        )

    def test_a_conversation_that_was_taken_over_cannot_release_the_lease(
        self, tmp_path: Path
    ) -> None:
        library = StoryLibrary(tmp_path)
        story = library.create("Tide", ROOM_A).story_id
        library.take_over(story, ROOM_B)

        assert library.release(story, ROOM_A) is False
        lease = library.load(story).lease
        assert lease is not None and lease.holder == ROOM_B


class TestTheWriterIsTheCallsConversation:
    """A tool's write is checked against the conversation on its `ToolContext`.

    The story id alone says what is written, not who writes it. An agent asked from
    another conversation -- a later A2A callee handed the story id and its caller's
    conversation id -- is checked against the same lease as the caller's own seats.
    """

    def test_an_agent_acting_for_the_holder_writes(self, tmp_path: Path) -> None:
        library = StoryLibrary(tmp_path)
        story = library.create("Tide", ROOM_A).story_id
        callee = ToolContext(
            agent_id="critic",
            session_id="sess_a2a__critic",
            workspace_root=tmp_path,
            room_id=ROOM_A,
            story_id=story,
        )

        library.write_for(callee, "notes.md", "n", expected_digest=None)

        assert library.read_file(story, "notes.md").text == "n"

    def test_an_agent_acting_for_another_conversation_is_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/library.py :: self._require_writer(folder, story_id, conversation_id)
        Becomes: pass
        """
        library = StoryLibrary(tmp_path)
        story = library.create("Tide", ROOM_A).story_id

        with pytest.raises(StoryReadOnlyError, match="Another conversation"):
            library.write_for(
                _ctx(tmp_path, conversation=ROOM_B, story=story),
                "notes.md",
                "n",
                expected_digest=None,
            )
        assert not (library.root(story) / "notes.md").exists()

    def test_a_write_with_no_open_story_or_conversation_is_refused(self, tmp_path: Path) -> None:
        library = StoryLibrary(tmp_path)
        with pytest.raises(NoOpenStoryError, match="No story is open"):
            library.write_for(_ctx(tmp_path, story=None), "a.md", "x", expected_digest=None)
        with pytest.raises(NoConversationError, match="not part of one"):
            library.write_for(
                _ctx(tmp_path, conversation=None, story="tide-0001"),
                "a.md",
                "x",
                expected_digest=None,
            )


class TestNoLostEdit:
    def test_a_file_edited_outside_the_conversation_is_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/library.py :: if current != expected_digest:
        Becomes: if False:
        """
        library = StoryLibrary(tmp_path)
        story = library.create("Tide", ROOM_A).story_id
        library.write_file(
            story, "chapter-01.md", "draft", conversation_id=ROOM_A, expected_digest=None
        )
        read = library.read_file(story, "chapter-01.md")
        (library.root(story) / "chapter-01.md").write_text("edited by hand", encoding="utf-8")

        with pytest.raises(StoryChangedError, match="changed after it was read"):
            library.write_file(
                story,
                "chapter-01.md",
                "model's version",
                conversation_id=ROOM_A,
                expected_digest=read.digest,
            )
        assert library.read_file(story, "chapter-01.md").text == "edited by hand"

    def test_a_new_file_does_not_replace_one_that_appeared(self, tmp_path: Path) -> None:
        library = StoryLibrary(tmp_path)
        story = library.create("Tide", ROOM_A).story_id
        (library.root(story) / "notes.md").write_text("mine", encoding="utf-8")

        with pytest.raises(StoryChangedError, match="already exists"):
            library.write_file(story, "notes.md", "x", conversation_id=ROOM_A, expected_digest=None)

    @pytest.mark.parametrize("bad", ["../other/x.md", "/tmp/x.md", "story.yaml"])
    def test_a_path_outside_the_story_or_its_record_is_refused(
        self, tmp_path: Path, bad: str
    ) -> None:
        library = StoryLibrary(tmp_path)
        story = library.create("Tide", ROOM_A).story_id
        with pytest.raises(Exception, match="story") as refused:
            library.write_file(story, bad, "x", conversation_id=ROOM_A, expected_digest=None)
        assert "Error" not in str(refused.value)


# --------------------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------------------


class TestStoryLibraryTool:
    @pytest.mark.asyncio
    async def test_create_opens_the_new_story_and_names_its_record(self, tmp_path: Path) -> None:
        result = await _call(StoryLibraryTool(), _ctx(tmp_path), action="create", title="Tide")

        assert result.success, result.error
        output: dict[str, Any] = dict(result.output)  # type: ignore[arg-type]
        story = output[OPEN_STORY_KEY]
        assert output["writable"] is True
        assert output["path"] == f"stories/{story}/story.yaml"
        assert (tmp_path / "stories" / story / "story.yaml").is_file()

    @pytest.mark.asyncio
    async def test_open_in_a_second_conversation_is_read_only_and_says_how_to_continue(
        self, tmp_path: Path
    ) -> None:
        tool = StoryLibraryTool()
        created = await _call(tool, _ctx(tmp_path), action="create", title="Tide")
        story = dict(created.output)[OPEN_STORY_KEY]  # type: ignore[arg-type]

        opened = await _call(
            tool, _ctx(tmp_path, conversation=ROOM_B), action="open", story_id=story
        )
        output: dict[str, Any] = dict(opened.output)  # type: ignore[arg-type]
        assert output["writable"] is False
        assert "take_over" in output["note"]
        assert "path" not in output  # nothing was written

        taken = await _call(
            tool, _ctx(tmp_path, conversation=ROOM_B), action="take_over", story_id=story
        )
        assert dict(taken.output)["taken_from_another_conversation"] is True  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_a_story_call_with_no_open_story_fails_with_a_plain_reason(
        self, tmp_path: Path
    ) -> None:
        """The refusal reaches the conversation as words, not as `NoOpenStoryError: ...`.

        Killed by: src/uclone_x/tools/base.py :: except PlainRefusalError as e:
        Becomes: except ZeroDivisionError as e:
        """
        result = await _call(StoryLibraryTool(), _ctx(tmp_path, story=None), action="close")

        assert result.success is False
        assert result.error == "No story is open in this conversation, so none was closed."

    @pytest.mark.asyncio
    async def test_outside_a_conversation_nothing_is_opened(self, tmp_path: Path) -> None:
        result = await _call(
            StoryLibraryTool(), _ctx(tmp_path, conversation=None), action="create", title="T"
        )
        assert result.success is False
        assert result.error is not None and "not part of one" in result.error
        assert not (tmp_path / "stories").exists()

    @pytest.mark.asyncio
    async def test_open_of_an_unknown_story_is_refused_with_a_reason(self, tmp_path: Path) -> None:
        result = await _call(
            StoryLibraryTool(), _ctx(tmp_path), action="open", story_id="../../etc"
        )
        assert result.success is False
        assert result.error is not None and result.error.startswith(
            "'../../etc' is not a story name"
        )

    @pytest.mark.asyncio
    async def test_opening_another_story_gives_back_the_first(self, tmp_path: Path) -> None:
        tool = StoryLibraryTool()
        library = StoryLibrary(tmp_path)
        first = library.create("One", ROOM_A).story_id
        second = library.create("Two", ROOM_B).story_id
        library.release(second, ROOM_B)

        opened = await _call(tool, _ctx(tmp_path, story=first), action="open", story_id=second)

        assert dict(opened.output)["released"] == first  # type: ignore[arg-type]
        assert library.load(first).lease is None

    @pytest.mark.asyncio
    async def test_close_releases_the_lease_and_leaves_the_files(self, tmp_path: Path) -> None:
        library = StoryLibrary(tmp_path)
        story = library.create("Tide", ROOM_A).story_id

        closed = await _call(StoryLibraryTool(), _ctx(tmp_path, story=story), action="close")

        assert dict(closed.output)[OPEN_STORY_KEY] is None  # type: ignore[arg-type]
        assert library.load(story).lease is None
        assert library.root(story).is_dir()


# --------------------------------------------------------------------------------------
# The story reaches the agent's tool calls
# --------------------------------------------------------------------------------------


class _ProbeParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _ProbeTool(BaseTool[_ProbeParams]):
    """Records the story and conversation each call was given."""

    name = "probe"
    writes_files: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__(name="probe", description="probe", params_type=_ProbeParams)
        self.seen: list[tuple[str | None, str | None]] = []

    async def run(self, params: _ProbeParams, context: ToolContext) -> dict[str, Any]:
        self.seen.append((context.room_id, context.story_id))
        return {"ok": True}


class _ImpostorTool(BaseTool[_ProbeParams]):
    """Answers with the story key without declaring `opens_story`."""

    name = "impostor"
    writes_files: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__(name="impostor", description="x", params_type=_ProbeParams)

    async def run(self, params: _ProbeParams, context: ToolContext) -> dict[str, Any]:
        return {OPEN_STORY_KEY: "somewhere-else"}


class _StepLLM(MockLLMConnector):
    """Asks for one step's tool calls at a time, then answers."""

    def __init__(self, steps: list[list[ToolCallRequest]]) -> None:
        super().__init__(default_response="done")
        self._steps = steps

    async def generate(self, request: LLMRequest) -> ModelResponse:
        spent = {m.tool_call_id for m in request.messages if m.tool_call_id}
        pending = [s for s in self._steps if not {c.id for c in s} <= spent]
        self._tool_calls = list(pending[0]) if pending else []
        return await super().generate(request)


def _writer(workspace: Path, llm: MockLLMConnector, *tools: BaseTool[Any]) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id="writer",
            name="Writer",
            workspace_dir=workspace,
            allowed_tools=tuple(t.name for t in tools),
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=llm,
        tools=ToolRegistry(tools=list(tools)),
    )


class TestTheStoryReachesToolCalls:
    @pytest.mark.asyncio
    async def test_the_turns_conversation_and_story_reach_the_tool_context(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/agent/base.py :: story_id=self._turn_story_id,
        Becomes: story_id=None,
        """
        probe = _ProbeTool()
        llm = _StepLLM([[ToolCallRequest(id="c1", name="probe", arguments={})]])
        agent = _writer(tmp_path, llm, probe)

        await agent.execute_turn("go", room_id=ROOM_A, story_id="tide-0001")

        assert probe.seen == [(ROOM_A, "tide-0001")]

    @pytest.mark.asyncio
    async def test_a_story_created_in_a_turn_is_the_one_its_later_steps_see(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/agent/base.py :: self._turn_story_id = story_after((rec,), self._turn_story_id)
        Becomes: pass
        """
        probe = _ProbeTool()
        llm = _StepLLM(
            [
                [
                    ToolCallRequest(
                        id="c1",
                        name="story_library",
                        arguments={"action": "create", "title": "Tide"},
                    )
                ],
                [ToolCallRequest(id="c2", name="probe", arguments={})],
            ]
        )
        agent = _writer(tmp_path, llm, StoryLibraryTool(), probe)

        result = await agent.execute_turn("start a story", room_id=ROOM_A)

        created = [s.story_id for s in StoryLibrary(tmp_path).list().stories]
        assert len(created) == 1
        assert probe.seen == [(ROOM_A, created[0])]
        assert [r.opens_story for r in result.tool_executions] == [True, False]

    def test_only_a_declaring_tool_that_succeeded_moves_the_story(self) -> None:
        """Killed by: src/uclone_x/story/__init__.py :: if not record.opens_story or record.status
        Becomes: if record.status
        """

        def rec(opens: bool, ok: bool, story: str | None) -> ToolExecutionRecord:
            return ToolExecutionRecord(
                tool_name="t",
                output={OPEN_STORY_KEY: story},
                status=ToolResultStatus.SUCCESS if ok else ToolResultStatus.ERROR,
                opens_story=opens,
            )

        assert story_after([rec(False, True, "x")], "a") == "a"
        assert story_after([rec(True, False, "x")], "a") == "a"
        assert story_after([rec(True, True, "x"), rec(True, True, None)], "a") is None
        assert story_after([rec(True, True, "x")], None) == "x"

    @pytest.mark.asyncio
    async def test_a_tool_that_does_not_declare_it_cannot_move_the_story(
        self, tmp_path: Path
    ) -> None:
        probe = _ProbeTool()
        llm = _StepLLM(
            [
                [ToolCallRequest(id="c1", name="impostor", arguments={})],
                [ToolCallRequest(id="c2", name="probe", arguments={})],
            ]
        )
        agent = _writer(tmp_path, llm, _ImpostorTool(), probe)

        await agent.execute_turn("go", room_id=ROOM_A, story_id="tide-0001")

        assert probe.seen == [(ROOM_A, "tide-0001")]
