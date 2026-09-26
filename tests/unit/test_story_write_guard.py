"""The general file tools do not write a story (#1583).

A story is written only by the story tools, which check the room's lease, the digest the
writer read and, for a codex change, a person's approval. The review of #1581 had the
Writer's `file_write` rewrite `stories/<id>/codex/characters/vane.yaml` during a turn,
skipping all three. What these pin:

* **Through a real turn**, with the shipped Writer persona and a scripted model, a
  `file_write` into the story is refused with a plain sentence naming the story tools, the
  codex file is unchanged, and a `file_read` of the same file in the same turn succeeds.
* **The library is recognised by what the path is**: another case or `ſ` spelling of the
  folder, a `..` detour, an absolute path, a link into the library, and a library that is
  itself a link to another folder. The spellings are refused even before a story exists.
* **Nothing else is caught**: a folder named `stories` deeper in the workspace, and names
  that only start with it, are written as before.
* **Every general tool that writes a path the model chose** refuses it: `file_edit` and
  both `generate_image` implementations as well as `file_write`.
* **A hardlink is not detected, and does not need to be**: `file_write`, `file_edit`,
  both `generate_image` implementations (single images and batches) and `character_sheet`
  write a new file and give it the name, so a name hardlinked to a story file outside the
  library is detached and the story's copy is left as it was.
* **`character_sheet` writes its fixed `characters/<id>.yaml` the same way** (#1589): a
  `characters/` link out of the workspace or into the library is refused.

* **The shell and local MCP servers** (#1589 items 1 and 2) run on macOS in a jail
  (`sandbox.story_jail`) that refuses any write in the library, however the path is
  named, and refuses renaming the workspace or a folder above it; the rest of the
  workspace is written as before. Where the jail should exist but cannot start, the
  command is refused in plain words instead of run without it.

Not covered: the shell and MCP servers on other systems, MCP servers reached over HTTP, a
local MCP server whose configured `workspace_root` is not the app's workspace, and a jailed
process asking one that is not jailed to write for it.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ToolCallRequest
from uclone_x.sandbox import story_jail
from uclone_x.sandbox.story_jail import JAIL_SETUP_REFUSAL
from uclone_x.story.tool import StoryLibraryTool
from uclone_x.tools.builtin.character import CharacterSheetTool
from uclone_x.tools.builtin.comfy_client import ComfyClient
from uclone_x.tools.builtin.comfy_image_tool import ComfyImageGenTool
from uclone_x.tools.builtin.filesystem import FileEditTool, FileWriteTool
from uclone_x.tools.builtin.image import (
    GenerateImageTool,
    ImageGenerationResult,
    ImagePipelineDispatcher,
)
from uclone_x.tools.builtin.shell import STORY_LIBRARY_SHELL_NOTE, BashRunTool
from uclone_x.tools.client import MCPClient
from uclone_x.tools.models import (
    MCPConnectionConfig,
    MCPTransport,
    NoIsolation,
    ToolContext,
    ToolResult,
    ToolResultStatus,
)
from uclone_x.tools.registry import create_default_registry

ROOM = "room_a"
VANE = "codex/characters/vane.yaml"
VANE_TEXT = yaml.safe_dump({"id": "vane", "name": "Lord Vane"})


def _refusal(path: str) -> str:
    return (
        f"'{path}' is in the story library, so it was not written. Stories are changed with "
        "the story tools (story_manuscript, story_outline, story_codex), which check which "
        "conversation is writing the story and ask a person before a codex change. Reading "
        "the file is still allowed."
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


async def _story(workspace: Path) -> str:
    """A story created by the story tools, with Vane in its codex as a person wrote him."""
    result = await StoryLibraryTool().execute(
        {"action": "create", "title": "The Salt Road"}, _ctx(workspace)
    )
    assert result.success, result.error
    assert isinstance(result.output, dict)
    story_id = result.output["open_story_id"]
    assert isinstance(story_id, str)
    vane = workspace / "stories" / story_id / VANE
    vane.parent.mkdir(parents=True)
    vane.write_text(VANE_TEXT, encoding="utf-8")
    return story_id


async def _write(workspace: Path, path: str, **extra: Any) -> tuple[bool, str | None]:
    args: dict[str, Any] = {"path": path, "content": "name: Nobody\n", "overwrite": True}
    args.update(extra)
    result = await FileWriteTool().execute(args, _ctx(workspace))
    return result.success, result.error


class TestTheWriterCannotRewriteTheCodexInATurn:
    async def test_file_write_is_refused_and_file_read_still_answers(self, tmp_path: Path) -> None:
        """The reproduction from the review of #1581, through `execute_turn`.

        Killed by: src/uclone_x/tools/base.py :: if in_story_library(resolved, workspace_root):
        Becomes: if False:
        Killed by: src/uclone_x/tools/builtin/filesystem.py :: safe_path = self.resolve_write_path(params.path, workspace)
        Becomes: safe_path = self.resolve_safe_path(params.path, workspace)
        """
        story_id = await _story(tmp_path)
        target = f"stories/{story_id}/{VANE}"
        llm = MockLLMConnector(
            responses=["", "Done."],
            tool_calls=[
                ToolCallRequest(
                    id="c1",
                    name="file_write",
                    arguments={
                        "path": target,
                        "content": "name: Vane\nstatus: dead\n",
                        "overwrite": True,
                    },
                ),
                ToolCallRequest(id="c2", name="file_read", arguments={"path": target}),
            ],
        )
        agent = BaseAgent(
            config=AgentConfig(
                agent_id="writer",
                name="writer",
                persona="writer",
                workspace_dir=tmp_path,
                llm_config=AgentLLMConfig(model_name="mock-model"),
            ),
            llm=llm,
            tools=create_default_registry(workspace_root=tmp_path, enable_mcp=False),
        )
        assert "file_write" in agent.config.allowed_tools  # the shipped persona grants it

        result = await agent.execute_turn(
            "Vane dies in chapter one; update his codex entry.", room_id=ROOM, story_id=story_id
        )

        by_tool = {r.tool_name: r for r in result.tool_executions}
        assert by_tool["file_write"].status == ToolResultStatus.ERROR
        assert by_tool["file_write"].error == _refusal(target)
        assert by_tool["file_read"].status == ToolResultStatus.SUCCESS
        assert (tmp_path / "stories" / story_id / VANE).read_text(encoding="utf-8") == VANE_TEXT


class TestTheLibraryIsKnownByWhatThePathIs:
    @pytest.mark.parametrize(
        "spelling",
        ["stories/{id}", "Stories/{id}", "STORIES/{id}", "ſtories/{id}", "notes/../stories/{id}"],
    )
    async def test_a_spelling_of_the_story_folder_is_refused(
        self, tmp_path: Path, spelling: str
    ) -> None:
        story_id = await _story(tmp_path)
        path = f"{spelling.format(id=story_id)}/{VANE}"
        assert await _write(tmp_path, path) == (False, _refusal(path))
        assert (tmp_path / "stories" / story_id / VANE).read_text(encoding="utf-8") == VANE_TEXT

    async def test_an_absolute_path_is_refused(self, tmp_path: Path) -> None:
        story_id = await _story(tmp_path)
        path = str(tmp_path / "stories" / story_id / VANE)
        assert await _write(tmp_path, path) == (False, _refusal(path))

    @pytest.mark.parametrize("spelling", ["Stories", "ſtories", "stories"])
    async def test_nothing_is_planted_where_a_story_will_be_created(
        self, tmp_path: Path, spelling: str
    ) -> None:
        """Before any story exists there is no folder to compare with, so the name decides.

        Killed by: src/uclone_x/tools/base.py :: if parts and parts[0].casefold() == STORIES_DIRNAME:
        Becomes: if parts and parts[0] == STORIES_DIRNAME:
        """
        path = f"{spelling}/salt-road-0000/story.yaml"
        assert await _write(tmp_path, path) == (False, _refusal(path))
        assert list(tmp_path.iterdir()) == []

    async def test_a_link_into_the_story_is_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/sandbox/path_validator.py :: resolved_target = (resolved_root / target_path).resolve()
        Becomes: resolved_target = resolved_root / target_path
        """
        story_id = await _story(tmp_path)
        (tmp_path / "notes").symlink_to(tmp_path / "stories" / story_id / "codex")
        path = "notes/characters/vane.yaml"
        assert await _write(tmp_path, path) == (False, _refusal(path))
        assert (tmp_path / "stories" / story_id / VANE).read_text(encoding="utf-8") == VANE_TEXT

    async def test_a_link_to_a_file_the_story_does_not_have_yet_is_refused(
        self, tmp_path: Path
    ) -> None:
        story_id = await _story(tmp_path)
        (tmp_path / "draft.md").symlink_to(tmp_path / "stories" / story_id / "manuscript/x.md")
        assert await _write(tmp_path, "draft.md") == (False, _refusal("draft.md"))
        assert not (tmp_path / "stories" / story_id / "manuscript").exists()

    async def test_a_library_that_is_a_link_is_refused_under_its_real_name(
        self, tmp_path: Path
    ) -> None:
        """`stories` links to `vault`; the story tools follow it, and so does the refusal.

        Killed by: src/uclone_x/tools/base.py :: if folder.exists() and folder.samefile(library):
        Becomes: if False:
        """
        (tmp_path / "vault").mkdir()
        (tmp_path / "stories").symlink_to(tmp_path / "vault")
        story_id = await _story(tmp_path)
        path = f"vault/{story_id}/{VANE}"
        assert await _write(tmp_path, path) == (False, _refusal(path))
        assert (tmp_path / "vault" / story_id / VANE).read_text(encoding="utf-8") == VANE_TEXT


class TestOnlyTheLibraryIsRefused:
    @pytest.mark.parametrize(
        "path", ["notes/stories/idea.md", "stories-notes/idea.md", "storiesque.md"]
    )
    async def test_other_paths_are_written_as_before(self, tmp_path: Path, path: str) -> None:
        """Killed by: src/uclone_x/tools/base.py :: if parts and parts[0].casefold() == STORIES_DIRNAME:
        Becomes: if any(part.casefold() == STORIES_DIRNAME for part in parts) or parts[0].startswith(STORIES_DIRNAME):
        """
        await _story(tmp_path)
        assert await _write(tmp_path, path) == (True, None)
        assert (tmp_path / path).read_text(encoding="utf-8") == "name: Nobody\n"


class TestEveryGeneralWriterRefuses:
    async def test_file_edit(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/tools/builtin/filesystem.py :: safe_path = self.resolve_write_path(params.path, context.require_workspace())
        Becomes: safe_path = self.resolve_safe_path(params.path, context.require_workspace())
        """
        story_id = await _story(tmp_path)
        path = f"stories/{story_id}/{VANE}"
        result = await FileEditTool().execute(
            {"path": path, "target_content": "Lord Vane", "replacement_content": "Nobody"},
            _ctx(tmp_path),
        )
        assert (result.success, result.error) == (False, _refusal(path))
        assert (tmp_path / "stories" / story_id / VANE).read_text(encoding="utf-8") == VANE_TEXT

    async def test_generate_image(self, tmp_path: Path) -> None:
        """Refused before any image is made.

        Killed by: src/uclone_x/tools/base.py :: if in_story_library(resolved, workspace_root):
        Becomes: if False:
        """
        story_id = await _story(tmp_path)
        dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
        path = f"stories/{story_id}/codex/characters/vane.png"
        result = await GenerateImageTool(dispatcher=dispatcher).execute(
            {"prompt": "Lord Vane", "output_path": path}, _ctx(tmp_path)
        )
        assert (result.success, result.error) == (False, _refusal(path))
        dispatcher.dispatch.assert_not_called()

    async def test_generate_image_as_a_batch(self, tmp_path: Path) -> None:
        """Each image of a batch is checked, and the image is refused before its metadata.

        Killed by: src/uclone_x/tools/builtin/image.py :: dest_path = self.resolve_write_path(clean_rel, workspace)
        Becomes: dest_path = self.resolve_safe_path(clean_rel, workspace)
        """
        story_id = await _story(tmp_path)
        dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
        folder = f"stories/{story_id}/codex/characters"
        result = await GenerateImageTool(dispatcher=dispatcher).execute(
            {"prompt": "Lord Vane", "output_path": f"{folder}/vane.png", "count": 2},
            _ctx(tmp_path),
        )
        assert (result.success, result.error) == (False, _refusal(f"{folder}/vane_1.png"))
        dispatcher.dispatch.assert_not_called()

    async def test_the_comfy_generate_image(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/tools/builtin/comfy_image_tool.py :: dest_path = self.resolve_write_path(params.output_path, context.require_workspace())
        Becomes: dest_path = self.resolve_safe_path(params.output_path, context.require_workspace())
        """
        story_id = await _story(tmp_path)
        path = f"stories/{story_id}/codex/characters/vane.png"
        result = await ComfyImageGenTool().execute(
            {"prompt": "Lord Vane", "output_path": path}, _ctx(tmp_path)
        )
        assert (result.success, result.error) == (False, _refusal(path))


class TestAHardlinkIsNotWrittenThrough:
    async def test_writing_the_outside_name_leaves_the_story_copy(self, tmp_path: Path) -> None:
        """Not detected, and harmless: `file_write` replaces the file, never writes into it.

        Killed by: src/uclone_x/tools/builtin/filesystem.py :: replace_file(safe_path, encoded)
        Becomes: safe_path.write_bytes(encoded)
        """
        story_id = await _story(tmp_path)
        os.link(tmp_path / "stories" / story_id / VANE, tmp_path / "vane-copy.yaml")
        assert await _write(tmp_path, "vane-copy.yaml") == (True, None)
        assert (tmp_path / "vane-copy.yaml").read_text(encoding="utf-8") == "name: Nobody\n"
        assert (tmp_path / "stories" / story_id / VANE).read_text(encoding="utf-8") == VANE_TEXT

    async def test_an_image_written_over_a_linked_name_leaves_the_story_copy(
        self, tmp_path: Path
    ) -> None:
        """The image and its metadata file are both replaced, not written into.

        Killed by: src/uclone_x/tools/base.py :: handle.write(data)
        Becomes: path.write_bytes(data); handle.write(data)
        Killed by: src/uclone_x/tools/builtin/image.py :: replace_file(meta_path, json.dumps(meta_data, indent=2, ensure_ascii=False).encode())
        Becomes: meta_path.write_text(json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8")
        """
        story_id = await _story(tmp_path)
        story_file = tmp_path / "stories" / story_id / VANE
        os.link(story_file, tmp_path / "pic.png")
        os.link(story_file, tmp_path / "pic.json")
        dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
        dispatcher.dispatch.return_value = ImageGenerationResult(
            image_bytes=b"png bytes",
            seed=7,
            engine_name="mock",
            device_info="cpu",
            duration_seconds=0.1,
            width=64,
            height=64,
        )
        result = await GenerateImageTool(dispatcher=dispatcher).execute(
            {"prompt": "Lord Vane", "output_path": "pic.png"}, _ctx(tmp_path)
        )
        assert result.success, result.error
        assert (tmp_path / "pic.png").read_bytes() == b"png bytes"
        assert '"prompt": "Lord Vane"' in (tmp_path / "pic.json").read_text(encoding="utf-8")
        assert story_file.read_text(encoding="utf-8") == VANE_TEXT

    async def test_a_comfy_image_written_over_a_linked_name_leaves_the_story_copy(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/tools/builtin/comfy_image_tool.py :: replace_file(dest_path, image_bytes)
        Becomes: dest_path.write_bytes(image_bytes)
        """
        story_id = await _story(tmp_path)
        story_file = tmp_path / "stories" / story_id / VANE
        os.link(story_file, tmp_path / "pic.png")
        client = AsyncMock(spec=ComfyClient)
        client.queue_prompt.return_value = "prompt_1"
        client.wait_for_output.return_value = ["pic_00001.png"]
        client.download_image.return_value = b"png bytes"
        result = await ComfyImageGenTool(client=client).execute(
            {"prompt": "Lord Vane", "output_path": "pic.png"}, _ctx(tmp_path)
        )
        assert result.success, result.error
        assert (tmp_path / "pic.png").read_bytes() == b"png bytes"
        assert story_file.read_text(encoding="utf-8") == VANE_TEXT

    async def test_a_batch_written_over_linked_names_leaves_the_story_copy(
        self, tmp_path: Path
    ) -> None:
        """Every image of a batch and every metadata file is replaced, not written into (#1589).

        Killed by: src/uclone_x/tools/builtin/image.py :: replace_file(dest_path, batch_image)
        Becomes: dest_path.write_bytes(batch_image)
        Killed by: src/uclone_x/tools/builtin/image.py :: replace_file(meta_path, batch_meta)
        Becomes: meta_path.write_bytes(batch_meta)
        """
        story_id = await _story(tmp_path)
        story_file = tmp_path / "stories" / story_id / VANE
        names = ["pic_1.png", "pic_1.json", "pic_2.png", "pic_2.json"]
        for name in names:
            os.link(story_file, tmp_path / name)
        dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
        dispatcher.dispatch.return_value = ImageGenerationResult(
            image_bytes=b"png bytes",
            seed=7,
            engine_name="mock",
            device_info="cpu",
            duration_seconds=0.1,
            width=64,
            height=64,
        )
        result = await GenerateImageTool(dispatcher=dispatcher).execute(
            {"prompt": "Lord Vane", "output_path": "pic.png", "count": 2}, _ctx(tmp_path)
        )
        assert result.success, result.error
        assert (tmp_path / "pic_2.png").read_bytes() == b"png bytes"
        assert '"prompt": "Lord Vane"' in (tmp_path / "pic_2.json").read_text(encoding="utf-8")
        assert story_file.read_text(encoding="utf-8") == VANE_TEXT
        assert story_file.stat().st_nlink == 1  # every linked name was detached


def _save_vane(workspace: Path) -> Any:
    return CharacterSheetTool().execute(
        {"action": "save", "character_id": "vane", "name": "Nobody"}, _ctx(workspace)
    )


class TestCharacterSheetWritesLikeTheFileTools:
    """`character_sheet` writes a fixed `characters/<id>.yaml` outside a story (#1589).

    The name is fixed, but `characters/` is an ordinary workspace folder that can be a link,
    so the sheet is resolved and written as the general file tools write theirs.
    """

    async def test_a_characters_link_into_a_story_is_refused(self, tmp_path: Path) -> None:
        """A story's characters are its codex, which only the story tools write.

        Killed by: src/uclone_x/tools/builtin/character.py :: sheet = self.resolve_write_path(rel, workspace)
        Becomes: sheet = self.resolve_safe_path(rel, workspace)
        """
        story_id = await _story(tmp_path)
        (tmp_path / "characters").symlink_to(tmp_path / "stories" / story_id / "codex/characters")
        result = await _save_vane(tmp_path)
        assert (result.success, result.error) == (False, _refusal("characters/vane.yaml"))
        assert (tmp_path / "stories" / story_id / VANE).read_text(encoding="utf-8") == VANE_TEXT

    async def test_a_characters_link_out_of_the_workspace_is_refused_plainly(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/tools/builtin/character.py :: except PathTraversalError:
        Becomes: except PlainRefusalError:
        Killed by: src/uclone_x/tools/builtin/character.py :: sheet = self.resolve_write_path(rel, workspace)
        Becomes: sheet = workspace / rel
        """
        workspace = tmp_path / "workspace"
        outside = tmp_path / "elsewhere"
        workspace.mkdir()
        outside.mkdir()
        (workspace / "characters").symlink_to(outside)
        result = await _save_vane(workspace)
        assert (result.success, result.error) == (
            False,
            "'characters/vane.yaml' leads outside the workspace through a link, so the "
            "character sheet was not saved.",
        )
        assert list(outside.iterdir()) == []

    async def test_a_sheet_hardlinked_to_a_story_file_leaves_the_story_copy(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/tools/builtin/character.py :: replace_file(char_file, sheet_text.encode("utf-8"))
        Becomes: char_file.write_bytes(sheet_text.encode("utf-8"))
        """
        story_id = await _story(tmp_path)
        story_file = tmp_path / "stories" / story_id / VANE
        (tmp_path / "characters").mkdir()
        os.link(story_file, tmp_path / "characters" / "vane.yaml")
        result = await _save_vane(tmp_path)
        assert result.success, result.error
        sheet = yaml.safe_load((tmp_path / "characters" / "vane.yaml").read_text(encoding="utf-8"))
        assert sheet["name"] == "Nobody"
        assert story_file.read_text(encoding="utf-8") == VANE_TEXT


# --- The shell and local MCP servers (#1589 items 1 and 2) ------------------------------

darwin_only = pytest.mark.skipif(
    sys.platform != "darwin", reason="the story library jail exists only on macOS"
)


async def _shell(workspace: Path, command: str) -> ToolResult:
    return await BashRunTool().execute({"command": command, "cwd": str(workspace)}, _ctx(workspace))


@darwin_only
class TestTheShellCannotChangeTheLibrary:
    """On macOS a shell command runs in a jail that refuses every write in the library."""

    async def test_a_write_into_a_story_is_refused_with_a_note(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/tools/builtin/shell.py :: if jail:
        Becomes: if False:
        Killed by: src/uclone_x/tools/builtin/shell.py :: if jail and _NOT_PERMITTED in stderr_str:
        Becomes: if False:
        """
        story_id = await _story(tmp_path)
        result = await _shell(tmp_path, f"printf 'name: Nobody' > stories/{story_id}/{VANE}")
        assert not result.success
        assert result.error is not None
        assert result.error.endswith(STORY_LIBRARY_SHELL_NOTE)
        assert (tmp_path / "stories" / story_id / VANE).read_text(encoding="utf-8") == VANE_TEXT

    async def test_every_way_of_changing_the_library_is_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/sandbox/story_jail.py :: rules.append(f'(subpath (param "LIBRARY_{index}"))')
        Becomes: pass
        """
        story_id = await _story(tmp_path)
        vane = tmp_path / "stories" / story_id / VANE
        (tmp_path / "into").symlink_to(tmp_path / "stories" / story_id)
        (tmp_path / "other").mkdir()
        commands = [
            f"printf x > Stories/{story_id}/{VANE}",
            f"printf x > other/../stories/{story_id}/{VANE}",
            f"printf x > '{vane}'",
            f"printf x > into/{VANE}",
            f"printf x >> stories/{story_id}/{VANE}",
            f"printf x > stories/{story_id}/new.md",
            f"ln stories/{story_id}/{VANE} stories/{story_id}/linked.yaml",
            f"mv stories/{story_id}/{VANE} moved.yaml",
            f"rm stories/{story_id}/{VANE}",
            f"chmod 777 stories/{story_id}/{VANE}",
            "mv stories gone",
            "rm -rf stories",
        ]
        for command in commands:
            result = await _shell(tmp_path, command)
            assert not result.success, command
        assert vane.read_text(encoding="utf-8") == VANE_TEXT
        assert vane.stat().st_nlink == 1
        assert not (tmp_path / "stories" / story_id / "new.md").exists()
        assert not (tmp_path / "stories" / story_id / "linked.yaml").exists()
        assert not (tmp_path / "moved.yaml").exists()
        assert not (tmp_path / "gone").exists()

    async def test_the_workspace_and_its_folders_cannot_be_renamed(self, tmp_path: Path) -> None:
        """Renaming either would move the library to a path the jail does not name.

        Killed by: src/uclone_x/sandbox/story_jail.py :: rules.append(f'(literal (param "FOLDER_{index}"))')
        Becomes: pass
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        await _story(workspace)
        for command in (
            f"mv '{workspace}' '{tmp_path}/moved'",
            f"mv '{tmp_path}' '{tmp_path}-moved'",
        ):
            result = await _shell(workspace, command)
            assert not result.success, command
        assert (workspace / "stories").is_dir()

    async def test_the_rest_of_the_workspace_is_written_as_before(self, tmp_path: Path) -> None:
        await _story(tmp_path)
        result = await _shell(
            tmp_path,
            "printf a > notes.txt && mkdir -p drafts/stories && printf b > drafts/stories/f.md"
            " && printf c > stories-old.txt",
        )
        assert result.success, result.error
        assert (tmp_path / "notes.txt").read_text() == "a"
        assert (tmp_path / "drafts/stories/f.md").read_text() == "b"
        assert (tmp_path / "stories-old.txt").read_text() == "c"

    async def test_a_library_that_is_a_link_is_protected_where_it_leads(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/sandbox/story_jail.py :: subpaths.append(target)
        Becomes: pass
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        story_id = await _story(workspace)
        (workspace / "stories").rename(tmp_path / "library")
        (workspace / "stories").symlink_to(tmp_path / "library")
        vane = tmp_path / "library" / story_id / VANE
        result = await _shell(workspace, f"printf x > '{vane}'")
        assert not result.success
        assert vane.read_text(encoding="utf-8") == VANE_TEXT


@darwin_only
class TestACommandThatRanIsNotReportedAsNotRun:
    """Only a jail that never started the command is reported as not running it."""

    async def test_a_command_that_prints_what_sandbox_exec_prints_is_reported_as_run(
        self, tmp_path: Path
    ) -> None:
        """A command may run `sandbox-exec` itself, fail, and exit 71 with its message.

        Killed by: src/uclone_x/tools/builtin/shell.py :: started = (started_dir / "started").exists()
        Becomes: started = False
        """
        result = await _shell(
            tmp_path,
            "printf ran > ran.txt; echo 'sandbox-exec: sandbox_apply: Operation not permitted' >&2;"
            " exit 71",
        )
        assert not result.success
        assert result.error is not None
        assert result.error.startswith("Command failed with exit code 71: sandbox-exec:")
        assert (tmp_path / "ran.txt").read_text() == "ran"

    async def test_the_started_marker_is_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killed by: src/uclone_x/tools/builtin/shell.py :: shutil.rmtree(started_dir, ignore_errors=True)
        Becomes: pass
        """
        temp = tmp_path / "temp"
        temp.mkdir()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp))
        for command in ("true", "false"):
            await _shell(workspace, command)
        assert list(temp.iterdir()) == []


class TestTheShellDoesNotRunWithoutTheJail:
    """Where the jail should exist but cannot start, the command is refused in plain words."""

    async def test_a_missing_sandbox_exec_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killed by: src/uclone_x/sandbox/story_jail.py :: raise PlainRefusalError(JAIL_SETUP_REFUSAL)
        Becomes: return []
        """
        monkeypatch.setattr(story_jail, "PLATFORM", "darwin")
        monkeypatch.setattr(story_jail, "SANDBOX_EXEC", tmp_path / "missing-sandbox-exec")
        result = await _shell(tmp_path, "printf x > ran.txt")
        assert (result.success, result.error) == (False, JAIL_SETUP_REFUSAL)
        assert not (tmp_path / "ran.txt").exists()

    async def test_a_jail_that_fails_to_start_is_reported_plainly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`sandbox-exec` exits 71 and names itself when it cannot apply the profile.

        Killed by: src/uclone_x/tools/builtin/shell.py :: if jail and not started:
        Becomes: if False:
        """
        fake = tmp_path / "fake-sandbox-exec"
        fake.write_text(
            "#!/bin/sh\necho 'sandbox-exec: sandbox_apply: Operation not permitted' >&2\nexit 71\n"
        )
        fake.chmod(0o755)
        monkeypatch.setattr(story_jail, "PLATFORM", "darwin")
        monkeypatch.setattr(story_jail, "SANDBOX_EXEC", fake)
        result = await _shell(tmp_path, "true")
        assert (result.success, result.error) == (False, JAIL_SETUP_REFUSAL)

    def test_other_systems_have_no_jail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(story_jail, "PLATFORM", "linux")
        assert story_jail.story_library_jail(tmp_path) == []

    def test_no_workspace_means_no_jail(self) -> None:
        assert story_jail.story_library_jail(None) == []


_PROBE_SERVER = """
import json, sys
target, report = sys.argv[1], sys.argv[2]
try:
    with open(target, "w") as handle:
        handle.write("name: Nobody")
    outcome = "written"
except OSError as err:
    outcome = type(err).__name__
with open(report, "w") as handle:
    handle.write(outcome)
for line in sys.stdin:
    req = json.loads(line)
    if req.get("method") == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "probe", "version": "1"}}
    elif req.get("method") == "tools/list":
        result = {"tools": []}
    else:
        continue
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}) + "\\n")
    sys.stdout.flush()
"""


@darwin_only
class TestALocalMCPServerCannotChangeTheLibrary:
    async def test_a_stdio_server_cannot_write_a_story(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/tools/client.py :: *jail,
        Becomes: *[],
        """
        story_id = await _story(tmp_path)
        vane = tmp_path / "stories" / story_id / VANE
        config = MCPConnectionConfig(
            server_name="probe",
            transport=MCPTransport.STDIO,
            command=sys.executable,
            args=("-c", _PROBE_SERVER, str(vane), str(tmp_path / "report.txt")),
            workspace_root=tmp_path,
        )
        async with MCPClient(config=config):
            pass
        assert (tmp_path / "report.txt").read_text() == "PermissionError"
        assert vane.read_text(encoding="utf-8") == VANE_TEXT

    async def test_a_missing_server_is_still_named_as_missing(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/tools/client.py :: if jail and not _command_exists(self._config.command, child_env, cwd_str):
        Becomes: if False:
        """
        config = MCPConnectionConfig(
            server_name="missing",
            transport=MCPTransport.STDIO,
            command="no-such-mcp-server-1589",
            workspace_root=tmp_path,
        )
        with pytest.raises(
            FileNotFoundError, match="MCP server executable not found: 'no-such-mcp-server-1589'"
        ):
            async with MCPClient(config=config):
                pass


class TestCharacterSheetFolderShapes:
    """A `characters` that is a link to nowhere or a file (#1589 follow-up b)."""

    async def test_a_link_to_a_missing_folder_in_the_workspace_is_saved_through(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/tools/builtin/character.py :: if not chars_dir.is_symlink():
        Becomes: if True:
        Killed by: src/uclone_x/tools/builtin/character.py :: sheet.parent.mkdir(parents=True, exist_ok=True)
        Becomes: pass
        """
        (tmp_path / "characters").symlink_to(tmp_path / "cast")
        result = await _save_vane(tmp_path)
        assert result.success, result.error
        sheet = yaml.safe_load((tmp_path / "cast" / "vane.yaml").read_text(encoding="utf-8"))
        assert sheet["name"] == "Nobody"

    async def test_a_link_to_a_missing_folder_outside_is_refused_plainly(
        self, tmp_path: Path
    ) -> None:
        """It used to fail with a raw `FileExistsError` naming the link's full path.

        Killed by: src/uclone_x/tools/builtin/character.py :: if not chars_dir.is_symlink():
        Becomes: if True:
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "characters").symlink_to(tmp_path / "elsewhere")
        result = await _save_vane(workspace)
        assert (result.success, result.error) == (
            False,
            "'characters/vane.yaml' leads outside the workspace through a link, so the "
            "character sheet was not saved.",
        )
        assert not (tmp_path / "elsewhere").exists()

    async def test_a_characters_file_is_refused_plainly(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/tools/builtin/character.py :: except FileExistsError:
        Becomes: except KeyError:
        """
        (tmp_path / "characters").write_text("not a folder", encoding="utf-8")
        result = await _save_vane(tmp_path)
        assert (result.success, result.error) == (
            False,
            "'characters' in the workspace is a file, not a folder, so no character sheet "
            "can be read or saved there.",
        )

    async def test_a_link_to_a_file_is_refused_plainly(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/tools/builtin/character.py :: except (FileExistsError, NotADirectoryError):
        Becomes: except KeyError:
        """
        (tmp_path / "cast.txt").write_text("not a folder", encoding="utf-8")
        (tmp_path / "characters").symlink_to(tmp_path / "cast.txt")
        result = await _save_vane(tmp_path)
        assert (result.success, result.error) == (
            False,
            "'characters' leads to a file, not a folder, so the character sheet was not saved.",
        )
        assert (tmp_path / "cast.txt").read_text(encoding="utf-8") == "not a folder"
