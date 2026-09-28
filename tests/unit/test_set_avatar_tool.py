"""`set_avatar`: a clone changes its own picture, and only its own.

What these pin:

* **Whose picture**: the persona of the agent running the tool, not `ToolContext.agent_id`.
  In a room the id is the seat, and a seat named after nothing must still change the
  persona's picture. With no agent there is no one to change, and the tool says so.
* **Which file**: a path in the workspace; one outside it is refused in plain words.
* **Undo**: the `previous_path` a set returns sets the old picture back, and a first set
  returns none, so the clone knows to reset instead. Each call returns the `change_id` of
  its change; an undo passing it as `undo_of` is refused, with nothing changed, once a
  later change was made -- by a person in the head, say, while the clone was talking (#1809).
* **Binding**: in the default catalog, not a writing tool, and named by every shipped
  clone that has a tool list.
* **No image model**: `generate_image` refuses in plain words with
  `reason_code: no_image_engine`, and names no engine, error class or setting variable.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from uclone_x.agent.avatar_tool import SetAvatarTool
from uclone_x.agent.persona_registry import BUILTIN_PERSONAS_DIR
from uclone_x.tools.builtin.image import (
    NO_IMAGE_ENGINE_TEXT,
    CheckpointResolution,
    ComfyUIImageEngine,
    GenerateImageTool,
    ImagePipelineDispatcher,
    LocalDiffusersImageEngine,
    RemoteCudaImageEngine,
)
from uclone_x.tools.models import NoIsolation, ToolContext
from uclone_x.tools.registry import create_default_registry

PERSONAS = Path(".uclone") / "personas"

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
OTHER_PNG = PNG + b"\x00"


def _install(workspace: Path, name: str) -> None:
    folder = workspace / PERSONAS
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "role": "Site Surveyor",
                "description": "Maps a site before anyone digs.",
                "system_prompt": "You survey.",
                "allowed_tools": [],
            }
        ),
        encoding="utf-8",
    )


def _ctx(workspace: Path, *, persona: str | None, agent_id: str = "surveyor") -> ToolContext:
    delegate = SimpleNamespace(persona_name=persona) if persona is not None else None
    return ToolContext(
        agent_id=agent_id,
        session_id="sess_1",
        workspace_root=workspace,
        isolation=NoIsolation(),
        agent_delegate=delegate,
    )


def _image(workspace: Path, rel: str, data: bytes = PNG) -> str:
    path = workspace / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return rel


class TestWhosePicture:
    async def test_a_room_seat_sets_its_personas_picture_not_one_named_after_the_seat(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/agent/avatar_tool.py :: persona: object = getattr(agent, "persona_name", None) if agent is not None else None
        Becomes: persona: object = context.agent_id
        """
        _install(tmp_path, "surveyor")
        _install(tmp_path, "seat-7")
        rel = _image(tmp_path, "artifacts/images/img_1.png")

        result = await SetAvatarTool().execute(
            {"image_path": rel}, _ctx(tmp_path, persona="surveyor", agent_id="seat-7")
        )

        assert result.success, result.error
        assert isinstance(result.output, dict)
        url = result.output["avatar_url"]
        assert isinstance(url, str) and url.startswith("/api/personas/surveyor/avatar?v=")
        assert (tmp_path / PERSONAS / "surveyor.png").read_bytes() == PNG
        assert not (tmp_path / PERSONAS / "seat-7.png").exists()

    async def test_with_no_clone_running_it_refuses_in_plain_words(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/avatar_tool.py :: if not isinstance(persona, str) or not persona:
        Becomes: if False:
        """
        _install(tmp_path, "surveyor")
        rel = _image(tmp_path, "artifacts/images/img_1.png")

        result = await SetAvatarTool().execute({"image_path": rel}, _ctx(tmp_path, persona=None))

        assert result.success is False
        assert result.error is not None
        assert result.error.startswith("Only a clone can change its own picture")
        assert not (tmp_path / PERSONAS / "surveyor.png").exists()

    async def test_a_path_outside_the_workspace_is_refused_in_plain_words(
        self, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "work"
        _install(workspace, "surveyor")
        outside = tmp_path / "outside.png"
        outside.write_bytes(PNG)

        result = await SetAvatarTool().execute(
            {"image_path": str(outside)}, _ctx(workspace, persona="surveyor")
        )

        assert result.success is False
        assert result.error is not None
        assert "outside the workspace" in result.error
        assert "Error" not in result.error
        assert not (workspace / PERSONAS / "surveyor.png").exists()


class TestUndo:
    async def test_previous_path_sets_the_old_picture_back(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/avatar_tool.py :: "previous_path": _relative(change.previous, workspace),
        Becomes: "previous_path": None,
        """
        _install(tmp_path, "surveyor")
        tool = SetAvatarTool()
        ctx = _ctx(tmp_path, persona="surveyor")
        first = await tool.execute({"image_path": _image(tmp_path, "a.png", PNG)}, ctx)
        assert isinstance(first.output, dict)
        assert first.output["previous_path"] is None

        second = await tool.execute({"image_path": _image(tmp_path, "b.png", OTHER_PNG)}, ctx)
        assert isinstance(second.output, dict)
        previous = second.output["previous_path"]
        assert previous == ".uclone/personas/surveyor.prev.png"

        undone = await tool.execute({"image_path": previous}, ctx)

        assert undone.success, undone.error
        assert (tmp_path / PERSONAS / "surveyor.png").read_bytes() == PNG

    async def test_each_call_returns_its_change_id_and_an_undo_passes_it_back(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/agent/avatar_tool.py :: "change_id": change.change_id,
        Becomes: "change_id": None,
        """
        _install(tmp_path, "surveyor")
        tool = SetAvatarTool()
        ctx = _ctx(tmp_path, persona="surveyor")
        first = await tool.execute({"image_path": _image(tmp_path, "a.png", PNG)}, ctx)
        second = await tool.execute({"image_path": _image(tmp_path, "b.png", OTHER_PNG)}, ctx)
        assert isinstance(first.output, dict) and isinstance(second.output, dict)
        assert (first.output["change_id"], second.output["change_id"]) == (1, 2)

        undone = await tool.execute(
            {"image_path": second.output["previous_path"], "undo_of": second.output["change_id"]},
            ctx,
        )

        assert undone.success, undone.error
        assert isinstance(undone.output, dict)
        assert undone.output["change_id"] == 3
        assert (tmp_path / PERSONAS / "surveyor.png").read_bytes() == PNG

    async def test_an_undo_after_a_later_change_is_refused_and_changes_nothing(
        self, tmp_path: Path
    ) -> None:
        """A person changed the picture in the head after the clone's change.

        Killed by: src/uclone_x/agent/avatar_tool.py :: change = store.set_from_path(persona, source, undo_of=params.undo_of)
        Becomes: change = store.set_from_path(persona, source)
        """
        _install(tmp_path, "surveyor")
        tool = SetAvatarTool()
        ctx = _ctx(tmp_path, persona="surveyor")
        await tool.execute({"image_path": _image(tmp_path, "a.png", PNG)}, ctx)
        mine = await tool.execute({"image_path": _image(tmp_path, "b.png", OTHER_PNG)}, ctx)
        assert isinstance(mine.output, dict)
        later = PNG + b"later"
        await tool.execute({"image_path": _image(tmp_path, "c.png", later)}, ctx)

        undone = await tool.execute(
            {"image_path": mine.output["previous_path"], "undo_of": mine.output["change_id"]}, ctx
        )

        assert undone.success is False
        assert undone.output == {"reason_code": "stale_change"}
        assert undone.error is not None and "changed again" in undone.error
        assert (tmp_path / PERSONAS / "surveyor.png").read_bytes() == later

    async def test_a_reset_undo_after_a_later_change_is_refused(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/avatar_tool.py :: change = store.reset(persona, undo_of=params.undo_of)
        Becomes: change = store.reset(persona)
        """
        _install(tmp_path, "surveyor")
        tool = SetAvatarTool()
        ctx = _ctx(tmp_path, persona="surveyor")
        mine = await tool.execute({"image_path": _image(tmp_path, "a.png", PNG)}, ctx)
        assert isinstance(mine.output, dict) and mine.output["previous_path"] is None
        await tool.execute({"image_path": _image(tmp_path, "b.png", OTHER_PNG)}, ctx)

        undone = await tool.execute({"reset": True, "undo_of": mine.output["change_id"]}, ctx)

        assert undone.success is False
        assert (tmp_path / PERSONAS / "surveyor.png").read_bytes() == OTHER_PNG

    async def test_reset_goes_back_to_the_shipped_picture(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path, persona="artist")
        await SetAvatarTool().execute({"image_path": _image(tmp_path, "a.png")}, ctx)

        result = await SetAvatarTool().execute({"reset": True}, ctx)

        assert result.success, result.error
        assert isinstance(result.output, dict)
        assert result.output["previous_path"] == ".uclone/personas/artist.prev.png"
        assert not (tmp_path / PERSONAS / "artist.png").exists()
        assert (BUILTIN_PERSONAS_DIR / "artist.png").is_file()

    @pytest.mark.parametrize("args", [{}, {"image_path": "a.png", "reset": True}])
    async def test_exactly_one_of_image_path_and_reset(
        self, tmp_path: Path, args: dict[str, object]
    ) -> None:
        result = await SetAvatarTool().execute(args, _ctx(tmp_path, persona="artist"))

        assert result.success is False
        assert result.error is not None
        assert "one of the two" in result.error


class TestBinding:
    def test_it_is_in_the_catalog_and_is_not_a_writing_tool(self, tmp_path: Path) -> None:
        registry = create_default_registry(workspace_root=tmp_path, enable_mcp=False)

        tool = registry.get("set_avatar")

        assert isinstance(tool, SetAvatarTool)
        assert SetAvatarTool.writes_files is False

    @pytest.mark.parametrize("name", ["artist", "guardian", "pioneer", "scout", "writer"])
    def test_every_shipped_clone_with_a_tool_list_names_it(self, name: str) -> None:
        data = yaml.safe_load((BUILTIN_PERSONAS_DIR / f"{name}.yaml").read_text(encoding="utf-8"))

        assert "set_avatar" in data["allowed_tools"]


def _no_engine_dispatcher() -> ImagePipelineDispatcher:
    """The real dispatcher with every engine declining, as on a fresh install."""
    remote = AsyncMock(spec=RemoteCudaImageEngine)
    remote.is_available.return_value = False
    remote.base_url = ""
    comfy = AsyncMock(spec=ComfyUIImageEngine)
    comfy.is_available.return_value = False
    comfy.base_url = "http://127.0.0.1:8188"
    local = AsyncMock(spec=LocalDiffusersImageEngine)
    local.is_available.return_value = False
    local.checkpoint_resolution = MagicMock(return_value=CheckpointResolution("unconfigured"))
    return ImagePipelineDispatcher(remote_engine=remote, comfy_engine=comfy, local_engine=local)


class TestNoImageModel:
    async def test_the_refusal_is_plain_words_with_a_reason_code(self, tmp_path: Path) -> None:
        """Through the real dispatcher with every engine down, as a fresh install sees it.

        Killed by: src/uclone_x/tools/builtin/image.py :: raise PlainRefusalError(NO_IMAGE_ENGINE_TEXT, reason_code="no_image_engine") from exc
        Becomes: raise
        Killed by: src/uclone_x/tools/base.py :: output={"reason_code": e.reason_code} if e.reason_code else None,
        Becomes: output=None,
        """
        tool = GenerateImageTool(dispatcher=_no_engine_dispatcher())

        result = await tool.execute(
            {"prompt": "a friendly face"}, _ctx(tmp_path, persona="artist", agent_id="artist")
        )

        assert result.success is False
        assert result.error == NO_IMAGE_ENGINE_TEXT
        assert result.output == {"reason_code": "no_image_engine"}
        for internal in (
            "ucx",
            "ImageGenerationError",
            "diffusers",
            "ComfyUI",
            "UCLONE_",
            "_URL",
            "pip",
            "Tool execution failed",
        ):
            assert internal not in result.error
        assert "Settings › Images" in result.error
        assert "upload a picture" in result.error
        assert list(tmp_path.rglob("*.png")) == []
