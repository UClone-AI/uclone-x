"""The ComfyUI connection's address reaches `generate_image` (#1976, model-gateway §3.5).

Before #1976 `generate_image`'s ComfyUI engine read only `UCX_COMFYUI_URL`. The saved
`comfyui_base_url` then reached it through the picture settings; since model-gateway step 5
a ComfyUI is a connection (`kind: comfyui`), and its address travels in the same picture
settings the dispatcher reads on every draw, with the variable still overriding it
(settings-single-source S4). With no ComfyUI connection, no ComfyUI is asked at all.
"""

from __future__ import annotations

import json
import struct
import time
import zlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from uclone_x.llm import MockLLMConnector
from uclone_x.tools.builtin.image import (
    COMFY_URL_ENV,
    LOCAL_PROBE_TTL_SECONDS,
    ComfyUIImageEngine,
    GenerateImageTool,
    ImageEngineChoice,
    ImagePipelineDispatcher,
    LocalDiffusersImageEngine,
    RemoteCudaImageEngine,
)
from uclone_x.tools.builtin.image_status import probe_image_engines
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry
from uclone_x.ui.app import create_ui_app

SAVED = "http://127.0.0.1:18188"


def _png() -> bytes:
    """A 1x1 PNG, so whatever reads the drawn bytes sees a real picture."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )


class _RecordingComfyClient:
    """A ComfyUI daemon that answers anywhere and records the address it was asked at."""

    addresses: list[str] = []

    def __init__(self, base_url: str, **_: Any) -> None:
        self.addresses.append(base_url)

    async def alive(self, timeout: float = 3.0) -> bool:
        return True

    async def queue_prompt(self, workflow: dict[str, Any]) -> str:
        return "prompt-1"

    async def wait_for_output(self, prompt_id: str, **_: Any) -> list[str]:
        return ["out.png"]

    async def download_image(self, filename: str, **_: Any) -> bytes:
        return _png()

    async def aclose(self) -> None:
        return None


def _absent(spec: type) -> AsyncMock:
    engine = AsyncMock(spec=spec)
    engine.is_available.return_value = False
    engine.base_url = ""
    return engine


def _dispatcher() -> ImagePipelineDispatcher:
    """A dispatcher whose only ready engine is a real `ComfyUIImageEngine`."""
    local = _absent(LocalDiffusersImageEngine)
    local.resolve_checkpoint.return_value = None
    return ImagePipelineDispatcher(
        remote_engine=_absent(RemoteCudaImageEngine),
        comfy_engine=ComfyUIImageEngine(),
        local_engine=local,
    )


@pytest.fixture(autouse=True)
def no_comfy_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(COMFY_URL_ENV, raising=False)
    monkeypatch.delenv("COMFYUI_BASE_URL", raising=False)
    _RecordingComfyClient.addresses = []


def _settings(path: Path, *rows: dict[str, Any]) -> Path:
    path.write_text(json.dumps({"connections": list(rows)}), encoding="utf-8")
    return path


def _comfy_row(address: str = SAVED) -> dict[str, Any]:
    return {"id": "comfyui", "kind": "comfyui", "base_url": address}


@pytest.mark.asyncio
@pytest.mark.usefixtures("builtin_personas_absent")
async def test_generate_image_draws_at_the_comfyui_connections_address(tmp_path: Path) -> None:
    """The dashboard's ComfyUI connection is where the next picture is drawn.

    Killed by: src/uclone_x/llm/connectors/factory.py :: comfyui_base_url=comfy.base_url if comfy is not None else None,
    Becomes: comfyui_base_url=None,
    Killed by: src/uclone_x/tools/builtin/image.py :: self._comfy_engine.use_saved_address(choice.comfyui_base_url)
    Becomes: pass
    """
    _settings(tmp_path / "settings.json", _comfy_row())
    tool = GenerateImageTool(dispatcher=_dispatcher())
    create_ui_app(
        static_dir=tmp_path,
        llm=MockLLMConnector(responses=["ok"]),
        tools=ToolRegistry(tools=[tool]),
        storage_dir=tmp_path,
        fallback_to_mock=True,
    )

    with patch("uclone_x.tools.builtin.image.ComfyClient", _RecordingComfyClient):
        result = await tool.execute(
            params={"prompt": "a lighthouse at dusk"},
            context=ToolContext(agent_id="a", workspace_root=tmp_path, session_id="s"),
        )

    assert result.success is True, result.error
    assert _RecordingComfyClient.addresses
    assert set(_RecordingComfyClient.addresses) == {SAVED}


@pytest.mark.asyncio
@pytest.mark.usefixtures("builtin_personas_absent")
async def test_with_no_comfyui_connection_no_comfyui_is_asked(tmp_path: Path) -> None:
    """A daemon on the default port is not used unless a connection names it (§3.5).

    Killed by: src/uclone_x/tools/builtin/image.py :: if not choice.from_connections or choice.comfyui_base_url:
    Becomes: if True:
    """
    tool = GenerateImageTool(dispatcher=_dispatcher())
    create_ui_app(
        static_dir=tmp_path,
        llm=MockLLMConnector(responses=["ok"]),
        tools=ToolRegistry(tools=[tool]),
        storage_dir=tmp_path,
        fallback_to_mock=True,
    )

    with patch("uclone_x.tools.builtin.image.ComfyClient", _RecordingComfyClient):
        result = await tool.execute(
            params={"prompt": "a lighthouse at dusk"},
            context=ToolContext(agent_id="a", workspace_root=tmp_path, session_id="s"),
        )

    assert result.success is False
    assert _RecordingComfyClient.addresses == []


def _bound(**fields: Any) -> ImageEngineChoice:
    return ImageEngineChoice(from_connections=True, **fields)


@pytest.mark.asyncio
async def test_the_environment_overrides_the_connections_comfyui_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`UCX_COMFYUI_URL` still wins over the saved address: an override, never storage (S4).

    Killed by: src/uclone_x/tools/builtin/image.py :: return self._override_url or self._saved_url or DEFAULT_COMFYUI_BASE_URL
    Becomes: return self._saved_url or self._override_url or DEFAULT_COMFYUI_BASE_URL
    """
    monkeypatch.setenv(COMFY_URL_ENV, "http://10.0.0.5:8188")
    dispatcher = _dispatcher()
    dispatcher.bind_engine_settings(lambda _own: _bound(comfyui_base_url=SAVED))

    with patch("uclone_x.tools.builtin.image.ComfyClient", _RecordingComfyClient):
        await dispatcher.dispatch("a lighthouse", "", "1:1", 7, "none")

    assert set(_RecordingComfyClient.addresses) == {"http://10.0.0.5:8188"}


class _ComfyAnsweringOnlyAtSaved(_RecordingComfyClient):
    """A ComfyUI daemon that answers only at `SAVED`, as a tunnel's port does once it connects."""

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__(base_url, **kwargs)
        self.base_url = base_url

    async def alive(self, timeout: float = 3.0) -> bool:
        return self.base_url == SAVED


class _CountingGemini:
    """A `GeminiImageClient` that counts the pictures it was asked for."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_image(self, prompt: str, aspect_ratio: str, model: str) -> tuple[bytes, str]:
        self.calls += 1
        return _png(), "image/png"


@pytest.mark.asyncio
async def test_a_changed_address_is_looked_at_again_before_the_next_draw() -> None:
    """Under `auto` with a cloud model, the first picture after a tunnel connects is local.

    The dispatcher caches "is any local engine ready" for `LOCAL_PROBE_TTL_SECONDS`. That
    look was taken at the old address, so a new address must not reuse it; otherwise the
    next picture goes to the cloud for up to 30 seconds while ComfyUI answers at the new
    one, breaking local-first without anyone seeing it (review of PR #1980, condition 1).
    The look is keyed by the addresses it was taken at.

    Killed by: src/uclone_x/tools/builtin/image.py :: return (choice.pin, choice.from_connections, choice.comfyui_base_url, choice.remote_url)
    Becomes: return (choice.pin, choice.from_connections, choice.remote_url)
    """
    gemini = _CountingGemini()
    saved = {"url": "http://127.0.0.1:9"}
    dispatcher = _dispatcher()
    dispatcher.bind_engine_settings(
        lambda _own: _bound(
            gemini=gemini, gemini_model="gemini-2.5-flash-image", comfyui_base_url=saved["url"]
        )
    )

    with patch("uclone_x.tools.builtin.image.ComfyClient", _ComfyAnsweringOnlyAtSaved):
        started = time.monotonic()
        await dispatcher.dispatch("a lighthouse", "", "1:1", 7, "none")
        assert gemini.calls == 1  # nothing answered at the old address
        assert SAVED not in _RecordingComfyClient.addresses

        saved["url"] = SAVED  # the tunnel connects and saves its address
        drawn = await dispatcher.dispatch("a lighthouse", "", "1:1", 7, "none")
        elapsed = time.monotonic() - started

    assert elapsed < LOCAL_PROBE_TTL_SECONDS  # the old look would still be fresh
    assert gemini.calls == 1
    assert drawn.engine_name == "comfyui-local"
    assert SAVED in _RecordingComfyClient.addresses


@pytest.mark.asyncio
async def test_no_answer_at_the_connection_names_the_connection_not_the_variable() -> None:
    """With nothing ready, the diagnosis says the address came from the ComfyUI connection.

    Telling the person to "set UCX_COMFYUI_URL" sends them to a variable that is not
    where the address came from (the same shape as PR #1096's reviewer finding).

    Killed by: src/uclone_x/tools/builtin/image.py :: elif self._comfy_saved_url:
    Becomes: elif False:
    """
    dispatcher = _dispatcher()
    dispatcher.bind_engine_settings(lambda _own: _bound(comfyui_base_url=SAVED))
    client = AsyncMock()
    client.alive.return_value = False

    with (
        patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client),
        pytest.raises(Exception, match="No image generation engine available") as caught,
    ):
        await dispatcher.dispatch("a lighthouse", "", "1:1", 7, "none")

    message = str(caught.value)
    assert f"No ComfyUI daemon answered at {SAVED}, the address of the ComfyUI connection" in (
        message
    )
    assert f"set {COMFY_URL_ENV}" not in message


def test_the_engine_status_probes_the_connections_comfyui_address(tmp_path: Path) -> None:
    """`/api/media/status` and `ucx media status` look where `generate_image` draws.

    Killed by: src/uclone_x/tools/builtin/image_status.py :: comfy_url = choice.comfyui_base_url
    Becomes: comfy_url = None
    """
    settings = _settings(tmp_path / "settings.json", _comfy_row())
    client = AsyncMock()
    client.alive.return_value = False

    with patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client):
        report = probe_image_engines(settings)

    assert report.comfy_url == SAVED
