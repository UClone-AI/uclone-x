"""The Gemini image engine, and the picture model that decides when it draws (model-gateway §3.5).

Every Gemini request here goes to an `httpx.MockTransport`; nothing reaches the network.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from uclone_x.core.remote_worker import PortMapping, SSHTunnelManager, TunnelSessionStatus
from uclone_x.errors import ImageNotReturnedError
from uclone_x.llm.connectors.factory import image_engine_choice
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.tools.base import artifact_path_from_url, linked_paths
from uclone_x.tools.builtin.image import (
    ComfyUIImageEngine,
    GenerateImageParams,
    GenerateImageTool,
    ImageEngineChoice,
    ImageEngineRefusal,
    ImageGenerationError,
    ImageGenerationResult,
    ImagePipelineDispatcher,
    ImageWhere,
    LocalDiffusersImageEngine,
    RemoteCudaImageEngine,
    gemini_profile,
    image_extension_for,
    image_where,
    jpeg_dimensions,
    picture_path_for,
    sidecar_path_for,
    sniff_picture,
    webp_dimensions,
)
from uclone_x.tools.builtin.image_status import (
    ImageEngineReport,
    KnownModel,
    media_status_payload,
    probe_image_engines,
    resolve_image_choice,
)
from uclone_x.tools.builtin.media_registry import ModelProfile, PromptFamily
from uclone_x.tools.models import ToolContext
from uclone_x.ui.app import AgentSessionManager, create_ui_app


def _picture_rel(url: object) -> str:
    """The workspace path a result's picture link serves; fails the test if it names none."""
    rel = artifact_path_from_url(url)
    assert rel is not None, f"not a picture link: {url!r}"
    return rel


def _sidecar(workspace: Path, url: object) -> dict[str, Any]:
    """The recipe sidecar saved beside the picture a result links to."""
    side: dict[str, Any] = json.loads(
        (workspace / sidecar_path_for(_picture_rel(url))).read_text(encoding="utf-8")
    )
    return side


def _png(width: int, height: int) -> bytes:
    """The first 24 bytes of a PNG -- signature and IHDR -- which is all a size is read from."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


#: A real 5x3 baseline JPEG as an encoder writes one (JFIF APP0, DQT, SOF0, DHT, SOS), so
#: the size reader has to walk past the segments ahead of SOF0 to find the size.
_JPEG_5X3 = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERX"
    "RTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2Nj"
    "Y2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAARCAADAAUDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAA"
    "AAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAk"
    "M2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKT"
    "lJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QA"
    "HwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdh"
    "cRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hp"
    "anN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk"
    "5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDHooorhPqD/9k="
)
#: Real 5x3 WebP files in each of the three first-chunk layouts an encoder writes.
_WEBP_LOSSLESS_5X3 = base64.b64decode("UklGRh4AAABXRUJQVlA4TBEAAAAvBIAAAAdQjyLXo/+BiOh/AAA=")
_WEBP_LOSSY_5X3 = base64.b64decode(
    "UklGRjYAAABXRUJQVlA4ICoAAACQAQCdASoFAAMAAsBMJaACdLoAA5gA/u2QP4hd7G2//PTP62/j/xkAAAA="
)
_WEBP_EXTENDED_5X3 = base64.b64decode(
    "UklGRloAAABXRUJQVlA4WAoAAAAQAAAABAAAAgAAQUxQSAoAAAABB1DAiAhERP8DVlA4ICoAAACQAQCdASoFAAMA"
    "AsBMJaACdLoAA5gA/u2QP4hd7G2//PTP62/j/xkAAAA="
)


def _image_reply(data: bytes, mime: str = "image/png") -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Here is your picture."},
                        {"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}},
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }


class _Recorder:
    """A MockTransport handler that records each request and answers with ``reply``."""

    def __init__(self, reply: dict[str, Any], status: int = 200) -> None:
        self.reply = reply
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.reply)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


class _FakeGemini:
    """A `GeminiImageClient` that counts calls and returns a 1024x1024 PNG."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def generate_image(self, prompt: str, aspect_ratio: str, model: str) -> tuple[bytes, str]:
        self.calls.append((prompt, aspect_ratio, model))
        return _png(1024, 1024), "image/png"


def _locals(*, ready: bool) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    """Three local engine mocks, all ready or none, the in-process one returning a result."""
    remote = AsyncMock(spec=RemoteCudaImageEngine)
    remote.is_available.return_value = False
    remote.base_url = ""
    comfy = AsyncMock(spec=ComfyUIImageEngine)
    comfy.is_available.return_value = False
    comfy.base_url = "http://127.0.0.1:8188"
    local = AsyncMock(spec=LocalDiffusersImageEngine)
    local.is_available.return_value = ready
    local.resolve_checkpoint.return_value = None
    local.checkpoint_resolution.return_value.usable = False
    local.checkpoint_resolution.return_value.describe.return_value = "No checkpoint."
    local.generate.return_value = ImageGenerationResult(
        image_bytes=b"local",
        seed=1,
        engine_name="diffusers-sdxl",
        device_info="cpu",
        duration_seconds=0.1,
        width=1024,
        height=1024,
    )
    return remote, comfy, local


def _dispatcher(choice: ImageEngineChoice, *, local_ready: bool) -> ImagePipelineDispatcher:
    remote, comfy, local = _locals(ready=local_ready)
    return ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=local,
        engine_settings=lambda _own: choice,
    )


class _FakeGeminiSending:
    """A `GeminiImageClient` that answers every request with ``data`` labelled ``mime``."""

    def __init__(self, data: bytes, mime: str) -> None:
        self._reply = (data, mime)

    async def generate_image(self, prompt: str, aspect_ratio: str, model: str) -> tuple[bytes, str]:
        return self._reply


def _pinned_gemini(gemini: Any, model: str = "gemini-2.5-flash-image") -> ImageEngineChoice:
    """The choice a picture model ref on a Google connection resolves to."""
    return ImageEngineChoice(
        chosen=f"gemini/{model}",
        pin="gemini",
        from_connections=True,
        gemini=gemini,
        gemini_model=model,
        gemini_connection="gemini",
    )


def _auto(gemini: Any = None, model: str | None = "gemini-2.5-flash-image") -> ImageEngineChoice:
    """`auto`, with ``gemini`` as the cloud model on a connection that has a key."""
    return ImageEngineChoice(
        gemini=gemini,
        gemini_model=model if gemini is not None else None,
        gemini_connection="gemini" if gemini is not None else None,
    )


def _gemini_tool(data: bytes, mime: str) -> GenerateImageTool:
    gemini = _FakeGeminiSending(data, mime)
    choice = _pinned_gemini(gemini, "gemini-3.1-flash-image")
    return GenerateImageTool(dispatcher=_dispatcher(choice, local_ready=False))


async def _draw(dispatcher: ImagePipelineDispatcher) -> ImageGenerationResult:
    return await dispatcher.dispatch(
        prompt="a lighthouse at dawn",
        negative_prompt="",
        aspect_ratio="16:9",
        seed=7,
        style="artistic",
    )


# -- The connector ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_request_asks_for_an_image_with_the_key_header() -> None:
    """The picture is asked for as an image modality, at the ratio, with the key in its header.

    Killed by: src/uclone_x/llm/connectors/gemini.py :: "responseModalities": ["IMAGE"],
    Becomes: "responseModalities": ["TEXT"],
    Killed by: src/uclone_x/llm/connectors/gemini.py :: "imageConfig": {"aspectRatio": aspect_ratio},
    Becomes: "imageConfig": {},
    """
    recorder = _Recorder(_image_reply(_png(8, 8)))
    connector = GeminiConnector(api_key="test-key", http_client=recorder.client())

    await connector.generate_image("a red kite", "9:16", "gemini-2.5-flash-image")

    (request,) = recorder.requests
    assert request.url.path.endswith("/models/gemini-2.5-flash-image:generateContent")
    assert request.headers["x-goog-api-key"] == "test-key"
    body = json.loads(request.content)
    assert body["generationConfig"]["responseModalities"] == ["IMAGE"]
    assert body["generationConfig"]["imageConfig"] == {"aspectRatio": "9:16"}
    assert body["contents"][0]["parts"] == [{"text": "a red kite"}]


@pytest.mark.asyncio
async def test_the_first_inline_image_is_decoded_with_its_mime_type() -> None:
    """The picture is the base64 `inlineData` part, decoded, after any text part.

    Killed by: src/uclone_x/llm/connectors/gemini.py :: raw = base64.b64decode(encoded, validate=True)
    Becomes: raw = encoded.encode()
    """
    picture = _png(640, 480)
    recorder = _Recorder(_image_reply(picture, mime="image/png"))
    connector = GeminiConnector(api_key="test-key", http_client=recorder.client())

    data, mime = await connector.generate_image("a red kite", "4:3", "gemini-2.5-flash-image")

    assert data == picture
    assert mime == "image/png"


@pytest.mark.asyncio
async def test_a_reply_without_an_image_part_raises_rather_than_returning_nothing() -> None:
    """Text only is how the model declines; that is an error, never an empty picture (P6).

    Killed by: src/uclone_x/llm/connectors/gemini.py :: raise ImageNotReturnedError(provider=_PROVIDER, model=model)
    Becomes: return b"", "image/png"
    """
    reply = {"candidates": [{"content": {"parts": [{"text": "I can't draw that."}]}}]}
    connector = GeminiConnector(api_key="test-key", http_client=_Recorder(reply).client())

    with pytest.raises(ImageNotReturnedError) as caught:
        await connector.generate_image("a red kite", "1:1", "gemini-2.5-flash-image")

    assert "without a picture" in str(caught.value)
    assert "inlineData" not in str(caught.value)


# -- `auto` ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_draws_with_the_cloud_model_on_a_connection_with_a_key() -> None:
    """With no own engine ready, `auto` uses the cloud model a keyed connection offers.

    It no longer asks what the chat model is (model-gateway §3.5): a cloud model on a
    connection with a key is enough, and without one nothing is drawn.

    Killed by: src/uclone_x/tools/builtin/image.py :: return self.pin is None and self.gemini is not None and self.gemini_model is not None
    Becomes: return False
    """
    gemini = _FakeGemini()
    drawn = await _draw(_dispatcher(_auto(gemini), local_ready=False))
    assert drawn.engine_name == "gemini"
    assert drawn.width == 1024
    assert gemini.calls[0][1:] == ("16:9", "gemini-2.5-flash-image")
    assert drawn.drawn_with == (
        "Drawn with Gemini 2.5 Flash Image in the cloud (Google) (picked automatically)."
    )

    with pytest.raises(ImageGenerationError):
        await _draw(_dispatcher(_auto(), local_ready=False))


@pytest.mark.asyncio
async def test_auto_prefers_a_ready_local_engine_over_gemini() -> None:
    """The cloud is the fallback under `auto`, not the first choice.

    Killed by: src/uclone_x/tools/builtin/image.py :: return not await self._any_local_ready(choice)
    Becomes: return True
    """
    gemini = _FakeGemini()
    drawn = await _draw(_dispatcher(_auto(gemini), local_ready=True))
    assert drawn.engine_name == "diffusers-sdxl"
    assert gemini.calls == []
    assert drawn.drawn_with is not None and drawn.drawn_with.endswith(
        "on this computer (picked automatically)."
    )


@pytest.mark.asyncio
async def test_under_auto_the_description_names_the_engine_that_then_draws(tmp_path: Path) -> None:
    """The description and the draw read one look at the local engines, so they agree.

    Killed by: src/uclone_x/tools/builtin/image.py :: parts = [self.GEMINI_BASE_DESCRIPTION if gemini else self.BASE_DESCRIPTION]
    Becomes: parts = [self.BASE_DESCRIPTION]
    Killed by: src/uclone_x/tools/builtin/image.py :: and probe[0] == self._probe_key(choice)
    Becomes: and False
    """
    gemini = _FakeGemini()
    remote, comfy, local = _locals(ready=False)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=local,
        engine_settings=lambda _own: _auto(gemini),
    )
    tool = GenerateImageTool(dispatcher=dispatcher)

    description = tool.description
    result = await tool.run(
        GenerateImageParams(prompt="a lighthouse at dawn"),
        ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path),
    )

    assert description.startswith(GenerateImageTool.GEMINI_BASE_DESCRIPTION)
    assert "never on a paid cloud service" not in description
    assert _sidecar(tmp_path, result["relative_url"])["engine"] == "gemini"
    # One look served the description and the draw.
    assert local.is_available.await_count == 1


@pytest.mark.asyncio
async def test_a_stale_look_does_not_hold_up_the_description(tmp_path: Path) -> None:
    """Under a running loop, a stale look is served and refreshed on the loop (#1769).

    The description is read inside a turn; looking at the engines there, after the cache
    has expired, would hold the loop for as long as the probes take to time out.

    Killed by: src/uclone_x/tools/builtin/image.py :: if last is None or last[0] != key:
    Becomes: if True:
    """
    gemini = _FakeGemini()
    remote, comfy, local = _locals(ready=False)
    choice = _auto(gemini)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=local,
        engine_settings=lambda _own: choice,
    )
    tool = GenerateImageTool(dispatcher=dispatcher)
    # A look long past its time, which found no local engine.
    key = dispatcher._probe_key(choice)  # pyright: ignore[reportPrivateUsage]
    dispatcher._local_probe = (key, time.monotonic() - 3600, False)  # pyright: ignore[reportPrivateUsage]

    description = tool.description

    # Answered from the last look, without looking again first.
    assert description.startswith(GenerateImageTool.GEMINI_BASE_DESCRIPTION)
    assert local.is_available.await_count == 0
    # The refresh runs on the loop, and the draw shares it rather than looking twice.
    result = await tool.run(
        GenerateImageParams(prompt="a lighthouse at dawn"),
        ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path),
    )
    assert _sidecar(tmp_path, result["relative_url"])["engine"] == "gemini"
    assert local.is_available.await_count == 1
    fresh = dispatcher._local_probe  # pyright: ignore[reportPrivateUsage]
    assert fresh is not None and time.monotonic() - fresh[1] < 60


# -- A cloud model chosen by name ----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_chosen_cloud_model_draws_there_and_records_it(tmp_path: Path) -> None:
    """A Google picture model chosen by name draws there even with a local engine ready.

    Killed by: src/uclone_x/tools/builtin/image.py :: return self._cloud_profile(choice.gemini_model or DEFAULT_IMAGE_MODEL)
    Becomes: return self._local_profile()
    Killed by: src/uclone_x/tools/builtin/image.py :: engine_name=GEMINI_ENGINE_NAME,
    Becomes: engine_name="diffusers-sdxl",
    """
    gemini = _FakeGemini()
    tool = GenerateImageTool(
        dispatcher=_dispatcher(_pinned_gemini(gemini, "gemini-test-image"), local_ready=True)
    )

    description = tool.description
    result = await tool.run(
        GenerateImageParams(prompt="a lighthouse at dawn", aspect_ratio="3:4"),
        ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path),
    )

    # The profile `load_skill` routes the prompt rules by; the description names no model.
    assert tool.active_profile().model_id == "gemini-test-image"
    assert "gemini-test-image" not in description
    assert "engine" not in result
    sidecar = _sidecar(tmp_path, result["relative_url"])
    assert sidecar["engine"] == "gemini"
    assert sidecar["drawn_with"] == result["drawn_with"]
    assert result["drawn_with"] == (
        "Drawn with gemini-test-image in the cloud (Google) (the picture model chosen in Settings)."
    )
    assert gemini.calls[0][1:] == ("3:4", "gemini-test-image")


@pytest.mark.asyncio
async def test_a_negative_gemini_cannot_use_is_reported_in_the_result(tmp_path: Path) -> None:
    """Gemini takes no negative prompt; the result says it went unused (P6, #1723).

    Killed by: src/uclone_x/tools/builtin/image.py ::                 prompt_changes=gemini_fill.changes,
    Becomes:                 prompt_changes=(),
    """
    from uclone_x.tools.builtin.media_registry import NEGATIVE_NOT_USED

    gemini = _FakeGemini()
    tool = GenerateImageTool(
        dispatcher=_dispatcher(_pinned_gemini(gemini, "gemini-test-image"), local_ready=False)
    )

    result = await tool.run(
        GenerateImageParams(prompt="a lighthouse at dawn", negative_prompt="people"),
        ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path),
    )

    assert result["prompt_changes"] == [NEGATIVE_NOT_USED]
    assert "people" not in gemini.calls[0][0]


def test_the_gemini_description_tells_the_model_where_the_notes_are() -> None:
    """The Gemini path returns prompt_changes, so its description says where to read them.

    Killed by: src/uclone_x/tools/builtin/image.py ::         "listed in the result under prompt_changes. "
    Becomes:         "listed in the result. "
    """
    tool = GenerateImageTool(
        dispatcher=_dispatcher(
            _pinned_gemini(_FakeGemini(), "gemini-test-image"), local_ready=False
        )
    )

    assert "prompt_changes" in tool.description
    assert "English" not in tool.description


@pytest.mark.asyncio
async def test_a_chosen_own_model_never_sends_a_picture_request_to_the_cloud() -> None:
    """A ComfyUI model chosen by name: a ready cloud client is never used in its place.

    Killed by: src/uclone_x/tools/builtin/image.py :: return choice.pin == "gemini"
    Becomes: return True
    """
    recorder = _Recorder(_image_reply(_png(8, 8)))
    armed = ImageEngineChoice(
        chosen="comfyui/anillustrious_v4",
        pin="comfyui",
        pinned_profile="anillustrious_v4",
        from_connections=True,
        comfyui_base_url="http://127.0.0.1:1",
        gemini=GeminiConnector(api_key="k", http_client=recorder.client()),
        gemini_model="gemini-2.5-flash-image",
    )
    with pytest.raises(ImageGenerationError):
        await _draw(_dispatcher(armed, local_ready=True))
    assert recorder.requests == []


#: A settings file holding only a Gemini connection with its key.
_GEMINI_KEY_SAVED: dict[str, Any] = {
    "connections": [{"id": "gemini", "kind": "gemini", "key": "saved-key"}]
}


def test_the_saved_gemini_key_builds_the_client_for_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key comes from the Gemini connection in settings.json, where keys are saved."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(_GEMINI_KEY_SAVED))

    choice = image_engine_choice(settings)

    assert (choice.chosen, choice.pin) == ("auto", None)
    assert choice.gemini_model == "gemini-2.5-flash-image"
    assert isinstance(choice.gemini, GeminiConnector)
    assert choice.gemini.api_key == "saved-key"
    assert image_engine_choice(tmp_path / "absent.json").gemini is None


def test_pictures_go_to_the_gemini_connections_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Google connection's saved address is the pictures' address too, in the dashboard (#1769).

    Killed by: src/uclone_x/llm/connectors/factory.py :: api_key=cloud.key, base_url=cloud.base_url, http_client=http_client
    Becomes: api_key=cloud.key, http_client=http_client
    """
    for name in ("GEMINI_BASE_URL", "LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "connections": [
                    {
                        "id": "gemini",
                        "kind": "gemini",
                        "base_url": "https://gemini-proxy.example/v1beta",
                        "key": "saved-key",
                    }
                ],
                "default_models": {"deep": "gemini/gemini-2.5-pro"},
            }
        )
    )
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    tool = mgr._tools.get("generate_image")  # pyright: ignore[reportPrivateUsage]
    assert isinstance(tool, GenerateImageTool)

    choice = tool._dispatcher.engine_choice()  # pyright: ignore[reportPrivateUsage]

    assert isinstance(choice.gemini, GeminiConnector)
    assert choice.gemini.base_url == "https://gemini-proxy.example/v1beta"


# -- Status -----------------------------------------------------------------------------------


def _report(**overrides: Any) -> ImageEngineReport:
    fields: dict[str, Any] = {
        "remote_url": None,
        "remote_alive": False,
        "comfy_url": "http://127.0.0.1:8188",
        "comfy_alive": False,
        "dependency_problems": (),
        "checkpoint": None,
    }
    fields.update(overrides)
    return ImageEngineReport(**fields)


def test_the_report_counts_the_cloud_as_the_dispatcher_does() -> None:
    """Ready and engine follow the `auto` rule and the pins, with a reason for each engine.

    Killed by: src/uclone_x/tools/builtin/image_status.py :: if self.pin is not None and pinned[self.pin] != name:
    Becomes: if False:
    """
    auto = _report(gemini_model="gemini-2.5-flash-image")
    assert auto.ready and auto.engine == "gemini"

    no_key = _report()
    assert not no_key.ready and no_key.engine == "none"
    assert no_key.engine_states()[-1] == ("gemini", False, "no_key")

    pinned = _report(
        pin="comfyui", comfy_alive=False, gemini_model="gemini-2.5-flash-image", checkpoint="x"
    )
    assert pinned.engine == "none"
    assert pinned.engine_states()[-1] == ("gemini", False, "disabled_by_setting")
    assert pinned.engine_states()[2] == ("diffusers-sdxl", False, "disabled_by_setting")

    only_gemini = _report(pin="gemini", gemini_model="gemini-x", checkpoint="x")
    assert only_gemini.engine == "gemini"


def test_the_probe_reads_the_picture_model_and_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe `ucx media status` prints includes the cloud model, from the settings file."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {**_GEMINI_KEY_SAVED, "default_models": {"image": "gemini/gemini-2.5-flash-image"}}
        )
    )
    with patch.object(LocalDiffusersImageEngine, "resolve_checkpoint", return_value=None):
        report = probe_image_engines(settings)

    assert (report.setting, report.pin) == ("gemini/gemini-2.5-flash-image", "gemini")
    assert report.gemini_model == "gemini-2.5-flash-image"
    assert report.engine == "gemini"
    assert resolve_image_choice(report)["create"] == {
        "engine": "gemini",
        "model_id": "gemini-2.5-flash-image",
        "label": "Gemini 2.5 Flash Image",
        "where": "cloud",
    }


@pytest.mark.asyncio
async def test_media_status_answers_without_the_cli_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server installed without `cli` (no `rich`) still answers the status route (#1769).

    Killed by: src/uclone_x/ui/app.py :: from uclone_x.tools.builtin.image_status import media_status_payload, probe_image_engines
    Becomes: from uclone_x.cli.commands.bootstrap import media_status_payload, probe_image_engines
    """
    monkeypatch.delenv("UCX_IMAGE_REMOTE_URL", raising=False)
    monkeypatch.delenv("UCX_MEDIA_REMOTE_URL", raising=False)
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    app = create_ui_app(static_dir=tmp_path, session_manager=mgr, storage_dir=tmp_path)
    # `None` in `sys.modules` makes an import of that name raise ImportError, as it
    # would on an install that lacks the module.
    withheld = dict.fromkeys(
        ("rich", "uclone_x.cli", "uclone_x.cli.commands", "uclone_x.cli.commands.bootstrap")
    )
    with (
        patch.dict(sys.modules, withheld),
        patch.object(ComfyUIImageEngine, "is_available", AsyncMock(return_value=False)),
        patch.object(LocalDiffusersImageEngine, "resolve_checkpoint", return_value=None),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            response = await client.get("/api/media/status")

    assert response.status_code == 200
    assert response.json()["setting"] == "auto"


@pytest.mark.asyncio
async def test_media_status_endpoint_shape(tmp_path: Path) -> None:
    """`GET /api/media/status` gives the probe's verdict and one entry per engine."""
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    app = create_ui_app(static_dir=tmp_path, session_manager=mgr, storage_dir=tmp_path)
    report = _report(gemini_model="gemini-2.5-flash-image")
    with patch(
        "uclone_x.tools.builtin.image_status.probe_image_engines", return_value=report
    ) as probe:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            response = await client.get("/api/media/status")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["engine"] == "gemini"
    assert body["setting"] == "auto"
    assert [entry["name"] for entry in body["engines"]] == [
        "remote-cuda",
        "comfyui-local",
        "diffusers-sdxl",
        "gemini",
    ]
    assert body["engines"][3] == {"name": "gemini", "ready": True, "reason_code": "ready"}
    assert body["engines"][0]["reason_code"] == "not_configured"
    assert probe.call_args.args[0] == mgr.settings_file


# -- The format a picture is saved as ------------------------------------------------------


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(agent_id="artist", session_id="s", workspace_root=tmp_path)


def test_a_picture_is_named_after_the_format_its_bytes_are() -> None:
    """The suffix follows the format; one that already names it is kept, a wrong one replaced.

    Killed by: src/uclone_x/tools/builtin/image.py :: if mime == "image/jpg":
    Becomes: if mime == "image/xxx":
    Killed by: src/uclone_x/tools/builtin/image.py :: if path.suffix.lower() in names:
    Becomes: if path.suffix.lower() in ():
    """
    assert image_extension_for("image/jpeg") == ".jpg"
    assert image_extension_for("image/jpg") == ".jpg"
    assert image_extension_for("image/webp") == ".webp"
    assert image_extension_for("image/png") == ".png"
    assert picture_path_for("artifacts/images/face.jpeg", "image/jpeg") == (
        "artifacts/images/face.jpeg"
    )
    assert (
        picture_path_for("artifacts/images/face.png", "image/jpeg") == "artifacts/images/face.jpg"
    )
    assert picture_path_for("artifacts/images/face", "image/jpeg") == "artifacts/images/face.jpg"
    assert jpeg_dimensions(b"\x00\x01\x02") is None
    assert jpeg_dimensions(_png(5, 3)) is None


@pytest.mark.parametrize(
    "data",
    [_WEBP_LOSSLESS_5X3, _WEBP_LOSSY_5X3, _WEBP_EXTENDED_5X3],
    ids=["VP8L", "VP8", "VP8X"],
)
def test_a_webp_size_is_read_from_each_first_chunk_layout(data: bytes) -> None:
    """Each WebP layout an encoder writes gives its real size, with no image library.

    Killed by: src/uclone_x/tools/builtin/image.py :: return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    Becomes: return (bits & 0x3FFF) + 0, ((bits >> 14) & 0x3FFF) + 0
    Killed by: src/uclone_x/tools/builtin/image.py :: width = int.from_bytes(data[26:28], "little") & 0x3FFF
    Becomes: width = int.from_bytes(data[24:26], "little") & 0x3FFF
    Killed by: src/uclone_x/tools/builtin/image.py :: width = 1 + int.from_bytes(data[24:27], "little")
    Becomes: width = 0 + int.from_bytes(data[24:27], "little")
    """
    assert webp_dimensions(data) == (5, 3)
    assert sniff_picture(data) == ("image/webp", (5, 3))


@pytest.mark.asyncio
async def test_a_gemini_jpeg_is_saved_as_a_jpg_that_its_sidecar_and_address_name(
    tmp_path: Path,
) -> None:
    """`gemini-3.1-flash-image` answers in JPEG; it is saved as `.jpg`, never under `.png`.

    Killed by: src/uclone_x/tools/builtin/image.py :: picture_rel = picture_path_for(clean_rel, gen_result.mime_type)
    Becomes: picture_rel = clean_rel
    Killed by: src/uclone_x/tools/builtin/image.py :: idx += seg_len
    Becomes: idx += 2
    """
    tool = _gemini_tool(_JPEG_5X3, "image/jpeg")

    result = await tool.run(GenerateImageParams(prompt="a friendly face"), _ctx(tmp_path))

    picture = _picture_rel(result["relative_url"])
    assert re.fullmatch(r"artifacts/s/images/img_[0-9a-f]{6}\.jpg", picture)
    assert (tmp_path / picture).read_bytes() == _JPEG_5X3
    assert list((tmp_path / "artifacts/images").glob("*.png")) == []
    assert result["relative_url"] == f"/api/artifacts/content?path={picture}"
    sidecar = _sidecar(tmp_path, result["relative_url"])
    assert sidecar["image_path"] == picture
    assert sidecar["mime_type"] == "image/jpeg"
    assert (sidecar["width"], sidecar["height"]) == (5, 3)


@pytest.mark.asyncio
async def test_a_batch_of_gemini_jpegs_is_saved_and_shown_as_jpgs(tmp_path: Path) -> None:
    """Every picture of a batch, its sidecar, its address and the gallery name a `.jpg`.

    Killed by: src/uclone_x/tools/builtin/image.py :: batch_rel = picture_path_for(clean_rel, gen_result.mime_type)
    Becomes: batch_rel = clean_rel
    """
    tool = _gemini_tool(_JPEG_5X3, "image/jpeg")

    result = await tool.run(GenerateImageParams(prompt="a cybernetic cat", count=2), _ctx(tmp_path))

    assert len(result["images"]) == 2
    for img in result["images"]:
        picture = _picture_rel(img["relative_url"])
        assert re.fullmatch(r"artifacts/s/images/img_[0-9a-f]{6}_[12]\.jpg", picture)
        assert (tmp_path / picture).read_bytes() == _JPEG_5X3
        assert img["relative_url"].endswith(".jpg")
        sidecar = _sidecar(tmp_path, img["relative_url"])
        assert sidecar["image_path"] == picture
        assert sidecar["mime_type"] == "image/jpeg"
    assert list((tmp_path / "artifacts/images").glob("*.png")) == []


@pytest.mark.asyncio
async def test_a_jpeg_asked_for_under_a_png_name_is_written_as_a_jpg(tmp_path: Path) -> None:
    """A chosen `face.png` that receives JPEG bytes becomes `face.jpg`; its sidecar stays.

    Killed by: src/uclone_x/tools/builtin/image.py :: return str(path.with_suffix(suffix))
    Becomes: return rel
    """
    tool = _gemini_tool(_JPEG_5X3, "image/jpeg")

    result = await tool.run(
        GenerateImageParams(prompt="a friendly face", output_path="artifacts/images/face.png"),
        _ctx(tmp_path),
    )

    assert _picture_rel(result["relative_url"]) == "artifacts/images/face.jpg"
    assert not (tmp_path / "artifacts/images/face.png").exists()
    side = json.loads((tmp_path / "artifacts/images/face.json").read_text())
    assert side["image_path"] == "artifacts/images/face.jpg"


@pytest.mark.asyncio
async def test_the_bytes_not_the_label_decide_the_format(tmp_path: Path) -> None:
    """JPEG bytes labelled `image/png` are still a JPEG: saved as `.jpg`, recorded as JPEG.

    Killed by: src/uclone_x/tools/builtin/image.py :: mime_type=real_mime,
    Becomes: mime_type=mime,
    """
    tool = _gemini_tool(_JPEG_5X3, "image/png")

    result = await tool.run(GenerateImageParams(prompt="a friendly face"), _ctx(tmp_path))

    assert _picture_rel(result["relative_url"]).endswith(".jpg")
    assert _sidecar(tmp_path, result["relative_url"])["mime_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_a_format_that_is_not_png_jpeg_or_webp_is_refused_in_plain_words(
    tmp_path: Path,
) -> None:
    """A GIF is refused in words written for a person, and nothing is written.

    Killed by: src/uclone_x/tools/builtin/image.py :: if picture is None:
    Becomes: if picture == ():
    """
    gif = (
        b"GIF89a\x05\x00\x03\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x00\x00\x00\x00\x00;"
    )
    tool = _gemini_tool(gif, "image/gif")

    result = await tool.execute({"prompt": "a friendly face"}, _ctx(tmp_path))

    assert result.success is False
    assert result.error == (
        "Google Gemini sent back a picture in a format that cannot be saved here. "
        "Try again, or choose another picture model."
    )
    assert not (tmp_path / "artifacts/images").exists() or not any(
        (tmp_path / "artifacts/images").iterdir()
    )
    with pytest.raises(ImageEngineRefusal):
        await tool.run(GenerateImageParams(prompt="a friendly face"), _ctx(tmp_path))


def _personas_refusal(path: str) -> str:
    return (
        f"'{path}' is where clones' definitions and pictures are kept, so it "
        "was not written. A clone changes its own picture with set_avatar, and a "
        "person edits a clone from its settings. Reading the file is still allowed."
    )


def _link_to_an_avatar(tmp_path: Path, link: str) -> Path:
    """``link`` in the workspace, pointing at a clone's picture in `.uclone/personas/`."""
    avatar = tmp_path / ".uclone/personas/artist.jpg"
    avatar.parent.mkdir(parents=True)
    avatar.write_bytes(b"the artist's own picture")
    (tmp_path / link).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / link).symlink_to(avatar)
    return avatar


@pytest.mark.asyncio
async def test_the_renamed_picture_is_checked_again_before_it_is_written(tmp_path: Path) -> None:
    """`face.png` passes; its JPEG is to go to `face.jpg`, a link to a clone's picture: refused.

    Killed by: src/uclone_x/tools/builtin/image.py :: resolve_dest = self.resolve_write_path
    Becomes: resolve_dest = self.resolve_safe_path
    """
    avatar = _link_to_an_avatar(tmp_path, "artifacts/images/face.jpg")
    tool = _gemini_tool(_JPEG_5X3, "image/jpeg")

    result = await tool.execute(
        {"prompt": "a friendly face", "output_path": "artifacts/images/face.png"}, _ctx(tmp_path)
    )

    assert (result.success, result.error) == (False, _personas_refusal("artifacts/images/face.jpg"))
    assert avatar.read_bytes() == b"the artist's own picture"


@pytest.mark.asyncio
async def test_each_renamed_batch_picture_is_checked_again_before_it_is_written(
    tmp_path: Path,
) -> None:
    """The batch's `face_1.png` passes; `face_1.jpg` links to a clone's picture: refused.

    Killed by: src/uclone_x/tools/builtin/image.py :: resolve_image = self.resolve_write_path
    Becomes: resolve_image = self.resolve_safe_path
    """
    avatar = _link_to_an_avatar(tmp_path, "artifacts/images/face_1.jpg")
    tool = _gemini_tool(_JPEG_5X3, "image/jpeg")

    result = await tool.execute(
        {"prompt": "a friendly face", "output_path": "artifacts/images/face.png", "count": 2},
        _ctx(tmp_path),
    )

    assert (result.success, result.error) == (
        False,
        _personas_refusal("artifacts/images/face_1.jpg"),
    )
    assert avatar.read_bytes() == b"the artist's own picture"


@pytest.mark.parametrize(
    ("asked", "picture", "sidecar"),
    [
        ("art/face", "art/face.png", "art/face.json"),
        ("art/notes.txt", "art/notes.txt.png", "art/notes.txt.json"),
        ("art/face.jpg", "art/face.png", "art/face.json"),
    ],
)
@pytest.mark.asyncio
async def test_a_chosen_name_for_a_png_gets_a_png_suffix_and_a_sidecar_beside_it(
    tmp_path: Path, asked: str, picture: str, sidecar: str
) -> None:
    """PNG bytes too are saved under `.png`, and the sidecar is named after the final picture.

    Killed by: src/uclone_x/tools/builtin/image.py :: meta_rel = sidecar_path_for(picture_rel)
    Becomes: meta_rel = sidecar_path_for(clean_rel)
    """
    tool = _gemini_tool(_png(5, 3), "image/png")

    result = await tool.run(
        GenerateImageParams(prompt="a friendly face", output_path=asked), _ctx(tmp_path)
    )

    assert _picture_rel(result["relative_url"]) == picture
    assert sidecar_path_for(picture) == sidecar
    assert json.loads((tmp_path / sidecar).read_text())["image_path"] == picture
    written = sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / "art").glob("*.json"))
    assert written == [sidecar]
    assert not (tmp_path / asked).exists() or asked == picture


@pytest.mark.asyncio
async def test_a_batch_sidecar_is_named_after_its_final_picture(tmp_path: Path) -> None:
    """`notes.txt` in a batch becomes `notes_1.txt.png`, with `notes_1.txt.json` beside it.

    Killed by: src/uclone_x/tools/builtin/image.py :: batch_meta_rel = sidecar_path_for(batch_rel)
    Becomes: batch_meta_rel = sidecar_path_for(clean_rel)
    """
    tool = _gemini_tool(_png(5, 3), "image/png")

    result = await tool.run(
        GenerateImageParams(prompt="a friendly face", output_path="art/notes.txt", count=2),
        _ctx(tmp_path),
    )

    pictures = ["art/notes_1.txt.png", "art/notes_2.txt.png"]
    sidecars = ["art/notes_1.txt.json", "art/notes_2.txt.json"]
    assert linked_paths(result) == pictures
    assert "meta_paths" not in result
    for picture, sidecar in zip(pictures, sidecars, strict=True):
        assert json.loads((tmp_path / sidecar).read_text())["image_path"] == picture
    written = sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / "art").glob("*.json"))
    assert written == sidecars


@pytest.mark.parametrize(
    "asked",
    [
        "artifacts/images/",
        "artifacts/images/.png",
        "artifacts/images/.",
        "artifacts/images/..",
        "a/.",
    ],
)
@pytest.mark.asyncio
async def test_a_name_with_no_file_in_it_is_refused_before_drawing(
    tmp_path: Path, asked: str
) -> None:
    r"""A folder, or a bare `.png`, names no file: refused in plain words, nothing drawn.

    Killed by: src/uclone_x/tools/builtin/image.py :: return rel.endswith(("/", "\\")) or last in _NO_FILE_NAMES
    Becomes: return False
    """
    gemini = AsyncMock()
    choice = _pinned_gemini(gemini, "gemini-3.1-flash-image")
    tool = GenerateImageTool(dispatcher=_dispatcher(choice, local_ready=False))
    gemini.generate_image.return_value = (_JPEG_5X3, "image/jpeg")

    result = await tool.execute({"prompt": "a friendly face", "output_path": asked}, _ctx(tmp_path))

    assert (result.success, result.error) == (
        False,
        f"'{asked}' has no file name, so no picture was made. "
        "Give a file name, such as 'artifacts/images/face.png'.",
    )
    gemini.generate_image.assert_not_called()


@pytest.mark.asyncio
async def test_a_batch_name_with_no_file_in_it_is_refused_before_drawing(
    tmp_path: Path,
) -> None:
    """A batch output path naming no file is refused before drawing."""
    gemini = AsyncMock()
    choice = _pinned_gemini(gemini, "gemini-3.1-flash-image")
    tool = GenerateImageTool(dispatcher=_dispatcher(choice, local_ready=False))
    gemini.generate_image.return_value = (_JPEG_5X3, "image/jpeg")

    result = await tool.execute(
        {"prompt": "a friendly face", "output_path": "artifacts/images/.", "count": 2},
        _ctx(tmp_path),
    )

    assert (result.success, result.error) == (
        False,
        "'artifacts/images/.' has no file name, so no picture was made. "
        "Give a file name, such as 'artifacts/images/face.png'.",
    )
    gemini.generate_image.assert_not_called()


# -- Where each engine draws -------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine", "url", "tunnel_port", "where"),
    [
        ("remote-cuda", None, None, "gpu_server"),
        ("gemini", None, None, "cloud"),
        ("diffusers-sdxl", None, None, "this_computer"),
        ("comfyui-local", "http://127.0.0.1:8188", None, "this_computer"),
        ("comfyui-local", "http://[::1]:8188", None, "this_computer"),
        ("comfyui-local", "http://localhost:8188", None, "this_computer"),
        ("comfyui-local", "http://LOCALHOST:8188", None, "this_computer"),
        ("comfyui-local", "http://192.168.1.20:8188", None, "gpu_server"),
        ("comfyui-local", "http://gpu-box.lan:8188", None, "gpu_server"),
        # The connected tunnel's own port is the GPU server; another loopback port is not.
        ("comfyui-local", "http://127.0.0.1:8188", 8188, "gpu_server"),
        ("comfyui-local", "http://[::1]:8190", 8190, "gpu_server"),
        ("comfyui-local", "http://localhost:8190", 8188, "this_computer"),
        # No port in the address is the scheme's default.
        ("comfyui-local", "http://127.0.0.1", 80, "gpu_server"),
        # A tunnel made by hand (`ssh -L 8188:...`) is not the connected one.
        ("comfyui-local", "http://127.0.0.1:8188", None, "this_computer"),
    ],
)
def test_where_each_engine_draws(
    engine: Any, url: str | None, tunnel_port: int | None, where: ImageWhere
) -> None:
    """One table for every engine (design §3.1).

    Killed by: src/uclone_x/tools/builtin/image.py :: if gpu_tunnel_comfy_port is not None and port == gpu_tunnel_comfy_port:
    Becomes: if False:
    Killed by: src/uclone_x/tools/builtin/image.py :: return ipaddress.ip_address(host).is_loopback
    Becomes: return host == "127.0.0.1"
    Killed by: src/uclone_x/tools/builtin/image.py :: if host.lower() == "localhost":
    Becomes: if False:
    """
    assert image_where(engine, url, tunnel_port) == where


def test_a_comfyui_address_with_no_host_is_an_error() -> None:
    """An address naming no host is a caller's mistake, not a place to guess."""
    with pytest.raises(ValueError, match="names no host"):
        image_where("comfyui-local", "not a url")


def test_the_tunnel_status_names_its_comfyui_port_only_while_connected() -> None:
    """The ui head's tunnel port comes from the connected tunnel's ComfyUI forward.

    Killed by: src/uclone_x/core/remote_worker.py :: if mapping.service_name == COMFYUI_SERVICE_NAME:
    Becomes: if True:
    """
    mappings = [PortMapping("ollama", 11434, 11435), PortMapping("comfyui", 8188, 8190)]
    assert TunnelSessionStatus("gpu", True, mappings=mappings).comfyui_local_port == 8190
    assert TunnelSessionStatus("gpu", False, mappings=mappings).comfyui_local_port is None
    assert TunnelSessionStatus("gpu", True, mappings=mappings[:1]).comfyui_local_port is None


def test_the_dashboard_reads_the_tunnel_port_from_the_tunnel(tmp_path: Path) -> None:
    """The dashboard's picture settings carry the connected tunnel's ComfyUI port.

    Killed by: src/uclone_x/ui/app.py :: session_mgr.bind_gpu_tunnel(lambda: tunnel_manager.get_status().comfyui_local_port)
    Becomes: pass
    """
    status = TunnelSessionStatus("gpu", True, mappings=[PortMapping("comfyui", 8188, 8190)])
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    assert mgr.gpu_tunnel_comfy_port() is None
    with patch.object(SSHTunnelManager, "get_status", return_value=status):
        create_ui_app(static_dir=tmp_path, session_manager=mgr, storage_dir=tmp_path)
        assert mgr.gpu_tunnel_comfy_port() == 8190


# -- Results carry where and which model ------------------------------------------------


@pytest.mark.asyncio
async def test_a_gemini_picture_records_the_cloud_and_its_model(tmp_path: Path) -> None:
    """Result and sidecar name where the picture was drawn and the model, single and batch.

    Killed by: src/uclone_x/tools/builtin/image.py :: model_id=cloud_model,
    Becomes: model_id=None,
    """
    choice = _pinned_gemini(_FakeGemini())
    tool = GenerateImageTool(dispatcher=_dispatcher(choice, local_ready=False))
    ctx = ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path)

    single = await tool.run(GenerateImageParams(prompt="a boat"), ctx)
    assert (single["where"], single["model_id"]) == ("cloud", "gemini-2.5-flash-image")
    sidecar = _sidecar(tmp_path, single["relative_url"])
    assert (sidecar["where"], sidecar["model_id"]) == ("cloud", "gemini-2.5-flash-image")

    # A batch says what its pictures share once, at the top (#2013).
    batch = await tool.run(GenerateImageParams(prompt="a boat", count=2), ctx)
    assert (batch["where"], batch["model_id"]) == ("cloud", "gemini-2.5-flash-image")
    for img in batch["images"]:
        entry = _sidecar(tmp_path, img["relative_url"])
        assert (entry["where"], entry["model_id"]) == ("cloud", "gemini-2.5-flash-image")


def _comfy_dispatcher(tunnel_port: int | None) -> ImagePipelineDispatcher:
    """A dispatcher whose ComfyUI daemon on 127.0.0.1:8188 answers and loads a known model."""
    remote, comfy, local = _locals(ready=False)
    comfy.is_available.return_value = True
    comfy.checkpoint = "anillustrious_v4.safetensors"
    comfy.generate.return_value = ImageGenerationResult(
        image_bytes=b"comfy",
        seed=1,
        engine_name="comfyui-local",
        device_info="ComfyUI",
        duration_seconds=0.1,
        width=1024,
        height=1024,
    )
    choice = ImageEngineChoice(gpu_tunnel_comfy_port=tunnel_port)
    return ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=local,
        engine_settings=lambda _own: choice,
    )


@pytest.mark.asyncio
async def test_a_comfyui_picture_records_its_place_and_registered_model() -> None:
    """ComfyUI on the tunnel's port is the GPU server; elsewhere on loopback, this computer."""
    tunnelled = await _draw(_comfy_dispatcher(8188))
    assert (tunnelled.where, tunnelled.model_id) == ("gpu_server", "anillustrious_v4")

    here = await _draw(_comfy_dispatcher(None))
    assert (here.where, here.model_id) == ("this_computer", "anillustrious_v4")


@pytest.mark.asyncio
async def test_an_unregistered_checkpoint_and_the_remote_worker_report_no_model() -> None:
    """The generic fallback is no model to name, and the remote worker reports none.

    Killed by: src/uclone_x/tools/builtin/image.py :: return profile.model_id if is_registered_profile(profile) else None
    Becomes: return profile.model_id
    """
    choice = ImageEngineChoice()
    remote, comfy, local = _locals(ready=True)
    local.resolve_checkpoint.return_value = "/models/mystery_mix.safetensors"
    in_process = await _draw(
        ImagePipelineDispatcher(
            remote_engine=remote,
            comfy_engine=comfy,
            local_engine=local,
            engine_settings=lambda _own: choice,
        )
    )
    assert (in_process.where, in_process.model_id) == ("this_computer", None)

    remote, comfy, local = _locals(ready=False)
    remote.is_available.return_value = True
    remote.generate.return_value = ImageGenerationResult(
        image_bytes=b"remote",
        seed=1,
        engine_name="remote-cuda",
        device_info="cuda",
        duration_seconds=0.1,
        width=1024,
        height=1024,
    )
    worker = await _draw(
        ImagePipelineDispatcher(
            remote_engine=remote,
            comfy_engine=comfy,
            local_engine=local,
            engine_settings=lambda _own: choice,
        )
    )
    assert (worker.where, worker.model_id) == ("gpu_server", None)


# -- The resolved model in the status ----------------------------------------------------


def test_the_status_resolves_the_model_and_the_place() -> None:
    """`resolved.create` names the engine, the model, a display label and where.

    Killed by: src/uclone_x/tools/builtin/image_status.py :: where = image_where("comfyui-local", report.comfy_url, report.gpu_tunnel_comfy_port)
    Becomes: where = image_where("comfyui-local", report.comfy_url)
    """
    known = KnownModel("anillustrious_v4", "Illustrious-XL v4 (Anime SDXL)")
    comfy = resolve_image_choice(
        _report(comfy_alive=True, comfy_model=known, gpu_tunnel_comfy_port=8188)
    )
    assert comfy == {
        "create": {
            "engine": "comfyui-local",
            "model_id": "anillustrious_v4",
            "label": "Illustrious-XL v4 (Anime SDXL)",
            "where": "gpu_server",
        },
        "reason_code": None,
        "refusal": None,
    }

    gemini = resolve_image_choice(_report(gemini_model="gemini-2.5-flash-image"))
    assert gemini["create"] is not None
    assert gemini["create"]["where"] == "cloud"
    assert gemini["create"]["model_id"] == "gemini-2.5-flash-image"

    remote = resolve_image_choice(_report(remote_url="http://gpu:9000", remote_alive=True))
    assert remote["create"] == {
        "engine": "remote-cuda",
        "model_id": None,
        "label": None,
        "where": "gpu_server",
    }


@pytest.mark.parametrize(
    ("pin", "reason"),
    [(None, "no_image_model"), ("comfyui", "chosen_not_ready"), ("gemini", "chosen_not_ready")],
)
def test_nothing_resolves_with_a_reason_for_auto_and_for_a_chosen_model(
    pin: str | None, reason: str
) -> None:
    """With nothing to draw, ``create`` is null, and a chosen model says it is not ready.

    Killed by: src/uclone_x/tools/builtin/image_status.py :: reason: ResolveReasonCode = "no_image_model" if report.pin is None else "chosen_not_ready"
    Becomes: reason: ResolveReasonCode = "no_image_model"
    """
    assert resolve_image_choice(_report(pin=pin)) == {
        "create": None,
        "reason_code": reason,
        "refusal": None,
    }


@pytest.mark.asyncio
async def test_the_endpoint_answers_the_resolved_model(tmp_path: Path) -> None:
    """`/api/media/status` carries `resolved`, the same object the CLI's `--json` prints."""
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    app = create_ui_app(static_dir=tmp_path, session_manager=mgr, storage_dir=tmp_path)
    known = KnownModel("anillustrious_v4", "Illustrious-XL v4 (Anime SDXL)")
    report = _report(comfy_alive=True, comfy_model=known)
    with patch("uclone_x.tools.builtin.image_status.probe_image_engines", return_value=report):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            response = await client.get("/api/media/status")

    body = response.json()
    assert body == json.loads(json.dumps(media_status_payload(report)))
    assert body["resolved"]["create"]["label"] == "Illustrious-XL v4 (Anime SDXL)"
    assert body["resolved"]["create"]["where"] == "this_computer"


def test_the_probe_names_the_registered_model_comfyui_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The label is the profile's display name, never the checkpoint file or engine name."""
    monkeypatch.delenv("UCX_COMFYUI_URL", raising=False)
    monkeypatch.setenv("UCX_COMFYUI_CHECKPOINT", "anillustrious_v4.safetensors")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "connections": [
                    {"id": "comfyui", "kind": "comfyui", "base_url": "http://127.0.0.1:1"}
                ]
            }
        )
    )
    with (
        patch.object(ComfyUIImageEngine, "is_available", AsyncMock(return_value=True)),
        patch.object(LocalDiffusersImageEngine, "resolve_checkpoint", return_value=None),
    ):
        report = probe_image_engines(settings)

    assert report.comfy_model == KnownModel("anillustrious_v4", "Illustrious-XL v4 (Anime SDXL)")


# -- Model profiles ---------------------------------------------------------------------


def test_profiles_say_what_they_can_do_and_gemini_is_its_own_engine() -> None:
    """A profile draws by default; a YAML list of capabilities is accepted.

    Killed by: src/uclone_x/tools/builtin/image.py :: engine_type="gemini",
    Becomes: engine_type="auto",
    """
    plain = ModelProfile(model_id="m", display_name="M", family=PromptFamily.GENERIC)
    assert plain.capabilities == ("create",)
    both = ModelProfile.model_validate(
        {
            "model_id": "m",
            "display_name": "M",
            "family": "generic",
            "capabilities": ["create", "edit"],
        }
    )
    assert both.capabilities == ("create", "edit")
    assert gemini_profile("gemini-2.5-flash-image").engine_type == "gemini"
