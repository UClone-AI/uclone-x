"""The Gemini image engine and the `image_engine` setting that decides when it draws.

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

from uclone_x.errors import ImageNotReturnedError
from uclone_x.llm.connectors.factory import image_engine_choice
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.tools.builtin.image import (
    ComfyUIImageEngine,
    GenerateImageParams,
    GenerateImageTool,
    ImageEngineChoice,
    ImageEngineRefusal,
    ImageGenerationError,
    ImageGenerationResult,
    ImagePipelineDispatcher,
    LocalDiffusersImageEngine,
    RemoteCudaImageEngine,
    image_extension_for,
    jpeg_dimensions,
    picture_path_for,
    sniff_picture,
    webp_dimensions,
)
from uclone_x.tools.builtin.image_status import ImageEngineReport, probe_image_engines
from uclone_x.tools.models import ToolContext
from uclone_x.ui.app import AgentSessionManager, create_ui_app


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
        engine_settings=lambda: choice,
    )


class _FakeGeminiSending:
    """A `GeminiImageClient` that answers every request with ``data`` labelled ``mime``."""

    def __init__(self, data: bytes, mime: str) -> None:
        self._reply = (data, mime)

    async def generate_image(self, prompt: str, aspect_ratio: str, model: str) -> tuple[bytes, str]:
        return self._reply


def _gemini_tool(data: bytes, mime: str) -> GenerateImageTool:
    gemini = _FakeGeminiSending(data, mime)
    choice = ImageEngineChoice(setting="gemini", model="gemini-3.1-flash-image", gemini=gemini)
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
async def test_auto_draws_with_gemini_only_for_a_gemini_chat_with_a_key() -> None:
    """With no local engine ready, `auto` needs both a Gemini chat and a key to use Gemini.

    Killed by: src/uclone_x/tools/builtin/image.py :: return self.chat_provider == GEMINI_ENGINE_NAME and self.gemini is not None
    Becomes: return self.gemini is not None
    """
    gemini = _FakeGemini()
    drawn = await _draw(
        _dispatcher(ImageEngineChoice(chat_provider="gemini", gemini=gemini), local_ready=False)
    )
    assert drawn.engine_name == "gemini"
    assert drawn.width == 1024
    assert gemini.calls[0][1] == "16:9"

    other_chat = _FakeGemini()
    with pytest.raises(ImageGenerationError):
        await _draw(
            _dispatcher(
                ImageEngineChoice(chat_provider="openai", gemini=other_chat), local_ready=False
            )
        )
    assert other_chat.calls == []

    with pytest.raises(ImageGenerationError):
        await _draw(_dispatcher(ImageEngineChoice(chat_provider="gemini"), local_ready=False))


@pytest.mark.asyncio
async def test_auto_prefers_a_ready_local_engine_over_gemini() -> None:
    """Gemini is the fallback under `auto`, not the first choice.

    Killed by: src/uclone_x/tools/builtin/image.py :: return not await self._any_local_ready()
    Becomes: return True
    """
    gemini = _FakeGemini()
    drawn = await _draw(
        _dispatcher(ImageEngineChoice(chat_provider="gemini", gemini=gemini), local_ready=True)
    )
    assert drawn.engine_name == "diffusers-sdxl"
    assert gemini.calls == []


@pytest.mark.asyncio
async def test_under_auto_the_description_names_the_engine_that_then_draws(tmp_path: Path) -> None:
    """The description and the draw read one look at the local engines, so they agree.

    Killed by: src/uclone_x/tools/builtin/image.py :: parts = [self.GEMINI_BASE_DESCRIPTION if gemini else self.BASE_DESCRIPTION]
    Becomes: parts = [self.BASE_DESCRIPTION]
    Killed by: src/uclone_x/tools/builtin/image.py :: if probe is not None and time.monotonic() - probe[0] < LOCAL_PROBE_TTL_SECONDS:
    Becomes: if False:
    """
    gemini = _FakeGemini()
    remote, comfy, local = _locals(ready=False)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=local,
        engine_settings=lambda: ImageEngineChoice(chat_provider="gemini", gemini=gemini),
    )
    tool = GenerateImageTool(dispatcher=dispatcher)

    description = tool.description
    result = await tool.run(
        GenerateImageParams(prompt="a lighthouse at dawn"),
        ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path),
    )

    assert description.startswith(GenerateImageTool.GEMINI_BASE_DESCRIPTION)
    assert "never on a paid cloud service" not in description
    assert result["engine"] == "gemini"
    # One look served the description and the draw.
    assert local.is_available.await_count == 1


@pytest.mark.asyncio
async def test_a_stale_look_does_not_hold_up_the_description(tmp_path: Path) -> None:
    """Under a running loop, a stale look is served and refreshed on the loop (#1769).

    The description is read inside a turn; looking at the engines there, after the cache
    has expired, would hold the loop for as long as the probes take to time out.

    Killed by: src/uclone_x/tools/builtin/image.py :: if last is None:
    Becomes: if True:
    """
    gemini = _FakeGemini()
    remote, comfy, local = _locals(ready=False)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=local,
        engine_settings=lambda: ImageEngineChoice(chat_provider="gemini", gemini=gemini),
    )
    tool = GenerateImageTool(dispatcher=dispatcher)
    # A look long past its time, which found no local engine.
    dispatcher._local_probe = (time.monotonic() - 3600, False)  # pyright: ignore[reportPrivateUsage]

    description = tool.description

    # Answered from the last look, without looking again first.
    assert description.startswith(GenerateImageTool.GEMINI_BASE_DESCRIPTION)
    assert local.is_available.await_count == 0
    # The refresh runs on the loop, and the draw shares it rather than looking twice.
    result = await tool.run(
        GenerateImageParams(prompt="a lighthouse at dawn"),
        ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path),
    )
    assert result["engine"] == "gemini"
    assert local.is_available.await_count == 1
    fresh = dispatcher._local_probe  # pyright: ignore[reportPrivateUsage]
    assert fresh is not None and time.monotonic() - fresh[0] < 60


# -- `gemini` and `local` -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_setting_draws_with_gemini_and_records_it(tmp_path: Path) -> None:
    """`gemini` draws there even with a local engine ready, and the sidecar says so.

    Killed by: src/uclone_x/tools/builtin/image.py :: return gemini_profile(choice.model)
    Becomes: return self._local_profile()
    Killed by: src/uclone_x/tools/builtin/image.py :: engine_name=GEMINI_ENGINE_NAME,
    Becomes: engine_name="diffusers-sdxl",
    """
    gemini = _FakeGemini()
    choice = ImageEngineChoice(setting="gemini", model="gemini-test-image", gemini=gemini)
    tool = GenerateImageTool(dispatcher=_dispatcher(choice, local_ready=True))

    description = tool.description
    result = await tool.run(
        GenerateImageParams(prompt="a lighthouse at dawn", aspect_ratio="3:4"),
        ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path),
    )

    # The profile `load_skill` routes the prompt rules by; the description names no model.
    assert tool.active_profile().model_id == "gemini-test-image"
    assert "gemini-test-image" not in description
    assert result["engine"] == "gemini"
    sidecar = json.loads((tmp_path / result["meta_path"]).read_text())
    assert sidecar["engine"] == "gemini"
    assert gemini.calls[0][1:] == ("3:4", "gemini-test-image")


@pytest.mark.asyncio
async def test_a_negative_gemini_cannot_use_is_reported_in_the_result(tmp_path: Path) -> None:
    """Gemini takes no negative prompt; the result says it went unused (P6, #1723).

    Killed by: src/uclone_x/tools/builtin/image.py ::             return replace(drawn, prompt_changes=gemini_fill.changes)
    Becomes:             return drawn
    """
    from uclone_x.tools.builtin.media_registry import NEGATIVE_NOT_USED

    gemini = _FakeGemini()
    choice = ImageEngineChoice(setting="gemini", model="gemini-test-image", gemini=gemini)
    tool = GenerateImageTool(dispatcher=_dispatcher(choice, local_ready=False))

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
            ImageEngineChoice(setting="gemini", model="gemini-test-image", gemini=_FakeGemini()),
            local_ready=False,
        )
    )

    assert "prompt_changes" in tool.description
    assert "English" not in tool.description


@pytest.mark.asyncio
async def test_local_setting_never_sends_a_picture_request(tmp_path: Path) -> None:
    """`local` builds no Gemini client from a saved key, and draws nothing over the network.

    Killed by: src/uclone_x/llm/connectors/factory.py :: if setting == "local" or not gemini_key_available(data):
    Becomes: if not gemini_key_available(data):
    Killed by: src/uclone_x/tools/builtin/image.py :: return choice.setting == GEMINI_ENGINE_NAME
    Becomes: return True
    """
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"image_engine": "local", "llm_api_keys": {"gemini": "saved-key"}})
    )
    recorder = _Recorder(_image_reply(_png(8, 8)))

    choice = image_engine_choice(settings, "gemini", http_client=recorder.client())
    assert choice.setting == "local"
    assert choice.gemini is None

    # Even handed a client, the dispatcher under `local` does not use it.
    armed = ImageEngineChoice(
        setting="local",
        chat_provider="gemini",
        gemini=GeminiConnector(api_key="k", http_client=recorder.client()),
    )
    with pytest.raises(ImageGenerationError):
        await _draw(_dispatcher(armed, local_ready=False))
    assert recorder.requests == []


def test_the_saved_gemini_key_builds_the_client_for_auto(tmp_path: Path) -> None:
    """The key comes from settings.json's `llm_api_keys`, the one place keys are saved."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"llm_api_keys": {"gemini": "saved-key"}}))

    choice = image_engine_choice(settings, "google")

    assert choice.setting == "auto"
    assert choice.model == "gemini-2.5-flash-image"
    assert choice.chat_provider == "gemini"
    assert isinstance(choice.gemini, GeminiConnector)
    assert choice.gemini.api_key == "saved-key"
    assert image_engine_choice(tmp_path / "absent.json", "gemini").gemini is None


def test_pictures_go_to_the_gemini_address_the_chat_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved Gemini base URL is the pictures' address too, in the dashboard (#1769).

    Killed by: src/uclone_x/llm/connectors/factory.py :: api_key=resolve_api_key("gemini", None, data), base_url=base_url, http_client=http_client
    Becomes: api_key=resolve_api_key("gemini", None, data), http_client=http_client
    Killed by: src/uclone_x/ui/app.py :: self.gemini_base_url_in_effect,
    Becomes: None,
    """
    for name in ("GEMINI_BASE_URL", "LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "llm_provider": "gemini",
                "llm_base_url": "https://gemini-proxy.example/v1beta",
                "llm_api_keys": {"gemini": "saved-key"},
            }
        )
    )
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    tool = mgr._tools.get("generate_image")  # pyright: ignore[reportPrivateUsage]
    assert isinstance(tool, GenerateImageTool)

    choice = tool._dispatcher.engine_choice()  # pyright: ignore[reportPrivateUsage]

    assert isinstance(choice.gemini, GeminiConnector)
    assert choice.gemini.base_url == "https://gemini-proxy.example/v1beta"


def test_a_saved_address_with_no_saved_provider_is_not_sent_the_gemini_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An address saved for no provider may be another server's; pictures keep Google's.

    Killed by: src/uclone_x/ui/app.py :: or same_provider(self._configured_provider, "gemini")
    Becomes: or True
    """
    for name in ("GEMINI_BASE_URL", "LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "llm_base_url": "http://127.0.0.1:11434",
                "image_engine": "gemini",
                "llm_api_keys": {"gemini": "saved-key"},
            }
        )
    )
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)

    assert mgr.gemini_base_url_in_effect() is None


# -- Settings ---------------------------------------------------------------------------------


def test_settings_accept_known_values_and_refuse_unknown_ones(tmp_path: Path) -> None:
    """Both fields are validated before anything is written; a refusal changes nothing.

    Killed by: src/uclone_x/tools/builtin/image.py :: if value == setting:
    Becomes: if True:
    Killed by: src/uclone_x/tools/builtin/image.py :: if isinstance(value, str) and _IMAGE_MODEL_ID.fullmatch(value):
    Becomes: if isinstance(value, str):
    Killed by: src/uclone_x/ui/app.py :: changes.update(image_changes)
    Becomes: pass
    """
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    assert mgr.get_settings()["image_engine"] == "auto"
    assert mgr.get_settings()["image_model"] == "gemini-2.5-flash-image"

    updated = mgr.update_settings(image_engine="gemini", image_model="gemini-3-pro-image")
    assert updated["image_engine"] == "gemini"
    assert updated["image_model"] == "gemini-3-pro-image"
    saved = json.loads(mgr.settings_file.read_text())
    assert saved["image_engine"] == "gemini"

    with pytest.raises(ValueError, match="auto, local or gemini"):
        mgr.update_settings(image_engine="cloud", ui_language="en")
    with pytest.raises(ValueError, match="Gemini model id"):
        mgr.update_settings(image_model="../../v1/files")
    assert json.loads(mgr.settings_file.read_text()) == saved


@pytest.mark.asyncio
async def test_the_settings_endpoint_refuses_an_unknown_engine_plainly(tmp_path: Path) -> None:
    """`POST /api/settings` answers 400 with the refusal, and takes a known value."""
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    app = create_ui_app(static_dir=tmp_path, session_manager=mgr, storage_dir=tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        bad = await client.post("/api/settings", json={"image_engine": "cloud"})
        good = await client.post("/api/settings", json={"image_engine": "local"})
        read = await client.get("/api/settings")

    assert bad.status_code == 400
    assert "auto, local or gemini" in bad.json()["detail"]
    assert good.status_code == 200
    assert read.json()["image_engine"] == "local"


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


def test_the_report_counts_gemini_as_the_dispatcher_does() -> None:
    """Ready and engine follow the `image_engine` rule, with a reason for each engine.

    Killed by: src/uclone_x/tools/builtin/image_status.py :: return self.local_ready or self.gemini_ready
    Becomes: return self.local_ready
    Killed by: src/uclone_x/tools/builtin/image_status.py :: gemini = (False, "chat_provider_not_gemini")
    Becomes: gemini = (True, "ready")
    """
    auto = _report(chat_provider="gemini", gemini_key=True)
    assert auto.ready and auto.engine == "gemini"

    other_chat = _report(chat_provider="openai", gemini_key=True)
    assert not other_chat.ready and other_chat.engine == "none"
    assert other_chat.engine_states()[-1] == ("gemini", False, "chat_provider_not_gemini")

    local = _report(image_engine="local", chat_provider="gemini", gemini_key=True, checkpoint="x")
    assert local.engine == "diffusers-sdxl"
    assert local.engine_states()[-1] == ("gemini", False, "disabled_by_setting")

    only_gemini = _report(image_engine="gemini", gemini_key=True, checkpoint="x")
    assert only_gemini.engine == "gemini"
    assert only_gemini.engine_states()[2] == ("diffusers-sdxl", False, "disabled_by_setting")


def test_the_probe_reads_the_picture_settings_and_the_key(tmp_path: Path) -> None:
    """The probe `ucx media status` prints includes the Gemini engine, from the settings file."""
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"image_engine": "gemini", "llm_api_keys": {"gemini": "saved-key"}})
    )
    with (
        patch.object(ComfyUIImageEngine, "is_available", AsyncMock(return_value=False)),
        patch.object(LocalDiffusersImageEngine, "resolve_checkpoint", return_value=None),
    ):
        report = probe_image_engines(settings, "gemini")

    assert report.image_engine == "gemini"
    assert report.gemini_key
    assert report.engine == "gemini"


@pytest.mark.asyncio
async def test_media_status_answers_without_the_cli_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server installed without `cli` (no `rich`) still answers the status route (#1769).

    Killed by: src/uclone_x/ui/app.py :: from uclone_x.tools.builtin.image_status import probe_image_engines
    Becomes: from uclone_x.cli.commands.bootstrap import probe_image_engines
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
    report = _report(chat_provider="gemini", gemini_key=True)
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

    assert re.fullmatch(r"artifacts/images/img_[0-9a-f]{6}\.jpg", result["path"])
    assert (tmp_path / result["path"]).read_bytes() == _JPEG_5X3
    assert list((tmp_path / "artifacts/images").glob("*.png")) == []
    assert (result["width"], result["height"]) == (5, 3)
    assert result["mime_type"] == "image/jpeg"
    assert result["relative_url"] == f"/api/artifacts/content?path={result['path']}"
    sidecar = json.loads((tmp_path / result["meta_path"]).read_text())
    assert sidecar["image_path"] == result["path"]
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
        assert re.fullmatch(r"artifacts/images/img_[0-9a-f]{6}_[12]\.jpg", img["path"])
        assert (tmp_path / img["path"]).read_bytes() == _JPEG_5X3
        assert img["relative_url"] in result["markdown_gallery"]
        assert img["relative_url"].endswith(".jpg")
        sidecar = json.loads((tmp_path / img["meta_path"]).read_text())
        assert sidecar["image_path"] == img["path"]
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

    assert result["path"] == "artifacts/images/face.jpg"
    assert not (tmp_path / "artifacts/images/face.png").exists()
    assert result["meta_path"] == "artifacts/images/face.json"


@pytest.mark.asyncio
async def test_the_bytes_not_the_label_decide_the_format(tmp_path: Path) -> None:
    """JPEG bytes labelled `image/png` are still a JPEG: saved as `.jpg`, recorded as JPEG.

    Killed by: src/uclone_x/tools/builtin/image.py :: mime_type=real_mime,
    Becomes: mime_type=mime,
    """
    tool = _gemini_tool(_JPEG_5X3, "image/png")

    result = await tool.run(GenerateImageParams(prompt="a friendly face"), _ctx(tmp_path))

    assert result["path"].endswith(".jpg")
    assert result["mime_type"] == "image/jpeg"


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

    assert (result["path"], result["meta_path"]) == (picture, sidecar)
    assert json.loads((tmp_path / sidecar).read_text())["image_path"] == picture
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

    assert result["paths"] == ["art/notes_1.txt.png", "art/notes_2.txt.png"]
    assert result["meta_paths"] == ["art/notes_1.txt.json", "art/notes_2.txt.json"]


@pytest.mark.parametrize("asked", ["artifacts/images/", "artifacts/images/.png"])
@pytest.mark.asyncio
async def test_a_name_with_no_file_in_it_is_refused_before_drawing(
    tmp_path: Path, asked: str
) -> None:
    r"""A folder, or a bare `.png`, names no file: refused in plain words, nothing drawn.

    Killed by: src/uclone_x/tools/builtin/image.py :: return rel.endswith(("/", "\\")) or Path(rel).name.lower() in _NO_FILE_NAMES
    Becomes: return False
    """
    gemini = AsyncMock()
    choice = ImageEngineChoice(setting="gemini", model="gemini-3.1-flash-image", gemini=gemini)
    tool = GenerateImageTool(dispatcher=_dispatcher(choice, local_ready=False))
    gemini.generate_image.return_value = (_JPEG_5X3, "image/jpeg")

    result = await tool.execute({"prompt": "a friendly face", "output_path": asked}, _ctx(tmp_path))

    assert (result.success, result.error) == (
        False,
        f"'{asked}' has no file name, so no picture was made. "
        "Give a file name, such as 'artifacts/images/face.png'.",
    )
    gemini.generate_image.assert_not_called()
