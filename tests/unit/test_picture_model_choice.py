"""Which model draws a picture: image engines as connections, per clone (model-gateway §3.5).

Model-gateway step 5. The image engines (`comfyui`, `remote_gpu`) are connections in the
provider table; the picture model is `default_models.image` or a clone's own `image_model`;
`auto` uses the first ready own model, else the first cloud picture model on a connection
with a key -- whatever the chat model is; a chosen model that cannot draw is refused in
plain words, never replaced (P6); and every picture says what drew it ("Drawn with …").

Nothing here reaches the network: every daemon and listing is a fake or a MockTransport.
"""

from __future__ import annotations

import json
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.core.provenance import Provenance
from uclone_x.errors import LLMProviderError
from uclone_x.llm.catalog import CatalogEntry, CatalogResult
from uclone_x.llm.connections import kind_rows
from uclone_x.llm.connectors.factory import create_llm_connector, image_engine_choice
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.gateway import ModelGateway
from uclone_x.llm.models import FinishReason, LLMRequest, ModelResponse, TokenUsage, ToolCallRequest
from uclone_x.tools.builtin.comfy_client import ComfyClient
from uclone_x.tools.builtin.image import (
    ComfyUIImageEngine,
    GenerateImageParams,
    GenerateImageTool,
    ImageEngineChoice,
    ImageEngineRefusal,
    ImageGenerationResult,
    ImagePipelineDispatcher,
    LocalDiffusersImageEngine,
    RemoteCudaImageEngine,
    drawn_with,
)
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

#: Words a person never reads in a refusal: class names, variables, codes.
_INTERNALS = ("Error", "UCX_", "comfyui-local", "remote-cuda", "image_engine", "Traceback")


def _plain(text: str) -> None:
    for word in _INTERNALS:
        assert word not in text, f"{word!r} reaches the person: {text!r}"


def _settings(tmp_path: Path, **data: Any) -> Path:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_PROVIDER",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_BASE_URL",
        "UCX_COMFYUI_URL",
        "UCX_IMAGE_REMOTE_URL",
        "UCX_MEDIA_REMOTE_URL",
        "OLLAMA_HOST",
        "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


# -- The provider table ------------------------------------------------------------------


def test_the_image_engines_are_kinds_that_draw_and_hold_no_conversation() -> None:
    """`GET /api/connections` lists both engines, each saying what it can do.

    Killed by: src/uclone_x/llm/connections.py :: "capabilities": list(spec.capabilities),
    Becomes: "capabilities": ["chat"],
    """
    kinds = {row["kind"]: row for row in kind_rows()}
    assert kinds["comfyui"]["capabilities"] == ["image_create"]
    assert kinds["remote_gpu"]["capabilities"] == ["image_create"]
    assert kinds["gemini"]["capabilities"] == ["chat", "image_create", "image_edit", "image_input"]
    assert (kinds["comfyui"]["needs_base_url"], kinds["comfyui"]["needs_key"]) == (True, False)
    assert kinds["comfyui"]["default_base_url"] == "http://127.0.0.1:8188"
    assert kinds["remote_gpu"]["default_base_url"] is None


def test_an_image_engine_is_never_built_as_a_conversation(tmp_path: Path) -> None:
    """Asked for a chat connector, an image engine is refused in plain words.

    Killed by: src/uclone_x/llm/connectors/factory.py :: if provider_id in IMAGE_ENGINE_KINDS:
    Becomes: if False:
    """
    with pytest.raises(LLMProviderError) as refused:
        create_llm_connector(provider="comfyui", saved_choice_file=tmp_path / "settings.json")
    assert "draws pictures and cannot hold a conversation" in str(refused.value)


# -- Which picture model a draw follows ------------------------------------------------------


_COMFY = {"id": "comfyui", "kind": "comfyui", "base_url": "http://127.0.0.1:18188"}


def test_a_clones_own_picture_model_wins_over_the_default(tmp_path: Path) -> None:
    """The clone's `image_model` is the one it draws with; without one, the default (§3.4).

    Killed by: src/uclone_x/llm/connectors/factory.py :: clean_own = own.strip() if own and own.strip() else None
    Becomes: clean_own = None
    """
    path = _settings(
        tmp_path,
        connections=[_COMFY, {"id": "gemini", "kind": "gemini", "key": "test-key"}],
        default_models={"image": "gemini/gemini-2.5-flash-image"},
    )

    own = image_engine_choice(path, "comfyui/anillustrious_v4")
    assert (own.pin, own.chosen_by, own.pinned_profile) == ("comfyui", "clone", "anillustrious_v4")
    assert own.comfyui_base_url == "http://127.0.0.1:18188"

    default = image_engine_choice(path, None)
    assert (default.pin, default.chosen_by, default.gemini_model) == (
        "gemini",
        "default",
        "gemini-2.5-flash-image",
    )


def test_auto_finds_a_google_key_on_a_differently_named_row_whatever_the_chat_model(
    tmp_path: Path,
) -> None:
    """The #2158 review's bug: a key on a row called `work` was not found for pictures.

    `auto` asks the connections, not the row whose id is the kind, and not the chat model:
    here the conversations run on Ollama.

    Killed by: src/uclone_x/llm/connectors/factory.py :: cloud = next((c for c in conns if c.kind == "gemini" and c.key), None)
    Becomes: cloud = next((c for c in conns if c.id == "gemini" and c.key), None)
    """
    path = _settings(
        tmp_path,
        connections=[
            {"id": "ollama", "kind": "ollama", "base_url": "http://127.0.0.1:11434"},
            {"id": "work", "kind": "gemini", "key": "work-key"},
        ],
        default_models={"deep": "ollama/qwen3:8b"},
    )

    choice = image_engine_choice(path)

    assert isinstance(choice.gemini, GeminiConnector)
    assert choice.gemini.api_key == "work-key"
    assert (choice.gemini_connection, choice.gemini_model) == ("work", "gemini-2.5-flash-image")

    pinned = image_engine_choice(path, "work/gemini-2.5-flash-image")
    assert isinstance(pinned.gemini, GeminiConnector)
    assert pinned.gemini.api_key == "work-key"


def test_the_gpu_server_takes_only_auto(tmp_path: Path) -> None:
    """The worker cannot be told which model to load, so a pin to one is refused (§3.5).

    Killed by: src/uclone_x/llm/connections.py :: if conn.kind == "remote_gpu" and ref.model != REMOTE_GPU_MODEL:
    Becomes: if False:
    """
    path = _settings(
        tmp_path,
        connections=[
            {"id": "remote_gpu", "kind": "remote_gpu", "base_url": "http://10.0.0.5:9000"}
        ],
    )

    assert image_engine_choice(path, "remote_gpu/auto").refusal is None
    pinned = image_engine_choice(path, "remote_gpu/sdxl-turbo")
    assert pinned.refusal is not None
    assert "remote_gpu/auto" in pinned.refusal
    _plain(pinned.refusal)


def test_a_conversation_connection_cannot_be_a_picture_model(tmp_path: Path) -> None:
    """A model on a connection that draws nothing is refused, saying so.

    Killed by: src/uclone_x/llm/connections.py :: if "image_create" not in spec.capabilities:
    Becomes: if False:
    """
    path = _settings(tmp_path, connections=[{"id": "openai", "kind": "openai", "key": "sk-test"}])

    choice = image_engine_choice(path, "openai/gpt-4o")

    assert choice.refusal is not None
    assert "does not draw pictures" in choice.refusal
    _plain(choice.refusal)


# -- The draw ------------------------------------------------------------------------------


def _result(engine: str) -> ImageGenerationResult:
    return ImageGenerationResult(
        image_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 24,
        seed=1,
        engine_name=engine,
        device_info="test",
        duration_seconds=0.1,
        width=832,
        height=1216,
    )


class _Comfy:
    """A ComfyUI engine for one pin: whether it answers, and what it was built with."""

    def __init__(self, alive: bool) -> None:
        self.alive = alive
        self.built: list[tuple[str, str]] = []

    def __call__(self, url: str, checkpoint: str) -> Any:
        self.built.append((url, checkpoint))
        engine = AsyncMock(spec=ComfyUIImageEngine)
        engine.is_available.return_value = self.alive
        engine.generate.return_value = _result("comfyui-local")
        return engine


def _engines() -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    """Every own engine of a head ready to draw, so a refusal cannot be "nothing was ready"."""
    remote = AsyncMock(spec=RemoteCudaImageEngine)
    remote.is_available.return_value = True
    remote.base_url = "http://10.0.0.5:9000"
    remote.generate.return_value = _result("remote-cuda")
    comfy = AsyncMock(spec=ComfyUIImageEngine)
    comfy.is_available.return_value = True
    comfy.base_url = "http://127.0.0.1:8188"
    comfy.checkpoint = "anillustrious_v4.safetensors"
    comfy.generate.return_value = _result("comfyui-local")
    local = AsyncMock(spec=LocalDiffusersImageEngine)
    local.is_available.return_value = True
    local.resolve_checkpoint.return_value = None
    local.generate.return_value = _result("diffusers-sdxl")
    return remote, comfy, local


class _Cloud:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_image(self, prompt: str, aspect_ratio: str, model: str) -> tuple[bytes, str]:
        self.calls += 1
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 24, "image/png"


def _pinned_comfy(cloud: _Cloud | None = None) -> ImageEngineChoice:
    return ImageEngineChoice(
        chosen="comfyui/anillustrious_v4",
        chosen_by="clone",
        pin="comfyui",
        pinned_profile="anillustrious_v4",
        from_connections=True,
        comfyui_base_url="http://127.0.0.1:18188",
        gemini=cloud,
        gemini_model="gemini-2.5-flash-image" if cloud is not None else None,
    )


async def _draw(dispatcher: ImagePipelineDispatcher, own: str | None = None) -> Any:
    return await dispatcher.dispatch("a lighthouse", "", "1:1", 7, "none", own=own)


@pytest.mark.asyncio
async def test_an_unreachable_chosen_model_is_refused_never_replaced() -> None:
    """A ComfyUI model chosen for a clone, its daemon silent: refused, naming the model.

    Every other engine is ready and a cloud model has a key; none of them draws instead
    (P6, model-gateway G7).

    Killed by: src/uclone_x/tools/builtin/image.py :: raise self._refuse_pin(choice, self._pin_silence(choice))
    Becomes: pass
    """
    remote, comfy, local = _engines()
    cloud = _Cloud()
    pinned = _Comfy(alive=False)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=local,
        engine_settings=lambda _own: _pinned_comfy(cloud),
        pinned_comfy_engine=pinned,
    )

    with pytest.raises(ImageEngineRefusal) as refused:
        await _draw(dispatcher)

    text = str(refused.value)
    assert "comfyui/anillustrious_v4" in text
    assert "chosen for this clone" in text
    assert "Settings › Models" in text
    _plain(text)
    assert cloud.calls == 0
    for engine in (remote, comfy, local):
        engine.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_chosen_comfyui_model_draws_with_its_own_checkpoint_at_its_connection() -> None:
    """The pin's checkpoint and its connection's address reach the engine, and the result
    says what drew it.

    Killed by: src/uclone_x/tools/builtin/image.py :: comfy_url, pinned.filename or pinned.model_id
    Becomes: comfy_url, pinned.model_id
    Killed by: src/uclone_x/tools/builtin/image.py :: return "the picture model chosen for this clone"
    Becomes: return "the picture model chosen in Settings"
    """
    remote, comfy, local = _engines()
    pinned = _Comfy(alive=True)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=local,
        engine_settings=lambda _own: _pinned_comfy(),
        pinned_comfy_engine=pinned,
    )

    drawn = await _draw(dispatcher)

    assert pinned.built == [("http://127.0.0.1:18188", "anillustrious_v4.safetensors")]
    assert (drawn.model_id, drawn.where) == ("anillustrious_v4", "this_computer")
    assert drawn.drawn_with == (
        "Drawn with Illustrious-XL v4 (Anime SDXL) on this computer "
        "(the picture model chosen for this clone)."
    )
    remote.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_chosen_comfyui_model_whose_connection_lost_its_address_is_refused(
    tmp_path: Path,
) -> None:
    """#2176: a pin on a ComfyUI connection with its address cleared is refused by name.

    Before the fix the shared ComfyUI engine drew at its built-in address with whatever
    checkpoint it had found (`anillustrious_v4`), and the result said "Drawn with
    Illustrious-XL v0.1" (P6). The GPU server's connection is held to the same rule.

    Killed by: src/uclone_x/llm/connections.py :: if conn.kind in BASE_URL_KINDS and not (conn.base_url or "").strip():
    Becomes: if False:
    Killed by: src/uclone_x/llm/connections.py :: if conn.kind in BASE_URL_KINDS and not (conn.base_url or "").strip():
    Becomes: if conn.kind == "comfyui" and not (conn.base_url or "").strip():
    """
    path = _settings(
        tmp_path,
        connections=[
            {"id": "comfyui", "kind": "comfyui", "base_url": ""},
            {"id": "remote_gpu", "kind": "remote_gpu", "base_url": ""},
        ],
        default_models={"image": "comfyui/Illustrious-XL-v0.1"},
    )
    remote, comfy, local = _engines()
    pinned = _Comfy(alive=True)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=local,
        engine_settings=lambda own: image_engine_choice(path, own),
        pinned_comfy_engine=pinned,
    )

    with pytest.raises(ImageEngineRefusal) as refused:
        await _draw(dispatcher)

    text = str(refused.value)
    assert text == (
        "The picture model comfyui/Illustrious-XL-v0.1 cannot be used: the connection "
        "comfyui has no address. Add its address to comfyui in Settings › Models."
    )
    _plain(text)
    assert pinned.built == []
    for engine in (remote, comfy, local):
        engine.generate.assert_not_awaited()
    gpu = image_engine_choice(path, "remote_gpu/auto").refusal
    assert gpu is not None and "remote_gpu has no address" in gpu


@pytest.mark.asyncio
async def test_a_comfyui_pin_without_an_address_never_falls_to_the_shared_engine() -> None:
    """The draw itself refuses a pin with no address, whatever produced the choice.

    The shared ComfyUI engine is ready at its built-in address with another checkpoint; it
    neither draws nor lends the pin its name (#2176).

    Killed by: src/uclone_x/tools/builtin/image.py :: if not choice.comfyui_base_url:
    Becomes: if False:
    """
    remote, comfy, local = _engines()
    pinned = _Comfy(alive=True)
    choice = dataclass_replace(_pinned_comfy(), comfyui_base_url=None)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=local,
        engine_settings=lambda _own: choice,
        pinned_comfy_engine=pinned,
    )

    with pytest.raises(ImageEngineRefusal) as refused:
        await _draw(dispatcher)

    text = str(refused.value)
    assert "comfyui/anillustrious_v4 (chosen for this clone)" in text
    assert "its connection has no address" in text
    _plain(text)
    assert pinned.built == []
    for engine in (remote, comfy, local):
        engine.generate.assert_not_awaited()


def test_the_words_say_whose_choice_drew() -> None:
    """The three choosers, and the GPU server that names no model.

    Killed by: src/uclone_x/tools/builtin/image.py :: return "picked automatically"
    Becomes: return "the picture model chosen in Settings"
    """
    auto = drawn_with(None, "gpu_server", ImageEngineChoice())
    assert auto == (
        "Drawn with the model your GPU server has loaded on your GPU server (picked automatically)."
    )
    default = drawn_with("X", "cloud", ImageEngineChoice(pin="gemini"))
    assert default == "Drawn with X in the cloud (Google) (the picture model chosen in Settings)."


@pytest.mark.asyncio
async def test_the_tool_passes_the_clones_picture_model_to_the_draw(tmp_path: Path) -> None:
    """`generate_image` asks for the picture model of the clone that called it.

    Killed by: src/uclone_x/tools/builtin/image.py :: image_model=context.image_model,  # the asking clone's picture model (§3.4)
    Becomes: image_model=None,
    """
    asked: list[str | None] = []

    def settings(own: str | None) -> ImageEngineChoice:
        asked.append(own)
        return ImageEngineChoice(from_connections=True, refusal="Not now.")

    remote, comfy, local = _engines()
    tool = GenerateImageTool(
        dispatcher=ImagePipelineDispatcher(
            remote_engine=remote, comfy_engine=comfy, local_engine=local, engine_settings=settings
        )
    )

    with pytest.raises(ImageEngineRefusal):
        await tool.run(
            GenerateImageParams(prompt="a lighthouse"),
            ToolContext(
                agent_id="a",
                session_id="s",
                workspace_root=tmp_path,
                image_model="comfyui/anillustrious_v4",
            ),
        )
    assert "comfyui/anillustrious_v4" in asked


class _ScriptedLLM(MockLLMConnector):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self._queue = list(responses)

    async def generate(self, request: LLMRequest) -> ModelResponse:
        if self._queue:
            return self._queue.pop(0)
        return await super().generate(request)


def _reply(text: str, calls: tuple[ToolCallRequest, ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=text,
        tool_calls=calls,
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        usage=TokenUsage(provider="mock"),
        provenance=Provenance.primary("mock"),
    )


@pytest.mark.asyncio
async def test_a_clones_turn_carries_its_picture_model_to_the_tool(tmp_path: Path) -> None:
    """The agent puts its own `image_model` on the tool context of every call.

    Killed by: src/uclone_x/agent/turn_executor.py :: image_model=self._config.llm_config.image_model,
    Becomes: image_model=None,
    """
    from uclone_x.agent.base import BaseAgent

    asked: list[str | None] = []

    def settings(own: str | None) -> ImageEngineChoice:
        asked.append(own)
        return ImageEngineChoice(from_connections=True, refusal="Not now.")

    remote, comfy, local = _engines()
    tool = GenerateImageTool(
        dispatcher=ImagePipelineDispatcher(
            remote_engine=remote, comfy_engine=comfy, local_engine=local, engine_settings=settings
        )
    )
    draw = ToolCallRequest(id="call_1", name="generate_image", arguments={"prompt": "a boat"})
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="artist",
            name="Artist",
            llm_config=AgentLLMConfig(
                model_name="mock", auto_compact=False, image_model="comfyui/anillustrious_v4"
            ),
        ),
        llm=_ScriptedLLM([_reply("", (draw,)), _reply("Done.")]),
        tools=ToolRegistry(tools=[tool]),
    )

    await agent.execute_turn("draw a boat")

    assert "comfyui/anillustrious_v4" in asked


# -- The picture model set (`GET /api/models?capability=image`) -------------------------------


@pytest.mark.asyncio
async def test_the_picture_set_lists_each_engines_models_from_its_own_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Own models the daemon has, the GPU server's one `auto`, and declared cloud models.

    A chat-only connection is not in the picture set, and a picture engine is not in the
    chat set (each kind's capabilities).

    Killed by: src/uclone_x/llm/gateway.py :: if p.engine_type in COMFYUI_ENGINE_TYPES and p.filename and p.filename in files
    Becomes: if p.engine_type in COMFYUI_ENGINE_TYPES and p.filename
    Killed by: src/uclone_x/llm/gateway.py :: or wanted in (spec.capabilities if (spec := spec_for(conn.kind)) else ())
    Becomes: or True
    """
    import uclone_x.llm.gateway as gateway_module

    async def checkpoints(_client: ComfyClient, timeout: float = 3.0) -> list[str]:
        return ["SDXL/anillustrious_v4.safetensors", "something_else.safetensors"]

    monkeypatch.setattr(ComfyClient, "checkpoints", checkpoints)
    monkeypatch.setattr(RemoteCudaImageEngine, "is_available", AsyncMock(return_value=True))

    async def catalog(kind: str, **_kw: Any) -> CatalogResult:
        entries = (CatalogEntry(id="gemini-2.5-flash-image"), CatalogEntry(id="gemini-3-pro"))
        return CatalogResult(provider=kind, status="live", entries=entries)

    monkeypatch.setattr(gateway_module, "read_provider_catalog", catalog)
    path = _settings(
        tmp_path,
        connections=[
            _COMFY,
            {"id": "remote_gpu", "kind": "remote_gpu", "base_url": "http://10.0.0.5:9000"},
            {"id": "gemini", "kind": "gemini", "key": "test-key"},
            {"id": "openai", "kind": "openai", "key": "sk-test"},
        ],
    )
    gateway = ModelGateway(path)

    pictures = await gateway.model_set("image")

    assert [(g.connection_id, [m.ref for m in g.models]) for g in pictures.groups] == [
        ("comfyui", ["comfyui/anillustrious_v4"]),
        ("remote_gpu", ["remote_gpu/auto"]),
        ("gemini", ["gemini/gemini-2.5-flash-image"]),
    ]
    chat = await gateway.model_set("chat")
    assert {g.connection_id for g in chat.groups} == {"gemini", "openai"}


@pytest.mark.asyncio
async def test_comfyuis_checkpoint_list_is_read_from_its_loader_node() -> None:
    """The daemon's own list, from `/object_info/CheckpointLoaderSimple`.

    Killed by: src/uclone_x/tools/builtin/comfy_client.py :: names = cast(list[object], names)[0]
    Becomes: pass
    """
    body: dict[str, Any] = {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"], {}]}}
        }
    }

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/CheckpointLoaderSimple"
        return httpx.Response(200, json=body)

    http = httpx.AsyncClient(transport=httpx.MockTransport(answer), base_url="http://comfy")
    client = ComfyClient(base_url="http://comfy", client=http)
    assert await client.checkpoints() == ["a.safetensors", "b.safetensors"]

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    down = httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://comfy")
    assert await ComfyClient(base_url="http://comfy", client=down).checkpoints() is None


# -- The dashboard's routes ----------------------------------------------------------------


def _api(tmp_path: Path, settings: dict[str, Any]) -> Any:
    from fastapi.testclient import TestClient

    from uclone_x.ui.app import create_ui_app

    storage = tmp_path / "sessions"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    app = create_ui_app(static_dir=tmp_path / "static", storage_dir=storage, workspace_dir=tmp_path)
    return TestClient(app)


@pytest.mark.usefixtures("builtin_personas_absent")
def test_a_clones_picture_model_on_the_gpu_server_must_be_auto(tmp_path: Path) -> None:
    """`PUT/POST /api/clones` refuses a pin the GPU server cannot keep, in plain words.

    Killed by: src/uclone_x/ui/app.py :: if slot == "image_model" and (problem := image_ref_problem(ref, conn)) is not None:
    Becomes: if False:
    """
    client = _api(
        tmp_path,
        {"connections": [{"id": "gpu", "kind": "remote_gpu", "base_url": "http://10.0.0.5:9000"}]},
    )
    base = {"name": "painter", "role": "Paints", "system_prompt": "x"}

    refused = client.post("/api/clones", json={**base, "image_model": "gpu/sdxl"})
    assert refused.status_code == 422
    assert "gpu/auto" in refused.json()["detail"]
    _plain(refused.json()["detail"])
    assert client.post("/api/clones", json={**base, "image_model": "gpu/auto"}).status_code == 201


def test_a_default_picture_model_must_be_on_a_connection_that_draws(tmp_path: Path) -> None:
    """`POST /api/settings` refuses a chat connection's model as the Pictures default.

    Killed by: src/uclone_x/ui/app.py :: if slot == "image" and (problem := image_ref_problem(ref, conn)) is not None:
    Becomes: if False:
    """
    client = _api(tmp_path, {"connections": [{"id": "openai", "kind": "openai", "key": "sk-t"}]})

    res = client.post("/api/settings", json={"default_models": {"image": "openai/gpt-4o"}})

    assert res.status_code == 400
    assert "does not draw pictures" in res.json()["detail"]
    _plain(res.json()["detail"])


def test_the_old_picture_settings_are_never_read(tmp_path: Path) -> None:
    """`image_engine`, `image_model` and `comfyui_base_url` are deleted, with no migration.

    A file that still holds them draws as one that does not: `auto`, and no ComfyUI.
    No kill declaration: it asserts an absence, and the readers it would revive are gone.
    """
    path = _settings(
        tmp_path,
        image_engine="gemini",
        image_model="gemini-3-pro-image",
        comfyui_base_url="http://10.0.0.9:8188",
    )
    choice = image_engine_choice(path)
    assert (choice.chosen, choice.pin, choice.comfyui_base_url, choice.gemini) == (
        "auto",
        None,
        None,
        None,
    )
    with patch.object(LocalDiffusersImageEngine, "resolve_checkpoint", return_value=None):
        from uclone_x.tools.builtin.image_status import probe_image_engines

        assert probe_image_engines(path).comfy_url is None


# -- A ComfyUI already running here: detected, offered, added only on a yes ---------------


def _found_report() -> Any:
    from uclone_x.tools.builtin.image_status import ImageEngineReport

    return ImageEngineReport(
        remote_url=None,
        remote_alive=False,
        comfy_url=None,
        comfy_alive=False,
        dependency_problems=("'torch' is not installed",),
        checkpoint=None,
        detected_comfyui="http://127.0.0.1:8188",
    )


@pytest.mark.parametrize(("answer", "added"), [(True, True), (False, False)])
def test_install_offers_a_running_comfyui_and_adds_it_only_on_a_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: bool, added: bool
) -> None:
    """`ucx install` asks before it connects someone's ComfyUI (agent-assisted-installation,
    rung 1: detect, never seize), and saves it as a `comfyui` connection on a yes.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: add_connection("comfyui", base_url=address)
    Becomes: pass
    Killed by: src/uclone_x/cli/commands/bootstrap.py :: agreed = ask_consent(
    Becomes: agreed = True or ask_consent(
    """
    from uclone_x.cli.commands import bootstrap
    from uclone_x.llm.connections import saved_connections
    from uclone_x.llm.connectors.saved_choice import settings_data

    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path))
    asked: list[str] = []

    def consent(question: str, **_kw: Any) -> bool:
        asked.append(question)
        return answer

    with (
        patch.object(bootstrap, "probe_image_engines", return_value=_found_report()),
        patch.object(bootstrap, "ask_consent", consent),
    ):
        setup = bootstrap.setup_local_image(interactive=False, allow_download=False)

    assert asked and "http://127.0.0.1:8188" in asked[0]
    rows = [
        (c.kind, c.base_url) for c in saved_connections(settings_data(tmp_path / "settings.json"))
    ]
    assert rows == ([("comfyui", "http://127.0.0.1:8188")] if added else [])
    assert (setup.ready, setup.engine) == ((True, "comfyui-local") if added else (False, "none"))


def test_the_status_offers_a_comfyui_found_here_only_while_none_is_connected(
    tmp_path: Path,
) -> None:
    """`/api/media/status` names a ComfyUI answering on the default address, to offer it.

    Killed by: src/uclone_x/tools/builtin/image_status.py :: detected = detect_local_comfyui() if detect_comfyui and comfy_url is None else None
    Becomes: detected = None
    """
    from uclone_x.tools.builtin.image_status import media_status_payload, probe_image_engines

    with (
        patch.object(ComfyUIImageEngine, "is_available", AsyncMock(return_value=True)),
        patch.object(LocalDiffusersImageEngine, "resolve_checkpoint", return_value=None),
    ):
        bare = probe_image_engines(tmp_path / "absent.json", detect_comfyui=True)
        connected = probe_image_engines(
            _settings(tmp_path, connections=[_COMFY]), detect_comfyui=True
        )

    assert media_status_payload(bare)["detected_comfyui"] == "http://127.0.0.1:8188"
    # Detected is not used: nothing draws there until the person adds it.
    assert bare.comfy_url is None and bare.comfy_alive is False
    assert media_status_payload(connected)["detected_comfyui"] is None
