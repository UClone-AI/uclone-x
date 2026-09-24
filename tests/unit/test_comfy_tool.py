"""Unit tests for ComfyUI client and image generation agent tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from uclone_x.errors import ComfyUIError, ComfyUIExecutionError, PathTraversalError
from uclone_x.sandbox.models import WorkspaceIsolation
from uclone_x.tools.builtin.comfy_client import (
    COMFY_CHECKPOINT_ENV,
    COMFY_DEFAULT_CHECKPOINT,
    DEFAULT_COMFYUI_BASE_URL,
    ComfyClient,
    ComfyTimeoutError,
    format_exec_error,
    format_node_errors,
)
from uclone_x.tools.builtin.comfy_image_tool import (
    ComfyImageGenParams,
    ComfyImageGenTool,
    build_txt2img_workflow,
)
from uclone_x.tools.builtin.image import ComfyUIImageEngine
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture
def tool_context(workspace: Path) -> ToolContext:
    return ToolContext(
        agent_id="test_agent",
        session_id="test_session",
        workspace_root=workspace,
        isolation=WorkspaceIsolation(),
    )


# ======================================================================================
# 1. Helper Formatting Tests
# ======================================================================================


def test_format_node_errors_with_detailed_payload() -> None:
    payload = {
        "error": {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": "Syntax check failed",
        },
        "node_errors": {
            "3": {
                "class_type": "KSampler",
                "errors": [
                    {
                        "type": "value_not_in_list",
                        "message": "Value not in list: sampler_name",
                        "details": "'invalid_euler' not in ['euler']",
                    }
                ],
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "errors": ["Non-dict error detail"],
            },
        },
    }
    msg = format_node_errors(payload)
    assert "Prompt outputs failed validation" in msg
    assert "Syntax check failed" in msg
    assert (
        "Node 3 (KSampler): Value not in list: sampler_name: 'invalid_euler' not in ['euler']"
        in msg
    )
    assert "Node 6 (CLIPTextEncode): Non-dict error detail" in msg


def test_format_node_errors_fallback_cases() -> None:
    assert format_node_errors({}) == "ComfyUI reported an error with no details"
    assert format_node_errors({"error": "Simple string error"}) == "Simple string error"
    assert (
        format_node_errors({"node_errors": {"5": "raw string node error"}})
        == "Node 5: raw string node error"
    )


def test_format_exec_error() -> None:
    status = {
        "status_str": "error",
        "messages": [
            [
                "execution_error",
                {
                    "node_id": "3",
                    "node_type": "KSampler",
                    "exception_type": "CUDAOutOfMemoryError",
                    "exception_message": "CUDA out of memory",
                },
            ]
        ],
    }
    msg = format_exec_error(status)
    assert "Node 3 (KSampler): CUDAOutOfMemoryError: CUDA out of memory" in msg

    empty_status: dict[str, Any] = {"status_str": "error", "messages": []}
    assert format_exec_error(empty_status) == "ComfyUI execution failed with an unspecified error"


# ======================================================================================
# 2. ComfyClient Unit Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_client_initialization_and_url_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    client_default = ComfyClient()
    assert client_default.base_url == DEFAULT_COMFYUI_BASE_URL

    monkeypatch.setenv("COMFYUI_BASE_URL", "http://remote-gpu:8188/")
    client_env = ComfyClient()
    assert client_env.base_url == "http://remote-gpu:8188"

    client_explicit = ComfyClient(base_url="http://custom-host:9000/")
    assert client_explicit.base_url == "http://custom-host:9000"


@pytest.mark.asyncio
async def test_client_alive_and_system_stats() -> None:
    stats_data = {
        "system": {"os": "posix", "comfyui_version": "0.1.0"},
        "devices": [{"name": "mps", "type": "mps", "vram_total": 16 * 1024**3}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/system_stats":
            return httpx.Response(200, json=stats_data)
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    assert await client.alive() is True
    stats = await client.system_stats()
    assert stats["system"]["os"] == "posix"
    assert stats["devices"][0]["name"] == "mps"

    await client.aclose()


@pytest.mark.asyncio
async def test_client_alive_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    assert await client.alive() is False
    with pytest.raises(ComfyUIError):
        await client.system_stats()

    await client.aclose()


@pytest.mark.asyncio
async def test_alive_propagates_internal_exceptions() -> None:
    """P6: alive() only catches httpx.HTTPError; internal defects propagate."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise ZeroDivisionError("internal defect")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)
    with pytest.raises(ZeroDivisionError, match="internal defect"):
        await client.alive()
    await client.aclose()


@pytest.mark.asyncio
async def test_alive_returns_false_on_connection_error() -> None:
    """alive() returns False on network/connection failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)
    assert await client.alive() is False
    await client.aclose()


@pytest.mark.asyncio
async def test_client_object_info_and_missing_nodes() -> None:
    obj_info_data: dict[str, Any] = {
        "CheckpointLoaderSimple": {"input": {}},
        "CLIPTextEncode": {"input": {}},
        "KSampler": {"input": {}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json=obj_info_data)
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    info = await client.object_info()
    assert "KSampler" in info

    missing = await client.missing_nodes(["KSampler", "VAEDecode", "CustomSecretNode"])
    assert missing == ["VAEDecode", "CustomSecretNode"]

    await client.aclose()


@pytest.mark.asyncio
async def test_client_missing_nodes_handles_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)
    with pytest.raises(ComfyUIError):
        await client.missing_nodes(["KSampler"])
    await client.aclose()


@pytest.mark.asyncio
async def test_missing_nodes_raises_on_connection_failure() -> None:
    """P6: missing_nodes() raises ComfyUIError on connection failure rather than returning []."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)
    with pytest.raises(ComfyUIError, match="Failed to fetch ComfyUI object info"):
        await client.missing_nodes(["KSampler"])
    await client.aclose()


@pytest.mark.asyncio
async def test_client_queue_prompt_success() -> None:
    received_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt" and request.method == "POST":
            data = json.loads(request.content.decode("utf-8"))
            received_requests.append(data)
            return httpx.Response(200, json={"prompt_id": "prompt-12345", "number": 1})
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    workflow = {"1": {"class_type": "EmptyLatentImage"}}
    prompt_id = await client.queue_prompt(workflow)

    assert prompt_id == "prompt-12345"
    assert len(received_requests) == 1
    assert received_requests[0]["prompt"] == workflow
    assert received_requests[0]["client_id"] == client.client_id

    await client.aclose()


@pytest.mark.asyncio
async def test_client_queue_prompt_missing_prompt_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"number": 1})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    with pytest.raises(ComfyUIError, match="without prompt_id"):
        await client.queue_prompt({})

    await client.aclose()


@pytest.mark.asyncio
async def test_client_queue_prompt_unpacks_node_errors() -> None:
    error_response = {
        "error": {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt validation failure",
            "details": "",
        },
        "node_errors": {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "errors": [
                    {
                        "type": "value_not_in_list",
                        "message": "Checkpoint not found: model.safetensors",
                        "details": "model.safetensors",
                    }
                ],
            }
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=error_response)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    with pytest.raises(ComfyUIError) as exc_info:
        await client.queue_prompt({})

    assert exc_info.value.status_code == 400
    assert "Checkpoint not found" in str(exc_info.value)
    assert exc_info.value.node_errors is not None
    assert "4" in exc_info.value.node_errors

    await client.aclose()


@pytest.mark.asyncio
async def test_client_queue_prompt_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by ComfyUI daemon")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    with pytest.raises(ComfyUIError, match="Failed to connect"):
        await client.queue_prompt({})

    await client.aclose()


@pytest.mark.asyncio
async def test_client_get_history() -> None:
    hist_data = {
        "prompt-1": {
            "outputs": {
                "9": {"images": [{"filename": "out_0001.png", "subfolder": "", "type": "output"}]}
            },
            "status": {"completed": True, "status_str": "success"},
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/history/prompt-1":
            return httpx.Response(200, json=hist_data)
        if request.url.path == "/history/missing":
            return httpx.Response(200, json={})
        return httpx.Response(500)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    entry = await client.get_history("prompt-1")
    assert "outputs" in entry

    missing_entry = await client.get_history("missing")
    assert missing_entry == {}

    with pytest.raises(ComfyUIError):
        await client.get_history("error-prompt")

    await client.aclose()


@pytest.mark.asyncio
async def test_client_get_history_malformed_body() -> None:
    def handler_non_dict(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"])

    client1 = ComfyClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handler_non_dict)))
    with pytest.raises(ComfyUIError, match="Unexpected non-dict /history response"):
        await client1.get_history("pid")
    await client1.aclose()

    def handler_entry_non_dict(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"pid": "not a dict entry"})

    client2 = ComfyClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler_entry_non_dict))
    )
    with pytest.raises(ComfyUIError, match="Unexpected non-dict history entry"):
        await client2.get_history("pid")
    await client2.aclose()

    def handler_invalid_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not valid json")

    client3 = ComfyClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler_invalid_json))
    )
    with pytest.raises(ComfyUIError, match="Failed to parse ComfyUI /history response JSON"):
        await client3.get_history("pid")
    await client3.aclose()


@pytest.mark.asyncio
async def test_client_wait_for_output_success() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            # Job still running
            return httpx.Response(200, json={"pid": {}})

        return httpx.Response(
            200,
            json={
                "pid": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {
                        "9": {
                            "images": [
                                {
                                    "filename": "render_001.png",
                                    "subfolder": "sub",
                                    "type": "output",
                                }
                            ]
                        }
                    },
                }
            },
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    filenames = await client.wait_for_output("pid", timeout_seconds=5.0, poll_interval=0.01)
    assert filenames == ["sub/render_001.png"]
    await client.aclose()


@pytest.mark.asyncio
async def test_client_wait_for_output_execution_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "pid": {
                    "status": {
                        "status_str": "error",
                        "messages": [
                            [
                                "execution_error",
                                {
                                    "node_id": "4",
                                    "node_type": "VAEDecode",
                                    "exception_type": "RuntimeError",
                                    "exception_message": "Tensor shape mismatch",
                                },
                            ]
                        ],
                    }
                }
            },
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    with pytest.raises(
        ComfyUIError, match="Node 4 \\(VAEDecode\\): RuntimeError: Tensor shape mismatch"
    ):
        await client.wait_for_output("pid", timeout_seconds=2.0, poll_interval=0.01)

    await client.aclose()


@pytest.mark.asyncio
async def test_client_wait_for_output_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    with pytest.raises(ComfyTimeoutError) as exc_info:
        await client.wait_for_output("pid", timeout_seconds=0.05, poll_interval=0.01)

    # Subclass of both ComfyUIError and TimeoutError
    assert isinstance(exc_info.value, TimeoutError)
    assert isinstance(exc_info.value, ComfyUIError)
    await client.aclose()


@pytest.mark.asyncio
async def test_client_download_image_with_subfolder() -> None:
    raw_png = b"\x89PNG\r\n\x1a\nfake-image-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/view":
            assert request.url.params["filename"] == "sample.png"
            assert request.url.params["subfolder"] == "custom/sub"
            assert request.url.params["type"] == "output"
            return httpx.Response(200, content=raw_png)
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    # Test auto-splitting subfolder from filename
    content = await client.download_image("custom/sub/sample.png")
    assert content == raw_png

    # Test explicit subfolder parameter
    content2 = await client.download_image("sample.png", subfolder="custom/sub")
    assert content2 == raw_png

    await client.aclose()


@pytest.mark.asyncio
async def test_client_download_image_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)

    with pytest.raises(ComfyUIError, match="Failed to download image"):
        await client.download_image("missing.png")

    await client.aclose()


@pytest.mark.asyncio
async def test_client_async_context_manager() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with ComfyClient(client=mock_client) as client:
        assert client is not None


# ======================================================================================
# 3. Workflow Construction Tests
# ======================================================================================


def test_build_txt2img_workflow_structure() -> None:
    wf = build_txt2img_workflow(
        prompt="A serene cyberpunk alley with neon lights",
        negative_prompt="blurry, low quality",
        width=768,
        height=1024,
        seed=42,
        steps=25,
        cfg=8.5,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        checkpoint="sdxl_base.safetensors",
    )

    # Validate CheckpointLoaderSimple
    assert wf["4"]["class_type"] == "CheckpointLoaderSimple"
    assert wf["4"]["inputs"]["ckpt_name"] == "sdxl_base.safetensors"

    # Validate EmptyLatentImage
    assert wf["5"]["class_type"] == "EmptyLatentImage"
    assert wf["5"]["inputs"]["width"] == 768
    assert wf["5"]["inputs"]["height"] == 1024

    # Validate CLIPTextEncode (pos & neg)
    assert wf["6"]["inputs"]["text"] == "A serene cyberpunk alley with neon lights"
    assert wf["6"]["inputs"]["clip"] == ["4", 1]
    assert wf["7"]["inputs"]["text"] == "blurry, low quality"
    assert wf["7"]["inputs"]["clip"] == ["4", 1]

    # Validate KSampler
    ksampler = wf["3"]["inputs"]
    assert ksampler["seed"] == 42
    assert ksampler["steps"] == 25
    assert ksampler["cfg"] == 8.5
    assert ksampler["sampler_name"] == "dpmpp_2m"
    assert ksampler["scheduler"] == "karras"
    assert ksampler["model"] == ["4", 0]
    assert ksampler["positive"] == ["6", 0]
    assert ksampler["negative"] == ["7", 0]
    assert ksampler["latent_image"] == ["5", 0]

    # Validate VAEDecode & SaveImage
    assert wf["8"]["inputs"]["samples"] == ["3", 0]
    assert wf["8"]["inputs"]["vae"] == ["4", 2]
    assert wf["9"]["inputs"]["images"] == ["8", 0]


# ======================================================================================
# 4. ComfyImageGenTool Unit & Integration Tests
# ======================================================================================


def test_tool_parameters_schema() -> None:
    tool = ComfyImageGenTool()
    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert "prompt" in schema["properties"]
    assert "negative_prompt" in schema["properties"]
    assert "width" in schema["properties"]
    assert "height" in schema["properties"]
    assert "seed" in schema["properties"]
    assert "steps" in schema["properties"]
    assert "checkpoint" in schema["properties"]
    assert schema["required"] == ["prompt"]


def test_tool_parameter_validation_bounds() -> None:
    # Width and height minimum is 64
    with pytest.raises(ValidationError):
        ComfyImageGenParams(prompt="cat", width=32)

    with pytest.raises(ValidationError):
        ComfyImageGenParams(prompt="cat", steps=0)

    with pytest.raises(ValidationError):
        ComfyImageGenParams(prompt="cat", cfg=0.5)


def test_tool_registry_registration() -> None:
    registry = ToolRegistry()
    tool = ComfyImageGenTool(name="generate_image")
    registry.register(tool)

    fetched = registry.get("generate_image")
    assert fetched is tool
    assert "generate_image" in [t.name for t in registry.list_tools()]


@pytest.mark.asyncio
async def test_tool_successful_execution(tool_context: ToolContext) -> None:
    test_png_bytes = b"\x89PNG\r\ntest-image-content"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "prompt-xyz-100"})
        if request.url.path == "/history/prompt-xyz-100":
            return httpx.Response(
                200,
                json={
                    "prompt-xyz-100": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {
                            "9": {
                                "images": [
                                    {
                                        "filename": "generated_0001.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                },
            )
        if request.url.path == "/view":
            return httpx.Response(200, content=test_png_bytes)
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)
    tool = ComfyImageGenTool(client=client)

    result = await tool.execute(
        params={
            "prompt": "Futuristic hovercar racing at sunset",
            "seed": 12345,
            "width": 832,
            "height": 1216,
        },
        context=tool_context,
    )

    assert result.success is True
    assert result.error is None
    assert isinstance(result.output, dict)
    assert result.output["prompt_id"] == "prompt-xyz-100"
    assert result.output["seed"] == 12345
    assert result.output["width"] == 832
    assert result.output["height"] == 1216
    assert result.output["bytes_written"] == len(test_png_bytes)

    # Check artifact provenance
    assert isinstance(result.output, dict)
    rel_path = str(result.output["path"])
    assert len(result.artifacts) == 1
    assert result.artifacts[0] == rel_path

    # Verify saved file content on filesystem
    saved_file = tool_context.require_workspace() / rel_path
    assert saved_file.exists()
    assert saved_file.read_bytes() == test_png_bytes

    await client.aclose()


@pytest.mark.asyncio
async def test_tool_custom_output_path(tool_context: ToolContext) -> None:
    test_png_bytes = b"\x89PNG\r\ntest-image-content"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p-1"})
        if request.url.path == "/history/p-1":
            return httpx.Response(
                200,
                json={
                    "p-1": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {"9": {"images": [{"filename": "out.png", "subfolder": ""}]}},
                    }
                },
            )
        if request.url.path == "/view":
            return httpx.Response(200, content=test_png_bytes)
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)
    tool = ComfyImageGenTool(client=client)

    result = await tool.execute(
        params={
            "prompt": "A medieval castle",
            "output_path": "images/nested/castle.png",
        },
        context=tool_context,
    )

    assert result.success is True
    assert isinstance(result.output, dict)
    assert result.output["path"] == "images/nested/castle.png"
    saved_file = tool_context.require_workspace() / "images" / "nested" / "castle.png"
    assert saved_file.exists()
    assert saved_file.read_bytes() == test_png_bytes

    await client.aclose()


@pytest.mark.asyncio
async def test_tool_path_traversal_protection(tool_context: ToolContext) -> None:
    tool = ComfyImageGenTool()

    # Path traversal attempt using execute()
    result = await tool.execute(
        params={
            "prompt": "An escaping agent",
            "output_path": "../../../etc/traversal.png",
        },
        context=tool_context,
    )

    assert result.success is False
    assert "Path traversal violation" in str(result.error)

    # Path traversal attempt directly via run()
    with pytest.raises(PathTraversalError):
        await tool.run(
            ComfyImageGenParams(
                prompt="test",
                output_path="../../outside.png",
            ),
            context=tool_context,
        )


@pytest.mark.asyncio
async def test_tool_comfyui_failure_handling(tool_context: ToolContext) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {"message": "Invalid workflow structure"},
                "node_errors": {},
            },
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)
    tool = ComfyImageGenTool(client=client)

    result = await tool.execute(
        params={"prompt": "A test prompt that fails on server"},
        context=tool_context,
    )

    assert result.success is False
    assert "Invalid workflow structure" in str(result.error)

    await client.aclose()


@pytest.mark.asyncio
async def test_tool_empty_output_handling(tool_context: ToolContext) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p-empty"})
        if request.url.path == "/history/p-empty":
            return httpx.Response(
                200,
                json={
                    "p-empty": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {},
                    }
                },
            )
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ComfyClient(client=mock_client)
    tool = ComfyImageGenTool(client=client)

    result = await tool.execute(
        params={"prompt": "A prompt yielding no images"},
        context=tool_context,
    )

    assert result.success is False
    assert "produced no image outputs" in str(result.error)

    await client.aclose()


@pytest.mark.asyncio
async def test_client_invalid_json_responses() -> None:
    # 1. system_stats returns a list instead of dict
    def handler_list_stats(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"])

    client1 = ComfyClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler_list_stats))
    )
    with pytest.raises(ComfyUIError, match="Unexpected non-dict system_stats"):
        await client1.system_stats()

    # 2. object_info returns a list
    def handler_list_obj(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"])

    client2 = ComfyClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handler_list_obj)))
    with pytest.raises(ComfyUIError, match="Unexpected non-dict object_info"):
        await client2.object_info()

    # 3. /prompt returns invalid JSON on 200
    def handler_bad_json_200(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json {")

    client3 = ComfyClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler_bad_json_200))
    )
    with pytest.raises(ComfyUIError, match="Failed to parse ComfyUI /prompt response JSON"):
        await client3.queue_prompt({})

    # 4. /prompt returns non-json error on 500
    def handler_raw_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Nginx Error")

    client4 = ComfyClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handler_raw_500)))
    with pytest.raises(ComfyUIError, match="Internal Nginx Error"):
        await client4.queue_prompt({})


@pytest.mark.asyncio
async def test_wait_for_output_raises_when_completed_with_zero_images() -> None:
    """P6: wait_for_output() raises ComfyUIExecutionError when completed with 0 images."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "pid-1": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {"9": {"images": []}},
                }
            },
        )

    client = ComfyClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(ComfyUIExecutionError, match="completed but produced no image outputs"):
        await client.wait_for_output("pid-1", timeout_seconds=1.0)
    await client.aclose()


@pytest.mark.asyncio
async def test_tool_default_client_and_base_url() -> None:
    tool = ComfyImageGenTool(base_url="http://custom-comfy:8188")
    client = tool.client
    assert client.base_url == "http://custom-comfy:8188"


# ======================================================================================
# 5. Default Checkpoint Agreement (#1223)
# ======================================================================================


def test_agent_tool_and_comfy_engine_ask_for_the_same_default_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One unconfigured default reaches both paths, or the daemon is asked for two files.

    `generate_image` used to name `v1-5-pruned-emaonly.safetensors` — a file nothing in this
    repository installs or documents — while `ComfyUIImageEngine` asked for the declared
    `anillustrious_v4.safetensors`. Both are checked here against the single declaration, so
    restating either one diverges from it and fails.

    Killed by: src/uclone_x/tools/builtin/comfy_image_tool.py :: default_factory=default_comfy_checkpoint,
    Becomes: default="v1-5-pruned-emaonly.safetensors",
    """
    monkeypatch.delenv(COMFY_CHECKPOINT_ENV, raising=False)

    tool_default = ComfyImageGenParams(prompt="a cat").checkpoint
    engine_default = ComfyUIImageEngine()._checkpoint  # pyright: ignore[reportPrivateUsage]

    assert tool_default == engine_default
    assert tool_default == COMFY_DEFAULT_CHECKPOINT


def test_comfy_engine_takes_its_default_from_the_shared_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine reads the one declaration too — the agreement is not the tool's alone.

    Killed by: src/uclone_x/tools/builtin/image.py :: self._checkpoint = checkpoint or default_comfy_checkpoint()
    Becomes: self._checkpoint = checkpoint or "v1-5-pruned-emaonly.safetensors"
    """
    monkeypatch.delenv(COMFY_CHECKPOINT_ENV, raising=False)

    engine = ComfyUIImageEngine()

    assert engine._checkpoint == COMFY_DEFAULT_CHECKPOINT  # pyright: ignore[reportPrivateUsage]


def test_ucx_comfyui_checkpoint_overrides_the_agent_tool_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented override reaches `generate_image`, not only the engine.

    Set **after** import, because that is the case a frozen default gets wrong: a plain
    pydantic `default=` is evaluated once at class definition, so it would answer with
    whatever the environment held when this module was first imported.

    Killed by: src/uclone_x/tools/builtin/comfy_image_tool.py :: default_factory=default_comfy_checkpoint,
    Becomes: default=COMFY_DEFAULT_CHECKPOINT,
    """
    monkeypatch.setenv(COMFY_CHECKPOINT_ENV, "chosen_by_the_operator.safetensors")

    params = ComfyImageGenParams(prompt="a cat")

    assert params.checkpoint == "chosen_by_the_operator.safetensors"
    assert params.checkpoint != COMFY_DEFAULT_CHECKPOINT


def test_workflow_builder_resolves_an_unnamed_checkpoint_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build_txt2img_workflow` with no checkpoint asks for the configured one, per call.

    A function parameter default would be evaluated once at import and freeze the answer, so
    the resolution happens in the body. An explicit checkpoint still wins outright.

    Killed by: src/uclone_x/tools/builtin/comfy_image_tool.py :: "ckpt_name": checkpoint or default_comfy_checkpoint(),
    Becomes: "ckpt_name": checkpoint or COMFY_DEFAULT_CHECKPOINT,
    """
    monkeypatch.setenv(COMFY_CHECKPOINT_ENV, "configured_late.safetensors")

    workflow = build_txt2img_workflow(prompt="a cat")

    assert workflow["4"]["inputs"]["ckpt_name"] == "configured_late.safetensors"

    explicit = build_txt2img_workflow(prompt="a cat", checkpoint="explicit.safetensors")

    assert explicit["4"]["inputs"]["ckpt_name"] == "explicit.safetensors"
