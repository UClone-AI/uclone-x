"""Unit tests for the remote / ComfyUI / in-process hybrid image pipeline (#849, #1095)."""

from __future__ import annotations

import json
import os
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uclone_x.core.tool_results import canonical_tool_text
from uclone_x.errors import PlainRefusalError
from uclone_x.tools.base import artifact_path_from_url, linked_paths
from uclone_x.tools.builtin.comfy_client import COMFY_DEFAULT_CHECKPOINT
from uclone_x.tools.builtin.image import (
    ACCELERATE_MISSING_SENTENCE,
    COMFY_URL_ENV,
    COUNT_FOR_VARIETY_REFUSAL,
    DEFAULT_CHECKPOINTS,
    DIFFUSERS_GUIDANCE,
    DIFFUSERS_STEPS,
    GENERIC_FALLBACK_MODEL_ID,
    GPU_OUT_OF_MEMORY_MESSAGE,
    IMAGE_CHECKPOINT_ENV,
    IN_PROCESS_INSTALL_REQUIREMENTS,
    IN_PROCESS_REQUIREMENTS,
    SDXL_MEGAPIXEL_BUCKETS,
    STYLE_SUFFIXES,
    CheckpointResolution,
    ComfyUIImageEngine,
    GenerateImageParams,
    GenerateImageTool,
    ImageGenerationError,
    ImageGenerationResult,
    ImagePipelineDispatcher,
    LocalDiffusersImageEngine,
    RemoteCudaImageEngine,
    accelerate_is_installed,
    compute_deterministic_seed,
    device_wide_free_bytes,
    diffusers_install_hint,
    expand_checkpoint_path,
    find_prompt_conflicts,
    in_process_dependency_problems,
    long_prompt_embeds,
    out_of_memory_message,
    parse_nvidia_smi_free_bytes,
    resolve_aspect_dimensions,
    resolve_sampling,
    running_under_wsl,
    select_torch_device,
    sidecar_path_for,
    style_guided_prompt,
    torch_out_of_memory_types,
    version_release,
)
from uclone_x.tools.builtin.image import (
    nvidia_smi_output as real_nvidia_smi_output,
)
from uclone_x.tools.builtin.image import (
    nvml_free_bytes as real_nvml_free_bytes,
)
from uclone_x.tools.models import ToolContext


def _picture_rel(url: object) -> str:
    """The workspace path a result's picture link serves; fails the test if it names none."""
    rel = artifact_path_from_url(url)
    assert rel is not None, f"not a picture link: {url!r}"
    return rel


def _sidecar(workspace: Path, url: object) -> dict[str, Any]:
    """The recipe sidecar saved beside the picture a result links to."""
    return cast(
        dict[str, Any],
        json.loads((workspace / sidecar_path_for(_picture_rel(url))).read_text(encoding="utf-8")),
    )


def test_deterministic_seeding_reproducibility_and_turn_entropy() -> None:
    """Deterministic seeding must produce identical seeds for same inputs and vary with turn/prompt."""
    seed1 = compute_deterministic_seed("sess_abc", 1, "a majestic mountain")
    seed2 = compute_deterministic_seed("sess_abc", 1, "a majestic mountain")
    assert seed1 == seed2

    # Case insensitivity & whitespace trimming
    seed3 = compute_deterministic_seed("sess_abc", 1, "  A Majestic Mountain  ")
    assert seed1 == seed3

    # Turn entropy variation
    seed_turn2 = compute_deterministic_seed("sess_abc", 2, "a majestic mountain")
    assert seed1 != seed_turn2

    # Session variation
    seed_other_sess = compute_deterministic_seed("sess_xyz", 1, "a majestic mountain")
    assert seed1 != seed_other_sess

    # Explicit override
    assert compute_deterministic_seed("sess_abc", 1, "mountain", seed_override=42) == 42


def test_aspect_ratio_dimension_resolution() -> None:
    """Dimensions must align with standard multiples."""
    assert resolve_aspect_dimensions("1:1") == (768, 768)
    assert resolve_aspect_dimensions("16:9") == (896, 512)
    assert resolve_aspect_dimensions("9:16") == (512, 896)
    assert resolve_aspect_dimensions("4:3") == (768, 576)
    assert resolve_aspect_dimensions("3:4") == (576, 768)


@pytest.mark.asyncio
async def test_remote_cuda_engine_is_available_when_healthy() -> None:
    """Remote CUDA engine reports available only when endpoint responds HTTP 200."""
    engine = RemoteCudaImageEngine(base_url="http://10.0.0.50:8000")

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        assert await engine.is_available() is True

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        assert await engine.is_available() is False


@pytest.mark.asyncio
async def test_remote_cuda_engine_generation_flow() -> None:
    """Remote CUDA worker returns structured image result."""
    engine = RemoteCudaImageEngine(base_url="http://10.0.0.50:8000")
    dummy_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/png", "x-device-info": "NVIDIA RTX 5070 Ti"}
        mock_resp.content = dummy_bytes
        mock_post.return_value = mock_resp

        res = await engine.generate(
            prompt="lake at dawn",
            negative_prompt="",
            width=1024,
            height=1024,
            seed=12345,
            style="photorealistic",
        )

        assert res.image_bytes == dummy_bytes
        assert res.seed == 12345
        assert res.engine_name == "remote-cuda-fastapi"
        assert res.device_info == "NVIDIA RTX 5070 Ti"


def _dispatcher_mocks(
    *, remote: bool, comfy: bool, local: bool
) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    """Three engine mocks with the given availability, each returning a named result."""
    mock_remote = AsyncMock(spec=RemoteCudaImageEngine)
    mock_remote.is_available.return_value = remote
    mock_remote.base_url = "http://gpu.local:9000" if remote else ""
    mock_remote.generate.return_value = ImageGenerationResult(
        image_bytes=b"remote_image_bytes",
        seed=100,
        engine_name="remote-cuda-fastapi",
        device_info="RTX 5070 Ti",
        duration_seconds=0.8,
        width=1024,
        height=1024,
    )

    mock_comfy = AsyncMock(spec=ComfyUIImageEngine)
    mock_comfy.is_available.return_value = comfy
    mock_comfy.base_url = "http://127.0.0.1:8188"
    mock_comfy.generate.return_value = ImageGenerationResult(
        image_bytes=b"comfy_image_bytes",
        seed=200,
        engine_name="comfyui-local",
        device_info="ComfyUI daemon at http://127.0.0.1:8188",
        duration_seconds=3.0,
        width=1024,
        height=1024,
    )

    mock_local = AsyncMock(spec=LocalDiffusersImageEngine)
    mock_local.is_available.return_value = local
    mock_local.resolve_checkpoint = MagicMock(return_value="/models/anillustrious_v4.safetensors")
    mock_local.generate.return_value = ImageGenerationResult(
        image_bytes=b"diffusers_image_bytes",
        seed=300,
        engine_name="diffusers-sdxl",
        device_info="mps (arm)",
        duration_seconds=9.0,
        width=1024,
        height=1024,
    )
    return mock_remote, mock_comfy, mock_local


@pytest.mark.asyncio
async def test_dispatcher_prefers_remote_cuda_over_every_local_engine() -> None:
    """A reachable remote worker wins even when both local engines are also ready.

    Killed by: src/uclone_x/tools/builtin/image.py :: ("Remote CUDA worker", "remote-cuda", self._remote_engine))
    Becomes: ("Remote CUDA worker", "remote-cuda", self._local_engine))
    """
    mock_remote, mock_comfy, mock_local = _dispatcher_mocks(remote=True, comfy=True, local=True)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_remote, comfy_engine=mock_comfy, local_engine=mock_local
    )

    res = await dispatcher.dispatch(
        prompt="cosmic nebula",
        negative_prompt="",
        aspect_ratio="1:1",
        seed=100,
        style="artistic",
    )

    assert res.engine_name == "remote-cuda-fastapi"
    mock_remote.generate.assert_awaited_once()
    mock_comfy.generate.assert_not_awaited()
    mock_local.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatcher_prefers_a_detected_comfyui_over_the_in_process_engine() -> None:
    """With no remote worker, a running ComfyUI is chosen ahead of in-process diffusers (#1095).

    Killed by: src/uclone_x/tools/builtin/image.py :: ("detected local ComfyUI daemon", "comfyui-local", self._comfy_engine))
    Becomes: ("detected local ComfyUI daemon", "comfyui-local", self._local_engine))
    """
    mock_remote, mock_comfy, mock_local = _dispatcher_mocks(remote=False, comfy=True, local=True)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_remote, comfy_engine=mock_comfy, local_engine=mock_local
    )

    res = await dispatcher.dispatch(
        prompt="cosmic nebula",
        negative_prompt="",
        aspect_ratio="1:1",
        seed=200,
        style="artistic",
    )

    assert res.engine_name == "comfyui-local"
    assert res.image_bytes == b"comfy_image_bytes"
    mock_local.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatcher_runs_in_process_when_no_daemon_is_running() -> None:
    """The daemon-free baseline: no remote, no ComfyUI, and an image is still produced.

    Killed by: src/uclone_x/tools/builtin/image.py :: ("in-process diffusers engine", "diffusers-sdxl", self._local_engine))
    Becomes: ("in-process diffusers engine", "diffusers-sdxl", self._comfy_engine))
    """
    mock_remote, mock_comfy, mock_local = _dispatcher_mocks(remote=False, comfy=False, local=True)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_remote, comfy_engine=mock_comfy, local_engine=mock_local
    )

    res = await dispatcher.dispatch(
        prompt="cosmic nebula",
        negative_prompt="",
        aspect_ratio="1:1",
        seed=300,
        style="anime",
    )

    assert res.engine_name == "diffusers-sdxl"
    assert res.image_bytes == b"diffusers_image_bytes"
    mock_local.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatcher_raises_typed_error_when_all_engines_unavailable() -> None:
    """Dispatcher fails fast when no engine of the three is ready."""
    mock_remote, mock_comfy, mock_local = _dispatcher_mocks(remote=False, comfy=False, local=False)
    mock_local.checkpoint_resolution = MagicMock(return_value=CheckpointResolution("unconfigured"))
    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_remote, comfy_engine=mock_comfy, local_engine=mock_local
    )

    with pytest.raises(ImageGenerationError) as exc_info:
        await dispatcher.dispatch(
            prompt="test prompt",
            negative_prompt="",
            aspect_ratio="1:1",
            seed=300,
            style="photorealistic",
        )
    assert "No image generation engine available" in str(exc_info.value)
    assert "Local Private-First enforcement" in str(exc_info.value)


def test_diagnostics_names_all_three_engines_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One "nothing available" is not actionable; each engine declines for its own reason (P6).

    Killed by: src/uclone_x/tools/builtin/image.py :: f"No ComfyUI daemon answered at {self._comfy_engine.base_url} "
    Becomes: ""
    """
    monkeypatch.delenv(COMFY_URL_ENV, raising=False)
    mock_remote, mock_comfy, mock_local = _dispatcher_mocks(remote=False, comfy=False, local=False)
    mock_local.checkpoint_resolution = MagicMock(return_value=CheckpointResolution("unconfigured"))
    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_remote, comfy_engine=mock_comfy, local_engine=mock_local
    )

    diagnostics = dispatcher.diagnostics()

    assert "UCX_IMAGE_REMOTE_URL is not set" in diagnostics
    assert "No ComfyUI daemon answered at http://127.0.0.1:8188" in diagnostics
    assert f"set {COMFY_URL_ENV}, to enable it" in diagnostics
    assert "ucx media status" in diagnostics or "diffusers" in diagnostics


def test_diagnostics_does_not_tell_a_user_to_set_a_variable_they_already_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured-but-silent daemon is a daemon problem, not an unset-variable problem.

    The advice was "start one, or set UCX_COMFYUI_URL" for every reader, including the one
    whose UCX_COMFYUI_URL is set and whose daemon is down (reviewer, PR #1096).

    Killed by: src/uclone_x/tools/builtin/image.py :: if os.getenv(COMFY_URL_ENV):
    Becomes: if False:
    """
    monkeypatch.setenv(COMFY_URL_ENV, "http://127.0.0.1:9999")
    mock_remote, mock_comfy, mock_local = _dispatcher_mocks(remote=False, comfy=False, local=False)
    mock_local.checkpoint_resolution = MagicMock(return_value=CheckpointResolution("unconfigured"))
    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_remote, comfy_engine=mock_comfy, local_engine=mock_local
    )

    diagnostics = dispatcher.diagnostics()

    assert f"the address {COMFY_URL_ENV} names" in diagnostics
    assert f"set {COMFY_URL_ENV}, to enable it" not in diagnostics


@pytest.mark.asyncio
async def test_generate_image_tool_run_writes_artifact_and_returns_provenance(
    tmp_path: Path,
) -> None:
    """GenerateImageTool writes image bytes to disk and returns complete provenance."""
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    mock_dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"fake_png_binary_data",
        seed=55555,
        engine_name="diffusers-sdxl",
        device_info="Apple M3",
        duration_seconds=2.2,
        width=1024,
        height=1024,
    )

    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="test-agent",
        workspace_root=tmp_path,
        session_id="sess_img_test",
    )

    params = GenerateImageParams(
        prompt="cyberpunk cityscape at dusk",
        style="artistic",
        seed_override=55555,
    )
    result = await tool.run(params, context)

    assert result["status"] == "success"
    assert result["seed"] == 55555
    # The provenance lives in the sidecar beside the picture, not in the result (#2013).
    sidecar = _sidecar(tmp_path, result["relative_url"])
    assert sidecar["engine"] == "diffusers-sdxl"
    assert sidecar["device"] == "Apple M3"
    assert sidecar["width"] == 1024
    assert sidecar["height"] == 1024
    assert sidecar["seed"] == 55555

    # Verify written file on disk, found from the link the result carries
    img_path = tmp_path / _picture_rel(result["relative_url"])
    assert img_path.is_file()
    assert img_path.read_bytes() == b"fake_png_binary_data"


@pytest.mark.asyncio
async def test_generate_image_recipe_hash_includes_negative_prompt(tmp_path: Path) -> None:
    """Recipe hash must differ when negative_prompt differs.

    Killed by: src/uclone_x/tools/builtin/image.py :: f"{params.prompt}__{params.negative_prompt}__{actual_seed}__{gen_result.engine_name}"
    Becomes: f"{params.prompt}____{actual_seed}__{gen_result.engine_name}"
    """
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    mock_dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"fake_png",
        seed=12345,
        engine_name="diffusers-sdxl",
        device_info="Apple M3",
        duration_seconds=1.0,
        width=512,
        height=512,
    )
    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="test-agent",
        workspace_root=tmp_path,
        session_id="sess_hash_test",
    )

    res1 = await tool.run(
        GenerateImageParams(
            prompt="portrait of a wizard",
            negative_prompt="blurry",
            seed_override=12345,
        ),
        context,
    )
    res2 = await tool.run(
        GenerateImageParams(
            prompt="portrait of a wizard",
            negative_prompt="cartoon, oversaturated",
            seed_override=12345,
        ),
        context,
    )

    hash1 = _sidecar(tmp_path, res1["relative_url"])["recipe_hash"]
    hash2 = _sidecar(tmp_path, res2["relative_url"])["recipe_hash"]
    assert hash1 and hash2
    assert hash1 != hash2


@pytest.mark.asyncio
async def test_generate_image_tool_execute_decorates_artifacts_field(
    tmp_path: Path,
) -> None:
    """BaseTool execute wrapper attaches image path to ToolResult.artifacts."""
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    mock_dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"png_data",
        seed=111,
        engine_name="remote-cuda-fastapi",
        device_info="RTX 5070 Ti",
        duration_seconds=0.7,
        width=1024,
        height=1024,
    )

    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="test-agent",
        workspace_root=tmp_path,
        session_id="sess_exec",
    )

    result = await tool.execute(
        params={"prompt": "ancient temple hidden in jungle"},
        context=context,
    )

    assert result.success is True
    assert len(result.artifacts) == 1
    assert "artifacts/sess_exec/images/img_" in result.artifacts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 3])
async def test_a_generated_picture_is_saved_in_the_session_directory(
    tmp_path: Path, count: int
) -> None:
    """Default saves land in `artifacts/<session id>/images/`, the directory the Docs &
    Artifacts listing reads for a session (#1390), for one picture and for a batch.

    Killed by: src/uclone_x/tools/builtin/image.py :: images_dir = f"artifacts/{session_id}/images"
    Becomes: images_dir = f"artifacts/images"
    """
    dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"png",
        seed=1,
        engine_name="mock-engine",
        device_info="cpu",
        duration_seconds=0.1,
        width=64,
        height=64,
    )
    context = ToolContext(agent_id="artist", workspace_root=tmp_path, session_id="sess_1390")

    result = await GenerateImageTool(dispatcher=dispatcher).execute(
        params={"prompt": "lighthouse", "count": count}, context=context
    )

    assert result.success is True
    assert len(result.artifacts) == count
    for rel in result.artifacts:
        assert rel.startswith("artifacts/sess_1390/images/img_"), rel
        assert (tmp_path / rel).is_file()


@pytest.mark.asyncio
async def test_generate_image_tool_run_batch_count(tmp_path: Path) -> None:
    """GenerateImageTool runs batch generation when count > 1."""
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)

    def _fake_batch_dispatch(
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        seed: int,
        style: str,
        own: str | None = None,
    ) -> ImageGenerationResult:
        return ImageGenerationResult(
            image_bytes=f"png_{seed}".encode(),
            seed=seed,
            engine_name="diffusers-sdxl",
            device_info="Apple M3",
            duration_seconds=1.0,
            width=768,
            height=576,
        )

    mock_dispatcher.dispatch.side_effect = _fake_batch_dispatch

    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="test-artist",
        workspace_root=tmp_path,
        session_id="sess_batch",
    )

    params = GenerateImageParams(
        prompt="gundam girl robot armor",
        count=3,
        seed_override=2000,
    )
    result = await tool.run(params, context)

    assert result["status"] == "success"
    assert result["count"] == 3
    assert len(result["images"]) == 3
    assert mock_dispatcher.dispatch.call_count == 3
    # Seeds should be progressive
    assert result["images"][0]["seed"] == 2000
    assert result["images"][1]["seed"] == 2001
    assert result["images"][2]["seed"] == 2002
    # The shared prompt is said once, at the top (#2013)
    assert result["prompt"] == "gundam girl robot armor"
    # No gallery text, path lists or link lists: each picture is named once, by its link
    for dropped in ("markdown_gallery", "paths", "relative_urls", "meta_paths"):
        assert dropped not in result
    paths = linked_paths(result)
    assert len(paths) == 3
    assert paths == [_picture_rel(img["relative_url"]) for img in result["images"]]

    # Verify all 3 files exist on disk, found from the links
    for img_meta, rel in zip(result["images"], paths, strict=True):
        f_path = tmp_path / rel
        assert f_path.is_file()
        assert f_path.read_bytes() == f"png_{img_meta['seed']}".encode()


_DROPPED_RESULT_KEYS = (
    "engine",
    "device",
    "path",
    "paths",
    "meta_path",
    "meta_paths",
    "recipe_hash",
    "markdown_gallery",
    "relative_urls",
    "prompts",
    "width",
    "height",
    "mime_type",
    "duration_seconds",
    "bytes_written",
)
_FILL_KEYS = {"prompt_changes", "prompt_added", "negative_added"}


def _batch_tool() -> GenerateImageTool:
    """A tool whose engine draws each picture from its seed, the same way every time."""
    dispatcher = AsyncMock(spec=ImagePipelineDispatcher)

    def _draw(
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        seed: int,
        style: str,
        own: str | None = None,
    ) -> ImageGenerationResult:
        return ImageGenerationResult(
            image_bytes=f"png_{seed}".encode(),
            seed=seed,
            engine_name="diffusers-sdxl",
            device_info="Apple M3",
            duration_seconds=1.0,
            width=768,
            height=576,
        )

    dispatcher.dispatch.side_effect = _draw
    return GenerateImageTool(dispatcher=dispatcher)


def _assert_no_dropped_key(value: object) -> None:
    """No dict anywhere in `value` carries a key the slim result dropped (#2013)."""
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        for key in _DROPPED_RESULT_KEYS:
            assert key not in mapping, key
        for inner in mapping.values():
            _assert_no_dropped_key(inner)
    elif isinstance(value, list):
        for inner in cast(list[Any], value):
            _assert_no_dropped_key(inner)


@pytest.mark.asyncio
async def test_a_same_prompt_batch_result_says_each_picture_once_and_the_prompt_once(
    tmp_path: Path,
) -> None:
    """Pins the slim batch shape the model is sent, and every later request resends (#2013).

    A count=4 batch of one prompt: the result's keys are exactly status, count, prompt,
    style, aspect_ratio and images (plus a fill key only when the fill produced one), each
    picture is exactly its link and seed, the prompt text appears once in the serialized
    result, and none of the dropped keys (engine, device, paths, sidecar paths, recipe hash,
    gallery text...) appear anywhere in it. The engine and recipe stay reachable from each
    picture's sidecar.
    """
    prompt = "gundam girl robot armor, standing on a rooftop at dusk"
    context = ToolContext(agent_id="a", workspace_root=tmp_path, session_id="s")

    result = await _batch_tool().run(
        GenerateImageParams(prompt=prompt, count=4, seed_override=2000), context
    )

    base = {"status", "count", "prompt", "style", "aspect_ratio", "images"}
    assert base <= set(result) <= base | _FILL_KEYS
    for key in _FILL_KEYS & set(result):
        assert result[key], f"{key} is sent only when it says something"
    assert result["count"] == 4
    assert result["prompt"] == prompt
    assert [set(img) for img in result["images"]] == [{"relative_url", "seed"}] * 4
    assert [img["seed"] for img in result["images"]] == [2000, 2001, 2002, 2003]
    assert json.dumps(result, ensure_ascii=False).count(prompt) == 1
    _assert_no_dropped_key(result)

    for img in result["images"]:
        sidecar = _sidecar(tmp_path, img["relative_url"])
        assert sidecar["prompt"] == prompt
        assert sidecar["seed"] == img["seed"]
        assert sidecar["engine"] == "diffusers-sdxl"
        assert sidecar["recipe_hash"]


@pytest.mark.asyncio
async def test_a_batch_of_distinct_prompts_puts_each_prompt_on_its_own_picture(
    tmp_path: Path,
) -> None:
    """Pins the slim shape for `prompts=[a, b]` (#2013): no shared prompt, one per picture.

    Pictures drawn from different prompts share none, so the result has no top-level
    `prompt` (and no `prompts` list); each picture carries its own prompt beside its link
    and seed, and each prompt appears once in the serialized result.
    """
    prompts = ["1girl, silver hair, hero pose", "1girl, silver hair, sitting on a chair"]
    context = ToolContext(agent_id="a", workspace_root=tmp_path, session_id="s")

    result = await _batch_tool().run(
        GenerateImageParams(prompts=prompts, seed_override=10), context
    )

    assert "prompt" not in result
    assert result["count"] == 2
    assert [img["prompt"] for img in result["images"]] == prompts
    for img in result["images"]:
        assert (
            {"relative_url", "seed", "prompt"}
            <= set(img)
            <= ({"relative_url", "seed", "prompt"} | _FILL_KEYS)
        )
    text = json.dumps(result, ensure_ascii=False)
    for p in prompts:
        assert text.count(p) == 1
    _assert_no_dropped_key(result)
    assert len(linked_paths(result)) == 2


def test_find_prompt_conflicts_detection_and_safeguards() -> None:
    """Exclusion tags from negative prompt are detected in positive prompt, ignoring boilerplate and substrings.

    Killed by: src/uclone_x/tools/builtin/image.py :: if len(clean) < 2 or clean in GENERIC_NEGATIVE_BOILERPLATE or clean in seen:
    Becomes: if len(clean) < 2 or clean in seen:
    """
    # 1. Contradiction detected
    assert find_prompt_conflicts(
        prompt="1girl, semi-transparent skin-toned underwear, solo",
        negative_prompt="worst quality, blurry, underwear, panties",
    ) == ["underwear"]

    # 2. Case-insensitive and weighted tags handled
    assert find_prompt_conflicts(
        prompt="1girl, Cute Panties, bedroom",
        negative_prompt="(panties:1.2), (bad anatomy:1.1)",
    ) == ["panties"]

    # 3. Substring false-positive prevention (e.g. pants vs panties)
    assert (
        find_prompt_conflicts(
            prompt="1girl, denim pants, walking outside",
            negative_prompt="panties, underwear",
        )
        == []
    )

    # 4. Generic quality boilerplate is ignored in conflict detection
    assert (
        find_prompt_conflicts(
            prompt="1girl, garden, blurry background, bokeh",
            negative_prompt="worst quality, blurry, bad anatomy",
        )
        == []
    )


@pytest.mark.asyncio
async def test_generate_image_tool_rejects_conflicting_positive_and_negative_terms(
    tmp_path: Path,
) -> None:
    """Contradictory positive and negative prompts fail fast with PlainRefusalError before GPU dispatch.

    Killed by: src/uclone_x/tools/builtin/image.py :: conflicts = find_prompt_conflicts(params.prompt, params.negative_prompt)
    Becomes: conflicts = []
    """
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="test-artist",
        workspace_root=tmp_path,
        session_id="sess_conflict",
    )

    params = GenerateImageParams(
        prompt="1girl, semi-transparent skin-toned underwear, solo",
        negative_prompt="worst quality, blurry, underwear",
    )

    with pytest.raises(PlainRefusalError) as exc_info:
        await tool.run(params, context)

    assert "Prompt conflict detected" in str(exc_info.value)
    assert "'underwear'" in str(exc_info.value)
    # GPU dispatch should never be called when conflict is present
    mock_dispatcher.dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_generate_image_tool_supports_diverse_prompts_list(
    tmp_path: Path,
) -> None:
    """GenerateImageTool accepts a list of distinct prompts for multi-pose / multi-scene generation.

    Killed by: src/uclone_x/tools/builtin/image.py :: params.prompts if params.prompts is not None else [params.prompt] * params.count
    Becomes: [params.prompt] * params.count
    """
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)

    def _fake_multi_prompt_dispatch(
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        seed: int,
        style: str,
        own: str | None = None,
    ) -> ImageGenerationResult:
        return ImageGenerationResult(
            image_bytes=f"png_{prompt[:10]}_{seed}".encode(),
            seed=seed,
            engine_name="diffusers-sdxl",
            device_info="Apple M3",
            duration_seconds=0.5,
            width=768,
            height=768,
        )

    mock_dispatcher.dispatch.side_effect = _fake_multi_prompt_dispatch

    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="test-artist",
        workspace_root=tmp_path,
        session_id="sess_multi_prompt",
    )

    distinct_prompts = [
        "1girl, silver hair, confident standing hero pose, eye level",
        "1girl, silver hair, sitting on chair crossed legs, casual",
        "1girl, silver hair, dynamic sprint action pose, low angle",
    ]
    params = GenerateImageParams(
        prompts=distinct_prompts,
        negative_prompt="worst quality, blurry",
        seed_override=5000,
    )

    result = await tool.run(params, context)

    assert result["status"] == "success"
    assert result["count"] == 3
    # Each picture carries its own prompt; there is no shared one and no `prompts` list (#2013)
    assert "prompts" not in result
    assert "prompt" not in result
    assert [img["prompt"] for img in result["images"]] == distinct_prompts
    assert mock_dispatcher.dispatch.call_count == 3

    # Dispatcher was called with each distinct prompt
    assert mock_dispatcher.dispatch.call_args_list[0].kwargs["prompt"] == distinct_prompts[0]
    assert mock_dispatcher.dispatch.call_args_list[1].kwargs["prompt"] == distinct_prompts[1]
    assert mock_dispatcher.dispatch.call_args_list[2].kwargs["prompt"] == distinct_prompts[2]

    # Verify per-image prompt in metadata
    assert result["images"][0]["prompt"] == distinct_prompts[0]
    assert result["images"][1]["prompt"] == distinct_prompts[1]
    assert result["images"][2]["prompt"] == distinct_prompts[2]


@pytest.mark.asyncio
async def test_generate_image_tool_rejects_crammed_multiple_poses_when_count_greater_than_one(
    tmp_path: Path,
) -> None:
    """A single prompt listing multiple poses with count > 1 fails fast with PlainRefusalError.

    Killed by: src/uclone_x/tools/builtin/image.py :: if re.search(composite_pose_pattern, params.prompt, re.IGNORECASE):
    Becomes: if False:
    """
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="test-artist",
        workspace_root=tmp_path,
        session_id="sess_crammed_poses",
    )

    params = GenerateImageParams(
        prompt="K-pop idol, dynamic poses including standing gracefully, seated on chair, reclining on sofa",
        count=3,
    )

    with pytest.raises(PlainRefusalError) as exc_info:
        await tool.run(params, context)

    assert "Multiple poses detected in a single prompt" in str(exc_info.value)
    mock_dispatcher.dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_generate_image_tool_execute_batch_artifacts(tmp_path: Path) -> None:
    """Execute wrapper registers all paths in artifacts tuple when count > 1."""
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)

    def _fake_exec_dispatch(
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        seed: int,
        style: str,
        own: str | None = None,
    ) -> ImageGenerationResult:
        return ImageGenerationResult(
            image_bytes=b"dummy",
            seed=seed,
            engine_name="mock",
            device_info="cpu",
            duration_seconds=0.1,
            width=512,
            height=512,
        )

    mock_dispatcher.dispatch.side_effect = _fake_exec_dispatch

    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="test-artist",
        workspace_root=tmp_path,
        session_id="sess_batch_exec",
    )

    result = await tool.execute(
        params={"prompt": "cybernetic warrior", "count": 2},
        context=context,
    )

    assert result.success is True
    assert len(result.artifacts) == 2
    assert isinstance(result.output, dict)
    assert result.output["count"] == 2


@pytest.mark.asyncio
async def test_generate_image_tool_turn_index_different_seeds(tmp_path: Path) -> None:
    """Distinct turn indices must yield distinct seeds and distinct artifact paths for identical prompts."""
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)

    def _fake_exec_dispatch(
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        seed: int,
        style: str,
        own: str | None = None,
    ) -> ImageGenerationResult:
        return ImageGenerationResult(
            image_bytes=f"image_{seed}".encode(),
            seed=seed,
            engine_name="mock",
            device_info="cpu",
            duration_seconds=0.1,
            width=512,
            height=512,
        )

    mock_dispatcher.dispatch.side_effect = _fake_exec_dispatch
    tool = GenerateImageTool(dispatcher=mock_dispatcher)

    ctx_turn1 = ToolContext(
        agent_id="artist",
        workspace_root=tmp_path,
        session_id="sess_turns",
        turn_index=1,
    )
    result1 = await tool.execute(
        params={"prompt": "cyberpunk city street", "count": 2},
        context=ctx_turn1,
    )

    ctx_turn2 = ToolContext(
        agent_id="artist",
        workspace_root=tmp_path,
        session_id="sess_turns",
        turn_index=2,
    )
    result2 = await tool.execute(
        params={"prompt": "cyberpunk city street", "count": 2},
        context=ctx_turn2,
    )

    assert result1.success is True
    assert result2.success is True
    out1 = cast(dict[str, Any], result1.output)
    out2 = cast(dict[str, Any], result2.output)
    images1 = cast(list[dict[str, Any]], out1["images"])
    images2 = cast(list[dict[str, Any]], out2["images"])
    seed1 = images1[0]["seed"]
    seed2 = images2[0]["seed"]
    assert seed1 != seed2
    assert result1.artifacts != result2.artifacts

    # Fallback to agent_delegate._turn_counter when context.turn_index is 0
    class DummyAgent:
        _turn_counter = 3

    ctx_turn3 = ToolContext(
        agent_id="artist",
        workspace_root=tmp_path,
        session_id="sess_turns",
        turn_index=0,
        agent_delegate=DummyAgent(),
    )
    result3 = await tool.execute(
        params={"prompt": "cyberpunk city street", "count": 2},
        context=ctx_turn3,
    )
    assert result3.success is True
    out3 = cast(dict[str, Any], result3.output)
    images3 = cast(list[dict[str, Any]], out3["images"])
    seed3 = images3[0]["seed"]
    assert seed3 != seed1
    assert seed3 != seed2


@pytest.mark.asyncio
async def test_output_path_leading_slash_normalized_safely(tmp_path: Path) -> None:
    """Leading slash in output_path is normalized safely to avoid false path traversal rejection."""
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    mock_dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"dummy_png",
        seed=42,
        engine_name="mock",
        device_info="test",
        duration_seconds=0.1,
        width=1024,
        height=1024,
    )

    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="test-agent",
        workspace_root=tmp_path,
        session_id="sess_slash",
    )

    result = await tool.execute(
        params={
            "prompt": "sunset over ocean",
            "output_path": "/artifacts/images/custom_img.png",
        },
        context=context,
    )

    assert result.success is True
    assert (tmp_path / "artifacts/images/custom_img.png").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 3])
async def test_the_result_the_model_reads_names_no_workspace_directory(
    tmp_path: Path, count: int
) -> None:
    """The model is handed the served link and nothing it could turn into a host (#1618).

    In a fresh-HOME run the Artist linked its picture as
    `https://ucx-fresh-test2-pypi022/work/api/artifacts/...`: the result carried the file's
    absolute path beside the link, and the model joined the workspace directory onto it.
    The text the model reads is checked, not the dict, because that is what it copies from.
    Both the single-image and the batch shape are covered; each carried the path twice.

    Killed by: src/uclone_x/tools/builtin/image.py :: rel_path = str(dest_path.relative_to(context.require_workspace().resolve()))
    Becomes: rel_path = str(dest_path)
    Killed by: src/uclone_x/tools/builtin/image.py :: batch_path = str(dest_path.relative_to(context.require_workspace().resolve()))
    Becomes: batch_path = str(dest_path)
    """
    workspace = tmp_path / "ucx-fresh-test2-pypi022" / "work"
    workspace.mkdir(parents=True)
    dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"png",
        seed=7,
        engine_name="mock",
        device_info="test",
        duration_seconds=0.1,
        width=64,
        height=64,
    )
    context = ToolContext(agent_id="artist", workspace_root=workspace, session_id="s")

    result = await GenerateImageTool(dispatcher=dispatcher).execute(
        params={"prompt": "night train", "count": count}, context=context
    )

    assert result.success is True
    text = canonical_tool_text(result.output)
    assert "ucx-fresh-test2-pypi022" not in text
    output = cast(dict[str, Any], result.output)
    links = (
        [img["relative_url"] for img in output["images"]] if count > 1 else [output["relative_url"]]
    )
    assert len(links) == count
    assert "relative_urls" not in output
    for link in links:
        assert link.startswith("/api/artifacts/content?path=artifacts/s/images/img_")


@pytest.mark.asyncio
async def test_a_file_name_with_a_space_or_parenthesis_still_makes_one_markdown_link(
    tmp_path: Path,
) -> None:
    """`](... (1).png)` would end the markdown link at the first `)`; the path is encoded.

    Killed by: src/uclone_x/tools/base.py :: quote(rel_path, safe='/')
    Becomes: rel_path
    """
    dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"png",
        seed=7,
        engine_name="mock",
        device_info="test",
        duration_seconds=0.1,
        width=64,
        height=64,
    )
    context = ToolContext(agent_id="artist", workspace_root=tmp_path, session_id="s")

    result = await GenerateImageTool(dispatcher=dispatcher).execute(
        params={"prompt": "night train", "output_path": "artifacts/images/night train (1).png"},
        context=context,
    )

    assert result.success is True
    output = cast(dict[str, Any], result.output)
    assert output["relative_url"] == (
        "/api/artifacts/content?path=artifacts/images/night%20train%20%281%29.png"
    )


@pytest.mark.asyncio
async def test_output_path_traversal_still_rejected(tmp_path: Path) -> None:
    """Escaping workspace root via ../ is strictly rejected with PathTraversalError."""
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="test-agent",
        workspace_root=tmp_path,
        session_id="sess_escape",
    )

    result = await tool.execute(
        params={
            "prompt": "escape attempt",
            "output_path": "../../escaped_file.png",
        },
        context=context,
    )

    assert result.success is False
    assert "escapes workspace root" in (result.error or "")


def test_style_guided_prompt_appends_the_preset_and_leaves_an_unknown_one_alone() -> None:
    """SDXL takes its style from the prompt, so an unappended preset would do nothing.

    Killed by: src/uclone_x/tools/builtin/image.py :: return f"{prompt}, {suffix}" if suffix else prompt
    Becomes: return prompt
    """
    guided = style_guided_prompt("a castle", "anime")

    assert guided.startswith("a castle, ")
    assert "anime illustration" in guided
    assert style_guided_prompt("a castle", "no-such-style") == "a castle"


def test_resolve_checkpoint_prefers_the_configured_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "custom.safetensors"
    checkpoint.write_bytes(b"x")

    engine = LocalDiffusersImageEngine(checkpoint_path=str(checkpoint))

    assert engine.resolve_checkpoint() == str(checkpoint)


def test_resolve_checkpoint_reads_the_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "env.safetensors"
    checkpoint.write_bytes(b"x")
    monkeypatch.setenv(IMAGE_CHECKPOINT_ENV, str(checkpoint))

    assert LocalDiffusersImageEngine().resolve_checkpoint() == str(checkpoint)


def test_resolve_checkpoint_returns_none_for_a_configured_path_that_is_not_there(
    tmp_path: Path,
) -> None:
    """A named-but-absent checkpoint must not be reported as the one that would load (P6).

    Killed by: src/uclone_x/tools/builtin/image.py :: state: CheckpointState = "present" if os.path.exists(path) else "missing"
    Becomes: state: CheckpointState = "present"
    """
    engine = LocalDiffusersImageEngine(checkpoint_path=str(tmp_path / "absent.safetensors"))

    assert engine.resolve_checkpoint() is None


def test_checkpoint_resolution_separates_absent_from_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three states must be three values, not two Nones and a path (#1120, P6).

    `resolve_checkpoint` answers "is there a file"; it cannot answer "did the user
    configure one", and every message built on it told a user who had mistyped
    `UCX_IMAGE_CHECKPOINT` to go and set `UCX_IMAGE_CHECKPOINT`.

    Killed by: src/uclone_x/tools/builtin/image.py :: return CheckpointResolution(state, path, source, literal)
    Becomes: return CheckpointResolution("unconfigured")
    """
    monkeypatch.delenv(IMAGE_CHECKPOINT_ENV, raising=False)
    present = tmp_path / "there.safetensors"
    present.write_bytes(b"x")
    absent = tmp_path / "typo.safetensors"

    configured_present = LocalDiffusersImageEngine(
        checkpoint_path=str(present)
    ).checkpoint_resolution()
    configured_absent = LocalDiffusersImageEngine(
        checkpoint_path=str(absent)
    ).checkpoint_resolution()
    with patch("os.path.exists", return_value=False):
        nothing_configured = LocalDiffusersImageEngine().checkpoint_resolution()

    assert (configured_present.state, configured_present.path) == ("present", str(present))
    assert (configured_absent.state, configured_absent.path) == ("missing", str(absent))
    assert (nothing_configured.state, nothing_configured.path) == ("unconfigured", None)
    # The deliverable is that a reader can tell them apart, so assert on the prose too.
    described = {
        configured_present.describe(),
        configured_absent.describe(),
        nothing_configured.describe(),
    }
    assert len(described) == 3
    assert str(absent) in configured_absent.describe()
    assert str(absent) not in nothing_configured.describe()


def test_checkpoint_resolution_names_the_variable_that_set_an_absent_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped `UCX_IMAGE_CHECKPOINT` is told to correct it, not to set it (#1120).

    Killed by: src/uclone_x/tools/builtin/image.py :: if self.source == IMAGE_CHECKPOINT_ENV:
    Becomes: if False:
    """
    absent = tmp_path / "typo.safetensors"
    monkeypatch.setenv(IMAGE_CHECKPOINT_ENV, str(absent))

    resolution = LocalDiffusersImageEngine().checkpoint_resolution()

    assert resolution.state == "missing"
    assert resolution.source == IMAGE_CHECKPOINT_ENV
    assert resolution.usable is False
    described = resolution.describe()
    assert f"{IMAGE_CHECKPOINT_ENV} is set to '{absent}'" in described
    assert "Correct that path" in described


def test_diagnostics_distinguishes_a_mistyped_checkpoint_from_an_unset_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatcher's failure text is the second reader of the collapsed state (#1120).

    Killed by: src/uclone_x/tools/builtin/image.py :: f"{resolution.describe()} Run `ucx media status` to see this from the "
    Becomes: "Run `ucx media status` to see this from the "
    """
    monkeypatch.delenv(COMFY_URL_ENV, raising=False)
    absent = str(tmp_path / "typo.safetensors")
    mock_remote, mock_comfy, mock_local = _dispatcher_mocks(remote=False, comfy=False, local=False)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_remote, comfy_engine=mock_comfy, local_engine=mock_local
    )

    mock_local.checkpoint_resolution = MagicMock(
        return_value=CheckpointResolution("missing", absent, IMAGE_CHECKPOINT_ENV)
    )
    with patch("uclone_x.tools.builtin.image.in_process_dependency_problems", return_value=()):
        mistyped = dispatcher.diagnostics()
        mock_local.checkpoint_resolution = MagicMock(
            return_value=CheckpointResolution("unconfigured")
        )
        unset = dispatcher.diagnostics()

    assert absent in mistyped
    assert absent not in unset
    assert mistyped != unset


def test_expand_checkpoint_path_is_the_shell_reading_of_tilde_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chosen reading of a configured path, pinned form by form (#1123).

    `~` and `~user` expand, because `DEFAULT_CHECKPOINTS` are written home-relative and have
    always been expanded — a user-typed path must not obey a narrower rule than the default
    it overrides. `$VAR` does not, because expanding it means deciding what an *unset*
    variable means, and substituting an empty string turns a typo into a silently different
    path (P6). A relative path and a non-leading `~` are untouched.

    Killed by: src/uclone_x/tools/builtin/image.py :: return os.path.expanduser(path)
    Becomes: return path
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    assert expand_checkpoint_path("~/models/a.safetensors") == str(
        tmp_path / "models/a.safetensors"
    )
    assert expand_checkpoint_path("~") == str(tmp_path)
    # Unresolvable `~user` stays literal rather than becoming some other user's home.
    assert expand_checkpoint_path("~nosuchuser/a.safetensors") == "~nosuchuser/a.safetensors"
    assert expand_checkpoint_path("$HOME/a.safetensors") == "$HOME/a.safetensors"
    assert expand_checkpoint_path("models/a.safetensors") == "models/a.safetensors"
    # A `~` that is a legitimate character in a filename is not a home reference.
    assert expand_checkpoint_path("/m/back~up.safetensors") == "/m/back~up.safetensors"


@pytest.mark.parametrize(
    "configured",
    ["$HOME/ai_models/x.safetensors", "~nosuchuser/x.safetensors", "relative/x.safetensors"],
)
def test_unexpanded_forms_resolve_as_missing_under_the_name_that_was_typed(
    configured: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A form this engine does not expand fails loudly as itself, not as something else.

    The point of #1123 is one answer per path, not that every path works. These three are
    deliberately not expanded, so the contract is that they are reported `missing` under the
    exact string the user set — checkable against their own config — and that the listing
    agrees by showing nothing.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(IMAGE_CHECKPOINT_ENV, configured)

    resolution = LocalDiffusersImageEngine().checkpoint_resolution()

    assert resolution.state == "missing"
    assert resolution.path == configured
    assert resolution.literal is None
    assert f"'{configured}'" in resolution.describe()
    assert "expanded to" not in resolution.describe()


def test_resolve_checkpoint_falls_back_to_the_default_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing configured, the first default path that exists on disk is chosen."""
    monkeypatch.delenv(IMAGE_CHECKPOINT_ENV, raising=False)
    second = tmp_path / Path(DEFAULT_CHECKPOINTS[1]).name
    second.write_bytes(b"x")
    real_exists = os.path.exists

    def fake_exists(path: str) -> bool:
        if path == os.path.expanduser(DEFAULT_CHECKPOINTS[0]):
            return False
        if path == os.path.expanduser(DEFAULT_CHECKPOINTS[1]):
            return True
        return real_exists(path)

    with patch("os.path.exists", side_effect=fake_exists):
        assert LocalDiffusersImageEngine().resolve_checkpoint() == os.path.expanduser(
            DEFAULT_CHECKPOINTS[1]
        )


def test_resolve_checkpoint_returns_none_when_no_candidate_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(IMAGE_CHECKPOINT_ENV, raising=False)

    with patch("os.path.exists", return_value=False):
        assert LocalDiffusersImageEngine().resolve_checkpoint() is None


@pytest.mark.asyncio
async def test_local_engine_is_unavailable_without_a_checkpoint_even_with_diffusers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The package importing is not enough; a load with no file would fail (P6).

    Killed by: src/uclone_x/tools/builtin/image.py :: return self.resolve_checkpoint() is not None
    Becomes: return True
    """
    monkeypatch.delenv(IMAGE_CHECKPOINT_ENV, raising=False)
    engine = LocalDiffusersImageEngine()

    with (
        patch("importlib.util.find_spec", return_value=MagicMock()),
        patch("os.path.exists", return_value=False),
    ):
        assert await engine.is_available() is False


@pytest.mark.asyncio
async def test_local_engine_is_unavailable_without_diffusers(tmp_path: Path) -> None:
    checkpoint = tmp_path / "c.safetensors"
    checkpoint.write_bytes(b"x")
    engine = LocalDiffusersImageEngine(checkpoint_path=str(checkpoint))

    with patch("importlib.util.find_spec", return_value=None):
        assert await engine.is_available() is False


def test_local_engine_load_names_the_checkpoint_it_could_not_find(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(IMAGE_CHECKPOINT_ENV, raising=False)
    engine = LocalDiffusersImageEngine()

    with patch("os.path.exists", return_value=False):
        with pytest.raises(ImageGenerationError) as exc_info:
            engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    assert IMAGE_CHECKPOINT_ENV in str(exc_info.value)


def test_comfy_engine_reads_its_address_and_checkpoint_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UCX_COMFYUI_URL", "http://10.0.0.5:8188")
    monkeypatch.setenv("UCX_COMFYUI_CHECKPOINT", "other.safetensors")

    engine = ComfyUIImageEngine()

    assert engine.base_url == "http://10.0.0.5:8188"
    assert engine._checkpoint == "other.safetensors"  # pyright: ignore[reportPrivateUsage]


def test_comfy_engine_defaults_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UCX_COMFYUI_URL", raising=False)
    monkeypatch.delenv("UCX_COMFYUI_CHECKPOINT", raising=False)

    engine = ComfyUIImageEngine()

    assert engine.base_url == "http://127.0.0.1:8188"
    assert engine._checkpoint == COMFY_DEFAULT_CHECKPOINT  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_comfy_engine_availability_is_a_live_probe_not_a_configured_address() -> None:
    """A configured URL says nothing; only a daemon that answers makes the engine available.

    Killed by: src/uclone_x/tools/builtin/image.py :: return await client.alive()
    Becomes: return True
    """
    engine = ComfyUIImageEngine(base_url="http://127.0.0.1:8188")
    client = AsyncMock()
    client.alive.return_value = False

    with patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client):
        assert await engine.is_available() is False
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_comfy_engine_availability_is_false_when_the_probe_raises() -> None:
    """A refused connection is an unavailable engine, not an exception out of the dispatcher."""
    engine = ComfyUIImageEngine(base_url="http://127.0.0.1:8188")
    client = AsyncMock()
    client.alive.side_effect = OSError("connection refused")

    with patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client):
        assert await engine.is_available() is False
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_comfy_engine_generate_queues_a_graph_and_downloads_the_result() -> None:
    engine = ComfyUIImageEngine(base_url="http://127.0.0.1:8188", checkpoint="ckpt.safetensors")
    client = AsyncMock()
    client.queue_prompt.return_value = "prompt-1"
    client.wait_for_output.return_value = ["ComfyUI_0001_.png"]
    client.download_image.return_value = b"comfy_png_bytes"

    with patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client):
        res = await engine.generate(
            prompt="a castle",
            negative_prompt="blurry",
            width=1024,
            height=1024,
            seed=77,
            style="anime",
        )

    assert res.image_bytes == b"comfy_png_bytes"
    assert res.engine_name == "comfyui-local"
    assert res.seed == 77
    client.download_image.assert_awaited_once_with("ComfyUI_0001_.png")
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_comfy_engine_generate_refuses_to_return_an_empty_image() -> None:
    """A finished job with no output file is an error, not an empty success (P6).

    Killed by: src/uclone_x/tools/builtin/image.py :: if not filenames:
    Becomes: if False:
    """
    engine = ComfyUIImageEngine(base_url="http://127.0.0.1:8188")
    client = AsyncMock()
    client.queue_prompt.return_value = "prompt-1"
    client.wait_for_output.return_value = []

    with patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client):
        with pytest.raises(ImageGenerationError) as exc_info:
            await engine.generate(
                prompt="a castle",
                negative_prompt="",
                width=1024,
                height=1024,
                seed=1,
                style="anime",
            )

    assert "without producing an image" in str(exc_info.value)
    client.download_image.assert_not_awaited()


def _fake_environment(monkeypatch: pytest.MonkeyPatch, installed: dict[str, str | None]) -> None:
    """Present `installed` as the only packages on the machine, mapping name -> version.

    A value of None means importable with no distribution metadata, which is the case the
    version check must not mistake for an outdated install.
    """
    import importlib.metadata
    import importlib.util

    def fake_find_spec(name: str, package: str | None = None) -> object | None:
        return MagicMock() if name in installed else None

    def fake_version(name: str) -> str:
        version = installed.get(name)
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib.metadata, "version", fake_version)


_ALL_PRESENT = {"diffusers": "0.40.0", "torch": "2.6.0", "transformers": "4.51.0"}


def test_in_process_requirements_name_torch_and_transformers() -> None:
    """`diffusers` alone was the old requirement, and it declares neither of the other two.

    Measured on 2026-09-17: installing `uclone-x[cli,media]` into a clean Python 3.13.14
    venv brought `diffusers` and not `torch` or `transformers`, after which
    `ucx media status` printed a ready engine that could not have generated anything.
    """
    assert {module for module, _, _ in IN_PROCESS_REQUIREMENTS} == {
        "diffusers",
        "torch",
        "transformers",
    }


def test_dependency_problems_are_empty_when_everything_is_new_enough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_environment(monkeypatch, dict(_ALL_PRESENT))

    assert in_process_dependency_problems() == ()


def test_dependency_problems_name_each_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A beginner environment may have any subset; each absence is reported by name.

    Killed by: src/uclone_x/tools/builtin/image.py :: problems.append(DependencyProblem(module, requirement, None))
    Becomes: pass
    """
    _fake_environment(monkeypatch, {"diffusers": "0.40.0"})

    described = [problem.describe() for problem in in_process_dependency_problems()]

    assert [problem.module for problem in in_process_dependency_problems()] == [
        "torch",
        "transformers",
    ]
    assert any("'torch' is not installed" in line for line in described)
    assert any("torch>=2.2.0" in line for line in described)


def test_dependency_problems_report_an_installed_but_outdated_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An old version resolves to an import that succeeds and an API that is not there.

    Killed by: src/uclone_x/tools/builtin/image.py :: if version_release(installed) < floor:
    Becomes: if False:
    """
    _fake_environment(monkeypatch, {**_ALL_PRESENT, "diffusers": "0.21.4"})

    problems = in_process_dependency_problems()

    assert [problem.module for problem in problems] == ["diffusers"]
    assert problems[0].installed == "0.21.4"
    assert "older than" in problems[0].describe()
    assert "0.21.4" in problems[0].describe()


def test_dependency_problems_accept_a_package_with_no_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An importable package whose version cannot be read is not evidence of an old one."""
    _fake_environment(monkeypatch, {**_ALL_PRESENT, "torch": None})

    assert in_process_dependency_problems() == ()


def test_release_reads_the_numeric_prefix_of_a_local_version() -> None:
    """Torch ships versions like '2.6.0+cpu', which must not compare as older than 2.2."""
    assert version_release("2.6.0+cpu") == (2, 6, 0)
    assert version_release("2.6.0+cpu") >= (2, 2)
    assert version_release("4.51.0.dev0") == (4, 51, 0)


@pytest.mark.asyncio
async def test_local_engine_is_unavailable_when_torch_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The P6 regression: a checkpoint plus `diffusers` alone must not report ready.

    Killed by: src/uclone_x/tools/builtin/image.py :: if in_process_dependency_problems():
    Becomes: if False:
    """
    checkpoint = tmp_path / "c.safetensors"
    checkpoint.write_bytes(b"x")
    engine = LocalDiffusersImageEngine(checkpoint_path=str(checkpoint))
    _fake_environment(monkeypatch, {"diffusers": "0.40.0"})

    assert await engine.is_available() is False


def test_local_engine_load_names_every_unmet_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure says which packages, not a bare ImportError from inside diffusers."""
    checkpoint = tmp_path / "c.safetensors"
    checkpoint.write_bytes(b"x")
    engine = LocalDiffusersImageEngine(checkpoint_path=str(checkpoint))
    _fake_environment(monkeypatch, {"diffusers": "0.40.0"})

    with pytest.raises(ImageGenerationError) as exc_info:
        engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    message = str(exc_info.value)
    assert "'torch' is not installed" in message
    assert "'transformers' is not installed" in message


def test_install_hint_names_every_requirement_with_its_floor() -> None:
    """A hint that installs only `diffusers` reproduces the bug it is printed for."""
    hint = diffusers_install_hint()

    for _, requirement, _ in IN_PROCESS_REQUIREMENTS:
        assert requirement in hint


def test_a_missing_accelerate_leaves_a_working_engine_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An MPS or CPU engine that generated before an upgrade must still report ready.

    `accelerate` only lets a CUDA card short of memory offload; the engine runs without it.
    """
    _fake_environment(monkeypatch, {**_ALL_PRESENT})

    assert in_process_dependency_problems() == ()
    assert "accelerate" not in {module for module, _, _ in IN_PROCESS_REQUIREMENTS}


def test_the_install_list_adds_accelerate_to_the_requirements() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: ("accelerate", "accelerate>=0.31.0", (0, 31)),
    Becomes: ("diffusers", "diffusers>=0.31.0", (0, 31)),
    """
    requirements = [requirement for _, requirement, _ in IN_PROCESS_INSTALL_REQUIREMENTS]

    assert requirements == [
        *(requirement for _, requirement, _ in IN_PROCESS_REQUIREMENTS),
        "accelerate>=0.31.0",
    ]


def test_the_install_hint_installs_accelerate_too() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: f"'{requirement}'" for _, requirement, _ in IN_PROCESS_INSTALL_REQUIREMENTS
    Becomes: f"'{requirement}'" for _, requirement, _ in IN_PROCESS_REQUIREMENTS
    """
    assert "'accelerate>=0.31.0'" in diffusers_install_hint()


def test_accelerate_is_installed_reads_this_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: return importlib.util.find_spec("accelerate") is not None
    Becomes: return importlib.util.find_spec("accelerate") is None
    """
    _fake_environment(monkeypatch, {"accelerate": "1.15.0"})
    assert accelerate_is_installed()

    _fake_environment(monkeypatch, {})
    assert not accelerate_is_installed()


@pytest.mark.asyncio
async def test_diagnostics_lists_the_missing_in_process_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Nothing is available" must still say which package to install (P6)."""
    monkeypatch.delenv("UCX_IMAGE_REMOTE_URL", raising=False)
    _fake_environment(monkeypatch, {"diffusers": "0.40.0"})
    dispatcher = ImagePipelineDispatcher()

    diagnostics = dispatcher.diagnostics()

    assert "'torch' is not installed" in diagnostics


@pytest.mark.asyncio
async def test_generate_image_tool_unique_short_id_and_sidecar_metadata(tmp_path: Path) -> None:
    """Images are given compact 6-char hex IDs and companion JSON sidecar metadata."""
    import json
    import re

    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    mock_dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"dummy_png",
        seed=123456791,
        engine_name="mock-engine",
        device_info="test-device",
        duration_seconds=0.25,
        width=1024,
        height=768,
    )

    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="artist",
        workspace_root=tmp_path,
        session_id="sess_room__room_test__artist",
    )

    result = await tool.execute(
        params={"prompt": "bikini girl crawling on sandy beach", "seed_override": 123456791},
        context=context,
    )

    assert result.success is True
    assert isinstance(result.output, dict)
    rel_path = _picture_rel(result.output["relative_url"])
    rel_meta = sidecar_path_for(rel_path)
    assert "meta_path" not in result.output
    assert result.artifacts == (rel_path,)

    # Compact short ID pattern, in the session's own directory (#1390)
    assert re.match(
        r"^artifacts/sess_room__room_test__artist/images/img_[0-9a-f]{6}\.png$", rel_path
    )
    assert re.match(
        r"^artifacts/sess_room__room_test__artist/images/img_[0-9a-f]{6}\.json$", rel_meta
    )

    # Verify files on disk
    png_file = tmp_path / rel_path
    json_file = tmp_path / rel_meta
    assert png_file.exists()
    assert json_file.exists()

    # Verify JSON content
    raw_json = json.loads(json_file.read_text(encoding="utf-8"))
    assert isinstance(raw_json, dict)
    meta = cast(dict[str, Any], raw_json)
    assert meta["prompt"] == "bikini girl crawling on sandy beach"
    assert meta["seed"] == 123456791
    assert meta["engine"] == "mock-engine"
    assert meta["width"] == 1024
    assert meta["height"] == 768
    assert "created_at" in meta


@pytest.mark.asyncio
async def test_generate_image_tool_no_collision_on_identical_seed(tmp_path: Path) -> None:
    """Repeated calls with identical seed produce distinct files without collision."""
    mock_dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    mock_dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"dummy_png",
        seed=123456791,
        engine_name="mock-engine",
        device_info="test-device",
        duration_seconds=0.2,
        width=768,
        height=576,
    )

    tool = GenerateImageTool(dispatcher=mock_dispatcher)
    context = ToolContext(
        agent_id="artist",
        workspace_root=tmp_path,
        session_id="sess_room__room_test__artist",
    )

    # Generation 1 (e.g. Gundam Girl with seed 123456791)
    res1 = await tool.execute(
        params={"prompt": "gundam girl with mechanical armor", "seed_override": 123456791},
        context=context,
    )
    # Generation 2 (e.g. Bikini Girl with same seed 123456791)
    res2 = await tool.execute(
        params={"prompt": "bikini girl crawling on sand", "seed_override": 123456791},
        context=context,
    )

    assert res1.success is True and res2.success is True
    assert isinstance(res1.output, dict) and isinstance(res2.output, dict)

    path1 = _picture_rel(res1.output["relative_url"])
    path2 = _picture_rel(res2.output["relative_url"])
    assert path1 != path2

    # Both images must exist simultaneously on disk (no overwriting!)
    assert (tmp_path / path1).exists()
    assert (tmp_path / path2).exists()


@pytest.mark.asyncio
async def test_local_diffusers_image_engine_generate_offloads_to_worker_thread(
    tmp_path: Path,
) -> None:
    """In-process diffusers execution must run off the asyncio event loop thread.

    Running heavy diffusion directly in the event loop blocks the entire server
    from responding to HTTP requests or heartbeat signals.

    Killed by: src/uclone_x/tools/builtin/image.py :: await asyncio.to_thread(
    Becomes: self._run_in_process_generation(
    """
    import threading

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"dummy")
    engine = LocalDiffusersImageEngine(checkpoint_path=str(checkpoint))

    main_thread_id = threading.get_ident()
    worker_thread_ids: list[int] = []

    def mock_run(
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        style: str,
        steps: int | None = None,
        cfg: float | None = None,
        family: Any = None,
    ) -> bytes:
        worker_thread_ids.append(threading.get_ident())
        return b"fake_png_bytes"

    with patch.object(engine, "_run_in_process_generation", side_effect=mock_run):
        result = await engine.generate(
            prompt="a serene sunset",
            negative_prompt="blurry",
            width=512,
            height=512,
            seed=42,
            style="photorealistic",
        )

    assert result.image_bytes == b"fake_png_bytes"
    assert result.seed == 42
    assert result.engine_name == "diffusers-sdxl"
    assert len(worker_thread_ids) == 1
    # Must have executed on a worker thread distinct from the event loop thread
    assert worker_thread_ids[0] != main_thread_id


@pytest.mark.asyncio
async def test_local_diffusers_image_engine_run_in_process_generation(tmp_path: Path) -> None:
    """_run_in_process_generation calls pipeline and returns valid PNG bytes."""
    from PIL import Image

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"dummy")
    engine = LocalDiffusersImageEngine(checkpoint_path=str(checkpoint))

    # Create dummy PIL image to return from pipeline
    pil_img = Image.new("RGB", (64, 64), color="blue")
    mock_pipeline = _sdxl_pipeline()
    mock_pipeline.return_value = MagicMock(images=[pil_img])

    with patch.object(engine, "_ensure_pipeline_loaded", return_value=mock_pipeline):
        png_bytes = engine._run_in_process_generation(  # pyright: ignore[reportPrivateUsage]
            prompt="test prompt",
            negative_prompt="",
            width=64,
            height=64,
            seed=123,
            style="anime",
        )

    assert png_bytes.startswith(b"\x89PNG")


def test_two_generations_never_run_the_pipeline_at_once(tmp_path: Path) -> None:
    """A second request waits for the first: a pipeline shared by two threads aborts on MPS.

    The first run holds the pipeline until the second thread has started (or 0.3 s pass).
    Unserialized, the second enters the pipeline in that window and two runs overlap.
    """
    import threading

    from PIL import Image

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"dummy")
    engine = LocalDiffusersImageEngine(checkpoint_path=str(checkpoint))

    first_inside = threading.Event()
    second_started = threading.Event()
    guard = threading.Lock()
    active = 0
    most_active = 0

    def run_pipeline(**_: Any) -> MagicMock:
        nonlocal active, most_active
        with guard:
            active += 1
            most_active = max(most_active, active)
        first_inside.set()
        second_started.wait(timeout=0.3)
        with guard:
            active -= 1
        return MagicMock(images=[Image.new("RGB", (8, 8))])

    mock_pipeline = _sdxl_pipeline()
    mock_pipeline.side_effect = run_pipeline

    def generate() -> None:
        engine._run_in_process_generation(  # pyright: ignore[reportPrivateUsage]
            prompt="p", negative_prompt="", width=8, height=8, seed=1, style="anime"
        )

    with patch.object(engine, "_ensure_pipeline_loaded", return_value=mock_pipeline):
        first = threading.Thread(target=generate)
        first.start()
        assert first_inside.wait(timeout=5)
        second = threading.Thread(target=generate)
        second.start()
        second_started.set()
        first.join(timeout=5)
        second.join(timeout=5)

    assert mock_pipeline.call_count == 2
    assert most_active == 1


class _FakeOutOfMemoryError(RuntimeError):
    """Stands in for `torch.OutOfMemoryError`, with torch's own allocator wording."""


_RAW_OOM_TEXT = (
    "CUDA out of memory. Tried to allocate 1.50 GiB. GPU 0 has a total capacity of "
    "15.99 GiB of which 812.00 MiB is free. See /opt/torch/docs PYTORCH_CUDA_ALLOC_CONF"
)


class _SdxlVaeSpec:
    """The `AutoencoderKL` methods the engine calls, named as in diffusers 0.40."""

    def enable_tiling(self) -> None: ...


class _SdxlPipelineSpec:
    """The `StableDiffusionXLPipeline` surface the engine calls, named as in diffusers 0.40.

    A mock specced from this raises `AttributeError` for any other name, which is how a
    call to the nonexistent `enable_vae_tiling` would have failed here instead of on
    every offloaded load. `test_the_pipeline_spec_names_only_what_diffusers_has` checks
    each name against the installed diffusers.
    """

    tokenizer: Any = None
    tokenizer_2: Any = None
    text_encoder: Any = None
    text_encoder_2: Any = None

    def to(self, device: str) -> _SdxlPipelineSpec: ...

    def enable_model_cpu_offload(self) -> None: ...

    def __call__(self, **kwargs: Any) -> Any: ...


@pytest.fixture(autouse=True)
def no_real_gpu_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here reads this machine's kernel, NVML or `nvidia-smi` (P8).

    The WSL detection reads `/proc` and `/dev/dxg`, and on WSL the free-memory check runs
    `nvidia-smi`. Pinned to native with no whole-device source; a test that needs WSL or a
    reading sets its own.
    """
    import uclone_x.tools.builtin.image as image_module

    def no_reading() -> None:
        raise AssertionError("a unit test reached a real GPU memory source")

    monkeypatch.setattr(image_module, "running_under_wsl", lambda: False)
    monkeypatch.setattr(image_module, "nvml_free_bytes", no_reading)
    monkeypatch.setattr(image_module, "nvidia_smi_output", no_reading)


def _sdxl_pipeline() -> MagicMock:
    """A specced SDXL pipeline whose `.to` returns itself, as diffusers' does."""
    pipeline = MagicMock(spec=_SdxlPipelineSpec)
    pipeline.vae = MagicMock(spec=_SdxlVaeSpec)
    pipeline.to.return_value = pipeline
    return pipeline


def test_the_pipeline_spec_names_only_what_diffusers_has() -> None:
    """The specs above are only as good as their agreement with the real classes."""
    diffusers = pytest.importorskip("diffusers")
    import inspect

    sdxl = diffusers.StableDiffusionXLPipeline
    for name in ("to", "enable_model_cpu_offload", "__call__"):
        assert callable(getattr(sdxl, name)), name
    assert callable(sdxl.from_single_file)
    parameters = inspect.signature(sdxl.__init__).parameters
    for component in ("tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2"):
        assert component in parameters, component
    vae = parameters["vae"]
    assert vae.annotation is diffusers.AutoencoderKL
    assert callable(diffusers.AutoencoderKL.enable_tiling)
    assert not hasattr(sdxl, "enable_vae_tiling")


class _FakeTorch:
    """Just enough of `torch` for device selection and placement.

    Two dtype sentinels, two availability probes, the free-memory reading, and the
    out-of-memory class. The default free memory is ample, so a test that does not name
    it gets the plain `.to("cuda")` placement.
    """

    float16 = "float16"
    float32 = "float32"
    OutOfMemoryError = _FakeOutOfMemoryError

    def __init__(
        self,
        *,
        cuda: bool,
        mps: bool,
        gpu_name: str = "NVIDIA GeForce RTX 5070 Ti",
        free_bytes: int = 15 * 1024**3,
    ):
        self.cuda = MagicMock()
        self.cuda.is_available.return_value = cuda
        self.cuda.get_device_name.return_value = gpu_name
        self.cuda.mem_get_info.return_value = (free_bytes, 16 * 1024**3)
        self.cuda.OutOfMemoryError = _FakeOutOfMemoryError
        self.Generator = MagicMock()
        self.backends = MagicMock()
        self.backends.mps.is_available.return_value = mps


def test_select_torch_device_prefers_cuda_in_half_precision() -> None:
    """The fresh-machine E2E defect: an RTX 5070 Ti ran SDXL on its CPU in float32.

    Killed by: src/uclone_x/tools/builtin/image.py :: if cuda is not None and cuda.is_available():
    Becomes: if False:
    """
    assert select_torch_device(_FakeTorch(cuda=True, mps=False)) == ("cuda", "float16")


def test_select_torch_device_prefers_cuda_over_mps() -> None:
    assert select_torch_device(_FakeTorch(cuda=True, mps=True)) == ("cuda", "float16")


def test_select_torch_device_keeps_mps_in_half_precision() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: return "mps", torch.float16
    Becomes: return "mps", torch.float32
    """
    assert select_torch_device(_FakeTorch(cuda=False, mps=True)) == ("mps", "float16")


def test_select_torch_device_falls_back_to_cpu_in_full_precision() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: return "cpu", torch.float32
    Becomes: return "cpu", torch.float16
    """
    assert select_torch_device(_FakeTorch(cuda=False, mps=False)) == ("cpu", "float32")


@pytest.mark.parametrize(
    ("cuda", "mps", "device", "dtype"),
    [
        (True, False, "cuda", "float16"),
        (False, True, "mps", "float16"),
        (False, False, "cpu", "float32"),
    ],
)
def test_pipeline_loads_onto_the_selected_device_and_reports_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda: bool,
    mps: bool,
    device: str,
    dtype: str,
) -> None:
    """The load uses the chosen device and dtype, and `device_info` names what ran.

    Killed by: src/uclone_x/tools/builtin/image.py :: return pipeline.to(device), False
    Becomes: return pipeline, False
    """
    import sys

    import uclone_x.tools.builtin.image as image_module

    checkpoint = tmp_path / "c.safetensors"
    checkpoint.write_bytes(b"x")
    fake_torch = _FakeTorch(cuda=cuda, mps=mps)
    pipeline = _sdxl_pipeline()
    fake_diffusers = MagicMock()
    fake_diffusers.StableDiffusionXLPipeline.from_single_file.return_value = pipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(image_module, "in_process_dependency_problems", lambda: ())
    engine = LocalDiffusersImageEngine(checkpoint_path=str(checkpoint))

    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    fake_diffusers.StableDiffusionXLPipeline.from_single_file.assert_called_once_with(
        str(checkpoint), torch_dtype=dtype
    )
    pipeline.to.assert_called_once_with(device)
    pipeline.enable_model_cpu_offload.assert_not_called()
    device_info = engine._device  # pyright: ignore[reportPrivateUsage]
    if device == "cuda":
        assert device_info == "cuda (NVIDIA GeForce RTX 5070 Ti)"
    else:
        assert device_info.startswith(f"{device} (")


def _engine_with_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: _FakeTorch,
    pipeline: MagicMock | None,
    make_pipeline: Callable[[], MagicMock] | None = None,
) -> LocalDiffusersImageEngine:
    """An in-process engine whose `torch` and `diffusers` imports are the given fakes.

    `make_pipeline`, when given, builds the pipeline at load time, so the fake diffusers
    holds no reference to it (a `return_value` would keep it alive for the whole test).
    """
    import sys

    import uclone_x.tools.builtin.image as image_module

    checkpoint = tmp_path / "c.safetensors"
    checkpoint.write_bytes(b"x")
    fake_diffusers = MagicMock()
    from_single_file = fake_diffusers.StableDiffusionXLPipeline.from_single_file
    if make_pipeline is not None:

        def build(*_args: object, **_kwargs: object) -> MagicMock:
            return make_pipeline()

        from_single_file.side_effect = build
    else:
        from_single_file.return_value = pipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(image_module, "in_process_dependency_problems", lambda: ())
    # Pinned: whether this machine has `accelerate` must not change the message asserted.
    monkeypatch.setattr(image_module, "accelerate_is_installed", lambda: True)
    return LocalDiffusersImageEngine(checkpoint_path=str(checkpoint))


def _assert_plain_out_of_memory(message: str, tmp_path: Path) -> None:
    assert message == GPU_OUT_OF_MEMORY_MESSAGE
    for internal in (
        "CUDA out of memory",
        "GiB",
        "PYTORCH",
        "OutOfMemoryError",
        "/",
        str(tmp_path),
    ):
        assert internal not in message


def test_a_card_short_of_memory_offloads_sdxl_to_the_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 16 GB card with qwen3:8b resident has ~8 GB free; `.to("cuda")` ran out there.

    Killed by: src/uclone_x/tools/builtin/image.py :: if device == "cuda" and not cuda_has_room_for_sdxl(torch):
    Becomes: if False:
    """
    pipeline = _sdxl_pipeline()
    fake_torch = _FakeTorch(cuda=True, mps=False, free_bytes=8 * 1024**3)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    assert engine._ensure_pipeline_loaded() is pipeline  # pyright: ignore[reportPrivateUsage]

    pipeline.enable_model_cpu_offload.assert_called_once_with()
    pipeline.vae.enable_tiling.assert_called_once_with()
    pipeline.to.assert_not_called()
    device_info = engine._device  # pyright: ignore[reportPrivateUsage]
    assert device_info == "cuda (NVIDIA GeForce RTX 5070 Ti), offloading to CPU"


def test_offload_decodes_the_image_in_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: pipeline.vae.enable_tiling()
    Becomes: pass
    """
    pipeline = _sdxl_pipeline()
    fake_torch = _FakeTorch(cuda=True, mps=False, free_bytes=4 * 1024**3)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    pipeline.vae.enable_tiling.assert_called_once_with()


def test_an_unreadable_free_memory_figure_offloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: free_bytes = 0
    Becomes: free_bytes = CUDA_RESIDENT_MIN_FREE_BYTES
    """
    pipeline = _sdxl_pipeline()
    fake_torch = _FakeTorch(cuda=True, mps=False)
    fake_torch.cuda.mem_get_info.side_effect = RuntimeError("no context")
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    pipeline.enable_model_cpu_offload.assert_called_once_with()
    pipeline.to.assert_not_called()


_GIB = 1024**3
#: What torch reported on the WSL2 host: 14.66 GiB free with qwen3:8b resident.
_WSL_TORCH_FREE = int(14.66 * _GIB)


def _on_wsl(
    monkeypatch: pytest.MonkeyPatch,
    *,
    nvml: int | None = None,
    smi: str | None = None,
) -> None:
    import uclone_x.tools.builtin.image as image_module

    monkeypatch.setattr(image_module, "running_under_wsl", lambda: True)
    monkeypatch.setattr(image_module, "nvml_free_bytes", lambda: nvml)
    monkeypatch.setattr(image_module, "nvidia_smi_output", lambda: smi)


def test_on_wsl_the_device_wide_reading_overrides_torch_and_offloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The WSL2 E2E: torch said 14.66 GiB free, `nvidia-smi` 8115 of 16303 MiB used.

    Killed by: src/uclone_x/tools/builtin/image.py :: free_bytes = min(int(free_bytes), device_free) if device_free is not None else 0
    Becomes: free_bytes = int(free_bytes) if device_free is not None else 0
    """
    _on_wsl(monkeypatch, smi="8115, 16303\n")
    pipeline = _sdxl_pipeline()
    fake_torch = _FakeTorch(cuda=True, mps=False, free_bytes=_WSL_TORCH_FREE)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    pipeline.enable_model_cpu_offload.assert_called_once_with()
    pipeline.to.assert_not_called()


def test_on_wsl_the_wsl_check_is_what_brings_in_the_device_wide_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: if running_under_wsl():
    Becomes: if False:
    """
    _on_wsl(monkeypatch, nvml=8 * _GIB)
    pipeline = _sdxl_pipeline()
    fake_torch = _FakeTorch(cuda=True, mps=False, free_bytes=_WSL_TORCH_FREE)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    pipeline.enable_model_cpu_offload.assert_called_once_with()


def test_on_wsl_with_no_device_wide_source_the_pipeline_offloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither NVML nor `nvidia-smi`: free memory is unknown, and unknown is not ample.

    Killed by: src/uclone_x/tools/builtin/image.py :: free_bytes = min(int(free_bytes), device_free) if device_free is not None else 0
    Becomes: free_bytes = min(int(free_bytes), device_free) if device_free is not None else free_bytes
    """
    _on_wsl(monkeypatch)
    pipeline = _sdxl_pipeline()
    fake_torch = _FakeTorch(cuda=True, mps=False, free_bytes=_WSL_TORCH_FREE)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    pipeline.enable_model_cpu_offload.assert_called_once_with()
    pipeline.to.assert_not_called()


def test_on_wsl_an_unparseable_nvidia_smi_reading_offloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: free_bytes = min(int(free_bytes), device_free) if device_free is not None else 0
    Becomes: free_bytes = min(int(free_bytes), device_free) if device_free is not None else free_bytes
    """
    _on_wsl(monkeypatch, smi="Failed to initialize NVML: Unknown Error\n")
    pipeline = _sdxl_pipeline()
    fake_torch = _FakeTorch(cuda=True, mps=False, free_bytes=_WSL_TORCH_FREE)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    pipeline.enable_model_cpu_offload.assert_called_once_with()
    pipeline.to.assert_not_called()


def test_on_wsl_nvml_is_read_before_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: if free_bytes is not None:
    Becomes: if False:
    """
    _on_wsl(monkeypatch, nvml=3 * _GIB, smi="0, 16303\n")

    assert device_wide_free_bytes() == 3 * _GIB


def test_native_linux_with_ample_free_memory_loads_whole_without_asking_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native: torch's figure alone; the autouse fixture fails any whole-device read."""
    pipeline = _sdxl_pipeline()
    fake_torch = _FakeTorch(cuda=True, mps=False, free_bytes=15 * _GIB)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    pipeline.to.assert_called_once_with("cuda")
    pipeline.enable_model_cpu_offload.assert_not_called()


def test_on_wsl_with_room_on_the_device_the_pipeline_loads_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The WSL branch does not offload a card that really is empty."""
    _on_wsl(monkeypatch, smi="500, 16303\n")
    pipeline = _sdxl_pipeline()
    fake_torch = _FakeTorch(cuda=True, mps=False, free_bytes=15 * _GIB)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    pipeline.to.assert_called_once_with("cuda")


def test_nvidia_smi_output_is_read_in_mib_per_card() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: free.append(max(total_mib - used_mib, 0) * 1024**2)
    Becomes: free.append(max(total_mib - used_mib, 0) * 1024**3)
    """
    assert parse_nvidia_smi_free_bytes("8115, 16303\n") == (16303 - 8115) * 1024**2


def test_nvidia_smi_output_with_several_cards_takes_the_tightest() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: return min(free, default=None)
    Becomes: return max(free, default=None)
    """
    assert parse_nvidia_smi_free_bytes("8115, 16303\n100, 16303\n") == 8188 * 1024**2


@pytest.mark.parametrize(
    "output",
    ["", "Failed to initialize NVML: Unknown Error\n", "8115, 16303\n[N/A], [N/A]\n"],
)
def test_nvidia_smi_output_that_does_not_parse_is_unknown(output: str) -> None:
    """One unreadable line voids the reading, even beside a readable one.

    Killed by: src/uclone_x/tools/builtin/image.py :: return None  # an unreadable line makes the whole reading unknown
    Becomes: continue  # an unreadable line makes the whole reading unknown
    """
    assert parse_nvidia_smi_free_bytes(output) is None


def test_nvidia_smi_is_run_with_a_timeout_and_a_failed_run_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No real process: `subprocess.run` is replaced and its call is recorded.

    Killed by: src/uclone_x/tools/builtin/image.py :: if completed.returncode != 0:
    Becomes: if False:
    """
    import subprocess

    import uclone_x.tools.builtin.image as image_module

    calls: list[dict[str, Any]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 9, stdout="8115, 16303\n", stderr="")

    monkeypatch.setattr(image_module.subprocess, "run", fake_run)

    def not_on_path(_name: str) -> None:
        return None

    monkeypatch.setattr(image_module.shutil, "which", not_on_path)

    assert real_nvidia_smi_output() is None
    assert calls[0]["argv"][0] == "/usr/lib/wsl/lib/nvidia-smi"
    assert calls[0]["timeout"] == 5.0


def test_a_hung_nvidia_smi_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: except (OSError, subprocess.SubprocessError):
    Becomes: except (OSError, ArithmeticError):
    """
    import subprocess

    import uclone_x.tools.builtin.image as image_module

    def hung(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(image_module.subprocess, "run", hung)

    def on_path(_name: str) -> str:
        return "/usr/bin/nvidia-smi"

    monkeypatch.setattr(image_module.shutil, "which", on_path)

    assert real_nvidia_smi_output() is None


def test_nvml_reports_total_minus_used_of_the_tightest_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fake `pynvml` module; nothing here loads the real library.

    Killed by: src/uclone_x/tools/builtin/image.py :: free.append(max(int(memory.total) - int(memory.used), 0))
    Becomes: free.append(max(int(memory.total) - 0, 0))
    """
    import sys
    from types import SimpleNamespace

    memories = [
        SimpleNamespace(total=16303 * 1024**2, used=8115 * 1024**2),
        SimpleNamespace(total=16303 * 1024**2, used=100 * 1024**2),
    ]
    fake = MagicMock()
    fake.nvmlDeviceGetCount.return_value = 2

    def handle_by_index(index: int) -> int:
        return index

    def memory_info(handle: int) -> SimpleNamespace:
        return memories[handle]

    fake.nvmlDeviceGetHandleByIndex.side_effect = handle_by_index
    fake.nvmlDeviceGetMemoryInfo.side_effect = memory_info
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    assert real_nvml_free_bytes() == 8188 * 1024**2
    fake.nvmlShutdown.assert_called_once_with()


def test_without_pynvml_nvml_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "pynvml", None)

    assert real_nvml_free_bytes() is None


def test_wsl_is_recognised_by_its_kernel_release(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: if "microsoft" in text or "wsl" in text:
    Becomes: if False:
    """
    osrelease = tmp_path / "osrelease"
    osrelease.write_text("6.6.87.2-microsoft-standard-WSL2\n")

    assert running_under_wsl(osrelease, tmp_path / "absent", tmp_path / "dxg")


def test_wsl_is_recognised_by_its_gpu_device_alone(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: return dxg.exists()
    Becomes: return False
    """
    (tmp_path / "dxg").touch()

    assert running_under_wsl(tmp_path / "absent", tmp_path / "absent", tmp_path / "dxg")


def test_a_native_linux_kernel_is_not_wsl(tmp_path: Path) -> None:
    osrelease = tmp_path / "osrelease"
    osrelease.write_text("6.8.0-45-generic\n")
    version = tmp_path / "version"
    version.write_text("Linux version 6.8.0-45-generic (buildd@lcy02-amd64-075) (gcc 13.2.0)\n")

    assert not running_under_wsl(osrelease, version, tmp_path / "dxg")


def test_without_accelerate_the_pipeline_is_loaded_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _sdxl_pipeline()
    pipeline.enable_model_cpu_offload.side_effect = ImportError("accelerate")
    fake_torch = _FakeTorch(cuda=True, mps=False, free_bytes=4 * 1024**3)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    pipeline.to.assert_called_once_with("cuda")
    device_info = engine._device  # pyright: ignore[reportPrivateUsage]
    assert device_info == "cuda (NVIDIA GeForce RTX 5070 Ti)"


def test_running_out_of_memory_while_loading_is_reported_in_plain_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: if not isinstance(exc, torch_out_of_memory_types(torch)):
    Becomes: if True:
    """
    pipeline = _sdxl_pipeline()
    pipeline.to.side_effect = _FakeOutOfMemoryError(_RAW_OOM_TEXT)
    fake_torch = _FakeTorch(cuda=True, mps=False)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    with pytest.raises(ImageGenerationError) as caught:
        engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    _assert_plain_out_of_memory(str(caught.value), tmp_path)
    fake_torch.cuda.empty_cache.assert_called_once_with()
    assert engine._pipeline is None  # pyright: ignore[reportPrivateUsage]


def test_running_out_of_memory_while_generating_drops_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next request must rebuild, re-reading free memory, not reuse a broken pipeline.

    Killed by: src/uclone_x/tools/builtin/image.py :: if torch is None or not isinstance(exc, torch_out_of_memory_types(torch)):
    Becomes: if True:
    """
    import uclone_x.tools.builtin.image as image_module

    pipeline = _sdxl_pipeline()
    pipeline.side_effect = _FakeOutOfMemoryError(_RAW_OOM_TEXT)
    fake_torch = _FakeTorch(cuda=True, mps=False)
    collected: list[bool] = []
    monkeypatch.setattr(image_module.gc, "collect", lambda: collected.append(True) or 0)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)

    with pytest.raises(ImageGenerationError) as caught:
        engine._run_in_process_generation(  # pyright: ignore[reportPrivateUsage]
            prompt="a cat",
            negative_prompt="",
            width=768,
            height=768,
            seed=1,
            style="photorealistic",
        )

    _assert_plain_out_of_memory(str(caught.value), tmp_path)
    assert engine._pipeline is None  # pyright: ignore[reportPrivateUsage]
    assert collected == [True]
    fake_torch.cuda.empty_cache.assert_called_once_with()


def test_out_of_memory_release_drops_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: self._pipeline = None
    Becomes: pass
    """
    pipeline = _sdxl_pipeline()
    fake_torch = _FakeTorch(cuda=True, mps=False)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)
    engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    engine._release_after_out_of_memory(fake_torch)  # pyright: ignore[reportPrivateUsage]

    assert engine._pipeline is None  # pyright: ignore[reportPrivateUsage]
    fake_torch.cuda.empty_cache.assert_called_once_with()


def test_the_pipeline_is_collected_before_the_cache_is_emptied_after_a_failed_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Released inside the `except`, the traceback kept the pipeline alive (#1541 review).

    `empty_cache` can only return memory nothing references, so the pipeline must be
    garbage by then.

    Killed by: src/uclone_x/tools/builtin/image.py :: del pipeline  # the half-placed one
    Becomes: pass
    """
    refs: list[weakref.ref[MagicMock]] = []

    def make_pipeline() -> MagicMock:
        pipeline = _sdxl_pipeline()
        pipeline.to.side_effect = _FakeOutOfMemoryError(_RAW_OOM_TEXT)
        refs.append(weakref.ref(pipeline))
        return pipeline

    fake_torch = _FakeTorch(cuda=True, mps=False)
    collected_first: list[bool] = []
    fake_torch.cuda.empty_cache.side_effect = lambda: collected_first.append(refs[0]() is None)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, None, make_pipeline)

    with pytest.raises(ImageGenerationError) as caught:
        engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    assert collected_first == [True]
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_the_pipeline_is_collected_before_the_cache_is_emptied_after_a_failed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: del pipeline  # the one that ran out
    Becomes: pass
    """
    refs: list[weakref.ref[MagicMock]] = []

    def make_pipeline() -> MagicMock:
        pipeline = _sdxl_pipeline()
        pipeline.side_effect = _FakeOutOfMemoryError(_RAW_OOM_TEXT)
        refs.append(weakref.ref(pipeline))
        return pipeline

    fake_torch = _FakeTorch(cuda=True, mps=False)
    collected_first: list[bool] = []
    fake_torch.cuda.empty_cache.side_effect = lambda: collected_first.append(refs[0]() is None)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, None, make_pipeline)

    with pytest.raises(ImageGenerationError) as caught:
        engine._run_in_process_generation(  # pyright: ignore[reportPrivateUsage]
            prompt="a cat",
            negative_prompt="",
            width=768,
            height=768,
            seed=1,
            style="photorealistic",
        )

    assert collected_first == [True]
    assert caught.value.__context__ is None


def test_out_of_memory_on_cuda_without_accelerate_says_how_to_add_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `accelerate` the card could not offload, so that is the remedy to name.

    Killed by: src/uclone_x/tools/builtin/image.py :: if device == "cuda" and not accelerate_is_installed():
    Becomes: if False:
    """
    import uclone_x.tools.builtin.image as image_module

    pipeline = _sdxl_pipeline()
    pipeline.to.side_effect = _FakeOutOfMemoryError(_RAW_OOM_TEXT)
    fake_torch = _FakeTorch(cuda=True, mps=False)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)
    monkeypatch.setattr(image_module, "accelerate_is_installed", lambda: False)

    with pytest.raises(ImageGenerationError) as caught:
        engine._ensure_pipeline_loaded()  # pyright: ignore[reportPrivateUsage]

    message = str(caught.value)
    assert message == f"{GPU_OUT_OF_MEMORY_MESSAGE} {ACCELERATE_MISSING_SENTENCE}"
    assert "install_package with package='accelerate'" in message
    for internal in ("CUDA out of memory", "GiB", "PYTORCH", "OutOfMemoryError", "/"):
        assert internal not in message


def test_out_of_memory_with_accelerate_installed_does_not_mention_it() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: if device == "cuda" and not accelerate_is_installed():
    Becomes: if device == "cuda":
    """
    with patch("uclone_x.tools.builtin.image.accelerate_is_installed", return_value=True):
        assert out_of_memory_message("cuda") == GPU_OUT_OF_MEMORY_MESSAGE


def test_out_of_memory_off_cuda_does_not_mention_accelerate() -> None:
    """Offload is a CUDA remedy; an MPS or CPU run gains nothing from `accelerate`.

    Killed by: src/uclone_x/tools/builtin/image.py :: if device == "cuda" and not accelerate_is_installed():
    Becomes: if not accelerate_is_installed():
    """
    with patch("uclone_x.tools.builtin.image.accelerate_is_installed", return_value=False):
        for device in ("mps", "cpu", None):
            assert out_of_memory_message(device) == GPU_OUT_OF_MEMORY_MESSAGE


def test_out_of_memory_while_generating_on_cuda_without_accelerate_says_how_to_add_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generate path names the device the pipeline was loaded onto.

    Killed by: src/uclone_x/tools/builtin/image.py :: self._torch_device = device
    Becomes: pass
    """
    import uclone_x.tools.builtin.image as image_module

    pipeline = _sdxl_pipeline()
    pipeline.side_effect = _FakeOutOfMemoryError(_RAW_OOM_TEXT)
    fake_torch = _FakeTorch(cuda=True, mps=False)
    engine = _engine_with_fakes(tmp_path, monkeypatch, fake_torch, pipeline)
    monkeypatch.setattr(image_module, "accelerate_is_installed", lambda: False)

    with pytest.raises(ImageGenerationError) as caught:
        engine._run_in_process_generation(  # pyright: ignore[reportPrivateUsage]
            prompt="a cat",
            negative_prompt="",
            width=768,
            height=768,
            seed=1,
            style="photorealistic",
        )

    assert str(caught.value) == f"{GPU_OUT_OF_MEMORY_MESSAGE} {ACCELERATE_MISSING_SENTENCE}"


def test_older_torch_out_of_memory_class_is_recognised() -> None:
    """torch before 2.5 has only `torch.cuda.OutOfMemoryError`."""

    class _OldCudaOom(RuntimeError):
        pass

    old_torch = MagicMock(spec=["cuda"])
    old_torch.cuda = MagicMock()
    old_torch.cuda.OutOfMemoryError = _OldCudaOom

    assert torch_out_of_memory_types(old_torch) == (_OldCudaOom,)


# --- Profile sizes and sampling reach the engines (Artist round 6, appendix F) -------------


def _default_registry(tmp_path: Path) -> Any:
    """The shipped registry, with this machine's `~/.uclone` override kept out (P8)."""
    from uclone_x.tools.builtin.media_registry import ModelRegistry

    return ModelRegistry(user_config_path=tmp_path / "no-user-models.yaml")


def test_the_illustrious_profile_renders_at_sdxl_megapixel_buckets(tmp_path: Path) -> None:
    """The shipped anillustrious_v4 profile gets ~1MP sizes for every offered ratio.

    Killed by: src/uclone_x/tools/builtin/image.py :: and profile.width * profile.height >= MEGAPIXEL_PROFILE_MIN_PIXELS
    Becomes: and profile.width * profile.height < MEGAPIXEL_PROFILE_MIN_PIXELS
    """
    profile = _default_registry(tmp_path).resolve("anillustrious_v4.safetensors")
    assert profile.model_id == "anillustrious_v4"

    sizes = {ratio: resolve_aspect_dimensions(ratio, profile) for ratio in SDXL_MEGAPIXEL_BUCKETS}

    assert sizes == {
        "1:1": (1024, 1024),
        "3:4": (896, 1152),
        "4:3": (1152, 896),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
    }
    for width, height in sizes.values():
        assert width % 64 == 0 and height % 64 == 0
        assert min(width, height) >= 768  # Illustrious's training floor
        assert 0.95 <= width * height / 1024**2 <= 1.0


def test_the_generic_fallback_keeps_the_legacy_sizes_and_engine_defaults(tmp_path: Path) -> None:
    """An unrecognised checkpoint's profile is a guess, so nothing about it changes.

    Killed by: src/uclone_x/tools/builtin/image.py :: return profile is not None and profile.model_id != GENERIC_FALLBACK_MODEL_ID
    Becomes: return profile is not None
    """
    fallback = _default_registry(tmp_path).resolve("mystery.safetensors")
    assert fallback.model_id == GENERIC_FALLBACK_MODEL_ID

    assert resolve_aspect_dimensions("1:1", fallback) == (768, 768)
    assert resolve_aspect_dimensions("3:4", fallback) == (576, 768)
    assert resolve_sampling(fallback) == (None, None)
    assert resolve_sampling(None) == (None, None)


def test_a_small_canvas_profile_keeps_the_legacy_sizes() -> None:
    """A registered SD 1.5-class profile (512x768) is not pushed to SDXL buckets."""
    from uclone_x.tools.builtin.media_registry import ModelProfile, PromptFamily

    small = ModelProfile(
        model_id="sd15_anime",
        display_name="SD 1.5",
        family=PromptFamily.DANBOORU,
        width=512,
        height=768,
    )

    assert resolve_aspect_dimensions("3:4", small) == (576, 768)


def test_a_registered_profile_supplies_its_steps_and_guidance(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: return profile.steps, profile.cfg
    Becomes: return None, None
    """
    profile = _default_registry(tmp_path).resolve("anillustrious_v4.safetensors")

    assert resolve_sampling(profile) == (30, 5.5)


def _generating_engine(
    tmp_path: Path, pipeline: MagicMock
) -> tuple[LocalDiffusersImageEngine, _FakeTorch]:
    """An in-process engine whose pipeline is `pipeline`, generating with a fake torch."""
    from PIL import Image

    checkpoint = tmp_path / "anillustrious_v4.safetensors"
    checkpoint.write_bytes(b"dummy")
    engine = LocalDiffusersImageEngine(checkpoint_path=str(checkpoint))
    pipeline.return_value = MagicMock(images=[Image.new("RGB", (8, 8))])
    engine._pipeline = pipeline  # pyright: ignore[reportPrivateUsage]
    return engine, _FakeTorch(cuda=False, mps=False)


@pytest.mark.asyncio
async def test_the_dispatcher_runs_the_illustrious_profile_at_its_own_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the real dispatcher and in-process engine: 30 steps, cfg 5.5, 1MP.

    Before, the engine used DIFFUSERS_STEPS=20 and DIFFUSERS_GUIDANCE=7.0 and rendered 3:4
    at 576x768 whatever the profile said.

    Killed by: src/uclone_x/tools/builtin/image.py :: steps, cfg = resolve_sampling(profile)
    Becomes: steps, cfg = None, None
    """
    import importlib

    pipeline = _sdxl_pipeline()
    engine, fake_torch = _generating_engine(tmp_path, pipeline)
    monkeypatch.setattr(engine, "is_available", AsyncMock(return_value=True))
    real_import = importlib.import_module

    def import_module(name: str, package: str | None = None) -> Any:
        return fake_torch if name == "torch" else real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    remote, comfy, _ = _dispatcher_mocks(remote=False, comfy=False, local=True)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=remote,
        comfy_engine=comfy,
        local_engine=engine,
        registry=_default_registry(tmp_path),
    )

    result = await dispatcher.dispatch(
        prompt="1girl, solo", negative_prompt="", aspect_ratio="3:4", seed=1, style="anime"
    )

    kwargs = pipeline.call_args.kwargs
    assert kwargs["num_inference_steps"] == 30
    assert kwargs["guidance_scale"] == 5.5
    assert (kwargs["width"], kwargs["height"]) == (896, 1152)
    assert (result.width, result.height) == (896, 1152)


def test_the_in_process_engine_uses_the_given_steps_and_guidance(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: guidance_scale=cfg if cfg is not None else DIFFUSERS_GUIDANCE,
    Becomes: guidance_scale=DIFFUSERS_GUIDANCE,
    """
    pipeline = _sdxl_pipeline()
    engine, fake_torch = _generating_engine(tmp_path, pipeline)

    with patch("importlib.import_module", return_value=fake_torch):
        engine._run_in_process_generation(  # pyright: ignore[reportPrivateUsage]
            prompt="1girl",
            negative_prompt="",
            width=896,
            height=1152,
            seed=1,
            style="anime",
            steps=30,
            cfg=5.5,
        )

    assert pipeline.call_args.kwargs["num_inference_steps"] == 30
    assert pipeline.call_args.kwargs["guidance_scale"] == 5.5


def test_without_a_profile_the_in_process_engine_keeps_its_defaults(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: num_inference_steps=steps if steps is not None else DIFFUSERS_STEPS,
    Becomes: num_inference_steps=steps,
    """
    pipeline = _sdxl_pipeline()
    engine, fake_torch = _generating_engine(tmp_path, pipeline)

    with patch("importlib.import_module", return_value=fake_torch):
        engine._run_in_process_generation(  # pyright: ignore[reportPrivateUsage]
            prompt="a cat",
            negative_prompt="",
            width=768,
            height=768,
            seed=1,
            style="photorealistic",
        )

    assert pipeline.call_args.kwargs["num_inference_steps"] == DIFFUSERS_STEPS == 20
    assert pipeline.call_args.kwargs["guidance_scale"] == DIFFUSERS_GUIDANCE == 7.0


@pytest.mark.asyncio
async def test_the_comfy_engine_puts_profile_steps_and_guidance_into_the_graph() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: sampling["cfg"] = cfg
    Becomes: pass
    """
    engine = ComfyUIImageEngine(base_url="http://127.0.0.1:8188", checkpoint="ckpt.safetensors")
    client = AsyncMock()
    client.queue_prompt.return_value = "prompt-1"
    client.wait_for_output.return_value = ["out.png"]
    client.download_image.return_value = b"png"

    with patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client):
        await engine.generate(
            prompt="a castle",
            negative_prompt="",
            width=1152,
            height=896,
            seed=7,
            style="anime",
            steps=12,
            cfg=3.0,
        )
        await engine.generate(
            prompt="a castle", negative_prompt="", width=768, height=768, seed=7, style="anime"
        )

    profiled, default = (call.args[0] for call in client.queue_prompt.await_args_list)
    assert (profiled["3"]["inputs"]["steps"], profiled["3"]["inputs"]["cfg"]) == (12, 3.0)
    assert (profiled["5"]["inputs"]["width"], profiled["5"]["inputs"]["height"]) == (1152, 896)
    # No profile: the workflow's own defaults, unchanged.
    assert (default["3"]["inputs"]["steps"], default["3"]["inputs"]["cfg"]) == (30, 5.5)


@pytest.mark.asyncio
async def test_the_remote_worker_is_sent_steps_and_guidance_only_when_a_profile_set_them() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: payload["steps"] = steps
    Becomes: pass
    """
    engine = RemoteCudaImageEngine(base_url="http://10.0.0.50:8000")
    response = MagicMock(status_code=200, headers={"content-type": "image/png"}, content=b"png")

    with patch("httpx.AsyncClient.post", return_value=response) as post:
        await engine.generate(
            prompt="p",
            negative_prompt="",
            width=896,
            height=1152,
            seed=1,
            style="anime",
            steps=30,
            cfg=5.5,
        )
        await engine.generate(
            prompt="p", negative_prompt="", width=768, height=768, seed=1, style="anime"
        )

    profiled, legacy = (call.kwargs["json"] for call in post.await_args_list)
    assert (profiled["steps"], profiled["cfg"]) == (30, 5.5)
    assert (profiled["width"], profiled["height"]) == (896, 1152)
    assert "steps" not in legacy and "cfg" not in legacy


# --- Prompts past one CLIP window are encoded in windows, not truncated -------------------


class _FakeClipTokenizer:
    """Whitespace tokenizer: word `wN` is token id N. CLIP's begin/end ids, a chosen pad."""

    bos_token_id = 49406
    eos_token_id = 49407

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id
        self.calls: list[dict[str, Any]] = []

    def __call__(self, text: str, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        self.calls.append(kwargs)
        return SimpleNamespace(input_ids=[int(word[1:]) for word in text.split()])


class _FakeTensor:
    """A (1, sequence, width) tensor recorded as one label per sequence position."""

    def __init__(self, positions: list[Any], width: int) -> None:
        self.positions = positions
        self.width = width


class _FakeIds:
    def __init__(self, ids: list[int], device: Any) -> None:
        self.ids = ids
        self.device = device


class _FakeClipEncoder:
    """Returns, per window, `[0]` (pooled) and hidden states tagged by layer and token."""

    def __init__(self, name: str, width: int) -> None:
        self.name = name
        self.width = width
        self.devices: list[Any] = []

    def __call__(self, ids: _FakeIds, output_hidden_states: bool = False) -> Any:
        assert output_hidden_states
        self.devices.append(ids.device)

        class _Output(tuple[_FakeTensor]):  # noqa: SLOT001 - a test double for CLIP's output tuple
            hidden_states: list[_FakeTensor]

        layers = [
            _FakeTensor([(self.name, layer, token) for token in ids.ids], self.width)
            for layer in ("early", "penultimate", "last")
        ]
        output = _Output((_FakeTensor([("pooled", self.name, tuple(ids.ids))], 1280),))
        output.hidden_states = layers
        return output


class _FakeTensorTorch:
    """`tensor`, `cat`, `zeros_like` and `no_grad` over `_FakeTensor`."""

    def __init__(self) -> None:
        self.grad_disabled = False

    def tensor(self, rows: list[list[int]], device: Any = None) -> _FakeIds:
        assert len(rows) == 1
        return _FakeIds(rows[0], device)

    def cat(self, tensors: list[_FakeTensor], dim: int) -> _FakeTensor:
        assert self.grad_disabled, "encoding must run under no_grad"
        if dim == 1:
            assert len({t.width for t in tensors}) == 1
            return _FakeTensor([p for t in tensors for p in t.positions], tensors[0].width)
        assert dim == -1
        assert len({len(t.positions) for t in tensors}) == 1
        joined = [tuple(position) for position in zip(*(t.positions for t in tensors), strict=True)]
        return _FakeTensor(joined, sum(t.width for t in tensors))

    def zeros_like(self, tensor: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(["zero"] * len(tensor.positions), tensor.width)

    def no_grad(self) -> Any:
        import contextlib

        @contextlib.contextmanager
        def disabled() -> Any:
            self.grad_disabled = True
            try:
                yield
            finally:
                self.grad_disabled = False

        return disabled()


def _words(count: int, start: int = 1) -> str:
    return " ".join(f"w{n}" for n in range(start, start + count))


def _clip_pipeline(*, force_zeros: bool = True) -> MagicMock:
    """A specced SDXL pipeline with fake tokenizers and encoders (768 + 1280 wide)."""
    pipeline = _sdxl_pipeline()
    pipeline.tokenizer = _FakeClipTokenizer(pad_token_id=49407)
    pipeline.tokenizer_2 = _FakeClipTokenizer(pad_token_id=0)
    pipeline.text_encoder = _FakeClipEncoder("te1", 768)
    pipeline.text_encoder_2 = _FakeClipEncoder("te2", 1280)
    pipeline.config = MagicMock(force_zeros_for_empty_prompt=force_zeros)
    return pipeline


def test_a_prompt_that_fits_one_window_keeps_the_plain_path() -> None:
    """Exactly 75 tokens still fits, so nothing about a short prompt's encoding changes.

    Killed by: src/uclone_x/tools/builtin/image.py :: if longest <= CLIP_CHUNK_TOKENS:
    Becomes: if longest < CLIP_CHUNK_TOKENS:
    """
    pipeline = _clip_pipeline()

    assert long_prompt_embeds(pipeline, _FakeTensorTorch(), _words(75), _words(10)) is None
    for tokenizer in (pipeline.tokenizer, pipeline.tokenizer_2):
        assert tokenizer.calls[0]["add_special_tokens"] is False
        assert tokenizer.calls[0]["truncation"] is False


def test_a_long_prompt_is_encoded_in_windows_and_keeps_every_token() -> None:
    """100 tokens: two windows of 77, both encoders' penultimate layers joined to 2048 wide.

    Killed by: src/uclone_x/tools/builtin/image.py :: hidden.append(output.hidden_states[-2])
    Becomes: hidden.append(output.hidden_states[-1])
    """
    pipeline = _clip_pipeline()
    embeds = long_prompt_embeds(pipeline, _FakeTensorTorch(), _words(100), "w900 w901")

    assert embeds is not None
    positive = embeds["prompt_embeds"]
    assert (len(positive.positions), positive.width) == (2 * 77, 768 + 1280)
    te1_tokens = [position[0][2] for position in positive.positions]
    te2_tokens = [position[1][2] for position in positive.positions]
    assert {position[0][1] for position in positive.positions} == {"penultimate"}
    # Window 1: begin, tokens 1-75, end. Window 2: begin, tokens 76-100, end, then pad.
    assert te1_tokens[:77] == [49406, *range(1, 76), 49407]
    assert te1_tokens[77:104] == [49406, *range(76, 101), 49407]
    assert te1_tokens[104:] == [49407] * 50  # tokenizer 1 pads with its end token
    assert te2_tokens[104:] == [0] * 50  # tokenizer 2 pads with its own pad token
    # The negative is brought to the same two windows, the second empty.
    negative = embeds["negative_prompt_embeds"]
    assert len(negative.positions) == 2 * 77
    assert [p[0][2] for p in negative.positions[77:79]] == [49406, 49407]


def test_the_pooled_embedding_is_text_encoder_2s_on_the_first_window() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: pooled = first  # the last encoder's is kept: text_encoder_2's
    Becomes: pooled = pooled or first
    """
    pipeline = _clip_pipeline(force_zeros=False)
    embeds = long_prompt_embeds(pipeline, _FakeTensorTorch(), _words(80), _words(3, 500))

    assert embeds is not None
    (pooled,) = embeds["pooled_prompt_embeds"].positions
    assert pooled[:2] == ("pooled", "te2")
    assert pooled[2] == (49406, *range(1, 76), 49407)
    (negative_pooled,) = embeds["negative_pooled_prompt_embeds"].positions
    assert negative_pooled[:2] == ("pooled", "te2")
    assert negative_pooled[2][:4] == (49406, 500, 501, 502)


def test_a_negative_longer_than_the_prompt_sets_the_window_count() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: longest = max(len(sequence) for pair in ids for sequence in pair)
    Becomes: longest = max(len(pair[0]) for pair in ids)
    """
    pipeline = _clip_pipeline()
    embeds = long_prompt_embeds(pipeline, _FakeTensorTorch(), _words(5), _words(160, 300))

    assert embeds is not None
    assert len(embeds["prompt_embeds"].positions) == 3 * 77
    assert len(embeds["negative_prompt_embeds"].positions) == 3 * 77


def test_an_empty_negative_is_zeros_when_the_checkpoint_asks_for_it() -> None:
    """Diffusers zeroes an absent negative when `force_zeros_for_empty_prompt` is set.

    Killed by: src/uclone_x/tools/builtin/image.py :: zero_negative = not negative_prompt and bool(
    Becomes: zero_negative = False and bool(
    """
    embeds = long_prompt_embeds(_clip_pipeline(), _FakeTensorTorch(), _words(90), "")
    assert embeds is not None
    assert set(embeds["negative_prompt_embeds"].positions) == {"zero"}
    assert len(embeds["negative_prompt_embeds"].positions) == 2 * 77
    assert embeds["negative_pooled_prompt_embeds"].positions == ["zero"]

    encoded = long_prompt_embeds(
        _clip_pipeline(force_zeros=False), _FakeTensorTorch(), _words(90), ""
    )
    assert encoded is not None
    assert "zero" not in encoded["negative_prompt_embeds"].positions


def test_a_long_prompt_reaches_the_pipeline_as_embeddings(tmp_path: Path) -> None:
    """The regression: the 76th token onward used to be cut by the pipeline's encoder.

    Killed by: src/uclone_x/tools/builtin/image.py :: embeds = long_prompt_embeds(pipeline, torch, guided, negative_prompt)
    Becomes: embeds = None
    """
    pipeline = _clip_pipeline()
    engine, _ = _generating_engine(tmp_path, pipeline)
    fake_torch = _FakeTensorTorch()
    fake_torch.Generator = MagicMock()  # type: ignore[attr-defined]

    with patch("importlib.import_module", return_value=fake_torch):
        engine._run_in_process_generation(  # pyright: ignore[reportPrivateUsage]
            prompt=_words(120),
            negative_prompt="w999",
            width=1024,
            height=1024,
            seed=1,
            style="unknown-style",
        )

    kwargs = pipeline.call_args.kwargs
    assert "prompt" not in kwargs and "negative_prompt" not in kwargs
    tokens = [position[0][2] for position in kwargs["prompt_embeds"].positions]
    assert 120 in tokens
    for key in ("negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"):
        assert key in kwargs


def test_a_short_prompt_reaches_the_pipeline_as_text(tmp_path: Path) -> None:
    pipeline = _clip_pipeline()
    engine, fake_torch = _generating_engine(tmp_path, pipeline)

    with patch("importlib.import_module", return_value=fake_torch):
        engine._run_in_process_generation(  # pyright: ignore[reportPrivateUsage]
            prompt="w1 w2",
            negative_prompt="",
            width=1024,
            height=1024,
            seed=1,
            style="unknown-style",
        )

    kwargs = pipeline.call_args.kwargs
    assert kwargs["prompt"] == "w1 w2"
    assert kwargs["negative_prompt"] is None
    assert "prompt_embeds" not in kwargs


# --- `count` is for seed variations of one scene, not for different scenes ----------------


def _context_with_request(tmp_path: Path, *requests: str) -> ToolContext:
    """A tool context whose agent's history holds `requests` as user messages, in order."""
    from types import SimpleNamespace

    from uclone_x.llm.models import ChatMessage, MessageRole

    history: list[ChatMessage] = []
    for request in requests:
        history.append(ChatMessage(role=MessageRole.USER, content=request))
        history.append(ChatMessage(role=MessageRole.ASSISTANT, content="ok"))
    return ToolContext(
        agent_id="artist",
        session_id="sess_variety",
        workspace_root=tmp_path,
        agent_delegate=SimpleNamespace(history=tuple(history[:-1])),
    )


def _counting_dispatcher() -> AsyncMock:
    dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"png",
        seed=1,
        engine_name="diffusers-sdxl",
        device_info="cpu",
        duration_seconds=0.1,
        width=1024,
        height=1024,
    )
    return dispatcher


VARIED_REQUESTS = (
    "이 캐릭터로 5개 장면 그려줘",
    "장면 5개 만들어줘",
    "다섯 장면으로 그려줘",
    "같은 캐릭터로 다양한 포즈 5장",
    "여러 가지 의상으로 4장",
    "각각 다른 배경으로 3장",
    "서로 다른 표정 4장",
    "3가지 버전으로 그려줘",
    "draw her in 5 different scenes",
    "various poses please, 4 images",
    "give me 3 distinct outfits",
)
SAME_SCENE_REQUESTS = (
    "같은 그림 5장 뽑아줘",
    "시드만 바꿔서 4장",
    "give me 5 variations of this",
    "이 그림 4장 더",
    "same image, 3 more seeds",
    "draw her 4 times",
    # A variety word is present, but the request is still one scene re-seeded.
    "포즈 그대로 시드만 바꿔서 4장",
    "same scene with different seeds",
    "각각 다른 시드로 같은 그림 3장",
    # Review of the first guard: bare nouns and "each" refused these seed requests.
    "같은 포즈로 5장 더 뽑아줘",
    "give me 4 versions, each looking at viewer",
    "여러 장 뽑아줘",
    "이 의상 그대로 4장",
)


@pytest.mark.parametrize("request_text", VARIED_REQUESTS)
@pytest.mark.asyncio
async def test_count_is_refused_when_the_user_asked_for_different_images(
    tmp_path: Path, request_text: str
) -> None:
    """The misuse: "5 different scenes" became one prompt rendered five times.

    Killed by: src/uclone_x/tools/builtin/image.py :: if request is not None and asks_for_varied_images(request):
    Becomes: if False:
    """
    dispatcher = _counting_dispatcher()
    tool = GenerateImageTool(dispatcher=dispatcher)

    with pytest.raises(PlainRefusalError) as caught:
        await tool.run(
            GenerateImageParams(prompt="1girl, solo", count=5),
            _context_with_request(tmp_path, request_text),
        )

    assert str(caught.value) == COUNT_FOR_VARIETY_REFUSAL
    dispatcher.dispatch.assert_not_awaited()


@pytest.mark.parametrize("request_text", SAME_SCENE_REQUESTS)
@pytest.mark.asyncio
async def test_count_is_allowed_for_variations_of_one_scene(
    tmp_path: Path, request_text: str
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image_set_intent.py :: if _SAME_PICTURE.search(message) is not None:
    Becomes: if False:
    """
    dispatcher = _counting_dispatcher()
    tool = GenerateImageTool(dispatcher=dispatcher)

    result = await tool.run(
        GenerateImageParams(prompt="1girl, solo", count=3),
        _context_with_request(tmp_path, request_text),
    )

    assert result["count"] == 3
    assert dispatcher.dispatch.await_count == 3


@pytest.mark.asyncio
async def test_only_the_latest_user_request_decides(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: for message in reversed(messages):
    Becomes: for message in messages:
    """
    tool = GenerateImageTool(dispatcher=_counting_dispatcher())
    params = GenerateImageParams(prompt="1girl, solo", count=2)

    allowed = await tool.run(
        params, _context_with_request(tmp_path, "5개 장면 그려줘", "같은 그림 2장 더")
    )
    assert allowed["count"] == 2

    with pytest.raises(PlainRefusalError):
        await tool.run(
            params, _context_with_request(tmp_path, "같은 그림 2장", "이번엔 서로 다른 포즈로")
        )


@pytest.mark.asyncio
async def test_the_variety_guard_leaves_prompts_single_images_and_direct_calls_alone(
    tmp_path: Path,
) -> None:
    dispatcher = _counting_dispatcher()
    tool = GenerateImageTool(dispatcher=dispatcher)
    varied = _context_with_request(tmp_path, "5개 장면 그려줘")

    await tool.run(GenerateImageParams(prompts=["scene one", "scene two"]), varied)
    await tool.run(GenerateImageParams(prompt="scene one"), varied)
    no_agent = ToolContext(agent_id="artist", session_id="s", workspace_root=tmp_path)
    await tool.run(GenerateImageParams(prompt="scene one", count=2), no_agent)

    assert dispatcher.dispatch.await_count == 5


def test_the_variety_refusal_is_plain_and_points_at_prompts() -> None:
    assert "'prompts'" in COUNT_FOR_VARIETY_REFUSAL
    assert "generate_image(prompts=" in COUNT_FOR_VARIETY_REFUSAL
    for internal in ("Error", "Traceback", "regex", "agent_delegate", "history", "_"):
        assert internal not in COUNT_FOR_VARIETY_REFUSAL.replace("generate_image", "")


def test_the_count_description_says_it_repeats_one_prompt() -> None:
    description = GenerateImageParams.model_fields["count"].description or ""
    assert "SAME" in description
    assert "'prompts'" in description
    for kind in ("scenes", "poses", "outfits"):
        assert kind in description


# --- Family x domain routing: image domain skills resolved per prompt family ---------------


def test_a_danbooru_prompt_gets_no_style_suffix_and_a_prose_one_does() -> None:
    """A tag list already carries the domain skill's style tags (design §3.6).

    Killed by: src/uclone_x/tools/builtin/image.py :: if family is PromptFamily.DANBOORU:
    Becomes: if False:
    """
    from uclone_x.tools.builtin.media_registry import PromptFamily

    tags = "modern architecture, glass facade, blue sky"

    assert style_guided_prompt(tags, "photorealistic", PromptFamily.DANBOORU) == tags
    assert style_guided_prompt(tags, "photorealistic", PromptFamily.NATURAL_PROSE) != tags
    assert style_guided_prompt(tags, "photorealistic") != tags


_PHOTO_SUFFIX = STYLE_SUFFIXES["photorealistic"]


@pytest.mark.asyncio
async def test_the_comfy_engine_submits_a_danbooru_prompt_without_the_style_suffix() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: prompt=style_guided_prompt(prompt, style, family),
    Becomes: prompt=style_guided_prompt(prompt, style),
    """
    from uclone_x.tools.builtin.media_registry import PromptFamily

    engine = ComfyUIImageEngine(base_url="http://127.0.0.1:8188", checkpoint="ckpt.safetensors")
    client = AsyncMock()
    client.queue_prompt.return_value = "prompt-1"
    client.wait_for_output.return_value = ["out.png"]
    client.download_image.return_value = b"png"

    with patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client):
        for family in (PromptFamily.DANBOORU, PromptFamily.NATURAL_PROSE, None):
            await engine.generate(
                prompt="1girl, solo",
                negative_prompt="",
                width=1024,
                height=1024,
                seed=7,
                style="photorealistic",
                family=family,
            )

    danbooru, prose, generic = (
        call.args[0]["6"]["inputs"]["text"] for call in client.queue_prompt.await_args_list
    )
    assert danbooru == "1girl, solo"
    assert prose == generic == f"1girl, solo, {_PHOTO_SUFFIX}"


@pytest.mark.asyncio
async def test_the_in_process_engine_submits_a_danbooru_prompt_without_the_style_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: guided = style_guided_prompt(prompt, style, family)
    Becomes: guided = style_guided_prompt(prompt, style)
    """
    import importlib

    from uclone_x.tools.builtin.media_registry import PromptFamily

    pipeline = _sdxl_pipeline()
    engine, fake_torch = _generating_engine(tmp_path, pipeline)
    real_import = importlib.import_module

    def import_module(name: str, package: str | None = None) -> Any:
        return fake_torch if name == "torch" else real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)

    submitted: list[str] = []
    for family in (PromptFamily.DANBOORU, PromptFamily.NATURAL_PROSE, None):
        await engine.generate(
            prompt="1girl, solo",
            negative_prompt="",
            width=1024,
            height=1024,
            seed=7,
            style="photorealistic",
            family=family,
        )
        submitted.append(pipeline.call_args.kwargs["prompt"])

    assert submitted == [
        "1girl, solo",
        f"1girl, solo, {_PHOTO_SUFFIX}",
        f"1girl, solo, {_PHOTO_SUFFIX}",
    ]


@pytest.mark.asyncio
async def test_the_dispatcher_tells_the_engine_the_active_prompt_family(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: family=profile.family,
    Becomes: family=None,
    """
    from uclone_x.tools.builtin.media_registry import PromptFamily

    mock_remote, mock_comfy, mock_local = _dispatcher_mocks(remote=False, comfy=False, local=True)
    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_remote,
        comfy_engine=mock_comfy,
        local_engine=mock_local,
        registry=_default_registry(tmp_path),
    )

    await dispatcher.dispatch(
        prompt="a house", negative_prompt="", aspect_ratio="1:1", seed=1, style="anime"
    )

    assert mock_local.generate.call_args.kwargs["family"] is PromptFamily.DANBOORU


class _SwitchableDispatcher(ImagePipelineDispatcher):
    """A dispatcher whose active profile a test sets directly."""

    def __init__(self, profile: Any) -> None:
        super().__init__()
        self.profile = profile

    def get_active_profile(self, own: str | None = None) -> Any:
        return self.profile


class _ListedSkills:
    """The one `SkillRegistryProtocol` method the description reads."""

    def __init__(self, skills: list[Any]) -> None:
        self._skills = skills

    def list_skills(self) -> list[Any]:
        return self._skills


def _listed_skill(name: str, *, family_sections: bool, status: Any = None) -> Any:
    from uclone_x.skills.models import SkillManifest, SkillOrigin, SkillStatus

    manifest = SkillManifest(
        name=name,
        description=name,
        origin=SkillOrigin.HUMAN,
        status=status or SkillStatus.ACTIVE,
        family_sections=family_sections,
    )
    return MagicMock(manifest=manifest)


def _profile(model_id: str, family: Any) -> Any:
    from uclone_x.tools.builtin.media_registry import ModelProfile

    return ModelProfile(model_id=model_id, display_name=model_id, family=family)


def test_the_description_stays_the_same_across_a_model_switch() -> None:
    """Switching the checkpoint leaves the tools layer of the request unchanged (#1723).

    Killed by: src/uclone_x/tools/builtin/image.py :: parts = [self.GEMINI_BASE_DESCRIPTION if gemini else self.BASE_DESCRIPTION]
    Becomes: parts = [self.GEMINI_BASE_DESCRIPTION if gemini else self.BASE_DESCRIPTION, self.active_profile().family.value]
    """
    from uclone_x.tools.builtin.media_registry import PromptFamily

    dispatcher = _SwitchableDispatcher(_profile("anime_model", PromptFamily.DANBOORU))
    tool = GenerateImageTool(dispatcher=dispatcher)

    first = tool.description
    dispatcher.profile = _profile("flux_model", PromptFamily.NATURAL_PROSE)
    second = tool.description

    assert first == second == GenerateImageTool.BASE_DESCRIPTION
    for text in (first, second):
        assert "anime_model" not in text and "flux_model" not in text
        assert "danbooru" not in text and "prose" not in text
        assert "load_skill" not in text  # no domain skills bound


def test_the_description_lists_only_active_family_sections_skills() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: if skill.manifest.family_sections and skill.manifest.status == SkillStatus.ACTIVE
    Becomes: if skill.manifest.status == SkillStatus.ACTIVE
    """
    from uclone_x.skills.models import SkillStatus
    from uclone_x.tools.builtin.media_registry import PromptFamily

    tool = GenerateImageTool(dispatcher=_SwitchableDispatcher(_profile("m", PromptFamily.DANBOORU)))
    tool.bind_skill_registry(
        cast(
            Any,
            _ListedSkills(
                [
                    _listed_skill("media-portrait", family_sections=True),
                    _listed_skill("media-architecture", family_sections=True),
                    _listed_skill("code_review", family_sections=False),
                    _listed_skill(
                        "media-draft", family_sections=True, status=SkillStatus.QUARANTINED
                    ),
                ]
            ),
        )
    )

    text = tool.description

    assert "load_skill('media-architecture'), load_skill('media-portrait')." in text
    assert "code_review" not in text
    assert "media-draft" not in text


def test_a_failing_settings_read_leaves_the_base_description_and_a_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: return self.BASE_DESCRIPTION
    Becomes: raise
    """

    class _Broken(ImagePipelineDispatcher):
        def draws_with_gemini(self, choice: Any = None) -> bool:
            raise OSError("settings file unreadable")

    tool = GenerateImageTool(dispatcher=_Broken())

    with caplog.at_level("ERROR"):
        text = tool.description

    assert text == GenerateImageTool.BASE_DESCRIPTION
    assert "Could not read the picture settings" in caplog.text


_ANATOMY_TERMS = ("anatomy", "hands", "fingers", "limbs", "animal", "deformed", "extra")


def test_the_heuristic_negative_is_domain_neutral() -> None:
    """A building, a chart or a product has no hands to get wrong (design §3.7).

    Killed by: src/uclone_x/tools/builtin/media_registry.py :: DOMAIN_NEUTRAL_NEGATIVE = "
    Becomes: DOMAIN_NEUTRAL_NEGATIVE = "bad anatomy, extra fingers, " + "
    """
    from uclone_x.tools.builtin.media_registry import ModelRegistry

    registry = ModelRegistry(user_config_path=Path("/nonexistent/no-user-models.yaml"))
    for checkpoint in ("my_pony_mix.safetensors", "mystery.safetensors"):
        negative = registry.resolve(checkpoint).default_negative.lower()
        assert negative, checkpoint
        for term in _ANATOMY_TERMS:
            assert term not in negative, (checkpoint, term)


def test_an_architecture_prompt_gets_no_anatomy_negative_from_any_shipped_profile(
    tmp_path: Path,
) -> None:
    """The negative an architecture render actually receives, per shipped model."""
    from uclone_x.tools.builtin.media_registry import fill_prompt_defaults

    registry = _default_registry(tmp_path)
    for checkpoint in (
        "anillustrious_v4.safetensors",
        "Illustrious-XL-v0.1.safetensors",
        "flux-2-klein-base-4b.safetensors",
        "mystery.safetensors",
    ):
        negative = fill_prompt_defaults(
            "modern architecture, glass facade, no humans", "", registry.resolve(checkpoint)
        ).negative_prompt
        for term in _ANATOMY_TERMS:
            assert term not in negative.lower(), (checkpoint, term)


def test_deterministic_seeding_call_index_variation() -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: f"{session_id}__{turn_idx}__{seed_prompt_key(prompt)}__{call_index}"
    Becomes: f"{session_id}__{turn_idx}__{seed_prompt_key(prompt)}"
    """
    seed0 = compute_deterministic_seed("sess_abc", 1, "prompt", call_index=0)
    seed1 = compute_deterministic_seed("sess_abc", 1, "prompt", call_index=1)
    seed2 = compute_deterministic_seed("sess_abc", 1, "prompt", call_index=2)
    assert seed0 != seed1
    assert seed1 != seed2
    assert seed0 == compute_deterministic_seed("sess_abc", 1, "prompt")


@pytest.mark.asyncio
async def test_generate_image_avoids_overwriting_existing_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: if not dest.exists():
    Becomes: if True:
    """
    import secrets
    from unittest.mock import AsyncMock

    from uclone_x.tools.builtin.image import ImageGenerationResult, ImagePipelineDispatcher

    def _fake_token_hex(_n: int = 16) -> str:
        return "fixed"

    monkeypatch.setattr(secrets, "token_hex", _fake_token_hex)
    dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"new_image_bytes",
        seed=123,
        engine_name="mock_engine",
        device_info="test_device",
        duration_seconds=0.1,
        width=512,
        height=512,
        mime_type="image/png",
    )
    context = ToolContext(agent_id="artist", workspace_root=tmp_path, session_id="sess_test")
    existing_file = tmp_path / "artifacts" / "sess_test" / "images" / "img_fixed.png"
    existing_file.parent.mkdir(parents=True, exist_ok=True)
    existing_file.write_bytes(b"existing_bytes")

    tool = GenerateImageTool(dispatcher=dispatcher)
    res = await tool.execute(
        params={"prompt": "test prompt"},
        context=context,
    )
    assert res.success is True
    output_dict = cast(dict[str, Any], res.output)
    assert _picture_rel(output_dict["relative_url"]) == "artifacts/sess_test/images/img_fixed_1.png"
    assert existing_file.read_bytes() == b"existing_bytes"
    assert (
        tmp_path / "artifacts" / "sess_test" / "images" / "img_fixed_1.png"
    ).read_bytes() == b"new_image_bytes"


@pytest.mark.asyncio
async def test_generate_image_avoids_overwriting_explicit_existing_artifact_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/builtin/image.py :: if dest_path.exists() and picture_rel.replace("\\", "/").startswith("artifacts/"):
    Becomes: if False:
    """
    from unittest.mock import AsyncMock

    from uclone_x.tools.builtin.image import ImageGenerationResult, ImagePipelineDispatcher

    dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"new_image_bytes",
        seed=123,
        engine_name="mock_engine",
        device_info="test_device",
        duration_seconds=0.1,
        width=512,
        height=512,
        mime_type="image/png",
    )
    context = ToolContext(agent_id="artist", workspace_root=tmp_path, session_id="sess_test")
    existing_file = tmp_path / "artifacts" / "images" / "img_explicit.png"
    existing_file.parent.mkdir(parents=True, exist_ok=True)
    existing_file.write_bytes(b"existing_bytes")

    tool = GenerateImageTool(dispatcher=dispatcher)
    res = await tool.execute(
        params={"prompt": "test prompt", "output_path": "artifacts/images/img_explicit.png"},
        context=context,
    )
    assert res.success is True
    output_dict = cast(dict[str, Any], res.output)
    assert artifact_path_from_url(output_dict["relative_url"]) == (
        "artifacts/images/img_explicit_1.png"
    )
    assert existing_file.read_bytes() == b"existing_bytes"
    assert (
        tmp_path / "artifacts" / "images" / "img_explicit_1.png"
    ).read_bytes() == b"new_image_bytes"


def _seed_recording_tool() -> tuple[GenerateImageTool, AsyncMock]:
    dispatcher = AsyncMock(spec=ImagePipelineDispatcher)
    dispatcher.dispatch.return_value = ImageGenerationResult(
        image_bytes=b"png",
        seed=123,
        engine_name="mock_engine",
        device_info="test_device",
        duration_seconds=0.1,
        width=512,
        height=512,
        mime_type="image/png",
    )
    return GenerateImageTool(dispatcher=dispatcher), dispatcher


async def _drawn_seeds(
    tool: GenerateImageTool,
    dispatcher: AsyncMock,
    context: ToolContext,
    prompts: list[str],
) -> list[int]:
    for prompt in prompts:
        res = await tool.execute(params={"prompt": prompt}, context=context)
        assert res.success is True
    return [c.kwargs["seed"] for c in dispatcher.dispatch.call_args_list]


@pytest.mark.asyncio
async def test_the_same_prompt_twice_in_a_turn_gets_two_seeds_and_replays(
    tmp_path: Path,
) -> None:
    """Review of #1966: through the tool, not `compute_deterministic_seed` alone.

    Killed by: src/uclone_x/tools/builtin/image.py :: index = counts.get(key, 0)
    Becomes: index = 0
    """
    context = ToolContext(
        agent_id="artist", workspace_root=tmp_path, session_id="sess_seed", turn_index=4
    )
    tool, dispatcher = _seed_recording_tool()
    first = await _drawn_seeds(tool, dispatcher, context, ["a red fox", "A red fox "])
    assert first[0] != first[1]

    replay_tool, replay_dispatcher = _seed_recording_tool()
    replay = await _drawn_seeds(
        replay_tool, replay_dispatcher, context, ["a red fox", "A red fox "]
    )
    assert replay == first


@pytest.mark.asyncio
async def test_a_prompts_seed_does_not_depend_on_which_prompts_ran_first(
    tmp_path: Path,
) -> None:
    """Calls of one step run concurrently, so the order they reach the tool is not fixed.

    Killed by: src/uclone_x/tools/builtin/image.py :: key = seed_prompt_key(prompt)
    Becomes: key = ""
    """
    context = ToolContext(
        agent_id="artist", workspace_root=tmp_path, session_id="sess_seed", turn_index=4
    )
    tool, dispatcher = _seed_recording_tool()
    fox, owl = await _drawn_seeds(tool, dispatcher, context, ["a red fox", "a snowy owl"])
    other_tool, other_dispatcher = _seed_recording_tool()
    owl_again, fox_again = await _drawn_seeds(
        other_tool, other_dispatcher, context, ["a snowy owl", "a red fox"]
    )
    assert (fox, owl) == (fox_again, owl_again)
    assert fox == compute_deterministic_seed("sess_seed", 4, "a red fox")


@pytest.mark.asyncio
async def test_the_repeat_count_keeps_only_each_sessions_latest_turn(
    tmp_path: Path,
) -> None:
    """A new turn starts the count again, and the table is bounded by session count.

    Killed by: src/uclone_x/tools/builtin/image.py :: counts = held[1] if held is not None and held[0] == turn_idx else {}
    Becomes: counts = held[1] if held is not None else {}
    Killed by: src/uclone_x/tools/builtin/image.py :: del self._turn_prompt_calls[next(iter(self._turn_prompt_calls))]
    Becomes: pass
    """
    tool, dispatcher = _seed_recording_tool()
    for turn in (1, 2):
        context = ToolContext(
            agent_id="artist", workspace_root=tmp_path, session_id="sess_seed", turn_index=turn
        )
        await tool.execute(params={"prompt": "a red fox"}, context=context)
    seeds = [c.kwargs["seed"] for c in dispatcher.dispatch.call_args_list]
    assert seeds == [
        compute_deterministic_seed("sess_seed", 1, "a red fox"),
        compute_deterministic_seed("sess_seed", 2, "a red fox"),
    ]

    for i in range(200):
        tool._repeat_index(f"sess_{i}", 1, "a red fox")  # pyright: ignore[reportPrivateUsage]
    held = tool._turn_prompt_calls  # pyright: ignore[reportPrivateUsage]
    assert len(held) == 64
    assert "sess_199" in held and "sess_seed" not in held


def test_a_link_is_read_back_as_the_workspace_path_it_was_made_from() -> None:
    """`artifact_path_from_url` inverts `artifact_content_url`, and reads nothing else (#2013).

    Killed by: src/uclone_x/tools/base.py :: if parts.scheme or parts.netloc or parts.path != "/api/artifacts/content":
    Becomes: if parts.path != "/api/artifacts/content":
    Killed by: src/uclone_x/tools/base.py :: if values is None or len(values) != 1 or not values[0]:
    Becomes: if values is None or not values[0]:
    """
    from uclone_x.tools.base import artifact_content_url, artifact_path_from_url, linked_paths

    for rel in ("images/cat.png", "images/a cat (1).png", "그림/고양이.png"):
        assert artifact_path_from_url(artifact_content_url(rel)) == rel
    for other in (
        None,
        3,
        "https://host/api/artifacts/content?path=a.png",
        "/api/artifacts/other?path=a.png",
        "/api/artifacts/content?path=a.png&path=b.png",
        "/api/artifacts/content?path=",
    ):
        assert artifact_path_from_url(other) is None
    one = artifact_content_url("a.png")
    two = artifact_content_url("b.png")
    output = {"relative_url": one, "images": [{"relative_url": one}, {"relative_url": two}, 7]}
    assert linked_paths(output) == ["a.png", "b.png"]
    assert linked_paths("a.png") == []
