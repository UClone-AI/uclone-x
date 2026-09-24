"""Unit tests for the remote / ComfyUI / in-process hybrid image pipeline (#849, #1095)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uclone_x.tools.builtin.comfy_client import COMFY_DEFAULT_CHECKPOINT
from uclone_x.tools.builtin.image import (
    COMFY_URL_ENV,
    DEFAULT_CHECKPOINTS,
    IMAGE_CHECKPOINT_ENV,
    IN_PROCESS_REQUIREMENTS,
    CheckpointResolution,
    ComfyUIImageEngine,
    GenerateImageParams,
    GenerateImageTool,
    ImageGenerationError,
    ImageGenerationResult,
    ImagePipelineDispatcher,
    LocalDiffusersImageEngine,
    RemoteCudaImageEngine,
    compute_deterministic_seed,
    diffusers_install_hint,
    expand_checkpoint_path,
    in_process_dependency_problems,
    resolve_aspect_dimensions,
    style_guided_prompt,
    version_release,
)
from uclone_x.tools.models import ToolContext


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

    Killed by: src/uclone_x/tools/builtin/image.py :: ("Remote CUDA worker", self._remote_engine),
    Becomes: ("Remote CUDA worker", self._local_engine),
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

    Killed by: src/uclone_x/tools/builtin/image.py :: ("detected local ComfyUI daemon", self._comfy_engine),
    Becomes: ("detected local ComfyUI daemon", self._local_engine),
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

    Killed by: src/uclone_x/tools/builtin/image.py :: ("in-process diffusers engine", self._local_engine),
    Becomes: ("in-process diffusers engine", self._comfy_engine),
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
    assert result["engine"] == "diffusers-sdxl"
    assert result["width"] == 1024
    assert result["height"] == 1024
    assert result["bytes_written"] == len(b"fake_png_binary_data")

    # Verify written file on disk
    img_path = tmp_path / result["path"]
    assert img_path.is_file()
    assert img_path.read_bytes() == b"fake_png_binary_data"


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
    assert "artifacts/images/img_" in result.artifacts[0]


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
    assert len(result["paths"]) == 3
    assert len(result["relative_urls"]) == 3
    assert mock_dispatcher.dispatch.call_count == 3
    # Seeds should be progressive
    assert result["images"][0]["seed"] == 2000
    assert result["images"][1]["seed"] == 2001
    assert result["images"][2]["seed"] == 2002
    assert "gundam girl robot armor #1" in result["markdown_gallery"]

    # Verify all 3 files exist on disk
    for img_meta in result["images"]:
        f_path = tmp_path / img_meta["path"]
        assert f_path.is_file()
        assert f_path.read_bytes() == f"png_{img_meta['seed']}".encode()


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
    rel_path = result.output["path"]
    rel_meta = result.output["meta_path"]
    assert isinstance(rel_path, str)
    assert isinstance(rel_meta, str)

    # Compact short ID pattern: artifacts/images/img_<6hex>.png
    assert re.match(r"^artifacts/images/img_[0-9a-f]{6}\.png$", rel_path)
    assert re.match(r"^artifacts/images/img_[0-9a-f]{6}\.json$", rel_meta)

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

    path1 = res1.output["path"]
    path2 = res2.output["path"]
    assert isinstance(path1, str)
    assert isinstance(path2, str)
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
    mock_pipeline = MagicMock()
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
