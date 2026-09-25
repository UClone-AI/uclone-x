"""Native and remote hybrid image generation tools for UClone-X agents."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import inspect
import json
import logging
import os
import platform
import secrets
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import UCloneXError
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.comfy_client import (
    DEFAULT_COMFYUI_BASE_URL,
    ComfyClient,
    default_comfy_checkpoint,
)
from uclone_x.tools.builtin.comfy_image_tool import build_txt2img_workflow
from uclone_x.tools.builtin.media_registry import (
    ModelProfile,
    ModelRegistry,
    PromptFamily,
    optimize_prompts,
)
from uclone_x.tools.models import ToolContext, ToolResult

logger = logging.getLogger(__name__)

#: Points the dispatcher at a remote CUDA worker.
REMOTE_URL_ENV = "UCX_IMAGE_REMOTE_URL"
#: Overrides the checkpoint the in-process engine loads.
IMAGE_CHECKPOINT_ENV = "UCX_IMAGE_CHECKPOINT"
#: Overrides the address the ComfyUI engine probes. The checkpoint it asks that daemon for is
#: overridden by `COMFY_CHECKPOINT_ENV`, which lives in `comfy_client` because the agent tool
#: reads it too and cannot import this module (#1223).
COMFY_URL_ENV = "UCX_COMFYUI_URL"
#: Single-file SDXL checkpoints looked for, in order, when nothing is configured. Single-file
#: is the requirement, not the specific weights: such a file carries UNet, CLIP and VAE
#: together, which is what lets the in-process engine run with no daemon at all (#1095).
DEFAULT_CHECKPOINTS = (
    "~/ai_models/checkpoints/anillustrious_v4.safetensors",
    "~/ai_models/checkpoints/Illustrious-XL-v0.1.safetensors",
    # Appended, not substituted: the two above are what a machine that already had a
    # checkpoint had, and dropping them would un-find a file that works today. This third
    # entry is the one the installer can actually *fetch* — `bootstrap.download_image_
    # checkpoint` writes SDXL base 1.0 here — so the path the download lands on and the
    # path the prober searches are the same string in one place (#1095 follow-up).
    "~/ai_models/checkpoints/sd_xl_base_1.0.safetensors",
)
DIFFUSERS_STEPS = 20
DIFFUSERS_GUIDANCE = 7.0
#: Free CUDA memory below which the SDXL pipeline is offloaded to the CPU instead of being
#: loaded onto the card whole. An estimate, not a measurement: SDXL's float16 weights are
#: about 7 GB (UNet ~5.1 GB, both text encoders ~1.6 GB, VAE ~0.2 GB), and denoising plus
#: the float32 VAE decode at up to 1024 px was budgeted at about 3 GB more. A 16 GB card
#: with qwen3:8b resident in Ollama at a 16K window has roughly 8 GB free, so it offloads.
CUDA_RESIDENT_MIN_FREE_BYTES = 10 * 1024**3
#: Where WSL2 puts the host driver's user-space tools, `nvidia-smi` among them. It is
#: usually on PATH there too; this is the fallback when a login shell did not add it.
WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"
#: The whole-device memory query, in MiB with no header or units: one `used, total` line
#: per GPU.
NVIDIA_SMI_MEMORY_QUERY = (
    "--query-gpu=memory.used,memory.total",
    "--format=csv,noheader,nounits",
)
#: How long the `nvidia-smi` query may take before its figure is treated as unknown. It
#: answers in well under a second on a working driver; a wedged one must not hold up the
#: image load, and an unknown figure only costs the slower offloaded path.
NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
#: What a person is told when the graphics card runs out of memory. Plain copy on purpose:
#: torch's own text names allocator internals and byte counts, which help nobody decide.
GPU_OUT_OF_MEMORY_MESSAGE = (
    "The graphics card ran out of memory while making the image. Close other programs "
    "using the graphics card, or try a smaller image, then ask again."
)


def expand_checkpoint_path(path: str) -> str:
    """The filesystem path a configured checkpoint string means — the only place that decides.

    A configured path and the listing of candidate paths used to answer this question
    separately: `media.local_checkpoints()` expanded `~`, `checkpoint_resolution()` did not,
    so `UCX_IMAGE_CHECKPOINT=~/ai_models/checkpoints/foo.safetensors` listed as present and
    resolved as missing (#1123). Two answers about one path is the defect, so both callers
    now ask here and there is no second reading to drift from.

    The reading is the shell's, for `~` and `~user` only:

    * `~` expands. `DEFAULT_CHECKPOINTS` are themselves written home-relative and have always
      been expanded, so refusing a user-typed `~` would make the configured path obey a
      narrower rule than the default it overrides.
    * `$VAR` does **not** expand, and is left to fail as the literal path it is. Expanding it
      would mean deciding what an *unset* variable means, and the only non-prompting answer —
      substituting an empty string — turns a typo into a silently different path, which is
      the class of failure P6 forbids. A literal `$HOME` that is not on disk is reported
      `missing` under the name the user typed, which is checkable.
    * A relative path stays relative, and a `~` that is not the first character (a file
      genuinely named `back~up.safetensors`) is left alone — both by `expanduser`, on both
      sides, because there is only one side now.
    """
    return os.path.expanduser(path)


#: Appended to the prompt per style preset. SDXL takes its style from the prompt rather than
#: from a parameter, so a preset that is not written into the prompt does nothing at all.
STYLE_SUFFIXES = {
    "photorealistic": "photorealistic, sharp focus, natural lighting, high detail",
    "anime": "anime illustration, clean line art, cel shading, vibrant colors",
    "artistic": "painterly, expressive brushwork, rich composition",
    "diagram": "clean vector diagram, flat colors, legible labels, plain background",
}


#: What the in-process engine imports when it generates, each with the floor it needs.
#: `diffusers` alone was not the requirement: it declares neither `torch` nor
#: `transformers`, so installing the `media` extra could leave the engine importable and
#: unable to run (#1095). A beginner's environment is not this one — a package may be
#: absent, or present at a version older than the API used here — and both are reported by
#: name instead of surfacing as a traceback part-way through a checkpoint load (P6).
IN_PROCESS_REQUIREMENTS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("diffusers", "diffusers>=0.31.0", (0, 31)),
    ("torch", "torch>=2.2.0", (2, 2)),
    ("transformers", "transformers>=4.40.0", (4, 40)),
)

#: What an install of the in-process engine asks for: every readiness requirement, plus
#: `accelerate`. It is installed but not required: the engine generates without it, and
#: only a CUDA card short of memory needs it, for `enable_model_cpu_offload` (see
#: `place_pipeline`). Were it in `IN_PROCESS_REQUIREMENTS`, an MPS or CPU engine that
#: worked before an upgrade would report itself not ready after it.
IN_PROCESS_INSTALL_REQUIREMENTS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    *IN_PROCESS_REQUIREMENTS,
    ("accelerate", "accelerate>=0.31.0", (0, 31)),
)


@dataclass(frozen=True)
class DependencyProblem:
    """One in-process requirement that is absent, or installed below its floor."""

    module: str
    requirement: str
    installed: str | None

    def describe(self) -> str:
        """A sentence naming the package, what is there, and what is wanted."""
        if self.installed is None:
            return f"'{self.module}' is not installed (needs {self.requirement})"
        return f"'{self.module}' is {self.installed}, older than the required {self.requirement}"


#: How the in-process engine's search for a checkpoint ended.
#:
#: `missing` and `unconfigured` are both "no file to load", which is why they were one
#: state until #1120; they are not the same mistake. Someone who mistyped
#: `UCX_IMAGE_CHECKPOINT` has already done the thing the `unconfigured` remedy tells them
#: to do, and collapsing the two sends them to re-check a variable that is set (P6).
CheckpointState = Literal["present", "missing", "unconfigured"]


@dataclass(frozen=True)
class CheckpointResolution:
    """The outcome of looking for the in-process engine's checkpoint, with its reason."""

    state: CheckpointState
    #: The file for `present`; the configured path that is not there for `missing`; None
    #: for `unconfigured`, where there is no path to name.
    path: str | None = None
    #: What named that path — `UCX_IMAGE_CHECKPOINT`, or the engine's own argument. None
    #: when nothing named it: a default that was found, or nothing found at all.
    source: str | None = None
    #: What the user actually typed, when `expand_checkpoint_path` changed it. None when
    #: there was nothing to expand. Reporting only the expanded path would name a string
    #: that appears in no config the user can go and edit.
    literal: str | None = None

    @property
    def usable(self) -> bool:
        """Whether a load may be attempted. Only `present` has a file behind it."""
        return self.state == "present"

    def describe(self) -> str:
        """One sentence: what was looked for, what was found, and what to do about it."""
        looked = ", ".join(DEFAULT_CHECKPOINTS)
        if self.state == "present":
            return f"Checkpoint '{self.path}' is on disk."
        if self.state == "missing":
            expanded = f" (expanded to '{self.path}')" if self.literal else ""
            named = self.literal or self.path
            if self.source == IMAGE_CHECKPOINT_ENV:
                return (
                    f"{IMAGE_CHECKPOINT_ENV} is set to '{named}'{expanded}, but no file is "
                    f"there. Correct that path, or unset {IMAGE_CHECKPOINT_ENV} to fall back "
                    f"to {looked}."
                )
            return (
                f"The engine was given checkpoint '{named}'{expanded}, but no file is there. "
                "Correct that path, or pass none to fall back to "
                f"{IMAGE_CHECKPOINT_ENV} and {looked}."
            )
        return (
            f"No checkpoint is configured and none was found on disk. Point "
            f"{IMAGE_CHECKPOINT_ENV} at a single-file SDXL checkpoint, or place one at "
            f"{looked}."
        )


def version_release(version: str) -> tuple[int, ...]:
    """The leading numeric components of a version string: '2.4.1+cpu' -> (2, 4, 1).

    Written out rather than taken from `packaging`, which is not a dependency of this
    package and so may be missing in exactly the thin environment this check is for.
    """
    parts: list[int] = []
    for chunk in version.split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def in_process_dependency_problems() -> tuple[DependencyProblem, ...]:
    """Every in-process requirement that is missing or too old, listed one by one.

    An importable package with no distribution metadata is left alone: its version cannot
    be read, and refusing on that would be a false alarm in the other direction.
    """
    import importlib.metadata
    import importlib.util

    problems: list[DependencyProblem] = []
    for module, requirement, floor in IN_PROCESS_REQUIREMENTS:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            problems.append(DependencyProblem(module, requirement, None))
            continue
        try:
            installed = importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError:
            continue
        if version_release(installed) < floor:
            problems.append(DependencyProblem(module, requirement, installed))
    return tuple(problems)


def diffusers_install_hint() -> str:
    """How to install the in-process image dependencies into the interpreter running UClone-X.

    A uv-created environment has no pip, so `pip install ...` is not an answer there; naming
    the interpreter keeps the packages landing where `ucx` imports from.
    """
    packages = " ".join(f"'{requirement}'" for _, requirement, _ in IN_PROCESS_INSTALL_REQUIREMENTS)
    return (
        "Install them into the environment running UClone-X: re-run `ucx start` and accept "
        f"the image generator, or run `uv pip install --python {sys.executable} {packages}`."
    )


def in_process_install_remedy() -> str:
    """The remedy for missing dependencies, addressed to the model reading a tool result.

    Separate from `diffusers_install_hint`, which `ucx media status` prints to a person and
    so rightly names a shell command. A tool result is read by the model, and an 8B model
    relayed that shell command to the user verbatim instead of repairing anything (#1107).
    Here the remedy is the tool the model can actually call.
    """
    return (
        "Call install_package with package='media' to install them, then try generating "
        "the image again."
    )


def style_guided_prompt(prompt: str, style: str) -> str:
    """The prompt with its style preset appended, or unchanged for an unknown preset."""
    suffix = STYLE_SUFFIXES.get(style)
    return f"{prompt}, {suffix}" if suffix else prompt


class ImageGenerationError(UCloneXError):
    """Raised when image generation fails due to engine, memory, or network errors."""


class GenerateImageParams(BaseModel):
    """Parameters for generating an image via the default image pipeline."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(
        ...,
        description="Positive text prompt describing the visual composition of the image to generate.",
    )
    style: Literal["photorealistic", "anime", "artistic", "diagram"] = Field(
        default="photorealistic",
        description="Visual style preset for aesthetic guidance.",
    )
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = Field(
        default="1:1",
        description="Aspect ratio of the generated image.",
    )
    negative_prompt: str = Field(
        default="",
        description="Negative text prompt specifying artifacts, text, or elements to exclude.",
    )
    seed_override: int | None = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description="Optional explicit seed to override deterministic turn-based entropy.",
    )
    output_path: str | None = Field(
        default=None,
        description="Optional relative file path within workspace. Defaults to 'artifacts/images/img_{seed}.png'.",
    )
    count: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of images to generate (1-10) with progressive seeds.",
    )


@dataclass
class ImageGenerationResult:
    """Internal result structure returned by image generation engines."""

    image_bytes: bytes
    seed: int
    engine_name: str
    device_info: str
    duration_seconds: float
    width: int
    height: int


class BaseImageEngine(ABC):
    """Abstract interface for image generation backends."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this engine can execute in the current runtime environment."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        style: str,
    ) -> ImageGenerationResult:
        """Generate an image returning raw image bytes and execution metadata."""


class RemoteCudaImageEngine(BaseImageEngine):
    """Offloads image generation to a remote CUDA workstation (e.g. Dell RTX 5070 Ti) via HTTP API."""

    def __init__(self, base_url: str | None = None, timeout_seconds: float = 30.0) -> None:
        self._base_url = (
            base_url or os.getenv(REMOTE_URL_ENV) or os.getenv("UCX_MEDIA_REMOTE_URL") or ""
        ).rstrip("/")
        self._timeout = timeout_seconds

    @property
    def base_url(self) -> str:
        return self._base_url

    async def is_available(self) -> bool:
        """Probe remote worker health endpoint."""
        if not self._base_url:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self._base_url}/health")
                return res.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        style: str,
    ) -> ImageGenerationResult:
        """Dispatch generation request to remote CUDA worker."""
        if not self._base_url:
            raise ImageGenerationError("Remote CUDA engine URL is not configured.")

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "style": style,
        }

        start_t = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._base_url}/v1/images/generations", json=payload)
                if resp.status_code != 200:
                    raise ImageGenerationError(
                        f"Remote CUDA worker returned HTTP {resp.status_code}: {resp.text}"
                    )
                # Parse response: supports raw bytes or JSON containing base64/bytes
                content_type = resp.headers.get("content-type", "")
                if "image/" in content_type:
                    raw_bytes = resp.content
                    device_info = resp.headers.get("x-device-info", "Remote CUDA GPU")
                else:
                    import base64

                    data = resp.json()
                    b64_data = data.get("image_base64") or data.get("data", [{}])[0].get("b64_json")
                    if not b64_data:
                        raise ImageGenerationError(
                            "Remote CUDA worker response did not contain image data."
                        )
                    raw_bytes = base64.b64decode(b64_data)
                    device_info = data.get("device_info", "Remote CUDA GPU")

        except httpx.TimeoutException as exc:
            raise ImageGenerationError(
                f"Remote CUDA worker timed out after {self._timeout}s."
            ) from exc
        except Exception as exc:
            if isinstance(exc, ImageGenerationError):
                raise
            raise ImageGenerationError(
                f"Failed to communicate with remote CUDA worker at {self._base_url}: {exc}"
            ) from exc

        duration = time.monotonic() - start_t
        return ImageGenerationResult(
            image_bytes=raw_bytes,
            seed=seed,
            engine_name="remote-cuda-fastapi",
            device_info=device_info,
            duration_seconds=round(duration, 3),
            width=width,
            height=height,
        )


def select_torch_device(torch: Any) -> tuple[str, Any]:
    """The device and dtype the in-process engine loads SDXL onto: CUDA, then MPS, then CPU.

    Until the fresh-machine test on an RTX 5070 Ti, only MPS was checked, so an NVIDIA
    machine with a working CUDA build of torch ran SDXL on its CPU in float32 while
    reporting nothing wrong. Half precision is used on either accelerator; the CPU keeps
    float32, because most CPU kernels have no fast float16 path.
    """
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda", torch.float16
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def describe_torch_device(torch: Any, device: str) -> str:
    """What `device_info` says about the device a pipeline was loaded onto.

    A CUDA device is named by its GPU, because "cuda (x86_64)" says which architecture
    the host CPU has and nothing about the card doing the work. Everything else keeps the
    host processor, which *is* the hardware for MPS (the SoC) and for the CPU.
    """
    if device == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    return f"{device} ({platform.processor() or platform.machine()})"


def in_process_device() -> str | None:
    """The device the in-process engine would load onto, or None when torch cannot say.

    Read the same way `_ensure_pipeline_loaded` chooses, so `ucx media status` names the
    device a generation would actually use rather than the one this module once assumed.
    """
    try:
        import importlib

        torch: Any = importlib.import_module("torch")
        device, dtype = select_torch_device(torch)
        precision = "float16" if dtype == torch.float16 else "float32"
        return f"{describe_torch_device(torch, device)}, {precision}"
    except Exception:
        return None


def running_under_wsl(
    osrelease: Path = Path("/proc/sys/kernel/osrelease"),
    version: Path = Path("/proc/version"),
    dxg: Path = Path("/dev/dxg"),
) -> bool:
    """Whether this is a Linux running under Windows' WSL2.

    The kernel names itself "microsoft" / "WSL" in its release string there, and the GPU
    reaches the guest through the `/dev/dxg` paravirtual device. Either is enough: a
    custom kernel can drop the name, and a WSL without GPU support has no `/dev/dxg`
    but then has no CUDA either.
    """
    for source in (osrelease, version):
        try:
            text = source.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        if "microsoft" in text or "wsl" in text:
            return True
    return dxg.exists()


def parse_nvidia_smi_free_bytes(output: str) -> int | None:
    """The least free memory of any GPU in `nvidia-smi`'s `used, total` MiB lines.

    None when there is no line or any line does not parse (`[N/A]`, an error message on
    stdout): a figure that cannot be read is unknown, never ample. With several cards the
    least free one is taken, because `nvidia-smi` numbers GPUs without regard to
    `CUDA_VISIBLE_DEVICES` and so cannot say which of them torch's device 0 is; the
    tightest card can only send the load down the slower offloaded path.
    """
    free: list[int] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            used_mib, total_mib = (int(field.strip()) for field in line.split(","))
        except ValueError:
            return None  # an unreadable line makes the whole reading unknown
        free.append(max(total_mib - used_mib, 0) * 1024**2)
    return min(free, default=None)


def nvidia_smi_output() -> str | None:
    """`nvidia-smi`'s whole-device memory query, or None when it cannot be run."""
    executable = shutil.which("nvidia-smi") or WSL_NVIDIA_SMI
    try:
        completed = subprocess.run(
            [executable, *NVIDIA_SMI_MEMORY_QUERY],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def nvml_free_bytes() -> int | None:
    """The least free memory of any GPU, read through NVML, or None when NVML is absent.

    `nvidia-ml-py` (imported as `pynvml`) is not an in-process requirement, so it is used
    when present and otherwise `nvidia-smi` -- a front end to the same library -- is.
    Free is taken as total minus used, the figures `nvidia-smi` shows.
    """
    try:
        import importlib

        pynvml: Any = importlib.import_module("pynvml")
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        free: list[int] = []
        for index in range(int(pynvml.nvmlDeviceGetCount())):
            memory = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index))
            free.append(max(int(memory.total) - int(memory.used), 0))
        return min(free) if free else None
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def device_wide_free_bytes() -> int | None:
    """Free GPU memory as the driver counts it for the whole device, or None if unknown."""
    free_bytes = nvml_free_bytes()
    if free_bytes is not None:
        return free_bytes
    output = nvidia_smi_output()
    return None if output is None else parse_nvidia_smi_free_bytes(output)


def cuda_has_room_for_sdxl(torch: Any) -> bool:
    """Whether the CUDA card has enough free memory to hold the whole SDXL pipeline.

    Read from `torch.cuda.mem_get_info`, which on native Linux and Windows counts what
    *other* processes hold too -- an Ollama model kept resident is exactly the case this
    exists for. When the figure cannot be read, the answer is no: offloading is slower,
    but it does not run out.

    Under WSL2 that reading is not trusted alone. On an RTX 5070 Ti 16 GB it reported
    14.66 GiB free while `nvidia-smi` showed 8115 MiB held by a resident qwen3:8b, so SDXL
    was loaded whole and peaked at 15860 of 16303 MiB. There the driver's whole-device
    figure (NVML, else `nvidia-smi`) is read as well and the smaller of the two is used;
    when neither can be read, free memory is unknown and the pipeline offloads.

    Native Linux keeps the torch figure alone, by decision: there `cudaMemGetInfo` is the
    driver's device-wide counter, the one NVML reports, so a second reading adds a
    subprocess per load and nothing observed -- and it would have to guess which
    `nvidia-smi` GPU is torch's device 0, which `CUDA_VISIBLE_DEVICES` renumbers.
    """
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:
        free_bytes = 0
    if running_under_wsl():
        device_free = device_wide_free_bytes()
        free_bytes = min(int(free_bytes), device_free) if device_free is not None else 0
    return int(free_bytes) >= CUDA_RESIDENT_MIN_FREE_BYTES


def place_pipeline(pipeline: Any, torch: Any, device: str) -> tuple[Any, bool]:
    """Put the pipeline on `device`, offloading to the CPU when the card is short of memory.

    Returns the pipeline and whether it was offloaded. Offload keeps each sub-model on the
    card only while it runs, and VAE tiling decodes the image in pieces; together they fit
    SDXL beside a resident Ollama model where `.to("cuda")` ran out of memory. Tiling is
    the VAE's own switch (`AutoencoderKL.enable_tiling`); diffusers 0.40 has no
    pipeline-level `enable_vae_tiling`. Offload needs `accelerate`, which is an in-process
    requirement; if diffusers still refuses it (an `ImportError`, as for a version below
    its floor), the whole pipeline is loaded onto the card and a card that is then too
    small is reported by the out-of-memory handling.
    """
    if device == "cuda" and not cuda_has_room_for_sdxl(torch):
        try:
            pipeline.enable_model_cpu_offload()
        except ImportError:
            logger.info("accelerate is not installed; loading SDXL onto the card whole")
        else:
            pipeline.vae.enable_tiling()
            return pipeline, True
    return pipeline.to(device), False


#: Added to the out-of-memory message when the card is CUDA and `accelerate` is absent,
#: because then the pipeline could not offload and was loaded onto the card whole.
#: diffusers checks for `accelerate` once, when it is imported, so installing it takes
#: effect only in a fresh process.
ACCELERATE_MISSING_SENTENCE = (
    "The package that lets image generation share the graphics card, accelerate, is not "
    "installed. Call install_package with package='accelerate' to add it; it takes effect "
    "after UClone-X restarts."
)


def accelerate_is_installed() -> bool:
    """Whether `accelerate`, which CUDA model offload needs, can be imported here."""
    import importlib.util

    try:
        return importlib.util.find_spec("accelerate") is not None
    except (ImportError, ValueError):
        return False


def out_of_memory_message(device: str | None) -> str:
    """The plain out-of-memory message, naming `accelerate` only when its absence mattered."""
    if device == "cuda" and not accelerate_is_installed():
        return f"{GPU_OUT_OF_MEMORY_MESSAGE} {ACCELERATE_MISSING_SENTENCE}"
    return GPU_OUT_OF_MEMORY_MESSAGE


def torch_out_of_memory_types(torch: Any) -> tuple[type[BaseException], ...]:
    """The exception classes torch raises when a device runs out of memory.

    `torch.OutOfMemoryError` arrived in torch 2.5; older builds only have
    `torch.cuda.OutOfMemoryError`. Both are collected, and whichever is absent is skipped.
    """
    candidates = (
        getattr(torch, "OutOfMemoryError", None),
        getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
    )
    return tuple(
        kind for kind in candidates if isinstance(kind, type) and issubclass(kind, BaseException)
    )


class LocalDiffusersImageEngine(BaseImageEngine):
    """In-process generation from a single-file SDXL checkpoint, with no daemon (#1095).

    This is the baseline the beginner path rests on: a checkpoint file and the `diffusers`
    package are the whole of the requirement. The checkpoint carries UNet, CLIP and VAE in
    one file, so `from_single_file` is a complete pipeline — unlike the split ComfyUI layout
    of FLUX.2 Klein, which is a graph and is served by `ComfyUIImageEngine` instead.
    """

    def __init__(self, checkpoint_path: str | None = None) -> None:
        self._configured = checkpoint_path
        self._pipeline: Any = None
        self._device: str = f"cpu ({platform.processor() or platform.machine()})"
        #: The torch device the loaded pipeline was placed on ("cuda", "mps", "cpu").
        self._torch_device: str | None = None
        self._lock = threading.Lock()

    def checkpoint_resolution(self) -> CheckpointResolution:
        """Where the checkpoint search ended, and why — the three states, kept apart.

        A configured path that is not on disk is `missing` and carries that path, so the
        caller can say which file it failed to find. Nothing configured and nothing found
        is `unconfigured`. Both are "no file to load", and until #1120 both came back as a
        bare None, so every message downstream told a user who had mistyped
        `UCX_IMAGE_CHECKPOINT` to set `UCX_IMAGE_CHECKPOINT`.

        Every path here — configured or default — goes through `expand_checkpoint_path`,
        which is also what `media.local_checkpoints()` asks, so the listing and this
        resolution cannot disagree about what a `~` means (#1123). `path` is therefore the
        expanded path, which is what a load and the `\u2192` marker both need; `literal`
        carries what was typed, for the message.
        """
        configured, source = self._configured, "argument"
        if not configured:
            configured, source = os.getenv(IMAGE_CHECKPOINT_ENV), IMAGE_CHECKPOINT_ENV
        if configured:
            path = expand_checkpoint_path(configured)
            literal = configured if path != configured else None
            state: CheckpointState = "present" if os.path.exists(path) else "missing"
            return CheckpointResolution(state, path, source, literal)
        for candidate in DEFAULT_CHECKPOINTS:
            path = expand_checkpoint_path(candidate)
            if os.path.exists(path):
                return CheckpointResolution("present", path)
        return CheckpointResolution("unconfigured")

    def resolve_checkpoint(self) -> str | None:
        """The checkpoint this engine would load, or None when no candidate is on disk.

        Absence is reported as None rather than a default string: naming a file that is not
        there would let `is_available` promise a load that must fail (P6). Callers that
        need to tell *why* there is no file want `checkpoint_resolution` instead.
        """
        resolution = self.checkpoint_resolution()
        return resolution.path if resolution.usable else None

    async def is_available(self) -> bool:
        """Whether every import this engine makes is satisfied and a checkpoint is on disk."""
        if in_process_dependency_problems():
            return False
        return self.resolve_checkpoint() is not None

    def _ensure_pipeline_loaded(self) -> Any:
        """Lazily build the SDXL pipeline on the fastest device `select_torch_device` finds."""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            resolution = self.checkpoint_resolution()
            if not resolution.usable:
                raise ImageGenerationError(f"No image checkpoint to load. {resolution.describe()}")
            checkpoint = resolution.path
            problems = in_process_dependency_problems()
            if problems:
                raise ImageGenerationError(
                    "In-process image generation cannot run: "
                    + "; ".join(problem.describe() for problem in problems)
                    + ". "
                    + in_process_install_remedy()
                )
            try:
                import importlib

                diffusers: Any = importlib.import_module("diffusers")
                torch: Any = importlib.import_module("torch")
            except ImportError as exc:
                raise ImageGenerationError(
                    "In-process image generation failed to import its dependencies: "
                    f"{exc}. " + in_process_install_remedy()
                ) from exc

            device, dtype = select_torch_device(torch)
            logger.info("Loading SDXL checkpoint %s onto %s...", checkpoint, device)
            pipeline: Any = None
            offloaded = False
            out_of_memory = False
            try:
                pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(
                    checkpoint, torch_dtype=dtype
                )
                pipeline, offloaded = place_pipeline(pipeline, torch, device)
            except Exception as exc:
                if not isinstance(exc, torch_out_of_memory_types(torch)):
                    raise ImageGenerationError(
                        f"Failed to load SDXL checkpoint '{checkpoint}': {exc}"
                    ) from exc
                logger.info("Ran out of memory loading SDXL onto %s: %s", device, str(exc))
                out_of_memory = True
            if out_of_memory:
                # Outside the `except`: there the exception's traceback still holds the
                # frames that reference the half-placed pipeline, and empty_cache would
                # free nothing.
                del pipeline  # the half-placed one
                self._release_after_out_of_memory(torch)
                raise ImageGenerationError(out_of_memory_message(device)) from None
            self._pipeline = pipeline
            self._torch_device = device
            described = describe_torch_device(torch, device)
            self._device = f"{described}, offloading to CPU" if offloaded else described
            return pipeline

    def _release_after_out_of_memory(self, torch: Any) -> None:
        """Drop the pipeline and hand its CUDA memory back after an out-of-memory error.

        Called outside the `except` block, once the caller has deleted its own reference,
        so that nothing but garbage still points at the pipeline when the cache is emptied.
        Keeping a half-placed pipeline would make every later request fail the same way,
        and the memory torch caches would stay taken from Ollama, which shares the card.
        The next request builds the pipeline again and re-reads how much memory is free.
        """
        self._pipeline = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("torch.cuda.empty_cache failed after running out of memory")

    def _run_in_process_generation(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        style: str,
    ) -> bytes:
        """Run the diffusion loop on a worker thread and return PNG bytes."""
        import io

        pipeline: Any = self._ensure_pipeline_loaded()
        torch: Any = None
        try:
            import importlib

            torch = importlib.import_module("torch")
            generator: Any = torch.Generator(device="cpu").manual_seed(seed)
            output: Any = pipeline(
                prompt=style_guided_prompt(prompt, style),
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=DIFFUSERS_STEPS,
                guidance_scale=DIFFUSERS_GUIDANCE,
                generator=generator,
            )
            images: Any = getattr(output, "images", None)
            if not images:
                raise ImageGenerationError("The diffusers pipeline returned no image.")
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:
            if isinstance(exc, ImageGenerationError):
                raise
            if torch is None or not isinstance(exc, torch_out_of_memory_types(torch)):
                raise ImageGenerationError(f"In-process image generation failed: {exc}") from exc
            logger.info("Ran out of memory generating an image: %s", str(exc))
        # Only an out-of-memory error reaches here: the `try` returns and every other
        # failure raises. The release runs outside the `except` so the traceback no
        # longer keeps the pipeline alive.
        del pipeline  # the one that ran out
        self._release_after_out_of_memory(torch)
        raise ImageGenerationError(out_of_memory_message(self._torch_device)) from None

    async def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        style: str,
    ) -> ImageGenerationResult:
        """Run the diffusion loop in a worker thread and return PNG bytes without blocking event loop."""
        start_t = time.monotonic()
        img_bytes = await asyncio.to_thread(
            self._run_in_process_generation,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            style=style,
        )
        duration = time.monotonic() - start_t
        return ImageGenerationResult(
            image_bytes=img_bytes,
            seed=seed,
            engine_name="diffusers-sdxl",
            device_info=self._device,
            duration_seconds=round(duration, 3),
            width=width,
            height=height,
        )


class ComfyUIImageEngine(BaseImageEngine):
    """A ComfyUI daemon that is already running, detected and used but never installed (#1095).

    Detection is the whole of the contract: if `/system_stats` answers on the configured
    address, the graph-capable engine is used; if it does not, nothing is installed, started
    or explained at length — the in-process engine covers the case. This machine runs a
    ComfyUI for other projects, so taking its port or ending its process is out of bounds.
    """

    def __init__(self, base_url: str | None = None, checkpoint: str | None = None) -> None:
        self._base_url = base_url or os.getenv(COMFY_URL_ENV) or DEFAULT_COMFYUI_BASE_URL
        self._checkpoint = checkpoint or default_comfy_checkpoint()

    @property
    def base_url(self) -> str:
        """Address the engine probes; never a daemon this process started."""
        return self._base_url

    async def is_available(self) -> bool:
        """Whether a ComfyUI daemon answers at `base_url`."""
        client = ComfyClient(base_url=self._base_url)
        try:
            return await client.alive()
        except Exception:
            return False
        finally:
            await client.aclose()

    async def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        style: str,
    ) -> ImageGenerationResult:
        """Queue a txt2img graph on the running daemon and fetch the produced PNG."""
        client = ComfyClient(base_url=self._base_url)
        start_t = time.monotonic()
        try:
            workflow = build_txt2img_workflow(
                prompt=style_guided_prompt(prompt, style),
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
                checkpoint=self._checkpoint,
            )
            prompt_id = await client.queue_prompt(workflow)
            filenames = await client.wait_for_output(prompt_id)
            if not filenames:
                raise ImageGenerationError(
                    f"ComfyUI at {self._base_url} finished without producing an image."
                )
            img_bytes = await client.download_image(filenames[0])
        except ImageGenerationError:
            raise
        except Exception as exc:
            raise ImageGenerationError(
                f"ComfyUI generation at {self._base_url} failed: {exc}"
            ) from exc
        finally:
            await client.aclose()

        duration = time.monotonic() - start_t
        return ImageGenerationResult(
            image_bytes=img_bytes,
            seed=seed,
            engine_name="comfyui-local",
            device_info=f"ComfyUI daemon at {self._base_url}",
            duration_seconds=round(duration, 3),
            width=width,
            height=height,
        )


def compute_deterministic_seed(
    session_id: str,
    turn_idx: int,
    prompt: str,
    seed_override: int | None = None,
) -> int:
    """Derive seed deterministically from session identifier, turn index, and prompt hash (P6)."""
    if seed_override is not None:
        return seed_override
    token = f"{session_id}__{turn_idx}__{prompt.strip().lower()}"
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


def resolve_aspect_dimensions(
    aspect_ratio: str,
    profile: ModelProfile | None = None,
) -> tuple[int, int]:
    """Calculate pixel dimensions aligned to 64-pixel multiples."""
    if profile is not None and profile.family == PromptFamily.DANBOORU:
        match aspect_ratio:
            case "16:9":
                return (896, 512)
            case "9:16":
                return (512, 896)
            case "4:3":
                return (768, 576)
            case "3:4":
                return (576, 768)
            case _:
                return (768, 768)
    match aspect_ratio:
        case "16:9":
            return (896, 512)
        case "9:16":
            return (512, 896)
        case "4:3":
            return (768, 576)
        case "3:4":
            return (576, 768)
        case _:
            return (768, 768)


class ImagePipelineDispatcher:
    """Selects an engine by priority: remote CUDA, then a running ComfyUI, then in-process.

    The order encodes the ruling behind #1095. The *baseline* is last on purpose: the
    in-process engine needs no daemon, so a beginner with nothing but a checkpoint file
    still generates images. A ComfyUI that happens to be running is preferred above it
    because it is already warm and carries the graph features, but it is only ever
    detected — this dispatcher never installs, starts or stops one.
    """

    def __init__(
        self,
        remote_engine: RemoteCudaImageEngine | None = None,
        comfy_engine: ComfyUIImageEngine | None = None,
        local_engine: LocalDiffusersImageEngine | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._remote_engine = remote_engine or RemoteCudaImageEngine()
        self._comfy_engine = comfy_engine or ComfyUIImageEngine()
        self._local_engine = local_engine or LocalDiffusersImageEngine()
        self._registry = registry or ModelRegistry()

    @property
    def registry(self) -> ModelRegistry:
        """The model registry managing model profiles and prompt families."""
        return self._registry

    def get_active_profile(self) -> ModelProfile:
        """Resolve profile for currently configured or detected checkpoint."""
        ckpt: Any = getattr(self._comfy_engine, "_checkpoint", None)
        if isinstance(ckpt, (str, Path)) and ckpt:
            profile = self._registry.resolve(str(ckpt))
            if profile.model_id != "generic_fallback":
                return profile

        local_ckpt: Any = None
        if hasattr(self._local_engine, "resolve_checkpoint"):
            try:
                candidate = self._local_engine.resolve_checkpoint
                if callable(candidate):
                    val = candidate()
                    if inspect.iscoroutine(val):
                        val.close()
                    elif isinstance(val, (str, Path)):
                        local_ckpt = val
            except Exception:
                pass
        if isinstance(local_ckpt, (str, Path)) and local_ckpt:
            return self._registry.resolve(str(local_ckpt))

        if isinstance(ckpt, (str, Path)) and ckpt:
            return self._registry.resolve(str(ckpt))

        return self._registry.resolve(None)

    async def dispatch(
        self,
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        seed: int,
        style: str,
    ) -> ImageGenerationResult:
        """Route to the highest priority available local-private engine."""
        profile = self.get_active_profile()
        effective_prompt, effective_negative = optimize_prompts(prompt, negative_prompt, profile)
        width, height = resolve_aspect_dimensions(aspect_ratio, profile=profile)
        attempts: list[tuple[str, BaseImageEngine]] = [
            ("Remote CUDA worker", self._remote_engine),
            ("detected local ComfyUI daemon", self._comfy_engine),
            ("in-process diffusers engine", self._local_engine),
        ]

        for label, engine in attempts:
            if await engine.is_available():
                logger.info("Dispatching image generation to %s...", label)
                return await engine.generate(
                    prompt=effective_prompt,
                    negative_prompt=effective_negative,
                    width=width,
                    height=height,
                    seed=seed,
                    style=style,
                )

        raise ImageGenerationError(
            "No image generation engine available. Local Private-First enforcement: "
            + self.diagnostics()
        )

    def diagnostics(self) -> str:
        """Why each engine declined, named one by one.

        A single "nothing is available" tells the user nothing actionable, and the three
        engines fail for unrelated reasons: an address, a daemon, and a file (P6).
        """
        parts: list[str] = []
        if self._remote_engine.base_url:
            parts.append(
                f"Configured remote worker at '{self._remote_engine.base_url}' was unreachable."
            )
        else:
            parts.append(f"{REMOTE_URL_ENV} is not set.")

        # Configured and silent is a different problem from never configured, and telling
        # someone to "set UCX_COMFYUI_URL" when they already set it sends them to check a
        # variable that is correct (reviewer, PR #1096).
        if os.getenv(COMFY_URL_ENV):
            parts.append(
                f"No ComfyUI daemon answered at {self._comfy_engine.base_url}, the address "
                f"{COMFY_URL_ENV} names (optional — start that daemon, or unset the variable "
                "to fall back to the in-process engine)."
            )
        else:
            parts.append(
                f"No ComfyUI daemon answered at {self._comfy_engine.base_url} "
                f"(optional — start one, or set {COMFY_URL_ENV}, to enable it)."
            )

        problems = in_process_dependency_problems()
        if problems:
            parts.append(
                "The in-process engine is missing dependencies: "
                + "; ".join(problem.describe() for problem in problems)
                + ". "
                + in_process_install_remedy()
            )
        elif not (resolution := self._local_engine.checkpoint_resolution()).usable:
            # A configured-but-absent path is named here rather than folded into the
            # nothing-configured remedy: telling someone to set a variable they have
            # already set sends them to check the one thing that is not the problem
            # (#1120, the same shape as the ComfyUI case above).
            parts.append(
                "The in-process engine has no checkpoint to load. "
                f"{resolution.describe()} Run `ucx media status` to see this from the "
                "command line."
            )
        else:
            parts.append("The in-process engine is installed but declined; see the log above.")
        return " ".join(parts)


class GenerateImageTool(BaseTool[GenerateImageParams]):
    """Agent tool to generate visual illustrations, diagrams, and photos locally."""

    name = "generate_image"
    writes_files: ClassVar[bool] = True  # can create, modify or delete a file on the host (#1167)
    description = (
        "Generate a local-private image, illustration, or diagram from a text prompt. "
        "Runs in this process from a local checkpoint, on a detected local ComfyUI daemon, or on "
        "a local-network CUDA GPU worker — never on a paid cloud service. "
        "Saves the resulting image as an artifact in the workspace. "
        "To present the result to the user, include standard markdown ![description](relative_url) or [title](relative_url)."
    )
    params_type = GenerateImageParams

    def __init__(
        self,
        dispatcher: ImagePipelineDispatcher | None = None,
    ) -> None:
        dispatcher = dispatcher or ImagePipelineDispatcher()
        tool_desc = self.description
        profile: ModelProfile | None = None
        if hasattr(dispatcher, "get_active_profile"):
            try:
                candidate: Any = dispatcher.get_active_profile()
                if isinstance(candidate, ModelProfile):
                    profile = candidate
            except Exception:
                profile = None

        if profile is not None:
            tool_desc = (
                f"{self.description} Active Model: '{profile.model_id}' "
                f"({profile.family.value} prompt family). "
            )
            if profile.family == PromptFamily.DANBOORU:
                tool_desc += (
                    "Prompt MUST use English Danbooru tags (e.g. 1girl, solo, armor, glowing sword) "
                    "rather than natural Korean sentences. Safety/anatomy defense tags are automatically merged into negative_prompt if omitted. "
                    "Consult skill 'media-prompt-danbooru'."
                )
            elif profile.family == PromptFamily.NATURAL_PROSE:
                tool_desc += (
                    "Prompt should be descriptive English prose detailing lighting, composition, "
                    "and subject. This model does not use negative prompts; leave negative_prompt empty. "
                    "Consult skill 'media-prompt-flux'."
                )
            elif profile.family == PromptFamily.GENERIC:
                tool_desc += "Formulate prompts in English. Consult skill 'media-prompt-generic'."

        super().__init__(
            name=self.name,
            description=tool_desc,
            params_type=GenerateImageParams,
        )
        self._dispatcher = dispatcher

    async def run(
        self,
        params: GenerateImageParams,
        context: ToolContext,
    ) -> dict[str, Any]:
        """Execute image generation, write artifact, and return structured metadata."""
        session_id = context.session_id or "sess_default"
        turn_idx = getattr(context, "turn_index", 0) or 0
        if not turn_idx and getattr(context, "agent_delegate", None) is not None:
            turn_idx = getattr(context.agent_delegate, "_turn_counter", 0) or 0

        actual_seed = compute_deterministic_seed(
            session_id=session_id,
            turn_idx=turn_idx,
            prompt=params.prompt,
            seed_override=params.seed_override,
        )

        if params.count > 1:
            images: list[dict[str, Any]] = []
            last_engine = ""
            last_device = ""
            batch_id = secrets.token_hex(3)
            for i in range(params.count):
                curr_seed = (actual_seed + i) % (2**32)
                if params.output_path is not None:
                    p = Path(params.output_path)
                    candidate = p.parent / f"{p.stem}_{i + 1}{p.suffix or '.png'}"
                    clean_rel = str(candidate).lstrip("/\\")
                    dest_path = self.resolve_safe_path(clean_rel, context.require_workspace())
                    meta_candidate = p.parent / f"{p.stem}_{i + 1}.json"
                    meta_rel = str(meta_candidate).lstrip("/\\")
                    meta_path = self.resolve_safe_path(meta_rel, context.require_workspace())
                else:
                    default_rel = f"artifacts/images/img_{batch_id}_{i + 1}.png"
                    dest_path = self.resolve_safe_path(default_rel, context.require_workspace())
                    meta_rel = f"artifacts/images/img_{batch_id}_{i + 1}.json"
                    meta_path = self.resolve_safe_path(meta_rel, context.require_workspace())

                gen_result = await self._dispatcher.dispatch(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    aspect_ratio=params.aspect_ratio,
                    seed=curr_seed,
                    style=params.style,
                )
                last_engine = gen_result.engine_name
                last_device = gen_result.device_info

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(gen_result.image_bytes)

                try:
                    rel_path = str(dest_path.relative_to(context.require_workspace().resolve()))
                except ValueError:
                    rel_path = str(dest_path)

                try:
                    rel_meta_path = str(
                        meta_path.relative_to(context.require_workspace().resolve())
                    )
                except ValueError:
                    rel_meta_path = str(meta_path)

                recipe_hash = hashlib.sha256(
                    f"{params.prompt}__{curr_seed}__{gen_result.engine_name}".encode()
                ).hexdigest()[:12]

                meta_data: dict[str, Any] = {
                    "id": f"{batch_id}_{i + 1}",
                    "image_path": rel_path,
                    "prompt": params.prompt,
                    "negative_prompt": params.negative_prompt,
                    "seed": curr_seed,
                    "style": params.style,
                    "aspect_ratio": params.aspect_ratio,
                    "width": gen_result.width,
                    "height": gen_result.height,
                    "engine": gen_result.engine_name,
                    "device": gen_result.device_info,
                    "duration_seconds": gen_result.duration_seconds,
                    "recipe_hash": recipe_hash,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(
                    json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                images.append(
                    {
                        "path": rel_path,
                        "absolute_path": str(dest_path),
                        "meta_path": rel_meta_path,
                        "meta_absolute_path": str(meta_path),
                        "relative_url": f"/api/artifacts/content?path={rel_path}",
                        "seed": curr_seed,
                        "recipe_hash": recipe_hash,
                        "bytes_written": len(gen_result.image_bytes),
                        "width": gen_result.width,
                        "height": gen_result.height,
                        "duration_seconds": gen_result.duration_seconds,
                    }
                )

            gallery_md = "\n".join(
                f"{idx + 1}. ![{params.prompt[:30]} #{idx + 1}]({img['relative_url']})"
                for idx, img in enumerate(images)
            )

            return {
                "status": "success",
                "count": len(images),
                "path": images[0]["path"],
                "paths": [img["path"] for img in images],
                "meta_path": images[0]["meta_path"],
                "meta_paths": [img["meta_path"] for img in images],
                "relative_url": images[0]["relative_url"],
                "relative_urls": [img["relative_url"] for img in images],
                "images": images,
                "markdown_gallery": gallery_md,
                "prompt": params.prompt,
                "style": params.style,
                "aspect_ratio": params.aspect_ratio,
                "engine": last_engine,
                "device": last_device,
            }

        short_id = secrets.token_hex(3)
        # 1. Resolve safe destination path
        if params.output_path is not None:
            clean_rel = params.output_path.lstrip("/\\") or f"artifacts/images/img_{short_id}.png"
            dest_path = self.resolve_safe_path(clean_rel, context.require_workspace())
            p = Path(clean_rel)
            meta_candidate = p.parent / f"{p.stem}.json"
            meta_rel = str(meta_candidate).lstrip("/\\")
            meta_path = self.resolve_safe_path(meta_rel, context.require_workspace())
        else:
            default_rel = f"artifacts/images/img_{short_id}.png"
            dest_path = self.resolve_safe_path(default_rel, context.require_workspace())
            meta_rel = f"artifacts/images/img_{short_id}.json"
            meta_path = self.resolve_safe_path(meta_rel, context.require_workspace())

        # 2. Dispatch generation
        gen_result = await self._dispatcher.dispatch(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            aspect_ratio=params.aspect_ratio,
            seed=actual_seed,
            style=params.style,
        )

        # 3. Write image artifact securely
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(gen_result.image_bytes)

        try:
            rel_path = str(dest_path.relative_to(context.require_workspace().resolve()))
        except ValueError:
            rel_path = str(dest_path)

        try:
            rel_meta_path = str(meta_path.relative_to(context.require_workspace().resolve()))
        except ValueError:
            rel_meta_path = str(meta_path)

        recipe_hash = hashlib.sha256(
            f"{params.prompt}__{actual_seed}__{gen_result.engine_name}".encode()
        ).hexdigest()[:12]

        meta_data = {
            "id": short_id,
            "image_path": rel_path,
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "seed": actual_seed,
            "style": params.style,
            "aspect_ratio": params.aspect_ratio,
            "width": gen_result.width,
            "height": gen_result.height,
            "engine": gen_result.engine_name,
            "device": gen_result.device_info,
            "duration_seconds": gen_result.duration_seconds,
            "recipe_hash": recipe_hash,
            "created_at": datetime.now(UTC).isoformat(),
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "status": "success",
            "path": rel_path,
            "absolute_path": str(dest_path),
            "meta_path": rel_meta_path,
            "meta_absolute_path": str(meta_path),
            "relative_url": f"/api/artifacts/content?path={rel_path}",
            "prompt": params.prompt,
            "style": params.style,
            "aspect_ratio": params.aspect_ratio,
            "width": gen_result.width,
            "height": gen_result.height,
            "seed": actual_seed,
            "recipe_hash": recipe_hash,
            "engine": gen_result.engine_name,
            "device": gen_result.device_info,
            "duration_seconds": gen_result.duration_seconds,
            "bytes_written": len(gen_result.image_bytes),
        }

    async def execute(
        self,
        params: dict[str, Any] | ToolContext | None = None,
        context: ToolContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute tool and decorate result with artifact path provenance."""
        result = await super().execute(params=params, context=context, **kwargs)
        if result.success and isinstance(result.output, dict):
            if "paths" in result.output and isinstance(result.output["paths"], list):
                paths_val = tuple(str(p) for p in result.output["paths"])
                return result.model_copy(update={"artifacts": paths_val})
            if "path" in result.output:
                path_val = str(result.output["path"])
                return result.model_copy(update={"artifacts": (path_val,)})
        return result
