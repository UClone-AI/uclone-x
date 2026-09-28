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
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import LLMError, PlainRefusalError, UCloneXError
from uclone_x.skills.models import SkillStatus
from uclone_x.skills.protocols import SkillRegistryProtocol
from uclone_x.tools.base import BaseTool, artifact_content_url, replace_file
from uclone_x.tools.builtin.comfy_client import (
    DEFAULT_COMFYUI_BASE_URL,
    ComfyClient,
    default_comfy_checkpoint,
)
from uclone_x.tools.builtin.comfy_image_tool import build_txt2img_workflow
from uclone_x.tools.builtin.image_set_intent import asks_for_variety
from uclone_x.tools.builtin.media_registry import (
    ModelProfile,
    ModelRegistry,
    PromptFamily,
    fill_prompt_defaults,
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
#: Denoising steps and guidance for the in-process engine when no registered model profile
#: is active (the generic fallback). With a profile, its own `steps` and `cfg` are used —
#: `default_models.yaml` gives Illustrious 30 steps at cfg 5.5, and these two constants
#: used to override that for every checkpoint.
DIFFUSERS_STEPS = 20
DIFFUSERS_GUIDANCE = 7.0
#: Text the CLIP encoders read per 77-token window: 75 prompt tokens between the
#: begin and end markers. Anything past it used to be dropped silently; longer prompts are
#: now encoded in windows of this size and joined (see `long_prompt_embeds`).
CLIP_CHUNK_TOKENS = 75
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


def style_guided_prompt(prompt: str, style: str, family: PromptFamily | None = None) -> str:
    """The prompt with its style preset appended, or unchanged for an unknown preset.

    A `danbooru` prompt is left unchanged whatever the preset: it is a tag list whose
    style tags the domain skill already supplied, and a prose suffix such as
    `photorealistic, sharp focus, natural lighting` contradicts them (design §3.6).
    """
    if family is PromptFamily.DANBOORU:
        return prompt
    suffix = STYLE_SUFFIXES.get(style)
    return f"{prompt}, {suffix}" if suffix else prompt


class ImageGenerationError(UCloneXError):
    """Raised when image generation fails due to engine, memory, or network errors."""


class NoImageEngineError(ImageGenerationError):
    """No engine the dispatcher tried could draw. The message is the engine-by-engine account."""


#: What a conversation shows when no engine can draw: plain words and what to do about it.
#: The engine-by-engine account goes to the log and `ucx media status`, not to the model,
#: which would otherwise hand it to the person as the reason.
NO_IMAGE_ENGINE_TEXT = (
    "No image model is connected, so I can't draw right now. To fix this, connect one in "
    "Settings › Images, or upload a picture instead."
)


#: Generic quality defense tags that should not trigger false-positive conflict errors when
#: appearing in negative prompts alongside photographic or artistic positive descriptors.
GENERIC_NEGATIVE_BOILERPLATE = frozenset(
    {
        "worst quality",
        "low quality",
        "normal quality",
        "bad quality",
        "poor quality",
        "bad anatomy",
        "bad hands",
        "bad feet",
        "bad proportions",
        "blurry",
        "blur",
        "deformed",
        "disfigured",
        "mutated",
        "extra limbs",
        "missing limbs",
        "extra fingers",
        "fewer fingers",
        "missing fingers",
        "extra digits",
        "fewer digits",
        "cropped",
        "jpeg artifacts",
        "watermark",
        "signature",
        "username",
        "artist name",
        "text",
        "error",
    }
)


def find_prompt_conflicts(prompt: str, negative_prompt: str) -> list[str]:
    """Find exclusion terms from negative_prompt that contradictorily appear in prompt.

    Ignores generic quality boilerplate tags (e.g. blurry, worst quality, bad hands).
    Cleans weights and formatting brackets before checking word boundaries.
    """
    if not prompt or not negative_prompt:
        return []

    raw_tags = re.split(r"[,;\n]", negative_prompt)
    conflicts: list[str] = []
    seen: set[str] = set()
    prompt_lower = prompt.lower()

    for raw in raw_tags:
        clean = re.sub(r"[:\d\.]+$", "", raw.strip(" ()[]{}\"'\t")).strip().lower()
        if len(clean) < 2 or clean in GENERIC_NEGATIVE_BOILERPLATE or clean in seen:
            continue

        pattern = r"(?<![\w-])" + re.escape(clean) + r"(?![\w-])"
        if re.search(pattern, prompt_lower):
            conflicts.append(clean)
            seen.add(clean)

    return conflicts


COUNT_FOR_VARIETY_REFUSAL = (
    "The request asks for images that differ from one another, but 'count' only renders "
    "the same prompt again with a new seed, so every image would show the same scene. "
    "Write one distinct prompt per image and pass them in 'prompts' instead, for example "
    "generate_image(prompts=['first scene ...', 'second scene ...'])."
)


def asks_for_varied_images(request: str) -> bool:
    """Whether the person asked for images that differ -- the reading the planner uses."""
    return asks_for_variety(request)


def latest_user_request(context: ToolContext) -> str | None:
    """The text of the newest user message in the calling agent's history, if it has one.

    Read through the agent's public `history`; None when the tool runs without an agent
    (a direct call) or the history holds no user message.
    """
    agent: Any = getattr(context, "agent_delegate", None)
    if agent is None:
        return None
    try:
        history: Any = agent.history
        messages = tuple(history)
    except Exception:
        return None
    for message in reversed(messages):
        role: Any = getattr(message, "role", None)
        if str(getattr(role, "value", role)) == "user":
            content = getattr(message, "content", None)
            return content if isinstance(content, str) else None
    return None


class GenerateImageParams(BaseModel):
    """Parameters for generating an image via the default image pipeline."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(
        default="",
        description=(
            "Positive text prompt describing the visual composition of the image to generate. "
            "Required unless 'prompts' is provided for diverse multi-image generation."
        ),
    )
    prompts: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of distinct prompts (1-10) for diverse multi-image generation "
            "(e.g. multiple distinct poses, scenes, or camera angles for the same character). "
            "When provided, an image is generated for each prompt, and 'count' is ignored."
        ),
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
        description="Optional relative file path within workspace. Defaults to 'artifacts/images/img_{id}.png'; the suffix follows the format drawn (.jpg for a JPEG).",
    )
    count: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "How many times to render the SAME 'prompt' (1-10), each with a different seed: "
            "variations of one scene. For different scenes, poses, characters, expressions "
            "or outfits, do not use count; pass one distinct prompt per image in 'prompts'."
        ),
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
    mime_type: str = "image/png"
    #: What the active model family's defaults added to the prompt, or left out of it,
    #: one plain sentence each (`fill_prompt_defaults`). Set by the dispatcher.
    prompt_changes: tuple[str, ...] = ()


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
        *,
        steps: int | None = None,
        cfg: float | None = None,
        family: PromptFamily | None = None,
    ) -> ImageGenerationResult:
        """Generate an image returning raw image bytes and execution metadata.

        `steps` and `cfg` come from the active model profile; None leaves each engine on
        its own default, which is what the generic fallback profile gets. `family` is the
        profile's prompt family, which decides whether the style suffix is appended
        (`style_guided_prompt`).
        """


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
        *,
        steps: int | None = None,
        cfg: float | None = None,
        family: PromptFamily | None = None,
    ) -> ImageGenerationResult:
        """Dispatch generation request to remote CUDA worker."""
        if not self._base_url:
            raise ImageGenerationError("Remote CUDA engine URL is not configured.")

        # `family` is not sent: the worker's payload has no such field, so a danbooru
        # prompt there still gets whatever the worker does with `style`.
        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "style": style,
        }
        # Sent only when a profile set them, so a worker that predates these fields gets
        # the exact payload it always did for the fallback profile.
        if steps is not None:
            payload["steps"] = steps
        if cfg is not None:
            payload["cfg"] = cfg

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


def clip_token_ids(tokenizer: Any, text: str) -> list[int]:
    """The CLIP token ids of `text`: no begin/end markers, no padding, no truncation."""
    if not text:
        return []
    encoded: Any = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)
    return [int(token) for token in encoded.input_ids]


def clip_windows(ids: list[int], tokenizer: Any, count: int) -> list[list[int]]:
    """`ids` cut into `count` windows of `CLIP_CHUNK_TOKENS`, each framed as CLIP expects.

    Every window is `[begin] + up to 75 tokens + [end]`, padded to 77 with the tokenizer's
    own pad token — the layout diffusers' `encode_prompt` builds for a single window.
    Windows past the end of `ids` are empty (`[begin, end, pad...]`), which is how a
    shorter prompt is brought to the same length as its longer counterpart.
    """
    begin, end = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end
    width = CLIP_CHUNK_TOKENS + 2
    windows: list[list[int]] = []
    for index in range(count):
        chunk = ids[index * CLIP_CHUNK_TOKENS : (index + 1) * CLIP_CHUNK_TOKENS]
        window = [begin, *chunk, end]
        windows.append(window + [pad] * (width - len(window)))
    return windows


def encode_clip_windows(
    torch: Any, encoder: Any, windows: list[list[int]], device: Any
) -> tuple[Any, Any]:
    """Each window's penultimate hidden state, joined along the sequence, and window 1's `[0]`.

    The penultimate layer is what SDXL conditions on (diffusers' `hidden_states[-2]` with
    no clip skip). Output `[0]` is the pooled, projected embedding on `text_encoder_2`; on
    `text_encoder` it is unused.
    """
    hidden: list[Any] = []
    first: Any = None
    for window in windows:
        output: Any = encoder(torch.tensor([window], device=device), output_hidden_states=True)
        if first is None:
            first = output[0]
        hidden.append(output.hidden_states[-2])
    return torch.cat(hidden, dim=1), first


def long_prompt_embeds(
    pipeline: Any, torch: Any, prompt: str, negative_prompt: str
) -> dict[str, Any] | None:
    """SDXL embeddings for a prompt longer than one CLIP window, or None when it fits.

    The pipeline's own encoder truncates at 77 tokens, so everything past the 75th prompt
    token was dropped without a word. Here both prompts are tokenized by both tokenizers,
    cut into 75-token windows, each window encoded by its text encoder, and the windows
    joined along the sequence axis; the negative prompt is given as many windows as the
    positive so the two can be batched for guidance. The pooled embedding is
    `text_encoder_2`'s on the first window. Weight syntax and BREAK are not parsed.

    None — the plain `prompt=` path, unchanged — when both prompts fit in one window, or
    when the pipeline lacks either tokenizer or encoder.
    """
    tokenizers = (getattr(pipeline, "tokenizer", None), getattr(pipeline, "tokenizer_2", None))
    encoders = (getattr(pipeline, "text_encoder", None), getattr(pipeline, "text_encoder_2", None))
    if any(part is None for part in (*tokenizers, *encoders)):
        return None
    ids = [
        (clip_token_ids(tok, prompt), clip_token_ids(tok, negative_prompt)) for tok in tokenizers
    ]
    longest = max(len(sequence) for pair in ids for sequence in pair)
    if longest <= CLIP_CHUNK_TOKENS:
        return None
    count = -(-longest // CLIP_CHUNK_TOKENS)
    device: Any = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", "cpu")
    config: Any = getattr(pipeline, "config", None)
    zero_negative = not negative_prompt and bool(
        getattr(config, "force_zeros_for_empty_prompt", False)
    )
    with torch.no_grad():
        positive: list[Any] = []
        negative: list[Any] = []
        pooled: Any = None
        negative_pooled: Any = None
        for tokenizer, encoder, (prompt_ids, negative_ids) in zip(
            tokenizers, encoders, ids, strict=True
        ):
            embeds, first = encode_clip_windows(
                torch, encoder, clip_windows(prompt_ids, tokenizer, count), device
            )
            positive.append(embeds)
            pooled = first  # the last encoder's is kept: text_encoder_2's
            if not zero_negative:
                embeds, first = encode_clip_windows(
                    torch, encoder, clip_windows(negative_ids, tokenizer, count), device
                )
                negative.append(embeds)
                negative_pooled = first
        prompt_embeds = torch.cat(positive, dim=-1)
        if zero_negative:
            # What diffusers does for an empty negative when the checkpoint's config asks.
            negative_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled = torch.zeros_like(pooled)
        else:
            negative_embeds = torch.cat(negative, dim=-1)
    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_embeds,
        "pooled_prompt_embeds": pooled,
        "negative_pooled_prompt_embeds": negative_pooled,
    }


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
        steps: int | None = None,
        cfg: float | None = None,
        family: PromptFamily | None = None,
    ) -> bytes:
        """Run the diffusion loop on a worker thread and return PNG bytes."""
        import io

        pipeline: Any = self._ensure_pipeline_loaded()
        torch: Any = None
        try:
            import importlib

            torch = importlib.import_module("torch")
            generator: Any = torch.Generator(device="cpu").manual_seed(seed)
            guided = style_guided_prompt(prompt, style, family)
            # Past 75 CLIP tokens the pipeline's own encoder truncates; those prompts are
            # encoded here in windows instead. Short ones keep the plain arguments.
            embeds = long_prompt_embeds(pipeline, torch, guided, negative_prompt)
            text_args: dict[str, Any] = (
                embeds
                if embeds is not None
                else {"prompt": guided, "negative_prompt": negative_prompt or None}
            )
            output: Any = pipeline(
                **text_args,
                width=width,
                height=height,
                num_inference_steps=steps if steps is not None else DIFFUSERS_STEPS,
                guidance_scale=cfg if cfg is not None else DIFFUSERS_GUIDANCE,
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
        *,
        steps: int | None = None,
        cfg: float | None = None,
        family: PromptFamily | None = None,
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
            steps=steps,
            cfg=cfg,
            family=family,
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
        *,
        steps: int | None = None,
        cfg: float | None = None,
        family: PromptFamily | None = None,
    ) -> ImageGenerationResult:
        """Queue a txt2img graph on the running daemon and fetch the produced PNG."""
        client = ComfyClient(base_url=self._base_url)
        start_t = time.monotonic()
        # None keeps the workflow's own defaults, as before profiles reached here.
        sampling: dict[str, Any] = {}
        if steps is not None:
            sampling["steps"] = steps
        if cfg is not None:
            sampling["cfg"] = cfg
        try:
            workflow = build_txt2img_workflow(
                prompt=style_guided_prompt(prompt, style, family),
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
                checkpoint=self._checkpoint,
                **sampling,
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


#: The settings.json fields choosing who draws a picture (`image_engine`) and, when Google
#: Gemini does, which of its image models (`image_model`). The shell reads them and hands
#: the dispatcher an `ImageEngineChoice`; this module never reads the file itself.
IMAGE_ENGINE_KEY = "image_engine"
IMAGE_MODEL_KEY = "image_model"
#: `auto` uses a ready local engine, and Gemini only when none is ready and Gemini is the
#: chat provider with a key; `local` never sends a picture request over the internet;
#: `gemini` draws with Gemini only.
ImageEngineSetting = Literal["auto", "local", "gemini"]
IMAGE_ENGINE_SETTINGS: tuple[ImageEngineSetting, ...] = ("auto", "local", "gemini")
DEFAULT_IMAGE_ENGINE: ImageEngineSetting = "auto"
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"
#: The engine name results and sidecars carry for a picture Gemini drew.
GEMINI_ENGINE_NAME = "gemini"
#: A model id is interpolated into a URL path, so it is held to the characters ids use.
_IMAGE_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
#: How long one look at the local engines stands for, under `auto` with Gemini as the
#: fallback. The tool description and the next draw read the same look, so the engine
#: the description names is the one that draws.
LOCAL_PROBE_TTL_SECONDS = 30.0
NO_GEMINI_KEY_MESSAGE = (
    "Pictures are set to be drawn by Google Gemini, but no Gemini API key is saved. "
    "Add a Gemini key in Settings, or set the picture engine back to automatic."
)


def parse_image_engine_setting(value: object) -> ImageEngineSetting:
    """`value` as an `image_engine` setting; absent means `auto`, anything unknown is refused."""
    if value is None:
        return DEFAULT_IMAGE_ENGINE
    for setting in IMAGE_ENGINE_SETTINGS:
        if value == setting:
            return setting
    raise PlainRefusalError(
        f"The picture engine must be auto, local or gemini; {value!r} is not one of them."
    )


def parse_image_model(value: object) -> str:
    """`value` as an `image_model` setting; absent means `DEFAULT_IMAGE_MODEL`."""
    if value is None:
        return DEFAULT_IMAGE_MODEL
    if isinstance(value, str) and _IMAGE_MODEL_ID.fullmatch(value):
        return value
    raise PlainRefusalError(
        f"The picture model must be a Gemini model id such as {DEFAULT_IMAGE_MODEL}; "
        f"{value!r} is not one."
    )


class GeminiImageClient(Protocol):
    """What `GeminiImageEngine` needs from the Gemini connector, which it cannot import."""

    async def generate_image(self, prompt: str, aspect_ratio: str, model: str) -> tuple[bytes, str]:
        """One picture as ``(bytes, MIME type)``, or a `ProviderFailureError`."""
        ...


@dataclass(frozen=True)
class ImageEngineChoice:
    """The picture settings in effect, as the shell read them.

    ``gemini`` is a client only when a Gemini key is saved or set; ``chat_provider`` is the
    canonical id of the chat provider in effect. The default is the choice of a head that
    binds nothing: `auto` with no Gemini client, which never leaves the local engines.
    """

    setting: ImageEngineSetting = DEFAULT_IMAGE_ENGINE
    model: str = DEFAULT_IMAGE_MODEL
    chat_provider: str | None = None
    gemini: GeminiImageClient | None = None

    @property
    def gemini_in_auto(self) -> bool:
        """Whether `auto` may fall back to Gemini: Gemini chat, and a key to draw with."""
        return self.chat_provider == GEMINI_ENGINE_NAME and self.gemini is not None


class ImageEngineRefusal(ImageGenerationError, PlainRefusalError):
    """A picture an engine could not draw, told in words already written for a person."""


def png_dimensions(data: bytes) -> tuple[int, int] | None:
    """Width and height from a PNG's IHDR chunk, or ``None`` for bytes that are no PNG."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Width and height from a JPEG's SOF marker, or ``None`` for bytes that are no JPEG."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    idx = 2
    while idx < len(data) - 8:
        if data[idx] != 0xFF:
            return None
        while idx < len(data) and data[idx] == 0xFF:
            idx += 1
        if idx >= len(data):
            return None
        marker = data[idx]
        idx += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if idx + 2 > len(data):
            return None
        seg_len = int.from_bytes(data[idx : idx + 2], "big")
        # SOF markers: C0, C1, C2, C3, C5, C6, C7, C9, CA, CB, CD, CE, CF
        if marker in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            if idx + 2 + 5 <= len(data):
                height = int.from_bytes(data[idx + 3 : idx + 5], "big")
                width = int.from_bytes(data[idx + 5 : idx + 7], "big")
                return width, height
        idx += seg_len
    return None


def webp_dimensions(data: bytes) -> tuple[int, int] | None:
    """Width and height from a WebP's first chunk (VP8, VP8L or VP8X), or ``None`` otherwise."""
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        return width, 1 + int.from_bytes(data[27:30], "little")
    if chunk == b"VP8L" and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        return width, int.from_bytes(data[28:30], "little") & 0x3FFF
    return None


#: The picture formats a drawn picture is saved as: MIME type, the suffix it is written
#: under, every suffix that already names it, and the reader that finds its size.
_SAVED_PICTURE_FORMATS: tuple[
    tuple[str, str, frozenset[str], Callable[[bytes], tuple[int, int] | None]], ...
] = (
    ("image/png", ".png", frozenset({".png"}), png_dimensions),
    ("image/jpeg", ".jpg", frozenset({".jpg", ".jpeg"}), jpeg_dimensions),
    ("image/webp", ".webp", frozenset({".webp"}), webp_dimensions),
)
_PICTURE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})


def sniff_picture(data: bytes) -> tuple[str, tuple[int, int]] | None:
    """The MIME type and size ``data`` is by its own bytes, or ``None`` for no saved format.

    Decided by content, never by the MIME type a reply claims: the file is named after
    what this returns, so a label that disagrees with the bytes cannot make it lie.
    """
    for mime, _suffix, _names, reader in _SAVED_PICTURE_FORMATS:
        size = reader(data)
        if size is not None and size[0] > 0 and size[1] > 0:
            return mime, size
    return None


def image_extension_for(mime: str) -> str:
    """File extension (with dot) for a picture MIME type; PNG, which local engines write, otherwise."""
    if mime == "image/jpg":
        mime = "image/jpeg"
    for known, suffix, _names, _reader in _SAVED_PICTURE_FORMATS:
        if known == mime:
            return suffix
    return ".png"


def picture_path_for(rel: str, mime: str) -> str:
    """``rel`` under a suffix naming the format written, so a file never lies about what it is.

    A suffix that already names the format (``.jpeg`` for a JPEG) is kept; another picture
    suffix is replaced (``face.png`` holding a JPEG becomes ``face.jpg``); anything else
    has the suffix added.
    """
    suffix = image_extension_for(mime)
    names = next(
        (n for known, _s, n, _r in _SAVED_PICTURE_FORMATS if known == mime), frozenset({suffix})
    )
    path = Path(rel)
    if path.suffix.lower() in names:
        return rel
    if path.suffix.lower() in _PICTURE_SUFFIXES:
        return str(path.with_suffix(suffix))
    return f"{rel}{suffix}"


def sidecar_path_for(picture_rel: str) -> str:
    """The recipe `.json` saved beside ``picture_rel``, named after the picture's final name."""
    path = Path(picture_rel)
    return str(path.parent / f"{path.stem}.json")


#: Last path parts that name no picture file: a folder, or a bare suffix like `.png`.
_NO_FILE_NAMES = _PICTURE_SUFFIXES | {"", ".", ".."}


def names_no_file(rel: str) -> bool:
    """Whether ``rel`` names a folder or a bare suffix rather than a picture file.

    An empty ``rel`` (or one that is only slashes) is not refused: it falls back to the
    default name, as it always has.
    """
    if not rel.strip("/\\"):
        return False
    return rel.endswith(("/", "\\")) or Path(rel).name.lower() in _NO_FILE_NAMES


def aspect_ratio_for(width: int, height: int) -> str:
    """The supported ratio string closest to ``width`` x ``height``."""
    target = width / height if height else 1.0
    ratios = {"1:1": 1.0, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4}
    return min(ratios, key=lambda name: abs(ratios[name] - target))


def gemini_profile(model: str) -> ModelProfile:
    """The model profile a Gemini image model is prompted by: prose, no negative prompt."""
    return ModelProfile(
        model_id=model,
        display_name=f"Google Gemini ({model})",
        family=PromptFamily.NATURAL_PROSE,
        suppress_negative=True,
    )


class GeminiImageEngine(BaseImageEngine):
    """Draws with a Google Gemini image model, over the internet, with the saved key.

    Gemini takes no seed, steps or guidance: the seed is recorded in the result as the
    other engines record theirs, but it does not make a Gemini picture reproducible.
    """

    def __init__(self, client: GeminiImageClient | None, model: str = DEFAULT_IMAGE_MODEL) -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        """The Gemini image model this engine asks."""
        return self._model

    async def is_available(self) -> bool:
        """Whether a Gemini key is there to draw with. Nothing is sent to find out."""
        return self._client is not None

    async def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        style: str,
        *,
        steps: int | None = None,
        cfg: float | None = None,
        family: PromptFamily | None = None,
    ) -> ImageGenerationResult:
        """`draw` at the ratio closest to ``width`` x ``height``; the negative prompt is unused."""
        return await self.draw(prompt, aspect_ratio_for(width, height), seed)

    async def draw(self, prompt: str, aspect_ratio: str, seed: int) -> ImageGenerationResult:
        """One picture at ``aspect_ratio``, or an `ImageEngineRefusal` saying why not."""
        if self._client is None:
            raise ImageEngineRefusal(NO_GEMINI_KEY_MESSAGE)
        start_t = time.monotonic()
        try:
            data, mime = await self._client.generate_image(
                prompt=prompt, aspect_ratio=aspect_ratio, model=self._model
            )
        except LLMError as exc:
            raise ImageEngineRefusal(str(exc)) from exc
        picture = sniff_picture(data)
        if picture is None:
            # Only PNG, JPEG and WebP are saved; any other bytes would be a file that no
            # picture viewer here opens.
            logger.info(
                "Gemini image model %s returned %s bytes that are no PNG, JPEG or WebP",
                self._model,
                mime,
            )
            raise ImageEngineRefusal(
                "Google Gemini sent back a picture in a format that cannot be saved here. "
                "Try again, or choose another picture model."
            )
        real_mime, size = picture
        if real_mime != mime:
            logger.info(
                "Gemini image model %s labelled a %s picture %s", self._model, real_mime, mime
            )
        return ImageGenerationResult(
            image_bytes=data,
            seed=seed,
            engine_name=GEMINI_ENGINE_NAME,
            device_info=f"Google Gemini API ({self._model})",
            duration_seconds=round(time.monotonic() - start_t, 3),
            width=size[0],
            height=size[1],
            mime_type=real_mime,
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


#: SDXL's ~1-megapixel training buckets, per aspect ratio. Illustrious excluded images under
#: 768x768 from training, and the community renders it at these sizes; the sub-megapixel
#: table below rendered it below the size it was trained at. 3:4 is 896x1152, the
#: bucket, rather than the profile's own 832x1216, which is 2:3: one rule for every ratio.
SDXL_MEGAPIXEL_BUCKETS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "16:9": (1344, 768),
}
#: The sub-megapixel sizes kept for the generic fallback and for small-canvas profiles.
LEGACY_ASPECT_DIMENSIONS: dict[str, tuple[int, int]] = {
    "1:1": (768, 768),
    "3:4": (576, 768),
    "4:3": (768, 576),
    "9:16": (512, 896),
    "16:9": (896, 512),
}
#: The model id `ModelRegistry` gives an unrecognised checkpoint. Its settings are a guess,
#: so it keeps the legacy sizes and the engines' own step and guidance defaults.
GENERIC_FALLBACK_MODEL_ID = "generic_fallback"
#: A profile whose native canvas is at least this many pixels is a megapixel-class model
#: (SDXL, Illustrious, FLUX) and is rendered at `SDXL_MEGAPIXEL_BUCKETS`. 90% of 1024x1024
#: admits 832x1216 (1,011,712 px) and excludes an SD 1.5-style 512x768 profile.
MEGAPIXEL_PROFILE_MIN_PIXELS = int(0.9 * 1024 * 1024)


def is_registered_profile(profile: ModelProfile | None) -> bool:
    """Whether `profile` describes a known checkpoint rather than the generic fallback."""
    return profile is not None and profile.model_id != GENERIC_FALLBACK_MODEL_ID


def resolve_aspect_dimensions(
    aspect_ratio: str,
    profile: ModelProfile | None = None,
) -> tuple[int, int]:
    """Pixel dimensions for an aspect ratio, all multiples of 64.

    A registered profile with a megapixel-class native canvas gets SDXL's ~1MP buckets;
    everything else — no profile, the generic fallback, a small-canvas profile — keeps the
    legacy sub-megapixel sizes. An unknown ratio reads as 1:1 in both tables.
    """
    megapixel = (
        is_registered_profile(profile)
        and profile is not None
        and profile.width * profile.height >= MEGAPIXEL_PROFILE_MIN_PIXELS
    )
    table = SDXL_MEGAPIXEL_BUCKETS if megapixel else LEGACY_ASPECT_DIMENSIONS
    return table.get(aspect_ratio, table["1:1"])


def resolve_sampling(profile: ModelProfile | None) -> tuple[int | None, float | None]:
    """The steps and guidance a registered profile asks for, or (None, None) for none.

    None means "the engine's own default": `DIFFUSERS_STEPS`/`DIFFUSERS_GUIDANCE` in
    process, the workflow's defaults on ComfyUI, and nothing sent to a remote worker.
    """
    if profile is None or not is_registered_profile(profile):
        return None, None
    return profile.steps, profile.cfg


def _log_failed_refresh(task: asyncio.Task[bool]) -> None:
    """Log a background look at the local engines that raised, so it is not lost."""
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.warning("Could not look at the local image engines: %s", error)


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
        engine_settings: Callable[[], ImageEngineChoice] | None = None,
    ) -> None:
        self._remote_engine = remote_engine or RemoteCudaImageEngine()
        self._comfy_engine = comfy_engine or ComfyUIImageEngine()
        self._local_engine = local_engine or LocalDiffusersImageEngine()
        self._registry = registry or ModelRegistry()
        self._engine_settings: Callable[[], ImageEngineChoice] = (
            engine_settings or ImageEngineChoice
        )
        #: When the local engines were last looked at, and whether any was ready.
        self._local_probe: tuple[float, bool] | None = None
        #: The look a stale description started on the running loop, while it runs.
        self._local_refresh: asyncio.Task[bool] | None = None

    @property
    def registry(self) -> ModelRegistry:
        """The model registry managing model profiles and prompt families."""
        return self._registry

    def bind_engine_settings(self, source: Callable[[], ImageEngineChoice]) -> None:
        """Read the picture settings from ``source`` on every draw and description.

        Called per read rather than once, so a setting changed in Settings applies to the
        next picture without rebuilding the tool.
        """
        self._engine_settings = source
        self._local_probe = None

    def engine_choice(self) -> ImageEngineChoice:
        """The picture settings in effect now."""
        return self._engine_settings()

    def _local_engines(self) -> list[tuple[str, BaseImageEngine]]:
        """The local engines, in the order they are tried."""
        return [
            ("Remote CUDA worker", self._remote_engine),
            ("detected local ComfyUI daemon", self._comfy_engine),
            ("in-process diffusers engine", self._local_engine),
        ]

    async def _probe_local_engines(self) -> bool:
        for _label, engine in self._local_engines():
            if await engine.is_available():
                return True
        return False

    def _fresh_local_probe(self) -> bool | None:
        probe = self._local_probe
        if probe is not None and time.monotonic() - probe[0] < LOCAL_PROBE_TTL_SECONDS:
            return probe[1]
        return None

    async def _look_at_local_engines(self) -> bool:
        ready = await self._probe_local_engines()
        self._local_probe = (time.monotonic(), ready)
        return ready

    async def _any_local_ready(self) -> bool:
        cached = self._fresh_local_probe()
        if cached is not None:
            return cached
        refresh = self._local_refresh
        if (
            refresh is not None
            and not refresh.done()
            and refresh.get_loop() is asyncio.get_running_loop()
        ):
            return await refresh
        return await self._look_at_local_engines()

    def _any_local_ready_without_blocking(self) -> bool:
        """`_any_local_ready` for synchronous code, never stalling a running event loop.

        With no loop running, the engines are looked at here. Under a running loop (the
        tool description is read inside a turn) only the very first look waits; after that
        a stale answer is served while one refresh runs on the loop, so a probe that takes
        its full timeout never holds a turn up (#1769). `dispatch` looks again itself.
        """
        cached = self._fresh_local_probe()
        if cached is not None:
            return cached
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            ready = asyncio.run(self._probe_local_engines())
            self._local_probe = (time.monotonic(), ready)
            return ready
        last = self._local_probe
        if last is None:
            with ThreadPoolExecutor(max_workers=1) as pool:
                ready = pool.submit(asyncio.run, self._probe_local_engines()).result()
            self._local_probe = (time.monotonic(), ready)
            return ready
        if self._local_refresh is None or self._local_refresh.done():
            self._local_refresh = loop.create_task(self._look_at_local_engines())
            self._local_refresh.add_done_callback(_log_failed_refresh)
        return last[1]

    def _setting_decides(self, choice: ImageEngineChoice) -> bool | None:
        """Whether Gemini draws, when the setting alone says; `None` when `auto` must look.

        `local` never does; `gemini` always does; `auto` does only when Gemini may be the
        fallback and no local engine is ready, which is looked at only in that case.
        """
        if choice.setting != "auto":
            return choice.setting == GEMINI_ENGINE_NAME
        if not choice.gemini_in_auto:
            return False
        return None

    def draws_with_gemini(self, choice: ImageEngineChoice | None = None) -> bool:
        """Whether the next picture goes to Gemini under ``choice`` (the settings by default)."""
        choice = choice if choice is not None else self.engine_choice()
        decided = self._setting_decides(choice)
        if decided is not None:
            return decided
        return not self._any_local_ready_without_blocking()

    async def _draws_with_gemini_now(self, choice: ImageEngineChoice) -> bool:
        decided = self._setting_decides(choice)
        if decided is not None:
            return decided
        return not await self._any_local_ready()

    def get_active_profile(self) -> ModelProfile:
        """The profile of the engine that draws next: Gemini's, or the local checkpoint's."""
        choice = self.engine_choice()
        if self.draws_with_gemini(choice):
            return gemini_profile(choice.model)
        return self._local_profile()

    def _local_profile(self) -> ModelProfile:
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
        """Route to the engine the picture settings name (`draws_with_gemini`).

        Otherwise the highest priority available local-private engine.
        """
        choice = self.engine_choice()
        if await self._draws_with_gemini_now(choice):
            gemini = GeminiImageEngine(choice.gemini, choice.model)
            family = PromptFamily.NATURAL_PROSE
            gemini_fill = fill_prompt_defaults(
                prompt, negative_prompt, gemini_profile(choice.model)
            )
            logger.info("Dispatching image generation to Google Gemini (%s)...", choice.model)
            drawn = await gemini.draw(
                style_guided_prompt(gemini_fill.prompt, style, family), aspect_ratio, seed
            )
            return replace(drawn, prompt_changes=gemini_fill.changes)
        profile = self.get_active_profile()
        fill = fill_prompt_defaults(prompt, negative_prompt, profile)
        width, height = resolve_aspect_dimensions(aspect_ratio, profile=profile)
        steps, cfg = resolve_sampling(profile)
        attempts = self._local_engines()

        for label, engine in attempts:
            if await engine.is_available():
                logger.info("Dispatching image generation to %s...", label)
                generated = await engine.generate(
                    prompt=fill.prompt,
                    negative_prompt=fill.negative_prompt,
                    width=width,
                    height=height,
                    seed=seed,
                    style=style,
                    steps=steps,
                    cfg=cfg,
                    family=profile.family,
                )
                return replace(generated, prompt_changes=fill.changes)

        raise NoImageEngineError(
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
        parts.append(self._gemini_diagnostic())
        return " ".join(parts)

    def _gemini_diagnostic(self) -> str:
        """Why Google Gemini did not draw, in the terms of the `image_engine` rule."""
        try:
            choice = self.engine_choice()
        except UCloneXError as exc:
            return str(exc)
        if choice.setting == "local":
            return "Google Gemini is not used, because image_engine is set to local."
        if choice.gemini is None:
            return "Google Gemini can draw only with a Gemini API key saved in Settings."
        if choice.setting == "auto" and not choice.gemini_in_auto:
            return (
                "With image_engine set to auto, Google Gemini draws only while Gemini is "
                "the chat provider."
            )
        return f"Google Gemini ({choice.model}) is ready."


class GenerateImageTool(BaseTool[GenerateImageParams]):
    """Agent tool to generate visual illustrations, diagrams, and photos locally.

    `description` names no model and no prompt family, so switching the checkpoint
    mid-session leaves the tools layer of the request as it was. The active family's rules
    reach the prompt in two other ways: `load_skill` returns the section for that family,
    and `fill_prompt_defaults` fills in, at call time, what the prompt left open, which
    the result reports under `prompt_changes` (owner ruling on #1723).
    """

    name = "generate_image"
    writes_files: ClassVar[bool] = True  # can create, modify or delete a file on the host (#1167)
    BASE_DESCRIPTION: ClassVar[str] = (
        "Generate a local-private image, illustration, or diagram from a text prompt. "
        "Runs in this process from a local checkpoint, on a detected local ComfyUI daemon, or on "
        "a local-network CUDA GPU worker — never on a paid cloud service. "
        "Saves the resulting image as an artifact in the workspace. "
        "Write the prompt in English, whatever language the person wrote in. "
        "What the prompt leaves out that the image model needs, such as quality tags or a "
        "negative prompt, is filled in, and the result lists it under prompt_changes. "
        "To present the result to the user, include standard markdown ![description](relative_url) or [title](relative_url)."
    )
    #: The base text while Google Gemini draws, which the local text would misstate.
    GEMINI_BASE_DESCRIPTION: ClassVar[str] = (
        "Generate an image, illustration, or diagram from a text prompt. "
        "Pictures are drawn by a Google Gemini image model over the internet with the "
        "Gemini API key saved in Settings, so the prompt is sent to Google. "
        "Saves the resulting image as an artifact in the workspace. "
        "Anything left out of the request, such as a negative prompt this model cannot use, is "
        "listed in the result under prompt_changes. "
        "To present the result to the user, include standard markdown ![description](relative_url) or [title](relative_url)."
    )
    params_type = GenerateImageParams

    def __init__(
        self,
        dispatcher: ImagePipelineDispatcher | None = None,
    ) -> None:
        super().__init__(name=self.name, params_type=GenerateImageParams)
        self._dispatcher = dispatcher or ImagePipelineDispatcher()
        self._skills: SkillRegistryProtocol | None = None

    def active_profile(self) -> ModelProfile:
        """The profile of the checkpoint a generation would use now (`ImageModelSource`)."""
        return self._dispatcher.get_active_profile()

    def bind_skill_registry(self, registry: SkillRegistryProtocol) -> None:
        """Set the skill store the description lists image domains from (`ImageModelSource`).

        The tool sits in a registry every agent of the process shares, so the last agent
        to bind wins. Every head passes the same runtime store, so the domain list does
        not depend on which agent that was.
        """
        self._skills = registry

    def bind_engine_settings(self, source: Callable[[], ImageEngineChoice]) -> None:
        """Read the picture settings (`image_engine`, `image_model`) from ``source``."""
        self._dispatcher.bind_engine_settings(source)

    def domain_skill_names(self) -> list[str]:
        """Names of the active skills declaring `family_sections: true`, sorted."""
        if self._skills is None:
            return []
        return sorted(
            skill.manifest.name
            for skill in self._skills.list_skills()
            if skill.manifest.family_sections and skill.manifest.status == SkillStatus.ACTIVE
        )

    # A read-only property where `BaseTool` declares a plain attribute: readers only read
    # it (`ToolProtocol.description` is a property), and nothing assigns this one, since
    # `__init__` passes no description to `BaseTool`.
    @property
    def description(self) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        """The base text for the engine that draws, plus the image domains.

        It never names the active model or its prompt family (owner ruling on #1723):
        it changes only with the picture settings and the skill store. A failure to read
        them is logged and the base text returned: it must not take the whole tool
        listing down with it, and `generate_image` itself reports that failure when it runs.
        """
        try:
            gemini = self._dispatcher.draws_with_gemini()
            domains = self.domain_skill_names()
        except Exception:
            logger.exception("Could not read the picture settings for generate_image")
            return self.BASE_DESCRIPTION
        parts = [self.GEMINI_BASE_DESCRIPTION if gemini else self.BASE_DESCRIPTION]
        if domains:
            listed = ", ".join(f"load_skill('{name}')" for name in domains)
            parts.append(
                f"Before writing the prompt, load the one skill for the image's domain: {listed}."
            )
        return " ".join(parts)

    async def _dispatch_or_refuse(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        seed: int,
        style: str,
    ) -> ImageGenerationResult:
        """Draw one image, refusing in plain words when no engine can.

        Raises:
            PlainRefusalError: `NO_IMAGE_ENGINE_TEXT`, with `reason_code="no_image_engine"`.
        """
        try:
            return await self._dispatcher.dispatch(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                seed=seed,
                style=style,
            )
        except NoImageEngineError as exc:
            logger.warning("generate_image: no engine could draw: %s", exc)
            raise PlainRefusalError(NO_IMAGE_ENGINE_TEXT, reason_code="no_image_engine") from exc

    async def run(
        self,
        params: GenerateImageParams,
        context: ToolContext,
    ) -> dict[str, Any]:
        """Execute image generation, write artifact, and return structured metadata."""
        if params.prompts is not None:
            if len(params.prompts) == 0:
                raise PlainRefusalError("The 'prompts' list must contain at least 1 prompt.")
            if len(params.prompts) > 10:
                raise PlainRefusalError(
                    "A maximum of 10 prompts can be generated in a single batch."
                )
            for idx, p_text in enumerate(params.prompts):
                if not p_text.strip():
                    raise PlainRefusalError(f"Prompt at index {idx} in 'prompts' cannot be empty.")
                conflicts = find_prompt_conflicts(p_text, params.negative_prompt)
                if conflicts:
                    term_str = ", ".join(f"'{c}'" for c in conflicts)
                    raise PlainRefusalError(
                        f"Prompt conflict detected in prompts[{idx}]: {term_str} appears in both positive prompt and negative_prompt. "
                        "If you want to exclude these elements, remove them from the positive prompt. "
                        "If you want to include them, remove them from the negative prompt."
                    )
        elif not params.prompt.strip():
            raise PlainRefusalError("Either 'prompt' or 'prompts' must be provided.")
        else:
            if params.count > 1:
                composite_pose_pattern = r"\b(dynamic poses? including|poses? including|poses? such as|various poses? including)\b"
                if re.search(composite_pose_pattern, params.prompt, re.IGNORECASE):
                    raise PlainRefusalError(
                        "Multiple poses detected in a single prompt for multi-image generation. "
                        "A single diffusion prompt can only depict one physical posture at a time. "
                        "Please plan separate prompts for each pose and pass them via the 'prompts' parameter: "
                        "generate_image(prompts=['[pose 1]...', '[pose 2]...', ...])."
                    )
                request = latest_user_request(context)
                if request is not None and asks_for_varied_images(request):
                    raise PlainRefusalError(COUNT_FOR_VARIETY_REFUSAL)

            conflicts = find_prompt_conflicts(params.prompt, params.negative_prompt)
            if conflicts:
                term_str = ", ".join(f"'{c}'" for c in conflicts)
                raise PlainRefusalError(
                    f"Prompt conflict detected: {term_str} appears in both positive prompt and negative_prompt. "
                    "If you want to exclude these elements, remove them from the positive prompt. "
                    "If you want to include them, remove them from the negative prompt."
                )

        if params.output_path is not None and names_no_file(params.output_path):
            raise PlainRefusalError(
                f"'{params.output_path}' has no file name, so no picture was made. "
                "Give a file name, such as 'artifacts/images/face.png'."
            )

        session_id = context.session_id or "sess_default"
        turn_idx = getattr(context, "turn_index", 0) or 0
        if not turn_idx and getattr(context, "agent_delegate", None) is not None:
            turn_idx = getattr(context.agent_delegate, "_turn_counter", 0) or 0

        base_prompt = params.prompts[0] if params.prompts is not None else params.prompt
        actual_seed = compute_deterministic_seed(
            session_id=session_id,
            turn_idx=turn_idx,
            prompt=base_prompt,
            seed_override=params.seed_override,
        )

        if params.prompts is not None or params.count > 1:
            prompt_list = (
                params.prompts if params.prompts is not None else [params.prompt] * params.count
            )
            images: list[dict[str, Any]] = []
            last_engine = ""
            last_device = ""
            batch_id = secrets.token_hex(3)
            for i, p_text in enumerate(prompt_list):
                curr_seed = (actual_seed + i) % (2**32)
                if params.output_path is not None:
                    p = Path(params.output_path)
                    candidate = p.parent / f"{p.stem}_{i + 1}{p.suffix or '.png'}"
                    clean_rel = str(candidate).lstrip("/\\")
                    workspace = context.require_workspace()
                    # Refused here, before anything is drawn; resolved again once the
                    # picture's own format has fixed the suffix.
                    self.resolve_write_path(clean_rel, workspace)
                    resolve_image = self.resolve_write_path
                else:
                    clean_rel = f"artifacts/images/img_{batch_id}_{i + 1}.png"
                    resolve_image = self.resolve_safe_path

                gen_result = await self._dispatch_or_refuse(
                    prompt=p_text,
                    negative_prompt=params.negative_prompt,
                    aspect_ratio=params.aspect_ratio,
                    seed=curr_seed,
                    style=params.style,
                )
                last_engine = gen_result.engine_name
                last_device = gen_result.device_info
                batch_changes = list(gen_result.prompt_changes)
                batch_rel = picture_path_for(clean_rel, gen_result.mime_type)
                dest_path = resolve_image(batch_rel, context.require_workspace())
                batch_meta_rel = sidecar_path_for(batch_rel)
                meta_path = self.resolve_write_path(batch_meta_rel, context.require_workspace())

                # Named apart from the single-image writes, so a test can pin these two.
                batch_image = gen_result.image_bytes
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                replace_file(dest_path, batch_image)

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
                    f"{p_text}__{params.negative_prompt}__{curr_seed}__{gen_result.engine_name}".encode()
                ).hexdigest()[:12]

                meta_data: dict[str, Any] = {
                    "id": f"{batch_id}_{i + 1}",
                    "image_path": rel_path,
                    "prompt": p_text,
                    "negative_prompt": params.negative_prompt,
                    "seed": curr_seed,
                    "style": params.style,
                    "aspect_ratio": params.aspect_ratio,
                    "width": gen_result.width,
                    "height": gen_result.height,
                    "engine": gen_result.engine_name,
                    "device": gen_result.device_info,
                    "mime_type": gen_result.mime_type,
                    "duration_seconds": gen_result.duration_seconds,
                    "recipe_hash": recipe_hash,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                batch_meta = json.dumps(meta_data, indent=2, ensure_ascii=False).encode()
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                replace_file(meta_path, batch_meta)

                images.append(
                    {
                        "path": rel_path,
                        "meta_path": rel_meta_path,
                        "relative_url": artifact_content_url(rel_path),
                        "seed": curr_seed,
                        "recipe_hash": recipe_hash,
                        "bytes_written": len(gen_result.image_bytes),
                        "width": gen_result.width,
                        "height": gen_result.height,
                        "mime_type": gen_result.mime_type,
                        "duration_seconds": gen_result.duration_seconds,
                        "prompt": p_text,
                        "prompt_changes": batch_changes,
                    }
                )

            gallery_md = "\n".join(
                f"{idx + 1}. ![{img['prompt'][:30]} #{idx + 1}]({img['relative_url']})"
                for idx, img in enumerate(images)
            )

            res_dict: dict[str, Any] = {
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
                "prompt": images[0]["prompt"],
                "style": params.style,
                "aspect_ratio": params.aspect_ratio,
                "engine": last_engine,
                "device": last_device,
            }
            if params.prompts is not None:
                res_dict["prompts"] = [img["prompt"] for img in images]
            return res_dict

        short_id = secrets.token_hex(3)
        # 1. Resolve and validate safe destination path when output_path is provided
        if params.output_path is not None:
            clean_rel = params.output_path.lstrip("/\\") or f"artifacts/images/img_{short_id}.png"
            # Refused here, before anything is drawn; resolved again once the picture's own
            # format has fixed the suffix.
            self.resolve_write_path(clean_rel, context.require_workspace())
            resolve_dest = self.resolve_write_path
        else:
            clean_rel = f"artifacts/images/img_{short_id}.png"
            resolve_dest = self.resolve_safe_path

        # 2. Dispatch generation
        gen_result = await self._dispatch_or_refuse(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            aspect_ratio=params.aspect_ratio,
            seed=actual_seed,
            style=params.style,
        )

        picture_rel = picture_path_for(clean_rel, gen_result.mime_type)
        dest_path = resolve_dest(picture_rel, context.require_workspace())
        meta_rel = sidecar_path_for(picture_rel)
        meta_path = self.resolve_write_path(meta_rel, context.require_workspace())

        # 3. Write image artifact securely
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        replace_file(dest_path, gen_result.image_bytes)

        try:
            rel_path = str(dest_path.relative_to(context.require_workspace().resolve()))
        except ValueError:
            rel_path = str(dest_path)

        try:
            rel_meta_path = str(meta_path.relative_to(context.require_workspace().resolve()))
        except ValueError:
            rel_meta_path = str(meta_path)

        recipe_hash = hashlib.sha256(
            f"{params.prompt}__{params.negative_prompt}__{actual_seed}__{gen_result.engine_name}".encode()
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
            "mime_type": gen_result.mime_type,
            "duration_seconds": gen_result.duration_seconds,
            "recipe_hash": recipe_hash,
            "created_at": datetime.now(UTC).isoformat(),
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        replace_file(meta_path, json.dumps(meta_data, indent=2, ensure_ascii=False).encode())

        return {
            "status": "success",
            "path": rel_path,
            "meta_path": rel_meta_path,
            "relative_url": artifact_content_url(rel_path),
            "prompt": params.prompt,
            "style": params.style,
            "aspect_ratio": params.aspect_ratio,
            "width": gen_result.width,
            "height": gen_result.height,
            "seed": actual_seed,
            "recipe_hash": recipe_hash,
            "engine": gen_result.engine_name,
            "device": gen_result.device_info,
            "mime_type": gen_result.mime_type,
            "duration_seconds": gen_result.duration_seconds,
            "bytes_written": len(gen_result.image_bytes),
            "prompt_changes": list(gen_result.prompt_changes),
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
