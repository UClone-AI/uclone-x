"""ComfyUI image generation agent tool implementing standard txt2img workflows."""

from __future__ import annotations

import secrets
import uuid
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import ComfyUIError
from uclone_x.tools.base import BaseTool, artifact_content_url, replace_file
from uclone_x.tools.builtin.comfy_client import (
    COMFY_CHECKPOINT_ENV,
    COMFY_DEFAULT_CHECKPOINT,
    ComfyClient,
    default_comfy_checkpoint,
)
from uclone_x.tools.models import ToolContext, ToolResult


class ComfyImageGenParams(BaseModel):
    """Parameters for generating an image via ComfyUI."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(description="Positive text prompt describing the image to generate")
    negative_prompt: str = Field(
        default="",
        description="Negative text prompt describing unwanted artifacts or elements",
    )
    width: int = Field(
        default=832,
        ge=64,
        le=4096,
        description="Image width in pixels (multiple of 8)",
    )
    height: int = Field(
        default=1216,
        ge=64,
        le=4096,
        description="Image height in pixels (multiple of 8)",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**63 - 1,
        description="Random seed for generation reproducibility. If None, randomly generated.",
    )
    steps: int = Field(
        default=30,
        ge=1,
        le=150,
        description="Sampling steps count (default: 30)",
    )
    cfg: float = Field(
        default=5.5,
        ge=1.0,
        le=30.0,
        description="Classifier-Free Guidance (CFG) scale (default: 5.5)",
    )
    sampler_name: str = Field(
        default="dpmpp_2m",
        description="KSampler algorithm name (e.g. euler, euler_ancestral, dpmpp_2m)",
    )
    scheduler: str = Field(
        default="karras",
        description="Noise scheduler (e.g. normal, karras, exponential, sgm_uniform)",
    )
    checkpoint: str = Field(
        # `default_factory`, not `default`: a plain default is evaluated once, when this class
        # is defined, which would pin the value to whatever the environment held at import.
        # The factory runs per instance — i.e. per tool call — so `UCX_COMFYUI_CHECKPOINT`
        # reaches this tool at the moment it is used, as it already reaches the engine (#1223).
        default_factory=default_comfy_checkpoint,
        description=(
            "Checkpoint model name to load in CheckpointLoaderSimple. Defaults to "
            f"${COMFY_CHECKPOINT_ENV} when set, otherwise {COMFY_DEFAULT_CHECKPOINT}."
        ),
    )
    output_path: str | None = Field(
        default=None,
        description="Relative destination file path within workspace. Defaults to 'artifacts/images/{seed}_{uuid}.png'.",
    )
    timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description="Timeout in seconds waiting for generation completion",
    )


def build_txt2img_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 1216,
    seed: int = 0,
    steps: int = 30,
    cfg: float = 5.5,
    sampler_name: str = "dpmpp_2m",
    scheduler: str = "karras",
    checkpoint: str | None = None,
    filename_prefix: str = "UCloneX",
) -> dict[str, Any]:
    """Construct a standard ComfyUI txt2img workflow graph.

    `checkpoint=None` means "whatever is configured", resolved here rather than in a parameter
    default, which Python would evaluate once at import and freeze (#1223).
    """
    return {
        "4": {
            "inputs": {
                "ckpt_name": checkpoint or default_comfy_checkpoint(),
            },
            "class_type": "CheckpointLoaderSimple",
        },
        "5": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
            "class_type": "EmptyLatentImage",
        },
        "6": {
            "inputs": {
                "text": prompt,
                "clip": ["4", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "7": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "3": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
            "class_type": "KSampler",
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
            "class_type": "VAEDecode",
        },
        "9": {
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["8", 0],
            },
            "class_type": "SaveImage",
        },
    }


class ComfyImageGenTool(BaseTool[ComfyImageGenParams]):
    """Agent tool generating images via ComfyUI with workspace containment."""

    name: str = "generate_image"
    writes_files: ClassVar[bool] = True  # can create, modify or delete a file on the host (#1167)
    description: str = (
        "Generate an image from a text prompt using ComfyUI standard txt2img workflow "
        "and save the output artifact securely within the workspace."
    )

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        client: ComfyClient | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self._client: ComfyClient | None = client
        self._base_url: str | None = base_url

    @property
    def base_url(self) -> str | None:
        """Configured ComfyUI base URL."""
        return self._base_url

    def update_base_url(self, base_url: str) -> None:
        """Update the ComfyUI endpoint URL and recreate the client instance."""
        self._base_url = base_url
        self._client = ComfyClient(base_url=base_url)

    @property
    def client(self) -> ComfyClient:
        """Return the ComfyClient instance used by this tool."""
        return self._get_client()

    def _get_client(self) -> ComfyClient:
        if self._client is not None:
            return self._client
        return ComfyClient(base_url=self._base_url)

    async def run(
        self,
        params: ComfyImageGenParams,
        context: ToolContext,
    ) -> dict[str, Any]:
        """Execute the text-to-image workflow, download result, and persist artifact."""
        actual_seed = params.seed if params.seed is not None else secrets.randbelow(2**32)

        # 1. Path containment check: resolve safe path before queuing workflow
        if params.output_path is not None:
            dest_path = self.resolve_write_path(params.output_path, context.require_workspace())
        else:
            default_rel = f"artifacts/images/{actual_seed}_{uuid.uuid4().hex[:8]}.png"
            dest_path = self.resolve_safe_path(default_rel, context.require_workspace())

        # 2. Build standard txt2img workflow graph
        workflow = build_txt2img_workflow(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            width=params.width,
            height=params.height,
            seed=actual_seed,
            steps=params.steps,
            cfg=params.cfg,
            sampler_name=params.sampler_name,
            scheduler=params.scheduler,
            checkpoint=params.checkpoint,
        )

        # 3. Queue workflow and await completion
        client = self._get_client()
        prompt_id = await client.queue_prompt(workflow)
        filenames = await client.wait_for_output(
            prompt_id,
            timeout_seconds=params.timeout_seconds,
        )
        if not filenames:
            raise ComfyUIError(f"ComfyUI produced no image outputs for prompt {prompt_id}")

        # 4. Download generated image artifact
        primary_filename = filenames[0]
        image_bytes = await client.download_image(primary_filename)

        # 5. Persist artifact safely within workspace
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        replace_file(dest_path, image_bytes)

        try:
            rel_path = str(dest_path.relative_to(context.require_workspace().resolve()))
        except ValueError:
            rel_path = str(dest_path)

        return {
            "path": rel_path,
            "relative_url": artifact_content_url(rel_path),
            "prompt_id": prompt_id,
            "filename": primary_filename,
            "width": params.width,
            "height": params.height,
            "seed": actual_seed,
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "checkpoint": params.checkpoint,
            "bytes_written": len(image_bytes),
        }

    async def execute(
        self,
        params: dict[str, Any] | ToolContext | None = None,
        context: ToolContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute tool and decorate result with artifact path provenance."""
        result = await super().execute(params=params, context=context, **kwargs)
        if result.success and isinstance(result.output, dict) and "path" in result.output:
            path_val = str(result.output["path"])
            return result.model_copy(update={"artifacts": (path_val,)})
        return result
