"""What each image engine offers right now, read without installing or starting anything.

`ucx media status`, `ucx install` and the dashboard's `GET /api/media/status` all read
this one probe. It lives outside `cli/` so a server installed without the `cli` extra
(no `rich`) can still answer the status route (#1769).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple


class ImageEngineReport(NamedTuple):
    """What each image engine offers right now, inspected rather than assumed (#1095)."""

    remote_url: str | None
    remote_alive: bool
    comfy_url: str
    comfy_alive: bool
    dependency_problems: tuple[str, ...]
    checkpoint: str | None
    #: The `image_engine` setting, the chat provider in effect, whether a Gemini key is
    #: there, and the Gemini image model. The defaults are a report with no Gemini at all.
    image_engine: str = "auto"
    chat_provider: str | None = None
    gemini_key: bool = False
    image_model: str = ""

    @property
    def local_ready(self) -> bool:
        """Whether a local engine -- remote worker, ComfyUI or in-process -- can draw."""
        return self.remote_ready or self.comfy_alive or self.in_process_ready

    @property
    def gemini_ready(self) -> bool:
        """Whether Gemini may draw under the setting: a key, and `gemini`, or `auto` with Gemini chat.

        Read from the key's presence, not a request: probing Google would itself be the
        network call `local` promises not to make, and costs a call on every status read.
        """
        if not self.gemini_key:
            return False
        return self.image_engine == "gemini" or (
            self.image_engine == "auto" and self.chat_provider == "gemini"
        )

    @property
    def ready(self) -> bool:
        """Whether the engine the `image_engine` setting allows can actually generate an image.

        A configured `UCX_IMAGE_REMOTE_URL` is an address, not a worker. It counts only
        once `/health` has answered, the same probe `RemoteCudaImageEngine.is_available`
        makes before the dispatcher will use it; reporting ready from the variable alone
        promised an engine that then refused to generate (P6).
        """
        if self.image_engine == "gemini":
            return self.gemini_ready
        return self.local_ready or self.gemini_ready

    @property
    def remote_ready(self) -> bool:
        """A remote worker that is both configured and answering."""
        return bool(self.remote_url) and self.remote_alive

    @property
    def dependencies_ok(self) -> bool:
        """Whether every package the in-process engine imports is present and new enough."""
        return not self.dependency_problems

    @property
    def in_process_ready(self) -> bool:
        """The daemon-free baseline: every import is satisfied and a checkpoint file exists."""
        return self.dependencies_ok and self.checkpoint is not None

    @property
    def engine(self) -> str:
        """The engine that would be selected, matching `ImagePipelineDispatcher`'s order."""
        if self.image_engine == "gemini":
            return "gemini" if self.gemini_ready else "none"
        if self.remote_ready:
            return "remote-cuda"
        if self.comfy_alive:
            return "comfyui-local"
        if self.in_process_ready:
            return "diffusers-sdxl"
        if self.gemini_ready:
            return "gemini"
        return "none"

    def engine_states(self) -> list[tuple[str, bool, str]]:
        """``(engine, ready, reason_code)`` for each engine, in the order they are tried.

        ``ready`` is whether that engine could draw under the current setting; the
        reason code says why not, or is ``ready``.
        """
        local_off = self.image_engine == "gemini"

        def local(problem: str | None) -> tuple[bool, str]:
            if local_off:
                return False, "disabled_by_setting"
            return (False, problem) if problem is not None else (True, "ready")

        if not self.remote_url:
            remote = local("not_configured")
        else:
            remote = local(None if self.remote_alive else "unreachable")
        comfy = local(None if self.comfy_alive else "not_running")
        if not self.dependencies_ok:
            in_process = local("missing_dependencies")
        else:
            in_process = local(None if self.checkpoint is not None else "no_checkpoint")
        if self.image_engine == "local":
            gemini = (False, "disabled_by_setting")
        elif not self.gemini_key:
            gemini = (False, "no_key")
        elif not self.gemini_ready:
            gemini = (False, "chat_provider_not_gemini")
        else:
            gemini = (True, "ready")
        named = (
            ("remote-cuda", remote),
            ("comfyui-local", comfy),
            ("diffusers-sdxl", in_process),
            ("gemini", gemini),
        )
        return [(name, state[0], state[1]) for name, state in named]


def probe_image_engines(
    settings_path: Path | None = None, chat_provider: str | None = None
) -> ImageEngineReport:
    """Inspect every image engine without installing, starting or downloading anything.

    The picture settings are read from ``settings_path`` (the session root's by default);
    ``chat_provider`` is the chat provider in effect, which `auto` needs to decide on
    Gemini. Gemini is judged by whether its key is there, and is never sent a request.

    Every field is read from this machine as it is: two HTTP probes -- the remote worker's
    `/health` and a ComfyUI daemon somebody else started -- an import, and a file on disk.
    Nothing here can report ready for an engine that would then fail to produce an image
    (P6), which is why the remote worker is probed rather than inferred from its variable.
    """
    import asyncio

    from uclone_x.llm.connectors.factory import gemini_key_available
    from uclone_x.llm.connectors.saved_choice import settings_data
    from uclone_x.llm.providers import canonical_provider
    from uclone_x.tools.builtin.image import (
        IMAGE_ENGINE_KEY,
        IMAGE_MODEL_KEY,
        REMOTE_URL_ENV,
        ComfyUIImageEngine,
        LocalDiffusersImageEngine,
        RemoteCudaImageEngine,
        in_process_dependency_problems,
        parse_image_engine_setting,
        parse_image_model,
    )

    # First, so a stored value the dispatcher would refuse is refused here too, before
    # anything is probed.
    data = settings_data(settings_path)
    image_engine = parse_image_engine_setting(data.get(IMAGE_ENGINE_KEY))
    image_model = parse_image_model(data.get(IMAGE_MODEL_KEY))

    remote_url = os.getenv(REMOTE_URL_ENV) or os.getenv("UCX_MEDIA_REMOTE_URL")
    try:
        # Built inside the `try` so that a client which rejects the configured address
        # reads as "not there", the same as a refused connection. No address, no request.
        remote_alive = (
            asyncio.run(RemoteCudaImageEngine(base_url=remote_url).is_available())
            if remote_url
            else False
        )
    except Exception:
        remote_alive = False
    comfy = ComfyUIImageEngine()
    try:
        comfy_alive = asyncio.run(comfy.is_available())
    except Exception:
        comfy_alive = False

    return ImageEngineReport(
        remote_url=remote_url or None,
        remote_alive=remote_alive,
        comfy_url=comfy.base_url,
        comfy_alive=comfy_alive,
        dependency_problems=tuple(
            problem.describe() for problem in in_process_dependency_problems()
        ),
        checkpoint=LocalDiffusersImageEngine().resolve_checkpoint(),
        image_engine=image_engine,
        chat_provider=canonical_provider(chat_provider),
        gemini_key=gemini_key_available(data),
        image_model=image_model,
    )
