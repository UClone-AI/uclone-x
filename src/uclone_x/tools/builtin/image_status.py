"""What each image engine offers right now, read without installing or starting anything.

`ucx media status`, `ucx install` and the dashboard's `GET /api/media/status` all read
this one probe. It lives outside `cli/` so a server installed without the `cli` extra
(no `rich`) can still answer the status route (#1769).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict

if TYPE_CHECKING:
    from uclone_x.tools.builtin.image import ImageWhere


class KnownModel(NamedTuple):
    """A registered model a local engine loads: its id and the name a person reads."""

    model_id: str
    label: str


#: Why nothing resolves to draw with: nothing at all under `auto`, or the chosen model is
#: not ready (its connection does not answer, has no key, or is gone).
ResolveReasonCode = Literal["no_image_model", "chosen_not_ready"]


class ResolvedModel(TypedDict):
    """The model that draws the next picture, as `/api/media/status` names it (design §3.8).

    ``model_id`` and ``label`` are `None` when the engine reports no model (the remote
    worker, an unregistered checkpoint); ``label`` is never an engine's internal name.
    """

    engine: str
    model_id: str | None
    label: str | None
    where: ImageWhere


class ResolvedChoice(TypedDict):
    """``create`` is what draws now, or `None` with the ``reason_code`` saying why.

    ``refusal`` is the plain sentence a draw would be refused with, for a chosen model
    that cannot be used at all (model-gateway §3.6); `None` otherwise.
    """

    create: ResolvedModel | None
    reason_code: ResolveReasonCode | None
    refusal: str | None


class ImageEngineReport(NamedTuple):
    """What each image engine offers right now, inspected rather than assumed (#1095).

    The engines are the connections (model-gateway §3.5): ``comfy_url`` and ``remote_url``
    are their connections' addresses, `None` when there is none.
    """

    remote_url: str | None
    remote_alive: bool
    comfy_url: str | None
    comfy_alive: bool
    dependency_problems: tuple[str, ...]
    checkpoint: str | None
    #: The picture model in effect (`auto` or a ref), the engine a ref pins, and the
    #: sentence a pin that cannot be used at all is refused with.
    setting: str = "auto"
    pin: str | None = None
    refusal: str | None = None
    #: The cloud picture model `auto` may use (on a connection with a key) or the pinned
    #: one; `None` when there is none.
    gemini_model: str | None = None
    #: The registered model each local engine loads (a ComfyUI pin's own model), `None`
    #: when none is known, and the local port of the connected remote-GPU tunnel's ComfyUI
    #: (`None` without one).
    comfy_model: KnownModel | None = None
    in_process_model: KnownModel | None = None
    gpu_tunnel_comfy_port: int | None = None
    gemini_label: str | None = None
    #: A ComfyUI answering on this computer's default address while no ComfyUI connection
    #: is saved: detected, never used until the person agrees to add it (never seize).
    detected_comfyui: str | None = None

    @property
    def gemini_ready(self) -> bool:
        """Whether the cloud can draw: a cloud picture model with a key (pins: `engine`).

        Read from the key's presence, not a request: probing Google costs a call on every
        status read.
        """
        return self.gemini_model is not None

    @property
    def ready(self) -> bool:
        """Whether the picture model in effect can actually generate an image.

        A configured remote worker address is an address, not a worker. It counts only
        once `/health` has answered, the same probe `RemoteCudaImageEngine.is_available`
        makes before the dispatcher will use it (P6).
        """
        return self.engine != "none"

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
        """The engine that would draw, matching `ImagePipelineDispatcher`'s rule."""
        if self.refusal is not None:
            return "none"
        if self.pin == "gemini":
            return "gemini" if self.gemini_ready else "none"
        if self.pin == "comfyui":
            return "comfyui-local" if self.comfy_alive else "none"
        if self.pin == "remote_gpu":
            return "remote-cuda" if self.remote_ready else "none"
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

        ``ready`` is whether that engine could draw under the picture model in effect; the
        reason code says why not, or is ``ready``. A pin to one engine leaves the others
        ``disabled_by_setting``.
        """
        pinned = {"remote_gpu": "remote-cuda", "comfyui": "comfyui-local", "gemini": "gemini"}

        def under_pin(name: str, problem: str | None) -> tuple[bool, str]:
            if self.pin is not None and pinned[self.pin] != name:
                return False, "disabled_by_setting"
            return (False, problem) if problem is not None else (True, "ready")

        if not self.remote_url:
            remote = under_pin("remote-cuda", "not_configured")
        else:
            remote = under_pin("remote-cuda", None if self.remote_alive else "unreachable")
        if self.comfy_url is None:
            comfy = under_pin("comfyui-local", "not_configured")
        else:
            comfy = under_pin("comfyui-local", None if self.comfy_alive else "not_running")
        if self.pin is not None:
            in_process = (False, "disabled_by_setting")
        elif not self.dependencies_ok:
            in_process = (False, "missing_dependencies")
        else:
            in_process = (
                (True, "ready") if self.checkpoint is not None else (False, "no_checkpoint")
            )
        gemini = under_pin("gemini", None if self.gemini_model is not None else "no_key")
        named = (
            ("remote-cuda", remote),
            ("comfyui-local", comfy),
            ("diffusers-sdxl", in_process),
            ("gemini", gemini),
        )
        return [(name, state[0], state[1]) for name, state in named]


def resolve_image_choice(report: ImageEngineReport) -> ResolvedChoice:
    """What draws the next picture under ``report``, and where; the one answer both heads give.

    `/api/media/status` and `ucx media status` both return this, so the dashboard and
    the command line cannot disagree about the model or the place (design §3.8).
    """
    from uclone_x.tools.builtin.image import image_where

    engine = report.engine
    if engine == "none":
        reason: ResolveReasonCode = "no_image_model" if report.pin is None else "chosen_not_ready"
        if report.refusal is not None:
            reason = "chosen_not_ready"
        return {"create": None, "reason_code": reason, "refusal": report.refusal}
    known: KnownModel | None = None
    if engine == "gemini":
        model = report.gemini_model or ""
        known = KnownModel(model, report.gemini_label or model)
        where = image_where("gemini")
    elif engine == "remote-cuda":
        where = image_where("remote-cuda")
    elif engine == "comfyui-local":
        known = report.comfy_model
        where = image_where("comfyui-local", report.comfy_url, report.gpu_tunnel_comfy_port)
    else:
        known = report.in_process_model
        where = image_where("diffusers-sdxl")
    create: ResolvedModel = {
        "engine": engine,
        "model_id": known.model_id if known is not None else None,
        "label": known.label if known is not None else None,
        "where": where,
    }
    return {"create": create, "reason_code": None, "refusal": None}


def media_status_payload(report: ImageEngineReport) -> dict[str, Any]:
    """The status object `/api/media/status` answers and `ucx media status --json` prints."""
    return {
        "ready": report.ready,
        "engine": report.engine,
        "setting": report.setting,
        "engines": [
            {"name": name, "ready": ready, "reason_code": code}
            for name, ready, code in report.engine_states()
        ],
        "resolved": resolve_image_choice(report),
        "detected_comfyui": report.detected_comfyui,
    }


def detect_local_comfyui() -> str | None:
    """The default address, when a ComfyUI answers there; else `None` (#1095: detect only).

    Nothing is started, installed or saved: a caller offers to add the connection and adds
    it only when the person says yes (agent-assisted-installation, rung 1).
    """
    import asyncio

    from uclone_x.llm.connections import DEFAULT_COMFYUI_ADDRESS
    from uclone_x.tools.builtin.image import ComfyUIImageEngine

    try:
        alive = asyncio.run(ComfyUIImageEngine(base_url=DEFAULT_COMFYUI_ADDRESS).is_available())
    except Exception:
        alive = False
    return DEFAULT_COMFYUI_ADDRESS if alive else None


def _known_model(checkpoint: str | None) -> KnownModel | None:
    """The registered profile ``checkpoint`` resolves to, or `None` for none or a fallback."""
    from uclone_x.tools.builtin.image import is_registered_profile
    from uclone_x.tools.builtin.media_registry import ModelRegistry

    if not checkpoint:
        return None
    profile = ModelRegistry().resolve(checkpoint)
    if not is_registered_profile(profile):
        return None
    return KnownModel(profile.model_id, profile.display_name)


def probe_image_engines(
    settings_path: Path | None = None,
    gpu_tunnel_comfy_port: int | None = None,
    *,
    detect_comfyui: bool = False,
) -> ImageEngineReport:
    """Inspect every image engine without installing, starting or downloading anything.

    The picture model in effect and the connections are read from ``settings_path`` (the
    session root's by default) through `image_engine_choice`, the rule every draw reads,
    so this cannot report an engine the dispatcher would not ask. The cloud is judged by
    whether its connection has a key, and is never sent a request.

    Every field is read from this machine as it is: two HTTP probes -- the remote worker's
    `/health` and the ComfyUI connection's daemon -- an import, and a file on disk. Nothing
    here can report ready for an engine that would then fail to produce an image (P6),
    which is why the remote worker is probed rather than inferred from its address.

    ``detect_comfyui`` also looks for a ComfyUI on the default address while no ComfyUI
    connection is saved, so a head can offer it (``detected_comfyui``); it is never used.
    """
    import asyncio

    from uclone_x.llm.connectors.factory import image_engine_choice
    from uclone_x.tools.builtin.image import (
        ComfyUIImageEngine,
        LocalDiffusersImageEngine,
        RemoteCudaImageEngine,
        in_process_dependency_problems,
    )
    from uclone_x.tools.builtin.media_registry import ModelRegistry

    choice = image_engine_choice(settings_path, gpu_tunnel_comfy_port=gpu_tunnel_comfy_port)
    remote_url = choice.remote_url
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
    comfy_url = choice.comfyui_base_url
    comfy_alive = False
    comfy_model: KnownModel | None = None
    # With no ComfyUI connection, one on the default address is only looked for, to offer.
    detected = detect_local_comfyui() if detect_comfyui and comfy_url is None else None
    if comfy_url:
        comfy = ComfyUIImageEngine(base_url=comfy_url)
        try:
            comfy_alive = asyncio.run(comfy.is_available())
        except Exception:
            comfy_alive = False  # a probe that raises is no daemon there
        registry = ModelRegistry()
        if choice.pinned_profile:
            pinned = registry.resolve(choice.pinned_profile)
            comfy_model = KnownModel(pinned.model_id, pinned.display_name)
        else:
            comfy_model = _known_model(comfy.checkpoint)

    gemini_label: str | None = None
    if choice.gemini_model is not None:
        cloud = ModelRegistry().resolve(choice.gemini_model)
        if cloud.model_id == choice.gemini_model:
            gemini_label = cloud.display_name
    checkpoint = LocalDiffusersImageEngine().resolve_checkpoint()
    return ImageEngineReport(
        remote_url=remote_url or None,
        remote_alive=remote_alive,
        comfy_url=comfy_url,
        comfy_alive=comfy_alive,
        dependency_problems=tuple(
            problem.describe() for problem in in_process_dependency_problems()
        ),
        checkpoint=checkpoint,
        setting=choice.chosen,
        pin=choice.pin,
        refusal=choice.refusal,
        gemini_model=choice.gemini_model if choice.gemini is not None else None,
        comfy_model=comfy_model,
        in_process_model=_known_model(checkpoint),
        gpu_tunnel_comfy_port=gpu_tunnel_comfy_port,
        gemini_label=gemini_label,
        detected_comfyui=detected,
    )
