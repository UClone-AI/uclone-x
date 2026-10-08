"""`ucx media` — what the local image engines have, read from this machine (#1095).

Every line this prints is inspected: an HTTP probe of the configured remote worker, one
HTTP probe of a ComfyUI daemon somebody else started, an import, and files on disk.
Nothing is downloaded, installed or started from here, and nothing is inferred — the command exists so that "why did I not
get an image?" has an answer that does not require reading the dispatcher's source.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from uclone_x.cli.commands.bootstrap import probe_image_engines
from uclone_x.errors import PlainRefusalError
from uclone_x.llm.connectors.factory import image_engine_choice
from uclone_x.tools.builtin.image import (
    COMFY_URL_ENV,
    DEFAULT_CHECKPOINTS,
    IMAGE_CHECKPOINT_ENV,
    ComfyUIImageEngine,
    ImageWhere,
    LocalDiffusersImageEngine,
    diffusers_install_hint,
    expand_checkpoint_path,
    in_process_device,
)
from uclone_x.tools.builtin.image_status import media_status_payload, resolve_image_choice

media_app = typer.Typer(
    name="media",
    help="Inspect the local image generation engines",
    no_args_is_help=True,
)
console = Console()

#: Where a picture is drawn, in the words Settings uses (design §3.1).
WHERE_WORDS: dict[ImageWhere, str] = {
    "this_computer": "This computer",
    "gpu_server": "Your GPU server",
    "cloud": "Cloud · Google",
}


def human_bytes(size: int) -> str:
    """A size in GB to two decimals, or MB below a gigabyte."""
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    return f"{size / 1024**2:.0f} MB"


def local_checkpoints() -> list[tuple[str, int]]:
    """Every candidate checkpoint that exists, with its size — configured one first.

    Expansion is `expand_checkpoint_path`'s decision, not this function's. It used to be
    `Path(...).expanduser()` here and nothing at all in `checkpoint_resolution()`, so a
    configured `~/...` listed as present and resolved as missing (#1123). One authority
    answers both now; keeping a second `expanduser()` here would only re-open the gap.
    """
    found: list[tuple[str, int]] = []
    seen: set[str] = set()
    candidates = [os.getenv(IMAGE_CHECKPOINT_ENV) or "", *DEFAULT_CHECKPOINTS]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(expand_checkpoint_path(candidate))
        resolved = str(path)
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        found.append((resolved, path.stat().st_size))
    return found


#: Why the cloud would not draw, by `ImageEngineReport.engine_states` reason code.
GEMINI_REASONS = {
    "disabled_by_setting": "off (another picture model is chosen)",
    "no_key": "no Google connection with a key",
}


@media_app.command("status")
def media_status(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the status object the dashboard reads, as JSON"),
    ] = False,
) -> None:
    """Report which image engine would run, and what each one is missing."""
    try:
        report = probe_image_engines()
    except PlainRefusalError as exc:
        console.print(f"[bold red]✖ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    if as_json:
        # The object `/api/media/status` answers, built by the same function.
        typer.echo(json.dumps(media_status_payload(report), indent=2))
        if not report.ready:
            raise typer.Exit(code=1)
        return

    console.print("[bold cyan]🎨 Local image engines[/bold cyan]")

    if report.remote_url is None:
        console.print("  1. Remote CUDA worker: [cyan]not configured[/cyan]")
    elif report.remote_alive:
        console.print(
            f"  1. Remote CUDA worker: [cyan]{report.remote_url}[/cyan] "
            "[bold green]reachable[/bold green]"
        )
    else:
        # Configured is not running. The dispatcher probes `/health` before it will use
        # this engine, so an address that does not answer is reported as what it is
        # rather than counted towards readiness (P6).
        console.print(
            f"  1. Remote CUDA worker: [cyan]{report.remote_url}[/cyan] "
            "[bold yellow]unreachable[/bold yellow]"
        )

    if report.comfy_url is None:
        console.print("  2. ComfyUI: [cyan]no connection[/cyan]")
    else:
        state = "[bold green]detected[/bold green]" if report.comfy_alive else "not running"
        console.print(f"  2. ComfyUI ({report.comfy_url}): {state}")
    if not report.comfy_alive:
        console.print(
            "     [dim]Optional. Add a ComfyUI connection in Settings › Models, or set "
            f"{COMFY_URL_ENV}; UClone-X never installs or starts one.[/dim]"
        )

    if report.dependencies_ok:
        console.print("  3. In-process (no daemon): [bold green]dependencies ready[/bold green]")
        device = in_process_device()
        if device is not None:
            console.print(f"     device: [cyan]{device}[/cyan]")
    else:
        console.print(
            "  3. In-process (no daemon): [bold yellow]dependencies missing[/bold yellow]"
        )
        for problem in report.dependency_problems:
            console.print(f"     [bold yellow]![/bold yellow] {problem}")
        console.print(f"     [dim]{diffusers_install_hint()}[/dim]")
    # The marker means "this is the file the selected engine would load", so it is printed
    # only when the in-process engine is the one that would run. Matching on the path alone
    # marked a checkpoint whose engine is missing `torch`, pointing at a load that must fail
    # (#1120, P6).
    in_process_selected = report.engine == "diffusers-sdxl"
    for path, size in local_checkpoints():
        marker = "→" if in_process_selected and path == report.checkpoint else " "
        console.print(f"     {marker} [cyan]{path}[/cyan] ({human_bytes(size)})")
    # Nothing configured and a configured path that is not there are different mistakes,
    # and the engine is the one that knows which happened. Asked unconditionally rather
    # than only when the listing is empty: a mistyped `UCX_IMAGE_CHECKPOINT` alongside a
    # default checkpoint on disk lists a file and still has nothing to load.
    resolution = LocalDiffusersImageEngine().checkpoint_resolution()
    if not resolution.usable:
        headline = "checkpoint missing" if resolution.state == "missing" else "no checkpoint found"
        console.print(f"     [bold yellow]{headline}[/bold yellow] — {resolution.describe()}")

    gemini_reason = next(code for name, _, code in report.engine_states() if name == "gemini")
    if gemini_reason == "ready":
        gemini_state = f"[bold green]ready[/bold green] ({report.gemini_model}, over the internet)"
    else:
        gemini_state = GEMINI_REASONS.get(gemini_reason, gemini_reason)
    console.print(f"  4. Google Gemini (cloud): {gemini_state}")
    console.print(f"     [dim]picture model: {escape(report.setting)}[/dim]")

    resolved = resolve_image_choice(report)
    if resolved["refusal"] is not None:
        console.print(f"  [bold yellow]![/bold yellow] {escape(resolved['refusal'])}")
    create = resolved["create"]
    if create is not None:
        label = create["label"] or "model not reported"
        console.print(
            f"  Draws with: [cyan]{escape(label)}[/cyan] · {WHERE_WORDS[create['where']]}"
        )

    if report.ready:
        console.print(f"[bold green]✔ Ready — '{report.engine}' would run.[/bold green]")
        return
    console.print("[bold yellow]✖ No image engine is ready.[/bold yellow]")
    raise typer.Exit(code=1)


@media_app.command("probe")
def media_probe() -> None:
    """Ask the ComfyUI daemon what it is, when one answers."""
    engine = ComfyUIImageEngine()
    engine.use_saved_address(image_engine_choice().comfyui_base_url)
    from uclone_x.tools.builtin.comfy_client import ComfyClient

    async def _stats() -> dict[str, object] | None:
        client = ComfyClient(base_url=engine.base_url)
        try:
            if not await client.alive():
                return None
            return await client.system_stats()
        finally:
            await client.aclose()

    try:
        stats = asyncio.run(_stats())
    except Exception as exc:
        console.print(f"[bold red]✖ Probe failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    if stats is None:
        console.print(f"[bold yellow]No ComfyUI answered at {engine.base_url}.[/bold yellow]")
        raise typer.Exit(code=1)
    console.print(f"[bold green]✔ ComfyUI at {engine.base_url}[/bold green]")
    console.print(stats)
