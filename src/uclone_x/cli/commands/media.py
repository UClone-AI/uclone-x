"""`ucx media` — what the local image engines have, read from this machine (#1095).

Every line this prints is inspected: an HTTP probe of the configured remote worker, one
HTTP probe of a ComfyUI daemon somebody else started, an import, and files on disk.
Nothing is downloaded, installed or started from here, and nothing is inferred — the command exists so that "why did I not
get an image?" has an answer that does not require reading the dispatcher's source.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console

from uclone_x.cli.commands.bootstrap import probe_image_engines
from uclone_x.tools.builtin.image import (
    COMFY_URL_ENV,
    DEFAULT_CHECKPOINTS,
    IMAGE_CHECKPOINT_ENV,
    ComfyUIImageEngine,
    LocalDiffusersImageEngine,
    diffusers_install_hint,
    expand_checkpoint_path,
)

media_app = typer.Typer(
    name="media",
    help="Inspect the local image generation engines",
    no_args_is_help=True,
)
console = Console()


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


@media_app.command("status")
def media_status() -> None:
    """Report which image engine would run, and what each one is missing."""
    report = probe_image_engines()

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

    state = "[bold green]detected[/bold green]" if report.comfy_alive else "not running"
    console.print(f"  2. Local ComfyUI ({report.comfy_url}): {state}")
    if not report.comfy_alive:
        console.print(
            f"     [dim]Optional. Start your own ComfyUI, or set {COMFY_URL_ENV}; "
            "UClone-X never installs or starts one.[/dim]"
        )

    if report.dependencies_ok:
        console.print("  3. In-process (no daemon): [bold green]dependencies ready[/bold green]")
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

    if report.ready:
        console.print(f"[bold green]✔ Ready — '{report.engine}' would run.[/bold green]")
        return
    console.print("[bold yellow]✖ No image engine is ready.[/bold yellow]")
    raise typer.Exit(code=1)


@media_app.command("probe")
def media_probe() -> None:
    """Ask the ComfyUI daemon what it is, when one answers."""
    engine = ComfyUIImageEngine()
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
