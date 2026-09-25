"""CLI commands for managing and testing local LLM endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import urllib.request

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from uclone_x.errors import LLMProviderError
from uclone_x.llm.connectors.ollama import delete_model, resolve_ollama_base_url
from uclone_x.llm.connectors.vllm import (
    VLLM_ENDPOINT_ENV_VARS,
    VLLM_MODEL_ENV_VAR,
    has_configured_vllm_endpoint,
    resolve_vllm_base_url,
)

llm_app = typer.Typer(
    name="llm",
    help="Manage and inspect local LLM endpoints (2-Tier Ollama, vLLM)",
    no_args_is_help=True,
)
console = Console()


def _check_ollama_endpoint(url: str, timeout: float | None = None) -> list[str] | None:
    """Check if an Ollama endpoint is reachable and return model names."""
    if timeout is None:
        try:
            timeout = float(os.getenv("OLLAMA_CHECK_TIMEOUT", "5.0"))
        except ValueError:
            timeout = 5.0
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/tags", headers={"User-Agent": "UCX-CLI"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return None
    return None


def _check_vllm_endpoint(url: str, timeout: float | None = None) -> list[str] | None:
    """The models a vLLM endpoint reports at `/v1/models`, or `None` when it did not answer.

    A separate probe from `_check_ollama_endpoint` because the two servers answer different
    paths — `/v1/models` against vLLM's OpenAI-compatible surface, `/api/tags` against
    Ollama's own — and a shared one would have to guess which. `VLLM_API_KEY` is sent only
    when it is set, since `vllm serve --api-key` is optional.

    An empty list means the endpoint answered with no models, which is a different fact from
    `None`; the caller prints them differently, exactly as the Ollama rows do.
    """
    if timeout is None:
        try:
            timeout = float(os.getenv("VLLM_CHECK_TIMEOUT", "5.0"))
        except ValueError:
            timeout = 5.0
    headers = {"User-Agent": "UCX-CLI"}
    api_key = (os.getenv("VLLM_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/models", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return [str(m.get("id", "")) for m in data.get("data", [])]
    except Exception:
        return None
    return None


def _add_vllm_row(table: Table) -> None:
    """Add the vLLM row: unconfigured, unreachable, or online with the model it serves.

    Three states, not two. Ollama's rows have a default endpoint to probe, so "no answer"
    is the only bad news they can carry. vLLM has none — this command refuses to probe a
    port nobody named, because the connection error that came back would describe a server
    the operator never claimed to be running (P6). The row says which variable to set
    instead, which is the only way a reader discovers this provider exists.
    """
    if not has_configured_vllm_endpoint():
        table.add_row(
            "🚀 vLLM",
            f"[dim]unset ({' / '.join(VLLM_ENDPOINT_ENV_VARS)})[/dim]",
            "[dim]— NOT CONFIGURED[/dim]",
            f"[dim]Set {VLLM_ENDPOINT_ENV_VARS[0]}=http://<host>:8000/v1 and "
            f"{VLLM_MODEL_ENV_VAR}=<the --model argument>[/dim]",
        )
        return

    vllm_url = resolve_vllm_base_url()
    models = _check_vllm_endpoint(vllm_url)
    if models is not None:
        table.add_row(
            "🚀 vLLM",
            f"Self-hosted ({vllm_url})",
            "[green]✔ ONLINE[/green]",
            ", ".join(models) if models else "[yellow]Serving no model[/yellow]",
        )
    else:
        table.add_row(
            "🚀 vLLM",
            f"Self-hosted ({vllm_url})",
            "[red]✖ UNREACHABLE[/red]",
            "[dim]Start with 'vllm serve <model> --port <port>'\n"
            "Check that the port matches the endpoint above[/dim]",
        )


@llm_app.command("status")
def llm_status() -> None:
    """Check health and model inventory across Local and Remote (Mac mini) LLM nodes."""
    # A private LAN address as the shipped default made `ucx llm status` probe
    # one particular developer's machine on every other installation. The
    # fast tier is configured, not guessed.
    fast_url = os.getenv("OLLAMA_FAST_BASE_URL", "http://localhost:11434").replace("/v1", "")
    indepth_url = os.getenv("OLLAMA_INDEPTH_BASE_URL", "http://localhost:11434").replace("/v1", "")

    table = Table(title="UClone-X Self-Hosted LLM Topology Status")
    table.add_column("Tier", style="bold cyan")
    table.add_column("Host / Endpoint", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Available Models", style="green")

    # 1. Fast Tier (Mac mini)
    fast_models = _check_ollama_endpoint(fast_url)
    if fast_models is not None:
        table.add_row(
            "⚡ Fast Tier",
            f"Mac mini ({fast_url})",
            "[green]✔ ONLINE[/green]",
            ", ".join(fast_models) if fast_models else "[yellow]No models pulled[/yellow]",
        )
    else:
        table.add_row(
            "⚡ Fast Tier",
            f"Mac mini ({fast_url})",
            "[red]✖ UNREACHABLE[/red]",
            "[dim]Check if 'OLLAMA_HOST=0.0.0.0 ollama serve' is running\n"
            "Check host sleep settings (caffeinate / pmset sleep 0)[/dim]",
        )

    # 2. In-Depth Tier (Local MacBook Pro)
    indepth_models = _check_ollama_endpoint(indepth_url)
    if indepth_models is not None:
        table.add_row(
            "🧠 In-Depth Tier",
            f"Localhost ({indepth_url})",
            "[green]✔ ONLINE[/green]",
            ", ".join(indepth_models) if indepth_models else "[yellow]No models pulled[/yellow]",
        )
    else:
        table.add_row(
            "🧠 In-Depth Tier",
            f"Localhost ({indepth_url})",
            "[red]✖ OFFLINE[/red]",
            "[dim]Start with 'ollama serve'[/dim]",
        )

    # 3. vLLM, when the operator has said where it is.
    _add_vllm_row(table)

    console.print(table)


def _resolved_daemon_url() -> str:
    """The Ollama daemon address, resolved the one documented way.

    This was a second implementation of the resolution order: it read `OLLAMA_HOST` alone
    and supplied the missing scheme itself, so `ucx llm pull` and `ucx llm rm` could not be
    pointed at a daemon named by `OLLAMA_BASE_URL` — the variable every other part of this
    project reads first. `normalize_ollama_base_url` now prepends the scheme, which was the
    only behaviour this function had that `resolve_ollama_base_url` lacked, so keeping a
    separate copy would only preserve the precedence bug.
    """
    return resolve_ollama_base_url()


def _require_ollama_daemon(daemon_url: str, action: str) -> None:
    """Exit(1) with a help message unless the Ollama daemon answers at `daemon_url`.

    Shared by `llm_pull` and `llm_rm` so the reachability guard exists in exactly one
    place — a `Killed by:` declaration anchored on its `if` needs the line to occur once
    in the file, which duplicating this block into both commands would break.
    """
    if _check_ollama_endpoint(daemon_url) is None:
        console.print(
            f"[bold red]Error:[/bold red] Ollama daemon is not responding at {daemon_url}.\n"
            f"Please start the Ollama service before {action}:\n"
            "  • Start daemon: [cyan]ollama serve[/cyan]\n"
            "  • Or launch the Ollama desktop application."
        )
        raise typer.Exit(code=1)


@llm_app.command("pull")
def llm_pull(
    tier: str = typer.Argument(
        "fast",
        help="Tier or model name to pull: 'fast' (qwen3:1.7b), 'indepth' (qwen3:8b), or exact model name",
    ),
) -> None:
    """Download recommended local models using Ollama."""
    target_model = tier
    if tier.lower() == "fast":
        target_model = "qwen3:1.7b"
    elif tier.lower() == "indepth":
        target_model = "qwen3:8b"

    if not shutil.which("ollama"):
        console.print(
            "[bold red]Error:[/bold red] 'ollama' CLI is not installed or not found on PATH.\n"
            "Please install Ollama before pulling models:\n"
            "  • macOS (Homebrew): [cyan]brew install ollama[/cyan]\n"
            "  • Official download: https://ollama.com"
        )
        raise typer.Exit(code=1)

    daemon_url = _resolved_daemon_url()
    _require_ollama_daemon(daemon_url, "pulling models")

    console.print(f"[bold green]Pulling model:[/bold green] [cyan]{target_model}[/cyan]...")
    try:
        cmd = ["ollama", "pull", target_model]
        subprocess.run(cmd, check=True)
        console.print(f"✔ Successfully pulled [bold green]{target_model}[/bold green]")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[bold red]Failed to pull model {target_model}:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@llm_app.command("rm")
def llm_rm(
    model: str = typer.Argument(..., help="Exact model name to remove, e.g. 'llama3.2:1b'"),
) -> None:
    """Remove a locally installed Ollama model.

    Unlike `pull`, this goes over Ollama's HTTP API (`DELETE /api/delete`) rather than
    shelling out to the `ollama` CLI binary — deletion has no equivalent progress output
    worth streaming to a subprocess, so there's nothing the subprocess route buys here.
    """
    daemon_url = _resolved_daemon_url()
    _require_ollama_daemon(daemon_url, "removing models")

    console.print(f"[bold green]Removing model:[/bold green] [cyan]{model}[/cyan]...")
    try:
        asyncio.run(delete_model(model, base_url=daemon_url))
        console.print(f"✔ Successfully removed [bold green]{model}[/bold green]")
    except LLMProviderError as e:
        console.print(f"[bold red]Failed to remove model {model}:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@llm_app.command("use")
def llm_use(
    model: str = typer.Argument(..., help="Model to make the default, e.g. 'qwen3:1.7b'"),
    provider: str = typer.Option(
        "ollama", "--provider", help="Provider that serves it: ollama, vllm, openai, ..."
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Address of the server, when it is not the usual one"
    ),
) -> None:
    """Make MODEL the default for `ucx run`, rooms and the dashboard.

    Saved where the dashboard's Settings keep the choice, so both change the same thing.
    Without `--base-url`, an address already saved for the same provider is kept. A saved
    API key is kept too, for the provider it was saved for.
    """
    from uclone_x.llm.connectors.factory import model_env_override, what_outranks_saved_choice
    from uclone_x.llm.connectors.saved_choice import (
        SAVED_PROVIDERS,
        read_saved_choice,
        save_choice,
        settings_file,
    )

    chosen = provider.strip().lower()
    if chosen not in SAVED_PROVIDERS:
        known = ", ".join(sorted(SAVED_PROVIDERS))
        console.print(
            f"[red]{escape(provider)} is not a provider this version knows: {known}.[/red]"
        )
        raise typer.Exit(code=2)
    try:
        before = read_saved_choice()
        address = base_url
        if address is None and before is not None and before.provider == chosen:
            address = before.base_url
        save_choice(provider=chosen, model=model, base_url=address)
    except (OSError, ValueError):
        console.print(
            f"[red]Could not save {escape(model)} as the default: {escape(str(settings_file()))} "
            "could not be read or written. Choose it in the dashboard's Settings instead, "
            f"or pass --model {escape(model)} to each command.[/red]"
        )
        raise typer.Exit(code=1) from None
    # Saved is not the same as in use: a variable outranks the file, so say which one.
    winner = what_outranks_saved_choice()
    model_variable = model_env_override(chosen)
    if winner is not None:
        console.print(
            f"[yellow]Saved {escape(model)} ({chosen}) as the default, but it is not used yet: "
            f"{winner} is set in this shell, and it takes priority over the saved default. "
            f"Unset {winner} to use {escape(model)}.[/yellow]"
        )
    elif model_variable is not None:
        console.print(
            f"[yellow]Saved {escape(model)} ({chosen}) as the default, but {model_variable} is "
            f"set in this shell and names another model, which is used instead. "
            f"Unset {model_variable} to use {escape(model)}.[/yellow]"
        )
    else:
        console.print(
            f"[green]✔ {escape(model)} ({chosen}) is now the default model for `ucx run`, "
            "rooms and the dashboard.[/green]"
        )
