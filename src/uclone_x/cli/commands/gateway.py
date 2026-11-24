"""CLI commands for running and inspecting the standalone Model Gateway server."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

from uclone_x.errors import LLMError
from uclone_x.llm.connectors.gateway_client import (
    DEFAULT_GATEWAY_URL,
    GatewayModelClient,
    resolve_gateway_base_url,
)

gateway_app = typer.Typer(
    name="gateway",
    help="Launch and inspect the standalone high-speed Model Gateway server",
    no_args_is_help=True,
)
console = Console()


@gateway_app.command("serve")
def gateway_serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind"),
    port: int = typer.Option(8765, "--port", "-p", help="TCP port to bind the Model Gateway"),
    settings_file: Annotated[
        Path | None,
        typer.Option(
            "--settings-file",
            help="Path to settings.json storing connections, default models, and usage limits",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="Optional shared bearer token required on incoming gateway requests",
        ),
    ] = None,
    remote_host: Annotated[
        str | None,
        typer.Option(
            "--remote-host",
            "-r",
            help="SSH host (e.g. dell or mac-mini) to auto-tunnel and connect at gateway startup",
        ),
    ] = None,
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level"),
) -> None:
    """Start the standalone Model Gateway HTTP/SSE server."""
    # Deferred: `uclone_x.shells.gateway_server` requires the `http` extra (FastAPI + Uvicorn).
    from uclone_x.shells.gateway_server import run_gateway_server

    console.print(f"[bold cyan]Starting UClone-X Model Gateway on http://{host}:{port}[/bold cyan]")
    run_gateway_server(
        host=host,
        port=port,
        settings_path=settings_file,
        api_key=api_key,
        remote_host=remote_host,
        log_level=log_level,
    )


def _render_remote_gpu_status(remote_payload: Mapping[str, Any]) -> None:
    """Render Remote GPU tunnel state, port mappings, or failure diagnostics."""
    connected = bool(remote_payload.get("connected"))
    reconnecting = bool(remote_payload.get("reconnecting"))
    host = str(remote_payload.get("host") or remote_payload.get("desired_host") or "").strip()
    err = str(remote_payload.get("error") or "").strip()

    if connected:
        pid = remote_payload.get("pid")
        gpu = remote_payload.get("gpu")
        gpu_label = ""
        if isinstance(gpu, Mapping):
            gpu_map = cast(Mapping[str, Any], gpu)
            name = gpu_map.get("name")
            vram = gpu_map.get("total_mb")
            if name:
                gpu_label = f" — {name}" + (f" ({vram} MB)" if vram is not None else "")
        mappings_raw = remote_payload.get("mappings")
        forwards: list[str] = []
        if isinstance(mappings_raw, list):
            for m_obj in cast(list[object], mappings_raw):
                if isinstance(m_obj, Mapping):
                    m = cast(Mapping[str, Any], m_obj)
                    forwards.append(
                        f"{m.get('service_name')}:127.0.0.1:{m.get('local_port')}->{m.get('remote_port')}"
                    )
        fwd_str = f" [{', '.join(forwards)}]" if forwards else ""
        console.print(
            f"[bold green]✔ Remote GPU connected:[/bold green] {host or 'remote'}"
            f" (pid={pid}){gpu_label}{fwd_str}"
        )
        return

    if reconnecting and host:
        console.print(f"[yellow]⏳ Remote GPU connecting:[/yellow] {host}...")
        return

    if host:
        auto_rec = " (auto-reconnect enabled)" if remote_payload.get("auto_reconnect") else ""
        reason = err if err and err != "No active remote tunnel" else "disconnected"
        console.print(
            f"[yellow]⚠ Remote GPU ({host}) not connected{auto_rec}:[/yellow] {reason}\n"
            f"  [dim]Hint: check SSH reachability with `ucx gateway probe {host}` or retry with `ucx gateway connect {host}`[/dim]"
        )
        return

    console.print(
        "[dim]Remote GPU: not configured (connect with `ucx gateway connect <ssh-host>`)[/dim]"
    )


@gateway_app.command("status")
def gateway_status(
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            help=f"Gateway base URL (default: UCX_GATEWAY_URL or {DEFAULT_GATEWAY_URL})",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Gateway API key (default: UCX_GATEWAY_KEY)"),
    ] = None,
) -> None:
    """Inspect connected servers, remote GPU tunnel, default model slots, and quota windows."""
    resolved_url = resolve_gateway_base_url(url)

    async def _probe() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        client = GatewayModelClient(base_url=resolved_url, api_key=api_key)
        try:
            conns = await client.list_connections()
            usage = await client.get_usage()
            try:
                remote = await client.remote_gpu_status()
            except LLMError:
                remote = {}
            return conns, usage, remote
        finally:
            await client.aclose()

    try:
        conns_payload, usage_payload, remote_payload = asyncio.run(_probe())
    except LLMError as exc:
        console.print(f"[red]✖ Gateway unreachable at {resolved_url}:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[bold green]✔ Model Gateway reachable at {resolved_url}[/bold green]")
    if remote_payload:
        _render_remote_gpu_status(remote_payload)
    connections = conns_payload.get("connections", [])
    if isinstance(connections, list) and connections:
        table = Table(title="Configured Connections")
        table.add_column("ID", style="cyan")
        table.add_column("Kind")
        table.add_column("Label")
        table.add_column("Status")
        table.add_column("Endpoint / Detail")
        for raw_row in cast(list[object], connections):
            if not isinstance(raw_row, Mapping):
                continue
            row = cast(Mapping[str, Any], raw_row)
            status_obj: object = row.get("status")
            if isinstance(status_obj, Mapping):
                status_map = cast(Mapping[str, Any], status_obj)
                state = str(status_map.get("state") or "unknown")
                detail = str(status_map.get("detail") or row.get("base_url") or "")
            else:
                state = str(status_obj or "unknown")
                detail = str(row.get("detail") or row.get("base_url") or "")
            table.add_row(
                str(row.get("id") or ""),
                str(row.get("kind") or ""),
                str(row.get("label") or ""),
                state,
                detail,
            )
        console.print(table)

    windows = usage_payload.get("windows", [])
    if isinstance(windows, list) and windows:
        w_table = Table(title="Quota Windows")
        w_table.add_column("Window", style="magenta")
        w_table.add_column("Used Tokens", justify="right")
        w_table.add_column("Limit", justify="right")
        w_table.add_column("Remaining", justify="right")
        for raw_win in cast(list[object], windows):
            if not isinstance(raw_win, Mapping):
                continue
            win = cast(Mapping[str, Any], raw_win)
            used_raw = win.get("used")
            used_val = int(used_raw) if isinstance(used_raw, int) else 0
            limit_val: object = win.get("limit")
            rem_val: object = win.get("remaining")
            if rem_val is None and isinstance(limit_val, int):
                rem_val = max(0, limit_val - used_val)
            w_table.add_row(
                str(win.get("window") or ""),
                str(used_val),
                str(limit_val) if limit_val is not None else "unlimited",
                str(rem_val) if rem_val is not None else "unlimited",
            )
        console.print(w_table)


@gateway_app.command("probe")
def gateway_probe(
    host: Annotated[str, typer.Argument(help="SSH host alias or address (e.g. dell, mac-mini)")],
    timeout: float = typer.Option(6.0, "--timeout", help="SSH probe timeout in seconds"),
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            help=f"Gateway base URL (default: UCX_GATEWAY_URL or {DEFAULT_GATEWAY_URL})",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Gateway API key (default: UCX_GATEWAY_KEY)"),
    ] = None,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output"),
) -> None:
    """Probe an SSH remote host for GPU/Metal hardware, Ollama models, and ComfyUI."""
    resolved_url = resolve_gateway_base_url(url)

    async def _run() -> dict[str, Any]:
        client = GatewayModelClient(base_url=resolved_url, api_key=api_key)
        try:
            return await client.probe_remote_gpu(host.strip(), timeout=timeout)
        finally:
            await client.aclose()

    try:
        res = asyncio.run(_run())
    except LLMError as exc:
        console.print(f"[red]✖ Gateway unreachable at {resolved_url}:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(json.dumps(res))
        if not bool(res.get("reachable")):
            raise typer.Exit(code=1)
        return

    if not bool(res.get("reachable")):
        console.print(f"[red]✖ Remote host '{host}' unreachable:[/red] {res.get('error')}")
        raise typer.Exit(code=1)

    gpu = res.get("gpu")
    gpu_desc = "none detected"
    if isinstance(gpu, Mapping):
        g = cast(Mapping[str, Any], gpu)
        gpu_desc = f"{g.get('name')} ({g.get('total_mb')} MB, driver {g.get('driver')})"
    console.print(f"[bold green]✔ Remote host '{host}' reachable[/bold green] — GPU: {gpu_desc}")
    ports = res.get("ports")
    if isinstance(ports, Mapping):
        for svc_name, svc_info in cast(Mapping[str, Any], ports).items():
            if isinstance(svc_info, Mapping):
                info = cast(Mapping[str, Any], svc_info)
                state = "listening" if info.get("listening") else "stopped"
                console.print(f"  • {svc_name} (port {info.get('port')}): {state}")
    models = res.get("ollama_models")
    if isinstance(models, list) and models:
        console.print(f"  • Ollama models: {', '.join(str(m) for m in cast(list[object], models))}")


@gateway_app.command("connect")
def gateway_connect(
    host: Annotated[
        str | None,
        typer.Argument(
            help="SSH host alias (e.g. dell). Omit to reconnect saved target or UCX_REMOTE_HOST."
        ),
    ] = None,
    auto_start_comfyui: bool = typer.Option(
        True,
        "--comfyui/--no-comfyui",
        help="Auto-start ComfyUI on the remote host if installed",
    ),
    sync_llm: bool = typer.Option(
        True,
        "--sync-llm/--no-sync-llm",
        help="Register the tunnel's Ollama as the `remote-gpu` chat connection",
    ),
    timeout: float = typer.Option(15.0, "--timeout", help="Connection timeout in seconds"),
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            help=f"Gateway base URL (default: UCX_GATEWAY_URL or {DEFAULT_GATEWAY_URL})",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Gateway API key (default: UCX_GATEWAY_KEY)"),
    ] = None,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output"),
) -> None:
    """Connect (or reconnect) the running Model Gateway to a remote GPU host over SSH."""
    resolved_url = resolve_gateway_base_url(url)

    async def _run() -> tuple[str, dict[str, Any], list[str]]:
        client = GatewayModelClient(base_url=resolved_url, api_key=api_key)
        try:
            target = (host or "").strip() or os.environ.get("UCX_REMOTE_HOST", "").strip()
            if not target:
                st = await client.remote_gpu_status()
                target = str(st.get("desired_host") or st.get("host") or "").strip()
            if not target:
                hosts_res = await client.remote_gpu_hosts()
                raw_hosts = hosts_res.get("hosts")
                candidates = (
                    [str(h) for h in cast(list[object], raw_hosts)]
                    if isinstance(raw_hosts, list)
                    else []
                )
                return "", {}, candidates
            res = await client.connect_remote_gpu(
                target,
                auto_start_comfyui=auto_start_comfyui,
                sync_llm=sync_llm,
                timeout=timeout,
            )
            return target, res, []
        finally:
            await client.aclose()

    try:
        target_host, res, candidates = asyncio.run(_run())
    except LLMError as exc:
        console.print(f"[red]✖ Gateway unreachable at {resolved_url}:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not target_host:
        hint = (
            f" Discovered SSH hosts: {', '.join(candidates)}"
            if candidates
            else " Configure a host in ~/.ssh/config first."
        )
        console.print(
            f"[red]✖ No remote GPU host specified and no saved session found.[/red]{hint}\n"
            "  Usage: ucx gateway connect <host> (e.g. `ucx gateway connect dell`)"
        )
        raise typer.Exit(code=1)

    if as_json:
        console.print_json(json.dumps(res))
        if not res.get("connected"):
            raise typer.Exit(code=1)
        return

    if not bool(res.get("connected")):
        console.print(
            f"[red]✖ Failed to connect gateway to '{target_host}':[/red] {res.get('error')}\n"
            f"  [dim]Target '{target_host}' is saved on the gateway and will auto-reconnect when reachable.[/dim]"
        )
        raise typer.Exit(code=1)

    tunnel = res.get("tunnel")
    if isinstance(tunnel, Mapping):
        _render_remote_gpu_status({**cast(Mapping[str, Any], tunnel), "connected": True})
    else:
        console.print(f"[bold green]✔ Connected gateway to {target_host}[/bold green]")


@gateway_app.command("disconnect")
def gateway_disconnect(
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            help=f"Gateway base URL (default: UCX_GATEWAY_URL or {DEFAULT_GATEWAY_URL})",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Gateway API key (default: UCX_GATEWAY_KEY)"),
    ] = None,
) -> None:
    """Disconnect the gateway's active remote GPU tunnel and clear auto-reconnect state."""
    resolved_url = resolve_gateway_base_url(url)

    async def _run() -> dict[str, Any]:
        client = GatewayModelClient(base_url=resolved_url, api_key=api_key)
        try:
            return await client.disconnect_remote_gpu()
        finally:
            await client.aclose()

    try:
        res = asyncio.run(_run())
    except LLMError as exc:
        console.print(f"[red]✖ Gateway unreachable at {resolved_url}:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    restored = res.get("restored_settings")
    note = f" (removed tunnel connections: {restored})" if restored else ""
    console.print(f"[bold green]✔ Disconnected remote GPU tunnel{note}[/bold green]")
