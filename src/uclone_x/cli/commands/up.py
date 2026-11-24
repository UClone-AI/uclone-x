"""Top-level declarative stack commands: `ucx up`, `ucx down`, and `ucx status --stack`.

Provides Terraform-style `plan` / `apply` incremental convergence across the 5-component
UClone-X cluster topology (`docker_infra`, `frontend_bundle`, `model_gateway`,
`agent_workers`, `web_ui_server`).

Keep module-level imports limited to the `cli` extra (`typer`, `rich`) and standard library so
that `pip install uclone-x[cli]` never requires `fastapi` (`http`) or `aiokafka` (`cluster`)
just to load `ucx --help`.
"""

from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from uclone_x.shells.reconciler import PlanAction, StackApplyResult, StackPlan

__all__ = ["down_command", "render_stack_status", "up_command"]

_ACTION_SYMBOL: dict[str, str] = {
    "create": "[bold green]+ create[/bold green]",
    "update": "[bold yellow]~ update[/bold yellow]",
    "replace": "[bold magenta]-/+ replace[/bold magenta]",
    "unchanged": "[dim]= unchanged[/dim]",
}


def _render_plan_table(plan: StackPlan) -> Table:
    table = Table(
        title=f"UClone-X Declarative Stack Plan (inspected in {plan.inspection_ms:.1f} ms)"
    )
    table.add_column("Action", no_wrap=True)
    table.add_column("Resource", style="bold cyan", no_wrap=True)
    table.add_column("Live", no_wrap=True)
    table.add_column("Endpoint / Target", style="dim")
    table.add_column("Reason / Drift")

    for res in plan.resources:
        live_badge = "[green]healthy[/green]" if res.live_healthy else "[yellow]down/stale[/yellow]"
        table.add_row(
            _ACTION_SYMBOL.get(res.action, res.action),
            res.name,
            live_badge,
            res.endpoint,
            res.reason,
        )
    return table


def _plan_summary_line(counts: dict[PlanAction, int]) -> str:
    return (
        f"Plan: [bold green]{counts['create']} to create[/bold green], "
        f"[bold yellow]{counts['update']} to update[/bold yellow], "
        f"[bold magenta]{counts['replace']} to replace[/bold magenta], "
        f"[dim]{counts['unchanged']} unchanged[/dim]."
    )


def _render_apply_table(apply_result: StackApplyResult) -> Table:
    table = Table(title="Declarative Apply Summary")
    table.add_column("Resource", style="bold cyan", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Duration (ms)", justify="right")
    table.add_column("Detail")

    for item in apply_result.results:
        status_str = (
            "[bold green]READY[/bold green]" if item.healthy else "[bold red]FAILED[/bold red]"
        )
        table.add_row(
            item.name,
            _ACTION_SYMBOL.get(item.action, item.action),
            status_str,
            f"{item.duration_ms:.1f}",
            item.detail,
        )
    return table


def _serialize_plan(plan: StackPlan) -> dict[str, Any]:
    counts = plan.counts()
    return {
        "has_changes": plan.has_changes,
        "inspection_ms": plan.inspection_ms,
        "counts": dict(counts),
        "resources": [r.model_dump(mode="json") for r in plan.resources],
    }


def up_command(
    plan_only: bool = typer.Option(
        False,
        "--plan",
        help="Compute and display the Terraform-style execution plan without applying changes",
    ),
    detailed_exitcode: bool = typer.Option(
        False,
        "--detailed-exitcode",
        help="With --plan, return exit code 0 when unchanged, 2 when changes are pending, 1 on error",
    ),
    workers: int = typer.Option(
        2,
        "--workers",
        "-w",
        min=0,
        max=64,
        help="Desired number of stateless background worker daemons (`ucx worker`)",
    ),
    remote_host: str | None = typer.Option(
        None,
        "--remote-host",
        "-r",
        help="Optional remote GPU SSH host (e.g. 'dell', 'macmini') managed by model_gateway",
    ),
    gateway_port: int = typer.Option(
        8765,
        "--gateway-port",
        min=1,
        max=65535,
        help="Port for the standalone Model Gateway (`ucx gateway serve`)",
    ),
    gateway_host: str = typer.Option(
        "127.0.0.1",
        "--gateway-host",
        help="Bind host for the standalone Model Gateway",
    ),
    port: int = typer.Option(
        5180,
        "--port",
        "-p",
        min=1,
        max=65535,
        help="Port for the cluster-backed Web UI & Room server",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Bind host for the cluster-backed Web UI & Room server",
    ),
    cwd: Annotated[
        Path | None,
        typer.Option(
            "--cwd",
            help="Optional workspace root directory passed to stateless workers and Web UI",
        ),
    ] = None,
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Run post-apply HTTP readiness checks on Model Gateway and Cluster Web UI",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Automatically open the Web UI URL in the default browser once healthy",
    ),
    wait_seconds: int = typer.Option(
        60,
        "--wait-seconds",
        min=1,
        help="Maximum seconds to wait for each stack resource to report healthy",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON on stdout (progress notices go to stderr)",
    ),
) -> None:
    """Converge the 5-component UClone-X cluster stack incrementally (reload only what changed)."""
    from uclone_x.adapters.cluster.config import ClusterConfigError
    from uclone_x.errors import MissingDependencyError
    from uclone_x.shells.reconciler import ReconcileError, StackReconciler

    out = Console(stderr=True) if as_json else Console()
    try:
        reconciler = StackReconciler.from_environment(
            ui_host=host,
            ui_port=port,
            gateway_host=gateway_host,
            gateway_port=gateway_port,
            remote_host=remote_host,
            workers=workers,
            workspace_dir=cwd,
            wait_seconds=wait_seconds,
        )
        computed_plan = reconciler.plan()
    except (ClusterConfigError, MissingDependencyError) as exc:
        if as_json:
            sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}, indent=2) + "\n")
        else:
            out.print(f"[bold red]✖ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    if plan_only:
        if as_json:
            sys.stdout.write(json.dumps(_serialize_plan(computed_plan), indent=2) + "\n")
        else:
            out.print(_render_plan_table(computed_plan))
            out.print(_plan_summary_line(computed_plan.counts()))
        if detailed_exitcode and computed_plan.has_changes:
            raise typer.Exit(code=2)
        return

    out.print(_render_plan_table(computed_plan))
    out.print(_plan_summary_line(computed_plan.counts()))

    if not computed_plan.has_changes:
        if verify:
            checks, checks_ok = reconciler._run_post_apply_verification()  # pyright: ignore[reportPrivateUsage]
        else:
            checks, checks_ok = (), True
        if as_json:
            payload = {
                "all_healthy": checks_ok,
                "changed": False,
                "ui_url": reconciler.spec.ui_url,
                "gateway_url": reconciler.spec.gateway_url,
                "plan": _serialize_plan(computed_plan),
                "verification_checks": list(checks),
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            out.print(
                Panel(
                    f"[bold green]✔ Stack already converged — 0 changes applied.[/bold green]\n"
                    f"  • [bold]Web Dashboard[/bold] : [cyan]{reconciler.spec.ui_url}[/cyan]\n"
                    f"  • [bold]Model Gateway[/bold] : [cyan]{reconciler.spec.gateway_url}[/cyan]",
                    border_style="green",
                )
            )
        if not checks_ok:
            raise typer.Exit(code=1)
        if open_browser:
            webbrowser.open(reconciler.spec.ui_url)
        return

    try:
        apply_result = reconciler.apply(precomputed_plan=computed_plan, verify=verify)
    except (ReconcileError, ClusterConfigError, MissingDependencyError) as exc:
        if as_json:
            sys.stdout.write(
                json.dumps(
                    {
                        "all_healthy": False,
                        "error": str(exc),
                        "plan": _serialize_plan(computed_plan),
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            out.print(f"[bold red]✖ Apply failed: {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    if as_json:
        payload = {
            "all_healthy": apply_result.all_healthy,
            "changed": True,
            "ui_url": apply_result.ui_url,
            "gateway_url": apply_result.gateway_url,
            "plan": _serialize_plan(apply_result.plan),
            "results": [r.model_dump(mode="json") for r in apply_result.results],
            "verification_checks": list(apply_result.verification_checks),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        out.print(_render_apply_table(apply_result))
        status_line = (
            "[bold green]✔ Declarative Stack Converged & Verified[/bold green]"
            if apply_result.all_healthy
            else "[bold red]✖ Stack converged with failing verification check(s)[/bold red]"
        )
        out.print(
            Panel(
                f"{status_line}\n"
                f"  • [bold]Web Dashboard[/bold]  : [cyan]{apply_result.ui_url}[/cyan]\n"
                f"  • [bold]Model Gateway[/bold]  : [cyan]{apply_result.gateway_url}[/cyan]\n"
                f"  • [bold]Check Drift[/bold]    : [dim]ucx up --plan[/dim]\n"
                f"  • [bold]Stop Stack[/bold]     : [dim]ucx down[/dim]",
                border_style="green" if apply_result.all_healthy else "red",
            )
        )

    if not apply_result.all_healthy:
        raise typer.Exit(code=1)
    if open_browser:
        webbrowser.open(apply_result.ui_url)


def down_command(
    volumes: bool = typer.Option(
        False,
        "--volumes",
        "-v",
        help="Also remove Docker volumes (`docker compose down --volumes`) and deploy-state.json",
    ),
    port: int = typer.Option(
        5180,
        "--port",
        "-p",
        min=1,
        max=65535,
        help="Port of the cluster Web UI server to stop",
    ),
    gateway_port: int = typer.Option(
        8765,
        "--gateway-port",
        min=1,
        max=65535,
        help="Port of the standalone Model Gateway to stop",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON on stdout",
    ),
) -> None:
    """Stop Web UI, stateless workers, Model Gateway, and Docker cluster services."""
    from uclone_x.adapters.cluster.config import ClusterConfigError
    from uclone_x.shells.reconciler import StackReconciler

    out = Console(stderr=True) if as_json else Console()
    try:
        reconciler = StackReconciler.from_environment(
            ui_port=port,
            gateway_port=gateway_port,
        )
    except ClusterConfigError as exc:
        if as_json:
            sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}, indent=2) + "\n")
        else:
            out.print(f"[bold red]✖ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    result = reconciler.down(volumes=volumes)
    if as_json:
        sys.stdout.write(json.dumps(result.model_dump(mode="json"), indent=2) + "\n")
        return

    for warn in result.warnings:
        out.print(f"[yellow]⚠️ {warn}[/yellow]")
    out.print(
        Panel(
            f"[bold green]✔ UClone-X Declarative Stack Stopped[/bold green]\n"
            f"  • [bold]Web UI PIDs stopped[/bold]      : {list(result.ui_stopped_pids)}\n"
            f"  • [bold]Worker PIDs stopped[/bold]      : {list(result.worker_stopped_pids)}\n"
            f"  • [bold]Gateway PIDs stopped[/bold]     : {list(result.gateway_stopped_pids)}\n"
            f"  • [bold]Docker cluster stopped[/bold]   : {result.docker_stopped}"
            + (" (volumes removed)" if result.volumes_removed else ""),
            border_style="green",
        )
    )


def render_stack_status(
    *,
    port: int = 5180,
    gateway_port: int = 8765,
    workers: int = 2,
    remote_host: str | None = None,
    json_output: bool = False,
) -> None:
    """Inspect live health and code/config drift across all 5 declarative stack resources."""
    from uclone_x.adapters.cluster.config import ClusterConfigError
    from uclone_x.shells.reconciler import StackReconciler

    out = Console(stderr=True) if json_output else Console()
    try:
        reconciler = StackReconciler.from_environment(
            ui_port=port,
            gateway_port=gateway_port,
            workers=workers,
            remote_host=remote_host,
        )
        computed_plan = reconciler.plan()
    except ClusterConfigError as exc:
        if json_output:
            sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}, indent=2) + "\n")
        else:
            out.print(f"[bold red]✖ {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    all_healthy = all(r.live_healthy for r in computed_plan.resources)
    in_sync = not computed_plan.has_changes
    if json_output:
        payload = {
            "all_healthy": all_healthy,
            "in_sync": in_sync,
            "ui_url": reconciler.spec.ui_url,
            "gateway_url": reconciler.spec.gateway_url,
            "resources": [
                {
                    "name": r.name,
                    "live_healthy": r.live_healthy,
                    "drift_state": "in-sync" if r.action == "unchanged" else f"drifted:{r.action}",
                    "planned_action": r.action,
                    "endpoint": r.endpoint,
                    "pids": list(r.pids),
                    "reason": r.reason,
                }
                for r in computed_plan.resources
            ],
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return

    table = Table(title="UClone-X Declarative Stack Status & Drift")
    table.add_column("Resource", style="bold cyan", no_wrap=True)
    table.add_column("Live Health", no_wrap=True)
    table.add_column("Code/Config Drift", no_wrap=True)
    table.add_column("Endpoint / Target", style="dim")
    table.add_column("Detail")

    for res in computed_plan.resources:
        health_badge = (
            "[bold green]HEALTHY[/bold green]"
            if res.live_healthy
            else "[bold red]OFFLINE[/bold red]"
        )
        drift_badge = (
            "[green]in-sync[/green]"
            if res.action == "unchanged"
            else f"[yellow]drifted ({res.action})[/yellow]"
        )
        table.add_row(res.name, health_badge, drift_badge, res.endpoint, res.reason)

    out.print(table)
    if not in_sync:
        out.print("[dim]Run `ucx up` to reconcile drifted or offline components.[/dim]")
