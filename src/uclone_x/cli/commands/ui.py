"""UI command group for starting and stopping the developer dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from uclone_x.errors import (
    DashboardNotIdentifiedError,
    DashboardStopUnconfirmedError,
    ListeningProcessLookupError,
)
from uclone_x.shells import ui_process
from uclone_x.shells.ui_process import MAX_TCP_PORT as _MAX_TCP_PORT
from uclone_x.shells.ui_process import MIN_TCP_PORT as _MIN_TCP_PORT

console = Console()

ui_app = typer.Typer(
    name="ui",
    help="Launch or manage the Developer Dashboard (React GUI + FastAPI Backend).",
    invoke_without_command=True,
)


@ui_app.callback(invoke_without_command=True)
def ui_start(
    ctx: typer.Context,
    port: int = typer.Option(5180, "--port", "-p", help="Port to bind the developer GUI backend"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind"),
    dev: bool = typer.Option(
        False, "--dev", "-d", help="Enable hot-reload development mode for fast iteration"
    ),
    vite_port: int = typer.Option(
        5173, "--vite-port", help="Port to bind the Vite HMR dev server in dev mode"
    ),
    cwd: Annotated[
        Path | None,
        typer.Option(
            "--cwd",
            "-w",
            help="Root workspace directory for agent tool operations (default: current directory)",
        ),
    ] = None,
) -> None:
    """Launch the Developer Dashboard (React GUI + FastAPI Backend)."""
    if ctx.invoked_subcommand is not None:
        return

    # Deferred: `uclone_x.ui` requires the `http` extra, and `cli/main.py` imports
    # this module at module scope. At the top it made `ucx --help` require FastAPI,
    # so a `uclone-x[cli]` install could reach no command at all.
    import uclone_x.cli.main as cli_main
    from uclone_x.ui import server as ui_server

    effective_cwd = cwd.resolve() if cwd is not None else Path.cwd().resolve()
    bound_port = cli_main.find_available_port(port)
    if bound_port != port:
        console.print(
            f"[yellow]⚠️ Port {port} is in use. Automatically falling back to port {bound_port}.[/yellow]"
        )
    # The backend port was probed and the Vite port was not, so a collision degraded
    # only the half that had no fallback. Probing is necessary and not sufficient here:
    # a wildcard bind by another process can leave `127.0.0.1:<port>` bindable while
    # still shadowing `localhost`, which is why `start_ui_server` also verifies after
    # start that the server answering is ours.
    bound_vite_port = cli_main.find_available_port(vite_port) if dev else vite_port
    if dev and bound_vite_port != vite_port:
        console.print(
            f"[yellow]⚠️ Vite port {vite_port} is in use. Falling back to {bound_vite_port}.[/yellow]"
        )
    ui_server.start_ui_server(
        port=bound_port,
        dev=dev,
        host=host,
        vite_port=bound_vite_port,
        workspace_dir=effective_cwd,
    )


@ui_app.command("stop")
def ui_stop(
    port: int = typer.Option(
        5180,
        "--port",
        "-p",
        min=_MIN_TCP_PORT,
        max=_MAX_TCP_PORT,
        help="Port of the developer GUI backend to stop",
    ),
) -> None:
    """Stop the UClone-X dashboard on the target port, and nothing else.

    Only the process that launched a dashboard with `ucx ui` or `ucx start`, as recorded
    when it started, is signalled. Anything else on the port is left alone, with the reason.
    """
    # #613, #927. `ui_process`, never `uclone_x.ui.server`: that package requires the `http`
    # extra, and signalling a PID needs none of it (#881). Pinned by the `ui stop` rows of
    # the distribution-install fitness lane.
    try:
        stopped = ui_process.stop_ui_server(port=port)
    except ListeningProcessLookupError as exc:
        # Not "nothing found": the port was never inspected, so the outcome is unknown.
        typer.echo(f"ucx ui stop: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except DashboardNotIdentifiedError as exc:
        # Something is there and it is not provably ours: refuse rather than guess.
        typer.echo(f"ucx ui stop: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except DashboardStopUnconfirmedError as exc:
        # SIGTERM went out; what followed is unknown, and the message must not deny the signal.
        typer.echo(f"ucx ui stop: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    record_path = escape(str(ui_process.dashboard_record_path(port)))
    if stopped.stale_pid is not None:
        console.print(
            f"[yellow]Removed a stale dashboard record: PID {stopped.stale_pid} "
            f"{stopped.stale_reason}.[/yellow]",
            soft_wrap=True,
        )
    if stopped.killed:
        # Not "Stopped": the dashboard did not shut down, it was killed, and under `--dev` a
        # worker it would have taken down may still be serving (P6).
        pid_list = ", ".join(map(str, stopped.stopped_pids))
        typer.echo(
            f"ucx ui stop: PID(s) {pid_list} did not exit within "
            f"{ui_process.TERMINATE_WAIT_S:g} s of SIGTERM and were sent SIGKILL (port {port}); "
            f"the dashboard's graceful shutdown did not complete.",
            err=True,
        )
        if stopped.still_accepting is not False:
            state = (
                "still accepts connections"
                if stopped.still_accepting
                else "could not be checked afterwards"
            )
            typer.echo(
                f"ucx ui stop: port {port} {state}; a `--dev` reload worker may have been "
                f"orphaned. Nothing further was signalled.",
                err=True,
            )
        # Exit 1 even when the port is free: a killed dashboard may have lost a turn its
        # shutdown would have saved, and a script calling `stop` must be able to tell.
        raise typer.Exit(code=1)
    if stopped.stopped_pids:
        pid_list = ", ".join(map(str, stopped.stopped_pids))
        console.print(
            f"[bold green]✔ Stopped UI dashboard server running on PID(s): {pid_list} (port {port}).[/bold green]",
            soft_wrap=True,
        )
    else:
        # Say what was looked at: a dashboard bound only to a LAN address, or recorded under
        # another UCLONE_UI_STATE_DIR, is invisible to both checks (P6).
        console.print(
            f"[dim]No active UI dashboard server found on port {port}: no dashboard record at "
            f"{record_path}, and nothing accepts connections on 127.0.0.1 or ::1. A dashboard "
            f"bound only to another address is not detected.[/dim]",
            soft_wrap=True,
        )
