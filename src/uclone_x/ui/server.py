"""Server launcher for the UClone-X developer UI dashboard."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from types import FrameType
from typing import Any

try:
    import uvicorn
    from rich.console import Console
except ImportError as exc:
    pkg = "uvicorn" if "uvicorn" in str(exc) else "rich"
    extra = "http" if pkg == "uvicorn" else "cli"
    from uclone_x.errors import MissingDependencyError

    raise MissingDependencyError(
        extra=extra,
        package=pkg,
        feature="UI server runner",
    ) from exc

from uclone_x import __version__
from uclone_x.agent.session import SESSION_STORAGE_DIR_ENV_VAR
from uclone_x.errors import DashboardNotIdentifiedError, FrontendBuildFailedError
from uclone_x.shells import ui_process
from uclone_x.shells.ui_process import UI_BIND_HOST_ENV_VAR

# Re-exported, not defined here: `ucx ui stop` must not import this module, whose package
# requires the `http` extra, so the one implementation lives in the shell layer (#881).
from uclone_x.shells.ui_process import stop_ui_server as stop_ui_server
from uclone_x.ui import build_inputs
from uclone_x.ui.app import create_ui_app, get_git_commit

console = Console()

DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_S: int = 2

if not getattr(uvicorn.Server.handle_exit, "_is_uclone_hook", False):
    _orig_handle_exit = uvicorn.Server.handle_exit

    def _shutdown_hook_handle_exit(self: uvicorn.Server, sig: int, frame: FrameType | None) -> None:
        from uclone_x.ui.app import trigger_ui_shutdown

        trigger_ui_shutdown()
        _orig_handle_exit(self, sig, frame)

    _shutdown_hook_handle_exit._is_uclone_hook = True  # pyright: ignore[reportFunctionMemberAccess]
    uvicorn.Server.handle_exit = _shutdown_hook_handle_exit


def _should_rebuild_frontend(frontend_dir: Path, static_dir: Path) -> bool:
    """Whether the committed bundle in `static_dir` is not what `frontend_dir` builds.

    A **content** question answered with content (#1075). The digest of the real build
    inputs is compared with the digest recorded beside the bundle by the build that
    produced it; `uclone_x.ui.build_inputs` says what is hashed and what is not. Before
    that, two structural cases: a bundle with no `index.html` or no `assets/` is not a
    bundle, whatever any record claims.

    The `st_size < 10000` check this replaces is gone rather than re-based. It asked a
    content question of a proxy, and it could not converge: had a legitimate build ever
    emitted CSS under 10 KB, every launch would have rebuilt and every rebuild would have
    re-emitted the same small file.
    """
    if not (static_dir / "index.html").is_file() or not (static_dir / "assets").is_dir():
        return True

    recorded = build_inputs.recorded_digest(static_dir.parent)
    if recorded is None:
        return True
    return recorded != build_inputs.build_input_digest(frontend_dir)


def _ensure_frontend_built() -> None:
    """Build the dashboard bundle if the committed one is not what the source builds.

    A failed build raises (P6). It used to be caught by a bare `except Exception` that
    printed a warning and let the launcher serve the **stale** bundle: a UI that looks
    fine and is not the code the user is running. Only the two exceptions a failed
    `subprocess.run` actually raises are caught here, and both are re-raised as
    `FrontendBuildFailedError`. Nothing wider may be caught: `tests/support/`'s
    `RealFrontendBuildInTestError` derives from `BaseException` *because* the old handler
    would have swallowed it (#1074), so a handler widened back to `Exception` — never mind
    `BaseException` — silently re-opens the hole this function is being fixed to close.
    """
    static_dir = Path(__file__).resolve().parent.parent / "ui_static"
    frontend_dir = Path(__file__).resolve().parents[3] / "frontend"

    if frontend_dir.is_dir() and (frontend_dir / "package.json").is_file():
        if _should_rebuild_frontend(frontend_dir, static_dir):
            console.print(
                "[bold cyan]📦 Preparing dashboard: building frontend assets...[/bold cyan]"
            )
            import subprocess

            try:
                subprocess.run(
                    ["npm", "run", "build"],
                    cwd=str(frontend_dir),
                    check=True,
                    capture_output=True,
                )
            except (subprocess.CalledProcessError, OSError) as exc:
                raise FrontendBuildFailedError(frontend_dir, exc) from exc
            # Recorded only after a build that succeeded, and beside the bundle rather
            # than inside it: `emptyOutDir` would delete a record kept inside, and gate
            # stage 6b compares the bundle with nothing ignored, so a file there that
            # `vite build` does not emit fails the gate.
            build_inputs.write_record(frontend_dir, static_dir.parent)
            console.print("[bold green]✔ Frontend built successfully![/bold green]")


VITE_IDENTITY_MARKER = "UClone-X"
"""A string only this project's `frontend/index.html` serves.

The check below asserts identity rather than liveness. A dev server answering on the
port is not evidence that it is *our* dev server: another project's Vite bound to
`*:<port>` and ours to `127.0.0.1:<port>` coexist happily, and `localhost` then resolves
to whichever the OS prefers. Measured on this machine: `localhost:5173` served a
different project's dashboard while `127.0.0.1:5173` served this one.
"""


def _diagnose_vite(host: str, port: int, timeout_s: float = 10.0) -> str | None:
    """Return `None` when the dev server on `host:port` is ours, else why it is not.

    Called after co-spawning Vite so the success line is printed on evidence rather than
    on having started a subprocess. Announcing HMR because `Popen` returned is an
    assertion independent of the outcome — the announcement is identical whether the
    server bound, died on startup, or lost the port to another project.
    """
    import httpx

    deadline = time.monotonic() + timeout_s
    last: str = "no response"
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"http://{host}:{port}/", timeout=2.0)
        except Exception as exc:  # transport error: still starting, or never will
            last = f"no server answered ({type(exc).__name__})"
        else:
            body = resp.text
            if VITE_IDENTITY_MARKER in body:
                return None
            title = "unknown app"
            if "<title>" in body and "</title>" in body:
                title = body.split("<title>", 1)[1].split("</title>", 1)[0].strip()
            return (
                f"port {port} is served by a different application ({title!r}). "
                f"Another project's dev server holds it; this one's HMR is not reachable "
                f"there. Free the port or pass --vite-port."
            )
        time.sleep(0.4)
    return last


def start_ui_server(
    port: int = 5180,
    dev: bool = False,
    host: str = "127.0.0.1",
    vite_port: int = 5173,
    storage_dir: Path | None = None,
    auto_open_browser: bool = False,
    workspace_dir: Path | None = None,
    timeout_graceful_shutdown: int = DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_S,
) -> None:
    """Start the UClone-X developer dashboard server.

    In development mode (`dev=True`), this automatically co-spawns the Vite HMR
    development server (`npm run dev`) alongside FastAPI/Uvicorn, ensuring instant
    React hot-reloading on `http://localhost:5173` while maintaining clean process lifecycle.

    Args:
        port: TCP port to bind the FastAPI backend server.
        dev: If True, enables hot-reloading for rapid Python & React development.
        host: Host IP address to bind.
        vite_port: TCP port for the Vite HMR server in dev mode.
        storage_dir: Optional custom session storage directory path.
        auto_open_browser: If True, automatically launches system web browser once started.
        workspace_dir: Optional workspace directory for tool executions (defaults to cwd).
        timeout_graceful_shutdown: Graceful shutdown timeout in seconds (default 2s, Issue #613).
    """
    if auto_open_browser:
        import threading
        import webbrowser

        target_url = f"http://{host}:{vite_port if dev else port}"
        threading.Timer(1.2, lambda: webbrowser.open(target_url)).start()

    if storage_dir is not None:
        os.environ[SESSION_STORAGE_DIR_ENV_VAR] = str(storage_dir.resolve())

    if workspace_dir is not None:
        os.environ["UCLONE_WORKSPACE_DIR"] = str(workspace_dir.resolve())

    _ensure_frontend_built()

    # Recorded before serving and removed after, so `ucx ui stop` can tell this dashboard's
    # launcher — this process — from anything else on the port or a reused PID (#927).
    record, unidentified = ui_process.new_dashboard_record(port=port, host=host)
    if unidentified is not None:
        console.print(
            f"[yellow]⚠️ `ucx ui stop` will not be able to stop this dashboard: {unidentified}. "
            f"Stop it with Ctrl-C.[/yellow]"
        )
    try:
        ui_process.write_dashboard_record(record)
    except DashboardNotIdentifiedError as exc:
        console.print(
            f"[yellow]⚠️ `ucx ui stop` will not be able to stop this dashboard: {exc}. "
            f"Stop it with Ctrl-C.[/yellow]"
        )
    try:
        _serve(
            port=port,
            dev=dev,
            host=host,
            vite_port=vite_port,
            storage_dir=storage_dir,
            workspace_dir=workspace_dir,
            timeout_graceful_shutdown=timeout_graceful_shutdown,
        )
    finally:
        ui_process.remove_dashboard_record(record)


def _serve(
    *,
    port: int,
    dev: bool,
    host: str,
    vite_port: int,
    storage_dir: Path | None,
    workspace_dir: Path | None,
    timeout_graceful_shutdown: int,
) -> None:
    """Run the dashboard in the foreground until it exits; `start_ui_server` wraps it."""
    if dev:
        frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
        vite_proc = None
        vite_log = Path(tempfile.gettempdir()) / f"uclone-x-vite-{vite_port}.log"

        if frontend_dir.is_dir() and (frontend_dir / "package.json").is_file():
            console.print(
                f"[bold cyan]⚡ Co-spawning Vite HMR Dev Server on[/bold cyan] "
                f"[underline green]http://{host}:{vite_port}[/underline green]"
            )
            import atexit
            import subprocess

            try:
                # stderr is kept, not discarded. It was `DEVNULL`, so a Vite that
                # failed to start produced no output anywhere while the success line
                # below printed regardless.
                vite_stderr = vite_log.open("w", encoding="utf-8")
                vite_proc = subprocess.Popen(
                    ["npm", "run", "dev", "--", "--port", str(vite_port), "--host", host],
                    cwd=str(frontend_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=vite_stderr,
                )

                def _cleanup_vite() -> None:
                    if vite_proc and vite_proc.poll() is None:
                        try:
                            vite_proc.terminate()
                            vite_proc.wait(timeout=2.0)
                        except Exception:
                            vite_proc.kill()

                atexit.register(_cleanup_vite)
            except Exception as e:
                console.print(
                    f"[bold yellow]⚠️ Could not auto-start Vite dev server: {e}[/bold yellow]"
                )

        commit_sha = get_git_commit()
        console.print(
            f"[bold yellow]🔥 Starting UClone-X Developer Mode (Hot-Reload Enabled) [cyan]v{__version__} ({commit_sha})[/cyan] on[/bold yellow] "
            f"[cyan]http://{host}:{port}[/cyan]"
        )
        if vite_proc is not None:
            problem = _diagnose_vite(host, vite_port)
            if problem is None:
                console.print(
                    f"[bold green]✨ Instant React HMR Active:[/bold green] Open "
                    f"[cyan]http://{host}:{vite_port}[/cyan] in your browser"
                )
                console.print(
                    f"[dim]   Use {host}, not 'localhost' — they can resolve to "
                    f"different servers when another project holds the port.[/dim]"
                )
            else:
                console.print(f"[bold yellow]⚠️ React HMR is not available:[/bold yellow] {problem}")
                console.print(f"[dim]   Vite output: {vite_log}[/dim]")
                console.print(
                    f"[dim]   The dashboard on http://{host}:{port} still works; it serves "
                    f"the built bundle, so frontend edits need a rebuild.[/dim]"
                )
        src_dir = str(Path(__file__).resolve().parents[2])
        # The factory below takes no arguments, so the bind address reaches
        # `create_ui_app` -- which guards a loopback-bound app against DNS rebinding
        # (#1413) -- through the environment, as the storage and workspace paths do.
        os.environ[UI_BIND_HOST_ENV_VAR] = host
        try:
            # The factory, not a module-level `app`. Reload mode needs an import string,
            # and `uclone_x.ui.app:app` made merely importing that module build a whole
            # dashboard -- a second `AgentSessionManager` bound to the real
            # `~/.uclone/sessions`, built before any fixture could redirect it, by every
            # test and tool that imports the module for one function.
            uvicorn.run(
                "uclone_x.ui.app:create_ui_app",
                factory=True,
                host=host,
                port=port,
                reload=True,
                reload_dirs=[src_dir],
                log_level="info",
                timeout_graceful_shutdown=timeout_graceful_shutdown,
            )
        finally:
            if vite_proc and vite_proc.poll() is None:
                try:
                    vite_proc.terminate()
                    vite_proc.wait(timeout=2.0)
                except Exception:
                    vite_proc.kill()
    else:
        commit_sha = get_git_commit()
        console.print(
            f"[bold green]🚀 Launching UClone-X Embedded Developer Dashboard [cyan]v{__version__} ({commit_sha})[/cyan] on[/bold green] "
            f"[cyan]http://{host}:{port}[/cyan]"
        )
        app_kwargs: dict[str, Any] = {"storage_dir": storage_dir, "bind_host": host}
        if workspace_dir is not None:
            app_kwargs["workspace_dir"] = workspace_dir
        app_instance = create_ui_app(**app_kwargs)
        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            log_level="info",
            timeout_graceful_shutdown=timeout_graceful_shutdown,
        )
