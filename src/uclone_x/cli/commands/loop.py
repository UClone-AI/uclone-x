"""Standalone CLI command for recurring loop execution (FR-Loop, P4, P6)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from uclone_x.agent.loop import (
    LoopJob,
    LoopScheduler,
    LoopStatus,
    LoopTickResult,
    parse_interval_string,
    parse_loop_command_input,
)
from uclone_x.agent.prompts import compose_system_prompt
from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.factory import create_llm_connector
from uclone_x.sandbox.models import WorkspaceIsolation
from uclone_x.telemetry import TelemetryTracer
from uclone_x.tools.registry import create_default_registry

logger = logging.getLogger(__name__)
console = Console(emoji=False)
err_console = Console(stderr=True, emoji=False)

loop_app = typer.Typer(
    name="loop",
    help="Execute recurring agent automation loops (Claude-style loop)",
    no_args_is_help=True,
)


async def _run_loop_agent(
    prompt: str,
    interval_seconds: float,
    max_runs: int | None = None,
    timeout_seconds: float = 600.0,
    clean_context: bool = False,
    until_pattern: str | None = None,
    max_consecutive_failures: int = 3,
    agent_name: str = "default",
    provider: str | None = None,
    model: str | None = None,
    workspace_dir: Path | None = None,
    session_id: str | None = None,
) -> int:
    """Async loop runner initializing agent and driving LoopScheduler."""
    effective_cwd = workspace_dir or Path.cwd().resolve()
    bus = EventBus()
    tools = create_default_registry()
    # Imported here, as `room` does: `ucx --help` imports this module, and `run` is kept
    # out of that path on purpose, by a fitness check on what `--help` reaches.
    from uclone_x.cli.commands.run import (
        apply_saved_model,
        loop_tick_turn,
        own_model_notice,
        record_in_room,
        report_set_aside,
    )

    saved_model, saved_notice = apply_saved_model(provider, model)
    if saved_notice is not None:
        console.print(f"[dim]{escape(saved_notice)}[/dim]")
    llm = create_llm_connector(provider=provider, model=model, fallback_to_mock=False)
    # `None` leaves the model to the connector, which refuses in plain words when it has
    # none configured, rather than sending a literal id to a provider that never served it.
    effective_model = saved_model

    store = SessionStore()
    tracer = TelemetryTracer()

    default_system = compose_system_prompt(model_name=effective_model)
    # Deferred for the reason `run` is: `ucx --help` imports this module.
    from uclone_x.agent.clone_builder import build_clone, local_app_scope, saved_models
    from uclone_x.core.agent_home import AgentHomeError
    from uclone_x.errors import PathTraversalError, RoomError
    from uclone_x.room.one_seat import open_head_room
    from uclone_x.room.store import RoomStore
    from uclone_x.skills.auditor import load_runtime_skill_registry

    # A loop is a one-seat room (§5.9, owner ruling 2026-09-27): `--session-id` names the
    # room, or a new one is started; the clone keeps a room seat's session in it. A
    # session kept under the earlier `loop_<clone>` name is left on disk and not resumed.
    rooms = RoomStore()
    try:
        room = open_head_room(agent_name, session_id, head="loop", store=rooms)
    except AgentHomeError as exc:
        err_console.print(f"[bold red]✖ Invalid --agent:[/bold red] {escape(str(exc))}")
        return 2
    except (PathTraversalError, RoomError) as exc:
        err_console.print(f"[bold red]✖ Invalid --session-id:[/bold red] {escape(str(exc))}")
        return 2
    if room.created and session_id is None:
        err_console.print(
            f"[dim]New conversation [bold]{escape(room.room_id)}[/bold]; "
            f"--session-id {escape(room.room_id)} resumes it.[/dim]"
        )

    # Built as the desktop app builds the same clone (#1731).
    agent = build_clone(
        local_app_scope(
            workspace_root=effective_cwd,
            llm=llm,
            tools=tools,
            global_models=saved_models(effective_model),
            bus=bus,
            tracer=tracer,
            store=store,
            # P9: the approved skills in the runtime store; without them no `load_skill`.
            skills=await load_runtime_skill_registry(),
        ),
        clone_id=agent_name,
        session_id=room.session_id,
        # `--model` wins over a persona's own model, as a model asked for in the app does;
        # the saved choice only fills what the persona leaves empty (`saved_models`).
        model_name=model,
        fallback_prompt=default_system,
        config_update={"isolation": WorkspaceIsolation()},
    ).agent
    own_model = own_model_notice(agent_name, agent.config.llm_config.model_name, saved_model)
    if own_model is not None:
        console.print(f"[dim]{escape(own_model)}[/dim]")

    agent.hydrate_session()
    await agent.start()

    console.print(
        Panel(
            f"[bold cyan]🔁 UClone-X Recurring Loop[/bold cyan]\n"
            f"[dim]Agent:[/dim] [green]{escape(agent_name)}[/green] | "
            f"[dim]Interval:[/dim] [yellow]{interval_seconds:.0f}s[/yellow] | "
            f"[dim]Max Runs:[/dim] [magenta]{max_runs or '∞'}[/magenta] | "
            f"[dim]Timeout:[/dim] [blue]{timeout_seconds:.0f}s[/blue]\n"
            f"[dim]Prompt:[/dim] {escape(prompt)}"
            + (
                f"\n[dim]Until Pattern:[/dim] [cyan]{escape(until_pattern)}[/cyan]"
                if until_pattern
                else ""
            )
            + (
                "\n[dim]Context Mode:[/dim] [yellow]Clean per tick[/yellow]"
                if clean_context
                else ""
            ),
            border_style="cyan",
        )
    )

    def _on_tick(job: LoopJob, result: LoopTickResult) -> None:
        if result.skipped:
            err_console.print(
                f"[dim]⏱ [Tick #{result.tick_index}] Skipped: prior tick still running.[/dim]"
            )
            return

        status_style = "green" if result.success else "red"
        console.print(
            f"\n[bold yellow]⏱ [Tick #{result.tick_index}][/bold yellow] "
            f"[dim]({result.duration_seconds:.1f}s at {result.finished_at.strftime('%H:%M:%S')})[/dim]"
        )
        if result.success:
            console.print(
                f"[{status_style}]{escape(result.content or '(No response content)')}[/{status_style}]"
            )
        else:
            err_console.print(
                f"[{status_style}]✖ Error: {escape(result.error or 'unknown error')}[/{status_style}]"
            )
        # Each tick is a turn in the loop's room (#1837), saved to the seat first as a room
        # seat's turn is, so the room never shows a tick the seat's session lacks.
        try:
            agent.persist_session()
        except Exception as persist_err:
            logger.warning("Failed to persist loop session after a tick: %s", persist_err)
        # Said as `ucx run` says it (#1877): the save kept an earlier record aside.
        set_aside = report_set_aside(store, room.session_id, err_console)
        record_in_room(
            rooms,
            room_id=room.room_id,
            clone_id=agent_name,
            turn=loop_tick_turn(job, result, session_set_aside=set_aside),
            out=err_console,
            head="loop",
        )

    scheduler = LoopScheduler(
        agent=agent, on_tick_completed=_on_tick, person_names=room.person_names
    )
    job = scheduler.add_job(
        interval_seconds=interval_seconds,
        prompt=prompt,
        max_runs=max_runs,
        timeout_seconds=timeout_seconds,
        clean_context=clean_context,
        until_pattern=until_pattern,
        max_consecutive_failures=max_consecutive_failures,
        run_immediately=True,
    )

    try:
        await scheduler.wait_job(job.job_id)
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Interrupted by user. Cancelling loop...[/yellow]")
        scheduler.cancel_job(job.job_id)
    finally:
        scheduler.cancel_all()
        try:
            agent.persist_session()
        except Exception as persist_err:
            logger.warning("Failed to persist loop session: %s", persist_err)
        report_set_aside(store, room.session_id, err_console)  # the save on the way out
        await agent.stop()

    # Summary
    success_count = sum(1 for r in job.history if r.success)
    fail_count = sum(1 for r in job.history if not r.success and not r.skipped)
    skipped_count = sum(1 for r in job.history if r.skipped)

    console.print(
        f"\n[bold cyan]Loop Summary ({escape(job.job_id)}):[/bold cyan] "
        f"[green]{success_count} succeeded[/green], "
        f"[red]{fail_count} failed[/red], "
        f"[yellow]{skipped_count} skipped[/yellow] "
        f"([dim]Final status: {job.status.value}[/dim])"
    )

    return 0 if job.status in (LoopStatus.COMPLETED, LoopStatus.ACTIVE, LoopStatus.CANCELLED) else 1


@loop_app.command("run")
def run_loop_cmd(
    prompt_or_text: str = typer.Argument(
        ...,
        help="Prompt to execute, or natural language command (e.g. '5분마다 test check 돌려줘')",
    ),
    interval: str | None = typer.Option(
        None,
        "--interval",
        "-i",
        help="Recurring interval (e.g. '30s', '5m', '1h'). If omitted, extracts from text.",
    ),
    max_runs: int | None = typer.Option(
        None,
        "--max-runs",
        "-n",
        help="Maximum number of loop iterations before stopping (default: unlimited)",
    ),
    timeout: float = typer.Option(
        600.0,
        "--timeout",
        "-t",
        help="Watchdog timeout per execution tick in seconds (default: 600s / 10m)",
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Reset session dialogue memory before each tick (uclone2 clean poller mode)",
    ),
    until: str | None = typer.Option(
        None,
        "--until",
        help="Stop loop if agent response matches this regex pattern",
    ),
    max_consecutive_failures: int = typer.Option(
        3,
        "--max-failures",
        help="Maximum consecutive failed turns before aborting (default: 3)",
    ),
    agent_name: str = typer.Option("default", "--agent", "-a", help="Agent identifier"),
    provider: str | None = typer.Option(None, "--provider", "-p", help="LLM Provider"),
    model: str | None = typer.Option(None, "--model", "-m", help="LLM Model"),
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", "-w", help="Workspace root directory"),
    ] = None,
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Conversation (one-seat room) to resume; a new one is started if omitted",
    ),
) -> None:
    """Run recurring agent automation loop with watchdog and concurrency protections."""
    effective_interval: float
    effective_prompt: str

    if interval is not None:
        try:
            effective_interval = parse_interval_string(interval)
        except ValueError as err:
            err_console.print(f"[bold red]✖ Invalid --interval argument:[/bold red] {err}")
            raise typer.Exit(code=2) from err
        effective_prompt = prompt_or_text.strip()
    else:
        try:
            effective_interval, effective_prompt = parse_loop_command_input(prompt_or_text)
        except ValueError as err:
            err_console.print(f"[bold red]✖ Could not determine interval:[/bold red] {err}")
            raise typer.Exit(code=2) from err

    effective_cwd = cwd.resolve() if cwd is not None else Path.cwd().resolve()

    exit_code = asyncio.run(
        _run_loop_agent(
            prompt=effective_prompt,
            interval_seconds=effective_interval,
            max_runs=max_runs,
            timeout_seconds=timeout,
            clean_context=clean,
            until_pattern=until,
            max_consecutive_failures=max_consecutive_failures,
            agent_name=agent_name,
            provider=provider,
            model=model,
            workspace_dir=effective_cwd,
            session_id=session_id,
        )
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)
