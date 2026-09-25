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

from uclone_x.agent.composition import HostDependencies, compose_agent
from uclone_x.agent.loop import (
    LoopJob,
    LoopScheduler,
    LoopStatus,
    LoopTickResult,
    parse_interval_string,
    parse_loop_command_input,
)
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.agent.prompts import compose_system_prompt
from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.factory import create_llm_connector
from uclone_x.memory.store import default_cross_session_memory
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
    from uclone_x.cli.commands.run import apply_saved_model

    saved_model, saved_notice = apply_saved_model(provider, model)
    if saved_notice is not None:
        console.print(f"[dim]{escape(saved_notice)}[/dim]")
    llm = create_llm_connector(provider=provider, fallback_to_mock=False)
    effective_model = saved_model or getattr(llm, "default_model", "qwen3:8b")

    store = SessionStore()
    effective_session_id = session_id or f"loop_{agent_name}"
    tracer = TelemetryTracer()

    default_system = compose_system_prompt(model_name=effective_model)
    config = AgentConfig(
        agent_id=agent_name,
        name=agent_name,
        system_prompt=default_system,
        llm_config=AgentLLMConfig(
            model_name=effective_model,
            temperature=0.7,
            max_tokens=2048,
        ),
        isolation=WorkspaceIsolation(),
    )
    context = AgentContext(
        session_id=effective_session_id,
        agent_id=agent_name,
        workspace_root=effective_cwd,
    )
    host = HostDependencies(
        bus=bus,
        llm=llm,
        tools=tools,
        tracer=tracer,
        store=store,
        memory=default_cross_session_memory(agent_name),
    )
    agent = compose_agent(config=config, host=host, context=context)

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

    scheduler = LoopScheduler(agent=agent, on_tick_completed=_on_tick)
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
    session_id: str | None = typer.Option(None, "--session-id", help="Session identifier"),
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
