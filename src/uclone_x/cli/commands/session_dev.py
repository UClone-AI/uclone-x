"""CLI commands for inspecting, diagnosing, and auditing agent conversation sessions and logs.

Provides:
- ./ucx dev session list
- ./ucx dev session show <session_id>
- ./ucx dev session status <session_id>
- ./ucx dev session check [session_id]
- ./ucx dev logs
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from uclone_x.agent.session import SessionStore
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.log_inspector import get_log_file, read_logs
from uclone_x.core.session_diagnostics import (
    DEFAULT_MAX_CONVERSATION_TURNS,
    check_session_health,
    inspect_session,
    list_session_summaries,
)
from uclone_x.llm.models import MessageRole

console = Console()

session_app = typer.Typer(
    name="session",
    help="Inspect, diagnose, and audit agent conversation sessions",
    no_args_is_help=True,
)


def _repo_root() -> Path:
    """Anchor paths to the repository root."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        candidate = out.stdout.strip()
        if candidate:
            return Path(candidate)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return Path.cwd()


def _format_status_badge(status: str) -> str:
    """Format session status string with Rich markup."""
    if status == "active":
        return "[bold green]Active[/bold green]"
    if status == "compacted":
        return "[bold yellow]Compacted[/bold yellow]"
    if status == "saturated":
        return "[bold red]Saturated[/bold red]"
    if status == "empty":
        return "[dim]Empty[/dim]"
    return f"[cyan]{status}[/cyan]"


@session_app.command("list")
def session_list(
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum number of sessions to display"),
    max_conversation_turns: int = typer.Option(
        DEFAULT_MAX_CONVERSATION_TURNS,
        "--max-turns",
        "-m",
        help="Advisory conversation-turn threshold for status derivation",
    ),
) -> None:
    """List recent agent conversation sessions with operational status."""
    store = SessionStore()
    summaries = list_session_summaries(
        store=store, max_conversation_turns=max_conversation_turns, limit=limit
    )

    if not summaries:
        console.print(
            f"[yellow]No sessions found in store ({store.storage_dir}).[/yellow]\n"
            "[dim]Start a chat session using './ucx run' or './ucx ui'.[/dim]"
        )
        return

    table = Table(
        title=f"📋 Persisted Agent Sessions ({len(summaries)} shown)",
        caption=f"Storage: {store.storage_dir}",
    )
    table.add_column("Session ID", style="bold cyan", no_wrap=True)
    table.add_column("Agent ID", style="magenta")
    table.add_column("Rev", justify="right", style="dim")
    table.add_column("Turns", justify="right")
    table.add_column("Msgs", justify="right")
    table.add_column("Updated At", style="dim")
    table.add_column("Status", justify="center")

    for s in summaries:
        turns_color = "red" if s.turn_counter >= max_conversation_turns else "green"
        turns_str = f"[{turns_color}]{s.turn_counter}/{max_conversation_turns}[/{turns_color}]"
        table.add_row(
            s.session_id,
            s.agent_id,
            str(s.revision),
            turns_str,
            str(s.message_count),
            s.updated_at[:19].replace("T", " "),
            _format_status_badge(s.status),
        )

    console.print(table)


@session_app.command("show")
def session_show(
    session_id: str = typer.Argument(..., help="Session ID to inspect"),
    tail: int = typer.Option(0, "--tail", "-n", help="Show only last N messages (0 for all)"),
    max_conversation_turns: int = typer.Option(
        DEFAULT_MAX_CONVERSATION_TURNS,
        "--max-turns",
        "-m",
        help="Advisory conversation-turn threshold for status derivation",
    ),
) -> None:
    """Render turn-by-turn conversation messages, roles, and tool calls."""
    store = SessionStore()
    details = inspect_session(
        session_id,
        store=store,
        workspace_root=_repo_root(),
        max_conversation_turns=max_conversation_turns,
    )
    if details is None:
        console.print(
            f"[bold red]✖ Session not found:[/bold red] '{session_id}' in {store.storage_dir}"
        )
        raise typer.Exit(code=1)

    s = details.summary
    status_str = _format_status_badge(s.status)
    header_text = (
        f"[bold cyan]Session:[/bold cyan] {s.session_id}  |  "
        f"[bold magenta]Agent:[/bold magenta] {s.agent_id}  |  "
        f"[bold]Turns:[/bold] {s.turn_counter}/{max_conversation_turns}  |  "
        f"[bold]Rev:[/bold] {s.revision}  |  "
        f"[bold]Status:[/bold] {status_str}\n"
        f"[dim]Updated: {s.updated_at} | Created: {s.created_at}[/dim]"
    )
    console.print(Panel(header_text, title="🔍 Session Turn Inspector", border_style="cyan"))

    messages = details.messages
    if tail > 0:
        messages = messages[-tail:]

    table = Table(box=None, padding=(0, 1))
    table.add_column("#", style="dim", justify="right", width=4)
    table.add_column("Role", width=12)
    table.add_column("Content / Action", ratio=3)
    table.add_column("Tool Details", ratio=2)

    role_styles: dict[MessageRole, str] = {
        MessageRole.SYSTEM: "bold yellow",
        MessageRole.USER: "bold cyan",
        MessageRole.ASSISTANT: "bold green",
        MessageRole.TOOL: "bold magenta",
    }

    start_idx = len(details.messages) - len(messages)
    for i, msg in enumerate(messages):
        turn_num = start_idx + i
        role_style = role_styles.get(msg.role, "white")
        role_name = msg.role.value.upper()
        if msg.compaction_ledger:
            role_name += " 📜"

        # Content formatting
        content_preview = ""
        if msg.content:
            clean_content = msg.content.strip().replace("\n", " ")
            if len(clean_content) > 120:
                content_preview = clean_content[:117] + "..."
            else:
                content_preview = clean_content
        elif msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            content_preview = (
                f"[italic dim]{len(msg.tool_calls)} tool call(s) requested[/italic dim]"
            )
        else:
            content_preview = "[dim italic](empty)[/dim italic]"

        # Tool details formatting
        tool_detail = ""
        if msg.tool_calls:
            tool_strs: list[str] = []
            for tc in msg.tool_calls:
                args_preview = json.dumps(unwrap_immutable(tc.arguments)) if tc.arguments else "{}"
                if len(args_preview) > 35:
                    args_preview = args_preview[:32] + "..."
                tool_strs.append(f"[bold]{tc.name}[/bold]({args_preview})")
            tool_detail = "\n".join(tool_strs)
        elif msg.role == MessageRole.TOOL:
            cid = msg.tool_call_id or "missing"
            tool_detail = f"[dim]id=[/dim]{cid}"
            if msg.content and "[Tool Output Offloaded" in msg.content:
                tool_detail += " [cyan](offloaded)[/cyan]"
            elif msg.content and "[Tool Output Truncated" in msg.content:
                tool_detail += " [yellow](truncated)[/yellow]"

        table.add_row(
            str(turn_num),
            f"[{role_style}]{role_name}[/{role_style}]",
            content_preview,
            tool_detail,
        )

    console.print(table)
    if tail > 0 and len(details.messages) > tail:
        console.print(f"[dim]Showing last {tail} of {len(details.messages)} total messages.[/dim]")


@session_app.command("status")
def session_status(
    session_id: str = typer.Argument(..., help="Session ID to inspect"),
    max_conversation_turns: int = typer.Option(
        DEFAULT_MAX_CONVERSATION_TURNS,
        "--max-turns",
        "-m",
        help="Advisory conversation-turn threshold to evaluate against",
    ),
) -> None:
    """Show turn budget utilization, message counts, plan, and artifact status."""
    store = SessionStore()
    details = inspect_session(
        session_id,
        store=store,
        workspace_root=_repo_root(),
        max_conversation_turns=max_conversation_turns,
    )
    if details is None:
        console.print(
            f"[bold red]✖ Session not found:[/bold red] '{session_id}' in {store.storage_dir}"
        )
        raise typer.Exit(code=1)

    s = details.summary
    pct = (
        round((s.turn_counter / max_conversation_turns) * 100, 1)
        if max_conversation_turns > 0
        else 0.0
    )
    bar_width = 25
    filled_width = (
        int(min(1.0, s.turn_counter / max_conversation_turns) * bar_width)
        if max_conversation_turns > 0
        else 0
    )
    bar_color = (
        "red" if s.turn_counter >= max_conversation_turns else ("yellow" if pct >= 70 else "green")
    )
    bar_str = f"[{bar_color}]{'█' * filled_width}{'░' * (bar_width - filled_width)}[/{bar_color}]"

    ledger_note = " (compaction ledger present)" if s.has_compaction else ""
    content_lines: list[str] = [
        f"[bold]Session ID:[/bold]     [cyan]{s.session_id}[/cyan]",
        f"[bold]Agent ID:[/bold]       [magenta]{s.agent_id}[/magenta]",
        f"[bold]Status:[/bold]         {_format_status_badge(s.status)}",
        f"[bold]Write Revision:[/bold] {s.revision}",
        f"[bold]Created At:[/bold]     {s.created_at}",
        f"[bold]Updated At:[/bold]     {s.updated_at}",
        "",
        f"[bold]Turn Budget:[/bold]    {bar_str}  [bold]{s.turn_counter}/{max_conversation_turns}[/bold] ({pct}%)",
        "",
        "[bold]Message Breakdown:[/bold]",
        f"  • System:     {details.role_counts.get(MessageRole.SYSTEM.value, 0)}{ledger_note}",
        f"  • User:       {details.role_counts.get(MessageRole.USER.value, 0)}",
        f"  • Assistant:  {details.role_counts.get(MessageRole.ASSISTANT.value, 0)}",
        f"  • Tool:       {details.role_counts.get(MessageRole.TOOL.value, 0)}",
        f"  • Total:      [bold]{len(details.messages)}[/bold]",
    ]

    # Plan section
    content_lines.append("")
    if details.has_plan:
        content_lines.extend(
            [
                f"[bold]Execution Plan:[/bold] [green]{details.plan_title or 'Untitled Plan'}[/green]",
                f"  • Progress:    {details.plan_steps_completed} / {details.plan_steps_total} steps completed",
            ]
        )
    else:
        content_lines.append("[bold]Execution Plan:[/bold] [dim]None attached[/dim]")

    # Tool artifacts section
    content_lines.append("")
    if details.artifacts_count > 0:
        size_kb = round(details.artifacts_total_bytes / 1024, 2)
        content_lines.extend(
            [
                f"[bold]Tool Output Artifacts:[/bold] [cyan]{details.artifacts_count} files[/cyan] ({size_kb} KB)",
            ]
        )
        for fname in details.artifact_files[:5]:
            content_lines.append(f"  • [dim]{fname}[/dim]")
        if len(details.artifact_files) > 5:
            content_lines.append(f"  [dim]... and {len(details.artifact_files) - 5} more[/dim]")
    else:
        content_lines.append("[bold]Tool Output Artifacts:[/bold] [dim]0 files[/dim]")

    panel = Panel(
        "\n".join(content_lines),
        title=f"📊 Session Status: {session_id}",
        border_style="cyan",
    )
    console.print(panel)


@session_app.command("check")
def session_check(
    session_id: str | None = typer.Argument(
        None, help="Specific session ID to check (omits to check all sessions)"
    ),
    max_conversation_turns: int = typer.Option(
        DEFAULT_MAX_CONVERSATION_TURNS,
        "--max-turns",
        "-m",
        help="Advisory conversation-turn threshold to evaluate against",
    ),
) -> None:
    """Execute health check for orphaned tool calls, saturation, and anomalies."""
    store = SessionStore()
    repo_root = _repo_root()

    sessions_to_check: list[str] = []
    if session_id:
        sessions_to_check = [session_id]
    else:
        sessions_to_check = list(store.list_session_ids())

    if not sessions_to_check:
        console.print(f"[yellow]No sessions found in store ({store.storage_dir}).[/yellow]")
        return

    total_checked = 0
    total_errors = 0
    total_warnings = 0

    for sid in sessions_to_check:
        report = check_session_health(
            session_id=sid,
            store=store,
            workspace_root=repo_root,
            max_conversation_turns=max_conversation_turns,
        )
        total_checked += 1
        total_errors += report.error_count
        total_warnings += report.warning_count

        if report.healthy and report.warning_count == 0:
            console.print(
                f"[bold green]✔[/bold green] [cyan]{sid}[/cyan]: [green]Healthy[/green] (0 issues)"
            )
        else:
            status_tag = (
                "[bold red]FAIL[/bold red]"
                if not report.healthy
                else "[bold yellow]WARN[/bold yellow]"
            )
            console.print(
                f"{status_tag} [cyan]{sid}[/cyan]: "
                f"{report.error_count} error(s), {report.warning_count} warning(s)"
            )
            for issue in report.issues:
                color = "red" if issue.severity == "error" else "yellow"
                turn_str = f" [turn #{issue.turn_index}]" if issue.turn_index is not None else ""
                console.print(f"    [{color}]• [{issue.code}]{turn_str} {issue.message}[/{color}]")

    console.print(
        f"\n[dim]Audit completed: {total_checked} session(s) checked. "
        f"{total_errors} error(s), {total_warnings} warning(s).[/dim]"
    )

    if total_errors > 0:
        raise typer.Exit(code=1)


def dev_logs(
    tail: int = typer.Option(50, "--tail", "-n", help="Number of recent log lines to display"),
    level: str | None = typer.Option(
        None, "--level", "-l", help="Minimum log level (DEBUG, INFO, WARN, ERROR, CRITICAL)"
    ),
    session_id: str | None = typer.Option(
        None, "--session", "-s", help="Filter logs by session ID"
    ),
    output_format: str = typer.Option(
        "text", "--format", "-f", help="Output format: 'text' or 'json'"
    ),
) -> None:
    """Stream or tail formatted logs from ~/.uclone/logs/ucx.log with structured filtering."""
    log_file = get_log_file()
    if not log_file.is_file():
        console.print(
            f"[yellow]ℹ No application log file found at:[/yellow] {log_file}\n"
            "[dim]Logs are recorded to ~/.uclone/logs/ucx.log when logging is active. "
            "Set UCX_LOG_DIR to point to a custom directory.[/dim]"
        )
        return

    entries = read_logs(log_file, tail=tail, min_level=level, session_id=session_id)
    if not entries:
        console.print(
            f"[yellow]No log entries matched criteria in {log_file}.[/yellow] "
            f"[dim](tail={tail}, level={level or 'all'}, session={session_id or 'all'})[/dim]"
        )
        return

    if output_format == "json":
        for e in entries:
            if e.raw and e.raw.startswith("{"):
                console.print(e.raw)
            else:
                rec: dict[str, Any] = {
                    "timestamp": e.timestamp,
                    "level": e.level,
                    "logger": e.logger,
                    "message": e.message,
                }
                if e.session_id:
                    rec["session_id"] = e.session_id
                rec.update(e.data)
                console.print(json.dumps(rec))
        return

    # Text format rendering
    level_colors: dict[str, str] = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARN": "bold yellow",
        "WARNING": "bold yellow",
        "ERROR": "bold red",
        "CRITICAL": "bold white on red",
    }

    console.print(f"[dim]── Application Logs: {log_file} ({len(entries)} lines) ──[/dim]")
    for e in entries:
        lvl_color = level_colors.get(e.level, "white")
        ts = e.timestamp[:19].replace("T", " ") if e.timestamp else " " * 19
        sid_badge = f" [cyan](sess={e.session_id})[/cyan]" if e.session_id else ""
        logger_name = f"[dim][{e.logger}][/dim]" if e.logger else ""
        console.print(
            f"[dim]{ts}[/dim] [{lvl_color}]{e.level:<7}[/{lvl_color}] {logger_name} {e.message}{sid_badge}"
        )
