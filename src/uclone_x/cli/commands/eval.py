"""CLI commands for managing and executing evaluation suites (./ucx eval)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from uclone_x.evaluation import (
    EvalBackendUnavailableError,
    EvalReportProtocol,
    EvalRunnerProtocol,
    create_eval_runner,
    default_live_reason,
    eval_backend_available,
)
from uclone_x.evaluation.answerer import (
    LIVE_EVAL_REQUEST_TIMEOUT_SECONDS,
    AgentAnswerer,
    EvalToolPreflightError,
    build_agent_answerer,
)

eval_app = typer.Typer(
    name="eval",
    help="Execute and inspect standardized evaluation benchmarks and scorecards",
    no_args_is_help=True,
)
console = Console()

_BACKEND_MISSING_HINT = (
    "No evaluation backend is installed; the evaluation suites are distributed "
    "separately from the UClone-X runtime."
)


def _runner(reports_dir: Path | None) -> EvalRunnerProtocol:
    """Build a runner, or exit with the reason no backend could supply one.

    The `eval` command group is registered unconditionally so that `--help`
    lists it and the failure names the missing backend, rather than the group
    silently disappearing from a distribution that ships no suites.
    """
    try:
        return create_eval_runner(reports_dir=reports_dir)
    except EvalBackendUnavailableError as exc:
        console.print(f"[bold yellow]⚠ {exc}[/bold yellow]")
        raise typer.Exit(1) from exc


@eval_app.command("list")
def eval_list(
    reports_dir: Annotated[
        Path | None, typer.Option("--reports-dir", help="Optional custom reports directory path")
    ] = None,
) -> None:
    """List available evaluation suites."""
    runner = _runner(reports_dir)
    suites = runner.list_suites()

    offline_suites = [s for s in suites if not s.get("requires_live", False)]
    live_suites = [s for s in suites if s.get("requires_live", False)]

    table = Table(title="Available Evaluation Suites")
    table.add_column("Suite Name", style="bold cyan")
    table.add_column("Status", style="bold")
    table.add_column("Description", style="white")

    # "live" on its own says a suite is gated without saying what gates it. The reason column
    # is added only when something is actually live, so the offline listing keeps its shape.
    if live_suites:
        table.add_column("Why Live", style="yellow")

    for s in offline_suites:
        row = [str(s["name"]), "[green]offline[/green]", str(s["description"])]
        if live_suites:
            row.append("[dim]-[/dim]")
        table.add_row(*row)

    if live_suites:
        table.add_section()
        for s in live_suites:
            reason = str(s.get("live_reason") or default_live_reason())
            table.add_row(
                str(s["name"]),
                "[yellow]live (--live)[/yellow]",
                str(s["description"]),
                reason,
            )

    console.print(table)


def _selection_has_live_suite(runner: EvalRunnerProtocol, suite: str) -> bool:
    """Whether the chosen suite -- or any suite, for `all` -- declares `requires_live`.

    Asked of the registry rather than inferred from the flags. `--live` is permission, not
    a description of the run: an offline suite invoked with it does not become live, and
    making it require a provider would break `ucx eval run ontology --live --provider
    openai`, which works today and uses no provider at all.
    """
    try:
        metas = runner.list_suites()
    except Exception:
        # The runner will fail on the same call a moment later with a better message; this
        # helper must not be the thing that reports it.
        return False
    if suite == "all":
        return any(bool(m.get("requires_live", False)) for m in metas)
    return any(str(m.get("name")) == suite and bool(m.get("requires_live", False)) for m in metas)


@eval_app.command("run")
def eval_run(
    suite: Annotated[
        str, typer.Argument(help="Evaluation suite name or 'all' to run all suites")
    ] = "all",
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m", help="Target model identifier override (e.g. qwen3:8b, hermes3:8b)"
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="Target model provider (e.g. ollama, mock); anything but 'mock' needs --live",
        ),
    ] = None,
    json_path: Annotated[
        Path | None,
        typer.Option("--json", "-j", help="Path to write structured JSON evaluation output"),
    ] = None,
    reps: Annotated[
        int,
        typer.Option(
            "--reps",
            "-r",
            help=(
                "Number of times to repeat the whole suite run; the report carries the "
                "spread across them"
            ),
        ),
    ] = 1,
    reports_dir: Annotated[
        Path | None, typer.Option("--reports-dir", help="Directory to save evaluation reports")
    ] = None,
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Enable live evaluation suites that spend tokens or require external services",
        ),
    ] = False,
    workspace_root: Annotated[
        Path | None,
        typer.Option(
            "--workspace-root",
            help=(
                "Repository the live agent's tools operate on "
                "(default: the checkout this command was invoked from)"
            ),
        ),
    ] = None,
    request_timeout: Annotated[
        float,
        typer.Option(
            "--request-timeout",
            help=(
                "Per-request ceiling in seconds for the live agent's provider; recorded "
                "in the report as request_timeout_s. Two runs at different ceilings are "
                "two different measurements"
            ),
        ),
    ] = LIVE_EVAL_REQUEST_TIMEOUT_SECONDS,
    tiers: Annotated[
        str | None,
        typer.Option(
            "--tiers",
            help=("Which frontier tiers to run, e.g. 0-10, L00-L04, 0,4,9; default is every tier"),
        ),
    ] = None,
    per_tier: Annotated[
        int | None,
        typer.Option(
            "--per-tier",
            help=(
                "How many problems to draw from each tier; default is every problem the tier holds"
            ),
        ),
    ] = None,
    tier_seed: Annotated[
        int,
        typer.Option(
            "--tier-seed",
            help=(
                "Seed for the per-tier draw; two runs are comparable only when they name "
                "the same seed"
            ),
        ),
    ] = 0,
) -> None:
    """Execute evaluation suite(s) and output structured JSON scorecard."""
    runner = _runner(reports_dir)

    # A live suite measures the caller's own agent and deliberately refuses to build one
    # for itself, so the composition is assembled here, at the call site, where it is
    # visible.
    #
    # Three conditions, each closing a different hole. `--live` is permission, not a
    # description: without the registry check, `ucx eval run ontology --live --provider
    # openai` -- which works today and touches no provider -- would start failing on a
    # missing key. Without `provider`, a live suite invoked bare would report a connector
    # failure instead of its own `precondition_error`, which names `--provider` and says
    # nothing was written; that message is better than anything raised from here.
    #
    # The tier selection is assembled separately and unconditionally, because it is not
    # part of that composition: `--tiers` and `--per-tier` say which problems to measure,
    # and a run that drops them because no agent was built would measure the whole set
    # while the command line said otherwise (#1040). Absent flags leave the keys out
    # entirely rather than writing their defaults in, so a plain run still reaches the
    # runner with `options=None` exactly as before.
    selection: dict[str, object] = {}
    if tiers is not None:
        selection["frontier_tiers"] = tiers
    if per_tier is not None:
        selection["frontier_per_tier"] = per_tier
    if tier_seed != 0:
        selection["frontier_seed"] = tier_seed

    options: dict[str, object] | None = dict(selection) if selection else None
    live_answerer: AgentAnswerer | None = None
    if live and provider and _selection_has_live_suite(runner, suite):
        try:
            live_answerer = build_agent_answerer(
                provider=provider,
                model=model,
                workspace_root=workspace_root,
                request_timeout_s=request_timeout,
            )
            options = {
                **selection,
                "answerer": live_answerer,
                "workspace_root": getattr(live_answerer, "workspace_root", workspace_root),
            }
        except EvalToolPreflightError as exc:
            # Deliberately before `runner.run`, so nothing is written. A run whose tools are
            # dead answers every question from the model's weights and still reports a
            # capability figure the report has no way to mark as wrong (#666).
            console.print(
                f"[bold red]✖ Tool preflight failed:[/bold red] {exc}\n"
                "No report was written. A capability run whose tools do not work measures "
                "the model, not the runtime, and the number it produces is indistinguishable "
                "from a real one."
            )
            sys.exit(1)
        except Exception as exc:
            console.print(
                f"[bold red]✖ Cannot build the agent to measure:[/bold red] {exc}\n"
                "A live suite needs a provider that can be constructed. Nothing falls back "
                "to a mock here: a capability figure produced from a mock would be worse "
                "than none."
            )
            sys.exit(1)

    # `--reps` multiplies the whole run, and under `--live` it multiplies the spend. The
    # flag is unbounded on purpose -- the useful number of repetitions is a judgement --
    # so the cost is echoed before the first token is spent rather than capped.
    if reps > 1:
        cost = (
            "each repetition is a full run and spends its own tokens"
            if live
            else "offline; each repetition is a full run"
        )
        console.print(f"[cyan]ℹ {reps} repetitions of {suite!r} — {cost}.[/cyan]")

    try:
        try:
            reports = runner.run(
                suite_name=suite,
                model=model,
                provider=provider,
                reps=reps,
                json_path=json_path,
                options=options,
                live=live,
            )
        except ValueError as exc:
            console.print(f"[bold red]✖ Evaluation failed:[/bold red] {exc}")
            sys.exit(1)
    finally:
        if live_answerer is not None and hasattr(live_answerer, "close"):
            live_answerer.close()

    if runner.excluded_suites:
        excluded_str = ", ".join(runner.excluded_suites)
        console.print(
            f"[yellow]ℹ Live suite(s) excluded: {excluded_str} (pass --live to execute).[/yellow]\n"
        )

    for report in reports:
        _render_suite_report(report)

    if len(reports) > 1:
        _render_aggregate_summary(reports)

    if json_path is not None:
        console.print(f"\n[green]✔ Evaluation JSON output saved to {json_path}[/green]")


@eval_app.command("status")
def eval_status(
    reports_dir: Annotated[
        Path | None, typer.Option("--reports-dir", help="Directory containing evaluation reports")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output latest scorecard in JSON format")
    ] = False,
) -> None:
    """Render the latest aggregate scorecard in CLI format/table."""
    runner = _runner(reports_dir)
    scorecard = runner.get_latest_scorecard()

    if not scorecard:
        if json_output:
            console.print_json(json.dumps({}))
        else:
            console.print(
                "[yellow]No evaluation reports found in evals/reports/.\n"
                "Run `./ucx eval run` to execute evaluation suites and generate reports.[/yellow]"
            )
        return

    if json_output:
        data = {k: v.model_dump() for k, v in scorecard.items()}
        console.print_json(json.dumps(data))
        return

    table = Table(title="UClone-X Latest Evaluation Scorecard")
    table.add_column("Suite", style="bold cyan", no_wrap=True)
    table.add_column("Timestamp", style="dim")
    table.add_column("Model / Provider", style="white")
    table.add_column("Probes", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Latency (p50 / worst)", justify="right")
    table.add_column("Status", style="bold")

    for suite_name, report in sorted(scorecard.items()):
        model_prov = f"{report.model or 'default'} ({report.provider or 'internal'})"
        probes_str = f"{report.summary.passed_probes}/{report.summary.total_probes}"
        rate_str = f"{report.summary.pass_rate:.1%}"

        p50_str = (
            f"{report.summary.p50_latency_s:.2f}s"
            if report.summary.p50_latency_s is not None
            else "-"
        )
        worst_str = (
            f"{report.summary.worst_latency_s:.2f}s"
            if report.summary.worst_latency_s is not None
            else "-"
        )
        latency_str = f"{p50_str} / {worst_str}"

        if report.summary.pass_rate >= 1.0:
            status_badge = "[green]✔ PASS[/green]"
        elif report.summary.pass_rate >= 0.8:
            status_badge = "[yellow]⚠ WARN[/yellow]"
        else:
            status_badge = "[red]✖ FAIL[/red]"

        table.add_row(
            suite_name,
            report.timestamp[:19].replace("T", " "),
            model_prov,
            probes_str,
            rate_str,
            latency_str,
            status_badge,
        )

    console.print(table)


@eval_app.command("view")
def eval_view(
    port: Annotated[
        int | None,
        typer.Option("--port", "-p", help="Port to host the Promptfoo web viewer (default: 15500)"),
    ] = None,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", "-n", help="Do not automatically open browser"),
    ] = False,
    print_only: Annotated[
        bool,
        typer.Option(
            "--print-only",
            help="Print instructions to run npx promptfoo view without launching subprocess",
        ),
    ] = False,
) -> None:
    """Launch the Promptfoo evaluation web viewer or display launch instructions."""
    # Nothing to view without suites to produce reports, and the config the
    # instructions point at ships with the backend rather than the runtime.
    if not eval_backend_available():
        console.print(f"[bold yellow]⚠ {_BACKEND_MISSING_HINT}[/bold yellow]")
        raise typer.Exit(1)

    npx_path = shutil.which("npx")
    if npx_path is None:
        console.print(
            "[bold yellow]⚠ Node.js / npx is not installed or not found on PATH.[/bold yellow]\n\n"
            "Promptfoo requires Node.js (>= 18.0.0) and npx to launch the local web viewer.\n\n"
            "[bold]Instructions to set up and view evaluations:[/bold]\n"
            "  1. Install Node.js: https://nodejs.org/\n"
            "  2. Run evaluations: [bold cyan]npx promptfoo eval -c evals/promptfooconfig.yaml[/bold cyan]\n"
            "  3. Launch web viewer: [bold cyan]npx promptfoo view[/bold cyan]\n\n"
            "Alternatively, run [bold cyan]./ucx eval status[/bold cyan] to inspect the CLI scorecard."
        )
        raise typer.Exit(1)

    cmd: list[str] = ["npx", "promptfoo", "view"]
    if port is not None:
        cmd.extend(["-p", str(port)])
    if no_browser:
        cmd.append("--no-browser")

    if print_only:
        console.print("[bold green]✔ Node.js / npx is available.[/bold green]\n")
        console.print("To run the Promptfoo web viewer manually, execute:")
        console.print(f"  [bold cyan]{' '.join(cmd)}[/bold cyan]\n")
        console.print("To run evaluations first, execute:")
        console.print("  [bold cyan]npx promptfoo eval -c evals/promptfooconfig.yaml[/bold cyan]")
        return

    viewer_port = port or 15500
    console.print(
        f"[bold green]Starting Promptfoo web viewer...[/bold green] (port: {viewer_port})"
    )
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[dim]Promptfoo web viewer stopped.[/dim]")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        console.print(f"[bold red]✖ Failed to launch Promptfoo web viewer:[/bold red] {exc}")
        raise typer.Exit(1) from exc


def _render_suite_report(report: EvalReportProtocol) -> None:
    """Render details of an individual evaluation suite report."""
    table = Table(
        title=f"Evaluation Suite: [bold cyan]{report.suite}[/bold cyan] ({report.model or 'system'})"
    )
    table.add_column("Probe", style="bold")
    table.add_column("Status", style="bold")
    table.add_column("Latency", justify="right")
    table.add_column("Details", style="dim")

    for probe in report.probes:
        status_str = "[green]PASS[/green]" if probe.passed else "[red]FAIL[/red]"
        lat_str = f"{probe.latency_s:.3f}s"
        table.add_row(probe.name, status_str, lat_str, probe.message)

    console.print(table)

    summary = report.summary
    p50_s = f"{summary.p50_latency_s:.2f}s" if summary.p50_latency_s is not None else "N/A"
    console.print(
        f"[dim]Summary: {summary.passed_probes}/{summary.total_probes} passed "
        f"({summary.pass_rate:.1%}) | p50 latency: {p50_s} | total time: {summary.duration_s:.2f}s[/dim]"
    )
    disclosure = report.metadata.get("disclosure")
    if isinstance(disclosure, str) and disclosure:
        console.print(f"[yellow]⚠ {disclosure}[/yellow]")
    _render_reported_metrics(report)
    console.print()


#: Metrics a suite measures rather than passes or fails on, rendered under the summary line
#: so the finding is legible without opening the report JSON. `context_ab` (#561) is the
#: case that motivates this: its pass rate is deliberately silent about what the ungrounded
#: arm invented, and the number that carries the finding lives only in `report.metadata`.
_REPORTED_METRIC_KEYS: tuple[tuple[str, str], ...] = (
    ("fabrication_rate_with_context", "fabrication (with context)"),
    ("fabrication_rate_without_context", "fabrication (without context)"),
    ("abstention_rate_with_context", "abstention (with context)"),
    ("abstention_rate_without_context", "abstention (without context)"),
    ("silence_recall", "silence recall"),
    ("silence_precision", "silence precision"),
    ("hallucination_rate", "hallucination rate"),
)


def _render_reported_metrics(report: EvalReportProtocol) -> None:
    """Print the measured-not-graded rates a report carries, when it carries any."""
    parts: list[str] = []
    for key, label in _REPORTED_METRIC_KEYS:
        value = report.metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{label} {float(value):.1%}")
    if parts:
        console.print(f"[dim]Reported (measured, not graded): {' | '.join(parts)}[/dim]")


def _render_aggregate_summary(reports: list[EvalReportProtocol]) -> None:
    """Render overall summary scorecard across multiple suites."""
    table = Table(title="Execution Scorecard Summary")
    table.add_column("Suite", style="bold cyan")
    table.add_column("Probes", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Status", style="bold")

    total_probes = sum(r.summary.total_probes for r in reports)
    total_passed = sum(r.summary.passed_probes for r in reports)
    total_duration = sum(r.summary.duration_s for r in reports)

    for r in reports:
        status_badge = (
            "[green]PASS[/green]"
            if r.summary.pass_rate >= 1.0
            else ("[yellow]WARN[/yellow]" if r.summary.pass_rate >= 0.8 else "[red]FAIL[/red]")
        )
        table.add_row(
            r.suite,
            f"{r.summary.passed_probes}/{r.summary.total_probes}",
            f"{r.summary.pass_rate:.1%}",
            f"{r.summary.duration_s:.2f}s",
            status_badge,
        )

    overall_rate = (total_passed / total_probes) if total_probes > 0 else 0.0
    overall_status = (
        "[green]PASS[/green]"
        if overall_rate >= 1.0
        else ("[yellow]WARN[/yellow]" if overall_rate >= 0.8 else "[red]FAIL[/red]")
    )
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_passed}/{total_probes}[/bold]",
        f"[bold]{overall_rate:.1%}[/bold]",
        f"[bold]{total_duration:.2f}s[/bold]",
        overall_status,
    )
    console.print(table)
