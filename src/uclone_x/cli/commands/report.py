"""`ucx report` -- show what failed, and help send it somewhere useful.

The command is deliberately not under `dev`: a report is a user's business,
and `dev` is not registered in an installed build, which is precisely where
reports come from.
"""

from __future__ import annotations

import shutil
import subprocess
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from uclone_x.core.diagnostic_report import (
    ISSUE_REPO,
    issue_url,
    render_report,
    report_title,
    search_url,
    summarise,
)
from uclone_x.core.failure_journal import (
    CONSENT_DENIED,
    CONSENT_GRANTED,
    CONSENT_UNASKED,
    clear_journal,
    consent_path,
    journal_path,
    read_consent,
    read_journal,
    set_consent,
)

report_app = typer.Typer(
    name="report",
    help="Review recorded failures and report them",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()

_CONSENT_EXPLANATION = (
    "Failure collection is [bold]off[/bold]. When on, UClone-X writes a line to "
    "[cyan]{path}[/cyan] each time something fails:\n"
    "  the error type, the code path, versions of ucx/Python/your OS, and the\n"
    "  error message with credentials and your home directory masked.\n\n"
    "  It is written [bold]on your machine only[/bold]. Nothing is uploaded, ever, by\n"
    "  itself: `ucx report` shows you the text and you decide whether to send it.\n\n"
    "  Turn it on with [bold cyan]ucx report --enable[/bold cyan], off with "
    "[bold cyan]ucx report --disable[/bold cyan]."
)


@report_app.callback()
def report(
    ctx: typer.Context,
    enable: Annotated[
        bool, typer.Option("--enable", help="Allow failures to be recorded locally")
    ] = False,
    disable: Annotated[
        bool, typer.Option("--disable", help="Stop recording failures locally")
    ] = False,
    open_issue: Annotated[
        bool, typer.Option("--open", help="Open a pre-filled GitHub issue in your browser")
    ] = False,
    submit: Annotated[
        bool, typer.Option("--submit", help="File the issue with the GitHub CLI (`gh`)")
    ] = False,
    save: Annotated[
        Path | None, typer.Option("--save", help="Write the report to a file instead")
    ] = None,
    clear: Annotated[bool, typer.Option("--clear", help="Delete the recorded failures")] = False,
) -> None:
    """Show recorded failures, and optionally report them."""
    if ctx.invoked_subcommand is not None:
        return

    if enable and disable:
        console.print("[bold red]✖ --enable and --disable are contradictory.[/bold red]")
        raise typer.Exit(2)

    if enable or disable:
        try:
            path = set_consent(enable)
        except OSError as exc:
            console.print("[bold red]✖ Your choice could not be saved.[/bold red]")
            console.print(f"{consent_path()}: {exc}", markup=False, highlight=False, style="dim")
            raise typer.Exit(1) from exc
        state = "on" if enable else "off"
        console.print(f"[green]✔ Failure collection is {state}.[/green] [dim]({path})[/dim]")
        if disable:
            console.print("[dim]Already-recorded failures are kept; --clear deletes them.[/dim]")
        return

    if clear:
        outcome = clear_journal()
        if outcome.error is not None:
            # `False` used to mean both "nothing there" and "could not delete",
            # and this printed the first while the file stayed on disk.
            console.print("[bold red]✖ The recorded failures could not be deleted.[/bold red]")
            console.print(outcome.error, markup=False, highlight=False, style="dim")
            raise typer.Exit(1)
        console.print(
            "[green]✔ Recorded failures deleted.[/green]"
            if outcome.deleted
            else "[dim]Nothing to delete.[/dim]"
        )
        return

    consent = read_consent()
    state = consent.state
    if consent.error is not None:
        # Saying "you have never been asked" when the answer is on disk and
        # unreadable is a false statement about the user's own choice.
        console.print("[yellow]⚠ Your recording preference could not be read.[/yellow]")
        console.print(consent.error, markup=False, highlight=False, style="dim")
        console.print("[dim]Collection is off until it can be read again.[/dim]\n")
    read = read_journal()
    entries = list(read.entries)

    if read.error is not None:
        # Distinct from "nothing recorded", and distinct exit code from the
        # nothing-to-report path: a user whose journal is unreadable has a
        # problem to fix, not a quiet day.
        console.print("[bold red]✖ The failure journal could not be read.[/bold red]")
        # The path and the OS error are data: printed without markup, so a
        # directory named `[dim]` renders as itself rather than as a style.
        console.print(read.error, markup=False, highlight=False, style="dim")
        raise typer.Exit(1)

    if read.recording_blocked is not None:
        console.print("[bold red]✖ New failures are not being recorded.[/bold red]")
        console.print(read.recording_blocked, markup=False, highlight=False, style="dim")
        console.print()

    if read.unreadable_lines:
        console.print(
            f"[yellow]⚠ {read.unreadable_lines} journal line(s) could not be parsed "
            "and are omitted below.[/yellow]"
        )

    if state != CONSENT_GRANTED and not entries:
        message = _CONSENT_EXPLANATION.format(path=journal_path())
        if state == CONSENT_DENIED:
            message = "Failure collection is off, by your choice.\n\n" + message
        elif state == CONSENT_UNASKED:
            message = "Nothing has been recorded yet.\n\n" + message
        console.print(message)
        return

    body = render_report(read)
    title = report_title(entries)
    # `markup=False`: the body carries an exception message, and an exception
    # message carries whatever the failure was about. Printed as markup, a
    # message mentioning `uclone-x[llm]` loses the extra to rich's tag parser --
    # measured, not hypothetical -- and one containing `[bold]` would restyle
    # the terminal. Report text is data.
    console.print(body, markup=False, highlight=False)
    console.print()

    if not entries:
        console.print("[dim]Nothing to report — no failures have been recorded.[/dim]")
        return

    newest_fingerprint = entries[-1].fingerprint
    console.print(f"[dim]Already reported? Search first: {search_url(newest_fingerprint)}[/dim]")

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
        console.print(f"[green]✔ Report written to {save}[/green]")
        return

    if submit:
        _submit_with_gh(title, body)
        return

    if open_issue:
        _open_prefilled_issue(title, body)
        return

    distinct = len(summarise(entries))
    console.print(
        f"\n[dim]{len(entries)} failure(s), {distinct} distinct. "
        "Report with [bold]ucx report --open[/bold] (browser) or "
        "[bold]ucx report --submit[/bold] (GitHub CLI).[/dim]"
    )


def _open_prefilled_issue(title: str, body: str) -> None:
    """Open a pre-filled issue form. The user still presses Submit."""
    url = issue_url(body, title)
    if url is None:
        console.print(
            "[yellow]⚠ The report is too long to carry in a URL.[/yellow]\n"
            "Save it with [bold cyan]ucx report --save report.md[/bold cyan] and attach "
            f"it to a new issue at https://github.com/{ISSUE_REPO}/issues/new"
        )
        raise typer.Exit(1)

    console.print(
        "[dim]Opening a pre-filled issue. Review it — nothing is sent until you press Submit.[/dim]"
    )
    if not webbrowser.open(url):
        console.print(f"Could not open a browser. Paste this URL:\n{url}")


def _submit_with_gh(title: str, body: str) -> None:
    """File the issue through the user's own authenticated `gh`.

    Their credentials, their account, their rate limit. No token ships with
    UClone-X, so there is nothing in the distribution to leak and no way for a
    report to be filed without an account that consented to it.
    """
    if shutil.which("gh") is None:
        console.print(
            "[yellow]⚠ The GitHub CLI (`gh`) is not installed.[/yellow]\n"
            "Use [bold cyan]ucx report --open[/bold cyan] instead, or install gh from "
            "https://cli.github.com/"
        )
        raise typer.Exit(1)

    result = subprocess.run(
        ["gh", "issue", "create", "--repo", ISSUE_REPO, "--title", title, "--body", body],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        console.print(
            "[bold red]✖ `gh issue create` failed:[/bold red]\n"
            f"{result.stderr.strip() or result.stdout.strip()}\n"
            "[dim]If this is an authentication error, run `gh auth login`.[/dim]"
        )
        raise typer.Exit(1)

    console.print(f"[green]✔ Reported:[/green] {result.stdout.strip()}")
