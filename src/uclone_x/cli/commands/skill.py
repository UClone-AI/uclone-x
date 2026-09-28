"""CLI commands for dynamic skills, audit reports, and human-in-the-loop approval gates."""

from __future__ import annotations

import asyncio
import datetime
import tempfile
from pathlib import Path
from typing import Annotated, TextIO

import typer
from rich.console import Console
from rich.table import Table

from uclone_x.errors import SkillAuditError
from uclone_x.skills.approvals import SkillApprovalLedger, SkillPin
from uclone_x.skills.auditor import (
    Skill,
    SkillAuditor,
    _is_file,  # pyright: ignore[reportPrivateUsage]
    compute_skill_sha256,
    copy_skill_package,
    load_skill_from_dir,
    runtime_skill_store_dir,
    save_skill,
)
from uclone_x.skills.models import (
    AuditVerdict,
    AutoApprovalPolicy,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.skills.synthesizer import SkillSynthesizer

skill_app = typer.Typer(
    name="skill",
    help="Manage modular dynamic skills, audit verification, and approval gates",
    no_args_is_help=True,
)

console = Console()


def _skills_dir(custom_path: Path | None = None) -> Path:
    """Resolve the skills root directory, creating it when it does not exist yet.

    The default is `runtime_skill_store_dir()`, the resolver the heads load approved skills
    from at startup, so what `ucx skill approve` promotes is what a running agent sees.
    """
    if custom_path is not None:
        return custom_path
    path = runtime_skill_store_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


#: What `skill list` says about a skill folder it may list but not look inside (#1824).
LIST_FOLDER_UNOPENED = (
    "The skill folder '{name}' could not be opened, so it is not listed. Check that you "
    "may open it."
)

#: What `skill approve` says when there is no terminal to ask the person at (#1589).
APPROVE_NEEDS_TERMINAL = (
    "Only you can approve a skill, so this command asks you in a terminal window, and it "
    "could not find one to ask in. Run it yourself in a terminal window. Nothing was changed."
)

#: What `skill approve` says when the package changed while the person was being asked (#1777).
APPROVE_CHANGED_MEANWHILE = (
    "The skill changed while you were being asked, so it was not approved. Check what "
    "changed, then run the command again."
)

#: What `skill approve` says when the person did not answer yes.
APPROVE_NOT_CONFIRMED = "The skill was not approved, because you did not answer yes."


def _open_terminal() -> TextIO:
    """This process's controlling terminal: the window a person typed the command in.

    Not standard input. A clone's `bash_run` starts every command in a new session
    (`setsid`), which leaves it with no controlling terminal, so this open fails there. Its
    standard input is another matter: it is inherited from the server, and is the person's
    terminal when `ucx ui` runs in one, so a check of `stdin.isatty()` would pass under
    `bash_run`, and `echo yes |` would answer a prompt read from it. Replaced in tests.
    """
    return open("/dev/tty", "r+", encoding="utf-8")  # noqa: SIM115 -- closed by the caller


def _confirmed_at_terminal(name: str) -> bool | None:
    """Whether the person answered yes at the terminal; None when there is none to ask at.

    What this stops, and what it does not (#1589): a program without a terminal, the
    model's `bash_run` included, cannot answer, and neither can one that pipes `yes` in.
    A program that makes itself a terminal (`script`, a pseudo-terminal) can, and so can
    one that edits the skill's files directly, which an unconfined shell may; only
    confining that shell closes those.
    """
    try:
        terminal = _open_terminal()
    except OSError:
        return None
    with terminal:
        terminal.write(f"Approve the skill '{name}'? Type yes to approve it: ")
        terminal.flush()
        answer = terminal.readline()
    return answer.strip().lower() == "yes"


def _is_pinned(ledger: SkillApprovalLedger, skill_dir: Path, name: str) -> bool:
    """Whether the package as it is now is the version approved for `name`.

    False when it cannot be told -- an unreadable package or ledger -- so `approve` goes on
    to audit it and ask, rather than calling an unapproved skill approved.
    """
    try:
        return compute_skill_sha256(skill_dir) in ledger.approved_digests(name, ledger.read())
    except SkillAuditError:
        return False


def _now() -> str:
    """Current UTC ISO timestamp."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _blank_if_unset(value: str | None) -> str:
    """Render an unset field as a dash."""
    return "-" if value in (None, "", "null") else str(value)


@skill_app.command("list")
def skill_list(
    pending_only: Annotated[
        bool,
        typer.Option(
            "--pending", "-p", help="Show only pending/quarantined skills awaiting approval"
        ),
    ] = False,
    skills_dir: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Custom skills directory path"),
    ] = None,
    show_all: Annotated[
        bool,
        typer.Option("--all", "-a", help="Show all skills including rejected ones"),
    ] = False,
) -> None:
    """List skills in the local registry with their quarantine and approval status."""
    root = _skills_dir(skills_dir)
    if not root.exists() or not root.is_dir():
        console.print(f"[yellow]Skills directory not found: {root}[/yellow]")
        return

    skill_folders: list[Path] = []
    unopened: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        # `_is_file`, not `Path.is_file()`: a folder that can be listed but not searched
        # raises on Python 3.11 to 3.13 and is answered False on 3.14. Either way the person
        # is told about it instead of the command failing or leaving it out unsaid (#1824).
        try:
            if _is_file(child / "SKILL.md"):
                skill_folders.append(child)
        except OSError:
            unopened.append(child.name)
    for folder_name in unopened:
        console.print(f"[yellow]{LIST_FOLDER_UNOPENED.format(name=folder_name)}[/yellow]")
    if not skill_folders:
        if not unopened:
            console.print(f"[yellow]No skills found in {root}[/yellow]")
        return

    table = Table(title="🧩 Dynamic Skill Registry (UClone-X)")
    table.add_column("Name", style="bold cyan", no_wrap=True)
    table.add_column("Version", style="magenta")
    table.add_column("Origin", style="blue")
    table.add_column("Status", style="bold")
    table.add_column("Isolation", style="dim")
    table.add_column("Approved By", style="dim")
    table.add_column("Description")

    shown = 0
    for folder in skill_folders:
        try:
            skill = load_skill_from_dir(folder)
        except Exception:
            continue

        manifest = skill.manifest
        status = manifest.status

        if pending_only and status not in (SkillStatus.PENDING, SkillStatus.QUARANTINED):
            continue
        if not show_all and not pending_only and status is SkillStatus.REJECTED:
            continue

        shown += 1
        if status is SkillStatus.ACTIVE:
            status_style = "[green]active[/green]"
        elif status is SkillStatus.PENDING:
            status_style = "[yellow]pending[/yellow]"
        elif status is SkillStatus.QUARANTINED:
            status_style = "[magenta]quarantined[/magenta]"
        else:
            status_style = "[red]rejected[/red]"

        origin_style = (
            "[cyan]human[/cyan]"
            if manifest.origin is SkillOrigin.HUMAN
            else "[dim]synthesized[/dim]"
        )
        isolation_str = (
            manifest.requested_isolation.value if manifest.requested_isolation else "default"
        )

        table.add_row(
            manifest.name,
            manifest.version,
            origin_style,
            status_style,
            isolation_str,
            _blank_if_unset(manifest.approved_by),
            manifest.description,
        )

    if shown == 0:
        if pending_only:
            console.print("[green]No pending skills awaiting review.[/green]")
        else:
            console.print(
                "[yellow]No skills to display. Use --all to include rejected skills.[/yellow]"
            )
        return

    console.print(table)


@skill_app.command("approve")
def skill_approve(
    name: Annotated[str, typer.Argument(help="Name of the skill to approve")],
    approver: Annotated[
        str, typer.Option("--approver", "-a", help="Approver identity")
    ] = "human:developer",
    skills_dir: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Custom skills directory path"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force approval even if auditor flags security risks"),
    ] = False,
) -> None:
    """Approve a pending/quarantined skill and promote it to active status.

    Asks the person to type yes in the terminal the command runs in, so a program with no
    terminal (a clone's `bash_run`) cannot approve on their behalf (#1589).
    """
    root = _skills_dir(skills_dir)
    skill_dir = root / name
    if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").is_file():
        console.print(f"[bold red]✖ Skill '{name}' not found in {root}[/bold red]")
        raise typer.Exit(code=1)

    try:
        skill = load_skill_from_dir(skill_dir)
    except Exception as exc:
        console.print(f"[bold red]✖ Failed to load skill '{name}':[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    manifest = skill.manifest
    ledger = SkillApprovalLedger()
    if (
        manifest.status is SkillStatus.ACTIVE
        and not force
        and _is_pinned(ledger, skill_dir, manifest.name)
    ):
        console.print(
            f"[yellow]Skill '{name}' is already active (approved by "
            f"{_blank_if_unset(manifest.approved_by)} at {_blank_if_unset(manifest.approved_at)}).[/yellow]"
        )
        return

    # Everything from the audit to the pin works on one copy of the package, each file read
    # once (#1777): the person is asked about the bytes the auditor checked, and the pin is
    # of those bytes, however long the question waits and whatever changes the package
    # meanwhile.
    with tempfile.TemporaryDirectory(prefix="ucx-skill-approve-") as scratch:
        checked_dir = Path(scratch) / skill_dir.name
        try:
            copy_skill_package(skill_dir, checked_dir)
            checked = load_skill_from_dir(checked_dir)
        except SkillAuditError as unread:
            console.print(f"[bold red]✖ Cannot approve skill '{name}':[/bold red] {unread}")
            raise typer.Exit(code=1) from unread
        _approve_checked(name, skill_dir, checked_dir, checked, approver, force, ledger)


def _write_skill_md(skill_dir: Path, data: bytes) -> None:
    """Put the approved `SKILL.md` in the package. Replaced in a test to change it meanwhile."""
    (skill_dir / "SKILL.md").write_bytes(data)


def _approve_checked(
    name: str,
    skill_dir: Path,
    checked_dir: Path,
    skill: Skill,
    approver: str,
    force: bool,
    ledger: SkillApprovalLedger,
) -> None:
    """Audit the copy in `checked_dir`, ask, and pin the digest of what was audited."""
    manifest = skill.manifest
    auditor = SkillAuditor(policy=AutoApprovalPolicy.SAFE_ONLY)
    try:
        report = asyncio.run(auditor.audit_skill(checked_dir))
    except SkillAuditError as exc:
        # The skill stays as it was on disk: an audit that could not finish approves nothing.
        console.print(f"[bold red]✖ Cannot approve skill '{name}':[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if report.recommendation is AuditVerdict.REJECT and not force:
        console.print(
            f"[bold red]✖ Cannot approve skill '{name}': Skill Auditor verdict is REJECT.[/bold red]"
        )
        if report.detected_risks:
            console.print("[yellow]Detected risks:[/yellow]")
            for risk in report.detected_risks:
                console.print(f"  • {risk}")
        console.print(
            "[yellow]Use --force to override auditor verdict and approve anyway.[/yellow]"
        )
        raise typer.Exit(code=1)

    # Asked last, after the audit's findings are on screen: the person decides with them.
    confirmed = _confirmed_at_terminal(name)
    if confirmed is None:
        console.print(f"[bold red]✖ {APPROVE_NEEDS_TERMINAL}[/bold red]")
        raise typer.Exit(code=1)
    if not confirmed:
        console.print(f"[yellow]{APPROVE_NOT_CONFIRMED}[/yellow]")
        raise typer.Exit(code=1)

    approved_at = _now()
    updated_manifest = manifest.model_copy(
        update={
            "status": SkillStatus.ACTIVE,
            "approved_by": approver,
            "approved_at": approved_at,
            "content_sha256": None,
            "rejected_by": None,
            "rejected_at": None,
            "rejection_reason": None,
        }
    )

    # The digest covers the file as approval leaves it -- `status: active` and the approver
    # included -- so it is taken after the write, of the checked copy. Writing it into the
    # file afterwards does not change it (the rule in `compute_skill_sha256`), and the pin in
    # the ledger, not the copy in the file, is what a load checks (#1720).
    try:
        save_skill(checked_dir, updated_manifest, skill.instructions_markdown)
        digest = compute_skill_sha256(checked_dir)
        save_skill(
            checked_dir,
            updated_manifest.model_copy(update={"content_sha256": digest}),
            skill.instructions_markdown,
        )
        if compute_skill_sha256(skill_dir) != report.content_sha256:
            console.print(f"[bold red]✖ {APPROVE_CHANGED_MEANWHILE}[/bold red]")
            raise typer.Exit(code=1)
        _write_skill_md(skill_dir, (checked_dir / "SKILL.md").read_bytes())
        ledger.pin(
            manifest.name,
            SkillPin(content_sha256=digest, approved_by=approver, approved_at=approved_at),
        )
    except (SkillAuditError, OSError) as exc:
        detail = str(exc) if isinstance(exc, SkillAuditError) else "its files could not be saved"
        console.print(f"[bold red]✖ Cannot approve skill '{name}':[/bold red] {detail}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[bold green]✔ Approved skill:[/bold green] [cyan]{name}[/cyan] "
        f"(status: [green]active[/green], approver: [magenta]{approver}[/magenta])"
    )


@skill_app.command("reject")
def skill_reject(
    name: Annotated[str, typer.Argument(help="Name of the skill to reject")],
    reason: Annotated[
        str, typer.Option("--reason", "-r", help="Reason for rejection")
    ] = "Rejected by developer review",
    rejecter: Annotated[
        str, typer.Option("--rejecter", help="Rejecter identity")
    ] = "human:developer",
    skills_dir: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Custom skills directory path"),
    ] = None,
) -> None:
    """Reject a skill and record rejection provenance in its manifest."""
    root = _skills_dir(skills_dir)
    skill_dir = root / name
    if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").is_file():
        console.print(f"[bold red]✖ Skill '{name}' not found in {root}[/bold red]")
        raise typer.Exit(code=1)

    try:
        skill = load_skill_from_dir(skill_dir)
    except Exception as exc:
        console.print(f"[bold red]✖ Failed to load skill '{name}':[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    manifest = skill.manifest
    try:
        SkillApprovalLedger().revoke(manifest.name)
    except SkillAuditError as error:
        console.print(f"[bold red]✖ Cannot reject skill '{name}':[/bold red] {error}")
        raise typer.Exit(code=1) from error
    updated_manifest = manifest.model_copy(
        update={
            "status": SkillStatus.REJECTED,
            "rejected_by": rejecter,
            "rejected_at": _now(),
            "rejection_reason": reason,
        }
    )

    save_skill(skill_dir, updated_manifest, skill.instructions_markdown)
    console.print(
        f"[bold green]✔ Rejected skill:[/bold green] [cyan]{name}[/cyan] "
        f"(reason: [yellow]{reason}[/yellow])"
    )


@skill_app.command("audit")
def skill_audit(
    name: Annotated[str, typer.Argument(help="Name of the skill to audit")],
    policy_str: Annotated[
        str,
        typer.Option(
            "--policy",
            "-p",
            help="Auto-approval policy to evaluate against: safe_only | never | always",
        ),
    ] = "safe_only",
    skills_dir: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Custom skills directory path"),
    ] = None,
) -> None:
    """Run security and safety audit on a skill package."""
    root = _skills_dir(skills_dir)
    skill_dir = root / name
    if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").is_file():
        console.print(f"[bold red]✖ Skill '{name}' not found in {root}[/bold red]")
        raise typer.Exit(code=1)

    try:
        policy = AutoApprovalPolicy(policy_str)
    except ValueError:
        console.print(
            f"[bold red]✖ Invalid policy:[/bold red] '{policy_str}'. "
            f"Must be one of: {', '.join(p.value for p in AutoApprovalPolicy)}"
        )
        raise typer.Exit(code=1) from None

    auditor = SkillAuditor(policy=policy)
    try:
        report = asyncio.run(auditor.audit_skill(skill_dir))
    except Exception as exc:
        console.print(f"[bold red]✖ Audit failed with error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"🛡️ Skill Audit Report: {name}")
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")

    is_safe_style = "[green]True[/green]" if report.is_safe else "[red]False[/red]"
    if report.recommendation is AuditVerdict.APPROVE:
        verdict_style = "[green]approve[/green]"
    elif report.recommendation is AuditVerdict.REQUIRE_HUMAN_REVIEW:
        verdict_style = "[yellow]require_human_review[/yellow]"
    else:
        verdict_style = "[red]reject[/red]"

    table.add_row("Skill Name", report.skill_name)
    table.add_row("Safe", is_safe_style)
    table.add_row("Verdict", verdict_style)
    table.add_row(
        "Risk Score",
        f"{report.risk_score:.2f}" if report.risk_score is not None else "[dim]None[/dim]",
    )
    table.add_row("Policy Evaluated", policy.value)
    table.add_row("Content SHA256", report.content_sha256 or "[dim]None[/dim]")

    if report.detected_risks:
        risks_formatted = "\n".join(f"[yellow]•[/yellow] {r}" for r in report.detected_risks)
        table.add_row("Detected Risks", risks_formatted)
    else:
        table.add_row("Detected Risks", "[green]None detected[/green]")

    console.print(table)

    if report.recommendation is AuditVerdict.APPROVE:
        console.print(
            f"[bold green]✔ Audit passed:[/bold green] Skill is safe and approved under '{policy.value}' policy."
        )
    elif report.recommendation is AuditVerdict.REQUIRE_HUMAN_REVIEW:
        console.print(
            f"[bold yellow]⚠️ Human Review Required:[/bold yellow] Skill requires explicit human approval. "
            f"Run [bold cyan]./ucx skill approve {name}[/bold cyan] to approve."
        )
    else:
        console.print(
            "[bold red]✖ Audit Failed:[/bold red] Skill was rejected due to detected risks."
        )


@skill_app.command("synthesize")
def skill_synthesize(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Name of the skill to synthesize"),
    ],
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", "-s", help="Session ID to extract workflow from"),
    ] = None,
    from_trace: Annotated[
        Path | None,
        typer.Option(
            "--from-trace",
            "-t",
            help="Path to trace file (JSON/YAML) to extract workflow from",
        ),
    ] = None,
    steps: Annotated[
        list[str] | None,
        typer.Option("--step", help="Explicit workflow step (can be specified multiple times)"),
    ] = None,
    description: Annotated[
        str | None,
        typer.Option("--description", help="Description for the synthesized skill"),
    ] = None,
    skills_dir: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Custom skills directory path"),
    ] = None,
    policy_str: Annotated[
        str,
        typer.Option(
            "--policy",
            "-p",
            help="Auto-approval policy to evaluate against: safe_only | never | always",
        ),
    ] = "safe_only",
) -> None:
    """Synthesize a pending, prompt-only skill from a session's trace or given steps.

    The package is left pending: `./ucx skill approve` is the only way it becomes active.
    """
    name_clean = name.strip()
    if not name_clean:
        console.print("[bold red]✖ Skill name cannot be empty.[/bold red]")
        raise typer.Exit(code=1)

    if not session_id and not from_trace and not steps:
        console.print(
            "[bold red]✖ Must provide either --session-id, --from-trace, or --step to synthesize a skill.[/bold red]"
        )
        raise typer.Exit(code=1)

    try:
        policy = AutoApprovalPolicy(policy_str)
    except ValueError:
        console.print(
            f"[bold red]✖ Invalid policy:[/bold red] '{policy_str}'. "
            f"Must be one of: {', '.join(p.value for p in AutoApprovalPolicy)}"
        )
        raise typer.Exit(code=1) from None

    root = _skills_dir(skills_dir)
    skill_dir = root / name_clean
    auditor = SkillAuditor(policy=policy)
    synthesizer = SkillSynthesizer(auditor=auditor)

    # 1. Extract workflow steps
    try:
        if from_trace is not None:
            workflow_steps = synthesizer.extract_workflow_from_trace(from_trace)
        elif session_id is not None:
            workflow_steps = synthesizer.extract_workflow_from_session(session_id)
        elif steps:
            workflow_steps = list(steps)
        else:
            workflow_steps = []
    except Exception as exc:
        console.print(f"[bold red]✖ Failed to extract workflow steps:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not workflow_steps:
        console.print("[bold red]✖ No workflow steps extracted.[/bold red]")
        raise typer.Exit(code=1)

    # 2. Synthesize skill package into quarantine
    try:
        manifest = asyncio.run(
            synthesizer.synthesize_skill(
                task_name=name_clean,
                workflow_steps=workflow_steps,
                quarantine_dir=root,
                description=description,
            )
        )
    except Exception as exc:
        console.print(f"[bold red]✖ Failed to synthesize skill '{name_clean}':[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold green]✔ Synthesized skill package in quarantine:[/bold green] [cyan]{name_clean}[/cyan] "
        f"(status: [yellow]{manifest.status.value}[/yellow])"
    )

    # 3. Mandatory security audit via SkillAuditor
    try:
        report = asyncio.run(auditor.audit_skill(skill_dir))
    except Exception as exc:
        console.print(f"[bold red]✖ Security audit failed for synthesized skill:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"🛡️ Skill Synthesis & Audit Report: {name_clean}")
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")

    is_safe_style = "[green]True[/green]" if report.is_safe else "[red]False[/red]"
    if report.recommendation is AuditVerdict.APPROVE:
        verdict_style = "[green]approve[/green]"
    elif report.recommendation is AuditVerdict.REQUIRE_HUMAN_REVIEW:
        verdict_style = "[yellow]require_human_review[/yellow]"
    else:
        verdict_style = "[red]reject[/red]"

    table.add_row("Skill Name", name_clean)
    table.add_row("Package Path", str(skill_dir))
    table.add_row("Workflow Steps", str(len(workflow_steps)))
    table.add_row("Safe", is_safe_style)
    table.add_row("Audit Verdict", verdict_style)
    table.add_row(
        "Risk Score",
        f"{report.risk_score:.2f}" if report.risk_score is not None else "[dim]None[/dim]",
    )
    table.add_row("Content SHA256", report.content_sha256 or "[dim]None[/dim]")

    if report.detected_risks:
        risks_formatted = "\n".join(f"[yellow]•[/yellow] {r}" for r in report.detected_risks)
        table.add_row("Detected Risks", risks_formatted)
    else:
        table.add_row("Detected Risks", "[green]None detected[/green]")

    console.print(table)

    # 4. Always left pending: an LLM-authored skill is never auto-approved (#1824, owner
    # ruling 2026-09-27). Only `approve` makes it active, and only its pin makes it load.
    console.print(
        f"[bold cyan]ℹ Skill is quarantined ([yellow]pending[/yellow]).[/bold cyan] "
        f"Run [bold cyan]./ucx skill approve {name_clean}[/bold cyan] to audit and activate."
    )
