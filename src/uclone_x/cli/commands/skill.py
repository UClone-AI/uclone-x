"""CLI commands for dynamic skills, audit reports, and human-in-the-loop approval gates."""

from __future__ import annotations

import asyncio
import datetime
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from uclone_x.skills.auditor import (
    SkillAuditor,
    SkillRegistry,
    load_skill_from_dir,
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


def _repo_root() -> Path:
    """Anchor paths to the repository root, or current directory if not in git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()


def _skills_dir(custom_path: Path | None = None) -> Path:
    """Resolve the skills root directory."""
    if custom_path is not None:
        return custom_path
    # `ucx-agent-skills`, not `skills`: the store belongs to the Runtime Layer
    # (`ucx agent`), and under the shorter name it twice collected Builder
    # workflow prose instead — which surfaces here as an unaudited `pending`
    # package and contradicts the threat model's premise that the store is empty.
    # Builder skills live in `swarm/skills/`. See `ucx-agent-skills/README.md`.
    path = _repo_root() / "ucx-agent-skills"
    path.mkdir(parents=True, exist_ok=True)
    return path


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

    skill_folders = sorted([p for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").is_file()])
    if not skill_folders:
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
    """Approve a pending/quarantined skill and promote it to active status."""
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
    if manifest.status is SkillStatus.ACTIVE and not force:
        console.print(
            f"[yellow]Skill '{name}' is already active (approved by "
            f"{_blank_if_unset(manifest.approved_by)} at {_blank_if_unset(manifest.approved_at)}).[/yellow]"
        )
        return

    # Run auditor check
    auditor = SkillAuditor(policy=AutoApprovalPolicy.SAFE_ONLY)
    report = asyncio.run(auditor.audit_skill(skill_dir))

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

    updated_manifest = manifest.model_copy(
        update={
            "status": SkillStatus.ACTIVE,
            "approved_by": approver,
            "approved_at": _now(),
            "content_sha256": report.content_sha256,
            "rejected_by": None,
            "rejected_at": None,
            "rejection_reason": None,
        }
    )

    save_skill(skill_dir, updated_manifest, skill.instructions_markdown)
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
    auto_approve: Annotated[
        bool,
        typer.Option("--auto-approve", help="Automatically approve and promote if audit passes"),
    ] = False,
    policy_str: Annotated[
        str,
        typer.Option(
            "--policy",
            "-p",
            help="Auto-approval policy to evaluate against: safe_only | never | always",
        ),
    ] = "safe_only",
) -> None:
    """Autonomously synthesize a new skill package from session traces or workflow steps."""
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

    # 4. Admission / Promotion logic
    if auto_approve:
        if report.is_safe and report.recommendation is AuditVerdict.APPROVE:
            updated_manifest = manifest.model_copy(
                update={
                    "status": SkillStatus.ACTIVE,
                    "approved_by": "synthesizer:auto",
                    "approved_at": _now(),
                    "content_sha256": report.content_sha256,
                }
            )
            skill = load_skill_from_dir(skill_dir)
            save_skill(skill_dir, updated_manifest, skill.instructions_markdown)
            registry = SkillRegistry(root)
            registry.register(load_skill_from_dir(skill_dir), report)
            console.print(
                f"[bold green]✔ Auto-approved and registered skill:[/bold green] [cyan]{name_clean}[/cyan] "
                f"(status: [green]active[/green])"
            )
        else:
            console.print(
                f"[bold yellow]⚠️ Auto-approval skipped:[/bold yellow] Audit verdict is '{report.recommendation.value}'. "
                f"Skill remains in quarantine ([yellow]pending[/yellow]). "
                f"Run [bold cyan]./ucx skill approve {name_clean}[/bold cyan] to review."
            )
    else:
        console.print(
            f"[bold cyan]ℹ Skill is quarantined ([yellow]pending[/yellow]).[/bold cyan] "
            f"Run [bold cyan]./ucx skill approve {name_clean}[/bold cyan] to audit and activate."
        )
