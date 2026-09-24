"""Builder-facing task commands, backed by the development repository's project board.

A task is a GitHub issue on the board configured in ``swarm/config.yaml``. There is no
second place to look: the Markdown registers these commands used to read and write —
``docs/tasks/`` and the ``docs/issues/`` findings register — are retired (#1037). The
findings register survives only as an immutable eval fixture, and nothing in this module
reads it.

``--github``/``-g`` selected the board back when a file register was the other option.
It is still accepted on every subcommand, as a no-op, so existing scripts and guides keep
working; it no longer selects anything.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Literal, cast

import typer
from rich.console import Console
from rich.table import Table

from uclone_x.cli.commands.session_dev import dev_logs, session_app
from uclone_x.cli.quality_gate import run_quality_gate

#: Help text for the retired backend selector. Named once so every subcommand says the
#: same thing: the flag is accepted, and it chooses nothing.
GITHUB_FLAG_HELP = "Accepted for compatibility and ignored: the project board is the only backend"


def _board_config() -> tuple[str | None, str | None]:
    """The project board URL and repository slug, from `swarm/config.yaml`.

    Read rather than hardcoded for two reasons. The slug was hardcoded as
    `UClone-AI/uclone-x` and went **stale at the repository rename** — it named the
    published repository while the development repository became `-lab`, so
    `ucx dev task list` would have queried the wrong one. And this module ships inside
    the distribution, where a hardcoded URL publishes the address of a board a public
    reader cannot open. `swarm/config.yaml` is the file the rename updates, and it is
    present exactly where these commands are usable: inside a checkout.
    """
    try:
        import yaml

        root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
                env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
            ).stdout.strip()
        )
        loaded: object = yaml.safe_load(
            (root / "swarm" / "config.yaml").read_text(encoding="utf-8")
        )
    except Exception:
        return None, None
    if not isinstance(loaded, dict):
        return None, None
    config = cast("dict[str, object]", loaded)
    board_value = config.get("board")
    board = cast("dict[str, object]", board_value) if isinstance(board_value, dict) else {}
    owner, number = board.get("owner"), board.get("project_number")
    url = f"https://github.com/orgs/{owner}/projects/{number}" if owner and number else None
    repo = config.get("repo")
    return url, repo if isinstance(repo, str) else None


dev_app = typer.Typer(
    name="dev",
    help="Autonomous Builder Task & Worktree Management",
    no_args_is_help=True,
)
task_app = typer.Typer(
    name="task",
    help="Manage Builder implementation tasks on the project board",
    no_args_is_help=True,
)
dev_app.add_typer(task_app, name="task")
dev_app.add_typer(session_app, name="session")
dev_app.command("logs")(dev_logs)

console = Console()


def _print_github_notice() -> None:
    """Point at the project board, naming it only when configuration supplies it."""
    url, _ = _board_config()
    where = f" [cyan]{url}[/cyan]" if url else " (see `swarm/config.yaml`)"
    console.print(
        "[bold yellow]📌 Notice: Tasks are managed on the project board:[/bold yellow]"
        f"{where}\n"
        "[dim]   The file-backed task and findings registers are retired; these commands "
        "read and write GitHub only.[/dim]"
    )


def _repo_root() -> Path:
    """Anchor paths to the repository root, not the caller's working directory."""
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


#: What `link_frontend_node_modules` did for a new worktree. `absent`: the tree has no
#: frontend. `present`: something is already at `frontend/node_modules`, left alone.
#: `linked`: the symlink was created. `unlinked`: it could not be, and the lines say how
#: to fix it.
FrontendDepsLink = Literal["absent", "present", "linked", "unlinked"]


def link_frontend_node_modules(worktree: Path) -> tuple[FrontendDepsLink, list[str]]:
    """Link a new worktree's `frontend/node_modules` to the primary workspace's copy.

    The gate's vitest stage fails when `frontend/node_modules` is missing (#915), and a
    fresh worktree never has it, because it is untracked. Without this link a builder's
    first push fails on a known, mechanical step (#930). The primary copy is shared the
    way the `.venv` is. The primary workspace is found through `--git-common-dir`, not
    `_repo_root()`, because a claim run from inside a worktree has that worktree as its
    toplevel.

    Anything already at `frontend/node_modules` is left alone and never followed: a real
    directory, a link, or a dangling link. A dangling link is reported as `unlinked`,
    because the gate reads it as missing dependencies. This is the same rule as the shell
    form in the dev-builder skill. A plain `ln -s` run where a link already exists follows
    it, and creates a `node_modules/node_modules` link inside the primary workspace's copy.

    When the primary workspace has no `node_modules`, nothing is linked, and the returned
    lines say so and give the commands that fix it, rather than skipping silently (P6).
    The claim still succeeds: the worktree exists and is usable, and the gate fails
    loudly for as long as the dependencies stay missing.
    """
    frontend = worktree / "frontend"
    if not (frontend / "package.json").is_file():
        return "absent", []
    link = frontend / "node_modules"
    not_linked = "[bold yellow]⚠️  frontend/node_modules was NOT linked:[/bold yellow]"
    if link.is_symlink() or link.exists():
        if not link.exists():
            # Not `present`: the gate reads a dangling link as missing dependencies.
            dangling = f"{not_linked} [yellow]{link} is a dangling symlink; left as is.[/yellow]"
            return "unlinked", [dangling, f"[yellow]  Fix: rm {link}, then link again.[/yellow]"]
        return "present", [f"[dim]{link} already exists; left as is, not followed.[/dim]"]

    install_here = [
        "[yellow]  Or install into this worktree (needed when its package-lock.json "
        "differs from the primary's):[/yellow]",
        f"[yellow]    npm ci --prefix {frontend}[/yellow]",
        "[yellow]  The gate's vitest stage fails until one of these is done (#915).[/yellow]",
    ]
    try:
        common_dir = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return "unlinked", [
            f"{not_linked} [yellow]the primary workspace could not be located: {exc}[/yellow]",
            *install_here,
        ]

    primary_frontend = Path(common_dir).parent / "frontend"
    source = primary_frontend / "node_modules"
    link_it = [
        "[yellow]  Fix: install the primary workspace's copy, then link it:[/yellow]",
        f"[yellow]    npm ci --prefix {primary_frontend}[/yellow]",
        f'[yellow]    ln -s "{source}" "{link}"[/yellow]',
    ]
    if not source.is_dir():
        missing = f"{not_linked} [yellow]{source} does not exist.[/yellow]"
        return "unlinked", [missing, *link_it, *install_here]
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        return "unlinked", [f"{not_linked} [yellow]{exc}[/yellow]", *link_it, *install_here]
    return "linked", [
        f"[bold green]✔ Linked frontend/node_modules[/bold green] → [cyan]{source}[/cyan]"
    ]


def _print_frontend_link(worktree: Path) -> None:
    for line in link_frontend_node_modules(worktree)[1]:
        # `soft_wrap`: the remedy is a shell command, and a hard wrap breaks the paste.
        console.print(line, soft_wrap=True)


def _render_github_issues(show_all: bool = False) -> None:
    """Fetch and render live tasks/issues from GitHub via gh CLI."""
    state_arg = "all" if show_all else "open"
    try:
        proc = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--state",
                state_arg,
                "--limit",
                "50",
                "--json",
                "number,title,state,assignees,labels",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        err_msg = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
            else str(exc)
        )
        console.print(
            f"[bold red]✖ GitHub CLI ('gh') failed to fetch issues:[/bold red] {err_msg}\n"
            "[yellow]Ensure 'gh' is installed and authenticated ('gh auth login').[/yellow]"
        )
        raise typer.Exit(code=1) from exc

    try:
        raw_issues = cast(list[dict[str, object]], json.loads(proc.stdout))
    except Exception as exc:
        console.print(f"[bold red]✖ Failed to parse GitHub issues JSON:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not raw_issues:
        console.print("[yellow]No issues found on GitHub.[/yellow]")
        return

    _, repo = _board_config()
    table = Table(title=f"🐙 Live GitHub Tasks / Issues ({repo or 'unconfigured'})")
    table.add_column("ID", style="bold cyan")
    table.add_column("State", style="bold")
    table.add_column("Assignees", style="dim")
    table.add_column("Labels / Module", style="blue")
    table.add_column("Title")

    for item in raw_issues:
        num = f"#{item.get('number', '?')}"
        state_str = str(item.get("state", "OPEN")).upper()
        state_style = "[green]OPEN[/green]" if state_str == "OPEN" else f"[dim]{state_str}[/dim]"

        raw_assignees = item.get("assignees")
        assignees_list: list[str] = []
        if isinstance(raw_assignees, list):
            for a in cast(list[object], raw_assignees):
                if isinstance(a, dict):
                    login_val = cast(dict[str, object], a).get("login")
                    if login_val is not None:
                        assignees_list.append(str(login_val))
                elif isinstance(a, str):
                    assignees_list.append(a)
        assignees_str = ", ".join(assignees_list) if assignees_list else "-"

        raw_labels = item.get("labels")
        labels_list: list[str] = []
        if isinstance(raw_labels, list):
            for label_obj in cast(list[object], raw_labels):
                if isinstance(label_obj, dict):
                    name_val = cast(dict[str, object], label_obj).get("name")
                    if name_val is not None:
                        labels_list.append(str(name_val))
                elif isinstance(label_obj, str):
                    labels_list.append(label_obj)
        labels_str = ", ".join(labels_list) if labels_list else "-"

        title = str(item.get("title", ""))
        table.add_row(num, state_style, assignees_str, labels_str, title)

    console.print(table)


@task_app.command("create")
def task_create(
    title: str = typer.Argument(..., help="Task title"),
    task_type: str = typer.Option("task", "--type", "-t", help="task | bug | refactor | security"),
    module: str = typer.Option("core", "--module", "-m", help="Target module"),
    description: str = typer.Option("", "--desc", "-d", help="Short description"),
    resolves: str = typer.Option(
        "", "--resolves", "-r", help="Comma-separated finding IDs this task addresses"
    ),
    github: bool = typer.Option(False, "--github", "-g", help=GITHUB_FLAG_HELP),
) -> None:
    """Create a task as a GitHub issue on the project board."""
    _print_github_notice()

    body = (
        f"## Description\n{description or title}\n\n## Module\n`{module}`\n\n## Type\n`{task_type}`"
    )
    if resolves:
        body += f"\n\n## Resolves\n{resolves}"
    try:
        cmd = [
            "gh",
            "issue",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--label",
            f"module:{module}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        err = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
            else str(exc)
        )
        console.print(f"[bold red]✖ Failed to create GitHub issue via gh:[/bold red] {err}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold green]✔ Created GitHub Issue:[/bold green] {proc.stdout.strip()}")


@task_app.command("list")
def task_list(
    show_all: bool = typer.Option(False, "--all", "-a", help="Include resolved and closed tasks"),
    github: bool = typer.Option(False, "--github", "-g", help=GITHUB_FLAG_HELP),
) -> None:
    """List Builder implementation tasks from the project board."""
    _print_github_notice()
    _render_github_issues(show_all=show_all)


@task_app.command("claim")
def task_claim(
    task_id: str = typer.Argument(..., help="GitHub issue number, e.g. 20"),
    assignee: str = typer.Option(
        ..., "--assignee", "-a", help="Builder identity, e.g. builder:<pool>-<n>"
    ),
    worktree: str | None = typer.Option(None, "--worktree", "-w", help="Worktree path"),
    create_worktree: bool = typer.Option(
        False, "--create-worktree", help="Automatically create git worktree"
    ),
    github: bool = typer.Option(False, "--github", "-g", help=GITHUB_FLAG_HELP),
) -> None:
    """Claim an open task before starting work on it."""
    _print_github_notice()

    if not assignee or not assignee.strip():
        console.print("[bold red]✖ Assignee cannot be empty.[/bold red]")
        raise typer.Exit(code=1)

    issue_num = task_id.lstrip("#")
    repo_root = _repo_root()
    should_create_worktree = create_worktree or (worktree is not None and bool(worktree.strip()))
    if should_create_worktree:
        builder_slug = re.sub(r"[^a-z0-9]+", "-", assignee.lower()).strip("-")
        if worktree is not None and worktree.strip():
            target_wt_str = worktree.strip()
        else:
            target_wt_str = f".worktrees/{builder_slug}"

        target_wt_path = (repo_root / target_wt_str).resolve()
        if target_wt_path.exists():
            console.print(f"[bold red]✖ Target worktree already exists:[/bold red] {target_wt_str}")
            raise typer.Exit(code=1)

        branch_name = f"task/{issue_num}-{builder_slug}"
        try:
            subprocess.run(
                ["git", "worktree", "add", target_wt_str, "-b", branch_name],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            err_msg = exc.stderr.strip() if exc.stderr else exc.stdout.strip()
            console.print(f"[bold red]✖ Failed to create worktree:[/bold red] {err_msg}")
            raise typer.Exit(code=1) from exc

        console.print(
            f"[bold green]✔ Created worktree:[/bold green] [cyan]{target_wt_str}[/cyan] on branch [yellow]{branch_name}[/yellow]"
        )
        _print_frontend_link(target_wt_path)

    try:
        cmd = [
            "gh",
            "issue",
            "comment",
            issue_num,
            "--body",
            f"Claimed by `{assignee.strip()}`",
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        err = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
            else str(exc)
        )
        console.print(f"[yellow]⚠️  Notice: GitHub comment update failed:[/yellow] {err}")

    console.print(
        f"[bold green]✔ Claimed GitHub Issue:[/bold green] [cyan]#{issue_num}[/cyan] by [magenta]{assignee.strip()}[/magenta]"
    )


@task_app.command("merge")
def task_merge(
    task_id: str = typer.Argument(..., help="GitHub issue/PR number, e.g. 20"),
    branch: str | None = typer.Option(
        None, "--branch", "-b", help="Branch or PR to merge (default: the task ID)"
    ),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Bypass automatic quality gate verification"
    ),
    github: bool = typer.Option(False, "--github", "-g", help=GITHUB_FLAG_HELP),
) -> None:
    """Merge the task's pull request, deleting its branch, after the quality gate passes.

    There is no `--comment`: the option only ever appended a resolution note to the
    `docs/tasks/` document, and with that register retired it would be a flag that
    accepts text and discards it (P6). Comment on the PR with `gh pr comment`.
    """
    _print_github_notice()

    if not no_verify:
        console.print("[bold cyan]🔍 Running automated quality gate before PR merge...[/bold cyan]")
        gate_code = run_quality_gate()
        if gate_code != 0:
            console.print(
                f"\n[bold red]✖ Cannot merge PR for task {task_id}: Quality gate failed (exit code {gate_code}).[/bold red]\n"
                "[yellow]Fix failing checks, type errors, or test regressions before merging, or use --no-verify.[/yellow]"
            )
            raise typer.Exit(code=1)

    target = branch or task_id.lstrip("#")
    try:
        cmd = ["gh", "pr", "merge", target, "--rebase", "--delete-branch"]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        err = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
            else str(exc)
        )
        console.print(f"[bold red]✖ Failed to merge PR via gh:[/bold red] {err}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold green]✔ Merged PR on GitHub:[/bold green] {target}")


@task_app.command("resolve")
def task_resolve(
    task_id: str = typer.Argument(..., help="GitHub issue number, e.g. 20"),
    comment: str | None = typer.Option(None, "--comment", "-c", help="Resolution comment"),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Bypass automatic quality gate verification"
    ),
    github: bool = typer.Option(False, "--github", "-g", help=GITHUB_FLAG_HELP),
) -> None:
    """Close the task's GitHub issue. Runs the quality gate unless --no-verify is passed."""
    _print_github_notice()

    if not no_verify:
        console.print(
            "[bold cyan]🔍 Running automated quality gate before task resolution...[/bold cyan]"
        )
        gate_code = run_quality_gate()
        if gate_code != 0:
            console.print(
                f"\n[bold red]✖ Cannot resolve task {task_id}: Quality gate failed (exit code {gate_code}).[/bold red]\n"
                "[yellow]Fix failing checks, type errors, or test regressions before resolving, or use --no-verify.[/yellow]"
            )
            raise typer.Exit(code=1)

    issue_num = task_id.lstrip("#")
    try:
        cmd = ["gh", "issue", "close", issue_num]
        if comment:
            cmd.extend(["--comment", comment])
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        err = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
            else str(exc)
        )
        console.print(f"[bold red]✖ Failed to resolve GitHub issue via gh:[/bold red] {err}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold green]✔ Resolved/closed GitHub Issue:[/bold green] #{issue_num}")
