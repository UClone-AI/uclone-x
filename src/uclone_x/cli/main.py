"""Main CLI entrypoint for UClone-X (ucx)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Annotated, cast

try:
    import typer
    from rich.console import Console
except ImportError as exc:
    pkg = "typer" if "typer" in str(exc) else "rich"
    from uclone_x.errors import MissingDependencyError

    raise MissingDependencyError(
        extra="cli",
        package=pkg,
        feature="CLI shell (ucx)",
    ) from exc

from uclone_x.agent.persona_store import DEFAULT_PERSONA_NAME
from uclone_x.cli.commands.a2a import a2a_app
from uclone_x.cli.commands.acp import AGENT_ID_HELP as ACP_AGENT_ID_HELP
from uclone_x.cli.commands.acp import acp_app, start_acp_server
from uclone_x.cli.commands.dev import dev_app
from uclone_x.cli.commands.eval import eval_app
from uclone_x.cli.commands.llm import llm_app
from uclone_x.cli.commands.loop import loop_app
from uclone_x.cli.commands.media import media_app
from uclone_x.cli.commands.ontology import ontology_app
from uclone_x.cli.commands.report import report_app
from uclone_x.cli.commands.room import room_app
from uclone_x.cli.commands.skill import skill_app
from uclone_x.cli.commands.ui import ui_app
from uclone_x.cli.quality_gate import (
    describe_installed_hook_drift,
    gate_run_can_record,
    record_gate_pass,
    run_quality_gate,
    snapshot_committed_tree,
)
from uclone_x.core.failure_journal import install_excepthook

PRE_COMMIT_HOOK = """#!/usr/bin/env bash
# UClone-X automated pre-commit quality gate
set -e
# Primary-workspace invariant (development repository, swarm guide §6): pinned
# to `main` and is read-only for every agent. Only there do --git-dir and
# --git-common-dir resolve to the same directory; in a worktree the first is
# .git/worktrees/<name>. Refuse the commit rather than let it land where nothing should.
git_dir="$(cd "$(git rev-parse --git-dir)" && pwd -P)"
common_dir="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"
if [ "$git_dir" = "$common_dir" ]; then
  echo "pre-commit: refusing to commit in the primary workspace ($(pwd))." >&2
  echo "pre-commit: all work happens in a worktree:" >&2
  echo "pre-commit:   git worktree add .worktrees/<slug> -b task/<id>-<slug> origin/main" >&2
  echo "pre-commit: see AGENTS.md §2, 'The Primary Workspace Is Pinned' (greppable prefix)." >&2
  exit 1
fi

# --- Refusal 2, independent of the one above: the commit's author identity. ---
# `ghx` makes GitHub *API* authorship structural, but `git commit` never goes through it,
# and the shared .git/config has carried a non-bot identity that GitHub matches to a real
# human account — the #99 / #105 / #122 collision class, arriving through the one door ghx
# cannot watch. Squash-merge rewrites the author to the merging identity, so `main` looks
# clean and the exposure lives only in branch commits: invisible exactly where people check.
# Per-commit `-c` overrides are not a control, for the reason ghx's own header gives —
# remembering to type it every time is not a control. This is.
# `|| true` on both: under `set -e` a failing substitution would abort the hook at the
# assignment, which exits 1 — fail-closed, but *silently*, printing none of the guidance
# below. A refusal a builder cannot read is a blocked builder, so the failure is captured
# here and reported by the emptiness check instead.
manifest="$(git rev-parse --show-toplevel 2>/dev/null)/swarm/config.yaml"
read_identity() {
  awk -v field="$1" '
    /^bot:/ {in_bot=1; next}
    /^[^[:space:]]/ {in_bot=0; in_id=0}
    in_bot && /^[[:space:]]+commit_identity:/ {in_id=1; next}
    in_id && $1 == field":" {
      sub(/^[[:space:]]*[a-z_]+:[[:space:]]*/, "")
      gsub(/^"|"$/, "")
      print; exit
    }
  ' "$manifest" 2>/dev/null || true
}
expected_name="$(read_identity name || true)"
expected_email="$(read_identity email || true)"
if [ -z "$expected_name" ] || [ -z "$expected_email" ]; then
  echo "pre-commit: cannot read bot.commit_identity from swarm/config.yaml." >&2
  echo "pre-commit: refusing rather than guessing — an unreadable identity rule must not" >&2
  echo "pre-commit: read as 'no rule'. Check the file exists and declares bot.commit_identity." >&2
  exit 1
fi
actual_name="$(git var GIT_AUTHOR_IDENT | sed 's/ <.*//')"
actual_email="$(git var GIT_AUTHOR_IDENT | sed 's/.*<//; s/>.*//')"
if [ "$actual_name" != "$expected_name" ] || [ "$actual_email" != "$expected_email" ]; then
  echo "pre-commit: refusing a commit authored as '$actual_name <$actual_email>'." >&2
  echo "pre-commit: this repository's commits are authored by the bot:" >&2
  echo "pre-commit:   $expected_name <$expected_email>" >&2
  echo "pre-commit: a non-bot author is matched by GitHub to a real human account and" >&2
  echo "pre-commit: manufactures a human-colliding record (#99, #105, #122)." >&2
  echo "pre-commit: commit with the bot identity:" >&2
  echo "pre-commit:   git -c user.name=\\"$expected_name\\" \\\\" >&2
  echo "pre-commit:       -c user.email=\\"$expected_email\\" \\\\" >&2
  echo "pre-commit:       commit -m \\"<your message>\\"" >&2
  echo "pre-commit: or set it once — note this writes the SHARED .git/config, which every" >&2
  echo "pre-commit: worktree and the primary workspace read (extensions.worktreeConfig is" >&2
  echo "pre-commit: unset, so there is no per-worktree config to write):" >&2
  echo "pre-commit:   ./ucx setup            # does exactly this, and installs the hooks" >&2
  echo "pre-commit: see swarm/config.yaml 'bot.commit_identity'." >&2
  exit 1
fi

# Static checks only (pre-check policy 4.1, development repository). Formatting, lint and
# strict typing survive a rebase; a test result does not. The tree a commit holds is not
# the squashed, rebased tree that reaches `main`, so running the suite here verifies
# something that never lands — measured on 2026-09-04, the gate ran 110 times on such
# trees and 0 times on the 60 merge results. The suite runs at the PR head immediately
# before merge (#966), and a passing run on a clean commit records itself.
./ucx test check --skip-tests
"""
"""Content of the pre-commit hook installed by `ucx setup`.

Kept as a module constant so the hook has a tracked, reviewable source: an edit to it is
a diff, which the in-place edit of the live `.git/hooks/pre-commit` on 2026-09-03 (#285)
was not. The test suite **executes** this constant against real commits rather than
asserting on its text — text assertions cannot tell a hook that refuses from one that
merely contains the words of a refusal. The guard is a discipline, not a control:
`git commit --no-verify` bypasses it (P8).
"""

PRE_PUSH_HOOK = """#!/usr/bin/env bash
# UClone-X pre-push guard: `main` changes only through a reviewed PR.
# In this repository merges land server-side via `gh pr merge`, so a local push that
# updates refs/heads/main is by construction work that skipped review — a direct commit,
# or a fast-forward merge made in the primary workspace, which the pre-commit hook cannot
# see (it fires on `git commit` only). Swarm Guide §8.2; AGENTS.md §2.
# stdin: <local ref> <local sha> <remote ref> <remote sha>, one line per ref.
#
# This is the hook's only duty. Until #966 it also ran `./ucx test check` on every
# branch push and wrote <git-common-dir>/gate-pass/<sha>; owner ruling 2026-09-15
# (decision ledger DL-020) removed that. The record is now written by
# `./ucx test check` itself, when a full run passes on a clean committed tree, and the
# gate is run at the PR head immediately before merge (builder-manager Phase 2 step 3).
while read -r local_ref local_sha remote_ref remote_sha; do
  if [ "$remote_ref" = "refs/heads/main" ]; then
    echo "pre-push: refusing to push to main ($local_sha)." >&2
    echo "pre-push: main changes only via a PR: push the task/ branch and open one." >&2
    echo "pre-push: see AGENTS.md §2, 'The Primary Workspace Is Pinned', and Swarm Guide §8.2." >&2
    exit 1
  fi
done
exit 0  # any other ref: no gate run, no gate-pass record (#966)
"""
"""Content of the pre-push hook installed by `ucx setup`.

Complements PRE_COMMIT_HOOK: that one refuses commits *in the primary workspace*; this
one refuses *landing anything on main locally*, which a fast-forward merge in the primary
workspace followed by `git push` did on 2026-09-03 (`5a34169`) without ever running
pre-commit. `git push --no-verify` bypasses it (P8).

It runs no tests and writes no gate-pass record (#966, owner ruling 2026-09-15). The
push-time gate re-ran ~3 minutes of suite on every push of every branch, and unrelated
environment noise refused pushes of finished work often enough that a builder bypassed
it with `--no-verify` (#919) — which also drops the `main` refusal above. The record now
comes from `./ucx test check` (`record_gate_pass` in `cli/quality_gate.py`).
"""


def running_from_source_checkout() -> bool:
    """Whether this `ucx` is running out of a checkout of its own repository.

    Decides which commands exist. An installed build is for using the agent; the
    development commands assume a checkout and misbehave without one, in three ways
    measured on a wheel installed into a clean environment:

    * `ucx setup` writes this project's `pre-commit`/`pre-push` hooks into whatever
      repository the current directory is the root of. That pre-commit refuses any
      commit not authored by `uclone-x-builder-bot[bot]`, so a user who runs it in
      their own project can no longer commit to it, for a reason naming a bot they
      have never heard of.
    * `ucx test check` raises a traceback out of `cli/main.py`, which reads as a bug
      in the tool rather than as "this needs the repository".
    * `ucx dev task list` exits **0** and prints an internal project-board URL — a
      false success that also discloses internal detail.

    Detected by looking for the `pyproject.toml` that declares this project above the
    package, not by looking for git: a source tree downloaded without git metadata is
    still a source tree, and `site-packages` never has one. An editable install points
    `__file__` back into the checkout, so it is correctly treated as one.
    """
    root = Path(__file__).resolve().parents[3]
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    # Parsed, not substring-matched: `name = 'uclone-x'` is equally valid TOML, and a
    # reformat that changed the quoting would have silently deleted `setup`, `test` and
    # `dev` from every checkout -- invisibly, because no test reads the real file.
    # `UnicodeDecodeError` is caught alongside `OSError` and `TOMLDecodeError`: a
    # pyproject that is not valid UTF-8 crashed `ucx` at import, before any command ran.
    try:
        parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    project = parsed.get("project")
    if not isinstance(project, dict):
        return False
    name = cast("dict[str, object]", project).get("name")
    return isinstance(name, str) and name == "uclone-x"


def _resolve_git_hooks_dir() -> Path | None:
    """Return the hooks directory of the checkout in the current directory, or None.

    `git rev-parse --git-path hooks` answers correctly from both the primary workspace
    and a linked worktree (hooks live in the common dir, which a worktree's `.git` file
    only points at); the old literal `.git/hooks` made `setup` a silent no-op from a
    worktree. Two guards keep the answer about *this* directory:

    * `.git` must exist here. `setup` is a repo-root command, and without this check a
      run from an unrelated directory would install into whatever repository encloses it.
    * `GIT_*` is stripped from the child environment. Inside a git hook `GIT_DIR` and
      `GIT_INDEX_FILE` are exported, and with them `rev-parse` answers for the repository
      running the hook regardless of the working directory — which is how the quality gate,
      run by the pre-commit hook, once had `setup` write the hook into the real repository
      instead of the test's temporary one.

    `--git-path hooks` also honours `core.hooksPath`, which is the directory git will
    actually run. It is created if it does not exist, parents included.

    **When git names a hooks directory that cannot be created, this aborts rather than
    falling back.** Git reads hooks from that directory and nowhere else, so writing into
    `.git/hooks` instead would print a success line for a hook that never fires — the
    same silent-permission failure the hooks exist to prevent, arriving through the
    installer. A `setup` that cannot install where git looks has not installed anything,
    and says so with a non-zero exit.

    The literal fallback is only for the case where git could not answer at all — it is
    unavailable, or `.git` is not a usable repository — which is how the unit test
    exercises it. It is never a second choice for a hooks directory git *did* name.
    """
    if not Path(".git").exists():
        return None
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError):
        completed = None
    if completed is not None:
        resolved = Path(completed.stdout.strip())
        if resolved.parent.is_file():
            console.print(
                f"✖ git reads hooks from {resolved}, whose parent is a file. "
                "Refusing to install into .git/hooks, where git would never read them."
            )
            raise typer.Exit(1)
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            console.print(
                f"✖ git reads hooks from {resolved}, which cannot be created ({exc}). "
                "Refusing to install into .git/hooks, where git would never read them."
            )
            raise typer.Exit(1) from exc
        return resolved
    fallback = Path(".git/hooks")
    return fallback if fallback.is_dir() else None


def read_bot_commit_identity(repo_root: Path | None = None) -> tuple[str, str] | None:
    """Return (name, email) from `swarm/config.yaml`'s `bot.commit_identity`, or None.

    Parsed with the same nesting rules the pre-commit hook's `awk` uses, so the installer
    and the guard agree about what the manifest says. Returns None rather than raising:
    a repository without the block is not an error for `setup`, it is simply a repository
    with no declared identity to normalise.
    """
    root = repo_root if repo_root is not None else Path(".")
    manifest = root / "swarm" / "config.yaml"
    try:
        raw = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    in_bot = in_identity = False
    found: dict[str, str] = {}
    for line in raw.splitlines():
        if line.startswith("bot:"):
            in_bot, in_identity = True, False
            continue
        if line and not line[0].isspace():
            in_bot = in_identity = False
            continue
        if in_bot and line.strip() == "commit_identity:":
            in_identity = True
            continue
        if in_identity:
            stripped = line.strip()
            for key in ("name", "email"):
                if stripped.startswith(f"{key}:"):
                    found[key] = stripped[len(key) + 1 :].strip().strip('"').strip("'")
    if "name" in found and "email" in found and found["name"] and found["email"]:
        return found["name"], found["email"]
    return None


def _normalise_commit_identity() -> None:
    """Point the repository's `user.name`/`user.email` at `bot.commit_identity`.

    Run in the same pass as the hook install, deliberately. The pre-commit hook refuses a
    commit whose author is not the bot, and a hook installed while the config still names
    someone else would refuse every commit in every worktree — so shipping the guard and
    fixing the identity have to happen together, or the repository spends time in a state
    where the control is installed and nothing can be committed.

    **This writes the shared `.git/config`**, which the primary workspace and every linked
    worktree read: `extensions.worktreeConfig` is unset, so there is no per-worktree config
    to write (measured — a `git config` from inside a worktree lands in `.git/config`). It
    says so out loud rather than doing it quietly, because a command that changes shared
    state while looking local is the shape this repository keeps paying for.
    """
    identity = read_bot_commit_identity()
    if identity is None:
        console.print(
            "  ! swarm/config.yaml declares no bot.commit_identity, so the commit identity "
            "was left alone. The pre-commit hook will refuse commits until it does."
        )
        return
    name, email = identity

    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    current: dict[str, str] = {}
    for key in ("user.name", "user.email"):
        try:
            probe = subprocess.run(
                ["git", "config", "--get", key],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
        except OSError:
            return
        current[key] = probe.stdout.strip()

    if current["user.name"] == name and current["user.email"] == email:
        console.print(f"✔ Commit identity already {name} <{email}>")
        return

    for key, value in (("user.name", name), ("user.email", email)):
        try:
            written = subprocess.run(
                ["git", "config", key, value],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
        except OSError:
            return
        if written.returncode != 0:
            console.print(f"  ! could not set {key}: {written.stderr.strip()}")
            return

    console.print(f"✔ Commit identity set to {name} <{email}> in the shared .git/config")
    console.print(
        f"  ! this replaced {current['user.name']} <{current['user.email']}> for EVERY "
        "worktree and the primary workspace — there is no per-worktree git config here"
    )


def find_available_port(preferred_port: int, max_attempts: int = 50) -> int:
    """Find the preferred port or the next available port if preferred is in use."""
    for p in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred_port


app = typer.Typer(
    name="ucx",
    help="UClone-X: High-Performance Event-Driven Agent Framework CLI",
    add_completion=False,
    no_args_is_help=True,
)
agent_app = typer.Typer(
    name="agent",
    help="Manage and interact with runtime autonomous agents and swarms",
    no_args_is_help=True,
)
test_app = typer.Typer(
    name="test",
    help="Automated quality gates and test suites",
    no_args_is_help=True,
)
app.add_typer(agent_app, name="agent")
app.add_typer(a2a_app, name="a2a")
app.add_typer(acp_app, name="acp")
app.add_typer(eval_app, name="eval")
app.add_typer(llm_app, name="llm")
app.add_typer(loop_app, name="loop")
agent_app.add_typer(loop_app, name="loop")
app.add_typer(media_app, name="media")
app.add_typer(ontology_app, name="ontology")
app.add_typer(report_app, name="report")
app.add_typer(room_app, name="room")
app.add_typer(skill_app, name="skill")
app.add_typer(ui_app, name="ui")

# `test`, `dev` and `setup` are registered only from a source checkout; see
# `running_from_source_checkout` and `register_developer_commands` at the end of
# this module. `eval` stays registered either way because it already refuses with an
# actionable message when its separately-distributed backend is absent, and a user
# may install one.

console = Console()


@app.command()
def version() -> None:
    """Print UClone-X version."""
    from uclone_x import __version__

    console.print(f"[bold cyan]UClone-X[/bold cyan] version [green]{__version__}[/green]")


@app.command("acp-server")
def acp_server(
    agent_id: str | None = typer.Option(None, "--agent-id", help=ACP_AGENT_ID_HELP),
    persona: str = typer.Option(DEFAULT_PERSONA_NAME, "--persona", help="Clone to answer as"),
) -> None:
    """Start standalone ACP stdio server for editor integration (Issue #649)."""
    start_acp_server(agent_id=agent_id, persona=persona)


def setup() -> None:
    """Bootstrap the local development environment (venv, dependencies, UI, git hooks)."""
    console.print("[bold green]Bootstrapping UClone-X environment...[/bold green]")

    # 1. Install Git pre-commit hook if inside git repo
    git_hooks_dir = _resolve_git_hooks_dir()
    if git_hooks_dir is not None:
        for hook_name, hook_content in (
            ("pre-commit", PRE_COMMIT_HOOK),
            ("pre-push", PRE_PUSH_HOOK),
        ):
            hook_path = git_hooks_dir / hook_name
            # `setup` overwrites whatever is there. On 2026-09-03 the live pre-commit hook
            # was narrowed by an in-place edit that left no diff and no history (#285), so
            # "installed" and "what git has been running" are not the same thing. Report
            # the difference rather than replacing it silently: whether the gate should
            # skip docs-only commits is #285's open question, but a hook that changes
            # under the swarm without anyone being told is the part that is not.
            try:
                previous = hook_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                previous = None
            hook_path.write_text(hook_content, encoding="utf-8")
            hook_path.chmod(0o755)
            console.print(f"✔ Installed Git {hook_name} hook ({hook_path})")
            if previous is not None and previous != hook_content:
                # Byte count, not character count: the identifying fact about the hook
                # edited in place on 2026-09-03 is its size on disk (1242 bytes, #285),
                # and that hook contains multi-byte characters.
                console.print(
                    f"  ! replaced a {hook_name} hook "
                    f"({len(previous.encode('utf-8'))} bytes) that differed from the "
                    "tracked source — see #285"
                )

        _normalise_commit_identity()

    console.print("✔ Python 3.11+ environment initialized")
    console.print("✔ Run [bold cyan]./ucx test check[/bold cyan] to verify.")


@app.command()
@agent_app.command("run")
def run(
    agent_name: str = typer.Argument("default", help="Name of the agent or swarm to run"),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help=(
            "LLM Provider: ollama | vllm | openai | anthropic | gemini | mock. "
            "Omit to resolve from LLM_PROVIDER, a credential variable, or a configured "
            "Ollama or vLLM endpoint."
        ),
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model name (default depends on provider)"
    ),
    system_prompt: str | None = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt. Omit to use the composed default (agent/prompts.py), "
        "adapted for --model's family.",
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", help="Execute single prompt non-interactively"
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Session to run in. Persisted under the Core session store, so a session "
        "resumed by ID keeps the conversation it had (#183).",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Reset the session to its system prompt before running, and persist that.",
    ),
    compact: bool = typer.Option(
        False,
        "--compact",
        help="Compact the session context before running (P5).",
    ),
    isolation: str = typer.Option(
        "workspace",
        "--isolation",
        "-i",
        help="Isolation level: workspace, container, wasm, or none",
    ),
    cwd: Annotated[
        Path | None,
        typer.Option(
            "--cwd",
            "-w",
            help="Root workspace directory for file and tool operations (default: current directory)",
        ),
    ] = None,
) -> None:
    """Launch the interactive terminal Chat REPL or execute a single prompt with BaseAgent."""
    from uclone_x.cli.commands.run import run_agent_repl

    effective_cwd = cwd.resolve() if cwd is not None else Path.cwd().resolve()
    run_agent_repl(
        agent_name=agent_name,
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        prompt=prompt,
        session_id=session_id,
        reset=reset,
        compact=compact,
        isolation=isolation,
        workspace_dir=effective_cwd,
    )


@app.command("install")
def install_models(
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Accept every prompt in advance. Required to proceed when there is no terminal.",
    ),
    llm: bool = typer.Option(True, "--llm/--no-llm", help="Install the local LLM (Ollama)"),
    image: bool = typer.Option(
        True, "--image/--no-image", help="Install the local image engine and its checkpoint"
    ),
) -> None:
    """Install the local models without a terminal: Ollama + its model, and the image engine.

    The non-interactive twin of what `ucx start` does on the way up, and the command an
    automated install test can actually drive. `ucx start` asks; this one takes `--yes`, so
    `ssh host 'ucx install --yes </dev/null'` completes or fails, and never blocks on a
    prompt nobody can answer.

    The last lines are fixed and greppable, one per requested part:

        setup llm: ready model=<name> endpoint=<url>
        setup llm: unavailable reason=<short>
        setup image: ready engine=<engine> checkpoint=<path>
        setup image: unavailable reason=<short>

    Exit status is 0 only when every requested part is ready — measured by the probers
    after the installs ran, not inferred from an installer's exit code.
    """
    from uclone_x.cli.commands.bootstrap import run_local_setup

    result = run_local_setup(llm=llm, image=image, assume_yes=assume_yes)
    for line in result.summary_lines():
        # `print`, not `console.print`: rich wraps at the terminal width and would fold a
        # long checkpoint path onto a second line, which is precisely what a grepping
        # install script cannot read. These lines are a machine interface.
        print(line)
    raise typer.Exit(0 if result.ok else 1)


@app.command("start")
def start(
    port: int = typer.Option(5180, "--port", "-p", help="Port to bind the developer GUI backend"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind"),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Automatically launch web browser on start (default: True)"
    ),
    setup_local: bool = typer.Option(
        True,
        "--setup-local/--skip-setup",
        help="Perform pre-flight system inspection and ensure local Ollama AI model is ready",
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
    """Zero-barrier one-click starter for beginners: inspects hardware, prepares local AI, and opens UI."""
    from uclone_x.cli.commands.bootstrap import run_local_setup
    from uclone_x.ui.server import start_ui_server

    effective_cwd = cwd.resolve() if cwd is not None else Path.cwd().resolve()
    console.print("[bold cyan]🚀 Initializing UClone-X Beginner-First Workspace...[/bold cyan]")
    console.print(f"[dim]📁 Working Directory: {effective_cwd}[/dim]")

    if setup_local:
        # The same function `ucx install` runs, with the prompts left on. Inline copies of
        # this sequence are how `start` and the install path drift apart; there is one.
        result = run_local_setup(llm=True, image=True, interactive=True, assume_yes=False)
        if result.llm is not None and result.llm.ready:
            console.print(
                f"[bold green]✔ Local Private AI Model Ready:[/bold green] "
                f"[cyan]{result.llm.model}[/cyan]"
            )
            os.environ.setdefault("LLM_PROVIDER", "ollama")
            os.environ.setdefault("OLLAMA_MODEL", result.llm.model)
            os.environ.setdefault("OLLAMA_BASE_URL", result.llm.endpoint)

    bound_port = find_available_port(port)
    if bound_port != port:
        console.print(
            f"[yellow]⚠️ Port {port} is in use. Automatically falling back to port {bound_port}.[/yellow]"
        )

    console.print(
        f"[bold green]✨ Starting UClone-X Web Dashboard at[/bold green] [cyan]http://{host}:{bound_port}[/cyan]"
    )
    start_ui_server(
        port=bound_port,
        dev=False,
        host=host,
        auto_open_browser=open_browser,
        workspace_dir=effective_cwd,
    )


ui = ui_app


@test_app.command("check")
def test_check(
    skip_tests: bool = typer.Option(False, "--skip-tests", help="Skip running the pytest suite"),
    check_frontend: bool = typer.Option(
        False, "--check-frontend", "-fe", help="Run frontend TypeScript build check"
    ),
    all_tests: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Deprecated: E2E is now included by default. Retained as a no-op synonym.",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        "-f",
        help="Skip the E2E Playwright suite (for the tight edit loop)",
    ),
    fail_fast: bool = typer.Option(
        True,
        "--fail-fast/--no-fail-fast",
        help="Stop on first failure (default: True). Use --no-fail-fast to run all checks.",
    ),
    serial: bool = typer.Option(
        False,
        "--serial",
        help="Run pytest in one process instead of parallel workers: slower, same selection "
        "and coverage. For debugging an ordering or isolation failure (#967).",
    ),
) -> None:
    """Run automated quality gates: Ruff lint/format, Pyright strict typing, and Pytest.

    The pytest selection includes the E2E Playwright suite. It used to be opt-in behind
    `--all`, which meant a green gate said nothing about the rendered UI; `--fast` is the
    opt-out for iterating, and `--all` is kept as a no-op synonym for the default.
    """
    # Reported here rather than inside `run_quality_gate`: this is presentation, not a
    # verification step, and putting a git call inside the gate changed the subprocess
    # accounting its own tests assert on.
    for line in describe_installed_hook_drift():
        console.print(line)
    # `--all` is a no-op synonym now that `gate` includes E2E; `--fast` drops it.
    test_scope = "fast" if fast else "gate"
    # The tree is inspected before the run as well as after: a record claims the gate ran on
    # exactly this commit, and a tree that was dirty while the suite ran and clean afterwards
    # (edits reverted, stashed) would otherwise earn one (#966).
    before = (
        snapshot_committed_tree()
        if gate_run_can_record(test_scope, skip_tests=skip_tests)
        else None
    )
    exit_code = run_quality_gate(
        skip_tests=skip_tests,
        check_frontend=check_frontend,
        fail_fast=fail_fast,
        test_scope=test_scope,
        serial=serial,
    )
    console.print(
        record_gate_pass(
            exit_code=exit_code, test_scope=test_scope, skip_tests=skip_tests, before=before
        ),
        markup=False,
        highlight=False,
        soft_wrap=True,
    )
    if exit_code != 0:
        sys.exit(exit_code)


@test_app.command("changed")
def test_changed(
    base: str = typer.Option(
        "origin/main", "--base", help="Diff against the merge base with this ref."
    ),
    fail_fast: bool = typer.Option(
        True,
        "--fail-fast/--no-fail-fast",
        help="Stop on first failure (default: True). Use --no-fail-fast to run all checks.",
    ),
) -> None:
    """Run the gate's checks over what the diff reaches. Records no gate pass.

    Ruff on the changed Python files, Pyright on them and their direct importers, and the
    tests that import a changed module or name a changed file. A change to a conftest, the
    shared test support or the dependency set runs `check --fast` instead. The full
    `./ucx test check` runs once per PR, at the head that merges.
    """
    from uclone_x.cli.changed_scope import run_changed_gate

    exit_code = run_changed_gate(base=base, fail_fast=fail_fast)
    if exit_code != 0:
        sys.exit(exit_code)


@test_app.command("unit")
def test_unit(
    check_frontend: bool = typer.Option(
        False, "--check-frontend", "-fe", help="Run frontend TypeScript build check"
    ),
    fail_fast: bool = typer.Option(
        True,
        "--fail-fast/--no-fail-fast",
        help="Stop on first failure (default: True). Use --no-fail-fast to run all checks.",
    ),
    serial: bool = typer.Option(
        False,
        "--serial",
        help="Run pytest in one process instead of parallel workers: slower, same selection "
        "and coverage. For debugging an ordering or isolation failure (#967).",
    ),
) -> None:
    """Run unit test suite with branch coverage >= 70% (excluding E2E tests)."""
    exit_code = run_quality_gate(
        skip_tests=False,
        check_frontend=check_frontend,
        fail_fast=fail_fast,
        test_scope="unit",
        serial=serial,
    )
    if exit_code != 0:
        sys.exit(exit_code)


@test_app.command("fitness")
def test_fitness(
    fail_fast: bool = typer.Option(
        True,
        "--fail-fast/--no-fail-fast",
        help="Stop on first failure (default: True). Use --no-fail-fast to run all checks.",
    ),
    serial: bool = typer.Option(
        False,
        "--serial",
        help="Run pytest in one process instead of parallel workers: slower, same selection "
        "and coverage. For debugging an ordering or isolation failure (#967).",
    ),
) -> None:
    """Run the fitness functions over the repository's own declarations (no coverage)."""
    exit_code = run_quality_gate(
        skip_tests=False,
        fail_fast=fail_fast,
        test_scope="fitness",
        serial=serial,
    )
    if exit_code != 0:
        sys.exit(exit_code)


@test_app.command("recorded")
def test_recorded(
    check_frontend: bool = typer.Option(
        False, "--check-frontend", "-fe", help="Run frontend TypeScript build check"
    ),
    fail_fast: bool = typer.Option(
        True,
        "--fail-fast/--no-fail-fast",
        help="Stop on first failure (default: True). Use --no-fail-fast to run all checks.",
    ),
) -> None:
    """Run the Tier 2 recorded-playback suite: replays committed cassettes, no network."""
    exit_code = run_quality_gate(
        skip_tests=False,
        check_frontend=check_frontend,
        fail_fast=fail_fast,
        test_scope="recorded",
    )
    if exit_code != 0:
        sys.exit(exit_code)


@test_app.command("live")
def test_live(
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help="LLM provider for the live run (sets LLM_PROVIDER). 'ollama' is free and local.",
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help="Permit cassette writes (sets RECORD=1). Without this a live run never "
        "mutates committed cassettes.",
    ),
    check_frontend: bool = typer.Option(
        False, "--check-frontend", "-fe", help="Run frontend TypeScript build check"
    ),
    fail_fast: bool = typer.Option(
        True,
        "--fail-fast/--no-fail-fast",
        help="Stop on first failure (default: True). Use --no-fail-fast to run all checks.",
    ),
) -> None:
    """Run the Tier 3 live suite against a real LLM endpoint (costs tokens unless --provider ollama)."""
    # Recording is gated on RECORD alone, never on being live. uclone2 keyed `record_mode`
    # off its LIVE flag, so every live run silently rewrote every cassette it replayed.
    os.environ["LLM_PROVIDER"] = provider
    if record:
        os.environ["RECORD"] = "1"
    exit_code = run_quality_gate(
        skip_tests=False,
        check_frontend=check_frontend,
        fail_fast=fail_fast,
        test_scope="live",
    )
    if exit_code != 0:
        sys.exit(exit_code)


@test_app.command("pre-release")
def test_pre_release(
    fail_fast: bool = typer.Option(
        True,
        "--fail-fast/--no-fail-fast",
        help="Stop on first failure (default: True). Use --no-fail-fast to run all checks.",
    ),
) -> None:
    """Run the release-qualification suite: clean-venv install matrix (slow, uses network)."""
    exit_code = run_quality_gate(
        skip_tests=False,
        fail_fast=fail_fast,
        test_scope="pre-release",
    )
    if exit_code != 0:
        sys.exit(exit_code)


@test_app.command("e2e")
def test_e2e(
    check_frontend: bool = typer.Option(
        False, "--check-frontend", "-fe", help="Run frontend TypeScript build check"
    ),
    fail_fast: bool = typer.Option(
        True,
        "--fail-fast/--no-fail-fast",
        help="Stop on first failure (default: True). Use --no-fail-fast to run all checks.",
    ),
    serial: bool = typer.Option(
        False,
        "--serial",
        help="Run pytest in one process instead of parallel workers: slower, same selection "
        "and coverage. For debugging an ordering or isolation failure (#967).",
    ),
) -> None:
    """Run E2E Playwright integration test suite."""
    exit_code = run_quality_gate(
        skip_tests=False,
        check_frontend=check_frontend,
        fail_fast=fail_fast,
        test_scope="e2e",
        serial=serial,
    )
    if exit_code != 0:
        sys.exit(exit_code)


@app.command("status")
def status_command(
    url: str = typer.Option(
        "http://127.0.0.1:8000", "--url", "-u", help="Target UClone-X UI server URL"
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output status in JSON format"),
) -> None:
    """Inspect runtime health and absorbed failure metrics (P6/P8)."""
    import json

    import httpx
    from rich.table import Table

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{url.rstrip('/')}/api/diagnostics")
            if resp.status_code != 200:
                resp = client.get(f"{url.rstrip('/')}/api/health")
            data = resp.json()
    except Exception as exc:
        console.print(
            f"[bold red]✖ Failed to connect to UClone-X server at {url}: {exc}[/bold red]"
        )
        sys.exit(1)

    if json_output:
        console.print_json(json.dumps(data))
        return

    console.print(
        f"[bold cyan]UClone-X Runtime Status[/bold cyan] (v{data.get('version', 'unknown')})"
    )

    absorbed = data.get("absorbed_failures", {})
    spans = absorbed.get("dropped_spans", {})
    bus_drops = absorbed.get("event_bus_drops", {})
    agent_errors = absorbed.get("agent_processing_errors", {})

    table = Table(title="Absorbed Failures & Discard Accounting")
    table.add_column("Subsystem", style="cyan")
    table.add_column("Drop / Error Count", justify="right")
    table.add_column("Details", style="magenta")

    table.add_row("Telemetry Tracer", str(spans.get("count", 0)), str(spans.get("reasons", {})))
    table.add_row("Event Bus", str(bus_drops.get("count", 0)), str(bus_drops.get("reasons", {})))
    table.add_row(
        "Agent Core", str(agent_errors.get("count", 0)), str(agent_errors.get("agents", {}))
    )

    console.print(table)


def register_developer_commands() -> None:
    """Expose the checkout-only commands, and tell an installed build where they went.

    Registered here rather than by decorator so that an installed build does not carry
    commands that cannot work in it. The alternative -- ship them and let them fail --
    was measured and rejected: see `running_from_source_checkout` for what each of the
    three actually does outside a checkout.

    An installed build still says where they are. A command that silently does not
    exist leaves a reader who followed older documentation with nothing to go on, so
    the help text names the checkout as the place these live.
    """
    if running_from_source_checkout():
        app.command()(setup)
        app.add_typer(test_app, name="test")
        app.add_typer(dev_app, name="dev")
        return

    app.info.epilog = (
        "Development commands (setup, test, dev) are available when ucx runs from a "
        "source checkout: git clone https://github.com/UClone-AI/uclone-x"
    )


register_developer_commands()


def main() -> None:
    """Run the CLI, recording an unhandled failure if the user allowed it.

    The hook is installed here rather than at import: importing `uclone_x.cli`
    must not change the process's exception handling for a program that merely
    embeds the library. `install_excepthook` chains to the previous hook, so the
    traceback a developer expects still prints, and `record_failure` is a no-op
    until consent exists.
    """
    install_excepthook()
    app()


if __name__ == "__main__":
    main()
