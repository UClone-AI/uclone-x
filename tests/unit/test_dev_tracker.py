"""Unit tests for the Builder task commands (``./ucx dev task``).

Every subcommand is board-backed: it reaches the project board through `gh`. The
Markdown registers these commands used to read and write — ``docs/tasks/`` and the
``docs/issues/`` findings register — are retired (#1037), so what is covered here is
what survived that removal: the `gh` bridge and its failure paths, the worktree a claim
creates, and the `frontend/node_modules` link that worktree needs (#915, #930).
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from uclone_x.cli.commands import dev

runner = CliRunner()


def _init_git_repo(path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True, env=env
    )
    subprocess.run(
        ["git", "config", "user.name", "Test Builder"],
        cwd=path,
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        ["git", "config", "user.email", "builder@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
        env=env,
    )
    readme = path / "README.md"
    readme.write_text("# Test Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True, env=env)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=path,
        check=True,
        capture_output=True,
        env=env,
    )


@pytest.fixture
def scratch_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real git repository the commands treat as the checkout.

    `GIT_*` is stripped so the scratch repository is not silently read as this one.
    """
    for key in list(os.environ.keys()):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key, raising=False)
    _init_git_repo(tmp_path)
    monkeypatch.setattr(dev, "_repo_root", lambda: tmp_path)
    return tmp_path


def _let_git_run_and_fake_gh(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub `gh` only, so a claim creates a worktree that really exists on disk.

    Returns the list the faked `gh` invocations accumulate in.
    """
    real_run = subprocess.run
    gh_calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "gh":
            gh_calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return cast("subprocess.CompletedProcess[str]", real_run(cmd, **kwargs))

    monkeypatch.setattr(subprocess, "run", run)
    return gh_calls


def _branch_exists(repo: Path, branch: str) -> bool:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
            cwd=repo,
            capture_output=True,
            env=env,
        ).returncode
        == 0
    )


# --- the retired registers are gone from the surface (#1037) ---------------------------


def test_dev_has_no_issue_command() -> None:
    """`ucx dev issue` does not exist: the register it browsed is retired.

    Asserted as a missing command rather than as an empty listing — a group that still
    exists and renders nothing reads as "no findings", which is a different claim.

    Killed by: src/uclone_x/cli/commands/dev.py :: dev_app.add_typer(task_app, name="task")
    Becomes: dev_app.add_typer(task_app, name="issue")
    """
    result = runner.invoke(dev.dev_app, ["issue", "list"])
    assert result.exit_code != 0
    assert "Design Review Findings Register" not in result.output

    help_result = runner.invoke(dev.dev_app, ["--help"])
    assert help_result.exit_code == 0
    assert "issue" not in help_result.output


def test_task_list_reaches_the_board_with_no_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """`task list` queries GitHub by default; `--github` no longer selects anything.

    Both spellings are asserted to issue the same `gh` query, because `-g` is kept only
    so existing scripts and guides keep working.

    Killed by: src/uclone_x/cli/commands/dev.py :: _render_github_issues(show_all=show_all)
    Becomes: pass
    """
    queries: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "gh":
            queries.append(cmd)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps([{"number": 20, "title": "Board task", "state": "OPEN"}]),
                stderr="",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    bare = runner.invoke(dev.dev_app, ["task", "list"])
    flagged = runner.invoke(dev.dev_app, ["task", "list", "-g"])

    assert bare.exit_code == 0, bare.output
    assert flagged.exit_code == 0, flagged.output
    assert "Live GitHub Tasks / Issues" in bare.output
    assert "Board task" in bare.output
    assert len(queries) == 2
    assert queries[0] == queries[1]


# --- the gh bridge ---------------------------------------------------------------------


def test_task_list_github_success(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_issues = [
        {
            "number": 20,
            "title": "Retire the file-based task register",
            "state": "OPEN",
            "assignees": [{"login": "builder:dev-junior-1"}],
            "labels": [{"name": "module:cli"}, {"name": "priority:medium"}],
        },
        {
            "number": 21,
            "title": "Closed Issue Example",
            "state": "CLOSED",
            "assignees": ["plain-user"],
            "labels": ["some-label"],
        },
    ]

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["gh", "issue"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps(mock_issues),
                stderr="",
            )
        raise ValueError(f"Unexpected cmd: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(dev.dev_app, ["task", "list", "--github"])
    assert result.exit_code == 0
    assert "Live GitHub Tasks / Issues" in result.output
    assert "#20" in result.output
    assert "Retire the" in result.output
    assert "module:cli" in result.output


def test_task_list_github_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=json.dumps([]), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(dev.dev_app, ["task", "list"])
    assert result.exit_code == 0
    assert "No issues found on GitHub" in result.output


def test_task_list_github_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd, stderr="Authentication error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(dev.dev_app, ["task", "list"])
    assert result.exit_code == 1
    assert "failed to fetch issues" in result.output


def test_task_list_github_json_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="not-json", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(dev.dev_app, ["task", "list"])
    assert result.exit_code == 1
    assert "Failed to parse GitHub issues JSON" in result.output


def test_task_create_github_success(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gh", "issue", "create"]:
            created.append(cmd)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="https://github.com/UClone-AI/uclone-x/issues/99\n",
                stderr="",
            )
        raise ValueError(f"Unexpected cmd: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(
        dev.dev_app,
        [
            "task",
            "create",
            "New GH Task",
            "--module",
            "cli",
            "--type",
            "task",
            "--resolves",
            "2026-09-02-001",
        ],
    )
    assert result.exit_code == 0
    assert "Created GitHub Issue:" in result.output
    assert "issues/99" in result.output
    # The finding this task addresses reaches the board in the issue body, which is now
    # the only place it is recorded.
    assert len(created) == 1
    assert any("2026-09-02-001" in arg for arg in created[0])


def test_task_create_github_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd, stderr="Network error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(dev.dev_app, ["task", "create", "New GH Task"])
    assert result.exit_code == 1
    assert "Failed to create GitHub issue via gh" in result.output


def test_task_claim_github_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_git_repo(tmp_path)
    monkeypatch.setattr(dev, "_repo_root", lambda: tmp_path)

    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(
        dev.dev_app,
        [
            "task",
            "claim",
            "20",
            "--assignee",
            "builder:dev-junior-1",
            "--create-worktree",
        ],
    )
    assert result.exit_code == 0
    assert "Claimed GitHub Issue:" in result.output
    assert "#20" in result.output
    assert "builder:dev-junior-1" in result.output
    assert len(calls) == 3


def test_task_resolve_github_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev, "run_quality_gate", lambda: 0)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(
        dev.dev_app,
        ["task", "resolve", "20", "--comment", "Resolved by builder"],
    )
    assert result.exit_code == 0
    assert "Resolved/closed GitHub Issue: #20" in result.output
    assert len(calls) == 2
    # The comment is not swallowed: it is the body `gh issue close` carries.
    assert calls[-1][:4] == ["gh", "issue", "close", "20"]
    assert "Resolved by builder" in calls[-1]


def test_task_resolve_github_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev, "run_quality_gate", lambda: 0)

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd, stderr="Not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(dev.dev_app, ["task", "resolve", "20"])
    assert result.exit_code == 1
    assert "Failed to resolve GitHub issue via gh" in result.output


def test_resolve_blocks_when_the_quality_gate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing gate stops the close, so the issue is not marked done over a red tree."""
    monkeypatch.setattr(dev, "run_quality_gate", lambda: 1)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(dev.dev_app, ["task", "resolve", "20"])

    assert result.exit_code == 1
    assert "Quality gate failed" in result.output
    assert not any(cmd[:1] == ["gh"] for cmd in calls)


def test_resolve_no_verify_bypasses_the_quality_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> int:
        raise AssertionError("--no-verify must not run the gate")

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dev, "run_quality_gate", boom)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(dev.dev_app, ["task", "resolve", "20", "--no-verify"])

    assert result.exit_code == 0
    assert "Resolved/closed GitHub Issue: #20" in result.output


def test_task_merge_github_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev, "run_quality_gate", lambda: 0)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(dev.dev_app, ["task", "merge", "20"])
    assert result.exit_code == 0
    assert "Merged PR on GitHub: 20" in result.output
    assert len(calls) == 2
    assert calls[-1] == ["gh", "pr", "merge", "20", "--rebase", "--delete-branch"]


def test_task_merge_github_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev, "run_quality_gate", lambda: 0)

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd, stderr="PR merge failed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(dev.dev_app, ["task", "merge", "20", "--github"])
    assert result.exit_code == 1
    assert "Failed to merge PR via gh" in result.output


def test_merge_blocks_on_a_failed_quality_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate runs before `gh pr merge`, so a red tree is never merged."""
    monkeypatch.setattr(dev, "run_quality_gate", lambda: 1)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(dev.dev_app, ["task", "merge", "20"])

    assert result.exit_code == 1
    assert "Quality gate failed" in result.output
    assert not any(cmd[:1] == ["gh"] for cmd in calls)


def test_merge_uses_the_branch_option_as_the_merge_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev, "run_quality_gate", lambda: 0)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(
        dev.dev_app, ["task", "merge", "20", "--branch", "task/20-builder-one", "--no-verify"]
    )

    assert result.exit_code == 0
    assert calls[-1][:4] == ["gh", "pr", "merge", "task/20-builder-one"]


# --- the worktree a claim creates -------------------------------------------------------


def test_claim_creates_a_worktree_on_a_task_branch(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default worktree is named for the builder, on a branch named for the issue."""
    gh_calls = _let_git_run_and_fake_gh(monkeypatch)

    result = runner.invoke(
        dev.dev_app,
        ["task", "claim", "20", "--assignee", "builder:one", "--create-worktree"],
    )

    assert result.exit_code == 0, result.output
    assert "Created worktree:" in result.output
    assert (scratch_repo / ".worktrees" / "builder-one").is_dir()
    assert _branch_exists(scratch_repo, "task/20-builder-one")
    assert any(cmd[:3] == ["gh", "issue", "comment"] for cmd in gh_calls)


def test_claim_with_custom_worktree_option(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _let_git_run_and_fake_gh(monkeypatch)

    result = runner.invoke(
        dev.dev_app,
        [
            "task",
            "claim",
            "20",
            "--assignee",
            "builder:one",
            "--worktree",
            ".worktrees/custom-wt",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (scratch_repo / ".worktrees" / "custom-wt").is_dir()


def test_claim_rejects_existing_worktree(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gh_calls = _let_git_run_and_fake_gh(monkeypatch)
    (scratch_repo / ".worktrees" / "builder-one").mkdir(parents=True)

    result = runner.invoke(
        dev.dev_app,
        ["task", "claim", "20", "--assignee", "builder:one", "--create-worktree"],
    )

    assert result.exit_code == 1
    assert "Target worktree already exists" in result.output
    # The claim aborts before announcing itself on the issue.
    assert gh_calls == []


def test_claim_fails_when_git_worktree_fails(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(
        ["git", "branch", "task/20-builder-one"],
        cwd=scratch_repo,
        check=True,
        capture_output=True,
        env=env,
    )
    gh_calls = _let_git_run_and_fake_gh(monkeypatch)

    result = runner.invoke(
        dev.dev_app,
        ["task", "claim", "20", "--assignee", "builder:one", "--create-worktree"],
    )

    assert result.exit_code == 1
    assert "Failed to create worktree" in result.output
    assert gh_calls == []


def test_claim_rejects_empty_assignee() -> None:
    result = runner.invoke(dev.dev_app, ["task", "claim", "20", "--assignee", "   "])

    assert result.exit_code == 1
    assert "Assignee cannot be empty" in result.output


# --- frontend/node_modules on a new worktree (#930) ------------------------------------
#
# The gate's vitest stage fails without `frontend/node_modules` (#915), and a new worktree
# never has one because it is untracked. `claim --create-worktree` is the only repository
# code that creates worktrees, so it links the primary workspace's copy. When there is no
# copy to link, it prints the fix instead of skipping silently.


def _commit_frontend(repo: Path) -> None:
    """Give the scratch repository a tracked `frontend/package.json`, as the real one has."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    (repo / "frontend").mkdir()
    (repo / "frontend" / "package.json").write_text('{"scripts": {"test": "vitest run"}}\n')
    for cmd in (["git", "add", "frontend/package.json"], ["git", "commit", "-m", "frontend"]):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True, env=env)


def test_claim_links_the_primary_workspace_node_modules_into_the_new_worktree(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new worktree's `frontend/node_modules` is a link to the primary's copy.

    Killed by: src/uclone_x/cli/commands/dev.py :: link.symlink_to(source, target_is_directory=True)
    Becomes: pass
    """
    _commit_frontend(scratch_repo)
    primary_deps = scratch_repo / "frontend" / "node_modules"
    (primary_deps / "vitest").mkdir(parents=True)
    _let_git_run_and_fake_gh(monkeypatch)

    result = runner.invoke(
        dev.dev_app,
        ["task", "claim", "20", "--assignee", "builder:one", "--create-worktree"],
    )

    assert result.exit_code == 0, result.output
    link = scratch_repo / ".worktrees" / "builder-one" / "frontend" / "node_modules"
    assert link.is_symlink()
    assert link.resolve() == primary_deps.resolve()
    assert (link / "vitest").is_dir()
    assert "Linked frontend/node_modules" in result.output


def test_claim_without_a_primary_node_modules_prints_the_fix_instead_of_skipping(
    scratch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No primary copy: nothing is linked, the claim succeeds, and the fix is printed (P6).

    Goes through the whole `claim` command, with only `gh` faked, so the worktree is real.

    Killed by: src/uclone_x/cli/commands/dev.py :: return "unlinked", [missing, *link_it, *install_here]
    Becomes: return "unlinked", []
    """
    _commit_frontend(scratch_repo)
    _let_git_run_and_fake_gh(monkeypatch)

    result = runner.invoke(
        dev.dev_app,
        ["task", "claim", "20", "--assignee", "builder:one", "--create-worktree"],
    )

    assert result.exit_code == 0, result.output
    worktree = scratch_repo / ".worktrees" / "builder-one"
    assert (worktree / "frontend" / "package.json").is_file()
    assert not (worktree / "frontend" / "node_modules").is_symlink()
    assert "frontend/node_modules was NOT linked" in result.output
    assert "does not exist" in result.output
    assert f"npm ci --prefix {scratch_repo.resolve() / 'frontend'}" in result.output


def test_a_worktree_without_a_frontend_gets_no_link_and_no_message(tmp_path: Path) -> None:
    """A tree that never claimed a frontend is left alone, the gate's `absent` case.

    Killed by: src/uclone_x/cli/commands/dev.py :: if not (frontend / "package.json").is_file():
    Becomes: if False:
    """
    (tmp_path / "frontend" / "node_modules").mkdir(parents=True)
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    assert dev.link_frontend_node_modules(worktree) == ("absent", [])
    assert not (worktree / "frontend").exists()


def test_an_existing_node_modules_in_the_worktree_is_left_alone(tmp_path: Path) -> None:
    """A worktree that already installed its own dependencies keeps them.

    Killed by: src/uclone_x/cli/commands/dev.py :: if link.is_symlink() or link.exists():
    Becomes: if False:
    """
    _init_git_repo(tmp_path)
    (tmp_path / "frontend" / "node_modules").mkdir(parents=True)
    own = tmp_path / "wt" / "frontend" / "node_modules"
    own.mkdir(parents=True)
    (own.parent / "package.json").write_text("{}\n")

    status, lines = dev.link_frontend_node_modules(own.parent.parent)

    assert status == "present"
    assert not own.is_symlink()
    assert "left as is" in lines[0]


def test_linking_twice_leaves_the_link_and_nests_nothing_in_the_primary_copy(
    tmp_path: Path,
) -> None:
    """A second link attempt over an existing link is a no-op, not a nested link.

    The hazard: a plain `ln -s <primary>/frontend/node_modules frontend/node_modules`,
    run where that link already exists (a manager pre-linked the worktree at dispatch),
    follows the link and creates `node_modules/node_modules` inside the primary
    workspace's copy. Measured with macOS `/bin/ln`: exit 0, and the nested link appeared.

    Killed by: src/uclone_x/cli/commands/dev.py :: if link.is_symlink() or link.exists():
    Becomes: if link.exists() and not link.is_symlink():
    """
    _init_git_repo(tmp_path)
    _commit_frontend(tmp_path)
    primary_deps = tmp_path / "frontend" / "node_modules"
    primary_deps.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    worktree = tmp_path / ".worktrees" / "twice"
    subprocess.run(
        ["git", "worktree", "add", str(worktree), "-b", "task/twice"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env=env,
    )

    first, _ = dev.link_frontend_node_modules(worktree)
    second, lines = dev.link_frontend_node_modules(worktree)

    assert (first, second) == ("linked", "present")
    assert "not followed" in lines[0]
    assert (worktree / "frontend" / "node_modules").resolve() == primary_deps.resolve()
    assert list(primary_deps.iterdir()) == []


def test_a_dangling_node_modules_link_is_reported_and_left_alone(tmp_path: Path) -> None:
    """A dangling link is not replaced and not followed, and the claim says it is broken.

    Killed by: src/uclone_x/cli/commands/dev.py :: if link.is_symlink() or link.exists():
    Becomes: if link.exists():
    """
    _init_git_repo(tmp_path)
    (tmp_path / "frontend" / "node_modules").mkdir(parents=True)
    worktree = tmp_path / "wt"
    (worktree / "frontend").mkdir(parents=True)
    (worktree / "frontend" / "package.json").write_text("{}\n")
    link = worktree / "frontend" / "node_modules"
    gone = tmp_path / "gone"
    link.symlink_to(gone, target_is_directory=True)

    status, lines = dev.link_frontend_node_modules(worktree)

    assert status == "unlinked"
    assert "is a dangling symlink" in lines[0]
    assert link.is_symlink() and link.readlink() == gone
    assert not gone.exists()


def test_a_worktree_outside_any_repository_gets_the_fix_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the primary workspace cannot be located, the claim still prints the fix."""
    for key in list(os.environ.keys()):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text("{}\n")

    status, lines = dev.link_frontend_node_modules(tmp_path)

    assert status == "unlinked"
    assert "the primary workspace could not be located" in lines[0]
    assert any(f"npm ci --prefix {tmp_path / 'frontend'}" in line for line in lines)
