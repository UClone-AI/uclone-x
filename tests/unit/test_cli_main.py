"""Unit tests for the UCX CLI main commands, setup, ui, run, version, llm, and dev commands."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import typer
import yaml
from typer.testing import CliRunner

from uclone_x.cli import main
from uclone_x.cli.commands import dev, llm
from uclone_x.errors import LLMProviderNotConfiguredError
from uclone_x.llm import create_llm_connector

runner = CliRunner()

_HOOK_ENV_KEYS = ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_PREFIX", "GIT_COMMON_DIR")


def _git_env_without_hook_vars() -> dict[str, str]:
    """Environment for git subprocesses in tests that build their own repositories.

    When the suite runs from the pre-commit hook, git exports GIT_DIR and GIT_INDEX_FILE
    for the repository running the hook. A `git init <dir>` or `git -C <dir> commit`
    inheriting them operates on *that* repository, not the temporary one — measured when
    a hook-run of this module reinitialised the real checkout. Every git call here uses
    this env, and each test also clears the variables from os.environ before `setup`.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _install_gate_stub(root: Path) -> Path:
    """Put a stand-in `./ucx` at `root` that records having been run, and return its marker.

    The hook's last line is `./ucx test check --skip-tests`, resolved against the working tree root git
    gives a hook as its cwd. A test cannot run the real gate (it would recurse into this
    suite), so the stub is the instrument that answers "was the gate reached?" — a
    question the hook's *text* cannot answer (§6.9 case 2: assert the instrument observed
    what it claims to observe). Absence of the marker is the assertion that the guard runs
    *before* the gate; presence is the assertion that a permitted commit still gets gated.
    """
    marker = root / "gate-ran"
    stub = root / "ucx"
    stub.write_text(
        f'#!/usr/bin/env bash\nprintf "%s" "$*" > {marker!s}\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return marker


def _bot_identity() -> tuple[str, str]:
    """The bot commit identity, read from the real `swarm/config.yaml`.

    Read rather than restated so the tests and the manifest cannot drift: the hook takes
    its expected value from that file, so a test carrying its own copy would keep passing
    after the manifest changed underneath it.

    A checkout with no swarm directory at all -- the published tree, which carries this
    module because the `cli` package's coverage floor depends on it -- gets a placeholder
    identity instead of a collection error. Every test here that uses the identity writes
    it into its own temporary manifest and repository, so the placeholder is consistent
    with itself; the drift this function guards against exists only where the real
    manifest does. The condition is the directory, not the file: a checkout that has
    the directory and lost its manifest still fails loudly.
    """
    swarm_dir = Path(__file__).resolve().parents[2] / "swarm"
    if not swarm_dir.is_dir():
        return "uclone-x-test-bot", "uclone-x-test-bot@example.invalid"
    manifest = swarm_dir / "config.yaml"
    loaded: object = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    identity = cast(dict[str, Any], cast(dict[str, Any], loaded)["bot"])["commit_identity"]
    return str(identity["name"]), str(identity["email"])


_BOT_NAME, _BOT_EMAIL = _bot_identity()

_MANIFEST_STUB = f"""version: 1
repo: "UClone-AI/uclone-x"
bot:
  login: "{_BOT_NAME}"
  commit_identity:
    name: "{_BOT_NAME}"
    email: "{_BOT_EMAIL}"
"""


def _stored_config(key: str, *, cwd: Path) -> str:
    """Read a git config value with NO `-c` overrides in play.

    `_git` injects `-c user.name=... -c user.email=...`, and `git -c user.name=X config
    --get user.name` returns **X**, not what is stored in the file. Reading identity
    through `_git` therefore echoes the test's own override back at it, and an assertion
    built that way passes whether or not `setup` wrote anything. Measured: it did.
    """
    return subprocess.run(
        ["git", "config", "--get", key],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_git_env_without_hook_vars(),
    ).stdout.strip()


def _git_as(name: str, email: str, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run git in `cwd` under an explicit author identity, GIT_* stripped."""
    return subprocess.run(
        ["git", "-c", f"user.email={email}", "-c", f"user.name={name}", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_git_env_without_hook_vars(),
    )


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run git in `cwd` as the bot, with the hook's GIT_* variables stripped.

    The bot identity is the default because it is what this repository's commits are
    authored as; the pre-commit hook refuses anything else. Tests that are *about* the
    identity check call `_git_as` with a deliberately wrong one.
    """
    return _git_as(_BOT_NAME, _BOT_EMAIL, *args, cwd=cwd)


def test_cli_version() -> None:
    result = runner.invoke(main.app, ["version"])
    assert result.exit_code == 0
    assert "UClone-X version" in result.output


def test_cli_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    git_dir = tmp_path / ".git" / "hooks"
    git_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(main.app, ["setup"])
    assert result.exit_code == 0
    hook_file = git_dir / "pre-commit"
    assert hook_file.exists()
    content = hook_file.read_text(encoding="utf-8")
    assert "./ucx test check" in content
    assert content == main.PRE_COMMIT_HOOK
    push_hook = git_dir / "pre-push"
    assert push_hook.read_text(encoding="utf-8") == main.PRE_PUSH_HOOK


def test_cli_setup_ignores_hook_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`setup` installs into the checkout in cwd even when a git hook's GIT_DIR is exported.

    Inside a running hook git exports GIT_DIR (and GIT_INDEX_FILE) for the repository
    running the hook. Left in the child environment, `git rev-parse --git-path hooks`
    answers for *that* repository regardless of cwd — measured when the pre-commit hook's
    own quality-gate run had this test write the hook into the real `.git/hooks`.
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    other_repo = tmp_path / "other"
    other_repo.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(other_repo)], check=True, env=_git_env_without_hook_vars()
    )
    target = tmp_path / "target"
    (target / ".git" / "hooks").mkdir(parents=True)
    monkeypatch.chdir(target)
    monkeypatch.setenv("GIT_DIR", str(other_repo / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(other_repo / ".git" / "index"))

    result = runner.invoke(main.app, ["setup"])
    assert result.exit_code == 0
    assert (target / ".git" / "hooks" / "pre-commit").exists()
    assert not (other_repo / ".git" / "hooks" / "pre-commit").exists()


def test_cli_setup_from_worktree_targets_common_hooks_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From a linked worktree, the hook lands in the common `.git/hooks`, not nowhere.

    A worktree's `.git` is a file, so the previous `Path(".git/hooks").is_dir()` check
    was false there and `setup` silently installed nothing.
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    env = _git_env_without_hook_vars()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, env=env)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "root",
        ],
        check=True,
        env=env,
    )
    worktree = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-q", "-b", "task/x", str(worktree)],
        check=True,
        env=env,
    )
    monkeypatch.chdir(worktree)

    result = runner.invoke(main.app, ["setup"])
    assert result.exit_code == 0
    assert (repo / ".git" / "hooks" / "pre-commit").exists()
    assert not (worktree / ".git").is_dir()


def test_cli_setup_reports_replacing_a_drifted_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing a hook whose content differs from the tracked source is announced.

    `setup` overwrites `.git/hooks/pre-commit` unconditionally, and the deployed hook is
    not necessarily the one this module defines: on 2026-09-03 the live one was narrowed
    in place, leaving no diff and no history (#285). This does not decide whether the
    narrowing was right — it decides that the replacement is not silent. A hook that
    matches the tracked source is replaced without the notice.
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    hooks = tmp_path / ".git" / "hooks"
    hooks.mkdir(parents=True)
    # The 2026-09-03 edit carried a 📄 emoji, so character count != byte count. The size
    # on disk is the identifying fact about that hook (1242 bytes), so the notice must
    # report bytes; `len(str)` would understate it.
    drifted = main.PRE_COMMIT_HOOK + '\necho "📄 docs-only"\nexit 0\n'
    assert len(drifted) != len(drifted.encode("utf-8"))
    (hooks / "pre-commit").write_text(drifted, encoding="utf-8")
    (hooks / "pre-push").write_text(main.PRE_PUSH_HOOK, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    output = " ".join(runner.invoke(main.app, ["setup"]).output.split())
    assert f"replaced a pre-commit hook ({len(drifted.encode('utf-8'))} bytes)" in output
    assert "replaced a pre-push hook" not in output
    assert (hooks / "pre-commit").read_text(encoding="utf-8") == main.PRE_COMMIT_HOOK


def test_cli_setup_fails_loudly_when_the_hooks_dir_git_names_is_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `core.hooksPath` cannot be created, `setup` aborts instead of falling back.

    Git reads hooks from `core.hooksPath` and nowhere else. Writing into `.git/hooks`
    instead produced `✔ Installed Git pre-commit hook (.git/hooks/pre-commit)` at exit 0
    for a hook git never reads — the silent-permission failure these hooks exist to
    prevent, arriving through the installer. `/nonexistent` is on the read-only root
    volume on darwin, so `mkdir(parents=True)` raises `OSError` there.
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("init", "-q", "-b", "main", ".", cwd=repo).returncode == 0
    unusable = "/nonexistent/deep/hooks"
    assert _git("config", "core.hooksPath", unusable, cwd=repo).returncode == 0
    # The instrument must observe an unusable directory, not merely an absent one.
    with pytest.raises(OSError):
        Path(unusable).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(repo)

    result = runner.invoke(main.app, ["setup"])
    assert result.exit_code != 0
    assert "cannot be created" in result.output
    assert not (repo / ".git" / "hooks" / "pre-commit").exists()
    assert "✔ Installed Git pre-commit hook" not in result.output


def test_cli_setup_fails_when_hooks_dir_parent_is_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When git resolves a hooks directory whose parent is a file, setup aborts.

    Killed by: src/uclone_x/cli/main.py :: if resolved.parent.is_file():
    Becomes: if not resolved.parent.is_file():

    In a linked worktree, `.git` is a pointer file. If `core.hooksPath=.git/hooks` is
    configured, git resolves hooks to `.git/hooks` whose parent is a regular file.
    Setup must refuse loudly rather than proceeding.
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("init", "-q", "-b", "main", ".", cwd=repo).returncode == 0
    assert _git("commit", "--allow-empty", "-m", "init", cwd=repo).returncode == 0
    worktree = tmp_path / "wt"
    assert (
        _git("worktree", "add", "-q", "-b", "task/wt-hooks", str(worktree), cwd=repo).returncode
        == 0
    )
    assert _git("config", "core.hooksPath", ".git/hooks", cwd=worktree).returncode == 0
    monkeypatch.chdir(worktree)

    result = runner.invoke(main.app, ["setup"])
    assert result.exit_code != 0
    assert "whose parent is a file" in result.output
    assert "Refusing to install" in result.output


def test_cli_setup_declines_from_a_subdirectory_of_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`setup` is a repo-root command and installs nothing when run from a subdirectory.

    `git rev-parse --git-path hooks` answers for the *enclosing* repository from anywhere
    inside it (measured: `../.git/hooks` from one level down), so without the `.git`
    existence check a `setup` run in some unrelated subdirectory would silently rewrite
    the hooks of whatever checkout happens to contain it.
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    assert _git("init", "-q", "-b", "main", ".", cwd=repo).returncode == 0
    monkeypatch.chdir(repo / "sub")

    result = runner.invoke(main.app, ["setup"])
    assert result.exit_code == 0
    assert "Installed Git" not in result.output
    assert not (repo / ".git" / "hooks" / "pre-commit").exists()
    assert not (repo / "sub" / ".git").exists()


def test_cli_setup_installs_where_core_hookspath_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `core.hooksPath` set, the hook lands where git looks — and git runs it.

    `git rev-parse --git-path hooks` honours `core.hooksPath` (measured on git 2.50.1),
    so the answer can name a directory that does not exist yet. Installing into
    `.git/hooks` instead would print `✔ Installed Git pre-commit hook` for a hook git
    never runs. The commit at the end is the proof that it is not merely written but live.
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("init", "-q", "-b", "main", ".", cwd=repo).returncode == 0
    assert _git("config", "core.hooksPath", "tools/githooks", cwd=repo).returncode == 0
    assert not (repo / "tools").exists()
    monkeypatch.chdir(repo)

    assert runner.invoke(main.app, ["setup"]).exit_code == 0
    installed = repo / "tools" / "githooks" / "pre-commit"
    assert installed.exists(), "setup installed nowhere git would look"
    assert os.access(installed, os.X_OK)
    assert not (repo / ".git" / "hooks" / "pre-commit").exists()

    _install_gate_stub(repo)
    (repo / "note.md").write_text("x\n", encoding="utf-8")
    assert _git("add", "note.md", cwd=repo).returncode == 0
    committed = _git("commit", "-m", "should be refused", cwd=repo)
    assert committed.returncode != 0
    assert "refusing to commit in the primary workspace" in committed.stderr


def test_pre_push_hook_refuses_main_and_allows_task_branches(tmp_path: Path) -> None:
    """Run the installed pre-push hook with git's stdin protocol.

    A fast-forward merge in the primary workspace followed by `git push` reached `main`
    on 2026-09-03 (`5a34169`) without pre-commit ever firing. The pre-push hook is the
    guard for that path: refuse when the remote ref is `refs/heads/main`, let every
    other ref through, and say what to do instead.
    """
    hook = tmp_path / "pre-push"
    hook.write_text(main.PRE_PUSH_HOOK, encoding="utf-8")
    hook.chmod(0o755)
    sha = "0" * 40

    refused = subprocess.run(
        [str(hook), "origin", "git@example:repo"],
        input=f"refs/heads/main {sha} refs/heads/main {sha}\n",
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 1
    assert "refusing to push to main" in refused.stderr
    assert "open one" in refused.stderr

    allowed = subprocess.run(
        [str(hook), "origin", "git@example:repo"],
        input=f"refs/heads/task/1-x {sha} refs/heads/task/1-x {sha}\n",
        capture_output=True,
        text=True,
    )
    assert allowed.returncode == 0
    assert allowed.stderr == ""

    # A push of several refs is refused as soon as one of them is main.
    mixed = subprocess.run(
        [str(hook), "origin", "git@example:repo"],
        input=f"refs/heads/task/1-x {sha} refs/heads/task/1-x {sha}\nrefs/heads/main {sha} refs/heads/main {sha}\n",
        capture_output=True,
        text=True,
    )
    assert mixed.returncode == 1


def _repo_with_hooks_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Build a real repository with one commit and both hooks installed by `ucx setup`.

    Installing through `setup` rather than writing the constants by hand is deliberate:
    it puts the install path — the hooks-directory resolver and the `chmod` — inside the
    behavioural test, so a hook that git would skip fails here rather than passing on
    its text. Returns (repo root, hooks dir).
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("init", "-q", "-b", "main", ".", cwd=repo).returncode == 0
    # The manifest is *committed*, not merely written: the hook resolves it through
    # `git rev-parse --show-toplevel`, which in a linked worktree is the worktree's own
    # checkout. An uncommitted file would exist in `repo` and be absent from every
    # worktree built from it, so the identity check would fail-closed there for the wrong
    # reason and the worktree tests would pass on a misread.
    (repo / "swarm").mkdir()
    (repo / "swarm" / "config.yaml").write_text(_MANIFEST_STUB, encoding="utf-8")
    assert _git("add", "swarm/config.yaml", cwd=repo).returncode == 0
    # A root commit before the hooks exist, so later tests have a parent to commit onto.
    assert _git("commit", "-q", "-m", "root", cwd=repo).returncode == 0

    monkeypatch.chdir(repo)
    result = runner.invoke(main.app, ["setup"])
    assert result.exit_code == 0
    hooks = repo / ".git" / "hooks"
    return repo, hooks


def test_setup_installs_hooks_git_will_actually_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both installed hooks are executable.

    Git skips a hook that is not executable **and prints nothing at all** — so a mode of
    0644 yields a repository that believes it is locked and is not, which is the same
    silent-permission failure the hooks exist to prevent. `content == CONSTANT` cannot
    see this; only the mode bit can.
    """
    _repo, hooks = _repo_with_hooks_installed(tmp_path, monkeypatch)
    for name in ("pre-commit", "pre-push"):
        assert os.access(hooks / name, os.X_OK), f"{name} installed non-executable"


def test_pre_commit_hook_refuses_a_real_commit_in_the_primary_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `git commit` in the primary workspace is refused, before the gate is reached.

    Invariant 1 (Swarm Guide §6): the primary workspace is read-only. This runs the hook
    the way git runs it — a real commit in a repository where `--git-dir` and
    `--git-common-dir` are the same directory — and asserts the commit did not happen,
    rather than asserting that the hook's source contains the words of a refusal. A hook
    with `exit 0` spliced in above the guard satisfies every text assertion and permits
    the commit; it fails here.
    """
    repo, _hooks = _repo_with_hooks_installed(tmp_path, monkeypatch)
    marker = _install_gate_stub(repo)
    # The instrument must be pointed at a primary workspace, not merely believed to be.
    git_dir = _git("rev-parse", "--absolute-git-dir", cwd=repo).stdout.strip()
    common = _git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=repo)
    assert git_dir == common.stdout.strip() != ""

    (repo / "note.md").write_text("x\n", encoding="utf-8")
    assert _git("add", "note.md", cwd=repo).returncode == 0
    committed = _git("commit", "-m", "commit in primary workspace", cwd=repo)

    assert committed.returncode != 0, committed.stdout
    assert "refusing to commit in the primary workspace" in committed.stderr
    assert "git worktree add" in committed.stderr
    assert not marker.exists(), "the gate ran before the guard refused"
    # The refusal is the whole point: no second commit exists.
    assert _git("rev-list", "--count", "HEAD", cwd=repo).stdout.strip() == "1"


def test_pre_commit_hook_lets_a_worktree_commit_through_to_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a linked worktree the guard passes and the quality gate is reached and run.

    The staged change here is documentation only, and that is the point of the case: the
    hook as this branch defines it gates **every** commit. On 2026-09-03 the live shared
    hook was edited in place to `exit 0` on a docs-only diff (#285) — a hole cut
    downstream of an intact guard, invisible to any assertion over the guard's text. This
    test fails if that edit is reintroduced into the constant.

    It pins current behaviour, not a ruling: #285 is undecided, and if the fast path is
    later adopted deliberately, this expectation changes in that PR, as a reviewable diff.
    """
    repo, _hooks = _repo_with_hooks_installed(tmp_path, monkeypatch)
    worktree = tmp_path / "wt"
    assert _git("worktree", "add", "-q", "-b", "task/1-x", str(worktree), cwd=repo).returncode == 0
    marker = _install_gate_stub(worktree)
    assert (
        _git("rev-parse", "--absolute-git-dir", cwd=worktree).stdout.strip()
        != _git(
            "rev-parse", "--path-format=absolute", "--git-common-dir", cwd=worktree
        ).stdout.strip()
    )

    # Staged under `docs/` with a `.md` suffix, so the change looks documentation-only to
    # a fast path keyed on either the path prefix or the file extension. The live
    # 14:04:56 edit keyed on extensions; a variant keyed on `docs/` is just as plausible,
    # and a file at the repo root would slip past that one.
    (worktree / "docs").mkdir()
    (worktree / "docs" / "doc.md").write_text("docs only\n", encoding="utf-8")
    assert _git("add", "docs/doc.md", cwd=worktree).returncode == 0
    committed = _git("commit", "-m", "docs-only commit in a worktree", cwd=worktree)

    assert committed.returncode == 0, committed.stderr
    assert "refusing to commit" not in committed.stderr
    # `--skip-tests` is the assertion, not an incidental argument: pre-commit runs the
    # static checks and must NOT run the suite. The suite runs at the PR head before merge
    # (#966; the pre-check redesign, sections 4.1 and 4.2). A commit-stage suite verifies a
    # tree that the squashed, rebased merge result never has.
    assert marker.read_text(encoding="utf-8") == "test check --skip-tests"


def test_pre_commit_hook_refuses_a_commit_authored_as_a_non_bot_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree commit authored as anything but the bot is refused before the gate.

    `ghx` makes GitHub *API* authorship structural, but `git commit` never passes through
    it. The shared `.git/config` carried `Test Builder <builder@example.com>`, which
    GitHub matches to a real human account — the #99 / #105 / #122 collision class. This
    exercises the exact value that was live, in a worktree so the primary-workspace guard
    is not what does the refusing.

    Mutation: delete the identity block from `PRE_COMMIT_HOOK` — this test fails, and
    `test_pre_commit_hook_refuses_a_real_commit_in_the_primary_workspace` still passes,
    which is what makes the two refusals separably pinned.
    """
    repo, _hooks = _repo_with_hooks_installed(tmp_path, monkeypatch)
    worktree = tmp_path / "wt"
    assert _git("worktree", "add", "-q", "-b", "task/1-x", str(worktree), cwd=repo).returncode == 0
    marker = _install_gate_stub(worktree)

    (worktree / "note.md").write_text("x\n", encoding="utf-8")
    assert _git("add", "note.md", cwd=worktree).returncode == 0
    committed = _git_as(
        "Test Builder", "builder@example.com", "commit", "-m", "wrong author", cwd=worktree
    )

    assert committed.returncode != 0, committed.stdout
    assert "refusing a commit authored as 'Test Builder <builder@example.com>'" in committed.stderr
    assert not marker.exists(), "the gate ran before the identity guard refused"
    # The refusal is the point: the worktree still has only the root commit.
    assert _git("rev-list", "--count", "HEAD", cwd=worktree).stdout.strip() == "1"


def test_pre_commit_hook_identity_refusal_names_a_remedy_that_actually_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal prints the bot identity, and committing with it succeeds.

    A refusal without a usable remedy converts a silent defect into a blocked builder, so
    the message is only load-bearing if what it names works. This runs the refusal, reads
    the expected identity **out of the refusal's own text**, and commits with that — so
    the test fails if the message names a value the hook would not accept.

    Mutation: print a stale literal in the message instead of `$expected_name` /
    `$expected_email` — this test fails at the second commit.
    """
    repo, _hooks = _repo_with_hooks_installed(tmp_path, monkeypatch)
    worktree = tmp_path / "wt"
    assert _git("worktree", "add", "-q", "-b", "task/1-x", str(worktree), cwd=repo).returncode == 0
    marker = _install_gate_stub(worktree)
    (worktree / "note.md").write_text("x\n", encoding="utf-8")
    assert _git("add", "note.md", cwd=worktree).returncode == 0

    refused = _git_as("Someone Else", "nope@example.com", "commit", "-m", "x", cwd=worktree)
    assert refused.returncode != 0

    # Parse the identity back out of the guidance rather than restating it here.
    quoted = re.findall(r"pre-commit:\s+(\S.*?) <(\S+?)>$", refused.stderr, re.MULTILINE)
    named = [pair for pair in quoted if pair[1] != "nope@example.com"]
    assert named, f"the refusal named no usable identity: {refused.stderr}"
    name, email = named[0]

    accepted = _git_as(name, email, "commit", "-m", "with the named identity", cwd=worktree)
    assert accepted.returncode == 0, accepted.stderr
    # `--skip-tests` is the assertion, not an incidental argument: pre-commit runs the
    # static checks and must NOT run the suite. The suite runs at the PR head before merge
    # (#966; the pre-check redesign, sections 4.1 and 4.2). A commit-stage suite verifies a
    # tree that the squashed, rebased merge result never has.
    assert marker.read_text(encoding="utf-8") == "test check --skip-tests"


def test_pre_commit_hook_fails_closed_and_loudly_when_the_manifest_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable identity rule refuses the commit AND says why.

    Two assertions, and the second is the one that was missing: an early revision aborted
    under `set -e` at the extraction itself, which exited 1 — closed, but **silently**,
    printing none of the guidance. Measured during development, which is why both halves
    are pinned here.

    Mutation: drop the `|| true` from the extraction — the commit is still refused, but
    the explanatory line disappears and this test fails on the stderr assertion.
    """
    repo, _hooks = _repo_with_hooks_installed(tmp_path, monkeypatch)
    worktree = tmp_path / "wt"
    assert _git("worktree", "add", "-q", "-b", "task/1-x", str(worktree), cwd=repo).returncode == 0
    marker = _install_gate_stub(worktree)
    # Remove the worktree's OWN copy: the hook resolves the manifest through
    # `--show-toplevel`, so deleting the parent checkout's copy would leave this one
    # readable and the test would pass without ever exercising the branch (§6.9 case 2).
    (worktree / "swarm" / "config.yaml").unlink()

    (worktree / "note.md").write_text("x\n", encoding="utf-8")
    assert _git("add", "note.md", cwd=worktree).returncode == 0
    committed = _git("commit", "-m", "manifest gone", cwd=worktree)

    assert committed.returncode != 0
    assert "cannot read bot.commit_identity" in committed.stderr
    assert not marker.exists()


def test_cli_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_repl = MagicMock()
    monkeypatch.setattr("uclone_x.cli.commands.run.run_agent_repl", mock_repl)
    result = runner.invoke(main.app, ["run", "test-agent"])
    assert result.exit_code == 0
    assert mock_repl.called


def test_cli_agent_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_repl = MagicMock()
    monkeypatch.setattr("uclone_x.cli.commands.run.run_agent_repl", mock_repl)
    result = runner.invoke(main.app, ["agent", "run", "test-agent"])
    assert result.exit_code == 0
    assert mock_repl.called


def test_the_provider_option_help_names_every_provider_an_installation_can_be_set_to() -> None:
    """`--provider`'s help is an enumeration, and an enumeration goes stale silently.

    `create_llm_connector`'s refusal is the one message that lists what an installation can
    actually be set to, and `./ucx run --provider`'s help is where an operator reads the same
    list before running anything. vLLM reached the factory and the refusal while this help
    still named five providers — a provider absent from it is a provider nobody discovers
    (the same reasoning as the refusal's own test). Deriving the expected names from the
    refusal keeps the pair together rather than asking the next provider's author to remember
    one help string.

    Killed by: src/uclone_x/cli/main.py :: "LLM Provider: ollama | vllm | openai | anthropic | gemini | mock. "
    Becomes: "LLM Provider: ollama | openai | anthropic | gemini | mock. "
    """
    with pytest.raises(LLMProviderNotConfiguredError) as excinfo:
        create_llm_connector()
    listed = re.search(r"LLM_PROVIDER=([a-z|]+)", str(excinfo.value))
    assert listed is not None, "the refusal no longer lists LLM_PROVIDER's accepted values"
    providers = listed.group(1).split("|")
    assert "vllm" in providers

    result = runner.invoke(main.app, ["run", "--help"])
    assert result.exit_code == 0
    help_text = re.sub(r"\s+", " ", result.output)
    for name in providers:
        assert name in help_text, f"--provider help omits {name}"


def test_cli_start(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find(preferred_port: int, max_attempts: int = 50) -> int:
        return 5180

    mock_start = MagicMock()
    mock_bootstrap = MagicMock(return_value=(True, "qwen3:8b"))
    monkeypatch.setattr(main, "find_available_port", fake_find)
    monkeypatch.setattr("uclone_x.ui.server.start_ui_server", mock_start)
    monkeypatch.setattr("uclone_x.cli.commands.bootstrap.ensure_local_profile", mock_bootstrap)

    result = runner.invoke(main.app, ["start", "--skip-setup", "--no-open", "--cwd", "/tmp"])
    assert result.exit_code == 0
    assert mock_start.called
    mock_start.assert_called_once_with(
        port=5180,
        dev=False,
        host="127.0.0.1",
        auto_open_browser=False,
        workspace_dir=Path("/tmp").resolve(),
    )


def test_cli_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find(preferred_port: int, max_attempts: int = 50) -> int:
        return 5181

    mock_start = MagicMock()
    monkeypatch.setattr(main, "find_available_port", fake_find)
    monkeypatch.setattr("uclone_x.ui.server.start_ui_server", mock_start)
    result = runner.invoke(main.app, ["ui", "--port", "5180", "--cwd", "/tmp"])
    assert result.exit_code == 0
    assert mock_start.called
    mock_start.assert_called_once_with(
        port=5181,
        dev=False,
        host="127.0.0.1",
        vite_port=5173,
        workspace_dir=Path("/tmp").resolve(),
    )


def test_cli_test_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Not a git checkout, on purpose: `test check` now records a gate pass for HEAD after a
    # passing run (#966), and with the gate faked to pass, running this from the real
    # worktree would write a record for a commit nothing verified.
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_gate(
        quiet: bool = False,
        skip_tests: bool = False,
        check_frontend: bool = False,
        fail_fast: bool = True,
        test_scope: str = "unit",
        serial: bool = False,
    ) -> int:
        captured["test_scope"] = test_scope
        captured["check_frontend"] = check_frontend
        captured["skip_tests"] = skip_tests
        captured["fail_fast"] = fail_fast
        captured["serial"] = serial
        return 0

    monkeypatch.setattr(main, "run_quality_gate", fake_gate)
    result = runner.invoke(main.app, ["test", "check"])
    assert result.exit_code == 0
    # `check` runs the offline gate — unit and integration tests plus the fitness functions —
    # rather than the `unit` level alone, which now excludes them.
    assert captured["test_scope"] == "gate"
    assert captured["fail_fast"] is True

    # `--all` is now a no-op synonym: the default gate already includes E2E. Kept because
    # it appears in AGENTS.md and the review checklist, and silently removing a documented
    # flag is worse than honouring it.
    result_all = runner.invoke(main.app, ["test", "check", "--all"])
    assert result_all.exit_code == 0
    assert captured["test_scope"] == "gate"

    # `--fast` is the opt-out that drops the browser suite.
    result_fast = runner.invoke(main.app, ["test", "check", "--fast"])
    assert result_fast.exit_code == 0
    assert captured["test_scope"] == "fast"

    # `--no-fail-fast` runs all checks before failing.
    result_no_ff = runner.invoke(main.app, ["test", "check", "--no-fail-fast"])
    assert result_no_ff.exit_code == 0
    assert captured["fail_fast"] is False
    # Parallel workers by default; `--serial` is the documented escape for debugging (#967).
    assert captured["serial"] is False

    result_serial = runner.invoke(main.app, ["test", "check", "--serial"])
    assert result_serial.exit_code == 0
    assert captured["serial"] is True
    assert captured["test_scope"] == "gate"


def test_cli_test_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_gate(
        quiet: bool = False,
        skip_tests: bool = False,
        check_frontend: bool = False,
        fail_fast: bool = True,
        test_scope: str = "unit",
        serial: bool = False,
    ) -> int:
        captured["test_scope"] = test_scope
        captured["serial"] = serial
        return 0

    monkeypatch.setattr(main, "run_quality_gate", fake_gate)
    result = runner.invoke(main.app, ["test", "unit"])
    assert result.exit_code == 0
    assert captured["test_scope"] == "unit"

    result_serial = runner.invoke(main.app, ["test", "unit", "--serial"])
    assert result_serial.exit_code == 0
    assert captured["serial"] is True


def test_cli_test_e2e(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_gate(
        quiet: bool = False,
        skip_tests: bool = False,
        check_frontend: bool = False,
        fail_fast: bool = True,
        test_scope: str = "unit",
        serial: bool = False,
    ) -> int:
        captured["test_scope"] = test_scope
        return 0

    monkeypatch.setattr(main, "run_quality_gate", fake_gate)
    result = runner.invoke(main.app, ["test", "e2e"])
    assert result.exit_code == 0
    assert captured["test_scope"] == "e2e"


def test_cli_test_pre_release(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_gate(
        quiet: bool = False,
        skip_tests: bool = False,
        check_frontend: bool = False,
        fail_fast: bool = True,
        test_scope: str = "unit",
        serial: bool = False,
    ) -> int:
        captured["test_scope"] = test_scope
        return 0

    monkeypatch.setattr(main, "run_quality_gate", fake_gate)
    result = runner.invoke(main.app, ["test", "pre-release"])
    assert result.exit_code == 0
    assert captured["test_scope"] == "pre-release"


def test_cli_test_check_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # not a checkout: see test_cli_test_check

    def fake_gate(
        quiet: bool = False,
        skip_tests: bool = False,
        check_frontend: bool = False,
        fail_fast: bool = True,
        test_scope: str = "unit",
        serial: bool = False,
    ) -> int:
        return 1

    monkeypatch.setattr(main, "run_quality_gate", fake_gate)
    result = runner.invoke(main.app, ["test", "check"])
    assert result.exit_code == 1


# --- `./ucx test check` writes the gate-pass record (#966) ------------------------------
#
# Owner ruling 2026-09-15: the pre-push hook no longer runs the gate, so the record the
# Builder Manager's pre-merge check (c) reads is written by the gate itself. These run the
# real CLI command and real git against a `tmp_path` repository; only the gate's verdict is
# faked, because the real gate would recurse into this suite.


def _committed_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository with one committed file, made the cwd. Returns its root."""
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("init", "-q", ".", cwd=repo).returncode == 0
    (repo / "tracked.txt").write_text("v1\n", encoding="utf-8")
    assert _git("add", "tracked.txt", cwd=repo).returncode == 0
    assert _git("commit", "-q", "-m", "root", cwd=repo).returncode == 0
    monkeypatch.chdir(repo)
    return repo


def _head_of(repo: Path) -> str:
    return _git("rev-parse", "HEAD", cwd=repo).stdout.strip()


def _fake_gate_verdict(
    monkeypatch: pytest.MonkeyPatch, exit_code: int, during: Any = None
) -> list[str]:
    """Replace the gate with one returning `exit_code`, running `during()` mid-run."""
    scopes: list[str] = []

    def fake_gate(
        quiet: bool = False,
        skip_tests: bool = False,
        check_frontend: bool = False,
        fail_fast: bool = True,
        test_scope: str = "gate",
        serial: bool = False,
    ) -> int:
        scopes.append(test_scope)
        if during is not None:
            during()
        return exit_code

    monkeypatch.setattr(main, "run_quality_gate", fake_gate)
    return scopes


def _recorded(repo: Path) -> list[str]:
    records = repo / ".git" / "gate-pass"
    return sorted(p.name for p in records.iterdir()) if records.exists() else []


def _flat(output: str) -> str:
    return " ".join(output.split())


@pytest.mark.parametrize(
    "args", [["test", "check"], ["test", "check", "--all"]], ids=["plain", "all"]
)
def test_a_passing_full_gate_on_a_clean_commit_records_head(
    args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record is written for exactly HEAD, and the run says so on one line.

    `--all` is a no-op synonym for the default scope, so it records too.

    Killed by: src/uclone_x/cli/quality_gate.py :: record.touch()
    Becomes: pass
    """
    repo = _committed_repo(tmp_path, monkeypatch)
    _fake_gate_verdict(monkeypatch, 0)

    result = runner.invoke(main.app, args)

    assert result.exit_code == 0, result.output
    sha = _head_of(repo)
    assert _recorded(repo) == [sha]
    assert f"Gate pass recorded for {sha}" in _flat(result.output)


def test_a_record_written_from_a_linked_worktree_lands_in_the_common_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every worktree shares one record store, which is where check (c) looks.

    Killed by: src/uclone_x/cli/quality_gate.py :: ["rev-parse", "--path-format=absolute", "--git-common-dir"]
    Becomes: ["rev-parse", "--path-format=absolute", "--git-dir"]
    """
    repo = _committed_repo(tmp_path, monkeypatch)
    worktree = tmp_path / "wt"
    assert _git("worktree", "add", "-q", "--detach", str(worktree), cwd=repo).returncode == 0
    monkeypatch.chdir(worktree)
    _fake_gate_verdict(monkeypatch, 0)

    result = runner.invoke(main.app, ["test", "check"])

    assert result.exit_code == 0, result.output
    assert _recorded(repo) == [_head_of(worktree)]


def _make_dirty(repo: Path, how: str) -> None:
    """Leave `repo`'s tree differing from HEAD in one of the three ways git reports."""
    if how == "modified":
        (repo / "tracked.txt").write_text("v2\n", encoding="utf-8")
    elif how == "staged":
        (repo / "staged.txt").write_text("s\n", encoding="utf-8")
        assert _git("add", "staged.txt", cwd=repo).returncode == 0
    else:
        (repo / "untracked.txt").write_text("u\n", encoding="utf-8")


@pytest.mark.parametrize("dirty", ["modified", "staged", "untracked"])
def test_a_passing_gate_on_a_dirty_tree_records_nothing(
    dirty: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record claims the gate ran on the commit; a dirty tree is not the commit.

    Untracked files count: the gate reads more than the paths it names, so an untracked
    test or fixture can change its verdict.

    Killed by: src/uclone_x/cli/quality_gate.py :: if changed:
    Becomes: if False:
    """
    repo = _committed_repo(tmp_path, monkeypatch)
    _make_dirty(repo, dirty)
    _fake_gate_verdict(monkeypatch, 0)

    result = runner.invoke(main.app, ["test", "check"])

    assert result.exit_code == 0, result.output
    assert _recorded(repo) == []
    assert "Gate pass not recorded: before the run, the tree is not clean" in _flat(result.output)


def test_a_failing_gate_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A record for a failed run would let a merge through on a verdict that was red.

    Killed by: src/uclone_x/cli/quality_gate.py :: if exit_code != 0:
    Becomes: if False:
    """
    repo = _committed_repo(tmp_path, monkeypatch)
    _fake_gate_verdict(monkeypatch, 1)

    result = runner.invoke(main.app, ["test", "check"])

    assert result.exit_code == 1
    assert _recorded(repo) == []
    assert "Gate pass not recorded: the gate failed." in _flat(result.output)


def test_a_fast_run_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--fast` drops the browser suite, so it did not verify what a merge needs.

    Killed by: src/uclone_x/cli/quality_gate.py :: GATE_PASS_SCOPES: Final[frozenset[str]] = frozenset({"gate"})
    Becomes: GATE_PASS_SCOPES: Final[frozenset[str]] = frozenset({"gate", "fast"})
    """
    repo = _committed_repo(tmp_path, monkeypatch)
    scopes = _fake_gate_verdict(monkeypatch, 0)

    result = runner.invoke(main.app, ["test", "check", "--fast"])

    assert result.exit_code == 0, result.output
    assert scopes == ["fast"]
    assert _recorded(repo) == []
    assert "Gate pass not recorded: a --fast run is not the full gate" in _flat(result.output)


def test_a_skip_tests_run_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--skip-tests` (the pre-commit hook's run) is static checks only.

    Killed by: src/uclone_x/cli/quality_gate.py :: return not skip_tests and test_scope in GATE_PASS_SCOPES
    Becomes: return test_scope in GATE_PASS_SCOPES
    """
    repo = _committed_repo(tmp_path, monkeypatch)
    _fake_gate_verdict(monkeypatch, 0)

    result = runner.invoke(main.app, ["test", "check", "--skip-tests"])

    assert result.exit_code == 0, result.output
    assert _recorded(repo) == []
    assert "Gate pass not recorded: a --skip-tests run" in _flat(result.output)


def test_a_tree_cleaned_during_the_run_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dirty while the suite ran, clean afterwards (edits reverted): the commit was not run.

    Killed by: src/uclone_x/cli/quality_gate.py :: head = before.head
    Becomes: head = snapshot_committed_tree(cwd).head
    """
    repo = _committed_repo(tmp_path, monkeypatch)
    (repo / "tracked.txt").write_text("edited while gating\n", encoding="utf-8")
    _fake_gate_verdict(
        monkeypatch, 0, during=lambda: (repo / "tracked.txt").write_text("v1\n", encoding="utf-8")
    )

    result = runner.invoke(main.app, ["test", "check"])

    assert result.exit_code == 0, result.output
    assert _recorded(repo) == []


def test_a_head_that_moved_during_the_run_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit made while the gate ran: neither the old nor the new HEAD is what passed.

    Killed by: src/uclone_x/cli/quality_gate.py :: if after.head != head:
    Becomes: if False:
    """
    repo = _committed_repo(tmp_path, monkeypatch)

    def commit_mid_run() -> None:
        (repo / "tracked.txt").write_text("v2\n", encoding="utf-8")
        assert _git("commit", "-q", "-am", "mid-run", cwd=repo).returncode == 0

    _fake_gate_verdict(monkeypatch, 0, during=commit_mid_run)

    result = runner.invoke(main.app, ["test", "check"])

    assert result.exit_code == 0, result.output
    assert _recorded(repo) == []
    assert "HEAD moved during the run" in _flat(result.output)


def test_cli_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_app = MagicMock()
    monkeypatch.setattr(main, "app", mock_app)
    main.main()
    assert mock_app.called


def test_find_available_port() -> None:
    port = main.find_available_port(5180)
    assert isinstance(port, int)
    assert port >= 5180


def test_llm_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_check(url: str, timeout: float = 2.0) -> list[str] | None:
        return ["qwen2.5-coder:7b"]

    monkeypatch.setattr(llm, "_check_ollama_endpoint", fake_check)
    result = runner.invoke(main.app, ["llm", "status"])
    assert result.exit_code == 0
    assert "UClone-X Self-Hosted LLM Topology Status" in result.output
    assert "qwen2.5-coder:7b" in result.output


def test_llm_status_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_check(url: str, timeout: float = 2.0) -> list[str] | None:
        return None

    monkeypatch.setattr(llm, "_check_ollama_endpoint", fake_check)
    result = runner.invoke(main.app, ["llm", "status"])
    assert result.exit_code == 0
    assert "UNREACHABLE" in result.output
    assert "OFFLINE" in result.output


def _fake_which(_cmd: str) -> str:
    return "/usr/local/bin/ollama"


def _fake_check_endpoint(_url: str) -> list[str]:
    return []


def test_llm_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = MagicMock()
    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr("shutil.which", _fake_which)
    monkeypatch.setattr("uclone_x.cli.commands.llm._check_ollama_endpoint", _fake_check_endpoint)

    result = runner.invoke(main.app, ["llm", "pull", "fast"])
    assert result.exit_code == 0
    assert mock_run.called

    result_indepth = runner.invoke(main.app, ["llm", "pull", "indepth"])
    assert result_indepth.exit_code == 0


def test_llm_pull_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, ["ollama", "pull"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", _fake_which)
    monkeypatch.setattr("uclone_x.cli.commands.llm._check_ollama_endpoint", _fake_check_endpoint)
    result = runner.invoke(main.app, ["llm", "pull", "custom-model"])
    assert result.exit_code == 1


def test_cli_dev_task_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """`dev task list` is reachable from the top-level app and queries the board.

    `gh` is stubbed rather than run: the assertion is about the wiring from `main.app`,
    and a real `gh` would make it depend on network and credentials.
    """

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(dev.subprocess, "run", fake_run)

    result = runner.invoke(main.app, ["dev", "task", "list"])
    assert result.exit_code == 0
    # The notice names the board generically now: the number and URL come from
    # `swarm/config.yaml`, because this module ships in the distribution and a
    # hardcoded URL published the address of a board a public reader cannot open.
    assert "the project board" in result.output


def test_cli_dev_issue_group_is_gone() -> None:
    """`ucx dev issue` was the file-backed findings register; it is retired (#1037)."""
    result = runner.invoke(main.app, ["dev", "issue", "list", "--all"])
    assert result.exit_code != 0
    assert "Design Review Findings Register" not in result.output


def test_cli_run_single_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    result = runner.invoke(
        main.app,
        ["run", "test_bot", "--provider", "mock", "--prompt", "Hello UClone-X"],
    )
    assert result.exit_code == 0
    assert "test_bot" in result.output
    assert "Hello UClone-X" in result.output


def test_cli_run_interactive_repl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`/reset` is asserted by its **effect**, not by its banner.

    This test previously ran `/reset` and checked only that a banner appeared in the
    output. That passes even when the reset does nothing at all: turning `BaseAgent`'s
    `_history` setter into a silent no-op left the whole suite green, and before this
    seam `run.py` reached the agent through exactly that setter. Ordering a turn before
    the reset and reading the message count out of `/status` on both sides is what makes
    the assertion about the reset rather than about the print statement (#211 review).

    The storage root is redirected because the REPL now persists through the Core
    `SessionStore`: a unit test that wrote a session under `~/.uclone` would have escaped
    its own sandbox.
    """
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    inputs = iter(["Hello world", "/status", "/reset", "/status", "/exit"])
    monkeypatch.setattr(
        "rich.prompt.Prompt.ask",
        lambda *args, **kwargs: next(inputs),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    )

    result = runner.invoke(main.app, ["run", "interactive_bot", "--provider", "mock"])
    assert result.exit_code == 0
    assert "UClone-X Autonomous Agent REPL" in result.output
    assert "interactive_bot" in result.output
    # `/reset` now reports the session it reset and its anchored message count, and it
    # goes through `BaseAgent.reset_session` rather than assigning `agent._history`.
    assert "reset" in result.output
    assert "Session finished" in result.output

    counts = [int(n) for n in re.findall(r"History Messages:\s+(\d+)", result.output)]
    assert len(counts) == 2, f"expected two /status readings, got {counts}"
    before_reset, after_reset = counts
    assert before_reset > after_reset, (
        f"/reset did not shrink the history ({before_reset} -> {after_reset}); the "
        "banner printed but nothing changed"
    )
    assert after_reset == 1, "reset should leave exactly the seeded SYSTEM message"


def test_cli_status_command_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    fake_payload: dict[str, Any] = {
        "version": "0.1.0",
        "absorbed_failures": {
            "dropped_spans": {"count": 2, "reasons": {"buffer_overflow": 2}},
            "event_bus_drops": {"count": 1, "reasons": {"ingress_drop_incoming": 1}},
            "agent_processing_errors": {"count": 0, "agents": {}},
        },
    }

    def _fake_get(self: Any, url: str, *args: object, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json=fake_payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", _fake_get)
    result = runner.invoke(main.app, ["status"])
    assert result.exit_code == 0
    assert "UClone-X Runtime Status" in result.output
    assert "Telemetry Tracer" in result.output
    assert "Event Bus" in result.output
    assert "Agent Core" in result.output


def test_cli_status_command_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    fake_payload: dict[str, Any] = {
        "version": "0.1.0",
        "absorbed_failures": {
            "dropped_spans": {"count": 0, "reasons": {}},
            "event_bus_drops": {"count": 0, "reasons": {}},
            "agent_processing_errors": {"count": 0, "agents": {}},
        },
    }

    def _fake_get(self: Any, url: str, *args: object, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json=fake_payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", _fake_get)
    result = runner.invoke(main.app, ["status", "--json"])
    assert result.exit_code == 0
    assert '"absorbed_failures"' in result.output


def test_cli_status_command_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def _fake_get(self: Any, url: str, *args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", _fake_get)
    result = runner.invoke(main.app, ["status"])
    assert result.exit_code == 1
    assert "Failed to connect to UClone-X server" in result.output


def test_setup_normalises_commit_identity_to_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`setup` points the repository's git identity at `bot.commit_identity`.

    Hook and identity must move together. Installing the identity refusal while the config
    still names someone else would refuse the next commit in every worktree at once, so
    `setup` does both in one pass and the "installed but nothing can be committed" state
    has no window to exist in.

    Mutation: delete the `_normalise_commit_identity()` call from `setup` — this test
    fails. It asserts the resulting config, not the console line: a message claiming the
    identity was set is exactly what a broken implementation would also print.
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("init", "-q", "-b", "main", ".", cwd=repo).returncode == 0
    (repo / "swarm").mkdir()
    (repo / "swarm" / "config.yaml").write_text(_MANIFEST_STUB, encoding="utf-8")
    assert (
        _git_as(
            "Someone Else", "human@example.com", "config", "user.name", "Someone Else", cwd=repo
        ).returncode
        == 0
    )
    assert _git("config", "user.email", "human@example.com", cwd=repo).returncode == 0

    monkeypatch.chdir(repo)
    assert runner.invoke(main.app, ["setup"]).exit_code == 0

    assert _stored_config("user.name", cwd=repo) == _BOT_NAME
    assert _stored_config("user.email", cwd=repo) == _BOT_EMAIL


def test_setup_says_out_loud_that_it_wrote_the_shared_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The identity change is announced, naming what it replaced.

    `extensions.worktreeConfig` is unset in this repository, so there is no per-worktree
    git config: the write lands in the shared `.git/config` that every worktree and the
    primary workspace read. A command that changes shared state while looking local is the
    shape this repository keeps paying for, so `setup` reports the scope and the previous
    value rather than doing it quietly.

    Mutation: drop the second `console.print` in `_normalise_commit_identity` — this test
    fails.
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("init", "-q", "-b", "main", ".", cwd=repo).returncode == 0
    (repo / "swarm").mkdir()
    (repo / "swarm" / "config.yaml").write_text(_MANIFEST_STUB, encoding="utf-8")
    assert _git("config", "user.name", "Someone Else", cwd=repo).returncode == 0
    assert _git("config", "user.email", "human@example.com", cwd=repo).returncode == 0

    monkeypatch.chdir(repo)
    output = " ".join(runner.invoke(main.app, ["setup"]).output.split())

    assert "shared .git/config" in output
    assert "Someone Else" in output and "human@example.com" in output
    assert "EVERY" in output


def test_setup_leaves_identity_alone_when_the_manifest_declares_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repository with no `bot.commit_identity` keeps its identity, and is told why.

    Mutation: fall back to a hardcoded default instead of returning None — this test fails.
    `setup` must not invent an identity for a repository that never declared one; the swarm
    is meant to run more than this repository, and each declares its own.
    """
    for key in _HOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("init", "-q", "-b", "main", ".", cwd=repo).returncode == 0
    assert _git("config", "user.name", "Someone Else", cwd=repo).returncode == 0
    assert _git("config", "user.email", "human@example.com", cwd=repo).returncode == 0

    monkeypatch.chdir(repo)
    output = " ".join(runner.invoke(main.app, ["setup"]).output.split())

    assert _stored_config("user.name", cwd=repo) == "Someone Else"
    assert "no bot.commit_identity" in output


def _repo_for_pre_push(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A real repo with the pre-push hook and a `./ucx` that fails loudly if anything runs it.

    Returns (repo, hook, marker). The stub writes `marker`, prints to stderr and exits 97,
    so a hook that still reached the gate would show up three ways: the marker, the stderr,
    and — for a hook that honours the gate's status — the exit code.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("init", "-q", cwd=repo).returncode == 0
    marker = repo / "gate-ran"
    stub = repo / "ucx"
    stub.write_text(
        f'#!/usr/bin/env bash\nprintf "%s" "$*" > {marker!s}\n'
        'echo "STUB GATE INVOKED: the pre-push hook must not run ./ucx" >&2\nexit 97\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    hook = repo / "pre-push"
    hook.write_text(main.PRE_PUSH_HOOK, encoding="utf-8")
    hook.chmod(0o755)
    return repo, hook, marker


def _run_pre_push(hook: Path, repo: Path, stdin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(hook), "origin", "git@example:repo"],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=repo,
        env=_git_env_without_hook_vars(),
    )


def _gate_pass_dir(repo: Path) -> Path:
    common = _git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=repo)
    return Path(common.stdout.strip()) / "gate-pass"


def test_pre_push_runs_no_gate_and_writes_no_record_for_a_branch_push(tmp_path: Path) -> None:
    """Pushing a task branch runs no test gate and writes no gate-pass record (#966).

    Owner ruling 2026-09-15: the push-time gate re-ran the suite on every push of every
    branch, and unrelated environment noise refused pushes of finished work. The gate moved
    to immediately before merge, and `./ucx test check` writes the record itself.

    Killed by: src/uclone_x/cli/main.py :: exit 0  # any other ref: no gate run, no gate-pass record (#966)
    Becomes: ./ucx test check || exit 1; exit 0
    """
    repo, hook, marker = _repo_for_pre_push(tmp_path)
    sha = "a" * 40

    result = _run_pre_push(
        hook, repo, f"refs/heads/task/1-x {sha} refs/heads/task/1-x {'b' * 40}\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert not marker.exists(), "the pre-push hook must not run the gate"
    assert not _gate_pass_dir(repo).exists(), "the pre-push hook must not write a record"


def test_pre_push_still_refuses_a_push_to_main_with_a_real_sha(tmp_path: Path) -> None:
    """Dropping the gate kept the hook's first duty: `main` changes only through a PR.

    Killed by: src/uclone_x/cli/main.py :: if [ "$remote_ref" = "refs/heads/main" ]; then
    Becomes: if [ "$remote_ref" = "refs/heads/master" ]; then
    """
    repo, hook, marker = _repo_for_pre_push(tmp_path)
    sha = "c" * 40

    result = _run_pre_push(hook, repo, f"refs/heads/main {sha} refs/heads/main {'d' * 40}\n")

    assert result.returncode == 1
    assert "refusing to push to main" in result.stderr
    assert not marker.exists()
    assert not _gate_pass_dir(repo).exists()


def test_pre_push_lets_a_branch_deletion_through(tmp_path: Path) -> None:
    """A deletion pushes the all-zero sha; it is not `main` and must exit 0 with nothing run.

    Killed by: src/uclone_x/cli/main.py :: exit 0  # any other ref: no gate run, no gate-pass record (#966)
    Becomes: exit 1
    """
    repo, hook, marker = _repo_for_pre_push(tmp_path)

    result = _run_pre_push(hook, repo, f"(delete) {'0' * 40} refs/heads/task/1-x {'e' * 40}\n")

    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    assert not _gate_pass_dir(repo).exists()


# --- what an installed build exposes (#605) -------------------------------------------

_USER_COMMANDS = {"version", "run", "ui", "status", "a2a", "eval", "llm", "ontology", "skill"}
_DEVELOPER_COMMANDS = {"setup", "test", "dev"}


def _registered_command_names(application: typer.Typer) -> set[str]:
    names = {
        c.name or (c.callback.__name__ if c.callback else "")
        for c in application.registered_commands
    }
    names |= {g.name or "" for g in application.registered_groups}
    return {n for n in names if n}


def test_a_checkout_exposes_the_developer_commands() -> None:
    """Running from a source checkout is how this test suite runs, so all 12 are here."""
    from uclone_x.cli.main import app, running_from_source_checkout

    assert running_from_source_checkout() is True
    registered = _registered_command_names(app)
    assert _DEVELOPER_COMMANDS <= registered, sorted(_DEVELOPER_COMMANDS - registered)
    assert _USER_COMMANDS <= registered, sorted(_USER_COMMANDS - registered)


def test_an_installed_build_exposes_only_the_user_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """The three developer commands misbehave outside a checkout, so they are not shipped.

    Measured on a wheel in a clean environment before this split: `ucx setup` wrote this
    project's git hooks into whatever repository the working directory was the root of --
    and that pre-commit refuses any commit not authored by the builder bot, so a user who
    ran it in their own project could no longer commit to it; `ucx test check` raised a
    traceback; `ucx dev task list` exited 0 while printing an internal board URL.
    """
    import uclone_x.cli.main as cli_main

    fresh = typer.Typer(name="ucx")
    monkeypatch.setattr(cli_main, "app", fresh)
    monkeypatch.setattr(cli_main, "running_from_source_checkout", lambda: False)
    # re-register onto the fresh app under the installed-build condition
    for name, sub in (
        ("a2a", cli_main.a2a_app),
        ("eval", cli_main.eval_app),
        ("llm", cli_main.llm_app),
        ("ontology", cli_main.ontology_app),
        ("skill", cli_main.skill_app),
    ):
        fresh.add_typer(sub, name=name)
    cli_main.register_developer_commands()

    registered = _registered_command_names(fresh)
    assert not (_DEVELOPER_COMMANDS & registered), sorted(_DEVELOPER_COMMANDS & registered)
    assert "eval" in registered, "eval refuses with an actionable message and stays shipped"


def test_an_installed_build_says_where_the_developer_commands_are(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command that silently does not exist leaves an older document's reader stuck.

    Killed by: src/uclone_x/cli/main.py :: app.info.epilog = (
    """
    import uclone_x.cli.main as cli_main

    fresh = typer.Typer(name="ucx")
    monkeypatch.setattr(cli_main, "app", fresh)
    monkeypatch.setattr(cli_main, "running_from_source_checkout", lambda: False)
    cli_main.register_developer_commands()

    epilog = fresh.info.epilog or ""
    assert "source checkout" in epilog
    assert "git clone" in epilog
    for name in sorted(_DEVELOPER_COMMANDS):
        assert name in epilog, f"{name} is withheld but not named"


def test_source_checkout_detection_reads_the_project_not_git(tmp_path: Path) -> None:
    """A source tree without git metadata is still a source tree; site-packages is not.

    Detection therefore looks for the `pyproject.toml` that declares this project, three
    levels above the package. Asserted by pointing the module's own path resolution at a
    fake tree, which is the only part of the decision that can be wrong in a way the
    checkout-based test above cannot see.
    """
    from uclone_x.cli.main import running_from_source_checkout

    # The real checkout: found.
    assert running_from_source_checkout() is True

    # A wheel layout: `<prefix>/lib/python3.x/site-packages/uclone_x/cli/main.py` has no
    # project file three levels up.
    fake = tmp_path / "lib" / "python3.12" / "site-packages" / "uclone_x" / "cli"
    fake.mkdir(parents=True)
    (fake / "main.py").write_text("", encoding="utf-8")
    root = fake.resolve().parents[3]
    assert not (root / "pyproject.toml").is_file()


@pytest.mark.parametrize(
    ("body", "expected", "why"),
    [
        ('[project]\nname = "uclone-x"\n', True, "the shipped spelling"),
        ("[project]\nname = 'uclone-x'\n", True, "single quotes are equally valid TOML"),
        ('[project]\nname = "uclone-x-plugin"\n', False, "a different project"),
        ('[project]\nversion = "1"\n', False, "no name at all"),
        ('[tool.poetry]\nname = "uclone-x"\n', False, "not a PEP 621 project table"),
        ("name = = broken\n", False, "unparseable TOML"),
    ],
)
def test_checkout_detection_parses_the_project_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, expected: bool, why: str
) -> None:
    """Detection reads the project table, so a reformat cannot silently disable commands.

    Raised in review of #605: the first implementation matched the exact substring
    `name = "uclone-x"`, so `name = 'uclone-x'` — valid TOML that a formatter could
    produce — would have answered False and deleted `setup`, `test` and `dev` from a
    checkout, with no test on the real file to notice.

    Killed by: src/uclone_x/cli/main.py :: parsed = tomllib.loads(
    """
    import uclone_x.cli.main as cli_main

    fake_pkg = tmp_path / "src" / "uclone_x" / "cli"
    fake_pkg.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(body, encoding="utf-8")
    monkeypatch.setattr(cli_main, "__file__", str(fake_pkg / "main.py"))

    assert cli_main.running_from_source_checkout() is expected, why


def test_checkout_detection_survives_a_pyproject_that_is_not_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A byte sequence that is not UTF-8 must answer False, not crash `ucx` at import.

    Also raised in review: the first implementation caught only `OSError`, so a
    non-UTF-8 `pyproject.toml` three levels above the package raised
    `UnicodeDecodeError` during module import — before any command could run.
    """
    import uclone_x.cli.main as cli_main

    fake_pkg = tmp_path / "src" / "uclone_x" / "cli"
    fake_pkg.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_bytes(b'[project]\nname = "\xff\xfe not utf-8"\n')
    monkeypatch.setattr(cli_main, "__file__", str(fake_pkg / "main.py"))

    assert cli_main.running_from_source_checkout() is False


def test_checkout_detection_answers_for_the_real_repository() -> None:
    """The one case the constructed layouts cannot cover: this repository's own file.

    A parser that works on fixtures and not on the shipped `pyproject.toml` would be
    invisible to every test above.
    """
    from uclone_x.cli.main import running_from_source_checkout

    assert running_from_source_checkout() is True


def _distribution_name(requirement: str) -> str:
    """The importable top-level name a requirement specifier installs, near enough.

    `"typer[all]>=0.12.0"` -> `typer`. Good enough to decide what to block, because
    the packages this repository depends on all import under their own name; a
    dependency that did not would simply not be blocked, which fails safe.
    """
    name = re.split(r"[<>=!~\[; ]", requirement.strip(), maxsplit=1)[0]
    return name.replace("-", "_")


def test_cli_imports_without_the_http_extra() -> None:
    """`ucx --help` must not require a package outside the base and `cli` sets.

    `cli/main.py` imports `a2a_app` and the `ui` group at module scope, and both
    modules used to reach the `http` extra at import time -- `a2a.py` through
    `import uvicorn` and the A2A shell, `ui.py` through `uclone_x.ui`. So a
    `uclone-x[cli]` install -- the extra whose whole purpose is the CLI -- died
    on `ModuleNotFoundError: No module named 'uvicorn'` before typer ran, and
    every command was unreachable, `--help` included.

    Not a released defect: 0.1.0 declares no extras at all and carries fastapi
    and uvicorn as base dependencies, so its CLI runs. The bug arrived on `main`
    with the extras split and is caught here before it ships.

    The blocked set is derived from `pyproject.toml` rather than hardcoded: the
    failure is "a module-scope import of something outside the base and `cli`
    sets", and a hardcoded `{"uvicorn", "fastapi"}` would let the same mistake
    through for opentelemetry, tree-sitter or an LLM SDK.

    Run in a subprocess because this process has every extra installed, and
    nothing else can express their absence.

    Killed by: src/uclone_x/cli/commands/a2a.py :: from uclone_x.shells.a2a_server import A2AServer
    """
    import tomllib

    import uclone_x

    src_root = Path(uclone_x.__file__).resolve().parents[1]
    repo_root = src_root.parent
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = cast("dict[str, Any]", pyproject["project"])
    extras = cast("dict[str, list[str]]", project["optional-dependencies"])

    allowed = {_distribution_name(spec) for spec in cast("list[str]", project["dependencies"])}
    allowed |= {_distribution_name(spec) for spec in extras["cli"]}
    blocked = sorted(
        {
            _distribution_name(spec)
            for name, specs in extras.items()
            if name not in {"cli", "dev", "all"}
            for spec in specs
        }
        - allowed
    )
    assert "uvicorn" in blocked, blocked

    script = textwrap.dedent(f"""
        import sys

        BLOCKED = {{module for module in {blocked!r}}}

        class Blocked:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0].replace("-", "_") in BLOCKED:
                    raise ImportError(f"blocked: {{name}}")
                return None

        sys.meta_path.insert(0, Blocked())
        for mod in [m for m in sys.modules if m.split(".")[0].replace("-", "_") in BLOCKED]:
            del sys.modules[mod]

        from uclone_x.cli.main import app

        assert app is not None
        print("imported")
    """)

    env = {**os.environ, "PYTHONPATH": str(src_root)}

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout
