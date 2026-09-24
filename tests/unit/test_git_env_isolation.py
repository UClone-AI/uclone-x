"""The suite must not stage into the repository that invoked it (#709).

`git` exports `GIT_DIR` to its hooks whenever the repository is not the plain `.git` beside
the current directory, which is always true of a linked worktree. A hook that runs
`./ucx test check` (`pre-push` did until #966) passes that on, so the whole suite inherits an
absolute `GIT_DIR` naming the checkout being pushed. `GIT_DIR` outranks `git -C <dir>` —
`-C` changes only the working directory — so a test that runs `git -C <tmp_path> add -A`
stages the scratch tree into the pushing worktree's index and collapses it to one path. The
push then fails on the damage rather than on the branch, and nothing reports the damage.

These tests run a **child pytest** against a freshly written test file that shells out to
`git` the naive way, with `GIT_DIR` set exactly as the hook sets it, and read the index of
a scratch "pushing" repository before and after. The child is given
`tests/conftest.py` as a plugin, so what is under test is the repo-wide scrub itself —
not a change to any particular call site. A fix applied call site by call site would leave
this test red the moment somebody adds the 1,001st `subprocess.run(["git", ...])`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The identity the pushing checkout must still carry when the child run is over.
PUSHING_EMAIL = "pushing-checkout@example.invalid"

# A test module that does exactly what the two culprit modules did: build a scratch
# repository and stage it, inheriting the ambient environment.
#
# It stages **twice**, once at import and once inside a test, because only the import-time
# scrub in `tests/conftest.py` can protect the first: module-level and collection-time
# `git` calls run before any fixture does. Measured: neutralising the import-time scrub
# fails this case; neutralising the autouse fixture does not, which is what
# `test_a_test_that_leaks_git_dir_cannot_damage_the_next_one` below exists to cover.
#
# It also rewrites `user.email`, the second damage class: under `GIT_DIR` that lands in the
# pushing checkout's config, and in this repository every worktree shares one.
_NAIVE_TEST_MODULE = """
import subprocess
from pathlib import Path

_AT_IMPORT = Path(__file__).parent / "scratch-import"
_AT_IMPORT.mkdir(exist_ok=True)
subprocess.run(["git", "init", "-q", str(_AT_IMPORT)], check=False)
subprocess.run(["git", "-C", str(_AT_IMPORT), "config", "user.email", "t@t"], check=False)
(_AT_IMPORT / "README.md").write_text("only file\\n")
subprocess.run(["git", "-C", str(_AT_IMPORT), "add", "-A"], check=False)


def test_stages_a_scratch_tree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=False)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=False)
    (tmp_path / "README.md").write_text("only file\\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=False)
"""


# The other direction: nothing in the environment at startup, and a test that puts
# `GIT_DIR` back by writing `os.environ` directly — no `monkeypatch`, so nothing undoes it.
# The import-time scrub cannot help here; only the autouse fixture re-clears it before the
# next test. This is the case that pins the fixture, which the import-time scrub otherwise
# makes unkillable.
_LEAKY_TEST_MODULE = """
import os
import subprocess


def test_a_leaks_git_dir():
    os.environ["GIT_DIR"] = os.environ["LEAKED_GIT_DIR"]


def test_b_stages_a_scratch_tree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=False)
    (tmp_path / "README.md").write_text("only file\\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=False)
"""


def _make_pushing_repo(root: Path) -> Path:
    """A stand-in for the worktree a `git push` runs its hook from."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (root / name).write_text(f"# {name}\n")
    env = _clean_git_env()
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", PUSHING_EMAIL], check=True, env=env
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "pushing"], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "i"], check=True, env=env, capture_output=True
    )
    return root


def _clean_git_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _configured_email(repo: Path) -> str:
    """`git -C <tmp> config user.email …` under GIT_DIR rewrites the *pushing* config.

    The second damage class, unreported until #709 was fixed: in this repository every
    worktree shares one `.git/config`, so a leaked `user.email` re-attributes every agent
    commit made afterwards — the #99 / #105 / #122 collision, from a test.
    """
    out = subprocess.run(
        ["git", "-C", str(repo), "config", "--get", "user.email"],
        check=False,
        capture_output=True,
        text=True,
        env=_clean_git_env(),
    )
    return out.stdout.strip()


def _tracked(repo: Path) -> int:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
        env=_clean_git_env(),
    )
    return len(out.stdout.split())


def _run_child_pytest(
    *,
    test_file: Path,
    git_dir: Path | None,
    load_repo_conftest: bool,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one test file in a child pytest, with `GIT_DIR` set as a git hook sets it.

    `git_dir=None` starts the child with no `GIT_DIR` at all, which is how the leak case
    below reaches the autouse fixture rather than the import-time scrub.
    """
    env = _clean_git_env()
    if git_dir is not None:
        env["GIT_DIR"] = str(git_dir)
    env.update(extra_env or {})
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # The child's rootdir is the temporary directory, so it reads no ini file and inherits
    # neither `addopts` nor the invocation guard's assumptions.
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if load_repo_conftest:
        argv += ["-p", "tests.conftest"]
    argv.append(str(test_file))
    return subprocess.run(argv, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))


def test_the_repo_conftest_stops_a_test_staging_into_the_pushing_checkout(tmp_path: Path) -> None:
    """The index of the repository that invoked the suite is the same size afterwards.

    Not "no exception raised": the naive `git add -A` succeeds either way. The only
    observable difference is whose index it landed in, so the assertion is a count of
    tracked files in the pushing repository, positive before and unchanged after.

    Killed by: tests/conftest.py :: _SCRUBBED_GIT_ENV
    """
    pushing = _make_pushing_repo(tmp_path / "pushing")
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "test_naive_git.py").write_text(_NAIVE_TEST_MODULE)

    before = _tracked(pushing)
    assert before == 3, "the pushing repository must start with something to lose"

    result = _run_child_pytest(
        test_file=workdir / "test_naive_git.py",
        git_dir=pushing / ".git",
        load_repo_conftest=True,
    )

    after = _tracked(pushing)
    # Count the tests, never the exit code: a filter that selects nothing exits 5.
    assert "1 passed" in result.stdout, f"child pytest did not run the test:\n{result.stdout}"
    assert after == before, (
        f"the child staged into the pushing repository's index: {before} tracked files "
        f"before, {after} after. GIT_DIR survived into the test environment.\n"
        f"{result.stdout}"
    )
    assert after > 0
    assert _configured_email(pushing) == PUSHING_EMAIL, (
        "the child rewrote the pushing checkout's commit identity through GIT_DIR"
    )


def test_without_the_conftest_the_same_child_does_wipe_the_index(tmp_path: Path) -> None:
    """Positive control: the harness above can see the damage it claims is absent.

    Identical child run with the repo-wide conftest withheld. If this ever stops wiping
    the index, the test above has stopped proving anything and both need rewriting —
    `GIT_DIR` would no longer be able to reach a test's subprocesses at all.
    """
    pushing = _make_pushing_repo(tmp_path / "pushing")
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "test_naive_git.py").write_text(_NAIVE_TEST_MODULE)

    before = _tracked(pushing)
    assert before == 3

    result = _run_child_pytest(
        test_file=workdir / "test_naive_git.py",
        git_dir=pushing / ".git",
        load_repo_conftest=False,
    )

    assert "1 passed" in result.stdout, f"child pytest did not run the test:\n{result.stdout}"
    assert _tracked(pushing) == 1, (
        "the unprotected child was expected to collapse the pushing index to the one file "
        "its scratch tree holds; it did not, so the guarded case above proves nothing"
    )
    assert _configured_email(pushing) == "t@t", (
        "the unprotected child was expected to rewrite the pushing checkout's commit "
        "identity too; it did not, so that half of the guarded assertion proves nothing"
    )


def test_a_test_that_leaks_git_dir_cannot_damage_the_next_one(tmp_path: Path) -> None:
    """One test writing `os.environ["GIT_DIR"]` must not follow the next one into `git`.

    The child starts with no `GIT_DIR`, so the import-time scrub has nothing to remove and
    this case reaches the autouse fixture alone. Without it, test B's `git add -A` lands in
    the pushing checkout exactly as the hook-inherited variable did.

    Killed by: tests/conftest.py :: _isolate_git_environment
    """
    pushing = _make_pushing_repo(tmp_path / "pushing")
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "test_leaky_git.py").write_text(_LEAKY_TEST_MODULE)

    before = _tracked(pushing)
    assert before == 3, "the pushing repository must start with something to lose"

    result = _run_child_pytest(
        test_file=workdir / "test_leaky_git.py",
        git_dir=None,
        load_repo_conftest=True,
        extra_env={"LEAKED_GIT_DIR": str(pushing / ".git")},
    )

    after = _tracked(pushing)
    assert "2 passed" in result.stdout, f"child pytest did not run both tests:\n{result.stdout}"
    assert after == before, (
        f"a leaked GIT_DIR reached the following test's subprocess: {before} tracked files "
        f"before, {after} after.\n{result.stdout}"
    )
    assert after > 0


def test_without_the_conftest_the_leak_does_reach_the_next_test(tmp_path: Path) -> None:
    """Positive control for the leak case, same shape as the one above."""
    pushing = _make_pushing_repo(tmp_path / "pushing")
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "test_leaky_git.py").write_text(_LEAKY_TEST_MODULE)

    assert _tracked(pushing) == 3

    result = _run_child_pytest(
        test_file=workdir / "test_leaky_git.py",
        git_dir=None,
        load_repo_conftest=False,
        extra_env={"LEAKED_GIT_DIR": str(pushing / ".git")},
    )

    assert "2 passed" in result.stdout, f"child pytest did not run both tests:\n{result.stdout}"
    assert _tracked(pushing) == 1, (
        "the unprotected leak was expected to collapse the pushing index; it did not, so "
        "the guarded case above proves nothing"
    )
