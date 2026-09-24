"""Unit tests for the child-process import-provenance guard (#679, #718).

The defect these pin is that the shared venv's editable install can name a tree that is
not the one under test — a deleted worktree, or the bare repository root's stale untracked
checkout — and that **nothing noticed**, because `pyproject.toml`'s
`pythonpath = [".", "src"]` applies in-process only. So the tests
below are split the way the module is:

* the decision function is exercised with every combination of probe result, in-process,
  because a branch per Python environment would cost a venv per case;
* the probe itself is exercised **for real**, against throwaway environments built in
  `tmp_path`, because a mocked probe is exactly the instrument that failed here — a guard
  that passes because its own probe stopped working is this issue reproduced inside its
  own fix.

The shared environment is never mutated. Each spawning test builds its own venv and writes
its own `.pth`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from uclone_x.cli.environment_provenance import (
    BLOCKING_STATUSES,
    ProvenanceStatus,
    VenvExceptionCheck,
    check_environment_provenance,
    describe_environment_provenance,
    find_declared_venv_exception,
    find_worktree_local_venv,
    probe_module_origin,
    repo_subprocess_env,
)

#: Stands in for "wherever `root`'s own `src` is" in `_describe` below, so each test
#: names only the one input it is varying.
_INSIDE = "<the tree under test>"


def _describe(
    root: Path,
    *,
    tree_origin: str | None = _INSIDE,
    tree_error: str = "",
    ambient_origin: str | None = _INSIDE,
    ambient_error: str = "",
    worktree_venv: Path | None = None,
    venv_exception: VenvExceptionCheck | None = None,
) -> tuple[ProvenanceStatus, list[str]]:
    """`describe_environment_provenance` with the all-healthy input as the default.

    Every test below therefore reads as "healthy, except for this one thing", which is
    what makes the severity of that one thing the assertion rather than a coincidence.
    """
    inside = str(root / "src" / "uclone_x" / "__init__.py")
    return describe_environment_provenance(
        root,
        tree_origin=inside if tree_origin == _INSIDE else tree_origin,
        tree_error=tree_error,
        ambient_origin=inside if ambient_origin == _INSIDE else ambient_origin,
        ambient_error=ambient_error,
        worktree_venv=worktree_venv,
        venv_exception=venv_exception,
    )


#: A branch that has declared AGENTS.md:82's Permitted Exception, and one that has not.
#: Written out rather than built by `find_declared_venv_exception` so that the severity
#: tests below vary one input and never a `git` invocation.
_DECLARED = VenvExceptionCheck(declared=True, detail="uv.lock differs from refs/heads/main")
_NOT_DECLARED = VenvExceptionCheck(declared=False, detail="uv.lock is identical to refs/heads/main")


# --------------------------------------------------------------------------------------
# repo_subprocess_env — the definition of "the environment a child needs"
# --------------------------------------------------------------------------------------


def test_repo_subprocess_env_puts_the_tree_under_test_on_pythonpath(tmp_path: Path) -> None:
    """The tree's own `src` is on PYTHONPATH, absolute and resolved.

    Killed by: src/uclone_x/cli/environment_provenance.py :: env["PYTHONPATH"] = f"{src}{os.pathsep}{inherited}" if inherited else src
    """
    env = repo_subprocess_env(tmp_path, base={})
    assert env["PYTHONPATH"] == str((tmp_path / "src").resolve())


def test_repo_subprocess_env_prepends_rather_than_replacing_an_inherited_path(
    tmp_path: Path,
) -> None:
    """A caller's own PYTHONPATH survives, and the tree under test still wins.

    Replacing it would break any caller that had put something on the path deliberately;
    appending would let a stale entry shadow the tree, which is the defect itself.
    """
    env = repo_subprocess_env(tmp_path, base={"PYTHONPATH": "/somewhere/else"})
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries == [str((tmp_path / "src").resolve()), "/somewhere/else"]


# --------------------------------------------------------------------------------------
# find_worktree_local_venv — a `.venv` that should not exist
# --------------------------------------------------------------------------------------


def test_a_venv_in_a_linked_worktree_is_reported(tmp_path: Path) -> None:
    """A linked worktree has `.git` as a file; a `.venv` beside it is never intended.

    Killed by: src/uclone_x/cli/environment_provenance.py :: return candidate if candidate.is_dir() else None
    """
    (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/w\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    assert find_worktree_local_venv(tmp_path) == tmp_path / ".venv"


def test_a_venv_at_the_repository_root_is_not_reported(tmp_path: Path) -> None:
    """`.git` as a directory means this is the common dir, where the shared venv belongs.

    Without this discrimination the guard would refuse in the one place the venv is
    correct, which is the repository root itself.
    """
    (tmp_path / ".git").mkdir()
    (tmp_path / ".venv").mkdir()
    assert find_worktree_local_venv(tmp_path) is None


def test_a_worktree_without_its_own_venv_is_reported_clean(tmp_path: Path) -> None:
    """The inherited-venv arrangement AGENTS.md describes is the passing case."""
    (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/w\n", encoding="utf-8")
    assert find_worktree_local_venv(tmp_path) is None


# --------------------------------------------------------------------------------------
# find_declared_venv_exception — residue or AGENTS.md:82's Permitted Exception (#968)
#
# Driven against **real** repositories built in `tmp_path`, never a mocked `git`. The
# whole content of this check is what `git` answers for a particular pair of revisions,
# so a test that stubbed `git` would be asserting the author's belief about merge bases
# rather than git's behaviour — and the merge-base choice is the one decision here that a
# reasonable reader would get wrong.
# --------------------------------------------------------------------------------------


def _run_git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _repo_on_main(tmp_path: Path) -> Path:
    """A real repository on `main`, carrying a committed `uv.lock`."""
    root = tmp_path / "repo"
    root.mkdir()
    _run_git(root, "init", "--initial-branch=main")
    _run_git(root, "config", "user.email", "builder@example.invalid")
    _run_git(root, "config", "user.name", "builder")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _run_git(root, "add", "uv.lock")
    _run_git(root, "commit", "--no-verify", "-m", "main: lock")
    return root


def test_a_branch_whose_lock_differs_from_main_declares_the_exception(tmp_path: Path) -> None:
    """The signal #968 decided on: this branch's `uv.lock` is not `main`'s.

    Killed by: src/uclone_x/cli/environment_provenance.py :: diff_code, changed = _git(root, "diff", "--name-only", base, "--", DEPENDENCY_LOCKFILE)
    Becomes: diff_code, changed = _git(root, "diff", "--name-only", base, "--", "no-such-path")
    """
    root = _repo_on_main(tmp_path)
    _run_git(root, "checkout", "-b", "task/example")
    (root / "uv.lock").write_text("version = 1\n# branch needs its own resolution\n", "utf-8")
    _run_git(root, "commit", "--no-verify", "-am", "branch: relock")

    result = find_declared_venv_exception(root)

    assert result.declared is True
    assert "uv.lock" in result.detail


def test_an_uncommitted_lock_change_declares_the_exception_too(tmp_path: Path) -> None:
    """The comparison reaches the working tree, not only committed history.

    Deliberate rather than incidental: the builder who needs a branch-local environment is
    mid-relock by definition, and a check that only saw commits would refuse them for the
    duration of the work it exists to permit.
    """
    root = _repo_on_main(tmp_path)
    _run_git(root, "checkout", "-b", "task/example")
    (root / "uv.lock").write_text("version = 1\n# not committed yet\n", encoding="utf-8")

    assert find_declared_venv_exception(root).declared is True


def test_a_branch_whose_lock_matches_main_declares_nothing(tmp_path: Path) -> None:
    """The negative control for the two tests above, and the common case by far.

    No `Killed by:` line: this branch of the function is shared with the merge-base test
    below, which owns the mutation. A declaration here would name an edit that kills two
    tests, and a `Killed by:` line that is true of more nodes than it names is the kind of
    claim #741 was filed about.
    """
    root = _repo_on_main(tmp_path)
    _run_git(root, "checkout", "-b", "task/example")
    (root / "README.md").write_text("no dependency change here\n", encoding="utf-8")
    _run_git(root, "add", "README.md")
    _run_git(root, "commit", "--no-verify", "-m", "branch: docs")

    result = find_declared_venv_exception(root)

    assert result.declared is False
    assert "identical" in result.detail


def test_a_lock_change_main_made_after_the_fork_is_not_this_branchs_declaration(
    tmp_path: Path,
) -> None:
    """The comparison is against the merge base, not against `main`'s tip.

    A two-dot `git diff main` also reports every `uv.lock` change `main` acquired *after*
    this branch forked. Under that rule a branch would come to declare an exception it
    never made, simply by ageing — a false allow that arrives on its own, with no builder
    action at all, in a repository where every worktree is cut from `origin/main` and
    `main` moves several times a day.

    Killed by: src/uclone_x/cli/environment_provenance.py :: base_code, base = _git(root, "merge-base", main_ref, "HEAD")
    Becomes: base_code, base = _git(root, "rev-parse", main_ref)
    """
    root = _repo_on_main(tmp_path)
    _run_git(root, "checkout", "-b", "task/example")
    _run_git(root, "checkout", "main")
    (root / "uv.lock").write_text("version = 1\n# main relocked since the fork\n", "utf-8")
    _run_git(root, "commit", "--no-verify", "-am", "main: relock")
    _run_git(root, "checkout", "task/example")

    assert find_declared_venv_exception(root).declared is False


def test_no_main_to_compare_against_fails_closed_and_says_so(tmp_path: Path) -> None:
    """Unable to tell is read as "not declared" — the pre-#968 answer — and is named.

    The direction of this default is the whole safety argument for the change. A false
    refusal costs a builder one actionable message; a false allow is a stray `.venv`
    poisoning every import behind a gate that passed (#679).

    Killed by: src/uclone_x/cli/environment_provenance.py :: detail=f"no main to compare against (tried {', '.join(_MAIN_REFS)})",
    Becomes: detail="",
    """
    root = tmp_path / "orphan"
    root.mkdir()
    _run_git(root, "init", "--initial-branch=task/example")
    _run_git(root, "config", "user.email", "builder@example.invalid")
    _run_git(root, "config", "user.name", "builder")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _run_git(root, "add", "uv.lock")
    _run_git(root, "commit", "--no-verify", "-m", "no main here")

    result = find_declared_venv_exception(root)

    assert result.declared is False
    assert "no main" in result.detail


# --------------------------------------------------------------------------------------
# describe_environment_provenance — severity per condition
# --------------------------------------------------------------------------------------


def test_a_child_resolving_under_the_tree_under_test_is_the_passing_case(tmp_path: Path) -> None:
    """Everything healthy reports `ok` and names the directory it checked."""
    status, lines = _describe(tmp_path)
    assert status == "ok"
    assert status not in BLOCKING_STATUSES
    assert str(tmp_path / "src") in " ".join(lines)


def test_a_worktree_local_venv_blocks_and_outranks_every_other_reading(tmp_path: Path) -> None:
    """The `.venv` is reported even when both probes look healthy, and it blocks.

    Ordering is the substance here, not an incidental: a local venv explains every other
    reading on the list, so surfacing a downstream symptom above it sends the reader after
    the wrong thing — which is what 258 phantom Pyright errors did to one builder (#679).

    Killed by: src/uclone_x/cli/environment_provenance.py :: if worktree_venv is not None:
    """
    status, lines = _describe(tmp_path, worktree_venv=tmp_path / ".venv")
    assert status == "worktree-venv"
    assert status in BLOCKING_STATUSES
    joined = " ".join(lines)
    assert str(tmp_path / ".venv") in joined
    # It must not tell the reader to `uv sync` their way out: that is the other half of
    # this very defect.
    assert "Do NOT `uv sync`" in joined


def test_an_undeclared_worktree_venv_is_refused_as_residue_and_says_why(tmp_path: Path) -> None:
    """The refusal names *which* of the two situations the builder is in, and its evidence.

    Half the defect #968 records was the wording, not the logic: the stage refused every
    worktree-local `.venv` while telling the reader "a local one is never intended", which
    contradicts AGENTS.md:82's Permitted Exception in so many words. A builder who really
    did need different dependency versions was told their arrangement does not exist. The
    message must now name the exception, say that this branch is not it, and carry the
    reason — otherwise the next reader cannot tell a branch that changed no dependencies
    from a checkout whose `origin/main` was never fetched.

    The declaration below restores the old wording rather than inverting the branch. The
    inversion also kills this node, but it kills four others with it — including two that
    die by `AttributeError` — so it is evidence about the branch and not about the
    sentence. Reverting the sentence is the regression this test exists to catch, and it
    kills this node alone.

    Killed by: src/uclone_x/cli/environment_provenance.py :: "A local one is permitted in exactly one case — AGENTS.md:82's Permitted "
    Becomes: "A local one is never intended. "
    """
    status, lines = _describe(
        tmp_path, worktree_venv=tmp_path / ".venv", venv_exception=_NOT_DECLARED
    )
    assert status == "worktree-venv"
    assert status in BLOCKING_STATUSES
    joined = " ".join(lines)
    assert "never intended" not in joined, "the wording AGENTS.md:82 contradicts (#968)"
    assert "Permitted Exception" in joined
    assert _NOT_DECLARED.detail in joined, "the refusal must carry its own evidence"


def test_a_worktree_venv_this_branch_declared_is_permitted_and_does_not_block(
    tmp_path: Path,
) -> None:
    """AGENTS.md:82's Permitted Exception, which stage 1b refused until #968.

    The decision recorded on #968 is Option (A): the gate honours a declared exception
    when the branch's dependency lock differs from `main`. Honouring it means a
    non-blocking status — a refusal with friendlier prose would leave the contradiction
    exactly where it was.

    Killed by: src/uclone_x/cli/environment_provenance.py :: return "worktree-venv-declared", [*notes, resolved]
    Becomes: return "worktree-venv", [*notes, resolved]
    """
    status, lines = _describe(tmp_path, worktree_venv=tmp_path / ".venv", venv_exception=_DECLARED)
    assert status == "worktree-venv-declared"
    assert status not in BLOCKING_STATUSES
    joined = " ".join(lines)
    assert "AGENTS.md:82" in joined
    assert _DECLARED.detail in joined
    assert str(tmp_path / ".venv") in joined


def test_a_declared_exception_warns_without_masking_an_unimportable_tree(tmp_path: Path) -> None:
    """The declared note is *prepended*, not returned instead of the remaining checks.

    This is the false allow the fix could easily have introduced. The refusing branch
    returns from the top of the function because a local venv explains every reading
    below it — but a non-blocking status returned from that same position would swallow a
    genuine `tree-unimportable` and turn a refusal into a pass, which is the silent green
    this whole stage exists to prevent (P6).

    Killed by: src/uclone_x/cli/environment_provenance.py :: notes = _declared_venv_notes(worktree_venv, venv_exception)
    Becomes: return "worktree-venv-declared", _declared_venv_notes(worktree_venv, venv_exception)
    """
    status, lines = _describe(
        tmp_path,
        tree_origin=None,
        tree_error="No module named 'uclone_x'",
        worktree_venv=tmp_path / ".venv",
        venv_exception=_DECLARED,
    )
    assert status == "tree-unimportable"
    assert status in BLOCKING_STATUSES
    joined = " ".join(lines)
    assert "AGENTS.md:82" in joined, "the declared venv is still the most explanatory line"
    assert "No module named 'uclone_x'" in joined


def test_a_worktree_venv_with_no_exception_check_at_all_is_refused(tmp_path: Path) -> None:
    """`venv_exception=None` means "nobody asked", and that is read as "not declared".

    The default matters because it is what any caller that has not been updated gets. A
    default of "permitted" would make the fix arrive by omission in exactly the callers
    that never considered it.
    """
    status, lines = _describe(tmp_path, worktree_venv=tmp_path / ".venv", venv_exception=None)
    assert status == "worktree-venv"
    assert "not checked" in " ".join(lines)


def test_a_child_that_cannot_import_the_tree_under_test_blocks(tmp_path: Path) -> None:
    """No origin from the helped child means nothing downstream is about this tree.

    Killed by: src/uclone_x/cli/environment_provenance.py :: if tree_origin is None:
    """
    status, lines = _describe(
        tmp_path, tree_origin=None, tree_error="ModuleNotFoundError: uclone_x"
    )
    assert status == "tree-unimportable"
    assert status in BLOCKING_STATUSES
    assert "ModuleNotFoundError: uclone_x" in " ".join(lines)


def test_a_child_shadowed_away_from_the_tree_under_test_blocks(tmp_path: Path) -> None:
    """An import that succeeds from the wrong tree is the quiet form and still blocks.

    This is the reading the whole issue turns on: the child answered, so nothing raised,
    but it answered about a different tree.

    Killed by: src/uclone_x/cli/environment_provenance.py :: if not _is_inside(tree_origin, src):
    """
    foreign = "/some/other/checkout/src/uclone_x/__init__.py"
    status, lines = _describe(tmp_path, tree_origin=foreign)
    assert status == "tree-unimportable"
    assert status in BLOCKING_STATUSES
    assert foreign in " ".join(lines)


def test_an_unimportable_shared_install_blocks_and_names_the_recovery(tmp_path: Path) -> None:
    """The original #679: the editable path named a worktree that had been deleted.

    Blocking is correct here because the condition is broken for every checkout at once
    and is fixable in one command from the repository root.

    Killed by: src/uclone_x/cli/environment_provenance.py :: if ambient_origin is None:
    """
    status, lines = _describe(
        tmp_path, ambient_origin=None, ambient_error="ModuleNotFoundError: uclone_x"
    )
    assert status == "ambient-unimportable"
    assert status in BLOCKING_STATUSES
    assert "uv sync --all-extras" in " ".join(lines)


def test_a_shared_install_pointing_elsewhere_warns_and_does_not_block(tmp_path: Path) -> None:
    """One venv holds one editable path, so this cannot be true in every worktree at once.

    This is the deliberate non-refusal, and it is the one decision in this module most
    likely to be "corrected" by a later reader into a refusal. Doing so would make the
    gate unrunnable in every worktree but whichever one last ran `uv sync` — so the test
    pins the severity, not just the message.

    Killed by: src/uclone_x/cli/environment_provenance.py :: if not _is_inside(ambient_origin, src):
    """
    elsewhere = "/Users/somebody/uclone-git/other-checkout/src/uclone_x/__init__.py"
    status, lines = _describe(tmp_path, ambient_origin=elsewhere)
    assert status == "ambient-foreign"
    assert status not in BLOCKING_STATUSES
    joined = " ".join(lines)
    assert elsewhere in joined
    assert "#718" in joined


def test_every_status_is_classified_as_blocking_or_not() -> None:
    """A new status cannot be added without deciding whether it stops the gate.

    Defaulting an unclassified status to non-blocking is how a check stops checking, so
    the enumeration and the blocking set are pinned against each other.

    `worktree-venv-declared` joined the non-blocking side under #968 — deliberately, and
    this node is where that was decided rather than noticed: adding the status without
    editing this line turns the suite red.
    """
    declared = set(ProvenanceStatus.__args__)  # type: ignore[attr-defined]
    assert BLOCKING_STATUSES <= declared
    assert declared - BLOCKING_STATUSES == {"ok", "ambient-foreign", "worktree-venv-declared"}


# --------------------------------------------------------------------------------------
# The probe itself, against real throwaway environments
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def throwaway_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A venv of our own, with no `uclone_x` in it and no access to the shared one.

    Built rather than mocked: the thing under test is what a real interpreter does with a
    real `.pth`, which is the step every previous investigation had to take by hand.
    """
    root = tmp_path_factory.mktemp("throwaway-venv")
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root / "venv")],
        check=True,
        capture_output=True,
    )
    python = root / "venv" / "bin" / "python"
    if not python.exists():  # pragma: no cover - Windows layout, not a supported host
        python = root / "venv" / "Scripts" / "python.exe"
    return python


def _site_packages(python: Path) -> Path:
    out = subprocess.run(
        [str(python), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def _shadow_on_pythonpath(tmp_path: Path) -> dict[str, str]:
    """An env whose PYTHONPATH names a tree that would win if the probe were not isolated.

    The two decoy tests below assert what an install *alone* resolves. Passing this makes
    that assertion independent of the ambient `PYTHONPATH`: under `-I` the shadow is
    invisible and the decoy answers, and without `-I` the shadow answers and the test
    fails — the same outcome whether the suite was started by `./ucx` (which exports
    `PYTHONPATH`) or by a bare `pytest` (which does not). Before this, those two tests
    were sensitive to dropping `-I` only when the runner happened to set `PYTHONPATH`,
    which is environment-dependent measurement inside the guard against exactly that.
    """
    shadow_src = tmp_path / "shadow" / "src"
    (shadow_src / "uclone_x").mkdir(parents=True)
    (shadow_src / "uclone_x" / "__init__.py").write_text("", encoding="utf-8")
    return {**os.environ, "PYTHONPATH": str(shadow_src)}


def test_the_probe_reports_the_tree_an_editable_install_actually_names(
    throwaway_python: Path, tmp_path: Path
) -> None:
    """The positive control: point an install at a decoy tree and watch the probe say so.

    Without this the guard could pass forever because its probe silently stopped
    resolving anything — the same shape as the defect it exists to catch. The decoy is a
    directory this repository has never heard of, so a probe that reported the real tree
    here would be reading something other than the environment it was given.

    Killed by: src/uclone_x/cli/environment_provenance.py :: argv.append("-I")
    """
    env = _shadow_on_pythonpath(tmp_path)
    decoy_src = tmp_path / "decoy" / "src"
    (decoy_src / "uclone_x").mkdir(parents=True)
    (decoy_src / "uclone_x" / "__init__.py").write_text("", encoding="utf-8")
    pth = _site_packages(throwaway_python) / "_decoy_editable.pth"
    pth.write_text(f"{decoy_src}\n", encoding="utf-8")
    try:
        origin, error = probe_module_origin(
            python_executable=str(throwaway_python), env=env, isolated=True
        )
    finally:
        pth.unlink()

    assert error == ""
    assert origin is not None
    assert Path(origin) == decoy_src / "uclone_x" / "__init__.py"

    # And the decision function turns exactly that reading into the warning, for a tree
    # that is not the decoy.
    status, _ = _describe(tmp_path, ambient_origin=origin)
    assert status == "ambient-foreign"


def test_the_probe_reports_a_broken_install_rather_than_an_empty_answer(
    throwaway_python: Path, tmp_path: Path
) -> None:
    """An editable path naming a directory that is gone — the original #679 state.

    The probe must return the child's diagnosis, not `None` with an empty string: a
    failure reported as silence is the one this issue spent hours on.

    Killed by: src/uclone_x/cli/environment_provenance.py :: argv.append("-I")
    """
    env = _shadow_on_pythonpath(tmp_path)
    pth = _site_packages(throwaway_python) / "_decoy_editable.pth"
    pth.write_text(f"{tmp_path / 'deleted-worktree' / 'src'}\n", encoding="utf-8")
    try:
        origin, error = probe_module_origin(
            python_executable=str(throwaway_python), env=env, isolated=True
        )
    finally:
        pth.unlink()

    assert origin is None
    assert "ModuleNotFoundError" in error

    status, _ = _describe(tmp_path, ambient_origin=origin, ambient_error=error)
    assert status == "ambient-unimportable"


def test_the_probe_ignores_pythonpath_when_isolated(throwaway_python: Path, tmp_path: Path) -> None:
    """`-I` measures the install and nothing else.

    If it honoured PYTHONPATH, the ambient reading and the helped reading would always
    agree and the warning could never fire — the guard would be decorative.

    Killed by: src/uclone_x/cli/environment_provenance.py :: argv.append("-I")
    """
    decoy_src = tmp_path / "onpath" / "src"
    (decoy_src / "uclone_x").mkdir(parents=True)
    (decoy_src / "uclone_x" / "__init__.py").write_text("", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(decoy_src)}

    isolated_origin, _ = probe_module_origin(
        python_executable=str(throwaway_python), env=env, isolated=True
    )
    helped_origin, helped_error = probe_module_origin(
        python_executable=str(throwaway_python), env=env, isolated=False
    )

    assert isolated_origin is None, "isolated probe must not see PYTHONPATH"
    assert helped_error == ""
    assert helped_origin == str(decoy_src / "uclone_x" / "__init__.py")


def test_the_probe_reports_a_missing_interpreter_instead_of_raising() -> None:
    """A probe that cannot run is a failure the gate reports, not a traceback."""
    origin, error = probe_module_origin(
        python_executable=str(Path(__file__).parent / "no-such-interpreter")
    )
    assert origin is None
    assert error != ""


# --------------------------------------------------------------------------------------
# The live environment this suite is running in
# --------------------------------------------------------------------------------------


def test_this_suite_is_running_in_an_environment_fit_to_be_measured() -> None:
    """The guard applied to the real tree, so a direct `pytest` sees it too.

    `./ucx test check` runs this check as gate stage 1b, but a direct `pytest` does not go
    through the gate — and a direct `pytest` is precisely how the twelve-failure floor was
    observed. Without this node, the environment defect is invisible to exactly the
    invocation that suffers from it.

    A warning-level reading (`ambient-foreign`) passes here by design: one shared venv
    holds one editable path and cannot name every worktree at once.
    """
    root = Path(__file__).resolve().parents[2]
    status, lines = check_environment_provenance(root)
    assert status not in BLOCKING_STATUSES, "\n".join(lines)
