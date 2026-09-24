"""Which `uclone_x` a **child process** gets, and whether it is the tree being tested.

Every check in this repository runs in the tree a builder is working in, but the Python
environment those checks are measured in is **shared**: one `.venv` at the repository
common dir, inherited by every worktree (AGENTS.md §3). That venv holds one editable
install, and an editable install names exactly one `src` directory. With N worktrees and
one `.pth`, the installed path can match at most one of them — so "the editable install
points at this worktree" is **not** an invariant this repository can hold, and a gate that
refused whenever it did not hold would refuse in every worktree but one. That is the
single most important thing to know before changing the severities below.

What *is* invariant is what a child process gets when the repository spawns one on
purpose: `repo_subprocess_env` puts `<root>/src` on `PYTHONPATH`, and the child must then
resolve `uclone_x` inside the tree under test. That is satisfiable in every worktree at
once, and it is exactly the property whose absence produced three irreconcilable bug
reports on one commit (#679, #718).

The reason none of this was visible: `pyproject.toml` sets `pythonpath = [".", "src"]`,
which applies to the pytest process and **not** to anything it spawns. So every in-process
import resolved to the tree under test and the gate stayed green, while subprocess imports
resolved to whatever the shared install happened to name. The in-process assertion that
would be the natural one to write is precisely the one that cannot see the problem, which
is why both probes here spawn.

Three symptoms, one mechanism, recorded so the next reader does not re-derive them:

* `.pth` naming a worktree that had since been deleted — `import uclone_x` failed in any
  subprocess, and twelve tests in `tests/unit/test_kernel_dependency_footprint.py` failed
  under a direct `pytest` while `./ucx test check` was green (#679);
* `.pth` repointed at the bare repository root, whose untracked tree is twenty-one modules
  behind `main` — subprocesses imported a stale `uclone_x` and failed only on reaching one
  of the missing modules, e.g. `No module named 'uclone_x.evaluation.answerer'` (#718);
* a worktree that had acquired a `.venv` of its own — `ucx` prefers it, and Pyright
  reported 258 unresolved imports against untouched code (#679, second comment).

Severities are assigned by *whether the condition is satisfiable and locally fixable*:

* **refuse** on a worktree-local `.venv` that this branch has not declared — residue by
  construction (the inheritance is the design), fixable by deleting one directory;
* **warn** on a worktree-local `.venv` that this branch *has* declared, by changing
  `uv.lock` — AGENTS.md:82's Permitted Exception, which the refusal above used to
  contradict in so many words ("a local one is never intended"); see
  `find_declared_venv_exception` for what "declared" is allowed to mean and #968 for the
  decision that made it mean anything at all;
* **refuse** when the helped child cannot resolve `uclone_x` under the tree under test —
  nothing measured afterwards would be about this tree;
* **refuse** when the *ambient* install is unimportable — broken for every checkout at
  once, and fixed by `uv sync --all-extras` from the repository root;
* **warn** when the ambient install merely resolves somewhere else — the unsatisfiable
  case above. It is still worth printing on every run, because it is what an ad-hoc
  script or any `subprocess.run` that skips `repo_subprocess_env` will silently get, and
  because it is the standing evidence for #718.

`ok` means **self-consistent, not current.** It says the helped child resolved `uclone_x`
inside the tree under test and the ambient install agrees — it says nothing about what
revision that tree is at. The sharpest instance is the repository root itself: it has no
`.git` file and so no worktree-local `.venv`, the editable install names its own `src`, and
this function returns `ok` there — in the one tree on this machine that is twenty-one
modules behind `main` (#718). Read the verdict as "nothing here will resolve to a
*different* tree", never as "this tree is up to date"; no probe that reads only what the
interpreter resolved can tell you the second thing.

A third option, weighed and declined rather than absent. The paragraph above rules out
"the install names *this* tree" as unsatisfiable, and that is right, but it rules out more
than it establishes: the case that actually cost the hours was an ambient tree that was
*stale*, not merely *different*, and staleness is mechanically separable from identity. A
rule of the shape "refuse when the ambient tree is missing modules this tree has" **is**
satisfiable in every worktree at once whenever the ambient tree is current, so the binary
framing does not dispose of it. Measured on this machine, against the bare root the
install names: the ambient tree is missing 22 `.py` files this one has and has 1 this one
does not. It is declined because the naive form of it misfires on exactly the change a
builder is most likely to be making — any branch that adds a module is "missing" that
module from an ambient tree that is perfectly current, and this branch is one of them.
Distinguishing "behind `main`" from "ahead of the install" needs a revision comparison
rather than a file-set comparison, which is a second source of truth (`git` state) inside
a check whose whole claim is that it reads only what the interpreter actually resolved.
That is a defensible thing to build; it is not this change, and the reason it is not is
this paragraph rather than the definition above.

All of this is detection at the next gate run, not prevention. Nothing stops `uv sync` or
`uv run` being typed inside a worktree in the first place, and the `.pth` rewrite the
first one causes is not locally repairable — one editable install cannot name every
worktree at once. That half is #737.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

#: What `check_environment_provenance` found. Everything except `ok`, `ambient-foreign`
#: and `worktree-venv-declared` is a refusal; see the module docstring for why that split
#: is where it is.
ProvenanceStatus = Literal[
    "ok",
    "ambient-foreign",
    "ambient-unimportable",
    "tree-unimportable",
    "worktree-venv",
    "worktree-venv-declared",
]

#: Statuses the gate must not run past.
BLOCKING_STATUSES: Final[frozenset[ProvenanceStatus]] = frozenset(
    {"ambient-unimportable", "tree-unimportable", "worktree-venv"}
)

#: Printed by the child; parsed by `probe_module_origin`. A sentinel rather than bare
#: `print(uclone_x.__file__)` because a package whose import emits to stdout would
#: otherwise be indistinguishable from the answer.
_ORIGIN_PREFIX: Final = "uclone-x-origin:"

_PROBE_SCRIPT: Final = (
    f"import uclone_x\nprint({_ORIGIN_PREFIX!r} + (uclone_x.__file__ or '<namespace package>'))\n"
)

#: Seconds. A child that has not answered by now is itself the failure; the gate must not
#: hang on an environment probe.
_PROBE_TIMEOUT: Final = 60.0


def repo_subprocess_env(
    root: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment a child process needs to import the tree at `root`.

    This is the one definition of that environment. `pyproject.toml`'s
    `pythonpath = [".", "src"]` configures the pytest process only, so a test that spawns
    a child and does not pass this gets whatever the shared editable install names — a
    different tree, at a different revision, on a machine-dependent basis (#679).

    `<root>/src` is **prepended** to any inherited `PYTHONPATH` rather than replacing it:
    a caller that has deliberately put something on the path keeps it, and the tree under
    test still wins. Prepending is what makes the result independent of the install.
    """
    env = dict(os.environ if base is None else base)
    src = str((root / "src").resolve())
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{inherited}" if inherited else src
    return env


def probe_module_origin(
    *,
    python_executable: str | None = None,
    env: Mapping[str, str] | None = None,
    isolated: bool = False,
) -> tuple[str | None, str]:
    """Ask a child process where its `uclone_x` comes from.

    Returns `(origin, error)`: the resolved `__file__` and an empty string on success, or
    `None` and a one-paragraph diagnosis on failure. Both halves are returned rather than
    raising because every caller here reports rather than propagates.

    `isolated=True` runs the child with `-I`, which ignores `PYTHONPATH` and does not
    prepend the working directory. That measures the **install** and nothing else, which
    is what an ad-hoc script or an un-helped `subprocess.run` will get. Measured rather
    than read out of the `.pth` file on purpose: the file records an intention, the probe
    records what actually answered.
    """
    argv = [python_executable or sys.executable]
    if isolated:
        argv.append("-I")
    argv.extend(["-c", _PROBE_SCRIPT])
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT,
            env=None if env is None else dict(env),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"

    for line in (result.stdout or "").splitlines():
        if line.startswith(_ORIGIN_PREFIX):
            return line[len(_ORIGIN_PREFIX) :].strip(), ""
    detail = ((result.stderr or "").strip() or (result.stdout or "").strip()).splitlines()
    return None, detail[-1] if detail else f"child exited {result.returncode} saying nothing"


def find_worktree_local_venv(root: Path) -> Path | None:
    """A `.venv` a worktree was not given by inheritance, or None.

    Only linked worktrees are eligible. A linked worktree has `.git` as a **file** (a
    `gitdir:` pointer); the repository common dir has it as a directory. So this reports
    nothing at the repository root, where a `.venv` is the shared one every worktree is
    meant to inherit, and reports it inside a worktree, where it came from a
    `uv run`/`uv sync`/`pip install` typed there (#679).

    Finding one is **not** by itself a verdict. It is usually residue, and it is
    occasionally AGENTS.md:82's Permitted Exception — a branch that genuinely requires
    different dependency versions. `find_declared_venv_exception` is what separates the
    two; this function only answers whether there is anything to separate.
    """
    if not (root / ".git").is_file():
        return None
    candidate = root / ".venv"
    return candidate if candidate.is_dir() else None


#: The file whose content decides whether a branch "genuinely requires different
#: dependency versions" (AGENTS.md:82).
#:
#: `uv.lock` alone, and leaving `pyproject.toml` out is the substance of the rule rather
#: than an omission. `pyproject.toml` carries Ruff, Pyright and pytest configuration
#: beside its dependency tables, so a branch that only retuned a linter would read as a
#: dependency exception. Nothing true is lost by excluding it: gate stage 1 refuses
#: whenever `uv lock --check` finds the lock no longer resolves what `pyproject.toml`
#: declares, so a branch that genuinely changed its dependencies cannot reach stage 1b
#: with an unchanged `uv.lock`. Strictly narrower, and therefore strictly fewer false
#: allows, at the cost of none of the true ones.
DEPENDENCY_LOCKFILE: Final = "uv.lock"

#: Which `main` to compare against, in order of preference. `origin/main` first because
#: it is the ref every worktree in this repository is cut from (AGENTS.md §3); the local
#: branch is the fallback for a checkout with no remote. When neither resolves the check
#: **fails closed** — see `find_declared_venv_exception`.
_MAIN_REFS: Final[tuple[str, ...]] = ("refs/remotes/origin/main", "refs/heads/main")


@dataclass(frozen=True)
class VenvExceptionCheck:
    """Whether this branch has declared AGENTS.md:82's Permitted Exception, and why.

    `detail` is carried beside the verdict because it is half the fix: a builder told
    only "not declared" cannot tell a branch that changed no dependencies from a
    repository whose `origin/main` was never fetched, and those two want opposite
    actions.
    """

    declared: bool
    detail: str


def _git(root: Path, *args: str) -> tuple[int, str]:
    """Run `git` in `root` and return `(returncode, stdout)`, never raising.

    Stdout is returned even on failure so that a caller can report what git said; a
    non-zero code is the only thing any caller here branches on.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return result.returncode, (result.stdout or "").strip()


def find_declared_venv_exception(root: Path) -> VenvExceptionCheck:
    """Does this branch declare that it needs dependencies of its own?

    AGENTS.md:82 permits a worktree-local `.venv` for "a branch that genuinely requires
    different dependency versions", and until #968 the gate had no way to read that, so
    it refused the permitted case along with the residue and told the builder a local
    venv "is never intended". The decision recorded on #968 is that the declaration is
    the branch's own `uv.lock`: if it differs from `main`, the branch is saying it needs
    a different dependency set, and a local environment follows from that.

    Three choices in here are load-bearing.

    **Against the merge base, not against `main`'s tip.** A two-dot `git diff main` also
    reports every `uv.lock` change `main` acquired after this branch forked — changes
    this branch never made — so a branch could inherit an exception it did not declare
    simply by being a few days old. The merge base is what isolates the branch's own
    edit. It also means an uncommitted `uv.lock` in the working tree counts, which is
    deliberate: the builder who needs the exception is mid-work by definition.

    **`uv.lock` only** — see `DEPENDENCY_LOCKFILE` for why that is narrower rather than
    weaker.

    **Fail closed.** Every path that cannot establish the comparison returns
    `declared=False`, which is the pre-#968 behaviour: refuse. The failure mode worth
    engineering against is the false *allow* — a stray `.venv` silently poisoning the
    imports of a gate that passed — and a false refusal costs a builder one named,
    actionable message.

    This check does admit one false allow it cannot rule out: a branch that changed
    `uv.lock` for a reason unrelated to needing its own environment, and that *also*
    carries residue, is allowed. The signal is the honest one the decision named, and
    the allowed case is still printed as a warning naming the venv, so it is visible
    rather than silent; separating "changed the lock" from "changed the lock *because*
    this branch needs its own environment" would need a second, forgeable artefact.
    """
    main_ref: str | None = None
    for ref in _MAIN_REFS:
        if _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")[0] == 0:
            main_ref = ref
            break
    if main_ref is None:
        return VenvExceptionCheck(
            declared=False,
            detail=f"no main to compare against (tried {', '.join(_MAIN_REFS)})",
        )

    base_code, base = _git(root, "merge-base", main_ref, "HEAD")
    if base_code != 0 or not base:
        return VenvExceptionCheck(
            declared=False, detail=f"no commit shared with {main_ref} to compare against"
        )

    diff_code, changed = _git(root, "diff", "--name-only", base, "--", DEPENDENCY_LOCKFILE)
    if diff_code != 0:
        return VenvExceptionCheck(
            declared=False, detail=f"could not compare {DEPENDENCY_LOCKFILE} against {main_ref}"
        )
    if not changed:
        return VenvExceptionCheck(
            declared=False,
            detail=(
                f"{DEPENDENCY_LOCKFILE} is identical to {main_ref} "
                f"(merge base {base[:12]}), so this branch declares no dependency change"
            ),
        )
    return VenvExceptionCheck(
        declared=True,
        detail=(
            f"{DEPENDENCY_LOCKFILE} on this branch differs from {main_ref} (merge base {base[:12]})"
        ),
    )


def _declared_venv_notes(worktree_venv: Path, exception: VenvExceptionCheck) -> list[str]:
    """What to print above everything else when the local venv is the permitted one.

    A separate function because it is prepended to whatever the remaining checks find
    rather than returned instead of them — see `describe_environment_provenance`.
    """
    return [
        f"[bold yellow]⚠ This worktree has a virtual environment of its own, and this "
        f"branch declares it: {worktree_venv}[/bold yellow]",
        f"[yellow]  AGENTS.md:82 Permitted Exception — {exception.detail} — so this is "
        "the documented arrangement for a branch that needs different dependency "
        "versions, not residue, and the gate does not refuse it (#968).[/yellow]",
        "[dim]  Everything below was measured against that local .venv rather than the "
        "shared one. If it was not built with `uv sync --all-extras`, missing dev/test "
        "extras will surface below as failures that have nothing to do with your "
        "change.[/dim]",
        f"[dim]  If you did not intend a branch-local environment, remove "
        f"{worktree_venv} and re-run.[/dim]",
    ]


def _is_inside(path: str, directory: Path) -> bool:
    try:
        Path(path).resolve().relative_to(directory.resolve())
    except (ValueError, OSError):
        return False
    return True


def describe_environment_provenance(
    root: Path,
    *,
    tree_origin: str | None,
    tree_error: str,
    ambient_origin: str | None,
    ambient_error: str,
    worktree_venv: Path | None,
    venv_exception: VenvExceptionCheck | None = None,
) -> tuple[ProvenanceStatus, list[str]]:
    """Turn two probe results and one directory check into a status and a report.

    Separated from the probes so that every branch below is reachable from a test without
    building a Python environment per case — and so the probes stay small enough to be
    exercised once each, for real, rather than mocked.

    Checks are ordered most-explanatory first: a worktree-local `.venv` explains every
    other reading on this list, so reporting a downstream symptom above it would send the
    reader after the wrong thing.

    A **declared** local venv (AGENTS.md:82, #968) is the one finding that does not
    return on its own. It is prepended as a note and the remaining checks still run,
    because the alternative — returning a non-blocking status from the top of the
    function — would mask a genuine `tree-unimportable` behind it and convert a refusal
    into a pass. The note is what the reader needs; the early return was never the point.

    `venv_exception` defaults to `None`, read as "not checked", which is treated exactly
    like "not declared": a caller that finds a venv and does not ask gets the refusal.
    """
    src = root / "src"
    notes: list[str] = []

    if worktree_venv is not None:
        if venv_exception is None or not venv_exception.declared:
            reason = "not checked" if venv_exception is None else venv_exception.detail
            return "worktree-venv", [
                f"[bold red]✖ This worktree has a virtual environment of its own, and "
                f"nothing on this branch asked for one: {worktree_venv}[/bold red]",
                "[red]  Worktrees inherit the repository's shared .venv (AGENTS.md §3). "
                "A local one is permitted in exactly one case — AGENTS.md:82's Permitted "
                "Exception, a branch that genuinely requires different dependency "
                f"versions — and this branch is not it: {reason}.[/red]",
                "[red]  So this is residue of a `uv run`, `uv sync` or `pip install` "
                "typed inside the worktree, and `ucx` prefers it — so the gate you are "
                "reading was measured against it (#679).[/red]",
                f"[red]  Fix: remove {worktree_venv} and re-run. Do NOT `uv sync` to "
                "repair it; that repoints the shared environment for every other "
                "checkout.[/red]",
                f"[red]  If this branch really does need different dependency versions, "
                f"the gate reads that from {DEPENDENCY_LOCKFILE}: change it on this "
                "branch, then rebuild the local venv with `uv sync --all-extras` "
                "(AGENTS.md:82).[/red]",
            ]
        notes = _declared_venv_notes(worktree_venv, venv_exception)

    if tree_origin is None:
        return "tree-unimportable", [
            *notes,
            "[bold red]✖ A child process cannot import `uclone_x` from the tree under "
            "test.[/bold red]",
            f"[red]  Tree: {src}[/red]",
            f"[red]  Child said: {tree_error}[/red]",
            "[red]  Nothing measured after this would be about this tree, so the gate "
            "stops here rather than reporting a result it cannot attribute.[/red]",
        ]

    if not _is_inside(tree_origin, src):
        return "tree-unimportable", [
            *notes,
            "[bold red]✖ A child process given this tree on PYTHONPATH imported "
            "`uclone_x` from somewhere else.[/bold red]",
            f"[red]  Expected under: {src}[/red]",
            f"[red]  Actually got:   {tree_origin}[/red]",
            "[red]  Something ahead of PYTHONPATH on sys.path is shadowing the tree — a "
            "non-editable `uclone_x` installed into the venv is the usual cause.[/red]",
        ]

    if ambient_origin is None:
        return "ambient-unimportable", [
            *notes,
            "[bold red]✖ The shared environment's own `uclone_x` install is broken.[/bold red]",
            f"[red]  An isolated child said: {ambient_error}[/red]",
            "[red]  The editable install names a directory that is gone — the signature "
            "of a `uv sync` run inside a worktree that was later removed (#679).[/red]",
            "[red]  Every checkout sharing this venv is affected, not just this one.[/red]",
            "[red]  Fix: `uv sync --all-extras` from the repository root — never from "
            "inside a worktree.[/red]",
        ]

    if not _is_inside(ambient_origin, src):
        return "ambient-foreign", [
            *notes,
            "[bold yellow]⚠ The shared editable install points outside this tree.[/bold yellow]",
            f"[yellow]  An isolated child imports: {ambient_origin}[/yellow]",
            f"[yellow]  This tree is:              {src}[/yellow]",
            "[dim]  Not a failure: one shared venv holds one editable path, so it cannot "
            "name every worktree at once. The gate's own children are given this tree "
            "explicitly (uclone_x.cli.environment_provenance.repo_subprocess_env).[/dim]",
            "[dim]  It is printed on every run because any subprocess that skips that "
            "helper — an ad-hoc script above all — silently gets the tree named above "
            "instead, at whatever revision it happens to be. See #718.[/dim]",
        ]

    resolved = f"[green]✔ A child process resolves `uclone_x` under {src}.[/green]"
    if notes:
        return "worktree-venv-declared", [*notes, resolved]
    return "ok", [resolved]


def check_environment_provenance(root: Path | None = None) -> tuple[ProvenanceStatus, list[str]]:
    """Run both probes against `root` (default: the working directory) and report."""
    tree = (root or Path.cwd()).resolve()
    tree_origin, tree_error = probe_module_origin(env=repo_subprocess_env(tree))
    ambient_origin, ambient_error = probe_module_origin(isolated=True)
    worktree_venv = find_worktree_local_venv(tree)
    return describe_environment_provenance(
        tree,
        tree_origin=tree_origin,
        tree_error=tree_error,
        ambient_origin=ambient_origin,
        ambient_error=ambient_error,
        worktree_venv=worktree_venv,
        # Only asked when there is something to decide. Two `git` calls are cheap, but
        # the overwhelmingly common case is a worktree with no local venv at all, and a
        # check that runs then is a check whose cost and failure modes are paid for
        # nothing.
        venv_exception=None if worktree_venv is None else find_declared_venv_exception(tree),
    )
