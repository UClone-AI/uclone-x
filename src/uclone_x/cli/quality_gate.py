"""Automated Quality Gate runner for UClone-X."""

from __future__ import annotations

import datetime
import errno
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from rich.console import Console

from uclone_x.cli.bundle_freshness import BUILD_ENVIRONMENT, check_bundle_freshness
from uclone_x.cli.environment_provenance import (
    BLOCKING_STATUSES,
    check_environment_provenance,
)

console = Console()

# Both paths are **deliberately relative**, and the reason is worktree isolation (#436).
#
# These name per-tree scratch state: the junit report is the artifact of *this* tree's last
# run, and the failure history is *this* builder's accumulated failures (#207). Resolving
# them against a stable absolute location instead — the repository common dir, say — would
# make every concurrent worktree share one junit report and one history, so the last writer
# would win on the report and each builder's failures would interleave with every other
# builder's in the log. Relative-to-cwd is the correct scope; what was accidental before
# #436 was that the choice was unrecorded and that a *test* depended on it.
#
# Because they are relative, they resolve against the process cwd, which makes them ambient
# input. Every function below therefore takes them as parameters rather than reading these
# globals directly, and tests MUST pass `tmp_path`-based values. A test that lets these
# defaults apply reads whatever the previous gate run left in the tree it happens to run in,
# which made the gate's own coverage figure depend on its own history (#436).
JUNIT_REPORT_PATH = Path(".pytest_cache/junit.xml")
FAILURE_LOG_PATH = Path(".pytest_cache/failure_history.jsonl")

# Marker expression per test scope — the *only* place a tier boundary is defined.
#
# `unit` must name every other tier negatively rather than relying on directory layout.
# Selecting with `-m "not e2e"` alone (the pre-#377 behaviour) meant that adding a
# `tests/live/` directory would have swept live, token-spending tests into
# `./ucx test check`, while the design document claimed Tier 1 makes zero network calls.
# A tier that is not named in a marker expression is not a boundary.
#
# `all` keeps its documented meaning ("Unit + E2E") and therefore also excludes the
# recorded and live tiers. Since `gate` now includes E2E, `--all` selects the same set;
# it is retained because it is in the guides, the review checklist and muscle memory, and
# silently removing a flag that still appears in AGENTS.md would be worse than a synonym.
#
# `gate` is what `./ucx test check` runs: everything that is offline and free. It is not a
# test level, and naming it separately is the point — `unit` now selects only tests of
# product code, while the fitness functions that check the repository's own declarations are
# selectable on their own. Both were previously the same undifferentiated selection, so
# neither could be run or excluded alone.
#
# `gate` now *includes* E2E. It previously excluded it, which meant the default gate's
# green said nothing about the UI: `./ucx test check` reported success while deselecting
# the only suite that exercises the rendered application. That is not a theoretical gap —
# the E2E suite is what caught the mislabelled step-budget gauge on PR #551, a defect the
# unit suite and Pyright both passed over. Its cost is inside the measured gate figure in
# AGENTS.md (#967), and this repository has no hosted CI, so a check that is skipped by
# default is a check that does not run. `--fast` is the opt-out for the tight edit loop;
# `gate` is what a commit is measured against.
_MARKER_EXPRESSIONS: Final[dict[str, str]] = {
    "gate": "not recorded and not live",
    # The tight-edit-loop selection: everything `gate` runs except the browser suite.
    "fast": "not e2e and not recorded and not live",
    "unit": "not e2e and not recorded and not live and not fitness",
    "fitness": "fitness",
    "recorded": "recorded",
    "live": "live",
    "e2e": "e2e",
    # Release qualification. Its members are collected by every scope above and skipped
    # there (`tests/support/live_optin.py`), so this is the only selection that runs them.
    # It exists because the checks it holds are the ones a push cannot afford: the
    # distribution-install lane installs the built wheel into a clean venv per published
    # extra. Measured at 4s against a warm `uv` cache and minutes against a cold one, and
    # a gate stage that only works when a cache happens to be warm is a network
    # dependency, which the `gate` scope may not take — the quality testing and evaluation
    # guide's scope table defines that scope as "everything offline and free". P8 is about
    # *local* verification and says nothing about the network, so it is not the citation
    # for this. #658 asked where such a check runs given this repository has no hosted CI;
    # the answer is that a release runs it by name, and naming it is what makes it
    # auditable rather than a step in someone's head.
    "pre-release": "pre_release",
    "all": "not recorded and not live",
}

# Scopes that must disable coverage. `addopts` in `pyproject.toml` carries
# `--cov-fail-under=70`, which is measured against the *whole* of `src/uclone_x`. Any
# partial selection therefore fails on coverage regardless of whether its tests passed,
# so this is a correctness requirement and not a convenience.
_NO_COVERAGE_SCOPES: Final[frozenset[str]] = frozenset(
    {"recorded", "live", "e2e", "fitness", "pre-release"}
)

# How each scope's pytest stage is run (#967).
#
# Run serially, the pytest stage was almost the whole of the gate's wall time, and that time
# was thousands of small tests rather than a few slow ones — the shape that distributes. A
# scope is distributed only if everything it selects is also selected by `gate`, because the
# gate's own repeated runs are then the evidence that those tests are parallel-safe.
#
# Whole run on workers: the scopes with no browser suite in them.
_PARALLEL_SCOPES: Final[frozenset[str]] = frozenset({"fast", "unit", "fitness"})

# Two steps: everything except the browser suite on workers, then the browser suite in one
# process, with coverage appended so the threshold and the floors judge the combined data.
#
# The browser suite is kept out of the workers by measurement, not caution. Its cases drive a
# real Chromium against a server on the same machine, and two of them race the page's own
# load-time requests (#942; #946 and #975). Running them beside eleven busy workers made those
# races fire: the fully parallel gate passed 1 run in 3 even with #942's test fix applied,
# against 3 in 3 for this split, which costs about 22 s of serial browser time.
_SPLIT_BROWSER_SCOPES: Final[frozenset[str]] = frozenset({"gate", "all"})

# Kept serial throughout, each for a reason of its own:
#   * `e2e` is the browser suite, for the reason above.
#   * `live` spends tokens against one real endpoint; N workers are N concurrent callers, which
#     multiplies the rate and the cost of a run whose point is a faithful single conversation.
#   * `pre-release` installs the built wheel into clean venvs through one shared `uv` cache and
#     the network. Concurrent installs contend for the cache and the link; they are not faster.
#   * `recorded` replays cassettes. It is empty or nearly so, and outside `gate`.

# The most workers a run starts, whatever the machine offers. Measured — see
# `pytest_worker_count` for the figures and why more stopped helping.
_MAX_PYTEST_WORKERS: Final = 12

# How tests are handed to workers: `load`, any pending test to any idle worker. Chosen by
# measurement over `loadgroup`, `loadfile` and `loadscope`, the modes that keep a module's
# tests together on one worker; `load` was the fastest of the four. The figures are on the
# #967 pull request.
_PYTEST_DIST_MODE: Final = "load"

#: The gate's exit code when a parallel scope was asked for and pytest-xdist is not installed.
_PARALLEL_RUNNER_MISSING_EXIT: Final = 1

TEST_SCOPES: Final[tuple[str, ...]] = (
    "gate",
    "fast",
    "unit",
    "fitness",
    "recorded",
    "live",
    "e2e",
    "pre-release",
    "all",
)

# pytest's EXIT_NOTESTSCOLLECTED. Reported honestly rather than as a coverage failure:
# an empty tier is a real problem (a release gate that passes because it collected
# nothing is the silent pass P6 forbids) but it is not the problem the generic message
# names, and the recorded/live tiers are legitimately empty until the roadmap seeds them.
_PYTEST_NO_TESTS_COLLECTED: Final = 5
PYTEST_NO_TESTS_COLLECTED: Final = _PYTEST_NO_TESTS_COLLECTED

SCOPE_LABELS: Final[dict[str, str]] = {
    "gate": "Offline Gate (unit, integration, fitness functions and E2E)",
    "fast": "Fast Gate (offline gate without the browser suite)",
    "unit": "Tier 1 Unit Suite (excluding fitness, recorded, live, E2E)",
    "fitness": "Fitness Functions (repository declarations; no product code)",
    "recorded": "Tier 2 Recorded Playback Suite (cassette replay, no network)",
    "live": "Tier 3 Live Suite (real LLM endpoint)",
    "e2e": "E2E Playwright Suite",
    "pre-release": "Release Qualification Suite (clean-venv install matrix; network)",
    "all": "Offline Gate + E2E Suite (same selection as the default gate)",
}

# Per-package branch-coverage floors, checked after the suite runs.
#
# The single global `--cov-fail-under=70` is satisfied by a repository-wide average, so a
# subsystem can rot for a long time behind a healthy total: at 89% overall, `ui/` sat at
# 82% and `a2a/` at 85%, and nothing in the gate could tell. A global floor also cannot be
# raised, because the weakest package sets the ceiling for the whole number.
#
# These are a **ratchet, not a target**: each is set a few points below what the package
# measures today, so the gate catches a regression without demanding new tests for code
# nobody is touching. Raise a floor when a package improves; never lower one to make a
# red gate green — that is the silent weakening P8 warns about.
#
# Measured 2026-09-07: agent 89, core 90, tools 90, engine 88, llm 94, ontology 93,
# sandbox 98, a2a 85, ui 82, telemetry 90, skills 89, cli 85.
PACKAGE_COVERAGE_FLOORS: Final[dict[str, int]] = {
    "a2a": 80,
    "agent": 85,
    "cli": 80,
    "core": 85,
    "engine": 84,
    "llm": 90,
    "ontology": 89,
    "sandbox": 94,
    "skills": 85,
    "telemetry": 86,
    "tools": 86,
    "ui": 78,
}


# Policy: every top-level directory holding Python that this repository ships or
# depends on is subject to static verification (Ruff format/lint, Pyright strict).
#
# The previous text claimed this set already was "all top-level directories
# containing Python code", and it was not: `evals/` (~6,000 lines - eight
# evaluation suites, the runner, the grading module, the promptfoo bridge) was
# absent, so Ruff never inspected it, while `[tool.pyright] include` in
# pyproject.toml did. `evals/suites/data_quality.py` reached `main` unformatted
# as a direct result (#588). A comment asserting coverage the tuple does not
# provide is worse than no comment: it answers the question a reader would
# otherwise go and check.
#
# `benchmarks/` is deliberately excluded and named in PYTHON_UNCHECKED_PATHS
# below, so the omission is a recorded decision rather than a silence.
PYTHON_CHECK_PATHS: Final[tuple[str, ...]] = (
    "src",
    "tests",
    "swarm",
    "scripts",
    "evals",
    "oss",
)

# Top-level Python directories intentionally outside the gate, with the reason.
# `test_quality_gate_scope_covers_every_python_directory` fails when a directory
# appears that is in neither tuple, so a new one cannot escape by being forgotten.
PYTHON_UNCHECKED_PATHS: Final[dict[str, str]] = {
    "benchmarks": (
        "ad-hoc measurement scripts, not shipped and not imported by src/. Also "
        "outside `[tool.pyright] include`. Holds 13 deliberate style violations "
        "(E701/E702 one-liners) that are legible in a throwaway script; enforcing "
        "the gate here would be churn against code whose value is being quick to "
        "write. Move a script into scripts/ when it stops being throwaway."
    ),
}


def measure_package_coverage(package: str) -> int | None:
    """Total branch coverage for one `uclone_x` subpackage, or None if unmeasurable.

    Reads the data file `pytest-cov` just wrote rather than re-running the suite. Returns
    None when the package has no measured files, which the caller reports as a problem in
    its own right: a floor declared for a package coverage cannot see is a floor that
    silently never applies.
    """
    result = subprocess.run(
        ["coverage", "report", f"--include=src/uclone_x/{package}/*"],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 2):
        return None
    total_rows = [ln for ln in result.stdout.splitlines() if ln.startswith("TOTAL")]
    if not total_rows:
        return None
    for field in reversed(total_rows[-1].split()):
        if field.endswith("%"):
            try:
                return int(field.rstrip("%"))
            except ValueError:
                return None
    return None


def check_package_coverage_floors(
    floors: dict[str, int] | None = None,
) -> list[tuple[str, int, int]]:
    """Return `(package, measured, floor)` for every package below its declared floor.

    A package whose coverage cannot be measured is returned with `measured=-1` rather
    than skipped, so a typo in the floor table fails loudly instead of quietly disabling
    the check for that package. Every shortfall is reported, not just the first: a gate
    that stopped at one would hide the rest of the work.
    """
    table = PACKAGE_COVERAGE_FLOORS if floors is None else floors
    shortfalls: list[tuple[str, int, int]] = []
    for package, floor in sorted(table.items()):
        measured = measure_package_coverage(package)
        if measured is None:
            shortfalls.append((package, -1, floor))
        elif measured < floor:
            shortfalls.append((package, measured, floor))
    return shortfalls


#: Returned when the gate is invoked where there is no project to check.
#:
#: Not a distinct code: pytest already exits 2 for an interrupted session, and
#: reserving a number the tools do not use would be a private convention no
#: caller could rely on. What separates the two cases is the message, which is
#: printed on stderr whether or not the run is quiet — see below.
SCOPE_RESOLUTION_EXIT_CODE: Final[int] = 2


def resolve_check_paths(root: Path | None = None) -> tuple[str, ...]:
    """The declared check paths that exist in this tree, in declaration order.

    Not every path exists in every tree. `swarm`, `scripts` and `oss` hold the
    build swarm and the publication tooling, which are not part of the
    published open-source subset, and `ruff` and `pyright` both fail on a path
    that is not there. Filtering keeps one gate honest in both trees rather
    than making the published one carry a different definition of "passing".

    `src` is not optional: a tree without it is not this project, and silently
    checking nothing would report success over an empty scope.
    """
    base = Path.cwd() if root is None else root
    existing = tuple(path for path in PYTHON_CHECK_PATHS if (base / path).exists())
    if "src" not in existing:
        raise FileNotFoundError(
            f"no `src` directory under {base}; run the quality gate from a repository root"
        )
    return existing


# The resolved-dependency manifest `uv` writes from `pyproject.toml`. Relative for the
# same reason the paths above are: it names a file in *this* tree, and a worktree's
# lockfile is the one its own `pyproject.toml` must agree with.
LOCKFILE_PATH = Path("uv.lock")

# What `check_lockfile_freshness` found.
#
# `absent` is a real third answer and not a flavour of `fresh`: a tree with no `uv.lock`
# has nothing to be stale against, and calling that "passed" would be the substituted
# default P6 forbids. `uv-missing` is likewise not a flavour of `absent` — one means the
# check does not apply, the other means it could not be run, and the gate treats them
# differently for exactly that reason.
LockfileStatus = Literal["fresh", "stale", "uv-missing", "absent"]

# The gate's exit code when the lockfile could not be shown to be current. Named rather
# than written as a bare `1` in two places: `uv`'s own exit codes are not the gate's, and
# the generic literal appears in other stages, so a mutation here would otherwise be
# indistinguishable from a mutation there.
_LOCKFILE_FAILURE_EXIT: Final = 1

#: Same value, separate name: both are "the environment is not fit to be measured", and a
#: reader tracing a `1` back to its origin should land on the stage that produced it.
_ENVIRONMENT_FAILURE_EXIT: Final = 1


def check_lockfile_freshness(root: Path | None = None) -> tuple[LockfileStatus, list[str]]:
    """Report whether `uv.lock` still resolves the dependencies `pyproject.toml` declares.

    `uv lock --check` re-resolves and exits non-zero when the result would differ from the
    committed lockfile. It writes nothing — the mutating form is `uv lock` — so this is
    safe to run in a worktree that shares the root `.venv` (AGENTS.md §3), and it does not
    touch the environment the rest of the gate is measured in.

    Why this is a gate stage at all: the drift is silent. `pyproject.toml` and `uv.lock`
    are edited by different actions — a human writes the first, a tool writes the second —
    and nothing in the repository noticed when a commit did only the first. #646 moved the
    shell and adapter frameworks into extras and #654 bumped the version; neither relocked,
    and the lock sat wrong on `main` until someone ran `uv lock --check` by hand. This was
    written down before it happened the second time: it is finding F2 of the review of
    #610, "No `uv lock --check` anywhere in the gate — the direct cause of the drift this
    PR cleans up". A finding filed and not built is a finding that gets re-filed.

    Missing `uv` is a **failure**, not a skip, whenever `uv.lock` is present.

    The temptation is to skip: the gate runs in developer checkouts, and a developer
    without `uv` would rather see the other stages than an error about a tool they do not
    have. That trade is exactly backwards. A check that quietly does nothing reports the
    same green as a check that ran and passed, so the one state a builder needs to
    distinguish is the one the output hides — and this repository has no hosted CI (P8),
    so this gate is the *only* place the lockfile is ever verified. A skip here does not
    degrade the check, it deletes it, on precisely the machines least likely to notice.
    `uv` is not an exotic dependency either: it is how the venv this gate runs inside was
    built, so its absence means the checkout is already in a state worth stopping on.

    A tree with no `uv.lock` is a different answer and is reported as `absent`, not run and
    not failed: the published open-source subset is built from `oss/manifest.yaml`, which
    does ship `uv.lock`, but a source export or a partial checkout may not, and failing on
    a file the tree never claimed to have would be an assertion about someone else's tree.

    Args:
        root: Directory to check. Defaults to the process cwd, matching `resolve_check_paths`.

    Returns:
        The status, and the lines to print for it (empty when there is nothing to say).
    """
    base = Path.cwd() if root is None else root
    lockfile = base / LOCKFILE_PATH
    if not lockfile.exists():
        return "absent", [
            f"[yellow]! No {LOCKFILE_PATH} in {base}; lockfile freshness not checked.[/yellow]"
        ]

    try:
        result = subprocess.run(
            ["uv", "lock", "--check"],
            cwd=base,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return "uv-missing", [
            f"[bold red]✖ {LOCKFILE_PATH} is present but `uv` could not be run: "
            f"{type(exc).__name__}: {exc}[/bold red]",
            "[red]  The lockfile cannot be verified, so it is reported as a failure and "
            "not as a pass.[/red]",
            "[red]  Install uv (https://docs.astral.sh/uv/) and re-run the gate.[/red]",
        ]

    if result.returncode == 0:
        return "fresh", []

    return "stale", [
        f"[bold red]✖ {LOCKFILE_PATH} is out of date with pyproject.toml.[/bold red]",
        "[red]  Fix: run [bold]uv lock[/bold] from the repository root and commit the "
        "result.[/red]",
        "[red]  Do NOT run `uv sync` in a worktree — it repoints the shared venv "
        "(AGENTS.md §3, #679).[/red]",
        f"[dim]{result.stderr.strip() or result.stdout.strip()}[/dim]",
    ]


# The gate's standard streams are put back into blocking mode around every stage (#993).
#
# `O_NONBLOCK` belongs to an open file description, not to a descriptor, so it is shared by
# every process that holds the same pipe: the gate, each stage it runs with inherited stdio,
# and anything outside the gate writing into the same pipe. When the flag is set and the
# pipe is full, a buffered Python write raises `BlockingIOError: [Errno 35] write could not
# complete without blocking`. In pytest's terminal writer that is an INTERNALERROR, which the
# junit report records as `pytest::internal` and the gate reports as a failed suite.
#
# Who sets it, measured on macOS on 2026-09-15:
#   * `ssh`, the transport `git push` runs for this repository's SSH origin, sets it on the
#     stderr it inherits and keeps it set for its whole life. It does not set it again once
#     another process clears it. When the pre-push hook still ran the gate (before #966), a
#     `git push | grep | tail` ran pytest into a non-blocking pipe, and the crash was
#     observed after 3171 passing tests.
#   * `node` sets it while it writes to an inherited pipe (pyright, vitest and vite are node
#     programs), clears it when it exits normally, and leaves it set when it is killed.
# A full gate piped into a reader stalled for 150 s passed at 6a5fbbf with nothing outside
# it holding the pipe: the flag was set only while pyright and the vitest stages ran, and
# they cleared it on exit.
#
# So the flag is cleared when the gate starts, before each stage inherits the streams, and
# after each stage exits. A process that sets it again while a stage is running is not
# covered by this; only giving each stage a pipe of its own, relayed by the gate, would be.
_STD_STREAM_FDS: Final[tuple[int, ...]] = (1, 2)


def restore_blocking_stdio(fds: tuple[int, ...] = _STD_STREAM_FDS) -> list[int]:
    """Clear `O_NONBLOCK` on the given descriptors; return the ones that had it set.

    A closed descriptor is skipped, since nothing can write to it. Any other error is
    raised: a stream the gate cannot put back into blocking mode is one whose next large
    write may fail, and saying nothing about that would hide the crash this exists to stop.
    """
    if os.name != "posix":
        return []
    restored: list[int] = []
    for fd in fds:
        try:
            if os.get_blocking(fd):
                continue
            os.set_blocking(fd, True)
        except OSError as exc:
            if exc.errno == errno.EBADF:
                continue
            raise
        restored.append(fd)
    return restored


def _describe_restored(fds: list[int], when: str) -> str:
    names = ", ".join({1: "stdout", 2: "stderr"}.get(fd, f"fd {fd}") for fd in fds)
    return (
        f"ucx: {names} was non-blocking {when}; restored blocking mode so a full pipe waits "
        "instead of failing the next write (#993)."
    )


def run_stage(
    argv: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    """Run one stage that inherits the gate's stdio, with that stdio in blocking mode.

    `cwd` and `env` are passed on only when given, so the call `subprocess.run` receives is
    the one each stage made before #993. `quiet` suppresses the report of a restore after
    the stage, as the gate's `quiet` suppresses the one when it starts (#999).
    """
    restore_blocking_stdio()  # before: the stage inherits these descriptors
    try:
        if cwd is None and env is None:
            return subprocess.run(argv)
        if env is None:
            return subprocess.run(argv, cwd=cwd)
        return subprocess.run(argv, cwd=cwd, env=env)
    finally:
        left_non_blocking = restore_blocking_stdio()
        if left_non_blocking and not quiet:
            print(
                _describe_restored(left_non_blocking, f"after `{argv[0]}` exited"), file=sys.stderr
            )


# The frontend vitest suite (#915).
#
# Until #915 it ran only behind `-fe`, which no hook, no documented command and no CI
# passed. The suite was written, reviewed and extended — PR #905 alone added ~1,150 lines
# of it — and executed by nothing, so a frontend regression reached a green gate by
# construction. It now runs in every scope below, which is every scope a commit is
# measured against: `gate` (plain `./ucx test check`), `fast` (the documented edit loop, which
# drops the browser suite "and nothing else") and `all` (the no-op synonym for `gate`).
# The named pytest tiers stay Python-only; `-fe` still adds the suite to any of them.
_FRONTEND_SUITE_SCOPES: Final[frozenset[str]] = frozenset({"gate", "fast", "all"})

# What `run_frontend_suite` found. `absent`, `npm-missing` and `deps-missing` are separate
# answers for the reason `LockfileStatus` separates its own: "does not apply" and "could
# not be run" must never print the same thing as "ran and passed" (P6).
FrontendSuiteStatus = Literal["passed", "failed", "absent", "npm-missing", "deps-missing"]

#: "The suite could not run": an environment answer, not a test verdict, so it gets a name
#: of its own rather than borrowing vitest's exit code.
_FRONTEND_UNRUNNABLE_EXIT: Final = 1


def frontend_suite_selected(test_scope: str, *, skip_tests: bool, check_frontend: bool) -> bool:
    """Whether a gate run with these options runs the frontend vitest suite.

    `skip_tests` excludes it because the pre-commit hook's reason for skipping pytest — the
    tree a commit holds is not the tree that reaches `main` — applies to vitest unchanged.
    `check_frontend` (`-fe`) adds it to any scope, as it always did.
    """
    return check_frontend or (not skip_tests and test_scope in _FRONTEND_SUITE_SCOPES)


def run_frontend_suite(
    root: Path | None = None, *, quiet: bool = False
) -> tuple[FrontendSuiteStatus, int, list[str]]:
    """Run the frontend vitest suite, or say precisely why it could not be run.

    A missing `node_modules` or a missing `npm` is a **failure**, not a skip. The argument
    is the lockfile stage's for a missing `uv`, with a sharper edge: `frontend/node_modules`
    is gitignored, so **every fresh worktree lacks it** — a stage that skipped when it was
    absent would skip in exactly the place builders run the gate, and print a green line
    over a suite that did not run, which is #915 over again. The message names the fix;
    in a worktree the cheap one is a symlink to the primary workspace's `node_modules`,
    the frontend analogue of the shared `.venv`.

    A tree with no `frontend/package.json` is `absent` and passes with a note: it never
    claimed a frontend suite, and failing on one would be an assertion about another tree.

    Vitest reads the source and writes nothing tracked. The production build does not have
    that property — `vite build` writes the committed `src/uclone_x/ui_static` — which is
    why the build stays behind `-fe` and out of this function. Whether that committed
    bundle is still what the source builds is stage 6b's question (#878).

    Args:
        root: Repository root. Defaults to the process cwd, matching `resolve_check_paths`.
        quiet: Passed to `run_stage`, which runs vitest on the gate's stdio (#999).

    Returns:
        The status, the exit code the gate should record (0 for `passed` and `absent`),
        and the lines to print.
    """
    # Resolved against `root` (default cwd) inside the call, not as a module-level relative
    # constant: it names *this* tree's frontend, which is the one a worktree's gate tests.
    base = Path.cwd() if root is None else root
    frontend = base / "frontend"
    if not (frontend / "package.json").is_file():
        absent = [f"[yellow]! No frontend/package.json in {base}; frontend suite not run.[/yellow]"]
        return "absent", 0, absent

    cannot_run = [
        "[bold red]✖ frontend/ has a vitest suite but `npm` could not be run.[/bold red]",
        "[red]  The suite cannot be verified, so it is reported as a failure and not as a "
        "pass.[/red]",
        "[red]  Install Node.js (which provides npm) and re-run the gate.[/red]",
    ]
    if shutil.which("npm") is None:
        not_on_path = [*cannot_run, "[dim]`npm` was not found on PATH.[/dim]"]
        return "npm-missing", _FRONTEND_UNRUNNABLE_EXIT, not_on_path

    if not (frontend / "node_modules").is_dir():
        no_deps = [
            "[bold red]✖ frontend/node_modules is missing, so the vitest suite cannot "
            "run.[/bold red]",
            "[red]  Reported as a failure, not a skip: a fresh worktree never has it, and a "
            "skip there is a suite nobody runs (#915).[/red]",
            "[red]  Fix, in a worktree — link the primary workspace's copy (like the shared "
            ".venv):[/red]",
            '[red]    ln -s "$(git rev-parse --path-format=absolute --git-common-dir)/../'
            'frontend/node_modules" frontend/node_modules[/red]',
            "[red]  Fix, where there is no primary copy (or package-lock.json differs from "
            "it):[/red]",
            "[red]    npm ci --prefix frontend[/red]",
        ]
        return "deps-missing", _FRONTEND_UNRUNNABLE_EXIT, no_deps

    try:
        # `npm test` is `vitest run` — non-watch, so it terminates.
        vitest = run_stage(["npm", "test", "--silent"], cwd=frontend, quiet=quiet)
    except OSError as npm_exc:
        raised = [*cannot_run, f"[dim]{type(npm_exc).__name__}: {npm_exc}[/dim]"]
        return "npm-missing", _FRONTEND_UNRUNNABLE_EXIT, raised
    if vitest.returncode == 0:
        return "passed", 0, ["[green]✔ Frontend unit tests passed.[/green]"]
    return "failed", vitest.returncode, ["[bold red]✖ Frontend unit tests failed.[/bold red]"]


def pytest_worker_count(cpu_count: int | None = None) -> int:
    """How many pytest-xdist workers a parallel scope starts: one per CPU, capped.

    Capped because the gain flattens well before the core count and the cost does not: each
    worker is a full interpreter that imports the suite, and this machine is shared by
    several builders running gates at once. Never below one, so a platform where
    `os.cpu_count()` is None still runs.

    Args:
        cpu_count: CPUs to plan for. Defaults to `os.cpu_count()`.
    """
    available = os.cpu_count() if cpu_count is None else cpu_count
    return max(1, min(available or 1, _MAX_PYTEST_WORKERS))


def pytest_runs_in_parallel(test_scope: str, *, serial: bool = False) -> bool:
    """Whether any part of this scope's pytest stage is distributed across workers."""
    return (test_scope in _PARALLEL_SCOPES or test_scope in _SPLIT_BROWSER_SCOPES) and not serial


def parallel_runner_available() -> bool:
    """Whether pytest-xdist can be imported by the interpreter running the gate.

    The gate and the `pytest` it spawns come from the same environment (`./ucx` puts that
    environment's `bin` first on PATH), so asking here answers for the child.
    """
    return importlib.util.find_spec("xdist") is not None


def describe_missing_parallel_runner(test_scope: str) -> list[str]:
    """The refusal printed when a parallel scope meets an environment without pytest-xdist."""
    return [
        f"ucx: the `{test_scope}` scope runs pytest on parallel workers (pytest-xdist, #967), "
        f"and pytest-xdist is not installed in {sys.prefix}.",
        "  It is declared in pyproject.toml's `dev` extra; this environment predates that.",
        "  Fix, from the primary workspace (the shared .venv): uv sync --all-extras",
        "  To run once without it, in one process: add --serial (slower; same selection and "
        "coverage).",
        "  Refused rather than run serially: a gate that silently became three times slower "
        "would hide the out-of-date environment that made it so.",
    ]


def gate_marker_expression() -> str:
    """The `gate` scope's marker expression, for a run that narrows the gate by path."""
    return _MARKER_EXPRESSIONS["gate"]


def build_pytest_command(
    test_scope: str,
    *,
    junit_path: Path = JUNIT_REPORT_PATH,
    serial: bool = False,
) -> list[str]:
    """Build the pytest argv for one test scope.

    Extracted from `run_quality_gate` so the tier boundaries are testable without
    executing a suite. An unknown scope raises rather than silently degrading to "run
    everything" — a mistyped scope that quietly widened the selection would be exactly
    the silent substitution P6 forbids.

    This is the scope's selection as **one** invocation. For `gate` and `all` the gate does
    not run it as it stands unless asked to with `serial`: it runs the two steps
    `build_pytest_steps` derives from the same marker expression. `serial` drops the worker
    flags from a parallel scope and changes nothing else.
    """
    if test_scope not in _MARKER_EXPRESSIONS:
        raise ValueError(
            f"unknown test scope {test_scope!r}; expected one of {', '.join(TEST_SCOPES)}"
        )

    cmd = [
        "pytest",
        "-v",
        f"--junitxml={junit_path}",
        "-o",
        "junit_family=xunit2",
    ]
    if test_scope in _PARALLEL_SCOPES and not serial:
        cmd.extend(["-n", str(pytest_worker_count()), f"--dist={_PYTEST_DIST_MODE}"])
    if test_scope in _NO_COVERAGE_SCOPES:
        cmd.append("--no-cov")
    if test_scope == "pre-release":
        # Same shape as `live`: the tier is skipped unless the run opts in, so the scope
        # carries its own opt-in rather than depending on the marker expression alone.
        cmd.append("--pre-release")
    if test_scope == "live":
        # Tier 3 tests are collected but skipped unless the run opts in explicitly
        # (`tests/conftest.py`), so the scope has to carry the opt-in itself.
        cmd.append("--live")
    cmd.extend(["-m", _MARKER_EXPRESSIONS[test_scope]])
    return cmd


@dataclass(frozen=True)
class PytestStep:
    """One pytest invocation of the gate's pytest stage."""

    label: str
    argv: tuple[str, ...]
    junit_path: Path


def build_pytest_steps(
    test_scope: str,
    *,
    junit_path: Path = JUNIT_REPORT_PATH,
    serial: bool = False,
) -> list[PytestStep]:
    """The invocations that make up one scope's pytest stage, in the order they run.

    One step for every scope except `gate` and `all`, whose step is `build_pytest_command`
    unchanged. Those two run in two steps whose marker expressions partition the scope's
    selection by the `e2e` marker, so together they select exactly what the scope selects:

    1. everything except the browser suite, on workers, with `--cov-fail-under=0` because
       this step's coverage is partial and must not be judged on its own;
    2. the browser suite, in one process, with `--cov-append`, so the `--cov-fail-under` in
       `addopts` is applied to the combined data.

    Each step writes its own report beside `junit_path`; `merge_junit_reports` combines them
    into `junit_path`, which is where the failure extraction and people look.
    """
    whole = build_pytest_command(test_scope, junit_path=junit_path, serial=serial)
    if serial or test_scope not in _SPLIT_BROWSER_SCOPES:
        return [
            PytestStep(label=SCOPE_LABELS[test_scope], argv=tuple(whole), junit_path=junit_path)
        ]

    expression = _MARKER_EXPRESSIONS[test_scope]
    workers = pytest_worker_count()
    workers_report = junit_path.with_name(f"{junit_path.stem}.workers{junit_path.suffix}")
    browser_report = junit_path.with_name(f"{junit_path.stem}.browser{junit_path.suffix}")
    return [
        PytestStep(
            label=f"everything except the browser suite, on {workers} workers",
            argv=(
                "pytest",
                "-v",
                f"--junitxml={workers_report}",
                "-o",
                "junit_family=xunit2",
                "-n",
                str(workers),
                f"--dist={_PYTEST_DIST_MODE}",
                "--cov-fail-under=0",
                "-m",
                f"({expression}) and not e2e",
            ),
            junit_path=workers_report,
        ),
        PytestStep(
            label="the browser suite, in one process (coverage appended and judged here)",
            argv=(
                "pytest",
                "-v",
                f"--junitxml={browser_report}",
                "-o",
                "junit_family=xunit2",
                "--cov-append",
                "-m",
                f"({expression}) and e2e",
            ),
            junit_path=browser_report,
        ),
    ]


def merge_junit_reports(
    sources: list[Path], destination: Path, *, written_after: float
) -> list[Path]:
    """Combine the step reports into one report at `destination`; return the sources left out.

    A source is left out when it is missing, unparseable, or older than `written_after` — the
    last is a report some earlier run left behind because this run's step died before writing
    its own, and reading it would report that run's failures as this one's (#436). When nothing
    is left, `destination` is not written, exactly as a single pytest run that died leaves it.
    """
    combined = ET.Element("testsuites")
    left_out: list[Path] = []
    for source in sources:
        try:
            if source.stat().st_mtime < written_after:
                left_out.append(source)
                continue
            root = ET.parse(source).getroot()
        except (OSError, ET.ParseError):
            left_out.append(source)
            continue
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        combined.extend(suites)
    if len(left_out) < len(sources):
        destination.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(combined).write(destination, encoding="utf-8", xml_declaration=True)
    return left_out


def record_and_extract_failures(
    junit_path: Path = JUNIT_REPORT_PATH,
    *,
    failure_log_path: Path = FAILURE_LOG_PATH,
    quiet: bool = False,
) -> list[str]:
    """Extract failed test names from junit XML report and append to durable failure history (#207).

    `failure_log_path` is a parameter rather than a read of the module global so that a
    caller — a test above all — can name both the input report and the durable output log.
    Reading the global made the durable history writable by any test that reached this
    function with the ambient report in scope (#436).
    """
    failed_tests: list[str] = []
    if not junit_path.exists():
        return failed_tests

    try:
        tree = ET.parse(junit_path)
        root = tree.getroot()
        for testcase in root.iter("testcase"):
            has_failure = testcase.find("failure") is not None or testcase.find("error") is not None
            if has_failure:
                classname = testcase.attrib.get("classname", "")
                name = testcase.attrib.get("name", "")
                nodeid = f"{classname}::{name}" if classname else name
                failed_tests.append(nodeid)

        if failed_tests:
            failure_log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                "failed_count": len(failed_tests),
                "failures": failed_tests,
            }
            with open(failure_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    # Narrowed from `except Exception` (#436), which is the shape #380, #385 and #397
    # removed from the connectors — it should not survive in the gate that judges them.
    #
    # What must reach the caller, and would have been swallowed before:
    #   * `AttributeError` / `TypeError` / `KeyError` from a future edit to the extraction
    #     loop — a broken extractor and an unparseable report are different failures, and
    #     catching both as one made them indistinguishable;
    #   * `MemoryError` and `RecursionError` on a pathological report;
    #   * anything raised by `json.dumps`, which cannot fail on this all-`str`/`int`/`list`
    #     payload, so a raise from it is a defect and not an input problem.
    # What is genuinely an input problem and is caught here:
    #   * `ET.ParseError` — a malformed or partially-written report, the real case: pytest
    #     writes the report at session end, so a crashed or killed run leaves a truncated
    #     one behind;
    #   * `OSError` — the report vanished or became unreadable between `exists()` above and
    #     `parse` here (a real race: concurrent worktrees clean their own caches), or the
    #     failure log could not be created or appended to.
    #
    # It no longer `pass`es silently. Returning an empty list without saying why is the
    # substituted-empty-result P6 forbids: the caller's report would then claim no failed
    # test cases when the truth is that the record of them could not be read.
    except (ET.ParseError, OSError) as exc:
        if not quiet:
            console.print(
                f"[bold yellow]⚠ Could not read the failure report at {junit_path}: "
                f"{type(exc).__name__}: {exc}[/bold yellow]"
            )
            console.print(
                "[dim]The list of failed test cases below is therefore incomplete or empty. "
                "This does not change the gate's verdict, which comes from pytest's exit "
                "code.[/dim]"
            )
        return []

    return failed_tests


def describe_installed_hook_drift(
    hooks_dir: Path | None = None, expected: str | None = None
) -> list[str]:
    """Report whether the *installed* pre-commit hook matches the tracked constant.

    Three copies of this hook exist: the `PRE_COMMIT_HOOK` constant, the tracked mirror
    `swarm/pre-commit.hook`, and the file git actually runs at `.git/hooks/pre-commit`.
    An existing test pins mirror-to-constant, so **the live copy was checked by nothing** —
    and on 2026-09-03 it was edited in place to add a documentation-only fast path (#285).
    The two sizes are deliberately not written down here: the constant's length changes
    whenever the hook does, and a number in prose goes stale silently. The report below
    measures both at call time and prints them.

    That gap matters more now that the hook carries the commit-identity refusal: a control
    that lives only in the constant is a control that is **not installed**. This surfaces
    the difference on every gate run, because nothing else tells a builder to re-run
    `./ucx setup`.

    Deliberately a report and not a failure. The installed hook is shared by every
    worktree, so making the gate red would block every builder in the repository for a
    state none of them created and only `./ucx setup` can clear — and `setup` is itself a
    shared-state change. Returns lines to print; empty when there is nothing to say.

    Args:
        hooks_dir: Directory to inspect. Defaults to the resolved git hooks directory.
        expected: Content to compare against. Defaults to the installed-hook constant.
    """
    # Private by name, and deliberately reused: this must inspect the directory git will
    # actually run hooks from (it honours `core.hooksPath`), which is exactly what that
    # resolver answers. A second implementation here could disagree with the installer.
    from uclone_x.cli.main import (
        PRE_COMMIT_HOOK,
        _resolve_git_hooks_dir,  # pyright: ignore[reportPrivateUsage]
    )

    expected_content = PRE_COMMIT_HOOK if expected is None else expected
    if hooks_dir is not None:
        resolved = hooks_dir
    else:
        # Broad by intent, and narrow in consequence: this is a best-effort *report*, and
        # a reporter that can break the gate it reports on is worse than no reporter. The
        # resolver shells out to git, so it can fail in ways that have nothing to do with
        # hook drift — no git on PATH, a partially-initialised repository, or a caller that
        # has replaced `subprocess.run` (which is how this first surfaced: six gate tests
        # mock it, and the mock's `stdout` is None). Silence on failure is correct here;
        # the only cost of a missed report is the message this run would have printed.
        try:
            resolved = _resolve_git_hooks_dir()
        except Exception:  # noqa: BLE001 - see above; reporting must never fail the gate
            return []
    if resolved is None:
        return []

    installed_path = resolved / "pre-commit"
    try:
        installed = installed_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # Absent is not drift: a fresh clone has no hook until `./ucx setup` runs.
        return []
    if installed == expected_content:
        return []

    return [
        "[bold yellow]! The installed pre-commit hook differs from the tracked source.[/bold yellow]",
        f"[yellow]  installed: {installed_path} ({len(installed.encode('utf-8'))} bytes)[/yellow]",
        f"[yellow]  tracked:   PRE_COMMIT_HOOK / swarm/pre-commit.hook "
        f"({len(expected_content.encode('utf-8'))} bytes)[/yellow]",
        "[yellow]  Git runs the installed copy, so any guard added to the tracked source "
        "is NOT in effect[/yellow]",
        "[yellow]  until you run [bold]./ucx setup[/bold]. See #285.[/yellow]",
    ]


# Gate-pass records (#966).
#
# `<git-common-dir>/gate-pass/<sha>` is what the Builder Manager's pre-merge check (c) and
# the swarm's merge observer read. Until #966 the pre-push hook wrote it after running the
# gate on every branch push; the owner ruling of 2026-09-15 (decision ledger DL-020) took the
# gate off push, so the gate now writes its own record. A record is read by someone about to
# merge on it, so it is written only when all three hold:
#
# 1. The run passed.
# 2. The run was the full default gate: `./ucx test check`, including with `--all` (a no-op
#    synonym for it), `-fe` or `--no-fail-fast`. `--fast` drops the browser suite and
#    `--skip-tests` drops every test, so neither verified what a merge needs. The named tiers
#    (`unit`, `e2e`, ...) never call this.
# 3. The tree was the commit: `HEAD` resolves, `git status --porcelain` is empty both before
#    and after the run, and `HEAD` did not move during it. Empty includes untracked files
#    that are not ignored. The gate reads more than the Python paths it names (pytest collects
#    from the rootdir, fitness tests read `docs/`, vitest reads `frontend/`), so a list of
#    "paths the gate reads" would go stale with the next test that reads a new one. Strictness
#    costs one unrecorded run with its reason printed; looseness costs a record for a commit
#    whose tree nobody ran.
GATE_PASS_SCOPES: Final[frozenset[str]] = frozenset({"gate"})


@dataclass(frozen=True)
class CommittedTree:
    """What `snapshot_committed_tree` saw: `head` when the tree is exactly HEAD, else `problem`."""

    head: str | None
    problem: str | None


def gate_run_can_record(test_scope: str, *, skip_tests: bool) -> bool:
    """Whether a passing run with these options is the full gate a record may claim."""
    return not skip_tests and test_scope in GATE_PASS_SCOPES


def _git_stdout(args: list[str], cwd: Path | None) -> str | None:
    """Run git and return its stdout, or None when it could not answer.

    `GIT_*` is stripped for the reason `_resolve_git_hooks_dir` in `cli/main.py` gives: with a
    hook's exported `GIT_DIR`, git answers about that repository whatever the cwd.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, env=env, check=False
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def snapshot_committed_tree(cwd: Path | None = None) -> CommittedTree:
    """Report the HEAD sha if the working tree is exactly that commit, or why it is not."""
    head = (_git_stdout(["rev-parse", "--verify", "--quiet", "HEAD"], cwd) or "").strip()
    if not head:
        return CommittedTree(None, "there is no commit here (not a git checkout, or no HEAD)")
    status = _git_stdout(["status", "--porcelain", "--untracked-files=normal"], cwd)
    if status is None:
        return CommittedTree(None, "`git status` failed, so the tree was not shown to be clean")
    changed = [line.strip() for line in status.splitlines() if line.strip()]
    if changed:
        return CommittedTree(
            None,
            f"the tree is not clean ({len(changed)} modified, staged or untracked "
            f"path(s); first: {changed[0]})",
        )
    return CommittedTree(head, None)


def record_gate_pass(
    *,
    exit_code: int,
    test_scope: str,
    skip_tests: bool,
    before: CommittedTree | None,
    cwd: Path | None = None,
) -> str:
    """Write `<git-common-dir>/gate-pass/<HEAD>` if this run earns it; return one line saying so.

    The rule is in the comment above `GATE_PASS_SCOPES`. Every refusal names its reason:
    a missing record at merge time sends someone to run the gate again, and they need to
    know which of the three conditions to fix.

    Args:
        exit_code: The gate's exit code.
        test_scope: The scope the gate ran.
        skip_tests: Whether the run skipped the tests.
        before: `snapshot_committed_tree()` taken before the gate ran; None if none was taken.
        cwd: Directory to ask git about. Defaults to the process cwd.
    """
    if not gate_run_can_record(test_scope, skip_tests=skip_tests):
        what = "--skip-tests" if skip_tests else ("--fast" if test_scope == "fast" else test_scope)
        return (
            f"Gate pass not recorded: a {what} run is not the full gate; "
            "only `./ucx test check` records one."
        )
    if exit_code != 0:
        return "Gate pass not recorded: the gate failed."
    if before is None:
        return "Gate pass not recorded: the tree was not inspected before the run."
    head = before.head
    if head is None:
        return f"Gate pass not recorded: before the run, {before.problem}."
    after = snapshot_committed_tree(cwd)
    if after.head is None:
        return f"Gate pass not recorded: after the run, {after.problem}."
    if after.head != head:
        return (
            f"Gate pass not recorded: HEAD moved during the run ({head[:12]} -> {after.head[:12]})."
        )
    common = (
        _git_stdout(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd) or ""
    ).strip()
    if not common:
        return "Gate pass not recorded: git named no common directory to record in."
    record = Path(common) / "gate-pass" / head
    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.touch()
    except OSError as exc:
        return f"Gate pass not recorded: could not write {record} ({exc})."
    return f"Gate pass recorded for {head} ({record})."


def run_quality_gate(
    *,
    quiet: bool = False,
    skip_tests: bool = False,
    check_frontend: bool = False,
    fail_fast: bool = True,
    test_scope: str = "gate",
    junit_path: Path = JUNIT_REPORT_PATH,
    failure_log_path: Path = FAILURE_LOG_PATH,
    serial: bool = False,
) -> int:
    """Run the automated quality gates, in order.

    Ruff format and lint, Pyright strict typing, Pytest, the frontend vitest suite, and
    the frontend production build when it is requested.

    Scope policy:
        Python static checks cover all top-level directories containing Python code:
        `src`, `tests`, `swarm`, and `scripts`. Directories outside this set
        (such as `frontend/`, `credentials/`, and `docs/`) are not covered by the
        static checks. The frontend's vitest suite runs as its own stage in the `gate`,
        `fast` and `all` scopes (#915), and so does the check that the committed bundle is
        what the source builds, which builds into a temporary directory (#878). The
        frontend production build into the committed directory runs only when explicitly
        requested (`check_frontend=True`), because it writes committed files.

    Args:
        junit_path: Where pytest writes its report and where the failure extraction reads
            it from. Defaults to the cwd-relative module constant; a test MUST override it
            with a `tmp_path` value, because leaving the default makes the run read the
            artifact of whatever ran in that directory before it (#436).
        failure_log_path: Where the durable failure history is appended. Same rule.
        serial: Run the pytest stage in one process even for a parallel scope (#967).

    Returns:
        0 if all quality gates pass, or the exit code of the first failed step.
    """
    # `ucx` is installed from PyPI as well as run from a checkout, and outside a
    # checkout there is nothing to gate. That is a usage error with a one-line
    # answer, so it is reported and returned like any other failed step. Letting
    # the exception escape printed a Python traceback at the first command a new
    # user reaches for after `--help`, which is the failure mode P6 names: the
    # message was right and the presentation made it look like a crash.
    try:
        check_paths = resolve_check_paths()
    except FileNotFoundError as exc:
        # Printed even when quiet. `quiet` suppresses the gate's progress
        # reporting; this is the gate declining to run at all, and a non-zero
        # exit with nothing on any stream is the silent failure P6 forbids.
        print(f"ucx: {exc}", file=sys.stderr)
        return SCOPE_RESOLUTION_EXIT_CODE

    # Before the gate's first write: a process that shares this pipe, such as the `ssh` of a
    # `git push`, may already have made it non-blocking (#993; see `restore_blocking_stdio`).
    restored_at_start = restore_blocking_stdio()
    if restored_at_start and not quiet:
        print(_describe_restored(restored_at_start, "when the gate started"), file=sys.stderr)

    scopes_str = ", ".join(check_paths)
    if not quiet:
        console.print("[bold cyan]══════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]  UClone-X Quality Gate: Static Verification     [/bold cyan]")
        console.print("[bold cyan]══════════════════════════════════════════════════[/bold cyan]")
        console.print(
            f"[dim]Scope: Python static checks cover ({scopes_str}). "
            "Non-Python paths (e.g. frontend, docs) are excluded from them; the frontend "
            "vitest suite runs as stage 6.[/dim]"
        )
    first_failure = 0

    # 1. Lockfile freshness.
    #
    # First, and before any tool that reads the environment. Every stage after this one is
    # measured against the packages actually installed; if `uv.lock` no longer agrees with
    # `pyproject.toml`, nobody knows which of the two that environment came from, so a
    # green Pyright or a green suite is a result about an unidentified dependency set. The
    # ordering also follows the gate's own economics: this costs well under a second and
    # `fail_fast` is the default, so a builder with a stale lock is told in one second
    # rather than after seventy, and is told the one thing they can fix without reading a
    # diff. It is deliberately not last for the same reason it is not optional — a check
    # that only runs when everything else passed is a check that stops running as soon as
    # the tree is red.
    if not quiet:
        console.print("\n[bold]1. Checking uv.lock is in sync with pyproject.toml...[/bold]")
    lock_status, lock_lines = check_lockfile_freshness()
    if not quiet:
        for line in lock_lines:
            console.print(line)
    if lock_status in ("stale", "uv-missing"):
        if fail_fast:
            return _LOCKFILE_FAILURE_EXIT
        if first_failure == 0:
            first_failure = _LOCKFILE_FAILURE_EXIT
    elif lock_status == "fresh" and not quiet:
        console.print("[green]✔ uv.lock matches pyproject.toml.[/green]")

    # 1b. Environment provenance.
    #
    # Numbered `1b` for the same reason `5b` is: it belongs with the stage above it. The
    # lockfile check asks whether the declared dependency set is the installed one; this
    # asks whether the installed environment resolves *this tree's* `uclone_x` in a child
    # process. Together they are the one question "is this environment fit to measure this
    # tree", and every stage below is an answer about whatever they found.
    #
    # It must precede Ruff and Pyright, not follow them. The worktree-local `.venv` case
    # presents as a wall of unresolved-import errors from Pyright against code the builder
    # never touched — a reading an author is primed to accept as their own doing. The
    # figure recorded on #679 is 258; treat it as the count one builder saw in one venv,
    # not as a re-derivable constant, because it depends on what that venv happened to
    # hold. The ordering argument does not rest on the number: any count above zero
    # against untouched code is the misdirection this stage exists to pre-empt. Two
    # subprocesses costing well under a second buy the difference between that and a named
    # error, and `fail_fast` is the default, so the cost is paid once.
    #
    # It refuses rather than warning for the blocking statuses because the failure this
    # comes from did not present as a red gate. It presented as a **green** one, for hours,
    # while subprocess imports were resolving to a tree twenty-one modules behind `main`.
    # A warning printed above a green verdict is the same silent pass in a different font
    # (P6). Two statuses are nonetheless non-blocking, and neither is an exception to that
    # reasoning: `ambient-foreign` is the reading that is not satisfiable in more than one
    # worktree at a time, and `worktree-venv-declared` is the arrangement AGENTS.md:82
    # explicitly permits, which this stage used to refuse while calling it "never
    # intended" (#968). See `uclone_x.cli.environment_provenance` for both.
    if not quiet:
        console.print("\n[bold]1b. Checking which `uclone_x` a child process gets...[/bold]")
    env_status, env_lines = check_environment_provenance()
    if not quiet:
        for line in env_lines:
            console.print(line)
    if env_status in BLOCKING_STATUSES:
        if fail_fast:
            return _ENVIRONMENT_FAILURE_EXIT
        if first_failure == 0:
            first_failure = _ENVIRONMENT_FAILURE_EXIT

    # 2. Ruff Format Check
    if not quiet:
        console.print(f"\n[bold]2. Running Ruff Format Check ({scopes_str})...[/bold]")
    res_format = run_stage(["ruff", "format", "--check", *check_paths], quiet=quiet)
    if res_format.returncode != 0:
        if not quiet:
            console.print("[bold red]✖ Ruff format check failed.[/bold red]")
        if fail_fast:
            return res_format.returncode
        if first_failure == 0:
            first_failure = res_format.returncode
    elif not quiet:
        console.print("[green]✔ Ruff format check passed.[/green]")

    # 3. Ruff Linter
    if not quiet:
        console.print(f"\n[bold]3. Running Ruff Linter ({scopes_str})...[/bold]")
    res_lint = run_stage(["ruff", "check", *check_paths], quiet=quiet)
    if res_lint.returncode != 0:
        if not quiet:
            console.print("[bold red]✖ Ruff linter check failed.[/bold red]")
        if fail_fast:
            return res_lint.returncode
        if first_failure == 0:
            first_failure = res_lint.returncode
    elif not quiet:
        console.print("[green]✔ Ruff linter check passed.[/green]")

    # 4. Pyright Strict Typing
    if not quiet:
        console.print(f"\n[bold]4. Running Pyright Strict Typing Check ({scopes_str})...[/bold]")
    res_pyright = run_stage(["pyright"], quiet=quiet)
    if res_pyright.returncode != 0:
        if not quiet:
            console.print("[bold red]✖ Pyright type check failed.[/bold red]")
        if fail_fast:
            return res_pyright.returncode
        if first_failure == 0:
            first_failure = res_pyright.returncode
    elif not quiet:
        console.print("[green]✔ Pyright strict typing passed with 0 errors.[/green]")

    # 5. Pytest Test Suite with Coverage >= 70%
    #
    # A parallel scope needs pytest-xdist. Without it the stage is refused, never quietly run
    # serially: that would turn an out-of-date environment into a slower gate nobody reads as
    # a symptom (#967). Printed even when quiet, like the refusal above — a non-zero exit with
    # nothing on any stream is the silent failure P6 forbids.
    runner_missing = (
        not skip_tests
        and pytest_runs_in_parallel(test_scope, serial=serial)
        and not parallel_runner_available()
    )
    if runner_missing:
        for line in describe_missing_parallel_runner(test_scope):
            print(line, file=sys.stderr)
        if fail_fast:
            return _PARALLEL_RUNNER_MISSING_EXIT
        if first_failure == 0:
            first_failure = _PARALLEL_RUNNER_MISSING_EXIT
    elif not skip_tests:
        steps = build_pytest_steps(test_scope, junit_path=junit_path, serial=serial)
        # Every coverage claim below is conditional on this. A scope run with `--no-cov`
        # measured nothing, so reporting a floor for it is a claim about a check that did
        # not run (#882).
        measures_coverage = test_scope not in _NO_COVERAGE_SCOPES
        if not quiet:
            scope_label = SCOPE_LABELS[test_scope]
            coverage_note = " (Branch Coverage >= 70%)" if measures_coverage else ""
            console.print(f"\n[bold]5. Running Pytest {scope_label}{coverage_note}...[/bold]")
        junit_path.parent.mkdir(parents=True, exist_ok=True)

        # A step that fails stops the rest under `fail_fast`, as any other stage does. The
        # result judged below is the first step that failed, or the last step if none did.
        started = time.time()
        step_results: list[subprocess.CompletedProcess[bytes]] = []
        for number, step in enumerate(steps, start=1):
            if len(steps) > 1 and not quiet:
                console.print(f"[bold]5.{number} Pytest step: {step.label}[/bold]")
            step_results.append(run_stage(list(step.argv), quiet=quiet))
            if step_results[-1].returncode != 0 and fail_fast:
                break
        if len(steps) > 1:
            left_out = merge_junit_reports(
                [step.junit_path for step in steps], junit_path, written_after=started - 1.0
            )
            if left_out and not quiet:
                console.print(
                    "[dim]Not in the combined report (the step did not run, or wrote no "
                    f"report): {', '.join(str(path) for path in left_out)}[/dim]"
                )
        res_pytest = next((r for r in step_results if r.returncode != 0), step_results[-1])
        if res_pytest.returncode == _PYTEST_NO_TESTS_COLLECTED:
            if not quiet:
                console.print(
                    f"[bold red]✖ No tests are marked '{test_scope}' — this tier is "
                    f"empty.[/bold red]"
                )
                console.print(
                    "[dim]An empty tier cannot verify anything, so it is reported as a "
                    "failure rather than a pass. See the quality-assurance design note "
                    "in the development repository, §8, for the seeding steps."
                    "[/dim]"
                )
            if fail_fast:
                return res_pytest.returncode
            if first_failure == 0:
                first_failure = res_pytest.returncode  # empty tier
        elif res_pytest.returncode != 0:
            failed_tests = record_and_extract_failures(
                junit_path, failure_log_path=failure_log_path, quiet=quiet
            )
            if not quiet:
                console.print(
                    "[bold red]✖ Pytest suite or branch coverage check (< 70%) failed.[/bold red]"
                    if measures_coverage
                    else "[bold red]✖ Pytest suite failed (no coverage measured).[/bold red]"
                )
                if failed_tests:
                    console.print(
                        "[bold red]── Failed Test Cases ──────────────────────────────[/bold red]"
                    )
                    for ft in failed_tests:
                        console.print(f"  [red]• {ft}[/red]")
                    console.print(
                        f"[dim]Durable failure record saved to: {failure_log_path}[/dim]\n"
                    )
            if fail_fast:
                return res_pytest.returncode
            if first_failure == 0:
                first_failure = res_pytest.returncode  # test failures
        elif not quiet:
            console.print(
                "[green]✔ Pytest tests passed with >= 70% branch coverage.[/green]"
                if measures_coverage
                else "[green]✔ Pytest tests passed (no coverage measured).[/green]"
            )

    # 5b. Per-package coverage floors.
    #
    # Skipped when the selected scope disabled coverage: there is no fresh data file, and
    # reporting a floor met by leftover data from a previous run is worse than reporting
    # nothing at all (#436 is this same class of bug).
    if not skip_tests and not runner_missing and test_scope not in _NO_COVERAGE_SCOPES:
        if not quiet:
            console.print("\n[bold]5b. Checking Per-Package Coverage Floors...[/bold]")
        shortfalls = check_package_coverage_floors()
        if shortfalls:
            if not quiet:
                for package, measured, floor in shortfalls:
                    where = "could not be measured" if measured < 0 else f"{measured}%"
                    console.print(
                        f"[bold red]✖ uclone_x/{package}: {where} is below its "
                        f"{floor}% floor[/bold red]"
                    )
            if fail_fast:
                return 1
            if first_failure == 0:
                first_failure = 1
        elif not quiet:
            console.print(
                f"[green]✔ All {len(PACKAGE_COVERAGE_FLOORS)} package coverage floors met.[/green]"
            )

    # 6. Frontend vitest suite — part of every scope a commit is measured against (#915).
    #
    # Gated on the exit code, not on the status being `failed`: `deps-missing` and
    # `npm-missing` carry a non-zero code precisely so that a suite which could not run
    # turns the gate red instead of disappearing from it.
    if frontend_suite_selected(test_scope, skip_tests=skip_tests, check_frontend=check_frontend):
        if not quiet:
            console.print("\n[bold]6. Running Frontend Vitest Suite...[/bold]")
        _, fe_code, fe_lines = run_frontend_suite(quiet=quiet)
        if not quiet:
            for line in fe_lines:
                # `soft_wrap`: the remedy is a shell command, and a hard wrap breaks the paste.
                console.print(line, soft_wrap=True)
        if fe_code != 0:
            if fail_fast:
                return fe_code
            if first_failure == 0:
                first_failure = fe_code

    # 6b. The committed UI bundle is what the source builds (#878).
    #
    # Same selection as stage 6, for the same reason: a check behind a flag is a check that
    # does not run. Since #966 no hook runs this at all — the pre-push hook only refuses
    # `refs/heads/main`, and the pre-commit hook runs `--skip-tests`, which selects neither
    # this stage nor stage 6 — so a hand-run `./ucx test check` (before a PR, and again at
    # the head before merge) is the only thing that reaches it. It costs about 1.5 s. It
    # builds into a temporary directory and never writes `src/uclone_x/ui_static`, and it
    # runs before stage 7, which does write it — so under `-fe` it still judges the bundle
    # as it was found.
    if frontend_suite_selected(test_scope, skip_tests=skip_tests, check_frontend=check_frontend):
        if not quiet:
            console.print(
                "\n[bold]6b. Checking src/uclone_x/ui_static is what frontend/ builds...[/bold]"
            )
        _, bundle_code, bundle_lines = check_bundle_freshness()
        if not quiet:
            for line in bundle_lines:
                console.print(line, soft_wrap=True)
        if bundle_code != 0:
            if fail_fast:
                return bundle_code
            if first_failure == 0:
                first_failure = bundle_code

    # 7. Frontend production build — only on request (`-fe`). It writes the committed
    # `src/uclone_x/ui_static`, so a default gate that built would rewrite tracked files on
    # every push. Stage 6b answers the freshness question without writing anything.
    if check_frontend:
        if not quiet:
            console.print("\n[bold]7. Running Frontend TypeScript Build Check...[/bold]")
        # `NODE_ENV` and `BROWSERSLIST` pinned as stage 6b pins them: this build writes the
        # committed bundle, and under a different value of either it writes different bytes.
        res_fe = run_stage(
            ["npm", "run", "build"],
            cwd="frontend",
            env={**os.environ, **BUILD_ENVIRONMENT},
            quiet=quiet,
        )
        if res_fe.returncode != 0:
            if not quiet:
                console.print("[bold red]✖ Frontend build failed.[/bold red]")
            if fail_fast:
                return res_fe.returncode
            if first_failure == 0:
                first_failure = res_fe.returncode
        elif not quiet:
            console.print("[green]✔ Frontend build passed.[/green]")

    if first_failure != 0:
        if not quiet:
            console.print("\n[bold red]✖ Quality gate failed.[/bold red]\n")
        return first_failure

    if not quiet:
        console.print(
            "\n[bold green]✨ ALL SYSTEMS GO: Quality gate passed with flying colors![/bold green]\n"
        )

    return 0
