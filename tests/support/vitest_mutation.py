"""Run one declared frontend mutation and say whether the declaring vitest test dies (#1077).

This is the vitest half of `swarm.core.mutation.run_isolated_mutation`, and it is deliberately
the same shape: measure an unmutated baseline first, refuse to score a run that failed for a
reason other than the mutation, restore the file afterwards, and classify from the *named
test's* result rather than from an exit code. A frontend declaration that only ever gets an
existence check is a recorded proof nobody has replayed — five of those shipped green across
three repositories before this module existed (#1077).

## Why the declaring test is named, and not merely the file

`vitest run <file>` would go green or red for the whole file, and #1073 is exactly the case
that defeats: a declaration attached to a *new* test whose mutation was already lethal through
a *pre-existing* test in the same file. The file dies either way and the new test evidences
nothing. So the verdict is read per test, out of vitest's own JSON reporter, for the one title
the declaration sits in.

## Why the run happens in a copy of the tree

Two reasons, and the second is the one that cannot be worked around:

1.  The gate runs pytest on several workers (#967) and any of them may read a `frontend/src`
    file. A mutant in the live tree is a failure on another worker in a file nobody changed.
2.  Gate stage 6b rebuilds `frontend/` and compares it byte for byte with the committed
    `src/uclone_x/ui_static`. A suite that can write into the live `frontend/src` decides 6b
    on what the tests just wrote (#1067). It must not be able to.

`tests.support.isolated_tree.isolated_copy_of_tree` gives that copy. `node_modules` is not in
it — it is gitignored, so the copy has no dependencies — and is symlinked in from the live tree
rather than copied, which is the difference between 0.2 s and minutes.

## What this module does **not** do, stated here so nothing reads as covered that is not

It runs **vitest**, which transforms `frontend/src` on the fly. It therefore says nothing about
a **browser** test: `tests/e2e/` drives the committed bundle in `src/uclone_x/ui_static` and
never rebuilds it, so mutating a `.tsx` under it changes nothing the browser loads and the test
passes either way. That is why e2e declarations are written in the backticked, non-parsing form
with the mutation described in prose — see
`tests/e2e/test_rail_responsive_e2e.py::test_the_rail_is_closed_on_a_first_run_at_every_width`.
Making an e2e declaration lethal needs a `vite build` inside the mutation loop, which the suite
is forbidden from running (#1067) and which costs two orders of magnitude more than this does.
The declaration ratchet that calls this module refuses such a declaration outright rather than
handing it to a harness that would call it escaped.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, cast

from tests.support.frontend_build_guard import run_outside_the_live_tree

#: vitest's own JSON reporter. `--outputFile` keeps it out of the captured stdout, which still
#: carries the human-readable failure text used in a problem message.
_JSON_REPORT: Final[str] = "vitest-lethality-report.json"

#: Long enough for a cold vite transform of a big component file on a loaded machine; short
#: enough that a hung run is a failure rather than a gate that never returns.
_TIMEOUT_SECONDS: Final[float] = 300.0


class VitestOutcome(Enum):
    """How one mutation run ended, read from the declaring test's own result."""

    #: The declaring test passed unmutated and failed mutated. The only pass.
    KILLED = "killed"
    #: The declaring test passed unmutated and passed mutated. The declaration evidences nothing.
    ESCAPED = "escaped"
    #: The declaring test did not pass before the mutation was applied. Not scored.
    BASELINE_FAILED = "baseline-failed"
    #: vitest reported no test by that name, or more than one. Not scored.
    NAME_NOT_RESOLVED = "name-not-resolved"
    #: vitest could not run the file, or the run produced no report. Not scored.
    SESSION_ERROR = "session-error"


@dataclass(frozen=True)
class VitestMutationSpec:
    """One executable frontend declaration: what to break, and which test(s) must die of it."""

    #: Repo-relative path of the file the mutation is applied to.
    target_file: str
    #: The substring replaced. Must occur exactly once in `target_file`.
    search_content: str
    #: What it becomes. Empty deletes it.
    replacement_content: str
    #: Repo-relative path of the vitest file the declaration lives in.
    test_file: str
    #: The title of the `it(...)` the declaration sits in, as written — for the message.
    test_name: str
    #: Which reported titles are that case. One for a plain `it`, one per row for `it.each`.
    title_pattern: re.Pattern[str]


@dataclass(frozen=True)
class VitestMutationResult:
    """The verdict, and enough text to act on it."""

    outcome: VitestOutcome
    message: str
    #: Tests vitest reported for the file at baseline. 0 is itself a reason to refuse to score.
    tests_reported: int


class VitestTreeUnavailableError(RuntimeError):
    """The isolated tree could not be prepared, so no declaration was audited.

    Fatal on purpose, exactly as `UnresolvableChangeBaseError` is: a harness that cannot run
    and says nothing is indistinguishable from a harness that ran and found nothing, and both
    are a green gate.
    """


def prepare_tree(live_root: Path, destination: Path) -> Path:
    """An isolated copy of `live_root` at `destination`, with `frontend/node_modules` linked in.

    The link is to the live tree's own `node_modules`. vitest only reads it, and copying it
    instead is the difference between 0.2 s and minutes.
    """
    from tests.support.isolated_tree import isolated_copy_of_tree

    modules = live_root / "frontend" / "node_modules"
    if not modules.is_dir():
        raise VitestTreeUnavailableError(
            f"{modules} is absent, so no frontend mutation can be run and the declarations "
            "this change touched would be scored by a harness that never executed.\n"
            "    Run `npm ci --prefix frontend`, or symlink it from another checkout.\n"
            "    This is fatal rather than a skip: a skipped lethality check and a clean one "
            "look identical from the gate banner (#1077)."
        )
    tree = isolated_copy_of_tree(live_root, destination)
    link = tree / "frontend" / "node_modules"
    if not link.exists():
        link.symlink_to(modules)
    return tree


@dataclass(frozen=True)
class _Report:
    """One vitest run: every reported `(title, status)`, and the console text behind it."""

    results: list[tuple[str, str]]
    console: str

    def matching(self, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
        return [entry for entry in self.results if pattern.match(entry[0])]


def _vitest(tree: Path, live_root: Path, test_file: str) -> _Report:
    """Run one vitest file in `tree` and read its JSON report."""
    frontend = tree / "frontend"
    report = frontend / _JSON_REPORT
    report.unlink(missing_ok=True)
    relative = Path(test_file).relative_to("frontend").as_posix()
    # vitest is not a build, but it is still Node under a suite that refuses the toolchain by
    # program name; `run_outside_the_live_tree` is the sanctioned lift and checks the cwd.
    done = run_outside_the_live_tree(
        ["npx", "vitest", "run", relative, "--reporter=json", f"--outputFile={_JSON_REPORT}"],
        cwd=frontend,
        live_root=live_root,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )
    console = f"{done.stdout}\n{done.stderr}".strip()
    if not report.is_file():
        return _Report([], console)
    try:
        parsed = cast("dict[str, Any]", json.loads(report.read_text(encoding="utf-8")))
    except json.JSONDecodeError:  # pragma: no cover - vitest writing invalid JSON is not a case
        return _Report([], console)
    finally:
        report.unlink(missing_ok=True)

    results: list[tuple[str, str]] = []
    for suite in cast("list[dict[str, Any]]", parsed.get("testResults", [])):
        for assertion in cast("list[dict[str, Any]]", suite.get("assertionResults", [])):
            results.append((str(assertion.get("title", "")), str(assertion.get("status"))))
    return _Report(results, console)


class VitestMutationHarness:
    """Runs declared frontend mutations in one prepared tree, reusing each file's baseline.

    The baseline is per *test file* and measured once: it is the same run for every declaration
    in that file, and it is what separates "my mutation killed this test" from "this test was
    already red" — the distinction #558 added to the Python harness for the same reason.
    """

    def __init__(self, tree: Path, live_root: Path) -> None:
        self._tree = tree
        self._live_root = live_root
        self._baselines: dict[str, _Report] = {}
        #: Every vitest invocation this harness made, for the cost figure the PR has to carry.
        self.runs = 0

    def _run(self, test_file: str) -> _Report:
        self.runs += 1
        return _vitest(self._tree, self._live_root, test_file)

    def _baseline(self, test_file: str) -> _Report:
        if test_file not in self._baselines:
            self._baselines[test_file] = self._run(test_file)
        return self._baselines[test_file]

    def run(self, spec: VitestMutationSpec) -> VitestMutationResult:
        """Apply `spec`, run its file, and classify from the declaring case's own results."""
        baseline = self._baseline(spec.test_file)
        total = len(baseline.results)
        if total == 0:
            return VitestMutationResult(
                VitestOutcome.SESSION_ERROR,
                f"vitest reported no tests for {spec.test_file} before any mutation was "
                f"applied, so nothing about the declaration was measured:\n"
                f"{baseline.console[-2000:]}",
                0,
            )
        mine = baseline.matching(spec.title_pattern)
        if not mine:
            return VitestMutationResult(
                VitestOutcome.NAME_NOT_RESOLVED,
                f"no test matching {spec.test_name!r} ran in {spec.test_file}. The declaration "
                "was read as sitting inside that case; vitest reported no such test, so the "
                "mutation was not applied.",
                total,
            )
        not_passing = [entry for entry in mine if entry[1] != "passed"]
        if not_passing:
            listed = ", ".join(f"{title!r} is {status}" for title, status in not_passing)
            return VitestMutationResult(
                VitestOutcome.BASELINE_FAILED,
                f"in {spec.test_file}, {listed} before the mutation is applied, so a failure "
                "afterwards would say nothing. Fix or un-skip it first.",
                total,
            )

        target = self._tree / spec.target_file
        if not target.is_file():
            return VitestMutationResult(
                VitestOutcome.SESSION_ERROR,
                f"{spec.target_file} does not exist in the isolated tree.",
                total,
            )
        original = target.read_text(encoding="utf-8")
        occurrences = original.count(spec.search_content)
        if occurrences != 1:
            return VitestMutationResult(
                VitestOutcome.SESSION_ERROR,
                f"{spec.search_content!r} occurs {occurrences} times in {spec.target_file} "
                "(must be exactly 1), so the mutation could not be applied.",
                total,
            )
        target.write_text(
            original.replace(spec.search_content, spec.replacement_content, 1), encoding="utf-8"
        )
        try:
            mutated = self._run(spec.test_file)
        finally:
            target.write_text(original, encoding="utf-8")

        if not mutated.results:
            return VitestMutationResult(
                VitestOutcome.SESSION_ERROR,
                f"the mutant of {spec.target_file} left vitest unable to run {spec.test_file} "
                "at all. A run that reports no test says nothing about the declaration and is "
                "not a kill.\n"
                "    The usual cause is a `Becomes:` value that describes the mutation instead "
                "of being it — `(line deleted)`, or a sentence. Substituted into the source "
                "that is not TypeScript, so nothing loads. A deletion is written as an "
                "**empty** `Becomes:` value; anything else is the literal replacement text.\n"
                f"{mutated.console[-2000:]}",
                total,
            )
        after = mutated.matching(spec.title_pattern)
        if any(status == "failed" for _, status in after):
            return VitestMutationResult(VitestOutcome.KILLED, "", total)
        if after and all(status == "passed" for _, status in after):
            return VitestMutationResult(VitestOutcome.ESCAPED, "", total)
        listed = ", ".join(f"{title!r} is {status}" for title, status in after) or "nothing"
        return VitestMutationResult(
            VitestOutcome.SESSION_ERROR,
            f"under the mutation, {spec.test_name!r} in {spec.test_file} came back as {listed} "
            "rather than passed or failed. A run that says nothing is not a pass.",
            total,
        )


def discard_tree(tree: Path) -> None:
    """Best effort removal. Failing to delete scratch must not replace a verdict."""
    shutil.rmtree(tree, ignore_errors=True)
