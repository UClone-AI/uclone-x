"""Whether the merge gate runs the browser suite, and the records that decide it.

The browser suite runs after everything else, on a few workers of its own (#967), and it is
the part of `./ucx test check` that grew: a real Chromium per case. Running all of it on every PR
spends that time on diffs that cannot change what the browser renders. So the gate decides,
per diff against the merge base with `origin/main`:

* **full** — a `[browser_suite]` or `[whole_suite]` rule in `tests/scope-rules.toml` matches a
  changed path; or the browser suite imports a changed module, directly or not; or a
  non-Python file under `src/` changed (package data); or no full
  pass on main is fresh (the lazy nightly); or the diff cannot be computed, or the rules
  cannot be read. Doubt runs everything.
* **files** — a fresh full pass on main vouches for the suite, and the diff changes browser
  test files: those files run, because a changed test always runs.
* **none** — a fresh pass vouches for the suite and the diff changes no browser test.

Records live under the git common directory, beside the gate-pass records, keyed by a commit
on main — never by a PR head, which a squash merge never makes an ancestor of main:

* `browser-pass/<sha>` — the full suite passed on `<sha>`, or on a PR tree whose merge base is
  `<sha>` (the file says which). Written by any clean full run that passed with nothing
  deselected. A pass on a PR tree vouches for merges built on `<sha>`; it is not a test of
  `<sha>`'s own tree, so it neither clears a red record nor excuses the nightly.
* `browser-red/<sha>` — the full suite failed on `<sha>` itself, listing the failing node ids.
  Written only by `--full-browser` on exactly `origin/main` (the nightly), and only when the
  report names every failure as a test in a file: a failure on a PR tree is that PR's to fix,
  and a crash that named no test tested nothing a record could vouch for.

A fresh red record vouches for the rest of the suite and names the tests main already fails.
A full run deselects those — unless the diff changes their file — and says so: a red main
must not fail every merge until it is fixed, and a test deselected in silence is a test
nobody knows is off.
"""

from __future__ import annotations

import os
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from uclone_x.cli.scope_rules import ScopeRules, ScopeRulesError, load_scope_rules

PASS_DIR: Final[str] = "browser-pass"
# The body of a pass record written by a run on the commit itself, not on a PR tree built on it.
_ON_THE_COMMIT: Final[str] = "tested-at the commit itself"
RED_DIR: Final[str] = "browser-red"

BrowserMode = Literal["full", "files", "none"]


@dataclass(frozen=True)
class BrowserRecord:
    """The newest fresh record on main's recent history: what it says, and about which commit."""

    kind: Literal["pass", "red"]
    sha: str
    age_hours: float
    red_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BrowserPlan:
    """What the merge gate's browser step runs, and why. `describe()` is what it prints."""

    mode: BrowserMode
    reason: str
    files: tuple[str, ...] = ()
    deselect: tuple[str, ...] = ()
    merge_base: str | None = None
    # Set when the plan was forced by `--full-browser`: only such a run may write a red record.
    forced: bool = False
    notes: tuple[str, ...] = field(default=())

    def describe(self) -> list[str]:
        """Lines for the gate's output: the decision, its reason, and anything deselected."""
        if self.mode == "full":
            head = f"Browser suite: full — {self.reason}."
        elif self.mode == "files":
            head = (
                f"Browser suite: the {len(self.files)} changed browser test file(s) only — "
                f"{self.reason}."
            )
        else:
            head = f"Browser suite: not run — {self.reason}."
        lines = [head, *self.notes]
        if self.deselect:
            lines.append(
                f"Deselected {len(self.deselect)} test(s) main already fails (a red nightly "
                "record); run them with `./ucx test e2e` once main is fixed:"
            )
            lines.extend(f"  - {node}" for node in self.deselect)
        return lines


def _git(args: list[str], cwd: Path | None) -> str | None:
    """git's stdout, or None when it could not answer. `GIT_*` stripped, as `_git_stdout` does."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, env=env, check=False
        )
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def common_dir(cwd: Path | None = None) -> Path | None:
    """The git common directory, where records are shared by every worktree."""
    out = (_git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd) or "").strip()
    return Path(out) if out else None


def newest_fresh_record(
    common: Path,
    history: list[str],
    *,
    max_age_hours: int,
    now: float | None = None,
) -> BrowserRecord | None:
    """The first fresh record along `history` (newest commit first), or None.

    On one commit a red record outranks a pass, unless the pass is on the commit itself and
    newer: a pass on a PR tree built on the commit is not main's tree, and the PR may be the
    one that fixes the red test.
    """
    clock = time.time() if now is None else now
    limit = max_age_hours * 3600
    for sha in history:
        pass_path, red_path = common / PASS_DIR / sha, common / RED_DIR / sha
        pass_age, red_age = _age(pass_path, clock), _age(red_path, clock)
        pass_fresh = pass_age is not None and pass_age <= limit
        red_fresh = red_age is not None and red_age <= limit
        if red_fresh and red_age is not None:
            beaten = (
                pass_fresh
                and pass_age is not None
                and pass_age < red_age
                and _reads(pass_path).strip() == _ON_THE_COMMIT
            )
            if not beaten:
                ids = tuple(line.strip() for line in _reads(red_path).splitlines() if line.strip())
                return BrowserRecord("red", sha, red_age / 3600, ids)
        if pass_fresh and pass_age is not None:
            return BrowserRecord("pass", sha, pass_age / 3600)
    return None


def _age(path: Path, clock: float) -> float | None:
    try:
        return clock - path.stat().st_mtime
    except OSError:
        return None


def _reads(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def plan_browser_suite(
    *,
    repo: Path,
    base: str = "origin/main",
    full: bool = False,
    rules: ScopeRules | None = None,
    now: float | None = None,
) -> BrowserPlan:
    """Decide the merge gate's browser step for the tree at `repo`. See the module docstring."""
    # Imported here: `changed_scope` imports `quality_gate`, which imports this module.
    from uclone_x.cli.changed_scope import changed_paths, first_path_imported_from

    if full:
        merge_base = (_git(["merge-base", base, "HEAD"], repo) or "").strip() or None
        return BrowserPlan("full", "--full-browser", merge_base=merge_base, forced=True)
    try:
        loaded = rules or load_scope_rules(repo)
    except ScopeRulesError as exc:
        return BrowserPlan("full", f"the scope rules could not be read ({exc})")
    merge_base = (_git(["merge-base", base, "HEAD"], repo) or "").strip()
    if not merge_base:
        return BrowserPlan("full", f"git names no merge base with {base}")
    try:
        changed = changed_paths(base, cwd=repo)
    except RuntimeError as exc:
        return BrowserPlan("full", f"the diff could not be computed ({exc})", merge_base=merge_base)

    changed_tests = tuple(
        p
        for p in changed
        if loaded.is_browser_test(p) and p.endswith(".py") and (repo / p).is_file()
    )
    common = common_dir(repo)
    history = (
        _git(["rev-list", "--first-parent", f"--max-count={loaded.max_merges}", merge_base], repo)
        or ""
    ).split()
    record = (
        newest_fresh_record(common, history, max_age_hours=loaded.max_age_hours, now=now)
        if common is not None
        else None
    )
    deselect: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    if record is not None and record.kind == "red":
        # A test whose own file the diff changes is the PR's to run, red record or not.
        deselect = tuple(
            node for node in record.red_ids if node.split("::", 1)[0] not in changed_tests
        )
        notes = (
            f"Main is known red: the nightly on {record.sha[:12]} "
            f"({record.age_hours:.1f} h ago) failed {len(record.red_ids)} browser test(s).",
        )

    trigger = next((p for p in changed if loaded.reaches_browser_suite(p)), None)
    reason = f"{trigger} can change what the browser renders"
    if trigger is None:
        # Package data (personas, bundled stories, config) is loaded at run time by code the
        # import graph cannot see through, so any non-Python file under `src/` counts.
        trigger = next((p for p in changed if p.startswith("src/") and not p.endswith(".py")), None)
        reason = f"{trigger} is package data the served app may load"
    if trigger is None:
        # The rules name what imports cannot show; the import graph names the rest. A module
        # the served app imports changes what the page does, whether or not a rule says so.
        others = [p for p in changed if not loaded.is_browser_test(p)]  # those run themselves
        trigger = first_path_imported_from(repo, loaded.browser_tests, others)
        reason = f"the browser suite imports {trigger}"
    if trigger is not None:
        return BrowserPlan(
            "full",
            reason,
            deselect=deselect,
            merge_base=merge_base,
            notes=notes,
        )
    if record is None:
        return BrowserPlan(
            "full",
            f"no full pass on main is fresh (none on the last {loaded.max_merges} merges within "
            f"{loaded.max_age_hours} h), so this run is the nightly",
            merge_base=merge_base,
        )
    vouch = (
        f"a full pass on {record.sha[:12]} ({record.age_hours:.1f} h ago) vouches for the suite"
        if record.kind == "pass"
        else f"the nightly on {record.sha[:12]} vouches for the rest of the suite"
    )
    if changed_tests:
        return BrowserPlan(
            "files",
            f"no browser rule matches the diff and {vouch}",
            files=changed_tests,
            merge_base=merge_base,
            notes=notes,
        )
    return BrowserPlan(
        "none",
        f"no browser rule matches the diff and {vouch}",
        merge_base=merge_base,
        notes=notes,
    )


def junit_node_ids(report: Path, repo: Path) -> list[str]:
    """Failed and errored cases of a junit report, as pytest node ids (`path.py::name`).

    junit names a case by a dotted `classname` that joins the module path and any class; the
    module part is the longest dotted prefix that is a file in `repo`.
    """
    try:
        root = ET.parse(report).getroot()
    except (ET.ParseError, OSError):
        return []
    ids: list[str] = []
    for case in root.iter("testcase"):
        if case.find("failure") is None and case.find("error") is None:
            continue
        parts = case.attrib.get("classname", "").split(".")
        name = case.attrib.get("name", "")
        for cut in range(len(parts), 0, -1):
            module = "/".join(parts[:cut]) + ".py"
            if (repo / module).is_file():
                ids.append("::".join([module, *parts[cut:], name]))
                break
        else:
            ids.append(name)
    return ids


def record_browser_result(
    plan: BrowserPlan,
    *,
    passed: bool,
    report: Path,
    repo: Path,
    base: str = "origin/main",
) -> str | None:
    """Write the record a finished full browser step earns, and return the line saying so.

    Returns None when the run earns no record and there is nothing worth saying: a run that
    was not full. Every other refusal names its reason, as the gate-pass record's do.
    """
    if plan.mode != "full":
        return None
    status = _git(["status", "--porcelain", "--untracked-files=normal"], repo)
    if status is None or status.strip():
        return "Browser record not written: the tree is not a clean commit."
    head = (_git(["rev-parse", "HEAD"], repo) or "").strip()
    main = (_git(["rev-parse", base], repo) or "").strip()
    common = common_dir(repo)
    if not head or not plan.merge_base or common is None:
        return "Browser record not written: git could not name HEAD, the merge base or its store."

    if passed and plan.deselect:
        return (
            "Browser record not written: tests were deselected, so this pass does not cover "
            "the suite."
        )
    if passed:
        path = common / PASS_DIR / plan.merge_base
        body = f"tested-at {head}\n" if head != plan.merge_base else f"{_ON_THE_COMMIT}\n"
        if body.strip() != _ON_THE_COMMIT and _reads(path).strip() == _ON_THE_COMMIT:
            # The pass on the commit itself is what outranks an older red record and lets the
            # nightly skip the commit; a PR tree's pass must not replace it, nor refresh it.
            return f"Browser pass record kept: {path.name} already has a pass on the commit itself."
    elif plan.forced and head == main:
        ids = junit_node_ids(report, repo)
        # A red record vouches for every test it does not name. A step that failed without
        # naming a test in a file — a crash, a collection error, a missing browser, only the
        # coverage threshold — tested nothing it could vouch for.
        if not ids or any("::" not in node for node in ids):
            return (
                "Browser red record not written: the browser step failed without naming each "
                "failing test in its file, so it vouches for nothing."
            )
        path = common / RED_DIR / head
        body = "".join(f"{node}\n" for node in ids)
    elif plan.forced:
        return (
            f"Browser red record not written: HEAD is not {base} "
            "(it moved during the run, or this is not main's tree)."
        )
    else:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    except OSError as exc:
        return f"Browser record not written: {path} ({exc})."
    kind = "pass" if passed else "red"
    return f"Browser {kind} record written for {path.name} ({path})."
