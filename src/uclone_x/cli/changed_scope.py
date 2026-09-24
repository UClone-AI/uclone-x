"""`./ucx test changed`: the gate's checks, narrowed to what a diff can reach.

The full gate runs once per PR, at the head that merges (owner ruling 2026-09-23). Every run
before that one is this: lint and types on the changed Python files and their direct
importers, and the tests that reach a changed file — by importing it, directly or through
other modules, or by naming its path. It records no gate pass, because a selection derived
from a diff can miss a test the full run would not.

It errs wide. A change to anything every test depends on — a `conftest.py`, the shared test
support, the dependency set — selects the whole suite rather than guessing.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# A change to any of these reaches every test, so the selection is the whole suite.
_WHOLE_SUITE_FILES = frozenset({"pyproject.toml", "uv.lock", "ucx"})
_WHOLE_SUITE_PREFIXES = ("tests/support/",)

# The top-level directories importable Python lives in. Under `src` the package root is
# one level down; everywhere else the directory is itself the top-level package.
_SOURCE_ROOTS = ("src", "swarm", "evals", "oss", "scripts", "tests")
_STRIPPED_ROOT = "src"

# A change here is judged by the frontend's own suite and the bundle check.
_FRONTEND_PREFIXES = ("frontend/", "src/uclone_x/ui_static/")

# A change to a governance document is judged by the fitness functions (R3).
_FITNESS_DIR = ("tests", "fitness")

_E2E_DIR = "tests/e2e/"

# Checks over the whole tree rather than over what they import: every tracked file is
# classified for export, and every test file is declared. A new file of any kind can
# fail either, and no import edge leads to them, so they run on every diff.
_TREE_WIDE_TESTS = ("tests/unit/test_oss_export.py", "tests/unit/test_swarm_manifest.py")


@dataclass
class ChangedSelection:
    """What a diff reaches. `whole_suite` set means the selection is everything."""

    changed: list[str]
    whole_suite: str | None = None
    test_files: list[str] = field(default_factory=list[str])
    lint_files: list[str] = field(default_factory=list[str])
    typecheck_files: list[str] = field(default_factory=list[str])
    frontend: bool = False


def changed_paths(base: str = "origin/main", cwd: Path | None = None) -> list[str]:
    """Paths that differ from the merge base with `base`: committed, staged, unstaged, new.

    Raises:
        RuntimeError: when git cannot name a merge base, so no diff can be taken.
    """

    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
        return done.stdout

    merge_base = git("merge-base", base, "HEAD").strip()
    diffed = git("diff", "--name-only", merge_base).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    return sorted({p for p in (*diffed, *untracked) if p})


def module_name(path: str) -> str | None:
    """The dotted module a repository-relative `.py` path is imported as, or None."""
    if not path.endswith(".py"):
        return None
    top, _, rest = path.partition("/")
    if top not in _SOURCE_ROOTS or not rest:
        return None
    inner = rest if top == _STRIPPED_ROOT else path
    return inner.removesuffix(".py").replace("/", ".").removesuffix(".__init__")


def _imports_of(path: Path, name: str, is_package: bool) -> set[str]:
    """Every module name `path` imports, anywhere in the file, including lazy imports."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    package = name if is_package else name.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                anchor = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join([*anchor, node.module] if node.module else anchor)
            else:
                base = node.module or ""
            if base:
                found.add(base)
                found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _import_graph(
    repo: Path, absent: frozenset[str] = frozenset()
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Module name → path, and module name → the modules it imports (known ones only).

    Importing `a.b.c` also runs `a/__init__.py` and `a/b/__init__.py`, so an import of
    `a.b.c` is an edge to every prefix of it that is a module. `absent` names modules with no file — deleted
    by the diff — that still count as known, so whatever imports them still depends on them.
    """
    files: dict[str, str] = {}
    for root in _SOURCE_ROOTS:
        top = repo / root
        if not top.is_dir():
            continue
        for file in top.rglob("*.py"):
            rel = file.relative_to(repo).as_posix()
            if "/node_modules/" in rel or "/.venv/" in rel:
                continue
            name = module_name(rel)
            if name:
                files[name] = rel
    graph: dict[str, set[str]] = {}
    for name, rel in files.items():
        deps: set[str] = set()
        for imported in _imports_of(repo / rel, name, rel.endswith("__init__.py")):
            parts = imported.split(".")
            for i in range(len(parts), 0, -1):
                candidate = ".".join(parts[:i])
                if candidate in files or candidate in absent:
                    deps.add(candidate)
        deps.discard(name)
        graph[name] = deps
    return files, graph


def _reaches(graph: dict[str, set[str]], start: str, targets: set[str]) -> bool:
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in targets:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, ()))
    return False


def _is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.startswith("tests/") and name.startswith("test_") and name.endswith(".py")


def select_for_diff(changed: list[str], repo: Path) -> ChangedSelection:
    """Decide what the changed paths reach. Pure over the tree at `repo`."""
    selection = ChangedSelection(changed=list(changed))
    for path in changed:
        if (
            path in _WHOLE_SUITE_FILES
            or path.rsplit("/", 1)[-1] == "conftest.py"
            or path.startswith(_WHOLE_SUITE_PREFIXES)
        ):
            selection.whole_suite = f"{path} reaches every test"
            break

    existing_py = [p for p in changed if p.endswith(".py") and (repo / p).is_file()]
    selection.lint_files = existing_py
    selection.frontend = any(p.startswith(_FRONTEND_PREFIXES) for p in changed)
    if selection.whole_suite:
        return selection

    changed_modules = {m for p in changed if (m := module_name(p))}
    files, graph = _import_graph(repo, frozenset(changed_modules))
    all_tests = sorted(rel for rel in files.values() if _is_test_file(rel))

    selected: set[str] = {p for p in existing_py if _is_test_file(p)}
    selected.update(_TREE_WIDE_TESTS)
    for rel in all_tests:
        name = module_name(rel)
        if name and _reaches(graph, name, changed_modules):
            selected.add(rel)

    # Tests that name a changed file by its path: scripts, documents, fixtures, templates.
    named = [p for p in changed if not p.endswith(".py")]
    if named:
        for rel in all_tests:
            try:
                text = (repo / rel).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if any(p in text for p in named):
                selected.add(rel)

    governance = any(
        p.endswith(".md") or p.startswith((".claude/", "docs/", "swarm/skills/")) for p in named
    )
    if governance:
        selected.update(rel for rel in all_tests if tuple(rel.split("/")[:2]) == _FITNESS_DIR)

    selection.test_files = sorted(p for p in selected if (repo / p).is_file())

    importers = {
        rel
        for name, rel in files.items()
        if graph.get(name, set()) & changed_modules and (repo / rel).is_file()
    }
    selection.typecheck_files = sorted(set(existing_py) | importers)
    return selection


def split_browser_tests(test_files: list[str]) -> tuple[list[str], list[str]]:
    """(tests that run on workers, browser tests that run in one process after them)."""
    browser = [p for p in test_files if p.startswith(_E2E_DIR)]
    return [p for p in test_files if not p.startswith(_E2E_DIR)], browser


_NOTHING_TO_TYPECHECK = 0


def run_changed_gate(
    *,
    base: str = "origin/main",
    fail_fast: bool = True,
    junit_path: Path = Path(".pytest_cache/junit.changed.xml"),
    repo: Path | None = None,
) -> int:
    """Run the gate's checks over what the diff against `base` reaches. Records nothing.

    The browser suite is left to the merge-time gate and `./ucx test e2e`, and said so: it is
    the slowest step of the full gate and the one a diff-scoped run is for avoiding.

    A selection that holds no test is reported and passes — this is not a tier, so R5's
    "an empty tier is a failure" does not apply; the full gate at the merge head is the
    verdict on the whole suite.
    """
    # Imported here: `quality_gate` is the heavier module, and this one's selection logic is
    # imported on its own by its tests.
    from uclone_x.cli import quality_gate as qg

    root = repo or Path.cwd()
    try:
        changed = changed_paths(base, cwd=root)
    except RuntimeError as exc:
        print(f"ucx: {exc}", file=qg.sys.stderr)
        return qg.SCOPE_RESOLUTION_EXIT_CODE

    console = qg.console
    console.print("[bold cyan]UClone-X diff-scoped check[/bold cyan] (records no gate pass)")
    if not changed:
        console.print(f"Nothing differs from {base}; nothing to check.")
        return 0
    selection = select_for_diff(changed, root)
    console.print(f"[dim]{len(changed)} path(s) differ from the merge base with {base}.[/dim]")

    if selection.whole_suite:
        console.print(
            f"[yellow]{selection.whole_suite}: running the whole gate without the browser "
            "suite (`./ucx test check --fast`).[/yellow]"
        )
        return qg.run_quality_gate(fail_fast=fail_fast, test_scope="fast")

    first_failure = 0

    def failed(code: int) -> bool:
        nonlocal first_failure
        if code != 0 and first_failure == 0:
            first_failure = code
        return code != 0 and fail_fast

    lock_status, lock_lines = qg.check_lockfile_freshness()
    for line in lock_lines:
        console.print(line)
    if lock_status in ("stale", "uv-missing") and failed(1):
        return first_failure
    env_status, env_lines = qg.check_environment_provenance()
    if env_status in qg.BLOCKING_STATUSES:
        for line in env_lines:
            console.print(line)
        if failed(1):
            return first_failure

    if selection.lint_files:
        console.print(f"\n[bold]Ruff over {len(selection.lint_files)} changed file(s)[/bold]")
        if failed(qg.run_stage(["ruff", "format", "--check", *selection.lint_files]).returncode):
            return first_failure
        if failed(qg.run_stage(["ruff", "check", *selection.lint_files]).returncode):
            return first_failure
    if selection.typecheck_files:
        console.print(
            f"\n[bold]Pyright strict over {len(selection.typecheck_files)} file(s): the changed "
            "ones and their direct importers[/bold]"
        )
        if failed(qg.run_stage(["pyright", *selection.typecheck_files]).returncode):
            return first_failure

    workers, browser = split_browser_tests(selection.test_files)
    if not selection.test_files:
        console.print("\n[dim]No test reaches this diff.[/dim]")
    marker = qg.gate_marker_expression()
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    if workers:
        console.print(f"\n[bold]Pytest over {len(workers)} test file(s) the diff reaches[/bold]")
        argv = ["pytest", "-q", "--no-cov", f"--junitxml={junit_path}", "-m", marker]
        if len(workers) > 1 and qg.parallel_runner_available():
            argv += ["-n", str(min(qg.pytest_worker_count(), len(workers))), "--dist=load"]
        code = qg.run_stage([*argv, *workers]).returncode
        if code == qg.PYTEST_NO_TESTS_COLLECTED:
            console.print("[dim]The selected files hold no gate-tier test.[/dim]")
        elif failed(code):
            return first_failure
    if browser:
        console.print(
            f"\n[dim]{len(browser)} browser test file(s) reach this diff and are left to "
            "`./ucx test e2e` and the merge-time gate.[/dim]"
        )

    if selection.frontend:
        console.print("\n[bold]Frontend vitest suite[/bold]")
        _, fe_code, fe_lines = qg.run_frontend_suite()
        for line in fe_lines:
            console.print(line, soft_wrap=True)
        if failed(fe_code):
            return first_failure
        console.print("\n[bold]src/uclone_x/ui_static is what frontend/ builds[/bold]")
        _, bundle_code, bundle_lines = qg.check_bundle_freshness()
        for line in bundle_lines:
            console.print(line, soft_wrap=True)
        if failed(bundle_code):
            return first_failure

    if first_failure:
        console.print("\n[bold red]✖ Diff-scoped check failed.[/bold red]")
        return first_failure
    console.print(
        "\n[bold green]✔ Diff-scoped check passed.[/bold green] "
        "[dim]Not a gate pass: the full `./ucx test check` runs at the head that merges.[/dim]"
    )
    return 0
