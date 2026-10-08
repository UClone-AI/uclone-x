"""Unit tests for the diff-scoped check (uclone_x.cli.changed_scope)."""

import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from uclone_x.cli import changed_scope, main, quality_gate
from uclone_x.cli.changed_scope import (
    changed_paths,
    module_name,
    run_changed_gate,
    select_for_diff,
    split_browser_tests,
)
from uclone_x.cli.scope_rules import RULES_PATH, load_scope_rules

_ROOT = Path(__file__).resolve().parents[2]


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


# A small repository: `test_app` reaches `core` only through `app`; `test_other` reaches
# nothing; `test_script` names a shell script by its path.
_REPO = {
    "src/pkg/__init__.py": "",
    "src/pkg/core.py": "VALUE = 1\n",
    "src/pkg/app.py": "from pkg.core import VALUE\n",
    "src/pkg/lazy.py": "def f():\n    from . import core\n    return core\n",
    "src/pkg/sub/__init__.py": "from pkg import core\n",
    "src/pkg/sub/leaf.py": "X = 2\n",
    "src/pkg/other.py": "Y = 3\n",
    "scripts/run.sh": "echo hi\n",
    "tests/unit/test_app.py": "from pkg.app import VALUE\n",
    "tests/unit/test_other.py": "from pkg.other import Y\n",
    "tests/unit/test_lazy.py": "def test():\n    import pkg.lazy\n",
    "tests/unit/test_leaf.py": "from pkg.sub.leaf import X\n",
    "tests/unit/test_script.py": "SCRIPT = 'scripts/run.sh'\n",
    "tests/fitness/test_docs.py": "",
    "tests/e2e/test_browser.py": "from pkg.app import VALUE\n",
    "tests/e2e/test_other_browser.py": "from pkg.app import VALUE\n",
    "tests/unit/test_paid.py": "import pytest\n\n@pytest.mark.live\ndef test(): ...\n",
    "tests/unit/test_oss_export.py": "",
    "tests/unit/test_swarm_manifest.py": "",
    "tests/fitness/test_oss_manifest.py": "",
    "tests/fitness/test_mutation_kill_declarations.py": "",
    "tests/fitness/test_frontend_kill_declarations.py": "",
    "tests/fitness/test_kill_declaration_uniqueness.py": "",
    "tests/fitness/test_kill_declaration_placement.py": "",
    "tests/fitness/test_kill_declaration_becomes.py": "",
}

_TREE_WIDE = sorted(
    [
        "tests/unit/test_oss_export.py",
        "tests/unit/test_swarm_manifest.py",
        "tests/fitness/test_oss_manifest.py",
        "tests/fitness/test_mutation_kill_declarations.py",
        "tests/fitness/test_frontend_kill_declarations.py",
        "tests/fitness/test_kill_declaration_uniqueness.py",
        "tests/fitness/test_kill_declaration_placement.py",
        "tests/fitness/test_kill_declaration_becomes.py",
    ]
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = _tree(tmp_path, _REPO)
    shutil.copy(_ROOT / RULES_PATH, root / RULES_PATH)
    return root


def test_module_name_strips_src_and_init() -> None:
    assert module_name("src/pkg/core.py") == "pkg.core"
    assert module_name("src/pkg/__init__.py") == "pkg"
    assert module_name("tests/unit/test_app.py") == "tests.unit.test_app"
    assert module_name("swarm/tool.py") == "swarm.tool"
    assert module_name("docs/x.md") is None
    assert module_name("frontend/a.py") is None


@pytest.mark.parametrize(
    "path", ["pyproject.toml", "uv.lock", "ucx", "tests/unit/conftest.py", "tests/support/x.py"]
)
def test_a_change_every_test_depends_on_selects_the_whole_suite(repo: Path, path: str) -> None:
    selection = select_for_diff([path], repo)
    assert selection.whole_suite == f"{path} reaches every test"
    assert selection.test_files == []


def test_a_test_reaching_the_change_only_transitively_is_selected(repo: Path) -> None:
    selection = select_for_diff(["src/pkg/core.py"], repo)
    assert selection.whole_suite is None
    assert "tests/unit/test_app.py" in selection.test_files
    assert "tests/unit/test_other.py" not in selection.test_files
    assert "tests/unit/test_script.py" not in selection.test_files


def test_a_lazy_relative_import_is_followed(repo: Path) -> None:
    assert "tests/unit/test_lazy.py" in select_for_diff(["src/pkg/core.py"], repo).test_files


def test_importing_a_module_depends_on_its_parent_packages(repo: Path) -> None:
    # test_leaf imports only pkg.sub.leaf, but that runs pkg/sub/__init__.py, which imports core.
    assert "tests/unit/test_leaf.py" in select_for_diff(["src/pkg/core.py"], repo).test_files


def test_an_unrelated_module_selects_only_its_own_tests(repo: Path) -> None:
    selection = select_for_diff(["src/pkg/other.py"], repo)
    assert selection.test_files == sorted(["tests/unit/test_other.py", *_TREE_WIDE])


def test_a_changed_test_file_selects_itself(repo: Path) -> None:
    selection = select_for_diff(["tests/unit/test_other.py"], repo)
    assert selection.test_files == sorted(["tests/unit/test_other.py", *_TREE_WIDE])


def test_the_tree_wide_checks_run_on_any_diff(repo: Path) -> None:
    # A new file of any kind can be unclassified for export or an undeclared test.
    selection = select_for_diff(["assets/new.png"], repo)
    assert selection.test_files == _TREE_WIDE


def test_a_source_change_runs_the_declaration_and_manifest_checks(repo: Path) -> None:
    """The full gate failed on these after a clean diff-scoped run: a new file missing from
    the export manifest, a kill declaration whose line moved. Nothing imports a changed
    module into them, so only `[always]` brings them into `./ucx test changed`.

    Killed by: tests/scope-rules.toml :: "tests/fitness/test_mutation_kill_declarations.py",
    Becomes:
    """
    selection = select_for_diff(["src/pkg/other.py"], repo)
    assert "tests/fitness/test_oss_manifest.py" in selection.test_files
    assert "tests/fitness/test_mutation_kill_declarations.py" in selection.test_files
    assert "tests/fitness/test_frontend_kill_declarations.py" in selection.test_files


def test_a_non_python_file_selects_the_tests_that_name_its_path(repo: Path) -> None:
    selection = select_for_diff(["scripts/run.sh"], repo)
    assert selection.test_files == sorted(["tests/unit/test_script.py", *_TREE_WIDE])
    assert selection.lint_files == []
    assert selection.typecheck_files == []


@pytest.mark.parametrize("path", ["docs/guide.md", "README.md", ".claude/skills/x/y.txt"])
def test_a_governance_document_selects_the_fitness_functions(repo: Path, path: str) -> None:
    assert "tests/fitness/test_docs.py" in select_for_diff([path], repo).test_files


def test_a_source_change_does_not_select_the_fitness_functions(repo: Path) -> None:
    assert "tests/fitness/test_docs.py" not in select_for_diff(["src/pkg/core.py"], repo).test_files


def test_typecheck_covers_the_changed_file_and_its_direct_importers(repo: Path) -> None:
    selection = select_for_diff(["src/pkg/core.py"], repo)
    assert selection.lint_files == ["src/pkg/core.py"]
    assert "src/pkg/core.py" in selection.typecheck_files
    assert "src/pkg/app.py" in selection.typecheck_files
    assert "src/pkg/sub/__init__.py" in selection.typecheck_files
    # test_app reaches core only through app: an indirect importer, not type-checked.
    assert "tests/unit/test_app.py" not in selection.typecheck_files


def test_a_deleted_module_still_selects_its_importers(repo: Path) -> None:
    (repo / "src/pkg/other.py").unlink()
    selection = select_for_diff(["src/pkg/other.py"], repo)
    assert selection.lint_files == []
    assert selection.test_files == sorted(["tests/unit/test_other.py", *_TREE_WIDE])


@pytest.mark.parametrize(
    ("path", "frontend"),
    [
        ("frontend/src/App.tsx", True),
        ("src/uclone_x/ui_static/index.html", True),
        ("src/pkg/other.py", False),
    ],
)
def test_the_frontend_flag_follows_the_frontend_paths(
    repo: Path, path: str, frontend: bool
) -> None:
    assert select_for_diff([path], repo).frontend is frontend


def test_browser_tests_are_split_from_the_worker_tests() -> None:
    workers, browser = split_browser_tests(
        ["tests/unit/test_a.py", "tests/e2e/test_b.py"], load_scope_rules(_ROOT)
    )
    assert workers == ["tests/unit/test_a.py"]
    assert browser == ["tests/e2e/test_b.py"]


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_changed_paths_holds_committed_uncommitted_and_new_files(tmp_path: Path) -> None:
    _tree(tmp_path, {"a.txt": "a\n", "b.txt": "b\n", "c.txt": "c\n"})
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    _git(tmp_path, "branch", "base")
    (tmp_path / "a.txt").write_text("a2\n", encoding="utf-8")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "one")
    (tmp_path / "b.txt").write_text("b2\n", encoding="utf-8")
    (tmp_path / "d.txt").write_text("d\n", encoding="utf-8")
    assert changed_paths("base", cwd=tmp_path) == ["a.txt", "b.txt", "d.txt"]


def test_changed_paths_raises_when_git_names_no_merge_base(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    with pytest.raises(RuntimeError, match="merge-base"):
        changed_paths("nope", cwd=tmp_path)


class _Done:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


@pytest.fixture
def stages(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub every step `run_changed_gate` delegates to; collect the argv of each stage."""
    ran: list[list[str]] = []

    def run_stage(argv: list[str]) -> _Done:
        ran.append(argv)
        return _Done(0)

    def refuse(**_: object) -> str:
        raise AssertionError("the diff-scoped check must record no gate pass")

    monkeypatch.setattr(quality_gate, "run_stage", run_stage)
    monkeypatch.setattr(quality_gate, "record_gate_pass", refuse)
    no_lines: list[str] = []
    monkeypatch.setattr(quality_gate, "check_lockfile_freshness", lambda: ("fresh", no_lines))
    monkeypatch.setattr(quality_gate, "check_environment_provenance", lambda: ("ok", no_lines))
    monkeypatch.setattr(quality_gate, "parallel_runner_available", lambda: True)
    monkeypatch.setattr(quality_gate, "pytest_worker_count", lambda: 12)
    return ran


def _run(repo: Path, monkeypatch: pytest.MonkeyPatch, changed: list[str]) -> int:
    def paths(base: str, cwd: Path | None = None) -> list[str]:
        return changed

    monkeypatch.setattr(changed_scope, "changed_paths", paths)
    return run_changed_gate(repo=repo, junit_path=repo / ".pytest_cache" / "j.xml")


def test_the_run_checks_the_selection_and_skips_the_unchanged_browser_tests(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stages: list[list[str]]
) -> None:
    assert _run(repo, monkeypatch, ["src/pkg/core.py"]) == 0
    tools = [argv[0] for argv in stages]
    assert tools == ["ruff", "ruff", "pyright", "pytest"]
    pytest_argv = stages[-1]
    assert pytest_argv[pytest_argv.index("-m") + 1] == quality_gate.gate_marker_expression()
    assert "-n" in pytest_argv
    assert int(pytest_argv[pytest_argv.index("-n") + 1]) <= 12
    assert "tests/unit/test_app.py" in pytest_argv
    assert "tests/e2e/test_browser.py" not in pytest_argv


def test_a_selection_without_gate_tier_tests_passes(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stages: list[list[str]]
) -> None:
    def run_stage(argv: list[str]) -> _Done:
        stages.append(argv)
        return _Done(quality_gate.PYTEST_NO_TESTS_COLLECTED if argv[0] == "pytest" else 0)

    monkeypatch.setattr(quality_gate, "run_stage", run_stage)
    assert _run(repo, monkeypatch, ["src/pkg/other.py"]) == 0


def test_a_failing_stage_fails_the_run(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stages: list[list[str]]
) -> None:
    def run_stage(argv: list[str]) -> _Done:
        stages.append(argv)
        return _Done(1 if argv[0] == "pytest" else 0)

    monkeypatch.setattr(quality_gate, "run_stage", run_stage)
    assert _run(repo, monkeypatch, ["src/pkg/other.py"]) == 1


def test_a_whole_suite_change_delegates_to_the_fast_gate(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stages: list[list[str]]
) -> None:
    calls: list[dict[str, object]] = []

    def gate(**kwargs: object) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(quality_gate, "run_quality_gate", gate)
    assert _run(repo, monkeypatch, ["pyproject.toml"]) == 0
    assert calls == [{"fail_fast": True, "test_scope": "fast"}]
    assert stages == []


def test_an_empty_diff_runs_nothing(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stages: list[list[str]]
) -> None:
    assert _run(repo, monkeypatch, []) == 0
    assert stages == []


def test_no_merge_base_is_a_scope_resolution_failure(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stages: list[list[str]]
) -> None:
    def fail(base: str, cwd: Path | None = None) -> list[str]:
        raise RuntimeError("git merge-base failed")

    monkeypatch.setattr(changed_scope, "changed_paths", fail)
    assert run_changed_gate(repo=repo) == quality_gate.SCOPE_RESOLUTION_EXIT_CODE
    assert stages == []


def test_the_cli_passes_base_and_fail_fast_and_exits_with_the_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def gate(*, base: str, fail_fast: bool) -> int:
        calls.append((base, fail_fast))
        return 3

    monkeypatch.setattr(changed_scope, "run_changed_gate", gate)
    result = CliRunner().invoke(main.app, ["test", "changed", "--base", "main", "--no-fail-fast"])
    assert result.exit_code == 3
    assert calls == [("main", False)]


def test_a_changed_browser_test_runs_in_one_process_and_the_rest_stay_deferred(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stages: list[list[str]]
) -> None:
    """A test the PR edits is run by the PR's own check, browser test or not.

    Killed by: src/uclone_x/cli/changed_scope.py :: if failed(run_changed_browser_tests()):
    Becomes: if False:
    """
    assert _run(repo, monkeypatch, ["src/pkg/core.py", "tests/e2e/test_browser.py"]) == 0
    browser_runs = [a for a in stages if a[0] == "pytest" and "tests/e2e/test_browser.py" in a]
    assert len(browser_runs) == 1
    assert "-n" not in browser_runs[0]
    assert "--no-cov" in browser_runs[0]
    # Reached through the import graph but not changed: left to the merge gate.
    assert not any("tests/e2e/test_other_browser.py" in argv for argv in stages)


def test_a_failing_changed_browser_test_fails_the_run(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stages: list[list[str]]
) -> None:
    def run_stage(argv: list[str]) -> _Done:
        stages.append(argv)
        return _Done(1 if "tests/e2e/test_browser.py" in argv else 0)

    monkeypatch.setattr(quality_gate, "run_stage", run_stage)
    assert _run(repo, monkeypatch, ["tests/e2e/test_browser.py"]) == 1


def test_a_whole_suite_change_still_runs_the_changed_browser_tests(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stages: list[list[str]]
) -> None:
    """`check --fast` has no browser suite, so the changed browser tests run after it.

    Killed by: src/uclone_x/cli/changed_scope.py :: browser_code = run_changed_browser_tests()
    Becomes: browser_code = 0
    """

    def _fake_gate(**_: object) -> int:
        return 0

    monkeypatch.setattr(quality_gate, "run_quality_gate", _fake_gate)
    assert _run(repo, monkeypatch, ["pyproject.toml", "tests/e2e/test_browser.py"]) == 0
    assert [a for a in stages if a[0] == "pytest" and "tests/e2e/test_browser.py" in a]


def test_a_changed_test_holding_an_opt_in_tier_is_named_as_not_run(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    stages: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A diff's own `live` test is skipped by the gate's marker; the run says so.

    Killed by: src/uclone_x/cli/changed_scope.py :: set(_OPT_IN_MARKER.findall(text))
    Becomes: set[str]()
    """
    selection = select_for_diff(["tests/unit/test_paid.py"], repo)
    assert selection.opt_in_tests == {"tests/unit/test_paid.py": ["live"]}
    assert _run(repo, monkeypatch, ["tests/unit/test_paid.py"]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "tests/unit/test_paid.py holds live test(s)" in out
    assert "./ucx test live" in out


def test_an_unreadable_rules_file_is_a_scope_resolution_failure(
    repo: Path, monkeypatch: pytest.MonkeyPatch, stages: list[list[str]]
) -> None:
    """Selecting by rules nobody wrote would be a silent narrowing; the run refuses instead."""
    (repo / RULES_PATH).write_text(
        "version = 1\n[whole_suite]\nprefixs = ['x/']\n", encoding="utf-8"
    )
    assert _run(repo, monkeypatch, ["src/pkg/core.py"]) == quality_gate.SCOPE_RESOLUTION_EXIT_CODE
    assert stages == []


@pytest.mark.parametrize(
    ("source", "tiers"),
    [
        ("pytestmark = pytest.mark.pre_release\n", ["pre_release"]),
        ("pytestmark = [pytest.mark.recorded, pytest.mark.slow]\n", ["recorded"]),
        ("    @pytest.mark.live\n    def test(self): ...\n", ["live"]),
        # A test that writes a fixture naming the marker marks nothing itself.
        ('FIXTURE = "import pytest\\n\\n@pytest.mark.live\\ndef test(): ..."\n', []),
        ("# pytest.mark.live is opt-in\n", []),
    ],
)
def test_only_a_marker_that_applies_names_an_opt_in_tier(
    tmp_path: Path, source: str, tiers: list[str]
) -> None:
    """Killed by: src/uclone_x/cli/changed_scope.py :: r"^[
    Becomes: r"[
    """
    path = tmp_path / "test_x.py"
    path.write_text(source, encoding="utf-8")
    assert changed_scope._opt_in_tiers(path) == tiers  # pyright: ignore[reportPrivateUsage]
