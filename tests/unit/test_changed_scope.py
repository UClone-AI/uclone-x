"""Unit tests for the diff-scoped check (uclone_x.cli.changed_scope)."""

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
    "tests/unit/test_oss_export.py": "",
    "tests/unit/test_swarm_manifest.py": "",
}

_TREE_WIDE = ["tests/unit/test_oss_export.py", "tests/unit/test_swarm_manifest.py"]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _tree(tmp_path, _REPO)


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


def test_a_non_python_file_selects_the_tests_that_name_its_path(repo: Path) -> None:
    selection = select_for_diff(["scripts/run.sh"], repo)
    assert selection.test_files == sorted(["tests/unit/test_script.py", *_TREE_WIDE])
    assert selection.lint_files == []
    assert selection.typecheck_files == []


@pytest.mark.parametrize("path", ["docs/guide.md", "README.md", "swarm/skills/x/y.txt"])
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
    workers, browser = split_browser_tests(["tests/unit/test_a.py", "tests/e2e/test_b.py"])
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


def test_the_run_checks_the_selection_and_skips_the_browser_suite(
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
