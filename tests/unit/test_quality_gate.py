"""Unit tests for the Quality Gate runner (uclone_x.cli.quality_gate)."""

import ast
import importlib.util
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final, cast
from unittest.mock import MagicMock

import pytest

from uclone_x.cli import quality_gate
from uclone_x.cli.bundle_freshness import BundleStatus
from uclone_x.cli.environment_provenance import BLOCKING_STATUSES, ProvenanceStatus
from uclone_x.cli.quality_gate import (
    PYTHON_CHECK_PATHS,
    PYTHON_UNCHECKED_PATHS,
    SCOPE_RESOLUTION_EXIT_CODE,
    record_and_extract_failures,
    resolve_check_paths,
    run_quality_gate,
)
from uclone_x.cli.quality_gate import (
    check_lockfile_freshness as real_check_lockfile_freshness,
)
from uclone_x.cli.quality_gate import (
    check_package_coverage_floors as real_check_package_coverage_floors,
)
from uclone_x.cli.quality_gate import (
    run_frontend_suite as real_run_frontend_suite,
)


@pytest.fixture(autouse=True)
def stub_package_coverage_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the per-package floor step for the argv-sequence tests.

    The step shells out to `coverage report` once per declared package, which would put
    a dozen calls into every mocked `subprocess.run` sequence and make the assertions
    about ruff/pyright/pytest unreadable. The step has its own tests below; here it is
    stubbed so these tests keep asserting the thing they are about.
    """

    def no_shortfalls() -> list[tuple[str, int, int]]:
        return []

    monkeypatch.setattr(quality_gate, "check_package_coverage_floors", no_shortfalls)


@pytest.fixture(autouse=True)
def stub_lockfile_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the lockfile stage for the argv-sequence tests, as the floors step is.

    The stage shells out to `uv lock --check`, which would put a real subprocess into
    every mocked `subprocess.run` sequence below and shift the index of every other
    command by one. Worse, it would make those assertions depend on whether the tree the
    suite happens to run in has a fresh lockfile — the ambient-input mistake #436 was
    about.

    Stubbing it here is safe only because the stage has its own tests, and those are
    written the other way round: they drive it stale and assert the gate goes red. A test
    that asserts a stage passes is satisfied by a stage that does nothing.
    """

    def fresh() -> tuple[quality_gate.LockfileStatus, list[str]]:
        return "fresh", []

    monkeypatch.setattr(quality_gate, "check_lockfile_freshness", fresh)


@pytest.fixture(autouse=True)
def stub_environment_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise gate stage 1b for the argv-sequence tests, for the same reason as above.

    The stage spawns two child interpreters. Left live, they would be the first two entries
    in every mocked `subprocess.run` sequence below, shifting every index; and because the
    stage's verdict depends on the shared venv's editable install, the gate tests would
    pass or fail according to which worktree last ran `uv sync` — which is the very defect
    the stage exists to report, admitted into the tests that judge the gate.

    Safe only because the stage has its own tests, written the other way round: they drive
    it into each blocking state and assert the report. See
    `tests/unit/test_cli_environment_provenance.py`.
    """

    def ok() -> tuple[ProvenanceStatus, list[str]]:
        return "ok", []

    monkeypatch.setattr(quality_gate, "check_environment_provenance", ok)


@pytest.fixture(autouse=True)
def stub_frontend_suite(monkeypatch: pytest.MonkeyPatch) -> list[Path | None]:
    """Neutralise the frontend vitest stage for the argv-sequence tests; return its call log.

    The stage decides from the filesystem (`frontend/package.json`, `frontend/node_modules`)
    and from `PATH`, so left live it would make every gate test below pass or fail according
    to whether the worktree the suite runs in has had its `node_modules` linked — the
    ambient-input mistake #436 was about. A fresh worktree has none; the primary workspace
    does.

    Safe only because the stage has its own tests, written against the failure direction
    (an unrunnable suite must fail the gate), and because the selection tests read this
    log to assert *which scopes* run it. Returned so those tests can do exactly that.
    """
    calls: list[Path | None] = []

    def passed(
        root: Path | None = None,
        *,
        quiet: bool = False,
    ) -> tuple[quality_gate.FrontendSuiteStatus, int, list[str]]:
        calls.append(root)
        return "passed", 0, []

    monkeypatch.setattr(quality_gate, "run_frontend_suite", passed)
    return calls


#: Captured at import, before the autouse stub below replaces it, for the one test that is
#: about the real lookup.
_REAL_PARALLEL_RUNNER_AVAILABLE = quality_gate.parallel_runner_available


@pytest.fixture(autouse=True)
def stub_parallel_runner_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report pytest-xdist as installed for the argv-sequence tests, as the stages above are.

    Left live, every gate test here would pass or fail on whether the venv running the suite
    has been synced since #967 declared the dependency — ambient input again (#436). The
    refusal has its own tests below, which drive this to False and assert the gate goes red.
    """
    monkeypatch.setattr(quality_gate, "parallel_runner_available", lambda: True)


@pytest.fixture(autouse=True)
def stub_bundle_freshness(monkeypatch: pytest.MonkeyPatch) -> list[Path | None]:
    """Neutralise gate stage 6b (the committed bundle is what the source builds); return its log.

    Left live it runs a real `vite build`, which needs `node_modules` and would put a
    subprocess into every mocked sequence below. The stage's own verdicts are tested in
    `tests/unit/test_frontend_bundle_freshness.py`, in the failure direction.
    """
    calls: list[Path | None] = []

    def fresh(root: Path | None = None) -> tuple[BundleStatus, int, list[str]]:
        calls.append(root)
        return "fresh", 0, []

    monkeypatch.setattr(quality_gate, "check_bundle_freshness", fresh)
    return calls


def test_quality_gate_passes_all_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(subprocess, "run", mock_run)

    exit_code = run_quality_gate(quiet=True)
    assert exit_code == 0
    assert mock_run.call_count == 5
    # Verify ruff format and check cover all policy directories (#386)
    format_call_args = cast(list[str], mock_run.call_args_list[0][0][0])
    # Derived, not restated: the declared list names directories that the
    # open-source tree does not contain, and the gate checks the ones that
    # exist. A hardcoded expectation here passes only in this repository.
    assert format_call_args == ["ruff", "format", "--check", *resolve_check_paths()]
    lint_call_args = cast(list[str], mock_run.call_args_list[1][0][0])
    assert lint_call_args == ["ruff", "check", *resolve_check_paths()]
    pyright_call_args = cast(list[str], mock_run.call_args_list[2][0][0])
    assert pyright_call_args == ["pyright"]
    # Verify the unit scope excludes every other tier (#377): the marker expression is
    # one argv element, so assert against its text rather than against list membership.
    # The default gate runs pytest in two steps (#967) whose expressions partition the scope
    # by the `e2e` marker, so both still carry the scope's exclusions.
    expressions: list[str] = []
    for index in (3, 4):
        pytest_call_args = cast(list[str], mock_run.call_args_list[index][0][0])
        assert "-m" in pytest_call_args
        expressions.append(pytest_call_args[pytest_call_args.index("-m") + 1])
    assert expressions == [
        "(not recorded and not live) and not e2e",
        "(not recorded and not live) and e2e",
    ]


def test_quality_gate_static_scope_includes_all_python_directories() -> None:
    """No directory once inside the gate may quietly leave it (#386).

    This test used to assert `PYTHON_CHECK_PATHS == ("src", "tests", "swarm", "scripts")`
    -- a frozen literal under a name promising completeness. The two disagreed, and the
    disagreement had teeth: `evals/` was missing from the tuple, so *adding* it presented
    as a test failure, and the test most likely to prompt the fix was the one discouraging
    it (#588). A superset assertion keeps #386's actual intent (the scope never shrinks)
    without pinning it against growth; completeness is checked by
    `test_quality_gate_scope_covers_every_python_directory`, which discovers directories
    instead of listing them.
    """
    for required in ("src", "tests", "swarm", "scripts", "evals", "oss"):
        assert required in PYTHON_CHECK_PATHS, f"{required!r} dropped from the gate scope"


def test_resolve_check_paths_skips_directories_that_are_absent(tmp_path: Path) -> None:
    """A declared path that is not in this tree is dropped, in declaration order.

    `swarm`, `scripts` and `oss` are not published as open source, and both
    `ruff` and `pyright` fail on a path that does not exist -- so an unfiltered
    list would make the published gate fail on the absence of code that was
    never meant to be there.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()

    assert resolve_check_paths(tmp_path) == ("src", "tests")

    (tmp_path / "oss").mkdir()
    assert resolve_check_paths(tmp_path) == ("src", "tests", "oss")


def test_resolve_check_paths_refuses_a_tree_without_src(tmp_path: Path) -> None:
    """Checking nothing must not be reported as checking everything.

    Mutation this exists to catch: return the filtered tuple unconditionally.
    An empty scope makes `ruff` and `pyright` exit 0 over no files, and the
    gate would print a green tick for a directory that holds no project.
    """
    (tmp_path / "tests").mkdir()

    with pytest.raises(FileNotFoundError, match="no `src` directory"):
        resolve_check_paths(tmp_path)


def test_gate_outside_a_repository_reports_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ucx test check` from a PyPI install has no project to check.

    It is the first command a new user reaches for after `--help`, and it used
    to end in a rich-rendered `FileNotFoundError` traceback. The message was
    always right; rendering it as a crash is what made it wrong. Pinned here
    because the condition only arises outside a checkout, which is the one
    place no developer runs the gate.

    Mutation this exists to catch: let `resolve_check_paths` raise through
    `run_quality_gate` again.
    """
    monkeypatch.chdir(tmp_path)

    exit_code = run_quality_gate(quiet=False, skip_tests=True)

    assert exit_code == SCOPE_RESOLUTION_EXIT_CODE
    captured = capsys.readouterr()
    output = " ".join((captured.out + captured.err).split())
    assert "repository root" in output
    assert "Traceback" not in output


def test_gate_outside_a_repository_still_says_why_when_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`quiet` silences progress reporting, not the refusal to run.

    Mutation this exists to catch: guard the message with `if not quiet`. The
    gate then exits non-zero with nothing on any stream, which is a silent
    failure — the caller cannot tell "wrong directory" from "checks failed"
    from a crash.
    """
    monkeypatch.chdir(tmp_path)

    exit_code = run_quality_gate(quiet=True, skip_tests=True)

    assert exit_code == SCOPE_RESOLUTION_EXIT_CODE
    assert "repository root" in capsys.readouterr().err


def test_quality_gate_console_output_names_scopes_and_exclusions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify console output explicitly reports verified Python scopes and excluded paths (#386)."""
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(subprocess, "run", mock_run)

    exit_code = run_quality_gate(quiet=False, skip_tests=True)
    assert exit_code == 0
    # Rich wraps the scope line, and the wrap point moves with the length of the
    # scope list, so the assertions below read the unwrapped text.
    captured = " ".join(capsys.readouterr().out.split())
    assert ", ".join(resolve_check_paths()) in captured
    assert "Scope: Python static checks cover" in captured
    assert "Non-Python paths" in captured
    assert "excluded" in captured


def test_quality_gate_test_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    # The worker flags a scope without the browser suite carries (#967).
    parallel = ["-n", str(quality_gate.pytest_worker_count()), "--dist=load"]

    # Gate scope: what `./ucx test check` runs — everything offline and free, in two steps:
    # everything but the browser suite on workers, then the browser suite in one process.
    calls.clear()
    assert run_quality_gate(quiet=True, test_scope="gate") == 0
    assert calls[3] == [
        "pytest",
        "-v",
        "--junitxml=.pytest_cache/junit.workers.xml",
        "-o",
        "junit_family=xunit2",
        *parallel,
        "--cov-fail-under=0",
        "-m",
        "(not recorded and not live) and not e2e",
    ]
    assert calls[4] == [
        "pytest",
        "-v",
        "--junitxml=.pytest_cache/junit.browser.xml",
        "-o",
        "junit_family=xunit2",
        "--cov-append",
        "-m",
        "(not recorded and not live) and e2e",
    ]
    assert len(calls) == 5

    # Unit scope: product-code tests only, with the fitness functions excluded.
    calls.clear()
    assert run_quality_gate(quiet=True, test_scope="unit") == 0
    assert calls[3] == [
        "pytest",
        "-v",
        "--junitxml=.pytest_cache/junit.xml",
        "-o",
        "junit_family=xunit2",
        *parallel,
        "-m",
        "not e2e and not recorded and not live and not fitness",
    ]

    # Fitness scope: exercises no product code, so coverage is disabled.
    calls.clear()
    assert run_quality_gate(quiet=True, test_scope="fitness") == 0
    assert calls[3] == [
        "pytest",
        "-v",
        "--junitxml=.pytest_cache/junit.xml",
        "-o",
        "junit_family=xunit2",
        *parallel,
        "--no-cov",
        "-m",
        "fitness",
    ]

    # Recorded scope (Tier 2): cassette replay, coverage disabled (#377)
    calls.clear()
    assert run_quality_gate(quiet=True, test_scope="recorded") == 0
    assert calls[3] == [
        "pytest",
        "-v",
        "--junitxml=.pytest_cache/junit.xml",
        "-o",
        "junit_family=xunit2",
        "--no-cov",
        "-m",
        "recorded",
    ]

    # Live scope (Tier 3): carries its own --live opt-in, coverage disabled (#377)
    calls.clear()
    assert run_quality_gate(quiet=True, test_scope="live") == 0
    assert calls[3] == [
        "pytest",
        "-v",
        "--junitxml=.pytest_cache/junit.xml",
        "-o",
        "junit_family=xunit2",
        "--no-cov",
        "--live",
        "-m",
        "live",
    ]

    # E2E scope: the browser suite, in one process (#967).
    calls.clear()
    assert run_quality_gate(quiet=True, test_scope="e2e") == 0
    assert calls[3] == [
        "pytest",
        "-v",
        "--junitxml=.pytest_cache/junit.xml",
        "-o",
        "junit_family=xunit2",
        "--no-cov",
        "-m",
        "e2e",
    ]

    # All scope: "all" is documented as Unit + E2E, so it still excludes the recorded and
    # live tiers rather than meaning "literally everything" (#377). Same two steps as gate.
    calls.clear()
    assert run_quality_gate(quiet=True, test_scope="all") == 0
    assert calls[3][-1] == "(not recorded and not live) and not e2e"
    assert calls[4][-1] == "(not recorded and not live) and e2e"
    assert len(calls) == 5


def test_quality_gate_fails_on_format(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=1))
    monkeypatch.setattr(subprocess, "run", mock_run)

    exit_code = run_quality_gate(quiet=False)
    assert exit_code == 1
    assert mock_run.call_count == 1


def test_quality_gate_fails_on_lint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "format" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0)
        if "check" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=2)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    exit_code = run_quality_gate(quiet=False)
    assert exit_code == 2


def test_quality_gate_fails_on_pyright(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "pyright" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=3)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    exit_code = run_quality_gate(quiet=False)
    assert exit_code == 3


def test_quality_gate_fails_on_pytest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pytest-failure path, with its report and history paths pinned to `tmp_path`.

    This test previously took the module defaults, which are cwd-relative, so it read
    whatever `.pytest_cache/junit.xml` the last gate run had left in the tree it ran in
    (#436). See the dedicated no-ambient-state test below for the property that pins it.
    """

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "pytest" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=4)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    exit_code = run_quality_gate(
        quiet=False,
        junit_path=tmp_path / "junit.xml",
        failure_log_path=tmp_path / "failure_history.jsonl",
    )
    assert exit_code == 4


def test_quality_gate_skip_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(subprocess, "run", mock_run)

    exit_code = run_quality_gate(quiet=True, skip_tests=True)
    assert exit_code == 0
    assert mock_run.call_count == 3


def test_quality_gate_with_frontend_pass(
    monkeypatch: pytest.MonkeyPatch, stub_frontend_suite: list[Path | None]
) -> None:
    """`-fe` runs vitest once and then the production build, with its environment pinned.

    Killed by: src/uclone_x/cli/quality_gate.py :: env={**os.environ, **BUILD_ENVIRONMENT},
    Becomes:
    """
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(subprocess, "run", mock_run)

    exit_code = run_quality_gate(quiet=True, check_frontend=True)
    assert exit_code == 0

    # Assert the command sequence, not just its length. `call_count == 5` passed for any
    # five subprocesses in any order, so it could not tell "the frontend suite ran" from
    # "ruff ran one extra time" — and it broke uninformatively when a step was added.
    commands = [call.args[0] for call in mock_run.call_args_list]
    assert [c[0] for c in commands] == ["ruff", "ruff", "pyright", "pytest", "pytest", "npm"]
    # The vitest suite ran exactly once — `-fe` on top of the gate scope, which already
    # selects it, must not run it twice — and the build is the one extra `-fe` buys.
    assert len(stub_frontend_suite) == 1
    assert commands[-1] == ["npm", "run", "build"]
    # It writes the committed bundle, so it pins the environment exactly as stage 6b does (#878).
    build_environment = mock_run.call_args_list[-1].kwargs["env"]
    assert build_environment["NODE_ENV"] == "production"
    assert build_environment["BROWSERSLIST"] == "defaults"


def test_quality_gate_with_frontend_build_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "npm" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=5)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    exit_code = run_quality_gate(quiet=False, check_frontend=True)
    assert exit_code == 5


def test_quality_gate_no_fail_fast_format_failure_propagates_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When format check fails with fail_fast=False, all steps run and non-zero exit is returned.

    Killed by: src/uclone_x/cli/quality_gate.py :: first_failure = res_format.returncode
    """
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if "format" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=1)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    exit_code = run_quality_gate(quiet=False, fail_fast=False)
    assert exit_code == 1
    # All 4 stages must have run: format, lint, pyright, pytest
    assert len(calls) == 5
    out = capsys.readouterr().out
    assert "✖ Ruff format check failed." in out
    assert "✖ Quality gate failed." in out
    assert "ALL SYSTEMS GO" not in out


def test_quality_gate_no_fail_fast_lint_failure_propagates_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When linter fails with fail_fast=False, all steps run and lint exit code is returned.

    Killed by: src/uclone_x/cli/quality_gate.py :: first_failure = res_lint.returncode
    """
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if "check" in cmd and "ruff" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=2)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    exit_code = run_quality_gate(quiet=False, fail_fast=False)
    assert exit_code == 2
    assert len(calls) == 5
    out = capsys.readouterr().out
    assert "✖ Ruff linter check failed." in out
    assert "✖ Quality gate failed." in out
    assert "ALL SYSTEMS GO" not in out


def test_quality_gate_no_fail_fast_pyright_failure_propagates_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When pyright fails with fail_fast=False, pytest still runs and pyright exit code is returned.

    Killed by: src/uclone_x/cli/quality_gate.py :: first_failure = res_pyright.returncode
    """
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if "pyright" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=3)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    exit_code = run_quality_gate(quiet=False, fail_fast=False)
    assert exit_code == 3
    assert len(calls) == 5
    out = capsys.readouterr().out
    assert "✖ Pyright type check failed." in out
    assert "✖ Quality gate failed." in out
    assert "ALL SYSTEMS GO" not in out


def test_quality_gate_no_fail_fast_pytest_failure_propagates_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When pytest fails with fail_fast=False, subsequent steps run and pytest exit code is returned.

    Killed by: src/uclone_x/cli/quality_gate.py :: first_failure = res_pytest.returncode  # test failures
    """
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if "pytest" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=4)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    exit_code = run_quality_gate(
        quiet=False,
        fail_fast=False,
        check_frontend=True,
        junit_path=tmp_path / "junit.xml",
        failure_log_path=tmp_path / "failure_history.jsonl",
    )
    assert exit_code == 4
    # format, lint, pyright, pytest ×2 (workers, then browser suite), npm build (vitest is stubbed)
    assert len(calls) == 6
    out = capsys.readouterr().out
    assert "✖ Pytest suite or branch coverage check (< 70%) failed." in out
    assert "✖ Quality gate failed." in out
    assert "ALL SYSTEMS GO" not in out


def test_quality_gate_no_fail_fast_package_floors_failure_propagates_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When package floors fail with fail_fast=False, subsequent steps run and exit code 1 is returned.

    Killed by: src/uclone_x/cli/quality_gate.py :: first_failure = 1
    """
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    def fake_shortfalls() -> list[tuple[str, int, int]]:
        return [("core", 65, 75)]

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(quality_gate, "check_package_coverage_floors", fake_shortfalls)

    exit_code = run_quality_gate(quiet=False, fail_fast=False, check_frontend=True)
    assert exit_code == 1
    assert len(calls) == 6
    out = capsys.readouterr().out
    assert "✖ uclone_x/core: 65% is below its 75% floor" in out
    assert "✖ Quality gate failed." in out
    assert "ALL SYSTEMS GO" not in out


def test_quality_gate_no_fail_fast_frontend_failure_propagates_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When frontend test fails with fail_fast=False, frontend build still runs and exit code is returned.

    Killed by: src/uclone_x/cli/quality_gate.py :: first_failure = fe_code
    Becomes: first_failure = 0
    """
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    def failed(
        root: Path | None = None,
        *,
        quiet: bool = False,
    ) -> tuple[quality_gate.FrontendSuiteStatus, int, list[str]]:
        return "failed", 6, ["[bold red]✖ Frontend unit tests failed.[/bold red]"]

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(quality_gate, "run_frontend_suite", failed)
    exit_code = run_quality_gate(quiet=False, fail_fast=False, check_frontend=True)
    assert exit_code == 6
    assert [c[0] for c in calls] == ["ruff", "ruff", "pyright", "pytest", "pytest", "npm"]
    out = capsys.readouterr().out
    assert "✖ Frontend unit tests failed." in out
    assert "✖ Quality gate failed." in out
    assert "ALL SYSTEMS GO" not in out


def test_quality_gate_no_fail_fast_returns_first_failure_when_multiple_stages_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When multiple stages fail with fail_fast=False, the first failed stage's returncode is returned.

    Killed by: src/uclone_x/cli/quality_gate.py :: if first_failure != 0:
    """
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if "format" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=11)
        if "pyright" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=22)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    exit_code = run_quality_gate(quiet=False, fail_fast=False)
    assert exit_code == 11
    assert len(calls) == 5
    out = capsys.readouterr().out
    assert "✖ Ruff format check failed." in out
    assert "✖ Pyright type check failed." in out
    assert "✖ Quality gate failed." in out
    assert "ALL SYSTEMS GO" not in out


def test_record_and_extract_failures(tmp_path: Path) -> None:
    junit_file = tmp_path / "junit.xml"
    log_file = tmp_path / "failures.jsonl"

    # 1. Missing XML returns empty
    assert (
        record_and_extract_failures(tmp_path / "nonexistent.xml", failure_log_path=log_file) == []
    )

    # 2. XML with passes and failures
    junit_file.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="0" failures="2" skipped="0" tests="3" time="1.0">
    <testcase classname="tests.unit.test_foo" name="test_pass" time="0.01" />
    <testcase classname="tests.unit.test_foo" name="test_fail_one" time="0.02">
      <failure message="assert 1 == 2">AssertionError</failure>
    </testcase>
    <testcase classname="tests.unit.test_bar" name="test_error_two" time="0.03">
      <error message="RuntimeError">Error</error>
    </testcase>
  </testsuite>
</testsuites>""",
        encoding="utf-8",
    )

    failures = record_and_extract_failures(junit_file, failure_log_path=log_file)
    assert failures == [
        "tests.unit.test_foo::test_fail_one",
        "tests.unit.test_bar::test_error_two",
    ]
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "tests.unit.test_foo::test_fail_one" in content
    assert "tests.unit.test_bar::test_error_two" in content


_A_REPORT_WITH_NO_FAILURES = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="0" failures="0" skipped="0" tests="1" time="1.0">
    <testcase classname="tests.unit.test_foo" name="test_pass" time="0.01" />
  </testsuite>
</testsuites>"""


def test_run_quality_gate_failure_path_consumes_no_ambient_cwd_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gate's failure path must not read or write cwd-relative default paths (#436).

    This is the *cause* of the history-dependent coverage figure, pinned directly rather
    than by measuring coverage. `JUNIT_REPORT_PATH` and `FAILURE_LOG_PATH` are relative, so
    a caller that leaves them defaulted reads the artifact of whatever ran in that directory
    before it. That made one branch inside `record_and_extract_failures` taken or not taken
    according to whether the gate had run in that tree before, which moved the gate's own
    reported branch coverage.

    The test plants exactly the poison — an ambient report naming a test that failed in some
    *earlier* run — and asserts it is not consumed. Note the second, worse half of the
    defect it also pins: because the ambient report names failures, the old code appended
    them to the real durable history with a *fresh* timestamp, so a green run manufactured
    a record of a failure that did not happen in it (P6's forbidden substituted result).
    """
    ambient = tmp_path / "cwd"
    (ambient / ".pytest_cache").mkdir(parents=True)
    # The gate refuses to run in a tree with no `src`, rather than reporting
    # success over an empty check scope. This test is about the failure path's
    # cwd handling, so it gives the ambient directory the minimum shape of a
    # repository root and nothing else.
    (ambient / "src").mkdir()
    (ambient / ".pytest_cache" / "junit.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="1" failures="1">
<testcase classname="tests.unit.test_phantom" name="test_from_an_earlier_red_run">
<failure message="assert 1 == 2">AssertionError</failure></testcase>
</testsuite></testsuites>""",
        encoding="utf-8",
    )
    monkeypatch.chdir(ambient)

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "pytest" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=4)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = run_quality_gate(
        quiet=False,
        junit_path=tmp_path / "report" / "junit.xml",
        failure_log_path=tmp_path / "report" / "failure_history.jsonl",
    )

    assert exit_code == 4
    # The ambient report was not read: nothing it names reaches the console.
    assert "test_from_an_earlier_red_run" not in capsys.readouterr().out
    # The ambient durable history was not written.
    assert not (ambient / ".pytest_cache" / "failure_history.jsonl").exists()
    # Nor was the pinned one, because the pinned report never existed.
    assert not (tmp_path / "report" / "failure_history.jsonl").exists()


def test_record_and_extract_failures_returns_empty_for_a_report_with_no_failures(
    tmp_path: Path,
) -> None:
    """A readable report with zero failures returns empty and writes no history (#436).

    This is the branch whose coverage used to depend on directory history — it was reached
    only when a *previous* run had left a green report behind. Driving it deliberately from
    a `tmp_path` fixture makes it covered on every run instead of on some runs.
    """
    junit_file = tmp_path / "junit.xml"
    junit_file.write_text(_A_REPORT_WITH_NO_FAILURES, encoding="utf-8")
    log_file = tmp_path / "failure_history.jsonl"

    assert record_and_extract_failures(junit_file, failure_log_path=log_file) == []
    assert not log_file.exists()


def test_record_and_extract_failures_reports_a_malformed_report_instead_of_passing_silently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A truncated report is named on the console, not swallowed by a bare `pass` (#436).

    pytest writes the report at session end, so a crashed or killed run leaves a partial
    one. Returning `[]` without saying why is the substituted-empty-result P6 forbids: the
    caller would print "no failed test cases" when the truth is that the record could not
    be read.
    """
    junit_file = tmp_path / "junit.xml"
    junit_file.write_text('<?xml version="1.0"?><testsuites><testsuite', encoding="utf-8")

    assert (
        record_and_extract_failures(junit_file, failure_log_path=tmp_path / "history.jsonl") == []
    )
    out = capsys.readouterr().out
    assert "Could not read the failure report" in out
    assert "ParseError" in out


def test_record_and_extract_failures_lets_a_programming_error_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The narrowed except must not catch defects in the extractor itself (#436).

    A broken extractor and an unparseable report are different failures. `except Exception`
    made them indistinguishable — both became "no failures found". This pins the narrowing:
    it fails if the handler is widened back to `Exception` or to `BaseException`.
    """
    junit_file = tmp_path / "junit.xml"
    junit_file.write_text(_A_REPORT_WITH_NO_FAILURES, encoding="utf-8")

    def exploding_parse(*args: object, **kwargs: object) -> object:
        raise AttributeError("simulated defect inside the extraction logic")

    monkeypatch.setattr(quality_gate.ET, "parse", exploding_parse)

    with pytest.raises(AttributeError, match="simulated defect"):
        record_and_extract_failures(junit_file, failure_log_path=tmp_path / "history.jsonl")


def _scopes_measuring_coverage(measured: bool) -> list[str]:
    """Scopes split by whether their pytest argv actually measures coverage.

    Read from the built command, not from the private scope set, so the split is the one
    the gate really runs.
    """
    return [
        scope
        for scope in quality_gate.TEST_SCOPES
        if ("--no-cov" not in quality_gate.build_pytest_command(scope)) is measured
    ]


def _gate_output_for(
    scope: str,
    pytest_returncode: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> str:
    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        code = pytest_returncode if cmd[0] == "pytest" else 0
        return subprocess.CompletedProcess(args=cmd, returncode=code)

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_quality_gate(
        quiet=False,
        test_scope=scope,
        junit_path=tmp_path / "junit.xml",
        failure_log_path=tmp_path / "failure_history.jsonl",
    )
    # Rich wraps long lines; read the unwrapped text.
    return " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("scope", _scopes_measuring_coverage(False))
def test_a_passing_scope_that_measures_no_coverage_reports_no_coverage_floor(
    scope: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A scope run with `--no-cov` must not claim a coverage floor it never measured (#882).

    Killed by: src/uclone_x/cli/quality_gate.py ::
        "[green]✔ Pytest tests passed (no coverage measured).[/green]"
    Becomes: "[green]✔ Pytest tests passed with >= 70% branch coverage.[/green]"
    """
    out = _gate_output_for(scope, 0, tmp_path, monkeypatch, capsys)
    assert "✔ Pytest tests passed" in out
    assert "70%" not in out, f"the {scope!r} scope ran with --no-cov yet reported: {out}"


@pytest.mark.parametrize("scope", _scopes_measuring_coverage(False))
def test_a_failing_scope_that_measures_no_coverage_does_not_blame_coverage(
    scope: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Naming a coverage check that did not run sends the reader to the wrong problem.

    Killed by: src/uclone_x/cli/quality_gate.py ::
        "[bold red]✖ Pytest suite failed (no coverage measured).[/bold red]"
    Becomes: "[bold red]✖ Pytest suite or branch coverage check (< 70%) failed.[/bold red]"
    """
    out = _gate_output_for(scope, 1, tmp_path, monkeypatch, capsys)
    assert "✖ Pytest suite failed" in out
    assert "70%" not in out, f"the {scope!r} scope ran with --no-cov yet reported: {out}"


@pytest.mark.parametrize("scope", _scopes_measuring_coverage(True))
def test_a_scope_that_measures_coverage_still_reports_its_floor(
    scope: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The converse, so the fix above cannot be satisfied by dropping the floor everywhere.

    The pass half names the whole message. `"70%" in out` was satisfied by the step-5
    header `(Branch Coverage >= 70%)` alone, so the pass message could change freely (#928).

    Killed by: src/uclone_x/cli/quality_gate.py ::
        "[green]✔ Pytest tests passed with >= 70% branch coverage.[/green]"
    Becomes: "[green]✔ Pytest tests passed (no coverage measured).[/green]"
    """
    passed = _gate_output_for(scope, 0, tmp_path, monkeypatch, capsys)
    assert "✔ Pytest tests passed with >= 70% branch coverage." in passed, (
        f"the {scope!r} scope measured coverage yet its pass message did not say so: {passed}"
    )
    assert "✖ Pytest suite or branch coverage check (< 70%) failed." in _gate_output_for(
        scope, 1, tmp_path, monkeypatch, capsys
    )


def test_empty_tier_is_reported_as_empty_rather_than_as_a_coverage_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pytest exit 5 means nothing was collected (#377).

    The recorded and live tiers are legitimately empty until the roadmap seeds them, and
    reporting that as "branch coverage < 70%" sends the reader to the wrong problem. It
    still fails: a tier that verified nothing must not report success.
    """

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        code = 5 if cmd[0] == "pytest" else 0
        return subprocess.CompletedProcess(args=cmd, returncode=code)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_quality_gate(quiet=True, test_scope="recorded") == 5


# ======================================================================================
# AST Sweep: Module-Level Relative Path Constants Invariant (#440)
# ======================================================================================


def is_relative_path_call(call: ast.Call) -> bool:
    """Check if an ast.Call is constructing a relative Path instance.

    Matches Path, PurePath, PosixPath, WindowsPath (both bare identifiers and
    attribute-qualified forms, e.g. pathlib.Path).

    A call is considered relative if:
    - It has no arguments (Path() defaults to Path(".")).
    - Its first positional argument is a string literal representing a relative path
      (i.e. not absolute on POSIX or Windows).
    """
    func = call.func
    target_name: str | None = None
    if isinstance(func, ast.Name):
        target_name = func.id
    elif isinstance(func, ast.Attribute):
        target_name = func.attr

    if target_name not in (
        "Path",
        "PurePath",
        "PosixPath",
        "WindowsPath",
        "PurePosixPath",
        "PureWindowsPath",
    ):
        return False

    if not call.args:
        # Path() with no args evaluates to Path(".")
        return True

    first_arg = call.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        val = first_arg.value
        if PurePosixPath(val).is_absolute() or PureWindowsPath(val).is_absolute():
            return False
        return True

    return False


def is_relative_path_expr(node: ast.AST) -> bool:
    """Check if an expression resolves to a relative Path instance.

    Handles direct calls (e.g. Path("...")) as well as Path division chaining
    (e.g. Path("...") / "sub").
    """
    if isinstance(node, ast.Call) and is_relative_path_call(node):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return is_relative_path_expr(node.left)
    return False


def extract_module_relative_path_constants(tree: ast.Module) -> set[str]:
    """Extract uppercase module-level constant names assigned to relative Path expressions.

    Traverses module-level statements including top-level If, Try, and With blocks,
    but explicitly ignores assignments inside FunctionDef, AsyncFunctionDef, or ClassDef
    scopes because those are not module-level constants.
    """
    constants: set[str] = set()

    def _scan_stmts(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        if is_relative_path_expr(stmt.value):
                            constants.add(target.id)
            elif isinstance(stmt, ast.AnnAssign):
                if (
                    isinstance(stmt.target, ast.Name)
                    and stmt.target.id.isupper()
                    and stmt.value is not None
                    and is_relative_path_expr(stmt.value)
                ):
                    constants.add(stmt.target.id)
            elif isinstance(stmt, (ast.If, ast.Try, ast.With)):
                for attr in ("body", "orelse", "finalbody"):
                    nested = getattr(stmt, attr, None)
                    if isinstance(nested, list):
                        _scan_stmts(cast(list[ast.stmt], nested))

    _scan_stmts(tree.body)
    return constants


def scan_src_relative_path_constants(
    src_dir: Path,
    repo_root: Path | None = None,
) -> dict[tuple[str, str], str]:
    """Scan Python files under src_dir for module-level relative Path constants.

    Returns a dictionary mapping (module_relative_path_posix, constant_name) to the module path.
    """
    if repo_root is None:
        repo_root = src_dir.parent if src_dir.name == "src" else src_dir
    findings: dict[tuple[str, str], str] = {}
    for py_file in sorted(src_dir.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        consts = extract_module_relative_path_constants(tree)
        rel_mod_path = py_file.resolve().relative_to(repo_root.resolve()).as_posix()
        for const in sorted(consts):
            findings[(rel_mod_path, const)] = rel_mod_path
    return findings


# Allowlist of known module-level relative Path constants.
# Each entry maps (module_path, constant_name) to a non-empty stated rationale.
# Stated reasons are asserted data (§6.9, #402).
RELATIVE_PATH_CONSTANT_ALLOWLIST: Final[dict[tuple[str, str], str]] = {
    ("src/uclone_x/cli/quality_gate.py", "JUNIT_REPORT_PATH"): (
        "Per-worktree default for pytest JUnit XML output; resolves against cwd "
        "so concurrent worktrees do not collide on shared reports, while functions "
        "accept explicit paths to allow tmp_path isolation in tests (#436, #437)."
    ),
    ("src/uclone_x/cli/quality_gate.py", "FAILURE_LOG_PATH"): (
        "Per-worktree default for failure history log; resolves against cwd "
        "so concurrent worktrees do not interleave failure histories, while functions "
        "accept explicit paths to allow tmp_path isolation in tests (#207, #436, #437)."
    ),
    ("src/uclone_x/cli/quality_gate.py", "LOCKFILE_PATH"): (
        "Per-worktree default for `uv.lock`; the lockfile a tree must agree with is its "
        "own, not the one in whatever checkout happens to be the repository root, and "
        "`check_lockfile_freshness` takes an explicit root for tmp_path isolation."
    ),
}


def verify_relative_path_constants(
    findings: set[tuple[str, str]],
    allowlist: Mapping[tuple[str, str], str],
) -> None:
    """Enforce that all relative Path constants in findings match allowlist in all four directions.

    Enforces:
    1. Reason validity (Direction 4): Every allowlist entry must carry a non-empty,
       non-whitespace string reason. Empty reasons or non-string values raise ValueError.
    2. Zero unallowlisted sites (Direction 1 & 2): Every detected relative Path constant
       must be present in allowlist. Any unexpected site raises AssertionError.
    3. Zero phantom entries (Direction 3): Every allowlist entry must correspond to an
       actual detected relative Path constant in source. Unused entries raise AssertionError.
    """
    for (mod_path, var_name), reason in allowlist.items():
        if not reason.strip():
            raise ValueError(
                f"Allowlist entry ({mod_path}, {var_name}) must have a non-empty reason string, got: {reason!r}"
            )

    allowlist_keys = set(allowlist.keys())

    unallowlisted = findings - allowlist_keys
    if unallowlisted:
        sites = ", ".join(f"{mod}:{var}" for mod, var in sorted(unallowlisted))
        raise AssertionError(
            f"Found module-level relative Path constants outside allowlist: {sites}"
        )

    phantom = allowlist_keys - findings
    if phantom:
        sites = ", ".join(f"{mod}:{var}" for mod, var in sorted(phantom))
        raise AssertionError(
            f"Allowlist contains entries that do not match any relative Path constant in source: {sites}"
        )


def test_src_has_no_unallowlisted_relative_path_constants() -> None:
    """Verify src/ has no module-level relative Path constants outside the allowlist.

    Acceptance criteria from Issue #440 / PR #437:
    - Scans src/ using AST for module-level assignments where target is uppercase
      and value constructs a relative Path.
    - Demonstrates non-blindness (§6.9 Case 2) by asserting > 50 files scanned and
      confirming the two known instances (JUNIT_REPORT_PATH and FAILURE_LOG_PATH)
      are matched before asserting absence of others.
    - Verifies findings match RELATIVE_PATH_CONSTANT_ALLOWLIST exactly.

    What this sweep covers:
    - Module-level uppercase constant assignments to relative Path expressions
      (e.g., Path("..."), pathlib.Path("..."), PurePath("..."), Path() / "...").
    - Assignments at top-level module scope, including within top-level if/try/with blocks.

    What this sweep does NOT cover (documented caveats):
    - Non-Path ambient state reads: os.environ, os.getcwd(), sys.argv, or direct
      ambient open(...) calls.
    - Tool-managed cache directories: .ruff_cache, or pytest internal test execution
      state under .pytest_cache/v/ (as rev-senior-52 noted, .pytest_cache/v/ is inert
      for quality gate selection because the gate selects by explicit -m markers
      rather than --lf).
    - Runtime-constructed dynamic paths (e.g., Path(os.getenv("..."))).
    - Path instances scoped inside functions, methods, or classes (e.g., fallback = Path(...)
      in _find_hooks_dir inside uclone_x.cli.main).
    """
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    assert src_dir.is_dir(), f"src directory not found at {src_dir}"

    py_files = list(src_dir.rglob("*.py"))
    # §6.9 Case 2: assert the instrument observed what it claims to observe
    assert len(py_files) > 50, f"Expected > 50 Python files in src/, found {len(py_files)}"

    findings = scan_src_relative_path_constants(src_dir, repo_root=repo_root)

    # Acceptance Criteria 4: Prove pattern matches the known instances before asserting absence of others
    expected_quality_gate = "src/uclone_x/cli/quality_gate.py"
    assert (expected_quality_gate, "JUNIT_REPORT_PATH") in findings, (
        f"Detector failed to find JUNIT_REPORT_PATH in {expected_quality_gate}"
    )
    assert (expected_quality_gate, "FAILURE_LOG_PATH") in findings, (
        f"Detector failed to find FAILURE_LOG_PATH in {expected_quality_gate}"
    )

    # Validate against allowlist in all dimensions
    verify_relative_path_constants(set(findings.keys()), RELATIVE_PATH_CONSTANT_ALLOWLIST)


def test_allowlist_direction_1_synthetic_violating_site_fails() -> None:
    """Direction 1: adding a synthetic violating site fails the test (unallowlisted)."""
    findings = {
        ("src/uclone_x/cli/quality_gate.py", "JUNIT_REPORT_PATH"),
        ("src/uclone_x/cli/quality_gate.py", "FAILURE_LOG_PATH"),
        ("src/uclone_x/cli/quality_gate.py", "NEW_UNALLOWLISTED_PATH"),
    }
    with pytest.raises(
        AssertionError,
        match="outside allowlist: src/uclone_x/cli/quality_gate.py:NEW_UNALLOWLISTED_PATH",
    ):
        verify_relative_path_constants(findings, RELATIVE_PATH_CONSTANT_ALLOWLIST)


def test_allowlist_direction_2_dropped_known_entry_fails() -> None:
    """Direction 2: dropping a known entry from the allowlist fails the test (load-bearing)."""
    findings = {
        ("src/uclone_x/cli/quality_gate.py", "JUNIT_REPORT_PATH"),
        ("src/uclone_x/cli/quality_gate.py", "FAILURE_LOG_PATH"),
    }
    # Drop JUNIT_REPORT_PATH from allowlist
    reduced_allowlist = {
        k: v
        for k, v in RELATIVE_PATH_CONSTANT_ALLOWLIST.items()
        if k != ("src/uclone_x/cli/quality_gate.py", "JUNIT_REPORT_PATH")
    }
    with pytest.raises(
        AssertionError,
        match="outside allowlist: src/uclone_x/cli/quality_gate.py:JUNIT_REPORT_PATH",
    ):
        verify_relative_path_constants(findings, reduced_allowlist)


def test_allowlist_direction_3_phantom_entry_fails() -> None:
    """Direction 3: adding a phantom entry to the allowlist fails the test (no unused entries)."""
    findings = {
        ("src/uclone_x/cli/quality_gate.py", "JUNIT_REPORT_PATH"),
        ("src/uclone_x/cli/quality_gate.py", "FAILURE_LOG_PATH"),
        ("src/uclone_x/cli/quality_gate.py", "LOCKFILE_PATH"),
    }
    phantom_allowlist = {
        **RELATIVE_PATH_CONSTANT_ALLOWLIST,
        ("src/uclone_x/cli/quality_gate.py", "STALE_PHANTOM_PATH"): "Documented phantom reason",
    }
    with pytest.raises(
        AssertionError,
        match=r"do not match any relative Path constant in source: src/uclone_x/cli/quality_gate.py:STALE_PHANTOM_PATH",
    ):
        verify_relative_path_constants(findings, phantom_allowlist)


def test_allowlist_direction_4_empty_or_missing_reason_fails() -> None:
    """Direction 4: removing or emptying an entry's reason fails the test (enforces reasons as required data)."""
    findings = {
        ("src/uclone_x/cli/quality_gate.py", "JUNIT_REPORT_PATH"),
        ("src/uclone_x/cli/quality_gate.py", "FAILURE_LOG_PATH"),
    }
    # Empty string reason
    empty_reason_allowlist = {
        **RELATIVE_PATH_CONSTANT_ALLOWLIST,
        ("src/uclone_x/cli/quality_gate.py", "JUNIT_REPORT_PATH"): "",
    }
    with pytest.raises(ValueError, match="must have a non-empty reason string"):
        verify_relative_path_constants(findings, empty_reason_allowlist)

    # Whitespace-only reason
    whitespace_reason_allowlist = {
        **RELATIVE_PATH_CONSTANT_ALLOWLIST,
        ("src/uclone_x/cli/quality_gate.py", "JUNIT_REPORT_PATH"): "   \n\t  ",
    }
    with pytest.raises(ValueError, match="must have a non-empty reason string"):
        verify_relative_path_constants(findings, whitespace_reason_allowlist)


@pytest.mark.parametrize(
    ("source_code", "expected_constants"),
    [
        ('SIMPLE = Path("rel/path")', {"SIMPLE"}),
        ('TYPED: Path = Path("rel/path")', {"TYPED"}),
        ('FINAL: Final[Path] = pathlib.Path("rel/path")', {"FINAL"}),
        ('PURE = PurePath("rel/path")', {"PURE"}),
        ('POSIX = PosixPath("rel/path")', {"POSIX"}),
        ('WIN = WindowsPath("rel/path")', {"WIN"}),
        ('MULTI1 = MULTI2 = Path("rel/path")', {"MULTI1", "MULTI2"}),
        ("NO_ARGS = Path()", {"NO_ARGS"}),
        ('DOT = Path(".")', {"DOT"}),
        ('CHAINED = Path("rel") / "sub"', {"CHAINED"}),
        ('if True:\n    IN_IF = Path("rel/path")', {"IN_IF"}),
        ('try:\n    IN_TRY = Path("rel/path")\nexcept Exception:\n    pass', {"IN_TRY"}),
        ('with dummy_ctx():\n    IN_WITH = Path("rel/path")', {"IN_WITH"}),
    ],
)
def test_relative_path_constant_detector_matches_claimed_shapes(
    source_code: str, expected_constants: set[str]
) -> None:
    """Verify that every shape claimed by the pattern is correctly detected."""
    tree = ast.parse(source_code)
    detected = extract_module_relative_path_constants(tree)
    assert detected == expected_constants, f"Failed on shape: {source_code}"


@pytest.mark.parametrize(
    "clean_source",
    [
        'ABS_POSIX = Path("/var/log")',
        'ABS_WIN = Path("C:/Users/test")',
        'local_var = Path("rel/path")',
        'def helper() -> None:\n    FUNC_CONST = Path("rel/path")',
        'class MyClass:\n    CLASS_CONST = Path("rel/path")',
        "TIMEOUT = 30",
        'NAME = "Path"',
        'ABS_CHAINED = Path("/var/log") / "sub"',
    ],
)
def test_relative_path_constant_detector_ignores_non_violating_shapes(clean_source: str) -> None:
    """Verify that non-violating shapes (absolute paths, lowercase, function/class scope) are ignored."""
    tree = ast.parse(clean_source)
    detected = extract_module_relative_path_constants(tree)
    assert detected == set(), f"Detector fired on clean shape: {clean_source}"


def test_cli_main_hooks_dir_falls_outside_module_level_constant_pattern() -> None:
    """Demonstrate that Path(".git") / Path(".git/hooks") in cli/main.py fall outside this pattern.

    As documented in Issue #440, Path(".git") and Path(".git/hooks") in cli/main.py
    were inspected to determine whether they belong in the allowlist or fall outside
    the module-level constant pattern.

    This test proves by AST inspection that:
    1. Neither occurrence is a module-level assignment; both reside inside the
       function symbol `_resolve_git_hooks_dir`.
    2. The variable bound to Path(".git/hooks") inside `_resolve_git_hooks_dir` is named
       `fallback` (lowercase identifier), which is a local variable rather than
       an uppercase module-level constant.
    3. The occurrence of Path(".git") is an expression inside an if condition, not an
       assignment target.
    4. Therefore, both occurrences fall outside the module-level relative Path
       constant pattern and do not require allowlisting.

    Rule #137 compliance: Located by symbol name `_resolve_git_hooks_dir`, with zero numeric line citations.
    """
    repo_root = Path(__file__).resolve().parents[2]
    main_py = repo_root / "src" / "uclone_x" / "cli" / "main.py"
    assert main_py.is_file(), f"main.py not found at {main_py}"

    tree = ast.parse(main_py.read_text(encoding="utf-8"), filename=str(main_py))

    # Module-level sweep on main.py yields no constants
    module_constants = extract_module_relative_path_constants(tree)
    assert "GIT" not in module_constants
    assert "HOOKS" not in module_constants
    assert len(module_constants) == 0

    # Locate the _resolve_git_hooks_dir function node by symbol
    hooks_func: ast.FunctionDef | None = None
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "_resolve_git_hooks_dir":
            hooks_func = stmt
            break

    assert hooks_func is not None, "Function _resolve_git_hooks_dir not found in cli/main.py"

    # Confirm that Path(".git") and Path(".git/hooks") exist inside _resolve_git_hooks_dir
    found_git_path = False
    found_fallback_hooks = False

    for node in ast.walk(hooks_func):
        if isinstance(node, ast.Call) and is_relative_path_call(node):
            if node.args and isinstance(node.args[0], ast.Constant):
                arg_val = node.args[0].value
                if arg_val == ".git":
                    found_git_path = True
                elif arg_val == ".git/hooks":
                    found_fallback_hooks = True

    assert found_git_path, "Path('.git') expected inside _resolve_git_hooks_dir"
    assert found_fallback_hooks, "Path('.git/hooks') expected inside _resolve_git_hooks_dir"


def test_installed_hook_drift_is_reported_with_both_byte_counts(tmp_path: Path) -> None:
    """A drifted installed hook is reported, naming installed and tracked sizes.

    The live copy of the pre-commit hook is checked by nothing: the existing drift test
    compares the tracked mirror to the constant, and git runs neither of those. On
    2026-09-03 the installed hook was edited in place (#285) and has differed ever since.
    Now that the hook carries the commit-identity refusal, an un-reinstalled hook means
    that control is simply not in effect, so the gate says so on every run.

    Mutation: return `[]` unconditionally from `describe_installed_hook_drift` — this test
    fails. Byte counts are asserted because they are the identifying fact about #285's
    hook, and because a report that cannot distinguish the two copies is not a report.
    """
    installed_text = "#!/usr/bin/env bash\nexit 0\n"
    tracked_text = "#!/usr/bin/env bash\n./ucx test check\n"
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text(installed_text, encoding="utf-8")

    lines = quality_gate.describe_installed_hook_drift(hooks_dir=hooks, expected=tracked_text)

    assert lines, "drift between installed and tracked was not reported"
    joined = "\n".join(lines)
    assert "differs from the tracked source" in joined
    assert "./ucx setup" in joined, "a report without the remedy leaves the builder stuck"
    # Computed, not transcribed: a hand-copied count silently stops matching the strings
    # above the moment either is edited, and would then assert nothing.
    assert f"{len(installed_text.encode('utf-8'))} bytes" in joined
    assert f"{len(tracked_text.encode('utf-8'))} bytes" in joined


def test_installed_hook_matching_the_constant_reports_nothing(tmp_path: Path) -> None:
    """An installed hook identical to the tracked source produces no noise.

    Mutation: compare with `!=` instead of `==` — this test fails. A warning that fires
    when nothing is wrong is one every builder learns to scroll past, which would cost the
    case above its only reader.
    """
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    content = "#!/usr/bin/env bash\n./ucx test check\n"
    (hooks / "pre-commit").write_text(content, encoding="utf-8")

    assert quality_gate.describe_installed_hook_drift(hooks_dir=hooks, expected=content) == []


def test_absent_installed_hook_is_not_reported_as_drift(tmp_path: Path) -> None:
    """No installed hook at all is silence, not a drift report.

    Mutation: treat a missing file as drift — this test fails. A fresh clone has no hook
    until `./ucx setup` runs, and reporting that as drift would train the reader to
    dismiss the message before the real case ever appears.
    """
    hooks = tmp_path / "hooks"
    hooks.mkdir()

    assert quality_gate.describe_installed_hook_drift(hooks_dir=hooks, expected="anything") == []


# --- Gate scope (#588) ----------------------------------------------------------------


def _top_level_python_dirs(repo_root: Path) -> set[str]:
    """Top-level directories holding at least one `.py` file, excluding vendored trees."""
    skip_parts = {"node_modules", "__pycache__", ".venv", ".git", ".worktrees"}
    found: set[str] = set()
    for entry in repo_root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name in skip_parts:
            continue
        for path in entry.rglob("*.py"):
            if skip_parts & set(path.parts):
                continue
            found.add(entry.name)
            break
    return found


def test_quality_gate_scope_covers_every_python_directory() -> None:
    """Every top-level Python directory is either checked or explicitly excused.

    The gate is the only verification this repository has -- there is no hosted CI -- so a
    directory in neither tuple is unchecked everywhere, and the omission is invisible
    precisely because the gate still prints green. `evals/` sat in that gap until #588:
    ~6,000 lines Ruff never inspected, while pyright did, which is why nobody noticed.
    """
    repo_root = Path(__file__).resolve().parents[2]
    discovered = _top_level_python_dirs(repo_root)
    accounted = set(PYTHON_CHECK_PATHS) | set(PYTHON_UNCHECKED_PATHS)

    unaccounted = discovered - accounted
    assert not unaccounted, (
        f"Top-level Python directories in neither PYTHON_CHECK_PATHS nor "
        f"PYTHON_UNCHECKED_PATHS: {sorted(unaccounted)}. Add each to one of them -- "
        "excluding a directory is a decision to record, not a silence to keep."
    )


def test_every_excused_directory_states_a_reason() -> None:
    """An exclusion without a reason is the silence this mechanism exists to prevent."""
    for name, reason in PYTHON_UNCHECKED_PATHS.items():
        assert reason.strip(), f"{name!r} is excused from the gate with no reason given"
        assert len(reason.strip()) > 40, (
            f"{name!r}'s exclusion reason is too terse to be checkable: {reason!r}"
        )


def test_check_and_unchecked_paths_are_disjoint() -> None:
    """A directory cannot be both checked and excused; that would make the excuse a lie."""
    overlap = set(PYTHON_CHECK_PATHS) & set(PYTHON_UNCHECKED_PATHS)
    assert not overlap, f"Directories in both tuples: {sorted(overlap)}"


def test_evals_is_inside_the_gate_scope() -> None:
    """Pin #588 directly: `evals/` must stay checked.

    Written as its own assertion rather than relying on the discovery test, which would
    also pass if `evals` were moved into PYTHON_UNCHECKED_PATHS.
    """
    assert "evals" in PYTHON_CHECK_PATHS


def test_measure_package_coverage_parses_the_total_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """The percentage is read from the TOTAL row's last %-suffixed field."""
    report = (
        "Name                          Stmts   Miss Branch BrPart  Cover   Missing\n"
        "------------------------------------------------------------------------\n"
        "src/uclone_x/agent/base.py      848     54    290     51    90%   266\n"
        "------------------------------------------------------------------------\n"
        "TOTAL                          1747    140    482     73    89%\n"
    )

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=report)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert quality_gate.measure_package_coverage("agent") == 89


def test_measure_package_coverage_returns_none_without_a_total_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`coverage report` prints no TOTAL when nothing matched the include pattern."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=2, stdout="No data to report.\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert quality_gate.measure_package_coverage("nope") is None


def test_package_floors_report_every_shortfall_not_just_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gate that stopped at the first shortfall would hide the rest of the work."""
    measured = {"agent": 70, "core": 95, "ui": 60}

    def fake_measure(package: str) -> int | None:
        return measured[package]

    monkeypatch.setattr(quality_gate, "measure_package_coverage", fake_measure)

    shortfalls = real_check_package_coverage_floors({"agent": 85, "core": 85, "ui": 78})
    assert shortfalls == [("agent", 70, 85), ("ui", 60, 78)]


def test_unmeasurable_package_is_a_shortfall_not_a_silent_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A floor declared for a package coverage cannot see must fail, not disappear.

    This is the failure mode the check exists to avoid: a typo in the floor table would
    otherwise switch the check off for that package while the gate still reported green.

    Killed by: src/uclone_x/cli/quality_gate.py :: shortfalls.append((package, -1, floor))
    """

    def unmeasurable(_package: str) -> int | None:
        return None

    monkeypatch.setattr(quality_gate, "measure_package_coverage", unmeasurable)

    assert real_check_package_coverage_floors({"agnet": 85}) == [("agnet", -1, 85)]


# ── Lockfile freshness stage (uv.lock vs pyproject.toml) ──────────────────────────────
#
# Written against drift, not against agreement. `uv lock --check` on this repository's own
# tree passes whenever the tree is healthy, so a test that asserted that would have been
# green on the day the lock was wrong -- which is the day it was wrong on `main`, twice.
# Each test below therefore constructs a lockfile that disagrees with its `pyproject.toml`
# and asserts the gate refuses it.


def _write_locked_project(root: Path) -> None:
    """A minimal, dependency-free project with a lockfile `uv` itself just generated.

    Dependency-free on purpose: resolution touches no index, so these tests stay inside
    the Tier 1 promise of zero network calls, and `UV_OFFLINE` below makes that a
    constraint rather than an expectation.
    """
    (root / "pyproject.toml").write_text(
        '[project]\nname = "drift-probe"\nversion = "0.1.0"\n'
        'requires-python = ">=3.11"\ndependencies = []\n',
        encoding="utf-8",
    )
    generated = subprocess.run(
        ["uv", "lock"], cwd=root, capture_output=True, text=True, check=False
    )
    assert generated.returncode == 0, generated.stderr


needs_uv = pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="`uv` is not installed; the stage's behaviour without it is pinned separately",
)


@needs_uv
def test_a_lockfile_that_disagrees_with_pyproject_is_reported_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real case: `pyproject.toml` moved and nobody relocked.

    This reproduces what happened on `main` in miniature -- #654 bumped `version` and left
    `uv.lock` naming the old one -- by doing the same edit to a throwaway project and
    running the same `uv lock --check` the gate runs. No mock: the point is that the stage
    detects genuine drift, and a stubbed `uv` would only prove that a stub returns what it
    was told to.

    Killed by: src/uclone_x/cli/quality_gate.py :: if result.returncode == 0:
    """
    monkeypatch.setenv("UV_OFFLINE", "1")
    _write_locked_project(tmp_path)

    assert real_check_lockfile_freshness(tmp_path)[0] == "fresh"

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "drift-probe"\nversion = "0.2.0"\n'
        'requires-python = ">=3.11"\ndependencies = []\n',
        encoding="utf-8",
    )

    status, lines = real_check_lockfile_freshness(tmp_path)
    assert status == "stale"
    # Naming the fix is the whole value of the message. A builder who reads "out of date"
    # and not "run `uv lock`" reaches for `uv sync`, which repoints the shared venv at
    # their worktree and breaks every other checkout (#679).
    assert any("uv lock" in line for line in lines)


@needs_uv
def test_a_drifted_lockfile_fails_the_whole_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Drift must make `./ucx test check` red, not merely print a note.

    The stage is wired in with `fail_fast`, so this also pins the ordering: every other
    command is mocked to succeed, and the gate must still stop before running any of them.

    Killed by: src/uclone_x/cli/quality_gate.py :: if lock_status in ("stale", "uv-missing"):
    """
    monkeypatch.setenv("UV_OFFLINE", "1")
    _write_locked_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "drift-probe"\nversion = "0.2.0"\n'
        'requires-python = ">=3.11"\ndependencies = []\n',
        encoding="utf-8",
    )

    # Resolved *before* `subprocess.run` is mocked below: the drift is measured by the
    # real `uv`, and the mock that neutralises the later stages would otherwise intercept
    # it and report the lockfile fresh.
    verdict = real_check_lockfile_freshness(tmp_path)
    assert verdict[0] == "stale", verdict

    def drifted() -> tuple[quality_gate.LockfileStatus, list[str]]:
        return verdict

    monkeypatch.setattr(quality_gate, "check_lockfile_freshness", drifted)
    ran: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        ran.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = run_quality_gate(quiet=False, junit_path=tmp_path / "junit.xml")

    assert exit_code != 0
    assert ran == [], "the gate ran later stages against an unidentified dependency set"
    captured = " ".join(capsys.readouterr().out.split())
    assert "uv lock" in captured


def test_a_missing_uv_fails_the_stage_instead_of_skipping_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `uv` on PATH with a lockfile present is a failure, not a quiet pass.

    This is the decision the stage turns on. A skip prints the same green as a check that
    ran, so the state a builder most needs to see -- "the lockfile was never verified" --
    would be the one state the output hides, and this repository has no hosted CI to
    verify it anywhere else (P8). The exception below is what `subprocess.run` raises when
    the executable is absent.

    Killed by: src/uclone_x/cli/quality_gate.py :: return "uv-missing", [
    """
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    def no_uv(cmd: list[str], *args: object, **kwargs: object) -> object:
        raise FileNotFoundError(2, "No such file or directory: 'uv'")

    monkeypatch.setattr(subprocess, "run", no_uv)

    status, lines = real_check_lockfile_freshness(tmp_path)
    assert status == "uv-missing"
    assert any("uv" in line for line in lines)
    assert any("failure" in line for line in lines), "it must say why it is not a skip"


def test_a_tree_without_a_lockfile_is_absent_and_not_a_failure(tmp_path: Path) -> None:
    """`absent` is a third answer, distinct from both `fresh` and `uv-missing`.

    A source export or partial checkout may carry no `uv.lock`; failing on a file the tree
    never claimed to have would be an assertion about someone else's tree. It still says
    so out loud rather than returning nothing, so the reader can tell "not applicable" from
    "checked and fine".

    Killed by: src/uclone_x/cli/quality_gate.py :: return "absent", [
    """
    status, lines = real_check_lockfile_freshness(tmp_path)
    assert status == "absent"
    assert lines, "an unrun check must say it did not run"


def test_stale_lock_without_fail_fast_still_fails_after_later_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `fail_fast=False` the remaining stages run, and the verdict is still red.

    The mutation this exists to catch is the one that makes the stage advisory: report the
    drift, record no failure, and let a green total swallow it.

    Killed by: src/uclone_x/cli/quality_gate.py :: first_failure = _LOCKFILE_FAILURE_EXIT
    """

    def stale() -> tuple[quality_gate.LockfileStatus, list[str]]:
        return "stale", ["stale"]

    monkeypatch.setattr(quality_gate, "check_lockfile_freshness", stale)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = run_quality_gate(
        quiet=True,
        fail_fast=False,
        junit_path=tmp_path / "junit.xml",
        failure_log_path=tmp_path / "failure_history.jsonl",
    )

    assert exit_code == 1
    assert [c[0] for c in calls] == ["ruff", "ruff", "pyright", "pytest", "pytest"]


# --- Gate stage 1b: environment provenance, as the gate wires it -----------------------
#
# Everything above about the stage's *verdict* lives in
# `tests/unit/test_cli_environment_provenance.py`. What follows is the other half: what
# `run_quality_gate` does with that verdict. Without these, `stub_environment_provenance`
# above answers `"ok"` for every test in this module and the wiring is measured by nothing
# — replacing `BLOCKING_STATUSES` with `frozenset()` in `quality_gate.py` killed 0 of the
# 3176 nodes in the `not recorded and not live` selection, while the gate printed stage
# 1b's red report and still ended "ALL SYSTEMS GO".
#
# Each test below re-patches `quality_gate.check_environment_provenance` **inside its own
# body**, which is how it gets past the module-wide autouse stub: the stub runs first at
# setup time, the body's `setattr` lands on top of it, and `monkeypatch` undoes both in
# reverse order. Disabling the fixture itself would be the other way to do it; this way
# the override is visible at the point of use rather than in a decorator.


def _blocking_provenance(
    monkeypatch: pytest.MonkeyPatch, status: ProvenanceStatus
) -> list[list[str]]:
    """Drive stage 1b to `status`, mock every later command to succeed, return the log."""

    def verdict() -> tuple[ProvenanceStatus, list[str]]:
        return status, [f"probe says {status}"]

    monkeypatch.setattr(quality_gate, "check_environment_provenance", verdict)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("status", sorted(BLOCKING_STATUSES))
def test_a_blocking_provenance_status_fails_the_whole_gate(
    status: ProvenanceStatus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An environment unfit to measure this tree must make the gate red, not merely noisy.

    The failure this comes from presented as a *green* gate with a warning above it, so a
    test that only asserted the report would be satisfied by the bug. This asserts the
    verdict: exit non-zero, and no later stage run at all — nothing measured after this
    point would have been about this tree.

    Killed by: src/uclone_x/cli/quality_gate.py :: if env_status in BLOCKING_STATUSES:

    Killed by: src/uclone_x/cli/quality_gate.py :: return _ENVIRONMENT_FAILURE_EXIT
    """
    calls = _blocking_provenance(monkeypatch, status)

    exit_code = run_quality_gate(quiet=True, junit_path=tmp_path / "junit.xml")

    assert exit_code == 1
    assert calls == [], "the gate ran later stages in an environment it had just refused"


def test_blocking_provenance_without_fail_fast_still_fails_after_later_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `fail_fast=False` the rest of the gate runs, and the verdict is still red.

    The mutation this exists to catch is the one that makes the stage advisory: print the
    refusal, record no failure, and let a green total swallow it — which is the shape the
    original defect had.

    Killed by: src/uclone_x/cli/quality_gate.py :: first_failure = _ENVIRONMENT_FAILURE_EXIT
    """
    calls = _blocking_provenance(monkeypatch, "worktree-venv")

    exit_code = run_quality_gate(
        quiet=True,
        fail_fast=False,
        junit_path=tmp_path / "junit.xml",
        failure_log_path=tmp_path / "failure_history.jsonl",
    )

    assert exit_code == 1
    assert [c[0] for c in calls] == ["ruff", "ruff", "pyright", "pytest", "pytest"]


@pytest.mark.parametrize("status", ["ok", "ambient-foreign", "worktree-venv-declared"])
def test_a_non_blocking_provenance_status_does_not_stop_the_gate(
    status: ProvenanceStatus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive control for the two tests above, and the other direction of the split.

    `ambient-foreign` is the unsatisfiable case — one editable install cannot name every
    worktree at once — so it warns. `worktree-venv-declared` is the case AGENTS.md:82
    permits outright, a branch that genuinely requires different dependency versions
    (#968); the gate refused it until then, so this parameter is the wiring half of that
    fix — `environment_provenance` can return the status all it likes if the gate still
    stops on it. If the two tests above went red because of the harness rather than
    because of the status, this one would go red with them.
    """
    calls = _blocking_provenance(monkeypatch, status)

    exit_code = run_quality_gate(quiet=True, junit_path=tmp_path / "junit.xml")

    assert exit_code == 0
    assert [c[0] for c in calls] == ["ruff", "ruff", "pyright", "pytest", "pytest"]


# --- Gate stage 6: the frontend vitest suite (#915) ------------------------------------
#
# Until #915 the suite ran only behind `-fe`, which no hook, no documented command and no
# CI passed — so ~125 vitest cases were written, and then consulted by nobody. The tests
# below are written in the failure direction on purpose: a stage that cannot run must turn
# the gate red, because a skipped suite prints the same green as a passing one (P6).
#
# The selection tests read the call log of `stub_frontend_suite` (autouse, above). The
# stage-function tests call the real `run_frontend_suite` against a `tmp_path` tree, so
# neither half depends on whether the worktree running them has `node_modules`.


def _gate_calls_with_passing_subprocesses(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize(
    ("scope", "runs_vitest"),
    [
        ("gate", True),
        ("fast", True),
        ("all", True),
        ("unit", False),
        ("fitness", False),
        ("recorded", False),
        ("live", False),
        ("e2e", False),
        ("pre-release", False),
    ],
)
def test_the_frontend_suite_runs_in_every_scope_a_commit_is_measured_against(
    scope: str,
    runs_vitest: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_frontend_suite: list[Path | None],
    stub_bundle_freshness: list[Path | None],
) -> None:
    """`gate` (plain `./ucx test check`), `--fast` and `--all` run vitest; tiers do not.

    The bundle freshness check (stage 6b, #878) is selected with it, scope for scope.

    `fast` is included deliberately: it is documented as "the gate without the browser suite
    and nothing else", and vitest costs about a second and a half. A `--fast` that also
    dropped the frontend suite would be a second, weaker policy nobody declared. The named
    tiers (`unit`, `e2e`, ...) are pytest selections for narrowing a failure and stay Python
    only, as they were.

    Killed by: src/uclone_x/cli/quality_gate.py :: frozenset({"gate", "fast", "all"})
    Becomes: frozenset({"fast", "all"})
    """
    _gate_calls_with_passing_subprocesses(monkeypatch)

    exit_code = run_quality_gate(
        quiet=True,
        test_scope=scope,
        junit_path=tmp_path / "junit.xml",
        failure_log_path=tmp_path / "failure_history.jsonl",
    )

    assert exit_code == 0
    assert len(stub_frontend_suite) == (1 if runs_vitest else 0)
    assert len(stub_bundle_freshness) == len(stub_frontend_suite)


def test_skip_tests_skips_the_frontend_suite_too(
    monkeypatch: pytest.MonkeyPatch, stub_frontend_suite: list[Path | None]
) -> None:
    """The pre-commit hook runs `--skip-tests`: static checks only, so no vitest either.

    The reason the suite is not run at commit time is the one the hook already gives for
    pytest — the tree a commit holds is not the tree that reaches `main` — and it applies to
    vitest unchanged. A plain `./ucx test check` is where it runs.

    Killed by: src/uclone_x/cli/quality_gate.py :: not skip_tests and test_scope in _FRONTEND_SUITE_SCOPES
    Becomes: test_scope in _FRONTEND_SUITE_SCOPES
    """
    _gate_calls_with_passing_subprocesses(monkeypatch)

    assert run_quality_gate(quiet=True, skip_tests=True) == 0
    assert stub_frontend_suite == []


def test_check_frontend_adds_the_suite_to_any_scope(
    monkeypatch: pytest.MonkeyPatch, stub_frontend_suite: list[Path | None]
) -> None:
    """`-fe` keeps its old meaning on scopes that do not select vitest themselves.

    Killed by: src/uclone_x/cli/quality_gate.py :: return check_frontend or (
    Becomes: return (
    """
    _gate_calls_with_passing_subprocesses(monkeypatch)

    assert run_quality_gate(quiet=True, test_scope="e2e", check_frontend=True) == 0
    assert len(stub_frontend_suite) == 1


def test_the_default_gate_never_runs_the_frontend_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production build stays behind `-fe`, and this is the #878 boundary.

    `vite build` writes `src/uclone_x/ui_static`, which is **committed**. A default gate
    that built would rewrite tracked files on every push — dirtying the tree it had just
    verified — and would silently settle #878 (whether the committed bundle may diverge
    from its source) by overwriting the evidence. Vitest reads the source and writes
    nothing tracked, which is why it alone joins the default gate.

    Killed by: src/uclone_x/cli/quality_gate.py :: if check_frontend:
    Becomes: if True:
    """
    calls = _gate_calls_with_passing_subprocesses(monkeypatch)

    exit_code = run_quality_gate(
        quiet=True,
        junit_path=tmp_path / "junit.xml",
        failure_log_path=tmp_path / "failure_history.jsonl",
    )

    assert exit_code == 0
    assert [c[0] for c in calls] == ["ruff", "ruff", "pyright", "pytest", "pytest"]


@pytest.mark.parametrize("fail_fast", [True, False])
@pytest.mark.parametrize("status", ["deps-missing", "npm-missing"])
def test_an_unrunnable_frontend_suite_fails_the_whole_gate(
    status: quality_gate.FrontendSuiteStatus,
    fail_fast: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A suite that could not run is a red gate, not a quiet green one (P6).

    This is the decision the stage turns on. A fresh worktree has no `frontend/node_modules`
    (it is gitignored, and ~216MB), so "skip when absent" would skip in exactly the place
    builders run the gate, and #915 would be back — a suite that exists and runs nowhere —
    with a green line printed over it.

    Killed by: src/uclone_x/cli/quality_gate.py :: if fe_code != 0:
    Becomes: if fe_code > _FRONTEND_UNRUNNABLE_EXIT:
    """
    _gate_calls_with_passing_subprocesses(monkeypatch)

    def unrunnable(
        root: Path | None = None,
        *,
        quiet: bool = False,
    ) -> tuple[quality_gate.FrontendSuiteStatus, int, list[str]]:
        return status, 1, [f"cannot run: {status}"]

    monkeypatch.setattr(quality_gate, "run_frontend_suite", unrunnable)

    exit_code = run_quality_gate(
        quiet=False,
        fail_fast=fail_fast,
        junit_path=tmp_path / "junit.xml",
        failure_log_path=tmp_path / "failure_history.jsonl",
    )

    assert exit_code != 0
    out = capsys.readouterr().out
    assert f"cannot run: {status}" in out
    assert "ALL SYSTEMS GO" not in out


def _frontend_tree(root: Path, *, node_modules: bool) -> Path:
    frontend = root / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text('{"scripts": {"test": "vitest run"}}\n')
    if node_modules:
        (frontend / "node_modules").mkdir()
    return frontend


def _npm_on_path(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if present else None

    monkeypatch.setattr(quality_gate.shutil, "which", which)


def _forbid_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_run(cmd: list[str], *args: object, **kwargs: object) -> object:
        raise AssertionError(f"the stage ran {cmd} when it should have refused first")

    monkeypatch.setattr(subprocess, "run", must_not_run)


def test_a_frontend_without_installed_dependencies_fails_instead_of_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `node_modules` is `deps-missing`, non-zero, and the message names the fix.

    Naming the fix is most of the value: the cheap one in a worktree is a symlink to the
    primary workspace's `node_modules` (the frontend analogue of the shared `.venv`), not a
    216MB `npm ci` per worktree.

    Killed by: src/uclone_x/cli/quality_gate.py :: return "deps-missing", _FRONTEND_UNRUNNABLE_EXIT, no_deps
    Becomes: return "passed", 0, no_deps
    """
    _frontend_tree(tmp_path, node_modules=False)
    _npm_on_path(monkeypatch, present=True)
    _forbid_subprocess(monkeypatch)

    status, code, lines = real_run_frontend_suite(tmp_path)

    assert status == "deps-missing"
    assert code != 0
    text = " ".join(lines)
    assert "node_modules" in text
    assert "ln -s" in text
    assert "npm ci" in text


def test_a_broken_node_modules_symlink_is_deps_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recommended remedy is a symlink, so a dangling one must not read as installed.

    Killed by: src/uclone_x/cli/quality_gate.py :: if not (frontend / "node_modules").is_dir():
    Becomes: if not (frontend / "node_modules").is_symlink() and not (frontend / "node_modules").exists():
    """
    frontend = _frontend_tree(tmp_path, node_modules=False)
    (frontend / "node_modules").symlink_to(tmp_path / "nowhere")
    _npm_on_path(monkeypatch, present=True)
    _forbid_subprocess(monkeypatch)

    status, code, _ = real_run_frontend_suite(tmp_path)

    assert (status, code != 0) == ("deps-missing", True)


def test_a_missing_npm_fails_the_frontend_stage_instead_of_skipping_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `npm` on PATH is `npm-missing` and non-zero — the lockfile stage's rule for `uv`.

    Killed by: src/uclone_x/cli/quality_gate.py :: return "npm-missing", _FRONTEND_UNRUNNABLE_EXIT, not_on_path
    Becomes: return "npm-missing", 0, not_on_path
    """
    _frontend_tree(tmp_path, node_modules=True)
    _npm_on_path(monkeypatch, present=False)
    _forbid_subprocess(monkeypatch)

    status, code, lines = real_run_frontend_suite(tmp_path)

    assert status == "npm-missing"
    assert code != 0
    assert any("npm" in line for line in lines)


def test_npm_vanishing_between_lookup_and_run_is_npm_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`subprocess.run` raising is the same answer as `which` failing, not a traceback.

    Killed by: src/uclone_x/cli/quality_gate.py :: except OSError as npm_exc:
    Becomes: except LookupError as npm_exc:
    """
    _frontend_tree(tmp_path, node_modules=True)
    _npm_on_path(monkeypatch, present=True)

    def no_npm(cmd: list[str], *args: object, **kwargs: object) -> object:
        raise FileNotFoundError(2, "No such file or directory: 'npm'")

    monkeypatch.setattr(subprocess, "run", no_npm)

    status, code, _ = real_run_frontend_suite(tmp_path)

    assert (status, code != 0) == ("npm-missing", True)


def test_a_tree_without_a_frontend_is_absent_and_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `frontend/package.json` is `absent`: the tree never claimed a frontend suite.

    Same third answer as the lockfile stage's `absent`, and it still says so out loud.

    Killed by: src/uclone_x/cli/quality_gate.py :: return "absent", 0, absent
    Becomes: return "absent", 1, absent
    """
    _npm_on_path(monkeypatch, present=True)
    _forbid_subprocess(monkeypatch)

    status, code, lines = real_run_frontend_suite(tmp_path)

    assert (status, code) == ("absent", 0)
    assert lines, "an unrun check must say it did not run"


@pytest.mark.parametrize(("returncode", "expected"), [(0, "passed"), (1, "failed")])
def test_the_stage_runs_vitest_in_the_frontend_directory_and_keeps_its_exit_code(
    returncode: int,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runnable path: `npm test` (`vitest run`, non-watch) in `frontend/`, verdict kept.

    Killed by: src/uclone_x/cli/quality_gate.py :: if vitest.returncode == 0:
    Becomes: if vitest.returncode != 0:
    """
    frontend = _frontend_tree(tmp_path, node_modules=True)
    _npm_on_path(monkeypatch, present=True)
    seen: list[tuple[list[str], object]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen.append((cmd, kwargs.get("cwd")))
        return subprocess.CompletedProcess(args=cmd, returncode=returncode)

    monkeypatch.setattr(subprocess, "run", fake_run)

    status, code, _ = real_run_frontend_suite(tmp_path)

    assert (status, code) == (expected, returncode)
    assert seen == [(["npm", "test", "--silent"], frontend)]


def test_the_vitest_stage_starts_blocking_and_a_quiet_one_reports_no_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Stage 6 runs on the gate's stdio through `run_stage`, like every other such stage (#999).

    Stdio is made non-blocking before the call, and the fake `npm` leaves it non-blocking when
    it exits, as a killed `node` does. The stage must start blocking, leave stdio blocking, and
    under `quiet` say nothing about the restore. `capfd` owns fds 1 and 2 for the duration.

    Killed by: src/uclone_x/cli/quality_gate.py :: vitest = run_stage(["npm", "test", "--silent"], cwd=frontend, quiet=quiet)
    Becomes: vitest = subprocess.run(["npm", "test", "--silent"], cwd=frontend)
    Killed by: src/uclone_x/cli/quality_gate.py :: vitest = run_stage(["npm", "test", "--silent"], cwd=frontend, quiet=quiet)
    Becomes: vitest = run_stage(["npm", "test", "--silent"], cwd=frontend)
    """
    _frontend_tree(tmp_path, node_modules=True)
    _npm_on_path(monkeypatch, present=True)
    started: list[tuple[bool, bool]] = []

    def npm(cmd: list[str], *args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        started.append((os.get_blocking(1), os.get_blocking(2)))
        _set_stdio_non_blocking()
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", npm)
    was_blocking = (os.get_blocking(1), os.get_blocking(2))
    _set_stdio_non_blocking()
    try:
        status, code, _ = real_run_frontend_suite(tmp_path, quiet=True)
        blocking_after = (os.get_blocking(1), os.get_blocking(2))
    finally:
        os.set_blocking(1, was_blocking[0])
        os.set_blocking(2, was_blocking[1])

    assert (status, code) == ("passed", 0)
    assert started == [(True, True)]
    assert blocking_after == (True, True)
    assert capfd.readouterr() == ("", "")


# --------------------------------------------------------------------------------------
# The parallel pytest stage (#967)
# --------------------------------------------------------------------------------------

_WHOLLY_PARALLEL_SCOPES = ("fast", "unit", "fitness")
_SPLIT_SCOPES = ("gate", "all")
_SERIAL_TEST_SCOPES = ("e2e", "recorded", "live", "pre-release")


def _worker_flags(cmd: Sequence[str]) -> list[str]:
    """The pytest-xdist arguments in a built argv, in order; empty for a serial command."""
    flags: list[str] = []
    if "-n" in cmd:
        index = list(cmd).index("-n")
        flags.extend(cmd[index : index + 2])
    flags.extend(arg for arg in cmd if arg.startswith("--dist"))
    return flags


def _marker(cmd: Sequence[str]) -> str:
    return cmd[list(cmd).index("-m") + 1]


def _junit_argument(cmd: Sequence[str]) -> Path:
    return Path(next(arg for arg in cmd if arg.startswith("--junitxml=")).split("=", 1)[1])


def _report(cases: dict[str, bool]) -> str:
    """A minimal junit report: `{test name: failed}`."""
    body = "".join(
        f'<testcase classname="tests.unit.test_x" name="{name}">'
        + ('<failure message="boom">AssertionError</failure>' if failed else "")
        + "</testcase>"
        for name, failed in cases.items()
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>'
    )


def _run_gate_recording_commands(
    monkeypatch: pytest.MonkeyPatch,
    *,
    serial: bool = False,
    fail_fast: bool = True,
    junit_path: Path | None = None,
    step_outcomes: Sequence[tuple[int, dict[str, bool] | None]] = (),
) -> tuple[int, list[list[str]]]:
    """Run the gate with every subprocess faked; the N-th pytest call gets `step_outcomes[N]`.

    An outcome is `(returncode, cases)`; `cases` are written to that call's `--junitxml` path
    as a real pytest would, or nothing is written when `cases` is None.
    """
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[0] != "pytest":
            return subprocess.CompletedProcess(args=cmd, returncode=0)
        index = sum(1 for c in calls if c[0] == "pytest") - 1
        code, cases = step_outcomes[index] if index < len(step_outcomes) else (0, None)
        if cases is not None:
            report = _junit_argument(cmd)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(_report(cases), encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=code)

    monkeypatch.setattr(subprocess, "run", fake_run)
    kwargs: dict[str, Any] = (
        {}
        if junit_path is None
        else {
            "junit_path": junit_path,
            "failure_log_path": junit_path.with_name("failure_history.jsonl"),
        }
    )
    code = run_quality_gate(
        quiet=True, test_scope="gate", serial=serial, fail_fast=fail_fast, **kwargs
    )
    return code, calls


def test_every_declared_scope_is_classified_exactly_once() -> None:
    """A scope added later has to be placed in one list, not fall through to a default."""
    groups = (_WHOLLY_PARALLEL_SCOPES, _SPLIT_SCOPES, _SERIAL_TEST_SCOPES)
    placed = [scope for group in groups for scope in group]
    assert sorted(placed) == sorted(quality_gate.TEST_SCOPES)


@pytest.mark.parametrize("scope", _WHOLLY_PARALLEL_SCOPES)
def test_a_scope_without_the_browser_suite_runs_in_one_step_on_workers(scope: str) -> None:
    """`fast`, `unit` and `fitness` run whole on workers, in one step.

    Killed by: src/uclone_x/cli/quality_gate.py ::
        cmd.extend(["-n", str(pytest_worker_count()), f"--dist={_PYTEST_DIST_MODE}"])
    Becomes: cmd.extend([])
    """
    steps = quality_gate.build_pytest_steps(scope)
    assert len(steps) == 1
    expected = ["-n", str(quality_gate.pytest_worker_count()), "--dist=load"]
    assert _worker_flags(steps[0].argv) == expected


@pytest.mark.parametrize("scope", _SERIAL_TEST_SCOPES)
def test_the_browser_suite_and_the_scopes_outside_the_gate_stay_serial(scope: str) -> None:
    """`e2e` races its own page loads under load (#942, #946); the rest have costs of their own.

    Killed by: src/uclone_x/cli/quality_gate.py :: frozenset({"fast", "unit", "fitness"})
    Becomes: frozenset({"fast", "unit", "fitness", "e2e", "live", "recorded", "pre-release"})
    """
    steps = quality_gate.build_pytest_steps(scope)
    assert len(steps) == 1
    assert _worker_flags(steps[0].argv) == []


@pytest.mark.parametrize("scope", _SPLIT_SCOPES)
def test_the_gate_runs_the_browser_suite_after_the_workers_in_one_process(scope: str) -> None:
    """Step 1 is everything but E2E on workers; step 2 is E2E alone, appending coverage.

    Killed by: src/uclone_x/cli/quality_gate.py :: f"({expression}) and e2e",
    Becomes: f"({expression})",
    """
    expression = _marker(quality_gate.build_pytest_command(scope))
    workers, browser = quality_gate.build_pytest_steps(scope)

    assert _worker_flags(workers.argv) == [
        "-n",
        str(quality_gate.pytest_worker_count()),
        "--dist=load",
    ]
    assert _marker(workers.argv) == f"({expression}) and not e2e"
    assert _worker_flags(browser.argv) == []
    assert _marker(browser.argv) == f"({expression}) and e2e"
    assert workers.junit_path != browser.junit_path
    assert _junit_argument(workers.argv) == workers.junit_path
    assert _junit_argument(browser.argv) == browser.junit_path


@pytest.mark.parametrize("scope", _SPLIT_SCOPES)
def test_coverage_is_judged_once_on_the_combined_data(scope: str) -> None:
    """The partial first step must not fail on coverage; the second must add to it, not replace it.

    Killed by: src/uclone_x/cli/quality_gate.py :: "--cov-append",
    Becomes: "--cov-report=term-missing",
    """
    workers, browser = quality_gate.build_pytest_steps(scope)
    assert "--cov-fail-under=0" in workers.argv
    assert "--cov-append" not in workers.argv
    assert "--cov-append" in browser.argv
    assert not any(arg.startswith("--cov-fail-under") for arg in browser.argv)
    assert "--no-cov" not in workers.argv and "--no-cov" not in browser.argv


@pytest.mark.parametrize("scope", (*_WHOLLY_PARALLEL_SCOPES, *_SPLIT_SCOPES))
def test_the_serial_escape_is_one_process_with_the_same_selection(scope: str) -> None:
    """`--serial` is for debugging an ordering failure, so it must change only the ordering.

    Killed by: src/uclone_x/cli/quality_gate.py :: if test_scope in _PARALLEL_SCOPES and not serial:
    Becomes: if test_scope in _PARALLEL_SCOPES:
    """
    steps = quality_gate.build_pytest_steps(scope, serial=True)
    assert len(steps) == 1
    serial = list(steps[0].argv)
    assert _worker_flags(serial) == []
    assert serial == quality_gate.build_pytest_command(scope, serial=True)
    parallel = quality_gate.build_pytest_command(scope)
    if "-n" in parallel:
        at = parallel.index("-n")
        parallel = parallel[:at] + parallel[at + len(_worker_flags(parallel)) :]
    assert parallel == serial


def test_the_worker_count_is_one_per_cpu_capped_and_never_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One worker per CPU, capped, and never zero.

    Killed by: src/uclone_x/cli/quality_gate.py :: return max(1, min(available or 1, _MAX_PYTEST_WORKERS))
    Becomes: return max(1, available or 1)
    """
    assert quality_gate.pytest_worker_count(1) == 1
    assert quality_gate.pytest_worker_count(4) == 4
    assert quality_gate.pytest_worker_count(0) == 1
    ceiling = quality_gate.pytest_worker_count(10_000)
    assert 1 <= ceiling < 10_000
    assert quality_gate.pytest_worker_count(ceiling + 1) == ceiling

    monkeypatch.setattr(quality_gate.os, "cpu_count", lambda: None)
    assert quality_gate.pytest_worker_count() == 1


def test_the_gate_merges_both_steps_into_the_report_it_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failure in either step must reach the combined report, the console and the history.

    Killed by: src/uclone_x/cli/quality_gate.py :: combined.extend(suites)
    Becomes: pass
    """
    junit = tmp_path / "report" / "junit.xml"
    code, calls = _run_gate_recording_commands(
        monkeypatch,
        fail_fast=False,
        junit_path=junit,
        step_outcomes=[(0, {"test_unit_ok": False}), (1, {"test_browser_broke": True})],
    )

    assert code == 1
    assert [len(_worker_flags(c)) > 0 for c in calls if c[0] == "pytest"] == [True, False]
    names = [tc.get("name") for tc in ET.parse(junit).getroot().iter("testcase")]
    assert names == ["test_unit_ok", "test_browser_broke"]
    history = junit.with_name("failure_history.jsonl").read_text(encoding="utf-8")
    assert "test_browser_broke" in history and "test_unit_ok" not in history


def test_a_failing_first_step_stops_the_browser_suite_under_fail_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under fail-fast a failing first step ends the stage, as any failing stage does.

    Killed by: src/uclone_x/cli/quality_gate.py :: if step_results[-1].returncode != 0 and fail_fast:
    Becomes: if False:
    """
    code, calls = _run_gate_recording_commands(
        monkeypatch,
        junit_path=tmp_path / "junit.xml",
        step_outcomes=[(1, {"test_unit_broke": True}), (0, {"test_browser_ok": False})],
    )
    assert code == 1
    assert len([c for c in calls if c[0] == "pytest"]) == 1


def test_a_first_step_failure_is_the_result_even_when_the_browser_suite_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without fail-fast both steps run; the gate must not report the last step's success.

    Killed by: src/uclone_x/cli/quality_gate.py ::
        res_pytest = next((r for r in step_results if r.returncode != 0), step_results[-1])
    Becomes: res_pytest = step_results[-1]
    """
    code, calls = _run_gate_recording_commands(
        monkeypatch,
        fail_fast=False,
        junit_path=tmp_path / "junit.xml",
        step_outcomes=[(1, {"test_unit_broke": True}), (0, {"test_browser_ok": False})],
    )
    assert code == 1
    assert len([c for c in calls if c[0] == "pytest"]) == 2


def test_a_step_report_left_by_an_earlier_run_is_not_read_as_this_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A step that dies before writing its report must not inherit the last run's (#436).

    Killed by: src/uclone_x/cli/quality_gate.py :: if source.stat().st_mtime < written_after:
    Becomes: if False:
    """
    junit = tmp_path / "report" / "junit.xml"
    old = junit.with_name("junit.workers.xml")
    old.parent.mkdir(parents=True)
    old.write_text(_report({"test_from_an_earlier_red_run": True}), encoding="utf-8")
    long_ago = 1_000_000_000
    os.utime(old, (long_ago, long_ago))

    code, _ = _run_gate_recording_commands(
        monkeypatch, fail_fast=False, junit_path=junit, step_outcomes=[(4, None), (4, None)]
    )

    assert code == 4
    assert not junit.exists()
    assert not junit.with_name("failure_history.jsonl").exists()


def test_merge_reports_what_it_left_out(tmp_path: Path) -> None:
    present = tmp_path / "a.xml"
    present.write_text(_report({"test_a": False}), encoding="utf-8")
    missing = tmp_path / "b.xml"
    broken = tmp_path / "c.xml"
    broken.write_text("<testsuites><testsuite", encoding="utf-8")
    destination = tmp_path / "out" / "junit.xml"

    left_out = quality_gate.merge_junit_reports(
        [present, missing, broken], destination, written_after=0.0
    )

    assert left_out == [missing, broken]
    assert [tc.get("name") for tc in ET.parse(destination).getroot().iter("testcase")] == ["test_a"]


def test_the_serial_flag_reaches_the_pytest_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """`serial=True` reaches the pytest stage as one serial invocation.

    Killed by: src/uclone_x/cli/quality_gate.py ::
        steps = build_pytest_steps(test_scope, junit_path=junit_path, serial=serial)
    Becomes: steps = build_pytest_steps(test_scope, junit_path=junit_path)
    """
    code, calls = _run_gate_recording_commands(monkeypatch, serial=True)
    pytest_calls = [cmd for cmd in calls if cmd[0] == "pytest"]
    assert code == 0
    assert len(pytest_calls) == 1
    assert _worker_flags(pytest_calls[0]) == []


def test_a_missing_parallel_runner_fails_the_gate_instead_of_running_serially(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without pytest-xdist the stage is refused and says how to fix it, even when quiet.

    Running serially instead would pass, three times slower, and hide the stale environment.

    Killed by: src/uclone_x/cli/quality_gate.py :: and not parallel_runner_available()
    Becomes: and False
    """
    monkeypatch.setattr(quality_gate, "parallel_runner_available", lambda: False)

    code, calls = _run_gate_recording_commands(monkeypatch)

    assert code == 1
    assert [cmd for cmd in calls if cmd[0] == "pytest"] == []
    stderr = capsys.readouterr().err
    assert "pytest-xdist is not installed" in stderr
    assert "uv sync --all-extras" in stderr
    assert "--serial" in stderr


def test_a_missing_parallel_runner_without_fail_fast_skips_the_floors_it_has_no_data_for(
    monkeypatch: pytest.MonkeyPatch, stub_frontend_suite: list[Path | None]
) -> None:
    """The later stages still run and the gate still fails; the floors are not judged on stale data.

    Killed by: src/uclone_x/cli/quality_gate.py ::
        if not skip_tests and not runner_missing and test_scope not in _NO_COVERAGE_SCOPES:
    Becomes: if not skip_tests and test_scope not in _NO_COVERAGE_SCOPES:
    """
    monkeypatch.setattr(quality_gate, "parallel_runner_available", lambda: False)
    floors_checked: list[bool] = []

    def recording_floors() -> list[tuple[str, int, int]]:
        floors_checked.append(True)
        return []

    monkeypatch.setattr(quality_gate, "check_package_coverage_floors", recording_floors)
    code, calls = _run_gate_recording_commands(monkeypatch, fail_fast=False)

    assert code == 1
    assert [cmd for cmd in calls if cmd[0] == "pytest"] == []
    assert floors_checked == []
    assert len(stub_frontend_suite) == 1


def test_the_serial_escape_does_not_need_the_parallel_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--serial` works in an environment without pytest-xdist.

    Killed by: src/uclone_x/cli/quality_gate.py :: and pytest_runs_in_parallel(test_scope, serial=serial)
    Becomes: and pytest_runs_in_parallel(test_scope)
    """
    monkeypatch.setattr(quality_gate, "parallel_runner_available", lambda: False)

    code, calls = _run_gate_recording_commands(monkeypatch, serial=True)

    assert code == 0
    assert len([cmd for cmd in calls if cmd[0] == "pytest"]) == 1


def test_parallel_runner_availability_is_a_lookup_of_xdist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def not_found(name: str, package: str | None = None) -> None:
        asked.append(name)

    monkeypatch.setattr(importlib.util, "find_spec", not_found)
    assert _REAL_PARALLEL_RUNNER_AVAILABLE() is False
    assert asked == ["xdist"]


# --- Gate stage 6b: the committed UI bundle is what the source builds (#878) ----------
# The stage itself: tests/unit/test_frontend_bundle_freshness.py.


@pytest.mark.parametrize("fail_fast", [True, False])
def test_a_stale_committed_bundle_fails_the_whole_gate(
    fail_fast: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bundle that is not what the source builds is a red gate, with the stage's report (#878).

    The browser suite serves `src/uclone_x/ui_static`, so a green gate over a stale bundle
    is a green browser suite about a UI the source no longer describes.

    Killed by: src/uclone_x/cli/quality_gate.py :: _, bundle_code, bundle_lines = check_bundle_freshness()
    Becomes: bundle_code, bundle_lines = 0, []
    Killed by: src/uclone_x/cli/quality_gate.py :: if bundle_code != 0:
    Becomes: if False:
    """
    _gate_calls_with_passing_subprocesses(monkeypatch)

    def stale(root: Path | None = None) -> tuple[BundleStatus, int, list[str]]:
        return "stale", 1, ["stale: assets/index-OLD.js"]

    monkeypatch.setattr(quality_gate, "check_bundle_freshness", stale)

    exit_code = run_quality_gate(
        quiet=False,
        fail_fast=fail_fast,
        junit_path=tmp_path / "junit.xml",
        failure_log_path=tmp_path / "failure_history.jsonl",
    )

    assert exit_code != 0
    out = capsys.readouterr().out
    assert "stale: assets/index-OLD.js" in out
    assert "ALL SYSTEMS GO" not in out


def test_under_fe_the_bundle_is_judged_before_the_build_rewrites_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 7 (`-fe`) writes `src/uclone_x/ui_static`; stage 6b must have read it first.

    In the other order `-fe` would compare a bundle it had just rebuilt, and pass by
    construction on exactly the run that asks for the most frontend checking.
    """
    events: list[str] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        events.append(" ".join(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    def fresh(root: Path | None = None) -> tuple[BundleStatus, int, list[str]]:
        events.append("bundle-freshness")
        return "fresh", 0, []

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(quality_gate, "check_bundle_freshness", fresh)

    assert run_quality_gate(quiet=True, check_frontend=True) == 0
    assert events.index("bundle-freshness") < events.index("npm run build")


def test_the_gate_clears_a_non_blocking_stdout_before_its_first_write(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A stdout another process made non-blocking is restored before stage 1 prints (#993).

    Under `git push` over SSH the transport held the shared pipe non-blocking before the gate
    started. Stage 1 is the first code after the gate's opening lines, so it observes the mode
    every write before the first stage ran in. `capfd` makes fd 1 a capture file for the
    duration, so the flag toggled here is never the worker's own stream.

    Killed by: src/uclone_x/cli/quality_gate.py :: restored_at_start = restore_blocking_stdio()
    Becomes: restored_at_start: list[int] = []
    """
    observed: list[bool] = []

    def fresh() -> tuple[quality_gate.LockfileStatus, list[str]]:
        observed.append(os.get_blocking(1))
        return "fresh", []

    monkeypatch.setattr(quality_gate, "check_lockfile_freshness", fresh)
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0)),
    )
    was_blocking = os.get_blocking(1)
    os.set_blocking(1, False)
    try:
        exit_code = run_quality_gate(quiet=True, skip_tests=True)
    finally:
        os.set_blocking(1, was_blocking)

    assert exit_code == 0
    assert observed == [True]
    assert capfd.readouterr().out == ""


def _set_stdio_non_blocking() -> None:
    os.set_blocking(1, False)
    os.set_blocking(2, False)


class _StageResult:
    """A passing stage whose `returncode`, when the gate reads it, makes stdio non-blocking.

    The gate reads it after `run_stage` has returned and before it starts the next stage, so
    this stands in for a process sharing the gate's pipe that sets the flag between stages,
    as `ssh` did under `git push` (#993). Only a stage started through `run_stage` has it
    cleared again before it runs.
    """

    def __init__(self, argv: list[str]) -> None:
        self.args = argv

    @property
    def returncode(self) -> int:
        _set_stdio_non_blocking()
        return 0


def _run_gate_with_stdio_made_non_blocking_between_and_by_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, quiet: bool
) -> tuple[int, list[tuple[str, bool, bool]]]:
    """Run every stage of `gate` plus `-fe`, each faked; return (exit code, what each saw).

    Each fake stage records whether fd 1 and fd 2 were blocking when it started, then exits
    leaving both non-blocking, as a killed `node` does. The result it returns sets the flag
    again when the gate reads it (`_StageResult`), and stage 1b sets it after the gate-start
    restore, so the first stage is preceded by it too. The caller's `capfd` owns fds 1 and 2,
    and their modes are put back afterwards.

    Stage 6 is the real `run_frontend_suite` over a `frontend/` in `tmp_path`, not the autouse
    `stub_frontend_suite`, because it runs vitest on the gate's stdio. The other stubbed
    helpers (stages 1, 1b, 5b, 6b) capture their subprocesses' output, so they share nothing.
    """
    started: list[tuple[str, bool, bool]] = []

    def vitest_stage(
        root: Path | None = None, *, quiet: bool = False
    ) -> tuple[quality_gate.FrontendSuiteStatus, int, list[str]]:
        return real_run_frontend_suite(tmp_path, quiet=quiet)

    _frontend_tree(tmp_path, node_modules=True)
    _npm_on_path(monkeypatch, present=True)
    monkeypatch.setattr(quality_gate, "run_frontend_suite", vitest_stage)

    def provenance_then_non_blocking() -> tuple[ProvenanceStatus, list[str]]:
        _set_stdio_non_blocking()
        return "ok", []

    def stage(argv: list[str], **_: object) -> _StageResult:
        started.append((" ".join(argv[:2]), os.get_blocking(1), os.get_blocking(2)))
        _set_stdio_non_blocking()
        return _StageResult(argv)

    monkeypatch.setattr(quality_gate, "check_environment_provenance", provenance_then_non_blocking)
    monkeypatch.setattr(subprocess, "run", stage)
    was_blocking = (os.get_blocking(1), os.get_blocking(2))
    try:
        exit_code = run_quality_gate(
            quiet=quiet,
            check_frontend=True,
            junit_path=tmp_path / "junit.xml",
            failure_log_path=tmp_path / "failures.jsonl",
        )
    finally:
        os.set_blocking(1, was_blocking[0])
        os.set_blocking(2, was_blocking[1])
    return exit_code, started


def test_every_stage_that_shares_the_gate_stdio_starts_with_it_blocking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """Each stage starts blocking even when stdio was made non-blocking just before it (#999).

    The gate's restore runs inside `run_stage`, so this holds only for a stage started
    through it. A stage started with a bare `subprocess.run` inherits whatever the previous
    reader of the pipe left, which the other stdio tests cannot see: there, the previous
    stage's own after-restore has already cleared it.

    Killed by: src/uclone_x/cli/quality_gate.py :: step_results.append(run_stage(list(step.argv), quiet=quiet))
    Becomes: step_results.append(subprocess.run(list(step.argv)))
    """
    exit_code, started = _run_gate_with_stdio_made_non_blocking_between_and_by_stages(
        monkeypatch, tmp_path, quiet=True
    )

    assert exit_code == 0
    # ruff format, ruff check, pyright, the two pytest steps of `gate` (#967), vitest, npm build.
    assert [name for name, *_ in started] == [
        "ruff format",
        "ruff check",
        "pyright",
        *[name for name, *_ in started[3:5]],
        "npm test",
        "npm run",
    ], started
    assert [entry for entry in started if entry[1:] != (True, True)] == []
    capfd.readouterr()


def test_a_quiet_gate_does_not_report_a_restore_after_a_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """`quiet` covers the after-stage restore report, as it covers the gate-start one (#999).

    Every stage exits leaving stdio non-blocking, so `run_stage` restores it after each one;
    a quiet gate still restores it and says nothing.

    Killed by: src/uclone_x/cli/quality_gate.py :: if left_non_blocking and not quiet:
    Becomes: if left_non_blocking:
    Killed by: src/uclone_x/cli/quality_gate.py :: run_stage(["pyright"], quiet=quiet)
    Becomes: run_stage(["pyright"])
    Killed by: src/uclone_x/cli/quality_gate.py :: _, fe_code, fe_lines = run_frontend_suite(quiet=quiet)
    Becomes: _, fe_code, fe_lines = run_frontend_suite()
    """
    exit_code, started = _run_gate_with_stdio_made_non_blocking_between_and_by_stages(
        monkeypatch, tmp_path, quiet=True
    )

    assert exit_code == 0
    assert len(started) == 7, started
    assert capfd.readouterr() == ("", "")
