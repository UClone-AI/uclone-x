"""Unit tests for the merge gate's browser plan and its records (uclone_x.cli.browser_plan)."""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from uclone_x.cli import quality_gate
from uclone_x.cli.browser_plan import (
    BrowserPlan,
    junit_node_ids,
    newest_fresh_record,
    plan_browser_suite,
    record_browser_result,
)
from uclone_x.cli.quality_gate import build_pytest_steps, hold_gate_lock
from uclone_x.cli.scope_rules import RULES_PATH

_ROOT = Path(__file__).resolve().parents[2]


def _git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return done.stdout.strip()


def _write(repo: Path, rel: str, text: str = "x\n") -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Two commits on main (`origin/main` at the second), and a branch checked out off it."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "tests").mkdir()
    shutil.copy(_ROOT / RULES_PATH, root / RULES_PATH)
    _write(root, "src/pkg/core.py")
    _write(root, "tests/e2e/test_page_e2e.py")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "one")
    _write(root, "src/pkg/core.py", "y\n")
    _git(root, "commit", "-qam", "two")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(root, "checkout", "-q", "-b", "task")
    return root


def _commit(repo: Path, rel: str, text: str = "changed\n") -> None:
    _write(repo, rel, text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", f"change {rel}")


def _main(repo: Path, rev: str = "origin/main") -> str:
    return _git(repo, "rev-parse", rev)


def _record(repo: Path, kind: str, sha: str, body: str = "", age_hours: float = 0.0) -> Path:
    path = repo / ".git" / f"browser-{kind}" / sha
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    stamp = time.time() - age_hours * 3600
    os.utime(path, (stamp, stamp))
    return path


def test_a_browser_path_in_the_diff_runs_the_whole_suite(repo: Path) -> None:
    """Killed by: src/uclone_x/cli/browser_plan.py :: if trigger is not None:
    Becomes: if False:
    """
    _record(repo, "pass", _main(repo))
    _commit(repo, "src/uclone_x/ui/server.py")
    plan = plan_browser_suite(repo=repo)
    assert plan.mode == "full"
    assert "src/uclone_x/ui/server.py can change what the browser renders" in plan.reason


def test_no_fresh_pass_on_main_makes_the_run_the_nightly(repo: Path) -> None:
    """The lazy nightly: without a recent full pass, a merge gate runs the suite in full.

    Killed by: src/uclone_x/cli/browser_plan.py :: if record is None:
    Becomes: if False:
    """
    _commit(repo, "src/pkg/core.py")
    plan = plan_browser_suite(repo=repo)
    assert plan.mode == "full"
    assert "no full pass on main is fresh" in plan.reason


def test_a_fresh_pass_on_the_merge_base_skips_the_suite(repo: Path) -> None:
    _record(repo, "pass", _main(repo), age_hours=2)
    _commit(repo, "src/pkg/core.py")
    plan = plan_browser_suite(repo=repo)
    assert plan.mode == "none"
    assert f"a full pass on {_main(repo)[:12]}" in plan.reason


def test_a_fresh_pass_still_runs_the_browser_tests_the_diff_changes(repo: Path) -> None:
    _record(repo, "pass", _main(repo))
    _commit(repo, "tests/e2e/test_page_e2e.py")
    plan = plan_browser_suite(repo=repo)
    assert plan.mode == "files"
    assert plan.files == ("tests/e2e/test_page_e2e.py",)


def test_a_pass_older_than_the_age_bound_is_not_fresh(repo: Path) -> None:
    """Killed by: src/uclone_x/cli/browser_plan.py :: pass_fresh = pass_age is not None and pass_age <= limit
    Becomes: pass_fresh = pass_age is not None
    """
    _record(repo, "pass", _main(repo), age_hours=25)
    _commit(repo, "src/pkg/core.py")
    assert plan_browser_suite(repo=repo).mode == "full"


def test_a_pass_beyond_the_merge_bound_is_not_fresh(repo: Path) -> None:
    """The first commit is two merges back; the rules below allow one."""
    rules_file = repo / RULES_PATH
    rules_file.write_text(
        rules_file.read_text(encoding="utf-8").replace("max_merges = 20", "max_merges = 1"),
        encoding="utf-8",
    )
    _record(repo, "pass", _main(repo, "origin/main~1"))
    _commit(repo, "src/pkg/core.py")
    assert plan_browser_suite(repo=repo).mode == "full"


def test_a_pass_on_an_earlier_main_commit_within_the_bounds_is_fresh(repo: Path) -> None:
    _record(repo, "pass", _main(repo, "origin/main~1"))
    _commit(repo, "src/pkg/core.py")
    assert plan_browser_suite(repo=repo).mode == "none"


def test_a_red_main_deselects_what_it_fails_except_in_files_the_diff_changes(
    repo: Path,
) -> None:
    """A red nightly must not fail every merge; the PR's own tests still run.

    Killed by: src/uclone_x/cli/browser_plan.py :: if node.split("::", 1)[0] not in changed_tests
    Becomes: if True
    """
    red = "tests/e2e/test_page_e2e.py::test_a\ntests/e2e/test_gone_e2e.py::test_b\n"
    _record(repo, "red", _main(repo), body=red)
    _commit(repo, "src/uclone_x/ui/server.py")
    _commit(repo, "tests/e2e/test_page_e2e.py")
    plan = plan_browser_suite(repo=repo)
    assert plan.mode == "full"
    assert plan.deselect == ("tests/e2e/test_gone_e2e.py::test_b",)
    text = "\n".join(plan.describe())
    assert "Main is known red" in text
    assert "tests/e2e/test_gone_e2e.py::test_b" in text


def test_a_red_main_vouches_for_the_rest_of_the_suite(repo: Path) -> None:
    _record(repo, "red", _main(repo), body="tests/e2e/test_page_e2e.py::test_a\n")
    _commit(repo, "src/pkg/core.py")
    assert plan_browser_suite(repo=repo).mode == "none"


def _both_records(common: Path, pass_body: str, pass_age: float, red_age: float) -> None:
    now = time.time()
    for kind, body, age in (("pass", pass_body, pass_age), ("red", "t.py::x\n", red_age)):
        path = common / f"browser-{kind}" / "abc"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        os.utime(path, (now - age * 3600, now - age * 3600))


def test_a_pass_on_a_pr_tree_does_not_clear_a_red_main(tmp_path: Path) -> None:
    """A PR that fixes main's red test passes; main itself is still red until it merges.

    Killed by: src/uclone_x/cli/browser_plan.py :: and _reads(pass_path).strip() == _ON_THE_COMMIT
    Becomes: and True
    """
    _both_records(tmp_path, "tested-at 0123abcd\n", pass_age=1, red_age=5)
    record = newest_fresh_record(tmp_path, ["abc"], max_age_hours=24)
    assert record is not None and record.kind == "red"


def test_a_later_pass_on_the_commit_itself_clears_its_red_record(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/cli/browser_plan.py :: and pass_age < red_age
    Becomes: and False
    """
    _both_records(tmp_path, "tested-at the commit itself\n", pass_age=1, red_age=5)
    record = newest_fresh_record(tmp_path, ["abc"], max_age_hours=24)
    assert record is not None and record.kind == "pass"
    _both_records(tmp_path, "tested-at the commit itself\n", pass_age=5, red_age=1)
    record = newest_fresh_record(tmp_path, ["abc"], max_age_hours=24)
    assert record is not None and record.kind == "red"


def test_a_module_the_browser_suite_imports_runs_the_whole_suite(repo: Path) -> None:
    """No rule names backend code; the import graph says the served page runs it.

    Killed by: src/uclone_x/cli/browser_plan.py :: trigger = first_path_imported_from(repo, loaded.browser_tests, others)
    Becomes: trigger = None
    """
    _write(repo, "src/pkg/__init__.py", "")
    _write(repo, "src/pkg/served.py")
    _write(repo, "tests/e2e/test_page_e2e.py", "import pkg.served\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "serve")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _record(repo, "pass", _main(repo))
    _commit(repo, "src/pkg/served.py")
    plan = plan_browser_suite(repo=repo)
    assert plan.mode == "full"
    assert plan.reason == "the browser suite imports src/pkg/served.py"


def test_a_deleted_module_the_browser_suite_imported_runs_the_whole_suite(repo: Path) -> None:
    _write(repo, "src/pkg/__init__.py", "")
    _write(repo, "src/pkg/served.py")
    _write(repo, "tests/e2e/test_page_e2e.py", "from pkg import served\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "serve")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _record(repo, "pass", _main(repo))
    _git(repo, "rm", "-q", "src/pkg/served.py")
    _git(repo, "commit", "-qm", "drop")
    assert plan_browser_suite(repo=repo).mode == "full"


def test_the_gates_own_code_runs_the_whole_suite(repo: Path) -> None:
    """A PR must not narrow its own gate by editing the code that decides it."""
    _record(repo, "pass", _main(repo))
    _commit(repo, "src/uclone_x/cli/browser_plan.py")
    assert plan_browser_suite(repo=repo).mode == "full"


def test_full_browser_forces_the_suite_and_deselects_nothing(repo: Path) -> None:
    _record(repo, "red", _main(repo), body="tests/e2e/test_page_e2e.py::test_a\n")
    plan = plan_browser_suite(repo=repo, full=True)
    assert (plan.mode, plan.forced, plan.deselect) == ("full", True, ())


def test_unreadable_rules_run_the_whole_suite(repo: Path) -> None:
    (repo / RULES_PATH).write_text("nonsense = [", encoding="utf-8")
    _record(repo, "pass", _main(repo))
    plan = plan_browser_suite(repo=repo)
    assert plan.mode == "full"
    assert "scope rules could not be read" in plan.reason


def test_no_merge_base_runs_the_whole_suite(repo: Path) -> None:
    plan = plan_browser_suite(repo=repo, base="refs/heads/absent")
    assert plan.mode == "full"


# --- records -------------------------------------------------------------------------


def test_a_full_pass_on_a_pr_tree_records_its_merge_base(repo: Path) -> None:
    """Keyed by the merge base: a squash merge never makes the PR head an ancestor of main.

    Killed by: src/uclone_x/cli/browser_plan.py :: path = common / PASS_DIR / plan.merge_base
    Becomes: path = common / PASS_DIR / head
    """
    _commit(repo, "src/pkg/core.py")
    plan = plan_browser_suite(repo=repo)
    line = record_browser_result(plan, passed=True, report=repo / "none.xml", repo=repo)
    assert line is not None and "Browser pass record written" in line
    written = repo / ".git" / "browser-pass" / _main(repo)
    assert written.read_text(encoding="utf-8") == f"tested-at {_main(repo, 'HEAD')}\n"
    assert plan_browser_suite(repo=repo).mode == "none"


def test_a_pass_with_tests_deselected_records_nothing(repo: Path) -> None:
    plan = BrowserPlan("full", "r", deselect=("t::x",), merge_base=_main(repo))
    line = record_browser_result(plan, passed=True, report=repo / "none.xml", repo=repo)
    assert line is not None and "deselected" in line
    assert not (repo / ".git" / "browser-pass").exists()


def test_a_dirty_tree_records_nothing(repo: Path) -> None:
    _write(repo, "untracked.txt")
    plan = BrowserPlan("full", "r", merge_base=_main(repo))
    line = record_browser_result(plan, passed=True, report=repo / "none.xml", repo=repo)
    assert line is not None and "not a clean commit" in line


def test_a_failure_on_a_pr_tree_records_no_red(repo: Path) -> None:
    """A PR's failure is the PR's to fix; recorded as main's, it would deselect its own defect.

    Killed by: src/uclone_x/cli/browser_plan.py :: elif plan.forced and head == main:
    Becomes: elif True:
    """
    _commit(repo, "src/pkg/core.py")
    plan = plan_browser_suite(repo=repo, full=True)
    line = record_browser_result(plan, passed=False, report=repo / "r.xml", repo=repo)
    assert line is not None and "HEAD is not origin/main" in line
    assert not (repo / ".git" / "browser-red").exists()


@pytest.mark.parametrize(
    "cases",
    [
        "",  # no report at all: the step died before writing it
        '<testcase classname="gone.module" name="test_a"><failure/></testcase>',
    ],
)
def test_a_red_step_that_names_no_test_in_a_file_records_nothing(repo: Path, cases: str) -> None:
    """A crash or a coverage-only failure tested nothing a red record could vouch for.

    Killed by: src/uclone_x/cli/browser_plan.py :: if not ids or any("::" not in node for node in ids):
    Becomes: if False:
    """
    _git(repo, "checkout", "-q", "--detach", "origin/main")
    report = repo / ".git" / "junit.browser.xml"
    if cases:
        report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>", "utf-8")
    plan = plan_browser_suite(repo=repo, full=True)
    line = record_browser_result(plan, passed=False, report=report, repo=repo)
    assert line is not None and "vouches for nothing" in line
    assert not (repo / ".git" / "browser-red").exists()


def test_the_nightly_failing_on_main_records_the_failing_ids(repo: Path) -> None:
    _git(repo, "checkout", "-q", "--detach", "origin/main")
    report = repo / ".git" / "junit.browser.xml"
    report.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="tests.e2e.test_page_e2e" name="test_a"><failure/></testcase>'
        '<testcase classname="tests.e2e.test_page_e2e" name="test_ok"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    plan = plan_browser_suite(repo=repo, full=True)
    line = record_browser_result(plan, passed=False, report=report, repo=repo)
    assert line is not None and "Browser red record written" in line
    red = repo / ".git" / "browser-red" / _main(repo)
    assert red.read_text(encoding="utf-8") == "tests/e2e/test_page_e2e.py::test_a\n"


def test_junit_names_become_node_ids_with_any_class(tmp_path: Path) -> None:
    _write(tmp_path, "tests/e2e/test_x.py")
    report = tmp_path / "r.xml"
    report.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="tests.e2e.test_x.TestPage" name="test_a[p1]"><error/></testcase>'
        '<testcase classname="tests.e2e.test_x" name="test_b"><failure/></testcase>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    assert junit_node_ids(report, tmp_path) == [
        "tests/e2e/test_x.py::TestPage::test_a[p1]",
        "tests/e2e/test_x.py::test_b",
    ]


# --- the gate's pytest steps under a plan -----------------------------------------------


def test_no_browser_step_means_the_workers_step_judges_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the browser step gone, `--cov-fail-under=0` would leave coverage unjudged.

    Killed by: src/uclone_x/cli/quality_gate.py :: *(("--cov-fail-under=0",) if browser_runs else ()),
    Becomes: "--cov-fail-under=0",
    """
    monkeypatch.setattr(quality_gate, "pytest_worker_count", lambda: 4)
    steps = build_pytest_steps("gate", browser=BrowserPlan("none", "r"))
    assert len(steps) == 1
    assert "--cov-fail-under=0" not in steps[0].argv
    assert "-n" in steps[0].argv


def test_a_files_plan_runs_only_those_files_and_a_full_plan_deselects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quality_gate, "pytest_worker_count", lambda: 4)
    files = build_pytest_steps("gate", browser=BrowserPlan("files", "r", files=("tests/e2e/a.py",)))
    assert files[1].argv[-1] == "tests/e2e/a.py"
    assert "--cov-fail-under=0" in files[0].argv
    full = build_pytest_steps("gate", browser=BrowserPlan("full", "r", deselect=("t.py::x",)))
    argv = list(full[1].argv)
    assert argv[argv.index("--deselect") + 1] == "t.py::x"
    assert build_pytest_steps("gate", browser=None)[1].argv == build_pytest_steps("gate")[1].argv


# --- one full gate at a time ----------------------------------------------------------------


def test_a_second_gate_waits_for_the_first_and_says_so(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/cli/quality_gate.py :: fcntl.flock(handle, fcntl.LOCK_EX)
    Becomes: pass
    """
    order: list[str] = []
    released = threading.Event()

    def second() -> None:
        with hold_gate_lock(tmp_path, announce=order.append):
            order.append("second ran")

    with hold_gate_lock(tmp_path, announce=order.append):
        thread = threading.Thread(target=second)
        thread.start()
        deadline = time.monotonic() + 10
        while not order and time.monotonic() < deadline:
            time.sleep(0.01)
        order.append("first done")
        released.set()
    thread.join(timeout=10)
    assert order[0].startswith("Waiting for the gate already running (pid ")
    assert order[1] == "first done"
    assert order[-1] == "second ran"


def test_outside_a_checkout_the_gate_runs_unlocked() -> None:
    with hold_gate_lock(None, announce=lambda _: None):
        pass


def test_package_data_under_src_runs_the_whole_suite(repo: Path) -> None:
    """A persona or bundled story is loaded at run time; no import edge leads to it.

    Killed by: src/uclone_x/cli/browser_plan.py :: if p.startswith("src/") and not p.endswith(".py")
    Becomes: if False
    """
    _record(repo, "pass", _main(repo))
    _commit(repo, "src/uclone_x/personas/artist.yaml")
    plan = plan_browser_suite(repo=repo)
    assert plan.mode == "full"
    assert "package data" in plan.reason


def test_a_pr_tree_pass_keeps_the_nightlys_pass_on_the_commit(repo: Path) -> None:
    """Overwritten, an older red record would return and deselect tests main now passes.

    Killed by: src/uclone_x/cli/browser_plan.py :: if body.strip() != _ON_THE_COMMIT and _reads(path).strip() == _ON_THE_COMMIT:
    Becomes: if False:
    """
    kept = _record(repo, "pass", _main(repo), body="tested-at the commit itself\n", age_hours=3)
    before = kept.stat().st_mtime
    _commit(repo, "src/pkg/core.py")
    plan = BrowserPlan("full", "r", merge_base=_main(repo))
    line = record_browser_result(plan, passed=True, report=repo / "none.xml", repo=repo)
    assert line is not None and "kept" in line
    assert kept.read_text(encoding="utf-8") == "tested-at the commit itself\n"
    assert kept.stat().st_mtime == before
