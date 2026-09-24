"""Tier boundary regression tests (#377).

The hybrid quality-assurance design note states that
a tier which cannot be named in a marker expression is not a boundary. These tests are
that statement, executed: they pin the marker expressions and the coverage flags so a
future edit to `quality_gate.py` cannot quietly let a token-spending tier into
`./ucx test check`.

The regression being pinned is concrete. Before #377 the unit scope was selected with
`-m "not e2e"` alone, so a `tests/live/` directory added per the document's own roadmap
would have run inside the default gate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from uclone_x.cli.quality_gate import (
    SCOPE_LABELS,
    TEST_SCOPES,
    build_pytest_command,
)


def _marker_expression(cmd: list[str]) -> str:
    """Return the `-m` expression from a built pytest argv."""
    assert "-m" in cmd, f"scope produced no marker expression: {cmd}"
    return cmd[cmd.index("-m") + 1]


def test_unit_scope_excludes_every_other_tier() -> None:
    """The default gate must name recorded and live negatively, not rely on layout."""
    expression = _marker_expression(build_pytest_command("unit"))
    for excluded in ("e2e", "recorded", "live"):
        assert f"not {excluded}" in expression, (
            f"unit scope does not exclude the {excluded!r} tier: {expression!r}"
        )


def test_unit_scope_excludes_the_fitness_functions() -> None:
    """Unit selects tests of product code, not checks over the repository's declarations.

    Before the fitness tier existed these were the same selection, so the suite could not
    say how much of its own runtime went to checking documents, and neither set could be run
    alone (R3, R13).
    """
    expression = _marker_expression(build_pytest_command("unit"))
    assert "not fitness" in expression, (
        f"unit scope does not exclude the fitness tier: {expression!r}"
    )


def test_gate_scope_runs_the_fitness_functions() -> None:
    """The default gate keeps running them: a tier nobody runs verifies nothing (P8).

    `gate` is deliberately the union rather than a level — the offline, free selection that
    `./ucx test check` executes.
    """
    expression = _marker_expression(build_pytest_command("gate"))
    assert "fitness" not in expression, (
        f"gate scope must not exclude the fitness tier: {expression!r}"
    )
    # E2E is deliberately NOT excluded here. It used to be, which made the default gate's
    # green silent about the rendered UI while it was the only suite exercising it.
    assert "not e2e" not in expression, (
        f"gate scope must include the E2E tier; use --fast to skip it: {expression!r}"
    )
    # The network and cost tiers stay out: the gate must be offline and free.
    for excluded in ("recorded", "live"):
        assert f"not {excluded}" in expression, (
            f"gate scope does not exclude the {excluded!r} tier: {expression!r}"
        )


def test_fast_scope_is_the_gate_without_the_browser_suite() -> None:
    """`--fast` is the opt-out, so it must differ from `gate` in exactly the E2E tier."""
    fast = _marker_expression(build_pytest_command("fast"))
    gate = _marker_expression(build_pytest_command("gate"))

    assert "not e2e" in fast, f"fast scope must exclude the E2E tier: {fast!r}"
    # Everything else the gate runs, fast runs too — a fast gate that also silently
    # dropped the fitness functions would be a second, weaker policy nobody declared.
    assert "fitness" not in fast
    for excluded in ("recorded", "live"):
        assert f"not {excluded}" in fast
    assert fast == f"not e2e and {gate}", (
        f"fast must be the gate minus E2E and nothing else: {fast!r} vs {gate!r}"
    )


def test_fitness_scope_disables_coverage() -> None:
    """Fitness functions import no product code, so measuring them against it is noise."""
    assert "--no-cov" in build_pytest_command("fitness")


def test_unit_scope_keeps_coverage_enforcement() -> None:
    """Tier 1 is the scope the >= 70% branch coverage floor is measured against."""
    assert "--no-cov" not in build_pytest_command("unit")


@pytest.mark.parametrize("scope", ["recorded", "live", "e2e"])
def test_partial_scopes_disable_coverage(scope: str) -> None:
    """`addopts` carries --cov-fail-under=70, so a partial selection must pass --no-cov.

    Without it these scopes fail on coverage even when every selected test passes, which
    reports a coverage problem for what is actually a scoping problem.
    """
    assert "--no-cov" in build_pytest_command(scope)


def test_recorded_scope_selects_only_recorded_tests() -> None:
    assert _marker_expression(build_pytest_command("recorded")) == "recorded"


def test_live_scope_selects_live_tests_and_carries_the_opt_in() -> None:
    """Tier 3 tests are skipped without --live, so the live scope has to pass it."""
    cmd = build_pytest_command("live")
    assert _marker_expression(cmd) == "live"
    assert "--live" in cmd


@pytest.mark.parametrize("scope", ["unit", "recorded", "e2e", "all"])
def test_only_the_live_scope_opts_into_live_execution(scope: str) -> None:
    """No other scope may spend tokens."""
    assert "--live" not in build_pytest_command(scope)


def test_pre_release_scope_selects_the_release_tier_and_carries_the_opt_in() -> None:
    """`pre_release` items are skipped without the opt-in, so the scope has to pass it.

    Without the flag this scope would collect its tests, skip every one of them, and
    report green — the silent pass P6 forbids, and the exact failure mode that makes a
    release-qualification lane worse than no lane at all.
    """
    cmd = build_pytest_command("pre-release")
    assert _marker_expression(cmd) == "pre_release"
    assert "--pre-release" in cmd
    assert "--no-cov" in cmd


@pytest.mark.parametrize("scope", [scope for scope in TEST_SCOPES if scope != "pre-release"])
def test_no_everyday_scope_opts_into_release_qualification(scope: str) -> None:
    """The install matrix resolves dependencies over the network; no push may pay for it.

    Derived from `TEST_SCOPES` rather than listed, so the statement is total: a hardcoded
    list of six scopes left `recorded` and `live` unchecked, and a scope added later would
    have been unchecked too (#882).

    Killed by: src/uclone_x/cli/quality_gate.py :: if test_scope == "pre-release":
    Becomes: if test_scope in ("pre-release", "live"):
    """
    assert "--pre-release" not in build_pytest_command(scope)


def test_all_scope_still_excludes_the_paid_and_recorded_tiers() -> None:
    """`--all` is documented as "Unit + E2E"; it must not silently mean "everything"."""
    expression = _marker_expression(build_pytest_command("all"))
    assert "not recorded" in expression
    assert "not live" in expression
    assert "not e2e" not in expression


def test_unknown_scope_raises_instead_of_widening_selection() -> None:
    """A mistyped scope must fail loudly; degrading to "run everything" is a P6 fallback."""
    with pytest.raises(ValueError, match="unknown test scope 'untit'"):
        build_pytest_command("untit")


def test_junit_path_is_threaded_through(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    assert f"--junitxml={junit}" in build_pytest_command("unit", junit_path=junit)


def test_every_declared_scope_is_buildable_and_labelled() -> None:
    """`TEST_SCOPES`, the marker table and the labels must not drift apart."""
    for scope in TEST_SCOPES:
        assert build_pytest_command(scope)[0] == "pytest"
        assert scope in SCOPE_LABELS, f"scope {scope!r} has no human label"


def test_declared_markers_cover_every_marker_the_scopes_reference() -> None:
    """A marker expression naming an unregistered marker is a silent no-op under
    `--strict-markers`, and a lie without it. Read the registrations from
    `pyproject.toml` rather than trusting a duplicated list."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    declared = {
        line.strip().strip('"').split(":", 1)[0]
        for line in pyproject.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith('"') and ":" in line
    }
    for marker in ("e2e", "recorded", "live"):
        assert marker in declared, f"marker {marker!r} is not registered in pyproject.toml"


class _StubItem:
    """Minimal `pytest.Item` stand-in for the collection hook.

    `keywords` is modelled the way pytest builds it — markers *plus* the node name and
    every parametrize id — because that conflation is the bug this stub exists to catch.
    """

    def __init__(self, nodeid: str, markers: tuple[str, ...]) -> None:
        self.nodeid = nodeid
        self._markers = frozenset(markers)
        name = nodeid.rsplit("::", 1)[-1]
        keywords: dict[str, object] = {name: True}
        keywords.update(dict.fromkeys(markers, True))
        if "[" in name:
            keywords.update(dict.fromkeys(name.split("[", 1)[1].rstrip("]").split("-"), True))
        self.keywords = keywords
        self.added: list[pytest.MarkDecorator] = []

    def get_closest_marker(self, name: str) -> object | None:
        return name if name in self._markers else None

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.added.append(marker)


def test_live_items_are_skipped_without_the_opt_in() -> None:
    from tests.support.live_optin import apply_live_skip

    live = _StubItem("tests/live/test_x.py::test_a", ("live",))
    unit = _StubItem("tests/unit/test_y.py::test_b", ())
    marker = pytest.mark.skip(reason="stub")

    skipped = apply_live_skip(live_enabled=False, items=[live, unit], skip_marker=marker)

    assert skipped == ("tests/live/test_x.py::test_a",)
    assert live.added == [marker]
    assert unit.added == []


def test_live_items_run_when_the_opt_in_is_given() -> None:
    from tests.support.live_optin import apply_live_skip

    live = _StubItem("tests/live/test_x.py::test_a", ("live",))

    skipped = apply_live_skip(
        live_enabled=True, items=[live], skip_marker=pytest.mark.skip(reason="stub")
    )

    assert skipped == ()
    assert live.added == []


def test_a_test_merely_named_live_is_not_skipped() -> None:
    """`item.keywords` conflates markers with node names and parametrize ids.

    Measured regression: with `"live" in item.keywords` the hook skipped this module's
    own `test_partial_scopes_disable_coverage[live]`, which carries no marker at all. A
    guard that skips unmarked tests keeps the run green while covering less, so it must
    consult the marker, not the keyword bag.
    """
    from tests.support.live_optin import apply_live_skip

    named = _StubItem("tests/unit/test_scopes.py::test_scope[live]", ())
    assert "live" in named.keywords, "stub does not reproduce pytest's keyword bag"

    skipped = apply_live_skip(
        live_enabled=False, items=[named], skip_marker=pytest.mark.skip(reason="stub")
    )

    assert skipped == ()
    assert named.added == []


def test_recorded_marker_alone_does_not_trigger_the_live_skip() -> None:
    """Tier 2 is free and offline; it must never be gated behind --live."""
    from tests.support.live_optin import apply_live_skip

    recorded = _StubItem("tests/recorded/test_x.py::test_a", ("recorded",))

    assert (
        apply_live_skip(
            live_enabled=False,
            items=[recorded],
            skip_marker=pytest.mark.skip(reason="stub"),
        )
        == ()
    )
    assert recorded.added == []


def test_pre_release_items_are_skipped_without_the_opt_in() -> None:
    from tests.support.live_optin import apply_pre_release_skip

    pre_rel = _StubItem(
        "tests/scenarios/test_user_creative_workflow_scenario.py::test_workflow", ("pre_release",)
    )
    unit = _StubItem("tests/unit/test_y.py::test_b", ())
    marker = pytest.mark.skip(reason="stub")

    skipped = apply_pre_release_skip(
        pre_release_enabled=False, items=[pre_rel, unit], skip_marker=marker
    )

    assert skipped == ("tests/scenarios/test_user_creative_workflow_scenario.py::test_workflow",)
    assert pre_rel.added == [marker]
    assert unit.added == []


def test_pre_release_items_run_when_the_opt_in_is_given() -> None:
    from tests.support.live_optin import apply_pre_release_skip

    pre_rel = _StubItem(
        "tests/scenarios/test_user_creative_workflow_scenario.py::test_workflow", ("pre_release",)
    )

    skipped = apply_pre_release_skip(
        pre_release_enabled=True, items=[pre_rel], skip_marker=pytest.mark.skip(reason="stub")
    )

    assert skipped == ()
    assert pre_rel.added == []


def test_integration_and_scenario_tests_are_selected_by_gate_and_unit_scopes() -> None:
    """L2 integration and scenario tests ride on the default gate without custom markers.

    Issue #479: tests under `tests/integration/` and `tests/scenarios/` are unmarked
    and therefore selected by both `gate` ("not e2e and not recorded and not live") and
    `unit` ("not e2e and not recorded and not live and not fitness"). They are NOT
    excluded and do NOT disable coverage.
    """
    gate_expr = _marker_expression(build_pytest_command("gate"))
    unit_expr = _marker_expression(build_pytest_command("unit"))

    for excluded in ("integration", "scenarios"):
        assert f"not {excluded}" not in gate_expr, f"gate scope must not exclude {excluded}"
        assert f"not {excluded}" not in unit_expr, f"unit scope must not exclude {excluded}"

    # Verify that integration and scenarios are NOT in _NO_COVERAGE_SCOPES
    assert "--no-cov" not in build_pytest_command("unit")
    assert "--no-cov" not in build_pytest_command("gate")


def test_invocation_guard_accepts_the_current_run() -> None:
    """The guard must not fire on a correct invocation — this suite is one.

    A guard that raised `UsageError` on `./ucx test check` would make the gate
    unrunnable, and the first draft of this one did exactly that: it compared
    `sys.executable` against `Path.resolve()`, which flags a venv's `bin/python` symlink
    to the uv-managed interpreter. Every correct invocation would have been rejected.
    """
    from tests.conftest import pytest_configure

    # The real config object this run was built with is not reachable here, so assert the
    # two properties the guard checks, against this process.
    assert os.path.isabs(sys.executable)
    assert os.path.normpath(sys.executable) == sys.executable, (
        f"sys.executable carries an un-normalised segment: {sys.executable!r}"
    )
    assert callable(pytest_configure)


def test_invocation_guard_is_opt_outable() -> None:
    """A deliberate bypass must exist: testing an installed wheel is a legitimate run.

    Pinned by name rather than by importing the private constant, so the documented
    escape hatch cannot be renamed without this failing — the variable is what someone
    reads out of an error message and types into their shell.
    """
    from tests import conftest

    opt_out = conftest._INVOCATION_GUARD_OPT_OUT  # pyright: ignore[reportPrivateUsage]
    assert opt_out == "UCLONE_SKIP_INVOCATION_GUARD"
