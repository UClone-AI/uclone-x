"""Structural contracts the runtime needs from an evaluation backend.

The evaluation suites themselves are *not* part of the runtime: they measure it.
The runtime therefore names only the shape it consumes — a runner that lists,
executes and reads back reports — and never imports the backend that implements
it. That keeps `uclone_x` installable, type-checkable and testable in a
distribution that ships no suites at all, while `./ucx eval` still works
wherever a backend is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EvalProbeResultProtocol(Protocol):
    """A single graded probe within a suite report."""

    name: str
    passed: bool
    latency_s: float
    message: str


@runtime_checkable
class EvalSummaryProtocol(Protocol):
    """Aggregate scorecard numbers for one suite execution."""

    total_probes: int
    passed_probes: int
    failed_probes: int
    pass_rate: float
    duration_s: float
    p50_latency_s: float | None
    worst_latency_s: float | None


@runtime_checkable
class EvalReportProtocol(Protocol):
    """One suite execution, as rendered by the CLI and served by the UI."""

    timestamp: str
    suite: str
    model: str | None
    provider: str | None
    summary: EvalSummaryProtocol
    probes: list[Any]
    metadata: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping."""
        ...


class EvalRunnerProtocol(Protocol):
    """The orchestrator surface `./ucx eval` and the dashboard drive.

    `excluded_suites` is read after `run()` to report what a missing `--live`
    flag skipped, so it is part of the contract rather than an implementation
    detail of any one backend.
    """

    excluded_suites: list[str]

    def list_suites(self) -> list[dict[str, Any]]:
        """Return metadata for every discoverable suite."""
        ...

    def run(
        self,
        suite_name: str = "all",
        model: str | None = None,
        provider: str | None = None,
        reps: int = 1,
        json_path: Path | None = None,
        options: dict[str, Any] | None = None,
        live: bool = False,
    ) -> list[EvalReportProtocol]:
        """Execute the requested suite (or all of them) and return the reports."""
        ...

    def get_reports(self) -> list[EvalReportProtocol]:
        """Return every persisted report, unordered."""
        ...

    def get_latest_scorecard(self) -> dict[str, EvalReportProtocol]:
        """Return the most recent report per suite, keyed by suite name."""
        ...


class EvalRunnerFactory(Protocol):
    """Constructor shape of a runner, as exported by a backend module."""

    def __call__(self, reports_dir: Path | None = None) -> EvalRunnerProtocol:
        """Build a runner that reads and writes reports under `reports_dir`."""
        ...
