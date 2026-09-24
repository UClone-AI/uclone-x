"""Discovery of the optional evaluation backend.

`uclone_x` is the runtime; the suites that grade it are shipped separately and
may be absent entirely. Every entry point that wants a runner asks for one here
and handles its absence, so a distribution without suites still imports, type
checks, starts the dashboard and answers `./ucx eval --help`.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Final, NamedTuple, cast

from uclone_x.evaluation.protocols import EvalRunnerFactory, EvalRunnerProtocol

#: Module a backend must expose. Anything providing these three names is a valid
#: backend; the in-repo `evals/` package is the reference implementation.
EVAL_BACKEND_MODULE: Final = "evals.runner"

#: Names read off the backend module.
_RUNNER_ATTR: Final = "EvalRunner"
_REPORTS_DIR_ATTR: Final = "DEFAULT_REPORTS_DIR"
_LIVE_REASON_ATTR: Final = "DEFAULT_LIVE_REASON"

#: Used when no backend is installed to describe a live suite, so help text and
#: tables keep their shape rather than rendering an empty column.
FALLBACK_LIVE_REASON: Final = (
    "declares requires_live (spends real tokens or needs an external service)"
)

#: Where reports are looked for when no backend is installed to name a directory.
_FALLBACK_REPORTS_SUBPATH: Final = ("evals", "reports")


class EvalBackendUnavailableError(RuntimeError):
    """Raised when an evaluation backend is required but none is installed."""


class EvalBackend(NamedTuple):
    """The three names the runtime consumes from an evaluation backend."""

    runner_factory: EvalRunnerFactory
    default_reports_dir: Path
    default_live_reason: str


def _import_backend() -> Any:
    """Import the backend module, or raise `EvalBackendUnavailableError`."""
    try:
        return import_module(EVAL_BACKEND_MODULE)
    except ImportError as exc:
        raise EvalBackendUnavailableError(
            f"No evaluation backend is installed: `{EVAL_BACKEND_MODULE}` is not importable "
            f"({exc}). The evaluation suites are distributed separately from the UClone-X "
            "runtime; install a package providing that module, or run from a checkout that "
            "contains one, to use `./ucx eval`."
        ) from exc


def load_eval_backend() -> EvalBackend:
    """Return the installed backend, raising if it is absent or incomplete.

    An incomplete backend is reported as precisely as a missing one: a module
    that imports but lacks a name is a wiring bug in that backend, and saying
    which name is missing is the difference between a fix and a bisect.
    """
    module = _import_backend()

    missing = [
        name
        for name in (_RUNNER_ATTR, _REPORTS_DIR_ATTR, _LIVE_REASON_ATTR)
        if not hasattr(module, name)
    ]
    if missing:
        raise EvalBackendUnavailableError(
            f"`{EVAL_BACKEND_MODULE}` is installed but does not export "
            f"{', '.join(missing)}; it is not a usable evaluation backend."
        )

    return EvalBackend(
        runner_factory=cast(EvalRunnerFactory, getattr(module, _RUNNER_ATTR)),
        default_reports_dir=Path(cast(Path, getattr(module, _REPORTS_DIR_ATTR))),
        default_live_reason=str(getattr(module, _LIVE_REASON_ATTR)),
    )


def eval_backend_available() -> bool:
    """Whether a usable evaluation backend is installed."""
    try:
        load_eval_backend()
    except EvalBackendUnavailableError:
        return False
    return True


def create_eval_runner(reports_dir: Path | None = None) -> EvalRunnerProtocol:
    """Build a runner from the installed backend.

    Raises `EvalBackendUnavailableError` when none is installed.
    """
    return load_eval_backend().runner_factory(reports_dir=reports_dir)


def default_reports_dir() -> Path:
    """Where reports live: the backend's directory, or a repo-relative default.

    The fallback exists so callers that only *display* a path (the dashboard's
    `eval_reports_dir`) keep working with no backend installed. Nothing writes
    there without a runner, which cannot be built in that state.
    """
    try:
        return load_eval_backend().default_reports_dir
    except EvalBackendUnavailableError:
        return Path.cwd().joinpath(*_FALLBACK_REPORTS_SUBPATH)


def default_live_reason() -> str:
    """The backend's description of a live suite, or the built-in wording."""
    try:
        return load_eval_backend().default_live_reason
    except EvalBackendUnavailableError:
        return FALLBACK_LIVE_REASON
