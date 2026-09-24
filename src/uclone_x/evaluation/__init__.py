"""Optional evaluation backend seam: protocols plus discovery, no suites."""

from __future__ import annotations

from uclone_x.evaluation.loader import (
    EVAL_BACKEND_MODULE,
    FALLBACK_LIVE_REASON,
    EvalBackend,
    EvalBackendUnavailableError,
    create_eval_runner,
    default_live_reason,
    default_reports_dir,
    eval_backend_available,
    load_eval_backend,
)
from uclone_x.evaluation.protocols import (
    EvalProbeResultProtocol,
    EvalReportProtocol,
    EvalRunnerFactory,
    EvalRunnerProtocol,
    EvalSummaryProtocol,
)

__all__ = [
    "EVAL_BACKEND_MODULE",
    "FALLBACK_LIVE_REASON",
    "EvalBackend",
    "EvalBackendUnavailableError",
    "EvalProbeResultProtocol",
    "EvalReportProtocol",
    "EvalRunnerFactory",
    "EvalRunnerProtocol",
    "EvalSummaryProtocol",
    "create_eval_runner",
    "default_live_reason",
    "default_reports_dir",
    "eval_backend_available",
    "load_eval_backend",
]
