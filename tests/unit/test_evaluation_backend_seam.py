"""The runtime must remain usable when no evaluation backend is installed.

The suites under `evals/` are private to this repository and are not part of the
distributed runtime. These tests pin the seam that makes that separation real:
with the backend hidden, the CLI still resolves, the dashboard still answers,
and every failure names the missing backend rather than raising `ImportError`
out of an unrelated call.
"""

from __future__ import annotations

import builtins
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from uclone_x.cli.commands.eval import eval_app
from uclone_x.evaluation import (
    EVAL_BACKEND_MODULE,
    FALLBACK_LIVE_REASON,
    EvalBackendUnavailableError,
    create_eval_runner,
    default_live_reason,
    default_reports_dir,
    eval_backend_available,
    load_eval_backend,
)
from uclone_x.evaluation.protocols import EvalRunnerProtocol
from uclone_x.ui.app import create_ui_app

runner = CliRunner()


@pytest.fixture
def no_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make `evals.runner` unimportable, as it is in a runtime-only install."""
    for name in list(sys.modules):
        if name == "evals" or name.startswith("evals."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name == "evals" or name.startswith("evals."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    yield


def test_backend_reports_itself_unavailable(no_backend: None) -> None:
    assert eval_backend_available() is False


def test_load_names_the_missing_module(no_backend: None) -> None:
    with pytest.raises(EvalBackendUnavailableError) as excinfo:
        load_eval_backend()
    assert EVAL_BACKEND_MODULE in str(excinfo.value)


def test_create_runner_raises_rather_than_import_error(no_backend: None) -> None:
    with pytest.raises(EvalBackendUnavailableError):
        create_eval_runner()


def test_defaults_fall_back_without_a_backend(no_backend: None) -> None:
    assert default_live_reason() == FALLBACK_LIVE_REASON
    assert default_reports_dir() == Path.cwd() / "evals" / "reports"


@pytest.mark.parametrize("command", [["list"], ["run"], ["status"], ["view"]])
def test_cli_exits_with_a_named_reason(no_backend: None, command: list[str]) -> None:
    result = runner.invoke(eval_app, command)
    assert result.exit_code == 1
    assert "No evaluation backend is installed" in result.output


def test_incomplete_backend_names_the_missing_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """A module that imports but exports nothing is not a usable backend."""
    stub = ModuleType(EVAL_BACKEND_MODULE)
    monkeypatch.setitem(sys.modules, EVAL_BACKEND_MODULE, stub)

    with pytest.raises(EvalBackendUnavailableError) as excinfo:
        load_eval_backend()

    message = str(excinfo.value)
    assert "EvalRunner" in message
    assert "DEFAULT_REPORTS_DIR" in message
    assert "DEFAULT_LIVE_REASON" in message


@pytest.mark.skipif(
    not eval_backend_available(),
    reason="no evaluation backend installed; this asserts the reference backend's shape",
)
def test_in_repo_backend_satisfies_the_contract() -> None:
    """The reference backend under `evals/` is a structural match for the seam.

    Skipped where no backend is installed, which is the normal state of a
    runtime-only distribution. Every other test in this file asserts behaviour
    *without* a backend and runs everywhere; this one needs a backend present
    to have a subject at all, and asserting the shape of something absent is
    not a stricter test, only a red one.

    Assigning the concrete runner to the protocol is what makes this a type-level
    check as well as a runtime one, so a backend that drifts from the contract
    fails the gate here rather than at a call site in the dashboard.
    """
    backend = load_eval_backend()
    built: EvalRunnerProtocol = backend.runner_factory(reports_dir=None)

    assert isinstance(built.list_suites(), list)
    assert isinstance(built.excluded_suites, list)
    assert backend.default_reports_dir.is_dir()
    assert backend.default_live_reason


@pytest.mark.asyncio
async def test_dashboard_says_no_backend_rather_than_failing(
    no_backend: None, tmp_path: Path
) -> None:
    """With no backend, the dashboard answers `no_backend` with no error, not a failure.

    A runtime shipped without suites is a normal state. It is also a different fact from
    "suites exist and none has run", so it gets its own status rather than `empty`.

    Killed by: src/uclone_x/ui/app.py :: return empty_response("no_backend")
    Becomes: return empty_response("empty")
    """
    app = create_ui_app(
        static_dir=tmp_path,
        fallback_to_mock=True,
        storage_dir=tmp_path,
        eval_reports_dir=tmp_path / "reports",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        latest = (await client.get("/api/evaluations/latest")).json()
        history = (await client.get("/api/evaluations/history")).json()

    assert (latest["status"], latest["error"], latest["suites"]) == ("no_backend", None, [])
    assert (history["status"], history["error"], history["reports"]) == ("no_backend", None, [])
