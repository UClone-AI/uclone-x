"""Unit tests for ucx loop CLI command (Issue FR-Loop, P4, P6)."""

from typer.testing import CliRunner

from uclone_x.cli.main import app

runner = CliRunner()


def test_ucx_loop_help() -> None:
    result = runner.invoke(app, ["loop", "--help"])
    assert result.exit_code == 0
    assert "Execute recurring agent automation loops" in result.output
    assert "run" in result.output


def test_ucx_loop_run_help() -> None:
    result = runner.invoke(app, ["loop", "run", "--help"])
    assert result.exit_code == 0
    assert "--interval" in result.output
    assert "--max-runs" in result.output
    assert "--timeout" in result.output
    assert "--clean" in result.output
    assert "--until" in result.output


def test_ucx_loop_run_invalid_interval() -> None:
    result = runner.invoke(app, ["loop", "run", "--interval", "invalid_val", "some prompt"])
    assert result.exit_code == 2
    assert (
        "Invalid --interval argument" in result.output or "Invalid interval format" in result.output
    )


def test_ucx_loop_run_unrecognized_interval_in_text() -> None:
    result = runner.invoke(app, ["loop", "run", "just do something without any interval"])
    assert result.exit_code == 2
    assert "Could not determine interval" in result.output
