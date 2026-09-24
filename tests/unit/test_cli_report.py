"""`ucx report` -- what it shows before consent, and what it never does on its own.

The command is the only place a report can leave the machine, so the tests that
matter are the ones about *not* leaving: no browser opened without the flag, no
issue filed without the user's own authenticated `gh`, and an explanation --
not an error -- for someone who has never been asked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from uclone_x.cli.commands import report as report_module
from uclone_x.cli.commands.report import report_app
from uclone_x.core.failure_journal import (
    CONSENT_DENIED,
    CONSENT_GRANTED,
    consent_state,
    read_entries,
    record_failure,
    set_consent,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UCLONE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))


def _record(message: str = "boom") -> None:
    try:
        raise ValueError(message)
    except ValueError as exc:
        record_failure(exc)


def test_unasked_user_gets_an_explanation_not_an_error() -> None:
    """Someone who runs this out of curiosity is the target reader.

    Exit 0: nothing failed. The output has to say what would be collected,
    where it would be written, and that it is never uploaded on its own --
    which is the whole basis on which the answer is given.
    """
    result = runner.invoke(report_app, [])

    assert result.exit_code == 0
    assert "Nothing has been recorded yet" in result.output
    assert "failures.jsonl" in result.output
    assert "--enable" in result.output


def test_enable_and_disable_are_recorded() -> None:
    assert runner.invoke(report_app, ["--enable"]).exit_code == 0
    assert consent_state() == CONSENT_GRANTED

    result = runner.invoke(report_app, ["--disable"])

    assert result.exit_code == 0
    assert consent_state() == CONSENT_DENIED
    # Turning collection off must not delete what was already collected: the
    # user may be turning it off *because* they are about to send a report.
    assert "--clear" in result.output


def test_contradictory_flags_are_refused() -> None:
    result = runner.invoke(report_app, ["--enable", "--disable"])

    assert result.exit_code == 2


def test_report_renders_recorded_failures() -> None:
    set_consent(True)
    _record()

    result = runner.invoke(report_app, [])

    assert result.exit_code == 0
    assert "ValueError" in result.output
    assert "Most recent" in result.output
    # The user is pointed at the duplicate check before being offered a way to
    # file anything.
    assert "Already reported?" in result.output


def test_nothing_is_sent_without_an_explicit_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default invocation must not open a browser or file an issue.

    Mutation: make `--open` the default. This fails, and it is the failure that
    separates "a tool that helps you report" from "a tool that reports on you".
    """
    set_consent(True)
    _record()
    opened: list[str] = []

    def fake_open(url: str, new: int = 0, autoraise: bool = True) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(report_module.webbrowser, "open", fake_open)

    result = runner.invoke(report_app, [])

    assert result.exit_code == 0
    assert opened == []


def test_open_uses_a_prefilled_url_the_user_still_submits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_consent(True)
    _record()
    opened: list[str] = []

    def fake_open(url: str, new: int = 0, autoraise: bool = True) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(report_module.webbrowser, "open", fake_open)

    result = runner.invoke(report_app, ["--open"])

    assert result.exit_code == 0
    assert len(opened) == 1
    assert opened[0].startswith("https://github.com/UClone-AI/uclone-x/issues/new?")
    assert "nothing is sent until you" in result.output.lower()


def test_save_writes_the_report_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_consent(True)
    _record()
    opened: list[str] = []

    def fake_open(url: str, new: int = 0, autoraise: bool = True) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(report_module.webbrowser, "open", fake_open)
    destination = tmp_path / "out" / "report.md"

    result = runner.invoke(report_app, ["--save", str(destination)])

    assert result.exit_code == 0
    assert destination.is_file()
    assert "ValueError" in destination.read_text(encoding="utf-8")
    assert opened == []


def test_submit_without_gh_explains_rather_than_failing_obscurely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_consent(True)
    _record()

    def no_gh(cmd: str, mode: int = 0, path: str | None = None) -> str | None:
        return None

    monkeypatch.setattr(report_module.shutil, "which", no_gh)

    result = runner.invoke(report_app, ["--submit"])

    assert result.exit_code == 1
    assert "gh" in result.output
    assert "--open" in result.output


def test_submit_uses_the_users_own_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    """No token ships with UClone-X, so a report is filed by an account that
    consented to filing it -- or not at all."""
    set_consent(True)
    _record()
    captured: dict[str, list[str]] = {}

    class Completed:
        returncode = 0
        stdout = "https://github.com/UClone-AI/uclone-x/issues/42"
        stderr = ""

    def fake_run(cmd: list[str], **kwargs: object) -> Completed:
        captured["cmd"] = cmd
        return Completed()

    def has_gh(cmd: str, mode: int = 0, path: str | None = None) -> str:
        return "/usr/bin/gh"

    monkeypatch.setattr(report_module.shutil, "which", has_gh)
    monkeypatch.setattr(report_module.subprocess, "run", fake_run)

    result = runner.invoke(report_app, ["--submit"])

    assert result.exit_code == 0
    assert captured["cmd"][:4] == ["gh", "issue", "create", "--repo"]
    assert captured["cmd"][4] == "UClone-AI/uclone-x"
    assert "issues/42" in result.output


def test_clear_deletes_recorded_failures() -> None:
    set_consent(True)
    _record()

    result = runner.invoke(report_app, ["--clear"])

    assert result.exit_code == 0
    assert "deleted" in result.output.lower()
    assert "ValueError" not in runner.invoke(report_app, []).output


def test_a_startup_failure_is_recorded_not_only_a_failed_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unconfigured provider is the failure a new user actually hits.

    It happens while the agent is being built, so the hooks around
    `execute_turn` never see it. Measured before this was pinned: with consent
    granted, `ucx run` against an unconfigured environment left the journal
    empty, and `ucx report` told the user there was nothing to report about the
    thing that had just failed in front of them.

    Killed by: src/uclone_x/cli/commands/run.py :: "phase": "startup"
    """
    from uclone_x.cli import main as cli_main

    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LLM_PROVIDER",
        "OLLAMA_BASE_URL",
        "OLLAMA_FAST_BASE_URL",
        "OLLAMA_INDEPTH_BASE_URL",
        "LOCAL_LLM_BASE_URL",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    set_consent(True)

    result = runner.invoke(cli_main.app, ["run", "hello"])

    assert result.exit_code == 1
    entries = read_entries()
    assert entries, "a startup failure left no record"
    assert entries[-1].context.get("phase") == "startup"
    assert "provider" in entries[-1].message.lower()
