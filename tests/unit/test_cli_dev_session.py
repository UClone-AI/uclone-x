"""Unit tests for ucx dev session and ucx dev logs CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from typer.testing import CliRunner

from uclone_x.agent.models import PlanState, PlanStep
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.cli import main
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest

runner = CliRunner()


def test_cli_dev_session_list_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as tmp:
        monkeypatch.setenv("UCLONE_SESSION_DIR", tmp)
        res = runner.invoke(main.app, ["dev", "session", "list"])
        assert res.exit_code == 0
        assert "No sessions found in store" in res.output


def test_cli_dev_session_list_and_show(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as tmp:
        monkeypatch.setenv("UCLONE_SESSION_DIR", tmp)
        store = SessionStore()
        state = SessionState.seed("sess_test1", "champion")
        state = state.with_messages(
            (
                ChatMessage(role=MessageRole.SYSTEM, content="System prompt"),
                ChatMessage(role=MessageRole.USER, content="Hello there!"),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=(
                        ToolCallRequest(id="c1", name="search", arguments={"query": "test"}),
                    ),
                ),
                ChatMessage(role=MessageRole.TOOL, content="Found results", tool_call_id="c1"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Search finished successfully."),
            ),
            turn_counter=2,
        )
        store.save(state)

        # 1. Test list
        res_list = runner.invoke(main.app, ["dev", "session", "list"])
        assert res_list.exit_code == 0
        assert "sess_test1" in res_list.output
        assert "champion" in res_list.output
        assert "Active" in res_list.output

        # 2. Test show
        res_show = runner.invoke(main.app, ["dev", "session", "show", "sess_test1"])
        assert res_show.exit_code == 0
        assert "Session Turn Inspector" in res_show.output
        assert "SYSTEM" in res_show.output
        assert "USER" in res_show.output
        assert "TOOL" in res_show.output
        assert "search" in res_show.output

        # 3. Test show non-existent
        res_missing = runner.invoke(main.app, ["dev", "session", "show", "sess_unknown"])
        assert res_missing.exit_code == 1
        assert "Session not found" in res_missing.output


def test_cli_dev_session_status(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as tmp:
        monkeypatch.setenv("UCLONE_SESSION_DIR", tmp)
        store = SessionStore()

        plan = PlanState(
            title="Migration Plan",
            steps=(PlanStep(index=0, description="Step 1", completed=True),),
        )
        state = (
            SessionState.seed("sess_status_test", "champion")
            .with_messages(
                (
                    ChatMessage(role=MessageRole.USER, content="Run task"),
                    ChatMessage(role=MessageRole.ASSISTANT, content="Done"),
                ),
                turn_counter=10,
            )
            .with_plan(plan)
        )
        store.save(state)

        res = runner.invoke(
            main.app, ["dev", "session", "status", "sess_status_test", "--max-turns", "20"]
        )
        assert res.exit_code == 0
        assert "Session Status: sess_status_test" in res.output
        assert "10/20 (50.0%)" in res.output
        assert "Migration Plan" in res.output
        assert "1 / 1 steps completed" in res.output


def test_cli_dev_session_check(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as tmp:
        monkeypatch.setenv("UCLONE_SESSION_DIR", tmp)
        store = SessionStore()

        # Healthy session
        state_ok = SessionState.seed("sess_healthy", "champion").with_messages(
            (
                ChatMessage(role=MessageRole.USER, content="Hello"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
            ),
            turn_counter=1,
        )
        store.save(state_ok)

        # Unhealthy session (orphaned tool result)
        state_bad = SessionState.seed("sess_bad", "champion").with_messages(
            (
                ChatMessage(
                    role=MessageRole.TOOL, content="Orphan", tool_call_id="call_unregistered"
                ),
            ),
            turn_counter=1,
        )
        store.save(state_bad)

        # Check specific healthy session
        res_ok = runner.invoke(main.app, ["dev", "session", "check", "sess_healthy"])
        assert res_ok.exit_code == 0
        assert "Healthy (0 issues)" in res_ok.output

        # Check specific bad session
        res_bad = runner.invoke(main.app, ["dev", "session", "check", "sess_bad"])
        assert res_bad.exit_code == 1
        assert "ORPHANED_TOOL_RESULT" in res_bad.output

        # Check all sessions (should exit 1 due to sess_bad)
        res_all = runner.invoke(main.app, ["dev", "session", "check"])
        assert res_all.exit_code == 1
        assert "sess_healthy" in res_all.output
        assert "sess_bad" in res_all.output


def test_cli_dev_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as tmp:
        monkeypatch.setenv("UCX_LOG_DIR", tmp)
        log_file = Path(tmp) / "ucx.log"

        # Missing log file
        res_empty = runner.invoke(main.app, ["dev", "logs"])
        assert res_empty.exit_code == 0
        assert "No application log file found" in res_empty.output

        # Create log file with structured records
        lines = [
            json.dumps(
                {
                    "timestamp": "2026-09-06T12:00:00Z",
                    "level": "INFO",
                    "logger": "agent",
                    "message": "Initialized",
                    "session_id": "sess_1",
                }
            ),
            json.dumps(
                {
                    "timestamp": "2026-09-06T12:01:00Z",
                    "level": "WARN",
                    "logger": "tool",
                    "message": "Slow request",
                    "session_id": "sess_1",
                }
            ),
            json.dumps(
                {
                    "timestamp": "2026-09-06T12:02:00Z",
                    "level": "ERROR",
                    "logger": "core",
                    "message": "Failed step",
                    "session_id": "sess_2",
                }
            ),
        ]
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Text output
        res_text = runner.invoke(main.app, ["dev", "logs", "--level", "WARN"])
        assert res_text.exit_code == 0
        assert "Slow request" in res_text.output
        assert "Failed step" in res_text.output
        assert "Initialized" not in res_text.output

        # JSON output
        res_json = runner.invoke(
            main.app, ["dev", "logs", "--format", "json", "--session", "sess_2"]
        )
        assert res_json.exit_code == 0
        assert "Failed step" in res_json.output
        assert "Slow request" not in res_json.output


# ======================================================================================
# Issue #665 (follow-up) — the shallow unwrap outside `llm/connectors/`
#
# The four connector fixes in this PR were not the whole defect. `dict(tc.arguments)` is
# a shallow copy of a *recursively* frozen mapping wherever it is written, and three more
# sites reached a JSON encoder with the nested `MappingProxyType`s still in place. The
# tests below pin those sites. They deliberately assert an **observable outcome** rather
# than "no exception was raised": two of the three failed *quietly* — a refused tool call
# and a transcript dropped from disk — so a test that only asserted the absence of an
# exception would have passed against the broken code.
# ======================================================================================


_NESTED_TOOL_ARGUMENTS: dict[str, Any] = {
    "path": "README.md",
    "options": {"encoding": "utf-8"},
    "ranges": [{"start": 1, "end": 20}],
}
"""The same shape as the connector tests: an object nested in an object and in a list."""


def test_cli_dev_session_show_renders_nested_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ucx dev session show` survives a transcript whose tool call has nested arguments.

    `json.dumps(dict(tc.arguments))` here was character-for-character the OpenAI defect,
    and unlike the connectors it sits in no `try`: the inspector died outright with
    `TypeError: Object of type mappingproxy is not JSON serializable` on any session
    carrying a structured tool call — which is every call the connector fix in this PR
    now successfully sends.

    Killed by: src/uclone_x/cli/commands/session_dev.py :: json.dumps(unwrap_immutable(tc.arguments))
    """
    with TemporaryDirectory() as tmp:
        monkeypatch.setenv("UCLONE_SESSION_DIR", tmp)
        store = SessionStore()
        state = SessionState.seed("sess_nested", "champion").with_messages(
            (
                ChatMessage(role=MessageRole.USER, content="Read the file"),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=(
                        ToolCallRequest(
                            id="call_1", name="read_file", arguments=_NESTED_TOOL_ARGUMENTS
                        ),
                    ),
                ),
                ChatMessage(role=MessageRole.TOOL, content="ok", tool_call_id="call_1"),
            ),
            turn_counter=1,
        )
        store.save(state)

        res = runner.invoke(main.app, ["dev", "session", "show", "sess_nested"])

        assert res.exception is None, f"inspector crashed: {res.exception!r}"
        assert res.exit_code == 0
        # The call is rendered, not merely survived: the preview is what the crash ate.
        assert "read_file" in res.output
