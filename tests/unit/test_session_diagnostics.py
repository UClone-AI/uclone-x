"""Unit tests for session inspection, summary, and health checks."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from uclone_x.agent.models import PlanState, PlanStep
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.core.session_diagnostics import (
    check_session_health,
    count_active_turns,
    inspect_session,
    list_session_summaries,
)
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest


def test_inspect_and_check_nonexistent_session() -> None:
    with TemporaryDirectory() as tmp:
        store = SessionStore(storage_dir=Path(tmp))
        details = inspect_session("nonexistent", store=store)
        assert details is None

        report = check_session_health("nonexistent", store=store)
        assert not report.healthy
        assert report.error_count == 1
        assert any(i.code == "SESSION_NOT_FOUND" for i in report.issues)


def test_check_invalid_session_id() -> None:
    with TemporaryDirectory() as tmp:
        store = SessionStore(storage_dir=Path(tmp))
        report = check_session_health("../escaped_id", store=store)
        assert not report.healthy
        assert report.error_count == 1
        assert any(i.code == "INVALID_SESSION_ID" for i in report.issues)


def test_inspect_and_check_clean_healthy_session() -> None:
    with TemporaryDirectory() as tmp:
        store = SessionStore(storage_dir=Path(tmp))
        state = SessionState.seed("sess_clean", "agent_1", "You are helpful.")
        state = state.with_messages(
            (
                ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
                ChatMessage(role=MessageRole.USER, content="Hello!"),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="Thinking",
                    tool_calls=(
                        ToolCallRequest(id="call_1", name="search", arguments={"q": "apple"}),
                    ),
                ),
                ChatMessage(role=MessageRole.TOOL, content="Found apple", tool_call_id="call_1"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Here is information on apples."),
            ),
            turn_counter=2,
        )
        store.save(state)

        details = inspect_session("sess_clean", store=store)
        assert details is not None
        assert details.summary.session_id == "sess_clean"
        assert details.summary.status == "active"
        assert details.summary.turn_counter == 2
        assert details.role_counts["system"] == 1
        assert details.role_counts["user"] == 1
        assert details.role_counts["assistant"] == 2
        assert details.role_counts["tool"] == 1

        report = check_session_health("sess_clean", store=store, max_conversation_turns=20)
        assert report.healthy
        assert report.error_count == 0
        assert report.warning_count == 0


def test_check_orphaned_tool_result_and_missing_id() -> None:
    with TemporaryDirectory() as tmp:
        store = SessionStore(storage_dir=Path(tmp))
        state = SessionState.seed("sess_orphaned", "agent_1")
        state = state.with_messages(
            (
                ChatMessage(role=MessageRole.USER, content="Query"),
                # Tool message with missing tool_call_id
                ChatMessage(
                    role=MessageRole.TOOL, content="Tool output without ID", tool_call_id=None
                ),
                # Tool message with orphaned ID
                ChatMessage(
                    role=MessageRole.TOOL, content="Orphaned result", tool_call_id="call_999"
                ),
            ),
            turn_counter=1,
        )
        store.save(state)

        report = check_session_health("sess_orphaned", store=store)
        assert not report.healthy
        assert report.error_count == 2
        codes = [i.code for i in report.issues]
        assert "MISSING_TOOL_CALL_ID" in codes
        assert "ORPHANED_TOOL_RESULT" in codes


def test_check_turn_budget_saturation() -> None:
    with TemporaryDirectory() as tmp:
        store = SessionStore(storage_dir=Path(tmp))
        state = SessionState.seed("sess_sat", "agent_1")
        messages: list[ChatMessage] = []
        for i in range(20):
            messages.append(ChatMessage(role=MessageRole.USER, content=f"Turn {i}"))
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=f"Answer {i}"))
        state = state.with_messages(
            tuple(messages),
            turn_counter=20,
        )
        store.save(state)

        report = check_session_health("sess_sat", store=store, max_conversation_turns=20)
        assert report.healthy  # warnings do not make healthy=False
        assert report.error_count == 0
        assert report.warning_count == 1
        assert any(i.code == "TURN_BUDGET_SATURATED" for i in report.issues)


def test_compacted_session_clears_saturation_and_status_is_compacted() -> None:
    """A compacted session past the turn budget reports 'compacted' and clears saturation (#883)."""
    with TemporaryDirectory() as tmp:
        store = SessionStore(storage_dir=Path(tmp))
        state = SessionState.seed("sess_compacted", "agent_1")
        # Session had 25 lifetime turns, but active context was compacted down to 2 messages + ledger
        state = state.with_messages(
            (
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="Compaction summary",
                    compaction_ledger=True,
                ),
                ChatMessage(role=MessageRole.USER, content="Latest query"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Latest response"),
            ),
            turn_counter=25,
        )
        store.save(state)

        details = inspect_session("sess_compacted", store=store, max_conversation_turns=20)
        assert details is not None
        assert details.summary.status == "compacted"
        assert details.summary.turn_counter == 25
        assert details.summary.has_compaction is True

        summaries = list_session_summaries(store=store, max_conversation_turns=20)
        assert len(summaries) == 1
        assert summaries[0].status == "compacted"
        assert summaries[0].turn_counter == 25

        report = check_session_health("sess_compacted", store=store, max_conversation_turns=20)
        assert report.healthy
        assert report.error_count == 0
        assert report.warning_count == 0
        assert not any(i.code == "TURN_BUDGET_SATURATED" for i in report.issues)


def test_count_active_turns() -> None:
    messages = (
        ChatMessage(role=MessageRole.SYSTEM, content="Ledger", compaction_ledger=True),
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ChatMessage(role=MessageRole.TOOL, content="Data", tool_call_id="call_1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Done"),
    )
    assert count_active_turns(messages) == 2


def test_check_unresolved_tool_calls() -> None:
    with TemporaryDirectory() as tmp:
        store = SessionStore(storage_dir=Path(tmp))
        state = SessionState.seed("sess_unresolved", "agent_1")
        state = state.with_messages(
            (
                ChatMessage(role=MessageRole.USER, content="Do work"),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=(
                        ToolCallRequest(id="call_x", name="bash", arguments={"cmd": "ls"}),
                    ),
                ),
                # User speaks before tool results arrive
                ChatMessage(role=MessageRole.USER, content="Are you done?"),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=(
                        ToolCallRequest(id="call_y", name="bash", arguments={"cmd": "pwd"}),
                    ),
                ),
            ),
            turn_counter=2,
        )
        store.save(state)

        report = check_session_health("sess_unresolved", store=store)
        assert report.healthy
        codes = [i.code for i in report.issues]
        assert "UNRESOLVED_TOOL_CALLS" in codes
        assert "UNRESOLVED_TOOL_CALLS_AT_END" in codes


def test_check_empty_message_and_missing_offload_artifact() -> None:
    with TemporaryDirectory() as tmp:
        store = SessionStore(storage_dir=Path(tmp))
        state = SessionState.seed("sess_empty", "agent_1")
        state = state.with_messages(
            (
                ChatMessage(role=MessageRole.USER, content=""),  # empty content
                ChatMessage(
                    role=MessageRole.TOOL,
                    content="[Tool Output Offloaded (path=offload, size=9999): Excerpt Full output saved to '.sandbox/tool_artifacts/missing.txt'. Use file_read to inspect.]",
                    tool_call_id="call_off",
                ),
            ),
            turn_counter=1,
        )
        store.save(state)

        report = check_session_health("sess_empty", store=store, workspace_root=Path(tmp))
        codes = [i.code for i in report.issues]
        assert "EMPTY_MESSAGE_CONTENT" in codes
        assert "ORPHANED_TOOL_RESULT" in codes
        assert "MISSING_OFFLOAD_ARTIFACT" in codes


def test_list_session_summaries_and_plan_inspection() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store = SessionStore(storage_dir=tmp_path)

        plan = PlanState(
            title="Refactor Session",
            steps=(
                PlanStep(index=0, description="Step 1", completed=True),
                PlanStep(index=1, description="Step 2", completed=False),
            ),
        )

        state1 = SessionState.seed("sess_1", "agent_a").with_messages(
            (ChatMessage(role=MessageRole.USER, content="First"),),
            turn_counter=1,
            updated_at="2026-09-01T10:00:00Z",
        )
        state2 = (
            SessionState.seed("sess_2", "agent_b")
            .with_messages(
                (
                    ChatMessage(role=MessageRole.SYSTEM, content="Ledger", compaction_ledger=True),
                    ChatMessage(role=MessageRole.USER, content="Second"),
                ),
                turn_counter=5,
                updated_at="2026-09-02T10:00:00Z",
            )
            .with_plan(plan)
        )

        store.save(state1)
        store.save(state2)

        summaries = list_session_summaries(store=store, limit=10)
        assert len(summaries) == 2
        # sorted descending by updated_at
        assert summaries[0].session_id == "sess_2"
        assert summaries[0].status == "compacted"
        assert summaries[0].has_compaction
        assert summaries[0].has_plan
        assert summaries[1].session_id == "sess_1"

        details = inspect_session("sess_2", store=store)
        assert details is not None
        assert details.has_plan
        assert details.plan_title == "Refactor Session"
        assert details.plan_steps_total == 2
        assert details.plan_steps_completed == 1
