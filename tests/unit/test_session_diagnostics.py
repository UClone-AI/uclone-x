"""Unit tests for session inspection, summary, and health checks."""

from __future__ import annotations

from collections.abc import Sequence
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
from uclone_x.core.session_log import LoggedMessage
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
                    content="[Stored tool result tr_00000000000000cc: 9,999 characters]\nx",
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
        # Output that only quotes a stored-result header is its own full text: it records
        # no kept result, so none is missing (#1974, item 8).
        assert "MISSING_OFFLOAD_ARTIFACT" not in codes


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


def _logged(
    state: SessionState, kept: Sequence[LoggedMessage], store: SessionStore | None = None
) -> SessionState:
    """`state` as a session writes it: a log of the `kept` texts, then each message,
    and the entry each message is (#1848) -- a record naming none is refused as old."""
    from uclone_x.core.session_log import SessionLogProvenance, logged_message, new_entry

    msg_logged = [logged_message(m) for m in state.messages]
    if store is not None:
        for lm in msg_logged:
            store.save_context_body(state.session_id, lm.digest, lm.body)
    logged = [*kept, *msg_logged]
    return state.model_copy(
        update={
            "session_log": tuple(
                new_entry(i, r, turn=1, provenance=SessionLogProvenance.RECORDED)
                for i, r in enumerate(logged)
            ),
            "history_entries": tuple(f"e{i}" for i in range(len(kept), len(logged))),
        }
    )


def test_check_a_stored_tool_result_whose_blob_is_missing(tmp_path: Path) -> None:
    """A `tr_` handle is checked against the session's own log and body store (#1848).

    A handle whose full text the log names but the store no longer holds is flagged, as
    is one the log never named -- a handle from before #1848, whose file is not looked at.
    The handle is the one each form records, never one read from its text (#1974, item 8).

    Killed by: src/uclone_x/core/session_diagnostics.py :: if entry is None or store.load_context_body(session_id, entry.digest) is None:
    Becomes: if entry is None:
    """
    from uclone_x.core.session_log import (
        SessionLogKind,
        logged_text,
    )
    from uclone_x.core.tool_results import result_handle
    from uclone_x.llm.models import RenderedFrom

    store = SessionStore(storage_dir=tmp_path / "sessions")
    kept_body, lost_body = "the kept body", "the lost body"
    present, lost = result_handle(kept_body), result_handle(lost_body)
    missing = "tr_00000000000000bb"
    kept = logged_text(SessionLogKind.TOOL_RESULT, kept_body, blob=present)
    gone = logged_text(SessionLogKind.TOOL_RESULT, lost_body, blob=lost)
    store.save_context_body("sess_tr", kept.digest, kept.body)
    calls = tuple(ToolCallRequest(id=f"c{i}", name="t") for i in range(3))
    state = SessionState.seed("sess_tr", "agent_1").with_messages(
        (
            ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=calls),
            *(
                ChatMessage(
                    role=MessageRole.TOOL,
                    tool_call_id=f"c{i}",
                    form="stub",
                    rendered_from=RenderedFrom(handle=handle, limit=100, readable=True),
                )
                for i, handle in enumerate((present, lost, missing))
            ),
        ),
        turn_counter=1,
    )
    store.save(_logged(state, (kept, gone), store=store))

    report = check_session_health("sess_tr", store=store, workspace_root=tmp_path)

    flagged = [i for i in report.issues if i.code == "MISSING_OFFLOAD_ARTIFACT"]
    assert [i.details.get("handle") for i in flagged] == [lost, missing]
    for issue in flagged:
        assert "/" not in issue.message and "Errno" not in issue.message


def test_output_that_quotes_a_stored_result_header_names_no_kept_result(tmp_path: Path) -> None:
    """A tool result whose own text begins like a stored-result header is full output
    (#1854). Its log entry names no blob, and the health check does not flag a kept result
    as missing for it: a handle is read from what a message records, never from its text
    (#1974, item 8).

    Killed by: src/uclone_x/core/session_log.py :: blob: str | None = message.rendered_from.handle if message.rendered_from is not None else None
    Becomes: blob: str | None = message.rendered_from.handle if message.rendered_from is not None else __import__("uclone_x.core.tool_results", fromlist=["_"]).handle_in(message.content)
    Killed by: src/uclone_x/core/session_diagnostics.py :: handle: str | None = msg.rendered_from.handle if msg.rendered_from is not None else None
    Becomes: handle: str | None = msg.rendered_from.handle if msg.rendered_from is not None else __import__("uclone_x.core.tool_results", fromlist=["_"]).handle_in(msg.content)
    """
    from uclone_x.core.session_log import SessionLogProvenance, logged_message, new_entry

    store = SessionStore(storage_dir=tmp_path / "sessions")
    quoting = ChatMessage(
        role=MessageRole.TOOL,
        content="[Stored tool result tr_00000000000000cc: 9,999 characters]\nquoted in a file",
        tool_call_id="c0",
    )
    call = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=(ToolCallRequest(id="c0", name="t"),),
    )
    logged = [logged_message(m) for m in (call, quoting)]
    assert logged[1].blob is None
    for item in logged:
        store.save_context_body("sess_quote", item.digest, item.body)
    state = SessionState.seed("sess_quote", "agent_1").with_messages(
        (call, quoting), turn_counter=1
    )
    state = state.model_copy(
        update={
            "session_log": tuple(
                new_entry(i, item, turn=1, provenance=SessionLogProvenance.RECORDED)
                for i, item in enumerate(logged)
            )
        }
    )
    store.save(state)

    report = check_session_health("sess_quote", store=store, workspace_root=tmp_path)

    assert not [i for i in report.issues if i.code == "MISSING_OFFLOAD_ARTIFACT"]


def test_check_a_form_recorded_with_no_text_by_the_result_it_records(tmp_path: Path) -> None:
    """A saved excerpt or stub holds no text, only the kept result it is rendered from
    (#1848). It is not an empty message, and its kept result is checked by the handle it
    records: present, it is healthy; lost, it is flagged.

    Killed by: src/uclone_x/core/session_diagnostics.py :: if not has_content and not has_calls and msg.rendered_from is None:
    Becomes: if not has_content and not has_calls:
    Killed by: src/uclone_x/core/session_diagnostics.py :: handle: str | None = msg.rendered_from.handle
    Becomes: handle: str | None = None
    """
    from typing import Literal

    from uclone_x.core.session_log import (
        SessionLogKind,
        logged_text,
    )
    from uclone_x.core.tool_results import result_handle
    from uclone_x.llm.models import RenderedFrom

    forms: tuple[tuple[str, Literal["excerpt", "stub"]], ...]

    store = SessionStore(storage_dir=tmp_path / "sessions")
    kept_body, lost_body = "the kept body", "the lost body"
    present, lost = result_handle(kept_body), result_handle(lost_body)
    kept = logged_text(SessionLogKind.TOOL_RESULT, kept_body, blob=present)
    gone = logged_text(SessionLogKind.TOOL_RESULT, lost_body, blob=lost)
    store.save_context_body("sess_form", kept.digest, kept.body)
    calls = tuple(ToolCallRequest(id=f"c{i}", name="t") for i in range(2))
    forms = ((present, "excerpt"), (lost, "stub"))
    state = SessionState.seed("sess_form", "agent_1").with_messages(
        (
            ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=calls),
            *(
                ChatMessage(
                    role=MessageRole.TOOL,
                    tool_call_id=f"c{i}",
                    form=form,
                    rendered_from=RenderedFrom(handle=handle, limit=100, readable=True),
                )
                for i, (handle, form) in enumerate(forms)
            ),
        ),
        turn_counter=1,
    )
    store.save(_logged(state, (kept, gone), store=store))

    report = check_session_health("sess_form", store=store, workspace_root=tmp_path)

    assert not [i for i in report.issues if i.code == "EMPTY_MESSAGE_CONTENT"]
    flagged = [i for i in report.issues if i.code == "MISSING_OFFLOAD_ARTIFACT"]
    assert [i.details.get("handle") for i in flagged] == [lost]


def test_inspect_lists_each_kept_result_once_and_skips_a_lost_body(tmp_path: Path) -> None:
    """The listing names each handle once, in log order, and only if its body is stored.

    A handle logged twice is one kept result, and one whose body the store no longer holds
    is not listed or counted (#1848).

    Killed by: src/uclone_x/core/session_diagnostics.py :: if store.load_context_body(state.session_id, entry.digest) is None:
    Becomes: if False:
    """
    from uclone_x.core.session_log import (
        SessionLogKind,
        SessionLogProvenance,
        logged_text,
        new_entry,
    )
    from uclone_x.core.tool_results import result_handle

    store = SessionStore(storage_dir=tmp_path / "sessions")
    first_body, second_body, lost_body = "the first body", "the second, longer body", "lost"
    first, second, lost = (result_handle(b) for b in (first_body, second_body, lost_body))
    records = (
        logged_text(SessionLogKind.TOOL_RESULT, first_body, blob=first),
        logged_text(SessionLogKind.TOOL_RESULT, lost_body, blob=lost),
        logged_text(SessionLogKind.TOOL_RESULT, second_body, blob=second),
        logged_text(SessionLogKind.TOOL_RESULT, first_body, blob=first),
    )
    for record in (records[0], records[2]):
        store.save_context_body("sess_list", record.digest, record.body)
    state = SessionState.seed("sess_list", "agent_1").model_copy(
        update={
            "session_log": tuple(
                new_entry(i, r, turn=1, provenance=SessionLogProvenance.RECORDED)
                for i, r in enumerate(records)
            )
        }
    )
    store.save(state)

    details = inspect_session("sess_list", store=store)

    assert details is not None
    assert details.artifact_files == [second, first]
    assert details.artifacts_count == 2
    assert details.artifacts_total_bytes == len(first_body) + len(second_body)


def test_measure_context_payload_and_context_payload_bloat_anomaly(tmp_path: Path) -> None:
    """`measure_context_payload` and `check_session_health` detect `CONTEXT_PAYLOAD_BLOAT`
    when active messages accumulate large `thoughtSignature` blobs or exceed the compaction
    token threshold.

    Killed by: src/uclone_x/core/session_diagnostics.py :: if est_tokens >= compaction_threshold_tokens or sig_bytes >= SIGNATURE_BLOAT_BYTES_THRESHOLD:
    Becomes: if False:
    """
    from uclone_x.core.session_diagnostics import measure_context_payload

    store = SessionStore(storage_dir=tmp_path / "sessions")
    sig_blob = "S" * 40_000
    messages = (
        ChatMessage(role=MessageRole.USER, content="Compose multi-bar music"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=(
                ToolCallRequest(
                    id="tc_1",
                    name="compose_bar",
                    arguments={"bar": 1},
                    signature=sig_blob,
                ),
                ToolCallRequest(
                    id="tc_2",
                    name="compose_bar",
                    arguments={"bar": 2},
                    signature=sig_blob,
                ),
            ),
        ),
        ChatMessage(role=MessageRole.TOOL, content="ok", tool_call_id="tc_1"),
        ChatMessage(role=MessageRole.TOOL, content="ok", tool_call_id="tc_2"),
    )
    est_tokens, payload_bytes, sig_count, sig_bytes = measure_context_payload(messages)
    assert sig_count == 2
    assert sig_bytes == 80_000
    assert payload_bytes > 80_000
    assert est_tokens >= 20_000

    state = SessionState.seed("sess_bloat", "composer").with_messages(messages, turn_counter=1)
    store.save(state)

    details = inspect_session("sess_bloat", store=store)
    assert details is not None
    assert details.signature_count == 2
    assert details.signature_bytes == 80_000
    assert details.payload_bytes == payload_bytes
    assert details.estimated_tokens == est_tokens

    report = check_session_health("sess_bloat", store=store)
    bloat = [i for i in report.issues if i.code == "CONTEXT_PAYLOAD_BLOAT"]
    assert len(bloat) == 1
    assert bloat[0].details["signature_count"] == 2
    assert bloat[0].details["signature_bytes"] == 80_000


def test_tool_retry_spiral_and_misrouted_install_package_detected(tmp_path: Path) -> None:
    """`check_session_health` flags `TOOL_RETRY_SPIRAL` when `generate_image` fails with a
    connection refusal and the agent calls `install_package` or retries in the same turn.

    Killed by: src/uclone_x/core/session_diagnostics.py :: misrouted = called_install_package and t_name in refusal_tools
    Becomes: misrouted = False
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    state = SessionState.seed("sess_fiona", "fiona").with_messages(
        (
            ChatMessage(role=MessageRole.USER, content="Draw a portrait"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Could not draw."),
        ),
        turn_counter=1,
    )
    store.save(state)

    events = [
        {
            "type": "TOOL_EXECUTION",
            "turn_index": 1,
            "payload": {
                "step": 1,
                "tool_name": "generate_image",
                "success": False,
                "error": (
                    "The picture model remote-gpu-pictures/anillustrious_v4 cannot be used: "
                    "there is no connection called remote-gpu-pictures."
                ),
            },
        },
        {
            "type": "TOOL_EXECUTION",
            "turn_index": 1,
            "payload": {
                "step": 2,
                "tool_name": "install_package",
                "success": False,
                "error": "Package 'diffusers' is not an optional dependency of this project.",
            },
        },
    ]

    report = check_session_health("sess_fiona", store=store, events=events)
    spirals = [i for i in report.issues if i.code == "TOOL_RETRY_SPIRAL"]
    assert len(spirals) == 1
    assert spirals[0].turn_index == 1
    assert spirals[0].details["tool_name"] == "generate_image"
    assert spirals[0].details["misrouted_install_package"] is True


def test_read_timeout_masked_and_dangling_model_ref_detected(tmp_path: Path) -> None:
    """`check_session_health` flags `READ_TIMEOUT_MASKED` when a ~60s step ends with an
    unreachable/timeout error, and flags `DANGLING_MODEL_REF` when a configured model ref
    points to a connection id missing from `settings.json`.

    Killed by: src/uclone_x/core/session_diagnostics.py :: if slow_steps and matched_errors:
    Becomes: if False:
    Killed by: src/uclone_x/core/session_diagnostics.py :: if parsed_ref.connection_id not in conn_ids:
    Becomes: if False:
    """
    from uclone_x.core.session_diagnostics import diagnose_session_forensics

    store = SessionStore(storage_dir=tmp_path / "sessions")
    state = SessionState.seed("sess_timeout", "fiona").with_messages(
        (ChatMessage(role=MessageRole.USER, content="Compose and illustrate"),),
        turn_counter=5,
    )
    store.save(state)

    events = [
        {
            "type": "USER_INPUT",
            "turn_index": 5,
            "payload": {"content": "Compose and illustrate"},
        },
        {
            "type": "MODEL_RESPONSE",
            "turn_index": 5,
            "payload": {
                "step": 1,
                "model_name": "gemini-3-flash-preview",
                "started_at": "2026-11-08T00:00:00+00:00",
                "ended_at": "2026-11-08T00:01:01.250000+00:00",
                "tool_calls": [{"name": "compose_music"}],
            },
        },
        {
            "type": "ERROR",
            "turn_index": 5,
            "payload": {
                "step": 2,
                "model_name": "gemini-3-flash-preview",
                "started_at": "2026-11-08T00:01:02+00:00",
                "ended_at": "2026-11-08T00:02:02.500000+00:00",
                "error": (
                    "We couldn't get an answer from gemini (gemini-3-flash-preview): "
                    "the service couldn't be reached."
                ),
            },
        },
    ]
    settings_data = {
        "connections": [{"id": "gemini", "kind": "gemini", "key": "k"}],
        "default_models": {"deep": "gemini/gemini-3-flash-preview", "image": "auto"},
    }
    clone_configs = {
        "fiona": {"image_model": "remote-gpu-pictures/anillustrious_v4"},
    }

    forensics = diagnose_session_forensics(
        "sess_timeout",
        store=store,
        events=events,
        settings_data=settings_data,
        clone_configs=clone_configs,
    )
    assert forensics is not None
    codes = {i.code for i in forensics.health.issues}
    assert "READ_TIMEOUT_MASKED" in codes
    assert "DANGLING_MODEL_REF" in codes
    assert len(forensics.turns) == 1
    assert forensics.turns[0].turn_index == 5
    assert forensics.turns[0].max_step_duration_seconds >= 60.0
    assert forensics.turns[0].stop_reason == "error"


def test_real_serialized_jsonl_schema_and_out_of_order_turns(tmp_path: Path) -> None:
    """`diagnose_session_forensics` handles real top-level `serialize_session_event` JSONL
    records (`TURN_START`, `USER_MESSAGE`, `TOOL_CALL`, `TOOL_RESULT` with `duration_ms` and
    `result_handle`, dict `error`, `TURN_ROLLED_BACK`, and out-of-order turn numbering).

    Killed by: src/uclone_x/core/session_diagnostics.py :: if any(marker in txt.lower() for marker in _TIMEOUT_ERROR_MARKERS)
    Becomes: if False
    """
    from uclone_x.core.session_diagnostics import diagnose_session_forensics
    from uclone_x.core.session_log import (
        SessionLogKind,
        SessionLogProvenance,
        logged_text,
        new_entry,
    )
    from uclone_x.core.tool_results import result_handle

    store = SessionStore(storage_dir=tmp_path / "sessions")
    refusal_body = (
        "Error: Tool 'generate_image' failed: The picture model "
        "remote-gpu-pictures/anillustrious_v4 cannot be used: "
        "there is no connection called remote-gpu-pictures."
    )
    handle = result_handle(refusal_body)
    record = logged_text(SessionLogKind.TOOL_RESULT, refusal_body, blob=handle)
    store.save_context_body("sess_real", record.digest, record.body)
    state = SessionState.seed("sess_real", "fiona").model_copy(
        update={
            "turn_counter": 41,
            "session_log": (
                new_entry(0, record, turn=41, provenance=SessionLogProvenance.RECORDED),
            ),
        }
    )
    store.save(state)

    events: list[dict[str, object]] = [
        {"type": "TURN_START", "turn_index": 48, "at": "2026-11-01T00:00:00+00:00"},
        {"type": "USER_MESSAGE", "content": "Old turn before compaction"},
        {
            "type": "MODEL_RESPONSE",
            "turn_index": 48,
            "step": 1,
            "started_at": "2026-11-01T00:00:00+00:00",
            "ended_at": "2026-11-01T00:01:00.140000+00:00",
            "model_name": "gemini-flash-latest",
            "finish_reason": "error",
            "tool_calls": [],
            "error": {
                "type": "LLMStreamInterruptedError",
                "message": (
                    "The streamed reply from gemini/gemini-flash-latest was interrupted after "
                    "0 chunk(s) by ProviderUnreachableError: Couldn't get an answer from Google."
                ),
            },
        },
        {"type": "TURN_ROLLED_BACK", "turn_index": 48, "reason": "turn_not_committed"},
        {"type": "TURN_START", "turn_index": 41, "at": "2026-11-08T00:00:00+00:00"},
        {"type": "USER_MESSAGE", "content": "Room header\n: 그림 또 만들어줘"},
        {"type": "ASSISTANT_ATTEMPT", "turn_index": 41, "step": 1},
        {"type": "TOOL_CALL", "tool_call_id": "tc_img_1", "name": "generate_image"},
        {
            "type": "TOOL_RESULT",
            "tool_call_id": "tc_img_1",
            "name": "generate_image",
            "result_handle": handle,
            "status": "error",
            "outcome": "errored",
            "duration_ms": 420,
        },
        {"type": "ASSISTANT_ATTEMPT", "turn_index": 41, "step": 2},
        {"type": "TOOL_CALL", "tool_call_id": "tc_pkg_1", "name": "install_package"},
        {
            "type": "TOOL_RESULT",
            "tool_call_id": "tc_pkg_1",
            "name": "install_package",
            "result_handle": "tr_pkg_1",
            "status": "error",
            "outcome": "errored",
            "duration_ms": 150,
        },
        {
            "type": "TURN_END",
            "turn_index": 41,
            "outcome": "completed",
            "steps": 2,
            "tool_executions": 2,
            "stop_reason": "model_stopped",
        },
    ]
    settings_data = {
        "connections": [{"id": "gemini", "kind": "gemini", "key": "k"}],
        "default_models": {"deep": "gemini/gemini-flash-latest", "image": "auto"},
    }

    forensics = diagnose_session_forensics(
        "sess_real",
        store=store,
        events=events,
        settings_data=settings_data,
        tail_turns=2,
    )
    assert forensics is not None
    codes = {i.code for i in forensics.health.issues}
    assert "READ_TIMEOUT_MASKED" in codes
    assert "TOOL_RETRY_SPIRAL" in codes
    assert "DANGLING_MODEL_REF" in codes
    assert [t.turn_index for t in forensics.turns] == [48, 41]
    assert forensics.turns[0].stop_reason == "rolled_back (turn_not_committed)"
    assert forensics.turns[1].user_prompt == ": 그림 또 만들어줘"
    assert forensics.turns[1].steps[0].duration_seconds == 0.42
