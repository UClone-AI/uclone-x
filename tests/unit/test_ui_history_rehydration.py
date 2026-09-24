# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
"""Unit tests for session history rehydration in the Developer UI (Issue #387).

Pins fidelity of ChatMessage reconstruction across USER, ASSISTANT, and TOOL roles:
- Preserves tool identity (name, tool_call_id, tool_calls).
- Distinguishes absent/None content from genuinely empty string content ("").
- Fails loudly with SessionHistoryRehydrationError when a TOOL message lacks identity
  and cannot be repaired, preventing unmappable messages from reaching LLM connectors (P6).
- Preserves tool identity during Core session history synthesis in AgentSessionManager.get_session_history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tests.support.ui_copy import PRINCIPLE_NUMBER
from uclone_x.errors import SessionHistoryRehydrationError
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest
from uclone_x.ui.app import AgentSessionManager, _translate_session_error, create_ui_app


@pytest.fixture(autouse=True)
def _declare_a_connector(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every test here builds an agent; none is about provider resolution.

    `tests/conftest.py` clears LLM configuration rather than pinning a provider, so an
    unconfigured build is refused (#533) instead of silently producing a connector that
    cannot work. Saying `mock` once for the module is the honest form of what these tests
    were relying on implicitly.
    """
    monkeypatch.setenv("LLM_PROVIDER", "mock")


def test_rehydration_user_message_fidelity(tmp_path: Path) -> None:
    """USER messages preserve content and optional user name verbatim.

    Mutation this catches: dropping user name or substituting a default role.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw_msgs = [
        {"role": "user", "content": "Hello world", "name": "alice"},
        {"role": "user", "content": "Follow up"},
    ]
    chat_msgs, typed_msgs = mgr.reconstruct_history(raw_msgs, session_id="test_sess")

    assert len(chat_msgs) == 2
    assert chat_msgs[0] == ChatMessage(
        role=MessageRole.USER,
        content="Hello world",
        name="alice",
    )
    assert chat_msgs[1] == ChatMessage(
        role=MessageRole.USER,
        content="Follow up",
        name=None,
    )
    assert len(typed_msgs) == 2


def test_rehydration_assistant_message_with_tool_calls(tmp_path: Path) -> None:
    """ASSISTANT messages reconstruct tool_calls as ToolCallRequest tuples.

    Mutation this catches: dropping tool_calls or failing to convert dicts to ToolCallRequest.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw_msgs = [
        {
            "role": "assistant",
            "content": "Running inspection",
            "tool_calls": [
                {
                    "id": "call_inspect_1",
                    "name": "inspect_directory",
                    "arguments": {"path": "/workspace"},
                }
            ],
        }
    ]
    chat_msgs, _ = mgr.reconstruct_history(raw_msgs, session_id="test_sess")

    assert len(chat_msgs) == 1
    msg = chat_msgs[0]
    assert msg.role == MessageRole.ASSISTANT
    assert msg.content == "Running inspection"
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0] == ToolCallRequest(
        id="call_inspect_1",
        name="inspect_directory",
        arguments={"path": "/workspace"},
    )


def test_rehydration_assistant_recovers_tool_calls_from_tool_executions(tmp_path: Path) -> None:
    """When tool_calls is absent but tool_executions is present, reconstruct tool calls.

    UI presentation records may hold tool execution records without an explicit tool_calls
    list. The rehydration path recovers the structured tool calls from execution metadata.

    Mutation this catches: ignoring tool_executions when tool_calls is absent.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw_msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_executions": [
                {
                    "tool_call_id": "call_exec_1",
                    "tool_name": "shell_exec",
                    "arguments": {"command": "pytest"},
                    "output": "1 passed",
                }
            ],
        }
    ]
    chat_msgs, _ = mgr.reconstruct_history(raw_msgs, session_id="test_sess")

    assert len(chat_msgs) == 1
    msg = chat_msgs[0]
    assert msg.role == MessageRole.ASSISTANT
    assert msg.content is None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0] == ToolCallRequest(
        id="call_exec_1",
        name="shell_exec",
        arguments={"command": "pytest"},
    )


def test_rehydration_tool_message_preserves_identity(tmp_path: Path) -> None:
    """TOOL message preserves name, tool_call_id, and content.

    Prior to Issue #387, rehydration omitted name and tool_call_id, yielding
    ChatMessage(role=TOOL, name=None, tool_call_id=None), which caused live raises
    in GeminiConnector._build_payload.

    Mutation this catches: omitting name or tool_call_id when constructing TOOL ChatMessage.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw_msgs = [
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "call_read_1",
            "content": "file contents",
        }
    ]
    chat_msgs, _ = mgr.reconstruct_history(raw_msgs, session_id="test_sess")

    assert len(chat_msgs) == 1
    msg = chat_msgs[0]
    assert msg.role == MessageRole.TOOL
    assert msg.name == "read_file"
    assert msg.tool_call_id == "call_read_1"
    assert msg.content == "file contents"


def test_rehydration_tool_message_repaired_from_tool_name(tmp_path: Path) -> None:
    """TOOL message missing 'name' but carrying 'tool_name' is safely repaired.

    Some UI execution records store the key as 'tool_name' rather than 'name'.
    Rehydration adopts 'tool_name' rather than rejecting the valid record.

    Mutation this catches: only inspecting item_dict['name'].
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw_msgs = [
        {
            "role": "tool",
            "tool_name": "web_search",
            "tool_call_id": "call_search_1",
            "content": "search results",
        }
    ]
    chat_msgs, typed_msgs = mgr.reconstruct_history(raw_msgs, session_id="test_sess")

    assert len(chat_msgs) == 1
    assert chat_msgs[0].name == "web_search"
    assert chat_msgs[0].tool_call_id == "call_search_1"
    # Verify in-memory transcript dict is also updated with repaired name
    assert typed_msgs[0]["name"] == "web_search"


def test_rehydration_tool_message_repaired_from_preceding_assistant_tool_call(
    tmp_path: Path,
) -> None:
    """TOOL message missing a name is repaired if tool_call_id matches a preceding tool call.

    Positive evidence in the same transcript establishes identity without guessing.

    Mutation this catches: omitting cross-turn known_tool_calls lookup.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw_msgs = [
        {
            "role": "assistant",
            "content": "Let me search",
            "tool_calls": [{"id": "call_lookup_1", "name": "vector_query", "arguments": {}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_lookup_1",
            "content": "match found",
        },
    ]
    chat_msgs, typed_msgs = mgr.reconstruct_history(raw_msgs, session_id="test_sess")

    assert len(chat_msgs) == 2
    tool_msg = chat_msgs[1]
    assert tool_msg.role == MessageRole.TOOL
    assert tool_msg.name == "vector_query"
    assert tool_msg.tool_call_id == "call_lookup_1"
    assert typed_msgs[1]["name"] == "vector_query"


def test_rehydration_tool_message_missing_identity_fails_loudly(tmp_path: Path) -> None:
    """A TOOL message with no name and no matching tool_call_id raises SessionHistoryRehydrationError.

    Principle 6 forbids silently fabricating a tool name (e.g. 'tool') or handing a
    nameless tool message to agent.load_history.

    Mutation this catches: falling back to a fabricated name or ChatMessage(role=TOOL, name=None).
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw_msgs = [
        {
            "role": "tool",
            "content": "orphaned result with no tool identity",
        }
    ]

    with pytest.raises(SessionHistoryRehydrationError) as exc_info:
        mgr.reconstruct_history(raw_msgs, session_id="damaged_sess")

    err_text = str(exc_info.value)
    assert "damaged_sess" in err_text
    assert "tool identity" in err_text
    # The refusal says what it refuses to do, in words. It used to be pinned to the string
    # "P6", which is how the principle number survived into a message the head renders
    # (#1027): `f"Error: {exc}"` becomes the turn's `response`, and the conversation shows it.
    assert "fabricating a tool name" in err_text
    assert not PRINCIPLE_NUMBER.search(err_text)


def test_rehydration_tool_message_blank_name_fails_loudly(tmp_path: Path) -> None:
    """A TOOL message with an empty or whitespace name also fails loudly.

    Mutation this catches: checking only 'if raw_name is not None:' without stripping blank strings.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    for blank_name in ("", "   ", "\t\n"):
        raw_msgs = [
            {
                "role": "tool",
                "name": blank_name,
                "content": "result",
            }
        ]
        with pytest.raises(SessionHistoryRehydrationError) as exc_info:
            mgr.reconstruct_history(raw_msgs, session_id="blank_name_sess")
        assert "lacks tool identity" in str(exc_info.value)


def test_rehydration_distinguishes_content_none_from_empty_string(tmp_path: Path) -> None:
    """Rehydration cleanly distinguishes content=None from content="".

    - 'content' absent or None -> ChatMessage.content is None
    - 'content'="" -> ChatMessage.content == ""

    Mutation this catches: 'content = item_dict.get("content") or ""' which coerces None to "".
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw_msgs = [
        # Explicit None
        {"role": "user", "content": None},
        # Absent content field
        {"role": "assistant", "tool_calls": [{"id": "c1", "name": "run", "arguments": {}}]},
        # Genuinely empty string content
        {"role": "assistant", "content": ""},
        # Tool message with genuinely empty result
        {"role": "tool", "name": "run", "tool_call_id": "c1", "content": ""},
        # Tool message with explicit None content
        {"role": "tool", "name": "run", "tool_call_id": "c1", "content": None},
    ]

    chat_msgs, _ = mgr.reconstruct_history(raw_msgs, session_id="content_distinction_sess")

    assert len(chat_msgs) == 5
    assert chat_msgs[0].content is None
    assert chat_msgs[1].content is None
    assert chat_msgs[2].content == ""
    assert chat_msgs[3].content == ""
    assert chat_msgs[4].content is None


def test_rehydration_preserves_compaction_ledger(tmp_path: Path) -> None:
    """Context compaction ledger system messages are preserved rather than discarded.

    Initial system prompt is prepended, while subsequent compaction ledgers in the transcript
    are restored as ChatMessage(role=SYSTEM, compaction_ledger=True).

    Mutation this catches: continuing unconditionally on role == 'system'.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw_msgs = [
        {
            "role": "system",
            "content": "[Context Auto-Compacted Summary: Heuristic turn 1..5]",
            "compaction_ledger": True,
        },
        {"role": "user", "content": "What was the summary?"},
    ]
    chat_msgs, _ = mgr.reconstruct_history(
        raw_msgs, system_prompt="You are an assistant.", session_id="test_sess"
    )

    assert len(chat_msgs) == 3
    assert chat_msgs[0].role == MessageRole.SYSTEM
    assert chat_msgs[0].content == "You are an assistant."
    assert not chat_msgs[0].compaction_ledger

    assert chat_msgs[1].role == MessageRole.SYSTEM
    assert chat_msgs[1].content == "[Context Auto-Compacted Summary: Heuristic turn 1..5]"
    assert chat_msgs[1].compaction_ledger

    assert chat_msgs[2].role == MessageRole.USER
    assert chat_msgs[2].content == "What was the summary?"


@pytest.mark.asyncio
async def test_get_or_create_agent_legacy_rehydration_end_to_end(tmp_path: Path) -> None:
    """Legacy UI transcript rehydrates into live BaseAgent with tool identity intact.

    Exercises the legacy fallback path in AgentSessionManager.get_or_create_agent when
    no Core store record exists on disk.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    transcript_path = mgr.get_session_path("sess_legacy_tool")
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    stored_transcript = {
        "session_id": "sess_legacy_tool",
        "agent_id": "test_agent",
        "created_at": "2026-09-04T00:00:00Z",
        "updated_at": "2026-09-04T00:01:00Z",
        "turns": 1,
        "messages": [
            {"role": "user", "content": "Check data"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_101", "name": "fetch_metric", "arguments": {"m": "cpu"}}],
            },
            {
                "role": "tool",
                "name": "fetch_metric",
                "tool_call_id": "tc_101",
                "content": "cpu=42%",
            },
            {"role": "assistant", "content": "CPU usage is 42%."},
        ],
    }
    transcript_path.write_text(json.dumps(stored_transcript), encoding="utf-8")

    agent = await mgr.get_or_create_agent(
        agent_id="test_agent",
        session_id="sess_legacy_tool",
        system_prompt="Test system prompt",
    )

    live_messages = agent.get_session("sess_legacy_tool").messages
    # system prompt (0) + 4 transcript messages = 5 messages total
    assert len(live_messages) == 5
    assert live_messages[0].role == MessageRole.SYSTEM
    assert live_messages[1].role == MessageRole.USER
    assert live_messages[1].content == "Check data"

    assert live_messages[2].role == MessageRole.ASSISTANT
    assert len(live_messages[2].tool_calls) == 1
    assert live_messages[2].tool_calls[0].name == "fetch_metric"
    assert live_messages[2].tool_calls[0].id == "tc_101"

    assert live_messages[3].role == MessageRole.TOOL
    assert live_messages[3].name == "fetch_metric"
    assert live_messages[3].tool_call_id == "tc_101"
    assert live_messages[3].content == "cpu=42%"

    assert live_messages[4].role == MessageRole.ASSISTANT
    assert live_messages[4].content == "CPU usage is 42%."

    cached = mgr._session_messages.get("sess_legacy_tool")
    assert cached is not None
    assert len(cached) == 4


@pytest.mark.asyncio
async def test_get_or_create_agent_legacy_rehydration_refuses_nameless_tool(
    tmp_path: Path,
) -> None:
    """Legacy UI transcript with unrepairable nameless tool message fails fast on get_or_create_agent.

    Verifies that corrupt legacy sessions are refused loudly rather than silently loaded.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    transcript_path = mgr.get_session_path("sess_corrupt_tool")
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    stored_transcript = {
        "session_id": "sess_corrupt_tool",
        "agent_id": "test_agent",
        "turns": 1,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "content": "unattributed result"},
        ],
    }
    transcript_path.write_text(json.dumps(stored_transcript), encoding="utf-8")

    with pytest.raises(SessionHistoryRehydrationError) as exc_info:
        await mgr.get_or_create_agent(
            agent_id="test_agent",
            session_id="sess_corrupt_tool",
        )

    assert "sess_corrupt_tool" in str(exc_info.value)
    assert "lacks tool identity" in str(exc_info.value)


def test_get_session_history_preserves_tool_identity_from_core_store(tmp_path: Path) -> None:
    """AgentSessionManager.get_session_history preserves tool identity and content fidelity.

    When synthesizing UI presentation messages from Core store's SessionState,
    name, tool_call_id, and tool_calls must be preserved on the message dicts,
    and content=None must not be coerced to empty string.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)

    # Hydrate Core store directly with a conversation containing tool calls
    from uclone_x.agent.session import SessionState

    core_state = SessionState(
        session_id="sess_core_tools",
        agent_id="agent_core",
        turn_counter=2,
        messages=(
            ChatMessage(role=MessageRole.USER, content="Run check"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(
                    ToolCallRequest(id="call_99", name="disk_check", arguments={"dev": "sda"}),
                ),
            ),
            ChatMessage(
                role=MessageRole.TOOL,
                name="disk_check",
                tool_call_id="call_99",
                content="disk ok",
            ),
            ChatMessage(role=MessageRole.ASSISTANT, content="All clear."),
        ),
    )
    mgr.core_store.save(core_state)

    # Ensure in-memory cache and UI transcript do not exist so get_session_history synthesizes
    history = mgr.get_session_history(agent_id="agent_core", session_id="sess_core_tools")

    assert len(history) == 4
    # User message
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Run check"

    # Assistant message with tool_calls and content=None preserved
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] is None
    assert "tool_calls" in history[1]
    assert history[1]["tool_calls"] == [
        {"id": "call_99", "name": "disk_check", "arguments": {"dev": "sda"}}
    ]

    # Tool message with name and tool_call_id preserved
    assert history[2]["role"] == "tool"
    assert history[2]["name"] == "disk_check"
    assert history[2]["tool_call_id"] == "call_99"
    assert history[2]["content"] == "disk ok"

    # Final assistant message
    assert history[3]["role"] == "assistant"
    assert history[3]["content"] == "All clear."


def test_translate_session_error_maps_rehydration_error_to_422() -> None:
    """_translate_session_error maps SessionHistoryRehydrationError to 422 Unprocessable Entity."""
    exc = SessionHistoryRehydrationError("Corrupt history record")
    http_exc = _translate_session_error(exc)
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 422
    assert "Corrupt history record" in http_exc.detail


def test_rehydration_does_not_mutate_raw_input_and_marks_inferred_name(tmp_path: Path) -> None:
    """reconstruct_history does not mutate input dicts in place and marks inferred names.

    Principle 6 and Issue #390 require:
    - Raw input dictionaries are never mutated by reconstruct_history.
    - typed[i] is a distinct object from raw[i] (typed[i] is not raw[i]).
    - When a tool name is inferred from preceding tool calls, it is explicitly marked
      as inferred on the typed transcript dict ("name_inferred": True).
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)
    raw_msgs: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "Running query",
            "tool_calls": [{"id": "call_q1", "name": "sql_query", "arguments": {}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_q1",
            "content": "1 row returned",
        },
    ]

    # Preserve reference to the raw second message
    raw_tool_msg = raw_msgs[1]
    assert "name" not in raw_tool_msg
    assert "name_inferred" not in raw_tool_msg

    chat_msgs, typed_msgs = mgr.reconstruct_history(raw_msgs, session_id="test_sess")

    # Assert object identity separation: typed dicts are copies, not the original input dicts
    assert len(typed_msgs) == 2
    assert typed_msgs[0] is not raw_msgs[0]
    assert typed_msgs[1] is not raw_msgs[1]

    # Assert raw input dictionary was NOT mutated in place
    assert "name" not in raw_tool_msg
    assert "name_inferred" not in raw_tool_msg
    assert "name" not in raw_msgs[1]
    assert "name_inferred" not in raw_msgs[1]

    # Assert typed session message holds the repaired name and explicitly marks it inferred
    assert typed_msgs[1]["name"] == "sql_query"
    assert typed_msgs[1]["name_inferred"] is True

    # Assert ChatMessage history correctly hydrated the tool name
    assert len(chat_msgs) == 2
    assert chat_msgs[1].role == MessageRole.TOOL
    assert chat_msgs[1].name == "sql_query"
    assert chat_msgs[1].tool_call_id == "call_q1"


def test_rehydration_unrecognized_role_fails_loudly(tmp_path: Path) -> None:
    """An unrecognized role string fails loudly rather than coercing to MessageRole.ASSISTANT.

    Prior to Issue #390, a ValueError when constructing MessageRole fell back to
    MessageRole.ASSISTANT. Principle 6 forbids silently coercing unrecognized message
    roles into assistant utterances.
    """
    mgr = AgentSessionManager(storage_dir=tmp_path)

    for invalid_role in ("moderator", "bot", "bogus", "unknown", "arbitrary_role"):
        raw_msgs = [
            {
                "role": invalid_role,
                "content": "some message content",
            }
        ]
        with pytest.raises(SessionHistoryRehydrationError) as exc_info:
            mgr.reconstruct_history(raw_msgs, session_id="invalid_role_sess")

        err_text = str(exc_info.value)
        assert "unrecognized role" in err_text
        assert invalid_role in err_text
        assert "invalid_role_sess" in err_text

    # Missing role with unrecognized sender also fails loudly
    raw_msgs_bad_sender = [
        {
            "sender": "unrecognized_entity",
            "content": "sender without role",
        }
    ]
    with pytest.raises(SessionHistoryRehydrationError) as exc_info:
        mgr.reconstruct_history(raw_msgs_bad_sender, session_id="bad_sender_sess")
    assert "unrecognized sender" in str(exc_info.value)

    # Missing both role and sender fails loudly
    raw_msgs_no_role_or_sender = [
        {
            "content": "no role or sender",
        }
    ]
    with pytest.raises(SessionHistoryRehydrationError) as exc_info:
        mgr.reconstruct_history(raw_msgs_no_role_or_sender, session_id="no_role_sess")
    assert "lacks both 'role' and 'sender'" in str(exc_info.value)


def test_chat_session_load_failure_reports_session_load_error_provenance(tmp_path: Path) -> None:
    """POST /api/turn reports SESSION_LOAD_ERROR provenance when session loading fails.

    When session loading fails (e.g. SessionHistoryRehydrationError due to damaged history),
    provenance.path must NOT read FAILOVER or OFFLINE_FALLBACK (Issue #390, P6).
    """
    storage_dir = tmp_path / "sessions"
    storage_dir.mkdir(parents=True, exist_ok=True)
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir)
    client = TestClient(app)

    # Persist a damaged session transcript containing an unrepairable nameless tool message
    damaged_transcript = {
        "session_id": "sess_damaged_load",
        "agent_id": "test-agent",
        "turns": 1,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "orphaned output with no name"},
        ],
    }
    transcript_file = storage_dir / "sess_damaged_load.json"
    transcript_file.write_text(json.dumps(damaged_transcript), encoding="utf-8")

    response = client.post(
        "/api/turn",
        json={
            "message": "Continue work",
            "agent_id": "test-agent",
            "session_id": "sess_damaged_load",
        },
    )
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())

    # Verify status is error
    assert data["status"] == "error"
    assert "Error:" in data["response"]

    # Verify provenance path is SESSION_LOAD_ERROR, NEVER FAILOVER or OFFLINE_FALLBACK
    prov = data["provenance"]
    assert prov["path"] == "SESSION_LOAD_ERROR"
    assert prov["path"] != "FAILOVER"
    assert prov["path"] != "OFFLINE_FALLBACK"
    assert prov["component"] == "uclone_x.agent.session"
    assert prov["degraded"] is True

    # Verify durability diagnostic records the rehydration error type
    durability = data["durability"]
    assert durability["persisted"] is False
    assert durability["error_type"] == "SessionHistoryRehydrationError"
    assert "lacks tool identity" in durability["error"]


def test_get_session_history_damaged_record_returns_422(tmp_path: Path) -> None:
    """GET /api/session/history refuses damaged transcript records with HTTP 422.

    When reading persisted session history containing damaged records (unrepairable tool,
    invalid role), GET /api/session/history validates history through reconstruct_history
    and translates SessionHistoryRehydrationError into 422 Unprocessable Entity (Issue #390).

    The body is copy: it reaches the conversation through the turn's `response` and a client
    through this 422. It names no principle number (#1027).

    Killed by: src/uclone_x/ui/app.py :: "message to the model."
    Becomes: "message to the model (P6)."
    """
    storage_dir = tmp_path / "sessions"
    storage_dir.mkdir(parents=True, exist_ok=True)
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir)
    client = TestClient(app)

    # 1. Persist a session transcript with an unrepairable nameless tool message
    damaged_transcript = {
        "session_id": "sess_damaged_tool",
        "agent_id": "agent-orchestrator",
        "messages": [
            {"role": "user", "content": "check"},
            {"role": "tool", "content": "damaged output without tool name"},
        ],
    }
    (storage_dir / "sess_damaged_tool.json").write_text(
        json.dumps(damaged_transcript), encoding="utf-8"
    )

    resp_tool = client.get(
        "/api/session/history?agent_id=agent-orchestrator&session_id=sess_damaged_tool"
    )
    assert resp_tool.status_code == 422
    tool_detail = resp_tool.json()["detail"]
    assert "lacks tool identity" in tool_detail
    assert "fabricating a tool name" in tool_detail
    assert not PRINCIPLE_NUMBER.search(tool_detail)

    # 2. Persist a session transcript with an unrecognized role string
    invalid_role_transcript = {
        "session_id": "sess_damaged_role",
        "agent_id": "agent-orchestrator",
        "messages": [
            {"role": "rogue_role", "content": "invalid role message"},
        ],
    }
    (storage_dir / "sess_damaged_role.json").write_text(
        json.dumps(invalid_role_transcript), encoding="utf-8"
    )

    resp_role = client.get(
        "/api/session/history?agent_id=agent-orchestrator&session_id=sess_damaged_role"
    )
    assert resp_role.status_code == 422
    role_detail = resp_role.json()["detail"]
    assert "unrecognized role" in role_detail
    assert "rogue_role" in role_detail


def test_chat_with_an_unusable_agent_name_does_not_blame_the_session_store(
    tmp_path: Path,
) -> None:
    """A name the request supplied is not a fault of `uclone_x.agent.session`.

    `AgentHomeError` is raised while the agent's home directory is resolved, inside the
    broad handler that wraps `get_or_create_agent` -- so it inherited `session_load_failed`
    and was recorded as `SESSION_LOAD_ERROR` with `component=uclone_x.agent.session`. That
    record tells whoever reads it to go and look at the session store, which holds nothing
    to do with the name and has nothing to fix: the mis-attribution P6 forbids (#539 is
    the same bug for the LLM provider, one branch above).

    Killed by: src/uclone_x/ui/app.py :: session_load_failed and not is_offline and not isinstance(exc, AgentHomeError)
    Becomes: session_load_failed and not is_offline
    """
    storage_dir = tmp_path / "sessions"
    storage_dir.mkdir(parents=True, exist_ok=True)
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir)
    client = TestClient(app)

    response = client.post(
        "/api/turn",
        json={"message": "hello", "agent_id": "Champion", "session_id": "sess_bad_name"},
    )

    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["status"] == "error"
    assert "Champion" in data["response"], "the caller cannot fix a name the reply withholds"

    prov = data["provenance"]
    assert prov["component"] != "uclone_x.agent.session", (
        "the session store did not refuse this name and has nothing to fix"
    )
    assert prov["path"] != "SESSION_LOAD_ERROR"
