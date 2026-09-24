# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
"""Unit tests for tool execution data on the session transcript (Issue #450).

Validates that tool-only turns (empty content with tool executions):
- Retain empty assistant content without generic fallback pollution in session manager and API endpoints.
- Return serialized tool executions with arguments, outputs, and status.

**The rendering half of #450 is gone, and these are what is left of it.** A third test here
asserted that the built `ui_static` bundle contained `tool-execution-summary`,
`inline-tool-execution`, `tool-output-result` and `tool-arguments`. Those testids existed
only in `frontend/src/components/PlaygroundTab.tsx`, which #1208 deletes on the owner's
§3.2.5 [Rev 23] ruling, so the assertion had no subject left and went with it (together with
`tests/e2e/test_ui_tool_execution_e2e.py`, which drove the same cards).

It is not replaced, and nothing below should be read as replacing it. The room transcript
renders no tool executions at all, and it cannot be made to without a Core change:
`RoomMessage` excludes tool calls and results *deliberately* (`room/models.py`) because a
seat's tool traces belong to that seat's own session. So the wire still carries the data —
that is precisely what the tests below pin — and no surface draws it. That gap is a
follow-up issue, not something #1208 delivered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from uclone_x.ui.app import AgentSessionManager, create_ui_app


def test_ui_session_history_preserves_tool_only_turn(tmp_path: Path) -> None:
    """Verify session storage and history API faithfully preserve empty content for tool-only turns."""
    mgr = AgentSessionManager(storage_dir=tmp_path)
    session_id = "sess_tool_only_test"

    tool_exec_data = [
        {
            "tool_call_id": "call_web_search_0",
            "tool_name": "web_search",
            "arguments": {"query": "current tech news", "max_results": 5},
            "output": {"results": ["Article A", "Article B"]},
            "status": "success",
            "duration_ms": 142.5,
        }
    ]

    # Save a tool-only turn directly as stored in real sessions
    raw_messages = [
        {
            "id": "msg-user-1",
            "sender": "user",
            "role": "user",
            "content": "Search tech news",
            "timestamp": "10:00:00 AM",
        },
        {
            "id": "msg-agent-1",
            "sender": "agent",
            "role": "assistant",
            "content": "",
            "timestamp": "10:00:01 AM",
            "tool_executions": tool_exec_data,
            "tool_calls": [
                {
                    "id": "call_web_search_0",
                    "name": "web_search",
                    "arguments": {"query": "current tech news", "max_results": 5},
                }
            ],
            "latency_ms": 142.5,
            "tokens_used": 85,
        },
    ]

    mgr.save_session_record(
        session_id=session_id,
        agent_id="agent-general",
        messages=raw_messages,
    )

    # Re-read session record via manager
    loaded_rec = mgr.load_session_record(session_id=session_id)
    assert loaded_rec is not None
    loaded_msgs = loaded_rec.get("messages", [])
    assert len(loaded_msgs) == 2
    assistant_msg = loaded_msgs[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == ""
    assert "tool_executions" in assistant_msg
    assert len(assistant_msg["tool_executions"]) == 1
    assert assistant_msg["tool_executions"][0]["tool_name"] == "web_search"
    assert assistant_msg["tool_executions"][0]["output"] == {"results": ["Article A", "Article B"]}


def test_ui_history_endpoint_returns_tool_executions(tmp_path: Path) -> None:
    """Verify /api/session/history returns tool executions and preserves empty content."""
    mgr = AgentSessionManager(storage_dir=tmp_path)
    session_id = "sess_api_history_tool"

    raw_messages = [
        {
            "id": "msg-u",
            "sender": "user",
            "role": "user",
            "content": "Execute tool",
        },
        {
            "id": "msg-a",
            "sender": "agent",
            "role": "assistant",
            "content": "",
            "tool_executions": [
                {
                    "tool_call_id": "call_1",
                    "tool_name": "calculator",
                    "arguments": {"expression": "40 + 2"},
                    "output": 42,
                    "status": "success",
                    "duration_ms": 12.0,
                }
            ],
        },
    ]
    mgr.save_session_record(
        session_id=session_id,
        agent_id="agent-general",
        messages=raw_messages,
    )

    app = create_ui_app(static_dir=tmp_path, storage_dir=tmp_path)
    client = TestClient(app)

    res = client.get(
        "/api/session/history",
        params={"agent_id": "agent-general", "session_id": session_id},
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert "messages" in data
    assert len(data["messages"]) == 2
    agent_msg = data["messages"][1]
    assert agent_msg["role"] == "assistant"
    assert agent_msg["content"] == ""
    assert "tool_executions" in agent_msg
    assert len(agent_msg["tool_executions"]) == 1
    assert agent_msg["tool_executions"][0]["tool_name"] == "calculator"
    assert agent_msg["tool_executions"][0]["output"] == 42


# ======================================================================================
# Issue #665 (follow-up) — nested tool-call arguments must survive the trip to disk
#
# Both UI sites fed `save_session_record`, whose `json.dumps` raises on a nested
# `MappingProxyType`. Both raised where a caller could see it — measured by reverting
# each line (#673), correcting an earlier claim here that the failure was silent:
# the truncate path lets the `TypeError` out of `truncate_session_history`, which sits
# in no `try`, and the turn path dies in FastAPI's response serialiser
# (`PydanticSerializationError`) because the same list is in the response body, before
# the endpoint's `except Exception: logger.warning(...)` around the save is reached.
#
# The assertions are on the persisted record anyway, and deliberately so: it is the
# claim the tests are actually making (the transcript is on disk with its nested
# arguments intact), and it survives a future refactor that stops putting
# `serialized_tool_calls` in the response body — at which point the `except Exception`
# around the save *would* make this quiet, and a status-code assertion would stop
# pinning anything.
# ======================================================================================


_NESTED_ARGUMENTS: dict[str, Any] = {
    "path": "README.md",
    "options": {"encoding": "utf-8", "limits": {"max_bytes": 4096}},
    "ranges": [{"start": 1, "end": 20}],
}


def _only_tool_call(record: dict[str, Any]) -> dict[str, Any]:
    """The single tool call in a persisted transcript record."""
    messages = cast(list[dict[str, Any]], record["messages"])
    calls = [
        tc for msg in messages for tc in cast(list[dict[str, Any]], msg.get("tool_calls") or [])
    ]
    assert len(calls) == 1, f"expected one persisted tool call, got {calls}"
    return calls[0]


def test_ui_truncate_persists_nested_tool_arguments_from_core_store(tmp_path: Path) -> None:
    """History synthesised from the Core store keeps nested tool arguments on disk.

    `truncate_session_history` synthesises the UI transcript with `get_session_history`
    and hands it straight to `save_session_record`. With the shallow `dict(tc.arguments)`
    the nested objects were still `MappingProxyType` when `json.dumps` saw them.

    Killed by: src/uclone_x/ui/app.py :: "arguments": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
    """
    from uclone_x.agent.session import SessionState
    from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest

    mgr = AgentSessionManager(storage_dir=tmp_path)
    session_id = "sess_nested_truncate"
    mgr.core_store.save(
        SessionState(
            session_id=session_id,
            agent_id="agent-general",
            turn_counter=1,
            messages=(
                ChatMessage(role=MessageRole.USER, content="Read the file"),
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=(
                        ToolCallRequest(
                            id="call_nested", name="read_file", arguments=_NESTED_ARGUMENTS
                        ),
                    ),
                ),
                ChatMessage(
                    role=MessageRole.TOOL,
                    name="read_file",
                    tool_call_id="call_nested",
                    content="file contents",
                ),
                ChatMessage(role=MessageRole.ASSISTANT, content="Done."),
            ),
        )
    )

    mgr.truncate_session_history(agent_id="agent-general", session_id=session_id, index=3)

    record = mgr.load_session_record(session_id=session_id)
    assert record is not None, "transcript never reached disk"
    assert _only_tool_call(record)["arguments"] == _NESTED_ARGUMENTS


@pytest.mark.usefixtures("builtin_personas_absent")
def test_ui_chat_turn_persists_nested_tool_arguments(tmp_path: Path) -> None:
    """A completed turn's tool call reaches disk with its nested arguments intact.

    `save_session_record` is called from `chat_with_agent` inside
    `except Exception: logger.warning("Failed to save transcript ...")`. An earlier
    version of this docstring concluded from that alone that the broken code "answered
    the request with a normal 200 and a correct-looking body while writing nothing", and
    that "asserting on the response would prove nothing". Both are wrong, and the #673
    review found it by reverting the line rather than by reading the code: the same
    `serialized_tool_calls` list goes into the **response body**, so the request dies in
    FastAPI's response serialiser with
    `PydanticSerializationError: Unable to serialize unknown type: <class 'mappingproxy'>`
    and never reaches a 200. `assert res.status_code == 200` below does have teeth.

    The assertion on the persisted file is kept as the primary one regardless, because
    it is the claim this test makes and it does not depend on `serialized_tool_calls`
    happening to be in the response body — a refactor that removed it from there would
    put this back inside the swallowed save, where only the file assertion still holds.

    The same turn also publishes a `TOOL_CALL` event whose payload is `ImmutableMapping`.
    That site was cleared as benign during review ("pydantic re-freezes it"); it is not —
    `JsonValue` validation runs before `AfterValidator(freeze_mapping)`, so the field
    *rejects* a nested `MappingProxyType`. `status == "success"` is what pins it: the
    `ValidationError` propagated out of the endpoint.

    Killed by: src/uclone_x/ui/app.py :: "arguments": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
    Killed by: src/uclone_x/ui/app.py :: tool_arguments = cast(dict[str, Any], unwrap_immutable(tc.arguments))
    """
    from uclone_x.core.provenance import Provenance
    from uclone_x.llm import MockLLMConnector
    from uclone_x.llm.models import ToolCallRequest
    from uclone_x.tools import LocalTool, ToolRegistry
    from uclone_x.tools.models import ToolContext, ToolResult

    class ReadFileTool(LocalTool):
        """Echoes its parameters back, so the turn completes with a real execution."""

        def __init__(self) -> None:
            super().__init__(name="read_file", description="Reads a file")

        async def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
            return ToolResult(
                success=True, output=params, provenance=Provenance.primary("read_file")
            )

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    llm = MockLLMConnector(
        responses=["Reading the file.", "Here is the file."],
        tool_calls=[
            ToolCallRequest(id="call_nested", name="read_file", arguments=_NESTED_ARGUMENTS)
        ],
    )

    app = create_ui_app(static_dir=tmp_path, llm=llm, tools=registry, storage_dir=tmp_path)
    client = TestClient(app)
    session_id = "sess_nested_turn"

    res = client.post(
        "/api/turn",
        json={
            "message": "Read README.md",
            "agent_id": "agent-general",
            "session_id": session_id,
        },
    )
    assert res.status_code == 200
    body = cast(dict[str, Any], res.json())
    assert body["status"] == "success", body
    assert len(body["tool_calls"]) == 1

    # The claim: the turn is on disk, not merely in the response body.
    mgr = AgentSessionManager(storage_dir=tmp_path)
    record = mgr.load_session_record(session_id=session_id)
    assert record is not None, (
        "the turn's transcript was dropped from disk — `save_session_record` failed and "
        "the endpoint swallowed it"
    )
    assert _only_tool_call(record)["arguments"] == _NESTED_ARGUMENTS
