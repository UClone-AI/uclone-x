# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
"""Unit tests for the developer UI backend, API endpoints, and server launcher."""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.support.vite_diagnosis import answering_as, one_line
from uclone_x import __version__
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentState, TurnResult
from uclone_x.cli import main
from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    Provenance,
    ServiceRef,
)
from uclone_x.engine.event_bus import EventBus, EventType
from uclone_x.errors import (
    LLMCredentialsNotConfiguredError,
    LLMProviderError,
    LLMTimeoutError,
    StaleSessionWriteError,
)
from uclone_x.llm import MockLLMConnector, create_llm_connector
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.connectors.vllm import VLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.models import (
    OntologyAxiom,
    OntologyConcept,
    OntologyRelation,
    OntologyTier,
)
from uclone_x.sandbox.models import IsolationLevel
from uclone_x.shells import ui_process
from uclone_x.skills.auditor import Skill, SkillRegistry
from uclone_x.skills.models import (
    AuditVerdict,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.tools.models import ToolResult
from uclone_x.tools.protocols import ToolProtocol
from uclone_x.tools.registry import ToolRegistry, create_default_registry
from uclone_x.ui.app import (
    AgentSessionManager,
    create_ui_app,
    fetch_available_models,
    get_ui_session_manager,
    vllm_request_headers,
)
from uclone_x.ui.server import VITE_IDENTITY_MARKER, start_ui_server

runner = CliRunner()


def stubbed_ollama_connector(reply: str = "Stubbed connector reply.") -> OllamaConnector:
    """A real `OllamaConnector` with only its socket replaced (#212).

    Injecting a `MagicMock(spec=LLMProviderProtocol)` would also make a chat test
    deterministic, but it would stop the connector's own code from running: the
    `Provenance` such a test asserts on would be one the test hand-wrote, so the test
    could no longer distinguish "the UI forwarded the connector's attribution" from
    "the UI synthesised an attribution that happens to match". Stubbing one layer
    lower — at `BaseLLMConnector`'s `http_client` seam, with `httpx.MockTransport` —
    keeps every line of `ollama.py`'s response mapping, including its
    `Provenance.primary(...)` call, on the path under test. Only the network is gone.

    The handler echoes the model name back out of the request payload, which is what a
    live Ollama does and which makes the stub independent of `OLLAMA_MODEL` in the
    environment. A consequence worth naming (swarm guide §6.9, "the coincidence case"):
    because `requested` and `served_by` are then equal, this stub cannot distinguish a
    correct `served_model=` mapping from the `model=served` defect repaired in #149.
    That alias case is pinned separately in `tests/unit/test_llm_connectors.py`; do not
    read provenance assertions made through this helper as covering it.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        payload: dict[str, Any] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "created_at": "2026-09-03T00:00:00.000000Z",
                "message": {"role": "assistant", "content": reply},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 5,
            },
        )

    return OllamaConnector(
        base_url="http://stub-ollama.invalid:11434",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture
def test_client(tmp_path: Path) -> TestClient:
    """Create a test client with a dummy static directory and MockLLMConnector."""
    static_dir = tmp_path / "ui_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>Test UI</body></html>", encoding="utf-8")
    storage_dir = tmp_path / "sessions"
    mock_llm = MockLLMConnector()
    app = create_ui_app(static_dir=static_dir, storage_dir=storage_dir, llm=mock_llm)
    return TestClient(app)


def test_ui_health_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/api/health")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["status"] == "healthy"
    assert data["version"] == __version__
    assert data["runtime"] == "uclone_x"
    assert "absorbed_failures" in data
    assert "dropped_spans" in data["absorbed_failures"]
    assert "event_bus_drops" in data["absorbed_failures"]
    assert "agent_processing_errors" in data["absorbed_failures"]


def test_ui_diagnostics_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/api/diagnostics")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["runtime"] == "uclone_x"
    assert "absorbed_failures" in data
    assert "event_bus" in data


def test_ui_agents_endpoint_empty(test_client: TestClient) -> None:
    """Assert /api/agents returns truthful empty topology when unpopulated (P6, P8)."""
    response = test_client.get("/api/agents")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["data_source"] == "live"
    assert data["agents"] == []
    assert data["topology"]["nodes"] == []
    assert data["topology"]["edges"] == []


@pytest.mark.asyncio
async def test_ui_agents_endpoint_populated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert /api/agents reflects live registered BaseAgent instances in session manager."""
    # This test builds an agent but is not about provider resolution, so it says which
    # connector it wants. `conftest` clears LLM configuration rather than pinning a
    # provider, so an unconfigured build is now refused (#533) instead of silently
    # producing one.
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions")
    app = create_ui_app(static_dir=tmp_path, session_manager=session_mgr)
    client = TestClient(app)

    # Initially empty
    res1 = client.get("/api/agents")
    assert res1.status_code == 200
    assert res1.json()["agents"] == []

    # Spawn root agent and subagent
    _root_agent = await session_mgr.get_or_create_agent(
        agent_id="champion",
        session_id="sess-1",
    )
    sub_agent = await session_mgr.get_or_create_agent(
        agent_id="agent-worker-1",
        session_id="sess-1",
    )
    # Configure parent relationship
    sub_agent._context = sub_agent.context.model_copy(update={"parent_agent_id": "champion"})

    res2 = client.get("/api/agents")
    assert res2.status_code == 200
    data = cast(dict[str, Any], res2.json())
    assert data["data_source"] == "live"
    assert len(data["agents"]) == 2
    agent_ids = {a["id"] for a in data["agents"]}
    assert agent_ids == {"champion", "agent-worker-1"}

    # Verify topology nodes and edges
    nodes = data["topology"]["nodes"]
    assert len(nodes) == 2
    edges = data["topology"]["edges"]
    assert len(edges) == 1
    assert edges[0]["source"] == "champion"
    assert edges[0]["target"] == "agent-worker-1"

    await session_mgr.clear()


@pytest.mark.asyncio
async def test_ui_agents_endpoint_deduplicates_across_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert /api/agents deduplicates agents with identical agent_id across sessions (#869)."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions")
    app = create_ui_app(static_dir=tmp_path, session_manager=session_mgr)
    client = TestClient(app)

    # Spawn root agent in session 1
    await session_mgr.get_or_create_agent(
        agent_id="champion",
        session_id="sess-1",
    )
    # Spawn root agent in session 2 (different instance, same agent_id)
    await session_mgr.get_or_create_agent(
        agent_id="champion",
        session_id="sess-2",
    )

    # Calling /api/agents without session_id returns deduplicated agent list (exactly 1 champion)
    res = client.get("/api/agents")
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert len(data["agents"]) == 1
    assert data["agents"][0]["id"] == "champion"
    assert len(data["topology"]["nodes"]) == 1

    # Calling /api/agents with session_id filters to that session
    res_sess1 = client.get("/api/agents?session_id=sess-1")
    assert res_sess1.status_code == 200
    data_sess1 = cast(dict[str, Any], res_sess1.json())
    assert len(data_sess1["agents"]) == 1
    assert data_sess1["agents"][0]["id"] == "champion"

    await session_mgr.clear()


def test_ui_chat_endpoint(test_client: TestClient) -> None:
    # Test valid chat request
    response = test_client.post(
        "/api/turn",
        json={"message": "Hello UClone-X", "agent_id": "champion"},
    )
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["status"] == "success"
    assert "Hello UClone-X" in data["response"]
    assert data["agent_id"] == "champion"
    assert "latency_ms" in data
    assert "tool_calls" in data
    assert data["turn_budget_max"] == 50
    assert data["turns_remaining"] == 49, "one answered request costs one step of the run"
    assert data["step_budget_max"] == 50
    assert data["steps_remaining"] == 49
    assert data["run_steps"] == 1
    assert "provenance" in data

    # Test empty chat request
    err_res = test_client.post("/api/turn", json={"message": "   ", "agent_id": "champion"})
    assert err_res.status_code == 200
    assert "error" in err_res.json()


def test_ui_chat_multi_turn_persistence(test_client: TestClient) -> None:
    # First turn
    res1 = test_client.post(
        "/api/turn",
        json={"message": "First message", "agent_id": "agent-multi-turn"},
    )
    assert res1.status_code == 200
    data1 = cast(dict[str, Any], res1.json())
    assert data1["status"] == "success"
    assert data1["turn_count"] == 1
    assert data1["turn_budget_max"] == 50
    assert data1["turns_remaining"] == 49, "one answered request costs one step of the run"
    assert "First message" in data1["response"]

    # Second turn with same agent
    res2 = test_client.post(
        "/api/turn",
        json={"message": "Second message", "agent_id": "agent-multi-turn"},
    )
    assert res2.status_code == 200
    data2 = cast(dict[str, Any], res2.json())
    assert data2["status"] == "success"
    assert data2["turn_count"] == 2
    assert data2["turn_budget_max"] == 50
    assert data2["turns_remaining"] == 49, "the next request starts the run over"
    assert "Second message" in data2["response"]


def test_ui_turn_endpoint_reports_dynamic_turn_budget(test_client: TestClient) -> None:
    """Verify /api/turn reports the dynamic turn budget and decrements what remains.

    The second turn was sent through `/api/chat/stream` until #1208 retired it. Both
    arms ran the same `_execute_turn_logic_impl`, so what the stream arm added was a
    second reading of one figure, not a second behaviour; the turn counter across two
    requests is pinned here.
    """
    agent_id = "agent-budget-stream"

    res1 = test_client.post(
        "/api/turn",
        json={"message": "Chat endpoint turn 1", "agent_id": agent_id},
    )
    assert res1.status_code == 200
    data1 = cast(dict[str, Any], res1.json())
    assert data1["turn_budget_max"] == 50
    assert data1["turn_count"] == 1
    assert data1["turns_remaining"] == 49, "one answered request costs one step of the run"

    res2 = test_client.post(
        "/api/turn",
        json={"message": "Turn 2", "agent_id": agent_id},
    )
    assert res2.status_code == 200
    data2 = cast(dict[str, Any], res2.json())
    assert data2["turn_budget_max"] == 50
    assert data2["turn_count"] == 2
    assert data2["turns_remaining"] == 49, "the next request starts the run over"


@pytest.mark.asyncio
async def test_ui_chat_custom_agent_max_turns_budget(tmp_path: Path) -> None:
    """Verify /api/turn dynamically reports a custom `AgentConfig.max_turns`.

    Turn 2 went through `/api/chat/stream` until #1208 retired it; the budget is read
    off the same turn result either way.
    """
    from uclone_x.agent import BaseAgent
    from uclone_x.agent.models import AgentConfig, AgentLLMConfig

    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions")
    app = create_ui_app(static_dir=tmp_path, session_manager=session_mgr)
    client = TestClient(app)

    custom_cfg = AgentConfig(
        agent_id="custom-budget-agent",
        name="Custom Budget Agent",
        max_turns=3,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    custom_agent = BaseAgent(config=custom_cfg, llm=MockLLMConnector())
    session_mgr._agents["custom-budget-agent"] = custom_agent

    r1 = client.post(
        "/api/turn",
        json={"message": "Turn 1", "agent_id": "custom-budget-agent"},
    )
    assert r1.status_code == 200
    d1 = cast(dict[str, Any], r1.json())
    assert d1["turn_budget_max"] == 3
    assert d1["turn_count"] == 1
    assert d1["turns_remaining"] == 2

    r2 = client.post(
        "/api/turn",
        json={"message": "Turn 2", "agent_id": "custom-budget-agent"},
    )
    assert r2.status_code == 200
    d2 = cast(dict[str, Any], r2.json())
    assert d2["turn_budget_max"] == 3
    assert d2["turn_count"] == 2
    assert d2["turns_remaining"] == 2

    # Turn 3 via /api/turn (reaches ceiling)
    r3 = client.post(
        "/api/turn",
        json={"message": "Turn 3", "agent_id": "custom-budget-agent"},
    )
    assert r3.status_code == 200
    d3 = cast(dict[str, Any], r3.json())
    assert d3["turn_budget_max"] == 3
    assert d3["turn_count"] == 3
    assert d3["turns_remaining"] == 2

    # Turn 4: past the ceiling in message count, and NOT refused. `max_turns` bounds a
    # self-driven run, not a conversation — this assertion is the defect it replaces,
    # where a person's fourth message was rejected with "Turn budget exceeded".
    r4 = client.post(
        "/api/turn",
        json={"message": "Turn 4", "agent_id": "custom-budget-agent"},
    )
    assert r4.status_code == 200
    d4 = cast(dict[str, Any], r4.json())
    # Not refused is the claim. `status` is "warning" for every turn in this test,
    # including the first three, because the agent is constructed without a
    # `SessionStore` and persistence is therefore unavailable — a property of the
    # fixture, not of the ceiling.
    assert d4["status"] != "error", d4.get("response")
    assert "Turn budget exceeded" not in d4["response"]
    assert d4["turn_budget_max"] == 3
    assert d4["turn_count"] == 4
    assert d4["run_turns"] == 1
    assert d4["run_steps"] == 1
    assert d4["turns_remaining"] == 2
    assert d4["step_budget_max"] == 3
    assert d4["steps_remaining"] == 2

    await session_mgr.clear()


@pytest.mark.asyncio
@pytest.mark.usefixtures("builtin_personas_absent")
async def test_ui_chat_with_tools_execution(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock

    mock_tool = MagicMock(spec=ToolProtocol)
    mock_tool.name = "ast_code_analyzer"
    mock_tool.description = "Analyzes AST"
    mock_tool.parameters_schema = {}
    tool_res = ToolResult(
        output={"status": "clean", "symbols": 42},
        success=True,
        execution_time_ms=8.0,
        isolation_level=IsolationLevel.WORKSPACE,
        provenance=Provenance(
            path=ExecutionPath.PRIMARY,
            requested=ServiceRef(provider="tools", model="ast_code_analyzer"),
            served_by=ServiceRef(provider="tools", model="ast_code_analyzer"),
        ),
    )
    mock_tool.execute = AsyncMock(return_value=tool_res)

    tools = ToolRegistry()
    tools.register(mock_tool)

    tool_call = ToolCallRequest(
        id="tc_123",
        name="ast_code_analyzer",
        arguments={"target": "test.py"},
    )
    mock_llm = MockLLMConnector(
        responses=["Analysis complete with 0 issues."],
        tool_calls=[tool_call],
    )

    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, tools=tools)
    client = TestClient(app)

    res = client.post(
        "/api/turn",
        json={"message": "Analyze test.py", "agent_id": "agent-tool-user"},
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["status"] == "success"
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["name"] == "ast_code_analyzer"
    assert data["tool_calls"][0]["id"] == "tc_123"
    # The reply is the model's answer after seeing its tool results, not the text it
    # emitted alongside the tool call (P4, amended 2026-09-05).
    assert data["response"], "a tool-using turn must still produce an answer"
    assert "Analysis complete" not in data["response"]
    assert mock_tool.execute.called

    # Issue #171: Enriched tool executions and debug info
    assert "tool_executions" in data
    assert len(data["tool_executions"]) == 1
    te = data["tool_executions"][0]
    assert te["tool_name"] == "ast_code_analyzer"
    assert te["tool_call_id"] == "tc_123"
    assert te["arguments"] == {"target": "test.py"}
    assert te["status"] == "success"
    assert te["output"] == {"status": "clean", "symbols": 42}
    assert te["error"] is None
    assert te["duration_ms"] == 8.0

    assert "debug_info" in data
    assert "active_invariants" in data["debug_info"]
    assert "prompt_tokens_used" in data["debug_info"]
    assert "system_prompt_excerpt" in data["debug_info"]


def test_ui_chat_error_handling(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock

    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("Provider failure"))

    app = create_ui_app(static_dir=tmp_path, llm=mock_llm)
    client = TestClient(app)

    res = client.post(
        "/api/turn",
        json={"message": "Crash please", "agent_id": "agent-error-test"},
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["status"] == "error"
    assert "Provider failure" in data["response"]


def test_a_manager_given_no_workspace_does_not_read_the_checkout(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Under pytest the default workspace is this run's own temporary one, not the checkout.

    With the checkout as workspace, a developer's untracked `.uclone/personas/artist.yaml`
    declaring tools the tests do not register failed 8 unit tests that build a manager
    without naming a workspace. Those tests stay green on a clean checkout either way, so
    this is what notices the isolation going.

    Asserted against this run's base temp rather than as "not the cwd": the mutation ratchet
    runs this test in a child pytest that inherits the parent's environment, so with the
    fixture removed the child still sees the parent's `UCLONE_WORKSPACE_DIR` -- a temporary
    directory, but not one of this run's.

    Killed by: tests/conftest.py :: monkeypatch.setenv("UCLONE_WORKSPACE_DIR", str(root))
    Becomes: pass
    """
    manager = AgentSessionManager(bus=EventBus(), llm=MockLLMConnector(), tools=ToolRegistry())

    base = tmp_path_factory.getbasetemp().resolve()
    assert manager.workspace_dir.is_relative_to(base), manager.workspace_dir


@pytest.mark.asyncio
@pytest.mark.usefixtures("builtin_personas_absent")
async def test_agent_session_manager() -> None:
    bus = EventBus()
    mock_llm = MockLLMConnector(default_response="Session Manager Reply")
    tools = ToolRegistry()
    manager = AgentSessionManager(bus=bus, llm=mock_llm, tools=tools)

    assert manager.bus is bus
    assert manager.tools is tools
    assert manager.llm is mock_llm
    assert manager.list_agents() == []

    agent = await manager.get_or_create_agent("test-agent-1", session_id="sess-1")
    assert agent.agent_id == "test-agent-1"
    assert manager.get_agent("test-agent-1") is agent
    assert len(manager.list_agents()) == 1

    # Fetching same agent returns cached instance
    same_agent = await manager.get_or_create_agent("test-agent-1")
    assert same_agent is agent

    # Stop specific agent
    await manager.stop_agent("test-agent-1")
    assert manager.get_agent("test-agent-1") is None
    assert agent.state == AgentState.TERMINATED

    # Create multiple and clear
    ag2 = await manager.get_or_create_agent("test-agent-2")
    ag3 = await manager.get_or_create_agent("test-agent-3")
    assert len(manager.list_agents()) == 2
    await manager.clear()
    assert len(manager.list_agents()) == 0
    assert ag2.state == AgentState.TERMINATED
    assert ag3.state == AgentState.TERMINATED

    # Test global singleton accessor
    global_mgr = get_ui_session_manager()
    assert global_mgr is not None


def test_ui_chat_history_empty_session(tmp_path: Path) -> None:
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=MockLLMConnector())
    client = TestClient(app)

    response = client.get("/api/session/history?agent_id=agent-test&session_id=sess_empty")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["session_id"] == "sess_empty"
    assert data["agent_id"] == "agent-test"
    assert data["messages"] == []


def test_ui_chat_history_persistence_and_retrieval(tmp_path: Path) -> None:
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=storage_dir,
        llm=MockLLMConnector(responses=["First canned reply", "Second canned reply"]),
    )
    client = TestClient(app)

    # 1. First chat message
    res1 = client.post(
        "/api/turn",
        json={
            "message": "Hello agent!",
            "agent_id": "agent-history-test",
            "session_id": "sess_hist_1",
        },
    )
    assert res1.status_code == 200

    # Verify session file was created on disk
    # Transcripts moved to `<root>/ui/` so they cannot collide with the Core record
    # the CLI writes at `<root>/<id>.json`; see UI_TRANSCRIPT_SUBDIR (#215 review).
    session_file = storage_dir / "ui" / "sess_hist_1.json"
    assert session_file.is_file()

    # 2. Second chat message
    res2 = client.post(
        "/api/turn",
        json={
            "message": "Tell me more.",
            "agent_id": "agent-history-test",
            "session_id": "sess_hist_1",
        },
    )
    assert res2.status_code == 200

    # 3. Retrieve chat history via GET /api/session/history
    hist_res = client.get("/api/session/history?agent_id=agent-history-test&session_id=sess_hist_1")
    assert hist_res.status_code == 200
    hist_data = cast(dict[str, Any], hist_res.json())
    assert hist_data["session_id"] == "sess_hist_1"
    assert hist_data["agent_id"] == "agent-history-test"
    messages = cast(list[dict[str, Any]], hist_data["messages"])
    assert len(messages) == 4  # 2 user messages + 2 agent responses

    assert messages[0]["sender"] == "user"
    assert messages[0]["content"] == "Hello agent!"
    assert messages[1]["sender"] == "agent"
    assert messages[1]["content"] == "First canned reply"
    assert messages[1]["turn_count"] == 1
    assert messages[2]["sender"] == "user"
    assert messages[2]["content"] == "Tell me more."
    assert messages[3]["sender"] == "agent"
    assert messages[3]["content"] == "Second canned reply"
    assert messages[3]["turn_count"] == 2


def test_ui_chat_history_hydration_across_app_reloads(tmp_path: Path) -> None:
    storage_dir = tmp_path / "sessions"
    # App instance 1: run turn 1
    mock_llm_1 = MockLLMConnector(responses=["Reply from app 1"])
    app_1 = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=mock_llm_1)
    client_1 = TestClient(app_1)

    r1 = client_1.post(
        "/api/turn",
        json={
            "message": "Step 1",
            "agent_id": "agent-hydrated",
            "session_id": "sess_persist",
        },
    )
    assert r1.status_code == 200
    assert r1.json()["turn_count"] == 1

    # App instance 2 (simulates server restart or new process with same storage_dir):
    mock_llm_2 = MockLLMConnector(responses=["Reply from app 2"])
    app_2 = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=mock_llm_2)
    client_2 = TestClient(app_2)

    # Verify history is immediately accessible before new chat
    hist = client_2.get("/api/session/history?agent_id=agent-hydrated&session_id=sess_persist")
    assert hist.status_code == 200
    assert len(hist.json()["messages"]) == 2

    # Execute turn 2: agent should be hydrated and turn count should be 2
    r2 = client_2.post(
        "/api/turn",
        json={
            "message": "Step 2",
            "agent_id": "agent-hydrated",
            "session_id": "sess_persist",
        },
    )
    assert r2.status_code == 200
    assert r2.json()["turn_count"] == 2

    # Verify history now contains all 4 messages
    hist2 = client_2.get("/api/session/history?agent_id=agent-hydrated&session_id=sess_persist")
    assert hist2.status_code == 200
    assert len(hist2.json()["messages"]) == 4


def test_ui_chat_history_clear_endpoint(tmp_path: Path) -> None:
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=storage_dir,
        llm=MockLLMConnector(responses=["Message to be cleared"]),
    )
    client = TestClient(app)

    # 1. Send message
    client.post(
        "/api/turn",
        json={
            "message": "Forget me",
            "agent_id": "agent-clear-test",
            "session_id": "sess_clear",
        },
    )
    session_file = storage_dir / "ui" / "sess_clear.json"
    assert session_file.is_file()

    # 2. Call DELETE /api/session/history
    del_res = client.delete("/api/session/history?agent_id=agent-clear-test&session_id=sess_clear")
    assert del_res.status_code == 200
    assert del_res.json() == {"status": "cleared"}
    assert not session_file.exists()

    # 3. Subsequent GET /api/session/history returns empty
    hist_res = client.get("/api/session/history?agent_id=agent-clear-test&session_id=sess_clear")
    assert hist_res.status_code == 200
    assert hist_res.json()["messages"] == []


def test_ui_chat_history_path_traversal_prevention(tmp_path: Path) -> None:
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector())
    client = TestClient(app)

    # Traversal in GET
    r_get = client.get("/api/session/history?agent_id=champion&session_id=../../etc/passwd")
    assert r_get.status_code == 400

    # Traversal in DELETE
    r_del = client.delete("/api/session/history?agent_id=champion&session_id=../../etc/passwd")
    assert r_del.status_code == 400


@pytest.mark.asyncio
async def test_agent_session_manager_persistence_and_corrupt_files(tmp_path: Path) -> None:
    storage_dir = tmp_path / "sessions"
    manager = AgentSessionManager(storage_dir=storage_dir, fallback_to_mock=True)

    # Save session
    manager.save_session_record(
        session_id="sess_manual",
        agent_id="test_agent",
        messages=[{"sender": "user", "content": "Manual message"}],
        turns=1,
    )
    assert (storage_dir / "ui" / "sess_manual.json").is_file()

    # Load session
    record = manager.load_session_record("sess_manual")
    assert record is not None
    assert record["session_id"] == "sess_manual"
    assert len(record["messages"]) == 1

    # Corrupt JSON file handling
    corrupt_file = storage_dir / "ui" / "sess_corrupt.json"
    corrupt_file.write_text("NOT_VALID_JSON{{{", encoding="utf-8")
    assert manager.load_session_record("sess_corrupt") is None
    assert manager.get_session_history("test_agent", "sess_corrupt") == []

    # Path traversal in session manager raises PathTraversalError
    from uclone_x.errors import PathTraversalError

    with pytest.raises(PathTraversalError):
        manager.get_session_path("../outside")


def test_ui_chat_transcript_and_core_persistence_failure_resilience(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chat endpoint is resilient to transcript read/save or core persist errors, logging warnings (#229)."""
    import logging

    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector())
    mgr: AgentSessionManager = app.state.session_manager
    client = TestClient(app)

    # 1. Mock save_session_record failure
    with caplog.at_level(logging.WARNING):
        with patch.object(mgr, "save_session_record", side_effect=OSError("Disk write failed")):
            res = client.post(
                "/api/turn",
                json={"message": "Test save failure", "agent_id": "agent-persist-fail"},
            )
            assert res.status_code == 200
            assert res.json()["status"] == "success"
            assert "Failed to save transcript" in caplog.text

    caplog.clear()

    # 2. Mock get_session_history failure
    with caplog.at_level(logging.WARNING):
        with patch.object(mgr, "get_session_history", side_effect=OSError("Disk read failed")):
            res = client.post(
                "/api/turn",
                json={"message": "Test read failure", "agent_id": "agent-read-fail"},
            )
            assert res.status_code == 200
            assert res.json()["status"] == "success"
            assert "Failed to read session history" in caplog.text


def test_ui_chat_offline_llm_warning_notice(tmp_path: Path) -> None:
    from uclone_x.llm.connectors.ollama import OllamaConnector
    from uclone_x.ui.app import OFFLINE_LLM_DIAGNOSTIC_MESSAGE

    def _fail_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_fail_connect),
        base_url="http://127.0.0.1:11434",
    )
    unreachable_ollama = OllamaConnector(http_client=mock_client)
    app = create_ui_app(static_dir=tmp_path, llm=unreachable_ollama)
    client = TestClient(app)

    res = client.post(
        "/api/turn",
        json={"message": "Hello when offline", "agent_id": "agent-offline-test"},
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["status"] == "warning"
    assert data["response"] == OFFLINE_LLM_DIAGNOSTIC_MESSAGE
    assert "⚠️ No active LLM provider connected." in data["response"]
    assert "ollama serve" in data["response"]
    assert "./ucx llm status" in data["response"]
    assert data["provenance"]["degraded"] is True
    assert data["provenance"]["path"] == "OFFLINE_FALLBACK"
    assert data["provenance"]["served_by"] == "offline_diagnostic"


def test_ui_chat_llm_provider_error_handling(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock

    from uclone_x.errors import LLMProviderError
    from uclone_x.ui.app import OFFLINE_LLM_DIAGNOSTIC_MESSAGE

    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(
        side_effect=LLMProviderError("Failed to connect to Ollama: Connection refused")
    )

    app = create_ui_app(static_dir=tmp_path, llm=mock_llm)
    client = TestClient(app)

    res = client.post(
        "/api/turn",
        json={"message": "Test Ollama down", "agent_id": "agent-ollama-down"},
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["status"] == "warning"
    assert data["response"] == OFFLINE_LLM_DIAGNOSTIC_MESSAGE
    assert data["provenance"]["degraded"] is True


@pytest.mark.asyncio
async def test_agent_session_manager_fallback_to_mock_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uclone_x.llm.connectors.ollama import OllamaConnector

    # `fallback_to_mock` is the unconfigured-case default, not an override, so the two halves
    # need different environments and the scoping is the assertion. With a provider named,
    # the flag must not shadow it; with none named, the flag chooses.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    mgr_default = AgentSessionManager()
    assert mgr_default.fallback_to_mock is False
    ag_default = await mgr_default.get_or_create_agent("default-agent")
    assert isinstance(ag_default._llm, OllamaConnector)
    await mgr_default.clear()

    # Explicit fallback_to_mock=True with nothing configured -> MockLLMConnector.
    # `conftest` pins the suite to `mock`, so this clears it to reach the unconfigured path
    # the flag is defined over.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    mgr_mock = AgentSessionManager(fallback_to_mock=True)
    assert mgr_mock.fallback_to_mock is True
    ag_mock = await mgr_mock.get_or_create_agent("mock-agent")
    assert isinstance(ag_mock._llm, MockLLMConnector)
    await mgr_mock.clear()

    # And with neither a provider nor the flag, the manager surfaces the refusal (#533)
    # rather than building an agent whose connector cannot work.
    from uclone_x.errors import LLMProviderNotConfiguredError

    mgr_unconfigured = AgentSessionManager()
    with pytest.raises(LLMProviderNotConfiguredError):
        await mgr_unconfigured.get_or_create_agent("unconfigured-agent")
    await mgr_unconfigured.clear()


@pytest.mark.asyncio
async def test_mock_llm_connector_and_factory() -> None:
    from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole

    # MockLLMConnector generate
    mock_llm = MockLLMConnector(responses=["First canned reply", "Second canned reply"])
    req1 = LLMRequest(
        model="test-model",
        messages=(ChatMessage(role=MessageRole.USER, content="Query 1"),),
    )
    resp1 = await mock_llm.generate(req1)
    assert resp1.content == "First canned reply"
    assert mock_llm.call_count == 1

    resp2 = await mock_llm.generate(req1)
    assert resp2.content == "Second canned reply"
    assert mock_llm.call_count == 2

    # MockLLMConnector stream
    chunks = []
    async for chunk in mock_llm.stream(req1):
        chunks.append(chunk)
    assert len(chunks) > 0

    # Factory with mock
    conn1 = create_llm_connector(provider="mock")
    assert isinstance(conn1, MockLLMConnector)

    # Factory with providers
    from uclone_x.llm.connectors.anthropic import AnthropicConnector
    from uclone_x.llm.connectors.gemini import GeminiConnector
    from uclone_x.llm.connectors.ollama import OllamaConnector
    from uclone_x.llm.connectors.openai import OpenAIConnector

    conn_oa = create_llm_connector(provider="openai", api_key="sk-test")
    assert isinstance(conn_oa, OpenAIConnector)

    conn_ant = create_llm_connector(provider="anthropic", api_key="sk-ant")
    assert isinstance(conn_ant, AnthropicConnector)

    conn_gem = create_llm_connector(provider="gemini", api_key="test-gemini")
    assert isinstance(conn_gem, GeminiConnector)

    conn_ol = create_llm_connector(provider="ollama", base_url="http://localhost:11434")
    assert isinstance(conn_ol, OllamaConnector)

    # With nothing configured the factory refuses rather than defaulting to a connector
    # that cannot work (#533). `conftest` pins `LLM_PROVIDER=mock` for the suite, so this
    # clears it to reach the unconfigured path.
    from uclone_x.errors import LLMProviderNotConfiguredError

    saved_provider = os.environ.pop("LLM_PROVIDER", None)
    try:
        with pytest.raises(LLMProviderNotConfiguredError):
            create_llm_connector()
    finally:
        if saved_provider is not None:
            os.environ["LLM_PROVIDER"] = saved_provider

    # An unsupported provider name is refused whatever `fallback_to_mock` says (#397, P6):
    # a mock returned there answers a request the factory could not understand.
    from uclone_x.errors import LLMProviderError as _LLMProviderError

    with pytest.raises(_LLMProviderError, match="Unsupported LLM provider"):
        create_llm_connector(provider="nonexistent_provider", fallback_to_mock=True)

    # Factory without fallback raising on error by default
    from uclone_x.errors import LLMProviderError

    with pytest.raises(LLMProviderError):
        create_llm_connector(provider="nonexistent_provider")

    with pytest.raises(LLMProviderError):
        create_llm_connector(provider="nonexistent_provider", fallback_to_mock=False)


def test_ui_dispatch_endpoint(test_client: TestClient) -> None:
    # A valid dispatch names its recipient: the spawn is published to that agent, and a
    # request that names nobody is refused rather than sent to an invented one (#1125).
    response = test_client.post(
        "/api/dispatch",
        json={"task": "Implement feature X", "role": "assistant", "agent_id": "champion"},
    )
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["status"] == "dispatched"
    assert data["agent_id"] == "champion"
    assert data["role"] == "assistant"
    assert "timestamp" in data

    # Test empty task dispatch
    err_res = test_client.post("/api/dispatch", json={"task": "", "agent_id": "champion"})
    assert err_res.status_code == 200
    assert "error" in err_res.json()


def test_ui_ontology_endpoint_empty(test_client: TestClient) -> None:
    """Assert /api/ontology returns truthful empty structure when unpopulated (P6, P8)."""
    response = test_client.get("/api/ontology")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert "concepts" in data
    assert "relations" in data
    assert "axioms" in data
    assert "summary" in data
    assert data["concepts"] == []
    assert data["relations"] == []
    assert data["axioms"] == []
    assert data["total_concepts"] == 0
    assert data["total_axioms"] == 0
    assert data["total_relations"] == 0
    assert data["summary"]["total_concepts"] == 0
    assert data["summary"]["asserted_count"] == 0


def test_ui_ontology_endpoint_populated(tmp_path: Path) -> None:
    """Assert /api/ontology reflects live concepts, relations, and axioms in OntologyEngine."""
    ontology_engine = OntologyEngine(agent_id="test-agent")
    ontology_engine.register_entity(
        OntologyConcept(
            name="TestTask",
            tier=OntologyTier.ASSERTED,
            attributes={"task_id": "string", "prompt": "string"},
            required_fields=("task_id",),
        )
    )
    ontology_engine.register_entity(
        OntologyConcept(
            name="InducedPattern",
            tier=OntologyTier.INDUCED_ENFORCING,
            attributes={"pattern_id": "string"},
            required_fields=("pattern_id",),
        )
    )
    ontology_engine.register_relation(
        OntologyRelation(
            source_entity="TestTask",
            predicate="generates",
            target_entity="InducedPattern",
            is_directed=True,
            tier=OntologyTier.ASSERTED,
        )
    )
    ontology_engine.register_axiom(
        OntologyAxiom(
            name="task_non_empty_prompt",
            subject_entity="TestTask",
            predicate="prompt != ''",
            rule_expression="len(prompt) > 0",
        )
    )

    app = create_ui_app(static_dir=tmp_path, ontology_engine=ontology_engine)
    client = TestClient(app)

    response = client.get("/api/ontology")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["total_concepts"] == 2
    assert data["total_relations"] == 1
    assert data["total_axioms"] == 1
    assert data["summary"]["asserted_count"] == 1
    assert data["summary"]["induced_enforcing_count"] == 1

    concept_names = {c["name"] for c in data["concepts"]}
    assert concept_names == {"TestTask", "InducedPattern"}
    assert data["relations"][0]["predicate"] == "generates"
    assert data["axioms"][0]["name"] == "task_non_empty_prompt"


def test_ui_skills_endpoint_empty(test_client: TestClient) -> None:
    """Assert /api/skills returns truthful empty list when unpopulated (P6, P8)."""
    response = test_client.get("/api/skills")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert "skills" in data
    assert "summary" in data
    assert data["skills"] == []
    assert data["total"] == 0
    assert data["summary"]["total_skills"] == 0
    assert data["summary"]["active_count"] == 0


def test_ui_skills_endpoint_populated(tmp_path: Path) -> None:
    """Assert /api/skills reflects live skills registered in SkillRegistry with audit reports."""
    skill_registry = SkillRegistry()
    manifest = SkillManifest(
        name="test_ast_analyzer",
        description="Analyzes AST safely",
        version="1.0.0",
        author="core",
        origin=SkillOrigin.HUMAN,
        status=SkillStatus.ACTIVE,
        content_sha256="abc123sha",
        scripts=("analyze.py",),
        tags=("ast", "lint"),
    )
    skill = Skill(manifest=manifest, instructions_markdown="# Instructions")
    report = SkillAuditReport(
        skill_name="test_ast_analyzer",
        is_safe=True,
        recommendation=AuditVerdict.APPROVE,
        risk_score=0.05,
        detected_risks=(),
        auditor_version="0.1.0",
        content_sha256="abc123sha",
    )
    skill_registry.register(skill, report)

    app = create_ui_app(static_dir=tmp_path, skill_registry=skill_registry)
    client = TestClient(app)

    response = client.get("/api/skills")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["total"] == 1
    assert data["summary"]["total_skills"] == 1
    assert data["summary"]["active_count"] == 1
    assert len(data["skills"]) == 1
    s = data["skills"][0]
    assert s["name"] == "test_ast_analyzer"
    assert s["audit_report"]["is_safe"] is True
    assert s["audit_report"]["recommendation"] == "approve"


def test_ui_budget_endpoint_empty(test_client: TestClient) -> None:
    """Assert /api/budget returns truthful zeroed usage when unpopulated (P6, P8)."""
    response = test_client.get("/api/budget")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert "session_budget" in data
    assert "providers" in data
    assert "roles" in data
    assert "compaction_history" in data
    assert data["total_tokens"] == 0
    assert data["prompt_tokens"] == 0
    assert data["completion_tokens"] == 0
    assert data["session_budget"]["used_input_tokens"] == 0
    assert data["session_budget"]["used_output_tokens"] == 0
    assert data["session_budget"]["total_used_tokens"] == 0
    assert data["providers"] == {}
    assert data["roles"] == {}
    assert data["compaction_history"] == []


def test_ui_budget_endpoint_populated(tmp_path: Path) -> None:
    """Assert /api/budget reflects live token usages, per-provider tokens, and compactions."""
    budget_tracker = TokenBudgetManager(default_max_tokens=500_000)
    usage = TokenUsage(
        input_tokens=150,
        output_tokens=50,
        provider="openai",
        model="gpt-4o",
    )
    budget_tracker.record_usage(session_id="sess_1", usage=usage)
    budget_tracker.record_compaction(
        reason="Threshold reached",
        original_tokens=1000,
        compacted_tokens=250,
        kept_turns=4,
    )

    app = create_ui_app(static_dir=tmp_path, budget_tracker=budget_tracker)
    client = TestClient(app)

    response = client.get("/api/budget")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["total_tokens"] == 200
    assert data["prompt_tokens"] == 150
    assert data["completion_tokens"] == 50
    # UClone-X counts tokens only (#1392): the route carries no cost figure.
    assert "total_cost_usd" not in data
    assert "cost_usd" not in data["providers"]["openai"]
    assert "openai" in data["providers"]
    assert data["providers"]["openai"]["input_tokens"] == 150
    assert data["providers"]["openai"]["output_tokens"] == 50
    assert data["providers"]["openai"]["models"] == ["gpt-4o"]
    assert len(data["compaction_history"]) == 1
    assert data["compaction_history"][0]["saved_tokens"] == 750
    assert data["compaction_history"][0]["compression_ratio_pct"] == 75.0


def test_ui_stream_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/api/stream?max_events=1")
    assert response.status_code == 200
    assert "SYSTEM_CONNECTED" in response.text
    assert "HEARTBEAT" in response.text


@pytest.mark.asyncio
async def test_ui_stream_endpoint_with_published_event(tmp_path: Path) -> None:
    import httpx

    from uclone_x.engine.event_bus import AgentEvent, EventSource, EventType
    from uclone_x.ui.app import create_ui_app, get_ui_event_bus

    app = create_ui_app(static_dir=tmp_path)
    bus = get_ui_event_bus()

    async def _publish_loop() -> None:
        for _ in range(5):
            await asyncio.sleep(0.01)
            await bus.publish(
                AgentEvent(
                    type=EventType.AGENT_REPLY,
                    source=EventSource.AGENT,
                    sender_id="test_agent",
                    topic="agent.test",
                    payload={"msg": "hello from test"},
                )
            )

    pub_task = asyncio.create_task(_publish_loop())
    chunks: list[str] = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as ac:
        async with ac.stream("GET", "/api/stream?max_events=3") as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                chunks.append(line)
    await pub_task
    full_text = "\n".join(chunks)
    assert "AGENT_EVENT" in full_text
    assert "agent.test" in full_text


@pytest.mark.asyncio
async def test_ui_stream_receives_chat_turn_events(tmp_path: Path) -> None:
    """SSE subscribers observe a chat turn's post-turn AGENT_REPLY event and provenance (#220).

    Asserts specifically on the post-turn AGENT_REPLY event rather than the pre-turn
    USER_INPUT event, and verifies that the SSE frame carries the structured provenance
    block produced by _sse_provenance_block (P6).
    """
    bus = EventBus()
    app = create_ui_app(static_dir=tmp_path, bus=bus, llm=stubbed_ollama_connector())

    async def _post_chat() -> None:
        await asyncio.sleep(0.05)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            await client.post(
                "/api/turn",
                json={"message": "Streaming test prompt", "agent_id": "stream-agent"},
            )

    post_task = asyncio.create_task(_post_chat())
    chunks: list[str] = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as ac:
        async with ac.stream("GET", "/api/stream?max_events=6") as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                chunks.append(line)
    await post_task
    full_text = "\n".join(chunks)
    assert "AGENT_EVENT" in full_text
    assert "AGENT_REPLY" in full_text or "agent_reply" in full_text

    # Parse and assert on the post-turn AGENT_REPLY frame
    parsed_events: list[dict[str, Any]] = []
    for line in chunks:
        if line.startswith("data: "):
            try:
                parsed_events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass

    reply_events = [
        e
        for e in parsed_events
        if e.get("event_type") in ("AGENT_REPLY", "agent_reply") or e.get("type") == "AGENT_REPLY"
    ]
    assert len(reply_events) >= 1, f"Expected AGENT_REPLY in SSE stream, got {parsed_events}"
    reply = reply_events[0]
    assert "provenance" in reply
    assert reply["provenance"]["component"] == "uclone_x.engine.event_bus"
    assert reply["provenance"]["producer"] == "stream-agent"
    assert reply["provenance"]["path"] == "primary"


def test_ui_static_serving(test_client: TestClient) -> None:
    response = test_client.get("/")
    assert response.status_code == 200
    assert "Test UI" in response.text


def test_ui_default_static_bundle_serving() -> None:
    # Test loading actual compiled static bundle from default location
    app = create_ui_app(llm=MockLLMConnector())
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>UClone-X Dashboard</title>" in response.text
    assert '<div id="root"></div>' in response.text


def test_ui_static_asset_serving() -> None:
    # Test serving assets from static directory
    app = create_ui_app(llm=MockLLMConnector())
    client = TestClient(app)
    # Check index.html is served and find asset links
    index_res = client.get("/")
    assert index_res.status_code == 200
    assert "/assets/" in index_res.text


def test_ui_fallback_index_when_no_static() -> None:
    app = create_ui_app(static_dir=Path("/non/existent/path"))
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert "message" in data
    assert "/api/ontology" in data["endpoints"]
    assert "/api/skills" in data["endpoints"]
    assert "/api/budget" in data["endpoints"]
    assert "/api/evaluations/latest" in data["endpoints"]
    assert "/api/evaluations/history" in data["endpoints"]


def test_ui_fallback_index_when_empty_static_dir(tmp_path: Path) -> None:
    # Directory exists but no index.html present
    empty_dir = tmp_path / "empty_static"
    empty_dir.mkdir()
    app = create_ui_app(static_dir=empty_dir)
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert "message" in data
    assert "/api/health" in data["endpoints"]


@pytest.mark.usefixtures("frontend_build_suppressed")
def test_start_ui_server_production(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_uvicorn = MagicMock()
    monkeypatch.setattr("uvicorn.run", mock_uvicorn)

    start_ui_server(port=5180, dev=False, host="127.0.0.1")
    assert mock_uvicorn.called
    _, kwargs = mock_uvicorn.call_args
    assert kwargs.get("timeout_graceful_shutdown") == 2


@pytest.mark.usefixtures("frontend_build_suppressed")
def test_start_ui_server_records_its_dashboard_while_serving_and_removes_it_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ucx ui stop` identifies a dashboard by this record, so it must track the server (#927).

    Written before the server runs, naming this process as `ps` sees it; removed when the
    server exits — here by failing to start, the path where a record left behind would name
    a dashboard that never served. Nothing about the record reaches the environment that
    agent tool subprocesses inherit.

    Killed by: src/uclone_x/ui/server.py ::
        ui_process.remove_dashboard_record(record)
    Becomes: pass
    """
    port = 5180
    path = ui_process.dashboard_record_path(port)
    assert path.parent != ui_process.DEFAULT_DASHBOARD_STATE_DIR
    environment_before = dict(os.environ)
    seen: list[ui_process.DashboardRecord | None] = []

    def failing_run(*_args: Any, **_kwargs: Any) -> None:
        seen.append(ui_process.read_dashboard_record(port))
        raise OSError("address already in use")

    monkeypatch.setattr("uvicorn.run", failing_run)

    with pytest.raises(OSError, match="address already in use"):
        start_ui_server(port=port, dev=False, host="127.0.0.1")

    record = seen[0]
    assert record is not None
    assert (record.pid, record.port, record.host) == (os.getpid(), port, "127.0.0.1")
    assert record.process == ui_process.process_identity(os.getpid())
    assert record.instance not in "".join(os.environ.values())
    assert set(os.environ) - set(environment_before) <= {"UCLONE_SESSION_DIR"}
    assert not path.exists()


def test_should_rebuild_frontend_missing_files(tmp_path: Path) -> None:
    from uclone_x.ui.server import _should_rebuild_frontend

    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    static_dir = tmp_path / "ui_static"
    # Static files missing -> should rebuild
    assert _should_rebuild_frontend(frontend_dir, static_dir) is True


def test_should_rebuild_frontend_missing_record(tmp_path: Path) -> None:
    """A bundle with no digest record beside it cannot be judged, so it is rebuilt.

    The trigger's decisions, and the mutations that pin them, are in
    `tests/unit/test_frontend_build_inputs.py`; this is the neighbour of the missing-files
    case above, kept here so the two structural refusals read together. The mtime case that
    stood here until #1075 asserted the defect: it built a tree whose source was newer than
    the bundle and required `True`, which is what a `git checkout` produces on a pristine
    tree.
    """
    from uclone_x.ui.server import _should_rebuild_frontend

    frontend_dir = tmp_path / "frontend"
    (frontend_dir / "src").mkdir(parents=True)
    static_dir = tmp_path / "ui_static"
    (static_dir / "assets").mkdir(parents=True)
    (static_dir / "index.html").write_text("<html></html>")

    assert _should_rebuild_frontend(frontend_dir, static_dir) is True


@pytest.mark.usefixtures("frontend_build_suppressed")
def test_start_ui_server_dev_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dev branch wires uvicorn for reload and reports HMR as active (#1082).

    `vite_proc` is a `MagicMock` here and so never `None`, which means the launcher always
    reaches `_diagnose_vite`. Left alone that is a real `httpx.get` against
    `127.0.0.1:5173` polled to a 10 s deadline, so the test cost ten seconds and took
    whichever of the three branches the developer's own port 5173 decided — ours, another
    project's app, or nothing answering. The response is fixed below and the branch it
    produces is asserted, which is the half that makes the speed-up more than a speed-up:
    before, every branch passed.

    Killed by: src/uclone_x/ui/server.py :: if VITE_IDENTITY_MARKER in body:
    Becomes: if VITE_IDENTITY_MARKER not in body:
    """
    # `subprocess.Popen` is mocked below for the co-spawned Vite; `_ensure_frontend_built`
    # uses `subprocess.run`, so that mock never reached it and the fixture is what does.
    mock_uvicorn = MagicMock()
    mock_popen = MagicMock()
    monkeypatch.setattr("uvicorn.run", mock_uvicorn)
    # The dashboard record reads this process's identity through `ps`, which the global
    # `Popen` mock below would otherwise answer with a `MagicMock`.
    identity = ui_process.process_identity(os.getpid())
    monkeypatch.setattr(ui_process, "process_identity", MagicMock(return_value=identity))
    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr(httpx, "get", answering_as(VITE_IDENTITY_MARKER))

    start_ui_server(port=5180, dev=True, host="127.0.0.1", vite_port=5173)
    assert mock_uvicorn.called
    _, kwargs = mock_uvicorn.call_args
    assert kwargs.get("reload") is True
    assert kwargs.get("reload_dirs") == [str(Path(__file__).resolve().parents[2] / "src")]
    assert kwargs.get("timeout_graceful_shutdown") == 2

    printed = one_line(capsys.readouterr().out)
    assert "Instant React HMR Active" in printed, printed
    assert "React HMR is not available" not in printed, printed


@pytest.mark.usefixtures("frontend_build_suppressed")
def test_reload_mode_is_given_the_factory_not_a_built_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing `uclone_x.ui.app` must not build a dashboard as a side effect (#1125).

    Reload mode needs an import string, and it used to be `uclone_x.ui.app:app` -- a
    module-level `create_ui_app()`. So every importer of that module, for any one function
    or type, also constructed an `AgentSessionManager` wired to the real
    `~/.uclone/sessions`, at import time, before any fixture could redirect it. Under the
    per-agent home layout that same construction creates directories, which is a test
    writing into the developer's own agents root (#453's leak, through the import graph).

    `factory=True` moves the construction to the server that actually serves.

    Killed by: src/uclone_x/ui/server.py :: factory=True,
    Becomes: factory=False,
    """
    import uclone_x.ui.app as ui_app_module

    assert not hasattr(ui_app_module, "app"), (
        "a module-level app is built by the import itself, which is what this closes"
    )

    mock_uvicorn = MagicMock()
    monkeypatch.setattr("uvicorn.run", mock_uvicorn)
    identity = ui_process.process_identity(os.getpid())
    monkeypatch.setattr(ui_process, "process_identity", MagicMock(return_value=identity))
    monkeypatch.setattr("subprocess.Popen", MagicMock())
    # The mocked `Popen` makes `vite_proc` non-None, so the launcher polls port 5173 for
    # our dev server before calling uvicorn: a real 10 s wait, as #1082 found in
    # `test_start_ui_server_dev_mode`. The Vite branch is not this test's subject.
    monkeypatch.setattr(httpx, "get", answering_as(VITE_IDENTITY_MARKER))

    start_ui_server(port=5180, dev=True, host="127.0.0.1", vite_port=5173)

    args, kwargs = mock_uvicorn.call_args
    assert args[0] == "uclone_x.ui.app:create_ui_app"
    assert kwargs.get("factory") is True


def test_cli_ui_dev_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_start = MagicMock()
    monkeypatch.setattr("uclone_x.ui.server.start_ui_server", mock_start)

    result = runner.invoke(main.app, ["ui", "--port", "5180", "--dev", "--vite-port", "5174"])
    assert result.exit_code == 0
    assert mock_start.called
    _, kwargs = mock_start.call_args
    assert kwargs.get("dev") is True
    assert kwargs.get("vite_port") == 5174


@pytest.mark.asyncio
async def test_ui_stream_generator_terminates_on_shutdown_event(tmp_path: Path) -> None:
    """Assert SSE generator terminates cleanly when shutdown event is triggered (#613).

    Killed by: src/uclone_x/ui/app.py :: shutdown_task = asyncio.create_task(effective_shutdown.wait())
    """
    from fastapi.routing import APIRoute
    from starlette.requests import Request
    from starlette.responses import StreamingResponse

    shutdown_event = asyncio.Event()
    app = create_ui_app(static_dir=tmp_path, shutdown_event=shutdown_event)

    route = next(r for r in app.routes if isinstance(r, APIRoute) and r.path == "/api/stream")
    mock_request = MagicMock(spec=Request)
    mock_request.app = app
    mock_request.app.state.shutdown_event = shutdown_event
    mock_request.is_disconnected = AsyncMock(return_value=False)

    response = await route.endpoint(request=mock_request)
    assert isinstance(response, StreamingResponse)

    start_time = asyncio.get_running_loop().time()
    # Trigger shutdown while generator is waiting for bus events
    asyncio.get_running_loop().call_later(0.05, shutdown_event.set)

    chunks: list[str] = []
    async for chunk in response.body_iterator:
        text_chunk = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        chunks.append(text_chunk)

    elapsed = asyncio.get_running_loop().time() - start_time
    full_text = "\n".join(chunks)
    assert "SYSTEM_CONNECTED" in full_text
    assert elapsed < 0.5, f"Generator took too long to terminate on shutdown: {elapsed:.2f}s"


def test_ui_server_terminates_cleanly_during_active_sse_stream() -> None:
    """Reproduction test: Uvicorn server terminates cleanly on shutdown during active SSE stream (#613).

    Killed by: src/uclone_x/ui/app.py :: "type": "SYSTEM_CONNECTED",
    Becomes: "type": "SYSTEM_HELLO",
    """
    import threading
    import time

    import uvicorn

    from uclone_x.ui.app import create_ui_app

    app = create_ui_app()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config=config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    while not server.started:
        time.sleep(0.05)

    assert server.servers and server.servers[0].sockets
    port = server.servers[0].sockets[0].getsockname()[1]

    connected = threading.Event()

    def _client() -> None:
        try:
            with httpx.stream("GET", f"http://127.0.0.1:{port}/api/stream", timeout=10.0) as r:
                for line in r.iter_lines():
                    if "SYSTEM_CONNECTED" in line:
                        connected.set()
        except Exception:
            pass

    client_thread = threading.Thread(target=_client, daemon=True)
    client_thread.start()

    assert connected.wait(timeout=5.0)

    # Signal shutdown
    server.should_exit = True
    thread.join(timeout=4.0)
    assert not thread.is_alive(), "UI server hanged on shutdown during active SSE stream!"


# ==============================================================================
# In-band provenance on the SSE stream and the UI's own publishers (#117)
# ==============================================================================


def test_sse_provenance_block_reports_only_what_the_event_states() -> None:
    """P6: this layer never asserts `degraded` on behalf of a producer that stated nothing.

    The block used to hard-code `"degraded": False` for every frame. Once `AgentEvent`
    began carrying real provenance that would have streamed a genuinely degraded reply
    to the UI as clean, so `path`/`degraded` are now present only when the event states
    them and absent otherwise — both are optional in `EventEnvelope`.
    """
    from uclone_x.engine.event_bus import AgentEvent, EventType
    from uclone_x.ui.app import _sse_provenance_block

    unstated = _sse_provenance_block(AgentEvent(type=EventType.AGENT_REPLY, sender_id="agt"))
    assert unstated["producer"] == "agt"
    assert "degraded" not in unstated
    assert "path" not in unstated

    degraded = _sse_provenance_block(
        AgentEvent(
            type=EventType.AGENT_REPLY,
            sender_id="agt",
            provenance=Provenance(
                path=ExecutionPath.FAILOVER,
                requested=ServiceRef(provider="primary_llm"),
                served_by=ServiceRef(provider="secondary_llm"),
                attempts=(AttemptRecord(provider="primary_llm", error_class="Timeout"),),
            ),
        )
    )
    assert degraded["degraded"] is True
    assert degraded["path"] == "failover"
    assert degraded["served_by"] == "secondary_llm"

    clean = _sse_provenance_block(
        AgentEvent(
            type=EventType.AGENT_REPLY,
            sender_id="agt",
            provenance=Provenance.primary("mock", "m-1"),
        )
    )
    assert clean["degraded"] is False
    assert clean["path"] == "primary"

    # Failed turn without provenance states degraded: True
    failed_unstated = _sse_provenance_block(
        AgentEvent(
            type=EventType.AGENT_REPLY,
            sender_id="agt",
            payload={"error": "Connection timed out", "is_completed": "False"},
        )
    )
    assert failed_unstated["degraded"] is True

    # Failed turn with is_completed="False" states degraded: True
    failed_clean_prov = _sse_provenance_block(
        AgentEvent(
            type=EventType.AGENT_REPLY,
            sender_id="agt",
            payload={"is_completed": "False"},
            provenance=Provenance.primary("mock", "m-1"),
        )
    )
    assert failed_clean_prov["degraded"] is True

    # Persona stated in payload is emitted in SSE provenance block
    with_persona = _sse_provenance_block(
        AgentEvent(
            type=EventType.AGENT_REPLY,
            sender_id="agt",
            payload={"persona": "champion"},
            provenance=Provenance.primary("mock", "m-1"),
        )
    )
    assert with_persona["persona"] == "champion"

    # When persona is not stated, it is omitted rather than fabricated (P6)
    assert "persona" not in clean
    assert "persona" not in unstated

    empty_persona = _sse_provenance_block(
        AgentEvent(
            type=EventType.AGENT_REPLY,
            sender_id="agt",
            payload={"persona": "   "},
            provenance=Provenance.primary("mock", "m-1"),
        )
    )
    assert "persona" not in empty_persona


@pytest.mark.asyncio
async def test_chat_endpoint_emits_persona_in_provenance_and_response(tmp_path: Path) -> None:
    """Verify /api/turn emits persona attribution in response and provenance (FR-13.4, P6)."""
    mock_llm = MockLLMConnector(
        default_model="mock-gpt-4o",
        default_response="Champion response.",
    )
    app = create_ui_app(storage_dir=tmp_path, llm=mock_llm)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        # 1. Clone persona is populated and emitted in /api/turn
        resp = await client.post(
            "/api/turn",
            json={
                "message": "Hello Clone",
                "agent_id": "clone",
                "session_id": "sess_clone_test",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["persona"] == "clone"
        assert data["provenance"]["persona"] == "clone"
        assert data["provenance"]["served_by"] == "mock:mock-gpt-4o"

        # 2. Chat history carries persona on assistant message
        hist_resp = await client.get(
            "/api/session/history?agent_id=clone&session_id=sess_clone_test"
        )
        assert hist_resp.status_code == 200
        history_data = hist_resp.json()
        messages = history_data["messages"]
        assert len(messages) >= 2
        agent_msg = [m for m in messages if m.get("sender") == "agent"][-1]
        assert agent_msg.get("persona") == "clone"
        assert agent_msg.get("provenance", {}).get("persona") == "clone"

        # 3. Generic unconfigured agent does NOT fabricate persona (P6)
        generic_resp = await client.post(
            "/api/turn",
            json={
                "message": "Hello Generic",
                "agent_id": "generic_custom_agent",
                "session_id": "sess_gen_test",
            },
        )
        assert generic_resp.status_code == 200
        gen_data = generic_resp.json()
        assert gen_data.get("persona") is None
        assert "persona" not in gen_data.get("provenance", {})


@pytest.mark.asyncio
async def test_chat_success_path_forwards_turn_provenance_to_the_bus(tmp_path: Path) -> None:
    """The success-path `AGENT_REPLY` publisher forwards the turn's own attribution (#117).

    This publisher previously emitted no provenance at all, while `turn_result.provenance`
    sat in scope fourteen lines below it, feeding the HTTP DTO.

    **What this proves, and what it stopped proving (#212).** The connector is now a real
    `OllamaConnector` over `httpx.MockTransport` rather than a live inference call, so the
    forwarding chain still runs end to end in production code —
    `ollama.py`'s `Provenance.primary(...)` -> `ModelResponse.provenance` ->
    `BaseAgent.execute_turn` -> `TurnResult.provenance` -> `AgentEvent.provenance` — and
    `served_by.provider` below is still the string `ollama.py` put there, not one this
    test wrote. What is no longer covered is the wire contract: that a live Ollama's
    actual response body has the shape `ollama.py` parses. That was only ever incidental
    coverage here, it is what made this test take ~54s and go red under load, and it is
    the sole reason `./ucx test check` was load-dependent.
    """
    from uclone_x.engine.event_bus import EventType

    bus = EventBus()
    app = create_ui_app(static_dir=tmp_path, bus=bus, llm=stubbed_ollama_connector())
    replies = bus.subscribe("agent.chat.reply")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post("/api/turn", json={"message": "hi", "agent_id": "prov-agent"})
    assert response.json()["status"] == "success"

    reply = await asyncio.wait_for(replies.get(), timeout=2.0)
    assert reply.type is EventType.AGENT_REPLY
    provenance = reply.provenance
    assert provenance is not None
    assert provenance.path is ExecutionPath.PRIMARY
    assert provenance.degraded is False
    # Forwarded verbatim: it names the connector that answered, not the UI.
    assert provenance.served_by.provider != "uclone_x.ui"


@pytest.mark.asyncio
async def test_chat_diagnostic_path_publishes_a_degraded_provenance(tmp_path: Path) -> None:
    """When this endpoint substitutes a diagnostic, it says so in band (#117, P6).

    The user is not looking at the agent's answer, so `requested` (the agent) differs
    from `served_by` (this endpoint) and `degraded` computes True rather than being
    asserted by hand.
    """
    from unittest.mock import AsyncMock

    import httpx

    from uclone_x.engine.event_bus import EventType
    from uclone_x.telemetry import TelemetryTracer

    bus = EventBus()
    tracer = TelemetryTracer()
    failing_llm = MagicMock(spec=LLMProviderProtocol)
    failing_llm.generate = AsyncMock(
        side_effect=LLMProviderError("Connection refused to Ollama at localhost:11434")
    )
    failing_mgr = AgentSessionManager(bus=bus, tracer=tracer, llm=failing_llm)
    app = create_ui_app(static_dir=tmp_path, bus=bus, tracer=tracer, session_manager=failing_mgr)
    replies = bus.subscribe("agent.chat.reply")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post("/api/turn", json={"message": "hi", "agent_id": "prov-agent"})
    assert response.json()["status"] in {"warning", "error"}

    reply = await asyncio.wait_for(replies.get(), timeout=2.0)
    assert reply.type is EventType.AGENT_REPLY
    provenance = reply.provenance
    assert provenance is not None
    assert provenance.path is ExecutionPath.FAILOVER
    assert provenance.degraded is True
    assert provenance.requested.provider == "prov-agent"
    assert provenance.served_by.provider == "agent.core"
    assert provenance.attempts[0].error_class == "LLMProviderError"
    assert provenance.attempts[0].span_id is not None
    assert provenance.attempts[0].span_id.startswith("spn_")


@pytest.mark.asyncio
async def test_ui_stream_receives_degraded_true_on_provider_error(tmp_path: Path) -> None:
    """UI SSE stream receives degraded: true when a provider error / failed turn occurs (Issue #150)."""
    import json
    from unittest.mock import AsyncMock

    import httpx

    mock_llm = MagicMock(spec=LLMProviderProtocol)
    mock_llm.generate = AsyncMock(side_effect=LLMProviderError("Connection refused to provider"))

    bus = EventBus()
    app = create_ui_app(static_dir=tmp_path, bus=bus, llm=mock_llm)

    async def _post_failing_chat() -> None:
        await asyncio.sleep(0.05)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            await client.post(
                "/api/turn",
                json={"message": "Crash prompt", "agent_id": "stream-fail-agent"},
            )

    post_task = asyncio.create_task(_post_failing_chat())
    chunks: list[str] = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as ac:
        async with ac.stream("GET", "/api/stream?max_events=8") as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                chunks.append(line)
    await post_task

    full_text = "\n".join(chunks)
    assert "AGENT_EVENT" in full_text
    assert "AGENT_REPLY" in full_text

    # Parse AGENT_REPLY event frame from SSE
    reply_frames: list[dict[str, Any]] = [
        cast(dict[str, Any], json.loads(line.removeprefix("data: ")))
        for line in chunks
        if line.startswith("data: ") and "AGENT_REPLY" in line
    ]
    assert len(reply_frames) >= 1
    reply_frame = reply_frames[0]
    payload = cast(dict[str, Any], reply_frame["payload"])
    assert payload["is_completed"] == "False"
    assert "provenance" in reply_frame
    provenance_block = cast(dict[str, Any], reply_frame["provenance"])
    assert provenance_block["degraded"] is True


@pytest.mark.asyncio
async def test_chat_diagnostic_path_publishes_failover_notice_ordered_before_reply(
    tmp_path: Path,
) -> None:
    """P6 Check 4: Chat fallback publishes PROVIDER_FAILOVER strictly before AGENT_REPLY."""
    from unittest.mock import AsyncMock

    import httpx

    from uclone_x.engine.event_bus import AgentEvent, EventType
    from uclone_x.telemetry import TelemetryTracer

    bus = EventBus()
    tracer = TelemetryTracer()
    failing_llm = MagicMock(spec=LLMProviderProtocol)
    failing_llm.generate = AsyncMock(
        side_effect=LLMProviderError("Connection refused to Ollama at localhost:11434")
    )
    failing_mgr = AgentSessionManager(bus=bus, tracer=tracer, llm=failing_llm)
    app = create_ui_app(
        static_dir=tmp_path,
        bus=bus,
        tracer=tracer,
        session_manager=failing_mgr,
    )
    all_events = bus.subscribe("agent.chat.*")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/api/turn", json={"message": "failover order test", "agent_id": "failover-agent"}
        )
    assert response.json()["status"] in {"warning", "error"}

    # Receive events from bus (USER_INPUT -> PROVIDER_FAILOVER -> AGENT_REPLY)
    events: list[AgentEvent] = []
    for _ in range(3):
        events.append(await asyncio.wait_for(all_events.get(), timeout=2.0))

    evt1 = next(e for e in events if e.type is EventType.PROVIDER_FAILOVER)
    evt2 = next(e for e in events if e.type is EventType.AGENT_REPLY)

    assert evt1.type is EventType.PROVIDER_FAILOVER
    assert evt2.type is EventType.AGENT_REPLY

    # P6 Check 4: strict total ordering (priority, sequence)
    assert evt1 < evt2
    assert evt1.sequence < evt2.sequence
    assert evt1.payload["requested_provider"] == "failover-agent"
    assert evt1.payload["served_provider"] == "agent.core"

    # P6 Check 5: telemetry span correlation
    spans = tracer.get_completed_spans()
    failover_spans = [s for s in spans if s.name == "failover.event"]
    assert len(failover_spans) >= 1
    failover_span = failover_spans[0]
    assert evt1.provenance is not None
    assert evt2.provenance is not None
    assert evt1.provenance.attempts[0].span_id == failover_span.span_id
    assert evt2.provenance.attempts[0].span_id == failover_span.span_id


@pytest.mark.asyncio
@pytest.mark.usefixtures("builtin_personas_absent")
async def test_ui_chat_tool_execution_error_records(tmp_path: Path) -> None:
    """Tool execution failures and exceptions are accurately recorded in tool_executions (Issue #171)."""
    from unittest.mock import AsyncMock

    mock_tool = MagicMock(spec=ToolProtocol)
    mock_tool.name = "failing_tool"
    mock_tool.description = "Failing tool"
    mock_tool.parameters_schema = {}
    mock_tool.execute = AsyncMock(
        return_value=ToolResult(
            output=None,
            success=False,
            error="Execution timeout in sandbox",
            execution_time_ms=12.5,
            isolation_level=IsolationLevel.WORKSPACE,
            provenance=None,
        )
    )

    tools = ToolRegistry()
    tools.register(mock_tool)

    tool_call1 = ToolCallRequest(
        id="tc_fail",
        name="failing_tool",
        arguments={"arg": "val"},
    )
    tool_call2 = ToolCallRequest(
        id="tc_not_found",
        name="nonexistent_tool",
        arguments={"x": 1},
    )
    mock_llm = MockLLMConnector(
        responses=["Attempted tools."],
        tool_calls=[tool_call1, tool_call2],
    )

    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, tools=tools)
    client = TestClient(app)

    res = client.post(
        "/api/turn",
        json={"message": "Run failing tools", "agent_id": "agent-tool-err"},
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert "tool_executions" in data
    assert len(data["tool_executions"]) == 2

    # First tool: executed with error result
    te1 = data["tool_executions"][0]
    assert te1["tool_name"] == "failing_tool"
    assert te1["tool_call_id"] == "tc_fail"
    assert te1["status"] == "error"
    assert te1["error"] == "Execution timeout in sandbox"
    assert te1["output"] is None
    assert te1["duration_ms"] == 12.5

    # Second tool: not found error
    te2 = data["tool_executions"][1]
    assert te2["tool_name"] == "nonexistent_tool"
    assert te2["tool_call_id"] == "tc_not_found"
    assert te2["status"] == "error"
    assert "not found" in str(te2["error"])


@pytest.mark.asyncio
async def test_ui_chat_debug_info_with_active_ontology_invariants(tmp_path: Path) -> None:
    """Debug info inspection returns active ontology invariants and prompt excerpts (Issue #171)."""
    from uclone_x.ontology.engine import OntologyEngine
    from uclone_x.ontology.models import OntologyAxiom, OntologyTier

    ontology = OntologyEngine()
    axiom = OntologyAxiom(
        name="A2A_ZeroBroker_Fastpath",
        subject_entity="Agent",
        tier=OntologyTier.ASSERTED,
        description="Co-located agents communicate in-process",
        predicate="transport_mode",
        object_value="in_memory_fastpath",
    )
    ontology.register_axiom(axiom)

    bus = EventBus()
    mock_llm = MockLLMConnector(responses=["Ontology aware reply"])
    session_mgr = AgentSessionManager(bus=bus, llm=mock_llm)

    # Initialize agent with ontology
    agent = await session_mgr.get_or_create_agent(
        agent_id="agent-ont-test",
        system_prompt="Custom system instructions for ontology testing",
    )
    agent._ontology = ontology  # Attach ontology engine

    app = create_ui_app(static_dir=tmp_path, bus=bus, session_manager=session_mgr)
    client = TestClient(app)

    res = client.post(
        "/api/turn",
        json={"message": "Ontology check prompt", "agent_id": "agent-ont-test"},
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert "debug_info" in data
    debug_info = cast(dict[str, Any], data["debug_info"])
    assert "active_invariants" in debug_info
    assert len(debug_info["active_invariants"]) >= 1
    assert any("A2A_ZeroBroker_Fastpath" in inv_str for inv_str in debug_info["active_invariants"])
    assert debug_info["prompt_tokens_used"] > 0
    assert "Custom system instructions" in debug_info["system_prompt_excerpt"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("builtin_personas_absent")
async def test_turn_events_carry_tool_executions_and_debug_info(tmp_path: Path) -> None:
    """AGENT_REPLY events published to the bus carry enriched tool_executions and debug_info (Issue #171)."""
    from unittest.mock import AsyncMock

    import httpx

    mock_tool = MagicMock(spec=ToolProtocol)
    mock_tool.name = "inspect_code"
    mock_tool.description = "Inspects code"
    mock_tool.parameters_schema = {}
    mock_tool.execute = AsyncMock(
        return_value=ToolResult(
            output={"lines": 120, "ast_clean": True},
            success=True,
            execution_time_ms=6.4,
            isolation_level=IsolationLevel.WORKSPACE,
            provenance=None,
        )
    )

    tools = ToolRegistry()
    tools.register(mock_tool)

    tool_call = ToolCallRequest(
        id="tc_inspect",
        name="inspect_code",
        arguments={"path": "main.py"},
    )
    mock_llm = MockLLMConnector(
        responses=["Inspection done."],
        tool_calls=[tool_call],
    )

    bus = EventBus()
    app = create_ui_app(static_dir=tmp_path, bus=bus, llm=mock_llm, tools=tools)
    reply_sub = bus.subscribe("agent.chat.reply")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.post(
            "/api/turn",
            json={"message": "Inspect main.py", "agent_id": "event-trace-agent"},
        )
    assert res.status_code == 200

    reply_event = await asyncio.wait_for(reply_sub.get(), timeout=2.0)
    assert reply_event.type is EventType.AGENT_REPLY
    payload = reply_event.payload
    assert "tool_executions" in payload
    tool_execs = cast(list[dict[str, Any]], payload["tool_executions"])
    assert len(tool_execs) == 1
    assert tool_execs[0]["tool_name"] == "inspect_code"
    assert tool_execs[0]["tool_call_id"] == "tc_inspect"
    assert tool_execs[0]["status"] == "success"
    assert tool_execs[0]["output"] == {"lines": 120, "ast_clean": True}
    assert tool_execs[0]["duration_ms"] == 6.4

    assert "debug_info" in payload
    debug_info = cast(dict[str, Any], payload["debug_info"])
    assert "prompt_tokens_used" in debug_info
    assert "system_prompt_excerpt" in debug_info


@pytest.mark.asyncio
async def test_ui_chat_successful_persist_reports_durability(tmp_path: Path) -> None:
    """Normal chat turn reports durability.persisted=True and degraded=False (P6, #247)."""
    mock_llm = MockLLMConnector(responses=["Everything is fine."])
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.post(
            "/api/turn",
            json={"message": "hello", "agent_id": "dur-agent", "session_id": "sess_dur_ok"},
        )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["status"] == "success"
    assert data["response"] == "Everything is fine."
    assert data["provenance"]["degraded"] is False
    assert "durability" in data
    assert data["durability"]["persisted"] is True
    assert data["durability"]["error"] is None
    assert data["durability"]["stale_conflict"] is False


@pytest.mark.asyncio
async def test_ui_chat_generic_persist_failure_reports_degraded_and_durability(
    tmp_path: Path,
) -> None:
    """A failed Core persist is visible in chat response as degraded and durability.persisted=False (#247)."""
    from uclone_x.agent.base import BaseAgent

    mock_llm = MockLLMConnector(responses=["Model replied successfully."])
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm)

    with patch.object(BaseAgent, "persist_session", side_effect=RuntimeError("Disk I/O error")):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post(
                "/api/turn",
                json={"message": "test", "agent_id": "dur-agent", "session_id": "sess_dur_err"},
            )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["status"] == "warning"
    assert data["response"] == "Model replied successfully."
    assert data["provenance"]["degraded"] is True
    assert "durability" in data
    assert data["durability"]["persisted"] is False
    assert "Disk I/O error" in str(data["durability"]["error"])
    assert data["durability"]["error_type"] == "RuntimeError"
    assert data["durability"]["stale_conflict"] is False


@pytest.mark.asyncio
async def test_ui_chat_stale_session_write_reports_stale_conflict_durability(
    tmp_path: Path,
) -> None:
    """StaleSessionWriteError during persist is distinguishable in durability metadata (#240, #247)."""
    from uclone_x.agent.base import BaseAgent
    from uclone_x.agent.session import SessionState

    mock_llm = MockLLMConnector(responses=["Contended turn reply."])
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm)

    stale_exc = StaleSessionWriteError(
        "stale write",
        session_id="sess_contended",
        expected_revision=1,
        actual_revision=2,
        current=SessionState(session_id="sess_contended", agent_id="dur-agent"),
    )
    with patch.object(BaseAgent, "persist_session", side_effect=stale_exc):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post(
                "/api/turn",
                json={
                    "message": "contended write",
                    "agent_id": "dur-agent",
                    "session_id": "sess_contended",
                },
            )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["status"] == "warning"
    assert data["response"] == "Contended turn reply."
    assert data["provenance"]["degraded"] is True
    assert "durability" in data
    assert data["durability"]["persisted"] is False
    assert data["durability"]["error_type"] == "StaleSessionWriteError"
    assert data["durability"]["stale_conflict"] is True


@pytest.mark.asyncio
async def test_get_settings_endpoint(tmp_path: Path) -> None:
    """GET /api/settings returns active endpoints and masked credentials (#350)."""
    mock_llm = MockLLMConnector(api_key="sk-abcdef123456", base_url="http://mock-llm.invalid:8000")
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.get("/api/settings")

    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["llm_provider"] == "mock"
    assert data["llm_base_url"] == "http://mock-llm.invalid:8000"
    assert data["llm_api_key_set"] is True
    assert data["llm_api_key_masked"] == "sk-...3456"
    assert "sk-abcdef123456" not in json.dumps(data)
    assert "comfyui_base_url" in data
    assert "providers_available" in data
    assert "mock" in data["providers_available"]
    assert "available_models" in data
    assert "mock-llm" in data["available_models"]


@pytest.mark.asyncio
async def test_get_models_endpoint(tmp_path: Path) -> None:
    """GET /api/models returns enumerated models for active provider (P0/Recognition over Recall)."""
    mock_llm = MockLLMConnector()
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.get("/api/models")

    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["provider"] == "mock"
    assert "models" in data
    assert "mock-llm" in data["models"]
    assert "current_model" in data


@pytest.mark.asyncio
async def test_pull_model_endpoint_success(tmp_path: Path) -> None:
    """POST /api/models/pull installs a model via the Ollama connector."""
    mock_llm = MockLLMConnector()
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, storage_dir=tmp_path)

    with patch("uclone_x.ui.app.pull_model", new=AsyncMock(return_value=None)) as mock_pull:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post("/api/models/pull", json={"model": "llama3.2:1b"})

    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    # `joined` says whether this caller did the work or attached to a pull that was
    # already running (#1233). Part of the contract, so it is asserted here too.
    assert data == {"status": "ok", "model": "llama3.2:1b", "joined": False}
    mock_pull.assert_awaited_once()
    assert mock_pull.await_args is not None
    assert mock_pull.await_args.args[0] == "llama3.2:1b"


@pytest.mark.asyncio
async def test_pull_model_endpoint_requires_model_name(tmp_path: Path) -> None:
    """POST /api/models/pull 400s when model is missing or blank."""
    mock_llm = MockLLMConnector()
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.post("/api/models/pull", json={"model": "  "})

    assert res.status_code == 400


@pytest.mark.asyncio
async def test_pull_model_endpoint_reports_provider_error_as_502(tmp_path: Path) -> None:
    """POST /api/models/pull surfaces an LLMProviderError as a 502 with its message (P6)."""
    mock_llm = MockLLMConnector()
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, storage_dir=tmp_path)

    failing_pull = AsyncMock(side_effect=LLMProviderError("Failed to connect to Ollama: refused"))
    with patch("uclone_x.ui.app.pull_model", new=failing_pull):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post("/api/models/pull", json={"model": "llama3.2:1b"})

    assert res.status_code == 502
    assert "Failed to connect to Ollama" in res.json()["detail"]


async def _yield_until(predicate: Callable[[], bool], *, yields: int = 2000) -> None:
    """Give the loop `yields` chances to make `predicate` true, then fail (#1233).

    Every step being waited for here is an in-memory ASGI await, so the number of
    them is fixed and does not grow with machine load. A wall-clock sleep in its
    place would be the thing the concurrency tests below exist to avoid.
    """
    for _ in range(yields):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"predicate never became true within {yields} event-loop yields")


@pytest.mark.asyncio
async def test_pull_route_keeps_no_wall_clock_ceiling_of_its_own(tmp_path: Path) -> None:
    """The route used to need its own ceiling. It does not any more, and that is #1243.

    `PULL_ROUTE_TIMEOUT_SECONDS = 900.0` existed because this route holds a socket
    and `ucx llm pull` does not, so it could not afford the connector's 1800 s. Both
    numbers measured elapsed time, which cannot tell a slow pull from a wedged one —
    splitting a ceiling that measures the wrong quantity only buys two callers two
    different wrong answers. `pull_model` now gives up on silence between progress
    lines, which is the same right answer for every caller, so the route passes no
    deadline at all.

    Asserted as "passes no deadline" rather than "passes the connector's number",
    because the second would still pass if the route reinstated a wall-clock ceiling
    that happened to equal one of the defaults.

    Killed by: src/uclone_x/ui/app.py :: return pull_model(model, base_url=base_url)
    Becomes: return pull_model(model, base_url=base_url, silence_timeout=900.0)
    """
    from uclone_x.ui import app as ui_app_module

    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)
    mock_pull = AsyncMock(return_value=None)

    with patch("uclone_x.ui.app.pull_model", new=mock_pull):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post("/api/models/pull", json={"model": "llama3.2:1b"})

    assert res.status_code == 200
    assert mock_pull.await_args is not None
    passed = mock_pull.await_args.kwargs
    assert "silence_timeout" not in passed, "the route has reinstated a ceiling of its own"
    assert "total_timeout" not in passed
    assert "timeout" not in passed
    assert not hasattr(ui_app_module, "PULL_ROUTE_TIMEOUT_SECONDS")


@pytest.mark.asyncio
async def test_pull_route_reports_a_stalled_pull_as_504_that_says_it_stalled(
    tmp_path: Path,
) -> None:
    """A refused daemon and a daemon that went quiet are different answers.

    Both used to be 502 carrying `Failed to connect to Ollama`, which told a user
    whose daemon was running and still downloading to go and start their daemon. So
    this answers 504 and names the path that has no ceiling — a refusal states its
    cause *and* its remedy.

    **What the operator reads has to have changed too (#1243).** A 504 saying only
    `did not finish within 900s` names a number nobody chose and describes a
    measurement the server no longer takes. The cause the connector now reports —
    the stream went quiet, and here is the last thing it said — is carried through
    to the response body intact rather than replaced by the route's own summary.

    Killed by: src/uclone_x/ui/app.py :: except LLMTimeoutError as exc:
    Becomes: except LLMProviderNotConfiguredError as exc:
    """
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)
    stalled = AsyncMock(
        side_effect=LLMTimeoutError(
            "Ollama stopped sending pull progress for 'qwen3:8b': nothing for 120s, and "
            "last status was 'pulling 797b70c4edf8'. This is a stalled pull, not a slow "
            "one — a pull that is merely slow keeps reporting.",
            seconds=120.0,
        )
    )

    with patch("uclone_x.ui.app.pull_model", new=stalled):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post("/api/models/pull", json={"model": "qwen3:8b"})

    assert res.status_code == 504
    detail = cast(str, res.json()["detail"])
    assert "stopped sending pull progress" in detail
    assert "last status was 'pulling 797b70c4edf8'" in detail
    assert "ucx llm pull qwen3:8b" in detail
    assert "within 900s" not in detail


@pytest.mark.asyncio
async def test_two_pulls_of_one_model_in_flight_together_reach_the_daemon_once(
    tmp_path: Path,
) -> None:
    """A second request for a model already downloading joins it instead of racing it.

    Nothing bounded concurrent pulls before #1233: a double-click, a second tab, or
    an impatient retry each started its own multi-gigabyte fetch of the same
    weights. The route is the seam that has to hold that, not the connector — a
    pull started from a terminal is a different process and is not this route's to
    bound.

    **The first pull is held open across the second request.** A version of this
    test that let the first finish would measure nothing: the second would arrive
    to an empty table and start its own run, which is what the unfixed code does,
    and the test would pass against it. `joined` in each response says which side
    of the de-duplication that caller landed on, and both are asserted — a second
    caller that had started its own run would answer `false`.

    Killed by: src/uclone_x/ui/app.py :: started = await pull_single_flight.run(model, pull)
    Becomes: started = await pull()
    """
    from uclone_x.ui.single_flight import SingleFlight

    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)
    flight = cast(SingleFlight, app.state.pull_single_flight)
    release = asyncio.Event()
    reached_daemon: list[str] = []

    async def held_pull(model: str, **kwargs: Any) -> None:
        reached_daemon.append(model)
        await release.wait()

    with patch("uclone_x.ui.app.pull_model", new=held_pull):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            first = asyncio.create_task(client.post("/api/models/pull", json={"model": "qwen3:8b"}))
            await _yield_until(lambda: flight.in_flight("qwen3:8b"))

            second = asyncio.create_task(
                client.post("/api/models/pull", json={"model": "qwen3:8b"})
            )
            await _yield_until(lambda: flight.joins == 1)

            assert reached_daemon == ["qwen3:8b"]  # the joiner built no request of its own

            release.set()
            res_first, res_second = await asyncio.gather(first, second)

    assert reached_daemon == ["qwen3:8b"]
    assert res_first.json() == {"status": "ok", "model": "qwen3:8b", "joined": False}
    assert res_second.json() == {"status": "ok", "model": "qwen3:8b", "joined": True}


@pytest.mark.asyncio
async def test_delete_model_endpoint_success(tmp_path: Path) -> None:
    """POST /api/models/delete removes a model via the Ollama connector."""
    mock_llm = MockLLMConnector()
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, storage_dir=tmp_path)

    with patch("uclone_x.ui.app.delete_model", new=AsyncMock(return_value=None)) as mock_delete:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post("/api/models/delete", json={"model": "llama3.2:1b"})

    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data == {"status": "ok", "model": "llama3.2:1b"}
    mock_delete.assert_awaited_once()
    assert mock_delete.await_args is not None
    assert mock_delete.await_args.args[0] == "llama3.2:1b"


@pytest.mark.asyncio
async def test_delete_model_endpoint_requires_model_name(tmp_path: Path) -> None:
    """POST /api/models/delete 400s when model is missing or blank."""
    mock_llm = MockLLMConnector()
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.post("/api/models/delete", json={})

    assert res.status_code == 400


@pytest.mark.asyncio
async def test_delete_model_endpoint_reports_provider_error_as_502(tmp_path: Path) -> None:
    """POST /api/models/delete surfaces an LLMProviderError as a 502 with its message (P6)."""
    mock_llm = MockLLMConnector()
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, storage_dir=tmp_path)

    failing_delete = AsyncMock(
        side_effect=LLMProviderError("Ollama provider returned status 404: not found")
    )
    with patch("uclone_x.ui.app.delete_model", new=failing_delete):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post("/api/models/delete", json={"model": "llama3.2:1b"})

    assert res.status_code == 502
    assert "status 404" in res.json()["detail"]


@pytest.mark.asyncio
async def test_pull_model_endpoint_is_refused_cross_origin(tmp_path: Path) -> None:
    """`ucx ui` listens on loopback with no auth and `allow_origins=["*"]`.

    Unguarded, a page open in any other tab could `fetch` this route and make the
    host download weights of that page's choosing — Ollama's `/api/pull` accepts
    `hf.co/<owner>/<repo>:<quant>`, and whatever lands shows up in the model
    picker afterwards. The refusal has to be server-side: the frontend's own
    spinner and confirmation are not in a cross-origin caller's path.

    `/api/diagnostics/consent` is the control: it was already guarded, so a run
    where it answers anything but 403 is measuring the harness, not the routes.

    Killed by: src/uclone_x/ui/app.py :: _refuse_cross_origin(request)  # a page in another tab must not install weights
    Becomes: pass
    """
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)
    evil = {"Origin": "https://evil.example.com"}

    with patch("uclone_x.ui.app.pull_model", new=AsyncMock(return_value=None)) as mock_pull:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post(
                "/api/models/pull", json={"model": "hf.co/evil/weights:Q4_K_M"}, headers=evil
            )
            control = await client.post(
                "/api/diagnostics/consent", json={"collect": True}, headers=evil
            )

    assert res.status_code == 403
    assert control.status_code == 403
    # The status code alone would pass on a route that refused *after* pulling.
    mock_pull.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_model_endpoint_is_refused_cross_origin(tmp_path: Path) -> None:
    """The same hole, pointed at the user's existing weights instead of new ones.

    A single cross-origin `POST {"model": ...}` would delete a local model; the
    `window.confirm` in `SettingsModal.tsx` is client-side and never runs for a
    caller that is not the dashboard. `/api/diagnostics/consent` is the control.

    Killed by: src/uclone_x/ui/app.py :: _refuse_cross_origin(request)  # a page in another tab must not delete weights
    Becomes: pass
    """
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)
    evil = {"Origin": "https://evil.example.com"}

    with patch("uclone_x.ui.app.delete_model", new=AsyncMock(return_value=None)) as mock_delete:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post(
                "/api/models/delete", json={"model": "llama3.2:1b"}, headers=evil
            )
            control = await client.post(
                "/api/diagnostics/consent", json={"collect": True}, headers=evil
            )

    assert res.status_code == 403
    assert control.status_code == 403
    mock_delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_routes_admit_the_dashboards_own_dev_server(tmp_path: Path) -> None:
    """`vite.config.ts` proxies `/api` with `changeOrigin`, so the dashboard in
    development arrives with a loopback `Origin` against a rewritten `Host`.

    Reusing `_refuse_cross_origin` rather than writing a strict origin/host
    comparison here is what keeps that working; nothing remote can forge a
    loopback origin.
    """
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)
    vite = {"Origin": "http://localhost:5173", "Host": "localhost:5180"}

    with (
        patch("uclone_x.ui.app.pull_model", new=AsyncMock(return_value=None)),
        patch("uclone_x.ui.app.delete_model", new=AsyncMock(return_value=None)),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            pulled = await client.post(
                "/api/models/pull", json={"model": "llama3.2:1b"}, headers=vite
            )
            deleted = await client.post(
                "/api/models/delete", json={"model": "llama3.2:1b"}, headers=vite
            )

    assert pulled.status_code == 200
    assert deleted.status_code == 200


@pytest.mark.asyncio
@pytest.mark.usefixtures("builtin_personas_absent")
async def test_update_settings_hot_reloads_runtime_and_publishes_event(tmp_path: Path) -> None:
    """POST /api/settings hot-reloads active agent LLM and ComfyUI, broadcasts event (#350)."""
    from uclone_x.tools.builtin.comfy_image_tool import ComfyImageGenTool

    event_bus = EventBus()
    sub = event_bus.subscribe("settings")

    tool = ComfyImageGenTool(base_url="http://127.0.0.1:8188")
    registry = ToolRegistry(tools=[tool])
    initial_llm = MockLLMConnector(responses=["Initial reply"])

    app = create_ui_app(
        static_dir=tmp_path,
        bus=event_bus,
        llm=initial_llm,
        tools=registry,
        storage_dir=tmp_path,
        fallback_to_mock=True,
    )

    session_mgr: AgentSessionManager = app.state.session_manager
    agent = await session_mgr.get_or_create_agent(agent_id="test-agent", session_id="sess_test")
    assert agent.llm is initial_llm

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.post(
            "/api/settings",
            json={
                "llm_provider": "mock",
                "llm_base_url": "http://updated-mock:9000",
                "llm_model": "test-model-reload",
                "llm_api_key": "sk-new-super-secret-key",
                "comfyui_base_url": "http://comfy-gpu:8188",
            },
        )

    assert res.status_code == 200
    updated = cast(dict[str, Any], res.json())
    assert updated["llm_provider"] == "mock"
    assert updated["llm_base_url"] == "http://updated-mock:9000"
    assert updated["llm_model"] == "test-model-reload"
    assert updated["comfyui_base_url"] == "http://comfy-gpu:8188"
    assert updated["llm_api_key_set"] is True

    # 1. Hot-reload verified: agent._llm has been swapped without restart
    assert agent.llm is not initial_llm
    assert agent.llm is session_mgr.llm
    assert agent.config.llm_config.model_name == "test-model-reload"

    # 2. Tool hot-reload verified: ComfyUI base_url updated
    assert tool.base_url == "http://comfy-gpu:8188"

    # 3. Event broadcast verified: settings.updated received on event bus
    event = await asyncio.wait_for(sub.get(), timeout=2.0)
    assert event.type is EventType.SETTINGS_UPDATED
    assert event.topic == "settings"
    assert event.payload["llm_model"] == "test-model-reload"
    assert event.payload["comfyui_base_url"] == "http://comfy-gpu:8188"


@pytest.mark.asyncio
async def test_update_settings_invalid_provider_returns_400(tmp_path: Path) -> None:
    """POST /api/settings with unsupported provider returns 400."""
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=False, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.post(
            "/api/settings",
            json={"llm_provider": "unsupported-cloud-unknown"},
        )
    assert res.status_code == 400
    data = cast(dict[str, Any], res.json())
    assert "Unsupported LLM provider" in data["detail"]


def test_update_settings_failed_connector_does_not_poison_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If connector creation fails during update_settings, os.environ['LLM_PROVIDER'] is not modified (#402, #410).

    Mutation this exists to catch:
        -   new_llm = create_llm_connector(...)
        -   for k, v in env_updates.items():
        -       os.environ[k] = v
        +   for k, v in env_updates.items():
        +       os.environ[k] = v
        +   new_llm = create_llm_connector(...)
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=False)
    assert os.getenv("LLM_PROVIDER") is None

    with pytest.raises(LLMCredentialsNotConfiguredError):
        mgr.update_settings(llm_provider="openai")

    assert os.getenv("LLM_PROVIDER") is None
    assert mgr._configured_provider != "openai"

    # Also verify that a pre-existing LLM_PROVIDER is preserved rather than overwritten on failure
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    mgr2 = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=False)
    assert os.getenv("LLM_PROVIDER") == "mock"

    with pytest.raises(LLMCredentialsNotConfiguredError):
        mgr2.update_settings(llm_provider="openai")

    assert os.getenv("LLM_PROVIDER") == "mock"
    assert mgr2._configured_provider != "openai"


@pytest.mark.asyncio
async def test_settings_persistence_across_manager_instances(tmp_path: Path) -> None:
    """Settings saved in one session manager instance persist to disk and rehydrate in another (#350)."""
    mgr1 = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    mgr1.update_settings(
        llm_provider="mock",
        llm_base_url="http://persisted-mock:8888",
        llm_model="persisted-model-v1",
        comfyui_base_url="http://persisted-comfy:8188",
    )

    # Instantiate fresh AgentSessionManager with the same storage directory
    mgr2 = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    settings2 = mgr2.get_settings()
    assert settings2["llm_provider"] == "mock"
    assert settings2["llm_base_url"] == "http://persisted-mock:8888"
    assert settings2["llm_model"] == "persisted-model-v1"
    assert settings2["comfyui_base_url"] == "http://persisted-comfy:8188"


@pytest.mark.asyncio
async def test_persisted_settings_initializes_active_llm_connector_on_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AgentSessionManager initializes active LLM connector and environment from settings.json (#891)."""
    # Ensure no ambient LLM env vars
    for k in ("LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "llm_provider": "ollama",
                "llm_base_url": "http://127.0.0.1:11434",
                "llm_model": "hermes3:8b",
            }
        ),
        encoding="utf-8",
    )

    # Initialize manager without explicit LLM or fallback_to_mock
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=False)

    # Active LLM connector should be initialized from settings.json
    assert mgr.default_llm is not None
    assert getattr(mgr.default_llm, "provider_name", "") == "ollama"
    assert getattr(mgr.default_llm, "base_url", "") == "http://127.0.0.1:11434"
    assert os.getenv("LLM_PROVIDER") == "ollama"
    assert os.getenv("OLLAMA_BASE_URL") == "http://127.0.0.1:11434"
    assert os.getenv("OLLAMA_MODEL") == "hermes3:8b"

    # get_or_create_agent should successfully resolve the LLM without raising LLMProviderNotConfiguredError
    agent = await mgr.get_or_create_agent("scout", session_id="test-session-891")
    assert agent is not None
    assert agent.llm == mgr.default_llm


@pytest.mark.asyncio
async def test_test_endpoint_connection(tmp_path: Path) -> None:
    """POST /api/settings/test verifies mock and ComfyUI connectivity diagnostics (#350)."""
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=True, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        # Test mock LLM connection
        res_llm = await client.post(
            "/api/settings/test",
            json={"target": "llm", "llm_provider": "mock"},
        )
        assert res_llm.status_code == 200
        data_llm = cast(dict[str, Any], res_llm.json())
        assert data_llm["status"] == "ok"
        assert data_llm["results"]["llm"]["status"] == "ok"

        # Test ComfyUI connection (unreachable port)
        res_comfy = await client.post(
            "/api/settings/test",
            json={"target": "comfyui", "comfyui_base_url": "http://127.0.0.1:59999"},
        )
        assert res_comfy.status_code == 200
        data_comfy = cast(dict[str, Any], res_comfy.json())
        assert data_comfy["results"]["comfyui"]["online"] is False


def test_diagnose_vite_accepts_our_own_dev_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identity, not liveness: the body must be this project's index.html."""
    import httpx as _httpx

    from uclone_x.ui import server as srv

    def fake_get(url: str, timeout: float = 2.0) -> _httpx.Response:
        return _httpx.Response(200, text="<title>UClone-X Dashboard</title>")

    monkeypatch.setattr(_httpx, "get", fake_get)
    assert srv._diagnose_vite("127.0.0.1", 5173) is None


def test_diagnose_vite_names_the_other_application_holding_the_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dev server answering is not evidence that it is *our* dev server.

    Measured on a developer machine: another project's Vite held `*:5173` while this
    one held `127.0.0.1:5173`, so `localhost:5173` served the other project and HMR
    looked broken. Reporting "HMR active" there is an assertion independent of the
    outcome.

    Mutation this exists to catch: return `None` for any 200 response.
    """
    import httpx as _httpx

    from uclone_x.ui import server as srv

    def fake_get(url: str, timeout: float = 2.0) -> _httpx.Response:
        return _httpx.Response(200, text="<html><title>Hexworld Deck Studio</title></html>")

    monkeypatch.setattr(_httpx, "get", fake_get)
    problem = srv._diagnose_vite("127.0.0.1", 5173)
    assert problem is not None
    assert "Hexworld Deck Studio" in problem, problem
    assert "--vite-port" in problem, problem


def test_diagnose_vite_reports_when_nothing_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Vite that died on startup is reported rather than announced as active.

    Mutation this exists to catch: swallow the transport error and return `None`.
    """
    import httpx as _httpx

    from uclone_x.ui import server as srv

    def fake_get(url: str, timeout: float = 2.0) -> _httpx.Response:
        raise _httpx.ConnectError("connection refused")

    monkeypatch.setattr(_httpx, "get", fake_get)
    problem = srv._diagnose_vite("127.0.0.1", 5173, timeout_s=0.5)
    assert problem is not None
    assert "no server answered" in problem, problem


def test_unconfigured_provider_is_translated_as_a_configuration_error_not_a_session_one() -> None:
    """The refusal must not be reported as a session fault.

    `LLMProviderNotConfiguredError` reaches the UI through the same broad handlers that
    wrap session loading, so #539 left it labelled `SESSION_LOAD_ERROR` with component
    `uclone_x.agent.session` and a bare 500 from `_translate_session_error`. Naming the
    wrong subsystem is the mis-attribution P6 forbids: the record says where to look and
    points at the wrong place.
    """
    from uclone_x.errors import LLMProviderNotConfiguredError
    from uclone_x.ui.app import _translate_session_error

    exc = _translate_session_error(LLMProviderNotConfiguredError("No LLM provider is configured."))
    assert exc.status_code == 503, "an unconfigured provider is not a 500 server fault"
    assert "provider" in str(exc.detail).lower()


def test_unconfigured_provider_is_recognised_as_an_llm_condition() -> None:
    """The offline-diagnostic path must recognise it.

    `_is_offline_llm_error` gates `OFFLINE_LLM_DIAGNOSTIC_MESSAGE`, which is the only text
    in the product naming `ollama serve` and `./ucx llm status`. None of its keywords match
    "No LLM provider is configured", so #539 made the UI message *less* actionable than the
    one it displaced.
    """
    from uclone_x.errors import LLMProviderNotConfiguredError
    from uclone_x.ui.app import _is_offline_llm_error

    assert _is_offline_llm_error(LLMProviderNotConfiguredError("No LLM provider is configured."))


def test_ui_artifacts_endpoints(tmp_path: Path) -> None:
    """Assert /api/artifacts lists only generated artifacts, and /api/artifacts/content streams them (RFC §6.1)."""
    # 1. Setup workspace with markdown files
    docs_dir = tmp_path / "docs" / "design"
    docs_dir.mkdir(parents=True, exist_ok=True)
    rfc_file = docs_dir / "rfc.md"
    rfc_file.write_text(
        "# Design RFC: User-Centric Refactoring\n\nContent of the RFC.\n", encoding="utf-8"
    )

    art_dir = tmp_path / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    report_file = art_dir / "report.md"
    report_file.write_text("## Unheaded Report\n\nReport details.\n", encoding="utf-8")

    readme_file = tmp_path / "README.md"
    readme_file.write_text("# Project UClone-X\n\nREADME text.\n", encoding="utf-8")

    # Session-specific artifact
    session_tool_dir = tmp_path / ".sandbox" / "tool_artifacts" / "sess_test_1"
    session_tool_dir.mkdir(parents=True, exist_ok=True)
    tool_art = session_tool_dir / "output.md"
    tool_art.write_text("# Tool Output Analysis\n\nTool run results.\n", encoding="utf-8")

    mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", workspace_dir=tmp_path)
    app = create_ui_app(session_manager=mgr)
    client = TestClient(app)

    # A. Enumerate all artifacts
    res = client.get("/api/artifacts")
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert "artifacts" in data
    assert "total" in data
    paths = {a["path"] for a in data["artifacts"]}
    assert "artifacts/report.md" in paths
    # The workspace's own docs are not artifacts: no clone generated them.
    assert "docs/design/rfc.md" not in paths
    assert "README.md" not in paths

    # Verify structured metadata fields
    report_item = next(a for a in data["artifacts"] if a["path"] == "artifacts/report.md")
    assert report_item["title"] == "Report"
    assert report_item["id"].startswith("art_")
    assert report_item["size_bytes"] > 0
    assert "created_at" in report_item
    assert "modified_at" in report_item

    # B. Enumerate with session_id
    res_sess = client.get("/api/artifacts?session_id=sess_test_1")
    assert res_sess.status_code == 200
    data_sess = cast(dict[str, Any], res_sess.json())
    sess_paths = {a["path"] for a in data_sess["artifacts"]}
    assert ".sandbox/tool_artifacts/sess_test_1/output.md" in sess_paths
    assert "docs/design/rfc.md" not in sess_paths
    tool_item = next(
        a
        for a in data_sess["artifacts"]
        if a["path"] == ".sandbox/tool_artifacts/sess_test_1/output.md"
    )
    assert tool_item["title"] == "Tool Output Analysis"

    # C. Fetch content (raw text)
    res_content = client.get("/api/artifacts/content?path=docs/design/rfc.md")
    assert res_content.status_code == 200
    assert "text/markdown" in res_content.headers.get("content-type", "")
    assert res_content.text == "# Design RFC: User-Centric Refactoring\n\nContent of the RFC.\n"

    # D. Fetch content (JSON accept header)
    res_json = client.get(
        "/api/artifacts/content?path=docs/design/rfc.md",
        headers={"accept": "application/json"},
    )
    assert res_json.status_code == 200
    json_data = cast(dict[str, Any], res_json.json())
    assert json_data["path"] == "docs/design/rfc.md"
    assert json_data["content"] == "# Design RFC: User-Centric Refactoring\n\nContent of the RFC.\n"

    # E. Fetch image content (returns Cache-Control: no-cache header)
    img_path = tmp_path / "artifacts" / "images" / "test.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")
    res_img = client.get("/api/artifacts/content?path=artifacts/images/test.png")
    assert res_img.status_code == 200
    assert "image/png" in res_img.headers.get("content-type", "")
    assert "no-cache" in res_img.headers.get("cache-control", "")


def test_ui_artifacts_path_traversal_rejection(tmp_path: Path) -> None:
    """Assert /api/artifacts/content strictly rejects path traversal (P6 security invariant).

    Killed by: src/uclone_x/ui/app.py :: resolved = validator.resolve_safe_path(Path(clean_path), self._workspace_dir)
    Becomes: resolved = Path(clean_path).resolve()
    """
    mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", workspace_dir=tmp_path)
    app = create_ui_app(session_manager=mgr)
    client = TestClient(app)

    # 1. Upward directory traversal
    r1 = client.get("/api/artifacts/content?path=../../etc/passwd")
    assert r1.status_code == 400
    assert (
        "escapes workspace root" in r1.json()["detail"]
        or "traversal" in r1.json()["detail"].lower()
    )

    # 2. Absolute path escaping workspace
    r2 = client.get("/api/artifacts/content?path=/etc/passwd")
    assert r2.status_code == 400

    # 3. Empty path
    r3 = client.get("/api/artifacts/content?path=")
    assert r3.status_code == 400

    # 4. Null byte injection
    r4 = client.get("/api/artifacts/content?path=docs/test%00.md")
    assert r4.status_code == 400

    # 5. Non-existent file within workspace
    r5 = client.get("/api/artifacts/content?path=docs/does_not_exist.md")
    assert r5.status_code == 404


def test_ui_knowledge_graph_endpoint(tmp_path: Path) -> None:
    """Assert /api/knowledge-graph returns dynamic triples and Force Graph network (RFC §6.1).

    Killed by: src/uclone_x/ui/knowledge.py :: "subject": r.source_entity,
    Becomes: "subject": "mutated_subject",
    """
    engine = OntologyEngine(agent_id="test-agent")
    engine.register_entity(
        OntologyConcept(
            name="ArtifactsDock",
            parent_type="SubpanelView",
            tier=OntologyTier.ASSERTED,
            attributes={"version": "string"},
            required_fields=("version",),
        )
    )
    engine.register_entity(
        OntologyConcept(
            name="DynamicGraph",
            tier=OntologyTier.INDUCED_ENFORCING,
            attributes={"node_count": "int"},
            required_fields=("node_count",),
        )
    )
    engine.induce_relation(
        source_entity="ArtifactsDock",
        predicate="renders",
        target_entity="DynamicGraph",
        source_session="sess_kg_1",
        confidence=0.95,
    )
    engine.register_axiom(
        OntologyAxiom(
            name="dock_has_max_nodes",
            subject_entity="DynamicGraph",
            predicate="node_count <= 150",
            object_value="150",
            rule_expression="node_count <= 150",
            description="Max rendered nodes capped at 150",
        )
    )

    mgr = AgentSessionManager(
        storage_dir=tmp_path / "sessions",
        workspace_dir=tmp_path,
        ontology_engine=engine,
    )
    app = create_ui_app(session_manager=mgr)
    client = TestClient(app)

    # 1. Unfiltered query
    res = client.get("/api/knowledge-graph")
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert "triples" in data
    assert "nodes" in data
    assert "edges" in data
    assert "summary" in data

    triples = data["triples"]
    assert len(triples) >= 2

    # Check relation triple
    rel_triple = next((t for t in triples if t["predicate"] == "renders"), None)
    assert rel_triple is not None
    assert rel_triple["subject"] == "ArtifactsDock"
    assert rel_triple["object"] == "DynamicGraph"
    assert rel_triple["provenance"]["source_session"] == "sess_kg_1"
    assert rel_triple["provenance"]["confidence"] == 0.95
    assert rel_triple["provenance"]["origin"] == "derived"

    # Check concept parent_type triple (is_a)
    isa_triple = next((t for t in triples if t["predicate"] == "is_a"), None)
    assert isa_triple is not None
    assert isa_triple["subject"] == "ArtifactsDock"
    assert isa_triple["object"] == "SubpanelView"

    # Check axiom triple
    ax_triple = next(
        (
            t
            for t in triples
            if t["subject"] == "DynamicGraph" and t["predicate"] == "node_count <= 150"
        ),
        None,
    )
    assert ax_triple is not None
    assert ax_triple["object"] == "150"

    # Check nodes and edges
    node_ids = {n["id"] for n in data["nodes"]}
    assert "ArtifactsDock" in node_ids
    assert "DynamicGraph" in node_ids
    assert "SubpanelView" in node_ids

    # 2. Session filter: matching session
    res_sess = client.get("/api/knowledge-graph?session_id=sess_kg_1")
    assert res_sess.status_code == 200
    data_sess = cast(dict[str, Any], res_sess.json())
    sess_triples = data_sess["triples"]
    assert any(t["predicate"] == "renders" for t in sess_triples)

    # Session filter: other session
    res_other = client.get("/api/knowledge-graph?session_id=sess_different")
    assert res_other.status_code == 200
    data_other = cast(dict[str, Any], res_other.json())
    assert not any(t["predicate"] == "renders" for t in data_other["triples"])

    # 3. Agent filter
    res_agent = client.get("/api/knowledge-graph?agent_id=test-agent")
    assert res_agent.status_code == 200
    data_agent = cast(dict[str, Any], res_agent.json())
    assert data_agent["summary"]["total_triples"] > 0


def test_chat_turn_respects_and_logs_requested_model(test_client: TestClient) -> None:
    """Verify that chat turn respects requested model override and includes model in turn result."""
    payload = {
        "message": "Hello from hermes test",
        "agent_id": "champion",
        "model": "hermes3:8b",
    }
    response = test_client.post("/api/turn", json=payload)
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["status"] == "success"
    assert data["model"] == "hermes3:8b"
    assert data["agent_id"] == "champion"


def test_chat_turn_propagates_model_not_found_error_without_silent_replacement(
    tmp_path: Path,
) -> None:
    """Verify that when a requested model fails with not found, the error is delivered directly without silent fallback to another model."""
    static_dir = tmp_path / "ui_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>Test UI</body></html>", encoding="utf-8")
    storage_dir = tmp_path / "sessions"

    def not_found_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": "model 'nonexistent-model' not found, try pulling it first"}
        )

    ollama_connector = OllamaConnector(
        base_url="http://stub-ollama.invalid:11434",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(not_found_handler)),
    )
    app = create_ui_app(static_dir=static_dir, storage_dir=storage_dir, llm=ollama_connector)
    client = TestClient(app)

    payload = {
        "message": "Hello to missing model",
        "agent_id": "champion",
        "model": "nonexistent-model",
    }
    response = client.post("/api/turn", json=payload)
    assert response.status_code == 200
    data = cast(dict[str, Any], response.json())
    assert data["status"] == "error"
    assert "model 'nonexistent-model' not found" in data["response"]


@pytest.mark.asyncio
async def test_ui_stream_ends_cleanly_when_its_subscription_closes(tmp_path: Path) -> None:
    """A closed subscription ends the stream; it does not escape the generator (#872).

    `sub.get()` raises `SubscriptionClosedError` once the bus has closed the
    subscription. Nothing caught it, so it left `event_generator` mid-response instead
    of being the ordinary end-of-stream it is.

    Killed by: src/uclone_x/ui/app.py :: except SubscriptionClosedError:
    Becomes: except ConnectionAbortedError:
    """
    from fastapi.routing import APIRoute
    from starlette.requests import Request
    from starlette.responses import StreamingResponse

    from uclone_x.engine.event_bus import EventBus

    bus = EventBus()
    app = create_ui_app(static_dir=tmp_path, bus=bus)

    closed_sub = bus.subscribe("*")
    closed_sub.close()

    def _hand_back_the_closed_subscription(*_args: object, **_kwargs: object) -> object:
        return closed_sub

    bus.subscribe = _hand_back_the_closed_subscription  # pyright: ignore[reportAttributeAccessIssue]

    route = next(r for r in app.routes if isinstance(r, APIRoute) and r.path == "/api/stream")
    mock_request = MagicMock(spec=Request)
    mock_request.app = app
    mock_request.is_disconnected = AsyncMock(return_value=False)

    response = await route.endpoint(request=mock_request)
    assert isinstance(response, StreamingResponse)

    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))

    full_text = "\n".join(chunks)
    assert "SYSTEM_CONNECTED" in full_text


@pytest.mark.asyncio
async def test_cancel_and_drain_reads_the_waiter_that_finished_before_the_break() -> None:
    """A waiter that is already `done` must still be read (#872).

    The leak is not about cancellation. Measured on CPython 3.12.13, cancelling a task
    whose awaited future has already resolved ends it **cancelled** -- `_must_cancel`
    raises `CancelledError` and replaces any pending exception -- on a bare
    `asyncio.Queue` and on a real `EventSubscription` alike. So a cancelled waiter never
    carried a `SubscriptionClosedError` for anyone to miss.

    What leaked is the waiter that never reached `pending` at all. When the subscription
    closes in the same window the shutdown event fires, `sub.get()` is woken by the close
    sentinel and completes with `SubscriptionClosedError`, so `asyncio.wait` returns it in
    **`done`**. The loop's `for t in pending: t.cancel()` therefore never touched it, and
    the `break` walked away without reading it -- which is the "Task exception was never
    retrieved" line in the issue, emitted when the task is finalized.

    This exercises the real `EventSubscription`, and never calls `.exception()` on the
    task it is making a claim about, because that call would itself retrieve it.

    Killed by: src/uclone_x/ui/app.py :: await asyncio.gather(*collected, return_exceptions=True)
    Becomes: pass
    """
    import gc

    from uclone_x.engine.event_bus import EventBus, SubscriptionClosedError
    from uclone_x.ui.app import _cancel_and_drain

    async def waiter_done_before_the_break() -> asyncio.Task[Any]:
        """Reproduce the loop's window and return the `done`, still-unread waiter."""
        bus = EventBus()
        sub = bus.subscribe()
        shutdown = asyncio.Event()

        get_task: asyncio.Task[Any] = asyncio.create_task(sub.get())
        shutdown_task = asyncio.create_task(shutdown.wait())
        await asyncio.sleep(0)  # let `get()` suspend on the queue
        sub.close()  # the close sentinel wakes it
        shutdown.set()  # ... in the same window the shutdown fires

        done, pending = await asyncio.wait(
            [get_task, shutdown_task],
            timeout=1.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # The shape the leak needs: the waiter is `done`, so cancelling `pending` and
        # breaking leaves it finished and unread.
        assert get_task in done
        assert get_task not in pending
        await _cancel_and_drain(pending)
        return get_task

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unretrieved: list[dict[str, Any]] = []

    def record_subscription_leaks(_loop: object, context: dict[str, Any]) -> None:
        """Record only our waiter's leak, not the bus's own dispatcher teardown noise."""
        if isinstance(context.get("exception"), SubscriptionClosedError):
            unretrieved.append(context)

    loop.set_exception_handler(record_subscription_leaks)
    try:
        # Positive control: without the drain, finalizing the waiter reports the leak.
        leaked = await waiter_done_before_the_break()
        del leaked
        gc.collect()
        await asyncio.sleep(0)
        assert len(unretrieved) == 1, (
            f"expected the unretrieved-exception report, got {unretrieved}"
        )
        assert isinstance(unretrieved[0].get("exception"), SubscriptionClosedError)

        # The guard: draining the `done` waiter reads it, and finalizing reports nothing.
        unretrieved.clear()
        drained = await waiter_done_before_the_break()
        await _cancel_and_drain([drained])
        del drained
        gc.collect()
        await asyncio.sleep(0)
        assert unretrieved == [], f"drained waiter still reported as unretrieved: {unretrieved}"
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_ui_stream_yields_event_when_shutdown_fires_in_same_window(tmp_path: Path) -> None:
    """An event taken by sub.get() before breaking on shutdown is delivered, not lost (#880, P6).

    When an event arrives in the same window that the shutdown event fires, get_task
    completes normally and lands in `done` alongside `shutdown_task`. The shutdown
    check previously drained the task but broke immediately without reading or yielding
    its event result, permanently discarding it.

    This test drives the race deterministically:
    1. A mock subscription returns an AgentEvent on `get()`.
    2. A custom shutdown_event is set in the same tick.
    3. The SSE stream yields SYSTEM_CONNECTED, then the AGENT_EVENT, then cleanly terminates.
    """
    from fastapi.routing import APIRoute
    from starlette.requests import Request
    from starlette.responses import StreamingResponse

    from uclone_x.engine.event_bus import AgentEvent, EventBus, EventSource, EventType

    bus = EventBus()
    app = create_ui_app(static_dir=tmp_path, bus=bus)

    test_event = AgentEvent(
        event_id="evt_test_shutdown_race_880",
        type=EventType.AGENT_REPLY,
        source=EventSource.AGENT,
        sender_id="agent-880",
        recipient_id="user",
        topic="agent.reply",
        payload={"message": "terminal reply before shutdown"},
    )

    class RaceSubscription:
        def __init__(self, shutdown_evt: asyncio.Event) -> None:
            self._shutdown_evt = shutdown_evt
            self._delivered = False

        async def get(self) -> AgentEvent:
            if not self._delivered:
                self._delivered = True
                # Set the shutdown event in the same tick as returning the event!
                self._shutdown_evt.set()
                return test_event
            # If called again, wait indefinitely
            await asyncio.Event().wait()
            raise RuntimeError("unreachable")

        def close(self) -> None:
            pass

    shutdown = asyncio.Event()
    app.state.shutdown_event = shutdown
    race_sub = RaceSubscription(shutdown)

    def _hand_back_race_sub(*_args: object, **_kwargs: object) -> object:
        return race_sub

    bus.subscribe = _hand_back_race_sub  # pyright: ignore[reportAttributeAccessIssue]

    route = next(r for r in app.routes if isinstance(r, APIRoute) and r.path == "/api/stream")
    mock_request = MagicMock(spec=Request)
    mock_request.app = app
    mock_request.is_disconnected = AsyncMock(return_value=False)

    response = await route.endpoint(request=mock_request)
    assert isinstance(response, StreamingResponse)

    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))

    full_text = "\n".join(chunks)
    # Both SYSTEM_CONNECTED and the terminal AGENT_EVENT must be present in the output
    assert "SYSTEM_CONNECTED" in full_text
    assert "evt_test_shutdown_race_880" in full_text
    assert "terminal reply before shutdown" in full_text


# --- #939: the chat turn's token figures are the ones the budget booked, labelled --------

_FIGURES_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="dummy", model="dummy"),
    served_by=ServiceRef(provider="dummy", model="dummy"),
)


def _figures_usage(input_tokens: int, output_tokens: int, source: TokenCountSource) -> TokenUsage:
    return TokenUsage(
        provider="dummy",
        model="dummy",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        count_source=source,
    )


class _BookingLLM(BaseLLMConnector):
    """Replays scripted responses, one per model step, cycling through the script."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self._responses = responses
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "dummy"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        resp = self._responses[self.calls % len(self._responses)]
        self.calls += 1
        return resp

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        yield StreamChunk(delta_content="")


def _figures_answer(usage: TokenUsage) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        content="The figures are in.",
        usage=usage,
        provenance=_FIGURES_PROV,
    )


def _figures_tool_call(usage: TokenUsage) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(ToolCallRequest(id="tc_fig", name="count_things", arguments={}),),
        usage=usage,
        provenance=_FIGURES_PROV,
    )


def _figures_tools() -> ToolRegistry:
    tool = MagicMock(spec=ToolProtocol)
    tool.name = "count_things"
    tool.description = "Counts things"
    tool.parameters_schema = {}
    tool.execute = AsyncMock(
        return_value=ToolResult(
            output={"things": 3},
            success=True,
            execution_time_ms=1.0,
            isolation_level=IsolationLevel.WORKSPACE,
            provenance=None,
        )
    )
    # Added to the default registry: a bare one fails the bundled personas' tool checks.
    registry = create_default_registry()
    registry.register(tool)
    return registry


@pytest.mark.parametrize(
    ("script", "tokens_used", "prompt_tokens_used", "source"),
    [
        pytest.param(
            [_figures_answer(_figures_usage(321, 45, TokenCountSource.PROVIDER))],
            366,
            321,
            "provider",
            id="provider-counted",
        ),
        pytest.param(
            [_figures_answer(_figures_usage(40, 9, TokenCountSource.ESTIMATE))],
            49,
            40,
            "estimate",
            id="estimated",
        ),
        pytest.param(
            [
                _figures_tool_call(_figures_usage(300, 20, TokenCountSource.PROVIDER)),
                _figures_answer(_figures_usage(350, 30, TokenCountSource.ESTIMATE)),
            ],
            700,
            650,
            "estimate",
            id="two-steps-one-estimated",
        ),
    ],
)
def test_a_chat_turn_reports_the_tokens_its_steps_booked_and_whose_count_they_are(
    tmp_path: Path,
    script: list[ModelResponse],
    tokens_used: int,
    prompt_tokens_used: int,
    source: str,
) -> None:
    """The chat response's token figures are the budget ledger's for that turn, labelled.

    They were `100 + len(reply) // 4` and `100 + len(message) // 4`: invented, reported as
    if measured, and unrelated to what the budget charged (#939 item 2). The rule is
    §6.7's: a figure the provider did not count is an estimate, labelled, and a turn with
    any estimated step is estimated as a whole. The turn is sent twice on one session, and
    the second report must hold that turn's steps only, not the session's.

    Killed by: src/uclone_x/ui/app.py :: return total, prompt, source.value
    Becomes: return 100 + total // 4, 100 + prompt // 4, source.value
    Killed by: src/uclone_x/ui/app.py :: estimated = any(u.count_source is TokenCountSource.ESTIMATE for u in usages)
    Becomes: estimated = False
    Killed by: src/uclone_x/ui/app.py :: prompt = sum(u.input_tokens for u in usages)
    Becomes: prompt = sum(u.output_tokens for u in usages)
    Killed by: src/uclone_x/ui/app.py :: token_count_source = _turn_token_figures(turn_booked)
    Becomes: token_count_source = _turn_token_figures(session_mgr.budget_tracker.get_turn_history(session_id))
    Killed by: src/uclone_x/ui/app.py :: "token_count_source": token_count_source,  # persisted
    Becomes: "token_count_source": None,  # persisted
    """
    llm = _BookingLLM(script)
    session_mgr = AgentSessionManager(bus=EventBus(), llm=llm, tools=_figures_tools())
    app = create_ui_app(static_dir=tmp_path, session_manager=session_mgr)
    client = TestClient(app)
    body = {"message": "count them", "agent_id": "figures-agent", "session_id": "sess_fig"}

    for _ in range(2):
        data = cast(dict[str, Any], client.post("/api/turn", json=body).json())
        assert data["status"] == "success", data
        assert data["tokens_used"] == tokens_used
        assert data["debug_info"]["prompt_tokens_used"] == prompt_tokens_used
        assert data["token_count_source"] == source

    saved = session_mgr.get_session_history("figures-agent", "sess_fig")[-1]
    assert saved["tokens_used"] == tokens_used
    assert saved["token_count_source"] == source


def test_a_chat_turn_that_booked_no_tokens_reports_no_figure_rather_than_zero(
    tmp_path: Path,
) -> None:
    """A turn that failed before any model step was booked has no token figure at all.

    Reported as `0` it reads as a count of nothing (#394's substitution); reported as
    `100 + len(...) // 4` it was a figure for work that never happened. `None` with no
    `token_count_source` is what the ledger holds: nothing.

    Killed by: src/uclone_x/ui/app.py :: return None, None, None
    Becomes: return 0, 0, None
    """
    failing = MagicMock(spec=LLMProviderProtocol)
    failing.provider_name = "dummy"
    failing.generate = AsyncMock(side_effect=LLMProviderError("provider exploded"))
    session_mgr = AgentSessionManager(bus=EventBus(), llm=failing)
    app = create_ui_app(static_dir=tmp_path, session_manager=session_mgr)
    client = TestClient(app)

    data = cast(
        dict[str, Any],
        client.post("/api/turn", json={"message": "hello", "agent_id": "figures-none"}).json(),
    )
    assert data["status"] != "success", data
    assert data["tokens_used"] is None
    assert data["debug_info"]["prompt_tokens_used"] is None
    assert data["token_count_source"] is None


def test_a_chat_turn_that_raises_after_booking_steps_still_reports_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn whose `execute_turn` raises reports the steps it booked first, labelled.

    The error path builds its response separately from the success path, and nothing
    else pins its figures: #981's review hard-coded them to `0` / `"provider"` and every
    test in this file still passed. Here the turn books a counted step and an estimated
    one, then raises, so the report must carry both figures and the `estimate` label.

    Killed by: src/uclone_x/ui/app.py :: "tokens_used": failed_tokens_used,
    Becomes: "tokens_used": 0,
    Killed by: src/uclone_x/ui/app.py :: "token_count_source": failed_count_source,
    Becomes: "token_count_source": "provider",
    Killed by: src/uclone_x/ui/app.py :: "prompt_tokens_used": failed_prompt_tokens_used,
    Becomes: "prompt_tokens_used": 0,
    """
    session_mgr = AgentSessionManager(bus=EventBus(), llm=MockLLMConnector())

    async def _book_two_steps_then_raise(
        self: BaseAgent, message: str, stream_callback: object = None
    ) -> TurnResult:
        session_mgr.budget_tracker.record_usage(
            "sess_raise", _figures_usage(200, 30, TokenCountSource.PROVIDER)
        )
        session_mgr.budget_tracker.record_usage(
            "sess_raise", _figures_usage(50, 5, TokenCountSource.ESTIMATE)
        )
        raise RuntimeError("the turn fell over after two steps")

    monkeypatch.setattr(BaseAgent, "execute_turn", _book_two_steps_then_raise)
    app = create_ui_app(static_dir=tmp_path, session_manager=session_mgr)
    client = TestClient(app)

    data = cast(
        dict[str, Any],
        client.post(
            "/api/turn",
            json={"message": "go", "agent_id": "figures-raise", "session_id": "sess_raise"},
        ).json(),
    )
    assert data["status"] == "error", data
    assert "fell over" in data["response"]
    assert data["tokens_used"] == 285
    assert data["debug_info"]["prompt_tokens_used"] == 250
    assert data["token_count_source"] == "estimate"


# --- #982: overlapping turns on one session id each report only the steps they booked ----

_ALPHA = "alpha turn"
_BETA = "beta turn"


class _HeldLLM(BaseLLMConnector):
    """Scripted steps per turn, found by the turn's own user message; any step can be held.

    `entered(turn, i)` is set when step `i` of `turn` reaches the model, and a step given a
    `hold(turn, i)` waits on it before answering. A test orders two turns' bookings by what
    each turn has reached, never by elapsed time (R7).
    """

    def __init__(self, scripts: dict[str, list[ModelResponse]]) -> None:
        super().__init__()
        self._scripts = scripts
        self._taken: dict[str, int] = {}
        self._entered: dict[tuple[str, int], asyncio.Event] = {}
        self._held: dict[tuple[str, int], asyncio.Event] = {}

    @property
    def provider_name(self) -> str:
        return "dummy"

    def entered(self, turn: str, index: int) -> asyncio.Event:
        return self._entered.setdefault((turn, index), asyncio.Event())

    def hold(self, turn: str, index: int) -> asyncio.Event:
        return self._held.setdefault((turn, index), asyncio.Event())

    async def generate(self, request: LLMRequest) -> ModelResponse:
        last_user = next(m for m in reversed(request.messages) if m.role == MessageRole.USER)
        turn = next(name for name in self._scripts if name in (last_user.content or ""))
        index = self._taken.get(turn, 0)
        self._taken[turn] = index + 1
        self.entered(turn, index).set()
        held = self._held.get((turn, index))
        if held is not None:
            await held.wait()
        return self._scripts[turn][index]

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        yield StreamChunk(delta_content="")


class _ContendedLock(asyncio.Lock):
    """An agent's turn lock that says when a second turn has arrived and is waiting on it."""

    def __init__(self) -> None:
        super().__init__()
        self.contended = asyncio.Event()

    async def acquire(self) -> Literal[True]:
        if self.locked():
            self.contended.set()
        return await super().acquire()


async def _reached(event: asyncio.Event) -> None:
    # A bound on a hang, not a synchronisation: every wait is on an event a turn sets.
    await asyncio.wait_for(event.wait(), timeout=30)


_Chat = Callable[[str, str], Coroutine[Any, Any, dict[str, Any]]]
_Reports = tuple[dict[str, Any], dict[str, Any]]


async def _one_agent_second_turn_sent_while_the_first_runs(
    llm: _HeldLLM, chat: _Chat, session_mgr: AgentSessionManager
) -> _Reports:
    # Two tabs on one agent: beta is requested before alpha books, then waits on the lock.
    agent = await session_mgr.get_or_create_agent("overlap-a", "sess_overlap")
    lock = _ContendedLock()
    agent._turn_lock = lock
    llm.hold(_ALPHA, 0)
    alpha = asyncio.create_task(chat(_ALPHA, "overlap-a"))
    await _reached(llm.entered(_ALPHA, 0))
    beta = asyncio.create_task(chat(_BETA, "overlap-a"))
    await _reached(lock.contended)
    llm.hold(_ALPHA, 0).set()
    return await alpha, await beta


async def _two_agents_book_between_each_others_steps(
    llm: _HeldLLM, chat: _Chat, session_mgr: AgentSessionManager
) -> _Reports:
    # Both turns start, alpha books its first step, beta books and ends, alpha books its last.
    for turn, index in ((_ALPHA, 0), (_ALPHA, 1), (_BETA, 0)):
        llm.hold(turn, index)
    alpha = asyncio.create_task(chat(_ALPHA, "overlap-a"))
    await _reached(llm.entered(_ALPHA, 0))
    beta = asyncio.create_task(chat(_BETA, "overlap-b"))
    await _reached(llm.entered(_BETA, 0))
    llm.hold(_ALPHA, 0).set()
    await _reached(llm.entered(_ALPHA, 1))
    llm.hold(_BETA, 0).set()
    beta_report = await beta
    llm.hold(_ALPHA, 1).set()
    return await alpha, beta_report


async def _two_agents_second_starts_after_the_first_booked(
    llm: _HeldLLM, chat: _Chat, session_mgr: AgentSessionManager
) -> _Reports:
    # Alpha books its first step; beta then runs whole; alpha books its last step.
    llm.hold(_ALPHA, 1)
    alpha = asyncio.create_task(chat(_ALPHA, "overlap-a"))
    await _reached(llm.entered(_ALPHA, 1))
    beta_report = await chat(_BETA, "overlap-b")
    llm.hold(_ALPHA, 1).set()
    return await alpha, beta_report


@pytest.mark.parametrize(
    "schedule",
    [
        pytest.param(_one_agent_second_turn_sent_while_the_first_runs, id="one-agent"),
        pytest.param(_two_agents_book_between_each_others_steps, id="two-agents-interleaved"),
        pytest.param(_two_agents_second_starts_after_the_first_booked, id="two-agents-staggered"),
    ],
)
async def test_overlapping_chat_turns_on_one_session_id_each_report_only_the_steps_they_booked(
    tmp_path: Path,
    schedule: Callable[[_HeldLLM, _Chat, AgentSessionManager], Awaitable[_Reports]],
) -> None:
    """Two turns overlapping on one session id report their own steps, and their own label.

    The figures were a slice of the session's ledger from an index taken before the agent's
    turn lock, so an overlapping turn's steps landed in both slices (#982): through this
    endpoint, one agent gave 110 + 330 = 440 and two agents on one session id gave 650 or
    550, and alpha's `provider` label turned `estimate` from beta's step. Alpha books 100
    then 10 tokens, both counted; beta books 220, estimated. Each schedule orders the
    bookings by what the turns have reached.

    Killed by: src/uclone_x/llm/budget.py :: if any(turn is mine for mine in ours):
    Becomes: if True:
    """
    llm = _HeldLLM(
        {
            _ALPHA: [
                _figures_tool_call(_figures_usage(90, 10, TokenCountSource.PROVIDER)),
                _figures_answer(_figures_usage(8, 2, TokenCountSource.PROVIDER)),
            ],
            _BETA: [_figures_answer(_figures_usage(200, 20, TokenCountSource.ESTIMATE))],
        }
    )
    session_mgr = AgentSessionManager(bus=EventBus(), llm=llm, tools=_figures_tools())
    app = create_ui_app(static_dir=tmp_path, session_manager=session_mgr)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:

        async def chat(message: str, agent_id: str) -> dict[str, Any]:
            body = {"message": message, "agent_id": agent_id, "session_id": "sess_overlap"}
            return cast(dict[str, Any], (await client.post("/api/turn", json=body)).json())

        alpha, beta = await schedule(llm, chat, session_mgr)

    booked = session_mgr.budget_tracker.get_turn_history("sess_overlap")
    assert sum(u.total_tokens for u in booked) == 330, booked  # every step ran and was booked
    figures = ("tokens_used", "token_count_source")
    assert [alpha[k] for k in figures] + [alpha["debug_info"]["prompt_tokens_used"]] == [
        110,
        "provider",
        98,
    ], alpha
    assert [beta[k] for k in figures] + [beta["debug_info"]["prompt_tokens_used"]] == [
        220,
        "estimate",
        200,
    ], beta


@pytest.mark.asyncio
async def test_ui_stream_cancelled_mid_wait_reads_its_abandoned_waiter(tmp_path: Path) -> None:
    """Cancelling the stream while it waits must not abandon `sub.get()` (#1038).

    This is the other exit from the `await asyncio.wait` that #872 fixed, and the drain
    #872 added cannot reach it. There the waiter was already `done` and the shutdown
    branch walked away without reading it. Here a client closes the stream while both
    waiters are still **pending**: `CancelledError` is thrown into the generator at the
    `await` itself, so `_cancel_and_drain(pending)` on the line below never runs, the
    `except (asyncio.CancelledError, ...)` breaks, and the `finally`'s `sub.close()` then
    puts the close sentinel on the queue -- which wakes the abandoned waiter with
    `SubscriptionClosedError` that nobody holds. asyncio reports it at finalization.

    The assertion is made through the loop's exception handler and `gc`, never by calling
    `.exception()` on the waiter, because that call would itself retrieve it and prove
    nothing.

    Killed by: src/uclone_x/ui/app.py :: _discard_waiter(get_task)
    Becomes: pass
    """
    import gc

    from fastapi.routing import APIRoute
    from starlette.requests import Request
    from starlette.responses import StreamingResponse

    from uclone_x.engine.event_bus import SubscriptionClosedError

    bus = EventBus()
    app = create_ui_app(static_dir=tmp_path, bus=bus)

    route = next(r for r in app.routes if isinstance(r, APIRoute) and r.path == "/api/stream")
    mock_request = MagicMock(spec=Request)
    mock_request.app = app
    mock_request.is_disconnected = AsyncMock(return_value=False)

    response = await route.endpoint(request=mock_request)
    assert isinstance(response, StreamingResponse)

    body = cast(AsyncIterator[Any], response.body_iterator)
    first = await anext(body)
    assert "SYSTEM_CONNECTED" in (first.decode("utf-8") if isinstance(first, bytes) else first)

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unretrieved: list[dict[str, Any]] = []

    def record_subscription_leaks(_loop: object, context: dict[str, Any]) -> None:
        """Record only a waiter's leak, not the bus's own dispatcher teardown noise."""
        if isinstance(context.get("exception"), SubscriptionClosedError):
            unretrieved.append(context)

    loop.set_exception_handler(record_subscription_leaks)
    try:
        # The stream has nothing to send, so this parks in the `await asyncio.wait`.
        async def next_chunk() -> Any:
            return await anext(body)

        pump: asyncio.Task[Any] = asyncio.create_task(next_chunk())
        for _ in range(5):
            await asyncio.sleep(0)

        pump.cancel()  # the client closes the stream
        with pytest.raises((asyncio.CancelledError, StopAsyncIteration)):
            await pump

        gc.collect()
        await asyncio.sleep(0)
        assert unretrieved == [], f"the cancelled stream left a waiter unread: {unretrieved}"
    finally:
        loop.set_exception_handler(previous_handler)


def test_an_unusable_agent_name_is_a_client_error_not_a_session_fault() -> None:
    """The agent id comes from the request, so its refusal is a 400 the caller can act on.

    `AgentHomeError` reaches this translator through the same broad handlers that wrap
    session loading, so without a branch it fell through to a 500 labelled "Session
    operation failed" -- a client-fixable request reported as a server fault, naming a
    subsystem that is not at fault (P6). The message already states the rule the name
    broke; the status has to agree with it.

    Killed by: src/uclone_x/ui/app.py :: if isinstance(exc, AgentHomeError):
    Becomes: if False:
    """
    from uclone_x.core.agent_home import AgentHomeError
    from uclone_x.ui.app import _translate_session_error

    exc = _translate_session_error(AgentHomeError("agent username 'Scout' is not usable"))

    assert exc.status_code == 400, "a name the client sent is not a server fault"
    assert "Scout" in str(exc.detail), "the caller cannot fix a name the refusal withholds"


# ======================================================================================
# vLLM in the settings surface (#1304)
# ======================================================================================

_VLLM_MODELS_PAYLOAD = {
    "object": "list",
    "data": [{"id": "qwen2.5-coder-32b-instruct", "object": "model", "owned_by": "vllm"}],
}
"""What a vLLM server answers at `/v1/models`: one entry, the model it was started with."""


def test_choosing_vllm_writes_vllms_own_variables_and_not_another_providers(
    tmp_path: Path,
) -> None:
    """Saving a vLLM configuration configures vLLM, under the names its connector reads.

    Each provider's endpoint, model and key live under their own variables, and
    `VLLMConnector` reads them there: it refuses a request that names no model unless
    `VLLM_MODEL` does. Writing the endpoint to `OPENAI_BASE_URL` instead would point
    OpenAI's connector at the operator's local server -- and send it OpenAI's key -- while
    vLLM stayed unconfigured, so the three assertions are about three separate failures
    rather than one restated.

    Killed by: src/uclone_x/ui/app.py :: env_updates["VLLM_BASE_URL"] = clean_base
    Becomes: env_updates["OPENAI_BASE_URL"] = clean_base

    Killed by: src/uclone_x/ui/app.py :: env_updates[VLLM_MODEL_ENV_VAR] = clean_model
    Becomes: env_updates["OLLAMA_MODEL"] = clean_model

    Killed by: src/uclone_x/ui/app.py :: env_updates["VLLM_API_KEY"] = clean_key
    Becomes: env_updates["OPENAI_API_KEY"] = clean_key
    """
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=False, storage_dir=tmp_path)
    session_mgr: AgentSessionManager = app.state.session_manager

    session_mgr.update_settings(
        llm_provider="vllm",
        llm_base_url="http://gpu-box.invalid:8000",
        llm_model="qwen2.5-coder-32b-instruct",
        llm_api_key="served-with-a-key",
    )

    assert os.environ["VLLM_BASE_URL"] == "http://gpu-box.invalid:8000"
    assert os.environ["VLLM_MODEL"] == "qwen2.5-coder-32b-instruct"
    assert os.environ["VLLM_API_KEY"] == "served-with-a-key"
    assert "OPENAI_BASE_URL" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ
    assert "OLLAMA_MODEL" not in os.environ
    assert isinstance(session_mgr.llm, VLLMConnector)
    assert session_mgr.llm.provider_name == "vllm"


@pytest.mark.asyncio
async def test_vllm_is_offered_as_a_provider_and_accepted_when_selected(tmp_path: Path) -> None:
    """`providers_available` names vllm, and a save naming vllm is not a 400.

    Two separately maintained lists -- one advertises providers to the panel, the other
    admits them -- and a provider in the second but not the first is one nobody can reach
    from the UI, while the reverse is a card that saves to an error. The pair is the
    contract, so both directions are asserted here.

    Killed by: src/uclone_x/ui/app.py :: available_providers = ["ollama", "vllm", "openai", "anthropic", "gemini"]
    Becomes: available_providers = ["ollama", "openai", "anthropic", "gemini"]

    Killed by: src/uclone_x/ui/app.py :: allowed = {"ollama", "vllm", "openai", "anthropic", "gemini", "google", "mock"}
    Becomes: allowed = {"ollama", "openai", "anthropic", "gemini", "google", "mock"}
    """
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=False, storage_dir=tmp_path)
    session_mgr: AgentSessionManager = app.state.session_manager

    assert "vllm" in session_mgr.get_settings()["providers_available"]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.post(
            "/api/settings",
            json={"llm_provider": "vllm", "llm_base_url": "http://gpu-box.invalid:8000"},
        )

    assert res.status_code == 200
    assert cast(dict[str, Any], res.json())["llm_provider"] == "vllm"


@pytest.mark.asyncio
async def test_the_model_list_reports_the_model_the_vllm_server_is_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dropdown is filled from `/v1/models`, which for vLLM says what *is* running.

    Ollama's `/api/tags` lists everything pulled and the operator picks one; a vLLM server
    serves the single model it was launched with, so this listing is not a catalogue but the
    answer to "what did I start?". It is also the only place the operator can read the exact
    string `--model` was given, which is what has to go in `VLLM_MODEL` for a turn to work.

    Killed by: src/uclone_x/ui/app.py :: return vllm_model_ids(resp.json())
    Becomes: return []

    Killed by: src/uclone_x/ui/app.py :: f"{vllm_url.rstrip('/')}/models", headers=vllm_request_headers()
    Becomes: f"{vllm_url.rstrip('/')}/api/tags", headers=vllm_request_headers()
    """
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box.invalid:8000")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_VLLM_MODELS_PAYLOAD)

    stub_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("uclone_x.ui.app.httpx.AsyncClient", return_value=stub_client):
        models = await fetch_available_models(provider="vllm")

    assert models == ["qwen2.5-coder-32b-instruct"]
    assert seen["url"] == "http://gpu-box.invalid:8000/v1/models"


@pytest.mark.asyncio
async def test_an_unconfigured_vllm_endpoint_is_not_probed_for_models() -> None:
    """With no endpoint configured, nothing is requested at all.

    The alternative is this module probing vLLM's documented default port to fill a
    dropdown: a request to whatever is listening on 8000 of the machine running the
    dashboard, authorised by nothing the operator said. That the refused connection would
    then be reported as "no models available" is the second defect, not the first (P6).

    Killed by: src/uclone_x/ui/app.py :: if not has_configured_vllm_endpoint(base_url):
    Becomes: if False:
    """
    with patch("uclone_x.ui.app.httpx.AsyncClient") as client_cls:
        models = await fetch_available_models(provider="vllm")

    assert models == []
    client_cls.assert_not_called()


def test_the_connectivity_test_reads_the_models_a_vllm_endpoint_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Test connection" against vLLM asks `/v1/models` and reports the model it names.

    The button exists to answer "is the thing I configured actually there?" before a turn
    depends on it. For vLLM the useful answer includes *which* model, because an endpoint
    answering with a different one than the operator typed is the failure most likely to be
    waiting -- and it is invisible in a bare ok/unreachable verdict.

    Killed by: src/uclone_x/ui/app.py :: "models": vllm_model_ids(resp.json()),
    Becomes: "models": [],
    """
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box.invalid:8000")
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=False, storage_dir=tmp_path)
    client = TestClient(app)
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=_VLLM_MODELS_PAYLOAD)

    stub_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("uclone_x.ui.app.httpx.AsyncClient", return_value=stub_client):
        res = client.post("/api/settings/test", json={"target": "llm", "llm_provider": "vllm"})

    assert res.status_code == 200
    llm_result = cast(dict[str, Any], res.json())["results"]["llm"]
    assert llm_result["status"] == "ok"
    assert llm_result["provider"] == "vllm"
    assert llm_result["models"] == ["qwen2.5-coder-32b-instruct"]
    assert paths == ["/v1/models"]


@pytest.mark.asyncio
async def test_the_connectivity_test_names_the_variable_when_no_vllm_endpoint_is_set(
    tmp_path: Path,
) -> None:
    """Unconfigured is reported as unconfigured, in the terms the panel can act on.

    Every other provider's test either reaches something or checks for a key. vLLM is the
    one that can fail before any request is made, and the operator who pressed the button
    needs the reason to be the missing endpoint rather than a socket error against a port
    this code picked for them (P6).

    The phrase asserted on is the load-bearing part, and the reason is worth stating.
    Without the check, `resolve_vllm_base_url` raises and the route's own `except` puts
    *its* message in the same field -- which also names `VLLM_BASE_URL`, so a test looking
    only for the variable cannot tell the two apart and passes either way (measured: the
    declaration below escaped until this assertion was added). What only the branch under
    test says is the other way to fix it: the field on the panel the operator is already
    looking at. A dashboard whose remediation is "set an environment variable" for a value
    it has an input box for is the defect being pinned.

    Killed by: src/uclone_x/ui/app.py :: if not has_configured_vllm_endpoint(eff_base):
    Becomes: if False:
    """
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=False, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.post(
            "/api/settings/test", json={"target": "llm", "llm_provider": "vllm"}
        )

    assert res.status_code == 200
    llm_result = cast(dict[str, Any], res.json())["results"]["llm"]
    assert llm_result["status"] == "error"
    assert "VLLM_BASE_URL" in llm_result["error"]
    assert "enter the endpoint above" in llm_result["error"]


def test_a_vllm_endpoint_is_sent_a_bearer_token_only_when_one_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`vllm serve --api-key` is optional, so the header is conditional.

    An empty `Bearer ` is not equivalent to sending no header: a server started without
    `--api-key` ignores both, but a gateway in front of one reads the empty credential and
    answers 401 about something nobody configured (#385).

    Killed by: src/uclone_x/ui/app.py :: return {"Authorization": f"Bearer {key}"} if key else {}
    Becomes: return {"Authorization": f"Bearer {key}"}
    """
    assert vllm_request_headers() == {}
    assert vllm_request_headers("explicit-key") == {"Authorization": "Bearer explicit-key"}

    monkeypatch.setenv("VLLM_API_KEY", "from-the-environment")
    assert vllm_request_headers() == {"Authorization": "Bearer from-the-environment"}


def test_a_saved_vllm_model_is_still_configured_after_a_restart(tmp_path: Path) -> None:
    """The model saved in the panel is exported again when persisted settings are reloaded.

    `VLLMConnector` reads `VLLM_MODEL` at request time and refuses when nothing names a
    model. A manager that restored the provider and the endpoint but not the model would
    come back from a restart holding a configuration the operator completed, and refuse the
    first turn for lacking the very part they filled in.

    Killed by: src/uclone_x/ui/app.py :: os.environ[VLLM_MODEL_ENV_VAR] = self._configured_model
    Becomes: os.environ[VLLM_MODEL_ENV_VAR] = ""
    """
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "llm_provider": "vllm",
                "llm_base_url": "http://gpu-box.invalid:8000/v1",
                "llm_model": "qwen2.5-coder-32b-instruct",
            }
        ),
        encoding="utf-8",
    )

    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=False, storage_dir=tmp_path)
    session_mgr: AgentSessionManager = app.state.session_manager

    assert os.environ["VLLM_MODEL"] == "qwen2.5-coder-32b-instruct"
    assert os.environ["VLLM_BASE_URL"] == "http://gpu-box.invalid:8000/v1"
    settings = session_mgr.get_settings()
    assert settings["llm_provider"] == "vllm"
    assert settings["llm_model"] == "qwen2.5-coder-32b-instruct"
    assert settings["llm_base_url"] == "http://gpu-box.invalid:8000/v1"
