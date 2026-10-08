# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
"""Unit tests for the developer UI backend, API endpoints, and server launcher."""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from typer.testing import CliRunner

from tests.support.app_clone import app_clone
from tests.support.clones import make_clones
from tests.support.vite_diagnosis import answering_as, one_line
from uclone_x import __version__
from uclone_x.agent.base import BaseAgent
from uclone_x.cli import main
from uclone_x.core.agent_home import seat_id_for
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    Provenance,
    ServiceRef,
)
from uclone_x.engine.event_bus import EventBus, EventType
from uclone_x.errors import (
    LLMProviderError,
    LLMTimeoutError,
)
from uclone_x.llm import MockLLMConnector, create_llm_connector
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.models import (
    LLMRequest,
    MessageRole,
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
from uclone_x.tools.registry import ToolRegistry
from uclone_x.ui.app import (
    AgentSessionManager,
    create_ui_app,
    fetch_available_models,
    get_ui_session_manager,
    vllm_request_headers,
)
from uclone_x.ui.server import VITE_IDENTITY_MARKER, start_ui_server

runner = CliRunner()

# The agent ids these tests chat as. Each is a clone now, since a name no clone carries is
# refused rather than given a home (clone-data-scopes §3.4); persona-less, so each still
# speaks as the fallback prompt the test was written against.
_CHAT_CLONES = (
    "agent-clear-test",
    "agent-tool-err",
    "agent-tool-user",
    "champion",
    "custom-budget-agent",
    "default-agent",
    "failover-agent",
    "follower",
    "generic_custom_agent",
    "mock-agent",
    "overlap-a",
    "overlap-b",
    "prov-agent",
    "stream-agent",
    "test-agent",
    "test-agent-1",
    "test-agent-2",
    "test-agent-3",
    "test_agent",
    "unconfigured-agent",
)


@pytest.fixture(autouse=True)
def _chat_clones() -> None:  # pyright: ignore[reportUnusedFunction]
    make_clones(*_CHAT_CLONES)


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
        model="qwen3:8b",
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


@pytest.mark.asyncio
async def test_ui_chat_custom_agent_max_steps_budget() -> None:
    """A custom `AgentConfig.max_steps` bounds a run of steps, not a conversation.

    Driven through `/api/turn` until that route was retired (every conversation is a
    room); the budget is the agent's own, so it is read off the agent directly. Each
    message starts the run over: one answered message costs one step of it.
    """
    from uclone_x.agent.models import AgentConfig, AgentLLMConfig

    custom_cfg = AgentConfig(
        agent_id="custom-budget-agent",
        name="Custom Budget Agent",
        max_steps=3,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    agent = BaseAgent(config=custom_cfg, llm=MockLLMConnector())

    # Turn 4 is past the ceiling in message count, and NOT refused. `max_steps` bounds a
    # self-driven run, not a conversation -- the defect this replaces rejected a person's
    # fourth message with "Turn budget exceeded".
    for n in range(1, 5):
        result = await agent.execute_turn(f"Turn {n}")
        assert "Turn budget exceeded" not in result.content, result.content
        assert result.error is None, result.error
        assert result.turn_index == n
        assert agent.config.max_steps == 3
        assert agent.run_steps == 1
        assert agent.steps_remaining == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("builtin_personas_absent")
async def test_ui_chat_with_tools_execution(tmp_path: Path) -> None:
    """A tool-using turn runs its tool and answers after seeing the result (#171, P4).

    Driven through `/api/turn` until that route was retired; the record is the turn's
    own, so it is read off `execute_turn` for an agent the session manager built.
    """
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
    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=mock_llm, tools=tools)
    agent = app_clone(session_mgr, "agent-tool-user")

    result = await agent.execute_turn("Analyze test.py")

    assert result.error is None, result.error
    assert result.is_completed
    assert [(tc.name, tc.id) for tc in result.tool_calls] == [("ast_code_analyzer", "tc_123")]
    # The reply is the model's answer after seeing its tool results, not the text it
    # emitted alongside the tool call (P4, amended 2026-09-05).
    assert result.content, "a tool-using turn must still produce an answer"
    assert "Analysis complete" not in result.content
    assert mock_tool.execute.called

    assert len(result.tool_executions) == 1
    te = result.tool_executions[0]
    assert te.tool_name == "ast_code_analyzer"
    assert te.tool_call_id == "tc_123"
    assert unwrap_immutable(te.arguments) == {"target": "test.py"}
    assert te.status == "success"
    assert unwrap_immutable(te.output) == {"status": "clean", "symbols": 42}
    assert te.error is None
    assert te.duration_ms == 8.0


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

    # Its running-agent cache went with `get_or_create_agent` (#1893); what a clone is
    # built from is pinned where one is built (`test_one_clone_builder.py`).
    assert manager.bus is bus
    assert manager.tools is tools
    # A connector handed in answers every clone that follows the system default.
    binding = manager.gateway.default_binding
    assert binding is not None and binding.connector is mock_llm
    # Test global singleton accessor
    global_mgr = get_ui_session_manager()
    assert global_mgr is not None


def test_ui_chat_history_clear_endpoint(tmp_path: Path) -> None:
    """Clearing a session removes its transcript, and the reader then finds nothing.

    Driven through `DELETE /api/session/history` until that route was retired;
    `clear_session_history` is what rooms still call, so it is called directly here.
    """
    manager = AgentSessionManager(storage_dir=tmp_path / "sessions", fallback_to_mock=True)
    session_file = _write_transcript(
        manager,
        session_id="sess_clear",
        agent_id="agent-clear-test",
        messages=[{"sender": "user", "content": "Forget me"}],
    )
    assert manager.load_session_record("sess_clear") is not None

    manager.clear_session_history(agent_id="agent-clear-test", session_id="sess_clear")

    assert not session_file.exists()
    assert manager.load_session_record("sess_clear") is None


def _write_transcript(
    manager: AgentSessionManager,
    *,
    session_id: str,
    agent_id: str,
    messages: list[dict[str, Any]],
    turns: int = 1,
) -> Path:
    """Put a UI transcript on disk in the shape the retired writer left, and return its path.

    Nothing in the product writes these any more (every conversation is a room), but
    installs carry them and `load_session_record` / `clear_session_history` still read and
    remove them, so the readers are exercised against a file written here.
    """
    path = manager.get_session_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "agent_id": agent_id,
                "created_at": "2026-09-27T00:00:00+00:00",
                "updated_at": "2026-09-27T00:00:00+00:00",
                "turns": turns,
                "messages": messages,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_ui_chat_history_path_traversal_prevention(tmp_path: Path) -> None:
    """A traversal id is refused by the transcript reader and by the clear, before either
    touches the disk. Driven through the retired `/api/session/history` routes until now."""
    from uclone_x.errors import PathTraversalError

    manager = AgentSessionManager(storage_dir=tmp_path / "sessions", fallback_to_mock=True)

    with pytest.raises(PathTraversalError):
        manager.load_session_record("../../etc/passwd")
    with pytest.raises(PathTraversalError):
        manager.clear_session_history(agent_id="champion", session_id="../../etc/passwd")


@pytest.mark.asyncio
async def test_agent_session_manager_persistence_and_corrupt_files(tmp_path: Path) -> None:
    storage_dir = tmp_path / "sessions"
    manager = AgentSessionManager(storage_dir=storage_dir, fallback_to_mock=True)

    # A transcript left on disk (nothing writes these any more) is found under `ui/`.
    _write_transcript(
        manager,
        session_id="sess_manual",
        agent_id="test_agent",
        messages=[{"sender": "user", "content": "Manual message"}],
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

    # Path traversal in session manager raises PathTraversalError
    from uclone_x.errors import PathTraversalError

    with pytest.raises(PathTraversalError):
        manager.get_session_path("../outside")


@pytest.mark.asyncio
async def test_mock_llm_connector_and_factory() -> None:
    from uclone_x.llm.models import ChatMessage

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


def _install_clone(*names: str) -> None:
    """Give each named clone a home, so the listing a developer-graph read checks has it."""
    make_clones(*names)


def test_ui_ontology_endpoint_empty(test_client: TestClient) -> None:
    """Assert /api/ontology returns truthful empty structure when unpopulated (P6, P8)."""
    _install_clone("scout")
    response = test_client.get("/api/ontology", params={"agent_id": "scout"})
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
    """/api/ontology reads the named clone's own rules engine, and no other clone's (#1869).

    Killed by: src/uclone_x/ui/app.py :: return session_mgr.ontology_for(_developer_graph_clone(agent_id)).export_graph()
    Becomes: return session_mgr.ontology_for("default").export_graph()
    """
    _install_clone("test-agent", "scout")
    app = create_ui_app(static_dir=tmp_path)
    manager = cast(AgentSessionManager, app.state.session_manager)
    ontology_engine = manager.ontology_for("test-agent")
    assert isinstance(ontology_engine, OntologyEngine)
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

    client = TestClient(app)

    other = client.get("/api/ontology", params={"agent_id": "scout"}).json()
    assert other["total_concepts"] == 0 and other["total_relations"] == 0

    response = client.get("/api/ontology", params={"agent_id": "test-agent"})
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


def test_ui_skills_endpoint_empty(tmp_path: Path) -> None:
    """Assert /api/skills returns truthful empty list when unpopulated (P6, P8).

    The registry is given, and the client entered so startup runs (#1721). The shared
    `test_client` fixture never enters its client, so its lifespan never loads the
    default store and an empty list there says nothing about an empty registry.
    """
    static_dir = tmp_path / "ui_static"
    static_dir.mkdir()
    app = create_ui_app(
        static_dir=static_dir,
        storage_dir=tmp_path / "sessions",
        llm=MockLLMConnector(),
        skill_registry=SkillRegistry(),
    )
    with TestClient(app) as client:
        response = client.get("/api/skills")
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
    """SSE subscribers observe an agent's landed reply and its provenance (#220, P6).

    Driven through `/api/turn` until that route was retired; a reply now lands through a
    room, so a room is opened and spoken into, and the stream is read for the room
    turn's `final` AGENT_REPLY frame rather than any earlier event of the turn.
    """
    bus = EventBus()
    app = create_ui_app(static_dir=tmp_path, bus=bus, llm=stubbed_ollama_connector())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        created = await client.post(
            "/api/rooms", json={"title": "Stream", "agent_ids": ["stream-agent"]}
        )
        assert created.status_code == 201, created.text
        room_id = str(created.json()["room_id"])

        async def _post_message() -> None:
            await asyncio.sleep(0.05)
            sent = await client.post(
                f"/api/rooms/{room_id}/messages", json={"content": "Streaming test prompt"}
            )
            assert sent.status_code == 202, sent.text

        post_task = asyncio.create_task(_post_message())
        chunks: list[str] = []
        # The turn publishes three frames -- generating, streaming, final -- and the
        # stream is buffered to its end by the ASGI transport, so the bound leaves room
        # for one heartbeat ahead of them and one after, not a wait on the clock.
        async with client.stream("GET", "/api/stream?max_events=5") as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                chunks.append(line)
        await post_task

    parsed_events: list[dict[str, Any]] = []
    for line in chunks:
        if line.startswith("data: "):
            parsed_events.append(cast(dict[str, Any], json.loads(line[6:])))

    finals = [
        e
        for e in parsed_events
        if e.get("event_type") in ("AGENT_REPLY", "agent_reply")
        and cast(dict[str, Any], e.get("payload") or {}).get("status") == "final"
    ]
    assert len(finals) == 1, f"Expected the turn's final AGENT_REPLY, got {parsed_events}"
    reply = finals[0]
    assert reply["topic"] == f"room.{room_id}"
    assert reply["provenance"]["component"] == "uclone_x.engine.event_bus"
    # The seat is keyed by the clone's id, which the room resolved the handle to.
    assert seat_id_for("stream-agent").startswith("agt_")
    assert reply["provenance"]["producer"] == seat_id_for("stream-agent")
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

    # Every port reads as free: the subject is that the flags are forwarded, and a real probe
    # turned this red whenever the developer's own Vite already held 5174.
    def every_port_is_free(port: int) -> int:
        return port

    monkeypatch.setattr(main, "find_available_port", every_port_is_free)

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
    # A real request's headers: none, as from the CLI. `/api/stream` reads `Origin` (#2146).
    mock_request.headers = Headers()
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
    """A clone's turn carries its persona in the result and the provenance (FR-13.4, P6).

    Driven through `/api/turn` until that route was retired; the attribution is the
    turn's own, so it is read off `execute_turn` for agents the manager built. The route
    also copied the persona into the provenance block it returned and wrote it to the
    transcript; both copies went with it, and `Provenance` itself has no persona field.
    """
    mock_llm = MockLLMConnector(
        default_model="mock-gpt-4o",
        default_response="Champion response.",
    )
    session_mgr = AgentSessionManager(storage_dir=tmp_path, llm=mock_llm)

    # 1. The clone persona is populated on the result and in its provenance.
    clone = app_clone(session_mgr, "clone", "sess_clone_test")
    result = await clone.execute_turn("Hello Clone")
    assert result.error is None, result.error
    assert result.persona == "clone"
    assert result.provenance is not None
    assert result.provenance.served_by.provider == "mock"
    assert result.provenance.served_by.model == "mock-gpt-4o"

    # 2. A generic unconfigured agent does NOT fabricate a persona (P6).
    generic = app_clone(session_mgr, "generic_custom_agent", "sess_gen_test")
    gen_result = await generic.execute_turn("Hello Generic")
    assert gen_result.persona is None


@pytest.mark.asyncio
async def test_chat_diagnostic_path_publishes_a_degraded_provenance() -> None:
    """When a turn substitutes a diagnostic, it says so in band (#117, P6).

    The user is not looking at the agent's answer, so `requested` (the agent) differs
    from `served_by` (the Core) and `degraded` computes True rather than being asserted
    by hand. Driven through `/api/turn` until that route was retired; the provenance is
    the turn's own, so it is read off `execute_turn`.
    """
    from uclone_x.telemetry import TelemetryTracer

    tracer = TelemetryTracer()
    failing_llm = MagicMock(spec=LLMProviderProtocol)
    failing_llm.generate = AsyncMock(
        side_effect=LLMProviderError("Connection refused to Ollama at localhost:11434")
    )
    failing_mgr = AgentSessionManager(bus=EventBus(), tracer=tracer, llm=failing_llm)
    agent = app_clone(failing_mgr, "prov-agent")

    result = await agent.execute_turn("hi")

    provenance = result.provenance
    assert provenance is not None
    assert provenance.path is ExecutionPath.FAILOVER
    assert provenance.degraded is True
    assert provenance.requested.provider == "prov-agent"
    assert provenance.served_by.provider == "agent.core"
    assert provenance.attempts[0].error_class == "LLMProviderError"
    assert provenance.attempts[0].span_id is not None
    assert provenance.attempts[0].span_id.startswith("spn_")


@pytest.mark.asyncio
async def test_chat_diagnostic_path_publishes_failover_notice_ordered_before_reply() -> None:
    """P6 Check 4/5: a failed turn publishes PROVIDER_FAILOVER, correlated to its span.

    Driven through `/api/turn` until that route was retired, which also published the
    reply on `agent.chat.reply` and pinned the notice's order against it. That reply was
    the removed endpoint's; a room reply is published by the orchestrator. What remains
    is the Core's: the notice names both sides and shares a span with the turn's
    provenance.
    """
    from uclone_x.telemetry import TelemetryTracer

    bus = EventBus()
    tracer = TelemetryTracer()
    failing_llm = MagicMock(spec=LLMProviderProtocol)
    failing_llm.generate = AsyncMock(
        side_effect=LLMProviderError("Connection refused to Ollama at localhost:11434")
    )
    failing_mgr = AgentSessionManager(bus=bus, tracer=tracer, llm=failing_llm)
    notices = bus.subscribe("agent.chat.failover")
    agent = app_clone(failing_mgr, "failover-agent")

    result = await agent.execute_turn("failover order test")

    notice = await asyncio.wait_for(notices.get(), timeout=2.0)
    assert notice.type is EventType.PROVIDER_FAILOVER
    assert notice.payload["requested_provider"] == "failover-agent"
    assert notice.payload["served_provider"] == "agent.core"

    # P6 Check 5: telemetry span correlation
    spans = tracer.get_completed_spans()
    failover_spans = [s for s in spans if s.name == "failover.event"]
    assert len(failover_spans) >= 1
    failover_span = failover_spans[0]
    assert notice.provenance is not None
    assert result.provenance is not None
    assert notice.provenance.attempts[0].span_id == failover_span.span_id
    assert result.provenance.attempts[0].span_id == failover_span.span_id


@pytest.mark.asyncio
@pytest.mark.usefixtures("builtin_personas_absent")
async def test_ui_chat_tool_execution_error_records(tmp_path: Path) -> None:
    """Tool execution failures and exceptions are accurately recorded in tool_executions (Issue #171).

    Driven through `/api/turn` until that route was retired; the records are the turn's
    own, so they are read off `execute_turn` for an agent the session manager built.
    """
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
    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=mock_llm, tools=tools)
    agent = app_clone(session_mgr, "agent-tool-err")

    result = await agent.execute_turn("Run failing tools")

    assert len(result.tool_executions) == 2

    # First tool: executed with error result
    te1 = result.tool_executions[0]
    assert te1.tool_name == "failing_tool"
    assert te1.tool_call_id == "tc_fail"
    assert te1.status == "error"
    assert te1.error == "Execution timeout in sandbox"
    assert te1.output is None
    assert te1.duration_ms == 12.5

    # Second tool: no tool has that name, answered in plain words (#2190)
    te2 = result.tool_executions[1]
    assert te2.tool_name == "nonexistent_tool"
    assert te2.tool_call_id == "tc_not_found"
    assert te2.status == "error"
    assert "There is no tool named 'nonexistent_tool'" in str(te2.error)


@pytest.mark.asyncio
async def test_get_settings_endpoint(tmp_path: Path) -> None:
    """GET /api/settings carries the defaults and the other settings, and no `llm_*` field.

    The connections and models have their own routes (model-gateway §3.7.1); the
    pre-gateway fields are deleted, not left empty (step 3).
    """
    mock_llm = MockLLMConnector(api_key="sk-abcdef123456", base_url="http://mock-llm.invalid:8000")
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.get("/api/settings")

    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["default_models"] == {"deep": None, "fast": None, "image": "auto"}
    # The picture settings went with model-gateway step 5: a ComfyUI is a connection, and
    # the picture model is `default_models.image`.
    assert not {"comfyui_base_url", "image_engine", "image_model"} & set(data)
    assert not [key for key in data if key.startswith("llm_")]
    assert "providers" not in data and "available_models" not in data
    assert "sk-abcdef123456" not in json.dumps(data)


@pytest.mark.asyncio
async def test_get_models_endpoint(tmp_path: Path) -> None:
    """GET /api/models answers the model set grouped by connection (model-gateway §3.7.1)."""
    (tmp_path / "settings.json").write_text(
        json.dumps({"connections": [{"id": "mock", "kind": "mock"}]}), encoding="utf-8"
    )
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.get("/api/models")

    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert set(data) == {"groups", "defaults", "recommended"}
    (group,) = data["groups"]
    assert group["connection_id"] == "mock"
    assert "mock/mock-llm" in [m["ref"] for m in group["models"]]


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
    """POST /api/models/pull surfaces an LLMProviderError as a 502 with plain words (#1460).

    Killed by: src/uclone_x/ui/app.py :: status_code=502, detail=f"Could not pull {model} from Ollama."
    Becomes: status_code=502, detail=str(exc)
    """
    mock_llm = MockLLMConnector()
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm, storage_dir=tmp_path)

    failing_pull = AsyncMock(side_effect=LLMProviderError("Failed to connect to Ollama: refused"))
    with patch("uclone_x.ui.app.pull_model", new=failing_pull):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            res = await client.post("/api/models/pull", json={"model": "llama3.2:1b"})

    assert res.status_code == 502
    assert res.json()["detail"] == "Could not pull llama3.2:1b from Ollama."
    assert "Failed to connect to Ollama" not in res.json()["detail"]
    assert "LLMProviderError" not in res.json()["detail"]


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
    """POST /api/models/delete surfaces an LLMProviderError as a 502 with plain words (#1460).

    Killed by: src/uclone_x/ui/app.py :: status_code=502, detail=f"Could not delete {model} from Ollama."
    Becomes: status_code=502, detail=str(exc)
    """
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
    assert res.json()["detail"] == "Could not delete llama3.2:1b from Ollama."
    assert "status 404" not in res.json()["detail"]
    assert "LLMProviderError" not in res.json()["detail"]


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
    """POST /api/settings saves the default model and broadcasts an event (#350).

    A clone built after the save answers on the saved default's connection and model. A
    room seat built before it is re-bound by its resolver (`test_room_llm_replaced.py`).
    """
    from uclone_x.llm.connections import ModelRef

    event_bus = EventBus()
    sub = event_bus.subscribe("settings")

    registry = ToolRegistry(tools=[])
    (tmp_path / "settings.json").write_text(
        json.dumps({"connections": [{"id": "mock", "kind": "mock"}]}), encoding="utf-8"
    )
    app = create_ui_app(static_dir=tmp_path, bus=event_bus, tools=registry, storage_dir=tmp_path)

    session_mgr: AgentSessionManager = app.state.session_manager

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.post(
            "/api/settings",
            json={"default_models": {"deep": "mock/mock-llm"}},
        )

    assert res.status_code == 200
    updated = cast(dict[str, Any], res.json())
    assert updated["default_models"]["deep"] == "mock/mock-llm"

    # 1. The default reaches a clone built now, on its connection's connector.
    agent = app_clone(session_mgr, "test-agent", "sess_test")
    assert agent.llm is session_mgr.gateway.connector_for(ModelRef.parse("mock/mock-llm"))
    assert agent.config.llm_config.model_name == "mock-llm"

    # 3. Event broadcast verified: settings.updated received on event bus
    event = await asyncio.wait_for(sub.get(), timeout=2.0)
    assert event.type is EventType.SETTINGS_UPDATED
    assert event.topic == "settings"
    defaults = cast(dict[str, Any], event.payload["default_models"])
    assert defaults["deep"] == "mock/mock-llm"


@pytest.mark.asyncio
async def test_update_settings_invalid_provider_returns_400(tmp_path: Path) -> None:
    """A default model that is not a ref is refused in plain words, and nothing is saved.

    Killed by: src/uclone_x/ui/app.py :: refusal = await _refuse_unlisted_defaults(defaults)
    Becomes: refusal = None
    """
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=False, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.post("/api/settings", json={"default_models": {"deep": "gpt-5"}})
    assert res.status_code == 400
    detail = cast(dict[str, Any], res.json())["detail"]
    assert "which connection" in detail
    assert "Error" not in detail
    assert not (tmp_path / "settings.json").exists()


def test_llm_provider_alone_gives_conversations_a_model_with_no_settings_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment overrides the file and does not need one (S4, #1899).

    `LLM_PROVIDER` makes an ephemeral connection and the provider's model variable names the
    default deep model as `<kind>/<id>` (model-gateway §3.2), so conversations have a model
    with no settings file at all.

    Killed by: src/uclone_x/llm/gateway.py :: return replace(saved, deep=f"{kind}/{model}", env_vars={"deep": variable})
    Becomes: return saved
    """
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=False)

    assert not (tmp_path / "settings.json").exists(), "the case is the one with no file"
    (conn,) = mgr.gateway.connections()
    assert (conn.id, conn.source, conn.env_var) == ("ollama", "env", "LLM_PROVIDER")
    assert mgr.get_settings()["default_models"]["deep"] == "ollama/qwen3:8b"
    agent = app_clone(mgr, "scout", "sess_env")
    assert getattr(agent.llm, "provider_name", None) == "ollama"
    assert agent.config.llm_config.model_name == "qwen3:8b"


@pytest.mark.asyncio
async def test_settings_persistence_across_manager_instances(tmp_path: Path) -> None:
    """Settings saved in one session manager instance persist to disk and rehydrate in another (#350)."""
    (tmp_path / "settings.json").write_text(
        json.dumps({"connections": [{"id": "mock", "kind": "mock"}]}), encoding="utf-8"
    )
    mgr1 = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    mgr1.update_settings(default_models={"deep": "mock/persisted-model-v1"}, ui_language="ko")

    # Instantiate fresh AgentSessionManager with the same storage directory
    mgr2 = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    settings2 = mgr2.get_settings()
    assert settings2["default_models"]["deep"] == "mock/persisted-model-v1"
    assert settings2["ui_language"] == "ko"
    assert [c.id for c in mgr2.gateway.connections()] == ["mock"]


@pytest.mark.asyncio
async def test_persisted_settings_initializes_active_llm_connector_on_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clone answers on the saved default's connection from startup (#891, model-gateway §3.3).

    From explicit arguments: the file's values are not copied into the environment.
    """
    for k in ("LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "connections": [
                    {"id": "ollama", "kind": "ollama", "base_url": "http://127.0.0.1:11434"}
                ],
                "default_models": {"deep": "ollama/hermes3:8b"},
            }
        ),
        encoding="utf-8",
    )

    mgr = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=False)

    llm, model = mgr.gateway.default_deep()
    assert getattr(llm, "provider_name", "") == "ollama"
    assert getattr(llm, "base_url", "") == "http://127.0.0.1:11434"
    assert model == "hermes3:8b"
    assert os.getenv("LLM_PROVIDER") is None
    assert os.getenv("OLLAMA_BASE_URL") is None
    assert os.getenv("OLLAMA_MODEL") is None

    agent = app_clone(mgr, "scout", "test-session-891")
    assert agent.llm is llm
    # The saved model reaches the agent explicitly, not through `OLLAMA_MODEL`.
    assert agent.config.llm_config.model_name == "hermes3:8b"


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
    session_dir = tmp_path / "artifacts" / "sess_test_1"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "output.md").write_text(
        "# Tool Output Analysis\n\nTool run results.\n", encoding="utf-8"
    )
    # A file where tool results were kept before #1848 is not a clone's document.
    old_store = tmp_path / ".sandbox" / "tool_artifacts" / "sess_test_1"
    old_store.mkdir(parents=True, exist_ok=True)
    (old_store / "legacy.md").write_text("# Legacy\n\nOld result.\n", encoding="utf-8")

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
    assert "artifacts/sess_test_1/output.md" in sess_paths
    assert ".sandbox/tool_artifacts/sess_test_1/legacy.md" not in sess_paths
    assert "docs/design/rfc.md" not in sess_paths
    tool_item = next(
        a for a in data_sess["artifacts"] if a["path"] == "artifacts/sess_test_1/output.md"
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

    Killed by: src/uclone_x/ui/app.py :: resolved = validator.resolve_safe_path(Path(clean_path), root or self._workspace_dir)
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

    Killed by: src/uclone_x/ui/app.py :: return knowledge_graph(self.ontology_for(agent_id), session_id=session_id)
    Becomes: return knowledge_graph(self.ontology_for("default"), session_id=session_id)
    """
    _install_clone("test-agent", "scout")
    mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", workspace_dir=tmp_path)
    engine = mgr.ontology_for("test-agent")
    assert isinstance(engine, OntologyEngine)
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

    app = create_ui_app(session_manager=mgr)
    client = TestClient(app)

    # 1. Unfiltered query of the clone's engine
    res = client.get("/api/knowledge-graph?agent_id=test-agent")
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
    res_sess = client.get("/api/knowledge-graph?agent_id=test-agent&session_id=sess_kg_1")
    assert res_sess.status_code == 200
    data_sess = cast(dict[str, Any], res_sess.json())
    sess_triples = data_sess["triples"]
    assert any(t["predicate"] == "renders" for t in sess_triples)

    # Session filter: other session
    res_other = client.get("/api/knowledge-graph?agent_id=test-agent&session_id=sess_different")
    assert res_other.status_code == 200
    data_other = cast(dict[str, Any], res_other.json())
    assert not any(t["predicate"] == "renders" for t in data_other["triples"])

    # 3. Another clone's engine holds none of it: there is no shared engine (#1869)
    res_agent = client.get("/api/knowledge-graph?agent_id=scout")
    assert res_agent.status_code == 200
    data_agent = cast(dict[str, Any], res_agent.json())
    assert data_agent["summary"]["total_triples"] == 0


@pytest.mark.parametrize("route", ["/api/ontology", "/api/knowledge-graph"])
def test_a_developer_graph_route_names_its_clone(tmp_path: Path, route: str) -> None:
    """With no shared engine, a developer-graph read names a clone, in plain words (#1869).

    No clone named: 400 saying what to add. A name no clone can have: 404.

    Killed by: src/uclone_x/ui/app.py :: raise HTTPException(status_code=400, detail="Name the clone to read with agent_id.")
    Becomes: agent_id = "default"
    """
    client = TestClient(create_ui_app(static_dir=tmp_path))

    unnamed = client.get(route)
    assert unnamed.status_code == 400
    assert unnamed.json()["detail"] == "Name the clone to read with agent_id."

    unusable = client.get(route, params={"agent_id": "../Scout"})
    assert unusable.status_code == 404
    assert unusable.json()["detail"] == "There is no clone with that name here."


@pytest.mark.parametrize("route", ["/api/ontology", "/api/knowledge-graph"])
def test_a_developer_graph_read_of_an_unknown_clone_is_refused_and_makes_no_engine(
    tmp_path: Path, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A well-formed name that is no clone here is a 404, and no engine is kept for it.

    Before, the name check alone let `nosuchclone0` through: 200 with an empty graph, and
    an empty engine left in the manager's map for the app's lifetime.

    Killed by: src/uclone_x/ui/app.py :: if agent_id in (clone.id, clone.name):
    Becomes: if True:
    """
    _install_clone("scout")
    app = create_ui_app(static_dir=tmp_path)
    manager = cast(AgentSessionManager, app.state.session_manager)
    engines_made_for: list[str] = []
    real_ontology_for = manager.ontology_for

    def recording_ontology_for(agent_id: str) -> Any:
        engines_made_for.append(agent_id)
        return real_ontology_for(agent_id)

    monkeypatch.setattr(manager, "ontology_for", recording_ontology_for)
    client = TestClient(app)

    unknown = client.get(route, params={"agent_id": "nosuchclone0"})
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == "There is no clone with that name here."
    assert "nosuchclone0" not in engines_made_for

    # The recorder sees the route's reads: an installed clone's read goes through it.
    # Engines are keyed by clone id; the handle is resolved to it first.
    assert client.get(route, params={"agent_id": "scout"}).status_code == 200
    assert seat_id_for("scout").startswith("agt_")
    assert engines_made_for == [seat_id_for("scout")]


@pytest.mark.parametrize("route", ["/api/ontology", "/api/knowledge-graph"])
def test_a_developer_graph_read_of_an_unreadable_clone_folder_is_refused(
    tmp_path: Path, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder the clone listing marks unreadable is not a clone a graph can be read for.

    The listing shows a folder named `Bad Name`, since a home is reported rather than
    hidden, but no clone can have that name. Before #1879 the route found the name in the
    listing, answered 200 and kept an engine for it.

    Killed by: src/uclone_x/ui/app.py :: if clone.status is CloneStatus.UNREADABLE:
    Becomes: if clone.status is None:
    """
    from uclone_x.core.agent_home import list_agent_homes

    _install_clone("scout")
    (list_agent_homes().root / "Bad Name").mkdir()
    app = create_ui_app(static_dir=tmp_path)
    manager = cast(AgentSessionManager, app.state.session_manager)
    engines_made_for: list[str] = []
    real_ontology_for = manager.ontology_for

    def recording_ontology_for(agent_id: str) -> Any:
        engines_made_for.append(agent_id)
        return real_ontology_for(agent_id)

    monkeypatch.setattr(manager, "ontology_for", recording_ontology_for)
    client = TestClient(app)
    listed = client.get("/api/clones").json()["clones"]
    assert {"name": "Bad Name", "status": "unreadable"}.items() <= next(
        c for c in listed if c["name"] == "Bad Name"
    ).items()

    refused = client.get(route, params={"agent_id": "Bad Name"})

    assert refused.status_code == 404
    assert refused.json()["detail"] == "There is no clone with that name here."
    assert engines_made_for == []
    assert client.get(route, params={"agent_id": "scout"}).status_code == 200


@pytest.mark.asyncio
async def test_chat_turn_respects_and_logs_requested_model(tmp_path: Path) -> None:
    """A requested model override reaches the agent the manager builds.

    Driven through `/api/turn` (which echoed it as `model`) until that route was
    retired; the override is `build_clone`'s, so it is read off the agent.
    """
    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=MockLLMConnector())
    agent = app_clone(session_mgr, "champion", model_name="hermes3:8b")

    assert agent.config.llm_config.model_name == "hermes3:8b"
    result = await agent.execute_turn("Hello from hermes test")
    assert result.error is None, result.error


@pytest.mark.asyncio
async def test_chat_turn_propagates_model_not_found_error_without_silent_replacement(
    tmp_path: Path,
) -> None:
    """When a requested model is not found, the error is delivered directly without a
    silent fallback to another model.

    Driven through `/api/turn` until that route was retired; the error is the turn's own,
    so it is read off `execute_turn` for an agent built with the missing model.
    """

    def not_found_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": "model 'nonexistent-model' not found, try pulling it first"}
        )

    ollama_connector = OllamaConnector(
        base_url="http://stub-ollama.invalid:11434",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(not_found_handler)),
        model="qwen3:8b",
    )
    session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=ollama_connector)
    agent = app_clone(session_mgr, "champion", model_name="nonexistent-model")

    result = await agent.execute_turn("Hello to missing model")

    assert result.error is not None
    assert "model 'nonexistent-model' not found" in result.error


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
    # A real request's headers: none, as from the CLI. `/api/stream` reads `Origin` (#2146).
    mock_request.headers = Headers()
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
    # A real request's headers: none, as from the CLI. `/api/stream` reads `Origin` (#2146).
    mock_request.headers = Headers()
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
    # A real request's headers: none, as from the CLI. `/api/stream` reads `Origin` (#2146).
    mock_request.headers = Headers()
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


# ======================================================================================
# vLLM in the settings surface (#1304)
# ======================================================================================

_VLLM_MODELS_PAYLOAD = {
    "object": "list",
    "data": [{"id": "qwen2.5-coder-32b-instruct", "object": "model", "owned_by": "vllm"}],
}
"""What a vLLM server answers at `/v1/models`: one entry, the model it was started with."""


@pytest.mark.asyncio
async def test_the_model_list_reports_the_model_the_vllm_server_is_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dropdown is filled from `/v1/models`, which for vLLM says what *is* running.

    Ollama's `/api/tags` lists everything pulled and the operator picks one; a vLLM server
    serves the single model it was launched with, so this listing is not a catalogue but the
    answer to "what did I start?". It is also the only place the operator can read the exact
    string `--model` was given, which is what has to go in `VLLM_MODEL` for a turn to work.

    Killed by: src/uclone_x/llm/model_listing.py :: return vllm_model_ids(resp.json())
    Becomes: return []

    Killed by: src/uclone_x/llm/model_listing.py :: listing_url = f"{vllm_url.rstrip('/')}/models"
    Becomes: listing_url = f"{vllm_url.rstrip('/')}/api/tags"
    """
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box.invalid:8000")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_VLLM_MODELS_PAYLOAD)

    stub_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch("uclone_x.llm.model_listing.httpx.AsyncClient", return_value=stub_client):
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

    Killed by: src/uclone_x/llm/model_listing.py :: if not has_configured_vllm_endpoint(base_url):
    Becomes: if False:
    """
    with patch("uclone_x.llm.model_listing.httpx.AsyncClient") as client_cls:
        models = await fetch_available_models(provider="vllm")

    assert models == []
    client_cls.assert_not_called()


def test_a_vllm_endpoint_is_sent_a_bearer_token_only_when_one_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`vllm serve --api-key` is optional, so the header is conditional.

    An empty `Bearer ` is not equivalent to sending no header: a server started without
    `--api-key` ignores both, but a gateway in front of one reads the empty credential and
    answers 401 about something nobody configured (#385).

    Killed by: src/uclone_x/llm/model_listing.py :: return {"Authorization": f"Bearer {key}"} if key else {}
    Becomes: return {"Authorization": f"Bearer {key}"}
    """
    assert vllm_request_headers() == {}
    assert vllm_request_headers("explicit-key") == {"Authorization": "Bearer explicit-key"}

    monkeypatch.setenv("VLLM_API_KEY", "from-the-environment")
    assert vllm_request_headers() == {"Authorization": "Bearer from-the-environment"}


async def test_ui_language_defaults_to_system_and_round_trips(tmp_path: Path) -> None:
    """The language choice is the Core's, saved beside the model choice."""
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=True, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        before = cast(dict[str, Any], (await client.get("/api/settings")).json())
        saved = await client.post("/api/settings", json={"ui_language": "ko"})

    assert before["ui_language"] == "system"
    assert saved.status_code == 200
    assert cast(dict[str, Any], saved.json())["ui_language"] == "ko"
    # Another reader of the same storage -- the CLI, a second head -- sees the choice.
    reloaded = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    assert reloaded.get_settings()["ui_language"] == "ko"


async def test_a_language_save_leaves_the_model_connector_alone(tmp_path: Path) -> None:
    """Switching the language changes no model: no open conversation is re-bound.

    Killed by: src/uclone_x/ui/app.py :: if default_models is not None:
    Becomes: if True:
    """
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=True, storage_dir=tmp_path)
    session_mgr: AgentSessionManager = app.state.session_manager
    rebound: list[None] = []
    session_mgr.on_models_changed(lambda: rebound.append(None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        saved = await client.post("/api/settings", json={"ui_language": "ko"})

    assert saved.status_code == 200
    assert cast(dict[str, Any], saved.json())["ui_language"] == "ko"
    assert rebound == []


@pytest.mark.parametrize("field", ["read_roots"])
async def test_a_save_of_one_other_field_leaves_the_model_connector_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Settings saves each field on its own; a folder or picture-service save moves no model."""
    monkeypatch.delenv("UCLONE_READ_ROOTS", raising=False)
    papers = tmp_path / "papers"
    papers.mkdir()
    body: dict[str, Any] = {field: [str(papers)]}
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=True, storage_dir=tmp_path / "store")
    session_mgr: AgentSessionManager = app.state.session_manager
    rebound: list[None] = []
    session_mgr.on_models_changed(lambda: rebound.append(None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        saved = await client.post("/api/settings", json=body)

    assert saved.status_code == 200
    assert cast(dict[str, Any], saved.json())[field] == body[field]
    assert rebound == []


async def test_an_unknown_ui_language_is_refused_and_not_saved(tmp_path: Path) -> None:
    """A language the heads have no catalog for is a 400, and the saved choice stays."""
    app = create_ui_app(static_dir=tmp_path, fallback_to_mock=True, storage_dir=tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        await client.post("/api/settings", json={"ui_language": "en"})
        res = await client.post("/api/settings", json={"ui_language": "fr"})

    assert res.status_code == 400
    assert (
        cast(dict[str, Any], res.json())["detail"]
        == "The settings could not be saved because the configuration is invalid."
    )
    reloaded = AgentSessionManager(storage_dir=tmp_path, fallback_to_mock=True)
    assert reloaded.get_settings()["ui_language"] == "en"


_KEY_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "VLLM_API_KEY",
    "LLM_PROVIDER",
    "GEMINI_MODEL",
    "OPENAI_MODEL",
)
