# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
"""Unit tests for FR-13 conversational chat surface, session APIs, and attribution badge."""

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from uclone_x.core.provenance import (
    AttemptRecord,
    ExecutionPath,
    Provenance,
    ServiceRef,
)
from uclone_x.errors import LLMProviderError, TokenBudgetExhaustedError
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    ModelResponse,
    ToolCallRequest,
)
from uclone_x.ui.app import (
    CANCELLED_TURN_TEXT,
    TRANSCRIPT_CANCELLED_ROLE,
    AgentSessionManager,
    _core_index_for_transcript,
    _is_presentable,
    create_ui_app,
)


def stubbed_ollama_with_model(
    reply: str = "Hello",
    model: str = "qwen3:8b",
) -> OllamaConnector:
    """Create an Ollama connector backed by httpx.MockTransport returning specific model attribution."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": model,
                "created_at": "2026-09-03T00:00:00.000000Z",
                "message": {"role": "assistant", "content": reply},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    return OllamaConnector(
        base_url="http://stub-ollama.invalid:11434",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_fr13_attribution_payload_independent_of_message_content(tmp_path: Path) -> None:
    """FR-13.4 Acceptance Test: per-turn attribution payload is independent of message text.

    When an assistant's generated text claims 'I am OpenAI GPT-4', but inference was
    served by Ollama (qwen3:8b), provenance.served_by faithfully reports
    'ollama:qwen3:8b' and NEVER derives attribution from message content.
    """
    claimed_identity_reply = "Hello! I am OpenAI GPT-4, a large language model trained by OpenAI."
    ollama_connector = stubbed_ollama_with_model(
        reply=claimed_identity_reply,
        model="qwen3:8b",
    )

    storage_dir = tmp_path / "sessions"
    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=storage_dir,
        llm=ollama_connector,
    )
    client = TestClient(app)

    res = client.post(
        "/api/turn",
        json={
            "message": "Who are you?",
            "agent_id": "agent-general",
            "session_id": "sess_fr13_attr",
        },
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["status"] == "success"
    # Reply text claims OpenAI GPT-4:
    assert "OpenAI GPT-4" in data["response"]

    # But provenance.served_by strictly reports the real serving provider:model:
    prov = data["provenance"]
    assert prov is not None
    assert prov["served_by"] == "ollama:qwen3:8b"
    assert "openai" not in prov["served_by"].lower()
    assert prov["degraded"] is False
    assert prov["path"] == "primary"
    assert prov["component"] == "uclone_x.llm.orchestrator"
    assert prov["producer"] == "agent-general"


def test_fr13_turn_alias_and_chat_endpoints(tmp_path: Path) -> None:
    """Verify /api/turn and /api/turn both execute turns and return P6 provenance."""
    mock_llm = MockLLMConnector(responses=["Response from turn endpoint"])
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=mock_llm)
    client = TestClient(app)

    # Test /api/turn POST endpoint
    res = client.post(
        "/api/turn",
        json={
            "message": "Execute turn via /api/turn",
            "agent_id": "agent-turn-test",
            "session_id": "sess_turn_1",
        },
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["status"] == "success"
    assert data["response"] == "Response from turn endpoint"
    assert data["turn_count"] == 1
    assert "provenance" in data
    assert data["provenance"]["served_by"] == "mock:mock-model"
    assert data["provenance"]["degraded"] is False
    assert "durability" in data
    assert data["durability"]["persisted"] is True


def test_per_agent_model_override_does_not_leak_across_agents(tmp_path: Path) -> None:
    """#1138: a per-request `model` override applies only to its own agent's session.

    The frontend sent `model` from `PlaygroundTab`'s "agent-model-override-select" until
    #1208 retired that surface; the route still accepts it, and the header picker the
    conversation kept still has no way to put it on the wire (a room's send posts
    `content` only). The route's own contract is what is pinned here: it sends `model`
    scoped to one agent's own turn requests. This pins the existing backend
    mechanism as sufficient for that -- no new endpoint or persistence was added:
    `_execute_turn_logic_impl` reads `model`/`llm_model` into `model_req`, threads it to
    `get_or_create_agent(..., model_name=model_req)` for a fresh agent, and re-asserts it
    every turn via `agent.hot_reload_llm(...)` for an already-cached one. Both apply only
    to the specific `agent_id:session_id`-keyed `BaseAgent` instance, in memory, never
    written to a persona file.

    Killed by: src/uclone_x/ui/app.py :: model_req = str(req.get("model") or req.get("llm_model") or "").strip() or None
    Becomes: model_req = None
    """
    mock_llm = MockLLMConnector()
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=mock_llm)
    client = TestClient(app)

    # agent-a picks its own model...
    res_a = client.post(
        "/api/turn",
        json={
            "message": "hello from a",
            "agent_id": "agent-a",
            "session_id": "sess_a",
            "model": "model-a",
        },
    )
    assert res_a.status_code == 200
    data_a = cast(dict[str, Any], res_a.json())
    assert data_a["provenance"]["served_by"] == "mock:model-a"

    # ...while agent-b, sending no override at all, is unaffected and keeps the default.
    res_b = client.post(
        "/api/turn",
        json={
            "message": "hello from b",
            "agent_id": "agent-b",
            "session_id": "sess_b",
        },
    )
    assert res_b.status_code == 200
    data_b = cast(dict[str, Any], res_b.json())
    assert data_b["provenance"]["served_by"] == "mock:mock-model"

    # A second turn on agent-a, with no `model` field this time, still uses the model it
    # was given earlier -- the per-turn re-application in `_execute_turn_logic_impl` reads
    # the cached agent's own `llm_config.model_name`, not agent-b's request or default.
    res_a2 = client.post(
        "/api/turn",
        json={
            "message": "hello again from a",
            "agent_id": "agent-a",
            "session_id": "sess_a",
        },
    )
    assert res_a2.status_code == 200
    data_a2 = cast(dict[str, Any], res_a2.json())
    assert data_a2["provenance"]["served_by"] == "mock:model-a"

    # And agent-b remains on the default, never having picked up agent-a's override.
    res_b2 = client.post(
        "/api/turn",
        json={
            "message": "hello again from b",
            "agent_id": "agent-b",
            "session_id": "sess_b",
        },
    )
    assert res_b2.status_code == 200
    data_b2 = cast(dict[str, Any], res_b2.json())
    assert data_b2["provenance"]["served_by"] == "mock:mock-model"


def test_fr13_session_history_and_sessions_listing(tmp_path: Path) -> None:
    """Verify /api/session/history and /api/sessions endpoints (FR-13.2)."""
    mock_llm = MockLLMConnector(responses=["First turn reply", "Second turn reply"])
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=mock_llm)
    client = TestClient(app)

    # Send 2 turns
    client.post(
        "/api/turn",
        json={
            "message": "Turn 1",
            "agent_id": "agent-general",
            "session_id": "sess_fr13_hist",
        },
    )
    client.post(
        "/api/turn",
        json={
            "message": "Turn 2",
            "agent_id": "agent-general",
            "session_id": "sess_fr13_hist",
        },
    )

    # Retrieve history via /api/session/history
    hist_res = client.get("/api/session/history?agent_id=agent-general&session_id=sess_fr13_hist")
    assert hist_res.status_code == 200
    hist_data = cast(dict[str, Any], hist_res.json())
    assert hist_data["session_id"] == "sess_fr13_hist"
    messages = cast(list[dict[str, Any]], hist_data["messages"])
    assert len(messages) == 4

    # Check agent message provenance in history
    agent_msg_1 = messages[1]
    assert agent_msg_1["sender"] == "agent"
    assert agent_msg_1["content"] == "First turn reply"
    assert "provenance" in agent_msg_1
    assert agent_msg_1["provenance"]["served_by"] == "mock:mock-model"
    assert agent_msg_1["provenance"]["degraded"] is False

    agent_msg_2 = messages[3]
    assert agent_msg_2["sender"] == "agent"
    assert agent_msg_2["content"] == "Second turn reply"
    assert "provenance" in agent_msg_2
    assert agent_msg_2["provenance"]["served_by"] == "mock:mock-model"

    # Retrieve sessions via /api/sessions (FR-13.2 view over Core store)
    sess_res = client.get("/api/sessions")
    assert sess_res.status_code == 200
    sess_data = cast(dict[str, Any], sess_res.json())
    assert "sessions" in sess_data
    session_list = cast(list[dict[str, Any]], sess_data["sessions"])
    found = [s for s in session_list if s["session_id"] == "sess_fr13_hist"]
    assert len(found) == 1
    assert found[0]["turn_counter"] == 2


def test_fr13_dispatch_endpoint_provenance(tmp_path: Path) -> None:
    """Verify /api/dispatch includes P6 structured provenance."""
    app = create_ui_app(static_dir=tmp_path)
    client = TestClient(app)

    res = client.post(
        "/api/dispatch",
        json={
            "task": "Perform security audit",
            "role": "analyst",
            "agent_id": "agent_worker_1",
        },
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["status"] == "dispatched"
    assert data["role"] == "analyst"
    assert "provenance" in data
    prov = data["provenance"]
    assert prov["component"] == "uclone_x.ui.dispatcher"
    assert prov["producer"] == "ui_dispatcher"
    assert prov["served_by"] == "dispatcher"
    assert prov["degraded"] is False
    assert prov["path"] == "primary"


def test_fr13_degraded_provenance_flag(tmp_path: Path) -> None:
    """Verify degraded=True is reported when failover occurs.

    Killed by: src/uclone_x/ui/app.py :: return ChatTurnOutcome.DEGRADED
    Becomes: return ChatTurnOutcome.COMPLETED
    """
    from uclone_x.agent.base import BaseAgent
    from uclone_x.agent.models import AgentConfig, AgentContext, AgentState, TurnResult

    # Mock agent that returns a degraded turn result
    storage_dir = tmp_path / "sessions"
    session_mgr = AgentSessionManager(storage_dir=storage_dir)

    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, session_manager=session_mgr)

    # Directly create degraded provenance
    degraded_prov = Provenance(
        path=ExecutionPath.FAILOVER,
        requested=ServiceRef(provider="openai", model="gpt-4o"),
        served_by=ServiceRef(provider="ollama", model="qwen2.5-coder:14b"),
        attempts=(
            AttemptRecord(
                provider="openai",
                model="gpt-4o",
                error_class="RateLimitError",
                span_id="spn_123456",
            ),
        ),
    )

    agent_config = AgentConfig(agent_id="agent-degraded-test", name="degraded-test")
    agent_context = AgentContext(
        session_id="sess_degraded",
        agent_id="agent-degraded-test",
        current_state=AgentState.IDLE,
    )
    agent = BaseAgent(
        config=agent_config,
        bus=session_mgr.bus,
        llm=MockLLMConnector(),
        tools=session_mgr.tools,
        context=agent_context,
        store=session_mgr.core_store,
    )

    async def mock_execute_turn(user_message: str) -> TurnResult:
        return TurnResult(
            turn_index=1,
            content="Failover answer from local model.",
            is_completed=True,
            provenance=degraded_prov,
        )

    agent.execute_turn = mock_execute_turn  # type: ignore[method-assign]
    session_mgr._agents["agent-degraded-test:sess_degraded"] = agent

    client = TestClient(app)
    res = client.post(
        "/api/turn",
        json={
            "message": "Trigger failover",
            "agent_id": "agent-degraded-test",
            "session_id": "sess_degraded",
        },
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    assert data["provenance"]["degraded"] is True
    assert data["provenance"]["served_by"] == "ollama:qwen2.5-coder:14b"
    # The Core's own meaning of degraded (#1007): an answer, from a model not requested.
    assert (data["status"], data["outcome"]) == ("success", "degraded")
    assert data["provenance"]["path"] == "failover"


def test_fr13_history_truncation_on_edit_and_resend(tmp_path: Path) -> None:
    """FR-13.7 Acceptance Test: Truncating history rolls back UI and Core session state."""
    mock_llm = MockLLMConnector(responses=["Reply 1", "Reply 2", "Reply 3", "Replacement Reply 2"])
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=mock_llm)
    client = TestClient(app)

    session_id = "sess_truncate_test"
    agent_id = "agent-general"

    # Send 3 conversational turns
    client.post(
        "/api/turn", json={"message": "Prompt 1", "agent_id": agent_id, "session_id": session_id}
    )
    client.post(
        "/api/turn", json={"message": "Prompt 2", "agent_id": agent_id, "session_id": session_id}
    )
    client.post(
        "/api/turn", json={"message": "Prompt 3", "agent_id": agent_id, "session_id": session_id}
    )

    # Verify history has 6 messages (3 user + 3 agent)
    hist_res = client.get(f"/api/session/history?agent_id={agent_id}&session_id={session_id}")
    assert hist_res.status_code == 200
    hist_data = cast(dict[str, Any], hist_res.json())
    assert len(hist_data["messages"]) == 6

    # Verify sessions endpoint reports 3 turns
    sess_res = client.get("/api/sessions")
    sess_data = cast(dict[str, Any], sess_res.json())
    target_sess = next(s for s in sess_data["sessions"] if s["session_id"] == session_id)
    assert target_sess["turn_counter"] == 3

    # Truncate back to index 2 (retaining prompt 1 + reply 1, removing prompt 2 & prompt 3 turns)
    trunc_res = client.post(
        "/api/session/history/truncate",
        json={"agent_id": agent_id, "session_id": session_id, "index": 2},
    )
    assert trunc_res.status_code == 200
    trunc_data = cast(dict[str, Any], trunc_res.json())
    assert trunc_data["status"] == "truncated"
    assert trunc_data["index"] == 2
    assert len(trunc_data["messages"]) == 2
    assert trunc_data["messages"][0]["content"] == "Prompt 1"
    assert trunc_data["messages"][1]["content"] == "Reply 1"

    # Verify GET /api/session/history reflects truncation
    hist_after = client.get(f"/api/session/history?agent_id={agent_id}&session_id={session_id}")
    assert hist_after.status_code == 200
    assert len(hist_after.json()["messages"]) == 2

    # Verify Core store was also truncated
    sess_after = client.get("/api/sessions")
    target_after = next(s for s in sess_after.json()["sessions"] if s["session_id"] == session_id)
    assert target_after["turn_counter"] == 1

    # Send replacement Turn 2
    res_turn2 = client.post(
        "/api/turn",
        json={"message": "Edited Prompt 2", "agent_id": agent_id, "session_id": session_id},
    )
    assert res_turn2.status_code == 200
    assert res_turn2.json()["response"] == "Replacement Reply 2"
    assert res_turn2.json()["turn_count"] == 2

    # History now has 4 messages [Prompt 1, Reply 1, Edited Prompt 2, Replacement Reply 2]
    hist_final = client.get(f"/api/session/history?agent_id={agent_id}&session_id={session_id}")
    assert len(hist_final.json()["messages"]) == 4
    assert hist_final.json()["messages"][2]["content"] == "Edited Prompt 2"
    assert hist_final.json()["messages"][3]["content"] == "Replacement Reply 2"


def test_fr13_truncate_via_delete_method(tmp_path: Path) -> None:
    """Verify DELETE /api/session/history with ?index=... also performs history truncation."""
    mock_llm = MockLLMConnector(responses=["Reply 1", "Reply 2"])
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=mock_llm)
    client = TestClient(app)

    session_id = "sess_del_trunc"
    agent_id = "agent-general"

    client.post("/api/turn", json={"message": "P1", "agent_id": agent_id, "session_id": session_id})
    client.post("/api/turn", json={"message": "P2", "agent_id": agent_id, "session_id": session_id})

    del_res = client.delete(
        f"/api/session/history?agent_id={agent_id}&session_id={session_id}&index=2"
    )
    assert del_res.status_code == 200
    del_data = cast(dict[str, Any], del_res.json())
    assert del_data["status"] == "truncated"
    assert len(del_data["messages"]) == 2


def test_fr13_truncate_validation_errors(tmp_path: Path) -> None:
    """Verify error handling on invalid truncate parameters."""
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir)
    client = TestClient(app)

    # Missing index
    res_missing = client.post("/api/session/history/truncate", json={"agent_id": "a1"})
    assert res_missing.status_code == 400

    # Negative index
    res_neg = client.post(
        "/api/session/history/truncate",
        json={"agent_id": "a1", "session_id": "s1", "index": -1},
    )
    assert res_neg.status_code == 400

    # Non-integer index
    res_invalid = client.post(
        "/api/session/history/truncate",
        json={"agent_id": "a1", "session_id": "s1", "index": "abc"},
    )
    assert res_invalid.status_code == 400


def test_fr13_unspecified_model_does_not_forge_baseagent(tmp_path: Path) -> None:
    """Verify that turns with unspecified model do NOT forge 'BaseAgent' as served_by."""
    from uclone_x.agent.base import BaseAgent
    from uclone_x.agent.models import AgentConfig, AgentContext, AgentState, TurnResult

    storage_dir = tmp_path / "sessions"
    session_mgr = AgentSessionManager(storage_dir=storage_dir)
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, session_manager=session_mgr)

    agent_config = AgentConfig(agent_id="agent-unspec", name="unspec")
    agent_context = AgentContext(
        session_id="sess_unspec",
        agent_id="agent-unspec",
        current_state=AgentState.IDLE,
    )
    agent = BaseAgent(
        config=agent_config,
        bus=session_mgr.bus,
        llm=MockLLMConnector(),
        tools=session_mgr.tools,
        context=agent_context,
        store=session_mgr.core_store,
    )

    # TurnResult with no provenance / unspecified model
    async def mock_execute_turn(user_message: str) -> TurnResult:
        return TurnResult(
            turn_index=1,
            content="Answer with unknown model provenance.",
            is_completed=True,
            provenance=None,
        )

    agent.execute_turn = mock_execute_turn  # type: ignore[method-assign]
    session_mgr._agents["agent-unspec:sess_unspec"] = agent

    client = TestClient(app)
    res = client.post(
        "/api/turn",
        json={
            "message": "Hello unknown model",
            "agent_id": "agent-unspec",
            "session_id": "sess_unspec",
        },
    )
    assert res.status_code == 200
    data = cast(dict[str, Any], res.json())
    prov = data["provenance"]
    assert prov["served_by"] is None
    assert prov["served_by"] != "BaseAgent"

    # Also test history read back when transcript is synthesized from Core state:
    # Persist core messages, then delete transcript file so get_session_history synthesizes from Core store
    from uclone_x.llm.models import ChatMessage, MessageRole

    agent.load_history(
        [
            ChatMessage(role=MessageRole.USER, content="Hello core"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Answer from core"),
        ],
        turn_counter=1,
        session_id="sess_unspec",
    )
    agent.persist_session("sess_unspec")

    transcript_path = session_mgr.get_session_path("sess_unspec")
    transcript_path.unlink(missing_ok=True)
    session_mgr._session_messages.pop("sess_unspec", None)

    hist_res = client.get("/api/session/history?agent_id=agent-unspec&session_id=sess_unspec")
    assert hist_res.status_code == 200
    hist_msgs = hist_res.json()["messages"]
    assert len(hist_msgs) == 2
    agent_msg = hist_msgs[1]
    assert agent_msg["sender"] == "agent"
    assert agent_msg["provenance"]["served_by"] is None
    assert agent_msg["provenance"]["served_by"] != "BaseAgent"


def test_health_and_diagnostics_contain_git_commit_and_started_at(tmp_path: Path) -> None:
    """Verify /api/health and /api/diagnostics return git_commit and started_at (#871)."""
    app = create_ui_app(static_dir=tmp_path, storage_dir=tmp_path / "sessions")
    client = TestClient(app)

    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    hdata = health_res.json()
    assert "git_commit" in hdata
    assert "started_at" in hdata
    assert isinstance(hdata["git_commit"], str)
    assert len(hdata["git_commit"]) > 0

    diag_res = client.get("/api/diagnostics")
    assert diag_res.status_code == 200
    ddata = diag_res.json()
    assert "git_commit" in ddata
    assert "started_at" in ddata


def test_sessions_sorted_by_recency(tmp_path: Path) -> None:
    """Verify /api/sessions returns sessions sorted by updated_at descending."""
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir)
    client = TestClient(app)

    session_mgr: AgentSessionManager = app.state.session_manager
    from uclone_x.agent.session import SessionState

    # Create older session
    s1 = SessionState(
        session_id="sess_older",
        agent_id="champion",
        created_at="2026-09-01T10:00:00Z",
        updated_at="2026-09-01T11:00:00Z",
        messages=(),
    )
    # Create newer session
    s2 = SessionState(
        session_id="sess_newer",
        agent_id="champion",
        created_at="2026-09-02T10:00:00Z",
        updated_at="2026-09-02T12:00:00Z",
        messages=(),
    )
    session_mgr.core_store.save(s1)
    session_mgr.core_store.save(s2)

    res = client.get("/api/sessions")
    assert res.status_code == 200
    sessions = res.json()["sessions"]
    assert len(sessions) == 2
    assert sessions[0]["session_id"] == "sess_newer"
    assert sessions[1]["session_id"] == "sess_older"


def test_hermes_model_adapts_system_prompt_for_steerability(tmp_path: Path) -> None:
    """Verify that requesting a Hermes model adapts the system prompt for steerability (#871)."""
    from uclone_x.agent.prompts import HERMES_STEERABILITY_POLICY, IDENTITY_GROUNDING

    mock_llm = MockLLMConnector(responses=["Creative story response"])
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=mock_llm)
    client = TestClient(app)

    res = client.post(
        "/api/turn",
        json={
            "message": "Creative request",
            "agent_id": "pioneer",
            "session_id": "sess_hermes_test",
            "model": "hermes3:8b",
        },
    )
    assert res.status_code == 200
    session_mgr: AgentSessionManager = app.state.session_manager
    agent = session_mgr._agents.get("pioneer:sess_hermes_test")
    assert agent is not None
    # Pioneer identity, grounding, and capabilities must be preserved;
    # steerability framing must adapt to HERMES_STEERABILITY_POLICY.
    assert "You are Pioneer" in agent.effective_system_prompt
    assert IDENTITY_GROUNDING in agent.effective_system_prompt
    assert HERMES_STEERABILITY_POLICY in agent.effective_system_prompt


def test_list_personas_api_returns_catalog(tmp_path: Path) -> None:
    """GET /api/personas returns the catalog of declarative personas discovered by PersonaRegistry (#897)."""
    mock_llm = MockLLMConnector()
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=mock_llm)
    client = TestClient(app)

    res = client.get("/api/personas")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["count"] > 0
    personas = {p["name"]: p for p in data["personas"]}
    assert "writer" in personas
    assert "artist" in personas
    assert "clone" in personas
    assert "file_write" in personas["writer"]["allowed_tools"]


def test_chat_creates_agent_from_declarative_persona(tmp_path: Path) -> None:
    """POST /api/turn with agent_id='writer' instantiates agent from PersonaRegistry (#897)."""
    mock_llm = MockLLMConnector(responses=["Once upon a time in a digital realm..."])
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=mock_llm)
    client = TestClient(app)

    res = client.post(
        "/api/turn",
        json={
            "message": "Write a story chapter",
            "agent_id": "writer",
            "session_id": "sess_writer_test",
        },
    )
    assert res.status_code == 200
    session_mgr: AgentSessionManager = app.state.session_manager
    agent = session_mgr._agents.get("writer:sess_writer_test")
    assert agent is not None
    assert (
        "Writer" in agent.effective_system_prompt
        or "storyteller" in agent.effective_system_prompt.lower()
    )
    assert "file_write" in agent.config.allowed_tools


def test_saved_prompt_keeps_the_client_turn_id_the_page_sent(tmp_path: Path) -> None:
    """#1000: a page finds its own turn's saved copy by the id it sent, not by the prompt.

    Two turns with the same prompt are told apart only by the id each was sent with, so the
    id must be on the saved user entry, and still there once the record is read from disk.
    A turn sent without one (an older page, `ucx`'s own clients) saves no such field.

    Killed by: src/uclone_x/ui/app.py ::
        user_msg_entry["client_turn_id"] = client_turn_id
    Becomes: pass
    """
    storage_dir = tmp_path / "sessions"
    replies = ["First", "Second", "Third"]
    app = create_ui_app(
        static_dir=tmp_path, storage_dir=storage_dir, llm=MockLLMConnector(responses=replies)
    )
    client = TestClient(app)
    turn = {"message": "Same prompt", "agent_id": "agent-general", "session_id": "sess_turn_id"}
    longest = "a" * 128
    for extra in ({"client_turn_id": "turn-abc"}, {}, {"client_turn_id": longest}):
        assert client.post("/api/turn", json={**turn, **extra}).json()["status"] == "success"

    served = client.get("/api/session/history?agent_id=agent-general&session_id=sess_turn_id")
    from_disk = AgentSessionManager(storage_dir=storage_dir).get_session_history(
        "agent-general", "sess_turn_id"
    )
    for messages in (cast(list[dict[str, Any]], served.json()["messages"]), from_disk):
        prompts = [m for m in messages if m["sender"] == "user"]
        assert [m["content"] for m in prompts] == ["Same prompt"] * 3
        assert [m.get("client_turn_id") for m in prompts] == ["turn-abc", None, longest]
        assert "client_turn_id" not in prompts[1]


REFUSED_TURN_IDS: list[tuple[str, object]] = [
    ("empty", ""),
    ("not a string", 42),
    ("null", None),
    ("longer than 128", "a" * 129),
    ("a space", "turn abc"),
    ("a path", "turn/../abc"),
    ("markup", "<b>turn</b>"),
    ("non-ASCII", "türn-abc"),
]


@pytest.mark.parametrize(("case", "turn_id"), REFUSED_TURN_IDS)
def test_a_turn_sent_with_an_unusable_client_turn_id_is_refused_before_it_runs(
    tmp_path: Path, case: str, turn_id: object
) -> None:
    """#1007: a `client_turn_id` the server would not keep is refused with 422, not dropped.

    The id is kept on the saved prompt, so it is bounded to 1-128 of `A-Z a-z 0-9 - _`.
    Anything else sent in the field is refused before the turn runs: the model is never
    asked, nothing is saved, and the refusal names the field and the rule.

    The second arm sent the same id to `/api/chat/stream`, where the point was that the
    refusal is the HTTP status rather than an error inside a 200 event stream. #1208
    retired that route, and with it the only endpoint that could have answered 200 here.

    Killed by: src/uclone_x/ui/app.py :: _CLIENT_TURN_ID.fullmatch(turn_id)
    Becomes: _CLIENT_TURN_ID.match(turn_id)
    Killed by: src/uclone_x/ui/app.py :: [A-Za-z0-9_-]{1,128}
    Becomes: [A-Za-z0-9_-]{0,129}
    Killed by: src/uclone_x/ui/app.py :: not isinstance(turn_id, str) or
    Becomes: False or
    Killed by: src/uclone_x/ui/app.py :: client_turn_id = _client_turn_id(req)
    Becomes: client_turn_id = req.get("client_turn_id")
    """
    llm = MockLLMConnector(responses=["Reply to the accepted turn"])
    app = create_ui_app(static_dir=tmp_path, storage_dir=tmp_path / "sessions", llm=llm)
    client = TestClient(app)
    turn = {"message": "Hello", "agent_id": "agent-general", "session_id": "sess_refused_id"}

    refused = client.post("/api/turn", json={**turn, "client_turn_id": turn_id})

    assert refused.status_code == 422, (case, refused.text)
    assert "client_turn_id" in refused.json()["detail"]
    history = "/api/session/history?agent_id=agent-general&session_id=sess_refused_id"
    assert client.get(history).json()["messages"] == []
    # The refused turn did not take the scripted reply: the model was never asked.
    accepted = client.post("/api/turn", json={**turn, "client_turn_id": "turn-mfk3x1a2-4fzyq0ab"})
    assert accepted.json()["response"] == "Reply to the accepted turn"


# --------------------------------------------------------------------------------------
# A failed chat turn (#969): saved as a failure rather than as a reply, kept out of what
# the model is shown, and retried without showing the model the prompt twice.
# --------------------------------------------------------------------------------------


class _ScriptedChatConnector(MockLLMConnector):
    """Raises a provider error on the calls in `fail_on` (1-based); records every request."""

    def __init__(self, fail_on: set[int], responses: list[str]) -> None:
        super().__init__(responses=responses)
        self._fail_on = fail_on
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) in self._fail_on:
            raise LLMProviderError("provider unavailable (scripted)")
        return await super().generate(request)


def _said(request: LLMRequest) -> list[tuple[str, str | None]]:
    """The conversation a request showed the model, without its system prompt."""
    return [(m.role.value, m.content) for m in request.messages if m.role is not MessageRole.SYSTEM]


def test_a_failed_chat_turn_is_saved_as_a_failure_and_not_as_a_reply(tmp_path: Path) -> None:
    """A failure is not something the agent said, and the saved record says so (#969).

    The chat head saved a failed turn as `role: "assistant"` with `Error: ...` content, so
    the saved conversation claimed the agent replied with an error message, and the only
    mark on it was that wording. The record now carries its own role and the error
    beside the text the page showed, and nothing of it reaches the model's next request
    -- neither from the live agent nor from a conversation reloaded from disk.

    Killed by: src/uclone_x/ui/app.py :: agent_msg_entry["role"] = TRANSCRIPT_FAILURE_ROLE
    Becomes: pass
    """
    storage_dir = tmp_path / "sessions"
    connector = _ScriptedChatConnector({1}, ["The answer", "The second answer"])
    client = TestClient(create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=connector))
    turn = {"agent_id": "agent-general", "session_id": "sess_failed"}

    failed = client.post("/api/turn", json={**turn, "message": "First question"}).json()
    assert failed["status"] == "error"

    served = client.get("/api/session/history?agent_id=agent-general&session_id=sess_failed")
    from_disk = AgentSessionManager(storage_dir=storage_dir).get_session_history(
        "agent-general", "sess_failed"
    )
    for messages in (cast(list[dict[str, Any]], served.json()["messages"]), from_disk):
        assert [(m["sender"], m["role"]) for m in messages] == [
            ("user", "user"),
            ("agent", "failure"),
        ]
        assert messages[1]["error"] == "provider unavailable (scripted)"
        assert messages[1]["content"] == failed["response"]

    answered = client.post("/api/turn", json={**turn, "message": "Second question"}).json()
    assert answered["status"] == "success"
    assert _said(connector.requests[-1]) == [
        ("user", "First question"),
        ("user", "Second question"),
    ]

    # The same conversation, reopened by a fresh process from what is on disk.
    reopened = TestClient(
        create_ui_app(static_dir=tmp_path / "again", storage_dir=storage_dir, llm=connector)
    )
    assert (
        reopened.post("/api/turn", json={**turn, "message": "Third"}).json()["status"] == "success"
    )
    assert all("provider unavailable" not in (c or "") for _, c in _said(connector.requests[-1]))


class _RaisingChatConnector(MockLLMConnector):
    """Raises `exc` on every call: a turn that ends in exactly that error."""

    def __init__(self, exc: Exception) -> None:
        super().__init__(responses=["unused"])
        self._exc = exc

    async def generate(self, request: LLMRequest) -> ModelResponse:
        raise self._exc


def test_a_chat_turn_a_spent_budget_refused_says_a_retry_is_refused_too(tmp_path: Path) -> None:
    """A budget refusal carries its structured reason, live and saved (#969 question 1).

    The chat head offered "Retry last turn" on every failed turn. On a spent token or cost
    budget the retry meets the same ceiling -- the ledger only grows -- so it could only
    append another failure. The turn's `refusal` is what the head decides it from: the
    `RoomTurnRefusal` rooms already carry (#1022), from the turn's `stop_reason`, never
    from the wording of `error`. The error itself is carried beside it, for the head to
    show with the remedy.

    Killed by: src/uclone_x/room/models.py :: if stop_reason == "budget_exceeded":
    Becomes: if False:
    Killed by: src/uclone_x/ui/app.py :: agent_msg_entry["refusal"] = refusal
    Becomes: pass
    """
    from uclone_x.llm.budget import TokenBudgetManager

    session_id = "sess_spent"
    budget = TokenBudgetManager()
    budget.configure_session(session_id, max_tokens=0)
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path,
            storage_dir=tmp_path / "sessions",
            llm=MockLLMConnector(responses=["unused"]),
            budget_tracker=budget,
        )
    )

    refused = client.post(
        "/api/turn",
        json={"message": "Hello", "agent_id": "agent-general", "session_id": session_id},
    ).json()

    assert refused["status"] == "error"
    assert refused["outcome"] == "failed"
    assert refused["refusal"] == "budget_exceeded"
    assert refused["error"] and "limit exceeded" in refused["error"].lower()
    saved = _transcript(client, session_id)[-1]
    assert (saved["role"], saved["outcome"], saved["refusal"]) == (
        "failure",
        "failed",
        "budget_exceeded",
    )
    assert saved["error"] == refused["error"]


@pytest.mark.parametrize(
    ("case", "exc"),
    [
        ("provider outage", LLMProviderError("provider unavailable (scripted)")),
        # An LLMError, not a BudgetExceededError: the model spent its own completion
        # allowance on one answer. A retry is a new request with a new allowance.
        ("token budget exhausted", TokenBudgetExhaustedError("reasoning ate it (scripted)")),
    ],
)
def test_a_failed_chat_turn_a_retry_can_get_past_carries_no_refusal(
    tmp_path: Path, case: str, exc: Exception
) -> None:
    """Every other failure keeps Retry: it states no refusal, live or saved (#969).

    Killed by: src/uclone_x/ui/app.py :: refusal = turn_refusal(turn_result.stop_reason) if turn_result.error is not None else None
    Becomes: refusal = turn_refusal("budget_exceeded") if turn_result.error is not None else None
    """
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path,
            storage_dir=tmp_path / "sessions",
            llm=_RaisingChatConnector(exc),
        )
    )
    turn = {"message": "Hello", "agent_id": "agent-general", "session_id": "sess_retryable"}

    failed = client.post("/api/turn", json=turn).json()

    assert (failed["status"], failed["outcome"], failed["refusal"]) == ("error", "failed", None), (
        case
    )
    assert failed["error"] == str(exc), case
    saved = _transcript(client, "sess_retryable")[-1]
    assert (saved["role"], saved["outcome"]) == ("failure", "failed"), case
    assert "refusal" not in saved, case


@pytest.mark.parametrize(
    ("stop_reason", "refused"),
    [
        ("budget_exceeded", True),
        # `run_steps` resets per turn: a retry starts with the whole step budget.
        ("step_budget_exceeded", False),
        ("blocked_by_hook", False),
        ("cancelled", False),
        (None, False),
    ],
)
def test_only_a_token_or_cost_ceiling_is_a_refusal_a_retry_meets_again(
    stop_reason: str | None, refused: bool
) -> None:
    """The one mapping both heads decide Retry from (#969), over the Core's stop reasons.

    Killed by: src/uclone_x/room/models.py :: if stop_reason == "budget_exceeded":
    Becomes: if stop_reason is not None and "budget_exceeded" in stop_reason:
    """
    from uclone_x.room.models import RoomTurnRefusal, turn_refusal

    assert turn_refusal(stop_reason) == (RoomTurnRefusal.BUDGET_EXCEEDED if refused else None)


def test_only_a_turn_that_reached_an_answer_is_named_by_the_core_s_degraded(
    tmp_path: Path,
) -> None:
    """The turn's `outcome` uses `degraded` in the Core's meaning only (#1007 item 4).

    `provenance.degraded` on a chat row is broader than the Core's: it is also true for a
    failed turn, an offline one, a cancelled one and a reply that answered but was not
    persisted. A head chipping that flag as "degraded" named all of those the same thing.
    `outcome` is the one field a head reads: `failed` for any turn that ended in an error,
    `degraded` only when the Core's own provenance says the served model is not the one
    requested, and `completed` otherwise -- a durability fault is reported on
    `durability`, not as a different answer.

    Killed by: src/uclone_x/ui/app.py :: outcome = _answered_outcome(turn_result)
    Becomes: outcome = ChatTurnOutcome.DEGRADED if is_degraded else _answered_outcome(turn_result)
    Killed by: src/uclone_x/ui/app.py :: if turn_result.error is not None or not turn_result.is_completed:
    Becomes: if False:
    """
    from unittest.mock import patch

    from uclone_x.agent.base import BaseAgent

    storage_dir = tmp_path / "sessions"
    connector = _ScriptedChatConnector({2}, ["The answer", "Another answer"])
    client = TestClient(create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=connector))
    turn = {"agent_id": "agent-general", "session_id": "sess_outcomes"}

    answered = client.post("/api/turn", json={**turn, "message": "one"}).json()
    failed = client.post("/api/turn", json={**turn, "message": "two"}).json()
    with patch.object(BaseAgent, "persist_session", side_effect=OSError("disk full (scripted)")):
        unpersisted = client.post("/api/turn", json={**turn, "message": "three"}).json()

    assert (answered["status"], answered["outcome"]) == ("success", "completed")
    assert (failed["status"], failed["outcome"]) == ("error", "failed")
    # Degraded in the head's old sense, and still an answer the served model gave.
    assert unpersisted["provenance"]["degraded"] is True
    assert (unpersisted["status"], unpersisted["outcome"]) == ("warning", "completed")
    assert "error" not in answered or answered["error"] is None
    saved = [m.get("outcome") for m in _transcript(client, "sess_outcomes") if m["role"] != "user"]
    assert saved == ["completed", "failed", "completed"]


def test_a_chat_turn_that_raised_is_a_failure_with_its_error(tmp_path: Path) -> None:
    """The path that never reached a turn result states the same structured outcome.

    Killed by: src/uclone_x/ui/app.py :: "outcome": ChatTurnOutcome.FAILED,  # raised
    Becomes: "outcome": None,
    """
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path, storage_dir=tmp_path / "sessions", llm=MockLLMConnector()
        )
    )

    raised = client.post(
        "/api/turn",
        json={"message": "hello", "agent_id": "Champion", "session_id": "sess_bad_name"},
    ).json()

    assert (raised["status"], raised["outcome"], raised["refusal"]) == ("error", "failed", None)
    assert raised["error"] and "Champion" in raised["error"]


def test_an_old_transcript_keeps_its_failed_turn_on_screen_and_out_of_the_model(
    tmp_path: Path,
) -> None:
    """A transcript saved before failure records existed still reads as it did (#969).

    Its failed turns are `role: "assistant"` rows reading `Error: ...` with degraded
    provenance. The page is still served them unchanged. A conversation with no Core record
    is rebuilt from that transcript, and the rebuild handed those rows to the model as
    replies the agent had given.

    Killed by: src/uclone_x/ui/app.py :: and content.startswith(LEGACY_FAILURE_PREFIX)
    Becomes: and False
    """
    storage_dir = tmp_path / "sessions"
    failure_row = {
        "id": "agent-1",
        "sender": "agent",
        "role": "assistant",
        "content": "Error: provider unavailable",
        "provenance": {"degraded": True, "path": "FAILOVER", "served_by": "agent.core"},
    }
    reply_row = {
        "id": "agent-2",
        "sender": "agent",
        "role": "assistant",
        "content": "Answer two",
        "provenance": {"degraded": False, "path": "PRIMARY", "served_by": "mock:mock-model"},
    }
    old = [
        {"id": "user-1", "sender": "user", "role": "user", "content": "Question one"},
        failure_row,
        {"id": "user-2", "sender": "user", "role": "user", "content": "Question two"},
        reply_row,
    ]
    AgentSessionManager(storage_dir=storage_dir).save_session_record(
        session_id="sess_old", agent_id="agent-general", messages=old, turns=2
    )
    connector = _ScriptedChatConnector(set(), ["Answer three"])
    client = TestClient(create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=connector))

    served = client.get("/api/session/history?agent_id=agent-general&session_id=sess_old")
    assert served.json()["messages"] == old

    turn = {"agent_id": "agent-general", "session_id": "sess_old", "message": "Question three"}
    assert client.post("/api/turn", json=turn).json()["status"] == "success"
    assert _said(connector.requests[-1]) == [
        ("user", "Question one"),
        ("user", "Question two"),
        ("assistant", "Answer two"),
        ("user", "Question three"),
    ]


def test_retrying_a_failed_chat_turn_shows_the_model_the_prompt_once(tmp_path: Path) -> None:
    """The page's Retry sends the failed prompt again; the model sees it once (#969).

    Reproduced before the fix: `BaseAgent.execute_turn` puts the prompt in history before
    calling the model and a failed turn leaves it there, so the retry's request read
    `[("user", "Same prompt"), ("user", "Same prompt")]`. Sending the same words after
    they were answered is a new message, and still reaches the model as one.

    Killed by: src/uclone_x/agent/base.py :: if not _repeats_unanswered_prompt(self._history, user_prompt):
    Becomes: if True:
    """
    connector = _ScriptedChatConnector({1}, ["Answered", "Answered again"])
    client = TestClient(
        create_ui_app(static_dir=tmp_path, storage_dir=tmp_path / "sessions", llm=connector)
    )
    turn = {"agent_id": "agent-general", "session_id": "sess_retry", "message": "Same prompt"}

    first = client.post("/api/turn", json={**turn, "client_turn_id": "turn-1"}).json()
    assert first["status"] == "error"
    retried = client.post("/api/turn", json={**turn, "client_turn_id": "turn-2"}).json()
    assert retried["status"] == "success"
    assert _said(connector.requests[1]) == [("user", "Same prompt")]

    again = client.post("/api/turn", json={**turn, "client_turn_id": "turn-3"}).json()
    assert again["status"] == "success"
    assert _said(connector.requests[2]) == [
        ("user", "Same prompt"),
        ("assistant", "Answered"),
        ("user", "Same prompt"),
    ]

    # The page's record keeps both sends: the failure and the retry that answered it.
    history = client.get("/api/session/history?agent_id=agent-general&session_id=sess_retry")
    roles = [m["role"] for m in history.json()["messages"]]
    assert roles == ["user", "failure", "user", "assistant", "user", "assistant"]


# --------------------------------------------------------------------------------------
# A failure row is a transcript row the Core has no message for (#1023): truncation counts
# it in the index it cuts the Core at, and compaction re-derives the transcript without it.
# --------------------------------------------------------------------------------------


def _transcript(client: TestClient, session_id: str) -> list[dict[str, Any]]:
    """The transcript `GET /api/session/history` serves for a session."""
    res = client.get(f"/api/session/history?agent_id=agent-general&session_id={session_id}")
    assert res.status_code == 200, res.text
    return cast(list[dict[str, Any]], res.json()["messages"])


def _core_dialogue(client: TestClient, session_id: str) -> list[tuple[str, str | None]]:
    """The Core's conversation for a session, without the anchored system prompt."""
    mgr = cast(Any, client.app).state.session_manager
    state = mgr.core_store.load(session_id)
    assert state is not None, f"no Core record for {session_id!r}"
    return [(m.role.value, m.content) for m in state.messages if m.role is not MessageRole.SYSTEM]


def _conversation_with_a_failed_turn(
    tmp_path: Path, session_id: str, connector: _ScriptedChatConnector
) -> TestClient:
    """Four sends on one session, the second of which fails.

    Eight transcript rows against seven Core messages: the Core holds the failed turn's
    prompt and no reply to it, so it has no message for the failure row at index 3.
    """
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path / session_id,
            storage_dir=tmp_path / "sessions",
            llm=connector,
        )
    )
    for i in range(4):
        posted = client.post(
            "/api/turn",
            json={"agent_id": "agent-general", "session_id": session_id, "message": f"msg {i}"},
        )
        assert posted.status_code == 200, posted.text
    return client


def test_truncating_at_or_after_a_failure_row_cuts_the_core_where_the_user_clicked(
    tmp_path: Path,
) -> None:
    """The truncation index is a transcript offset, and a failure row is transcript-only (#1023).

    `_truncate_core_messages` cuts the Core at the number the page counted over the
    transcript it renders, which is the mapping #872 established. A failed turn is in that
    transcript and not in the Core -- the Core holds the prompt and no reply to it -- so
    from the first failure onward the two lists have different lengths and one number cuts
    them in different places.

    Measured before the fix, on the conversation this test builds: truncating to index 4,
    where the failure row is the last row kept, left `msg 2` -- the prompt of the turn the
    user had just deleted -- in the Core for the next turn to answer against; to index 5 it
    left `reply 1` there as well. Truncating before the failure row was, and stays, right.

    The stored transcript is not renumbered. The rows are what they were; the index is
    mapped onto the messages the Core actually holds.

    Killed by: src/uclone_x/ui/app.py :: core_index = _core_index_for_transcript(truncated_transcript, core_messages)
    Becomes: core_index = index
    """
    failure_text = "Error: provider unavailable (scripted)"
    cases: list[tuple[int, list[tuple[str, str]], list[tuple[str, str | None]]]] = [
        # Before the failure row the two lists still agree, and must go on agreeing.
        (
            2,
            [("user", "msg 0"), ("assistant", "reply 0")],
            [("user", "msg 0"), ("assistant", "reply 0")],
        ),
        # The row before the failure row: the whole failed turn is dropped from both.
        (
            3,
            [("user", "msg 0"), ("assistant", "reply 0"), ("user", "msg 1")],
            [("user", "msg 0"), ("assistant", "reply 0"), ("user", "msg 1")],
        ),
        # At the failure row -- it is kept, and the Core has nothing to keep for it.
        (
            4,
            [
                ("user", "msg 0"),
                ("assistant", "reply 0"),
                ("user", "msg 1"),
                ("failure", failure_text),
            ],
            [("user", "msg 0"), ("assistant", "reply 0"), ("user", "msg 1")],
        ),
        # After it: one more transcript row, one more Core message, the offset unchanged.
        (
            5,
            [
                ("user", "msg 0"),
                ("assistant", "reply 0"),
                ("user", "msg 1"),
                ("failure", failure_text),
                ("user", "msg 2"),
            ],
            [
                ("user", "msg 0"),
                ("assistant", "reply 0"),
                ("user", "msg 1"),
                ("user", "msg 2"),
            ],
        ),
    ]

    for index, expected_rows, expected_core in cases:
        session_id = f"sess_1023_truncate_{index}"
        connector = _ScriptedChatConnector({2}, ["reply 0", "reply 1", "reply 2"])
        client = _conversation_with_a_failed_turn(tmp_path, session_id, connector)
        assert [(m["role"], m["content"]) for m in _transcript(client, session_id)] == [
            ("user", "msg 0"),
            ("assistant", "reply 0"),
            ("user", "msg 1"),
            ("failure", failure_text),
            ("user", "msg 2"),
            ("assistant", "reply 1"),
            ("user", "msg 3"),
            ("assistant", "reply 2"),
        ]

        truncated = client.post(
            "/api/session/history/truncate",
            json={"agent_id": "agent-general", "session_id": session_id, "index": index},
        )
        assert truncated.status_code == 200, truncated.text
        # The page reads the truncated conversation back, failure row included.
        assert [(m["role"], m["content"]) for m in _transcript(client, session_id)] == (
            expected_rows
        ), index
        assert _core_dialogue(client, session_id) == expected_core, index

    # The harm the Core cut prevents, where it is actually observable: the next turn.
    session_id = "sess_1023_truncate_next_turn"
    connector = _ScriptedChatConnector({2}, ["reply 0", "reply 1", "reply 2", "after"])
    client = _conversation_with_a_failed_turn(tmp_path, session_id, connector)
    assert (
        client.post(
            "/api/session/history/truncate",
            json={"agent_id": "agent-general", "session_id": session_id, "index": 4},
        ).status_code
        == 200
    )
    client.post(
        "/api/turn",
        json={
            "agent_id": "agent-general",
            "session_id": session_id,
            "message": "after the cut",
        },
    )
    assert _said(connector.requests[-1]) == [
        ("user", "msg 0"),
        ("assistant", "reply 0"),
        ("user", "msg 1"),
        ("user", "after the cut"),
    ]


def test_an_old_transcripts_error_row_is_counted_the_same_way_when_truncating(
    tmp_path: Path,
) -> None:
    """A transcript saved before failure records existed truncates identically (#1023, #969).

    Its failed turns are `role: "assistant"` rows reading `Error: ...` under degraded
    provenance, and `reconstruct_history` keeps them out of the history it rebuilds, so the
    Core has no message for them either. The mapping reaches both spellings through
    `_is_failure_entry`, reached through `_records_a_turn_not_spoken`, rather than knowing
    only about the new one.

    Killed by: src/uclone_x/ui/app.py :: core_index = _core_index_for_transcript(truncated_transcript, core_messages)
    Becomes: core_index = index
    """
    storage_dir = tmp_path / "sessions"
    session_id = "sess_1023_legacy_truncate"
    old: list[dict[str, Any]] = [
        {"id": "user-1", "sender": "user", "role": "user", "content": "Question one"},
        {
            "id": "agent-1",
            "sender": "agent",
            "role": "assistant",
            "content": "Error: provider unavailable",
            "provenance": {"degraded": True, "path": "FAILOVER", "served_by": "agent.core"},
        },
        {"id": "user-2", "sender": "user", "role": "user", "content": "Question two"},
        {
            "id": "agent-2",
            "sender": "agent",
            "role": "assistant",
            "content": "Answer two",
            "provenance": {"degraded": False, "path": "PRIMARY", "served_by": "mock:mock-model"},
        },
    ]
    AgentSessionManager(storage_dir=storage_dir).save_session_record(
        session_id=session_id, agent_id="agent-general", messages=old, turns=2
    )
    connector = _ScriptedChatConnector(set(), ["Answer three"])
    client = TestClient(create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=connector))
    # One turn brings the conversation up as a live agent, rebuilt from that transcript.
    assert (
        client.post(
            "/api/turn",
            json={
                "agent_id": "agent-general",
                "session_id": session_id,
                "message": "Question three",
            },
        ).status_code
        == 200
    )
    assert _core_dialogue(client, session_id) == [
        ("user", "Question one"),
        ("user", "Question two"),
        ("assistant", "Answer two"),
        ("user", "Question three"),
        ("assistant", "Answer three"),
    ]

    truncated = client.post(
        "/api/session/history/truncate",
        json={"agent_id": "agent-general", "session_id": session_id, "index": 2},
    )
    assert truncated.status_code == 200, truncated.text
    # The page keeps the prompt and the row that says the turn failed, unchanged.
    assert [(m["role"], m["content"]) for m in _transcript(client, session_id)] == [
        ("user", "Question one"),
        ("assistant", "Error: provider unavailable"),
    ]
    assert _core_dialogue(client, session_id) == [("user", "Question one")]


def test_a_degraded_reply_beginning_error_is_never_cut_from_the_core(tmp_path: Path) -> None:
    """A real reply the Core holds must survive a truncation that deletes nothing (#1023).

    `_is_failure_entry` calls an assistant row a failed turn on the *wording* of its
    content plus degraded provenance. Answering "may the model be shown this row?" that
    way costs one row of context when it is wrong (#1022 accepted that). Answering "does
    the Core hold a message for this row?" the same way **deletes** one, because the Core
    does hold that reply and the count decides where the Core is cut.

    `degraded is True` on a perfectly successful reply is ordinary: `ui/app.py` sets it
    when the Core session failed to persist (driven here), and `Provenance.primary()`
    yields it for plain provider-side model alias resolution. The only other condition is
    a reply that opens with `Error: ` -- a coding assistant quoting a compiler error.

    Measured before the fix, truncating to `index == len(transcript)` -- the user deletes
    nothing -- cut `Go returns an error value.` out of the Core and off the disk while the
    page went on showing it.

    Killed by: src/uclone_x/ui/app.py :: matched = candidate < len(presentable) and _core_key(presentable[candidate]) == key
    Becomes: matched = candidate < len(presentable) and _core_key(presentable[candidate]) == key and not _is_failure_entry(key[0], entry)
    """
    from unittest.mock import patch

    from uclone_x.agent.base import BaseAgent

    storage_dir = tmp_path / "sessions"
    session_id = "sess_1023_degraded"
    connector = _ScriptedChatConnector(
        set(),
        [
            "Error: handling in Python uses try/except blocks.",
            "Go returns an error value.",
            "Third answer",
        ],
    )
    client = TestClient(create_ui_app(static_dir=tmp_path, storage_dir=storage_dir, llm=connector))
    turn = {"agent_id": "agent-general", "session_id": session_id}

    # A successful turn whose reply begins `Error: `, marked degraded by a failing
    # `persist_session` -- genuine app code, and the reply itself is real and in the Core.
    with patch.object(BaseAgent, "persist_session", side_effect=OSError("disk full (scripted)")):
        first = client.post("/api/turn", json={**turn, "message": "how do I handle errors?"}).json()
    # The turn answered; only its durability was degraded, which is exactly the shape.
    assert first["status"] == "warning"
    assert first["response"] == "Error: handling in Python uses try/except blocks."
    assert client.post("/api/turn", json={**turn, "message": "and in Go?"}).status_code == 200

    rows = [(m["role"], m["content"]) for m in _transcript(client, session_id)]
    assert rows == [
        ("user", "how do I handle errors?"),
        ("assistant", "Error: handling in Python uses try/except blocks."),
        ("user", "and in Go?"),
        ("assistant", "Go returns an error value."),
    ]
    # The row the wording test misreads is degraded, and the Core holds it.
    assert _transcript(client, session_id)[1]["provenance"]["degraded"] is True
    before = _core_dialogue(client, session_id)
    assert len(before) == 4

    truncated = client.post(
        "/api/session/history/truncate",
        json={**turn, "index": len(rows)},
    )
    assert truncated.status_code == 200, truncated.text

    # Nothing was deleted, so nothing may be missing -- in memory or on disk.
    assert _core_dialogue(client, session_id) == before
    reopened = AgentSessionManager(storage_dir=storage_dir).core_store.load(session_id)
    assert reopened is not None
    assert "Go returns an error value." in [m.content for m in reopened.messages]

    # And the model is still shown the reply on the next turn.
    client.post("/api/turn", json={**turn, "message": "and in Rust?"})
    assert ("assistant", "Go returns an error value.") in _said(connector.requests[-1])


def test_the_mapping_counts_a_row_the_core_holds_at_every_index(tmp_path: Path) -> None:
    """The count is derived from the Core, not from what a row's text looks like (#1023).

    The same degraded `Error: ` reply, cut at every index rather than only at the end: a
    row the Core holds advances the mapping wherever it sits, so no prefix of this
    transcript maps to fewer Core messages than it has rows.

    Killed by: src/uclone_x/ui/app.py :: matched = candidate < len(presentable) and _core_key(presentable[candidate]) == key
    Becomes: matched = candidate < len(presentable) and _core_key(presentable[candidate]) == key and not _is_failure_entry(key[0], entry)
    """
    from uclone_x.agent.session import SessionState
    from uclone_x.llm.models import ChatMessage

    storage_dir = tmp_path / "sessions"
    mgr = AgentSessionManager(storage_dir=storage_dir)
    dialogue = [
        ("user", "how do I handle errors?"),
        ("assistant", "Error: handling in Python uses try/except blocks."),
        ("user", "and in Go?"),
        ("assistant", "Go returns an error value."),
    ]
    for index in range(len(dialogue) + 1):
        session_id = f"sess_1023_every_{index}"
        transcript: list[dict[str, Any]] = [
            {
                "id": f"m-{i}",
                "sender": "user" if role == "user" else "agent",
                "role": role,
                "content": content,
                "provenance": {"degraded": True, "path": "PRIMARY", "served_by": "m"},
            }
            for i, (role, content) in enumerate(dialogue)
        ]
        mgr.save_session_record(
            session_id=session_id, agent_id="agent-general", messages=transcript, turns=2
        )
        mgr.core_store.save(
            SessionState(
                session_id=session_id,
                agent_id="agent-general",
                turn_counter=2,
                messages=(
                    ChatMessage(role=MessageRole.SYSTEM, content="You are Champion."),
                    *(
                        ChatMessage(role=MessageRole(role), content=content)
                        for role, content in dialogue
                    ),
                ),
            )
        )

        mgr.truncate_session_history(agent_id="agent-general", session_id=session_id, index=index)
        state = mgr.core_store.load(session_id)
        assert state is not None
        kept = [
            (m.role.value, m.content) for m in state.messages if m.role is not MessageRole.SYSTEM
        ]
        # Every row of this transcript is Core-backed, so the Core keeps exactly as many.
        assert kept == dialogue[:index], index


def test_a_failed_turn_still_counts_as_a_turn_after_truncation(tmp_path: Path) -> None:
    """The turn counter counts a failed turn; the Core's own counter does too (#1023).

    Two numbers in `truncate_session_history` are deliberately different, and this pins
    which is which. `new_turn_counter` counts the transcript **raw** -- a failure row has
    `sender: "agent"` and is counted -- because `turn_counter` is the lifetime figure for
    turns this session has *taken*, and a turn that failed was still taken: measured, four
    sends of which one failed leave the Core's own `turn_counter` at 4. The line beside it
    maps the index onto the messages the Core holds, which is a different question.

    Killed by: src/uclone_x/ui/app.py :: if entry.get("role") in ("assistant", "agent") or entry.get("sender") == "agent"
    Becomes: if entry.get("role") in ("assistant", "agent")
    """
    session_id = "sess_1023_turn_counter"
    connector = _ScriptedChatConnector({2}, ["reply 0", "reply 1", "reply 2"])
    client = _conversation_with_a_failed_turn(tmp_path, session_id, connector)

    truncated = client.post(
        "/api/session/history/truncate",
        json={"agent_id": "agent-general", "session_id": session_id, "index": 4},
    )
    assert truncated.status_code == 200, truncated.text

    rows = [m["role"] for m in _transcript(client, session_id)]
    assert rows == ["user", "assistant", "user", "failure"]

    # One answered turn and one failed turn: two turns taken, and the record says so.
    record = AgentSessionManager(storage_dir=tmp_path / "sessions").load_session_record(session_id)
    assert record is not None
    assert record["turns"] == 2
    mgr = cast(Any, client.app).state.session_manager
    state = mgr.core_store.load(session_id)
    assert state is not None
    assert state.turn_counter == 2


# --------------------------------------------------------------------------------------
# A tool-using turn is the mirror of a failed one: one transcript row over several Core
# messages. The mapping must account for all of them (#1023, #1024 review B2).
# --------------------------------------------------------------------------------------


class _ToolCallingConnector(MockLLMConnector):
    """Returns `tool_calls` until they are spent, with whatever text is scripted.

    The tool is deliberately not registered: the call still puts an `ASSISTANT` message
    carrying the calls and a `TOOL` result into the Core, which is the shape under test,
    and it keeps the fixture free of the app's persona/tool wiring.
    """

    def __init__(
        self,
        responses: list[str],
        tool_calls: list[ToolCallRequest],
        fail_on: set[int] | None = None,
    ) -> None:
        super().__init__(responses=responses, tool_calls=list(tool_calls))
        self.requests: list[LLMRequest] = []
        self._fail_on = fail_on or set()

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) in self._fail_on:
            raise LLMProviderError("provider unavailable (scripted)")
        return await super().generate(request)


def _tool_turn(
    tmp_path: Path,
    session_id: str,
    responses: list[str],
    tool_calls: list[ToolCallRequest],
    fail_on: set[int] | None = None,
) -> TestClient:
    """One turn that calls a tool, driven through `/api/turn`."""
    connector = _ToolCallingConnector(responses, tool_calls, fail_on)
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path / session_id,
            storage_dir=tmp_path / "sessions",
            llm=connector,
        )
    )
    posted = client.post(
        "/api/turn",
        json={"agent_id": "agent-general", "session_id": session_id, "message": "look it up"},
    )
    assert posted.status_code == 200, posted.text
    return client


def _call(call_id: str, query: str) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name="lookup_tool", arguments={"query": query})


def test_a_tool_turn_that_also_printed_text_keeps_its_whole_turn_in_the_core(
    tmp_path: Path,
) -> None:
    """A step may print text *and* call a tool, and the row still stands for both (#1023).

    `/api/turn` writes two transcript rows per turn, so every Core message of a tool-using
    turn except the reply is folded into the agent row. The rule named the wrong property:
    it folded an `ASSISTANT` message whose `content is None`, but `agent/base.py` appends
    `ChatMessage(ASSISTANT, content=resp_content or None, tool_calls=tool_calls)` under
    `if tool_calls or resp_content:` -- so a step that printed a preamble *and* called a
    tool has content, was not folded, and stalled the walk exactly as the strict-lockstep
    candidate does.

    Measured before the fix, truncating at `index == len(transcript)` -- the user deletes
    nothing -- the Core kept **1** message where 4 is right, losing the preamble, the tool
    result, and the turn's **own final reply**. What makes these messages foldable is that
    they carry tool calls, not that they lack content.

    A preamble beside a tool call is ordinary model behaviour, which is why this is the
    same severity as the row-wording defect it replaced.

    Killed by: src/uclone_x/ui/app.py :: msg.role == MessageRole.ASSISTANT and (msg.content is None or bool(msg.tool_calls))
    Becomes: msg.role == MessageRole.ASSISTANT and msg.content is None
    """
    cases: list[tuple[str, list[str], list[ToolCallRequest], list[str], list[str]]] = [
        # The defect: a preamble beside the call, so the message has content *and* calls.
        (
            "preamble",
            ["Let me look that up.", "The answer is 42."],
            [_call("c1", "q")],
            ["user", "assistant", "tool", "assistant"],
            ["Let me look that up.", "The answer is 42."],
        ),
        # Pins that already held and must go on holding.
        (
            "silent",
            ["", "The answer is 42."],
            [_call("c1", "q")],
            ["user", "assistant", "tool", "assistant"],
            ["The answer is 42."],
        ),
        (
            "two_calls",
            ["", "Both done."],
            [_call("c1", "a"), _call("c2", "b")],
            ["user", "assistant", "tool", "tool", "assistant"],
            ["Both done."],
        ),
    ]

    for name, responses, tool_calls, expected_roles, must_survive in cases:
        session_id = f"sess_1023_tool_{name}"
        client = _tool_turn(tmp_path, session_id, responses, tool_calls)

        rows = _transcript(client, session_id)
        assert [m["role"] for m in rows] == ["user", "assistant"], name
        before = _core_dialogue(client, session_id)
        # One row stands for the whole turn: the call, its result, and the reply.
        assert [role for role, _ in before] == expected_roles, (name, before)

        truncated = client.post(
            "/api/session/history/truncate",
            json={"agent_id": "agent-general", "session_id": session_id, "index": len(rows)},
        )
        assert truncated.status_code == 200, truncated.text

        # A cut that deletes nothing may not delete anything.
        after = _core_dialogue(client, session_id)
        assert after == before, name
        kept = [content for _, content in after]
        for text in must_survive:
            assert text in kept, (name, text, kept)


def test_a_turn_that_failed_after_calling_a_tool_keeps_what_the_core_recorded(
    tmp_path: Path,
) -> None:
    """The Core's record of a failed tool turn belongs to the row that reports it (#1023).

    A turn that fails after its tool step leaves the `ASSISTANT` message carrying the call
    and the `TOOL` result in the Core, and the page shows one failure row for the turn.
    That row matches no Core message, so folding it is not enough on its own: the walk
    stepped over those two messages and then, finding nothing to match, did not count
    them. Truncating at `index == len(transcript)` therefore cut the Core to **1** message
    where 3 is right.

    The rows stepped over are the Core's own record of the turn this row reports, so they
    are kept when the row is kept -- which is the same rule the restored failure rows
    follow, and it is what makes a failed tool turn survive a cut that deletes nothing.

    Killed by: src/uclone_x/ui/app.py :: index = candidate + 1 if matched else candidate
    Becomes: index = candidate + 1 if matched else index
    """
    session_id = "sess_1023_tool_failed"
    client = _tool_turn(
        tmp_path, session_id, ["", "never reached"], [_call("c1", "q")], fail_on={2}
    )

    rows = _transcript(client, session_id)
    assert [m["role"] for m in rows] == ["user", "failure"]
    before = _core_dialogue(client, session_id)
    # The prompt, the message carrying the call, and the tool's own result.
    assert [role for role, _ in before] == ["user", "assistant", "tool"]

    truncated = client.post(
        "/api/session/history/truncate",
        json={"agent_id": "agent-general", "session_id": session_id, "index": len(rows)},
    )
    assert truncated.status_code == 200, truncated.text
    assert _core_dialogue(client, session_id) == before


def test_truncating_after_a_retried_prompt_does_not_keep_the_reply_it_deleted(
    tmp_path: Path,
) -> None:
    """A repeated prompt must not be matched against a later copy of itself (#1023).

    `BaseAgent` does not append a prompt identical to the unanswered one already ending
    its history (#1022), so a retried send is a transcript row the Core has no message
    for. Aligning by searching ahead for the next match of any kind resolves that row
    against the *later* Core copy of the same prompt and swallows the reply between them:
    truncating to index 3 then keeps `reply 0`, the answer the user had just deleted.

    The walk therefore steps over only the messages a turn's row folds in, never over a
    message that could still match a row of its own.

    Killed by: src/uclone_x/ui/app.py :: and _is_folded_into_its_turns_row(presentable[candidate])
    Becomes: and True
    """
    session_id = "sess_1023_retry_cut"
    connector = _ScriptedChatConnector({1}, ["reply 0", "reply 1"])
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path / session_id,
            storage_dir=tmp_path / "sessions",
            llm=connector,
        )
    )
    for _ in range(3):
        posted = client.post(
            "/api/turn",
            json={"agent_id": "agent-general", "session_id": session_id, "message": "Same"},
        )
        assert posted.status_code == 200, posted.text

    assert [(m["role"], m["content"]) for m in _transcript(client, session_id)] == [
        ("user", "Same"),
        ("failure", "Error: provider unavailable (scripted)"),
        ("user", "Same"),
        ("assistant", "reply 0"),
        ("user", "Same"),
        ("assistant", "reply 1"),
    ]
    assert _core_dialogue(client, session_id) == [
        ("user", "Same"),
        ("assistant", "reply 0"),
        ("user", "Same"),
        ("assistant", "reply 1"),
    ]

    # Keep the first prompt, its failure row, and the retry: the Core holds one message
    # for all three, because the retry's prompt was never appended a second time.
    truncated = client.post(
        "/api/session/history/truncate",
        json={"agent_id": "agent-general", "session_id": session_id, "index": 3},
    )
    assert truncated.status_code == 200, truncated.text
    assert _core_dialogue(client, session_id) == [("user", "Same")]


# --------------------------------------------------------------------------------------
# #1026: the mapping as a property, over generated transcripts, at every index.
#
# A retry's repeated prompt is a transcript row the Core deliberately never held --
# `_repeats_unanswered_prompt` (#1022) does not re-append it -- which is the same class of
# skew as a failure row. These shapes mix that with failures, legacy `Error:` rows, tool
# turns and cancellations, and check the mapping against a reference derived from the Core.
# --------------------------------------------------------------------------------------


@dataclass
class _Step:
    """One `generate()` call within a turn: what the model returns, or how it fails."""

    content: str = ""
    tool_calls: tuple[ToolCallRequest, ...] = ()
    raises: str | None = None  # "provider" or "cancel"


@dataclass
class _Turn:
    """One `/api/turn` send, and the steps the model takes answering it."""

    prompt: str
    steps: tuple[_Step, ...]


class _StepScriptedConnector(MockLLMConnector):
    """Answers each `generate()` from a script of steps, in order.

    `_ToolCallingConnector` above returns one fixed set of calls until they are spent,
    which cannot express a turn whose *second* step calls a tool, nor a turn cancelled
    part-way through. This probe needs both, so the script is per-step here.
    """

    def __init__(self, steps: list[_Step]) -> None:
        super().__init__(responses=[])
        self._script = list(steps)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        i = len(self.requests) - 1
        step = self._script[i] if i < len(self._script) else _Step(content="(script spent)")
        if step.raises == "provider":
            raise LLMProviderError("provider unavailable (scripted)")
        if step.raises == "cancel":
            # What Stop delivers into a turn: cancelling the task makes
            # `execute_turn` re-raise `CancelledError` instead of recording a result.
            raise asyncio.CancelledError()
        self._responses = [step.content]
        self._tool_calls = list(step.tool_calls)
        return await super().generate(request)


def _probe_core(client: TestClient, session_id: str) -> list[ChatMessage]:
    """The Core as `truncate_session_history` reads it: the live agent, else the store.

    The live agent is the writer, so it is never behind the record and may be ahead of it --
    and it is the sequence a truncation would actually cut. (Before #1031 a cancelled turn
    also never reached `persist_session`, which is why this helper had to exist; it does
    reach it now, and the store agrees, which
    `test_a_stopped_turns_core_reaches_disk_before_anything_else_happens` pins.)
    """
    mgr = cast(Any, client.app).state.session_manager
    agent = mgr.get_agent("agent-general", session_id)
    state = agent.get_session(session_id) if agent is not None else mgr.core_store.load(session_id)
    return list(state.messages) if state is not None else []


def _probe_rows(client: TestClient, session_id: str) -> list[dict[str, Any]]:
    """The served transcript, tolerating a session that does not exist yet."""
    res = client.get(f"/api/session/history?agent_id=agent-general&session_id={session_id}")
    return cast(list[dict[str, Any]], res.json()["messages"]) if res.status_code == 200 else []


@dataclass
class _Shape:
    """A driven conversation, with the mapping and its reference at every index."""

    name: str
    client: TestClient
    session_id: str
    rows: list[dict[str, Any]]
    presentable: list[ChatMessage]
    mapped: list[int]
    reference: list[int]
    connector: _StepScriptedConnector


def _drive_shape(
    tmp_path: Path,
    name: str,
    turns: list[_Turn],
    prior_rows: list[dict[str, Any]] | None = None,
) -> _Shape:
    """Drive one conversation, and compute the mapping and its reference at every index.

    **The reference is turn-aware and observed, never re-derived from the walk under
    test.** The Core is snapshotted around every send, so each turn's Core delta is
    measured rather than assumed, and a row accounts for Core messages like this:

    * the prompt row accounts for the turn's first Core message when that message is the
      prompt. A retry's repeated prompt is *not* re-appended (#1022), so that turn's delta
      does not begin with the prompt and the row accounts for nothing -- which is the
      whole of what #1026 asks about;
    * the agent row accounts for every remaining message of that turn's delta, which is
      how a tool turn's several Core messages fold into one row;
    Every turn writes exactly two rows, a cancelled one included since #1031 -- so a turn
    that writes none is not modelled here, it is a defect, and the walk below says so rather
    than quietly scoring the mapping against a reference that has absorbed the loss.

    reference(k) = 1 + the highest Core position accounted for by the first k rows, or 0.
    Truncation cuts a *prefix*, so keeping a message means keeping everything before it.
    """
    session_id = f"sess_1026_{name}"
    storage = tmp_path / "sessions"
    steps: list[_Step] = []
    for turn in turns:
        steps.extend(turn.steps)

    if prior_rows is not None:
        AgentSessionManager(storage_dir=storage).save_session_record(
            session_id=session_id,
            agent_id="agent-general",
            messages=prior_rows,
            turns=sum(1 for row in prior_rows if row.get("sender") == "agent"),
        )

    connector = _StepScriptedConnector(steps)
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path / name,
            storage_dir=storage,
            llm=connector,
        )
    )

    measured: list[tuple[str, int, int, int]] = []
    for turn in turns:
        before = len([m for m in _probe_core(client, session_id) if _is_presentable(m)])
        rows_before = len(_probe_rows(client, session_id))
        try:
            client.post(
                "/api/turn",
                json={
                    "agent_id": "agent-general",
                    "session_id": session_id,
                    "message": turn.prompt,
                },
            )
        except BaseException as exc:  # a cancelled turn leaves the endpoint uncaught
            # The test portal re-raises `concurrent.futures.CancelledError`, which on 3.12
            # is not `asyncio.CancelledError`; the name is what both spellings share.
            if type(exc).__name__ != "CancelledError":
                raise
        after = len([m for m in _probe_core(client, session_id) if _is_presentable(m)])
        new_rows = len(_probe_rows(client, session_id)) - rows_before
        measured.append((turn.prompt, before, after, new_rows))

    rows = _probe_rows(client, session_id)
    core = _probe_core(client, session_id)
    presentable = [m for m in core if _is_presentable(m)]

    account: list[int | None] = []
    consumed = 0
    if prior_rows is not None:
        # Rows saved by an older head, aligned in order against the Core that the first
        # live turn rebuilt from them. Valid here: none of those rows is a tool turn.
        for row in prior_rows:
            key = (str(row.get("role") or ""), row.get("content"))
            if (
                consumed < len(presentable)
                and (
                    presentable[consumed].role.value,
                    presentable[consumed].content,
                )
                == key
            ):
                account.append(consumed)
                consumed += 1
            else:
                account.append(None)
        # That first live turn is what materialised the Core, so its measured `before` was
        # 0 and its delta swallowed the rebuilt prefix -- which belongs to the rows above.
        if measured:
            prompt, _, after, n_rows = measured[0]
            measured[0] = (prompt, consumed, after, n_rows)

    for prompt, before, after, n_new_rows in measured:
        delta = presentable[before:after]
        prompt_pos: int | None = None
        rest_start = before
        if delta and delta[0].role is MessageRole.USER and delta[0].content == prompt:
            prompt_pos = before
            rest_start = before + 1
        agent_pos: int | None = after - 1 if after - 1 >= rest_start else None
        if n_new_rows != 2:
            raise AssertionError(
                f"{name}: the turn {prompt!r} wrote {n_new_rows} transcript rows, not 2. "
                "A turn with no row of its own leaves Core messages nothing can place (#1031)."
            )
        account.extend([prompt_pos, agent_pos])

    assert len(account) == len(rows), f"{name}: accounted {len(account)} of {len(rows)} rows"

    reference = [0]
    best = -1
    for position in account:
        if position is not None:
            best = max(best, position)
        reference.append(best + 1)

    mapped = [_core_index_for_transcript(rows[:k], core) for k in range(len(rows) + 1)]
    return _Shape(name, client, session_id, rows, presentable, mapped, reference, connector)


def _probe_call(cid: str) -> ToolCallRequest:
    return ToolCallRequest(id=cid, name="lookup_tool", arguments={"q": cid})


_PROBE_FAILS = _Step(raises="provider")
_PROBE_CANCELS = _Step(raises="cancel")

_PROBE_LEGACY_ROWS: list[dict[str, Any]] = [
    {"id": "u1", "sender": "user", "role": "user", "content": "Q one"},
    {
        "id": "a1",
        "sender": "agent",
        "role": "assistant",
        "content": "Error: provider unavailable",
        "provenance": {"degraded": True, "path": "FAILOVER", "served_by": "agent.core"},
    },
    {"id": "u2", "sender": "user", "role": "user", "content": "Q two"},
    {
        "id": "a2",
        "sender": "agent",
        "role": "assistant",
        "content": "A two",
        "provenance": {"degraded": False, "path": "PRIMARY", "served_by": "mock:m"},
    },
]


def _enumerated_shapes() -> list[tuple[str, list[_Turn], list[dict[str, Any]] | None]]:
    """The shapes named in #1026, each a case someone reasoned about rather than a sample."""
    ok = [_Step("reply 0")]
    return [
        ("plain", [_Turn(f"msg {i}", (_Step(f"reply {i}"),)) for i in range(3)], None),
        (
            "fail_middle",
            [
                _Turn("msg 0", (_Step("reply 0"),)),
                _Turn("msg 1", (_PROBE_FAILS,)),
                _Turn("msg 2", (_Step("reply 2"),)),
            ],
            None,
        ),
        ("fail_first", [_Turn("msg 0", (_PROBE_FAILS,)), _Turn("msg 1", (_Step("r"),))], None),
        ("fail_last", [_Turn("msg 0", (_Step("r"),)), _Turn("msg 1", (_PROBE_FAILS,))], None),
        ("fail_all", [_Turn(f"msg {i}", (_PROBE_FAILS,)) for i in range(3)], None),
        # A retry repeats the unanswered prompt: the page records both sends, the Core one.
        (
            "retry_once",
            [
                _Turn("Same", (_PROBE_FAILS,)),
                _Turn("Same", (_Step("reply 0"),)),
                _Turn("Same", (_Step("reply 1"),)),
            ],
            None,
        ),
        (
            "retry_twice",
            [
                _Turn("Same", (_PROBE_FAILS,)),
                _Turn("Same", (_PROBE_FAILS,)),
                _Turn("Same", (_Step("reply 0"),)),
                _Turn("Same", (_Step("reply 1"),)),
            ],
            None,
        ),
        (
            "retry_then_other",
            [
                _Turn("Same", (_PROBE_FAILS,)),
                _Turn("Same", (_Step("reply 0"),)),
                _Turn("other", (_Step("reply 1"),)),
            ],
            None,
        ),
        # The same words *after* they were answered are a new message, and are appended.
        ("dup_answered_prompt", [_Turn("Same", (_Step(f"r{i}"),)) for i in range(3)], None),
        (
            "tool_silent",
            [_Turn("look", (_Step("", (_probe_call("c1"),)), _Step("The answer is 42.")))],
            None,
        ),
        (
            "tool_preamble",
            [_Turn("look", (_Step("Let me look.", (_probe_call("c1"),)), _Step("The answer.")))],
            None,
        ),
        (
            "tool_two_calls_one_step",
            [
                _Turn(
                    "look",
                    (
                        _Step("", (_probe_call("c1"), _probe_call("c2"))),
                        _Step("Both done."),
                    ),
                )
            ],
            None,
        ),
        (
            "tool_two_steps",
            [
                _Turn(
                    "look",
                    (
                        _Step("", (_probe_call("c1"),)),
                        _Step("more", (_probe_call("c2"),)),
                        _Step("Both done."),
                    ),
                )
            ],
            None,
        ),
        (
            "tool_then_fail",
            [_Turn("look", (_Step("", (_probe_call("c1"),)), _PROBE_FAILS))],
            None,
        ),
        (
            "tool_then_fail_then_retry",
            [
                _Turn("look", (_Step("", (_probe_call("c1"),)), _PROBE_FAILS)),
                _Turn("look", (_Step("recovered"),)),
                _Turn("next", (_Step("reply"),)),
            ],
            None,
        ),
        ("legacy_then_live", [_Turn("Q three", (_Step("A three"),))], _PROBE_LEGACY_ROWS),
        (
            "legacy_then_tool",
            [_Turn("Q three", (_Step("", (_probe_call("c1"),)), _Step("A three")))],
            _PROBE_LEGACY_ROWS,
        ),
        (
            "legacy_then_fail_then_retry",
            [_Turn("Q three", (_PROBE_FAILS,)), _Turn("Q three", (_Step("A three"),))],
            _PROBE_LEGACY_ROWS,
        ),
        ("single_ok", [_Turn("only", tuple(ok))], None),
        # #1031: Stop between a tool call and the next step. Before the cancelled turn had
        # a row of its own its prompt sat in the Core matchable by nothing and foldable by
        # nothing, and the walk stopped there for the rest of the conversation.
        (
            "cancel_after_tool",
            [
                _Turn("msg 0", (_Step("reply 0"),)),
                _Turn("cancel me", (_Step("", (_probe_call("c1"),)), _PROBE_CANCELS)),
                _Turn("msg 2", (_Step("reply 2"),)),
            ],
            None,
        ),
        (
            "cancel_first_step",
            [
                _Turn("msg 0", (_Step("reply 0"),)),
                _Turn("cancel me", (_PROBE_CANCELS,)),
                _Turn("msg 2", (_Step("reply 2"),)),
            ],
            None,
        ),
        # The last turn cancelled after a tool step. The probe's reference scores this
        # "exact" even when the mapping is wrong -- there is no later row to strand -- so it
        # is the no-op-cut invariant, not the index comparison, that has teeth here: measured
        # before the fix, that cut took the Core 5 -> 2 (#1031, PR #1033 review).
        (
            "cancel_only_last",
            [
                _Turn("msg 0", (_Step("reply 0"),)),
                _Turn("cancel me", (_Step("", (_probe_call("c1"),)), _PROBE_CANCELS)),
            ],
            None,
        ),
        (
            "cancel_last_first_step",
            [_Turn("msg 0", (_Step("reply 0"),)), _Turn("cancel me", (_PROBE_CANCELS,))],
            None,
        ),
        ("cancel_only_turn", [_Turn("cancel me", (_PROBE_CANCELS,))], None),
        # A cancelled prompt goes unanswered, so the resend repeats it and the Core does not
        # re-append it (#1022) -- a repeated prompt and a turn-outcome row in one shape.
        (
            "cancel_then_retry",
            [
                _Turn("Same", (_Step("", (_probe_call("c1"),)), _PROBE_CANCELS)),
                _Turn("Same", (_Step("reply 0"),)),
                _Turn("next", (_Step("reply 1"),)),
            ],
            None,
        ),
        (
            "legacy_then_cancel",
            [
                _Turn("Q three", (_Step("", (_probe_call("c1"),)), _PROBE_CANCELS)),
                _Turn("Q four", (_Step("A four"),)),
            ],
            _PROBE_LEGACY_ROWS,
        ),
    ]


def _generated_shapes(count: int, seed: int) -> list[tuple[str, list[_Turn], None]]:
    """Seeded mixtures of the same turn kinds, for the orderings nobody thought to name."""
    rng = random.Random(seed)
    kinds = [
        "ok",
        "fail",
        "tool",
        "tool_text",
        "tool_two",
        "tool_fail",
        "cancel",
        "tool_cancel",
        "retry",
        "dup",
    ]
    shapes: list[tuple[str, list[_Turn], None]] = []
    for s in range(count):
        turns: list[_Turn] = []
        last_prompt: str | None = None
        last_failed = False
        for t in range(rng.randint(2, 6)):
            kind = rng.choice(kinds)
            if last_prompt is not None and (kind == "dup" or (kind == "retry" and last_failed)):
                prompt = last_prompt
            else:
                prompt = f"p{s}_{t}"
            cid = f"c{s}_{t}"
            if kind == "fail":
                steps: tuple[_Step, ...] = (_PROBE_FAILS,)
            elif kind == "tool":
                steps = (_Step("", (_probe_call(cid),)), _Step(f"r{s}_{t}"))
            elif kind == "tool_text":
                steps = (_Step(f"pre{s}_{t}", (_probe_call(cid),)), _Step(f"r{s}_{t}"))
            elif kind == "tool_two":
                steps = (
                    _Step("", (_probe_call(cid + "a"), _probe_call(cid + "b"))),
                    _Step(f"r{s}_{t}"),
                )
            elif kind == "tool_fail":
                steps = (_Step("", (_probe_call(cid),)), _PROBE_FAILS)
            elif kind == "cancel":
                steps = (_PROBE_CANCELS,)
            elif kind == "tool_cancel":
                steps = (_Step("", (_probe_call(cid),)), _PROBE_CANCELS)
            else:
                steps = (_Step(f"r{s}_{t}"),)
            turns.append(_Turn(prompt, steps))
            last_prompt = prompt
            # A cancelled turn leaves its prompt unanswered too, so a resend repeats it.
            last_failed = any(step.raises is not None for step in steps)
        shapes.append((f"gen{s}", turns, None))
    return shapes


def test_the_transcript_to_core_mapping_is_exact_at_every_index(tmp_path: Path) -> None:
    """Every prefix of every shape maps to the Core messages that prefix covers (#1026).

    #1026 asked whether a retry's repeated prompt -- a transcript row the Core
    deliberately never held, because `_repeats_unanswered_prompt` (#1022) does not
    re-append it -- is still mapped one late. It is not: measured against a turn-aware
    reference computed from the Core itself, `retry_once` reads `[0, 1, 1, 1, 2, 3, 4]`
    and `retry_twice` `[0, 1, 1, 1, 1, 1, 2, 3, 4]`, agreeing at every index, and so does
    every other shape here. No marker distinguishing a repeated-prompt row was needed;
    inventing one would have pinned a defect that the Core-derived count had closed.

    The shapes mix what #1026 names: failures first, last, middle, consecutive and
    throughout; legacy `Error:` rows from a transcript saved before failure records
    existed; repeated prompts, retried and answered; tool turns with a contentless step,
    a step that printed text beside its call, two calls in one step and two across two
    steps; a turn that failed after its tool step; turns cancelled at the first step, after
    a tool call, as the last turn and as the only turn (#1031); and transcripts rebuilt from
    the Core.
    Twenty-six enumerated shapes, plus twenty-four seeded mixtures for the orderings
    nobody thought to name -- 50 in all, `len(_enumerated_shapes())` being the check on the
    first figure. (#1033's own body miscounted this as "44 shapes, 42 exact"; it was 43
    shapes -- 19 enumerated plus 24 seeded -- all exact, beside one cancellation shape
    driven by a test of its own. #1031 corrects the figure where it repeats.)

    The invariant asserted last is the one that caught all three of #1024's blockers: a
    truncation at `index == len(transcript)` deletes nothing, so it must delete nothing
    from the Core either.

    Killed by: src/uclone_x/ui/app.py :: and _is_folded_into_its_turns_row(presentable[candidate])
    Becomes: and True
    """
    shapes = _enumerated_shapes() + _generated_shapes(24, seed=1026)
    assert len(shapes) == 50, f"the probe drives {len(shapes)} shapes"
    wrong: list[str] = []
    for name, turns, prior in shapes:
        shape = _drive_shape(tmp_path, name, turns, prior)
        if shape.mapped != shape.reference:
            wrong.append(
                f"{name}: right={shape.reference} measured={shape.mapped}\n"
                f"    rows={[(r.get('role'), r.get('content')) for r in shape.rows]}\n"
                f"    core={[(m.role.value, m.content) for m in shape.presentable]}"
            )

        # A cut that deletes nothing may not delete anything -- through the endpoint,
        # because the mapping being right is only half of the claim. It runs for **every**
        # shape, a mapping mismatch above included: for a cancelled turn that is the last
        # one there is no later row to strand, so the mapping agrees with the reference
        # while still being wrong and this is the only evidence there. Skipping it on a
        # mismatch would disable it exactly where it carries the weight, so a failure is
        # collected rather than raised.
        before = [(m.role.value, m.content) for m in shape.presentable]
        truncated = shape.client.post(
            "/api/session/history/truncate",
            json={
                "agent_id": "agent-general",
                "session_id": shape.session_id,
                "index": len(shape.rows),
            },
        )
        if truncated.status_code != 200:
            wrong.append(f"{name}: the no-op truncation answered {truncated.status_code}")
            continue
        after = [
            (m.role.value, m.content)
            for m in _probe_core(shape.client, shape.session_id)
            if _is_presentable(m)
        ]
        if after != before:
            wrong.append(f"{name}: a no-op truncation cut the Core\n    {before}\n -> {after}")

    assert not wrong, "the mapping disagrees with the Core-derived reference:\n" + "\n".join(wrong)


# --------------------------------------------------------------------------------------
# #1031: a turn Stop cancels leaves a record of its own.
#
# `asyncio.CancelledError` is a `BaseException`, so it walked past the handler that saves a
# failed turn's rows: the Core kept the prompt, the `ASSISTANT` message carrying the tool
# calls and the `TOOL` result, and the transcript recorded nothing. The page showed no sign
# a tool had run, and the orphaned prompt -- matchable by no row, foldable by no rule --
# stalled `_core_index_for_transcript` for the rest of the conversation.
# --------------------------------------------------------------------------------------


_CANCELLED_SHAPE = [
    _Turn("msg 0", (_Step("reply 0"),)),
    _Turn("cancel me", (_Step("", (_probe_call("c1"),)), _PROBE_CANCELS)),
    _Turn("msg 2", (_Step("reply 2"),)),
]
"""The measured shape: Stop between a tool call and the step that would have used it."""


def _core_on_disk(client: TestClient, session_id: str) -> list[tuple[str, str | None]]:
    """The Core as the store holds it, which is what a restart would read back."""
    mgr = cast(Any, client.app).state.session_manager
    state = mgr.core_store.load(session_id)
    assert state is not None, f"no Core record for {session_id!r}"
    return [(m.role.value, m.content) for m in state.messages if _is_presentable(m)]


def test_a_cancelled_turn_is_recorded_so_a_truncation_that_deletes_nothing_deletes_nothing(
    tmp_path: Path,
) -> None:
    """Stop writes a row of its own, and the Core survives the cut that follows (#1031).

    Measured before this change, on exactly this shape -- one send, a send cancelled between
    a tool call and the next step, another send -- and then a truncation at
    `index == len(transcript)`, which deletes nothing:

    | | before | after the no-op cut |
    |---|---|---|
    | Core | 7 messages | **2** |
    | `msg 2` / `reply 2`, still on the page | present | **gone, in memory and on disk** |
    | next turn's request | carried them | **did not** |

    The cancelled turn wrote no transcript row, so its prompt sat in the Core as a message
    no row could match and no rule could fold, and the walk stopped on it: every later row
    mapped to nothing, and the cut took the conversation back to the last row the walk had
    reached. With a row of its own the cancelled turn is shaped like every other turn --
    prompt row, then one row for the turn -- and the walk that already handles a failed turn
    handles this one with no rule about cancellation in it at all.

    The Core is not touched by the record: what the turn had already put there stays, in
    memory and on disk, and the next request carries it.

    Killed by: src/uclone_x/ui/app.py :: turn_messages: list[ChatMessage] = []
    Becomes: return
    Killed by: src/uclone_x/ui/app.py :: "outcome": ChatTurnOutcome.INTERRUPTED,
    Becomes: "outcome": ChatTurnOutcome.COMPLETED,
    """
    shape = _drive_shape(tmp_path, "cancel_recorded", _CANCELLED_SHAPE)

    # Two rows for the cancelled turn, in the order the page showed them.
    assert [(m["role"], m["content"]) for m in shape.rows] == [
        ("user", "msg 0"),
        ("assistant", "reply 0"),
        ("user", "cancel me"),
        (TRANSCRIPT_CANCELLED_ROLE, CANCELLED_TURN_TEXT),
        ("user", "msg 2"),
        ("assistant", "reply 2"),
    ]
    # Its structured outcome, which the head names "Interrupted" (#1007 item 4).
    assert [m.get("outcome") for m in shape.rows if m["role"] != "user"] == [
        "completed",
        "interrupted",
        "completed",
    ]
    # The Core keeps what the cancelled turn left it, which is what made the row necessary.
    assert [(m.role.value, m.content) for m in shape.presentable] == [
        ("user", "msg 0"),
        ("assistant", "reply 0"),
        ("user", "cancel me"),
        ("assistant", None),
        ("tool", "Tool 'lookup_tool' not found"),
        ("user", "msg 2"),
        ("assistant", "reply 2"),
    ]
    # Exact at every index, the cancelled turn's folded messages included.
    assert shape.reference == [0, 1, 2, 3, 5, 6, 7]
    assert shape.mapped == shape.reference

    before = [(m.role.value, m.content) for m in shape.presentable]
    truncated = shape.client.post(
        "/api/session/history/truncate",
        json={
            "agent_id": "agent-general",
            "session_id": shape.session_id,
            "index": len(shape.rows),
        },
    )
    assert truncated.status_code == 200, truncated.text
    after = [
        (m.role.value, m.content)
        for m in _probe_core(shape.client, shape.session_id)
        if _is_presentable(m)
    ]
    assert after == before, "a truncation that deletes nothing cut the Core"
    assert _core_on_disk(shape.client, shape.session_id) == before

    # And the turn after the cut is still answered against the whole conversation.
    shape.client.post(
        "/api/turn",
        json={"agent_id": "agent-general", "session_id": shape.session_id, "message": "msg 3"},
    )
    carried = _said(shape.connector.requests[-1])
    assert ("user", "msg 2") in carried
    assert ("assistant", "reply 2") in carried
    # The record of the cancellation is the page's; it is not conversation.
    assert all(CANCELLED_TURN_TEXT not in (content or "") for _, content in carried)


def test_a_no_op_cut_keeps_the_core_when_the_cancelled_turn_is_the_last_one(
    tmp_path: Path,
) -> None:
    """The shape the mapping cannot be scored on, measured through the endpoint (#1031).

    With the cancelled turn last there is no later row for the stall to strand, so the
    transcript-to-Core mapping agrees with a Core-derived reference at every index **even
    while it is wrong**: the probe scores `cancel_only_last` exact either way. What it loses
    is the Core. Measured before this change, one send followed by a send cancelled after
    its tool call, then a truncation at `index == len(transcript)`:

    | | before | after the no-op cut |
    |---|---|---|
    | Core | 5 messages | **2** |

    So passing the probe is not sufficient evidence for a cancellation, and this asserts the
    invariant that is: a cut that deletes nothing deletes nothing, here and on disk.

    Killed by: src/uclone_x/ui/app.py :: turn_messages: list[ChatMessage] = []
    Becomes: return
    """
    shape = _drive_shape(
        tmp_path,
        "cancel_last_only",
        [
            _Turn("msg 0", (_Step("reply 0"),)),
            _Turn("cancel me", (_Step("", (_probe_call("c1"),)), _PROBE_CANCELS)),
        ],
    )
    before = [
        ("user", "msg 0"),
        ("assistant", "reply 0"),
        ("user", "cancel me"),
        ("assistant", None),
        ("tool", "Tool 'lookup_tool' not found"),
    ]
    assert [(m.role.value, m.content) for m in shape.presentable] == before

    truncated = shape.client.post(
        "/api/session/history/truncate",
        json={
            "agent_id": "agent-general",
            "session_id": shape.session_id,
            "index": len(shape.rows),
        },
    )
    assert truncated.status_code == 200, truncated.text
    assert [
        (m.role.value, m.content)
        for m in _probe_core(shape.client, shape.session_id)
        if _is_presentable(m)
    ] == before
    assert _core_on_disk(shape.client, shape.session_id) == before


def test_a_cancellation_at_the_first_step_is_recorded_like_any_other(tmp_path: Path) -> None:
    """A stop before any tool ran stalls the walk identically, and is fixed identically.

    #1031 was reported on a cancellation *between a tool call and the next step*, but the
    Core message the walk stalls on is the **prompt**, which every cancelled turn leaves
    behind. Measured on a first-step cancellation with no tool messages at all: right
    `[0, 1, 2, 4, 5]`, measured `[0, 1, 2, 2, 2]` (PR #1033 review). The remedy is the same
    row, and it is not a rule about tools.

    Killed by: src/uclone_x/ui/app.py :: turn_messages: list[ChatMessage] = []
    Becomes: return
    """
    shape = _drive_shape(
        tmp_path,
        "cancel_first_step_only",
        [
            _Turn("msg 0", (_Step("reply 0"),)),
            _Turn("cancel me", (_PROBE_CANCELS,)),
            _Turn("msg 2", (_Step("reply 2"),)),
        ],
    )
    assert [(m["role"], m["content"]) for m in shape.rows] == [
        ("user", "msg 0"),
        ("assistant", "reply 0"),
        ("user", "cancel me"),
        (TRANSCRIPT_CANCELLED_ROLE, CANCELLED_TURN_TEXT),
        ("user", "msg 2"),
        ("assistant", "reply 2"),
    ]
    # Nothing was called, so the row says nothing was called.
    assert shape.rows[3]["tool_calls"] == []
    assert shape.mapped == shape.reference == [0, 1, 2, 3, 3, 4, 5]

    before = [(m.role.value, m.content) for m in shape.presentable]
    truncated = shape.client.post(
        "/api/session/history/truncate",
        json={
            "agent_id": "agent-general",
            "session_id": shape.session_id,
            "index": len(shape.rows),
        },
    )
    assert truncated.status_code == 200, truncated.text
    assert [
        (m.role.value, m.content)
        for m in _probe_core(shape.client, shape.session_id)
        if _is_presentable(m)
    ] == before


def test_a_cancelled_turn_reports_only_the_calls_its_own_turn_made(tmp_path: Path) -> None:
    """The row names this turn's calls, never the previous turn's (#1031).

    A cancelled turn has no `TurnResult`, so the row's `tool_calls` are read off the Core
    messages the turn appended -- which requires knowing where this turn's messages *begin*.
    Read that boundary wrongly (0, say) and an earlier completed turn's call is presented as
    a call the stopped turn made: measured, a first turn calling `lookup_tool` and answering,
    then a turn Stopped at its first step having called nothing, saved a row reporting
    `tool_calls: ['lookup_tool']`. The code's own comment names that hazard; this is what
    makes the claim fail when it stops being true.

    Killed by: src/uclone_x/ui/app.py :: core_len_before = len(agent.get_session(session_id).messages)
    Becomes: core_len_before = 0
    """
    shape = _drive_shape(
        tmp_path,
        "cancel_after_a_tool_turn",
        [
            _Turn("look", (_Step("", (_probe_call("c1"),)), _Step("The answer."))),
            _Turn("cancel me", (_PROBE_CANCELS,)),
        ],
    )

    rows = shape.rows
    assert [(m["role"], m["content"]) for m in rows] == [
        ("user", "look"),
        ("assistant", "The answer."),
        ("user", "cancel me"),
        (TRANSCRIPT_CANCELLED_ROLE, CANCELLED_TURN_TEXT),
    ]
    # The completed turn's call belongs to the completed turn's row, and to no other.
    assert [(c["id"], c["name"]) for c in rows[1]["tool_calls"]] == [("c1", "lookup_tool")]
    assert rows[3]["tool_calls"] == []


def test_a_stopped_turns_core_reaches_disk_before_anything_else_happens(tmp_path: Path) -> None:
    """The Core the stopped turn left is durable at once, not once something else writes.

    Nothing else on this path writes the Core: `execute_turn` leaves its messages on the
    live agent, and the endpoint's own `persist_session` runs only on the paths that return.
    Measured by removing the writer's call: a session whose **first** turn is stopped has no
    Core record on disk at all afterwards, and a session with an earlier turn keeps only that
    turn -- three messages against the six the agent holds -- so the saved row would describe
    a turn a restart could not find. Every later check of the Core on disk in this file runs
    after a truncation, which persists on its own, which is why this one reads the store
    immediately and drives no truncation at all.

    Killed by: src/uclone_x/ui/app.py :: agent.persist_session(session_id=session_id)
    Becomes: pass
    """
    shape = _drive_shape(
        tmp_path,
        "cancel_durable",
        [
            _Turn("msg 0", (_Step("reply 0"),)),
            _Turn("cancel me", (_Step("", (_probe_call("c1"),)), _PROBE_CANCELS)),
        ],
    )

    left_behind = [
        ("user", "msg 0"),
        ("assistant", "reply 0"),
        ("user", "cancel me"),
        ("assistant", None),
        ("tool", "Tool 'lookup_tool' not found"),
    ]
    assert [(m.role.value, m.content) for m in shape.presentable] == left_behind
    assert _core_on_disk(shape.client, shape.session_id) == left_behind
    # And the row says the write happened, rather than claiming it without having tried.
    assert shape.rows[3]["durability"]["persisted"] is True


def test_truncating_before_at_and_after_a_cancellation_row_cuts_where_the_page_says(
    tmp_path: Path,
) -> None:
    """Each cut keeps exactly the turns whose rows the user kept (#1031, FR-13.7).

    The transcript reads `msg 0, reply 0, cancel me, <cancelled>, msg 2, reply 2` over a
    Core of seven messages, four of which have no row of their own: the cancelled turn's
    `ASSISTANT`-with-calls and `TOOL` result are folded into its row, exactly as a tool
    turn's are folded into its reply's row.

    | rows kept | last row kept | Core kept |
    |---|---|---|
    | 2 | `reply 0` | 2 |
    | 3 | the cancelled turn's prompt | 3 |
    | 4 | the cancellation row | 5 -- its folded tool messages go with it |
    | 5 | `msg 2` | 6 |

    A cut that keeps the cancellation row keeps the messages it reports, for the same
    reason a failed tool turn's do: the row is the page's record *of* those messages
    (#1023 Rev 17). A cut that drops it drops them.

    Killed by: src/uclone_x/ui/app.py :: return [prompt_row, cancelled_row]
    Becomes: return [cancelled_row]
    """
    kept_core = {
        2: [("user", "msg 0"), ("assistant", "reply 0")],
        3: [("user", "msg 0"), ("assistant", "reply 0"), ("user", "cancel me")],
        4: [
            ("user", "msg 0"),
            ("assistant", "reply 0"),
            ("user", "cancel me"),
            ("assistant", None),
            ("tool", "Tool 'lookup_tool' not found"),
        ],
        5: [
            ("user", "msg 0"),
            ("assistant", "reply 0"),
            ("user", "cancel me"),
            ("assistant", None),
            ("tool", "Tool 'lookup_tool' not found"),
            ("user", "msg 2"),
        ],
    }
    for index, expected in kept_core.items():
        shape = _drive_shape(tmp_path, f"cancel_cut_{index}", _CANCELLED_SHAPE)
        truncated = shape.client.post(
            "/api/session/history/truncate",
            json={
                "agent_id": "agent-general",
                "session_id": shape.session_id,
                "index": index,
            },
        )
        assert truncated.status_code == 200, truncated.text
        core = [
            (m.role.value, m.content)
            for m in _probe_core(shape.client, shape.session_id)
            if _is_presentable(m)
        ]
        assert core == expected, f"cut at {index} kept {core}"
        assert _core_on_disk(shape.client, shape.session_id) == expected
        # The page keeps the rows the user kept, the cancellation record included.
        assert [(m["role"], m["content"]) for m in _transcript(shape.client, shape.session_id)] == [
            (m["role"], m["content"]) for m in shape.rows[:index]
        ]


def test_the_history_endpoint_serves_the_cancelled_turn_with_the_call_it_made(
    tmp_path: Path,
) -> None:
    """The row says a turn was stopped and which tool it had called (#1031, P0, P6).

    "No failure row, no partial reply, no sign a tool ran" is the harm #1031 names, so the
    row carries the calls the Core records for the turn. It carries **no** `tool_executions`:
    the Core states which calls were made, not how each one ended, and a status nothing
    measured would be a substituted default (P6). The turn's own `TOOL` messages stay in the
    Core and off the page, which is where #1024 left a failed turn's -- this does not decide
    that question.

    Killed by: src/uclone_x/ui/app.py :: call for msg in turn_messages for call in _serialized_tool_calls(msg.tool_calls or ())
    Becomes: call for msg in () for call in _serialized_tool_calls(msg.tool_calls or ())
    """
    shape = _drive_shape(tmp_path, "cancel_served", _CANCELLED_SHAPE)
    rows = _transcript(shape.client, shape.session_id)
    row = rows[3]

    assert row["role"] == TRANSCRIPT_CANCELLED_ROLE
    assert row["sender"] == "agent"
    assert row["content"] == CANCELLED_TURN_TEXT
    assert [(c["id"], c["name"]) for c in row["tool_calls"]] == [("c1", "lookup_tool")]
    assert "tool_executions" not in row
    # A cancelled turn was still a turn taken: `execute_turn` counts it before it calls the
    # model, and the saved figure is the Core's own counter (#1023).
    assert row["turn_count"] == 2
    assert row["provenance"]["degraded"] is True
    assert row["provenance"]["path"] == "CANCELLED"
    assert row["durability"]["persisted"] is True
    # The prompt is saved beside it, so a page that reloads recognises its own copy (#1000).
    assert (rows[2]["role"], rows[2]["content"]) == ("user", "cancel me")


def test_a_cancelled_turn_is_shown_on_reload_and_never_rebuilt_into_model_context(
    tmp_path: Path,
) -> None:
    """A reopened conversation shows the stop and does not re-send it (#1031, #969).

    `reconstruct_history` drops a cancellation row through the same predicate that drops a
    failure row: neither is anything the agent said. Without that the rebuild would also
    meet a role that is no `MessageRole` at all.

    The prompt of the cancelled turn *is* conversation and comes back, which is what the
    Core held all along.

    Killed by: src/uclone_x/ui/app.py :: return role == TRANSCRIPT_CANCELLED_ROLE or _is_failure_entry(role, entry)
    Becomes: return _is_failure_entry(role, entry)
    """
    storage = tmp_path / "sessions"
    shape = _drive_shape(tmp_path, "cancel_reload", _CANCELLED_SHAPE)
    session_id = shape.session_id

    # A second app over the same storage: the live agent is gone, the record is not.
    reopened = _StepScriptedConnector([_Step("reply 3")])
    client = TestClient(
        create_ui_app(static_dir=tmp_path / "reopen", storage_dir=storage, llm=reopened)
    )
    rows = _transcript(client, session_id)
    assert [(m["role"], m["content"]) for m in rows][2:4] == [
        ("user", "cancel me"),
        (TRANSCRIPT_CANCELLED_ROLE, CANCELLED_TURN_TEXT),
    ]

    posted = client.post(
        "/api/turn",
        json={"agent_id": "agent-general", "session_id": session_id, "message": "msg 3"},
    )
    assert posted.status_code == 200, posted.text
    said = _said(reopened.requests[-1])
    assert ("user", "cancel me") in said
    assert all(CANCELLED_TURN_TEXT not in (content or "") for _, content in said)


def test_a_transcript_saved_before_cancellation_rows_still_maps_and_truncates(
    tmp_path: Path,
) -> None:
    """A conversation written by an older head is read, mapped and cut exactly as before.

    The shapes a transcript could hold before #1031 are unchanged by it: no row carries the
    new role, so `_records_a_turn_not_spoken` answers precisely what `_is_failure_entry`
    answered, and the legacy `Error: ` spelling of a failed turn keeps its own behaviour.

    Four legacy rows -- a failed turn in the old spelling, then an answered one -- brought
    up as a live agent by one send, then cut back to the failure. The Core keeps the prompt
    that went unanswered and nothing else, which is the #1023 answer, unmoved.

    Killed by: src/uclone_x/ui/app.py :: return role == TRANSCRIPT_CANCELLED_ROLE or _is_failure_entry(role, entry)
    Becomes: return role == TRANSCRIPT_CANCELLED_ROLE
    """
    shape = _drive_shape(
        tmp_path,
        "legacy_no_cancellation",
        [_Turn("Q three", (_Step("A three"),))],
        _PROBE_LEGACY_ROWS,
    )
    assert shape.mapped == shape.reference == [0, 1, 1, 2, 3, 4, 5]

    truncated = shape.client.post(
        "/api/session/history/truncate",
        json={"agent_id": "agent-general", "session_id": shape.session_id, "index": 2},
    )
    assert truncated.status_code == 200, truncated.text
    assert [(m["role"], m["content"]) for m in _transcript(shape.client, shape.session_id)] == [
        ("user", "Q one"),
        ("assistant", "Error: provider unavailable"),
    ]
    assert [
        (m.role.value, m.content)
        for m in _probe_core(shape.client, shape.session_id)
        if _is_presentable(m)
    ] == [("user", "Q one")]
