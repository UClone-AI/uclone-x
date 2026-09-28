# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
"""Unit tests for the FR-13 head surface: dispatch, health, sessions, personas, attribution."""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from uclone_x.core.provenance import (
    ExecutionPath,
)
from uclone_x.errors import LLMProviderError, SessionHistoryRehydrationError
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.models import (
    LLMRequest,
    MessageRole,
    ModelResponse,
)
from uclone_x.ui.app import (
    TRANSCRIPT_CANCELLED_ROLE,
    AgentSessionManager,
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
        model="qwen3:8b",
    )


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


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/turn"),
        ("GET", "/api/session/history"),
        ("POST", "/api/session/history/truncate"),
        ("DELETE", "/api/session/history"),
    ],
)
def test_the_retired_chat_routes_are_not_served(tmp_path: Path, method: str, path: str) -> None:
    """Every conversation is a room (D1 Rev 23); the single-agent chat routes are gone (#1731).

    The static directory holds an `index.html` so the page mount is live: a retired path
    must not be answered by it either.
    """
    (tmp_path / "index.html").write_text("<!doctype html><title>UClone</title>", encoding="utf-8")
    client = TestClient(create_ui_app(static_dir=tmp_path, storage_dir=tmp_path / "sessions"))

    # The control: the same app answers a live route and serves the page, so the refusal
    # below is about the path and not a client that reaches nothing.
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200

    res = client.request(method, path, json={"message": "hi", "agent_id": "scout"})

    assert res.status_code in {404, 405}, (res.status_code, res.text[:200])


def test_the_live_agents_listing_is_not_served(tmp_path: Path) -> None:
    """`GET /api/agents` is gone (owner ruling 2026-09-27, #1775).

    It overlaid live-instance state onto the persona rows, and after #1731 took the chat
    path away it answered an empty list for every install. The rail lists clones from
    `GET /api/personas`; a head still asking the old path must see a 404, not an answer
    from some other route or from the page mount.

    Killed by: src/uclone_x/ui/app.py :: @app.get("/api/personas")
    Becomes: @app.get("/api/agents")
    """
    (tmp_path / "index.html").write_text("<!doctype html><title>UClone</title>", encoding="utf-8")
    client = TestClient(create_ui_app(static_dir=tmp_path, storage_dir=tmp_path / "sessions"))

    # The control: the same app answers a live route and serves the page.
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200

    res = client.get("/api/agents")

    assert res.status_code == 404, (res.status_code, res.text[:200])


def _said(request: LLMRequest) -> list[tuple[str, str | None]]:
    """The conversation a request showed the model, without its system prompt."""
    return [(m.role.value, m.content) for m in request.messages if m.role is not MessageRole.SYSTEM]


@pytest.mark.parametrize(
    ("stop_reason", "refused"),
    [
        ("budget_exceeded", True),
        ("model_without_tools", True),
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

    expected = RoomTurnRefusal(stop_reason) if refused and stop_reason else None
    assert turn_refusal(stop_reason) == expected


# --------------------------------------------------------------------------------------
# Ported from the retired `POST /api/turn` (#1731). The chat route is gone; what it pinned
# about the agent it built, and about rebuilding an old transcript, is driven here through
# `AgentSessionManager.get_or_create_agent` and `BaseAgent.execute_turn` directly.
# --------------------------------------------------------------------------------------


async def _turns(
    mgr: AgentSessionManager,
    agent_id: str,
    session_id: str,
    messages: list[str],
    model_name: str | None = None,
) -> list[Any]:
    """Send `messages` in order to one agent; a turn that raised yields its exception."""
    agent = await mgr.get_or_create_agent(agent_id, session_id=session_id, model_name=model_name)
    results: list[Any] = []
    for message in messages:
        try:
            results.append(await agent.execute_turn(message))
        except Exception as exc:  # noqa: BLE001 -- the failed turn is the observation
            results.append(exc)
    return results


def _write_legacy_transcript(
    storage_dir: Path, session_id: str, messages: list[dict[str, Any]], turns: int
) -> None:
    """A UI transcript as the retired chat path saved it, with no Core record beside it."""
    path = AgentSessionManager(storage_dir=storage_dir).get_session_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "agent_id": "agent-general",
                "created_at": "2026-09-01T00:00:00Z",
                "updated_at": "2026-09-01T00:00:00Z",
                "turns": turns,
                "messages": messages,
            }
        ),
        encoding="utf-8",
    )


def test_fr13_attribution_is_independent_of_message_content(tmp_path: Path) -> None:
    """FR-13.4: a reply claiming 'I am OpenAI GPT-4' is still attributed to what served it."""
    mgr = AgentSessionManager(
        storage_dir=tmp_path / "sessions",
        llm=stubbed_ollama_with_model(
            reply="Hello! I am OpenAI GPT-4, a large language model trained by OpenAI.",
            model="qwen3:8b",
        ),
    )
    (result,) = asyncio.run(_turns(mgr, "agent-general", "sess_fr13_attr", ["Who are you?"]))
    assert "OpenAI GPT-4" in result.content
    prov = result.provenance
    assert prov is not None
    assert (prov.served_by.provider, prov.served_by.model) == ("ollama", "qwen3:8b")
    assert prov.degraded is False
    assert prov.path is ExecutionPath.PRIMARY


def test_hermes_model_adapts_system_prompt_for_steerability(tmp_path: Path) -> None:
    """Asking for a Hermes model adapts the system prompt for steerability (#871)."""
    from uclone_x.agent.prompts import HERMES_STEERABILITY_POLICY, IDENTITY_GROUNDING

    mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=MockLLMConnector())
    agent = asyncio.run(
        mgr.get_or_create_agent("pioneer", session_id="sess_hermes", model_name="hermes3:8b")
    )
    assert "You are Pioneer" in agent.effective_system_prompt
    assert IDENTITY_GROUNDING in agent.effective_system_prompt
    assert HERMES_STEERABILITY_POLICY in agent.effective_system_prompt


def test_an_agent_is_built_from_its_declarative_persona(tmp_path: Path) -> None:
    """`agent_id='writer'` instantiates the agent from PersonaRegistry (#897)."""
    mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=MockLLMConnector())
    agent = asyncio.run(mgr.get_or_create_agent("writer", session_id="sess_writer"))
    prompt = agent.effective_system_prompt
    assert "Writer" in prompt or "storyteller" in prompt.lower()
    assert "file_write" in agent.config.allowed_tools


def test_retrying_a_failed_turn_shows_the_model_the_prompt_once(tmp_path: Path) -> None:
    """A retry of a failed prompt reaches the model once (#969).

    `BaseAgent.execute_turn` puts the prompt in history before calling the model and a
    failed turn leaves it there, so without the check the retry's request read
    `[("user", "Same prompt"), ("user", "Same prompt")]`. Sending the same words after
    they were answered is a new message, and still reaches the model as one.

    Killed by: src/uclone_x/agent/turn_executor.py :: if not _repeats_unanswered_prompt(self._history, user_prompt):
    Becomes: if True:
    """
    connector = _ScriptedChatConnector({1}, ["Answered", "Answered again"])
    mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=connector)
    first, retried, again = asyncio.run(
        _turns(mgr, "agent-general", "sess_retry", ["Same prompt"] * 3)
    )
    assert isinstance(first, Exception) or first.error is not None
    assert retried.content == "Answered"
    assert _said(connector.requests[1]) == [("user", "Same prompt")]
    assert again.content == "Answered again"
    assert _said(connector.requests[2]) == [
        ("user", "Same prompt"),
        ("assistant", "Answered"),
        ("user", "Same prompt"),
    ]


def test_an_old_transcripts_failed_turn_is_kept_out_of_the_model(tmp_path: Path) -> None:
    """A transcript saved before failure records existed is rebuilt without its failures (#969).

    Its failed turns are `role: "assistant"` rows reading `Error: ...` with degraded
    provenance. A conversation with no Core record is rebuilt from that transcript, and the
    rebuild handed those rows to the model as replies the agent had given.

    Killed by: src/uclone_x/ui/app.py :: and content.startswith(LEGACY_FAILURE_PREFIX)
    Becomes: and False
    """
    storage_dir = tmp_path / "sessions"
    _write_legacy_transcript(
        storage_dir,
        "sess_old",
        [
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
                "provenance": {"degraded": False, "path": "PRIMARY", "served_by": "mock:mock"},
            },
        ],
        turns=2,
    )
    connector = _ScriptedChatConnector(set(), ["Answer three"])
    mgr = AgentSessionManager(storage_dir=storage_dir, llm=connector)
    (result,) = asyncio.run(_turns(mgr, "agent-general", "sess_old", ["Question three"]))
    assert result.content == "Answer three"
    assert _said(connector.requests[-1]) == [
        ("user", "Question one"),
        ("user", "Question two"),
        ("assistant", "Answer two"),
        ("user", "Question three"),
    ]


def test_an_old_transcripts_cancelled_turn_is_never_rebuilt_into_model_context(
    tmp_path: Path,
) -> None:
    """A reopened transcript's cancellation row is not re-sent as a reply (#1031, #969).

    `reconstruct_history` drops a cancellation row through the same predicate that drops a
    failure row: neither is anything the agent said. The prompt of the cancelled turn *is*
    conversation and comes back.

    Killed by: src/uclone_x/ui/app.py :: return role == TRANSCRIPT_CANCELLED_ROLE or _is_failure_entry(role, entry)
    Becomes: return _is_failure_entry(role, entry)
    """
    storage_dir = tmp_path / "sessions"
    stopped = "Stopped before an answer."
    _write_legacy_transcript(
        storage_dir,
        "sess_cancel",
        [
            {"id": "user-1", "sender": "user", "role": "user", "content": "msg 0"},
            {"id": "agent-1", "sender": "agent", "role": "assistant", "content": "reply 0"},
            {"id": "user-2", "sender": "user", "role": "user", "content": "cancel me"},
            {
                "id": "agent-2",
                "sender": "agent",
                "role": TRANSCRIPT_CANCELLED_ROLE,
                "content": stopped,
            },
        ],
        turns=2,
    )
    connector = _ScriptedChatConnector(set(), ["reply 3"])
    mgr = AgentSessionManager(storage_dir=storage_dir, llm=connector)
    try:
        (result,) = asyncio.run(_turns(mgr, "agent-general", "sess_cancel", ["msg 3"]))
    except SessionHistoryRehydrationError as exc:
        # Kept as a row, `cancelled` is no `MessageRole`: the rebuild refuses the history.
        raise AssertionError(f"the cancellation row reached the rebuild: {exc}") from exc
    assert not isinstance(result, Exception), result
    said = _said(connector.requests[-1])
    assert ("user", "cancel me") in said
    assert all(stopped not in (content or "") for _, content in said)
