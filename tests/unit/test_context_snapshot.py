"""What was sent is recorded, and can be rebuilt (#1421).

A request is five layers: tools, identity, slow context, conversation, turn context. The
session record now holds one `ContextSnapshot` per distinct set of the non-conversation
layers, with the large bodies stored once by content hash, and each `REQUEST_CONTEXT`
event names its snapshot and carries only the conversation the request added. These tests
pin the five things the issue asks for, and what review found around them:

*   one builder composes the identity prompt, so the anchor a session stores is the
    identity its requests send -- for a room seat and for a 1:1 clone, each built the way
    the product builds it;
*   every request of a session, tools included, is rebuilt from the persisted record and
    the log, across turns and a restart;
*   a turn's first step no longer re-records the conversation, so the log is linear in the
    requests of a session, not just in the steps of one turn;
*   a record written before snapshots existed still loads, and its session still runs;
*   resetting or deleting a session removes the snapshot bodies with its log.
*   a rebuild whose record was redacted on the way to disk says it is not verified, for
    the tool schemas (which the messages digest does not cover) and the turn context;
*   every save that writes a record writes the bodies it names first.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentConfig, AgentLLMConfig, PersonaDefinition
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.request_record import (
    RequestRecordError,
    compose_system_message,
    rebuild_requests,
    serialize_tools,
)
from uclone_x.agent.session import (
    CORE_RECORD_SUBDIR,
    SessionState,
    SessionStore,
    content_digest,
)
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.log.reader import read_session_log
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.room.models import Participant, ParticipantKind
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="agent", model="dummy"),
    served_by=ServiceRef(provider="agent", model="dummy"),
    attempts=(),
)
_USAGE = TokenUsage(provider="dummy", model="dummy", input_tokens=0, output_tokens=0)


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        content=text,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )


def _call(call_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(ToolCallRequest(id=call_id, name="count_tool", arguments={"x": 1}),),
        usage=_USAGE,
        provenance=_PROV,
    )


class RecordingLLM(BaseLLMConnector):
    """Replays a script, then answers `"done"`, and records every request it is sent."""

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        super().__init__()
        self.responses = list(responses or [])
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "dummy"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0) if self.responses else _answer("done")

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        yield StreamChunk(delta_content="")


class CountParams(BaseModel):
    x: int = Field(default=0)


class CountTool(BaseTool[CountParams]):
    name = "count_tool"
    description = "Adds one"

    def run(self, params: CountParams, context: ToolContext) -> dict[str, Any]:
        return {"result": params.x + 1}


#: A credential shape the store redacts on the way to disk. Not a real key.
_FAKE_KEY = "sk-1234567890123456789012345678901234567890"


class KeyedTool(BaseTool[CountParams]):
    """A tool whose schema carries a credential-shaped string, as a pasted MCP one might."""

    name = "keyed_tool"
    description = f"Calls the service with {_FAKE_KEY}"

    def run(self, params: CountParams, context: ToolContext) -> dict[str, Any]:
        return {"result": params.x}


def _agent(store: SessionStore, llm: RecordingLLM, session_id: str = "sess_snap") -> BaseAgent:
    from uclone_x.agent.models import AgentContext, AgentState

    registry = ToolRegistry()
    registry.register(CountTool())
    config = AgentConfig(
        agent_id="agent_snap",
        name="Agent",
        system_prompt="You are a careful counter.",
        llm_config=AgentLLMConfig(model_name="dummy", temperature=0.3, max_tokens=512),
        max_steps=8,
    )
    context = AgentContext(
        session_id=session_id, agent_id="agent_snap", current_state=AgentState.IDLE
    )
    # A remembered fact gives every request a turn-context block at its tail.
    memory = CrossSessionMemory()
    memory.record_fact(
        subject="project",
        predicate="uses",
        object_value="postgres",
        provenance=_PROV,
        source_session_id="earlier",
    )
    return BaseAgent(
        config=config, llm=llm, tools=registry, store=store, context=context, memory=memory
    )


def _log(store: SessionStore, session_id: str = "sess_snap") -> Path:
    path = store.event_log_path(session_id)
    assert path is not None
    return path


def _request_contexts(store: SessionStore) -> list[dict[str, Any]]:
    return [dict(e) for e in read_session_log(_log(store)) if e.get("type") == "REQUEST_CONTEXT"]


def _dump(messages: tuple[ChatMessage, ...] | list[ChatMessage]) -> list[dict[str, Any]]:
    return json.loads(json.dumps([m.model_dump() for m in messages], default=str))


def _conversation(request: LLMRequest) -> list[ChatMessage]:
    return [m for m in request.messages if m.role is not MessageRole.SYSTEM]


async def _three_turns_with_a_restart(tmp_path: Path) -> tuple[SessionStore, list[LLMRequest]]:
    """Two turns with tool steps, a restart, then a third: every request that was sent."""
    store = SessionStore(tmp_path)
    first = RecordingLLM([_call("c1"), _call("c2"), _answer("two"), _call("c3"), _answer("one")])
    agent = _agent(store, first)
    await agent.start()
    assert (await agent.execute_turn("count twice")).is_completed
    assert (await agent.execute_turn("count once")).is_completed
    agent.persist_session()

    second = RecordingLLM([_call("c4"), _answer("again")])
    restarted = _agent(store, second)
    assert restarted.hydrate_session() is not None
    await restarted.start()
    assert (await restarted.execute_turn("once more")).is_completed
    restarted.persist_session()
    return store, [*first.requests, *second.requests]


# --------------------------------------------------------------------------------------
# One builder: the stored anchor is the identity that is sent
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_room_seats_stored_anchor_is_the_identity_its_request_sends(
    tmp_path: Path,
) -> None:
    """The seat's anchor carries its framing and persona, and the request sends that anchor.

    The resolver used to build the agent, which seeded the anchor from a bare prompt, and
    then define the persona and patch the config: the anchor lacked the seat framing the
    request carried. The persona now reaches the agent at construction, so the anchor and
    the request come from one builder.

    Killed by: src/uclone_x/room/resolver.py :: persona_definitions=(persona_def,),
    Becomes: persona_definitions=(),
    """
    registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
    registry.register_persona(
        PersonaDefinition(
            name="novelist",
            role="Creative Fiction Writer",
            description="Writes prose.",
            system_prompt="You are a novelist. Focus on sensory detail.",
        )
    )
    llm = RecordingLLM()
    store = SessionStore(tmp_path / "sessions")
    host = HostDependencies(
        bus=EventBus(), llm=llm, tools=ToolRegistry(), tracer=TelemetryTracer(), store=store
    )
    seat = Participant(
        id="author",
        kind=ParticipantKind.AGENT,
        display_name="Author",
        persona="novelist",
        session_id="sess_room__r1__author",
    )
    agent = await RoomAgentResolver(host, persona_registry=registry).resolve(seat)
    assert isinstance(agent, BaseAgent)
    await agent.execute_turn("Open the chapter.")
    agent.persist_session(seat.session_id)

    record = store.load(seat.session_id)
    assert record is not None
    anchor = record.messages[0].content or ""
    assert "one participant in a shared multi-agent conversation" in anchor
    assert "[Persona Instructions: Creative Fiction Writer]" in anchor

    sent = llm.requests[0].messages[0]
    assert sent.role is MessageRole.SYSTEM
    snapshot = record.context_snapshots[-1]
    slow = store.load_context_body(seat.session_id, snapshot.slow_context_digest)
    assert slow is not None
    assert snapshot.identity_digest == content_digest(anchor)
    assert sent.content == compose_system_message(anchor, slow)
    if not slow:
        assert sent.content == anchor


@pytest.mark.usefixtures("builtin_personas_absent")
def test_a_one_to_one_clones_stored_anchor_is_the_identity_its_request_sends(
    tmp_path: Path,
) -> None:
    """The same for an agent the UI builds for a direct conversation, through its turn route.

    Killed by: src/uclone_x/agent/base.py :: if not seat_framing:
    Becomes: if False:
    """
    from uclone_x.ui.app import create_ui_app

    llm = RecordingLLM()
    app = create_ui_app(static_dir=tmp_path, llm=llm, storage_dir=tmp_path)
    client = TestClient(app)
    session_id = "sess_clone"
    turn = {"message": "hello", "agent_id": "agent-clone", "session_id": session_id}
    assert client.post("/api/turn", json=turn).json()["status"] == "success"

    store = SessionStore(tmp_path / CORE_RECORD_SUBDIR)
    record = store.load(session_id)
    assert record is not None
    anchor = record.messages[0].content or ""
    assert anchor.startswith("You are agent-clone,"), anchor[:80]

    snapshot = record.context_snapshots[-1]
    slow = store.load_context_body(session_id, snapshot.slow_context_digest)
    assert slow is not None
    assert snapshot.identity_digest == content_digest(anchor)
    sent = llm.requests[0].messages[0]
    assert sent.role is MessageRole.SYSTEM
    assert sent.content == compose_system_message(anchor, slow)


# --------------------------------------------------------------------------------------
# Every request is rebuilt from the record
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_request_of_a_session_is_rebuilt_from_its_record(tmp_path: Path) -> None:
    """Messages, tools, model and settings of each request, across turns and a restart.

    Killed by: src/uclone_x/agent/request_record.py :: conversation = conversation[:kept] + list(event["appended_messages"])
    Becomes: conversation = list(event["appended_messages"])
    Killed by: src/uclone_x/agent/base.py :: turn_context_digest=content_digest(layers.turn_context),
    Becomes: turn_context_digest=content_digest(""),
    """
    store, sent = await _three_turns_with_a_restart(tmp_path)
    state = store.load("sess_snap")
    assert state is not None

    rebuilt = rebuild_requests(store, state, read_session_log(_log(store)))

    assert len(rebuilt) == len(sent) == 7
    for number, (request, original) in enumerate(zip(rebuilt, sent, strict=True)):
        assert request.verified, f"request {number} did not hash to what was sent"
        assert _dump(request.request.messages) == _dump(original.messages), number
        assert request.request.tools == original.tools, number
        assert request.request.tools, "no tool schemas were sent, so this proves nothing"
        assert request.request.model == original.model
        assert request.request.temperature == original.temperature
        assert request.request.max_tokens == original.max_tokens


@pytest.mark.asyncio
async def test_a_turn_adds_one_snapshot_and_a_changed_layer_adds_another(
    tmp_path: Path,
) -> None:
    """A snapshot is recorded when what it holds changes, not once per step.

    Killed by: src/uclone_x/agent/base.py :: if not session.context_snapshots or session.context_snapshots[-1] != candidate:
    Becomes: if True:
    """
    store = SessionStore(tmp_path)
    agent = _agent(store, RecordingLLM([_call("c1"), _call("c2"), _answer("two")]))
    await agent.start()
    await agent.execute_turn("count twice")
    await agent.execute_turn("and again")
    agent.persist_session()
    state = store.load("sess_snap")
    assert state is not None

    # Four requests in the first turn and one in the second, all with the same layers but
    # the turn index, so one snapshot per turn.
    assert len(_request_contexts(store)) == 4
    assert [s.turn_index for s in state.context_snapshots] == [1, 2]
    assert len({s.identity_digest for s in state.context_snapshots}) == 1
    # The turn context is a body like the other layers, so the same text is stored once
    # rather than copied into every snapshot of the record.
    assert len({s.turn_context_digest for s in state.context_snapshots}) == 1
    turn_context = store.load_context_body(
        "sess_snap", state.context_snapshots[0].turn_context_digest
    )
    assert turn_context and "[Turn Context]" in turn_context
    assert "[Turn Context]" not in (tmp_path / "sess_snap.json").read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# The log is linear in the requests of a session
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_turns_first_step_records_only_what_the_turn_added(tmp_path: Path) -> None:
    """Across a session, each conversation message is recorded once.

    A turn's first step used to record its whole request, so every turn re-recorded the
    conversation so far and a long session's log grew with the square of its turns.

    Killed by: src/uclone_x/agent/base.py :: kept, appended = _request_context_delta(session.last_conversation, conversation)
    Becomes: kept, appended = _request_context_delta([] if step == 1 else session.last_conversation, conversation)
    """
    store = SessionStore(tmp_path)
    llm = RecordingLLM()
    agent = _agent(store, llm)
    await agent.start()
    turns = 6
    for index in range(turns):
        assert (await agent.execute_turn(f"message {index}")).is_completed
    agent.persist_session()

    contexts = _request_contexts(store)
    assert len(contexts) == turns
    recorded = sum(len(e["appended_messages"]) for e in contexts)
    assert recorded == len(_conversation(llm.requests[-1])), recorded
    # Each turn after the first adds its question and the previous turn's answer.
    assert all(len(e["appended_messages"]) <= 2 for e in contexts)
    assert all("system" not in json.dumps(e["appended_messages"]) for e in contexts[1:])


# --------------------------------------------------------------------------------------
# Old records, and clearing
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_record_written_before_snapshots_loads_and_runs(tmp_path: Path) -> None:
    """A record in the shape written before #1421, with no `context_snapshots` key.

    Killed by: src/uclone_x/agent/session.py :: default_factory=tuple,
    Becomes:
    """
    store = SessionStore(tmp_path)
    store.save(
        SessionState(
            session_id="sess_snap",
            agent_id="agent_snap",
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content="You are a careful counter."),
                ChatMessage(role=MessageRole.USER, content="hello"),
                ChatMessage(role=MessageRole.ASSISTANT, content="hi"),
            ),
            turn_counter=1,
        )
    )
    path = tmp_path / "sess_snap.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("context_snapshots")
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = store.load("sess_snap")
    assert loaded is not None
    assert loaded.context_snapshots == ()
    assert len(loaded.messages) == 3

    agent = _agent(store, RecordingLLM())
    assert agent.hydrate_session() is not None
    await agent.start()
    assert (await agent.execute_turn("again")).is_completed
    agent.persist_session()
    after = store.load("sess_snap")
    assert after is not None and len(after.context_snapshots) == 1


@pytest.mark.asyncio
async def test_reset_and_delete_remove_the_snapshot_bodies(tmp_path: Path) -> None:
    """The bodies are history, as the log is, and go with it.

    Killed by: src/uclone_x/agent/session.py :: shutil.rmtree(body_dir)
    Becomes: pass
    """
    store = SessionStore(tmp_path)
    agent = _agent(store, RecordingLLM([_call("c1"), _answer("one")]))
    await agent.start()
    await agent.execute_turn("count")
    agent.persist_session()
    bodies = store.context_body_dir("sess_snap")
    assert any(bodies.iterdir()), "no bodies were written, so this proves nothing"

    agent.reset_session()
    assert not bodies.exists(), "a reset left the cleared conversation's bodies on disk"

    await agent.execute_turn("count")
    agent.persist_session()
    assert any(bodies.iterdir())
    assert store.delete("sess_snap")
    assert not bodies.exists(), "a delete left the session's bodies on disk"


@pytest.mark.asyncio
async def test_a_missing_body_or_request_is_refused_in_plain_words(tmp_path: Path) -> None:
    """A rebuild that would have to guess stops, and says so without internals.

    Killed by: src/uclone_x/agent/request_record.py :: if base is not None and base != previous_seq:
    Becomes: if False:
    """
    store, _ = await _three_turns_with_a_restart(tmp_path)
    state = store.load("sess_snap")
    assert state is not None
    log = _log(store)
    original = log.read_text(encoding="utf-8")

    # A request missing from the middle of a chain.
    lines = original.splitlines(keepends=True)
    second = [i for i, line in enumerate(lines) if '"REQUEST_CONTEXT"' in line][1]
    log.write_text("".join(lines[:second] + lines[second + 1 :]), encoding="utf-8")
    with pytest.raises(RequestRecordError) as gap:
        rebuild_requests(store, state, read_session_log(log))
    assert "/" not in str(gap.value) and "Error" not in str(gap.value)
    log.write_text(original, encoding="utf-8")

    # A body missing from the store.
    (store.context_body_dir("sess_snap") / state.context_snapshots[0].tools_digest).unlink()
    with pytest.raises(RequestRecordError) as missing:
        rebuild_requests(store, state, read_session_log(log))
    assert "/" not in str(missing.value) and "Error" not in str(missing.value)
    assert state.context_snapshots[0].tools_digest in missing.value.detail


# --------------------------------------------------------------------------------------
# A rebuild says when redaction changed what it rebuilt
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_tool_schema_redacted_on_disk_is_not_called_verified(tmp_path: Path) -> None:
    """The tools are sent as they are and stored redacted; the rebuild says they differ.

    The messages digest does not cover the tool schemas, so a body that no longer hashes
    to its name is the only sign that the rebuilt tools are not the ones sent.

    Killed by: src/uclone_x/agent/request_record.py :: body_intact[digest] = content_digest(text) == digest
    Becomes: body_intact[digest] = True
    """
    store = SessionStore(tmp_path)
    llm = RecordingLLM()
    agent = _agent(store, llm)
    await agent.start()
    assert (await agent.execute_turn("count")).is_completed
    tools = agent.tools
    assert isinstance(tools, ToolRegistry)
    tools.register(KeyedTool())
    assert (await agent.execute_turn("count again")).is_completed
    agent.persist_session()
    state = store.load("sess_snap")
    assert state is not None
    rebuilt = rebuild_requests(store, state, read_session_log(_log(store)))
    sent = llm.requests
    assert len(rebuilt) == len(sent) == 2

    assert _FAKE_KEY in serialize_tools(sent[1].tools), "the schema went out redacted"
    on_disk = "".join(
        p.read_text(encoding="utf-8") for p in store.context_body_dir("sess_snap").iterdir()
    )
    assert _FAKE_KEY not in on_disk

    assert rebuilt[0].verified, "a request with nothing redacted was not verified"
    assert rebuilt[0].request.tools == sent[0].tools
    assert not rebuilt[1].verified, "a redacted tool schema was called what was sent"
    assert rebuilt[1].request.tools != sent[1].tools
    assert _dump(rebuilt[1].request.messages) == _dump(sent[1].messages)


@pytest.mark.asyncio
async def test_a_turn_context_redacted_on_disk_is_not_called_verified(tmp_path: Path) -> None:
    """Text inside a message body that goes out unredacted is caught the same way.

    A message the user types is redacted before it is ever sent (`redact_message`), so
    the conversation cannot differ from its record. The turn context can: a remembered
    fact reaches the tail of the last user message as it was recorded, and the store
    redacts the turn-context body on disk.

    Both halves of `verified` catch it -- the body no longer hashes to its name, and the
    messages no longer hash to the recorded digest -- so neither alone is what pins this;
    the redaction on disk is.

    Killed by: src/uclone_x/agent/session.py :: staging.write_text(redact_credentials(body), encoding="utf-8")
    Becomes: staging.write_text(body, encoding="utf-8")
    """
    store = SessionStore(tmp_path)
    llm = RecordingLLM()
    agent = _agent(store, llm)
    await agent.start()
    assert (await agent.execute_turn("count")).is_completed
    memory = agent.memory
    assert isinstance(memory, CrossSessionMemory)
    memory.record_fact(
        subject="service",
        predicate="key",
        object_value=_FAKE_KEY,
        provenance=_PROV,
        source_session_id="earlier",
    )
    assert (await agent.execute_turn("count again")).is_completed
    agent.persist_session()
    state = store.load("sess_snap")
    assert state is not None
    rebuilt = rebuild_requests(store, state, read_session_log(_log(store)))
    sent = llm.requests

    assert len(rebuilt) == len(sent) == 2
    assert _FAKE_KEY in (sent[1].messages[-1].content or ""), "the turn context went out redacted"
    assert _FAKE_KEY not in json.dumps(_dump(rebuilt[1].request.messages))
    assert rebuilt[0].verified, "a request with nothing redacted was not verified"
    assert not rebuilt[1].verified, "a redacted turn context was called what was sent"
    assert rebuilt[1].request.tools == sent[1].tools


# --------------------------------------------------------------------------------------
# Every save writes the bodies its record names
# --------------------------------------------------------------------------------------


def _missing_bodies(store: SessionStore, session_id: str = "sess_snap") -> list[str]:
    state = store.load(session_id)
    assert state is not None and state.context_snapshots, "no snapshots, so this proves nothing"
    named = {
        digest
        for s in state.context_snapshots
        for digest in (
            s.tools_digest,
            s.identity_digest,
            s.slow_context_digest,
            s.turn_context_digest,
        )
    }
    return sorted(d for d in named if store.load_context_body(session_id, d) is None)


@pytest.mark.asyncio
async def test_a_history_loaded_before_a_save_keeps_the_bodies_to_write(tmp_path: Path) -> None:
    """A rewind replaces the working copy and then saves it; the record's bodies are there.

    Killed by: src/uclone_x/agent/base.py :: replaced.pending_bodies = live.pending_bodies
    Becomes: pass
    """
    store = SessionStore(tmp_path)
    agent = _agent(store, RecordingLLM())
    await agent.start()
    assert (await agent.execute_turn("count")).is_completed
    kept = list(agent.get_session().messages)
    agent.load_history(kept, turn_counter=1)
    agent.persist_session()

    assert _missing_bodies(store) == []


@pytest.mark.asyncio
async def test_a_compaction_writes_the_bodies_its_record_names(tmp_path: Path) -> None:
    """Compaction saves the record itself; that save writes the bodies too.

    Killed by: src/uclone_x/agent/base.py :: self._write_pending_bodies(sid)  # before the record that names them
    Becomes: pass
    """
    store = SessionStore(tmp_path)
    agent = _agent(store, RecordingLLM())
    await agent.start()
    for index in range(3):
        assert (await agent.execute_turn(f"message {index}")).is_completed
    await agent.compact_session()

    assert _missing_bodies(store) == []
