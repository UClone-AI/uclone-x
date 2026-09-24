"""The durable turn events reach disk in every production store configuration (#1442).

`BaseAgent` queues an event for every step of a turn, and `persist_session` hands the queue
to `SessionStore.save`. Until #1442 the store wrote them only when a log writer had been
injected, and no production construction injected one, so every turn's tool calls, full
tool outputs and nudges were drained and dropped with nothing saying so.

What these pin:

* the stores the heads actually build -- the UI's (which room seats share) and a bare
  `SessionStore()`, which is what the ACP shell, the CLI and the A2A path build -- write a
  readable per-session event log under their own storage directory;
* a store that is handed events it cannot write refuses, and writes nothing;
* `REQUEST_CONTEXT` records each step's delta, so the log grows linearly in a turn's steps,
  and the full request of every step can still be rebuilt from it and checked;
* the log is redacted on write, offsets are contiguous across saves, and deleting a session
  deletes its log.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import (
    CORE_RECORD_SUBDIR,
    EVENT_LOG_SUBDIR,
    SESSION_STORAGE_DIR_ENV_VAR,
    SessionState,
    SessionStore,
)
from uclone_x.core.log_writer import RedactingLogWriter
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.engine.event_bus import EventBus
from uclone_x.errors import SessionEventLogNotConfiguredError
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.log.file_allocator import FileLogOffsetAllocator
from uclone_x.log.reader import read_session_log
from uclone_x.room.models import ParticipantKind
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import MentionSelector
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

# A shape `redact_credentials` recognises (see test_log_redaction_credentials.py); not a key.
_FAKE_KEY = "sk-1234567890123456789012345678901234567890"


def _events(log_path: Path) -> list[dict[str, Any]]:
    """Every event in a log, read through the shipped reader (which checks the header)."""
    return list(read_session_log(log_path))


def _types(events: list[dict[str, Any]]) -> list[str]:
    return [str(e["type"]) for e in events]


# --------------------------------------------------------------------------------------
# A scripted agent, for the turns whose shape the tests need to control.
# --------------------------------------------------------------------------------------

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="scripted", model="scripted"),
    served_by=ServiceRef(provider="scripted", model="scripted"),
    attempts=(),
)
_USAGE = TokenUsage(provider="scripted", model="scripted", input_tokens=0, output_tokens=0)


class _ScriptedLLM(BaseLLMConnector):
    """Calls `echo` `tool_steps` times, then answers. Keeps every request it received."""

    def __init__(self, tool_steps: int) -> None:
        super().__init__()
        self._tool_steps = tool_steps
        self.calls: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.calls.append(request)
        n = len(self.calls)
        if n <= self._tool_steps:
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                content=None,
                tool_calls=(ToolCallRequest(id=f"tc_{n}", name="echo", arguments={"x": n}),),
                usage=_USAGE,
                provenance=_PROV,
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content="done",
            tool_calls=(),
            usage=_USAGE,
            provenance=_PROV,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta_content="unused")


class _EchoParams(BaseModel):
    x: int = Field(default=0)


class _EchoTool(BaseTool[_EchoParams]):
    """Returns a ~1 KB body carrying a credential shape, so steps have weight to measure."""

    name = "echo"
    description = "Echo tool"

    def run(self, params: _EchoParams, context: ToolContext) -> dict[str, Any]:
        return {"x": params.x, "body": "y" * 1000, "leaked": _FAKE_KEY}


async def _run_turn(store: SessionStore, tool_steps: int) -> tuple[BaseAgent, _ScriptedLLM]:
    llm = _ScriptedLLM(tool_steps)
    registry = ToolRegistry()
    registry.register(_EchoTool())
    config = AgentConfig(
        agent_id="agent_1",
        name="Agent",
        llm_config=AgentLLMConfig(model_name="scripted"),
        max_steps=tool_steps + 5,
    )
    agent = BaseAgent(config=config, llm=llm, tools=registry, store=store)
    await agent.start()
    result = await agent.execute_turn("start")
    assert result.is_completed, result.error
    agent.persist_session()
    return agent, llm


def _only_log(store: SessionStore) -> Path:
    logs = sorted((store.storage_dir / EVENT_LOG_SUBDIR).glob("*.jsonl"))
    assert len(logs) == 1, logs
    return logs[0]


# --------------------------------------------------------------------------------------
# The construction sites
# --------------------------------------------------------------------------------------


class TestTheStoresTheHeadsBuildWriteEvents:
    @pytest.mark.usefixtures("builtin_personas_absent")
    def test_a_ui_turn_with_a_tool_call_leaves_tool_call_and_tool_result_on_disk(
        self, tmp_path: Path
    ) -> None:
        """The UI's own store, driven through its own turn route.

        Killed by: src/uclone_x/agent/session.py :: if log_writer is None and log_allocator is None:
        Becomes: if False:
        """
        from uclone_x.core.provenance import Provenance as _Prov
        from uclone_x.tools import LocalTool
        from uclone_x.tools.models import ToolResult

        class ReadFileTool(LocalTool):
            def __init__(self) -> None:
                super().__init__(name="read_file", description="Reads a file")

            async def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
                return ToolResult(
                    success=True, output="file body", provenance=_Prov.primary("read_file")
                )

        registry = ToolRegistry()
        registry.register(ReadFileTool())
        llm = MockLLMConnector(
            responses=["Reading.", "Read it."],
            tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"p": "a"})],
        )
        from uclone_x.ui.app import create_ui_app

        app = create_ui_app(static_dir=tmp_path, llm=llm, tools=registry, storage_dir=tmp_path)
        session_id = "sess_ui_events"
        res = TestClient(app).post(
            "/api/turn",
            json={"message": "read a", "agent_id": "agent-general", "session_id": session_id},
        )
        assert res.status_code == 200, res.text
        assert cast(dict[str, Any], res.json())["status"] == "success", res.text

        log_path = tmp_path / CORE_RECORD_SUBDIR / EVENT_LOG_SUBDIR / f"{session_id}.jsonl"
        assert log_path.is_file(), f"no event log at {log_path}"
        events = _events(log_path)
        types = _types(events)
        assert "TOOL_CALL" in types and "TOOL_RESULT" in types, types
        result = next(e for e in events if e["type"] == "TOOL_RESULT")
        assert result["tool_call_id"] == "call_1"
        assert result["output"] == "file body"

    @pytest.mark.asyncio
    async def test_a_default_store_writes_under_the_session_directory_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`SessionStore()` with no arguments: what the ACP shell, the CLI and A2A build.

        Killed by: src/uclone_x/agent/session.py :: if log_writer is None and log_allocator is None:
        Becomes: if False:
        """
        monkeypatch.setenv(SESSION_STORAGE_DIR_ENV_VAR, str(tmp_path))
        from uclone_x.shells.acp.server import ACPServer

        store = cast(SessionStore, ACPServer()._store)  # pyright: ignore[reportPrivateUsage]
        assert store.storage_dir == (tmp_path / CORE_RECORD_SUBDIR).resolve()

        agent, _ = await _run_turn(store, tool_steps=1)
        log_path = store.event_log_path(agent.session_id)
        assert log_path is not None
        assert log_path == store.storage_dir / EVENT_LOG_SUBDIR / f"{agent.session_id}.jsonl"
        types = _types(_events(log_path))
        assert types[0] == "TURN_START" and types[-1] == "TURN_END", types
        assert "TOOL_CALL" in types and "TOOL_RESULT" in types, types

    @pytest.mark.asyncio
    async def test_a_room_seat_turn_lands_in_the_seat_session_log(self, tmp_path: Path) -> None:
        """Seats persist through the same store as chat, under the seat's own session id.

        Killed by: src/uclone_x/agent/session.py :: if log_writer is None and log_allocator is None:
        Becomes: if False:
        """
        room_store = RoomStore(tmp_path / "rooms")
        service = RoomService(room_store)
        service.create("Events", room_id="room_ev")
        service.add_participant("room_ev", "user", kind=ParticipantKind.HUMAN)
        service.add_participant("room_ev", "scout")
        seat = next(p for p in service.get("room_ev").participants if p.id == "scout")
        store = SessionStore(tmp_path / "sessions")
        host = HostDependencies(
            bus=EventBus(),
            llm=MockLLMConnector(default_response="the index is fine"),
            tools=ToolRegistry(),
            tracer=TelemetryTracer(),
            store=store,
        )
        orchestrator = RoomOrchestrator(
            store=room_store,
            selectors=(MentionSelector(),),
            resolver=RoomAgentResolver(host),
        )
        await orchestrator.post("room_ev", "user", "@scout check the index")

        log_path = store.event_log_path(seat.session_id)
        assert log_path is not None and log_path.is_file(), f"no seat log at {log_path}"
        events = _events(log_path)
        assert {e["session_id"] for e in events} == {seat.session_id}
        assert any(
            e["type"] == "ASSISTANT_MESSAGE" and e["content"] == "the index is fine" for e in events
        ), _types(events)


# --------------------------------------------------------------------------------------
# The store's contract
# --------------------------------------------------------------------------------------


class TestTheStoreRefusesEventsItCannotWrite:
    @pytest.mark.parametrize("wired", ["writer_only", "allocator_only"])
    def test_a_half_wired_store_raises_and_writes_no_record(
        self, tmp_path: Path, wired: str
    ) -> None:
        """Raised, not logged: the caller's queue keeps the events and nothing is written.

        Killed by: src/uclone_x/agent/session.py :: and self._event_log_dir is None
        Becomes: and False
        """
        store = SessionStore(
            tmp_path / "sessions",
            log_writer=RedactingLogWriter(tmp_path / "log.jsonl")
            if wired == "writer_only"
            else None,
            log_allocator=FileLogOffsetAllocator(tmp_path / "cursors")
            if wired == "allocator_only"
            else None,
        )
        state = SessionState(session_id="s1", agent_id="a1")
        with pytest.raises(SessionEventLogNotConfiguredError):
            store.save(state, pending_events=[{"type": "TURN_START"}])
        assert store.load("s1") is None, "the record was written without its events"
        if wired == "writer_only":
            assert not (tmp_path / "log.jsonl").exists()
        else:
            assert not (tmp_path / "cursors" / "s1.cursor").exists()

    @pytest.mark.asyncio
    async def test_a_refused_persist_keeps_the_events_in_the_agent_queue(
        self, tmp_path: Path
    ) -> None:
        """Through `persist_session`: the refusal propagates and the queue is not drained.

        Killed by: src/uclone_x/agent/session.py :: and self._event_log_dir is None
        Becomes: and False
        """
        store = SessionStore(
            tmp_path / "sessions", log_writer=RedactingLogWriter(tmp_path / "log.jsonl")
        )
        llm = _ScriptedLLM(tool_steps=1)
        registry = ToolRegistry()
        registry.register(_EchoTool())
        config = AgentConfig(
            agent_id="agent_1", name="Agent", llm_config=AgentLLMConfig(model_name="scripted")
        )
        agent = BaseAgent(config=config, llm=llm, tools=registry, store=store)
        await agent.start()
        assert (await agent.execute_turn("start")).is_completed
        queued = agent.pending_durable_events
        assert queued, "the turn queued no events, so this test would prove nothing"

        with pytest.raises(SessionEventLogNotConfiguredError):
            agent.persist_session()
        assert agent.pending_durable_events == queued
        assert store.load(agent.session_id) is None

    def test_a_half_wired_store_still_saves_a_record_with_no_events(self, tmp_path: Path) -> None:
        """The refusal is about events it would drop, not about the store as a whole."""
        store = SessionStore(
            tmp_path / "sessions", log_writer=RedactingLogWriter(tmp_path / "log.jsonl")
        )
        saved = store.save(SessionState(session_id="s1", agent_id="a1"))
        assert saved.revision == 1


class TestTheDefaultEventLog:
    def test_offsets_continue_across_saves_and_the_header_is_written_once(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/agent/session.py :: {**cast("Mapping[str, Any]", event), "offset": int(cursor)}
        Becomes: {**cast("Mapping[str, Any]", event)}
        Killed by: src/uclone_x/agent/session.py :: if not log_path.is_file() or log_path.stat().st_size == 0:
        Becomes: if False:
        Killed by: src/uclone_x/agent/session.py :: if not log_path.is_file() or log_path.stat().st_size == 0:
        Becomes: if True:
        """
        store = SessionStore(tmp_path)
        saved = store.save(
            SessionState(session_id="s1", agent_id="a1"),
            pending_events=[{"type": "TURN_START"}, {"type": "TURN_END"}],
        )
        store.save(saved, pending_events=[{"type": "TURN_START"}])

        log_path = store.event_log_path("s1")
        assert log_path is not None
        lines = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert sum(1 for line in lines if "schema" in line) == 1, lines
        assert [e["offset"] for e in _events(log_path)] == [1, 2, 3]

    def test_a_store_that_writes_no_event_creates_no_event_directory(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.save(SessionState(session_id="s1", agent_id="a1"))
        store.load("s1")
        assert not (tmp_path / EVENT_LOG_SUBDIR).exists()

    def test_deleting_a_session_deletes_its_log_and_cursor(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/session.py :: log_path.unlink(missing_ok=True)
        Becomes: pass
        Killed by: src/uclone_x/agent/session.py :: (log_path.parent / f"{session_id}.cursor").unlink(missing_ok=True)
        Becomes: pass
        """
        store = SessionStore(tmp_path)
        store.save(
            SessionState(session_id="s1", agent_id="a1"), pending_events=[{"type": "TURN_START"}]
        )
        log_path = store.event_log_path("s1")
        assert log_path is not None and log_path.is_file()
        cursor = tmp_path / EVENT_LOG_SUBDIR / "s1.cursor"
        assert cursor.is_file()

        assert store.delete("s1") is True
        assert not log_path.exists()
        assert not cursor.exists()
        # A new session under the same id starts a new log, not the old one's tail.
        store.save(
            SessionState(session_id="s1", agent_id="a1"), pending_events=[{"type": "TURN_START"}]
        )
        assert [e["offset"] for e in _events(log_path)] == [1]

    def test_a_failed_append_keeps_offsets_unique_on_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A save that fails partway leaves its written lines' offsets used, not reused.

        The retry writes the events again (content is duplicated, which the log does not
        hide), but under new offsets, so offsets still name one line each.

        Killed by: src/uclone_x/agent/session.py :: log_allocator.record_appended(state.session_id, written)
        Becomes: pass
        """
        real_write = RedactingLogWriter.write_entry
        calls = {"n": 0}

        def flaky(self: RedactingLogWriter, entry: Any) -> str:
            calls["n"] += 1
            if calls["n"] == 3:  # header, first event, then the second event fails
                raise OSError("disk full")
            return real_write(self, entry)

        store = SessionStore(tmp_path)
        state = SessionState(session_id="s1", agent_id="a1")
        events = [{"type": "TURN_START"}, {"type": "TURN_END"}]
        monkeypatch.setattr(RedactingLogWriter, "write_entry", flaky)
        with pytest.raises(OSError, match="disk full"):
            store.save(state, pending_events=events)
        assert store.load("s1") is None
        monkeypatch.setattr(RedactingLogWriter, "write_entry", real_write)
        store.save(state, pending_events=events)

        log_path = store.event_log_path("s1")
        assert log_path is not None
        written = [(e["type"], e["offset"]) for e in _events(log_path)]
        assert written == [("TURN_START", 1), ("TURN_START", 2), ("TURN_END", 3)]

    def test_a_cursor_that_cannot_advance_fails_the_save_before_the_record_moves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise the save raises over a record already at the next revision, and the
        caller's retry is refused as stale against its own write.

        Declared by hand: the mutation moves a statement below the record commit, which
        a single-substring replacement cannot express.
        """

        def refuse(self: FileLogOffsetAllocator, session_id: str, offsets: Any) -> None:
            raise OSError("cannot fsync the cursor")

        monkeypatch.setattr(FileLogOffsetAllocator, "record_appended", refuse)
        store = SessionStore(tmp_path)
        with pytest.raises(OSError, match="cursor"):
            store.save(
                SessionState(session_id="s1", agent_id="a1"),
                pending_events=[{"type": "TURN_START"}],
            )
        assert store.load("s1") is None, "the record was committed by a save that failed"

    @pytest.mark.usefixtures("builtin_personas_absent")
    def test_clearing_history_with_a_live_agent_removes_the_log(self, tmp_path: Path) -> None:
        """ "Clear history" leaves the same disk with or without a live agent.

        With no agent the UI deletes the record, and `delete` removes the log. With a live
        one it resets the agent's session in place, and before this the log survived: the
        cleared conversation's tool outputs stayed on disk, and the next turn continued
        the same log after them.

        Killed by: src/uclone_x/agent/base.py :: self._store.clear_event_log(sid)
        Becomes: pass
        """
        from uclone_x.core.provenance import Provenance as _Prov
        from uclone_x.tools import LocalTool
        from uclone_x.tools.models import ToolResult
        from uclone_x.ui.app import create_ui_app

        class ReadFileTool(LocalTool):
            """Answers differently each call, so each conversation's output is its own."""

            def __init__(self) -> None:
                super().__init__(name="read_file", description="Reads a file")
                self.calls = 0

            async def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
                self.calls += 1
                body = "first conversation body" if self.calls == 1 else "second body"
                return ToolResult(success=True, output=body, provenance=_Prov.primary("read_file"))

        registry = ToolRegistry()
        registry.register(ReadFileTool())
        llm = MockLLMConnector(
            responses=["Reading.", "Read it.", "A new conversation."],
            tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"p": "a"})],
        )
        app = create_ui_app(static_dir=tmp_path, llm=llm, tools=registry, storage_dir=tmp_path)
        client = TestClient(app)
        session_id = "sess_clear"
        turn = {"message": "read a", "agent_id": "agent-general", "session_id": session_id}
        assert client.post("/api/turn", json=turn).json()["status"] == "success"
        log_path = tmp_path / CORE_RECORD_SUBDIR / EVENT_LOG_SUBDIR / f"{session_id}.jsonl"
        assert "first conversation body" in log_path.read_text()
        manager = app.state.session_manager
        assert manager.get_agent("agent-general", session_id) is not None, (
            "no live agent, so this would test the delete branch instead"
        )

        res = client.delete(
            "/api/session/history",
            params={"agent_id": "agent-general", "session_id": session_id},
        )
        assert res.status_code == 200, res.text
        assert not log_path.exists(), "the cleared conversation's events are still on disk"
        assert not log_path.with_suffix(".cursor").exists()

        turn["message"] = "hello again"
        assert client.post("/api/turn", json=turn).json()["status"] == "success"
        text = log_path.read_text()
        assert "first conversation body" not in text
        events = _events(log_path)
        assert events[0]["type"] == "TURN_START" and events[0]["offset"] == 1
        assert _types(events).count("TURN_START") == 1

    @pytest.mark.asyncio
    async def test_a_credential_in_a_tool_output_is_redacted_on_disk(self, tmp_path: Path) -> None:
        """The writer redacts twice: per string in `write_entry`, then per line in
        `write_line`. Disabling either one alone survives this test, because the other
        still redacts, so no single-line kill is declared. Disabling both at once
        (`clean = line` together with `return payload` in `redact_log_payload`) fails it.
        """
        store = SessionStore(tmp_path)
        await _run_turn(store, tool_steps=1)
        text = _only_log(store).read_text()
        assert "TOOL_RESULT" in text
        assert _FAKE_KEY not in text
        assert "[REDACTED]" in text


# --------------------------------------------------------------------------------------
# REQUEST_CONTEXT: the delta, and what it must still let a reader rebuild
# --------------------------------------------------------------------------------------


def _request_contexts(store: SessionStore) -> list[dict[str, Any]]:
    return [e for e in _events(_only_log(store)) if e["type"] == "REQUEST_CONTEXT"]


def _digest(messages: list[dict[str, Any]]) -> str:
    canonical = json.dumps(messages, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TestRequestContextIsADelta:
    @pytest.mark.asyncio
    async def test_every_step_request_can_be_rebuilt_from_the_log(self, tmp_path: Path) -> None:
        """Each step's request is rebuilt from the log and the snapshots, and matches what was sent.

        The log carries the conversation as a delta; the tools, identity, slow context and
        turn context come from the session's snapshots (#1421).

        Killed by: src/uclone_x/agent/base.py :: return kept, list(current[kept:])
        Becomes: return kept, list(current[kept + 1:])
        Killed by: src/uclone_x/agent/base.py :: return kept, list(current[kept:])
        Becomes: return kept, list(current)
        """
        from uclone_x.agent.request_record import rebuild_requests
        from uclone_x.core.log_writer import redact_log_payload

        store = SessionStore(tmp_path)
        agent, llm = await _run_turn(store, tool_steps=3)
        state = store.load(agent.get_session().session_id)
        assert state is not None
        rebuilt = rebuild_requests(store, state, read_session_log(_only_log(store)))
        assert len(rebuilt) == len(llm.calls) == 4

        for request, sent in zip(rebuilt, llm.calls, strict=True):
            # Redaction runs on write, so compare against what was sent after the same pass.
            expected = json.loads(
                json.dumps(redact_log_payload([m.model_dump() for m in sent.messages]))
            )
            assert json.loads(json.dumps([m.model_dump() for m in request.request.messages])) == (
                expected
            )
            assert request.request.tools == sent.tools
        # Nothing credential-shaped is in the first request, so it is proven, not just rebuilt.
        assert rebuilt[0].verified
        # The digest is over the unredacted request, so it identifies what was sent.
        contexts = _request_contexts(store)
        last_sent = [m.model_dump() for m in llm.calls[-1].messages]
        assert contexts[-1]["digest"] == _digest(last_sent)

    @pytest.mark.asyncio
    async def test_event_log_bytes_grow_linearly_in_the_steps_of_a_turn(
        self, tmp_path: Path
    ) -> None:
        """Four times the steps costs about four times the bytes, not sixteen.

        Each step here adds ~1 KB of tool output. Measured on this fixture: re-recording the
        whole request per step (the pre-#1442 shape) made the 16-step log 6.8x the 4-step
        one (276,020 vs 40,430 bytes); the delta makes it 3.1x (53,168 vs 16,950). The
        bound of 5 sits between the two. The count assertion below is the sharper check.

        Killed by: src/uclone_x/agent/base.py :: session.last_conversation = conversation
        Becomes: session.last_conversation = []
        """
        small_store = SessionStore(tmp_path / "small")
        await _run_turn(small_store, tool_steps=4)
        large_store = SessionStore(tmp_path / "large")
        _, llm = await _run_turn(large_store, tool_steps=16)

        small = _only_log(small_store).stat().st_size
        large = _only_log(large_store).stat().st_size
        assert large / small < 5, (small, large)

        # And the stronger form: across the turn, every conversation message of the final
        # request is recorded once, not once per step it was present in. The system message
        # is not conversation: it is in the snapshot.
        contexts = _request_contexts(large_store)
        recorded = sum(len(e["appended_messages"]) for e in contexts)
        final = [m for m in llm.calls[-1].messages if m.role is not MessageRole.SYSTEM]
        assert recorded == len(final), (recorded, len(final))
