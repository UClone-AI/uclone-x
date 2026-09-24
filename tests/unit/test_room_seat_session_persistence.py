"""A room seat's session survives a restart.

Every seated agent owns a `sess_room__{room}__{agent}` session (design doc
`multi-agent-conversational-orchestration.md` §3.2, G3), and `RoomAgentResolver.resolve`
hydrates it before the seat's first turn so a room that outlives its process resumes. The
resume only works if something *wrote* the record. Nothing did: the orchestrator ran the
turn and saved the room transcript, and the seat's own session -- the model context, tool
calls and results the transcript deliberately leaves out -- stayed in memory. A runtime
probe found `sessions/core/` empty after room turns, so a restarted room's agents answered
from a blank history while the transcript on screen said otherwise.

What these pin, in order:

* a seat's session is on disk after its turn, and a fresh resolver over the same store --
  what a restart is -- hands back an agent holding it;
* the same through the head, since that is where the probe found the gap;
* a failed turn is written too, as the chat route writes one: the agent in memory is what
  the next turn uses, and a restart must not see a different conversation -- what is
  written is the seat with the failed turn rolled back out of it (#1423);
* a write that fails says so on the row it concerns (P6), and the utterance still lands.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import TurnResult
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.room.models import Participant, ParticipantKind
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import MentionSelector
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.registry import ToolRegistry

ROOM = "room_persist"
SEAT = "scout"


def _host(sessions_dir: Path, reply: str) -> HostDependencies:
    return HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(default_response=reply),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(sessions_dir),
    )


def _seat_session_id(service: RoomService) -> str:
    seat = next(p for p in service.get(ROOM).participants if p.id == SEAT)
    return seat.session_id


def _seated_room(tmp_path: Path) -> tuple[RoomStore, RoomService]:
    store = RoomStore(tmp_path / "rooms")
    service = RoomService(store)
    service.create("Persistence", room_id=ROOM)
    service.add_participant(ROOM, "user", kind=ParticipantKind.HUMAN)
    service.add_participant(ROOM, SEAT)
    return store, service


class TestASeatSessionSurvivesARestart:
    @pytest.mark.asyncio
    async def test_a_seat_session_is_written_and_reloaded_by_a_fresh_resolver(
        self, tmp_path: Path
    ) -> None:
        """Run one turn, then rebuild the resolver over the same directory and ask again.

        Killed by: src/uclone_x/room/orchestrator.py :: persist_error = rollback_error or self._persist_seat(agent, speaker)
        Becomes: persist_error = rollback_error
        """
        room_store, service = _seated_room(tmp_path)
        sessions_dir = tmp_path / "sessions"
        session_id = _seat_session_id(service)

        orchestrator = RoomOrchestrator(
            store=room_store,
            selectors=(MentionSelector(),),
            resolver=RoomAgentResolver(_host(sessions_dir, "the index is fine")),
        )
        await orchestrator.post(ROOM, "user", "@scout check the index")

        record = SessionStore(sessions_dir).load(session_id)
        assert record is not None, f"no Core record for {session_id} under {sessions_dir}"
        assert any("the index is fine" in str(m.content) for m in record.messages)

        # A restart: a new store object and a new resolver, nothing carried over in memory.
        restarted = RoomAgentResolver(_host(sessions_dir, "unused"))
        seat = next(p for p in service.get(ROOM).participants if p.id == SEAT)
        agent = await restarted.resolve(seat)
        resumed = cast(Any, agent).get_session(session_id)
        contents = [str(m.content) for m in resumed.messages]
        assert any("check the index" in c for c in contents), contents
        assert any("the index is fine" in c for c in contents), contents


def _wait_for_rows(client: TestClient, room_id: str, rows: int) -> dict[str, Any]:
    deadline = time.monotonic() + 10.0
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = client.get(f"/api/rooms/{room_id}").json()
        if len(latest.get("transcript", [])) >= rows:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"room never reached {rows} rows; last was {latest}")


class TestTheHeadWritesTheSeat:
    def test_a_room_turn_through_the_head_leaves_a_core_record_for_the_seat(
        self, tmp_path: Path
    ) -> None:
        """The probe's own observation, as a test: `sessions/core/` after a room turn.

        Killed by: src/uclone_x/room/orchestrator.py :: persist_error = rollback_error or self._persist_seat(agent, speaker)
        Becomes: persist_error = rollback_error
        """
        from uclone_x.ui.app import AgentSessionManager, create_ui_app

        app = create_ui_app(
            static_dir=tmp_path,
            storage_dir=tmp_path / "sessions",
            llm=MockLLMConnector(default_response="looked at it"),
        )
        with TestClient(app) as client:
            created = client.post("/api/rooms", json={"title": "Probe", "agent_ids": [SEAT]})
            assert created.status_code == 201, created.text
            room_id = created.json()["room_id"]
            client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
            landed = _wait_for_rows(client, room_id, 2)

            seat_session = next(
                p["session_id"] for p in created.json()["participants"] if p["id"] == SEAT
            )
            manager = cast(AgentSessionManager, cast(Any, client.app).state.session_manager)
            record = manager.core_store.load(seat_session)
            assert record is not None, (
                f"no Core record for {seat_session}; "
                f"core dir holds {sorted(p.name for p in manager.core_store.storage_dir.iterdir())}"
            )
            assert landed["transcript"][-1].get("persist_error") is None


# --------------------------------------------------------------------------------------
# Orchestrator-level: a failed turn is written, and a failed write is said.
# --------------------------------------------------------------------------------------


class _RecordingAgent:
    """The protocol surface the orchestrator touches, recording each persist."""

    def __init__(self, *, result_error: str | None = None) -> None:
        self.result_error = result_error
        self.persisted: list[str | None] = []
        self.persist_fails_with: Exception | None = None

    async def execute_turn(
        self, prompt: str, *, stream_callback: Any = None, **kwargs: Any
    ) -> TurnResult:
        if self.result_error is not None:
            return TurnResult(turn_index=1, content="", error=self.result_error, provenance=None)
        return TurnResult(turn_index=1, content="done", provenance=None)

    def checkpoint_turn(self, session_id: str | None = None) -> SessionState:
        return SessionState(session_id=session_id or "x", agent_id=SEAT)

    def roll_back_turn(self, checkpoint: SessionState, *, reason: str) -> int:
        return 0

    def persist_session(self, session_id: str | None = None) -> SessionState:
        self.persisted.append(session_id)
        if self.persist_fails_with is not None:
            raise self.persist_fails_with
        return SessionState(session_id=session_id or "x", agent_id=SEAT)


class _OneAgentResolver:
    def __init__(self, agent: _RecordingAgent) -> None:
        self.agent = agent

    async def resolve(self, participant: Participant) -> Any:
        return self.agent


def _orchestrator(tmp_path: Path, agent: _RecordingAgent) -> tuple[RoomOrchestrator, str]:
    room_store, service = _seated_room(tmp_path)
    return (
        RoomOrchestrator(
            store=room_store, selectors=(MentionSelector(),), resolver=_OneAgentResolver(agent)
        ),
        _seat_session_id(service),
    )


class TestTheOrchestratorWritesEverySeatTurn:
    @pytest.mark.asyncio
    async def test_a_failed_turn_is_written_to_the_seats_own_session(self, tmp_path: Path) -> None:
        """The chat route persists a failed turn; a seat must too, or a restart diverges.

        What is written is the seat with the failed turn rolled back out of it (#1423), so
        a restart and the agent in memory still agree; the write itself is what this pins.

        Killed by: src/uclone_x/room/orchestrator.py :: persist_error = rollback_error or self._persist_seat(agent, speaker)
        Becomes: persist_error = rollback_error or (None if error is not None else self._persist_seat(agent, speaker))
        """
        agent = _RecordingAgent(result_error="LLMError: provider unreachable")
        orchestrator, session_id = _orchestrator(tmp_path, agent)

        state = await orchestrator.post(ROOM, "user", "@scout go")

        assert state.transcript[-1].error == "LLMError: provider unreachable"
        assert agent.persisted == [session_id]

    @pytest.mark.asyncio
    async def test_a_seat_write_that_fails_is_stated_on_the_row_it_concerns(
        self, tmp_path: Path
    ) -> None:
        """P6: the reply is real and lands; that it will not survive a restart is said.

        Not folded into `error`: the turn succeeded, and `error` is what `retry` and
        `last_seen_seq` read as "this turn failed" -- re-running a good answer because its
        bookkeeping write failed would spend a turn to produce a second, different reply.

        Killed by: src/uclone_x/room/orchestrator.py :: return f"{type(exc).__name__}: {exc}"
        Becomes: return None
        """
        agent = _RecordingAgent()
        agent.persist_fails_with = OSError("disk full")
        orchestrator, session_id = _orchestrator(tmp_path, agent)

        state = await orchestrator.post(ROOM, "user", "@scout go")

        row = state.transcript[-1]
        assert row.sender_id == SEAT
        assert row.content == "done"
        assert row.error is None
        assert row.persist_error == "OSError: disk full"
        assert agent.persisted == [session_id]
