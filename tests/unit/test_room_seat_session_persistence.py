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
* a room stored before seats were keyed by clone id is rewritten once on read, each
  message keeping the session it was written in (clone-data-scopes §3.8 step 3; section below).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from tests.support.clones import make_clones
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import TurnResult
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.core.agent_home import (
    default_agents_root,
    handle_of,
    replace_clone_file,
    resolve_handle,
    seat_id_for,
)
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.room import store as store_module
from uclone_x.room.clone_ids import migrate_room_file, needs_clone_id_migration
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomMessage,
    RoomPolicy,
    SpeakerRequest,
    TurnState,
)
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import MentionSelector
from uclone_x.room.service import RoomService, participant_session_id
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

            # The head seats a handle as its clone's id (clone-data-scopes §4 step 3).
            seat_id = seat_id_for(SEAT)
            seat_session = next(
                p["session_id"] for p in created.json()["participants"] if p["id"] == seat_id
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

    async def resolve(self, participant: Participant, *, one_seat: bool = False) -> Any:
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


# ======================================================================================
# A room stored before seats were keyed by clone id, read by this build (clone-data-scopes §3.8 step 3).
#
# Every room written before §4 step 3 seats each clone by its handle and carries a `persona`
# on each seat. This build's model refuses that key, and the store rewrites the record once,
# in one save, the first time it reads it. What these pin, in order of what it would cost:
#
# * **Two seats of one clone keep the first.** The rewrite does not go through
#   `RoomService.add_participant`, the only place that refuses a duplicate id, so the
#   migration checks it itself -- and names the drop in its report.
# * **A message's session is the one it was written in.** It is derived from the handle the
#   message was written under, *before* that sender is rewritten to the id; derived after,
#   it names a session that never existed and the trace of every old turn is lost.
# * **A kept seat keeps its stored `session_id`.** It was derived from the old id, and the
#   seat's whole history is recorded under it.
# * **A rename after the migration leaves the room working.** The seat is the id; the handle
#   is read from the clone, so `@new-name` reaches it and nothing in the room is rewritten.
# ======================================================================================

OLD_ROOM = "room_old0000001"


def _old_seat(seat_id: str, persona: str | None = None) -> dict[str, Any]:
    """A seat as a build before §4 step 3 stored it: keyed by handle, with `persona`."""
    return {
        "id": seat_id,
        "kind": "agent",
        "display_name": seat_id,
        "persona": seat_id if persona is None else persona,
        "session_id": participant_session_id(OLD_ROOM, seat_id),
    }


def _human() -> dict[str, Any]:
    return {"id": "user", "kind": "human", "display_name": "You", "persona": ""}


def _said(seq: int, sender: str, content: str, *, turn: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"seq": seq, "sender_id": sender, "content": content}
    if turn is not None:
        row["turn_id"] = turn
    return row


def _old_room(
    tmp_path: Path, seats: list[dict[str, Any]], transcript: list[dict[str, Any]]
) -> RoomStore:
    """Write an old-shape room record into a fresh store and return the store."""
    store = RoomStore(tmp_path / "rooms")
    store.storage_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "room_id": OLD_ROOM,
        "title": "an old conversation",
        "participants": [_human(), *seats],
        "transcript": transcript,
        "revision": 3,
    }
    store.room_path(OLD_ROOM).write_text(json.dumps(record), encoding="utf-8")
    assert needs_clone_id_migration(record)
    return store


def _stored(store: RoomStore) -> dict[str, Any]:
    return json.loads(store.room_path(OLD_ROOM).read_text(encoding="utf-8"))


def _report() -> str:
    logs = sorted(default_agents_root().glob(".migration-*.log"))
    return "".join(log.read_text(encoding="utf-8") for log in logs)


def test_a_second_seat_of_one_clone_leaves_and_the_drop_is_reported(tmp_path: Path) -> None:
    """`writer-1` played `scout`; with seats keyed by clone there is one `scout` seat.

    The first seat is kept, the second leaves the membership, and its turns stay in the
    transcript under the author they were written by -- and the report names the drop.

    Killed by: src/uclone_x/room/clone_ids.py :: if first is not None:
    Becomes: if False:
    """
    make_clones("scout")
    scout = resolve_handle("scout")
    store = _old_room(
        tmp_path,
        [_old_seat("scout"), _old_seat("writer-1", persona="scout")],
        [
            _said(1, "user", "hello"),
            _said(2, "scout", "from the first seat", turn="t1"),
            _said(3, "writer-1", "from the second seat", turn="t2"),
        ],
    )

    state = store.load(OLD_ROOM)

    assert state is not None
    agents = [p.id for p in state.participants if p.kind is ParticipantKind.AGENT]
    assert agents == [scout], "one clone, one seat"
    senders = [m.sender_id for m in state.transcript]
    assert senders == ["user", scout, "writer-1"], "a dropped seat's turns keep their author"
    assert "writer-1" in _report() and "left the room" in _report()


def test_each_clone_message_records_the_session_of_the_handle_it_was_written_under(
    tmp_path: Path,
) -> None:
    """The backfill reads the sender before the rewrite, for a kept seat and a dropped one.

    Derived after the rewrite, a kept seat's message would name `sess_room__…__agt_…`, a
    session nobody ever wrote to, and its trace would be lost for good.

    Killed by: src/uclone_x/room/clone_ids.py :: message["session_id"] = participant_session_id(room_id, sender)
    Becomes: message["session_id"] = participant_session_id(room_id, rename.get(sender, sender))
    """
    make_clones("scout")
    store = _old_room(
        tmp_path,
        [_old_seat("scout"), _old_seat("writer-1", persona="scout")],
        [
            _said(1, "user", "hello"),
            _said(2, "scout", "kept seat", turn="t1"),
            _said(3, "writer-1", "dropped seat", turn="t2"),
        ],
    )

    state = store.load(OLD_ROOM)

    assert state is not None
    by_seq = {m.seq: m for m in state.transcript}
    assert by_seq[2].session_id == participant_session_id(OLD_ROOM, "scout")
    assert by_seq[3].session_id == participant_session_id(OLD_ROOM, "writer-1")
    assert by_seq[1].session_id is None, "a person's message was never a clone's session"
    assert "session_id" not in _stored(store)["transcript"][0], "None is left out of the file"


def test_a_kept_seat_keeps_the_session_it_was_given(tmp_path: Path) -> None:
    """The seat's history is recorded under the session derived from its old id.

    A seat whose session were re-derived from its new id would open with no history, and
    the next turn would answer as if the conversation had just begun.

    Killed by: src/uclone_x/room/clone_ids.py :: seat["id"] = target
    Becomes: seat.update(id=target, session_id=participant_session_id(room_id, target))
    """
    make_clones("scout", "critic")
    store = _old_room(
        tmp_path,
        [_old_seat("scout"), _old_seat("critic")],
        [_said(1, "user", "hello"), _said(2, "critic", "hmm", turn="t1")],
    )

    state = store.load(OLD_ROOM)

    assert state is not None
    seats = {p.id: p.session_id for p in state.participants if p.kind is ParticipantKind.AGENT}
    assert seats == {
        resolve_handle("scout"): participant_session_id(OLD_ROOM, "scout"),
        resolve_handle("critic"): participant_session_id(OLD_ROOM, "critic"),
    }
    assert _stored(store)["revision"] == 3, "a new naming, not a new revision"
    again = store.load(OLD_ROOM)
    assert again is not None
    assert again.participants == state.participants
    assert "persona" not in json.dumps(_stored(store)["participants"]), "rewritten once"


def test_a_persona_naming_no_clone_is_created_as_one(tmp_path: Path) -> None:
    """A seat played a persona nothing has installed: the clone is made, not dropped.

    Killed by: src/uclone_x/room/clone_ids.py :: target = _created(named, base, lines, room_id, old)
    Becomes: target = None
    """
    make_clones("scout")
    store = _old_room(
        tmp_path,
        [_old_seat("helper-1", persona="archivist")],
        [_said(1, "user", "hello"), _said(2, "helper-1", "on it", turn="t1")],
    )

    state = store.load(OLD_ROOM)

    assert state is not None
    archivist = resolve_handle("archivist")
    assert [p.id for p in state.participants if p.kind is ParticipantKind.AGENT] == [archivist]
    assert state.transcript[1].sender_id == archivist
    assert "archivist" in _report()


@pytest.mark.asyncio
async def test_a_renamed_clone_keeps_its_rooms_and_answers_its_new_handle(
    tmp_path: Path,
) -> None:
    """After the migration a rename rewrites the clone's own file and nothing else.

    The seat is the clone's id, so the room still loads and still seats it; `@` reaches
    it by the handle it has now, read from the clone rather than from the room.

    Killed by: src/uclone_x/room/selectors.py :: if handle is not None and handle.lower() == lowered:
    Becomes: if False:
    """
    make_clones("scout")
    scout = resolve_handle("scout")
    store = _old_room(
        tmp_path,
        [_old_seat("scout")],
        [_said(1, "user", "hello"), _said(2, "scout", "hi", turn="t1")],
    )
    assert store.load(OLD_ROOM) is not None  # migrated

    replace_clone_file(scout, "handle: pathfinder\n")
    state = store.load(OLD_ROOM)

    assert state is not None
    seat = next(p for p in state.participants if p.kind is ParticipantKind.AGENT)
    assert seat.id == scout
    assert handle_of(scout) == "pathfinder"
    human = next(p for p in state.participants if p.kind is ParticipantKind.HUMAN)
    decision = await MentionSelector().select(
        SpeakerRequest(
            room_id=OLD_ROOM,
            participants=(human, seat),
            transcript=(RoomMessage(seq=1, sender_id=human.id, content="@pathfinder go"),),
            turn_state=TurnState(),
            policy=RoomPolicy(),
        )
    )
    assert decision.speaker_id == scout


def test_a_room_that_already_seats_by_id_is_not_rewritten(tmp_path: Path) -> None:
    """The trigger is the `persona` key; a record without it is read as it is."""
    make_clones("scout")
    scout = resolve_handle("scout")
    store = RoomStore(tmp_path / "rooms")
    store.storage_dir.mkdir(parents=True, exist_ok=True)
    seat = Participant(
        id=scout,
        kind=ParticipantKind.AGENT,
        display_name="Scout",
        session_id=participant_session_id(OLD_ROOM, scout),
    )
    record = {
        "room_id": OLD_ROOM,
        "title": "new",
        "participants": [seat.model_dump(mode="json")],
        "transcript": [],
        "revision": 1,
    }
    store.room_path(OLD_ROOM).write_text(json.dumps(record), encoding="utf-8")

    assert not needs_clone_id_migration(record)
    assert store.load(OLD_ROOM) is not None
    assert _stored(store)["revision"] == 1


def test_two_readers_opening_one_old_room_at_once_both_read_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader that loses the migration race reads the winner's record, not an error.

    Both readers fail validation before either rewrites; the barrier makes that certain
    rather than likely. The first rewrites the room, the second finds nothing left to
    rewrite and gets `False` -- which is not a broken room, since the file is now valid.
    The first launch after an upgrade does exactly this: the survey and the open room load
    one room on two threads of the head's pool.

    Killed by: src/uclone_x/room/store.py :: migrate_room_file(path)
    Becomes: if not migrate_room_file(path): raise UnreadableRoomRecordError(room_id)
    """
    make_clones("scout")
    scout = resolve_handle("scout")
    store = _old_room(
        tmp_path,
        [_old_seat("scout")],
        [_said(1, "user", "hello"), _said(2, "scout", "hi", turn="t1")],
    )
    both_failed_validation = threading.Barrier(2, timeout=10)
    outcomes: list[bool] = []

    def migrate_after_both(path: Path) -> bool:
        both_failed_validation.wait()
        migrated = migrate_room_file(path)
        outcomes.append(migrated)
        return migrated

    monkeypatch.setattr(store_module, "migrate_room_file", migrate_after_both)

    with ThreadPoolExecutor(max_workers=2) as pool:
        loads = [pool.submit(store.load, OLD_ROOM) for _ in range(2)]
        states = [load.result(timeout=30) for load in loads]

    assert sorted(outcomes) == [False, True], "one reader rewrote, the other found it done"
    for state in states:
        assert state is not None
        assert [p.id for p in state.participants if p.kind is ParticipantKind.AGENT] == [scout]
