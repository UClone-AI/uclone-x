"""A room seat's knowledge survives a restart (#1367).

A seat is composed with an ontology engine of its own, and until #1367 that engine lived only
in the running agent: after a restart the room's knowledge read answered `not_running` until
the seat spoke again, and even then the engine was rebuilt from nothing. The seat's *session*
was already written after every turn (#1361); its knowledge is now written beside it, by the
same step, and read back in two places without anything else changing:

* a fresh resolver loads it into the seat's engine before the seat's first turn, so the seat
  continues from what it had learned rather than from an empty graph;
* the knowledge read answers from the saved record when no agent is running, and does not
  build one to find out.

What these pin, in order:

* a fact learned in a turn is on disk after it, and a fresh resolver's engine holds it;
* the same through the head, across a real restart (a second app over the same storage), with
  no turn taken after it and no agent built by the read;
* a save that fails is stated on the row it concerns, and the reply still lands (P6);
* a saved record that cannot be read is set aside -- renamed, never deleted -- when the seat
  is built; the read reports it as unreadable and changes nothing. Neither says where the file is or what the parser said:
  that is the log's (P6, and the rule in `reader_facing_reason`);
* a seat with nothing saved says so, instead of claiming it remembers nothing (pinned in
  `test_ui_room_dock_routes.py`).
"""

from __future__ import annotations

import asyncio
import os
import re
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import TurnResult
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.errors import SeatKnowledgeUnreadableError
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.models import OntologyRelation
from uclone_x.room.knowledge import KnowledgeLoad
from uclone_x.room.knowledge_store import SeatKnowledgeStore
from uclone_x.room.models import Participant, ParticipantKind
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import MentionSelector
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.registry import ToolRegistry
from uclone_x.ui.rooms import reader_facing_reason

ROOM = "room_knows"
SEAT = "scout"


def _fact() -> OntologyRelation:
    return OntologyRelation(source_entity="Postgres", predicate="is_a", target_entity="Database")


class _LearningLLM(MockLLMConnector):
    """Answers, and while answering has the seat learn a fact: a turn in which it learns.

    The runtime has no step that induces into a seat's engine on its own yet, so the test
    stands in for one -- inside the turn, before the orchestrator's post-turn writes run,
    which is where any real induction would also have to land.
    """

    def __init__(self) -> None:
        super().__init__(default_response="noted")
        self.learn: Callable[[], None] | None = None

    async def generate(self, request: LLMRequest) -> ModelResponse:
        if self.learn is not None:
            self.learn()
            self.learn = None
        return await super().generate(request)


def _host(sessions_dir: Path, llm: MockLLMConnector) -> HostDependencies:
    return HostDependencies(
        bus=EventBus(),
        llm=llm,
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(sessions_dir),
    )


def _store(tmp_path: Path) -> SeatKnowledgeStore:
    return SeatKnowledgeStore(
        tmp_path / "knowledge",
        engine_factory=lambda namespace: OntologyEngine(namespace_iri=namespace),
    )


def _resolver(
    tmp_path: Path, llm: MockLLMConnector, knowledge: SeatKnowledgeStore | None = None
) -> RoomAgentResolver:
    return RoomAgentResolver(
        _host(tmp_path / "sessions", llm),
        ontology_factory=lambda namespace: OntologyEngine(namespace_iri=namespace),
        knowledge=knowledge if knowledge is not None else _store(tmp_path),
    )


#: A record the YAML parser refuses, and one it reads but whose second relation the engine's
#: models refuse -- after the first has already been loaded.
_NOT_YAML = "concepts: [unterminated"
_HALF_A_RECORD = (
    "relations:\n"
    "- {source_entity: Stale, predicate: is_a, target_entity: Fact}\n"
    "- {predicate: is_a}\n"
)


def _assert_plain(text: str, path: Path) -> None:
    """Copy for a non-expert: no file location, no Python class name, no parser output."""
    assert str(path.parent) not in text and path.name not in text, text
    assert "/" not in text and "\\" not in text, text
    assert not re.search(r"[A-Za-z]*(Error|Exception)\b", text), text
    for parser_word in ("yaml", "YAML", "line ", "column", "while ", "expected", "unterminated"):
        assert parser_word not in text, (parser_word, text)


def _set_aside(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.stem}.unreadable-*.yaml"))


def _seated_room(tmp_path: Path) -> tuple[RoomStore, RoomService, Participant]:
    store = RoomStore(tmp_path / "rooms")
    service = RoomService(store)
    service.create("Knowledge", room_id=ROOM)
    service.add_participant(ROOM, "user", kind=ParticipantKind.HUMAN)
    service.add_participant(ROOM, SEAT)
    seat = next(p for p in service.get(ROOM).participants if p.id == SEAT)
    return store, service, seat


class TestASeatsKnowledgeSurvivesARestart:
    @pytest.mark.asyncio
    async def test_a_fact_learned_in_a_turn_is_reloaded_by_a_fresh_resolver(
        self, tmp_path: Path
    ) -> None:
        """Learn in one turn; rebuild the resolver over the same directory; the engine has it.

        Killed by: src/uclone_x/room/orchestrator.py :: knowledge_persist_error = self._persist_knowledge(agent, speaker)
        Becomes: knowledge_persist_error = None
        Killed by: src/uclone_x/room/resolver.py :: self._knowledge.load_into(participant.session_id, engine)
        Becomes: None
        """
        room_store, _service, seat = _seated_room(tmp_path)
        llm = _LearningLLM()
        resolver = _resolver(tmp_path, llm)
        orchestrator = RoomOrchestrator(
            store=room_store,
            selectors=(MentionSelector(),),
            resolver=resolver,
            knowledge=_store(tmp_path),
        )

        def learn() -> None:
            live = resolver.live_agent(seat.session_id)
            assert live is not None and live.ontology is not None
            live.ontology.register_relation(_fact())

        llm.learn = learn
        state = await orchestrator.post(ROOM, "user", "@scout what is Postgres?")
        assert state.transcript[-1].knowledge_persist_error is None

        # A restart: new objects over the same directories, nothing carried in memory.
        restarted = _resolver(tmp_path, MockLLMConnector(default_response="unused"))
        agent = await restarted.resolve(seat)
        assert agent.ontology is not None
        relations = cast(OntologyEngine, agent.ontology).list_relations()
        assert [(r.source_entity, r.target_entity) for r in relations] == [("Postgres", "Database")]

    @pytest.mark.asyncio
    async def test_an_unreadable_record_is_set_aside_and_the_seat_starts_over(
        self, tmp_path: Path
    ) -> None:
        """The record is renamed, kept byte for byte; the seat speaks, learns, and says so.

        Refusing the seat left a clone that could never speak again in that conversation;
        building it over an empty engine without the rename would save that emptiness over
        the record. The notice is on the first row after the set-aside, and on no later one.

        Killed by: src/uclone_x/room/knowledge_store.py :: os.rename(path, aside)
        Becomes: pass
        Killed by: src/uclone_x/room/orchestrator.py :: knowledge_set_aside=self._knowledge_set_aside(speaker),
        Becomes: knowledge_set_aside=False,
        Killed by: src/uclone_x/room/knowledge_store.py :: self._set_aside.discard(session_id)
        Becomes: pass
        """
        room_store, _service, seat = _seated_room(tmp_path)
        knowledge = _store(tmp_path)
        record = knowledge.path(seat.session_id)
        record.parent.mkdir(parents=True)
        record.write_text(_NOT_YAML, encoding="utf-8")
        original = record.read_bytes()
        llm = _LearningLLM()
        # One store for both, as the head composes it: the set-aside is taken by the turn.
        resolver = _resolver(tmp_path, llm, knowledge)
        orchestrator = RoomOrchestrator(
            store=room_store,
            selectors=(MentionSelector(),),
            resolver=resolver,
            knowledge=knowledge,
        )

        def learn() -> None:
            live = resolver.live_agent(seat.session_id)
            assert live is not None and live.ontology is not None
            live.ontology.register_relation(_fact())

        llm.learn = learn
        state = await orchestrator.post(ROOM, "user", "@scout what is Postgres?")

        row = state.transcript[-1]
        assert row.sender_id == SEAT and row.content == "noted", row
        assert row.error is None and row.knowledge_persist_error is None
        assert row.knowledge_set_aside is True
        kept = _set_aside(record)
        assert len(kept) == 1 and kept[0].read_bytes() == original
        # What it learned after starting over is the record now.
        assert knowledge.read(seat.session_id, seat.ontology_namespace) is not None

        state = await orchestrator.post(ROOM, "user", "@scout and again?")
        assert state.transcript[-1].knowledge_set_aside is False
        assert _set_aside(record) == kept and kept[0].read_bytes() == original

    @pytest.mark.asyncio
    async def test_a_half_loaded_record_is_not_what_the_seat_starts_over_from(
        self, tmp_path: Path
    ) -> None:
        """The engine the record failed in may hold part of it; the seat gets a fresh one.

        Killed by: src/uclone_x/room/resolver.py :: engine = fresh(participant.ontology_namespace)
        Becomes: pass
        """
        _room_store, _service, seat = _seated_room(tmp_path)
        record = _store(tmp_path).path(seat.session_id)
        record.parent.mkdir(parents=True)
        record.write_text(_HALF_A_RECORD, encoding="utf-8")

        agent = await _resolver(tmp_path, MockLLMConnector()).resolve(seat)

        assert agent.ontology is not None
        assert cast(OntologyEngine, agent.ontology).list_relations() == []
        assert len(_set_aside(record)) == 1

    @pytest.mark.asyncio
    async def test_a_record_that_cannot_be_set_aside_refuses_the_seat_in_plain_words(
        self, tmp_path: Path
    ) -> None:
        """If the rename fails, the seat is refused -- and the words shown for it are plain.

        The refusal is a `RoomError`, which `reader_facing_reason` passes to the
        conversation as it is; the path and the cause are on the exception for the log.

        Killed by: src/uclone_x/room/knowledge_store.py :: raise unreadable from exc
        Becomes: return KnowledgeLoad.ABSENT
        Killed by: src/uclone_x/room/resolver.py :: f"{participant.display_name}'s knowledge record for this "
        Becomes: f"{exc.path} knowledge record for this "
        """
        _room_store, _service, seat = _seated_room(tmp_path)
        record = _store(tmp_path).path(seat.session_id)
        record.parent.mkdir(parents=True)
        record.write_text(_NOT_YAML, encoding="utf-8")
        original = record.read_bytes()
        mode = stat.S_IMODE(os.stat(record.parent).st_mode)
        os.chmod(record.parent, stat.S_IRUSR | stat.S_IXUSR)  # no rename in here
        try:
            with pytest.raises(SeatKnowledgeUnreadableError) as refused:
                await _resolver(tmp_path, MockLLMConnector()).resolve(seat)
        finally:
            os.chmod(record.parent, mode)

        shown = reader_facing_reason(refused.value)
        assert "could not be read" in shown and "could not be set aside" in shown, shown
        # Only this conversation's record is concerned; the clone's saved memory is not (#1434).
        assert "knowledge record for this conversation" in shown, shown
        assert "nothing remembered" not in shown and "start over" not in shown, shown
        _assert_plain(shown, record)
        assert refused.value.path == record and refused.value.cause, "the log's half is missing"
        assert record.read_bytes() == original and _set_aside(record) == []

    def test_the_store_answers_a_set_aside_once(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/room/knowledge_store.py :: return KnowledgeLoad.SET_ASIDE
        Becomes: return KnowledgeLoad.ABSENT
        """
        knowledge = _store(tmp_path)
        record = knowledge.path("sess_x")
        record.parent.mkdir(parents=True)
        record.write_text(_NOT_YAML, encoding="utf-8")

        assert knowledge.take_set_aside("sess_x") is False
        assert knowledge.load_into("sess_x", OntologyEngine()) is KnowledgeLoad.SET_ASIDE
        assert knowledge.take_set_aside("sess_x") is True
        assert knowledge.take_set_aside("sess_x") is False
        assert knowledge.load_into("sess_x", OntologyEngine()) is KnowledgeLoad.ABSENT


# --------------------------------------------------------------------------------------
# Through the head, across a real restart.
# --------------------------------------------------------------------------------------


def _app(storage: Path, llm: MockLLMConnector) -> Any:
    from uclone_x.ui.app import create_ui_app

    return create_ui_app(
        static_dir=storage / "static",
        storage_dir=storage / "sessions",
        workspace_dir=storage / "workspace",
        llm=llm,
    )


def _wait_for_reply(client: TestClient, room_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10.0
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = client.get(f"/api/rooms/{room_id}").json()
        rows = latest.get("transcript", [])
        if rows and rows[-1]["sender_id"] == SEAT:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"{SEAT} never replied; last was {latest}")


def _knowledge(client: TestClient, room_id: str) -> dict[str, Any]:
    answer = client.get(f"/api/rooms/{room_id}/knowledge", params={"agent_id": SEAT})
    assert answer.status_code == 200, answer.text
    return cast(dict[str, Any], answer.json())


class TestTheKnowledgeReadAfterARestart:
    def test_what_a_seat_learned_is_read_after_a_restart_without_a_new_turn(
        self, tmp_path: Path
    ) -> None:
        """The issue's acceptance: learn, restart, read -- `ok`, with the fact, no turn taken.

        Killed by: src/uclone_x/ui/room_dock.py :: engine = stack.knowledge.read(seat.session_id, seat.ontology_namespace)
        Becomes: engine = None
        Killed by: src/uclone_x/ui/rooms.py :: knowledge=self.knowledge,  # written after each turn (#1367)
        Becomes: knowledge=None,  # written after each turn (#1367)
        Killed by: src/uclone_x/ui/rooms.py :: knowledge=self.knowledge,  # read before a seat's first turn (#1367)
        Becomes: knowledge=None,  # read before a seat's first turn (#1367)
        """
        llm = _LearningLLM()
        with TestClient(_app(tmp_path, llm)) as client:
            created = client.post("/api/rooms", json={"title": "Probe", "agent_ids": [SEAT]})
            assert created.status_code == 201, created.text
            room_id = created.json()["room_id"]
            session_id = next(
                p["session_id"] for p in created.json()["participants"] if p["id"] == SEAT
            )
            stack = cast(Any, client.app).state.room_stack

            def learn() -> None:
                stack.live_agent(room_id, session_id).ontology.register_relation(_fact())

            llm.learn = learn
            client.post(f"/api/rooms/{room_id}/messages", json={"content": "what is Postgres?"})
            landed = _wait_for_reply(client, room_id)
            rows_before_restart = len(landed["transcript"])
            assert landed["transcript"][-1]["knowledge_persist_error"] is None
            assert llm.learn is None, "the seat's turn never reached the model"

        with TestClient(_app(tmp_path, MockLLMConnector(default_response="unused"))) as client:
            body = _knowledge(client, room_id)
            stack = cast(Any, client.app).state.room_stack

            assert body["status"] == "ok", body
            assert [s["statement"] for s in body["remembers"]] == ["Postgres is a Database"]
            assert body["reason"] is None
            # The read did not build the seat to answer: no turn, no agent.
            assert stack.live_agent(room_id, session_id) is None
            transcript = client.get(f"/api/rooms/{room_id}").json()["transcript"]
            assert len(transcript) == rows_before_restart

            # And when the seat is next built, it continues from what it knew.
            state = stack.service.get(room_id)
            seat = next(p for p in state.participants if p.id == SEAT)
            agent = asyncio.run(stack.resolve_agent(state, seat))
            assert [r.source_entity for r in agent.ontology.list_relations()] == ["Postgres"]

    def test_an_unreadable_saved_record_is_stated_not_shown_as_empty(self, tmp_path: Path) -> None:
        """The read says the record is damaged, in plain words, and leaves it where it is.

        Killed by: src/uclone_x/ui/room_dock.py :: except SeatKnowledgeUnreadableError:
        Becomes: except ZeroDivisionError:
        Killed by: src/uclone_x/ui/room_dock.py :: f"{name}'s knowledge record for this conversation could not be read, "
        Becomes: f"{stack.knowledge.path(seat.session_id)} could not be read, "
        """
        with TestClient(_app(tmp_path, MockLLMConnector())) as client:
            created = client.post("/api/rooms", json={"title": "Broken", "agent_ids": [SEAT]})
            room_id = created.json()["room_id"]
            session_id = next(
                p["session_id"] for p in created.json()["participants"] if p["id"] == SEAT
            )
            stack = cast(Any, client.app).state.room_stack
            path: Path = stack.knowledge.path(session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_NOT_YAML, encoding="utf-8")

            body = _knowledge(client, room_id)

            assert body["status"] == "unreadable", body
            assert body["remembers"] is None
            assert "could not be read" in body["reason"]
            _assert_plain(body["reason"], path)
            # A read changes nothing: the record is where it was, and nothing was set aside.
            assert path.read_text(encoding="utf-8") == _NOT_YAML and _set_aside(path) == []


# --------------------------------------------------------------------------------------
# Orchestrator-level: a failed save is said on the row it concerns.
# --------------------------------------------------------------------------------------


class _Seat:
    """The protocol surface the orchestrator touches: a turn, a session write, an engine."""

    def __init__(self) -> None:
        self.ontology = OntologyEngine()

    async def execute_turn(
        self, prompt: str, *, stream_callback: Any = None, **kwargs: Any
    ) -> TurnResult:
        return TurnResult(turn_index=1, content="done", provenance=None)

    def checkpoint_turn(self, session_id: str | None = None) -> SessionState:
        return SessionState(session_id=session_id or "x", agent_id=SEAT)

    def roll_back_turn(self, checkpoint: SessionState, *, reason: str) -> int:
        return 0

    def persist_session(self, session_id: str | None = None) -> SessionState:
        return SessionState(session_id=session_id or "x", agent_id=SEAT)


class _OneSeat:
    def __init__(self, agent: _Seat) -> None:
        self.agent = agent

    async def resolve(self, participant: Participant) -> Any:
        return self.agent


class _FailingKnowledge:
    def load_into(self, session_id: str, engine: Any) -> KnowledgeLoad:
        return KnowledgeLoad.ABSENT

    def take_set_aside(self, session_id: str) -> bool:
        return False

    def read(self, session_id: str, namespace: str) -> None:
        return None

    def save(self, session_id: str, engine: Any) -> None:
        raise OSError("disk full")


class TestAFailedKnowledgeSaveIsStated:
    @pytest.mark.asyncio
    async def test_the_row_names_the_failure_and_the_reply_still_lands(
        self, tmp_path: Path
    ) -> None:
        """P6: the reply is real; that what the seat learned will not survive is said.

        Not folded into `error` for the reason `persist_error` is not: the turn succeeded.

        Killed by: src/uclone_x/room/orchestrator.py :: return stated
        Becomes: return None
        """
        room_store, _service, _seat = _seated_room(tmp_path)
        orchestrator = RoomOrchestrator(
            store=room_store,
            selectors=(MentionSelector(),),
            resolver=_OneSeat(_Seat()),
            knowledge=_FailingKnowledge(),
        )

        state = await orchestrator.post(ROOM, "user", "@scout go")

        row = state.transcript[-1]
        assert row.sender_id == SEAT
        assert row.content == "done"
        assert row.error is None
        assert row.persist_error is None
        assert row.knowledge_persist_error == "OSError: disk full"
        assert row.knowledge_set_aside is False
