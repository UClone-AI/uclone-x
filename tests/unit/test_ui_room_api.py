"""Tests for the head's room surface — `/api/rooms`.

Written before `uclone_x.ui.rooms` existed. What they pin, in order of what it would
cost to get wrong:

* **A half-seated room is not left behind.** `RoomService.create` takes a title and
  nothing else; the roster arrives through `add_participant` afterwards. A route that
  creates and then fails to seat leaves a titled, empty room in the list with no way for
  a user to tell it from one they meant to make.
* **A refusal keeps its reason.** `RoomPolicy` refuses knobs that would starve an address,
  and the roster refuses a second human. Both messages say *why*; a route that answers
  422 with a validation dump throws that away, and the surface can then only say "invalid".
* **Sending does not hold the request open.** `post()` returns once the whole cascade has
  landed, so a synchronous route is one HTTP request held across several model calls --
  unabortable, and a proxy timeout on a long exchange.
* **A room's internals are not conversations.** Every seated agent gets its own
  `sess_room__{room}__{agent}` session. Unfiltered, `/api/sessions` shows them as N
  phantom chats per room.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.support.app_clone import app_clone
from tests.support.clones import make_clones
from tests.support.repo import REPO_ROOT
from uclone_x.agent.base import BaseAgent
from uclone_x.core.agent_home import seat_id_for
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.context_window import OllamaContextWindows
from uclone_x.llm.models import LLMRequest, ModelResponse, TokenUsage
from uclone_x.room.models import Participant, ParticipantKind, RoomLoop, RoomPolicy, RoomState
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore, room_storage_dir_under
from uclone_x.ui import rooms as rooms_module
from uclone_x.ui.app import create_ui_app
from uclone_x.ui.rooms import RoomStack, seated_agents

# The seats and agent ids these tests use. Each is a clone now, since a name no clone
# carries is refused memory rather than given a home (clone-data-scopes §3.4).
_TEST_CLONES = ("alpha", "beta", "bob", "champion", "critic", "dba")


@pytest.fixture(autouse=True)
def _test_clones() -> None:  # pyright: ignore[reportUnusedFunction]
    make_clones(*_TEST_CLONES)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A client whose requests share one event loop.

    Context-managed on purpose: sending answers 202 and leaves the cascade running as a
    task. Without the shared portal each request gets its own loop, and the task is left
    holding a queue belonging to a loop that has closed -- which is a property of the test
    harness and not of the route.
    """
    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=tmp_path / "sessions",
        llm=MockLLMConnector(),
    )
    with TestClient(app) as started:
        yield started


@pytest.fixture
def answering_client(tmp_path: Path) -> Iterator[TestClient]:
    """`client`, but a route failure is *answered* rather than re-raised into the test.

    The default `TestClient` lets an unhandled route exception out of `client.get(...)`,
    which a test reads as an error and not as the answer a browser receives. A test whose
    whole subject is what reaches the caller when the Core refuses (#1212) has to observe
    the response, and a mutation that removes the translation has to show up as the bare
    500 it really is rather than as a traceback that never reaches an assertion.
    """
    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=tmp_path / "sessions",
        llm=MockLLMConnector(),
    )
    with TestClient(app, raise_server_exceptions=False) as started:
        yield started


_HANGUL = re.compile(r"[\uac00-\ud7a3]")


def _notes(client: TestClient, room_id: str) -> list[dict[str, Any]]:
    """The application's notes in a room's transcript, oldest first.

    Not the person's typed commands, which are notes too (#1661) but are written by them.
    """
    room = client.get(f"/api/rooms/{room_id}").json()
    return [m for m in room["transcript"] if m.get("kind") == "note" and m["sender_id"] == "system"]


def _create(client: TestClient, **kwargs: Any) -> Any:
    payload: dict[str, Any] = {"title": "Index tuning", "agent_ids": ["scout"]}
    payload.update(kwargs)
    return client.post("/api/rooms", json=payload)


def _stack(client: TestClient) -> RoomStack:
    return cast(RoomStack, cast(Any, client.app).state.room_stack)


def _wait_for_transcript(client: TestClient, room_id: str, rows: int) -> dict[str, Any]:
    """Poll the room until the cascade has written `rows`, or give up loudly.

    The send route answers 202 and runs the turns behind it, so there is no response to
    await. Polling the record is what a head does too.
    """
    deadline = time.monotonic() + 10.0
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = client.get(f"/api/rooms/{room_id}").json()
        if len(latest.get("transcript", [])) >= rows:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"room never reached {rows} rows; last was {latest}")


def _wait_for_loop(
    client: TestClient, room_id: str, *, runs: int, failed: bool = False
) -> dict[str, Any]:
    """Poll the room until its saved loop has started `runs` runs (and, if `failed`, has
    recorded an error), or give up loudly. A run is a background task, like a cascade."""
    deadline = time.monotonic() + 10.0
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = client.get(f"/api/rooms/{room_id}").json()
        loop = latest.get("loop")
        if loop and loop["runs"] >= runs and (not failed or loop.get("last_error")):
            return loop
        time.sleep(0.05)
    raise AssertionError(f"the room's loop never reached {runs} run(s); last was {latest}")


def _wait_for_cascade_to_settle(stack: RoomStack, room_id: str) -> None:
    """Block until no cascade this stack started is still running in the room, or fail loudly.

    Enabling autonomous mode answers 200 and leaves a `resume` running behind it, and that
    run saves the room. A test that loads the record and writes it back through the store
    while the run is still going holds a revision the run is about to pass, and the store
    rightly refuses the write with `StaleRoomWriteError` (#1593). Waiting here is what makes
    the test's own load the latest one; the revision check is not the thing to relax.
    """
    deadline = time.monotonic() + 10.0
    while stack.turn_in_flight(room_id):
        if time.monotonic() >= deadline:
            raise AssertionError(f"room {room_id} still had a cascade running after 10s")
        time.sleep(0.01)


class _SelfDiscardingTask:
    """A task that leaves the room's running set while it is being asked whether it is
    done -- what the loop thread's `discard` does to a reader on another thread."""

    def __init__(self, running: set[Any]) -> None:
        self._running = running

    def done(self) -> bool:
        self._running.discard(self)
        return True


def test_turn_in_flight_reads_a_copy_of_the_running_set() -> None:
    """A task leaving the set mid-read does not end the read in "Set changed size during
    iteration": `_wait_for_cascade_to_settle` polls from the test thread while the loop
    thread discards finished tasks (#1602).

    Killed by: src/uclone_x/ui/rooms.py :: self._running.get(room_id, set()).copy())
    Becomes: self._running.get(room_id, set()))
    """
    running: set[Any] = set()
    running.update({_SelfDiscardingTask(running), _SelfDiscardingTask(running)})
    stack = cast(RoomStack, type("_Stack", (), {"_running": {"room": running}})())

    assert RoomStack.turn_in_flight(stack, "room") is False
    assert running == set()


class TestCreateAndList:
    def test_a_created_room_is_seated_and_listed(self, client: TestClient) -> None:

        created = _create(client, title="Index tuning", agent_ids=["scout", "critic"])

        assert created.status_code == 201, created.text
        room_id = created.json()["room_id"]
        # A handle is seated by its clone's id (clone-data-scopes §4 step 3).
        seats = [seat_id_for("scout"), seat_id_for("critic")]
        assert all(seat.startswith("agt_") for seat in seats)
        assert [p["id"] for p in created.json()["participants"]] == ["user", *seats]

        listed = client.get("/api/rooms").json()["rooms"]
        assert [r["room_id"] for r in listed] == [room_id]
        assert listed[0]["title"] == "Index tuning"
        assert listed[0]["agent_ids"] == seats
        assert listed[0]["human_ids"] == ["user"]

    def test_a_room_that_cannot_be_seated_is_not_left_behind(self, client: TestClient) -> None:
        """A room is written once, seated, so a refused seat leaves nothing (#1885 item 3).

        Killed by: src/uclone_x/room/service.py ::
            state = self._seated(state, participant_id, kind=kind)
        Becomes: state = self._store.save(self._seated(state, participant_id, kind=kind))
        """

        # The second agent's id is one the session derivation cannot carry, so the roster
        # refuses it -- after the human and the first agent were seated.
        refused = _create(client, agent_ids=["scout", "bad__id"])

        assert refused.status_code == 400, refused.text
        assert client.get("/api/rooms").json()["rooms"] == [], (
            "a room nobody could be seated in survived the failure that created it"
        )

    def test_a_blank_title_is_refused_with_its_reason(self, client: TestClient) -> None:

        refused = _create(client, title="   ")

        assert refused.status_code == 400
        assert "needs a title" in refused.json()["detail"]

    def test_a_starving_policy_is_refused_with_its_reason(self, client: TestClient) -> None:
        """The Core explains which two knobs disagree; the route must not flatten that."""

        refused = _create(
            client,
            policy={"max_agent_turns_per_human_message": 9, "transcript_window": 3},
        )

        assert refused.status_code == 400
        assert "transcript_window" in refused.json()["detail"]

    def test_an_unknown_room_is_a_404(self, client: TestClient) -> None:

        assert client.get("/api/rooms/room_missing").status_code == 404
        assert client.delete("/api/rooms/room_missing").status_code == 404

    def test_a_missing_room_refusal_names_its_cause_and_its_remedy(
        self, client: TestClient
    ) -> None:
        """The head shows this `detail` verbatim when a delete or rename finds nothing (#1058).

        It used to say only that the room was missing. A person who pressed Delete on a
        row a second window had already removed was told the fact and not what to do.
        """
        for refused in (
            client.delete("/api/rooms/room_missing"),
            client.patch("/api/rooms/room_missing", json={"title": "Index tuning"}),
        ):
            detail = refused.json()["detail"]
            assert "room_missing" in detail
            assert "deleted" in detail
            assert "list" in detail.lower()


class TestRename:
    def test_a_room_can_be_renamed(self, client: TestClient) -> None:
        room_id = _create(client, title="Check the index on the users table").json()["room_id"]

        renamed = client.patch(f"/api/rooms/{room_id}", json={"title": "Index tuning"})

        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["title"] == "Index tuning"
        assert client.get("/api/rooms").json()["rooms"][0]["title"] == "Index tuning"

    def test_a_rename_carries_the_same_refusals_as_creation(self, client: TestClient) -> None:
        room_id = _create(client).json()["room_id"]

        refused = client.patch(f"/api/rooms/{room_id}", json={"title": "two\nlines"})

        assert refused.status_code == 400
        assert "newlines" in refused.json()["detail"]


def _assert_plain_detail(detail: str, room_file: Path) -> None:
    """What the head shows as the Core's own words: no location, class name or parser output.

    Every check here is one the parser's own text fails -- the field dump carries a dotted
    field path and pydantic's words, the decoder's complaint a line and column -- so none
    of them passes because the text was replaced by something empty.
    """
    assert detail.strip(), "an empty detail says nothing, and every check below would pass"
    assert str(room_file.parent) not in detail and room_file.name not in detail, detail
    assert "/" not in detail and "\\" not in detail, detail
    assert not re.search(r"[A-Za-z]*(Error|Exception)\b", detail), detail
    assert not re.search(r"\b\d{3} [A-Z]", detail), detail
    assert not re.search(r"\w+\.\d+\.\w+", detail), f"a field path: {detail!r}"
    for parser_word in (
        "validation error",
        "Input should be",
        "Invalid JSON",
        "json_invalid",
        "line ",
        "column",
        "EOF",
        "pydantic",
        "RoomState",
    ):
        assert parser_word not in detail, (parser_word, detail)


class TestARoomWhoseRecordWillNotLoad:
    """A stored record this build cannot read is a fault, and is answered as one (#1411).

    The store used to let the parser's `ValidationError` out of `load`, and the route
    translator answers that class as the caller's malformed request: a 400 whose `detail`
    was the field dump. The head shows a `detail` as the Core's own words, so the reader
    was shown `transcript.0.kind: Input should be ...` for a conversation they only
    clicked on. The record is broken the two ways a real one is: cut short mid-write or
    by hand, and written by a build whose shape this one no longer admits.
    """

    @staticmethod
    def _break(client: TestClient, how: str) -> tuple[str, Path]:
        room_id = _create(client, title="Kept").json()["room_id"]
        stack = cast(Any, client.app).state.room_stack
        room_file: Path = stack.store.room_path(room_id)
        text = room_file.read_text(encoding="utf-8")
        if how == "cut short":
            room_file.write_text(text[: len(text) // 2], encoding="utf-8")
        else:
            raw = json.loads(text)
            raw["transcript"] = [{"seq": 1, "sender_id": "user", "kind": "shout"}]
            room_file.write_text(json.dumps(raw), encoding="utf-8")
        return room_id, room_file

    @pytest.mark.parametrize("how", ["cut short", "an older shape"])
    def test_opening_it_says_so_plainly_and_logs_the_cause(
        self, answering_client: TestClient, caplog: pytest.LogCaptureFixture, how: str
    ) -> None:
        """Killed by: src/uclone_x/ui/rooms.py :: if isinstance(exc, UnreadableRoomRecordError):
        Becomes: if False:
        Killed by: src/uclone_x/room/store.py :: raise UnreadableRoomRecordError(room_id) from rewritten
        Becomes: raise
        """
        room_id, room_file = self._break(answering_client, how)

        with caplog.at_level(logging.ERROR, logger=rooms_module.logger.name):
            opened = answering_client.get(f"/api/rooms/{room_id}")

        assert opened.status_code == 500, opened.text
        detail = opened.json()["detail"]
        assert detail == rooms_module.UNREADABLE_ROOM_DETAIL
        _assert_plain_detail(detail, room_file)
        logged = [r for r in caplog.records if r.exc_info and room_id in r.getMessage()]
        assert logged, "the parser's cause must reach the log, since the answer leaves it out"
        exc_info = logged[0].exc_info
        assert exc_info is not None and exc_info[1] is not None
        assert isinstance(exc_info[1].__cause__, ValidationError), exc_info[1].__cause__

    def test_renaming_it_is_answered_the_same_way(self, answering_client: TestClient) -> None:
        room_id, room_file = self._break(answering_client, "an older shape")

        renamed = answering_client.patch(f"/api/rooms/{room_id}", json={"title": "New"})

        assert renamed.status_code == 500, renamed.text
        _assert_plain_detail(renamed.json()["detail"], room_file)

    def test_the_listing_still_shows_the_other_conversations(
        self, answering_client: TestClient
    ) -> None:
        self._break(answering_client, "cut short")
        readable = _create(answering_client, title="Readable").json()["room_id"]

        listed = answering_client.get("/api/rooms")

        assert listed.status_code == 200, listed.text
        assert [r["room_id"] for r in listed.json()["rooms"]] == [readable]

    @pytest.mark.parametrize("how", ["cut short", "an older shape"])
    def test_the_listing_names_it_as_unreadable(
        self, answering_client: TestClient, how: str
    ) -> None:
        """It is left out of `rooms` and named in `unreadable`, so it does not vanish (#1440).

        Killed by: src/uclone_x/room/service.py :: unreadable.append(room_id)
        Becomes: pass
        """
        room_id, _ = self._break(answering_client, how)
        readable = _create(answering_client, title="Readable").json()["room_id"]

        listed = answering_client.get("/api/rooms")

        assert listed.status_code == 200, listed.text
        assert [r["room_id"] for r in listed.json()["rooms"]] == [readable]
        assert listed.json()["unreadable"] == [room_id]

    def test_a_listing_with_nothing_unreadable_says_so(self, client: TestClient) -> None:
        _create(client)

        assert client.get("/api/rooms").json()["unreadable"] == []

    @pytest.mark.parametrize("how", ["cut short", "an older shape"])
    def test_it_can_be_deleted_without_being_loaded(
        self, answering_client: TestClient, how: str
    ) -> None:
        """Delete used to load the record first, so it answered 500 and the file stayed (#1440).

        Killed by: src/uclone_x/ui/rooms.py :: except UnreadableRoomRecordError:  # deleted anyway (#1440)
        Becomes: except KeyError:
        """
        room_id, room_file = self._break(answering_client, how)

        deleted = answering_client.delete(f"/api/rooms/{room_id}")

        assert deleted.status_code == 204, deleted.text
        assert not room_file.exists()
        assert answering_client.get("/api/rooms").json()["unreadable"] == []

    def test_deleting_a_room_that_is_not_there_is_still_refused(
        self, answering_client: TestClient
    ) -> None:
        """The unreadable case is the only one let through: a missing room still answers 404."""
        gone = answering_client.delete("/api/rooms/room_never_made")

        assert gone.status_code == 404, gone.text


class TestParticipants:
    def test_an_agent_can_be_added_to_a_running_conversation(self, client: TestClient) -> None:
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]

        added = client.post(f"/api/rooms/{room_id}/participants", json={"agent_id": "dba"})

        assert added.status_code == 200, added.text
        assert [p["id"] for p in added.json()["participants"]] == [
            "user",
            seat_id_for("scout"),
            seat_id_for("dba"),
        ]

    def test_a_second_human_is_refused_and_says_why(self, client: TestClient) -> None:
        room_id = _create(client).json()["room_id"]

        refused = client.post(
            f"/api/rooms/{room_id}/participants", json={"agent_id": "bob", "kind": "human"}
        )

        assert refused.status_code == 409
        assert "human" in refused.json()["detail"].lower()

    def test_an_agent_can_leave(self, client: TestClient) -> None:
        room_id = _create(client, agent_ids=["scout", "critic"]).json()["room_id"]

        # By handle: the route resolves it to the seat, which is keyed by the clone's id.
        left = client.delete(f"/api/rooms/{room_id}/participants/critic")

        assert left.status_code == 200, left.text
        assert [p["id"] for p in left.json()["participants"]] == ["user", seat_id_for("scout")]


class TestSending:
    def test_sending_answers_before_the_turns_have_run(self, client: TestClient) -> None:
        """202 and not the final state: `post()` spans every turn of the cascade.

        Killed by: src/uclone_x/ui/rooms.py :: JSONResponse(status_code=202
        Becomes: JSONResponse(status_code=200
        """
        room_id = _create(client).json()["room_id"]

        sent = client.post(f"/api/rooms/{room_id}/messages", json={"content": "check the index"})

        # Rows 1 and 2 are the two joins, so the message is row 3 and the reply is row 4.
        assert sent.status_code == 202, sent.text
        assert sent.json() == {"room_id": room_id, "seq": 3}

        landed = _wait_for_transcript(client, room_id, rows=4)
        scout = seat_id_for("scout")
        assert [m["sender_id"] for m in landed["transcript"]] == ["user", scout, "user", scout]
        assert landed["transcript"][3]["content"]

    def test_the_human_utterance_is_recorded_before_the_route_answers(
        self, client: TestClient
    ) -> None:
        """A message that is only in flight is a message the user cannot see they sent."""
        room_id = _create(client).json()["room_id"]

        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})

        state = client.get(f"/api/rooms/{room_id}").json()
        assert state["transcript"][2]["sender_id"] == "user"
        assert state["transcript"][2]["content"] == "hello"

    def test_sending_to_an_unknown_room_is_a_404(self, client: TestClient) -> None:

        assert (
            client.post("/api/rooms/room_missing/messages", json={"content": "x"}).status_code
            == 404
        )

    def test_typing_and_stop_are_reachable(self, client: TestClient) -> None:
        """Human interjection priority is an invariant; without controls it is a claim."""
        room_id = _create(client).json()["room_id"]

        assert client.post(f"/api/rooms/{room_id}/typing").status_code == 204
        assert client.post(f"/api/rooms/{room_id}/stop").status_code == 200


class TestRoomSessionsAreNotConversations:
    def test_a_rooms_agent_sessions_are_kept_out_of_the_session_list(
        self, client: TestClient
    ) -> None:
        """Otherwise one room adds one phantom chat per seated agent.

        The session is written straight into the store rather than produced by running a
        room, and deliberately so. Until the orchestrator wrote each seat's session after
        its turn, a room's agents persisted nothing, and a test that ran a room and then
        asserted the listing was empty passed against a filter that did nothing at all --
        which is exactly what it did before a mutation run said so. What is under test is
        the filter, and the filter needs a record to filter that does not depend on
        whether a room run writes one.

        Killed by: src/uclone_x/ui/app.py :: if sid.startswith(ROOM_SESSION_PREFIX)
        Becomes: if False
        """
        from uclone_x.agent.session import SessionState, SessionStore
        from uclone_x.ui.app import AgentSessionManager

        manager = cast(AgentSessionManager, cast(Any, client.app).state.session_manager)
        store: SessionStore = manager.core_store
        store.save(SessionState(session_id="sess_room__room_x__scout", agent_id="scout"))
        store.save(SessionState(session_id="sess_room__room_x__critic", agent_id="critic"))
        store.save(SessionState(session_id="sess_ordinary_chat", agent_id="champion"))

        listed = [s["session_id"] for s in client.get("/api/sessions").json()["sessions"]]

        assert listed == ["sess_ordinary_chat"], listed


class TestTheOrchestratorIsStableAndPerRoom:
    """Two cache invariants that were argued for in prose and held by nothing.

    An audit dropped `reset_orchestrator`'s body and replaced the per-room key with a
    constant, and the suite stayed green through both.
    """

    def test_a_conversation_keeps_one_orchestrator_across_a_roster_edit(
        self, client: TestClient
    ) -> None:
        """The floor lives on the instance, so replacing it re-opens the overlap.

        `reset_orchestrator` used to run on every roster edit, on the argument that the
        selector chain depends on the roster. It does not: `SoleAgentSelector` reads
        `request.participants` live and abstains as soon as a second agent joins, and the
        chain is built from the *policy* alone. Dropping the instance therefore bought
        nothing and cost the room its floor lock and its interrupt registry -- a Stop
        served by the replacement found no turn and said it had stopped one.

        Killed by: src/uclone_x/ui/rooms.py :: existing = self._rooms.get(state.room_id)
        Becomes: existing = None
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        stack = cast(Any, client.app).state.room_stack
        before = stack.orchestrator(stack.service.get(room_id))

        client.post(f"/api/rooms/{room_id}/participants", json={"agent_id": "dba"})

        assert stack.orchestrator(stack.service.get(room_id)) is before

    def test_two_conversations_do_not_share_one_orchestrator(self, client: TestClient) -> None:
        """A shared instance routes every room by whichever policy was read first.

        The mutation is on the *read*, and the first draft of this declaration had it on
        the write -- where it does not kill: a shared write key leaves the per-room read
        missing, so a fresh instance is built every time and the two rooms still differ.
        The ratchet said so.

        Killed by: src/uclone_x/ui/rooms.py :: existing = self._rooms.get(state.room_id)
        Becomes: existing = next(iter(self._rooms.values()), None)
        """
        first = _create(client, title="One").json()["room_id"]
        second = _create(client, title="Two").json()["room_id"]
        stack = cast(Any, client.app).state.room_stack

        assert stack.orchestrator(stack.service.get(first)) is not stack.orchestrator(
            stack.service.get(second)
        )


class TestStopAndTypingDoSomething:
    """Both routes answered 200/204 with their bodies replaced by `pass`.

    The first version of these tests asserted the status code, which is route existence
    and not a control. "An invariant without a control is a claim" was the docstring; the
    test made the control itself one.
    """

    def test_stop_signals_the_interrupt_the_room_will_read(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/rooms.py :: await self.orchestrator(state).interrupt(room_id)
        Becomes: pass
        """
        room_id = _create(client).json()["room_id"]
        stack = cast(Any, client.app).state.room_stack

        assert client.post(f"/api/rooms/{room_id}/stop").status_code == 200

        orch = stack.orchestrator(stack.service.get(room_id))
        assert room_id in orch._interrupted_rooms, (  # noqa: SLF001
            "Stop answered 200 without signalling anything the turn loop reads"
        )

    def test_stop_refuses_invalid_room_id_with_400(self, answering_client: TestClient) -> None:
        """POST /api/rooms/bad..id/stop returns 400 (bad request), not 500 (#1996)."""
        refused = answering_client.post("/api/rooms/bad..id/stop")
        assert refused.status_code == 400

    def test_typing_advances_the_activity_mark_a_hesitating_room_reads(
        self, client: TestClient
    ) -> None:
        """Killed by: src/uclone_x/ui/rooms.py :: await stack.orchestrator(state).note_human_activity(
        Becomes: await _ignore(
        """
        room_id = _create(client).json()["room_id"]
        before = client.get(f"/api/rooms/{room_id}").json()["turn_state"]["last_activity_ts"]

        assert client.post(f"/api/rooms/{room_id}/typing").status_code == 204

        after = client.get(f"/api/rooms/{room_id}").json()["turn_state"]["last_activity_ts"]
        assert after > before, "typing answered 204 without recording that anybody typed"


class TestRetryAndDelete:
    def test_retry_refuses_with_its_reason_when_nothing_failed(self, client: TestClient) -> None:
        room_id = _create(client).json()["room_id"]

        refused = client.post(f"/api/rooms/{room_id}/retry")

        assert refused.status_code == 409, refused.text
        assert refused.json()["detail"]

    def test_deleting_a_conversation_removes_it_from_the_list(self, client: TestClient) -> None:
        room_id = _create(client).json()["room_id"]

        assert client.delete(f"/api/rooms/{room_id}").status_code == 204

        assert client.get("/api/rooms").json()["rooms"] == []
        assert client.get(f"/api/rooms/{room_id}").status_code == 404

    def test_deleting_a_room_removes_seat_sessions_from_session_store(
        self, client: TestClient
    ) -> None:
        from uclone_x.agent.session import SessionState

        created = _create(client, agent_ids=["scout"]).json()
        room_id = created["room_id"]
        agent_participant = next(p for p in created["participants"] if p["kind"] == "agent")
        seat_session_id = agent_participant["session_id"]

        stack = _stack(client)
        core_store = stack._session_mgr.core_store  # pyright: ignore[reportPrivateUsage]
        core_store.save(SessionState(session_id=seat_session_id, agent_id=agent_participant["id"]))
        session_file = core_store.session_path(seat_session_id)
        assert session_file.exists()

        res = client.delete(f"/api/rooms/{room_id}")
        assert res.status_code == 204, res.text
        assert not session_file.exists()


class TestTheHeadDoesNotSubstituteValuesTheCoreWouldRefuse:
    """Every one of these answered 2xx, with the head supplying what the caller did not.

    `str(req.get("title", ""))` turns `null` into the string `"None"`, which is the one
    thing `RoomService.create` refuses a blank title in order to prevent. The pattern
    repeated for `content`, `kind` and the human's id.
    """

    def test_a_null_title_is_refused_rather_than_stringified(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/rooms.py :: title = _required_str(req, "title", allow_blank=True)
        Becomes: title = str(req.get("title", ""))
        """
        refused = client.post("/api/rooms", json={"title": None, "agent_ids": ["scout"]})

        assert refused.status_code == 400, refused.text
        assert client.get("/api/rooms").json()["rooms"] == [], (
            "a conversation called 'None' was created"
        )

    def test_a_non_string_title_is_refused(self, client: TestClient) -> None:
        refused = client.post("/api/rooms", json={"title": {"a": 1}, "agent_ids": []})

        assert refused.status_code == 400

    def test_an_empty_message_does_not_spend_a_turn(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/rooms.py :: content = _required_str(req, "content")
        Becomes: content = str(req.get("content", ""))
        """
        room_id = _create(client).json()["room_id"]

        refused = client.post(f"/api/rooms/{room_id}/messages", json={})

        assert refused.status_code == 400, refused.text
        state = client.get(f"/api/rooms/{room_id}").json()
        assert len(state["transcript"]) == 2, "an empty message was recorded and answered"

    def test_a_non_string_message_is_refused_rather_than_repr_ed(self, client: TestClient) -> None:
        room_id = _create(client).json()["room_id"]

        refused = client.post(f"/api/rooms/{room_id}/messages", json={"content": {"x": 1}})

        assert refused.status_code == 400
        assert "{'x': 1}" not in client.get(f"/api/rooms/{room_id}").text

    def test_an_unrecognised_kind_is_refused_rather_than_defaulted_to_agent(
        self, client: TestClient
    ) -> None:
        """A person modelled as an agent gets a derived session and never speaks.

        Killed by: src/uclone_x/ui/rooms.py :: raise HTTPException(status_code=400, detail=_KIND_REFUSAL)
        Becomes: pass
        """
        room_id = _create(client).json()["room_id"]

        refused = client.post(
            f"/api/rooms/{room_id}/participants", json={"agent_id": "kenny", "kind": "Human"}
        )

        assert refused.status_code == 400, refused.text
        assert "kind" in refused.json()["detail"]

    def test_a_participant_id_that_cannot_be_addressed_in_a_url_is_refused(
        self, client: TestClient
    ) -> None:
        """Otherwise the seat is filled by somebody no route can remove.

        A room seats one human, and removal is the only way to free the seat, so a human
        id carrying a slash made that room permanently unusable.

        Killed by: src/uclone_x/ui/rooms.py :: _refuse_an_unaddressable_id(human_id)
        Becomes: pass
        """
        refused = client.post("/api/rooms", json={"title": "T", "human_id": "a/b", "agent_ids": []})

        assert refused.status_code == 400, refused.text
        assert client.get("/api/rooms").json()["rooms"] == []

    def test_typing_in_a_conversation_with_no_human_says_so(self, client: TestClient) -> None:
        """`_sole_human` answered with the name `"user"` for a room that seats nobody.

        Killed by: src/uclone_x/ui/rooms.py :: raise HTTPException(status_code=409, detail=_NO_HUMAN_REFUSAL)
        Becomes: pass
        """
        room_id = _create(client).json()["room_id"]
        client.delete(f"/api/rooms/{room_id}/participants/user")

        refused = client.post(f"/api/rooms/{room_id}/typing")

        assert refused.status_code == 409, refused.text


class TestACascadeFailureIsAnnouncedWithoutLeaking:
    """The room topic is forwarded to every subscriber and rendered as copy.

    The announcement carried `f"{type(exc).__name__}: {exc}"`, so a fault put a Python
    class name and whatever the exception interpolated -- a store path, for instance -- in
    front of a non-expert reader. A Core refusal is written *for* that reader and is
    passed through; a fault is not.
    """

    def test_a_core_refusal_keeps_its_own_sentence(self) -> None:
        from uclone_x.errors import UnknownRoomParticipantError
        from uclone_x.ui.rooms import reader_facing_reason

        refusal = UnknownRoomParticipantError("'phantm' is not a participant of room 'r1'")

        assert reader_facing_reason(refusal) == str(refusal)

    def test_an_unexpected_fault_does_not_reach_the_reader(self) -> None:
        """Killed by: src/uclone_x/ui/rooms.py :: return "This conversation stopped because of a problem in the agent runtime."
        Becomes: return str(exc)
        """
        from uclone_x.ui.rooms import reader_facing_reason

        fault = OSError("[Errno 13] Permission denied: '/Users/someone/.uclone-x/rooms/r1.json'")

        reason = reader_facing_reason(fault)

        assert "Permission denied" not in reason
        assert "/Users/" not in reason
        assert reason.strip()

    @pytest.mark.asyncio
    async def test_a_failure_notice_that_could_not_be_published_is_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A lost failure notice is itself a failure, and is logged as one (#929).

        The announcement is how the surface learns that a conversation stopped. When it
        cannot be published the head shows a speaker that writes forever, and the only
        record of why was a `debug` line nobody runs with.

        Killed by: src/uclone_x/ui/rooms.py :: logger.warning("Room %s could not announce its failure", room_id, exc_info=True)
        Becomes: logger.debug("Room %s could not announce its failure", room_id, exc_info=True)
        """
        from types import SimpleNamespace

        from uclone_x.errors import UnknownRoomParticipantError
        from uclone_x.ui.rooms import RoomStack

        attempted = asyncio.Event()

        class DownBus:
            def register_publisher(self, *args: Any, **kwargs: Any) -> Any:
                attempted.set()
                raise RuntimeError("the bus is down")

        def ignore_llm(_: object) -> None:
            return None

        stack = RoomStack(
            cast(
                Any,
                SimpleNamespace(
                    storage_dir=tmp_path,
                    workspace_dir=tmp_path / "workspace",
                    bus=DownBus(),
                    on_models_changed=ignore_llm,
                ),
            )
        )

        async def refused() -> None:
            raise UnknownRoomParticipantError("'phantm' is not a participant of room 'r1'")

        with caplog.at_level(logging.DEBUG, logger="uclone_x.ui.rooms"):
            stack.drive("r1", refused)
            # Set inside the failing call; the log line follows in the same step.
            await asyncio.wait_for(attempted.wait(), 2.0)
            await stack.close()

        lost = [r for r in caplog.records if "could not announce its failure" in r.getMessage()]
        assert [r.levelno for r in lost] == [logging.WARNING], [
            (r.levelname, r.getMessage()) for r in lost
        ]
        assert lost[0].exc_info is not None, "the reason the notice was lost must be logged too"


class TestTheCoreRefusalSentenceBelongsToTheReader:
    """A Core refusal reaches the reader either through reader_facing_reason or _http_error.

    Every reachable RoomError must be written for a non-expert reader, without developer
    jargon, path separators, or exception class names (#1438).
    """

    def test_all_reachable_room_errors_produce_plain_reader_facing_reasons_and_route_details(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/room/store.py :: f"The conversation {state.room_id!r} could not be saved because it was changed by another action. Refresh and try again."
        Becomes: f"The conversation {state.room_id!r} could not be saved: held revision conflict with orchestrator."
        Killed by: src/uclone_x/room/store.py :: "The conversation identifier is not valid: it must not contain path separators."
        Becomes: "The conversation identifier is not valid: path/traversal error."
        Killed by: src/uclone_x/ui/rooms.py :: return "The next speaker could not be chosen. Try sending your message again."
        Becomes: return "SpeakerSelectionError: could not choose next speaker"
        """
        from uclone_x.errors import (
            HeadRoomWriteError,
            NothingToRetryError,
            ParticipantNotResolvableError,
            RoomAlreadyExistsError,
            RoomError,
            RoomIdError,
            RoomNotFoundError,
            RoomWorkspaceRefusedError,
            SecondHumanInRoomError,
            SpeakerSelectionError,
            StaleRoomWriteError,
            TurnNotLandedError,
            TurnNotStartedError,
            UnknownRoomParticipantError,
        )
        from uclone_x.room.turn_summary import TurnNotFoundError
        from uclone_x.ui.rooms import (
            _http_error,  # pyright: ignore[reportPrivateUsage]
            reader_facing_reason,
        )

        store = RoomStore(tmp_path / "rooms")
        state = RoomState(room_id="r1", revision=0, participants=())
        store.save(state)
        stale_err: StaleRoomWriteError | None = None
        room_id_err: RoomIdError | None = None
        try:
            store.save(state)
        except StaleRoomWriteError as exc:
            stale_err = exc
        try:
            store.room_path("..")
        except RoomIdError as exc:
            room_id_err = exc

        assert stale_err is not None
        assert room_id_err is not None

        sample_errors: list[RoomError] = [
            stale_err,
            room_id_err,
            RoomNotFoundError("No room 'r1' in the store."),
            RoomAlreadyExistsError(
                "Room 'r1' already exists; refusing to overwrite its conversation."
            ),
            SecondHumanInRoomError(
                "Room 'r1' already seats a human, so 'user2' cannot join: a room serves one human."
            ),
            NothingToRetryError(
                "The last thing said in room 'r1' did not fail. A retry re-runs a failed turn; there is none to re-run."
            ),
            HeadRoomWriteError("This conversation is managed by another process."),
            TurnNotStartedError(
                "The turn was not started, because the conversation could not save that it was starting."
            ),
            TurnNotLandedError(
                "The reply could not be saved to this conversation, so it was not kept."
            ),
            UnknownRoomParticipantError("'phantm' is not a participant of room 'r1'"),
            ParticipantNotResolvableError("No live agent can be produced for 'bot'"),
            SpeakerSelectionError("The next speaker could not be chosen."),
            TurnNotFoundError("r1", 1),
            RoomWorkspaceRefusedError(
                "That is not an existing folder given by its full path, so the "
                "conversation's workspace was not changed.",
                "not_a_folder",
            ),
        ]

        assert {type(err) for err in sample_errors} == set(RoomError.__subclasses__())

        for err in sample_errors:
            reason = reader_facing_reason(err)
            detail = _http_error(err).detail
            assert isinstance(detail, str), f"detail is not a str for {type(err).__name__}"
            for text in (reason, detail):
                assert text, f"empty text for {type(err).__name__}"
                assert "/" not in text and "\\" not in text, f"separator in {text!r}"
                assert not re.search(r"[A-Za-z]*(Error|Exception)\b", text), (
                    f"error class in {text!r}"
                )
                assert "Errno" not in text, f"Errno in {text!r}"
                assert "held revision" not in text, f"held revision in {text!r}"
                assert "orchestrator" not in text, f"orchestrator in {text!r}"


class TestTheDesktopSaysNobodyAnswersApprovals:
    """Owner decision 2026-09-26: the desktop app does not ask for approval mid-conversation.

    Nothing in the desktop app answers an approval request (a person approves story changes
    in the story's view instead, #1560), so every agent it builds must say so, and a call
    that needs a person's approval is then refused at once rather than after the timeout.
    """

    def test_a_chat_agent_and_a_room_seat_are_built_saying_so(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/ui/app.py :: approvals_answered=False,
        Becomes: approvals_answered=True,

        One declaration covers both since #1731: the chat agent and the room seat are
        built from the one `AgentSessionManager.app_scope()`.
        """
        from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
        from uclone_x.ui.app import AgentSessionManager

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))
            make_clones(*_TEST_CLONES)
            mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=MockLLMConnector())
            chat = app_clone(mgr, "writer")
            stack = RoomStack(mgr)
            state = stack.service.create(title="Chapter one")
            stack.service.add_participant(
                state.room_id, participant_id="writer", kind=ParticipantKind.AGENT
            )
            seated = stack.store.load(state.room_id)
            assert seated is not None
            resolver = stack.orchestrator(seated)._resolver  # pyright: ignore[reportPrivateUsage]
            participant = next(p for p in seated.participants if p.id == "writer")
            seat = cast("BaseAgent", asyncio.run(resolver.resolve(participant)))

        assert chat.approvals_answered is False
        assert seat.approvals_answered is False


class TestRoomMemoryIsHeldPerAgent:
    """One store per agent id, and exactly one map of them in the process.

    `RoomStack.orchestrator` builds a `RoomAgentResolver` per room and caches it by room
    id, so a store created inside the resolver would be created again for the same
    participant seated in a second room. But holding the map on `RoomStack` is wrong for
    the same reason one level up: `AgentSessionManager` already keys stores by agent id
    for the chat surface, and the ids collide by design -- `champion`, `scout` and
    `critic` are both chat agents and the canonical room seats. `CrossSessionMemory.save()`
    rewrites the whole document, so two objects over one file are two whole-document
    writers and each silently drops what the other recorded (P6).
    """

    def test_a_room_seat_and_a_chat_session_share_one_store(self, tmp_path: Path) -> None:
        """The room must reach the session manager's map, not keep one of its own.

        Killed by: src/uclone_x/ui/app.py :: existing = self._agent_memories.get(key)
        Becomes: existing = None
        """
        from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
        from uclone_x.ui.app import AgentSessionManager
        from uclone_x.ui.rooms import RoomStack

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))
            make_clones(*_TEST_CLONES)
            mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=MockLLMConnector())
            stack = RoomStack(mgr)
            state = stack.service.create(title="Design review")
            # A seat is keyed by its clone's id; the service seats whatever id it is given.
            champion = seat_id_for("champion")
            stack.service.add_participant(
                state.room_id, participant_id=champion, kind=ParticipantKind.AGENT
            )
            seated = stack.store.load(state.room_id)
            assert seated is not None
            resolver = stack.orchestrator(seated)._resolver  # pyright: ignore[reportPrivateUsage]
            participant = next(p for p in seated.participants if p.id == champion)
            agent = cast("BaseAgent", asyncio.run(resolver.resolve(participant)))
            by_handle = mgr.memory_for("champion")

        # The same object the chat surface would hand `champion`, not merely one over the
        # same path: two objects would each hold the whole fact set and clobber the other.
        assert agent.memory is mgr.memory_for(champion)
        assert by_handle is agent.memory, "a handle reaches its clone's one store"

    def test_two_seats_in_one_room_do_not_share_a_store(self, tmp_path: Path) -> None:
        """Sharing one store across seats is the opposite failure: recollection bleed.

        Killed by: src/uclone_x/agent/clone_builder.py :: host = dataclasses.replace(host, memory=app.memory_for(clone_id))
        Becomes: host = host
        """
        from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
        from uclone_x.ui.app import AgentSessionManager
        from uclone_x.ui.rooms import RoomStack

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))
            make_clones(*_TEST_CLONES)
            mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=MockLLMConnector())
            stack = RoomStack(mgr)
            state = stack.service.create(title="Design review")
            # Seats are keyed by clone id; the service seats whatever id it is given.
            alpha, beta = seat_id_for("alpha"), seat_id_for("beta")
            for participant_id in (alpha, beta):
                stack.service.add_participant(
                    state.room_id, participant_id=participant_id, kind=ParticipantKind.AGENT
                )
            seated = stack.store.load(state.room_id)
            assert seated is not None
            resolver = stack.orchestrator(seated)._resolver  # pyright: ignore[reportPrivateUsage]
            agents = {
                p.id: cast("BaseAgent", asyncio.run(resolver.resolve(p)))
                for p in seated.participants
                if p.id in {alpha, beta}
            }

        assert agents[alpha].memory is not None
        assert agents[alpha].memory is not agents[beta].memory
        assert agents[alpha].memory is mgr.memory_for(alpha)
        assert agents[beta].memory is mgr.memory_for(beta)

    def test_a_seated_agent_can_record_into_its_own_store(self, tmp_path: Path) -> None:
        """The wiring, end to end: the head's resolver hands each seat a store at all.

        Without `memory_factory` the room's agents are composed with no memory at all, so
        `record_memory_fact` is neither advertised to them nor resolvable — the defect the
        false "wired by every head" comment on `HostDependencies.memory` concealed.

        Killed by: src/uclone_x/ui/app.py :: memory_for=self.memory_for,
        Becomes: memory_for=None,
        """
        from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
        from uclone_x.ui.app import AgentSessionManager
        from uclone_x.ui.rooms import RoomStack

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))
            make_clones(*_TEST_CLONES)
            mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", llm=MockLLMConnector())
            stack = RoomStack(mgr)
            state = stack.service.create(title="Design review")
            stack.service.add_participant(
                state.room_id, participant_id="alpha", kind=ParticipantKind.AGENT
            )
            seated = stack.store.load(state.room_id)
            assert seated is not None
            resolver = stack.orchestrator(seated)._resolver  # pyright: ignore[reportPrivateUsage]
            participant = next(p for p in seated.participants if p.id == "alpha")
            agent = cast("BaseAgent", asyncio.run(resolver.resolve(participant)))
            record = asyncio.run(
                agent.execute_tool_call(
                    "record_memory_fact",
                    {"subject": "Design review", "predicate": "held in", "object_value": "a room"},
                )
            )
            # Read back inside the context: with `MEMORY_STORAGE_DIR` restored, a
            # mutant would resolve this against the developer's own home instead,
            # where a stale `alpha.json` makes it pass.
            recorded = [fact.subject for fact in mgr.memory_for("alpha").list_facts()]

        assert record.status == "success", record.error
        assert recorded == ["Design review"]
        assert isinstance(seated.policy, RoomPolicy)


# --------------------------------------------------------------------------------------
# The history controls the retirement of `/api/turn` moves onto the conversation (#1208)
# --------------------------------------------------------------------------------------


def _seat(room: dict[str, Any], participant_id: str) -> dict[str, Any]:
    return next(p for p in room["participants"] if p["id"] == participant_id)


def _live_seat_agent(client: TestClient, room_id: str, participant_id: str) -> BaseAgent | None:
    """The agent the room's own resolver built for a seat, reached the way a route reaches it."""
    stack = cast(Any, client.app).state.room_stack
    room = client.get(f"/api/rooms/{room_id}").json()
    return cast(
        "BaseAgent | None", stack.live_agent(room_id, _seat(room, participant_id)["session_id"])
    )


def _always_answering(self: RoomStack, room_id: str) -> bool:
    """A stand-in for `RoomStack.turn_in_flight` that always reports a running cascade."""
    return True


def _endpoint(app: Any, path: str, method: str) -> Any:
    """The route's own coroutine function, to be awaited on the *test's* event loop.

    `TestClient` runs the app on a portal of its own, so a task the test schedules is not
    on the loop the handler runs on and cannot be used to ask whether that handler yielded.
    The two concurrency tests below need exactly that question answered, so they call the
    registered handler directly: it is the same object the ASGI app dispatches to, minus
    the transport.
    """
    for route in cast(list[Any], app.routes):
        if getattr(route, "path", None) == path and method in (
            getattr(route, "methods", None) or ()
        ):
            return route.endpoint
    raise AssertionError(f"no route {method} {path}")


def _seated_room(stack: RoomStack) -> str:
    """A room with one human and one agent, created through the service rather than HTTP.

    The concurrency tests below own their event loop and so cannot use the `client`
    fixture; `RoomService` is synchronous, which is what makes this the short way in.
    """
    room_id = stack.service.create("Index tuning").room_id
    stack.service.add_participant(room_id, "user", kind=ParticipantKind.HUMAN)
    stack.service.add_participant(room_id, "scout")
    return room_id


def _default_model(mgr: Any, kind: str, model: str, *, base_url: str | None = None) -> None:
    """Save one connection of ``kind`` and make ``model`` on it the default deep (§3.2)."""
    row: dict[str, str] = {"id": kind, "kind": kind}
    if base_url is not None:
        row["base_url"] = base_url
    if kind == "anthropic":
        row["key"] = "sk-ant-test"
    mgr.settings_file.write_text(
        json.dumps({"connections": [row], "default_models": {"deep": f"{kind}/{model}"}}),
        encoding="utf-8",
    )
    # The fixture's app was handed a mock for every clone that follows the default; this
    # readout is about the saved default, so the gateway answers from the file instead.
    mgr.gateway.set_default_binding(None)


class TestConversationContextReadout:
    def test_a_fresh_conversation_reports_a_seat_nobody_has_spoken_in(
        self, client: TestClient
    ) -> None:
        """Zero turns and `live: false` are different facts and are both stated (P6).

        `live` is not decoration: a seat with no agent built yet is answered from its stored
        record, which is behind whatever a previous process did not persist. A readout that
        hid which copy it read would present the two as one number.
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]

        body = client.get(f"/api/rooms/{room_id}/context").json()
        scout = seat_id_for("scout")  # a handle is seated by its clone's id

        assert body["seats"] == [
            {
                "participant_id": scout,
                "session_id": f"sess_room__{room_id}__{scout}",
                "active_turns": 0,
                "is_saturated": False,
                # `None`, not `0`: nothing has been booked against this session in this
                # process, which is a different fact from having spent nothing. The case
                # below shows the seat that really has spent nothing reporting `0`.
                "used_tokens": None,
                "cumulative_tokens": None,
                # No window either, and for a reason the surface states: this fixture's
                # provider publishes no context window and serves no daemon that could
                # report one. A figure here would have to have been invented.
                "max_context_tokens": None,
                "context_window_source": None,
                "live": False,
            }
        ]
        assert body["is_saturated"] is False
        assert body["saturation_threshold"] == 20

    def test_a_seat_with_no_booking_is_told_apart_from_one_that_spent_nothing(
        self, client: TestClient
    ) -> None:
        """P6, on the figure the composer draws beside Send.

        `TokenBudgetManager` holds one process's bookings. A conversation reopened after a
        restart has a full transcript and nothing booked against its session, and reporting
        that as `0` would describe a fresh conversation where there is an expensive one.
        The two are `None` and `0` here, which is why the route reads `get_budget` and not
        `get_summary` -- the summary flattens both to `0`.

        Killed by: src/uclone_x/ui/rooms.py :: booked = mgr.budget_tracker.get_budget(participant.session_id)
        Becomes: booked = mgr.budget_tracker._get_or_create_budget(participant.session_id)  # pyright: ignore[reportPrivateUsage]
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        session_id = f"sess_room__{room_id}__{seat_id_for('scout')}"

        assert client.get(f"/api/rooms/{room_id}/context").json()["seats"][0]["used_tokens"] is None

        tracker = cast(Any, client.app).state.room_stack.session_manager().budget_tracker
        tracker.record_usage(
            session_id,
            TokenUsage(provider="mock", input_tokens=120, output_tokens=33),
        )

        seat = client.get(f"/api/rooms/{room_id}/context").json()["seats"][0]
        assert seat["used_tokens"] == 153

    def test_the_human_seat_is_not_reported_as_a_context(self, client: TestClient) -> None:
        """A person has no session to saturate; listing one would invite clearing it.

        Through the API this holds for the weaker of the two reasons `seated_agents` gives:
        `RoomService` stamps a session id only on agents, so the human is filtered out by
        carrying none, and the kind check never has to fire. The case where it does is
        exercised directly below — stated here so this test is not read as covering it.
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]

        body = client.get(f"/api/rooms/{room_id}/context").json()

        assert [seat["participant_id"] for seat in body["seats"]] == [seat_id_for("scout")]

    def test_a_refusal_about_a_seats_record_arrives_in_the_cores_own_words(
        self, answering_client: TestClient
    ) -> None:
        """This readout was the one room route with no `except` at all (#1212).

        A seat nobody has spoken in is answered from its **stored** record, and
        `SessionStore.load` refuses a record that identifies a different session -- the
        #256 collision, which is a conflict the caller can act on and whose message names
        the two ids and the one file they fold onto. With no `except` here that refusal
        left the route as a bare 500: no status describing the conflict, no body, and
        nothing saying which record to move. A failure with no reason and no remedy is
        what P6 forbids, and it is what every other handler in the module already avoids
        by routing its Core call through `_http_error`.

        The record is planted rather than waited for. The fold #256 describes needs a
        case-insensitive filesystem to occur naturally, and the guard it trips is a
        comparison of the record's *contents* against the id asked for -- so writing one
        record whose `session_id` disagrees with its own path reproduces the refusal on
        any filesystem, which is also the only form that is deterministic in CI.

        Both halves are load-bearing and neither alone gives this answer: without the
        `except` the refusal never reaches a translator, and without the translator's
        branch it is translated into a 500 carrying a sentence of ours instead of the
        Core's.

        Killed by: src/uclone_x/ui/rooms.py :: except Exception as exc:  # the readout's one Core call (#1212)
        Becomes: except SystemExit as exc:  # narrowed past the refusal, as before #1212
        Killed by: src/uclone_x/ui/rooms.py :: if isinstance(exc, SessionIdCollisionError):
        Becomes: if False:
        """
        from uclone_x.agent.session import SessionState
        from uclone_x.ui.app import AgentSessionManager

        room_id = _create(answering_client, agent_ids=["scout"]).json()["room_id"]
        manager = cast(AgentSessionManager, cast(Any, answering_client.app).state.session_manager)
        scout = seat_id_for("scout")
        asked = f"sess_room__{room_id}__{scout}"
        squatter = asked.upper()
        record = manager.core_store.session_path(asked)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            SessionState(session_id=squatter, agent_id=scout).model_dump_json(),
            encoding="utf-8",
        )

        refused = answering_client.get(f"/api/rooms/{room_id}/context")

        assert refused.status_code == 409, refused.text
        detail = refused.json()["detail"]
        assert asked in detail and squatter in detail, (
            f"the refusal must name both ids, since the whole defect is that nothing else "
            f"tells them apart; got {detail!r}"
        )
        assert "identifies session" in detail, detail

    def test_a_hosted_model_reports_the_window_its_provider_publishes(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ring's denominator, and the fact that it says where it came from (P6).

        `used_tokens` is a count of something, and until this it was a count against
        nothing: the surface drew turns because no context window was reachable from this
        runtime. A hosted provider's window is a published figure its API enforces, so for
        those the table *is* the measurement and the seat reports it.

        The default model is written to the settings file the gateway reads
        (model-gateway §3.2); no seat has spoken, so the readout uses the default deep.

        Killed by: src/uclone_x/ui/rooms.py :: window_tokens, window_source = declared, "published"
        Becomes: window_tokens, window_source = declared, "loaded"
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        mgr = cast(Any, client.app).state.room_stack.session_manager()
        _default_model(mgr, "anthropic", "claude-3-5-sonnet-20241022")

        seat = client.get(f"/api/rooms/{room_id}/context").json()["seats"][0]

        assert seat["max_context_tokens"] == 200_000
        assert seat["context_window_source"] == "published"

    def test_a_model_nobody_publishes_a_window_for_reports_none(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unknown model has an unknown window, and is never given a typical one.

        This is the case a provider default would have swallowed. A window that is somewhat
        wrong is a ring drawn to the wrong fraction, and nothing on screen would show the
        reader it was wrong.

        Killed by: src/uclone_x/llm/context_window.py :: return found
        Becomes: return found if found is not None else 200_000
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        mgr = cast(Any, client.app).state.room_stack.session_manager()
        _default_model(mgr, "anthropic", "claude-9-unreleased")

        seat = client.get(f"/api/rooms/{room_id}/context").json()["seats"][0]

        assert seat["max_context_tokens"] is None
        assert seat["context_window_source"] is None

    def test_a_locally_served_model_reports_the_window_the_daemon_gave_it(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Read from the server, never from a table -- see `llm/context_window.py`.

        The number is whatever that daemon chose when it loaded that model, which is why
        the route asks it rather than declaring it. `refresh` is left to fail against a
        daemon that is not there; the observation seeded here is what a successful one
        would have left behind, and the route must read that store rather than the
        published table, which has no entry for Ollama at all.

        Killed by: src/uclone_x/ui/rooms.py :: observed = OLLAMA_CONTEXT_WINDOWS.get(llm_base_url, seat_model)
        Becomes: observed = None
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        mgr = cast(Any, client.app).state.room_stack.session_manager()
        # A port nothing listens on, so `refresh` is refused instantly and the only
        # figure in the store is the seeded observation this test is about.
        _default_model(mgr, "ollama", "qwen3:8b", base_url="http://127.0.0.1:1")
        store = OllamaContextWindows()
        store.remember("http://127.0.0.1:1", "qwen3:8b", 40_960)
        monkeypatch.setattr(rooms_module, "OLLAMA_CONTEXT_WINDOWS", store)

        seat = client.get(f"/api/rooms/{room_id}/context").json()["seats"][0]

        assert seat["max_context_tokens"] == 40_960
        assert seat["context_window_source"] == "loaded"


class _ModelRecorder(MockLLMConnector):
    """A mock that records the model every request named, for the seat-model tests."""

    def __init__(self) -> None:
        super().__init__()
        self.models: list[str | None] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.models.append(request.model)
        return await super().generate(request)


class TestASeatRunsOnTheSettingsModel:
    """A seat whose persona names no model sends the Settings deep model.

    Before this, the seat sent no model and the connector filled in one written in source
    -- a retired Gemini model -- while Settings showed the model the person had picked.
    """

    def test_a_seat_sends_the_settings_model_and_the_new_one_after_a_save(
        self, tmp_path: Path
    ) -> None:
        """The default deep ref's connection answers, and a saved default reaches the seat.

        Killed by: src/uclone_x/ui/app.py :: gateway=self._gateway,
        Becomes: gateway=None,
        """
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        (sessions / "settings.json").write_text(
            json.dumps(
                {
                    "connections": [{"id": "mock", "kind": "mock"}],
                    "default_models": {"deep": "mock/mock-gpt-4o"},
                }
            ),
            encoding="utf-8",
        )
        app = create_ui_app(static_dir=tmp_path, storage_dir=sessions)
        mgr = cast(Any, app).state.session_manager
        recorders: dict[str, _ModelRecorder] = {}

        def build(**kwargs: Any) -> _ModelRecorder:
            return recorders.setdefault(str(kwargs["model"]), _ModelRecorder())

        mgr.gateway._factory = build  # pyright: ignore[reportPrivateUsage]
        with TestClient(app) as client:
            room_id = _create(client).json()["room_id"]
            client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
            _wait_for_transcript(client, room_id, rows=4)

            assert "mock-gpt-4o" in recorders["mock-gpt-4o"].models

            saved = client.post("/api/settings", json={"default_models": {"deep": "mock/mock-llm"}})
            assert saved.status_code == 200, saved.text

            client.post(f"/api/rooms/{room_id}/messages", json={"content": "again"})
            _wait_for_transcript(client, room_id, rows=6)

            second = recorders.get("mock-llm")
            assert second is not None and second.models, "the seat never moved to the new default"
            assert set(second.models) == {"mock-llm"}, second.models


class TestWhichSeatsAHistoryControlActsOn:
    """`seated_agents` directly, for the records the routes above cannot produce."""

    def test_a_human_carrying_a_session_id_is_still_not_a_seat(self) -> None:
        """`Participant` permits it, so the filter cannot be left to the convention.

        "Empty for a human" is the field's description and what `RoomService` writes — not
        a validator. A record written by hand, restored from an older schema, or produced by
        a later roster path could carry one, and every history control would then reset it:
        a person's own chat session cleared because they joined a conversation.

        Killed by: src/uclone_x/ui/rooms.py :: p.kind is ParticipantKind.AGENT and p.session_id
        Becomes: p.session_id
        """
        state = RoomState(
            room_id="room_x",
            participants=(
                Participant(
                    id="alice",
                    kind=ParticipantKind.HUMAN,
                    display_name="Alice",
                    session_id="sess_alices_own_chat",
                ),
                Participant(
                    id="scout",
                    kind=ParticipantKind.AGENT,
                    display_name="Scout",
                    session_id="sess_room__room_x__scout",
                ),
            ),
        )

        assert [p.id for p in seated_agents(state)] == ["scout"]

    def test_an_agent_with_no_session_of_its_own_is_skipped(self) -> None:
        """`RoomAgentResolver` refuses one because it would share the host's default
        session, and a history control acting on that default would cut back a session
        belonging to something else entirely.

        Killed by: src/uclone_x/ui/rooms.py :: p.kind is ParticipantKind.AGENT and p.session_id
        Becomes: p.kind is ParticipantKind.AGENT
        """
        state = RoomState(
            room_id="room_x",
            participants=(
                Participant(id="scout", kind=ParticipantKind.AGENT, display_name="Scout"),
            ),
        )

        assert seated_agents(state) == ()

    def test_a_seat_that_has_spoken_is_counted_from_the_live_agent(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hazard this slice was written around, on the read side.

        A seat's agent is built and cached by `RoomAgentResolver`, and nothing else holds it
        (the session manager's own agent registry, `_agents` and `get_agent`, was removed
        in #1899). The readout therefore has to be handed the agent the resolver holds; the
        fallback it would otherwise take reads a stored copy, and reports a conversation as
        empty while the agent driving it is one turn from its ceiling.

        `live` still reads `True` without it — the resolver holds the agent either way —
        while `active_turns` is answered from the record rather than from the writer. The
        record is made unreadable here, as for a seat that has not been persisted yet, so
        the only source of the turn is the live agent.

        Killed by: src/uclone_x/ui/rooms.py :: session_id=participant.session_id, agent=live
        Becomes: session_id=participant.session_id
        """
        from uclone_x.agent.session import SessionStore

        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_transcript(client, room_id, rows=4)

        def _unpersisted(self: SessionStore, session_id: str) -> None:
            return None

        monkeypatch.setattr(SessionStore, "load", _unpersisted)

        seat = client.get(f"/api/rooms/{room_id}/context").json()["seats"][0]

        assert seat["live"] is True
        assert seat["active_turns"] >= 1

    def test_an_unknown_conversation_is_a_404(self, client: TestClient) -> None:
        assert client.get("/api/rooms/room_missing/context").status_code == 404


class TestASeatThatWouldNotReset:
    def test_the_answer_names_the_seat_that_kept_its_own_history(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A partial reset is the divergence this slice exists to prevent, so it is said aloud.

        The transcript is already cut when a seat's reset raises, and the loop deliberately
        carries on rather than half-applying. What the caller then reads is 200 and a room
        with an empty transcript, while one participant still answers from the turns that
        were dropped. Logging it tells the operator; it does not tell the only party holding
        the response. An absence states its cause (P6).

        Killed by: src/uclone_x/ui/rooms.py :: kept_stale.append(participant.id)
        Becomes: pass
        """

        from uclone_x.ui.app import AgentSessionManager

        def _refuse(*args: object, **kwargs: object) -> None:
            raise RuntimeError("the store is read-only")

        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_transcript(client, room_id, rows=4)
        monkeypatch.setattr(AgentSessionManager, "clear_session_history", _refuse)

        body = client.delete(f"/api/rooms/{room_id}/history").json()

        assert body["transcript"] == []
        assert body["participants_not_reset"] == [seat_id_for("scout")]

    def test_a_reset_that_worked_says_so_rather_than_omitting_the_key(
        self, client: TestClient
    ) -> None:
        """Present and empty, not absent.

        A key that appears only on failure is one a caller discovers by hitting the failure,
        so the rendering of the bad case is the least-exercised path in the head.

        Killed by: src/uclone_x/ui/rooms.py :: return {**state.model_dump(mode="json"), "participants_not_reset": list(kept_stale)}
        Becomes: return state.model_dump(mode="json")
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_transcript(client, room_id, rows=4)

        body = client.post(f"/api/rooms/{room_id}/history/truncate", json={"seq": 3}).json()

        assert body["participants_not_reset"] == []


class TestClearingAConversation:
    def test_the_conversation_survives_its_own_emptying(self, client: TestClient) -> None:
        """Not `DELETE /api/rooms/{id}`, and not a new room alongside the old one."""
        room_id = _create(client, title="Keep me", agent_ids=["scout"]).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_transcript(client, room_id, rows=4)

        body = client.delete(f"/api/rooms/{room_id}/history").json()

        assert body["transcript"] == []
        assert body["title"] == "Keep me"
        assert [p["id"] for p in body["participants"]] == ["user", seat_id_for("scout")]
        assert [r["room_id"] for r in client.get("/api/rooms").json()["rooms"]] == [room_id]

    def test_the_live_agent_forgets_what_the_record_no_longer_holds(
        self, client: TestClient
    ) -> None:
        """The hazard this slice was written around, on the write side.

        `clear_session_history` reaches its agent through `AgentSessionManager.get_agent`,
        which cannot see a seat: `RoomAgentResolver` caches it and never registers it there.
        Taking the no-agent branch deletes the stored record while the live agent goes on
        holding the messages — and persists them back over the deletion at its next turn, so
        the clear presents as a success and then silently undoes itself.

        Without it the route still answers an empty transcript and this agent still holds
        the turn it took — which is the whole failure, visible nowhere in the response.

        Killed by: src/uclone_x/ui/rooms.py :: agent=stack.live_agent(state.room_id, participant.session_id),
        Becomes: agent=None,
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_transcript(client, room_id, rows=4)
        scout = seat_id_for("scout")
        agent = _live_seat_agent(client, room_id, scout)
        assert agent is not None, "the cascade should have built the seat's agent"
        session_id = f"sess_room__{room_id}__{scout}"
        assert [m.role.value for m in agent.get_session(session_id).messages] != ["system"]

        client.delete(f"/api/rooms/{room_id}/history")

        assert [m.role.value for m in agent.get_session(session_id).messages] == ["system"]

    def test_every_read_mark_goes_with_the_record(self, client: TestClient) -> None:
        """`seq` restarts at 1, so a surviving mark hides the cleared room's next turns."""
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_transcript(client, room_id, rows=4)

        assert client.delete(f"/api/rooms/{room_id}/history").json()["last_seen_seq"] == {}

    def test_an_unknown_conversation_is_a_404(self, client: TestClient) -> None:
        assert client.delete("/api/rooms/room_missing/history").status_code == 404


class TestRewindingAConversation:
    def test_a_rewind_keeps_the_named_message(self, client: TestClient) -> None:
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_transcript(client, room_id, rows=4)

        body = client.post(f"/api/rooms/{room_id}/history/truncate", json={"seq": 3})

        assert body.status_code == 200, body.text
        kept = body.json()["transcript"]
        assert [m["seq"] for m in kept] == [1, 2, 3]
        assert kept[-1]["content"] == "hello", "the named message is kept, not cut with the rest"

    def test_the_seats_are_cut_back_with_the_record(self, client: TestClient) -> None:
        """A rewound conversation whose speakers still remember the dropped turns is the
        failure the room-level route exists to prevent."""
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_transcript(client, room_id, rows=4)
        scout = seat_id_for("scout")
        agent = _live_seat_agent(client, room_id, scout)
        assert agent is not None

        client.post(f"/api/rooms/{room_id}/history/truncate", json={"seq": 3})

        session_id = f"sess_room__{room_id}__{scout}"
        assert [m.role.value for m in agent.get_session(session_id).messages] == ["system"]

    def test_a_seq_the_conversation_does_not_hold_keeps_the_cores_sentence(
        self, client: TestClient
    ) -> None:
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_transcript(client, room_id, rows=4)

        refused = client.post(f"/api/rooms/{room_id}/history/truncate", json={"seq": 99})

        assert refused.status_code == 400
        assert "no message at seq 99" in refused.json()["detail"]

    def test_a_seq_that_is_not_a_number_is_refused_rather_than_coerced(
        self, client: TestClient
    ) -> None:
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]

        refused = client.post(f"/api/rooms/{room_id}/history/truncate", json={"seq": "1"})

        assert refused.status_code == 400
        assert "must be the number of the message" in refused.json()["detail"]

    def test_a_boolean_is_not_a_seq(self, client: TestClient) -> None:
        """`True` is an `int` in Python, and `seq=True` would rewind to message 1.

        Killed by: src/uclone_x/ui/rooms.py :: if not isinstance(raw, int) or isinstance(raw, bool):
        Becomes: if not isinstance(raw, int):
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]

        refused = client.post(f"/api/rooms/{room_id}/history/truncate", json={"seq": True})

        assert refused.status_code == 400


class TestCompactingAConversation:
    def test_every_seat_is_reported_on_its_own(self, client: TestClient) -> None:
        """Never one merged figure: compaction carries the provenance of whichever producer
        wrote its ledger, and averaging four of those invents an attribution (P6)."""
        room_id = _create(client, agent_ids=["scout", "critic"]).json()["room_id"]

        body = client.post(f"/api/rooms/{room_id}/compact", json={})

        assert body.status_code == 200, body.text
        results = body.json()["results"]
        assert [r["participant_id"] for r in results] == [
            seat_id_for("scout"),
            seat_id_for("critic"),
        ]
        assert all("provenance" in r for r in results)

    def test_one_seat_can_be_named(self, client: TestClient) -> None:
        """U0 asks for the conversation and U1 refines to a seat, one argument away.

        The seat is its clone's id, and the handle a person types names it too, as the
        dock routes and removing a seat already accept.

        Killed by: src/uclone_x/ui/rooms.py :: seat = seat_id_for(raw)  # a handle names its clone's seat
        Becomes: seat = raw
        """
        room_id = _create(client, agent_ids=["scout", "critic"]).json()["room_id"]

        critic = seat_id_for("critic")  # a seat is named by its clone's id
        for named in (critic, "critic"):
            body = client.post(f"/api/rooms/{room_id}/compact", json={"participant_id": named})

            assert body.status_code == 200, body.text
            assert [r["participant_id"] for r in body.json()["results"]] == [critic]

    def test_a_seat_that_is_not_in_the_conversation_is_refused_with_its_reason(
        self, client: TestClient
    ) -> None:
        """Compacting a seat the conversation does not hold would otherwise report a
        success over an empty fan-out.

        Killed by: src/uclone_x/ui/rooms.py :: if not seats:
        Becomes: if False:
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]

        refused = client.post(f"/api/rooms/{room_id}/compact", json={"participant_id": "critic"})

        assert refused.status_code == 400
        assert "not a seated agent" in refused.json()["detail"]

    def test_an_unknown_conversation_is_a_404(self, client: TestClient) -> None:
        assert client.post("/api/rooms/room_missing/compact", json={}).status_code == 404


class TestHistoryIsRefusedWhileSomebodyIsAnswering:
    @pytest.mark.asyncio
    async def test_a_running_cascade_is_what_turn_in_flight_reports(self, tmp_path: Path) -> None:
        """The input to the refusal, pinned on the real thing rather than on a patch.

        A cascade is a task this stack started and has not seen finish; a turn between two
        agents holds no floor and is still about to write, which is why this reads `_running`
        rather than the orchestrator's floor.
        """
        from uclone_x.ui.app import AgentSessionManager
        from uclone_x.ui.rooms import RoomStack

        stack = RoomStack(
            AgentSessionManager(storage_dir=tmp_path / "sessions", llm=MockLLMConnector())
        )
        assert stack.turn_in_flight("room_x") is False
        release = asyncio.Event()
        stack.drive("room_x", release.wait)
        await asyncio.sleep(0)

        assert stack.turn_in_flight("room_x") is True

        release.set()
        await asyncio.sleep(0.05)
        assert stack.turn_in_flight("room_x") is False

    @pytest.mark.parametrize(
        ("method", "path", "payload"),
        [
            ("post", "/compact", {}),
            ("post", "/history/truncate", {"seq": 3}),
            ("delete", "/history", None),
        ],
    )
    def test_each_history_control_declines_and_names_the_remedy(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> None:
        """What this pins is the *wiring*: that every one of the three asks.

        The condition itself is pinned above, on a real task. Producing a genuinely
        in-flight cascade here would mean a model that blocks until the test releases it,
        which makes the fixture the thing most likely to break the test. A control that
        forgot to ask would let a turn land after the cut, writing a reply into a
        conversation that no longer holds the messages it answered.

        Killed by: src/uclone_x/ui/rooms.py :: _refuse_during_turn(stack, room_id, "Clearing")
        Becomes: pass
        """
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_transcript(client, room_id, rows=4)
        monkeypatch.setattr(RoomStack, "turn_in_flight", _always_answering)

        refused = client.request(method.upper(), f"/api/rooms/{room_id}{path}", json=payload)

        assert refused.status_code == 409, refused.text
        assert "Stop the conversation first" in refused.json()["detail"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("path", "method", "writer", "payload"),
        [
            ("/api/rooms/{room_id}/history/truncate", "POST", "truncate_transcript", True),
            ("/api/rooms/{room_id}/history", "DELETE", "clear_transcript", False),
        ],
    )
    async def test_the_two_history_routes_never_yield_between_the_question_and_the_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        path: str,
        method: str,
        writer: str,
        payload: bool,
    ) -> None:
        """The property that makes `_refuse_during_turn` a *guard* here, pinned (#1213).

        These two routes ask whether a turn is running and then cut the transcript with no
        `await` in between, so the whole handler is one uninterrupted pass of the event
        loop and a cascade cannot begin inside it. Nothing below re-asks: `RoomService`
        sees the store and has never heard of `RoomStack._running`, so if a turn could
        start between the question and the write, it would land afterwards carrying a
        `seq` read from a transcript length that no longer exists -- the exact defect the
        refusal is there to prevent. The atomicity is therefore load-bearing, and until
        now it was an accident of how the handlers happened to be written.

        Observed rather than asserted about: a task created *before* the handler is
        awaited cannot run until the handler yields. `create_task` schedules it and
        nothing more, so `tripwire.done()` being false after `await handler(...)` says the
        handler never gave the loop a turn -- with the seated cascade the tripwire would
        have started recorded at the write for good measure. Any `await` added anywhere in
        either handler, for any reason, flips both.

        `/compact` is deliberately absent from this parametrization and is not an omission:
        it awaits by construction, which is why its guard is the Core's and not this one.
        The test below is its half.

        Killed by: src/uclone_x/ui/rooms.py :: state = stack.service.truncate_transcript(room_id, raw)
        Becomes: await asyncio.sleep(0); state = stack.service.truncate_transcript(room_id, raw)
        Killed by: src/uclone_x/ui/rooms.py :: state = stack.service.clear_transcript(room_id)
        Becomes: await asyncio.sleep(0); state = stack.service.clear_transcript(room_id)
        """
        app = create_ui_app(
            static_dir=tmp_path, storage_dir=tmp_path / "sessions", llm=MockLLMConnector()
        )
        stack = cast(RoomStack, cast(Any, app).state.room_stack)
        room_id = _seated_room(stack)
        state = await stack.orchestrator(stack.service.get(room_id)).accept(
            room_id, "user", "hello"
        )
        seq = state.transcript[-1].seq

        at_the_write: list[bool] = []
        unmutated: Any = getattr(RoomService, writer)

        def spy(self: Any, room_id: str, *args: Any) -> Any:
            at_the_write.append(stack.turn_in_flight(room_id))
            return unmutated(self, room_id, *args)

        monkeypatch.setattr(RoomService, writer, spy)

        async def a_cascade_starts_now() -> None:
            stack.drive(room_id, asyncio.Event().wait)

        tripwire = asyncio.create_task(a_cascade_starts_now())
        handler = _endpoint(app, path, method)

        await (handler(room_id, {"seq": seq}) if payload else handler(room_id))

        assert tripwire.done() is False, (
            "the handler yielded to the event loop between its refusal and its write, so a "
            "cascade can now begin inside the window this refusal exists to close"
        )
        assert at_the_write == [False], at_the_write
        await tripwire
        await stack.close()

    @pytest.mark.asyncio
    async def test_compaction_is_refused_by_the_core_when_this_check_is_already_stale(
        self, tmp_path: Path
    ) -> None:
        """`/compact`'s half of the asymmetry: its guard is not the one above (#1213).

        The route awaits between asking `_refuse_during_turn` and rewriting a seat, so its
        answer can be out of date by the time the write happens -- and for the second and
        later seats of a fan-out it routinely is. What makes that safe is not a narrower
        window but a *different guard*: `BaseAgent.compact_session` runs
        `_refuse_session_mutation_during_turn` synchronously against the seat it is about
        to rewrite, at the instant it rewrites it, so it cannot be stale by construction.

        This stages precisely that: a seat that is answering while the room-level question
        answers `False`, which is the window the card describes. Compaction is still
        refused, and the sentence the caller gets is the Core's -- naming the agent, the
        session and the remedy. Before this change that refusal reached the browser as
        `500 The conversation service failed`, which is a guard a caller cannot act on and
        the same divergence #1212 closed for `/context`.

        `_turn_lock` is reached directly because it *is* the Core's definition of a turn in
        flight -- `_refuse_session_mutation_during_turn` reads `self._turn_lock.locked()`
        and nothing else. Driving a real cascade and catching it mid-turn would make the
        model double the thing most likely to break the test.

        Killed by: src/uclone_x/ui/rooms.py :: if isinstance(exc, SessionMutationDuringTurnError):
        Becomes: if False:
        Killed by: src/uclone_x/agent/compaction_driver.py :: self._refuse_session_mutation_during_turn(sid, "compact")
        Becomes: pass
        """
        app = create_ui_app(
            static_dir=tmp_path, storage_dir=tmp_path / "sessions", llm=MockLLMConnector()
        )
        stack = cast(RoomStack, cast(Any, app).state.room_stack)
        room_id = _seated_room(stack)
        state = stack.service.get(room_id)
        seat = seated_agents(state)[0]
        agent = await stack.resolve_agent(state, seat)

        async with cast(Any, agent)._turn_lock:
            assert stack.turn_in_flight(room_id) is False, (
                "the room-level question must answer False here, or this test is not "
                "standing in the window it claims to be standing in"
            )
            with pytest.raises(HTTPException) as refused:
                await _endpoint(app, "/api/rooms/{room_id}/compact", "POST")(room_id, {})

        assert refused.value.status_code == 409, refused.value.detail
        detail = str(refused.value.detail)
        assert "while a reasoning turn is in flight" in detail, detail
        assert seat.session_id in detail, detail
        await stack.close()


class TestRoomLoop:
    def test_room_slash_loop_help(self, client: TestClient, tmp_path: Path) -> None:
        """Sending /loop or /loop help records help text in transcript without agent cascade.

        The typed command is a note from the person, not their message, so no seat is
        handed it as something to answer (#1661).

        Killed by: src/uclone_x/ui/rooms.py :: return _notice("loop.help")
        Becomes: return _notice("loop.no_interval")
        Killed by: src/uclone_x/ui/rooms.py :: await orch.accept_command(room_id, sender_id, content)
        Becomes: await orch.accept(room_id, sender_id, content)
        """
        room_id = _create(client).json()["room_id"]
        res = client.post(f"/api/rooms/{room_id}/messages", json={"content": "/loop help"})
        assert res.status_code == 202

        room = client.get(f"/api/rooms/{room_id}").json()
        assert [m for m in room["transcript"] if m.get("kind") == "utterance"] == []
        command, help_note = [m for m in room["transcript"] if m["kind"] == "note"]
        assert (command["kind"], command["sender_id"], command["content"]) == (
            "note",
            "user",
            "/loop help",
        )
        assert "code" not in command  # left out while unset (#1885)
        # Stored as a note, and not only read as one: loading converts a command saved as
        # speech by an older build, which would hide this route saving it as speech.
        stored = json.loads((tmp_path / "sessions" / "rooms" / f"{room_id}.json").read_text())
        assert [m["kind"] for m in stored["transcript"] if m["sender_id"] == "user"][-1] == "note"
        # The help is a note, not speech: shown in the room, kept out of seat spans (#1641).
        notes = _notes(client, room_id)
        assert notes == [help_note]
        assert notes[0]["sender_id"] == "system"
        assert notes[0]["code"] == "loop.help"
        assert "params" not in notes[0]  # left out while unset (#1885)
        # The stored fallback is English, for exports and heads that predate the code.
        assert "`/loop list`" in notes[0]["content"]
        assert not _HANGUL.search(notes[0]["content"])

    def test_a_word_that_starts_with_loop_is_one_message(self, client: TestClient) -> None:
        """`/loopy` is not a command; it was recorded as one and then again as a message.

        Killed by: src/uclone_x/ui/rooms.py :: if is_loop_command(clean_content):
        Becomes: if clean_content.startswith("/loop"):
        """
        room_id = _create(client).json()["room_id"]
        res = client.post(f"/api/rooms/{room_id}/messages", json={"content": "/loopy idea"})
        assert res.status_code == 202

        room = client.get(f"/api/rooms/{room_id}").json()
        rows = [
            (m["kind"], m["content"])
            for m in room["transcript"]
            if m["sender_id"] == "user" and m["kind"] != "join"
        ]
        assert rows == [("utterance", "/loopy idea")]

    def test_room_slash_loop_schedule_and_stop(self, client: TestClient) -> None:
        """Sending /loop <interval> <prompt> registers recurring loop and /loop stop cancels it.

        Each reply is a code with its values, which the head words in the reader's language;
        the interval travels as seconds for the head to format.

        Killed by: src/uclone_x/ui/rooms.py :: "interval_seconds": interval_seconds,
        Becomes: "interval_seconds": 0,
        Killed by: src/uclone_x/ui/rooms.py :: return _notice("loop.stopped" if stopped else "loop.nothing_to_stop")
        Becomes: return _notice("loop.nothing_to_stop" if stopped else "loop.stopped")
        """
        room_id = _create(client).json()["room_id"]
        res = client.post(
            f"/api/rooms/{room_id}/messages",
            json={"content": "/loop 10s status check"},
        )
        assert res.status_code == 202

        notes = _notes(client, room_id)
        assert notes[-1]["code"] == "loop.registered"
        params = notes[-1]["params"]
        assert params["interval_seconds"] == 10.0
        assert params["prompt"] == "status check"
        job_id = params["job_id"]
        assert isinstance(job_id, str) and job_id
        assert notes[-1]["content"].startswith("🔄 **Repeating task started** (every 10 seconds")

        client.post(f"/api/rooms/{room_id}/messages", json={"content": "/loop list"})
        notes = _notes(client, room_id)
        assert notes[-1]["code"] == "loop.active"
        assert notes[-1]["params"] == {
            "job_id": job_id,
            "interval_seconds": 10.0,
            "prompt": "status check",
            "runs": 1,
        }

        client.post(f"/api/rooms/{room_id}/messages", json={"content": "/loop stop"})
        notes = _notes(client, room_id)
        assert notes[-1]["code"] == "loop.stopped"

        client.post(f"/api/rooms/{room_id}/messages", json={"content": "/loop stop"})
        notes = _notes(client, room_id)
        assert notes[-1]["code"] == "loop.nothing_to_stop"
        assert all(not _HANGUL.search(n["content"]) for n in notes)

    @pytest.mark.parametrize(
        ("command", "code", "params"),
        [
            ("/loop not_an_interval prompt", "loop.no_interval", None),
            ("/loop 5m", "loop.missing_prompt", None),
            ("/loop 0.5s check", "loop.interval_too_short", {"interval_seconds": 1.0}),
        ],
    )
    def test_room_slash_loop_invalid_syntax(
        self, client: TestClient, command: str, code: str, params: dict[str, float] | None
    ) -> None:
        """A command that cannot be read is answered with why, as a code; never the parser's text.

        The parser's message names internals ("spin loops", the unit table) and is English on
        every screen, so the conversation carries the reason instead.

        Killed by: src/uclone_x/ui/rooms.py :: if err.reason == "missing_prompt":
        Becomes: if err.reason == "no_prompt":
        Killed by: src/uclone_x/ui/rooms.py :: if err.reason == "interval_too_short":
        Becomes: if err.reason == "too_short":
        """
        room_id = _create(client).json()["room_id"]
        res = client.post(f"/api/rooms/{room_id}/messages", json={"content": command})
        assert res.status_code == 202
        notes = _notes(client, room_id)
        assert [n["code"] for n in notes] == [code]
        assert notes[0].get("params") == params  # left out while unset (#1885)
        assert "spin loop" not in notes[0]["content"]
        assert "Supported units" not in notes[0]["content"]

    def test_room_slash_loop_stopped_via_room_stop_button(self, client: TestClient) -> None:
        """POST /api/rooms/{id}/stop cancels any active recurring loop for that room.

        Killed by: src/uclone_x/ui/rooms.py :: return _notice("loop.none_active")
        Becomes: return _notice("loop.active")
        """
        room_id = _create(client).json()["room_id"]
        client.post(
            f"/api/rooms/{room_id}/messages",
            json={"content": "/loop 60s check something"},
        )
        assert _notes(client, room_id)[-1]["content"].startswith(
            "🔄 **Repeating task started** (every 1 minute,"
        )
        # Calling stop endpoint
        stop_res = client.post(f"/api/rooms/{room_id}/stop")
        assert stop_res.status_code == 200

        # /loop list now says no active loop
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "/loop list"})
        notes = _notes(client, room_id)
        assert notes[-1]["code"] == "loop.none_active"
        assert "params" not in notes[-1]  # left out while unset (#1885)

    def test_a_loop_is_on_the_room_record_until_it_is_stopped(self, client: TestClient) -> None:
        """The loop is read from the room, not typed for: it lived only in memory (#1936).

        Killed by: src/uclone_x/ui/rooms.py :: self.store.save(state.model_copy(update={"loop": loop}))
        Becomes: pass
        Killed by: src/uclone_x/ui/rooms.py :: self.store.save(state.model_copy(update={"loop": None}))
        Becomes: pass
        """
        room_id = _create(client).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "/loop 60s status check"})
        loop = _wait_for_loop(client, room_id, runs=1)
        assert loop["interval_seconds"] == 60.0
        assert loop["prompt"] == "status check"
        assert loop["job_id"] == _notes(client, room_id)[0]["params"]["job_id"]
        assert loop["next_run_at"] is not None

        client.post(f"/api/rooms/{room_id}/messages", json={"content": "/loop stop"})
        assert "loop" not in client.get(f"/api/rooms/{room_id}").json()

    def test_a_restarted_app_continues_the_loop_and_says_so(self, tmp_path: Path) -> None:
        """`ucx ui --dev` reloads on a `git pull`, and the loop ended without a word (#1936).

        Killed by: src/uclone_x/ui/app.py :: resumed = loop_stack.resume_room_loops()
        Becomes: resumed = 0
        """

        def _app() -> Any:
            return create_ui_app(
                static_dir=tmp_path, storage_dir=tmp_path / "sessions", llm=MockLLMConnector()
            )

        with TestClient(_app()) as first:
            room_id = _create(first).json()["room_id"]
            first.post(f"/api/rooms/{room_id}/messages", json={"content": "/loop 60s status check"})
            _wait_for_loop(first, room_id, runs=1)
            _wait_for_cascade_to_settle(cast(Any, first.app).state.room_stack, room_id)

        store = RoomStore(room_storage_dir_under(tmp_path / "sessions"))
        state = store.load(room_id)
        assert state is not None and state.loop is not None, "shutting down took the loop off"
        store.save(
            state.model_copy(
                update={
                    "loop": state.loop.model_copy(
                        update={"next_run_at": "2020-01-01T00:00:00+00:00"}
                    )
                }
            )
        )

        with TestClient(_app()) as second:
            loop = _wait_for_loop(second, room_id, runs=2)
            assert loop["job_id"] == state.loop.job_id
            resumed = [n for n in _notes(second, room_id) if n["code"] == "loop.resumed"]
            assert len(resumed) == 1
            assert resumed[0]["params"]["prompt"] == "status check"
            prompts = [
                m
                for m in second.get(f"/api/rooms/{room_id}").json()["transcript"]
                if m.get("kind") == "utterance" and m["content"] == "status check"
            ]
            assert len(prompts) == 2
            _wait_for_cascade_to_settle(cast(Any, second.app).state.room_stack, room_id)

    def test_a_restarted_app_waits_for_future_next_run_at(self, tmp_path: Path) -> None:
        """A future next_run_at is honored across restart so reloads do not re-run all loops (#1996).

        Killed by: src/uclone_x/ui/rooms.py :: delay = max(0.0, (due - datetime.now(UTC)).total_seconds())
        Becomes: delay = 0.0
        """

        def _app() -> Any:
            return create_ui_app(
                static_dir=tmp_path, storage_dir=tmp_path / "sessions", llm=MockLLMConnector()
            )

        with TestClient(_app()) as first:
            room_id = _create(first).json()["room_id"]
            first.post(f"/api/rooms/{room_id}/messages", json={"content": "/loop 60s status check"})
            _wait_for_loop(first, room_id, runs=1)
            _wait_for_cascade_to_settle(cast(Any, first.app).state.room_stack, room_id)

        store = RoomStore(room_storage_dir_under(tmp_path / "sessions"))
        state = store.load(room_id)
        assert state is not None and state.loop is not None
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        store.save(
            state.model_copy(update={"loop": state.loop.model_copy(update={"next_run_at": future})})
        )

        with TestClient(_app()) as second:
            stack: Any = cast(Any, second.app).state.room_stack
            assert room_id in stack._room_loops
            job_id, _ = stack._room_loops[room_id]
            assert job_id == state.loop.job_id

            resumed = [n for n in _notes(second, room_id) if n["code"] == "loop.resumed"]
            assert len(resumed) == 1
            assert resumed[0]["params"]["prompt"] == "status check"

            time.sleep(0.05)

            room_data = second.get(f"/api/rooms/{room_id}").json()
            assert room_data["loop"]["runs"] == 1
            prompts = [
                m
                for m in room_data["transcript"]
                if m.get("kind") == "utterance" and m["content"] == "status check"
            ]
            assert len(prompts) == 1
            _wait_for_cascade_to_settle(stack, room_id)

    def test_a_failed_run_is_said_in_the_conversation_and_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed run was a log line on a terminal nobody reads; now the room shows it.

        Killed by: src/uclone_x/ui/rooms.py :: last_error=error,
        Becomes: last_error=None,
        Killed by: src/uclone_x/ui/rooms.py :: "loop.run_failed",
        Becomes: "loop.stopped",
        Killed by: src/uclone_x/ui/rooms.py :: exc, context={"surface": "ui.room_loop", "room_id": room_id, "run": run}
        Becomes: exc, context={}
        """
        journaled: list[dict[str, object]] = []

        def _journal(error: BaseException, *, context: dict[str, object] | None = None) -> None:
            journaled.append(dict(context or {}))

        monkeypatch.setattr(rooms_module, "record_failure", _journal)
        sessions = tmp_path / "sessions"
        with TestClient(
            create_ui_app(static_dir=tmp_path, storage_dir=sessions, llm=MockLLMConnector())
        ) as setup:
            room_id = _create(setup).json()["room_id"]
        store = RoomStore(room_storage_dir_under(sessions))
        state = store.load(room_id)
        assert state is not None
        ghost = RoomLoop(job_id="loop-ghost", sender_id="ghost", interval_seconds=60.0, prompt="hi")
        store.save(state.model_copy(update={"loop": ghost}))

        with TestClient(
            create_ui_app(static_dir=tmp_path, storage_dir=sessions, llm=MockLLMConnector())
        ) as client:
            loop = _wait_for_loop(client, room_id, runs=1, failed=True)
            assert loop["last_error"], loop
            failed = [n for n in _notes(client, room_id) if n["code"] == "loop.run_failed"]
            assert len(failed) == 1
            assert failed[0]["params"]["run"] == 1
            assert "ghost" in failed[0]["params"]["reason"]
        assert journaled == [{"surface": "ui.room_loop", "room_id": room_id, "run": 1}]

    def test_the_stop_button_says_it_stopped_a_loop(self, client: TestClient) -> None:
        """Stop ends the loop and says so; with none running it adds nothing (#1936).

        Killed by: src/uclone_x/ui/rooms.py :: if self.cancel_room_loop(room_id):
        Becomes: if self.cancel_room_loop(room_id) or True:
        """
        room_id = _create(client).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "/loop 60s status check"})
        _wait_for_loop(client, room_id, runs=1)
        client.post(f"/api/rooms/{room_id}/stop")
        client.post(f"/api/rooms/{room_id}/stop")
        codes = [n["code"] for n in _notes(client, room_id)]
        assert codes == ["loop.registered", "loop.stopped"]
        assert "loop" not in client.get(f"/api/rooms/{room_id}").json()


class TestAutonomousAndPresence:
    def test_room_toggle_autonomous_and_presence(self, client: TestClient) -> None:
        """POST /api/rooms/{id}/autonomous toggles policy, /presence updates presence."""
        room_id = _create(client).json()["room_id"]

        # Initially autonomous is False
        room = client.get(f"/api/rooms/{room_id}").json()
        assert "autonomous" not in room["policy"]  # left out while off (#1885)

        # Enable autonomous
        res = client.post(f"/api/rooms/{room_id}/autonomous", json={"enabled": True})
        assert res.status_code == 200
        assert res.json()["policy"]["autonomous"] is True

        # Presence report
        pres_res = client.post(f"/api/rooms/{room_id}/presence", json={"active": True})
        assert pres_res.status_code == 200
        assert pres_res.json()["active"] is True

        # Disable autonomous
        res_off = client.post(f"/api/rooms/{room_id}/autonomous", json={"enabled": False})
        assert res_off.status_code == 200
        assert "autonomous" not in res_off.json()["policy"]

        # Re-enabling autonomous resets turn count if circuit breaker was reached
        from uclone_x.room.orchestrator import AUTONOMOUS_CIRCUIT_BREAKER_TURNS

        stack = cast(RoomStack, cast(Any, client.app).state.room_stack)
        _wait_for_cascade_to_settle(stack, room_id)
        state = stack.store.load(room_id)
        assert state is not None
        exhausted = state.turn_state.model_copy(
            update={"agent_turns_since_human": AUTONOMOUS_CIRCUIT_BREAKER_TURNS}
        )
        stack.store.save(state.model_copy(update={"turn_state": exhausted}))

        res_on = client.post(f"/api/rooms/{room_id}/autonomous", json={"enabled": True})
        assert res_on.status_code == 200
        reloaded = stack.store.load(room_id)
        assert reloaded is not None
        assert reloaded.turn_state.agent_turns_since_human == 0

    def test_presence_does_not_loop_when_not_presence_paused(self, client: TestClient) -> None:
        """POST /api/rooms/{id}/presence only resumes when previously paused due to inactive presence."""
        from unittest.mock import MagicMock

        from uclone_x.room.models import SelectionVerdict, SpeakerDecision

        room_id = _create(client).json()["room_id"]
        client.post(f"/api/rooms/{room_id}/autonomous", json={"enabled": True})

        stack = cast(RoomStack, cast(Any, client.app).state.room_stack)
        _wait_for_cascade_to_settle(stack, room_id)
        state = stack.store.load(room_id)
        assert state is not None

        # Set last_decision to normal silence
        normal_silence = SpeakerDecision(
            verdict=SelectionVerdict.SILENCE,
            selector="orchestrator",
            reasoning="conversation reached natural resting point",
        )
        stack.store.save(state.model_copy(update={"last_decision": normal_silence}))

        # Mock drive to verify it is NOT called on normal presence heartbeat
        orig_drive = stack.drive
        stack.drive = MagicMock()
        try:
            res = client.post(f"/api/rooms/{room_id}/presence", json={"active": True})
            assert res.status_code == 200
            stack.drive.assert_not_called()
        finally:
            stack.drive = orig_drive

    def test_presence_resumes_when_presence_paused(self, client: TestClient) -> None:
        """POST /api/rooms/{id}/presence resumes when previously paused due to user being away."""
        from unittest.mock import MagicMock

        from uclone_x.room.models import SelectionVerdict, SpeakerDecision

        room_id = _create(client).json()["room_id"]
        stack = cast(RoomStack, cast(Any, client.app).state.room_stack)
        state = stack.store.load(room_id)
        assert state is not None

        # Directly configure policy with autonomous=True and last_decision to presence paused
        presence_paused = SpeakerDecision(
            verdict=SelectionVerdict.SILENCE,
            selector="orchestrator",
            reasoning="autonomous discussion paused: user is not actively viewing the room",
        )
        stack.store.save(
            state.model_copy(
                update={
                    "policy": state.policy.model_copy(update={"autonomous": True}),
                    "last_decision": presence_paused,
                }
            )
        )

        orig_drive = stack.drive
        stack.drive = MagicMock()
        try:
            res = client.post(f"/api/rooms/{room_id}/presence", json={"active": True})
            assert res.status_code == 200
            stack.drive.assert_called_once()
        finally:
            stack.drive = orig_drive

    @pytest.mark.asyncio
    async def test_drive_records_cascade_failure_decision(self) -> None:
        """A cascade failure records a SILENCE last_decision explaining the stop."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock

        from uclone_x.errors import SpeakerSelectionError
        from uclone_x.room.models import RoomPolicy, RoomState, SelectionVerdict, TurnState

        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = MagicMock()
            session_mgr.storage_dir = Path(tmpdir)
            session_mgr.bus = MagicMock()
            stack = RoomStack(session_mgr=session_mgr)

            state = RoomState(
                room_id="r_fail",
                policy=RoomPolicy(),
                turn_state=TurnState(),
            )
            stack.store.save(state)

            async def failing_cascade() -> None:
                raise SpeakerSelectionError("Selector test failed")

            stack.drive("r_fail", failing_cascade)
            # Give background task time to run
            await asyncio.sleep(0.05)

            reloaded = stack.store.load("r_fail")
            assert reloaded is not None
            assert reloaded.last_decision is not None
            assert reloaded.last_decision.verdict == SelectionVerdict.SILENCE
            assert "cascade stopped" in reloaded.last_decision.reasoning

    @pytest.mark.asyncio
    async def test_get_room_reports_active_turn_in_flight_and_clears_on_completion(
        self, client: TestClient
    ) -> None:
        """GET /api/rooms/{id} returns active_turn when a turn runs, None when idle."""
        room_id = _create(client, agent_ids=["scout"]).json()["room_id"]
        res = client.get(f"/api/rooms/{room_id}").json()
        assert res["active_turn"] is None

        stack = cast(RoomStack, cast(Any, client.app).state.room_stack)
        ev = asyncio.Event()

        async def pause_task() -> None:
            await ev.wait()

        stack.drive(room_id, pause_task)
        try:
            assert stack.turn_in_flight(room_id) is True
            active = client.get(f"/api/rooms/{room_id}").json()["active_turn"]
            assert active is not None
            assert active["in_flight"] is True
            assert active["agent_id"] == seat_id_for("scout")
        finally:
            ev.set()
            for _ in range(50):
                if not stack.turn_in_flight(room_id):
                    break
                await asyncio.sleep(0.01)

        assert client.get(f"/api/rooms/{room_id}").json()["active_turn"] is None


class TestAHeadRoomInTheApp:
    """The app reads a head's room and cannot post into it (#1885)."""

    @staticmethod
    def _head_room(client: TestClient) -> str:
        from uclone_x.room.one_seat import HeadTurn, record_head_turn

        stack = cast(Any, client.app).state.room_stack
        state = record_head_turn(
            stack.store,
            room_id="room_from_the_terminal",
            clone_id="scout",
            turn=HeadTurn(prompt="what is in the index?", content="three tables"),
            head="run",
        )
        assert state.head == "run"
        return state.room_id

    @pytest.mark.parametrize(
        ("route", "body"),
        [("messages", {"content": "hello from the app"}), ("retry", None)],
    )
    def test_a_post_is_refused_with_the_plain_reason_and_nothing_is_written(
        self, answering_client: TestClient, route: str, body: dict[str, Any] | None
    ) -> None:
        """Unmapped, the refusal reaches the head as a bare 500 with no reason to show.

        Killed by: src/uclone_x/ui/rooms.py :: (NothingToRetryError, HeadRoomWriteError)
        Becomes: (NothingToRetryError,)
        """
        room_id = self._head_room(answering_client)
        before = answering_client.get(f"/api/rooms/{room_id}").json()

        refused = answering_client.post(f"/api/rooms/{room_id}/{route}", json=body)

        assert refused.status_code == 409, refused.text
        assert refused.json()["detail"] == (
            "This conversation belongs to ucx run, so only ucx run can continue it. "
            "You can read it here."
        )
        assert answering_client.get(f"/api/rooms/{room_id}").json() == before

    def test_every_route_that_changes_a_room_refuses_a_head_room(self, client: TestClient) -> None:
        """One guard, every writing route: the enumeration is of the app, not of a list here.

        Each non-read route under `/api/rooms/{room_id}` is sent to a fresh head room, and
        each must answer 409 with the plain refusal and leave the stored room as it was --
        compact, rewind, clear, roster, rename, autonomy and presence included, which write
        without the orchestrator. The one exception is deleting the room itself (author's
        choice, `_HEAD_ROOM_OWNER_ACTIONS`). The named paths below only guard against the
        enumeration going quietly empty.

        Killed by: src/uclone_x/ui/rooms.py :: if head is not None:
        Becomes: if False:
        Killed by: src/uclone_x/ui/rooms.py :: if request.method in _READ_METHODS:
        Becomes: if True:
        Killed by: src/uclone_x/ui/rooms.py :: return self._app.patch(path, dependencies=self._dependencies, **kwargs)
        Becomes: return self._app.patch(path, **kwargs)
        """
        from uclone_x.room.one_seat import HeadTurn, record_head_turn

        stack = cast(Any, client.app).state.room_stack
        writes = sorted(
            (method, route.path)
            for route in cast(list[Any], cast(Any, client.app).routes)
            if str(getattr(route, "path", "")).startswith("/api/rooms/{room_id}")
            for method in (getattr(route, "methods", None) or ())
            if method not in ("GET", "HEAD", "OPTIONS")
        )
        assert {
            ("PATCH", "/api/rooms/{room_id}"),
            ("POST", "/api/rooms/{room_id}/participants"),
            ("DELETE", "/api/rooms/{room_id}/participants/{participant_id}"),
            ("POST", "/api/rooms/{room_id}/compact"),
            ("POST", "/api/rooms/{room_id}/history/truncate"),
            ("DELETE", "/api/rooms/{room_id}/history"),
            ("POST", "/api/rooms/{room_id}/autonomous"),
            ("POST", "/api/rooms/{room_id}/presence"),
        } <= set(writes)

        for n, (method, path) in enumerate(writes):
            room_id = record_head_turn(
                stack.store,
                room_id=f"room_head_{n}",
                clone_id="scout",
                turn=HeadTurn(prompt="what is in the index?", content="three tables"),
                head="run",
            ).room_id
            before = stack.service.get(room_id)
            url = path.format(room_id=room_id, participant_id="scout", seq=1)

            answer = client.request(method, url, json={})

            if (method, path) == ("DELETE", "/api/rooms/{room_id}"):
                assert answer.status_code == 204, (method, path, answer.text)
                continue
            assert answer.status_code == 409, (method, path, answer.text)
            assert answer.json()["detail"] == (
                "This conversation belongs to ucx run, so only ucx run can continue it. "
                "You can read it here."
            )
            assert stack.service.get(room_id) == before, (method, path)


class TestTheAppAndTheCliShareOneRoomStore:
    """`RoomStack` resolves the room folder as the CLI does (#1885)."""

    def test_over_the_default_session_root_it_honours_the_room_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise the app lists a folder no head writes to while `UCLONE_ROOM_DIR` is set.

        Killed by: src/uclone_x/room/store.py :: return default_room_storage_dir()
        Becomes: return session_root / ROOMS_SUBDIR
        """
        from uclone_x.room.store import ROOM_STORAGE_DIR_ENV_VAR, RoomStore

        elsewhere = tmp_path / "rooms-elsewhere"
        monkeypatch.setenv(ROOM_STORAGE_DIR_ENV_VAR, str(elsewhere))
        app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector())
        stack = cast(Any, app).state.room_stack

        assert stack.store.storage_dir == elsewhere
        assert stack.store.storage_dir == RoomStore().storage_dir

    def test_a_storage_folder_of_its_own_keeps_its_rooms_inside_it(self, tmp_path: Path) -> None:
        """An app pointed at a folder keeps that folder whole (author's choice).

        Killed by: src/uclone_x/room/store.py :: if session_root.resolve() == default_session_root().resolve():
        Becomes: if True:
        """
        app = create_ui_app(
            static_dir=tmp_path, storage_dir=tmp_path / "sessions", llm=MockLLMConnector()
        )
        stack = cast(Any, app).state.room_stack

        assert stack.store.storage_dir == (tmp_path / "sessions").resolve() / "rooms"


def test_a_workspace_switch_is_saved_and_imports_that_folders_personas(
    client: TestClient, tmp_path: Path
) -> None:
    """The room keeps the folder's full path; its personas become clones (§3.8 step 2).

    Killed by: src/uclone_x/ui/rooms.py :: ensure_clone_store(workspace, builtin_dir=None, install=False)
    Becomes: None
    """
    from uclone_x.agent.clone_store import clone_handles
    from uclone_x.core.agent_home import default_agents_root

    project = tmp_path / "project"
    folder = project / ".uclone" / "personas"
    folder.mkdir(parents=True)
    (folder / "elsewhere.yaml").write_text(
        "name: elsewhere\nrole: Helper\nsystem_prompt: I am elsewhere.\n", encoding="utf-8"
    )
    room_id = _create(client).json()["room_id"]

    switched = client.patch(f"/api/rooms/{room_id}/workspace", json={"workspace": str(project)})

    assert switched.status_code == 200
    assert client.get(f"/api/rooms/{room_id}").json()["workspace"] == str(project.resolve())
    assert "elsewhere" in clone_handles(default_agents_root())


def test_a_refused_workspace_says_which_refusal_it_is(client: TestClient, tmp_path: Path) -> None:
    """The head words a refusal from its `code`; the English sentence is the fallback.

    Killed by: src/uclone_x/ui/rooms.py :: return JSONResponse({"detail": str(exc), "code": exc.code}, status_code=400)
    Becomes: raise _http_error(exc) from exc
    """
    room_id = _create(client).json()["room_id"]

    refused = client.patch(
        f"/api/rooms/{room_id}/workspace", json={"workspace": str(tmp_path / "missing")}
    )

    assert refused.status_code == 400
    assert refused.json()["code"] == "not_a_folder"
    assert "is not an existing folder" in refused.json()["detail"]


def test_every_workspace_refusal_has_words_in_the_head() -> None:
    """A code the Core can send and the head cannot word would show English on a Korean
    screen, or nothing; each code is a key of `conversation.workspace.refusals`."""
    import json
    from typing import get_args

    from uclone_x.errors import WorkspaceRefusalCode

    for locale in ("en", "ko"):
        copy = json.loads(
            (
                REPO_ROOT / "frontend" / "src" / "i18n" / "locales" / locale / "conversation.json"
            ).read_text(encoding="utf-8")
        )
        assert set(copy["workspace"]["refusals"]) == set(get_args(WorkspaceRefusalCode)), locale


def test_another_sites_page_cannot_move_a_rooms_workspace(
    client: TestClient, tmp_path: Path
) -> None:
    """It moves the file tools' write bound, so a cross-origin request is refused.

    Killed by: src/uclone_x/ui/rooms.py :: refuse_cross_origin(request)  # another tab must not move the file tools' write bound
    Becomes: pass
    """
    room_id = _create(client).json()["room_id"]

    moved = client.patch(
        f"/api/rooms/{room_id}/workspace",
        json={"workspace": str(tmp_path)},
        headers={"origin": "https://evil.example"},
    )

    assert moved.status_code == 403
    assert client.get(f"/api/rooms/{room_id}").json()["workspace"] is None
