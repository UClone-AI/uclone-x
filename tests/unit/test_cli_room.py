"""Tests for `ucx room` — the first user-reachable path into a multi-agent room.

The CLI is a shell (P8): every assertion below is about what landed in the **store**, not
about what was printed, because the command's job is to call the Core and render the
result. A test that only checked the output would pass over a command that computed the
roster itself.

`say` is exercised against a stub orchestrator factory rather than a live model: what the
command owes its caller is that it drives `RoomOrchestrator.post` with the right arguments
and renders what came back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from uclone_x.cli.commands import room as room_cmd
from uclone_x.room.models import ParticipantKind, RoomMessageKind, RoomState
from uclone_x.room.store import RoomStore

runner = CliRunner()


@pytest.fixture
def rooms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RoomStore:
    """Point the command group's default store at a temporary directory."""
    monkeypatch.setenv("UCLONE_ROOM_DIR", str(tmp_path / "rooms"))
    return RoomStore(tmp_path / "rooms")


def _run(*args: str) -> Any:
    return runner.invoke(room_cmd.room_app, list(args))


class TestCreateAndList:
    def test_create_writes_a_titled_room(self, rooms: RoomStore) -> None:
        result = _run("create", "Cache strategy", "--id", "room_cli")
        assert result.exit_code == 0, result.output

        state = rooms.load("room_cli")
        assert state is not None
        assert state.title == "Cache strategy"
        assert "room_cli" in result.output

    def test_create_seats_the_participants_it_was_given(self, rooms: RoomStore) -> None:
        result = _run(
            "create",
            "Review",
            "--id",
            "room_seed",
            "--human",
            "alice",
            "--agent",
            "scout",
            "--agent",
            "critic",
            "--responder",
            "scout",
        )
        assert result.exit_code == 0, result.output

        state = rooms.load("room_seed")
        assert state is not None
        assert [(p.id, p.kind) for p in state.participants] == [
            ("alice", ParticipantKind.HUMAN),
            ("scout", ParticipantKind.AGENT),
            ("critic", ParticipantKind.AGENT),
        ]
        assert state.policy.default_responder_id == "scout"
        assert [m.kind for m in state.transcript] == [RoomMessageKind.JOIN] * 3

    def test_create_refuses_a_responder_who_is_not_seated(self, rooms: RoomStore) -> None:
        """A room created against a responder it does not have is broken on arrival.

        `DefaultResponderSelector` raises on a responder that is not an agent of the room,
        so accepting this would produce a room whose very first unaddressed message fails.
        """
        result = _run("create", "Bad", "--id", "room_bad", "--responder", "ghost")
        assert result.exit_code == 1
        assert rooms.load("room_bad") is None, "a refused create must leave nothing behind"

    def test_list_shows_titles_not_only_ids(self, rooms: RoomStore) -> None:
        _run("create", "Cache strategy", "--id", "room_1")
        _run("create", "Retry semantics", "--id", "room_2")

        result = _run("list")

        assert result.exit_code == 0, result.output
        assert "Cache strategy" in result.output
        assert "Retry semantics" in result.output

    def test_list_says_so_when_there_are_no_rooms(self, rooms: RoomStore) -> None:
        result = _run("list")
        assert result.exit_code == 0
        assert "no rooms" in result.output.lower()


class TestShow:
    def test_show_renders_the_transcript_and_the_roster(self, rooms: RoomStore) -> None:
        _run("create", "Room", "--id", "room_s", "--human", "alice", "--agent", "scout")
        result = _run("show", "room_s")

        assert result.exit_code == 0, result.output
        assert "alice" in result.output
        assert "scout" in result.output

    def test_show_exits_nonzero_for_an_unknown_room(self, rooms: RoomStore) -> None:
        result = _run("show", "room_nope")
        assert result.exit_code == 1
        assert "room_nope" in result.output


def _provenance(model: str, requested: str | None = None) -> Any:
    """A `PRIMARY` provenance served by `model`, optionally under a different request."""
    from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef

    return Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="anthropic", model=requested or model),
        served_by=ServiceRef(provider="anthropic", model=model),
    )


def _seed(rooms: RoomStore, room_id: str, *rows: Any, decision: Any = None) -> RoomState:
    """Write a room whose transcript is exactly `rows`, and its closing decision."""
    from uclone_x.room.models import Participant

    state = RoomState(
        room_id=room_id,
        title="Cache strategy",
        participants=(
            Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice"),
            Participant(id="scout", kind=ParticipantKind.AGENT, display_name="Scout"),
        ),
        transcript=tuple(rows),
        last_decision=decision,
    )
    return rooms.save(state)


class TestShowMakesTheRecordedOutcomesVisible:
    """The three things the Core records faithfully and no surface rendered.

    Each was, to a reader of `ucx room show`, indistinguishable from the room having done
    nothing at all — which is the failure P6 forbids, reintroduced at the one place a user
    actually looks.
    """

    def test_show_names_the_model_that_answered(self, rooms: RoomStore) -> None:
        """`Provenance` rides on the message; the head renders the *served* model.

        Never what the text claims to be, and never the requested model: P6 makes the two
        legally different on a `primary` path, and the difference is what an operator
        needs when routing degrades.

        Killed by: src/uclone_x/cli/commands/room.py :: _service_label(provenance.served_by)
        Becomes: _service_label(provenance.requested)
        """
        from uclone_x.room.models import RoomMessage

        _seed(
            rooms,
            "room_prov",
            RoomMessage(seq=1, sender_id="alice", content="caching?"),
            RoomMessage(
                seq=2,
                sender_id="scout",
                content="TTL is simplest",
                provenance=_provenance("claude-3-haiku", requested="claude-3-opus"),
            ),
        )

        result = _run("show", "room_prov")

        assert result.exit_code == 0, result.output
        assert "claude-3-haiku" in result.output

    def test_show_names_the_model_that_chose_the_speaker(self, rooms: RoomStore) -> None:
        """Attribution is twofold: who answered, and who decided that they should."""
        from uclone_x.room.models import RoomMessage, SelectionVerdict, SpeakerDecision

        _seed(
            rooms,
            "room_dec",
            RoomMessage(seq=1, sender_id="alice", content="caching?"),
            RoomMessage(
                seq=2,
                sender_id="scout",
                content="TTL is simplest",
                decision=SpeakerDecision(
                    verdict=SelectionVerdict.SPEAK,
                    speaker_id="scout",
                    selector="llm",
                    reasoning="scout reads code",
                    provenance=_provenance("claude-3-5-sonnet"),
                ),
            ),
        )

        result = _run("show", "room_dec")

        assert result.exit_code == 0, result.output
        assert "llm" in result.output
        assert "claude-3-5-sonnet" in result.output

    def test_show_renders_a_decided_silence_and_its_reason(self, rooms: RoomStore) -> None:
        """A room that chose to be quiet must not read as a room that stopped.

        Killed by: src/uclone_x/cli/commands/room.py :: escape(decision.reasoning)
        Becomes: decision.verdict.value
        """
        from uclone_x.room.models import RoomMessage, SelectionVerdict, SpeakerDecision

        _seed(
            rooms,
            "room_quiet",
            RoomMessage(seq=1, sender_id="alice", content="thanks all"),
            decision=SpeakerDecision(
                verdict=SelectionVerdict.SILENCE,
                selector="orchestrator",
                reasoning=(
                    "the selector chain was exhausted without a judgement "
                    "(abstained: mention, default-responder)"
                ),
            ),
        )

        result = _run("show", "room_quiet")

        assert result.exit_code == 0, result.output
        assert "exhausted without a judgement" in result.output
        assert "orchestrator" in result.output

    def test_show_offers_the_retry_for_a_failed_turn(self, rooms: RoomStore) -> None:
        """The failure is shown *and* the way out of it is named, in the same place.

        Rendering the error without the remedy leaves the user where the issue found
        them: retyping the question, which spends a fresh turn budget on a turn that was
        already paid for.

        Killed by: src/uclone_x/cli/commands/room.py :: f"[dim]— retry that turn with:[/dim] ucx room retry {escape(state.room_id)}"
        Becomes: ""
        """
        from uclone_x.room.models import RoomMessage

        _seed(
            rooms,
            "room_fail",
            RoomMessage(seq=1, sender_id="alice", content="caching?"),
            RoomMessage(seq=2, sender_id="scout", content="", error="RuntimeError: boom"),
        )

        result = _run("show", "room_fail")

        assert result.exit_code == 0, result.output
        assert "RuntimeError: boom" in result.output
        assert "room retry" in result.output
        assert "room_fail" in result.output

    def test_show_does_not_offer_a_retry_when_nothing_failed(self, rooms: RoomStore) -> None:
        """An offer the Core would refuse is worse than no offer."""
        from uclone_x.room.models import RoomMessage

        _seed(
            rooms,
            "room_ok",
            RoomMessage(seq=1, sender_id="alice", content="caching?"),
            RoomMessage(seq=2, sender_id="scout", content="TTL is simplest"),
        )

        result = _run("show", "room_ok")

        assert result.exit_code == 0, result.output
        assert "room retry" not in result.output


class TestRetryCommand:
    def test_retry_drives_the_core_and_renders_what_came_back(
        self, rooms: RoomStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The command is a shell: it calls `retry`, and shows the rows that appeared.

        Killed by: src/uclone_x/cli/commands/room.py :: asyncio.run(orchestrator.retry(room_id))
        Becomes: before
        """
        from uclone_x.room.models import RoomMessage

        _seed(
            rooms,
            "room_r",
            RoomMessage(seq=1, sender_id="alice", content="caching?"),
            RoomMessage(seq=2, sender_id="scout", content="", error="RuntimeError: boom"),
        )
        retried: list[str] = []

        class StubOrchestrator:
            async def retry(self, room_id: str) -> RoomState:
                retried.append(room_id)
                state = room_cmd.build_service().get(room_id)
                return state.model_copy(
                    update={
                        "transcript": (
                            *state.transcript,
                            RoomMessage(seq=3, sender_id="scout", content="TTL is simplest"),
                        )
                    }
                )

        def _stub(**_kwargs: object) -> StubOrchestrator:
            return StubOrchestrator()

        monkeypatch.setattr(room_cmd, "build_orchestrator", _stub)

        result = _run("retry", "room_r")

        assert result.exit_code == 0, result.output
        assert retried == ["room_r"]
        assert "TTL is simplest" in result.output

    def test_retry_reports_the_core_refusal_rather_than_a_traceback(
        self, rooms: RoomStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from uclone_x.errors import NothingToRetryError
        from uclone_x.room.models import RoomMessage

        _seed(rooms, "room_rn", RoomMessage(seq=1, sender_id="alice", content="caching?"))

        class Refusing:
            async def retry(self, room_id: str) -> RoomState:
                raise NothingToRetryError("the last turn in 'room_rn' did not fail")

        def _stub(**_kwargs: object) -> Refusing:
            return Refusing()

        monkeypatch.setattr(room_cmd, "build_orchestrator", _stub)

        result = _run("retry", "room_rn")

        assert result.exit_code == 1
        assert "did not fail" in result.output


class TestRoster:
    def test_add_seats_an_agent_with_its_own_session(self, rooms: RoomStore) -> None:
        _run("create", "Room", "--id", "room_a")
        result = _run("add", "room_a", "scout", "--persona", "reads code")

        assert result.exit_code == 0, result.output
        state = rooms.load("room_a")
        assert state is not None
        scout = next(p for p in state.participants if p.id == "scout")
        assert scout.session_id == "sess_room__room_a__scout"
        assert scout.persona_summary == "reads code"

    def test_add_can_seat_a_human(self, rooms: RoomStore) -> None:
        _run("create", "Room", "--id", "room_ah")
        _run("add", "room_ah", "alice", "--human")

        state = rooms.load("room_ah")
        assert state is not None
        assert state.participants[0].kind is ParticipantKind.HUMAN

    def test_remove_records_the_departure(self, rooms: RoomStore) -> None:
        _run("create", "Room", "--id", "room_r", "--agent", "critic")
        result = _run("remove", "room_r", "critic")

        assert result.exit_code == 0, result.output
        state = rooms.load("room_r")
        assert state is not None
        assert state.participants == ()
        assert state.transcript[-1].kind is RoomMessageKind.LEAVE

    def test_remove_exits_nonzero_for_someone_who_is_not_there(self, rooms: RoomStore) -> None:
        _run("create", "Room", "--id", "room_rn")
        result = _run("remove", "room_rn", "ghost")
        assert result.exit_code == 1


class TestSay:
    def test_say_drives_the_orchestrator_and_renders_the_reply(
        self, rooms: RoomStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The command is a shell: it posts, and shows what the Core returned.

        Killed by: src/uclone_x/cli/commands/room.py :: asyncio.run(orchestrator.post(room_id, sender_id, message))
        Becomes: before
        """
        _run("create", "Room", "--id", "room_say", "--human", "alice", "--agent", "scout")
        posted: list[tuple[str, str, str]] = []

        class StubOrchestrator:
            async def post(self, room_id: str, sender_id: str, content: str) -> RoomState:
                posted.append((room_id, sender_id, content))
                service = room_cmd.build_service()
                state = service.get(room_id)
                message = state.transcript
                from uclone_x.room.models import RoomMessage

                return state.model_copy(
                    update={
                        "transcript": (
                            *message,
                            RoomMessage(seq=len(message) + 1, sender_id="alice", content=content),
                            RoomMessage(
                                seq=len(message) + 2,
                                sender_id="scout",
                                content="TTL is simplest",
                            ),
                        )
                    }
                )

        def _stub(**_kwargs: object) -> StubOrchestrator:
            return StubOrchestrator()

        monkeypatch.setattr(room_cmd, "build_orchestrator", _stub)

        result = _run("say", "room_say", "alice", "what about caching?")

        assert result.exit_code == 0, result.output
        assert posted == [("room_say", "alice", "what about caching?")]
        assert "TTL is simplest" in result.output

    def test_say_exits_nonzero_when_the_sender_is_not_in_the_room(
        self, rooms: RoomStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A room error is reported, not raised as a traceback at the user."""
        from uclone_x.errors import UnknownRoomParticipantError

        _run("create", "Room", "--id", "room_bad_sender", "--agent", "scout")

        class Failing:
            async def post(self, room_id: str, sender_id: str, content: str) -> RoomState:
                raise UnknownRoomParticipantError(f"{sender_id!r} is not a participant")

        def _stub(**_kwargs: object) -> Failing:
            return Failing()

        monkeypatch.setattr(room_cmd, "build_orchestrator", _stub)

        result = _run("say", "room_bad_sender", "mallory", "hi")

        assert result.exit_code == 1
        assert "mallory" in result.output


class TestWiring:
    def test_the_group_is_registered_on_the_root_app(self) -> None:
        """`ucx room` must be reachable; the group existing in a module is not the feature."""
        from uclone_x.cli import main

        result = runner.invoke(main.app, ["room", "--help"])

        assert result.exit_code == 0, result.output
        for command in ("create", "list", "show", "add", "remove", "say", "retry"):
            assert command in result.output


def _loaded(rooms: RoomStore) -> Any:
    """The one room the test just created."""
    ids = rooms.list_room_ids()
    assert len(ids) == 1, ids
    state = rooms.load(ids[0])
    assert state is not None
    return state


class TestCreateSeatsRoutableAgents:
    """A room built in one command must be able to have both personas and a responder.

    `--persona` lived only on `add`, and `--responder` may only name an agent seated by
    `create`, so the two were mutually exclusive: an agent seated at creation had an empty
    `persona_summary` — which `Participant` calls a routing input and not decoration, since
    a selector reasoning about fit cannot route to a participant it knows nothing about —
    and giving it one meant giving up the responder.
    """

    def test_create_accepts_a_persona_for_a_seated_agent(self, rooms: RoomStore) -> None:
        result = _run("create", "R", "--agent", "scout", "--persona", "scout=explores the codebase")
        assert result.exit_code == 0, result.output
        state = _loaded(rooms)
        scout = next(p for p in state.participants if p.id == "scout")
        assert scout.persona_summary == "explores the codebase"

    def test_create_accepts_an_alias(self, rooms: RoomStore) -> None:
        result = _run("create", "R", "--agent", "critic", "--alias", "critic=reviewer")
        assert result.exit_code == 0, result.output
        critic = next(p for p in _loaded(rooms).participants if p.id == "critic")
        assert "reviewer" in critic.aliases

    def test_a_persona_for_an_unseated_agent_is_refused_by_name(self, rooms: RoomStore) -> None:
        result = _run("create", "R", "--agent", "scout", "--persona", "critic=reviews things")
        assert result.exit_code != 0
        assert "critic" in result.output

    def test_a_persona_without_the_keyed_form_is_refused(self, rooms: RoomStore) -> None:
        result = _run("create", "R", "--agent", "scout", "--persona", "explores")
        assert result.exit_code != 0
        assert "AGENT=" in result.output

    def test_personas_and_a_responder_can_be_given_together(self, rooms: RoomStore) -> None:
        """The combination that was impossible, and the reason this exists."""
        result = _run(
            "create",
            "R",
            "--human",
            "kenny",
            "--agent",
            "scout",
            "--persona",
            "scout=explores",
            "--agent",
            "critic",
            "--persona",
            "critic=reviews",
            "--alias",
            "critic=reviewer",
            "--responder",
            "scout",
        )
        assert result.exit_code == 0, result.output
        state = _loaded(rooms)
        assert state.policy.default_responder_id == "scout"
        assert {p.persona_summary for p in state.participants if p.kind.value == "agent"} == {
            "explores",
            "reviews",
        }


class TestTheResponderCanBeChangedAfterCreation:
    """`remove` clears the responder, and nothing could set one again."""

    def test_responder_names_an_agent(self, rooms: RoomStore) -> None:
        _run("create", "R", "--agent", "scout", "--agent", "critic")
        room = _loaded(rooms).room_id
        result = _run("responder", room, "critic")
        assert result.exit_code == 0, result.output
        assert _loaded(rooms).policy.default_responder_id == "critic"

    def test_responder_clear_leaves_the_room_without_one(self, rooms: RoomStore) -> None:
        _run("create", "R", "--agent", "scout", "--responder", "scout")
        room = _loaded(rooms).room_id
        result = _run("responder", room, "--clear")
        assert result.exit_code == 0, result.output
        assert _loaded(rooms).policy.default_responder_id == ""

    def test_naming_someone_who_is_not_a_seated_agent_is_refused(self, rooms: RoomStore) -> None:
        _run("create", "R", "--agent", "scout", "--human", "kenny")
        room = _loaded(rooms).room_id
        for who in ("ghost", "kenny"):
            result = _run("responder", room, who)
            assert result.exit_code != 0, who
            assert who in result.output


class TestRoomCommandsSpeakOfRooms:
    """A room's user has no notion of a session, and should not be shown one."""

    def test_a_blank_room_id_is_refused_in_the_room_s_own_vocabulary(self, tmp_path: Any) -> None:
        result = _run("show", "")
        assert result.exit_code != 0
        assert "session" not in result.output.lower(), result.output
        assert "room" in result.output.lower()


class TestUserTextIsRenderedNotInterpreted:
    """Nothing a person or an agent wrote is a rendering instruction.

    Every line this module prints is an f-string handed to `Console.print`, which parses
    `[...]` as markup. A title, a room id, a participant id, a transcript row and a Core
    refusal are all text somebody else chose, so square brackets in any of them are read
    as tags: balanced ones silently restyle and **delete** the characters that were
    written, and an unbalanced one raises `MarkupError` — a traceback in place of the
    record, at the one surface a user has for reading the room back.
    """

    def test_a_message_that_looks_like_markup_is_shown_as_written(self, rooms: RoomStore) -> None:
        """`[/]` in a transcript row must not end the room's only reader.

        An agent citing `[1]`, quoting a template, or closing a tag is ordinary output.
        Rendered as markup it either vanishes or raises, and the raise is permanent: the
        row is in the store, and no `ucx room` command can then read the room at all.

        Killed by: src/uclone_x/cli/commands/room.py :: f"{escape(message.content)}"
        Becomes: f"{message.content}"
        """
        from uclone_x.room.models import RoomMessage

        _seed(
            rooms,
            "room_markup",
            RoomMessage(seq=1, sender_id="alice", content="use [/] to close"),
            RoomMessage(seq=2, sender_id="scout", content="call [bold]now[/bold]"),
        )

        result = _run("show", "room_markup")

        assert result.exit_code == 0, result.output
        assert "use [/] to close" in result.output
        assert "call [bold]now[/bold]" in result.output

    def test_a_title_that_looks_like_markup_is_stored_and_echoed_whole(
        self, rooms: RoomStore
    ) -> None:
        """The confirmation names the room the user actually made."""
        result = _run("create", "TTL [/red] rules", "--id", "room_t")

        assert result.exit_code == 0, result.output
        state = rooms.load("room_t")
        assert state is not None
        assert state.title == "TTL [/red] rules"
        assert "TTL [/red] rules" in result.output

    def test_one_odd_title_does_not_take_the_whole_listing_down(self, rooms: RoomStore) -> None:
        """The listing is the only way to find any of the *other* rooms.

        `RoomService.list_rooms` already refuses to let one unreadable record cost the
        survey; a renderer that raises on one row gives that property back.

        Killed by: src/uclone_x/cli/commands/room.py :: escape(s.title)
        Becomes: s.title
        """
        rooms.save(RoomState(room_id="room_odd", title="TTL [/red] rules"))
        _run("create", "Retry semantics", "--id", "room_fine")

        result = _run("list")

        assert result.exit_code == 0, result.output
        assert "Retry semantics" in result.output

    def test_an_unknown_room_whose_id_looks_like_markup_is_reported(self, rooms: RoomStore) -> None:
        """A refusal must survive the text it quotes back.

        Killed by: src/uclone_x/cli/commands/room.py :: {escape(message)}
        Becomes: {message}
        """
        result = _run("show", "[bold]ghost")

        assert result.exit_code == 1
        assert "[bold]ghost" in result.output, "the refusal must quote the id as typed"


class TestCreateLeavesNothingBehindWhenItRefuses:
    """`create` is one command to a user, and must be one outcome.

    It writes the room first and seats the roster afterwards, one call at a time, so a
    refusal at any seat exits non-zero over a room that already exists and is already
    half-seated. The id is then taken, and the corrected command — the obvious next thing
    to type — is refused as a duplicate, leaving the user with a room they did not ask for
    and no way to make the one they did under the name they chose.
    """

    def test_a_second_human_leaves_no_room_behind(self, rooms: RoomStore) -> None:
        """Killed by: src/uclone_x/cli/commands/room.py :: service.delete(state.room_id)
        Becomes: pass
        """
        result = _run("create", "Pairing", "--id", "room_2h", "--human", "alice", "--human", "bob")

        assert result.exit_code == 1
        assert rooms.load("room_2h") is None, "a refused create must leave nothing behind"

    def test_the_same_agent_twice_is_refused_and_leaves_no_room(self, rooms: RoomStore) -> None:
        result = _run("create", "Dup", "--id", "room_dup", "--agent", "scout", "--agent", "scout")

        assert result.exit_code == 1
        assert "scout" in result.output
        assert rooms.load("room_dup") is None

    def test_one_id_cannot_be_both_the_human_and_an_agent(self, rooms: RoomStore) -> None:
        """Two seats under one id is the collision `Participant.id` exists to prevent."""
        result = _run("create", "Same", "--id", "room_same", "--human", "al", "--agent", "al")

        assert result.exit_code == 1
        assert rooms.load("room_same") is None


class TestTheTurnCeilingArgument:
    """`--max-turns` is a number a user types, so its refusals belong to the flag."""

    def test_a_negative_ceiling_is_refused_with_a_message(self, rooms: RoomStore) -> None:
        """Killed by: src/uclone_x/cli/commands/room.py :: except ValidationError as exc:
        Becomes: except RoomError as exc:
        """
        result = _run("create", "Bad", "--id", "room_neg", "--max-turns", "-1")

        assert result.exit_code == 1
        assert "--max-turns" in result.output
        assert rooms.load("room_neg") is None

    def test_a_ceiling_the_selector_window_cannot_hold_is_refused(self, rooms: RoomStore) -> None:
        """A ceiling at or above `transcript_window` starves the addressing message.

        The Core refuses the pair; the CLI owed that refusal a sentence rather than a
        pydantic traceback.
        """
        result = _run("create", "Wide", "--id", "room_wide", "--max-turns", "20")

        assert result.exit_code == 1
        assert "--max-turns" in result.output
        assert rooms.load("room_wide") is None


class TestTailArgument:
    def test_a_negative_tail_is_refused_rather_than_silently_meaning_all(
        self, rooms: RoomStore
    ) -> None:
        """`--tail -5` reads as "five rows"; rendering the whole room instead is a lie.

        Killed by: src/uclone_x/cli/commands/room.py :: if tail < 0:
        Becomes: if False:
        """
        _run("create", "Room", "--id", "room_tail")

        result = _run("show", "room_tail", "--tail", "-5")

        assert result.exit_code == 1
        assert "--tail" in result.output


class TestSettingUpTheTurnIsReportedLikeAnyOtherRefusal:
    """`say` and `retry` build an orchestrator before they enter their `try`.

    Assembling it resolves a model client, so a mistyped `--provider`, an unreachable
    endpoint or a missing configuration — the ordinary ways this command is used wrongly —
    reach the user as a traceback rather than as the one-line refusal `_fail` exists for.
    """

    def test_say_reports_a_provider_failure_rather_than_a_traceback(
        self, rooms: RoomStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killed by: src/uclone_x/cli/commands/room.py :: except UCloneXError as setup_failure:
        Becomes: except RoomError as setup_failure:
        """
        from uclone_x.errors import LLMProviderError

        _run("create", "Room", "--id", "room_prov_fail", "--human", "alice", "--agent", "scout")

        def _boom(**_kwargs: object) -> object:
            raise LLMProviderError("Unsupported LLM provider: bogus")

        monkeypatch.setattr(room_cmd, "build_orchestrator", _boom)

        result = _run("say", "room_prov_fail", "alice", "hi", "--provider", "bogus")

        assert result.exit_code == 1
        assert "Unsupported LLM provider: bogus" in result.output

    def test_retry_reports_a_provider_failure_rather_than_a_traceback(
        self, rooms: RoomStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from uclone_x.errors import LLMProviderError

        _run("create", "Room", "--id", "room_prov_retry", "--agent", "scout")

        def _boom(**_kwargs: object) -> object:
            raise LLMProviderError("no Ollama endpoint is configured")

        monkeypatch.setattr(room_cmd, "build_orchestrator", _boom)

        result = _run("retry", "room_prov_retry")

        assert result.exit_code == 1
        assert "no Ollama endpoint is configured" in result.output


class TestASilenceWithNothingToSayForItself:
    def test_a_silence_with_no_reasoning_does_not_render_a_dangling_colon(
        self, rooms: RoomStore
    ) -> None:
        """`reasoning` is required by no validator, so the renderer must survive an empty one.

        Killed by: src/uclone_x/cli/commands/room.py :: if decision.reasoning else " "
        Becomes: if decision.reasoning else ": "
        """
        from uclone_x.room.models import RoomMessage, SelectionVerdict, SpeakerDecision

        _seed(
            rooms,
            "room_mute",
            RoomMessage(seq=1, sender_id="alice", content="thanks all"),
            decision=SpeakerDecision(
                verdict=SelectionVerdict.SILENCE, selector="llm", reasoning=""
            ),
        )

        result = _run("show", "room_mute")

        assert result.exit_code == 0, result.output
        assert "the room fell quiet" in result.output
        assert "quiet: " not in result.output


class TestTheRoomIsDescribedInItsUsersVocabulary:
    def test_show_does_not_report_the_rooms_write_counter(self, rooms: RoomStore) -> None:
        """`revision` is the store's compare-and-swap bookkeeping, not a room's property.

        A person reading a conversation back has no notion of one, and the header offered
        no explanation of what the number counted — so it read as a version of the
        conversation, which is the one thing it is not.

        Killed by: src/uclone_x/cli/commands/room.py :: {escape(state.room_id)} · updated {state.updated_at}
        Becomes: {escape(state.room_id)} · rev {state.revision} · updated {state.updated_at}
        """
        from uclone_x.room.models import RoomMessage

        _seed(rooms, "room_vocab", RoomMessage(seq=1, sender_id="alice", content="hello"))

        result = _run("show", "room_vocab")

        assert result.exit_code == 0, result.output
        assert "rev " not in result.output
        assert "revision" not in result.output

    def test_show_traversal_explains_itself_in_room_vocabulary(self, rooms: RoomStore) -> None:
        """`ucx room show 'a/b'` must explain itself in room vocabulary, not session vocabulary.

        Killed by: src/uclone_x/room/store.py :: .replace("session ID", "room id")
        Becomes:
        """
        result = _run("show", "a/b")

        assert result.exit_code == 1
        assert "session" not in result.output.lower()
        assert "room id" in result.output.lower()


class TestTheCliSeatsAgentsWithMemory:
    def test_build_orchestrator_gives_each_participant_its_own_store(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """`ucx room say` is a head too, and it wired no memory either.

        `HostDependencies` is built here with no `memory=` on purpose — a single store
        there is handed to every seat — so the per-participant factory is what gives a
        room agent a memory at all.

        Killed by: src/uclone_x/cli/commands/room.py :: memory_factory=default_cross_session_memory,
        Becomes: memory_factory=None,
        """
        from uclone_x.cli.commands.room import build_orchestrator
        from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
        from uclone_x.room.models import RoomPolicy
        from uclone_x.room.resolver import RoomAgentResolver
        from uclone_x.room.store import RoomStore

        monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))
        orchestrator = build_orchestrator(
            store=RoomStore(tmp_path / "rooms"), policy=RoomPolicy(), provider="mock"
        )

        resolver = cast(
            "RoomAgentResolver",
            orchestrator._resolver,  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType]
        )
        factory = resolver._memory_factory  # pyright: ignore[reportPrivateUsage]
        assert factory is not None

        alpha, again, beta = factory("alpha"), factory("alpha"), factory("beta")
        assert alpha.storage_path == again.storage_path
        assert alpha.storage_path != beta.storage_path
        assert alpha.storage_path is not None
        assert str(tmp_path) in str(alpha.storage_path)
