"""Tests for room lifecycle: creation, the roster, and the record that explains both.

Written before `uclone_x.room.service` existed. What they pin, in order of how much they
would cost to get wrong:

* **A membership event is not an utterance.** A join or a leave lands in the transcript so
  that "why did critic stop replying?" is answerable from the conversation alone — and it
  must not be readable as somebody *talking*. Three consumers read the transcript looking
  for speech (the span an agent is handed, the interjection check, the mention scan), and
  a membership line that slipped past any of them would put words in a participant's mouth
  or make the orchestrator believe a human had cut in.
* **A room has a name.** `list_room_ids()` returns hex; a person reopening yesterday's
  room recognises a title.
* **Removing the designated responder does not break the room.** The policy field would
  otherwise keep naming a departed agent, and `DefaultResponderSelector` raises on that —
  so a leave would turn every later unaddressed message into an error.
* **Isolation is derived, not trusted.** The service stamps each agent's own session id
  and ontology namespace, so G3/G4 are properties of the record rather than of whoever
  assembled it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.support.clones import make_clone
from uclone_x.errors import (
    RoomAlreadyExistsError,
    RoomError,
    RoomIdError,
    RoomNotFoundError,
    SecondHumanInRoomError,
    StaleRoomWriteError,
    UnknownRoomParticipantError,
    UnreadableRoomRecordError,
)
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomMessage,
    RoomMessageKind,
    RoomPolicy,
    RoomState,
)
from uclone_x.room.service import (
    RoomService,
    participant_session_id,
    seat_sessions,
)
from uclone_x.room.store import RoomStore


@pytest.fixture
def service(tmp_path: Path) -> RoomService:
    return RoomService(RoomStore(tmp_path / "rooms"))


def _stamp_updated_at(store: RoomStore, room_id: str, updated_at: str) -> None:
    """Set a stored room's `updated_at` directly; `RoomStore.save` always stamps *now*."""
    path = store.room_path(room_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["updated_at"] = updated_at
    path.write_text(json.dumps(raw), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Creation and identity
# --------------------------------------------------------------------------------------


class TestCreate:
    def test_create_persists_a_titled_room(self, service: RoomService) -> None:
        state = service.create("Cache strategy review")

        assert state.title == "Cache strategy review"
        assert state.revision == 1, "a created room is on disk, not in hand"
        reloaded = service.get(state.room_id)
        assert reloaded.title == "Cache strategy review"
        assert reloaded.participants == ()
        assert reloaded.transcript == ()

    def test_create_accepts_an_explicit_id_and_policy(self, service: RoomService) -> None:
        policy = RoomPolicy(max_agent_turns_per_human_message=7, transcript_window=9)
        state = service.create("Named", room_id="room_fixed", policy=policy)

        assert state.room_id == "room_fixed"
        assert service.get("room_fixed").policy.max_agent_turns_per_human_message == 7

    def test_create_refuses_a_blank_title(self, service: RoomService) -> None:
        """A blank title is the state the field exists to end, so it is refused at birth.

        `RoomState.title` defaults to empty *only* so that records written before the field
        existed still load. Letting the service mint a new one that way would reintroduce
        the opaque-id problem while looking like a feature.

        Killed by: src/uclone_x/room/service.py :: if not cleaned:
        Becomes: if False:
        """
        with pytest.raises(RoomError, match="title"):
            service.create("   ")

    def test_create_refuses_newlines_in_title(self, service: RoomService) -> None:
        """A newline breaks single-line table formatting and confirmation lines.

        Killed by: src/uclone_x/room/service.py :: if "\n" in cleaned or "\r" in cleaned:
        Becomes: if False:
        """
        with pytest.raises(RoomError, match="newlines"):
            service.create("Line 1\nLine 2")
        with pytest.raises(RoomError, match="newlines"):
            service.create("Line 1\rLine 2")

    def test_create_refuses_ansi_escapes_in_title(self, service: RoomService) -> None:
        r"""ANSI escape sequences store raw escape codes that alter terminal display.

        Killed by: src/uclone_x/room/service.py :: if re.search(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", cleaned):
        Becomes: if False:
        """
        with pytest.raises(RoomError, match="ANSI escape"):
            service.create("Cache \x1b[31mred")

    def test_create_refuses_to_overwrite_an_existing_room(self, service: RoomService) -> None:
        service.create("First", room_id="room_dup")
        with pytest.raises(RoomAlreadyExistsError, match="room_dup"):
            service.create("Second", room_id="room_dup")
        assert service.get("room_dup").title == "First"

    def test_list_rooms_reports_titles_and_shape(self, service: RoomService) -> None:
        service.create("Beta room", room_id="room_b")
        service.create("Alpha room", room_id="room_a")
        service.add_participant("room_a", "alice", kind=ParticipantKind.HUMAN)
        service.add_participant("room_a", "scout")

        summaries = {s.room_id: s for s in service.list_rooms()}

        assert set(summaries) == {"room_a", "room_b"}
        assert summaries["room_a"].title == "Alpha room"
        assert summaries["room_a"].agent_ids == ("scout",)
        assert summaries["room_a"].human_ids == ("alice",)
        # Two joins, and both are in the record.
        assert summaries["room_a"].message_count == 2
        # Killed by: src/uclone_x/room/service.py :: utterance_count=sum(1 for m in state.transcript if m.is_utterance),
        # Becomes: utterance_count=len(state.transcript),
        assert summaries["room_a"].utterance_count == 0
        assert summaries["room_b"].agent_ids == ()

    def test_list_rooms_puts_the_most_recently_updated_first(self, tmp_path: Path) -> None:
        """The listing's order is recency, decided in the Core (#1053).

        The ids are chosen so that every order that is *not* recency differs from it: the
        id sort `list_room_ids` hands back is a, b, c, and so is creation order. A listing
        that dropped the sort would therefore read a, b, c and not a, c, b.

        Killed by: src/uclone_x/room/service.py :: rooms=tuple(_by_recency(summaries))
        Becomes: rooms=tuple(summaries)
        """
        store = RoomStore(tmp_path / "rooms")
        service = RoomService(store)
        for room_id in ("room_a", "room_b", "room_c"):
            service.create(room_id.upper(), room_id=room_id)
        _stamp_updated_at(store, "room_a", "2026-09-19T09:00:00+00:00")
        _stamp_updated_at(store, "room_b", "2026-09-17T09:00:00+00:00")
        _stamp_updated_at(store, "room_c", "2026-09-18T09:00:00+00:00")

        assert [s.room_id for s in service.list_rooms()] == ["room_a", "room_c", "room_b"]

    def test_list_rooms_compares_instants_and_not_strings(self, tmp_path: Path) -> None:
        """Two stamps written in different offsets are ordered by the moment they name.

        `10:00+09:00` is 01:00 UTC, four hours *before* `05:00+00:00`, and sorts after it
        as a string. The Core only ever writes UTC, but a record carried over from another
        machine or edited by hand need not, and a string sort would put it wrong silently.

        Killed by: src/uclone_x/room/service.py :: key=lambda s: _updated_instant(s.updated_at)
        Becomes: key=lambda s: s.updated_at
        """
        store = RoomStore(tmp_path / "rooms")
        service = RoomService(store)
        service.create("Seoul", room_id="room_seoul")
        service.create("Utc", room_id="room_utc")
        _stamp_updated_at(store, "room_seoul", "2026-09-19T10:00:00+09:00")
        _stamp_updated_at(store, "room_utc", "2026-09-19T05:00:00+00:00")

        assert [s.room_id for s in service.list_rooms()] == ["room_utc", "room_seoul"]

    def test_list_rooms_breaks_a_tie_by_room_id(self, tmp_path: Path) -> None:
        """Equal stamps order by `room_id`, ascending, so the listing is deterministic.

        `RoomStoreProtocol.list_room_ids` promises no order -- `RoomStore` happens to sort
        -- so the store here lists its ids backwards: a tiebreak that leaned on the
        store's order would pass against `RoomStore` and fail against the next store.

        Killed by: src/uclone_x/room/service.py :: ordered = sorted(summaries, key=lambda s: s.room_id)
        Becomes: ordered = list(summaries)
        """

        class _BackwardsStore(RoomStore):
            def list_room_ids(self) -> tuple[str, ...]:
                return tuple(reversed(super().list_room_ids()))

        store = _BackwardsStore(tmp_path / "rooms")
        service = RoomService(store)
        for room_id in ("room_z", "room_m", "room_a"):
            service.create(room_id.upper(), room_id=room_id)
            _stamp_updated_at(store, room_id, "2026-09-19T09:00:00+00:00")

        assert [s.room_id for s in service.list_rooms()] == ["room_a", "room_m", "room_z"]

    def test_list_rooms_puts_an_unreadable_stamp_last_rather_than_failing(
        self, tmp_path: Path
    ) -> None:
        """A stamp that is not a date costs its room its place, not the listing.

        Same reasoning as the unloadable record below: the listing is how every other room
        is found. A naive stamp is read as UTC rather than refused, for the same reason.
        """
        store = RoomStore(tmp_path / "rooms")
        service = RoomService(store)
        for room_id in ("room_a", "room_b", "room_c"):
            service.create(room_id.upper(), room_id=room_id)
        _stamp_updated_at(store, "room_a", "not a date")
        _stamp_updated_at(store, "room_b", "2026-09-18T09:00:00")
        _stamp_updated_at(store, "room_c", "2026-09-19T09:00:00+00:00")

        assert [s.room_id for s in service.list_rooms()] == ["room_c", "room_b", "room_a"]

    def test_get_refuses_an_unknown_room(self, service: RoomService) -> None:
        with pytest.raises(RoomNotFoundError, match="room_missing"):
            service.get("room_missing")


class TestRename:
    """A title a person can fix.

    `RoomState.title` refuses to be *derived* from the opening utterance -- "an identifier
    that moves is not one" -- so a head that wants a conversation to be findable without
    demanding a title before the user has said anything must seed one once and then let it
    be corrected. Seeding without a repair is worse than no seeding: the first sentence a
    person happened to type becomes the permanent name of the conversation.

    Validation is the same as `create`'s, and that sameness is the point: a title reachable
    only through `rename` could otherwise carry a newline into every single-line table and
    prompt that `create` spends three checks keeping them out of.
    """

    def test_rename_replaces_the_title_and_keeps_the_conversation(
        self, service: RoomService
    ) -> None:
        service.create("Check the index on the users table", room_id="room_seeded")
        service.add_participant("room_seeded", "scout")

        renamed = service.rename("room_seeded", "  Index tuning  ")

        assert renamed.title == "Index tuning"
        # The roster and its record survive: a rename is a label change, not a new room.
        assert [p.id for p in renamed.participants] == ["scout"]
        assert len(renamed.transcript) == 1
        assert service.get("room_seeded").title == "Index tuning"

    def test_rename_refuses_an_unknown_room(self, service: RoomService) -> None:
        with pytest.raises(RoomNotFoundError, match="room_missing"):
            service.rename("room_missing", "Anything")

    @pytest.mark.parametrize(
        ("title", "reason"),
        [
            ("   ", "needs a title"),
            ("two\nlines", "must not contain newlines"),
            ("\x1b[31mred", "ANSI"),
        ],
    )
    def test_rename_applies_the_same_title_rules_as_create(
        self, service: RoomService, title: str, reason: str
    ) -> None:
        service.create("Original", room_id="room_rules")

        with pytest.raises(RoomError, match=reason):
            service.rename("room_rules", title)

        # Refused means unchanged, not partially applied.
        assert service.get("room_rules").title == "Original"


# --------------------------------------------------------------------------------------
# The roster
# --------------------------------------------------------------------------------------


class TestRoster:
    def test_adding_an_agent_derives_its_own_session(self, service: RoomService) -> None:
        """G3 is stamped by the service, not left to whoever builds the record.

        G4's per-seat namespace is gone (clone-knowledge-graph step 6): a clone's rules are
        keyed by its id (`clone_builder.clone_namespace`), whatever room it is in.

        Killed by: src/uclone_x/room/service.py :: f"{SESSION_ID_PREFIX}{SESSION_ID_SEPARATOR}{room_id}{SESSION_ID_SEPARATOR}{participant_id}"
        Becomes: f"{SESSION_ID_PREFIX}{SESSION_ID_SEPARATOR}{room_id}"
        """
        service.create("Room", room_id="room_iso")
        state = service.add_participant("room_iso", "scout", persona_summary="reads code")
        service.add_participant("room_iso", "critic")

        state = service.get("room_iso")
        sessions = {p.id: p.session_id for p in state.participants}

        assert sessions == {
            "scout": "sess_room__room_iso__scout",
            "critic": "sess_room__room_iso__critic",
        }

    def test_a_human_gets_no_session(self, service: RoomService) -> None:
        service.create("Room", room_id="room_h")
        state = service.add_participant("room_h", "alice", kind=ParticipantKind.HUMAN)

        alice = next(p for p in state.participants if p.id == "alice")
        assert alice.kind is ParticipantKind.HUMAN
        assert alice.session_id == ""

    def test_adding_the_same_id_twice_is_refused(self, service: RoomService) -> None:
        service.create("Room", room_id="room_x")
        service.add_participant("room_x", "scout")
        with pytest.raises(RoomError, match="scout"):
            service.add_participant("room_x", "scout")
        assert len(service.get("room_x").participants) == 1

    def test_a_second_human_is_refused_and_the_refusal_says_why(self, service: RoomService) -> None:
        """A room seats one human, and the roster is where that is enforced.

        The model admitted a second one while the runtime could not serialise two posters:
        `post()` loads, appends and saves, so two humans typing at once means the second
        write is refused by the store's compare-and-swap and that user's message simply
        fails. Refusing at the roster turns a runtime surprise into a stated boundary — so
        the message has to carry the reason and name who is already seated, or it is just
        an error where a trap used to be.

        Killed by: src/uclone_x/room/service.py :: if kind is ParticipantKind.HUMAN and seated_human is not None:
        Becomes: if False:
        """
        service.create("Room", room_id="room_2h")
        service.add_participant("room_2h", "alice", kind=ParticipantKind.HUMAN)

        with pytest.raises(SecondHumanInRoomError) as caught:
            service.add_participant("room_2h", "bob", kind=ParticipantKind.HUMAN)

        message = str(caught.value)
        assert "alice" in message, "the refusal must name the human already seated"
        assert "bob" in message
        # The reason, not merely the refusal: a caller told only "no" cannot tell a policy
        # from a bug, and this boundary is a policy.
        assert "one human" in message
        assert len(service.get("room_2h").participants) == 1

    def test_an_agent_is_not_refused_by_the_one_human_rule(self, service: RoomService) -> None:
        """The bound is on humans only; a room's agents are why it exists."""
        service.create("Room", room_id="room_2a")
        service.add_participant("room_2a", "alice", kind=ParticipantKind.HUMAN)
        service.add_participant("room_2a", "scout")
        state = service.add_participant("room_2a", "critic")

        assert [p.id for p in state.participants] == ["alice", "scout", "critic"]

    def test_the_seat_is_free_again_once_the_human_leaves(self, service: RoomService) -> None:
        """A handover is not blocked; only two humans at once are.

        Worth pinning because the cheap way to write the guard — "this room has ever had a
        human" — would read the same on the happy path and make the refusal permanent.
        """
        service.create("Room", room_id="room_hh")
        service.add_participant("room_hh", "alice", kind=ParticipantKind.HUMAN)
        service.remove_participant("room_hh", "alice")

        state = service.add_participant("room_hh", "bob", kind=ParticipantKind.HUMAN)

        assert [p.id for p in state.participants] == ["bob"]

    def test_removing_someone_who_is_not_there_is_refused(self, service: RoomService) -> None:
        service.create("Room", room_id="room_y")
        with pytest.raises(UnknownRoomParticipantError, match="ghost"):
            service.remove_participant("room_y", "ghost")

    def test_remove_drops_the_participant_and_keeps_its_high_water_mark(
        self, service: RoomService
    ) -> None:
        """A departure does not reset what the agent was already shown.

        Its session still holds that conversation, so re-showing it on a rejoin would
        duplicate the room inside the agent's own record.
        """
        service.create("Room", room_id="room_z")
        service.add_participant("room_z", "scout")
        state = service.get("room_z")
        service._store.save(  # pyright: ignore[reportPrivateUsage]
            state.model_copy(update={"last_seen_seq": {"scout": "4"}})
        )

        after = service.remove_participant("room_z", "scout")

        assert [p.id for p in after.participants] == []
        assert after.last_seen_seq.get("scout") == "4"

    def test_removing_the_default_responder_clears_the_policy(self, service: RoomService) -> None:
        """Otherwise the leave leaves a policy naming an agent that is gone.

        `DefaultResponderSelector` *raises* on a responder who is not an agent of the room,
        so an uncleared field converts a departure into a room where every later
        unaddressed message fails selection. The clearing is stated in the leave line,
        because a policy that changed itself silently is the kind of thing a reader later
        cannot account for.

        Killed by: src/uclone_x/room/service.py :: if state.policy.default_responder_id == participant_id:
        Becomes: if False:
        """
        service.create("Room", room_id="room_dr", policy=RoomPolicy(default_responder_id="scout"))
        service.add_participant("room_dr", "scout")
        service.add_participant("room_dr", "critic")

        after = service.remove_participant("room_dr", "scout")

        assert after.policy.default_responder_id == ""
        assert "responder" in after.transcript[-1].content
        # A different agent leaving must not touch the field.
        service.create("Other", room_id="room_dr2", policy=RoomPolicy(default_responder_id="scout"))
        service.add_participant("room_dr2", "scout")
        service.add_participant("room_dr2", "critic")
        kept = service.remove_participant("room_dr2", "critic")
        assert kept.policy.default_responder_id == "scout"


# --------------------------------------------------------------------------------------
# Membership is recorded, and is not speech
# --------------------------------------------------------------------------------------


class TestMembershipIsRecorded:
    def test_a_join_and_a_leave_are_written_to_the_transcript(self, service: RoomService) -> None:
        """The conversation alone must answer 'why did critic stop replying?'.

        Killed by: src/uclone_x/room/service.py :: RoomMessageKind.LEAVE, text
        Becomes: RoomMessageKind.JOIN, text
        """
        service.create("Room", room_id="room_m")
        service.add_participant("room_m", "critic", display_name="Critic")
        state = service.remove_participant("room_m", "critic")

        kinds = [m.kind for m in state.transcript]
        assert kinds == [RoomMessageKind.JOIN, RoomMessageKind.LEAVE]
        assert [m.seq for m in state.transcript] == [1, 2], "seq stays 1-based and gap-free"
        assert all(m.sender_id == "critic" for m in state.transcript)
        assert "critic" in state.transcript[1].content

    def test_a_membership_event_is_never_an_utterance(self, service: RoomService) -> None:
        """The one predicate every transcript consumer filters on.

        Killed by: src/uclone_x/room/models.py :: return self.kind is RoomMessageKind.UTTERANCE
        Becomes: return True
        """
        service.create("Room", room_id="room_u")
        state = service.add_participant("room_u", "scout")

        join = state.transcript[-1]
        assert join.is_utterance is False
        assert join.decision is None, "nobody decided that a join should happen"

    def test_membership_does_not_move_the_floor(self, service: RoomService) -> None:
        """A join is not a turn: it neither spends the agent budget nor takes the floor."""
        service.create("Room", room_id="room_f")
        service.add_participant("room_f", "alice", kind=ParticipantKind.HUMAN)
        state = service.add_participant("room_f", "scout")

        assert state.turn_state.agent_turns_since_human == 0
        assert state.turn_state.last_speaker_id is None

    @pytest.mark.usefixtures("builtin_personas_absent")
    def test_a_clone_made_after_the_registry_loaded_is_seated_by_its_handle(
        self, service: RoomService
    ) -> None:
        """A seat is named from the clone's own file, not the registry's older copy (#1815).

        The registry is a process-wide cache, so a clone created after it loaded is not in
        it, and a seat named from it showed the clone's `agt_…` id.

        Killed by: src/uclone_x/room/service.py :: fallback_name = handle_of(participant_id) or participant_id
        Becomes: fallback_name = __import__("uclone_x.agent.persona_registry", fromlist=["_"]).get_default_persona_registry().handle_for(participant_id)
        """
        from uclone_x.agent.persona_registry import get_default_persona_registry

        get_default_persona_registry()
        clone_id = make_clone("latecomer").path.name
        service.create("Room", room_id="room_late")

        state = service.add_participant("room_late", clone_id)

        assert clone_id.startswith("agt_")
        assert state.participants[-1].display_name == "latecomer"


# --------------------------------------------------------------------------------------
# The model itself
# --------------------------------------------------------------------------------------


class TestRoomStateTitle:
    def test_a_record_written_before_titles_still_loads(self, tmp_path: Path) -> None:
        """The default is empty for exactly one reason: forward-compatibility of old JSON.

        A stored room predating the field has no `title` key, and `extra="forbid"` plus
        `strict` would reject it outright if the field had no default.
        """
        store = RoomStore(tmp_path / "rooms")
        store.save(RoomState(room_id="room_old"))
        raw = store.room_path("room_old").read_text(encoding="utf-8")
        assert '"title"' in raw

        stripped = "\n".join(line for line in raw.splitlines() if '"title"' not in line)
        store.room_path("room_old").write_text(stripped, encoding="utf-8")

        loaded = store.load("room_old")
        assert loaded is not None
        assert loaded.title == ""


class TestOneHumanPerRoom:
    """The roster model itself refuses the shape the runtime cannot serve."""

    def test_two_humans_cannot_be_constructed(self) -> None:
        """Refused by `RoomState`, not only by the service that writes it.

        A model that represents something the runtime does not support is a trap for
        whoever reads it next: the roster looked like a shared-room feature and nothing in
        it said otherwise. Closing it at the service alone would leave the model still
        advertising it.

        Killed by: src/uclone_x/room/models.py :: if len(humans) > 1:
        Becomes: if False:
        """
        alice = Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice")
        bob = Participant(id="bob", kind=ParticipantKind.HUMAN, display_name="Bob")

        with pytest.raises(ValidationError, match="one human"):
            RoomState(room_id="room_mm", participants=(alice, bob))

    def test_one_human_and_many_agents_is_the_supported_shape(self) -> None:
        alice = Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice")
        scout = Participant(id="scout", kind=ParticipantKind.AGENT, display_name="Scout")
        critic = Participant(id="critic", kind=ParticipantKind.AGENT, display_name="Critic")

        state = RoomState(room_id="room_ok", participants=(alice, scout, critic))

        assert len(state.participants) == 3

    def test_a_stored_two_human_room_is_refused_on_load(self, tmp_path: Path) -> None:
        """A document hand-written or left by an older build does not load silently.

        Loading it would put the runtime back in the state this change removed, with the
        refusal bypassed by the one path that never went through the service.
        """
        store = RoomStore(tmp_path / "rooms")
        store.save(RoomState(room_id="room_raw"))
        raw = json.loads(store.room_path("room_raw").read_text(encoding="utf-8"))
        raw["participants"] = [
            {"id": "alice", "kind": "human", "display_name": "Alice"},
            {"id": "bob", "kind": "human", "display_name": "Bob"},
        ]
        store.room_path("room_raw").write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(UnreadableRoomRecordError) as excinfo:
            store.load("room_raw")

        # The refusal is the model's, chained for the log; the raised class is the store's
        # own, so no route above reads it as a malformed request (#1411).
        assert isinstance(excinfo.value.__cause__, ValidationError)
        assert "one human" in str(excinfo.value.__cause__)
        assert excinfo.value.room_id == "room_raw"

    def test_listing_survives_a_room_that_will_not_validate(self, tmp_path: Path) -> None:
        """One unreadable record must not take the listing down with it.

        The listing is the only way to find any of the *other* rooms, so a document that
        refuses to load has to cost its own row and nothing more.

        Killed by: src/uclone_x/room/service.py :: except UnreadableRoomRecordError:
        Becomes: except RoomNotFoundError:
        """
        store = RoomStore(tmp_path / "rooms")
        service = RoomService(store)
        service.create("Readable", room_id="room_good")
        store.save(RoomState(room_id="room_bad"))
        raw = json.loads(store.room_path("room_bad").read_text(encoding="utf-8"))
        raw["participants"] = [
            {"id": "alice", "kind": "human", "display_name": "Alice"},
            {"id": "bob", "kind": "human", "display_name": "Bob"},
        ]
        store.room_path("room_bad").write_text(json.dumps(raw), encoding="utf-8")

        summaries = service.list_rooms()

        assert [s.room_id for s in summaries] == ["room_good"]
        # Left out of the rooms, and still reported by id (#1440).
        listing = service.survey_rooms()
        assert [s.room_id for s in listing.rooms] == ["room_good"]
        assert listing.unreadable == ("room_bad",)

    def test_listing_survives_a_file_whose_stem_is_not_an_addressable_id(
        self, tmp_path: Path
    ) -> None:
        """A stray file in the directory must not take the whole listing down (#1258).

        `list_room_ids` reports every `*.json` stem it finds, and `...json` yields the stem
        `..` (on 3.12; on a pathlib that stems it `...json`, that too carries the forbidden
        `..`), which the store's path guard refuses with `RoomIdError`. Unskipped, that
        propagates through a route with no `except` and `GET /api/rooms` answers a bare
        500 — the listing being the only way to find any of the *readable* rooms.

        Killed by: src/uclone_x/room/service.py :: except RoomIdError:
        Becomes: except RoomNotFoundError:
        """
        store = RoomStore(tmp_path / "rooms")
        service = RoomService(store)
        service.create("Readable", room_id="room_good")
        planted = store.storage_dir / "...json"
        planted.write_text("{}", encoding="utf-8")
        # The premise, asserted rather than assumed: the stray file is one the listing
        # looks at, and the stem it yields is one the store refuses to address. The stem
        # is read back rather than written down, so a pathlib that splits the name
        # differently fails on the refusal below and not on a stale literal.
        assert planted.stem in store.list_room_ids()
        with pytest.raises(RoomIdError):
            store.load(planted.stem)

        summaries = service.list_rooms()

        assert [s.room_id for s in summaries] == ["room_good"]
        # Not reported as unreadable: no route can address it by that stem.
        assert service.survey_rooms().unreadable == ()


class TestSetDefaultResponder:
    """The Core half, pinned separately from the CLI's.

    The command refuses an unseated responder before the service is reached, so a mutation
    that removes the service's own check killed nothing — the guard existed and nothing
    held it. `RoomService` is a public seam that the CLI is only one caller of.
    """

    def test_it_names_a_seated_agent(self, service: RoomService) -> None:
        state = service.create("R", room_id="r1")
        service.add_participant("r1", "scout", kind=ParticipantKind.AGENT)
        service.add_participant("r1", "critic", kind=ParticipantKind.AGENT)

        state = service.set_default_responder("r1", "critic")

        assert state.policy.default_responder_id == "critic"

    def test_an_empty_id_clears_it(self, service: RoomService) -> None:
        service.create("R", room_id="r1")
        service.add_participant("r1", "scout", kind=ParticipantKind.AGENT)
        service.set_default_responder("r1", "scout")

        state = service.set_default_responder("r1", "")

        assert state.policy.default_responder_id == ""

    def test_it_refuses_someone_who_is_not_a_seated_agent(self, service: RoomService) -> None:
        """A responder who is not in the room fails the room's next unaddressed message.

        Killed by: src/uclone_x/room/service.py :: if agent_id not in seated:
        Becomes: if False:
        """
        service.create("R", room_id="r1")
        service.add_participant("r1", "scout", kind=ParticipantKind.AGENT)

        with pytest.raises(RoomError, match="ghost"):
            service.set_default_responder("r1", "ghost")

    def test_it_refuses_a_human(self, service: RoomService) -> None:
        service.create("R", room_id="r1")
        service.add_participant("r1", "kenny", kind=ParticipantKind.HUMAN)

        with pytest.raises(RoomError, match="kenny"):
            service.set_default_responder("r1", "kenny")

    def test_an_unknown_room_is_refused(self, service: RoomService) -> None:
        with pytest.raises(RoomNotFoundError):
            service.set_default_responder("nope", "scout")


# --------------------------------------------------------------------------------------
# Ids the derivation cannot survive
# --------------------------------------------------------------------------------------


class TestParticipantIdsTheRosterMustRefuse:
    """`add_participant` is where G3 is *established*, so it owns the ids it can establish it for.

    Every id seated here becomes a session id and a namespace IRI by string concatenation.
    An id the concatenation cannot survive is not a room problem discovered in the room: it
    surfaces inside the session store, or in a knowledge graph, several steps from the
    roster edit that caused it — which is the distance this service exists to close.
    """

    def test_an_id_that_escapes_the_stores_path_guard_is_refused(
        self, service: RoomService
    ) -> None:
        """`../../evil` was seated, and its session id carried the traversal.

        `validate_session_id` forbids a path separator and a `..` segment, so the derived
        `sess_room__room_esc__../../evil` is refused by the session store — but only at the
        first write, by which point the participant is on the roster, the join is in the
        transcript, and the error a person sees names a session id nobody typed. The guard
        belongs at the seat.

        Killed by: src/uclone_x/room/service.py :: validate_session_id(participant_session_id(room_id, participant_id))
        Becomes: pass
        """
        service.create("Room", room_id="room_esc")

        with pytest.raises(RoomError, match="session"):
            service.add_participant("room_esc", "../../evil")

        assert service.get("room_esc").participants == ()
        assert service.get("room_esc").transcript == (), "a refused seat writes no join row"

    def test_a_blank_id_is_refused(self, service: RoomService) -> None:
        """An empty or whitespace id derives a session the store will not name.

        `validate_session_id` refuses an empty id because it resolves to the storage
        directory itself; a whitespace one is worse, because it *is* nameable and renders
        as nothing — a participant the roster shows as a blank row and that no mention can
        ever address.

        Killed by: src/uclone_x/room/service.py :: if not value or value != value.strip():
        Becomes: if False:
        """
        service.create("Room", room_id="room_blank")

        for blank in ("", "   ", "\t"):
            with pytest.raises(RoomError, match="id"):
                service.add_participant("room_blank", blank)
        # A padded id renders as the unpadded one and addresses differently. Refused for
        # the same reason two ids that print the same are refused below.
        with pytest.raises(RoomError, match="id"):
            service.add_participant("room_blank", " scout ")

        assert service.get("room_blank").participants == ()

    def test_two_ids_differing_only_by_case_are_refused(self, service: RoomService) -> None:
        """`critic` and `Critic` derive two session ids that are one file on macOS.

        The exact-match duplicate check let both sit on the roster with distinct-looking
        sessions, `sess_room__r__critic` and `sess_room__r__Critic`. On a case-insensitive filesystem
        — the default on macOS and on Windows — those are one record, and `BaseAgent.
        hydrate_session` refuses a record identifying a different session (#256), so the
        second agent cannot hydrate at all. The roster is where the two ids are still
        visible side by side, so the roster is where they are refused.

        Killed by: src/uclone_x/room/service.py :: (p for p in state.participants if p.id.casefold() == participant_id.casefold()),
        Becomes: (p for p in state.participants if p.id == participant_id),
        """
        service.create("Room", room_id="room_case")
        service.add_participant("room_case", "critic")

        with pytest.raises(RoomError) as caught:
            service.add_participant("room_case", "Critic")

        assert "critic" in str(caught.value)
        assert [p.id for p in service.get("room_case").participants] == ["critic"]

    def test_the_derived_session_id_is_unambiguous_across_rooms(self, service: RoomService) -> None:
        """`sess_room__{room}__{participant}` is only injective if neither half holds `__`.

        Room `a` seating `b__c`, and room `a__b` seating `c`, both derive
        `sess_room__a__b__c` — two participants of two *different* rooms on one `SessionState`,
        which is G3 broken by string concatenation rather than by any roster edit that
        looks wrong. The resolver cannot catch it: it holds one room's claims, and each
        room's claim is internally consistent.

        Killed by: src/uclone_x/room/service.py :: if SESSION_ID_SEPARATOR in value:
        Becomes: if False:
        """
        service.create("Room", room_id="room_sep")
        with pytest.raises(RoomError, match="__"):
            service.add_participant("room_sep", "b__c")

        # The room half of the same concatenation, refused where the room is named.
        with pytest.raises(RoomError, match="__"):
            service.create("Ambiguous", room_id="a__b")

    def test_a_unicode_id_is_seated_as_a_human_and_refused_as_an_agent(
        self, service: RoomService
    ) -> None:
        """The two kinds do not carry the same obligation, so they do not get one rule.

        A human is a name on a roster: the filesystem can hold `評論家` and a mention can
        address it, so refusing it for its script would be a rule nobody asked for. An
        agent's id is additionally the name of the directory holding its id and its
        memory, and that directory name is refused rather than repaired -- so seating
        `評論家` as an agent would put a participant on the roster whose first turn cannot
        load its own memory, with the refusal arriving from the memory factory rather than
        from the seat that caused it.

        Killed by: src/uclone_x/room/service.py :: refuse_an_unusable_username(participant_id)
        Becomes: pass
        """
        service.create("Room", room_id="room_uni")
        state = service.add_participant(
            "room_uni", "評論家", display_name="評論家", kind=ParticipantKind.HUMAN
        )

        seated = state.participants[0]
        assert seated.id == "評論家"

        with pytest.raises(RoomError) as caught:
            service.add_participant("room_uni", "評論家2")

        assert "as an agent" in str(caught.value)
        assert [p.id for p in service.get("room_uni").participants] == ["評論家"], (
            "a refused seat leaves the roster as it was"
        )

    def test_a_blank_display_name_falls_back_to_the_id(self, service: RoomService) -> None:
        """A whitespace display name rendered as a leading gap in the join row.

        `display_name or participant_id` treats `"   "` as a name, so the transcript read
        `    (scout) joined the room as an agent` and every later render of that
        participant showed nothing at all where a name goes.

        Killed by: src/uclone_x/room/service.py :: display_name=display_name.strip() or fallback_name,
        Becomes: display_name=display_name or fallback_name,
        """
        service.create("Room", room_id="room_dn")
        state = service.add_participant("room_dn", "scout", display_name="   ")

        assert state.participants[0].display_name == "scout"
        assert state.transcript[-1].content.startswith("scout (scout) joined")


# --------------------------------------------------------------------------------------
# Lifecycle edges, and the record they leave
# --------------------------------------------------------------------------------------


class TestRosterLifecycleEdges:
    def test_removing_the_last_agent_leaves_a_readable_empty_room(
        self, service: RoomService
    ) -> None:
        """An empty roster is a state, not a failure: the room and its record survive it."""
        service.create("Room", room_id="room_last")
        service.add_participant("room_last", "scout")
        after = service.remove_participant("room_last", "scout")

        assert after.participants == ()
        assert [m.kind for m in after.transcript] == [RoomMessageKind.JOIN, RoomMessageKind.LEAVE]
        assert service.list_rooms()[0].agent_ids == ()

    def test_a_re_seated_participant_keeps_the_mark_it_left_with(
        self, service: RoomService
    ) -> None:
        """Documented, and worth pinning because it is the surprising half of the contract.

        `remove_participant` keeps `last_seen_seq` so that a rejoin does not replay the
        whole room into an agent's own session. The cost is that whatever was said while it
        was away is now behind its mark and will never be shown to it — the agent rejoins
        with a gap it cannot see. That is a design question (§3.2.2 records the keep, not
        the gap), not something this service can decide alone, so the behaviour is pinned
        here rather than quietly changed.
        """
        service.create("Room", room_id="room_re")
        service.add_participant("room_re", "scout")
        state = service.get("room_re")
        service._store.save(  # pyright: ignore[reportPrivateUsage]
            state.model_copy(update={"last_seen_seq": {"scout": "9"}})
        )
        service.remove_participant("room_re", "scout")

        after = service.add_participant("room_re", "scout")

        assert after.last_seen_seq.get("scout") == "9"

    def test_membership_rows_keep_the_seq_the_transcript_expects(
        self, service: RoomService
    ) -> None:
        """Gap-free and 1-based across every roster change, not only the first.

        Killed by: src/uclone_x/room/service.py :: seq=len(state.transcript) + 1,
        Becomes: seq=len(state.transcript),
        """
        service.create("Room", room_id="room_seq")
        service.add_participant("room_seq", "alice", kind=ParticipantKind.HUMAN)
        service.add_participant("room_seq", "scout")
        service.add_participant("room_seq", "critic")
        service.remove_participant("room_seq", "scout")
        state = service.remove_participant("room_seq", "critic")

        assert [m.seq for m in state.transcript] == [1, 2, 3, 4, 5]
        assert [m.sender_id for m in state.transcript] == [
            "alice",
            "scout",
            "critic",
            "scout",
            "critic",
        ]

    def test_two_services_over_one_store_do_not_lose_a_roster_edit(self, tmp_path: Path) -> None:
        """CAS is the guard on the single-writer claim, and a roster edit is a writer.

        Both services hold the room at the same revision; the second save would otherwise
        drop the first's join row and its participant silently. The refusal is the design's
        stated outcome (§3.9) — a stale room write means two writers are driving one room —
        so what this pins is that a roster edit goes through that guard rather than around
        it.
        """
        store = RoomStore(tmp_path / "rooms")
        first = RoomService(store)
        second = RoomService(store)
        first.create("Contended", room_id="room_cas")

        # Both read the same revision before either writes.
        held_by_second = second.get("room_cas")
        first.add_participant("room_cas", "scout")

        with pytest.raises(StaleRoomWriteError, match="room_cas"):
            store.save(held_by_second.model_copy(update={"title": "Renamed"}))

        # The first edit survived, whole.
        assert [p.id for p in first.get("room_cas").participants] == ["scout"]


# --------------------------------------------------------------------------------------
# History: rewinding and clearing the record (#1208)
# --------------------------------------------------------------------------------------


def _with_transcript(
    service: RoomService, room_id: str, rows: list[tuple[str, str]], *, spent: int
) -> RoomState:
    """Give a room a transcript directly, and mark every agent as having read all of it.

    `spent` is `agent_turns_since_human` as the orchestrator would have left it, and every
    caller states it by hand rather than deriving it. Deriving it would compute the fixture
    with the arithmetic under test, and a bug shared by both would pass every assertion.

    Written through the store rather than through a turn loop: what these tests pin is the
    service's arithmetic over a record, and standing up an orchestrator and four fake models
    to produce that record would make the test's own subject the thing most likely to break
    it. `seq` is assigned exactly as every writer in the tree assigns it — `len(transcript) +
    1` — because the reuse that follows from it is half of what is under test.
    """
    state = service.get(room_id)
    agents = [p.id for p in state.participants if p.kind is ParticipantKind.AGENT]
    transcript = tuple(
        RoomMessage(
            seq=index + 1,
            sender_id=sender,
            content=content,
            turn_id=f"t{index + 1}" if sender in agents else None,
        )
        for index, (sender, content) in enumerate(rows)
    )
    return service._store.save(  # pyright: ignore[reportPrivateUsage]
        state.model_copy(
            update={
                "transcript": transcript,
                "last_seen_seq": {agent: str(len(transcript)) for agent in agents},
                "turn_state": state.turn_state.model_copy(
                    update={"agent_turns_since_human": spent}
                ),
            }
        )
    )


@pytest.fixture
def seated(service: RoomService) -> RoomService:
    """A room of one human and two agents, with six utterances in it."""
    service.create("Rewind me", room_id="room_hist")
    service.add_participant("room_hist", "alice", kind=ParticipantKind.HUMAN)
    service.add_participant("room_hist", "scout")
    service.add_participant("room_hist", "critic")
    _with_transcript(
        service,
        "room_hist",
        [
            ("alice", "one"),
            ("scout", "two"),
            ("critic", "three"),
            ("alice", "four"),
            ("scout", "five"),
            ("critic", "six"),
        ],
        spent=2,
    )
    return service


class TestTruncateTranscript:
    def test_keeps_the_named_message_and_drops_what_follows(self, seated: RoomService) -> None:
        state = seated.truncate_transcript("room_hist", 3)

        assert [m.content for m in state.transcript] == ["one", "two", "three"]

    def test_clamps_every_read_mark_to_the_new_tail(self, seated: RoomService) -> None:
        """The whole reason a rewind is not just a slice.

        `seq` is `len(transcript) + 1` at every writer, so cutting six rows to three frees
        seqs 4-6 for reuse. `RoomOrchestrator._unseen_span` hands a speaker only
        `m.seq > last_seen`, so a mark left at 6 would hide the three *replacement* turns
        from that agent entirely — three messages every other participant can see, one
        participant cannot, and nothing anywhere saying so. The room would look like an
        agent that had stopped listening.

        Killed by: src/uclone_x/room/service.py :: pid: str(min(int(mark), seq)) for pid, mark in state.last_seen_seq.items()
        Becomes: pid: mark for pid, mark in state.last_seen_seq.items()
        """
        state = seated.truncate_transcript("room_hist", 3)

        assert dict(state.last_seen_seq) == {"scout": "3", "critic": "3"}

    def test_leaves_a_mark_that_is_already_behind_the_cut_alone(self, service: RoomService) -> None:
        """Clamping is a ceiling, not an assignment.

        An agent that had not caught up before the rewind has an *older* mark, and raising it
        to the cut would silently mark conversation as read that it was never handed.

        Killed by: src/uclone_x/room/service.py :: pid: str(min(int(mark), seq)) for pid, mark in state.last_seen_seq.items()
        Becomes: pid: str(seq) for pid, mark in state.last_seen_seq.items()
        """
        service.create("Behind", room_id="room_behind")
        service.add_participant("room_behind", "alice", kind=ParticipantKind.HUMAN)
        service.add_participant("room_behind", "scout")
        state = service.get("room_behind")
        service._store.save(  # pyright: ignore[reportPrivateUsage]
            state.model_copy(
                update={
                    "transcript": tuple(
                        RoomMessage(seq=n, sender_id="alice", content=str(n)) for n in range(1, 7)
                    ),
                    "last_seen_seq": {"scout": "2"},
                }
            )
        )

        rewound = service.truncate_transcript("room_behind", 4)

        assert dict(rewound.last_seen_seq) == {"scout": "2"}

    def test_recounts_the_cascade_ceiling_from_what_survives(self, seated: RoomService) -> None:
        """`agent_turns_since_human` bounds agent-to-agent traffic, so it cannot be inherited.

        The room's six rows end with two agent turns after alice's fourth message. Rewinding
        to seq 4 — alice's own message — leaves zero agent turns since a human, and a count
        carried over from the dropped turns would strand the rewound room at its ceiling with
        nothing in the record to explain the refusal.

        The count would otherwise stay at whatever the pre-rewind state held, which is why
        the fixture below stamps a non-zero one.

        Killed by: src/uclone_x/room/service.py :: "turn_state": state.turn_state.model_copy(update=floor),
        Becomes: "turn_state": state.turn_state,
        """
        pre = seated.get("room_hist")
        seated._store.save(  # pyright: ignore[reportPrivateUsage]
            pre.model_copy(
                update={
                    "turn_state": pre.turn_state.model_copy(
                        update={"agent_turns_since_human": 2, "last_speaker_id": "critic"}
                    )
                }
            )
        )

        state = seated.truncate_transcript("room_hist", 4)

        assert state.turn_state.agent_turns_since_human == 0
        assert state.turn_state.last_speaker_id == "alice"

    def test_counts_the_agent_turns_that_do_survive(self, seated: RoomService) -> None:
        state = seated.truncate_transcript("room_hist", 6)

        assert state.turn_state.agent_turns_since_human == 2
        assert state.turn_state.last_speaker_id == "critic"

    def test_a_retried_turn_is_not_charged_a_second_time_by_the_recount(
        self, service: RoomService
    ) -> None:
        """The record is not sufficient to recompute the ceiling, so the recount only lowers it.

        `RoomOrchestrator.retry` refunds `agent_turns_since_human` and **keeps** the failed
        row ("The failed row is kept", `orchestrator.py`). The refund is therefore written
        nowhere in the transcript, and a recount bills the failure and the retry both. Here
        the live counter is 1 — one charge, one refund, one charge — while the rows alone say
        2. Uncapped, rewinding to the message already at the tail *raises* the count, and at
        the default ceiling of 3 two retried failures plus a rewind is enough to refuse the
        next turn with nothing in the record accounting for it.

        Killed by: src/uclone_x/room/service.py :: "agent_turns_since_human": min(since_human, spent),
        Becomes: "agent_turns_since_human": since_human,
        """
        service.create("Retried", room_id="room_retry")
        service.add_participant("room_retry", "alice", kind=ParticipantKind.HUMAN)
        service.add_participant("room_retry", "scout")
        pre = service.get("room_retry")
        service._store.save(  # pyright: ignore[reportPrivateUsage]
            pre.model_copy(
                update={
                    "transcript": (
                        RoomMessage(seq=1, sender_id="alice", content="hi"),
                        RoomMessage(
                            seq=2,
                            sender_id="scout",
                            turn_id="t2",
                            content="",
                            error="provider down",
                        ),
                        RoomMessage(seq=3, sender_id="scout", turn_id="t3", content="here you go"),
                    ),
                    "turn_state": pre.turn_state.model_copy(
                        update={"agent_turns_since_human": 1, "last_speaker_id": "scout"}
                    ),
                }
            )
        )

        state = service.truncate_transcript("room_retry", 3)

        assert state.turn_state.agent_turns_since_human == 1

    def test_a_departed_humans_messages_do_not_become_agent_turns(
        self, service: RoomService
    ) -> None:
        """`humans` is the *current* roster, and `remove_participant` has no kind guard.

        Once alice leaves, nothing in the loop stops at her rows and every one of them counts
        as an agent turn. On a long conversation that recount runs to the length of the
        transcript, which would strand the room above its ceiling permanently — the cap is
        what makes a roster change unable to invent turns nobody took.

        Killed by: src/uclone_x/room/service.py :: "agent_turns_since_human": min(since_human, spent),
        Becomes: "agent_turns_since_human": since_human,
        """
        service.create("Departed", room_id="room_gone")
        service.add_participant("room_gone", "alice", kind=ParticipantKind.HUMAN)
        service.add_participant("room_gone", "scout")
        _with_transcript(
            service,
            "room_gone",
            [("alice", "one"), ("scout", "two"), ("alice", "three"), ("scout", "four")],
            spent=1,
        )
        service.remove_participant("room_gone", "alice")

        state = service.truncate_transcript("room_gone", 4)

        assert state.turn_state.agent_turns_since_human == 1

    def test_a_join_between_turns_is_not_counted_as_one(self, service: RoomService) -> None:
        """A membership row is not a turn, and the orchestrator does not count it as one.

        The room's own prose about who arrived would otherwise be billed to the cascade
        ceiling, and a room could be pushed to its limit by nothing but participants joining.

        Killed by: src/uclone_x/room/service.py :: if not message.is_utterance:
        Becomes: if False:
        """
        service.create("Joined", room_id="room_join")
        service.add_participant("room_join", "alice", kind=ParticipantKind.HUMAN)
        service.add_participant("room_join", "critic")
        pre = service.get("room_join")
        service._store.save(  # pyright: ignore[reportPrivateUsage]
            pre.model_copy(
                update={
                    "transcript": (
                        RoomMessage(seq=1, sender_id="alice", content="one"),
                        RoomMessage(
                            seq=2,
                            sender_id="critic",
                            turn_id="t-critic",
                            content="critic joined the room",
                            kind=RoomMessageKind.JOIN,
                        ),
                    ),
                    "turn_state": pre.turn_state.model_copy(
                        update={"agent_turns_since_human": 1, "last_speaker_id": "alice"}
                    ),
                }
            )
        )

        state = service.truncate_transcript("room_join", 2)

        assert state.turn_state.agent_turns_since_human == 0
        assert state.turn_state.last_speaker_id == "alice"

    def test_refuses_a_seq_the_transcript_does_not_hold(self, seated: RoomService) -> None:
        """Not clamped in either direction, and the refusal says what is there.

        Clamping down destroys more than was asked for; clamping up answers success over a
        no-op. Both are silent, and a rewind is the one control whose mistakes cannot be
        undone.
        """
        with pytest.raises(RoomError, match=r"no message at seq 9.*present: 1-6"):
            seated.truncate_transcript("room_hist", 9)

    def test_refuses_zero_rather_than_emptying_the_room(self, seated: RoomService) -> None:
        """`truncate_transcript(0)` is not a way to clear: `clear_transcript` is.

        Without the guard a mistyped seq empties the conversation and reports success.

        Killed by: src/uclone_x/room/service.py :: if not any(m.seq == seq for m in state.transcript):
        Becomes: if False:
        """
        with pytest.raises(RoomError, match="no message at seq 0"):
            seated.truncate_transcript("room_hist", 0)

        assert len(seated.get("room_hist").transcript) == 6

    def test_says_the_room_is_empty_rather_than_naming_a_range(self, service: RoomService) -> None:
        """P6: an empty room has no range to print, and `"present: "` alone states nothing."""
        service.create("Silent", room_id="room_silent")

        with pytest.raises(RoomError, match="present: none; the room is empty"):
            service.truncate_transcript("room_silent", 1)

    def test_refuses_a_room_that_does_not_exist(self, service: RoomService) -> None:
        with pytest.raises(RoomNotFoundError):
            service.truncate_transcript("room_missing", 1)


class TestClearTranscript:
    def test_empties_the_record_but_keeps_the_conversation(self, seated: RoomService) -> None:
        """The distinction from `DELETE /api/rooms`: the conversation is still there."""
        state = seated.clear_transcript("room_hist")

        assert state.transcript == ()
        assert state.room_id == "room_hist"
        assert state.title == "Rewind me"
        assert [p.id for p in state.participants] == ["alice", "scout", "critic"]

    def test_drops_every_read_mark(self, seated: RoomService) -> None:
        """Unlike `remove_participant`, which keeps them deliberately.

        A mark surviving a clear indexes a transcript that no longer exists, and since `seq`
        restarts at 1 every replacement message falls below it — the cleared room's agents
        would be handed nothing at all.

        Both marks would otherwise stay at `"6"`, and the room's next six turns would be
        invisible to the agents holding them.

        Killed by: src/uclone_x/room/service.py :: "last_seen_seq": {},
        Becomes: "last_seen_seq": state.last_seen_seq,
        """
        assert dict(seated.clear_transcript("room_hist").last_seen_seq) == {}

    def test_resets_the_cascade_ceiling(self, seated: RoomService) -> None:
        pre = seated.get("room_hist")
        seated._store.save(  # pyright: ignore[reportPrivateUsage]
            pre.model_copy(
                update={
                    "turn_state": pre.turn_state.model_copy(
                        update={"agent_turns_since_human": 3, "last_speaker_id": "critic"}
                    )
                }
            )
        )

        state = seated.clear_transcript("room_hist")

        assert state.turn_state.agent_turns_since_human == 0
        assert state.turn_state.last_speaker_id is None

    def test_refuses_a_room_that_does_not_exist(self, service: RoomService) -> None:
        with pytest.raises(RoomNotFoundError):
            service.clear_transcript("room_missing")


def _with_tool_record(service: RoomService, room_id: str) -> None:
    """Give rows 2 and 5 of `seated` a turn id, a tool call each, and a written file each."""
    from uclone_x.room.models import RoomToolUse, RoomWrittenFile

    state = service.get(room_id)
    turns = {2: "turn_two", 5: "turn_five"}
    transcript = tuple(
        m.model_copy(update={"turn_id": turns[m.seq], "tools_recorded": True})
        if m.seq in turns
        else m
        for m in state.transcript
    )
    service._store.save(  # pyright: ignore[reportPrivateUsage]
        state.model_copy(
            update={
                "transcript": transcript,
                "tool_uses": tuple(
                    RoomToolUse(
                        turn_id=turn_id,
                        participant_id="scout",
                        tool_name="file_write",
                        status="success",
                        written_path=f"{turn_id}.md",
                    )
                    for turn_id in turns.values()
                ),
                "written_files": tuple(
                    RoomWrittenFile(
                        path=f"{turn_id}.md",
                        participant_id="scout",
                        tool_name="file_write",
                        turn_id=turn_id,
                    )
                    for turn_id in turns.values()
                ),
            }
        )
    )


class TestAHistoryChangeAndTheToolRecord:
    """A rewind takes back what was said and the calls behind it, not the files (#1353/#1354)."""

    def test_a_rewind_drops_the_tool_calls_of_the_turns_it_removes(
        self, seated: RoomService
    ) -> None:
        """Left behind, a removed turn's calls would be shown under a seat that never made them.

        Killed by: src/uclone_x/room/service.py :: "tool_uses": tuple(u for u in state.tool_uses if u.turn_id in kept_turns),
        Becomes: "tool_uses": state.tool_uses,
        """
        _with_tool_record(seated, "room_hist")

        state = seated.truncate_transcript("room_hist", 3)

        assert [u.turn_id for u in state.tool_uses] == ["turn_two"]
        assert [f.path for f in state.written_files] == ["turn_two.md", "turn_five.md"]

    def test_a_clear_drops_every_tool_call_and_keeps_the_files(self, seated: RoomService) -> None:
        """The files are still on disk and are still this conversation's output.

        Killed by: src/uclone_x/room/service.py :: "tool_uses": (),
        Becomes: "tool_uses": state.tool_uses,
        """
        _with_tool_record(seated, "room_hist")

        state = seated.clear_transcript("room_hist")

        assert state.tool_uses == ()
        assert [f.path for f in state.written_files] == ["turn_two.md", "turn_five.md"]


class TestTheFileRecordOnlyGrows:
    """What a clear or a rewind removes is counted, and nothing resets the count (#1366)."""

    def test_a_new_room_is_the_only_one_born_complete(self, service: RoomService) -> None:
        """Killed by: src/uclone_x/room/service.py :: "file_record": RoomFileRecord(kept_since_creation=True),
        Becomes: "file_record": RoomFileRecord(),
        """
        from uclone_x.room.models import RoomState

        assert service.create("Fresh").file_record.kept_since_creation is True
        # A record stored before the field existed loads as incomplete, never as fresh.
        assert RoomState.model_validate({"title": "old"}).file_record.kept_since_creation is False

    def test_a_clear_and_a_rewind_are_counted_and_neither_resets_the_other(
        self, seated: RoomService
    ) -> None:
        """Killed by: src/uclone_x/room/service.py :: if len(kept) < len(state.transcript):
        Becomes: if False:
        """
        state = seated.truncate_transcript("room_hist", 3)
        assert (state.file_record.rewinds, state.file_record.clears) == (1, 0)
        # Rewinding to the last row removes nothing, so it hides nothing.
        last = state.transcript[-1].seq
        assert seated.truncate_transcript("room_hist", last).file_record.rewinds == 1

        cleared = seated.clear_transcript("room_hist")
        assert (cleared.file_record.rewinds, cleared.file_record.clears) == (1, 1)
        # An empty room cleared again loses nothing.
        assert seated.clear_transcript("room_hist").file_record.clears == 1


_FEATURE_FIELDS = ("tool_uses", "written_files", "file_record", "story_id", "head", "workspace")


def test_a_default_room_is_saved_without_the_feature_fields(tmp_path: Path) -> None:
    """A room using none of the later features is written in the shape an older build reads.

    `RoomState` forbids unknown fields, so a key an older build does not know makes it
    refuse the whole room. Each feature field is therefore left out while it holds its
    default (#1885), and a room that does set one keeps it -- the round trip loses nothing.

    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda head: head is None,
    Becomes: exclude_if=lambda head: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda story: story is None,
    Becomes: exclude_if=lambda story: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda record: record == RoomFileRecord(),
    Becomes: exclude_if=lambda record: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda uses: not uses,
    Becomes: exclude_if=lambda uses: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda written: not written,
    Becomes: exclude_if=lambda written: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda head: head is None,
    Becomes: exclude_if=lambda head: True,
    """
    store = RoomStore(tmp_path / "rooms")
    saved = store.save(RoomState(room_id="room_plain"))
    store.save(RoomState(room_id="room_of_a_head", head="run", story_id="story_1"))

    plain = json.loads((tmp_path / "rooms" / "room_plain.json").read_text())
    marked = json.loads((tmp_path / "rooms" / "room_of_a_head.json").read_text())

    assert [key for key in _FEATURE_FIELDS if key in plain] == []
    assert (marked["head"], marked["story_id"]) == ("run", "story_1")
    assert store.load("room_plain") == saved
    reloaded = store.load("room_of_a_head")
    assert reloaded is not None
    assert reloaded.head == "run"


#: The keys each nested record had at #1366 (bfc93861), the oldest build that reads a room
#: the service creates: every such room carries `file_record` from its first save.
_KEYS_AT_1366 = {
    "RoomMessage": {
        "seq",
        "sender_id",
        "content",
        "kind",
        "created_at",
        "decision",
        "provenance",
        "usage",
        "error",
        "refusal",
        "completed",
        "persist_error",
        "rendered_through",
        "turn_id",
        "tools_recorded",
    },
    "RoomPolicy": {
        "max_agent_turns_per_human_message",
        "max_span_messages",
        "transcript_window",
        "hesitation_seconds",
        "default_responder_id",
        "auto_routing",
        "selector_llm",
    },
    "RoomToolUse": {
        "turn_id",
        "participant_id",
        "tool_name",
        "tool_call_id",
        "status",
        "error",
        "duration_ms",
        "arguments_preview",
        "output_preview",
        "truncated",
        "written_path",
        "wrote_unnamed",
        "subagent_id",
        "recorded_at",
    },
}


def test_nested_records_are_saved_in_the_older_shape(service: RoomService) -> None:
    """A room's rows, policy and tool calls are written in the keys #1366 knew (#1885).

    Each record forbids unknown keys, so one newer key anywhere in a room makes an older
    build refuse the whole room. A room using none of the newer features therefore holds
    none of their keys, and one using all of them keeps every value through a round trip.

    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda code: code is None,
    Becomes: exclude_if=lambda code: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda params: params is None,
    Becomes: exclude_if=lambda params: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda failure: failure is None,
    Becomes: exclude_if=lambda failure: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda tried: tried == 0,
    Becomes: exclude_if=lambda tried: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda unsaved: unsaved == 0,
    Becomes: exclude_if=lambda unsaved: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda tokens: tokens == DEFAULT_MAX_SPAN_TOKENS,
    Becomes: exclude_if=lambda tokens: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda autonomous: not autonomous,
    Becomes: exclude_if=lambda autonomous: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda paths: not paths,
    Becomes: exclude_if=lambda paths: False,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda failure: failure is None,
    Becomes: exclude_if=lambda failure: True,
    Killed by: src/uclone_x/room/models.py :: exclude_if=lambda paths: not paths,
    Becomes: exclude_if=lambda paths: True,
    """
    from uclone_x.agent.models import ProviderFailure
    from uclone_x.errors import ProviderFailureKind
    from uclone_x.room.models import RoomToolUse

    store = service._store  # pyright: ignore[reportPrivateUsage]
    assert isinstance(store, RoomStore)
    created = service.create(
        "Index tuning",
        seats=[("user", ParticipantKind.HUMAN), ("scout", ParticipantKind.AGENT)],
    )
    joins = created.transcript
    asked = RoomMessage(seq=len(joins) + 1, sender_id="user", content="what is slow?")
    answered = RoomMessage(
        seq=len(joins) + 2, sender_id="scout", content="the join", turn_id="t-join"
    )
    call = RoomToolUse(turn_id="turn_1", participant_id="scout", tool_name="grep", status="ok")
    plain = store.save(
        created.model_copy(update={"transcript": (*joins, asked, answered), "tool_uses": (call,)})
    )

    raw = json.loads(store.room_path(plain.room_id).read_text(encoding="utf-8"))

    assert raw["transcript"] and raw["tool_uses"]
    for row in raw["transcript"]:
        assert set(row) - _KEYS_AT_1366["RoomMessage"] == set(), row
    assert set(raw["policy"]) - _KEYS_AT_1366["RoomPolicy"] == set()
    assert set(raw["tool_uses"][0]) - _KEYS_AT_1366["RoomToolUse"] == set()
    assert "turn_id" not in raw["transcript"][-2]  # the human's row carries none
    assert "tools_recorded" not in raw["transcript"][-1]

    note = RoomMessage(
        seq=len(joins) + 3,
        sender_id="scout",
        content="Stopped.",
        kind=RoomMessageKind.NOTE,
        code="loop.stopped",
        params={"interval_seconds": 60},
    )
    failed = answered.model_copy(
        update={
            "turn_id": "turn_1",
            "tools_recorded": True,
            "error": "quota",
            "provider_failure": ProviderFailure(
                kind=ProviderFailureKind.PROVIDER_QUOTA, message="Out of quota.", retryable=False
            ),
            "memory_facts_tried": 2,
            "memory_facts_unsaved": 1,
        }
    )
    featured = store.save(
        plain.model_copy(
            update={
                "transcript": (*joins, asked, failed, note),
                "tool_uses": (call.model_copy(update={"written_paths": ("a.md", "b.md")}),),
                "policy": plain.policy.model_copy(
                    update={"autonomous": True, "max_span_tokens": 4_000}
                ),
            }
        )
    )

    assert store.load(featured.room_id) == featured


def test_a_room_is_created_with_its_seats_in_one_save(service: RoomService) -> None:
    """Three saves left a seatless room behind whenever the second one failed (#1885).

    Killed by: src/uclone_x/room/service.py :: state = self._seated(state, participant_id, kind=kind)
    Becomes: state = self._store.save(self._seated(state, participant_id, kind=kind))
    """
    state = service.create(
        "Index tuning",
        seats=[("user", ParticipantKind.HUMAN), ("scout", ParticipantKind.AGENT)],
    )

    assert state.revision == 1
    assert [p.id for p in service.get(state.room_id).participants] == ["user", "scout"]


_SERVER_HEAD_ID = {"acp": "acp_" + "0a" * 12, "a2a": "a2a_" + "0b" * 12}
_RESERVED_FOR = {
    "acp": "conversations an editor opens",
    "a2a": "conversations another agent starts with a clone",
}


@pytest.mark.parametrize("head", ["acp", "a2a"])
@pytest.mark.parametrize("creator", [None, "run"])
def test_only_its_head_creates_a_room_under_a_server_head_s_id(
    service: RoomService, head: str, creator: str | None
) -> None:
    """Only its head makes a room under that id, since that head alone derives it.

    `ucx room create --id acp_<24 hex>` made a room its maker could then neither rename
    nor add to (#1885). Author's choice: another head is refused too, so the shape is one
    only its own head mints.

    Killed by: src/uclone_x/room/service.py :: if owner is not None and owner != head:
    Becomes: if False:
    Killed by: src/uclone_x/room/service.py :: if owner is not None and owner != head:
    Becomes: if owner is not None:
    """
    room_id = _SERVER_HEAD_ID[head]

    with pytest.raises(RoomError) as refused:
        service.create("scout", room_id=room_id, head=creator)

    assert str(refused.value) == (
        f"Room id {room_id!r} is reserved: ids of this form belong to "
        f"{_RESERVED_FOR[head]}. Choose another id, or leave it out to get a new one."
    )
    assert service.list_rooms() == ()
    assert service.create("scout", room_id=room_id, head=head).head == head


def test_a_head_is_refused_another_head_s_id_before_its_turn(service: RoomService) -> None:
    """`ucx run --session-id acp_<24 hex>` stops before the model is asked (#1885).

    The room is written only after the reply, so a refusal there came after the turn had
    run and its reply had been shown.

    Killed by: src/uclone_x/room/one_seat.py :: refuse_another_heads_id(room_id, head, label=id_label)
    Becomes: pass
    """
    from uclone_x.room.one_seat import resolve_one_seat_room

    with pytest.raises(RoomError, match="is reserved"):
        resolve_one_seat_room(service, room_id=_SERVER_HEAD_ID["acp"], clone_id="scout", head="run")

    own = resolve_one_seat_room(
        service, room_id=_SERVER_HEAD_ID["acp"], clone_id="scout", head="acp"
    )
    assert own.state is None


def test_run_s_refusal_calls_the_id_what_the_person_typed(tmp_path: Path) -> None:
    """`ucx run --session-id acp_<24 hex>` is told about its session id, not a room id (#1900).

    Killed by: src/uclone_x/room/one_seat.py :: refuse_another_heads_id(room_id, head, label=id_label)
    Becomes: refuse_another_heads_id(room_id, head)
    Killed by: src/uclone_x/room/one_seat.py :: id_label="Session id",
    Becomes: id_label="Room id",
    """
    from uclone_x.room.one_seat import open_head_room
    from uclone_x.room.store import RoomStore

    room_id = _SERVER_HEAD_ID["acp"]
    with pytest.raises(RoomError) as refused:
        open_head_room("scout", room_id, head="run", store=RoomStore(tmp_path / "rooms"))

    assert str(refused.value) == (
        f"Session id {room_id!r} is reserved: ids of this form belong to "
        f"{_RESERVED_FOR['acp']}. Choose another id, or leave it out to get a new one."
    )


# --------------------------------------------------------------------------------------
# A room a head keeps (#1885)
# --------------------------------------------------------------------------------------


class TestAHeadRoomIsWrittenOnlyByItsHead:
    """Every change a person asks the service for is refused in a head's room.

    The app's route guard reads the room from the path; this is the check under it, so a
    route that names the room in its body and the `ucx room` commands refuse the same way.
    """

    #: Methods that save a room without `_writable`, each for a stated reason: `create` is
    #: how a head makes its room, and `forget_story` only takes a deleted story out of the
    #: rooms that name it. `set_story` checks only when it opens a story, not when it
    #: clears one (author's choice: clearing is the library's cleanup, not a person
    #: writing into the room).
    _EXEMPT = frozenset({"create", "forget_story"})

    def test_every_method_that_saves_a_room_goes_through_the_head_check(self) -> None:
        """Read from the service's source, so a new saving method fails until it is decided.

        Killed by: src/uclone_x/room/service.py :: state = self.get(room_id) if story_id is None else self._writable(room_id)
        Becomes: state = self.get(room_id)
        """
        import ast
        import inspect

        import uclone_x.room.service as service_module

        tree = ast.parse(inspect.getsource(service_module))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoomService")
        saving: dict[str, bool] = {}
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            calls = {ast.unparse(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)}
            if "self._store.save" in calls:
                saving[fn.name] = "self._writable" in calls

        assert {"rename", "add_participant", "remove_participant"} <= set(saving)
        unguarded = {name for name, guarded in saving.items() if not guarded}
        assert unguarded == self._EXEMPT, unguarded

    @staticmethod
    def _head_room(service: RoomService, *, head: str) -> str:
        from uclone_x.room.one_seat import HeadTurn, conversation_room_id, record_head_turn
        from uclone_x.room.store import RoomStore

        store = service._store  # pyright: ignore[reportPrivateUsage]
        assert isinstance(store, RoomStore)
        state = record_head_turn(
            store,
            room_id="room_from_the_terminal"
            if head == "run"
            else conversation_room_id(head, "scout", "c1"),
            clone_id="scout",
            turn=HeadTurn(prompt="what is in the index?", content="three tables"),
            head=head,
        )
        return state.room_id

    @pytest.mark.parametrize("head", ["run", "a2a"])
    @pytest.mark.parametrize(
        "change",
        ["rename", "add", "remove", "responder", "truncate", "clear", "open_story"],
    )
    def test_each_change_is_refused_plainly_and_nothing_is_saved(
        self, service: RoomService, change: str, head: str
    ) -> None:
        """A terminal head's room and a server head's: both are the head's alone.

        Killed by: src/uclone_x/room/service.py :: head = state.head
        Becomes: head = None
        """
        from uclone_x.errors import HeadRoomWriteError

        room_id = self._head_room(service, head=head)
        before = service.get(room_id)
        calls = {
            "rename": lambda: service.rename(room_id, "Mine now"),
            "add": lambda: service.add_participant(room_id, "critic"),
            "remove": lambda: service.remove_participant(room_id, "scout"),
            "responder": lambda: service.set_default_responder(room_id, "scout"),
            "truncate": lambda: service.truncate_transcript(room_id, 1),
            "clear": lambda: service.clear_transcript(room_id),
            "open_story": lambda: service.set_story(room_id, "story_night_train"),
        }

        with pytest.raises(HeadRoomWriteError) as refused:
            calls[change]()

        keeper = "ucx run" if head == "run" else "the agent that called this clone"
        assert str(refused.value) == (
            f"This conversation belongs to {keeper}, so only {keeper} can continue it. "
            "You can read it here."
        )
        assert service.get(room_id) == before


class TestDeleteRoom:
    def test_seat_sessions_lists_roster_and_departed_agents(self) -> None:
        """Every seat session used by the room is returned: roster agents first, then departed.

        Human participants are excluded.

        Killed by: src/uclone_x/room/service.py :: _add(participant_session_id(state.room_id, m.sender_id))
        Becomes: pass
        """
        state = RoomState(
            room_id="room_seats",
            title="Seats",
            participants=(
                Participant(id="user", display_name="User", kind=ParticipantKind.HUMAN),
                Participant(
                    id="alpha",
                    display_name="Alpha",
                    kind=ParticipantKind.AGENT,
                    session_id=participant_session_id("room_seats", "alpha"),
                ),
                Participant(
                    id="beta",
                    display_name="Beta",
                    kind=ParticipantKind.AGENT,
                    session_id=participant_session_id("room_seats", "beta"),
                ),
            ),
            transcript=(
                RoomMessage(seq=1, sender_id="user", content="hello"),
                RoomMessage(seq=2, sender_id="alpha", turn_id="t1", content="hi"),
                RoomMessage(
                    seq=3,
                    sender_id="departed_agent",
                    kind=RoomMessageKind.JOIN,
                    content="departed joined",
                ),
                RoomMessage(
                    seq=4,
                    sender_id="departed_agent",
                    turn_id="t2",
                    content="departed message",
                ),
                RoomMessage(
                    seq=5,
                    sender_id="departed_agent",
                    kind=RoomMessageKind.LEAVE,
                    content="departed left",
                ),
            ),
        )

        sessions = seat_sessions(state)
        assert sessions == (
            participant_session_id("room_seats", "alpha"),
            participant_session_id("room_seats", "beta"),
            participant_session_id("room_seats", "departed_agent"),
        )

    def test_deleting_a_room_removes_each_seats_session_record_before_the_room_record(
        self, tmp_path: Path
    ) -> None:
        """Seat session records are deleted before the room record itself.

        Killed by: src/uclone_x/room/service.py :: self._sessions.delete(session_id)
        Becomes: pass
        """
        events: list[str] = []

        class RecordingSessionStore:
            def delete(self, session_id: str) -> bool:
                events.append(f"delete_session:{session_id}")
                return True

        class RecordingRoomStore(RoomStore):
            def delete(self, room_id: str) -> bool:
                events.append(f"delete_room:{room_id}")
                return super().delete(room_id)

        store = RecordingRoomStore(tmp_path / "rooms")
        session_store = RecordingSessionStore()
        service = RoomService(store, sessions=session_store)

        state = service.create(
            "Discussion",
            room_id="room_del",
            seats=[("user", ParticipantKind.HUMAN), ("scout", ParticipantKind.AGENT)],
        )

        assert service.delete(state.room_id) is True
        scout_session_id = participant_session_id("room_del", "scout")
        assert events == [
            f"delete_session:{scout_session_id}",
            "delete_room:room_del",
        ]

    def test_deleting_a_room_keeps_the_room_when_seat_session_removal_fails(
        self, tmp_path: Path
    ) -> None:
        """When removing a seat's session fails, RoomError is raised and the room is kept."""

        class FailingSessionStore:
            def delete(self, session_id: str) -> bool:
                raise OSError("simulated disk error")

        store = RoomStore(tmp_path / "rooms")
        service = RoomService(store, sessions=FailingSessionStore())

        state = service.create(
            "Persistent Room",
            room_id="room_keep",
            seats=[("user", ParticipantKind.HUMAN), ("scout", ParticipantKind.AGENT)],
        )

        with pytest.raises(RoomError, match="could not be deleted because its seat session"):
            service.delete(state.room_id)

        assert service.get("room_keep").room_id == "room_keep"


# --------------------------------------------------------------------------------------
# Workspace per conversation (clone-data-scopes §3.6)
# --------------------------------------------------------------------------------------


def test_a_workspace_that_is_not_a_folder_or_with_a_story_open_is_refused(
    service: RoomService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    room = service.create("Refused")
    with pytest.raises(RoomError):
        service.set_workspace(room.room_id, "relative/path")
    with pytest.raises(RoomError):
        service.set_workspace(room.room_id, str(tmp_path / "missing"))
    # Inside the app's own state, which the store writes and no workspace may hold.
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "state"))
    (tmp_path / "state" / "rooms").mkdir(parents=True)
    with pytest.raises(RoomError):
        service.set_workspace(room.room_id, str(tmp_path / "state" / "rooms"))
    service.set_story(room.room_id, "story_1")
    with pytest.raises(RoomError):
        service.set_workspace(room.room_id, str(tmp_path))
    assert service.get(room.room_id).workspace is None
