"""Room lifecycle: making a room, naming it, and changing who is in it.

**Why this is not on `RoomOrchestrator`.** The orchestrator is the *turn* authority — it
runs the selector chain, gives the floor, and is the room's single writer while a loop is
running. Constructing it costs a selector chain and an agent resolver, and neither has
anything to say about whether a room exists or who is seated in it. Hanging `create` on it
would mean a caller that only wants to make an empty room must first assemble a model
client and an agent registry, which is the shape that kept `ucx` from having a `room`
command at all. So roster lifecycle takes a `RoomStoreProtocol` and nothing else, and the
CLI can be the thin shell over the Core that P8 asks for.

The two do not race. Both write through the store's compare-and-swap, and every method
here re-reads immediately before it writes, so a roster edit made while a turn loop is
running is refused as a stale write rather than silently losing the loop's utterances —
the same refusal, for the same reason, as two orchestrators on one room.

**Every roster change is written into the transcript.** A reader asks "why did critic stop
replying?" of the conversation, so the conversation is where the answer has to be; see
`RoomMessageKind` for why that row is a kind rather than a side-channel, and what it costs.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.agent.session import validate_session_id
from uclone_x.core.agent_home import AgentHomeError, refuse_an_unusable_username
from uclone_x.errors import (
    PathTraversalError,
    RoomAlreadyExistsError,
    RoomError,
    RoomIdError,
    RoomNotFoundError,
    SecondHumanInRoomError,
    UnknownRoomParticipantError,
    UnreadableRoomRecordError,
)
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomFileRecord,
    RoomMessage,
    RoomMessageKind,
    RoomPolicy,
    RoomState,
)
from uclone_x.room.protocols import RoomStoreProtocol

__all__ = [
    "ONTOLOGY_NAMESPACE_ROOT",
    "SESSION_ID_PREFIX",
    "SESSION_ID_SEPARATOR",
    "RoomListing",
    "RoomService",
    "RoomSummary",
    "participant_session_id",
]

logger = logging.getLogger(__name__)

#: Root IRI under which each room participant's own ontology namespace is minted. One
#: namespace per `(room, agent)` pair, never shared: merging two agents' induced concepts
#: destroys the per-agent grounding P7 requires.
ONTOLOGY_NAMESPACE_ROOT = "https://uclone-x.ai/ontology"


#: Prefix for derived room-scoped session ids. Follows the UClone-X `sess_*` convention
#: to prevent namespace confusion with room identifiers (`room_*`).
SESSION_ID_PREFIX = "sess_room"

#: Separator between the parts of a derived session id. `__` and not `:`:
#: `validate_session_id` forbids path separators and `..` but admits a colon, which then
#: fails to be a filename on Windows.
SESSION_ID_SEPARATOR = "__"


def participant_session_id(room_id: str, participant_id: str) -> str:
    """The session an agent keeps for its part in one room.

    Derived here rather than chosen by a caller so that two participants cannot be given
    one session — the failure mode is two writers on one `SessionState`, which the session
    store refuses on its revision precondition, at the end of a turn rather than at the
    roster edit that caused it.

    **Injective only over halves that hold no `__` themselves**, which is why the two write
    paths refuse one that does. Room `a` seating `b__c` and room `a__b` seating `c` both
    land on `sess_room__a__b__c`: two participants of two different rooms on one session, with
    neither room's own roster looking wrong and no resolver able to see it — each holds one
    room's claims, and each set is internally consistent.
    """
    return (
        f"{SESSION_ID_PREFIX}{SESSION_ID_SEPARATOR}{room_id}{SESSION_ID_SEPARATOR}{participant_id}"
    )


def _refuse_an_id_the_derivation_cannot_carry(label: str, value: str) -> None:
    """Refuse an id that cannot become one unambiguous session name.

    Applied to both halves of `participant_session_id`, at the two places that write them.
    Every refusal here is a failure that would otherwise surface somewhere else entirely —
    inside the session store, or in another room — with an error naming the derived id and
    not the roster edit that produced it.

    Raises:
        RoomError: The id is blank, carries surrounding whitespace, or contains the
            separator the derivation reserves.
    """
    if not value or value != value.strip():
        raise RoomError(
            f"{label} {value!r} is not usable: an id is blank, or padded with whitespace "
            f"that renders as nothing and addresses as something. It becomes a session "
            f"name and a namespace IRI, so it has to be a name."
        )
    if SESSION_ID_SEPARATOR in value:
        raise RoomError(
            f"{label} {value!r} contains {SESSION_ID_SEPARATOR!r}, which separates the two "
            f"halves of a derived session id. An id holding it makes the derivation "
            f"ambiguous: room 'a' seating 'b__c' and room 'a__b' seating 'c' would share "
            f"one session, which is two writers on one record and the isolation the room "
            f"derives these ids to guarantee."
        )


def _floor_after(
    transcript: tuple[RoomMessage, ...],
    participants: tuple[Participant, ...],
    *,
    spent: int,
) -> dict[str, object]:
    """The floor state a transcript implies, for the two history routes to write back.

    Only the two fields a rewind actually invalidates. `agent_turns_since_human` is the
    cascade ceiling's counter, so carrying one from dropped turns either strands the rewound
    room at its limit or hands it a budget it never spent; `last_speaker_id` names a turn
    that may no longer be in the record.

    Membership rows are skipped, matching the orchestrator, which counts a *turn* and a join
    is not one. A sender who is not a seated human is counted as an agent rather than looked
    up in the roster, so an agent that has since left still counts the turns it took — the
    ceiling bounds agent-to-agent traffic, and a departure does not retroactively unspend it.

    **The count is capped at `spent`, the counter the room already carried, because the
    transcript is not sufficient to recompute it.** Two turns cost the live counter less than
    a recount charges them:

    * a *retried* turn. `RoomOrchestrator.retry` refunds the counter and **keeps** the failed
      row (its own docstring: "The failed row is kept"), so the refund exists nowhere in the
      record and a recount bills the failure and its retry both;
    * an utterance by a human who has since left, which no longer matches `humans` and so
      reads as an agent turn.

    A rewind only ever *removes* rows, so the ceiling it implies can never honestly exceed
    the one already in force — the cap is that inequality, not a guess. It matters in one
    direction only: uncapped, a rewind can push a room to its ceiling and the next agent turn
    is refused with nothing in the record accounting for it, which is exactly the silent
    divergence between reader and speakers these two methods exist to prevent.
    """
    humans = {p.id for p in participants if p.kind is ParticipantKind.HUMAN}
    since_human = 0
    for message in reversed(transcript):
        if not message.is_utterance:
            continue
        if message.sender_id in humans:
            break
        since_human += 1
    last = next((m for m in reversed(transcript) if m.is_utterance), None)
    return {
        "agent_turns_since_human": min(since_human, spent),
        "last_speaker_id": last.sender_id if last is not None else None,
    }


def _clean_title(title: str) -> str:
    """Strip a title and refuse the three shapes a room title must never carry.

    Shared by `RoomService.create` and `RoomService.rename` rather than written twice. The
    two are the only ways a title is ever set, and a rule enforced at one of them is not a
    rule: a newline refused at creation and admitted at rename still reaches every
    single-line table and every prompt that renders the title, by the later door.

    Raises:
        RoomError: The title is blank, carries a newline, or carries an ANSI escape.
    """
    cleaned = title.strip()
    if not cleaned:
        raise RoomError(
            "A room needs a title: it is how a person finds this conversation again, "
            "and an untitled room is a hex id in a list"
        )
    if "\n" in cleaned or "\r" in cleaned:
        raise RoomError(
            "A room title must not contain newlines: it is rendered in single-line tables and prompts."
        )
    if re.search(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", cleaned):
        raise RoomError(
            "A room title must not contain ANSI escape sequences: it is stored and rendered to terminals."
        )
    return cleaned


def participant_namespace(room_id: str, participant_id: str) -> str:
    """The ontology namespace an agent keeps for its part in one room (P7, G4)."""
    return f"{ONTOLOGY_NAMESPACE_ROOT}/{room_id}/{participant_id}"


class RoomSummary(BaseModel):
    """What a room looks like in a list, without loading the conversation to find out.

    Exists so the listing surface is a Core value rather than a rendering decision made
    twice — once by the CLI table and again by whatever HTTP route follows it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    room_id: str
    title: str = Field(description="Empty for a room written before titles existed.")
    agent_ids: tuple[str, ...]
    human_ids: tuple[str, ...]
    message_count: int = Field(description="Every transcript row, membership rows included.")
    updated_at: str = Field(
        description="The room record's last write, as the store stamped it: a message, a "
        "reply, a rename, a roster change, or a composing notice. What `list_rooms` orders by."
    )


class RoomListing(BaseModel):
    """Every stored room: the ones that load, and the ids of the ones that do not (#1440).

    `unreadable` exists so a record that will not load is still a row somewhere. Left out
    of the listing with only a log line, the reader's conversation vanished with no trace
    on screen, and nothing offered a way to remove it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    rooms: tuple[RoomSummary, ...]
    unreadable: tuple[str, ...] = Field(
        description="Ids of records that are there and will not load, sorted."
    )


#: Where a room whose `updated_at` will not parse sorts: after every readable one.
_UNREADABLE_INSTANT = datetime.min.replace(tzinfo=UTC)


def _updated_instant(updated_at: str) -> datetime:
    """The moment `updated_at` names, compared as an instant rather than as a string.

    The store only writes UTC, but a record carried from another machine or edited by hand
    need not be, and `10:00+09:00` sorts after `05:00+00:00` as text while naming a moment
    four hours earlier. A naive stamp is read as UTC; one that is not a date sorts last
    rather than failing the listing, which is the only way to find the other rooms.
    """
    try:
        parsed = datetime.fromisoformat(updated_at)
    except ValueError:
        return _UNREADABLE_INSTANT
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _by_recency(summaries: Iterable[RoomSummary]) -> list[RoomSummary]:
    """Most recently updated first; equal stamps by `room_id` ascending (#1053).

    Two stable sorts: the tiebreak first, then the key, so the key's ties keep the
    tiebreak's order.
    """
    ordered = sorted(summaries, key=lambda s: s.room_id)
    ordered.sort(key=lambda s: _updated_instant(s.updated_at), reverse=True)
    return ordered


class RoomService:
    """Creates rooms and edits their rosters. The Core half of `ucx room`."""

    def __init__(self, store: RoomStoreProtocol) -> None:
        self._store = store

    # -- rooms -------------------------------------------------------------------------

    def create(
        self,
        title: str,
        *,
        room_id: str | None = None,
        policy: RoomPolicy | None = None,
    ) -> RoomState:
        """Create and persist an empty, titled room.

        Empty on purpose: a participant arrives through `add_participant`, so the joins
        that formed the room are in its transcript like any other roster change, and there
        is one code path that seats somebody rather than two that must agree.

        Raises:
            RoomError: `title` is blank. A room whose title is whitespace is the opaque
                record the field was added to end, and minting one here would reintroduce
                it while looking like a feature. Or `room_id` is one the session-id
                derivation cannot carry unambiguously.
            RoomAlreadyExistsError: `room_id` already names a stored room.
        """
        cleaned = _clean_title(title)

        # `room_id` omitted means "mint one", which `RoomState`'s default factory does.
        # Passing `None` through would be a different thing — an explicitly null id — so
        # the argument is left out rather than forwarded.
        # The one place a room is born with a complete file record: every row it will ever
        # hold is written by a build that counts what the record cannot see (#1366).
        seed: dict[str, object] = {
            "title": cleaned,
            "policy": policy or RoomPolicy(),
            "file_record": RoomFileRecord(kept_since_creation=True),
        }
        if room_id is not None:
            _refuse_an_id_the_derivation_cannot_carry("Room id", room_id)
            seed["room_id"] = room_id
        state = RoomState.model_validate(seed)
        if self._store.load(state.room_id) is not None:
            raise RoomAlreadyExistsError(
                f"Room {state.room_id!r} already exists; refusing to overwrite its "
                f"conversation. Pick another id, or open the one that is there."
            )
        return self._store.save(state)

    def get(self, room_id: str) -> RoomState:
        """Return the stored room.

        Raises:
            RoomNotFoundError: No room under that id.
        """
        state = self._store.load(room_id)
        if state is None:
            # Cause and remedy both: a head shows this verbatim when a row it still lists
            # was deleted elsewhere, and "missing" alone leaves that reader nowhere to go.
            raise RoomNotFoundError(
                f"No room {room_id!r} in the store: it has been deleted, or was never "
                f"created. List the rooms to see the ones that exist."
            )
        return state

    def rename(self, room_id: str, title: str) -> RoomState:
        """Give a stored room a new title, leaving everything else alone.

        Exists because `RoomState.title` refuses to be *derived* from the conversation --
        a derived title moves when the opening message is edited or compacted, and an
        identifier that moves is not one. A head that does not want to demand a title
        before a first-time user has said anything therefore seeds one at creation and
        needs this to repair it; seeding with no repair makes whatever sentence somebody
        happened to type first the permanent name of the conversation.

        A label change and nothing more: the roster, the transcript and the floor are
        carried through untouched, so a rename is not a place a conversation can be lost.

        Raises:
            RoomNotFoundError: No room under that id.
            RoomError: The title is blank, carries a newline, or carries an ANSI escape --
                the same three refusals `create` makes, by the same helper.
            StaleRoomWriteError: Another writer moved the room first.
        """
        cleaned = _clean_title(title)
        state = self.get(room_id)
        return self._store.save(state.model_copy(update={"title": cleaned}))

    def delete(self, room_id: str) -> bool:
        """Remove the room; return whether one was there to remove."""
        return self._store.delete(room_id)

    def list_rooms(self) -> tuple[RoomSummary, ...]:
        """The rooms that load, as `survey_rooms` lists them."""
        return self.survey_rooms().rooms

    def survey_rooms(self) -> RoomListing:
        """Summarise every stored room, by title rather than by id, most recent first.

        **The order is the Core's, not each head's** (#1053). Most recently updated first,
        equal stamps by `room_id` ascending. It is decided here because a second head on
        the same Core needs the same order, and one that sorted for itself would be free
        to disagree with this one; see `docs/ui-dashboard-architecture.md` §3.

        A room whose document will not validate is skipped rather than raising: one
        unreadable record must not make the listing — the only way to find any of the
        others — fail wholesale. A stem the store will not address is skipped on the same
        reasoning (#1258): the tolerance is about the *directory*, and a name the guard
        refuses fails one row earlier than a body that will not parse.

        A skipped unreadable record is still *reported*, by id, in `unreadable` (#1440): it
        costs its row in `rooms` and nothing more, and the listing says it is there. A stem
        the store will not address is not reported: no route can act on it by that id.
        """
        summaries: list[RoomSummary] = []
        unreadable: list[str] = []
        for room_id in self._store.list_room_ids():
            try:
                state = self._store.load(room_id)
            except RoomIdError:
                # The same tolerance as the unreadable record below, one step earlier: that
                # one is a document the loader will not parse, this one is a *stem* the path
                # guard will not address. `list_room_ids` reports whatever `*.json` is in the
                # directory, so a file nobody's room wrote — `...json`, whose stem is `..` —
                # otherwise took the whole listing down with a bare 500 (#1258). Translating
                # it in the route was the wrong fix: a 400 tells the caller they made a bad
                # request, and they did not.
                #
                # **It is logged, and the silence next door is not the precedent to follow.**
                # The question is whether a skipped stem can be a real room disappearing, and
                # it can: `room_path` has two halves, and only the *name* half is about junk
                # files. The containment half refuses a lexically clean stem whose record is
                # a symlink pointing out of the directory — that is a real room's filename,
                # one `get` still answers by id, silently absent from the only view that
                # lists it. A room that vanishes from the listing with no trace anywhere is
                # not a thing to be quiet about. And the exception is a `PathTraversalError`
                # besides: discarding a containment refusal without a record destroys the one
                # event whose trace is worth most.
                logger.warning(
                    "Room file %r is named something the store will not address; it is "
                    "omitted from the listing",
                    room_id,
                    exc_info=True,
                )
                continue
            except UnreadableRoomRecordError:
                # Caught, because the listing is the only way to find any of the *other*
                # rooms: a document that refuses to validate — an older shape, a hand-edited
                # file, a roster this build no longer admits — must cost its own row and
                # nothing more. Not swallowed elsewhere: `get` still raises, so a caller who
                # names the bad room is told, and only the survey is made tolerant.
                logger.warning(
                    "Room %r is stored in a shape this build will not load; it is listed "
                    "as unreadable",
                    room_id,
                    exc_info=True,
                )
                unreadable.append(room_id)
                continue
            if state is None:
                continue
            summaries.append(
                RoomSummary(
                    room_id=state.room_id,
                    title=state.title,
                    agent_ids=tuple(
                        p.id for p in state.participants if p.kind is ParticipantKind.AGENT
                    ),
                    human_ids=tuple(
                        p.id for p in state.participants if p.kind is ParticipantKind.HUMAN
                    ),
                    message_count=len(state.transcript),
                    updated_at=state.updated_at,
                )
            )
        return RoomListing(rooms=tuple(_by_recency(summaries)), unreadable=tuple(unreadable))

    # -- the roster --------------------------------------------------------------------

    def add_participant(
        self,
        room_id: str,
        participant_id: str,
        *,
        kind: ParticipantKind = ParticipantKind.AGENT,
        display_name: str = "",
        persona_summary: str = "",
        aliases: tuple[str, ...] = (),
        persona: str = "",
    ) -> RoomState:
        """Seat a participant, deriving its own session and ontology namespace.

        The isolation obligation (G3, G4) is discharged *here*, where the roster is
        written, rather than by the resolver that later builds the agent: a `RoomState`
        then answers "do these two share a session?" by inspection, which is what putting
        the fields on `Participant` was for.

        Raises:
            RoomNotFoundError: No such room.
            RoomError: The id is one the derivation cannot carry, it collides
                case-insensitively with a seated participant, or the session id it derives
                is not one the session store will name.
        """
        _refuse_an_id_the_derivation_cannot_carry("Participant id", participant_id)
        state = self.get(room_id)
        # Case-insensitively, because two ids that differ only by case derive two session
        # ids that differ only by case, and on macOS and Windows those are one file.
        # `BaseAgent.hydrate_session` then refuses a record identifying a different session
        # (#256), so the second agent cannot hydrate at all — a departure from the roster
        # that the roster is the last place able to see.
        clash = next(
            (p for p in state.participants if p.id.casefold() == participant_id.casefold()),
            None,
        )
        if clash is not None:
            raise RoomError(
                f"{participant_id!r} collides with participant {clash.id!r} of room "
                f"{room_id!r}. Ids are how the room addresses a participant, so two of "
                f"them cannot share one — and two that differ only by case derive two "
                f"session records that a case-insensitive filesystem stores as one."
            )

        seated_human = next(
            (p for p in state.participants if p.kind is ParticipantKind.HUMAN), None
        )
        if kind is ParticipantKind.HUMAN and seated_human is not None:
            # Refused *here*, with the reason, rather than several steps later as a stale
            # write nobody can act on. `RoomState` refuses the same roster, but it is the
            # backstop for a record built by another route; this is the message a person
            # reads. The seat is not permanently claimed — removing the human frees it.
            raise SecondHumanInRoomError(
                f"Room {room_id!r} already seats the human {seated_human.id!r}, so "
                f"{participant_id!r} cannot join it: a room serves one human. Two humans "
                f"posting at once collide on the transcript's single-writer precondition "
                f"and the second message is refused rather than merged, and a second human "
                f"has no record of what they have already read. Give {participant_id!r} "
                f"their own room, or remove {seated_human.id!r} from this one first."
            )

        is_agent = kind is ParticipantKind.AGENT
        if is_agent:
            # The name rule the session store applies, applied at the seat instead of at
            # the agent's first write. Left to the store, a traversal in a participant id
            # is refused several steps later, by which point the join is in the transcript
            # and the error names a derived session id nobody typed.
            try:
                validate_session_id(participant_session_id(room_id, participant_id))
            except PathTraversalError as exc:
                raise RoomError(
                    f"{participant_id!r} cannot be seated in room {room_id!r}: the session "
                    f"id it derives is not one the session store will name ({exc})."
                ) from exc
            # And the name rule its *home directory* applies, for the same reason and
            # last, so that an id breaking one of the older rules is still refused by the
            # rule it breaks. An agent's id is the directory holding its id and memory, so
            # an id no directory can carry is a seat whose first turn cannot load memory --
            # a refusal that would otherwise arrive from the memory factory, naming a rule
            # this call never applied.
            try:
                refuse_an_unusable_username(participant_id)
            except AgentHomeError as exc:
                raise RoomError(
                    f"{participant_id!r} cannot be seated in room {room_id!r} as an agent: {exc}"
                ) from exc
        effective_summary = persona_summary
        if is_agent and not effective_summary:
            lookup_key = persona or participant_id
            try:
                from uclone_x.agent.persona_registry import get_default_persona_registry

                persona_def = get_default_persona_registry().get_persona(lookup_key)
                if persona_def is not None:
                    effective_summary = persona_def.description or persona_def.role
            except Exception:
                pass

        participant = Participant(
            id=participant_id,
            kind=kind,
            # Stripped, and falling back to the id: `"   "` is truthy, so an all-whitespace
            # name was kept and every render of that participant — the join row included —
            # showed a gap where a name goes.
            display_name=display_name.strip() or participant_id,
            persona_summary=effective_summary,
            aliases=aliases,
            # A human has no agent session and no ontology of its own; stamping one would
            # claim a record that nothing writes.
            session_id=participant_session_id(room_id, participant_id) if is_agent else "",
            ontology_namespace=participant_namespace(room_id, participant_id) if is_agent else "",
            persona=persona if is_agent else "",
        )
        note = self._membership_row(
            state,
            participant_id,
            RoomMessageKind.JOIN,
            f"{participant.display_name} ({participant_id}) joined the room "
            f"as {'an' if kind is ParticipantKind.AGENT else 'a'} {kind.value}",
        )
        return self._store.save(
            state.model_copy(
                update={
                    "participants": (*state.participants, participant),
                    "transcript": (*state.transcript, note),
                }
            )
        )

    def remove_participant(self, room_id: str, participant_id: str) -> RoomState:
        """Unseat a participant and record the departure.

        `last_seen_seq` is deliberately **kept**. The mark says what that agent has already
        been shown, and its own session still holds it; clearing it would replay the whole
        room into that session on a rejoin, duplicating the conversation inside the record
        that is supposed to be the agent's memory of it.

        A departing agent that the policy names as the room's default responder also clears
        that field, and the leave row says so. `DefaultResponderSelector` *raises* on a
        responder who is not an agent of the room — correctly, since it is a
        misconfiguration — so leaving the field set would convert an ordinary departure
        into a room where every later unaddressed message fails selection, at a distance of
        several turns from the removal that caused it.

        Raises:
            RoomNotFoundError: No such room.
            UnknownRoomParticipantError: Nobody with that id is seated.
        """
        state = self.get(room_id)
        leaving = next((p for p in state.participants if p.id == participant_id), None)
        if leaving is None:
            roster = ", ".join(p.id for p in state.participants)
            raise UnknownRoomParticipantError(
                f"{participant_id!r} is not a participant of room {room_id!r} "
                f"(participants: {roster or 'none'})"
            )

        update: dict[str, object] = {}
        text = f"{leaving.display_name} ({participant_id}) left the room"
        if state.policy.default_responder_id == participant_id:
            update["policy"] = state.policy.model_copy(update={"default_responder_id": ""})
            text += "; it was this room's default responder, so the room now has none"

        note = self._membership_row(state, participant_id, RoomMessageKind.LEAVE, text)
        update["participants"] = tuple(p for p in state.participants if p.id != participant_id)
        update["transcript"] = (*state.transcript, note)
        return self._store.save(state.model_copy(update=update))

    def set_default_responder(self, room_id: str, agent_id: str) -> RoomState:
        """Name the agent that answers an unaddressed message, or clear it with `""`.

        The setting was previously write-once and lose-once: only `create` could give it,
        and `remove_participant` clears it when that agent leaves — correctly, since
        `DefaultResponderSelector` raises on a responder who is not in the room, so an
        uncleared one turns an ordinary departure into a room that fails its next
        unaddressed message. With nothing able to name another, a room that lost its
        responder had lost it for good.

        The membership check belongs to the caller that can report it usefully: this
        refuses only what would corrupt the record.

        Raises:
            RoomNotFoundError: No such room.
            RoomError: `agent_id` is not a seated agent of the room.
        """
        state = self.get(room_id)
        if agent_id:
            seated = {p.id for p in state.participants if p.kind is ParticipantKind.AGENT}
            if agent_id not in seated:
                raise RoomError(
                    f"{agent_id!r} is not a seated agent of room {room_id!r} "
                    f"(agents: {', '.join(sorted(seated)) or 'none'})"
                )
        policy = state.policy.model_copy(update={"default_responder_id": agent_id})
        return self._store.save(state.model_copy(update={"policy": policy}))

    # -- history -----------------------------------------------------------------------

    def truncate_transcript(self, room_id: str, seq: int) -> RoomState:
        """Rewind the conversation to `seq`, keeping that message and dropping what follows.

        **The transcript is one record with several speakers, so a rewind is the room's act
        and not a seat's.** Cutting one participant's context back would leave every row it
        produced still on screen with nobody left who remembers writing them — a
        conversation whose reader and whose speakers disagree about what happened. The
        caller is responsible for the other half, resetting each seated agent's own session;
        this method owns the record.

        **`last_seen_seq` is clamped to the new tail, and that is not bookkeeping.** `seq` is
        assigned as `len(transcript) + 1` everywhere it is assigned, so a truncation *reuses*
        the numbers it freed: rewinding a room of twelve to seven means the next utterance is
        seq 8 again. An agent's mark left at 12 is read by
        `RoomOrchestrator._unseen_span` as `m.seq > last_seen`, so that agent would be handed
        nothing until the room grew past twelve a second time — five replacement turns that
        every other participant can see and it cannot, with no notice anywhere saying so. The
        mark is therefore cut back with the record it indexes.

        **`turn_state` is recounted, not carried.** `agent_turns_since_human` bounds the
        agent-to-agent cascade (`RoomPolicy.max_agent_turns_per_human_message`); a count
        inherited from turns that have just been dropped either strands the rewound room at
        its ceiling or spends a budget nobody used. It is re-derived from what survives, as
        is `last_speaker_id` — but only ever *downwards*: the record alone cannot recompute
        the counter, because a retry's refund is not written into it, so the re-derived count
        is capped at the one already in force (`_floor_after`). `last_activity_ts` and
        `races_forgiven` are kept: the first
        records when the human was last at the keyboard, which a rewind does not change, and
        the second is a lifetime count of departures forgiven, not a property of the
        transcript.

        Parameters:
            room_id: The conversation to rewind.
            seq: The last message to keep. It must be a `seq` that is in the transcript —
                see the refusal below.

        Raises:
            RoomNotFoundError: No such room.
            RoomError: No message in this room carries that `seq`. An out-of-range rewind is
                refused rather than clamped: clamping down silently destroys more than the
                caller asked for, and clamping up answers 200 over a no-op. The one thing it
                will not do is empty the room — that is `clear_transcript`, which is a
                different act on the record and says so.
        """
        state = self.get(room_id)
        if not any(m.seq == seq for m in state.transcript):
            present = [m.seq for m in state.transcript]
            span = f"{present[0]}-{present[-1]}" if present else "none; the room is empty"
            raise RoomError(
                f"Room {room_id!r} has no message at seq {seq}, so there is nothing to "
                f"rewind to (present: {span}). Name a message that is in the conversation, "
                f"or clear it instead."
            )
        kept = tuple(m for m in state.transcript if m.seq <= seq)
        spent = state.turn_state.agent_turns_since_human
        floor = _floor_after(kept, state.participants, spent=spent)
        # The calls behind the removed turns go with them, joined by turn id rather than by
        # `seq`, which the replacement turns reuse. The files they wrote do not: those are
        # still on disk, and `written_files` is the room's record of what it produced.
        kept_turns = {m.turn_id for m in kept if m.turn_id is not None}
        # And the fact that they were removed is kept, because what they showed -- a turn
        # that never reported its tools, a shell's unnamed write -- goes with them (#1366).
        record = state.file_record
        if len(kept) < len(state.transcript):
            record = record.model_copy(update={"rewinds": record.rewinds + 1})
        return self._store.save(
            state.model_copy(
                update={
                    "transcript": kept,
                    "file_record": record,
                    "tool_uses": tuple(u for u in state.tool_uses if u.turn_id in kept_turns),
                    "last_seen_seq": {
                        pid: str(min(int(mark), seq)) for pid, mark in state.last_seen_seq.items()
                    },
                    "turn_state": state.turn_state.model_copy(update=floor),
                }
            )
        )

    def clear_transcript(self, room_id: str) -> RoomState:
        """Empty the conversation in place, keeping its id, its title and its roster.

        **Not the `seq = 0` case of `truncate_transcript`**, and the difference is the one a
        caller is choosing between: this is the room you are already in, emptied, and
        `RoomService.create` is the other reset — a new conversation, with a new id, that
        does not replace this one in the list. A rewind names a message to keep and cannot
        name none; this names none and cannot be asked for a message.

        Every mark goes with the record. Unlike `remove_participant`, which keeps
        `last_seen_seq` deliberately because the agent's own session still holds what the
        mark refers to, a clear is the caller's statement that none of it is to be kept —
        and the caller resets those sessions in the same breath. A mark surviving here would
        index a transcript that no longer exists, and every replacement message would fall
        below it.

        Raises:
            RoomNotFoundError: No such room.
        """
        state = self.get(room_id)
        record = state.file_record
        if state.transcript or state.tool_uses:
            # Counted, never reset: see the same step in `truncate_transcript` (#1366).
            record = record.model_copy(update={"clears": record.clears + 1})
        return self._store.save(
            state.model_copy(
                update={
                    "transcript": (),
                    # Every call's turn is gone; the files it wrote are not (see rewind).
                    "tool_uses": (),
                    "file_record": record,
                    "last_seen_seq": {},
                    "turn_state": state.turn_state.model_copy(
                        update=_floor_after((), state.participants, spent=0)
                    ),
                }
            )
        )

    # -- helpers -----------------------------------------------------------------------

    @staticmethod
    def _membership_row(
        state: RoomState, subject_id: str, kind: RoomMessageKind, text: str
    ) -> RoomMessage:
        """Build the transcript row for a roster change.

        Carries no `decision`: nobody selected a join, and an empty decision is the honest
        record of that. `TurnState` is untouched by the caller for the same reason — a join
        is not a turn, so it neither spends the agent budget nor takes the floor.
        """
        return RoomMessage(
            seq=len(state.transcript) + 1,
            sender_id=subject_id,
            content=text,
            kind=kind,
        )
