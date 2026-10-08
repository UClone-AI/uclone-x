"""One-seat rooms for the heads that are not the desktop app (§5.9, owner ruling 2026-09-27).

Every head is a room with one clone seat: the CLI's `run` and `loop`, and the ACP and A2A
servers. A head's conversation is therefore a `RoomState` with a person and a clone
seated, and the clone keeps the session a room seat keeps (`participant_session_id`), not
a session named after the head. Sessions a head kept before this (`sess_<clone>`,
`loop_<clone>`) are left on disk as they are and are not resumed.

This module decides which room a head's conversation is, and records each turn a head
runs in that room's transcript (#1837), as a room seat's turn is recorded. The clone is
built by the caller, as each head already builds it.

A head's room is written with its first recorded turn, not when the head starts
(author's choice, #1846): a run that never reaches a turn, an ACP session that is never
prompted or whose start fails, and an A2A context that is never answered leave no room in
the list.

A room a head creates carries that head's name (`RoomState.head`), and the room
orchestrator refuses a post or a retry from anywhere else (#1885): the app shows the
conversation and cannot add to it. A room has a single owner (owner ruling); marking the
room and refusing in the Core is how this build keeps it so (author's choice).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomMessage,
    RoomState,
    RoomTurnRefusal,
    head_keeper,
    is_one_seat,
    turn_refusal,
)
from uclone_x.room.service import (
    RoomService,
    participant_session_id,
    refuse_an_underivable_id,
    refuse_another_heads_id,
)

if TYPE_CHECKING:
    from uclone_x.agent.models import ProviderFailure, ToolExecutionRecord, TurnResult
    from uclone_x.core.provenance import Provenance
    from uclone_x.llm.models import TokenUsage
    from uclone_x.room.store import RoomStore

__all__ = [
    "ONE_SEAT_HUMAN_ID",
    "HeadTurn",
    "OneSeatRoom",
    "conversation_room_id",
    "head_person_names",
    "head_room_person_names",
    "new_one_seat_room_id",
    "open_head_room",
    "open_one_seat_room",
    "record_head_turn",
    "resolve_one_seat_room",
]

#: The id the person holds in a head's one-seat room. A head has one person on the other
#: end, and the desktop app seats its person under the same id.
ONE_SEAT_HUMAN_ID = "user"


@dataclass(frozen=True, slots=True)
class OneSeatRoom:
    """A head's room: its id, the clone seat's session id, and the stored record if any.

    `state` is `None` for a room not written yet -- a new conversation, whose room is
    written with its first recorded turn (`record_head_turn`).
    """

    room_id: str
    session_id: str
    state: RoomState | None

    @property
    def created(self) -> bool:
        """Whether this is a new conversation: no room was stored under `room_id`."""
        return self.state is None

    @property
    def person_names(self) -> tuple[str, ...]:
        """The names the room's person goes by, for each turn's tools (#1893 item 1)."""
        return head_person_names(self.state)


@dataclass(frozen=True, slots=True)
class HeadTurn:
    """One turn a head ran, as its room records it: the person's prompt and the answer.

    `error` is the raw cause, as a room seat's failed row keeps it; a head that shows the
    room (the app) renders its own plain copy of it. `session_set_aside` says the save
    after this turn kept an earlier record of the seat's session aside (#1877): the clone's
    row carries it, and the app says so there, as it does on a room seat's row.
    """

    prompt: str
    content: str = ""
    error: str | None = None
    refusal: RoomTurnRefusal | None = None
    provider_failure: ProviderFailure | None = None
    provenance: Provenance | None = None
    usage: TokenUsage | None = None
    executions: tuple[ToolExecutionRecord, ...] = ()
    tools_recorded: bool = False
    completed: bool = True
    session_set_aside: bool = False

    @classmethod
    def from_result(cls, prompt: str, result: TurnResult) -> HeadTurn:
        """The turn `result` reports, read as the room orchestrator reads a seat's."""
        failed = result.error is not None
        return cls(
            prompt=prompt,
            content=result.content,
            error=result.error,
            refusal=turn_refusal(result.stop_reason) if failed else None,
            provider_failure=result.provider_failure if failed else None,
            provenance=result.provenance,
            usage=result.usage,
            executions=result.tool_executions,
            tools_recorded=result.tool_executions_complete,
        )

    @classmethod
    def failed(cls, prompt: str, cause: str, *, completed: bool = True) -> HeadTurn:
        """A turn that raised, or was interrupted (`completed=False`): no result to read.

        Its tools stay unrecorded -- with no `TurnResult` there is no list of them, and an
        empty list stored as recorded would claim the clone used none (P6).
        """
        return cls(prompt=prompt, error=cause, completed=completed)


def head_person_names(state: RoomState | None) -> tuple[str, ...]:
    """The names that mean the person in a head's room, as a room seat's turn is given them.

    A head is a one-seat room (§5.9), so its turns are given the person's names the way
    the room orchestrator gives a seat's (`_person_names`, #1857, #1868): the person's id,
    display name and aliases, less any another participant goes by. `record_memory_fact`
    then files a fact about the person under `user`, not under a name. A room not stored
    yet seats the person under `ONE_SEAT_HUMAN_ID` alone.
    """
    from uclone_x.room.orchestrator import _person_names  # pyright: ignore[reportPrivateUsage]

    return (ONE_SEAT_HUMAN_ID,) if state is None else _person_names(state)


def head_room_person_names(service: RoomService, room_id: str) -> tuple[str, ...]:
    """`head_person_names` of the room stored under `room_id`, if one is.

    For a server head (ACP, A2A), which reads its room per turn: a room is stored with its
    first turn, so the first turn of a conversation finds none. A room that will not load
    gives the person's id alone; the turn's own room write reports that failure.
    """
    from uclone_x.errors import RoomError, UnreadableRoomRecordError

    try:
        return head_person_names(service.get(room_id))
    except (RoomError, UnreadableRoomRecordError):
        return head_person_names(None)


def new_one_seat_room_id() -> str:
    """A fresh room id, for a head started without one."""
    return f"room_{uuid.uuid4().hex[:12]}"


def conversation_room_id(head: str, clone_id: str, conversation_id: str) -> str:
    """The room a server head keeps for one clone in one caller conversation.

    Keyed by both, so two clones called under one conversation id never share a room, and
    one clone called under two never does either. Digested, because a caller's id is not
    ours to trust as a file name or as a half of `participant_session_id`: it may hold the
    `__` the derivation reserves, or a path separator. `head` keeps two protocols' ids apart.
    """
    digest = hashlib.sha256(f"{head}\0{clone_id}\0{conversation_id}".encode()).hexdigest()
    return f"{head}_{digest[:24]}"


def _named(clone_id: str) -> str:
    """What a refusal calls a clone: the handle a person typed, else the id it has."""
    from uclone_x.core.agent_home import handle_of, is_agent_id

    if is_agent_id(clone_id):
        return handle_of(clone_id) or clone_id
    return clone_id


def _seat_in(state: RoomState, room_id: str, clone_id: str, head: str) -> Participant:
    """`clone_id`'s seat in the stored room, refused unless the room is its alone and `head`'s.

    A room has a single owner (owner ruling). A stored room that `head` did not create --
    one the app keeps, which carries no head, or another head's -- is refused, so `ucx run
    --session-id <a room the app shows>` cannot become a second writer beside the app
    (#1885).
    """
    from uclone_x.errors import RoomError

    seat = next(
        (p for p in state.participants if p.kind is ParticipantKind.AGENT and p.id == clone_id),
        None,
    )
    if seat is None:
        raise RoomError(
            f"Room {room_id!r} does not seat {_named(clone_id)!r}. Name a room this clone is in, "
            f"or leave the id out to start a new one."
        )
    if not is_one_seat(state.participants):
        raise RoomError(
            f"Room {room_id!r} seats other clones besides {_named(clone_id)!r}. Name a room this "
            f"clone is in alone, or leave the id out to start a new one."
        )
    if state.head != head:
        keeper = "the app" if state.head is None else head_keeper(state.head)
        raise RoomError(
            f"Room {room_id!r} belongs to {keeper}, so {head_keeper(head)} cannot continue "
            f"it. Continue it there, or leave the id out to start a new one."
        )
    return seat


def resolve_one_seat_room(
    service: RoomService, *, room_id: str, clone_id: str, head: str, id_label: str = "Room id"
) -> OneSeatRoom:
    """The room `room_id` names for `clone_id`, checked, with nothing written.

    A stored room must seat `clone_id` alone; an id with no room is a new conversation,
    written by its first `record_head_turn`. `id_label` is what an id refusal calls
    `room_id`: the name the person typed it under.

    Raises:
        RoomError: as `open_one_seat_room`.
    """
    from uclone_x.errors import RoomNotFoundError

    refuse_an_underivable_id(id_label, room_id)
    refuse_an_underivable_id("Participant id", clone_id)
    try:
        state = service.get(room_id)
    except RoomNotFoundError:
        # Refused now, not when the first turn's room is written after the reply (#1885).
        refuse_another_heads_id(room_id, head, label=id_label)
        return OneSeatRoom(
            room_id=room_id, session_id=participant_session_id(room_id, clone_id), state=None
        )
    seat = _seat_in(state, room_id, clone_id, head)
    return OneSeatRoom(
        room_id=room_id,
        session_id=seat.session_id or participant_session_id(room_id, clone_id),
        state=state,
    )


def _stored_or_created(
    service: RoomService, *, room_id: str, clone_id: str, title: str, head: str
) -> RoomState:
    """The stored room under `room_id`, or a new one seating the person and `clone_id`.

    A new room is marked with `head`, the head that keeps it (#1885). A stored room keeps
    the mark it has.
    """
    from uclone_x.errors import RoomNotFoundError

    try:
        return service.get(room_id)
    except RoomNotFoundError:
        # One save, seated (#1885 item 3): a crash cannot leave a room with nobody in it.
        return service.create(
            title=title or clone_id,
            room_id=room_id,
            head=head,
            seats=((ONE_SEAT_HUMAN_ID, ParticipantKind.HUMAN), (clone_id, ParticipantKind.AGENT)),
        )


def open_one_seat_room(
    service: RoomService,
    *,
    room_id: str,
    clone_id: str,
    head: str,
    title: str = "",
) -> OneSeatRoom:
    """Open `room_id`, or create it with the person and `clone_id` seated.

    Raises:
        RoomError: `room_id` or `clone_id` cannot carry a seat session id, or the stored
            room under `room_id` does not seat `clone_id` -- it is another clone's
            conversation, and answering in it would start a second seat there -- or it
            seats another clone beside `clone_id`. A head answers alone, without the seat
            framing and without the room's transcript, so a turn there would be a seat
            speaking past the room it sits in.
    """
    state = _stored_or_created(service, room_id=room_id, clone_id=clone_id, title=title, head=head)
    seat = _seat_in(state, room_id, clone_id, head)
    return OneSeatRoom(
        room_id=room_id,
        session_id=seat.session_id or participant_session_id(room_id, clone_id),
        state=state,
    )


def record_head_turn(
    store: RoomStore,
    *,
    room_id: str,
    clone_id: str,
    turn: HeadTurn,
    head: str,
    title: str = "",
) -> RoomState:
    """Append `turn` to the room's transcript: the person's row, then the clone's (#1837).

    `head` is the head recording it (`run`, `loop`, `acp`, `a2a`). A room this creates is
    marked as that head's, and no other surface drives a turn in it (#1885).

    Written as the room orchestrator lands a seat's turn, in one save: the clone's row
    answers the person's (`rendered_through`), its tool calls and the files they named go
    to the room's ledger under the row's `turn_id`, and the file record counts the turn as
    started and landed. `last_seen_seq` moves only on a turn without an error, so the app
    renders a failed turn's span again, as it would a seat's. The room is created here if
    this is its first turn.

    The seat's own session is the head's to save; this writes only the room.

    Raises:
        RoomError: the room cannot be opened for `clone_id` (see `open_one_seat_room`),
            or another writer moved it since it was read (`StaleRoomWriteError`).
        OSError: the store could not be written.
    """
    from uclone_x.room.orchestrator import (
        image_prompt_additions,
        memory_save_outcome,
        record_turn_tools,
    )

    state = _stored_or_created(
        RoomService(store), room_id=room_id, clone_id=clone_id, title=title, head=head
    )
    seat = _seat_in(state, room_id, clone_id, head)
    turn_id = uuid.uuid4().hex
    asked = RoomMessage(
        seq=len(state.transcript) + 1, sender_id=ONE_SEAT_HUMAN_ID, content=turn.prompt
    )
    tried, unsaved = memory_save_outcome(turn.executions)
    prompt_added, negative_added = image_prompt_additions(turn.executions)
    answered = RoomMessage(
        seq=asked.seq + 1,
        sender_id=clone_id,
        session_id=seat.session_id or None,  # the trace reads it (clone-data-scopes §4)
        content=turn.content,
        provenance=turn.provenance,
        usage=turn.usage,
        error=turn.error,
        refusal=turn.refusal,
        provider_failure=turn.provider_failure,
        completed=turn.completed,
        rendered_through=asked.seq,
        turn_id=turn_id,
        tools_recorded=turn.tools_recorded,
        memory_facts_tried=tried,
        memory_facts_unsaved=unsaved,
        image_prompt_added=prompt_added,
        image_negative_added=negative_added,
        session_set_aside=turn.session_set_aside,
    )
    uses, written = record_turn_tools(seat, turn_id, turn.executions)
    seen = dict(state.last_seen_seq)
    if turn.error is None:
        seen[clone_id] = str(asked.seq)
    record = state.file_record
    return store.save(
        state.model_copy(
            update={
                "transcript": (*state.transcript, asked, answered),
                "turn_state": state.turn_state.model_copy(
                    update={
                        "agent_turns_since_human": 1,
                        "last_speaker_id": clone_id,
                        "last_activity_ts": time.time(),
                    }
                ),
                "last_seen_seq": seen,
                "last_decision": None,
                "tool_uses": (*state.tool_uses, *uses),
                "written_files": (*state.written_files, *written),
                "file_record": record.model_copy(
                    update={
                        "turns_started": record.turns_started + 1,
                        "turns_landed": record.turns_landed + 1,
                        "unrecorded_turns": record.unrecorded_turns
                        + (0 if turn.tools_recorded else 1),
                        "unattributed_writes": record.unattributed_writes
                        + sum(1 for u in uses if u.wrote_unnamed),
                    }
                ),
            }
        )
    )


def open_head_room(
    clone_id: str, room_id: str | None = None, *, head: str, store: RoomStore | None = None
) -> OneSeatRoom:
    """The one-seat room a CLI head (`run`, `loop`) converses in: `room_id`'s, or a new one.

    Nothing is written: a new room is stored with its first turn (`record_head_turn`).

    `clone_id` and `room_id` are checked by the rules they met before they named a room --
    the agent home's and the session name's -- so `ucx run n/x` and `--session-id ../x`
    are refused as before, naming what was typed, before any room is written.

    Raises:
        AgentHomeError: `clone_id` cannot name an agent home.
        PathTraversalError: `room_id` is not a legal session name.
        RoomError: see `resolve_one_seat_room`.
    """
    from uclone_x.core.agent_home import refuse_an_unusable_username
    from uclone_x.core.session import validate_session_id
    from uclone_x.room.store import RoomStore

    # The clone's name first: it is the seat's id as well as its home directory, and a
    # name no directory can carry is the agent name's fault, not the room's.
    refuse_an_unusable_username(clone_id)
    if room_id is not None:
        validate_session_id(room_id)
    return resolve_one_seat_room(
        RoomService(store if store is not None else RoomStore()),
        room_id=room_id if room_id is not None else new_one_seat_room_id(),
        clone_id=clone_id,
        head=head,
        # `run` and `loop` take the id as `--session-id`, so a refusal names it so (#1900).
        id_label="Session id",
    )
