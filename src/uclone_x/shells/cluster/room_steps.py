"""One room step on a stateless worker: a `ucx.cmd.room` command in, one commit, what it sends.

A room's cascade is a chain of commands, one speaker per link (`stateless-worker-and-rooms.md`
§4.1). `ROOM_POST` and `ROOM_TURN_LANDED` each make one commit, which writes the room's change,
the count of the next speaker's start (#1366) and the floor, with a receipt holding what the step
sends.

**Redelivery changes nothing and sends again.** A command with a receipt goes straight to sending
what the receipt holds, so a commit that succeeded and a publish that did not cannot stall the
room: the record's offset stays until the publish is acknowledged. Every copy of what it sends is
known by its own id in turn (a seat turn's receipt, a landing's receipt), so the chain settles.

**How a cascade ends.** The floor goes back to the person when the chain says nobody speaks, when
the turn ceiling is reached, when the selector meets a usage limit, when a seat turn was given up
on or interrupted, or when the step itself is given up on (`run(..., can_select=False)`). A person's
message that arrives while a seat holds the floor waits for that seat's landing, because one floor
has one speaker. Its cascade starts at the landing, unless a stop came after it.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from uclone_x.core.cluster_protocols import (
    LEARN_JOB_ID_PREFIX,
    SEAT_TURN_ID_PREFIX,
    ClusterRoomStoreProtocol,
    ClusterSessionStoreProtocol,
    CommandBrokerProtocol,
    InterruptControlEnvelope,
    LearnCommandEnvelope,
    RoomCommandEnvelope,
    RoomStepOutbox,
    SeatTurnRef,
    SSEStreamPublisherProtocol,
    TurnCommandEnvelope,
    TurnLeaseProtocol,
)
from uclone_x.engine.event_bus import AgentEvent, EventType
from uclone_x.errors import SpeakerSelectionError, UCloneXError, UsageLimitReachedError
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomFloor,
    RoomMessage,
    RoomState,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerStepOutcome,
)
from uclone_x.room.orchestrator import (
    RoomOrchestrator,
    append_human_row,
    ceiling_decision,
    land_speaker_step,
    landed_payload,
    row_committed,
    validate_speaker,
)

__all__ = ["DecideSpeakerFn", "RoomStepExecutor"]

logger = logging.getLogger(__name__)

#: How long a seat may hold the floor before a person's new message takes it back. The seat, if it
#: is alive, is stopped first.
DEFAULT_FLOOR_RECLAIM_SECONDS: Final[float] = 3600.0

#: What a person reads. Plain words, and no exception, path or id: those are in the log.
_COPY_POST_REFUSED: Final[str] = "This message could not be posted to this conversation."
_COPY_ROOM_GAVE_UP: Final[str] = (
    "The room could not work out who should answer, so it has stopped. Send a message to try again."
)
_COPY_CEILING: Final[str] = "The room has paused after its turn limit. Send a message to continue."
_COPY_SEAT_LOST: Final[str] = "This reply never arrived."
_COPY_NOT_FINISHED: Final[str] = "The turn could not be finished. Please try again."
_INTERRUPTED: Final[str] = "Turn was interrupted"


def _copy_seat_gave_up(name: str) -> str:
    return f"{name} could not answer, so the room has stopped. Send a message to try again."


DecideSpeakerFn = Callable[[RoomState, str, str], Awaitable[SpeakerDecision]]


@dataclass
class _Step:
    """One command's transition, not yet committed."""

    state: RoomState | None
    outbox: RoomStepOutbox = field(default_factory=RoomStepOutbox)


@dataclass(frozen=True)
class _Cascade:
    """What the next link of a cascade needs to know, wherever it was reached from."""

    cascade_id: str
    user_id: str
    baseline_seq: int
    learn_from_seq: int
    traceparent: str | None


def _seat_turn_id(room_id: str, command_id: str) -> str:
    """The same on every retry of `command_id`, so a seat turn is known by its receipt."""
    return (
        SEAT_TURN_ID_PREFIX + hashlib.sha256(f"{room_id}\n{command_id}".encode()).hexdigest()[:32]
    )


def _row_event(room_id: str, message: RoomMessage, turn_id: str) -> dict[str, Any]:
    return {
        "type": EventType.AGENT_REPLY.value,
        "sender_id": message.sender_id,
        "payload": landed_payload(room_id, message, turn_id),
    }


def _room_event(room_id: str, event: str, cascade_id: str, **fields: Any) -> dict[str, Any]:
    return {
        "type": EventType.AGENT_REPLY.value,
        "sender_id": "room",
        "payload": {"room_id": room_id, "event": event, "cascade_id": cascade_id, **fields},
    }


def _outcome_of(
    terminal: Mapping[str, Any] | None, floor: RoomFloor, state: RoomState
) -> SpeakerStepOutcome:
    """The row's account of a seat turn: its receipt's, or what its end says when it never ran."""
    landing = terminal.get("landing") if terminal is not None else None
    if isinstance(landing, str):
        return SpeakerStepOutcome.model_validate_json(landing)
    interrupted = terminal is not None and terminal.get("event") == "TURN_INTERRUPTED"
    told = terminal.get("error") if terminal is not None else None
    return SpeakerStepOutcome(
        turn_id=floor.turn_id,
        rendered_through=floor.baseline_seq,
        error=_INTERRUPTED if interrupted else str(told or _COPY_NOT_FINISHED),
        completed=not interrupted,
        story_id=state.story_id,
    )


def _last_human_seq(state: RoomState) -> int:
    return next(m.seq for m in reversed(state.transcript) if m.is_utterance and m.decision is None)


def _as_asked(state: RoomState) -> RoomState:
    """`state` for choosing a speaker: a reply that never saw the latest message comes before it.

    A message sent while a seat holds the floor is written ahead of that seat's row. Read in that
    order the row looks like its answer, and spends its turn budget. This copy is only read: the
    room keeps its rows as they were written.
    """
    rows = state.transcript
    at = next(
        (
            i
            for i in reversed(range(len(rows)))
            if rows[i].is_utterance and rows[i].decision is None
        ),
        None,
    )
    if at is None:
        return state
    seq = rows[at].seq
    later = rows[at + 1 :]
    overtaken = tuple(m for m in later if m.is_utterance and m.rendered_through < seq)
    if not overtaken:
        return state
    answers = tuple(m for m in later if m not in overtaken)
    budget = state.turn_state.agent_turns_since_human - len(overtaken)
    return state.model_copy(
        update={
            "transcript": (*rows[:at], *overtaken, rows[at], *answers),
            "turn_state": state.turn_state.model_copy(update={"agent_turns_since_human": budget}),
        }
    )


class RoomStepExecutor:
    """Runs `ucx.cmd.room` commands; holds nothing about any room between them."""

    def __init__(
        self,
        *,
        rooms: ClusterRoomStoreProtocol,
        sessions: ClusterSessionStoreProtocol,
        broker: CommandBrokerProtocol,
        sse: SSEStreamPublisherProtocol,
        leases: TurnLeaseProtocol,
        decide: DecideSpeakerFn,
        clone_learns: Callable[[str], bool],
        floor_reclaim_seconds: float = DEFAULT_FLOOR_RECLAIM_SECONDS,
    ) -> None:
        self._rooms = rooms
        self._sessions = sessions
        self._broker = broker
        self._sse = sse
        self._leases = leases
        self._decide = decide
        self._clone_learns = clone_learns
        self._floor_reclaim_seconds = floor_reclaim_seconds

    async def seat_on_floor(self, cmd: TurnCommandEnvelope, seat: SeatTurnRef) -> bool:
        """Whether this seat turn is still the one the room is waiting for.

        Past this the seat runs, so this is the durable half of an interrupt: a tombstone expires,
        and a command can wait longer than that in a queue.
        """
        state = await self._rooms.read_room(seat.room_id)
        floor = state.floor if state is not None else None
        return (
            state is not None
            and floor is not None
            and floor.turn_id == cmd.turn_id
            and not floor.interrupted
            and RoomOrchestrator._participant(state, floor.speaker_id) is not None  # pyright: ignore[reportPrivateUsage]
        )

    async def send_if_committed(self, cmd: RoomCommandEnvelope) -> bool:
        """A command with a receipt: send what it holds again. Returns whether there was one."""
        stored = await self._rooms.step_receipt(cmd.room_id, cmd.turn_id)
        if stored is None:
            return False
        await self._send(cmd.room_id, stored)
        return True

    async def run(self, cmd: RoomCommandEnvelope, *, can_select: bool = True) -> None:
        """One command: commit its one transition, then send what it says.

        `can_select=False` is the worker giving up on the command: everything but the choice of
        the next speaker still happens, and the floor goes back to the person.
        """
        if await self.send_if_committed(cmd):
            return
        step = await self._step(cmd, can_select)
        await self._rooms.commit_step(
            step.state, room_id=cmd.room_id, command_id=cmd.turn_id, outbox=step.outbox
        )
        await self._send(cmd.room_id, step.outbox)

    async def tell_unread(self, room_id: str, message: str) -> None:
        """A record that named this room could not be read at all."""
        await self._publish_event(
            room_id, _room_event(room_id, "ROOM_POST_REFUSED", "", message=message)
        )

    async def _read_given_up(self, room_id: str) -> RoomState | None:
        """The room, or `None` if its record cannot be read: giving up must still end."""
        try:
            return await self._rooms.read_room(room_id)
        except ValueError:  # pydantic's `ValidationError`: an unreadable record, not an outage
            logger.exception("Room %s cannot be read; giving up without it", room_id)
            return None

    async def _step(self, cmd: RoomCommandEnvelope, can_select: bool) -> _Step:
        state = (
            await self._rooms.read_room(cmd.room_id)
            if can_select
            else await self._read_given_up(cmd.room_id)
        )
        if cmd.kind == "ROOM_POST":
            return await self._post(cmd, state, can_select)
        if cmd.kind == "ROOM_TURN_LANDED":
            return await self._landed(cmd, state, can_select)
        return await self._interrupt(cmd, state)

    # -- a person's message ------------------------------------------------------------

    async def _post(
        self, cmd: RoomCommandEnvelope, state: RoomState | None, can_select: bool
    ) -> _Step:
        sender = (
            RoomOrchestrator._participant(state, cmd.sender_id or "")  # pyright: ignore[reportPrivateUsage]
            if state is not None
            else None
        )
        if (
            state is None
            or sender is None
            or sender.kind is not ParticipantKind.HUMAN
            or state.head is not None
            or cmd.human_utterance is None
        ):
            # Checked on the record: only the room's human posts, and a head's room has one
            # writer (`RoomOrchestrator._require_human_sender`). No retry changes either.
            logger.warning("Refusing post %s to room %s", cmd.turn_id, cmd.room_id)
            refused = _room_event(
                cmd.room_id,
                "ROOM_POST_REFUSED",
                cmd.cascade_id,
                turn_id=cmd.turn_id,
                message=_COPY_POST_REFUSED,
            )
            return _Step(None, RoomStepOutbox(events=[refused]))

        written = append_human_row(state, sender.id, cmd.human_utterance)
        row = written.transcript[-1]
        events = [
            _room_event(
                cmd.room_id,
                "ROOM_MESSAGE",
                cmd.cascade_id,
                turn_id=cmd.turn_id,
                seq=row.seq,
                agent_id=row.sender_id,
                content=row.content,
            )
        ]
        floor = written.floor
        learn_from_seq = row.seq - 1
        if floor is not None:
            age_seconds = time.time() - floor.given_at_ms / 1000
            if age_seconds < self._floor_reclaim_seconds:
                # One floor, one speaker: this message's cascade starts at the seat's landing,
                # under this person.
                waiting = floor.model_copy(
                    update={"cascade_id": cmd.cascade_id, "user_id": cmd.user_id}
                )
                return _Step(
                    written.model_copy(update={"floor": waiting}), RoomStepOutbox(events=events)
                )
            written, lost = await self._reclaim(written, floor)
            events.append(_row_event(cmd.room_id, lost, floor.turn_id))
            learn_from_seq = floor.learn_from_seq
        return await self._advance(
            written,
            cmd,
            _Cascade(cmd.cascade_id, cmd.user_id, row.seq, learn_from_seq, cmd.traceparent),
            events,
            can_select,
        )

    async def _reclaim(self, state: RoomState, floor: RoomFloor) -> tuple[RoomState, RoomMessage]:
        """Take the floor from a seat turn that has not landed in all this time."""
        logger.error("Room %s takes the floor back from turn %s", state.room_id, floor.turn_id)
        await self._stop_seat(floor)
        landing, message = land_speaker_step(
            state,
            speaker_id=floor.speaker_id,
            session_id=floor.session_id,
            decision=floor.decision,
            outcome=SpeakerStepOutcome(
                turn_id=floor.turn_id,
                rendered_through=floor.baseline_seq,
                error=_COPY_SEAT_LOST,
                completed=False,
                story_id=state.story_id,
            ),
        )
        return landing.model_copy(update={"floor": None}), message

    async def _stop_seat(self, floor: RoomFloor) -> None:
        """A queued seat turn finds a tombstone; a running one is cancelled where it runs."""
        await self._leases.write_cancel_tombstone(floor.session_id, floor.turn_id)
        await self._broker.broadcast_interrupt(
            InterruptControlEnvelope(
                conversation_id=floor.session_id,
                turn_id=floor.turn_id,
                requested_by=floor.user_id,
                issued_at_ms=int(time.time() * 1000),
            )
        )

    # -- a seat's turn ended -----------------------------------------------------------

    async def _landed(
        self, cmd: RoomCommandEnvelope, state: RoomState | None, can_select: bool
    ) -> _Step:
        floor = state.floor if state is not None else None
        if state is None or floor is None or floor.turn_id != cmd.seat_turn_id:
            # Not the turn the room is waiting for: the floor was taken back, or the room is gone.
            logger.warning("Room %s: landing of %s is not on its floor", cmd.room_id, cmd.turn_id)
            return _Step(None)
        terminal = await self._sessions.settled_terminal(floor.session_id, floor.turn_id)
        if terminal is None and can_select:
            raise UCloneXError(f"The seat turn {floor.turn_id} landed with no receipt")
        landing, message = self._land(state, floor, terminal, can_select)
        cleared = landing.model_copy(update={"floor": None})
        events = [_row_event(cmd.room_id, message, floor.turn_id)]
        cascade = _Cascade(
            floor.cascade_id,
            floor.user_id,
            floor.baseline_seq,
            floor.learn_from_seq,
            cmd.traceparent,
        )

        said_by_then = max(floor.baseline_seq, floor.stopped_through)
        if RoomOrchestrator._interjected(cleared, said_by_then):  # pyright: ignore[reportPrivateUsage]
            # The person spoke while this seat held the floor, and did not stop the room after.
            # Their message starts its own cascade now, whatever became of this turn.
            again = _Cascade(
                cascade.cascade_id,
                cascade.user_id,
                _last_human_seq(cleared),
                cascade.learn_from_seq,
                cascade.traceparent,
            )
            return await self._advance(cleared, cmd, again, events, can_select)
        if terminal is not None and terminal.get("dead_lettered"):
            name = self._name_of(state, floor.speaker_id)
            return self._end(
                cleared, cmd, cascade, events, "seat_gave_up", _copy_seat_gave_up(name)
            )
        if floor.interrupted or message.completed is False:
            return self._end(cleared, cmd, cascade, events, "interrupted", None)
        return await self._advance(cleared, cmd, cascade, events, can_select)

    @staticmethod
    def _land(
        state: RoomState,
        floor: RoomFloor,
        terminal: Mapping[str, Any] | None,
        giving_up: bool,
    ) -> tuple[RoomState, RoomMessage]:
        def land(outcome: SpeakerStepOutcome) -> tuple[RoomState, RoomMessage]:
            return land_speaker_step(
                state,
                speaker_id=floor.speaker_id,
                session_id=floor.session_id,
                decision=floor.decision,
                outcome=outcome,
            )

        try:
            return land(_outcome_of(terminal, floor, state))
        except ValueError:
            if not giving_up:
                raise
        # The turn's account itself will not read: its row says only that it did not finish.
        logger.exception("Room %s could not read the account of %s", state.room_id, floor.turn_id)
        return land(_outcome_of(None, floor, state))

    @staticmethod
    def _name_of(state: RoomState, participant_id: str) -> str:
        found: Participant | None = RoomOrchestrator._participant(state, participant_id)  # pyright: ignore[reportPrivateUsage]
        return (found.display_name or found.id) if found is not None else participant_id

    # -- a person stopped the room -----------------------------------------------------

    async def _interrupt(self, cmd: RoomCommandEnvelope, state: RoomState | None) -> _Step:
        floor = state.floor if state is not None else None
        if state is None or floor is None:
            return _Step(None)
        # Before the commit, and safe to repeat: a redelivery that got this far did not commit.
        await self._stop_seat(floor)
        stopped = floor.model_copy(
            update={"interrupted": True, "stopped_through": state.transcript[-1].seq}
        )
        return _Step(state.model_copy(update={"floor": stopped}))

    # -- who speaks next ---------------------------------------------------------------

    async def _advance(
        self,
        state: RoomState,
        cmd: RoomCommandEnvelope,
        cascade: _Cascade,
        events: list[dict[str, Any]],
        can_select: bool,
    ) -> _Step:
        """The next link of the cascade: give the floor to one speaker, or end it."""
        if not can_select:
            return self._end(state, cmd, cascade, events, "room_gave_up", _COPY_ROOM_GAVE_UP)
        asked = _as_asked(state)
        stop = ceiling_decision(asked)  # the circuit breaker, before anyone is consulted
        if stop is not None:
            return self._end(
                state.model_copy(update={"last_decision": stop}),
                cmd,
                cascade,
                events,
                "turn_ceiling",
                _COPY_CEILING,
            )
        try:
            decision = await self._decide(asked, cascade.user_id, cmd.turn_id)
        except SpeakerSelectionError as exc:
            if not isinstance(exc.__cause__, UsageLimitReachedError):
                raise
            # Not retried (P6). Its own sentence says what stopped and what to do.
            limit = str(exc.__cause__)
            return self._end(state, cmd, cascade, events, "usage_limit", limit)
        if decision.verdict is SelectionVerdict.SILENCE:
            return self._end(
                state.model_copy(update={"last_decision": decision}),
                cmd,
                cascade,
                events,
                "silence",
                None,
            )
        speaker = validate_speaker(state, state, decision)
        if speaker is None:  # the chain saw this very state, so its roster did not move
            raise UCloneXError(f"Room {state.room_id}: the speaker it chose is not seated")
        turn_id = _seat_turn_id(state.room_id, cmd.turn_id)
        seat = TurnCommandEnvelope(
            turn_id=turn_id,
            session_id=speaker.session_id,
            agent_id=speaker.id,
            user_id=cascade.user_id,
            prompt="",
            seat=SeatTurnRef(room_id=state.room_id, cascade_id=cascade.cascade_id),
            traceparent=cascade.traceparent,
            enqueued_at_ms=int(time.time() * 1000),
        )
        record = state.file_record
        started = state.model_copy(
            update={
                # Counted before anyone is told to run it, in the one commit that gives the floor,
                # so a crash anywhere in the turn leaves `turns_started` ahead of `turns_landed`
                # (#1366) and a redelivery cannot count it twice.
                "file_record": record.model_copy(
                    update={"turns_started": record.turns_started + 1}
                ),
                "floor": RoomFloor(
                    turn_id=turn_id,
                    session_id=speaker.session_id,
                    speaker_id=speaker.id,
                    cascade_id=cascade.cascade_id,
                    user_id=cascade.user_id,
                    decision=decision,
                    baseline_seq=cascade.baseline_seq,
                    learn_from_seq=cascade.learn_from_seq,
                    given_at_ms=seat.enqueued_at_ms,
                ),
            }
        )
        return _Step(started, RoomStepOutbox(events=events, seat_turns=[seat]))

    def _end(
        self,
        state: RoomState,
        cmd: RoomCommandEnvelope,
        cascade: _Cascade,
        events: list[dict[str, Any]],
        reason: str,
        told: str | None,
    ) -> _Step:
        """The floor goes back to the person, and what the cascade taught goes to be learned."""
        ended = _room_event(
            state.room_id,
            "ROOM_CASCADE_ENDED",
            cascade.cascade_id,
            reason=reason,
            **({"message": told} if told is not None else {}),
        )
        try:
            job = self._learn_job(state, cmd, cascade)
        except Exception:  # a lesson lost costs a lesson; ending the cascade must not depend on it
            logger.warning("Room %s: could not make a learn job", state.room_id, exc_info=True)
            job = None
        return _Step(
            state,
            RoomStepOutbox(events=[*events, ended], learn_jobs=[job] if job is not None else []),
        )

    def _learn_job(
        self, state: RoomState, cmd: RoomCommandEnvelope, cascade: _Cascade
    ) -> LearnCommandEnvelope | None:
        """`ROOM_LEARN_JOB`, when a clone that learns landed a turn in this cascade (#1840)."""
        up_to_seq = state.transcript[-1].seq if state.transcript else 0
        speakers = {
            row.sender_id
            for row in state.transcript
            if row.seq > cascade.learn_from_seq and row_committed(state, row)
        }
        if not any(self._clone_learns(clone_id) for clone_id in speakers):
            return None
        return LearnCommandEnvelope(
            job_id=f"{LEARN_JOB_ID_PREFIX}{cmd.turn_id}",
            room_id=state.room_id,
            user_id=cascade.user_id,
            from_seq=cascade.learn_from_seq,
            up_to_seq=up_to_seq,
            traceparent=cascade.traceparent,
        )

    # -- what a step sends -------------------------------------------------------------

    async def _publish_event(self, room_id: str, event: Mapping[str, Any]) -> None:
        await self._sse.publish_event(
            room_id,
            AgentEvent(
                type=EventType(str(event["type"])),
                topic=f"room.{room_id}",
                sender_id=str(event["sender_id"]),
                payload=event["payload"],
            ),
        )

    async def _send(self, room_id: str, outbox: RoomStepOutbox) -> None:
        for event in outbox.events:
            await self._publish_event(room_id, event)
        for seat in outbox.seat_turns:
            await self._broker.publish_turn(seat)
        for job in outbox.learn_jobs:
            try:
                await self._broker.publish_learn_job(job)
            except Exception:  # a lesson lost costs a lesson; the room has gone on without it
                logger.warning("Could not publish learn job %s", job.job_id, exc_info=True)
