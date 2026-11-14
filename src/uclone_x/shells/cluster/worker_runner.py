"""Stateless agent and room worker execution loop (`Rehydrate -> Execute -> Commit -> Evict`)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, TypeVar

from uclone_x.adapters.cluster.kafka_broker import TOPIC_BG_LEARN, TOPIC_CMD_ROOM
from uclone_x.core.cluster_protocols import (
    LANDED_ID_PREFIX,
    ClusterSessionStoreProtocol,
    CommandBrokerProtocol,
    DeadLetterEnvelope,
    RoomCommandEnvelope,
    SeatTurnRef,
    SSEStreamPublisherProtocol,
    TurnCommandEnvelope,
    TurnLeaseConflictError,
    TurnLeaseProtocol,
    TurnLeaseToken,
    TurnPreemptedError,
    wire_id_problem,
)
from uclone_x.core.session_state import SessionState
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventType
from uclone_x.engine.protocols import EventBusProtocol
from uclone_x.errors import (
    PlainRefusalError,
    StaleRoomWriteError,
    StaleSessionWriteError,
    UCloneXError,
)

if TYPE_CHECKING:
    from uclone_x.adapters.cluster.kafka_broker import ConsumedRecord
    from uclone_x.agent.models import TurnResult
    from uclone_x.shells.cluster.room_steps import RoomStepExecutor
    from uclone_x.shells.cluster.turn_executor import TurnExecution

__all__ = [
    "StatelessAgentWorker",
    "TurnDisposition",
    "TurnExecutorFn",
    "TurnOutcome",
]

logger = logging.getLogger(__name__)

#: Commits refused by the revision check before the turn is reported failed. Each attempt runs
#: the model again, so this stays small; a redelivered copy needs only the second one.
_MAX_STALE_ATTEMPTS: Final[int] = 3
#: Times one delivery may reach its executor and fail. Each can bill a model, so this is small.
DEFAULT_MAX_TURN_ATTEMPTS: Final[int] = 5
#: How long after its record was produced a failure ends a turn instead of a retry. A failure
#: before the executor bills nothing and uses no attempt, so this alone bounds an outage.
DEFAULT_TURN_DEADLINE_SECONDS: Final[float] = 900.0
#: The longest a backoff doubles to.
_RETRY_CAP_SECONDS: Final[float] = 30.0
#: Starts of a delivery before it is dead-lettered without running. An attempt that comes to any
#: end gives its start back, so what stays counted is a process dying under it (killed, say).
DEFAULT_MAX_TURN_STARTS: Final[int] = 10
#: How much of a record a dead letter quotes. The rest stays in the source topic.
_DEAD_LETTER_HEAD_BYTES: Final[int] = 4096
_DEAD_LETTER_KEY_CHARS: Final[int] = 256

#: What a person reads. Plain words, and no exception, path or id: those are in the log.
_COPY_NOT_FINISHED: Final[str] = "The turn could not be finished. Please try again."
_COPY_NOT_READ: Final[str] = "This message could not be read, and trying again may not help."
_COPY_NOT_ANSWERED: Final[str] = "This message could not be answered. Please try again later."
_COPY_KEPT_CHANGING: Final[str] = (
    "This conversation kept changing during the turn. Please try again."
)
_COPY_CONVERSATION_CHANGED: Final[str] = "This conversation has changed. Please try again."
_LEASE_RETRY_FIRST_SECONDS: Final[float] = 0.25
_LEASE_RETRY_MAX_SECONDS: Final[float] = 2.0

TurnExecutorFn = Callable[[TurnCommandEnvelope, EventBusProtocol], Awaitable["TurnExecution"]]
WaitFn = Callable[[float], Awaitable[None]]
_T = TypeVar("_T")


class TurnDisposition(StrEnum):
    """How a turn ended; but for `DUPLICATE`, the value is the `event` of its terminal SSE event."""

    COMPLETED = "TURN_COMPLETED"
    #: Its usage ceiling stopped it at a step boundary; it continues in place (#2431).
    STOPPED = "TURN_STOPPED"
    FAILED = "TURN_FAILED"
    INTERRUPTED = "TURN_INTERRUPTED"
    #: A redelivered turn that already has a receipt: nothing ran, and its receipt's terminal
    #: event went out again.
    DUPLICATE = "TURN_ALREADY_SETTLED"


@dataclass(frozen=True)
class TurnOutcome:
    """What became of a turn command. Returning one means its offset may advance."""

    disposition: TurnDisposition
    state: SessionState | None = None
    result: TurnResult | None = None


class _TurnLeaseLostError(UCloneXError):
    """The lease lapsed or moved under a running turn."""


class _TurnCancelledInsideError(UCloneXError):
    """A running turn's task was cancelled, and nobody here asked for it."""


def _terminal(disposition: TurnDisposition, **fields: Any) -> dict[str, Any]:
    """The event a client ends the turn on, without its `turn_id`; an empty field is left out."""
    return {
        "event": disposition.value,
        **{name: value for name, value in fields.items() if value is not None},
    }


def _landed_command(cmd: TurnCommandEnvelope, seat: SeatTurnRef) -> RoomCommandEnvelope:
    """`ROOM_TURN_LANDED` for a seat turn that ended: the room reads how from its receipt."""
    return RoomCommandEnvelope(
        kind="ROOM_TURN_LANDED",
        turn_id=f"{LANDED_ID_PREFIX}{cmd.turn_id}",
        room_id=seat.room_id,
        user_id=cmd.user_id,
        cascade_id=seat.cascade_id,
        seat_turn_id=cmd.turn_id,
        traceparent=cmd.traceparent,
        enqueued_at_ms=int(time.time() * 1000),
    )


def _plain_failure(result: TurnResult) -> str:
    """What the person is told of a failed turn. Raw `error` text may name internals (#2234).

    A provider's failure has a sentence written to be shown, as does the usage limit. Anything
    else is said generically; the raw text is in the log, and `error_code` lets a head translate.
    """
    if result.provider_failure is not None:
        return result.provider_failure.message
    if result.stop_reason == "usage_limit" and result.error is not None:
        return result.error
    return _COPY_NOT_FINISHED


@dataclass
class _Delivery:
    """One delivery of a turn on this worker: what bounds its retries."""

    produced_at_ms: int
    #: Attempts that reached the executor. In memory: a process that dies is the starts' to count.
    attempts: int = 0
    failures: int = 0
    letter_published: bool = False


@dataclass
class _RunningTurn:
    """The turn this worker is running on one conversation, and why it was stopped."""

    turn_id: str
    task: asyncio.Task[Any]
    #: The room this turn is a seat's turn in, when it is one.
    room_id: str | None = None
    interrupted: bool = False
    lease_lost: bool = False

    def stop(self, *, interrupted: bool) -> None:
        if interrupted:
            self.interrupted = True
        else:
            self.lease_lost = True
        self.task.cancel()


class StatelessAgentWorker:
    """Executes single-agent turns and single-hop room steps without retaining resident state."""

    def __init__(
        self,
        *,
        worker_id: str,
        session_store: ClusterSessionStoreProtocol,
        lease_manager: TurnLeaseProtocol,
        sse_publisher: SSEStreamPublisherProtocol,
        turn_executor: TurnExecutorFn,
        room_steps: RoomStepExecutor | None = None,
        broker: CommandBrokerProtocol | None = None,
        max_turn_attempts: int = DEFAULT_MAX_TURN_ATTEMPTS,
        turn_deadline_seconds: float = DEFAULT_TURN_DEADLINE_SECONDS,
        max_turn_starts: int = DEFAULT_MAX_TURN_STARTS,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be blank")
        self._worker_id = worker_id
        self._session_store = session_store
        self._lease_manager = lease_manager
        self._sse_publisher = sse_publisher
        self._turn_executor = turn_executor
        self._room_steps = room_steps
        self._broker = broker
        self._max_turn_attempts = max_turn_attempts
        self._turn_deadline_seconds = turn_deadline_seconds
        self._max_turn_starts = max_turn_starts
        self._retry_delay_seconds = retry_delay_seconds
        self._running: dict[str, _RunningTurn] = {}

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def turn_executor(self) -> TurnExecutorFn:
        return self._turn_executor

    @property
    def room_steps(self) -> RoomStepExecutor | None:
        return self._room_steps

    def interrupt(self, conversation_id: str, turn_id: str | None = None) -> bool:
        """Stop the turn running here on `conversation_id` (Invariant 4, case A).

        `turn_id` of `None` stops whichever turn is running. A turn still queued is not
        touched: its tombstone refuses it when it reaches the lease. `conversation_id` may be a
        room, which stops the seat turn running here in it. Returns whether a running turn was
        stopped.
        """
        seated = [r for r in self._running.values() if r.room_id == conversation_id]
        for running in seated:
            running.stop(interrupted=True)
        if seated:
            return True
        running = self._running.get(conversation_id)
        if running is None or (turn_id is not None and running.turn_id != turn_id):
            return False
        running.stop(interrupted=True)
        return True

    async def execute_single_turn(
        self,
        cmd: TurnCommandEnvelope,
        source: ConsumedRecord | None = None,
        *,
        wait: WaitFn = asyncio.sleep,
    ) -> TurnOutcome:
        """Run one turn to a disposition; whatever returns may have its offset committed.

        **Every terminal event has a receipt written first**, and the receipt holds the event, so
        a redelivery cannot rerun a turn the person was told ended, and tells them again if the
        publish failed. That holds for each way a turn ends: a commit writes the receipt in its
        own transaction, and every other end writes one in `_settle`.

        **A failure is retried here,** with a backoff that holds up this session's queue, until it
        has used its attempts or passed its deadline. Then the turn is given up on (`_give_up`).
        Raises only when cancelled: the process is leaving.
        """
        delivery = self._delivery_of(source)

        async def after_stale(exc: Exception) -> None:
            if isinstance(exc, StaleSessionWriteError):
                # Someone else moved the record. If they ended this turn, the next attempt
                # finds its receipt; if not, it runs on their state. The cached head may be
                # what was stale (async replication), so it goes first.
                await self._session_store.invalidate_head(cmd.session_id)
            else:
                logger.warning("Lost the lease of turn %s; trying it again", cmd.turn_id)

        async def kept_changing() -> TurnOutcome:
            await self._settle(cmd, TurnDisposition.FAILED, error=_COPY_KEPT_CHANGING)
            return TurnOutcome(TurnDisposition.FAILED)

        async def give_up(reason: str, exc: Exception) -> TurnOutcome:
            return await self._give_up(cmd, source, delivery, reason, exc, wait)

        return await self._within_bounds(
            f"Turn {cmd.turn_id}",
            delivery,
            lambda: self._attempt_turn(cmd, source, delivery),
            conflicts=(StaleSessionWriteError, _TurnLeaseLostError),
            after_conflict=after_stale,
            kept_changing=kept_changing,
            give_up=give_up,
            wait=wait,
        )

    @staticmethod
    def _delivery_of(source: ConsumedRecord | None) -> _Delivery:
        return _Delivery(
            produced_at_ms=(
                source.timestamp_ms
                if source is not None and source.timestamp_ms is not None
                else int(time.time() * 1000)
            )
        )

    async def _within_bounds(
        self,
        what: str,
        delivery: _Delivery,
        attempt: Callable[[], Awaitable[_T]],
        *,
        conflicts: tuple[type[Exception], ...],
        after_conflict: Callable[[Exception], Awaitable[None]],
        kept_changing: Callable[[], Awaitable[_T]],
        give_up: Callable[[str, Exception], Awaitable[_T]],
        wait: WaitFn,
    ) -> _T:
        """The one policy for a command that fails (§2.3): a turn's and a room step's.

        Retried here, with a backoff that holds up its conversation's queue, until it has used
        its attempts or passed its deadline; then `give_up`. Someone else moving the record is
        not a failure and has a small bound of its own. Raises only when cancelled.
        """
        stale_attempts = 0
        while True:
            try:
                try:
                    return await attempt()
                except conflicts as exc:
                    await after_conflict(exc)
                    stale_attempts += 1
                if stale_attempts >= _MAX_STALE_ATTEMPTS:
                    return await kept_changing()
            except Exception as exc:
                logger.exception("%s failed on worker %s", what, self._worker_id)
                reason = self._out_of_bounds(delivery)
                if reason is not None:
                    return await give_up(reason, exc)
                await wait(self._backoff(delivery.failures))
                delivery.failures += 1

    def _backoff(self, failures: int) -> float:
        return min(self._retry_delay_seconds * 2 ** min(failures, 16), _RETRY_CAP_SECONDS)

    def _out_of_bounds(self, delivery: _Delivery) -> str | None:
        """Why this delivery gets no more retries, if it does not. Only a failure asks this."""
        if delivery.attempts >= self._max_turn_attempts:
            return "attempts_used"
        age_seconds = time.time() - delivery.produced_at_ms / 1000
        return "deadline_passed" if age_seconds >= self._turn_deadline_seconds else None

    async def _until_done(
        self, what: str, step: Callable[[], Awaitable[None]], wait: WaitFn
    ) -> None:
        """Retry `step` with no bound: the offset stays until the person can be told."""
        failures = 0
        while True:
            try:
                return await step()
            except Exception:
                logger.exception("Could not %s; trying again", what)
                await wait(self._backoff(failures))
                failures += 1

    async def _give_up(
        self,
        cmd: TurnCommandEnvelope,
        source: ConsumedRecord | None,
        delivery: _Delivery,
        reason: str,
        exc: Exception,
        wait: WaitFn,
    ) -> TurnOutcome:
        """End a turn out of bounds, however long the stores take to let it be written."""

        async def step() -> None:
            settled = await self._session_store.settled_terminal(cmd.session_id, cmd.turn_id)
            if settled is not None:  # the failure was telling them: nothing to give up on
                await self._publish_terminal(cmd, settled)
                return
            await self._dead_letter(cmd, source, delivery, reason, error_kind=type(exc).__name__)

        await self._until_done(f"give up on turn {cmd.turn_id}", step, wait)
        return TurnOutcome(TurnDisposition.FAILED)

    async def _dead_letter(
        self,
        cmd: TurnCommandEnvelope,
        source: ConsumedRecord | None,
        delivery: _Delivery,
        reason: str,
        *,
        error_kind: str | None = None,
        starts: int = 0,
    ) -> None:
        """Publish a description of the record, write its receipt, then tell the person."""
        if not delivery.letter_published:
            await self._publish_dead_letter(
                key=cmd.session_id,
                value=source.raw_value if source is not None else cmd.model_dump_json().encode(),
                source=source,
                reason=reason,
                error_kind=error_kind,
                attempts=delivery.attempts,
                starts=starts,
            )
            delivery.letter_published = True
        await self._settle(
            cmd, TurnDisposition.FAILED, error=_COPY_NOT_ANSWERED, dead_lettered=True
        )

    async def _publish_dead_letter(
        self,
        *,
        key: str,
        value: bytes,
        source: ConsumedRecord | None,
        reason: str,
        error_kind: str | None = None,
        attempts: int = 0,
        starts: int = 0,
    ) -> None:
        if self._broker is None:
            raise UCloneXError("StatelessAgentWorker needs a broker to dead-letter a command")
        await self._broker.publish_dead_letter(
            DeadLetterEnvelope(
                source_topic=source.topic if source is not None else "ucx.cmd.turn",
                source_partition=source.partition if source is not None else None,
                source_offset=source.offset if source is not None else None,
                key=key[:_DEAD_LETTER_KEY_CHARS],
                value_size=len(value),
                value_sha256=hashlib.sha256(value).hexdigest(),
                value_head_base64=base64.b64encode(value[:_DEAD_LETTER_HEAD_BYTES]).decode("ascii"),
                reason=reason,
                error_kind=error_kind,
                attempts=attempts,
                starts=starts,
                worker_id=self._worker_id,
                dead_lettered_at_ms=int(time.time() * 1000),
            )
        )

    async def dead_letter_malformed(
        self, record: ConsumedRecord, *, error_kind: str, wait: WaitFn = asyncio.sleep
    ) -> None:
        """Publish a record that is not a valid command; no retry can make it one.

        Its session is told only when the record names one that can be written to safely.
        """
        published = False

        async def step() -> None:
            nonlocal published
            if not published:
                await self._publish_dead_letter(
                    key=record.key,
                    value=record.raw_value,
                    source=record,
                    reason="malformed",
                    error_kind=error_kind,
                )
                published = True
            if record.topic == TOPIC_CMD_ROOM:
                room_id: object = record.payload.get("room_id")
                if self._room_steps is not None and isinstance(room_id, str):
                    if wire_id_problem(room_id) is None:
                        await self._room_steps.tell_unread(room_id, _COPY_NOT_READ)
                return
            if record.topic == TOPIC_BG_LEARN:
                return  # nobody is waiting on a lesson
            session_id: object = record.payload.get("session_id")
            turn_id: object = record.payload.get("turn_id")
            if (
                isinstance(session_id, str)
                and isinstance(turn_id, str)
                and wire_id_problem(session_id) is None
                and wire_id_problem(turn_id) is None
            ):
                terminal = _terminal(
                    TurnDisposition.FAILED, error=_COPY_NOT_READ, dead_lettered=True
                )
                await self._session_store.settle_turn(session_id, turn_id, terminal)
                await self._sse_publisher.publish_event(
                    session_id,
                    AgentEvent(
                        type=EventType.AGENT_REPLY,
                        topic=f"session.{session_id}",
                        session_id=session_id,
                        sender_id=self._worker_id,
                        payload={**terminal, "turn_id": turn_id},
                    ),
                )
                await self._land_malformed_seat(record, session_id, turn_id)

        await self._until_done(f"dead-letter the record at offset {record.offset}", step, wait)

    async def _land_malformed_seat(
        self, record: ConsumedRecord, session_id: str, turn_id: str
    ) -> None:
        """A seat turn that cannot be read still ends, or its room would wait for it."""
        if record.payload.get("seat") is None or self._broker is None:
            return
        try:
            seat = SeatTurnRef.model_validate(record.payload["seat"])
            ended = TurnCommandEnvelope(
                turn_id=turn_id,
                session_id=session_id,
                agent_id="unread",
                user_id=str(record.payload.get("user_id")),
                prompt="",
                enqueued_at_ms=0,
            )
            landed = _landed_command(ended, seat)
        except ValueError:
            logger.error("A seat turn at offset %d names no room it can land in", record.offset)
            return
        await self._broker.publish_room_step(landed)

    async def _acquire_lease(self, cmd: TurnCommandEnvelope) -> TurnLeaseToken:
        """Wait out another holder: its turn ends, or its deadline passes after a crash.

        Unbounded on purpose. Giving up would advance the offset of a turn nobody ran, and a
        holder that keeps renewing is a turn that is really running.
        """
        delay = _LEASE_RETRY_FIRST_SECONDS
        while True:
            try:
                return await self._lease_manager.acquire(
                    cmd.session_id, cmd.turn_id, self._worker_id
                )
            except TurnLeaseConflictError:
                await asyncio.sleep(delay)
                delay = min(delay * 2, _LEASE_RETRY_MAX_SECONDS)

    async def _attempt_turn(
        self, cmd: TurnCommandEnvelope, source: ConsumedRecord | None, delivery: _Delivery
    ) -> TurnOutcome:
        try:
            lease = await self._acquire_lease(cmd)
        except TurnPreemptedError:
            await self._settle(cmd, TurnDisposition.INTERRUPTED)
            return TurnOutcome(TurnDisposition.INTERRUPTED)
        started = False
        try:
            told = await self._session_store.settled_terminal(cmd.session_id, cmd.turn_id)
            if told is not None:
                # Redelivered: said again, because the failure that brought it back may have come
                # before the publish. At-least-once, so a client takes one per `turn_id`.
                await self._publish_terminal(cmd, told)
                return TurnOutcome(TurnDisposition.DUPLICATE)
            # Counted under the lease, so a copy waiting behind a running turn counts nothing,
            # and before anything can kill the process, so a record that takes the worker down
            # with it (out of memory, say) is bounded too.
            starts = await self._session_store.record_turn_start(cmd.session_id, cmd.turn_id)
            started = True
            if starts > self._max_turn_starts:
                await self._dead_letter(cmd, source, delivery, "started_repeatedly", starts=starts)
                return TurnOutcome(TurnDisposition.FAILED)
            if cmd.seat is not None and not await self._seat_may_run(cmd, cmd.seat):
                await self._settle(cmd, TurnDisposition.INTERRUPTED)
                return TurnOutcome(TurnDisposition.INTERRUPTED)
            state = await self._session_store.prefetch_session(cmd.session_id)
            found = state.revision if state is not None else 0
            if cmd.expected_revision is not None and found != cmd.expected_revision:
                # The person answered a conversation that has moved on; running it anyway
                # would answer the wrong context, and retrying meets the same mismatch.
                await self._settle(cmd, TurnDisposition.FAILED, error=_COPY_CONVERSATION_CHANGED)
                return TurnOutcome(TurnDisposition.FAILED)
            return await self._run_and_commit(cmd, lease, delivery)
        finally:
            try:
                self._session_store.evict_local_turn_cache(cmd.session_id)
                if started:
                    await self._refund_start(cmd)
            finally:
                await self._lease_manager.release(lease)

    async def _seat_may_run(self, cmd: TurnCommandEnvelope, seat: SeatTurnRef) -> bool:
        if self._room_steps is None:
            raise UCloneXError("StatelessAgentWorker needs room steps to run a seat turn")
        return await self._room_steps.seat_on_floor(cmd, seat)

    async def _refund_start(self, cmd: TurnCommandEnvelope) -> None:
        """This process lived to see the attempt end, however it ended."""
        await self._refund_delivery(cmd.session_id, cmd.turn_id)

    async def _refund_delivery(self, conversation_id: str, command_id: str) -> None:
        try:
            await self._session_store.refund_turn_start(conversation_id, command_id)
        except Exception:
            logger.warning("Could not refund a start of %s", command_id, exc_info=True)

    async def _run_and_commit(
        self, cmd: TurnCommandEnvelope, lease: TurnLeaseToken, delivery: _Delivery
    ) -> TurnOutcome:
        bus = EventBus()
        try:
            async with self._sse_publisher.attach_to_bus(bus, cmd.session_id):
                await self._emit(
                    cmd, EventType.USER_INPUT, {"event": "TURN_STARTED", "prompt": cmd.prompt}
                )
                delivery.attempts += 1
                ran = await self._run_turn(cmd, lease, bus)
        finally:
            await bus.stop()
        if isinstance(ran, TurnOutcome):
            return ran
        # The last look before the write (Invariant 4): an interrupt that arrived while the
        # model was answering still wins, and nothing of the turn lands.
        interrupted_late = await self._lease_manager.is_cancelled(cmd.session_id, cmd.turn_id)
        if interrupted_late:
            await self._settle(cmd, TurnDisposition.INTERRUPTED)
            return TurnOutcome(TurnDisposition.INTERRUPTED)
        disposition, terminal = self._committed_terminal(cmd, ran.result)
        if ran.landing is not None:
            # What the room's row says, in the receipt the room reads it from. Its error is the
            # plain one the person is told: raw `TurnResult.error` text stays in the log.
            told = terminal.get("error")
            terminal["landing"] = ran.landing.model_copy(update={"error": told}).model_dump_json()
        saved = await ran.commit(terminal)
        await self._publish_terminal(cmd, {**terminal, "revision": saved.revision})
        return TurnOutcome(disposition, state=saved, result=ran.result)

    async def _run_turn(
        self, cmd: TurnCommandEnvelope, lease: TurnLeaseToken, bus: EventBus
    ) -> TurnExecution | TurnOutcome:
        task = asyncio.ensure_future(self._turn_executor(cmd, bus))
        running = _RunningTurn(
            turn_id=cmd.turn_id,
            task=task,
            room_id=cmd.seat.room_id if cmd.seat is not None else None,
        )
        self._running[cmd.session_id] = running
        keeper = asyncio.create_task(self._keep_lease(cmd, lease, running))
        try:
            # Cancelling this coroutine cancels `task` through the await; `running.stop`
            # cancels only `task`, which is how an interrupt is told from the process leaving.
            return await task
        except asyncio.CancelledError:
            if running.lease_lost:
                raise _TurnLeaseLostError(cmd.turn_id) from None
            if running.interrupted:
                await self._settle(cmd, TurnDisposition.INTERRUPTED)
                return TurnOutcome(TurnDisposition.INTERRUPTED)
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise  # the process is leaving: the turn stays pending, and that is the only way
            # Nobody asked: something inside the turn was cancelled. A failure like any other.
            raise _TurnCancelledInsideError(cmd.turn_id) from None
        except PlainRefusalError as exc:  # written for a person, and no retry changes it
            logger.warning("Turn %s refused: %s", cmd.turn_id, exc)
            await self._settle(cmd, TurnDisposition.FAILED, error=str(exc))
            return TurnOutcome(TurnDisposition.FAILED)
        finally:
            keeper.cancel()
            del self._running[cmd.session_id]

    async def _keep_lease(
        self, cmd: TurnCommandEnvelope, lease: TurnLeaseToken, running: _RunningTurn
    ) -> None:
        """Renew the lease through a long turn, and catch an interrupt the broadcast missed."""
        while True:
            await asyncio.sleep(lease.ttl_seconds / 3)
            try:
                if await self._lease_manager.is_cancelled(cmd.session_id, cmd.turn_id):
                    running.stop(interrupted=True)
                    return
                if not await self._lease_manager.renew(lease):
                    running.stop(interrupted=False)
                    return
            except Exception:
                # A blip is not a lost lease; the commit's revision check is the real guard.
                logger.warning("Could not renew the lease of %s", cmd.turn_id, exc_info=True)

    @staticmethod
    def _committed_terminal(
        cmd: TurnCommandEnvelope, result: TurnResult
    ) -> tuple[TurnDisposition, dict[str, Any]]:
        if result.stop_reason == "usage_ceiling":
            disposition = TurnDisposition.STOPPED
        elif result.error is not None:
            disposition = TurnDisposition.FAILED
        else:
            disposition = TurnDisposition.COMPLETED
        if result.error is not None:
            logger.warning("Turn %s ended with an error: %s", cmd.turn_id, result.error)
        return disposition, _terminal(
            disposition,
            error=_plain_failure(result) if disposition is TurnDisposition.FAILED else None,
            error_code=result.error_code,
            content=result.content if disposition is TurnDisposition.COMPLETED else None,
            stop_reason=result.stop_reason,
        )

    async def _settle(
        self, cmd: TurnCommandEnvelope, disposition: TurnDisposition, **fields: Any
    ) -> None:
        """End a turn with no commit: the receipt first, then the person is told."""
        terminal = _terminal(disposition, **fields)
        await self._session_store.settle_turn(cmd.session_id, cmd.turn_id, terminal)
        await self._publish_terminal(cmd, terminal)

    async def _emit(
        self, cmd: TurnCommandEnvelope, event_type: EventType, payload: dict[str, Any]
    ) -> None:
        await self._sse_publisher.publish_event(
            cmd.session_id,
            AgentEvent(
                type=event_type,
                topic=f"session.{cmd.session_id}",
                session_id=cmd.session_id,
                sender_id=cmd.agent_id,
                payload={**payload, "turn_id": cmd.turn_id},
            ),
        )

    async def _publish_terminal(
        self, cmd: TurnCommandEnvelope, terminal: Mapping[str, Any]
    ) -> None:
        event_type = (
            EventType.INTERRUPT
            if terminal["event"] == TurnDisposition.INTERRUPTED.value
            else EventType.AGENT_REPLY
        )
        await self._emit(cmd, event_type, {k: v for k, v in terminal.items() if k != "landing"})
        if cmd.seat is not None:
            # Every way a turn ends comes through here, its receipt written, so a seat turn's room
            # hears of each of them, and hears again when this is redelivered.
            if self._broker is None:
                raise UCloneXError("StatelessAgentWorker needs a broker to land a seat turn")
            await self._broker.publish_room_step(_landed_command(cmd, cmd.seat))

    async def execute_room_command(
        self,
        cmd: RoomCommandEnvelope,
        source: ConsumedRecord | None = None,
        *,
        wait: WaitFn = asyncio.sleep,
    ) -> None:
        """Run one `ucx.cmd.room` command to its receipt, under the same policy as a turn."""
        if self._room_steps is None:
            raise UCloneXError("StatelessAgentWorker needs room steps to run a room command")
        steps = self._room_steps
        delivery = self._delivery_of(source)

        async def give_up(reason: str, exc: Exception | None) -> None:
            await self._give_up_on_room(
                cmd, source, delivery, reason, type(exc).__name__ if exc else None, wait
            )

        async def attempt() -> None:
            # Counted durably, as a turn's starts are: only a process dying under this command
            # keeps one, so a record that takes the worker down with it is bounded too.
            starts = await self._session_store.record_turn_start(cmd.room_id, cmd.turn_id)
            kept_dying = starts > self._max_turn_starts
            try:
                if kept_dying:
                    await give_up("started_repeatedly", None)
                    return
                delivery.attempts += 1
                await steps.run(cmd)
            finally:
                await self._refund_delivery(cmd.room_id, cmd.turn_id)

        async def after_stale(exc: Exception) -> None:
            # The learner or another copy of this command moved the room: the next attempt reads
            # it again, and finds this command's receipt if that is what moved it.
            logger.info("Room %s moved under command %s", cmd.room_id, cmd.turn_id)

        async def kept_changing() -> None:
            await give_up("kept_changing", None)

        await self._within_bounds(
            f"Room command {cmd.turn_id}",
            delivery,
            attempt,
            conflicts=(StaleRoomWriteError,),
            after_conflict=after_stale,
            kept_changing=kept_changing,
            give_up=give_up,
            wait=wait,
        )

    async def _give_up_on_room(
        self,
        cmd: RoomCommandEnvelope,
        source: ConsumedRecord | None,
        delivery: _Delivery,
        reason: str,
        error_kind: str | None,
        wait: WaitFn,
    ) -> None:
        """End a room command out of bounds: the letter, its receipt, then the person is told."""
        steps = self._room_steps
        assert steps is not None
        lettered = False

        async def step() -> None:
            nonlocal lettered
            if await steps.send_if_committed(cmd):
                return  # the failure was sending what it committed: nothing to give up on
            if not lettered:
                await self._publish_dead_letter(
                    key=cmd.room_id,
                    value=source.raw_value
                    if source is not None
                    else cmd.model_dump_json().encode(),
                    source=source,
                    reason=reason,
                    error_kind=error_kind,
                    attempts=delivery.attempts,
                )
                lettered = True
            await steps.run(cmd, can_select=False)

        await self._until_done(f"give up on room command {cmd.turn_id}", step, wait)
