"""The `ucx worker` process: Kafka in, `StatelessAgentWorker` per record, watermark out.

The poll loop never awaits a turn. Each `ucx.cmd.turn` and `ucx.cmd.room` record goes to the keyed
dispatcher, which runs one per session or room at a time and commits only contiguous finished
offsets (Invariant 3). A second consumer with no group reads `ucx.ctrl.interrupt`, so every worker
sees every interrupt (Invariant 4). With a learner, a third reads `ucx.bg.learn`, with a dispatcher
of its own, so a lesson holds no turn's slot and no room's offset.

**A turn's failure does not stop this loop.** `StatelessAgentWorker` retries it in place and ends
it within its bounds. Commands queued behind a failing turn still count toward the pause on
fetching. The process exits for a shutdown, and nonzero when a Kafka consumer raises.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Final, Protocol

from pydantic import ValidationError

from uclone_x.adapters.cluster.kafka_broker import (
    TOPIC_BG_LEARN,
    TOPIC_CMD_ROOM,
    TOPIC_CMD_TURN,
    ConsumedRecord,
    PartitionsRevokedFn,
)
from uclone_x.core.cluster_protocols import (
    InterruptControlEnvelope,
    LearnCommandEnvelope,
    RoomCommandEnvelope,
    TurnCommandEnvelope,
)
from uclone_x.shells.cluster.dispatcher import KeyedPartitionDispatcher
from uclone_x.shells.cluster.learn_worker import RoomLearnWorker
from uclone_x.shells.cluster.worker_runner import StatelessAgentWorker, TurnOutcome

__all__ = ["ClusterWorkerProcess", "CommandConsumerProtocol", "ControlConsumerProtocol"]

logger = logging.getLogger(__name__)

_POLL_TIMEOUT_MS: Final[int] = 200
#: Lessons learned at once: each is a model call nobody is waiting on.
_LEARN_CONCURRENCY: Final[int] = 2


class CommandConsumerProtocol(Protocol):
    """The consumer-group half of `KafkaCommandBroker` this process uses."""

    async def subscribe(
        self, topics: Sequence[str], *, on_revoked: PartitionsRevokedFn | None = None
    ) -> None: ...
    async def poll(
        self, timeout_ms: int = 1000, max_records: int = 500
    ) -> tuple[ConsumedRecord, ...]: ...
    async def commit_partition_offset(self, topic: str, partition: int, offset: int) -> None: ...
    def pause_all(self) -> None: ...
    def resume_all(self) -> None: ...
    async def stop(self) -> None: ...


class ControlConsumerProtocol(Protocol):
    """The broadcast half of `KafkaCommandBroker` this process uses."""

    async def subscribe_broadcast(self, topics: Sequence[str]) -> None: ...
    async def poll(
        self, timeout_ms: int = 1000, max_records: int = 500
    ) -> tuple[ConsumedRecord, ...]: ...
    async def stop(self) -> None: ...


class ClusterWorkerProcess:
    """One worker's consume loop, control listener and shutdown."""

    def __init__(
        self,
        *,
        worker: StatelessAgentWorker,
        consumer: CommandConsumerProtocol,
        control: ControlConsumerProtocol,
        concurrency: int = 16,
        control_topic: str,
        drain_seconds: float = 30.0,
        learner: RoomLearnWorker | None = None,
        learn_consumer: CommandConsumerProtocol | None = None,
    ) -> None:
        self._worker = worker
        self._consumer = consumer
        self._control = control
        self._control_topic = control_topic
        self._drain_seconds = drain_seconds
        self._dispatcher = KeyedPartitionDispatcher(self._commit, max_concurrent_turns=concurrency)
        self._paused = False
        self._learner = learner
        self._learn_consumer = learn_consumer
        self._learn_dispatcher = KeyedPartitionDispatcher(
            self._commit_learned, max_concurrent_turns=_LEARN_CONCURRENCY
        )

    async def _commit_learned(self, topic: str, partition: int, offset: int) -> None:
        assert self._learn_consumer is not None
        try:
            await self._learn_consumer.commit_partition_offset(topic, partition, offset)
        except Exception:  # a job's receipt makes the redelivered copy a no-op
            logger.warning("Could not commit %s[%d] at %d", topic, partition, offset, exc_info=True)

    async def _commit(self, topic: str, partition: int, offset: int) -> None:
        try:
            await self._consumer.commit_partition_offset(topic, partition, offset)
        except Exception:
            # Commits fail during a rebalance. Nothing is lost: the session committed, so the
            # redelivered record finds the turn's receipt.
            logger.warning("Could not commit %s[%d] at %d", topic, partition, offset, exc_info=True)

    def _on_revoked(self, partitions: Sequence[tuple[str, int]]) -> None:
        for topic, partition in partitions:
            self._dispatcher.forget_partition(topic, partition)

    @staticmethod
    def _note_if_broken(task: asyncio.Task[object]) -> None:
        """A failure that escapes work is a bug here: its offset is stuck."""
        if not task.cancelled() and task.exception() is not None:
            logger.critical("A turn's work raised; its offset stays", exc_info=task.exception())

    def _dispatch(self, record: ConsumedRecord) -> None:
        malformed = record.decode_error
        cmd: TurnCommandEnvelope | RoomCommandEnvelope | None
        try:
            cmd = (
                RoomCommandEnvelope.model_validate(record.payload)
                if record.topic == TOPIC_CMD_ROOM
                else TurnCommandEnvelope.model_validate(record.payload)
            )
        except Exception as exc:  # only parsing is in here, and no value may stop this loop
            # Not a command, and no retry makes it one. The log names where; the value may hold
            # anything, so it goes to the dead-letter topic and not here.
            logger.error(
                "Dead-lettering a malformed record at %s[%d]@%d",
                record.topic,
                record.partition,
                record.offset,
            )
            cmd = None
            malformed = malformed or type(exc).__name__

        async def work() -> TurnOutcome | None:
            if cmd is None:
                await self._worker.dead_letter_malformed(
                    record, error_kind=str(malformed), wait=self._dispatcher.wait_outside_slot
                )
                return None
            if isinstance(cmd, RoomCommandEnvelope):
                await self._worker.execute_room_command(
                    cmd, record, wait=self._dispatcher.wait_outside_slot
                )
                return None
            return await self._worker.execute_single_turn(
                cmd, source=record, wait=self._dispatcher.wait_outside_slot
            )

        task: asyncio.Task[object] = self._dispatcher.dispatch(
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            conversation_id=record.key,
            work=work,
        )
        task.add_done_callback(self._note_if_broken)

    def _dispatch_learn(self, record: ConsumedRecord) -> None:
        learner = self._learner
        assert learner is not None
        job: LearnCommandEnvelope | None
        malformed = record.decode_error
        try:
            job = LearnCommandEnvelope.model_validate(record.payload)
        except Exception as exc:  # a job that does not parse is a letter, and stops nothing
            job = None
            malformed = malformed or type(exc).__name__

        async def work() -> None:
            if job is not None:
                await learner.process_learn_job(job)
                return
            try:
                await self._worker.dead_letter_malformed(
                    record,
                    error_kind=str(malformed),
                    wait=self._learn_dispatcher.wait_outside_slot,
                )
            except Exception:
                logger.exception("Could not dead-letter a learn job at %d", record.offset)

        self._learn_dispatcher.dispatch(
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            conversation_id=record.key,
            work=work,
        ).add_done_callback(self._note_if_broken)

    def _on_learn_revoked(self, partitions: Sequence[tuple[str, int]]) -> None:
        for topic, partition in partitions:
            self._learn_dispatcher.forget_partition(topic, partition)

    async def _consume_learn_jobs(self) -> None:
        consumer = self._learn_consumer
        assert consumer is not None
        await consumer.subscribe([TOPIC_BG_LEARN], on_revoked=self._on_learn_revoked)
        paused = False
        while True:
            full = self._learn_dispatcher.in_flight >= self._learn_dispatcher.max_concurrent_turns
            if full != paused:
                (consumer.pause_all if full else consumer.resume_all)()
                paused = full
            for record in await consumer.poll(timeout_ms=_POLL_TIMEOUT_MS):
                self._dispatch_learn(record)

    def _apply_backpressure(self) -> None:
        full = self._dispatcher.in_flight >= self._dispatcher.max_concurrent_turns
        if full and not self._paused:
            self._consumer.pause_all()
            self._paused = True
        elif not full and self._paused:
            self._consumer.resume_all()
            self._paused = False

    async def _listen_for_interrupts(self) -> None:
        while True:
            for record in await self._control.poll(timeout_ms=_POLL_TIMEOUT_MS):
                try:
                    ctrl = InterruptControlEnvelope.model_validate(record.payload)
                except ValidationError:
                    logger.error("Skipping a malformed interrupt at offset %d", record.offset)
                    continue
                self._worker.interrupt(ctrl.conversation_id, ctrl.turn_id)

    async def run(self, stop: asyncio.Event) -> None:
        """Consume until `stop` is set, then drain; a consumer that raises ends this at once."""
        await self._consumer.subscribe(
            [TOPIC_CMD_TURN, TOPIC_CMD_ROOM], on_revoked=self._on_revoked
        )
        await self._control.subscribe_broadcast([self._control_topic])
        listener = asyncio.create_task(self._listen_for_interrupts())
        lessons = (
            asyncio.create_task(self._consume_learn_jobs())
            if self._learner is not None and self._learn_consumer is not None
            else None
        )
        background = [t for t in (listener, lessons) if t is not None]
        try:
            while not stop.is_set():
                for task in background:
                    if task.done():
                        task.result()  # a consumer died: raise why
                self._apply_backpressure()
                for record in await self._consumer.poll(timeout_ms=_POLL_TIMEOUT_MS):
                    self._dispatch(record)
            await self._dispatcher.drain(self._drain_seconds)
        finally:
            await self._dispatcher.drain(0.0)  # a cancelled `run` leaves nothing behind it
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            await self._learn_dispatcher.drain(0.0)  # a lesson redelivered is a lesson not lost
            await self._consumer.stop()
            await self._control.stop()
            if self._learn_consumer is not None:
                await self._learn_consumer.stop()
