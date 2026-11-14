"""`CommandBrokerProtocol` over Apache Kafka, through `aiokafka` (Apache-2.0).

The wire protocol, partitioning (Kafka's default murmur2 partitioner on the record key),
idempotent `acks=all` production, consumer groups and offset commits are all the library's.
This module only maps the wire envelopes onto topics and keys.

`aiokafka` is imported lazily, so `uclone_x` imports without the `cluster` extra installed.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, cast

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.core.cluster_protocols import (
    CommandBrokerProtocol,
    DeadLetterEnvelope,
    InterruptControlEnvelope,
    LearnCommandEnvelope,
    RoomCommandEnvelope,
    SSEStreamEnvelope,
    TurnCommandEnvelope,
)
from uclone_x.core.loop_local import LoopLocal

TOPIC_CMD_TURN: Final[str] = "ucx.cmd.turn"
TOPIC_CMD_ROOM: Final[str] = "ucx.cmd.room"
TOPIC_BG_LEARN: Final[str] = "ucx.bg.learn"
TOPIC_CTRL_INTERRUPT: Final[str] = "ucx.ctrl.interrupt"
TOPIC_STREAM_SSE: Final[str] = "ucx.stream.sse"
TOPIC_DEAD_LETTER: Final[str] = "ucx.dlq.turn"

#: Local partition counts (event-backbone design, topic table). The interrupt topic is a
#: single-partition broadcast; every worker reads it with its own consumer group.
DEFAULT_TOPIC_PARTITIONS: Final[Mapping[str, int]] = {
    TOPIC_CMD_TURN: 16,
    TOPIC_CMD_ROOM: 16,
    TOPIC_BG_LEARN: 8,
    TOPIC_STREAM_SSE: 16,
    TOPIC_CTRL_INTERRUPT: 1,
    TOPIC_DEAD_LETTER: 4,
}

__all__ = [
    "DEFAULT_TOPIC_PARTITIONS",
    "TOPIC_BG_LEARN",
    "TOPIC_CMD_ROOM",
    "TOPIC_CMD_TURN",
    "TOPIC_CTRL_INTERRUPT",
    "TOPIC_DEAD_LETTER",
    "TOPIC_STREAM_SSE",
    "ConsumedRecord",
    "KafkaCommandBroker",
    "PartitionsRevokedFn",
]


class ConsumedRecord(BaseModel):
    """One record read back from a topic partition."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    topic: str
    partition: int = Field(ge=0)
    offset: int = Field(ge=0)
    key: str
    payload: Mapping[str, Any]
    #: The value as consumed, for a dead letter: `payload` is empty when it was not JSON.
    raw_value: bytes = b""
    #: The class of what stopped the value being parsed, when something did.
    decode_error: str | None = None
    #: When the record was produced, by Kafka's clock for it; a turn's deadline counts from here.
    timestamp_ms: int | None = None


def _aiokafka() -> Any:
    return importlib.import_module("aiokafka")


#: Told which `(topic, partition)` pairs this consumer just lost.
PartitionsRevokedFn = Callable[[Sequence[tuple[str, int]]], None]


def _revocation_listener(on_revoked: PartitionsRevokedFn | None) -> Any:
    """A `ConsumerRebalanceListener` calling `on_revoked`, built here to keep the import lazy."""
    base: Any = _aiokafka().ConsumerRebalanceListener

    class _Listener(base):
        async def on_partitions_revoked(self, revoked: Any) -> None:
            if on_revoked is not None:
                on_revoked([(tp.topic, tp.partition) for tp in revoked])

        async def on_partitions_assigned(self, assigned: Any) -> None:
            return None

    return _Listener()


def _encode(payload: Mapping[str, Any]) -> bytes:
    # `default=str`: a stream event may carry an enum or a timestamp; a command never does,
    # because a command is a validated envelope dumped in JSON mode.
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")


class KafkaCommandBroker(CommandBrokerProtocol):
    """Kafka producer (and, when given a group, consumer) for the command and stream topics."""

    def __init__(
        self,
        *,
        bootstrap_servers: Sequence[str],
        group_id: str | None = None,
        topic_partitions: Mapping[str, int] | None = None,
        replication_factor: int = 1,
        request_timeout_ms: int = 15000,
        session_timeout_ms: int = 10000,
    ) -> None:
        if not bootstrap_servers:
            raise ValueError("KafkaCommandBroker needs at least one bootstrap server")
        self._bootstrap = ",".join(bootstrap_servers)
        self._group_id = group_id
        self._topic_partitions = dict(topic_partitions or DEFAULT_TOPIC_PARTITIONS)
        self._replication_factor = replication_factor
        self._request_timeout_ms = request_timeout_ms
        self._session_timeout_ms = session_timeout_ms
        self._producers: LoopLocal[Any] = LoopLocal(self._start_producer, self._stop_client)
        self._consumer: Any = None

    async def _start_producer(self) -> Any:
        producer: Any = _aiokafka().AIOKafkaProducer(
            bootstrap_servers=self._bootstrap,
            acks="all",
            enable_idempotence=True,
            linger_ms=5,
            request_timeout_ms=self._request_timeout_ms,
        )
        try:
            await producer.start()
        except BaseException:
            await producer.stop()  # a producer whose start failed still holds a client
            raise
        return producer

    @staticmethod
    async def _stop_client(client: Any) -> None:
        await client.stop()

    async def start(self) -> None:
        """Connect the producer on the current event loop."""
        await self._producers.get()

    async def stop(self) -> None:
        """Flush and close every producer, and the consumer if one was started."""
        await self._producers.aclose()
        if self._consumer is not None:
            consumer, self._consumer = self._consumer, None
            await consumer.stop()

    async def ensure_topics(self) -> tuple[str, ...]:
        """Create the declared topics that do not exist yet; return the ones created."""
        admin_mod: Any = importlib.import_module("aiokafka.admin")
        admin: Any = admin_mod.AIOKafkaAdminClient(
            bootstrap_servers=self._bootstrap, request_timeout_ms=self._request_timeout_ms
        )
        await admin.start()
        try:
            existing: set[str] = set(await admin.list_topics())
            missing = [
                admin_mod.NewTopic(name, partitions, self._replication_factor)
                for name, partitions in self._topic_partitions.items()
                if name not in existing
            ]
            if missing:
                await admin.create_topics(missing)
            return tuple(sorted(t.name for t in missing))
        finally:
            await admin.close()

    async def ping(self) -> bool:
        """Return True when the cluster answers a metadata request."""
        producer = await self._producers.get()
        await producer.client.fetch_all_metadata()
        return True

    async def _send(self, topic: str, key: str, payload: Mapping[str, Any]) -> int:
        producer = await self._producers.get()
        metadata: Any = await producer.send_and_wait(
            topic, _encode(payload), key=key.encode("utf-8")
        )
        offset: int = metadata.offset
        return offset

    async def publish_turn(self, cmd: TurnCommandEnvelope) -> int:
        """Publish a turn command keyed by `session_id`; returns the acknowledged offset."""
        return await self._send(TOPIC_CMD_TURN, cmd.session_id, cmd.model_dump(mode="json"))

    async def publish_room_step(self, cmd: RoomCommandEnvelope) -> int:
        """Publish one room speaker hop keyed by `room_id`."""
        return await self._send(TOPIC_CMD_ROOM, cmd.room_id, cmd.model_dump(mode="json"))

    async def publish_learn_job(self, cmd: LearnCommandEnvelope) -> int:
        """Publish a background learning job keyed by `room_id`."""
        return await self._send(TOPIC_BG_LEARN, cmd.room_id, cmd.model_dump(mode="json"))

    async def broadcast_interrupt(self, ctrl: InterruptControlEnvelope) -> None:
        """Publish an interrupt on the single-partition broadcast topic."""
        await self._send(TOPIC_CTRL_INTERRUPT, ctrl.conversation_id, ctrl.model_dump(mode="json"))

    async def publish_sse(self, conversation_id: str, envelope: SSEStreamEnvelope) -> None:
        """Publish a Centrifugo `publish` API command for Centrifugo's built-in Kafka consumer."""
        command = {
            "method": envelope.method,
            "payload": {
                "channel": envelope.channel,
                "data": dict(envelope.data),
                "idempotency_key": envelope.idempotency_key,
            },
        }
        await self._send(TOPIC_STREAM_SSE, conversation_id, command)

    async def publish_dead_letter(self, envelope: DeadLetterEnvelope) -> None:
        """Publish a command given up on, keyed like the original so one session stays ordered."""
        await self._send(TOPIC_DEAD_LETTER, envelope.key, envelope.model_dump(mode="json"))

    async def subscribe(
        self, topics: Sequence[str], *, on_revoked: PartitionsRevokedFn | None = None
    ) -> None:
        """Join `group_id` on `topics`, reading from the earliest uncommitted offset.

        `StickyPartitionAssignor` is the closest `aiokafka` has to the soft affinity the design
        wants: it keeps partitions where they were across a rebalance. `aiokafka` has no
        cooperative protocol, so a rebalance still revokes every partition first (eager);
        `on_revoked` is told. Its other assignors are round-robin and range.
        """
        if self._group_id is None:
            raise ValueError("KafkaCommandBroker was built without a consumer group_id")
        if self._consumer is not None:
            raise RuntimeError("KafkaCommandBroker consumer is already subscribed")
        sticky: Any = importlib.import_module(
            "aiokafka.coordinator.assignors.sticky.sticky_assignor"
        )
        consumer: Any = _aiokafka().AIOKafkaConsumer(
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            request_timeout_ms=self._request_timeout_ms,
            session_timeout_ms=self._session_timeout_ms,
            heartbeat_interval_ms=max(1000, self._session_timeout_ms // 3),
            partition_assignment_strategy=(sticky.StickyPartitionAssignor,),
        )
        try:
            await consumer.start()
            consumer.subscribe(topics=list(topics), listener=_revocation_listener(on_revoked))
        except BaseException:
            await consumer.stop()
            raise
        self._consumer = consumer

    async def subscribe_broadcast(self, topics: Sequence[str]) -> None:
        """Read `topics` with no consumer group, from now on: every worker sees every record.

        Nothing is committed and no group outlives the process. A record published before
        this returned is not seen, which is why an interrupt writes its tombstone first.
        """
        if self._consumer is not None:
            raise RuntimeError("KafkaCommandBroker consumer is already subscribed")
        consumer: Any = _aiokafka().AIOKafkaConsumer(
            *topics,
            bootstrap_servers=self._bootstrap,
            group_id=None,
            enable_auto_commit=False,
            auto_offset_reset="latest",
            request_timeout_ms=self._request_timeout_ms,
        )
        try:
            await consumer.start()
        except BaseException:
            await consumer.stop()
            raise
        self._consumer = consumer

    def pause_all(self) -> None:
        """Stop fetching from every assigned partition; polling keeps the membership alive."""
        if self._consumer is not None:
            self._consumer.pause(*self._consumer.assignment())

    def resume_all(self) -> None:
        """Fetch again from every assigned partition."""
        if self._consumer is not None:
            self._consumer.resume(*self._consumer.assignment())

    async def poll(
        self, timeout_ms: int = 1000, max_records: int = 500
    ) -> tuple[ConsumedRecord, ...]:
        """Return the records fetched within `timeout_ms` (possibly none)."""
        if self._consumer is None:
            raise RuntimeError("call subscribe() before poll()")
        batches: Mapping[Any, Sequence[Any]] = await self._consumer.getmany(
            timeout_ms=timeout_ms, max_records=max_records
        )
        records: list[ConsumedRecord] = []
        for batch in batches.values():
            for message in batch:
                key_raw: bytes | None = message.key
                decode_error: str | None = None
                try:
                    payload: Any = json.loads(message.value or b"")
                except Exception as exc:  # `RecursionError`, too, on a deeply nested value
                    # A broken record must not stop the partition behind it.
                    payload = {}
                    decode_error = type(exc).__name__
                records.append(
                    ConsumedRecord(
                        topic=message.topic,
                        partition=message.partition,
                        offset=message.offset,
                        # A key that is not UTF-8 must not stop the poll loop for everyone.
                        key=key_raw.decode("utf-8", errors="replace")
                        if key_raw is not None
                        else "",
                        payload=(
                            cast(dict[str, Any], payload)
                            if isinstance(payload, dict)
                            else {"value": payload}
                        ),
                        raw_value=message.value or b"",
                        decode_error=decode_error,
                        timestamp_ms=message.timestamp if message.timestamp > 0 else None,
                    )
                )
        return tuple(records)

    async def commit_partition_offset(self, topic: str, partition: int, offset: int) -> None:
        """Commit `offset` as the next offset to read on `(topic, partition)` (Kafka semantics).

        `PartitionOffsetTracker.mark_completed` returns exactly this value: the highest
        contiguous completed offset plus one.
        """
        if partition < 0 or offset < 0:
            raise ValueError(
                f"partition and offset must be non-negative, got ({partition}, {offset})"
            )
        if self._consumer is None:
            raise RuntimeError("call subscribe() before committing offsets")
        tp: Any = _aiokafka().TopicPartition(topic, partition)
        await self._consumer.commit({tp: offset})
