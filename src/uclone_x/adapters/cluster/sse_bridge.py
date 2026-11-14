"""Event bus to `ucx.stream.sse` bridge with cluster-monotonic sequence stamping (#565).

Centrifugo consumes `ucx.stream.sse` with its built-in Kafka consumer and keeps channel history
in the Valkey cluster, so this bridge only stamps the sequence and hands the envelope to the
broker; there is no HTTP publisher.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from uclone_x.core.cluster_protocols import (
    CacheClientProtocol,
    CommandBrokerProtocol,
    SSEStreamEnvelope,
    SSEStreamPublisherProtocol,
)
from uclone_x.core.immutable import unwrap_immutable

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from uclone_x.engine.event_bus import AgentEvent
    from uclone_x.engine.protocols import EventBusProtocol

SSEEnvelopeSink = Callable[[SSEStreamEnvelope], Awaitable[None] | None]

__all__ = [
    "MAX_EVENT_BYTES",
    "KafkaSSEBridge",
    "SSEEnvelopeSink",
]

#: An event's data in encoded bytes, past which it is cut. Kafka refuses a record over 1 MiB, so
#: this leaves room for the rest of the record.
MAX_EVENT_BYTES: Final[int] = 900_000
#: What is left of a text field that was cut.
_CUT_TO_CHARS: Final[int] = 16_000
#: What says which event this is; never cut.
_NEVER_CUT: Final[frozenset[str]] = frozenset({"event", "turn_id"})


def _encoded_size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), default=str))


def _fitting(data: dict[str, Any]) -> dict[str, Any]:
    """`data`, or when it is too big to publish, the same event with its bulkiest fields cut.

    Largest first: text keeps its start, and anything else goes. `truncated` says so, and the
    whole of a reply stays in the session's state.
    """
    if _encoded_size(data) <= MAX_EVENT_BYTES:
        return data
    payload: dict[str, Any] = {**data["payload"], "truncated": True}
    bulkiest = sorted(
        (name for name in payload if name not in _NEVER_CUT),
        key=lambda name: _encoded_size(payload[name]),
        reverse=True,
    )
    for name in bulkiest:
        value = payload[name]
        payload[name] = value[:_CUT_TO_CHARS] if isinstance(value, str) else None
        if _encoded_size({**data, "payload": payload}) <= MAX_EVENT_BYTES:
            break
    return {**data, "payload": payload}


@dataclass
class _ConversationOrder:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class KafkaSSEBridge(SSEStreamPublisherProtocol):
    """Bridges an in-process `EventBusProtocol` to `ucx.stream.sse` with monotonic sequence numbers.

    Why Valkey `INCR` is used instead of `AgentEvent.sequence`: `AgentEvent.sequence` is a
    process-local FIFO tie-breaker that resets on every stateless worker turn (#565).
    `INCR seq:sse:{conversation_id}` gives one strictly monotonic sequence per conversation
    across every worker pod. The key is coordination state and carries no TTL, so
    `volatile-lru` never evicts it and a sequence never restarts at 1.
    """

    def __init__(
        self,
        cache: CacheClientProtocol,
        broker: CommandBrokerProtocol,
        channel_kind: Literal["session", "room"] = "session",
        sink: SSEEnvelopeSink | None = None,
    ) -> None:
        if channel_kind not in ("session", "room"):
            raise ValueError(f"channel_kind must be 'session' or 'room', got {channel_kind!r}")
        self._cache = cache
        self._broker = broker
        self._channel_kind: Literal["session", "room"] = channel_kind
        self._sink = sink
        # The sequence and the send are one step per conversation: without this, two events
        # of a turn publishing at once can reach the topic in the opposite order to their
        # sequence numbers (a bus callback and the turn's stream callback run concurrently).
        self._order: dict[str, _ConversationOrder] = {}

    @staticmethod
    def sequence_key(conversation_id: str) -> bytes:
        return f"seq:sse:{{{conversation_id}}}".encode()

    async def publish_event(self, conversation_id: str, event: AgentEvent) -> int:
        """Allocate the next cluster-wide sequence number and publish an `SSEStreamEnvelope`."""
        clean_id = conversation_id.strip()
        if not clean_id:
            raise ValueError("conversation_id must not be blank")
        order = self._order.setdefault(clean_id, _ConversationOrder())
        order.users += 1
        try:
            async with order.lock:
                return await self._publish_in_order(clean_id, event)
        finally:
            order.users -= 1
            if order.users == 0:
                del self._order[clean_id]  # zero idle state: no entry outlives its events

    async def _publish_in_order(self, clean_id: str, event: AgentEvent) -> int:
        seq = await self._cache.incr(self.sequence_key(clean_id), 1)
        unwrapped_payload: Any = unwrap_immutable(event.payload)
        payload_dict: dict[str, Any] = (
            dict(cast(dict[str, Any], unwrapped_payload))
            if isinstance(unwrapped_payload, dict)
            else {}
        )
        envelope = SSEStreamEnvelope(
            channel=f"ucx:{self._channel_kind}:{clean_id}",
            idempotency_key=f"{clean_id}:{seq}",
            data=_fitting(
                {
                    "seq": seq,
                    "event_id": event.event_id,
                    "topic": event.topic,
                    "type": str(event.type),
                    "sender_id": event.sender_id,
                    "payload": payload_dict,
                }
            ),
        )
        await self._broker.publish_sse(clean_id, envelope)
        if self._sink is not None:
            outcome = self._sink(envelope)
            if inspect.isawaitable(outcome):
                await outcome
        return seq

    def attach_to_bus(
        self, bus: EventBusProtocol, conversation_id: str
    ) -> AbstractAsyncContextManager[None]:
        """Subscribe to `bus` for a turn and forward every `AgentEvent` to `publish_event`."""

        @asynccontextmanager
        async def _lifecycle() -> AsyncGenerator[None, None]:
            async def _forward(event: AgentEvent) -> None:
                await self.publish_event(conversation_id, event)

            unsubscribe = bus.subscribe_callback(
                "*",
                _forward,
                capabilities=frozenset({"wildcard_subscribe"}),
            )
            try:
                yield
                await bus.wait_until_idle()
            finally:
                unsubscribe()

        return _lifecycle()
