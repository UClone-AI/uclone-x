"""Trivial fakes at the cluster Protocol boundary, for offline unit tests.

They record calls and hold values in dicts. They do not emulate Valkey scripts, SQL or
Kafka: behaviour that depends on the real server (atomic lease scripts, eviction, revision
OCC, consumer offsets) is tested against the docker compose cluster in
`tests/integration/cluster_live/`. `FakeCache.run_script` therefore raises.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from uclone_x.core.cluster_protocols import (
    CacheClientProtocol,
    CommandBrokerProtocol,
    DeadLetterEnvelope,
    InterruptControlEnvelope,
    LearnCommandEnvelope,
    RoomCommandEnvelope,
    SSEStreamEnvelope,
    TurnCommandEnvelope,
)

__all__ = ["FakeBroker", "FakeCache"]


class FakeCache(CacheClientProtocol):
    """A dict. `ttls` records the TTL each cache write asked for (None for `incr`)."""

    def __init__(self) -> None:
        self.values: dict[bytes, bytes] = {}
        self.ttls: dict[bytes, int | None] = {}
        self.closed = False

    async def get(self, key: bytes) -> bytes | None:
        return self.values.get(key)

    async def multi_get(self, *keys: bytes) -> tuple[bytes | None, ...]:
        return tuple(self.values.get(k) for k in keys)

    async def set_cache(self, key: bytes, value: bytes, *, ttl_seconds: int) -> None:
        self.values[key] = value
        self.ttls[key] = ttl_seconds

    async def add_cache(self, key: bytes, value: bytes, *, ttl_seconds: int) -> bool:
        if key in self.values:
            return False
        await self.set_cache(key, value, ttl_seconds=ttl_seconds)
        return True

    async def hash_get_all(self, key: bytes) -> Mapping[bytes, bytes]:
        raise AssertionError("no fake here keeps a hash")

    async def delete(self, *keys: bytes) -> int:
        removed = 0
        for key in keys:
            if self.values.pop(key, None) is not None:
                removed += 1
            self.ttls.pop(key, None)
        return removed

    async def incr(self, key: bytes, amount: int = 1) -> int:
        value = int(self.values.get(key, b"0")) + amount
        self.values[key] = str(value).encode()
        self.ttls[key] = None
        return value

    async def run_script(
        self, source: str, keys: Sequence[bytes], args: Sequence[bytes | str | int]
    ) -> Any:
        raise NotImplementedError(
            "FakeCache does not emulate Valkey scripts; test this against the live cluster "
            "(tests/integration/cluster_live)"
        )

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


class FakeBroker(CommandBrokerProtocol):
    """Records every publish; offsets are list positions."""

    def __init__(self) -> None:
        self.turns: list[TurnCommandEnvelope] = []
        self.room_steps: list[RoomCommandEnvelope] = []
        self.learn_jobs: list[LearnCommandEnvelope] = []
        self.interrupts: list[InterruptControlEnvelope] = []
        self.sse: list[tuple[str, SSEStreamEnvelope]] = []
        self.dead_letters: list[DeadLetterEnvelope] = []
        self.commits: list[tuple[str, int, int]] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def publish_turn(self, cmd: TurnCommandEnvelope) -> int:
        self.turns.append(cmd)
        return len(self.turns) - 1

    async def publish_room_step(self, cmd: RoomCommandEnvelope) -> int:
        self.room_steps.append(cmd)
        return len(self.room_steps) - 1

    async def publish_learn_job(self, cmd: LearnCommandEnvelope) -> int:
        self.learn_jobs.append(cmd)
        return len(self.learn_jobs) - 1

    async def broadcast_interrupt(self, ctrl: InterruptControlEnvelope) -> None:
        self.interrupts.append(ctrl)

    async def publish_dead_letter(self, envelope: DeadLetterEnvelope) -> None:
        self.dead_letters.append(envelope)

    async def publish_sse(self, conversation_id: str, envelope: SSEStreamEnvelope) -> None:
        self.sse.append((conversation_id, envelope))

    async def commit_partition_offset(self, topic: str, partition: int, offset: int) -> None:
        self.commits.append((topic, partition, offset))
