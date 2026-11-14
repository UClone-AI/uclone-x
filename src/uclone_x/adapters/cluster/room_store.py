"""Distributed room store: immediate PostgreSQL OCC flush (#1366), then a revisioned Valkey head."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

from uclone_x.adapters.cluster.config import ClusterCachePolicy
from uclone_x.core.cluster_protocols import (
    VERSIONED_HEAD_SET_SCRIPT,
    CacheClientProtocol,
    ClusterRoomStoreProtocol,
    RoomStepOutbox,
    SqlPoolProtocol,
    decode_versioned_head,
)
from uclone_x.core.loop_local import run_on_bridge_loop
from uclone_x.errors import StaleRoomWriteError
from uclone_x.room.models import RoomState

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

__all__ = ["ValkeyPostgresRoomStore"]

_T = TypeVar("_T")


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RevisionCarrier:
    revision: int


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ValkeyPostgresRoomStore(ClusterRoomStoreProtocol):
    """Cluster-safe room persistence: PostgreSQL revision OCC, then a revisioned Valkey head.

    Every `save` commits to PostgreSQL before returning, so a pre-turn `turns_started`
    increment (`#1366`) survives a worker crash before any model or tool call begins.
    """

    def __init__(
        self,
        pool: SqlPoolProtocol,
        cache: CacheClientProtocol,
        *,
        policy: ClusterCachePolicy | None = None,
        portal: BlockingPortal | None = None,
    ) -> None:
        self._pool = pool
        self._cache = cache
        self._policy = policy or ClusterCachePolicy()
        self._portal = portal

    def _run_sync(self, fn: Callable[..., Coroutine[Any, Any, _T]], *args: Any) -> _T:
        if self._portal is not None:
            return self._portal.call(fn, *args)
        return run_on_bridge_loop(fn(*args))

    @staticmethod
    def _room_key(room_id: str) -> bytes:
        return f"room:head:{{{room_id}}}".encode()

    async def _cache_head(self, room_id: str, revision: int, state_json: str) -> None:
        await self._cache.run_script(
            VERSIONED_HEAD_SET_SCRIPT,
            [self._room_key(room_id)],
            [revision, state_json, self._policy.head_ttl_seconds],
        )

    async def _load_async(self, room_id: str) -> RoomState | None:
        raw = await self._cache.get(self._room_key(room_id))
        if raw is not None:
            try:
                _, state_json = decode_versioned_head(raw)
                return RoomState.model_validate_json(state_json)
            except (ValueError, RecursionError):
                logger.warning("Dropping an unreadable cached head of room %s", room_id)
                await self._cache.delete(self._room_key(room_id))

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT revision AS committed_rev, state_json FROM room_states WHERE room_id = $1",
                room_id,
            )
        if row is None or not row.get("state_json"):
            return None
        state_json = str(row["state_json"])
        await self._cache_head(room_id, int(row["committed_rev"]), state_json)
        return RoomState.model_validate_json(state_json)

    def load(self, room_id: str) -> RoomState | None:
        """Return the stored `RoomState` for `room_id`, or `None` if absent."""
        return self._run_sync(self._load_async, room_id)

    async def read_room(self, room_id: str) -> RoomState | None:
        """The room in PostgreSQL. A commit moves the head after it, so the head can be behind."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state_json FROM room_states WHERE room_id = $1", room_id
            )
        return RoomState.model_validate_json(str(row["state_json"])) if row else None

    async def step_receipt(self, room_id: str, command_id: str) -> RoomStepOutbox | None:
        """What `command_id`'s commit left to send, read from the source of truth."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT outbox_json FROM room_step_receipts WHERE room_id = $1 AND command_id = $2",
                room_id,
                command_id,
            )
        return RoomStepOutbox.model_validate_json(str(row["outbox_json"])) if row else None

    async def commit_step(
        self, state: RoomState | None, *, room_id: str, command_id: str, outbox: RoomStepOutbox
    ) -> RoomState | None:
        """Save `state` (if any) and the step's receipt in one transaction."""
        receipt = (command_id, outbox.model_dump_json())
        if state is not None:
            return await self._save_async(state, receipt)
        async with self._pool.acquire() as conn:
            await self._insert_receipt(conn, room_id, receipt)
        return None

    @staticmethod
    async def _insert_receipt(conn: Any, room_id: str, receipt: tuple[str, str]) -> None:
        status = await conn.execute(
            "INSERT INTO room_step_receipts (room_id, command_id, outbox_json) "
            "VALUES ($1, $2, $3) ON CONFLICT (room_id, command_id) DO NOTHING",
            room_id,
            *receipt,
        )
        if not status.endswith(" 1"):
            # Another worker committed this command. Rolled back with the rest; the retry finds
            # the receipt.
            raise StaleRoomWriteError(
                f"The step {receipt[0]!r} of conversation {room_id!r} was already committed."
            )
        # The step has come to an end, so its starts are not a process dying under it.
        await conn.execute(
            "DELETE FROM turn_delivery_attempts WHERE session_id = $1 AND turn_id = $2",
            room_id,
            receipt[0],
        )

    async def _save_async(
        self, state: RoomState, receipt: tuple[str, str] | None = None
    ) -> RoomState:
        room_id = state.room_id
        carrier = _RevisionCarrier(state.revision + 1)
        stamped = state.model_copy(
            update={
                "updated_at": _now_iso(),
                "revision": carrier.revision,
            }
        )
        state_json = stamped.model_dump_json()
        validated = RoomState.model_validate_json(state_json)
        record = validated.file_record

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                if state.revision == 0:
                    status = await conn.execute(
                        "INSERT INTO room_states (room_id, revision, turns_started, "
                        "turns_landed, state_json, updated_at) VALUES ($1, $2, $3, $4, $5, $6) "
                        "ON CONFLICT (room_id) DO NOTHING",
                        room_id,
                        validated.revision,
                        record.turns_started,
                        record.turns_landed,
                        state_json,
                        validated.updated_at,
                    )
                else:
                    status = await conn.execute(
                        "UPDATE room_states SET revision = $2, turns_started = $3, "
                        "turns_landed = $4, state_json = $5, updated_at = $6 "
                        "WHERE room_id = $1 AND revision = $7",
                        room_id,
                        validated.revision,
                        record.turns_started,
                        record.turns_landed,
                        state_json,
                        validated.updated_at,
                        state.revision,
                    )
                if not status.endswith(" 1"):
                    row = await conn.fetchrow(
                        "SELECT revision AS committed_rev FROM room_states WHERE room_id = $1",
                        room_id,
                    )
                    if row is None:
                        raise StaleRoomWriteError(
                            f"The conversation {room_id!r} could not be saved: expected revision "
                            f"{state.revision}, but no room exists."
                        )
                    raise StaleRoomWriteError(
                        f"The conversation {room_id!r} could not be saved: expected revision "
                        f"{state.revision}, actual revision {int(row['committed_rev'])}."
                    )
                if receipt is not None:
                    await self._insert_receipt(conn, room_id, receipt)

        await self._cache_head(room_id, validated.revision, state_json)
        return validated

    def save(self, state: RoomState) -> RoomState:
        """Persist `state` at the next revision with immediate PostgreSQL durability."""
        return self._run_sync(self._save_async, state)

    async def _delete_async(self, room_id: str) -> bool:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM room_step_receipts WHERE room_id = $1", room_id)
            status = await conn.execute("DELETE FROM room_states WHERE room_id = $1", room_id)
        await self._cache.delete(self._room_key(room_id))
        return status == "DELETE 1"

    def delete(self, room_id: str) -> bool:
        """Remove `room_id` from PostgreSQL and Valkey."""
        return self._run_sync(self._delete_async, room_id)

    async def _list_room_ids_async(self) -> tuple[str, ...]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT room_id FROM room_states ORDER BY room_id")
        return tuple(str(r["room_id"]) for r in rows)

    def list_room_ids(self) -> tuple[str, ...]:
        """Return all stored room IDs in sorted order."""
        return self._run_sync(self._list_room_ids_async)
