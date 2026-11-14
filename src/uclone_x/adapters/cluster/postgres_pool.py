"""PostgreSQL pool for the cluster stores, through `asyncpg` (Apache-2.0), and the schema DDL.

`asyncpg`'s `Pool` already satisfies `SqlPoolProtocol` (`acquire()` as an async context
manager yielding a connection with `fetchrow`/`fetch`/`execute`/`executemany`/`transaction`),
so this module only creates one per event loop. psycopg is not used: it is LGPL.

`asyncpg` is imported lazily, so `uclone_x` imports without the `cluster` extra installed.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Final, cast

from uclone_x.core.cluster_protocols import SqlConnProtocol, SqlPoolProtocol
from uclone_x.core.loop_local import LoopLocal

__all__ = ["CLUSTER_SCHEMA_SQL", "PostgresPool", "bootstrap_schema"]

CLUSTER_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS session_snapshots (
    session_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    revision BIGINT NOT NULL,
    state_json TEXT NOT NULL,
    budget_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_turn_receipts (
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    revision BIGINT,
    terminal_json TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_id)
);

CREATE TABLE IF NOT EXISTS turn_delivery_attempts (
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    starts BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, turn_id)
);

CREATE TABLE IF NOT EXISTS session_context_blobs (
    session_id TEXT NOT NULL,
    digest TEXT NOT NULL,
    body TEXT NOT NULL,
    PRIMARY KEY (session_id, digest)
);

CREATE TABLE IF NOT EXISTS session_events (
    session_id TEXT NOT NULL,
    seq BIGINT NOT NULL,
    event_json TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS room_states (
    room_id TEXT PRIMARY KEY,
    revision BIGINT NOT NULL,
    turns_started BIGINT NOT NULL DEFAULT 0,
    turns_landed BIGINT NOT NULL DEFAULT 0,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS room_step_receipts (
    room_id TEXT NOT NULL,
    command_id TEXT NOT NULL,
    outbox_json TEXT NOT NULL,
    PRIMARY KEY (room_id, command_id)
);

CREATE TABLE IF NOT EXISTS knowledge_triples (
    triple_id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    up_to_revision BIGINT NOT NULL DEFAULT 0,
    triple_json TEXT
);

CREATE TABLE IF NOT EXISTS ucx_knowledge_triples (
    clone_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_val TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    source_room_id TEXT,
    PRIMARY KEY (clone_id, subject, predicate, object_val)
);

CREATE TABLE IF NOT EXISTS usage_ledger (
    id BIGSERIAL PRIMARY KEY,
    scope TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    tokens BIGINT NOT NULL,
    prompt_tokens BIGINT,
    completion_tokens BIGINT,
    cached_tokens BIGINT,
    count_source TEXT NOT NULL,
    counts_against_limits BOOLEAN NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS usage_ledger_scope_recorded_at
    ON usage_ledger (scope, recorded_at);
""".strip()


class PostgresPool(SqlPoolProtocol):
    """`SqlPoolProtocol` over one `asyncpg` pool per event loop (pools are loop-bound)."""

    def __init__(
        self, dsn: str, *, min_size: int = 1, max_size: int = 8, command_timeout: float = 30.0
    ) -> None:
        if not dsn:
            raise ValueError("PostgresPool needs a DSN")
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._command_timeout = command_timeout
        self._pools: LoopLocal[Any] = LoopLocal(self._open, self._shut)

    async def _open(self) -> Any:
        asyncpg: Any = importlib.import_module("asyncpg")
        return await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            command_timeout=self._command_timeout,
        )

    @staticmethod
    async def _shut(pool: Any) -> None:
        await pool.close()

    @asynccontextmanager
    async def _acquire(self) -> AsyncGenerator[SqlConnProtocol, None]:
        pool = await self._pools.get()
        async with pool.acquire() as conn:
            yield cast(SqlConnProtocol, conn)

    def acquire(self) -> AbstractAsyncContextManager[SqlConnProtocol]:
        """Borrow a connection from this event loop's pool."""
        return self._acquire()

    async def close(self) -> None:
        """Close the pools of every event loop."""
        await self._pools.aclose()


async def bootstrap_schema(pool: SqlPoolProtocol) -> None:
    """Create every cluster table that does not exist yet (idempotent)."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(CLUSTER_SCHEMA_SQL)
