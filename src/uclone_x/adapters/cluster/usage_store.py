"""Cluster-wide paid-token windows in Valkey, and the PostgreSQL `usage_ledger`.

`TurnUsageStore` is what backs `UsageGate` on a worker: it implements the gate's own
`UsageStore` protocol over one turn, so the limit logic (`llm/usage/limits.py`, `gate.py`, `admission.py`)
runs unchanged and only where usage is kept differs.

**The ledger is the record.** Every completed model call inserts one row, first, so a Valkey
failure costs a window increment and never a row. Calls on a model that costs nothing are
ledger rows too, with `counts_against_limits` false, and never reach a window.

**The window is what the check reads.** One hash per scope, `usage:win:{scope}:<bucket_seconds>`,
holding `<bucket> -> tokens` and `c<bucket> -> cached tokens`, with finished buckets older than
`USAGE_RETENTION` pruned by the script that writes them. It carries no TTL, so `volatile-lru`
never evicts it. A bucket is read back as one `UsageEntry` stamped with the bucket's end, so a
limit lifts up to one bucket later than it would on exact rows, and never earlier.

**A lost window reads through.** Replication is asynchronous: a failover can lose the last
increments of a hash, and those stay lost until they age out (the check reads slightly low, the
ledger stays exact). A hash that is gone or unreachable reads from the ledger instead and, when
it is gone, is rebuilt from it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from uclone_x.core.cluster_protocols import (
    CacheClientProtocol,
    DistributedUsageStoreProtocol,
    SqlPoolProtocol,
)
from uclone_x.core.loop_local import run_on_bridge_loop
from uclone_x.core.usage_entry import USAGE_RETENTION, UsageEntry
from uclone_x.llm.models import TokenCountSource

__all__ = ["DistributedUsageStore", "TurnUsageStore"]

logger = logging.getLogger(__name__)

#: Width of one window bucket. The shortest limit is 10 minutes, so 60 s reads it to a tenth.
BUCKET_SECONDS: Final[int] = 60

#: KEYS: usage hash. ARGV: bucket, tokens, cached tokens, oldest bucket to keep. Adds to the
#: bucket, drops every field older than the oldest kept, returns the bucket's new total.
RECORD_SCRIPT: Final[str] = """
local total = redis.call('HINCRBY', KEYS[1], ARGV[1], ARGV[2])
if tonumber(ARGV[3]) > 0 then
  redis.call('HINCRBY', KEYS[1], 'c' .. ARGV[1], ARGV[3])
end
local oldest = tonumber(ARGV[4])
for _, field in ipairs(redis.call('HKEYS', KEYS[1])) do
  if tonumber((string.gsub(field, '^c', ''))) < oldest then
    redis.call('HDEL', KEYS[1], field)
  end
end
return total
"""

#: KEYS: usage hash. ARGV: field, value, field, value, ... Writes only fields not there yet, so
#: a rebuild racing a new call never replaces what that call just added.
REBUILD_SCRIPT: Final[str] = """
for i = 1, #ARGV, 2 do
  redis.call('HSETNX', KEYS[1], ARGV[i], ARGV[i + 1])
end
return 1
"""


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class DistributedUsageStore(DistributedUsageStoreProtocol):
    """Atomic Valkey window counters and the durable PostgreSQL usage ledger."""

    def __init__(
        self,
        cache: CacheClientProtocol,
        pool: SqlPoolProtocol,
        *,
        clock: Callable[[], float] | None = None,
        bucket_seconds: int = BUCKET_SECONDS,
    ) -> None:
        if bucket_seconds < 1:
            raise ValueError(f"bucket_seconds must be >= 1, got {bucket_seconds}")
        self._cache = cache
        self._pool = pool
        self._clock: Callable[[], float] = clock or time.time
        self._bucket_seconds = bucket_seconds

    def window_key(self, scope: str) -> bytes:
        return f"usage:win:{{{scope}}}:{self._bucket_seconds}".encode()

    async def record_completed_call(
        self,
        *,
        scope: str,
        session_id: str,
        turn_id: str,
        entry: UsageEntry,
        counts_against_limits: bool,
    ) -> None:
        """Insert the ledger row, then add the call to `scope`'s window when it counts."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO usage_ledger (scope, session_id, turn_id, provider, model, tokens, "
                "prompt_tokens, completion_tokens, cached_tokens, count_source, "
                "counts_against_limits, recorded_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
                scope,
                session_id,
                turn_id,
                entry.provider,
                entry.model,
                entry.tokens,
                entry.input_tokens,
                entry.output_tokens,
                entry.cached_tokens,
                entry.count_source.value,
                counts_against_limits,
                entry.at,
            )
        if not counts_against_limits:
            return
        bucket = int(entry.at.timestamp() // self._bucket_seconds)
        await self._cache.run_script(
            RECORD_SCRIPT,
            [self.window_key(scope)],
            [
                bucket,
                entry.tokens,
                entry.cached_tokens or 0,
                self._oldest_kept_bucket(),
            ],
        )

    def _oldest_kept_bucket(self) -> int:
        return int((self._clock() - USAGE_RETENTION.total_seconds()) // self._bucket_seconds)

    def _bucket_entries(self, fields: Mapping[str, int], since: datetime) -> list[UsageEntry]:
        width = self._bucket_seconds
        entries: list[UsageEntry] = []
        for field, tokens in fields.items():
            if field.startswith("c"):
                continue
            ended = datetime.fromtimestamp((int(field) + 1) * width, UTC)
            if ended > since:
                entries.append(
                    UsageEntry(
                        at=ended,
                        tokens=tokens,
                        provider="cluster-window",
                        count_source=TokenCountSource.PROVIDER,
                        cached_tokens=fields.get(f"c{field}"),
                    )
                )
        return sorted(entries, key=lambda e: e.at)

    async def _read_window(self, scope: str) -> dict[str, int] | None:
        try:
            raw: Mapping[Any, Any] = await self._cache.hash_get_all(self.window_key(scope))
            return {_text(key): int(value) for key, value in raw.items()}
        except Exception:
            logger.warning(
                "The usage window of %r could not be read; using the ledger", scope, exc_info=True
            )
            return None

    async def _ledger_fields(self, scope: str, since: datetime) -> dict[str, int]:
        """`scope`'s ledger rows since `since`, summed into window fields."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT recorded_at, tokens, cached_tokens FROM usage_ledger "
                "WHERE scope = $1 AND recorded_at > $2 AND counts_against_limits",
                scope,
                since,
            )
        fields: dict[str, int] = {}
        for row in rows:
            at: datetime = row["recorded_at"]
            bucket = str(int(at.timestamp() // self._bucket_seconds))
            fields[bucket] = fields.get(bucket, 0) + int(row["tokens"])
            cached = int(row["cached_tokens"] or 0)
            if cached:
                fields[f"c{bucket}"] = fields.get(f"c{bucket}", 0) + cached
        return fields

    async def entries_since(self, scope: str, since: datetime) -> Sequence[UsageEntry]:
        """The window's buckets ending after `since`, oldest first; see the module docstring."""
        fields = await self._read_window(scope)
        if fields:
            return self._bucket_entries(fields, since)
        # Gone (a failover, a new scope) or unreachable: the ledger answers.
        floor = datetime.fromtimestamp(self._clock(), UTC) - USAGE_RETENTION
        ledger = await self._ledger_fields(scope, max(since - timedelta(seconds=1), floor))
        if fields is not None and ledger:
            await self._cache.run_script(
                REBUILD_SCRIPT,
                [self.window_key(scope)],
                [part for field, value in ledger.items() for part in (field, value)],
            )
        return self._bucket_entries(ledger, since)

    async def list_ledger_entries(self, session_id: str) -> tuple[Mapping[str, Any], ...]:
        """The ledger rows of `session_id`, oldest first."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM usage_ledger WHERE session_id = $1 ORDER BY id",
                session_id,
            )
        return tuple(rows)


@dataclass(frozen=True)
class TurnUsageStore:
    """The gate's synchronous `UsageStore`, over one turn's scope, session and id."""

    usage: DistributedUsageStoreProtocol
    scope: str
    session_id: str
    turn_id: str
    counts_against_limits: bool = True

    def add(self, entry: UsageEntry) -> None:
        run_on_bridge_loop(
            self.usage.record_completed_call(
                scope=self.scope,
                session_id=self.session_id,
                turn_id=self.turn_id,
                entry=entry,
                counts_against_limits=self.counts_against_limits,
            )
        )

    def entries_since(self, since: datetime) -> Sequence[UsageEntry]:
        return run_on_bridge_loop(self.usage.entries_since(self.scope, since))
