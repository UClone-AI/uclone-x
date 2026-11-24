"""Distributed session store: PostgreSQL revision OCC first, then a revision-monotonic Valkey head."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

from uclone_x.adapters.cluster.config import ClusterCachePolicy
from uclone_x.core.cluster_protocols import (
    VERSIONED_HEAD_SET_SCRIPT,
    CacheClientProtocol,
    ClusterSessionStoreProtocol,
    SqlPoolProtocol,
    TokenBudgetSnapshot,
    decode_versioned_head,
)
from uclone_x.core.loop_local import run_on_bridge_loop
from uclone_x.core.secrets import serialize_session_event
from uclone_x.core.session_log import with_image_data
from uclone_x.core.session_state import SessionState, content_digest
from uclone_x.errors import StaleSessionWriteError
from uclone_x.llm.models import ChatMessage

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

__all__ = ["ValkeyPostgresSessionStore"]

_T = TypeVar("_T")


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RevisionCarrier:
    revision: int


@dataclass(slots=True)
class _PrefetchedTurnBuffer:
    state: SessionState
    blobs: dict[str, str]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ValkeyPostgresSessionStore(ClusterSessionStoreProtocol):
    """Two-tier session store: PostgreSQL is the source of truth, Valkey a revisioned cache.

    - `save` commits PostgreSQL first with an atomic `UPDATE ... WHERE revision = $expected`
      (or `INSERT ... ON CONFLICT DO NOTHING` for revision 0); zero rows is a
      `StaleSessionWriteError`. Only then is the cache head written, through
      `VERSIONED_HEAD_SET_SCRIPT`, which never replaces a newer cached revision (Invariant 1).
    - `prefetch_session` reads the head and every referenced blob (`#1848`) in one `MGET` per
      session (all of a session's keys share the `{session_id}` hash tag), falling back to
      PostgreSQL for whatever the cache lost.
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
        self._local_turn_cache: dict[str, _PrefetchedTurnBuffer] = {}
        self._known_digests: dict[str, list[str]] = {}
        self.multi_get_calls: int = 0

    def _run_sync(self, fn: Callable[..., Coroutine[Any, Any, _T]], *args: Any) -> _T:
        if self._portal is not None:
            return self._portal.call(fn, *args)
        return run_on_bridge_loop(fn(*args))

    @staticmethod
    def _head_key(session_id: str) -> bytes:
        return f"sess:head:{{{session_id}}}".encode()

    @staticmethod
    def _blob_key(session_id: str, digest: str) -> bytes:
        return f"blob:{{{session_id}}}:{digest}".encode()

    @staticmethod
    def _budget_key(session_id: str) -> bytes:
        return f"sess:budget:{{{session_id}}}".encode()

    async def _cache_head(self, session_id: str, revision: int, state_json: str) -> bool:
        written = await self._cache.run_script(
            VERSIONED_HEAD_SET_SCRIPT,
            [self._head_key(session_id)],
            [revision, state_json, self._policy.head_ttl_seconds],
        )
        return bool(written)

    def _collect_digests(self, session_id: str, state: SessionState) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for entry in state.session_log:
            seen.setdefault(entry.digest, None)
        for snapshot in state.context_snapshots:
            for digest in (
                snapshot.tools_digest,
                snapshot.identity_digest,
                snapshot.slow_context_digest,
                snapshot.turn_context_digest,
            ):
                seen.setdefault(digest, None)
        for msg in state.messages:
            for part in msg.images:
                seen.setdefault(part.digest, None)
        for digest in self._known_digests.get(session_id, ()):
            seen.setdefault(digest, None)
        return tuple(seen)

    def _hydrate_messages_from_blobs(
        self, session_id: str, state: SessionState, blobs: dict[str, str]
    ) -> SessionState:
        if not state.history_entries or state.messages:
            return state
        by_id = {entry.id: entry.digest for entry in state.session_log}
        reconstructed: list[ChatMessage] = []
        for entry_id in state.history_entries:
            digest = by_id.get(entry_id)
            if digest is None:
                return state
            body = blobs.get(digest)
            if body is None:
                body = self.load_context_body(session_id, digest)
            if body is None:
                return state
            message = ChatMessage.model_validate_json(body)
            reconstructed.append(
                with_image_data(
                    message,
                    lambda img_digest: (
                        blobs.get(img_digest) or self.load_context_body(session_id, img_digest)
                    ),
                )
            )
        return state.model_copy(update={"messages": tuple(reconstructed)})

    async def _cached_head(self, session_id: str, raw: bytes) -> SessionState | None:
        """The cached head, or `None` after dropping one that is not this session's record."""
        try:
            _, state_json = decode_versioned_head(raw)
            state = SessionState.model_validate_json(state_json)
            if state.session_id == session_id:
                return state
        except (ValueError, RecursionError):
            pass
        logger.warning("Dropping an unreadable cached head of %s", session_id)
        await self._cache.delete(self._head_key(session_id))
        return None

    async def _cached_body(self, session_id: str, digest: str, raw: bytes) -> str | None:
        """The cached body, or `None` after dropping one that is not what its digest names."""
        try:
            body = raw.decode("utf-8")
            if content_digest(body) == digest:
                return body
        except UnicodeDecodeError:
            pass
        logger.warning("Dropping an unreadable cached body of %s", session_id)
        await self._cache.delete(self._blob_key(session_id, digest))
        return None

    async def _cached_budget(self, session_id: str, raw: bytes) -> TokenBudgetSnapshot | None:
        """The cached budget, or `None` after dropping one that does not read."""
        try:
            _, body = decode_versioned_head(raw)
            return TokenBudgetSnapshot.model_validate_json(body)
        except (ValueError, RecursionError):
            logger.warning("Dropping an unreadable cached budget of %s", session_id)
            await self._cache.delete(self._budget_key(session_id))
            return None

    async def prefetch_session(self, session_id: str) -> SessionState | None:
        """Two-phase prefetch: the head, then every referenced blob in one `MGET` (#1848)."""
        raw_head = await self._cache.get(self._head_key(session_id))
        state = await self._cached_head(session_id, raw_head) if raw_head is not None else None
        if state is None:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT revision AS committed_rev, state_json FROM session_snapshots WHERE session_id = $1",
                    session_id,
                )
                if row is None or not row.get("state_json"):
                    return None
                state_json = str(row["state_json"])
                state = SessionState.model_validate_json(state_json)
                blob_rows = await conn.fetch(
                    "SELECT digest FROM session_context_blobs WHERE session_id = $1",
                    session_id,
                )
            known = self._known_digests.setdefault(session_id, [])
            for b_row in blob_rows:
                digest_val = str(b_row["digest"])
                if digest_val not in known:
                    known.append(digest_val)
            # Read-through populate: never replaces a newer head another worker cached.
            await self._cache_head(session_id, int(row["committed_rev"]), state_json)

        digests = self._collect_digests(session_id, state)
        blobs: dict[str, str] = {}
        if digests:
            cached_blobs = await self._cache.multi_get(
                *(self._blob_key(session_id, d) for d in digests)
            )
            self.multi_get_calls += 1
            missing_digests: list[str] = []
            for digest, raw_blob in zip(digests, cached_blobs, strict=True):
                body = (
                    await self._cached_body(session_id, digest, raw_blob)
                    if raw_blob is not None
                    else None
                )
                if body is not None:
                    blobs[digest] = body
                else:
                    missing_digests.append(digest)

            if missing_digests:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT digest, body FROM session_context_blobs "
                        "WHERE session_id = $1 AND digest = ANY($2::text[])",
                        session_id,
                        missing_digests,
                    )
                for b_row in rows:
                    d_str = str(b_row["digest"])
                    body_str = str(b_row["body"])
                    blobs[d_str] = body_str
                    await self._cache.add_cache(
                        self._blob_key(session_id, d_str),
                        body_str.encode("utf-8"),
                        ttl_seconds=self._policy.blob_ttl_seconds,
                    )

        hydrated_state = self._hydrate_messages_from_blobs(session_id, state, blobs)
        self._local_turn_cache[session_id] = _PrefetchedTurnBuffer(
            state=hydrated_state, blobs=blobs
        )
        return hydrated_state

    def evict_local_turn_cache(self, session_id: str) -> None:
        """Discard everything this worker kept of the session (zero idle state, §1)."""
        self._local_turn_cache.pop(session_id, None)
        self._known_digests.pop(session_id, None)

    async def invalidate_head(self, session_id: str) -> None:
        """Drop the cached head: a failover can leave one older than PostgreSQL's record."""
        await self._cache.delete(self._head_key(session_id))

    def load(self, session_id: str) -> SessionState | None:
        """Return the stored session from worker-local turn buffer, Valkey, or PostgreSQL."""
        buffered = self._local_turn_cache.get(session_id)
        if buffered is not None:
            return buffered.state
        return self._run_sync(self.prefetch_session, session_id)

    async def _stale(
        self, conn: Any, session_id: str, expected_revision: int, no_record: SessionState
    ) -> StaleSessionWriteError:
        row = await conn.fetchrow(
            "SELECT revision AS committed_rev, state_json FROM session_snapshots WHERE session_id = $1",
            session_id,
        )
        actual = int(row["committed_rev"]) if row is not None else 0
        current = (
            SessionState.model_validate_json(str(row["state_json"]))
            if row is not None
            else no_record
        )
        return StaleSessionWriteError(
            f"Refusing to persist session {session_id!r}: expected revision "
            f"{expected_revision}, actual revision {actual}",
            session_id=session_id,
            expected_revision=expected_revision,
            actual_revision=actual,
            current=current,
        )

    async def settled_terminal(self, session_id: str, turn_id: str) -> Mapping[str, Any] | None:
        """The terminal event in `turn_id`'s receipt: read from PostgreSQL, never from the cache.

        `None` means no receipt.

        A table rather than `session_snapshots.last_turn_id`, which the design named: offsets
        complete out of order across sessions, so turns A then B of one session can both be
        committed behind an unfinished offset of another. A crash redelivers A and B, and A
        is not the last turn any more.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT terminal_json FROM session_turn_receipts "
                "WHERE session_id = $1 AND turn_id = $2",
                session_id,
                turn_id,
            )
        return json.loads(str(row["terminal_json"])) if row is not None else None

    async def settle_turn(self, session_id: str, turn_id: str, terminal: Mapping[str, Any]) -> None:
        """Write the receipt of a turn ending with no commit; a receipt already there stands."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO session_turn_receipts (session_id, turn_id, outcome, "
                    "terminal_json) VALUES ($1, $2, $3, $4) "
                    "ON CONFLICT (session_id, turn_id) DO NOTHING",
                    session_id,
                    turn_id,
                    str(terminal["event"]),
                    json.dumps(terminal, default=str),
                )
                await self._forget_attempts(conn, session_id, turn_id)

    @staticmethod
    async def _forget_attempts(conn: Any, session_id: str, turn_id: str) -> None:
        await conn.execute(
            "DELETE FROM turn_delivery_attempts WHERE session_id = $1 AND turn_id = $2",
            session_id,
            turn_id,
        )

    async def record_turn_start(self, session_id: str, turn_id: str) -> int:
        """Count one start of this delivery, in PostgreSQL, so it survives the worker dying."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO turn_delivery_attempts (session_id, turn_id, starts) "
                "VALUES ($1, $2, 1) ON CONFLICT (session_id, turn_id) "
                "DO UPDATE SET starts = turn_delivery_attempts.starts + 1 "
                "RETURNING starts AS counted",
                session_id,
                turn_id,
            )
        return int(row["counted"]) if row is not None else 0

    async def refund_turn_start(self, session_id: str, turn_id: str) -> None:
        """Take back a start that came to an end, so only a process that died keeps one."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE turn_delivery_attempts SET starts = GREATEST(starts - 1, 0) "
                "WHERE session_id = $1 AND turn_id = $2",
                session_id,
                turn_id,
            )

    async def _save_async(
        self,
        state: SessionState,
        pending_events: Sequence[Any] | None = None,
        turn_id: str | None = None,
        budget: TokenBudgetSnapshot | None = None,
        terminal: Mapping[str, Any] | None = None,
    ) -> SessionState:
        session_id = state.session_id

        carrier = _RevisionCarrier(state.revision + 1)
        stamped = state.model_copy(
            update={
                "updated_at": _now_iso(),
                "revision": carrier.revision,
            }
        )
        payload_str = json.dumps(stamped.model_dump(mode="json"))
        validated = SessionState.model_validate_json(payload_str)
        if stamped.messages and not validated.messages:
            validated = validated.model_copy(
                update={
                    "messages": tuple(
                        partial(
                            with_image_data,
                            load=partial(self.load_context_body, session_id),
                        )(m)
                        for m in stamped.messages
                    )
                }
            )
        if pending_events:
            for evt in pending_events:
                serialize_session_event(evt, offset=0)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # The OCC check is the statement itself: concurrent writers that loaded the
                # same revision serialize on the row lock, and all but the first match no row.
                if state.revision == 0:
                    status = await conn.execute(
                        "INSERT INTO session_snapshots "
                        "(session_id, agent_id, revision, state_json, updated_at) "
                        "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (session_id) DO NOTHING",
                        session_id,
                        validated.agent_id,
                        validated.revision,
                        payload_str,
                        validated.updated_at,
                    )
                else:
                    status = await conn.execute(
                        "UPDATE session_snapshots SET agent_id = $2, revision = $3, "
                        "state_json = $4, updated_at = $5 "
                        "WHERE session_id = $1 AND revision = $6",
                        session_id,
                        validated.agent_id,
                        validated.revision,
                        payload_str,
                        validated.updated_at,
                        state.revision,
                    )
                if not status.endswith(" 1"):
                    raise await self._stale(conn, session_id, state.revision, state)

                if turn_id is not None:
                    receipt = await conn.execute(
                        "INSERT INTO session_turn_receipts (session_id, turn_id, outcome, "
                        "revision, terminal_json) VALUES ($1, $2, 'committed', $3, $4) "
                        "ON CONFLICT (session_id, turn_id) DO NOTHING",
                        session_id,
                        turn_id,
                        validated.revision,
                        json.dumps(
                            {**(terminal or {}), "revision": validated.revision}, default=str
                        ),
                    )
                    if not receipt.endswith(" 1"):
                        # Another worker already ended this turn. Rolled back with the rest;
                        # the retry finds the receipt.
                        raise await self._stale(conn, session_id, state.revision, state)
                    await self._forget_attempts(conn, session_id, turn_id)
                budget_json = budget.model_dump_json() if budget is not None else None
                if budget_json is not None:
                    await conn.execute(
                        "UPDATE session_snapshots SET budget_json = $2 WHERE session_id = $1",
                        session_id,
                        budget_json,
                    )

                if pending_events:
                    next_seq_row = await conn.fetchrow(
                        "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq FROM session_events "
                        "WHERE session_id = $1",
                        session_id,
                    )
                    next_seq = int(next_seq_row["next_seq"]) if next_seq_row is not None else 0
                    for idx, evt in enumerate(pending_events):
                        await conn.execute(
                            "INSERT INTO session_events (session_id, seq, event_json) "
                            "VALUES ($1, $2, $3)",
                            session_id,
                            next_seq + idx,
                            serialize_session_event(evt, offset=next_seq + idx),
                        )

        # PostgreSQL committed; the cache follows and can only move forward.
        await self._cache_head(session_id, validated.revision, payload_str)
        if budget_json is not None:
            await self._cache_budget(session_id, validated.revision, budget_json)

        if session_id in self._local_turn_cache:
            self._local_turn_cache[session_id].state = validated

        return validated

    def save(
        self, state: SessionState, pending_events: Sequence[Any] | None = None
    ) -> SessionState:
        """Persist `state` with PostgreSQL OCC, then advance the cached head."""
        return self._run_sync(self._save_async, state, pending_events)

    def commit_turn(
        self,
        state: SessionState,
        *,
        turn_id: str,
        terminal: Mapping[str, Any],
        budget: TokenBudgetSnapshot | None = None,
        pending_events: Sequence[Any] | None = None,
    ) -> SessionState:
        """Save `state` with the turn's receipt, terminal event and budget, in one OCC transaction."""
        return self._run_sync(self._save_async, state, pending_events, turn_id, budget, terminal)

    async def _delete_async(self, session_id: str) -> bool:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                status = await conn.execute(
                    "DELETE FROM session_snapshots WHERE session_id = $1",
                    session_id,
                )
                await conn.execute(
                    "DELETE FROM session_events WHERE session_id = $1",
                    session_id,
                )
                await conn.execute(
                    "DELETE FROM session_context_blobs WHERE session_id = $1",
                    session_id,
                )
                await conn.execute(
                    "DELETE FROM session_turn_receipts WHERE session_id = $1",
                    session_id,
                )
                await conn.execute(
                    "DELETE FROM turn_delivery_attempts WHERE session_id = $1",
                    session_id,
                )
        await self._cache.delete(self._head_key(session_id), self._budget_key(session_id))
        self._local_turn_cache.pop(session_id, None)
        self._known_digests.pop(session_id, None)
        return status == "DELETE 1"

    def delete(self, session_id: str) -> bool:
        """Remove the session from PostgreSQL and Valkey."""
        return self._run_sync(self._delete_async, session_id)

    async def _list_session_ids_async(self) -> tuple[str, ...]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT session_id FROM session_snapshots ORDER BY session_id")
        return tuple(str(r["session_id"]) for r in rows)

    def list_session_ids(self) -> tuple[str, ...]:
        """Return all stored session IDs in sorted order."""
        return self._run_sync(self._list_session_ids_async)

    async def _clear_event_log_async(self, session_id: str) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM session_events WHERE session_id = $1",
                    session_id,
                )
                await conn.execute(
                    "DELETE FROM session_context_blobs WHERE session_id = $1",
                    session_id,
                )
        digests = self._known_digests.pop(session_id, [])
        if digests:
            await self._cache.delete(*(self._blob_key(session_id, d) for d in digests))
        if session_id in self._local_turn_cache:
            self._local_turn_cache[session_id].blobs.clear()

    def clear_event_log(self, session_id: str) -> None:
        """Remove durable turn events and context bodies for `session_id`."""
        self._run_sync(self._clear_event_log_async, session_id)

    async def _save_context_body_async(self, session_id: str, digest: str, body: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO session_context_blobs (session_id, digest, body) VALUES ($1, $2, $3) "
                "ON CONFLICT (session_id, digest) DO NOTHING",
                session_id,
                digest,
                body,
            )
        await self._cache.add_cache(
            self._blob_key(session_id, digest),
            body.encode("utf-8"),
            ttl_seconds=self._policy.blob_ttl_seconds,
        )
        known = self._known_digests.setdefault(session_id, [])
        if digest not in known:
            known.append(digest)
        if session_id in self._local_turn_cache:
            self._local_turn_cache[session_id].blobs[digest] = body

    def save_context_body(self, session_id: str, digest: str, body: str) -> None:
        """Store a content-addressed context body under `(session_id, digest)`."""
        self._run_sync(self._save_context_body_async, session_id, digest, body)

    async def _load_context_body_async(self, session_id: str, digest: str) -> str | None:
        blob_key = self._blob_key(session_id, digest)
        raw = await self._cache.get(blob_key)
        cached = await self._cached_body(session_id, digest, raw) if raw is not None else None
        if cached is not None:
            return cached
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT body FROM session_context_blobs WHERE session_id = $1 AND digest = $2",
                session_id,
                digest,
            )
        if row is None or row.get("body") is None:
            return None
        body_str = str(row["body"])
        await self._cache.add_cache(
            blob_key, body_str.encode("utf-8"), ttl_seconds=self._policy.blob_ttl_seconds
        )
        return body_str

    def load_context_body(self, session_id: str, digest: str) -> str | None:
        """Return the context body for `(session_id, digest)` from local buffer, Valkey, or PG."""
        buffered = self._local_turn_cache.get(session_id)
        if buffered is not None and digest in buffered.blobs:
            return buffered.blobs[digest]
        body = self._run_sync(self._load_context_body_async, session_id, digest)
        if body is not None and buffered is not None:
            buffered.blobs[digest] = body
        return body

    async def _cache_budget(self, session_id: str, revision: int, budget_json: str) -> None:
        """Cache the snapshot the same way as the head: an older revision never replaces it."""
        await self._cache.run_script(
            VERSIONED_HEAD_SET_SCRIPT,
            [self._budget_key(session_id)],
            [revision, budget_json, self._policy.head_ttl_seconds],
        )

    async def _save_budget_snapshot_async(
        self, session_id: str, snapshot: TokenBudgetSnapshot, expected_revision: int
    ) -> None:
        budget_json = snapshot.model_dump_json()
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                "UPDATE session_snapshots SET budget_json = $2 "
                "WHERE session_id = $1 AND revision = $3",
                session_id,
                budget_json,
                expected_revision,
            )
            if not status.endswith(" 1"):
                raise await self._stale(
                    conn,
                    session_id,
                    expected_revision,
                    SessionState(session_id=session_id, agent_id=""),
                )
        await self._cache_budget(session_id, expected_revision, budget_json)

    def save_budget_snapshot(
        self, session_id: str, snapshot: TokenBudgetSnapshot, *, expected_revision: int
    ) -> None:
        """Persist the snapshot iff the session record is still at `expected_revision`."""
        self._run_sync(self._save_budget_snapshot_async, session_id, snapshot, expected_revision)

    async def _load_budget_snapshot_async(self, session_id: str) -> TokenBudgetSnapshot | None:
        budget_key = self._budget_key(session_id)
        raw = await self._cache.get(budget_key)
        cached = await self._cached_budget(session_id, raw) if raw is not None else None
        if cached is not None:
            return cached
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT revision AS committed_rev, budget_json FROM session_snapshots "
                "WHERE session_id = $1",
                session_id,
            )
        if row is None or not row.get("budget_json"):
            return None
        budget_str = str(row["budget_json"])
        await self._cache_budget(session_id, int(row["committed_rev"]), budget_str)
        return TokenBudgetSnapshot.model_validate_json(budget_str)

    def load_budget_snapshot(self, session_id: str) -> TokenBudgetSnapshot | None:
        """Load the stored `TokenBudgetSnapshot` for `session_id`, if any."""
        return self._run_sync(self._load_budget_snapshot_async, session_id)

    def take_set_aside(self, session_id: str) -> bool:
        """Satisfy `SeatSessionSetAsideProtocol`; cluster records are versioned in PostgreSQL."""
        _ = session_id
        return False

    def keeps_copy_of(self, session_id: str) -> bool:
        """Satisfy `SessionStore` kept-copy checks; cluster records do not use local set-aside files."""
        _ = session_id
        return False

    def event_log_path(self, session_id: str) -> None:
        """Cluster session events are stored in PostgreSQL `session_events`, not a local file path."""
        _ = session_id
        return None

    async def _read_event_log_lines_async(self, session_id: str) -> tuple[str, ...] | None:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT event_json FROM session_events WHERE session_id = $1 ORDER BY seq ASC",
                session_id,
            )
        if not rows:
            return None
        header = json.dumps(
            {"schema": "uclone_x.log.v0", "version": "0.1.0-dev", "session_id": session_id}
        )
        return (header, *(str(r["event_json"]) for r in rows))

    def read_event_log_lines(self, session_id: str) -> tuple[str, ...] | None:
        """Return the session's stored event log lines from PostgreSQL `session_events`, if any."""
        return self._run_sync(self._read_event_log_lines_async, session_id)
