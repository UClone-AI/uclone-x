"""Where paid-model usage is recorded: one row per call, shared by every process.

See the token-gateway design §4.1. The store only appends rows and returns the
rows recorded since a moment. Deciding whether a limit is reached is `limits.check`'s job,
so a store has no notion of a window or a limit.

`SqliteUsageStore` is the default: a file beside `settings.json`, so the dashboard, `ucx
run`, ACP and an eval running at the same time count against one total, and a restart
keeps the count. A JSON file was rejected because concurrent appends to a rewritten file
lose rows; SQLite is in the standard library and serialises writers itself.

Each row also keeps, apart, how many of its input tokens the provider served from its prompt
cache (#2429): an agent loop resends its whole conversation on every call, and in one measured
session 94% of the input was a cached prefix. `tokens` stays the whole count.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from uclone_x.core.usage_entry import USAGE_RETENTION, UsageEntry
from uclone_x.llm.models import TokenCountSource

__all__ = [
    "USAGE_FILE_NAME",
    "USAGE_RETENTION",
    "MemoryUsageStore",
    "SqliteUsageStore",
    "UsageEntry",
    "UsageStore",
    "default_usage_file",
]

#: The store's file name, in the session root beside `settings.json`.
USAGE_FILE_NAME = "usage.sqlite3"


class UsageStore(Protocol):
    """Append usage rows, and return the rows recorded since a moment."""

    def add(self, entry: UsageEntry) -> None: ...

    def entries_since(self, since: datetime) -> Sequence[UsageEntry]:
        """Rows with `at` after `since`, oldest first."""
        ...


def default_usage_file() -> Path:
    """`<session root>/usage.sqlite3`, beside `settings.json`.

    Resolved through `uclone_x.core.session`, and imported here rather than at module
    level, for the reason `saved_choice.settings_file` gives.
    """
    from uclone_x.core.session import default_session_root

    return default_session_root() / USAGE_FILE_NAME


class MemoryUsageStore:
    """A store held in this process only, for tests."""

    def __init__(self) -> None:
        self._entries: list[UsageEntry] = []
        self._lock = threading.Lock()

    def add(self, entry: UsageEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def entries_since(self, since: datetime) -> Sequence[UsageEntry]:
        with self._lock:
            return sorted((e for e in self._entries if e.at > since), key=lambda e: e.at)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    at REAL NOT NULL,
    tokens INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    count_source TEXT NOT NULL,
    cached_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS usage_at ON usage (at);
"""


def _add_cache_column(conn: sqlite3.Connection) -> None:
    """Give a file written before #2429 its `cached_tokens` column; its rows stay `NULL`.

    The store had no schema version, so the column's presence is the version. `NULL` is the
    honest value for an old row: the cached share was never recorded, so it reads as all new.
    Run inside the caller's write transaction, so two processes opening one old file at once
    cannot both add the column.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(usage)")}
    if "cached_tokens" not in columns:
        conn.execute("ALTER TABLE usage ADD COLUMN cached_tokens INTEGER")


class SqliteUsageStore:
    """The default store: a SQLite file every process on the machine shares.

    A connection is opened per operation rather than held, so the store is safe to share
    across threads and event loops, and holds no file open between calls. Opening a local
    SQLite file costs far less than the network round trip each call accompanies.

    Rows older than `USAGE_RETENTION` are removed when the store is first opened.
    """

    def __init__(self, path: Path, *, now: datetime | None = None) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        cutoff = (now or datetime.now(UTC)) - USAGE_RETENTION
        conn = self._connect()
        try:
            # `executescript` commits first, so the write lock is taken explicitly: the
            # column check and the `ALTER` must be one step across processes.
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        conn.execute(statement)
                _add_cache_column(conn)
                conn.execute("DELETE FROM usage WHERE at < ?", (cutoff.timestamp(),))
            except BaseException:
                conn.rollback()
                raise
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        # `timeout` waits for another process's write lock instead of failing at once.
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def add(self, entry: UsageEntry) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO usage (at, tokens, provider, model, count_source, cached_tokens) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        entry.at.timestamp(),
                        entry.tokens,
                        entry.provider,
                        entry.model,
                        entry.count_source.value,
                        entry.cached_tokens,
                    ),
                )
        finally:
            conn.close()

    def entries_since(self, since: datetime) -> Sequence[UsageEntry]:
        conn = self._connect()
        try:
            rows: list[tuple[float, int, str, str | None, str, int | None]] = conn.execute(
                "SELECT at, tokens, provider, model, count_source, cached_tokens FROM usage "
                "WHERE at > ? ORDER BY at",
                (since.timestamp(),),
            ).fetchall()
        finally:
            conn.close()
        return [
            UsageEntry(
                at=datetime.fromtimestamp(at, UTC),
                tokens=tokens,
                provider=provider,
                model=model,
                count_source=TokenCountSource(source),
                cached_tokens=cached,
            )
            for at, tokens, provider, model, source, cached in rows
        ]
