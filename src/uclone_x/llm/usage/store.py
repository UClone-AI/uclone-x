"""Where paid-model usage is recorded: one row per call, shared by every process.

See the token-gateway design §4.1. The store only appends rows and returns the
rows recorded since a moment. Deciding whether a limit is reached is `limits.check`'s job,
so a store has no notion of a window or a limit.

`SqliteUsageStore` is the default: a file beside `settings.json`, so the dashboard, `ucx
run`, ACP and an eval running at the same time count against one total, and a restart
keeps the count. A JSON file was rejected because concurrent appends to a rewritten file
lose rows; SQLite is in the standard library and serialises writers itself.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

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

#: How long rows are kept: the longest window (7 days) and a day to spare.
USAGE_RETENTION = timedelta(days=8)


class UsageEntry(BaseModel):
    """One paid call's tokens, input and output together."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    at: datetime = Field(description="When the call finished, timezone-aware.")
    tokens: int = Field(ge=0)
    provider: str
    model: str | None = None
    count_source: TokenCountSource = TokenCountSource.PROVIDER


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
    count_source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_at ON usage (at);
"""


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
            with conn:
                conn.executescript(_SCHEMA)
                conn.execute("DELETE FROM usage WHERE at < ?", (cutoff.timestamp(),))
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
                    "INSERT INTO usage (at, tokens, provider, model, count_source) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        entry.at.timestamp(),
                        entry.tokens,
                        entry.provider,
                        entry.model,
                        entry.count_source.value,
                    ),
                )
        finally:
            conn.close()

    def entries_since(self, since: datetime) -> Sequence[UsageEntry]:
        conn = self._connect()
        try:
            rows: list[tuple[float, int, str, str | None, str]] = conn.execute(
                "SELECT at, tokens, provider, model, count_source FROM usage "
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
            )
            for at, tokens, provider, model, source in rows
        ]
