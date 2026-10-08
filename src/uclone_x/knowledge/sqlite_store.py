"""A uGraph store in one SQLite file: the clone's vessel (design §3.1, §3.3, G8).

One file is one scope. `SqliteKnowledgeStore` opens a file for `clone:<agent id>` and
refuses a file made for another scope, so two vessels cannot share storage by accident (G2).

**One write is one transaction (G8).** Nothing is held between calls: every read and every
write opens its own connection, so there is no cached copy to go stale and no store object
to share between processes. A write is `BEGIN IMMEDIATE`, which takes the file's one write
lock up front; a second writer waits (`BUSY_TIMEOUT_SECONDS`) instead of failing, then
reads what the first one committed, so two processes writing alternately lose nothing. This
replaces `memory.json`'s read-compare-replace merge, which was a check and not a lock.

**Every write goes through the resolver (G11).** `KnowledgeTransaction.add_entity` takes an
`EntityRef`, which only `EntityResolver` makes, and `entity_for` is `resolve` followed by
`add_entity`. There is no method that inserts a raw name.

**A file that cannot be read is an error, not an empty store.** `KnowledgeUnreadableError`
is raised for a file SQLite says is not a database (or is malformed, or of a schema this
build does not know); the caller decides what to do with it, and this module never deletes
or recreates a file it cannot read.

Edges carry an opaque `record`: a JSON object the caller owns, for what the clone's fact
has that an `Edge` does not (its provenance, tags, session). The store keeps it and gives it
back; it reads none of it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import unicodedata
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from uclone_x.knowledge.models import (
    Axis,
    Edge,
    EdgeStatus,
    Entity,
    EntityRef,
    Episode,
    Interval,
    Position,
)
from uclone_x.knowledge.policy import CLONE_POLICY, ClonePolicy
from uclone_x.knowledge.resolver import EntityResolver, name_key
from uclone_x.knowledge.store import EdgeRow, edges_at, holds_at

__all__ = [
    "BUSY_TIMEOUT_SECONDS",
    "SCHEMA_VERSION",
    "EdgeRow",
    "KnowledgeTransaction",
    "KnowledgeUnreadableError",
    "ScopeMismatchError",
    "SqliteKnowledgeStore",
]

logger = logging.getLogger(__name__)

#: The file's schema. A file of another version is reported unreadable, never migrated in
#: place: there is no older format in the wild (owner 2026-09-28).
SCHEMA_VERSION = "1"
#: How long a writer waits for another writer's transaction before giving up. A transaction
#: here is a few statements, so a wait this long means a process died holding the lock.
BUSY_TIMEOUT_SECONDS = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]', summary TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, ref TEXT NOT NULL, at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT,
    value TEXT,
    axis TEXT NOT NULL,
    valid_start TEXT,
    valid_end TEXT,
    recorded_at TEXT NOT NULL,
    expired_at TEXT,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    origin TEXT NOT NULL,
    record TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS edges_subject ON edges (subject_id);
CREATE TABLE IF NOT EXISTS edge_evidence (
    edge_id TEXT NOT NULL, episode_id TEXT NOT NULL, quote TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_edge ON edge_evidence (edge_id);
"""


class KnowledgeUnreadableError(Exception):
    """The file is there and SQLite cannot use it. `path` and `cause` are for the log."""

    def __init__(self, message: str, *, path: Path | None, cause: str) -> None:
        super().__init__(message)
        self.path = path
        self.cause = cause


class ScopeMismatchError(ValueError):
    """A store was opened on a file made for another scope, or given an object of one (G2)."""


def _collapse(name: str) -> str:
    return " ".join(unicodedata.normalize("NFC", name).split())


def _edge_of(row: sqlite3.Row, scope: str, evidence: tuple[tuple[str, str], ...]) -> Edge:
    return Edge(
        id=row["id"],
        scope=scope,
        subject_id=row["subject_id"],
        predicate=row["predicate"],
        object_id=row["object_id"],
        value=row["value"],
        valid=Interval(axis=row["axis"], start=row["valid_start"], end=row["valid_end"]),
        recorded_at=row["recorded_at"],
        expired_at=row["expired_at"],
        status=row["status"],
        confidence=row["confidence"],
        origin=row["origin"],
        evidence=evidence,
    )


class _Reader:
    """The reads, over one connection. A store's read and a transaction's both are this."""

    def __init__(self, conn: sqlite3.Connection, scope: str, policy: ClonePolicy) -> None:
        self._conn = conn
        self.scope = scope
        self.policy = policy

    def entities(self) -> list[Entity]:
        rows = self._conn.execute("SELECT * FROM entities ORDER BY rowid").fetchall()
        return [
            Entity(
                id=r["id"],
                scope=self.scope,
                kind=r["kind"],
                name=r["name"],
                aliases=tuple(json.loads(r["aliases"])),
                summary=r["summary"],
            )
            for r in rows
        ]

    def meta(self, key: str) -> str | None:
        """A value the caller kept in the file's `meta` table, or `None`."""
        found = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if found is None else str(found["value"])

    def entity(self, entity_id: str) -> Entity | None:
        return next((e for e in self.entities() if e.id == entity_id), None)

    def resolve(self, name: str) -> EntityRef:
        """What `name` is in this scope: the resolver's stages 1-3, nothing written."""
        return EntityResolver(self.scope, self.entities()).resolve(name)

    def rows(self) -> list[EdgeRow]:
        """Every edge, oldest written first, with its record."""
        evidence: dict[str, list[tuple[str, str]]] = {}
        for ev in self._conn.execute("SELECT * FROM edge_evidence ORDER BY rowid"):
            evidence.setdefault(ev["edge_id"], []).append((ev["episode_id"], ev["quote"]))
        found = self._conn.execute("SELECT * FROM edges ORDER BY rowid").fetchall()
        return [
            EdgeRow(
                _edge_of(r, self.scope, tuple(evidence.get(r["id"], ()))),
                json.loads(r["record"]),
            )
            for r in found
        ]

    def row(self, edge_id: str) -> EdgeRow | None:
        found = self._conn.execute("SELECT * FROM edges WHERE id = ?", (edge_id,)).fetchone()
        if found is None:
            return None
        evidence = tuple(
            (ev["episode_id"], ev["quote"])
            for ev in self._conn.execute(
                "SELECT * FROM edge_evidence WHERE edge_id = ? ORDER BY rowid", (edge_id,)
            )
        )
        return EdgeRow(_edge_of(found, self.scope, evidence), json.loads(found["record"]))

    def edges(self) -> list[Edge]:
        return [row.edge for row in self.rows()]

    def statements_at(
        self, axis: Axis, point: Position, *, proposed: bool = False
    ) -> dict[str, tuple[str, str, str]]:
        """`edge id -> (subject name, predicate, object name or value)` of what holds at `point`.

        The same edges `facts_at` names (`knowledge.store.holds_at`: approved, or with
        `proposed` approved and proposed; unexpired; valid at `point` on `axis`), with the
        ids kept, so a reasoner's answer can say which facts it rests on. Names, not
        entity ids: a rule is written about `user`, not about an id minted for someone.
        """
        names = {e.id: e.name for e in self.entities()}
        held: dict[str, tuple[str, str, str]] = {}
        for edge in self.edges():
            if not holds_at(edge, axis, point, proposed=proposed):
                continue
            target = (
                names.get(edge.object_id, edge.object_id)
                if edge.object_id is not None
                else str(edge.value)
            )
            held[edge.id] = (names.get(edge.subject_id, edge.subject_id), edge.predicate, target)
        return held

    def edges_at(self, axis: Axis, point: Position, *, proposed: bool = False) -> list[Edge]:
        return edges_at(self.edges(), axis, point, proposed=proposed)

    def facts_at(
        self, axis: Axis, point: Position, *, proposed: bool = False
    ) -> list[tuple[str, str, str]]:
        return list(self.statements_at(axis, point, proposed=proposed).values())


class KnowledgeTransaction(_Reader):
    """The reads and writes of one `BEGIN IMMEDIATE` transaction. Obtain it from `transaction()`."""

    def entity_for(self, name: str, kind: str = "thing") -> Entity:
        """The entity `name` is, made if no stage found it: `resolve`, then `add_entity`."""
        return self.add_entity(self.resolve(name), kind)

    def add_entity(self, ref: EntityRef, kind: str = "thing") -> Entity:
        """The entity `ref` stands for; a `new` ref is created, as the clone's policy says.

        `existing` returns that entity. `new` creates one, with the reserved id when the
        name is a reserved one. A near candidate on the ref is ignored: merging it is step
        5, so a near name is its own entity. `ambiguous` returns the oldest of the entities
        that matched and logs it: a fact write is not refused for a name two entities share,
        and the oldest is the one that was there first.
        """
        if ref.kind == "existing" and ref.entity_id is not None:
            found = self.entity(ref.entity_id)
            if found is None:
                raise KeyError(f"entity {ref.entity_id!r} is not in scope {self.scope}")
            return found
        if ref.kind == "ambiguous":
            matched = [e for e in self.entities() if e.id in ref.ambiguous_ids]
            logger.warning(
                "Name %r matches %d entities in %s; the oldest is used",
                ref.name,
                len(matched),
                self.scope,
            )
            return matched[0]
        reserved = dict(self.policy.reserved)
        key = name_key(ref.name)
        if key in reserved:
            entity = Entity(id=key, scope=self.scope, kind=reserved[key], name=key)
        else:
            entity = Entity(
                id=f"ent_{uuid.uuid4().hex[:12]}",
                scope=self.scope,
                kind=kind,
                name=_collapse(ref.name),
            )
        self._conn.execute(
            "INSERT INTO entities (id, kind, name, aliases, summary) VALUES (?, ?, ?, '[]', '')",
            (entity.id, entity.kind, entity.name),
        )
        return entity

    def add_episode(self, episode: Episode) -> None:
        """Keep an episode; one already kept under that id is left as it is."""
        self._check_scope(episode.scope)
        self._conn.execute(
            "INSERT OR IGNORE INTO episodes (id, kind, ref, at) VALUES (?, ?, ?, ?)",
            (episode.id, episode.kind, episode.ref, episode.at),
        )

    def add_edge(self, edge: Edge, record: dict[str, Any] | None = None) -> None:
        """Write one edge. Its subject (and object) must be entities of this scope already."""
        self._check_scope(edge.scope)
        for entity_id in (edge.subject_id, edge.object_id):
            if entity_id is not None and self.entity(entity_id) is None:
                raise KeyError(f"entity {entity_id!r} is not in scope {self.scope}")
        self._conn.execute(
            "INSERT INTO edges (id, subject_id, predicate, object_id, value, axis, valid_start,"
            " valid_end, recorded_at, expired_at, status, confidence, origin, record)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                edge.id,
                edge.subject_id,
                edge.predicate,
                edge.object_id,
                edge.value,
                edge.valid.axis,
                edge.valid.start,
                edge.valid.end,
                edge.recorded_at,
                edge.expired_at,
                edge.status,
                edge.confidence,
                edge.origin,
                json.dumps(record or {}, ensure_ascii=False),
            ),
        )
        for episode_id, quote in edge.evidence:
            self._conn.execute(
                "INSERT INTO edge_evidence (edge_id, episode_id, quote) VALUES (?, ?, ?)",
                (edge.id, episode_id, quote),
            )

    def update_edge(
        self,
        edge_id: str,
        *,
        status: EdgeStatus | None = None,
        expired_at: str | None = None,
        record: dict[str, Any] | None = None,
    ) -> None:
        """Change an edge's status, its end of belief, or its record; what is not given stays.

        `record` is merged over the held one, key by key. History is never rewritten: this
        has no way to change a fact's subject, predicate, value or `recorded_at`.
        """
        held = self.row(edge_id)
        if held is None:
            raise KeyError(f"edge {edge_id!r} is not in scope {self.scope}")
        merged = {**held.record, **(record or {})}
        self._conn.execute(
            "UPDATE edges SET status = ?, expired_at = ?, record = ? WHERE id = ?",
            (
                status if status is not None else held.edge.status,
                expired_at if expired_at is not None else held.edge.expired_at,
                json.dumps(merged, ensure_ascii=False),
                edge_id,
            ),
        )

    def set_meta(self, key: str, value: str) -> None:
        """Keep `value` under `key` in the file's `meta` table, inside this transaction."""
        self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def _check_scope(self, scope: str) -> None:
        if scope != self.scope:
            raise ScopeMismatchError(f"a store of {self.scope} was given an object of {scope}")


def _first_row(conn: sqlite3.Connection, sql: str) -> sqlite3.Row | None:
    return conn.execute(sql).fetchone()


class SqliteKnowledgeStore:
    """One clone's entities, episodes and edges in `path`, or in memory when `path` is `None`.

    `agent_id` names the scope (`clone:<agent id>`). Opening a file made for another scope
    raises `ScopeMismatchError`. `create=False` opens an existing file read-only and
    creates nothing, for a reader that must not make a home: it raises
    `FileNotFoundError` for a file that is not there.

    Raises:
        KnowledgeUnreadableError: The file is not a SQLite database, is malformed, or is of
            another schema version. Nothing is changed.
        sqlite3.OperationalError: The file cannot be opened or written (permissions, a
            lock held past `BUSY_TIMEOUT_SECONDS`): not damage, so not "unreadable".
    """

    def __init__(
        self,
        path: Path | str | None,
        agent_id: str,
        policy: ClonePolicy = CLONE_POLICY,
        *,
        create: bool = True,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._scope = f"clone:{agent_id}"
        self.policy = policy
        self._writable = create or self._path is None
        self._anchor: sqlite3.Connection | None = None
        if self._path is None:
            # An in-memory store lives as long as one connection to it does, so this one is
            # held; every operation still opens its own beside it.
            self._uri = f"file:uclone_knowledge_{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._anchor = sqlite3.connect(self._uri, uri=True, isolation_level=None)
        elif create:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        elif not self._path.is_file():
            raise FileNotFoundError(str(self._path))
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    @property
    def scope(self) -> str:
        return self._scope

    @property
    def path(self) -> Path | None:
        return self._path

    def close(self) -> None:
        """Release the in-memory anchor; a file-backed store holds nothing to release."""
        if self._anchor is not None:
            self._anchor.close()
            self._anchor = None

    def __del__(self) -> None:
        self.close()

    # -- connections -------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._path is None:
            conn = sqlite3.connect(
                self._uri, uri=True, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None
            )
        elif self._writable:
            conn = sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
        else:
            conn = sqlite3.connect(
                f"{self._path.as_uri()}?mode=ro",
                uri=True,
                timeout=BUSY_TIMEOUT_SECONDS,
                isolation_level=None,
            )
        conn.row_factory = sqlite3.Row
        return conn

    def _open(self) -> None:
        """Check the file, and for a writer make the schema and fix the scope in it."""
        conn = self._connect()
        try:
            try:
                self._prepare(conn)
            except sqlite3.OperationalError:
                raise  # a lock or a permission, not damage
            except sqlite3.DatabaseError as exc:
                raise KnowledgeUnreadableError(
                    "The saved knowledge could not be read.",
                    path=self._path,
                    cause=f"{type(exc).__name__}: {exc}",
                ) from exc
        finally:
            conn.close()

    def _prepare(self, conn: sqlite3.Connection) -> None:
        if self._path is not None and self._writable:
            conn.execute("PRAGMA journal_mode=WAL")
        verdict = _first_row(conn, "PRAGMA quick_check")
        if verdict is None or verdict[0] != "ok":
            raise sqlite3.DatabaseError(f"quick_check: {None if verdict is None else verdict[0]}")
        if self._writable:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in filter(None, (s.strip() for s in _SCHEMA.split(";"))):
                    conn.execute(statement)
                conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema', ?)",
                    (SCHEMA_VERSION,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('scope', ?)", (self._scope,)
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        if meta.get("schema") != SCHEMA_VERSION:
            raise sqlite3.DatabaseError(f"schema {meta.get('schema')!r}, expected {SCHEMA_VERSION}")
        if meta.get("scope") != self._scope:
            raise ScopeMismatchError(
                f"{self._path or 'the store'} holds {meta.get('scope')}, not {self._scope}"
            )

    # -- the one-statement reads -------------------------------------------------------

    @contextmanager
    def _reading(self) -> Generator[_Reader]:
        conn = self._connect()
        try:
            yield _Reader(conn, self._scope, self.policy)
        finally:
            conn.close()

    def entities(self) -> list[Entity]:
        with self._reading() as reader:
            return reader.entities()

    def edges(self) -> list[Edge]:
        with self._reading() as reader:
            return reader.edges()

    def rows(self) -> list[EdgeRow]:
        with self._reading() as reader:
            return reader.rows()

    def row(self, edge_id: str) -> EdgeRow | None:
        with self._reading() as reader:
            return reader.row(edge_id)

    def resolve(self, name: str) -> EntityRef:
        with self._reading() as reader:
            return reader.resolve(name)

    def meta(self, key: str) -> str | None:
        with self._reading() as reader:
            return reader.meta(key)

    def edges_at(self, axis: Axis, point: Position, *, proposed: bool = False) -> list[Edge]:
        with self._reading() as reader:
            return reader.edges_at(axis, point, proposed=proposed)

    def facts_at(
        self, axis: Axis, point: Position, *, proposed: bool = False
    ) -> list[tuple[str, str, str]]:
        with self._reading() as reader:
            return reader.facts_at(axis, point, proposed=proposed)

    def statements_at(
        self, axis: Axis, point: Position, *, proposed: bool = False
    ) -> dict[str, tuple[str, str, str]]:
        with self._reading() as reader:
            return reader.statements_at(axis, point, proposed=proposed)

    # -- the write ---------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Generator[KnowledgeTransaction]:
        """One write: everything inside commits together, or none of it does.

        Takes the file's write lock first, so what is read inside is what is there when the
        writes land.
        """
        if not self._writable:
            raise PermissionError("this store was opened read-only")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")  # the one write lock, held to COMMIT
            try:
                yield KnowledgeTransaction(conn, self._scope, self.policy)
            except BaseException:
                conn.execute("ROLLBACK")  # a failed write leaves nothing
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()
