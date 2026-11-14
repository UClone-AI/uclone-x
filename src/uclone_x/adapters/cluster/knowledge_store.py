"""PostgreSQL durable knowledge triple store adapter for stateless background learners."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uclone_x.core.cluster_protocols import SqlPoolProtocol

__all__ = [
    "KnowledgeTripleRecord",
    "PostgresKnowledgeStore",
]


class KnowledgeTripleRecord(BaseModel):
    """Immutable relational knowledge triple persisted in `ucx_knowledge_triples`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    clone_id: str
    subject: str
    predicate: str
    object_val: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_room_id: str | None = None

    @field_validator("clone_id", "subject", "predicate", "object_val")
    @classmethod
    def _non_empty_field(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("KnowledgeTripleRecord fields must not be blank")
        return clean


class PostgresKnowledgeStore:
    """Durable relational store for clone knowledge triples backed by `SqlPoolProtocol`."""

    _UPSERT_SQL = (
        "INSERT INTO ucx_knowledge_triples "
        "(clone_id, subject, predicate, object_val, confidence, source_room_id) "
        "VALUES ($1, $2, $3, $4, $5, $6) "
        "ON CONFLICT (clone_id, subject, predicate, object_val) DO UPDATE SET "
        "confidence = EXCLUDED.confidence, source_room_id = EXCLUDED.source_room_id"
    )
    _SELECT_BY_CLONE_SQL = (
        "SELECT clone_id, subject, predicate, object_val, confidence, source_room_id "
        "FROM ucx_knowledge_triples WHERE clone_id = $1 "
        "ORDER BY subject ASC, predicate ASC, object_val ASC"
    )
    _DELETE_BY_CLONE_SQL = "DELETE FROM ucx_knowledge_triples WHERE clone_id = $1"

    def __init__(self, sql_pool: SqlPoolProtocol) -> None:
        self._sql_pool = sql_pool

    async def upsert_triple(self, record: KnowledgeTripleRecord) -> None:
        """Insert or update a single `KnowledgeTripleRecord` in PostgreSQL."""
        async with self._sql_pool.acquire() as conn:
            await conn.execute(
                self._UPSERT_SQL,
                record.clone_id,
                record.subject,
                record.predicate,
                record.object_val,
                record.confidence,
                record.source_room_id,
            )

    async def upsert_triples(self, records: Sequence[KnowledgeTripleRecord]) -> None:
        """Atomically upsert a batch of `KnowledgeTripleRecord` rows inside one transaction."""
        if not records:
            return
        async with self._sql_pool.acquire() as conn:
            async with conn.transaction():
                for rec in records:
                    await conn.execute(
                        self._UPSERT_SQL,
                        rec.clone_id,
                        rec.subject,
                        rec.predicate,
                        rec.object_val,
                        rec.confidence,
                        rec.source_room_id,
                    )

    async def list_triples(self, clone_id: str) -> tuple[KnowledgeTripleRecord, ...]:
        """Return all stored knowledge triples for `clone_id` in deterministic order."""
        clean_clone = clone_id.strip()
        if not clean_clone:
            raise ValueError("clone_id must not be blank")
        async with self._sql_pool.acquire() as conn:
            rows = await conn.fetch(self._SELECT_BY_CLONE_SQL, clean_clone)
        return tuple(
            KnowledgeTripleRecord(
                clone_id=str(row["clone_id"]),
                subject=str(row["subject"]),
                predicate=str(row["predicate"]),
                object_val=str(row["object_val"]),
                confidence=float(row.get("confidence", 1.0)),
                source_room_id=(
                    str(row["source_room_id"]) if row.get("source_room_id") is not None else None
                ),
            )
            for row in rows
        )

    async def delete_clone_triples(self, clone_id: str) -> None:
        """Delete all stored knowledge triples for `clone_id`."""
        clean_clone = clone_id.strip()
        if not clean_clone:
            raise ValueError("clone_id must not be blank")
        async with self._sql_pool.acquire() as conn:
            await conn.execute(self._DELETE_BY_CLONE_SQL, clean_clone)
