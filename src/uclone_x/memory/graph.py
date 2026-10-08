"""A `MemoryFact` as an uGraph edge, and back (uGraph design §3.3, §5 step 3).

The clone's facts are edges of its `SqliteKnowledgeStore`. A fact has more than an `Edge`
does (provenance, tags, the session it came from), so the store keeps the whole fact as the
edge's opaque `record`, and the edge carries what the graph reads: its subject entity, its
predicate and value, the interval it holds in, when it was recorded and when it stopped
being believed, and whether it was retracted.

What is derived and what is stored:

- `retracted` is the edge: status `retracted` (a person said forget it), or `expired_at`
  set (a later fact replaced it). The record's own `retracted` is never read.
- `valid_until` is `valid.end`: the end a person stated.
- Everything else comes from the record.

This module is the only place that knows that mapping.
"""

from __future__ import annotations

import json
from typing import Any

from uclone_x.knowledge.models import Edge, Interval
from uclone_x.knowledge.sqlite_store import EdgeRow, KnowledgeTransaction
from uclone_x.memory.models import MemoryFact

__all__ = ["fact_of", "retract_edge", "supersede_edge", "write_fact"]


def write_fact(tx: KnowledgeTransaction, fact: MemoryFact, *, subject_kind: str = "thing") -> None:
    """Write `fact` as an edge of `tx`'s scope, its subject resolved through the resolver (G11).

    A retracted fact is written as a retracted edge, so history survives an import.
    """
    subject = tx.entity_for(fact.subject, subject_kind)
    edge = Edge(
        id=fact.fact_id,
        scope=tx.scope,
        subject_id=subject.id,
        predicate=fact.predicate,
        value=fact.object_value,
        valid=Interval(axis=tx.policy.axis, start=fact.created_at, end=fact.valid_until),
        recorded_at=fact.created_at,
        status="retracted" if fact.retracted else tx.policy.initial_status,
        confidence=fact.confidence,
        origin=fact.origin,
    )
    tx.add_edge(edge, fact.model_dump(mode="json"))


def fact_of(row: EdgeRow) -> MemoryFact:
    """The fact an edge and its record say."""
    data: dict[str, Any] = dict(row.record)
    data["retracted"] = row.edge.status == "retracted" or row.edge.expired_at is not None
    data["valid_until"] = row.edge.valid.end
    return MemoryFact.model_validate_json(json.dumps(data))


def retract_edge(
    tx: KnowledgeTransaction, fact_id: str, *, reason: str, at: str, updated_at: str
) -> None:
    """A person withdrew the fact: the edge is `retracted`, with the reason in its record."""
    tx.update_edge(
        fact_id,
        status="retracted",
        record={"retraction_reason": reason, "retracted_at": at, "updated_at": updated_at},
    )


def supersede_edge(
    tx: KnowledgeTransaction, fact_id: str, *, by: str, at: str, reason: str | None = None
) -> None:
    """A later fact replaced this one: it stops being believed at `at`, and stays `approved`.

    The old edge is history, not a mistake: its `expired_at` is the new edge's `recorded_at`.
    """
    tx.update_edge(
        fact_id,
        expired_at=at,
        record={
            "retraction_reason": reason or f"Superseded by fact {by}",
            "retracted_at": at,
            "updated_at": at,
        },
    )
