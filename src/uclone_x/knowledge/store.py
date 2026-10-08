"""What a uGraph store gives a reader: the edges and facts that hold at one point.

Two views of one point, and no third:

- **approved** (the default): what the vessel believes. `facts_at` gives the reasoner
  these and nothing else.
- **approved and proposed** (`proposed=True`): what it believes and what is waiting for a
  decision, for a check that must see a change before a person approves it (a death a
  scene wrote, still a proposal). A proposed edge stays `proposed` in the result, so the
  caller can tell the two apart.

A `retracted` edge, and an edge whose belief has ended (`expired_at`), is in neither.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final, NamedTuple, Protocol

from uclone_x.knowledge.models import Axis, Edge, EdgeStatus, Entity, Position

__all__ = [
    "APPROVED",
    "APPROVED_OR_PROPOSED",
    "EdgeRow",
    "KnowledgeStore",
    "edges_at",
    "facts_at",
    "holds_at",
]

#: The statuses each view reads. `retracted` is in neither.
APPROVED: Final[frozenset[EdgeStatus]] = frozenset({"approved"})
APPROVED_OR_PROPOSED: Final[frozenset[EdgeStatus]] = frozenset({"approved", "proposed"})


class EdgeRow(NamedTuple):
    """An edge with the opaque record its writer kept beside it. The store reads none of it."""

    edge: Edge
    record: dict[str, Any]


def holds_at(edge: Edge, axis: Axis, point: Position, *, proposed: bool = False) -> bool:
    """Whether `edge` is in the view of `point` on `axis`: believed (or, with `proposed`,
    waiting for a decision), unexpired, and valid there."""
    wanted = APPROVED_OR_PROPOSED if proposed else APPROVED
    return edge.status in wanted and edge.expired_at is None and edge.valid.contains(axis, point)


def edges_at(
    edges: Iterable[Edge], axis: Axis, point: Position, *, proposed: bool = False
) -> list[Edge]:
    """The edges that hold at `point` on `axis`, in the order given (`holds_at`)."""
    return [e for e in edges if holds_at(e, axis, point, proposed=proposed)]


def facts_at(
    edges: Iterable[Edge], axis: Axis, point: Position, *, proposed: bool = False
) -> list[tuple[str, str, str]]:
    """`(subject, predicate, object or value)` of every edge that holds at `point`.

    The shared `facts_at`, for a store that has its edges in hand.
    """
    return [
        (e.subject_id, e.predicate, e.object_id if e.object_id is not None else str(e.value))
        for e in edges_at(edges, axis, point, proposed=proposed)
    ]


class KnowledgeStore(Protocol):
    """One scope's entities and edges. A store opens one scope and no other (G2)."""

    @property
    def scope(self) -> str: ...

    def entities(self) -> Iterable[Entity]: ...

    def edges(self) -> Iterable[Edge]: ...

    def edges_at(self, axis: Axis, point: Position, *, proposed: bool = False) -> list[Edge]:
        """The edges in the view of `point` on `axis` (`holds_at`)."""
        ...

    def facts_at(
        self, axis: Axis, point: Position, *, proposed: bool = False
    ) -> list[tuple[str, str, str]]:
        """`(subject, predicate, object or value)` of every edge `edges_at` gives."""
        ...
