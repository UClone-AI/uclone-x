"""The uGraph data model: entities, episodes, and edges with two time axes.

The shape follows the uGraph design, §3.1. A scope is a vessel (`clone:<agent id>`,
`story:<story id>`); every object carries one, and a store opens one. This module is pure
data: it has no storage and no policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

__all__ = [
    "Axis",
    "Edge",
    "EdgeStatus",
    "Entity",
    "EntityRef",
    "Episode",
    "EpisodeKind",
    "Interval",
    "Position",
    "mint_entity_ref",
]

Axis = Literal["wall", "story"]
EdgeStatus = Literal["proposed", "approved", "retracted"]
EpisodeKind = Literal["turn", "scene", "tool", "manual"]

#: A point on one axis. Totally ordered within its axis and never compared across axes:
#: an ISO-8601 UTC string on `wall`, and `story`'s `(TimeKey, int)` scene place.
Position = Any


@dataclass(frozen=True)
class Entity:
    id: str
    scope: str
    kind: str
    name: str
    aliases: tuple[str, ...] = ()
    summary: str = ""


@dataclass(frozen=True)
class Episode:
    id: str
    scope: str
    kind: EpisodeKind
    ref: str
    at: str


@dataclass(frozen=True)
class Interval:
    """When a fact holds: `[start, end)` on one axis, either end open (`None`)."""

    axis: Axis
    start: Position | None = None
    end: Position | None = None

    def contains(self, axis: Axis, point: Position) -> bool:
        """Whether `point` on `axis` is in the interval; a different axis never is."""
        if axis != self.axis:
            return False
        if self.start is not None and point < self.start:
            return False
        return self.end is None or point < self.end


@dataclass(frozen=True)
class Edge:
    """One fact: `subject predicate object` (an entity) or `value` (a literal).

    `valid` is when the fact holds as the vessel believes it. `proposed_valid`, when set,
    is when it holds once every pending proposal is taken as made: an approved value that
    a proposed change (or a proposed removal) replaces ends there in the approved-and-
    proposed view, and only there (`store.holds_at`). `None` is the same as `valid`.
    """

    id: str
    scope: str
    subject_id: str
    predicate: str
    valid: Interval
    recorded_at: str
    object_id: str | None = None
    value: str | None = None
    expired_at: str | None = None
    status: EdgeStatus = "approved"
    confidence: float = 1.0
    origin: str = ""
    evidence: tuple[tuple[str, str], ...] = ()  # (episode id, quote)
    proposed_valid: Interval | None = None

    def __post_init__(self) -> None:
        if (self.object_id is None) == (self.value is None):
            raise ValueError("an edge has exactly one of object_id and value")
        if self.proposed_valid is not None and self.proposed_valid.axis != self.valid.axis:
            raise ValueError("an edge's two intervals are on one axis")

    def valid_in(self, *, proposed: bool) -> Interval:
        """When the edge holds in the approved view, or in the approved-and-proposed one."""
        if proposed and self.proposed_valid is not None:
            return self.proposed_valid
        return self.valid


_MINT: Final = object()


@dataclass(frozen=True)
class EntityRef:
    """What a name resolved to. Only `EntityResolver` makes one (G11).

    `existing`: `entity_id` is the entity. `ambiguous`: `ambiguous_ids` all matched and
    nothing was guessed. `new`: no stage found it; `candidates` are near matches a later
    stage may confirm, `(entity id, score)` best first.
    """

    kind: Literal["existing", "new", "ambiguous"]
    name: str
    entity_id: str | None = None
    ambiguous_ids: tuple[str, ...] = ()
    candidates: tuple[tuple[str, float], ...] = ()
    stage: int = 0
    _mint: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._mint is not _MINT:
            raise TypeError("an EntityRef is made by EntityResolver, not by hand")


def mint_entity_ref(**kwargs: Any) -> EntityRef:
    """For `EntityResolver` alone."""
    return EntityRef(_mint=_MINT, **kwargs)
