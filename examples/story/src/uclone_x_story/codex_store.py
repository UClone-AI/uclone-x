"""The story codex read as a uGraph store: the YAML stays the source, and is read as edges.

The uGraph design, §3.2. A person edits the codex as YAML files, and nothing here writes
one: an entry file, a proposal file and the outline are read, and the store is what they
say, in the shared fact model (`uclone_x.knowledge`). Writing stays where it was: a
proposal is a file, and approving it writes the entry file (`uclone_x_story.proposals`).

How the files become the model:

- each codex entry is an `Entity` (its kind is its folder, its summary its profile); a
  name a pending proposal adds to it (`ProposalChange.aliases`) is one of its aliases, as
  a pending new entry is an entity, so the next reading resolves it;
- each `state` value is an edge `(entry, key, value)`; a list is one edge per element. A
  value that is a set of named fields, or `null`, is kept as an edge whose record says it
  is not a single value (`record["single"]` false), so a reader can say it did not check
  it rather than drop it;
- each `relations` value is an edge `(entry, word, other entry)`: an edge between two
  entities, `object_id` the other's id;
- a progression ends the edge it replaces and starts the new one, both at the scene's
  place in story time (`scene_point`). An entry's own values hold from the story's start;
- a **pending** proposal is the same edges with status `proposed`: a progression's from
  its scene until the entry's next change of that value -- its own or another pending
  proposal's -- and a new entry's from the start. A **rejected** one is `retracted`, with
  `expired_at` the time it was decided. An applied one is already in the entry file, so it
  is not read twice;
- in the approved-and-proposed view the store is the codex as it would be with every
  pending proposal made: a pending change ends the value it replaces there, and a pending
  removal (`null`) ends it with nothing after (`Edge.proposed_valid`). The approved view is
  untouched by either. A pending change is placed after the entry's own changes at the same
  scene, and after the pending proposals made before it, as approving them in turn would.

A point on the story axis is `(TimeKey, reading index, phase)` (`scene_point`): phase 0 is
the scene as it starts, 1 as it has ended. A change placed at a scene holds from that
scene's end, so it is in the view of the scene's end and not of its start, which is what
`timeline.entry_snapshot` says with `through_scene`.

Each edge's record is where it came from, in the words the audit reports it (`origin`):
the file, the field, and the scene of the progression that set it.

This module is pure: it is given the files' contents and reads nothing itself. A caller
with a `StoryWork` passes `require_outline()`, `codex()` and the loaded `proposals()`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast

from uclone_x.knowledge import (
    Edge,
    EdgeRow,
    Entity,
    EntityRef,
    EntityResolver,
    Interval,
    edges_at,
    facts_at,
    holds_at,
)
from uclone_x.knowledge.models import Axis, EdgeStatus, Position
from uclone_x_story.context import CodexIndex, CodexItem, UnreadableFile
from uclone_x_story.schemas import CodexEntry, Outline, Progression, Proposal
from uclone_x_story.timeline import Placement, place_scenes

__all__ = [
    "STORY_AXIS",
    "CodexYamlStore",
    "pending_aliases",
    "scene_point",
    "story_scope",
    "with_pending_aliases",
]

STORY_AXIS: Final[Axis] = "story"

_STARTS: Final = 0
_ENDED: Final = 1

#: `_Builder.add`'s "no other end once the pending proposals are made".
_SAME: Final = object()


def story_scope(story_id: str) -> str:
    """The scope a story's facts live in, and no other vessel's (G2)."""
    return f"story:{story_id}"


def scene_point(placement: Placement, *, ended: bool) -> Position:
    """Scene `placement` on the story axis: as it starts, or once it has ended."""
    key, index = placement.position
    return (key, index, _ENDED if ended else _STARTS)


def _single(value: object) -> str | None:
    """A single value as text; `None` for a set of named fields, a list, or `null`."""
    if isinstance(value, dict | list) or value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass
class _Cell:
    """One value of one entry, as it changes: `(section, key)` is `("state", "status")` or
    `("relations", "harin")`."""

    value: Any
    start: Position | None
    set_at: str | None
    proposal: str | None = None


@dataclass
class _Builder:
    """The edges of one source (an entry file, or a proposal), in the order they open."""

    scope: str
    status: EdgeStatus
    recorded_at: str
    origin: dict[str, Any]
    expired_at: str | None = None
    evidence: tuple[tuple[str, str], ...] = ()
    applied_evidence: Mapping[str, tuple[tuple[str, str], ...]] = field(
        default_factory=dict[str, tuple[tuple[str, str], ...]]
    )
    id_prefix: str = ""
    rows: list[tuple[int, int, EdgeRow]] = field(default_factory=list[tuple[int, int, EdgeRow]])
    _order: dict[tuple[str, str], int] = field(default_factory=dict[tuple[str, str], int])

    def add(
        self,
        subject: str,
        section: str,
        key: str,
        cell: _Cell,
        end: Position | None,
        proposed_end: Position | None | object = _SAME,
    ) -> None:
        """One edge per element of `cell`'s value, from its start to `end`.

        `proposed_end` is where it ends once every pending proposal is made, when a pending
        change ends it before `end` (`_SAME`: it does not). A proposed edge holds only in
        that view, so it is given its end there as `end`.
        """
        if self.status == "proposed" and proposed_end is not _SAME:
            end, proposed_end = cast("Position | None", proposed_end), _SAME
        if cell.start is not None and end is not None and not cell.start < end:
            return  # replaced at the same moment it was set: it never held
        proposed_valid = (
            None
            if proposed_end is _SAME or proposed_end == end
            else Interval(STORY_AXIS, cell.start, cast("Position | None", proposed_end))
        )
        order = self._order.setdefault((section, key), len(self._order))
        origin = {**self.origin, "field": f"{section}.{key}"}
        if cell.set_at is not None:
            origin["set_by_progression_at"] = cell.set_at
        evidence = self.evidence
        if cell.proposal is not None:
            origin["proposal"] = cell.proposal
            evidence = self.applied_evidence.get(cell.proposal, ())
        raw: object = cell.value
        values = cast("list[object]", raw) if isinstance(raw, list) else [raw]
        for element in values:
            text = _single(element)
            single = text is not None
            if text is None:
                text = json.dumps(element, ensure_ascii=False)
            number = len(self.rows)
            edge = Edge(
                id=f"{self.id_prefix}{subject}/{section}.{key}/{number}",
                scope=self.scope,
                subject_id=subject,
                predicate=text if section == "relations" else key,
                object_id=key if section == "relations" else None,
                value=None if section == "relations" else text,
                valid=Interval(STORY_AXIS, cell.start, end),
                recorded_at=self.recorded_at,
                expired_at=self.expired_at,
                status=self.status,
                origin=str(self.origin.get("from", "")),
                evidence=evidence,
                proposed_valid=proposed_valid,
            )
            self.rows.append((order, number, EdgeRow(edge, {"origin": origin, "single": single})))

    def ordered(self) -> list[EdgeRow]:
        return [row for _, _, row in sorted(self.rows, key=lambda r: (r[0], r[1]))]


def _placed(
    progressions: Sequence[Progression], placements: Mapping[str, Placement]
) -> list[tuple[Position, int, Progression]]:
    """The progressions whose scene is in the outline, in story order; the rest never hold."""
    found = [
        (scene_point(placements[p.at], ended=True), order, p)
        for order, p in enumerate(progressions)
        if p.at in placements
    ]
    return sorted(found, key=lambda step: (step[0], step[1]))


def _sets(progression: Progression) -> list[tuple[str, str, Any]]:
    """`(section, key, value)` of every value `progression` sets; `None` removes it."""
    sets: list[tuple[str, str, Any]] = [("state", k, v) for k, v in progression.set.items()]
    sets += [("relations", k, v) for k, v in progression.relations.items()]
    return sets


@dataclass(frozen=True)
class _Pending:
    """A pending change to one value of an entry: where it is placed, and its turn among
    the pending proposals (the order they would be approved in)."""

    point: Position
    order: int


#: Per entry id, per `(section, key)`, the pending changes to that value.
_PendingChanges = dict[str, dict[tuple[str, str], list[_Pending]]]


def _earliest(a: Position | None, b: Position | None) -> Position | None:
    """The earlier of two ends; `None` is no end."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _ended_by_pending(
    pending: Sequence[_Pending], start: Position | None, *, order: int | None = None
) -> Position | None:
    """Where the first pending change ends a value held from `start`, or `None`.

    A pending change at `start` itself ends the entry's own value set there (it is placed
    after the entry's own changes), and a pending value only when it comes later in turn
    (`order`): a later proposal replaces an earlier one at the same scene.
    """
    ends = [
        p.point
        for p in pending
        if start is None
        or p.point > start
        or (p.point == start and (order is None or p.order > order))
    ]
    return min(ends, default=None)


def _entry_edges(
    entry: CodexEntry,
    placements: Mapping[str, Placement],
    out: _Builder,
    pending: Mapping[tuple[str, str], Sequence[_Pending]] | None = None,
) -> dict[tuple[str, str], list[Position]]:
    """Every edge `entry` holds over the story, into `out`; returned, where each of its
    values changes (for a proposed change to end at the next one).

    `pending` are the pending changes to the entry's values: each value ends at the first
    of them in the approved-and-proposed view."""
    waiting = pending or {}
    cells: dict[tuple[str, str], _Cell] = {}
    for key, value in entry.state.items():
        cells[("state", key)] = _Cell(value, None, None)
    for other, word in entry.relations.items():
        cells[("relations", other)] = _Cell(word, None, None)
    changes: dict[tuple[str, str], list[Position]] = {}
    for point, _, progression in _placed(entry.progressions, placements):
        for section, key, value in _sets(progression):
            changes.setdefault((section, key), []).append(point)
            held = cells.pop((section, key), None)
            if held is not None:
                ended = _ended_by_pending(waiting.get((section, key), ()), held.start)
                out.add(entry.id, section, key, held, point, _earliest(point, ended))
            if value is not None:
                cells[(section, key)] = _Cell(value, point, progression.at, progression.proposal)
    for (section, key), held in cells.items():
        ended = _ended_by_pending(waiting.get((section, key), ()), held.start)
        out.add(entry.id, section, key, held, None, ended)
    return changes


class CodexYamlStore:
    """One story's codex, as entities and edges in the story's scope. Read-only.

    Build it with `read` from what a story's files hold.
    """

    def __init__(
        self,
        story_id: str,
        placements: Mapping[str, Placement],
        entities: Iterable[Entity],
        rows: Iterable[EdgeRow],
        items: Iterable[CodexItem] = (),
        unreadable: Iterable[UnreadableFile] = (),
    ) -> None:
        self.story_id = story_id
        self._scope = story_scope(story_id)
        self.placements: Mapping[str, Placement] = placements
        self._entities = tuple(entities)
        self._rows = tuple(rows)
        self._items = tuple(items)
        self._unreadable = tuple(unreadable)

    @classmethod
    def read(
        cls,
        story_id: str,
        outline: Outline,
        codex: CodexIndex,
        proposals: Iterable[Proposal] = (),
    ) -> CodexYamlStore:
        """The store `outline`, `codex` and `proposals` make. A proposal that is neither
        pending nor rejected is in its entry file already, and is not read again; its
        evidence stays linked to the progression that names it (uGraph §6 Q3)."""
        scope = story_scope(story_id)
        placements = place_scenes(outline)
        all_proposals = list(proposals)
        live = [p for p in all_proposals if p.status != "applied"]
        applied_evidence = {
            p.id: tuple((e.scene_id, e.quote) for e in p.evidence)
            for p in all_proposals
            if p.status == "applied"
        }
        waiting = _pending_changes(live, placements)
        called = pending_aliases(live)
        entities: list[Entity] = []
        items: list[CodexItem] = [with_pending_aliases(item, called) for item in codex.items]
        rows: list[EdgeRow] = []
        changes: dict[str, dict[tuple[str, str], list[Position]]] = {}
        for item in items:
            entities.append(_entity(scope, item))
            built = _Builder(
                scope=scope,
                status="approved",
                recorded_at="",
                origin={"from": "codex", "file": f"codex/{item.kind}/{item.entry.id}.yaml"},
                applied_evidence=applied_evidence,
                id_prefix=f"{item.kind}:",
            )
            changes[item.entry.id] = _entry_edges(
                item.entry, placements, built, waiting.get(item.entry.id)
            )
            rows += built.ordered()
        known = {e.id for e in entities}
        for order, proposal in enumerate(live):
            pending = proposal.status == "pending"
            built = _Builder(
                scope=scope,
                status="proposed" if pending else "retracted",
                recorded_at=proposal.proposed_at,
                expired_at=None if pending else (proposal.decided_at or proposal.proposed_at),
                origin={
                    "from": "proposal",
                    "proposal": proposal.id,
                    "file": f"proposals/{proposal.id}.yaml",
                },
                evidence=tuple((e.scene_id, e.quote) for e in proposal.evidence),
                id_prefix=f"{proposal.id}:",
            )
            if proposal.change.new_entry is not None:
                try:
                    entry = proposal.new_entry()
                except ValueError:
                    continue
                if entry.id in known:
                    continue  # the entry is there: the codex's own wins
                if pending:
                    known.add(entry.id)
                    item = with_pending_aliases(CodexItem(kind=proposal.kind, entry=entry), called)
                    entities.append(_entity(scope, item))
                    items.append(item)
                    changes[entry.id] = _entry_edges(
                        entry, placements, built, waiting.get(entry.id)
                    )
                else:
                    _entry_edges(entry, placements, built)
            elif proposal.change.progression is not None:
                _proposed_change(
                    proposal.entry_id,
                    proposal.change.progression,
                    placements,
                    changes.get(proposal.entry_id, {}),
                    built,
                    waiting.get(proposal.entry_id, {}) if pending else {},
                    order,
                )
            rows += built.ordered()
        return cls(story_id, placements, entities, rows, items, codex.unreadable)

    # -- the store -------------------------------------------------------------------------

    @property
    def scope(self) -> str:
        return self._scope

    def entities(self) -> list[Entity]:
        return list(self._entities)

    def rows(self) -> list[EdgeRow]:
        return list(self._rows)

    def edges(self) -> list[Edge]:
        return [row.edge for row in self._rows]

    def resolve(self, name: str) -> EntityRef:
        """What `name` names among the story's entities (`EntityResolver`, stages 1-3)."""
        return self.resolver().resolve(name)

    def resolver(self) -> EntityResolver:
        """The resolver over every entity, the codex's and those pending proposals add."""
        return EntityResolver(self._scope, self._entities)

    def codex(self) -> CodexIndex:
        """The codex as the approved-and-proposed view has it: every entry file, and each
        entry a pending proposal would add, after them. What reads entries by name or kind
        (`CodexNames`, a prompt's list of entries) reads them here."""
        return CodexIndex(items=self._items, unreadable=self._unreadable)

    def kind_of(self, entity_id: str) -> str | None:
        """The kind (codex folder) of entity `entity_id`, or `None` for no such entity."""
        return next((e.kind for e in self._entities if e.id == entity_id), None)

    def values_at(
        self, subject: str, key: str, point: Position, *, proposed: bool = True
    ) -> list[EdgeRow]:
        """The edges of state value `key` of `subject` in the view of `point`."""
        return [
            r
            for r in self.rows_at(point, proposed=proposed)
            if r.edge.subject_id == subject and r.edge.object_id is None and r.edge.predicate == key
        ]

    def relations_at(
        self, subject: str, point: Position, *, proposed: bool = True
    ) -> dict[str, EdgeRow]:
        """What `subject` is to each other entity in the view of `point`, by its id."""
        return {
            r.edge.object_id: r
            for r in self.rows_at(point, proposed=proposed)
            if r.edge.subject_id == subject and r.edge.object_id is not None
        }

    def point(self, scene_id: str, *, ended: bool = True) -> Position:
        """Scene `scene_id` on the story axis (`scene_point`).

        Raises:
            ValueError: the outline has no scene `scene_id`.
        """
        placement = self.placements.get(scene_id)
        if placement is None:
            raise ValueError(f"the outline has no scene '{scene_id}'")
        return scene_point(placement, ended=ended)

    def rows_at(self, point: Position, *, proposed: bool = False) -> list[EdgeRow]:
        """The edges in the view of `point` on the story axis, with their records."""
        return [r for r in self._rows if holds_at(r.edge, STORY_AXIS, point, proposed=proposed)]

    def edges_at(self, axis: Axis, point: Position, *, proposed: bool = False) -> list[Edge]:
        return edges_at(self.edges(), axis, point, proposed=proposed)

    def facts_at(
        self, axis: Axis, point: Position, *, proposed: bool = False
    ) -> list[tuple[str, str, str]]:
        return facts_at(self.edges(), axis, point, proposed=proposed)


def _entity(scope: str, item: CodexItem) -> Entity:
    entry = item.entry
    return Entity(
        id=entry.id,
        scope=scope,
        kind=item.kind,
        name=entry.name,
        aliases=tuple(entry.aliases),
        summary=entry.profile,
    )


def pending_aliases(proposals: Sequence[Proposal]) -> dict[str, list[str]]:
    """The names pending proposals add to each entry, by its id, in proposal order."""
    out: dict[str, list[str]] = {}
    for proposal in proposals:
        if proposal.status == "pending" and proposal.change.aliases is not None:
            out.setdefault(proposal.entry_id, []).extend(proposal.change.aliases)
    return out


def with_pending_aliases(item: CodexItem, aliases: Mapping[str, Sequence[str]]) -> CodexItem:
    """`item` with the names pending proposals add to it among its aliases."""
    entry = item.entry
    added = [a for a in aliases.get(entry.id, ()) if a not in entry.aliases and a != entry.name]
    if not added:
        return item
    named = entry.model_copy(update={"aliases": [*entry.aliases, *dict.fromkeys(added)]})
    return CodexItem(kind=item.kind, entry=named)


def _pending_changes(
    proposals: Sequence[Proposal], placements: Mapping[str, Placement]
) -> _PendingChanges:
    """Every pending change to an entry's value, by entry and value, in proposal order.

    Each proposal's turn is its index in `proposals`, as `CodexYamlStore.read` numbers it.
    """
    out: _PendingChanges = {}
    for order, proposal in enumerate(proposals):
        progression = proposal.change.progression
        if proposal.status != "pending" or progression is None:
            continue
        if progression.at not in placements:
            continue
        point = scene_point(placements[progression.at], ended=True)
        for section, key, _ in _sets(progression):
            by_value = out.setdefault(proposal.entry_id, {})
            by_value.setdefault((section, key), []).append(_Pending(point, order))
    return out


def _proposed_change(
    entry_id: str,
    progression: Progression,
    placements: Mapping[str, Placement],
    changes: Mapping[tuple[str, str], list[Position]],
    out: _Builder,
    pending: Mapping[tuple[str, str], Sequence[_Pending]],
    order: int,
) -> None:
    """A pending (or rejected) change to an entry: from its scene's end until the entry's
    own next change of that value, as it would hold once approved (it is placed after the
    entry's own). A pending one also ends at the next pending change of that value
    (`pending`); a pending removal makes no edge, and ends the value it removes there."""
    if progression.at not in placements:
        return
    start = scene_point(placements[progression.at], ended=True)
    for section, key, value in _sets(progression):
        if value is None:
            continue
        later = [p for p in changes.get((section, key), ()) if p > start]
        end = min(later, default=None)
        ended = _ended_by_pending(pending.get((section, key), ()), start, order=order)
        out.add(
            entry_id,
            section,
            key,
            _Cell(value, start, progression.at),
            end,
            _earliest(end, ended),
        )
