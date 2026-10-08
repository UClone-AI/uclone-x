"""The story codex read as a uGraph store: the YAML stays the source, and is read as edges.

The uGraph design, §3.2. A person edits the codex as YAML files, and nothing here writes
one: an entry file, a proposal file and the outline are read, and the store is what they
say, in the shared fact model (`uclone_x.knowledge`). Writing stays where it was: a
proposal is a file, and approving it writes the entry file (`uclone_x.story.proposals`).

How the files become the model:

- each codex entry is an `Entity` (its kind is its folder, its summary its profile);
- each `state` value is an edge `(entry, key, value)`; a list is one edge per element. A
  value that is a set of named fields, or `null`, is kept as an edge whose record says it
  is not a single value (`record["single"]` false), so a reader can say it did not check
  it rather than drop it;
- each `relations` value is an edge `(entry, word, other entry)`: an edge between two
  entities, `object_id` the other's id;
- a progression ends the edge it replaces and starts the new one, both at the scene's
  place in story time (`scene_point`). An entry's own values hold from the story's start;
- a **pending** proposal is the same edges with status `proposed`: a progression's from
  its scene until the entry's next change of that value, a new entry's from the start. A
  **rejected** one is `retracted`, with `expired_at` the time it was decided. An applied
  one is already in the entry file, so it is not read twice.

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
from uclone_x.story.context import CodexIndex, CodexItem
from uclone_x.story.schemas import CodexEntry, Outline, Progression, Proposal
from uclone_x.story.timeline import Placement, place_scenes

__all__ = [
    "STORY_AXIS",
    "CodexYamlStore",
    "scene_point",
    "story_scope",
]

STORY_AXIS: Final[Axis] = "story"

_STARTS: Final = 0
_ENDED: Final = 1


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


@dataclass
class _Builder:
    """The edges of one source (an entry file, or a proposal), in the order they open."""

    scope: str
    status: EdgeStatus
    recorded_at: str
    origin: dict[str, Any]
    expired_at: str | None = None
    evidence: tuple[tuple[str, str], ...] = ()
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
    ) -> None:
        if cell.start is not None and end is not None and not cell.start < end:
            return  # replaced at the same moment it was set: it never held
        order = self._order.setdefault((section, key), len(self._order))
        origin = {**self.origin, "field": f"{section}.{key}"}
        if cell.set_at is not None:
            origin["set_by_progression_at"] = cell.set_at
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
                evidence=self.evidence,
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


def _entry_edges(
    entry: CodexEntry, placements: Mapping[str, Placement], out: _Builder
) -> dict[tuple[str, str], list[Position]]:
    """Every edge `entry` holds over the story, into `out`; returned, where each of its
    values changes (for a proposed change to end at the next one)."""
    cells: dict[tuple[str, str], _Cell] = {}
    for key, value in entry.state.items():
        cells[("state", key)] = _Cell(value, None, None)
    for other, word in entry.relations.items():
        cells[("relations", other)] = _Cell(word, None, None)
    changes: dict[tuple[str, str], list[Position]] = {}
    for point, _, progression in _placed(entry.progressions, placements):
        sets = [("state", k, v) for k, v in progression.set.items()]
        sets += [("relations", k, v) for k, v in progression.relations.items()]
        for section, key, value in sets:
            changes.setdefault((section, key), []).append(point)
            held = cells.pop((section, key), None)
            if held is not None:
                out.add(entry.id, section, key, held, point)
            if value is not None:
                cells[(section, key)] = _Cell(value, point, progression.at)
    for (section, key), held in cells.items():
        out.add(entry.id, section, key, held, None)
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
    ) -> None:
        self._scope = story_scope(story_id)
        self.placements: Mapping[str, Placement] = placements
        self._entities = tuple(entities)
        self._rows = tuple(rows)

    @classmethod
    def read(
        cls,
        story_id: str,
        outline: Outline,
        codex: CodexIndex,
        proposals: Iterable[Proposal] = (),
    ) -> CodexYamlStore:
        """The store `outline`, `codex` and `proposals` make. A proposal that is neither
        pending nor rejected is in its entry file already, and is not read again."""
        scope = story_scope(story_id)
        placements = place_scenes(outline)
        entities: list[Entity] = []
        rows: list[EdgeRow] = []
        changes: dict[str, dict[tuple[str, str], list[Position]]] = {}
        for item in codex.items:
            entities.append(_entity(scope, item))
            built = _Builder(
                scope=scope,
                status="approved",
                recorded_at="",
                origin={"from": "codex", "file": f"codex/{item.kind}/{item.entry.id}.yaml"},
                id_prefix=f"{item.kind}:",
            )
            changes[item.entry.id] = _entry_edges(item.entry, placements, built)
            rows += built.ordered()
        known = {e.id for e in entities}
        for proposal in proposals:
            if proposal.status == "applied":
                continue
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
                    entities.append(_entity(scope, CodexItem(kind=proposal.kind, entry=entry)))
                _entry_edges(entry, placements, built)
            elif proposal.change.progression is not None:
                _proposed_change(
                    proposal.entry_id,
                    proposal.change.progression,
                    placements,
                    changes.get(proposal.entry_id, {}),
                    built,
                )
            rows += built.ordered()
        return cls(story_id, placements, entities, rows)

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
        return EntityResolver(self._scope, self._entities).resolve(name)

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


def _proposed_change(
    entry_id: str,
    progression: Progression,
    placements: Mapping[str, Placement],
    changes: Mapping[tuple[str, str], list[Position]],
    out: _Builder,
) -> None:
    """A pending change to an entry: from its scene's end until the entry's own next change
    of that value, as it would hold once approved (it is placed after the entry's own)."""
    if progression.at not in placements:
        return
    start = scene_point(placements[progression.at], ended=True)
    sets = [("state", k, v) for k, v in progression.set.items()]
    sets += [("relations", k, v) for k, v in progression.relations.items()]
    for section, key, value in sets:
        if value is None:
            continue
        later = [p for p in changes.get((section, key), ()) if p > start]
        out.add(
            entry_id, section, key, _Cell(value, start, progression.at), min(later, default=None)
        )
