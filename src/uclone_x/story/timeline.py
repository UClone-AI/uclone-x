"""When each scene happens, and what the codex says is true at that moment (#1557).

A book is read in one order and happens in another: a flashback in chapter 6 happens before
chapter 1. A progression (`at: ch03.s04`) is in force **once the scene it names has ended in
story time**, so the state a scene sees is the entry's starting state plus every progression
placed at a scene that happens before it -- in `story_time` order, not reading order.

How scenes are ordered:

- `story_time` is compared naturally: `day 3` comes before `day 10`, `5` equals `"5"`,
  and `-5` comes before `3`. Dotted numbers compare part by part, so `1.2.1` comes after
  `1.2` and before `1.10`, and `2024.9.30` before `2024.10.01`: a dot is never a decimal
  point. A number of any length compares by its digits, so no `story_time` is too long
  to order (#1601).
- A scene without a `story_time` happens when the scene before it in reading order does
  (it continues it). A scene with none and no timed scene before it happens first. Each
  such placement is an **assumption**, and `assumptions` lists them so the context can say
  so rather than decide in silence (P6).
- Scenes at the same time are in reading order.

This module is pure: it is given the outline and the entries and reads nothing itself.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from uclone_x.story.schemas import (
    CharacterEntry,
    CodexEntry,
    Outline,
    Progression,
    VisualProgression,
)

__all__ = [
    "EntrySnapshot",
    "Placement",
    "assumptions",
    "entry_snapshot",
    "place_scenes",
    "time_key",
]

#: A number's place in the order: its sign class (negative, zero, positive), then its
#: length and digits, arranged so that tuple order is numeric order (`_number`).
NumberKey = tuple[int, int, str]
TimeKey = tuple[tuple[int, NumberKey | str], ...]

#: A number, with its sign and dotted parts, or a run of anything else. A `-` that is not
#: part of a number is text.
_TOKENS = re.compile(r"-?\d+(?:\.\d+)*|[^\d-]+|-")


#: A digit's complement (`0` <-> `9`), so a longer or larger negative number sorts first.
_COMPLEMENT = str.maketrans("0123456789", "9876543210")


def _number(digits: str, negative: bool) -> NumberKey:
    """A sort key for the whole number `digits`, compared by its digits, not by `int`.

    `int` refuses a numeral of more than 4300 digits, and a `story_time` is free text
    (#1601). Leading zeros go, so `01` equals `1` and `-0` equals `0`; a digit of any
    script counts as its value, so `１２` equals `12`.
    """
    plain = "".join(str(unicodedata.decimal(ch)) for ch in digits).lstrip("0")
    if not plain:
        return (1, 0, "")
    if negative:
        return (0, -len(plain), plain.translate(_COMPLEMENT))
    return (2, len(plain), plain)


def _number_parts(token: str, signed: bool) -> list[NumberKey]:
    """The numbers one numeric token stands for, its first carrying the sign.

    A dot separates parts compared one by one, whatever their number, so `1.1 < 1.2 <
    1.2.1 < 1.5 < 1.10` and `2024.9.30 < 2024.10.01`: a dot is never a decimal point.
    """
    parts = token.split(".")
    return [_number(part, signed and not index) for index, part in enumerate(parts)]


def time_key(value: str | int | None) -> TimeKey:
    """A sort key for a `story_time`: numbers compare as numbers, the rest as text.

    A number may carry a sign, and its dotted parts compare one by one (`_number_parts`):
    `-5` comes before `3`, and `1.5` before `1.10`. A `-` is a minus sign only at the start
    or after white space and just before a digit, so the dashes in `2024-05-01` stay
    separators. `None` sorts before every time, which is where a scene with no time and no
    timed scene before it is placed.
    """
    if value is None:
        return ()
    if isinstance(value, int):
        return ((0, _number(str(abs(value)), value < 0)),)
    text = value.strip()
    key: list[tuple[int, NumberKey | str]] = []
    pending = ""

    def flush() -> None:
        nonlocal pending
        words = pending.strip().casefold()
        if words:
            key.append((1, words))
        pending = ""

    for match in _TOKENS.finditer(text):
        token = match.group()
        # `isdecimal`, the digits `\d` matches: `²` is a digit to `isdigit` and not a number.
        if not (token[0].isdecimal() or (token[0] == "-" and len(token) > 1)):
            pending += token
            continue
        signed = token[0] == "-"
        if signed and match.start() > 0 and not text[match.start() - 1].isspace():
            # A dash inside a word or a date is not a minus sign.
            pending += "-"
            token, signed = token[1:], False
        flush()
        numbers = _number_parts(token.lstrip("-"), signed)
        for index, number in enumerate(numbers):
            if index:
                key.append((1, "."))
            key.append((0, number))
    flush()
    return tuple(key)


@dataclass(frozen=True)
class Placement:
    """Where one scene sits in story time."""

    scene_id: str
    reading_index: int
    key: TimeKey
    #: The scene whose `story_time` this one took, when it has none of its own.
    inherited_from: str | None = None
    #: True when the scene has no time and no timed scene comes before it.
    untimed: bool = False

    @property
    def position(self) -> tuple[TimeKey, int]:
        """The order scenes happen in: time first, then reading order."""
        return (self.key, self.reading_index)


def place_scenes(outline: Outline) -> dict[str, Placement]:
    """Every scene of `outline`, placed in story time."""
    placements: dict[str, Placement] = {}
    current: TimeKey | None = None
    source: str | None = None
    for index, (_, scene) in enumerate(outline.scenes_in_order()):
        if scene.story_time is not None:
            current = time_key(scene.story_time)
            source = scene.id
            placements[scene.id] = Placement(scene.id, index, current)
        elif current is None:
            placements[scene.id] = Placement(scene.id, index, (), untimed=True)
        else:
            placements[scene.id] = Placement(scene.id, index, current, inherited_from=source)
    return placements


def assumptions(placements: Mapping[str, Placement]) -> list[dict[str, Any]]:
    """The scenes placed without a `story_time` of their own, and where they were put."""
    out: list[dict[str, Any]] = []
    for placement in sorted(placements.values(), key=lambda p: p.reading_index):
        if placement.untimed:
            out.append(
                {
                    "scene_id": placement.scene_id,
                    "placed": "first: it has no story_time and no scene before it has one",
                }
            )
        elif placement.inherited_from is not None:
            out.append(
                {
                    "scene_id": placement.scene_id,
                    "placed": f"at the same time as '{placement.inherited_from}': it has no "
                    "story_time of its own",
                }
            )
    return out


@dataclass(frozen=True)
class EntrySnapshot:
    """An entry as it stands at one moment of the story."""

    state: dict[str, Any]
    #: The character's visual tags at that moment; `None` for an entry with no visual block.
    visual_tags: list[str] | None
    #: The progressions in force, in the order they were applied.
    applied: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    #: The progressions placed at the scene itself (the change it shows happening).
    in_this_scene: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    #: The progressions whose `at` is not a scene of the outline, so they are not applied.
    not_placed: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    #: For each state key a progression set, the scene of the progression that set it last;
    #: a key not here holds the entry's starting value.
    set_at: dict[str, str] = field(default_factory=dict[str, str])


def _described(at: str, kind: str, note: str | None, change: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"at": at, "kind": kind, **change}
    if note:
        out["note"] = note
    return out


def entry_snapshot(
    entry: CodexEntry,
    placements: Mapping[str, Placement],
    scene_id: str,
    *,
    through_scene: bool,
) -> EntrySnapshot:
    """`entry` as it stands at `scene_id`.

    With `through_scene` false, the state is the one the scene *starts* from: progressions
    at scenes that happen before it. The scene's own progressions are listed apart, as the
    change it shows. With `through_scene` true they are applied too: the state once the
    scene has ended, which is what a scene's facts are checked against.

    Raises:
        ValueError: `scene_id` is not placed (the caller checks the outline first).
    """
    if scene_id not in placements:
        raise ValueError(f"the outline has no scene '{scene_id}'")
    here = placements[scene_id]

    visual = entry.visual if isinstance(entry, CharacterEntry) else None
    changes: list[tuple[str, Progression | VisualProgression]] = [
        ("state", p) for p in entry.progressions
    ]
    if visual is not None:
        changes += [("visual", p) for p in visual.progressions]

    ready: list[tuple[tuple[TimeKey, int], int, Progression | VisualProgression, dict[str, Any]]]
    ready = []
    not_placed: list[dict[str, Any]] = []
    in_this_scene: list[dict[str, Any]] = []
    for order, (kind, progression) in enumerate(changes):
        if isinstance(progression, Progression):
            change: dict[str, Any] = {"set": dict(progression.set)}
        else:
            change = {
                "add_tags": list(progression.add_tags),
                "remove_tags": list(progression.remove_tags),
            }
        described = _described(progression.at, kind, progression.note, change)
        placement = placements.get(progression.at)
        if placement is None:
            not_placed.append(described)
            continue
        if placement.scene_id == scene_id:
            in_this_scene.append(described)
            if not through_scene:
                continue
        elif placement.position > here.position:
            continue
        ready.append((placement.position, order, progression, described))

    state: dict[str, Any] = dict(entry.state)
    tags: list[str] | None = list(visual.tags) if visual is not None else None
    applied: list[dict[str, Any]] = []
    set_at: dict[str, str] = {}
    for _, _, progression, described in sorted(ready, key=lambda step: (step[0], step[1])):
        if isinstance(progression, Progression):
            for key, value in progression.set.items():
                if value is None:
                    state.pop(key, None)
                else:
                    state[key] = value
                set_at[key] = progression.at
        elif tags is not None:
            removed = set(progression.remove_tags)
            tags = [t for t in tags if t not in removed]
            tags += [t for t in progression.add_tags if t not in tags]
        applied.append(described)
    return EntrySnapshot(state, tags, applied, in_this_scene, not_placed, set_at)
