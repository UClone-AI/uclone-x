"""Temporal Knowledge Graph (TKG) state-transition planning and validation for stories (#2358).

In the text-first flow, outlines and chapter plans hold only prose summaries (`summary`,
`beats`), leaving `EntrySnapshot.in_this_scene` empty when a scene is drafted and relying on
post-hoc fact extraction (`grow_scene`) after the prose is already written.

This module implements the TKG state-transition core (`G_0 -> G_1 -> ... -> G_n`):

1. `effective_codex`: builds a `CodexIndex` that combines approved codex files, pending
   `new_entry` proposals, pending `aliases` proposals, pending `progression` proposals from
   earlier scenes (`grow_scene`), and planned scene transitions (`SceneTransition`). Passing
   this index to `scene_context` populates both `progressions_applied` ($G_{t-1}$) and
   `changes_in_this_scene` ($\\Delta G_t$) before a scene is drafted.
2. `parse_plot_transitions`, `parse_detail_transitions`, `thread_transitions`: extract
   scene-level state and relation deltas ($\\Delta G_t$) from plot-lock and chapter-detail
   plans, mapping character, item and thread names to codex entry ids.
3. `validate_kg_trajectory`: deterministically simulates $G_0 \\to G_1 \\to \\dots \\to G_n$ in
   story-time order (`place_scenes`), dropping transitions on dead actors (`dead_actor`),
   stripping immutable identity keys (`immutable_identity`), stripping state/relation values
   that already hold in $G_{t-1}$ (`redundant_transition`), and blocking premature twist or
   conflict payoff (`premature_twist`, `premature_resolution`).
4. `kg_scene_guidance`: renders the entering world state ($G_{t-1}$) and the required target
   transitions ($\\Delta G_t$) with entity names for the Writer's scene prompt.

This module is pure: it touches no files and calls no model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import JsonValue, ValidationError

from uclone_x.knowledge.fold import bigram_overlap
from uclone_x_story.arc import Arc
from uclone_x_story.codex_store import pending_aliases, with_pending_aliases
from uclone_x_story.context import CodexIndex, CodexItem
from uclone_x_story.schemas import (
    ENTRY_ID_PATTERN,
    CharacterEntry,
    CodexKind,
    Outline,
    PlaceEntry,
    Progression,
    Proposal,
    ThreadEntry,
    Visual,
    VisualChange,
    VisualProgression,
    status_of,
)
from uclone_x_story.timeline import EntrySnapshot, entry_snapshot, place_scenes

__all__ = [
    "SceneTransition",
    "TrajectoryReport",
    "candidate_names",
    "chronological_kg_log",
    "effective_codex",
    "entering_world_state",
    "format_chronological_kg_log",
    "kg_scene_guidance",
    "merge_transitions",
    "parse_detail_transitions",
    "parse_plot_transitions",
    "parse_scene_transitions",
    "split_entity_name",
    "thread_transitions",
    "unplaced_transitions",
    "validate_kg_trajectory",
]

_ENTRY_ID_RE = re.compile(ENTRY_ID_PATTERN)
_IMMUTABLE_KEYS: frozenset[str] = frozenset({"gender"})
_NARRATIVE_STATE_KEYS: frozenset[str] = frozenset(
    {"stance", "goal", "condition", "milestone", "knowledge"}
)
#: What a thread's planned stage says in the Writer's scene prompt, in the story's
#: language (#2374): Korean words in an English story's prompt draw Korean prose.
_THREAD_NOTES: Mapping[str, Mapping[str, str]] = {
    "ko": {"planted": "{name}: 단서/갈등 제시", "paid_off": "{name}: 회수/결말"},
    "en": {"planted": "{name}: set up as a clue or a conflict", "paid_off": "{name}: paid off"},
}


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _valid_entry_id(value: str) -> bool:
    return bool(value) and len(value) <= 80 and bool(_ENTRY_ID_RE.match(value))


def _clean_relation_word(value: object) -> str | None:
    word = _text(value)
    if not word:
        return None
    return word[:80]


def _clean_state_value(value: object) -> JsonValue | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, list):
        items = [
            s
            for item in cast("list[object]", value)
            if isinstance(item, str) and (s := item.strip())
        ]
        return cast("JsonValue", items) if items else None
    return None


@dataclass(frozen=True)
class SceneTransition:
    """One planned state/relation change ($\\Delta G_t$) on `entity_id` at the end of `scene_id`."""

    scene_id: str
    entity_id: str
    kind: CodexKind = "characters"
    set: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    relations: dict[str, str | None] = field(default_factory=dict[str, str | None])
    note: str | None = None

    def is_empty(self) -> bool:
        return not self.set and not self.relations

    def edge_count(self) -> int:
        return len(self.set) + len(self.relations)

    def to_progression(self) -> Progression | None:
        if self.is_empty() or not _valid_entry_id(self.scene_id):
            return None
        clean_relations = {
            target: word
            for target, word in self.relations.items()
            if _valid_entry_id(target) and (word is None or len(word.strip()) >= 1)
        }
        if not self.set and not clean_relations:
            return None
        try:
            return Progression(
                at=self.scene_id,
                set=dict(self.set),
                relations=clean_relations,
                note=self.note or None,
            )
        except ValidationError:
            return None

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scene_id": self.scene_id,
            "entity_id": self.entity_id,
            "kind": self.kind,
        }
        if self.set:
            out["set"] = dict(self.set)
        if self.relations:
            out["relations"] = dict(self.relations)
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_record(cls, raw: Mapping[str, Any]) -> SceneTransition | None:
        scene_id = _text(raw.get("scene_id"))
        entity_id = _text(raw.get("entity_id"))
        kind_raw = _text(raw.get("kind")) or "characters"
        if not (_valid_entry_id(scene_id) and _valid_entry_id(entity_id)):
            return None
        if kind_raw not in ("characters", "places", "items", "threads"):
            return None
        raw_set = raw.get("set")
        state_set: dict[str, JsonValue] = {}
        if isinstance(raw_set, dict):
            for k, v in cast("dict[object, object]", raw_set).items():
                key = _text(k)
                cleaned = _clean_state_value(v)
                if key and cleaned is not None:
                    state_set[key] = cleaned
        raw_rel = raw.get("relations")
        relations: dict[str, str | None] = {}
        if isinstance(raw_rel, dict):
            for k, v in cast("dict[object, object]", raw_rel).items():
                target = _text(k)
                if not _valid_entry_id(target):
                    continue
                if v is None:
                    relations[target] = None
                elif word := _clean_relation_word(v):
                    relations[target] = word
        note = _text(raw.get("note")) or None
        transition = cls(
            scene_id=scene_id,
            entity_id=entity_id,
            kind=kind_raw,
            set=state_set,
            relations=relations,
            note=note,
        )
        return None if transition.is_empty() else transition


@dataclass(frozen=True)
class TrajectoryReport:
    """Summary of deterministic simulation over $G_0 \\to G_1 \\to \\dots \\to G_n$."""

    total_scenes: int
    active_scenes: tuple[str, ...]
    stagnant_scenes: tuple[str, ...]
    edge_deltas: int
    dropped: tuple[dict[str, str], ...] = ()

    @property
    def transition_density(self) -> float:
        return self.edge_deltas / self.total_scenes if self.total_scenes > 0 else 0.0

    @property
    def active_ratio(self) -> float:
        return len(self.active_scenes) / self.total_scenes if self.total_scenes > 0 else 0.0

    def record(self) -> dict[str, Any]:
        return {
            "total_scenes": self.total_scenes,
            "active_scenes": list(self.active_scenes),
            "stagnant_scenes": list(self.stagnant_scenes),
            "edge_deltas": self.edge_deltas,
            "transition_density": round(self.transition_density, 3),
            "active_ratio": round(self.active_ratio, 3),
            "dropped": [dict(d) for d in self.dropped],
        }


def _merge_progression_list(
    existing: Sequence[Progression], added: Sequence[Progression]
) -> list[Progression]:
    """`existing` plus `added`, coalescing progressions at the same `at` scene."""
    if not added:
        return list(existing)
    by_scene: dict[str, Progression] = {}
    order: list[str] = []
    for prog in (*existing, *added):
        if prog.at not in by_scene:
            by_scene[prog.at] = prog
            order.append(prog.at)
            continue
        prev = by_scene[prog.at]
        merged_set = {**prev.set, **prog.set}
        merged_rel = {**prev.relations, **prog.relations}
        notes = [n for n in (prev.note, prog.note) if n]
        merged_note = "; ".join(dict.fromkeys(notes)) if notes else None
        by_scene[prog.at] = prev.model_copy(
            update={"set": merged_set, "relations": merged_rel, "note": merged_note}
        )
    return [by_scene[at] for at in order]


def effective_codex(
    codex: CodexIndex,
    proposals: Iterable[Proposal] = (),
    planned_transitions: Iterable[Mapping[str, Any] | SceneTransition] = (),
    *,
    rejected: set[str] | None = None,
) -> CodexIndex:
    """The story's full temporal `CodexIndex` combining approved entries, pending proposals,
    and planned scene transitions.

    Unlike `StoryWork.codex()` (which reads only approved files under `codex/`) or
    `CodexYamlStore.codex()` (which adds pending `new_entry` proposals without attaching
    pending `progression` proposals onto `entry.progressions`), `effective_codex` attaches
    both pending `progression` proposals (e.g. from `grow_scene`) and validated
    `planned_transitions` onto each `CodexEntry.progressions`. When passed to
    `scene_context`, `timeline.entry_snapshot` computes both `applied` ($G_{t-1}$) and
    `in_this_scene` ($\\Delta G_t$).
    """
    excluded = rejected or set()
    all_proposals = [p for p in proposals if p.id not in excluded and p.status == "pending"]
    aliases_by_id = pending_aliases(all_proposals)
    progressions_by_id: dict[str, list[Progression]] = {}
    visual_progs_by_id: dict[str, list[VisualProgression]] = {}
    visual_changes_by_id: dict[str, list[VisualChange]] = {}

    for proposal in all_proposals:
        if proposal.change.progression is not None:
            progressions_by_id.setdefault(proposal.entry_id, []).append(proposal.change.progression)
        if proposal.change.visual_progression is not None:
            visual_progs_by_id.setdefault(proposal.entry_id, []).append(
                proposal.change.visual_progression
            )
        if proposal.change.visual is not None:
            visual_changes_by_id.setdefault(proposal.entry_id, []).append(proposal.change.visual)

    for item in planned_transitions:
        transition = (
            item if isinstance(item, SceneTransition) else SceneTransition.from_record(item)
        )
        if transition is None:
            continue
        prog = transition.to_progression()
        if prog is not None:
            progressions_by_id.setdefault(transition.entity_id, []).append(prog)

    items: list[CodexItem] = []
    known: set[str] = set()

    for codex_item in codex.items:
        known.add(codex_item.entry.id)
        items.append(
            _enrich_codex_item(
                codex_item,
                aliases_by_id,
                progressions_by_id,
                visual_progs_by_id,
                visual_changes_by_id,
            )
        )

    for proposal in all_proposals:
        if proposal.change.new_entry is None:
            continue
        try:
            entry = proposal.new_entry()
        except ValueError:
            continue
        if entry.id in known:
            continue
        known.add(entry.id)
        # Mark core plot threads (`central_conflict`, `twist`) as `always_include` when they
        # are materialized from proposals so `scene_context` includes their state in every scene.
        if isinstance(entry, ThreadEntry) and entry.id in ("central_conflict", "twist"):
            entry = entry.model_copy(update={"always_include": True})
        items.append(
            _enrich_codex_item(
                CodexItem(kind=proposal.kind, entry=entry),
                aliases_by_id,
                progressions_by_id,
                visual_progs_by_id,
                visual_changes_by_id,
            )
        )

    return CodexIndex(items=tuple(items), unreadable=codex.unreadable)


def _enrich_codex_item(
    item: CodexItem,
    aliases_by_id: Mapping[str, Sequence[str]],
    progressions_by_id: Mapping[str, Sequence[Progression]],
    visual_progs_by_id: Mapping[str, Sequence[VisualProgression]] = {},
    visual_changes_by_id: Mapping[str, Sequence[VisualChange]] = {},
) -> CodexItem:
    """`item` with the names pending proposals add (`codex_store.with_pending_aliases`) and
    its pending and planned progressions (including character visual progressions)."""
    named = with_pending_aliases(item, aliases_by_id)
    entry = named.entry
    extra_progs = progressions_by_id.get(entry.id, ())
    extra_vprogs = visual_progs_by_id.get(entry.id, ())
    extra_vchanges = visual_changes_by_id.get(entry.id, ())
    if isinstance(entry, CharacterEntry) and (extra_vprogs or extra_vchanges):
        visual = entry.visual or Visual()
        for vchange in extra_vchanges:
            visual = visual.model_copy(update=vchange.model_dump(exclude_none=True))
        if extra_vprogs:
            visual = visual.model_copy(
                update={"progressions": [*visual.progressions, *extra_vprogs]}
            )
        entry = entry.model_copy(update={"visual": visual})
        named = CodexItem(kind=named.kind, entry=entry)
    if not extra_progs:
        return named
    merged = _merge_progression_list(entry.progressions, extra_progs)
    return CodexItem(kind=named.kind, entry=entry.model_copy(update={"progressions": merged}))


def _entity_lookup(
    cast_lines: Sequence[Mapping[str, Any]],
    codex: CodexIndex | None = None,
) -> dict[str, tuple[str, CodexKind]]:
    """Map folded entity name or id to `(entry_id, kind)`."""
    lookup: dict[str, tuple[str, CodexKind]] = {}
    for line in cast_lines:
        entry_id = _text(line.get("id"))
        name = _text(line.get("name"))
        kind_raw = _text(line.get("kind")) or "characters"
        if not (
            _valid_entry_id(entry_id) and kind_raw in ("characters", "places", "items", "threads")
        ):
            continue
        kind: CodexKind = kind_raw
        lookup[entry_id.casefold()] = (entry_id, kind)
        if name:
            lookup[name.casefold()] = (entry_id, kind)
        aliases = line.get("aliases")
        if isinstance(aliases, list):
            for alias in cast("list[object]", aliases):
                if isinstance(alias, str) and alias.strip():
                    lookup[alias.strip().casefold()] = (entry_id, kind)
    if codex is not None:
        for item in codex.items:
            lookup[item.entry.id.casefold()] = (item.entry.id, item.kind)
            lookup[item.entry.name.casefold()] = (item.entry.id, item.kind)
            for alias in item.entry.aliases:
                if alias.strip():
                    lookup[alias.strip().casefold()] = (item.entry.id, item.kind)
    return lookup


_EXACT_STATUS_WORDS: frozenset[str] = frozenset(
    {
        "alive",
        "dead",
        "missing",
        "unknown",
        "생존",
        "살아있음",
        "살아 있음",
        "살아남음",
        "사망",
        "죽음",
        "숨짐",
        "전사",
        "실종",
        "행방불명",
        "실종됨",
        "사라짐",
        "불명",
        "생사불명",
        "알 수 없음",
        "알수없음",
    }
)


def _has_status_word(value: str) -> bool:
    """Whether `value` is a standalone lifecycle status label (`alive`, `missing`, `dead`, `unknown`)
    rather than a narrative clause or condition (e.g. ``"죽을 만큼 괴로움"``, ``"기억을 잃고 이동"``)."""
    word = re.sub(r"[.!?,;:()\"'‘’“”\[\]]+", "", value).strip().casefold()
    return word in _EXACT_STATUS_WORDS


_OUTER_QUOTES = "'\"‘’“”「」『』"
_PAREN_ENTITY = re.compile(r"^(.+?)\s*\(([^()]+)\)$")
_QUOTED_ENTITY = re.compile(r"^(.+?)\s*['\"‘“「『]([^'\"’”」』]+)['\"’”」』]$")


def split_entity_name(raw: str) -> tuple[str, list[str]]:
    """Split a combined role/name string into `(primary_name, extra_aliases)`.

    Handles outer quotes (``"'먹물'"``), role + quoted name (``"검은 고양이 '먹물'"``),
    quoted name + parenthetical role (``"'먹물' (검은 고양이)"``), and name + parenthetical
    role (``"이준호 (시계 수리공)"``).
    """
    text = raw.strip()
    if not text:
        return "", []
    if m := _QUOTED_ENTITY.match(text):
        role = m.group(1).strip().strip(_OUTER_QUOTES).strip()
        proper = m.group(2).strip().strip(_OUTER_QUOTES).strip()
        if proper and role and proper.casefold() != role.casefold():
            return proper, [role]
    if m := _PAREN_ENTITY.match(text):
        head = m.group(1).strip().strip(_OUTER_QUOTES).strip()
        paren = m.group(2).strip().strip(_OUTER_QUOTES).strip()
        if head and paren and head.casefold() != paren.casefold():
            return head, [paren]
    stripped = text.strip(_OUTER_QUOTES).strip()
    return (stripped or text), []


def candidate_names(raw: str) -> tuple[str, ...]:
    """Candidate entity names from `raw`: exact string first, then stripped of outer quotes,
    then split into primary name and parenthetical/quoted role."""
    text = raw.strip()
    if not text:
        return ()
    out: list[str] = [text]
    unquoted = text.strip(_OUTER_QUOTES).strip()
    if unquoted and unquoted not in out:
        out.append(unquoted)
    primary, extras = split_entity_name(text)
    for item in (primary, *extras):
        if item and item not in out:
            out.append(item)
    return tuple(out)


def _lookup_entity(
    raw: str, lookup: Mapping[str, tuple[str, CodexKind]]
) -> tuple[str, CodexKind] | None:
    """Resolve `raw` against `lookup` using `candidate_names`."""
    for cand in candidate_names(raw):
        if (found := lookup.get(cand.casefold())) is not None:
            return found
    return None


def _scene_of_label(outline: Outline, label: object) -> str | None:
    """The outline scene the label the prompt showed names (`[ch01.s02]`), matched exactly;
    `None` for a label the outline does not have.

    The scene is never guessed from a chapter number or from the words of a note (#2204,
    #2374): a transition the reply does not place is not planned.
    """
    text = _text(label).strip("[] ")
    text = text.partition("]")[0].strip()
    return text if text and outline.find(text) is not None else None


def _plot_items(
    found: Mapping[str, Any] | None, outline: Outline
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int]]:
    """Each transition of the plot reading with the scene it is under, and how many were
    not placed, by why: `label_missing` (a listed one with no `scene`), `label_unknown`
    (a label the outline does not have), `not_a_transition` (a string or other value
    where a transition object belongs). One object under a label is a list of one.

    The reading is asked for its transitions keyed by each scene's label (`_kg_prompt`,
    #2378), so the scene is the key the code wrote; a flat list whose items carry a
    `scene` is read too.
    """
    placed: list[tuple[str, dict[str, Any]]] = []
    unplaced = {"label_missing": 0, "label_unknown": 0, "not_a_transition": 0}
    raw = found.get("transitions") if found is not None else None
    if isinstance(raw, dict):
        for label, value in cast("dict[object, object]", raw).items():
            listed: list[object] = cast("list[object]", value) if isinstance(value, list) else []
            listed = [value] if isinstance(value, dict) else listed
            entries = [cast("dict[str, Any]", e) for e in listed if isinstance(e, dict)]
            unplaced["not_a_transition"] += len(listed) - len(entries)
            scene_id = _scene_of_label(outline, label)
            if scene_id is None:
                unplaced["label_unknown"] += len(entries)
                continue
            placed += [(scene_id, e) for e in entries]
    elif isinstance(raw, list):
        for item in cast("list[object]", raw):
            if not isinstance(item, dict):
                unplaced["not_a_transition"] += 1
                continue
            entry = cast("dict[str, Any]", item)
            label = entry.get("scene") or entry.get("scene_id")
            scene_id = _scene_of_label(outline, label)
            if scene_id is not None:
                placed.append((scene_id, entry))
            elif _text(label):
                unplaced["label_unknown"] += 1
            else:
                unplaced["label_missing"] += 1
    return placed, unplaced


def _parse_single_transition(
    raw: Mapping[str, Any],
    scene_id: str,
    lookup: Mapping[str, tuple[str, CodexKind]],
) -> SceneTransition | None:
    who = _text(raw.get("character") or raw.get("entity") or raw.get("subject"))
    if not who:
        return None
    resolved = _lookup_entity(who, lookup)
    if resolved is None:
        return None
    entity_id, kind = resolved
    has_known_items = any(k == "items" for _, k in lookup.values())
    state_set: dict[str, JsonValue] = {}
    raw_state = raw.get("state") or raw.get("set")
    state_items: dict[object, object] = {}
    if isinstance(raw_state, dict):
        state_items = dict(cast("dict[object, object]", raw_state))
        if set(state_items.keys()) == {"key", "value"} and isinstance(state_items.get("key"), str):
            state_items = {state_items["key"]: state_items["value"]}
    for direct_key in ("stance", "goal", "knowledge", "possesses", "condition", "location"):
        if direct_key in raw and direct_key not in state_items:
            state_items[direct_key] = raw.get(direct_key)
    rel_items: dict[object, object] = {}
    raw_rel = raw.get("relations")
    if isinstance(raw_rel, dict):
        rel_items.update(cast("dict[object, object]", raw_rel))
    rel_target = state_items.pop("relation", None) or state_items.pop("relative", None)
    rel_is = state_items.pop("is", None)
    if isinstance(rel_target, str) and isinstance(rel_is, str):
        rel_items.setdefault(rel_target, rel_is)
    for k, v in state_items.items():
        key = _text(k)
        cleaned = _clean_state_value(v)
        if key and cleaned is not None:
            if key == "status" and (
                kind != "characters"
                or not isinstance(cleaned, str)
                or not _has_status_word(cleaned)
            ):
                key = "condition"
            if key == "possesses" and isinstance(cleaned, str) and has_known_items:
                item_match = _lookup_entity(cleaned, lookup)
                if item_match is None or item_match[1] != "items":
                    key = "knowledge"
            # A planned status is one of `STATUSES`, as a cast's is: "사망" is dead.
            cleaned = status_of(cleaned) if key == "status" else cleaned
            state_set[key] = cleaned
    relations: dict[str, str | None] = {}
    for k, v in rel_items.items():
        target_name = _text(k)
        target = _lookup_entity(target_name, lookup)
        if target is None or target[0] == entity_id:
            continue
        word = _clean_relation_word(v)
        if word:
            relations[target[0]] = word
    note = _text(raw.get("note") or raw.get("change")) or None
    if not state_set and not relations and note:
        state_set["milestone"] = note
    transition = SceneTransition(
        scene_id=scene_id,
        entity_id=entity_id,
        kind=kind,
        set=state_set,
        relations=relations,
        note=note,
    )
    return None if transition.is_empty() else transition


def parse_plot_transitions(
    found: Mapping[str, Any] | None,
    outline: Outline,
    cast_lines: Sequence[Mapping[str, Any]],
    codex: CodexIndex | None = None,
) -> list[SceneTransition]:
    """Parse scene transitions from the settled-plot KG reading (`_kg_prompt`)."""
    if found is None or not outline.chapters:
        return []
    lookup = _entity_lookup(cast_lines, codex)
    out: list[SceneTransition] = []
    for scene_id, raw in _plot_items(found, outline)[0]:
        parsed = _parse_single_transition(raw, scene_id, lookup)
        if parsed is not None:
            out.append(parsed)
    return out


def unplaced_transitions(found: Mapping[str, Any] | None, outline: Outline) -> dict[str, int]:
    """How many of the plot reading's transitions were not planned, by why (`_plot_items`):
    what the exact label match drops (#2376, #2378)."""
    return _plot_items(found, outline)[1]


def parse_detail_transitions(
    found: Mapping[str, Any] | None,
    outline: Outline,
    chapter_number: int,
    cast_lines: Sequence[Mapping[str, Any]],
    codex: CodexIndex | None = None,
) -> list[SceneTransition]:
    """Parse per-scene transitions from a chapter plan reply (`_propose_details` / `_revise_details`)."""
    if found is None or not 1 <= chapter_number <= len(outline.chapters):
        return []
    raw_scenes = found.get("scenes")
    if not isinstance(raw_scenes, list):
        return []
    chapter = outline.chapters[chapter_number - 1]
    lookup = _entity_lookup(cast_lines, codex)
    out: list[SceneTransition] = []
    for idx, sc_obj in enumerate(cast("list[object]", raw_scenes)):
        if idx >= len(chapter.scenes) or not isinstance(sc_obj, dict):
            continue
        sc = cast("dict[str, Any]", sc_obj)
        scene_id = chapter.scenes[idx].id
        raw_transitions = sc.get("transitions")
        if isinstance(raw_transitions, list):
            for tr_obj in cast("list[object]", raw_transitions):
                if not isinstance(tr_obj, dict):
                    continue
                parsed = _parse_single_transition(cast("dict[str, Any]", tr_obj), scene_id, lookup)
                if parsed is not None:
                    out.append(parsed)
    return out


_CONJOINED_SUBJECT_RE = re.compile(
    r"\s*(?:,|/|&|\band\b|와\s+|과\s+|및\s+|하고\s+)\s*",
    re.IGNORECASE,
)


def parse_scene_transitions(
    raw_transitions: object,
    scene_id: str,
    cast_lines: Sequence[Mapping[str, Any]],
    codex: CodexIndex | None = None,
) -> list[SceneTransition]:
    """Parse a list of transition dicts for a single `scene_id` (e.g. from an RPG scene choice)."""
    if not scene_id or not isinstance(raw_transitions, list):
        return []
    lookup = _entity_lookup(cast_lines, codex)
    out: list[SceneTransition] = []
    for tr_obj in cast("list[object]", raw_transitions):
        if not isinstance(tr_obj, dict):
            continue
        raw_dict = cast("dict[str, Any]", tr_obj)
        parsed = _parse_single_transition(raw_dict, scene_id, lookup)
        if parsed is not None:
            out.append(parsed)
            continue
        who = _text(raw_dict.get("character") or raw_dict.get("entity") or raw_dict.get("subject"))
        parts = [p.strip() for p in _CONJOINED_SUBJECT_RE.split(who) if p.strip()]
        if len(parts) >= 2:
            for part in parts:
                sub_dict = {**raw_dict, "subject": part, "character": part, "entity": part}
                sub_parsed = _parse_single_transition(sub_dict, scene_id, lookup)
                if sub_parsed is not None:
                    out.append(sub_parsed)
    return out


def thread_transitions(
    threads: Sequence[tuple[Any, Mapping[str, str | None]]],
    outline: Outline,
    *,
    id_for_name_fn: Callable[[str], str | None],
    language: str,
) -> list[SceneTransition]:
    """Deterministic lifecycle transitions (`planted` -> `paid_off`) for plot threads, their
    notes in the story's `language`."""
    notes = _THREAD_NOTES.get(language, _THREAD_NOTES["en"])
    out: list[SceneTransition] = []
    for params, fields in threads:
        kind = getattr(params, "kind", None)
        if kind != "threads":
            continue
        name = _text(getattr(params, "name", ""))
        entry_id = _text(getattr(params, "id", "")) or (
            (id_for_name_fn(name) or "") if name else ""
        )
        if not _valid_entry_id(entry_id):
            continue
        planted_in = _text(fields.get("planted_in"))
        pay_off_by = _text(fields.get("pay_off_by"))
        if planted_in and outline.find(planted_in) is not None:
            out.append(
                SceneTransition(
                    scene_id=planted_in,
                    entity_id=entry_id,
                    kind="threads",
                    set={"thread_stage": "planted"},
                    note=notes["planted"].format(name=name or entry_id),
                )
            )
        if pay_off_by and outline.find(pay_off_by) is not None and pay_off_by != planted_in:
            out.append(
                SceneTransition(
                    scene_id=pay_off_by,
                    entity_id=entry_id,
                    kind="threads",
                    set={"thread_stage": "paid_off"},
                    note=notes["paid_off"].format(name=name or entry_id),
                )
            )
    return out


def merge_transitions(
    existing: Sequence[Mapping[str, Any] | SceneTransition],
    added: Sequence[Mapping[str, Any] | SceneTransition],
    *,
    replace_scenes: set[str] | None = None,
) -> list[SceneTransition]:
    """Merge `existing` and `added` transitions, coalescing `(scene_id, entity_id)` pairs."""
    kept: list[SceneTransition] = []
    for raw in existing:
        tr = raw if isinstance(raw, SceneTransition) else SceneTransition.from_record(raw)
        if tr is None:
            continue
        if replace_scenes and tr.scene_id in replace_scenes and tr.kind != "threads":
            continue
        kept.append(tr)
    by_key: dict[tuple[str, str], SceneTransition] = {}
    order: list[tuple[str, str]] = []
    for raw in (*kept, *added):
        tr = raw if isinstance(raw, SceneTransition) else SceneTransition.from_record(raw)
        if tr is None:
            continue
        key = (tr.scene_id, tr.entity_id)
        if key not in by_key:
            by_key[key] = tr
            order.append(key)
            continue
        prev = by_key[key]
        merged_set = {**prev.set, **tr.set}
        merged_rel = {**prev.relations, **tr.relations}
        notes = [n for n in (prev.note, tr.note) if n]
        by_key[key] = SceneTransition(
            scene_id=tr.scene_id,
            entity_id=tr.entity_id,
            kind=tr.kind,
            set=merged_set,
            relations=merged_rel,
            note="; ".join(dict.fromkeys(notes)) if notes else None,
        )
    return [by_key[k] for k in order]


def validate_kg_trajectory(
    outline: Outline,
    codex: CodexIndex,
    transitions: Sequence[Mapping[str, Any] | SceneTransition],
    *,
    arc: Arc | None = None,
    cast_lines: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[SceneTransition], TrajectoryReport]:
    """Deterministically simulate $G_0 \\to G_1 \\to \\dots \\to G_n$ across `outline` in
    story-time order and filter out invalid or redundant transitions.

    Checks enforced:
    - `unknown_scene` / `unknown_entity`: transition references a missing scene or entity.
    - `kind_mismatch`: the transition's kind is not its entry's (a thread's stage on a place).
    - `dead_actor`: a character dead in $G_{t-1}$ (`status_of`) cannot perform state or
      relation transitions in scene $s_t$.
    - `immutable_identity`: fixed identity keys (`gender`) cannot be overwritten.
    - `premature_twist` / `premature_resolution`: `twist` cannot transition to `paid_off`
      before `arc.twist_chapter`, and `central_conflict` cannot transition to `paid_off`
      before `arc.climax_chapter`.
    - `redundant_transition`: state or relation values that already hold in $G_{t-1}$ are
      stripped so consecutive scenes do not repeat the same transition.

    The state at each scene is `timeline.entry_snapshot` of the entry with the transitions
    kept so far added to its progressions: the codex's own progressions (approved and
    pending) count, by the same rule `scene_context` and the codex store read them by.
    """
    placements = place_scenes(outline)
    ordered_scenes = [p.scene_id for p in sorted(placements.values(), key=lambda pl: pl.position)]
    chapter_of_scene: dict[str, int] = {
        scene.id: ch_idx
        for ch_idx, chapter in enumerate(outline.chapters, start=1)
        for scene in chapter.scenes
    }

    entries: dict[str, CodexItem] = {item.entry.id: item for item in codex.items}
    known_entities: dict[str, CodexKind] = {eid: item.kind for eid, item in entries.items()}
    for line in cast_lines:
        cid = _text(line.get("id"))
        cname = _text(line.get("name")) or cid
        ckind = _text(line.get("kind")) or "characters"
        if not (_valid_entry_id(cid) and cid not in entries and ckind in ("characters", "places")):
            continue
        if ckind == "characters":
            c_entry = CharacterEntry(
                id=cid,
                name=cname,
                state={"status": status_of(line.get("status"))},
            )
            entries[cid] = CodexItem(kind="characters", entry=c_entry)
            known_entities[cid] = "characters"
        else:
            p_entry = PlaceEntry(id=cid, name=cname)
            entries[cid] = CodexItem(kind="places", entry=p_entry)
            known_entities[cid] = "places"
    planned: dict[str, list[Progression]] = {}

    def held(eid: str, scene_id: str, *, through_scene: bool) -> EntrySnapshot:
        """`eid` at `scene_id`: as the scene starts, or with the scene's changes so far."""
        item = entries[eid]
        added = planned.get(eid, [])
        entry = item.entry.model_copy(update={"progressions": [*item.entry.progressions, *added]})
        return entry_snapshot(entry, placements, scene_id, through_scene=through_scene)

    by_scene: dict[str, list[SceneTransition]] = {sid: [] for sid in ordered_scenes}
    dropped: list[dict[str, str]] = []

    for raw in transitions:
        tr = raw if isinstance(raw, SceneTransition) else SceneTransition.from_record(raw)
        if tr is None:
            continue
        if tr.scene_id not in by_scene:
            dropped.append(
                {
                    "scene_id": tr.scene_id,
                    "entity_id": tr.entity_id,
                    "reason": "unknown_scene",
                }
            )
            continue
        if tr.entity_id not in known_entities:
            dropped.append(
                {
                    "scene_id": tr.scene_id,
                    "entity_id": tr.entity_id,
                    "reason": "unknown_entity",
                }
            )
            continue
        if tr.kind != known_entities[tr.entity_id]:
            # A thread named like a place takes the place's id (#2378): its stage is not
            # the place's to hold.
            dropped.append(
                {"scene_id": tr.scene_id, "entity_id": tr.entity_id, "reason": "kind_mismatch"}
            )
            continue
        by_scene[tr.scene_id].append(tr)

    kept: list[SceneTransition] = []
    active_scenes: list[str] = []
    stagnant_scenes: list[str] = []
    edge_deltas = 0

    for scene_id in ordered_scenes:
        scene_chapter = chapter_of_scene.get(scene_id, 1)
        scene_kept: list[SceneTransition] = []

        for tr in by_scene.get(scene_id, ()):
            eid = tr.entity_id
            # 1. Dead actor check: dead as the scene starts (one who dies in it still acts)
            entering = held(eid, scene_id, through_scene=False).state
            if status_of(entering.get("status")) == "dead":
                dropped.append({"scene_id": scene_id, "entity_id": eid, "reason": "dead_actor"})
                continue

            # 2. Premature twist / conflict payoff check
            stage_val = str(tr.set.get("thread_stage") or "").strip().casefold()
            if arc is not None and stage_val == "paid_off":
                if eid == "twist" and scene_chapter < arc.twist_chapter:
                    dropped.append(
                        {
                            "scene_id": scene_id,
                            "entity_id": eid,
                            "reason": "premature_twist",
                        }
                    )
                    continue
                if eid == "central_conflict" and scene_chapter < arc.climax_chapter:
                    dropped.append(
                        {
                            "scene_id": scene_id,
                            "entity_id": eid,
                            "reason": "premature_resolution",
                        }
                    )
                    continue

            # 3. Filter immutable keys and redundant state/relation assignments
            now = held(eid, scene_id, through_scene=True)
            state_now, rel_now = now.state, now.relations
            new_set: dict[str, JsonValue] = {}
            for k, v in tr.set.items():
                if k in _IMMUTABLE_KEYS and k in state_now:
                    dropped.append(
                        {
                            "scene_id": scene_id,
                            "entity_id": eid,
                            "reason": "immutable_identity",
                        }
                    )
                    continue
                if state_now.get(k) == v:
                    continue
                if (
                    k in _NARRATIVE_STATE_KEYS
                    and isinstance(v, str)
                    and len(v) >= 6
                    and any(
                        isinstance(prev_v := state_now.get(pk), str)
                        and bigram_overlap(prev_v, v) >= 0.65
                        for pk in _NARRATIVE_STATE_KEYS
                    )
                ):
                    continue
                new_set[k] = v

            new_rel: dict[str, str | None] = {}
            for target, word in tr.relations.items():
                if target not in known_entities or target == eid:
                    continue
                if rel_now.get(target) == word:
                    continue
                new_rel[target] = word

            if not new_set and not new_rel:
                dropped.append(
                    {
                        "scene_id": scene_id,
                        "entity_id": eid,
                        "reason": "redundant_transition",
                    }
                )
                continue

            cleaned = SceneTransition(
                scene_id=scene_id,
                entity_id=eid,
                kind=known_entities.get(eid, tr.kind),
                set=new_set,
                relations=new_rel,
                note=tr.note,
            )
            # Advance the world state G_{t-1} -> G_t: later scenes read it from here.
            progression = cleaned.to_progression()
            if progression is not None:
                planned.setdefault(eid, []).append(progression)
            scene_kept.append(cleaned)
            kept.append(cleaned)
            edge_deltas += cleaned.edge_count()

        if scene_kept:
            active_scenes.append(scene_id)
        else:
            stagnant_scenes.append(scene_id)

    report = TrajectoryReport(
        total_scenes=len(ordered_scenes),
        active_scenes=tuple(active_scenes),
        stagnant_scenes=tuple(stagnant_scenes),
        edge_deltas=edge_deltas,
        dropped=tuple(dropped),
    )
    return kept, report


def _format_value(v: object, names: Mapping[str, str]) -> object:
    """Resolve entity IDs to human-readable names in both scalar strings and string lists."""
    if isinstance(v, str):
        return names.get(v, v)
    if isinstance(v, list):
        items = [names.get(s, s) if isinstance(s, str) else str(s) for s in cast("list[object]", v)]
        return ", ".join(items) if len(items) == 1 else f"[{', '.join(items)}]"
    return v


def _fold_applied_changes(changes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prune superseded state keys and relation targets from earlier applied progressions so
    only the latest active value (and its `[at]` timestamp) appears in entering world state."""
    if len(changes) <= 1:
        return [dict(c) for c in changes]
    last_state_idx: dict[str, int] = {}
    last_rel_idx: dict[str, int] = {}
    for idx, change in enumerate(changes):
        raw_set = change.get("set")
        if isinstance(raw_set, dict):
            for k in cast("dict[str, Any]", raw_set):
                last_state_idx[k] = idx
        raw_rel = change.get("relations")
        if isinstance(raw_rel, dict):
            for k in cast("dict[str, Any]", raw_rel):
                last_rel_idx[k] = idx
    folded: list[dict[str, Any]] = []
    for idx, change in enumerate(changes):
        raw_set = change.get("set")
        kept_set = (
            {
                k: v
                for k, v in cast("dict[str, Any]", raw_set).items()
                if last_state_idx.get(k) == idx
            }
            if isinstance(raw_set, dict)
            else {}
        )
        raw_rel = change.get("relations")
        kept_rel = (
            {k: v for k, v in cast("dict[str, Any]", raw_rel).items() if last_rel_idx.get(k) == idx}
            if isinstance(raw_rel, dict)
            else {}
        )
        if not kept_set and not kept_rel:
            continue
        updated = dict(change)
        if "set" in updated:
            updated["set"] = kept_set
        if "relations" in updated:
            updated["relations"] = kept_rel
        folded.append(updated)
    return folded


def _format_changes(
    changes: Sequence[Mapping[str, Any]],
    names: Mapping[str, str],
    *,
    before_state: Mapping[str, Any] | None = None,
    before_relations: Mapping[str, str] | None = None,
    show_at: bool = False,
) -> str:
    parts: list[str] = []
    for change in changes:
        kind = change.get("kind")
        step_parts: list[str] = []
        if kind == "state":
            raw_set = change.get("set")
            if isinstance(raw_set, dict):
                kv: list[str] = []
                for k, v in cast("dict[str, Any]", raw_set).items():
                    val_str = _format_value(v, names)
                    prev = before_state.get(k) if before_state is not None else None
                    prev = None if prev in ("", []) else prev
                    if prev is not None and prev != v and k not in ("gender",):
                        prev_str = _format_value(prev, names)
                        kv.append(f"{k}={val_str} (from {prev_str})")
                    else:
                        kv.append(f"{k}={val_str}")
                if kv:
                    step_parts.append(", ".join(kv))
        if kind in ("state", "relations"):
            raw_rel = change.get("relations")
            if isinstance(raw_rel, dict):
                rv: list[str] = []
                for k, v in cast("dict[str, Any]", raw_rel).items():
                    target_label = names.get(k, k)
                    prev_rel = before_relations.get(k) if before_relations is not None else None
                    if prev_rel and prev_rel != v:
                        rv.append(f"relation({target_label})={v} (from {prev_rel})")
                    else:
                        rv.append(f"relation({target_label})={v}")
                if rv:
                    step_parts.append(", ".join(rv))
        note = _text(change.get("note"))
        if note and note not in step_parts:
            step_parts.append(f"({note})")
        at = _text(change.get("at")) if show_at else ""
        if at and step_parts:
            step_parts[-1] = f"{step_parts[-1]} [{at}]"
        parts.extend(step_parts)
    return "; ".join(parts)


def entering_world_state(codex: CodexIndex, outline: Outline, scene_id: str) -> str:
    """Render the net world state ($G_{t-1}$) entering `scene_id` across all codex items so
    chapter planning (`_propose_details` / `_revise_details`) and RPG scene choices see both
    current state/relations and the scene notes where they were enacted."""
    placements = place_scenes(outline)
    if scene_id not in placements:
        return ""
    names = {item.entry.id: item.entry.name for item in codex.items}
    lines: list[str] = []
    for item in codex.items:
        snap = entry_snapshot(item.entry, placements, scene_id, through_scene=False)
        state_items = [
            f"{k}={_format_value(v, names)}"
            for k, v in snap.state.items()
            if k not in ("gender",) and v is not None and v not in ("", [])
        ]
        rel_items = [f"relation({names.get(k, k)})={v}" for k, v in snap.relations.items() if v]
        note_items = [
            f"[{_text(c.get('at'))}: {n}]" if _text(c.get("at")) else f"[{n}]"
            for c in _fold_applied_changes(snap.applied)
            if (n := _text(c.get("note")))
        ]
        summary = ", ".join([*state_items, *rel_items, *note_items])
        if summary:
            lines.append(f"- {item.entry.name} ({item.kind}): {summary}")
    return "\n".join(lines)


def kg_scene_guidance(bundle: Mapping[str, Any], codex: CodexIndex) -> str:
    """Format the entering world state ($G_{t-1}$) and required target transitions ($\\Delta G_t$)
    from `scene_context`'s `bundle` so the Writer renders the causal state transition."""
    raw_manifest = bundle.get("manifest")
    if not isinstance(raw_manifest, dict):
        return ""
    manifest = cast("dict[str, Any]", raw_manifest)
    names = {item.entry.id: item.entry.name for item in codex.items}
    entering_states: dict[str, Mapping[str, Any]] = {}
    entering_relations: dict[str, Mapping[str, str]] = {}
    raw_codex = bundle.get("codex")
    if isinstance(raw_codex, list):
        for entry_obj in cast("list[object]", raw_codex):
            if not isinstance(entry_obj, dict):
                continue
            c_item = cast("dict[str, Any]", entry_obj)
            cid = _text(c_item.get("id"))
            if not cid:
                continue
            if isinstance(c_item.get("state"), dict):
                entering_states[cid] = cast("dict[str, Any]", c_item["state"])
            if isinstance(c_item.get("relations"), dict):
                entering_relations[cid] = cast("dict[str, str]", c_item["relations"])

    lines: list[str] = []

    applied = manifest.get("progressions_applied")
    if isinstance(applied, list) and applied:
        entering: list[str] = []
        for entry_obj in cast("list[object]", applied):
            if not isinstance(entry_obj, dict):
                continue
            item = cast("dict[str, Any]", entry_obj)
            eid = _text(item.get("id"))
            label = names.get(eid, eid)
            raw_applied = cast("list[dict[str, Any]]", item.get("applied") or [])
            desc = _format_changes(
                _fold_applied_changes(raw_applied),
                names,
                show_at=True,
            )
            if desc:
                entering.append(f"- {label}: {desc}")
        if entering:
            lines.append(
                "World state entering this scene (changes from earlier scenes already in force):\n"
                + "\n".join(entering)
            )

    in_this_scene = manifest.get("changes_in_this_scene")
    if isinstance(in_this_scene, list) and in_this_scene:
        targets: list[str] = []
        for entry_obj in cast("list[object]", in_this_scene):
            if not isinstance(entry_obj, dict):
                continue
            item = cast("dict[str, Any]", entry_obj)
            eid = _text(item.get("id"))
            label = names.get(eid, eid)
            desc = _format_changes(
                cast("list[dict[str, Any]]", item.get("changes") or []),
                names,
                before_state=entering_states.get(eid),
                before_relations=entering_relations.get(eid),
            )
            if desc:
                targets.append(f"- {label}: {desc}")
        if targets:
            lines.append(
                "Target state/relationship transitions in THIS scene (show these changes "
                "happening naturally by the end of the scene):\n" + "\n".join(targets)
            )

    return "\n\n".join(lines)


def chronological_kg_log(
    codex: CodexIndex,
    outline: Outline,
    before_scene_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the chronological sequence of KG state and relation transitions across scenes
    in story-time order ($G_0 \\to G_1 \\to \\dots$), stopping before `before_scene_id` when given.
    """
    placements = place_scenes(outline)
    ordered = [p.scene_id for p in sorted(placements.values(), key=lambda pl: pl.position)]
    names = {item.entry.id: item.entry.name for item in codex.items}
    titles = {sc.id: sc.title for _, sc in outline.scenes_in_order()}
    log: list[dict[str, Any]] = []
    for sid in ordered:
        if sid == before_scene_id:
            break
        scene_changes: list[str] = []
        for item in codex.items:
            before_snap = entry_snapshot(item.entry, placements, sid, through_scene=False)
            after_snap = entry_snapshot(item.entry, placements, sid, through_scene=True)
            if not after_snap.in_this_scene:
                continue
            desc = _format_changes(
                after_snap.in_this_scene,
                names,
                before_state=before_snap.state,
                before_relations=before_snap.relations,
            )
            if desc:
                scene_changes.append(f"{item.entry.name} ({item.kind}): {desc}")
        if scene_changes:
            log.append(
                {
                    "scene_id": sid,
                    "title": titles.get(sid, sid),
                    "changes": scene_changes,
                }
            )
    return log


def format_chronological_kg_log(
    codex: CodexIndex,
    outline: Outline,
    before_scene_id: str | None = None,
) -> str:
    """Format the chronological KG transition log and open plot threads into a prompt-ready
    summary so scene choices and dynamic KG updates maintain story-wide completeness.
    """
    log = chronological_kg_log(codex, outline, before_scene_id=before_scene_id)
    placements = place_scenes(outline)
    target_sid = before_scene_id
    if target_sid is None and placements:
        target_sid = max(placements.values(), key=lambda pl: pl.position).scene_id

    blocks: list[str] = []
    if log:
        step_lines = [
            f"- [{entry['scene_id']}] {entry['title']}: " + " | ".join(entry["changes"])
            for entry in log
        ]
        blocks.append(
            "Chronological KG transition history so far (in story-time order):\n"
            + "\n".join(step_lines)
        )

    thread_lines: list[str] = []
    names = {item.entry.id: item.entry.name for item in codex.items}
    for item in codex.items:
        if item.kind != "threads" or not isinstance(item.entry, ThreadEntry):
            continue
        stage = str(item.entry.state.get("thread_stage") or "open")
        if target_sid and target_sid in placements:
            snap = entry_snapshot(
                item.entry, placements, target_sid, through_scene=before_scene_id is None
            )
            stage = str(snap.state.get("thread_stage") or stage)
        planted = item.entry.planted_in or "-"
        payoff = item.entry.pay_off_by or "-"
        involved = ", ".join(
            names.get(i, i) for i, rel in item.entry.relations.items() if i and rel
        )
        inv_text = f", involves=[{involved}]" if involved else ""
        thread_lines.append(
            f"- {item.entry.name}: stage={stage} (planted_in={planted}, pay_off_by={payoff}{inv_text})"
        )
    if thread_lines:
        blocks.append(
            "Plot threads & narrative completeness status (advance planted threads toward their payoff):\n"
            + "\n".join(thread_lines)
        )
    return "\n\n".join(blocks)
