"""What the Writer is given before it writes a scene, and where a new conversation picks up (#1556).

`scene_context` is the bundle for one scene: the scene and its neighbours, the end of the
scene before it, and the codex entries it needs, with a **manifest** that says which
entries went in and why, which were left out and why, and what was not applied. Nothing is
cut in silence: an entry over the budget is listed as left out (P6).

`recap` is the start of a new conversation: what the story is, what the last conversations
did, how far the manuscript has got, and the next scene's bundle, so that a conversation
with no memory of the last one can continue from the recap alone.

Progressions (a change to an entry from a scene on) are stored but **not applied** in this
phase: the manifest lists each one, so the Writer knows the state it sees is the entry's
starting state.

This module is pure: it is given the files' contents and reads nothing itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from uclone_x.story.schemas import (
    Chapter,
    CharacterEntry,
    CodexEntry,
    CodexKind,
    Outline,
    Scene,
    SessionsFile,
    ThreadEntry,
)

__all__ = [
    "MAX_ENTRIES",
    "RECENT_SESSIONS",
    "TAIL_CHARS",
    "CodexIndex",
    "CodexItem",
    "UnreadableFile",
    "last_and_next",
    "recap",
    "render_entry",
    "scene_context",
    "tail",
]

#: How much of the previous scene's end the Writer is given, in characters.
TAIL_CHARS = 1200
#: How many codex entries one scene's context holds; the rest are listed as left out.
MAX_ENTRIES = 12
#: How many of the latest conversations a recap describes.
RECENT_SESSIONS = 3
#: A name shorter than this is not looked for in the text: one letter matches everything.
_MIN_NAME = 2


@dataclass(frozen=True)
class CodexItem:
    """A codex entry that loaded, and which kind it is."""

    kind: CodexKind
    entry: CodexEntry


@dataclass(frozen=True)
class UnreadableFile:
    """A story file that did not load, and the reason, naming the field."""

    file: str
    reason: str


@dataclass(frozen=True)
class CodexIndex:
    """Every codex entry that loaded, and every file that did not -- none is skipped."""

    items: tuple[CodexItem, ...] = ()
    unreadable: tuple[UnreadableFile, ...] = ()

    def find(self, entry_id: str, kind: CodexKind | None = None) -> list[CodexItem]:
        """The entries called `entry_id`, of `kind` when given."""
        return [
            item
            for item in self.items
            if item.entry.id == entry_id and (kind is None or item.kind == kind)
        ]


def tail(text: str, limit: int = TAIL_CHARS) -> str:
    """The end of `text`, at most `limit` characters, starting at a word when it is cut."""
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    space = cut.find(" ")
    if 0 <= space < limit // 4:
        cut = cut[space + 1 :]
    return "..." + cut


def render_entry(item: CodexItem) -> dict[str, Any]:
    """A codex entry as the Writer reads it; empty fields are left out."""
    entry = item.entry
    out: dict[str, Any] = {"id": entry.id, "kind": item.kind, "name": entry.name}
    if entry.aliases:
        out["aliases"] = list(entry.aliases)
    if entry.profile:
        out["profile"] = entry.profile
    if entry.state:
        out["state"] = dict(entry.state)
    if entry.notes:
        out["notes"] = entry.notes
    if isinstance(entry, ThreadEntry):
        for key in ("planted_in", "pay_off_by", "paid_off_in"):
            value = getattr(entry, key)
            if value is not None:
                out[key] = value
    if isinstance(entry, CharacterEntry) and entry.visual is not None:
        out["has_visual"] = True
    return out


def _scene_line(chapter: Chapter, scene: Scene) -> dict[str, Any]:
    line: dict[str, Any] = {"scene_id": scene.id, "chapter_id": chapter.id, "title": scene.title}
    if scene.summary:
        line["summary"] = scene.summary
    return line


def _mentions(entry: CodexEntry, haystack: str) -> bool:
    return any(
        len(name) >= _MIN_NAME and name.casefold() in haystack
        for name in (entry.name, *entry.aliases)
    )


def scene_context(
    outline: Outline,
    scene_id: str,
    codex: CodexIndex,
    *,
    previous_text: str | None,
    story: Mapping[str, Any],
) -> dict[str, Any]:
    """The bundle the Writer reads before writing `scene_id`, with its manifest.

    Entries go in in this order until `MAX_ENTRIES`: the characters and places the scene
    names, then every `always_include` entry, then every entry whose name or alias the
    scene's title, summary or beats -- or the end of the scene before -- mention.

    Raises:
        ValueError: the outline has no scene `scene_id` (the caller checks first).
    """
    ordered = outline.scenes_in_order()
    index = next((i for i, (_, s) in enumerate(ordered) if s.id == scene_id), None)
    if index is None:
        raise ValueError(f"the outline has no scene '{scene_id}'")
    chapter, scene = ordered[index]
    previous_tail = tail(previous_text) if previous_text else None

    candidates: list[tuple[CodexItem, str]] = []
    named_missing: list[dict[str, str]] = []
    named: tuple[tuple[CodexKind, list[str]], ...] = (
        ("characters", scene.characters),
        ("places", scene.places),
    )
    for kind, ids in named:
        for entry_id in ids:
            found = codex.find(entry_id, kind)
            if found:
                candidates.append((found[0], "named in the scene"))
            else:
                named_missing.append({"id": entry_id, "kind": kind})
    for item in codex.items:
        if item.entry.always_include:
            candidates.append((item, "always included"))
    haystack = " ".join([scene.title, scene.summary, *scene.beats, previous_tail or ""]).casefold()
    for item in codex.items:
        if _mentions(item.entry, haystack):
            candidates.append((item, "mentioned in the scene or the end of the scene before"))

    seen: set[tuple[str, str]] = set()
    included: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    left_out: list[dict[str, Any]] = []
    for item, reason in candidates:
        key = (item.kind, item.entry.id)
        if key in seen:
            continue
        seen.add(key)
        if len(entries) >= MAX_ENTRIES:
            left_out.append(
                {
                    "id": item.entry.id,
                    "kind": item.kind,
                    "reason": f"over the budget of {MAX_ENTRIES} entries ({reason})",
                }
            )
            continue
        entries.append(render_entry(item))
        included.append({"id": item.entry.id, "kind": item.kind, "reason": reason})
    for item in codex.items:
        if (item.kind, item.entry.id) not in seen:
            left_out.append(
                {
                    "id": item.entry.id,
                    "kind": item.kind,
                    "reason": "not named, always-included or mentioned in this scene",
                }
            )

    in_bundle = {(i["kind"], i["id"]) for i in included}
    not_applied: list[dict[str, Any]] = []
    for item in codex.items:
        if (item.kind, item.entry.id) not in in_bundle:
            continue
        ats = [p.at for p in item.entry.progressions]
        if isinstance(item.entry, CharacterEntry) and item.entry.visual is not None:
            ats += [p.at for p in item.entry.visual.progressions]
        if ats:
            not_applied.append({"id": item.entry.id, "kind": item.kind, "at_scenes": ats})

    manifest: dict[str, Any] = {"included": included, "left_out": left_out}
    if named_missing:
        manifest["named_without_an_entry"] = named_missing
    if not_applied:
        manifest["progressions_not_applied"] = not_applied
        manifest["progressions_note"] = (
            "These entries change from the scenes listed. Changes are not applied yet: the "
            "state shown is each entry's starting state, so follow the manuscript where the "
            "two differ."
        )
    if codex.unreadable:
        manifest["unreadable_files"] = [
            {"file": u.file, "reason": u.reason} for u in codex.unreadable
        ]

    bundle: dict[str, Any] = {
        "story": dict(story),
        "chapter": {"chapter_id": chapter.id, "title": chapter.title},
        "scene": scene.model_dump(mode="json", exclude_defaults=True),
        "previous_scene": _scene_line(*ordered[index - 1]) if index > 0 else None,
        "next_scene": _scene_line(*ordered[index + 1]) if index + 1 < len(ordered) else None,
        "end_of_previous_scene": previous_tail,
        "codex": entries,
        "manifest": manifest,
    }
    if chapter.act is not None:
        bundle["chapter"]["act"] = chapter.act
    return bundle


def last_and_next(
    outline: Outline, written: Sequence[str]
) -> tuple[tuple[Chapter, Scene] | None, tuple[Chapter, Scene] | None]:
    """The last written scene in reading order, and the first unwritten scene after it.

    With nothing written, the next scene is the first one. With every scene after the last
    written one written too, there is no next scene.
    """
    ordered = outline.scenes_in_order()
    done = set(written)
    last_index = max((i for i, (_, s) in enumerate(ordered) if s.id in done), default=None)
    start = 0 if last_index is None else last_index + 1
    following = next((pair for pair in ordered[start:] if pair[1].id not in done), None)
    return (ordered[last_index] if last_index is not None else None), following


def recap(
    *,
    story: Mapping[str, Any],
    outline: Outline | None,
    written: Sequence[str],
    sessions: SessionsFile,
    last_text: str | None,
    next_context: Mapping[str, Any] | None,
    problems: Sequence[UnreadableFile] = (),
) -> dict[str, Any]:
    """Where the story stands, for a conversation that remembers none of it.

    `last_text` is the text of the last written scene `last_and_next` names, and
    `next_context` the `scene_context` of the next one; the caller reads both.
    """
    recent = list(sessions.sessions[-RECENT_SESSIONS:])
    out: dict[str, Any] = {
        "story": dict(story),
        "recent_sessions": [s.model_dump(mode="json", exclude_defaults=True) for s in recent],
    }
    older = len(sessions.sessions) - len(recent)
    if older:
        out["older_sessions_not_shown"] = older
    if problems:
        out["unreadable_files"] = [{"file": p.file, "reason": p.reason} for p in problems]
    if outline is None:
        out["progress"] = {"scenes_in_outline": 0, "scenes_written": len(written)}
        out["how_to_continue"] = (
            "The story has no outline yet. Start one with story_outline 'init', then add "
            "scenes with 'set_scene'."
        )
        return out
    in_outline = [s.id for _, s in outline.scenes_in_order()]
    progress: dict[str, Any] = {
        "scenes_in_outline": len(in_outline),
        "scenes_written": sum(1 for s in in_outline if s in set(written)),
    }
    stray = sorted(set(written) - set(in_outline))
    if stray:
        progress["written_but_not_in_outline"] = stray
    out["progress"] = progress
    last, following = last_and_next(outline, written)
    if last is not None:
        out["last_written_scene"] = {
            **_scene_line(*last),
            "end_of_text": tail(last_text) if last_text else None,
        }
    if following is None:
        out["next_scene"] = None
        out["how_to_continue"] = (
            "Every scene in the outline after the last written one has text. Add the next "
            "scene with story_outline 'set_scene', or revise a written one."
        )
        return out
    out["next_scene"] = dict(next_context) if next_context is not None else None
    out["how_to_continue"] = (
        f"Write scene '{following[1].id}' next: its context is under next_scene. Save it "
        "with story_manuscript 'write', and give a one-paragraph session_summary of where "
        "the story stands."
    )
    return out
