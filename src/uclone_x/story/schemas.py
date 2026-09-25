"""The shapes of a story's files, and how a file that does not fit is reported (#1556).

A story's files are the user's: YAML a person can edit by hand. Each is validated every
time it is read, and a file that does not fit is refused with **the file and the field**
named, in words a person reads (P6). Nothing is defaulted in place of a wrong value.

This module is pure: it parses text it is given and touches no file. Where the text comes
from is `uclone_x.story.work`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from uclone_x.errors import PlainRefusalError

__all__ = [
    "CODEX_KINDS",
    "ENTRY_ID_PATTERN",
    "Chapter",
    "CharacterEntry",
    "CodexEntry",
    "CodexKind",
    "EntryId",
    "Outline",
    "PlaceEntry",
    "ItemEntry",
    "Progression",
    "Revision",
    "RevisionLog",
    "Scene",
    "SessionEntry",
    "SessionsFile",
    "StoryFileError",
    "ThreadEntry",
    "Visual",
    "VisualProgression",
    "describe_invalid",
    "dump_file",
    "entry_model",
    "parse_file",
]

#: An id a person or the model gives an outline scene or a codex entry: lowercase letters
#: and digits, joined by `.`, `_` or `-` (`ch03.s04`, `lord_vane`). It is also a file name.
ENTRY_ID_PATTERN = r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$"
EntryId = Annotated[str, Field(pattern=ENTRY_ID_PATTERN, max_length=80)]

CodexKind = Literal["characters", "places", "items", "threads"]
CODEX_KINDS: tuple[CodexKind, ...] = ("characters", "places", "items", "threads")


class StoryFileError(PlainRefusalError):
    """A story file is there and does not fit its shape; the message names file and field."""


class _Shape(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


# -- outline.yaml ----------------------------------------------------------------------


class Scene(_Shape):
    """One scene: its fixed id, what happens, and who is in it."""

    id: EntryId
    title: str = Field(min_length=1)
    summary: str = ""
    characters: list[EntryId] = Field(default_factory=list[str])
    places: list[EntryId] = Field(default_factory=list[str])
    beats: list[str] = Field(default_factory=list[str])
    #: When the scene happens in the story, as opposed to where it sits in the book.
    #: Kept for the phase that applies progressions; nothing orders by it yet.
    story_time: str | int | None = None


class Chapter(_Shape):
    """A chapter and its scenes, in reading order. `act` groups chapters into acts."""

    id: EntryId
    title: str = Field(min_length=1)
    act: str | None = None
    scenes: list[Scene] = Field(default_factory=list[Scene])


class Outline(_Shape):
    """`outline.yaml`: chapters in reading order, each holding its scenes."""

    chapters: list[Chapter] = Field(default_factory=list[Chapter])

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Outline:
        seen_chapters: set[str] = set()
        seen_scenes: set[str] = set()
        for chapter in self.chapters:
            if chapter.id in seen_chapters:
                raise ValueError(f"the chapter id '{chapter.id}' is used more than once")
            seen_chapters.add(chapter.id)
            for scene in chapter.scenes:
                if scene.id in seen_scenes:
                    raise ValueError(f"the scene id '{scene.id}' is used more than once")
                seen_scenes.add(scene.id)
        return self

    def scenes_in_order(self) -> list[tuple[Chapter, Scene]]:
        """Every scene with its chapter, in reading order."""
        return [(chapter, scene) for chapter in self.chapters for scene in chapter.scenes]

    def find(self, scene_id: str) -> tuple[Chapter, Scene] | None:
        """The scene `scene_id` and its chapter, or `None` when the outline has no such scene."""
        for chapter, scene in self.scenes_in_order():
            if scene.id == scene_id:
                return chapter, scene
        return None


# -- codex/<kind>/<id>.yaml -------------------------------------------------------------


class Progression(_Shape):
    """A change to an entry's state from a scene on. Stored now, applied by a later phase."""

    at: EntryId
    set: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    note: str | None = None


class VisualProgression(_Shape):
    """A change to how a character looks from a scene on. Stored now, applied later."""

    at: EntryId
    add_tags: list[str] = Field(default_factory=list[str])
    remove_tags: list[str] = Field(default_factory=list[str])
    note: str | None = None


class Visual(_Shape):
    """How a character looks, for illustrations: the block `character_sheet` reads."""

    tags: list[str] = Field(default_factory=list[str])
    prose: str | None = None
    negative_tags: list[str] = Field(default_factory=list[str])
    base_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    gender: Literal["female", "male", "other"] | None = None
    default_style: Literal["photorealistic", "anime", "artistic", "diagram"] | None = None
    progressions: list[VisualProgression] = Field(default_factory=list[VisualProgression])


class CodexEntry(_Shape):
    """What every codex entry has: who or what it is, and what it is called."""

    id: EntryId
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list[str])
    profile: str = ""
    #: Put in every scene's context, whether or not the scene mentions it (a world rule).
    always_include: bool = False
    state: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    progressions: list[Progression] = Field(default_factory=list[Progression])
    notes: str | None = None


class CharacterEntry(CodexEntry):
    """`codex/characters/<id>.yaml`."""

    visual: Visual | None = None


class PlaceEntry(CodexEntry):
    """`codex/places/<id>.yaml`."""


class ItemEntry(CodexEntry):
    """`codex/items/<id>.yaml`."""


class ThreadEntry(CodexEntry):
    """`codex/threads/<id>.yaml`: a thread planted in one scene and paid off in another."""

    planted_in: EntryId | None = None
    pay_off_by: EntryId | None = None
    paid_off_in: EntryId | None = None


_ENTRY_MODELS: dict[CodexKind, type[CodexEntry]] = {
    "characters": CharacterEntry,
    "places": PlaceEntry,
    "items": ItemEntry,
    "threads": ThreadEntry,
}


def entry_model(kind: CodexKind) -> type[CodexEntry]:
    """The shape of a codex entry of `kind`."""
    return _ENTRY_MODELS[kind]


# -- sessions.yaml and the manuscript's revision log ------------------------------------


class SessionEntry(_Shape):
    """One conversation's work on the story, for the next conversation's recap."""

    room_id: str = Field(min_length=1)
    opened_at: str = Field(min_length=1)
    closed_at: str | None = None
    scenes_written: list[EntryId] = Field(default_factory=list[str])
    #: One paragraph on where the story stands, left by the conversation's latest write.
    summary: str | None = None


class SessionsFile(_Shape):
    """`sessions.yaml`: the conversations that wrote the story, oldest first."""

    sessions: list[SessionEntry] = Field(default_factory=list[SessionEntry])


class Revision(_Shape):
    """One write of a scene: which text, when, and which conversation and turn wrote it."""

    digest: str = Field(min_length=1)
    written_at: str = Field(min_length=1)
    room_id: str = Field(min_length=1)
    agent_id: str | None = None
    turn_index: int | None = None
    #: The history file holding the text this write replaced, when it replaced one.
    replaced: str | None = None


class RevisionLog(_Shape):
    """`manuscript/.history/<scene_id>/revisions.yaml`, oldest first."""

    scene_id: EntryId
    revisions: list[Revision] = Field(default_factory=list[Revision])


# -- reading and reporting -------------------------------------------------------------

_M = TypeVar("_M", bound=BaseModel)

_PROBLEMS: dict[str, str] = {
    "missing": "is missing",
    "extra_forbidden": "is not a field this file has",
    "string_type": "should be text (put it in quotes if it looks like a number)",
    "int_type": "should be a whole number",
    "int_parsing": "should be a whole number",
    "bool_type": "should be true or false",
    "list_type": "should be a list",
    "dict_type": "should be a set of named fields",
    "model_type": "should be a set of named fields",
    "string_too_short": "should not be empty",
    "string_too_long": "is too long",
    "string_pattern_mismatch": "is not a valid id: use lowercase letters and digits, "
    "joined by '.', '_' or '-'",
    "literal_error": "is not one of the allowed values",
    "greater_than_equal": "is too small",
    "less_than_equal": "is too large",
}


def _where(loc: tuple[int | str, ...]) -> str:
    text = ""
    for part in loc:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else part
    return text


def describe_invalid(where: str, exc: ValidationError) -> str:
    """A sentence naming the file and every field that does not fit, for a person.

    The field is written as a path into the file (`chapters[0].scenes[1].id`). What is
    wrong is said in words; the validator's own wording, type names and links are not
    shown.
    """
    problems: list[str] = []
    for err in exc.errors():
        loc = tuple(p for p in err["loc"] if not (isinstance(p, str) and "[" in p))
        field = _where(loc)
        if err["type"] == "value_error":
            ctx = err.get("ctx") or {}
            reason = str(ctx.get("error", "has a value that does not fit"))
            problems.append(f"{field}: {reason}" if field else reason)
            continue
        reason = _PROBLEMS.get(err["type"], "has a value of the wrong kind")
        problems.append(f"'{field}' {reason}" if field else f"the file {reason}")
    shown = problems[:5]
    more = len(problems) - len(shown)
    tail = f" (and {more} more)" if more else ""
    return f"{where} does not fit its shape: {'; '.join(shown)}{tail}."


def parse_file(model: type[_M], text: str, where: str) -> _M:
    """`text`, the content of the story file `where`, as a `model`.

    Raises:
        StoryFileError: the text is not YAML, is not a set of named fields, or does not
            fit `model`. The message names `where` and the field.
    """
    try:
        raw: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = getattr(mark, "line", None)
        at = f" near line {line + 1}" if isinstance(line, int) else ""
        raise StoryFileError(f"{where} could not be read as YAML{at}.") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise StoryFileError(f"{where} is not a set of named fields.")
    data = {str(k): v for k, v in cast(dict[object, Any], raw).items()}
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise StoryFileError(describe_invalid(where, exc)) from exc


def dump_file(model: BaseModel) -> str:
    """`model` as the YAML text of its file, fields in declaration order."""
    return yaml.safe_dump(
        model.model_dump(mode="json", exclude_defaults=False),
        sort_keys=False,
        allow_unicode=True,
    )
