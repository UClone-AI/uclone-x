"""`muse_spark`: draw idea cards from a genre's table, reproducibly from a seed (#1559).

A card takes one entry from each slot of the genre's table. Which entry is a pure function of
`(seed, card number, slot name)`, computed with SHA-256 rather than `random`, so the same seed
draws the same card on any machine and any Python version, and adding a slot to a table does
not change what the other slots draw. The seed is always returned, so a draw the model made
without one can be repeated.

The tables are read on every call from the bundled data and from each active skill that
carries a `resources/story/muse/` folder (#1572), so the tool's description does not list
genres -- it would be stale the moment a skill added one -- and says to call `genres` instead.
A draw without a genre takes the open story's, and the result says so (P6).
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import PlainRefusalError
from uclone_x.story.library import StoryLibrary
from uclone_x.story.skill_data import (
    BUNDLED_DATA_ROOT,
    MuseTable,
    SkillDataError,
    Sourced,
    data_roots,
    load_muse_tables_sourced,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext

__all__ = [
    "MAX_CARDS",
    "MAX_SEED",
    "MuseDataError",
    "MuseSparkParams",
    "MuseSparkTool",
    "NoGenreError",
    "UnknownGenreError",
    "draw_cards",
    "normalize_genre",
]

logger = logging.getLogger(__name__)

MAX_SEED = 2**32 - 1
MAX_CARDS = 5


class UnknownGenreError(PlainRefusalError):
    """The draw named a genre no table covers. The message is written for a person."""


class NoGenreError(PlainRefusalError):
    """No genre was given and the open story names none, or no story is open."""


class MuseDataError(PlainRefusalError):
    """A muse table could not be loaded. The message names the file within its data root."""


def normalize_genre(genre: str) -> str:
    """`"Science Fiction"` and `"science_fiction"` name the table `science-fiction`."""
    return re.sub(r"[\s_]+", "-", genre.strip().lower())


class MuseSparkParams(BaseModel):
    """What to draw."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["draw", "genres"] = Field(
        description="'draw' idea cards from a genre's table; 'genres' lists the genres there "
        "are tables for."
    )
    genre: str | None = Field(
        default=None,
        description="For 'draw': the genre whose table to draw from. Leave it out to use the "
        "open story's genre.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SEED,
        description="Repeat an earlier draw by passing the seed it returned. Leave it out "
        "for a new draw.",
    )
    count: int = Field(default=1, ge=1, le=MAX_CARDS, description="How many cards to draw.")


def _pick(seed: int, card: int, slot: str, size: int) -> int:
    digest = hashlib.sha256(f"{seed}:{card}:{slot}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


def draw_cards(table: MuseTable, seed: int, count: int) -> list[dict[str, str]]:
    """`count` cards from `table`: the same table, seed and count always give the same cards."""
    return [
        {
            slot.slot: slot.entries[_pick(seed, card, slot.slot, len(slot.entries))]
            for slot in table.slots
        }
        for card in range(count)
    ]


class MuseSparkTool(BaseTool[MuseSparkParams]):
    """Idea cards for a stuck writer, reproducible from their seed."""

    name = "muse_spark"
    writes_files: ClassVar[bool] = False  # reads its tables and story.yaml; writes nothing
    params_type = MuseSparkParams
    description = (
        "Draw idea cards for a story from a genre's table: each card combines one entry "
        "from every column of the table, such as a character, a place, a conflict and a "
        "twist. Call 'genres' to see which genres have a table; a skill can add more. "
        "Leave 'genre' out to use the open story's genre. The result carries the seed; "
        "pass it again to get the same cards."
    )

    def __init__(self, data_roots: Sequence[Path] = (BUNDLED_DATA_ROOT,)) -> None:
        self._data_roots = tuple(data_roots)
        super().__init__(name=self.name, description=self.description, params_type=MuseSparkParams)

    def _tables(self, context: ToolContext) -> dict[str, Sourced[MuseTable]]:
        # Read on every call, so an edited table takes effect, a skill approved since the
        # last call counts, and a broken table is reported by file and field instead of
        # drawing from what was loaded before.
        try:
            return load_muse_tables_sourced(data_roots(context.skill_dirs, self._data_roots))
        except SkillDataError as exc:
            logger.warning("muse_spark could not load its tables: %s", exc)
            raise MuseDataError(
                f"The idea tables could not be loaded, so nothing was drawn: {exc.plain}."
            ) from exc

    def run(self, params: MuseSparkParams, context: ToolContext) -> dict[str, Any]:
        tables = self._tables(context)
        if params.action == "genres":
            return {
                "genres": [
                    {"genre": genre, "title": found.item.title, "source": found.source}
                    for genre, found in sorted(tables.items())
                ]
            }
        names = ", ".join(sorted(tables))
        note: str | None = None
        if params.genre is not None:
            genre = params.genre
            whose = f"the genre '{genre}'"
        else:
            genre = _story_genre(context, names)
            whose = f"the open story's genre '{genre}'"
            note = f"No genre was given, so the open story's genre '{genre}' was used."
        found = tables.get(normalize_genre(genre))
        if found is None:
            raise UnknownGenreError(
                f"There is no table for {whose}, so nothing was drawn. Genres: {names}."
            )
        table = found.item
        seed = params.seed if params.seed is not None else secrets.randbelow(MAX_SEED + 1)
        result: dict[str, Any] = {
            "genre": table.genre,
            "source": found.source,
            "seed": seed,
            "table_digest": table.digest(),
            "cards": draw_cards(table, seed, params.count),
        }
        if note is not None:
            result["note"] = note
        return result


def _story_genre(context: ToolContext, names: str) -> str:
    """The open story's genre, for a draw that named none; refuses when there is none."""
    if context.story_id is None or context.workspace_root is None:
        raise NoGenreError(
            "Name a genre to draw from: none was given, and no story is open to take one "
            f"from. Genres: {names}."
        )
    record = StoryLibrary(context.workspace_root).load(context.story_id)
    if record.genre is None or not record.genre.strip():
        raise NoGenreError(
            f"Name a genre to draw from: none was given, and the open story '{record.title}' "
            f"has no genre. Genres: {names}."
        )
    return record.genre.strip()
