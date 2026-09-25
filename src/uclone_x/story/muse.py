"""`muse_spark`: draw idea cards from a genre's table, reproducibly from a seed (#1559).

A card takes one entry from each slot of the genre's table. Which entry is a pure function of
`(seed, card number, slot name)`, computed with SHA-256 rather than `random`, so the same seed
draws the same card on any machine and any Python version, and adding a slot to a table does
not change what the other slots draw. The seed is always returned, so a draw the model made
without one can be repeated.
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
from uclone_x.story.skill_data import (
    BUNDLED_DATA_ROOT,
    MUSE_DIRNAME,
    MuseTable,
    SkillDataError,
    load_muse_tables,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext

__all__ = [
    "MAX_CARDS",
    "MAX_SEED",
    "MuseDataError",
    "MuseSparkParams",
    "MuseSparkTool",
    "UnknownGenreError",
    "draw_cards",
    "normalize_genre",
]

logger = logging.getLogger(__name__)

MAX_SEED = 2**32 - 1
MAX_CARDS = 5


class UnknownGenreError(PlainRefusalError):
    """The draw named a genre no table covers. The message is written for a person."""


class MuseDataError(PlainRefusalError):
    """A muse table could not be loaded. The message names the file within its data root."""


def normalize_genre(genre: str) -> str:
    """`"Science Fiction"` and `"science_fiction"` name the table `science-fiction`."""
    return re.sub(r"[\s_]+", "-", genre.strip().lower())


class MuseSparkParams(BaseModel):
    """What to draw."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["draw"] = Field(description="'draw' idea cards from a genre's table.")
    genre: str = Field(description="The genre whose table to draw from.")
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
    writes_files: ClassVar[bool] = False  # reads its tables; writes nothing
    params_type = MuseSparkParams

    def __init__(self, data_roots: Sequence[Path] = (BUNDLED_DATA_ROOT,)) -> None:
        self._data_roots = tuple(data_roots)
        genres = sorted(
            {p.stem for root in self._data_roots for p in (root / MUSE_DIRNAME).glob("*.yaml")}
        )
        super().__init__(
            name=self.name,
            description=(
                "Draw idea cards for a story from a genre's table: each card combines one "
                "entry from every column of the table, such as a character, a place, a "
                "conflict and a twist. Genres: "
                f"{', '.join(genres)}. The result carries the seed; pass it again to get "
                "the same cards."
            ),
            params_type=MuseSparkParams,
        )

    def run(self, params: MuseSparkParams, context: ToolContext) -> dict[str, Any]:
        # Read on every call, so an edited table takes effect and a broken one is reported
        # by file and field instead of drawing from what was loaded before.
        try:
            tables = load_muse_tables(self._data_roots)
        except SkillDataError as exc:
            logger.warning("muse_spark could not load its tables: %s", exc)
            raise MuseDataError(
                f"The idea tables could not be loaded, so nothing was drawn: {exc.plain}."
            ) from exc
        table = tables.get(normalize_genre(params.genre))
        if table is None:
            raise UnknownGenreError(
                f"There is no table for the genre '{params.genre}', so nothing was drawn. "
                f"Genres: {', '.join(sorted(tables))}."
            )
        seed = params.seed if params.seed is not None else secrets.randbelow(MAX_SEED + 1)
        return {
            "genre": table.genre,
            "seed": seed,
            "table_digest": table.digest(),
            "cards": draw_cards(table, seed, params.count),
        }
