"""`muse_spark` and the story structure templates: data files, a validating loader, and
seed-reproducible idea cards (#1559, design #1552 phase 4)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from uclone_x.story.muse import MuseSparkTool, draw_cards
from uclone_x.story.skill_data import (
    BUNDLED_DATA_ROOT,
    MuseSlot,
    MuseTable,
    SkillDataError,
    load_muse_tables,
    load_structure_templates,
)
from uclone_x.tools.models import ToolContext

_FIXTURE_TABLE = MuseTable(
    genre="fixture",
    title="Fixture",
    slots=(
        MuseSlot(slot="character", entries=tuple(f"character {i}" for i in range(10))),
        MuseSlot(slot="place", entries=tuple(f"place {i}" for i in range(10))),
    ),
)


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(agent_id="writer", session_id="s1", workspace_root=tmp_path)


def _draw(tool: MuseSparkTool, tmp_path: Path, **params: Any) -> dict[str, Any]:
    result = asyncio.run(tool.execute({"action": "draw", **params}, _context(tmp_path)))
    assert result.success, result.error
    assert isinstance(result.output, dict)
    return result.output


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- the acceptance criterion: the same seed draws the same card ------------------------


def test_the_same_seed_draws_the_same_cards(tmp_path: Path) -> None:
    """Two draws with one seed return identical cards; another seed draws others.

    Killed by: src/uclone_x/story/muse.py :: seed = params.seed if params.seed is not None else
    Becomes: seed = params.seed if params.seed is None else
    """
    tool = MuseSparkTool()
    first = _draw(tool, tmp_path, genre="fantasy", seed=81123, count=3)
    again = _draw(tool, tmp_path, genre="fantasy", seed=81123, count=3)
    other = _draw(tool, tmp_path, genre="fantasy", seed=81124, count=3)

    assert first["seed"] == again["seed"] == 81123
    assert first["cards"] == again["cards"]
    assert len(first["cards"]) == 3
    assert other["cards"] != first["cards"]


def test_a_draw_without_a_seed_returns_the_seed_that_repeats_it(tmp_path: Path) -> None:
    """The model need not choose a seed; the one the tool chose comes back and reproduces.

    Killed by: src/uclone_x/story/muse.py :: "seed": seed,
    Becomes: "seed": 1234,
    """
    tool = MuseSparkTool()
    unseeded = _draw(tool, tmp_path, genre="mystery", count=2)
    repeated = _draw(tool, tmp_path, genre="mystery", seed=unseeded["seed"], count=2)

    assert repeated["cards"] == unseeded["cards"]


def test_a_card_is_a_fixed_function_of_the_seed_not_of_the_interpreter() -> None:
    """Pinned literally: a card must not depend on `random`, hash seeding or the Python version.

    Killed by: src/uclone_x/story/muse.py :: f"{seed}:{card}:{slot}"
    Becomes: f"{seed}:{slot}:{card}"
    """
    assert draw_cards(_FIXTURE_TABLE, 81123, 2) == [
        {"character": "character 1", "place": "place 7"},
        {"character": "character 1", "place": "place 3"},
    ]


def test_adding_a_slot_to_a_table_leaves_the_other_slots_draws_alone() -> None:
    """A pick depends on the slot's name, not its position, so a skill can extend a table
    without changing what earlier seeds drew for the columns that were already there."""
    extended = MuseTable(
        genre="fixture",
        title="Fixture",
        slots=(MuseSlot(slot="twist", entries=("a", "b", "c")), *_FIXTURE_TABLE.slots),
    )
    before = draw_cards(_FIXTURE_TABLE, 7, 4)
    after = draw_cards(extended, 7, 4)
    assert [{k: v for k, v in card.items() if k != "twist"} for card in after] == before


def test_the_result_carries_a_digest_that_changes_when_the_table_does(tmp_path: Path) -> None:
    """A seed reproduces a card only against the same table; the digest shows which one."""
    root = tmp_path / "skill"
    table = root / "muse" / "fantasy.yaml"
    body = "genre: fantasy\ntitle: F\nslots:\n  - slot: character\n    entries: [a, b]\n"
    _write(table, body)
    tool = MuseSparkTool((BUNDLED_DATA_ROOT, root))
    first = _draw(tool, tmp_path, genre="fantasy", seed=1)["table_digest"]
    _write(table, body.replace("[a, b]", "[a, c]"))
    second = _draw(tool, tmp_path, genre="fantasy", seed=1)["table_digest"]

    assert first != second


def test_an_unknown_genre_draws_nothing_and_names_the_genres_there_are(tmp_path: Path) -> None:
    """P6: no default table stands in for a genre that has none.

    Killed by: src/uclone_x/story/muse.py :: if found is None:
    Becomes: if found is True:
    """
    result = asyncio.run(
        MuseSparkTool().execute(
            {"action": "draw", "genre": "western", "seed": 1}, _context(tmp_path)
        )
    )
    assert not result.success
    assert result.output is None
    assert result.error is not None
    assert "nothing was drawn" in result.error
    assert "fantasy, horror, mystery, romance, science-fiction" in result.error


def test_an_unknown_genre_is_refused_in_plain_words_only(tmp_path: Path) -> None:
    """The whole refusal, as the conversation shows it: no class name, no prefix.

    Killed by: src/uclone_x/story/muse.py :: class UnknownGenreError(PlainRefusalError):
    Becomes: class UnknownGenreError(Exception):
    """
    result = asyncio.run(
        MuseSparkTool().execute(
            {"action": "draw", "genre": "western", "seed": 1}, _context(tmp_path)
        )
    )
    assert result.error == (
        "There is no table for the genre 'western', so nothing was drawn. "
        "Genres: fantasy, horror, mystery, romance, science-fiction."
    )


def test_a_broken_table_is_refused_naming_the_file_in_its_root_and_the_field(
    tmp_path: Path,
) -> None:
    """A skill's damaged table is reported by file and field, relative to the skill's
    folder: no class name, no absolute path, no validator text.

    Killed by: src/uclone_x/story/muse.py :: class MuseDataError(PlainRefusalError):
    Becomes: class MuseDataError(Exception):
    """
    root = tmp_path / "skill"
    _write(
        root / "muse" / "fantasy.yaml",
        "genre: fantasy\ntitle: F\nslots:\n  - slot: omen\n    entries: [crow]\n",
    )
    result = asyncio.run(
        MuseSparkTool((BUNDLED_DATA_ROOT, root)).execute(
            {"action": "draw", "genre": "fantasy", "seed": 1}, _context(tmp_path)
        )
    )
    assert not result.success
    assert result.error == (
        "The idea tables could not be loaded, so nothing was drawn: the field "
        "'slots[0].entries' in muse/fantasy.yaml is missing or holds a value that does not fit."
    )
    assert str(tmp_path) not in result.error


@pytest.mark.parametrize("spelling", ["Science Fiction", "science_fiction", "  SCIENCE-fiction "])
def test_a_genre_matches_however_it_is_spaced_or_cased(tmp_path: Path, spelling: str) -> None:
    """The model writes a genre as a person would; the table's id uses hyphens.

    Killed by: src/uclone_x/story/muse.py :: found = tables.get(normalize_genre(genre))
    Becomes: found = tables.get(genre)
    """
    tool = MuseSparkTool()
    drawn = _draw(tool, tmp_path, genre=spelling, seed=5, count=2)
    canonical = _draw(tool, tmp_path, genre="science-fiction", seed=5, count=2)

    assert drawn["genre"] == "science-fiction"
    assert drawn["cards"] == canonical["cards"]


def test_the_description_says_how_to_list_genres_instead_of_listing_them(
    tmp_path: Path,
) -> None:
    """A list written into the description at start-up goes stale once a skill adds a
    genre (#1572), so the description sends the model to 'genres', which reads the tables
    the draw reads."""
    tool = MuseSparkTool()
    assert "Call 'genres'" in tool.description
    assert "science-fiction" not in tool.description

    listed = asyncio.run(tool.execute({"action": "genres"}, _context(tmp_path)))
    assert listed.success, listed.error
    assert listed.output == {
        "genres": [
            {"genre": "fantasy", "title": "Fantasy", "source": "bundled"},
            {"genre": "horror", "title": "Horror", "source": "bundled"},
            {"genre": "mystery", "title": "Mystery", "source": "bundled"},
            {"genre": "romance", "title": "Romance", "source": "bundled"},
            {"genre": "science-fiction", "title": "Science Fiction", "source": "bundled"},
        ]
    }


# --- the bundled data -------------------------------------------------------------------


def test_the_bundled_templates_and_tables_load() -> None:
    """The four structures the design names, and five genres whose cards have four columns."""
    templates = load_structure_templates()
    assert sorted(templates) == ["heros-journey", "kishotenketsu", "save-the-cat", "three-act"]
    assert len([b for act in templates["save-the-cat"].acts for b in act.beats]) == 15
    assert [act.id for act in templates["kishotenketsu"].acts] == ["ki", "sho", "ten", "ketsu"]

    tables = load_muse_tables()
    assert sorted(tables) == ["fantasy", "horror", "mystery", "romance", "science-fiction"]
    for table in tables.values():
        assert [s.slot for s in table.slots] == ["character", "place", "conflict", "twist"]


# --- the loader reports the file and the field ------------------------------------------

_TEMPLATE = """\
id: tiny
title: Tiny
description: Two beats.
acts:
  - id: only
    title: Only
    beats:
      - id: start
        title: Start
        purpose: Begin.
      - id: end
        title: End
        purpose: Finish.
"""


def test_a_schema_error_names_the_file_and_the_field(tmp_path: Path) -> None:
    """A missing beat title is reported where it is, not replaced by a default.

    Killed by: src/uclone_x/story/skill_data.py :: _field_path(first["loc"]) or None
    Becomes: None
    """
    bad = _write(
        tmp_path / "structures" / "tiny.yaml", _TEMPLATE.replace("        title: End\n", "")
    )

    with pytest.raises(SkillDataError) as caught:
        load_structure_templates((tmp_path,))
    assert caught.value.file == bad
    assert caught.value.field == "acts[0].beats[1].title"
    assert str(bad) in str(caught.value)
    assert "acts[0].beats[1].title" in str(caught.value)


def test_an_id_that_disagrees_with_its_file_name_is_refused(tmp_path: Path) -> None:
    """The id is what `story.yaml` will store, and the file name is how it is found again.

    Killed by: src/uclone_x/story/skill_data.py :: if getattr(loaded, id_field) != path.stem:
    Becomes: if False:
    """
    bad = _write(tmp_path / "structures" / "small.yaml", _TEMPLATE)

    with pytest.raises(SkillDataError) as caught:
        load_structure_templates((tmp_path,))
    assert (caught.value.file, caught.value.field) == (bad, "id")


def test_a_repeated_beat_id_is_refused(tmp_path: Path) -> None:
    """An outline addresses beats by id, so two with one id would be one beat.

    Killed by: src/uclone_x/story/skill_data.py :: raise ValueError(f"{what} repeat '{item}'")
    Becomes: pass
    """
    _write(tmp_path / "structures" / "tiny.yaml", _TEMPLATE.replace("id: end", "id: start"))

    with pytest.raises(SkillDataError, match="beats repeat 'start'"):
        load_structure_templates((tmp_path,))


def test_a_file_that_is_not_yaml_is_named(tmp_path: Path) -> None:
    bad = _write(tmp_path / "muse" / "noir.yaml", "genre: [unclosed\n")

    with pytest.raises(SkillDataError, match="is not valid YAML") as caught:
        load_muse_tables((tmp_path,))
    assert caught.value.file == bad


def test_a_later_root_replaces_a_table_and_adds_a_genre(tmp_path: Path) -> None:
    """How a skill supplies data: same id replaces, a new id adds.

    Killed by: src/uclone_x/story/skill_data.py :: loaded[path.stem] = Sourced(item, _source(root))
    Becomes: loaded.setdefault(path.stem, Sourced(item, _source(root)))
    """
    _write(
        tmp_path / "muse" / "fantasy.yaml",
        "genre: fantasy\ntitle: Mine\nslots:\n  - slot: omen\n    entries: [crow, comet]\n",
    )
    _write(
        tmp_path / "muse" / "noir.yaml",
        "genre: noir\ntitle: Noir\nslots:\n  - slot: vice\n    entries: [gin, dice]\n",
    )

    tables = load_muse_tables((BUNDLED_DATA_ROOT, tmp_path))
    assert tables["fantasy"].title == "Mine"
    assert tables["noir"].title == "Noir"
    assert "horror" in tables


def test_a_root_that_does_not_exist_is_an_error_not_an_empty_set(tmp_path: Path) -> None:
    """A mistyped skill path would otherwise load nothing, silently.

    Killed by: src/uclone_x/story/skill_data.py :: if not _is_folder(root, root):
    Becomes: if False:
    """
    missing = tmp_path / "nowhere"
    with pytest.raises(SkillDataError) as caught:
        load_muse_tables((BUNDLED_DATA_ROOT, missing))
    assert caught.value.file == missing
