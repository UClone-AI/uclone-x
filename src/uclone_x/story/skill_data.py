"""Story structure templates and muse tables: data, not code (#1552, #1559).

Both are YAML files under a *data root*::

    <root>/structures/<template_id>.yaml
    <root>/muse/<genre>.yaml

The bundled root ships with the package. A caller may pass further roots after it; a file in
a later root replaces the file with the same id in an earlier one, which is how a skill adds
or replaces a template or a genre table.

Every read validates. A file that does not fit its schema is never replaced by a default:
loading fails with `SkillDataError`, which names the file and the field (P6).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from uclone_x.errors import UCloneXError

__all__ = [
    "BUNDLED_DATA_ROOT",
    "MUSE_DIRNAME",
    "STRUCTURES_DIRNAME",
    "Beat",
    "MuseSlot",
    "MuseTable",
    "SkillDataError",
    "StructureAct",
    "StructureTemplate",
    "load_muse_tables",
    "load_structure_templates",
]

#: The data root that ships with the package.
BUNDLED_DATA_ROOT = Path(__file__).parent / "bundled"
STRUCTURES_DIRNAME = "structures"
MUSE_DIRNAME = "muse"

#: An id is a file name, so it is kept to what every file system and a model's JSON agree on.
_ID = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
_ID_RE = re.compile(_ID)


class SkillDataError(UCloneXError):
    """A structure template or muse table could not be used; names the file and the field.

    `str()` is for the log: the full path and the validator's own text. `plain` is for a
    person: the file relative to its data root, the field, and the reason in plain words.
    """

    def __init__(
        self,
        file: Path,
        field: str | None,
        message: str,
        *,
        reason: str,
        root: Path | None = None,
    ) -> None:
        self.file = file
        self.field = field
        self.root = root
        self.reason = reason
        where = f"{file}" if field is None else f"{file}, field '{field}'"
        super().__init__(f"{where}: {message}")

    @property
    def relative_file(self) -> str:
        """The file as its data root names it, e.g. `muse/fantasy.yaml`."""
        # Display only, not a containment check: the loader sets `root` to the folder the
        # file was found under.
        if self.root is None:
            return self.file.name
        try:
            return self.file.relative_to(self.root).as_posix()
        except ValueError:
            return self.file.name

    @property
    def plain(self) -> str:
        """The file, the field and the reason, with no path outside the data root."""
        where = self.relative_file
        if self.field is not None:
            where = f"the field '{self.field}' in {where}"
        return f"{where} {self.reason}"


class _Strict(BaseModel):
    # Not `strict=True`: YAML gives lists where these models keep tuples, and every field
    # is text, which lax mode does not coerce from a number either.
    model_config = ConfigDict(frozen=True, extra="forbid")


class Beat(_Strict):
    """One beat of a structure: a turn the story has to take, not its content."""

    id: str = Field(pattern=_ID)
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


class StructureAct(_Strict):
    """A named group of beats, in order."""

    id: str = Field(pattern=_ID)
    title: str = Field(min_length=1)
    beats: tuple[Beat, ...] = Field(min_length=1)


class StructureTemplate(_Strict):
    """A story structure (three acts, Save the Cat, ...) that an outline is built from."""

    id: str = Field(pattern=_ID)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    acts: tuple[StructureAct, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> Self:
        _require_unique("acts", [act.id for act in self.acts])
        _require_unique("beats", [beat.id for act in self.acts for beat in act.beats])
        return self


class MuseSlot(_Strict):
    """One column of a genre's table: a card draws one entry from each slot."""

    slot: str = Field(pattern=_ID)
    entries: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _entries_distinct(self) -> Self:
        if any(not entry.strip() for entry in self.entries):
            raise ValueError("an entry is blank")
        _require_unique("entries", list(self.entries))
        return self


class MuseTable(_Strict):
    """A genre's idea table."""

    genre: str = Field(pattern=_ID)
    title: str = Field(min_length=1)
    slots: tuple[MuseSlot, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _slots_unique(self) -> Self:
        _require_unique("slots", [slot.slot for slot in self.slots])
        return self

    def digest(self) -> str:
        """A short fingerprint of the table's content.

        A seed reproduces a card only against the same table, so a draw reports this beside
        the seed: if the table was edited since, the digest says so.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _require_unique(what: str, ids: list[str]) -> None:
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            raise ValueError(f"{what} repeat '{item}'")
        seen.add(item)


_M = TypeVar("_M", bound=BaseModel)


def _field_path(loc: tuple[int | str, ...]) -> str:
    path = ""
    for part in loc:
        path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else part)
    return path


def _load_file(root: Path, path: Path, model: type[_M], id_field: str) -> _M:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise SkillDataError(
            path, None, f"could not be read ({exc})", reason="could not be read", root=root
        ) from exc
    except yaml.YAMLError as exc:
        raise SkillDataError(
            path, None, f"is not valid YAML ({exc})", reason="is not valid YAML", root=root
        ) from exc
    if not isinstance(raw, dict):
        raise SkillDataError(
            path,
            None,
            "must be a YAML mapping",
            reason="is not laid out as named fields",
            root=root,
        )
    try:
        loaded = model.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = _field_path(first["loc"]) or None
        raise SkillDataError(
            path,
            field,
            first["msg"],
            reason="is missing or holds a value that does not fit",
            root=root,
        ) from exc
    if getattr(loaded, id_field) != path.stem:
        raise SkillDataError(
            path,
            id_field,
            f"is '{getattr(loaded, id_field)}' but the file is named '{path.stem}'",
            reason="does not match the file's name",
            root=root,
        )
    return loaded


def _load_all(roots: Sequence[Path], dirname: str, model: type[_M], id_field: str) -> dict[str, _M]:
    loaded: dict[str, _M] = {}
    for root in roots:
        if not root.is_dir():
            raise SkillDataError(
                root, None, "the data root does not exist", reason="is not a folder that exists"
            )
        directory = root / dirname
        if not directory.is_dir():
            # A root may carry templates and no muse tables, or the reverse. The bundled
            # root carries both, so a missing half there is damage, not a choice.
            if root == BUNDLED_DATA_ROOT:
                raise SkillDataError(
                    directory,
                    None,
                    "the bundled data is missing",
                    reason="is missing from the bundled data",
                    root=root,
                )
            continue
        for path in sorted(directory.glob("*.yaml")):
            if not _ID_RE.match(path.stem):
                raise SkillDataError(
                    path,
                    None,
                    f"the file name must match {_ID}",
                    reason="has a name that is not lowercase letters, digits and hyphens",
                    root=root,
                )
            loaded[path.stem] = _load_file(root, path, model, id_field)
    return loaded


def load_structure_templates(
    roots: Sequence[Path] = (BUNDLED_DATA_ROOT,),
) -> dict[str, StructureTemplate]:
    """Every structure template under `roots`, by id; a later root replaces an earlier one."""
    return _load_all(roots, STRUCTURES_DIRNAME, StructureTemplate, "id")


def load_muse_tables(roots: Sequence[Path] = (BUNDLED_DATA_ROOT,)) -> dict[str, MuseTable]:
    """Every genre's muse table under `roots`, by genre; a later root replaces an earlier one."""
    return _load_all(roots, MUSE_DIRNAME, MuseTable, "genre")
