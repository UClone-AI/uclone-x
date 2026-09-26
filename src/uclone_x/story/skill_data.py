"""Story structure templates and muse tables: data, not code (#1552, #1559).

Both are YAML files under a *data root*::

    <root>/structures/<template_id>.yaml
    <root>/muse/<genre>.yaml

The bundled root ships with the package. A caller may pass further roots after it; a file in
a later root replaces the file with the same id in an earlier one, which is how a skill adds
or replaces a template or a genre table (#1572). A skill package carries its root at
`<skill>/resources/story/` (`SKILL_DATA_PARTS`); `data_roots` puts the active skills' roots
after the bundled one, in skill folder-name order, and the `*_sourced` loaders say which root each id
came from, so a result can name the skill whose table it used.

What the loader checks on a skill's data, refusing the whole call when one fails:

* the data root, the `muse`/`structures` folder and each file resolve inside the skill's
  folder, and each file is then opened one folder at a time from the skill's folder
  without following a link, so a folder swapped for a link after that check is refused
  rather than followed out;
* the file name follows the id rule, so it does not start with a dot;
* the file is a regular file, not a folder, FIFO or other special file;
* the file is at most `MAX_DATA_FILE_BYTES`, and repeats nothing by YAML alias;
* a data file that is a link that loops, a `muse` or `structures` folder that cannot be
  listed, and a data folder inside a folder the process may not search are refused as
  "could not be read" rather than passed on as an exception. A data folder that is
  missing, or is itself a link that loops, supplies nothing.

These checks are meant to keep what is read within what `compute_skill_sha256` walks at
approval; that walk fails on a folder it cannot list, or on a link that loops under a
name that does not start with a dot, rather than skipping it. Nothing is
re-hashed when it is read.

Every read validates. A file that does not fit its schema is never replaced by a default:
loading fails with `SkillDataError`, which names the file and the field (P6).
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Self, TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from uclone_x.errors import PathTraversalError, UCloneXError
from uclone_x.sandbox.path_validator import PathValidator

__all__ = [
    "BUNDLED_DATA_ROOT",
    "MAX_DATA_FILE_BYTES",
    "MUSE_DIRNAME",
    "STRUCTURES_DIRNAME",
    "Beat",
    "MuseSlot",
    "MuseTable",
    "SKILL_DATA_PARTS",
    "SkillDataError",
    "StructureAct",
    "StructureTemplate",
    "data_roots",
    "load_muse_tables",
    "load_structure_templates",
    "Sourced",
    "load_muse_tables_sourced",
    "load_structure_templates_sourced",
]

#: The data root that ships with the package.
BUNDLED_DATA_ROOT = Path(__file__).parent / "bundled"
STRUCTURES_DIRNAME = "structures"
MUSE_DIRNAME = "muse"
#: Where a skill package keeps its story data: `<skill>/resources/story/structures/*.yaml`
#: and `<skill>/resources/story/muse/*.yaml`. `resources/` is the package folder
#: `docs/skill-system-architecture.md` §2 names for templates. Kept as parts, not a relative
#: `Path`, because it only ever means something joined onto a skill's own folder.
SKILL_DATA_PARTS: tuple[str, ...] = ("resources", "story")

#: An id is a file name, so it is kept to what every file system and a model's JSON agree on.
#: A data file larger than this is refused before it is read. The biggest bundled table is a
#: few kilobytes; the cap keeps one oversized file a skill carries from filling the prompt.
MAX_DATA_FILE_BYTES = 256 * 1024

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
            relative = self.file.relative_to(self.root).as_posix()
        except ValueError:
            return self.file.name
        if relative == "." and _skill_home(self.root) is not None:
            return "/".join(SKILL_DATA_PARTS)
        return relative

    @property
    def plain(self) -> str:
        """The file, the field and the reason, with no path outside the data root.

        A file a skill supplied also names the skill, since that is where a person fixes it.
        """
        where = self.relative_file
        skill = _skill_of(self.root)
        if skill is not None:
            where = f"{where} of the skill '{skill}'"
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


def _uses_alias(text: str) -> bool:
    """Whether the YAML repeats a node by alias (`*name`), which is how a small file expands."""
    # PyYAML ships no types for its event stream; each item is only tested with isinstance.
    events = cast(
        "Iterable[object]",
        yaml.parse(text, Loader=yaml.SafeLoader),  # pyright: ignore[reportUnknownMemberType]
    )
    return any(isinstance(event, yaml.AliasEvent) for event in events)


def _read_regular(root: Path, path: Path, read_from: Path, beneath: Path | None = None) -> str:
    """The text of `read_from`, read through one descriptor: a regular file, at most the cap.

    Opened without blocking, so a FIFO cannot hang the call, and without following a link,
    so the file checked is the file read. `fstat` on that descriptor decides; the read is
    bounded, so a file that grew past the cap after any earlier look is still refused. The
    descriptor is closed on every path, a refusal included.

    With `beneath` (a skill's resolved folder), `read_from` is opened one folder at a time
    from there, none of them through a link (`_open_beneath`), so a folder swapped for a
    link after `read_from` was resolved is refused rather than followed out of the skill.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(read_from, flags) if beneath is None else _open_beneath(beneath, read_from, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SkillDataError(
                path,
                None,
                "is not a regular file (a folder, FIFO, device or socket)",
                reason="is not a regular file",
                root=root,
            )
        chunks: list[bytes] = []
        remaining = MAX_DATA_FILE_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    if len(data) > MAX_DATA_FILE_BYTES:
        raise SkillDataError(
            path,
            None,
            f"is over the {MAX_DATA_FILE_BYTES}-byte limit",
            reason=f"is larger than {MAX_DATA_FILE_BYTES // 1024} KB",
            root=root,
        )
    return data.decode("utf-8")


def _open_beneath(folder: Path, target: Path, flags: int) -> int:
    """Open `target`, a resolved path inside the resolved `folder`, walking down from `folder`.

    Each folder on the way is opened relative to the one before it (`dir_fd`) and without
    following a link, and the file itself is opened with `flags`. `target` was resolved, so
    none of its parts was a link then; a part that is a link now was swapped in since, and
    the open fails (`ELOOP` or `ENOTDIR`) instead of leaving the skill. `folder` itself is
    opened by its path, so this covers the folders inside the skill, not the skill's own
    folder or those above it.

    `target` is opened by its path, as before this walk existed, where the platform cannot
    open relative to a folder (Windows), and when it is `folder` itself (a data file that
    links to the skill's folder), which `fstat` then refuses as not a regular file.
    """
    try:
        parts = target.relative_to(folder).parts
    except ValueError:  # the skill's folder resolves elsewhere than it did a moment ago
        raise PermissionError(errno.EACCES, "not inside the skill's folder", str(target)) from None
    if not parts or os.open not in os.supports_dir_fd:
        return os.open(target, flags)
    walk_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    folder_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            above, folder_fd = folder_fd, os.open(part, walk_flags, dir_fd=folder_fd)
            os.close(above)
        return os.open(parts[-1], flags, dir_fd=folder_fd)
    finally:
        os.close(folder_fd)


def _load_file(
    root: Path,
    path: Path,
    read_from: Path,
    model: type[_M],
    id_field: str,
    beneath: Path | None = None,
) -> _M:
    try:
        text = _read_regular(root, path, read_from, beneath)
        if _uses_alias(text):
            raise SkillDataError(
                path,
                None,
                "repeats a node by alias ('*name')",
                reason="repeats part of itself with a YAML alias ('*'), which is not allowed",
                root=root,
            )
        raw: Any = yaml.safe_load(text)
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


def _load_all(
    roots: Sequence[Path], dirname: str, model: type[_M], id_field: str
) -> dict[str, Sourced[_M]]:
    loaded: dict[str, Sourced[_M]] = {}
    for root in roots:
        home = _skill_home(root)
        if not _is_folder(root, root):
            raise SkillDataError(
                root, None, "the data root does not exist", reason="is not a folder that exists"
            )
        beneath: Path | None = None
        if home is not None:
            _require_inside(home, root, root)
            beneath = _resolved(root, home)
        directory = root / dirname
        if not _is_folder(directory, root):
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
        if home is not None:
            _require_inside(home, root, directory)
        for path in _yaml_files(directory, root):
            if not _ID_RE.match(path.stem):
                raise SkillDataError(
                    path,
                    None,
                    f"the file name must match {_ID}",
                    reason="has a name that is not lowercase letters, digits and hyphens",
                    root=root,
                )
            read_from = _resolved(root, path) if home is None else _require_inside(home, root, path)
            item = _load_file(root, path, read_from, model, id_field, beneath)
            loaded[path.stem] = Sourced(item, _source(root))
    return loaded


def _require_inside(home: Path, root: Path, path: Path) -> Path:
    """The resolved path of a skill's data file or folder; refused if it is outside the skill.

    The loader calls this on the data root, on the `muse`/`structures` folder and on each
    file, so a file is refused when it, or a data folder it is listed through, resolves
    outside the skill's folder, wherever the file itself then resolves. The containment
    itself is the one workspace guard, not a second copy.
    """
    try:
        return PathValidator().resolve_safe_path(path.absolute(), home)
    except PathTraversalError as exc:
        raise SkillDataError(
            path,
            None,
            f"resolves outside the skill's folder {home} ({exc})",
            reason="leads out of the skill's folder through a link, which is not allowed",
            root=root,
        ) from exc
    except (RuntimeError, OSError) as exc:  # a link that loops: RuntimeError before 3.13
        raise _unreadable(root, path, exc) from exc


def _resolved(root: Path, path: Path) -> Path:
    """`path.resolve()`, with a link that loops refused in plain words, not as an exception."""
    try:
        return path.resolve()
    except (RuntimeError, OSError) as exc:
        raise _unreadable(root, path, exc) from exc


#: The errors that mean "there is no folder here" rather than "the folder could not be looked
#: at", as `Path.is_dir()` treats them on Python 3.11 to 3.13.
_ABSENT = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})


def _is_folder(path: Path, root: Path) -> bool:
    """Whether `path` is a folder, with a stat that is refused turned into a plain refusal.

    A missing path, or a link that loops, is not a folder. Any other error, such as a
    parent the process may not search, is refused. This calls `os.stat` rather than
    `Path.is_dir()` because Python 3.14's `is_dir()` answers False for every error, which
    would let a folder the process may not search supply nothing instead of being refused.
    """
    try:
        mode = os.stat(path).st_mode
    except OSError as exc:
        if exc.errno in _ABSENT:
            return False
        raise _unreadable(root, path, exc) from exc  # stat refused
    return stat.S_ISDIR(mode)


def _yaml_files(directory: Path, root: Path) -> list[Path]:
    """The `*.yaml` entries of a data folder, sorted; a folder that cannot be listed is refused.

    `Path.glob` answers nothing for a folder it may not list (e.g. mode 000), which would let
    a skill silently supply nothing; `os.scandir` raises, and that becomes the plain refusal.
    """
    try:
        with os.scandir(directory) as entries:
            names = [entry.name for entry in entries if entry.name.endswith(".yaml")]
    except OSError as exc:  # listing refused
        raise _unreadable(root, directory, exc) from exc
    return sorted(directory / name for name in names)


def _unreadable(root: Path, path: Path, exc: BaseException) -> SkillDataError:
    """The plain "could not be read" refusal; the exception and full path go to the log only."""
    return SkillDataError(
        path, None, f"could not be read ({exc!r})", reason="could not be read", root=root
    )


def _skill_home(root: Path | None) -> Path | None:
    """The skill package folder a data root sits in, as written; `None` for other roots."""
    if root is None or root == BUNDLED_DATA_ROOT:
        return None
    if root.parts[-len(SKILL_DATA_PARTS) :] != SKILL_DATA_PARTS:
        return None
    return root.parents[len(SKILL_DATA_PARTS) - 1]


def _skill_of(root: Path | None) -> str | None:
    """The skill package a data root belongs to, by its folder name; `None` for others."""
    home = _skill_home(root)
    return None if home is None else home.name


def data_roots(
    skill_dirs: Sequence[Path] = (), base: Sequence[Path] = (BUNDLED_DATA_ROOT,)
) -> tuple[Path, ...]:
    """`base` (the bundled root), then the story data root of each skill that carries one.

    `skill_dirs` are the package folders of the skills active for the calling agent
    (`ToolContext.skill_dirs`). They are sorted by folder name here, so which of two skills
    wins an id they both supply does not depend on the order they were registered in, and
    `Sourced.source` names the winner. A skill with no `resources/story/` folder
    supplies nothing and is not an error: most skills are not about stories.
    """
    found = sorted(
        (
            d.joinpath(*SKILL_DATA_PARTS)
            for d in skill_dirs
            if _is_folder(d.joinpath(*SKILL_DATA_PARTS), d.joinpath(*SKILL_DATA_PARTS))
        ),
        key=lambda root: root.parents[len(SKILL_DATA_PARTS) - 1].name,
    )
    return (*base, *found)


def _source(root: Path) -> str:
    """A data root in words, for a result: never a path outside it."""
    if root == BUNDLED_DATA_ROOT:
        return "bundled"
    skill = _skill_of(root)
    return f"skill '{skill}'" if skill is not None else f"folder '{root.name}'"


@dataclass(frozen=True)
class Sourced(Generic[_M]):
    """A loaded template or table and where it came from: `bundled` or `skill '<folder>'`."""

    item: _M
    source: str


def load_structure_templates(
    roots: Sequence[Path] = (BUNDLED_DATA_ROOT,),
) -> dict[str, StructureTemplate]:
    """Every structure template under `roots`, by id; a later root replaces an earlier one."""
    return {k: v.item for k, v in load_structure_templates_sourced(roots).items()}


def load_structure_templates_sourced(
    roots: Sequence[Path] = (BUNDLED_DATA_ROOT,),
) -> dict[str, Sourced[StructureTemplate]]:
    """`load_structure_templates`, with the root each template came from."""
    return _load_all(roots, STRUCTURES_DIRNAME, StructureTemplate, "id")


def load_muse_tables(roots: Sequence[Path] = (BUNDLED_DATA_ROOT,)) -> dict[str, MuseTable]:
    """Every genre's muse table under `roots`, by genre; a later root replaces an earlier one."""
    return {k: v.item for k, v in load_muse_tables_sourced(roots).items()}


def load_muse_tables_sourced(
    roots: Sequence[Path] = (BUNDLED_DATA_ROOT,),
) -> dict[str, Sourced[MuseTable]]:
    """`load_muse_tables`, with the root each table came from."""
    return _load_all(roots, MUSE_DIRNAME, MuseTable, "genre")
