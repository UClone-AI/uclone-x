"""The workspace's artifact folders, listed and managed apart from any conversation (#1554).

A conversation's Docs dock shows what *that* conversation wrote. This service answers the
questions the dock cannot: what is in the workspace now, including what a conversation
that was since deleted wrote; how to put a file away and bring it back; how to remove one
for good; and how to open a story in a new conversation.

**What it covers is stated, never implied** (#1366, P6). The survey carries a
`scope_note` saying which folders it walked and a `record_gaps` list naming each part it
could not read. A file is linked to a conversation only while that conversation exists,
because the link is the room's own record of what it wrote, and a deleted room takes that
record with it; a file with no conversation has "no recorded writer", never "written by
nobody". Names that start with a dot are not listed, and the scope note says so.

**Operations stay inside the artifact folders.** Every path is resolved through
`PathValidator.resolve_safe_path`, the one containment guard, first against the workspace
and then against the folder it names. Archive moves a file or a story folder under
`<workspace>/.archive/`, keeping its relative path, and restore moves it back; neither
overwrites anything. Delete removes for good and needs `confirm` to be exactly `True`.
Both act on one file or one whole story; any other folder is refused, since the list
never offers one and removing it whole would take files nobody chose (#1578).

**A story being written is not taken silently.** A story's `story.yaml` names the
conversation holding its writing lease. Archiving or deleting that story while the holder
still exists is refused with `StoryInUseError`, which names the conversation, unless the
caller passes `release_writer=True`, in which case the lease is given back and the
conversation's open story is cleared first -- but never while that conversation is
answering, because the running turn was given the story when it started and keeps it until
it finishes, so its story tools would go on working on a story that was just released,
moved or deleted. A lease whose holder no longer exists is stale and is released without
asking. Once the story has moved, every conversation that had it open to read has it
cleared too. A story is recognised by its folder on disk, not by the name as typed, so any
spelling the disk takes to be that folder -- another case, or `ſ` for `s` -- is still that
story, lease and all. Where more than one entry is that same file (hard links), the one
whose name matches the spelling is chosen. Neither archive nor restore moves anything
through a link. Once a story has moved, clearing it from the conversations that had it open
comes after the move, so a failure there cannot be reported as the move failing. It is
logged, and the result carries a plain `note` saying some conversations may still show
the story as open. Only a store failure is handled that way; any other error is a defect
and propagates (#1578).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import (
    PathTraversalError,
    PlainRefusalError,
    RoomNotFoundError,
    StaleRoomWriteError,
    UnreadableRoomRecordError,
)
from uclone_x.room.models import RoomState
from uclone_x.room.service import RoomService
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.story.library import (
    STORIES_DIRNAME,
    STORY_FILE,
    StoryError,
    StoryLibrary,
    StoryRecord,
    UnknownStoryError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ARCHIVE_DIRNAME",
    "ARTIFACT_ROOTS",
    "SCOPE_NOTE",
    "ArtifactContent",
    "ArtifactEntry",
    "ArtifactChanged",
    "ArtifactError",
    "ArtifactLibrary",
    "ArtifactNotFoundError",
    "ArtifactSurvey",
    "ConversationRef",
    "DeleteNotConfirmedError",
    "READERS_NOT_CLEARED_NOTE",
    "StoryFileEntry",
    "StoryInUseError",
    "StoryInfo",
    "StoryOpened",
    "StoryWriter",
]

#: The workspace folders this service lists and changes. `artifacts/` is where the image
#: and document tools save by default; `stories/` is the story library (#1555).
ARTIFACT_ROOTS: tuple[str, ...] = ("artifacts", STORIES_DIRNAME)

#: Where an archived file waits, under the workspace, at its original relative path.
ARCHIVE_DIRNAME = ".archive"

SCOPE_NOTE = (
    "This list shows the files in the workspace's artifacts and stories folders, and files "
    "a current conversation saved by name somewhere else. A file saved somewhere else by a "
    "conversation that was since deleted, or written by a shell command, may not appear "
    "here. A file is linked to a conversation only while that conversation exists. Files "
    "and folders whose names start with a dot are not listed."
)

#: The note an archive or delete carries when the conversations that had the story open
#: could not all be cleared (#1578). The move itself happened; this says what did not.
READERS_NOT_CLEARED_NOTE = (
    "Some conversations that had this story open could not be updated, so they may still "
    "show it as open."
)

#: The largest file `open_file` returns as text.
MAX_OPEN_BYTES = 1_000_000

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"})
_DOCUMENT_SUFFIXES = frozenset({".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".html"})
_STORY_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

Kind = Literal["document", "image", "story", "file"]


# -- refusals -------------------------------------------------------------------------


class ArtifactError(PlainRefusalError):
    """A file operation refused, for a reason written for a person."""


class ArtifactNotFoundError(ArtifactError):
    """The named file, story or conversation is not there."""


class DeleteNotConfirmedError(ArtifactError):
    """A permanent delete was asked for without confirmation."""


class StoryInUseError(ArtifactError):
    """The story's writing lease is held by a conversation that still exists."""

    def __init__(self, message: str, *, room_id: str, title: str | None) -> None:
        super().__init__(message)
        self.room_id = room_id
        self.title = title


# -- values ---------------------------------------------------------------------------


class ConversationRef(BaseModel):
    """A conversation that still exists, by id and title."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    room_id: str
    title: str


class StoryWriter(BaseModel):
    """The conversation holding a story's writing lease."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    room_id: str
    title: str | None = Field(description="None when the conversation is gone or unreadable.")
    exists: bool


class StoryFileEntry(BaseModel):
    """One file inside a story folder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    name: str
    kind: Kind
    size_bytes: int


class StoryInfo(BaseModel):
    """What a story entry adds to a file entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    story_id: str
    title: str | None = Field(description="None when `story.yaml` did not load.")
    unreadable_reason: str | None = None
    writer: StoryWriter | None = None
    files: tuple[StoryFileEntry, ...] = ()


class ArtifactEntry(BaseModel):
    """One listed file, or one story folder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(description="Workspace-relative, where it is now.")
    original_path: str = Field(description="Where it was before archiving; `path` if not.")
    name: str
    kind: Kind
    archived: bool
    managed: bool = Field(
        description="False for a file outside the artifact folders: it can be opened here, "
        "not archived or deleted."
    )
    size_bytes: int | None
    modified_at: str | None
    conversations: tuple[ConversationRef, ...]
    story: StoryInfo | None = None


class ArtifactSurvey(BaseModel):
    """Every listed entry, what the list covers, and what it could not read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[ArtifactEntry, ...]
    scope_note: str
    record_gaps: tuple[str, ...]


class ArtifactContent(BaseModel):
    """An opened file: its text, or for an image only its kind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    name: str
    kind: Kind
    text: str | None


class ArtifactChanged(BaseModel):
    """What an archive or delete did: where the entry is now, and anything left undone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(description="Where it is now; for a delete, the path that was removed.")
    note: str | None = Field(
        default=None,
        description="A plain sentence about a step after the change that did not complete; "
        "None when everything did.",
    )


class StoryOpened(BaseModel):
    """A story opened into a conversation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    room_id: str
    story_id: str
    title: str
    writable: bool
    note: str | None


@dataclass(frozen=True)
class _Located:
    absolute: Path
    relative: str
    root: str
    archived: bool
    #: The path without the archive prefix.
    original: str
    #: The story id when this is a story folder itself.
    story_id: str | None

    @property
    def name(self) -> str:
        """The last part of the path, which is what a message names."""
        return self.absolute.name


def _kind(path: str) -> Kind:
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _DOCUMENT_SUFFIXES:
        return "document"
    return "file"


def _stamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()


def _is_story_folder(folder: Path) -> bool:
    return (
        _STORY_ID.fullmatch(folder.name) is not None
        and not folder.is_symlink()
        and (folder / STORY_FILE).is_file()
    )


def _same_entry(entry: Path, candidate: Path) -> bool:
    """Whether the directory entry `entry` is the file `candidate` names, links unfollowed.

    Not following links is the point: a link beside the story folder names the same
    folder when followed, and a story reached through it is not recognised as one.
    """
    try:
        return os.path.samestat(os.lstat(entry), os.lstat(candidate))
    except OSError:
        return False


def _fold(name: str) -> str:
    """`name` as a disk that ignores case and normalisation compares it."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def _pick_entry(part: str, same: list[str]) -> str:
    """Of the entries that are the file `part` names, the one spelled like `part`.

    More than one entry is the same file only through hard links. Choosing among those by
    sort order would act on a different name from the one asked for (#1578).
    """
    spelled = [entry for entry in same if _fold(entry) == _fold(part)]
    return (spelled or same)[0]


class ArtifactLibrary:
    """The artifact folders of one workspace, and the conversations that wrote into them."""

    def __init__(
        self,
        workspace: Path,
        rooms: RoomService,
        *,
        turn_in_flight: Callable[[str], bool],
    ) -> None:
        """`turn_in_flight(room_id)` says whether that conversation is answering now.

        It is required, not defaulted, so the caller must say how it knows.
        """
        self._workspace = workspace.resolve()
        self._rooms = rooms
        self._turn_in_flight = turn_in_flight

    # -- listing ------------------------------------------------------------------------

    def survey(self) -> ArtifactSurvey:
        """List the artifact folders, current and archived, newest change first."""
        gaps: list[str] = []
        states = self._readable_rooms(gaps)
        titles = {state.room_id: state.title for state in states}
        written: dict[str, list[ConversationRef]] = {}
        for state in states:
            ref = ConversationRef(room_id=state.room_id, title=state.title)
            for row in state.written_files:
                relative = self._relative(row.path)
                if relative is None:
                    continue
                refs = written.setdefault(relative, [])
                if ref not in refs:
                    refs.append(ref)
        story_rooms: dict[str, list[ConversationRef]] = {}
        for state in states:
            if state.story_id is not None:
                story_rooms.setdefault(state.story_id, []).append(
                    ConversationRef(room_id=state.room_id, title=state.title)
                )

        entries: list[ArtifactEntry] = []
        skipped_links = 0
        for archived in (False, True):
            base = self._workspace / ARCHIVE_DIRNAME if archived else self._workspace
            for root in ARTIFACT_ROOTS:
                folder = base / root
                if not folder.is_dir() or folder.is_symlink():
                    continue
                listed, links = self._walk(folder, root, archived, written, gaps)
                skipped_links += links
                entries.extend(listed)
                if root == STORIES_DIRNAME:
                    entries.extend(
                        self._stories(folder, archived, written, story_rooms, titles, gaps)
                    )
        entries.extend(self._recorded_elsewhere(written, gaps))
        if skipped_links:
            gaps.append(
                f"{skipped_links} linked file(s) or folder(s) in the artifact folders were not "
                "followed, so what they point to is not listed."
            )
        entries.sort(key=lambda e: (e.modified_at or "", e.path), reverse=True)
        return ArtifactSurvey(
            entries=tuple(entries), scope_note=SCOPE_NOTE, record_gaps=tuple(gaps)
        )

    def _readable_rooms(self, gaps: list[str]) -> list[RoomState]:
        listing = self._rooms.survey_rooms()
        unreadable = len(listing.unreadable)
        states: list[RoomState] = []
        for summary in listing.rooms:
            try:
                states.append(self._rooms.get(summary.room_id))
            except RoomNotFoundError:
                continue  # deleted between the listing and the load
            except UnreadableRoomRecordError:
                unreadable += 1
        if unreadable:
            gaps.append(
                f"{unreadable} conversation(s) could not be read, so the files they saved "
                "are not linked to them here."
            )
        return states

    def _relative(self, path: str) -> str | None:
        """`path` as a workspace-relative POSIX path, or None if it is outside."""
        if not path or "\0" in path:
            return None
        try:
            resolved = PathValidator().resolve_safe_path(Path(path), self._workspace)
        except (PathTraversalError, OSError, ValueError):
            return None
        relative = resolved.relative_to(self._workspace).as_posix()
        return None if relative == "." else relative

    def _walk(
        self,
        folder: Path,
        root: str,
        archived: bool,
        written: dict[str, list[ConversationRef]],
        gaps: list[str],
    ) -> tuple[list[ArtifactEntry], int]:
        """Every plain file under `folder`, skipping dot-names, links and story folders."""
        entries: list[ArtifactEntry] = []
        links = 0
        stack = [folder]
        while stack:
            current = stack.pop()
            try:
                children = sorted(current.iterdir())
            except OSError:
                gaps.append(
                    f"The folder {current.name} could not be read, so its files are not listed."
                )
                continue
            for child in children:
                if child.name.startswith("."):
                    continue
                if child.is_symlink():
                    links += 1
                    continue
                if child.is_dir():
                    if current == folder and root == STORIES_DIRNAME and _is_story_folder(child):
                        continue  # listed whole by `_stories`
                    stack.append(child)
                    continue
                entry = self._file_entry(child, archived, written, gaps)
                if entry is not None:
                    entries.append(entry)
        return entries, links

    def _display(self, path: Path) -> str:
        return path.relative_to(self._workspace).as_posix()

    def _file_entry(
        self,
        path: Path,
        archived: bool,
        written: dict[str, list[ConversationRef]],
        gaps: list[str],
        *,
        managed: bool = True,
    ) -> ArtifactEntry | None:
        relative = self._display(path)
        original = relative.removeprefix(f"{ARCHIVE_DIRNAME}/") if archived else relative
        try:
            found = path.stat()
        except OSError:
            gaps.append(f"The file {path.name} could not be read, so it is not listed.")
            return None
        return ArtifactEntry(
            path=relative,
            original_path=original,
            name=path.name,
            kind=_kind(relative),
            archived=archived,
            managed=managed,
            size_bytes=found.st_size,
            modified_at=_stamp(found.st_mtime),
            conversations=tuple(written.get(original, ())),
        )

    def _stories(
        self,
        folder: Path,
        archived: bool,
        written: dict[str, list[ConversationRef]],
        story_rooms: dict[str, list[ConversationRef]],
        titles: dict[str, str],
        gaps: list[str],
    ) -> list[ArtifactEntry]:
        """One entry per story folder, carrying its files and who is writing it."""
        # The story library reads `<base>/stories`; for the archive, base is `.archive`.
        library = StoryLibrary(folder.parent)
        entries: list[ArtifactEntry] = []
        try:
            children = sorted(folder.iterdir())
        except OSError:
            return entries  # `_walk` already reported this folder
        for child in children:
            if child.name.startswith(".") or not child.is_dir() or not _is_story_folder(child):
                continue
            story_id = child.name
            relative = self._display(child)
            original = relative.removeprefix(f"{ARCHIVE_DIRNAME}/") if archived else relative
            title: str | None = None
            reason: str | None = None
            writer: StoryWriter | None = None
            try:
                record = library.load(story_id)
            except StoryError as exc:
                reason = str(exc)
            else:
                title = record.title
                if record.lease is not None and not archived:
                    holder = record.lease.holder
                    writer = StoryWriter(
                        room_id=holder, title=titles.get(holder), exists=holder in titles
                    )
            files: list[StoryFileEntry] = []
            newest = 0.0
            size = 0
            conversations: list[ConversationRef] = list(story_rooms.get(story_id, ()))
            for path in sorted(child.rglob("*")):
                inner = path.relative_to(child)
                if any(part.startswith(".") for part in inner.parts) or path.is_symlink():
                    continue
                if not path.is_file():
                    continue
                try:
                    found = path.stat()
                except OSError:
                    gaps.append(f"The file {path.name} could not be read, so it is not listed.")
                    continue
                newest = max(newest, found.st_mtime)
                size += found.st_size
                file_path = self._display(path)
                files.append(
                    StoryFileEntry(
                        path=file_path,
                        name=inner.as_posix(),
                        kind=_kind(file_path),
                        size_bytes=found.st_size,
                    )
                )
                file_original = f"{original}/{inner.as_posix()}"
                for ref in written.get(file_original, ()):
                    if ref not in conversations:
                        conversations.append(ref)
            entries.append(
                ArtifactEntry(
                    path=relative,
                    original_path=original,
                    name=title or story_id,
                    kind="story",
                    archived=archived,
                    managed=True,
                    size_bytes=size,
                    modified_at=_stamp(newest) if newest else None,
                    conversations=tuple(conversations),
                    story=StoryInfo(
                        story_id=story_id,
                        title=title,
                        unreadable_reason=reason,
                        writer=writer,
                        files=tuple(files),
                    ),
                )
            )
        return entries

    def _recorded_elsewhere(
        self, written: dict[str, list[ConversationRef]], gaps: list[str]
    ) -> list[ArtifactEntry]:
        """Files a current conversation recorded outside the artifact folders."""
        entries: list[ArtifactEntry] = []
        for relative in sorted(written):
            head = relative.split("/", 1)[0]
            if head in ARTIFACT_ROOTS or head == ARCHIVE_DIRNAME:
                continue
            path = self._workspace / relative
            if path.is_symlink() or not path.is_file():
                continue  # removed since, or not a plain file
            entry = self._file_entry(path, False, written, gaps, managed=False)
            if entry is not None:
                entries.append(entry)
        return entries

    # -- opening --------------------------------------------------------------------------

    def open_file(self, path: str) -> ArtifactContent:
        """The text of a listed file, or for an image only its kind.

        Opens a file in the artifact folders (current or archived), or one a current
        conversation recorded writing; nothing else in the workspace.
        """
        relative = self._relative(path)
        if relative is None:
            raise ArtifactNotFoundError("That file is not in the workspace, so it was not opened.")
        head = relative.split("/", 1)[0]
        if head == ARCHIVE_DIRNAME:
            head = relative.split("/", 2)[1] if relative.count("/") >= 1 else ""
        if head not in ARTIFACT_ROOTS and not self._recorded(relative):
            raise ArtifactError("That file is not one the clones saved, so it is not opened here.")
        absolute = self._workspace / relative
        if not absolute.is_file():
            raise ArtifactNotFoundError(
                f"There is no file called {absolute.name} there any more. Refresh the list to see "
                "what is there."
            )
        kind = _kind(relative)
        if kind == "image":
            return ArtifactContent(path=relative, name=absolute.name, kind=kind, text=None)
        size = absolute.stat().st_size
        if size > MAX_OPEN_BYTES:
            raise ArtifactError(
                f"{absolute.name} is too large to show here ({size // 1000} KB; the limit is "
                f"{MAX_OPEN_BYTES // 1000} KB)."
            )
        try:
            text = absolute.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactError(
                f"{absolute.name} is not text, so it cannot be shown here."
            ) from exc
        return ArtifactContent(path=relative, name=absolute.name, kind=kind, text=text)

    def _recorded(self, relative: str) -> bool:
        for state in self._readable_rooms([]):
            for row in state.written_files:
                if self._relative(row.path) == relative:
                    return True
        return False

    # -- archive, restore, delete -----------------------------------------------------------

    def archive(self, path: str, *, release_writer: bool = False) -> ArtifactChanged:
        """Move a file or story folder under `.archive/`; say where it is now."""
        located = self._locate(path)
        if located.archived:
            raise ArtifactError(f"{located.name} is already archived.")
        self._refuse_plain_folder(located)
        target = self._workspace / ARCHIVE_DIRNAME / located.relative
        self._check_destination(target, located.name)
        if target.exists() or target.is_symlink():
            raise ArtifactError(
                f"An archived copy of {located.name} is already there. Restore or delete "
                "that copy first."
            )
        if located.story_id is not None:
            self._settle_lease(located.story_id, release_writer)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(located.absolute, target)
        note = None
        if located.story_id is not None:
            note = self._forget_story(located.story_id, "archived")
        return ArtifactChanged(path=self._display(target), note=note)

    def restore(self, path: str) -> str:
        """Move an archived file or story folder back; return where it is now."""
        located = self._locate(path)
        if not located.archived:
            raise ArtifactError(f"{located.name} is not archived, so there is nothing to restore.")
        target = self._workspace / located.original
        self._check_destination(target, located.name)
        if target.exists() or target.is_symlink():
            raise ArtifactError(
                f"Something is already where {located.name} was. Move or delete it, then restore."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(located.absolute, target)
        return located.original

    def delete(self, path: str, *, confirm: bool, release_writer: bool = False) -> ArtifactChanged:
        """Remove a file or story folder for good. Needs `confirm=True`."""
        if confirm is not True:
            raise DeleteNotConfirmedError(
                "Deleting cannot be undone, so it needs your confirmation. Nothing was deleted."
            )
        located = self._locate(path)
        self._refuse_plain_folder(located)
        leaving = None if located.archived else located.story_id
        if leaving is not None:
            self._settle_lease(leaving, release_writer)
        if located.absolute.is_dir():
            shutil.rmtree(located.absolute)
        else:
            located.absolute.unlink()
        note = None
        if leaving is not None:
            note = self._forget_story(leaving, "deleted")
        return ArtifactChanged(path=located.relative, note=note)

    @staticmethod
    def _refuse_plain_folder(located: _Located) -> None:
        """Refuse a folder that is not a story: it would go whole, with files nobody chose."""
        if located.story_id is None and located.absolute.is_dir():
            raise ArtifactError(
                f"{located.name} is a folder. Only a single file or a whole story can be "
                "archived or deleted here, so nothing was changed."
            )

    def _forget_story(self, story_id: str, done: Literal["archived", "deleted"]) -> str | None:
        """Clear a story that has just left from the conversations that had it open.

        The archive or delete has already happened, so a failure here must not read as it
        failing ("refresh and try again" for something done). It is logged, and returned as
        a note for the result to carry, so the person is told as well as the log (#1578).
        Only what `forget_story` raises for a store it could not read or write is handled so:
        any other error is a defect, and a defect caught here would be a warning nobody reads.
        """
        try:
            self._rooms.forget_story(story_id)
        except (StaleRoomWriteError, OSError):
            logger.warning(
                "Story %r was %s, but the conversations that had it open were not all updated",
                story_id,
                done,
                exc_info=True,
            )
            return READERS_NOT_CLEARED_NOTE
        return None

    def _check_destination(self, target: Path, name: str) -> None:
        """Refuse a move whose destination is reached through a link.

        `target` is built from the workspace and a checked relative path, so it is inside
        the workspace as written; a link along the way (a planted `.archive`, or a linked
        folder under a root) would carry the file somewhere else, where this service
        could neither restore nor delete it. Resolving it must change nothing.
        """
        try:
            resolved = PathValidator().resolve_safe_path(target, self._workspace)
        except PathTraversalError as exc:
            raise ArtifactError(
                f"The place {name} would move to is reached through a link, so it was not moved."
            ) from exc
        if resolved != target:
            raise ArtifactError(
                f"The place {name} would move to is reached through a link, so it was not moved."
            )

    def _true_parts(self, resolved: Path) -> tuple[str, ...]:
        """`resolved`'s parts under the workspace, spelled as the folders spell them.

        On a disk that ignores case, `stories/NIGHT-TRAIN` reaches `stories/night-train`,
        and a check made on the name as typed would not see the story, or its lease. Each
        part that exists is replaced by the directory entry that is the same file. The
        entry is chosen by what it is, never by comparing names: APFS also takes
        `stories/ſunſet` to be `stories/sunset`, which `str.lower()` does not (#1578).
        """
        current = self._workspace
        spelled: list[str] = []
        for part in resolved.relative_to(self._workspace).parts:
            candidate = current / part
            name = part
            if candidate.exists():
                entries = sorted(os.listdir(current))
                if part not in entries:
                    same = [e for e in entries if _same_entry(current / e, candidate)]
                    if same:
                        name = _pick_entry(part, same)
            spelled.append(name)
            current = current / name
        return tuple(spelled)

    def _locate(self, path: str) -> _Located:
        """Resolve `path` to a file or story folder this service may change."""
        cleaned = path.strip()
        if not cleaned or "\0" in cleaned:
            raise ArtifactError("No file was named, so nothing was changed.")
        lexical = self._workspace / cleaned
        if lexical.is_symlink():
            raise ArtifactError(
                "That path is a link to somewhere else, so it was not changed here."
            )
        validator = PathValidator()
        try:
            resolved = validator.resolve_safe_path(Path(cleaned), self._workspace)
        except (PathTraversalError, ValueError) as exc:
            raise ArtifactError(
                "That path is outside the workspace, so nothing was changed."
            ) from exc
        parts = self._true_parts(resolved)
        resolved = self._workspace.joinpath(*parts)
        archived = bool(parts) and parts[0] == ARCHIVE_DIRNAME
        body = parts[1:] if archived else parts
        if len(body) < 2 or body[0] not in ARTIFACT_ROOTS:
            raise ArtifactError(
                "Only files in the artifacts and stories folders can be archived or deleted here."
            )
        base = self._workspace / ARCHIVE_DIRNAME if archived else self._workspace
        try:
            validator.resolve_safe_path(resolved, base / body[0])
        except PathTraversalError as exc:  # a root reached through a link
            raise ArtifactError(
                "Only files in the artifacts and stories folders can be archived or deleted here."
            ) from exc
        original = "/".join(body)
        if not resolved.exists():
            raise ArtifactNotFoundError(
                f"There is no file called {resolved.name} there any more. Refresh the list to see "
                "what is there."
            )
        story_id: str | None = None
        if body[0] == STORIES_DIRNAME:
            story_folder = base / STORIES_DIRNAME / body[1]
            if _is_story_folder(story_folder):
                if len(body) > 2:
                    raise ArtifactError(
                        f"{resolved.name} is part of the story '{body[1]}'. Archive or delete the "
                        "whole story instead."
                    )
                story_id = body[1]
        return _Located(
            absolute=resolved,
            relative="/".join(parts),
            root=body[0],
            archived=archived,
            original=original,
            story_id=story_id,
        )

    def _settle_lease(self, story_id: str, release_writer: bool) -> None:
        """Refuse, or give back, the writing lease before a story leaves the library."""
        library = StoryLibrary(self._workspace)
        try:
            record = library.load(story_id)
        except StoryError:
            # `story.yaml` does not load, so no lease can be read -- and none is honoured:
            # the story tools refuse an unreadable story the same way.
            return
        if record.lease is None:
            return
        holder = record.lease.holder
        state: RoomState | None = None
        exists = True
        try:
            state = self._rooms.get(holder)
        except RoomNotFoundError:
            exists = False
        except UnreadableRoomRecordError:
            state = None  # there, and unreadable: treated as existing
        name = f"“{state.title}”" if state is not None and state.title else "another conversation"
        title = state.title if state is not None else None
        if exists and not release_writer:
            raise StoryInUseError(
                f"The conversation {name} is writing this story. Delete that conversation "
                "first, or go ahead anyway to stop it writing to this story.",
                room_id=holder,
                title=title,
            )
        if exists and self._turn_in_flight(holder):
            # The running turn keeps this story until it finishes, and its story tools
            # would go on working on a story that was just released, moved or deleted.
            raise StoryInUseError(
                f"The conversation {name} is answering right now, so it cannot be stopped "
                "from writing this story yet. Wait for the answer to finish, then try again.",
                room_id=holder,
                title=title,
            )
        library.release(story_id, holder)
        if state is not None and state.story_id == story_id:
            self._rooms.set_story(holder, None)
        logger.info("Released story %r from conversation %s before moving it", story_id, holder)

    # -- stories into conversations ---------------------------------------------------------

    def open_story_in_conversation(self, story_id: str, room_id: str) -> StoryOpened:
        """Open `story_id` in the conversation `room_id` and make it the room's story.

        Writable when the lease is free, already this room's, or held by a conversation
        that no longer exists (a stale lease, taken over). Otherwise opened to read, and
        `note` says which conversation is writing it.
        """
        try:
            self._rooms.get(room_id)
        except RoomNotFoundError as exc:
            raise ArtifactNotFoundError(
                "That conversation no longer exists, so the story was not opened."
            ) from exc
        library = StoryLibrary(self._workspace)
        try:
            opening = library.open(story_id, room_id)
        except UnknownStoryError as exc:
            raise ArtifactNotFoundError(
                f"There is no story called '{story_id}' any more. Refresh the list to see the "
                "stories there are."
            ) from exc
        record: StoryRecord = opening.story
        writable = opening.writable
        note: str | None = None
        if not writable and record.lease is not None:
            holder = record.lease.holder
            try:
                holder_title: str | None = self._rooms.get(holder).title
                holder_exists = True
            except RoomNotFoundError:
                holder_title, holder_exists = None, False
            except UnreadableRoomRecordError:
                holder_title, holder_exists = None, True
            if holder_exists:
                name = f"“{holder_title}”" if holder_title else "another conversation"
                note = (
                    f"The conversation {name} is writing this story, so this one can read it "
                    "but not change it."
                )
            else:
                record, _ = library.take_over(story_id, room_id)
                writable = True
        self._rooms.set_story(room_id, story_id)
        return StoryOpened(
            room_id=room_id,
            story_id=story_id,
            title=record.title,
            writable=writable,
            note=note,
        )
