"""Stories as workspace artifacts, and the one conversation that may write each (#1555).

A story lives in `<workspace>/stories/<story_id>/` and belongs to no conversation: a room
opens it, and deleting the room leaves it where it is. The design is #1552, section 3.

**One writer at a time.** `story.yaml` carries a lease naming the conversation that may
write. Opening a story takes a free lease; a story whose lease another conversation holds
opens read-only, and `take_over` moves the lease on a person's request. The lease has no
expiry: a stopped conversation is released by `take_over` or by deleting it.

**Every file write is checked twice.** Against the lease, so a conversation that was taken
over is refused rather than writing over its successor; and against the digest the
writer read, so a change made outside the conversation -- a person editing the file by
hand -- is refused rather than lost.

**These are checks, not locks.** Each operation reads, compares and then replaces the file,
with no lock across the three, which is the room store's contract too
(`room/store.py`). Inside one process the operations are synchronous and never await, so
two conversations served by one Core cannot interleave inside one. Two processes writing
the same story at the same instant can; the owner's ruling (2026-09-24) is that stories are
written by sequential sessions only, and this is the window that ruling leaves.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uclone_x.errors import PathTraversalError, PlainRefusalError
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.story.schemas import describe_invalid

if TYPE_CHECKING:
    from uclone_x.tools.models import ToolContext

__all__ = [
    "STORIES_DIRNAME",
    "STORY_FILE",
    "NoConversationError",
    "NoOpenStoryError",
    "StoryError",
    "StoryFile",
    "StoryLease",
    "StoryLibrary",
    "StoryListing",
    "StoryOpening",
    "StoryReadOnlyError",
    "StoryRecord",
    "StoryChangedError",
    "UnknownStoryError",
    "UnreadableStory",
    "UnreadableStoryError",
    "digest_of",
]

STORIES_DIRNAME = "stories"
STORY_FILE = "story.yaml"

_STORY_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MAX_ID_LENGTH = 80
_MAX_SLUG_LENGTH = 40


class StoryError(PlainRefusalError):
    """A story operation was refused. The message is written for a person."""


class UnknownStoryError(StoryError):
    """The id names no story in this workspace, or a place outside the story library."""


class UnreadableStoryError(StoryError):
    """A story's `story.yaml` is there and does not validate; the message names the field."""


class NoConversationError(StoryError):
    """The call runs outside a conversation, and a story is opened by a conversation."""


class NoOpenStoryError(StoryError):
    """The conversation has no story open."""


class StoryReadOnlyError(StoryError):
    """This conversation does not hold the story's lease, so it may not write."""


class StoryChangedError(StoryError):
    """The file changed after the writer read it, so the write would lose that change."""


class StoryLease(BaseModel):
    """Which conversation may write the story, and since when."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    holder: str = Field(min_length=1, description="The conversation (room) id.")
    since: str = Field(min_length=1, description="ISO 8601 UTC time the lease was taken.")


class StoryRecord(BaseModel):
    """`story.yaml`: what the story is, and who is writing it.

    `extra="allow"` because later phases add fields to it (`axioms`, the structure
    template). What is typed is validated strictly, and a failure names the field (P6);
    nothing is defaulted in its place.
    """

    model_config = ConfigDict(frozen=True, extra="allow", strict=True)

    story_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    genre: str | None = None
    #: The story in one or two sentences.
    logline: str | None = None
    #: How the story is written: voice, tense, point of view, what to avoid.
    style_notes: str | None = None
    lease: StoryLease | None = None


@dataclass(frozen=True)
class StoryOpening:
    """The result of opening a story: the record, and whether this conversation may write."""

    story: StoryRecord
    writable: bool
    #: True when this call took the lease and wrote `story.yaml`.
    acquired: bool


@dataclass(frozen=True)
class UnreadableStory:
    """A story folder whose `story.yaml` did not load, and why."""

    story_id: str
    reason: str


@dataclass(frozen=True)
class StoryListing:
    """Every story in the library that loads, and every one that did not."""

    stories: tuple[StoryRecord, ...]
    unreadable: tuple[UnreadableStory, ...]


@dataclass(frozen=True)
class StoryFile:
    """A file's text and the digest a later write must quote."""

    text: str
    digest: str


def digest_of(path: Path) -> str | None:
    """The digest of the file at `path`, or `None` when there is no file.

    A digest of the bytes rather than a counter or a modification time: an edit made
    outside the Core moves it without the Core having to see the edit happen, and an
    mtime can stay put across an edit made inside one second.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    return hashlib.sha256(data).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _when(iso: str) -> str:
    """A lease time as a person reads it."""
    try:
        return datetime.fromisoformat(iso).astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso


def _slug(title: str) -> str:
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]+", folded.lower())
    slug = "-".join(words)[:_MAX_SLUG_LENGTH].strip("-")
    return slug or "story"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class StoryLibrary:
    """The stories in one workspace: where each lives, and who may write it."""

    def __init__(self, workspace_root: Path) -> None:
        self._dir = workspace_root.resolve() / STORIES_DIRNAME

    @property
    def directory(self) -> Path:
        """`<workspace>/stories`."""
        return self._dir

    # -- where a story is ------------------------------------------------------------

    def root(self, story_id: str) -> Path:
        """The folder of the story `story_id`, refusing an id that names none.

        Raises:
            UnknownStoryError: `story_id` is not a story id, resolves outside the library
                (a symlinked folder), or names no story.
        """
        if len(story_id) > _MAX_ID_LENGTH or not _STORY_ID.fullmatch(story_id):
            raise UnknownStoryError(
                f"'{story_id}' is not a story name. Story names use lowercase letters, "
                "digits and hyphens; list the stories to see them."
            )
        library = self._dir.resolve()
        folder = (library / story_id).resolve()
        if folder.parent != library:
            raise UnknownStoryError(
                f"The story folder '{story_id}' points outside the story library, so it "
                "was not opened."
            )
        if not (folder / STORY_FILE).is_file():
            raise UnknownStoryError(
                f"There is no story called '{story_id}'. List the stories to see the ones "
                "there are."
            )
        return folder

    def load(self, story_id: str) -> StoryRecord:
        """The story's `story.yaml`, validated.

        Raises:
            UnknownStoryError: see `root`.
            UnreadableStoryError: the file does not parse or does not validate.
        """
        return self._load(self.root(story_id), story_id)

    def list(self) -> StoryListing:
        """Every story in the library, by id; an unreadable one is reported, not skipped."""
        stories: list[StoryRecord] = []
        unreadable: list[UnreadableStory] = []
        if not self._dir.is_dir():
            return StoryListing((), ())
        for folder in sorted(self._dir.iterdir()):
            if not _STORY_ID.fullmatch(folder.name) or not (folder / STORY_FILE).is_file():
                continue  # not a story folder
            try:
                stories.append(self.load(folder.name))
            except StoryError as exc:
                unreadable.append(UnreadableStory(folder.name, str(exc)))
        return StoryListing(tuple(stories), tuple(unreadable))

    # -- opening and the lease -------------------------------------------------------

    def create(
        self,
        title: str,
        conversation_id: str,
        *,
        genre: str | None = None,
        logline: str | None = None,
        style_notes: str | None = None,
    ) -> StoryRecord:
        """Make a new story, open for writing in `conversation_id`."""
        cleaned = title.strip()
        if not cleaned:
            raise StoryError("A new story needs a title.")
        self._dir.mkdir(parents=True, exist_ok=True)
        slug = _slug(cleaned)
        for _ in range(16):
            story_id = f"{slug}-{secrets.token_hex(2)}"
            folder = self._dir / story_id
            try:
                folder.mkdir()
            except FileExistsError:
                continue
            now = _now_iso()
            record = StoryRecord(
                story_id=story_id,
                title=cleaned,
                created_at=now,
                genre=genre,
                logline=logline,
                style_notes=style_notes,
                lease=StoryLease(holder=conversation_id, since=now),
            )
            self._save(folder, record)
            return record
        raise StoryError("Could not find a free name for the new story. Try another title.")

    def open(self, story_id: str, conversation_id: str) -> StoryOpening:
        """Open a story in `conversation_id`: writable if the lease is free or already ours."""
        folder = self.root(story_id)
        record = self._load(folder, story_id)
        if record.lease is not None and record.lease.holder == conversation_id:
            return StoryOpening(record, writable=True, acquired=False)
        if record.lease is not None:
            return StoryOpening(record, writable=False, acquired=False)
        taken = self._with_lease(record, conversation_id)
        self._save(folder, taken)
        return StoryOpening(taken, writable=True, acquired=True)

    def take_over(self, story_id: str, conversation_id: str) -> tuple[StoryRecord, str | None]:
        """Move the lease to `conversation_id`; return the record and the previous holder."""
        folder = self.root(story_id)
        record = self._load(folder, story_id)
        previous = record.lease.holder if record.lease is not None else None
        if previous == conversation_id:
            return record, previous
        taken = self._with_lease(record, conversation_id)
        self._save(folder, taken)
        return taken, previous

    def release(self, story_id: str, conversation_id: str) -> bool:
        """Give up the lease if `conversation_id` holds it; return whether it did.

        A conversation that no longer holds the lease releases nothing: it must not free a
        story another conversation took over.
        """
        folder = self.root(story_id)
        record = self._load(folder, story_id)
        if record.lease is None or record.lease.holder != conversation_id:
            return False
        self._save(folder, record.model_copy(update={"lease": None}))
        return True

    # -- the story's files -----------------------------------------------------------

    def read_file(self, story_id: str, relative: str) -> StoryFile:
        """A file in the story, with the digest a write of it must quote."""
        target = self._member(self.root(story_id), relative)
        try:
            data = target.read_bytes()
        except FileNotFoundError as exc:
            raise StoryError(f"The story has no file '{relative}'.") from exc
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StoryError(
                f"'{relative}' is not saved as UTF-8 text, so it was not read. Save it as "
                "UTF-8 text and try again."
            ) from exc
        return StoryFile(text=text, digest=hashlib.sha256(data).hexdigest()[:16])

    def read_file_if_present(self, story_id: str, relative: str) -> StoryFile | None:
        """`read_file`, or `None` when the story has no such file (yet)."""
        target = self._member(self.root(story_id), relative)
        if not target.exists():
            return None
        return self.read_file(story_id, relative)

    def files_in(self, story_id: str, relative_dir: str) -> list[str]:
        """The story-relative names of the files directly in `relative_dir`, sorted.

        Hidden files (a leading `.`) are left out; a missing directory has none.
        """
        folder = self.root(story_id)
        directory = self._member(folder, relative_dir)
        if not directory.is_dir():
            return []
        prefix = PurePosixPath(relative_dir)
        return sorted(
            str(prefix / entry.name)
            for entry in directory.iterdir()
            if not entry.name.startswith(".") and entry.is_file()
        )

    def write_file(
        self,
        story_id: str,
        relative: str,
        text: str,
        *,
        conversation_id: str,
        expected_digest: str | None,
    ) -> str:
        """Write a file in the story; return its new digest.

        `expected_digest` is the digest `read_file` returned, or `None` for a file that
        must not exist yet.

        Raises:
            StoryReadOnlyError: `conversation_id` does not hold the lease.
            StoryChangedError: the file is not at `expected_digest`.
        """
        folder = self.root(story_id)
        target = self._member(folder, relative)
        self._require_writer(folder, story_id, conversation_id)
        current = digest_of(target)
        if current != expected_digest:
            if expected_digest is None:
                reason = f"'{relative}' already exists, and this write would replace it"
            elif current is None:
                reason = f"'{relative}' was removed after it was read"
            else:
                reason = (
                    f"'{relative}' changed after it was read, perhaps edited outside "
                    f"this conversation"
                )
            raise StoryChangedError(
                f"{reason}, so nothing was written. Read it again and make the change "
                "on the current version."
            )
        _write_atomic(target, text)
        new = digest_of(target)
        assert new is not None
        return new

    def require_writer(self, story_id: str, conversation_id: str) -> StoryRecord:
        """The story's record, if `conversation_id` holds its lease; refuse otherwise.

        The one check every write goes through. `conversation_id` is the *writer's*
        conversation -- the room a seat is in, or the conversation an agent was asked from
        -- never the story id alone, so an agent working for a conversation that was taken
        over is refused like the conversation itself.

        Raises:
            StoryReadOnlyError: another conversation, or none, holds the lease.
        """
        return self._require_writer(self.root(story_id), story_id, conversation_id)

    def write_for(
        self, context: ToolContext, relative: str, text: str, *, expected_digest: str | None
    ) -> str:
        """`write_file` for a tool call: the story and the writer both come from `context`.

        A story tool takes neither from the model. The story is the one the call's
        conversation has open and the writer is that conversation, so an agent acting for
        any other conversation -- or for one that was taken over -- is refused with a reason.

        Raises:
            NoConversationError: the call is not part of a conversation.
            NoOpenStoryError: the conversation has no story open.
            StoryReadOnlyError, StoryChangedError: see `write_file`.
        """
        story_id, conversation_id = self.writer_of(context)
        return self.write_file(
            story_id,
            relative,
            text,
            conversation_id=conversation_id,
            expected_digest=expected_digest,
        )

    @staticmethod
    def writer_of(context: ToolContext) -> tuple[str, str]:
        """The story a tool call works on and the conversation writing it, or a refusal."""
        if context.room_id is None:
            raise NoConversationError(
                "Stories are written inside a conversation, and this call is not part of "
                "one, so nothing was written."
            )
        if context.story_id is None:
            raise NoOpenStoryError(
                "No story is open in this conversation, so nothing was written. Open or "
                "create a story first."
            )
        return context.story_id, context.room_id

    # -- internals -------------------------------------------------------------------

    def _require_writer(self, folder: Path, story_id: str, conversation_id: str) -> StoryRecord:
        record = self._load(folder, story_id)
        lease = record.lease
        if lease is None:
            raise StoryReadOnlyError(
                f"'{record.title}' is not open for writing in this conversation, so "
                f"nothing was written. Open the story again to write to it."
            )
        if lease.holder != conversation_id:
            raise StoryReadOnlyError(
                f"Another conversation has been writing '{record.title}' since "
                f"{_when(lease.since)}, so nothing was written here. Take the story over "
                f"in this conversation to keep writing it here."
            )
        return record

    @staticmethod
    def _with_lease(record: StoryRecord, conversation_id: str) -> StoryRecord:
        return record.model_copy(
            update={"lease": StoryLease(holder=conversation_id, since=_now_iso())}
        )

    def _member(self, folder: Path, relative: str) -> Path:
        """A path inside the story, refusing an escape and `story.yaml` itself.

        `story.yaml` is refused by what the path *is*, not by how it is spelled: on a
        case-insensitive disk `Story.yaml` is the same file, and so is a link inside the
        story that points at it. Comparing names alone let both through (#1556).
        """
        pure = PurePosixPath(relative)
        if not relative or pure.is_absolute() or ".." in pure.parts:
            raise StoryError(f"'{relative}' is not a file name inside the story.")
        try:
            target = PathValidator().resolve_safe_path(Path(*pure.parts), folder)
        except PathTraversalError as exc:
            raise StoryError(f"'{relative}' points outside the story, so it was refused.") from exc
        record = folder / STORY_FILE
        if target == record or (target.exists() and target.samefile(record)):
            raise StoryError(
                f"'{relative}' is the story's own record, which holds who is writing the "
                "story. It changes when the story is opened or taken over, and is not "
                "written directly."
            )
        return target

    @staticmethod
    def _load(folder: Path, story_id: str) -> StoryRecord:
        where = f"{STORIES_DIRNAME}/{story_id}/{STORY_FILE}"
        try:
            raw: object = yaml.safe_load((folder / STORY_FILE).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise UnreadableStoryError(f"{where} could not be read as YAML.") from exc
        if not isinstance(raw, dict):
            raise UnreadableStoryError(f"{where} is not a set of named fields.")
        data = {str(k): v for k, v in cast(dict[object, Any], raw).items()}
        try:
            record = StoryRecord.model_validate(data)
        except ValidationError as exc:
            raise UnreadableStoryError(describe_invalid(where, exc)) from exc
        if record.story_id != story_id:
            raise UnreadableStoryError(
                f"{where} says it is the story '{record.story_id}', not '{story_id}'."
            )
        return record

    @staticmethod
    def _save(folder: Path, record: StoryRecord) -> None:
        text = yaml.safe_dump(record.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        _write_atomic(folder / STORY_FILE, text)
