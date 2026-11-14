"""Songs as workspace artifacts, and the one conversation that may write each.

A song lives in `<workspace>/songs/<song_id>/` and belongs to no conversation: a room
opens it, and deleting the room leaves it where it is.

**One writer at a time.** `song.yaml` carries a lease naming the conversation that may
write. Opening a song takes a free lease; a song whose lease another conversation holds
opens read-only, and `take_over` moves the lease on a person's request.

**Every file write is checked twice.** Against the lease, so a conversation that was taken
over is refused rather than writing over its successor; and against the digest the
writer read, so a change made outside the conversation -- a person editing the file by
hand or in MusicView -- is refused rather than lost.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import tempfile
import threading
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uclone_x.errors import PathTraversalError, PlainRefusalError
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x_music.schemas import (
    SONGS_DIRNAME,
    StyleAxioms,
    StyleProfile,
    default_style_axioms,
    describe_invalid,
)

if TYPE_CHECKING:
    from uclone_x.tools.models import ToolContext

__all__ = [
    "SONGS_DIRNAME",
    "SONG_FILE",
    "MusicError",
    "NoConversationError",
    "NoOpenSongError",
    "SongChangedError",
    "SongFile",
    "SongLease",
    "SongLibrary",
    "SongListing",
    "SongOpening",
    "SongReadOnlyError",
    "SongRecord",
    "UnknownSongError",
    "UnreadableSong",
    "UnreadableSongError",
    "digest_of",
    "text_digest",
]

SONG_FILE = "song.yaml"

_SONG_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# One lock for every lease change in this process, so a check of who holds a song and the
# write that moves it cannot interleave with another such change.
_LEASE_LOCK = threading.Lock()
_MAX_ID_LENGTH = 80
_MAX_SLUG_LENGTH = 40
_SYSTEM_FILES = frozenset({"thumbs.db", "ehthumbs.db", "desktop.ini", "icon\r"})


class MusicError(PlainRefusalError):
    """A song operation was refused. The message is written for a person."""


class UnknownSongError(MusicError):
    """The id names no song in this workspace, or a place outside the song library."""


class UnreadableSongError(MusicError):
    """A song's `song.yaml` is there and does not validate; the message names the field."""


class NoConversationError(MusicError):
    """The call runs outside a conversation, and a song is opened by a conversation."""


class NoOpenSongError(MusicError):
    """The conversation has no song open."""


class SongReadOnlyError(MusicError):
    """This conversation does not hold the song's lease, so it may not write."""


class SongChangedError(MusicError):
    """The file changed on disk since it was read, so writing was refused."""


class SongLease(BaseModel):
    """The conversation that may currently write to this song."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    holder: str = Field(min_length=1)
    acquired_at: str = Field(min_length=1)


class SongRecord(BaseModel):
    """The metadata of a song (`song.yaml`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str = Field(min_length=1, max_length=_MAX_ID_LENGTH)
    title: str = Field(min_length=1)
    genre: str = "pop_ballad"
    logline: str = ""
    created_at: str = Field(min_length=1)
    style_axioms: StyleAxioms = Field(default_factory=default_style_axioms)
    lease: SongLease | None = None


@dataclass(frozen=True)
class SongFile:
    """A file read from a song folder, carrying its SHA-256 digest."""

    relative_path: str
    text: str
    digest: str


@dataclass(frozen=True)
class SongOpening:
    """The result of opening a song in a conversation."""

    song: SongRecord
    writable: bool


@dataclass(frozen=True)
class UnreadableSong:
    """A song folder whose `song.yaml` could not be parsed."""

    id: str
    reason: str


@dataclass(frozen=True)
class SongListing:
    """All songs found under `<workspace>/songs/`."""

    songs: tuple[SongRecord, ...]
    unreadable: tuple[UnreadableSong, ...] = ()


def digest_of(data: bytes) -> str:
    """Return the hex SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def text_digest(text: str) -> str:
    """Return the hex SHA-256 digest of UTF-8 `text`."""
    return digest_of(text.encode("utf-8"))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slugify(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title).casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    if not slug:
        slug = "song"
    return slug[:_MAX_SLUG_LENGTH].rstrip("-") or "song"


def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", text=True)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        tmp_path.replace(target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


class SongLibrary:
    """Manages `<workspace>/songs/<song_id>/` folders, leases, and guarded file writes."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace = workspace_root.resolve()
        self._songs_root = self._workspace / SONGS_DIRNAME
        self._validator = PathValidator()

    @property
    def workspace_root(self) -> Path:
        return self._workspace

    @property
    def songs_root(self) -> Path:
        return self._songs_root

    def song_dir(self, song_id: str) -> Path:
        if not song_id or len(song_id) > _MAX_ID_LENGTH or _SONG_ID.fullmatch(song_id) is None:
            raise UnknownSongError(
                f"{song_id!r} is not a valid song id (use lowercase letters, digits, and hyphens)."
            )
        library = self._songs_root.resolve()
        candidate = self._songs_root / song_id
        if candidate.is_symlink():
            raise UnknownSongError(f"Song {song_id!r} may not be a symbolic link.")
        resolved = candidate.resolve()
        if resolved.parent != library:
            raise UnknownSongError(f"Song {song_id!r} is outside the song library.")
        return resolved

    def _resolve_inside(self, song_id: str, relative_path: str) -> tuple[Path, str]:
        s_dir = self.song_dir(song_id)
        rel = PurePosixPath(relative_path)
        if not relative_path or rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise MusicError(f"Invalid song-relative path {relative_path!r}.")
        clean_rel = rel.as_posix()
        try:
            full = self._validator.resolve_safe_path(Path(*rel.parts), s_dir)
        except PathTraversalError as exc:
            raise MusicError(f"Path {relative_path!r} escapes song {song_id!r}.") from exc
        if not full.is_relative_to(s_dir):
            raise MusicError(f"Path {relative_path!r} escapes song {song_id!r}.")
        return full, clean_rel

    def load(self, song_id: str) -> SongRecord:
        """Load and validate `songs/<song_id>/song.yaml`."""
        s_dir = self.song_dir(song_id)
        manifest_path = s_dir / SONG_FILE
        if not s_dir.is_dir() or not manifest_path.is_file():
            raise UnknownSongError(f"There is no song {song_id!r} in this workspace.")
        try:
            raw_text = manifest_path.read_text(encoding="utf-8")
            loaded: Any = yaml.safe_load(raw_text)
        except Exception as exc:
            raise UnreadableSongError(
                f"'{SONGS_DIRNAME}/{song_id}/{SONG_FILE}' could not be read: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise UnreadableSongError(
                f"'{SONGS_DIRNAME}/{song_id}/{SONG_FILE}' must be a YAML mapping."
            )
        try:
            record = SongRecord.model_validate(loaded)
        except ValidationError as exc:
            raise UnreadableSongError(
                describe_invalid(f"{SONGS_DIRNAME}/{song_id}/{SONG_FILE}", exc)
            ) from exc
        if record.id != song_id:
            raise UnreadableSongError(
                f"'{SONGS_DIRNAME}/{song_id}/{SONG_FILE}' has id {record.id!r}, "
                f"which does not match its folder {song_id!r}."
            )
        return record

    def save_record(self, record: SongRecord) -> None:
        s_dir = self.song_dir(record.id)
        data = record.model_dump(mode="json", exclude_none=True)
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        _atomic_write(s_dir / SONG_FILE, text)

    def create(
        self,
        title: str,
        room_id: str,
        *,
        genre: str = "pop_ballad",
        logline: str = "",
        style_profile: StyleProfile = "pop_ballad",
        song_id: str | None = None,
    ) -> SongRecord:
        """Create a new song folder and grant its lease to `room_id`."""
        if not room_id or not room_id.strip():
            raise NoConversationError("A song can only be created inside a conversation.")
        clean_title = title.strip()
        if not clean_title:
            raise MusicError("Song title must not be empty.")
        self._songs_root.mkdir(parents=True, exist_ok=True)

        if song_id is not None:
            candidate = song_id
            if (self._songs_root / candidate).exists():
                raise MusicError(f"Song {candidate!r} already exists.")
        else:
            slug = _slugify(clean_title)
            candidate = f"{slug}-{secrets.token_hex(2)}"
            while (self._songs_root / candidate).exists():
                candidate = f"{slug}-{secrets.token_hex(2)}"

        now = _now_iso()
        record = SongRecord(
            id=candidate,
            title=clean_title,
            genre=genre,
            logline=logline,
            created_at=now,
            style_axioms=default_style_axioms(style_profile),
            lease=SongLease(holder=room_id, acquired_at=now),
        )
        s_dir = self.song_dir(candidate)
        (s_dir / "orchestra").mkdir(parents=True, exist_ok=True)
        (s_dir / "tracks").mkdir(parents=True, exist_ok=True)
        (s_dir / "proposals").mkdir(parents=True, exist_ok=True)
        self.save_record(record)
        return record

    def list_songs(self) -> SongListing:
        """List all songs in `<workspace>/songs/`."""
        if not self._songs_root.is_dir():
            return SongListing(songs=(), unreadable=())
        songs: list[SongRecord] = []
        unreadable: list[UnreadableSong] = []
        for child in sorted(self._songs_root.iterdir(), key=lambda p: p.name):
            if child.name.startswith(".") or child.name.casefold() in _SYSTEM_FILES:
                continue
            if not child.is_dir() or child.is_symlink():
                continue
            if _SONG_ID.fullmatch(child.name) is None:
                continue
            if not (child / SONG_FILE).is_file():
                continue
            try:
                songs.append(self.load(child.name))
            except MusicError as exc:
                unreadable.append(UnreadableSong(id=child.name, reason=str(exc)))
        return SongListing(songs=tuple(songs), unreadable=tuple(unreadable))

    def open(self, song_id: str, room_id: str) -> SongOpening:
        """Open `song_id` in `room_id`, acquiring the lease if free."""
        if not room_id or not room_id.strip():
            raise NoConversationError("A song can only be opened inside a conversation.")
        record = self.load(song_id)
        if record.lease is None:
            leased = record.model_copy(
                update={"lease": SongLease(holder=room_id, acquired_at=_now_iso())}
            )
            self.save_record(leased)
            return SongOpening(song=leased, writable=True)
        if record.lease.holder == room_id:
            return SongOpening(song=record, writable=True)
        return SongOpening(song=record, writable=False)

    def take_over(self, song_id: str, room_id: str) -> tuple[SongRecord, str | None]:
        """Move `song_id`'s lease to `room_id`, returning `(updated_record, previous_holder)`."""
        if not room_id or not room_id.strip():
            raise NoConversationError("A song lease can only be taken over by a conversation.")
        record = self.load(song_id)
        previous = record.lease.holder if record.lease is not None else None
        updated = record.model_copy(
            update={"lease": SongLease(holder=room_id, acquired_at=_now_iso())}
        )
        self.save_record(updated)
        return updated, previous

    def take_over_if_held_by(
        self, song_id: str, room_id: str, *, expected_holder: str | None
    ) -> bool:
        """Move the lease to `room_id` only if `expected_holder` still holds it (`None`: nobody).

        The check and the write happen under one lock, so a song another conversation took in
        between is left with it. Returns whether the lease moved.
        """
        if not room_id or not room_id.strip():
            raise NoConversationError("A song lease can only be taken over by a conversation.")
        with _LEASE_LOCK:
            record = self.load(song_id)
            current = record.lease.holder if record.lease is not None else None
            if current != expected_holder:
                return False
            self.save_record(
                record.model_copy(
                    update={"lease": SongLease(holder=room_id, acquired_at=_now_iso())}
                )
            )
            return True

    def release(self, song_id: str, holder: str) -> bool:
        """Release `song_id`'s lease if `holder` currently holds it."""
        record = self.load(song_id)
        if record.lease is None or record.lease.holder != holder:
            return False
        updated = record.model_copy(update={"lease": None})
        self.save_record(updated)
        return True

    def require_writable(self, song_id: str, room_id: str | None) -> SongRecord:
        """Verify that `room_id` holds the lease on `song_id`."""
        if not room_id or not room_id.strip():
            raise NoConversationError("Writing to a song requires an active conversation.")
        record = self.load(song_id)
        if record.lease is None or record.lease.holder != room_id:
            holder = record.lease.holder if record.lease is not None else "no conversation"
            raise SongReadOnlyError(
                f"Song {song_id!r} is read-only in this conversation (lease is held by {holder!r}). "
                "Take over the lease to write."
            )
        return record

    def read_file(self, song_id: str, relative_path: str) -> SongFile | None:
        """Read a file inside `songs/<song_id>/`, returning `None` if it does not exist."""
        self.load(song_id)
        full, clean_rel = self._resolve_inside(song_id, relative_path)
        if not full.is_file():
            return None
        raw = full.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MusicError(f"'{clean_rel}' in song {song_id!r} is not UTF-8 text.") from exc
        return SongFile(relative_path=clean_rel, text=text, digest=digest_of(raw))

    def write_file(
        self,
        song_id: str,
        relative_path: str,
        content: str,
        *,
        room_id: str | None,
        expected_digest: str | None,
    ) -> SongFile:
        """Write `content` to `relative_path` after checking both the lease and `expected_digest`.

        Pass `expected_digest=None` when creating a new file that must not exist yet.
        """
        self.require_writable(song_id, room_id)
        full, clean_rel = self._resolve_inside(song_id, relative_path)
        if clean_rel == SONG_FILE:
            raise MusicError(f"'{SONG_FILE}' is managed by the song library directly.")

        new_digest = text_digest(content)
        if full.exists():
            current_digest = digest_of(full.read_bytes())
            if expected_digest is None:
                raise SongChangedError(
                    f"'{clean_rel}' already exists on disk (digest {current_digest[:12]}), "
                    "so creating it as new was refused."
                )
            if current_digest != expected_digest:
                raise SongChangedError(
                    f"'{clean_rel}' changed on disk since it was read "
                    f"(expected {expected_digest[:12]}, found {current_digest[:12]}). "
                    "Read it again before writing."
                )
            if current_digest == new_digest:
                # The file already holds exactly this: nothing is written, and the
                # caller reads `digest == expected_digest` as "nothing changed" (#2429).
                return SongFile(relative_path=clean_rel, text=content, digest=new_digest)
        else:
            if expected_digest is not None:
                raise SongChangedError(
                    f"'{clean_rel}' was deleted on disk since it was read "
                    f"(expected {expected_digest[:12]})."
                )

        _atomic_write(full, content)
        return SongFile(relative_path=clean_rel, text=content, digest=new_digest)

    @classmethod
    def for_context(cls, context: ToolContext) -> tuple[SongLibrary, str, str | None]:
        """Return `(SongLibrary, open_song_id, room_id)` from `ToolContext`."""
        workspace = context.require_workspace()
        if not context.story_id:
            raise NoOpenSongError(
                "No song is open in this conversation. Use 'music_library' to create or open a song first."
            )
        return cls(workspace), context.story_id, context.room_id
