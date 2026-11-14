"""Video projects as workspace artifacts, and the one conversation that may write each.

A video project lives in `<workspace>/videos/<project_id>/` and belongs to no conversation:
a room opens it, and deleting the room leaves it where it is.

**One writer at a time.** `project.yaml` carries a lease naming the conversation that may
write. Opening a project takes a free lease; a project whose lease another conversation
holds opens read-only, and `take_over` moves the lease on a person's request.

**Every file write is checked twice.** Against the lease, so a conversation that was taken
over is refused rather than writing over its successor; and against the digest the writer
read, so a change made outside the conversation -- a person editing the file by hand or in
VideoView -- is refused rather than lost.
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
from uclone_x_video.schemas import (
    VIDEOS_DIRNAME,
    ProjectQualityTiers,
    QualityTier,
    StyleProfile,
    default_quality_tiers,
    describe_invalid,
)

if TYPE_CHECKING:
    from uclone_x.tools.models import ToolContext

__all__ = [
    "PROJECT_FILE",
    "VIDEOS_DIRNAME",
    "NoConversationError",
    "NoOpenVideoError",
    "UnknownVideoError",
    "UnreadableVideo",
    "UnreadableVideoError",
    "VideoChangedError",
    "VideoError",
    "VideoFile",
    "VideoLease",
    "VideoLibrary",
    "VideoListing",
    "VideoOpening",
    "VideoProjectRecord",
    "VideoReadOnlyError",
    "digest_of",
    "text_digest",
]

PROJECT_FILE = "project.yaml"

_PROJECT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# One lock for every lease change in this process, so a check of who holds a project and
# the write that moves it cannot interleave with another such change.
_LEASE_LOCK = threading.Lock()
_MAX_ID_LENGTH = 80
_MAX_SLUG_LENGTH = 40
_SYSTEM_FILES = frozenset({"thumbs.db", "ehthumbs.db", "desktop.ini", "icon\r"})


class VideoError(PlainRefusalError):
    """A video project operation was refused. The message is written for a person."""


class UnknownVideoError(VideoError):
    """The id names no video project in this workspace, or a place outside the library."""


class UnreadableVideoError(VideoError):
    """A project's `project.yaml` is there and does not validate; the message names the field."""


class NoConversationError(VideoError):
    """The call runs outside a conversation, and a project is opened by a conversation."""


class NoOpenVideoError(VideoError):
    """The conversation has no video project open."""


class VideoReadOnlyError(VideoError):
    """This conversation does not hold the project's lease, so it may not write."""


class VideoChangedError(VideoError):
    """The file changed on disk since it was read, so writing was refused."""


class VideoLease(BaseModel):
    """The conversation that may currently write to this video project."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    holder: str = Field(min_length=1)
    acquired_at: str = Field(min_length=1)


class VideoProjectRecord(BaseModel):
    """The metadata of a video project (`project.yaml`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str = Field(min_length=1, max_length=_MAX_ID_LENGTH)
    title: str = Field(min_length=1)
    style_profile: StyleProfile = "cinematic_film"
    logline: str = ""
    fps: int = Field(default=30, ge=1, le=240)
    color_space: str = "Rec.709"
    active_playback_tier: QualityTier = "preview"
    quality_tiers: ProjectQualityTiers = Field(default_factory=default_quality_tiers)
    created_at: str = Field(min_length=1)
    lease: VideoLease | None = None


@dataclass(frozen=True)
class VideoFile:
    """A file read from a video project folder, carrying its SHA-256 digest."""

    relative_path: str
    text: str
    digest: str


@dataclass(frozen=True)
class VideoOpening:
    """The result of opening a video project in a conversation."""

    project: VideoProjectRecord
    writable: bool


@dataclass(frozen=True)
class UnreadableVideo:
    """A video project folder whose `project.yaml` could not be parsed."""

    id: str
    reason: str


@dataclass(frozen=True)
class VideoListing:
    """All video projects found under `<workspace>/videos/`."""

    projects: tuple[VideoProjectRecord, ...]
    unreadable: tuple[UnreadableVideo, ...] = ()


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
        slug = "video"
    return slug[:_MAX_SLUG_LENGTH].rstrip("-") or "video"


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


class VideoLibrary:
    """Manages `<workspace>/videos/<project_id>/` folders, leases, and guarded file writes."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace = workspace_root.resolve()
        self._videos_root = self._workspace / EOS_DIRNAME_INTERNAL
        self._validator = PathValidator()

    @property
    def workspace_root(self) -> Path:
        return self._workspace

    @property
    def videos_root(self) -> Path:
        return self._videos_root

    def project_dir(self, project_id: str) -> Path:
        if (
            not project_id
            or len(project_id) > _MAX_ID_LENGTH
            or _PROJECT_ID.fullmatch(project_id) is None
        ):
            raise UnknownVideoError(
                f"{project_id!r} is not a valid video project id (use lowercase letters, digits, and hyphens)."
            )
        library = self._videos_root.resolve()
        candidate = self._videos_root / project_id
        if candidate.is_symlink():
            raise UnknownVideoError(f"Video project {project_id!r} may not be a symbolic link.")
        resolved = candidate.resolve()
        if resolved.parent != library:
            raise UnknownVideoError(
                f"Video project {project_id!r} is outside the video project library."
            )
        return resolved

    def _resolve_inside(self, project_id: str, relative_path: str) -> tuple[Path, str]:
        p_dir = self.project_dir(project_id)
        rel = PurePosixPath(relative_path)
        if not relative_path or rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise VideoError(f"Invalid project-relative path {relative_path!r}.")
        clean_rel = rel.as_posix()
        try:
            full = self._validator.resolve_safe_path(Path(*rel.parts), p_dir)
        except PathTraversalError as exc:
            raise VideoError(
                f"Path {relative_path!r} escapes video project {project_id!r}."
            ) from exc
        if not full.is_relative_to(p_dir):
            raise VideoError(f"Path {relative_path!r} escapes video project {project_id!r}.")
        return full, clean_rel

    def load(self, project_id: str) -> VideoProjectRecord:
        """Load and validate `videos/<project_id>/project.yaml`."""
        p_dir = self.project_dir(project_id)
        manifest_path = p_dir / PROJECT_FILE
        if not p_dir.is_dir() or not manifest_path.is_file():
            raise UnknownVideoError(f"There is no video project {project_id!r} in this workspace.")
        try:
            raw_text = manifest_path.read_text(encoding="utf-8")
            loaded: Any = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise UnreadableVideoError(
                f"'{VIDEOS_DIRNAME}/{project_id}/{PROJECT_FILE}' is not valid YAML."
            ) from exc
        except OSError as exc:
            raise UnreadableVideoError(
                f"'{VIDEOS_DIRNAME}/{project_id}/{PROJECT_FILE}' could not be read from disk."
            ) from exc
        if not isinstance(loaded, dict):
            raise UnreadableVideoError(
                f"'{VIDEOS_DIRNAME}/{project_id}/{PROJECT_FILE}' must be a YAML mapping."
            )
        try:
            record = VideoProjectRecord.model_validate(loaded)
        except ValidationError as exc:
            raise UnreadableVideoError(
                describe_invalid(f"{VIDEOS_DIRNAME}/{project_id}/{PROJECT_FILE}", exc)
            ) from exc
        if record.id != project_id:
            raise UnreadableVideoError(
                f"'{VIDEOS_DIRNAME}/{project_id}/{PROJECT_FILE}' has id {record.id!r}, "
                f"which does not match its folder {project_id!r}."
            )
        return record

    def save_record(self, record: VideoProjectRecord) -> None:
        p_dir = self.project_dir(record.id)
        data = record.model_dump(mode="json", exclude_none=True)
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        _atomic_write(p_dir / PROJECT_FILE, text)

    def create(
        self,
        title: str,
        room_id: str,
        *,
        style_profile: StyleProfile = "cinematic_film",
        logline: str = "",
        fps: int = 30,
        project_id: str | None = None,
    ) -> VideoProjectRecord:
        """Create a new video project folder and grant its lease to `room_id`."""
        if not room_id or not room_id.strip():
            raise NoConversationError("A video project can only be created inside a conversation.")
        clean_title = title.strip()
        if not clean_title:
            raise VideoError("Video project title must not be empty.")
        self._videos_root.mkdir(parents=True, exist_ok=True)

        if project_id is not None:
            candidate = project_id
            if (self._videos_root / candidate).exists():
                raise VideoError(f"Video project {candidate!r} already exists.")
        else:
            slug = _slugify(clean_title)
            candidate = f"{slug}-{secrets.token_hex(2)}"
            while (self._videos_root / candidate).exists():
                candidate = f"{slug}-{secrets.token_hex(2)}"

        now = _now_iso()
        record = VideoProjectRecord(
            id=candidate,
            title=clean_title,
            style_profile=style_profile,
            logline=logline,
            fps=fps,
            created_at=now,
            quality_tiers=default_quality_tiers(),
            lease=VideoLease(holder=room_id, acquired_at=now),
        )
        p_dir = self.project_dir(candidate)
        for sub in ("media", "vault", "proxies", "exports", "index", "tracks", "proposals"):
            (p_dir / sub).mkdir(parents=True, exist_ok=True)
        self.save_record(record)
        return record

    def list_projects(self) -> VideoListing:
        """List all video projects in `<workspace>/videos/`."""
        if not self._videos_root.is_dir():
            return VideoListing(projects=(), unreadable=())
        projects: list[VideoProjectRecord] = []
        unreadable: list[UnreadableVideo] = []
        for child in sorted(self._videos_root.iterdir(), key=lambda p: p.name):
            if child.name.startswith(".") or child.name.casefold() in _SYSTEM_FILES:
                continue
            if not child.is_dir() or child.is_symlink():
                continue
            if _PROJECT_ID.fullmatch(child.name) is None:
                continue
            if not (child / PROJECT_FILE).is_file():
                continue
            try:
                projects.append(self.load(child.name))
            except VideoError as exc:
                unreadable.append(UnreadableVideo(id=child.name, reason=str(exc)))
        return VideoListing(projects=tuple(projects), unreadable=tuple(unreadable))

    def open(self, project_id: str, room_id: str) -> VideoOpening:
        """Open `project_id` in `room_id`, acquiring the lease if free."""
        if not room_id or not room_id.strip():
            raise NoConversationError("A video project can only be opened inside a conversation.")
        with _LEASE_LOCK:
            record = self.load(project_id)
            if record.lease is None:
                leased = record.model_copy(
                    update={"lease": VideoLease(holder=room_id, acquired_at=_now_iso())}
                )
                self.save_record(leased)
                return VideoOpening(project=leased, writable=True)
            if record.lease.holder == room_id:
                return VideoOpening(project=record, writable=True)
            return VideoOpening(project=record, writable=False)

    def take_over(self, project_id: str, room_id: str) -> tuple[VideoProjectRecord, str | None]:
        """Move `project_id`'s lease to `room_id`, returning `(updated_record, previous_holder)`."""
        if not room_id or not room_id.strip():
            raise NoConversationError(
                "A video project lease can only be taken over by a conversation."
            )
        with _LEASE_LOCK:
            record = self.load(project_id)
            previous = record.lease.holder if record.lease is not None else None
            updated = record.model_copy(
                update={"lease": VideoLease(holder=room_id, acquired_at=_now_iso())}
            )
            self.save_record(updated)
            return updated, previous

    def take_over_if_held_by(
        self, project_id: str, room_id: str, *, expected_holder: str | None
    ) -> bool:
        """Move the lease to `room_id` only if `expected_holder` still holds it (`None`: nobody).

        The check and the write happen under one lock, so a project another conversation
        took in between is left with it. Returns whether the lease moved.
        """
        if not room_id or not room_id.strip():
            raise NoConversationError(
                "A video project lease can only be taken over by a conversation."
            )
        with _LEASE_LOCK:
            record = self.load(project_id)
            current = record.lease.holder if record.lease is not None else None
            if current != expected_holder:
                return False
            self.save_record(
                record.model_copy(
                    update={"lease": VideoLease(holder=room_id, acquired_at=_now_iso())}
                )
            )
            return True

    def release(self, project_id: str, holder: str) -> bool:
        """Release `project_id`'s lease if `holder` currently holds it."""
        with _LEASE_LOCK:
            record = self.load(project_id)
            if record.lease is None or record.lease.holder != holder:
                return False
            updated = record.model_copy(update={"lease": None})
            self.save_record(updated)
            return True

    def require_writable(self, project_id: str, room_id: str | None) -> VideoProjectRecord:
        """Verify that `room_id` holds the lease on `project_id`."""
        if not room_id or not room_id.strip():
            raise NoConversationError("Writing to a video project requires an active conversation.")
        record = self.load(project_id)
        if record.lease is None or record.lease.holder != room_id:
            holder = record.lease.holder if record.lease is not None else "no conversation"
            raise VideoReadOnlyError(
                f"Video project {project_id!r} is read-only in this conversation "
                f"(lease is held by {holder!r}). Take over the lease to write."
            )
        return record

    def read_file(self, project_id: str, relative_path: str) -> VideoFile | None:
        """Read a UTF-8 file inside `videos/<project_id>/`, returning `None` if absent."""
        self.load(project_id)
        full, clean_rel = self._resolve_inside(project_id, relative_path)
        if not full.is_file():
            return None
        raw = full.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VideoError(
                f"'{clean_rel}' in video project {project_id!r} is not UTF-8 text."
            ) from exc
        return VideoFile(relative_path=clean_rel, text=text, digest=digest_of(raw))

    def write_file(
        self,
        project_id: str,
        relative_path: str,
        content: str,
        *,
        room_id: str | None,
        expected_digest: str | None,
    ) -> VideoFile:
        """Write `content` to `relative_path` after checking both the lease and `expected_digest`.

        Pass `expected_digest=None` when creating a new file that must not exist yet.
        """
        self.require_writable(project_id, room_id)
        full, clean_rel = self._resolve_inside(project_id, relative_path)
        if clean_rel == PROJECT_FILE:
            raise VideoError(f"'{PROJECT_FILE}' is managed by the video library directly.")
        if clean_rel.startswith("vault/"):
            raise VideoError(
                f"'{clean_rel}' is inside the immutable media vault and cannot be overwritten."
            )

        new_digest = text_digest(content)
        if full.exists():
            current_digest = digest_of(full.read_bytes())
            if expected_digest is None:
                raise VideoChangedError(
                    f"'{clean_rel}' already exists on disk (digest {current_digest[:12]}), "
                    "so creating it as new was refused."
                )
            if current_digest != expected_digest:
                raise VideoChangedError(
                    f"'{clean_rel}' changed on disk since it was read "
                    f"(expected {expected_digest[:12]}, found {current_digest[:12]}). "
                    "Read it again before writing."
                )
            if current_digest == new_digest:
                return VideoFile(relative_path=clean_rel, text=content, digest=new_digest)
        else:
            if expected_digest is not None:
                raise VideoChangedError(
                    f"'{clean_rel}' was deleted on disk since it was read "
                    f"(expected {expected_digest[:12]})."
                )

        _atomic_write(full, content)
        return VideoFile(relative_path=clean_rel, text=content, digest=new_digest)

    @classmethod
    def for_context(cls, context: ToolContext) -> tuple[VideoLibrary, str, str | None]:
        """Return `(VideoLibrary, open_project_id, room_id)` from `ToolContext`."""
        workspace = context.require_workspace()
        if not context.story_id:
            raise NoOpenVideoError(
                "No video project is open in this conversation. "
                "Use 'video_library' to create or open a project first."
            )
        return cls(workspace), context.story_id, context.room_id


EOS_DIRNAME_INTERNAL = VIDEOS_DIRNAME
