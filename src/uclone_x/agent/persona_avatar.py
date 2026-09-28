"""Where a clone's picture is, and how it is changed.

A clone's picture is a file named after it: `<name>.<ext>` in the workspace personas
directory for one the user or the clone chose, and beside the shipped definition for a
built-in's own. `PersonaAvatarStore` is the one place that finds it and the one place that
writes it. The head's routes and the `set_avatar` tool both go through it, and general
writing tools cannot reach the directory at all (`BaseTool.resolve_write_path`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.persona_store import BUILTIN_PERSONAS_DIR
from uclone_x.errors import PlainRefusalError
from uclone_x.tools.base import replace_file

__all__ = [
    "AVATAR_FORMATS",
    "MAX_AVATAR_BYTES",
    "AvatarPersonaNotFound",
    "AvatarRecord",
    "AvatarRefused",
    "PersonaAvatarStore",
    "avatar_url",
    "sniff_image_format",
]

#: The picture formats a clone's avatar may be in, looked for in this order. Images only:
#: nothing here draws a face from the clone's name, so a clone with no picture gets the one
#: default the head ships rather than a generated one that would differ between heads.
AVATAR_FORMATS: tuple[tuple[str, str], ...] = (
    (".png", "image/png"),
    (".webp", "image/webp"),
    (".jpg", "image/jpeg"),
    (".jpeg", "image/jpeg"),
    (".gif", "image/gif"),
)

#: The largest picture accepted: 10 MiB. A face for a small round frame needs far less.
MAX_AVATAR_BYTES = 10 * 1024 * 1024

#: The suffix a format is written under, keyed by what the bytes say they are.
_SUFFIX_FOR_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

_ACCEPTED = "a PNG, JPEG, WebP or GIF image"


class AvatarRefused(PlainRefusalError):
    """A picture was not set or reset, for a reason written for a person.

    `reason_code` names which reason, so the head can say what to do without reading the
    sentence: `no_clone`, `not_saved`, `no_workspace`, `too_large`, `not_an_image`,
    `no_file`, `outside_workspace`, `no_source`.
    """


class AvatarPersonaNotFound(AvatarRefused):
    """The name is not a clone this head has loaded from a file."""


@dataclass(frozen=True)
class AvatarRecord:
    """One clone's picture as found on disk."""

    path: Path
    mime: str
    version: str


def sniff_image_format(data: bytes) -> str | None:
    """The image type `data` starts as, by its first bytes, or `None` for anything else.

    Decided by content, never by a file name: an SVG, or an HTML page named `face.png`,
    is not a picture here, because a browser opening it runs what is inside.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def avatar_url(name: str, record: AvatarRecord | None) -> str | None:
    """The address a head shows the picture from, changing whenever the picture does."""
    if record is None:
        return None
    return f"/api/personas/{name}/avatar?v={record.version}"


def _record(path: Path, mime: str) -> AvatarRecord:
    stat = path.stat()
    return AvatarRecord(path=path, mime=mime, version=f"{stat.st_mtime_ns}-{stat.st_size}")


class PersonaAvatarStore:
    """Finds, sets and resets clones' pictures for one persona registry.

    A picture path is composed from a name only after `source_of(name)` confirms the name
    is a clone loaded from a file, so `../` in a request names nothing.
    """

    def __init__(self, registry: PersonaRegistry) -> None:
        self._registry = registry

    def _custom_dir(self) -> Path | None:
        return self._registry.writable_dir()

    def _custom_files(self, name: str, *, stem: str) -> list[Path]:
        folder = self._custom_dir()
        if folder is None:
            return []
        candidates = (folder / f"{stem}{suffix}" for suffix, _ in AVATAR_FORMATS)
        return [candidate for candidate in candidates if candidate.is_file()]

    def find(self, name: str) -> AvatarRecord | None:
        """The picture in force for `name`, or `None` for a clone with none.

        Looked for in this order, first hit wins: the one chosen for it in the workspace
        personas directory; the one beside the file it was loaded from; the one beside the
        package's own definition of that name. The last is what keeps a built-in's shipped
        picture after its definition is edited, which moves the definition to the workspace.
        """
        source = self._registry.source_of(name)
        if source is None:
            return None
        folders: list[tuple[Path, str]] = []
        custom = self._custom_dir()
        if custom is not None:
            folders.append((custom, name))
        folders.append((source.parent, source.stem))
        if self._registry.has_builtin(name):
            folders.append((BUILTIN_PERSONAS_DIR, name))
        for folder, stem in folders:
            for suffix, mime in AVATAR_FORMATS:
                candidate = folder / f"{stem}{suffix}"
                if candidate.is_file():
                    return _record(candidate, mime)
        return None

    def _require_settable(self, name: str) -> Path:
        if self._registry.source_of(name) is None:  # only a loaded clone's name makes a path
            if self._registry.get_persona(name) is None:
                raise AvatarPersonaNotFound(
                    f"There is no clone named '{name}' here, so no picture was set. "
                    "Check the name in the clone list.",
                    reason_code="no_clone",
                )
            raise AvatarRefused(
                f"'{name}' has no saved definition, so it cannot keep a picture. "
                "Save it from its settings first.",
                reason_code="not_saved",
            )
        folder = self._custom_dir()
        if folder is None:
            raise AvatarRefused(
                "This head has no workspace to keep pictures in, so no picture was set. "
                "Open a workspace and try again.",
                reason_code="no_workspace",
            )
        return folder

    def set(self, name: str, data: bytes) -> AvatarRecord:
        """Make `data` the picture for `name`, keeping the one it replaces as `<name>.prev.<ext>`.

        Raises:
            AvatarPersonaNotFound: `name` is not a clone loaded here.
            AvatarRefused: the picture is too large or not a PNG, JPEG, WebP or GIF image,
                or the clone has nowhere to keep it.
        """
        folder = self._require_settable(name)
        if len(data) > MAX_AVATAR_BYTES:
            raise AvatarRefused(
                f"That picture is {len(data) / (1024 * 1024):.1f} MB; the most a clone's "
                f"picture can be is {MAX_AVATAR_BYTES // (1024 * 1024)} MB. Choose a smaller one.",
                reason_code="too_large",
            )
        mime = sniff_image_format(data)
        if mime is None:
            raise AvatarRefused(
                f"That file is not {_ACCEPTED}, so it was not set as the picture. "
                "Choose a picture in one of those formats.",
                reason_code="not_an_image",
            )
        folder.mkdir(parents=True, exist_ok=True)
        current = self._custom_files(name, stem=name)
        if current:
            self._keep_previous(name, current[0])
        target = folder / f"{name}{_SUFFIX_FOR_MIME[mime]}"
        replace_file(target, data)
        for other in current:
            if other != target:
                other.unlink(missing_ok=True)
        return _record(target, mime)

    def set_from_path(self, name: str, path: Path) -> AvatarRecord:
        """`set` with the bytes of `path`, a file the caller has already resolved safely."""
        if not path.is_file():
            raise AvatarRefused(
                f"There is no picture at '{path.name}', so none was set. Check the file's name.",
                reason_code="no_file",
            )
        if path.stat().st_size > MAX_AVATAR_BYTES:
            raise AvatarRefused(
                f"That picture is larger than {MAX_AVATAR_BYTES // (1024 * 1024)} MB, the most "
                "a clone's picture can be. Choose a smaller one.",
                reason_code="too_large",
            )
        return self.set(name, path.read_bytes())

    def reset(self, name: str) -> Path | None:
        """Put the chosen picture aside as `<name>.prev.<ext>`; return where it went.

        The clone then shows its shipped picture, or the head's default. `None` when there
        was no chosen picture to put aside.
        """
        self._require_settable(name)
        current = self._custom_files(name, stem=name)
        if not current:
            return None
        kept = self._keep_previous(name, current[0])
        for other in current:
            other.unlink(missing_ok=True)
        return kept

    def chosen(self, name: str) -> Path | None:
        """The picture chosen for `name` in the workspace, not the shipped one, if any."""
        if self._registry.source_of(name) is None:
            return None
        found = self._custom_files(name, stem=name)
        return found[0] if found else None

    def previous(self, name: str) -> Path | None:
        """The picture most recently replaced or reset for `name`, if one is kept."""
        if self._registry.source_of(name) is None:
            return None
        found = self._custom_files(name, stem=f"{name}.prev")
        return found[0] if found else None

    def _keep_previous(self, name: str, current: Path) -> Path:
        """Copy `current` to `<name>.prev.<ext>`, dropping any older previous picture."""
        kept = current.with_name(f"{name}.prev{current.suffix}")
        data = current.read_bytes()
        for older in self._custom_files(name, stem=f"{name}.prev"):
            if older != kept:
                older.unlink(missing_ok=True)
        replace_file(kept, data)
        return kept
