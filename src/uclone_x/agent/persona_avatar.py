"""Where a clone's picture is, and how it is changed.

A clone's picture is `avatar.<ext>` in its own directory under the agents root, for one
the user or the clone chose, with the one it replaced kept as `avatar.prev.<ext>`
(clone-data-scopes §3.2). A builtin's shipped picture stays beside its package definition
and is found by the clone's template. `PersonaAvatarStore` is the one place that finds it
and the one place that writes it. The head's routes and the `set_avatar` tool both go
through it, and general writing tools cannot reach the agents root at all
(`BaseTool.resolve_write_path`).
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.core.agent_home import peer_handles
from uclone_x.errors import PlainRefusalError
from uclone_x.tools.base import replace_file

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX (Windows): changes go unserialised
    fcntl = None

__all__ = [
    "AVATAR_FORMATS",
    "MAX_AVATAR_BYTES",
    "AvatarChange",
    "AvatarPersonaNotFound",
    "AvatarRecord",
    "AvatarRefused",
    "AvatarStaleChange",
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

#: The tool a clone draws with, and the one it asks a peer with (`tools/builtin/a2a.py`).
#: Names only: this module is below the tools that define them.
DRAWING_TOOL = "generate_image"
PEER_CALL_TOOL = "a2a_call"
#: The builtin that draws for clones that cannot.
ARTIST = "artist"

#: What a clone with no picture at the path it gave can do next, by what it holds (#2290).
#: None names a tool: the reader may be the person, who sees the refusal too (#2165).
DRAW_IT_FIRST = "Draw the picture first, then use the path the drawing returned."
ASK_ARTIST_FIRST = "Ask Artist to draw one first, then use the path Artist's answer gives."
CANNOT_DRAW = "This clone cannot draw pictures; the person can ask Artist for one."
#: For a caller that is no clone -- the head, setting a picture a person chose.
_FROM_A_PATH = "Use the path of a picture that is in the workspace."


def missing_picture_advice(agent: object) -> str:
    """What `agent`, the clone that called, can do to get a picture, as one sentence.

    Read from the running agent itself -- the tools it holds (`held_tools`, which its
    `allowed_tools` range already narrows) and whether it may reach Artist now (#2290) --
    never from the text of the request.
    """
    held = _held_tool_names(agent)
    if DRAWING_TOOL in held:
        return DRAW_IT_FIRST
    if _may_ask_artist(agent, held):
        return ASK_ARTIST_FIRST
    return CANNOT_DRAW


def _held_tool_names(agent: object) -> set[str]:
    """The names of the tools `agent` holds; none for a caller that is no agent."""
    held_tools: object = getattr(agent, "held_tools", None)
    if not callable(held_tools):
        return set()
    tools: object = held_tools()
    if not isinstance(tools, list):
        return set()
    return {str(getattr(tool, "name", "")) for tool in cast(list[object], tools)}


def _may_ask_artist(agent: object, held: set[str]) -> bool:
    """Whether `a2a_call` from `agent` would reach Artist: it holds the tool, Artist a peer.

    `held_tools` already leaves `a2a_call` out for an agent with no transport -- one a peer
    call built, which may not call another (`BaseAgent._may_call_peers`) -- so the seat,
    not only the persona file, decides.
    """
    if PEER_CALL_TOOL not in held:
        return False
    persona = getattr(agent, "persona_definition", None)
    peers: object = getattr(persona, "a2a_peers", ())
    if not isinstance(peers, tuple):
        return False
    return ARTIST in peer_handles([str(peer) for peer in cast(tuple[object, ...], peers)])


class AvatarRefused(PlainRefusalError):
    """A picture was not set or reset, for a reason written for a person.

    `reason_code` names which reason, so the head can say what to do without reading the
    sentence: `no_clone`, `clone_unreadable`, `not_saved`, `too_large`, `not_an_image`, `no_file`,
    `outside_workspace`, `no_source`, `stale_change`. (`no_workspace` went with the
    workspace pictures folder, 2026-09-27: a clone keeps its picture in its own directory.)
    """


class AvatarPersonaNotFound(AvatarRefused):
    """The name is not a clone this head has loaded from a file."""


class AvatarStaleChange(AvatarRefused):
    """An undo named a change that is no longer the clone's latest, so nothing was done.

    Another tab, another device, or the clone itself changed the picture after the change
    being undone; putting back what that one replaced would now undo the wrong picture.
    """

    def __init__(self) -> None:
        super().__init__(
            "The picture was changed again after that change, so it was not undone. "
            "Choose the picture you want from the clone's profile instead.",
            reason_code="stale_change",
        )


@dataclass(frozen=True)
class AvatarRecord:
    """One clone's picture as found on disk."""

    path: Path
    mime: str
    version: str


@dataclass(frozen=True)
class AvatarChange:
    """One change to a clone's picture, as the store made it.

    `change_id` names this change among every change to this clone's picture, counted on
    disk, so it survives a restart and two changes to the same picture (A, B, A) still get
    different ids. An undo passes it back as `undo_of`. `previous` is the kept picture
    that undoing it would put back, or `None` when the clone had no chosen picture before,
    so undoing is a reset.
    """

    change_id: int
    previous: Path | None


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


def avatar_url(clone: str, record: AvatarRecord | None) -> str | None:
    """The address a head shows the picture from, changing whenever the picture does.

    `clone` is the clone's id where it has one, so the address survives a handle rename;
    the route reads a handle as well.
    """
    if record is None:
        return None
    return f"/api/clones/{clone}/avatar?v={record.version}"


def _record(path: Path, mime: str) -> AvatarRecord:
    stat = path.stat()
    return AvatarRecord(path=path, mime=mime, version=f"{stat.st_mtime_ns}-{stat.st_size}")


class PersonaAvatarStore:
    """Finds, sets and resets clones' pictures for one persona registry.

    A picture path is composed only inside the directory the registry names for a clone,
    and never from the name itself, so `../` in a request names nothing.
    """

    #: The chosen picture's stem in a clone directory, and the replaced one's.
    _CHOSEN = "avatar"
    _PREVIOUS = "avatar.prev"

    def __init__(self, registry: PersonaRegistry) -> None:
        self._registry = registry

    def _clone_dir(self, name: str) -> Path | None:
        record = self._registry.clone_record(name)
        return record.path if record is not None else None

    def _files(self, name: str, *, stem: str) -> list[Path]:
        folder = self._clone_dir(name)
        if folder is None:
            return []
        candidates = (folder / f"{stem}{suffix}" for suffix, _ in AVATAR_FORMATS)
        return [candidate for candidate in candidates if candidate.is_file()]

    def find(self, name: str) -> AvatarRecord | None:
        """The picture in force for `name`, or `None` for a clone with none.

        Looked for in this order, first hit wins: the one chosen for it in its clone
        directory; for a clone installed from the package, or one whose handle a builtin
        carries, the shipped picture of that builtin; for a persona read from an extra
        directory, the one beside its file.
        """
        source = self._registry.source_of(name)
        if source is None:
            return None
        folders: list[tuple[Path, str]] = []
        record = self._registry.clone_record(name)
        if record is not None:
            folders.append((record.path, self._CHOSEN))
            # An imported workspace edit of a builtin has no template but keeps the
            # shipped picture of its handle, as it did before clones were stored. The
            # builtin may ship with the package or with an extension (#2208).
            template = record.template or (name if self._registry.has_builtin(name) else None)
            package = self._registry.shipped_dir(template) if template is not None else None
            if template is not None and package is not None:
                folders.append((package, template))
        else:
            folders.append((source.parent, source.stem))
        for folder, stem in folders:
            for suffix, mime in AVATAR_FORMATS:
                candidate = folder / f"{stem}{suffix}"
                if candidate.is_file():
                    return _record(candidate, mime)
        return None

    def _require_settable(self, name: str) -> Path:
        folder = self._clone_dir(name)
        if folder is not None:
            return folder
        unreadable = self._registry.unreadable_clone(name)
        if unreadable is not None:
            # The clone is there; it is its settings file that cannot be read (#2241).
            raise AvatarRefused(
                f"The clone '{unreadable.handle}' cannot keep a picture until it can be "
                f"loaded, so no picture was set. {unreadable.reason}",
                reason_code="clone_unreadable",
            )
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

    def latest_change(self, name: str) -> int:
        """The id of the latest change to `name`'s picture; `0` before any, or for no clone."""
        folder = self._clone_dir(name)
        return 0 if folder is None else _read_change(folder)

    def set(self, name: str, data: bytes, *, undo_of: int | None = None) -> AvatarChange:
        """Make `data` the picture for `name`, keeping the one it replaces as `avatar.prev.<ext>`.

        With `undo_of`, this is the undo of that change, and it is made only while that
        change is still the latest; checked and made under one lock.

        Raises:
            AvatarPersonaNotFound: `name` is not a clone loaded here.
            AvatarStaleChange: `undo_of` is not the latest change; nothing was written.
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
        with _changing(folder, undo_of) as change_id:
            current = self._files(name, stem=self._CHOSEN)
            replaced = self._keep_previous(name, current[0]) if current else None
            target = folder / f"{self._CHOSEN}{_SUFFIX_FOR_MIME[mime]}"
            replace_file(target, data)
            for other in current:
                if other != target:
                    other.unlink(missing_ok=True)
            return AvatarChange(change_id=change_id, previous=replaced)

    def set_from_path(
        self, name: str, path: Path, *, undo_of: int | None = None, advice: str = _FROM_A_PATH
    ) -> AvatarChange:
        """`set` with the bytes of `path`, a file the caller has already resolved safely.

        `advice` ends the refusal when there is no file at `path`: what the caller can do
        next. A clone passes `missing_picture_advice(agent)` (#2290).
        """
        if undo_of is not None and undo_of != self.latest_change(name):
            # Said before the file checks, so a stale undo is not reported as a missing file;
            # `set` checks again under the lock, which is what makes the refusal hold.
            raise AvatarStaleChange()
        if not path.is_file():
            raise AvatarRefused(
                f"There is no picture at '{path.name}', so none was set. {advice}",
                reason_code="no_file",
            )
        if path.stat().st_size > MAX_AVATAR_BYTES:
            raise AvatarRefused(
                f"That picture is larger than {MAX_AVATAR_BYTES // (1024 * 1024)} MB, the most "
                "a clone's picture can be. Choose a smaller one.",
                reason_code="too_large",
            )
        return self.set(name, path.read_bytes(), undo_of=undo_of)

    def reset(self, name: str, *, undo_of: int | None = None) -> AvatarChange:
        """Put the chosen picture aside as `avatar.prev.<ext>`; `previous` says where it went.

        The clone then shows its shipped picture, or the head's default. `previous` is
        `None` when there was no chosen picture to put aside; the call still counts as a
        change, so an Undo offered before it no longer matches. `undo_of` is as for `set`.
        """
        folder = self._require_settable(name)
        with _changing(folder, undo_of) as change_id:
            current = self._files(name, stem=self._CHOSEN)
            if not current:
                return AvatarChange(change_id=change_id, previous=None)
            kept = self._keep_previous(name, current[0])
            for other in current:
                other.unlink(missing_ok=True)
            return AvatarChange(change_id=change_id, previous=kept)

    def chosen(self, name: str) -> Path | None:
        """The picture chosen for `name` in its clone directory, not the shipped one, if any."""
        found = self._files(name, stem=self._CHOSEN)
        return found[0] if found else None

    def previous(self, name: str) -> Path | None:
        """The picture most recently replaced or reset for `name`, if one is kept."""
        found = self._files(name, stem=self._PREVIOUS)
        return found[0] if found else None

    def named_previous(self, name: str, raw: str) -> Path | None:
        """The kept picture of `name` when `raw` names it, as `previous_path` answers it.

        The kept picture lives in the clone's directory, outside the workspace, so the
        `previous_path` a change answers with is accepted back here by what it names; any
        other path is left to the caller's workspace rule.
        """
        kept = self.previous(name)
        if kept is None:
            return None
        try:
            same = Path(raw).expanduser().resolve() == kept.resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        return kept if same else None

    def _keep_previous(self, name: str, current: Path) -> Path:
        """Copy `current` to `avatar.prev.<ext>`, dropping any older previous picture."""
        kept = current.with_name(f"{self._PREVIOUS}{current.suffix}")
        data = current.read_bytes()
        for older in self._files(name, stem=self._PREVIOUS):
            if older != kept:
                older.unlink(missing_ok=True)
        replace_file(kept, data)
        return kept


def _change_file(folder: Path) -> Path:
    """The change counter of the clone whose directory is `folder`, beside its picture."""
    return folder / ".avatar-change"


def _read_change(folder: Path) -> int:
    """The counted id of the clone's latest change; `0` when none was counted or it is unreadable."""
    try:
        text = _change_file(folder).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return 0
    return int(text) if text.isascii() and text.isdigit() else 0


@contextmanager
def _changing(folder: Path, undo_of: int | None) -> Generator[int]:
    """Hold the picture lock of the clone in `folder` for one change; yield that change's id.

    The id is counted on disk, beside the picture, and written before the change is made.
    A change that fails part-way then still takes its id, which only makes older Undos
    stale; counting afterwards would let a crash between the two leave an older Undo
    matching a picture it no longer describes. With `undo_of`, the change goes ahead only
    while `undo_of` is still the latest id, checked under the same lock that covers the
    write, so a change from another tab or the clone itself cannot land in between.

    The lock is a sidecar file locked with `flock`, across processes as well as threads;
    where `fcntl` does not exist, changes are not serialised.
    """
    folder.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    if fcntl is not None:
        try:
            fd = os.open(folder / ".avatar-lock", os.O_RDONLY | os.O_CREAT, 0o600)
        except OSError:
            fd = None
    try:
        if fd is not None and fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        latest = _read_change(folder)
        if undo_of is not None and (undo_of < 1 or undo_of != latest):
            raise AvatarStaleChange()
        change_id = latest + 1
        replace_file(_change_file(folder), f"{change_id}\n".encode("ascii"))
        yield change_id
    finally:
        if fd is not None:
            os.close(fd)  # closing the descriptor releases the lock
