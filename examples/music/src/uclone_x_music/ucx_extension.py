"""The music extension: what songs add to the core, declared in one place.

The core never imports `uclone_x_music`. It finds `EXTENSION` through the entry point this
example's `pyproject.toml` declares in the group `uclone_x.extensions`:

* **Tools** -- `music_library`, `music_start`, `music_score`, `music_orchestra`,
  `music_track`, `music_context`, `music_proposal`, and `music_audit`.
* **Turn lifecycle hooks** -- `MusicLifecycleHook`, which moves the song a turn has open
  between its steps, and `MusicTurnHook`, which provides turn-level score context and
  selection parsing.
* **A protected folder** -- `songs/`, which general tools may read and never write.
* **A leased folder kind** -- the song library, which the Files screen lists one song per
  folder and opens into a conversation.
* **Routes** -- the interactive `MusicView` endpoints.
* **Personas** -- Composer (`personas/composer.yaml`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from uclone_x.artifacts.kinds import (
    FolderItem,
    FolderItemUnreadableError,
    UnknownFolderItemError,
)
from uclone_x.extensions import Extension, ProtectedRoot
from uclone_x_music.library import (
    SONG_FILE,
    SONGS_DIRNAME,
    MusicError,
    SongLibrary,
    SongRecord,
    UnknownSongError,
)

if TYPE_CHECKING:
    from uclone_x.agent.protocols import TurnLifecycleHookProtocol
    from uclone_x.tools.protocols import ToolProtocol

__all__ = ["EXTENSION", "PERSONAS_DIR", "SongFolderKind"]

PERSONAS_DIR = Path(__file__).resolve().parent / "personas"

_SONG_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

_WRITE_REFUSAL = (
    "{path} is in the song library, so it was not written. Songs are changed with the "
    "music tools (music_score, music_orchestra, music_track, music_proposal), which check "
    "which conversation is writing the song and ask a person before applying a proposal. "
    "Reading the file is still allowed."
)


def _item(record: SongRecord) -> FolderItem:
    return FolderItem(
        title=record.title,
        holder=record.lease.holder if record.lease is not None else None,
    )


class SongFolderKind:
    """The song library as the Files screen's leased folder kind (`LeasedFolderKind`)."""

    @property
    def root(self) -> str:
        """`songs`, under the workspace and under `.archive/`."""
        return SONGS_DIRNAME

    def is_item(self, folder: Path) -> bool:
        """A song folder: a song id for a name, not a link, with its `song.yaml`."""
        return (
            _SONG_ID.fullmatch(folder.name) is not None
            and not folder.is_symlink()
            and (folder / SONG_FILE).is_file()
        )

    def load(self, base: Path, item_id: str) -> FolderItem:
        """The song `item_id` under `<base>/songs`."""
        try:
            return _item(SongLibrary(base).load(item_id))
        except MusicError as exc:
            raise FolderItemUnreadableError(str(exc)) from exc

    def open(self, workspace: Path, item_id: str, room_id: str) -> tuple[FolderItem, bool]:
        """Open the song in `room_id`: writable when its lease is free or already the room's."""
        try:
            opening = SongLibrary(workspace).open(item_id, room_id)
        except UnknownSongError as exc:
            raise UnknownFolderItemError(str(exc)) from exc
        return _item(opening.song), opening.writable

    def take_over(self, workspace: Path, item_id: str, room_id: str) -> FolderItem:
        """Give `room_id` the lease of a song whose writer no longer exists."""
        record, _ = SongLibrary(workspace).take_over(item_id, room_id)
        return _item(record)

    def release(self, workspace: Path, item_id: str, holder: str) -> bool:
        """Give up the song's lease if `holder` holds it."""
        return SongLibrary(workspace).release(item_id, holder)


def _tools() -> Sequence[ToolProtocol]:
    from uclone_x_music.start import MusicStartTool
    from uclone_x_music.tool import MusicLibraryTool
    from uclone_x_music.tools import (
        MusicContextTool,
        MusicOrchestraTool,
        MusicProposalTool,
        MusicScoreTool,
        MusicTheoryAuditTool,
        MusicTrackTool,
    )

    return (
        MusicLibraryTool(),
        MusicStartTool(),
        MusicScoreTool(),
        MusicOrchestraTool(),
        MusicTrackTool(),
        MusicContextTool(),
        MusicProposalTool(),
        MusicTheoryAuditTool(),
    )


def _lifecycle_hooks() -> Sequence[TurnLifecycleHookProtocol]:
    from uclone_x_music import MusicLifecycleHook
    from uclone_x_music.section_turn import MusicTurnHook

    return (MusicLifecycleHook(), MusicTurnHook())


def _routes(context: Any) -> None:
    from uclone_x_music.routes import register_music_routes

    register_music_routes(context)


def _leased_folders() -> tuple[SongFolderKind, ...]:
    from uclone_x.extensions import loaded_extensions

    if any(ext.name != "music" and bool(ext.leased_folders()) for ext in loaded_extensions()):
        return ()
    return (SongFolderKind(),)


EXTENSION = Extension(
    name="music",
    tools=_tools,
    lifecycle_hooks=_lifecycle_hooks,
    protected_roots=(ProtectedRoot(dirname=SONGS_DIRNAME, refusal=_WRITE_REFUSAL),),
    leased_folders=_leased_folders,
    routes=_routes,
    personas=(PERSONAS_DIR,),
)
