"""A leased folder kind: a folder of items the Files screen lists one per folder (#2205).

Each item is a folder under the kind's `root` in the workspace, written by at most one
conversation at a time: the one holding the item's writing lease. The Files screen lists
an item whole, refuses to archive or delete one a live conversation is writing unless told
to stop it, and opens one into a conversation (`artifacts.library.ArtifactLibrary`).

The core names only this protocol. An extension supplies the kind
(`extensions.Extension.leased_folders`); the story library is the first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from uclone_x.errors import PlainRefusalError

__all__ = [
    "FolderItem",
    "FolderLease",
    "FolderItemUnreadableError",
    "LeasedFolderKind",
    "UnknownFolderItemError",
]


@dataclass(frozen=True)
class FolderItem:
    """One item as the Files screen shows it."""

    #: Its title.
    title: str
    #: The conversation holding its writing lease, when one does.
    holder: str | None


class FolderItemUnreadableError(PlainRefusalError):
    """The item's own record does not load; the message is the plain reason."""


class UnknownFolderItemError(PlainRefusalError):
    """There is no item by that id."""


class LeasedFolderKind(Protocol):
    """The operations the Files screen needs on one kind of leased folder."""

    @property
    def root(self) -> str:
        """The folder under the workspace (and under `.archive/`) that holds the items."""
        ...

    def is_item(self, folder: Path) -> bool:
        """Whether `folder`, a directory directly under `root`, is an item (links are not)."""
        ...

    def load(self, base: Path, item_id: str) -> FolderItem:
        """Read item `item_id` under `<base>/<root>`; `FolderItemUnreadableError` if it won't."""
        ...

    def open(self, workspace: Path, item_id: str, room_id: str) -> tuple[FolderItem, bool]:
        """Open the item in conversation `room_id`: the item, and whether it may write it.

        Raises `UnknownFolderItemError` when there is no such item.
        """
        ...

    def take_over(self, workspace: Path, item_id: str, room_id: str) -> FolderItem:
        """Give `room_id` the item's lease, from a holder that no longer exists."""
        ...

    def release(self, workspace: Path, item_id: str, holder: str) -> bool:
        """Give up the lease if `holder` holds it; return whether it did."""
        ...


@dataclass(frozen=True)
class FolderLease:
    """A kind's leases in one workspace, in the shape a room releases one with.

    `room.protocols.StoryLeaseProtocol`: a deleted conversation gives back its open item's
    lease through this (`room.service.RoomService`).
    """

    kind: LeasedFolderKind
    workspace: Path

    def release(self, story_id: str, conversation_id: str) -> bool:
        """Give up the lease on `story_id` if `conversation_id` holds it."""
        return self.kind.release(self.workspace, story_id, conversation_id)
