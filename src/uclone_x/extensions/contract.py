"""What an extension may add to the core, as data the core reads (#2205).

An extension is one `Extension` value. Every field is optional, and each says what the
core does with it; the design for #2205 states the contract in prose. The fields
that would load an extension's own modules (tools, hooks, folder kinds, routes) are
zero-argument factories, so finding an extension costs one small import and its tools are
built only when a registry is.

This module names no extension and imports nothing at run time beyond the error root:
the types it mentions are the core's own protocols, read only by the type checker.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from uclone_x.errors import UCloneXError

if TYPE_CHECKING:
    from uclone_x.agent.protocols import TurnLifecycleHookProtocol
    from uclone_x.artifacts.kinds import LeasedFolderKind
    from uclone_x.tools.builtin.a2a import CharacterLookup
    from uclone_x.tools.protocols import ToolProtocol

__all__ = ["Extension", "ExtensionError", "ProtectedRoot"]

#: An extension's name: lowercase words joined by `_` or `-`, as a package name is.
_NAME = re.compile(r"[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*")

#: The placeholder a `ProtectedRoot.refusal` puts the refused path in.
PATH_PLACEHOLDER = "{path}"


class ExtensionError(UCloneXError):
    """An extension could not be loaded, or what it adds conflicts with the core or another.

    Raised, never logged and skipped: a registry missing an extension's tools would look
    like a working one (P6). The message names the extension and where it came from.
    """


def _nothing() -> tuple[()]:
    return ()


@dataclass(frozen=True)
class ProtectedRoot:
    """A workspace folder that general tools may read but never write (#1583, #1589).

    `dirname` is the folder's name directly under the workspace. The file tools refuse a
    write there (`tools.base.in_protected_root`), and a shell command or local MCP server
    runs in a jail that cannot write it (`sandbox.story_jail`). `refusal` is the sentence
    the model is given, with `{path}` where the path it asked for goes.
    """

    dirname: str
    refusal: str

    def __post_init__(self) -> None:
        if not self.dirname or "/" in self.dirname or "\\" in self.dirname:
            raise ExtensionError(
                f"A protected folder must be one folder name, not {self.dirname!r}."
            )
        if self.dirname.startswith(".") or self.dirname != self.dirname.casefold():
            raise ExtensionError(
                f"A protected folder name must be lowercase and visible, not {self.dirname!r}."
            )
        if PATH_PLACEHOLDER not in self.refusal:
            raise ExtensionError(
                f"The refusal for the protected folder {self.dirname!r} must say which path "
                f"was refused, with {PATH_PLACEHOLDER}."
            )

    def refusal_for(self, path: object) -> str:
        """The refusal sentence for a write to `path`."""
        return self.refusal.replace(PATH_PLACEHOLDER, str(path))


@dataclass(frozen=True)
class Extension:
    """What one extension adds to the core. Found by `extensions.registry`.

    * `tools` -- built with every default tool registry, after the core's own tools.
    * `replaces_tools` -- names of core tools one of `tools` takes the place of, kept at
      the core tool's position. A tool whose name a core tool or another extension already
      has, and that is not named here, is an error.
    * `lifecycle_hooks` -- turn lifecycle hooks every clone is composed with
      (`agent.clone_builder.with_app_lifecycle_hooks`), ahead of the core's own.
    * `protected_roots` -- workspace folders general tools may not write.
    * `leased_folders` -- folder kinds the Files screen lists as one item each, written by
      the one conversation holding the item's lease (`artifacts.kinds.LeasedFolderKind`).
      The conversation's open item is the one a room keeps; at most one kind is allowed
      across all extensions, because a room keeps one.
    * `a2a_characters` -- the lookup `a2a_call` uses to hand a peer the characters a task
      names (`tools.builtin.a2a.CharacterLookup`).
    * `routes` -- mounts the extension's HTTP routes on the web head. Called once, with the
      head's route context (`ui.extension_routes.ExtensionRouteContext`).
    """

    name: str
    tools: Callable[[], Sequence[ToolProtocol]] = _nothing
    replaces_tools: tuple[str, ...] = ()
    lifecycle_hooks: Callable[[], Sequence[TurnLifecycleHookProtocol]] = _nothing
    protected_roots: tuple[ProtectedRoot, ...] = ()
    leased_folders: Callable[[], Sequence[LeasedFolderKind]] = _nothing
    a2a_characters: Callable[[], CharacterLookup] | None = None
    routes: Callable[[Any], None] | None = None
    #: Where the extension was found; set by discovery, for error messages.
    source: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None:
            raise ExtensionError(
                f"An extension's name must be lowercase words such as 'story', not {self.name!r}."
            )
