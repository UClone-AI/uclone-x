"""The workspace a host may provide, as a contract rather than a path.

`ToolContext.workspace_root` is a required `Path` today, so every tool receives a
filesystem location whether or not it touches one, and a host with no user filesystem has
no honest value to supply. That is what drives the kernel to invent one
(the core/shell architecture note C3, C5).

A workspace is therefore optional and, when present, is an object that can refuse: it
resolves a relative path *and* enforces containment, so the containment rule lives with
the root rather than being re-derived by each caller.

Design reference: the core/shell architecture note §6.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

__all__ = ["WorkspaceProtocol"]


class WorkspaceProtocol(Protocol):
    """A rooted, containment-enforcing filesystem location supplied by the host."""

    @property
    def root(self) -> Path:
        """The absolute root. Every path this workspace resolves is inside it."""
        ...

    def resolve(self, relative: Path | str) -> Path:
        """Resolve `relative` against `root`, refusing anything that escapes it.

        Raises `PathTraversalError` rather than clamping (§6.5). A path that escapes the
        root is a caller error, and silently returning the clamped result would answer a
        question the caller did not ask — the substitution P6 forbids. Naming the exception
        matters here: an implementer with only "raises" to go on picks their own, and two
        backends that refuse differently are two backends a caller cannot handle uniformly.
        """
        ...
