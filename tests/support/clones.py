"""Clones a test names, made the way the app makes one (clone-data-scopes §3.4).

There is no lazy home any more: memory, the CLI and the room resolve a name through
`resolve_handle`, and a name no clone carries is refused rather than given a directory. A
test that runs an agent under a name of its own therefore creates that clone first, here,
in the per-test agents root the autouse fixture in `tests/conftest.py` points at.

The clone these make has no persona of its own (its `clone.yaml` holds only its handle),
so it speaks as its host's fallback prompt -- exactly what the same test's name did before
clones were stored, and what the test was written to check.
"""

from __future__ import annotations

from pathlib import Path

from uclone_x.core.agent_home import AgentHome, clone_handles, create_clone


def make_clone(handle: str, *, root: Path | None = None) -> AgentHome:
    """Create the persona-less clone `handle`, or return it when it already exists."""
    existing = clone_handles(root).get(handle)
    if existing:
        return AgentHome.for_handle(handle, root)
    return create_clone(handle, f"handle: {handle}\n", root=root)


def make_clones(*handles: str, root: Path | None = None) -> None:
    """Create each of `handles` that does not exist yet."""
    for handle in handles:
        make_clone(handle, root=root)
