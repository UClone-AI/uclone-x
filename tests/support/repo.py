"""One way to locate the repository, and one way to read a file out of it.

Tests that read repository files — governance documents, hooks, manifests — need to resolve
them the same way every time (R12). Two habits had grown side by side: a bare relative
`Path("docs/...")`, which resolves against whatever directory the runner was invoked from,
and `Path(__file__).resolve().parents[N]`, which silently means a different directory as
soon as the file moves.

Both are fixed coordinates in the sense R10 rules out: the result depends on something other
than the repository itself. `REPO_ROOT` is anchored to this module's own location, so a test
answers the same way from any worktree and any working directory.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def repo_path(relative: str) -> Path:
    """Return an absolute path to `relative`, interpreted from the repository root."""
    return REPO_ROOT / relative


def read_repo_text(relative: str) -> str:
    """Read a repository file as UTF-8, failing with the resolved path if it is missing.

    Naming the resolved path matters: a check over a document that has been renamed
    otherwise fails with a bare `FileNotFoundError` naming a path the reader must
    reconstruct by hand.
    """
    path = repo_path(relative)
    if not path.is_file():
        raise FileNotFoundError(f"repository file not found: {path}")
    return path.read_text(encoding="utf-8")
