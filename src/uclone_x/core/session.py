"""Session identity and location: the pure part of the session model (#1734).

The session *family root* (`~/.uclone/sessions`, or `UCLONE_SESSION_DIR`), the
subdirectory names under it, the name rule for a session id, and the containment check
that turns an id into a path under a storage directory (`resolve_session_path`). Nothing
here reads or writes a session record; that is `SessionStore` in `agent/session.py`, which re-exports
every name below under its old spelling.

Lowered out of `agent/session.py` so that `core/` and `llm/` can locate the session root
and validate an id without importing the single-agent package: `llm/connectors/
saved_choice.py` and `llm/usage/store.py` need the root for `settings.json` and
`usage.sqlite3`, and `core/session_diagnostics.py` needs the id rule.

**Patch this module, not the re-export.** `default_session_root` reads
`DEFAULT_SESSION_STORAGE_DIR` from *this* module's globals, so a test that redirects the
default must patch `uclone_x.core.session.DEFAULT_SESSION_STORAGE_DIR`; patching the name
re-exported from `agent/session.py` changes nothing. `UCLONE_SESSION_DIR` is read
at call time, so an environment override needs no patch at all.
"""

from __future__ import annotations

import os
from pathlib import Path

from uclone_x.errors import PathTraversalError

__all__ = [
    "CORE_RECORD_SUBDIR",
    "DEFAULT_SESSION_STORAGE_DIR",
    "SESSION_STORAGE_DIR_ENV_VAR",
    "UI_TRANSCRIPT_SUBDIR",
    "default_session_root",
    "default_session_storage_dir",
    "resolve_session_path",
    "validate_session_id",
]

# Kept identical to the directory `AgentSessionManager` already defaults to, so the
# extraction does not move anybody's existing session files.
DEFAULT_SESSION_STORAGE_DIR = Path.home() / ".uclone" / "sessions"

# Subdirectories under the family root. Both artifacts are namespaced, and the root
# itself is **never written** by either.
#
# This is a correction: an earlier arrangement put the Core record for the UI under
# `core/` while leaving the UI's *transcript* on the root path — which is the path the
# CLI's own Core store owns. So `ucx run` wrote a Core record to
# `<root>/<id>.json` and the UI wrote a UI transcript to the same file. Each destroyed
# the other, and because a record that does not validate reads back as absent, the loss
# was silent by construction: the conversation was destroyed and then reported as "no
# session here". Measured on a real `~/.uclone/sessions`: 8 Core-shaped records, 16
# UI-shaped, one hybrid (a UI envelope wrapping Core `ChatMessage`s), and `core/` empty.
#
# Namespacing both fixes it, and it also fixes the deeper error. The CLI and the UI
# should share **one** Core record for a session — that is what P8's single store means
# — not hold isolated copies. They now both resolve to `<root>/core/<id>.json`. The
# transcript, which is genuinely a different artifact with a different schema, gets its
# own `<root>/ui/`. Nothing reads or writes a record at the root itself.
CORE_RECORD_SUBDIR = "core"
UI_TRANSCRIPT_SUBDIR = "ui"

# Environment override for the default storage root. Exists because `SessionStore()` is
# now constructed by the CLI, which has no `storage_dir` argument to pass down from a
# command line, and a headless run must not be forced to write into the invoking user's
# home directory — tests especially, since a unit test that persists a session under
# `~/.uclone` has escaped its own sandbox.
SESSION_STORAGE_DIR_ENV_VAR = "UCLONE_SESSION_DIR"


def default_session_root() -> Path:
    """Resolve the session **family root**, honouring `UCLONE_SESSION_DIR`.

    Both layers derive from this one resolver, which is the point. `ui.app.
    AgentSessionManager` used to compute its own default inline as
    `Path.home() / ".uclone" / "sessions"`, so it ignored `UCLONE_SESSION_DIR`
    entirely — the variable existed to keep a headless run out of the invoking user's
    home, and the busiest writer to that directory was not consulting it. Two defaults
    for one location is the same shape as two copies of a guard: one of them gets fixed.
    """
    override = os.environ.get(SESSION_STORAGE_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_SESSION_STORAGE_DIR


def default_session_storage_dir() -> Path:
    """Resolve the Core store's default root, honouring `UCLONE_SESSION_DIR`.

    Returns `<family root>/core`, **not** the family root. A `SessionStore()` built with
    no argument is a Core store, and Core records live in the `core/` subdirectory so
    they cannot collide with the UI transcript that used to occupy the root — see
    `CORE_RECORD_SUBDIR`.

    `UCLONE_SESSION_DIR` names the **family root**, so redirecting it moves the Core
    records and the UI transcripts together and keeps the CLI and the UI agreeing on
    which record is which session.
    """
    return default_session_root() / CORE_RECORD_SUBDIR


# Characters that must never appear in a session ID, because it is interpolated into a
# filename. Checked before any path arithmetic: `Path` normalises away a `..` segment
# during `resolve()`, so a containment check alone cannot report *why* an ID was
# rejected, and on a directory that is itself a symlink it can be made to pass.
_FORBIDDEN_ID_SUBSTRINGS = ("..", "/", "\\", "\x00")


def validate_session_id(session_id: str) -> str:
    """Return `session_id` if it is a legal session identifier, else raise (P3).

    Separated from path resolution so that a caller holding no storage directory can
    apply the same rule. `BaseAgent` is that caller: it addresses sessions by ID long
    before anything is persisted, and without this it would accept `""` or
    `"../../../etc/evil"` as a live session name and only discover the problem at the
    first write — by which point the conversation exists and cannot be saved.

    The rule is a *name* rule, not a containment check: containment additionally
    requires the resolved path, which `resolve_session_path` does.

    Raises:
        PathTraversalError: If the ID is empty, or carries a path separator, a `..`
            segment or a NUL.
    """
    # `None` already refused with a `PathTraversalError` via the falsiness check below,
    # while any other non-`str` fell through to `bad in session_id` and raised a raw
    # `TypeError`. Nothing was written either way, so this is error-type hygiene rather
    # than a hole — but a guard that refuses one class of bad input with two different
    # exception types is inconsistent with itself, and a caller catching
    # `PathTraversalError` would not catch the other.
    #
    # Pyright is correct that a *typed* caller cannot reach this: the parameter is `str`
    # and the boundary is checked statically. The guard is for the untyped arrivals the
    # annotation cannot police — a session id lifted out of a deserialized JSON payload,
    # or passed through `**kwargs` — which is exactly where a non-`str` comes from in
    # practice. So the branch is deliberately kept and the diagnostic suppressed here
    # rather than the check being dropped.
    if not isinstance(session_id, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise PathTraversalError(
            f"Session ID must be a string, got {type(session_id).__name__}: {session_id!r}"
        )
    if not session_id:
        raise PathTraversalError(
            "Session ID must be non-empty: an empty ID resolves to the storage directory itself."
        )
    for bad in _FORBIDDEN_ID_SUBSTRINGS:
        if bad in session_id:
            raise PathTraversalError(
                f"Invalid session ID containing path traversal characters: {session_id!r}"
            )
    return session_id


def resolve_session_path(storage_dir: Path, session_id: str, suffix: str = ".json") -> Path:
    """Resolve `session_id` to a file under `storage_dir`, refusing anything that escapes (P3).

    Module-level and public because it is a **security control with more than one
    caller**: `SessionStore` uses it for Core session records and `ui.app.
    AgentSessionManager.get_session_path` uses it for the UI's presentation transcript.
    A duplicated containment guard is one that gets fixed in a single copy, so there is
    one implementation and two callers rather than two implementations.

    The name half is delegated to `validate_session_id` for the same reason, rather than
    copied here: `BaseAgent` needs the name rule without a storage directory, and two
    copies of a rule are two things to fix.

    The containment check is the half that a hostile *string* never reaches — every such
    string is stopped by the name rule above it. What reaches containment is a lexically
    innocent ID whose record path is a planted symlink pointing out of the directory,
    which is the case `test_a_lexically_clean_id_whose_record_symlinks_outside_is_refused`
    pins.

    Raises:
        PathTraversalError: If the ID is illegal by name, or resolves outside
            `storage_dir`. The check *raises* rather than returning `None` or silently
            skipping the operation: a containment guard whose failure mode is "do
            nothing" is fail-open, and the caller cannot tell a refused write from a
            completed one.
    """
    validate_session_id(session_id)
    root = storage_dir.resolve()
    target = (root / f"{session_id}{suffix}").resolve()
    if not target.is_relative_to(root):
        raise PathTraversalError(f"Path traversal violation for session ID: {session_id!r}")
    return target
