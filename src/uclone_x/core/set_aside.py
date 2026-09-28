"""Keep a record this build cannot read, under a name nothing reads as a live record (#1844).

**Why a record is moved instead of left or replaced.** A durable record written by one build
can fail validation in another: a field added under `extra="forbid"` makes every record the
newer build writes unreadable to an older one. A store that answers "absent" for such a
record and then writes a fresh one over it destroys the only copy -- a whole conversation,
after one rollback. Moving the file aside first keeps its bytes, unchanged, beside where it
was.

**The name is `<name>.unreadable-<UTC time>`**, with `-<n>` added when that name is taken.
The marker goes *after* the whole file name, extension included, so a store that finds its
records by extension (`*.json`, `*.yaml`) never lists a set-aside copy as a record of its
own, and never tries to load it again.

**A copy is never replaced.** `os.rename` replaces an existing destination on POSIX, so a
name picked from a listing is not a claim: two processes that list before either renames
pick the same name, and the second rename replaces the first copy -- the original record,
gone (#1860). So the name is claimed first by creating an empty placeholder with `O_EXCL`,
which exactly one process can do, and the rename then replaces only that placeholder. A
name another process claimed first is skipped for the next counter. `os.link` would claim
it too, but the `unlink(path)` it needs afterwards could delete a record another process
wrote at `path` in between; a rename moves whatever is there into the copy instead.

**Retention: the newest `KEEP_SET_ASIDE` copies of each record (#1860).** Each set-aside
deletes that record's older copies beyond the newest three. Without a bound, two builds that
take turns saving one conversation -- the newer writes a field the older refuses, the older
sets it aside and saves, and so on -- leave a whole copy per round, and every copy also keeps
the conversation's tool results alive (`SessionStore._has_record`). The newest copies are
the ones a build that can read them wants back; a copy older than three later ones is a
state of the conversation that was superseded three times over. A name that does not parse
as `<name>.unreadable-<stamp>[-<n>]` was not written here and is never deleted. Deleting is
best effort: a copy that cannot be removed is logged and left. Two copies are never
deleted: the one just made, whatever its name says -- a clock set back would otherwise sort
it oldest and delete it at once -- and a fresh empty file, which may be another process's
claimed name waiting for its rename.

**Newest by the time the file last changed, not by its name (#1877).** Renaming a file
sets its change time (`st_ctime`) from the one kernel clock, so the order is the order the
copies were made even when two processes disagree about the time they write into the
name. Copies whose change times tie keep the order of their names.

**An empty copy is not a kept copy (#1877).** A crash between claiming the name and the
rename leaves an empty `<name>.unreadable-<stamp>` behind. It holds nothing, so
`kept_copies` leaves it out: a store that spares a record's history while a copy is kept
does not spare it for one of these. An empty copy older than `PLACEHOLDER_MAX_AGE` is
deleted by `expire_set_aside` and by the next set-aside of that record; a younger one may
still be a name another process has claimed, so it is left.

**Copies of a record that is gone expire (#1877).** Retention runs when a record is set
aside again. A record deleted after it was set aside -- a conversation deleted in this
build -- is never set aside again, so its copies would be kept forever, and the store
keeps what they refer to with them. `expire_set_aside` deletes a copy once it is
`EXPIRED_COPY_MAX_AGE` (30 days) old *and* the record it was set aside from no longer
exists. A record that still exists keeps its newest `KEEP_SET_ASIDE` copies however old
they are. 30 days is the author's choice (#1877): a build that could read the copy has to
be run again within that time for it to be restored.

**The same naming keeps a record that was moved elsewhere** (`set_aside`, with the marker
the caller names): the seat knowledge import renames each file it imported to
`<name>.imported-<UTC time>` (clone-knowledge-graph step 6), so it is kept and never read
as a record again. The name is claimed the same way; retention applies only to
`.unreadable-` copies, so an imported file is never deleted here.

**What this does not do: restore.** A later build that can read the copy may be given it
back by renaming it over `<name>` -- by hand today. Nothing here merges the copy with a
record written since, because the two are different conversations.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "EXPIRED_COPY_MAX_AGE",
    "KEEP_SET_ASIDE",
    "PLACEHOLDER_MAX_AGE",
    "SESSION_SET_ASIDE_NOTICE",
    "SET_ASIDE_MARKER",
    "expire_set_aside",
    "kept_copies",
    "set_aside",
    "set_aside_copies",
    "set_aside_unreadable",
]

logger = logging.getLogger(__name__)

#: Said once when a save had to keep a conversation's earlier record aside (#1860): the
#: plain line a room says on the seat's row, for a head with no rows (the CLI, ACP). No
#: path, no cause. Kept here so every head says the same words (#1877).
SESSION_SET_ASIDE_NOTICE = (
    "This version could not open this conversation's earlier saved record. It was kept, "
    "not written over, and this conversation carried on without it."
)

#: What separates a record's own name from the time it was set aside.
SET_ASIDE_MARKER = ".unreadable-"

#: How many set-aside copies of one record are kept; older ones are deleted (#1860).
KEEP_SET_ASIDE = 3

#: Seconds before an empty copy is deleted as a crashed set-aside's placeholder (#1877). A
#: claimed name is renamed onto within one call, so a day-old one was left by a crash.
PLACEHOLDER_MAX_AGE = 24 * 60 * 60.0

#: Seconds a copy of a record that no longer exists is kept (#1877; author's choice).
EXPIRED_COPY_MAX_AGE = 30 * 24 * 60 * 60.0

#: Names tried before giving up; each taken one was claimed by another process since the
#: listing, so running out means something other than a race is answering "taken".
_CLAIM_ATTEMPTS = 1000

#: The part after the marker: the UTC stamp, and the counter added when that name was taken.
_SUFFIX = re.compile(r"(\d{8}T\d{12}Z)(?:-(\d+))?")


def set_aside_unreadable(path: Path, *, now: datetime | None = None) -> Path:
    """Rename `path` to a free `<name>.unreadable-<UTC time>` beside it; return the new path.

    Then deletes this record's copies older than the newest `KEEP_SET_ASIDE`, the new one
    included in the count.

    Raises:
        OSError: The rename failed; the record is still at `path` and no name is left
            claimed. A caller about to write over `path` must not write.
    """
    aside = set_aside(path, SET_ASIDE_MARKER, now=now)
    _delete_older_copies(path, keep=aside)
    return aside


def set_aside(path: Path, marker: str, *, now: datetime | None = None) -> Path:
    """Rename `path` to a free `<name><marker><UTC time>` beside it; return the new path.

    `marker` says why it was moved (`.unreadable-`, `.imported-`). It must start with `.`
    and end with `-`, so the copy's name keeps the record's own whole and says when. The
    name is claimed before the rename, so no existing copy is ever replaced.

    Raises:
        ValueError: `marker` is not of that shape; nothing was renamed.
        OSError: The rename failed; the record is still at `path` and no name is left
            claimed.
    """
    if not (len(marker) > 2 and marker.startswith(".") and marker.endswith("-")):
        raise ValueError(f"a set-aside marker is '.<word>-', not {marker!r}")
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    base = f"{path.name}{marker}{stamp}"
    # `os.rename` replaces an existing destination, so the name must be free. Past the
    # highest counter already used in this instant, not the lowest free one: once retention
    # has deleted the bare name or `-1`, reusing it would name the newest copy as the oldest.
    # Every name that could collide -- the bare one or `-<n>` -- parses, so it is counted.
    n = 1 + _highest_counter(path, marker, stamp)
    for _ in range(_CLAIM_ATTEMPTS):
        aside = path.with_name(base if n == 0 else f"{base}-{n}")
        try:
            os.close(os.open(aside, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        except FileExistsError:
            n += 1  # claimed since the listing, by another process: never replace it
            continue
        break
    else:
        raise FileExistsError(
            f"no free name to keep {path.name} under after {_CLAIM_ATTEMPTS} tries"
        )
    try:
        os.rename(path, aside)
    except OSError:
        aside.unlink(missing_ok=True)  # the empty placeholder, not a copy
        raise
    return aside


def _highest_counter(path: Path, marker: str, stamp: str) -> int:
    """The highest `-<n>` among `path`'s `marker` copies made at `stamp`, the bare name 0.

    -1 when no copy was set aside at `stamp`.
    """
    prefix = f"{path.name}{marker}"
    counters = [
        key[1]
        for entry in set_aside_copies(path, marker=marker)
        if (key := _age_key(prefix, entry.name)) is not None and key[0] == stamp
    ]
    return max(counters, default=-1)


def _age_key(prefix: str, name: str) -> tuple[str, int] | None:
    """`(stamp, n)` for a copy's name, which sorts oldest first; `None` for a foreign name."""
    matched = _SUFFIX.fullmatch(name[len(prefix) :])
    if matched is None:
        return None
    return matched.group(1), int(matched.group(2) or 0)


def _changed_at(entry: Path) -> float:
    """When `entry` last changed: its change time, or its modification time if later.

    A rename sets the change time, so for a copy it is when it was set aside. The later of
    the two is taken so that a file system which reports something else there (Windows
    reports the creation time) still reads a file written since as recent.
    """
    stat = entry.stat()
    return max(stat.st_ctime, stat.st_mtime)


def _delete_older_copies(path: Path, *, keep: Path) -> None:
    """Delete `path`'s set-aside copies beyond the newest `KEEP_SET_ASIDE`; log, never raise.

    Newest by the time each copy last changed (#1877), then by its name. `keep`, the copy
    just made, counts as one of those kept and is never deleted. An empty copy is not
    counted; one older than `PLACEHOLDER_MAX_AGE` is deleted, and a younger one is left, as
    it may be a name another process has claimed.
    """
    prefix = f"{path.name}{SET_ASIDE_MARKER}"
    now = time.time()
    dated: list[tuple[float, tuple[str, int], Path]] = []
    try:
        for entry in set_aside_copies(path):
            key = _age_key(prefix, entry.name)
            if entry == keep or key is None or not entry.is_file():
                continue
            try:
                if entry.stat().st_size == 0:  # not counted: nothing is kept in it
                    _delete_stale_placeholder(entry, now)
                    continue
                dated.append((_changed_at(entry), key, entry))
            except FileNotFoundError:
                continue  # deleted by another process since the listing
    except OSError as exc:
        logger.warning("Could not list the copies kept beside %s (%s); none deleted", path, exc)
        return
    dated.sort(key=lambda item: (item[0], item[1]))
    ours = [entry for _, _, entry in dated]
    for old in ours[: max(0, len(ours) - (KEEP_SET_ASIDE - 1))]:
        try:
            old.unlink()
        except OSError as exc:
            logger.warning("Could not delete the old kept copy %s (%s); it stays", old, exc)
        else:
            logger.info("Deleted %s: %d newer copies of that record are kept", old, KEEP_SET_ASIDE)


def _delete_stale_placeholder(entry: Path, now: float) -> bool:
    """Delete `entry`, an empty copy, if it is older than `PLACEHOLDER_MAX_AGE`; whether it was.

    Never raises: an empty file that cannot be removed holds nothing, and is logged.
    """
    try:
        if now - _changed_at(entry) < PLACEHOLDER_MAX_AGE:
            return False
        entry.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("Could not delete the empty copy %s (%s); it stays", entry, exc)
        return False
    logger.info("Deleted %s: an empty name left by a set-aside that did not finish", entry)
    return True


def kept_copies(path: Path) -> tuple[Path, ...]:
    """`path`'s set-aside copies that hold something, oldest first by name (#1877).

    An empty copy is a name claimed by a set-aside that has not renamed yet, or one that
    crashed before it did: nothing is kept in it. A copy whose size cannot be read is
    counted, so a caller sparing what a copy needs keeps it on a guess rather than delete it.

    Raises:
        OSError: The folder could not be listed.
    """
    kept: list[Path] = []
    for entry in set_aside_copies(path):
        try:
            if entry.stat().st_size == 0:  # a claimed name, or one a crash left
                continue
        except FileNotFoundError:
            continue
        except OSError:
            pass  # cannot tell, so counted
        kept.append(entry)
    return tuple(kept)


def expire_set_aside(
    directory: Path,
    *,
    suffix: str,
    exists: Callable[[Path], bool] | None = None,
    max_age: float = EXPIRED_COPY_MAX_AGE,
    now: float | None = None,
) -> tuple[Path, ...]:
    """Delete, in `directory`, the copies nothing will restore; the records they belonged to.

    Looks at every `<record><SET_ASIDE_MARKER><stamp>[-<n>]` whose record's name ends with
    `suffix` (`.json`), and deletes (#1877):

    * an empty copy older than `PLACEHOLDER_MAX_AGE`, which holds nothing;
    * a copy older than `max_age` whose record does not exist (`exists`, by default
      `Path.is_file`). Its record was deleted or never written again, so no later set-aside
      will apply retention to it.

    A copy of a record that exists is never deleted here: `set_aside_unreadable` keeps the
    newest `KEEP_SET_ASIDE` of those. A name that does not parse as one this module wrote is
    never deleted. Best effort: a copy that cannot be removed is logged and left, and a
    folder that cannot be listed deletes nothing.

    Safe beside another process: a copy is never written after it is made, so one that is
    old stays old, and a copy another process deletes first is skipped. The record is
    checked for each copy just before it is deleted.

    Returns:
        The records at least one non-empty copy was deleted for, so the caller can remove
        what those copies referred to once none is left.
    """
    record_exists = exists if exists is not None else Path.is_file
    clock = time.time() if now is None else now
    try:
        entries = sorted(directory.iterdir()) if directory.is_dir() else []
    except OSError as exc:
        logger.warning(
            "Could not list %s for expired kept copies (%s); none deleted", directory, exc
        )
        return ()
    expired: dict[Path, None] = {}
    for entry in entries:
        name, marker, _ = entry.name.rpartition(SET_ASIDE_MARKER)
        if not marker or not name.endswith(suffix):
            continue
        record = directory / name
        if _age_key(f"{name}{SET_ASIDE_MARKER}", entry.name) is None:
            continue
        try:
            if not entry.is_file():
                continue
            if entry.stat().st_size == 0:  # a placeholder, whatever its record
                _delete_stale_placeholder(entry, clock)
                continue
            if clock - _changed_at(entry) < max_age or record_exists(record):
                continue
            entry.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Could not delete the expired kept copy %s (%s); it stays", entry, exc)
            continue
        logger.info(
            "Deleted %s: its record is gone and the copy was kept for %d days",
            entry,
            int(max_age // 86400),
        )
        expired[record] = None
    return tuple(expired)


def set_aside_copies(path: Path, *, marker: str = SET_ASIDE_MARKER) -> tuple[Path, ...]:
    """Every copy of `path` set aside so far under `marker`, oldest first.

    Ordered by the time in the name and then by its counter, so `-10` follows `-9`. A name
    after the marker that does not parse sorts first, by name.
    """
    if not path.parent.is_dir():
        return ()
    prefix = f"{path.name}{marker}"
    found = [entry for entry in path.parent.iterdir() if entry.name.startswith(prefix)]

    def order(entry: Path) -> tuple[int, str, int, str]:
        key = _age_key(prefix, entry.name)
        return (0, "", 0, entry.name) if key is None else (1, key[0], key[1], "")

    return tuple(sorted(found, key=order))
