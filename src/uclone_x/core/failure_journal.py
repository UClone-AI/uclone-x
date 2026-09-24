"""Local, consent-gated record of runtime failures.

A user who hits a bug in an installed build has nothing to send: the traceback
scrolled away, the log is a file they were never told about, and the failure
that mattered is indistinguishable from the ones that did not. This module is
the collection half of the answer. The transmission half is not here and is
never automatic — `uclone_x.core.diagnostic_report` renders what was collected,
and a person decides whether it leaves the machine.

Three rules the rest of the design follows from:

* **Nothing is recorded before consent is given.** Not buffered, not written
  "just in case" pending an answer. `consent_state()` returns `unasked` until
  someone says yes or no, and `record_failure` is a no-op in that state.
* **Fields are allow-listed, not redacted.** Redaction is a filter over
  arbitrary text and the pattern list cannot anticipate what a prompt contains.
  An allow-list inverts the default: a field reaches the journal because it was
  named here. The one free-text field, the exception message, passes through
  `redact_credentials` *and* home-directory masking on top of that.
* **Recording a failure may not cause one.** Every write is guarded; a journal
  that cannot be written is reported by `ucx report` as unavailable rather than
  raised into the command the user was actually running. Nothing is substituted
  in its place -- the absence is shown, which is what P6 asks for.

  This was a claim before it was true. Review of the first version found that a
  journal made unreadable (`chmod 000`) produced "No failures recorded." --
  byte-identical to a genuinely empty one -- because the write swallowed the
  error and the read returned `[]`. Separately each is defensible; together
  they are a substituted empty result. `read_journal` now returns the failure
  alongside whatever it could parse, and every surface prints it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Final, cast

from uclone_x.core.secrets import redact_credentials

#: Where the journal and the consent record live. `UCLONE_DIAGNOSTICS_DIR` exists
#: for the same reason `UCLONE_SESSION_DIR` does: a test that writes into the
#: invoking user's home directory has escaped its own sandbox.
DIAGNOSTICS_DIR_ENV_VAR: Final = "UCLONE_DIAGNOSTICS_DIR"

_DEFAULT_DIR_NAME: Final = ".uclone"
_DIAGNOSTICS_SUBDIR: Final = "diagnostics"
JOURNAL_FILENAME: Final = "failures.jsonl"
CONSENT_FILENAME: Final = "consent.json"

#: Entries kept. A journal is a recent-failure record, not an archive, and an
#: unbounded file in the user's home directory is a bug of its own.
MAX_ENTRIES: Final = 200

#: Longest exception message kept. Long messages are usually a payload echoed
#: back, which is exactly what must not be collected.
MAX_MESSAGE_CHARS: Final = 300

#: Stack frames kept, in traceback order -- outermost first, the failing frame
#: last, which is how Python prints them. Enough to identify the path through
#: the code, short enough to stay reviewable at a glance.
MAX_FRAMES: Final = 5

CONSENT_GRANTED: Final = "granted"
CONSENT_DENIED: Final = "denied"
CONSENT_UNASKED: Final = "unasked"


@dataclass(frozen=True)
class FailureEntry:
    """One recorded failure. Every field here is one the allow-list admits."""

    ts: str
    code: str
    fingerprint: str
    message: str
    frames: tuple[str, ...]
    ucx_version: str
    python_version: str
    os_name: str
    context: dict[str, str | int | bool] = field(default_factory=dict[str, "str | int | bool"])

    def to_json(self) -> dict[str, object]:
        """Serialise for the journal file."""
        return {
            "ts": self.ts,
            "code": self.code,
            "fingerprint": self.fingerprint,
            "message": self.message,
            "frames": list(self.frames),
            "ucx": self.ucx_version,
            "python": self.python_version,
            "os": self.os_name,
            "context": self.context,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> FailureEntry:
        """Rebuild an entry from a journal line."""
        raw_context: object = payload.get("context", {})
        context: dict[str, str | int | bool] = {}
        if isinstance(raw_context, dict):
            entries = cast(dict[object, object], raw_context)
            for key, value in entries.items():
                if isinstance(value, (str, int, bool)):
                    context[str(key)] = value
        raw_frames: object = payload.get("frames", [])
        frames: tuple[str, ...] = ()
        if isinstance(raw_frames, list):
            frames = tuple(str(item) for item in cast(list[object], raw_frames))
        return cls(
            ts=str(payload.get("ts", "")),
            code=str(payload.get("code", "")),
            fingerprint=str(payload.get("fingerprint", "")),
            message=str(payload.get("message", "")),
            frames=frames,
            ucx_version=str(payload.get("ucx", "")),
            python_version=str(payload.get("python", "")),
            os_name=str(payload.get("os", "")),
            context=context,
        )


def diagnostics_dir() -> Path:
    """Directory holding the journal and the consent record."""
    override = os.environ.get(DIAGNOSTICS_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path.home() / _DEFAULT_DIR_NAME / _DIAGNOSTICS_SUBDIR


def journal_path() -> Path:
    """The journal file."""
    return diagnostics_dir() / JOURNAL_FILENAME


def consent_path() -> Path:
    """The consent record."""
    return diagnostics_dir() / CONSENT_FILENAME


def _inspect(path: Path) -> tuple[int | None, str | None]:
    """`(st_mode, error)` for a path: exactly one of the two is set.

    `st_mode is None, error is None` means the path is genuinely absent.

    Every function in this module that used to open with `path.is_file()` and
    swallow `OSError` now asks this instead. `is_file()` cannot be that
    question: CPython ignores `ENOENT`, `ENOTDIR`, `ELOOP` and `EBADF` inside
    it and answers `False`, so "broken" and "absent" arrive identical — and
    "absent" is then rendered to a person as "nothing has gone wrong", which is
    the substitution this module has been repaired for four times, in four
    different functions. `stat()` raises for all four, and a dangling symlink
    is told apart from an absent path with `lstat`, because only one of them
    means nothing is there.
    """
    try:
        return path.stat().st_mode, None
    except FileNotFoundError:
        try:
            path.lstat()
        except FileNotFoundError:
            return None, None
        except OSError as exc:
            # `stat` said absent and `lstat` cannot say anything: that is not
            # the same as nothing being there, and reporting it as absent is
            # the substitution this helper exists to prevent.
            return None, f"{path} could not be examined: {type(exc).__name__}: {exc}"
        return None, f"{path} is a symbolic link whose target does not exist"
    except OSError as exc:
        return None, f"{path} could not be read: {type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class ConsentRead:
    """The consent answer, and why it could not be read if it could not.

    Both halves are needed and they are not the same decision. The *state*
    falls back to `unasked` whenever the record cannot be read, which fails
    closed: nothing is collected, and the question is put again. The *error*
    exists because telling someone "you have never been asked" when their
    answer is on disk and unreadable is a false statement about their own
    choice -- they granted consent, and the panel says "Currently off."
    """

    state: str
    error: str | None = None


def read_consent() -> ConsentRead:
    """The consent answer with its read failure, if there was one."""
    path = consent_path()
    mode, error = _inspect(path)
    if error is not None:
        return ConsentRead(CONSENT_UNASKED, error)
    if mode is None:
        return ConsentRead(CONSENT_UNASKED)
    if not stat.S_ISREG(mode):
        return ConsentRead(
            CONSENT_UNASKED, f"{path} exists but is not a regular file, so it cannot be read"
        )

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return ConsentRead(CONSENT_UNASKED, f"{path} could not be read: {exc}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ConsentRead(
            CONSENT_UNASKED,
            f"{path} is not readable JSON ({exc}); `ucx report --enable` writes it again",
        )

    if not isinstance(payload, dict):
        return ConsentRead(CONSENT_UNASKED, f"{path} does not contain a JSON object")
    collect: object = cast(dict[str, object], payload).get("collect")
    if collect is True:
        return ConsentRead(CONSENT_GRANTED)
    if collect is False:
        return ConsentRead(CONSENT_DENIED)
    return ConsentRead(
        CONSENT_UNASKED, f"{path} has no `collect` field, so no answer can be read from it"
    )


def consent_state() -> str:
    """`granted`, `denied`, or `unasked`. Fails closed; see `read_consent`."""
    return read_consent().state


def set_consent(collect: bool) -> Path:
    """Record the answer, and return where it was written."""
    path = consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"collect": collect, "answered_at": datetime.now(UTC).isoformat(timespec="seconds")},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


#: Home-directory shapes, not just *this* home directory. Review found the
#: literal-prefix version letting `/Users/alice/...` and `C:\Users\alice\...`
#: through untouched -- a message can name an account that is not the one
#: running the process, and on a shared machine that is someone else's name.
_HOME_SHAPES: Final = re.compile(
    r"(?:/(?:Users|home)/[^/\s:\"\']+|[A-Za-z]:\\Users\\[^\\\s:\"\']+)",
    re.IGNORECASE,
)


def _mask_paths(text: str) -> str:
    """Replace home directories, this process's and anyone else's, with `~`.

    A traceback's message frequently carries an absolute path, and an absolute
    path carries an account name. This is a heuristic like the credential
    redactor beside it, and is documented as one: it recognises the shapes that
    occur, not every path a message could contain.
    """
    home = str(Path.home())
    if home and home != "/":
        # This process's home first, so a case difference between the recorded
        # path and `Path.home()` does not leave the longer form unmasked.
        text = re.sub(re.escape(home), "~", text, flags=re.IGNORECASE)
    return _HOME_SHAPES.sub("~", text)


def _frame_label(filename: str, function: str) -> str:
    """`uclone_x/llm/router.py:route` -- package-relative, never absolute.

    The path is cut at the package root so the label is the same on every
    machine, which is what lets two reports of one bug share a fingerprint.
    """
    path = Path(filename)
    parts = path.parts
    if "uclone_x" in parts:
        index = len(parts) - 1 - parts[::-1].index("uclone_x")
        path = Path(*parts[index:])
    else:
        path = Path(path.name)
    return f"{path.as_posix()}:{function}"


def _frames_of(tb: TracebackType | None) -> tuple[str, ...]:
    """The last `MAX_FRAMES` frames of the traceback, failing frame last."""
    if tb is None:
        return ()
    summary = traceback.extract_tb(tb)
    labels = [_frame_label(frame.filename, frame.name or "?") for frame in summary]
    return tuple(labels[-MAX_FRAMES:])


def fingerprint_of(code: str, frames: tuple[str, ...]) -> str:
    """A stable short id for "the same bug", for grouping and for de-duplication.

    Derived from the exception type and the package-relative frames, so it is
    identical across machines and across runs, and changes when the code path
    changes. Line numbers are deliberately excluded: an unrelated edit above the
    failure would otherwise split one bug into two.
    """
    digest = hashlib.sha256("|".join([code, *frames]).encode("utf-8")).hexdigest()
    return digest[:8]


def build_entry(
    error: BaseException,
    *,
    context: Mapping[str, object] | None = None,
) -> FailureEntry:
    """Build a journal entry from an exception. Pure; writes nothing."""
    from uclone_x import __version__

    code = type(error).__name__
    frames = _frames_of(error.__traceback__)
    message = _mask_paths(redact_credentials(str(error)))[:MAX_MESSAGE_CHARS]

    # The signature already says `str | int | bool`, and a caller that respects
    # it needs no filtering. The check is here because the caller is a failure
    # path in someone else's module: a `Path` or an exception object passed by
    # mistake is exactly the kind of value that must not reach the file, and a
    # type annotation is not enforcement at runtime.
    allowed_context: dict[str, str | int | bool] = {}
    for key, value in (context or {}).items():
        if isinstance(value, str):
            # The same two filters the message gets, and the same cap. Review
            # found context strings going in raw: `sk-ant-api03-…` passed
            # through verbatim and a 5000-character value was stored whole.
            # No caller does that today, which is exactly why it would not have
            # been noticed when one started.
            allowed_context[key] = _mask_paths(redact_credentials(value))[:MAX_MESSAGE_CHARS]
        elif isinstance(value, (int, bool)):
            allowed_context[key] = value

    return FailureEntry(
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        code=code,
        fingerprint=fingerprint_of(code, frames),
        message=message,
        frames=frames,
        ucx_version=__version__,
        python_version=platform.python_version(),
        # Release, not `platform.platform()`: the latter carries the machine's
        # processor string and, on some systems, its hostname.
        os_name=f"{platform.system()} {platform.release()}",
        context=allowed_context,
    )


def record_failure(
    error: BaseException,
    *,
    context: Mapping[str, object] | None = None,
) -> FailureEntry | None:
    """Append a failure to the journal, if consent was given.

    Returns the entry written, or `None` when consent is absent or the write
    could not be made. Never raises: this runs on the failure path of whatever
    the user was actually doing, and a diagnostics writer that turns a handled
    error into a crash has made the product worse in the name of observing it.
    """
    if consent_state() != CONSENT_GRANTED:
        return None

    try:
        entry = build_entry(error, context=context)
        path = journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_json()) + "\n")
        _trim(path)
        return entry
    except Exception:  # noqa: BLE001 - see the docstring: this path cannot raise
        return None


def _trim(path: Path) -> None:
    """Keep the journal to `MAX_ENTRIES`, discarding the oldest."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_ENTRIES:
            path.write_text("\n".join(lines[-MAX_ENTRIES:]) + "\n", encoding="utf-8")
    except OSError:
        return


@dataclass(frozen=True)
class JournalRead:
    """What a read of the journal found, including what it could not read.

    `error` is the point of this type. A list on its own cannot distinguish
    "nothing has failed" from "the record of what failed is unreadable", and
    those two want opposite responses from the person looking at them.
    """

    entries: tuple[FailureEntry, ...] = ()
    error: str | None = None
    unreadable_lines: int = 0
    #: Why new failures are not being kept, if they are not. Separate from
    #: `error`, which is about reading: a journal can be perfectly readable
    #: while nothing more can be appended to it, and a reader shown only the
    #: entries would conclude that nothing else has gone wrong.
    recording_blocked: str | None = None

    @property
    def is_available(self) -> bool:
        """Whether the journal could be read at all."""
        return self.error is None


def _recording_blocked() -> str | None:
    """Why a failure could not be recorded right now, if it could not.

    This asks the invariant — *with consent granted, recording must be
    possible* — rather than a branch of some other question. Attaching it to
    "the journal is absent" was the fifth instance of this module's recurring
    defect: a journal that exists but cannot be appended to (mode 444, or
    root-owned after one `sudo ucx ...`) had `record_failure` dropping every
    failure while the read reported an available, empty journal. Same
    conjunction, reached through `S_ISREG` instead of `mode is None`.

    `os.access` is advisory and uses the real uid, so this is a best-effort
    answer to a question the filesystem only truly answers on write: it misses
    ENOSPC, quota and a read-only remount. It is still worth asking, because
    the states it does catch are the ones that persist silently.
    """
    if consent_state() != CONSENT_GRANTED:
        return None

    path = journal_path()
    mode, error = _inspect(path)
    if error is not None:
        return error

    if mode is not None:
        if not stat.S_ISREG(mode):
            return f"{path} is not a regular file, so failures cannot be appended to it"
        if not os.access(path, os.W_OK):
            return (
                f"failures cannot be recorded: {path} is not writable, so nothing new is "
                "being kept. This is not the same as nothing having failed."
            )
        return None

    directory = path.parent
    dir_mode, dir_error = _inspect(directory)
    if dir_error is not None:
        return dir_error
    if dir_mode is None:
        # Not created yet; `record_failure` makes it. Nothing to report until
        # something tries and fails.
        return None
    if not stat.S_ISDIR(dir_mode):
        return f"{directory} is not a directory, so no journal can be created in it"
    if not os.access(directory, os.W_OK):
        return (
            f"failures cannot be recorded: {directory} is not writable, so nothing has been "
            "kept. This is not the same as nothing having failed."
        )
    return None


def _read_failure(error: str) -> JournalRead:
    """A read failure, carrying the recording state when it adds something.

    Every error return in `read_journal` goes through here. Three of the four
    did not, so the commonest case -- a journal `chmod 000` -- reported only
    that it could not be read while recording was equally blocked, and the
    commit that introduced the field claimed otherwise.

    The deduplication matters as much as the addition: on a path that fails
    `_inspect`, `_recording_blocked()` re-runs the same inspection and returns
    the same sentence, so setting both printed it twice under two headings.
    Saying one thing twice is its own way of making a report harder to read.
    """
    blocked = _recording_blocked()
    return JournalRead(error=error, recording_blocked=None if blocked == error else blocked)


def read_journal() -> JournalRead:
    """Read the journal, reporting failure rather than reporting emptiness.

    Every way the read can fail ends in `error`, and the first version of this
    function only handled the one that had been reported. Review found two more
    a branch later: a path that exists but is a directory returned an empty
    read, and a file of non-UTF-8 bytes raised `UnicodeDecodeError` -- a
    `ValueError`, outside the `OSError` that was caught -- straight through the
    CLI as a traceback and out of the API as a 500. A write interrupted
    mid-multibyte character reaches that second case more easily than the
    permission error that prompted any of this.

    The guard is built on `stat()` rather than on `exists()`, and that is the
    correction a third review forced. Wrapping `exists()` in a `try` looks like
    a guard and is not one: CPython swallows `ENOENT`, `ENOTDIR`, `ELOOP` and
    `EBADF` inside `Path.exists()` and returns `False`, so a symlink loop, a
    dangling symlink, and a diagnostics directory whose parent is a regular
    file all arrived here as "the journal does not exist" -- available, empty,
    nothing wrong. In the `ENOTDIR` case `record_failure` is also failing to
    write, so failures were being dropped *and* reported as none: the exact
    conjunction this module was twice repaired to stop.

    `stat()` raises for all four, which is why the decision is made from it.
    The decode stays separate from the read, because "unreadable bytes" and
    "unreadable file" are different sentences to show someone.
    """
    path = journal_path()

    mode, error = _inspect(path)
    if error is not None:
        return _read_failure(error)

    if mode is None:
        return JournalRead(recording_blocked=_recording_blocked())

    if not stat.S_ISREG(mode):
        return _read_failure(
            f"{path} exists but is not a regular file, so no record can be read from it"
        )

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return _read_failure(f"{path} could not be read: {type(exc).__name__}: {exc}")

    blocked = _recording_blocked()

    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        return _read_failure(
            f"{path} is not valid UTF-8 and cannot be parsed: {exc}. "
            "A run interrupted mid-write leaves the file this way; "
            "`ucx report --clear` discards it."
        )

    entries: list[FailureEntry] = []
    unreadable = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            payload: object = json.loads(line)
        except json.JSONDecodeError:
            unreadable += 1
            continue
        if isinstance(payload, dict):
            entries.append(FailureEntry.from_json(cast(dict[str, object], payload)))
        else:
            unreadable += 1

    return JournalRead(
        entries=tuple(entries), unreadable_lines=unreadable, recording_blocked=blocked
    )


def read_entries() -> list[FailureEntry]:
    """The entries alone, for callers that have already handled unavailability.

    Kept narrow on purpose: anything that renders the journal to a person must
    go through `read_journal` and say when it could not be read.
    """
    return list(read_journal().entries)


@dataclass(frozen=True)
class ClearOutcome:
    """What a delete did, distinguishing "nothing there" from "could not".

    The previous `bool` collapsed them, and the surfaces rendered `False` as
    success: `ucx report --clear` printed "Nothing to delete." at exit 0 with
    the file still on disk, and the API answered `200 {"cleared": false}` --
    which the dashboard's new response check could not see, because the
    request had succeeded.
    """

    deleted: bool
    error: str | None = None


def clear_journal() -> ClearOutcome:
    """Delete the journal, reporting a failure to delete as one."""
    path = journal_path()

    mode, error = _inspect(path)
    if error is not None:
        return ClearOutcome(False, error)
    if mode is None:
        return ClearOutcome(False)

    try:
        path.unlink()
    except OSError as exc:
        return ClearOutcome(False, f"{path} could not be deleted: {type(exc).__name__}: {exc}")
    return ClearOutcome(True)


def record_unhandled_exception(
    exc_type: type[BaseException],
    exc: BaseException,
    tb: TracebackType | None,
) -> FailureEntry | None:
    """Record an exception that reached the top of the process.

    Separate from `record_failure` only because `sys.excepthook` hands over the
    three-part form and the traceback may not be attached to the exception.
    """
    if exc.__traceback__ is None and tb is not None:
        exc = exc.with_traceback(tb)
    return record_failure(exc, context={"unhandled": True})


def install_excepthook() -> None:
    """Route unhandled exceptions through the journal, then to the real hook."""
    previous = sys.excepthook

    def hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        record_unhandled_exception(exc_type, exc, tb)
        previous(exc_type, exc, tb)

    sys.excepthook = hook
