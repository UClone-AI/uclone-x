"""Core-owned session state and its single persistence boundary (P5, P8).

**Why this module exists in `agent/` and not in `ui/`.** P8 states that "all business
logic, session state, conversation histories, tool execution outcomes, and reasoning
loops MUST reside exclusively in the Core Engine", and that the Developer UI "MUST NEVER
hold primary state, manage independent conversation lifecycles, or implement logic that
is absent from the Core". Before this module the only session persistence in the tree
was `uclone_x.ui.app.AgentSessionManager` — a FastAPI-layer class holding the storage
directory, the path-traversal guard, the atomic write and its own reset semantics. That
placement is the P8 violation, so the store moves here and the UI becomes its client.

A second store beside the UI's was rejected rather than merely avoided: it would give
the repository four reset semantics with no rule for which wins, two copies of one
path-traversal guard (a duplicated security control is one that gets fixed in a single
copy), and divergent on-disk schemas, so that a UI-written session would be unreadable
by the headless CLI. The inverse — Core importing `uclone_x.ui` — was rejected too: no
module in the tree has `agent/` depending on `ui/`, and it would make headless CLI and
A2A persistence depend on FastAPI.

**One reset, defined once.** `SessionState.reset` is the sole reset semantics in the
Core. It exists because there were three divergent ones: `AgentSessionManager.
clear_session_history` (which re-seeded the system prompt in two duplicated blocks),
the CLI REPL's `/reset` (which assigned `agent._history` through a private-access pragma
and seeded `content=config.system_prompt or ""`, inserting an *empty* `SYSTEM` message —
a state `BaseAgent.__init__` never produces), and `BaseAgent.__init__` itself, which
neither of the others called. `reset` reproduces `__init__`'s seeding exactly: a `SYSTEM`
message if and only if the system prompt is non-empty.

Re-seeding the system prompt is **not** optional and **not** automatic.
`BaseAgent._prepare_turn_messages` recomposes only the P7 asserted invariants on every
turn; `config.system_prompt` is written into history once in `__init__` and never
recomposed. A reset that dropped it would silently strip the agent's instructions while
leaving the ontology grounding intact.

**Timestamps are ISO-8601 strings, not `datetime`.** These models are `strict=True`,
under which Python-mode validation refuses a `str` where a `datetime` is declared, so a
string field stays readable by either validation mode. `SessionStore.load` documents the
related constraint on enums.

**Validation is a property of the constructor, not of the type.** These models are
`strict=True`, but that buys nothing on a route that never runs validation. Each such
route is a way to put a `str` where an `int` is declared and have it stick. There are
two doors:

* **`model_copy(update=...)` on the frozen model.** Closed in `SessionState` (defined in
  `uclone_x.core.session_state` since #1734, re-exported here) and in this module:
  `SessionState.with_messages` and `SessionState.reset` construct rather than copy, and
  `SessionStore.save` re-validates the serialized payload at the persistence boundary as
  the backstop — which is also what lets it return a value identical to what `load` will
  yield.
* **A direct attribute write into the mutable working copy that `BaseAgent` keeps for a
  session's messages.** Closed in `BaseAgent.load_history`, which routes its arguments
  through `SessionState`. That door and its fix both belong to the `BaseAgent`
  multi-session seam of #183, **not to this module** — nothing here imports it or
  depends on it, so do not expect to find either in this file.

Closing one door leaves the other open, and neither call site's docstring used to
mention that the other existed, so a reader of either could not tell there was a family
at all. Every function named above carries a pointer back to this paragraph, and
`tests/unit/test_agent_session_multitenancy.py` asserts that by introspecting the live
docstrings — an earlier version of this paragraph claimed the pointers existed when
three of the four were missing, which is the same unfalsifiable-prose defect it was
written to fix. Read this before changing either door.

**What this family does *not* cover: the forgeable revision precondition (#248).** It
looks like a third door and it is not one, so it is named here to stop the next reader
placing it. Both doors above are about a value that **never passed validation** — a `str`
where an `int` is declared. A forged `revision` is a perfectly valid `int`, so the closure
this family takes (route the update through `model_validate`, as `ontology.models`,
`skills.models` and `engine.event_bus` already do for #31/#42) leaves it untouched:
measured, `model_copy(update={"revision": "2"})` starts raising and
`model_copy(update={"revision": 2})` stays accepted. The decision on #248, with the three
rejected options and their costs, is in `SessionStore.save`.

**A session id is a filename, and two filesystems in common use fold it (#256).** APFS
and NTFS compare names case-insensitively and without regard to Unicode normalization, so
`"SessA"` and `"sessa"`, and NFC `"séance"` against its NFD spelling, are **one file** and
two distinct `str`s. Both are legal single path components, so `validate_session_id` and
`resolve_session_path` pass them: the P3 control is not implicated and there is nothing
there to fix. The record's own `session_id` field is what settles ownership, checked by
`verify_record_identity` at every door that reaches a record by id — `SessionStore.load`,
`save` (via `load`, *before* the revision precondition), `delete`, and
`ui.app.AgentSessionManager.load_session_record`.

**Four filename encodings were considered and all four rejected**, recorded here because
the option that was chosen looks like the absence of a decision unless the alternatives
are written down:

* **Percent-encode the id.** Does not fix the case arm *at all* — an unreserved ASCII
  letter is not escaped, so `SessA` and `sessa` still fold — while adding a decode step to
  `list_session_ids` and invalidating existing filenames. It fixes only the normalization
  arm, and half a fix here reads as a whole one.
* **Hash the id** (`sha256(id)[:n].json`, id inside the record). Genuinely collision-free
  on every filesystem, and the most expensive migration available: **every** existing
  filename becomes wrong and a session id stops being visible in `ls`. (Its other cost —
  opening every record to enumerate — turned out **not** to be a point of difference:
  `list_session_ids` has to do that anyway, because a stem is not trustworthy as an id.)
  Decisive against it: `load` reports
  an unreadable record as *absent*, so a rename pass that half-completes presents as every
  session having vanished — and a record whose name is wrong is not merely unreadable, it
  is invisible to enumeration.
* **Case-fold the filename and reject a variant that collides.** Discards the case the
  user chose, so `list_session_ids` can no longer report the real id; still needs a rename
  pass; still silent on normalization unless it also normalizes; and still cannot help a
  tree that already holds a variant pair.
* **Refuse an id that normalizes onto an existing record.** Write-time only, and the
  defect is a *read*: a caller that loads before writing holds a current `revision`, so
  its write passes every check `save` has (measured — see `verify_record_identity`). A
  variant pair is also legal on ext4 and becomes a collision only when that tree is read
  on macOS, so no write-time rule makes an existing tree safe.

**What existing records get: nothing to do.** Every record already on disk carries
`session_id`, so no file is renamed, no filename becomes invalid, and no record becomes
invisible to `list_session_ids`. That is the deliberate contrast with `revision`'s default
of `0` in #240 — there, a default was needed precisely because existing records lacked the
field.

**This record is NOT interchangeable with the UI's.** An earlier draft of this docstring
claimed the on-disk key set was aligned with the one `ui.app.AgentSessionManager`
writes. That is false, and measurement is what settled it: the UI writes `turns` where
this writes `turn_counter`, and its `messages` are UI *presentation* records — `id`,
`sender`, `timestamp`, `latency_ms`, `tokens_used`, a display `provenance` block — not
`ChatMessage`s. They are two different artifacts with two different schemas, which is
why the Core record lives under a `core/` subdirectory rather than sharing the UI's
filename; pointing both at one path would have them silently overwrite each other.
Nothing here reads or writes a UI transcript.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.core.log_offset import LogOffset, LogOffsetAllocatorProtocol
from uclone_x.core.log_writer import LogWriterProtocol, RedactingLogWriter
from uclone_x.core.provenance import Provenance
from uclone_x.core.secrets import redact_credentials
from uclone_x.core.session import (
    CORE_RECORD_SUBDIR,
    DEFAULT_SESSION_STORAGE_DIR,
    SESSION_STORAGE_DIR_ENV_VAR,
    UI_TRANSCRIPT_SUBDIR,
    default_session_root,
    default_session_storage_dir,
    resolve_session_path,
    validate_session_id,
)
from uclone_x.core.session_log import with_image_data
from uclone_x.core.session_state import (
    AnchorAuthor,
    AnchorProvenance,
    ContextSnapshot,
    SessionState,
    content_digest,
    history_entries_not_their_messages,
    log_position,
    recorded_before_history_entries,
    redact_message,
)
from uclone_x.core.set_aside import expire_set_aside, kept_copies, set_aside_unreadable
from uclone_x.errors import (
    PathTraversalError,
    SessionEventLogNotConfiguredError,
    SessionIdCollisionError,
    SessionRecordUnreadableError,
    StaleSessionWriteError,
)
from uclone_x.llm.models import ChatMessage, LedgerSource

logger = logging.getLogger(__name__)

# The names imported from `uclone_x.core.session` and `uclone_x.core.session_state` above
# are re-exported here under their old spelling (#1734); every caller of
# `uclone_x.agent.session` keeps working.
__all__ = [
    "AnchorAuthor",
    "AnchorProvenance",
    "DEFAULT_SESSION_STORAGE_DIR",
    "CONTEXT_BODY_SUBDIR",
    "CompactionResult",
    "ContextSnapshot",
    "SessionState",
    "SessionStore",
    "CORE_RECORD_SUBDIR",
    "EVENT_LOG_SUBDIR",
    "MAX_PID",
    "SESSION_STORAGE_DIR_ENV_VAR",
    "UI_TRANSCRIPT_SUBDIR",
    "content_digest",
    "default_session_root",
    "default_session_storage_dir",
    "is_pid_alive",
    "reap_orphaned_temp_files",
    "redact_message",
    "resolve_session_path",
    "validate_session_id",
    "verify_record_identity",
]

# Where a store that was given no log writer keeps each session's durable turn events:
# `<storage_dir>/events/<session_id>.jsonl`, with the offset cursor beside it as
# `<session_id>.cursor`. Under the store's own directory, so a store redirected to a
# temporary directory takes its event log with it, and a subdirectory so that
# `list_session_ids`, which reads `*.json` at the top level, never sees a log.
EVENT_LOG_SUBDIR = "events"

# The bodies a session refers to by hash -- context-snapshot layers (tool schemas,
# identity, slow context), session-log entries, and the full text of every tool result
# too long to keep whole in the history (#1848) -- live at
# `<storage_dir>/context/<session_id>/<sha256>`, once each.
CONTEXT_BODY_SUBDIR = "context"

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _now_iso() -> str:
    """Current UTC instant as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


_MAX_PID = 2**31 - 1
"""The largest pid `os.kill` will accept without raising `OverflowError`.

`pid_t` is `int32_t`, so this is its maximum. Measured on this platform:
`os.kill(2**31 - 1, 0)` raises `ProcessLookupError` (in range, no such process) and
`os.kill(2**31, 0)` raises `OverflowError` (out of range) — so the bound is exact and no
real pid is excluded by it.
"""
MAX_PID = _MAX_PID


def is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is currently alive on the system.

    Safe and non-raising across POSIX and Windows platforms.
    """
    if pid <= 0 or pid > _MAX_PID:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but belongs to a different user
        return True
    except (OverflowError, OSError):
        return False


_TEMP_FILE_PATTERN = re.compile(r"^.+\.tmp\.(\d+)\.[0-9a-fA-F]{8}$")


def reap_orphaned_temp_files(storage_dir: Path, max_age_seconds: float = 300.0) -> int:
    """Reap orphaned session temporary files left behind by crashed writers (#219, #257, #269).

    Recognizes temp files matching `<name>.tmp.<pid>.<uuid8>` across the session storage
    directory. A temp file is safely unlinked if and only if:
    1. The PID encoded in the filename is no longer alive on the system, OR
    2. The temp file's modification age exceeds `max_age_seconds`.

    Committed session records whose session ID happens to contain `.tmp.<pid>.` (e.g.
    `chat.tmp.999999.deadbeef.json`) do not match the terminal temp file pattern and are
    never unlinked (#269).

    Returns the number of orphaned temp files successfully unlinked.
    """
    if not storage_dir.is_dir():
        return 0
    reaped_count = 0
    now = datetime.now(UTC).timestamp()
    for entry in storage_dir.glob("*.tmp.*"):
        if not entry.is_file():
            continue
        match = _TEMP_FILE_PATTERN.match(entry.name)
        if match is None:
            continue
        try:
            pid = int(match.group(1))
        except ValueError:
            continue

        # An out-of-range pid cannot name a process, so the file is debris. It is still put
        # through the age guard rather than deleted outright (identical to
        # log/file_allocator.py): an uninterpretable pid is not treated as unconditionally
        # orphaned, because doing so could delete a file belonging to a different naming scheme,
        # but neither is it skipped forever, which would leak.
        uninterpretable_pid = pid > _MAX_PID
        is_orphan = not uninterpretable_pid and not is_pid_alive(pid)

        if not is_orphan and max_age_seconds > 0:
            try:
                mtime = entry.stat().st_mtime
                if (now - mtime) >= max_age_seconds:
                    is_orphan = True
            except OSError:
                continue

        if is_orphan:
            try:
                entry.unlink(missing_ok=True)
                reaped_count += 1
            except OSError as err:
                logger.warning("Failed to reap orphaned session temp file %s: %s", entry, err)
    return reaped_count


def verify_record_identity(asked_session_id: str, record_session_id: str, path: Path) -> None:
    """Refuse a record that identifies a session other than the one asked for (#256).

    Module-level and public for the same reason `resolve_session_path` is: it is one rule
    with more than one caller — `SessionStore.load` for Core records and
    `ui.app.AgentSessionManager.load_session_record` for the UI's presentation transcript
    — and two copies of a rule are two things to fix. `resolve_session_path` cannot host
    it, because the rule needs the record's *content* and a path resolver has none.

    **What it closes, and why a filename encoding does not.** A session id is interpolated
    into a filename, and the default macOS and Windows filesystems are case-insensitive
    *and* normalization-insensitive, so `"SessA"` and `"sessa"` name one file while
    remaining two distinct `str`s. Both are legal single path components, so the P3 name
    and containment rules above pass them correctly — the collision is below the guard, in
    the filesystem. Measured on `e8b3e2f`: a `save` of `"SessA"` followed by
    `load("SESSA")` returned `SessA`'s whole conversation, and the returned record
    identified itself as `SessA`, so the caller adopted another session's history and then
    persisted onto it with a revision the precondition legitimately accepted.

    So the comparison is against the id **inside** the record, never against the filename:

    * It is independent of how a given filesystem folds names. APFS preserves the
      normalization a name was created with while comparing insensitively; HFS+ rewrote
      names to NFD on write; ext4 does neither. A filename-derived check would have to
      model all three. The record's `session_id` field is the id the caller supplied,
      byte for byte, on every one of them.
    * It needs **no migration**. Every record already on disk carries the field, so every
      existing filename stays valid and no record has to be renamed to become reachable.
      An encoding scheme — percent-encoding or a hash — makes every existing filename
      wrong instead, and a record whose *name* becomes invalid is worse than unreadable:
      `load` reports unparseable as absent and enumeration cannot see it at all, so a
      partial rename pass looks exactly like the sessions vanishing.

      What this does **not** buy is a filename that can be trusted as an id.
      `SessionStore.list_session_ids` therefore reads the id out of each record too,
      rather than off its stem — an earlier version of it read stems and inverted itself
      on a tree where a filename and its record disagree. The rule is the same in both
      places for the same reason; see that method.
    * It closes the arm that only *reads*. Refusing at write time cannot: a variant-id
      caller that loads first holds a current `revision`, so its write passes every check
      `save` has. And a variant pair created on ext4 is legal there, becoming a collision
      only when the tree is later read on macOS, so no write-time rule can make an
      existing tree safe.

    Comparison is **exact**, not case-folded and not normalization-folded. Folding would
    make two distinct ids deliberately resolve to one record, which is the defect with a
    policy attached rather than a fix: a Linux user with genuinely separate `SessA` and
    `sessa` sessions would find them merged, silently, which is what P6 forbids.

    Args:
        asked_session_id: The id the caller addressed.
        record_session_id: The `session_id` the record on disk carries.
        path: The single file both ids resolve to, named in the error so a report can
            point at the record rather than leave the user to find it.

    Raises:
        SessionIdCollisionError: If the two ids differ.
    """
    if asked_session_id == record_session_id:
        return
    raise SessionIdCollisionError(
        f"Refusing to use the record at {path} for session {asked_session_id!r}: it "
        f"identifies session {record_session_id!r}"
        f"{_describe_id_fold(asked_session_id, record_session_id)}. The two ids name one "
        f"file on this filesystem, so they cannot both be stored here and reading either "
        f"one would return the other's conversation. Choose a session id that differs "
        f"from {record_session_id!r} by more than case or Unicode normalization.",
        asked_session_id=asked_session_id,
        record_session_id=record_session_id,
        path=path,
    )


def _describe_id_fold(asked: str, found: str) -> str:
    """Name which filesystem fold made two distinct ids collide, for the error message.

    Without this the normalization case reads as a self-contradiction. NFC `"séance"` and
    its NFD spelling render **identically** in a terminal and even under `repr`, so the
    unqualified message says *"for session 'séance': it identifies session 'séance'"* and
    a user has no way to see what is being complained about. Naming the fold, and showing
    the code points when it is a normalization difference, is the difference between a
    report and a riddle.
    """
    if asked.casefold() == found.casefold():
        return ", which differs only by letter case"
    nfc_asked = unicodedata.normalize("NFC", asked)
    nfc_found = unicodedata.normalize("NFC", found)
    if nfc_asked == nfc_found:
        # The escaped forms, because the printable forms are indistinguishable.
        return (
            f", which differs only by Unicode normalization "
            f"(asked {asked.encode('unicode_escape').decode('ascii')}, "
            f"record {found.encode('unicode_escape').decode('ascii')})"
        )
    if nfc_asked.casefold() == nfc_found.casefold():
        return ", which differs only by letter case and Unicode normalization"
    # No fold this function knows about explains it, so it does not claim one. Reachable
    # when a record has been renamed or hand-edited so its filename and its `session_id`
    # disagree for a reason unrelated to #256 — a real collision to refuse, with a cause
    # this function must not guess at.
    return ""


class CompactionResult(BaseModel):
    """What one session-level compaction did (P5, P6).

    Two types describe a compaction and the split is the P5 boundary, not duplication:
    `CompactionOutcome` is the LLM layer's answer about the pass it ran — which producer
    wrote the ledger, and its attribution. This is the Core's answer about the *session*:
    which session, how many tokens and messages it held before and after, and why the
    pass was run. The Core owns none of the algorithm and all of the session bookkeeping,
    which is exactly the division P5 requires.

    Token counts come from the compactor's own `estimate_tokens`, never from a
    reimplementation: a saving computed by a different estimator than the one that
    decided to compact is not a measurement of anything.

    `provenance` is forwarded verbatim from the outcome. The Core does not build it —
    for an LLM-written ledger the attribution belongs to the summarizer, and naming
    `agent.core` as the server of text a model produced is the substitution P6 forbids.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    session_id: str
    reason: str = Field(
        description="Why the pass ran — `auto_threshold` when `execute_turn` tripped "
        "the configured token threshold before a turn, or a caller-supplied reason for an "
        "explicit `compact_session`. A record written before #1443 may also say "
        "`auto_threshold_mid_turn`, a pass between two steps, which no longer runs.",
    )
    ledger_source: LedgerSource = Field(
        description="Which producer wrote the ledger, forwarded from the outcome.",
    )
    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int
    keep_recent_turns: int = Field(
        description="The compactor's recent-turn window, read off the compactor rather "
        "than assumed, so a non-default configuration is reported accurately.",
    )
    superseded_ledger_count: int = Field(
        default=0,
        description="Prior ledgers this pass stopped carrying forward (issue #196).",
    )
    provenance: Provenance | None = Field(
        description="In-band attribution required by Principle 6, forwarded verbatim "
        "from the compaction outcome. Explicit with no default.",
    )

    @property
    def saved_tokens(self) -> int:
        """Tokens the pass removed from the context. Never negative."""
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def compression_ratio_pct(self) -> float:
        """Proportion of the estimated context the pass removed, as a percentage."""
        if self.tokens_before <= 0:
            return 0.0
        return round(self.saved_tokens / self.tokens_before * 100.0, 2)


class SessionStore:
    """The single persistence boundary for Core session state (P8).

    Writes are atomic by `os.replace` onto a same-directory temporary file. The
    temporary name is built from the process id and a random suffix, deliberately not
    from `asyncio.get_running_loop().time()` as the UI transcript writer once did: that
    call raises `RuntimeError` outside a running event loop, which makes the write
    unavailable to exactly the headless synchronous callers — the CLI REPL and the
    A2A path — that P8 says must be able to reach session state without the UI.

    Writes are also **optimistically concurrency-controlled** on `SessionState.revision`,
    which is a different property from atomicity and was the one missing: see `save`. The
    two are kept distinct on purpose, because "the write is atomic" was read as "two
    writers are safe" for long enough to lose a compaction (#219).
    """

    def __init__(
        self,
        storage_dir: str | os.PathLike[str] | None = None,
        log_writer: LogWriterProtocol | None = None,
        log_allocator: LogOffsetAllocatorProtocol | None = None,
    ) -> None:
        """Build a store rooted at `storage_dir`, a `Path` or a path string.

        **The durable event log is on by default (#1442).** With neither `log_writer` nor
        `log_allocator` given, each session's turn events are appended to
        `<storage_dir>/events/<session_id>.jsonl` through a `RedactingLogWriter`, and
        their offsets are allocated by a `FileLogOffsetAllocator` over the same
        directory. Before this every production store was built without a writer, and
        `save` drained the agent's event queue into nothing: no call site passed one, so
        making it the default is what reaches all of them, including the ones added
        later. Passing both still injects a single writer and allocator, as tests do.
        Passing only one leaves the store unable to write events, and `save` refuses
        events it cannot write rather than dropping them.
        """
        self._log_writer = log_writer
        self._log_allocator = log_allocator
        #: Set only in the default mode: the per-session log files live here. Nothing is
        #: created until the first event is written, so a store that only reads (the
        #: diagnostics commands) leaves no empty directory behind.
        self._event_log_dir: Path | None = None
        self._storage_dir: Path = (
            Path(storage_dir).resolve()
            if storage_dir is not None
            else default_session_storage_dir().resolve()
        )
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        if log_writer is None and log_allocator is None:
            self._event_log_dir = self._storage_dir / EVENT_LOG_SUBDIR
        #: Session ids whose unreadable record `save` moved aside, until `take_set_aside`
        #: reports each one once (#1844).
        self._set_aside: set[str] = set()
        reap_orphaned_temp_files(self._storage_dir)
        self._expire_kept_copies()

    @property
    def storage_dir(self) -> Path:
        """Root directory holding this store's session records."""
        return self._storage_dir

    def session_path(self, session_id: str) -> Path:
        """Resolve `session_id` to its record path, refusing anything that escapes (P3).

        Delegates to `resolve_session_path`, the single implementation of this guard.

        Raises:
            PathTraversalError: See `resolve_session_path`.
        """
        return resolve_session_path(self._storage_dir, session_id)

    def event_log_path(self, session_id: str) -> Path | None:
        """Where this store appends `session_id`'s turn events, or `None` if injected.

        `None` means the store was built with its own writer, whose destination it does
        not know. The id is guarded by `session_path` first, so an id that could not name
        a record cannot name a log either.
        """
        if self._event_log_dir is None:
            return None
        self.session_path(session_id)
        return self._event_log_dir / f"{session_id}.jsonl"

    def context_body_dir(self, session_id: str) -> Path:
        """Where `session_id`'s context-snapshot bodies are stored, one file per SHA-256."""
        self.session_path(session_id)
        return self._storage_dir / CONTEXT_BODY_SUBDIR / session_id

    def save_context_body(self, session_id: str, digest: str, body: str) -> None:
        """Store one layer body under its SHA-256, unless it is already stored.

        Written as soon as a snapshot names it, before the record that holds the snapshot,
        so a saved record never names a body that is not on disk. Content-addressed, so a
        second write of the same text is a no-op, and a body never changes once written.

        The body is redacted like a message is. A redacted body no longer hashes to its
        name, and `rebuild_requests` reports that request as not verified rather than
        presenting the redacted text as what was sent.
        """
        if not _SHA256_HEX.fullmatch(digest):
            raise ValueError("A context body is named by its SHA-256, 64 lowercase hex characters.")
        path = self.context_body_dir(session_id) / digest
        if path.is_file():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f"{digest}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            staging.write_text(redact_credentials(body), encoding="utf-8")
            os.replace(staging, path)
        except Exception:
            staging.unlink(missing_ok=True)
            raise

    def load_context_body(self, session_id: str, digest: str) -> str | None:
        """The body stored under `digest` for `session_id`, or `None` if there is none."""
        if not _SHA256_HEX.fullmatch(digest):
            return None
        path = self.context_body_dir(session_id) / digest
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def clear_event_log(self, session_id: str) -> None:
        """Remove `session_id`'s event log and its offset cursor, if this store owns them.

        Clearing a conversation clears its history, and the event log is history too: it
        holds every tool output the conversation produced. `delete` calls this, and so
        does `BaseAgent.reset_session`, so "clear history" leaves the same disk whether or
        not an agent was live when it ran (#1442). The next event starts a new log, with
        a header and offsets from 1. A store with an injected writer does not know where
        that writer puts its lines, so it removes no log.

        The context-snapshot bodies go too, in either case (#1421). They are the rest of
        what the log's `REQUEST_CONTEXT` events refer to, and the store always owns them.
        """
        body_dir = self.context_body_dir(session_id)
        if body_dir.is_dir():
            shutil.rmtree(body_dir)
        log_path = self.event_log_path(session_id)
        if log_path is None:
            return
        log_path.unlink(missing_ok=True)
        (log_path.parent / f"{session_id}.cursor").unlink(missing_ok=True)

    def _event_log_for(
        self, session_id: str
    ) -> tuple[LogWriterProtocol, LogOffsetAllocatorProtocol] | None:
        """The writer and allocator for `session_id`'s events, or `None` if unwired."""
        log_path = self.event_log_path(session_id)
        if log_path is None:
            if self._log_writer is None or self._log_allocator is None:
                return None
            return self._log_writer, self._log_allocator
        # Deferred: filesystem adapters that only a store actually writing an event needs.
        from uclone_x.log.file_allocator import FileLogOffsetAllocator
        from uclone_x.log.reader import CURRENT_LOG_FORMAT_VERSION, CURRENT_LOG_SCHEMA

        allocator = FileLogOffsetAllocator(log_path.parent)
        writer = RedactingLogWriter(log_path)
        # The reader refuses a log with no header line (#568), so a new log starts with one.
        if not log_path.is_file() or log_path.stat().st_size == 0:
            writer.write_entry(
                {
                    "schema": CURRENT_LOG_SCHEMA,
                    "version": CURRENT_LOG_FORMAT_VERSION,
                    "session_id": session_id,
                }
            )
        return writer, allocator

    def load(self, session_id: str) -> SessionState | None:
        """Read a session record, or `None` if no readable record exists.

        A corrupt or unparseable file reads as absent rather than raising. That is a
        deliberate and narrow exception: the alternative is an agent that cannot start
        because a cache file is truncated, and the value returned is `None` — the honest
        "no session here" — not a fabricated empty session presented as a real one, which
        is what P6 forbids. The corruption is logged.

        **Absent to a reader, never to a writer (#1844).** The commonest unreadable record
        is not a damaged one: it is a conversation a newer build wrote, with a field this
        build refuses under `extra="forbid"`. `load` leaves it where it is, so a newer
        build started again still finds it; `save` and `delete` move it to
        `<name>.unreadable-<UTC time>` before they would replace or remove it
        (`uclone_x.core.set_aside`).

        The `is_file()` probe is inside the `try` for the same reason. An ID long enough
        to exceed the filesystem's name limit makes `is_file()` itself raise
        `OSError(ENAMETOOLONG)`, which used to escape past this documented `None` — the
        docstring promised "no readable record" and the method raised instead. No record
        can exist at such a path, so absent is the correct answer.

        Validation goes through `model_validate_json`, not `model_validate` on a parsed
        `dict`. These models are `strict=True`, under which Python-mode validation
        requires a `MessageRole` *instance* and rejects the `"system"` string that
        `model_dump(mode="json")` writes — so the dict route cannot read back the records
        this store itself produces. Pydantic's JSON mode accepts the JSON
        representation of an enum, which is exactly what is on disk.

        **A record belonging to a different session raises rather than reading as absent
        (#256).** It is the one failure here that is not "no session at this id" — the
        file is present, parses, and holds somebody else's conversation, because the
        filesystem folded two distinct ids onto one name. Returning `None` for it was
        rejected twice over: it substitutes an empty result for a condition that is an
        error (P6), and it defers the refusal to the worst possible moment. The caller
        would take `None` as "new session", conduct a whole conversation, and only find
        out at the closing `save` — which is where the fold is unavoidable — that the id
        was never usable. Raising at the first touch spends no turns. See
        `verify_record_identity` for why the check compares record contents rather than
        encoding the filename.

        Raises:
            PathTraversalError: See `resolve_session_path`.
            SessionIdCollisionError: If a readable record exists at this id's path but
                identifies a different session.
        """
        path = self.session_path(session_id)
        state, cause = self._read(session_id, path)
        if cause is not None:
            # Left where it is: a read changes nothing on disk. `save` and `delete` move it
            # aside before they would replace or remove it (#1844).
            logger.warning(
                "Session record at %s cannot be read (%s); treating as absent", path, cause
            )
        return state

    def _read(self, session_id: str, path: Path) -> tuple[SessionState | None, str | None]:
        """The record at `path`, and why it could not be read, for the log, when it could not.

        `(None, None)` when there is no record, including a path too long to exist.
        `(None, cause)` when a file is there and does not load: undecodable, unopenable, or
        a shape this build refuses -- most often a field a newer build added, or a form
        that holds text, as a record written before forms were recorded does (#1848). No
        body is read: a form is judged by what it records, so a body the store cannot
        read never fails the read of the record (#1974).

        Raises:
            SessionIdCollisionError: The record loads and identifies a different session.
        """
        try:
            if not path.is_file():
                return None, None
        except OSError:
            return None, None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return None, f"{type(exc).__name__}: {exc}"
        try:
            state = SessionState.model_validate_json(text)
        except ValueError as exc:
            return None, f"it does not match SessionState: {exc}"
        # After parsing, not before: an unparseable file has no id to compare, and it is
        # reported unreadable above for a reason that has nothing to do with #256.
        verify_record_identity(session_id, state.session_id, path)
        # A record an earlier build wrote, whose history is not named by log entry, is
        # refused like any shape this build does not write (#1848, #1974 item 5).
        old = recorded_before_history_entries(state)
        if old is not None:
            return None, old
        if state.history_entries and not state.messages:
            reconstructed: list[ChatMessage] = []
            for entry_id in state.history_entries:
                index = log_position(entry_id)
                if index is None or index >= len(state.session_log):
                    return None, f"history_entries names {entry_id!r}, not in session_log"
                digest = state.session_log[index].digest
                body = self.load_context_body(session_id, digest)
                if body is None:
                    return None, f"no context body {digest} for log entry {entry_id}"
                try:
                    message = ChatMessage.model_validate_json(body)
                except ValueError as exc:
                    return None, f"context body {digest} could not be read: {exc}"
                # Image bytes are bodies of their own, named by digest (#2107); one that
                # is gone leaves its image unavailable and does not fail the record.
                reconstructed.append(
                    with_image_data(message, partial(self.load_context_body, session_id))
                )
            state = state.model_copy(update={"messages": tuple(reconstructed)})
        # A record whose message is not the entry it names would be refused on every
        # resume and never set aside; it is refused here instead, like the rest (#1985).
        mismatch = history_entries_not_their_messages(state)
        if mismatch is not None:
            return None, mismatch
        return state, None

    def _set_aside_unreadable(self, path: Path, cause: str) -> None:
        """Move the unreadable record at `path` aside, or refuse whatever was about to replace it.

        Raises:
            SessionRecordUnreadableError: The rename failed; the record is still at `path`.
        """
        try:
            aside = set_aside_unreadable(path)
        except OSError as exc:
            logger.warning(
                "Session record at %s cannot be read (%s) and could not be moved aside (%s); "
                "nothing is written over it.",
                path,
                cause,
                exc,
            )
            refusal = SessionRecordUnreadableError(
                "This conversation's saved record could not be opened by this version and "
                "could not be kept aside, so nothing was saved over it.",
                path=path,
                cause=cause,
            )
            raise refusal from exc
        logger.warning(
            "Session record at %s cannot be read (%s); it was moved aside, unchanged, to %s",
            path,
            cause,
            aside,
        )

    def _expire_kept_copies(self) -> None:
        """Delete expired copies of deleted records, then what the last of them referred to.

        A delete keeps a record's event log, context bodies and tool results while a copy of
        it is kept (#1860), and a deleted record is never set aside again, so nothing else
        would ever remove them. `expire_set_aside` deletes a copy 30 days after it was set
        aside once its record is gone (#1877); when no copy of that id is left and no record
        has been saved under it since, this removes what `delete` would have.

        The record and the copies are looked at again just before the history goes. A
        session saved under the same id between that look and the removal loses its event
        log and context bodies -- the window `delete` itself has. Never raises: this runs
        when a store opens, and a store that cannot tidy up still opens.
        """
        for record in expire_set_aside(self._storage_dir, suffix=".json"):
            session_id = record.name.removesuffix(".json")
            try:
                if self.session_path(session_id) != record:
                    continue
                if record.is_file() or kept_copies(record):
                    continue
                self.clear_event_log(session_id)  # the last copy is gone
            except Exception as exc:
                logger.warning(
                    "Could not remove the history of %r after its last kept copy expired (%s)",
                    session_id,
                    exc,
                )
                continue
            logger.info(
                "Removed the event log and tool results of %r: its record is gone and its "
                "last kept copy expired",
                session_id,
            )

    @staticmethod
    def _has_kept_copy(path: Path) -> bool:
        """Whether a copy of the record at `path` is set aside; `True` when that cannot be told.

        Answering `True` when the directory cannot be listed keeps what a copy might need;
        answering `False` would delete it on a guess. An empty copy is not one (#1877).
        """
        try:
            return bool(kept_copies(path))
        except OSError:
            return True  # cannot tell, so keep what a copy may need

    def keeps_copy_of(self, session_id: str) -> bool:
        """Whether a copy of `session_id`'s record is set aside; `True` when that cannot be told.

        A reset asks before it clears the session's history (#1921): a kept copy names the
        same event log, context bodies and tool artifacts, as `delete` knows.
        """
        return self._has_kept_copy(self.session_path(session_id))

    def take_set_aside(self, session_id: str) -> bool:
        """Whether `save` moved this session's unreadable record aside since last asked; once each."""
        if session_id not in self._set_aside:
            return False
        self._set_aside.discard(session_id)
        return True

    def save(
        self, state: SessionState, pending_events: Sequence[Any] | None = None
    ) -> SessionState:
        """Atomically persist `state` and return exactly what `load` will return.

        **Invariant: `save(state)` returns the same value a subsequent `load` yields.**
        Two halves, and an earlier version of this docstring held only the first:

        * The record on disk round-trips. The serialized payload is validated *before* it
          reaches the disk, so a state carrying a value that never passed validation is
          refused here rather than written as a record `load` would report as absent. A
          write that returns successfully followed by a read that says "no session here"
          is the silent-wipe shape, and strictly worse than a refusal, because the caller
          has no reason to look again.
        * **The returned object round-trips too.** This is the half that was claimed and
          not held. `save` used to return `stamped` — the *unvalidated* object it was
          handed. So `model_copy(update={"messages": [{"role": "system", ...}]})` wrote a
          correct record, `load` returned a correct state, and `save` still handed back a
          `SessionState` whose `messages` were raw `dict`s where `ChatMessage` was
          declared. No data was lost and the disk was right; a caller holding the return
          value simply got something `load` would never produce. Returning the validated
          payload closes it, and makes the return value a value the caller can rely on
          rather than one that merely happens to be correct on the typed paths.

        This revalidation is the backstop for the first door in the family described
        under "Validation is a property of the constructor, not of the type" in this
        module's docstring; `model_copy(update=...)` stays reachable by any caller, so
        the boundary refuses what the constructor never saw.

        **Isolation: a compare-and-swap on `revision`, not last-write-wins (#219).**
        Atomicity was never the missing half. `os.replace` is still the only route to the
        destination, so a crash leaves the previous record whole rather than a half-written
        one — but that says nothing about two writers, and until #219 there was no version
        field and no precondition, so two writers that both `load`, both modify and both
        `save` produced one surviving record and one **silently** discarded update.

        The gap was live rather than theoretical, which is what makes the precondition
        worth its cost: the CLI and the UI share one Core record per session by design
        (P8's single store) and both persist every turn, so `ucx run` and `ucx ui` on one
        session id is an ordinary configuration. On the compaction path it was worse than
        a lost turn. Measured: a writer compacted 25 messages to 6 and committed, a second
        writer holding the pre-compaction read persisted its 25 back, and the result was
        6 in one process's memory, 25 on disk, and the compaction ledger present in
        neither — no participant holding the truth, and no exception anywhere.

        So `save` now reads the record it is about to overwrite and requires
        `state.revision` to match what is on disk. On success the record is written at
        `disk + 1` and that incremented state is what comes back, so a caller that keeps
        the return value stays able to write again. On mismatch nothing is written and
        `StaleSessionWriteError` carries the on-disk state for the caller to rebase onto.

        Three consequences worth stating rather than discovering:

        * **No snapshot door advances `revision`, and that is the whole of the claim.**
          `with_messages` and `reset` carry it through unchanged and expose no parameter
          for it, so a state built from a read still claims the revision it read and a
          stale write is recognisably stale. That is the property the fix rests on, and it
          is the only one asserted.

          What it is **not** is tamper-proof, stated here because the wider reading is
          false and was published as true once. `SessionState(..., revision=n)` and
          `state.model_copy(update={"revision": n})` both produce a state carrying a
          revision no read ever returned, and `save` **accepts** it when `n` happens to
          equal the revision on disk — silently destroying the committed update the
          precondition exists to protect. The pre-write `model_validate_json` backstop
          cannot see it, because a forged revision is a perfectly valid `int`. So what the
          store carries is in effect a `force=True`, which is the option #219's fix
          explicitly rejected. It was undocumented; it is documented here rather than left
          implied, and **#248 decided to keep it** — see the decision below, which is where
          the alternatives and their costs are.

          Bounded, and the bound is what the decision below rests on, so it is *enforced*
          rather than described. There is exactly one, and it is a static sweep of
          `src/uclone_x` run as a test — not a runtime control, and not a check any
          caller passes through:

          - **No module outside this one names a revision value.** Every `SessionState`
            built under `src/` either omits `revision` or forwards one it was handed;
            `test_no_module_under_src_names_a_revision_value` sweeps for the two ways
            source can name the field — a `revision=` keyword, and a `"revision"` dict
            key — rather than for a list of callees, and in **both** cases a forward
            (`<expr>.revision`) is permitted while a literal or an expression is not,
            since `revision=state.revision` and `{"revision": state.revision}` are one
            act written two ways. That is deliberate: an earlier version of the sweep watched
            `SessionState(...)`, `model_copy` and `__setattr__`, and a reviewer forged a
            revision straight past it with `SessionState.model_validate({**raw,
            "revision": 2})`. A callee allow-list is incomplete by construction and reads
            as if it were not, which is this very card's defect shape. An earlier wording
            here said "no code under `src/` constructs or `model_copy`s a `SessionState`
            outside this module", which is simply **false**: `agent.base` does, and must,
            because that is where a live working copy becomes a record. What is true is
            that it *forwards* the revision it is holding, which is only ever written from
            a value a `SessionStore` produced. The false sentence is exactly why the bound
            is now a sweep — an unenforced bound is how this card was produced. The
            sweep's own two name matchers are asserted live for the same reason: a matcher
            that has stopped matching reports a clean tree, and both were shown to do
            exactly that before the assertions were added.

          There was a second bound here, and it is struck rather than deleted because
          deleting it is how the wrong version stays believable. It read: ~~"only an
          **exact** guess is accepted, and an exact guess requires knowing the revision on
          disk, which requires reading the record — so there is no route to it that a
          legitimate rebase does not also take."~~ The first clause is true and buys
          nothing; the conclusion drawn from it is **false, by measurement, twice over**:

          - A stale writer that never re-reads lands the forge with arithmetic. Holding a
            state read at revision 1 while disk has moved to 2,
            `stale.model_copy(update={"revision": stale.revision + 1})` is **accepted**
            and the winner's turn is gone. Its only read is the stale one it already had,
            so it never takes the rebase route it is supposed to be indistinguishable
            from.
          - A caller that has **never read the record** reaches an exact guess by trying.
            Against a record at revision 2, guesses of 1 and 3 raise
            `StaleSessionWriteError` and 2 is **accepted**. `revision` is a small
            monotonic integer, so "knowing the revision on disk" is not a cost.

          What would the code have to do for that sentence to be true? **Two changes, not
          one**, and this change measured both and adopted neither. The conclusion fails on
          a route that never enters `save` at all: a caller that has read **nothing**,
          writing the record path directly with the default `revision=0` — a value no read
          returned and which is *wrong*, since disk is at 2 — destroys the winner's
          committed turn, raises nothing, and leaves a record `load` accepts at revision 0.
          **No control placed inside `save` reaches that, because the record is a plain
          file**, and `test_the_revision_precondition_is_a_protocol_not_a_boundary` asserts
          it.

          Closing *that* route means making `save` the only way to reach a loadable record
          — sealing the payload on write and having **`load`** refuse a record the store did
          not seal. That is a real design and not a hypothetical: measured here as a
          throwaway subclass, it refuses the direct write above while the stock store adopts
          it. Note where it lives, because it is the reason the narrower sentence above
          survives — the seal is enforced in `load`, so it is not a control inside `save`.
          And sealing alone is still not enough: an exact guess would go on being accepted
          *through* `save`, so the first clause additionally needs the store-remembers
          closure rejected below.

          So the sentence is two repairs from true, and this change adopted neither. Only
          one of them was ever #248's to decline: the closure is rejected below, on a cost
          measured for it. **Sealing is not among the card's four options** — it surfaced
          later, as a mutation of this change, and is named here because it shows the route
          is closable, not because it was weighed. That is a claim about **cost**, not about
          what is possible, and the
          distinction is load-bearing: an earlier wording here said ~~"nothing available
          would do it"~~ and ~~"a property no reachable design can deliver"~~, which
          quantifies over every design that could exist, pins nothing, and is contradicted
          by the sealed store above. The cause is unchanged — the sentence claimed more than
          the design bought, which is an overclaim — and so is the capability #248 accepted.

          What changes is the decision's stated evidence. **It now rests on the single bound
          above, plus the argument that follows** — which is the same observation arriving
          as a premise rather than as a footnote: `save` is not a boundary. The sweep is
          therefore load-bearing alone, where the text above split the weight over two.

          **Decision on #248: accept the capability, and enforce the bound — option 1 of
          the four the card listed.** Recorded here rather than in a separate document for
          the reason the filename-encoding decision above is: the option that was chosen
          reads as the absence of a decision unless the rejected ones sit beside it.

          What settles it is that `save` is **not a boundary**. It is one door onto a file
          any in-process caller can open, and writing that file directly destroys the
          winner's committed turn with no exception raised — the same damage the forge
          does, without needing to guess anything.
          `test_the_revision_precondition_is_a_protocol_not_a_boundary` pins it. So
          `revision` is a cooperation protocol between honest writers, and each option is
          a change to that protocol rather than a security control.

          * **2 — a declared `force=True` parameter.** Rejected. It does not close the
            forge: an exact revision is still accepted under `force=False`, so the
            undeclared route survives *beside* the declared one. What it adds is a
            sanctioned way to discard another writer's committed turn, in a tree with no
            caller that wants one. Its stated benefit is auditability, and the sweep above
            buys strictly more of it — a forge site cannot land at all, where a
            `force=True` site would land and merely be greppable.
          * **3 — move the token out of the model.** Rejected on two costs. It breaks the
            `save(state) == load(...)` invariant #210 pinned, by putting a required
            argument outside the state; and it buys no unforgeability, since a handle a
            second process can present is one that process can also compute. A handle kept
            in the store object instead is the closure measured below.
          * **4 — wait for the constructor-door family to close centrally.** Rejected on a
            measurement of its *premise*, not on scope. The tree already carries that
            closure: `ontology.models`, `skills.models` and `engine.event_bus` override
            `model_copy` so `update=` is routed through `model_validate` (#31/#42). Give
            `SessionState` the same override and `model_copy(update={"revision": "2"})`
            raises — while `model_copy(update={"revision": 2})` stays accepted, because a
            forged revision is a valid `int` and validation is the whole of what that
            family adds. #248 shares the family's shape and not its cause: those doors are
            about a value that never passed validation, this one about a valid value the
            caller should not be choosing.
            `test_the_constructor_door_family_closure_would_not_close_this_one` pins that,
            so nobody spends the central fix expecting this to arrive with it.
          * **1 — accept and document. Chosen**, with the surviving bound moved out of
            prose and into the sweep.

          The closure that does work is rejected with the others, and by execution: have
          the store remember the revision it handed out per session id and compare against
          *that*. Measured on `3c23e5f` — it closes the hole, and takes **nine** tests with
          it, because it also refuses every writer that did not obtain its state from that
          same `SessionStore` **object**. Eight of the nine are legitimate writers; the
          ninth is the #248 pin, which is that pin's documented signal.
          `test_two_genuine_os_processes_on_one_session_cannot_lose_an_update` is among
          the eight, and it is the shipped configuration: two real OS processes, neither
          able to see the other's store. (#240's write-up recorded seven, measured against
          its own spelling of the closure; the count moves with the spelling, the
          two-process failure does not.)
        * **An absent record accepts any revision.** There is no update to lose when the
          file is gone, and refusing would make a session unsavable after a legitimate
          `delete`. A record that exists but does not load holds no revision to compare
          either, and is **moved aside, not written over** (#1844): it is most often a
          conversation a newer build wrote, and this write would otherwise destroy it. The
          move happens once the new record has validated, so a write refused for any
          other reason leaves it where it was. `take_set_aside` then reports it once.
        * **This is a precondition, not a lock.** It closes the read-modify-write window
          that spans a turn, which is the one this defect actually arrives through, and
          leaves a sub-millisecond window in which two writers both pass the check and
          then both `os.replace`. Closing that needs mutual exclusion, which was rejected
          deliberately: an `O_EXCL` lockfile reproduces the crash-orphan shape of #219's
          defect 3 as an availability failure, and `fcntl.flock` — which is stale-proof,
          since the kernel drops the lock when the fd dies — has no Windows equivalent, so
          it would ship a platform branch that cannot be exercised on the machine P8
          confines verification to. A documented window is honest; an unexercised branch
          is not. `test_the_precondition_is_a_check_not_a_lock` pins the limit so this
          docstring cannot drift into claiming serialisability.

        **Identity before revision, and the order is the requirement (#256).** The
        pre-write `load` below raises `SessionIdCollisionError` when the record occupying
        this id's path identifies a different session, and it does so *before* the revision
        comparison. That ordering is what a genuinely new session depends on. Measured on
        `e8b3e2f`: saving a fresh `"sessa"` beside an existing `"SessA"` was refused with
        `StaleSessionWriteError` — *"it was read at revision 0 and the record is now at
        revision 1, so this write would discard another writer's update"* — naming a writer
        and a revision for a session the user had never heard of, when the actual problem
        was that the two ids are one filename here. The refusal was accidental (a side
        effect of #240) and it described the wrong defect. Now the id collision is reported
        as an id collision and only a real revision mismatch reports one.

        Raises:
            ValueError: If `state` cannot be serialized to a record `load` can read.
            PathTraversalError: See `resolve_session_path`.
            SessionIdCollisionError: If the record on disk at this id's path identifies a
                different session — checked before the revision precondition, so a new
                session never reports a phantom revision conflict.
            StaleSessionWriteError: If the record on disk is at a different revision than
                the one `state` was read at.
            SessionEventLogNotConfiguredError: If `pending_events` is non-empty and the
                store was built with only one of `log_writer` and `log_allocator`, so it
                has nowhere to write them (#1442).
            SessionRecordUnreadableError: If the record at this id's path cannot be read
                and could not be moved aside. Nothing is written over it (#1844).
        """
        path = self.session_path(state.session_id)
        # Before anything is read or written, for the same reason as the revision check
        # below: a refusal must leave no trace. The events stay in the caller's queue.
        if (
            pending_events
            and self._event_log_dir is None
            and (self._log_writer is None or self._log_allocator is None)
        ):
            raise SessionEventLogNotConfiguredError(
                f"Refusing to persist session {state.session_id!r}: {len(pending_events)} "
                f"turn event(s) were handed to a store with no log writer or no offset "
                f"allocator, and writing the record without them would drop them "
                f"silently. Build the store with both, or with neither to use the "
                f"default per-session event log."
            )
        # Read-then-check *before* serializing: a refused write must leave no trace, and
        # the cheapest way to guarantee that is to decide before anything is built. A
        # record that is there and will not load holds no revision to compare, so this
        # write replaces nothing of it: it is moved aside below, once the write is known to
        # be valid, and never written over (#1844).
        on_disk, unreadable = self._read(state.session_id, path)
        if on_disk is not None and on_disk.revision != state.revision:
            raise StaleSessionWriteError(
                f"Refusing to persist session {state.session_id!r}: it was read at "
                f"revision {state.revision} and the record is now at revision "
                f"{on_disk.revision}, so this write would discard another writer's "
                f"update. Re-apply this change onto the record carried on the "
                f"exception's `current` attribute and save again; "
                f"`BaseAgent.hydrate_session` does that wholesale when adopting the "
                f"record as-is is acceptable.",
                session_id=state.session_id,
                expected_revision=state.revision,
                actual_revision=on_disk.revision,
                current=on_disk,
            )
        next_revision = 0 if on_disk is None else on_disk.revision
        stamped = state.model_copy(
            update={
                "updated_at": _now_iso(),
                "revision": next_revision + 1,
            }
        )
        payload = json.dumps(stamped.model_dump(mode="json"), indent=2)
        try:
            validated = SessionState.model_validate_json(payload)
        except ValueError as exc:
            raise ValueError(
                f"Refusing to persist session {state.session_id!r}: the serialized record "
                f"does not validate as SessionState, so `load` would report it as absent. "
                f"This usually means a field was set through `model_copy(update=...)`, "
                f"which does not validate. Underlying error: {exc}"
            ) from exc
        clean_payload = json.dumps(validated.model_dump(mode="json"), indent=2)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        if unreadable is not None:
            self._set_aside_unreadable(path, unreadable)
            self._set_aside.add(state.session_id)
        offsets: list[LogOffset] = []
        event_log = self._event_log_for(state.session_id) if pending_events else None
        if pending_events and event_log is not None:
            log_writer, log_allocator = event_log
            cursor = log_allocator.next_offset(state.session_id)
            try:
                for event in pending_events:
                    # The offset goes into the line, so the log can be read back in order
                    # without the allocator.
                    entry: Mapping[str, Any] | str = (
                        {**cast("Mapping[str, Any]", event), "offset": int(cursor)}
                        if isinstance(event, Mapping)
                        else str(event)
                    )
                    log_writer.write_entry(entry)
                    offsets.append(cursor)
                    cursor = LogOffset(cursor + 1)
            except BaseException:
                # Lines that reached the log keep their offsets, so a retry of this save
                # allocates past them instead of writing the same offsets a second time.
                # The retry still writes those events again, under new offsets.
                written = tuple(offsets)
                if written:
                    log_allocator.record_appended(state.session_id, written)
                raise
            # Before the record is committed, not after: a cursor that cannot advance
            # fails the save while the record is still at its previous revision, instead
            # of raising over a record that already moved on.
            log_allocator.record_appended(state.session_id, tuple(offsets))

        # Temp-and-replace, not a direct write. `os.replace` is atomic within a
        # filesystem, so a crash at any point leaves the *previous* record intact rather
        # than a half-written one; a direct write to `path` would truncate the only copy
        # before the new bytes landed. Atomicity only — the concurrent writer is handled
        # by the revision precondition above, not here, and the two are separate
        # properties: atomicity is about a crash mid-write, the precondition is about
        # another writer having committed since this one read.
        #
        # Durability decision on `fsync`-before-`replace` (#219, #257): Adopted `flush()`
        # and `os.fsync()` on the temp file descriptor before `os.replace`. The latency
        # cost (~1ms per turn) is negligible, guaranteeing that replaced records are
        # physically committed to non-volatile storage and immune to crash truncation.
        #
        # The temp name carries the pid and a random suffix so concurrent writers to one
        # session ID cannot splice each other's bytes, and deliberately not
        # `asyncio.get_running_loop().time()`, which raises outside a running loop and
        # would make this synchronous method unusable from the CLI.
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(clean_payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        # The validated payload, not `stamped`: what a caller receives is what `load`
        # will hand back, byte for byte and type for type.
        if stamped.history_entries and not validated.messages:
            validated = validated.model_copy(update={"messages": stamped.messages})
        return validated

    def reap_orphaned_temp_files(self, max_age_seconds: float = 300.0) -> int:
        """Reap crash-orphaned session temp files under this store's storage directory (#219, #257)."""
        return reap_orphaned_temp_files(self._storage_dir, max_age_seconds=max_age_seconds)

    def delete(self, session_id: str) -> bool:
        """Remove a session record. Returns whether a record was actually removed.

        `False` covers "no such record", including an ID too long for the filesystem to
        hold, whose `is_file()` probe raises `OSError(ENAMETOOLONG)` rather than
        answering. A path that cannot exist holds no record to remove. A failure of the
        `unlink` itself is *not* swallowed: that is a record that exists and could not be
        deleted, which the caller must hear about.

        **A record belonging to a different session is refused, not removed (#256).** This
        is the most destructive of the variant-id doors and it was measured, not inferred:
        on `e8b3e2f`, `delete("SESSA")` returned `True` having unlinked `SessA.json`, so a
        caller could annihilate a conversation it had never named and be told it succeeded.
        `#256`'s table does not list this door — it was found by enumerating the store's
        surface rather than the card's cases. Routed through the same reader as `load` so
        the ownership rule has one implementation.

        **An unreadable record is moved aside, not unlinked (#1844).** It holds no id to
        compare, and refusing would make the session impossible to clear; but the person
        clearing it is clearing a conversation this build showed as empty, and the record
        is most often one a newer build wrote. So its name is freed and its bytes are kept.

        **What a kept copy refers to is kept with it (#1860).** A record's event log and
        context bodies -- among them the full text of its tool results (#1848) -- are found
        by session id, not by file name, so a copy set aside -- by this delete or by an
        earlier save -- names the same ones. Removing them would leave a copy that,
        restored by hand, has lost its tool results. So while any copy of this id is
        kept, the delete removes the record and nothing it refers to. Kept copies are
        bounded (`uclone_x.core.set_aside.KEEP_SET_ASIDE`), so this is too.

        Raises:
            PathTraversalError: See `resolve_session_path`.
            SessionIdCollisionError: If the record at this id's path identifies a different
                session. Nothing is unlinked.
            SessionRecordUnreadableError: If the record cannot be read and could not be
                moved aside. Nothing is removed.
        """
        path = self.session_path(session_id)
        # Ownership before destruction. `_read` raises on a foreign record and returns no
        # state for an absent or unreadable one, so this is the whole check.
        _, unreadable = self._read(session_id, path)
        try:
            if not path.is_file():
                return False
        except OSError:
            logger.warning("Cannot probe session record for %r; treating as absent", session_id)
            return False
        if unreadable is not None:
            # Most likely a conversation a newer build wrote: kept, not unlinked (#1844).
            self._set_aside_unreadable(path, cause=unreadable)
        else:
            path.unlink(missing_ok=True)
        if self._has_kept_copy(path):
            # A kept copy names this id's event log and context bodies, tool results
            # among them; they stay while it does (#1860).
            logger.info(
                "Session %r deleted; its event log and tool results are kept for the copy "
                "set aside beside %s",
                session_id,
                path,
            )
            return True
        # The event log is this session's history too; deleting the record and keeping
        # every tool output it produced would make "delete" a claim the disk contradicts.
        # The full tool results are context bodies, so they go with it (#1848).
        self.clear_event_log(session_id)
        return True

    def list_session_ids(self) -> tuple[str, ...]:
        """Session IDs that `load` will actually return a record for, in sorted order.

        **Read out of each record, not off its filename** — and the correction that made
        this so is worth stating, because the first version of this method was the one
        mistake this whole change exists to reject. It returned `p.stem`, i.e. it treated
        the *filename* as the authority on a record's identity, while every read door
        below and above it verifies the `session_id` **inside** the record. Enumerating by
        stem and loading by content is not a small inconsistency: it makes the two
        disagree in exactly the scenario this fix's own reasoning invokes.

        The case, measured by a reviewer on #263: a variant pair created on
        ext4 and carried onto APFS collapses to one file, and the survivor can perfectly
        well be named `SessA.json` while holding **`sessa`**'s record. Enumerating by stem
        then reported `("SessA",)` — the one id `load` **refuses** — while hiding `sessa`,
        the one id `load` **accepts**. Precisely inverted, and reachable in three lines on
        plain APFS through any rename, restore, or archive extraction, so it is not an
        exotic-filesystem concern.

        So the rule here is the same rule as everywhere else: a record's id is what the
        record says it is. `samefile` is what makes it exact rather than approximately
        right — it compares device and inode, so it answers "would `load(id)` open *this*
        file?" without needing to model how a given filesystem folds names. `Path.resolve`
        cannot substitute: it does not canonicalise case on macOS, so two paths naming one
        file resolve to two different strings.

        **The property this gives, stated as narrowly as it actually holds:** every id
        returned here is an id `load` returns a record for, and every record `load` can
        reach is represented by exactly one id. A file whose record names an id that
        resolves to a *different* file is skipped rather than reported — it is
        unreachable through `load`, so listing it would re-create the same disagreement in
        the other direction. Unparseable and unreadable records are skipped for the same
        reason: `load` reports them absent, and an enumeration that disagreed with `load`
        about which sessions exist is the defect, not the feature.

        **Every skip is logged**, which it was not at first. `load` warns
        `"…treating as absent"` for the same class of condition two methods above, and this
        method had **zero** logger calls — so a record dropped from the listing left no
        trace anywhere. Nothing is dropped from *disk* on this path, so it is not a
        correctness bug; it is the difference between a diagnosable omission and a silent
        one, and the inconsistency with `load` is the argument on its own.

        This is the answer to #256's criterion that `list_session_ids()` report the real
        ids. The alternative on the table was to narrow the docstring's claim to records
        the store itself wrote; that would have left the criterion unmet in the very case
        the reviewer measured, so it was rejected in favour of making the two agree.

        Cost, named rather than hidden: one `read_text` plus one id validation plus one
        `stat` per record, instead of a bare `glob`. Records are small, this is not on the
        turn path, and the alternative is an enumeration that lies.

        An earlier version of this paragraph added that hashing would have forced the same
        cost anyway. That was too generous to this option and it is corrected here:
        a reviewer measured that a hashed scheme reads every record too but
        needs **no** `resolve` + `samefile` round-trip, because a hex digest cannot fold —
        one read plus a `sha256`, against one read plus a validation plus a `stat`. So on
        enumeration cost alone the column tips marginally **toward** hashing. It changes
        nothing, because enumeration cost was never the ground hashing was rejected on;
        see the module docstring for the ground that was.
        """
        if not self._storage_dir.is_dir():
            return ()
        ids: set[str] = set()
        for path in self._storage_dir.glob("*.json"):
            try:
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                logger.warning("Unreadable session record at %s; omitting from listing", path)
                continue
            try:
                state = SessionState.model_validate_json(text)
            except ValueError:
                logger.warning(
                    "Session record at %s does not match SessionState; omitting from listing",
                    path,
                )
                continue
            try:
                # The id must round-trip: resolving it has to land back on *this* file.
                # An id that is illegal by name (a hand-edited record) cannot be loaded at
                # all, so `PathTraversalError` is a skip rather than a failure.
                candidate = resolve_session_path(self._storage_dir, state.session_id)
                reachable = candidate.samefile(path)
            except (PathTraversalError, OSError):
                logger.warning(
                    "Session record at %s names id %r, which cannot be resolved back to it; "
                    "omitting from listing",
                    path,
                    state.session_id,
                )
                continue
            if not reachable:
                logger.warning(
                    "Session record at %s names id %r, which resolves to a different file; "
                    "omitting from listing because `load` cannot reach it",
                    path,
                    state.session_id,
                )
                continue
            ids.add(state.session_id)
        return tuple(sorted(ids))
