"""The log offset's three properties, each of which `AgentEvent.sequence` fails.

Written before the implementation. Each test names the property and the way the bus counter
does not have it, because the design that preceded #526 asserted the envelope was already
log-ready and was wrong on all three counts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from uclone_x.core.log_offset import FIRST_LOG_OFFSET, MAX_LOG_OFFSET, LogOffset
from uclone_x.errors import (
    LogOffsetContiguityError,
    LogOffsetCorruptCursorError,
    LogOffsetSessionMismatchError,
    PathTraversalError,
)
from uclone_x.log.file_allocator import FileLogOffsetAllocator


def test_a_fresh_session_starts_at_the_first_offset(tmp_path: Path) -> None:
    alloc = FileLogOffsetAllocator(root=tmp_path)
    assert alloc.next_offset("sess-a") == FIRST_LOG_OFFSET


def test_offsets_are_per_session_not_global(tmp_path: Path) -> None:
    """Two sessions do not share a sequence space.

    `EventBus._next_sequence` is one counter for the whole bus, so under it a session's
    second event could be numbered 47 because other sessions were busy. An offset is a
    position in *one* log.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    alloc.record_appended("sess-a", (FIRST_LOG_OFFSET,))
    alloc.record_appended("sess-a", (LogOffset(2),))

    assert alloc.next_offset("sess-a") == LogOffset(3)
    assert alloc.next_offset("sess-b") == FIRST_LOG_OFFSET, "sess-b saw sess-a's traffic"


def test_offsets_survive_a_restart(tmp_path: Path) -> None:
    """A second process continues rather than starting over.

    This is the property the bus counter most clearly lacks: it is constructed at zero, so
    offset 1 in run two collides with offset 1 in run one, and a log with two events numbered
    1 cannot be folded.
    """
    first = FileLogOffsetAllocator(root=tmp_path)
    first.record_appended("sess-a", (FIRST_LOG_OFFSET, LogOffset(2), LogOffset(3)))

    reopened = FileLogOffsetAllocator(root=tmp_path)
    assert reopened.next_offset("sess-a") == LogOffset(4)


def test_a_gap_is_refused(tmp_path: Path) -> None:
    """A hole in an append-only log is later indistinguishable from an event never written."""
    alloc = FileLogOffsetAllocator(root=tmp_path)
    with pytest.raises(LogOffsetContiguityError) as excinfo:
        alloc.validate_batch("sess-a", (LogOffset(2),), cursor=FIRST_LOG_OFFSET)

    message = str(excinfo.value)
    assert "sess-a" in message
    assert "1" in message, "the expected offset must be named, not just the wrong one"


def test_an_out_of_order_batch_is_refused(tmp_path: Path) -> None:
    """A batch whose *first* offset is wrong is refused by the start check."""
    alloc = FileLogOffsetAllocator(root=tmp_path)
    with pytest.raises(LogOffsetContiguityError):
        alloc.validate_batch("sess-a", (LogOffset(2), FIRST_LOG_OFFSET), cursor=FIRST_LOG_OFFSET)


@pytest.mark.parametrize(
    ("batch", "why"),
    [
        ((1, 3), "a gap inside the batch"),
        ((1, 2, 2), "a repeat inside the batch"),
        ((1, 3, 2), "a rewind inside the batch"),
    ],
)
def test_a_batch_that_starts_correctly_but_breaks_inside_is_refused(
    tmp_path: Path, batch: tuple[int, ...], why: str
) -> None:
    """Intra-batch contiguity, reached — which the out-of-order test above never gets to.

    That test passes `(2, 1)` against cursor 1, so the *start* check refuses it and the
    pairwise loop is never entered: delete the loop entirely and it still passes. Coverage
    confirmed the loop body was unreached by the whole suite. These cases all start at the
    expected offset, so the start check lets them through and only the loop can refuse them.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    offsets = tuple(LogOffset(n) for n in batch)
    with pytest.raises(LogOffsetContiguityError) as excinfo:
        alloc.validate_batch("sess-a", offsets, cursor=FIRST_LOG_OFFSET)
    assert "jumps from" in str(excinfo.value), f"{why} must be refused by the pairwise check"


def test_an_empty_batch_is_accepted_and_moves_nothing(tmp_path: Path) -> None:
    """Appending nothing is not an error, and must not advance the cursor."""
    alloc = FileLogOffsetAllocator(root=tmp_path)
    alloc.record_appended("sess-a", (FIRST_LOG_OFFSET,))

    alloc.validate_batch("sess-a", (), cursor=LogOffset(2))
    alloc.record_appended("sess-a", ())

    assert alloc.next_offset("sess-a") == LogOffset(2)


def test_record_appended_refuses_a_batch_for_the_wrong_session(tmp_path: Path) -> None:
    """Identity is refused on the *write* path too, not only on `validate_batch`.

    Only `validate_batch` was covered before, and `record_appended` is the method that
    persists. This pins the write path's *behaviour*, deliberately not a particular internal
    check: an earlier version of this test could not tell the two apart, because
    `record_appended` did its own identity check and then called `validate_batch`, which did
    it again. Deleting either one left this passing. The duplicate read is gone and both
    paths now share `_check_batch`, so there is one check to delete and this test sees it.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    alloc.record_appended("sess-a", (FIRST_LOG_OFFSET,))
    (tmp_path / "sess-b.cursor").write_text(
        (tmp_path / "sess-a.cursor").read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(LogOffsetSessionMismatchError):
        alloc.record_appended("sess-b", (LogOffset(2),))


def test_a_wrong_session_is_refused_even_when_the_numbers_line_up(tmp_path: Path) -> None:
    """Identity is compared, never inferred from a cursor agreeing.

    The first version of this test used an empty `sess-b` with a mismatched cursor, so it
    passed on the *contiguity* check and never exercised identity at all. With the numbers
    aligned, the earlier implementation accepted a batch for the wrong session silently —
    which is the failure the storage format now prevents by holding the session id.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    alloc.record_appended("sess-a", (FIRST_LOG_OFFSET,))

    # Same directory, same cursor value; only the identity differs.
    (tmp_path / "sess-b.cursor").write_text(
        (tmp_path / "sess-a.cursor").read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(LogOffsetSessionMismatchError) as excinfo:
        alloc.validate_batch("sess-b", (LogOffset(2),), cursor=LogOffset(2))
    assert "sess-a" in str(excinfo.value), "the error must name the session that owns the log"


def test_a_stale_cursor_for_the_right_session_is_a_contiguity_error(tmp_path: Path) -> None:
    """A caller that names the correct session must not be told it named the wrong one.

    The earlier implementation compared cursors and called the result an identity mismatch,
    so this case reported that `sess-a`'s cursor "does not name session 'sess-a''s log" — a
    message that is false on its face.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    alloc.record_appended("sess-a", (FIRST_LOG_OFFSET, LogOffset(2), LogOffset(3)))

    with pytest.raises(LogOffsetContiguityError):
        alloc.validate_batch("sess-a", (LogOffset(2),), cursor=LogOffset(2))


@pytest.mark.parametrize(
    ("offsets", "why"),
    [
        ((), "an empty batch reaches no other check"),
        ((LogOffset(4),), "a batch that starts correctly reaches no other check"),
    ],
)
def test_a_stale_cursor_is_refused_by_the_cursor_check_alone(
    tmp_path: Path, offsets: tuple[LogOffset, ...], why: str
) -> None:
    """The `cursor != expected` arm, reached — which the test above never gets to.

    That test passes `offsets=(2,)` against a record at 3, so the *start* check refuses it and
    the cursor arm is never evaluated: delete the arm entirely and it still passes. It is the
    same shape as the `(2, 1)` batch, in a test written a round earlier and not re-examined.

    These two inputs cannot be refused by anything else. The empty batch returns before the
    start check, and `(4,)` satisfies it — so only a stale `cursor` can refuse either, which is
    the check the module docstring leans on for detecting a coarse interleaving of two writers.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    alloc.record_appended("sess-a", (FIRST_LOG_OFFSET, LogOffset(2), LogOffset(3)))

    with pytest.raises(LogOffsetContiguityError) as excinfo:
        alloc.validate_batch("sess-a", offsets, cursor=LogOffset(2))
    assert "stale" in str(excinfo.value), why


def test_next_offset_refuses_a_log_that_belongs_to_another_session(tmp_path: Path) -> None:
    """`next_offset`'s own identity check, which nothing exercised.

    A reported mutation table in this PR claimed deleting this check failed two tests. That
    was false — measured again, the whole suite passed, and no test outside this file even
    calls `next_offset`. The claim came from reading an aggregate pass/fail line across a batch
    of mutations instead of confirming each one applied to clean code, which is how a number
    that looks like evidence gets published.

    Without the check, `next_offset` hands out a position in a log the caller does not own, and
    the caller then writes there.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    alloc.record_appended("sess-a", (FIRST_LOG_OFFSET,))
    (tmp_path / "sess-b.cursor").write_text(
        (tmp_path / "sess-a.cursor").read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(LogOffsetSessionMismatchError) as excinfo:
        alloc.next_offset("sess-b")
    assert "sess-a" in str(excinfo.value), "the error must name the session that owns the log"


def test_an_empty_batch_for_the_wrong_session_is_refused_on_both_paths(tmp_path: Path) -> None:
    """The two methods must not disagree about the same input.

    `record_appended` returned on an empty batch *before* reading anything, so an empty batch
    for the wrong session was silently accepted there and refused by `validate_batch`. Nothing
    was written either way, but "accepted" and "refused" cannot both be the answer for one call.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    alloc.record_appended("sess-a", (FIRST_LOG_OFFSET,))
    (tmp_path / "sess-b.cursor").write_text(
        (tmp_path / "sess-a.cursor").read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(LogOffsetSessionMismatchError):
        alloc.validate_batch("sess-b", (), cursor=LogOffset(2))
    with pytest.raises(LogOffsetSessionMismatchError):
        alloc.record_appended("sess-b", ())


def test_a_batch_cannot_move_the_cursor_backward(tmp_path: Path) -> None:
    """A rewind makes the next batch re-issue offsets the log already holds.

    Duplicate positions in an append-only log are not recoverable by reading it back, and
    `validate_batch` cannot catch them afterwards because it reads the same cursor.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    alloc.record_appended("sess-a", (FIRST_LOG_OFFSET, LogOffset(2), LogOffset(3)))

    with pytest.raises(LogOffsetContiguityError):
        alloc.record_appended("sess-a", (FIRST_LOG_OFFSET,))
    assert alloc.next_offset("sess-a") == LogOffset(4), "the cursor moved despite the refusal"

    # A forward jump, not a rewind: the same check refuses both, and naming only the rewind
    # would leave the reader thinking one of these two cases is untested.
    with pytest.raises(LogOffsetContiguityError):
        alloc.record_appended("sess-a", (LogOffset(9999),))
    assert alloc.next_offset("sess-a") == LogOffset(4)


@pytest.mark.parametrize(
    "contents",
    [
        '{"session_id": "sess-a", "cursor": -5}',
        '{"session_id": "sess-a", "cursor": true}',
        '{"session_id": "sess-a"}',
        '{"cursor": 3}',
        "not json at all",
        "[]",
    ],
)
def test_a_corrupt_cursor_is_refused_not_treated_as_absent(tmp_path: Path, contents: str) -> None:
    """Absent hands back the first offset and lets a writer overwrite a log that exists."""
    alloc = FileLogOffsetAllocator(root=tmp_path)
    (tmp_path / "sess-a.cursor").write_text(contents, encoding="utf-8")
    with pytest.raises(LogOffsetCorruptCursorError):
        alloc.next_offset("sess-a")


def test_a_traversing_session_id_is_a_path_error_not_a_session_mismatch(tmp_path: Path) -> None:
    """A path-safety refusal reported as a session mismatch names the wrong cause."""
    alloc = FileLogOffsetAllocator(root=tmp_path)
    with pytest.raises(PathTraversalError):
        alloc.next_offset("../evil")
    with pytest.raises(PathTraversalError):
        alloc.next_offset("")


def test_a_contiguous_batch_is_accepted(tmp_path: Path) -> None:
    alloc = FileLogOffsetAllocator(root=tmp_path)
    alloc.validate_batch("sess-a", (FIRST_LOG_OFFSET, LogOffset(2)), cursor=FIRST_LOG_OFFSET)


# ---------------------------------------------------------------------------
# Reaping temp records. Every case here was reachable and untested in the
# round-2 review: the reaper deleted a committed cursor and the log silently
# restarted at offset 1.
# ---------------------------------------------------------------------------


def test_reaping_never_deletes_a_committed_record_whose_id_looks_like_debris(
    tmp_path: Path,
) -> None:
    """The exact defect: a session id containing the temp infix must survive a reap.

    Before the anchor, `*.cursor.tmp.*` matched `s.cursor.tmp.9.cursor` — a *committed*
    record — and unlinking it restarted the log at offset 1 with no error. That is the
    silent-wipe shape the module docstring says this backend refuses, arrived at through the
    code meant to keep the directory clean. `agent/session.py` anchors for the same reason
    (#269), and this is that finding re-opened in a new file.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    # This id is chosen so the *unanchored* pattern matches its committed record and the
    # anchored one does not. An earlier version used "s.cursor.tmp.9", whose committed name
    # ends `.9.cursor` — and `cursor` is not eight hex characters, so *neither* pattern matched
    # it. That test passed against a reaper with no pattern at all, so it pinned the glob-wide
    # unlink and was blind to the anchor the docstring calls load-bearing.
    session_id = "x.cursor.tmp.999999999.deadbeefz"
    alloc.record_appended(session_id, (FIRST_LOG_OFFSET, LogOffset(2)))
    assert alloc.next_offset(session_id) == LogOffset(3)

    assert alloc.reap_orphaned_temp_files() == 0, "a committed record is not debris"
    assert alloc.next_offset(session_id) == LogOffset(3), "the log was silently rewound"


def test_reaping_never_deletes_a_live_writers_in_flight_temp_file(tmp_path: Path) -> None:
    """Our own pid is alive, so a temp file naming it is a write in progress, not debris.

    Unlinking it would make that writer's `os.replace` fail. A reaper that can break a
    concurrent write is worse than the debris it collects.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    live = tmp_path / f"sess-a.cursor.tmp.{os.getpid()}.deadbeef"
    live.write_text("{}", encoding="utf-8")

    assert alloc.reap_orphaned_temp_files() == 0
    assert live.exists()


def test_reaping_removes_a_temp_file_whose_writer_is_gone(tmp_path: Path) -> None:
    alloc = FileLogOffsetAllocator(root=tmp_path)
    # pid 1 exists; a pid that cannot be allocated does not. `os.kill(pid, 0)` on a free pid
    # raises ProcessLookupError, which is how the reaper decides.
    dead = tmp_path / "sess-a.cursor.tmp.999999999.deadbeef"
    dead.write_text("{}", encoding="utf-8")

    assert alloc.reap_orphaned_temp_files() == 1
    assert not dead.exists()


def test_reaping_removes_a_live_pids_temp_file_once_it_is_old_enough(tmp_path: Path) -> None:
    """A pid is reused, so liveness alone would keep debris forever. Age is the second guard."""
    alloc = FileLogOffsetAllocator(root=tmp_path)
    stale = tmp_path / f"sess-a.cursor.tmp.{os.getpid()}.deadbeef"
    stale.write_text("{}", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(stale, (old, old))

    assert alloc.reap_orphaned_temp_files() == 1
    assert not stale.exists()


def test_reaping_ignores_a_name_that_does_not_match_the_temp_pattern(tmp_path: Path) -> None:
    alloc = FileLogOffsetAllocator(root=tmp_path)
    for name in ("sess-a.cursor.tmp.notapid.deadbeef", "sess-a.cursor.tmp.1.xyz", "x.cursor"):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    assert alloc.reap_orphaned_temp_files() == 0


# ---------------------------------------------------------------------------
# Adversarial reads. Each of these escaped `LogOffsetError` entirely before,
# so a caller writing `except LogOffsetError` caught nothing and fell back to
# the one behaviour that must never happen: treating the log as absent.
# ---------------------------------------------------------------------------


def test_a_cursor_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    alloc = FileLogOffsetAllocator(root=tmp_path)
    (tmp_path / "sess-a.cursor").write_bytes(b'{"session_id":"sess-a","cursor":1,"x":"\xff\xfe"}')

    with pytest.raises(LogOffsetCorruptCursorError):
        alloc.next_offset("sess-a")


def test_a_directory_where_the_cursor_belongs_is_refused(tmp_path: Path) -> None:
    """An `IsADirectoryError` is an `OSError`, not a `FileNotFoundError`, and must not read
    as an absent log."""
    alloc = FileLogOffsetAllocator(root=tmp_path)
    (tmp_path / "sess-a.cursor").mkdir()

    with pytest.raises(LogOffsetCorruptCursorError):
        alloc.next_offset("sess-a")
    with pytest.raises(LogOffsetCorruptCursorError):
        alloc.record_appended("sess-a", (FIRST_LOG_OFFSET,))


def test_the_write_path_refuses_an_offset_the_read_path_would_not_accept(tmp_path: Path) -> None:
    """Otherwise one write bricks the session permanently, with no repair in the API.

    The read path bounds the cursor at 2**53; without the same bound on the write path a
    record lands that every later call — including `next_offset` — then refuses to read.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    (tmp_path / "sess-a.cursor").write_text(
        '{"session_id": "sess-a", "cursor": 9007199254740992}', encoding="utf-8"
    )

    with pytest.raises(LogOffsetContiguityError):
        alloc.record_appended("sess-a", (LogOffset(9007199254740993),))

    # The refusal left the record alone, so it still reads back. That is all this asserts.
    # `next_offset` now returns MAX_LOG_OFFSET + 1, which no write path will ever accept, so
    # the session can take no further append — see the terminal-refusal note below.
    assert alloc.next_offset("sess-a") == LogOffset(9007199254740993)


def test_both_paths_agree_about_the_largest_acceptable_offset(tmp_path: Path) -> None:
    """`validate_batch` and `record_appended` must draw the bound in the same place.

    An earlier fix put the bound on the write path only, so `validate_batch` accepted a batch
    `record_appended` then refused: a caller could validate, extend the log, and be refused
    with the events already written. That is the leading-cursor failure the design exists to
    avoid, reintroduced by the fix for its mirror image.

    This replaces a test that asserted facts about `json` and `float` and imported nothing from
    the module — it could not fail for any change to `file_allocator.py`, while its docstring
    claimed it stopped the docstring from drifting back. A test that cannot fail does not pin
    anything, and claiming it does is worse than not having it.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    at_bound = LogOffset(MAX_LOG_OFFSET)
    over = LogOffset(MAX_LOG_OFFSET + 1)

    # The bound itself is acceptable, on both paths, and reads back.
    (tmp_path / "sess-a.cursor").write_text(
        json.dumps({"session_id": "sess-a", "cursor": MAX_LOG_OFFSET - 1}), encoding="utf-8"
    )
    alloc.validate_batch("sess-a", (at_bound,), cursor=at_bound)
    alloc.record_appended("sess-a", (at_bound,))
    assert alloc.next_offset("sess-a") == over

    # One past it is refused by *both*, not by one of them.
    with pytest.raises(LogOffsetContiguityError):
        alloc.validate_batch("sess-a", (over,), cursor=over)
    with pytest.raises(LogOffsetContiguityError):
        alloc.record_appended("sess-a", (over,))
    # The refusals left the record readable. **They did not leave the session usable**, and
    # an earlier version of this comment claimed they did — the same false "not wedged" claim
    # a review had already blocked once, re-committed in the fix for it.
    #
    # What is true: `next_offset` returns MAX_LOG_OFFSET + 1, and every non-empty batch at or
    # from that offset is refused, forever, with no reset in the API. Verified below rather
    # than asserted in prose, because prose is how this got through the first time.
    assert alloc.next_offset("sess-a") == over
    for batch in ((over,), (over, LogOffset(over + 1)), (at_bound,)):
        with pytest.raises(LogOffsetContiguityError):
            alloc.record_appended("sess-a", batch)
    alloc.record_appended("sess-a", ())  # an empty batch is still accepted, and writes nothing
    assert alloc.next_offset("sess-a") == over

    # This is the same terminal-refusal tradeoff `_require_identity` documents for a
    # case-insensitive id collision: refusing is right, the state it leaves is unrecoverable
    # through the API, and calling that "not wedged" would be the comfortable lie.


def test_a_pid_too_large_for_the_platform_does_not_abort_the_whole_reap(tmp_path: Path) -> None:
    """`os.kill` raises `OverflowError` on such a pid, which is not an `OSError`.

    Unguarded, one junk filename aborted the reap for every real orphan in the directory and
    threw outside this module's error taxonomy — the property `_read_record` argues at length
    must hold.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    junk = tmp_path / "sess-a.cursor.tmp.99999999999999999999.deadbeef"
    junk.write_text("{}", encoding="utf-8")
    real = tmp_path / "sess-b.cursor.tmp.999999999.deadbeef"
    real.write_text("{}", encoding="utf-8")

    assert alloc.reap_orphaned_temp_files() == 1, "the junk name must not abort the reap"
    assert not real.exists(), "the real orphan is still collected"
    assert junk.exists(), "fresh: not debris by age yet — see the age-fallback test below"


# ---------------------------------------------------------------------------
# Arms a round-4 review showed the suite could delete and stay green. Each
# mutation named in a comment was reproduced before the test was written.
# ---------------------------------------------------------------------------


def test_max_age_seconds_of_zero_disables_the_age_guard(tmp_path: Path) -> None:
    """The sentinel reading, pinned — round 4 documented it and tested nothing.

    `0` does not mean "any age". It disables the guard, so a live pid's temp file is kept
    however old it is. That is the opposite of what "older than 0" suggests, which is why the
    docstring says so — and a documented surprise with no test is just a comment.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    stale = tmp_path / f"sess-a.cursor.tmp.{os.getpid()}.deadbeef"
    stale.write_text("{}", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(stale, (old, old))

    assert alloc.reap_orphaned_temp_files(max_age_seconds=0) == 0
    assert stale.exists(), "0 disables the age guard rather than meaning 'any age'"
    # The same file, with the guard enabled, is debris.
    assert alloc.reap_orphaned_temp_files() == 1


def test_a_pid_too_large_is_still_collected_once_it_is_old_enough(tmp_path: Path) -> None:
    """An uninterpretable pid must not make a file uncollectable forever.

    An earlier version returned early here and called it caution, reasoning that the pid is
    the only evidence of ownership. But a *live* pid's file is deleted on age alone, so age is
    already accepted as sufficient evidence — declining it only produced a permanent leak, and
    the reaper is the sole unlink path for this debris.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    junk = tmp_path / "sess-a.cursor.tmp.99999999999999999999.deadbeef"
    junk.write_text("{}", encoding="utf-8")

    assert alloc.reap_orphaned_temp_files() == 0, "fresh: not yet debris by age"
    old = time.time() - 10_000
    os.utime(junk, (old, old))
    assert alloc.reap_orphaned_temp_files() == 1, "old: collectable, not leaked forever"


def test_a_pid_of_zero_is_not_treated_as_a_live_process(tmp_path: Path) -> None:
    r"""`os.kill(0, 0)` *succeeds*, so pid 0 would read back as alive and its debris be kept.

    Signal 0 delivers nothing — it performs only the existence and permission check — so the
    hazard is not that the process group gets signalled. An earlier version of this docstring
    said it would; that is precisely what signal 0 exists to avoid. The hazard is that the
    check returns cleanly, `_is_pid_alive(0)` answers True, and a temp file naming pid 0 is
    never collected.

    The guard is `pid <= 0` while `_TEMP_SUFFIX_PATTERN` captures `(\d+)`, so the negative
    half is unreachable from a filename. It is kept as a precondition on a private helper, and
    only the reachable half is tested — hence the name says zero rather than "or below".
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    zero = tmp_path / "sess-a.cursor.tmp.0.deadbeef"
    zero.write_text("{}", encoding="utf-8")

    assert alloc.reap_orphaned_temp_files() == 1
    assert not zero.exists()


def test_a_directory_named_like_debris_is_not_unlinked(tmp_path: Path) -> None:
    """A directory matching the temp pattern survives and does not stop the reap.

    This pins the *behaviour*, and deliberately not the `is_file` guard: removing that guard
    leaves this green, because `unlink` on a directory raises `OSError` and the handler below
    already catches it, logs, and continues. The guard's value is avoiding a spurious warning,
    not correctness — measured, rather than asserted the other way round.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    (tmp_path / "sess-a.cursor.tmp.999999999.deadbeef").mkdir()
    real = tmp_path / "sess-b.cursor.tmp.999999999.deadbeef"
    real.write_text("{}", encoding="utf-8")

    assert alloc.reap_orphaned_temp_files() == 1
    assert (tmp_path / "sess-a.cursor.tmp.999999999.deadbeef").is_dir()
    assert not real.exists()


def test_a_record_naming_an_empty_session_is_refused(tmp_path: Path) -> None:
    """`""` is falsy but is a `str`, so an `isinstance` check alone lets it through."""
    alloc = FileLogOffsetAllocator(root=tmp_path)
    (tmp_path / "sess-a.cursor").write_text('{"session_id": "", "cursor": 1}', encoding="utf-8")

    with pytest.raises(LogOffsetCorruptCursorError):
        alloc.next_offset("sess-a")


@pytest.mark.parametrize("session_id", ["..", "a..b", "", "a\x00b"])
def test_a_traversal_shaped_id_is_refused_on_every_path(tmp_path: Path, session_id: str) -> None:
    """All three public methods, and ids the `/` guard does not already catch.

    The only traversal test covered `"../evil"`, which the `"/"` entry refuses — so removing
    `".."` from the forbidden set left the suite green while the comment above `_cursor_path`
    presents `..` as the reason the check precedes path arithmetic.
    """
    alloc = FileLogOffsetAllocator(root=tmp_path)
    with pytest.raises(PathTraversalError):
        alloc.next_offset(session_id)
    with pytest.raises(PathTraversalError):
        alloc.validate_batch(session_id, (FIRST_LOG_OFFSET,), cursor=FIRST_LOG_OFFSET)
    with pytest.raises(PathTraversalError):
        alloc.record_appended(session_id, (FIRST_LOG_OFFSET,))
    with pytest.raises(PathTraversalError):
        alloc.record_appended(session_id, ())


def test_a_temp_file_owned_by_another_users_live_process_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`PermissionError` from `os.kill` means the process exists and is someone else's.

    That is alive for our purposes — the file is a write in progress, not debris — and the
    `_is_pid_alive` docstring argues exactly this case. Nothing pinned it: flipping the arm to
    `return False` left the suite green, and the consequence is unlinking a live foreign
    writer's temp file.

    Monkeypatched because provoking a real cross-user `PermissionError` needs a second user.
    """
    from uclone_x.log import file_allocator

    def _kill(pid: int, sig: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(file_allocator.os, "kill", _kill)

    alloc = FileLogOffsetAllocator(root=tmp_path)
    foreign = tmp_path / "sess-a.cursor.tmp.4242.deadbeef"
    foreign.write_text("{}", encoding="utf-8")

    assert alloc.reap_orphaned_temp_files() == 0
    assert foreign.exists(), "a live process owned by another user is not a dead writer"


def test_a_failed_write_does_not_leave_its_temp_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `except`-not-`finally` cleanup on the write path, which nothing exercised.

    A temp file left by a *failed* write is debris the reaper only collects later, and only
    once its pid is gone or it is old enough — so leaking it here is visible for minutes.
    """
    from uclone_x.log import file_allocator

    def _boom(src: object, dst: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(file_allocator.os, "replace", _boom)

    alloc = FileLogOffsetAllocator(root=tmp_path)
    with pytest.raises(OSError, match="No space left"):
        alloc.record_appended("sess-a", (FIRST_LOG_OFFSET,))

    assert list(tmp_path.glob("*.tmp.*")) == [], "the temp file outlived the failed write"
    assert alloc.next_offset("sess-a") == FIRST_LOG_OFFSET, "no partial record was committed"
