"""Unit tests for `PartitionOffsetTracker` and `KeyedPartitionDispatcher`."""

from __future__ import annotations

import asyncio

import pytest

from uclone_x.shells.cluster.dispatcher import KeyedPartitionDispatcher
from uclone_x.shells.cluster.offset_tracker import PartitionOffsetTracker


def test_partition_offset_tracker_advances_watermark_only_on_contiguous_completion() -> None:
    """`PartitionOffsetTracker` withholds offset commits until the lowest in-flight offset finishes.

    Killed by: src/uclone_x/shells/cluster/offset_tracker.py :: self._next_commit_offset = highest_contiguous_offset + 1
    Becomes: self._next_commit_offset = highest_contiguous_offset
    """
    tracker = PartitionOffsetTracker()
    assert tracker.next_commit_offset is None

    tracker.register_dispatched(10)
    tracker.register_dispatched(11)
    tracker.register_dispatched(12)
    assert tracker.in_flight_count == 3

    # Offsets 11 and 12 finish out of order while 10 is still running.
    assert tracker.mark_completed(11) is None
    assert tracker.next_commit_offset is None
    assert tracker.mark_completed(12) is None
    assert tracker.next_commit_offset is None

    # Once offset 10 finishes, the contiguous watermark jumps across 10, 11, 12 -> 13.
    assert tracker.mark_completed(10) == 13
    assert tracker.next_commit_offset == 13
    assert tracker.in_flight_count == 0


def test_partition_offset_tracker_rejects_negative_and_stale_offsets() -> None:
    """`PartitionOffsetTracker.register_dispatched` rejects negative or already-committed offsets.

    Killed by: src/uclone_x/shells/cluster/offset_tracker.py :: if offset < 0:
    Becomes: if False:
    """
    tracker = PartitionOffsetTracker()
    with pytest.raises(ValueError, match="non-negative"):
        tracker.register_dispatched(-1)

    tracker.register_dispatched(5)
    assert tracker.mark_completed(5) == 6

    with pytest.raises(ValueError, match="below committed watermark"):
        tracker.register_dispatched(5)

    with pytest.raises(ValueError, match="not registered as in-flight"):
        tracker.mark_completed(99)


async def test_keyed_partition_dispatcher_serializes_same_session_and_parallels_distinct_sessions() -> (
    None
):
    """Same `conversation_id` runs strictly FIFO while distinct conversations run concurrently.

    Killed by: src/uclone_x/shells/cluster/dispatcher.py :: async with lock, self._semaphore:
    Becomes: async with self._semaphore:
    Killed by: src/uclone_x/shells/cluster/dispatcher.py :: if commit_offset is not None and still_owned:
    Becomes: if False:
    """
    committed: list[tuple[str, int, int]] = []

    async def record_commit(topic: str, partition: int, offset: int) -> None:
        committed.append((topic, partition, offset))

    dispatcher = KeyedPartitionDispatcher(record_commit, max_concurrent_turns=4)

    events: list[str] = []
    sess_a_first_started = asyncio.Event()
    release_sess_a_first = asyncio.Event()
    sess_b_finished = asyncio.Event()

    async def turn_a1() -> str:
        events.append("a1:start")
        sess_a_first_started.set()
        await release_sess_a_first.wait()
        events.append("a1:end")
        return "a1"

    async def turn_a2() -> str:
        events.append("a2:start")
        events.append("a2:end")
        return "a2"

    async def turn_b1() -> str:
        await sess_a_first_started.wait()
        events.append("b1:start")
        events.append("b1:end")
        sess_b_finished.set()
        return "b1"

    task_a1 = dispatcher.dispatch(
        topic="ucx.cmd.turn",
        partition=0,
        offset=10,
        conversation_id="sess-a",
        work=turn_a1,
    )
    task_a2 = dispatcher.dispatch(
        topic="ucx.cmd.turn",
        partition=0,
        offset=11,
        conversation_id="sess-a",
        work=turn_a2,
    )
    task_b1 = dispatcher.dispatch(
        topic="ucx.cmd.turn",
        partition=0,
        offset=12,
        conversation_id="sess-b",
        work=turn_b1,
    )

    # `sess-b` completes while `turn_a1` is still blocked; `turn_a2` must NOT have started.
    await sess_b_finished.wait()
    assert events == ["a1:start", "b1:start", "b1:end"]
    assert committed == []

    # Release `turn_a1`; `turn_a2` must run strictly after `a1:end`.
    release_sess_a_first.set()
    results = await asyncio.gather(task_a1, task_a2, task_b1)
    assert results == ["a1", "a2", "b1"]
    assert events == ["a1:start", "b1:start", "b1:end", "a1:end", "a2:start", "a2:end"]
    assert committed[-1] == ("ucx.cmd.turn", 0, 13)


async def test_cancelled_or_failed_work_keeps_its_offset_and_drain_cancels_what_is_left() -> None:
    """Only work that returned advances the watermark: the rest must be redelivered.

    Killed by: src/uclone_x/shells/cluster/dispatcher.py :: if finished:
    Becomes: if True:
    """
    committed: list[tuple[str, int, int]] = []

    async def record_commit(topic: str, partition: int, offset: int) -> None:
        committed.append((topic, partition, offset))

    dispatcher = KeyedPartitionDispatcher(record_commit, max_concurrent_turns=2)
    started = asyncio.Event()
    never_set = asyncio.Event()

    async def long_running_turn() -> None:
        started.set()
        await never_set.wait()

    async def failing_turn() -> None:
        raise OSError("gone")

    task = dispatcher.dispatch(
        topic="ucx.cmd.turn",
        partition=1,
        offset=40,
        conversation_id="sess-cancel",
        work=long_running_turn,
    )
    failing = dispatcher.dispatch(
        topic="ucx.cmd.turn",
        partition=2,
        offset=7,
        conversation_id="sess-fail",
        work=failing_turn,
    )
    await started.wait()
    with pytest.raises(OSError, match="gone"):
        await failing
    await asyncio.sleep(0)  # the done callback
    assert dispatcher.in_flight == 1

    await dispatcher.drain(0.0)
    assert task.cancelled()
    assert dispatcher.in_flight == 0
    assert committed == []


async def test_a_forgotten_partition_commits_nothing_and_starts_over() -> None:
    """After a revocation, the old tracker's offsets are the new owner's to commit.

    Killed by: src/uclone_x/shells/cluster/dispatcher.py :: self._trackers.pop((topic, partition), None)
    Becomes: pass
    """
    committed: list[tuple[str, int, int]] = []

    async def record_commit(topic: str, partition: int, offset: int) -> None:
        committed.append((topic, partition, offset))

    dispatcher = KeyedPartitionDispatcher(record_commit)
    release = asyncio.Event()

    async def held() -> None:
        await release.wait()

    running = dispatcher.dispatch(topic="t", partition=0, offset=5, conversation_id="a", work=held)
    await asyncio.sleep(0)
    dispatcher.forget_partition("t", 0)
    release.set()
    await running
    assert committed == []

    # The partition came back and Kafka read from the last commit: the same offset is new again.
    async def quick() -> None:
        return None

    await dispatcher.dispatch(topic="t", partition=0, offset=5, conversation_id="a", work=quick)
    assert committed == [("t", 0, 6)]
