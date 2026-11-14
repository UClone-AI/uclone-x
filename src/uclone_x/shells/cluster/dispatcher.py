"""Keyed partition dispatcher with per-conversation FIFO locks and watermark commits."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from uclone_x.shells.cluster.offset_tracker import PartitionOffsetTracker

__all__ = ["CommitOffsetFn", "KeyedPartitionDispatcher"]

T = TypeVar("T")

CommitOffsetFn = Callable[[str, int, int], Awaitable[None]]


class KeyedPartitionDispatcher:
    """Demultiplexes partition commands with per-conversation FIFO ordering and bounded concurrency.

    Commands sharing a `conversation_id` (`session_id` or `room_id`) run strictly sequentially
    in dispatch order, while commands for distinct `conversation_id`s on the same partition
    run concurrently up to `max_concurrent_turns`.

    **An offset advances only when its work returned** (Invariant 3). Work that raises or is
    cancelled keeps its offset in flight, so the watermark stays below it and Kafka redelivers
    it to whoever owns the partition next. "Done" is the work's decision, and an interrupted or
    refused turn is one that returned.
    """

    def __init__(
        self,
        commit_offset_fn: CommitOffsetFn,
        *,
        max_concurrent_turns: int = 16,
    ) -> None:
        if max_concurrent_turns < 1:
            raise ValueError("max_concurrent_turns must be >= 1")
        self._commit_offset_fn = commit_offset_fn
        self._max_concurrent_turns = max_concurrent_turns
        self._semaphore = asyncio.Semaphore(max_concurrent_turns)
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._conversation_waiters: dict[str, int] = {}
        self._trackers: dict[tuple[str, int], PartitionOffsetTracker] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._waiting = 0

    @property
    def max_concurrent_turns(self) -> int:
        """Maximum concurrent turns executing across all partitions on this worker."""
        return self._max_concurrent_turns

    @property
    def in_flight(self) -> int:
        """Dispatched commands not finished yet, but for those waiting out a backoff."""
        return len(self._tasks) - self._waiting

    async def wait_outside_slot(self, seconds: float) -> None:
        """Wait with this work's concurrency slot given up, for work that holds one.

        A turn backing off holds up its conversation: its slot goes to someone else, and it does
        not count toward the backpressure on fetching. Commands queued behind it still count.
        """
        self._semaphore.release()
        self._waiting += 1
        try:
            await asyncio.sleep(seconds)
        finally:
            self._waiting -= 1
            await self._semaphore.acquire()

    def tracker_for(self, topic: str, partition: int) -> PartitionOffsetTracker:
        """Return (or create) the `PartitionOffsetTracker` for `(topic, partition)`."""
        key = (topic, partition)
        tracker = self._trackers.get(key)
        if tracker is None:
            tracker = PartitionOffsetTracker()
            self._trackers[key] = tracker
        return tracker

    def forget_partition(self, topic: str, partition: int) -> None:
        """This consumer lost `(topic, partition)`: drop its tracker and commit no more for it.

        A turn still running finishes and commits its session state, but its offset is the new
        owner's now. If the partition comes back, its offsets are read again from the last
        commit, and the redelivered copy queues behind the one running here, then finds the
        turn committed.
        """
        self._trackers.pop((topic, partition), None)

    def dispatch(
        self,
        *,
        topic: str,
        partition: int,
        offset: int,
        conversation_id: str,
        work: Callable[[], Awaitable[T]],
    ) -> asyncio.Task[T]:
        """Register `offset` as dispatched and schedule `work` under the conversation's FIFO lock."""
        tracker = self.tracker_for(topic, partition)
        tracker.register_dispatched(offset)

        lock = self._conversation_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[conversation_id] = lock
        self._conversation_waiters[conversation_id] = (
            self._conversation_waiters.get(conversation_id, 0) + 1
        )

        async def _runner() -> T:
            finished = False
            try:
                async with lock, self._semaphore:
                    outcome = await work()
                finished = True
                return outcome
            finally:
                remaining = self._conversation_waiters.get(conversation_id, 1) - 1
                if remaining <= 0:
                    self._conversation_waiters.pop(conversation_id, None)
                    self._conversation_locks.pop(conversation_id, None)
                else:
                    self._conversation_waiters[conversation_id] = remaining

                if finished:
                    commit_offset = tracker.mark_completed(offset)
                    still_owned = self._trackers.get((topic, partition)) is tracker
                    if commit_offset is not None and still_owned:
                        await self._commit_offset_fn(topic, partition, commit_offset)

        task = asyncio.create_task(_runner())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self, timeout_seconds: float) -> None:
        """Wait up to `timeout_seconds` for dispatched work, then cancel what is left.

        Cancelled work keeps its offset uncommitted, so it is redelivered: nothing is lost by
        leaving early, and nothing is reported done that was not.
        """
        pending = set(self._tasks)
        if pending:
            await asyncio.wait(pending, timeout=timeout_seconds)
        leftover = [task for task in self._tasks if not task.done()]
        for task in leftover:
            task.cancel()
        if leftover:
            await asyncio.gather(*leftover, return_exceptions=True)
