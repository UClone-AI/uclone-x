"""Contiguous partition offset watermark tracker for out-of-order turn completions."""

from __future__ import annotations

__all__ = ["PartitionOffsetTracker"]


class PartitionOffsetTracker:
    """Tracks in-flight and completed Kafka offsets for a single partition.

    Turns on different conversations assigned to the same partition may complete out of
    order. Kafka offset commits are cumulative (`commit(offset)` commits all messages
    below `offset`), so the worker may only commit `highest_contiguous_offset + 1` once
    every lower dispatched offset has also completed.
    """

    def __init__(self) -> None:
        self._in_flight: set[int] = set()
        self._completed: set[int] = set()
        self._cursor: int | None = None
        self._next_commit_offset: int | None = None

    @property
    def next_commit_offset(self) -> int | None:
        """Current contiguous committed watermark (`highest_contiguous_offset + 1`), or `None`."""
        return self._next_commit_offset

    @property
    def in_flight_count(self) -> int:
        """Number of currently in-flight offsets on this partition."""
        return len(self._in_flight)

    def register_dispatched(self, offset: int) -> None:
        """Record `offset` as in-flight, rejecting negative or already-committed offsets."""
        if offset < 0:
            raise ValueError(f"Partition offset must be non-negative, got {offset}")
        if self._next_commit_offset is not None and offset < self._next_commit_offset:
            raise ValueError(
                f"Offset {offset} is below committed watermark {self._next_commit_offset}"
            )
        if offset in self._in_flight or offset in self._completed:
            raise ValueError(f"Offset {offset} is already tracked on this partition")

        if self._cursor is None:
            self._cursor = offset
        elif offset < self._cursor:
            raise ValueError(f"Offset {offset} is below active watermark cursor {self._cursor}")
        elif not self._in_flight and not self._completed and offset > self._cursor:
            self._cursor = offset

        self._in_flight.add(offset)

    def mark_completed(self, offset: int) -> int | None:
        """Mark `offset` completed and return `highest_contiguous_offset + 1` if watermark advanced."""
        if offset not in self._in_flight:
            raise ValueError(f"Offset {offset} is not registered as in-flight")

        self._in_flight.remove(offset)
        self._completed.add(offset)

        # If any lower dispatched offset is still in-flight, the contiguous watermark cannot move.
        if any(inflight < offset for inflight in self._in_flight) and (
            self._cursor is None or self._cursor not in self._completed
        ):
            return None

        highest_contiguous_offset: int | None = None
        while self._cursor is not None and self._cursor in self._completed:
            highest_contiguous_offset = self._cursor
            self._completed.remove(self._cursor)
            self._cursor += 1

        if highest_contiguous_offset is not None:
            self._next_commit_offset = highest_contiguous_offset + 1
            return self._next_commit_offset
        return None
