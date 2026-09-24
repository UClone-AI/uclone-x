"""A non-blocking stdout that a reply overfills, whatever the pipe's capacity (#1008).

`tests/unit/test_cli_run.py` needs a reply that goes into an empty non-blocking pipe part-way
and then fails with `BlockingIOError`. It used to send a fixed 96 KiB, on the assumption that
a pipe holds at most 64 KiB. That holds on macOS and on 4 KiB-page Linux. Linux sizes a
default pipe at 16 pages, so an arm64 kernel with 16 KiB or 64 KiB pages gives 256 KiB or
1 MiB, the whole reply fits, and the test fails there.

So the capacity is measured on the pipe the child will get, and the reply is sized from it.
It is measured by filling the pipe rather than looked up: Linux has `F_GETPIPE_SZ`, macOS has
no equivalent, and one path that runs on both is one path the gate exercises.
"""

from __future__ import annotations

import os
import select
import time

#: How far past the measured capacity the prompt is sized, so the reply is longer than the
#: pipe whatever the connector wraps around it. The child's write then goes in part-way and
#: `BlockingIOError` follows, raised by the write or by the flush after it; the test needs
#: that error and a strict prefix, not any particular size of remainder. A default Linux
#: pipe is 16 pages and a single argument may be 32 pages including its NUL, so a prompt of
#: `capacity + OVERFILL_MARGIN` fits on the command line on any page size larger than
#: 2 KiB. On macOS and 4 KiB-page Linux it gives the 96 KiB the test always sent.
OVERFILL_MARGIN = 32 * 1024

#: Bytes offered per write while measuring. Larger than any capacity measured here, so a
#: write accepts all the room there is and the next one fails with EAGAIN.
_PROBE_WRITE = 4 * 1024 * 1024


def nonblocking_pipe() -> tuple[int, int]:
    """A new pipe as `(read_fd, write_fd)`, with the write end non-blocking."""
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    return read_fd, write_fd


def capacity_of(read_fd: int, write_fd: int) -> int:
    """Bytes a non-blocking `write_fd` accepts before EAGAIN: the room left in the channel.

    Measured by filling the channel, then reading back as many bytes as the probe put in, so
    the channel holds as many bytes as it did before. A pipe is first in, first out: if it
    was not empty, what remains is the probe's tail, not the bytes that were there. The
    partial-write test measures an empty pipe, where there is no difference.
    """
    probe = bytes(_PROBE_WRITE)
    accepted = 0
    while True:
        try:
            accepted += os.write(write_fd, probe)
        except BlockingIOError:
            break
    drained = 0
    while drained < accepted:
        drained += len(os.read(read_fd, accepted - drained))
    return accepted


def read_to_eof(read_fd: int, timeout: float) -> bytes:
    """Everything `read_fd` delivers until EOF, or `TimeoutError` after `timeout` seconds.

    EOF arrives only once every write end is closed, so a process that still holds one, such
    as a helper a child left behind, would otherwise block the reader for good.
    """
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    while True:
        readable, _, _ = select.select([read_fd], [], [], max(deadline - time.monotonic(), 0))
        if not readable:
            raise TimeoutError(f"no EOF within {timeout} s: a write end is still open")
        chunk = os.read(read_fd, 64 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
