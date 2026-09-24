"""One run per key, shared by everyone who asks for it while it lasts (#1233).

`POST /api/models/pull` is the case this exists for. Before it, nothing bounded
how many pulls ran at once: a double-click, a second dashboard tab, or an
impatient retry each started its own multi-gigabyte download of the *same*
weights, competing for the same disk and the same link, and the surface gave no
sign that one was already under way.

**Why joining and not refusing.** A cap that refuses the second caller is the
other legible policy, and it is the wrong one here. The caller asking to install
`qwen3:8b` while `qwen3:8b` is installing wants the model present; that wish is
already being granted, so refusing it reports a failure where there is none and
invites exactly the retry that caused the duplicate. Refusing also needs a
number — two concurrent pulls? three? — and every such number is a guess that
turns some request which would have succeeded into an error. Keying on the model
needs no number: the thing being bounded is duplicated work, and duplicated work
is defined by the key, not by a count.

**What is deliberately not bounded.** Two *different* models pulled at once still
both run. The caller who could fan out over many models is a page in another tab,
and `_refuse_cross_origin` already closes that door; the dashboard's own Settings
modal disables its input while an install is in flight. Bounding distinct models
would defend a door that is shut, at the cost of refusing a user who genuinely
wants two models.

**Cancellation does not travel.** The shared work runs as a `Task` and every
caller awaits it through `asyncio.shield`, so a caller that goes away — a browser
that aborted, a request that was cancelled — leaves the run alone for the callers
that are still waiting. The download is not the waiter's to cancel on everyone
else's behalf, and the daemon would keep fetching the blobs regardless.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["SingleFlight"]


class SingleFlight:
    """Run at most one coroutine per key, and let concurrent callers share its result.

    Not thread-safe and not process-shared: it holds one event loop's in-flight
    work. That is the scope it is asked to cover — `ucx ui` is one process, and a
    pull started from a terminal is another program entirely.
    """

    def __init__(self) -> None:
        self._in_flight: dict[str, asyncio.Task[None]] = {}
        self._joins = 0

    @property
    def joins(self) -> int:
        """How many callers have so far joined a run started by someone else.

        Monotonic for the life of the instance. It is the one externally visible
        sign that de-duplication happened at all — a caller cannot tell from its
        own 200 whether it did the work — so it is what an operator reads and
        what a test waits on.
        """
        return self._joins

    def in_flight(self, key: str) -> bool:
        """Whether a run for `key` is under way right now."""
        return key in self._in_flight

    async def run(self, key: str, work: Callable[[], Coroutine[Any, Any, None]]) -> bool:
        """Await `work()` for `key`, or join the run already under way for it.

        Returns `True` when this caller started the run and `False` when it
        joined one. Either way it sees the same outcome: the run's exception is
        raised to every caller, because a joiner that was told `ok` while the
        pull it joined failed would be the reporting hole P6 forbids.

        `work` is only called when this caller starts the run, so a joiner never
        builds a second request to the daemon.
        """
        existing = self._in_flight.get(key)
        if existing is not None:
            self._joins += 1
            logger.debug("single-flight: joining the run already under way for %r", key)
            await asyncio.shield(existing)
            return False

        task = asyncio.create_task(work())
        self._in_flight[key] = task
        task.add_done_callback(self._release(key))
        await asyncio.shield(task)
        return True

    def _release(self, key: str) -> Callable[[asyncio.Task[None]], None]:
        """Drop `key` when its run ends, and account for a run nobody is left waiting on."""

        def released(task: asyncio.Task[None]) -> None:
            if self._in_flight.get(key) is task:
                del self._in_flight[key]
            if not task.cancelled() and task.exception() is not None:
                # Reading it marks it retrieved. Every caller still waiting is
                # raised into by `run`; this is for the case where the last one
                # went away first, which asyncio would otherwise report at exit
                # as a failure nobody handled.
                logger.debug("single-flight: the run for %r ended in failure", key)

        return released
