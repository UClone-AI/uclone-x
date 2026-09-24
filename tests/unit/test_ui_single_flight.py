"""`SingleFlight`: one run per key, shared by everyone who asks for it (#1233).

Every test here holds the shared run open on an `asyncio.Event` and releases it by
hand, so the ordering is produced rather than waited for. A concurrency test that
lets the work finish on its own has measured nothing: the second caller arrives
after the first is gone and de-duplication is never asked for, which is exactly
how a test of this shape passes against code that does not de-duplicate at all.

`_yield_until` is the same rule applied to the other direction — it hands control
back to the loop a bounded number of times rather than sleeping for a duration, so
a run where the expected thing never happens fails in microseconds instead of
waiting on a clock.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from uclone_x.ui.single_flight import SingleFlight


async def _yield_until(predicate: Callable[[], bool], *, yields: int = 1000) -> None:
    """Give the loop `yields` chances to make `predicate` true, then fail.

    No wall-clock sleep: the steps being waited for are all in-memory awaits, so
    their number is fixed and does not grow with machine load.
    """
    for _ in range(yields):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"predicate never became true within {yields} event-loop yields")


@pytest.mark.asyncio
async def test_a_second_caller_joins_the_run_already_under_way() -> None:
    """The second request for a model already downloading must not start a second download.

    This is the whole of the concurrency half of #1233. A double-click, a second
    dashboard tab, or an impatient retry each used to start its own multi-gigabyte
    pull of the same weights, competing for the same disk and the same link.

    The run is held open across the second call, so "did it de-duplicate" is
    actually asked. Both return values are asserted, which is what makes a
    scheduling accident fail rather than pass: a second caller that started its own
    run would answer `True` here, not `False`.

    Killed by: src/uclone_x/ui/single_flight.py :: existing = self._in_flight.get(key)
    Becomes: existing = None
    """
    flight = SingleFlight()
    release = asyncio.Event()
    runs: list[str] = []

    async def work() -> None:
        runs.append("ran")
        await release.wait()

    first = asyncio.create_task(flight.run("qwen3:8b", work))
    await _yield_until(lambda: flight.in_flight("qwen3:8b"))

    second = asyncio.create_task(flight.run("qwen3:8b", work))
    await _yield_until(lambda: flight.joins == 1)

    assert runs == ["ran"]  # the joiner has arrived and has not built a request of its own

    release.set()
    assert await first is True
    assert await second is False
    assert runs == ["ran"]
    assert not flight.in_flight("qwen3:8b")


@pytest.mark.asyncio
async def test_a_joiner_is_raised_into_by_the_run_it_joined() -> None:
    """A caller that joined a failed run is told it failed.

    Returning quietly would be the reporting hole P6 forbids, and a worse one than
    the original bug: the surface would say `Installed model "X".` about a pull
    that had just died, because *this* request never went near the daemon.

    Killed by: src/uclone_x/ui/single_flight.py :: await asyncio.shield(existing)
    Becomes: pass
    """
    flight = SingleFlight()
    release = asyncio.Event()

    async def failing_work() -> None:
        await release.wait()
        raise RuntimeError("the daemon gave up")

    first = asyncio.create_task(flight.run("qwen3:8b", failing_work))
    await _yield_until(lambda: flight.in_flight("qwen3:8b"))
    second = asyncio.create_task(flight.run("qwen3:8b", failing_work))
    await _yield_until(lambda: flight.joins == 1)

    release.set()
    with pytest.raises(RuntimeError, match="the daemon gave up"):
        await first
    with pytest.raises(RuntimeError, match="the daemon gave up"):
        await second


@pytest.mark.asyncio
async def test_a_caller_that_goes_away_leaves_the_run_for_the_others() -> None:
    """The browser that aborted does not cancel the download for the tab still waiting.

    `AbortController` on the dashboard ends *that* request; the pull it started is
    shared, and the caller that walked away has no standing to end it for everyone
    else. `asyncio.shield` is what separates the two — without it, cancelling the
    starter's await cancels the task every joiner is waiting on.

    Killed by: src/uclone_x/ui/single_flight.py :: await asyncio.shield(task)
    Becomes: await task
    """
    flight = SingleFlight()
    release = asyncio.Event()
    finished: list[str] = []

    async def work() -> None:
        await release.wait()
        finished.append("done")

    first = asyncio.create_task(flight.run("qwen3:8b", work))
    await _yield_until(lambda: flight.in_flight("qwen3:8b"))
    second = asyncio.create_task(flight.run("qwen3:8b", work))
    await _yield_until(lambda: flight.joins == 1)

    first.cancel()  # the browser aborted, or the request was dropped
    with pytest.raises(asyncio.CancelledError):
        await first

    release.set()
    assert await second is False
    assert finished == ["done"]


@pytest.mark.asyncio
async def test_two_different_models_are_not_made_to_wait_for_each_other() -> None:
    """De-duplication is keyed on the model, not on "a pull is happening".

    A user who wants two models gets two downloads. The thing being bounded is
    duplicated work, and what makes work duplicated is the key.

    The mutation is the same line test 1 pins, replaced differently: a lookup that
    answers with *any* run under way rather than this key's. That still
    de-duplicates a repeat of the same model, so test 1 stays green under it —
    which is the point of declaring both.

    Killed by: src/uclone_x/ui/single_flight.py :: existing = self._in_flight.get(key)
    Becomes: existing = next(iter(self._in_flight.values()), None)
    """
    flight = SingleFlight()
    release = asyncio.Event()
    runs: list[str] = []

    def work_for(model: str) -> Callable[[], Coroutine[Any, Any, None]]:
        async def work() -> None:
            runs.append(model)
            await release.wait()

        return work

    first = asyncio.create_task(flight.run("qwen3:8b", work_for("qwen3:8b")))
    await _yield_until(lambda: flight.in_flight("qwen3:8b"))
    second = asyncio.create_task(flight.run("qwen3:1.7b", work_for("qwen3:1.7b")))
    await _yield_until(lambda: flight.in_flight("qwen3:1.7b"))

    release.set()
    assert await first is True
    assert await second is True
    assert sorted(runs) == ["qwen3:1.7b", "qwen3:8b"]
    assert flight.joins == 0
