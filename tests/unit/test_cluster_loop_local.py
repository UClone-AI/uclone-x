"""`LoopLocal`: one client per event loop, none kept for a loop that closed (#2468)."""

from __future__ import annotations

import asyncio

from uclone_x.core.loop_local import LoopLocal, run_on_bridge_loop


def test_a_closed_loops_client_is_dropped_on_the_next_get() -> None:
    """Killed by: src/uclone_x/core/loop_local.py :: self._forget_closed_loops()
    Becomes: pass
    """
    created: list[object] = []

    async def create() -> object:
        client = object()
        created.append(client)
        return client

    async def close(_client: object) -> None:
        return None

    local: LoopLocal[object] = LoopLocal(create, close)
    for _ in range(3):
        asyncio.run(local.get())

    assert len(created) == 3
    assert len(local._by_loop) == 1  # pyright: ignore[reportPrivateUsage]
    assert len(local._pending) == 1  # pyright: ignore[reportPrivateUsage]


def test_the_bridge_runs_a_coroutine_from_a_blocking_caller_on_a_running_loop() -> None:
    async def value() -> int:
        return 42

    async def caller() -> int:
        return run_on_bridge_loop(value())  # would deadlock if it used this loop

    assert asyncio.run(caller()) == 42
