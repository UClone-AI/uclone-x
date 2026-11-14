"""One async client per event loop, for libraries whose clients are bound to the loop that made them.

`asyncpg` pools, `valkey.asyncio` clients and `aiokafka` producers all belong to the event
loop they were created on; using one from another loop fails or hangs. The cluster stores are
called from two loops at once: the caller's loop (`await store.prefetch_session(...)`) and the
background loop that bridges the synchronous `SessionStoreProtocol` methods. `LoopLocal` keeps
one client per loop and closes every one of them on `aclose()`.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Generic, TypeVar

__all__ = ["LoopLocal", "run_on_bridge_loop"]

_T = TypeVar("_T")

_bridge_guard = threading.Lock()
_bridge_loop: asyncio.AbstractEventLoop | None = None


def _ensure_bridge_loop() -> asyncio.AbstractEventLoop:
    global _bridge_loop  # noqa: PLW0603
    with _bridge_guard:
        if _bridge_loop is None or _bridge_loop.is_closed():
            ready = threading.Event()
            created: list[asyncio.AbstractEventLoop] = []

            def _runner() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                created.append(loop)
                ready.set()
                loop.run_forever()

            thread = threading.Thread(target=_runner, name="ucx-cluster-sync-bridge", daemon=True)
            thread.start()
            ready.wait()
            _bridge_loop = created[0]
        return _bridge_loop


def run_on_bridge_loop(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run `coro` to completion on the process's bridge loop and block for its result.

    The kernel's `SessionStoreProtocol` and `UsageStore` are synchronous; the cluster clients
    are async. This is the one bridge between them: a dedicated background loop, so a blocking
    caller on the turn's own loop cannot deadlock against it. Blocking that caller is the
    accepted cost of keeping those protocols synchronous (core/session_store.py).
    """
    return asyncio.run_coroutine_threadsafe(coro, _ensure_bridge_loop()).result()


class LoopLocal(Generic[_T]):
    """Lazily creates one `_T` per running event loop and closes them all on `aclose()`."""

    def __init__(
        self,
        create: Callable[[], Awaitable[_T]],
        close: Callable[[_T], Coroutine[Any, Any, None]],
    ) -> None:
        self._create = create
        self._close = close
        self._guard = threading.Lock()
        self._by_loop: dict[asyncio.AbstractEventLoop, _T] = {}
        self._pending: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}

    def _forget_closed_loops(self) -> None:
        """Drop what closed loops left behind (#2468); their connections went with them.

        Called with `_guard` held. A process that creates loops as it goes (`asyncio.run` per
        call, a test per loop) would otherwise keep every client it ever made.
        """
        for loop in [held for held in self._by_loop if held.is_closed()]:
            del self._by_loop[loop]
        for loop in [held for held in self._pending if held.is_closed()]:
            del self._pending[loop]

    async def get(self) -> _T:
        """Return this loop's instance, creating it on first use."""
        loop = asyncio.get_running_loop()
        with self._guard:
            self._forget_closed_loops()
            existing = self._by_loop.get(loop)
            if existing is not None:
                return existing
            creating = self._pending.setdefault(loop, asyncio.Lock())
        async with creating:
            with self._guard:
                existing = self._by_loop.get(loop)
            if existing is not None:
                return existing
            created = await self._create()
            with self._guard:
                self._by_loop[loop] = created
            return created

    async def aclose(self) -> None:
        """Close the instance of every loop: directly on this loop, threadsafe on the others."""
        current = asyncio.get_running_loop()
        with self._guard:
            items = list(self._by_loop.items())
            self._by_loop.clear()
            self._pending.clear()
        for loop, instance in items:
            if loop is current:
                await self._close(instance)
            elif loop.is_running() and not loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(self._close(instance), loop)
                await asyncio.wrap_future(future)
            # A loop that has already stopped took its connections with it.
