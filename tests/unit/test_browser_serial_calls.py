"""Browser calls on one clone's tab run one at a time, even when a step makes several (#2392).

The agent runs every tool call of a step together (`asyncio.gather`). Two `browser` calls
on the same tab that overlap race: a `type` and a `click` sent together lost the click and
reported the type's change as the click's. The calls are now taken in the order the model
made them; calls for different clones still run together.
"""

from __future__ import annotations

import asyncio
from typing import Any

from uclone_x.browser.service import BrowserService, TabKey
from uclone_x.browser.tool import BrowserParams, BrowserTool
from uclone_x.tools.models import ToolContext


class _Timed(BrowserService):
    """Records when each call starts and ends; each call takes a little while."""

    def __init__(self) -> None:
        async def _no_link() -> Any:
            raise AssertionError("no browser in this test")

        super().__init__(_no_link)
        self.log: list[tuple[str, TabKey, str]] = []

    async def _slow(self, key: TabKey, name: str) -> dict[str, Any]:
        self.log.append(("start", key, name))
        await asyncio.sleep(0.05)
        self.log.append(("end", key, name))
        return {"url": "https://example.com/"}

    async def type_text(self, key: TabKey, ref: str | None, text: str, **_: Any) -> dict[str, Any]:
        return await self._slow(key, f"type {ref}")

    async def click(self, key: TabKey, ref: str, **_: Any) -> dict[str, Any]:
        return await self._slow(key, f"click {ref}")


def _context(agent: str) -> ToolContext:
    return ToolContext(agent_id=agent, session_id="s-1")


async def test_two_calls_in_one_step_on_one_tab_run_in_order_not_together() -> None:
    """Killed by: src/uclone_x/browser/service.py :: async with entry.lock:
    Becomes: async with asyncio.Lock():
    """
    service = _Timed()
    tool = BrowserTool(service)

    await asyncio.gather(
        tool.run(BrowserParams(action="type", ref="e1", text="7Q-4821"), _context("scout")),
        tool.run(BrowserParams(action="click", ref="e2"), _context("scout")),
    )

    assert [(event, name) for event, _, name in service.log] == [
        ("start", "type e1"),
        ("end", "type e1"),
        ("start", "click e2"),
        ("end", "click e2"),
    ]


async def test_calls_for_different_clones_still_run_together() -> None:
    service = _Timed()
    tool = BrowserTool(service)

    await asyncio.gather(
        tool.run(BrowserParams(action="click", ref="e1"), _context("scout")),
        tool.run(BrowserParams(action="click", ref="e1"), _context("writer")),
    )

    assert [event for event, _, _ in service.log[:2]] == ["start", "start"]


async def test_a_lock_is_dropped_once_its_calls_end_and_no_tab_is_left() -> None:
    """The service keeps no lock per conversation forever (#2396 review): a key with no tabs
    loses its lock when the last call on it ends, and one with tabs keeps it.

    Killed by: src/uclone_x/browser/service.py :: if entry.users == 0 and key not in self._windows:
    Becomes: if False:
    """
    service = _Timed()
    tool = BrowserTool(service)

    await asyncio.gather(
        tool.run(BrowserParams(action="click", ref="e1"), _context("scout")),
        tool.run(BrowserParams(action="click", ref="e2"), _context("scout")),
    )
    assert service._call_locks == {}  # pyright: ignore[reportPrivateUsage]

    from uclone_x.browser.service import BrowserTab, BrowserWindow

    key = ("s-1", "scout")
    service._windows[key] = BrowserWindow(  # pyright: ignore[reportPrivateUsage]
        tabs=[BrowserTab(target="t", events=asyncio.Queue())]
    )
    await tool.run(BrowserParams(action="click", ref="e1"), _context("scout"))
    assert key in service._call_locks  # pyright: ignore[reportPrivateUsage]


async def test_a_lock_with_a_call_waiting_is_kept_when_the_first_call_ends() -> None:
    """The safe half of dropping a lock: while one call is still queued, the lock stays, so
    a call arriving after the first ends still lines up behind the queued one instead of
    taking a fresh lock and running beside it (#2398 review).

    Killed by: src/uclone_x/browser/service.py :: if entry.users == 0 and key not in self._windows:
    Becomes: if key not in self._windows:
    """
    service = _Timed()
    tool = BrowserTool(service)
    context = _context("scout")

    first = asyncio.create_task(tool.run(BrowserParams(action="click", ref="e1"), context))
    second = asyncio.create_task(tool.run(BrowserParams(action="click", ref="e2"), context))
    await first  # the second is still waiting for the lock, or just got it
    third = asyncio.create_task(tool.run(BrowserParams(action="click", ref="e3"), context))
    await asyncio.gather(second, third)

    order = [(event, name) for event, _, name in service.log]
    assert order == [
        ("start", "click e1"),
        ("end", "click e1"),
        ("start", "click e2"),
        ("end", "click e2"),
        ("start", "click e3"),
        ("end", "click e3"),
    ]
