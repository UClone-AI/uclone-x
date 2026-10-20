"""Each guard `sign_in` keeps between a saved password and the wrong page, alone (#2388).

A real browser cannot time a page move to land exactly between two checks, so these run
against a scripted link: the page's address changes when the test says, a call hangs when
the test says, and every value the Core types is recorded. Each test fails when only its
own guard is removed (the kill declarations below).
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

import pytest

from uclone_x.browser.actions import IS_FOCUSED_JS
from uclone_x.browser.cdp import CdpEvent
from uclone_x.browser.service import (
    BOX_JS,
    VALUE_JS,
    BrowserService,
    BrowserTab,
    BrowserWindow,
    TabKey,
)
from uclone_x.browser.sign_ins import SavedSignIn
from uclone_x.browser.signin import CAPTCHA_JS, PAGE_PROBE_JS, field_expression

SECRET = "sentinel-pw-2Wq8-guarded"  # noqa: S105
SAVED = SavedSignIn("shop.example.com", "kenny", SECRET)
KEY: TabKey = ("room-1", "scout")
HOME = "https://shop.example.com/login"
ELSEWHERE = "https://elsewhere.example.net/login"


class ScriptedPage:
    """A sign-in page as CDP shows it: its address, its fields, and what was typed."""

    def __init__(self) -> None:
        self.where: str | None = HOME
        self.fields = {"user": "user-1", "password": "pw-1", "code": None}
        self.typed: list[str] = []
        self.focused = True
        self.hang_focus_on: str | None = None
        self.hang_after_typing: str | None = None
        self.released = asyncio.Event()
        self.after_typing: dict[str, Callable[[], None]] = {}
        self.queues: list[asyncio.Queue[CdpEvent]] = []

    async def open_tab(self, url: str) -> str:
        return "tab-1"

    async def send(
        self, tab: str, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        p = dict(params or {})
        if method == "Runtime.evaluate":
            return self._evaluate(str(p.get("expression", "")))
        if method == "Runtime.callFunctionOn":
            fn = str(p.get("functionDeclaration", ""))
            if fn == VALUE_JS:
                return {"result": {"value": ""}}
            if fn == BOX_JS:
                return {"result": {"value": {"x": 1, "y": 2, "width": 3, "height": 4}}}
            if fn == IS_FOCUSED_JS:
                return {"result": {"value": self.focused}}
            return {"result": {"value": None}}
        if method == "DOM.focus" and p.get("objectId") == self.hang_focus_on:
            await self.released.wait()
        if method == "Input.insertText":
            text = str(p.get("text", ""))
            self.typed.append(text)
            hook = self.after_typing.pop(text, None)
            if hook is not None:
                hook()
            if text == self.hang_after_typing:
                await self.released.wait()
        if method == "Page.getFrameTree":
            return {"frameTree": {"frame": {"id": "F", "loaderId": "L1"}}}
        if method == "Accessibility.getFullAXTree":
            return {"nodes": []}
        return {}

    def _evaluate(self, expression: str) -> dict[str, Any]:
        if expression == "location.href":
            return {"result": {"value": self.where}} if self.where else {"result": {}}
        if expression == PAGE_PROBE_JS:
            return {"result": {"value": [self.where or "", "Sign in", 200, True]}}
        if expression == CAPTCHA_JS:
            return {"result": {"value": False}}
        for role, handle in self.fields.items():
            if expression == field_expression(role):
                if handle is None:
                    return {"result": {"type": "object", "subtype": "null"}}
                return {"result": {"type": "object", "subtype": "node", "objectId": handle}}
        return {"result": {"value": None}}

    def subscribe(self, tab: str, *, frames: bool = False) -> asyncio.Queue[CdpEvent]:
        queue: asyncio.Queue[CdpEvent] = asyncio.Queue()
        self.queues.append(queue)
        return queue

    def unsubscribe(self, tab: str, queue: asyncio.Queue[CdpEvent]) -> None:
        return

    async def adopt_tab(self, opener: str, tab: str) -> str:
        return tab

    async def close_tab(self, tab: str) -> None:
        return

    async def close(self) -> None:
        return


def _service(page: ScriptedPage) -> tuple[BrowserService, BrowserTab]:
    async def _factory() -> Any:
        return page

    service = BrowserService(_factory, sign_ins=lambda: [SAVED])
    service._link = page  # pyright: ignore[reportPrivateUsage]
    tab = BrowserTab(target="tab-1", events=asyncio.Queue(), url=HOME, loader="L1")
    service._windows[KEY] = BrowserWindow(tabs=[tab])  # pyright: ignore[reportPrivateUsage]
    return service, tab


async def _signed_in_or_handed_over(service: BrowserService) -> tuple[dict[str, Any], str]:
    """Run `sign_in`, playing the person who gives the tab back. The result and the ask."""
    running = asyncio.create_task(service.sign_in(KEY))
    asked = ""
    for _ in range(400):
        if running.done():
            break
        state = service.live_state(KEY[0]).get("asked_user")
        if state:
            asked = str(state["message"])
            service.give_back(*KEY)
            break
        await asyncio.sleep(0.01)
    return await asyncio.wait_for(running, 10), asked


async def test_a_page_that_moved_host_since_the_last_step_is_named_as_moved() -> None:
    """Typing done, the page is now on another host showing a sign-in form again: the step
    check says so, instead of blaming the saved site for refusing the sign-in.

    Killed by: src/uclone_x/browser/service.py :: moved = self._refuse_page(saved, await self._here(link, tab))
    Becomes: moved = None
    """
    page = ScriptedPage()
    page.after_typing[SECRET] = lambda: setattr(page, "where", ELSEWHERE)
    service, _ = _service(page)

    result, asked = await _signed_in_or_handed_over(service)

    assert "elsewhere.example.net" in asked
    assert "did not accept" not in result["message"]


async def test_a_page_that_moves_between_finding_a_field_and_typing_gets_nothing() -> None:
    """The username is typed on the saved host; the page then moves before the password.

    Killed by: src/uclone_x/browser/service.py :: refused = self._refuse_page(saved, await self._here(link, tab))
    Becomes: refused = None
    """
    page = ScriptedPage()
    page.after_typing["kenny"] = lambda: setattr(page, "where", ELSEWHERE)
    service, _ = _service(page)

    result, asked = await _signed_in_or_handed_over(service)

    assert page.typed == ["kenny"]
    assert "elsewhere.example.net" in asked
    assert "Signed in" not in result["message"]


async def test_a_fill_stalled_by_a_dialog_is_cancelled_and_never_types_later() -> None:
    """The fill hangs before the password, a dialog opens, the page moves on: once the stall
    lifts, the abandoned fill must not type the password into whatever page is there now.

    Killed by: src/uclone_x/browser/service.py :: _abandon(tab.pending)
    Becomes: pass
    """
    page = ScriptedPage()
    page.hang_focus_on = "pw-1"
    service, tab = _service(page)

    async def dialog_once_stalled() -> None:
        while not page.typed:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        tab.events.put_nowait(
            CdpEvent("Page.javascriptDialogOpening", {"type": "alert", "message": "Hi"}, "S")
        )

    opener = asyncio.create_task(dialog_once_stalled())
    result, asked = await _signed_in_or_handed_over(service)
    await opener
    page.where = ELSEWHERE
    page.released.set()
    for _ in range(20):
        await asyncio.sleep(0.01)

    assert SECRET not in page.typed
    assert "dialog" in asked
    assert tab.pending is None
    assert "Signed in" not in result["message"]


async def test_an_address_that_cannot_be_read_is_not_the_saved_site() -> None:
    """Killed by: src/uclone_x/browser/service.py :: return href if isinstance(href, str) else ""
    Becomes: return href if isinstance(href, str) else tab.url
    """
    page = ScriptedPage()
    service, _ = _service(page)
    page.where = None

    _, asked = await _signed_in_or_handed_over(service)

    assert page.typed == []
    assert asked


async def test_focus_that_left_the_field_gets_no_password() -> None:
    """Focus moved off the field (into a frame, say) between focusing it and typing: the
    text would land wherever focus is, so nothing is typed.

    Killed by: src/uclone_x/browser/service.py :: if await self._call(link, tab, element, IS_FOCUSED_JS) is not True:
    Becomes: if False:
    """
    page = ScriptedPage()
    page.focused = False
    service, _ = _service(page)

    _, asked = await _signed_in_or_handed_over(service)

    assert page.typed == []
    assert "focus" in asked.lower()


async def test_an_abandoned_fill_leaves_no_unretrieved_exception() -> None:
    """A pending fill that already failed is abandoned without asyncio reporting a lost
    exception (#2388 item 4)."""
    from uclone_x.browser.service import _abandon  # pyright: ignore[reportPrivateUsage]

    lost: list[Mapping[str, Any]] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _, context: lost.append(context))
    try:

        async def failing() -> None:
            raise RuntimeError("the page moved")

        task = asyncio.ensure_future(failing())
        await asyncio.sleep(0)
        _abandon(task)
        hanging = asyncio.ensure_future(asyncio.sleep(10))
        _abandon(hanging)
        await asyncio.sleep(0)
        del task, hanging
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(None)
    assert lost == []


async def test_a_fill_that_fails_as_its_wait_times_out_is_not_reported_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The race #2388 item 4 is about, through `sign_in`. The fill is slow, so `_perform`
    waits for it. In the same instant the wait times out, the fill ends with `_PageMoved`
    (the page moved before the password). `sign_in` then holds a fill that is already over,
    and no exception may be reported as lost.

    No kill is declared on `_abandon(tab.pending)`, because none exists through `sign_in`
    (#2390). `_perform` waits through `asyncio.shield`, and a shield whose wait timed out
    reads the inner task's exception when the task ends. The one other way to a pending
    fill is a dialog, which leaves the task running, so cancelling it ends it with
    `CancelledError`. Reverting the call site to `tab.pending.cancel()` therefore loses
    nothing either, and this test passes both ways. `_abandon` itself is pinned by
    `test_an_abandoned_fill_leaves_no_unretrieved_exception`.
    """
    import uclone_x.browser.service as service_module

    page = ScriptedPage()
    page.after_typing["kenny"] = lambda: setattr(page, "where", ELSEWHERE)
    page.hang_after_typing = "kenny"
    service, _ = _service(page)
    real_wait_for = asyncio.wait_for

    async def times_out_as_the_fill_ends(awaitable: Any, timeout: float | None) -> Any:
        # Only `_perform`'s wait on the fill (a shielded future, its load timeout); every
        # other wait, the settle loop's included, is the real one.
        is_fill_wait = (
            isinstance(awaitable, asyncio.Future) and timeout == service_module.LOAD_TIMEOUT_S
        )
        if page.released.is_set() or not is_fill_wait:
            return await real_wait_for(cast(Awaitable[Any], awaitable), timeout)
        # The fill is let go, finishes (raising), and the wait reports a timeout anyway.
        page.released.set()
        with contextlib.suppress(Exception):
            await awaitable
        raise TimeoutError

    monkeypatch.setattr(service_module.asyncio, "wait_for", times_out_as_the_fill_ends)
    lost: list[Mapping[str, Any]] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _, context: lost.append(context))
    try:
        result, asked = await _signed_in_or_handed_over(service)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(None)

    assert page.typed == ["kenny"]
    assert "stopped responding" in asked
    assert "Signed in" not in result["message"]
    assert [c.get("message") for c in lost] == []


@pytest.fixture(autouse=True)
def _quiet_settle(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """The page is scripted, so its quiet period need not be waited out at full length."""
    import uclone_x.browser.service as service_module

    original = service_module.settle

    async def quick(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("quiet", 0.01)
        kwargs.setdefault("idle_cap", 0.2)
        kwargs.setdefault("load_timeout", 0.3)
        return await original(*args, **kwargs)

    monkeypatch.setattr(service_module, "settle", quick)
