"""The session rail at a phone width and a desktop width, with the dock open and closed (#1062).

Until #1062 the rail was a 240px column of the workspace row at every width, opened on every
load, and forgot a collapse on reload. At a 400px window that left the conversation -- the one
surface with no condition on it -- 160px. Below 600px the rail is now drawn over the
conversation instead of beside it, a first run there starts with it closed, and the user's own
choice outlives a reload.

Every assertion here is on rendered geometry, or on which element the browser would hit at a
point, because both halves of the change are visual: a rail that stayed in the row with its
`data-overlay` set would pass any attribute check and still leave a 160px conversation.
"""

from __future__ import annotations

from typing import Any

import pytest
from playwright.async_api import Page, async_playwright

from tests.e2e.conftest import TWO_FRAMES, dock_locator

pytestmark = pytest.mark.e2e

#: Where the rail, the conversation and the dock are, and what is on top at the centre of the
#: rail and of the dock's close control. `null` for a region that is not rendered.
_LAYOUT = """() => {
  const box = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { left: Math.round(r.left), right: Math.round(r.right), width: Math.round(r.width) };
  };
  const onTop = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return hit !== null && (hit === el || el.contains(hit));
  };
  const rail = document.querySelector("[data-testid='chat-sidebar']");
  const dock = document.querySelector("[data-testid='artifacts-dock']");
  const close = document.querySelector("[data-testid='dock-close']");
  return {
    window: window.innerWidth,
    rail: box(rail),
    railOnTop: onTop(rail),
    main: box(document.querySelector('main')),
    dock: box(dock),
    closeOnTop: onTop(close),
    closeInside: close
      ? close.getBoundingClientRect().right <= window.innerWidth &&
        close.getBoundingClientRect().left >= 0
      : null,
    stored: localStorage.getItem('uclone-x.rail.open'),
  };
}"""


async def _layout(page: Page) -> dict[str, Any]:
    await page.evaluate(TWO_FRAMES)
    state: dict[str, Any] = await page.evaluate(_LAYOUT)
    return state


async def _load(page: Page, url: str) -> None:
    await page.goto(url, wait_until="commit")
    await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
    await page.locator("[data-testid='toggle-sidebar']").wait_for(state="visible")


async def _toggle_rail(page: Page, *, to: str) -> None:
    await page.click("[data-testid='toggle-sidebar']")
    await page.locator("[data-testid='chat-sidebar']").wait_for(
        state="visible" if to == "open" else "detached"
    )


async def _open_the_dock(page: Page) -> None:
    await page.click("[data-testid='toggle-dock']")
    dock = await dock_locator(page)
    await dock.wait_for(state="visible", timeout=5000)


@pytest.mark.asyncio
@pytest.mark.parametrize("dock", ["closed", "open"])
@pytest.mark.parametrize("width", [320, 400])
async def test_at_a_phone_width_the_rail_starts_closed_and_opens_over_the_conversation(
    ui_test_server: str, width: int, dock: str
) -> None:
    """320px and 400px: closed on a first run; opened, it covers the conversation's left.

    The conversation is the full window both before and after the rail opens -- that is what
    "overlays rather than divides" means on screen. The rail is the element a click at its
    centre reaches, so it is drawn over the conversation rather than under it; with the dock
    open, the dock's close control is still inside the window and still takes a click.

    #1173 added 320px, the narrowest window the conversation is laid out for. Every other
    320px case in the suite sets the rail's state before it measures, so none of them would
    notice a first run there that opened the rail. `browser.new_page` opens a new context, so
    the page starts with no stored choice, which is asserted rather than assumed: the default
    is decided by the width only when nothing is stored.

    Mutation-checked by hand, not as a kill declaration, because the lethality ratchet runs a
    browser test against the committed bundle without rebuilding it (see
    `test_room_conversation_layout_e2e.py`'s
    `test_a_160px_column_between_the_rail_and_the_dock_keeps_its_controls`, which records the
    same reason; that test was deleted with `test_chat_viewport_regression.py` by #1234 and
    this line pointed at nothing until #1227 restored it); rebuilt with
    `vite build`, this fails `[320-*]` and passes `[400-*]`:
    `Killed by:` frontend/src/lib/rail.ts :: `return !railOverlays(windowWidth);`
    becoming `return windowWidth <= 320 || !railOverlays(windowWidth);`
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": width, "height": 900})
            await _load(page, ui_test_server)
            if dock == "open":
                await _open_the_dock(page)

            before = await _layout(page)
            assert before["stored"] is None, f"not a first run: a choice is stored: {before}"
            assert before["rail"] is None, f"the rail opened on a first run at {width}px: {before}"
            assert before["main"] == {"left": 0, "right": width, "width": width}, before

            await _toggle_rail(page, to="open")
            after = await _layout(page)
            assert after["rail"] == {"left": 0, "right": 240, "width": 240}, after
            assert after["railOnTop"], f"the open rail is drawn under something: {after}"
            assert after["main"] == {"left": 0, "right": width, "width": width}, (
                f"the open rail took a share of the row at {width}px: {after}"
            )
            if dock == "open":
                assert after["dock"] is not None, after
                assert after["closeInside"], f"the dock's close control left the window: {after}"
                assert after["closeOnTop"], f"the rail covers the dock's close control: {after}"
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("dock", ["closed", "open"])
async def test_at_1280px_the_rail_starts_open_and_takes_its_share_of_the_row(
    ui_test_server: str, dock: str
) -> None:
    """1280px: open on a first run, and the conversation starts where the rail ends.

    With the dock open the conversation also ends where the dock begins: all three regions
    share the row and none is drawn over another.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1280, "height": 900})
            await _load(page, ui_test_server)
            if dock == "open":
                await _open_the_dock(page)

            state = await _layout(page)
            assert state["rail"] == {"left": 0, "right": 240, "width": 240}, state
            assert state["main"] is not None and state["main"]["left"] == 240, (
                f"the conversation does not start where the rail ends at 1280px: {state}"
            )
            if dock == "open":
                assert state["dock"] is not None, state
                assert state["main"]["right"] == state["dock"]["left"], state
                assert state["dock"]["right"] == 1280, state
                assert state["closeOnTop"], state
            else:
                assert state["main"]["right"] == 1280, state
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("width", "choice"), [(1280, "closed"), (400, "open")])
async def test_the_users_choice_survives_a_reload(
    ui_test_server: str, width: int, choice: str
) -> None:
    """A collapse at 1280px stays collapsed, and a rail opened at 400px stays open, on reload.

    Both are the opposite of what a first run at that width would choose, so a reload that fell
    back to the width would show the other state.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": width, "height": 900})
            await _load(page, ui_test_server)
            await _toggle_rail(page, to=choice)

            await page.reload(wait_until="commit")
            await _load(page, ui_test_server)
            state = await _layout(page)
            assert state["stored"] == ("true" if choice == "open" else "false"), state
            if choice == "open":
                assert state["rail"] is not None, f"the rail opened at {width}px closed on reload"
                assert state["main"]["left"] == 0, state
            else:
                assert state["rail"] is None, f"the rail closed at {width}px reopened on reload"
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_narrowing_the_window_moves_the_open_rail_over_the_conversation(
    ui_test_server: str,
) -> None:
    """1280px to 400px and back: the rail stays open throughout and changes only its position.

    A resize is not the user's choice, so it neither closes the rail nor writes anything.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1280, "height": 900})
            await _load(page, ui_test_server)

            await page.set_viewport_size({"width": 400, "height": 900})
            narrow = await _layout(page)
            assert narrow["rail"] == {"left": 0, "right": 240, "width": 240}, narrow
            assert narrow["main"] == {"left": 0, "right": 400, "width": 400}, narrow
            assert narrow["stored"] is None, narrow

            await page.set_viewport_size({"width": 1280, "height": 900})
            wide = await _layout(page)
            assert wide["main"] is not None and wide["main"]["left"] == 240, wide
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("width", "overlaid"), [(400, True), (1280, False)])
async def test_new_or_a_conversation_picked_from_the_overlaid_rail_closes_it(
    fresh_ui_server: str, width: int, overlaid: bool
) -> None:
    """#1062 review N3: navigating from the overlaid rail closes it; from the docked one, not.

    At 400px the open rail covers the left 240px of the conversation it has just opened, so
    New, and picking a conversation, close it as a phone-width drawer does. At 1280px the rail
    is in the row and covers nothing, and both leave it open. Neither writes the stored choice:
    the user did not choose it. The conversation itself must open either way, so the assertion
    is on the room's own surface as well as on the rail.

    Mutation-checked by hand (a kill declaration here could not be replayed: the ratchet does
    not rebuild the bundle): in `frontend/src/App.tsx`, deleting
    `if (railIsOverlaid) setIsSidebarOpen(false);` fails `[400-True]`, and making it
    unconditional fails `[1280-False]`. The vitest pair in `App.rail.test.tsx` declares both.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": width, "height": 900})
            await _load(page, fresh_ui_server)
            if overlaid:
                await _toggle_rail(page, to="open")
            rail = page.locator("[data-testid='chat-sidebar']")
            expected = "detached" if overlaid else "visible"

            await rail.locator("[data-testid='new-conversation-button']").click()
            await page.locator("[data-testid='room-conversation']").wait_for()
            await rail.wait_for(state=expected, timeout=5000)
            after_new = await _layout(page)
            assert (after_new["rail"] is None) == overlaid, f"after New: {after_new}"
            assert after_new["stored"] == ("true" if overlaid else None), after_new

            # Enough conversations that picking one that is not open is a navigation.
            #
            # The count is three, not two. A `fresh_ui_server` has none, and since #1208
            # the head will not render "no conversation open" at all: it opens the most
            # recent on load, and makes one when there is none. So the install already
            # holds one before this test clicks anything, and the two New clicks make the
            # second and the third.
            if overlaid:
                await _toggle_rail(page, to="open")
            await rail.locator("[data-testid='new-conversation-button']").click()
            await rail.wait_for(state=expected, timeout=5000)
            if overlaid:
                await _toggle_rail(page, to="open")
            renames = rail.locator("[data-testid^='rename-conversation-']")
            await renames.nth(2).wait_for()
            ids = [
                (await r.get_attribute("data-testid") or "").removeprefix("rename-conversation-")
                for r in await renames.all()
            ]
            assert len(ids) == 3, ids
            # The row that is not open: the list is by recency, so the first one made.
            await rail.locator(f"[data-testid='conversation-{ids[2]}']").click()
            await rail.wait_for(state=expected, timeout=5000)
            after_pick = await _layout(page)
            assert (after_pick["rail"] is None) == overlaid, f"after a pick: {after_pick}"
            await page.locator("[data-testid='room-conversation']").wait_for()
        finally:
            await browser.close()
