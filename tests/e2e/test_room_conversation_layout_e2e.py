"""Layout regression tests for the conversation column, on the surface that survives (#1208).

`test_chat_viewport_regression.py` guards these defects on `PlaygroundTab`, which the
retirement deletes. Its cases were written against a different component tree -- `chat-input`,
`send-button`, `messages-container`, and a markdown renderer emitting `.prose` -- so none of
them runs against `RoomConversation`, and the surface that is being kept has never been held
to any of them. This module ports the column's own cases across **before** the deletion, so
that what the retirement removes is a second implementation and not the coverage.

Content is decided by scripting the model (`scripted_reply_ui_server`) rather than by
answering the head's request with `page.route`. The room suites mock nothing at the HTTP
layer on purpose: `test_room_chat_e2e` exists because a fixture whose shape disagreed with
the real payload let a blank screen through a green gate. Scripting the model keeps route,
service, transcript and render real, and still decides the words.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import pytest
from playwright.async_api import Page, Route, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from tests.e2e.conftest import UIServerFactory, dock_locator, set_the_rail

pytestmark = pytest.mark.e2e

#: A token with no break opportunity in it (#1007). Long enough to run past any column here.
LONG_TOKEN = "0123456789abcdef" * 40

#: Since #1062 the narrowest conversation column with the dock closed is the narrowest window
#: itself: below 600px the rail is drawn over the conversation rather than beside it.
NARROWEST_WINDOW = 320

#: The window width at and above which the rail takes a share of the row instead of covering
#: the conversation (#1062, `RAIL_OVERLAY_BELOW_PX` in `frontend/src/lib/rail.ts`). At exactly
#: this width the rail leaves the narrowest column it ever leaves beside itself.
RAIL_OVERLAY_BELOW_PX = 600

#: A served-by tag as a locally pulled model reports it, with a long unbroken run in it.
#:
#: The run is what makes this a measurement. A name spelled the way the hub writes them --
#: `Qwen3-Coder-30B-A3B-Instruct` -- breaks after every hyphen on its own, so a column carrying
#: it cannot overrun whatever the head does, and a case built on one is green before anything
#: is written. Measured: with the hyphenated name, removing `min-w-0 overflow-hidden` from the
#: header strip changed nothing this file could see.
LONG_MODEL = "hf.co/unsloth/" + "Qwen3CoderA3BInstructGGUFQ4KMlocaltune" * 2

#: A title with the same property, for the same reason: `RoomConversation`'s `<h2>` is what
#: holds it to the column, and a title that breaks on its own would not ask it to.
LONG_TITLE = "Conversation" + LONG_TOKEN[:96]

#: Two animation frames, after which layout the browser was asked to recompute is readable.
TWO_FRAMES = "() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))"


async def _create_room(page: Page, base_url: str, title: str, agents: list[str]) -> str:
    """Seat a conversation through the API, as any other client would."""
    response = await page.request.post(
        f"{base_url}/api/rooms",
        data={"title": title, "agent_ids": agents},
    )
    assert response.ok, await response.text()
    body: dict[str, Any] = await response.json()
    return str(body["room_id"])


async def _open_room(page: Page, base_url: str, *, title: str = "Layout") -> None:
    """Seat a conversation and leave it on screen, at a width where the rail is in the row.

    The conversation is reached at 1280px and the caller narrows afterwards, rather than the
    page being opened narrow. Below 600px a first run starts with the rail closed (#1062), so
    the row that opens the conversation is not on screen at all -- and it is the *column at
    that width* these cases are about, which is the state either route leaves the page in.
    """
    room_id = await _create_room(page, base_url, title, ["scout"])
    await page.set_viewport_size({"width": 1280, "height": 900})
    await page.goto(base_url, wait_until="commit")
    await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
    await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)


async def _say(page: Page, words: str) -> None:
    """Send one message and wait for the reply to be on screen."""
    await page.fill("[data-testid='room-composer']", words)
    await page.click("[data-testid='send-message']")
    # Rows 1 and 2 are the joins, 3 is this message and 4 is the reply to it.
    await page.wait_for_selector("[data-testid='row-4']", timeout=20000)


async def _open_room_and_say(page: Page, base_url: str, words: str, *, then_width: int) -> None:
    """`_open_room`, one message, and then the window narrowed to `then_width`."""
    await _open_room(page, base_url)
    await _say(page, words)
    await page.set_viewport_size({"width": then_width, "height": 900})


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [NARROWEST_WINDOW, 400, 1280])
async def test_a_long_unbroken_token_wraps_inside_its_bubble(
    scripted_reply_ui_server: UIServerFactory, width: int
) -> None:
    """#1007, on the conversation: a token with no break opportunity stays inside its bubble.

    Both bubble kinds carry it -- what the user sent and what the agent replied -- because a
    rule that holds for one and not the other still puts text past the edge of the column.
    """
    base_url = scripted_reply_ui_server(f"Read {LONG_TOKEN}")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room_and_say(page, base_url, f"Please read {LONG_TOKEN}", then_width=width)

            transcript = page.locator("[data-testid='transcript']")
            carrying = await transcript.evaluate(
                """(container) => [...container.querySelectorAll('p')]
                    .filter((el) => (el.textContent || '').includes('0123456789abcdef')).length"""
            )
            assert carrying == 2, f"expected the prompt and the reply, found {carrying}"

            overflowing = await transcript.evaluate(
                """(container) => {
                    const column = container.getBoundingClientRect().right;
                    return [...container.querySelectorAll('p')]
                        .filter((el) => (el.textContent || '').includes('0123456789abcdef'))
                        .filter((el) => el.scrollWidth > el.clientWidth + 1
                            || el.getBoundingClientRect().right > column + 1)
                        .map((el) => `${el.scrollWidth}>${el.clientWidth}, right `
                            + `${Math.round(el.getBoundingClientRect().right)}>${Math.round(column)}`);
                }"""
            )
            assert overflowing == [], f"text wider than its bubble at {width}px: {overflowing}"
        finally:
            await browser.close()


_SIDEWAYS_SCROLL = """(container) => ({
    scrollWidth: container.scrollWidth,
    clientWidth: container.clientWidth,
    past: [...container.querySelectorAll('*')]
        .filter((el) => el.getBoundingClientRect().right > container.getBoundingClientRect().right + 1)
        .slice(0, 3)
        .map((el) => `${el.tagName} "${(el.textContent || '').slice(0, 30)}"`),
})"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "rail"),
    [(400, "open"), (400, "closed"), (NARROWEST_WINDOW, "closed"), (NARROWEST_WINDOW, "open")],
)
async def test_the_conversation_never_scrolls_sideways_at_a_narrow_window(
    scripted_reply_ui_server: UIServerFactory, width: int, rail: str
) -> None:
    """#1010, on the conversation: it does not scroll sideways, rail open or closed.

    Measured on the whole surface rather than the transcript alone, because the model name
    that overruns a narrow column does not sit in the transcript here. In a 1:1 whose model
    has never changed, `attributionLines` prints no line on the rows at all -- one value true
    of every row beneath it goes in the header instead -- so the element carrying a locally
    pulled model's name is the header's, and a case that watched only the transcript would be
    green whatever the header did.

    Checked twice on one conversation: with the join rows alone, and again once a reply has
    put the served-by name on screen. Since #1062 the open rail at these widths is drawn
    *over* the conversation rather than beside it, so the column is the window's width in
    both rail cases -- and the rail is asserted still in the state it was put in after the
    turn, so an open case measures the column with the rail over it rather than one the turn
    closed it on.

    The model name is the fixture's input rather than a route's: this suite answers no request
    the head makes, and a connector reporting a long name produces the same attribution by the
    path a real one would.
    """
    base_url = scripted_reply_ui_server("Hello there.", model=LONG_MODEL)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room(page, base_url, title=LONG_TITLE)
            await page.set_viewport_size({"width": width, "height": 900})
            await set_the_rail(page, rail)

            column = page.locator("[data-testid='room-conversation']")
            await page.evaluate(TWO_FRAMES)
            joined = await column.evaluate(_SIDEWAYS_SCROLL)
            assert joined["scrollWidth"] <= joined["clientWidth"], (
                f"the joins alone, {width}px, rail {rail}: {joined}"
            )

            await _say(page, "hi")
            await page.locator("[data-testid='header-attribution']").wait_for()

            rail_is_open = await page.locator("[data-testid='chat-sidebar']").count() == 1
            assert rail_is_open == (rail == "open"), f"the turn changed the rail from {rail}"
            await page.evaluate(TWO_FRAMES)
            replied = await column.evaluate(_SIDEWAYS_SCROLL)
            assert replied["scrollWidth"] <= replied["clientWidth"], (
                f"once a model is named, {width}px, rail {rail}: {replied}"
            )

            # The header clips the name rather than carrying it in a `title`, so what keeps a
            # truncated tag naming what served the turn (FR-13.4) is the reply's own `why`
            # control. That route is the assertion: a name the column cut in half is only
            # attribution if it is still readable somewhere, and this is the somewhere.
            #
            # The somewhere moved. `why` used to unfold the record inside the row, under
            # `why-detail-<seq>`; it now opens the Turn surface on the dock, where the model
            # is `turn-detail-served`. What this case is about is unchanged -- the route from
            # a clipped header to a readable name -- so it follows the surface rather than
            # keeping a control that no longer exists.
            #
            # The rail is put away first, and only now: an open overlay covers the control,
            # which is what an overlay is for and not a defect to assert about. The widths
            # above were measured with the rail in the state the case names, which is the
            # part of this test the rail is material to.
            await set_the_rail(page, "closed")
            await page.click("[data-testid='why-4']")
            await page.locator("[data-testid='turn-detail']").wait_for()
            disclosed = await page.locator("[data-testid='turn-detail-served']").inner_text()
            assert LONG_MODEL in disclosed, (
                f"the reply does not disclose the model that served it: {disclosed!r}"
            )
        finally:
            await browser.close()


#: The conversation's own controls. Each must lie inside the column and be what a click at its
#: centre reaches. `clear-history` is here because it is the control #1015 found running off the
#: column's edge.
#:
#: `room-model-select` is not, because it is not on screen: the room route carries no model, so
#: the picker is held behind `A_ROOM_SEAT_CAN_CARRY_A_MODEL` until #1235 settles whether a seat
#: should carry one. It was the widest control in the header -- a picker is as wide as its
#: longest option -- so whoever brings it back brings this measurement back with it.
_COLUMN_CONTROLS = [
    "add-someone",
    "clear-history",
    "room-composer",
    "send-message",
]

_UNREACHABLE = """(ids) => {
    const column = document
        .querySelector("[data-testid='room-conversation']").getBoundingClientRect();
    const problems = [];
    for (const id of ids) {
        const el = document.querySelector(`[data-testid='${id}']`);
        if (!el) {
            problems.push(`${id}: not rendered`);
            continue;
        }
        const r = el.getBoundingClientRect();
        const box = `x ${r.left.toFixed(1)}-${r.right.toFixed(1)}`
            + ` y ${r.top.toFixed(1)}-${r.bottom.toFixed(1)}`;
        if (r.width < 1 || r.height < 1) problems.push(`${id}: no size, ${box}`);
        if (r.left < column.left - 0.5 || r.right > column.right + 0.5
            || r.top < 0 || r.bottom > window.innerHeight + 0.5) {
            problems.push(`${id}: ${box} is outside the column x ${column.left}-${column.right}`);
        }
        const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        if (!hit || !el.contains(hit)) {
            const what = hit ? hit.dataset.testid || hit.tagName : 'nothing';
            problems.push(`${id}: a click at its centre reaches ${what}`);
        }
    }
    return problems;
}"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "rail"),
    [
        (NARROWEST_WINDOW, "closed"),
        (400, "closed"),
        (RAIL_OVERLAY_BELOW_PX, "open"),
        (768, "open"),
        (1280, "open"),
    ],
)
async def test_the_header_and_composer_controls_are_on_screen_and_uncovered(
    scripted_reply_ui_server: UIServerFactory, width: int, rail: str
) -> None:
    """#1015, on the conversation: every header and composer control takes a click where it is.

    The column this guards is narrower than any the playground's copy of this case measured,
    because the surface that survives adapts to it differently: `PlaygroundTab` carries the
    `.column-scope` container query, whose rules wrap its header, drop its separators and take
    its button labels down to icons under 320px. `RoomConversation` carries none of them -- it
    wraps its header with `flex-wrap` and marks the buttons `shrink-0 whitespace-nowrap` -- so
    what the controls do when the column runs out of room has never been measured here at all.

    The rail cases are the ones #1062 leaves meaningful: at 320px and 400px the rail covers
    the column, and covering what is under it is what an overlay is for, so those are measured
    closed; from 600px it is in the row and the column is what is left beside it.

    Nothing is sent. The prompt is typed only so Send is enabled. The long model name this
    case used to be given is gone with the picker it was measuring (see `_COLUMN_CONTROLS`):
    with nothing sent there is no served-by line either, so it reached no element on screen.
    """
    base_url = scripted_reply_ui_server("Hello there.")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room(page, base_url)
            await page.set_viewport_size({"width": width, "height": 900})
            await set_the_rail(page, rail)
            await page.fill("[data-testid='room-composer']", "hi")
            await page.wait_for_selector(
                "[data-testid='send-message']:not([disabled])", timeout=10000
            )
            await page.evaluate(TWO_FRAMES)

            problems = await page.evaluate(_UNREACHABLE, _COLUMN_CONTROLS)
            assert problems == [], f"{width}px, rail {rail}: {problems}"
        finally:
            await browser.close()


#: The four windows #339 was written against, from a phone to a desktop.
VIEWPORTS = [
    ("mobile", 375, 667),
    ("tablet", 768, 1024),
    ("laptop", 1280, 800),
    ("desktop", 1920, 1080),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "width", "height"), VIEWPORTS)
async def test_the_composer_is_pinned_and_whole_on_every_viewport(
    scripted_reply_ui_server: UIServerFactory, name: str, width: int, height: int
) -> None:
    """#339, on the conversation: the text box is on screen, whole, and takes typing.

    The first thing a first-time user does with this surface is type in it, so this is the
    one property whose absence makes the conversation unusable rather than untidy -- and the
    playground's copy of it is the only place it has ever been asserted.

    The rail's own first state is not asserted here, although the playground's copy of this
    case does. It cannot be: reaching a conversation at a phone's width is what #1062 makes
    impossible -- the row that opens it is behind a rail that starts closed -- so this opens
    wide and narrows, and the rail it then carries is one a click opened, not the state a
    first run would have. That question belongs to `test_rail_responsive_e2e.py`, which loads
    the page at the width it is asking about. What is asserted instead is the harder version
    of the same worry: the box is whole and reachable *with* the rail over it.

    Nothing is sent. The reply is scripted so the server has one to give, and the prompt is
    typed only so Send has to decide whether it is enabled.
    """
    base_url = scripted_reply_ui_server("Hello there.")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda err: errors.append(str(err)))

            await _open_room(page, base_url)
            await page.set_viewport_size({"width": width, "height": height})
            await page.wait_for_selector("[data-testid='room-composer']", timeout=10000)

            composer = page.locator("[data-testid='room-composer']")
            assert await composer.is_visible(), f"{name}: the text box is not on screen"
            assert await composer.is_editable(), f"{name}: the text box does not take typing"
            box = await composer.bounding_box()
            assert box is not None, f"{name}: the text box has no box"
            assert box["y"] >= 0, f"{name}: the text box starts above the window ({box['y']})"
            assert box["y"] + box["height"] <= height + 2, (
                f"{name}: the text box runs past the bottom of the window "
                f"({box['y'] + box['height']} > {height})"
            )
            assert box["height"] > 30, f"{name}: the text box is {box['height']}px tall"

            send = page.locator("[data-testid='send-message']")
            assert await send.is_visible(), f"{name}: Send is not on screen"
            assert await send.is_disabled(), f"{name}: Send offers to send nothing"
            await composer.fill(f"Testing input on {name} ({width}x{height})")
            await page.wait_for_selector(
                "[data-testid='send-message']:not([disabled])", timeout=10000
            )

            assert errors == [], f"{name}: uncaught page errors: {errors}"
        finally:
            await browser.close()


#: A window where the rail and the dock are both seated in the row: 920 - 240 - 520 leaves the
#: conversation 160px, the width #1015 was written against and #1227 found orphaned.
RAIL_AND_DOCK_WINDOW = 920

#: What decides whether the dock is open, which the column cannot show. Each must be inside the
#: window and be what a click at its centre reaches.
_DOCK_CONTROLS_OUTSIDE_THE_COLUMN = """(ids) => ids.flatMap((id) => {
    const el = document.querySelector(`[data-testid='${id}']`);
    if (!el) return [`${id}: not rendered`];
    const r = el.getBoundingClientRect();
    const problems = [];
    if (r.left < 0 || r.right > window.innerWidth + 0.5) {
        problems.push(`${id}: x ${r.left}-${r.right} is outside the window`);
    }
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    if (!hit || !el.contains(hit)) {
        problems.push(`${id}: a click at its centre reaches ${hit ? hit.tagName : 'nothing'}`);
    }
    return problems;
})"""


@pytest.mark.asyncio
async def test_a_160px_column_between_the_rail_and_the_dock_keeps_its_controls(
    scripted_reply_ui_server: UIServerFactory,
) -> None:
    """#1227: the column adapts to its **container**, which a wide window still narrows.

    Every other case in this file resizes the *viewport*, and the UI skill's hard rule is about
    the container. The two come apart here, and only here: the rail takes 240px from the row
    and the dock seats itself whenever what is left is wider than its own width
    (`ArtifactsDock`'s `overlaid`), keeping no floor for the conversation -- the hole
    `lib/rail.ts` records in prose. At 920px with both open the column is therefore
    920 - 240 - 520 = 160px while every Tailwind viewport breakpoint still reads "tablet or
    wider", so a header that answers the viewport does not adapt at all.

    This case existed against `PlaygroundTab`, in `test_chat_viewport_regression.py`, and
    #1234 deleted that file; the port that replaced it (#1224, this module) carried the
    viewport cases across and not this one, which is how #1227's `column-*` rules came to be
    orphaned with nothing measuring what their absence cost. `test_rail_responsive_e2e.py`
    still names this test in a docstring, which is how it was found again.

    Measured on `193fafee`, before `RoomConversation` took the scope: `clear-history` rendered
    at x 377-441 against a column ending at 400, and a click at its centre reached the dock
    drawn over it. The assertion is that consequence -- can this control be reached -- rather
    than a class or a `scrollWidth`, which #1241 showed can report success for clipped and
    unreachable content either way.

    The premise is asserted first, so a change to either region's width that stops this column
    being 160px fails here rather than passing over a comfortable one. With the dock open the
    floating opener is not rendered, so it is asserted absent and the controls that do decide
    the dock's state are checked instead.

    Mutation-checked by hand rather than as a kill declaration, because the lethality ratchet
    runs a browser test against the **committed bundle** without rebuilding it, so a
    declaration naming `frontend/src` reads as escaped there (measured, and the reason
    `test_rail_responsive_e2e.py` records its own the same way). Each was run with a real
    `vite build`, and each fails this case and no other in this file:
    `Killed by:` frontend/src/index.css :: `@container (max-width: 320px) {` becoming
    `@container (max-width: 100px) {`
    `Killed by:` frontend/src/components/rooms/RoomConversation.tsx ::
    `column-scope flex flex-col h-full min-h-0` becoming `flex flex-col h-full min-h-0`

    The first is the sharper of the two: it leaves `container-type: inline-size` in place and
    only stops the query firing, so what it proves is that the *query* carries the column,
    not the containment the class also brings.
    """
    base_url = scripted_reply_ui_server("Hello there.")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            room_id = await _create_room(page, base_url, "Layout", ["scout"])
            await page.set_viewport_size({"width": RAIL_AND_DOCK_WINDOW, "height": 900})
            await page.goto(base_url, wait_until="commit")
            await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
            await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)
            await set_the_rail(page, "open")
            await page.click("[data-testid='toggle-dock']")
            await (await dock_locator(page)).wait_for(state="visible")
            await page.fill("[data-testid='room-composer']", "hi")
            await page.wait_for_selector(
                "[data-testid='send-message']:not([disabled])", timeout=10000
            )
            await page.evaluate(TWO_FRAMES)

            column = await page.locator("[data-testid='room-conversation']").bounding_box()
            assert column is not None
            assert (round(column["x"]), round(column["width"])) == (240, 160), (
                f"the premise: rail and dock seated, a 160px column at x 240; got {column}"
            )
            assert await page.locator("[data-testid='floating-dock-btn']").count() == 0

            problems = await page.evaluate(_UNREACHABLE, _COLUMN_CONTROLS)
            problems += await page.evaluate(
                _DOCK_CONTROLS_OUTSIDE_THE_COLUMN, ["toggle-dock", "dock-close"]
            )
            assert problems == [], f"{RAIL_AND_DOCK_WINDOW}px, rail and dock open: {problems}"
        finally:
            await browser.close()


#: A reply with enough paragraphs to overflow the transcript at any window this file opens.
#: The transcript has to be genuinely scrollable before "it did not move while the transcript
#: scrolled" is a measurement rather than a tautology over a column with nothing to scroll.
TALL_REPLY = "\n\n".join(
    f"Paragraph {n} of a reply long enough to give the transcript something to scroll."
    for n in range(1, 61)
)

#: The pause between the reply's streamed words in the composer case (#1380). The reply is
#: about 840 words, so it streams for 8 seconds or so, and overflows the transcript within
#: the first two: several seconds of words still arriving after the reader has scrolled up.
SLOW_CHUNK_S = 0.01

#: The transcript is overflowing and the reply is still streaming into it.
_STREAMING_AND_OVERFLOWING = """() => {
    const t = document.querySelector("[data-testid='transcript']");
    const live = document.querySelector("[data-testid='live-turn']");
    return Boolean(t && live) && t.scrollHeight > t.clientHeight + 200;
}"""

#: The words in flight have grown past `before` characters, and are still in flight.
_LIVE_TEXT_GREW_PAST = """(before) => {
    const live = document.querySelector("[data-testid='live-turn']");
    return Boolean(live) && (live.textContent || '').length > before + 200;
}"""

#: The transcript is scrolled to its bottom edge, within a pixel's rounding.
_AT_THE_BOTTOM = """() => {
    const t = document.querySelector("[data-testid='transcript']");
    return Boolean(t) && t.scrollHeight - t.scrollTop - t.clientHeight <= 2;
}"""


@pytest.mark.asyncio
async def test_the_composer_does_not_move_while_the_transcript_scrolls(
    scripted_reply_ui_server: UIServerFactory,
) -> None:
    """#1236: scrolling the transcript leaves the composer's box exactly where it was, and a
    reader who scrolled up mid-reply stays where they scrolled to (#1380).

    `test_chat_viewport_regression.py::test_multi_turn_scroll_and_composer_pinned_immobility`
    measured this on `PlaygroundTab` and #1234 deleted it with that surface; the port (#1224,
    this module) carried the composer's *load-time* case across as
    `test_the_composer_is_pinned_and_whole_on_every_viewport` and not this one. The two are
    not the same property: that case asks whether the box is on screen and whole when the
    page settles, which a column that grows with its content still satisfies until something
    fills it. This one asks whether it stays there once something does.

    What pins it is that the transcript is the scroller and the composer is its sibling:
    `flex-1 overflow-y-auto min-h-0` inside `flex flex-col h-full min-h-0`. The scroller is
    the whole of it, so the premise is asserted first and the failure that follows a broken
    pin is "the transcript never overflowed", not "the composer moved": a column whose
    transcript is no longer bounded and scrollable has nothing left to hold the composer
    still. The assertions on the box are what say where "still" is once the scrolling is real.

    Measured, against the belief this was written on: the `min-h-0` on the transcript is
    **not** load-bearing. Removing it alone leaves every case in this file green, because a
    flex item whose `overflow` is not `visible` already has an automatic minimum size of
    zero -- the `overflow-y-auto` on the same element is doing that job.

    #1380: the scroll-up happens **while the reply is still streaming**. Before, the case
    scrolled after `row-4` had landed, and whether any deltas were left to pull the reader
    back down was down to how far the stream lagged the row -- #1378's fix for exactly that
    pull could be reverted and this case would often stay green. Now the model streams slowly
    (`SLOW_CHUNK_S`), the reader scrolls up once the reply overflows and while `live-turn` is
    still on screen, and the position is read only after the words in flight have grown by
    a measured amount, so "stayed at the top" is asserted over deltas that really arrived.
    The same reader is then offered "Jump to latest", which stays offered, not acted on,
    until pressed, and pressing it takes them to the bottom.

    One case, not a viewport sweep. The mechanism is one flex rule and does not vary with
    the window; four near-copies of it would cost four browsers to say the same thing once.

    Mutation-checked by hand rather than as a kill declaration, for the reason
    `test_a_160px_column_between_the_rail_and_the_dock_keeps_its_controls` above records: the
    lethality ratchet runs a browser test against the **committed bundle** in
    `src/uclone_x/ui_static` and never rebuilds it, so a declaration naming `frontend/src`
    reads as escaped there and is refused outright by the kill-declaration fitness check
    since #1245. Each was run with a real `vite build`, and each fails this case:
    `Killed by:` frontend/src/components/rooms/RoomConversation.tsx ::
    `flex-1 overflow-y-auto min-h-0 px-4` becoming
    `flex-1 min-h-0 px-4` (the transcript never overflows)
    `Killed by:` frontend/src/components/rooms/RoomConversation.tsx ::
    `{unseenBelow ? (` becoming `{unseenBelow && false ? (` (no "Jump to latest")
    and #1378's own change to this component reverted wholesale, which fails the
    "stayed at the top" assertion: the stream scrolls the reader straight back down.
    """
    base_url = scripted_reply_ui_server(TALL_REPLY, streaming_chunk_delay=SLOW_CHUNK_S)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room(page, base_url)
            await page.fill("[data-testid='room-composer']", "Say something long.")
            await page.click("[data-testid='send-message']")
            await page.wait_for_function(_STREAMING_AND_OVERFLOWING, timeout=20000)

            window_height = await page.evaluate("() => window.innerHeight")
            transcript = page.locator("[data-testid='transcript']")
            composer = page.locator("[data-testid='room-composer']")
            live = page.locator("[data-testid='live-turn']")
            jump = page.get_by_role("button", name="Jump to latest")

            following = await transcript.evaluate("(el) => el.scrollTop")
            assert following > 0, (
                f"the transcript did not follow the streaming reply down (scrollTop {following})"
            )
            assert await jump.count() == 0, "offered a jump to a reader already at the bottom"
            before = await composer.bounding_box()
            assert before is not None, "the composer has no box"

            streamed = len(await live.text_content() or "")
            await transcript.evaluate("(el) => { el.scrollTop = 0; }")
            # Words keep arriving below the reader. Without this wait the reads below could
            # all happen between two deltas, and a pull back down would never be seen.
            await page.wait_for_function(_LIVE_TEXT_GREW_PAST, arg=streamed, timeout=20000)
            await page.evaluate(TWO_FRAMES)

            moved_to = await transcript.evaluate("(el) => el.scrollTop")
            assert moved_to == 0, (
                f"the transcript did not stay at the top it was scrolled to (scrollTop "
                f"{moved_to}) while the reply was still streaming: either it is not the "
                "scroller, or auto-scroll pulled a reader who had scrolled up back down"
            )
            assert await live.count() == 1, "the premise: the reply is still arriving"

            after = await composer.bounding_box()
            assert after is not None, "the composer lost its box while the transcript scrolled"
            assert abs(after["y"] - before["y"]) <= 1.0, (
                f"the composer moved from y {before['y']} to {after['y']} while the "
                "transcript scrolled"
            )
            assert abs(after["height"] - before["height"]) <= 1.0, (
                f"the composer changed height from {before['height']} to {after['height']} "
                "while the transcript scrolled"
            )
            assert after["y"] + after["height"] <= window_height + 2, (
                f"the composer ran past the bottom of the {window_height}px window "
                f"({after['y'] + after['height']}) once the transcript scrolled"
            )

            # Told there is more below, in words, as a button the keyboard reaches.
            await jump.wait_for(state="visible", timeout=5000)

            # And it stays there for the rest of the reply. Only scrolling down, sending, or
            # pressing the button resumes following.
            await page.wait_for_selector("[data-testid='row-4']", timeout=60000)
            await live.wait_for(state="detached", timeout=60000)
            await page.evaluate(TWO_FRAMES)
            settled_at = await transcript.evaluate("(el) => el.scrollTop")
            assert settled_at == 0, (
                f"the reply finishing moved a reader who had scrolled to the top to "
                f"scrollTop {settled_at}"
            )

            await jump.focus()
            await page.keyboard.press("Enter")
            await page.wait_for_function(_AT_THE_BOTTOM, timeout=5000)
            await jump.wait_for(state="detached", timeout=5000)
        finally:
            await browser.close()


#: Records, on every animation frame, whether the reply's row and its bubble are on screen,
#: as runs of `[state, frames]`: `R` for the row, `L` for the bubble, `-` for neither.
_RECORD_FRAMES = """() => {
    window.__frames = [];
    const tick = () => {
        const row = Boolean(document.querySelector("[data-testid='row-4']"));
        const live = Boolean(document.querySelector("[data-testid='live-turn']"));
        const key = (row ? 'R' : '-') + (live ? 'L' : '-');
        const runs = window.__frames;
        if (runs.length && runs[runs.length - 1][0] === key) runs[runs.length - 1][1] += 1;
        else runs.push([key, 1]);
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}"""


#: How long each read of one room is held before it goes out, in the case below.
_SLOW_READ_S = 0.15

#: A read of one room, `/api/rooms/<id>`, and not its sub-routes or the list.
_ONE_ROOM = re.compile(r"/api/rooms/[^/?]+$")


async def _held_read(route: Route) -> None:
    """Let a GET of the room through after `_SLOW_READ_S`; anything else at once."""
    if route.request.method == "GET":
        await asyncio.sleep(_SLOW_READ_S)
    await route.continue_()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_delay", [0.0, 0.005], ids=["fast-stream", "slow-stream"])
async def test_a_streamed_reply_is_on_screen_once_at_every_frame(
    scripted_reply_ui_server: UIServerFactory, chunk_delay: float
) -> None:
    """#1379: at no frame is a reply drawn twice, and at no frame between its first word and
    its row is it missing.

    One reply reaches the page by two channels that do not arrive in step: the landed row
    comes with a read of the room, and the bubble is cleared by the stream's `final`.
    Measured with this case on the bundle before the fix, 5 runs of each: on the fast
    stream, 15 to 144 consecutive frames with both `row-4` and `live-turn` on screen -- the
    send's own re-read returned the row while deltas were still arriving; on the slow one,
    18 to 20 frames with neither between the bubble and the row -- `final` cleared the
    bubble a round trip before the row arrived. 10 of 10 red.

    The two delays are the two orders. What is asserted is the whole recorded sequence: once
    the bubble has appeared, the page shows the bubble, then the row, and nothing else.

    Every read of the room is held for `_SLOW_READ_S` before it goes out. Unheld, the
    handoff gap on localhost is a few milliseconds and falls inside one frame about half the
    time -- 3 of 6 runs of the slow stream were red on the unfixed bundle -- so the case
    would catch the flicker by luck. Held, the gap is the round trip a real network gives
    it. The request is only delayed, never answered by the test: the row that arrives is the
    server's own.

    Mutation-checked by hand, for the committed-bundle reason the cases above record. With a
    real `vite build`:
    `Killed by:` frontend/src/components/rooms/RoomConversation.tsx ::
    `!room.transcript.some((row) => row.turn_id === inFlight.turnId)` becoming `true`
    fails the fast stream (`RL` frames), and
    `Killed by:` frontend/src/components/rooms/RoomConversation.tsx ::
    `const inFlight = live.turn ?? heldRef.current?.turn ?? null;` becoming
    `const inFlight = live.turn;` fails the slow one (a `--` frame at the handoff), and
    the composer case above as well. Each fails nothing else in this file.
    """
    base_url = scripted_reply_ui_server(TALL_REPLY, streaming_chunk_delay=chunk_delay)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room(page, base_url)
            await page.route(_ONE_ROOM, _held_read)
            await page.fill("[data-testid='room-composer']", "Say something long.")
            await page.evaluate(_RECORD_FRAMES)
            await page.click("[data-testid='send-message']")
            await page.wait_for_selector("[data-testid='row-4']", timeout=60000)
            await page.wait_for_selector(
                "[data-testid='live-turn']", state="detached", timeout=60000
            )
            # A few more frames, so a bubble that came back after the row would be recorded.
            await page.wait_for_timeout(300)
            recorded: list[list[Any]] = await page.evaluate("() => window.__frames")
        finally:
            await browser.close()

    runs = [(str(state), int(frames)) for state, frames in recorded]
    states = [state for state, _ in runs]
    assert "RL" not in states, f"the reply was on screen twice: {runs}"
    assert "-L" in states, f"the premise: the reply streamed into a bubble first; got {runs}"
    after_first_word = states[states.index("-L") :]
    assert after_first_word == ["-L", "R-"], (
        f"expected the bubble and then the row, with nothing between or after: {runs}"
    )


#: How long after the stream's `final` has fired its own read the send's re-read is handed to
#: the page, in the case below. Well inside `_SLOW_READ_S`, so it lands before that read does.
_STALE_READ_LAG_S = 0.03


class _StaleSendReread:
    """Answer the send's re-read of the room at once, and hand it over after `final`'s read.

    The first `GET /api/rooms/<id>` after the conversation is open is the one `onSend` in
    `App.tsx` makes after its POST. It is performed against the real server as soon as it is
    asked (`route.fetch`), so its answer is the transcript as it stood then, before the reply
    was recorded. The page receives those same bytes only once the next read of the room --
    the one the stream's `final` fires -- has gone out, and that later read is held for
    `_SLOW_READ_S`, so the older answer arrives between `final` and the row (#1412).
    """

    def __init__(self) -> None:
        self.stale_body = ""
        self.later_read_issued = asyncio.Event()
        self._reads = 0

    async def handle(self, route: Route) -> None:
        if route.request.method != "GET":
            await route.continue_()
            return
        self._reads += 1
        if self._reads > 1:
            self.later_read_issued.set()
            await asyncio.sleep(_SLOW_READ_S)
            await route.continue_()
            return
        response = await route.fetch()
        self.stale_body = await response.text()
        try:
            await asyncio.wait_for(self.later_read_issued.wait(), 60)
        finally:
            await asyncio.sleep(_STALE_READ_LAG_S)
            await route.fulfill(response=response)


@pytest.mark.asyncio
async def test_a_read_from_before_final_does_not_take_the_reply_off_screen(
    scripted_reply_ui_server: UIServerFactory,
) -> None:
    """#1412: a read of the room that went out before the reply landed, and resolves after
    the stream's `final`, must not replace the bubble with a transcript that lacks the row.

    `RoomConversation` holds the streamed words after `final` until the transcript next
    changes, so the reply stays on screen for the round trip before its row arrives. The
    send's own re-read is in flight across that moment. Before the fix it was committed as
    an act: landing after `final` it put a transcript without the row on screen, which
    dropped the held words (`--`), and bumped the generation, which retired the read `final`
    fired -- so the row did not arrive at all until something else read the room again.

    The slow stream only, so that the send's re-read is sure to go out before `final`; the
    premise below checks that it did.

    Mutation-checked by hand, with a real `vite build`, for the committed-bundle reason the
    cases above record:
    `Killed by:` frontend/src/App.tsx ::
    `commitRoom(ticket, await roomsApi.get(currentRoomId));` becoming
    `commitRoomAct(ticket.roomId, ticket.generation, await roomsApi.get(currentRoomId));`
    fails this case (the row never lands), and
    `Killed by:` frontend/src/App.tsx ::
    `roomReadsRef.current.isStale(ticket, roomGenerationRef.current, currentRoomIdRef.current)`
    becoming
    `roomResponseIsStale(ticket.generation, roomGenerationRef.current, ticket.roomId, currentRoomIdRef.current)`
    fails it too (a `--` run between the bubble and the row).
    """
    base_url = scripted_reply_ui_server(TALL_REPLY, streaming_chunk_delay=0.005)
    reads = _StaleSendReread()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room(page, base_url)
            await page.route(_ONE_ROOM, reads.handle)
            await page.fill("[data-testid='room-composer']", "Say something long.")
            await page.evaluate(_RECORD_FRAMES)
            await page.click("[data-testid='send-message']")
            landed = True
            try:
                await page.wait_for_selector("[data-testid='row-4']", timeout=30000)
                await page.wait_for_selector(
                    "[data-testid='live-turn']", state="detached", timeout=30000
                )
            except PlaywrightTimeoutError:
                landed = False
            await page.wait_for_timeout(300)
            recorded: list[list[Any]] = await page.evaluate("() => window.__frames")
        finally:
            await browser.close()

    held = json.loads(reads.stale_body)
    assert [row["seq"] for row in held["transcript"]] == [1, 2, 3], (
        "the premise: the read handed over late describes the conversation before the reply "
        f"was recorded; it held {[row['seq'] for row in held['transcript']]}"
    )
    assert reads.later_read_issued.is_set(), "the premise: `final` fired a read of its own"
    runs = [(str(state), int(frames)) for state, frames in recorded]
    states = [state for state, _ in runs]
    assert landed, f"the reply's row never reached the screen: {runs}"
    assert "-L" in states, f"the premise: the reply streamed into a bubble first; got {runs}"
    after_first_word = states[states.index("-L") :]
    assert after_first_word == ["-L", "R-"], (
        f"expected the bubble and then the row, with nothing between or after: {runs}"
    )
