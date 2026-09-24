"""The conversation renders a reply's markdown as structure, in a browser (#1225).

Retiring `PlaygroundTab` (#1234, `289a0c43`) replaced a surface that put every message through
`RichText` with one that printed `message.content` into a `<p>`. The regression was invisible to
the suite: `test_chat_viewport_regression.py` held the old surface to #1010, #1015 and #1018 --
three cases about tables in a reply -- and the port that replaced it,
`test_room_conversation_layout_e2e.py`, carried the *column* cases across and not the table ones,
saying so in its own docstring, because by then no renderer existed for them to run against.

So this is not a component test's job alone. The failure was end-to-end in shape: every unit
test passed, and what broke was what a reader saw. These assert the rendered *structure* -- a
`<pre>`, a `<table>`, a scroller -- rather than the absence of backticks, because a row that
rendered nothing at all also contains no backticks.

Content is scripted through the model (`scripted_reply_ui_server`) rather than through
`page.route`, for the reason the sibling layout module gives: a test that answers the head's own
request renders a payload the test wrote, and the blank screen `test_room_chat_e2e` was opened
for was a real response whose shape the fixtures disagreed with.
"""

from __future__ import annotations

from typing import Any

import pytest
from playwright.async_api import Page, async_playwright

from tests.e2e.conftest import UIServerFactory

pytestmark = pytest.mark.e2e

#: A fenced block whose body is not prose, so the language badge and the `<pre>` are both real.
FENCED_REPLY = 'Use this:\n\n```python\nprint("indexed")\n```\n\nThat is the whole change.\n'

#: A table wider than any column here, so #1010's scroller is the thing under test and not a
#: table that happens to fit. The long cell has no break opportunity in it, which is what made
#: `overflow-wrap: anywhere` shrink every other column before `RichText` exempted tables.
LONG_CELL = "0123456789abcdef" * 12
TABLE_REPLY = (
    "| column one | column two | column three |\n"
    "| --- | --- | --- |\n"
    f"| {LONG_CELL} | second value here | third value here |\n"
)

#: A fenced block holding one line far too long for any column here. A code block must not wrap
#: it -- that would change what the code says -- so `PreBlock`'s `overflow-x-auto` has to absorb
#: it. If it does not, the line pushes the conversation itself sideways, which is #1007's defect
#: arriving by a different route: `[overflow-wrap:anywhere]` cannot help inside a `<pre>`.
LONG_CODE_REPLY = "```python\nvalue = " + '"%s"' % ("x" * 600) + "\n```\n"


async def _open_room_and_say(page: Page, base_url: str, words: str) -> None:
    """Seat a conversation, open it, send one message and wait for the reply to land."""
    response = await page.request.post(
        f"{base_url}/api/rooms", data={"title": "Markdown", "agent_ids": ["scout"]}
    )
    assert response.ok, await response.text()
    body: dict[str, Any] = await response.json()
    room_id = str(body["room_id"])

    await page.set_viewport_size({"width": 1280, "height": 900})
    await page.goto(base_url, wait_until="commit")
    await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
    await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)

    await page.fill("[data-testid='room-composer']", words)
    await page.click("[data-testid='send-message']")
    # Rows 1 and 2 are the joins, 3 is this message and 4 is the reply to it.
    await page.wait_for_selector("[data-testid='row-4']", timeout=20000)


@pytest.mark.asyncio
async def test_a_fenced_code_block_in_a_reply_renders_as_a_code_block(
    scripted_reply_ui_server: UIServerFactory,
) -> None:
    """A reply's fenced block is a `<pre>` on screen, not three backticks and a language word.

    The badge is asserted as well as the element: `RichText` routes a fence through `PreBlock`,
    which is what supplies the language and the copy control, so a bare `<pre>` from some other
    path would not be the renderer being back.

    Mutation-checked by hand, not as a kill declaration, because the lethality ratchet runs a
    browser test against the committed bundle without rebuilding it (the convention
    `test_rail_responsive_e2e.py` established). Rebuilt with `npm run build`, this fails:
    `Killed by:` frontend/src/components/rooms/RoomConversation.tsx ::
    `<MessageBody text={message.content} />` becoming `<p>{message.content}</p>`.
    """
    base_url = scripted_reply_ui_server(FENCED_REPLY)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room_and_say(page, base_url, "How do I index this?")

            row = page.locator("[data-testid='row-body-4']")
            await row.locator("pre").first.wait_for(timeout=10000)

            assert await row.locator("pre code").count() >= 1, "no <code> inside the <pre>"
            badge = await row.locator("text=python").first.text_content()
            assert badge is not None and "python" in badge

            text = await row.text_content() or ""
            assert "```" not in text, f"backticks are still on screen: {text[:200]!r}"
            assert 'print("indexed")' in text, "the code itself is missing from the block"
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_wide_table_in_a_reply_scrolls_in_its_own_box_rather_than_widening_the_column(
    scripted_reply_ui_server: UIServerFactory,
) -> None:
    """#1010/#1015 on the surviving surface: a table is a table, and it stays in the column.

    Three assertions, because each is a different way the row can be wrong: the table is parsed
    at all, it sits inside `TableScroll` (which is what keeps a wide one off the column), and
    the conversation itself does not scroll sideways as a result.

    Mutation-checked by hand for the reason the case above gives. Rebuilt with `npm run build`,
    this fails: `Killed by:` frontend/src/components/rooms/RoomConversation.tsx ::
    `<MessageBody text={message.content} />` becoming `<p>{message.content}</p>`.
    """
    base_url = scripted_reply_ui_server(TABLE_REPLY)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room_and_say(page, base_url, "Show me the columns")

            row = page.locator("[data-testid='row-body-4']")
            table = row.locator("table").first
            await table.wait_for(timeout=10000)

            assert await table.locator("th").count() == 3, "the header row was not parsed"
            text = await row.text_content() or ""
            assert "| --- |" not in text, "the table's own pipes are still on screen"

            # The table is inside the scroller, and the scroller is what overflows -- not the
            # conversation column, which is the defect #1010 was opened for.
            in_scroller = await table.evaluate(
                "(el) => el.closest('[data-testid=\"table-scroll\"]') !== null"
            )
            assert in_scroller, "a wide table is not inside TableScroll"

            column = page.locator("[data-testid='transcript']")
            sideways = await column.evaluate(
                "(el) => ({ scroll: el.scrollWidth, client: el.clientWidth })"
            )
            assert sideways["scroll"] <= sideways["client"] + 1, (
                f"the conversation scrolls sideways under a wide table: {sideways}"
            )
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [400, 1280])
async def test_a_long_code_line_scrolls_inside_its_block_rather_than_widening_the_column(
    scripted_reply_ui_server: UIServerFactory, width: int
) -> None:
    """A code line too long for the column stays inside its own block.

    This is the code half of #1010's rule, and it is not covered by the table case: a `<pre>`
    is exempt from `[overflow-wrap:anywhere]` by construction, since wrapping code would alter
    what it says. So the block itself has to be the scroller, and if it is not, the reply pushes
    the whole conversation sideways.

    Both widths, because the defect is a ratio rather than a length: a column wide enough for
    the line hides it.

    Mutation-checked by hand, for the reason the cases above give. Rebuilt with `npm run build`,
    this fails at both widths when `PreBlock`'s `<pre>` in `frontend/src/components/RichText.tsx`
    loses `overflow-x-auto`: `p-3.5 overflow-x-auto text-slate-200 leading-relaxed font-mono`
    becoming `p-3.5 text-slate-200 leading-relaxed font-mono`. That mutation is invisible to
    every other assertion in this module, which is why the `scrollLeft` one is here.
    """
    base_url = scripted_reply_ui_server(LONG_CODE_REPLY)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room_and_say(page, base_url, "Show me the literal")
            await page.set_viewport_size({"width": width, "height": 900})

            row = page.locator("[data-testid='row-body-4']")
            await row.locator("pre").first.wait_for(timeout=10000)

            column = page.locator("[data-testid='transcript']")
            sideways = await column.evaluate(
                "(el) => ({ scroll: el.scrollWidth, client: el.clientWidth })"
            )
            assert sideways["scroll"] <= sideways["client"] + 1, (
                f"a long code line pushed the conversation sideways at {width}px: {sideways}"
            )

            # And the block is a *scroller*, not merely a box the overflow is clipped by.
            # `scrollWidth > clientWidth` does not distinguish the two -- an element with
            # `overflow: visible` reports that too, and `PreBlock`'s wrapper carries
            # `overflow-hidden`, so the column stays put either way and the reader still
            # cannot reach the end of the line. Driving `scrollLeft` is what tells them
            # apart: it stays at 0 unless the element actually scrolls. Measured -- removing
            # `overflow-x-auto` from the `<pre>` leaves every other assertion here passing.
            scrolled = await row.locator("pre").first.evaluate(
                """(el) => {
                    if (el.scrollWidth <= el.clientWidth + 1) return -1;
                    el.scrollLeft = 9999;
                    const moved = el.scrollLeft;
                    el.scrollLeft = 0;
                    return moved;
                }"""
            )
            assert scrolled > 0, (
                f"the code line is clipped rather than scrollable (scrollLeft reached {scrolled})"
            )
        finally:
            await browser.close()
