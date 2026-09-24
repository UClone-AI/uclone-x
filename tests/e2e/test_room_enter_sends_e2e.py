"""Enter sends the message; Shift+Enter starts a line. In a real browser, on real keys.

The composer had no key handler at all, so Enter only ever grew the draft and a message
could be sent exactly one way: by travelling to the button in the corner of the box. Every
surface a reader arrives from -- Claude Desktop, Slack, Discord, iMessage -- has already
taught their hands otherwise, which made the most ordinary action on the screen the one
that did not work.

The component tests beside this (`RoomConversation.test.tsx`, `describe('the send key')`)
synthesise the event and pin the branches, including the ones a browser cannot be made to
perform on demand -- an IME mid-composition, most of all. What they cannot show is that a
real keypress reaches the handler at all and that calling `preventDefault` on it actually
stops the textarea from taking the newline: both are the browser's behaviour, not React's.
So this drives the key itself, through the real routes, and reads the transcript.
"""

from __future__ import annotations

from typing import Any

import pytest
from playwright.async_api import Page, async_playwright

pytestmark = pytest.mark.e2e


async def _seat_a_conversation(page: Page, base_url: str) -> str:
    response = await page.request.post(
        f"{base_url}/api/rooms",
        data={"title": "Enter sends", "agent_ids": ["scout"]},
    )
    assert response.ok, await response.text()
    body: dict[str, Any] = await response.json()
    return str(body["room_id"])


async def _open_the_conversation(page: Page, base_url: str, room_id: str) -> None:
    await page.goto(base_url, wait_until="commit")
    await page.wait_for_selector(f"[data-testid='conversation-{room_id}']", timeout=15000)
    await page.click(f"[data-testid='conversation-{room_id}']")
    await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)


@pytest.mark.asyncio
async def test_enter_sends_the_message_without_touching_the_button(
    ui_test_server: str,
) -> None:
    """The key does what the button does: the message lands and the box empties."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _seat_a_conversation(page, ui_test_server)
        await _open_the_conversation(page, ui_test_server, room_id)

        await page.fill("[data-testid='room-composer']", "check the index on users")
        await page.press("[data-testid='room-composer']", "Enter")

        # Row 3 is the message, row 4 the reply: rows 1 and 2 are the two joins.
        await page.wait_for_selector("[data-testid='row-4']", timeout=20000)
        assert (
            "check the index on users" in await page.locator("[data-testid='row-3']").inner_text()
        )

        # The box empties on a send that landed, exactly as the button's path leaves it --
        # and, decisively, it does not hold the newline the textarea would have taken had
        # `preventDefault` not run. A composer reading "\n" here would mean the key both
        # sent the message and typed into the next one.
        assert await page.locator("[data-testid='room-composer']").input_value() == ""
        assert not errors, f"the conversation raised in the browser: {errors}"
        await browser.close()


@pytest.mark.asyncio
async def test_shift_enter_writes_a_second_line_and_sends_nothing(
    ui_test_server: str,
) -> None:
    """A message of more than one line is not an edge case: the box is two rows tall.

    If Shift+Enter sent, a paragraph could not be written in this surface at all. The
    transcript is read at the end rather than the composer alone, because "no message was
    sent" is the claim, and an empty transcript is the only thing that establishes it.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _seat_a_conversation(page, ui_test_server)
        await _open_the_conversation(page, ui_test_server, room_id)

        await page.fill("[data-testid='room-composer']", "first line")
        await page.press("[data-testid='room-composer']", "Shift+Enter")
        await page.keyboard.type("second line")

        assert (
            await page.locator("[data-testid='room-composer']").input_value()
            == "first line\nsecond line"
        )
        # Rows 1 and 2 are the joins the room opens with; a third row would be the message
        # this test says was never sent.
        assert await page.locator("[data-testid='row-3']").count() == 0, (
            "Shift+Enter sent the message instead of starting a line"
        )
        assert not errors, f"the conversation raised in the browser: {errors}"
        await browser.close()
