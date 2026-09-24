"""End-to-end tests for clearing a conversation's history (#1208).

`PlaygroundTab` carried the clear control and the conversation surface did not, which is
the reason the retirement could not simply delete it: a user who moved to the conversation
lost the ability to empty a thread without deleting it. The head now calls
`DELETE /api/rooms/{id}/history`.

What only a browser can check, and so what this case is for:

* **Clearing empties the conversation in place.** `DELETE /api/rooms/{id}` and
  `DELETE /api/rooms/{id}/history` differ by one path segment and by everything else: one
  removes the conversation, the other empties the one you are in. Only the real routes can
  say the head called the second.
* **The control is drawn without hover**, the defect #1058 guards for the rail's own
  controls, checked here for the ones this change adds.
"""

from __future__ import annotations

from typing import Any

import pytest
from playwright.async_api import Locator, Page, async_playwright

pytestmark = pytest.mark.e2e


async def _create_room(page: Page, base_url: str, title: str, agents: list[str]) -> str:
    """Seat a conversation through the API, as any other client would."""
    response = await page.request.post(
        f"{base_url}/api/rooms",
        data={"title": title, "agent_ids": agents},
    )
    assert response.ok, await response.text()
    body: dict[str, Any] = await response.json()
    return str(body["room_id"])


async def _drawn(control: Locator) -> bool:
    """Whether the control is painted, judged from its computed style and its ancestors'.

    Playwright's `is_visible` counts `opacity: 0` as visible, which is exactly how a
    hover-revealed control is usually hidden, so it cannot answer this on its own.
    """
    return bool(
        await control.evaluate(
            """(el) => {
                const box = el.getBoundingClientRect();
                if (box.width === 0 || box.height === 0) return false;
                for (let node = el; node; node = node.parentElement) {
                    const style = getComputedStyle(node);
                    if (style.display === 'none' || style.visibility === 'hidden') return false;
                    if (parseFloat(style.opacity) === 0) return false;
                }
                return true;
            }"""
        )
    )


async def _say(page: Page, words: str, lands_as: int) -> None:
    """Send one message and return once the reply to it has landed."""
    await page.fill("[data-testid='room-composer']", words)
    await page.click("[data-testid='send-message']")
    await page.wait_for_selector(f"[data-testid='row-{lands_as}']", timeout=20000)


@pytest.mark.asyncio
async def test_clearing_empties_the_conversation_in_place_rather_than_deleting_it(
    ui_test_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _create_room(page, ui_test_server, "Throwaway thread", ["scout"])
        await page.goto(ui_test_server, wait_until="commit")
        await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
        await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)
        await _say(page, "check the index on users", lands_as=4)

        clear = page.locator("[data-testid='clear-history']")
        assert await _drawn(clear), "Clear is not drawn until something hovers the header"

        # Asked first, and the question carries the number the control's label cannot.
        await clear.click()
        asked = page.locator("[data-testid='confirm-history-change']")
        await asked.wait_for(timeout=10000)
        said = await asked.inner_text()
        assert "Its 4 messages are removed for good" in said, said
        assert "keeps its name and who is in it" in said

        # "Keep everything" keeps everything.
        await page.click("[data-testid='confirm-history-change-no']")
        await asked.wait_for(state="detached", timeout=10000)
        assert await page.locator("[data-testid='row-4']").count() == 1

        await clear.click()
        await asked.wait_for(timeout=10000)
        await page.click("[data-testid='confirm-history-change-yes']")

        await page.locator("[data-testid='transcript-empty']").wait_for(timeout=10000)
        assert await page.locator("[data-testid='row-4']").count() == 0
        assert await page.locator("[data-testid='room-conversation']").count() == 1, (
            "clearing the history closed the conversation"
        )
        await page.locator(f"[data-testid='conversation-{room_id}']").wait_for(timeout=10000)

        standing = await page.request.get(f"{ui_test_server}/api/rooms/{room_id}")
        assert standing.status == 200, "the head deleted the conversation instead of emptying it"
        stored = await standing.json()
        assert stored["transcript"] == []
        assert stored["title"] == "Throwaway thread"
        assert len(stored["participants"]) == 2
        assert not errors, f"the conversation raised in the browser: {errors}"
        await browser.close()
