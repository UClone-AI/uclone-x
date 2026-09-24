"""End-to-end tests for renaming and deleting a conversation from its rail row (#1058).

The Core had both routes -- `PATCH` and `DELETE /api/rooms/{room_id}` -- and the head
called neither from any control, so a conversation, once started, stayed in the list for
good under whatever its first message happened to say.

What only a browser can check, and so what these cases are for:

* **The controls are drawn without hover.** jsdom applies no stylesheet, so a component
  test can only read class names; here the computed style is read with the pointer nowhere
  near the row. uclone2's hover-only trash icon was the defect this guards against: a
  touch screen could never reach it.
* **Each input reaches them**: touch (a tap in a touch-enabled context), keyboard (focus and
  Enter, Tab, Escape) and pointer.
* **A refusal arrives in the Core's own words** through the real route, not a fixture.
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


@pytest.mark.asyncio
async def test_a_conversation_is_renamed_by_touch_and_a_blank_title_is_refused_in_the_cores_words(
    ui_test_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(has_touch=True, viewport={"width": 1600, "height": 900})
        page = await context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _create_room(page, ui_test_server, "check the index on", ["scout"])
        await page.goto(ui_test_server, wait_until="commit")
        row = page.locator(f"[data-testid='conversation-{room_id}']")
        await row.wait_for(timeout=15000)

        rename = page.locator(f"[data-testid='rename-conversation-{room_id}']")
        delete = page.locator(f"[data-testid='delete-conversation-{room_id}']")
        # The pointer has not been near the rail: nothing here is hovered.
        assert await _drawn(rename), "Rename is not drawn until something hovers the row"
        assert await _drawn(delete), "Delete is not drawn until something hovers the row"

        await rename.tap()
        title = page.get_by_role("textbox", name="Conversation title")
        await title.fill("   ")
        await title.press("Enter")

        refusal = page.locator("[data-testid='conversation-title-editor'] [role='alert']")
        await refusal.wait_for(timeout=10000)
        assert "A room needs a title" in await refusal.inner_text()

        await title.fill("Index tuning")
        await page.get_by_role("button", name="Save title").tap()
        await page.locator(
            f"[data-testid='conversation-{room_id}']", has_text="Index tuning"
        ).wait_for(timeout=10000)

        stored = await page.request.get(f"{ui_test_server}/api/rooms/{room_id}")
        assert (await stored.json())["title"] == "Index tuning"
        assert not errors, f"the rail raised in the browser: {errors}"
        await browser.close()


@pytest.mark.asyncio
async def test_deleting_the_open_conversation_by_keyboard_asks_first_and_leaves_nothing_behind(
    ui_test_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _create_room(page, ui_test_server, "Throwaway draft", ["scout"])
        await page.goto(ui_test_server, wait_until="commit")
        await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
        await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)

        delete = page.locator(f"[data-testid='delete-conversation-{room_id}']")
        await delete.focus()
        await page.keyboard.press("Enter")

        dialog = page.get_by_role("alertdialog")
        await dialog.wait_for(timeout=10000)
        said = await dialog.inner_text()
        assert "Throwaway draft" in said
        assert "scout" in said
        assert "cannot be undone" in said

        # Escape keeps everything.
        await page.keyboard.press("Escape")
        await dialog.wait_for(state="detached", timeout=10000)
        assert (await page.request.get(f"{ui_test_server}/api/rooms/{room_id}")).status == 200

        # Cancel holds the focus, so it takes a deliberate Tab to reach Delete.
        await delete.focus()
        await page.keyboard.press("Enter")
        await dialog.wait_for(timeout=10000)
        await page.keyboard.press("Tab")
        await page.keyboard.press("Enter")

        await dialog.wait_for(state="detached", timeout=10000)
        await page.locator(f"[data-testid='conversation-{room_id}']").wait_for(
            state="detached", timeout=10000
        )
        # The deleted conversation is not the open one. Asserted as "not this room" and no
        # longer as "no room at all": since #1208 the head has one conversation surface and
        # no second screen to fall back to, so deleting what is open releases the auto-open
        # latch and the next conversation takes its place. What must not survive the delete
        # is *this* room, and the rail row detaching above plus the 404 below is that claim
        # end to end.
        assert await page.locator(f"[data-testid='conversation-{room_id}']").count() == 0, (
            "the deleted conversation is still in the rail"
        )
        assert await page.locator("[data-testid='room-notice']").count() == 0
        assert (await page.request.get(f"{ui_test_server}/api/rooms/{room_id}")).status == 404
        assert not errors, f"the rail raised in the browser: {errors}"
        await browser.close()


@pytest.mark.asyncio
async def test_a_delete_the_core_refuses_says_why_and_what_to_do(ui_test_server: str) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _create_room(page, ui_test_server, "Deleted elsewhere", ["scout"])
        await page.goto(ui_test_server, wait_until="commit")
        await page.click(f"[data-testid='delete-conversation-{room_id}']", timeout=15000)
        dialog = page.get_by_role("alertdialog")
        await dialog.wait_for(timeout=10000)

        # A second window gets there first.
        gone = await page.request.delete(f"{ui_test_server}/api/rooms/{room_id}")
        assert gone.status == 204, await gone.text()

        await page.click("[data-testid='confirm-delete-conversation']")

        refusal = dialog.get_by_role("alert")
        await refusal.wait_for(timeout=10000)
        reason = await refusal.inner_text()
        assert room_id in reason
        assert "has been deleted" in reason
        assert "List the rooms" in reason
        # The refusal said the conversation is gone, so its row goes too.
        await page.locator(f"[data-testid='conversation-{room_id}']").wait_for(
            state="detached", timeout=10000
        )
        assert not errors, f"the rail raised in the browser: {errors}"
        await browser.close()
