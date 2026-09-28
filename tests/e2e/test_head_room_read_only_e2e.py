"""A head's room opens in the app read-only: the note stands where the composer was (#1885).

A room `ucx run` keeps is written only by `ucx run`, and the server refuses the app's
writes to it. The
refusal alone would let a person type a message and only then learn it cannot be sent, so
the app draws no composer there and says why in its place. The unit tests of
`RoomConversation` pin the branch; this pins it on the served bundle, against a room written
the way the head writes one (`record_head_turn`), in English and in Korean.

Mutation, declared by hand (the ratchet does not rebuild the bundle): in
`frontend/src/components/rooms/RoomConversation.tsx`, drawing the composer whatever
`room.head` holds fails this test on `room-composer` being present.

The same room opens with the person's own join row, which read "나이(가) 이 대화에
참여했습니다" in Korean (#1900): the third-person sentence around the reader's name for
themselves. The second case pins the sentence of their own. Mutation, declared by hand: in
`RoomConversation.tsx`'s `MembershipRow`, `ownRow`'s `=== 'human'` changed to `=== 'agent'`
fails it in both locales.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.async_api import Page, ViewportSize, async_playwright

from tests.e2e.conftest import mock_llm, running_ui

pytestmark = pytest.mark.e2e

VIEWPORT: ViewportSize = {"width": 1280, "height": 900}
ROOM_ID = "room_from_the_terminal"
MEMBERSHIP_ROWS = {
    "en-US": ["You joined this conversation", "scout joined this conversation"],
    # A Latin name keeps the paired particle (author's choice, #1900).
    "ko-KR": ["이 대화에 참여했습니다", "scout이(가) 이 대화에 참여했습니다"],
}
NOTE = {
    "en-US": "This conversation continues in ucx run, in the terminal. "
    "You can read it here, but not write in it.",
    "ko-KR": "이 대화는 터미널의 ucx run에서 이어집니다. 여기서는 읽을 수만 있고 쓸 수는 없습니다.",
}


@pytest.fixture
def head_room_ui(tmp_path: Path) -> Iterator[str]:
    """The app over a storage folder holding one room `ucx run` wrote."""
    from uclone_x.room.one_seat import HeadTurn, record_head_turn
    from uclone_x.room.store import RoomStore

    storage_dir = tmp_path / "state"
    record_head_turn(
        RoomStore(storage_dir / "rooms"),
        room_id=ROOM_ID,
        clone_id="scout",
        turn=HeadTurn(prompt="what is in the index?", content="three tables"),
        head="run",
    )
    with running_ui(storage_dir=storage_dir, llm=mock_llm()) as url:
        yield url


async def _open(page: Page, base_url: str) -> None:
    await page.goto(base_url, wait_until="commit")
    await page.click(f"[data-testid='conversation-{ROOM_ID}']", timeout=15000)
    await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["en-US", "ko-KR"])
async def test_a_head_room_shows_the_note_in_place_of_the_composer(
    head_room_ui: str, tmp_path: Path, locale: str
) -> None:
    """The transcript reads as usual; the note says where the conversation goes on."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            context = await browser.new_context(viewport=VIEWPORT, locale=locale)
            page = await context.new_page()
            await _open(page, head_room_ui)

            note = page.locator("[data-testid='head-room-note']")
            await note.wait_for(timeout=15000)
            assert (await note.inner_text()).strip() == NOTE[locale]
            await page.get_by_text("three tables").wait_for(timeout=15000)
            assert await page.locator("[data-testid='room-composer']").count() == 0
            assert await page.locator("[data-testid='send-message']").count() == 0
            await page.screenshot(path=str(tmp_path / f"head-room-{locale}.png"))
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["en-US", "ko-KR"])
async def test_the_reader_s_own_join_row_is_worded_as_theirs(
    head_room_ui: str, tmp_path: Path, locale: str
) -> None:
    """The person's own join row has a sentence of its own, in each language (#1900)."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            context = await browser.new_context(viewport=VIEWPORT, locale=locale)
            page = await context.new_page()
            await _open(page, head_room_ui)

            rows = page.locator("[data-testid^='membership-']")
            await rows.first.wait_for(timeout=15000)
            texts = [text.strip() for text in await rows.all_inner_texts()]
            # The reader's row first, then the head's, which keeps the third-person sentence.
            assert texts == MEMBERSHIP_ROWS[locale]
            await page.screenshot(path=str(tmp_path / f"own-join-{locale}.png"))
        finally:
            await browser.close()
