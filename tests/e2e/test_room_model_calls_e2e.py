"""End-to-end: the Turn surface's developer-mode "Model calls" reads the real trace (#1492).

`ModelCalls.test.tsx` draws the section from fixtures shaped like the trace routes' answers,
so every one of its tests would still pass if the routes answered some other shape, or if the
request the Core recorded were not the one the section splits into layers. Only a turn through
the real Core -- `BaseAgent` recording the call (#1489), `turn_trace.py` rebuilding it and
`room_dock.py` serving it (#1490) -- shows that the head and the Core agree.

The second test holds the cost rule of turn-inspection.md §6: the trace reads a whole session
log, so with developer mode off the head must not ask for it at all, and component tests can
only see the fetch mock they were handed, not the requests a real page sends.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, Request, ViewportSize, async_playwright

from tests.e2e.conftest import (
    SURFACE_OVERFLOW,
    TWO_FRAMES,
    dock_locator,
    mock_llm,
    running_ui,
    turn_on_developer_mode,
)

pytestmark = pytest.mark.e2e

VIEWPORT: ViewportSize = {"width": 1400, "height": 900}

#: Rows 1 and 2 are the two joins, 3 is the prompt, 4 is the turn that answered it.
ANSWER_SEQ = 4

PROMPT = "Which model answered this?"

#: What `mock_llm()` answers every call with.
MOCK_ANSWER = "Mock response from UClone-X BaseAgent."


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    with running_ui(storage_dir=tmp_path, llm=mock_llm()) as url:
        yield url


async def _one_answered_turn(page: Page, base_url: str) -> None:
    """Seat a 1:1 through the API, open it, and let the Core answer one message."""
    response = await page.request.post(
        f"{base_url}/api/rooms",
        data={"title": "Model calls", "agent_ids": ["scout"]},
    )
    assert response.ok, await response.text()
    body: dict[str, Any] = await response.json()
    await page.goto(base_url, wait_until="commit")
    await page.click(f"[data-testid='conversation-{body['room_id']}']", timeout=15000)
    await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)
    await page.fill("[data-testid='room-composer']", PROMPT)
    await page.click("[data-testid='send-message']")
    await page.wait_for_selector(f"[data-testid='why-{ANSWER_SEQ}']", timeout=30000)


@pytest.mark.asyncio
async def test_model_calls_shows_the_request_as_sent_and_the_response_it_got(
    server: str,
) -> None:
    """Step 1 of the answered turn: its system layers, the user's words, and the reply."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            await _one_answered_turn(page, server)
            await turn_on_developer_mode(page)

            await page.click(f"[data-testid='why-{ANSWER_SEQ}']")
            dock = await dock_locator(page)
            detail = dock.locator("[data-testid='turn-detail']")
            await detail.wait_for(timeout=10000)

            await detail.locator("[data-testid='model-calls-toggle']").click()
            row = detail.locator("[data-testid='model-call-row'][data-step='1']")
            await row.wait_for(timeout=15000)
            headline = await row.locator("[data-testid='model-call-headline']").inner_text()
            assert headline.startswith("step 1"), headline
            assert "mock-gpt-4o" in headline, headline

            await row.locator("[data-testid='model-call-toggle-1']").click()
            call = row.locator("[data-testid='model-call-detail']")
            await call.wait_for(timeout=15000)

            blocks = call.locator("[data-testid='request-block']")
            first = blocks.first
            assert await first.get_attribute("data-role") == "system"
            assert await first.get_attribute("data-layer") == "identity"
            assert (await first.inner_text()).strip(), "the identity layer drew no text"
            users = call.locator("[data-testid='request-block'][data-role='user']")
            user_text = "\n".join(await users.all_inner_texts())
            assert PROMPT in user_text, user_text

            await call.locator("[data-testid='model-call-tab-response']").click()
            content = await call.locator("[data-testid='model-call-response-content']").inner_text()
            assert MOCK_ANSWER in content, content

            # The request and response are long text inside a dock column: they scroll in
            # their own boxes, not by widening the dock. The app root clips its own overflow,
            # so the document never scrolls sideways whatever the dock holds; the dock's
            # surface is what would, and it is measured with each tab open.
            found: dict[str, list[str]] = {}
            for tab in ("request", "tools", "response", "raw"):
                await call.locator(f"[data-testid='model-call-tab-{tab}']").click()
                await page.evaluate(TWO_FRAMES)
                problems = await page.evaluate(SURFACE_OVERFLOW)
                if problems:
                    found[tab] = problems
            assert found == {}, found
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_with_developer_mode_off_the_head_never_asks_for_a_trace(server: str) -> None:
    """Opening a turn's record without developer mode sends no trace request at all."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            traced: list[str] = []

            def note(request: Request) -> None:
                if "/trace" in request.url:
                    traced.append(request.url)

            page.on("request", note)
            await _one_answered_turn(page, server)

            await page.click(f"[data-testid='why-{ANSWER_SEQ}']")
            dock = await dock_locator(page)
            detail = dock.locator("[data-testid='turn-detail']")
            await detail.wait_for(timeout=10000)
            await detail.locator("[data-testid='turn-detail-served']").wait_for(timeout=10000)

            assert await detail.locator("[data-testid='model-calls']").count() == 0
            assert traced == [], traced
        finally:
            await browser.close()
