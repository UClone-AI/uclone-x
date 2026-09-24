"""A fact a clone saves in a conversation is listed on the dock's Remembers tab (#1401).

Before #1401 the tab listed only the seat's knowledge record for the conversation, which
nothing in a room turn adds to (#1404), so in real use it never filled: a clone could save
"Kenny's favourite colour is teal" to its own memory and Remembers still said nothing was
listed. The tab now lists the facts the clone saved to memory as well.

The provider is scripted to ask for one `record_memory_fact`; everything after it -- the
tool, the agent loop, the orchestrator, the clone's memory file, `GET
/api/rooms/{id}/knowledge` and the render -- is the shipped code.

Mutation-checked by hand rather than as a kill declaration, because the lethality ratchet
runs a browser test against the committed bundle without rebuilding it (see
`tests/e2e/test_room_failed_turn_row_e2e.py`). The Core half is declared in
`tests/unit/test_ui_room_dock_routes.py`; rebuilt with `vite build`, this fails when
`frontend/src/components/artifacts/RemembersPanel.tsx`'s `{saved && saved.length > 0 && (`
becomes `{saved && saved.length > 99 && (`. It also fails against the bundle this change
replaced, which had no saved-facts group.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import ViewportSize, async_playwright

from tests.e2e.conftest import dock_locator, running_ui
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ToolCallRequest

pytestmark = pytest.mark.e2e

#: Wide enough that the dock seats itself beside the conversation.
VIEWPORT: ViewportSize = {"width": 1400, "height": 900}

#: What a person must never be shown on this surface: transport text, class names, paths.
TECHNICAL = re.compile(
    r"Failed to fetch|HTTP \d|Internal Server Error|Traceback|Error\b|record_memory_fact|/api/|\.json"
)


@pytest.fixture
def saving_ui_server(tmp_path: Path) -> Iterator[str]:
    """A provider that saves one fact to the clone's memory, then answers."""
    llm = MockLLMConnector(
        default_model="mock-gpt-4o",
        responses=["", "Noted: teal."],
        tool_calls=[
            ToolCallRequest(
                id="save-1",
                name="record_memory_fact",
                arguments={
                    "subject": "Kenny",
                    "predicate": "favourite_colour",
                    "object_value": "teal",
                },
            )
        ],
    )
    with running_ui(storage_dir=tmp_path, llm=llm) as url:
        yield url


@pytest.mark.asyncio
async def test_a_fact_saved_in_the_conversation_is_listed_on_remembers(
    saving_ui_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            # `clone` names no shipped persona, so it has every tool, as a new clone does.
            response = await page.request.post(
                f"{saving_ui_server}/api/rooms",
                data={"title": "Memory", "agent_ids": ["clone"]},
            )
            assert response.ok, await response.text()
            body: dict[str, Any] = await response.json()
            room_id = str(body["room_id"])

            await page.goto(saving_ui_server, wait_until="commit")
            await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
            await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)
            await page.fill("[data-testid='room-composer']", "My favourite colour is teal")
            await page.click("[data-testid='send-message']")
            await (
                page.locator("[data-testid='row-4']")
                .get_by_text("Noted: teal.")
                .wait_for(timeout=20000)
            )

            dock = await dock_locator(page)
            if not await dock.is_visible():
                await page.click("[data-testid='toggle-dock']")
            await dock.wait_for(state="visible", timeout=10000)
            await dock.locator("[data-testid='tab-remembers']").click()

            fact = dock.locator("[data-testid='saved-fact']")
            await fact.first.wait_for(timeout=15000)
            assert await fact.count() == 1
            text = await fact.first.inner_text()
            assert "Kenny favourite colour teal" in text
            assert "saved in this conversation" in text

            panel = await dock.locator("[data-testid='remembers-panel']").inner_text()
            assert not TECHNICAL.search(panel), f"Remembers shows technical text: {panel}"
        finally:
            await browser.close()
