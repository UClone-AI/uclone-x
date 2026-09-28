"""A fact a clone saves is listed on Remembers, and can be corrected and forgotten there.

#1401 put the facts a clone saved to memory on the dock's Remembers tab. #1638 step 3 made
that the tab's one list, grouped by where each fact was learned, with a `⋯` on each fact
that corrects or forgets it (clone-knowledge-graph §3.6).

The provider is scripted to ask for one `record_memory_fact`; everything after it -- the
tool, the agent loop, the orchestrator, the clone's memory file, `GET
/api/rooms/{id}/knowledge`, `PATCH`/`DELETE /api/agents/{id}/memory/{fact}` and the render
-- is the shipped code.

The design's end-to-end row also forgets a fact from the turn row that learned it; that row
line arrives with the extractor (step 4), so this test forgets from Remembers.

Mutation-checked by hand rather than as a kill declaration, because the lethality ratchet
runs a browser test against the committed bundle without rebuilding it (see
`tests/e2e/test_room_failed_turn_row_e2e.py`). The Core half is declared in
`tests/unit/test_ui_room_dock_routes.py`; rebuilt with `vite build`, this fails when
`frontend/src/components/artifacts/RemembersPanel.tsx`'s
`const onChanged = () => setEdits((n) => n + 1);` becomes
`const onChanged = () => setEdits((n) => n);`.
It also fails against the bundle this change replaced, which had no `known-fact` rows.
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

            fact = dock.locator("[data-testid='known-fact']")
            await fact.first.wait_for(timeout=15000)
            assert await fact.count() == 1
            here = dock.locator("[data-testid='remembers-here']")
            assert "Learned in this conversation" in await here.inner_text()
            text = await fact.first.inner_text()
            assert "Kenny favourite colour teal" in text
            assert "saved" in text

            # Correct: the value is edited in place, and the list is read again.
            await fact.first.locator("[data-testid='fact-actions']").click()
            await fact.first.locator("[data-testid='fact-correct']").click()
            await fact.first.locator("[data-testid='fact-correct-input']").fill("navy")
            await fact.first.locator("[data-testid='fact-correct-save']").click()
            await dock.get_by_text("Kenny favourite colour navy").wait_for(timeout=10000)
            assert await fact.count() == 1
            assert "corrected" in await fact.first.inner_text()

            # Forget: asked first, then gone from the list.
            await fact.first.locator("[data-testid='fact-actions']").click()
            await fact.first.locator("[data-testid='fact-forget']").click()
            confirm = await fact.first.locator("[data-testid='fact-forget-confirm']").inner_text()
            assert confirm == "clone will stop using this. Forget it?"
            await fact.first.locator("[data-testid='fact-forget-yes']").click()
            reason = dock.locator("[data-testid='remembers-reason']")
            await reason.get_by_text("No facts are listed for clone.").wait_for(timeout=10000)
            assert await fact.count() == 0

            panel = await dock.locator("[data-testid='remembers-panel']").inner_text()
            assert not TECHNICAL.search(panel), f"Remembers shows technical text: {panel}"

            # Durable: a fresh read of the clone's memory lists nothing either.
            answer = await page.request.get(
                f"{saving_ui_server}/api/rooms/{room_id}/knowledge?agent_id=clone"
            )
            assert (await answer.json())["facts"] == []
        finally:
            await browser.close()
