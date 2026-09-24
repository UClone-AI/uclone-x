"""A reply that claims a save the Core refused is not left to stand alone (#1375).

On qwen3:8b the clone answered "I've saved your favourite colour" after both of its
`record_memory_fact` calls had come back as errors. The errors were only in Activity; the
conversation, which is what the user reads, carried the claim and nothing else.

The model's words are not rewritten. The Core counts the facts the turn tried to save and
the ones it never saved (`RoomMessage.memory_facts_tried` / `memory_facts_unsaved`), and the
conversation says so under the reply. This case holds that end to end: the provider is
scripted, and everything after it -- the tool's validation, the agent loop, the
orchestrator, the store, `GET /api/rooms/{id}` and the render -- is the shipped code.

It also holds the other half: the notice is written for the person reading, not the model.
The tool's error for this call is a pydantic dump -- the params class, `type=missing`, the
argument dict, a docs URL -- and the model is right to get it. The conversation is not
(#1400 review).

Mutation-checked by hand rather than as a kill declaration, because the lethality ratchet
runs a browser test against the committed bundle without rebuilding it (see
`tests/e2e/test_room_failed_turn_row_e2e.py`). Rebuilt with `vite build`, each fails this
case:
`Killed by:` frontend/src/components/rooms/RoomConversation.tsx ::
`{unsavedNotice ? (` becoming `{false && unsavedNotice ? (`
`Killed by:` the reviewed head a78110a1 as a whole (Core and bundle), whose notice rendered
the tool's error after "The reason given:" -- the no-internals assertion fails there on
`RecordMemoryFactParams`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import ViewportSize, async_playwright

from tests.e2e.conftest import running_ui
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ToolCallRequest

pytestmark = pytest.mark.e2e

VIEWPORT: ViewportSize = {"width": 1280, "height": 900}

CLAIM = "I've saved your favourite colour."

#: What the tool's error for a missing value carries, and what a person must never be shown
#: as copy: the params class, pydantic's type code and argument dump, its docs URL, the
#: tool's internal id and the argument's field name, and an exception class name.
INTERNALS = (
    "RecordMemoryFactParams",
    "type=",
    "input_value",
    "http",
    "record_memory_fact",
    "object_value",
    "Error",
    "pydantic",
)


@pytest.fixture
def failed_save_ui_server(tmp_path: Path) -> Iterator[str]:
    """A provider that asks for one save with no value, then says it saved.

    The missing `object_value` is refused by the tool's own argument validation, so nothing
    reaches memory -- the same outcome the qwen3 run met, reached through the Core rather
    than asserted at the row.
    """
    llm = MockLLMConnector(
        default_model="mock-gpt-4o",
        responses=["", CLAIM],
        tool_calls=[
            ToolCallRequest(
                id="save-1",
                name="record_memory_fact",
                arguments={"subject": "user", "predicate": "favourite colour"},
            )
        ],
    )
    with running_ui(storage_dir=tmp_path, llm=llm) as url:
        yield url


@pytest.mark.asyncio
async def test_a_reply_after_a_failed_save_says_nothing_was_saved_in_plain_words(
    failed_save_ui_server: str,
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            response = await page.request.post(
                f"{failed_save_ui_server}/api/rooms",
                data={"title": "Memory", "agent_ids": ["clone"]},
            )
            assert response.ok, await response.text()
            body: dict[str, Any] = await response.json()
            room_id = str(body["room_id"])

            await page.goto(failed_save_ui_server, wait_until="commit")
            await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
            await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)
            await page.fill("[data-testid='room-composer']", "Remember my favourite colour is teal")
            await page.click("[data-testid='send-message']")

            # Rows 1 and 2 are the joins, 3 the prompt, 4 the reply.
            row = page.locator("[data-testid='row-4']")
            await row.get_by_text(CLAIM).wait_for(timeout=20000)
            notice = row.locator("[data-testid='row-memory-unsaved-4']")
            assert await notice.count() == 1, "the reply claimed a save and the row said nothing"
            text = await notice.inner_text()
            assert "nothing was saved" in text
            leaked = [internal for internal in INTERNALS if internal in text]
            assert not leaked, f"the notice shows the reader internals {leaked}: {text}"
        finally:
            await browser.close()
