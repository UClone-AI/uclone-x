"""End-to-end: `why ›` puts the turn's record on the dock, and the composer names its units.

Two facts the assembled product is the only place to check.

**The record moved off the row.** It used to open inside the turn it belonged to, three lines
of a longer record in a column shared with the conversation, and opening it pushed every turn
below it down the screen. `RoomConversation.test.tsx` can see that the control now hands a
`seq` outwards and `TurnDetail.test.tsx` can see what the surface draws from one, but neither
can see the two together, and neither can see a row's height: jsdom lays nothing out, so
"the turn you were reading stayed where it was" is not a claim it can hold.

**The token count is booked by the Core and read back.** The figure beside the ring is the
seat's own spend, which `BaseAgent` records against its session and
`GET /api/rooms/{id}/context` reads out of the budget manager. Every component test supplies
that number as a fixture, so all of them would still pass if the endpoint reported a seat the
head cannot match to the session that spent -- the head would then say "tokens not counted
this run" about a conversation that had just cost something, which is a P6 sentence told about
the wrong state. Only a real turn through the real Core distinguishes the two.

The ring's digits are asserted here as well as in the component test, because the unit is the
whole point of them: a bare `0/20` an inch above Send was read as tokens, and now that a token
count is drawn beside it, the two figures have to say which is which on the rendered screen.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, ViewportSize, async_playwright, expect

from tests.e2e.conftest import TWO_FRAMES, dock_locator, mock_llm, running_ui

pytestmark = pytest.mark.e2e

#: Wide enough that the rail is beside the conversation and the dock seats itself rather than
#: overlaying it, as in `test_room_failed_turn_row_e2e.py`.
VIEWPORT: ViewportSize = {"width": 1400, "height": 900}

#: Rows 1 and 2 are the two joins, 3 is the prompt, 4 is the turn that answered it.
ANSWER_SEQ = 4

PROMPT = "What did you use to answer this?"

#: How long the strip above Send is given to show the read that follows a landed turn.
#:
#: The strip is drawn from `GET /api/rooms/{id}/context`, and the head asks it twice: once
#: when the conversation opens, before any seat has spoken -- which is why the strip starts
#: at `0/20 turns` and `tokens not counted this run` -- and again when the answer's final
#: `AGENT_REPLY` arrives on the stream (`App.tsx`, `contextReadIsDue`). That same event is
#: what puts `why-4` on screen, so `_one_answered_turn` returning says nothing about whether
#: the second read has answered yet. Read at that instant, the strip showed the post-turn
#: figures on a quiet machine and the opening read's under a concurrent gate -- the two
#: failures #1415 recorded. So the assertions below wait for the post-turn figure to be
#: drawn rather than reading the strip once.
#:
#: A wait, not a weakening: the defect these tests exist for is a head that never finds the
#: seat's booking, and then the opening read's sentence is the last one ever drawn and the
#: wait runs out on it, with the same message the one-shot read gave.
STRIP_SETTLES_MS = 15000


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    with running_ui(storage_dir=tmp_path, llm=mock_llm()) as url:
        yield url


async def _one_answered_turn(page: Page, base_url: str) -> None:
    """Seat a 1:1 through the API, open it, and let the Core answer one message."""
    response = await page.request.post(
        f"{base_url}/api/rooms",
        data={"title": "Provenance", "agent_ids": ["scout"]},
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
async def test_why_opens_the_turn_s_record_on_the_dock_without_moving_the_conversation(
    server: str,
) -> None:
    """The record arrives on the dock, the row it belongs to does not move, and it is richer.

    No mutation is declared for the height assertion, and the reason is that the edit which
    would break it -- restoring the inline block -- is an addition rather than a change to a
    line, so there is no single needle it could name. What the two declared component tests
    hold between them is the wiring (`RoomConversation.test.tsx`'s *...and hands it to the
    dock*) and the drawing (`TurnDetail.test.tsx`'s six). What is left over, and only
    reachable here, is that those two are connected through `App.tsx` and that the connection
    costs the reader nothing on screen.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            await _one_answered_turn(page, server)

            row = page.locator(f"[data-testid='row-{ANSWER_SEQ}']")
            await page.evaluate(TWO_FRAMES)
            before = await row.bounding_box()
            assert before is not None, "the answered turn is not rendered"
            assert await page.locator("[data-testid='turn-detail']").count() == 0, (
                "the turn's record was on screen before anybody asked for it"
            )

            await page.click(f"[data-testid='why-{ANSWER_SEQ}']")
            dock = await dock_locator(page)
            await dock.wait_for(state="visible", timeout=10000)
            detail = dock.locator("[data-testid='turn-detail']")
            await detail.wait_for(timeout=10000)

            # It is on the dock, and only there: the row that offered it draws nothing.
            assert await row.locator("[data-testid='turn-detail']").count() == 0
            served = await detail.locator("[data-testid='turn-detail-served']").inner_text()
            assert "mock-gpt-4o" in served, served

            # And the turn the reader was looking at did not move to make room for it.
            await page.evaluate(TWO_FRAMES)
            after = await row.bounding_box()
            assert after is not None
            assert abs(after["height"] - before["height"]) < 1.0, (
                f"the row grew from {before['height']}px to {after['height']}px when its "
                "record opened; the record is meant to be drawn beside the conversation, "
                "not inside it"
            )
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_the_composer_names_the_turns_it_counts_and_the_tokens_the_core_booked(
    server: str,
) -> None:
    """The two figures beside Send say which is which, and the token one is the Core's.

    `used_tokens` comes from `TokenBudgetManager.get_budget` for the seat's own session, and
    the head says "tokens not counted this run" when there is no booking -- the honest thing
    to say about a conversation restored after a restart, and the wrong thing to say about
    the turn this test has just run. The distinction rests on
    `room/service.py::participant_session_id` naming the same session `BaseAgent` spends
    against, which no fixture can stand in for.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            await _one_answered_turn(page, server)

            count = page.locator("[data-testid='answering-context-count']")
            await count.wait_for(timeout=10000)
            # The unit is on screen with the digits, which is the whole reason they are there.
            assert re.search(r"\d+/\d+\s*turns", await count.inner_text()), await count.inner_text()

            tokens = page.locator("[data-testid='answering-tokens']")
            await tokens.wait_for(timeout=10000)
            try:
                # Waited for, not read once: see `STRIP_SETTLES_MS`.
                await expect(tokens).to_have_text(
                    re.compile(r"[\d,]+\s*tokens"), timeout=STRIP_SETTLES_MS
                )
            except AssertionError as timed_out:
                raise AssertionError(
                    "the composer reported no token record for a seat that had just answered a "
                    "turn; the session the head reads is not the one the Core spent against "
                    f"(it still reads {await tokens.inner_text()!r})"
                ) from timed_out
            spoken = await tokens.inner_text()
            assert "not counted" not in spoken, spoken
            spent = re.search(r"([\d,]+)\s*tokens", spoken)
            assert spent is not None, spoken
            assert int(spent.group(1).replace(",", "")) > 0, spoken
        finally:
            await browser.close()


@pytest.fixture
def server_serving_a_published_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    """The same app, configured as if the operator had pointed it at a hosted model.

    `update_settings` is the product's own path to this and is deliberately not used: it also
    writes `LLM_PROVIDER` and `ANTHROPIC_MODEL` into `os.environ`, which would outlive the
    test and reconfigure every suite that ran after it. The two fields the readout reads are
    set directly instead. The connector that answers the turn is still the mock, because
    nothing re-resolves it once the app is built -- which is what makes this a test of the
    readout rather than of Anthropic.
    """

    def configure(app: Any) -> None:
        manager = app.state.room_stack.session_manager()
        monkeypatch.setattr(manager, "_configured_provider", "anthropic")
        monkeypatch.setattr(manager, "_configured_model", "claude-3-5-sonnet-20241022")

    with running_ui(storage_dir=tmp_path, llm=mock_llm(), configure=configure) as url:
        yield url


@pytest.mark.asyncio
async def test_the_ring_counts_tokens_against_the_model_s_own_window_when_there_is_one(
    server_serving_a_published_window: str,
) -> None:
    """With a window for the seat's model, the ring's denominator is that window.

    The ring had no denominator but a turn ceiling, which is a cost ceiling wearing a
    context ceiling's clothes: a seat one turn into a 200,000-token window and a seat one
    turn from the end of one drew the same arc. `RoomConversation.test.tsx` can see the
    component switch units when a fixture hands it a window, and
    `test_ui_room_api.py` can see the route resolve one from the configured model, but
    neither can see that the field the route writes is the field the component reads --
    they agree because I wrote both fixtures, and a rename on either side leaves both
    green. Only the assembled product spends a real turn, books real tokens against it,
    and draws them against a window nobody in this test supplied.

    Both figures stay on screen: the ring takes the tokens, the count beside it keeps the
    turns, so the turn ceiling this replaced as a denominator is not lost.

    Breaking it: drop `max_context_tokens` from the seat dict in
    `src/uclone_x/ui/rooms.py::read_context` and the readout falls back to turns, so the
    `tokens` assertion below fails while
    `test_the_composer_names_the_turns_it_counts_and_the_tokens_the_core_booked` -- which
    runs against a provider nobody publishes a window for -- still passes. No needle is
    declared because the E2E suite serves the committed bundle, which the ratchet does not
    rebuild.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            await _one_answered_turn(page, server_serving_a_published_window)

            count = page.locator("[data-testid='answering-context-count']")
            await count.wait_for(timeout=10000)
            try:
                # Waited for, not read once: see `STRIP_SETTLES_MS`.
                await expect(count).to_have_text(
                    re.compile(r"[\d,]+/[\d,]+\s*tokens"), timeout=STRIP_SETTLES_MS
                )
            except AssertionError as timed_out:
                raise AssertionError(
                    f"the ring read out {await count.inner_text()!r}; with a window for this "
                    "model it is counting tokens, and the unit has to say so on screen"
                ) from timed_out
            spoken = await count.inner_text()
            drawn = re.search(r"([\d,]+)/([\d,]+)\s*tokens", spoken)
            assert drawn is not None, spoken
            assert drawn.group(2) == "200,000", spoken
            assert int(drawn.group(1).replace(",", "")) > 0, (
                "the numerator is the seat's own spend and it had just answered a turn"
            )

            # The turn ceiling did not leave the screen when the ring stopped drawing it.
            turns = page.locator("[data-testid='answering-tokens']")
            await turns.wait_for(timeout=10000)
            assert re.search(r"\d+/\d+\s*turns", await turns.inner_text()), await turns.inner_text()

            # And the arc is that fraction, not the turn fraction it would otherwise be.
            ring = page.locator("[data-testid='answering-context-ring']")
            filled = await ring.get_attribute("data-filled")
            assert filled is not None
            expected = int(drawn.group(1).replace(",", "")) / 200_000
            assert abs(float(filled) - expected) < 0.002, (
                f"the ring is {filled} full while the figure beside it reads {spoken!r}; "
                "the arc and the digits are drawn from different numbers"
            )
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_why_on_turn_with_documents_shows_step_and_doc_and_open_in_docs(
    tmp_path: Path,
) -> None:
    """Opening why on a turn that wrote a document shows its step and doc rows, and Open lands in Docs."""
    from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
    from uclone_x.room.models import (
        Participant,
        ParticipantKind,
        RoomMessage,
        RoomMessageKind,
        RoomState,
        RoomToolUse,
        RoomWrittenFile,
        SelectionVerdict,
        SpeakerDecision,
        TurnState,
    )
    from uclone_x.room.store import RoomStore

    room_id = "room_docs_e2e"
    storage_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    doc_path = workspace_dir / "docs" / "guide.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("# Project Guide\n\nWelcome to the documentation.\n", encoding="utf-8")

    room_store = RoomStore(storage_dir / "rooms")
    room_store.storage_dir.mkdir(parents=True, exist_ok=True)

    state = RoomState(
        room_id=room_id,
        title="Documentation Room",
        participants=(
            Participant(id="user", kind=ParticipantKind.HUMAN, display_name="Kenny"),
            Participant(id="scout", kind=ParticipantKind.AGENT, display_name="Scout"),
        ),
        transcript=(
            RoomMessage(seq=1, kind=RoomMessageKind.JOIN, sender_id="user", content=""),
            RoomMessage(seq=2, kind=RoomMessageKind.JOIN, sender_id="scout", content=""),
            RoomMessage(
                seq=3, kind=RoomMessageKind.UTTERANCE, sender_id="user", content="Write guide.md"
            ),
            RoomMessage(
                seq=4,
                kind=RoomMessageKind.UTTERANCE,
                sender_id="scout",
                content="I wrote the guide for you.",
                turn_id="turn_4",
                completed=True,
                tools_recorded=True,
                provenance=Provenance(
                    path=ExecutionPath.PRIMARY,
                    requested=ServiceRef(provider="ollama", model="qwen3:8b"),
                    served_by=ServiceRef(provider="ollama", model="qwen3:8b"),
                ),
                decision=SpeakerDecision(
                    verdict=SelectionVerdict.SPEAK,
                    speaker_id="scout",
                    confidence=1.0,
                    selector="sole_agent",
                    reasoning="",
                ),
            ),
        ),
        tool_uses=(
            RoomToolUse(
                turn_id="turn_4",
                participant_id="scout",
                tool_name="file_write",
                tool_call_id="call_write_guide",
                status="success",
                duration_ms=180,
                arguments_preview='{"TargetFile": "docs/guide.md"}',
                output_preview="Saved 45 bytes",
                written_path="docs/guide.md",
            ),
        ),
        written_files=(
            RoomWrittenFile(
                path="docs/guide.md",
                participant_id="scout",
                tool_name="file_write",
                turn_id="turn_4",
            ),
        ),
        turn_state=TurnState(agent_turns_since_human=1, last_speaker_id="scout"),
    )
    room_store.save(state)

    with running_ui(
        storage_dir=storage_dir, llm=mock_llm(), workspace_dir=workspace_dir
    ) as base_url:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page = await browser.new_page(viewport=VIEWPORT)
                await page.goto(base_url, wait_until="commit")
                await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
                await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)

                # Find why-4 and click it
                why_button = page.locator("[data-testid='why-4']")
                await why_button.wait_for(timeout=10000)
                await why_button.click()

                dock = await dock_locator(page)
                await dock.wait_for(state="visible", timeout=10000)

                # Verify turn detail is visible on the dock
                turn_detail = dock.locator("[data-testid='turn-detail']")
                await turn_detail.wait_for(timeout=10000)

                # Verify step row shows classifyTool label or filename
                step_row = turn_detail.locator("[data-testid='turn-step-row']")
                await step_row.wait_for(timeout=10000)
                step_text = await step_row.inner_text()
                assert "File Mutation" in step_text or "guide.md" in step_text, step_text

                # Verify document row shows document path
                doc_row = turn_detail.locator("[data-testid='turn-doc-row']")
                await doc_row.wait_for(timeout=10000)
                doc_text = await doc_row.inner_text()
                assert "docs/guide.md" in doc_text, doc_text

                # Click Open on the document row
                open_btn = doc_row.locator("[data-testid='open-doc-docs/guide.md']")
                await open_btn.wait_for(timeout=10000)
                await open_btn.click()

                # Verify Docs surface (doc-viewer) is mounted on the dock
                doc_viewer = dock.locator("[data-testid='doc-viewer']")
                await doc_viewer.wait_for(timeout=10000)
                assert await doc_viewer.is_visible()
            finally:
                await browser.close()
