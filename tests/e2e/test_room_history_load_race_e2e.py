"""A read of the conversation answered before the user acted must not undo what they did.

`test_history_load_race_e2e.py` guards #946 and #975 on `PlaygroundTab`, whose load-time
`GET /api/session/history` could be answered before a Clear or a Send and delivered after it,
putting the removed turn back or taking the sent one away. The retirement (#1208) deletes that
surface, so the race is checked here against the one that survives before it goes.

The room path held the same shape in a different place, and these two cases were written red
against it. `RoomConversation` renders nothing but what the server last said, and every read of
the record landed through `commitRoom` in `App.tsx`, which drops an answer only when
`roomResponseIsStale` says the *conversation* moved on -- a different generation, or a
different open room. Clearing, rewinding and sending moved neither, so a
`GET /api/rooms/{id}` issued before one of those acts was still accepted after it. A read is
in flight at exactly the moment the user regains the controls: the stream's `AGENT_REPLY`
`final` both clears `live.turn` -- which is what re-enables Clear and Rewind, `running` in
`RoomConversation` -- and fires `refreshRoom`.

What answers it is `commitRoomAct`, which the acts commit through instead: it bumps the
generation, so every read that started before the act carries a number that is no longer
current and is dropped. `commitRoom` keeps the old rule and is now the read path's alone.

Nothing here decides content at the HTTP layer. Each read is performed against the real server
(`route.fetch`) and its own bytes are handed to the page (`route.fulfill(response=...)`); only
*when* the page receives them is the test's. The reply is chosen by scripting the model
(`scripted_reply_ui_server`), as the other room suites do.

Three of the five playground cases have no room counterpart, and the module docstring of the
original says why in its own terms; the reasons are recorded at the foot of this file.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from playwright.async_api import Page, Request, Route, ViewportSize, async_playwright

from tests.e2e.conftest import UIServerFactory

pytestmark = pytest.mark.e2e

TRANSCRIPT = "[data-testid='transcript']"
COMPOSER = "[data-testid='room-composer']"
SEND = "[data-testid='send-message']"
REPLY = "Mock response from UClone-X BaseAgent."
FIRST = "Said before the read was owed"
SECOND = "Sent while an older read was still owed"
_EVENT_TIMEOUT_S = 20.0

#: The rail row is clicked, so the window must be one the rail is open in: below 600px it
#: starts closed (#1062) and the row is not on screen at all.
VIEWPORT: ViewportSize = {"width": 1280, "height": 900}


async def _next_frame(page: Page) -> None:
    """Let the page run what a delivered response queued, and paint it."""
    await page.evaluate(
        "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 0))))"
    )


@dataclass
class _Held:
    """One read, stopped between the server answering it and the page receiving it."""

    request: Request
    body: str = ""
    answered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


class HeldRoomReads:
    """Hold `GET /api/rooms/{room_id}` between the server's answer and the page.

    `pass_through` reads go by untouched, because the surface does not exist until the first
    of them lands: `App.tsx` renders `room-loading` while `room` is null, so there is no
    composer and no Clear to race an unanswered opening read with. What is raced here is a
    later read, which is the only kind a user can act against.

    Only the delivery is the test's. The read is made against the real server and its own
    bytes are what the page finally gets, so what lands is a real answer that has merely
    become old -- which is the whole premise.
    """

    def __init__(
        self,
        page: Page,
        room_id: str,
        *,
        pass_through: int,
        spare_the_sends_reread: bool = False,
        hold_at_most: int | None = None,
    ) -> None:
        self._page = page
        self._hold_at_most = hold_at_most
        self._room_id = room_id
        self._pass_through = pass_through
        self._spare_the_sends_reread = spare_the_sends_reread
        self._a_send_is_in_flight = False
        self._sends_reread_due = False
        self._seen = 0
        self._holding = True
        self.held: list[_Held] = []
        self._any_answered = asyncio.Event()

    async def install(self) -> None:
        if self._spare_the_sends_reread:
            self._page.on("request", self._note)
        await self._page.route(f"**/api/rooms/{self._room_id}", self._handle)

    def _note(self, request: Request) -> None:
        """Watch for the shape that identifies `onSend`'s own re-read of the conversation.

        `onSend` in `App.tsx` does three things in order: `POST .../messages`, then
        `await fetchRooms()` -- a `GET /api/rooms` -- and only then re-reads the conversation.
        So the room read that follows the *list* read that follows the *send* is the one it
        is waiting on. Watched rather than counted, because the two reads a landed turn makes
        (`refreshRoom` from the stream's `final`, and this one) race each other, and an
        ordinal cannot tell them apart. Watched from the POST rather than from the list alone,
        because the head lists the rooms on load as well, and that list precedes no send.
        """
        if request.method == "POST" and request.url.endswith(
            f"/api/rooms/{self._room_id}/messages"
        ):
            self._a_send_is_in_flight = True
        elif (
            self._a_send_is_in_flight
            and request.method == "GET"
            and request.url.endswith("/api/rooms")
        ):
            self._a_send_is_in_flight = False
            self._sends_reread_due = True

    async def _handle(self, route: Route) -> None:
        # The same URL carries the rename `PATCH` that a conversation's first message sends
        # (`seededTitle` in `App.tsx`). Only a read is a read.
        if route.request.method != "GET":
            await route.continue_()
            return
        if self._sends_reread_due:
            # Never held, and never counted either: holding it would leave `onSend` pending,
            # so the composer would stay disabled and the user could not act at all -- which
            # is not the race under test but the absence of one.
            self._sends_reread_due = False
            await route.continue_()
            return
        self._seen += 1
        held_enough = self._hold_at_most is not None and len(self.held) >= self._hold_at_most
        if self._seen <= self._pass_through or not self._holding or held_enough:
            await route.continue_()
            return
        entry = _Held(request=route.request)
        self.held.append(entry)
        response = await route.fetch()
        entry.body = await response.text()
        entry.answered.set()
        self._any_answered.set()
        await entry.release.wait()
        await route.fulfill(response=response)

    async def an_answer_is_owed(self) -> None:
        """Return once the server has answered a held read that the page has not received."""
        await asyncio.wait_for(self._any_answered.wait(), _EVENT_TIMEOUT_S)

    def stop_holding(self) -> None:
        """Let every later read through, leaving what is already held still owed."""
        self._holding = False

    async def deliver(self) -> None:
        """Hand every held answer to the page, and let the page act on each."""
        for entry in self.held:
            await self._hand_over(entry)

    async def _hand_over(self, entry: _Held) -> None:
        """Release one answer and return once the page has finished receiving it.

        Waiting on `requestfinished` rather than on what it renders: the page decides what to
        do with an answer, and a test that waited for a row would only ever observe the
        outcome it expected.
        """
        await asyncio.wait_for(entry.answered.wait(), _EVENT_TIMEOUT_S)
        async with self._page.expect_event(
            "requestfinished",
            lambda request: request == entry.request,
            timeout=_EVENT_TIMEOUT_S * 1000,
        ):
            entry.release.set()
        await _next_frame(self._page)


async def _create_room(page: Page, base_url: str, title: str, agents: list[str]) -> str:
    """Seat a conversation through the API, as any other client would."""
    response = await page.request.post(
        f"{base_url}/api/rooms", data={"title": title, "agent_ids": agents}
    )
    assert response.ok, await response.text()
    body: dict[str, Any] = await response.json()
    return str(body["room_id"])


async def _open(page: Page, base_url: str, room_id: str) -> None:
    """Load the head and wait for it to open the one conversation this install has.

    It used to click the rail row. Retiring `PlaygroundTab` (#1208) made the conversation
    the centre column unconditionally, so `App.tsx` now opens the most recent one on load
    -- here the only one -- and a click would be a *second* `GET /api/rooms/{id}` on top
    of the one the auto-open already made. The read budget below is what the held-reads
    harness counts, so an extra read is not cosmetic: it spends a pass-through and the
    turn never renders. The row is still asserted to exist, since a rail that lost it
    would otherwise go unnoticed.
    """
    await page.goto(base_url, wait_until="commit")
    await page.wait_for_selector(f"[data-testid='conversation-{room_id}']", timeout=15000)
    await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)


async def _say(page: Page, words: str) -> None:
    await page.fill(COMPOSER, words)
    await page.click(SEND)


async def _showing(page: Page, text: str) -> int:
    return await page.locator(TRANSCRIPT).get_by_text(text, exact=True).count()


@pytest.mark.asyncio
async def test_a_clear_is_not_undone_by_a_read_the_server_answered_before_it(
    scripted_reply_ui_server: UIServerFactory,
) -> None:
    """#946 on the room: the answer that was already old when Clear ran must not restore it.

    Rows 1 and 2 are the joins, 3 is the message and 4 the reply. Two reads of the record go
    out once the reply lands -- `refreshRoom`, from the stream's `final`, and the one `onSend`
    makes after its own POST -- and the same `final` is what puts Clear back within reach by
    clearing `live.turn`. So one of them is held, the conversation is cleared while it is
    owed, and then it is paid.

    Since #1412 it is the *first* of the two that is held. A read lands only while it is the
    last one issued for its room, so holding the second would keep the turn off screen and
    leave nothing to clear. The first is still an answer from before the Clear, which is the
    premise. What drops it now is both guards: it was overtaken, and the Clear moved the
    generation. The generation alone is held by `App.test.tsx`, which can issue a read that
    is the last one out when the Clear lands.
    """
    base_url = scripted_reply_ui_server(REPLY)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            room_id = await _create_room(page, base_url, "Race", ["scout"])
            # One read opens the conversation; of the two the turn makes, the first is held and
            # the second, the last one issued, renders the turn.
            reads = HeldRoomReads(page, room_id, pass_through=1, hold_at_most=1)
            await reads.install()
            await _open(page, base_url, room_id)

            await _say(page, FIRST)
            await page.locator(TRANSCRIPT).get_by_text(FIRST, exact=True).wait_for(timeout=20000)
            await reads.an_answer_is_owed()

            # The premise, stated rather than assumed: what is owed really does describe the
            # conversation as it was before the clear. Without this the test could pass on an
            # answer that happened to be empty, which proves nothing about the ordering.
            assert any(FIRST in entry.body for entry in reads.held), (
                "no held read carries the message, so nothing owed could undo the clear"
            )

            await page.click("[data-testid='clear-history']")
            # Auto-waited rather than raced: the confirmation refuses while a turn is running
            # (`running` in `RoomConversation`), and that is the same flag the landed reply
            # clears.
            await page.click("[data-testid='confirm-history-change-yes']")
            await page.wait_for_selector("[data-testid='transcript-empty']", timeout=15000)

            await reads.deliver()

            assert await _showing(page, FIRST) == 0, (
                "the cleared message came back when the older read landed"
            )
            assert await _showing(page, REPLY) == 0, (
                "the cleared reply came back when the older read landed"
            )
            assert await page.locator("[data-testid='transcript-empty']").count() == 1, (
                "the conversation stopped saying it was empty after an older read landed"
            )
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_message_sent_while_an_older_read_was_owed_stays_on_screen(
    scripted_reply_ui_server: UIServerFactory,
) -> None:
    """#975 on the room: paying a read from the previous turn must not take the next one away.

    The read held here was answered while the first exchange was the whole conversation, so
    it is a truthful answer that has simply gone out of date. The user then sends a second
    message, which lands, is recorded, and renders. Delivering the owed answer afterwards
    replaces the transcript with the one that predates it -- and the message the user watched
    arrive leaves the screen, with nothing on the room path holding a local copy of it the way
    `localTurnsRef` does on the playground.
    """
    base_url = scripted_reply_ui_server(REPLY)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            room_id = await _create_room(page, base_url, "Race", ["scout"])
            # One read opens the conversation; the next read that is not the send's own is
            # held. This case has to *send again* while an answer is owed, and `onSend` in
            # `App.tsx` awaits a read of its own after the POST -- holding that one would
            # leave `onSend` pending, so the composer would never re-enable and the test
            # would fail on a disabled Send rather than on the race it is about. Sparing it
            # puts the hold on the stream's `refreshRoom`, which equally predates `SECOND`
            # and is asserted to below.
            reads = HeldRoomReads(page, room_id, pass_through=1, spare_the_sends_reread=True)
            await reads.install()
            # A room read lands only while it is the last one issued for the room (#1412). The
            # held refresh must therefore go out *before* the send's re-read, or it is the last
            # read out, nothing else can land, and the turn never closes. The first send's
            # answer is kept from the page until the refresh is held, which puts the send's
            # re-read after it on every run rather than on the runs that happen to race that way.
            first_send = True

            async def _answer_the_first_send_after_the_hold(route: Route) -> None:
                nonlocal first_send
                if not first_send:
                    await route.continue_()
                    return
                first_send = False
                response = await route.fetch()
                await reads.an_answer_is_owed()
                await route.fulfill(response=response)

            await page.route(
                f"**/api/rooms/{room_id}/messages", _answer_the_first_send_after_the_hold
            )
            await _open(page, base_url, room_id)

            await _say(page, FIRST)
            await page.locator(TRANSCRIPT).get_by_text(FIRST, exact=True).wait_for(timeout=20000)
            await reads.an_answer_is_owed()
            assert any(FIRST in entry.body for entry in reads.held), (
                "no held read describes the conversation, so nothing owed could undo the send"
            )
            assert not any(SECOND in entry.body for entry in reads.held), (
                "the held read already knows the second message, so it is not the older answer"
            )
            # Only the one read is owed. The second message has to reach the screen, and on
            # this surface it reaches it through a read of the record like any other.
            reads.stop_holding()

            await _say(page, SECOND)
            await page.locator(TRANSCRIPT).get_by_text(SECOND, exact=True).wait_for(timeout=20000)

            await reads.deliver()

            assert await _showing(page, SECOND) == 1, (
                "the message the user sent left the screen when the older read was paid"
            )
            assert await _showing(page, FIRST) == 1, (
                "the earlier message was lost or doubled when the older read was paid"
            )
        finally:
            await browser.close()


# Left unported, deliberately.
#
# `test_a_quick_turn_that_fails_stays_on_screen_and_the_earlier_turns_come_back`: the
# playground renders a failed turn out of its own state, and owes itself a history load it
# deferred while the turn ran -- so replaying that load after the failure can erase it, and
# take the earlier turns with it.
#
# A failed turn *is* in `room` here: the Core mints a transcript row carrying `error`, which
# `RoomConversation` renders as `row-error-{seq}` (`test_room_failed_turn_row_e2e.py` asserts
# `row-error-4`). What is absent is the other half -- there is no owed-load bookkeeping on this
# path. Nothing is deferred and replayed: the transcript is whatever the server last said, so
# every read that lands, current or stale, carries the failed row with it rather than a state
# that predates it. (`live.error`/`cascade-error` is a second, transport-level slice that
# `commitRoom` never writes, but it is not what this case was about.) A room version would be
# asserting that a row the server holds is still there after the server is read again.
#
# `test_an_unsaved_retry_of_the_saved_prompt_leaves_the_exchange_it_repeats_on_screen` and
# `test_a_saved_retry_of_the_saved_prompt_is_shown_once_beside_the_exchange_it_repeats`
# (#1000): both are about reconciling a locally rendered turn with the server's copy of it,
# and they differ only in what the match is keyed on -- prompt text, which took the wrong
# exchange, against the turn id the server kept. `RoomConversation` reconciles nothing: it
# maps `room.transcript` keyed on `message.seq`, every row of which the Core minted, so a
# retry that repeats an earlier prompt word for word is a different `seq` and there is no
# matching step to key wrongly. Writing either case here would describe a situation that
# cannot arise.
