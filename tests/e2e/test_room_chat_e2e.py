"""End-to-end tests for a conversation that can seat more than one agent.

Written after a review found the surface blank-screened on its first agent reply while
the whole gate was green: 3,811 unit and component tests, pyright strict clean, and no
browser case that ever rendered a landed reply. The component tests could not catch it
because they build their own fixtures, and the fixture said `served_by` was a string
where the API sends an object. Only a real render against a real response can.

So the point of this module is narrow and deliberate: drive the actual routes, render the
actual payloads, and fail on an uncaught page error. It mocks nothing. The model is the
fixture's scripted connector, because a browser case that waits on a real provider is a
flaky gate and there is no CI here to re-run it.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, Response, Route, async_playwright

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


@pytest.mark.asyncio
async def test_a_conversation_renders_a_landed_reply_without_a_page_error(
    ui_test_server: str,
) -> None:
    """The case the review found: one real reply, rendered.

    The defect was that the attribution line rendered `provenance.served_by` -- a
    `ServiceRef` object -- as a React child. React throws, there is no error boundary, and
    the conversation unmounts: a blank screen on the first successful answer. `pageerror`
    is therefore an assertion here and not diagnostics.

    Where that line is printed moved with the two shapes of §3.2.3 [Rev 20]: a 1:1 states
    its model once in the header rather than under every turn, so that is where the object
    now reaches React, and that is what this reads. The object is the same object.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _create_room(page, ui_test_server, "Index tuning", ["scout"])
        await page.goto(ui_test_server, wait_until="commit")
        await page.wait_for_selector(f"[data-testid='conversation-{room_id}']", timeout=15000)
        await page.click(f"[data-testid='conversation-{room_id}']")
        await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)

        await page.fill("[data-testid='room-composer']", "check the index on users")
        await page.click("[data-testid='send-message']")

        # Row 3 is the message, row 4 the reply: rows 1 and 2 are the two joins.
        await page.wait_for_selector("[data-testid='row-4']", timeout=20000)
        reply = await page.locator("[data-testid='row-4']").inner_text()
        assert reply.strip(), "the reply rendered no text at all"

        # The attribution line is the thing that threw. It must render, and it must
        # render text rather than an object's shape.
        served_by = await page.locator("[data-testid='header-attribution']").inner_text()
        assert served_by.strip(), "a landed reply carried no attribution"
        assert "[object" not in served_by

        # And it is stated once, not under each turn: the 1:1 has one model and it is in
        # the header above every row it is true of.
        assert await page.locator("[data-testid='served-by-4']").count() == 0, (
            "a 1:1 restated its unchanging model under the reply"
        )
        assert "scout" not in reply, (
            "a 1:1 named the only agent on its own turn, which the header already names"
        )

        assert not errors, f"the conversation raised in the browser: {errors}"
        assert await page.locator("[data-testid='transcript']").is_visible(), (
            "the conversation unmounted after its first reply"
        )
        await browser.close()


@pytest.mark.asyncio
async def test_a_group_conversation_names_each_speaker_and_the_model_that_served_them(
    ui_test_server: str,
) -> None:
    """The other shape of §3.2.3 [Rev 20], rendered against the same routes.

    Two agents seated, so the left-hand side of the transcript holds more than one sender:
    each turn is named, and each turn carries its own attribution, because with the
    speaker changing turn to turn there is nothing constant for a header to hoist.

    The message addresses somebody on purpose. With two agents, no address and no
    designated responder, the Core's selector chain abstains and the conversation settles
    on silence -- which is what the surface then renders, correctly, and is not this case.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _create_room(
            page, ui_test_server, "Architecture triage", ["scout", "champion"]
        )
        await page.goto(ui_test_server, wait_until="commit")
        await page.wait_for_selector(f"[data-testid='conversation-{room_id}']", timeout=15000)
        await page.click(f"[data-testid='conversation-{room_id}']")
        await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)

        await page.fill("[data-testid='room-composer']", "@scout check the index on users")
        await page.click("[data-testid='send-message']")

        # The reply's `seq` depends on how many joins the conversation opened with, so it
        # is read off the attribution line rather than counted: a hard-coded row number
        # fails as a timeout, which says nothing about what is on screen.
        attribution = page.locator("[data-testid^='served-by-']").first
        await attribution.wait_for(timeout=20000)
        served_by = await attribution.inner_text()
        seq = (await attribution.get_attribute("data-testid") or "").removeprefix("served-by-")

        reply = page.locator(f"[data-testid='row-{seq}']")
        speaker = await reply.inner_text()
        assert "scout" in speaker or "champion" in speaker, (
            f"a group turn named nobody, so two senders share one column: {speaker!r}"
        )

        assert served_by.strip(), "a group turn carried no attribution"
        assert "[object" not in served_by
        assert await page.locator("[data-testid='header-attribution']").count() == 0, (
            "a group hoisted one model into the header, where it is false of the next turn"
        )

        assert not errors, f"the conversation raised in the browser: {errors}"
        await browser.close()


@pytest.mark.asyncio
async def test_a_conversation_can_seat_another_agent_while_it_is_open(
    ui_test_server: str,
) -> None:
    """F2: add someone mid-conversation, chosen from a list rather than typed."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        # An agent is only offered once the runtime has one running, and it reports none
        # until one has answered something. One ordinary chat turn is what makes the
        # invite list have anybody in it -- which is a fact about this product, not about
        # this test, and is why the empty list states that cause rather than claiming
        # everyone is already seated.
        warmed = await page.request.post(
            f"{ui_test_server}/api/turn",
            data={"message": "hello", "agent_id": "champion"},
        )
        assert warmed.ok, await warmed.text()

        room_id = await _create_room(page, ui_test_server, "Architecture triage", ["scout"])
        await page.goto(ui_test_server, wait_until="commit")
        await page.wait_for_selector(f"[data-testid='conversation-{room_id}']", timeout=15000)
        await page.click(f"[data-testid='conversation-{room_id}']")
        await page.wait_for_selector("[data-testid='participant-strip']", timeout=15000)

        await page.click("[data-testid='add-someone']")
        await page.wait_for_selector("[data-testid='invite-list']", timeout=10000)
        invitable = page.locator("[data-testid='invite-list'] button").first
        invited = (await invitable.inner_text()).strip()
        assert invited, "the invite list offered nobody, so nothing was chosen from it"
        await invitable.click()

        strip = page.locator("[data-testid='participant-strip']")
        await strip.get_by_text("scout").first.wait_for(timeout=10000)
        assert not errors, f"the conversation raised in the browser: {errors}"
        await browser.close()


def _is_persona_load(response: Response) -> bool:
    """The persona list the head adopts its first name from, at load (`App.tsx`)."""
    return response.request.method == "GET" and "/api/personas" in response.url


def _is_room_create(response: Response) -> bool:
    """The conversation New posts (`handleNewRoom` in `App.tsx`)."""
    return response.request.method == "POST" and response.url.endswith("/api/rooms")


#: The rail row of the name the head has adopted. `aria-current` is set on it and nowhere
#: else (`WorkspaceSidebar.tsx`, #1143), so it is the page saying which name New will seat
#: -- not a colour, and not this test's guess at which one that is.
ADOPTED_NAME_ROW = "[data-testid^='persona-item-'][aria-current='true']"


@pytest.mark.asyncio
async def test_new_starts_a_conversation_that_its_first_message_names(
    ui_test_server: str,
) -> None:
    """The way a user starts a conversation, driven the way a user drives it.

    Every other case here seats its conversation through `POST /api/rooms`, so the one path
    a first-time user takes had no browser coverage. It was gated behind a native
    `window.prompt`, which Playwright dismisses by default -- so under that build this case
    opens nothing, and a dialog is recorded. On a fresh install the New control lived inside
    a list that rendered only once a conversation already existed; that is checked first.

    The wait before the click is the fix for #1145, and it is not caution. New seats
    `selectedAgent ? [selectedAgent] : []` (`handleNewRoom`), and `selectedAgent` is adopted
    only once the whole load-time `Promise.all` in `App.tsx` has resolved -- nine requests,
    of which the persona list is the one the adopted name comes from. The conversation list
    renders before that, so a click on New as soon as it appears seated *nobody*, and the
    conversation then correctly said `No one is in this conversation yet.` rather than
    `Nothing has been said here yet` -- two different empty states, and this case read the
    wrong one about one lane run in four.

    So the adoption is waited for and then asserted, rather than raced: the name the rail
    says the head has adopted must be the agent the conversation New created holds. Widening
    the assertion to accept either line would go green while saying nothing about which
    state the page was in, which is the failure this case exists to catch.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        dialogs: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.on("dialog", lambda dialog: dialogs.append(dialog.message))

        async with page.expect_response(_is_persona_load, timeout=15000) as loading:
            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='conversation-list']", timeout=15000)
        assert (await loading.value).ok, "the head's load-time persona request failed"

        new_control = page.locator("[data-testid='new-conversation-button']")
        assert await new_control.is_visible(), "a fresh install offers no way to start one"
        assert await page.locator("[data-testid='new-session-button']").count() == 0, (
            "the rail still offers a second New control beside the conversation list's"
        )

        # The response having arrived is not the name having been adopted: the head sets
        # `selectedAgent` after the whole bootstrap resolves. The rail marks the adopted row,
        # so that is what is waited on, and its name is what the conversation is checked
        # against below.
        adopted_row = page.locator(ADOPTED_NAME_ROW).first
        await adopted_row.wait_for(timeout=15000)
        marked = await adopted_row.get_attribute("data-testid")
        assert marked is not None, "the rail marked the adopted row with no test id"
        adopted = marked.removeprefix("persona-item-")
        assert adopted, f"the rail marked a row that names nobody: {marked!r}"

        async with page.expect_response(_is_room_create, timeout=15000) as creating:
            await new_control.click()
        created = await creating.value
        assert created.ok, f"New could not start a conversation: {await created.text()}"
        # The tie. New seats whoever the head had adopted when it was clicked, so a
        # conversation holding nobody means the click beat the adoption -- and every
        # assertion below would then be reading the *no participants* empty state instead of
        # the one this case is about.
        body: dict[str, Any] = await created.json()
        seated = [p["id"] for p in body["participants"] if p["kind"] == "agent"]
        assert seated == [adopted], (
            f"New seated {seated} in the conversation while the rail said the head had "
            f"adopted {adopted!r}, so the conversation read below is not the one set up here"
        )

        conversation = page.locator("[data-testid='room-conversation']")
        await conversation.wait_for(timeout=15000)
        assert not dialogs, f"starting a conversation asked for input in a dialog: {dialogs}"
        # The other half of the tie. `seated` above is a claim about the conversation the
        # create answered with; everything below reads the one on screen. Those were only
        # ever the same room by the head not having opened another, which on a fresh install
        # it could (#1288). Asserted rather than assumed.
        assert await conversation.get_attribute("data-room-id") == body["room_id"], (
            f"the conversation on screen is not the one New created ({body['room_id']!r}), "
            f"so the assertions below are about a different room"
        )

        # On screen as well as in the record: the conversation shows who it seated.
        strip = page.locator("[data-testid='participant-strip']")
        await strip.get_by_text(adopted).first.wait_for(timeout=10000)

        # A conversation with nothing in it says so, rather than rendering a blank column.
        # Which of the two lines it says is what #1145 was reading at random: this one is the
        # *has participants, no messages* state, and the tie above is what makes it the state
        # the page must be in.
        empty = page.locator("[data-testid='transcript-empty']")
        await empty.wait_for(timeout=10000)
        assert "Nothing has been said here yet" in await empty.inner_text()

        await page.fill("[data-testid='room-composer']", "check the index on users")
        await page.click("[data-testid='send-message']")

        # Rows 1 and 2 are the two joins, 3 the message, 4 the reply.
        await page.wait_for_selector("[data-testid='row-4']", timeout=20000)
        assert await empty.count() == 0, "the empty-conversation line outlived the first message"

        # The first message named the conversation in the rail.
        rail = page.locator(
            "[data-testid='sessions-list'] button", has_text="check the index on users"
        )
        await rail.first.wait_for(timeout=10000)

        assert not errors, f"the conversation raised in the browser: {errors}"
        await browser.close()


async def _next_frame(page: Page) -> None:
    """Let the page run what a delivered response queued, and paint it."""
    await page.evaluate(
        "() => new Promise(r => requestAnimationFrame("
        "() => requestAnimationFrame(() => setTimeout(r, 0))))"
    )


@pytest.mark.asyncio
async def test_new_on_a_fresh_install_starts_one_conversation_and_not_two(
    fresh_ui_server: str,
) -> None:
    """The first thing a new user does, with the load still finishing underneath it (#1288).

    Every other case in this module runs on the module server, where three conversations
    already exist: the head reopens one, `currentRoomId` is set before the user can reach
    New, and the window this case is about has closed before the page is even interactive.
    A fresh install is the one run where it is open -- and it is the run a new user makes.

    The head auto-opens a conversation once both lists have landed, and `handleNewRoom` is
    what it calls when there are none. Between a user's click on New and `openRoom` setting
    `currentRoomId` there are two awaits, and the load finishes inside them: the effect then
    reads *no conversation open, nothing on its way* and starts a second conversation beside
    the one already being created. The rail ends up holding an empty `New conversation` the
    user never asked for.

    What closes that window is `createInFlightRef` (`App.tsx`, #1288): taken synchronously on
    entry to `handleNewRoom`, read as a third term in the auto-open effect's guard, released
    in `finally`. Both halves are load-bearing here and each is checked below on its own.

    Nothing here decides content. The first `GET /api/rooms` and the create are answered by
    the real server; the test owns only *when* the page receives them, which is what puts the
    finish of the load inside the create rather than before or after it. Held from the
    browser rather than slowed on the server, because the window is sub-frame on a warm
    localhost and a sleep would only make it likely.

    Recovered from the closed, unmerged `task/1288-builder-dev-ui-1` (PR #1299, `f1ea1b6f`)
    per #1301. That version held `GET /api/sessions`, which #1374 took out of the load, so it
    held nothing: at `ae75e25e` it failed its own precondition 5 of 5 runs, never reaching
    the window. It now holds the listing read, the other flag the auto-open waits on.

    Mutation-checked by hand rather than as a kill declaration, because the lethality ratchet
    runs a browser test against the **committed bundle** without rebuilding it, so a
    declaration naming `frontend/src` reads as escaped there (the reason
    `tests/e2e/test_rail_responsive_e2e.py` records). Measured at `ae75e25e` with the bundle
    rebuilt for each: each mutation below failed this case 5 of 5 standalone runs and in a
    full-file run (`1 failed, 8 passed`), always at the two-creates assertion. So did
    removing #1288's own lines from `handleNewRoom` and the guard. A wholesale revert of
    `App.tsx` to `8cf40d0f` no longer compiles against the rest of `frontend/src`, so it
    was not run.
    `Killed by:` frontend/src/App.tsx ::
    `if (currentRoomId !== null || autoOpenRef.current || createInFlightRef.current) return;`
    becoming `if (currentRoomId !== null || autoOpenRef.current) return;`
    `Killed by:` frontend/src/App.tsx ::
    `createInFlightRef.current = true;` becoming `createInFlightRef.current = false;`
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        # The auto-open waits on two flags: `roomsListed`, set once the first
        # `GET /api/rooms` settles, and `metadataListed`, set once `fetchAllMetadata`
        # settles. The listing read is held, so the load is still unfinished when the user
        # clicks New, and its finish can then be put inside the create.
        finish_the_load = asyncio.Event()
        # `handleNewRoom`'s own `POST /api/rooms`, held so the create is still in flight when
        # the load finishes. A create that has already returned is not the case under test:
        # `openRoom` has set `currentRoomId` by then, which is the guard the effect reads.
        finish_the_create = asyncio.Event()
        listing_reads: list[str] = []
        creates: list[str] = []

        async def hold_the_rooms_route(route: Route) -> None:
            if route.request.method == "POST":
                creates.append(route.request.url)
                await finish_the_create.wait()
            else:
                # Every read before the load is let finish is the load's own; the create's
                # list re-read comes after the create, which is held past that point, so
                # it is never held here.
                listing_reads.append(route.request.url)
                await finish_the_load.wait()
            await route.continue_()

        await page.route("**/api/rooms", hold_the_rooms_route)

        await page.goto(fresh_ui_server, wait_until="commit")
        new_control = page.locator("[data-testid='new-conversation-button']")
        await new_control.wait_for(timeout=15000)
        # The rail saying which name it has adopted is the agent list applied; the header's
        # refresh control is disabled while `fetchAllMetadata` runs and re-enabled in the
        # same commit that sets `metadataListed`. Both together say the metadata half of
        # the load is done, so what is still outstanding is the held listing read alone.
        await page.locator(ADOPTED_NAME_ROW).first.wait_for(timeout=15000)
        await page.locator("button[title='Refresh runtime state']:enabled").wait_for(timeout=15000)
        # The hold has to hold something. This case once held a read the page had stopped
        # making (#1374 removed `/api/sessions` from the load), and then the load finished
        # before the click and the window it is about never opened.
        assert listing_reads, "the page never asked for its conversation list, so no hold"
        assert not creates, (
            "the head started a conversation before its load had finished, so this case "
            "never reaches the window it is about"
        )

        await new_control.click()
        await _next_frame(page)
        assert len(creates) == 1, f"New did not start a conversation: {creates}"

        # The load finishes while that create is still owed, and the page is given frames to
        # act on it. This is the moment the head used to start a second conversation.
        finish_the_load.set()
        await _next_frame(page)
        await _next_frame(page)
        finish_the_create.set()

        await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)
        await _next_frame(page)

        assert len(creates) == 1, (
            f"New started {len(creates)} conversations, so the head raced the user's own "
            f"click: {creates}"
        )
        # On screen, not only in the request record: one row in the rail, not the user's
        # conversation with an empty `New conversation` sitting beside it.
        rows = page.locator("[data-testid='sessions-list'] button[data-testid^='conversation-']")
        assert await rows.count() == 1, (
            f"the rail holds {await rows.count()} conversations after one click on New"
        )

        assert not errors, f"the first run raised in the browser: {errors}"
        await browser.close()


@pytest.mark.asyncio
async def test_a_reply_builds_up_while_it_is_being_written(streaming_ui_server: str) -> None:
    """The live bubble accumulates what the room streams, rather than replacing it.

    It replaced it. The head keyed the bubble on `seq`, which the room sends only on the
    landed row, so every delta looked like the start of a new turn and the user watched
    single words flicker past until the whole reply landed. No browser case observed the
    bubble at all -- each waited for the landed row, which renders correctly either way.

    So the bubble's text is recorded on every DOM change while the reply streams, and at
    least one recording must hold more than the first word.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _create_room(page, streaming_ui_server, "Streaming", ["scout"])
        await page.goto(streaming_ui_server, wait_until="commit")
        await page.wait_for_selector(f"[data-testid='conversation-{room_id}']", timeout=15000)
        await page.click(f"[data-testid='conversation-{room_id}']")
        await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)

        await page.evaluate(
            """() => {
                window.__liveBubble = [];
                new MutationObserver(() => {
                    const bubble = document.querySelector("[data-testid='live-turn'] p");
                    if (bubble) window.__liveBubble.push(bubble.textContent);
                }).observe(document.body, { subtree: true, childList: true, characterData: true });
            }"""
        )

        await page.fill("[data-testid='room-composer']", "check the index on users")
        await page.click("[data-testid='send-message']")
        await page.wait_for_selector("[data-testid='row-4']", timeout=20000)

        seen: list[str] = await page.evaluate("() => window.__liveBubble")
        assert seen, "the live bubble never rendered, so nothing about streaming was observed"
        assert any(text.startswith("Mock response") for text in seen), (
            f"the live bubble never held more than one delta at a time: {seen}"
        )
        assert not errors, f"the conversation raised in the browser: {errors}"
        await browser.close()


async def _settled_on_silence(page: Page, base_url: str, room_id: str) -> dict[str, Any]:
    """Wait until the room has recorded the silence that ends its cascade, and return it.

    Nothing is published for a silence (§3.6), so a head learns of it only by reading the
    room. The browser assertions below are therefore made on a conversation opened after
    the Core has finished, rather than raced against a write the stream never announces.
    """
    state: dict[str, Any] = {}
    for _ in range(100):
        response = await page.request.get(f"{base_url}/api/rooms/{room_id}")
        assert response.ok, await response.text()
        state = await response.json()
        decision: dict[str, Any] = state.get("last_decision") or {}
        if decision.get("verdict") == "silence":
            return state
        await page.wait_for_timeout(100)
    raise AssertionError(f"room {room_id} never recorded a silence: {state}")


async def _reopen(page: Page, base_url: str, room_id: str) -> None:
    """Load the head afresh and open `room_id`, so what renders is the settled record."""
    await page.goto(base_url, wait_until="commit")
    await page.wait_for_selector(f"[data-testid='conversation-{room_id}']", timeout=15000)
    await page.click(f"[data-testid='conversation-{room_id}']")
    await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)


@pytest.mark.asyncio
async def test_a_one_agent_conversation_that_was_answered_does_not_say_nobody_answered(
    ui_test_server: str,
) -> None:
    """#920: the agent replies, and the conversation must not then claim nobody did.

    After the reply the chain has nobody left to give the floor to and records a silence.
    The head read that silence alone and told a new user, under their first answer, that
    no one had answered their message and no one was going to.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _create_room(page, ui_test_server, "Answered", ["scout"])
        await _reopen(page, ui_test_server, room_id)
        await page.fill("[data-testid='room-composer']", "check the index on users")
        await page.click("[data-testid='send-message']")
        # Rows 1 and 2 are the two joins, 3 the message, 4 the reply.
        await page.wait_for_selector("[data-testid='row-4']", timeout=20000)

        state = await _settled_on_silence(page, ui_test_server, room_id)
        assert state["transcript"][-1]["sender_id"] == "scout", (
            f"the premise is a silence recorded after scout's reply: {state['transcript']}"
        )

        await _reopen(page, ui_test_server, room_id)
        await page.wait_for_selector("[data-testid='row-4']", timeout=15000)
        assert await page.locator("[data-testid='silence-notice']").count() == 0, (
            "the conversation said nobody answered, under the reply it had just rendered"
        )
        assert not errors, f"the conversation raised in the browser: {errors}"
        await browser.close()


@pytest.mark.asyncio
async def test_a_message_no_agent_answers_says_so(ui_test_server: str) -> None:
    """The other side of #920: a message that really went unanswered still says so.

    Two agents, no address and no designated responder: every rule in the chain abstains,
    so the room records a silence without anyone taking the floor.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _create_room(page, ui_test_server, "Unanswered", ["scout", "critic"])
        await _reopen(page, ui_test_server, room_id)
        await page.fill("[data-testid='room-composer']", "anyone around?")
        await page.click("[data-testid='send-message']")

        state = await _settled_on_silence(page, ui_test_server, room_id)
        assert state["transcript"][-1]["sender_id"] == "user", (
            f"the premise is a message no agent answered: {state['transcript']}"
        )

        await _reopen(page, ui_test_server, room_id)
        # Rows 1-3 are the joins, 4 the message.
        await page.wait_for_selector("[data-testid='row-4']", timeout=15000)
        notice = page.locator("[data-testid='silence-notice']")
        await notice.wait_for(timeout=10000)
        assert "No one answered your last message" in await notice.inner_text()
        assert not errors, f"the conversation raised in the browser: {errors}"
        await browser.close()


def _stamp_updated_at(storage_dir: Path, room_id: str, updated_at: datetime) -> None:
    """Back-date a stored conversation. `RoomStore.save` always stamps *now*."""
    path = storage_dir / "rooms" / f"{room_id}.json"
    assert path.exists(), f"the conversation is not stored where this test looks: {path}"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["updated_at"] = updated_at.isoformat()
    path.write_text(json.dumps(raw), encoding="utf-8")


async def _rail_order(page: Page) -> list[str]:
    """The room ids in the rail, top to bottom, as the page lays them out."""
    ids: list[str] = await page.eval_on_selector_all(
        "[data-testid='sessions-list'] button[data-testid^='conversation-']",
        "rows => rows.map(r => r.dataset.testid.slice('conversation-'.length))",
    )
    return ids


@pytest.mark.asyncio
async def test_the_rail_lists_the_most_recently_active_conversation_first(
    fresh_ui_server: str, tmp_path: Path
) -> None:
    """#1053: most recent first, saying when, and moving when a conversation is used.

    The rooms are back-dated so that recency is the *reverse* of the id order. The listing
    used to be the id order -- a sort of a random hex string -- so dropping the Core's sort
    puts the oldest conversation first here and fails, rather than passing by the luck of
    whatever ids were minted. Then the oldest one is used, and has to rise to the top
    without a reload: a rail that only re-sorts when the page loads is sorted once.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        created = [
            await _create_room(page, fresh_ui_server, title, ["scout"])
            for title in ("Index tuning", "Release notes", "Cache strategy")
        ]
        oldest, middle, newest = sorted(created)
        now = datetime.now(UTC)
        _stamp_updated_at(tmp_path, oldest, now - timedelta(days=10))
        _stamp_updated_at(tmp_path, middle, now - timedelta(days=3))
        _stamp_updated_at(tmp_path, newest, now - timedelta(minutes=5, seconds=5))

        await page.goto(fresh_ui_server, wait_until="commit")
        await page.wait_for_selector(f"[data-testid='conversation-{oldest}']", timeout=15000)

        assert await _rail_order(page) == [newest, middle, oldest]
        first = await page.locator(f"[data-testid='conversation-{newest}']").inner_text()
        assert "5m" in first, f"the top row does not say when it was active: {first!r}"
        assert "3d" in await page.locator(f"[data-testid='conversation-{middle}']").inner_text()

        await page.click(f"[data-testid='conversation-{oldest}']")
        await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)
        await page.fill("[data-testid='room-composer']", "is the index still slow?")
        await page.click("[data-testid='send-message']")

        # No reload: the rail re-reads the listing when the message lands.
        await page.wait_for_function(
            """(id) => {
                const top = document.querySelector(
                    "[data-testid='sessions-list'] button[data-testid^='conversation-']");
                return top && top.dataset.testid === 'conversation-' + id;
            }""",
            arg=oldest,
            timeout=15000,
        )
        assert await _rail_order(page) == [oldest, newest, middle]
        used = await page.locator(f"[data-testid='conversation-{oldest}']").inner_text()
        assert "now" in used, f"the conversation just used does not say so: {used!r}"

        assert not errors, f"the rail raised in the browser: {errors}"
        await browser.close()
