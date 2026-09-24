"""End-to-end tests for the two routes the saturation banner rests on (#1230).

`GET /api/rooms/{id}/context` and `POST /api/rooms/{id}/compact` reached the head only
through fixtures the head's own tests wrote. At `f55a1020` a reviewer read both sides and
found the payloads agreed field for field, so nothing was broken -- and that is exactly the
shape this file exists to change. A fixture that agrees with the server by coincidence keeps
agreeing after the server changes, which is the defect class
the unified-conversations design note's Phase 4 **[Rev 3]** records: every rule
was held by component tests over fixtures the tests wrote themselves, the fixture disagreed
with the API about a payload's shape, and the first successful reply blanked the screen with
the whole gate green.

So these two routes are pinned where a rename cannot be agreed with: the browser renders the
server's own answer, and every field the head reads is read back off the screen.

* **`context.is_saturated`** decides whether the banner exists at all, and
  **`context.saturation_threshold`** is printed in it as a number.
* **`context.seats[].participant_id`** is what the banner names, resolved through the room's
  own roster -- a field arriving under another name names nobody.
* **`compaction.results`** is counted (*"1 clone"*) and **`results[].saved_tokens`** is summed
  into the freed figure. Renamed, the sum is `NaN` and the sentence says so.
* The banner coming **down** afterwards is the second `GET /context`, read against a Core
  that really did shorten.

The banner's absence is asserted **before** it is waited for and again after the shortening,
so a green run cannot mean the surface was never on screen: the two negative assertions are
bracketed by a positive one that has to hold in between.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from playwright.async_api import Page, async_playwright

pytestmark = pytest.mark.e2e

#: The Core saturates a seat at this many turns held in its active context
#: (`DEFAULT_MAX_CONVERSATION_TURNS`). Spelled out rather than imported, so that what the
#: browser checks is the number the route **sent**: imported, a threshold the route stopped
#: sending would still agree with the constant it was read from.
SATURATION_TURNS = 20

#: Padding on every message. Compaction reports what it freed, and a conversation of
#: twenty two-word turns is already shorter than the ledger that would replace it -- the
#: Core answers `saved_tokens: 0`, which is a true figure that no rename could change.
#: With this padding the run measured 3240 tokens before and 2281 after, so the figure on
#: screen is one only a working field can produce.
_PADDING = "padding words that make the transcript worth shortening " * 4


async def _create_room(page: Page, base_url: str, title: str, agents: list[str]) -> str:
    """Seat a conversation through the API, as any other client would."""
    response = await page.request.post(
        f"{base_url}/api/rooms",
        data={"title": title, "agent_ids": agents},
    )
    assert response.ok, await response.text()
    body: dict[str, Any] = await response.json()
    return str(body["room_id"])


async def _say(page: Page, words: str, lands_as: int) -> None:
    """Send one message and return once the reply to it has landed."""
    await page.fill("[data-testid='room-composer']", words)
    await page.click("[data-testid='send-message']")
    await page.wait_for_selector(f"[data-testid='row-{lands_as}']", timeout=30000)


@pytest.mark.asyncio
async def test_a_saturated_seat_raises_the_banner_and_shortening_reports_what_the_core_freed(
    fresh_ui_server: str,
) -> None:
    """The banner and the figure under it are the server's own answer, not a fixture's.

    Killed by: src/uclone_x/ui/rooms.py :: "saved_tokens": outcome.saved_tokens,
    Becomes: "freed_tokens": outcome.saved_tokens,

    Killed by: src/uclone_x/ui/rooms.py :: "saturation_threshold": SATURATION_TURNS_THRESHOLD,
    Becomes: "turn_limit": SATURATION_TURNS_THRESHOLD,

    Both mutations are the one the card asks about -- a server-side rename of a field the
    head reads -- one per route. They are **Python** edits, so a browser case sees them
    without the bundle being rebuilt, which is why these declarations are written in the
    parsing form that the mutation-kill-declaration fitness check actually runs
    rather than in the backticked prose form. That form is reserved for mutations landing
    in `frontend/src`, which no pytest run can reach because `tests/e2e/` drives the
    committed bundle in `src/uclone_x/ui_static` and never rebuilds it.

    Under the first, `done.results[i].saved_tokens` is `undefined`, the head's `reduce`
    yields `NaN`, and the notice reads *"freeing about NaN tokens"* -- which the assertion
    below rejects because it asks for digits. Under the second, `context.saturation_threshold`
    is `undefined` and the banner reads *"reached the -turn limit on context"*.

    Since #1256 removed the reopen, this case also covers the retry -- but **probabilistically**,
    so the third mutation is recorded in prose rather than as a scored declaration. Written
    out it would be `Killed by:` `frontend/src/lib/rooms.ts` ::
    `export const CONTEXT_READ_ATTEMPTS = 3;` becoming `= 1`, which is the pre-#1256 code
    exactly: issue the read, refuse a stale answer, stop. That target is under `frontend/src`,
    which no pytest run can reach -- `tests/e2e/` drives the committed bundle in
    `src/uclone_x/ui_static` and never rebuilds it -- so the declaration is written in the
    backticked non-parsing form and the mutation was replayed by hand, rebuilding the bundle.
    Measured at `2d3f7207` + this change, 12 runs each: **12/12 pass with the retry, 7/12 pass
    without it** (5 failures, every one `Locator.wait_for` timing out on
    `[data-testid='saturation-banner']`). A 5-in-12 kill is evidence about the flow and is
    **not** a declaration the ratchet could score. The deterministic case for the retry is
    `frontend/src/App.test.tsx`, which holds the read in flight and forces the interleaving.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        room_id = await _create_room(page, fresh_ui_server, "Long haul", ["scout"])
        await page.goto(fresh_ui_server, wait_until="commit")
        await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
        await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)

        banner = page.locator("[data-testid='saturation-banner']")
        assert await banner.count() == 0, "the banner is up before a single turn has been taken"

        # Rows 1 and 2 are the two joins, so message `i` lands its reply on row `2i + 2`.
        # Saturation counts *assistant* messages held in the seat's Core session, so it takes
        # one landed reply per turn and nothing shorter will do.
        for turn in range(1, SATURATION_TURNS + 1):
            await _say(page, f"question {turn}: {_PADDING}", lands_as=2 * turn + 2)

        # Waited for on the turn that caused it, with no reopen in between (#1256). This
        # used to reload the page and click back in, because the head's read after the
        # *last* turn was not reliably kept: `readRoomContext` captured the room generation
        # before it called and dropped its own answer if the generation moved meanwhile,
        # and nothing after the last turn issued another read. Measured on this flow at 2
        # of 8 runs, with the route answering `is_saturated: true` while the screen showed
        # no banner. The reopen hid it by resetting `roomContext` to `null` and forcing a
        # read -- a test arranging the exact condition that concealed the defect. #1256 made
        # the discarded read re-ask instead, so the banner now arrives on the turn that
        # saturated the seat and the reopen is gone. (#1286 then removed the four-turn
        # sampling cadence entirely, which does not change what this line waits for.)
        #
        # Which means this line is load-bearing twice over: it still reads the route's own
        # `is_saturated` and `saturation_threshold` off the screen, and it is now also the
        # browser-level evidence that the retry runs. The forced-interleaving case for the
        # retry itself is `frontend/src/App.test.tsx`, which can hold the read in flight;
        # this one exercises the real timing end to end.
        await banner.wait_for(timeout=30000)
        said = await banner.inner_text()
        assert f"scout reached the {SATURATION_TURNS}-turn limit on context" in said, said

        await page.click("[data-testid='compact-room']")

        notice = page.locator("[data-testid='room-notice']")
        await notice.wait_for(timeout=30000)
        reported = await notice.inner_text()
        freed = re.search(r"Shortened 1 clone, freeing about (\d+) tokens\.", reported)
        assert freed is not None, (
            f"the shortening did not report one seat and a token figure: {reported!r}"
        )
        assert int(freed.group(1)) > 0, (
            f"the Core freed nothing, so the figure proves nothing: {reported!r}"
        )

        # The read the head issues after a shortening, against a Core that really did cut.
        await banner.wait_for(state="detached", timeout=30000)
        assert not errors, f"the conversation raised in the browser: {errors}"
        await browser.close()
