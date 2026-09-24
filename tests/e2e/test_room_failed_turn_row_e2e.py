"""The failed-turn row on the surface that survives (#1208): Retry only where it can succeed.

`test_failed_turn_row_e2e.py` guards #969's rule on `PlaygroundTab`, which the retirement
deletes. Its cases are written against that component tree -- `chat-input`, `send-button`,
`messages-container`, `turn-chip`, `turn-remedy` -- so not one of them runs against
`RoomConversation`, and the rule has never been held on the surface being kept by anything
above the component level. This module ports it across **before** the deletion, so that what
the retirement removes is a second implementation and not the coverage.

The rule itself: a turn the Core refused on a spent token or cost budget is refused again on
every retry -- the ledger only grows -- so its row says why in a plain sentence, with the remedy, and
offers no Retry. A turn that failed for any other reason keeps Retry, because running it again
can succeed. Both are decided from the structured `refusal` the Core states on the row, never
from the error's wording.

**Nothing is mocked at the HTTP layer, and that is the point of the port.** The playground
cases fixed two of the three answers with `page.route`, which decides the very field the rule
reads: a scripted `refusal` proves the head branches on a value the test wrote, not that the
Core ever produces it. Here the refusal is the real one -- a Core budget of zero refuses the
turn, `BaseAgent.execute_turn` returns `stop_reason="budget_exceeded"`, and
`RoomOrchestrator._take_turn` maps that through `turn_refusal` onto the row. The control is a
real provider failure for the same reason. Route, orchestrator, store, transcript and render
are all the shipped ones; only the model and the ceiling are the test's.

The two fixtures live here rather than in `conftest.py` because each is one module's input
(R12), following `test_persona_editor_e2e.py`. The playground's shared `spent_budget_ui_server`
could not have served this file and is deleted with that suite: it configured `sess_default`,
and a room seat holds its own session -- `sess_room__<room_id>__<agent_id>`, derived by
`room/service.py::participant_session_id` from a room id minted at `POST /api/rooms`, so it
cannot be named before the server is running. A zero *default* ceiling is the honest way to
reach a session whose name is not knowable in advance: it refuses the seat's first turn for
the seat's own session, through the same check.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, ViewportSize, async_playwright

from tests.e2e.conftest import mock_llm, running_ui
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse, StreamChunk

pytestmark = pytest.mark.e2e

#: Wide enough that the rail is beside the conversation rather than over it (#1062): below
#: 600px a first run starts with the rail closed and the row that opens a conversation is not
#: on screen at all.
VIEWPORT: ViewportSize = {"width": 1280, "height": 900}

#: What `refusalRemedy` says about `budget_exceeded`, in the head's own words. Asserted in
#: full rather than by substring: the remedy is the whole reason the row withholds Retry, and
#: a row that showed the refusal and no way forward would pass a laxer check.
REMEDY = "Trying again would be refused too. Start a new conversation to continue."

#: What the row says about a spent token ceiling, in the head's words (#1408). The Core's own
#: refusal text ("Session token limit exceeded: 0/0") stays on the row's `error` field for the
#: log; the row line states the structured `refusal` instead, so a reader is not shown the
#: budget manager's counters.
CEILING = "it has used all the tokens this conversation allows"
#: The Core's refusal text, which must not reach the row line (#1408).
RAW_CEILING = "limit exceeded"

PROMPT = "Please answer this"
SECOND_TIME = "Second time lucky."


class _FailsOnceThenAnswers(MockLLMConnector):
    """A provider that drops the first call it is given and serves every later one.

    The control needs a turn that genuinely failed and that a retry could genuinely repair --
    the playground's case scripted `refusal: null` beside a step-budget message, which tests
    the head against its own fixture. A dropped connection is a failure `BaseAgent` reports
    with no `stop_reason` the refusal mapping recognises, so the row it lands on carries
    `error` and `refusal: null`: exactly the shape the rule must keep Retry for, arrived at
    through the Core rather than asserted at it.

    Failing *once* rather than always is what makes the retry observable. Against a provider
    that always fails, a Retry that ran and a Retry that did nothing both leave a failed row
    on screen; with the second call answered, the reply is the evidence the click reached the
    Core.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.drops = 0

    def _drop_the_first_call(self) -> None:
        if self.drops == 0:
            self.drops += 1
            raise RuntimeError("the provider dropped the connection (scripted)")

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self._drop_the_first_call()
        return await super().generate(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self._drop_the_first_call()
        async for chunk in super().stream(request):
            yield chunk


@pytest.fixture
def zero_budget_ui_server(tmp_path: Path) -> Iterator[str]:
    """A UI server on which every session -- a room seat's included -- has no tokens at all.

    The ceiling is the manager's *default* rather than one configured session, because the
    session a room seat spends on is derived from the room id and the room does not exist
    until the test creates one over HTTP. `TokenBudgetManager._get_or_create_budget` hands
    that default to whatever session first asks, so `check_budget` refuses the seat's opening
    turn on `0/0` before a token is spent.
    """
    budget = TokenBudgetManager(default_max_tokens=0)
    with running_ui(storage_dir=tmp_path, llm=mock_llm(), budget_tracker=budget) as url:
        yield url


@pytest.fixture
def dropped_first_turn_ui_server(tmp_path: Path) -> Iterator[str]:
    """A UI server whose provider drops the first turn asked of it and answers the next."""
    llm = _FailsOnceThenAnswers(default_model="mock-gpt-4o", default_response=SECOND_TIME)
    with running_ui(storage_dir=tmp_path, llm=llm) as url:
        yield url


async def _open_a_conversation(page: Page, base_url: str) -> str:
    """Seat a 1:1 through the API and open it, as any other client would."""
    response = await page.request.post(
        f"{base_url}/api/rooms",
        data={"title": "Refusals", "agent_ids": ["scout"]},
    )
    assert response.ok, await response.text()
    body: dict[str, Any] = await response.json()
    room_id = str(body["room_id"])
    await _reopen(page, base_url, room_id)
    return room_id


async def _reopen(page: Page, base_url: str, room_id: str) -> None:
    """Load the head afresh and open `room_id` from the rail."""
    await page.goto(base_url, wait_until="commit")
    await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
    await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)


async def _say(page: Page, words: str) -> None:
    await page.fill("[data-testid='room-composer']", words)
    await page.click("[data-testid='send-message']")


async def _assert_the_refusal_is_on_the_row(page: Page) -> None:
    """The refused turn states its error and its remedy on its own row, and offers no Retry.

    Rows 1 and 2 are the two joins, 3 is the prompt and 4 is the turn that answered it, so the
    failure is row 4 -- a `seq`, not a position in the rendered list.

    `cascade-error` and `send-error` are checked as absent rather than left unmentioned. They
    are the two other places this failure could surface, and either of them would be wrong in
    a way a reader feels: the conversation announcing that it stopped, or the composer
    refusing to have sent a message that was recorded and answered. A refusal belongs to the
    turn that met it.
    """
    row = page.locator("[data-testid='row-4']")
    await row.locator("[data-testid='row-error-4']").wait_for(timeout=20000)
    stated = await row.locator("[data-testid='row-error-4']").inner_text()
    assert CEILING in stated, stated
    assert RAW_CEILING not in stated, f"the row printed the Core's raw refusal: {stated!r}"
    # Counted before it is read. A missing remedy read straight through `inner_text` fails as
    # a 30-second Playwright timeout naming a selector, which reports "the page was slow"
    # where what happened is that the row said nothing about what to do next.
    assert await row.locator("[data-testid='row-remedy-4']").count() == 1, (
        "the refused turn showed its error with no remedy beside it"
    )
    assert (await row.locator("[data-testid='row-remedy-4']").inner_text()).strip() == REMEDY
    assert await row.locator("[data-testid='retry-turn']").count() == 0, (
        "the row offered a Retry for a refusal every retry meets again"
    )
    assert await page.locator("[data-testid='cascade-error']").count() == 0, (
        "the refusal was announced as the conversation stopping instead of on its turn"
    )
    assert await page.locator("[data-testid='send-error']").count() == 0, (
        "the composer reported a message as unsent that was recorded and answered"
    )


@pytest.mark.asyncio
async def test_the_core_s_budget_refusal_offers_no_retry_live_or_after_a_reload(
    zero_budget_ui_server: str,
) -> None:
    """End to end through the real Core: its refusal, live and as stored, withholds Retry.

    The reload half checks the same row on a cold mount: a page that holds nothing from the
    send, rebuilding the conversation from scratch. Measured, it is **not** a second
    serialisation of the refusal -- `App.tsx` answers the turn's `final` event by re-reading
    `GET /api/rooms/{id}`, so the live row already comes from the stored transcript, and a
    mutation that dropped `refusal` from what `RoomStore` writes killed the live assertion
    before the reload was reached. What the reload adds is therefore narrower than it looks
    and is stated rather than assumed: the remedy is a property of the stored turn, derived
    again on mount, and not state the running page kept beside the turn it had just watched
    fail. That distinction is invisible until a reload asks for it.

    Nothing re-sends on its own either: the transcript holds one failed turn, not a second
    written by a head that quietly retried what the Core had already refused.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            room_id = await _open_a_conversation(page, zero_budget_ui_server)

            await _say(page, PROMPT)
            await _assert_the_refusal_is_on_the_row(page)
            assert await page.locator("[data-testid='row-5']").count() == 0, (
                "a second turn was run against a budget that had already refused the first"
            )

            await _reopen(page, zero_budget_ui_server, room_id)
            await page.locator("[data-testid='row-3']").get_by_text(PROMPT).wait_for(timeout=15000)
            await _assert_the_refusal_is_on_the_row(page)
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_turn_that_failed_without_a_refusal_keeps_a_retry_that_runs_it_again(
    dropped_first_turn_ui_server: str,
) -> None:
    """The control, and the half that makes the rule a distinction rather than a policy.

    Without it, "no Retry on a refused turn" is satisfied by a surface that offers Retry
    nowhere, which would pass the case above while removing the remedy for every ordinary
    provider failure. So this asserts the opposite of each of that test's three claims on a
    row that differs from it in one field: the row carries no remedy, it does carry Retry, and
    the click runs the turn again -- observed as the reply that lands, not as the request the
    button fired, since a request the Core refused would fire identically.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            await _open_a_conversation(page, dropped_first_turn_ui_server)

            await _say(page, PROMPT)
            row = page.locator("[data-testid='row-4']")
            await row.locator("[data-testid='row-error-4']").wait_for(timeout=20000)
            assert await row.locator("[data-testid='row-remedy-4']").count() == 0, (
                "a failure a retry could repair was presented as one it could not"
            )
            assert await row.locator("[data-testid='retry-turn']").count() == 1

            await row.locator("[data-testid='retry-turn']").click()

            landed = page.locator("[data-testid='row-5']")
            await landed.wait_for(timeout=20000)
            assert SECOND_TIME in await landed.inner_text(), (
                "Retry appended a row without running the turn again"
            )
            assert await landed.locator("[data-testid='row-error-5']").count() == 0
        finally:
            await browser.close()


#: What a reader scanning the transcript can see about a row without reading it: the mark down
#: its edge, and the box it sits in. Read as computed style, so it is what the browser drew and
#: not what a class name promised.
_HOW_THE_ROW_LOOKS = """(seq) => {
    const el = document.querySelector(`[data-testid='row-body-${seq}']`);
    if (!el) return null;
    const s = getComputedStyle(el);
    return {
        state: el.dataset.state ?? null,
        borderLeftWidth: s.borderLeftWidth,
        borderLeftStyle: s.borderLeftStyle,
        borderLeftColor: s.borderLeftColor,
        animation: s.animationName,
        boxShadow: s.boxShadow,
    };
}"""


@pytest.mark.asyncio
async def test_a_failed_row_and_a_good_one_differ_by_more_than_their_words(
    dropped_first_turn_ui_server: str,
) -> None:
    """#1228: the failed turn is marked, not merely described.

    `PlaygroundTab` marked a failed turn structurally -- `turn-chip` with `data-state` -- and
    #1234 retired it. What came across was the prose, which is what the two cases above hold.
    A reader scanning a long conversation for the turn that failed cannot scan prose, and in a
    1:1 `bubbleClass` draws no bubble, so before this the failed row and the good one were the
    same box with different sentences in it.

    Both rows are taken from **one transcript in one run**, which is what makes this a
    difference rather than two measurements: the same provider drops turn 4 and answers turn
    5, so the rows share their conversation, their shape, their width and their stylesheet,
    and the only thing that differs is that one of them failed.

    The assertion is what the browser drew -- computed style -- not the class list or the
    `data-state` attribute, either of which can be present while the rule is invisible. #1241
    is the case for reading it this way: a layout assertion there passed under its own
    mutation because it checked a proxy that an ancestor satisfied either way.

    The restraint half is asserted too, and on the failed row rather than in review: no
    animation and no shadow, so the marker cannot quietly become a pulse or a glow
    (#1061 names the same rule for the rail).

    Mutation-checked by hand rather than as a kill declaration, because the lethality ratchet
    runs a browser test against the **committed bundle** without rebuilding it, so a
    declaration naming `frontend/src` reads as escaped there (measured; see
    `tests/e2e/test_rail_responsive_e2e.py`). Rebuilt with `vite build`, each of these fails
    this case and no other in this file:
    `Killed by:` frontend/src/components/rooms/RoomConversation.tsx ::
    `message.error && 'border-l-2 border-rose-800 pl-3',` becoming
    `message.error && 'pl-3',`
    `Killed by:` frontend/src/components/rooms/RoomConversation.tsx ::
    `data-state={message.error ? 'failed' : undefined}` becoming `data-state={undefined}`
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=VIEWPORT)
            await _open_a_conversation(page, dropped_first_turn_ui_server)

            await _say(page, PROMPT)
            row = page.locator("[data-testid='row-4']")
            await row.locator("[data-testid='row-error-4']").wait_for(timeout=20000)
            await row.locator("[data-testid='retry-turn']").click()
            await page.locator("[data-testid='row-5']").wait_for(timeout=20000)

            failed = await page.evaluate(_HOW_THE_ROW_LOOKS, 4)
            good = await page.evaluate(_HOW_THE_ROW_LOOKS, 5)
            assert failed is not None and good is not None

            assert failed["state"] == "failed", f"the failed row is unmarked: {failed}"
            assert good["state"] is None, f"a turn that succeeded is marked failed: {good}"

            drawn = failed["borderLeftStyle"] != "none" and failed["borderLeftWidth"] != "0px"
            assert drawn, f"the mark on the failed row is not drawn: {failed}"
            assert (failed["borderLeftWidth"], failed["borderLeftColor"]) != (
                good["borderLeftWidth"],
                good["borderLeftColor"],
            ), f"the two rows are drawn alike: failed {failed}, good {good}"

            assert failed["animation"] == "none", f"the marker moves: {failed}"
            assert failed["boxShadow"] == "none", f"the marker glows: {failed}"
        finally:
            await browser.close()
