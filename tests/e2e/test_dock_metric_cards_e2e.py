"""The Evaluations scorecard at its container's width, not the window's (#1300 follow-up, #1358).

The four metric cards are laid out by `grid-cols-1 sm:grid-cols-2 lg:grid-cols-4`, and those
are *viewport* breakpoints. They were first measured in the dock, a panel whose width the
reader drags: on a wide screen a narrow dock matched `lg:` and took four cards across it, and
the readings sat over their labels with the ends cut off. The rule ui-authoring §3 states --
layout answers to its container, not to the viewport -- is carried by `.dock-scope`.

Since #1358 the scorecard is not a dock surface. It is the Evals half of Settings' Diagnostics
section, shown only while developer mode is on, and that section carries `.dock-scope` so the
same rule holds there: the Settings dialog is at most `max-w-2xl` wide, so on a 1600px window,
where every viewport breakpoint says four across, the container says two; on a phone-width
window it says one. This is the panel with seeded numbers on every machine.

Measured in a browser because nothing else can measure it: jsdom resolves no stylesheet and
evaluates no container query, so a component test can assert a className and never find out
what it drew.
"""

from __future__ import annotations

from typing import Any

import pytest
from playwright.async_api import Page, async_playwright

from tests.e2e.conftest import TWO_FRAMES, open_diagnostics

pytestmark = pytest.mark.e2e

#: Where each metric card sits and whether it holds more than it can show. `scrollWidth`
#: exceeding `clientWidth` is content past the card's own edge -- which is what the reader
#: saw as a cut-off word -- and it reports that whether or not the card clips it.
_CARDS = """() => {
  const cards = [...document.querySelectorAll("[data-testid='eval-metric-card']")];
  return {
    count: cards.length,
    columns: new Set(cards.map((c) => Math.round(c.getBoundingClientRect().left))).size,
    container: Math.round(
      document.querySelector("[data-testid='settings-diagnostics']").getBoundingClientRect().width,
    ),
    overflowing: cards
      .filter((c) => c.scrollWidth > c.clientWidth + 1)
      .map((c) => `${c.textContent.slice(0, 40)}: ${c.scrollWidth} in ${c.clientWidth}`),
  };
}"""


async def _scorecard(page: Page, url: str) -> dict[str, Any]:
    await page.goto(url, wait_until="commit")
    await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
    # Evals is in Settings' Diagnostics section, which only developer mode adds (#1358).
    section = await open_diagnostics(page)
    await section.locator("[data-testid='eval-metric-card']").first.wait_for(timeout=10000)
    await page.evaluate(TWO_FRAMES)
    state: dict[str, Any] = await page.evaluate(_CARDS)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("window_width", "expected_columns"),
    [
        # A wide window: every viewport breakpoint says four, the dialog's width says two.
        (1600, 2),
        # A phone-width window: the container is under 480px, so one card per row. Without
        # this case the rule could be "two columns forever" and nothing would notice.
        (420, 1),
    ],
)
async def test_the_scorecard_answers_to_the_section_it_is_in_not_the_window(
    ui_test_server: str, window_width: int, expected_columns: int
) -> None:
    """Settings -> Diagnostics, with developer mode on, at two window widths.

    `Killed by:` frontend/src/components/settings/DiagnosticsSection.tsx ::
    `<div className="dock-scope space-y-6">` becoming `<div className="space-y-6">` -- the
    container query then has no container, the window's `lg:grid-cols-4` answers at 1600px and
    the wide case reads four columns.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": window_width, "height": 900})
            state = await _scorecard(page, ui_test_server)

            assert state["count"] == 4, f"metric cards: {state}"
            assert state["columns"] == expected_columns, (
                f"a {state['container']}px Diagnostics section at a {window_width}px window "
                f"laid the four cards in {state['columns']} column(s), expected {expected_columns}"
            )
            assert state["overflowing"] == [], (
                f"a {state['container']}px section holds card content past its edge: "
                f"{state['overflowing']}"
            )
        finally:
            await browser.close()
