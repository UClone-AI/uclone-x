"""End-to-end Playwright regression tests for the UClone-X developer UI dashboard."""

from __future__ import annotations

import re
from typing import Final

import pytest
from playwright.async_api import Page, async_playwright

import uclone_x
from tests.e2e.conftest import dock_locator, turn_on_developer_mode

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_dashboard_initial_load_and_focus(ui_test_server: str) -> None:
    """Verify that the dashboard loads cleanly and the input is auto-focused."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page: Page = await browser.new_page()

        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        await page.goto(ui_test_server, wait_until="commit")
        await page.wait_for_selector("input, textarea", timeout=10000)

        title = await page.title()
        assert "UClone-X" in title

        # Verify header logo
        app_logo = page.locator("[data-testid='header-logo']")
        assert await app_logo.is_visible()
        assert await app_logo.get_attribute("src") == "/uclone_logo_circle.png"
        assert await app_logo.get_attribute("alt") == "UClone Logo"

        # Verify header title and version badge with accessibility tooltips
        app_title = page.locator("[data-testid='header-app-title']")
        assert await app_title.is_visible()
        assert await app_title.get_attribute("title") == "UClone-X Agentic Runtime Platform"

        # Asserted against the package version rather than a literal. Both this
        # test and the badge used to carry `v0.1.0`, so a release that bumped the
        # package left the dashboard showing the old number and the gate stayed
        # green because the two stale copies agreed with each other.
        version_badge = page.locator("[data-testid='header-version-badge']")
        assert await version_badge.is_visible()
        assert f"v{uclone_x.__version__}" in (await version_badge.inner_text())
        assert await version_badge.get_attribute("title") == (
            f"Current Release Version: v{uclone_x.__version__}"
        )

        # Verify initial focus on input field (Issue #114, #339)
        input_focused = await page.evaluate(
            "document.activeElement === document.querySelector('input, textarea')"
        )
        assert input_focused is True
        assert len(errors) == 0

        await browser.close()


@pytest.mark.asyncio
async def test_topology_tab_renders_without_crash(ui_test_server: str) -> None:
    """Verify that switching to the Topology tab renders DAG nodes without throwing TypeError (Issue #116)."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page: Page = await browser.new_page()

        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        await page.goto(ui_test_server, wait_until="commit")

        # The DAG is a developer instrument: it is in the dock's developer drawer, which exists
        # only in developer mode, and developer mode is off by default (owner ruling
        # 2026-09-22). So: switch it on, then open the dock, which starts closed.
        await page.wait_for_selector("[data-testid='toggle-dock']", timeout=10000)
        await turn_on_developer_mode(page)
        await page.locator("[data-testid='toggle-dock']").click()
        dock = await dock_locator(page)
        await dock.wait_for(state="visible", timeout=10000)
        await dock.locator("[data-testid='dev-tab-topology']").click()

        # Header must be visible and no uncaught exceptions thrown. Waiting on the heading
        # itself rather than on a duration: the fixed 500ms this replaced was both slower
        # than the render on an idle machine and shorter than it under load (R7).
        heading = dock.locator("h2")
        await heading.first.wait_for(state="visible", timeout=10000)
        h2_text = await heading.first.inner_text()
        assert "How this conversation ran" in h2_text
        assert len(errors) == 0

        # The graph is the conversation on screen (#1355): either its seats, or the sentence
        # saying why there is nothing to draw (P6). Never a blank panel. The panel says
        # "Reading this conversation…" until its fetch lands, so wait for either settled
        # state before reading; a single read raced that fetch under gate load.
        settled = dock.locator("[data-testid='topology-reason']").or_(
            dock.get_by_text(re.compile(r" seated · "))
        )
        await settled.first.wait_for(state="visible", timeout=15000)
        dock_text = await dock.inner_text()
        assert " seated · " in dock_text or (
            await dock.locator("[data-testid='topology-reason']").count() == 1
        ), dock_text[:400]

        await browser.close()


# What each panel must actually render.
#
# This replaces `assert len(main.inner_text()) > 100`. That threshold does catch a panel
# that renders nothing at all, but it cannot tell a working panel from one showing the
# *wrong* content — the previous tab's panel left on screen, an error boundary's message,
# or an empty-state placeholder where data belongs. All three clear 100 characters
# comfortably. Verified: replacing the ledger panel with the topology panel leaves
# `main` at 480 characters, so the threshold passes while the tab is plainly broken.
#
# The strings are taken from what each panel actually renders, not from the tab's own
# name — several panels never print their own label.
#: The conversation surface itself. It was the welcome message's first line
#: ("UClone-X is ready.") until #1208: that copy was `createWelcomeMessage`'s, rendered as
#: a row by the retired `PlaygroundTab`, and the room transcript has no such row. Selecting
#: the surface rather than a string is what these two uses were always asking anyway --
#: that the conversation is present, and that opening a dock surface does not displace it.
_CONVERSATION_SURFACE: Final[str] = "[data-testid='room-conversation']"

_DOCK_CONTENT_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    # The DAG of the conversation on screen (#1355); the heading is the only fixed text.
    "topology": ("How this conversation ran",),
    "ledger": ("EventBus SSE Stream Ledger", "Last Heartbeat"),
    "ontology": ("Total Concepts", "Asserted (Axiomatic)"),
    # ACP and Skills were dock surfaces here until #1358 moved them into Settings; their
    # markers are asserted there by
    # `test_workspace_layout_e2e.py::test_settings_holds_skills_always_and_diagnostics_only_in_developer_mode`.
    # Resource absorbed the retired Budget surface, whose own marker was its token quota.
    "resource": ("Resource & Budget Summary", "Token Quota"),
}

# Text that means the panel failed rather than rendered. A length check counts these
# characters as evidence of success.
_RENDER_FAILURE_MARKERS: Final[tuple[str, ...]] = (
    "Something went wrong",
    "Traceback",
    "TypeError:",
)


@pytest.mark.asyncio
async def test_tab_navigation_all_tabs(ui_test_server: str) -> None:
    """Each internals surface renders its own panel, and none shows another's content.

    The conversation is checked separately and without a click, because it is no longer a
    tab: it is the centre column and is present whatever the dock is showing. That is the
    change this test had to absorb — previously every marker was asserted against `main`,
    which worked only while the seven panels took turns occupying it.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page: Page = await browser.new_page()

        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        await page.goto(ui_test_server, wait_until="commit")

        # The conversation is always present, with no tab to click.
        main = page.locator("main")
        await main.wait_for(state="visible", timeout=10000)
        await page.wait_for_selector(_CONVERSATION_SURFACE, timeout=20000)
        assert await main.locator(_CONVERSATION_SURFACE).count() == 1, (
            f"the conversation did not render:\n{(await main.inner_text())[:400]}"
        )

        # Every surface below but Resource is in the developer drawer (owner ruling 2026-09-22).
        await turn_on_developer_mode(page)
        await page.locator("[data-testid='toggle-dock']").click()
        dock = await dock_locator(page)
        await dock.wait_for(state="visible", timeout=10000)

        for surface, markers in _DOCK_CONTENT_MARKERS.items():
            testid = f"tab-{surface}" if surface == "resource" else f"dev-tab-{surface}"
            btn = dock.locator(f"[data-testid='{testid}']")
            assert await btn.count() == 1, f"expected exactly one {testid} control"
            await btn.click()
            # Wait for the panel's primary marker to be visible in the dock before asserting
            # to prevent flaky race conditions under load (#680).
            await dock.get_by_text(markers[0]).first.wait_for(state="visible", timeout=10000)
            dock_text = await dock.inner_text()

            for marker in markers:
                assert marker in dock_text, (
                    f"surface '{surface}' did not render its own panel; "
                    f"expected {marker!r} in:\n{dock_text[:400]}"
                )
            for failure_marker in _RENDER_FAILURE_MARKERS:
                assert failure_marker not in dock_text, (
                    f"surface '{surface}' rendered a failure state: {failure_marker}"
                )

            # No other surface's panel is left on screen. This is the assertion a length
            # threshold cannot express at all.
            for other, other_markers in _DOCK_CONTENT_MARKERS.items():
                if other == surface:
                    continue
                leaked = [m for m in other_markers if m in dock_text]
                assert not leaked, f"surface '{surface}' also shows '{other}' content: {leaked}"

            # And the conversation is still there, which is the point of the dock.
            assert await main.locator(_CONVERSATION_SURFACE).count() == 1, (
                f"opening '{surface}' displaced the conversation"
            )

        assert len(errors) == 0
        await browser.close()


# Two cases retired with `PlaygroundTab` (#1208), neither replaced:
#
#   * `test_welcome_claims_only_what_the_product_does` read `_WELCOME_CLAIMS` out of
#     `messages-container`. The welcome is still built (`createWelcomeMessage` in
#     `App.tsx`) and still feeds the dock, but nothing renders it as a conversation row
#     any more, so the copy discipline #1015/#1018 established has no browser surface to
#     be checked on. A follow-up should re-home the welcome or drop it; this PR does
#     neither, because #1208 is a deletion and inventing an empty-state welcome for the
#     room is a design decision the redesign has not made.
#
#   * `test_fr13_persona_badge_rendered_when_provided_and_absent_when_none` clicked
#     `chat-input`/`send-button` and read `persona-badge`. That testid existed only in
#     `PlaygroundTab`. The room states who answered through `served-by-{seq}` and
#     `header-attribution`, which `RoomConversation.test.tsx` and
#     `tests/e2e/test_room_chat_e2e.py` pin, but a *persona* badge specifically has no
#     room equivalent -- FR-13.4's own claim, that the persona is visible in the turn
#     without opening a panel, is not currently true of the surviving surface.
