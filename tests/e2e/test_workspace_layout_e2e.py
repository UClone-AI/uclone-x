"""End-to-end tests for the three-region workspace: rail, conversation, workspace dock."""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.async_api import Locator, Page, async_playwright

from tests.e2e.conftest import (
    SURFACE_OVERFLOW,
    TWO_FRAMES,
    dock_locator,
    open_diagnostics,
    set_the_rail,
    turn_on_developer_mode,
)

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_3_panel_workspace_layout_and_toggles(ui_test_server: str, tmp_path: Path) -> None:
    """Verify the three regions, their toggles, and the dock's surface switching.

    The screenshot used to be written to a hardcoded path under a Builder tool's home
    directory, naming one Antigravity session's UUID. That is Builder metadata in the
    runtime test suite (`AGENTS.md`, "Rule of Separation"), and it made the test write
    outside the repository on every run. `tmp_path` is per-run and cleaned up.
    """
    scratch_dir = tmp_path

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Open in desktop viewport
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})

        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        await page.goto(ui_test_server, wait_until="networkidle")
        await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)

        # 1. Verify Left Sidebar elements.
        #
        # Selected by testid rather than by its heading. The rail said "Sessions" until
        # the conversation surface landed and it became "Chats" -- a label change that is
        # the whole point of FR-13.9 and that broke this assertion, which is the same
        # coupling #864 and #877 already paid for once. What this test is about is that
        # the rail is present and toggles, so it selects the rail.
        assert await page.locator("[data-testid='chat-sidebar']").first.is_visible()

        # The gauge is not here any more (#1059). It reports agent *steps* against the step
        # ceiling, which is an instrument, and ui-authoring §2 keeps instruments one
        # deliberate action away from U0's default screen rather than on it. It is in the
        # dock's Resource surface, opened below; what the rail must show is that it shows
        # none of it.
        #
        # The dock's own testids, not the ones #1272 deleted: an absence assertion on a
        # testid no build renders passes on every page and guards nothing.
        for testid in (
            "room-reply-budget-readout",
            "room-seat-turns",
            "room-step-count-absent",
            "room-token-total-absent",
        ):
            assert await page.locator(f"[data-testid='{testid}']").count() == 0
        # And never under the old wrong name, wherever it ends up living: the meter was
        # labelled "Turn Budget" and fed the conversation's turn count, which filled a red
        # bar to 51/50 during an ordinary conversation.
        assert await page.locator("text=Turn Budget").count() == 0

        # The gauge's own reading after a send went with the playground composer (#1208):
        # it was driven by `POST /api/chat`'s `turn_budget_max` / `turns_remaining`, and a
        # room's send returns a `RoomMessage` carrying neither. The readings themselves are
        # pinned closer in -- `ResourceSummary.test.tsx` renders the room's reply-cascade
        # meter, the per-seat turn rows and the two absence sentences from props, and
        # `test_ui_server.py` pins what `GET /api/rooms/{id}/context` reports on the wire.
        # What this test still owns is the three regions, their toggles and the dock's
        # surfaces, which is what its name says.

        # 2. Toggle Left Sidebar
        toggle_sidebar_btn = page.locator("[data-testid='toggle-sidebar']")
        sidebar = page.locator("[data-testid='chat-sidebar']")
        await toggle_sidebar_btn.click()
        await sidebar.wait_for(state="hidden", timeout=5000)

        # 3. Toggle the workspace dock.
        #
        # `dock_locator` counts the retired dock *eagerly*, when it is awaited, so it is
        # called after the click rather than before it. The dock mounts on toggle — probed
        # live against the bundle at 833a09e, where both docks were present: closed, both
        # selectors count 0; open, both count 1. A retired-dock count taken while the dock
        # is still closed therefore reads 0 on a doubled page and passes vacuously, and the
        # test then dies later and elsewhere on a `wait_for` timeout. Called after the
        # click, it fails on its own assertion, which is the one that names the defect.
        toggle_dock_btn = page.locator("[data-testid='toggle-dock']")
        await toggle_dock_btn.click()
        dock = await dock_locator(page)
        await dock.wait_for(state="visible", timeout=5000)
        assert await dock.is_visible()

        # 3b. And the readings the rail gave up are here, one click further in (#1059). This
        # is the other half of the absence asserted above: without it, a build that deleted
        # the step budget outright would pass this test.
        #
        # What the surface reports changed in #1272: these are the open room's own readings,
        # and the two quantities a room does not have are stated as sentences rather than as
        # numbers borrowed from the retired single-agent store (P6). The dock is opened on
        # U0's default screen with a conversation already open, so the room block renders.
        await dock.locator("[data-testid='tab-resource']").click()
        await dock.locator("[data-testid='room-resource-readings']").wait_for(
            state="visible", timeout=5000
        )
        await dock.locator("[data-testid='room-step-count-absent']").wait_for(
            state="visible", timeout=5000
        )
        assert await dock.locator("[data-testid='room-token-total-absent']").count() == 1
        # "Step Budget" named a per-run step count the room does not report; the reply
        # cascade meter beside it is a different quantity and must not wear that label.
        assert await dock.locator("text=Step Budget").count() == 0

        # 4. Switch dock surfaces. EventBus and DAG are developer instruments: the drawer that
        # holds them exists only in developer mode, which is off by default (owner ruling
        # 2026-09-22), so the test switches it on the way a user does and the dock stays open.
        await turn_on_developer_mode(page)
        await dock.locator("[data-testid='dev-tab-ledger']").click()
        event_stream = dock.locator("text=EventBus SSE Stream Ledger")
        await event_stream.wait_for(state="visible", timeout=5000)
        assert await event_stream.is_visible()

        await dock.locator("[data-testid='dev-tab-topology']").click()
        # The DAG reads the conversation on screen (#1355); its heading says so.
        swarm_nodes = dock.locator("text=How this conversation ran")
        await swarm_nodes.wait_for(state="visible", timeout=5000)
        assert await swarm_nodes.is_visible()

        # 5. Progressive disclosure from a tool chip to the dock went with #1208, and is
        # not replaced. The chip it clicked was `PlaygroundTab`'s inline tool card, and the
        # room transcript has no tool rendering to click: `RoomMessage` excludes tool calls
        # and results deliberately (`room/models.py`), so a seat's executions never reach
        # the transcript at all. The data still reaches the wire -- `test_ui_tool_rendering.py`
        # is what pins that -- and no surface draws it. That gap is a follow-up issue.

        # 6. Verify zero window scroll invariant
        scroll_y = await page.evaluate("window.scrollY")
        assert scroll_y == 0, f"Window scrolled by {scroll_y}px; should be 0"

        # 7. Take screenshot of the three-region workspace
        # Re-open sidebar for full 3-panel screenshot
        await toggle_sidebar_btn.click()
        await sidebar.wait_for(state="visible", timeout=5000)
        screenshot_path = scratch_dir / "workspace_3panel_desktop.png"
        await page.screenshot(path=str(screenshot_path))

        assert len(errors) == 0, f"Page errors encountered: {errors}"
        await browser.close()


@pytest.mark.asyncio
async def test_reopening_the_dock_keeps_the_active_surface(
    ui_test_server: str,
) -> None:
    """#1055: reopening the dock returns to the surface last picked.

    It used to force the surface back to Docs & Artifacts on every open, silently
    discarding whatever the header's own Workspace toggle had left selected. The
    conversation toolbar's separate "Internals" button -- a second control for the same
    one dock, under a second name a few pixels from the header's -- is gone outright, and
    so is the floating opener that hovered over the reader's own text: nothing on the page
    should carry that label, and the header's toggle is the one way in.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": 1400, "height": 900})
            await page.goto(ui_test_server, wait_until="networkidle")
            await page.wait_for_selector("textarea", timeout=10000)

            assert await page.get_by_text("Internals", exact=True).count() == 0

            # Knowledge Graph is a developer surface, in the drawer developer mode adds.
            await turn_on_developer_mode(page)
            await page.click("[data-testid='toggle-dock']")
            dock = await dock_locator(page)
            await dock.wait_for(state="visible", timeout=5000)
            await dock.locator("[data-testid='dev-tab-knowledge_graph']").click()
            await dock.locator("[data-testid='knowledge-graph-viewer']").wait_for(
                state="visible", timeout=5000
            )

            await dock.locator("[data-testid='dock-close']").click()
            await dock.wait_for(state="hidden", timeout=5000)

            assert await page.locator("[data-testid='floating-dock-btn']").count() == 0

            await page.click("[data-testid='toggle-dock']")

            dock = await dock_locator(page)
            await dock.wait_for(state="visible", timeout=5000)
            await dock.locator("[data-testid='knowledge-graph-viewer']").wait_for(
                state="visible", timeout=5000
            )
            assert await dock.locator("[data-testid='doc-viewer']").count() == 0
        finally:
            await browser.close()


async def _open_the_resource_surface(page: Page) -> Locator:
    """Open the dock and select Resource, where the conversation's readings live since #1059.

    Two clicks, because the dock is shut on U0's default screen and the readings are
    instruments (ui-authoring §2). This is the "how a user reaches it" of the move, written
    as the steps rather than as a claim.
    """
    await page.locator("[data-testid='toggle-dock']").click()
    dock = await dock_locator(page)
    await dock.wait_for(state="visible", timeout=5000)
    await dock.locator("[data-testid='tab-resource']").click()
    return dock


@pytest.mark.asyncio
async def test_the_resource_surface_states_the_room_s_missing_totals_in_words(
    ui_test_server: str,
) -> None:
    """A room's absent token total is said, not filled from the retired store (#1272, P6).

    **This test replaces the `#939` reload arm that stood here.** That arm loaded a
    transcript row carrying `token_count_source: "provider"` and asserted the dock's
    `token-total-readout` showed `1,000` with no `estimated` label beside it. The figure it
    read was `GET /api/session/history` for whichever clone the rail had selected -- the
    retired single-agent store -- while the centre column showed a room, so the number on
    screen described a transcript the user could not open. #1272 removed it rather than
    re-source it, because a room has no token total to re-source it from:
    `RoomMessage.usage` is declared and nothing fills it on the orchestrated path, so a
    room's turns carry no token count to total, and no HTTP route serves one.

    So the end-to-end claim is now the absence itself, which is the part a user sees. The
    `token_count_source` mapping #939 was about still exists in `App.tsx` and is still
    pinned on the wire by `test_ui_server.py`; what no longer exists is a surface that
    renders its result, and that is stated in the report rather than hidden here.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})

        await page.goto(ui_test_server, wait_until="networkidle")
        # Nowhere on the default screen, then one dock away: the move (#1059) still holds.
        assert await page.locator("[data-testid='room-token-total-absent']").count() == 0
        dock = await _open_the_resource_surface(page)

        absent = dock.locator("[data-testid='room-token-total-absent']")
        await absent.wait_for(state="visible", timeout=10000)
        said = await absent.inner_text()
        # The sentence has to do two things: deny the figure, and name the Token Quota
        # beside it as a different quantity so the reader does not read that one as this.
        assert "No token total" in said, said
        assert "different quantity" in said, said

        # And no number anywhere in the conversation block claiming to be that total. The
        # session ledger's own figure is 0 on a fresh server, so this checks the stronger
        # thing: the retired readout is gone rather than renamed.
        assert await page.locator("[data-testid='token-total-readout']").count() == 0
        assert await page.locator("[data-testid='token-total-estimated']").count() == 0

        await browser.close()


#: Every surface the dock offers in developer mode: the primary row's, then the drawer's.
_DOCK_TABS = (
    "artifacts",
    "remembers",
    "knowledge_graph",
    "activity",
    "resource",
    "topology",
    "ledger",
    "ontology",
)

#: The dock against the window it is drawn in. Geometry is reported, and so is what a click at
#: each control's centre actually reaches -- a control pushed past the window's edge overlaps
#: nothing measurable and is simply not there to be clicked, which is the form this defect took.
#: Called with the testid of the tab that selects the surface: `tab-` or, in the drawer, `dev-tab-`.
_DOCK_OFF_SCREEN = """(tabTestId) => {
    const dock = document.querySelector("[data-testid='artifacts-dock']");
    if (!dock) return ['artifacts-dock: not rendered'];
    const r = dock.getBoundingClientRect();
    const problems = [];
    const box = `x ${r.left.toFixed(1)}-${r.right.toFixed(1)}`;
    if (r.width < 1 || r.height < 1) problems.push(`the dock has no size, ${box}`);
    if (r.left < -0.5 || r.right > window.innerWidth + 0.5) {
        problems.push(`the dock ${box} is outside the window 0-${window.innerWidth}`);
    }
    if (document.documentElement.scrollWidth > window.innerWidth + 0.5) {
        problems.push(`the page scrolls sideways: `
            + `${document.documentElement.scrollWidth} > ${window.innerWidth}`);
    }
    const surface = dock.querySelector("[data-testid='dock-surface']");
    if (!surface) {
        problems.push('dock-surface: not rendered');
    } else if (surface.scrollWidth > surface.clientWidth + 1
               && !['auto', 'scroll'].includes(getComputedStyle(surface).overflowX)) {
        problems.push(`the ${tabTestId} surface holds ${surface.scrollWidth}px of content in `
            + `${surface.clientWidth}px and cannot be scrolled to reach it`);
    }
    for (const id of [tabTestId, 'dock-close']) {
        const el = document.querySelector(`[data-testid='${id}']`);
        if (!el) { problems.push(`${id}: not rendered`); continue; }
        const b = el.getBoundingClientRect();
        const at = `x ${b.left.toFixed(1)}-${b.right.toFixed(1)}`;
        if (b.left < -0.5 || b.right > window.innerWidth + 0.5) {
            problems.push(`${id}: ${at} is outside the window 0-${window.innerWidth}`);
        }
        const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
        if (!hit || !el.contains(hit)) {
            const what = hit ? hit.dataset.testid || hit.tagName : 'nothing';
            problems.push(`${id}: a click at its centre reaches ${what}`);
        }
    }
    return problems;
}"""

#: The surfaces in the developer drawer, which exists only in developer mode (owner ruling
#: 2026-09-22). Their tabs carry `dev-tab-`, the primary row's `tab-`. ACP, Skills and Evals
#: were drawer surfaces until #1358 moved them into Settings.
_DEVELOPER_TABS = frozenset({"knowledge_graph", "topology", "ledger", "ontology"})


def _tab_testid(tab: str) -> str:
    """The testid of the control that selects `tab`, in the drawer or in the primary row."""
    return f"dev-tab-{tab}" if tab in _DEVELOPER_TABS else f"tab-{tab}"


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [320, 375, 400])
async def test_the_open_dock_fits_a_narrow_window_on_every_surface(
    ui_test_server: str, width: int
) -> None:
    """#1019: at 320, 375 and 400px the open dock is on screen and can be operated.

    The dock took its stored 520px whatever the window was. The workspace row cannot scroll
    sideways, so the excess simply left the window: measured at 400px on the Budget surface the
    dock ran `x -120..400`, which is the reading #1019 was filed on, and on the first surface
    it ran `x 240..760` instead -- the row overflows whichever way the surfaces' own widths push
    it, so a test that pinned one number would pass while the other still failed. What is pinned
    here instead is the property: the dock lies inside the window, the page does not scroll
    sideways to hide it, and the surface's own tab and the close button each take a click.

    Every surface is visited because the width is not the only thing that decides the box -- at
    320px with the rail open, all eleven had a control that `elementFromPoint` could not reach.

    With the rail closed only. This case used to run both rail states, because the rail was 240px
    *of the row* at these widths and was what made the row impossible. Since #1062 the rail at
    these widths is drawn over the workspace and holds none of the row, so the dock's box is the
    same whichever state it is in; what differs is that an open rail covers the dock's left
    240px, which is what an open overlay does and is not the dock's to fix. The dock's close
    control, on its right edge, staying on top of an open rail is pinned in
    `test_rail_responsive_e2e.py`.

    The surface is checked for *reachability* rather than for fitting, and the difference is the
    whole of what #1019 asks ("fits within the viewport ... **or** scrolls to reveal all of its
    content"). When #1019 landed, several panels headed themselves with a non-wrapping row that
    was wider than the dock (Skills 546px and Evaluations 575px in the default 520px dock at
    1280px); that row was reachable only because the surface's `overflow-x` computes to `auto`,
    and a change that made it `hidden` would put content permanently out of reach, which is the
    failure this clause exists to catch. #1029 made those rows wrap, so they now also *fit* --
    which `test_no_dock_surface_holds_a_control_past_its_edge` pins -- but this case keeps the
    weaker, reachability reading: it is the one #1019 asks for.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": width, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda err: errors.append(str(err)))

            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            await set_the_rail(page, "closed")
            await turn_on_developer_mode(page)

            await page.click("[data-testid='toggle-dock']")
            dock = await dock_locator(page)
            await dock.wait_for(state="visible", timeout=5000)

            for tab in _DOCK_TABS:
                await page.click(f"[data-testid='{_tab_testid(tab)}']")
                await page.evaluate(TWO_FRAMES)
                problems = await page.evaluate(_DOCK_OFF_SCREEN, _tab_testid(tab))
                assert problems == [], f"{width}px, {tab}: {problems}"

            assert errors == [], f"{width}px: page errors {errors}"
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("rail", ["open", "closed"])
@pytest.mark.parametrize(("width", "stored"), [(1280, None), (400, None), (1280, 360), (320, None)])
async def test_no_dock_surface_holds_a_control_past_its_edge(
    ui_test_server: str, width: int, stored: int | None, rail: str
) -> None:
    """#1029: at every dock and window width, every surface's controls lie inside the dock.

    The panels headed themselves with single-line rows of search, selects and buttons that
    stacked only below the *viewport's* `sm:` breakpoint. The dock is 360-900px whatever the
    viewport is, so on the bundle this was filed against the default 520px dock at 1280px held
    Skills' row at 546px and Evaluations' at 575px in a 519px surface, and at 400px Topology's
    ran 470px and Skills' 480px in 399px. The surface scrolled sideways and the last control of
    each row -- a filter, "Poll Topology", the view switcher -- sat past its edge with nothing
    saying it was there. The rows now wrap; this pins the property, not the numbers: nothing in
    any surface overflows it, and no control is outside it.

    1280 and 400 are the widths the issue names. The other two are where the rest of the rows
    overflowed, which the issue did not list: the narrowest dock the handle allows (360px at
    1280, where Topology ran 470px, Ontology 467px and Ledger 406px) and a 320px window (where
    Activity's filters ran 326px). Both rail states, because the rail is what overlays the dock
    in a narrow window and only changes the conversation's share of a wide one.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": width, "height": 900})
            if stored is not None:
                await page.add_init_script(
                    f"try {{ localStorage.setItem('uclone-x.dock.width', '{stored}') }} "
                    "catch (e) {}"
                )
            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            await set_the_rail(page, "closed")
            await turn_on_developer_mode(page)
            await _open_the_dock(page)

            found: dict[str, list[str]] = {}
            for tab in _DOCK_TABS:
                # Below 600px an open rail overlays the dock's tab strip (#1062) and takes the
                # click, so the tab is chosen with the rail closed and the rail set after.
                await set_the_rail(page, "closed")
                await page.click(f"[data-testid='{_tab_testid(tab)}']")
                await set_the_rail(page, rail)
                await page.evaluate(TWO_FRAMES)
                problems = await page.evaluate(SURFACE_OVERFLOW)
                if problems:
                    found[tab] = problems
            assert found == {}, f"{width}px, dock {stored or 'default'}, rail {rail}: {found}"
        finally:
            await browser.close()


#: Settings' Diagnostics section, where the ACP report and the Evals scorecard live (#1358).
_DIAGNOSTICS = "[data-testid='settings-diagnostics']"

#: The column count of every grid in the surface that sets its columns with a `grid-cols-`
#: class, which is every grid `.dock-scope`'s container query answers for.
_SURFACE_GRID_COLUMNS = """() => {
    const surface = document.querySelector("[data-testid='dock-surface']");
    if (!surface) return null;
    return [...surface.querySelectorAll("[class*='grid-cols-']")]
        .filter((g) => getComputedStyle(g).display === 'grid'
                       && g.getBoundingClientRect().width > 0)
        .map((g) => getComputedStyle(g).gridTemplateColumns.split(' ').length);
}"""

#: The surfaces that render a column grid with the test server's data.
_GRID_TABS = ("resource", "topology", "ontology")


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [360, 480, 481, 520])
async def test_a_dock_narrower_than_its_grids_lays_them_out_in_one_column(
    ui_test_server: str, stored: int
) -> None:
    """#1029: the dock's grids answer to the dock's width, not the window's.

    The grids choose their columns at the viewport's `sm:`/`md:`/`lg:` breakpoints, so at
    1280px a 360px dock drew three and four columns of 72-96px each. `.dock-scope` was meant
    to stop that, and its rule sat in `index.css` with no element carrying the class: #1055
    retired the dock it had been on. `ArtifactsDock` now carries it, and at a content width of
    480px or less every grid is one column -- the same threshold `.column-scope` uses (#1015).

    The dock's default 520px keeps its columns: that is the other half of the decision, and the
    one that says this did not collapse the dock the issue says should not collapse. 480 and
    481 are the boundary as the query sees it: the aside's 1px border comes off its content box,
    so a 481px dock is 480px of content and is single-column, and a 482px one would not be.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": 1280, "height": 900})
            await page.add_init_script(
                f"try {{ localStorage.setItem('uclone-x.dock.width', '{stored}') }} catch (e) {{}}"
            )
            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            await set_the_rail(page, "closed")
            await turn_on_developer_mode(page)
            dock = await _open_the_dock(page)
            box = await dock.bounding_box()
            assert box is not None and round(box["width"]) == stored, box

            columns: dict[str, list[int]] = {}
            for tab in _GRID_TABS:
                await page.click(f"[data-testid='{_tab_testid(tab)}']")
                await page.evaluate(TWO_FRAMES)
                counts = await page.evaluate(_SURFACE_GRID_COLUMNS)
                assert counts, f"{stored}px, {tab}: no column grid rendered to measure"
                columns[tab] = counts
            if stored <= 481:
                assert all(c == 1 for cs in columns.values() for c in cs), (
                    f"a {stored}px dock draws more than one column: {columns}"
                )
            else:
                assert all(max(cs) > 1 for cs in columns.values()), (
                    f"the {stored}px dock lost its columns: {columns}"
                )
        finally:
            await browser.close()


#: The dock and the conversation as boxes. Whether the dock sits *in* the workspace row or
#: *over* it is not readable from the dock alone -- at 1280px the two render the same box --
#: so the column is measured with it: they meet at an edge only while they share the row.
_DOCK_AND_COLUMN = """() => {
    const dock = document.querySelector("[data-testid='artifacts-dock']");
    const column = document.querySelector("[data-testid='room-conversation']");
    if (!dock || !column) return null;
    const d = dock.getBoundingClientRect();
    const c = column.getBoundingClientRect();
    return {
        width: +d.width.toFixed(1),
        left: +d.left.toFixed(1),
        right: +d.right.toFixed(1),
        columnRight: +c.right.toFixed(1),
        windowWidth: window.innerWidth,
    };
}"""


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [768, 1024, 1280])
async def test_a_window_wider_than_the_dock_still_seats_it_beside_the_conversation(
    ui_test_server: str, width: int
) -> None:
    """#1019's control: at 768px and wider nothing about the dock changes.

    The narrow-window fix takes the dock out of the workspace row, and a fix that did that at
    every width would satisfy the case above while quietly rewriting the layout every existing
    user has. So this pins the other side of the same decision: the dock keeps its stored 520px,
    it still ends at the window's right edge, and the conversation column ends exactly where the
    dock begins -- which is true while they share the row and false the moment the dock is drawn
    over it.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": width, "height": 900})
            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)

            await page.click("[data-testid='toggle-dock']")
            dock = await dock_locator(page)
            await dock.wait_for(state="visible", timeout=5000)
            await page.evaluate(TWO_FRAMES)

            boxes = await page.evaluate(_DOCK_AND_COLUMN)
            assert boxes is not None, f"{width}px: the dock or the conversation is not rendered"
            assert boxes["width"] == 520, f"{width}px: the dock is {boxes['width']}px, not 520"
            assert boxes["right"] == boxes["windowWidth"], f"{width}px: {boxes}"
            assert abs(boxes["columnRight"] - boxes["left"]) <= 1, (
                f"{width}px: the conversation ends at {boxes['columnRight']} and the dock "
                f"begins at {boxes['left']}; the dock is drawn over the row, not in it"
            )
        finally:
            await browser.close()


#: The dock's box, how it is positioned, and the width the browser has stored -- in one
#: reading, because the two halves of #1030's F1 are only a defect together. A dock that
#: survives a drag says nothing about what was written under it, and a stored width that
#: survives says nothing about the box on screen.
_DOCK_AND_STORAGE = """() => {
    const dock = document.querySelector("[data-testid='artifacts-dock']");
    if (!dock) return null;
    const r = dock.getBoundingClientRect();
    const rail = document.querySelector("[data-testid='chat-sidebar']");
    return {
        left: +r.left.toFixed(1),
        right: +r.right.toFixed(1),
        width: +r.width.toFixed(1),
        position: getComputedStyle(dock).position,
        stored: window.localStorage.getItem('uclone-x.dock.width'),
        windowWidth: window.innerWidth,
        railRight: rail ? +rail.getBoundingClientRect().right.toFixed(1) : null,
        insideWindow: r.left >= -0.5 && r.right <= window.innerWidth + 0.5,
    };
}"""


#: The narrowest window with the rail in the workspace row (#1062; `RAIL_OVERLAY_BELOW_PX` in
#: `frontend/src/lib/rail.ts`). Below it the rail is drawn over the row and holds none of it.
_RAIL_IN_THE_ROW = 600


async def _open_the_dock(page: Page) -> Locator:
    """Open the dock and return it, once the browser has recomputed the layout.

    Two frames rather than a sleep (R7), and `dock_locator` rather than a bare selector, so
    a page still mounting the retired dock is refused here as it is everywhere else.
    """
    await page.click("[data-testid='toggle-dock']")
    dock = await dock_locator(page)
    await dock.wait_for(state="visible", timeout=5000)
    await page.evaluate(TWO_FRAMES)
    return dock


@pytest.mark.asyncio
async def test_a_drag_at_a_narrow_window_cannot_return_the_dock_to_a_row_that_cannot_seat_it(
    ui_test_server: str,
) -> None:
    """#1030: a drag must not undo #1019's fix, and must not persist having undone it.

    The overlay decision reads the dock's **stored** width, and `MIN_WIDTH` is 360. At a 400px
    window the handle sits at `x 1..5`, so one drag rightward set the width to 360, `400 > 360`
    turned the overlay off, and the dock went back into the workspace row that still cannot
    seat it: measured `x 240..600`, with `dock-close` off the window again. That is #1019's own
    shape, reached by a drag. The drag's mouse-up then wrote 360 to `localStorage`, so a reload
    found it there and the dock never overlaid again.

    The reload is what makes this a defect rather than a transient, so it is part of the case
    rather than a second test: the state the drag reaches is the state the next session starts
    in.

    **At 600px since #1062.** The defect needs the rail in the row, and below 600px the rail is
    now drawn over the workspace and holds none of it -- at 400px a 360px dock beside no rail
    does fit, so the case would lose its premise. 600px is the narrowest window with the rail
    in the row, and there the shape is the same: a window-only rule reads `600 > 360` and seats
    the dock beside a rail that leaves it exactly 360, which the dock's `<=` rule overlays.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": _RAIL_IN_THE_ROW, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda err: errors.append(str(err)))

            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            await set_the_rail(page, "open")
            await _open_the_dock(page)

            before = await page.evaluate(_DOCK_AND_STORAGE)
            assert before is not None, "the dock is not rendered"
            assert before["position"] == "absolute", (
                f"the dock is not overlaid before the drag, so this test would pass without "
                f"exercising anything: {before}"
            )

            handle = page.locator("[data-testid='dock-resize-handle']")
            box = await handle.bounding_box()
            assert box is not None, "the resize handle has no box, so nothing was dragged"
            middle = box["y"] + box["height"] / 2
            await page.mouse.move(box["x"] + box["width"] / 2, middle)
            await page.mouse.down()
            # A width of 200 is below `MIN_WIDTH`, so the drag asks for the narrowest width the
            # dock allows: 360, which is narrower than the window and is exactly what took the
            # overlay off.
            await page.mouse.move(_RAIL_IN_THE_ROW - 200, middle, steps=10)
            await page.mouse.up()
            await page.evaluate(TWO_FRAMES)

            dragged = await page.evaluate(_DOCK_AND_STORAGE)
            assert await page.evaluate(_DOCK_OFF_SCREEN, "tab-artifacts") == [], dragged
            # The gesture is honoured -- what the drag asked for is what is stored. It is the
            # layout that declines to seat 360px beside a 240px rail in a 600px window, not the
            # control; an earlier revision refused the gesture instead and made the width
            # unrecoverable from inside the surface at every window up to MAX_WIDTH.
            assert dragged["stored"] == "360", dragged

            await page.reload(wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            await _open_the_dock(page)

            reloaded = await page.evaluate(_DOCK_AND_STORAGE)
            assert reloaded is not None, "the dock is not rendered after the reload"
            assert reloaded["railRight"] == 240, (
                f"the rail is not in the row after the reload, so this reading has no premise: "
                f"{reloaded}"
            )
            assert reloaded["position"] == "absolute", (
                f"after a reload the dock is back in the workspace row: {reloaded}"
            )
            assert await page.evaluate(_DOCK_OFF_SCREEN, "tab-artifacts") == [], reloaded

            assert errors == [], f"page errors {errors}"
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "position"),
    [(519, "absolute"), (520, "absolute"), (521, "relative")],
)
async def test_the_overlay_threshold_is_the_docks_own_width(
    ui_test_server: str, width: int, position: str
) -> None:
    """#1030: the window exactly as wide as the dock is overlaid, and one pixel wider is not.

    The dock's stored width is 520 by default, and every width any suite used was comfortably
    on one side or the other -- 320/375/400 and 768/1024/1280 in the browser, 400 and 1280 in
    vitest. So the threshold could be written `<` instead of `<=` and nothing anywhere went
    red, while the window exactly as wide as the dock changed sides. 519 and 521 are the
    controls that keep this a boundary rather than a single reading: an overlay that fired at
    every width would satisfy 519 and 520 on its own.

    The rail is closed throughout -- which since #1062 is also how a first run at these widths
    starts, and an open rail here would be drawn over the row and reserve none of it -- so the
    rail is taken out of the reading rather than measured by it.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": width, "height": 900})
            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            await set_the_rail(page, "closed")
            await _open_the_dock(page)

            state = await page.evaluate(_DOCK_AND_STORAGE)
            assert state is not None, f"{width}px: the dock is not rendered"
            assert state["windowWidth"] == width, (
                f"the browser reports a {state['windowWidth']}px window, not {width}px, so "
                f"this case does not sit where it claims to: {state}"
            )
            assert state["position"] == position, (
                f"{width}px against a 520px dock: the dock is {state['position']}, not "
                f"{position} -- the overlay threshold is not the dock's own width: {state}"
            )
            assert state["right"] <= width + 0.5, f"{width}px: the dock leaves the window: {state}"
        finally:
            await browser.close()


async def _drag_handle_to(page: Page, x: float) -> None:
    """Drag the dock's resize handle from wherever it is to `x`, in real mouse events."""
    box = await page.locator("[data-testid='dock-resize-handle']").bounding_box()
    assert box is not None, "the resize handle has no box, so nothing was dragged"
    middle = box["y"] + box["height"] / 2
    await page.mouse.move(box["x"] + box["width"] / 2, middle)
    await page.mouse.down()
    await page.mouse.move(x, middle, steps=10)
    await page.mouse.up()
    await page.evaluate(TWO_FRAMES)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "rail", "position_after"),
    [
        (400, "closed", "relative"),
        (520, "closed", "relative"),
        (_RAIL_IN_THE_ROW, "open", "absolute"),
        (700, "open", "relative"),
        (768, "open", "relative"),
    ],
)
async def test_a_dock_dragged_over_the_workspace_can_always_be_dragged_back_out_of_it(
    ui_test_server: str, width: int, rail: str, position_after: str
) -> None:
    """#1030 B1: no gesture may leave the dock's width unrecoverable from inside the surface.

    Dragging the handle to the window's left edge asks for the whole window, which overlays the
    dock at any window up to `MAX_WIDTH`. A revision of this fix froze the width whenever the
    dock was overlaid, so that first drag was a one-way door: the handle went inert at that
    window size **permanently** -- through further drags, a reload, and closing and reopening
    the dock -- leaving the dock covering the whole window with no rail and no conversation, and
    the only way out was to enlarge the OS window past the stored width.

    **Two drags, because one drag can never see it.** Every case in both suites performed at
    most one gesture, which is why the freeze and a guard that applied only to the first gesture
    both passed everything. The second drag is the whole point of this test.

    What the second drag does depends on what is left of the row once the rail has taken its
    share, and both answers are pinned here. At 600 the 240px rail leaves exactly 360, which
    does not seat the dock, so it stays drawn over the workspace; at 700 and 768 the narrowest
    width does fit, so it returns to the row and the rail is beside it again. At 400 and 520 the
    rail holds none of the row since #1062 -- it starts closed there, and open it is drawn over
    the row -- so 360 fits and the dock returns to the row. (Until #1062 those two were the
    "stays overlaid" cases, with the rail in the row; 600 took that role.) Either way the dock
    is inside the window and the width the user asked for is the width that is stored.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": width, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda err: errors.append(str(err)))

            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            await set_the_rail(page, rail)
            await _open_the_dock(page)

            # Drag 1 — to the window's left edge: `window - 0`, capped at MAX_WIDTH.
            await _drag_handle_to(page, 0)
            overlaid = await page.evaluate(_DOCK_AND_STORAGE)
            assert overlaid["position"] == "absolute", (
                f"{width}px: dragging to the edge did not overlay the dock, so this case never "
                f"reaches the state it exists to escape from: {overlaid}"
            )
            assert overlaid["insideWindow"], overlaid
            assert overlaid["stored"] == str(min(width, 900)), overlaid

            # Drag 2 — back to MIN_WIDTH. This is the drag the freeze made inert.
            await _drag_handle_to(page, width - 360)
            recovered = await page.evaluate(_DOCK_AND_STORAGE)
            assert recovered["stored"] == "360", (
                f"{width}px: the second drag did not change the stored width "
                f"({recovered['stored']!r}) — the handle is inert, and the width reached by the "
                f"first drag cannot be undone from inside the surface: {recovered}"
            )
            assert recovered["insideWindow"], f"{width}px: the dock left the window: {recovered}"
            assert recovered["position"] == position_after, (
                f"{width}px: after narrowing to 360 the dock is {recovered['position']}; with the "
                f"rail {rail} in a {width}px window it should be {position_after}: {recovered}"
            )
            assert await page.evaluate(_DOCK_OFF_SCREEN, "tab-artifacts") == [], recovered
            if position_after == "relative" and rail == "open":
                assert recovered["railRight"] == 240, (
                    f"{width}px: the dock is back in the row, so the rail must be beside it "
                    f"rather than covered: {recovered}"
                )

            assert errors == [], f"{width}px: page errors {errors}"
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [600, 700, 760])
async def test_the_open_dock_stays_in_the_window_where_the_rail_leaves_it_no_room(
    ui_test_server: str, width: int
) -> None:
    """#1030: between the dock's width and the rail-plus-dock total, the dock still fits.

    At 600 and 700 the default 520px dock and the 240px rail cannot share the row, and until
    this change the dock simply took its 520 and left the window -- `x 240..760` at 700, with
    the close control past the right edge. That reading was identical on `main` and on the
    first revision of this fix, because both decided the overlay from the window and the dock's
    own width alone, which cannot express "the rail is already holding 240 of this".

    This is the band #1028 recorded as its N3 and left open. It is not the rail question
    (#1013/#1018) -- nothing here collapses the rail or changes any state it owns; it is the
    same #1019 defect at the widths the window-only rule could not reach.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": width, "height": 900})
            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            await _open_the_dock(page)

            state = await page.evaluate(_DOCK_AND_STORAGE)
            assert state is not None, f"{width}px: the dock is not rendered"
            assert state["stored"] is None, (
                f"{width}px: this case is about the *default* width, and something has already "
                f"stored {state['stored']!r}: {state}"
            )
            assert state["insideWindow"], f"{width}px: the dock leaves the window: {state}"
            assert await page.evaluate(_DOCK_OFF_SCREEN, "tab-artifacts") == [], state
        finally:
            await browser.close()


#: The dock's close control against the window: where its box is, and what a click at its
#: centre reaches. Narrower than `_DOCK_OFF_SCREEN` on purpose -- it does not read the surface's
#: tab, because below 600px an *open* rail is drawn over the dock's left 240px and takes the
#: click on the tab strip. That is what an open overlay does (#1062) and is settled by
#: `test_the_open_dock_fits_a_narrow_window_on_every_surface`, which runs the rail closed for
#: exactly that reason; the close control sits at the dock's right edge, outside the rail, and
#: is the control #1035 states the invariant over.
_CLOSE_CONTROL_UNREACHABLE = """() => {
    const dock = document.querySelector("[data-testid='artifacts-dock']");
    if (!dock) return ['artifacts-dock: not rendered'];
    const problems = [];
    const r = dock.getBoundingClientRect();
    if (r.left < -0.5 || r.right > window.innerWidth + 0.5) {
        problems.push(`the dock x ${r.left.toFixed(1)}-${r.right.toFixed(1)} is outside the `
            + `window 0-${window.innerWidth}`);
    }
    if (document.documentElement.scrollWidth > window.innerWidth + 0.5) {
        problems.push(`the page scrolls sideways: `
            + `${document.documentElement.scrollWidth} > ${window.innerWidth}`);
    }
    const close = document.querySelector("[data-testid='dock-close']");
    if (!close) return [...problems, 'dock-close: not rendered'];
    const b = close.getBoundingClientRect();
    const at = `x ${b.left.toFixed(1)}-${b.right.toFixed(1)}`;
    if (b.left < -0.5 || b.right > window.innerWidth + 0.5) {
        problems.push(`dock-close: ${at} is outside the window 0-${window.innerWidth}`);
    }
    const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
    if (!hit || !close.contains(hit)) {
        problems.push(`dock-close: ${at}, a click at its centre reaches `
            + `${hit ? hit.dataset.testid || hit.tagName : 'nothing'}`);
    }
    return problems;
}"""


#: The window widths a dragged dock is narrowed through: the ends of the range the workspace is
#: laid out for (320 and 1280), and both sides of the two thresholds inside it -- the rail's own
#: (600, `RAIL_OVERLAY_BELOW_PX`) and the width at which the open rail leaves exactly the default
#: dock (760, where `windowWidth - reservedWidth == width`). A sweep rather than one width per
#: case: the window is narrowed under one already-dragged dock, which is the gesture, and
#: reloading the page between widths would take the live narrowing out of it.
_NARROWED_TO = (320, 400, 520, 599, 600, 601, 700, 759, 760, 761, 900, 1024, 1280)


@pytest.mark.asyncio
@pytest.mark.parametrize("rail", ["open", "closed"])
@pytest.mark.parametrize("dragged", [360, 520, 900])
async def test_no_width_dragged_at_a_wide_window_leaves_the_close_control_off_a_narrow_one(
    ui_test_server: str, dragged: int, rail: str
) -> None:
    """#1035: drag the dock at a wide window, narrow the window, and the close control is there.

    The reported reading is a width stored while the window was wide that the window can no
    longer seat: dragged to its 360px minimum at 900px and narrowed to 400px, the dock was
    measured at `x 240..600` with `dock-close` past the right edge and the state surviving a
    reload. **That reading no longer reproduces.** Two changes since it was filed each remove a
    term of it -- #1042 made the dock subtract the rail's share of the row rather than compare
    the window with its own width, and #1062 stopped the rail taking a share of the row at all
    below 600px -- and this whole sweep passes on `main` at `ce02adda` unchanged.

    What was missing was the guard, not the fix. Every browser case in this suite fixes the
    viewport before the dock mounts and never moves it again, so the narrowing itself -- the
    half of the gesture that makes a stored width outlive the window it was chosen in -- was
    pinned only in `ArtifactsDock.test.tsx`, where jsdom reports a width and lays nothing out.
    A dock whose `style.width` is right and whose box is pushed past the window's edge by the
    row it sits in reads identically there; `x 240..600` *is* that shape, and it is the one this
    measures. The drag is a real mouse gesture and every reading is geometry the browser
    computed, including what a click at the close control's centre reaches.

    The three widths are the ones the handle can reach: `MIN_WIDTH`, the default, and
    `MAX_WIDTH`. 360 alone would not exercise much -- a 240px rail and a 360px dock fit in
    every window that still has the rail in the row -- and it is the default and the maximum
    that the row cannot seat once the window narrows.

    Both rail states, at every width, because the rail is what the dock's share of the row is
    measured against and the two answers differ between 600 and 1140. Nothing here decides
    whether the rail *should* collapse below a breakpoint (#1013/#1018): it asserts the same
    property under both of the rail's current states, so it holds whichever way that is
    answered. With the rail closed the whole surface is read, tab strip included; with it open
    the reading is the close control alone, because below 600px the open rail is drawn over the
    dock's left 240px and takes the click on the tab under it -- measured, and what an open
    overlay does rather than anything this narrowing caused.

    Mutation-checked by hand, not as a kill declaration, because the lethality ratchet runs a
    browser test against the committed bundle without rebuilding it. In
    `frontend/src/components/layout/ArtifactsDock.tsx`, `windowWidth - reservedWidth <= width`
    becoming `windowWidth - reservedWidth * 0 <= width` -- the pre-#1042 window-only rule;
    deleting the term outright instead fails `tsc` on the now-unused prop, which is #1042's own
    guard -- rebuilt with `vite build`, fails `[520-open]` at 600, 601, 700 and 759 and
    `[900-open]` at 1024, in the reported shape and words: `the dock x 240.0-760.0 is outside
    the window 0-700`, `dock-close: x 724.0-748.0, a click at its centre reaches nothing`.
    (`ArtifactsDock.test.tsx` declares that same mutation and dies with it; what is added here
    is that the box, not the style attribute, is what goes wrong, and that it goes wrong under
    a narrowing rather than at a mount.)
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": 1280, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda err: errors.append(str(err)))

            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            await set_the_rail(page, rail)
            await _open_the_dock(page)

            await _drag_handle_to(page, 1280 - dragged)
            wide = await page.evaluate(_DOCK_AND_STORAGE)
            assert wide["stored"] == str(dragged), (
                f"the drag asked for {dragged}px and {wide['stored']!r} was stored, so the "
                f"width this case narrows under is not the one it names: {wide}"
            )
            assert await page.evaluate(_DOCK_OFF_SCREEN, "tab-artifacts") == [], wide

            found: dict[int, list[str]] = {}
            for narrowed in _NARROWED_TO:
                await page.set_viewport_size({"width": narrowed, "height": 900})
                await page.evaluate(TWO_FRAMES)
                problems: list[str] = await page.evaluate(_CLOSE_CONTROL_UNREACHABLE)
                # With the rail closed nothing is drawn over the dock, so the whole surface is
                # read there: the same narrowing must not put the tab strip out of reach either.
                if rail == "closed":
                    problems += await page.evaluate(_DOCK_OFF_SCREEN, "tab-artifacts")
                if problems:
                    state = await page.evaluate(_DOCK_AND_STORAGE)
                    found[narrowed] = [*problems, f"state {state}"]
            assert found == {}, f"dragged to {dragged}px at 1280px, rail {rail}: {found}"

            assert errors == [], f"page errors {errors}"
        finally:
            await browser.close()


#: The drawer's surfaces and the label each tab reads, in drawer order.
_DEVELOPER_TAB_LABELS = (
    ("knowledge_graph", "Knowledge Graph"),
    ("topology", "DAG"),
    ("ledger", "EventBus"),
    ("ontology", "Ontology"),
)

#: Surfaces the drawer held until #1358 moved them into Settings. No tab may offer them.
_MOVED_TO_SETTINGS = ("acp", "skills", "evaluations")

#: How many text nodes on the page carry the drawer's label, not counting the dock surface's own
#: content. The surface is excluded because Docs & Artifacts renders the workspace's Markdown, and
#: `docs/ui-dashboard-architecture.md` names the drawer: when it is the document on show, a
#: page-wide text search finds the label in prose about the drawer. That is a reader's document,
#: not the dock offering the drawer, so it is the dock's chrome that is searched.
_DRAWER_LABEL_OUTSIDE_THE_SURFACE = """() => {
    const surface = document.querySelector("[data-testid='dock-surface']");
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let found = 0;
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        if (!/developer tools/i.test(n.textContent)) continue;
        if (surface && surface.contains(n)) continue;
        found += 1;
    }
    return found;
}"""

#: The primary row: the only surfaces the dock offers while developer mode is off.
_USER_TABS = ("clone", "turn", "artifacts", "remembers", "activity", "resource")


@pytest.mark.asyncio
async def test_developer_mode_is_off_by_default_and_one_setting_turns_it_on(
    ui_test_server: str,
) -> None:
    """Owner ruling 2026-09-22: developer mode is off by default, and one setting turns it on.

    Four things a browser has to show, in the order a user meets them.

    1. **Off by default.** The dock offers its six user surfaces and nothing else: no drawer,
       no "Developer Tools" label, and no tab of any kind -- `tab-` or `dev-tab-` -- for the
       four instruments. Checked with the dock *open*, because a closed
       dock renders no tabs at all and every absence would hold vacuously.
    2. **One setting turns it on.** The Settings switch adds the drawer, with all four tabs
       reading their own labels, and a drawer tab opens its own surface. ACP, Skills and Evals
       are not among them in either state: #1358 moved them into Settings.
    3. **Kept across a reload**, which is what a preference is.
    4. **Switched off with an instrument open, the dock shows Docs & Artifacts** rather than
       the instrument with no tab to reach it, or nothing.

    Mutation-checked with a real `vite build` on each side, since the ratchet runs e2e cases
    against the committed bundle and so reads a declaration naming `frontend/src` as escaped.
    `Killed by:` frontend/src/components/layout/ArtifactsDock.tsx :: `{developerMode && (`
    becoming `{true && (` (arm 1: the drawer is on screen by default -- and with the `dev-drawer`
    count taken out of the test, the label count alone still fails it, 1 == 0);
    `Killed by:` frontend/src/lib/developerMode.ts ::
    `window.localStorage.setItem(DEVELOPER_MODE_KEY, String(on));` becoming `void on;`
    (arm 3: the drawer is gone after the reload);
    `Killed by:` frontend/src/lib/developerMode.ts :: `? 'artifacts' : active` becoming
    `? active : active` (arm 4: the dock keeps the ledger and Docs & Artifacts never renders).
    Each fails this case at its own arm and no other case in this file,
    `test_dock_metric_cards_e2e.py` or `test_dashboard_e2e.py` (1 failed, 47 passed).
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            dock = await _open_the_dock(page)

            # 1. Off by default: the six user tabs, and nothing of the drawer's.
            for tab in _USER_TABS:
                assert await dock.locator(f"[data-testid='tab-{tab}']").count() == 1, tab
            assert await page.locator("[data-testid='dev-drawer']").count() == 0
            assert await page.evaluate(_DRAWER_LABEL_OUTSIDE_THE_SURFACE) == 0
            for surface, _label in _DEVELOPER_TAB_LABELS:
                for testid in (f"tab-{surface}", f"dev-tab-{surface}"):
                    assert await page.locator(f"[data-testid='{testid}']").count() == 0, testid

            # 2. One setting turns it on, and the drawer holds all four, each under its label.
            await turn_on_developer_mode(page)
            drawer = dock.locator("[data-testid='dev-drawer']")
            await drawer.wait_for(state="visible", timeout=5000)
            assert await drawer.get_by_text("Developer Tools", exact=True).count() == 1
            for surface, label in _DEVELOPER_TAB_LABELS:
                tab_el = drawer.locator(f"[data-testid='dev-tab-{surface}']")
                assert await tab_el.is_visible(), surface
                assert (await tab_el.inner_text()).strip() == label, surface
                # In the drawer only, never in the primary row as well.
                assert await page.locator(f"[data-testid='tab-{surface}']").count() == 0
            for moved in _MOVED_TO_SETTINGS:
                for testid in (f"tab-{moved}", f"dev-tab-{moved}"):
                    assert await page.locator(f"[data-testid='{testid}']").count() == 0, testid
            await drawer.locator("[data-testid='dev-tab-ledger']").click()
            ledger = dock.get_by_text("EventBus SSE Stream Ledger")
            await ledger.wait_for(state="visible", timeout=5000)

            # 3. Kept across a reload. The dock's open state is not kept, so it is reopened.
            await page.reload(wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            dock = await _open_the_dock(page)
            drawer = dock.locator("[data-testid='dev-drawer']")
            await drawer.wait_for(state="visible", timeout=5000)
            for surface, _label in _DEVELOPER_TAB_LABELS:
                assert await drawer.locator(f"[data-testid='dev-tab-{surface}']").is_visible()

            # 4. Switched off with an instrument open: Docs & Artifacts, not the instrument.
            await drawer.locator("[data-testid='dev-tab-ledger']").click()
            await dock.get_by_text("EventBus SSE Stream Ledger").wait_for(
                state="visible", timeout=5000
            )
            await page.locator("[data-testid='open-settings']").click()
            switch = page.locator("[data-testid='developer-mode-switch']")
            await switch.wait_for(state="visible")
            await switch.click()
            await page.wait_for_selector(
                "[data-testid='developer-mode-switch'][aria-checked='false']", timeout=5000
            )
            await page.locator("[role='dialog'] button[title='Close']").click()
            await page.locator("[role='dialog']").wait_for(state="detached")

            await dock.locator("[data-testid='doc-viewer']").wait_for(state="visible", timeout=5000)
            assert await dock.get_by_text("EventBus SSE Stream Ledger").count() == 0
            assert await page.locator("[data-testid='dev-drawer']").count() == 0
        finally:
            await browser.close()


#: The ACP report's own words on a build with no ACP shell, which is what the test server is.
_ACP_REPORT = ("Agent Client Protocol", "Not serving ACP")

#: What the Skills section says with the test server's catalogue, which registers no skill. The
#: section states that absence in words rather than drawing four zero-count cards; that it lists
#: a registered skill is `SettingsModal.test.tsx`'s case, against a mocked `/api/skills`.
_SKILLS_EMPTY = "[data-testid='settings-skills-empty']"


@pytest.mark.asyncio
async def test_settings_holds_skills_always_and_diagnostics_only_in_developer_mode(
    ui_test_server: str,
) -> None:
    """#1358: Skills is a Settings section; ACP and Evals are Diagnostics, behind developer mode.

    Both states, in the order a user meets them.

    1. **Developer mode off (the default).** Settings shows the Skills section, and neither
       the Diagnostics section nor any piece of it -- the ACP report, the Evals scorecard --
       is on the page. No dock tab offers any of the three.
    2. **Developer mode on.** The same dialog, without closing it, now shows Diagnostics with
       the ACP report stating what this build serves and the Evals scorecard's four cards; the
       dock's drawer still offers none of the three.

    Mutation-checked by hand with a real `vite build` on each side, because the ratchet runs a
    browser test against the committed bundle and so reads a declaration naming `frontend/src`
    as escaped. `Killed by:` frontend/src/components/SettingsModal.tsx ::
    `{developerMode && <DiagnosticsSection />}` becoming `{<DiagnosticsSection />}` (arm 1:
    Diagnostics is on screen with developer mode off); `Killed by:`
    frontend/src/components/SettingsModal.tsx :: `<SkillsSection />` becoming
    `{false && <SkillsSection />}` (arm 1: no Skills section; removing the element outright
    leaves its import unused, which `tsc` refuses before the bundle is built).
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": 1280, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda err: errors.append(str(err)))
            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)

            # 1. Off: Skills is there, Diagnostics is not, and no dock tab offers any of them.
            await page.locator("[data-testid='open-settings']").click()
            switch = page.locator("[data-testid='developer-mode-switch']")
            await switch.wait_for(state="visible")
            assert await switch.get_attribute("aria-checked") == "false"
            skills = page.locator("[data-testid='settings-skills']")
            await skills.locator(_SKILLS_EMPTY).wait_for(timeout=10000)
            assert "No skills are registered" in await skills.inner_text()
            assert await page.locator(_DIAGNOSTICS).count() == 0
            for marker in _ACP_REPORT:
                assert await page.get_by_text(marker).count() == 0, marker
            assert await page.locator("[data-testid='eval-metric-card']").count() == 0

            # 2. On: the same dialog grows Diagnostics, with both instruments in it.
            await switch.click()
            section = page.locator(_DIAGNOSTICS)
            await section.wait_for(state="visible", timeout=5000)
            await section.get_by_text(_ACP_REPORT[1]).first.wait_for(timeout=10000)
            for marker in _ACP_REPORT:
                assert await section.get_by_text(marker).count() >= 1, marker
            await section.locator("[data-testid='eval-metric-card']").first.wait_for(timeout=10000)
            assert await section.locator("[data-testid='eval-metric-card']").count() == 4
            assert await skills.locator(_SKILLS_EMPTY).count() == 1

            await page.locator("[role='dialog'] button[title='Close']").click()
            await page.locator("[role='dialog']").wait_for(state="detached")
            dock = await _open_the_dock(page)
            await dock.locator("[data-testid='dev-drawer']").wait_for(state="visible")
            for moved in _MOVED_TO_SETTINGS:
                for testid in (f"tab-{moved}", f"dev-tab-{moved}"):
                    assert await page.locator(f"[data-testid='{testid}']").count() == 0, testid

            assert errors == [], f"page errors {errors}"
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_failed_skills_read_names_its_cause_and_can_be_tried_again(
    ui_test_server: str,
) -> None:
    """#1369: a Skills read that fails says so, with its cause, and "Try again" reads again.

    The first `/api/skills` request is aborted at the network, the way a stopped runtime
    fails it; nothing about the answer is written by the test. The section must say the
    skills could not be loaded and why, in plain words -- never the browser's transport
    message ("Failed to fetch"), which means nothing to someone who does not read the code,
    and never the empty-catalogue sentence. The route is then removed, so "Try again" reaches the real test server, whose
    catalogue registers no skill -- and the section says that, in place of the failure,
    without Settings being closed.

    Mutation-checked by hand with a real `vite build`, because the ratchet runs a browser test
    against the committed bundle and so reads a declaration naming `frontend/src` as escaped.
    `Killed by:` frontend/src/components/settings/SkillsSection.tsx :: `onRetry={reload}`
    becoming `onRetry={() => {}}` (the failure stays on screen after "Try again").
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": 1280, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda err: errors.append(str(err)))
            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)

            await page.route("**/api/skills", lambda route: route.abort())
            await page.locator("[data-testid='open-settings']").click()
            skills = page.locator("[data-testid='settings-skills']")
            failure = skills.locator("[data-testid='settings-skills-error']")
            await failure.wait_for(timeout=10000)
            text = await failure.inner_text()
            assert "Your skills could not be loaded." in text
            assert "UClone-X could not be reached." in text
            assert "Failed to fetch" not in text
            assert await skills.locator(_SKILLS_EMPTY).count() == 0

            await page.unroute("**/api/skills")
            await failure.get_by_role("button", name="Try again").click()
            await skills.locator(_SKILLS_EMPTY).wait_for(timeout=10000)
            assert await failure.count() == 0
            assert "No skills are registered" in await skills.inner_text()
            assert errors == [], f"page errors {errors}"
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [375, 1280])
async def test_the_settings_sections_hold_their_content_inside_their_edges(
    ui_test_server: str, width: int
) -> None:
    """#1358: Skills and Diagnostics carry `.dock-scope`, and hold every control inside them.

    The panels were drawn for the dock and are now drawn in a dialog. At a phone width the
    dialog is the window's width, so this is the case where a grid of four would push its last
    card or a control past the section's edge; at 1280px the dialog is `max-w-2xl`.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page: Page = await browser.new_page(viewport={"width": width, "height": 900})
            await page.goto(ui_test_server, wait_until="commit")
            await page.wait_for_selector("[data-testid='room-composer']", timeout=20000)
            section = await open_diagnostics(page)
            await section.locator("[data-testid='eval-metric-card']").first.wait_for(timeout=10000)
            await page.locator(_SKILLS_EMPTY).wait_for(timeout=10000)
            await page.evaluate(TWO_FRAMES)

            found = {
                name: await page.evaluate(SURFACE_OVERFLOW, selector)
                for name, selector in (
                    ("skills", "[data-testid='settings-skills']"),
                    ("diagnostics", _DIAGNOSTICS),
                )
            }
            assert found == {"skills": [], "diagnostics": []}, f"{width}px: {found}"
            assert await page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth + 0.5"
            ), f"{width}px: the page scrolls sideways"
        finally:
            await browser.close()
