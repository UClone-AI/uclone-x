"""End-to-end: picking a clone puts that clone on the dock, with its picture (#1300).

Two things only the assembled product can show.

The first is the picture's whole loop: a file installed beside a clone's definition, read by
`GET /api/personas/{name}/avatar`, addressed by the shipped bundle, and decoded by the
browser. `tests/unit/test_persona_avatar.py` pins the route and
`frontend/src/ui-kit.avatar.test.tsx` pins the component, but jsdom loads no images at all --
an `<img>` whose source 404s looks exactly like one that decoded, so neither end can tell a
picture that arrived from one that did not.

The second is the reason the clone's card moved out of the rail. It carries a model, two
ceilings, a tool list and two permission sentences, and it used to render inside a 240px
column, where the tool pills wrapped four deep. The assertion is on rendered width, because
that is the whole of what was wrong: a card with every field present and 240px to draw them
in passes every selector written about it.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from playwright.async_api import FloatRect, Page, async_playwright

from tests.e2e.conftest import TWO_FRAMES, dock_locator, mock_llm, running_ui

pytestmark = pytest.mark.e2e

#: A 1x1 transparent PNG, as in `tests/unit/test_persona_avatar.py`. A real one: the browser
#: reports `naturalWidth` 0 for bytes it could not decode, so anything else would pass the
#: request half of this test and fail the point of it.
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

#: The rail's width. The card being wider than this is the move, not a decoration.
RAIL_WIDTH_PX = 240


def _install(workspace: Path, name: str, role: str) -> Path:
    directory = workspace / ".uclone" / "personas"
    directory.mkdir(parents=True, exist_ok=True)
    definition = directory / f"{name}.yaml"
    definition.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "role": role,
                "description": f"{name} does one thing, and this sentence says what.",
                "system_prompt": "You work.",
                "allowed_tools": [],
                "enable_write_tools": False,
                "enable_subagent_tools": False,
            }
        ),
        encoding="utf-8",
    )
    return definition


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Two clones: one with a picture installed beside it, one with none."""
    path = tmp_path / "workspace"
    path.mkdir()
    _install(path, "surveyor", "Site Surveyor").with_suffix(".png").write_bytes(ONE_PIXEL_PNG)
    _install(path, "courier", "Dispatch Runner")
    return path


@pytest.fixture
def server(tmp_path: Path, workspace: Path) -> Iterator[str]:
    with running_ui(
        storage_dir=tmp_path / "sessions", llm=mock_llm(), workspace_dir=workspace
    ) as url:
        yield url


async def _pick(page: Page, clone: str) -> None:
    await page.get_by_test_id(f"clone-avatar-{clone}").click()
    await (await dock_locator(page)).wait_for(state="visible", timeout=5_000)
    await page.get_by_test_id(f"clone-profile-{clone}").wait_for(timeout=5_000)


@pytest.mark.asyncio
async def test_a_picked_clone_arrives_on_the_dock_with_the_picture_installed_for_it(
    server: str,
) -> None:
    """The bytes reach the screen, and the card has room for what it says.

    `Killed by:` frontend/src/ui-kit/rail/Rail.tsx :: `imageSrc={avatarSrc?.(clone.id)}`
    becoming `imageSrc={avatarSrc?.('')}` -- every row then asks for the same empty name,
    the route refuses it, and the rail draws the default beside every clone whatever is
    installed. No unit test sees that: jsdom fetches no images, so an element pointed at a
    URL that answers 404 is indistinguishable there from one whose picture decoded.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(server, wait_until="networkidle")
            await page.get_by_test_id("room-composer").wait_for(timeout=20_000)

            # The rail's own row, which is the binding this head owns.
            row_picture = page.get_by_test_id("clone-avatar-surveyor").locator("img")
            await row_picture.wait_for(timeout=10_000)
            assert await row_picture.evaluate("(el) => el.complete && el.naturalWidth > 0"), (
                "the rail's avatar element is on screen but its picture never decoded"
            )

            await _pick(page, "surveyor")

            assert await page.get_by_test_id("clone-profile-name").inner_text() == "surveyor"
            # `inner_text` is what the reader sees, and the line is drawn in small caps:
            # the assertion is on the words, not on the CSS that cases them.
            role = await page.get_by_test_id("clone-profile-role").inner_text()
            assert role.lower() == "site surveyor", role

            profile_picture = page.get_by_test_id("clone-profile-avatar").locator("img")
            assert await profile_picture.evaluate("(el) => el.complete && el.naturalWidth > 0")

            # And the card has more room than the column it came out of.
            await page.evaluate(TWO_FRAMES)
            box = await page.get_by_test_id("persona-detail-surveyor").bounding_box()
            assert box is not None, "the clone's configuration card is not rendered"
            assert box["width"] > RAIL_WIDTH_PX, (
                f"the card is {box['width']}px wide, no wider than the {RAIL_WIDTH_PX}px rail "
                "it was moved out of"
            )
            assert await page.get_by_test_id("clone-profile-start").is_visible()
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_clone_with_no_picture_installed_shows_the_default_and_no_broken_image(
    server: str,
) -> None:
    """The ordinary case, and the one a fallback written in jsdom cannot be trusted on.

    Most clones have no picture. The route answers 404, the browser fires `error`, and the
    component drops the element rather than leaving the broken-image mark on the row -- a
    wrong-looking screen that says nothing about why, which is what P6 forbids.

    No mutation is declared for it, and the reason is what it adds: that the *browser's* own
    `error` event fires on the real route's real 404. `ui-kit.avatar.test.tsx` fires that
    event by hand, which is the only way jsdom can, and carries the declaration for the
    component's response to it -- `setFailedSrc(imageSrc);` becoming `setFailedSrc(null);`.
    The same edit reddens this case, so declaring it here would be a claim about that one
    that is not true; what is left over here is the platform's half, which no edit to this
    repository isolates.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(server, wait_until="networkidle")
            await page.get_by_test_id("room-composer").wait_for(timeout=20_000)

            await _pick(page, "courier")

            avatar = page.get_by_test_id("clone-profile-avatar")
            await page.evaluate(TWO_FRAMES)
            assert await avatar.locator("img").count() == 0, (
                "a clone with no picture installed is still holding an image element"
            )
            # Not an empty circle either: the default is drawn, and the name is beside it.
            assert await avatar.locator("svg").count() == 1
            assert await page.get_by_test_id("clone-profile-name").inner_text() == "courier"
        finally:
            await browser.close()


async def _box(page: Page, test_id: str) -> FloatRect:
    box = await page.get_by_test_id(test_id).bounding_box()
    assert box is not None, f"{test_id} is not rendered"
    return box


@pytest.mark.asyncio
async def test_studio_mode_takes_the_conversation_column_and_escape_gives_it_back(
    server: str,
) -> None:
    """Studio mode is the conversation column plus the dock, and Escape returns both (#1377).

    The defect #1347 fixed was invisible to every component test: the editor drew itself as
    a `position: fixed` panel from inside the dock, and the dock's `backdrop-blur-md` makes
    the dock the containing block of a fixed descendant, so "full screen" was the dock's own
    box less a margin. jsdom does no layout, so only a browser can say where the editor went.
    The assertions are therefore on rendered boxes: the editor must be wider than the dock it
    was opened from and must start left of it, in the column the conversation had.

    Escape then leaves Studio mode without leaving the editor, so what was typed is still in
    the field (P6: leaving a mode never discards an edit silently).

    `Killed by:` frontend/src/components/clones/CloneProfile.tsx ::
    `useEscapeOwner('studio', studio && (isCreate || isEdit) && onStudioChange !== undefined, (event) => {`
    becoming `useEscapeOwner('studio', false, (event) => {` -- Escape then does nothing and
    the editor stays in Studio mode. The layout half was shown by reintroducing the trapped
    overlay (see the PR for #1377); a frontend mutation needs a rebuild, so it is recorded
    there rather than run by the ratchet.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(server, wait_until="networkidle")
            await page.get_by_test_id("room-composer").wait_for(timeout=20_000)

            await _pick(page, "surveyor")
            await page.get_by_test_id("clone-profile-edit").click()
            editor = page.get_by_test_id("clone-profile-editor")
            await editor.wait_for(timeout=5_000)
            await page.evaluate(TWO_FRAMES)
            main = page.locator("main")
            main_before = await main.bounding_box()
            assert main_before is not None, "the conversation column is not rendered"
            dock_before = await _box(page, "artifacts-dock")

            await page.get_by_test_id("toggle-studio-mode").click()
            await page.wait_for_selector(
                "[data-testid='clone-profile-editor'][data-studio='true']", timeout=5_000
            )
            await page.evaluate(TWO_FRAMES)

            assert not await main.is_visible(), "the conversation is still beside the editor"
            studio = await _box(page, "clone-profile-editor")
            assert studio["width"] > dock_before["width"], (
                f"the Studio editor is {studio['width']}px wide, no wider than the "
                f"{dock_before['width']}px dock it was opened from"
            )
            assert studio["x"] < dock_before["x"], (
                f"the Studio editor starts at x={studio['x']}, inside the dock "
                f"(x={dock_before['x']}), not in the conversation's column"
            )
            dock_studio = await _box(page, "artifacts-dock")
            assert abs(dock_studio["x"] - main_before["x"]) <= 2, (
                f"in Studio mode the workspace starts at x={dock_studio['x']}, not where the "
                f"conversation column did (x={main_before['x']})"
            )

            role = page.locator("#persona-role")
            await role.fill("Night surveyor")
            await page.keyboard.press("Escape")
            await page.wait_for_selector(
                "[data-testid='clone-profile-editor'][data-studio='false']", timeout=5_000
            )
            await page.evaluate(TWO_FRAMES)

            assert await main.is_visible(), "Escape left Studio mode but not the conversation"
            dock_after = await _box(page, "artifacts-dock")
            assert abs(dock_after["width"] - dock_before["width"]) <= 2, (
                f"the dock came back {dock_after['width']}px wide, not {dock_before['width']}px"
            )
            assert await role.input_value() == "Night surveyor", (
                "leaving Studio mode discarded what was typed"
            )
        finally:
            await browser.close()
