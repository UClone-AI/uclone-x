"""End-to-end: choosing a clone's picture from the head, and every picture on screen following.

What only the assembled product can show is the loop: a change made in one place -- the card
under a drawn picture, or the profile's avatar menu -- lands through the real route, and the
rail, which reads the listing, shows the new picture without a reload. The unit tests pin each
end (`tests/unit/test_persona_avatar.py` the route, the vitest files the components), but jsdom
decodes no image, so a rail pointed at an address that still serves the old bytes -- or none --
looks the same there as one that shows the new picture.

Also here: "Ask … to make one" opens a conversation with the request waiting in the box and
sends nothing, and Settings › Images says what can draw now in the head's own words.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from playwright.async_api import Page, async_playwright

from tests.e2e.conftest import dock_locator, mock_llm, running_ui
from uclone_x.llm.connectors.mock import MockLLMConnector

pytestmark = pytest.mark.e2e

#: A 1x1 PNG the browser decodes, as in `test_clone_profile_e2e.py`.
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

DRAWN = "artifacts/images/img_self_portrait.png"
REPLY = f"Here is one.\n\n![Me](/api/artifacts/content?path={DRAWN})"

#: What `rail picture` answers for a row that draws the default: no `<img>` at all.
DEFAULT = "default"


def _install(workspace: Path, name: str) -> None:
    directory = workspace / ".uclone" / "personas"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "role": "Illustrator",
                "description": f"{name} draws.",
                "system_prompt": "You draw.",
                "allowed_tools": [],
                "enable_write_tools": False,
                "enable_subagent_tools": False,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Two clones with no picture yet, and a picture one of them drew."""
    path = tmp_path / "workspace"
    path.mkdir()
    _install(path, "painter")
    _install(path, "courier")
    drawn = path / DRAWN
    drawn.parent.mkdir(parents=True)
    drawn.write_bytes(ONE_PIXEL_PNG)
    return path


@pytest.fixture
def server(tmp_path: Path, workspace: Path) -> Iterator[str]:
    llm = MockLLMConnector(default_model="mock-gpt-4o", default_response=REPLY)
    with running_ui(storage_dir=tmp_path / "sessions", llm=llm, workspace_dir=workspace) as url:
        yield url


@pytest.fixture
def plain_server(tmp_path: Path, workspace: Path) -> Iterator[str]:
    with running_ui(
        storage_dir=tmp_path / "sessions", llm=mock_llm(), workspace_dir=workspace
    ) as url:
        yield url


async def _rail_picture(page: Page, clone: str) -> str:
    """The rail row's picture: its address once decoded, or `DEFAULT` when it draws none."""
    return str(
        await page.get_by_test_id(f"clone-avatar-{clone}").evaluate(
            """(el) => {
                const img = el.querySelector('img');
                if (!img) return 'default';
                return img.complete && img.naturalWidth > 0 ? img.getAttribute('src') : 'loading';
            }"""
        )
    )


async def _wait_rail(page: Page, clone: str, *, shows_picture: bool) -> str:
    """Wait for the rail row to show a decoded picture, or the default, and return which."""
    now = ""
    for _ in range(100):
        now = await _rail_picture(page, clone)
        if (now == DEFAULT) != shows_picture and now != "loading":
            return now
        await page.wait_for_timeout(100)
    raise AssertionError(f"the rail's picture for {clone} stayed {now!r}")


@pytest.mark.asyncio
async def test_use_as_avatar_gives_the_drawn_picture_and_undo_takes_it_back(server: str) -> None:
    """The picture a clone drew becomes its face in the rail at once, and Undo restores the default.

    `Killed by:` frontend/src/App.tsx :: `onChanged: () => void fetchAllMetadata(),` becoming
    `onChanged: () => {},` -- the change lands, the card says so, and the rail keeps drawing
    the default until the next 15-second poll. The vitest files see only that `onChanged` was
    called; nothing there holds a rail that reads the listing again.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1400, "height": 900})
            response = await page.request.post(
                f"{server}/api/rooms", data={"title": "Studio", "agent_ids": ["painter"]}
            )
            assert response.ok, await response.text()
            room_id = str((await response.json())["room_id"])

            await page.goto(server, wait_until="commit")
            await page.click(f"[data-testid='conversation-{room_id}']", timeout=15_000)
            await page.fill("[data-testid='room-composer']", "Draw yourself, please.")
            await page.click("[data-testid='send-message']")
            await page.get_by_test_id("use-as-avatar-btn").wait_for(timeout=20_000)
            assert await _rail_picture(page, "painter") == DEFAULT

            await page.get_by_test_id("use-as-avatar-btn").click()
            done = page.get_by_test_id("use-as-avatar-done")
            await done.wait_for(timeout=10_000)
            assert "painter" in await done.inner_text()
            # Well inside the 15-second poll: the change itself asked for the listing again.
            src = await _wait_rail(page, "painter", shows_picture=True)
            assert "/api/personas/painter/avatar" in src

            await page.get_by_test_id("use-as-avatar-undo").click()
            await page.get_by_test_id("use-as-avatar-undone").wait_for(timeout=10_000)
            assert await _wait_rail(page, "painter", shows_picture=False) == DEFAULT
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_picture_uploaded_on_the_profile_is_the_clone_s_face_until_it_is_reset(
    plain_server: str, tmp_path: Path
) -> None:
    """Upload from the profile's menu, then Reset to default, each followed by the rail.

    `Killed by:` frontend/src/components/clones/CloneProfile.tsx ::
    `imageSrc={personaPicture(name, persona)}` becoming
    `imageSrc={personaPicture(name, persona) && undefined}` -- the rail
    follows the change while the profile's own picture never shows one, which jsdom cannot
    tell apart from a picture that decoded.
    """
    upload = tmp_path / "me.png"
    upload.write_bytes(ONE_PIXEL_PNG)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(plain_server, wait_until="networkidle")
            await page.get_by_test_id("room-composer").wait_for(timeout=20_000)
            await page.get_by_test_id("clone-avatar-courier").click()
            await (await dock_locator(page)).wait_for(state="visible", timeout=5_000)
            await page.get_by_test_id("clone-profile-courier").wait_for(timeout=5_000)

            await page.get_by_test_id("clone-profile-avatar").click()
            assert await page.get_by_test_id("clone-avatar-reset").is_disabled()
            # As a person does it: the menu item opens the browser's file chooser.
            async with page.expect_file_chooser() as chooser:
                await page.get_by_test_id("clone-avatar-upload").click()
            await (await chooser.value).set_files(str(upload))
            await page.get_by_test_id("clone-avatar-done").wait_for(timeout=10_000)
            await _wait_rail(page, "courier", shows_picture=True)
            profile = page.get_by_test_id("clone-profile-avatar").locator("img")
            await profile.wait_for(timeout=5_000)
            assert await profile.evaluate("(el) => el.complete && el.naturalWidth > 0")

            await page.get_by_test_id("clone-profile-avatar").click()
            await page.get_by_test_id("clone-avatar-reset").click()
            await page.get_by_test_id("clone-avatar-done").wait_for(timeout=10_000)
            assert await _wait_rail(page, "courier", shows_picture=False) == DEFAULT
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_asking_a_clone_for_pictures_waits_in_the_box_unsent(plain_server: str) -> None:
    """ "Ask … to make one" opens a conversation with the clone, the request typed and unsent.

    `Killed by:` frontend/src/App.tsx :: `void handleNewRoom(name, false, askText);` becoming
    `void handleNewRoom(name, false);` -- a conversation opens with an empty box, and the menu
    item looks like it did half its job.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(plain_server, wait_until="networkidle")
            await page.get_by_test_id("room-composer").wait_for(timeout=20_000)
            await page.get_by_test_id("clone-avatar-courier").click()
            await page.get_by_test_id("clone-profile-courier").wait_for(timeout=5_000)

            await page.get_by_test_id("clone-profile-avatar").click()
            await page.get_by_test_id("clone-avatar-ask").click()

            composer = page.get_by_test_id("room-composer")
            for _ in range(100):
                if "profile pictures" in await composer.input_value():
                    break
                await page.wait_for_timeout(100)
            assert "profile pictures" in await composer.input_value()
            # Unsent: the conversation holds the join and nothing the person said.
            await page.wait_for_timeout(500)
            rooms = await (await page.request.get(f"{plain_server}/api/rooms")).json()
            newest = max(rooms["rooms"], key=lambda r: str(r.get("updated_at", "")))
            room = await (
                await page.request.get(f"{plain_server}/api/rooms/{newest['room_id']}")
            ).json()
            said = [m for m in room["transcript"] if m.get("kind", "utterance") == "utterance"]
            assert said == [], said
            assert any(p["id"] == "courier" for p in room["participants"])
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_settings_images_says_what_draws_in_plain_words(plain_server: str) -> None:
    """The Images tab names what can draw, and never an engine id or a reason code.

    What draws depends on the machine -- a ComfyUI may be running, a model may be installed --
    so the assertion is on the words, whichever sentence it is.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1400, "height": 900})
            await page.goto(plain_server, wait_until="networkidle")
            await page.get_by_test_id("open-settings").click()
            tab = page.get_by_test_id("settings-tab-tools")
            assert (await tab.inner_text()).strip() == "Images"
            await tab.click()
            status = page.get_by_test_id("settings-images-status")
            await status.wait_for(timeout=15_000)
            text = await status.inner_text()
            assert "Pictures are drawn with" in text or "Nothing can draw pictures" in text
            for internal in ("comfyui-local", "diffusers-sdxl", "remote-cuda", "no_key", "_"):
                assert internal not in text, f"{internal!r} is on the screen: {text!r}"
        finally:
            await browser.close()
