"""End-to-end: the Files screen, apart from any conversation (#1554).

`tests/unit/test_artifact_library.py` pins the Core's decisions and
`frontend/src/components/artifacts/ArtifactLibrary.test.tsx` pins the screen against stubbed
answers. What only the assembled product shows is that the header's Files button reaches a
screen fed by the real routes, over a real workspace: a file no living conversation links
is listed and opens, archive and restore move it on disk, delete asks first, and a story
opens into a new conversation whose record carries the story's id.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Locator, Page, async_playwright

from tests.e2e.conftest import mock_llm, running_ui
from uclone_x.artifacts.library import READERS_NOT_CLEARED_NOTE
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.story.library import StoryLibrary

pytestmark = pytest.mark.e2e


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A file whose conversation is gone, and a story whose last writer is gone too."""
    path = tmp_path / "workspace"
    (path / "artifacts").mkdir(parents=True)
    (path / "artifacts" / "orphan.txt").write_text("written before the room went", "utf-8")
    StoryLibrary(path).create("Night Train", "room-deleted-long-ago")
    return path


@pytest.fixture
def server(tmp_path: Path, workspace: Path) -> Iterator[str]:
    with running_ui(
        storage_dir=tmp_path / "sessions", llm=mock_llm(), workspace_dir=workspace
    ) as url:
        yield url


def _row(page: Page, path: str) -> Locator:
    return page.locator(f'[data-testid="files-entry"][data-path="{path}"]')


async def _json(page: Page, url: str) -> Any:
    response = await page.request.get(url)
    assert response.ok, await response.text()
    return await response.json()


@pytest.mark.asyncio
async def test_files_are_managed_apart_from_any_conversation(server: str, workspace: Path) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(server, wait_until="networkidle")
            await page.get_by_test_id("open-artifact-library").click()
            await page.get_by_test_id("artifact-library").wait_for(timeout=10_000)
            await page.get_by_test_id("files-scope-note").wait_for(timeout=10_000)

            # A file from a conversation that no longer exists is listed and opens.
            orphan = _row(page, "artifacts/orphan.txt")
            await orphan.wait_for(timeout=10_000)
            assert "No recorded writer" in await orphan.inner_text()
            await orphan.get_by_test_id("files-open").click()
            preview = page.get_by_test_id("files-preview-text")
            await preview.wait_for(timeout=5_000)
            assert await preview.inner_text() == "written before the room went"

            # Archive moves it aside; restore puts it back.
            await orphan.get_by_test_id("files-archive").click()
            await page.get_by_text("orphan.txt was archived.").wait_for(timeout=5_000)
            assert (workspace / ".archive" / "artifacts" / "orphan.txt").is_file()
            assert not (workspace / "artifacts" / "orphan.txt").exists()
            await page.get_by_test_id("files-show-archived").check()
            archived = _row(page, ".archive/artifacts/orphan.txt")
            await archived.get_by_test_id("files-restore").click()
            await page.get_by_text("orphan.txt was restored.").wait_for(timeout=5_000)
            assert (workspace / "artifacts" / "orphan.txt").is_file()
            await page.get_by_test_id("files-show-archived").uncheck()

            # Delete asks first, and nothing is gone until the answer is yes.
            await orphan.get_by_test_id("files-delete").click()
            await page.get_by_role("alertdialog").wait_for(timeout=5_000)
            assert (workspace / "artifacts" / "orphan.txt").is_file()
            await page.get_by_test_id("files-confirm-delete").click()
            await page.get_by_text("orphan.txt was deleted.").wait_for(timeout=5_000)
            assert not (workspace / "artifacts" / "orphan.txt").exists()

            # A story opens into a new conversation that carries its id.
            stories = await _json(page, f"{server}/api/artifacts/library")
            story_id = next(e["story"]["story_id"] for e in stories["entries"] if e["story"])
            await _row(page, f"stories/{story_id}").get_by_test_id("files-open-story").click()
            await page.get_by_test_id("artifact-library").wait_for(state="hidden", timeout=10_000)

            listing = await _json(page, f"{server}/api/rooms")
            opened = [r for r in listing["rooms"] if r["title"] == "Night Train"]
            assert len(opened) == 1, listing
            room = await _json(page, f"{server}/api/rooms/{opened[0]['room_id']}")
            assert room["story_id"] == story_id
        finally:
            await browser.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes through a read-only folder"
)
async def test_an_archive_whose_readers_could_not_be_cleared_says_so(
    server: str, workspace: Path, tmp_path: Path
) -> None:
    """The story is archived and reported archived, and the note says what was left (#1578).

    A conversation has the story open to read. The conversations' folder is read-only, so
    clearing it fails after the story has already moved. One server is enough for this.

    Killed by: src/uclone_x/ui/artifacts.py :: return {"path": moved.path, "note": moved.note}
    Becomes: return {"path": moved.path, "note": None}

    Checked by hand, with the bundle rebuilt, because the ratchet does not rebuild it: in
    `ArtifactLibrary.tsx`, the notice set as `setNotice(note ? done : done);` fails this test.
    """
    rooms_dir = tmp_path / "sessions" / "rooms"
    store = RoomStore(rooms_dir)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(server, wait_until="networkidle")
            listing = await _json(page, f"{server}/api/artifacts/library")
            story_id = next(e["story"]["story_id"] for e in listing["entries"] if e["story"])
            reader = RoomService(store).create("Reading room")
            store.save(reader.model_copy(update={"story_id": story_id}))

            await page.get_by_test_id("open-artifact-library").click()
            row = _row(page, f"stories/{story_id}")
            await row.wait_for(timeout=10_000)
            mode = stat.S_IMODE(rooms_dir.stat().st_mode)
            rooms_dir.chmod(0o555)
            try:
                await row.get_by_test_id("files-archive").click()
                notice = page.get_by_test_id("files-notice")
                await notice.wait_for(timeout=5_000)
                text = await notice.inner_text()
            finally:
                rooms_dir.chmod(mode)

            assert "was archived." in text, text
            assert READERS_NOT_CLEARED_NOTE in text, text
            assert (workspace / ".archive" / "stories" / story_id).is_dir()
            assert not (workspace / "stories" / story_id).exists()
            still = store.load(reader.room_id)
            assert still is not None and still.story_id == story_id  # what the note says
        finally:
            await browser.close()
