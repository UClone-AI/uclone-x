"""End-to-end: the Files screen, apart from any conversation (#1554).

`tests/unit/test_artifact_library.py` pins the Core's decisions and
`frontend/src/components/artifacts/ArtifactLibrary.test.tsx` pins the screen against stubbed
answers. What only the assembled product shows is that the header's Files button reaches a
screen fed by the real routes, over a real workspace: a file no living conversation links
is listed and opens, archive and restore move it on disk, delete asks first, and a story
opens into a new conversation whose record carries the story's id.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Locator, Page, async_playwright

from tests.e2e.conftest import mock_llm, running_ui
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
            assert "Not linked to a conversation that still exists" in await orphan.inner_text()
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
