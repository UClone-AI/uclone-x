"""End-to-end: a story's view in Files, and a person deciding its proposals (#1560).

`tests/unit/test_story_view.py` pins the Core's decisions and
`frontend/src/components/artifacts/StoryView.test.tsx` pins the screen against stubbed
answers. What only the assembled product shows is that a person reaches the view from
Files, sees the proposed change before and after with the quote it rests on and the
conversation that proposed it, and that pressing Approve or Reject changes the story's
files on disk -- written as the conversation writing the story.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from playwright.async_api import Page, async_playwright

from tests.e2e.conftest import mock_llm, running_ui
from uclone_x.story.library import StoryLibrary
from uclone_x.story.schemas import Chapter, Outline, Proposal, Scene, dump_file

pytestmark = pytest.mark.e2e


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def server(tmp_path: Path, workspace: Path) -> Iterator[str]:
    with running_ui(
        storage_dir=tmp_path / "sessions", llm=mock_llm(), workspace_dir=workspace
    ) as url:
        yield url


async def _create_room(page: Page, base_url: str, title: str) -> str:
    response = await page.request.post(
        f"{base_url}/api/rooms", data={"title": title, "agent_ids": ["scout"]}
    )
    assert response.ok, await response.text()
    body: dict[str, Any] = await response.json()
    return str(body["room_id"])


def _write_story(workspace: Path, room_id: str) -> Path:
    """The Salt Road, written by `room_id`: Lord Vane, and two proposals about him."""
    library = StoryLibrary(workspace)
    story_id = library.create("The Salt Road", room_id).story_id
    root = workspace / "stories" / story_id

    def put(relative: str, text: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    outline = Outline(
        chapters=[
            Chapter(
                id="ch01",
                title="The Crossing",
                scenes=[Scene(id="ch01.s01", title="The gate", story_time="day 3")],
            )
        ]
    )
    put("outline.yaml", dump_file(outline))
    put("manuscript/ch01.s01.md", "An arrow grazed Lord Vane at the gate.")
    put(
        "codex/characters/vane.yaml",
        yaml.safe_dump({"id": "vane", "name": "Lord Vane", "state": {"status": "alive"}}),
    )
    digest = library.read_file(story_id, "codex/characters/vane.yaml").digest
    for number, (key, value) in enumerate([("wounded", True), ("mood", "furious")], start=1):
        proposal = Proposal.model_validate(
            {
                "id": f"p{number:03d}",
                "kind": "characters",
                "entry_id": "vane",
                "change": {"progression": {"at": "ch01.s01", "set": {key: value}}},
                "evidence": [{"scene_id": "ch01.s01", "quote": "An arrow grazed Lord Vane"}],
                "proposed_at": "2026-09-25T10:00:00+00:00",
                "room_id": room_id,
                "agent_id": "writer",
                "entry_digest": digest,
            }
        )
        put(f"proposals/p{number:03d}.yaml", dump_file(proposal, compact=True))
    return root


@pytest.mark.asyncio
async def test_a_person_decides_a_proposal_in_the_story_view(server: str, workspace: Path) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(server, wait_until="networkidle")
            room_id = await _create_room(page, server, "Writing room")
            root = _write_story(workspace, room_id)

            await page.get_by_test_id("open-artifact-library").click()
            row = page.locator(f'[data-testid="files-entry"][data-path="stories/{root.name}"]')
            await row.wait_for(timeout=10_000)
            await row.get_by_test_id("files-view-story").click()
            await page.get_by_test_id("story-view").wait_for(timeout=10_000)

            # The change as it is and as it would be, its quote, and who proposed it.
            first = page.locator('[data-testid="story-proposal"][data-proposal="p001"]')
            await first.wait_for(timeout=10_000)
            text = await first.inner_text()
            assert "Lord Vane" in text
            assert "Proposed in “Writing room”" in text
            assert "An arrow grazed Lord Vane" in text
            change = first.get_by_test_id("story-change")
            assert await change.get_by_test_id("story-change-before").inner_text() == "not set"
            assert await change.get_by_test_id("story-change-after").inner_text() == "true"
            assert "after “The gate”" in await change.inner_text()

            # Approving writes the entry and closes the proposal, as decided here.
            await first.get_by_test_id("story-approve").click()
            notice = page.get_by_test_id("story-view-notice")
            await notice.get_by_text("The change to Lord Vane was applied.").wait_for(
                timeout=10_000
            )
            vane = yaml.safe_load((root / "codex/characters/vane.yaml").read_text("utf-8"))
            assert vane["progressions"] == [{"at": "ch01.s01", "set": {"wounded": True}}]
            decided = yaml.safe_load((root / "proposals/p001.yaml").read_text("utf-8"))
            assert (decided["status"], decided["decided_in"]) == ("applied", "story_view")

            # The other proposal was made against the entry as it was, so it cannot be
            # approved now; rejecting it records the reason given.
            second = page.locator('[data-testid="story-proposal"][data-proposal="p002"]')
            await second.get_by_test_id("story-blocked").wait_for(timeout=10_000)
            assert await second.get_by_test_id("story-approve").is_disabled()
            await second.get_by_test_id("story-reject").click()
            await second.get_by_test_id("story-reject-reason").fill("He stays calm.")
            await second.get_by_test_id("story-confirm-reject").click()
            await notice.get_by_text("The change to Lord Vane was rejected.").wait_for(
                timeout=10_000
            )
            rejected = yaml.safe_load((root / "proposals/p002.yaml").read_text("utf-8"))
            assert (rejected["status"], rejected["reason"]) == ("rejected", "He stays calm.")

            items = page.get_by_test_id("story-decided-item")
            assert await items.count() == 2
            assert (
                "No change is waiting for a decision"
                in await page.get_by_test_id("story-pending").inner_text()
            )
        finally:
            await browser.close()
