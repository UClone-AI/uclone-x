"""End-to-end: approving a Composer proposal in the docked score view (#2421 item 7).

`tests/unit/test_music_ai_scenarios.py` pins the routes and `MusicView.test.tsx` pins the
screen against stubbed answers. What only the assembled product shows is the click the owner
reported doing nothing: a song whose composing conversation is gone showed its proposal with
an Approve button that was disabled, with no word of why. Here the person opens the score
view beside a conversation, presses Approve, reads the notice, and the track on disk changes.
It also shows the person gate (#2421 round 2): from a window the server did not open, the click
is refused in plain words and nothing changes; the window it opens on request may decide.

Mutations, checked by hand because the ratchet does not rebuild the bundle this test loads
(each side built with `NODE_ENV=production BROWSERSLIST=defaults npm run build`):

* `Killed by:` in `frontend/src/components/artifacts/MusicView.tsx`, the line
  `const canDecide = Boolean(song && (song.writable || conversationId));` becoming
  `const canDecide = Boolean(song && song.writable);` -- the test fails at its first click,
  which waits on a disabled Approve button: disabled again, as the owner saw it.
* `Killed by:` in `frontend/src/components/artifacts/MusicView.tsx`, the line
  `className="overflow-x-auto rounded-lg border border-slate-800 bg-[#090d16] shadow-xl"`
  losing `overflow-x-auto` -- `test_a_long_score_scrolls_to_its_last_bar` fails at "bar 16 should be
  reachable by scrolling": the last bar stays cut off, as the owner saw it.
* `self._library.take_over_if_held_by(song_id, room_id, expected_holder=holder)` in
  `examples/music/src/uclone_x_music/view.py` becoming `pass` -- the test fails at the notice, which reads the plain refusal instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from playwright.async_api import Page, async_playwright, expect

from tests.e2e.conftest import mock_llm, running_ui
from tests.unit.test_music_ai_scenarios import (
    _ctx as _tool_context,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.test_music_ai_scenarios import song_with_pending_violin_patch
from uclone_x.extensions import use_extensions
from uclone_x_music.tools import MusicScoreParams, MusicScoreTool
from uclone_x_music.ucx_extension import EXTENSION as MUSIC_EXTENSION
from uclone_x_story.ucx_extension import EXTENSION as STORY_EXTENSION

pytestmark = pytest.mark.e2e


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@dataclass
class _Server:
    url: str
    app: Any
    #: The addresses the server asked the person's browser to open.
    opened: list[str] = field(default_factory=list[str])


@pytest.fixture
def server(tmp_path: Path, workspace: Path) -> Iterator[_Server]:
    held: dict[str, Any] = {}
    opened: list[str] = []

    def capture(app: Any) -> None:
        held["app"] = app

        def _open(url: str) -> bool:
            opened.append(url)
            return True

        app.state.person_gate.opener = _open

    # The suite runs with the story as its one extension; this server also mounts the music
    # example, whose routes and dock the test drives.
    with (
        use_extensions([STORY_EXTENSION, MUSIC_EXTENSION]),
        running_ui(
            storage_dir=tmp_path / "sessions",
            llm=mock_llm(),
            workspace_dir=workspace,
            configure=capture,
        ) as url,
    ):
        yield _Server(url=url, app=held["app"], opened=opened)


async def _create_room(page: Page, base_url: str, title: str) -> str:
    response = await page.request.post(
        f"{base_url}/api/rooms", data={"title": title, "agent_ids": ["composer"]}
    )
    assert response.ok, await response.text()
    body: dict[str, Any] = await response.json()
    return str(body["room_id"])


async def _open_music_score(page: Page) -> None:
    # The dock remembers being open, so a second window of the same browser may have it open.
    music = page.get_by_test_id("dock-mode-music")
    if not await music.is_visible():
        await page.get_by_role("button", name="Open workspace dock").click()
    await music.click()
    await page.get_by_test_id("music-proposals-section").wait_for(timeout=10_000)


@pytest.mark.asyncio
async def test_a_person_approves_a_composer_proposal_in_the_docked_score_view(
    server: _Server, workspace: Path
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            context = await browser.new_context(viewport={"width": 1600, "height": 900})
            page = await context.new_page()
            await page.goto(server.url, wait_until="networkidle")
            room_id = await _create_room(page, server.url, "Swan Lake")
            # The conversation that staged the proposal has since been deleted; the song is
            # the one open in this conversation.
            proposed, proposal_id = await song_with_pending_violin_patch(
                workspace, "room_deleted_000"
            )
            server.app.state.room_stack.service.set_story(room_id, "midnight-nocturne")
            song_dir = workspace / "songs" / "midnight-nocturne"
            proposal_file = song_dir / f"proposals/{proposal_id}.yaml"

            # A window the server did not open cannot approve, and nothing changes.
            await page.goto(server.url, wait_until="networkidle")
            before = proposal_file.read_bytes()
            await _open_music_score(page)
            await page.get_by_test_id(f"proposal-approve-{proposal_id}").click()
            notice = page.get_by_test_id("music-proposal-notice")
            await expect(notice).to_contain_text("Only you can make this decision", timeout=10_000)
            assert proposal_file.read_bytes() == before
            await notice.get_by_test_id("music-open-confirmed-window").click()
            await expect(notice).to_contain_text("A confirmed window opened", timeout=10_000)
            assert len(server.opened) == 1

            # The window the server opened confirms itself and may decide.
            page = await context.new_page()
            await page.goto(server.opened[0], wait_until="networkidle")
            await _open_music_score(page)
            approve = page.get_by_test_id(f"proposal-approve-{proposal_id}")
            await expect(page.get_by_test_id("music-proposals-writer-note")).to_contain_text(
                "opens it in this conversation"
            )
            await expect(approve).to_be_enabled()

            await approve.click()
            await expect(page.get_by_test_id("music-proposal-notice")).to_contain_text(
                "Approved: Lyrical violin opening", timeout=10_000
            )
            await expect(page.get_by_test_id(f"music-proposal-{proposal_id}")).to_have_count(0)

            track = yaml.safe_load((song_dir / "tracks/theme_a_violin_solo.yaml").read_text())
            assert [b["notes"] for b in track["bars"][:2]] == [b.notes for b in proposed]
            decided = yaml.safe_load(proposal_file.read_text())
            assert decided["status"] == "approved"
            record = yaml.safe_load((song_dir / "song.yaml").read_text())
            assert record["lease"]["holder"] == room_id
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_long_score_scrolls_to_its_last_bar(server: _Server, workspace: Path) -> None:
    """A score longer than the dock is reachable: it scrolls sideways to its last bar."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await (
                await browser.new_context(viewport={"width": 1600, "height": 900})
            ).new_page()
            await page.goto(server.url, wait_until="networkidle")
            room_id = await _create_room(page, server.url, "Swan Lake")
            await song_with_pending_violin_patch(workspace, room_id)
            # Twelve more bars take the four-bar section to sixteen, wider than any dock.
            await MusicScoreTool().run(
                MusicScoreParams(
                    action="append_bars",
                    section_id="theme_a",
                    count=12,
                    chords=["Dm7", "Gm7", "A7", "Dm7"] * 3,
                    sketch=True,
                ),
                _tool_context(workspace, room_id=room_id, song_id="midnight-nocturne"),
            )
            server.app.state.room_stack.service.set_story(room_id, "midnight-nocturne")
            await page.goto(server.url, wait_until="networkidle")
            await _open_music_score(page)
            await expect(page.get_by_test_id("score-playhead")).to_contain_text(
                "Bar 1 / 16", timeout=10_000
            )

            scroller = page.get_by_test_id("music-score-scroller")
            geometry = await scroller.evaluate(
                "e => ({ scrollable: e.scrollWidth > e.clientWidth, client: e.clientWidth })"
            )
            assert geometry["scrollable"], geometry
            last_cell = page.get_by_test_id("score-cell-violin_solo-16")
            assert not await _inside(scroller, last_cell), "bar 16 should start off screen"

            await scroller.evaluate("e => { e.scrollLeft = e.scrollWidth; }")
            assert await _inside(scroller, last_cell), "bar 16 should be reachable by scrolling"
            await expect(page.get_by_test_id("music-transport-follow")).to_have_attribute(
                "aria-pressed", "true"
            )
        finally:
            await browser.close()


async def _inside(scroller: Any, cell: Any) -> bool:
    """Whether `cell` lies fully inside `scroller`'s visible box, left of its sticky labels aside."""
    outer = await scroller.bounding_box()
    inner = await cell.bounding_box()
    assert outer is not None and inner is not None
    return inner["x"] >= outer["x"] and inner["x"] + inner["width"] <= outer["x"] + outer["width"]
