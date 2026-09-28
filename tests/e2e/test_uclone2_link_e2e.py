"""End-to-end: a clone linked to uClone2 in Settings shows as online, and can be unlinked.

`tests/unit/test_links_api.py` pins the routes and `LinksSection.test.tsx` pins the section.
What only the assembled product can show is the loop through the shipped bundle: the
Connections tab reaches `POST /api/links/uclone2`, the supervisor starts the session at
once, the card and the clone's own page say it is online, the switch takes it offline, and
unlinking empties the list -- with the token on no screen along the way.

uClone2 is the in-process fake (`tests/support/uclone2_fake.py`), and the socket session is
a stub that reports online as soon as it is started, so nothing leaves the machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import yaml
from playwright.async_api import async_playwright

from tests.e2e.conftest import dock_locator, mock_llm, running_ui
from tests.support.uclone2_fake import CONNECT_URL, TOKEN, TOKEN_TAIL, FakeUclone2
from uclone_x.link.uclone2.client import Uclone2LinkClient
from uclone_x.link.uclone2.models import LinkRecord
from uclone_x.link.uclone2.session import LinkSession, LinkSessionState
from uclone_x.link.uclone2.store import LinkStore
from uclone_x.link.uclone2.supervisor import LinkSupervisor

pytestmark = pytest.mark.e2e


class _OnlineSession:
    """A session that dials nothing and is online from the moment it starts."""

    def __init__(self, record: LinkRecord) -> None:
        self.link_id = record.link_id
        self.state = LinkSessionState.CONNECTING

    def start(self) -> None:
        self.state = LinkSessionState.ONLINE

    async def stop(self, *, logout: bool, final: LinkSessionState) -> None:
        del logout
        self.state = final


def _session(record: LinkRecord, _store: LinkStore, _client: Uclone2LinkClient) -> LinkSession:
    return cast(LinkSession, _OnlineSession(record))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """One local clone named like the uClone2 clone the fake links (`haru`)."""
    path = tmp_path / "workspace"
    directory = path / ".uclone" / "personas"
    directory.mkdir(parents=True)
    (directory / "haru.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "haru",
                "role": "Gardener",
                "description": "haru tends a garden, and this sentence says so.",
                "system_prompt": "You garden.",
                "allowed_tools": [],
                "enable_write_tools": False,
                "enable_subagent_tools": False,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def server(tmp_path: Path, workspace: Path) -> Iterator[str]:
    supervisor = LinkSupervisor(
        store=LinkStore(tmp_path / "links" / "uclone2.json"),
        client=Uclone2LinkClient(transport=FakeUclone2().transport),
        session_factory=_session,
    )
    with running_ui(
        storage_dir=tmp_path / "sessions",
        llm=mock_llm(),
        workspace_dir=workspace,
        link_supervisor=supervisor,
    ) as url:
        yield url


@pytest.mark.asyncio
async def test_a_clone_linked_in_settings_is_online_on_its_page_and_can_be_unlinked(
    server: str,
) -> None:
    """Paste the connect URL; the card and the clone page say online; offline; unlink.

    Killed by: src/uclone_x/ui/links.py :: "state": link_state(record, session_states, elsewhere=elsewhere),
    Becomes: "state": "online",
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(server, wait_until="networkidle")
            await page.get_by_test_id("room-composer").wait_for(timeout=20_000)

            await page.get_by_test_id("open-settings").click()
            await page.get_by_test_id("settings-tab-links").click()
            section = page.get_by_test_id("settings-links")
            await section.get_by_test_id("settings-links-empty").wait_for(timeout=10_000)

            await section.get_by_test_id("settings-links-input").fill(CONNECT_URL)
            await section.get_by_test_id("settings-links-submit").click()
            await section.get_by_test_id("settings-links-linked").wait_for(timeout=10_000)

            card = section.locator("[data-testid^='link-card-lnk']")
            state = card.get_by_test_id("link-card-state")
            await state.get_by_text("Online — active in uClone2").wait_for(timeout=10_000)
            assert await card.get_by_test_id("link-card-page").inner_text() == "@haru"

            shown = await page.locator("body").inner_text()
            assert TOKEN not in shown and "ucl_" not in shown and TOKEN_TAIL not in shown

            # The clone's own page carries the line while the link is online.
            await page.locator("[role='dialog'] button[title='Close']").click()
            await page.get_by_test_id("clone-avatar-haru").click()
            await (await dock_locator(page)).wait_for(state="visible", timeout=5_000)
            line = page.get_by_test_id("clone-link-line")
            await line.wait_for(timeout=10_000)
            assert await line.inner_text() == "Active in uClone2 · @haru"

            await page.get_by_test_id("open-settings").click()
            await page.get_by_test_id("settings-tab-links").click()
            await card.get_by_test_id("link-card-toggle").click()
            await state.get_by_text("Switched offline").wait_for(timeout=10_000)
            assert await card.get_by_test_id("link-card-toggle").inner_text() == "Go online"

            await card.get_by_test_id("link-card-unlink").click()
            await (
                card.get_by_test_id("link-unlink-confirm")
                .get_by_role("button", name="Yes, unlink")
                .click()
            )
            await section.get_by_test_id("settings-links-empty").wait_for(timeout=10_000)

            listed = await page.request.get(f"{server}/api/links")
            assert listed.ok and (await listed.json())["links"] == []
        finally:
            await browser.close()
