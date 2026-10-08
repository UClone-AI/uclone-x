"""End-to-end: a Settings field is kept once it is changed, with no Save button to press.

The unit tests (`frontend/src/components/SettingsModal.test.tsx`, `tests/unit/test_ui_server.py`)
pin the hook, the request body and the route. What only the assembled product can show is the
loop through the shipped bundle: a field changed reaches the settings file, and a fresh page
reads it back.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from tests.e2e.conftest import running_ui
from uclone_x.llm.connectors.mock import MockLLMConnector

pytestmark = pytest.mark.e2e


@pytest.fixture
def storage(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture
def server(storage: Path) -> Iterator[str]:
    llm = MockLLMConnector(default_model="mock-gpt-4o", default_response="ok")
    with running_ui(storage_dir=storage, llm=llm) as url:
        yield url


@pytest.mark.asyncio
async def test_a_field_changed_is_kept_across_a_reload(
    server: str, storage: Path, tmp_path: Path
) -> None:
    """Add a folder clones may read, reload: it is still there, with nothing pressed to save.

    The ComfyUI address this test used to type went with model-gateway step 5 (a ComfyUI is
    a connection now). Mutation-checked by hand, not as a kill declaration, because the
    lethality ratchet runs a browser test against the committed bundle without rebuilding
    it: rebuilt with `vite build` after `readRootsSave.commit(next);` in
    `handleAddReadRoot` of `frontend/src/components/SettingsModal.tsx` became nothing, this
    fails waiting for the save.
    """
    papers = tmp_path / "papers"
    papers.mkdir()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1400, "height": 900})
            await page.goto(server, wait_until="networkidle")

            await page.get_by_test_id("open-settings").click()
            field = page.get_by_label("Folder to add")
            await field.wait_for(state="visible", timeout=10_000)
            assert await page.get_by_role("button", name="Save & Apply Settings").count() == 0

            await field.fill(str(papers))
            async with page.expect_response(
                lambda r: r.url.endswith("/api/settings") and r.request.method == "POST"
            ) as saved:
                await field.press("Enter")
            response = await saved.value
            assert response.ok
            assert json.loads(response.request.post_data or "{}") == {"read_roots": [str(papers)]}

            await page.reload(wait_until="networkidle")
            await page.get_by_test_id("open-settings").click()
            await expect(page.get_by_test_id("settings-read-roots")).to_contain_text(
                str(papers), timeout=10_000
            )
        finally:
            await browser.close()

    written = json.loads((storage / "settings.json").read_text())
    assert written["read_roots"] == [str(papers)]
