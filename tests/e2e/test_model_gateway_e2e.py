"""End-to-end: Settings › Models, the clone editor's pickers and "Use system default".

model-gateway.md §3.7 (step 4) against the assembled product: the shipped bundle talking to
the Core's §3.7.1 routes, with nothing faked between them. Every connection is a `mock` row
(no model server) or a row of a kind this version does not know, which must be listed as
such rather than dropped (P6).

The unit tests pin each route (`tests/unit/test_model_gateway.py`) and each component
(`frontend/src/components/settings/*.test.tsx`); what only the assembled product shows is
that the screen's requests are the ones the Core answers, and that a choice reaches the file.

Mutation-checked by hand, not as kill declarations, because the lethality ratchet runs a
browser test against the committed bundle without rebuilding it: rebuilt with `vite build`
after `const unsupported = row.status === 'unsupported';` in
`frontend/src/components/settings/ConnectionsSection.tsx` became `const unsupported = false;`,
the first test fails; after `|| !ref) return null;` in
`frontend/src/components/rooms/UseSystemDefault.tsx` became `|| !ref || clone !== '') return
null;`, the third fails waiting for the button.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import Page, ViewportSize, async_playwright, expect

from tests.e2e.conftest import running_ui

pytestmark = pytest.mark.e2e

_VIEWPORT: ViewportSize = {"width": 1400, "height": 900}
_WIDE: ViewportSize = {"width": 1600, "height": 900}

#: A row of a kind this version does not know, with a field it cannot know either.
_FUTURE = {"id": "future", "kind": "martian-llm", "zone": "olympus"}


@pytest.fixture
def storage(tmp_path: Path) -> Path:
    path = tmp_path / "sessions"
    path.mkdir(parents=True)
    (path / "settings.json").write_text(
        json.dumps(
            {
                "connections": [{"id": "mock", "kind": "mock"}, _FUTURE],
                "default_models": {"deep": "mock/mock-gpt-4o"},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def server(storage: Path, tmp_path: Path) -> Iterator[str]:
    # No connector handed in: every turn goes through the settings file's connections.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with running_ui(storage_dir=storage, llm=None, workspace_dir=workspace) as url:
        yield url


def _settings(storage: Path) -> dict[str, Any]:
    return json.loads((storage / "settings.json").read_text(encoding="utf-8"))


async def _open_models(page: Page, server: str) -> None:
    await page.goto(server, wait_until="networkidle")
    await page.get_by_test_id("open-settings").click()
    await page.get_by_test_id("settings-connections").wait_for(timeout=10_000)


@pytest.mark.asyncio
async def test_connections_are_listed_and_a_default_model_chosen_reaches_the_file(
    server: str, storage: Path
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=_VIEWPORT)
            await _open_models(page, server)

            # The unknown kind is on the list, says so, and offers Remove but not Change.
            await expect(page.get_by_test_id("connection-status-future")).to_have_text(
                "This kind of connection is not supported", timeout=10_000
            )
            await expect(page.get_by_test_id("connection-remove-future")).to_be_visible()
            assert await page.get_by_test_id("connection-edit-future").count() == 0
            await expect(page.get_by_test_id("connection-row-mock")).to_be_visible()

            # Conversations: chosen from the mock connection's own listing.
            select = page.get_by_test_id("settings-default-deep-select")
            await expect(select.locator("option[value='mock/mock-llm']")).to_have_count(1)
            async with page.expect_response(
                lambda r: r.url.endswith("/api/settings") and r.request.method == "POST"
            ) as saved:
                await select.select_option("mock/mock-llm")
            response = await saved.value
            assert response.ok, await response.text()
            assert json.loads(response.request.post_data or "{}") == {
                "default_models": {"deep": "mock/mock-llm"}
            }

            await page.reload(wait_until="networkidle")
            await page.get_by_test_id("open-settings").click()
            await expect(page.get_by_test_id("settings-default-deep-select")).to_have_value(
                "mock/mock-llm", timeout=10_000
            )
        finally:
            await browser.close()

    written = _settings(storage)
    assert written["default_models"]["deep"] == "mock/mock-llm"
    # The row this version cannot read was written back exactly as it was.
    assert _FUTURE in written["connections"]


@pytest.mark.asyncio
async def test_a_clone_saved_with_its_own_model_keeps_that_ref(server: str) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=_WIDE)
            await _open_models(page, server)
            section = page.get_by_test_id("settings-personas")
            await section.get_by_role("button", name="New clone").click()
            await page.locator("#persona-name").fill("cartographer")
            await page.locator("#persona-role").fill("Map maker")
            await page.locator("#persona-prompt").fill("You draw maps.")
            picker = page.locator("#persona-model-name")
            # It starts on the system default, naming what that is now.
            await expect(picker.locator("option:checked")).to_contain_text(
                "System default", timeout=10_000
            )
            await picker.select_option("mock/mock-llm")
            await page.get_by_test_id("persona-editor").get_by_role("button", name="Save").click()
            await section.get_by_test_id("persona-row-cartographer").wait_for(timeout=10_000)

            read = await page.request.get(f"{server}/api/clones/cartographer")
            persona = (await read.json())["persona"]
        finally:
            await browser.close()

    assert (persona["model_name"], persona["fast_model"], persona["image_model"]) == (
        "mock/mock-llm",
        None,
        None,
    )


@pytest.mark.asyncio
async def test_a_turn_on_an_unusable_own_model_offers_use_system_default(server: str) -> None:
    """§3.6: the row names the clone's own model and offers the one action, which clears it."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=_VIEWPORT)
            await page.goto(server, wait_until="networkidle")
            made = await page.request.post(
                f"{server}/api/clones",
                data={
                    "name": "rover",
                    "role": "Explorer",
                    "system_prompt": "You explore.",
                    "model_name": "future/big-model",
                },
            )
            assert made.status == 201, await made.text()
            created = await page.request.post(
                f"{server}/api/rooms", data={"title": "Mars", "agent_ids": ["rover"]}
            )
            assert created.ok, await created.text()
            room_id = str((await created.json())["room_id"])

            await page.goto(server, wait_until="commit")
            await page.click(f"[data-testid='conversation-{room_id}']", timeout=15_000)
            await page.fill("[data-testid='room-composer']", "hello")
            await page.click("[data-testid='send-message']")

            button = page.get_by_test_id("row-use-default-4")
            await button.wait_for(timeout=20_000)
            stated = await page.get_by_test_id("row-error-4").inner_text()
            for internal in ("Error", "Traceback", "{"):
                assert internal not in stated, stated
            # Built by the dashboard from the failure's fields, naming the clone's own model
            # without its connection prefix (#2167).
            assert "it is set to use its own model, big-model," in " ".join(stated.split())
            await button.click()
            await page.get_by_test_id("row-use-default-4-done").wait_for(timeout=10_000)

            # The done line is drawn once the save answered, so one read sees it.
            read = await page.request.get(f"{server}/api/clones/rover")
            persona = (await read.json())["persona"]
        finally:
            await browser.close()

    assert persona["model_name"] is None
    assert (persona["role"], persona["system_prompt"]) == ("Explorer", "You explore.")


# -- Install and remove models on an Ollama connection (#2167) --


class _FakeOllama(BaseHTTPRequestHandler):
    """A test Ollama: `/api/tags` lists `installed`, `/api/pull` adds, `/api/delete` removes."""

    installed: dict[str, list[str]] = {}

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the base's name
        return

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _send(self, payload: object, content_type: str = "application/json") -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - the base's name
        if self.path == "/api/tags":
            models = [{"name": n, "capabilities": c} for n, c in self.installed.items()]
            self._send({"models": models})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        body = self._body()
        if self.path == "/api/pull":
            self.installed[str(body["model"])] = ["completion"]
            lines = [{"status": "pulling manifest"}, {"status": "success"}]
            self._send(
                b"".join(json.dumps(x).encode() + b"\n" for x in lines), "application/x-ndjson"
            )
        else:
            self.send_error(404)

    def do_DELETE(self) -> None:  # noqa: N802
        body = self._body()
        if self.path == "/api/delete" and self.installed.pop(str(body["model"]), None) is not None:
            self._send({})
        else:
            self.send_error(404)


@pytest.fixture
def ollama_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    _FakeOllama.installed = {"qwen3:14b": ["completion", "tools"], "bge-m3:latest": ["embedding"]}
    fake = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    storage = tmp_path / "sessions"
    storage.mkdir(parents=True)
    address = f"http://127.0.0.1:{fake.server_address[1]}"
    (storage / "settings.json").write_text(
        json.dumps({"connections": [{"id": "ollama", "kind": "ollama", "base_url": address}]}),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        with running_ui(storage_dir=storage, llm=None, workspace_dir=workspace) as url:
            yield url, storage
    finally:
        fake.shutdown()
        fake.server_close()


@pytest.mark.asyncio
async def test_an_ollama_connection_installs_and_removes_models_from_settings(
    ollama_server: tuple[str, Path],
) -> None:
    """Settings lists what the Ollama has, embedder said to be one, installs and removes.

    Mutation-checked by hand (the bundle is not rebuilt by the ratchet): rebuilt after
    `{row.kind === 'ollama' && (row.status === 'connected' || row.status === 'unchecked') && (` in
    `frontend/src/components/settings/ConnectionsSection.tsx` became `{false && (`, this
    test fails waiting for the open button.
    """
    server, _ = ollama_server
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport=_VIEWPORT)
            await _open_models(page, server)
            # The conversation picker offers the chat model and not the embedder (part 2).
            deep = page.get_by_test_id("settings-default-deep-select")
            await expect(deep.locator("option[value='ollama/qwen3:14b']")).to_have_count(
                1, timeout=10_000
            )
            assert await deep.locator("option[value='ollama/bge-m3:latest']").count() == 0

            await page.get_by_test_id("ollama-models-open-ollama").click()
            await expect(page.get_by_test_id("ollama-model-ollama-bge-m3:latest")).to_contain_text(
                "Not a conversation model", timeout=10_000
            )

            await page.get_by_test_id("ollama-install-input-ollama").fill("gemma3:4b")
            await page.get_by_test_id("ollama-install-ollama").click()
            await expect(page.get_by_test_id("ollama-models-notice-ollama")).to_have_text(
                "gemma3:4b is installed on Ollama.", timeout=15_000
            )
            await expect(page.get_by_test_id("ollama-model-ollama-gemma3:4b")).to_be_visible()
            await expect(page.get_by_test_id("connection-status-ollama")).to_contain_text(
                "2 models", timeout=10_000
            )

            await page.get_by_test_id("ollama-model-remove-ollama-qwen3:14b").click()
            await page.get_by_test_id("ollama-model-remove-yes-ollama-qwen3:14b").click()
            await expect(page.get_by_test_id("ollama-models-notice-ollama")).to_have_text(
                "qwen3:14b was removed from Ollama.", timeout=10_000
            )
        finally:
            await browser.close()

    assert set(_FakeOllama.installed) == {"bge-m3:latest", "gemma3:4b"}
