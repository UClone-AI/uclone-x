"""R1 end to end: the real extension in a real Chromium, paired with a Core on loopback.

The Chromium Playwright installed for the e2e suite loads `chrome_extension/` unpacked,
headless. A plain `R2Link` to that same Chromium drives the extension's options page the
way a person would (paste a code, press Connect); the clone's tabs then go through the
extension, and the service's fallback is a stub that fails, so a pass cannot have come
from R2.

Skips when the extension does not load (a Chromium build without `--load-extension`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import AsyncIterator, Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from playwright.sync_api import sync_playwright

from uclone_x.browser import BrowserService, R2Link
from uclone_x.browser.chrome import free_port
from uclone_x.browser.extension import (
    EXTENSION_FOLDER,
    REFUSED_MESSAGE,
    ExtensionHub,
    RoutedLink,
)
from uclone_x.browser.link import BrowserLink
from uclone_x.ui.browser import pairing_token, register_browser_routes

pytestmark = pytest.mark.e2e

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>확장 시험</title></head>
<body><main><h1>R1 페이지</h1><p>확장으로 읽은 문장</p><a href="/page.html">다시</a></main></body></html>"""


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("site")
    (root / "page.html").write_text(_PAGE, encoding="utf-8")

    class _Quiet(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def chromium() -> Path:
    with sync_playwright() as p:
        return Path(p.chromium.executable_path)


@pytest.fixture
async def core(tmp_path: Path) -> AsyncIterator[tuple[ExtensionHub, str]]:
    """A Core with only the browser routes, on a real loopback port."""
    settings = tmp_path / "settings.json"
    hub = ExtensionHub(lambda: pairing_token(settings))
    app = FastAPI()

    def _allow(request: Request) -> None:
        return None

    register_browser_routes(app, hub=hub, settings_file=settings, refuse_cross_origin=_allow)
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    )
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.05)
    yield hub, f"http://127.0.0.1:{port}"
    await hub.disconnect()
    server.should_exit = True
    await task


def _unpacked_id(folder: Path) -> str:
    """Chrome's id for an unpacked extension: its path's SHA-256, spelled with a-p."""
    digest = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:32]
    return "".join(chr(ord("a") + int(c, 16)) for c in digest)


@pytest.fixture
async def chrome(tmp_path: Path, chromium: Path) -> AsyncIterator[R2Link]:
    folder = str(EXTENSION_FOLDER.resolve())
    started = await R2Link.start(
        tmp_path / "profile",
        chrome=chromium,
        extra_args=(
            "--headless=new",
            f"--disable-extensions-except={folder}",
            f"--load-extension={folder}",
        ),
    )
    yield started
    await started.shutdown()


async def _evaluate(chrome: R2Link, tab: str, expression: str) -> Any:
    result = await chrome.send(
        tab,
        "Runtime.evaluate",
        {"expression": expression, "returnByValue": True, "awaitPromise": True},
    )
    return result.get("result", {}).get("value")


async def _status_after(chrome: R2Link, tab: str, code: str, expect: str) -> str:
    """Paste `code` into the options page, press Connect, and wait for `expect` status."""
    await _evaluate(
        chrome,
        tab,
        f"document.getElementById('code').value = {json.dumps(code)};"
        "document.getElementById('save').click(); true",
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 15
    status = ""
    while loop.time() < deadline:
        status = await _evaluate(chrome, tab, "document.getElementById('status').dataset.status")
        if status == expect:
            break
        await asyncio.sleep(0.1)
    return str(await _evaluate(chrome, tab, "document.getElementById('status').textContent"))


async def _open_options(chrome: R2Link) -> str:
    tab = await chrome.open_tab(f"chrome-extension://{_unpacked_id(EXTENSION_FOLDER)}/options.html")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while loop.time() < deadline:
        if await _evaluate(chrome, tab, "!!document.getElementById('code')"):
            return tab
        await asyncio.sleep(0.1)
    pytest.skip("This Chromium did not load the unpacked extension.")


async def test_the_extension_pairs_and_carries_a_clones_tab(
    core: tuple[ExtensionHub, str], chrome: R2Link, site: str
) -> None:
    hub, base = core
    async with httpx.AsyncClient() as client:
        code = (await client.get(f"{base}/api/browser/extension")).json()["pairing_code"]
    port = code.split("-", 1)[0]
    options = await _open_options(chrome)

    refused = await _status_after(chrome, options, f"{port}-{'0' * 32}", "refused")
    assert hub.status()["connected"] is False
    assert refused == REFUSED_MESSAGE  # the extension's copy and the Core's say the same

    connected = await _status_after(chrome, options, code, "connected")
    await hub.wait_connected(15)
    assert connected == "Connected. Your clones can use this Chrome."

    async def _no_r2() -> BrowserLink:
        raise AssertionError("the fallback must not start while the extension is paired")

    async def _routed() -> BrowserLink:
        return RoutedLink(hub, _no_r2)

    service = BrowserService(_routed)
    key = ("conversation", "clone")
    opened = await service.open(key, f"{site}/page.html")
    read = await service.read(key, 10_000)

    assert opened["title"] == "확장 시험"
    assert "R1 페이지" in opened["snapshot"]
    assert "확장으로 읽은 문장" in read["content"]


async def test_a_new_pairing_code_drops_the_extension(
    core: tuple[ExtensionHub, str], chrome: R2Link
) -> None:
    hub, base = core
    async with httpx.AsyncClient() as client:
        code = (await client.get(f"{base}/api/browser/extension")).json()["pairing_code"]
        options = await _open_options(chrome)
        await _status_after(chrome, options, code, "connected")
        await hub.wait_connected(15)

        await client.post(f"{base}/api/browser/extension/pairing-code")

    assert hub.status()["connected"] is False
