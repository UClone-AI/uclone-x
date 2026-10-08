"""The dock's Browser tab, live (design `browser-agent.md` §6 step 3), over R2 against a real
Chromium: a screencast frame is a JPEG of the clone's tab, and an action shows its overlay.

The page is a fixture served on loopback; nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import AsyncIterator, Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import sync_playwright

from uclone_x.browser import BrowserService, R2Link
from uclone_x.browser.live import LiveView
from uclone_x.browser.tool import BrowserParams, BrowserTool
from uclone_x.tools.models import ToolContext

pytestmark = pytest.mark.e2e

_PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>찾기</title></head>
<body><main><label for="q">검색어</label><input id="q">
<button onclick="document.getElementById('out').textContent='찾았습니다'">검색</button>
<p id="out"></p></main></body></html>"""


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("site")
    (root / "search.html").write_text(_PAGE, encoding="utf-8")

    class _Quiet(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def chromium() -> Path:
    with sync_playwright() as p:
        return Path(p.chromium.executable_path)


@pytest.fixture
async def service(tmp_path: Path, chromium: Path) -> AsyncIterator[BrowserService]:
    link = await R2Link.start(tmp_path / "profile", chrome=chromium, extra_args=("--headless=new",))

    async def factory() -> R2Link:
        return link

    yield BrowserService(factory, downloads_dir=tmp_path / "downloads")
    await link.shutdown()


class _Head:
    def __init__(self) -> None:
        self.inbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.json: list[dict[str, Any]] = []
        self.binary: list[bytes] = []

    async def send_json(self, message: dict[str, Any]) -> None:
        self.json.append(message)

    async def send_bytes(self, data: bytes) -> None:
        self.binary.append(data)

    async def receive_json(self) -> dict[str, Any] | None:
        return await self.inbox.get()

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.json if m.get("type") == kind]


async def _until(check: Any, timeout: float = 10.0) -> None:
    async def wait() -> None:
        while not check():
            await asyncio.sleep(0.05)

    await asyncio.wait_for(wait(), timeout)


async def test_the_clones_tab_is_screencast_and_its_click_is_outlined(
    service: BrowserService, site: str
) -> None:
    tool = BrowserTool(service=service)
    context = ToolContext(agent_id="scout", session_id="s-1", room_id="room-1")
    await tool.run(BrowserParams(action="open", url=f"{site}/search.html"), context)
    head = _Head()
    view = asyncio.create_task(LiveView(service, "room-1", head).run())
    head.inbox.put_nowait({"type": "view", "on": True, "width": 800})

    await _until(lambda: head.binary)
    assert head.binary[0][:2] == b"\xff\xd8"  # a JPEG
    size = head.of("frame")[0]
    assert size["width"] > 0 and size["height"] > 0

    matches = (await service.find(("room-1", "scout"), "검색"))["matches"]
    ref = re.search(r'button "검색"[^\n]*\[ref=(e\d+)\]', matches)
    assert ref, matches
    result = await tool.run(BrowserParams(action="click", ref=ref.group(1)), context)

    assert isinstance(result, dict)
    assert result["element"] == "검색"
    await _until(lambda: head.of("step") and head.of("overlay"))
    overlay = head.of("overlay")[-1]
    assert overlay["label"] == "검색"
    box = overlay["box"]
    assert box["width"] > 0 and box["height"] > 0 and box["x"] >= 0 and box["y"] >= 0
    assert head.of("step")[-1] == {
        "type": "step",
        "clone": "scout",
        "action": "click",
        "element": "검색",
        "ok": True,
    }

    head.inbox.put_nowait(None)
    await asyncio.wait_for(view, 5)
