"""Sign-in detection and the path B hand-off (design `browser-agent.md` §3.5, §6 step 4b).

Real Chromium over R2, pages on loopback. The person's part — typing into the page while the
tab is theirs — is played through the page itself, as the dock's forwarded input would land.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.support.chromium import chromium_executable
from uclone_x.browser import BrowserService, R2Link
from uclone_x.browser.live import LiveView

pytestmark = pytest.mark.e2e

_PLAIN = """<!doctype html><html><head><meta charset="utf-8"><title>Home</title></head>
<body><h1>Welcome</h1><label for="q">Search</label><input id="q"></body></html>"""
_LOGIN = """<!doctype html><html><head><meta charset="utf-8"><title>Sign in</title></head>
<body><form><label for="u">Username</label><input id="u" autocomplete="username">
<label for="p">Password</label><input id="p" type="password">
<button type="button">Sign in</button></form></body></html>"""
_HIDDEN = """<!doctype html><html><head><meta charset="utf-8"><title>Saved</title></head>
<body><p>Your settings</p><input type="password" style="display:none"></body></html>"""
_DENIED = b"""<!doctype html><html><head><meta charset="utf-8"><title>Members</title></head>
<body><p>Members only.</p></body></html>"""

#: What the person types while the tab is theirs. It must come back in nothing the clone reads.
SENTINEL = "sentinel-pw-7Qx2-never-in-a-result"  # noqa: S105

KEY = ("conversation-1", "scout")


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("site")
    (root / "plain.html").write_text(_PLAIN, encoding="utf-8")
    (root / "login.html").write_text(_LOGIN, encoding="utf-8")
    (root / "hidden.html").write_text(_HIDDEN, encoding="utf-8")

    class _Handler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:
            if self.path == "/members":
                self.send_response(401)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_DENIED)))
                self.end_headers()
                self.wfile.write(_DENIED)
                return
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Handler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def chromium() -> Path:
    return chromium_executable()


@pytest.fixture
async def link(tmp_path: Path, chromium: Path) -> AsyncIterator[R2Link]:
    started = await R2Link.start(
        tmp_path / "profile", chrome=chromium, extra_args=("--headless=new",)
    )
    yield started
    await started.shutdown()


@pytest.fixture
async def service(link: R2Link) -> BrowserService:
    async def _factory() -> R2Link:
        return link

    return BrowserService(_factory)


@pytest.mark.parametrize("page", ["login.html", "members"])
async def test_a_sign_in_page_and_a_refused_page_say_a_sign_in_is_likely(
    service: BrowserService, site: str, page: str
) -> None:
    opened = await service.open(KEY, f"{site}/{page}")

    assert "127.0.0.1" in opened["sign_in_likely"]
    assert "Call sign_in" in opened["sign_in_likely"]
    assert "sign_in_likely" in await service.snapshot(KEY)


@pytest.mark.parametrize("page", ["plain.html", "hidden.html"])
async def test_an_ordinary_page_and_a_hidden_password_field_say_nothing(
    service: BrowserService, site: str, page: str
) -> None:
    opened = await service.open(KEY, f"{site}/{page}")

    assert "sign_in_likely" not in opened


async def test_what_the_person_types_while_signing_in_reaches_nothing_the_clone_reads(
    service: BrowserService, site: str, link: R2Link, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    await service.open(KEY, f"{site}/login.html")
    notices = service.watch(KEY[0])
    asking = asyncio.create_task(service.ask_user(KEY, message="I need you to sign in"))
    for _ in range(50):
        if service.live_state(KEY[0]).get("asked_user"):
            break
        await asyncio.sleep(0.05)
    state = service.live_state(KEY[0])
    assert state["asked_user"]["site"] == "127.0.0.1"
    assert not asking.done()

    # The person types into the page while it is theirs.
    target = service.live_target(KEY[0], KEY[1], 1)
    assert target is not None
    await link.send(
        target[1],
        "Runtime.evaluate",
        {
            "expression": "document.getElementById('u').value = 'person';"
            f"document.getElementById('p').value = {json.dumps(SENTINEL)};"
        },
    )
    service.give_back(KEY[0], KEY[1])
    returned = await asyncio.wait_for(asking, timeout=10)
    found = await service.find(KEY, "Password")

    heard: list[dict[str, Any]] = []
    while not notices.empty():
        heard.append(notices.get_nowait())
    for what in (returned, found, heard, service.live_state(KEY[0]), caplog.text):
        assert SENTINEL not in json.dumps(what, ensure_ascii=False)
    assert "gave the tab back" in returned["message"]
    # Still on the sign-in page, so the observation still says so.
    assert "sign_in_likely" in returned


class _Head:
    """The dock's side of the live socket: what it was sent, and what it sends."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.json: list[dict[str, Any]] = []

    async def send_json(self, message: dict[str, Any]) -> None:
        self.json.append(message)

    async def send_bytes(self, data: bytes) -> None:
        return

    async def receive_json(self) -> dict[str, Any] | None:
        return await self.inbox.get()


async def test_a_password_typed_through_the_dock_reaches_the_page_and_nothing_else(
    service: BrowserService, site: str, link: R2Link, caplog: pytest.LogCaptureFixture
) -> None:
    """The dock's own input route (`LiveView` forwarding `input`), not page script: the
    person's keystrokes during a hand-off land in the field and nowhere the clone or a log
    can read them."""
    caplog.set_level(logging.DEBUG)
    await service.open(KEY, f"{site}/login.html")
    target = service.live_target(KEY[0], KEY[1], 1)
    assert target is not None
    await link.send(
        target[1], "Runtime.evaluate", {"expression": "document.getElementById('p').focus()"}
    )
    notices = service.watch(KEY[0])
    head = _Head()
    view = asyncio.create_task(LiveView(service, KEY[0], head).run())
    try:
        head.inbox.put_nowait({"type": "take_over"})
        for _ in range(100):
            if service.live_state(KEY[0])["tabs"][0]["controller"] == "user":
                break
            await asyncio.sleep(0.02)
        head.inbox.put_nowait({"type": "input", "event": "text", "text": SENTINEL})
        typed: object = None
        for _ in range(100):
            value = await link.send(
                target[1],
                "Runtime.evaluate",
                {"expression": "document.getElementById('p').value", "returnByValue": True},
            )
            typed = value.get("result", {}).get("value")
            if typed == SENTINEL:
                break
            await asyncio.sleep(0.02)
        assert typed == SENTINEL, "the dock's input never reached the page"
        head.inbox.put_nowait({"type": "give_back"})
        await asyncio.sleep(0.2)
        after = await service.snapshot(KEY)
    finally:
        head.inbox.put_nowait(None)
        await asyncio.wait_for(view, 5)

    heard = [notices.get_nowait() for _ in range(notices.qsize())]
    for what in (head.json, heard, service.live_state(KEY[0]), after, caplog.text):
        assert SENTINEL not in json.dumps(what, ensure_ascii=False, default=str)
