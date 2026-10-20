"""A `type` and a `click` sent in one step reach a real page in order (#2392).

In the first live run of the browser eval, `js_rendered_read` typed the order number and
clicked Confirm, the form was never sent, and the reply said the button did nothing. That
is consistent with the two calls racing in one step. Sent together through the tool in the
run's recorded reproduction, the type's result read "Nothing on the page changed.", the
click's result carried the type's change under the type's element, and no form was
submitted. Here the same pair goes through `asyncio.gather`, as the agent runs a step's
calls.
"""

from __future__ import annotations

import asyncio
import threading
import urllib.parse
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.support.chromium import chromium_executable
from uclone_x.browser import BrowserService, R2Link
from uclone_x.browser.tool import BrowserParams, BrowserTool
from uclone_x.tools.models import ToolContext

pytestmark = pytest.mark.e2e

_FORM = b"""<!doctype html><html><head><meta charset="utf-8"><title>Order status</title></head>
<body><main><h1>Order status</h1>
<form method="post" action="/order">
<label for="order">Order number</label><input id="order" name="order" type="text">
<button type="submit">Confirm</button></form></main></body></html>"""
_DONE = b"""<!doctype html><html><head><meta charset="utf-8"><title>Received</title></head>
<body><p>Thank you.</p></body></html>"""


@pytest.fixture(scope="module")
def site() -> Iterator[tuple[str, list[dict[str, str]]]]:
    posted: list[dict[str, str]] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._send(_FORM)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            fields = urllib.parse.parse_qs(self.rfile.read(length).decode())
            posted.append({k: v[-1] for k, v in fields.items()})
            self._send(_DONE)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", posted
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


async def test_type_then_click_in_one_step_sends_the_form_with_what_was_typed(
    link: R2Link, site: tuple[str, list[dict[str, str]]]
) -> None:
    base, posted = site

    async def _factory() -> R2Link:
        return link

    tool = BrowserTool(BrowserService(_factory))
    context = ToolContext(agent_id="scout", session_id="s-1")
    await tool.run(BrowserParams(action="open", url=base + "/order"), context)

    typed, clicked = await asyncio.gather(
        tool.run(BrowserParams(action="type", ref="e1", text="7Q-4821"), context),
        tool.run(BrowserParams(action="click", ref="e2"), context),
    )

    assert posted == [{"order": "7Q-4821"}]
    assert isinstance(typed, dict) and isinstance(clicked, dict)
    assert typed["element"] == "Order number"
    assert "7Q-4821" in typed["changes"]
    assert clicked["element"] == "Confirm"
    assert clicked["title"] == "Received"
