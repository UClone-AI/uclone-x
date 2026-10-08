"""R1: the extension hub, its pairing, R1Link and the route choice (design §3.6, §3.7, §3.10).

The extension is played by a websockets client speaking the extension's wire: a hello with
the token, then CDP framing with the tab id as the session id.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from uclone_x.browser.cdp import CdpClosedError
from uclone_x.browser.extension import (
    EXTENSION_FOLDER,
    POPUP_NOT_FOLLOWED,
    REFUSED_MESSAGE,
    ExtensionHub,
    RoutedLink,
)
from uclone_x.browser.link import BrowserLink
from uclone_x.errors import PlainRefusalError
from uclone_x.ui.browser import (
    TOKEN_KEY,
    pairing_token,
    regenerate_pairing_token,
    register_browser_routes,
)

TOKEN = "ab" * 16


class FakeExtension:
    """The extension's side of the wire: answers commands, records them, emits events."""

    def __init__(self, ws: ClientConnection) -> None:
        self.ws = ws
        self.received: list[dict[str, Any]] = []
        self.next_tab = 100
        self.task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        with contextlib.suppress(Exception):
            async for raw in self.ws:
                message = json.loads(raw)
                self.received.append(message)
                await self._answer(message)

    async def _answer(self, message: dict[str, Any]) -> None:
        method = message["method"]
        if method == "UClone.openTab":
            self.next_tab += 1
            result: dict[str, Any] = {"tabId": str(self.next_tab)}
        elif method == "Runtime.evaluate":
            await self.ws.send(
                json.dumps(
                    {
                        "method": "Page.loadEventFired",
                        "params": {},
                        "sessionId": message["sessionId"],
                    }
                )
            )
            result = {"result": {"value": f"tab {message['sessionId']}"}}
        else:
            result = {}
        await self.ws.send(json.dumps({"id": message["id"], "result": result}))

    def methods(self) -> list[str]:
        return [m["method"] for m in self.received]

    async def stop(self) -> None:
        await self.ws.close()
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task


@pytest.fixture
async def hub_url() -> AsyncIterator[tuple[ExtensionHub, str]]:
    hub = ExtensionHub(lambda: TOKEN)

    async def _serve(ws: ServerConnection) -> None:
        await hub.serve(ws)

    async with serve(_serve, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        yield hub, f"ws://127.0.0.1:{port}"
        await hub.disconnect()


async def _pair(url: str, token: str = TOKEN) -> tuple[ClientConnection, dict[str, Any]]:
    ws = await connect(url)
    await ws.send(json.dumps({"type": "hello", "token": token, "version": "0.1.0"}))
    reply = json.loads(await ws.recv())
    return ws, reply


async def test_the_current_token_pairs_and_the_hub_reports_connected(
    hub_url: tuple[ExtensionHub, str],
) -> None:
    hub, url = hub_url
    assert hub.status()["connected"] is False

    ws, reply = await _pair(url)
    await hub.wait_connected(5)

    assert reply == {"type": "paired"}
    status = hub.status()
    assert status["connected"] is True
    assert status["version"] == "0.1.0"
    await ws.close()


async def test_a_wrong_token_is_refused_in_plain_words(hub_url: tuple[ExtensionHub, str]) -> None:
    hub, url = hub_url

    ws, reply = await _pair(url, token="cd" * 16)

    assert reply["type"] == "refused"
    assert reply["message"] == REFUSED_MESSAGE
    for internal in ("token", "hmac", "Exception", "WebSocket", "4401", "Traceback"):
        assert internal not in reply["message"]
    with pytest.raises(ConnectionClosed):
        await asyncio.wait_for(ws.recv(), 5)
    assert ws.close_code == 4401
    assert hub.status()["connected"] is False


async def test_a_hello_that_is_not_json_is_refused(hub_url: tuple[ExtensionHub, str]) -> None:
    hub, url = hub_url
    ws = await connect(url)
    await ws.send("hello")

    reply = json.loads(await ws.recv())

    assert reply["type"] == "refused"
    assert hub.status()["connected"] is False
    await ws.close()


async def test_without_a_saved_token_nothing_pairs() -> None:
    hub = ExtensionHub(lambda: None)

    async def _serve(ws: ServerConnection) -> None:
        await hub.serve(ws)

    async with serve(_serve, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        ws, reply = await _pair(f"ws://127.0.0.1:{port}", token="")

    assert reply["type"] == "refused"
    await ws.close()


async def test_r1_link_routes_commands_and_events_by_tab(
    hub_url: tuple[ExtensionHub, str],
) -> None:
    hub, url = hub_url
    ws, _ = await _pair(url)
    extension = FakeExtension(ws)
    await hub.wait_connected(5)
    link = hub.link()
    assert link is not None
    assert isinstance(link, BrowserLink)

    tab = await link.open_tab("about:blank")
    events = link.subscribe(tab)
    result = await link.send(tab, "Runtime.evaluate", {"expression": "1"})

    assert tab == "101"
    assert result == {"result": {"value": "tab 101"}}
    event = await asyncio.wait_for(events.get(), 5)
    assert event.method == "Page.loadEventFired"
    opened = extension.received[0]
    assert (opened["method"], opened["params"]) == ("UClone.openTab", {"url": "about:blank"})
    assert extension.received[1]["sessionId"] == "101"
    await extension.stop()


async def test_closing_r1_link_releases_its_tabs_and_keeps_the_extension(
    hub_url: tuple[ExtensionHub, str],
) -> None:
    hub, url = hub_url
    ws, _ = await _pair(url)
    extension = FakeExtension(ws)
    await hub.wait_connected(5)
    link = hub.link()
    assert link is not None
    first = await link.open_tab("about:blank")
    second = await link.open_tab("about:blank")
    await link.close_tab(second)

    await link.close()

    assert extension.methods()[-2:] == ["UClone.closeTab", "UClone.releaseTab"]
    assert extension.received[-1]["params"] == {"tabId": first}
    assert hub.status()["connected"] is True
    await extension.stop()


async def test_a_new_extension_connection_replaces_the_old_one(
    hub_url: tuple[ExtensionHub, str],
) -> None:
    hub, url = hub_url
    old_ws, _ = await _pair(url)
    old = FakeExtension(old_ws)
    await hub.wait_connected(5)
    old_link = hub.link()
    assert old_link is not None
    tab = await old_link.open_tab("about:blank")

    new_ws, reply = await _pair(url)
    new = FakeExtension(new_ws)

    assert reply == {"type": "paired"}
    await asyncio.wait_for(old.task, 5)  # the hub closed the old socket
    with pytest.raises(CdpClosedError):
        await old_link.send(tab, "Runtime.evaluate")
    current = hub.link()
    assert current is not None and current is not old_link
    assert await current.open_tab("about:blank") == "101"
    await new.stop()


async def test_disconnect_drops_the_extension(hub_url: tuple[ExtensionHub, str]) -> None:
    hub, url = hub_url
    ws, _ = await _pair(url)
    extension = FakeExtension(ws)
    await hub.wait_connected(5)

    await hub.disconnect()

    await asyncio.wait_for(extension.task, 5)
    assert hub.link() is None
    assert hub.status()["connected"] is False


# -- the route choice ------------------------------------------------------------------


class FakeR2:
    """A stand-in for the launched Chrome: counts what it was asked to do."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.adopted: list[tuple[str, str]] = []
        self.closed = False
        self.next_tab = 0

    async def open_tab(self, url: str) -> str:
        self.next_tab += 1
        self.opened.append(url)
        return f"r2-{self.next_tab}"

    async def send(self, tab: str, method: str, params: Any = None) -> dict[str, Any]:
        return {"route": "r2", "tab": tab}

    def subscribe(self, tab: str, *, frames: bool = False) -> asyncio.Queue[Any]:
        return asyncio.Queue()

    def unsubscribe(self, tab: str, queue: asyncio.Queue[Any]) -> None:
        pass

    async def close_tab(self, tab: str) -> None:
        pass

    async def adopt_tab(self, opener: str, tab: str) -> str:
        self.adopted.append((opener, tab))
        return tab

    async def close(self) -> None:
        self.closed = True


async def test_without_an_extension_tabs_open_in_the_launched_chrome() -> None:
    hub = ExtensionHub(lambda: TOKEN)
    r2 = FakeR2()
    starts = 0

    async def start() -> FakeR2:
        nonlocal starts
        starts += 1
        return r2

    link = RoutedLink(hub, start)
    tab = await link.open_tab("about:blank")
    again = await link.open_tab("about:blank")

    assert (tab, again) == ("r2-1", "r2-2")
    assert starts == 1
    assert await link.send(tab, "Page.enable") == {"route": "r2", "tab": "r2-1"}
    await link.close()
    assert r2.closed


async def test_a_paired_extension_gets_the_new_tabs_and_r2_is_never_started(
    hub_url: tuple[ExtensionHub, str],
) -> None:
    hub, url = hub_url
    ws, _ = await _pair(url)
    extension = FakeExtension(ws)
    await hub.wait_connected(5)

    async def start() -> FakeR2:
        raise AssertionError("R2 must not start while the extension is paired")

    link = RoutedLink(hub, start)
    tab = await link.open_tab("about:blank")
    result = await link.send(tab, "Runtime.evaluate")

    assert tab == "101"
    assert result == {"result": {"value": "tab 101"}}
    await link.close()
    await extension.stop()


async def test_a_tab_whose_extension_left_reads_as_a_closed_tab(
    hub_url: tuple[ExtensionHub, str],
) -> None:
    hub, url = hub_url
    ws, _ = await _pair(url)
    extension = FakeExtension(ws)
    await hub.wait_connected(5)
    r2 = FakeR2()

    async def start() -> FakeR2:
        return r2

    link = RoutedLink(hub, start)
    tab = await link.open_tab("about:blank")
    await hub.disconnect()
    await asyncio.wait_for(extension.task, 5)

    # The service treats KeyError as "U0 closed the tab" and opens a fresh one.
    with pytest.raises(KeyError):
        await link.send(tab, "Page.navigate", {"url": "https://example.com/"})
    assert await link.open_tab("about:blank") == "r2-1"


async def test_r1_and_r2_tabs_live_side_by_side(hub_url: tuple[ExtensionHub, str]) -> None:
    hub, url = hub_url
    r2 = FakeR2()

    async def start() -> FakeR2:
        return r2

    link = RoutedLink(hub, start)
    before = await link.open_tab("about:blank")
    ws, _ = await _pair(url)
    extension = FakeExtension(ws)
    await hub.wait_connected(5)
    after = await link.open_tab("about:blank")

    assert await link.send(before, "Page.enable") == {"route": "r2", "tab": "r2-1"}
    assert await link.send(after, "Runtime.evaluate") == {"result": {"value": "tab 101"}}
    await extension.stop()


async def test_a_pop_up_from_an_r2_tab_is_adopted_on_r2_and_routed_there() -> None:
    hub = ExtensionHub(lambda: TOKEN)
    r2 = FakeR2()

    async def start() -> FakeR2:
        return r2

    link = RoutedLink(hub, start)
    opener = await link.open_tab("about:blank")

    popup = await link.adopt_tab(opener, "T-POPUP")

    assert popup == "T-POPUP"
    assert r2.adopted == [(opener, "T-POPUP")]
    assert await link.send(popup, "Page.enable") == {"route": "r2", "tab": "T-POPUP"}


async def test_a_pop_up_in_u0s_chrome_is_refused_in_plain_words(
    hub_url: tuple[ExtensionHub, str],
) -> None:
    # The extension relay attaches only to tabs it opened itself (design §3.10, step 2).
    hub, url = hub_url
    ws, _ = await _pair(url)
    extension = FakeExtension(ws)
    await hub.wait_connected(5)

    async def start() -> FakeR2:
        raise AssertionError("R2 must not start for an R1 pop-up")

    link = RoutedLink(hub, start)
    opener = await link.open_tab("about:blank")
    sent = len(extension.received)

    with pytest.raises(PlainRefusalError) as refused:
        await link.adopt_tab(opener, "T-POPUP")

    assert str(refused.value) == POPUP_NOT_FOLLOWED
    assert refused.value.reason_code == "popup_not_followed"
    for internal in ("CDP", "target", "Target", "debugger", "R1", "Error"):
        assert internal not in str(refused.value)
    assert len(extension.received) == sent  # nothing was sent to the extension
    await extension.stop()


async def test_adopting_from_an_unknown_opener_reads_as_a_closed_tab() -> None:
    link = RoutedLink(ExtensionHub(lambda: TOKEN), FakeR2Factory())
    with pytest.raises(KeyError):
        await link.adopt_tab("gone", "T-POPUP")


class FakeR2Factory:
    async def __call__(self) -> FakeR2:
        return FakeR2()


# -- the pairing token -------------------------------------------------------------------


def test_the_token_is_made_once_and_kept(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"llm_provider": "openai"}), encoding="utf-8")

    first = pairing_token(settings)
    second = pairing_token(settings)

    assert first == second
    assert len(first) == 32 and int(first, 16) >= 0
    saved = json.loads(settings.read_text(encoding="utf-8"))
    assert saved["llm_provider"] == "openai"


def test_a_new_token_replaces_the_old(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    old = pairing_token(settings)

    new = regenerate_pairing_token(settings)

    assert new != old
    assert pairing_token(settings) == new


def test_the_extension_folder_ships_a_loadable_manifest() -> None:
    manifest = json.loads((EXTENSION_FOLDER / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == 3
    assert "debugger" in manifest["permissions"]
    assert (EXTENSION_FOLDER / manifest["background"]["service_worker"]).is_file()
    assert (EXTENSION_FOLDER / "options.html").is_file()


def test_the_extension_messages_are_translated_alike() -> None:
    en = json.loads((EXTENSION_FOLDER / "_locales/en/messages.json").read_text(encoding="utf-8"))
    ko = json.loads((EXTENSION_FOLDER / "_locales/ko/messages.json").read_text(encoding="utf-8"))

    assert set(en) == set(ko)


# -- the routes --------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    settings = tmp_path / "settings.json"
    hub = ExtensionHub(lambda: pairing_token(settings))

    def refuse(request: Any) -> None:
        origin = request.headers.get("origin")
        if origin and "127.0.0.1" not in origin:
            from fastapi import HTTPException

            raise HTTPException(status_code=403)

    register_browser_routes(app, hub=hub, settings_file=settings, refuse_cross_origin=refuse)
    app.state.hub = hub
    return TestClient(app)


def test_status_shows_a_pairing_code_with_the_port(client: TestClient) -> None:
    body = client.get("/api/browser/extension").json()

    assert body["connected"] is False
    port, token = body["pairing_code"].split("-")
    assert port == "80"  # TestClient's server port
    assert len(token) == 32
    assert body["extension_folder"] == str(EXTENSION_FOLDER)


def test_a_new_pairing_code_replaces_the_shown_one(client: TestClient) -> None:
    before = client.get("/api/browser/extension").json()["pairing_code"]

    after = client.post("/api/browser/extension/pairing-code").json()["pairing_code"]

    assert after != before
    assert client.get("/api/browser/extension").json()["pairing_code"] == after


def test_a_web_page_cannot_read_the_pairing_code(client: TestClient) -> None:
    response = client.get("/api/browser/extension", headers={"origin": "https://evil.example"})

    assert response.status_code == 403


def test_the_link_accepts_only_the_extension_origin(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/api/browser/extension", headers={"origin": "https://evil.example"}
        ) as ws:
            ws.receive_text()


def test_the_extension_pairs_over_the_link(client: TestClient) -> None:
    code = client.get("/api/browser/extension").json()["pairing_code"]
    token = code.split("-", 1)[1]

    with client.websocket_connect(
        "/api/browser/extension", headers={"origin": "chrome-extension://abcdef"}
    ) as ws:
        ws.send_text(json.dumps({"type": "hello", "token": token, "version": "0.1.0"}))
        assert json.loads(ws.receive_text()) == {"type": "paired"}
        assert client.get("/api/browser/extension").json()["connected"] is True


def test_a_stale_code_is_refused_over_the_link(client: TestClient) -> None:
    with client.websocket_connect(
        "/api/browser/extension", headers={"origin": "chrome-extension://abcdef"}
    ) as ws:
        ws.send_text(json.dumps({"type": "hello", "token": "0" * 32}))
        reply = json.loads(ws.receive_text())

    assert reply == {"type": "refused", "reason": "pairing_code", "message": REFUSED_MESSAGE}


def test_the_dashboard_serves_the_pairing_code_from_its_settings_file(tmp_path: Path) -> None:
    from uclone_x.browser.extension import default_extension_hub
    from uclone_x.llm.connectors.mock import MockLLMConnector
    from uclone_x.ui.app import create_ui_app

    static_dir = tmp_path / "ui_static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    storage_dir = tmp_path / "sessions"
    app = create_ui_app(static_dir=static_dir, storage_dir=storage_dir, llm=MockLLMConnector())
    client = TestClient(app)

    body = client.get("/api/browser/extension").json()

    token = body["pairing_code"].split("-", 1)[1]
    settings = json.loads((storage_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings[TOKEN_KEY] == token
    # The hub checks pairings against that same file.
    assert default_extension_hub()._token() == token  # pyright: ignore[reportPrivateUsage]
