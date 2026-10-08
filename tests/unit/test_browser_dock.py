"""Dock ▸ Browser tab WebSocket route (/api/browser/{conversation_id}), served by `LiveView`."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from uclone_x.browser.cdp import CdpEvent
from uclone_x.browser.extension import ExtensionHub
from uclone_x.browser.link import BrowserLink
from uclone_x.browser.service import BrowserService, BrowserTab, BrowserWindow
from uclone_x.ui.browser import register_browser_routes


class FakeLink:
    """Fake BrowserLink that records commands and simulates CDP screencast frames."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, dict[str, Any]]] = []
        self._queues: dict[str, list[asyncio.Queue[CdpEvent]]] = {}

    async def open_tab(self, url: str) -> str:
        return "tab-1"

    async def send(
        self, tab: str, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self.sent.append((tab, method, dict(params) if params is not None else {}))
        return {}

    def subscribe(self, tab: str, *, frames: bool = False) -> asyncio.Queue[CdpEvent]:
        q: asyncio.Queue[CdpEvent] = asyncio.Queue()
        self._queues.setdefault(tab, []).append(q)
        return q

    def unsubscribe(self, tab: str, queue: asyncio.Queue[CdpEvent]) -> None:
        if tab in self._queues and queue in self._queues[tab]:
            self._queues[tab].remove(queue)

    def emit(self, tab: str, event: CdpEvent) -> None:
        for q in self._queues.get(tab, []):
            q.put_nowait(event)

    async def adopt_tab(self, opener: str, tab: str) -> str:
        return tab

    async def close_tab(self, tab: str) -> None:
        pass

    async def close(self) -> None:
        pass


@pytest.fixture
def fake_link() -> FakeLink:
    return FakeLink()


@pytest.fixture
def browser_service(fake_link: FakeLink) -> BrowserService:
    async def _factory() -> BrowserLink:
        return fake_link

    service = BrowserService(_factory)
    service._link = fake_link  # pyright: ignore[reportPrivateUsage]
    return service


@pytest.fixture
def test_app(tmp_path: Path, browser_service: BrowserService) -> FastAPI:
    app = FastAPI()
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    hub = ExtensionHub(lambda: "test-token")

    def refuse(request: Any) -> None:
        origin = request.headers.get("origin")
        if origin and "127.0.0.1" not in origin and "localhost" not in origin:
            raise HTTPException(status_code=403)

    register_browser_routes(
        app,
        hub=hub,
        settings_file=settings,
        refuse_cross_origin=refuse,
        service=browser_service,
    )
    return app


def test_cross_origin_websocket_is_closed(test_app: FastAPI) -> None:
    client = TestClient(test_app)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/api/browser/room-1", headers={"origin": "https://malicious-site.example"}
        ) as ws:
            ws.receive_text()
    assert exc.value.code == 1008


def _receive_state(ws: Any) -> dict[str, Any]:
    while True:
        message = json.loads(ws.receive_text())
        if message["type"] == "state":
            return message


def _until(check: Any) -> None:
    for _ in range(200):
        if check():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met")


def test_an_empty_conversation_is_told_codes_not_copy(test_app: FastAPI) -> None:
    client = TestClient(test_app)
    with client.websocket_connect("/api/browser/room-empty") as ws:
        msg = json.loads(ws.receive_text())
    assert msg == {
        "type": "state",
        "tabs": [],
        "acting": None,
        "problem": None,
        "shown": None,
        "extension": False,
    }


def test_take_over_give_back_and_tab_pick(
    test_app: FastAPI, browser_service: BrowserService, fake_link: FakeLink
) -> None:
    tab1 = BrowserTab(target="tab-1", events=asyncio.Queue(), url="https://example.com")
    tab2 = BrowserTab(target="tab-2", events=asyncio.Queue(), url="https://example.org")
    window = BrowserWindow(tabs=[tab1, tab2], active=0)
    browser_service._windows[("room-1", "scout")] = window  # pyright: ignore[reportPrivateUsage]

    client = TestClient(test_app)
    with client.websocket_connect("/api/browser/room-1") as ws:
        first = _receive_state(ws)
        assert first["shown"] == {"clone": "scout", "index": 1}
        assert [t["controller"] for t in first["tabs"]] == ["clone", "clone"]

        ws.send_text(json.dumps({"type": "take_over"}))
        assert _receive_state(ws)["tabs"][0]["controller"] == "user"
        assert tab1.controller == "user"

        ws.send_text(json.dumps({"type": "give_back"}))
        assert _receive_state(ws)["tabs"][0]["controller"] == "clone"

        ws.send_text(json.dumps({"type": "tab", "clone": "scout", "index": 2}))
        assert _receive_state(ws)["shown"] == {"clone": "scout", "index": 2}


def test_frames_flow_only_while_the_head_looks_and_are_acknowledged(
    test_app: FastAPI, browser_service: BrowserService, fake_link: FakeLink
) -> None:
    tab = BrowserTab(target="tab-screen", events=asyncio.Queue(), url="https://test.com")
    window = BrowserWindow(tabs=[tab], active=0)
    browser_service._windows[("room-screen", "scout")] = window  # pyright: ignore[reportPrivateUsage]

    client = TestClient(test_app)
    with client.websocket_connect("/api/browser/room-screen") as ws:
        _receive_state(ws)
        assert not any(m == "Page.startScreencast" for _, m, _ in fake_link.sent)

        ws.send_text(json.dumps({"type": "view", "on": True, "width": 640}))
        _receive_state(ws)
        _until(lambda: any(m == "Page.startScreencast" for _, m, _ in fake_link.sent))
        start = next(p for _, m, p in fake_link.sent if m == "Page.startScreencast")
        assert start["maxWidth"] == 640

        fake_jpeg = b"\xff\xd8\xff\xe0testjpeg"
        fake_link.emit(
            "tab-screen",
            CdpEvent(
                method="Page.screencastFrame",
                params={
                    "data": base64.b64encode(fake_jpeg).decode("ascii"),
                    "sessionId": 101,
                    "metadata": {"deviceWidth": 640, "deviceHeight": 480},
                },
                session_id="tab-screen",
            ),
        )
        assert json.loads(ws.receive_text()) == {"type": "frame", "width": 640, "height": 480}
        assert ws.receive_bytes() == fake_jpeg
        _until(
            lambda: any(
                m == "Page.screencastFrameAck" and p.get("sessionId") == 101
                for _, m, p in fake_link.sent
            )
        )


def test_an_overlay_of_the_shown_tab_reaches_the_head(
    test_app: FastAPI, browser_service: BrowserService, fake_link: FakeLink
) -> None:
    tab = BrowserTab(target="tab-o", events=asyncio.Queue(), url="https://test.com")
    browser_service._windows[("room-overlay", "scout")] = BrowserWindow(tabs=[tab])  # pyright: ignore[reportPrivateUsage]

    client = TestClient(test_app)
    with client.websocket_connect("/api/browser/room-overlay") as ws:
        _receive_state(ws)
        ws.send_text(json.dumps({"type": "view", "on": True, "width": 800}))
        _receive_state(ws)
        _until(lambda: any(m == "Page.startScreencast" for _, m, _ in fake_link.sent))
        box = {"x": 10.0, "y": 20.0, "width": 50.0, "height": 30.0}
        browser_service._notify(  # pyright: ignore[reportPrivateUsage]
            "room-overlay",
            {
                "type": "overlay",
                "clone": "scout",
                "target": "tab-o",
                "box": box,
                "label": "Search",
                "action": "click",
            },
        )
        overlay = json.loads(ws.receive_text())
    assert overlay == {
        "type": "overlay",
        "clone": "scout",
        "box": box,
        "label": "Search",
        "action": "click",
    }


def test_browser_routes_default_service_is_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fake_default_service() -> BrowserService:
        nonlocal called
        called = True
        return BrowserService(AsyncMock())

    monkeypatch.setattr("uclone_x.ui.browser.default_browser_service", fake_default_service)

    app = FastAPI()
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    hub = ExtensionHub(lambda: "test-token")

    register_browser_routes(
        app,
        hub=hub,
        settings_file=settings,
        refuse_cross_origin=lambda r: None,
        service=None,
    )
    assert not called


def test_stop_action_over_websocket_invokes_on_stop(
    tmp_path: Path, browser_service: BrowserService
) -> None:
    app = FastAPI()
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    hub = ExtensionHub(lambda: "test-token")
    stopped: list[str] = []

    async def fake_on_stop(room_id: str) -> None:
        stopped.append(room_id)

    register_browser_routes(
        app,
        hub=hub,
        settings_file=settings,
        refuse_cross_origin=lambda r: None,
        service=browser_service,
        on_stop=fake_on_stop,
    )

    client = TestClient(app)
    with client.websocket_connect("/api/browser/room-stop") as ws:
        _receive_state(ws)
        ws.send_text(json.dumps({"type": "stop"}))
        _until(lambda: "room-stop" in stopped)

    assert stopped == ["room-stop"]
