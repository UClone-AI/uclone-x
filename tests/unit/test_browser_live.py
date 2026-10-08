"""The dock's Browser tab, live (design `browser-agent.md` §3.4, §3.6; §6 step 3).

A fake link stands in for Chrome (either link: the view speaks only `BrowserLink`), and a
fake socket for the head. The real-Chrome screencast is in `tests/e2e/test_browser_live_e2e.py`.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from uclone_x.browser import live
from uclone_x.browser.cdp import SCREENCAST_FRAME, CdpConnection, CdpEvent
from uclone_x.browser.extension import ExtensionHub, R1Link, RoutedLink
from uclone_x.browser.live import LiveView
from uclone_x.browser.service import (
    BrowserService,
    Notice,
    TabKey,
    _Tab,  # pyright: ignore[reportPrivateUsage]
    _Window,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.browser.tool import BrowserParams, BrowserTool
from uclone_x.errors import PlainRefusalError
from uclone_x.tools.models import ToolContext
from uclone_x.ui.browser import register_browser_routes

KEY: TabKey = ("room-1", "scout")
JPEG = b"\xff\xd8\xff\xe0fake-jpeg"
_INTERNALS = ("CDP", "Error", "Traceback", "backendNode", "objectId", "Page.", "net::")


class FakeLink:
    """A `BrowserLink` that records commands and lets a test push a tab's events."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, dict[str, Any]]] = []
        self.queues: dict[str, list[tuple[asyncio.Queue[CdpEvent], bool]]] = {}
        self.screenshot: str | None = None

    async def open_tab(self, url: str) -> str:
        return "t-new"

    async def send(
        self, tab: str, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self.sent.append((tab, method, dict(params or {})))
        if method == "Page.captureScreenshot":
            return {"data": self.screenshot or ""}
        if method == "Page.getLayoutMetrics":
            return {"cssLayoutViewport": {"clientWidth": 800, "clientHeight": 600}}
        return {}

    def subscribe(self, tab: str, *, frames: bool = False) -> asyncio.Queue[CdpEvent]:
        queue: asyncio.Queue[CdpEvent] = asyncio.Queue()
        self.queues.setdefault(tab, []).append((queue, frames))
        return queue

    def unsubscribe(self, tab: str, queue: asyncio.Queue[CdpEvent]) -> None:
        self.queues[tab] = [(q, f) for q, f in self.queues.get(tab, []) if q is not queue]

    async def adopt_tab(self, opener: str, tab: str) -> str:
        return tab

    async def close_tab(self, tab: str) -> None:
        pass

    async def close(self) -> None:
        pass

    def frame(self, tab: str, data: bytes = JPEG, width: int = 1024, height: int = 768) -> None:
        event = CdpEvent(
            method=SCREENCAST_FRAME,
            params={
                "data": base64.b64encode(data).decode(),
                "sessionId": 7,
                "metadata": {"deviceWidth": width, "deviceHeight": height},
            },
            session_id=tab,
        )
        for queue, frames in self.queues.get(tab, []):
            if frames:
                queue.put_nowait(event)

    def methods(self, tab: str | None = None) -> list[str]:
        return [m for t, m, _ in self.sent if tab is None or t == tab]


class FakeSocket:
    """The head: what it was sent, and what it sends."""

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

    async def until(self, check: Any, timeout: float = 2.0) -> None:
        async def wait() -> None:
            while not check():
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait(), timeout)


def _service(link: FakeLink, *tabs: tuple[str, str, str], key: TabKey = KEY) -> BrowserService:
    async def factory() -> Any:
        return link

    service = BrowserService(factory)
    service._link = link  # pyright: ignore[reportPrivateUsage]
    if tabs:
        service._windows[key] = _Window(  # pyright: ignore[reportPrivateUsage]
            [_Tab(target=t, events=asyncio.Queue(), url=u, title=title) for t, u, title in tabs]
        )
    return service


@pytest.fixture
async def running() -> AsyncIterator[tuple[FakeLink, BrowserService, FakeSocket]]:
    link = FakeLink()
    service = _service(link, ("t1", "https://example.com/", "Example"))
    socket = FakeSocket()
    task = asyncio.create_task(LiveView(service, "room-1", socket).run())
    await socket.until(lambda: socket.of("state"))
    yield link, service, socket
    socket.inbox.put_nowait(None)
    await asyncio.wait_for(task, 2)


# -- the service's notices ---------------------------------------------------------------


async def test_a_call_is_announced_while_it_runs_and_reported_when_it_ends() -> None:
    link = FakeLink()
    service = _service(link, ("t1", "https://example.com/", "Example"))
    notices = service.watch("room-1")

    async with service.acting(KEY, "click") as step:
        step["element"] = "Search"
        assert service.live_state("room-1")["acting"] == {"clone": "scout", "action": "click"}

    assert service.live_state("room-1")["acting"] is None
    seen = [notices.get_nowait() for _ in range(notices.qsize())]
    assert seen == [
        {"type": "state"},
        {"type": "step", "clone": "scout", "action": "click", "ok": True, "element": "Search"},
        {"type": "state"},
    ]


async def test_another_conversation_hears_nothing() -> None:
    service = _service(FakeLink(), ("t1", "", ""))
    elsewhere = service.watch("room-2")

    async with service.acting(KEY, "open"):
        pass

    assert elsewhere.empty()
    assert service.live_state("room-2")["tabs"] == []


async def test_a_missing_chrome_becomes_a_problem_code_until_a_call_works() -> None:
    service = _service(FakeLink())
    notices = service.watch("room-1")

    with pytest.raises(PlainRefusalError):
        async with service.acting(KEY, "open"):
            raise PlainRefusalError("Google Chrome is not installed.", reason_code="no_chrome")

    assert service.live_state("room-1")["problem"] == "chrome_missing"
    steps = [
        n for n in (notices.get_nowait() for _ in range(notices.qsize())) if n["type"] == "step"
    ]
    assert steps == [{"type": "step", "clone": "scout", "action": "open", "ok": False}]

    async with service.acting(KEY, "open"):
        pass
    assert service.live_state("room-1")["problem"] is None


async def test_an_ordinary_refusal_is_not_a_browser_problem() -> None:
    service = _service(FakeLink(), ("t1", "", ""))
    with pytest.raises(PlainRefusalError):
        async with service.acting(KEY, "click"):
            raise PlainRefusalError("That ref is stale.", reason_code="stale_ref")
    assert service.live_state("room-1")["problem"] is None


async def test_the_state_lists_tabs_as_data_not_copy() -> None:
    service = _service(
        FakeLink(), ("t1", "https://a.example/", "A"), ("t2", "https://b.example/", "B")
    )
    service._windows[KEY].active = 1  # pyright: ignore[reportPrivateUsage]

    state = service.live_state("room-1")

    assert state == {
        "type": "state",
        "tabs": [
            {
                "clone": "scout",
                "index": 1,
                "url": "https://a.example/",
                "title": "A",
                "current": False,
                "controller": "clone",
            },
            {
                "clone": "scout",
                "index": 2,
                "url": "https://b.example/",
                "title": "B",
                "current": True,
                "controller": "clone",
            },
        ],
        "acting": None,
        "problem": None,
    }


async def test_an_element_is_named_by_the_snapshot_and_shown_in_the_overlay() -> None:
    link = FakeLink()
    service = _service(link, ("t1", "", ""))
    tab = service._windows[KEY].tab  # pyright: ignore[reportPrivateUsage]
    ref = tab.refs.ref_for(41, "Search")
    notices = service.watch("room-1")

    async def frame_tree(
        t: str, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if method == "Page.getFrameTree":
            return {"frameTree": {"frame": {"id": "f", "loaderId": tab.loader}}}
        if method == "DOM.resolveNode":
            return {"object": {"objectId": "obj-1"}}
        if method == "Runtime.callFunctionOn":
            return {"result": {"value": {"reason": "", "x": 15, "y": 25, "box": [10, 20, 10, 10]}}}
        return {}

    link.send = frame_tree  # type: ignore[method-assign]
    async with service.acting(KEY, "click") as step:
        backend, element = await service._element(link, tab, ref)  # pyright: ignore[reportPrivateUsage]
        _, box = await service._point(link, tab, element, backend)  # pyright: ignore[reportPrivateUsage]
        await service.broadcast_overlay("room-1", box, step["element"], tab=tab, clone="scout")
        assert step["element"] == "Search"

    overlay = [
        n for n in (notices.get_nowait() for _ in range(notices.qsize())) if n["type"] == "overlay"
    ]
    assert overlay == [
        {
            "type": "overlay",
            "clone": "scout",
            "target": "t1",
            "box": {"x": 10.0, "y": 20.0, "width": 10.0, "height": 10.0},
            "label": "Search",
            "action": "click",
        }
    ]


class _Clicking(BrowserService):
    def __init__(self) -> None:
        async def factory() -> Any:
            raise AssertionError("no Chrome in this test")

        super().__init__(factory)

    async def click(self, key: TabKey, ref: str, **flags: bool) -> dict[str, Any]:
        self._steps[key]["element"] = "Search"  # pyright: ignore[reportPrivateUsage]
        return {"url": "https://example.com/", "changes": '+ text "3 results"'}


async def test_the_tool_result_leads_with_the_element_it_touched() -> None:
    service = _Clicking()
    notices = service.watch("room-1")
    result = await BrowserTool(service=service).run(
        BrowserParams(action="click", ref="e1"),
        ToolContext(agent_id="scout", session_id="s-1", room_id="room-1"),
    )

    assert isinstance(result, dict)
    assert list(result)[0] == "element"
    assert result["element"] == "Search"
    assert {"type": "state"} in [notices.get_nowait() for _ in range(notices.qsize())]


# -- the live view -----------------------------------------------------------------------


async def test_the_head_is_told_the_state_at_once_with_the_tab_shown(
    running: tuple[FakeLink, BrowserService, FakeSocket],
) -> None:
    link, _, socket = running
    state = socket.of("state")[0]

    assert state["shown"] == {"clone": "scout", "index": 1}
    assert state["extension"] is False
    assert state["acting"] is None
    # Nothing is screencast until the head says the tab is on screen.
    assert "Page.startScreencast" not in link.methods()
    for internal in _INTERNALS:
        assert internal not in json.dumps(state)


async def test_frames_are_relayed_as_jpeg_and_acknowledged_while_the_head_looks(
    running: tuple[FakeLink, BrowserService, FakeSocket],
) -> None:
    link, _, socket = running
    socket.inbox.put_nowait({"type": "view", "on": True, "width": 600})
    await socket.until(lambda: "Page.startScreencast" in link.methods("t1"))
    start = next(p for _, m, p in link.sent if m == "Page.startScreencast")
    assert start["format"] == "jpeg"
    assert start["maxWidth"] == 600

    link.frame("t1")
    await socket.until(lambda: socket.binary)

    assert socket.binary == [JPEG]
    assert socket.of("frame") == [{"type": "frame", "width": 1024, "height": 768}]
    await socket.until(lambda: "Page.screencastFrameAck" in link.methods("t1"))
    ack = next(p for _, m, p in link.sent if m == "Page.screencastFrameAck")
    assert ack == {"sessionId": 7}

    socket.inbox.put_nowait({"type": "view", "on": False})
    await socket.until(lambda: "Page.stopScreencast" in link.methods("t1"))
    assert [q for q, frames in link.queues["t1"] if frames] == []


async def test_a_tab_that_sends_no_frames_is_shown_by_stills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live, "STILL_AFTER_S", 0.01)
    link = FakeLink()
    link.screenshot = base64.b64encode(JPEG).decode()
    service = _service(link, ("t1", "", ""))
    socket = FakeSocket()
    task = asyncio.create_task(LiveView(service, "room-1", socket).run())
    socket.inbox.put_nowait({"type": "view", "on": True})

    await socket.until(lambda: socket.binary)

    assert socket.binary[0] == JPEG
    assert socket.of("frame")[0] == {"type": "frame", "width": 800, "height": 600}
    socket.inbox.put_nowait(None)
    await asyncio.wait_for(task, 2)


async def test_the_acting_clones_tab_is_followed(
    running: tuple[FakeLink, BrowserService, FakeSocket],
) -> None:
    link, service, socket = running
    service._windows[("room-1", "ada")] = _Window(  # pyright: ignore[reportPrivateUsage]
        [_Tab(target="t9", events=asyncio.Queue(), url="https://ada.example/")]
    )
    socket.inbox.put_nowait({"type": "view", "on": True})
    await socket.until(lambda: "Page.startScreencast" in link.methods("t1"))

    async with service.acting(("room-1", "ada"), "click"):
        await socket.until(lambda: "Page.startScreencast" in link.methods("t9"))
        assert socket.of("state")[-1]["shown"] == {"clone": "ada", "index": 1}
        assert socket.of("state")[-1]["acting"] == {"clone": "ada", "action": "click"}
    assert "Page.stopScreencast" in link.methods("t1")


async def test_the_head_can_pick_another_tab_to_watch(
    running: tuple[FakeLink, BrowserService, FakeSocket],
) -> None:
    link, service, socket = running
    service._windows[KEY].tabs.append(  # pyright: ignore[reportPrivateUsage]
        _Tab(target="t2", events=asyncio.Queue(), url="https://b.example/")
    )
    socket.inbox.put_nowait({"type": "view", "on": True})
    socket.inbox.put_nowait({"type": "tab", "clone": "scout", "index": 2})

    await socket.until(lambda: "Page.startScreencast" in link.methods("t2"))
    assert socket.of("state")[-1]["shown"] == {"clone": "scout", "index": 2}


async def test_overlays_of_the_shown_tab_and_every_step_reach_the_head(
    running: tuple[FakeLink, BrowserService, FakeSocket],
) -> None:
    link, service, socket = running
    socket.inbox.put_nowait({"type": "view", "on": True})
    await socket.until(lambda: "Page.startScreencast" in link.methods("t1"))
    tab = service._windows[KEY].tab  # pyright: ignore[reportPrivateUsage]

    async with service.acting(KEY, "click") as step:
        step["element"] = "Search"
        box = {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}
        await service.broadcast_overlay("room-1", box, "Search", tab=tab, clone="scout")
        other = _Tab(target="elsewhere", events=asyncio.Queue())
        await service.broadcast_overlay("room-1", box, "Elsewhere", tab=other, clone="scout")

    await socket.until(lambda: socket.of("step"))
    assert socket.of("overlay") == [
        {
            "type": "overlay",
            "clone": "scout",
            "box": {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
            "label": "Search",
            "action": "click",
        }
    ]
    assert socket.of("step") == [
        {"type": "step", "clone": "scout", "action": "click", "element": "Search", "ok": True}
    ]


async def test_input_reaches_the_page_only_while_u0_has_taken_over(
    running: tuple[FakeLink, BrowserService, FakeSocket],
) -> None:
    link, _, socket = running
    click = {"type": "input", "event": "mouse", "x": 5, "y": 6}
    socket.inbox.put_nowait(click)
    socket.inbox.put_nowait({"type": "take_over"})
    await socket.until(lambda: socket.of("state")[-1]["tabs"][0]["controller"] == "user")
    socket.inbox.put_nowait(click)
    socket.inbox.put_nowait({"type": "input", "event": "text", "text": "hi"})
    await socket.until(lambda: "Input.insertText" in link.methods("t1"))
    assert link.methods("t1").count("Input.dispatchMouseEvent") == 1

    socket.inbox.put_nowait({"type": "give_back"})
    await socket.until(lambda: socket.of("state")[-1]["tabs"][0]["controller"] == "clone")
    socket.inbox.put_nowait({"type": "input", "event": "key", "key": "a"})
    socket.inbox.put_nowait({"type": "stop"})
    await socket.until(lambda: socket.of("state")[-1]["tabs"][0]["controller"] == "user")
    assert "Input.dispatchKeyEvent" not in link.methods("t1")


async def test_messages_the_view_does_not_know_are_ignored(
    running: tuple[FakeLink, BrowserService, FakeSocket],
) -> None:
    link, _, socket = running
    before = len(socket.json)
    socket.inbox.put_nowait({"type": "navigate", "url": "https://example.com/"})
    socket.inbox.put_nowait({"type": "input", "event": "mouse"})
    socket.inbox.put_nowait({"type": "view", "on": False})
    await socket.until(lambda: len(socket.json) > before)
    assert link.methods() == []


async def test_leaving_stops_watching_the_conversation() -> None:
    link = FakeLink()
    service = _service(link, ("t1", "", ""))
    socket = FakeSocket()
    task = asyncio.create_task(LiveView(service, "room-1", socket).run())
    socket.inbox.put_nowait({"type": "view", "on": True})
    await socket.until(lambda: "Page.startScreencast" in link.methods())

    socket.inbox.put_nowait(None)
    await asyncio.wait_for(task, 2)

    assert service._watchers == {}  # pyright: ignore[reportPrivateUsage]
    assert "Page.stopScreencast" in link.methods()


# -- frames stay out of a tab's own queue ------------------------------------------------


class _IdleTransport:
    async def send(self, message: str) -> None:
        pass

    def __aiter__(self) -> AsyncIterator[str]:
        return self._never()

    async def _never(self) -> AsyncIterator[str]:
        await asyncio.Event().wait()
        yield ""

    async def close(self) -> None:
        pass


async def test_screencast_frames_reach_only_queues_that_asked_for_them() -> None:
    conn = CdpConnection(_IdleTransport())
    plain = conn.subscribe("s1")
    frames = conn.subscribe("s1", frames=True)

    conn._dispatch(  # pyright: ignore[reportPrivateUsage]
        json.dumps({"method": SCREENCAST_FRAME, "params": {"data": ""}, "sessionId": "s1"})
    )
    conn._dispatch(  # pyright: ignore[reportPrivateUsage]
        json.dumps({"method": "Page.loadEventFired", "params": {}, "sessionId": "s1"})
    )

    assert [e.method for e in [plain.get_nowait()]] == ["Page.loadEventFired"]
    assert plain.empty()
    assert [frames.get_nowait().method, frames.get_nowait().method] == [
        SCREENCAST_FRAME,
        "Page.loadEventFired",
    ]
    await conn.close()


# -- the route ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    link = FakeLink()
    service = _service(link, ("t1", "https://example.com/", "Example"))

    def refuse(request: Any) -> None:
        origin = request.headers.get("origin")
        if origin and "127.0.0.1" not in origin:
            raise HTTPException(status_code=403)

    register_browser_routes(
        app,
        hub=ExtensionHub(lambda: None),
        settings_file=tmp_path / "settings.json",
        refuse_cross_origin=refuse,
        service=service,
    )
    return TestClient(app)


def test_the_live_view_refuses_a_page_of_another_site(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect(
            "/api/browser/room-1", headers={"origin": "https://evil.example"}
        ) as ws:
            ws.receive_text()
    assert refused.value.code == 1008


def test_the_live_view_tells_the_dashboard_the_conversation_state(client: TestClient) -> None:
    with client.websocket_connect(
        "/api/browser/room-1", headers={"origin": "http://127.0.0.1:5180"}
    ) as ws:
        state: Notice = json.loads(ws.receive_text())

    assert state["type"] == "state"
    assert state["tabs"][0]["url"] == "https://example.com/"
    assert state["shown"] == {"clone": "scout", "index": 1}
    assert state["extension"] is False


def test_the_extension_route_is_not_taken_for_a_conversation(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/api/browser/extension", headers={"origin": "http://127.0.0.1:5180"}
        ) as ws:
            ws.receive_text()


async def test_dock_resize_restarts_screencast_with_new_width() -> None:
    link = FakeLink()
    service = _service(link, ("t1", "https://example.com/", "Example"))
    socket = FakeSocket()
    view = LiveView(service, "room-1", socket)
    task = asyncio.create_task(view.run())
    try:
        await socket.until(lambda: socket.of("state"))
        socket.inbox.put_nowait({"type": "view", "on": True, "width": 640})
        await socket.until(lambda: any(m == "Page.startScreencast" for _, m, _ in link.sent))
        first_start = next(p for _, m, p in link.sent if m == "Page.startScreencast")
        assert first_start["maxWidth"] == 640

        socket.inbox.put_nowait({"type": "view", "on": True, "width": 800})
        await socket.until(
            lambda: len([p for _, m, p in link.sent if m == "Page.startScreencast"]) >= 2
        )
        starts = [p for _, m, p in link.sent if m == "Page.startScreencast"]
        assert starts[-1]["maxWidth"] == 800
    finally:
        socket.inbox.put_nowait(None)
        await asyncio.wait_for(task, 2)


async def test_two_live_views_share_screencast_without_killing_the_survivor() -> None:
    link = FakeLink()
    service = _service(link, ("t1", "https://example.com/", "Example"))
    socket1 = FakeSocket()
    socket2 = FakeSocket()
    view1 = LiveView(service, "room-1", socket1)
    view2 = LiveView(service, "room-1", socket2)
    task1 = asyncio.create_task(view1.run())
    task2 = asyncio.create_task(view2.run())
    try:
        await socket1.until(lambda: socket1.of("state"))
        await socket2.until(lambda: socket2.of("state"))

        socket1.inbox.put_nowait({"type": "view", "on": True, "width": 640})
        await socket1.until(lambda: any(m == "Page.startScreencast" for _, m, _ in link.sent))
        assert service._screencast_subscribers.get("t1") == 1  # pyright: ignore[reportPrivateUsage]

        socket2.inbox.put_nowait({"type": "view", "on": True, "width": 640})
        await socket2.until(
            lambda: service._screencast_subscribers.get("t1") == 2  # pyright: ignore[reportPrivateUsage]
        )

        socket1.inbox.put_nowait(None)
        await asyncio.wait_for(task1, 2)
        assert service._screencast_subscribers.get("t1") == 1  # pyright: ignore[reportPrivateUsage]
        assert "Page.stopScreencast" not in link.methods("t1")

        socket2.inbox.put_nowait(None)
        await asyncio.wait_for(task2, 2)
        assert service._screencast_subscribers.get("t1", 0) == 0  # pyright: ignore[reportPrivateUsage]
        assert "Page.stopScreencast" in link.methods("t1")
    finally:
        if not task2.done():
            socket2.inbox.put_nowait(None)
            await task2


async def test_screencast_frames_reach_r1_and_routed_link_subscribers() -> None:
    conn = CdpConnection(_IdleTransport())
    r1 = R1Link(conn)

    plain_q = r1.subscribe("t1")
    frame_q = r1.subscribe("t1", frames=True)

    conn._dispatch(  # pyright: ignore[reportPrivateUsage]
        json.dumps({"method": SCREENCAST_FRAME, "params": {"data": "jpeg-data"}, "sessionId": "t1"})
    )

    assert plain_q.empty()
    frame_event = frame_q.get_nowait()
    assert frame_event.method == SCREENCAST_FRAME
    assert frame_event.params["data"] == "jpeg-data"

    hub = ExtensionHub(lambda: "token")
    routed = RoutedLink(hub, AsyncMock())
    routed._routes["t1"] = r1  # pyright: ignore[reportPrivateUsage]

    routed_plain = routed.subscribe("t1")
    routed_frame = routed.subscribe("t1", frames=True)

    conn._dispatch(  # pyright: ignore[reportPrivateUsage]
        json.dumps(
            {"method": SCREENCAST_FRAME, "params": {"data": "jpeg-data-2"}, "sessionId": "t1"}
        )
    )

    assert routed_plain.empty()
    frame_event_2 = routed_frame.get_nowait()
    assert frame_event_2.method == SCREENCAST_FRAME
    assert frame_event_2.params["data"] == "jpeg-data-2"

    await conn.close()
