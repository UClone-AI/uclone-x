"""Unit tests for BrowserService: room windows, _point box, last_step, and overlay listeners."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock

import pytest

from uclone_x.browser.cdp import CdpEvent
from uclone_x.browser.link import BrowserLink
from uclone_x.browser.service import (
    BrowserService,
    BrowserTab,
    BrowserWindow,
    TabKey,
    _absorb,  # pyright: ignore[reportPrivateUsage]
    _consume_hand_over,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.browser.snapshot import Entry


class FakeLink:
    """Minimal fake BrowserLink for unit testing BrowserService."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, dict[str, Any]]] = []
        self._queues: dict[str, list[asyncio.Queue[CdpEvent]]] = {}

    async def open_tab(self, url: str) -> str:
        return "tab-1"

    async def send(
        self, tab: str, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        params_dict = dict(params) if params is not None else {}
        self.sent.append((tab, method, params_dict))
        if method == "Runtime.callFunctionOn":
            fn = str(params_dict.get("functionDeclaration", ""))
            if "rectOf" in fn or "hitTest" in fn:
                return {
                    "result": {
                        "value": {
                            "reason": "",
                            "x": 150.0,
                            "y": 250.0,
                            "box": [100.0, 200.0, 100.0, 100.0],
                        }
                    }
                }
            if "selectNodeContents" in fn or "CARET_TO_END" in fn or "IS_FOCUSED" in fn:
                return {"result": {"value": True}}
            return {"result": {"value": {}}}
        if method == "Accessibility.getFullAXTree":
            return {"nodes": []}
        if method == "Page.getNavigationHistory":
            return {"currentIndex": 0, "entries": [{"id": 1, "url": "about:blank"}]}
        return {}

    def subscribe(self, tab: str, *, frames: bool = False) -> asyncio.Queue[CdpEvent]:
        q: asyncio.Queue[CdpEvent] = asyncio.Queue()
        self._queues.setdefault(tab, []).append(q)
        return q

    def unsubscribe(self, tab: str, queue: asyncio.Queue[CdpEvent]) -> None:
        if tab in self._queues and queue in self._queues[tab]:
            self._queues[tab].remove(queue)

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
def service(fake_link: FakeLink) -> BrowserService:
    async def _factory() -> BrowserLink:
        return fake_link

    return BrowserService(_factory)


def test_get_room_window_empty(service: BrowserService) -> None:
    assert service.get_room_window("room-123") is None


def test_get_room_window_present(service: BrowserService) -> None:
    tab = BrowserTab(target="tab-1", events=asyncio.Queue())
    window = BrowserWindow(tabs=[tab], active=0)
    service._windows[("room-123", "scout")] = window  # pyright: ignore[reportPrivateUsage]

    found = service.get_room_window("room-123")
    assert found is not None
    assert found is window
    assert found.tab.target == "tab-1"


@pytest.mark.asyncio
async def test_point_returns_center_and_bounding_box(
    service: BrowserService, fake_link: FakeLink
) -> None:
    tab = BrowserTab(target="tab-1", events=asyncio.Queue())
    (cx, cy), box = await service._point(fake_link, tab, "element-1", 42)  # pyright: ignore[reportPrivateUsage]

    assert cx == 150.0
    assert cy == 250.0
    assert box == {"x": 100.0, "y": 200.0, "width": 100.0, "height": 100.0}


@pytest.mark.asyncio
async def test_overlay_listener_and_broadcast(service: BrowserService) -> None:
    received: list[tuple[str, dict[str, Any], str]] = []

    def on_overlay(room_id: str, box: dict[str, Any], label: str) -> None:
        received.append((room_id, box, label))

    unregister = service.register_overlay_listener(on_overlay)
    sample_box = {"x": 10.0, "y": 20.0, "width": 50.0, "height": 30.0}
    await service.broadcast_overlay("room-1", sample_box, "Search Button")

    assert len(received) == 1
    assert received[0] == ("room-1", sample_box, "Search Button")

    unregister()
    await service.broadcast_overlay("room-1", sample_box, "Search Button")
    assert len(received) == 1


@pytest.mark.asyncio
async def test_overlay_listener_exception_is_logged_and_does_not_break_others(
    service: BrowserService, caplog: pytest.LogCaptureFixture
) -> None:
    received: list[tuple[str, dict[str, Any], str]] = []

    def broken_listener(room_id: str, box: dict[str, Any], label: str) -> None:
        raise RuntimeError("boom")

    def good_listener(room_id: str, box: dict[str, Any], label: str) -> None:
        received.append((room_id, box, label))

    service.register_overlay_listener(broken_listener)
    service.register_overlay_listener(good_listener)

    sample_box = {"x": 1.0, "y": 2.0, "width": 10.0, "height": 20.0}
    with caplog.at_level("WARNING"):
        await service.broadcast_overlay("room-1", sample_box, "Btn")

    assert len(received) == 1
    assert received[0] == ("room-1", sample_box, "Btn")
    assert "overlay listener failed for room-1" in caplog.text


def test_screencast_subscriber_refcounting(service: BrowserService) -> None:
    assert service.acquire_screencast("target-1") == 1
    assert service.acquire_screencast("target-1") == 2
    assert service.release_screencast("target-1") == 1
    assert service.release_screencast("target-1") == 0
    assert service.release_screencast("target-1") == 0


@pytest.mark.asyncio
async def test_action_broadcasts_overlay(service: BrowserService, fake_link: FakeLink) -> None:
    service._link = fake_link  # pyright: ignore[reportPrivateUsage]
    tab = BrowserTab(
        target="tab-1",
        events=asyncio.Queue(),
        entries=[Entry(role="button", name="Search", depth=0, ref="e1")],
    )
    tab.refs.ref_for(42)  # assign ref e1
    window = BrowserWindow(tabs=[tab], active=0)
    key: TabKey = ("room-1", "scout")
    service._windows[key] = window  # pyright: ignore[reportPrivateUsage]

    received: list[tuple[str, dict[str, Any], str]] = []
    service.register_overlay_listener(lambda room, box, label: received.append((room, box, label)))

    # Mock _element to return backend node 42
    service._element = AsyncMock(return_value=(42, "handle-1"))  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue] # type: ignore[method-assign]
    service._observe = AsyncMock(return_value={"url": "about:blank"})  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue] # type: ignore[method-assign]
    service._perform = AsyncMock(return_value=AsyncMock(navigated=False, dialog=None, chooser=None))  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue] # type: ignore[method-assign]

    await service.click(key, "e1")

    assert len(received) == 1
    assert received[0][0] == "room-1"
    assert received[0][2] == "Search"


@pytest.mark.asyncio
async def test_wait_for_control_blocks_until_give_back(service: BrowserService) -> None:
    """Reactive wait suspends clone execution while tab is user-controlled.

    Killed by: src/uclone_x/browser/service.py :: while tab.controller == "user":
    Becomes: while False:
    """
    tab = BrowserTab(target="tab-1", events=asyncio.Queue())
    window = BrowserWindow(tabs=[tab], active=0)
    key: TabKey = ("room-1", "scout")
    service._windows[key] = window  # pyright: ignore[reportPrivateUsage]
    service.take_over("room-1", "scout")
    assert tab.controller == "user"

    waiting = asyncio.create_task(service.wait_for_control(key))
    await asyncio.sleep(0.01)
    assert not waiting.done()

    service.give_back("room-1", "scout")
    await asyncio.wait_for(waiting, timeout=1.0)
    assert waiting.done()
    assert tab.controller == "scout"


@pytest.mark.asyncio
async def test_hand_over_note_reports_pages_visited_while_in_user_control_and_clears(
    service: BrowserService,
) -> None:
    """Hand-over note reports visited pages and clears pending flag on first observation.

    Killed by: src/uclone_x/browser/service.py :: tab.hand_over_pending = False
    Becomes: pass
    """
    tab = BrowserTab(
        target="tab-1",
        events=asyncio.Queue(),
        url="https://original.example.com",
        title="Original",
    )
    window = BrowserWindow(tabs=[tab], active=0)
    key: TabKey = ("room-1", "scout")
    service._windows[key] = window  # pyright: ignore[reportPrivateUsage]

    # User takes over
    service.take_over("room-1", "scout")
    assert tab.controller == "user"

    # User visits pages while in control
    tab.events.put_nowait(
        CdpEvent(
            method="Page.frameNavigated",
            params={"frame": {"url": "https://auth.example.com"}},
            session_id="tab-1",
        )
    )
    tab.events.put_nowait(
        CdpEvent(
            method="Page.frameNavigated",
            params={"frame": {"url": "https://app.example.com"}},
            session_id="tab-1",
        )
    )
    _absorb(tab)

    assert len(tab.visited_while_user) == 2
    assert tab.visited_while_user[0][0] == "https://auth.example.com"
    assert tab.visited_while_user[1][0] == "https://app.example.com"

    # User gives back
    service.give_back("room-1", "scout")
    assert tab.controller == "scout"
    assert tab.hand_over_pending is True

    # First observation consumes hand-over note
    res1 = _consume_hand_over(tab, {"url": tab.url})
    assert "hand_over_note" in res1
    assert "https://auth.example.com" in res1["hand_over_note"]
    assert "https://app.example.com" in res1["hand_over_note"]
    assert tab.hand_over_pending is False
    assert len(tab.visited_while_user) == 0

    # Second observation does NOT include hand-over note
    res2 = _consume_hand_over(tab, {"url": tab.url})
    assert "hand_over_note" not in res2


@pytest.mark.asyncio
async def test_ask_user_suspends_until_give_back_and_returns_snapshot(
    service: BrowserService, fake_link: FakeLink
) -> None:
    """The ask_user action sets asked_user state and suspends until give_back.

    Killed by: src/uclone_x/browser/service.py :: "kind": kind,
    Becomes: "kind": "",
    """
    service._link = fake_link  # pyright: ignore[reportPrivateUsage]
    tab = BrowserTab(
        target="tab-1",
        events=asyncio.Queue(),
        url="https://example.com/login",
        title="Login",
    )
    window = BrowserWindow(tabs=[tab], active=0)
    key: TabKey = ("room-1", "scout")
    service._windows[key] = window  # pyright: ignore[reportPrivateUsage]

    # Mock snapshot
    service.snapshot = AsyncMock(  # pyright: ignore[method-assign]
        return_value={"url": "https://example.com/dashboard", "snapshot": "Dashboard content"}
    )

    task = asyncio.create_task(service.ask_user(key, kind="sign_in", message="Please log in"))
    await asyncio.sleep(0.01)

    assert not task.done()
    assert tab.controller == "user"
    assert tab.asked_user is not None
    assert tab.asked_user["clone"] == "scout"
    assert tab.asked_user["kind"] == "sign_in"
    assert tab.asked_user["site"] == "example.com"
    assert tab.asked_user["message"] == "Please log in"

    # Give back wakes up ask_user
    service.give_back("room-1", "scout")
    result = await asyncio.wait_for(task, timeout=1.0)

    assert result["status"] == "ok"
    assert "Signed in" in result["message"]
    assert result["snapshot"] == "Dashboard content"


def test_cdp_detached_canceled_by_user_takes_over_tab() -> None:
    """A debugger detach canceled by user gives control to the user.

    Killed by: src/uclone_x/browser/service.py :: if event.params.get("reason") == "canceled_by_user":
    Becomes: if False:
    """
    tab = BrowserTab(target="tab-1", events=asyncio.Queue(), controller="scout")
    tab.events.put_nowait(
        CdpEvent(
            method="UClone.detached",
            params={"reason": "canceled_by_user"},
            session_id="tab-1",
        )
    )
    _absorb(tab)
    assert tab.controller == "user"
    assert not tab.controller_resumed.is_set()


@pytest.mark.asyncio
async def test_open_user_tab_creates_tab_with_user_controller(
    service: BrowserService, fake_link: FakeLink
) -> None:
    """Opening a user tab sets controller to user and records window under user key.

    Killed by: src/uclone_x/browser/service.py :: window = _Window(tabs=[tab], active=0)
    Becomes: window = None
    """
    service._link = fake_link  # pyright: ignore[reportPrivateUsage]
    tab = await service.open_user_tab("room-1", "about:blank")
    assert tab.controller == "user"
    assert not tab.controller_resumed.is_set()
    key: TabKey = ("room-1", "user")
    win = service._windows.get(key)  # pyright: ignore[reportPrivateUsage]
    assert win is not None
    assert win.tab is tab


@pytest.mark.asyncio
async def test_navigate_user_tab_navigates_and_tracks_history(
    service: BrowserService, fake_link: FakeLink
) -> None:
    service._link = fake_link  # pyright: ignore[reportPrivateUsage]
    tab = await service.open_user_tab("room-1", "about:blank")
    assert tab.controller == "user"

    service._navigate = AsyncMock()  # pyright: ignore[reportPrivateUsage,method-assign]
    await service.navigate_user_tab("room-1", "example.com/home")

    service._navigate.assert_awaited_once_with(fake_link, tab, "https://example.com/home")  # pyright: ignore[reportPrivateUsage]
    assert len(tab.visited_while_user) == 1


def test_hand_to_transfers_tab_to_target_clone(service: BrowserService) -> None:
    """Transferring control appends tab to target clone window.

    Killed by: src/uclone_x/browser/service.py :: target_win.tabs.append(tab)
    Becomes: pass
    """
    tab = BrowserTab(target="tab-1", events=asyncio.Queue(), controller="user")
    window = BrowserWindow(tabs=[tab], active=0)
    service._windows[("room-1", "scout")] = window  # pyright: ignore[reportPrivateUsage]

    # Pre-populate an analyst window to test append
    other_tab = BrowserTab(target="tab-2", events=asyncio.Queue(), controller="analyst")
    service._windows[("room-1", "analyst")] = BrowserWindow(tabs=[other_tab], active=0)  # pyright: ignore[reportPrivateUsage]

    service.hand_to("room-1", "analyst")
    assert tab.controller == "analyst"
    assert tab.controller_resumed.is_set()
    assert tab.hand_over_pending is True
    assert ("room-1", "scout") not in service._windows  # pyright: ignore[reportPrivateUsage]
    analyst_win = service._windows[("room-1", "analyst")]  # pyright: ignore[reportPrivateUsage]
    assert tab in analyst_win.tabs
