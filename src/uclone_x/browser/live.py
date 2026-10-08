"""The dock's Browser tab, live (design `browser-agent.md` §3.4, §3.6; §6 step 3).

One `LiveView` serves one head's socket for one conversation. It tells the head, as codes
the head words itself (no copy and no error text cross this socket):

- `{"type": "state", ...}`: the conversation's tabs, which clone is acting, any problem
  (`chrome_missing`, `browser_closed`), whether U0's Chrome is paired, and which tab is shown;
- `{"type": "frame", "width", "height"}`, then binary JPEG frames of the shown tab, from
  `Page.startScreencast` (acknowledged frame by frame), only while the head says it is
  looking (`{"type": "view", "on": true}`);
- `{"type": "overlay", "box", "label", "action"}` when a clone is about to touch an element
  of the shown tab, in the page's CSS pixels;
- `{"type": "step", "clone", "action", "element", "ok"}` when a browser call ends.

The head may send `{"type": "view", "on": bool, "width": int}` and
`{"type": "tab", "clone": str, "index": int}` (which tab to watch), and #2122's controls on
the shown tab: `take_over`, `give_back`, `stop` (marks the tab as U0's) and, while the tab
is U0's, `input` (mouse, key or text, forwarded to the page). Step 4 (§3.4 controller)
makes the clone honour the controller; for now it is only shown.

Both links carry it: R2 is Chrome's own CDP, and the extension relays every event of the
tabs it attached (`chrome_extension/background.js`), screencast frames included. A tab
that is not painting (a background tab in a visible window) sends no frames, so a shown tab
silent for `STILL_AFTER_S` is shown by `Page.captureScreenshot` stills instead.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast

from uclone_x.browser.cdp import SCREENCAST_FRAME, CdpClosedError, CdpError
from uclone_x.browser.link import BrowserLink
from uclone_x.browser.service import BrowserService, Notice

logger = logging.getLogger(__name__)

JPEG_QUALITY = 60
DEFAULT_WIDTH = 1280
MIN_WIDTH, MAX_WIDTH = 320, 1920
MAX_HEIGHT = 1600
FRAME_INTERVAL_S = 0.1
"""At most ten frames a second: the next frame is acknowledged, then this pause."""
STILL_AFTER_S = 2.0
STILL_TIMEOUT_S = 3.0
STOP_TIMEOUT_S = 2.0

Shown = tuple[str, int]
"""(clone id, tab number from 1): the tab the head is shown."""


class LiveSocket(Protocol):
    """The head's socket, as `LiveView` uses it."""

    async def send_json(self, message: dict[str, Any]) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...

    async def receive_json(self) -> dict[str, Any] | None:
        """The head's next message; `None` once the socket has closed."""
        ...


class LiveView:
    """One head watching one conversation's browser."""

    def __init__(
        self,
        service: BrowserService,
        conversation: str,
        socket: LiveSocket,
        *,
        extension_connected: Callable[[], bool] = lambda: False,
        on_stop: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._service = service
        self._conversation = conversation
        self._socket = socket
        self._extension_connected = extension_connected
        self._on_stop = on_stop
        self._send_lock = asyncio.Lock()
        self._viewing = False
        self._width = DEFAULT_WIDTH
        self._pinned: Shown | None = None
        self._shown: Shown | None = None
        self._cast: asyncio.Task[None] | None = None
        self._cast_on: tuple[BrowserLink, str] | None = None
        self._cast_width: int | None = None
        self._frame_size: tuple[int, int] | None = None

    async def run(self) -> None:
        """Serve the socket until the head leaves."""
        inbox = self._service.watch(self._conversation)
        reader = asyncio.create_task(self._read(inbox))
        try:
            await self._send_state()
            while True:
                notice = await inbox.get()
                if notice.get("type") == "_closed":
                    break
                await self._handle(notice)
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
            await self._stop_cast()
            self._service.unwatch(self._conversation, inbox)

    async def _read(self, inbox: asyncio.Queue[Notice]) -> None:
        """The head's messages, put in the same queue as the service's notices."""
        while True:
            message = await self._socket.receive_json()
            if message is None:
                inbox.put_nowait({"type": "_closed"})
                return
            kind = message.get("type")
            if kind == "view":
                inbox.put_nowait(
                    {
                        "type": "_view",
                        "on": message.get("on") is True,
                        "width": message.get("width"),
                    }
                )
            elif kind == "tab":
                inbox.put_nowait(
                    {"type": "_tab", "clone": message.get("clone"), "index": message.get("index")}
                )
            elif kind in ("take_over", "give_back", "stop", "hand_to"):
                inbox.put_nowait({"type": "_control", "to": kind, "target": message.get("target")})
            elif kind == "new_tab":
                inbox.put_nowait({"type": "_new_tab", "url": message.get("url") or "about:blank"})
            elif kind == "navigate":
                inbox.put_nowait({"type": "_navigate", "url": message.get("url") or ""})
            elif kind == "input":
                inbox.put_nowait({"type": "_input", "message": message})

    async def _handle(self, notice: Notice) -> None:
        kind = notice.get("type")
        if kind == "state":
            await self._send_state()
        elif kind == "step":
            element = notice.get("element")
            await self._send(
                {
                    "type": "step",
                    "clone": notice.get("clone"),
                    "action": notice.get("action"),
                    "element": element if isinstance(element, str) else "",
                    "ok": notice.get("ok") is True,
                }
            )
        elif kind == "overlay":
            if self._cast_on is not None and notice.get("target") == self._cast_on[1]:
                await self._send(
                    {
                        "type": "overlay",
                        "clone": notice.get("clone"),
                        "box": notice.get("box"),
                        "label": notice.get("label", ""),
                        "action": notice.get("action", ""),
                    }
                )
        elif kind == "_view":
            self._viewing = bool(notice.get("on"))
            width = notice.get("width")
            if isinstance(width, int | float) and not isinstance(width, bool):
                self._width = max(MIN_WIDTH, min(MAX_WIDTH, int(width)))
            await self._send_state()
        elif kind == "_tab":
            clone, index = notice.get("clone"), notice.get("index")
            if isinstance(clone, str) and isinstance(index, int) and not isinstance(index, bool):
                self._pinned = (clone, index)
            await self._send_state()
        elif kind == "_control":
            action = notice.get("to")
            target = notice.get("target")
            shown = self._shown
            if action == "take_over":
                self._service.take_over(
                    self._conversation,
                    shown[0] if shown else None,
                    shown[1] if shown else None,
                )
            elif action == "give_back":
                clone = shown[0] if shown and shown[0] != "user" else None
                index = shown[1] if shown else None
                self._service.give_back(self._conversation, clone, index)
            elif action == "hand_to" and isinstance(target, str):
                index = shown[1] if shown else None
                self._service.hand_to(self._conversation, target, index)
            elif action == "stop":
                self._service.take_over(
                    self._conversation,
                    shown[0] if shown else None,
                    shown[1] if shown else None,
                )
                if self._on_stop is not None:
                    try:
                        await self._on_stop(self._conversation)
                    except Exception:
                        logger.warning("on_stop failed for %s", self._conversation, exc_info=True)
        elif kind == "_new_tab":
            url = str(notice.get("url") or "about:blank")
            await self._service.open_user_tab(self._conversation, url)
        elif kind == "_navigate":
            url = str(notice.get("url") or "")
            if url:
                await self._service.navigate_user_tab(self._conversation, url)
        elif kind == "_input":
            await self._forward(cast(dict[str, Any], notice.get("message") or {}))

    async def _forward(self, message: dict[str, Any]) -> None:
        """U0's mouse, key or text, to the shown tab while it is theirs (#2122's input)."""
        if self._shown is None:
            return
        tab = self._service.live_tab(self._conversation, *self._shown)
        target = self._service.live_target(self._conversation, *self._shown)
        if tab is None or target is None or tab.controller != "user":
            return
        link, page = target
        event = message.get("event") or message.get("action")
        if event == "mouse":
            method = "Input.dispatchMouseEvent"
            params: dict[str, Any] = {
                "type": str(message.get("mouse_type") or "mousePressed"),
                "x": _number(message.get("x")),
                "y": _number(message.get("y")),
                "button": str(message.get("button") or "left"),
                "clickCount": int(_number(message.get("click_count"), 1)),
            }
        elif event == "key":
            method = "Input.dispatchKeyEvent"
            params = {
                "type": str(message.get("key_type") or "keyDown"),
                "key": str(message.get("key") or ""),
            }
        elif event in ("text", "insertText"):
            method = "Input.insertText"
            params = {"text": str(message.get("text") or "")}
        else:
            return
        with contextlib.suppress(CdpError, CdpClosedError, KeyError, TimeoutError):
            await asyncio.wait_for(link.send(page, method, params), STOP_TIMEOUT_S)

    # -- state -----------------------------------------------------------------------

    async def _send_state(self) -> None:
        state = self._service.live_state(self._conversation)
        self._shown = self._choose(state)
        state["shown"] = {"clone": self._shown[0], "index": self._shown[1]} if self._shown else None
        state["extension"] = self._extension_connected()
        await self._send(state)
        await self._retarget()

    def _choose(self, state: Notice) -> Shown | None:
        """The acting clone's current tab; else the head's pick; else what was shown."""
        tabs = cast(list[dict[str, Any]], state.get("tabs", []))
        present = {(str(t["clone"]), int(t["index"])) for t in tabs}
        acting = cast(dict[str, Any] | None, state.get("acting"))
        if acting is not None:
            current = next(
                (
                    (str(t["clone"]), int(t["index"]))
                    for t in tabs
                    if t["clone"] == acting["clone"] and t["current"]
                ),
                None,
            )
            if current is not None:
                self._pinned = None
                return current
        for candidate in (self._pinned, self._shown):
            if candidate is not None and candidate in present:
                return candidate
        return next(((str(t["clone"]), int(t["index"])) for t in tabs if t["current"]), None)

    # -- the screencast --------------------------------------------------------------

    async def _retarget(self) -> None:
        target = (
            self._service.live_target(self._conversation, *self._shown)
            if self._viewing and self._shown is not None
            else None
        )
        running = self._cast is not None and not self._cast.done()
        if target == self._cast_on and running and self._cast_width == self._width:
            return
        await self._stop_cast()
        if target is not None:
            self._cast_on = target
            self._cast_width = self._width
            self._cast = asyncio.create_task(self._screencast(*target))

    async def _stop_cast(self) -> None:
        task = self._cast
        self._cast, self._cast_on, self._cast_width, self._frame_size = None, None, None, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _screencast(self, link: BrowserLink, target: str) -> None:
        queue = link.subscribe(target, frames=True)
        self._service.acquire_screencast(target)
        try:
            await link.send(
                target,
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": JPEG_QUALITY,
                    "maxWidth": self._width,
                    "maxHeight": MAX_HEIGHT,
                },
            )
            framed = False
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=None if framed else STILL_AFTER_S
                    )
                except TimeoutError:
                    await self._still(link, target)
                    continue
                if event.method != SCREENCAST_FRAME:
                    continue
                framed = True
                params = event.params
                with contextlib.suppress(CdpError):
                    await link.send(
                        target, "Page.screencastFrameAck", {"sessionId": params.get("sessionId")}
                    )
                meta = cast(dict[str, Any], params.get("metadata") or {})
                await self._send_frame(
                    str(params.get("data", "")),
                    meta.get("deviceWidth"),
                    meta.get("deviceHeight"),
                )
                await asyncio.sleep(FRAME_INTERVAL_S)
        except (CdpError, CdpClosedError, KeyError):
            # The tab or the browser went away; the next state notice says so.
            logger.debug("Screencast of a browser tab ended", exc_info=True)
        finally:
            link.unsubscribe(target, queue)
            if self._service.release_screencast(target) == 0:
                with contextlib.suppress(CdpError, CdpClosedError, KeyError, TimeoutError):
                    await asyncio.wait_for(link.send(target, "Page.stopScreencast"), STOP_TIMEOUT_S)

    async def _still(self, link: BrowserLink, target: str) -> None:
        """One screenshot, for a tab that sends no screencast frames."""
        try:
            shot = await asyncio.wait_for(
                link.send(
                    target, "Page.captureScreenshot", {"format": "jpeg", "quality": JPEG_QUALITY}
                ),
                STILL_TIMEOUT_S,
            )
            metrics = await asyncio.wait_for(
                link.send(target, "Page.getLayoutMetrics"), STILL_TIMEOUT_S
            )
        except (CdpError, TimeoutError):
            return
        view = cast(dict[str, Any], metrics.get("cssLayoutViewport") or {})
        await self._send_frame(
            str(shot.get("data", "")), view.get("clientWidth"), view.get("clientHeight")
        )

    async def _send_frame(self, data: str, width: object, height: object) -> None:
        try:
            jpeg = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            return
        if not jpeg:
            return
        size = (_whole(width), _whole(height))
        if size != self._frame_size:
            self._frame_size = size
            await self._send({"type": "frame", "width": size[0], "height": size[1]})
        async with self._send_lock:
            await self._socket.send_bytes(jpeg)

    async def _send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            await self._socket.send_json(message)


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default


def _whole(value: object) -> int:
    """A page dimension in CSS pixels, or 0 when the frame did not say."""
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return round(value)
    return 0
