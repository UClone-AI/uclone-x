"""R1: the UClone-X extension in U0's own Chrome (design `browser-agent.md` §3.6, §3.7).

The extension (`chrome_extension/`, loaded unpacked for now) dials the Core at
`/api/browser/extension` over loopback and says hello with the pairing token Settings ▸
Browser shows. Once paired, the socket carries CDP's own framing, with the extension's tab
id as the session id, so `CdpConnection` runs over it unchanged and `R1Link` is a second
implementation of `BrowserLink`, not a second code path (§3.10).

`RoutedLink` is what the service holds: each new tab goes to the extension while one is
paired and connected, and to the Chrome the Core launches (R2) otherwise. The choice is per
tab, so pairing while a clone is browsing takes effect at that clone's next tab.

The pairing code is `<port>-<token>`: the extension needs both where to dial and what to
say, and one string is one thing to copy.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from websockets.exceptions import ConnectionClosed

from uclone_x.browser.actions import POPUP_NOT_FOLLOWED
from uclone_x.browser.cdp import CdpClosedError, CdpConnection, CdpEvent
from uclone_x.browser.link import BrowserLink
from uclone_x.errors import PlainRefusalError

EXTENSION_FOLDER = Path(__file__).parent / "chrome_extension"
"""The unpacked extension, for Chrome's "Load unpacked"."""

HELLO_TIMEOUT_S = 10.0
REFUSED_CLOSE_CODE = 4401

REFUSED_MESSAGE = (
    "UClone-X did not accept this pairing code. It may have been replaced: copy the code "
    "from Settings ▸ Browser again."
)
"""What the extension shows when its code is not the current one. Plain words only."""


class ExtensionSocket(Protocol):
    """The extension's socket: a websockets connection, or `StarletteSocket` around FastAPI's."""

    async def recv(self) -> str | bytes: ...

    async def send(self, message: str) -> None: ...

    def __aiter__(self) -> AsyncIterator[str | bytes]: ...

    async def close(self, code: int = 1000, reason: str = "") -> None: ...


class ExtensionHub:
    """The one extension connection the Core holds, and the pairing check in front of it."""

    def __init__(self, token: Callable[[], str | None]) -> None:
        self._token = token
        self._conn: CdpConnection | None = None
        self._link: R1Link | None = None
        self._version: str | None = None
        self._since: float | None = None
        self._waiters: list[asyncio.Future[None]] = []

    def use_token(self, token: Callable[[], str | None]) -> None:
        """Check pairings against `token()` from now on."""
        self._token = token

    async def serve(self, socket: ExtensionSocket) -> None:
        """Pair the socket and hold it until it closes. A new pairing replaces the old one."""
        hello = await self._hello(socket)
        expected = self._token()
        offered = hello.get("token") if hello is not None else None
        if (
            not expected
            or not isinstance(offered, str)
            or not hmac.compare_digest(offered.encode(), expected.encode())
        ):
            with contextlib.suppress(Exception):
                await socket.send(
                    json.dumps(
                        {"type": "refused", "reason": "pairing_code", "message": REFUSED_MESSAGE}
                    )
                )
                await socket.close(REFUSED_CLOSE_CODE, "pairing code")
            return
        await socket.send(json.dumps({"type": "paired"}))
        conn = CdpConnection(socket)
        previous = self._conn
        version = hello.get("version") if hello is not None else None
        self._conn = conn
        self._link = R1Link(conn)
        self._version = version if isinstance(version, str) else None
        self._since = time.time()
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._waiters.clear()
        if previous is not None:
            await previous.close()
        try:
            await conn.wait_closed()
        finally:
            if self._conn is conn:
                self._conn = None
                self._link = None
                self._version = None
                self._since = None

    def link(self) -> R1Link | None:
        """The link to the paired extension, or None when none is connected."""
        if self._conn is None or self._conn.closed:
            return None
        return self._link

    def status(self) -> dict[str, Any]:
        """For Settings ▸ Browser: whether an extension is connected, and since when."""
        connected = self.link() is not None
        return {
            "connected": connected,
            "version": self._version if connected else None,
            "since": self._since if connected else None,
        }

    async def wait_connected(self, timeout: float) -> None:
        """Wait until an extension is paired. For tests and the e2e."""
        if self.link() is not None:
            return
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        await asyncio.wait_for(waiter, timeout)

    async def disconnect(self) -> None:
        """Drop the connected extension, as a new pairing code does."""
        conn = self._conn
        if conn is not None:
            await conn.close()

    async def _hello(self, socket: ExtensionSocket) -> dict[str, Any] | None:
        try:
            raw = await asyncio.wait_for(socket.recv(), HELLO_TIMEOUT_S)
            decoded = json.loads(raw)
        except (TimeoutError, ValueError, ConnectionClosed):
            return None
        if not isinstance(decoded, dict):
            return None
        hello: dict[str, Any] = {str(k): v for k, v in decoded.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        return hello if hello.get("type") == "hello" else None


class R1Link:
    """A `BrowserLink` to tabs the extension opens in U0's Chrome, in the "UClone-X" group."""

    def __init__(self, conn: CdpConnection) -> None:
        self._conn = conn
        self._tabs: set[str] = set()

    @property
    def closed(self) -> bool:
        """Whether the extension's connection has ended."""
        return self._conn.closed

    async def open_tab(self, url: str) -> str:
        """Open a background tab in the group, with the debugger attached."""
        opened = await self._conn.send("UClone.openTab", {"url": url})
        tab = str(opened["tabId"])
        self._tabs.add(tab)
        return tab

    async def send(
        self, tab: str, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """A CDP command to the tab, through `chrome.debugger`."""
        return await self._conn.send(method, params, session_id=tab)

    def subscribe(self, tab: str, *, frames: bool = False) -> asyncio.Queue[CdpEvent]:
        """The tab's CDP events, and the extension's `UClone.detached` / `UClone.tabClosed`."""
        return self._conn.subscribe(tab, frames=frames)

    def unsubscribe(self, tab: str, queue: asyncio.Queue[CdpEvent]) -> None:
        """Stop delivering the tab's events to `queue`."""
        self._conn.unsubscribe(tab, queue)

    async def adopt_tab(self, opener: str, tab: str) -> str:
        """Refused for now: the extension attaches only to tabs it opened (`UClone.openTab`).

        Following a pop-up needs the extension to report the tabs its tabs open and attach
        to them; until then the clone is told, in plain words, that it stays on its tab.
        """
        raise PlainRefusalError(POPUP_NOT_FOLLOWED, reason_code="popup_not_followed")

    async def close_tab(self, tab: str) -> None:
        """Close the tab."""
        self._tabs.discard(tab)
        await self._conn.send("UClone.closeTab", {"tabId": tab})

    async def close(self) -> None:
        """Detach from this link's tabs and leave them, and the extension, as they are."""
        for tab in sorted(self._tabs):
            with contextlib.suppress(Exception):
                await self._conn.send("UClone.releaseTab", {"tabId": tab})
        self._tabs.clear()


class RoutedLink:
    """R1 for each new tab while the extension is connected, R2 otherwise (design §3.7)."""

    def __init__(self, hub: ExtensionHub, fallback: Callable[[], Awaitable[BrowserLink]]) -> None:
        self._hub = hub
        self._fallback_factory = fallback
        self._fallback: BrowserLink | None = None
        self._routes: dict[str, BrowserLink] = {}

    async def open_tab(self, url: str) -> str:
        """Open the tab in U0's Chrome when paired, else in the Chrome the Core launches."""
        link: BrowserLink | None = self._hub.link()
        if link is not None:
            try:
                tab = await link.open_tab(url)
            except CdpClosedError:
                link = None  # the extension left just now: fall back
            else:
                self._routes[tab] = link
                return tab
        if self._fallback is None:
            self._fallback = await self._fallback_factory()
        tab = await self._fallback.open_tab(url)
        self._routes[tab] = self._fallback
        return tab

    async def send(
        self, tab: str, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send to the tab over the route it was opened on."""
        return await self._route(tab).send(tab, method, params)

    def subscribe(self, tab: str, *, frames: bool = False) -> asyncio.Queue[CdpEvent]:
        """The tab's events."""
        return self._route(tab).subscribe(tab, frames=frames)

    async def adopt_tab(self, opener: str, tab: str) -> str:
        """A pop-up lives in its opener's Chrome, so it is adopted on the opener's route."""
        link = self._route(opener)
        adopted = await link.adopt_tab(opener, tab)
        self._routes[adopted] = link
        return adopted

    def unsubscribe(self, tab: str, queue: asyncio.Queue[CdpEvent]) -> None:
        """Stop delivering the tab's events to `queue`."""
        link = self._routes.get(tab)
        if link is not None:
            link.unsubscribe(tab, queue)

    async def close_tab(self, tab: str) -> None:
        """Close the tab."""
        link = self._routes.pop(tab, None)
        if link is not None:
            await link.close_tab(tab)

    async def close(self) -> None:
        """Drop both routes. Both Chromes stay open."""
        r1_links = {id(link): link for link in self._routes.values() if link is not self._fallback}
        self._routes.clear()
        for link in r1_links.values():
            await link.close()
        if self._fallback is not None:
            await self._fallback.close()
            self._fallback = None

    def _route(self, tab: str) -> BrowserLink:
        link = self._routes.get(tab)
        if link is None:
            raise KeyError(f"no open tab {tab!r}")
        if isinstance(link, R1Link) and link.closed:
            # The extension went away (Chrome quit, or a new pairing code): the tab is gone
            # for the Core, as a tab U0 closed is, and the next open picks a route afresh.
            del self._routes[tab]
            raise KeyError(f"tab {tab!r} was in a Chrome that is no longer connected")
        return link


def pairing_code(port: int, token: str) -> str:
    """What Settings shows and the extension takes: where to dial, and what to say."""
    return f"{port}-{token}"


_default_hub: ExtensionHub | None = None


def default_extension_hub() -> ExtensionHub:
    """The process's hub. The UI app points its token at the settings file (`use_token`)."""
    global _default_hub
    if _default_hub is None:
        _default_hub = ExtensionHub(lambda: None)
    return _default_hub


async def default_browser_link() -> BrowserLink:
    """The link `BrowserService` opens: the extension's tabs when paired, else R2."""
    from uclone_x.browser.chrome import R2Link

    return RoutedLink(default_extension_hub(), R2Link.start)
