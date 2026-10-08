"""The one seam the two browser routes share (design `browser-agent.md` §3.10).

R2 (`chrome.R2Link`, a Chrome the Core launches) and R1 (the extension in U0's own Chrome,
step 1b) both implement `BrowserLink`. Everything above it speaks CDP method names to a tab,
so R1 is a second implementation of this protocol, not a second code path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from uclone_x.browser.cdp import CdpEvent


@runtime_checkable
class BrowserLink(Protocol):
    """A connection to a Chrome that can open tabs and carry CDP commands to them."""

    async def open_tab(self, url: str) -> str:
        """Open a tab at `url` and attach to it. Returns the tab's id."""
        ...

    async def send(
        self, tab: str, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send one CDP command to a tab and return its result."""
        ...

    def subscribe(self, tab: str, *, frames: bool = False) -> asyncio.Queue[CdpEvent]:
        """A queue receiving every CDP event of the tab; screencast frames only if `frames`."""
        ...

    def unsubscribe(self, tab: str, queue: asyncio.Queue[CdpEvent]) -> None:
        """Stop delivering the tab's events to `queue`."""
        ...

    async def adopt_tab(self, opener: str, tab: str) -> str:
        """Attach to `tab`, which the page in `opener` opened itself (a pop-up); its id.

        A link that cannot attach to it raises `PlainRefusalError` (`popup_not_followed`).
        """
        ...

    async def close_tab(self, tab: str) -> None:
        """Close the tab."""
        ...

    async def close(self) -> None:
        """Drop the link. The browser keeps running: it is U0's window, not the Core's."""
        ...
