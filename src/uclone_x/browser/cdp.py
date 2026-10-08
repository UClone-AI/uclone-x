"""A Chrome DevTools Protocol client over one WebSocket (design `browser-agent.md` §3.10).

Raw CDP, not Playwright: the Core speaks CDP method names to Chrome whichever link carries
them (R2, a Chrome the Core launched; R1, the extension relaying `chrome.debugger`), so a
second page model on top would have to be mapped back onto CDP for R1 anyway.

One connection serves every tab. Commands to a tab carry its flattened `sessionId`
(`Target.attachToTarget` with `flatten: true`), and events are routed to the queues
subscribed for that session.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

DEFAULT_COMMAND_TIMEOUT_S = 15.0

SCREENCAST_FRAME = "Page.screencastFrame"
"""Delivered only to queues subscribed with `frames=True`: a tab's own queue is drained
only between a clone's calls, and frames (tens of kilobytes each) would pile up there."""


class CdpError(Exception):
    """Chrome answered a command with an error."""

    def __init__(self, method: str, code: int, message: str) -> None:
        super().__init__(f"{method} failed ({code}): {message}")
        self.method = method
        self.code = code
        self.message = message


class CdpClosedError(Exception):
    """The connection to Chrome ended: Chrome quit, or the link was closed."""


@dataclass(frozen=True)
class CdpEvent:
    """One CDP event. `session_id` is the tab's session, `None` for browser-level events."""

    method: str
    params: dict[str, Any]
    session_id: str | None


class CdpTransport(Protocol):
    """What CdpConnection needs of a socket: a websockets connection, or the extension's link.

    A transport that ends raises websockets' `ConnectionClosed` from `send`, and its iterator
    stops (or raises the same).
    """

    async def send(self, message: str) -> None: ...

    def __aiter__(self) -> AsyncIterator[str | bytes]: ...

    async def close(self) -> None: ...


class CdpConnection:
    """JSON-RPC over a CDP WebSocket: request ids, flattened sessions, event subscriptions."""

    def __init__(self, ws: CdpTransport | ClientConnection) -> None:
        self._ws = ws
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[str, asyncio.Future[dict[str, Any]]]] = {}
        self._subscribers: dict[str | None, list[asyncio.Queue[CdpEvent]]] = {}
        self._frame_queues: set[int] = set()
        self._closed = False
        self._reader = asyncio.create_task(self._read_loop())

    @classmethod
    async def connect(cls, ws_url: str) -> CdpConnection:
        """Open a connection to a CDP endpoint (`webSocketDebuggerUrl`)."""
        # CDP messages carry whole accessibility trees and page HTML: no size cap.
        ws = await connect(ws_url, max_size=None, ping_interval=None)
        return cls(ws)

    @property
    def closed(self) -> bool:
        """Whether the connection has ended."""
        return self._closed

    async def send(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        session_id: str | None = None,
        timeout: float = DEFAULT_COMMAND_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Send one command and wait for its result."""
        if self._closed:
            raise CdpClosedError("The browser connection is closed.")
        msg_id = next(self._ids)
        message: dict[str, Any] = {"id": msg_id, "method": method, "params": dict(params or {})}
        if session_id is not None:
            message["sessionId"] = session_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = (method, future)
        try:
            await self._ws.send(json.dumps(message))
            return await asyncio.wait_for(future, timeout)
        except ConnectionClosed as exc:
            raise CdpClosedError("The browser connection is closed.") from exc
        finally:
            self._pending.pop(msg_id, None)

    def subscribe(self, session_id: str | None, *, frames: bool = False) -> asyncio.Queue[CdpEvent]:
        """A queue that receives every event of one session (or browser-level events).

        Screencast frames are left out unless `frames` is set (see `SCREENCAST_FRAME`).
        """
        queue: asyncio.Queue[CdpEvent] = asyncio.Queue()
        self._subscribers.setdefault(session_id, []).append(queue)
        if frames:
            self._frame_queues.add(id(queue))
        return queue

    def unsubscribe(self, session_id: str | None, queue: asyncio.Queue[CdpEvent]) -> None:
        """Stop delivering events to `queue`."""
        queues = self._subscribers.get(session_id, [])
        if queue in queues:
            queues.remove(queue)
        self._frame_queues.discard(id(queue))

    async def close(self) -> None:
        """Close the WebSocket. Chrome itself keeps running."""
        self._closed = True
        await self._ws.close()
        self._reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader
        self._fail_pending()

    async def wait_closed(self) -> None:
        """Return once the other end has gone (or `close` was called)."""
        await asyncio.wait({self._reader})

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                self._dispatch(raw)
        except ConnectionClosed:
            pass
        finally:
            self._closed = True
            self._fail_pending()

    def _dispatch(self, raw: str | bytes) -> None:
        try:
            decoded: object = json.loads(raw)
        except ValueError:
            logger.warning("CDP sent a message that is not JSON; ignored")
            return
        if not isinstance(decoded, dict):
            return
        message = cast(dict[str, Any], decoded)
        msg_id = message.get("id")
        if isinstance(msg_id, int):
            entry = self._pending.get(msg_id)
            if entry is None or entry[1].done():
                return
            method, future = entry
            error = message.get("error")
            if isinstance(error, dict):
                err = cast(dict[str, Any], error)
                future.set_exception(
                    CdpError(method, int(err.get("code", 0)), str(err.get("message", "")))
                )
            else:
                result = message.get("result")
                future.set_result(cast(dict[str, Any], result) if isinstance(result, dict) else {})
            return
        method_name = message.get("method")
        if not isinstance(method_name, str):
            return
        params = message.get("params")
        session = message.get("sessionId")
        event = CdpEvent(
            method=method_name,
            params=cast(dict[str, Any], params) if isinstance(params, dict) else {},
            session_id=session if isinstance(session, str) else None,
        )
        frame = method_name == SCREENCAST_FRAME
        for queue in self._subscribers.get(event.session_id, []):
            if frame and id(queue) not in self._frame_queues:
                continue
            queue.put_nowait(event)

    def _fail_pending(self) -> None:
        for _method, future in list(self._pending.values()):
            if not future.done():
                future.set_exception(CdpClosedError("The browser connection is closed."))
